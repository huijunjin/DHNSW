#!/usr/bin/env python3
"""DHNSW baseline runner (Chapter 3).

Reproduces the experiment driver of the original `main.py` -- same constants,
same call order, same metrics -- but with the dataset directory, the dataset
selection and the sample sizes exposed as command-line arguments instead of
module-level constants. The original `main.py` hardcodes `DATASET_DIR =
"../dataset"`, a path that does not exist on this machine, so it cannot be run
as-is; `dhnsw.py` (the algorithm) is used unmodified.

Metrics follow the paper: build time, peak memory (tracemalloc), and recall.
QPS is deliberately not measured -- the DHNSW paper does not report it.

Example (golden master):
    python scripts/run_baseline.py --datasets mnist --num-vectors 60000 \
        --datasets-dir ~/rag/hnsw-python --seed 42 --out baseline/dhnsw_mnist.csv
"""
import argparse
import os
import sys
import time
import tracemalloc

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.random_projection import GaussianRandomProjection

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dhnsw import HNSW, DynamicHNSW

# Defaults below are the constants from the original main.py, unchanged.
M_VANILLA = 16
EF_VANILLA_BASE = 100
SCALE_FACTOR = 1.5          # lambda in the paper
EF_SCALE_FACTOR = 100       # alpha in the paper
TOP_K = 100


# ---------------------------------------------------------------- loaders
def _read_fvecs(path, num_vectors):
    with open(path, "rb") as f:
        vectors = []
        while len(vectors) < num_vectors:
            dim_bytes = f.read(4)
            if not dim_bytes:
                break
            dim = int(np.frombuffer(dim_bytes, dtype=np.int32)[0])
            vectors.append(np.frombuffer(f.read(dim * 4), dtype=np.float32))
    return np.array(vectors)


def _read_idx_images(path, num_vectors):
    with open(path, "rb") as f:
        f.read(16)  # magic, count, rows, cols
        buf = f.read(num_vectors * 28 * 28)
    return np.frombuffer(buf, dtype=np.uint8).reshape(-1, 28 * 28).astype(np.float64)


def load_mnist(data_dir, n_train, n_query):
    d = os.path.join(data_dir, "mnist")
    return (_read_idx_images(os.path.join(d, "train-images.idx3-ubyte"), n_train),
            _read_idx_images(os.path.join(d, "t10k-images.idx3-ubyte"), n_query))


def load_glove100k(data_dir, n_train, n_query, dim=300):
    path = os.path.join(data_dir, "glove100k", f"glove.6B.{dim}d.txt")

    def read(n):
        out = []
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= n:
                    break
                out.append([float(x) for x in line.split()[1:]])
        return np.array(out)

    return read(n_train), read(n_query)


def load_sift1m(data_dir, n_train, n_query):
    d = os.path.join(data_dir, "sift1m")
    return (_read_fvecs(os.path.join(d, "sift_base.fvecs"), n_train),
            _read_fvecs(os.path.join(d, "sift_query.fvecs"), n_query))


def load_gist(data_dir, n_train, n_query):
    d = os.path.join(data_dir, "gist")
    return (_read_fvecs(os.path.join(d, "gist_base.fvecs"), n_train),
            _read_fvecs(os.path.join(d, "gist_query.fvecs"), n_query))


LOADERS = {"mnist": load_mnist, "glove100k": load_glove100k,
           "sift1m": load_sift1m, "gist": load_gist}


# ------------------------------------------------------ paper's Algorithm 1
def calculate_density_random_projection(data, k, seed=None):
    """RP-KNN local density estimate (paper Sec. III, Line 1).

    `seed` is an addition: the original constructs GaussianRandomProjection with
    no random_state, so the density estimate -- and therefore every per-node M
    and ef -- differs between runs. Leaving it None reproduces that behaviour
    exactly; setting it makes a run repeatable, which is what the baseline
    measurement needs.
    """
    n_components = max(1, data.shape[1] // 3)
    rp = GaussianRandomProjection(n_components=n_components, random_state=seed)
    reduced = rp.fit_transform(data)
    knn = NearestNeighbors(n_neighbors=k, algorithm="brute", metric="euclidean")
    knn.fit(reduced)
    distances, _ = knn.kneighbors(reduced)
    return np.mean(distances, axis=1)


DENSITY_METHODS = ("rp", "pca", "lsh")


def calculate_density(data, k, seed=None, method="rp"):
    """Local density estimate with a swappable projection (C8 ablation).

    The paper projects with a dense Gaussian random projection ("rp"); C8 asks
    what a data-dependent (PCA) or cheaper sparse (LSH-style) projection buys
    or costs. Every method keeps the same target dimension and the same kNN
    step, so only the projection differs. "rp" delegates to the original
    function, unchanged.
    """
    if method == "rp":
        return calculate_density_random_projection(data, k, seed)

    n_components = max(1, data.shape[1] // 3)
    if method == "pca":
        from sklearn.decomposition import PCA
        proj = PCA(n_components=n_components, random_state=seed)
    elif method == "lsh":
        # Sparse random projection: the same JL guarantee as the Gaussian one
        # at a fraction of the projection cost, and the primitive underneath
        # most LSH families.
        from sklearn.random_projection import SparseRandomProjection
        proj = SparseRandomProjection(n_components=n_components, random_state=seed)
    else:
        raise ValueError(f"unknown density method: {method}")

    reduced = proj.fit_transform(data)
    knn = NearestNeighbors(n_neighbors=k, algorithm="brute", metric="euclidean")
    knn.fit(reduced)
    distances, _ = knn.kneighbors(reduced)
    return np.mean(distances, axis=1)


def adjust_ef_by_dim(ef_base, dim):
    """ef_ref, paper Eq. (1)."""
    return ef_base + (dim // EF_SCALE_FACTOR) ** 2


CV_TRANSFORMS = ("linear", "exp", "sqrt", "ratio")


def cv_dispersion(densities, cv_transform):
    """The CV term itself (paper Eq. 2-5 uses std/mean). `ratio` (C7 ablation)
    swaps in a bounded alternative, sigma / (mu + sigma), instead."""
    mu = np.mean(densities)
    sigma = np.std(densities)
    if cv_transform == "ratio":
        return sigma / (mu + sigma)
    return sigma / mu


def cv_delta_factor(cv, scale_factor, cv_transform):
    """How much of m_vanilla / ef_vanilla the CV term expands the range by.
    `linear` is the paper's formula, unchanged; the others are C7 ablation
    variants (see PLAN.md) that reshape the same CV -> delta mapping."""
    if cv_transform == "exp":
        return 1 - np.exp(-cv * scale_factor)
    if cv_transform == "sqrt":
        return np.sqrt(cv) * scale_factor
    return cv * scale_factor  # linear, ratio


def set_dynamic_hnsw_params_by_std(densities, m_vanilla, ef_vanilla,
                                   scale_factor=SCALE_FACTOR, cv_transform="linear"):
    """M_low/M_high and ef_low/ef_high, paper Eq. (2)-(5). The CV term is
    std/mean of the local densities.

    `cv_transform` is a C7 ablation addition (see PLAN.md): it swaps how the
    CV term is derived and how it scales into a range-expansion factor. The
    default, "linear", is byte-for-byte the paper's original formula.
    """
    cv = cv_dispersion(densities, cv_transform)
    delta = cv_delta_factor(cv, scale_factor, cv_transform)
    m_start = max(2, int(m_vanilla - m_vanilla * delta))
    m_end = int(m_vanilla + m_vanilla * delta)
    ef_start = max(10, int(ef_vanilla - ef_vanilla * delta))
    ef_end = int(ef_vanilla + ef_vanilla * delta)
    print(f"  CV[{cv_transform}]: {cv:.4f} (delta={delta:.4f}), "
          f"M range: [{m_start}, {m_end}], EF range: [{ef_start}, {ef_end}]")
    return cv, m_start, m_end, ef_start, ef_end


def calculate_recall(true_neighbors, retrieved):
    true_set, got = set(true_neighbors), set(retrieved)
    return len(true_set & got) / len(true_set) * 100


def measure_performance(hnsw_class, data, queries, true_neighbors, label, k,
                        densities=None, ef_vanilla=None, cv_transform="linear"):
    # Matches the original driver: vanilla HNSW runs at EF_VANILLA_BASE, while
    # the dimension-adjusted ef_ref (Eq. 1) is DHNSW's own contribution and is
    # passed in only for the dynamic variant.
    if ef_vanilla is None:
        ef_vanilla = EF_VANILLA_BASE

    tracemalloc.start()
    start = time.time()
    if hnsw_class is DynamicHNSW:
        cv, m_start, m_end, ef_start, ef_end = set_dynamic_hnsw_params_by_std(
            densities, M_VANILLA, ef_vanilla, cv_transform=cv_transform)
        hnsw = hnsw_class("l2", densities, m_start=m_start, m_end=m_end,
                          ef_start=ef_start, ef_end=ef_end)
    else:
        cv = float("nan")
        hnsw = hnsw_class("l2", m=M_VANILLA, ef=ef_vanilla)

    for point in data:
        hnsw.add(point)
    build_time = time.time() - start

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    memory_mb = peak / 10 ** 6

    recalls = [calculate_recall(true_neighbors[i], [idx for idx, _ in hnsw.search(q, k)])
               for i, q in enumerate(queries)]
    avg_recall = float(np.mean(recalls))

    print(f"{label} - Build Time: {build_time:.2f}s, Memory Usage: {memory_mb:.2f} MB, "
          f"Recall: {avg_recall:.2f}%")
    return {"Build_Time_s": build_time, "Memory_MB": memory_mb,
            "Recall_pct": avg_recall, "CV": cv, "Avg_Degree": hnsw.get_average_neighbors()}


def build_arg_parser():
    p = argparse.ArgumentParser(
        description="DHNSW baseline: build time / memory / recall vs vanilla HNSW.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--datasets-dir", default=os.path.expanduser("~/rag/hnsw-python"),
                   help="Directory holding mnist/ glove100k/ sift1m/ gist/ (read-only).")
    p.add_argument("--datasets", nargs="+", default=["mnist"], choices=list(LOADERS))
    p.add_argument("--num-vectors", type=int, default=60000,
                   help="Base vectors. Paper: MNIST 60000, GloVe100K 100000, SIFT1M/GIST1M 1000000.")
    p.add_argument("--num-query-vectors", type=int, default=100)
    p.add_argument("--top-k", type=int, default=TOP_K)
    p.add_argument("--seed", type=int, default=None,
                   help="random_state for the RP density estimate. The original passes none; "
                        "set it for a repeatable baseline.")
    p.add_argument("--cv-transform", default="linear", choices=list(CV_TRANSFORMS),
                   help="C7 ablation (see PLAN.md): how the CV term scales into the M/ef "
                        "range. 'linear' is the paper's original formula, unchanged.")
    p.add_argument("--density-method", default="rp", choices=list(DENSITY_METHODS),
                   help="C8 ablation: projection used for the density estimate. "
                        "'rp' is the paper's Gaussian random projection, unchanged.")
    p.add_argument("--out", default=None, help="CSV path to write.")
    return p


def main():
    args = build_arg_parser().parse_args()
    rows = []

    for name in args.datasets:
        print(f"\nLoading {name} dataset...")
        train, queries = LOADERS[name](os.path.expanduser(args.datasets_dir),
                                       args.num_vectors, args.num_query_vectors)
        print(f"Running experiments for {name}...  train={train.shape} query={queries.shape}")

        ef_vanilla = adjust_ef_by_dim(EF_VANILLA_BASE, train.shape[1])
        densities = calculate_density(train, k=M_VANILLA, seed=args.seed,
                                      method=args.density_method)

        knn = NearestNeighbors(n_neighbors=args.top_k, algorithm="brute", metric="euclidean")
        knn.fit(train)
        true_neighbors = knn.kneighbors(queries, n_neighbors=args.top_k, return_distance=False)

        # Same order as the original driver: Dynamic first, then vanilla.
        dyn = measure_performance(DynamicHNSW, train, queries, true_neighbors,
                                  f"Dynamic HNSW ({name})", k=args.top_k,
                                  densities=densities, ef_vanilla=ef_vanilla,
                                  cv_transform=args.cv_transform)
        van = measure_performance(HNSW, train, queries, true_neighbors,
                                  f"Vanilla HNSW ({name})", k=args.top_k)

        for variant, res in (("DHNSW", dyn), ("Vanilla HNSW", van)):
            rows.append({"Dataset": name, "Variant": variant, "N": len(train),
                         "Dim": train.shape[1], "ef_ref": ef_vanilla, "seed": args.seed,
                         "CV_Transform": args.cv_transform,
                         "Density_Method": args.density_method, **res})

        t_imp = (van["Build_Time_s"] - dyn["Build_Time_s"]) / van["Build_Time_s"] * 100
        m_imp = (van["Memory_MB"] - dyn["Memory_MB"]) / van["Memory_MB"] * 100
        print(f"\n  [{name}] Build time improvement: {t_imp:+.2f}%   "
              f"Memory improvement: {m_imp:+.2f}%   "
              f"Recall: {van['Recall_pct']:.2f}% -> {dyn['Recall_pct']:.2f}%")

    df = pd.DataFrame(rows)
    print("\n" + df.to_string(index=False))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        df.to_csv(args.out, index=False)
        print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
