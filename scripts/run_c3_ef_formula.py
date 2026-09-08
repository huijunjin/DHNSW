#!/usr/bin/env python3
"""C3 follow-up: does a different ef_ref formula fix the high-dimensional case?

Equation (1) sets the reference ef from the ambient dimension,
`ef_ref = ef_base + (d / alpha)^2` with alpha = 100. The term is quadratic and
was only ever exercised between 128d and 960d, where it yields 101-181. At
3072d it yields 1000, and DHNSW inherits the cost of an operating point no one
chose. C3 recorded that as a limitation; this script asks whether it is fixable.

Five candidate formulas are built and measured against the same vanilla
baseline on the same data. A candidate has to clear two bars, not one:

  (i)  it must not blow up at 3072d, and
  (ii) it must leave 128-960d essentially where the paper already has it.

A formula that only satisfies (i) has traded the paper's validated range for
its unvalidated one, which is not an improvement.

`intrinsic` is the interesting candidate. SimpleWiki is 3072-dimensional but
has the *lowest* density dispersion of any dataset here (CV 0.071) -- its
neighbourhood structure is simple despite the ambient width. Estimating the
intrinsic dimension with the two-NN maximum-likelihood estimator (Facco et al.,
2017) and feeding that to Eq. (1) tests whether the ambient `d` was the wrong
variable all along, rather than the exponent being the wrong shape.

Everything runs against the C++ port, so the whole matrix is minutes rather
than the hours the pure-Python engine needs at 3072d.

Example:
    python scripts/run_c3_ef_formula.py --phase 3
"""
import argparse
import gc
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cpp"))
import hnswlib  # noqa: E402

from run_baseline import (  # noqa: E402
    EF_SCALE_FACTOR, EF_VANILLA_BASE, LOADERS, M_VANILLA, TOP_K,
    calculate_density, set_dynamic_hnsw_params_by_std,
)
from run_c11_threads import build_index, dynamic_params  # noqa: E402
from hdf5_loaders import HDF5_DATASETS, load_hdf5  # noqa: E402
from exp_common import (  # noqa: E402
    add_common_args, order_columns, resolve_outputs, run_metadata, save_csv,
)


# ---------------------------------------------------------------- candidates
def f_paper(d, _int_d):
    """Eq. (1) unchanged."""
    return EF_VANILLA_BASE + (d // EF_SCALE_FACTOR) ** 2


def f_hardcap(d, _int_d, cap=200):
    """Eq. (1) with a ceiling. Identical below the ceiling, flat above it."""
    return EF_VANILLA_BASE + min((d // EF_SCALE_FACTOR) ** 2, cap)


def f_softcap(d, _int_d, cap=200):
    """`cap * (1 - exp(-x/cap))` with `x = (d/alpha)^2`.

    Approaches x for small x -- so it tracks Eq. (1) closely in the validated
    range -- and saturates at `cap` instead of stopping abruptly. Chosen over
    the hard cap because the derivative stays continuous, which matters if the
    formula is ever fitted rather than set by hand.
    """
    x = (d / EF_SCALE_FACTOR) ** 2
    return EF_VANILLA_BASE + int(round(cap * (1.0 - np.exp(-x / cap))))


def f_intrinsic(_d, int_d):
    """Eq. (1) evaluated at the estimated intrinsic dimension, alpha unchanged.

    Worth running even though the outcome is easy to predict: every dataset
    here estimates out at an intrinsic dimension of 12-20, so `(d_int / 100)^2`
    truncates to zero and this collapses onto `none`. That collapse is the
    result -- with the paper's alpha, "use the intrinsic dimension" and "drop
    the dimension term" are the same proposal.
    """
    return EF_VANILLA_BASE + (int(int_d) // EF_SCALE_FACTOR) ** 2


# alpha for the intrinsic dimension, anchored so that GIST -- the widest
# dataset inside the paper's validated 128-960d range -- lands on the ef_ref
# Eq. (1) already gives it (181). GIST's two-NN estimate is 12.7, so
# 100 + (12.7 / a)^2 = 181 gives a = 12.7 / 9.
ALPHA_INTRINSIC = 12.7 / 9.0


def f_intrinsic_scaled(_d, int_d):
    """The intrinsic dimension with alpha recalibrated to the validated range.

    This is the version of the idea that can actually be compared against the
    paper: keep the quadratic form, swap the variable, and refit the one
    constant so the widest validated dataset is unchanged. Whether the rest of
    that range stays put is the test -- intrinsic dimension need not order the
    datasets the way ambient dimension does, and if it does not, this is a
    different formula rather than a recalibration of the same one.
    """
    return EF_VANILLA_BASE + int(round((int_d / ALPHA_INTRINSIC) ** 2))


# gamma set so d = 960 reproduces the paper's ef_ref of 181:
# 100 * (1 + g * ln(1 + 9.6)) = 181  ->  g = 0.81 / ln(10.6).
GAMMA_LOG = 0.81 / np.log(10.6)


def f_log(d, _int_d):
    """`ef_base * (1 + gamma * ln(1 + d/alpha))` -- grows without bound but
    slowly enough that 3072d stays near 200 instead of reaching 1000."""
    return int(round(EF_VANILLA_BASE * (1.0 + GAMMA_LOG * np.log(1.0 + d / EF_SCALE_FACTOR))))


def f_none(_d, _int_d):
    """Drop the dimension term. Equivalent to the `dhnsw_ef100` arm of C11,
    which is already known to beat vanilla in C++ on all four raw datasets."""
    return EF_VANILLA_BASE


FORMULAS = {
    "paper": f_paper,
    "hardcap200": f_hardcap,
    "softcap200": f_softcap,
    "log": f_log,
    "intrinsic": f_intrinsic,
    "intrinsic_scaled": f_intrinsic_scaled,
    "none": f_none,
}


# ------------------------------------------------------- intrinsic dimension
def twonn_dimension(data, sample=20000, seed=42):
    """Two-NN maximum-likelihood intrinsic dimension (Facco et al., 2017).

    For each point take the two nearest neighbour distances r1 <= r2 and form
    mu = r2 / r1. Under a locally uniform density on a d-dimensional manifold,
    mu is Pareto(d), whose MLE is n / sum(log mu). Points with r1 = 0
    (duplicates) carry no information and are dropped.
    """
    rng = np.random.default_rng(seed)
    if len(data) > sample:
        data = data[rng.choice(len(data), sample, replace=False)]
    x = np.ascontiguousarray(data, dtype=np.float32)

    try:
        import faiss
        index = faiss.IndexFlatL2(x.shape[1])
        index.add(x)
        d2, _ = index.search(x, 3)
        dist = np.sqrt(np.maximum(d2[:, 1:3], 0.0))
    except ImportError:
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=3).fit(x)
        dist = nn.kneighbors(x, return_distance=True)[0][:, 1:3]

    r1, r2 = dist[:, 0], dist[:, 1]
    keep = (r1 > 0) & (r2 > r1)
    mu = r2[keep] / r1[keep]
    return float(len(mu) / np.sum(np.log(mu)))


def pca95_dimension(data, sample=20000, seed=42):
    """Components needed for 95 % of the variance -- a cruder, linear reading
    of the same question, reported alongside two-NN as a sanity check."""
    from sklearn.decomposition import PCA
    rng = np.random.default_rng(seed)
    if len(data) > sample:
        data = data[rng.choice(len(data), sample, replace=False)]
    n_comp = min(len(data), data.shape[1], 512)
    ratio = PCA(n_components=n_comp).fit(np.asarray(data, np.float32)).explained_variance_ratio_
    return int(np.searchsorted(np.cumsum(ratio), 0.95) + 1)


# ------------------------------------------------------------------ measure
def recall_at(index, queries, gt, k, ef, threads):
    index.set_ef(ef)
    labels, _ = index.knn_query(queries, k=k, num_threads=threads)
    hits = sum(len(set(labels[i]) & set(gt[i][:k])) for i in range(len(queries)))
    return 100.0 * hits / (len(queries) * k)


def exact_gt(train, queries, k):
    try:
        import faiss
        index = faiss.IndexFlatL2(train.shape[1])
        index.add(np.ascontiguousarray(train, np.float32))
        return index.search(np.ascontiguousarray(queries, np.float32), k)[1]
    except ImportError:
        from sklearn.neighbors import NearestNeighbors
        return NearestNeighbors(n_neighbors=k).fit(train).kneighbors(
            queries, return_distance=False)


def load_any(name, args):
    if name in LOADERS:
        return LOADERS[name](args.datasets_dir, args.num_vectors, args.num_query_vectors)
    return load_hdf5(name, args.num_vectors, args.num_query_vectors,
                     seed=args.seed, data_dir=args.hdf5_dir)


def run_dataset(name, args, rows):
    print(f"\n{'=' * 74}\n{name}\n{'=' * 74}", flush=True)
    train, queries = load_any(name, args)
    train = np.ascontiguousarray(train, dtype=np.float32)
    queries = np.ascontiguousarray(queries, dtype=np.float32)
    dim = train.shape[1]
    gt = exact_gt(train, queries, args.top_k)

    t0 = time.time()
    int_twonn = twonn_dimension(train, args.id_sample, args.seed)
    int_pca = pca95_dimension(train, args.id_sample, args.seed)
    id_time = time.time() - t0
    print(f"  dim={dim}  intrinsic: two-NN {int_twonn:.1f} | PCA-95 {int_pca}  "
          f"({id_time:.1f}s)", flush=True)

    densities = calculate_density(train, k=M_VANILLA, seed=args.seed)

    idx, build_s = build_index(train, dim, "vanilla", args.threads,
                               M_VANILLA, EF_VANILLA_BASE)
    base = dict(build_s=build_s, edges=idx.get_total_edge_count(),
                recall=recall_at(idx, queries, gt, args.top_k, args.ef_search, args.threads))
    rows.append(dict(dataset=name, dim=dim, formula="vanilla", ef_ref=EF_VANILLA_BASE,
                     intrinsic_twonn=round(int_twonn, 2), intrinsic_pca95=int_pca,
                     m_range="-", ef_range="-", build_s=round(build_s, 3),
                     edges=base["edges"], avg_degree=round(base["edges"] / len(train), 3),
                     recall=round(base["recall"], 3), build_vs_vanilla_pct=0.0,
                     edges_vs_vanilla_pct=0.0, recall_vs_vanilla_pp=0.0))
    print(f"  vanilla            build {build_s:7.2f}s  edges {base['edges']:>9,}  "
          f"recall {base['recall']:.2f}%", flush=True)
    del idx
    gc.collect()

    for key in args.formulas:
        ef_ref = int(FORMULAS[key](dim, int_twonn))
        cv, m_lo, m_hi, ef_lo, ef_hi = set_dynamic_hnsw_params_by_std(
            densities, M_VANILLA, ef_ref)
        m_arr, ef_arr = dynamic_params(densities, m_lo, m_hi, ef_lo, ef_hi)
        idx, build_s = build_index(train, dim, "dhnsw", args.threads,
                                   m_hi, ef_hi, m_arr=m_arr, ef_arr=ef_arr)
        edges = idx.get_total_edge_count()
        rec = recall_at(idx, queries, gt, args.top_k, args.ef_search, args.threads)
        rows.append(dict(
            dataset=name, dim=dim, formula=key, ef_ref=ef_ref, cv=round(cv, 4),
            intrinsic_twonn=round(int_twonn, 2), intrinsic_pca95=int_pca,
            m_range=f"[{m_lo},{m_hi}]", ef_range=f"[{ef_lo},{ef_hi}]",
            build_s=round(build_s, 3), edges=edges,
            avg_degree=round(edges / len(train), 3), recall=round(rec, 3),
            build_vs_vanilla_pct=round((build_s - base["build_s"]) / base["build_s"] * 100, 2),
            edges_vs_vanilla_pct=round((edges - base["edges"]) / base["edges"] * 100, 2),
            recall_vs_vanilla_pp=round(rec - base["recall"], 3)))
        print(f"  {key:<18} ef_ref {ef_ref:>5}  build {build_s:7.2f}s "
              f"({rows[-1]['build_vs_vanilla_pct']:+6.1f}%)  edges {edges:>9,}  "
              f"recall {rec:.2f}% ({rows[-1]['recall_vs_vanilla_pp']:+.2f}pp)", flush=True)
        del idx
        gc.collect()


def build_arg_parser():
    p = argparse.ArgumentParser(
        description="C3 follow-up: ef_ref formula alternatives at high dimension.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_common_args(p)
    p.add_argument("--datasets", nargs="+",
                   default=["sift1m", "glove100k", "landmark-dino", "landmark-nomic",
                            "mnist", "gist", "simplewiki-openai"],
                   choices=list(LOADERS) + list(HDF5_DATASETS))
    p.add_argument("--formulas", nargs="+", default=list(FORMULAS), choices=list(FORMULAS))
    p.add_argument("--datasets-dir", default=os.path.expanduser("~/rag/hnsw-python"))
    p.add_argument("--hdf5-dir", default=None)
    p.add_argument("--num-vectors", type=int, default=60000)
    p.add_argument("--num-query-vectors", type=int, default=100)
    p.add_argument("--top-k", type=int, default=TOP_K)
    p.add_argument("--ef-search", type=int, default=200)
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--id-sample", type=int, default=20000,
                   help="Points sampled for the intrinsic-dimension estimates.")
    p.add_argument("--stem", default="c3_ef_formula")
    return p


def main():
    args = build_arg_parser().parse_args()
    rows = []
    for name in args.datasets:
        try:
            run_dataset(name, args, rows)
        except Exception as exc:            # one dataset must not sink the sweep
            print(f"  !! {name} failed: {type(exc).__name__}: {exc}", flush=True)

    import pandas as pd
    for r in rows:
        r.update(run_metadata(args.phase, r["dataset"], r["formula"], args.seed))
    df = order_columns(pd.DataFrame(rows))
    csv_path, _ = resolve_outputs(args, args.stem)
    save_csv(df, csv_path)
    print("\n" + "=" * 74)
    print(df.drop(columns=[c for c in ("git_sha", "timestamp", "host", "seed")
                           if c in df.columns]).to_string(index=False))
    print(f"\nSaved: {csv_path}")


if __name__ == "__main__":
    main()
