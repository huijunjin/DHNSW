#!/usr/bin/env python3
"""C2: where DHNSW's build time actually goes (Chapter 3, review comment C2).

The paper reports DHNSW's build time as the graph-construction loop. DHNSW
also has a step vanilla HNSW does not: the RP-KNN local density estimate that
produces the per-node M and ef. A reviewer is entitled to ask whether the
reported saving survives once that preprocessing is charged to DHNSW, so this
runner times it separately and reports both.

Phases timed:
  rp_projection   GaussianRandomProjection.fit_transform  (Algorithm 1, line 1)
  knn_density     brute-force kNN over the projected data + the row means
  cv_params       Eq. (2)-(5), the M/ef ranges -- expected to be negligible
  graph_build     the add() loop, which is what the paper's figures report

Vanilla runs the graph build alone; that is the honest comparison denominator.

Example:
    python scripts/run_c2_phases.py --seed 42
    # -> results/phase1/c2_phase_overhead_mnist.{csv,png}
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_baseline import (  # noqa: E402
    EF_VANILLA_BASE, LOADERS, M_VANILLA, TOP_K, adjust_ef_by_dim,
    calculate_recall, set_dynamic_hnsw_params_by_std,
)
from dhnsw import HNSW, DynamicHNSW  # noqa: E402
from exp_common import (  # noqa: E402
    add_common_args, order_columns, resolve_outputs, run_metadata, save_csv,
)

PHASES = ["rp_projection", "knn_density", "cv_params", "graph_build"]


def timed_density(data, k, seed):
    """calculate_density_random_projection, split into its two phases.

    Kept arithmetically identical to run_baseline's version -- same component
    count, same estimator arguments, same mean -- so the timings describe the
    code the paper's numbers came from.
    """
    t0 = time.time()
    n_components = max(1, data.shape[1] // 3)
    rp = GaussianRandomProjection(n_components=n_components, random_state=seed)
    reduced = rp.fit_transform(data)
    t_rp = time.time() - t0

    t0 = time.time()
    knn = NearestNeighbors(n_neighbors=k, algorithm="brute", metric="euclidean")
    knn.fit(reduced)
    distances, _ = knn.kneighbors(reduced)
    densities = np.mean(distances, axis=1)
    t_knn = time.time() - t0

    return densities, t_rp, t_knn


def build_arg_parser():
    p = argparse.ArgumentParser(
        description="C2: RP-KNN density estimation vs graph build, as a share of build time.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--datasets-dir", default=os.path.expanduser("~/rag/hnsw-python"))
    p.add_argument("--dataset", default="mnist", choices=list(LOADERS))
    p.add_argument("--num-vectors", type=int, default=60000)
    p.add_argument("--num-query-vectors", type=int, default=100)
    p.add_argument("--top-k", type=int, default=TOP_K)
    add_common_args(p)
    return p


def _plot(df, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    van = df[df["variant"] == "vanilla"].iloc[0]
    dyn = df[df["variant"] == "dhnsw"].iloc[0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))

    # Left: every phase on a log axis -- the preprocessing phases are three
    # orders of magnitude below the build and would be invisible linearly.
    colors = {"rp_projection": "#56B4E9", "knn_density": "#0072B2",
              "cv_params": "#CC79A7", "graph_build": "#E69F00"}
    labels = PHASES + ["vanilla build"]
    values = [dyn[p] for p in PHASES] + [van["graph_build"]]
    bar_colors = [colors[p] for p in PHASES] + ["#888888"]
    ax1.barh(labels, values, color=bar_colors)
    ax1.set_xscale("log")
    for i, v in enumerate(values):
        share = "" if labels[i] == "vanilla build" else \
            f"   {v / dyn['total'] * 100:.1f}% of DHNSW"
        secs = f"{v:.2f}s" if v >= 0.01 else f"{v * 1000:.1f}ms"
        ax1.annotate(f"{secs}{share}", (v, i), textcoords="offset points",
                     xytext=(6, 0), va="center", fontsize=8.5)
    ax1.set_xlabel("Time (s, log scale)")
    ax1.set_title("Time: preprocessing is a rounding error")
    ax1.set_xlim(right=max(values) * 60)
    ax1.grid(alpha=0.3, axis="x")

    # Right: memory tells the opposite story. The paper's figure compares the
    # graphs; the preprocessing peak is transient but real, and it is what the
    # pipeline's high-water mark actually is.
    names = ["vanilla\ngraph", "DHNSW\ngraph", "DHNSW\npreprocessing"]
    mems = [van["Memory_MB"], dyn["Memory_MB"], dyn["Memory_MB_preproc"]]
    ax2.bar(names, mems, color=["#888888", "#0072B2", "#56B4E9"])
    for i, v in enumerate(mems):
        ax2.annotate(f"{v:.1f} MB", (i, v), textcoords="offset points",
                     xytext=(0, 3), ha="center", fontsize=9)
    peak = max(dyn["Memory_MB"], dyn["Memory_MB_preproc"])
    ax2.axhline(peak, ls="--", color="#D55E00", lw=1.2)
    ax2.annotate(f"pipeline high-water mark {peak:.1f} MB", (-0.45, peak),
                 textcoords="offset points", xytext=(0, 5), ha="left",
                 color="#D55E00", fontsize=8.5)
    graph_saving = (van["Memory_MB"] - dyn["Memory_MB"]) / van["Memory_MB"] * 100
    peak_saving = (van["Memory_MB"] - peak) / van["Memory_MB"] * 100
    ax2.set_ylabel("Peak memory (MB)")
    ax2.set_title(f"Memory: {graph_saving:.1f}% saved on the graph, "
                  f"{peak_saving:.1f}% on the pipeline")
    ax2.grid(alpha=0.3, axis="y")
    ax2.margins(y=0.20)

    fig.suptitle(f"C2 -- phase overhead ({van['dataset']}, N={int(van['N'])}, "
                 f"seed {int(van['seed'])})")
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot: {out_path}")


def main():
    args = build_arg_parser().parse_args()
    out, plot_path = resolve_outputs(args, f"c2_phase_overhead_{args.dataset}")

    if args.smoke:
        args.num_vectors = min(args.num_vectors, 2000)
        args.num_query_vectors = min(args.num_query_vectors, 20)

    if args.resume and os.path.exists(out):
        print(f"Nothing to measure -- {out} exists. Regenerating the figure.")
        df = pd.read_csv(out)
        if plot_path:
            _plot(df, plot_path)
        return

    print(f"Loading {args.dataset}...")
    train, queries = LOADERS[args.dataset](
        os.path.expanduser(args.datasets_dir), args.num_vectors, args.num_query_vectors)
    print(f"train={train.shape} query={queries.shape} seed={args.seed}")

    ef_vanilla = adjust_ef_by_dim(EF_VANILLA_BASE, train.shape[1])

    knn = NearestNeighbors(n_neighbors=args.top_k, algorithm="brute", metric="euclidean")
    knn.fit(train)
    true_neighbors = knn.kneighbors(queries, n_neighbors=args.top_k, return_distance=False)

    def recall_of(index):
        return float(np.mean([
            calculate_recall(true_neighbors[i], [i2 for i2, _ in index.search(q, args.top_k)])
            for i, q in enumerate(queries)]))

    rows = []

    # ---- DHNSW: density estimate, parameter derivation, then the graph build.
    # Memory is traced per stage rather than across both: the graph peak stays
    # comparable to the paper's figure, and the preprocessing peak -- which the
    # paper never reports -- becomes visible on its own.
    tracemalloc.start()
    densities, t_rp, t_knn = timed_density(train, k=M_VANILLA, seed=args.seed)
    _, peak_pre = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"  rp_projection {t_rp:.2f}s   knn_density {t_knn:.2f}s   "
          f"peak {peak_pre / 10 ** 6:.1f} MB")

    t0 = time.time()
    cv, m_start, m_end, ef_start, ef_end = set_dynamic_hnsw_params_by_std(
        densities, M_VANILLA, ef_vanilla)
    t_cv = time.time() - t0

    tracemalloc.start()
    t0 = time.time()
    dyn = DynamicHNSW("l2", densities, m_start=m_start, m_end=m_end,
                      ef_start=ef_start, ef_end=ef_end)
    for point in train:
        dyn.add(point)
    t_build = time.time() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"  graph_build {t_build:.2f}s")

    rows.append({**run_metadata(args.phase, args.dataset, "dhnsw", args.seed),
                 "N": len(train), "Dim": train.shape[1], "ef_ref": ef_vanilla,
                 "rp_projection": t_rp, "knn_density": t_knn, "cv_params": t_cv,
                 "graph_build": t_build, "total": t_rp + t_knn + t_cv + t_build,
                 "Memory_MB": peak / 10 ** 6, "Memory_MB_preproc": peak_pre / 10 ** 6,
                 "Recall_pct": recall_of(dyn),
                 "CV": cv, "Avg_Degree": dyn.get_average_neighbors()})

    # ---- Vanilla: graph build only; it has no density phase.
    tracemalloc.start()
    t0 = time.time()
    van = HNSW("l2", m=M_VANILLA, ef=EF_VANILLA_BASE)
    for point in train:
        van.add(point)
    t_build_v = time.time() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"  vanilla graph_build {t_build_v:.2f}s")

    rows.append({**run_metadata(args.phase, args.dataset, "vanilla", args.seed),
                 "N": len(train), "Dim": train.shape[1], "ef_ref": ef_vanilla,
                 "rp_projection": 0.0, "knn_density": 0.0, "cv_params": 0.0,
                 "graph_build": t_build_v, "total": t_build_v,
                 "Memory_MB": peak / 10 ** 6, "Memory_MB_preproc": 0.0,
                 "Recall_pct": recall_of(van),
                 "CV": float("nan"), "Avg_Degree": van.get_average_neighbors()})

    df = save_csv(order_columns(pd.DataFrame(rows)), out)
    print("\n" + df.to_string(index=False))

    d, v = df.iloc[0], df.iloc[1]
    print(f"\n  Graph build only:        {v['graph_build']:.2f}s -> {d['graph_build']:.2f}s "
          f"({(v['graph_build'] - d['graph_build']) / v['graph_build'] * 100:+.2f}%)")
    print(f"  Including preprocessing: {v['total']:.2f}s -> {d['total']:.2f}s "
          f"({(v['total'] - d['total']) / v['total'] * 100:+.2f}%)")
    print(f"  Density estimation is {(d['rp_projection'] + d['knn_density']) / d['total'] * 100:.1f}% "
          f"of DHNSW's total")

    if plot_path:
        _plot(df, plot_path)


if __name__ == "__main__":
    main()
