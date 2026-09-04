#!/usr/bin/env python3
"""C8: is the RP-KNN density estimate worth its cost? (review comment C8)

DHNSW estimates local density by projecting with a dense Gaussian random
projection and taking the mean distance to k nearest neighbours in the
projected space. C8 asks what happens if that projection is replaced -- by a
data-dependent one (PCA) or by a cheaper sparse one (the primitive under most
LSH families) -- and whether the extra work the paper's choice does pays for
itself.

Only the projection changes. The target dimension (d/3), the kNN step and
everything downstream are held fixed, so a difference in the result is a
difference in the projection alone. Vanilla HNSW is built once as the shared
reference; the density estimate is timed separately from the graph build,
because the whole question is cost against benefit.

Example:
    python scripts/run_c8_density_method.py --seed 42
    # -> results/phase1/c8_density_method_mnist.{csv,png}
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_baseline import (  # noqa: E402
    DENSITY_METHODS, EF_VANILLA_BASE, LOADERS, M_VANILLA, SCALE_FACTOR, TOP_K,
    adjust_ef_by_dim, calculate_density, cv_delta_factor, measure_performance,
)
from dhnsw import HNSW, DynamicHNSW  # noqa: E402
from exp_common import (  # noqa: E402
    add_common_args, order_columns, resolve_outputs, run_metadata, save_csv,
)

VANILLA_KEY = "vanilla"


def build_arg_parser():
    p = argparse.ArgumentParser(
        description="C8: RP vs PCA vs LSH for the density estimate.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--datasets-dir", default=os.path.expanduser("~/rag/hnsw-python"))
    p.add_argument("--dataset", default="mnist", choices=list(LOADERS))
    p.add_argument("--num-vectors", type=int, default=60000)
    p.add_argument("--num-query-vectors", type=int, default=100)
    p.add_argument("--top-k", type=int, default=TOP_K)
    p.add_argument("--methods", nargs="+", default=list(DENSITY_METHODS),
                   choices=list(DENSITY_METHODS))
    add_common_args(p)
    return p


def _load_existing(out_path):
    if os.path.exists(out_path):
        df = pd.read_csv(out_path)
        return df, set(df["variant"])
    return pd.DataFrame(), set()


def add_mef_columns(df):
    """The M range each method's CV produced (paper Eq. 2-3, lambda = 1.5).

    A pure function of the CV already stored per row, so it backfills onto an
    existing CSV. This runner always uses the paper's linear CV transform; the
    transform itself is C7's subject, not C8's.
    """
    lo, hi = [], []
    for _, r in df.iterrows():
        if r["variant"] == VANILLA_KEY or pd.isna(r["CV"]):
            lo.append(float("nan"))
            hi.append(float("nan"))
            continue
        delta = cv_delta_factor(r["CV"], SCALE_FACTOR, "linear")
        lo.append(max(2, int(M_VANILLA - M_VANILLA * delta)))
        hi.append(int(M_VANILLA + M_VANILLA * delta))
    df["M_low"], df["M_high"] = lo, hi
    return df


def _plot(df, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    van = df[df["variant"] == VANILLA_KEY]
    var = df[df["variant"] != VANILLA_KEY].sort_values("density_time_s")
    if van.empty or var.empty:
        print("Skipping plot: need the vanilla row and at least one method.")
        return
    van = van.iloc[0]

    # Three panels in the order the argument runs: the estimators differ in
    # cost, agree on the CV they produce, and therefore agree on the graph.
    # Axes start at zero throughout -- zoomed axes would dramatise differences
    # that are, and are meant to be seen as, negligible.
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13.5, 4.3))
    x = np.arange(len(var))
    names = list(var["variant"])

    ax1.bar(x, var["density_time_s"], 0.55, color="#0072B2")
    for i, d in enumerate(var["density_time_s"]):
        ax1.annotate(f"{d:.1f}s\n({d / (d + var['Build_Time_s'].iloc[i]) * 100:.1f}% of build)",
                     (i, d), textcoords="offset points", xytext=(0, 3),
                     ha="center", fontsize=8.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names)
    ax1.set_ylabel("Density estimate (s)")
    ax1.set_title("Cost differs")
    ax1.grid(alpha=0.3, axis="y")
    ax1.margins(y=0.30)

    ax2.bar(x, var["CV"], 0.55, color="#56B4E9")
    for i, (cv, lo, hi) in enumerate(zip(var["CV"], var["M_low"], var["M_high"])):
        ax2.annotate(f"{cv:.4f}\nM∈[{int(lo)},{int(hi)}]", (i, cv),
                     textcoords="offset points", xytext=(0, 3), ha="center",
                     fontsize=8.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names)
    ax2.set_ylabel("CV of the density estimate")
    spread = (var["CV"].max() - var["CV"].min()) / var["CV"].mean() * 100
    ax2.set_title(f"CV agrees (spread {spread:.1f}%)")
    ax2.grid(alpha=0.3, axis="y")
    ax2.margins(y=0.30)

    width = 0.38
    ax3.bar(x - width / 2, var["Recall_pct"], width, color="#0072B2", label="recall")
    ax3.bar(x + width / 2, var["Avg_Degree"], width, color="#E69F00",
            label="avg degree")
    ax3.axhline(van["Recall_pct"], ls="--", color="#666666", lw=1)
    ax3.axhline(van["Avg_Degree"], ls="--", color="#666666", lw=1)
    ax3.annotate(f"vanilla recall {van['Recall_pct']:.2f}", (-0.45, van["Recall_pct"]),
                 textcoords="offset points", xytext=(0, 3), ha="left",
                 color="#666666", fontsize=8)
    ax3.annotate(f"vanilla degree {van['Avg_Degree']:.1f}", (-0.45, van["Avg_Degree"]),
                 textcoords="offset points", xytext=(0, 3), ha="left",
                 color="#666666", fontsize=8)
    for i, (r, dg) in enumerate(zip(var["Recall_pct"], var["Avg_Degree"])):
        ax3.annotate(f"{r:.2f}", (i - width / 2, r), textcoords="offset points",
                     xytext=(0, 3), ha="center", fontsize=8)
        ax3.annotate(f"{dg:.2f}", (i + width / 2, dg), textcoords="offset points",
                     xytext=(0, 3), ha="center", fontsize=8)
    ax3.set_xticks(x)
    ax3.set_xticklabels(names)
    ax3.set_title("So the graph agrees")
    ax3.legend(frameon=False, fontsize=9, loc="center left")
    ax3.grid(alpha=0.3, axis="y")
    ax3.margins(y=0.22)

    fig.suptitle(f"C8 -- density estimate: RP vs PCA vs LSH "
                 f"({van['dataset']}, N={int(van['N'])}, seed {int(van['seed'])})")
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot: {out_path}")


def main():
    args = build_arg_parser().parse_args()
    out, plot_path = resolve_outputs(args, f"c8_density_method_{args.dataset}")

    if args.smoke:
        args.num_vectors = min(args.num_vectors, 2000)
        args.num_query_vectors = min(args.num_query_vectors, 20)

    existing_df, done = (_load_existing(out) if args.resume
                         else (pd.DataFrame(), set()))
    rows = existing_df.to_dict("records")

    needed = ([] if VANILLA_KEY in done else [VANILLA_KEY]) + \
             [m for m in args.methods if m not in done]
    if not needed:
        print(f"Nothing to measure -- {out} already has every requested method.")
        df = save_csv(order_columns(add_mef_columns(existing_df)), out)
        print("\n" + df.to_string(index=False))
        if plot_path:
            _plot(df, plot_path)
        return

    print(f"Loading {args.dataset}...")
    train, queries = LOADERS[args.dataset](
        os.path.expanduser(args.datasets_dir), args.num_vectors, args.num_query_vectors)
    print(f"train={train.shape} query={queries.shape} seed={args.seed}  to run: {needed}")

    ef_vanilla = adjust_ef_by_dim(EF_VANILLA_BASE, train.shape[1])

    knn = NearestNeighbors(n_neighbors=args.top_k, algorithm="brute", metric="euclidean")
    knn.fit(train)
    true_neighbors = knn.kneighbors(queries, n_neighbors=args.top_k, return_distance=False)

    def save():
        return save_csv(order_columns(add_mef_columns(pd.DataFrame(rows))), out)

    common = {"N": len(train), "Dim": train.shape[1], "ef_ref": ef_vanilla}

    for variant in needed:
        if variant == VANILLA_KEY:
            res = measure_performance(HNSW, train, queries, true_neighbors,
                                      f"Vanilla HNSW ({args.dataset})", k=args.top_k)
            extra = {"density_time_s": 0.0}
        else:
            t0 = time.time()
            densities = calculate_density(train, k=M_VANILLA, seed=args.seed,
                                          method=variant)
            t_density = time.time() - t0
            print(f"  density[{variant}]: {t_density:.2f}s")
            res = measure_performance(DynamicHNSW, train, queries, true_neighbors,
                                      f"DHNSW[{variant}] ({args.dataset})", k=args.top_k,
                                      densities=densities, ef_vanilla=ef_vanilla)
            extra = {"density_time_s": t_density}

        rows.append({**run_metadata(args.phase, args.dataset, variant, args.seed),
                     **common, **extra, **res})
        save()

    df = save()
    print("\n" + df.to_string(index=False))

    van = df[df["variant"] == VANILLA_KEY].iloc[0]
    for _, r in df[df["variant"] != VANILLA_KEY].iterrows():
        total = r["Build_Time_s"] + r["density_time_s"]
        print(f"  {r['variant']:5s} estimate {r['density_time_s']:7.2f}s  "
              f"build {r['Build_Time_s']:7.2f}s  total {total:7.2f}s "
              f"({(van['Build_Time_s'] - total) / van['Build_Time_s'] * 100:+6.2f}% vs vanilla)  "
              f"recall {r['Recall_pct']:.2f}%  CV {r['CV']:.4f}")

    if plot_path:
        _plot(df, plot_path)


if __name__ == "__main__":
    main()
