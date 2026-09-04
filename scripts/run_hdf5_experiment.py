#!/usr/bin/env python3
"""Vanilla HNSW vs DHNSW on the HDF5 datasets -- serves C3 and C10.

Both comments ask the same question of different data, so they share a runner:

  C3  does the gain survive at 1536d+?          simplewiki-openai (3072d)
  C10 does it survive a change of backbone?     landmark-dino vs landmark-nomic
      (same corpus, same dimension, different encoder -- a controlled pair)

Metrics stay the paper's: build time, peak memory, recall. Vectors are
unit-normalised at load (see hdf5_loaders), so the "l2" distance function is
cosine ranking and the code path matches the paper's MNIST runs exactly.

Scale is a range-dial-1 compromise: pure-Python DHNSW cannot build a 760K
graph, so --num-vectors subsamples. The comparison is vanilla vs DHNSW on
identical data, which is what the improvement rate needs; absolute numbers are
not comparable to the published table and are not meant to be.

Examples:
    python scripts/run_hdf5_experiment.py --datasets simplewiki-openai \
        --num-vectors 20000 --stem c3_highdim
    python scripts/run_hdf5_experiment.py --datasets landmark-dino landmark-nomic \
        --num-vectors 30000 --stem c10_backbone
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
    EF_VANILLA_BASE, M_VANILLA, TOP_K, adjust_ef_by_dim, calculate_density,
    measure_performance,
)
from dhnsw import HNSW, DynamicHNSW  # noqa: E402
from hdf5_loaders import HDF5_DATASETS, load_hdf5  # noqa: E402
from exp_common import (  # noqa: E402
    add_common_args, order_columns, resolve_outputs, run_metadata, save_csv,
)

VANILLA_KEY = "vanilla"
MATCHED_KEY = "vanilla_ef_ref"
DHNSW_KEY = "dhnsw"
ALL_VARIANTS = (VANILLA_KEY, MATCHED_KEY, DHNSW_KEY)


def build_arg_parser():
    p = argparse.ArgumentParser(
        description="Vanilla vs DHNSW on the HDF5 datasets (C3, C10).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--datasets", nargs="+", default=["simplewiki-openai"],
                   choices=list(HDF5_DATASETS))
    p.add_argument("--data-dir", default=None)
    p.add_argument("--num-vectors", type=int, default=20000,
                   help="Subsample size. Pure-Python cost is roughly linear in this.")
    p.add_argument("--num-query-vectors", type=int, default=100)
    p.add_argument("--top-k", type=int, default=TOP_K)
    p.add_argument("--stem", default="hdf5_experiment",
                   help="Output filename stem, e.g. c3_highdim or c10_backbone.")
    p.add_argument("--variants", nargs="+", default=[VANILLA_KEY, DHNSW_KEY],
                   choices=list(ALL_VARIANTS),
                   help="Add 'vanilla_ef_ref' to also build vanilla at DHNSW's ef_ref. "
                        "That control matters wherever Eq. (1) puts ef_ref far from the "
                        "vanilla ef=100 -- at 3072d it asks for ef=1000, so without it "
                        "a slower DHNSW build cannot be told apart from a larger ef.")
    add_common_args(p)
    return p


def _load_existing(out_path):
    if os.path.exists(out_path):
        df = pd.read_csv(out_path)
        return df, set(zip(df["dataset"], df["variant"]))
    return pd.DataFrame(), set()


def _plot(df, out_path, title_text):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    datasets = list(dict.fromkeys(df["dataset"]))
    variants = [v for v in ALL_VARIANTS if v in set(df["variant"])]
    if not datasets or not variants:
        print("Skipping plot: nothing measured yet.")
        return

    colors = {VANILLA_KEY: "#888888", MATCHED_KEY: "#009E73", DHNSW_KEY: "#0072B2"}
    metrics = [("Build_Time_s", "Build time (s)"),
               ("Memory_MB", "Peak memory (MB)"),
               ("Recall_pct", "Recall@k (%)")]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
    x = np.arange(len(datasets))
    width = 0.8 / len(variants)

    for ax, (col, title) in zip(axes, metrics):
        for j, variant in enumerate(variants):
            vals, offs = [], []
            for i, ds in enumerate(datasets):
                sub = df[(df["dataset"] == ds) & (df["variant"] == variant)]
                vals.append(sub.iloc[0][col] if not sub.empty else np.nan)
                offs.append(i + (j - (len(variants) - 1) / 2) * width)
            ax.bar(offs, vals, width, color=colors[variant],
                   label=variant if ax is axes[0] else None)
            for xo, v in zip(offs, vals):
                if not np.isnan(v):
                    fmt = f"{v:.1f}" if col != "Build_Time_s" else f"{v:.0f}"
                    ax.annotate(fmt, (xo, v), textcoords="offset points",
                                xytext=(0, 3), ha="center", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, fontsize=9)
        ax.set_title(title)
        ax.grid(alpha=0.3, axis="y")
        ax.margins(y=0.20)
    axes[0].legend(frameon=False, fontsize=9)

    fig.suptitle(title_text)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot: {out_path}")


def main():
    args = build_arg_parser().parse_args()
    out, plot_path = resolve_outputs(args, args.stem)

    if args.smoke:
        args.num_vectors = min(args.num_vectors, 1500)
        args.num_query_vectors = min(args.num_query_vectors, 10)

    existing_df, done = (_load_existing(out) if args.resume
                         else (pd.DataFrame(), set()))
    rows = existing_df.to_dict("records")

    def save():
        return save_csv(order_columns(pd.DataFrame(rows)), out)

    for name in args.datasets:
        needed = [v for v in args.variants if (name, v) not in done]
        if not needed:
            print(f"{name}: already measured, skipping.")
            continue

        print(f"\nLoading {name} (subsample {args.num_vectors}, seed {args.seed})...")
        t0 = time.time()
        train, queries = load_hdf5(name, args.num_vectors, args.num_query_vectors,
                                   seed=args.seed, data_dir=args.data_dir)
        print(f"  train={train.shape} query={queries.shape}  load {time.time() - t0:.1f}s")

        ef_vanilla = adjust_ef_by_dim(EF_VANILLA_BASE, train.shape[1])

        t0 = time.time()
        knn = NearestNeighbors(n_neighbors=args.top_k, algorithm="brute",
                               metric="euclidean")
        knn.fit(train)
        true_neighbors = knn.kneighbors(queries, n_neighbors=args.top_k,
                                        return_distance=False)
        print(f"  ground truth recomputed over the subsample in {time.time() - t0:.1f}s")

        densities = None
        if DHNSW_KEY in needed:
            t0 = time.time()
            densities = calculate_density(train, k=M_VANILLA, seed=args.seed)
            t_density = time.time() - t0
            print(f"  density estimate {t_density:.1f}s")
        else:
            t_density = float("nan")

        common = {"N": len(train), "Dim": train.shape[1], "ef_ref": ef_vanilla}
        for variant in needed:
            if variant == VANILLA_KEY:
                res = measure_performance(HNSW, train, queries, true_neighbors,
                                          f"Vanilla HNSW ({name})", k=args.top_k)
                extra = {"density_time_s": 0.0, "ef_used": EF_VANILLA_BASE}
            elif variant == MATCHED_KEY:
                # Vanilla at DHNSW's ef budget: isolates Eq. (1)'s effect on
                # build cost from DHNSW's per-node adaptation.
                res = measure_performance(HNSW, train, queries, true_neighbors,
                                          f"Vanilla HNSW @ef_ref={ef_vanilla} ({name})",
                                          k=args.top_k, ef_vanilla=ef_vanilla)
                extra = {"density_time_s": 0.0, "ef_used": ef_vanilla}
            else:
                res = measure_performance(DynamicHNSW, train, queries, true_neighbors,
                                          f"DHNSW ({name})", k=args.top_k,
                                          densities=densities, ef_vanilla=ef_vanilla)
                extra = {"density_time_s": t_density, "ef_used": ef_vanilla}
            rows.append({**run_metadata(args.phase, name, variant, args.seed),
                         **common, **extra, **res})
            save()

    df = save()
    print("\n" + df.to_string(index=False))

    for name in dict.fromkeys(df["dataset"]):
        sub = df[df["dataset"] == name]
        v = sub[sub["variant"] == VANILLA_KEY]
        d = sub[sub["variant"] == DHNSW_KEY]
        if v.empty or d.empty:
            continue
        v, d = v.iloc[0], d.iloc[0]
        print(f"\n  [{name}] build {(v['Build_Time_s'] - d['Build_Time_s']) / v['Build_Time_s'] * 100:+.2f}%   "
              f"memory {(v['Memory_MB'] - d['Memory_MB']) / v['Memory_MB'] * 100:+.2f}%   "
              f"recall {v['Recall_pct']:.2f}% -> {d['Recall_pct']:.2f}%   "
              f"degree {v['Avg_Degree']:.2f} -> {d['Avg_Degree']:.2f}")

    if plot_path:
        _plot(df, plot_path, f"{args.stem} (N={args.num_vectors}, seed {args.seed})")


if __name__ == "__main__":
    main()
