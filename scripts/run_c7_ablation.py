#!/usr/bin/env python3
"""C7 ablation: CV-term transform variants (Chapter 3, review comment C7).

The paper's Eq. (2)-(5) scale M/ef ranges by `cv * lambda`, where cv =
std/mean of the RP-KNN local densities ("linear" below -- the paper's
formula, unchanged). This was validated once during the original research
but no results were kept, so C7 asks for it to be re-run and reported.

This script reuses scripts/run_baseline.py unmodified: it loads the dataset,
computes the RP-KNN densities and the brute-force ground truth *once*, runs
vanilla HNSW *once* (it does not depend on the CV term), and then runs
DynamicHNSW once per `--cv-transform` variant against that shared density
array -- so the comparison isolates the CV-term formula, and the sweep costs
one vanilla build plus N dynamic builds rather than 2N.

Example (full MNIST, matches the golden master's seed):
    python scripts/run_c7_ablation.py --seed 42 \
        --out ablation/c7_cv_transform_mnist.csv \
        --plot ablation/c7_cv_transform_mnist.png

Smoke test (~60s):
    python scripts/run_c7_ablation.py --smoke
"""
import argparse
import os
import sys

import pandas as pd
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_baseline import (  # noqa: E402
    CV_TRANSFORMS, EF_VANILLA_BASE, LOADERS, M_VANILLA, SCALE_FACTOR, TOP_K,
    adjust_ef_by_dim, calculate_density_random_projection, cv_delta_factor,
    measure_performance,
)
from dhnsw import HNSW, DynamicHNSW  # noqa: E402

VANILLA_KEY = "vanilla"


def build_arg_parser():
    p = argparse.ArgumentParser(
        description="C7 ablation: compare CV-term transforms for DHNSW's M/ef scaling.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--datasets-dir", default=os.path.expanduser("~/rag/hnsw-python"),
                   help="Directory holding mnist/ glove100k/ sift1m/ gist/ (read-only).")
    p.add_argument("--dataset", default="mnist", choices=list(LOADERS),
                   help="Range dial 1 (PLAN.md): a single dataset to start.")
    p.add_argument("--num-vectors", type=int, default=60000)
    p.add_argument("--num-query-vectors", type=int, default=100)
    p.add_argument("--top-k", type=int, default=TOP_K)
    p.add_argument("--seed", type=int, default=42,
                   help="Shared random_state for the RP density estimate, so every "
                        "transform sees the same densities. Default matches the golden "
                        "master (baseline/dhnsw_mnist.csv).")
    p.add_argument("--transforms", nargs="+", default=list(CV_TRANSFORMS),
                   choices=list(CV_TRANSFORMS))
    p.add_argument("--smoke", action="store_true",
                   help="Small run (~60s) to sanity-check the pipeline: caps "
                        "num-vectors/num-query-vectors instead of using the full dataset.")
    p.add_argument("--resume", action="store_true",
                   help="Skip variants already present in --out and append the rest.")
    p.add_argument("--out", default="ablation/c7_cv_transform_mnist.csv",
                   help="CSV path to write (appended to incrementally, per variant).")
    p.add_argument("--plot", default="ablation/c7_cv_transform_mnist.png",
                   help="Comparison chart path. Empty string skips plotting.")
    return p


def _load_existing(out_path):
    if os.path.exists(out_path):
        df = pd.read_csv(out_path)
        return df, set(df["CV_Transform"])
    return pd.DataFrame(), set()


def add_derived_columns(df):
    """Recover the M/ef ranges each variant actually ran with.

    These are pure functions of the CV already stored per row (and of ef_ref),
    so they can be backfilled onto an existing CSV without re-running anything.
    Kept identical to set_dynamic_hnsw_params_by_std's arithmetic, including
    the int() truncation, so the recorded ranges are the ones that ran.
    Vanilla is left blank: it uses fixed M=16 / ef=100, not a range.
    """
    cols = {"Delta": [], "M_low": [], "M_high": [], "ef_low": [], "ef_high": []}
    for _, r in df.iterrows():
        if r["CV_Transform"] == VANILLA_KEY:
            for c in cols:
                cols[c].append(float("nan"))
            continue
        delta = cv_delta_factor(r["CV"], SCALE_FACTOR, r["CV_Transform"])
        ef_ref = r["ef_ref"]
        cols["Delta"].append(delta)
        cols["M_low"].append(max(2, int(M_VANILLA - M_VANILLA * delta)))
        cols["M_high"].append(int(M_VANILLA + M_VANILLA * delta))
        cols["ef_low"].append(max(10, int(ef_ref - ef_ref * delta)))
        cols["ef_high"].append(int(ef_ref + ef_ref * delta))
    for c, v in cols.items():
        df[c] = v
    return df


def _plot(df, dataset, out_path):
    """Plot every variant against delta, the range-expansion factor it produces.

    A bar chart per metric hides the result: the transforms do not differ in
    kind, only in the delta they map the same CV onto, and every metric is
    monotone in that one scalar. Putting delta on the x-axis shows that -- and
    shows the recall cliff that opens once M_low is driven too low.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    van = df[df["CV_Transform"] == VANILLA_KEY]
    var = df[df["CV_Transform"] != VANILLA_KEY].sort_values("Delta")
    if van.empty or var.empty:
        print("Skipping plot: need the vanilla row and at least one variant.")
        return
    van = van.iloc[0]

    build_imp = (van["Build_Time_s"] - var["Build_Time_s"]) / van["Build_Time_s"] * 100
    mem_imp = (van["Memory_MB"] - var["Memory_MB"]) / van["Memory_MB"] * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))

    ax1.plot(var["Delta"], mem_imp, "o-", color="#0072B2", label="Memory saved")
    ax1.plot(var["Delta"], build_imp, "s-", color="#E69F00", label="Build time saved")
    ax1.set_xlabel(r"$\delta$  (range-expansion factor)")
    ax1.set_ylabel("Improvement vs vanilla (%)")
    ax1.set_title("Savings grow monotonically with $\\delta$")
    ax1.legend(loc="upper left", frameon=False)
    ax1.grid(alpha=0.3)

    ax2.plot(var["Delta"], var["Recall_pct"], "o-", color="#0072B2")
    ax2.axhline(van["Recall_pct"], ls="--", color="#888888", lw=1)
    ax2.annotate("vanilla", (var["Delta"].max(), van["Recall_pct"]),
                 textcoords="offset points", xytext=(0, 4), ha="right",
                 color="#888888", fontsize=9)
    ax2.set_xlabel(r"$\delta$  (range-expansion factor)")
    ax2.set_ylabel("Recall@k (%)")
    ax2.set_title("Recall holds, then falls off a cliff")
    ax2.grid(alpha=0.3)

    # exp and ratio land at nearly the same delta, so alternate the label side
    # to keep neighbouring annotations from printing on top of each other.
    for ax, ys, show_mlow in ((ax1, mem_imp, False), (ax2, var["Recall_pct"], True)):
        ax.margins(x=0.14, y=0.22)
        for i, (x, y, name, mlow) in enumerate(
                zip(var["Delta"], ys, var["CV_Transform"], var["M_low"])):
            label = f"{name}\n$M_{{low}}$={int(mlow)}" if show_mlow else name
            ax.annotate(label, (x, y), textcoords="offset points",
                        xytext=(0, 13 if i % 2 else -31), ha="center", fontsize=8.5)

    fig.suptitle(f"C7 -- CV-term transform ablation ({dataset}, N={int(van['N'])}, "
                 f"seed {int(van['seed'])})")
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot: {out_path}")


def main():
    args = build_arg_parser().parse_args()

    if args.smoke:
        args.num_vectors = min(args.num_vectors, 2000)
        args.num_query_vectors = min(args.num_query_vectors, 20)

    existing_df, done = (_load_existing(args.out) if args.resume
                         else (pd.DataFrame(), set()))
    rows = existing_df.to_dict("records")

    needed = ([] if VANILLA_KEY in done else [VANILLA_KEY]) + \
             [t for t in args.transforms if t not in done]
    if not needed:
        # Everything measured already: refresh the derived columns and the
        # figure from the stored CSV, so the chart can be regenerated without
        # paying for the sweep again.
        print(f"Nothing to measure -- {args.out} already has every requested variant.")
        df = add_derived_columns(existing_df)
        df.to_csv(args.out, index=False)
        print("\n" + df.to_string(index=False))
        if args.plot:
            _plot(df, str(df["Dataset"].iloc[0]), args.plot)
        return

    print(f"Loading {args.dataset} dataset...")
    train, queries = LOADERS[args.dataset](
        os.path.expanduser(args.datasets_dir), args.num_vectors, args.num_query_vectors)
    print(f"train={train.shape} query={queries.shape}  seed={args.seed}  "
          f"variants to run: {needed}")

    ef_vanilla = adjust_ef_by_dim(EF_VANILLA_BASE, train.shape[1])
    densities = calculate_density_random_projection(train, k=M_VANILLA, seed=args.seed)

    knn = NearestNeighbors(n_neighbors=args.top_k, algorithm="brute", metric="euclidean")
    knn.fit(train)
    true_neighbors = knn.kneighbors(queries, n_neighbors=args.top_k, return_distance=False)

    def save():
        df = add_derived_columns(pd.DataFrame(rows))
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        df.to_csv(args.out, index=False)
        return df

    common = {"Dataset": args.dataset, "N": len(train), "Dim": train.shape[1],
              "ef_ref": ef_vanilla, "seed": args.seed}

    for variant in needed:
        if variant == VANILLA_KEY:
            res = measure_performance(HNSW, train, queries, true_neighbors,
                                      f"Vanilla HNSW ({args.dataset})", k=args.top_k)
        else:
            res = measure_performance(DynamicHNSW, train, queries, true_neighbors,
                                      f"DHNSW[{variant}] ({args.dataset})", k=args.top_k,
                                      densities=densities, ef_vanilla=ef_vanilla,
                                      cv_transform=variant)
        rows.append({**common, "CV_Transform": variant, **res})
        save()  # incremental, so --resume is safe against a mid-sweep interruption

    df = save()
    print("\n" + df.to_string(index=False))

    if args.plot:
        _plot(df, args.dataset, args.plot)


if __name__ == "__main__":
    main()
