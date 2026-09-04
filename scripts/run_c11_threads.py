#!/usr/bin/env python3
"""C11: how build time scales with threads (review comment C11).

This is a build-time experiment, not a search one -- the DHNSW paper reports
build time, memory and recall, and never QPS. The question is whether DHNSW's
build-time advantage survives parallel construction, or whether it is an
artefact of single-threaded measurement.

Requires the C++ port in cpp/, since pure-Python DHNSW has no parallel
construction at all. Vanilla is stock hnswlib; DHNSW passes the per-node M and
ef arrays derived from the same RP-KNN density estimate the Python
implementation uses (the estimate itself is not ported -- it stays in sklearn).

Pin to the P-cores before running. This machine is an i7-12700F: logical CPUs
0-15 are the 8 P-cores (two threads each), 16-19 are the 4 E-cores at a lower
clock, so a sweep past 16 threads measures core heterogeneity rather than
scaling:

    taskset -c 0-15 python scripts/run_c11_threads.py --seed 42
    # -> results/phase1/c11_threads_mnist.{csv,png}

Memory is reported as the actual number of stored edges. hnswlib allocates
maxM slots per node whether or not they are filled, so allocated bytes cannot
show the saving the paper attributes to reduced average degree (Fig. 5).
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hnswlib  # noqa: E402
from run_baseline import (  # noqa: E402
    EF_VANILLA_BASE, LOADERS, M_VANILLA, TOP_K, adjust_ef_by_dim,
    calculate_density, set_dynamic_hnsw_params_by_std,
)
from exp_common import (  # noqa: E402
    add_common_args, order_columns, resolve_outputs, run_metadata, save_csv,
)

VANILLA_KEY = "vanilla"
DHNSW_KEY = "dhnsw"
# DHNSW's per-node M with ef held at the vanilla 100. DHNSW moves two knobs at
# once -- M down (cheaper) and ef up (dearer, since ef_ref > 100 by Eq. 1) --
# and in C++ they do not cancel the way they do in Python. This arm separates
# them: it is DHNSW's degree adaptation alone.
DHNSW_FIXED_EF_KEY = "dhnsw_ef100"
VARIANTS = (VANILLA_KEY, DHNSW_KEY, DHNSW_FIXED_EF_KEY)


def dynamic_params(densities, m_start, m_end, ef_start, ef_end):
    """Per-node M and ef, matching DynamicHNSW._get_dynamic_m/_get_dynamic_ef.

    Same linear interpolation between the range endpoints on the normalised
    density, same clamping, same int() truncation.
    """
    lo, hi = densities.min(), densities.max()
    scaled = (densities - lo) / (hi - lo) if hi > lo else np.zeros_like(densities)
    m = np.clip((m_start + scaled * (m_end - m_start)).astype(np.int64), m_start, m_end)
    ef = np.clip((ef_start + scaled * (ef_end - ef_start)).astype(np.int64),
                 ef_start, ef_end)
    return m.astype(np.uint64), ef.astype(np.uint64)


def build_index(data, dim, variant, threads, m_index, ef_c, m_arr=None, ef_arr=None,
                m0=0):
    idx = hnswlib.Index(space="l2", dim=dim)
    idx.init_index(max_elements=len(data), M=m_index, ef_construction=ef_c,
                   random_seed=100)
    idx.set_num_threads(threads)
    ids = np.arange(len(data))
    t0 = time.time()
    if variant == VANILLA_KEY:
        idx.add_items(data, ids, num_threads=threads)
    else:
        idx.add_items_dynamic(data, m_arr, ef_arr, m0, ids, num_threads=threads)
    return idx, time.time() - t0


def build_arg_parser():
    p = argparse.ArgumentParser(
        description="C11: build-time thread scaling, vanilla vs DHNSW (C++ port).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--datasets-dir", default=os.path.expanduser("~/rag/hnsw-python"))
    p.add_argument("--dataset", default="mnist", choices=list(LOADERS))
    p.add_argument("--num-vectors", type=int, default=60000)
    p.add_argument("--num-query-vectors", type=int, default=100)
    p.add_argument("--top-k", type=int, default=TOP_K)
    p.add_argument("--threads", nargs="+", type=int,
                   default=list(range(1, 17)),
                   help="Thread counts to sweep. Keep at or below 16 under "
                        "taskset -c 0-15; beyond that the E-cores join in.")
    p.add_argument("--repeats", type=int, default=3,
                   help="Builds per point. C++ builds are cheap enough to repeat, "
                        "which is what makes the build-time noise measurable.")
    p.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    add_common_args(p)
    return p


def _plot(df, out_path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    agg = (df.groupby(["variant", "threads"])["build_time_s"]
             .agg(["median", "min", "max"]).reset_index())
    colors = {VANILLA_KEY: "#888888", DHNSW_KEY: "#0072B2",
              DHNSW_FIXED_EF_KEY: "#009E73"}

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13.5, 4.3))

    for variant in [v for v in VARIANTS if v in set(agg["variant"])]:
        a = agg[agg["variant"] == variant].sort_values("threads")
        ax1.plot(a["threads"], a["median"], "o-", color=colors[variant], label=variant)
        ax1.fill_between(a["threads"], a["min"], a["max"], color=colors[variant],
                         alpha=0.25)
        base = a[a["threads"] == a["threads"].min()]["median"].iloc[0]
        ax2.plot(a["threads"], base / a["median"], "o-", color=colors[variant],
                 label=variant)
    ax1.set_xlabel("Threads")
    ax1.set_ylabel("Build time (s)")
    ax1.set_title("Build time")
    ax1.legend(frameon=False, fontsize=9)
    ax1.grid(alpha=0.3)

    t = sorted(set(agg["threads"]))
    ax2.plot(t, t, "--", color="#333333", lw=1, label="ideal")
    ax2.set_xlabel("Threads")
    ax2.set_ylabel("Speedup vs 1 thread")
    ax2.set_title("Parallel scaling")
    ax2.legend(frameon=False, fontsize=9)
    ax2.grid(alpha=0.3)

    # DHNSW's advantage as a function of thread count: the actual C11 question.
    piv = agg.pivot(index="threads", columns="variant", values="median")
    if VANILLA_KEY in piv:
        for variant in (DHNSW_KEY, DHNSW_FIXED_EF_KEY):
            if variant in piv:
                gain = (piv[VANILLA_KEY] - piv[variant]) / piv[VANILLA_KEY] * 100
                ax3.plot(piv.index, gain, "o-", color=colors[variant], label=variant)
        ax3.axhline(0, color="#333333", lw=0.8)
        ax3.set_xlabel("Threads")
        ax3.set_ylabel("Build time saved vs vanilla (%)")
        ax3.set_title("Does the advantage survive threading?")
        ax3.legend(frameon=False, fontsize=9)
        ax3.grid(alpha=0.3)
        ax3.margins(y=0.25)

    fig.suptitle(title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot: {out_path}")


def main():
    args = build_arg_parser().parse_args()
    out, plot_path = resolve_outputs(args, f"c11_threads_{args.dataset}")

    if args.smoke:
        args.num_vectors = min(args.num_vectors, 10000)
        args.threads = [1, 2, 4]
        args.repeats = 1

    done = set()
    rows = []
    if args.resume and os.path.exists(out):
        prev = pd.read_csv(out)
        rows = prev.to_dict("records")
        done = set(zip(prev["variant"], prev["threads"], prev["repeat"]))

    print(f"Loading {args.dataset}...")
    train, queries = LOADERS[args.dataset](
        os.path.expanduser(args.datasets_dir), args.num_vectors, args.num_query_vectors)
    train = np.ascontiguousarray(train, dtype=np.float32)
    queries = np.ascontiguousarray(queries, dtype=np.float32)
    dim = train.shape[1]
    print(f"train={train.shape} query={queries.shape}  threads={args.threads}  "
          f"repeats={args.repeats}")

    knn = NearestNeighbors(n_neighbors=args.top_k, algorithm="brute", metric="euclidean")
    knn.fit(train)
    truth = knn.kneighbors(queries, n_neighbors=args.top_k, return_distance=False)

    ef_ref = adjust_ef_by_dim(EF_VANILLA_BASE, dim)
    densities = calculate_density(train, k=M_VANILLA, seed=args.seed)
    cv, m_start, m_end, ef_start, ef_end = set_dynamic_hnsw_params_by_std(
        densities, M_VANILLA, ef_ref)
    m_arr, ef_arr = dynamic_params(densities, m_start, m_end, ef_start, ef_end)
    print(f"  per-node M mean {m_arr.mean():.2f} in [{m_arr.min()}, {m_arr.max()}], "
          f"ef mean {ef_arr.mean():.1f}")

    def recall_of(idx):
        idx.set_ef(max(EF_VANILLA_BASE, args.top_k))
        labels, _ = idx.knn_query(queries, k=args.top_k, num_threads=1)
        return float(np.mean([len(set(t) & set(l)) / len(t)
                              for t, l in zip(truth, labels)])) * 100

    ef_flat = np.full(len(train), EF_VANILLA_BASE, dtype=np.uint64)
    for variant in args.variants:
        # Vanilla is built at the paper's M=16/ef=100. DHNSW's index is sized
        # at m_end so the link lists can hold its largest per-node degree; the
        # per-node arrays then hold each insertion to its own budget.
        m_index = M_VANILLA if variant == VANILLA_KEY else m_end
        ef_c = EF_VANILLA_BASE if variant in (VANILLA_KEY, DHNSW_FIXED_EF_KEY) else ef_end
        ef_this = ef_flat if variant == DHNSW_FIXED_EF_KEY else ef_arr
        # Level-0 budget: the paper's Python fixes it at 2*m_start for every
        # node (HNSW.__init__ derives _m0 from m_start and DynamicHNSW.add only
        # varies _m), and level 0 holds every node, so this constant is where
        # most of the reported memory saving comes from.
        m0 = 2 * m_start
        for threads in args.threads:
            for rep in range(args.repeats):
                if (variant, threads, rep) in done:
                    continue
                idx, elapsed = build_index(train, dim, variant, threads, m_index, ef_c,
                                           m_arr, ef_this, m0)
                edges = idx.get_total_edge_count()
                rec = recall_of(idx) if rep == 0 else float("nan")
                print(f"  {variant:7s} threads={threads:2d} rep={rep} "
                      f"{elapsed:7.2f}s  edges={edges}"
                      + (f"  recall={rec:.2f}%" if rep == 0 else ""))
                rows.append({**run_metadata(args.phase, args.dataset, variant, args.seed),
                             "N": len(train), "Dim": dim, "ef_ref": ef_ref,
                             "threads": threads, "repeat": rep,
                             "build_time_s": elapsed, "edges": edges,
                             "avg_degree": edges / len(train),
                             "Recall_pct": rec, "CV": cv,
                             "M_index": m_index, "ef_construction": ef_c})
                save_csv(order_columns(pd.DataFrame(rows)), out)
                del idx

    df = save_csv(order_columns(pd.DataFrame(rows)), out)

    agg = df.groupby(["variant", "threads"])["build_time_s"].median().unstack(0)
    print("\nMedian build time (s):\n" + agg.to_string())
    if VANILLA_KEY in agg:
        for variant in (DHNSW_KEY, DHNSW_FIXED_EF_KEY):
            if variant in agg:
                gain = (agg[VANILLA_KEY] - agg[variant]) / agg[VANILLA_KEY] * 100
                print(f"\n{variant} build time saved vs vanilla (%):\n"
                      + gain.round(2).to_string())

    e = df.groupby("variant")["edges"].median()
    r = df.groupby("variant")["Recall_pct"].median()
    if VANILLA_KEY in e:
        for variant in (DHNSW_KEY, DHNSW_FIXED_EF_KEY):
            if variant in e:
                print(f"\nEdges: vanilla {e[VANILLA_KEY]:.0f} -> {variant} {e[variant]:.0f} "
                      f"({(e[VANILLA_KEY] - e[variant]) / e[VANILLA_KEY] * 100:+.2f}%)   "
                      f"recall {r[VANILLA_KEY]:.2f}% -> {r[variant]:.2f}%")

    if plot_path:
        _plot(df, plot_path,
              f"C11 -- build-time thread scaling ({args.dataset}, N={len(train)}, "
              f"P-cores only, median of {args.repeats})")


if __name__ == "__main__":
    main()
