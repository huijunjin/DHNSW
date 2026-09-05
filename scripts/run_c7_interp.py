#!/usr/bin/env python3
"""C7, second axis: the interpolation curve inside the [M_low, M_high] range.

The CV-term ablation (`run_c7_ablation.py`) found that all four transforms
reduce to a single scalar delta, so swapping one for another is equivalent to
retuning lambda. That axis is degenerate. This one is not: Eq. (6)-(7) place
each node *linearly* between the range endpoints, and changing that curve
changes which nodes get the high M -- not just how wide the range is.

`DynamicHNSW` is subclassed rather than edited; `dhnsw.py` stays as published.
`--interp linear` reproduces it exactly, so it doubles as the correctness check.

    python scripts/run_c7_interp.py --seed 42
    # -> results/phase1/c7_interp_mnist.csv
"""
import argparse
import os
import subprocess
import sys
import time
import tracemalloc

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dhnsw import HNSW, DynamicHNSW
import run_baseline as rb


# --------------------------------------------------------------- shape curves
def _logistic(x, k=10.0, x0=0.5):
    """Steep in the middle, flat at both ends -- the curve `my_hnsw_v4.py` used.
    Renormalised so f(0) = 0 and f(1) = 1; the raw logistic only spans
    0.0067-0.9933 over [0, 1] and would otherwise shrink the effective range."""
    raw = 1.0 / (1.0 + np.exp(-k * (x - x0)))
    lo = 1.0 / (1.0 + np.exp(k * x0))
    hi = 1.0 / (1.0 + np.exp(-k * (1.0 - x0)))
    return (raw - lo) / (hi - lo)


def _inv_logistic(x, p=0.5):
    """Steep at both ends, flat in the middle -- the logistic turned inside out."""
    t = 2.0 * x - 1.0
    return 0.5 + 0.5 * np.sign(t) * np.abs(t) ** p


SHAPES = {
    "linear":       lambda x: x,                  # Eq. (6)-(7) as published
    "logistic":     _logistic,                    # middle steep, ends flat
    "convex":       lambda x: x ** 2,             # flat then steep
    "concave":      lambda x: np.sqrt(x),         # steep then flat
    "inv_logistic": _inv_logistic,                # ends steep, middle flat
}


class ShapedDynamicHNSW(DynamicHNSW):
    """DynamicHNSW with the linear placement of Eq. (6)-(7) replaced by `shape`.

    Only the position inside [m_start, m_end] changes; the range itself still
    comes from the CV term, and every other code path is the published one.
    """

    def __init__(self, *args, shape="linear", **kwargs):
        super().__init__(*args, **kwargs)
        self._shape = SHAPES[shape]
        self._span = float(self.max_density - self.min_density) or 1.0

    def _frac(self, index):
        x = (self.densities[index] - self.min_density) / self._span
        x = min(1.0, max(0.0, float(x)))
        if self.invert_density:
            x = 1.0 - x
        return float(self._shape(x))

    def _get_dynamic_m(self, index):
        m = self.m_start + self._frac(index) * (self.m_end - self.m_start)
        return int(min(self.m_end, max(self.m_start, m)))

    def _get_dynamic_ef(self, index):
        ef = self.ef_start + self._frac(index) * (self.ef_end - self.ef_start)
        return int(min(self.ef_end, max(self.ef_start, ef)))


def measure(cls, data, queries, gt, label, k, ef_vanilla, densities=None, shape=None):
    tracemalloc.start()
    t0 = time.time()
    if cls is ShapedDynamicHNSW:
        cv, m_lo, m_hi, ef_lo, ef_hi = rb.set_dynamic_hnsw_params_by_std(
            densities, rb.M_VANILLA, ef_vanilla)
        idx = cls("l2", densities, m_start=m_lo, m_end=m_hi,
                  ef_start=ef_lo, ef_end=ef_hi, shape=shape)
    else:
        cv = float("nan")
        idx = cls("l2", m=rb.M_VANILLA, ef=ef_vanilla)

    for p in data:
        idx.add(p)
    build = time.time() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    rec = float(np.mean([
        rb.calculate_recall(gt[i], [j for j, _ in idx.search(q, k)])
        for i, q in enumerate(queries)]))
    print(f"  {label:<28} build {build:8.2f}s  mem {peak/1e6:7.2f}MB  "
          f"recall {rec:6.2f}%  degree {idx.get_average_neighbors():6.3f}", flush=True)
    return {"Build_Time_s": build, "Memory_MB": peak / 1e6, "Recall_pct": rec,
            "CV": cv, "Avg_Degree": idx.get_average_neighbors()}


def main():
    p = argparse.ArgumentParser(description="C7 second axis: interpolation shape.")
    p.add_argument("--datasets-dir", default=os.path.expanduser("~/rag/hnsw-python"))
    p.add_argument("--datasets", nargs="+", default=["mnist"], choices=list(rb.LOADERS))
    p.add_argument("--num-vectors", type=int, default=60000)
    p.add_argument("--num-query-vectors", type=int, default=100)
    p.add_argument("--top-k", type=int, default=rb.TOP_K)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--interp", nargs="+", default=list(SHAPES), choices=list(SHAPES))
    p.add_argument("--phase", type=int, default=1, choices=[1, 2, 3])
    p.add_argument("--out", default=None)
    a = p.parse_args()

    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip() or "nogit"
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    host = os.uname().nodename
    rows = []

    for name in a.datasets:
        print(f"\n=== {name} ===", flush=True)
        train, q = rb.LOADERS[name](os.path.expanduser(a.datasets_dir),
                                    a.num_vectors, a.num_query_vectors)
        ef_v = rb.adjust_ef_by_dim(rb.EF_VANILLA_BASE, train.shape[1])
        dens = rb.calculate_density_random_projection(train, k=rb.M_VANILLA, seed=a.seed)
        from sklearn.neighbors import NearestNeighbors
        knn = NearestNeighbors(n_neighbors=a.top_k, algorithm="brute", metric="euclidean")
        knn.fit(train)
        gt = knn.kneighbors(q, n_neighbors=a.top_k, return_distance=False)

        meta = {"phase": a.phase, "dataset": name, "seed": a.seed, "git_sha": sha,
                "timestamp": stamp, "host": host, "N": len(train), "Dim": train.shape[1],
                "ef_ref": ef_v}
        rows.append({**meta, "variant": "vanilla", "interp": "n/a",
                     **measure(HNSW, train, q, gt, f"vanilla ({name})", a.top_k, ef_v)})
        for shape in a.interp:
            rows.append({**meta, "variant": "DHNSW", "interp": shape,
                         **measure(ShapedDynamicHNSW, train, q, gt,
                                   f"{shape} ({name})", a.top_k, ef_v,
                                   densities=dens, shape=shape)})

    df = pd.DataFrame(rows)
    out = a.out or f"results/phase{a.phase}/c7_interp_{'_'.join(a.datasets)}.csv"
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    df.to_csv(out, index=False)
    print("\n" + df.drop(columns=["git_sha", "timestamp", "host"]).to_string(index=False))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
