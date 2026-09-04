#!/usr/bin/env python3
"""Sanity checks for the DHNSW C++ port before it is used for measurements.

Three things have to hold before any number the port produces is worth
reporting:

  1. The upstream path is untouched -- `add_items` must still build exactly
     the graph stock hnswlib builds. The additions are defaulted parameters,
     so this is the additive-only guarantee, checked rather than assumed.
  2. `add_items_dynamic` with uniform M and ef equal to the index's own must
     reproduce that same graph. If it does not, the parameter threading is
     wrong somewhere.
  3. Per-node M must actually change the graph in the expected direction:
     smaller M for some nodes means fewer stored edges.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hnswlib  # noqa: E402

DIM, N, M, EF_C, SEED = 32, 3000, 16, 100, 42


def build(data, mode, m_arr=None, ef_arr=None, m_index=M, m0=0):
    idx = hnswlib.Index(space="l2", dim=DIM)
    idx.init_index(max_elements=N, M=m_index, ef_construction=EF_C, random_seed=SEED)
    idx.set_num_threads(1)  # single thread: insertion order is then deterministic
    if mode == "static":
        idx.add_items(data, np.arange(N))
    else:
        idx.add_items_dynamic(data, m_arr, ef_arr, m0, np.arange(N))
    return idx


def recall_of(idx, data, queries, truth, k=10):
    idx.set_ef(100)
    labels, _ = idx.knn_query(queries, k=k)
    return float(np.mean([len(set(t[:k]) & set(l)) / k for t, l in zip(truth, labels)]))


def main():
    rng = np.random.default_rng(SEED)
    data = rng.random((N, DIM), dtype=np.float32)
    queries = rng.random((20, DIM), dtype=np.float32)
    truth = np.argsort(((data[None, :, :] - queries[:, None, :]) ** 2).sum(-1), axis=1)

    ok = True

    static = build(data, "static")
    e_static = static.get_total_edge_count()
    print(f"1. upstream add_items                edges={e_static}  "
          f"recall={recall_of(static, data, queries, truth):.4f}")

    # m0 = 2*M reproduces upstream's level-0 budget exactly.
    uniform = build(data, "dynamic",
                    np.full(N, M, dtype=np.uint64), np.full(N, EF_C, dtype=np.uint64),
                    m0=2 * M)
    e_uniform = uniform.get_total_edge_count()
    same = e_uniform == e_static
    ok &= same
    print(f"2. add_items_dynamic at uniform M/ef edges={e_uniform}  "
          f"recall={recall_of(uniform, data, queries, truth):.4f}  "
          f"{'MATCHES upstream' if same else 'DIFFERS from upstream -- BUG'}")

    # Per-node M drawn from [8, 24] around the index maximum of 24, the shape
    # DHNSW produces from a density estimate.
    m_arr = rng.integers(8, 25, size=N).astype(np.uint64)
    ef_arr = rng.integers(60, 141, size=N).astype(np.uint64)
    dyn = build(data, "dynamic", m_arr, ef_arr, m_index=24, m0=2 * 8)
    e_dyn = dyn.get_total_edge_count()

    static24 = build(data, "static", m_index=24)
    e_static24 = static24.get_total_edge_count()
    fewer = e_dyn < e_static24
    ok &= fewer
    print(f"3. per-node M in [8,24]              edges={e_dyn}  "
          f"vs uniform M=24 edges={e_static24}  "
          f"({(e_static24 - e_dyn) / e_static24 * 100:+.1f}%)  "
          f"recall={recall_of(dyn, data, queries, truth):.4f}  "
          f"{'fewer edges as expected' if fewer else 'NOT fewer -- BUG'}")

    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
