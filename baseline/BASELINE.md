# Golden master — DHNSW

Reference measurements taken from the unmodified algorithm (`dhnsw.py`) before
any further change, so that later work has something to be checked against.
Re-run the command below after touching anything and diff the numbers: recall
should be identical, timings within a couple of percent. A larger gap means the
change altered behaviour.

```bash
python scripts/run_baseline.py --datasets mnist --num-vectors 60000 \
    --num-query-vectors 100 --seed 42 --out baseline/dhnsw_mnist.csv
```

Environment: Hallasan (i7-12700F, 94 GB), conda env `dhnsw`
(Python 3.11.7 · numpy 1.26.4 · scikit-learn 1.4.1.post1), 2026-09-03.

## MNIST, 60,000 base vectors, 100 queries

| | Vanilla HNSW | DHNSW | Change |
|---|---|---|---|
| Build time | 610.73 s | 549.29 s | **−10.06 %** |
| Peak memory (tracemalloc) | 153.21 MB | 97.36 MB | **−36.45 %** |
| Recall@100 | 99.29 % | 98.17 % | −1.12 %p |
| Average degree | 31.00 | 17.79 | −42.6 % |

Density statistics for this run: CV = 0.2634, giving M ∈ [9, 22] and
ef ∈ [90, 207]. ef_ref = 149 (Eq. 1 at d = 784); vanilla runs at ef = 100.

## Against the published numbers

| | Paper (Fig. 2/3) | This run |
|---|---|---|
| Vanilla recall | 99.33 % | **99.29 %** |
| DHNSW recall | 98.64 % | 98.17 % |
| Build time improvement | 13.29 % | 10.06 % |
| Memory improvement | 32.44 % | 36.45 % |

Vanilla recall lands within 0.01 %p of the published value, which is the useful
signal here: the data loading, the distance function and the unmodified
algorithm all behave as they did for the paper, so the environment is sound.

The two improvement figures differ, and the reason is in the code rather than
the environment. `calculate_density_random_projection` builds a
`GaussianRandomProjection` with no `random_state`, so every run draws a
different projection, gets different local densities, and therefore a different
CV — which is what sets the M and ef ranges each node is scaled between. The
published numbers come from an unseeded run; this one is pinned to seed 42 so it
can be reproduced. Expect a spread across seeds, not a single number.

**On the timing figures.** These come from a re-run on an idle machine
(2026-09-04). An earlier run had a 7 GB download in flight and reported 8.33 %;
memory, recall and degree were identical to within 0.03 %, so only the build
times had been affected. Quote the idle-machine numbers.

## What to compare against later

Average degree is the one to watch through the C++ port: the paper attributes
the memory saving to reduced average degree (Fig. 5), and 31.00 → 17.78 is that
effect. hnswlib allocates a fixed `maxM` slots per node, so a port will not
reproduce the memory figure directly — report the edge count instead.
