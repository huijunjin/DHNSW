# Vendored dependency: hnswlib

`cpp/hnswlib/` and `cpp/python_bindings/` are a vendored copy of upstream
hnswlib with DHNSW's changes applied — not a submodule, and not something the
build fetches at install time.

| | |
|---|---|
| Upstream | https://github.com/nmslib/hnswlib |
| Commit | `c1b9b79af3d10c6ee7b5d0afa1ce851ae975254c` (2024-06-17) |
| Files modified | `hnswlib/hnswalg.h`, `python_bindings/bindings.cpp` |
| Everything else | upstream, unchanged |

The commit is the same one PRO-HNSW pins, so the two chapters' C++ results rest
on the same baseline implementation.

## Why vendored

`hnswalg.h` includes `visited_list_pool.h`, `hnswlib.h` and the space headers
from whatever copy it is built against. A clone-and-overwrite overlay leaves
the build pinned to upstream's default branch on the day it runs: an upstream
change breaks the build, or worse changes behaviour silently. Vendoring at a
recorded commit keeps these results reproducible for as long as this
repository exists.

## What DHNSW changes

The paper's contribution is per-node M and ef derived from a local density
estimate. Upstream applies one M and one `ef_construction_` to every
insertion, so three call paths need to carry per-node values. **The RP-KNN
density estimate itself is deliberately not ported** — it stays in sklearn on
the Python side and arrives as two arrays, one entry per point.

| Location | Change |
|---|---|
| `hnswalg.h` `searchBaseLayer` | `ef_c` parameter; zero means `ef_construction_` |
| `hnswalg.h` `mutuallyConnectNewElement` | `M_node` / `M0_node` parameters for the degree budget |
| `hnswalg.h` `addPoint` | threads `M_node`, `ef_node`, `M0_node` through |
| `hnswalg.h` `getTotalEdgeCount` | new: counts stored edges (the memory metric) |
| `bindings.cpp` `add_items_dynamic` | new: `(data, m_per_node, ef_per_node, m0, ids, num_threads)` |
| `bindings.cpp` `get_total_edge_count` | new accessor |

Every added parameter defaults to zero, and zero means upstream behaviour. The
untouched path is therefore still stock hnswlib, which
`scripts/verify_cpp_port.py` checks rather than assumes.

### Two details that are easy to get wrong

**Level 0 uses a constant budget, not a per-node one.** The paper's Python
fixes the level-0 degree cap at `2 * m_start` for every node — `HNSW.__init__`
derives `self._m0` from `m_start`, and `DynamicHNSW.add` only ever varies
`self._m`. Level 0 holds every node, so this constant is where most of the
reported memory saving comes from. Scaling it per node instead (`2 * M_node`)
reproduces almost none of the effect.

**Upstream is asymmetric at level 0.** A new node selects `M_` neighbours, but
an existing neighbour's list may grow to `maxM0_ = 2 * M_`. The two caps are
kept separate in the port, so passing `M_node = M_` and `M0_node = 2 * M_`
reproduces upstream exactly. Collapsing them into one budget does not.

**The neighbour-list assertion had to be relaxed.** Upstream throws if a
neighbour's list exceeds `Mcurmax`, which is sound only when every insertion
shares one budget. With per-node M a neighbour's list may have been filled by
a higher-M insertion and legitimately exceed a later, smaller one; it is then
pruned down, which is what the Python `_select` does. The check now tests the
allocation bound (`maxM_` / `maxM0_`), which is the invariant that must
actually hold.

## Sizing the index

Link lists are allocated for `maxM_` / `maxM0_` slots, so the index must be
constructed with `M` at least the largest per-node M (DHNSW uses `m_end`).
Per-node values are clamped to those maxima and can never overflow.

This is also why memory is reported as **edge count** rather than allocated
bytes: hnswlib allocates `maxM` slots per node whether or not they are filled,
so allocated bytes cannot show a saving the paper attributes to reduced
average degree (Fig. 5). `get_total_edge_count()` measures the quantity that
actually changes.

## Building

```bash
cd cpp && pip install --no-build-isolation --no-deps .
python scripts/verify_cpp_port.py   # must print PASS
```

`--no-deps` matters: without it pip will happily upgrade NumPy out of the
pinned environment (`numpy==1.26.4`, `scikit-learn==1.4.1.post1`), which
changes the RP-KNN density estimate and therefore every downstream number.
