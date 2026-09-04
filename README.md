# DHNSW: Data-Adaptive Parameter Adjustment for HNSW

Source code and experimental artifacts for the DATE 2025 paper
*"Efficient Approximate Nearest Neighbor Search via Data-Adaptive Parameter
Adjustment in Hierarchical Navigable Small Graphs."*

Standard HNSW fixes `M` and `ef` globally, which over-connects dense regions and
under-connects sparse ones. DHNSW estimates each node's local density with
RP-KNN and scales its `M` and `ef` between density-derived bounds, cutting graph
build time and memory while holding recall.

## Setup

```bash
conda create -n dhnsw -c conda-forge python=3.11.7 -y
conda activate dhnsw
pip install -r requirements.txt
```

The algorithm is pure Python (`dhnsw.py`) — nothing to compile.

## Datasets

MNIST (784d), GloVe100K (300d), SIFT1M (128d) and GIST1M (960d), laid out as

```
<datasets-dir>/mnist/{train,t10k}-images.idx3-ubyte
<datasets-dir>/glove100k/glove.6B.300d.txt
<datasets-dir>/sift1m/sift_{base,query}.fvecs
<datasets-dir>/gist/gist_{base,query}.fvecs
```

Pass the directory with `--datasets-dir`; nothing is copied.

## Running

```bash
python scripts/run_baseline.py --datasets mnist --num-vectors 60000 \
    --datasets-dir /path/to/datasets --seed 42 --out baseline/dhnsw_mnist.csv
```

Reports **build time, peak memory and recall** against vanilla HNSW — the three
metrics of the paper. Throughput is deliberately not measured.

`--seed` fixes `random_state` for the RP density estimate. The original driver
leaves it unset, so per-node parameters differ between runs; set it when a
measurement needs to be repeatable.

## Layout

```
dhnsw.py             algorithm: HNSW and DynamicHNSW (unmodified)
main_original.py     the paper's original driver, kept verbatim for reference
scripts/             parameterized runners
baseline/            reference measurements, committed — see PLAN.md
results/             experiment output
```

`main_original.py` hardcodes `DATASET_DIR = "../dataset"` and a 1000-vector
default; it is kept as-is for provenance and is not the entry point.

## Review-comment experiments (thesis Chapter 3)

Each experiment writes data to `results/phase1/` and an interpretation to
`findings/`. Every runner takes `--smoke`, `--resume` and `--phase`.

| Comment | Question | Runner | Writeup |
|---|---|---|---|
| C2 | What does the density estimate cost? | `run_c2_phases.py` | `findings/C2_PHASE_OVERHEAD.md` |
| C3 | Does the gain hold at 1536d+? | `run_hdf5_experiment.py` | `findings/C3_HIGH_DIMENSIONAL.md` |
| C7 | Is the CV term's form justified? | `run_c7_ablation.py` | `findings/C7_CV_TRANSFORM.md` |
| C8 | Is RP-KNN worth its cost? | `run_c8_density_method.py` | `findings/C8_DENSITY_METHOD.md` |
| C10 | Does it survive a backbone change? | `run_hdf5_experiment.py` | `findings/C10_BACKBONE.md` |
| C11 | Does it survive parallel construction? | `run_c11_threads.py` | `findings/C11_THREAD_SCALING.md` |

`scripts/make_synthesis.py` draws the cross-experiment result from those CSVs:
DHNSW's saving is monotone in the corpus's density contrast (CV) across four
datasets, three dimensionalities and four encoders.

The C++ port used by C11 lives in `cpp/` — see `cpp/VENDORING.md` for what was
changed and `scripts/verify_cpp_port.py` for the checks that the untouched
path still reproduces stock hnswlib.
