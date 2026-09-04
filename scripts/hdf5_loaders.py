#!/usr/bin/env python3
"""Loaders for the ann-benchmarks-style HDF5 datasets used by C3 and C10.

Each file holds `train`, `test`, and a `neighbors` ground truth computed
against the *full* train set. These experiments subsample train (pure-Python
DHNSW cannot build a 760K-point graph in reasonable time), which invalidates
the stored ground truth -- so the caller recomputes it over the subsample.

Normalisation: `landmark-dino-768` ships unnormalised (norms ~66) while
`landmark-nomic-768` and `simplewiki-openai-3072` ship unit-norm. All three
declare an angular metric, so this module L2-normalises whatever is not
already normalised. On the unit sphere ||a-b||^2 = 2 - 2*a.b, so Euclidean
ranking is cosine ranking: the experiments can then use the same "l2" distance
function the paper's MNIST/SIFT runs used, which removes a confound from the
C10 backbone comparison rather than introducing one.
"""
import os

import h5py
import numpy as np

DATA_DIR = os.path.expanduser("~/PRO-HNSW-Release/data")

HDF5_DATASETS = {
    "landmark-dino": "landmark-dino-768-cosine.hdf5",
    "landmark-nomic": "landmark-nomic-768-normalized.hdf5",
    "simplewiki-openai": "simplewiki-openai-3072-normalized.hdf5",
}


def _normalize(x):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


def load_hdf5(name, n_train, n_query, seed=42, data_dir=None):
    """Return (train, queries) as float64, unit-norm, subsampled to n_train.

    The subsample is drawn at random under `seed` rather than taken as the
    first n rows: these corpora are stored in ingestion order, and a prefix
    would over-represent whatever was ingested first.
    """
    path = os.path.join(data_dir or DATA_DIR, HDF5_DATASETS[name])
    with h5py.File(path, "r") as h:
        total = h["train"].shape[0]
        n_train = min(n_train, total)
        idx = np.sort(np.random.default_rng(seed).choice(total, n_train, replace=False))
        train = np.asarray(h["train"][idx], dtype=np.float64)
        queries = np.asarray(h["test"][:n_query], dtype=np.float64)
    return _normalize(train), _normalize(queries)
