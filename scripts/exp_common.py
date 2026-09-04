#!/usr/bin/env python3
"""Shared plumbing for the Chapter 3 experiment runners.

Implements the result-file convention in PLAN.md: results are written under
`results/phase{1,2,3}/` so that widening the range dial never overwrites an
earlier phase, and every result row carries the metadata needed to tell two
runs apart after the fact (which code, which machine, when, which seed).
"""
import os
import socket
import subprocess
from datetime import datetime

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Metadata every result CSV must carry (PLAN.md, "결과 파일 규약").
META_COLUMNS = ["phase", "dataset", "variant", "seed", "git_sha", "timestamp", "host"]


def git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL, text=True).strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def run_metadata(phase, dataset, variant, seed):
    """The columns PLAN.md requires on every result row."""
    return {"phase": phase, "dataset": dataset, "variant": variant,
            "seed": "NA" if seed is None else seed, "git_sha": git_sha(),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "host": socket.gethostname()}


def phase_path(phase, filename):
    """results/phase<N>/<filename>, as an absolute path."""
    return os.path.join(REPO_ROOT, "results", f"phase{phase}", filename)


def add_common_args(p, default_seed=42):
    """The arguments PLAN.md requires on every runner."""
    p.add_argument("--phase", type=int, default=1, choices=(1, 2, 3),
                   help="Range dial. Decides the results/phase<N>/ output directory.")
    p.add_argument("--seed", type=int, default=default_seed,
                   help="random_state for the density estimate.")
    p.add_argument("--smoke", action="store_true",
                   help="Small run (~60s) to sanity-check the pipeline end to end.")
    p.add_argument("--resume", action="store_true",
                   help="Skip variants already present in the output CSV.")
    p.add_argument("--out", default=None,
                   help="CSV path. Defaults to the phase directory.")
    p.add_argument("--plot", default=None,
                   help="Figure path. Defaults next to the CSV. Empty string skips it.")
    return p


def resolve_outputs(args, stem):
    """Fill in --out / --plot from --phase when they were not given."""
    out = args.out or phase_path(args.phase, f"{stem}.csv")
    if args.plot is None:
        plot = os.path.splitext(out)[0] + ".png"
    else:
        plot = args.plot
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    return out, plot


def save_csv(df, out):
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    df.to_csv(out, index=False)
    return df


def order_columns(df):
    """Metadata first, then the measurements -- so the CSVs read consistently."""
    lead = [c for c in META_COLUMNS if c in df.columns]
    return df[lead + [c for c in df.columns if c not in lead]]
