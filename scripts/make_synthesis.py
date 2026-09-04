#!/usr/bin/env python3
"""Cross-experiment synthesis: DHNSW's saving as a function of density contrast.

C7 found that DHNSW's behaviour depends on the CV term only through one scalar
(delta). C8 found that scalar is insensitive to how the density is estimated.
C3 and C10 then measured it on data the paper never used, and the savings line
up on one curve against CV -- across four datasets, three dimensionalities and
four unrelated encoders.

This reads the phase CSVs the individual runners wrote rather than re-running
anything, so the figure cannot drift from the measurements.

    python scripts/make_synthesis.py
    # -> results/phase1/synthesis_cv_vs_saving.png
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_common import phase_path  # noqa: E402

# (label, csv stem, dataset, dhnsw variant, vanilla variant)
SOURCES = [
    ("MNIST\n784d", "c7_cv_transform_mnist", "mnist", "linear", "vanilla"),
    ("landmark-nomic\n768d", "c10_backbone", "landmark-nomic", "dhnsw", "vanilla"),
    ("landmark-dino\n768d", "c10_backbone", "landmark-dino", "dhnsw", "vanilla"),
    ("simplewiki-openai\n3072d", "c3_highdim", "simplewiki-openai", "dhnsw", "vanilla"),
]


def collect(phase=1):
    rows = []
    for label, stem, dataset, dyn_key, van_key in SOURCES:
        path = phase_path(phase, f"{stem}.csv")
        if not os.path.exists(path):
            print(f"  missing {path}, skipping {label}")
            continue
        df = pd.read_csv(path)
        df = df[df["dataset"] == dataset]
        dyn = df[df["variant"] == dyn_key]
        van = df[df["variant"] == van_key]
        if dyn.empty or van.empty:
            print(f"  no {dyn_key}/{van_key} pair for {dataset}, skipping")
            continue
        dyn, van = dyn.iloc[0], van.iloc[0]
        rows.append({
            "label": label.replace("\n", " "),
            "plot_label": label,
            "dim": int(dyn["Dim"]),
            "CV": float(dyn["CV"]),
            "degree_drop": (van["Avg_Degree"] - dyn["Avg_Degree"]) / van["Avg_Degree"] * 100,
            "memory_saved": (van["Memory_MB"] - dyn["Memory_MB"]) / van["Memory_MB"] * 100,
            "recall_delta": dyn["Recall_pct"] - van["Recall_pct"],
        })
    return pd.DataFrame(rows).sort_values("CV")


def main():
    df = collect()
    if len(df) < 2:
        print("Not enough datasets to synthesise.")
        return 1

    print(df[["label", "dim", "CV", "degree_drop", "memory_saved",
              "recall_delta"]].to_string(index=False))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6))

    for ax, col, title, ylab in (
            (ax1, "degree_drop", "Graph gets sparser in proportion to CV",
             "Average degree reduction (%)"),
            (ax2, "memory_saved", "And the memory saving follows",
             "Peak memory saved vs vanilla (%)")):
        ax.plot(df["CV"], df[col], "o-", color="#0072B2", markersize=9, zorder=3)
        for _, r in df.iterrows():
            ax.annotate(r["plot_label"], (r["CV"], r[col]),
                        textcoords="offset points", xytext=(0, -34),
                        ha="center", fontsize=8.5)
        ax.set_xlabel("CV of local density (data property, measurable up front)")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=0.3)
        ax.margins(x=0.20, y=0.30)

    fig.suptitle("DHNSW's benefit is a function of the corpus's density contrast\n"
                 "four datasets, three dimensionalities, four unrelated encoders",
                 fontsize=12)
    fig.tight_layout()
    out = phase_path(1, "synthesis_cv_vs_saving.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"\nSaved: {out}")

    csv_out = phase_path(1, "synthesis_cv_vs_saving.csv")
    df.drop(columns=["plot_label"]).to_csv(csv_out, index=False)
    print(f"Saved: {csv_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
