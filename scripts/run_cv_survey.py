#!/usr/bin/env python3
"""CV survey -- measure the local-density spread without building any graph.

C7 showed DHNSW's behaviour is governed by delta = CV * lambda, and C3 showed
CV collapses in high dimension. Turning that into a claim needs more than the
four points measured so far, and CV is cheap: it needs only the density
estimate, which C2 put at 0.9% of a build. This adds points at survey cost.

    python scripts/run_cv_survey.py --datasets mnist glove100k sift1m gist
    # -> results/phase2/cv_survey.csv
"""
import argparse, os, subprocess, sys, time
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_baseline as rb


def main():
    p = argparse.ArgumentParser(description="Measure CV of local density per dataset.")
    p.add_argument("--datasets-dir", default=os.path.expanduser("~/rag/hnsw-python"))
    p.add_argument("--datasets", nargs="+", default=list(rb.LOADERS), choices=list(rb.LOADERS))
    p.add_argument("--num-vectors", type=int, default=60000,
                   help="Capped for survey cost; CV is a dispersion ratio and is "
                        "stable well below full corpus size.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lam", type=float, default=rb.SCALE_FACTOR)
    p.add_argument("--phase", type=int, default=2, choices=[1, 2, 3])
    p.add_argument("--out", default=None)
    a = p.parse_args()

    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip() or "nogit"
    stamp, host = time.strftime("%Y-%m-%dT%H:%M:%S"), os.uname().nodename
    rows = []

    for name in a.datasets:
        t0 = time.time()
        train, _ = rb.LOADERS[name](os.path.expanduser(a.datasets_dir), a.num_vectors, 10)
        dens = rb.calculate_density_random_projection(train, k=rb.M_VANILLA, seed=a.seed)
        cv = float(np.std(dens) / np.mean(dens))
        ef_ref = rb.adjust_ef_by_dim(rb.EF_VANILLA_BASE, train.shape[1])
        delta = cv * a.lam
        m_lo = max(2, int(rb.M_VANILLA - rb.M_VANILLA * delta))
        m_hi = int(rb.M_VANILLA + rb.M_VANILLA * delta)
        rows.append({"phase": a.phase, "dataset": name, "variant": "cv_survey",
                     "seed": a.seed, "git_sha": sha, "timestamp": stamp, "host": host,
                     "N": len(train), "Dim": int(train.shape[1]), "CV": cv,
                     "lambda": a.lam, "delta": delta, "M_low": m_lo, "M_high": m_hi,
                     "M_span": m_hi - m_lo, "ef_ref": ef_ref,
                     "density_time_s": time.time() - t0})
        print(f"  {name:<12} d={train.shape[1]:>5}  N={len(train):>7,}  "
              f"CV={cv:.4f}  delta={delta:.3f}  M=[{m_lo},{m_hi}]  "
              f"ef_ref={ef_ref}  ({time.time()-t0:.1f}s)", flush=True)

    df = pd.DataFrame(rows)
    out = a.out or f"results/phase{a.phase}/cv_survey.csv"
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    df.to_csv(out, index=False)
    print("\n" + df.drop(columns=["git_sha", "timestamp", "host"]).to_string(index=False))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
