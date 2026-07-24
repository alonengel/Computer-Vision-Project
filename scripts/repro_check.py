"""Reproducibility check: re-derive every summary number from the saved raw arrays.

Loads results/metrics/raw/*.npy (per-run top-1 accuracies) and verifies that the
mean and standard deviation reported in summary.csv match, and that the per-run
table in runs.csv contains exactly those accuracies. Independent of the training
and feature-extraction code paths.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.evaluation import metrics_dir, summarize


def main():
    summary = pd.read_csv(metrics_dir() / "summary.csv")
    runs = pd.read_csv(metrics_dir() / "runs.csv")
    bad = 0
    for _, row in summary.iterrows():
        name = f"{row['head']}_{row['dataset']}_{row['encoder']}_{row['k_shot']}"
        raw = np.load(metrics_dir("raw") / f"{name}.npy")
        m, s = summarize(raw)
        if len(raw) != row["n_runs"] or not np.isclose(m, row["mean_acc"], atol=1e-9) \
                or not np.isclose(s, row["std_acc"], atol=1e-9):
            bad += 1
            print(f"MISMATCH {name}: table {row['n_runs']} runs "
                  f"{row['mean_acc']:.6f}±{row['std_acc']:.6f} vs raw {len(raw)} runs "
                  f"{m:.6f}±{s:.6f}")
            continue
        sel = runs[(runs["dataset"] == row["dataset"]) & (runs["encoder"] == row["encoder"])
                   & (runs["head"] == row["head"]) & (runs["k_shot"] == row["k_shot"])]
        if not np.allclose(np.sort(sel["test_acc"].to_numpy()), np.sort(raw), atol=1e-9):
            bad += 1
            print(f"MISMATCH {name}: runs.csv accuracies differ from raw array")
    print(f"summary: {len(summary)} rows checked, {bad} mismatches")
    if bad:
        sys.exit(f"repro check FAILED: {bad} mismatching rows")
    print("repro check PASSED: all summary numbers re-derived from raw artifacts")


if __name__ == "__main__":
    main()
