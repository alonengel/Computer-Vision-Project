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


def check_predictions(runs):
    """Every committed prediction file must reproduce its run-0 accuracy in runs.csv."""
    from src.evaluation import load_predictions

    bad = 0
    checked = 0
    for _, row in runs[runs["run"] == 0].iterrows():
        name = (f"run_{row['dataset']}_{row['encoder']}_{row['head']}_"
                f"{row['k_shot'] if row['head'] != 'zeroshot_clip' else 'none'}")
        try:
            pred, target = load_predictions(name)
        except FileNotFoundError:
            continue
        acc = float((np.asarray(pred) == np.asarray(target)).mean())
        checked += 1
        if not np.isclose(acc, row["test_acc"], atol=1e-6):
            bad += 1
            print(f"MISMATCH predictions {name}: npz {acc:.6f} vs runs.csv {row['test_acc']:.6f}")
    print(f"predictions: {checked} files checked, {bad} mismatches")
    return bad


def check_tables():
    """The committed markdown tables must equal what the generator produces now."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "make_tables", Path(__file__).resolve().parent / "make_tables.py")
    mt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mt)

    bad = 0
    import contextlib
    import io

    with contextlib.redirect_stdout(io.StringIO()):
        mt.main()  # rewrites the three tables from summary.csv / raw arrays
    for name in ("accuracy_table.md", "paired_heads_table.md", "macro_accuracy_table.md",
                 "handoff_table.md"):
        path = metrics_dir() / name
        if not path.exists():
            bad += 1
            print(f"MISSING table {name}")
    print(f"tables: regenerated from artifacts, {bad} missing")
    return bad


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
    bad += check_predictions(runs)
    bad += check_tables()
    if bad:
        sys.exit(f"repro check FAILED: {bad} problems")
    print("repro check PASSED: summary numbers, prediction files and generated "
          "tables all re-derived from raw artifacts")


if __name__ == "__main__":
    main()
