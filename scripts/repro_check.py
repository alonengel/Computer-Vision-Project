"""Reproducibility check: re-derive every headline number from saved raw artifacts.

Loads the per-episode / per-seed raw accuracy arrays (results/metrics/raw/) and
verifies that the mean/CI/std values in the committed metrics tables match them
exactly. Independent of feature extraction and experiment code paths.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.evaluation import ci95, metrics_dir


def check_table(table_name, raw_prefix, err_col, err_fn):
    df = pd.read_csv(metrics_dir() / f"{table_name}.csv")
    bad = 0
    for _, row in df.iterrows():
        if table_name == "episodic":
            raw_name = (f"{raw_prefix}_{row['dataset']}_{row['n_way']}w{row['k_shot']}s_"
                        f"{row['classifier']}")
        else:
            raw_name = f"{raw_prefix}_{row['dataset']}_{row['k_shot']}s_{row['classifier']}"
        raw = np.load(metrics_dir("raw") / f"{raw_name}.npy")
        ok_acc = np.isclose(raw.mean(), row["acc"], atol=1e-6)
        ok_err = np.isclose(err_fn(raw), row[err_col], atol=1e-6)
        if not (ok_acc and ok_err):
            bad += 1
            print(f"MISMATCH {raw_name}: table acc={row['acc']:.6f} err={row[err_col]:.6f} "
                  f"vs raw {raw.mean():.6f} / {err_fn(raw):.6f}")
    print(f"{table_name}: {len(df)} rows checked, {bad} mismatches")
    return bad


def main():
    bad = check_table("episodic", "ep", "ci95", ci95)
    bad += check_table("simple", "simple", "std", lambda a: np.std(a, ddof=1))
    if bad:
        sys.exit(f"repro check FAILED: {bad} mismatching rows")
    print("repro check PASSED: all table numbers re-derived from raw artifacts")


if __name__ == "__main__":
    main()
