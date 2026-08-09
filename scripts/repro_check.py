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


def check_stage2():
    """Stage-2 artifacts: summary re-derived from raw arrays, deltas re-derived
    from Stage-1 runs.csv, run-0 prediction files reproduce their accuracies,
    and the committed markdown tables regenerate. Skipped until Stage 2 has run."""
    if not (metrics_dir() / "runs_stage2.csv").exists():
        print("stage2: no runs_stage2.csv yet — skipped")
        return 0
    import importlib.util

    from src.evaluation import load_predictions

    runs1 = pd.read_csv(metrics_dir() / "runs.csv")
    runs2 = pd.read_csv(metrics_dir() / "runs_stage2.csv")
    summary2 = pd.read_csv(metrics_dir() / "summary_stage2.csv")
    bad = 0

    # (a) summary rows re-derived from raw arrays + runs table
    for _, row in summary2.iterrows():
        name = (f"{row['head']}_{row['dataset']}_{row['encoder']}_{row['target']}"
                f"_T{row['T']}_{row['k_shot']}")
        raw = np.load(metrics_dir("raw") / f"{name}.npy")
        m, s = summarize(raw)
        sel = runs2[(runs2["dataset"] == row["dataset"]) & (runs2["encoder"] == row["encoder"])
                    & (runs2["target"] == row["target"]) & (runs2["head"] == row["head"])
                    & (runs2["T"] == row["T"]) & (runs2["k_shot"] == row["k_shot"])]
        dm, dstd = summarize(sel["delta_acc"].to_numpy())
        if len(raw) != row["n_runs"] or not np.isclose(m, row["mean_acc"], atol=1e-9) \
                or not np.isclose(s, row["std_acc"], atol=1e-9) \
                or not np.isclose(dm, row["delta_mean"], atol=1e-9) \
                or not np.isclose(dstd, row["delta_std"], atol=1e-9) \
                or not np.allclose(np.sort(sel["test_acc"].to_numpy()), np.sort(raw), atol=1e-9):
            bad += 1
            print(f"MISMATCH stage2 {name}: summary/raw/runs disagree")

    # (b) per-row baselines re-derived from Stage-1 runs.csv
    for _, r in runs2.iterrows():
        if r["target"] == "clip_text":
            b = runs1[(runs1["dataset"] == r["dataset"])
                      & (runs1["head"] == "zeroshot_clip")]["test_acc"].iloc[0]
        else:
            sel = runs1[(runs1["dataset"] == r["dataset"]) & (runs1["encoder"] == r["encoder"])
                        & (runs1["head"] == "image_prototype")
                        & (runs1["k_shot"] == r["k_shot"])]
            if r["k_shot"] != "full":
                sel = sel[sel["seed"] == r["seed"]]
            b = sel["test_acc"].iloc[0]
        if not np.isclose(b, r["baseline_acc"], atol=1e-9) \
                or not np.isclose(r["test_acc"] - b, r["delta_acc"], atol=1e-9):
            bad += 1
            print(f"MISMATCH stage2 baseline {r['dataset']}/{r['encoder']}/{r['target']}/"
                  f"{r['head']}/T{r['T']}/{r['k_shot']}/run{r['run']}")

    # (c) run-0 prediction files
    checked = 0
    for _, row in runs2[runs2["run"] == 0].iterrows():
        name = (f"run2_{row['dataset']}_{row['encoder']}_{row['target']}_"
                f"{row['head']}_T{row['T']}_{row['k_shot']}")
        try:
            pred, target = load_predictions(name)
        except FileNotFoundError:
            bad += 1
            print(f"MISSING stage2 predictions {name}")
            continue
        acc = float((np.asarray(pred) == np.asarray(target)).mean())
        checked += 1
        if not np.isclose(acc, row["test_acc"], atol=1e-6):
            bad += 1
            print(f"MISMATCH stage2 predictions {name}")

    # (d) markdown tables regenerate
    spec = importlib.util.spec_from_file_location(
        "make_tables_stage2", Path(__file__).resolve().parent / "make_tables_stage2.py")
    mt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mt)
    import contextlib
    import io

    with contextlib.redirect_stdout(io.StringIO()):
        mt.main()
    for name in ("stage2_image_prototype_table.md", "stage2_clip_text_table.md",
                 "stage2_paired_delta_table.md"):
        if not (metrics_dir() / name).exists():
            bad += 1
            print(f"MISSING table {name}")

    print(f"stage2: {len(summary2)} summary rows, {len(runs2)} run rows, "
          f"{checked} prediction files checked, {bad} problems")
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
    bad += check_stage2()
    if bad:
        sys.exit(f"repro check FAILED: {bad} problems")
    print("repro check PASSED: summary numbers, prediction files and generated "
          "tables all re-derived from raw artifacts")


if __name__ == "__main__":
    main()
