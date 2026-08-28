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


def _regenerate_and_compare(script_name, table_names):
    """Snapshot the committed tables, re-run the generator, and diff content —
    a hand-edited table must be flagged, not silently repaired."""
    import contextlib
    import importlib.util
    import io

    before = {}
    for name in table_names:
        path = metrics_dir() / name
        before[name] = path.read_text(encoding="utf-8") if path.exists() else None

    spec = importlib.util.spec_from_file_location(
        script_name, Path(__file__).resolve().parent / f"{script_name}.py")
    mt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mt)
    with contextlib.redirect_stdout(io.StringIO()):
        mt.main()

    bad = 0
    for name in table_names:
        path = metrics_dir() / name
        if before[name] is None:
            bad += 1
            print(f"MISSING table {name} (created only now by the generator)")
        elif before[name] != path.read_text(encoding="utf-8"):
            bad += 1
            print(f"MISMATCH table {name}: committed content differed from what "
                  f"the generator produces (now overwritten with the regenerated one)")
    return bad


def check_tables():
    """The committed markdown tables must equal what the generator produces now."""
    bad = _regenerate_and_compare(
        "make_tables", ("accuracy_table.md", "paired_heads_table.md",
                        "macro_accuracy_table.md", "handoff_table.md"))
    print(f"tables: regenerated from artifacts and compared, {bad} problems")
    return bad


def check_stage2_variant(suffix):
    """One Stage-2 results variant ("" = published normalized grid, "_raw" =
    the literal-spec raw-feature version): summary re-derived from raw arrays,
    deltas re-derived from Stage-1 runs.csv, run-0 prediction files reproduce
    their accuracies."""
    from src.evaluation import load_predictions

    runs1 = pd.read_csv(metrics_dir() / "runs.csv")
    runs2 = pd.read_csv(metrics_dir() / f"runs_stage2{suffix}.csv")
    summary2 = pd.read_csv(metrics_dir() / f"summary_stage2{suffix}.csv")
    bad = 0

    # (a) summary rows re-derived from raw arrays + runs table
    for _, row in summary2.iterrows():
        name = (f"{row['head']}{suffix}_{row['dataset']}_{row['encoder']}_{row['target']}"
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
        name = (f"run2{suffix}_{row['dataset']}_{row['encoder']}_{row['target']}_"
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

    label = "stage2" + (suffix or "")
    print(f"{label}: {len(summary2)} summary rows, {len(runs2)} run rows, "
          f"{checked} prediction files checked, {bad} problems")
    return bad


def check_stage2():
    """Stage-2 checks for every present variant plus the generated tables.
    Skipped until Stage 2 has run."""
    if not (metrics_dir() / "runs_stage2.csv").exists():
        print("stage2: no runs_stage2.csv yet — skipped")
        return 0
    bad = check_stage2_variant("")
    if (metrics_dir() / "runs_stage2_raw.csv").exists():
        bad += check_stage2_variant("_raw")

    # markdown tables regenerate AND match the committed content
    tables = ["stage2_image_prototype_table.md", "stage2_clip_text_table.md",
              "stage2_paired_delta_table.md"]
    if (metrics_dir() / "summary_stage2_raw.csv").exists():
        tables += ["stage2_raw_image_prototype_table.md", "stage2_raw_clip_text_table.md"]
    bad += _regenerate_and_compare("make_tables_stage2", tuple(tables))
    return bad


def check_stage3():
    """Stage-3 artifacts: summary re-derived from raw arrays and the per-run
    table; delta consistency against the STORED pinned-probe baseline (the
    mathematical baseline per ADR 0008 — runs.csv is a separate audit, checked
    at run time); run-0 prediction files; generated tables regenerate and match.
    Skipped until Stage 3 has run."""
    if not (metrics_dir() / "runs_stage3.csv").exists():
        print("stage3: no runs_stage3.csv yet — skipped")
        return 0
    from src.evaluation import load_predictions

    runs3 = pd.read_csv(metrics_dir() / "runs_stage3.csv")
    summary3 = pd.read_csv(metrics_dir() / "summary_stage3.csv")
    ok3 = runs3[runs3.get("status", "ok") == "ok"]
    bad = 0

    for _, row in summary3.iterrows():
        name = f"stage3_{row['head']}_{row['dataset']}_{row['encoder']}_{row['k_shot']}"
        raw = np.load(metrics_dir("raw") / f"{name}.npy")
        m, s = summarize(raw)
        sel = ok3[(ok3["dataset"] == row["dataset"]) & (ok3["encoder"] == row["encoder"])
                  & (ok3["head"] == row["head"]) & (ok3["k_shot"] == row["k_shot"])]
        dm, dstd = summarize(sel["delta_acc"].to_numpy())
        if len(raw) != row["n_runs"] or not np.isclose(m, row["mean_acc"], atol=1e-9) \
                or not np.isclose(s, row["std_acc"], atol=1e-9) \
                or not np.isclose(dm, row["delta_mean"], atol=1e-9) \
                or not np.isclose(dstd, row["delta_std"], atol=1e-9) \
                or not np.allclose(np.sort(sel["test_acc"].to_numpy()), np.sort(raw),
                                   atol=1e-9):
            bad += 1
            print(f"MISMATCH stage3 {name}: summary/raw/runs disagree")

    # delta = test - pinned-probe baseline, and the probe rows anchor baseline_acc
    for (ds, enc, seed), g in ok3.groupby(["dataset", "encoder", "seed"]):
        base = g[g["head"] == "pinned_probe"]
        if len(base) != 1:
            bad += 1
            print(f"MISSING stage3 pinned-probe row for {ds}/{enc}/seed{seed}")
            continue
        b = float(base["test_acc"].iloc[0])
        for _, r in g.iterrows():
            if not np.isclose(r["baseline_acc"], b, atol=1e-9) \
                    or not np.isclose(r["test_acc"] - b, r["delta_acc"], atol=1e-9):
                bad += 1
                print(f"MISMATCH stage3 delta {ds}/{enc}/{r['head']}/seed{seed}")

    checked = 0
    for _, row in ok3[ok3["run"] == 0].iterrows():
        name = f"run3_{row['dataset']}_{row['encoder']}_{row['head']}_{row['k_shot']}"
        try:
            pred, target = load_predictions(name)
        except FileNotFoundError:
            bad += 1
            print(f"MISSING stage3 predictions {name}")
            continue
        acc = float((np.asarray(pred) == np.asarray(target)).mean())
        checked += 1
        if not np.isclose(acc, row["test_acc"], atol=1e-6):
            bad += 1
            print(f"MISMATCH stage3 predictions {name}")

    # optional joint extension: same re-derivation pattern, if present
    if (metrics_dir() / "runs_stage3_joint.csv").exists():
        rj = pd.read_csv(metrics_dir() / "runs_stage3_joint.csv")
        sj = pd.read_csv(metrics_dir() / "summary_stage3_joint.csv")
        for _, row in sj.iterrows():
            name = f"stage3_{row['head']}_{row['dataset']}_{row['encoder']}_{row['k_shot']}"
            raw = np.load(metrics_dir("raw") / f"{name}.npy")
            m, s = summarize(raw)
            sel = rj[(rj["dataset"] == row["dataset"]) & (rj["encoder"] == row["encoder"])
                     & (rj["head"] == row["head"])]
            dm, dstd = summarize(sel["delta_acc"].to_numpy())
            if len(raw) != row["n_runs"] or not np.isclose(m, row["mean_acc"], atol=1e-9) \
                    or not np.isclose(s, row["std_acc"], atol=1e-9) \
                    or not np.isclose(dm, row["delta_mean"], atol=1e-9) \
                    or not np.isclose(dstd, row["delta_std"], atol=1e-9):
                bad += 1
                print(f"MISMATCH stage3-joint {name}")
            for _, r in sel.iterrows():
                if not np.isclose(r["test_acc"] - r["baseline_acc"], r["delta_acc"],
                                  atol=1e-9):
                    bad += 1
                    print(f"MISMATCH stage3-joint delta {name} seed{r['seed']}")

    tables3 = ["stage3_main_table.md", "stage3_lambda_ablation_table.md",
               "stage3_sweep_table.md"]
    if (metrics_dir() / "summary_stage3_joint.csv").exists():
        tables3.append("stage3_joint_table.md")
    bad += _regenerate_and_compare("make_tables_stage3", tuple(tables3))
    print(f"stage3: {len(summary3)} summary rows, {len(runs3)} run rows, "
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
    bad += check_stage3()
    if bad:
        sys.exit(f"repro check FAILED: {bad} problems")
    print("repro check PASSED: summary numbers, prediction files and generated "
          "tables all re-derived from raw artifacts")


if __name__ == "__main__":
    main()
