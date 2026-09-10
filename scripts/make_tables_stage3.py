"""Stage-3 tables (spec: report top-1 for the probe and both methods + the
change relative to the probe baseline).

Generated programmatically from results/metrics/runs_stage3.csv /
summary_stage3.csv / stage3_sweep.csv (never typed by hand). Layout per
ADR 0008 §10: one clean main table (accuracies in %, Δ in percentage points,
paired per seed); the λ-ablation and the sweep in separate tables.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.evaluation import metrics_dir
from src.utils import load_config
from src.visualize import dataset_label, encoder_label

HEAD_LABELS = {"pinned_probe": "Pinned Stage-1 probe (baseline)",
               "fm_s1": "Rolled strategy — end-to-end rolled-out CE training",
               "fm_s1_lambda0": "Rolled strategy, λ = 0 (no displacement penalty)",
               "fm_s2": "Guided strategy — classifier-guided targets"}
# Display names of the two mandatory methods: the Rolled strategy is the
# specification's Strategy 1, the Guided strategy its Strategy 2.
STRATEGY_NAMES = {"s1": "Rolled", "s2": "Guided"}


def head_params(runs, ds, enc, head):
    r = runs[(runs["dataset"] == ds) & (runs["encoder"] == enc)
             & (runs["head"] == head)]
    if r.empty:
        return ""
    r = r.iloc[0]
    if head.startswith("fm_s1") and pd.notna(r.get("lam")):
        return f" (λ = {r['lam']:g})"
    if head == "fm_s2" and pd.notna(r.get("beta")):
        return f" (β = {r['beta']:g}, m = {int(r['m'])})"
    return ""


def fmt_pm(mean, std, n):
    txt = f"{100 * mean:.2f}"
    if n > 1:
        txt += f" ± {100 * std:.2f}"
    return txt


def main_table():
    cfg = load_config()
    runs = pd.read_csv(metrics_dir() / "runs_stage3.csv")
    s = pd.read_csv(metrics_dir() / "summary_stage3.csv")
    k = f"{cfg['stage3']['k_shot']}shot"

    lines = ["| Dataset / encoder | Head | Top-1 (%) | Δ vs pinned probe (pts, paired) |",
             "|---|---|---|---|"]
    for ds, enc in cfg["stage3"]["settings"]:
        for head in ("pinned_probe", "fm_s1", "fm_s2"):
            row = s[(s["dataset"] == ds) & (s["encoder"] == enc) & (s["head"] == head)]
            if row.empty:
                lines.append(f"| {dataset_label(ds)} / {encoder_label(enc, short=True)} "
                             f"| {HEAD_LABELS[head]} | failed | — |")
                continue
            r = row.iloc[0]
            acc = fmt_pm(r["mean_acc"], r["std_acc"], r["n_runs"])
            if head == "pinned_probe":
                delta = "—"
            else:
                delta = f"{100 * r['delta_mean']:+.2f}"
                if r["n_runs"] > 1:
                    delta += f" ± {100 * r['delta_std']:.2f}"
                d = 100 * runs[(runs["dataset"] == ds) & (runs["encoder"] == enc)
                               & (runs["head"] == head)].sort_values("seed")["delta_acc"]
                delta += f" ({int((d > 0).sum())}/{len(d)} seeds > 0)"
            lines.append(f"| {dataset_label(ds)} / {encoder_label(enc, short=True)} "
                         f"| {HEAD_LABELS[head]}{head_params(runs, ds, enc, head)} "
                         f"| {acc} | {delta} |")
    lines.append("")
    lines.append(
        f"Top-1 accuracy (%) on the complete official test split, K = "
        f"{cfg['stage3']['k_shot']}, T = {cfg['stage3']['T']}; mean ± sample std over "
        f"the 3 subset seeds. Δ is **paired per seed** against the exact pinned probe "
        f"of that seed — the identical frozen classifier inside the pipeline (the "
        f"pipeline at initialization equals it exactly, by the zero-velocity init). "
        f"Hyperparameters (λ; β, m) were selected per dataset on **seed-0 validation** "
        f"only; seed 0 is therefore partly a development run, and the 3-seed mean is a "
        f"summary, not an independent confirmatory estimate. n = 3 — no significance "
        f"claims.")
    out = metrics_dir() / "stage3_main_table.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return "\n".join(lines), out


def ablation_table():
    """Rolled-strategy winner vs the pre-registered λ = 0 pair (with/without regularization)."""
    cfg = load_config()
    runs = pd.read_csv(metrics_dir() / "runs_stage3.csv")
    s = pd.read_csv(metrics_dir() / "summary_stage3.csv")
    lines = ["| Dataset / encoder | Variant | Top-1 (%) | Δ vs pinned probe (pts) |",
             "|---|---|---|---|"]
    degenerate = []
    for ds, enc in cfg["stage3"]["settings"]:
        heads = [("fm_s1", "validation-selected λ"), ("fm_s1_lambda0", "λ = 0")]
        present = [h for h, _ in heads
                   if not s[(s["dataset"] == ds) & (s["encoder"] == enc)
                            & (s["head"] == h)].empty]
        if "fm_s1_lambda0" not in present:
            degenerate.append(f"{dataset_label(ds)}")
        for head, label in heads:
            row = s[(s["dataset"] == ds) & (s["encoder"] == enc) & (s["head"] == head)]
            if row.empty:
                continue
            r = row.iloc[0]
            delta = f"{100 * r['delta_mean']:+.2f}"
            if r["n_runs"] > 1:
                delta += f" ± {100 * r['delta_std']:.2f}"
            suffix = head_params(runs, ds, enc, head) if head == "fm_s1" else ""
            lines.append(f"| {dataset_label(ds)} / {encoder_label(enc, short=True)} "
                         f"| {label}{suffix} "
                         f"| {fmt_pm(r['mean_acc'], r['std_acc'], r['n_runs'])} "
                         f"| {delta} |")
    lines.append("")
    note = ("Pre-registered pair: the validation winner AND λ = 0 both receive a test "
            "read-out, so the with/without-regularization comparison involves no "
            "post-hoc choice.")
    if degenerate:
        note += (f" On {', '.join(degenerate)} the winner IS λ = 0, so the pair "
                 f"degenerates to a single run and the contrast is absent by "
                 f"construction (stated in ADR 0008 §7).")
    lines.append(note)
    out = metrics_dir() / "stage3_lambda_ablation_table.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return "\n".join(lines), out


def sweep_table():
    """Validation sweep (seed 0) — full transparency of the selection."""
    sw = pd.read_csv(metrics_dir() / "stage3_sweep.csv")
    lines = ["| Dataset | Strategy | Configuration | Val top-1 (%) | Checkpoint epoch | Fallback |",
             "|---|---|---|---|---|---|"]
    for _, r in sw.iterrows():
        if r.get("status") == "failed":
            lines.append(f"| {dataset_label(r['dataset'])} | {STRATEGY_NAMES.get(r['strategy'], r['strategy'].upper())} "
                         f"| {r['param']} | failed | — | — |")
            continue
        lines.append(f"| {dataset_label(r['dataset'])} | {STRATEGY_NAMES.get(r['strategy'], r['strategy'].upper())} "
                     f"| {r['param']} | {100 * r['val_acc']:.2f} "
                     f"| {int(r['checkpoint_epoch'])} | {int(r['fallback'])} |")
    lines.append("")
    lines.append("Seed-0 **validation** accuracy of the full pipeline per swept "
                 "configuration (test untouched during selection). Winners per dataset "
                 "by highest validation accuracy (ties → lowest validation CE → grid "
                 "order). Fallback level 0 = the default recipe (ADR 0008 §7).")
    out = metrics_dir() / "stage3_sweep_table.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return "\n".join(lines), out


def joint_table():
    """Optional extension: joint FM+classifier fine-tuning vs the classifier-only
    continued-training control (attribution), vs the frozen-classifier results."""
    cfg = load_config()
    s = pd.read_csv(metrics_dir() / "summary_stage3.csv")
    sj = pd.read_csv(metrics_dir() / "summary_stage3_joint.csv")
    labels = {"fm_joint": "Joint FM + classifier fine-tuning",
              "clf_only_continued": "Control: classifier-only continued training",
              "fm_s2": "(reference) Guided strategy, frozen classifier"}
    lines = ["| Dataset / encoder | Variant | Top-1 (%) | Δ vs pinned probe (pts) |",
             "|---|---|---|---|"]
    for ds, enc in cfg["stage3"]["settings"]:
        for head in ("fm_joint", "clf_only_continued", "fm_s2"):
            src = sj if head in ("fm_joint", "clf_only_continued") else s
            row = src[(src["dataset"] == ds) & (src["encoder"] == enc)
                      & (src["head"] == head)]
            if row.empty:
                continue
            r = row.iloc[0]
            delta = f"{100 * r['delta_mean']:+.2f}"
            if r["n_runs"] > 1:
                delta += f" ± {100 * r['delta_std']:.2f}"
            lines.append(f"| {dataset_label(ds)} / {encoder_label(enc, short=True)} "
                         f"| {labels[head]} "
                         f"| {fmt_pm(r['mean_acc'], r['std_acc'], r['n_runs'])} "
                         f"| {delta} |")
    lines.append("")
    lines.append(
        "Optional extension (ADR 0008 §9), run after the mandatory comparison: the "
        "classifier is unfrozen (lr 1e-4; FM lr 1e-3; same policy, 3 seeds). The "
        "**control** trains the pinned classifier further *alone* with the same "
        "budget — without it, a joint gain could not be attributed to the FM rather "
        "than to the classifier simply training longer. Same pinned-probe baselines "
        "as the main table.")
    out = metrics_dir() / "stage3_joint_table.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return "\n".join(lines), out


def main():
    fns = [main_table, ablation_table, sweep_table]
    if (metrics_dir() / "summary_stage3_joint.csv").exists():
        fns.append(joint_table)
    for fn in fns:
        tbl, path = fn()
        print(tbl)
        print(f"\nwritten: {path}\n")


if __name__ == "__main__":
    main()
