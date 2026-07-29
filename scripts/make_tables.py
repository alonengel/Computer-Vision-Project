"""Spec item 1 — the accuracy table: top-1 accuracy for every dataset, encoder,
training-set size and implemented baseline.

Generated programmatically from results/metrics/summary.csv (never typed by hand)
and written to results/metrics/accuracy_table.md for inclusion in the report and
notebook.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.evaluation import metrics_dir
from src.utils import load_config
from src.visualize import dataset_label, encoder_label, head_label

K_COLS = ["5shot", "10shot", "full"]
K_HEAD = {"5shot": "K = 5", "10shot": "K = 10", "full": "full train split"}


def cell(row):
    if row is None or row.empty:
        return "—"
    r = row.iloc[0]
    if r["n_runs"] > 1:
        return f"{100 * r['mean_acc']:.2f} ± {100 * r['std_acc']:.2f}"
    return f"{100 * r['mean_acc']:.2f}"


def paired_table():
    """Paired per-seed head comparison (image prototypes − linear probe).

    Legitimate pairing: at a given (dataset, encoder, K) both heads are trained on
    the *same* committed subset indices and evaluated on the same test split, so
    the per-seed differences are matched and far tighter than the marginal spreads.
    """
    import numpy as np

    cfg = load_config()
    lines = ["| Dataset | Encoder | K | Prototypes − probe (paired) | Seeds favouring prototypes |",
             "|---|---|---|---|---|"]
    for ds in cfg["datasets"]:
        spec = "" if cfg["datasets"][ds]["spec_selected"] else " ‡"
        for enc in ("resnet18", "dinov2_vits14"):
            for k in ("5shot", "10shot"):
                try:
                    a = np.load(metrics_dir("raw") / f"image_prototype_{ds}_{enc}_{k}.npy")
                    b = np.load(metrics_dir("raw") / f"linear_probe_{ds}_{enc}_{k}.npy")
                except FileNotFoundError:
                    continue
                d = 100 * (a - b)
                lines.append(f"| {dataset_label(ds)}{spec} | {encoder_label(enc, short=True)} "
                             f"| {k.replace('shot', '')} | {d.mean():+.2f} ± {d.std(ddof=1):.2f} "
                             f"| {int((d > 0).sum())} / {len(d)} |")
    lines.append("")
    lines.append("Mean ± sample standard deviation (ddof = 1) of the per-seed difference in "
                 "top-1 accuracy, both heads trained on identical subset indices. Positive "
                 "favours image prototypes. The `full` setting is omitted because the "
                 "prototype head runs once there and the probe varies only by initialization, "
                 "so the runs are not paired.")
    out = metrics_dir() / "paired_heads_table.md"
    table = "\n".join(lines)
    out.write_text(table, encoding="utf-8")
    return table, out


def macro_table():
    """Balanced (macro-averaged) accuracy alongside top-1, for the plotted
    full-split probe run. Relevant because Flowers-102's official test split is
    class-imbalanced (20-238 images per class) while top-1 is image-weighted."""
    import numpy as np

    from src.evaluation import load_predictions

    cfg = load_config()
    runs = pd.read_csv(metrics_dir() / "runs.csv")
    lines = ["| Dataset | Encoder | Top-1 (%) | Balanced / macro (%) |", "|---|---|---|---|"]
    for ds in cfg["datasets"]:
        g = runs[(runs["dataset"] == ds) & (runs["head"] == "linear_probe")
                 & (runs["k_shot"] == "full")]
        enc = g.groupby("encoder")["val_acc"].mean().idxmax()
        pred, target = load_predictions(f"run_{ds}_{enc}_linear_probe_full")
        pred, target = np.asarray(pred), np.asarray(target)
        macro = float(np.mean([(pred[target == c] == c).mean() for c in np.unique(target)]))
        spec = "" if cfg["datasets"][ds]["spec_selected"] else " ‡"
        lines.append(f"| {dataset_label(ds)}{spec} | {encoder_label(enc, short=True)} "
                     f"| {100 * (pred == target).mean():.2f} | {100 * macro:.2f} |")
    lines.append("")
    lines.append("Single full-split linear-probe run (initialization seed 0) on each dataset's "
                 "best-by-validation encoder. The two columns coincide when the test split is "
                 "balanced (DTD, FGVC-Aircraft) and diverge for Flowers-102, whose official "
                 "test split is class-imbalanced.")
    out = metrics_dir() / "macro_accuracy_table.md"
    table = "\n".join(lines)
    out.write_text(table, encoding="utf-8")
    return table, out


def handoff_table():
    """Stage-2/3 handoff: selected encoder, prototype target and linear-probe
    baseline per dataset. Encoder selection uses **validation** accuracy only;
    the test column is the already-published one-time read-out that Stage 2/3
    must beat, not a selection criterion.
    """
    cfg = load_config()
    runs = pd.read_csv(metrics_dir() / "runs.csv")
    s = pd.read_csv(metrics_dir() / "summary.csv")
    lines = ["| Dataset | Selected encoder (by validation) | Stage-2 prototype target "
             "(branch A) | Validation headroom (probe − prototypes, full) | "
             "Stage-3 baseline: linear probe, full split (test) |",
             "|---|---|---|---|---|"]
    for ds in cfg["datasets"]:
        g = runs[(runs["dataset"] == ds) & (runs["k_shot"] == "full")]
        probe_val = g[g["head"] == "linear_probe"].groupby("encoder")["val_acc"].mean()
        enc = probe_val.idxmax()
        proto_val = g[(g["head"] == "image_prototype") & (g["encoder"] == enc)]["val_acc"]
        headroom = ""
        if len(proto_val) and pd.notna(proto_val.iloc[0]) and str(proto_val.iloc[0]) != "":
            headroom = (f"{100 * probe_val[enc]:.2f} − {100 * float(proto_val.iloc[0]):.2f} "
                        f"= {100 * (probe_val[enc] - float(proto_val.iloc[0])):.2f} pts")
        probe_test = s[(s["dataset"] == ds) & (s["head"] == "linear_probe")
                       & (s["k_shot"] == "full") & (s["encoder"] == enc)].iloc[0]
        spec = "" if cfg["datasets"][ds]["spec_selected"] else " ‡"
        target = (f"class-mean prototypes $\\mu_c$ of the selected training subset, "
                  f"{encoder_label(enc, short=True)} features")
        lines.append(f"| {dataset_label(ds)}{spec} | {encoder_label(enc, short=True)} "
                     f"| {target} | {headroom} "
                     f"| {100 * probe_test['mean_acc']:.2f} ± {100 * probe_test['std_acc']:.2f} |")
    lines.append("")
    lines.append("Encoder selection: highest mean **validation** accuracy of the full-split "
                 "linear probe (`runs.csv`, `val_acc`); test accuracy plays no role. "
                 "Validation headroom: full-split probe validation accuracy minus full-split "
                 "image-prototype validation accuracy on the same encoder — the gap a "
                 "Stage-2 Flow-Matching decision layer has room to close, measured without "
                 "touching the test split. The Stage-3 baseline column is the one-time test "
                 "read-out published in the accuracy table. ‡ = beyond our selected pair.")
    out = metrics_dir() / "handoff_table.md"
    table = "\n".join(lines)
    out.write_text(table, encoding="utf-8")
    return table, out


def main():
    cfg = load_config()
    s = pd.read_csv(metrics_dir() / "summary.csv")
    header = [K_HEAD[k] for k in K_COLS] + ["no training images"]
    lines = ["| Dataset | Encoder | Baseline | " + " | ".join(header) + " |",
             "|---|---|---|" + "---|" * len(header)]
    for ds in cfg["datasets"]:
        g = s[s["dataset"] == ds]
        spec = "" if cfg["datasets"][ds]["spec_selected"] else " ‡"
        # Best supervised configuration per (dataset, K), marked in bold. Computed
        # here so the emphasis is generated, never hand-applied.
        best = {k: g[(g["k_shot"] == k) & (g["head"] != "zeroshot_clip")]["mean_acc"].max()
                for k in K_COLS}
        for (enc, head), gg in g.groupby(["encoder", "head"], sort=False):
            if head == "zeroshot_clip":
                cells = ["—", "—", "—", cell(gg[gg["k_shot"] == "none"])]
            else:
                cells = []
                for k in K_COLS:
                    row = gg[gg["k_shot"] == k]
                    txt = cell(row)
                    if not row.empty and row.iloc[0]["mean_acc"] == best[k]:
                        txt = f"**{txt}**"
                    cells.append(txt)
                cells.append("—")
            lines.append(f"| {dataset_label(ds)}{spec} | {encoder_label(enc, short=True)} "
                         f"| {head_label(head)} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("**Bold** marks the best supervised configuration in each "
                 "(dataset, training-set size) column; zero-shot CLIP is excluded from "
                 "that comparison because it uses no training images. "
                 "Top-1 accuracy (%) on the complete official test split. "
                 "5-shot / 10-shot: mean ± std over 3 training-subset seeds; "
                 "full linear probe: mean ± std over 3 initialization seeds; "
                 "full image prototypes and zero-shot CLIP are single deterministic runs. "
                 "Zero-shot CLIP uses no labeled training images, so it has one value only. "
                 "‡ = beyond our selected dataset pair (the specification asks for any two "
                 "of the three; we selected DTD + FGVC-Aircraft and additionally ran the third). "
                 "A standard deviation of exactly 0.00 at Flowers-102 K = 10 is not a rounding "
                 "artifact: that dataset's official training split holds exactly 10 images per "
                 "class, so all three 10-shot subsets are the same set of images.")
    table = "\n".join(lines)
    out = metrics_dir() / "accuracy_table.md"
    out.write_text(table, encoding="utf-8")
    print(table)
    print(f"\nwritten: {out}")

    for tbl, path in (paired_table(), macro_table(), handoff_table()):
        print("\n" + tbl)
        print(f"\nwritten: {path}")


if __name__ == "__main__":
    main()
