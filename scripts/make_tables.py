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


def main():
    cfg = load_config()
    s = pd.read_csv(metrics_dir() / "summary.csv")
    header = [K_HEAD[k] for k in K_COLS] + ["no training images"]
    lines = ["| Dataset | Encoder | Baseline | " + " | ".join(header) + " |",
             "|---|---|---|" + "---|" * len(header)]
    for ds in cfg["datasets"]:
        g = s[s["dataset"] == ds]
        spec = "" if cfg["datasets"][ds]["spec_selected"] else " ‡"
        for (enc, head), gg in g.groupby(["encoder", "head"], sort=False):
            if head == "zeroshot_clip":
                cells = ["—", "—", "—", cell(gg[gg["k_shot"] == "none"])]
            else:
                cells = [cell(gg[gg["k_shot"] == k]) for k in K_COLS] + ["—"]
            lines.append(f"| {dataset_label(ds)}{spec} | {encoder_label(enc, short=True)} "
                         f"| {head_label(head)} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("Top-1 accuracy (%) on the complete official test split. "
                 "5-shot / 10-shot: mean ± std over 3 training-subset seeds; "
                 "full linear probe: mean ± std over 3 initialization seeds; "
                 "full image prototypes and zero-shot CLIP are single deterministic runs. "
                 "Zero-shot CLIP uses no labeled training images, so it has one value only. "
                 "‡ = beyond the spec's required pair of datasets. "
                 "A standard deviation of exactly 0.00 at Flowers-102 K = 10 is not a rounding "
                 "artifact: that dataset's official training split holds exactly 10 images per "
                 "class, so all three 10-shot subsets are the same set of images.")
    table = "\n".join(lines)
    out = metrics_dir() / "accuracy_table.md"
    out.write_text(table, encoding="utf-8")
    print(table)
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
