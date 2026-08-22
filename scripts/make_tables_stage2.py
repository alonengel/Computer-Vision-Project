"""Stage-2 tables (spec item 1): FM vs the Stage-1 prototype baseline, with the
change relative to the baseline, DeltaAcc = Acc_FM - Acc_baseline.

Generated programmatically from results/metrics/summary_stage2.csv and
runs_stage2.csv (never typed by hand), written to results/metrics/ as markdown
for the report and notebook.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.evaluation import metrics_dir
from src.utils import load_config
from src.visualize import dataset_label, encoder_label, head_label

K_COLS = ["5shot", "10shot", "full"]
K_HEAD = {"5shot": "K = 5", "10shot": "K = 10", "full": "full train split"}
HEAD_ORDER = [("fm_standard", 4), ("fm_standard", 12), ("fm_rollout", 4), ("fm_rollout", 12)]


def fm_cell(r):
    if r.empty:
        return "—"
    r = r.iloc[0]
    acc = f"{100 * r['mean_acc']:.2f}"
    if r["n_runs"] > 1:
        acc += f" ± {100 * r['std_acc']:.2f}"
    d = f"Δ {100 * r['delta_mean']:+.2f}"
    if r["n_runs"] > 1 and r["delta_std"] > 0:
        d += f" ± {100 * r['delta_std']:.2f}"
    return f"{acc} ({d})"


def branch_table(target):
    cfg = load_config()
    s2 = pd.read_csv(metrics_dir() / "summary_stage2.csv")
    s1 = pd.read_csv(metrics_dir() / "summary.csv")
    g_all = s2[s2["target"] == target]
    mark = "" if target == cfg["stage2"]["spec_selected_branch"] else " ‡"

    lines = ["| Dataset | Encoder | Head | " + " | ".join(K_HEAD[k] for k in K_COLS) + " |",
             "|---|---|---|---|---|---|"]
    for (ds, enc), g in g_all.groupby(["dataset", "encoder"], sort=False):
        ds_cell = f"{dataset_label(ds)}{mark}"
        best = {k: max(g[g["k_shot"] == k]["mean_acc"].max(),
                       g[g["k_shot"] == k]["baseline_mean"].max())
                for k in K_COLS}
        # Stage-1 baseline row
        if target == "image_prototype":
            b = s1[(s1["dataset"] == ds) & (s1["encoder"] == enc)
                   & (s1["head"] == "image_prototype")]
            cells = []
            for k in K_COLS:
                r = b[b["k_shot"] == k].iloc[0]
                txt = f"{100 * r['mean_acc']:.2f}"
                if r["n_runs"] > 1:
                    txt += f" ± {100 * r['std_acc']:.2f}"
                cells.append(f"**{txt}**" if np.isclose(r["mean_acc"], best[k]) else txt)
            base_label = f"Stage-1 {head_label('image_prototype')} (baseline)"
        else:
            zs = s1[(s1["dataset"] == ds) & (s1["head"] == "zeroshot_clip")].iloc[0]
            txt = f"{100 * zs['mean_acc']:.2f}"
            cells = [(f"**{txt}**" if np.isclose(zs["mean_acc"], best[k]) else txt)
                     for k in K_COLS]
            base_label = f"Stage-1 {head_label('zeroshot_clip')} (reference, K-independent)"
        lines.append(f"| {ds_cell} | {encoder_label(enc, short=True)} "
                     f"| {base_label} | " + " | ".join(cells) + " |")
        for head, T in HEAD_ORDER:
            cells = []
            for k in K_COLS:
                r = g[(g["head"] == head) & (g["T"] == T) & (g["k_shot"] == k)]
                txt = fm_cell(r)
                if not r.empty and np.isclose(r.iloc[0]["mean_acc"], best[k]):
                    txt = f"**{txt}**"
                cells.append(txt)
            lines.append(f"| {ds_cell} | {encoder_label(enc, short=True)} "
                         f"| {head_label(head)}, T = {T} | " + " | ".join(cells) + " |")
    lines.append("")
    if target == "image_prototype":
        lines.append(
            "Top-1 accuracy (%) on the complete official test split; Δ = change vs the "
            "Stage-1 image-prototype baseline of the **identical** setting (same subset "
            "indices, seeds and prototypes). K ∈ {5, 10}: mean ± sample std over the 3 "
            "subset seeds, Δ paired per seed (mean ± std of per-seed differences). "
            "full: 3 FM initialization seeds against the single deterministic full-split "
            "baseline, so the Δ spread there measures FM training stochasticity only. "
            "The two Standard-FM rows of a setting share one trained network (standard "
            "FM training is independent of T; only inference differs). **Bold** marks "
            "the best value per column within each dataset–encoder block.")
    else:
        lines.append(
            "‡ Extension beyond the specification (the spec's Stage-2 branch is the "
            "image-prototype branch selected in Stage 1, ADR 0006). FM here transports "
            "**CLIP RN50 image features** toward the fixed CLIP **text** prototypes. "
            "Unlike the Stage-1 zero-shot reference, FM training consumes K labeled "
            "images per class — these rows are *supervised transport on CLIP features*, "
            "never \"zero-shot\". Δ therefore answers \"does supervised transport toward "
            "text prototypes beat zero-shot classification?\", **not** \"does the FM "
            "layer help?\" — only the image-prototype branch isolates the FM effect, "
            "because its baseline uses the identical labeled subset. The reference "
            "value is a single deterministic run, so Δ carries no pairing; at K ∈ "
            "{5, 10} the spread over the 3 subset seeds is reported on the accuracy."
        )
    out = metrics_dir() / f"stage2_{target}_table.md"
    table = "\n".join(lines)
    out.write_text(table, encoding="utf-8")
    return table, out


def raw_branch_table(target):
    """The raw-feature (literal-spec, no input normalization) version of the
    grid, side by side with the published normalized version. Generated only
    once runs_stage2_raw.csv exists."""
    cfg = load_config()
    s_raw = pd.read_csv(metrics_dir() / "summary_stage2_raw.csv")
    s_norm = pd.read_csv(metrics_dir() / "summary_stage2.csv")
    g_all = s_raw[s_raw["target"] == target]
    mark = "" if target == cfg["stage2"]["spec_selected_branch"] else " ‡"

    lines = ["| Dataset | Encoder | Head | " + " | ".join(K_HEAD[k] for k in K_COLS) + " |",
             "|---|---|---|---|---|---|"]
    for (ds, enc), g in g_all.groupby(["dataset", "encoder"], sort=False):
        ds_cell = f"{dataset_label(ds)}{mark}"
        for head, T in HEAD_ORDER:
            cells = []
            for k in K_COLS:
                r = g[(g["head"] == head) & (g["T"] == T) & (g["k_shot"] == k)]
                n = s_norm[(s_norm["dataset"] == ds) & (s_norm["encoder"] == enc)
                           & (s_norm["target"] == target) & (s_norm["head"] == head)
                           & (s_norm["T"] == T) & (s_norm["k_shot"] == k)]
                if r.empty or n.empty:
                    cells.append("—")
                    continue
                r, n = r.iloc[0], n.iloc[0]
                acc = f"{100 * r['mean_acc']:.2f}"
                if r["n_runs"] > 1:
                    acc += f" ± {100 * r['std_acc']:.2f}"
                cells.append(f"{acc} (norm {100 * (r['mean_acc'] - n['mean_acc']):+.2f})")
            lines.append(f"| {ds_cell} | {encoder_label(enc, short=True)} "
                         f"| {head_label(head)}, T = {T} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(
        "**Raw-feature version** of the full Stage-2 grid — the literal-spec "
        "formulation (ẑ₀ = z, no input L2-normalization), run under the identical "
        "protocol: same velocity network, recipe, prototypes, committed subsets, "
        "seeds, checkpoint rule and test split as the published normalized version. "
        "Top-1 (%) on the complete official test split; mean ± sample std over the "
        "same 3 runs. The parenthesis gives the raw-minus-normalized difference of "
        "setting means (negative = the normalized version is better). The Stage-1 "
        "baselines are identical for both versions (cosine classification is "
        "scale-invariant). The normalized version remains the primary published "
        "result (ADR 0007 §3).")
    out = metrics_dir() / f"stage2_raw_{target}_table.md"
    table = "\n".join(lines)
    out.write_text(table, encoding="utf-8")
    return table, out


def paired_delta_table():
    """Per-seed-paired deltas, image branch only, K in {5,10} — the regime where
    pairing is genuine (baseline and FM share subset seeds)."""
    runs2 = pd.read_csv(metrics_dir() / "runs_stage2.csv")
    g = runs2[(runs2["target"] == "image_prototype") & (runs2["k_shot"] != "full")]
    lines = ["| Dataset | Encoder | Head | K | Δ per seed (paired) | Seeds favouring FM |",
             "|---|---|---|---|---|---|"]
    for (ds, enc, head, T, k), gg in g.groupby(["dataset", "encoder", "head", "T", "k_shot"],
                                               sort=False):
        d = 100 * gg.sort_values("seed")["delta_acc"].to_numpy()
        lines.append(f"| {dataset_label(ds)} | {encoder_label(enc, short=True)} "
                     f"| {head_label(head)}, T = {T} | {k.replace('shot', '')} "
                     f"| {d.mean():+.2f} ± {d.std(ddof=1):.2f} "
                     f"| {int((d > 0).sum())} / {len(d)} |")
    lines.append("")
    lines.append("Mean ± sample std (ddof = 1) of the per-seed difference "
                 "FM − Stage-1 image-prototype baseline, both computed from identical "
                 "committed subset indices and prototypes. Positive favours FM. "
                 "n = 3 seeds; no confidence intervals are quoted at this sample size.")
    out = metrics_dir() / "stage2_paired_delta_table.md"
    table = "\n".join(lines)
    out.write_text(table, encoding="utf-8")
    return table, out


def main():
    for target in ("image_prototype", "clip_text"):
        tbl, path = branch_table(target)
        print(tbl)
        print(f"\nwritten: {path}\n")
    tbl, path = paired_delta_table()
    print(tbl)
    print(f"\nwritten: {path}")
    if (metrics_dir() / "summary_stage2_raw.csv").exists():
        for target in ("image_prototype", "clip_text"):
            tbl, path = raw_branch_table(target)
            print("\n" + tbl)
            print(f"\nwritten: {path}")


if __name__ == "__main__":
    main()
