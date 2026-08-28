"""Stage-3 figures: training behaviour + feature-space visualization.

Per ADR 0008 §10: for both strategies the COMPARABLE end-to-end metrics
(train/validation pipeline CE and accuracy) share axes; strategy-internal
quantities (S1 displacement penalty, S2 FM-regression loss, S2 target CE
before/after, displacements, trust-region hit rate) are drawn on separate
axes — they measure different things. Feature viz: one PCA fitted jointly on
[z, z_hat_S1, z_hat_S2] in RAW space, the shared viz_selection classes and
class colours, representative seed 0.

Usage: python scripts/make_figures_stage3.py [curves|diag|features]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np

from make_figures import viz_selection
from make_figures_stage2 import _joint_pca, pc_labels
from src.embeddings import load_features
from src.evaluation import artifacts_dir
from src.stage3 import load_pinned_probe, load_stage3_fm
from src.utils import load_config
from src.visualize import (_save, dataset_label, encoder_label, feature_projection)

S1_COLOR, S2_COLOR = "#0173B2", "#D55E00"
REP_SEED = 0


def k_label():
    return f"{load_config()['stage3']['k_shot']}shot"


def curve(ds, enc, head, seed=REP_SEED):
    p = artifacts_dir("curves_stage3") / f"{ds}_{enc}_{head}_seed{seed}.json"
    with open(p) as f:
        return json.load(f)


def curve_charts():
    """Comparable end-to-end metrics for both strategies (representative seed 0)."""
    cfg = load_config()
    for ds, enc in cfg["stage3"]["settings"]:
        h1, h2 = curve(ds, enc, "fm_s1"), curve(ds, enc, "fm_s2")
        fig, (ax_ce, ax_acc) = plt.subplots(1, 2, figsize=(12.8, 4.6))
        for h, color, name in ((h1, S1_COLOR, "Strategy 1"), (h2, S2_COLOR, "Strategy 2")):
            ax_ce.plot(h["epoch"], h["train_pipeline_ce"], color=color, ls="--",
                       lw=1.8, label=f"{name} — train")
            ax_ce.plot(h["epoch"], h["val_ce"], color=color, ls="-", lw=2.2,
                       label=f"{name} — validation")
            ax_acc.plot(h["epoch"], [100 * v for v in h["train_pipeline_acc"]],
                        color=color, ls="--", lw=1.8, label=f"{name} — train")
            ax_acc.plot(h["epoch"], [100 * v for v in h["val_acc"]], color=color,
                        ls="-", lw=2.2, label=f"{name} — validation")
        ax_ce.set_xlabel("epoch"); ax_ce.set_ylabel("pipeline cross-entropy")
        ax_acc.set_xlabel("epoch"); ax_acc.set_ylabel("pipeline top-1 accuracy (%)")
        for ax in (ax_ce, ax_acc):
            ax.legend(fontsize=9)
        fig.suptitle(f"{dataset_label(ds)} — {encoder_label(enc, short=True)}: "
                     f"end-to-end training behaviour (both strategies, seed 0)",
                     y=1.03, fontsize=13)
        fig.tight_layout()
        print("figure:", _save(fig, f"stage3_curves_{ds}_{enc}.png"))


def diag_charts():
    """Strategy-internal diagnostics — separate axes (not comparable metrics)."""
    cfg = load_config()
    for ds, enc in cfg["stage3"]["settings"]:
        h1, h2 = curve(ds, enc, "fm_s1"), curve(ds, enc, "fm_s2")
        fig, axes2d = plt.subplots(2, 2, figsize=(11.6, 8.0))
        axes = axes2d.ravel()
        a = axes[0]
        a.plot(h1["epoch"], h1["mean_disp"], color=S1_COLOR, lw=2,
               label="Strategy 1: mean ‖ẑ−z‖ (train)")
        a.plot(h2["epoch"], h2["mean_disp"], color=S2_COLOR, lw=2,
               label="Strategy 2: mean ‖ẑ−z‖ (train)")
        a.plot(h2["epoch"], h2["mean_target_disp"], color=S2_COLOR, ls="--", lw=1.6,
               label="Strategy 2: mean ‖target−z‖")
        a.set_title("displacement (feature units)", fontsize=11)
        a.legend(fontsize=8)
        a = axes[1]
        a.semilogy(h1["epoch"], h1["train_penalty"], color=S1_COLOR, lw=2)
        a.set_title("S1 relative displacement penalty (log)", fontsize=11)
        a = axes[2]
        a.semilogy(h2["epoch"], h2["fm_loss"], color=S2_COLOR, lw=2)
        a.set_title("S2 FM regression loss (log)", fontsize=11)
        a = axes[3]
        a.plot(h2["epoch"], h2["ce_unprojected_zhat"], color="#949494", lw=1.6,
               label="CE(ẑ) unprojected (diagnostic)")
        a.plot(h2["epoch"], h2["target_ce_before"], color=S2_COLOR, ls="--", lw=1.8,
               label="CE at projected $u_0$")
        a.plot(h2["epoch"], h2["target_ce_after"], color=S2_COLOR, lw=2.2,
               label="CE at selected target")
        a2 = a.twinx()
        a2.plot(h2["epoch"], [100 * v for v in h2["hit_rate"]], color="#029E73",
                ls=":", lw=1.8)
        a2.set_ylabel("trust-region hit rate (%)", color="#029E73", fontsize=9)
        a2.tick_params(axis="y", labelcolor="#029E73")
        a2.grid(False)
        a.set_title("S2 target construction (phase-1 snapshot)", fontsize=11)
        a.legend(fontsize=8)
        for ax in axes:
            ax.set_xlabel("epoch")
        fig.suptitle(f"{dataset_label(ds)} — {encoder_label(enc, short=True)}: "
                     f"strategy-internal diagnostics (separate axes)",
                     y=1.0, fontsize=13)
        fig.tight_layout()
        print("figure:", _save(fig, f"stage3_diag_{ds}_{enc}.png"))


def feature_charts():
    cfg = load_config()
    vz = cfg["feature_viz"]
    mdir = artifacts_dir("stage3_models")
    for ds, enc in cfg["stage3"]["settings"]:
        classes, idx = viz_selection(ds, vz["n_classes"], vz["class_seed"],
                                     vz["max_per_class"])
        f = load_features(ds, "test", enc)
        names_all = f["class_names"]
        X = f["features"][idx].float()          # RAW space — no normalization
        remap = {int(c): j for j, c in enumerate(classes)}
        y = np.array([remap[int(v)] for v in f["labels"][idx].numpy()])

        probe = load_pinned_probe(mdir / f"probe_{ds}_{enc}_{k_label()}_seed{REP_SEED}.pt")
        fms = {h: load_stage3_fm(mdir / f"{ds}_{enc}_{h}_{k_label()}_seed{REP_SEED}.pt",
                                 probe) for h in ("fm_s1", "fm_s2")}
        Z1 = fms["fm_s1"].transport(X)
        Z2 = fms["fm_s2"].transport(X)
        pca, (xy0, xy1, xy2) = _joint_pca([X.numpy(), Z1.numpy(), Z2.numpy()])
        panels = [{"title": "Original features z", "xy": xy0, "labels": y},
                  {"title": "After Strategy 1 (ẑ)", "xy": xy1, "labels": y},
                  {"title": "After Strategy 2 (ẑ)", "xy": xy2, "labels": y}]
        print("figure:", feature_projection(
            panels, [names_all[c] for c in classes],
            f"{dataset_label(ds)} — {encoder_label(enc, short=True)}: features "
            f"before and after the Stage-3 FM (joint PCA, seed 0)",
            f"stage3_features_{ds}_{enc}.png", axis_labels=pc_labels(pca)))


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    steps = {"curves": curve_charts, "diag": diag_charts, "features": feature_charts}
    for name, fn in steps.items():
        if only and name != only:
            continue
        print(f"--- {name} ---", flush=True)
        fn()


if __name__ == "__main__":
    main()
