"""Stage-3 figures: training behaviour + feature-space visualization.

Per ADR 0008 §10: for both strategies the COMPARABLE end-to-end metrics
(train/validation pipeline CE and accuracy) share axes; strategy-internal
quantities (Rolled displacement penalty, Guided FM-regression loss, Guided target CE
before/after, displacements, trust-region hit rate) are drawn on separate
axes — they measure different things. Feature viz: one PCA fitted jointly on
[z, z_hat_S1, z_hat_S2] in RAW space, the shared viz_selection classes and
class colours, representative seed 0. Sweep chart: the seed-0 validation sweep
as one panel per (dataset, strategy), replacing the sweep table in the notebook.
Checkpoint chart: placement + relative displacement of the final models
(replaces the notebook table). Optional extension: joint-vs-control curves and a joint-PCA
feature figure (needs the models checkpointed by run_stage3_joint.py).

Usage: python scripts/make_figures_stage3.py
       [curves|diag|features|sweep|checkpoints|joint_curves|joint_features]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from make_figures import viz_selection
from make_figures_stage2 import _joint_pca, pc_labels
from src.embeddings import load_features
from src.evaluation import artifacts_dir
from src.stage3 import load_pinned_probe, load_stage3_fm
from src.utils import load_config
from src.visualize import (_save, dataset_label, encoder_label, feature_overlay,
                           feature_projection)

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
        for h, color, name in ((h1, S1_COLOR, "Rolled strategy"), (h2, S2_COLOR, "Guided strategy")):
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
               label="Rolled strategy: mean ‖ẑ−z‖ (train)")
        a.plot(h2["epoch"], h2["mean_disp"], color=S2_COLOR, lw=2,
               label="Guided strategy: mean ‖ẑ−z‖ (train)")
        a.plot(h2["epoch"], h2["mean_target_disp"], color=S2_COLOR, ls="--", lw=1.6,
               label="Guided strategy: mean ‖target−z‖")
        a.set_title("displacement (feature units)", fontsize=11)
        a.legend(fontsize=8)
        a = axes[1]
        a.semilogy(h1["epoch"], h1["train_penalty"], color=S1_COLOR, lw=2)
        a.set_title("Rolled strategy: relative displacement penalty (log)", fontsize=11)
        a = axes[2]
        a.semilogy(h2["epoch"], h2["fm_loss"], color=S2_COLOR, lw=2)
        a.set_title("Guided strategy: FM regression loss (log)", fontsize=11)
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
        a.set_title("Guided strategy: target construction (phase-1 snapshot)", fontsize=11)
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
                  {"title": "After the Rolled strategy (ẑ)", "xy": xy1, "labels": y},
                  {"title": "After the Guided strategy (ẑ)", "xy": xy2, "labels": y}]
        print("figure:", feature_projection(
            panels, [names_all[c] for c in classes],
            f"{dataset_label(ds)} — {encoder_label(enc, short=True)}: features "
            f"before and after the Stage-3 FM (joint PCA, seed 0)",
            f"stage3_features_{ds}_{enc}.png", axis_labels=pc_labels(pca)))
        # Same joint fit, overlaid: original (faded) -> transported (solid), per strategy.
        print("figure:", feature_overlay(
            [{"title": "Rolled strategy: original z (faded) → ẑ (solid)", "xy_before": xy0,
              "xy_after": xy1, "labels": y},
             {"title": "Guided strategy: original z (faded) → ẑ (solid)", "xy_before": xy0,
              "xy_after": xy2, "labels": y}],
            [names_all[c] for c in classes],
            f"{dataset_label(ds)} — {encoder_label(enc, short=True)}: before/after overlay "
            f"in the same joint-PCA plane (seed 0)",
            f"stage3_features_overlay_{ds}_{enc}.png", axis_labels=pc_labels(pca)))


# --------------------------------------------------------------------------- #
# Seed-0 validation sweep as a chart (replaces the sweep table in the notebook)
# --------------------------------------------------------------------------- #
PARAM_SYMBOL = {"lam": "λ", "beta": "β", "m": "m"}


def _config_label(param, sep="\n"):
    """'lam=1' -> 'λ = 1'; 'beta=0.25,m=3' -> 'β = 0.25<sep>m = 3'."""
    return sep.join(f"{PARAM_SYMBOL[k]} = {v}" for k, v in (kv.split("=") for kv in param.split(",")))


def _selected_param(rows):
    """Pre-registered selection rule (ADR 0008 §7): highest validation accuracy,
    ties -> lowest validation CE -> grid order (= file order)."""
    best = None
    for _, r in rows.iterrows():
        key = (round(float(r["val_acc"]), 12), -round(float(r["val_ce"]), 12))
        if best is None or key > best[0]:
            best = (key, r["param"])
    return best[1]


def sweep_chart():
    """Selection transparency as a figure: one panel per (dataset, strategy) —
    seed-0 validation top-1 of every swept configuration (markers), the
    validation-selected configuration as a starred marker, the checkpoint epoch
    under each configuration, and the pinned probe's seed-0 validation accuracy
    (= the pipeline at its identity initialization) as a dashed reference.
    Reads stage3_sweep.csv / runs_stage3.csv only; the table of record remains
    stage3_sweep_table.md."""
    from src.evaluation import metrics_dir

    cfg = load_config()
    sw = pd.read_csv(metrics_dir() / "stage3_sweep.csv")
    r3 = pd.read_csv(metrics_dir() / "runs_stage3.csv")
    settings = cfg["stage3"]["settings"]
    strategies = (("s1", S1_COLOR, "Rolled strategy — end-to-end rolled-out CE"),
                  ("s2", S2_COLOR, "Guided strategy — classifier-guided targets"))
    fig, axes = plt.subplots(len(settings), 2, figsize=(14.5, 4.9 * len(settings)),
                             squeeze=False)
    for i, (ds, enc) in enumerate(settings):
        probe = r3[(r3["dataset"] == ds) & (r3["encoder"] == enc)
                   & (r3["head"] == "pinned_probe") & (r3["seed"] == 0)].iloc[0]
        base = 100 * float(probe["val_acc"])
        for j, (strat, color, name) in enumerate(strategies):
            ax = axes[i][j]
            rows = sw[(sw["dataset"] == ds) & (sw["encoder"] == enc)
                      & (sw["strategy"] == strat)].reset_index(drop=True)
            assert (rows["status"] == "ok").all() and (rows["fallback"] == 0).all()
            sel = _selected_param(rows)
            # The configuration of record used for the three-seed test runs must
            # be the one the rule selects (same check as the repro check).
            rec = r3[(r3["dataset"] == ds) & (r3["encoder"] == enc)
                     & (r3["head"] == f"fm_{strat}") & (r3["seed"] == 0)].iloc[0]
            rec_param = (f"lam={int(rec['lam'])}" if strat == "s1"
                         else f"beta={rec['beta']},m={int(rec['m'])}")
            assert sel == rec_param, (ds, strat, sel, rec_param)

            accs = 100 * rows["val_acc"].to_numpy()
            x = np.arange(len(rows))
            lo, hi = min(accs.min(), base), max(accs.max(), base)
            pad = max(0.5, 0.22 * (hi - lo))
            ax.axhline(base, color="#555555", ls="--", lw=1.4, zorder=1)
            ax.text(-0.5, base - 0.06 * pad,
                    f"pinned probe (identity FM): {base:.2f}", ha="left", va="top",
                    fontsize=9.5, color="#555555")
            for k, (acc, param, ep) in enumerate(zip(accs, rows["param"], rows["checkpoint_epoch"])):
                is_sel = param == sel
                ax.scatter([k], [acc], s=(420 if is_sel else 130), color=color,
                           marker=("*" if is_sel else "o"),
                           edgecolors="black" if is_sel else "none", linewidths=1.3,
                           zorder=3, alpha=1.0 if is_sel else 0.75)
                ax.text(k, acc + 0.22 * pad, f"{acc:.2f}", ha="center", va="bottom",
                        fontsize=10.5, fontweight="bold" if is_sel else "normal")
            ax.set_xticks(x)
            ax.set_xticklabels([f"{_config_label(p)}\nep {e}"
                                for p, e in zip(rows["param"], rows["checkpoint_epoch"])],
                               fontsize=10)
            ax.set_xlim(-0.6, len(rows) - 0.4)
            ax.set_ylim(lo - 0.9 * pad, hi + 1.3 * pad)
            ax.set_ylabel("seed-0 validation top-1 (%)", fontsize=11)
            ties = int(np.isclose(rows["val_acc"], rows["val_acc"].max()).sum())
            tie_note = "  (tie on accuracy → lower validation CE)" if ties > 1 else ""
            ax.set_title(f"{dataset_label(ds)} — {encoder_label(enc, short=True)} · {name}\n"
                         f"selected: {_config_label(sel, sep=', ')}{tie_note}",
                         fontsize=11.5)
            ax.tick_params(axis="y", labelsize=10)
    fig.suptitle("Seed-0 validation sweep — every configuration the winners were chosen from "
                 "(star = validation-selected; ep = checkpoint epoch; y-axes zoomed to each "
                 "sweep; all runs completed at fallback level 0)", fontsize=12.5, y=1.0)
    fig.tight_layout()
    print("figure:", _save(fig, "stage3_sweep.png"))


# --------------------------------------------------------------------------- #
# Checkpoint placement + displacement (replaces the §4 table in the notebook)
# --------------------------------------------------------------------------- #
def checkpoint_chart():
    """One row per dataset: (left) validation-selected checkpoint epoch per subset
    seed, annotated with the best validation top-1; (right) the RELATIVE
    displacement of that checkpointed model on its training subset — the
    per-sample mean of ‖ẑ−z‖ / ‖z‖ in % (label: % and, in parentheses, the
    absolute mean ‖ẑ−z‖ in feature units) — grouped bars, Rolled vs Guided
    strategy, with the Guided trust-region radius (alpha · ‖z‖) as a dashed
    reference. Relative units make the two encoders comparable (mean feature
    norms ≈24 vs ≈50) and are the quantity both methods are defined in.
    Checkpoint epochs come from runs_stage3.csv (the table of record, full
    tie-break rule); displacements are recomputed from the checkpointed models
    and asserted to agree with the training-history value within 0.1 units."""
    from src.data import training_indices
    from src.evaluation import metrics_dir

    cfg = load_config()
    s3 = cfg["stage3"]
    r3 = pd.read_csv(metrics_dir() / "runs_stage3.csv")
    settings, seeds = s3["settings"], s3["subset_seeds"]
    n_epochs = s3["training"]["epochs"]
    alpha = s3["strategy2"]["alpha_trust_region"]
    mdir = artifacts_dir("stage3_models")
    heads = (("fm_s1", S1_COLOR, "Rolled strategy"), ("fm_s2", S2_COLOR, "Guided strategy"))
    width = 0.36
    fig, axes = plt.subplots(len(settings), 2, figsize=(14.5, 4.7 * len(settings)),
                             squeeze=False)
    for i, (ds, enc) in enumerate(settings):
        ax_ep, ax_disp = axes[i]
        f = load_features(ds, "train", enc)
        Xtr, ytr = f["features"], f["labels"].numpy()
        rel_max = 0.0
        for j, (head, color, name) in enumerate(heads):
            eps, rels, absd, accs = [], [], [], []
            for seed in seeds:
                row = r3[(r3["dataset"] == ds) & (r3["encoder"] == enc)
                         & (r3["head"] == head) & (r3["seed"] == seed)].iloc[0]
                h = curve(ds, enc, head, seed)
                ep = int(row["checkpoint_epoch"])
                assert h["epoch"][ep] == ep
                idx = training_indices(ds, s3["k_shot"], seed, ytr,
                                       fingerprint=f.get("pool_fingerprint"))
                X = Xtr[idx].float()
                probe = load_pinned_probe(mdir / f"probe_{ds}_{enc}_{k_label()}_seed{seed}.pt")
                fm = load_stage3_fm(mdir / f"{ds}_{enc}_{head}_{k_label()}_seed{seed}.pt", probe)
                d = (fm.transport(X) - X).norm(dim=-1)
                n = X.norm(dim=-1)
                # The checkpointed model must reproduce the history's displacement.
                assert abs(d.mean().item() - h["mean_disp"][ep]) < 0.1, (ds, head, seed)
                eps.append(ep)
                rels.append(100 * (d / n).mean().item())
                absd.append(d.mean().item())
                accs.append(100 * max(h["val_acc"]))
            x = np.arange(len(seeds)) + (j - 0.5) * width
            ax_ep.bar(x, eps, width, color=color, label=name)
            ax_disp.bar(x, rels, width, color=color, label=name)
            for xb, ep, acc in zip(x, eps, accs):
                ax_ep.text(xb, ep + 0.015 * n_epochs, f"ep {ep}\n{acc:.2f}%",
                           ha="center", va="bottom", fontsize=9.5)
            for xb, r, a in zip(x, rels, absd):
                ax_disp.text(xb, r * 1.02, f"{r:.1f}%\n({a:.2f})", ha="center",
                             va="bottom", fontsize=9.5,
                             bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=0.3))
            rel_max = max(rel_max, max(rels))
        ax_disp.axhline(100 * alpha, color="0.35", ls="--", lw=1.4, zorder=1,
                        label=f"Guided trust-region radius ({100 * alpha:.0f}% of ‖z‖)")
        for ax in (ax_ep, ax_disp):
            ax.set_xticks(range(len(seeds)))
            ax.set_xticklabels([f"subset seed {s}" for s in seeds], fontsize=10.5)
            ax.tick_params(axis="y", labelsize=10)
            ax.legend(fontsize=9.5, loc="upper left")
        ax_ep.set_ylim(0, n_epochs * 1.28)
        ax_ep.set_ylabel(f"checkpoint epoch (of {n_epochs})", fontsize=11)
        ax_ep.set_title(f"{dataset_label(ds)} — {encoder_label(enc, short=True)}: "
                        f"validation-selected checkpoint\n(label: epoch · best validation top-1)",
                        fontsize=11.5)
        ax_disp.set_ylim(0, max(rel_max, 100 * alpha) * 1.6)
        ax_disp.set_ylabel("mean ‖ẑ−z‖ / ‖z‖ at the checkpoint (%)", fontsize=11)
        ax_disp.set_title(f"{dataset_label(ds)} — {encoder_label(enc, short=True)}: "
                          f"relative displacement of the checkpointed model\n"
                          f"(training subset; label: % of feature norm · absolute units)",
                          fontsize=11.5)
    fig.suptitle("Checkpoint placement and relative displacement of the two mandatory "
                 "strategies' final models, all subset seeds", fontsize=12.5, y=1.0)
    fig.tight_layout()
    print("figure:", _save(fig, "stage3_checkpoints.png"))


# --------------------------------------------------------------------------- #
# Optional extension (ADR 0008 §9): training behaviour + feature space
# --------------------------------------------------------------------------- #
JOINT_COLOR, CTRL_COLOR = "#6A3D9A", "#029E73"     # distinct from the Rolled / Guided colours


def joint_curve_charts():
    """Joint FM + classifier fine-tuning vs the classifier-only continued-training
    control, seed 0: post-epoch full-train (dashed) and validation (solid)
    pipeline CE / top-1, with the validation-selected checkpoint of record
    (runs_stage3_joint.csv) starred."""
    from src.evaluation import metrics_dir

    cfg = load_config()
    rj = pd.read_csv(metrics_dir() / "runs_stage3_joint.csv")
    series = (("joint", "fm_joint", JOINT_COLOR, "Joint FM + classifier"),
              ("clf_only", "clf_only_continued", CTRL_COLOR, "Classifier-only control"))
    for ds, enc in cfg["stage3"]["settings"]:
        fig, (ax_ce, ax_acc) = plt.subplots(1, 2, figsize=(12.8, 4.6))
        for fname, head, color, name in series:
            h = curve(ds, enc, fname)
            ep = int(rj[(rj["dataset"] == ds) & (rj["encoder"] == enc)
                        & (rj["head"] == head) & (rj["seed"] == REP_SEED)]
                     ["checkpoint_epoch"].iloc[0])
            tr_ce = h.get("train_pipeline_ce", h["train_ce"])
            ax_ce.plot(h["epoch"], tr_ce, color=color, ls="--", lw=1.8,
                       label=f"{name} — train")
            ax_ce.plot(h["epoch"], h["val_ce"], color=color, ls="-", lw=2.2,
                       label=f"{name} — validation")
            if "train_pipeline_acc" in h:
                ax_acc.plot(h["epoch"], [100 * v for v in h["train_pipeline_acc"]],
                            color=color, ls="--", lw=1.8, label=f"{name} — train")
            ax_acc.plot(h["epoch"], [100 * v for v in h["val_acc"]], color=color,
                        ls="-", lw=2.2, label=f"{name} — validation")
            ax_acc.plot([ep], [100 * h["val_acc"][ep]], ls="none", marker="*", ms=16,
                        color=color, mec="black", zorder=5,
                        label=f"{name} — checkpoint (epoch {ep})")
        ax_ce.set_xlabel("epoch"); ax_ce.set_ylabel("pipeline cross-entropy")
        ax_acc.set_xlabel("epoch"); ax_acc.set_ylabel("pipeline top-1 accuracy (%)")
        ax_ce.legend(fontsize=8.5)
        ax_acc.legend(fontsize=8.5)
        fig.suptitle(f"{dataset_label(ds)} — {encoder_label(enc, short=True)}: "
                     f"optional extension — joint fine-tuning vs its classifier-only "
                     f"control (seed 0)", y=1.03, fontsize=13)
        fig.tight_layout()
        print("figure:", _save(fig, f"stage3_curves_joint_{ds}_{enc}.png"))


def joint_feature_charts():
    """Feature space of the optional extension: original z, after the mandatory
    Guided-strategy FM (frozen classifier), after the jointly fine-tuned FM — one PCA
    fitted jointly on the three sets, the shared viz_selection classes and
    class colours, seed 0. Uses the models checkpointed by run_stage3_joint.py."""
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
        fm_s2 = load_stage3_fm(mdir / f"{ds}_{enc}_fm_s2_{k_label()}_seed{REP_SEED}.pt", probe)
        joint_clf = load_pinned_probe(
            mdir / f"{ds}_{enc}_joint_clf_{k_label()}_seed{REP_SEED}.pt")
        fm_joint = load_stage3_fm(
            mdir / f"{ds}_{enc}_fm_joint_{k_label()}_seed{REP_SEED}.pt", joint_clf)
        Z2 = fm_s2.transport(X)
        Zj = fm_joint.transport(X)
        pca, (xy0, xy1, xy2) = _joint_pca([X.numpy(), Z2.numpy(), Zj.numpy()])
        panels = [{"title": "Original features z", "xy": xy0, "labels": y},
                  {"title": "After the Guided strategy (ẑ, frozen classifier)", "xy": xy1,
                   "labels": y},
                  {"title": "After joint FM + classifier fine-tuning (ẑ)", "xy": xy2,
                   "labels": y}]
        print("figure:", feature_projection(
            panels, [names_all[c] for c in classes],
            f"{dataset_label(ds)} — {encoder_label(enc, short=True)}: features before "
            f"and after the optional joint extension (joint PCA, seed 0)",
            f"stage3_features_joint_{ds}_{enc}.png", axis_labels=pc_labels(pca)))
        print("figure:", feature_overlay(
            [{"title": "Guided strategy (frozen classifier): z (faded) → ẑ (solid)",
              "xy_before": xy0, "xy_after": xy1, "labels": y},
             {"title": "Joint FM + classifier fine-tuning: z (faded) → ẑ (solid)",
              "xy_before": xy0, "xy_after": xy2, "labels": y}],
            [names_all[c] for c in classes],
            f"{dataset_label(ds)} — {encoder_label(enc, short=True)}: before/after overlay "
            f"of the optional extension in the same joint-PCA plane (seed 0)",
            f"stage3_features_joint_overlay_{ds}_{enc}.png", axis_labels=pc_labels(pca)))


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    steps = {"curves": curve_charts, "diag": diag_charts, "features": feature_charts,
             "sweep": sweep_chart, "checkpoints": checkpoint_chart,
             "joint_curves": joint_curve_charts, "joint_features": joint_feature_charts}
    for name, fn in steps.items():
        if only and name != only:
            continue
        print(f"--- {name} ---", flush=True)
        fn()


if __name__ == "__main__":
    main()
