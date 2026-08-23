"""Regenerate every Stage 2 figure from saved metrics, curves and trained FM models.

Covers spec items 1-4: accuracy vs training-set size (with the Stage-1 baseline),
FM training-loss curves (standard vs rolled-out), feature-space comparisons
(original / after standard FM / after rolled-out FM, jointly projected), and flow
trajectories in a joint PCA plane.

Representative setting for figures 3-4: K = 10, subset seed 0, T = 12 — fixed a
priori, mirroring Stage 1's representative-curve convention (10-shot seed 0).

Usage: python scripts/make_figures_stage2.py [acc|curves|features|traj]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import torch.nn.functional as F

from make_figures import viz_selection
from src.embeddings import load_features
from src.evaluation import artifacts_dir, metrics_dir
from src.flow_matching import load_fm_head
from src.utils import load_config
from src.visualize import (STAGE2_COLORS, TARGET_NAMES, dataset_label, encoder_label,
                           feature_projection, flow_trajectory_chart,
                           fm_training_curves, stage2_accuracy_chart)

REP_K, REP_SEED, REP_T = "10shot", 0, 12  # representative setting, fixed a priori


def target_label(target):
    """Display name for a transport target; the CLIP branch carries its ‡ mark."""
    from src.visualize import TARGET_NAMES
    return TARGET_NAMES[target] + (" ‡" if target == "clip_text" else "")


def s2_pairs():
    cfg = load_config()
    for target, pairs in cfg["stage2"]["branches"].items():
        for ds, enc in pairs:
            yield target, ds, enc


def model_path(ds, enc, target, mode, k, seed, T=None):
    suffix = f"_T{T}" if mode == "rollout" else ""
    return artifacts_dir("fm_models") / f"{ds}_{enc}_{target}_fm_{mode}{suffix}_{k}_seed{seed}.pt"


# --------------------------------------------------------------------------- #

def acc_charts():
    s1 = pd.read_csv(metrics_dir() / "summary.csv")
    s2 = pd.read_csv(metrics_dir() / "summary_stage2.csv")
    for target, ds, enc in s2_pairs():
        rows = s2[(s2["dataset"] == ds) & (s2["encoder"] == enc) & (s2["target"] == target)]
        if target == "image_prototype":
            b = s1[(s1["dataset"] == ds) & (s1["encoder"] == enc)
                   & (s1["head"] == "image_prototype")]
            baseline = {"series": [(r["k_shot"], r["mean_acc"], r["std_acc"], r["n_runs"])
                                   for _, r in b.iterrows()]}
            subtitle = ("(error bars: ± sample std over 3 runs; full-split baseline and "
                        "both its FM deltas share one deterministic baseline run)")
        else:
            zs = s1[(s1["dataset"] == ds) & (s1["head"] == "zeroshot_clip")]
            baseline = {"hline": float(zs["mean_acc"].iloc[0])}
            subtitle = ("(‡ extension — supervised transport on CLIP features; the "
                        "reference uses no training images)")
        print("figure:", stage2_accuracy_chart(ds, enc, target, rows, baseline,
                                               f"stage2_acc_{ds}_{enc}_{target}.png",
                                               subtitle))


# --------------------------------------------------------------------------- #

def _curve(ds, enc, target, mode, k, seed, T=None):
    suffix = f"_T{T}" if mode == "rollout" else ""
    p = artifacts_dir("curves_stage2") / \
        f"{ds}_{enc}_{target}_fm_{mode}{suffix}_{k}_seed{seed}.json"
    with open(p) as f:
        return json.load(f)


def curve_charts():
    """Representative training-loss curves + the ADR 0007 §7 stability criterion
    evaluated over EVERY saved full-run curve (final loss vs 1.05 x running min)."""
    panels = []
    for target, ds, enc in s2_pairs():
        curves = [("Standard FM", STAGE2_COLORS[("fm_standard", 12)],
                   _curve(ds, enc, target, "standard", REP_K, REP_SEED), "-"),
                  ("Rolled-out FM, T = 4", STAGE2_COLORS[("fm_rollout", 4)],
                   _curve(ds, enc, target, "rollout", REP_K, REP_SEED, T=4), "--"),
                  ("Rolled-out FM, T = 12", STAGE2_COLORS[("fm_rollout", 12)],
                   _curve(ds, enc, target, "rollout", REP_K, REP_SEED, T=12), "-.")]
        panels.append({"title": f"{dataset_label(ds)} — {encoder_label(enc, short=True)}"
                                f"\n(toward {target_label(target)})", "curves": curves})
    for i in range(0, len(panels), 3):
        print("figure:", fm_training_curves(
            panels[i:i + 3], f"stage2_curves_{i // 3}.png",
            "FM training loss — representative 10-shot run (subset seed 0); "
            "reported checkpoint = minimum-training-loss epoch (ADR 0007 §7 fallback)"))

    # stability criterion over all published-grid curves (the raw-feature
    # version's curves carry a _raw suffix and are reported in their own section)
    curves = [p for p in sorted(artifacts_dir("curves_stage2").glob("*.json"))
              if "_smoke" not in p.name and "_raw" not in p.name]
    unstable = []
    for p in curves:
        with open(p) as f:
            h = json.load(f)["train_loss"]
        if h[-1] > 1.05 * min(h):
            unstable.append((p.name, h[-1], min(h)))
    print(f"stability check (ADR 0007 §7): {len(unstable)} unstable "
          f"of {len(curves)} curves")
    for name, fin, mn in unstable:
        print(f"  UNSTABLE {name}: final {fin:.4f} > 1.05 x min {mn:.4f}")
    return unstable


# --------------------------------------------------------------------------- #

def _joint_pca(parts, seed=0):
    """One PCA fitted jointly on the concatenation of all compared sets; returns
    the projections of each part in the SAME plane (spec requirement)."""
    from sklearn.decomposition import PCA

    X = np.concatenate([np.asarray(p) for p in parts])
    pca = PCA(n_components=2, random_state=seed).fit(X)
    return pca, [pca.transform(np.asarray(p)) for p in parts]


def _rep_setting(ds, enc, target):
    """Load the representative models + selected test features for one setting."""
    cfg = load_config()
    vz = cfg["feature_viz"]
    classes, idx = viz_selection(ds, vz["n_classes"], vz["class_seed"], vz["max_per_class"])
    f = load_features(ds, "test", enc)
    names_all = f["class_names"]
    X = F.normalize(f["features"][idx].float(), dim=-1)
    remap = {int(c): j for j, c in enumerate(classes)}
    y = np.array([remap[int(v)] for v in f["labels"][idx].numpy()])

    std = load_fm_head(model_path(ds, enc, target, "standard", REP_K, REP_SEED))
    roll = load_fm_head(model_path(ds, enc, target, "rollout", REP_K, REP_SEED, T=REP_T))
    protos = std.prototypes.cpu()[classes]
    Xstd = std.transport(X, REP_T)
    Xroll = roll.transport(X, REP_T)
    sel_names = [names_all[c] for c in classes]
    return classes, idx, X, Xstd, Xroll, protos, y, sel_names, std, roll


def pc_labels(pca):
    v = 100 * pca.explained_variance_ratio_
    return (f"PC1 ({v[0]:.1f}% var.)", f"PC2 ({v[1]:.1f}% var.)")


def feature_charts():
    for target, ds, enc in s2_pairs():
        classes, idx, X, Xstd, Xroll, protos, y, sel_names, _, _ = _rep_setting(ds, enc, target)
        pca, (xy0, xy1, xy2, pxy) = _joint_pca([X.numpy(), Xstd.numpy(), Xroll.numpy(),
                                                protos.numpy()])
        panels = [{"title": "Original features", "xy": xy0, "labels": y, "proto_xy": pxy},
                  {"title": f"After Standard FM (T = {REP_T})", "xy": xy1, "labels": y,
                   "proto_xy": pxy},
                  {"title": f"After Rolled-out FM (T = {REP_T})", "xy": xy2, "labels": y,
                   "proto_xy": pxy}]
        proto_kind = ("training-subset image prototypes (K = 10, seed 0)"
                      if target == "image_prototype" else "CLIP text prototypes ‡")
        print("figure:", feature_projection(
            panels, sel_names,
            f"{dataset_label(ds)} — {encoder_label(enc, short=True)}: features before "
            f"and after FM transport, with {proto_kind}\n(one PCA fitted jointly to all "
            f"three feature sets and the prototypes — every panel shares the same plane; "
            f"10-shot models, subset seed 0; qualitative view)",
            f"stage2_features_{ds}_{enc}_{target}.png", axis_labels=pc_labels(pca)))


def traj_charts(n_examples=4):
    for target, ds, enc in s2_pairs():
        classes, idx, X, Xstd, Xroll, protos, y, sel_names, std, roll = \
            _rep_setting(ds, enc, target)
        pca, (xy0, _, _, pxy) = _joint_pca([X.numpy(), Xstd.numpy(), Xroll.numpy(),
                                            protos.numpy()])
        rng = np.random.default_rng(0)
        picked_classes = rng.choice(len(classes), size=n_examples, replace=False)
        ex = [int(np.flatnonzero(y == c)[0]) for c in picked_classes]

        panels = []
        for head, label in ((std, "Standard FM"), (roll, "Rolled-out FM")):
            _, traj = head.transport(X[ex], REP_T, return_traj=True)  # [T+1, n, D]
            trajs = [(int(y[e]), pca.transform(traj[:, i].numpy()))
                     for i, e in enumerate(ex)]
            panels.append({"title": f"{label} (T = {REP_T})", "trajs": trajs})
        bg = {"xy": xy0, "labels": y, "proto_xy": pxy}
        print("figure:", flow_trajectory_chart(
            panels, bg, sel_names,
            f"{dataset_label(ds)} — {encoder_label(enc, short=True)}: Euler flow "
            f"trajectories toward {target_label(target)}\n(joint PCA plane shared with "
            f"the feature-comparison figure; background: original test features; "
            f"{n_examples} representative examples, 10-shot models, subset seed 0)",
            f"stage2_traj_{ds}_{enc}_{target}.png", axis_labels=pc_labels(pca)))


def reverse_charts():
    """Spec's OPTIONAL reverse exploration (one representative setting, both
    training modes): integrate the learned field backward from each selected
    class prototype (reverse_transport), drawn in the SAME jointly fitted PCA
    plane as the forward feature/trajectory figures — the projection is fitted
    once on the identical inputs, never refit for the reverse plots. Also
    writes the intermediate-time comparison (reverse-prototype state vs the
    forward centroid of the same class's selected test samples at the matching
    Euler time): cosine similarity + L2, per class per time, to CSV, plus a
    summary chart. Deterministic; uses saved checkpoints only."""
    import matplotlib.pyplot as plt

    from src.visualize import (STAGE2_COLORS, _save, head_label, reverse_flow_chart)

    ds, enc, target = "fgvc_aircraft", "dinov2_vits14", "image_prototype"
    tag = f"{ds}_{enc}_T{REP_T}"
    classes, idx, X, Xstd, Xroll, protos, y, sel_names, std, roll = \
        _rep_setting(ds, enc, target)
    pca, (xy0, _, _, pxy) = _joint_pca([X.numpy(), Xstd.numpy(), Xroll.numpy(),
                                        protos.numpy()])
    bg = {"xy": xy0, "labels": y, "proto_xy": pxy}

    panels, rows = [], []
    for head, label in ((std, "Standard FM"), (roll, "Rolled-out FM")):
        # reverse trajectories from the selected class prototypes (descending t)
        _, rtraj, times = head.reverse_transport(protos, REP_T, return_traj=True)
        panels.append({"title": f"{label} (T = {REP_T})",
                       "trajs": [(j, pca.transform(rtraj[:, j].numpy()))
                                 for j in range(len(classes))]})
        # forward states of the selected test samples, per class centroid per time
        _, ftraj = head.transport(X, REP_T, return_traj=True)  # [T+1, N, D]
        for j in range(len(classes)):
            fc = ftraj[:, y == j].mean(dim=1)                  # [T+1, D], t = k/T
            for k in range(REP_T + 1):
                rev = rtraj[REP_T - k, j]                      # reverse state at t = k/T
                cos = float(F.cosine_similarity(rev, fc[k], dim=0))
                rows.append({"head": f"fm_{head.mode}", "class_idx": j,
                             "class_name": sel_names[j], "t": k / REP_T,
                             "cosine": cos, "l2": float((rev - fc[k]).norm())})

    print("figure:", reverse_flow_chart(
        panels, bg, sel_names,
        f"{dataset_label(ds)} — {encoder_label(enc, short=True)}: approximate "
        f"backward integration from the class prototypes\n(reverse Euler on the "
        f"learned field, trained forward-only; same joint-PCA plane as the "
        f"feature/trajectory figures; 10-shot models, subset seed 0; qualitative)",
        f"stage2_reverse_flow_{tag}.png", axis_labels=pc_labels(pca)))

    df = pd.DataFrame(rows)
    csv_path = metrics_dir() / f"stage2_reverse_intermediate_{tag}.csv"
    df.to_csv(csv_path, index=False)
    print("written:", csv_path)

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.6))
    for ax, metric, ylabel in ((axes[0], "cosine", "cosine similarity"),
                               (axes[1], "l2", "$L_2$ distance")):
        for mode in ("fm_standard", "fm_rollout"):
            g = df[df["head"] == mode].groupby("t")[metric].agg(["mean", "std"])
            color = STAGE2_COLORS[(mode, 12)]
            ax.plot(g.index, g["mean"], color=color, lw=2.2,
                    label=f"{head_label(mode)} (mean ± std over "
                          f"{df['class_idx'].nunique()} classes)")
            ax.fill_between(g.index, g["mean"] - g["std"], g["mean"] + g["std"],
                            color=color, alpha=0.18)
        ax.set_xlabel("flow time $t$ (reverse runs $1 \\to 0$)")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=9)
    fig.suptitle(f"{dataset_label(ds)} — {encoder_label(enc, short=True)}: "
                 f"reverse-prototype state vs forward same-class centroid at "
                 f"matching Euler times (384-d space, not the PCA plane)",
                 y=1.03, fontsize=12.5)
    fig.tight_layout()
    print("figure:", _save(fig, f"stage2_reverse_intermediate_{tag}.png"))


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    steps = {"acc": acc_charts, "curves": curve_charts,
             "features": feature_charts, "traj": traj_charts,
             "reverse": reverse_charts}
    for name, fn in steps.items():
        if only and name != only:
            continue
        print(f"--- {name} ---", flush=True)
        fn()


if __name__ == "__main__":
    main()
