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
                   _curve(ds, enc, target, "standard", REP_K, REP_SEED)),
                  ("Rolled-out FM, T = 4", STAGE2_COLORS[("fm_rollout", 4)],
                   _curve(ds, enc, target, "rollout", REP_K, REP_SEED, T=4)),
                  ("Rolled-out FM, T = 12", STAGE2_COLORS[("fm_rollout", 12)],
                   _curve(ds, enc, target, "rollout", REP_K, REP_SEED, T=12))]
        panels.append({"title": f"{dataset_label(ds)} — {encoder_label(enc, short=True)}"
                                f"\n(toward {target_label(target)})", "curves": curves})
    for i in range(0, len(panels), 3):
        print("figure:", fm_training_curves(
            panels[i:i + 3], f"stage2_curves_{i // 3}.png",
            "FM training loss — representative 10-shot run (subset seed 0); "
            "reported checkpoint = minimum-training-loss epoch (ADR 0007 §7 fallback)"))

    # stability criterion over all full-run curves
    unstable = []
    for p in sorted(artifacts_dir("curves_stage2").glob("*.json")):
        if "_smoke" in p.name:
            continue
        with open(p) as f:
            h = json.load(f)["train_loss"]
        if h[-1] > 1.05 * min(h):
            unstable.append((p.name, h[-1], min(h)))
    print(f"stability check (ADR 0007 §7): {len(unstable)} unstable "
          f"of {len(list(artifacts_dir('curves_stage2').glob('*.json')))} curves")
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


def feature_charts():
    for target, ds, enc in s2_pairs():
        classes, idx, X, Xstd, Xroll, protos, y, sel_names, _, _ = _rep_setting(ds, enc, target)
        _, (xy0, xy1, xy2, pxy) = _joint_pca([X.numpy(), Xstd.numpy(), Xroll.numpy(),
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
            f"stage2_features_{ds}_{enc}_{target}.png"))


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
            f"stage2_traj_{ds}_{enc}_{target}.png"))


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    steps = {"acc": acc_charts, "curves": curve_charts,
             "features": feature_charts, "traj": traj_charts}
    for name, fn in steps.items():
        if only and name != only:
            continue
        print(f"--- {name} ---", flush=True)
        fn()


if __name__ == "__main__":
    main()
