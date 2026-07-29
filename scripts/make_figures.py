"""Regenerate every Stage 1 figure from cached features, saved metrics and artifacts.

Covers spec items 2-5: accuracy versus training-set size, linear-probe training
curves, row-normalized confusion matrices, and 2-D feature visualizations of a
readable subset of classes shown together with their class prototypes.

Usage: python scripts/make_figures.py [acc|curves|confusion|features|samples]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.classifiers import PrototypeClassifier
from src.data import training_indices
from src.embeddings import clip_text_path, encoders_for, load_features, supervised_encoders_for
from src.evaluation import (artifacts_dir, load_predictions, metrics_dir,
                            row_normalized_confusion)
from src.utils import load_config
from src.visualize import (accuracy_vs_trainsize, confusion, dataset_label, encoder_label,
                           feature_projection, training_curves)


def summary():
    return pd.read_csv(metrics_dir() / "summary.csv")


# --------------------------------------------------------------------------- #

def acc_charts():
    s = summary()
    for ds, g in s.groupby("dataset"):
        print("figure:", accuracy_vs_trainsize(ds, g))


def curve_charts():
    """Representative 10-shot training/validation loss curves per dataset-encoder."""
    cfg = load_config()
    panels = []
    for ds in cfg["datasets"]:
        for enc in supervised_encoders_for(ds):
            path = artifacts_dir("curves") / f"{ds}_{enc}_10shot_seed0.json"
            if not path.exists():
                continue
            with open(path) as f:
                panels.append({"dataset": ds, "encoder": enc, "history": json.load(f)})
    for i in range(0, len(panels), 2):
        chunk = panels[i:i + 2]
        tag = "_".join(f"{p['dataset']}-{p['encoder']}" for p in chunk)
        print("figure:", training_curves(
            chunk, f"training_curves_{i // 2}.png",
            "Linear-probe training curves — representative 10-shot run "
            "(subset seed 0); dashed line = selected checkpoint"))
    return [p for p in panels]


def confusion_charts():
    """One row-normalized confusion matrix per dataset, for the full-split linear
    probe on that dataset's best encoder. The encoder is chosen by **validation**
    accuracy (never test), and the title reports the accuracy of the run actually
    plotted rather than the 3-seed mean."""
    runs = pd.read_csv(metrics_dir() / "runs.csv")
    for ds, g in runs.groupby("dataset"):
        cand = g[(g["head"] == "linear_probe") & (g["k_shot"] == "full")]
        enc = (cand.groupby("encoder")["val_acc"].mean().idxmax())
        pred, target = load_predictions(f"run_{ds}_{enc}_linear_probe_full")
        names = load_features(ds, "test", enc)["class_names"]
        cm = row_normalized_confusion(pred, target, len(names))
        plotted_acc = float((np.asarray(pred) == np.asarray(target)).mean())
        off = cm.copy()
        np.fill_diagonal(off, 0.0)
        flat = np.dstack(np.unravel_index(np.argsort(off, axis=None)[::-1], off.shape))[0]
        top = [(int(t), int(p), float(off[t, p])) for t, p in flat[:6]]
        print("figure:", confusion(
            cm, names,
            f"{dataset_label(ds)} — row-normalized confusion matrix\n"
            f"linear probe, full training split, {encoder_label(enc, short=True)} "
            f"(plotted run: top-1 {100 * plotted_acc:.2f}%)",
            f"confusion_{ds}.png", top_confusions=top))


# --------------------------------------------------------------------------- #

def viz_selection(dataset, n_classes, class_seed, max_per_class):
    """Deterministic class subset + test-example indices, shared by every encoder
    of a dataset so panels are directly comparable (spec requirement)."""
    labels = load_features(dataset, "test", "resnet18")["labels"].numpy()
    rng = np.random.default_rng(class_seed)
    classes = np.sort(rng.choice(np.unique(labels), size=n_classes, replace=False))
    idx = []
    for c in classes:
        pool = np.flatnonzero(labels == c)
        take = min(max_per_class, len(pool))
        idx.append(rng.choice(pool, size=take, replace=False))
    return classes, np.sort(np.concatenate(idx))


# One fixed t-SNE configuration for every panel of every figure, so panels are
# produced under identical conditions (they remain qualitative either way: t-SNE
# preserves local neighbourhoods, not global distances, and coordinates are not
# comparable across separately fitted panels).
TSNE_PARAMS = {"n_components": 2, "random_state": 0, "init": "pca", "perplexity": 30.0}


def _project(X, seed=0):
    """Return {'PCA': xy, 't-SNE': xy}, each fitted JOINTLY to features+prototypes.

    PCA is the primary view (deterministic, linear, globally interpretable);
    t-SNE is supplementary. Every panel uses the same fixed parameters.
    """
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    assert len(X) > 3 * TSNE_PARAMS["perplexity"], \
        f"too few points ({len(X)}) for perplexity {TSNE_PARAMS['perplexity']}"
    out = {}
    out["PCA"] = PCA(n_components=2, random_state=seed).fit_transform(X)
    out["t-SNE"] = TSNE(**TSNE_PARAMS).fit_transform(X)
    return out


def feature_charts():
    cfg = load_config()
    vz = cfg["feature_viz"]
    for ds in cfg["datasets"]:
        classes, idx = viz_selection(ds, vz["n_classes"], vz["class_seed"],
                                     vz["max_per_class"])
        names_all = load_features(ds, "test", "resnet18")["class_names"]
        sel_names = [names_all[c] for c in classes]
        remap = {int(c): j for j, c in enumerate(classes)}

        for enc in encoders_for(ds):
            f = load_features(ds, "test", enc)
            X = F.normalize(f["features"][idx].float(), dim=-1)  # cosine space
            y = np.array([remap[int(v)] for v in f["labels"][idx].numpy()])

            if enc == "clip_rn50":
                text = torch.load(clip_text_path(ds), weights_only=True)
                assert text["class_names"] == names_all, \
                    f"{ds}: text-prototype class order differs from the feature caches"
                protos = text["text_prototypes"][classes].float()
                proto_kind = "text prototypes"
            else:
                ftr = load_features(ds, "train", enc)
                tr_idx = training_indices(ds, "full", 0, ftr["labels"].numpy(),
                                          fingerprint=ftr.get("pool_fingerprint"))
                p = PrototypeClassifier(len(names_all)).fit(
                    ftr["features"][tr_idx], ftr["labels"][tr_idx].long())
                protos = p.prototypes[classes]
                proto_kind = "image prototypes (full training split)"

            stacked = torch.cat([X, protos]).numpy()
            proj = _project(stacked)
            titles = {"PCA": f"PCA (primary) — {encoder_label(enc, short=True)}",
                      "t-SNE": f"t-SNE (supplementary) — {encoder_label(enc, short=True)}"}
            panels = [{"title": titles[method],
                       "xy": xy[:len(X)], "labels": y, "proto_xy": xy[len(X):]}
                      for method, xy in proj.items()]
            print("figure:", feature_projection(
                panels, sel_names,
                f"{dataset_label(ds)} — {vz['n_classes']} classes, test features with "
                f"{proto_kind}\n(L2-normalized features; projections fitted jointly to "
                f"features and prototypes; fixed seeds; qualitative view — t-SNE shows "
                f"local neighbourhoods only)",
                f"features_{ds}_{enc}.png"))


def sample_charts():
    from src.data import load_pool
    from src.visualize import sample_grid

    cfg = load_config()
    vz = cfg["feature_viz"]
    for ds in cfg["datasets"]:
        classes, _ = viz_selection(ds, vz["n_classes"], vz["class_seed"], vz["max_per_class"])
        pool = load_pool(ds, "test", download=False)
        rng = np.random.default_rng(0)
        picks = [rng.choice(np.flatnonzero(pool.labels == c)) for c in classes]
        print("figure:", sample_grid(
            pool, picks, pool.class_names,
            f"{dataset_label(ds)} — one test image from each of the "
            f"{vz['n_classes']} visualization classes", f"samples_{ds}.png",
            n_cols=5))


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    steps = {"acc": acc_charts, "curves": curve_charts, "confusion": confusion_charts,
             "features": feature_charts, "samples": sample_charts}
    for name, fn in steps.items():
        if only and name != only:
            continue
        print(f"--- {name} ---", flush=True)
        fn()


if __name__ == "__main__":
    main()
