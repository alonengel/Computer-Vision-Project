"""Regenerate every Stage 1 figure from cached features, saved episodes and metrics tables."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch

from src.classifiers import ZeroShotCLIP
from src.data import episodic_path, load_pool
from src.embeddings import clip_text_path, load_features
from src.evaluation import metrics_dir
from src.utils import load_config, set_seed
from src.visualize import (accuracy_vs_k, clip_qualitative, confusion,
                           embedding_scatter_panels, episode_grid, failure_gallery,
                           grouped_bars)

BACKBONES = ["clip_vitb32", "dinov2_vits14", "resnet50"]
PRIMARY = "clip_vitb32"


def tsne_panels(datasets, n_per_class=150):
    from sklearn.manifold import TSNE

    for ds in datasets:
        panels = []
        for bb in BACKBONES:
            f = load_features(ds, "test", bb)
            feats, labels = f["features"].numpy(), f["labels"].numpy()
            rng = np.random.default_rng(0)
            keep = np.concatenate([
                rng.choice(np.flatnonzero(labels == c),
                           size=min(n_per_class, (labels == c).sum()), replace=False)
                for c in np.unique(labels)])
            X, y = feats[keep], labels[keep]
            proto = np.stack([X[y == c].mean(0) for c in np.unique(y)])
            xy_all = TSNE(n_components=2, random_state=0, init="pca",
                          perplexity=30).fit_transform(np.vstack([X, proto]))
            panels.append({"title": bb, "xy": xy_all[:len(X)], "labels": y,
                           "class_names": f["class_names"],
                           "proto_xy": xy_all[len(X):]})
        p = embedding_scatter_panels(
            panels, f"{ds}: t-SNE of frozen test embeddings (black star = class prototype)",
            f"tsne_{ds}.png")
        print(f"figure: {p}")


def result_curves():
    ep = pd.read_csv(metrics_dir() / "episodic.csv")
    si = pd.read_csv(metrics_dir() / "simple.csv")
    for ds, g in ep.groupby("dataset"):
        d = g.rename(columns={"ci95": "err"})
        print("figure:", accuracy_vs_k(d, ds, name=f"acc_vs_k_episodic_{ds}.png"))
    for ds, g in si.groupby("dataset"):
        d = g.rename(columns={"std": "err"})
        print("figure:", accuracy_vs_k(d, ds, name=f"acc_vs_k_simple_{ds}.png"))

    k5 = ep[ep["k_shot"] == 5].rename(columns={"ci95": "err"})
    main = k5[k5["classifier"].str.contains(r"clip_vitb32")]
    print("figure:", grouped_bars(main, "5-way 5-shot episodic accuracy ± 95% CI "
                                        "(CLIP ViT-B/32 embeddings)", "bars_episodic_5shot.png"))
    proto = k5[k5["classifier"].str.startswith("proto_cos")]
    print("figure:", grouped_bars(proto, "Backbone comparison — prototype (cosine), "
                                         "5-way 5-shot ± 95% CI", "bars_backbones.png"))


def confusion_and_failures():
    """Simple-protocol confusion matrix + failure gallery (cifar10, prototype-cosine, K=10)."""
    from sklearn.metrics import confusion_matrix

    from src.classifiers import PrototypeClassifier
    from src.data import simple_path
    from src.evaluation import run_simple

    cfg = load_config()
    for ds in ("mnist", "cifar10"):
        train_f = load_features(ds, "train", PRIMARY)
        test_f = load_features(ds, "test", PRIMARY)
        support = torch.load(simple_path(ds, 10, cfg["simple"]["seeds"][0]), weights_only=True)
        acc, pred, true = run_simple(train_f, test_f, support, PrototypeClassifier("cosine"))
        cm = confusion_matrix(true, pred)
        print("figure:", confusion(cm, test_f["class_names"],
                                   f"{ds}: prototype (cosine), 10-shot — acc {100 * acc:.1f}%",
                                   f"confusion_{ds}_proto10s.png"))
        pool = load_pool(ds, "test")
        wrong = np.flatnonzero(pred != true)
        scores = None  # margin-based hardest: use distance of top prediction
        pick = wrong[:12]
        items = [(pool.get_image(int(i)), test_f["class_names"][int(true[i])],
                  test_f["class_names"][int(pred[i])]) for i in pick]
        print("figure:", failure_gallery(items, f"{ds}: prototype 10-shot misclassifications",
                                         f"failures_{ds}.png"))


def clip_zeroshot_panel():
    set_seed(0)
    for ds in ("cifar10", "mini_imagenet"):
        test_f = load_features(ds, "test", PRIMARY)
        text = torch.load(clip_text_path(ds), weights_only=True)
        clf = ZeroShotCLIP(text["ensemble"])
        Xq = test_f["features"].unsqueeze(0)
        probs = (100.0 * clf.predict(None, None, Xq)).softmax(-1).squeeze(0).numpy()
        pool = load_pool(ds, "test")
        rng = np.random.default_rng(1)
        pick = rng.choice(len(pool), size=5, replace=False)
        images = [pool.get_image(int(i)) for i in pick]
        true_names = [test_f["class_names"][int(test_f["labels"][i])] for i in pick]
        print("figure:", clip_qualitative(images, true_names, probs[pick],
                                          test_f["class_names"],
                                          f"{ds}: CLIP zero-shot predictions (prompt ensemble)",
                                          f"clip_zeroshot_{ds}.png"))


def episode_grids():
    cfg = load_config()["episodic"]
    for ds in ("mnist", "cifar10", "mini_imagenet"):
        pool = load_pool(ds, "test")
        episode = torch.load(episodic_path(ds, cfg["n_way"], max(cfg["shots"]), cfg["seed"]),
                             weights_only=True)
        print("figure:", episode_grid(pool, episode))


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    steps = {"grids": episode_grids, "tsne": lambda: tsne_panels(["mnist", "cifar10", "mini_imagenet"]),
             "curves": result_curves, "confusion": confusion_and_failures,
             "clip": clip_zeroshot_panel}
    for name, fn in steps.items():
        if only and name != only:
            continue
        print(f"--- {name} ---", flush=True)
        fn()


if __name__ == "__main__":
    main()
