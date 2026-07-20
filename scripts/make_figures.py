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


BACKBONE_NAMES = {"clip_vitb32": "CLIP ViT-B/32", "dinov2_vits14": "DINOv2 ViT-S/14",
                  "resnet50": "ResNet-50"}


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
            panels.append({"title": BACKBONE_NAMES[bb], "xy": xy_all[:len(X)], "labels": y,
                           "class_names": f["class_names"],
                           "proto_xy": xy_all[len(X):]})
        from src.visualize import dataset_label

        note = " (colors recycle across the 20 classes)" if ds == "mini_imagenet" else ""
        p = embedding_scatter_panels(
            panels, f"{dataset_label(ds)}: t-SNE of frozen test embeddings{note}",
            f"tsne_{ds}.png")
        print(f"figure: {p}")


def result_curves():
    from src.visualize import ABLATION_ONLY

    ep = pd.read_csv(metrics_dir() / "episodic.csv")
    si = pd.read_csv(metrics_dir() / "simple.csv")
    ep = ep[~ep["classifier"].str.startswith(ABLATION_ONLY)]
    si = si[~si["classifier"].str.startswith(ABLATION_ONLY)]
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


def kmeans_ncenters():
    """n_centers ∈ {1,2,3} multi-prototype comparison, episodic 5-shot, on each
    dataset's selected prototype backbone (n=1 = the plain cosine prototype)."""
    import json

    import matplotlib.pyplot as plt

    from src.visualize import DPI, dataset_label, figures_dir

    with open(Path(__file__).resolve().parent.parent / "results" / "artifacts"
              / "best_baselines.json") as f:
        best = json.load(f)["selection"]
    ep = pd.read_csv(metrics_dir() / "episodic.csv")
    datasets = ["mnist", "cifar10", "mini_imagenet"]
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    import numpy as np

    x = np.arange(len(datasets))
    w = 0.26
    colors = {1: "#0173B2", 2: "#CC78BC", 3: "#ECE133"}
    for i, n in enumerate((1, 2, 3)):
        vals, errs = [], []
        for ds in datasets:
            bb = best[ds]["prototype"]["backbone"]
            name = f"proto_cos__{bb}" if n == 1 else f"kmeans{n}_cos__{bb}"
            r = ep[(ep["dataset"] == ds) & (ep["k_shot"] == 5)
                   & (ep["classifier"] == name)].iloc[0]
            vals.append(100 * r["acc"]); errs.append(100 * r["ci95"])
        ax.bar(x + (i - 1) * w, vals, w, yerr=errs, capsize=4,
               label=f"$n$ = {n}" + (" (= prototype)" if n == 1 else ""),
               color=colors[n], edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels([dataset_label(d) for d in datasets])
    ax.set_ylabel("accuracy (%)"); ax.set_ylim(bottom=70)
    ax.set_title("Multi-prototype ablation: $n$ k-means centers per class\n"
                 "(5-way 5-shot episodes, each dataset's selected prototype backbone)")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)
    p = figures_dir() / "kmeans_ncenters.png"
    fig.savefig(p, dpi=DPI, bbox_inches="tight"); plt.close(fig)
    print(f"figure: {p}")


def confusion_and_failures():
    """Simple-protocol confusion matrix + failure gallery (cifar10, prototype-cosine, K=10)."""
    from sklearn.metrics import confusion_matrix

    from src.classifiers import PrototypeClassifier
    from src.data import simple_path
    from src.evaluation import run_simple

    from src.utils import get_device

    cfg = load_config()
    for ds in ("mnist", "cifar10"):
        train_f = load_features(ds, "train", PRIMARY)
        test_f = load_features(ds, "test", PRIMARY)
        support = torch.load(simple_path(ds, 10, cfg["simple"]["seeds"][0]), weights_only=True)
        clf = PrototypeClassifier("cosine")
        acc, pred, true = run_simple(train_f, test_f, support, clf)
        cm = confusion_matrix(true, pred)
        from src.visualize import dataset_label

        print("figure:", confusion(cm, test_f["class_names"],
                                   f"{dataset_label(ds)}: prototype (cosine), 10-shot, "
                                   f"seed 0 — acc {100 * acc:.1f}%",
                                   f"confusion_{ds}_proto10s.png"))
        # Hardest misclassifications = largest (predicted - true) score margin.
        device = get_device()
        Xs = train_f["features"][support["support_idx"]].unsqueeze(0).to(device)
        ys = train_f["labels"][support["support_idx"]].unsqueeze(0).long().to(device)
        Xq = test_f["features"].unsqueeze(0).to(device)
        scores = clf.predict(Xs, ys, Xq).squeeze(0).cpu().numpy()
        wrong = np.flatnonzero(pred != true)
        margin = scores[wrong, pred[wrong]] - scores[wrong, true[wrong]]
        order = wrong[np.argsort(-margin)]
        # Top-confidence error per (true, pred) pair first — shows distinct failure
        # modes rather than 12 copies of the single worst one; fill up by margin.
        pick, seen_pairs = [], set()
        for i in order:
            pair = (int(true[i]), int(pred[i]))
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                pick.append(i)
        pick.extend(i for i in order if i not in pick)
        pick = pick[:12]
        pool = load_pool(ds, "test")
        items = [(pool.get_image(int(i)), test_f["class_names"][int(true[i])],
                  test_f["class_names"][int(pred[i])]) for i in pick]
        print("figure:", failure_gallery(
            items, f"{dataset_label(ds)}: prototype 10-shot — most confident "
                   f"misclassification per (true, predicted) pair",
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
        # Lowest top-1-vs-top-2 margin = the most uncertain predictions; saturated
        # 100%-correct examples carry no information.
        top2 = np.sort(probs, axis=1)[:, -2:]
        pick = np.argsort(top2[:, 1] - top2[:, 0])[:5]
        images = [pool.get_image(int(i)) for i in pick]
        true_names = [test_f["class_names"][int(test_f["labels"][i])] for i in pick]
        from src.visualize import dataset_label

        print("figure:", clip_qualitative(images, true_names, probs[pick],
                                          test_f["class_names"],
                                          f"{dataset_label(ds)}: least confident CLIP "
                                          f"zero-shot predictions (prompt ensemble)",
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
    def arch():
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import make_architecture_figs as af

        af.baselines(); af.stage2_advised()

    steps = {"grids": episode_grids, "tsne": lambda: tsne_panels(["mnist", "cifar10", "mini_imagenet"]),
             "curves": result_curves, "kmeans": kmeans_ncenters,
             "confusion": confusion_and_failures, "clip": clip_zeroshot_panel,
             "arch": arch}
    for name, fn in steps.items():
        if only and name != only:
            continue
        print(f"--- {name} ---", flush=True)
        fn()


if __name__ == "__main__":
    main()
