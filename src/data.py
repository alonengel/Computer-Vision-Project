"""Datasets and training-subset sampling for Stage 1 (spec: `_docs/stage_1.pdf`).

Three datasets, all classes, **official** train / validation / test splits:
  dtd            — torchvision DTD, official partition 1
  fgvc_aircraft  — torchvision FGVCAircraft, `variant` annotation level
  flowers102     — torchvision Flowers102

Training and validation splits are never merged. The validation split is used for
model selection (linear-probe checkpointing); the test split is used only for the
final evaluation.

For methods that use labeled training examples we evaluate K in {5, 10, full}:
K = number of training images per class. For K in {5, 10} a *balanced* subset is
sampled from the official training split with seeds {0, 1, 2}; the subset indices
are saved to results/artifacts/subsets/ so every encoder and every head sees the
identical training images. "full" is the complete official training split.
"""
import hashlib
import json
import os

import numpy as np
import torch

from .utils import load_config, repo_path

# Keep any HF cache next to the repo on D:.
os.environ.setdefault("HF_HOME", str(repo_path("data", "hf")))

SPLITS = ("train", "val", "test")


class Pool:
    """Uniform view of one dataset split: get_image(i) -> PIL RGB, labels, class_names."""

    def __init__(self, name, split, get_image, labels, class_names):
        self.name = name
        self.split = split
        self._get_image = get_image
        self.labels = np.asarray(labels, dtype=np.int64)
        self.class_names = list(class_names)

    def __len__(self):
        return len(self.labels)

    def get_image(self, i):
        return self._get_image(int(i))


def labels_fingerprint(labels):
    """Stable fingerprint of a split's label array; stored in subset files and
    checked against feature caches so a dataset change fails loudly."""
    arr = np.ascontiguousarray(np.asarray(labels, dtype=np.int64))
    return f"{len(arr)}:{hashlib.sha256(arr.tobytes()).hexdigest()[:16]}"


def _flowers_class_names():
    with open(repo_path("config", "flowers102_classes.json"), encoding="utf-8-sig") as f:
        names = json.load(f)["classes"]
    assert len(names) == 102, f"expected 102 Flowers-102 class names, got {len(names)}"
    return names


def load_pool(dataset, split, download=True):
    """Official split of one dataset as a Pool. split in {'train','val','test'}."""
    from torchvision import datasets as tvd

    assert split in SPLITS, split
    cfg = load_config()["datasets"][dataset]
    root = str(repo_path(load_config()["paths"]["data_root"]))

    if dataset == "dtd":
        ds = tvd.DTD(root, split=split, partition=cfg["partition"], download=download)
        class_names = [c.replace("_", " ") for c in ds.classes]
        labels = list(ds._labels)
    elif dataset == "fgvc_aircraft":
        ds = tvd.FGVCAircraft(root, split=split,
                              annotation_level=cfg["annotation_level"], download=download)
        class_names = list(ds.classes)
        labels = list(ds._labels)
    elif dataset == "flowers102":
        ds = tvd.Flowers102(root, split=split, download=download)
        class_names = _flowers_class_names()
        labels = list(ds._labels)
    else:
        raise ValueError(dataset)

    assert len(class_names) == cfg["n_classes"], \
        f"{dataset}/{split}: {len(class_names)} classes, expected {cfg['n_classes']}"

    def get_image(i):
        return ds[i][0].convert("RGB")

    return Pool(dataset, split, get_image, labels, class_names)


# --------------------------------------------------------------------------- #
# Balanced K-shot training subsets (sampled from the official train split only)
# --------------------------------------------------------------------------- #

def subsets_dir():
    d = repo_path(load_config()["paths"]["artifacts_dir"], "subsets")
    d.mkdir(parents=True, exist_ok=True)
    return d


def subset_path(dataset, k_shot, seed):
    return subsets_dir() / f"{dataset}_k{k_shot}_seed{seed}.pt"


def sample_balanced_subset(labels, k_shot, seed):
    """K training indices per class, sampled without replacement from the train split.

    Classes with fewer than K available images contribute all of theirs (recorded
    in `short_classes` so the report can state it rather than silently truncate).
    """
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    idx, short = [], {}
    for c in np.unique(labels):
        pool = np.flatnonzero(labels == c)
        take = min(k_shot, len(pool))
        if take < k_shot:
            short[int(c)] = int(len(pool))
        idx.append(rng.choice(pool, size=take, replace=False))
    idx = np.sort(np.concatenate(idx))
    return {"indices": torch.from_numpy(idx), "k_shot": k_shot, "seed": seed,
            "short_classes": short, "pool_fingerprint": labels_fingerprint(labels)}


def training_indices(dataset, k_shot, seed, train_labels):
    """Indices into the official train split for one (K, seed) setting.

    k_shot == 'full' uses the complete official training split (seed irrelevant).
    Otherwise the saved balanced subset is loaded, or created on first use.
    """
    if k_shot == "full":
        return torch.arange(len(train_labels))
    path = subset_path(dataset, k_shot, seed)
    if not path.exists():
        torch.save(sample_balanced_subset(train_labels, k_shot, seed), path)
    d = torch.load(path, weights_only=True)
    actual = labels_fingerprint(train_labels)
    assert d["pool_fingerprint"] == actual, (
        f"{path.name}: subset was sampled from train pool {d['pool_fingerprint']} "
        f"but the current train split is {actual} — regenerate the subsets")
    return d["indices"]


def build_all_subsets(train_labels_by_dataset):
    """Create every K-shot subset file that does not exist yet."""
    cfg = load_config()
    created = []
    for ds, labels in train_labels_by_dataset.items():
        for k in cfg["shots"]:
            if k == "full":
                continue
            for seed in cfg["subset_seeds"]:
                path = subset_path(ds, k, seed)
                if not path.exists():
                    torch.save(sample_balanced_subset(labels, k, seed), path)
                    created.append(path.name)
    return created
