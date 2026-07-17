"""Dataset pools (MNIST, CIFAR-10, Mini-ImageNet) and episode/support-set sampling.

Every dataset is exposed as a `Pool`: a flat indexable collection of (PIL RGB image,
int label) with class names. Episode and support-set indices are sampled once per
(dataset, protocol, seed) and saved under results/artifacts/episodes/ so that all
classifiers — and the Flow Matching stages later — evaluate on identical data (ADR 0002).
"""
import json

import numpy as np
import torch

from .utils import load_config, repo_path

# Keep the HF datasets cache on D: next to the repo (config data root).
import os
os.environ.setdefault("HF_HOME", str(repo_path("data", "hf")))

MINI_IMAGENET_HF_ID = "timm/mini-imagenet"
# Pinned revision: upstream changes must never silently reorder the pool (ADR 0002).
MINI_IMAGENET_REVISION = "bd8779f9d33c061ea6e75fdd3bce4e43dd679060"


def labels_fingerprint(labels):
    """Stable fingerprint of a pool's label array; stored in episode files and
    checked against feature caches at evaluation time to catch misalignment."""
    import hashlib

    arr = np.ascontiguousarray(np.asarray(labels, dtype=np.int64))
    return f"{len(arr)}:{hashlib.sha256(arr.tobytes()).hexdigest()[:16]}"


class Pool:
    """Uniform dataset view: get_image(i) -> PIL RGB, labels -> np.int64 array."""

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


def _torchvision_pool(dataset_name, split):
    from torchvision import datasets as tvd

    root = str(repo_path(load_config()["paths"]["data_root"]))
    train = split == "train"
    if dataset_name == "mnist":
        ds = tvd.MNIST(root, train=train, download=True)
        class_names = [str(d) for d in range(10)]
        get_image = lambda i: ds[i][0].convert("RGB")  # 1-channel -> 3-channel
    elif dataset_name == "cifar10":
        ds = tvd.CIFAR10(root, train=train, download=True)
        class_names = list(ds.classes)
        get_image = lambda i: ds[i][0].convert("RGB")
    else:
        raise ValueError(dataset_name)
    labels = [int(ds[i][1]) for i in range(len(ds))]
    return Pool(dataset_name, split, get_image, labels, class_names)


def load_mini_imagenet_splits():
    """Canonical Ravi & Larochelle 64/16/20 class split: {split: {wnid: readable name}}."""
    with open(repo_path("config", "mini_imagenet_splits.json"), encoding="utf-8-sig") as f:
        return json.load(f)


def _mini_imagenet_pool(split):
    """Mini-ImageNet pool for one R&L class split ('train'/'val'/'test').

    timm/mini-imagenet ships the 100 classes partitioned by *image* (50k/10k/5k);
    we merge all images and re-partition by *class* per the canonical few-shot split.
    Labels are re-indexed 0..C-1 in sorted-wnid order.
    """
    from datasets import concatenate_datasets, load_dataset

    class_split = load_mini_imagenet_splits()[split]
    wnids = sorted(class_split)
    class_names = [class_split[w] for w in wnids]

    hf = load_dataset(MINI_IMAGENET_HF_ID, revision=MINI_IMAGENET_REVISION)
    merged = concatenate_datasets([hf[s] for s in hf])
    hf_names = merged.features["label"].names
    assert len(hf_names) == 100, f"expected 100 classes, got {len(hf_names)}"
    wanted = {hf_names.index(w): new for new, w in enumerate(wnids)}

    hf_labels = np.asarray(merged["label"])
    keep = np.flatnonzero(np.isin(hf_labels, list(wanted)))
    merged = merged.select(keep)
    labels = [wanted[int(l)] for l in hf_labels[keep]]

    get_image = lambda i: merged[i]["image"].convert("RGB")
    return Pool("mini_imagenet", split, get_image, labels, class_names)


def load_pool(dataset_name, split):
    """split: 'train'/'test' for mnist & cifar10; 'train'/'val'/'test' (class splits) for mini_imagenet."""
    if dataset_name in ("mnist", "cifar10"):
        return _torchvision_pool(dataset_name, split)
    if dataset_name == "mini_imagenet":
        return _mini_imagenet_pool(split)
    raise ValueError(dataset_name)


# --------------------------------------------------------------------------- #
# Episode / support-set sampling (saved to disk, ADR 0002)
# --------------------------------------------------------------------------- #

def episodes_dir():
    d = repo_path(load_config()["paths"]["artifacts_dir"], "episodes")
    d.mkdir(parents=True, exist_ok=True)
    return d


def episodic_path(dataset, n_way, k_shot, seed):
    return episodes_dir() / f"{dataset}_ep_{n_way}w{k_shot}s_seed{seed}.pt"


def simple_path(dataset, k_shot, seed):
    return episodes_dir() / f"{dataset}_simple_{k_shot}s_seed{seed}.pt"


def selection_path(dataset, n_way, k_shot, seed):
    """Validation-selection episodes (disjoint from test episodes by pool and seed):
    MNIST/CIFAR-10 sampled from the train split, Mini-ImageNet from the R&L val classes."""
    return episodes_dir() / f"{dataset}_valsel_{n_way}w{k_shot}s_seed{seed}.pt"


def ensure_selection_episodes(dataset, labels):
    """Create the validation-selection episode files from a label array if missing.
    `labels` must come from the selection pool (train split / val classes)."""
    cfg = load_config()
    ep, sel = cfg["episodic"], cfg["selection"]
    paths = []
    for k in ep["shots"]:
        path = selection_path(dataset, ep["n_way"], k, sel["seed"])
        if not path.exists():
            torch.save(sample_episodes(labels, ep["n_way"], k, ep["n_query"],
                                       sel["n_episodes"], sel["seed"]), path)
        paths.append(path)
    return paths


def sample_episodes(labels, n_way, k_shot, n_query, n_episodes, seed):
    """Disjoint support/query index tensors for n_episodes N-way K-shot episodes."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    classes = np.unique(labels)
    by_class = {c: np.flatnonzero(labels == c) for c in classes}
    ep_classes = np.empty((n_episodes, n_way), dtype=np.int64)
    support = np.empty((n_episodes, n_way * k_shot), dtype=np.int64)
    query = np.empty((n_episodes, n_way * n_query), dtype=np.int64)
    for e in range(n_episodes):
        cls = rng.choice(classes, size=n_way, replace=False)
        ep_classes[e] = cls
        for j, c in enumerate(cls):
            pick = rng.choice(by_class[c], size=k_shot + n_query, replace=False)
            support[e, j * k_shot:(j + 1) * k_shot] = pick[:k_shot]
            query[e, j * n_query:(j + 1) * n_query] = pick[k_shot:]
    return {
        "classes": torch.from_numpy(ep_classes),
        "support_idx": torch.from_numpy(support),
        "query_idx": torch.from_numpy(query),
        "n_way": n_way, "k_shot": k_shot, "n_query": n_query, "seed": seed,
        "pool_fingerprint": labels_fingerprint(labels),
    }


def sample_support(labels, k_shot, seed):
    """All-classes support set (K per class) for the simple protocol; eval uses the full test split."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    idx = np.concatenate([
        rng.choice(np.flatnonzero(labels == c), size=k_shot, replace=False)
        for c in np.unique(labels)
    ])
    return {"support_idx": torch.from_numpy(idx), "k_shot": k_shot, "seed": seed,
            "pool_fingerprint": labels_fingerprint(labels)}


def build_all_episode_files(pools_by_dataset):
    """Generate every episode/support index file that does not exist yet.

    pools_by_dataset: {dataset: {"episodic": Pool, "simple_train": Pool or None}}.
    Episodic pools: mnist/cifar10 test split, mini_imagenet R&L test classes.
    Simple protocol (mnist/cifar10 only): support drawn from the train split.
    """
    cfg = load_config()
    ep, si = cfg["episodic"], cfg["simple"]
    created = []
    for ds, pools in pools_by_dataset.items():
        pool = pools["episodic"]
        for k in ep["shots"]:
            path = episodic_path(ds, ep["n_way"], k, ep["seed"])
            if not path.exists():
                data = sample_episodes(pool.labels, ep["n_way"], k, ep["n_query"],
                                       ep["n_episodes"], ep["seed"])
                torch.save(data, path)
                created.append(path.name)
        train_pool = pools.get("simple_train")
        if train_pool is None:
            continue
        for k in si["shots"]:
            for seed in si["seeds"]:
                path = simple_path(ds, k, seed)
                if not path.exists():
                    torch.save(sample_support(train_pool.labels, k, seed), path)
                    created.append(path.name)
    return created
