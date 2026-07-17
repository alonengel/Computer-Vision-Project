"""Evaluation runners: episodic (mean ± 95% CI) and simple all-classes K-shot (mean ± std).

Both runners consume the *saved* episode/support index files (ADR 0002) and cached
embeddings — never raw images — so every classifier and every stage sees identical data.
Raw per-episode / per-seed accuracies are saved under results/metrics/raw/ so the
headline numbers can be independently re-derived (scripts/repro_check.py).
"""
import json

import numpy as np
import torch

from .utils import get_device, load_config, repo_path


def metrics_dir(sub=None):
    d = repo_path(load_config()["paths"]["metrics_dir"])
    if sub:
        d = d / sub
    d.mkdir(parents=True, exist_ok=True)
    return d


def ci95(per_episode_acc):
    """Half-width of the 95% CI of the mean over episode accuracies."""
    a = np.asarray(per_episode_acc)
    return 1.96 * a.std(ddof=1) / np.sqrt(len(a))


def episode_tensors(features, episode, device=None):
    """Build batched support/query tensors from cached features + a saved episode file.

    Episode class ids are remapped to 0..N-1 (position in episode['classes']).
    Returns Xs [B,S,D], ys [B,S], Xq [B,Q,D], yq [B,Q], classes [B,N] (original ids).
    """
    from .data import labels_fingerprint

    device = device or get_device()
    feats = features["features"].to(device)
    labels = features["labels"]
    fp = episode.get("pool_fingerprint")
    if fp is not None:
        actual = labels_fingerprint(labels.numpy())
        assert actual == fp, (f"episode/feature pool mismatch: episode was sampled from "
                              f"pool {fp} but features have {actual} — regenerate one of them")
    sup, qry, cls = episode["support_idx"], episode["query_idx"], episode["classes"]
    Xs, Xq = feats[sup], feats[qry]

    def remap(idx):
        orig = labels[idx]  # [B, S]
        hit = orig.unsqueeze(-1) == cls.unsqueeze(1)
        assert bool(hit.any(-1).all()), \
            "episode sample label not in episode classes — episode/feature mismatch"
        return hit.float().argmax(-1)

    return Xs, remap(sup).long().to(device), Xq, remap(qry).long().to(device), cls.to(device)


def run_episodic(features, episode, classifier, class_subset_aware=False):
    """Per-episode accuracies [n_episodes] for one classifier on one episode file."""
    Xs, ys, Xq, yq, cls = episode_tensors(features, episode)
    if class_subset_aware:  # zero-shot CLIP: score only the episode's classes
        scores = classifier.predict(Xs, ys, Xq, class_subset=cls)
    else:
        scores = classifier.predict(Xs, ys, Xq)
    pred = scores.argmax(-1)
    return (pred == yq).float().mean(dim=1).cpu().numpy()


def run_simple(train_features, test_features, support, classifier):
    """Simple protocol: support = saved indices into the train split, eval on the full test split.

    All classes are present, so zero-shot CLIP needs no class subset here.
    Returns (accuracy, predictions, true_labels) — predictions kept for confusion matrices.
    """
    from .data import labels_fingerprint

    device = get_device()
    fp = support.get("pool_fingerprint")
    if fp is not None:
        actual = labels_fingerprint(train_features["labels"].numpy())
        assert actual == fp, f"support/feature pool mismatch: {fp} vs {actual}"
    Xs = train_features["features"][support["support_idx"]].unsqueeze(0).to(device)
    ys = train_features["labels"][support["support_idx"]].unsqueeze(0).long().to(device)
    Xq = test_features["features"].unsqueeze(0).to(device)
    yq = test_features["labels"].long().to(device)
    scores = classifier.predict(Xs, ys, Xq)
    pred = scores.argmax(-1).squeeze(0)
    return (pred == yq).float().mean().item(), pred.cpu().numpy(), yq.cpu().numpy()


def save_raw(name, arr):
    np.save(metrics_dir("raw") / f"{name}.npy", np.asarray(arr))


def save_table(rows, name):
    """Write a list-of-dicts result table to CSV + JSON under results/metrics/."""
    import pandas as pd

    df = pd.DataFrame(rows)
    df.to_csv(metrics_dir() / f"{name}.csv", index=False)
    with open(metrics_dir() / f"{name}.json", "w") as f:
        json.dump(rows, f, indent=2)
    return df


def save_prototype_artifacts(features, dataset, split, backbone):
    """Class-mean prototypes from a full feature split — stage-2 FM targets."""
    feats, labels = features["features"], features["labels"]
    classes = torch.arange(int(labels.max()) + 1)
    proto = torch.stack([feats[labels == c].mean(0) for c in classes])
    d = repo_path(load_config()["paths"]["artifacts_dir"])
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"prototypes_{dataset}_{split}_{backbone}.pt"
    torch.save({"prototypes": proto, "class_names": features["class_names"]}, path)
    return path
