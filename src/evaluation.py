"""Evaluation helpers: top-1 accuracy on the official test split and run aggregation.

Per the spec, every reported number is top-1 accuracy on the *complete official
test split*. 5-shot and 10-shot settings are summarized as mean +/- standard
deviation over the 3 runs; the full linear probe over its 3 initialization seeds;
the full image-prototype result and the zero-shot CLIP result are single runs.
"""
import json

import numpy as np
import torch

from .utils import load_config, repo_path


def metrics_dir(sub=None):
    d = repo_path(load_config()["paths"]["metrics_dir"])
    if sub:
        d = d / sub
    d.mkdir(parents=True, exist_ok=True)
    return d


def artifacts_dir(sub=None):
    d = repo_path(load_config()["paths"]["artifacts_dir"])
    if sub:
        d = d / sub
    d.mkdir(parents=True, exist_ok=True)
    return d


def top1(pred, target):
    pred = torch.as_tensor(pred).cpu()
    target = torch.as_tensor(target).cpu()
    return float((pred == target).float().mean())


def summarize(runs):
    """mean and (sample) std over run accuracies; std is 0.0 for a single run."""
    a = np.asarray(runs, dtype=np.float64)
    return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0


def row_normalized_confusion(pred, target, n_classes):
    """Confusion matrix normalized per true class (rows sum to 1)."""
    cm = np.zeros((n_classes, n_classes), dtype=np.float64)
    pred = np.asarray(torch.as_tensor(pred).cpu())
    target = np.asarray(torch.as_tensor(target).cpu())
    for t, p in zip(target, pred):
        cm[t, p] += 1
    counts = cm.sum(axis=1, keepdims=True)
    return np.divide(cm, counts, out=np.zeros_like(cm), where=counts > 0)


def save_table(rows, name):
    """Write a list-of-dicts table to CSV + JSON under results/metrics/."""
    import pandas as pd

    df = pd.DataFrame(rows)
    df.to_csv(metrics_dir() / f"{name}.csv", index=False)
    with open(metrics_dir() / f"{name}.json", "w") as f:
        json.dump(rows, f, indent=2)
    return df


def save_raw(name, arr):
    np.save(metrics_dir("raw") / f"{name}.npy", np.asarray(arr))


def save_curves(name, history):
    with open(artifacts_dir("curves") / f"{name}.json", "w") as f:
        json.dump(history, f)


def save_predictions(name, pred, target):
    np.savez(artifacts_dir("predictions") / f"{name}.npz",
             pred=np.asarray(torch.as_tensor(pred).cpu()),
             target=np.asarray(torch.as_tensor(target).cpu()))


def load_predictions(name):
    d = np.load(artifacts_dir("predictions") / f"{name}.npz")
    return d["pred"], d["target"]
