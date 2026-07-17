"""Quantitative embedding-quality metrics per dataset × backbone (test features).

t-SNE plots are qualitative; these numbers are the quantitative counterpart:
  - silhouette score (cosine)
  - leave-one-out 1-NN accuracy (cosine)
  - mean within-class / between-class cosine distance ratio (lower = better separated)
Subsampled to 2000 points per (dataset, backbone), seed 0.
Writes results/metrics/embedding_quality.csv and prints a markdown table.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.embeddings import BACKBONES, load_features
from src.evaluation import metrics_dir
from src.utils import load_config

N_SUB = 2000


def metrics_for(feats, labels):
    X = F.normalize(feats, dim=-1)
    D = 1 - X @ X.T  # cosine distance
    same = labels.unsqueeze(0) == labels.unsqueeze(1)
    off_diag = ~torch.eye(len(X), dtype=torch.bool)

    within = D[same & off_diag].mean().item()
    between = D[~same].mean().item()

    nn_idx = (D + torch.eye(len(X)) * 10).argmin(dim=1)  # exclude self
    nn_acc = (labels[nn_idx] == labels).float().mean().item()

    from sklearn.metrics import silhouette_score

    sil = silhouette_score(X.numpy(), labels.numpy(), metric="cosine")
    return {"silhouette": round(float(sil), 3), "nn1_acc": round(nn_acc, 3),
            "within_between_ratio": round(within / between, 3)}


def main():
    cfg = load_config()
    rng = np.random.default_rng(0)
    rows = []
    for ds in cfg["datasets"]:
        for bb in BACKBONES:
            f = load_features(ds, "test", bb)
            n = len(f["labels"])
            keep = torch.from_numpy(rng.choice(n, size=min(N_SUB, n), replace=False))
            m = metrics_for(f["features"][keep].float(), f["labels"][keep])
            rows.append({"dataset": ds, "backbone": bb, **m})
            print(f"{ds} × {bb}: {m}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(metrics_dir() / "embedding_quality.csv", index=False)
    print("\n" + df.to_string(index=False))  # to_markdown needs tabulate (not in venv)


if __name__ == "__main__":
    main()
