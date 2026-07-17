CELLS = [
    ("markdown", """
## 2 · Datasets & fixed episodes

| Dataset | Classes (episodic pool) | Images | Notes |
|---|---|---|---|
| MNIST | 10 (test split) | 10,000 | grayscale → 3-channel |
| CIFAR-10 | 10 (test split) | 10,000 | |
| Mini-ImageNet | 20 R&L **test** classes | 13,000 (650/class) | `timm/mini-imagenet`, re-partitioned by the canonical Ravi & Larochelle 64/16/20 *class* split (`config/mini_imagenet_splits.json`) |

Episode support/query indices were sampled **once** (seed 42) and saved to `results/artifacts/episodes/` (ADR 0002); every classifier — and stages 2–3 later — evaluates on these exact files. Below: episode 0 of the 5-way 5-shot file for each dataset.
"""),
    ("code", """
from IPython.display import Image, display

for ds in ("mnist", "cifar10", "mini_imagenet"):
    display(Image(str(REPO / "results" / "figures" / f"episode_grid_{ds}.png"), width=880))
"""),
]
