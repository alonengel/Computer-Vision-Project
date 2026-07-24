CELLS = [
    ("markdown", "## 1 · Environment and configuration"),
    ("code", """
import sys
from pathlib import Path

REPO = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(REPO))

import json

import pandas as pd
from IPython.display import Image, display

from src.utils import load_config

pd.set_option("display.precision", 2)
pd.set_option("display.max_rows", 200)

with open(REPO / "results" / "runtime_summary.json") as f:
    runtime = json.load(f)
print("Runtime:", json.dumps(runtime, indent=2))

cfg = load_config()
print("\\nTraining-set sizes K:", cfg["shots"])
print("Subset seeds (5/10-shot):", cfg["subset_seeds"],
      "| initialization seeds (full):", cfg["init_seeds"])
print("Linear probe:", json.dumps(cfg["linear_probe"], indent=2))
"""),
    ("markdown", """
All experiments ran on the machine above. Every encoder is **frozen** and used with the preprocessing associated with its own checkpoint:

| Encoder | Source | Representation | Used on |
|---|---|---|---|
| ResNet-18 (ImageNet-1K) | torchvision `ResNet18_Weights.IMAGENET1K_V1` | 512-d, before the final classification layer | all datasets |
| DINOv2 ViT-S/14 | `facebook/dinov2-small` | final class token (384-d) | FGVC-Aircraft |
| CLIP RN50 | official OpenAI `clip` | image encoder (1024-d) + text encoder | zero-shot branch only |

Train, validation and test features are extracted **once** per (dataset, encoder) and cached; all classifier training and evaluation runs on those caches.
"""),
]
