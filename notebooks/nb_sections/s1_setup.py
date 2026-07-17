CELLS = [
    ("markdown", "## 1 · Environment & configuration"),
    ("code", """
import sys
from pathlib import Path

REPO = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(REPO))

import json

import pandas as pd

from src.utils import load_config

pd.set_option("display.precision", 2)

with open(REPO / "results" / "runtime_summary.json") as f:
    runtime = json.load(f)
print("Runtime:", json.dumps(runtime, indent=2))
cfg = load_config()
print("\\nEpisodic protocol:", cfg["episodic"])
print("Simple protocol:  ", cfg["simple"])
"""),
    ("markdown", """
All experiments ran on the machine above (AMD RX 7900 XTX, torch+ROCm). Backbones — all **frozen**, used only as feature extractors (ADR 0001):

| Backbone | Source | Dim |
|---|---|---|
| CLIP ViT-B/32 (primary) | official OpenAI `clip` | 512 |
| DINOv2 ViT-S/14 | `facebook/dinov2-small` | 384 |
| ResNet-50 (ImageNet) | torchvision, penultimate layer | 2048 |
"""),
]
