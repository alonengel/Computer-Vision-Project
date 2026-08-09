CELLS = [
    ("markdown", "## 1 · Environment and configuration"),
    ("code", """
import sys
from pathlib import Path

REPO = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(REPO))

import json

import numpy as np
import pandas as pd
from IPython.display import Image, Markdown, display

from src.utils import load_config

pd.set_option("display.precision", 2)
pd.set_option("display.max_rows", 200)

with open(REPO / "results" / "runtime_summary.json") as f:
    runtime = json.load(f)
print("Runtime:", json.dumps(runtime, indent=2))

cfg = load_config()
s2 = cfg["stage2"]
print("\\nStage-2 grid — datasets:", s2["datasets"], "| T values:", s2["T_values"])
print("Branches:", json.dumps(s2["branches"], indent=2))
print("Velocity network:", json.dumps(s2["velocity_net"], indent=2))
print("Training:", json.dumps(s2["training"], indent=2))
"""),
    ("markdown", """
**Hyperparameter provenance.** The velocity network is the specification's suggestion (2 hidden layers of width 512, SiLU, scalar $t$ concatenated to the input feature, output dimension = feature dimension). The training configuration — AdamW, lr $10^{-3}$, weight decay $10^{-4}$, batch 64, 200 epochs — **reuses the Stage-1 linear-probe recipe verbatim**, fixed a priori (ADR 0007 §7); the spec asks for stable training and a fair comparison, not hyperparameter search. One deliberate difference from the probe: **no validation-based checkpointing — the final-epoch model is used.** The spec specifies no selection rule for FM; a per-$T$ validation selection would treat standard FM (whose training is $T$-independent) and rolled-out FM asymmetrically; and the fixed budget keeps the standard-vs-rolled comparison clean. Training-loss curves for **every** model are saved and checked in §4 against a stability criterion that was also fixed a priori.

Everything runs on the Stage-1 feature caches — no images are touched in Stage 2.
"""),
]
