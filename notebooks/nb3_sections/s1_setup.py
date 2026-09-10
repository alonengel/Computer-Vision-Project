CELLS = [
    ("markdown", "## 1 · Environment, configuration, and the pre-registered protocol"),
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
    print("Runtime:", json.dumps(json.load(f), indent=2))

cfg = load_config()
s3 = cfg["stage3"]
print("\\nSettings:", s3["settings"], "| K =", s3["k_shot"], "| T =", s3["T"],
      "| feature space:", s3["feature_space"])
print("Training policy:", json.dumps(s3["training"], indent=2))
print("Rolled-strategy lambda grid:", s3["strategy1"]["lambda_grid"],
      "| Guided-strategy grid: beta", s3["strategy2"]["beta_grid"], "x m", s3["strategy2"]["m_grid"],
      "| trust region alpha =", s3["strategy2"]["alpha_trust_region"])
"""),
    ("markdown", """
**Provenance of every choice.** The FM operates in **raw feature space** — the probe was trained on raw features ($s = Wz + b$, norms ≈ 10–40), so normalizing on entry would break the required identity-at-initialization (note the deliberate contrast with Stage 2, where the downstream classifier was cosine-based and the FM therefore lived on the unit sphere: *the FM operates in the space its downstream classifier acts in*). The velocity network and Euler procedure are the Stage-2 design, as the spec instructs (MLP $d{+}1 \\to 512 \\to 512 \\to d$, SiLU, scalar $t$ concatenated; $\\hat{z}_{k+1} = \\hat{z}_k + \\frac{1}{T}v_\\theta(\\hat{z}_k, k/T)$). **Identity initialization is exact**: the final velocity layer (weight *and* bias) starts at zero, so $\\hat{z} = z$ bit-for-bit — asserted, not assumed (§2).

**Training policy (uniform for both strategies, fixed a priori):** AdamW, lr $10^{-3}$, weight decay $10^{-4}$, batch 64, exactly 200 epochs with the **best-validation-accuracy checkpoint** retained (ties → lowest validation CE → earliest epoch). Validation-based selection is deliberate and symmetric: the baseline probe itself was selected the same way in Stage 1, and both strategies expose the identical full-pipeline validation curve. Failure policy (objective trigger — NaN/Inf loss only): restart with the next pre-registered fallback (cumulative ladder: grad-clip 1.0 → lr 3e-4 → internal input standardization, which preserves the exact identity); finite late deterioration is absorbed by checkpointing; exhausted ladder ⇒ run marked failed. **Hyperparameters were selected per dataset on seed-0 validation only**; winners applied unchanged to seeds 1–2; the **test split stayed sealed** during all training and selection, and was evaluated in **one final pass per pre-registered phase** (the mandatory grid; then the optional extension), each over its already-locked checkpoints. Disclosed: seed 0 is therefore partly a development run, and the 3-seed mean is a summary, not an independent confirmatory estimate.
"""),
]
