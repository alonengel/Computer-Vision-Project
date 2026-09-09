# v2 talk — section 1: title, datasets, pipeline, research question; hidden setup cell.

TITLE = r"""
<style>
h1, h2, h3 { color: #1f3a5f; }
</style>

# <span style="color:#1f3a5f;">Stage 3 · Flow Matching Before a Frozen Linear Classifier</span>

<p style="font-size:1.15em; color:#333333; margin:0.2em 0 0.8em 0;">
<b>DTD</b> with a frozen <b>ResNet-18</b> &nbsp;·&nbsp; <b>FGVC-Aircraft</b> with a frozen <b>DINOv2 ViT-S/14</b> &nbsp;·&nbsp; the Stage-1 linear classifier kept frozen
</p>

<div style="text-align:center; font-size:1.4em; line-height:2.4; margin:1.0em 0 1.0em 0;">
<span style="border:2px solid #1f3a5f; border-radius:6px; padding:4px 14px;">Image</span> →
<span style="border:2px solid #1f3a5f; border-radius:6px; padding:4px 14px;">Frozen Encoder</span> →
<b>z</b> →
<span style="background:#1f3a5f; color:#ffffff; border-radius:6px; padding:5px 16px; font-weight:bold;">Flow Matching</span> →
<b>ẑ</b> →
<span style="border:2px solid #1f3a5f; border-radius:6px; padding:4px 14px;">Frozen Linear Classifier</span> →
<span style="border:2px solid #1f3a5f; border-radius:6px; padding:4px 14px;">Prediction</span>
</div>

<p style="font-size:1.25em; margin-top:1.0em;"><b>Research question.</b> Can Flow Matching transform frozen encoder features into a representation that the existing frozen Stage-1 linear classifier separates better?</p>
"""

SETUP_CODE = r"""
# Shared setup: repository paths, configuration and display-name maps.
# Every later cell reads tables, training histories, figures and models of
# record under results/ — nothing is recomputed or written back.
import io
import json
import sys
from pathlib import Path

REPO = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Image, Markdown, display

from src.utils import load_config
from src.visualize import class_palette, dataset_label, encoder_label

pd.set_option("display.precision", 2)
pd.set_option("display.max_rows", 200)

cfg = load_config()
MET = REPO / "results" / "metrics"
FIG = REPO / "results" / "figures"
CURVES = REPO / "results" / "artifacts" / "curves_stage3"
MODELS = REPO / "results" / "artifacts" / "stage3_models"
K_LABEL = f"{cfg['stage3']['k_shot']}shot"
S1_COLOR, S2_COLOR = "#0173B2", "#D55E00"      # Strategy 1 blue, Strategy 2 orange (as in the figures of record)


def show(fig, dpi=110):
    # Render a matplotlib figure as an embedded PNG (independent of the active backend).
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    display(Image(data=buf.getvalue()))
"""

CELLS = [
    ("markdown", TITLE),
    ("code", SETUP_CODE),
]
