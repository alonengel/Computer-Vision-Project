"""Architecture diagrams: the three baseline heads + the advised Stage-2/3 designs.

Outputs:
  results/figures/arch_baselines.png  — prototype / linear probe / zero-shot CLIP
  results/figures/arch_stage2_advised.png — advised FM-as-final-layer + FM-before-linear
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from src.visualize import figures_dir

C_FROZEN = "#B0B0B0"   # frozen modules
C_DATA = "#DDEBF7"     # tensors / data
C_HEAD = "#029E73"     # trained head
C_FIXED = "#0173B2"    # parameter-free computation
C_TEXT = "#D55E00"     # text/semantic path
C_FM = "#CC78BC"       # flow-matching module


def box(ax, x, y, w, h, text, color, fontsize=10.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                facecolor=color, edgecolor="black", linewidth=1.1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize,
            fontweight="bold" if bold else "normal", wrap=True)


def arrow(ax, x1, y1, x2, y2, text=None, fontsize=9):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=16, linewidth=1.4, color="black"))
    if text:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.025, text, ha="center",
                va="bottom", fontsize=fontsize, style="italic")


def panel(ax, title):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_title(title, fontsize=13, fontweight="bold", loc="left")


def baselines():
    fig, axes = plt.subplots(3, 1, figsize=(12, 10.5))

    # --- Prototype ---
    ax = axes[0]
    panel(ax, "(a) Prototype head — parameter-free, per episode")
    box(ax, 0.01, 0.55, 0.13, 0.22, "query\nimage $x$", C_DATA)
    box(ax, 0.20, 0.55, 0.17, 0.22, "frozen encoder\n$f$", C_FROZEN, bold=True)
    box(ax, 0.43, 0.55, 0.15, 0.22, "embedding\n$f(x)\\in\\mathbb{R}^D$", C_DATA)
    box(ax, 0.01, 0.10, 0.13, 0.22, "support set\n$S_k$ ($K$ imgs/class)", C_DATA)
    box(ax, 0.20, 0.10, 0.17, 0.22, "frozen encoder\n$f$", C_FROZEN, bold=True)
    box(ax, 0.43, 0.10, 0.15, 0.22, "class means\n$c_k=\\frac{1}{K}\\sum_{x\\in S_k}f(x)$", C_DATA)
    box(ax, 0.65, 0.32, 0.18, 0.24, "cosine similarity\n$\\cos(f(x),\\,c_k)$\n(Euclidean: ablation)", C_FIXED)
    box(ax, 0.88, 0.34, 0.11, 0.20, "argmax\n→ class", C_DATA)
    arrow(ax, 0.14, 0.66, 0.20, 0.66); arrow(ax, 0.37, 0.66, 0.43, 0.66)
    arrow(ax, 0.14, 0.21, 0.20, 0.21); arrow(ax, 0.37, 0.21, 0.43, 0.21)
    arrow(ax, 0.58, 0.62, 0.65, 0.50); arrow(ax, 0.58, 0.25, 0.65, 0.38)
    arrow(ax, 0.83, 0.44, 0.88, 0.44)
    ax.text(0.65, 0.05, "no trained parameters — prototypes recomputed from support only",
            fontsize=9.5, style="italic")

    # --- Linear probe ---
    ax = axes[1]
    panel(ax, "(b) Linear probe — trained on the support set only")
    box(ax, 0.01, 0.40, 0.13, 0.24, "query\nimage $x$", C_DATA)
    box(ax, 0.20, 0.40, 0.17, 0.24, "frozen encoder\n$f$", C_FROZEN, bold=True)
    box(ax, 0.43, 0.40, 0.15, 0.24, "embedding\n$f(x)\\in\\mathbb{R}^D$", C_DATA)
    box(ax, 0.64, 0.40, 0.20, 0.24, "linear head\n$W\\in\\mathbb{R}^{C\\times D},\\,b$", C_HEAD, bold=True)
    box(ax, 0.88, 0.42, 0.11, 0.20, "argmax\n→ class", C_DATA)
    arrow(ax, 0.14, 0.52, 0.20, 0.52); arrow(ax, 0.37, 0.52, 0.43, 0.52)
    arrow(ax, 0.58, 0.52, 0.64, 0.52); arrow(ax, 0.84, 0.52, 0.88, 0.52)
    box(ax, 0.30, 0.05, 0.46, 0.20,
        "training: CrossEntropy on support embeddings — Adam, 300 steps, lr 0.01,\n"
        "wd 0, init $0.01\\cdot\\mathcal{N}(0,1)$ (fixed a priori; episodic: 600 heads batched)", C_DATA, fontsize=9)
    arrow(ax, 0.63, 0.25, 0.70, 0.40)

    # --- Zero-shot CLIP ---
    ax = axes[2]
    panel(ax, "(c) Zero-shot CLIP — semantic reference baseline (no support images)")
    box(ax, 0.01, 0.55, 0.13, 0.22, "query\nimage $x$", C_DATA)
    box(ax, 0.20, 0.55, 0.19, 0.22, "frozen CLIP\nimage encoder", C_FROZEN, bold=True)
    box(ax, 0.45, 0.55, 0.13, 0.22, "image emb.\n$f(x)$", C_DATA)
    box(ax, 0.01, 0.10, 0.13, 0.22, "class names\n(text!)", C_TEXT)
    box(ax, 0.20, 0.10, 0.19, 0.22, "prompts → frozen\nCLIP text encoder", C_TEXT, bold=True)
    box(ax, 0.45, 0.10, 0.13, 0.22, "text emb.\n$t_c$ (cached)", C_DATA)
    box(ax, 0.65, 0.32, 0.18, 0.24, "cosine similarity\n$\\cos(f(x),\\,t_c)$", C_FIXED)
    box(ax, 0.88, 0.34, 0.11, 0.20, "argmax\n→ class", C_DATA)
    arrow(ax, 0.14, 0.66, 0.20, 0.66); arrow(ax, 0.39, 0.66, 0.45, 0.66)
    arrow(ax, 0.14, 0.21, 0.20, 0.21); arrow(ax, 0.39, 0.21, 0.45, 0.21)
    arrow(ax, 0.58, 0.62, 0.65, 0.50); arrow(ax, 0.58, 0.25, 0.65, 0.38)
    arrow(ax, 0.83, 0.44, 0.88, 0.44)
    ax.text(0.65, 0.05, "information source: class names + pretrained image–text alignment",
            fontsize=9.5, style="italic")

    fig.suptitle("Stage 1 baseline architectures (encoders always frozen)", fontsize=15, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    p = figures_dir() / "arch_baselines.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"figure: {p}")


def stage2_advised():
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # --- Stage 2: FM as the final layer ---
    ax = axes[0]
    panel(ax, "(a) Stage 2 (advised) — Flow Matching as the decision layer")
    box(ax, 0.01, 0.42, 0.11, 0.24, "image $x$", C_DATA)
    box(ax, 0.16, 0.42, 0.15, 0.24, "frozen encoder $f$\n(per-dataset pick,\nbest_baselines.json)", C_FROZEN, bold=True, fontsize=9)
    box(ax, 0.35, 0.42, 0.12, 0.24, "$x_0 = f(x)$", C_DATA)
    box(ax, 0.51, 0.38, 0.22, 0.32,
        "Flow Matching\n$\\dot{x}=v_\\theta(x_t,t)$\nintegrate $t: 0\\to 1$", C_FM, bold=True)
    box(ax, 0.77, 0.42, 0.10, 0.24, "$x_1$", C_DATA)
    box(ax, 0.90, 0.42, 0.09, 0.24, "nearest\ntarget\n→ class", C_FIXED)
    arrow(ax, 0.12, 0.54, 0.16, 0.54); arrow(ax, 0.31, 0.54, 0.35, 0.54)
    arrow(ax, 0.47, 0.54, 0.51, 0.54); arrow(ax, 0.73, 0.54, 0.77, 0.54)
    arrow(ax, 0.87, 0.54, 0.90, 0.54)
    box(ax, 0.51, 0.04, 0.38, 0.22,
        "targets (ADR 0003): per-episode support prototypes $c_k$\nor CLIP text embeddings $t_c$ — never eval-split statistics", C_DATA, fontsize=9)
    arrow(ax, 0.80, 0.26, 0.88, 0.42)
    ax.text(0.01, 0.10, "training (train classes only): standard FM loss on random $t$\n"
                        "vs. rolled-out trajectory loss — both compared vs the selected baselines",
            fontsize=9.5, style="italic")

    # --- Stage 3: FM before linear ---
    ax = axes[1]
    panel(ax, "(b) Stage 3 (advised) — Flow Matching before the linear head, trained jointly")
    box(ax, 0.01, 0.42, 0.11, 0.24, "image $x$", C_DATA)
    box(ax, 0.16, 0.42, 0.15, 0.24, "frozen encoder\n$f$", C_FROZEN, bold=True)
    box(ax, 0.35, 0.42, 0.20, 0.28, "Flow Matching\nmodule $v_\\theta$", C_FM, bold=True)
    box(ax, 0.59, 0.42, 0.17, 0.24, "reorganized\nembedding", C_DATA)
    box(ax, 0.80, 0.42, 0.10, 0.24, "nn.Linear\n$(D\\to C)$", C_HEAD, bold=True)
    box(ax, 0.93, 0.44, 0.06, 0.20, "CE\n→ class", C_DATA)
    arrow(ax, 0.12, 0.54, 0.16, 0.54); arrow(ax, 0.31, 0.54, 0.35, 0.54)
    arrow(ax, 0.55, 0.54, 0.59, 0.54); arrow(ax, 0.76, 0.54, 0.80, 0.54)
    arrow(ax, 0.90, 0.54, 0.93, 0.54)
    ax.text(0.16, 0.12, "θ and (W, b) trained jointly with CrossEntropy — the encoder stays frozen\n"
                        "(end-to-end refers to FM + linear only); evaluated on the same fixed episodes",
            fontsize=9.5, style="italic")

    fig.suptitle("Advised Stage-2/3 architectures — built on each dataset's selected embedding\n"
                 "(best_baselines.json: per dataset × head), beating that config paired on identical episodes",
                 fontsize=13.5, y=1.00)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    p = figures_dir() / "arch_stage2_advised.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"figure: {p}")


if __name__ == "__main__":
    baselines()
    stage2_advised()
