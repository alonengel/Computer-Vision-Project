"""Figures for Stage 1 (spec: `_docs/stage_1.pdf`).

Covers the five required presentation items: the accuracy table (built in
`scripts/make_tables.py`), accuracy versus training-set size with error bars,
linear-probe training curves, row-normalized confusion matrices, and 2-D feature
visualizations of ~10 classes with the corresponding class prototypes.

Display-name maps keep raw pipeline identifiers out of every figure.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .utils import load_config, repo_path

sns.set_theme(style="whitegrid", context="talk", palette="colorblind")
DPI = 150

DATASET_NAMES = {"dtd": "DTD", "fgvc_aircraft": "FGVC-Aircraft",
                 "flowers102": "Oxford Flowers-102"}
ENCODER_NAMES = {"resnet18": "ResNet-18 (ImageNet-1K)",
                 "dinov2_vits14": "DINOv2 ViT-S/14",
                 "clip_rn50": "CLIP RN50"}
ENCODER_SHORT = {"resnet18": "ResNet-18", "dinov2_vits14": "DINOv2", "clip_rn50": "CLIP RN50"}
HEAD_NAMES = {"linear_probe": "Linear probe", "image_prototype": "Image prototypes",
              "zeroshot_clip": "Zero-shot CLIP",
              "fm_standard": "Standard FM", "fm_rollout": "Rolled-out FM"}
TARGET_NAMES = {"image_prototype": "image prototypes", "clip_text": "CLIP text prototypes"}

# Stage-2 series styles: blues = standard FM, oranges = rolled-out FM; darker =
# more Euler steps. The Stage-1 baseline is always the neutral dashed line.
STAGE2_COLORS = {("fm_standard", 4): "#56B4E9", ("fm_standard", 12): "#0173B2",
                 ("fm_rollout", 4): "#E69F00", ("fm_rollout", 12): "#D55E00"}
STAGE2_BASELINE_COLOR = "#333333"

# One fixed colour per encoder and one linestyle/marker per head, used identically
# in every chart so a series can be identified across figures.
ENCODER_COLORS = {"resnet18": "#0173B2", "dinov2_vits14": "#CC78BC", "clip_rn50": "#D55E00"}
HEAD_STYLE = {"linear_probe": ("-", "o"), "image_prototype": ("--", "s"),
              "zeroshot_clip": (":", "^")}
K_ORDER = ["5shot", "10shot", "full"]
K_LABELS = {"5shot": "5", "10shot": "10", "full": "full"}


def class_palette(n_classes):
    """Class colours for feature/trajectory visualizations: the colourblind
    palette with three swaps for clearer separation (user request) — every
    colour must be tellable apart at small marker sizes. The near-duplicates
    are resolved as: vermillion -> true red, medium pink -> deep purple, and
    light tan -> dark brown (the tan read as a second orange). One shared
    source of truth so every figure and animation agrees."""
    from matplotlib.colors import to_rgb

    pal = list(sns.color_palette("colorblind", n_colors=max(10, n_classes)))
    pal[3] = to_rgb("#D62728")   # vermillion -> red
    pal[4] = to_rgb("#6A3D9A")   # medium pink -> deep purple
    pal[5] = to_rgb("#8C564B")   # light tan -> dark brown (vs bright orange)
    return pal


def dataset_label(name):
    return DATASET_NAMES.get(name, name)


def encoder_label(name, short=False):
    return (ENCODER_SHORT if short else ENCODER_NAMES).get(name, name)


def head_label(name):
    return HEAD_NAMES.get(name, name)


def figures_dir():
    d = repo_path(load_config()["paths"]["figures_dir"])
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save(fig, name):
    path = figures_dir() / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# 2 · accuracy versus training-set size
# --------------------------------------------------------------------------- #

def accuracy_vs_trainsize(dataset, summary, name=None):
    """summary: rows with encoder, head, k_shot, mean_acc, std_acc for one dataset.

    Supervised heads are drawn as accuracy-vs-K lines with error bars; zero-shot
    CLIP (which uses no labeled training images) as a horizontal reference line.
    """
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    x = np.arange(len(K_ORDER))
    sup = summary[summary["head"] != "zeroshot_clip"]
    for (enc, head), g in sup.groupby(["encoder", "head"]):
        g = g.set_index("k_shot").reindex(K_ORDER)
        ls, marker = HEAD_STYLE[head]
        ax.errorbar(x, 100 * g["mean_acc"], yerr=100 * g["std_acc"], marker=marker,
                    linestyle=ls, capsize=4, linewidth=2.2, color=ENCODER_COLORS[enc],
                    label=f"{head_label(head)} — {encoder_label(enc, short=True)}")
    zs = summary[summary["head"] == "zeroshot_clip"]
    for _, r in zs.iterrows():
        ax.axhline(100 * r["mean_acc"], linestyle=":", linewidth=2.2,
                   color=ENCODER_COLORS["clip_rn50"],
                   label="Zero-shot CLIP RN50 (no training images)")
    ax.set_xticks(x)
    ax.set_xticklabels([K_LABELS[k] for k in K_ORDER])
    ax.set_xlabel("training images per class (K)")
    ax.set_ylabel("top-1 test accuracy (%)")
    ax.set_title(f"{dataset_label(dataset)}: accuracy vs. training-set size\n"
                 "(error bars: ± sample std over 3 runs; the full-split prototype and\n"
                 "the zero-shot points are single deterministic runs)", fontsize=12)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=10.5, frameon=True)
    return _save(fig, name or f"acc_vs_trainsize_{dataset}.png")


# --------------------------------------------------------------------------- #
# 3 · linear-probe training curves
# --------------------------------------------------------------------------- #

def training_curves(panels, name, suptitle):
    """panels: list of dicts with 'dataset', 'encoder', 'history' (train/val loss+acc)."""
    fig, axes = plt.subplots(1, len(panels), figsize=(6.4 * len(panels), 4.6), squeeze=False)
    for ax, p in zip(axes[0], panels):
        h = p["history"]
        ax.plot(h["epoch"], h["train_loss"], label="training loss", color="#0173B2", lw=2)
        ax.plot(h["epoch"], h["val_loss"], label="validation loss", color="#D55E00", lw=2)
        best = int(np.argmax(h["val_acc"]))
        ax.axvline(best, color="gray", linestyle="--", lw=1.4)
        late = best > 0.6 * len(h["epoch"])  # keep the label inside the axes
        ax.annotate(f"checkpoint\n(best val acc, ep {best})", xy=(best, ax.get_ylim()[1]),
                    xytext=(-6 if late else 6, -8), textcoords="offset points", fontsize=9,
                    va="top", ha="right" if late else "left", color="gray")
        ax.set_title(f"{dataset_label(p['dataset'])} — {encoder_label(p['encoder'], short=True)}",
                     fontsize=12)
        ax.set_xlabel("epoch")
        # Validation accuracy on a twin axis: the checkpoint is selected on this
        # curve, and it can keep rising while the validation loss already grows.
        ax2 = ax.twinx()
        ax2.plot(h["epoch"], [100 * v for v in h["val_acc"]], color="#029E73",
                 lw=1.8, linestyle=":", label="validation accuracy")
        ax2.set_ylabel("validation accuracy (%)", color="#029E73", fontsize=11)
        ax2.tick_params(axis="y", labelcolor="#029E73")
        ax2.grid(False)
    axes[0][0].set_ylabel("cross-entropy loss")
    handles, labels_ = axes[0][0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color="#029E73", lw=1.8, linestyle=":"))
    labels_.append("validation accuracy")
    axes[0][0].legend(handles, labels_, fontsize=9.5, loc="center right")
    fig.suptitle(suptitle, y=1.04, fontsize=13.5)
    fig.tight_layout()
    return _save(fig, name)


# --------------------------------------------------------------------------- #
# 4 · row-normalized confusion matrix
# --------------------------------------------------------------------------- #

def confusion(cm, class_names, title, name, top_confusions=None):
    """Row-normalized confusion matrix heatmap (rows sum to 1). Class counts here
    are large (47–102), so cells are not annotated; the most frequent confusions
    are listed beside the matrix instead."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.2),
                             gridspec_kw={"width_ratios": [3, 1.35]})
    ax = axes[0]
    ax.grid(False)
    im = ax.imshow(cm, cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
    ax.set_xlabel("predicted class"); ax.set_ylabel("true class")
    ax.set_title(title, fontsize=12)
    n = len(class_names)
    if n <= 50:  # class names still legible — far easier to read than indices
        ticks = np.arange(n)
        ax.set_xticks(ticks); ax.set_yticks(ticks)
        ax.set_xticklabels(class_names, fontsize=6, rotation=90)
        ax.set_yticklabels(class_names, fontsize=6)
    else:
        step = max(1, n // 20)
        ticks = np.arange(0, n, step)
        ax.set_xticks(ticks); ax.set_yticks(ticks)
        ax.set_xticklabels(ticks, fontsize=8); ax.set_yticklabels(ticks, fontsize=8)
        ax.set_xlabel("predicted class (index)"); ax.set_ylabel("true class (index)")
    fig.colorbar(im, ax=ax, fraction=0.046, label="fraction of true-class test images")

    axes[1].axis("off")
    if top_confusions:
        lines = ["Most frequent confusions", ""]
        for t, p, v in top_confusions:
            lines.append(f"{class_names[t]}\n   → {class_names[p]}  ({100 * v:.0f}%)")
        axes[1].text(0, 1, "\n".join(lines), va="top", ha="left", fontsize=10.5,
                     family="monospace")
    fig.tight_layout()
    return _save(fig, name)


# --------------------------------------------------------------------------- #
# 5 · feature visualizations (image/text features + class prototypes)
# --------------------------------------------------------------------------- #

def feature_projection(panels, class_names, title, name, axis_labels=None):
    """panels: list of dicts with 'title', 'xy' [N,2], 'labels' [N], 'proto_xy' [C,2].

    The projection is fitted jointly to the plotted image features and prototypes
    (done by the caller), so prototype positions are comparable to the points.
    Colours are fixed per class index across every panel and figure.
    axis_labels: optional (xlabel, ylabel) shown on every panel — used by the
    PCA-only Stage-2 figures to state the axes and explained variance.
    """
    # Shared class palette (see class_palette); marker shape also varies so
    # class identity does not rest on hue alone.
    palette = class_palette(len(class_names))
    markers = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "*"]
    fig, axes = plt.subplots(1, len(panels), figsize=(6.4 * len(panels), 6.0), squeeze=False)
    for ax, p in zip(axes[0], panels):
        labels = np.asarray(p["labels"])
        for j, c in enumerate(sorted(np.unique(labels))):
            m = labels == c
            ax.scatter(p["xy"][m, 0], p["xy"][m, 1], s=18, alpha=0.65,
                       color=palette[j], marker=markers[j % len(markers)],
                       label=class_names[j])
            if p.get("proto_xy") is not None:
                ax.scatter(p["proto_xy"][j, 0], p["proto_xy"][j, 1], marker="*", s=430,
                           color=palette[j], edgecolors="black", linewidths=1.3, zorder=5)
        ax.set_title(p["title"], fontsize=12)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        if axis_labels:
            ax.set_xlabel(axis_labels[0], fontsize=10)
            ax.set_ylabel(axis_labels[1], fontsize=10)
    handles, labels_ = axes[0][0].get_legend_handles_labels()
    from matplotlib.lines import Line2D

    handles.append(Line2D([], [], linestyle="", marker="*", markersize=15,
                          color="white", markeredgecolor="black", label="class prototype"))
    labels_.append("class prototype")
    fig.legend(handles, labels_, loc="upper center", bbox_to_anchor=(0.5, 0.0),
               ncol=min(6, len(labels_)), fontsize=10, frameon=True, markerscale=1.6)
    fig.suptitle(title, y=1.02, fontsize=13.5)
    fig.tight_layout()
    return _save(fig, name)


# --------------------------------------------------------------------------- #
# Stage 2 — FM vs baseline accuracy, FM training curves, flow trajectories
# --------------------------------------------------------------------------- #

def stage2_accuracy_chart(dataset, encoder, target, rows, baseline, name, subtitle=""):
    """Accuracy vs training-set size for one (dataset, encoder, target) setting.

    rows: summary_stage2 rows for this setting (head, T, k_shot, mean/std).
    baseline: {"series": [(k_shot, mean, std, n_runs)]} for the image-prototype
    baseline, or {"hline": acc} for the zero-shot CLIP reference.
    """
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    x = np.arange(len(K_ORDER))
    if "series" in baseline:
        ks, means, stds, ns = zip(*baseline["series"])
        order = [ks.index(k) for k in K_ORDER]
        ax.errorbar(x, [100 * means[i] for i in order],
                    yerr=[100 * stds[i] if ns[i] > 1 else 0 for i in order],
                    marker="s", linestyle="--", capsize=4, linewidth=2.2,
                    color=STAGE2_BASELINE_COLOR,
                    label=f"Stage-1 baseline: {head_label('image_prototype')}")
    else:
        ax.axhline(100 * baseline["hline"], linestyle=":", linewidth=2.2,
                   color=STAGE2_BASELINE_COLOR,
                   label="Stage-1 reference: Zero-shot CLIP (no training images)")
    for (head, T), g in rows.groupby(["head", "T"]):
        g = g.set_index("k_shot").reindex(K_ORDER)
        ax.errorbar(x, 100 * g["mean_acc"], yerr=100 * g["std_acc"], marker="o",
                    linestyle="-" if head == "fm_standard" else "-.",
                    capsize=4, linewidth=2.0, color=STAGE2_COLORS[(head, T)],
                    label=f"{head_label(head)}, T = {T}")
    ax.set_xticks(x)
    ax.set_xticklabels([K_LABELS[k] for k in K_ORDER])
    ax.set_xlabel("training images per class (K)")
    ax.set_ylabel("top-1 test accuracy (%)")
    ax.set_title(f"{dataset_label(dataset)} — {encoder_label(encoder, short=True)}, "
                 f"FM toward {TARGET_NAMES[target]}\n{subtitle}", fontsize=12)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=10.5, frameon=True)
    return _save(fig, name)


def fm_training_curves(panels, name, suptitle):
    """panels: list of dicts with 'title' and 'curves':
    [(label, color, history)] or [(label, color, history, linestyle)].
    Training loss only; log scale because the two objectives live on different
    magnitudes; line styles differ so series identity does not rest on hue
    alone. Every panel carries its own y-label (scales are independent)."""
    fig, axes = plt.subplots(1, len(panels), figsize=(6.0 * len(panels), 4.4), squeeze=False)
    for ax, p in zip(axes[0], panels):
        for curve in p["curves"]:
            label, color, h = curve[:3]
            ls = curve[3] if len(curve) > 3 else "-"
            ax.semilogy(h["epoch"], h["train_loss"], label=label, color=color,
                        lw=2, linestyle=ls)
        ax.set_title(p["title"], fontsize=12)
        ax.set_xlabel("epoch")
        ax.set_ylabel("training loss (log scale)", fontsize=10)
        ax.legend(fontsize=9.5)
    fig.suptitle(suptitle, y=1.04, fontsize=13.5)
    fig.tight_layout()
    return _save(fig, name)


def flow_trajectory_chart(panels, bg, class_names, title, name, axis_labels=None):
    """Flow trajectories in a joint PCA plane (spec item 4).

    bg: {'xy' [N,2], 'labels' [N], 'proto_xy' [C,2]} — shared background of test
    features and prototypes. panels: list of {'title', 'trajs': [(class_idx,
    xy [S+1, 2])]}. All coordinates must come from ONE jointly fitted projection.
    """
    palette = class_palette(len(class_names))
    fig, axes = plt.subplots(1, len(panels), figsize=(6.8 * len(panels), 6.2), squeeze=False)
    for ax, p in zip(axes[0], panels):
        labels = np.asarray(bg["labels"])
        for j in sorted(np.unique(labels)):
            m = labels == j
            ax.scatter(bg["xy"][m, 0], bg["xy"][m, 1], s=12, alpha=0.18,
                       color=palette[j], marker="o")
            ax.scatter(bg["proto_xy"][j, 0], bg["proto_xy"][j, 1], marker="*", s=430,
                       color=palette[j], edgecolors="black", linewidths=1.3, zorder=5)
        for class_idx, xy in p["trajs"]:
            c = palette[class_idx]
            ax.plot(xy[:, 0], xy[:, 1], color=c, lw=1.8, alpha=0.95, zorder=4)
            ax.scatter(xy[1:-1, 0], xy[1:-1, 1], color=c, s=26, zorder=4,
                       edgecolors="white", linewidths=0.5)
            ax.scatter(xy[0, 0], xy[0, 1], color=c, s=90, marker="o",
                       edgecolors="black", linewidths=1.2, zorder=6)
            ax.scatter(xy[-1, 0], xy[-1, 1], color=c, s=110, marker="X",
                       edgecolors="black", linewidths=1.2, zorder=6)
        ax.set_title(p["title"], fontsize=12)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        if axis_labels:
            ax.set_xlabel(axis_labels[0], fontsize=10)
            ax.set_ylabel(axis_labels[1], fontsize=10)
    from matplotlib.lines import Line2D

    handles = [Line2D([], [], linestyle="", marker="o", markersize=10, color="#AAAAAA",
                      markeredgecolor="black", label="original feature $\\hat{z}_0$"),
               Line2D([], [], linestyle="", marker="X", markersize=11, color="#AAAAAA",
                      markeredgecolor="black", label="transported feature $\\hat{z}_T$"),
               Line2D([], [], linestyle="", marker="*", markersize=16, color="white",
                      markeredgecolor="black", label="class prototype")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0),
               ncol=3, fontsize=10.5, frameon=True)
    fig.suptitle(title, y=1.02, fontsize=13.5)
    fig.tight_layout()
    return _save(fig, name)


def sample_grid(pool, indices, class_names, title, name, n_cols=8):
    """Qualitative dataset preview: one row of example images per selected class."""
    n = len(indices)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.9 * n_cols, 2.2 * n_rows),
                             squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, i in zip(axes.ravel(), indices):
        ax.imshow(pool.get_image(int(i)))
        ax.set_title(class_names[int(pool.labels[i])], fontsize=8)
        ax.axis("off")
    fig.suptitle(title, y=1.01)
    fig.tight_layout()
    return _save(fig, name)
