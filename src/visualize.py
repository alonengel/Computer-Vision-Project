"""All figures. One consistent style; every figure saved to results/figures/ with a caption-ready title."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .utils import load_config, repo_path

sns.set_theme(style="whitegrid", context="talk", palette="colorblind")
DPI = 150

# Presentation names for datasets and methods — raw pipeline IDs must never
# appear in figure text (they may appear in filenames).
DATASET_NAMES = {"mnist": "MNIST", "cifar10": "CIFAR-10", "mini_imagenet": "Mini-ImageNet"}

# One global style per method: fixed color + marker + linestyle, independent of
# which series appear in a figure, so cross-figure comparison is unambiguous
# and series remain distinguishable without color (CVD-safe).
METHOD_STYLES = {
    "proto_cos__clip_vitb32":       ("Prototype (cos), CLIP",      "#0173B2", "o", "-"),
    "proto_eucl__clip_vitb32":      ("Prototype (eucl), CLIP",     "#0173B2", "s", "--"),
    "linear__clip_vitb32":          ("Linear probe, CLIP",         "#029E73", "D", "-"),
    "clip_zeroshot__clip_vitb32":   ("Zero-shot CLIP (1 prompt)",  "#D55E00", "^", "-"),
    "clip_zeroshot_ens__clip_vitb32": ("Zero-shot CLIP (ensemble)", "#D55E00", "v", "--"),
    "proto_cos__dinov2_vits14":     ("Prototype (cos), DINOv2",    "#CC78BC", "o", "-"),
    "proto_eucl__dinov2_vits14":    ("Prototype (eucl), DINOv2",   "#CC78BC", "s", "--"),
    "proto_cos__resnet50":          ("Prototype (cos), ResNet-50", "#ECE133", "o", "-"),
    "proto_eucl__resnet50":         ("Prototype (eucl), ResNet-50", "#ECE133", "s", "--"),
}


def method_label(name):
    return METHOD_STYLES.get(name, (name,))[0]


def dataset_label(name):
    return DATASET_NAMES.get(name, name)


def figures_dir():
    d = repo_path(load_config()["paths"]["figures_dir"])
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save(fig, name):
    path = figures_dir() / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def episode_grid(pool, episode, k_show=5, q_show=4, name=None):
    """Support | query grid for episode 0 of a saved episodic file.

    Rows = episode classes; left block = up to k_show support images,
    right block (after a gap) = q_show query images.
    """
    n_way = episode["n_way"]
    k = min(episode["k_shot"], k_show)
    cls = episode["classes"][0].tolist()
    sup = episode["support_idx"][0].reshape(n_way, episode["k_shot"])
    qry = episode["query_idx"][0].reshape(n_way, episode["n_query"])
    n_cols = k + q_show
    fig, axes = plt.subplots(n_way, n_cols, figsize=(1.9 * n_cols + 1.2, 1.9 * n_way))
    for r in range(n_way):
        for c in range(n_cols):
            ax = axes[r, c]
            idx = sup[r, c] if c < k else qry[r, c - k]
            ax.imshow(pool.get_image(int(idx)))
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_edgecolor("tab:blue" if c < k else "tab:orange"); s.set_linewidth(2.5)
        axes[r, 0].set_ylabel(pool.class_names[cls[r]], rotation=0, ha="right",
                              va="center", fontsize=12)
    fig.text(0.5 * (k / n_cols), 1.005, "support", color="tab:blue", fontsize=14,
             ha="center", transform=fig.transFigure)
    fig.text((k + 0.5 * q_show) / n_cols, 1.005, "query", color="tab:orange",
             fontsize=14, ha="center", transform=fig.transFigure)
    fig.suptitle(f"{dataset_label(pool.name)}: {n_way}-way {episode['k_shot']}-shot "
                 f"episode (support = blue, query = orange)", y=1.06)
    return _save(fig, name or f"episode_grid_{pool.name}.png")


def embedding_scatter_panels(panel_data, title, name):
    """2-D embedding panels (t-SNE/UMAP). panel_data: list of dicts with keys
    'title', 'xy' [N,2], 'labels' [N], 'class_names', optional 'proto_xy' [C,2]."""
    n = len(panel_data)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6.5), squeeze=False)
    for ax, p in zip(axes[0], panel_data):
        labels = np.asarray(p["labels"])
        for c in np.unique(labels):
            m = labels == c
            ax.scatter(p["xy"][m, 0], p["xy"][m, 1], s=8, alpha=0.55,
                       label=p["class_names"][c])
        if p.get("proto_xy") is not None:
            ax.scatter(p["proto_xy"][:, 0], p["proto_xy"][:, 1], marker="*", s=420,
                       c="black", edgecolors="white", linewidths=1.2, zorder=5)
        ax.set_title(p["title"]); ax.set_xticks([]); ax.set_yticks([])
    # Shared legend centered below all panels, with proxy handles so the
    # prototype star renders at legend scale (not data scale).
    from matplotlib.lines import Line2D

    p0 = panel_data[0]
    labels0 = np.asarray(p0["labels"])
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    handles = [Line2D([], [], linestyle="", marker="o", markersize=7,
                      color=colors[i % len(colors)], label=p0["class_names"][c])
               for i, c in enumerate(np.unique(labels0))]
    handles.append(Line2D([], [], linestyle="", marker="*", markersize=13,
                          color="black", markeredgecolor="white", label="class prototype"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.02),
               ncol=min(len(handles), 7), fontsize=10, frameon=True)
    fig.suptitle(title, y=1.02)
    return _save(fig, name)


def accuracy_vs_k(df, dataset, name=None):
    """Accuracy-vs-K curves. df columns: classifier, k_shot, acc, err (CI half-width or std)."""
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    for clf in [c for c in METHOD_STYLES if c in set(df["classifier"])]:
        g = df[df["classifier"] == clf].sort_values("k_shot")
        label, color, marker, ls = METHOD_STYLES[clf]
        if (g["k_shot"] == 0).all():  # zero-shot: K-independent -> reference line
            ax.axhline(100 * g["acc"].iloc[0], linestyle=ls, color=color,
                       label=f"{label} (K-indep.)")
            continue
        ax.errorbar(g["k_shot"], 100 * g["acc"], yerr=100 * g["err"], marker=marker,
                    linestyle=ls, color=color, capsize=3, label=label)
    ax.set_xlabel("K (shots per class)"); ax.set_ylabel("accuracy (%)")
    ax.set_xticks(sorted(df.loc[df["k_shot"] > 0, "k_shot"].unique()))
    ax.set_title(f"{dataset_label(dataset)}: accuracy vs. shots")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=11, frameon=True)
    return _save(fig, name or f"acc_vs_k_{dataset}.png")


def grouped_bars(df, title, name):
    """Grouped bar chart. df columns: dataset, classifier, acc, err."""
    datasets = list(df["dataset"].unique())
    classifiers = [c for c in METHOD_STYLES if c in set(df["classifier"])]
    x = np.arange(len(datasets)); w = 0.8 / len(classifiers)
    fig, ax = plt.subplots(figsize=(2.9 * len(datasets) + 3.5, 5.5))
    for i, clf in enumerate(classifiers):
        g = df[df["classifier"] == clf].set_index("dataset").reindex(datasets)
        label, color, _, ls = METHOD_STYLES[clf]
        ax.bar(x + (i - (len(classifiers) - 1) / 2) * w, 100 * g["acc"], w,
               yerr=100 * g["err"], capsize=4, label=label, color=color,
               edgecolor="white", linewidth=0.5,
               hatch="//" if ls == "--" else None)
    ax.set_xticks(x); ax.set_xticklabels([dataset_label(d) for d in datasets])
    ax.set_ylabel("accuracy (%)")
    ax.set_title(title)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=11, frameon=True)
    return _save(fig, name)


def confusion(cm, class_names, title, name):
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    sns.heatmap(cm, annot=len(class_names) <= 20, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax,
                annot_kws={"fontsize": 9}, cbar_kws={"label": "test images"})
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=9)
    return _save(fig, name)


def clip_qualitative(images, true_names, probs, class_names, title, name, top=5):
    """Per-image CLIP zero-shot panel: image + horizontal probability bars."""
    n = len(images)
    fig, axes = plt.subplots(n, 2, figsize=(9.5, 2.6 * n),
                             gridspec_kw={"width_ratios": [1, 2.2]})
    axes = np.atleast_2d(axes)
    for r in range(n):
        axes[r, 0].imshow(images[r]); axes[r, 0].axis("off")
        axes[r, 0].set_title(f"true: {true_names[r]}", fontsize=11)
        order = np.argsort(probs[r])[::-1][:top]
        colors = ["tab:green" if class_names[j] == true_names[r] else "tab:gray" for j in order]
        axes[r, 1].barh(range(top), probs[r][order], color=colors)
        axes[r, 1].set_yticks(range(top))
        axes[r, 1].set_yticklabels([class_names[j] for j in order], fontsize=10)
        axes[r, 1].invert_yaxis(); axes[r, 1].set_xlim(0, 1)
    axes[-1, 1].set_xlabel("softmax probability over class prompts", fontsize=11)
    fig.suptitle(title + "\n(green = ground-truth class)", y=1.02)
    fig.tight_layout()
    return _save(fig, name)


def failure_gallery(items, title, name, n_cols=6):
    """Hardest misclassifications. items: list of (PIL image, true_name, pred_name)."""
    n = len(items)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.3 * n_cols, 2.9 * n_rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (img, t, p) in zip(axes.ravel(), items):
        ax.imshow(img); ax.axis("off")
        ax.set_title(f"true: {t}\npred: {p}", fontsize=9, color="tab:red")
    fig.suptitle(title, y=1.01)
    fig.tight_layout()
    return _save(fig, name)
