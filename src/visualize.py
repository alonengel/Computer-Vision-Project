"""All figures. One consistent style; every figure saved to results/figures/ with a caption-ready title."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .utils import load_config, repo_path

sns.set_theme(style="whitegrid", context="talk", palette="colorblind")
DPI = 150


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
    axes[0, max(k // 2 - 1, 0)].set_title("support", color="tab:blue", fontsize=13)
    axes[0, k + q_show // 2].set_title("query", color="tab:orange", fontsize=13)
    fig.suptitle(f"{pool.name}: {n_way}-way {episode['k_shot']}-shot episode "
                 f"(support = blue, query = orange)", y=1.02)
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
                       c="black", edgecolors="white", linewidths=1.2, label="prototype", zorder=5)
        ax.set_title(p["title"]); ax.set_xticks([]); ax.set_yticks([])
    axes[0, -1].legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=10,
                       markerscale=2.0, frameon=True)
    fig.suptitle(title, y=1.02)
    return _save(fig, name)


def accuracy_vs_k(df, dataset, name=None):
    """Accuracy-vs-K curves. df columns: classifier, k_shot, acc, err (CI half-width or std)."""
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for clf, g in df.groupby("classifier"):
        g = g.sort_values("k_shot")
        if (g["k_shot"] == 0).all():  # zero-shot: K-independent -> reference line
            ax.axhline(100 * g["acc"].iloc[0], linestyle="--", alpha=0.7,
                       color="gray" if "ens" not in clf else "black",
                       label=f"{clf} (K-independent)")
            continue
        ax.errorbar(g["k_shot"], 100 * g["acc"], yerr=100 * g["err"], marker="o",
                    capsize=4, label=clf)
    ax.set_xlabel("K (shots per class)"); ax.set_ylabel("accuracy (%)")
    ax.set_xticks(sorted(df.loc[df["k_shot"] > 0, "k_shot"].unique()))
    ax.set_title(f"{dataset}: accuracy vs. shots"); ax.legend()
    return _save(fig, name or f"acc_vs_k_{dataset}.png")


def grouped_bars(df, title, name):
    """Grouped bar chart. df columns: dataset, classifier, acc, err."""
    datasets = list(df["dataset"].unique())
    classifiers = list(df["classifier"].unique())
    x = np.arange(len(datasets)); w = 0.8 / len(classifiers)
    fig, ax = plt.subplots(figsize=(2.6 * len(datasets) + 3, 5.5))
    for i, clf in enumerate(classifiers):
        g = df[df["classifier"] == clf].set_index("dataset").reindex(datasets)
        ax.bar(x + (i - (len(classifiers) - 1) / 2) * w, 100 * g["acc"], w,
               yerr=100 * g["err"], capsize=4, label=clf)
    ax.set_xticks(x); ax.set_xticklabels(datasets); ax.set_ylabel("accuracy (%)")
    ax.set_title(title); ax.legend()
    return _save(fig, name)


def confusion(cm, class_names, title, name):
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    sns.heatmap(cm, annot=len(class_names) <= 20, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax,
                annot_kws={"fontsize": 9})
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
    fig.suptitle(title, y=1.005)
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
