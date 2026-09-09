# v2 talk — section 10: feature space, FGVC-Aircraft (joint PCA).

HEAD = r"""
## <span style="color:#1f3a5f;">9 · Feature space — FGVC-Aircraft / DINOv2</span>

Ten classes of the shared visualization selection — the same test examples, classes and colours as in Stages 1–2. One PCA is fitted **jointly** over all three feature sets (original, after Strategy 1, after Strategy 2; seed-0 models), so the three panels share the same plane.
"""

CODE = r"""
from make_figures import viz_selection
from make_figures_stage2 import _joint_pca, pc_labels
from src.embeddings import load_features
from src.stage3 import load_pinned_probe, load_stage3_fm

MARKERS = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "*"]


def feature_figure(ds, enc, expected_axes, seed=0):
    # Same selection, models and joint PCA as the figure of record
    # stage3_features_<ds>_<enc>.png (Appendix A.8); the explained-variance labels
    # are asserted to match that figure.
    vz = cfg["feature_viz"]
    classes, idx = viz_selection(ds, vz["n_classes"], vz["class_seed"], vz["max_per_class"])
    feats = load_features(ds, "test", enc)
    X = feats["features"][idx].float()                      # raw feature space
    remap = {int(c): j for j, c in enumerate(classes)}
    y = np.array([remap[int(v)] for v in feats["labels"][idx].numpy()])
    probe = load_pinned_probe(MODELS / f"probe_{ds}_{enc}_{K_LABEL}_seed{seed}.pt")
    Z1 = load_stage3_fm(MODELS / f"{ds}_{enc}_fm_s1_{K_LABEL}_seed{seed}.pt", probe).transport(X)
    Z2 = load_stage3_fm(MODELS / f"{ds}_{enc}_fm_s2_{K_LABEL}_seed{seed}.pt", probe).transport(X)
    pca, (xy0, xy1, xy2) = _joint_pca([X.numpy(), Z1.numpy(), Z2.numpy()])
    axes_labels = pc_labels(pca)
    assert axes_labels == expected_axes, axes_labels
    names = [feats["class_names"][c] for c in classes]
    palette = class_palette(len(names))

    fig, axes = plt.subplots(1, 3, figsize=(19, 6.8))
    panels = ((xy0, "Original features z"), (xy1, "After Strategy 1 (ẑ)"), (xy2, "After Strategy 2 (ẑ)"))
    for ax, (xy, title) in zip(axes, panels):
        for j, name in enumerate(names):
            m = y == j
            ax.scatter(xy[m, 0], xy[m, 1], s=36, alpha=0.7, color=palette[j],
                       marker=MARKERS[j % len(MARKERS)], label=name)
        ax.set_title(title, fontsize=19)
        ax.set_xlabel(axes_labels[0], fontsize=16)
        ax.set_ylabel(axes_labels[1], fontsize=16)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=5,
               fontsize=15, frameon=True, markerscale=2.0)
    fig.suptitle(f"{dataset_label(ds)} — {encoder_label(enc, short=True)}: features before and after "
                 f"the Stage-3 FM (joint PCA, seed {seed})", fontsize=19, y=1.02)
    fig.tight_layout()
    show(fig)


feature_figure("fgvc_aircraft", "dinov2_vits14", ("PC1 (20.6% var.)", "PC2 (12.4% var.)"))
"""

AFTER = r"""
- The first two components explain 33.0% of the variance.
- The visible movement is subtle despite the +2.50-point accuracy gain.
- Useful movement may occur along classifier-relevant directions outside the displayed plane; the projection does not establish the mechanism.
"""

CELLS = [
    ("markdown", HEAD),
    ("code", CODE),
    ("markdown", AFTER),
]
