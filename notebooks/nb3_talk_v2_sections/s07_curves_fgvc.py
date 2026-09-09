# v2 talk — section 8: training behaviour, FGVC-Aircraft.

HEAD = r"""
## <span style="color:#1f3a5f;">7 · Training behaviour — FGVC-Aircraft / DINOv2</span>

Validation-selected models, subset seed 0. Strategy 1 in blue, Strategy 2 in orange; dashed = training, solid = validation.
"""

CODE = r"""
RUNS = pd.read_csv(MET / "runs_stage3.csv")


def _history(ds, enc, head, seed=0):
    with open(CURVES / f"{ds}_{enc}_{head}_seed{seed}.json") as f:
        return json.load(f)


def _checkpoint_epoch(ds, enc, head, seed=0):
    r = RUNS[(RUNS["dataset"] == ds) & (RUNS["encoder"] == enc)
             & (RUNS["head"] == head) & (RUNS["seed"] == seed)]
    return int(r["checkpoint_epoch"].iloc[0])


def training_figure(ds, enc, seed=0):
    # Training histories of record (results/artifacts/curves_stage3/*.json), the same
    # series as the figure of record stage3_curves_<ds>_<enc>.png (Appendix A.8),
    # drawn one dataset at a time with the validation-selected checkpoint of record
    # (runs_stage3.csv) marked.
    plt.rcParams.update({"font.size": 15, "axes.titlesize": 19, "axes.labelsize": 18,
                         "xtick.labelsize": 16, "ytick.labelsize": 16, "legend.fontsize": 14})
    fig, (ax_ce, ax_acc, ax_zoom) = plt.subplots(1, 3, figsize=(16.5, 6.0))
    for head, color, name in (("fm_s1", S1_COLOR, "Strategy 1"), ("fm_s2", S2_COLOR, "Strategy 2")):
        h, ep = _history(ds, enc, head, seed), _checkpoint_epoch(ds, enc, head, seed)
        assert h["epoch"][ep] == ep
        acc_train = [100 * v for v in h["train_pipeline_acc"]]
        acc_val = [100 * v for v in h["val_acc"]]
        ax_ce.plot(h["epoch"], h["train_pipeline_ce"], color=color, ls="--", lw=1.6, alpha=0.65,
                   label=f"{name} — training")
        ax_ce.plot(h["epoch"], h["val_ce"], color=color, lw=3.0, label=f"{name} — validation")
        ax_acc.plot(h["epoch"], acc_train, color=color, ls="--", lw=1.6, alpha=0.65,
                    label=f"{name} — training")
        ax_acc.plot(h["epoch"], acc_val, color=color, lw=3.0, label=f"{name} — validation")
        ax_zoom.plot(h["epoch"], acc_val, color=color, lw=3.0, label=f"{name} — validation")
        ax_zoom.plot([ep], [acc_val[ep]], ls="none", marker="*", ms=20, color=color, mec="black",
                     label=f"{name} — selected checkpoint: epoch {ep} ({acc_val[ep]:.2f}%)")
    ax_ce.set_title("pipeline cross-entropy")
    ax_ce.set_ylabel("cross-entropy")
    ax_acc.set_title("pipeline top-1 accuracy")
    ax_acc.set_ylabel("accuracy (%)")
    ax_zoom.set_title("validation accuracy — zoom")
    ax_zoom.set_ylabel("accuracy (%)")
    for ax in (ax_ce, ax_acc, ax_zoom):
        ax.set_xlabel("epoch")
    handles, labels = ax_acc.get_legend_handles_labels()
    ax_zoom.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=1, frameon=False)
    fig.tight_layout()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.36, ax_ce.get_position().y0 - 0.16))
    fig.suptitle(f"{dataset_label(ds)} — {encoder_label(enc, short=True)}: training behaviour of the "
                 f"validation-selected models (seed {seed})", fontsize=19, y=1.06)
    show(fig)


training_figure("fgvc_aircraft", "dinov2_vits14")
"""

AFTER = r"""
- Both strategies fit the training data almost completely (dashed lines: accuracy ≈ 100%, cross-entropy ≈ 0).
- Strategy 2 maintains higher validation accuracy through most of training (solid lines); ★ marks the validation-selected checkpoints.
"""

CELLS = [
    ("markdown", HEAD),
    ("code", CODE),
    ("markdown", AFTER),
]
