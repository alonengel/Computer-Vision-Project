# Talk section 6 — training behaviour, one dataset at a time, re-rendered from the
# histories of record with large fonts and validation emphasised.


def NOTES(body):
    return ('\n<details><summary><b>Speaker notes · הערות למרצה (עברית)</b></summary>\n'
            '<div dir="rtl" style="text-align:right; line-height:1.65; margin:0.6em 1em;">\n'
            + body.strip() + '\n</div></details>\n')


HEAD_FGVC = r"""
## 6 · Training behaviour — FGVC-Aircraft / DINOv2 (validation-selected models, seed 0)
"""

CURVES_CODE = r"""
import matplotlib.pyplot as plt

S1_COLOR, S2_COLOR = "#0173B2", "#D55E00"        # colours of the figures of record
CURVES = REPO / "results" / "artifacts" / "curves_stage3"
RUNS3 = pd.read_csv(MET / "runs_stage3.csv")


def _history(ds, enc, head, seed=0):
    with open(CURVES / f"{ds}_{enc}_{head}_seed{seed}.json") as f:
        return json.load(f)


def _checkpoint_epoch(ds, enc, head, seed=0):
    r = RUNS3[(RUNS3["dataset"] == ds) & (RUNS3["encoder"] == enc)
              & (RUNS3["head"] == head) & (RUNS3["seed"] == seed)]
    return int(r["checkpoint_epoch"].iloc[0])


def talk_curves(ds, enc, seed=0):
    # Presentation rendering of the seed-0 training histories of record
    # (results/artifacts/curves_stage3/*.json): the SAME series as the figure of
    # record stage3_curves_<ds>_<enc>.png (Appendix A.5), drawn one dataset at a
    # time, larger fonts, validation emphasised, and the validation-selected
    # checkpoint of record (runs_stage3.csv) marked. Nothing is recomputed.
    plt.rcParams.update({"font.size": 15, "axes.titlesize": 19, "axes.labelsize": 18,
                         "xtick.labelsize": 16, "ytick.labelsize": 16,
                         "legend.fontsize": 14})
    fig, (ax_ce, ax_acc, ax_zoom) = plt.subplots(1, 3, figsize=(16, 5.8))
    for head, color, name in (("fm_s1", S1_COLOR, "Strategy 1"),
                              ("fm_s2", S2_COLOR, "Strategy 2")):
        h, ep = _history(ds, enc, head, seed), _checkpoint_epoch(ds, enc, head, seed)
        assert h["epoch"][ep] == ep
        acc_tr = [100 * v for v in h["train_pipeline_acc"]]
        acc_va = [100 * v for v in h["val_acc"]]
        ax_ce.plot(h["epoch"], h["train_pipeline_ce"], color=color, ls="--", lw=1.6,
                   alpha=0.65, label=f"{name} — train")
        ax_ce.plot(h["epoch"], h["val_ce"], color=color, lw=3.0,
                   label=f"{name} — validation")
        ax_acc.plot(h["epoch"], acc_tr, color=color, ls="--", lw=1.6, alpha=0.65,
                    label=f"{name} — train")
        ax_acc.plot(h["epoch"], acc_va, color=color, lw=3.0, label=f"{name} — validation")
        ax_zoom.plot(h["epoch"], acc_va, color=color, lw=3.0, label=f"{name} — validation")
        ax_zoom.plot([ep], [acc_va[ep]], ls="none", marker="*", ms=20, color=color,
                     mec="black", label=f"{name} — checkpoint: epoch {ep} ({acc_va[ep]:.2f}%)")
    ax_ce.set_title("pipeline cross-entropy")
    ax_ce.set_ylabel("cross-entropy")
    ax_acc.set_title("pipeline top-1 accuracy")
    ax_acc.set_ylabel("accuracy (%)")
    ax_zoom.set_title("validation accuracy — zoom")
    ax_zoom.set_ylabel("accuracy (%)")
    for ax in (ax_ce, ax_acc, ax_zoom):
        ax.set_xlabel("epoch")
    handles, labels = ax_acc.get_legend_handles_labels()      # shared by the two left panels
    ax_zoom.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=1, frameon=False)
    fig.tight_layout()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.36, ax_ce.get_position().y0 - 0.16))
    fig.suptitle(f"{dataset_label(ds)} — {encoder_label(enc, short=True)}: training "
                 f"behaviour of the validation-selected models, seed {seed} "
                 f"(dashed = train, solid = validation)", fontsize=19, y=1.02)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    display(Image(data=buf.getvalue()))


talk_curves("fgvc_aircraft", "dinov2_vits14")
"""

HEAD_DTD = r"""
### 6b · Training behaviour — DTD / ResNet-18 (validation-selected models, seed 0)
"""

DTD_CODE = r"""
talk_curves("dtd", "resnet18")
"""

AFTER = r"""
- **Both strategies fit the training set completely** — train accuracy ≈ 100%, train cross-entropy ≈ 0 (dashed lines), on both datasets.
- **Validation stays far lower** (≈ 50–56%): at K = 10 the pipeline is data-starved, so the difference between the strategies is a difference in **generalization**, not in fit.
- On FGVC-Aircraft, Strategy 2's validation curve sits above Strategy 1's for essentially the whole run; on DTD the two stay within about a point of each other. ★ marks the best-validation-accuracy checkpoint of record. Strategy-internal diagnostics (displacements, trust-region hit rate, losses): Appendix A.5.
"""

HEB = """
<p><b>הנקודה המרכזית:</b> שתי האסטרטגיות מגיעות ל-100% על סט האימון; ההבדל ביניהן הוא בהכללה (ולידציה).</p>
<p><b>מונחים בפשטות:</b> <b>Pipeline CE / accuracy</b> — ה-CE והדיוק של הצינור המלא (z → FM → probe קפוא), לא של ה-FM לבד. <b>Train (מקווקו) מול validation (מלא)</b>. <b>הכוכב</b> — ה-epoch של ה-checkpoint שנבחר לפי דיוק ולידציה מרבי (מטבלת הרשומות).</p>
<p><b>מה לומר על הגרפים:</b> פאנל שמאלי ואמצעי — הפער הענק בין train ל-validation (K = 10, מעט מאוד דאטה). הפאנל הימני — זום על הוולידציה: ב-FGVC הכתום (S2) מעל הכחול (S1) לאורך כל הריצה; ב-DTD העקומות קרובות (כנקודה אחת). זה מסביר למה ההפרש ב-DTD בתוך הרעש.</p>
<p><b>שאלה צפויה:</b> "מה הקפיצה בסוף של אסטרטגיה 1 ב-FGVC?" — <b>תשובה:</b> חוסר יציבות סופי (ערכים סופיים, לא NaN) של האובייקטיב הישיר סביב epoch ~195; ה-checkpoint שנבחר (epoch 86 ב-seed 0) הרבה לפני כן, ולכן זה לא משפיע על התוצאה. הקפיצה נראית גם בדיאגנוסטיקת התזוזה בנספח A.5.</p>
<p><b>זמן משוער:</b> ~3 דקות.</p>
"""

CELLS = [
    ("markdown", HEAD_FGVC),
    ("code", CURVES_CODE),
    ("markdown", HEAD_DTD),
    ("code", DTD_CODE),
    ("markdown", AFTER + NOTES(HEB)),
]
