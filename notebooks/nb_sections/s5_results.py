CELLS = [
    ("markdown", """
## 5 · Results

### 5.1 Accuracy table

Top-1 accuracy (%) on the complete official test split, for every dataset, encoder, training-set size and implemented baseline. Generated programmatically from `results/metrics/summary.csv`; ‡ marks the dataset beyond our selected pair (the specification allows any two of the three). Statistics are the sample standard deviation (ddof = 1) over the 3 runs.
"""),
    ("code", """
from IPython.display import Markdown

display(Markdown((REPO / "results" / "metrics" / "accuracy_table.md").read_text(encoding="utf-8")))
"""),
    ("markdown", """
**External sanity check.** Our CLIP RN50 zero-shot results (DTD 39.8, FGVC-Aircraft 17.0, Flowers-102 63.6) sit just below the values commonly reported for this checkpoint (≈ 41.7, 19.3, 65.9 — Radford et al. 2021, Table 11, and the prompt-ensemble notebook in the official CLIP repository), which is what we expect from using the specification's single prompt rather than an ensemble of 80.

**Paired head comparison.** At a given (dataset, encoder, K) both supervised heads train on the *same* committed subset indices and are evaluated on the same test split, so per-seed differences are matched and far tighter than the marginal spreads. This is the statistic that actually supports the one prototype win:
"""),
    ("code", """
display(Markdown((REPO / "results" / "metrics" / "paired_heads_table.md").read_text(encoding="utf-8")))
"""),
    ("markdown", """
The single prototype win — DTD at K = 5 — is **+0.90 ± 0.14 with all three seeds agreeing in sign**, a much stronger statement than the overlapping marginal intervals (46.51 ± 0.64 vs 45.60 ± 0.64) would justify on their own. Everywhere else the linear probe wins, by a margin that grows with the training-set size and reaches −23.5 points on FGVC-Aircraft/DINOv2 at K = 10.

**Balanced accuracy.** Flowers-102's official test split is class-imbalanced (20–238 images per class), so its image-weighted top-1 and per-class macro accuracy differ; DTD and FGVC-Aircraft are balanced and unaffected. The spec asks for plain top-1 (reported above); this matters only when comparing top-1 against the row-normalized confusion matrix in §5.4.
"""),
    ("code", """
display(Markdown((REPO / "results" / "metrics" / "macro_accuracy_table.md").read_text(encoding="utf-8")))
"""),
    ("markdown", """
### 5.2 Accuracy versus training-set size

Error bars are ± sample std over the 3 runs; the full-split prototype point and the zero-shot line are single deterministic runs. Zero-shot CLIP uses no labeled training images, so it is a horizontal reference rather than a curve.

**What to look for:** on DTD prototypes lead at K = 5 and the probe overtakes them by K = 10. On FGVC-Aircraft the *encoder* separates the curves far more than the head does — the DINOv2 probe at K = 5 (36.5%) already matches the ResNet-18 probe on the full split (36.6%). On Flowers-102 the 10-shot and full points coincide because the official training split holds exactly 10 images per class.
"""),
    ("code", """
for ds in ("dtd", "fgvc_aircraft", "flowers102"):
    display(Image(str(REPO / "results" / "figures" / f"acc_vs_trainsize_{ds}.png"), width=820))
"""),
    ("markdown", """
### 5.3 Linear-probe training curves

Representative 10-shot run (subset seed 0) for each dataset–encoder combination. The dashed line marks the selected checkpoint (highest validation accuracy); the dotted green curve is validation accuracy on the right-hand axis.

**What to look for:** training is smooth and monotone everywhere — the probe is stable. On DTD and FGVC-Aircraft the validation *loss* bottoms out near epoch 20–30 and then rises while training loss goes to zero: the probe overfits at this training-set size, and checkpoint selection on validation accuracy is what absorbs it. On FGVC-Aircraft/DINOv2 the validation loss rises from ~epoch 25 while validation *accuracy* keeps creeping up to its peak at epoch 187 — the probe grows over-confident on examples it already gets wrong, which cross-entropy punishes and top-1 accuracy does not. Loss and accuracy are therefore *not* interchangeable checkpoint criteria.
"""),
    ("code", """
import glob

for p in sorted(glob.glob(str(REPO / "results" / "figures" / "training_curves_*.png"))):
    display(Image(p, width=980))
"""),
    ("markdown", """
**Is the specification's 200-epoch budget adequate?** 8 of the 36 probe runs peak at epoch ≥ 190 (4 of them the degenerate Flowers-102 runs), so the cap is mildly binding. Measuring late-training drift as *validation accuracy at the final epoch minus that at epoch 100*, on the four fully-saved curves:
"""),
    ("code", """
import json

rows = []
for p in sorted(glob.glob(str(REPO / "results" / "artifacts" / "curves" / "*.json"))):
    h = json.load(open(p))
    va = h["val_acc"]
    rows.append({"curve": Path(p).stem.replace("_10shot_seed0", ""),
                 "drift over last 100 epochs (pts)": round(100 * (va[-1] - va[99]), 2),
                 "best epoch": int(max(range(len(va)), key=lambda i: va[i]))})
display(pd.DataFrame(rows))
"""),
    ("markdown", """
Every value is within about one accuracy point of the plateau and comparable to the run-to-run spread; DTD in fact drifts slightly downward, having peaked at epoch 31. The suggested configuration therefore behaves reasonably at every training-set size and was **kept unchanged** — no deviation from the specification is reported.

### 5.4 Confusion matrices

One **row-normalized** matrix per dataset (rows sum to 1, so each row shows how one true class's test images were distributed). The representative setting is the full-split linear probe on that dataset's best encoder, with the encoder chosen by **validation** accuracy — never test — and the title reporting the accuracy of the run actually plotted.

**What to look for:** on DTD every one of the six most frequent confusions is a semantically adjacent texture pair (dotted ↔ polka-dotted, lined → banded, woven → braided, grid → meshed, stained → marbled). On FGVC-Aircraft errors concentrate within airframe families — variants of the same aircraft — which is exactly the difficulty the dataset was built to pose.
"""),
    ("code", """
for ds in ("dtd", "fgvc_aircraft", "flowers102"):
    display(Image(str(REPO / "results" / "figures" / f"confusion_{ds}.png"), width=1000))
"""),
]
