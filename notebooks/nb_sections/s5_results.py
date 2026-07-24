CELLS = [
    ("markdown", """
## 5 · Results

### 5.1 Accuracy table

Top-1 accuracy (%) on the complete official test split, for every dataset, encoder, training-set size and implemented baseline. Generated programmatically from `results/metrics/summary.csv`. ‡ marks the dataset beyond the spec's required pair.
"""),
    ("code", """
from IPython.display import Markdown

display(Markdown((REPO / "results" / "metrics" / "accuracy_table.md").read_text(encoding="utf-8")))
"""),
    ("markdown", """
### 5.2 Accuracy versus training-set size

Error bars are ± std over the 3 runs. Zero-shot CLIP uses no labeled training images, so it appears as a horizontal reference line rather than a curve.
"""),
    ("code", """
for ds in ("dtd", "fgvc_aircraft", "flowers102"):
    display(Image(str(REPO / "results" / "figures" / f"acc_vs_trainsize_{ds}.png"), width=820))
"""),
    ("markdown", """
### 5.3 Linear-probe training curves

Representative 10-shot run (subset seed 0) for each dataset–encoder combination. The dashed line marks the epoch whose checkpoint was selected — the highest validation accuracy. These curves are the evidence that training is stable and show how much the validation loss diverges from the training loss, i.e. how much overfitting the probe incurs at this training-set size.
"""),
    ("code", """
import glob

for p in sorted(glob.glob(str(REPO / "results" / "figures" / "training_curves_*.png"))):
    display(Image(p, width=900))
"""),
    ("markdown", """
### 5.4 Confusion matrices

One **row-normalized** confusion matrix per dataset (rows sum to 1, so each row shows how the test images of one true class were distributed). The representative setting is the best full-training-split linear probe on that dataset — the configuration whose errors are most informative. The most frequent off-diagonal confusions are listed beside each matrix.
"""),
    ("code", """
for ds in ("dtd", "fgvc_aircraft", "flowers102"):
    display(Image(str(REPO / "results" / "figures" / f"confusion_{ds}.png"), width=980))
"""),
]
