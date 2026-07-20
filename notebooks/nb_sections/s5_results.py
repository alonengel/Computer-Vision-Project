CELLS = [
    ("markdown", """
## 5 · Episodic results — 5-way, K ∈ {1, 5}, 600 fixed episodes

All numbers in this section come from `episodic.csv` (mean ± 95% CI over episodes). Every figure in this section shows **episodic results only**.
"""),
    ("code", """
import pandas as pd

from src.visualize import method_label

pd.set_option("display.precision", 2)
ep = pd.read_csv(REPO / "results" / "metrics" / "episodic.csv")
ep["accuracy"] = (100 * ep["acc"]).round(2).astype(str) + " ± " + (100 * ep["ci95"]).round(2).astype(str)
ep["method"] = ep["classifier"].map(method_label)
for k in sorted(ep["k_shot"].unique()):
    t = ep[ep["k_shot"] == k].pivot(index="method", columns="dataset", values="accuracy")
    print(f"\\n=== 5-way {k}-shot ===")
    display(t)
"""),
    ("markdown", """
**Overview:** zero-shot CLIP wins on both natural-image datasets but is the *worst* head on MNIST — the one dataset far from its training distribution. In the backbone chart, ResNet-50's Mini-ImageNet bar is inflated by ImageNet-label overlap (§6).
"""),
    ("code", """
display(Image(str(REPO / "results" / "figures" / "bars_episodic_5shot.png"), width=880))
display(Image(str(REPO / "results" / "figures" / "bars_backbones.png"), width=880))
"""),
    ("markdown", """
**Per-classifier episodic charts** (one line per embedding; colors consistent everywhere — blue CLIP, pink DINOv2, yellow ResNet-50):

- **Prototype**: backbone ranking is dataset-dependent; the ResNet-50 Mini-ImageNet lead is label contamination (†).
- **Linear probe**: strongest on CLIP embeddings for MNIST/CIFAR-10; note the probe-vs-prototype ranking is encoder-dependent (§6–7).
- **Zero-shot**: prompt variants compared on episode queries — no shots axis (zero-shot uses class names, not support images).
- **K-means multi-prototype**: at 5-shot, splitting 5 support samples into n centers never helps (each center gets ~5/n points).
"""),
    ("code", """
display(Image(str(REPO / "results" / "figures" / "ep_acc_prototype.png"), width=940))
display(Image(str(REPO / "results" / "figures" / "ep_acc_linear.png"), width=940))
display(Image(str(REPO / "results" / "figures" / "ep_zeroshot_variants.png"), width=700))
display(Image(str(REPO / "results" / "figures" / "ep_kmeans_ncenters.png"), width=760))
"""),
    ("markdown", """
## 5b · Simple K-shot results — all 10 classes, K ∈ {1, 5, 10} (MNIST / CIFAR-10)

All numbers in this section come from `simple.csv` (full 10,000-image test set, mean ± std over 10 support seeds). Every figure in this section shows **simple-protocol results only**; Mini-ImageNet has no simple protocol (its R&L test pool has no train/test image split).
"""),
    ("code", """
si = pd.read_csv(REPO / "results" / "metrics" / "simple.csv")
# Zero-shot CLIP is deterministic (no support sampling), so no std is reported
# for it — displaying "± 0.00" would misleadingly suggest a seed sweep.
si["accuracy"] = (100 * si["acc"]).round(2).astype(str) + " ± " + (100 * si["std"]).round(2).astype(str)
si.loc[si["k_shot"] == 0, "accuracy"] = (100 * si.loc[si["k_shot"] == 0, "acc"]).round(2).astype(str)
si["method"] = si["classifier"].map(method_label)
for ds in ("mnist", "cifar10"):
    t = si[si["dataset"] == ds].pivot(index="method", columns="k_shot", values="accuracy")
    print(f"\\n=== {ds} (all classes, full test set; k_shot=0 = zero-shot, "
          f"deterministic -> no std) ===")
    display(t)
"""),
    ("markdown", """
**Per-classifier simple-protocol charts** (same embedding colors as above):

- **Prototype / linear probe**: accuracy vs support size over K ∈ {1, 5, 10}; the probe's CLIP advantage widens with K.
- **Zero-shot**: single prompt vs ensemble on the full 10-class test set — the ensemble *hurts* on MNIST (generic templates dilute the digit-specific prompt).
- **K-means multi-prototype**: the K=10 column is where multi-modality finally pays — on MNIST, n=3 beats the single prototype by a paired +1.18 ± 1.08 over 10 seeds.
"""),
    ("code", """
display(Image(str(REPO / "results" / "figures" / "simple_acc_prototype.png"), width=800))
display(Image(str(REPO / "results" / "figures" / "simple_acc_linear.png"), width=800))
display(Image(str(REPO / "results" / "figures" / "simple_zeroshot_variants.png"), width=620))
display(Image(str(REPO / "results" / "figures" / "simple_kmeans_ncenters.png"), width=800))
"""),
    ("markdown", """
### Qualitative results (simple protocol, seed 0)

The confusion matrices show *where* the prototype head fails: 2/7/9 on MNIST, the cat↔dog and bird↔deer pairs on CIFAR-10. The zero-shot panels show the *least confident* CLIP predictions — even the hardest cases are two-way ambiguities. The failure galleries show the most confident misclassification per (true, predicted) pair.
"""),
    ("code", """
for name in ("confusion_mnist_proto10s.png", "confusion_cifar10_proto10s.png",
             "clip_zeroshot_cifar10.png", "clip_zeroshot_mini_imagenet.png",
             "failures_mnist.png", "failures_cifar10.png"):
    display(Image(str(REPO / "results" / "figures" / name), width=820))
"""),
]
