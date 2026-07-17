CELLS = [
    ("markdown", "## 5 · Results — episodic protocol (mean ± 95% CI, 600 episodes)"),
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
**How to read the next two charts:** zero-shot CLIP wins on both natural-image datasets but is the *worst* head on MNIST — the one dataset far from its training distribution. In the backbone chart, ResNet-50's Mini-ImageNet bar is inflated by ImageNet-label overlap (§6), not few-shot skill.
"""),
    ("code", """
display(Image(str(REPO / "results" / "figures" / "bars_episodic_5shot.png"), width=880))
display(Image(str(REPO / "results" / "figures" / "bars_backbones.png"), width=880))
"""),
    ("markdown", """
**Accuracy vs. shots:** watch the gap between the linear probe (green) and the prototype heads — it widens with K on MNIST but closes on CIFAR-10/Mini-ImageNet, where class means are already near-optimal at K=5.
"""),
    ("code", """
for ds in ("mnist", "cifar10", "mini_imagenet"):
    display(Image(str(REPO / "results" / "figures" / f"acc_vs_k_episodic_{ds}.png"), width=820))
"""),
    ("markdown", "## 5b · Results — simple all-classes protocol (mean ± std, 10 seeds)"),
    ("code", """
si = pd.read_csv(REPO / "results" / "metrics" / "simple.csv")
si["accuracy"] = (100 * si["acc"]).round(2).astype(str) + " ± " + (100 * si["std"]).round(2).astype(str)
si["method"] = si["classifier"].map(method_label)
for ds in ("mnist", "cifar10"):
    t = si[si["dataset"] == ds].pivot(index="method", columns="k_shot", values="accuracy")
    print(f"\\n=== {ds} (all classes, full test set; k_shot=0 = zero-shot) ===")
    display(t)
for ds in ("mnist", "cifar10"):
    display(Image(str(REPO / "results" / "figures" / f"acc_vs_k_simple_{ds}.png"), width=820))
"""),
    ("markdown", """
### Qualitative results

The confusion matrices show *where* the prototype head fails: 2/7/9 on MNIST, the cat↔dog and bird↔deer pairs on CIFAR-10. The zero-shot panels show the *least confident* CLIP predictions — even the hardest cases are two-way ambiguities. The failure galleries show the most confident misclassification per (true, predicted) pair.
"""),
    ("code", """
for name in ("confusion_mnist_proto10s.png", "confusion_cifar10_proto10s.png",
             "clip_zeroshot_cifar10.png", "clip_zeroshot_mini_imagenet.png",
             "failures_mnist.png", "failures_cifar10.png"):
    display(Image(str(REPO / "results" / "figures" / name), width=820))
"""),
]
