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
**Accuracy vs. shots:** compare each backbone's linear probe (diamond markers) with its own prototype heads (circles/squares, same color). On CLIP the probe leads everywhere; on DINOv2 the prototype leads at K=1 — the probe-vs-prototype ranking is encoder-dependent (§6–7).
"""),
    ("code", """
for ds in ("mnist", "cifar10", "mini_imagenet"):
    display(Image(str(REPO / "results" / "figures" / f"acc_vs_k_episodic_{ds}.png"), width=820))
"""),
    ("markdown", "## 5b · Results — simple all-classes protocol (mean ± std, 10 seeds)"),
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
for ds in ("mnist", "cifar10"):
    display(Image(str(REPO / "results" / "figures" / f"acc_vs_k_simple_{ds}.png"), width=820))
"""),
    ("markdown", """
### Multi-prototype ablation: does splitting a class into n k-means centers help?

**Not at 5-shot** — paired against the single prototype on each dataset's selected backbone: MNIST tie (−0.32 ± 0.39 for n=2, −0.12 ± 0.43 for n=3), CIFAR-10 significantly worse (−0.94 ± 0.20 / −1.64 ± 0.23), Mini-ImageNet marginally worse. With 5 support samples per class, each center gets ~5/n points and estimation noise dominates. **The reversal at MNIST 10-shot**: 3 centers win by a paired +1.18 ± 1.08 over 10 seeds — digit styles are genuinely multi-modal, but modeling that needs enough shots per center.
"""),
    ("code", """
display(Image(str(REPO / "results" / "figures" / "kmeans_ncenters.png"), width=760))
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
