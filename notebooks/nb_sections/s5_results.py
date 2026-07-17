CELLS = [
    ("markdown", "## 5 · Results — episodic protocol (mean ± 95% CI, 600 episodes)"),
    ("code", """
import pandas as pd

pd.set_option("display.precision", 2)
ep = pd.read_csv(REPO / "results" / "metrics" / "episodic.csv")
ep["accuracy"] = (100 * ep["acc"]).round(2).astype(str) + " ± " + (100 * ep["ci95"]).round(2).astype(str)
for k in sorted(ep["k_shot"].unique()):
    t = ep[ep["k_shot"] == k].pivot(index="classifier", columns="dataset", values="accuracy")
    print(f"\\n=== 5-way {k}-shot ===")
    display(t)
"""),
    ("code", """
display(Image(str(REPO / "results" / "figures" / "bars_episodic_5shot.png"), width=880))
display(Image(str(REPO / "results" / "figures" / "bars_backbones.png"), width=880))
for ds in ("mnist", "cifar10", "mini_imagenet"):
    display(Image(str(REPO / "results" / "figures" / f"acc_vs_k_episodic_{ds}.png"), width=760))
"""),
    ("markdown", "## 5b · Results — simple all-classes protocol (mean ± std, 10 seeds)"),
    ("code", """
si = pd.read_csv(REPO / "results" / "metrics" / "simple.csv")
si["accuracy"] = (100 * si["acc"]).round(2).astype(str) + " ± " + (100 * si["std"]).round(2).astype(str)
for ds in ("mnist", "cifar10"):
    t = si[si["dataset"] == ds].pivot(index="classifier", columns="k_shot", values="accuracy")
    print(f"\\n=== {ds} (all classes, full test set) ===")
    display(t)
for ds in ("mnist", "cifar10"):
    display(Image(str(REPO / "results" / "figures" / f"acc_vs_k_simple_{ds}.png"), width=760))
"""),
    ("markdown", "### Qualitative results"),
    ("code", """
for name in ("clip_zeroshot_cifar10.png", "clip_zeroshot_mini_imagenet.png",
             "confusion_mnist_proto10s.png", "confusion_cifar10_proto10s.png",
             "failures_mnist.png", "failures_cifar10.png"):
    display(Image(str(REPO / "results" / "figures" / name), width=820))
"""),
]
