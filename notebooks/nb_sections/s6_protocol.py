CELLS = [
    ("markdown", """
## 6 · Protocol integrity & provenance

*(This section summarizes guarantees that live in the repository — ADRs, config, raw
artifacts — so the notebook is self-contained for review.)*

**Data separation.** Support/query indices per episode were sampled once, committed
(`results/artifacts/episodes/`), and are integrity-fingerprinted against the feature
caches at load time. Support and query sets are disjoint by construction; query labels
are used *only* for the final accuracy computation — never in training, prototype
construction, or any selection decision. For MNIST/CIFAR-10, simple-protocol support
comes from the train split and evaluation from the full test split.

**Stage 2 target policy (ADR 0003).** Every episodic class prototype is computed only
from that episode's support set, $c_k = \\frac{1}{K}\\sum_{x \\in S_k} f(x)$. The cached
`prototypes_*.pt` artifacts are computed from **train splits only** (MNIST/CIFAR-10)
and exist for train-split use; no eval-split statistic may ever become a Flow-Matching
training target. Stage 2 trainable modules for Mini-ImageNet will train on the 64 R&L
train classes, tune on the 16 validation classes, and evaluate on the 20 test classes.
The encoder stays frozen in Stages 2–3.

**Hyperparameter provenance.** All hyperparameters were fixed a priori and never
adjusted after observing any test metric: linear probe = Adam, 300 steps, lr 0.01,
weight decay 0, init 0.01·N(0,1) (seeded), full-batch on the support set, loss =
per-episode mean CE (sum over episodes / S), identical configuration for every
dataset and K. Prompt templates come from the published OpenAI CLIP lists. No
hyperparameter or selection decision uses test data: the final embedding selection
(§7) runs on **validation episodes** — the 16 R&L validation classes for
Mini-ImageNet (their canonical purpose) and train-split episodes for MNIST/CIFAR-10
(seed 123, disjoint from all test evaluation).

**Paired statistical comparisons.** All methods share identical episodes, so method
comparisons use the per-episode accuracy *differences* (mean ± 95% CI), which is far
more informative than comparing two marginal CIs. Positive = linear probe beats the
cosine prototype on the same episodes:
"""),
    ("code", """
import numpy as np

raw = REPO / "results" / "metrics" / "raw"
rows = []
for bb, bb_name in (("clip_vitb32", "CLIP"), ("dinov2_vits14", "DINOv2"), ("resnet50", "ResNet-50")):
    row = {"embeddings": bb_name}
    for ds in ("mnist", "cifar10", "mini_imagenet"):
        for k in (1, 5):
            a = np.load(raw / f"ep_{ds}_5w{k}s_linear__{bb}.npy")
            b = np.load(raw / f"ep_{ds}_5w{k}s_proto_cos__{bb}.npy")
            d = a - b
            ci = 1.96 * d.std(ddof=1) / np.sqrt(len(d))
            row[f"{ds} {k}s"] = f"{100 * d.mean():+.2f} ± {100 * ci:.2f}"
    rows.append(row)
display(pd.DataFrame(rows).set_index("embeddings"))
"""),
    ("markdown", """
On CLIP embeddings the probe wins every setting; on DINOv2 the prototype wins five of
six. The probe-vs-prototype ranking is **encoder-dependent** — see §7 for discussion.
(ResNet-50 × Mini-ImageNet cells are ImageNet-label-contaminated, as flagged in §5.)
"""),
]
