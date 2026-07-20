CELLS = [
    ("markdown", """
# Stage 1 — Few-Shot Classification Baselines

**MNIST · CIFAR-10 · Mini-ImageNet | frozen pretrained embeddings**

This notebook presents the Stage 1 baselines of the project (see `_docs/bigPicture.txt`). Its purpose is **selection, not ranking**: find the strongest configuration of each classifier function per dataset — on validation data — so that Stages 2–3 (Flow Matching) are measured against the hardest possible baseline on identical episodes.

Classifier heads (each evaluated on CLIP ViT-B/32, DINOv2 ViT-S/14 and ResNet-50 embeddings):

1. **Prototype classifier** — class prototype = mean support embedding; nearest-prototype decision (cosine primary, Euclidean ablation), plus a **multi-prototype k-means ablation** (n centers per class from support only).
2. **Linear probe** — PyTorch `nn.Linear` trained with `CrossEntropyLoss` on the support set.
3. **Zero-shot CLIP** — a *semantic reference baseline*: image–text cosine similarity with dataset-specific prompts (single prompt + prompt-ensemble variants; uses class names, not support images).

Two evaluation protocols:

- **Episodic**: 5-way, K ∈ {1, 5}, 15 queries, 600 episodes → mean accuracy ± 95% CI. Mini-ImageNet uses the 20 Ravi & Larochelle *test* classes; MNIST/CIFAR-10 included for comparison.
- **Simple K-shot**: all classes, K ∈ {1, 5, 10} support per class (drawn from the train split), full test set, 10 seeds → mean ± std.

Everything downstream of the frozen encoders is reproducible from committed artifacts: fixed episode index files (`results/artifacts/episodes/`, fingerprint-verified), cached embeddings, train-split prototypes, CLIP text embeddings, and the final validation-based embedding selection (`results/artifacts/best_baselines.json`) — the exact inputs Stages 2–3 will reuse.
"""),
]
