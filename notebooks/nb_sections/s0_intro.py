CELLS = [
    ("markdown", """
# Stage 1 — Few-Shot Classification Baselines

**MNIST · CIFAR-10 · Mini-ImageNet | frozen pretrained embeddings**

This notebook presents the Stage 1 baselines of the project (see `_docs/bigPicture.txt`):

1. **Prototype classifier** — class prototype = mean support embedding; nearest-prototype decision (cosine primary, Euclidean ablation).
2. **Linear probe** — PyTorch `nn.Linear` trained with `CrossEntropyLoss` on the support set.
3. **Zero-shot CLIP** — image–text cosine similarity with dataset-specific prompts (single prompt + prompt-ensemble ablation).

Two evaluation protocols:

- **Episodic**: 5-way, K ∈ {1, 5}, 15 queries, 600 episodes → mean accuracy ± 95% CI. Mini-ImageNet uses the 20 Ravi & Larochelle *test* classes; MNIST/CIFAR-10 included for comparison.
- **Simple K-shot**: all classes, K ∈ {1, 5, 10} support per class (drawn from the train split), full test set, 10 seeds → mean ± std.

Everything downstream of the frozen encoders is reproducible from committed artifacts: fixed episode index files (`results/artifacts/episodes/`), cached embeddings, class prototypes and CLIP text embeddings — the exact inputs stages 2–3 (Flow Matching) will reuse.
"""),
]
