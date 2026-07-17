CELLS = [
    ("markdown", """
## 4 · Classifiers

All heads share one interface — `predict(Xs, ys, Xq) → scores [B, Q, C]` over cached embeddings — and are pure torch (differentiable), so stage 2/3 can insert a Flow-Matching module upstream without touching evaluation.

**Prototype** — class prototype $c_k = \\frac{1}{|S_k|}\\sum_{x \\in S_k} f(x)$; query scored by cosine similarity (primary) or negative squared Euclidean distance (ablation) to each prototype. No training.

**Linear probe** — `nn.Linear(dim, C)` + `CrossEntropyLoss`, Adam (300 steps, lr 0.01) on the support set only. Episodic evaluation trains all 600 episode heads jointly as a batched tensor `[600, C, dim]` — mathematically identical to 600 independent `nn.Linear` heads, ~100× faster.

**Zero-shot CLIP** — cached text embeddings of dataset-specific prompts (e.g. `'a photo of the number: "{}".'` for MNIST); query scored by image–text cosine similarity. Single-prompt vs. prompt-ensemble reported separately. In episodic mode only the 5 episode classes are scored.
"""),
]
