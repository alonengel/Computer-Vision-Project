CELLS = [
    ("markdown", """
## 4 · Classifiers

All heads share one interface — `predict(Xs, ys, Xq) → scores [B, Q, C]` over cached embeddings — and are pure torch (differentiable), so stage 2/3 can insert a Flow-Matching module upstream without touching evaluation.

**Prototype** — class prototype $c_k = \\frac{1}{|S_k|}\\sum_{x \\in S_k} f(x)$; query scored by cosine similarity (primary) or negative squared Euclidean distance (ablation) to each prototype. No training.

**Linear probe** — `nn.Linear(dim, C)` + `CrossEntropyLoss`, Adam (300 steps, lr 0.01) on the support set only. Episodic evaluation trains all 600 episode heads jointly as a batched tensor `[600, C, dim]` — provably identical to 600 independent heads and 373–455× faster (benchmark below: same seeded init, identical predictions on all 45,000 queries per config; full probe grid 9.1 min → 1.3 s).

**Zero-shot CLIP** — a *semantic reference baseline* based on class names and pretrained image–text alignment: cached text embeddings of dataset-specific prompts (e.g. `'a photo of the number: "{}".'` for MNIST), query scored by image–text cosine similarity. Unlike the two heads above it receives **no support images** — its information source is class names + CLIP pretraining, so it is a reference point rather than an equivalent few-shot method. Single-prompt vs. prompt-ensemble reported separately. In episodic mode only the 5 episode classes are scored.
"""),
    ("code", """
display(Image(str(REPO / "results" / "figures" / "bench_probe.png"), width=880))
"""),
]
