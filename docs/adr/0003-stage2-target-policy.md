# ADR 0003 — Stage 2 Flow-Matching targets must not use evaluation-split statistics

**Status:** accepted (2026-07-17, following cv-expert methodology review)

**Context.** Stage 2 trains a Flow Matching model to transport embeddings toward class representations. An early version of the extraction script cached class-mean prototypes computed from full *test* splits (including the 13,000 Mini-ImageNet test-class images) labeled as "stage-2 FM targets". Training FM against full-eval-split prototypes would leak test data into training, and no Stage 1 baseline sees that information (they compute prototypes from K support shots only) — the comparison would be unfair by construction.

**Decision.** Permissible Stage 2 target representations, in the episodic setting:
1. **Per-episode support prototypes** — recomputed from the same K support shots the baselines see (from the committed episode files);
2. **CLIP text embeddings** — class-name prompts, no image data at all.

Full-split class prototypes are cached **only from train splits** (`prototypes_mnist_train_*.pt`, `prototypes_cifar10_train_*.pt`) for the simple protocol / stage-3 training, where train-split usage is legitimate. No test-split prototype artifacts are produced or committed.

**Consequences.** Stage 2 comparisons against Stage 1 remain exactly fair (identical support information per episode). The Flow Matching model for Mini-ImageNet trains on the 64 R&L *train* classes (or via text embeddings) and is evaluated zero-shot-style on the 20 test classes, matching the few-shot paradigm.
