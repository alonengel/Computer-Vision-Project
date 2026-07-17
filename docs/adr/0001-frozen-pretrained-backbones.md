# ADR 0001 — Frozen pretrained backbones as feature extractors

**Status:** accepted (2026-07-17)

**Context.** Stage 1 needs embeddings for few-shot classification on MNIST, CIFAR-10 and Mini-ImageNet; stages 2–3 insert a Flow Matching model on top of the same embeddings.

**Decision.** Use frozen pretrained encoders only — CLIP ViT-B/32 (primary, official OpenAI `clip` package), DINOv2 ViT-S/14, and ImageNet ResNet-50 (penultimate layer) for a backbone comparison. Never fine-tune. Embeddings are extracted once and cached to `results/features/*.pt`.

**Consequences.** Few-shot results measure the classifier head, not encoder adaptation — the quantity the project actually studies. Cached embeddings make all later experiments (including stages 2–3) cheap and guarantee every method sees identical features. The cost is a ceiling on absolute accuracy (notably CLIP on MNIST digits), which is a discussion point rather than a flaw.
