# ADR 0001 — Frozen pretrained encoders as feature extractors

**Status:** accepted (2026-07-17); encoder list updated 2026-07-24 for ADR 0004.

**Context.** Stage 1 establishes classification baselines over pretrained representations; Stages 2–3 insert a Flow Matching model on top of the same representations. The specification requires publicly available pretrained checkpoints, each used with its own associated preprocessing, and states that all encoder parameters must remain frozen.

**Decision.** Use frozen pretrained encoders only, never fine-tuned:

| Encoder | Source | Representation | Used for |
|---|---|---|---|
| ResNet-18 | torchvision `ResNet18_Weights.IMAGENET1K_V1` | 512-d, before the final classification layer | all datasets, linear probe + image prototypes |
| DINOv2 ViT-S/14 | `facebook/dinov2-small` | final class token (384-d) | FGVC-Aircraft, linear probe + image prototypes |
| CLIP RN50 | official OpenAI `clip` package | image encoder (1024-d) + text encoder | zero-shot branch only, per spec |

Train, validation and test features are extracted once per (dataset, encoder) and cached to `results/features/*.pt`; every classifier is trained and evaluated on the caches.

**Consequences.** Measured differences are attributable to the classification method and the representation, not to encoder adaptation. Caching makes the whole experiment grid re-runnable in minutes and guarantees that every head sees bit-identical features. The cost is a ceiling on absolute accuracy — especially on the fine-grained FGVC-Aircraft task with ImageNet-supervised ResNet-18 features — which is a discussion point rather than a defect.
