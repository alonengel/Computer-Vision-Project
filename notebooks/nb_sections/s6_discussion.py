CELLS = [
    ("markdown", """
## 6 · Discussion & handoff to Stage 2

Observations the tables above support (numbers auto-loaded from `results/metrics/`):

- **Frozen-embedding quality dominates.** Ranking across backbones follows embedding separability (§3) — the classifier head matters less than the encoder. Caveat: ResNet-50's near-perfect Mini-ImageNet scores reflect ImageNet-1k *label* overlap with the R&L test classes, not few-shot skill.
- **Zero-shot CLIP vs. MNIST.** CLIP zero-shot beats every CLIP-embedding head on natural images (CIFAR-10, Mini-ImageNet) but is markedly weaker on MNIST digits — handwritten digits are far from CLIP's training distribution, and the prompt ensemble even *hurts* there. A planned discussion point, not a bug.
- **Prototype vs. linear.** The trained probe beats the prototype at every dataset × K tested (paired per-episode CIs). The gap grows with K on MNIST but shrinks on CIFAR-10/Mini-ImageNet: with well-separated embeddings, class means become near-optimal at K=5.

**Artifacts handed to Stage 2/3** (all committed or reproducible via `tasks.ps1 extract`):

| Artifact | Path |
|---|---|
| Fixed episode indices | `results/artifacts/episodes/*.pt` |
| Class prototypes (per dataset/split/backbone) | `results/artifacts/prototypes_*.pt` |
| CLIP text embeddings (primary + ensemble) | `results/artifacts/clip_text_*.pt` |
| Cached embeddings | `results/features/*.pt` (regenerable) |

Stage 2 replaces the decision layer with a Flow-Matching transport toward prototypes / text embeddings; Stage 3 inserts FM before the linear head, trained end-to-end — both evaluated on the **same episode files** for an exact comparison.
"""),
]
