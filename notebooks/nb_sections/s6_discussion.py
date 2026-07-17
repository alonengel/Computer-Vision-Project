CELLS = [
    ("markdown", """
## 6 · Discussion & handoff to Stage 2

Observations the tables above support (numbers auto-loaded from `results/metrics/`):

- **Frozen-embedding quality dominates.** Ranking across backbones follows embedding separability (§3) — the classifier head matters less than the encoder.
- **Zero-shot CLIP vs. MNIST.** CLIP zero-shot is strong on natural images (CIFAR-10) but markedly weaker on MNIST digits — handwritten digits are far from CLIP's training distribution; prompt engineering only partially closes the gap. A planned discussion point, not a bug.
- **Prototype vs. linear.** With very few shots the prototype classifier is competitive with (or beats) the trained linear probe; the probe catches up as K grows.

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
