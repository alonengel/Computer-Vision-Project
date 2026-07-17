CELLS = [
    ("markdown", """
## 7 · Discussion & handoff to Stage 2

Observations the tables above support (numbers auto-loaded from `results/metrics/`):

- **Frozen-embedding quality dominates.** Ranking across backbones follows embedding separability (§3, both the qualitative t-SNE and the quantitative metrics) — the classifier head matters less than the encoder. Note the setting throughout is *few-shot classification over frozen pretrained foundation-model embeddings*: near-ceiling Mini-ImageNet numbers are not comparable to the traditional few-shot literature, where encoders never see such broad pretraining. The strongest caveat is ResNet-50's ImageNet-1k *label* overlap with the R&L test classes; a weaker form (semantically overlapping pretraining data) applies to CLIP and DINOv2 too.
- **Zero-shot CLIP is a semantic reference baseline**, not an equivalent few-shot method: it consumes class *names* + pretrained image–text alignment, while the other heads consume labeled support *images*. It beats every CLIP-embedding support head on natural images (CIFAR-10, Mini-ImageNet) but is markedly weaker on MNIST digits — far from CLIP's training distribution — and the prompt ensemble even *hurts* there. A planned discussion point, not a bug.
- **Prototype vs. linear is encoder-dependent** (paired per-episode CIs, §6). Using CLIP embeddings the probe wins at every dataset × K; on DINOv2 the prototype wins five of six settings; on ResNet-50 the outcome is split. With a fixed a-priori training budget, no head is universally better — on MNIST's poorly separated embeddings extra shots favor the trained boundary, while on well-clustered embeddings the prototype is the more robust default, especially at K=1.

### Final embedding selection — the configurations Stage 2 must beat

Stage 1's purpose is **selection, not ranking**, and the selection is leakage-free: for each (dataset, head) **one** configuration is chosen across all K — by mean accuracy on **validation episodes** (Mini-ImageNet: the 16 R&L validation classes; MNIST/CIFAR-10: train-split episodes; seed 123) — with a paired per-episode check against the runner-up (statistical tie → the smaller embedding wins). Zero-shot CLIP is selected by *prompt variant* (single vs ensemble; K does not apply — it uses no support). The selected configuration's **test** numbers are then read out once as the Stage-2 reference (`results/artifacts/best_baselines.json`). ResNet-50 is excluded on Mini-ImageNet: the val classes are ImageNet-1k classes too, so selection would inherit the label contamination.
"""),
    ("code", """
import json

with open(REPO / "results" / "artifacts" / "best_baselines.json") as f:
    best = json.load(f)["selection"]
rows = []
for ds, heads in best.items():
    for head, info in heads.items():
        test = " / ".join(f"{k}: {100 * v['acc']:.2f} ± {100 * v['ci95']:.2f}"
                          for k, v in info["test_reference"].items())
        rows.append({"dataset": ds, "head": head,
                     "selected (on validation)": info["classifier"],
                     "val acc (mean over K)": f"{100 * info['val_acc_mean']:.2f}",
                     "margin vs runner-up": info["paired_margin_vs_runner_up"]
                     + (" (tie → smaller emb.)" if info["tie_break_smaller_embedding"] else ""),
                     "test reference (one-time read-out)": test})
display(pd.DataFrame(rows))
"""),
    ("markdown", """
### Advised Stage-2/3 architectures

Each variant is built on **its dataset's selected embedding** (table above) and must beat that configuration, paired on the same episode files:
"""),
    ("code", """
display(Image(str(REPO / "results" / "figures" / "arch_stage2_advised.png"), width=920))
"""),
    ("markdown", """

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
