# ADR 0004 — The Stage-1 specification document is the single source of truth

**Status:** accepted (2026-07-24). Supersedes ADR 0002 and the Stage-1 half of ADR 0003.

**Context.** Stage 1 was first built from `_docs/bigPicture.txt` (a short four-paragraph outline) before the official assignment document existed. On 2026-07-24 the course document `_docs/stage_1.pdf` ("CVLAB Summer Project — Stage 1: Classification Baselines") became available. It specifies a materially different experiment: different datasets, different encoders, official train/validation/test splits instead of sampled episodes, K ∈ {5, 10, full} training images per class, and only **two** of the three baselines per group.

**Decision.** `_docs/stage_1.pdf` is the sole source of truth for Stage 1. Where the earlier outline, prior ADRs, or the previous implementation conflict with it, the document wins. Concretely, Stage 1 is now:

| | Superseded (v1) | Current (spec) |
|---|---|---|
| Datasets | MNIST, CIFAR-10, Mini-ImageNet | DTD (partition 1), FGVC-Aircraft (`variant`), Oxford Flowers-102 |
| Protocol | 5-way episodic + all-classes K-shot | All classes, official train/val/test splits |
| Training sizes | K ∈ {1, 5, 10} | K ∈ {5, 10, full}, balanced subsets, seeds {0, 1, 2} |
| Encoders | CLIP ViT-B/32, DINOv2 ViT-S/14, ResNet-50 | ResNet-18 (ImageNet-1K), DINOv2 ViT-S/14, CLIP RN50 |
| Model selection | validation episodes | official validation split (linear-probe checkpointing) |
| Metric | mean ± 95% CI over episodes | top-1 on the complete official test split, mean ± std over 3 runs |
| Linear probe | Adam, 300 steps, lr 0.01 | AdamW, lr 1e-3, wd 1e-4, batch 64, ≤200 epochs, best-val-accuracy checkpoint |

**Consequences.** The v1 work is preserved, not deleted: git tag `stage1-v1-episodic`, branch `archive/stage1-v1-episodic` (both pushed to the remote), and a physical copy under `D:\_backups\`. Its artifacts (episode index files, old feature caches, metrics, figures) were removed from the working tree so nothing stale can be cited by mistake. `docs/LAB_NOTEBOOK.md` keeps the full chronological history of both versions.

Rules that survive unchanged because they are good practice rather than protocol details: frozen encoders (ADR 0001), features extracted and cached once, training-subset index files committed so every encoder and head sees identical images, all tunables in `config/config.json`, and no selection or tuning on test data.
