# Stage 1 Report — Few-Shot Classification Baselines on Frozen Embeddings

**Project:** Few-Shot Classification with Flow Matching (Stage 1 of 3)
**Datasets:** MNIST · CIFAR-10 · Mini-ImageNet
**Date:** July 2026

---

## 1 · Introduction

Few-shot classification asks a model to recognize classes from a handful of labeled examples ("shots"). A strong and now-standard recipe is to *freeze* a pretrained encoder and learn only a lightweight decision rule over its embeddings. Stage 1 establishes rigorous baselines for exactly this setting; stages 2–3 will replace or augment the decision rule with a Flow Matching model and must be compared against these baselines on identical data.

We evaluate three classifier heads over frozen embeddings:

1. **Prototype classifier** (Snell et al., 2017 style): each class is represented by the mean of its support embeddings; queries are assigned to the nearest prototype (cosine similarity primary; negative squared Euclidean as an ablation).
2. **Linear probe**: a PyTorch `nn.Linear` layer trained with `CrossEntropyLoss` on the support set only.
3. **Zero-shot CLIP** (Radford et al., 2021): no support at all — queries are scored by cosine similarity between image embeddings and text embeddings of class-name prompts.

## 2 · Methods

### 2.1 Frozen backbones (ADR 0001)

| Backbone | Source | Embedding dim |
|---|---|---|
| CLIP ViT-B/32 (**primary**) | official OpenAI `clip` package | 512 |
| DINOv2 ViT-S/14 | `facebook/dinov2-small` (CLS token) | 384 |
| ResNet-50 (ImageNet-1k) | torchvision, penultimate layer | 2048 |

All backbones are frozen; embeddings are extracted once per (dataset, split, backbone) with each backbone's own preprocessing and cached to disk (fp32). MNIST images are replicated to 3 channels. The full classifier suite runs on CLIP embeddings; the prototype classifier is additionally compared across all three backbones.

### 2.2 Datasets and splits

- **MNIST / CIFAR-10** (torchvision): official train/test splits. Episodic evaluation samples from the test split; simple-protocol support sets are drawn from the train split and evaluated on the full test split.
- **Mini-ImageNet**: `timm/mini-imagenet` (100 ImageNet classes, 65,000 images), re-partitioned by *class* according to the canonical Ravi & Larochelle 64/16/20 split (committed as `config/mini_imagenet_splits.json`, with wnid → readable-name mapping for CLIP prompts). Episodic evaluation uses the 20 **test** classes (13,000 images, 650 per class), which are disjoint from any class used elsewhere.

### 2.3 Evaluation protocols

- **Episodic**: 5-way, K ∈ {1, 5} shots, 15 queries per class, 600 episodes. Support and query sets are disjoint by construction. Reported: mean accuracy ± 95% CI (1.96 · s/√600 over per-episode accuracies). For MNIST/CIFAR-10, episodic support *and* query are drawn from the test split — standard for episodic evaluation and leakage-free since nothing is trained or tuned on them beyond the per-episode support fit; the simple protocol below instead draws support from the train split (the asymmetry is deliberate and protocol-conventional).
- **Simple all-classes K-shot** (MNIST/CIFAR-10): K ∈ {1, 5, 10} support images per class drawn from the train split, full 10,000-image test set, 10 seeds. Reported: mean ± sample std over seeds. Zero-shot CLIP ignores support, so it is reported once per dataset (it is exactly K- and seed-independent).

**Cross-method comparisons.** Because all methods share identical episodes (ADR 0002), any claim of the form "method A beats method B" — in this stage or when stages 2–3 compare against these baselines — is to be made with **paired** statistics: a CI on the per-episode accuracy *differences*, not by eyeballing overlap of two marginal CIs (overlapping marginal CIs do not imply non-significance, and paired CIs are far tighter). The saved raw per-episode arrays make this free.

**Fixed episodes (ADR 0002).** All support/query indices were sampled once (seed 42; simple-protocol seeds 0–9) and committed to `results/artifacts/episodes/`. Every classifier in every stage of this project is evaluated on these exact files.

### 2.4 Classifier details

- **Prototype**: no training; prototypes recomputed per episode from support only.
- **Linear probe**: linear head (weights + bias as in `nn.Linear`), CrossEntropyLoss, Adam (300 steps, lr 0.01), trained on support embeddings only. For episodic evaluation the 600 per-episode heads are optimized jointly as one batched tensor `[600, C, dim]` with the loss summed over episodes and scaled by 1/S — each head then receives exactly its independent-head mean-CE gradient, so this is equivalent to (and two orders of magnitude faster than) 600 independent heads. Weight init is 0.01·N(0,1) (not `nn.Linear`'s default Kaiming-uniform), fixed a priori.
- **Zero-shot CLIP**: dataset-specific prompt templates (e.g. `'a photo of the number: "{}".'` for MNIST, `'a photo of a {}.'` for CIFAR-10/Mini-ImageNet); we report a single-prompt variant and a prompt-ensemble variant (mean of per-template normalized embeddings, re-normalized). In episodic mode only the episode's 5 classes are scored — the correct protocol for N-way comparability (every head faces the same 5-way decision), but note these numbers are therefore *not* comparable to full-label-set zero-shot accuracies in the literature.

### 2.4b Hyperparameter provenance

All hyperparameters were fixed **a priori**, before observing any test metric, and were not adjusted afterwards: the linear-probe budget (Adam, 300 steps, lr 0.01) follows standard frozen-feature linear-probe practice (cf. the linear-evaluation protocols in Radford et al., 2021 and Chen et al., 2020) and was carried over from prior coursework on other datasets; cosine similarity as the primary prototype metric is the modern default (with the classic Euclidean form of Snell et al., 2017 reported as an ablation); prompt templates are drawn from the published OpenAI CLIP prompt lists for MNIST/CIFAR/ImageNet-style datasets. The Mini-ImageNet validation classes (16 classes of the R&L split) were **not** used for any tuning in Stage 1; no hyperparameter was selected by observing accuracy on any test data, episodic or simple.

### 2.5 Reproducibility

Per-episode / per-seed raw accuracies are saved (`results/metrics/raw/`); `scripts/repro_check.py` re-derives every table number from these artifacts. Re-running experiments on the same GPU reproduces the raw arrays bit-exactly; CPU recomputation can flip isolated queries that are genuine float ties (top-2 cosine margins ~1e-7), so cross-device comparisons in stages 2–3 should use the committed arrays, not device re-runs. `results/runtime_summary.json` records versions and hardware (AMD RX 7900 XTX, torch 2.9.1+rocm7.2.1, Python 3.12.3). All randomness is seeded.

## 3 · Results

All numbers are re-derivable from the committed raw artifacts via `scripts/repro_check.py` (last run: 100/100 rows match). Figures referenced below live in `results/figures/`.

### 3.1 Episodic protocol

**5-way 1-shot (600 episodes, accuracy % ± 95% CI):**

| Classifier | MNIST | CIFAR-10 | Mini-ImageNet |
|---|---|---|---|
| Prototype (cosine) — CLIP | 56.56 ± 0.87 | 72.92 ± 0.85 | 89.15 ± 0.64 |
| Prototype (Euclidean) — CLIP | 56.64 ± 0.87 | 74.07 ± 0.82 | 90.09 ± 0.61 |
| Linear probe — CLIP | 59.96 ± 0.86 | 76.86 ± 0.80 | 91.40 ± 0.57 |
| Zero-shot CLIP (single prompt) | 59.61 ± 0.88 | 93.14 ± 0.28 | 99.00 ± 0.10 |
| Zero-shot CLIP (prompt ensemble) | 56.98 ± 0.89 | 93.44 ± 0.27 | 99.08 ± 0.10 |
| Prototype (cosine) — DINOv2 | 56.54 ± 0.92 | 77.34 ± 0.72 | 94.42 ± 0.47 |
| Prototype (Euclidean) — DINOv2 | 56.76 ± 0.92 | 75.30 ± 0.77 | 94.57 ± 0.47 |
| Prototype (cosine) — ResNet-50 | 55.96 ± 0.82 | 62.30 ± 0.76 | 97.38 ± 0.26 |
| Prototype (Euclidean) — ResNet-50 | 55.79 ± 0.83 | 57.12 ± 0.80 | 78.69 ± 1.13 |

**5-way 5-shot (600 episodes, accuracy % ± 95% CI):**

| Classifier | MNIST | CIFAR-10 | Mini-ImageNet |
|---|---|---|---|
| Prototype (cosine) — CLIP | 78.01 ± 0.60 | 89.16 ± 0.42 | 97.70 ± 0.20 |
| Prototype (Euclidean) — CLIP | 77.93 ± 0.60 | 89.23 ± 0.42 | 97.66 ± 0.20 |
| Linear probe — CLIP | 86.60 ± 0.44 | 90.82 ± 0.37 | 98.26 ± 0.16 |
| Zero-shot CLIP (single prompt) | 59.53 ± 0.89 | 93.03 ± 0.28 | 99.02 ± 0.11 |
| Zero-shot CLIP (prompt ensemble) | 55.96 ± 0.89 | 93.22 ± 0.27 | 99.10 ± 0.10 |
| Prototype (cosine) — DINOv2 | 75.61 ± 0.67 | 91.95 ± 0.37 | 98.56 ± 0.15 |
| Prototype (Euclidean) — DINOv2 | 75.60 ± 0.67 | 91.91 ± 0.36 | 98.01 ± 0.17 |
| Prototype (cosine) — ResNet-50 | 76.86 ± 0.70 | 81.68 ± 0.48 | 99.51 ± 0.08 |
| Prototype (Euclidean) — ResNet-50 | 76.87 ± 0.69 | 81.21 ± 0.48 | 99.29 ± 0.10 |

Figures: `bars_episodic_5shot.png` (main comparison), `bars_backbones.png` (backbone comparison), `acc_vs_k_episodic_*.png`.

### 3.2 Simple all-classes protocol (MNIST / CIFAR-10)

Accuracy % on the full 10,000-image test set (10 seeds, ± sample std):

| Classifier | MNIST K=1 | MNIST K=5 | MNIST K=10 | CIFAR-10 K=1 | CIFAR-10 K=5 | CIFAR-10 K=10 |
|---|---|---|---|---|---|---|
| Prototype (cosine) — CLIP | 44.31 ± 5.75 | 66.53 ± 2.70 | 73.32 ± 1.88 | 61.07 ± 6.01 | 83.05 ± 1.84 | 86.26 ± 1.80 |
| Prototype (Euclidean) — CLIP | 44.60 ± 5.66 | 66.37 ± 2.72 | 73.22 ± 1.95 | 62.36 ± 5.48 | 83.02 ± 1.88 | 86.25 ± 1.72 |
| Linear probe — CLIP | 49.95 ± 5.47 | 79.70 ± 1.41 | 88.62 ± 1.41 | 66.69 ± 4.59 | 86.37 ± 1.36 | 88.93 ± 0.79 |
| Prototype (cosine) — DINOv2 | 44.17 ± 4.16 | 64.10 ± 4.23 | 70.08 ± 1.57 | 65.36 ± 3.95 | 87.48 ± 1.62 | 90.25 ± 0.76 |
| Prototype (Euclidean) — DINOv2 | 44.52 ± 4.14 | 64.10 ± 4.19 | 70.03 ± 1.62 | 62.53 ± 4.22 | 87.69 ± 1.51 | 90.22 ± 0.88 |
| Prototype (cosine) — ResNet-50 | 43.16 ± 6.06 | 64.69 ± 4.73 | 72.68 ± 3.01 | 48.59 ± 3.03 | 69.28 ± 2.26 | 75.25 ± 2.16 |
| Prototype (Euclidean) — ResNet-50 | 42.96 ± 5.88 | 64.84 ± 4.63 | 72.70 ± 3.01 | 42.91 ± 3.41 | 68.83 ± 2.31 | 75.07 ± 2.42 |

Zero-shot CLIP (support- and therefore K/seed-independent): MNIST 48.25% (single) / 47.40% (ensemble); CIFAR-10 88.31% / 88.75% — consistent with commonly reported ViT-B/32 zero-shot CIFAR-10 results (≈ 89–90%, e.g. Radford et al. 2021 Table 9 and the LAION clip_benchmark; our prompt set is smaller than the full published ensemble). Figures: `acc_vs_k_simple_*.png` (zero-shot drawn as reference lines), `confusion_*_proto10s.png`, `failures_*.png`, `clip_zeroshot_*.png`.

## 4 · Discussion

**Embedding quality dominates head choice.** The spread across backbones (e.g. CIFAR-10 1-shot prototype: DINOv2 77.3 vs CLIP 72.9 vs ResNet-50 62.3) is larger than the spread across heads on a fixed backbone — consistent with the t-SNE panels (`tsne_*.png`), where DINOv2/CLIP separate CIFAR-10 classes cleanly and ResNet-50 does not.

**Zero-shot CLIP is a very strong baseline on natural images — and fails on MNIST.** On CIFAR-10 and Mini-ImageNet episodic tasks, zero-shot CLIP beats every head that shares its embedding space (93.2 / 99.1% on 5-shot episodes; 93.4 / 99.1% on 1-shot episodes), because its "prototypes" (text embeddings) suffer no 1-or-5-sample estimation noise. (The one head that nominally exceeds it — ResNet-50 prototypes at 99.51% on 5-shot Mini-ImageNet, paired diff +0.40 ± 0.11 vs. the ensemble — is the label-contaminated backbone discussed below, not an honest few-shot comparison.) On MNIST zero-shot collapses to 56–60% episodic / 48% all-classes: handwritten digits are far from CLIP's web-image training distribution. Notably, the prompt *ensemble* hurts MNIST (−2.6 pts episodic 1-shot, −0.85 all-classes): the generic templates dilute the digit-specific prompt — prompt engineering does not transfer blindly across domains.

**Prototype vs. linear probe.** The trained probe beats the prototype classifier at every dataset × K tested (paired per-episode CIs, §2.3), but how the gap evolves with K depends on embedding quality: on MNIST it grows (+3.40 ± 0.29 at 1-shot → +8.59 ± 0.41 at 5-shot), while on CIFAR-10 (+3.94 ± 0.34 → +1.66 ± 0.21) and Mini-ImageNet (+2.26 ± 0.30 → +0.56 ± 0.13) it shrinks — where embeddings already cluster tightly, five shots make class means near-optimal and training adds little; on MNIST's poorly separated embeddings, extra shots benefit the trained boundary more than the mean.

**The ResNet-50 Mini-ImageNet caveat.** ResNet-50's 97.4% 1-shot prototype accuracy is *not* evidence of a great few-shot method: the 20 R&L test classes are ImageNet-1k classes, so supervised ResNet-50 saw them, labeled, during pretraining. CLIP/DINOv2 numbers are the honest few-shot references here (neither trains on ImageNet labels). ResNet-50 also shows a 18.7-point cosine-vs-Euclidean gap on 1-shot Mini-ImageNet (97.38 vs 78.69) — its unnormalized feature magnitudes make Euclidean prototype distances noisy at K=1, a classic argument for cosine as the primary metric.

**Headroom for stages 2–3.** Mini-ImageNet 5-way is near ceiling (≥ 97% for most heads) and will not differentiate Flow-Matching variants; MNIST (78% prototype / 86.6% probe at 5-shot, 73/88.6% at all-classes 10-shot) and the CIFAR-10 support-based heads leave the clearest headroom. This is where stage 2/3 gains should be demonstrated.

## 5 · Limitations

- Frozen encoders bound absolute accuracy; no fine-tuning or adaptation is attempted (by design — the head is the object of study).
- Episodic results on MNIST/CIFAR-10 sample 5-way tasks from only 10 classes, so episodes share classes (standard, but CIs are correlated across episodes).
- Mini-ImageNet's simple protocol is omitted: its R&L test pool has no train/test image split; adding one would deviate from the canonical protocol.

## 6 · Extensions beyond the specification

| Extension | Justification |
|---|---|
| Backbone comparison (CLIP vs DINOv2 vs ResNet-50) | isolates embedding quality from head choice |
| Euclidean prototype ablation | classic ProtoNet uses Euclidean; cosine is the modern default — both reported |
| Prompt-ensemble ablation for zero-shot CLIP | quantifies prompt engineering's contribution |
| Committed episode index files | exact (not just statistical) comparability for stages 2–3 |
| Independent repro check from raw artifacts | table numbers are verifiable without re-running experiments |
| Episodic protocol also run on MNIST/CIFAR-10 | the spec assigns them the simple protocol only; episodic runs added for cross-dataset comparability with Mini-ImageNet |
| Episode/feature integrity fingerprints + pinned dataset revision | any upstream dataset change fails loudly instead of silently corrupting labels |
