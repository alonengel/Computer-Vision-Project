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

Per-episode / per-seed raw accuracies are saved (`results/metrics/raw/`); `scripts/repro_check.py` re-derives every table number from these artifacts. `results/runtime_summary.json` records versions and hardware (AMD RX 7900 XTX, torch 2.9.1+rocm7.2.1, Python 3.12.3). All randomness is seeded.

## 3 · Results

> **[TO FILL after full runs]** — episodic tables (5-way 1-shot / 5-shot, all classifiers × datasets, ± 95% CI), simple-protocol tables (MNIST/CIFAR-10, K ∈ {1,5,10}, ± std), backbone comparison, and referenced figures from `results/figures/`.

## 4 · Discussion

> **[TO FILL]** — embedding quality vs. head choice; CLIP zero-shot's MNIST weakness; prototype vs. linear as K grows; prompt-ensemble effect; backbone ranking.

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
