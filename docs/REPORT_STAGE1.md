# Stage 1 Report — Classification Baselines on Frozen Pretrained Encoders

**Project:** CVLAB Summer Project — Stage 1
**Author:** Alon Engel
**Datasets:** DTD · FGVC-Aircraft · Oxford Flowers-102
**Specification:** `_docs/stage_1.pdf` (single source of truth, ADR 0004)
**Date:** July 2026

---

## Abstract

*[TO FILL after the full runs — setting, design, two or three headline findings, and the branch selected for Stage 2.]*

## 1 · Introduction

The goal of Stage 1 is a reliable and reproducible classification pipeline built on **frozen** pretrained encoders; no flow-matching component appears at this stage. The specification names three candidate baselines — linear probing, classification with image-derived class prototypes, and zero-shot CLIP classification with text-derived prototypes — and asks each group to implement the linear probe plus **one** of the two prototype branches. The selected prototype branch is carried into Stage 2, and the linear-probe setting into Stage 3.

We implement the required linear probe and **both** prototype branches (ADR 0005), so that the branch continuing into Stage 2 can be chosen from measured evidence rather than assumed. Concretely:

1. **Linear probe** (required) — a multiclass linear classifier $s = Wz + b$ over the frozen feature $z$, trained with softmax cross-entropy; only $W$ and $b$ are trained.
2. **Image-derived class prototypes** (branch A) — cosine classification against class means of $L_2$-normalized training features.
3. **Zero-shot CLIP** (branch B) — cosine classification against CLIP RN50 text prototypes built from class-name prompts; uses no labeled training images.

## 2 · Experimental setup

### 2.1 Datasets and official splits

The specification asks for two of three datasets; we run all three and mark the third (‡) as beyond the required pair (ADR 0005). All classes are used, with the **official** train / validation / test splits; DTD uses partition 1 and FGVC-Aircraft the `variant` annotation level. Training and validation splits are never merged.

*[TABLE: dataset split sizes — generated into `results/metrics/dataset_splits.csv`]*

The spec-selected pair is **DTD + FGVC-Aircraft**, chosen because their training splits (40 and ~33 images per class) make $K \in \{5, 10, \text{full}\}$ three genuinely distinct settings. Flowers-102's official training split holds exactly 10 images per class, so for that dataset the 10-shot setting *is* the full setting — a structural degeneracy we report rather than hide.

### 2.2 Training-set sizes and subset sampling

For every method that uses labeled training examples we evaluate $K \in \{5, 10, \text{full}\}$ training images per class. For $K \in \{5, 10\}$ a **balanced** subset is sampled from the official training split with seeds $\{0, 1, 2\}$; the resulting indices are committed to `results/artifacts/subsets/` so that every encoder and every head is trained on bit-identical images. The `full` setting is the complete official training split.

### 2.3 Frozen encoders (ADR 0001)

All encoder parameters remain frozen, and each checkpoint is used with its own associated preprocessing. Train, validation and test features are extracted once per (dataset, encoder) and cached; classifier training and evaluation only ever read the caches.

| Encoder | Source | Representation | Used on |
|---|---|---|---|
| ResNet-18 (ImageNet-1K) | torchvision `ResNet18_Weights.IMAGENET1K_V1` | 512-d, taken before the final classification layer | all three datasets |
| DINOv2 ViT-S/14 | `facebook/dinov2-small` | final class-token representation (384-d) | FGVC-Aircraft |
| CLIP RN50 | official OpenAI `clip` package | image encoder (1024-d) + text encoder | zero-shot branch only |

DINOv2 is applied to FGVC-Aircraft, the fine-grained task, where the contrast between ImageNet-supervised and self-supervised features is most informative. CLIP RN50 is used strictly for the zero-shot branch, as the specification restricts it.

### 2.4 Baselines

**Linear probe.** $s = Wz + b$ with softmax cross-entropy; only $W$ and $b$ are trained. The specification's suggested configuration was adopted unchanged and fixed a priori: AdamW, learning rate $10^{-3}$, weight decay $10^{-4}$, batch size 64, at most 200 epochs, and **checkpoint selection by highest validation accuracy**. No hyperparameter search was performed — the objective is a reasonable and stable probe, since the comparison that matters later concerns the Flow Matching layer.

**Image-derived class prototypes.** Features are $L_2$-normalized, then for every class $c$

$$\mu_c = \mathrm{normalize}\left(\frac{1}{|S_c|}\sum_{i \in S_c} \mathrm{normalize}(z_i)\right), \qquad \hat{y} = \arg\max_c \cos(z, \mu_c),$$

where $S_c$ is the selected **training** subset for class $c$. Nothing is trained, so the only run-to-run variation is the choice of training subset.

**Zero-shot CLIP.** One text prototype per class from the frozen CLIP RN50 text encoder, using the specified prompts — `a photo of a {class} texture` (DTD), `a photo of a {class} aircraft` (FGVC-Aircraft), `a photo of a {class} flower` (Flowers-102). Image and text embeddings are normalized and classified by $\hat{y} = \arg\max_c \cos(z, t_c)$. This branch consumes no labeled training images and therefore yields exactly one result per dataset.

### 2.5 Runs, seeds and evaluation

Every reported number is **top-1 accuracy on the complete official test split**. Each training-set size is run three times:

| Setting | The 3 runs vary | Held fixed | Reported as |
|---|---|---|---|
| 5-shot, 10-shot | balanced-subset seed ∈ {0,1,2} | classifier initialization | mean ± std |
| full — linear probe | initialization seed ∈ {0,1,2} | training set | mean ± std |
| full — image prototypes | — (deterministic) | — | single run |
| zero-shot CLIP | — (deterministic, no training images) | — | single run |

Each setting therefore has exactly one interpretable source of variance: training-subset sampling at 5/10-shot, initialization at full. Single-run settings are reported without a standard deviation rather than as "± 0.00".

The validation split is used **only** for linear-probe checkpoint selection. No hyperparameter, encoder, subset, or checkpoint was chosen by observing test accuracy.

### 2.6 Reproducibility

Balanced training subsets are saved as index files carrying a fingerprint of the training-split labels, so a dataset change fails loudly instead of silently shifting the data. Per-run accuracies are written to `results/metrics/raw/`, per-run details to `runs.csv`, and aggregates to `summary.csv`; `scripts/repro_check.py` re-derives every mean and standard deviation in this report from the raw arrays. `results/runtime_summary.json` records versions and hardware. The whole pipeline is re-runnable with `tasks.ps1 data | extract | run | tables | figures | check`.

## 3 · Results

*[TO FILL after the full runs: 3.1 accuracy table, 3.2 accuracy versus training-set size, 3.3 training curves, 3.4 row-normalized confusion matrices, 3.5 feature visualizations.]*

## 4 · Discussion

*[TO FILL — encoder comparison, probe vs prototypes as a function of training-set size, where zero-shot CLIP stands, overfitting evidence from the training curves, the main confusions, and the branch selected for Stage 2 with its justification.]*

## 5 · Limitations

- Frozen encoders bound absolute accuracy by design; no adaptation of the representation is attempted at this stage.
- The linear-probe configuration is the specification's suggested baseline, adopted without search. Better numbers are certainly reachable per dataset, but tuning was explicitly out of scope and would have to be done on the validation split.
- Flowers-102's official training split contains exactly 10 images per class, so its 10-shot and full settings coincide; its three 10-shot runs are identical by construction and carry zero spread.
- Two-dimensional feature projections are qualitative only: both PCA and t-SNE distort the geometry of the frozen feature spaces.

## 6 · Deviations from, and extensions beyond, the specification

| Item | Status | Justification |
|---|---|---|
| Third dataset (Flowers-102) | extension ‡ | the spec asks for two; the third is cheap, and its 10-images-per-class training split is an instructive degenerate case. The required pair (DTD + FGVC-Aircraft) is marked and can be read on its own. |
| Both prototype branches | extension | the spec asks for one; implementing both lets the Stage-2 branch be chosen on evidence. The exactly-compliant subset is the linear probe plus either branch. |
| Linear-probe configuration | as specified | AdamW / 1e-3 / 1e-4 / 64 / 200 epochs / best-val-accuracy checkpoint, unchanged. |
| DINOv2 coverage | as specified | one dataset (FGVC-Aircraft). |
| CLIP RN50 usage | as specified | zero-shot branch only; never used as a probe or prototype backbone. |
| Feature visualizations | as specified, both projections | PCA *and* t-SNE panels are shown; the spec allows either. |

## References

- Cimpoi, M., Maji, S., Kokkinos, I., Mohamed, S., & Vedaldi, A. (2014). *Describing Textures in the Wild* (DTD). CVPR 2014.
- Maji, S., Rahtu, E., Kannala, J., Blaschko, M., & Vedaldi, A. (2013). *Fine-Grained Visual Classification of Aircraft* (FGVC-Aircraft). arXiv:1306.5151.
- Nilsback, M.-E., & Zisserman, A. (2008). *Automated Flower Classification over a Large Number of Classes* (Oxford Flowers-102). ICVGIP 2008.
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition* (ResNet). CVPR 2016.
- Oquab, M., Darcet, T., Moutakanni, T., et al. (2023). *DINOv2: Learning Robust Visual Features without Supervision*. TMLR 2024.
- Radford, A., Kim, J. W., Hallacy, C., et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision* (CLIP). ICML 2021.
- Snell, J., Swersky, K., & Zemel, R. (2017). *Prototypical Networks for Few-shot Learning*. NeurIPS 2017.
- Loshchilov, I., & Hutter, F. (2019). *Decoupled Weight Decay Regularization* (AdamW). ICLR 2019.
