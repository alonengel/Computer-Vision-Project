# Stage 1 Report — Classification Baselines on Frozen Pretrained Encoders

**Project:** CVLAB Summer Project — Stage 1
**Author:** Alon Engel
**Datasets:** DTD · FGVC-Aircraft · Oxford Flowers-102
**Specification:** `_docs/stage_1.pdf` (single source of truth, ADR 0004)
**Date:** July 2026

---

## Abstract

We establish reproducible classification baselines on frozen pretrained encoders for DTD, FGVC-Aircraft and Oxford Flowers-102, using all classes and the official train/validation/test splits with training-set sizes of 5, 10 and all available images per class. Three heads are compared on cached frozen features: the required linear probe, image-derived class prototypes, and zero-shot CLIP — the specification asks for the probe plus one prototype branch, and we implement both so the branch continuing into Stage 2 can be chosen from evidence. Two results dominate. First, the representation matters far more than the head: on FGVC-Aircraft, replacing ImageNet-supervised ResNet-18 with self-supervised DINOv2 ViT-S/14 lifts the full-split linear probe from 36.6% to 67.2%, and the DINOv2 probe trained on five images per class already equals the ResNet-18 probe trained on the entire split. Second, image prototypes are competitive only in the lowest-data regime — they lead at DTD 5-shot (46.5 vs 45.6) and lose everywhere else, by a margin that widens with training-set size. Zero-shot CLIP RN50, which uses no labeled training images, reaches 63.6% on Flowers-102, 39.8% on DTD and 17.0% on FGVC-Aircraft. Linear-probe training curves show clear overfitting absorbed by validation-accuracy checkpointing. On the measured headroom we recommend the image-prototype branch for Stage 2. All numbers are re-derivable from committed raw artifacts.

## 1 · Introduction

The goal of Stage 1 is a reliable and reproducible classification pipeline built on **frozen** pretrained encoders; no flow-matching component appears at this stage. The specification names three candidate baselines — linear probing, classification with image-derived class prototypes, and zero-shot CLIP classification with text-derived prototypes — and asks each group to implement the linear probe plus **one** of the two prototype branches. The selected prototype branch is carried into Stage 2, and the linear-probe setting into Stage 3.

We implement the required linear probe and **both** prototype branches (ADR 0005), so that the branch continuing into Stage 2 can be chosen from measured evidence rather than assumed. Concretely:

1. **Linear probe** (required) — a multiclass linear classifier $s = Wz + b$ over the frozen feature $z$, trained with softmax cross-entropy; only $W$ and $b$ are trained.
2. **Image-derived class prototypes** (branch A) — cosine classification against class means of $L_2$-normalized training features.
3. **Zero-shot CLIP** (branch B) — cosine classification against CLIP RN50 text prototypes built from class-name prompts; uses no labeled training images.

## 2 · Experimental setup

### 2.1 Datasets and official splits

The specification allows any two of the three datasets; we run all three and mark the third (‡) as beyond our selected pair (ADR 0005). All classes are used, with the **official** train / validation / test splits; DTD uses partition 1 and FGVC-Aircraft the `variant` annotation level. Training and validation splits are never merged.

| Dataset | Classes | Train | Validation | Test | Train images per class |
|---|---|---|---|---|---|
| DTD (partition 1) | 47 | 1,880 | 1,880 | 1,880 | 40 |
| FGVC-Aircraft (`variant`) | 100 | 3,334 | 3,333 | 3,333 | 33–34 |
| Oxford Flowers-102 ‡ | 102 | 1,020 | 1,020 | 6,149 | 10 |

(Generated into `results/metrics/dataset_splits.csv`. Flowers-102's test split is class-imbalanced, 20–238 images per class, as published.)

Our selected pair is **DTD + FGVC-Aircraft**, chosen because their training splits (40 and ~33 images per class) make $K \in \{5, 10, \text{full}\}$ three genuinely distinct settings. Flowers-102's official training split holds exactly 10 images per class, so for that dataset the 10-shot setting *is* the full setting — a structural degeneracy we report rather than hide. The pair was provisional until the runs completed; §4 records its confirmation on the evidence (ADR 0006).

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
| 5-shot, 10-shot | balanced-subset seed ∈ {0,1,2} | classifier seed (initialization and batch order) | mean ± std |
| full — linear probe | classifier seed ∈ {0,1,2}, driving both weight initialization and minibatch ordering | training set (the complete official split) | mean ± std |
| full — image prototypes | — (deterministic) | — | single run |
| zero-shot CLIP | — (deterministic, no training images) | — | single run |

Each setting therefore has exactly one interpretable source of variance: training-subset sampling at 5/10-shot, and classifier stochasticity at full. Note that one seed governs both the weight initialization and the epoch shuffling, so the "full" spread measures optimization stochasticity as a whole rather than initialization in isolation. Statistics are the sample standard deviation (ddof = 1) over the 3 runs; single-run settings are reported without a standard deviation rather than as "± 0.00".

The validation split is used **only** for linear-probe checkpoint selection. No number in the accuracy table was influenced by test accuracy: no hyperparameter, training subset, checkpoint, or encoder used to produce a reported result was chosen by observing the test split, and where a representative encoder had to be picked for a figure it was picked by **validation** accuracy (`scripts/make_figures.py`).

Decisions that feed forward into Stage 2/3 are likewise made without test data: the representative encoder for each figure and the Stage-2 branch selection in §4 are argued from **validation** accuracies (recorded in `runs.csv` for both heads) and from methodological considerations, never from test accuracy. The test split's only role is the one-time read-out reported in §3.

### 2.6 Reproducibility

Balanced training subsets are saved as index files carrying a fingerprint of the training-split labels, so a dataset change fails loudly instead of silently shifting the data. Per-run accuracies are written to `results/metrics/raw/`, per-run details to `runs.csv`, and aggregates to `summary.csv`; `scripts/repro_check.py` re-derives every mean and standard deviation in this report from the raw arrays. `results/runtime_summary.json` records versions and hardware. The whole pipeline is re-runnable with `tasks.ps1 data | extract | run | tables | figures | check`.

## 3 · Results

Every number below is top-1 accuracy (%) on the complete official test split, re-derivable from the committed raw arrays via `scripts/repro_check.py` (last run: 27/27 summary rows match).

### 3.1 Accuracy table

| Dataset | Encoder | Baseline | K = 5 | K = 10 | full train split | no training images |
|---|---|---|---|---|---|---|
| DTD | ResNet-18 | Linear probe | 45.60 ± 0.64 | **54.31 ± 0.28** | **62.84 ± 0.45** | — |
| DTD | ResNet-18 | Image prototypes | **46.51 ± 0.64** | 53.37 ± 1.04 | 58.78 | — |
| DTD | CLIP RN50 | Zero-shot CLIP | — | — | — | 39.79 |
| FGVC-Aircraft | ResNet-18 | Linear probe | 19.90 ± 1.72 | 27.36 ± 0.83 | 36.62 ± 0.27 | — |
| FGVC-Aircraft | ResNet-18 | Image prototypes | 16.04 ± 0.83 | 19.85 ± 0.47 | 25.20 | — |
| FGVC-Aircraft | DINOv2 | Linear probe | **36.47 ± 0.70** | **51.30 ± 0.95** | **67.21 ± 0.11** | — |
| FGVC-Aircraft | DINOv2 | Image prototypes | 23.11 ± 1.36 | 27.80 ± 1.17 | 34.41 | — |
| FGVC-Aircraft | CLIP RN50 | Zero-shot CLIP | — | — | — | 17.04 |
| Oxford Flowers-102 ‡ | ResNet-18 | Linear probe | **75.58 ± 0.84** | **83.22 ± 0.00** | **83.28 ± 0.11** | — |
| Oxford Flowers-102 ‡ | ResNet-18 | Image prototypes | 70.19 ± 0.14 | 75.22 ± 0.00 | 75.22 | — |
| Oxford Flowers-102 ‡ | CLIP RN50 | Zero-shot CLIP | — | — | — | 63.64 |

**Bold** marks the best supervised configuration in each (dataset, training-set size) column, applied programmatically in `scripts/make_tables.py`; zero-shot CLIP is excluded from that comparison because it uses no training images. 5-shot / 10-shot: mean ± sample standard deviation (ddof = 1) over the 3 training-subset seeds. Full linear probe: same statistic over the 3 initialization seeds. Full image prototypes and zero-shot CLIP are single deterministic runs and carry no spread. ‡ beyond our selected dataset pair (the specification allows any two of the three). The std of exactly 0.00 at Flowers-102 K = 10 is not rounding: that dataset's official training split holds exactly 10 images per class, so all three 10-shot subsets are the same images.

External sanity check: our CLIP RN50 zero-shot numbers (DTD 39.8, FGVC-Aircraft 17.0, Flowers-102 63.6) sit just below the values commonly reported for this checkpoint (≈ 41.7, 19.3, 65.9; Radford et al., 2021, Table 11, and the prompt-ensemble notebook in the official CLIP repository), consistent with our use of the specification's single prompt rather than an ensemble of 80.

**Paired head comparison.** At a given (dataset, encoder, K) the two supervised heads are trained on the *same* committed subset indices and evaluated on the same test split, so their per-seed differences are matched and much tighter than the marginal spreads above. Generated into `results/metrics/paired_heads_table.md`:

| Dataset | Encoder | K | Prototypes − probe (paired) | Seeds favouring prototypes |
|---|---|---|---|---|
| DTD | ResNet-18 | 5 | **+0.90 ± 0.14** | 3 / 3 |
| DTD | ResNet-18 | 10 | −0.94 ± 0.83 | 0 / 3 |
| FGVC-Aircraft | ResNet-18 | 5 | −3.86 ± 0.95 | 0 / 3 |
| FGVC-Aircraft | ResNet-18 | 10 | −7.51 ± 1.11 | 0 / 3 |
| FGVC-Aircraft | DINOv2 | 5 | −13.36 ± 0.81 | 0 / 3 |
| FGVC-Aircraft | DINOv2 | 10 | −23.49 ± 0.94 | 0 / 3 |
| Flowers-102 ‡ | ResNet-18 | 5 | −5.39 ± 0.95 | 0 / 3 |
| Flowers-102 ‡ | ResNet-18 | 10 | −8.00 ± 0.00 | 0 / 3 |

The single prototype win (DTD, K = 5) is +0.90 ± 0.14 with all three seeds agreeing in sign — a far stronger statement than the overlapping marginal intervals 46.51 ± 0.64 vs 45.60 ± 0.64 would support on their own. The `full` setting is excluded because the prototype head runs once there while the probe varies only by initialization, so those runs are not paired.

**Balanced accuracy.** Flowers-102's official test split is class-imbalanced (20–238 images per class), so its image-weighted top-1 and its per-class macro accuracy differ; DTD and FGVC-Aircraft have balanced test splits and are unaffected (`results/metrics/macro_accuracy_table.md`):

| Dataset | Encoder | Top-1 (%) | Balanced / macro (%) |
|---|---|---|---|
| DTD | ResNet-18 | 63.35 | 63.35 |
| FGVC-Aircraft | DINOv2 | 67.09 | 67.07 |
| Flowers-102 ‡ | ResNet-18 | 83.22 | 85.38 |

The specification asks for plain top-1, which is what the main table reports; this is supplementary and matters only when comparing the top-1 figure with the row-normalized (per-class) confusion matrix of Figure 8.

### 3.2 Accuracy versus training-set size

![Figure 1](../results/figures/acc_vs_trainsize_dtd.png)
*Figure 1 — DTD. Image prototypes lead at K = 5; the trained probe overtakes them by K = 10 and pulls clearly ahead on the full split. Zero-shot CLIP (dotted) sits below every supervised setting.*

![Figure 2](../results/figures/acc_vs_trainsize_fgvc_aircraft.png)
*Figure 2 — FGVC-Aircraft. The encoder, not the head, dominates: the DINOv2 probe with **5 images per class** (36.5%) already matches the ResNet-18 probe trained on the **full** split (36.6%). Zero-shot CLIP RN50 (17.0%) is competitive with 5-shot ResNet-18 prototypes.*

![Figure 3](../results/figures/acc_vs_trainsize_flowers102.png)
*Figure 3 — Flowers-102 ‡. The 10-shot and full points coincide because the official training split contains exactly 10 images per class; the apparent plateau is a property of the split, not of the method.*

### 3.3 Training curves

![Figure 4](../results/figures/training_curves_0.png)
![Figure 5](../results/figures/training_curves_1.png)
*Figures 4–5 — Representative 10-shot linear-probe runs (subset seed 0), one per dataset–encoder combination; the dashed line marks the selected checkpoint. Training is smooth and monotone in every case. On DTD and FGVC-Aircraft the validation loss reaches a minimum near epoch 20–30 and then rises while the training loss goes to zero — clear overfitting of the probe at this training-set size, which is precisely what checkpoint selection on validation accuracy is there to absorb.*

A detail worth noting in Figure 5 (FGVC-Aircraft / DINOv2): validation *loss* rises from about epoch 25 onward, yet the best validation *accuracy* occurs at epoch 187. Loss and accuracy diverge because the probe becomes increasingly over-confident on the examples it already gets wrong — cross-entropy penalises that, top-1 accuracy does not. Selecting on accuracy, as the specification requires, is therefore not the same as selecting on loss.

**Is the 200-epoch budget adequate?** 8 of the 36 probe runs reach their best validation accuracy at epoch ≥ 190, so the cap is mildly binding — though 4 of those 8 are the three duplicated Flowers-102 10-shot runs plus one Flowers-102 full run, i.e. the degenerate dataset. To quantify the residual head-room we define the late-training drift as *validation accuracy at the final epoch minus validation accuracy at epoch 100*, measured on the four representative 10-shot curves that are saved in full (`results/artifacts/curves/`; curves are stored only for those runs):

| Curve | Drift over the last 100 epochs | Best epoch |
|---|---|---|
| DTD — ResNet-18 | −0.21 | 31 |
| FGVC-Aircraft — ResNet-18 | +0.81 | 195 |
| FGVC-Aircraft — DINOv2 | +0.63 | 187 |
| Flowers-102 ‡ — ResNet-18 | +0.98 | 199 |

Every value is within about one accuracy point of the plateau, and comparable to the run-to-run spread of the corresponding setting (e.g. ± 1.72 for FGVC-Aircraft/ResNet-18 at K = 5). DTD in fact drifts slightly *downward*, having peaked at epoch 31. The suggested configuration therefore behaves reasonably at every training-set size and was **kept unchanged**; no deviation from the specification is reported.

### 3.4 Confusion matrices

![Figure 6](../results/figures/confusion_dtd.png)
*Figure 6 — DTD, row-normalized, full-split linear probe (encoder chosen by validation accuracy; the title reports the accuracy of the plotted run). All six of the most frequent confusions are semantically adjacent texture pairs: dotted → polka-dotted (40% of that class's test images), polka-dotted → dotted (25%), lined → banded (22%), woven → braided (20%), grid → meshed (20%), stained → marbled (20%).*

![Figure 7](../results/figures/confusion_fgvc_aircraft.png)
*Figure 7 — FGVC-Aircraft, row-normalized. Confusions concentrate within airframe families (variants of the same aircraft), which is the defining difficulty of this dataset.*

![Figure 8](../results/figures/confusion_flowers102.png)
*Figure 8 — Oxford Flowers-102 ‡, row-normalized. The matrix is strongly diagonal, consistent with the 83% top-1 accuracy.*

### 3.5 Feature visualizations

![Figure 9](../results/figures/features_fgvc_aircraft_dinov2_vits14.png)
*Figure 9 — FGVC-Aircraft, DINOv2 features for 10 classes with their image-derived prototypes (stars), PCA and t-SNE, projections fitted jointly to the plotted features and prototypes. Distinctive light aircraft (Cessna 172, DHC-1, Fokker 50) form tight, well-separated clusters, while the Boeing/Airbus narrow- and wide-bodies (737-300, 737-700, 767-300, A320, A340-200) overlap heavily — a direct visual account of where the remaining 33% of errors come from.*

![Figure 10](../results/figures/features_fgvc_aircraft_resnet18.png)
*Figure 10 — The same 10 classes, same test images and same colours, with ResNet-18 features. Cluster structure is markedly weaker than in Figure 9. Since both figures draw **image prototypes**, the matching quantity is the prototype-head gap on this dataset (34.41 vs 25.20, i.e. 9.2 points); the corresponding linear-probe gap on the same features is 30.6 points.*

In every figure PCA is presented as the primary view — deterministic, linear, with globally meaningful axes — and t-SNE as a supplementary one, read for local neighbourhood structure only: t-SNE does not preserve global distances or cluster sizes, and coordinates are not comparable across separately fitted panels. All t-SNE panels share one fixed configuration (perplexity 30, PCA initialization, seed 0).

Corresponding figures for the remaining dataset–encoder combinations are in `results/figures/features_*.png`, including the CLIP panels that show test-image embeddings together with their **text** prototypes. In those CLIP panels the ten text prototypes appear close to one another and separated from the image points, in both projections. This is *consistent with* the modality gap documented for CLIP-style models (image and text embeddings concentrating in different regions of the joint space; Liang et al., 2022), though a 2-D projection cannot establish that by itself — confirming it would require distance measurements in the original 1024-d space. It does not indicate a broken classifier either way: zero-shot classification depends only on the *relative* cosine ordering of a query against the text prototypes, not on absolute image–text proximity.

## 4 · Discussion

**The representation dominates the head.** On FGVC-Aircraft, swapping ResNet-18 for DINOv2 while holding everything else fixed moves the full-split linear probe from 36.62% to 67.21% — a 30.6-point gain, far larger than any difference between heads on a fixed encoder. The sharpest way to state it: the DINOv2 probe trained on **5 images per class** (36.47 ± 0.70) matches, within run-to-run noise, the ResNet-18 probe trained on the **entire** training split (36.62 ± 0.27) — a 0.15-point difference that these three-run spreads cannot resolve. Self-supervised features transfer to fine-grained recognition in a way ImageNet-supervised ResNet-18 features do not, and Figures 9–10 show the reason directly in the feature space.

**Prototypes win only in the low-data regime, and only once.** The image-prototype head beats the linear probe at exactly one of the twelve grid cells — DTD at K = 5, paired **+0.90 ± 0.14** with all three seeds agreeing in sign (§3.1). At DTD K = 10 the two heads are **statistically indistinguishable at n = 3** (paired −0.94 ± 0.83; per-seed −1.60 / 0.00 / −1.22, one exact tie, 95% t-interval [−3.01, +1.13] straddling zero). At every remaining cell the probe wins decisively and by a margin that grows with the training-set size, up to −23.49 ± 0.94 on FGVC-Aircraft/DINOv2 at K = 10. This is the expected behaviour: a class mean is a well-conditioned estimator when five examples are all one has, but it cannot exploit additional data the way a discriminatively trained boundary can, and it is blind to the fact that some feature directions separate classes better than others. The effect is strongest where classes are entangled *and* the representation is rich — with DINOv2 features on FGVC-Aircraft the prototype head recovers barely half the probe's accuracy (34.41 vs 67.21), whereas with the weaker ResNet-18 features on the same dataset it retains about 69% of it (25.20 vs 36.62).

**Zero-shot CLIP is a genuinely different trade-off.** It uses no labeled training images at all. On Flowers-102 it reaches 63.64%, within 12 points of a ResNet-18 probe trained on **five images per class** (75.58%) and 19.6 points below that probe trained on the full split (83.28%). On DTD it reaches 39.79%, below every supervised setting though only 5.8 points below the 5-shot probe. On FGVC-Aircraft it manages 17.04% — above 5-shot ResNet-18 prototypes (16.04%) — which says more about the difficulty of the task than about CLIP. The pattern is consistent with CLIP's pretraining distribution: flowers and textures are describable in natural language, aircraft *variants* essentially are not ("a photo of a 737-300 aircraft" carries little visual signal).

**Overfitting is real where it appears, and the protocol absorbs it.** Full training histories are retained only for the four representative 10-shot runs (`run_experiments.py` saves curves for `K = 10, seed 0`), so the evidence below is scoped to that training-set size. In three of those four — DTD/ResNet-18, FGVC-Aircraft/ResNet-18 and FGVC-Aircraft/DINOv2 — training loss goes to zero while validation loss reaches a minimum at epoch 30, 35 and 26 respectively and then rises: textbook overfitting, absorbed by checkpointing on validation accuracy. The fourth, Flowers-102/ResNet-18, does **not** overfit at this size: its validation loss decreases monotonically to the final epoch. The divergence between validation loss and validation accuracy on FGVC-Aircraft/DINOv2 (loss minimum at epoch 26, best accuracy at epoch 187) is a reminder that the two are not interchangeable selection criteria.

**Which branch carries into Stage 2 — selected on methodological grounds and validation data, never test accuracy.** We select **Option A, image-derived class prototypes**. Three of the four reasons are structural properties of the branches that require no accuracy numbers at all; the fourth is measured on the **validation** split only:

1. *Experimental surface (methodological).* Image prototypes depend on both the encoder and K, so Stage 2 inherits a grid of transport targets (3 training sizes × the cached encoders) rather than a single fixed text embedding per class — more settings in which to characterize the Flow Matching layer.
2. *Encoder freedom (methodological).* Option A can be built on any cached representation, including DINOv2 on the fine-grained task, whereas Option B is locked to CLIP RN50 by construction.
3. *Continuity with Stage 3 (methodological).* Option A reuses precisely the caches the linear probe uses, so the Stage-2 and Stage-3 comparisons rest on the same representation.
4. *Headroom, measured on validation.* On the validation split — never the test split — the full-split prototype head sits well below the full-split probe on the same features: 55.37 vs 60.71 on DTD/ResNet-18 (5.3 points), 32.91 vs 68.62 on FGVC-Aircraft/DINOv2 (35.7 points), 78.04 vs 86.50 on Flowers-102/ResNet-18 (8.5 points) (`runs.csv`, `val_acc`). That validation gap is the space a Flow Matching decision layer has room to close.

Both branches are implemented and reported, so this selection can be revisited without re-running anything.

**Which two datasets carry forward — confirmed on structural evidence, never test accuracy.** The specification asks for two of the three datasets; we ran all three (§1, ADR 0005) precisely so this choice could be made on evidence rather than blind. The completed runs confirm the provisional pair: **DTD + FGVC-Aircraft** (ADR 0006). The evidence is structural — properties of the official splits that the runs made concrete, not a ranking of test accuracies: (i) Flowers-102's training split holds exactly 10 images per class, so its 10-shot setting *is* its full setting — the deterministic prototype head scores identically at both (75.22 = 75.22), the probe's three 10-shot "subset seeds" select the same images and carry exactly zero spread (83.22 ± 0.00), and its accuracy-versus-training-size curve has two distinct points where the other datasets have three, which is precisely the axis Stage 2 argues along; (ii) DTD (40 images/class) and FGVC-Aircraft (~33/class) give three genuinely distinct K settings; (iii) FGVC-Aircraft carries the required DINOv2 encoder and the largest validation headroom (35.7 points); (iv) Flowers-102's class-imbalanced test split (20–238 images per class) makes its top-1 the least clean single number of the three (§3.1). Flowers-102 remains in this report as the ‡ extension; Stages 2 and 3 build on the selected pair.

**Stage-2/3 handoff.** The concrete configuration each later stage builds on, per dataset — encoder selected by validation accuracy of the full-split probe, prototype target fixed by the branch selection above, and the linear-probe baseline as the already-published one-time test read-out (generated into `results/metrics/handoff_table.md`):

| Dataset | Selected encoder (by validation) | Stage-2 prototype target (branch A) | Validation headroom (probe − prototypes, full) | Stage-3 baseline: linear probe, full split (test) |
|---|---|---|---|---|
| DTD | ResNet-18 | class-mean prototypes $\mu_c$ of the selected training subset, ResNet-18 features | 60.71 − 55.37 = 5.34 pts | 62.84 ± 0.45 |
| FGVC-Aircraft | DINOv2 | class-mean prototypes $\mu_c$ of the selected training subset, DINOv2 features | 68.62 − 32.91 = 35.70 pts | 67.21 ± 0.11 |
| Flowers-102 ‡ | ResNet-18 | class-mean prototypes $\mu_c$ of the selected training subset, ResNet-18 features | 86.50 − 78.04 = 8.46 pts | 83.28 ± 0.11 |

Stage 2 trains the Flow Matching model to transport frozen embeddings toward the selected training subset's class prototypes and must beat the corresponding prototype baseline of §3.1; Stage 3 inserts the module before the linear probe and must beat the baseline in the last column — both on the identical cached features and committed subset files.

## 5 · Limitations

- Frozen encoders bound absolute accuracy by design; no adaptation of the representation is attempted at this stage.
- The linear-probe configuration is the specification's suggested baseline, adopted without search. Better numbers are certainly reachable per dataset, but tuning was explicitly out of scope and would have to be done on the validation split.
- Flowers-102's official training split contains exactly 10 images per class, so its 10-shot and full settings coincide; its three 10-shot runs are identical by construction and carry zero spread.
- Two-dimensional feature projections are qualitative only: both PCA and t-SNE distort the geometry of the frozen feature spaces.
- One seed governs both the weight initialization and the minibatch ordering of the linear probe, so the "full" spread measures optimization stochasticity as a whole rather than initialization in isolation.
- The official splits themselves contain a small number of content-identical images across split boundaries (verified by hashing feature rows: one train↔test pair in DTD partition 1, `dotted_0143` ≡ `dotted_0133`, plus six val↔test pairs; one train↔test pair in Flowers-102). Filename overlap between splits is exactly zero, so this is a property of the published datasets the specification mandates rather than a pipeline fault, and at 1/1880 and 1/6149 the effect on the reported accuracies is negligible — but it is stated rather than presented as a perfectly clean partition.
- `results/features/` is gitignored, so an external reviewer can re-derive every number from the committed metrics and prediction artifacts, but re-deriving the prototype and zero-shot accuracies from *images* requires re-running `tasks.ps1 extract` (~5 minutes).

## 6 · Deviations from, and extensions beyond, the specification

| Item | Status | Justification |
|---|---|---|
| Third dataset (Flowers-102) | extension ‡ | the spec allows any two; the third is cheap, and its 10-images-per-class training split is an instructive degenerate case. Running all three let the pair be chosen on evidence: DTD + FGVC-Aircraft, confirmed post-results on structural grounds (§4, ADR 0006). |
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
- Liang, V. W., Zhang, Y., Kwon, Y., Yeung, S., & Zou, J. (2022). *Mind the Gap: Understanding the Modality Gap in Multi-modal Contrastive Representation Learning*. NeurIPS 2022.
