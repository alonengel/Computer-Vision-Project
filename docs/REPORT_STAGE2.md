# Stage 2 — Flow Matching to Class Prototypes

**CVLAB Summer Project · Stage 2 report**
Datasets: DTD · FGVC-Aircraft (the pair selected on evidence in Stage 1, ADR 0006)
Specification: `_docs/stage_2.pdf` (source of truth); design decisions fixed a priori: `docs/adr/0007-stage2-design-decisions.md`

## Abstract

Stage 2 inserts a flow-matching (FM) decision layer above the Stage-1 prototype classifier: a small velocity network transports each frozen image feature toward the fixed prototype of its class, and the transported feature is classified with the identical cosine rule as Stage 1. We compare the Stage-1 baseline against standard FM training (velocity supervision at random interpolation points) and rolled-out training (backpropagation through the full T-step Euler inference sequence with a loss on the final state), at T ∈ {4, 12}, on the identical feature caches, committed training subsets, seeds and test splits as Stage 1. On the spec branch (image-derived prototypes), the results are sharply structured: (RESULTS-SUMMARY-PLACEHOLDER). As a marked ‡ extension, the same grid runs on CLIP RN50 features toward the fixed text prototypes, where FM must cross the image–text modality gap; (CLIP-SUMMARY-PLACEHOLDER).

## 1 · Setup

**Inherited from Stage 1, unchanged.** Frozen encoders and cached features (ResNet-18 on both datasets; DINOv2 ViT-S/14 on FGVC-Aircraft; CLIP RN50 for the ‡ branch); official train/validation/test splits; balanced K-shot subsets from the committed index files, K ∈ {5, 10, full}, subset seeds {0, 1, 2}; the class-prototype formula $\mu_c = \mathrm{normalize}(\frac{1}{|S_c|}\sum_{i\in S_c}\mathrm{normalize}(z_i))$ per (K, seed); top-1 accuracy on the complete official test split; 3 runs per setting (K ∈ {5,10}: subset seeds with FM initialization fixed; full: three FM initialization seeds on the fixed training set).

**The FM layer** (spec, followed exactly). Standard training: $z_t = (1-t)z_i + t\,p_{y_i}$ with $t \sim \mathcal{U}(0,1)$, loss $\lVert v_\theta(z_t,t) - (p_{y_i} - z_i)\rVert_2^2$. Inference: $\hat z_{k+1} = \hat z_k + \frac{1}{T} v_\theta(\hat z_k, k/T)$, $k = 0..T{-}1$, starting from the test feature; $\hat z_T$ classified by $\arg\max_c \cos(\hat z_T, p_c)$. Rolled-out training: the same T-step sequence applied to each training feature, loss $\lVert \hat z_T - p_{y_i}\rVert_2^2$ through the whole sequence; training T equals inference T. Velocity network: MLP $d{+}1 \to 512 \to 512 \to d$, SiLU, scalar $t$ concatenated (the spec's suggestion); one architecture everywhere. Standard FM training involves no T, so one standard model per setting is trained and evaluated at both T values — those two table rows share one set of weights.

**Training configuration, fixed a priori (ADR 0007 §7).** The Stage-1 probe recipe verbatim: AdamW, lr $10^{-3}$, weight decay $10^{-4}$, batch 64, 200 epochs. No validation-based checkpointing — the final-epoch model is used: the spec specifies no selection rule for FM; a per-T validation selection would treat the two training modes asymmetrically; and a fixed budget keeps the standard-vs-rolled comparison clean. Every model's training-loss curve is saved; §4 evaluates the a-priori stability criterion (final-epoch loss within 1.05× of the running minimum) over all of them.

**One disclosed deviation from the literal spec (ADR 0007 §3): FM operates on $L_2$-normalized features.** The spec writes $\hat z_0 = z$ on the raw frozen feature; we set $\hat z_0 = z/\lVert z\rVert$. The classifier the transported feature must serve is cosine similarity — it acts on the sphere — while raw feature norms are ≈10–40 versus unit-norm prototypes, so raw-space interpolation would traverse mostly scale rather than class structure. The decision was fixed before any training; no renormalization occurs between Euler steps; and with T = 0 the pipeline reduces exactly to Stage 1.

**Integrity guard.** Before any training, the run script asserts — for every setting — that classifying the untransported test features against that setting's prototypes reproduces the Stage-1 baseline accuracy from `runs.csv` to within $10^{-6}$. All settings passed.

**Branches.** Spec branch: FM toward image-derived prototypes (the Stage-1 Option A selection, ADR 0006) on DTD/ResNet-18, FGVC-Aircraft/ResNet-18 and FGVC-Aircraft/DINOv2. Extension ‡ (group decision, mirroring ADR 0005): the identical grid on CLIP RN50 image features toward the fixed CLIP text prototypes. The ‡ FM rows consume K labeled images per class and are therefore *supervised transport on CLIP features*, never "zero-shot"; their reference is the Stage-1 zero-shot number, and Δ there answers "does supervised transport toward text prototypes beat zero-shot classification?", not "does the FM layer help?" — only the image-prototype branch isolates the FM effect, because its baseline uses the identical labeled subset.

## 2 · Results

(RESULTS-SECTION-PLACEHOLDER)

## 3 · Training stability

(STABILITY-PLACEHOLDER)

## 4 · Geometry

(GEOMETRY-PLACEHOLDER)

## 5 · Limitations

(LIMITATIONS-PLACEHOLDER)

## 6 · Deviations from, and extensions beyond, the specification

(DEVIATIONS-PLACEHOLDER)

## References

- Lipman, Y., Chen, R. T. Q., Ben-Hamu, H., Nickel, M., & Le, M. (2023). *Flow Matching for Generative Modeling*. ICLR 2023.
- Radford, A., Kim, J. W., Hallacy, C., et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision* (CLIP). ICML 2021.
- Cimpoi, M., Maji, S., Kokkinos, I., Mohamed, S., & Vedaldi, A. (2014). *Describing Textures in the Wild* (DTD). CVPR 2014.
- Maji, S., Rahtu, E., Kannala, J., Blaschko, M., & Vedaldi, A. (2013). *Fine-Grained Visual Classification of Aircraft* (FGVC-Aircraft). arXiv:1306.5151.
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition* (ResNet). CVPR 2016.
- Oquab, M., Darcet, T., Moutakanni, T., et al. (2023). *DINOv2: Learning Robust Visual Features without Supervision*. TMLR 2024.
