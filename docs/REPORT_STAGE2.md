# Stage 2 — Flow Matching to Class Prototypes

**CVLAB Summer Project · Stage 2 report**
Datasets: DTD · FGVC-Aircraft (the pair selected on evidence in Stage 1, ADR 0006)
Specification: `_docs/stage_2.pdf` (source of truth); design decisions fixed a priori: `docs/adr/0007-stage2-design-decisions.md`

## Abstract

Stage 2 inserts a flow-matching (FM) decision layer above the Stage-1 prototype classifier: a small velocity network transports each frozen image feature toward the fixed prototype of its class, and the transported feature is classified with the identical cosine rule as Stage 1. We compare the Stage-1 baseline against standard FM training (velocity supervision at random interpolation points) and rolled-out training (backpropagation through the full T-step Euler inference sequence with a loss on the final state), at T ∈ {4, 12}, on the identical feature caches, committed training subsets, seeds and test splits as Stage 1. The results are sharply structured: FM gains track exactly the headroom Stage 1 measured between the class-mean prototype and a trained boundary on the same features. On FGVC-Aircraft/DINOv2 (largest headroom) the FM layer gains **+10.4 to +23.2 points** over the prototype baseline (3/3 seeds at every K), lifting the full-split pipeline from 34.4% to 57.6%; on FGVC-Aircraft/ResNet-18 standard FM gains +1.6 to +6.7; on DTD (smallest headroom) FM gains +0.5 to +0.8 at full split and *loses* 2.4–12.9 points in the low-shot settings (0/3 seeds). Standard FM is the better or equal training mode in every spec-branch setting (statistically indistinguishable from rolled-out only at FGVC-Aircraft/DINOv2 K ∈ {5,10}), and rolled-out training produced the grid's only training divergences — caught by a pre-registered stability criterion whose uniform fallback (minimum-training-loss checkpointing) was applied to every model. As a marked ‡ extension, the same grid on CLIP RN50 features toward the fixed text prototypes shows supervised transport beating the zero-shot reference in every cell (up to +23.9), with rolled-out training winning decisively exactly where transport must cross the image–text modality gap.

## 1 · Setup

**Inherited from Stage 1, unchanged.** Frozen encoders and cached features (ResNet-18 on both datasets; DINOv2 ViT-S/14 on FGVC-Aircraft; CLIP RN50 for the ‡ branch); official train/validation/test splits; balanced K-shot subsets from the committed index files, K ∈ {5, 10, full}, subset seeds {0, 1, 2}; the class-prototype formula $\mu_c = \mathrm{normalize}(\frac{1}{|S_c|}\sum_{i\in S_c}\mathrm{normalize}(z_i))$ per (K, seed); top-1 accuracy on the complete official test split; 3 runs per setting (K ∈ {5,10}: subset seeds with FM initialization fixed; full: three FM initialization seeds on the fixed training set).

**The FM layer** (spec, followed exactly). Standard training: $z_t = (1-t)z_i + t\,p_{y_i}$ with $t \sim \mathcal{U}(0,1)$, loss $\lVert v_\theta(z_t,t) - (p_{y_i} - z_i)\rVert_2^2$. Inference: $\hat z_{k+1} = \hat z_k + \frac{1}{T} v_\theta(\hat z_k, k/T)$, $k = 0..T{-}1$, starting from the test feature; $\hat z_T$ classified by $\arg\max_c \cos(\hat z_T, p_c)$. Rolled-out training: the same T-step sequence applied to each training feature, loss $\lVert \hat z_T - p_{y_i}\rVert_2^2$ through the whole sequence; training T equals inference T. Velocity network: MLP $d{+}1 \to 512 \to 512 \to d$, SiLU, scalar $t$ concatenated (the spec's suggestion); one architecture everywhere. Standard FM training involves no T, so one standard model per setting is trained and evaluated at both T values — those two table rows share one set of weights.

**Training configuration, fixed a priori (ADR 0007 §7).** The Stage-1 probe recipe verbatim: AdamW, lr $10^{-3}$, weight decay $10^{-4}$, batch 64, 200 epochs. No validation-based checkpointing anywhere: the spec specifies no selection rule for FM, and a per-T validation selection would treat the two training modes asymmetrically. The a-priori policy was the final-epoch model with one pre-registered contingency: if any run's final-epoch training loss exceeded 1.05× its own running minimum, the uniform fallback for **all** models would be the minimum-training-loss epoch. **The contingency triggered**: on the first grid, 3 of 135 runs genuinely diverged (all rolled-out at K = full; the worst at 62× its minimum, test accuracy collapsed to ~6% — recorded at trigger time, the first grid's artifacts were superseded by the re-run), alongside 20 marginal 1.05–1.13× cases (18 standard-FM last-epoch-noise runs, 2 rolled-out runs just past the threshold). As pre-registered, every model — both modes, both branches — was re-run under minimum-training-loss selection; no validation or test data enters the selection, and the first grid's numbers were never published. All numbers in this report come from the fallback grid.

**One disclosed deviation from the literal spec (ADR 0007 §3): FM operates on $L_2$-normalized features.** The spec writes $\hat z_0 = z$ on the raw frozen feature; we set $\hat z_0 = z/\lVert z\rVert$. The classifier the transported feature must serve is cosine similarity — it acts on the sphere — while raw feature norms are ≈10–40 versus unit-norm prototypes, so raw-space interpolation would traverse mostly scale rather than class structure. The decision was fixed before any training; no renormalization occurs between Euler steps; and with T = 0 the pipeline reduces exactly to Stage 1. **The rationale is measured, not asserted** (post-audit control, beyond spec): retraining the identical networks — same recipe, prototypes, subsets and seeds, only the input normalization differing — on **raw** features degrades **all 18 of 18** settings of the seed-0 slice, mostly by double digits (e.g. DTD full standard: 42.7% raw vs 59.5% normalized; FGVC-Aircraft/DINOv2 full: 37.3% vs 57.8%); at low K, raw-feature FM frequently falls below the Stage-1 baseline itself (`results/metrics/stage2_raw_ablation.md`, embedded in the notebook §2).

**Integrity guard.** Before any training, the run script asserts — for every setting — that classifying the untransported test features against that setting's prototypes reproduces the Stage-1 baseline accuracy from `runs.csv` to within $10^{-6}$. All settings passed.

**Branches.** Spec branch: FM toward image-derived prototypes (the Stage-1 Option A selection, ADR 0006) on DTD/ResNet-18, FGVC-Aircraft/ResNet-18 and FGVC-Aircraft/DINOv2. Extension ‡ (group decision, mirroring ADR 0005): the identical grid on CLIP RN50 image features toward the fixed CLIP text prototypes. The ‡ FM rows consume K labeled images per class and are therefore *supervised transport on CLIP features*, never "zero-shot"; their reference is the Stage-1 zero-shot number, and Δ there answers "does supervised transport toward text prototypes beat zero-shot classification?", not "does the FM layer help?" — only the image-prototype branch isolates the FM effect, because its baseline uses the identical labeled subset.

## 2 · Results

### 2.1 Spec branch — FM toward image-derived prototypes

*Table 1 — generated into `results/metrics/stage2_image_prototype_table.md`:*

| Dataset | Encoder | Head | K = 5 | K = 10 | full train split |
|---|---|---|---|---|---|
| DTD | ResNet-18 | Stage-1 Image prototypes (baseline) | **46.51 ± 0.64** | **53.37 ± 1.04** | 58.78 |
| DTD | ResNet-18 | Standard FM, T = 4 | 43.81 ± 1.35 (Δ -2.70 ± 1.46) | 48.69 ± 0.62 (Δ -4.68 ± 1.61) | 59.26 ± 0.32 (Δ +0.48 ± 0.32) |
| DTD | ResNet-18 | Standard FM, T = 12 | 44.10 ± 1.36 (Δ -2.41 ± 1.36) | 49.27 ± 0.52 (Δ -4.10 ± 1.34) | **59.56 ± 0.11 (Δ +0.78 ± 0.11)** |
| DTD | ResNet-18 | Rolled-out FM, T = 4 | 38.95 ± 0.94 (Δ -7.55 ± 0.75) | 40.50 ± 0.91 (Δ -12.87 ± 0.45) | 54.80 ± 1.17 (Δ -3.97 ± 1.17) |
| DTD | ResNet-18 | Rolled-out FM, T = 12 | 39.47 ± 0.97 (Δ -7.04 ± 0.74) | 40.74 ± 0.52 (Δ -12.62 ± 0.53) | 55.60 ± 0.21 (Δ -3.17 ± 0.21) |
| FGVC-Aircraft | ResNet-18 | Stage-1 Image prototypes (baseline) | 16.04 ± 0.83 | 19.85 ± 0.47 | 25.20 |
| FGVC-Aircraft | ResNet-18 | Standard FM, T = 4 | 17.59 ± 0.66 (Δ +1.55 ± 0.18) | 23.76 ± 0.61 (Δ +3.91 ± 0.77) | 31.67 ± 0.17 (Δ +6.47 ± 0.17) |
| FGVC-Aircraft | ResNet-18 | Standard FM, T = 12 | **17.70 ± 0.79 (Δ +1.66 ± 0.06)** | **23.82 ± 0.65 (Δ +3.97 ± 0.63)** | **31.87 ± 0.14 (Δ +6.67 ± 0.14)** |
| FGVC-Aircraft | ResNet-18 | Rolled-out FM, T = 4 | 15.87 ± 0.73 (Δ -0.17 ± 0.76) | 20.63 ± 0.31 (Δ +0.78 ± 0.45) | 25.16 ± 0.92 (Δ -0.04 ± 0.92) |
| FGVC-Aircraft | ResNet-18 | Rolled-out FM, T = 12 | 15.86 ± 0.72 (Δ -0.18 ± 0.92) | 21.23 ± 1.26 (Δ +1.38 ± 1.04) | 25.34 ± 0.80 (Δ +0.14 ± 0.80) |
| FGVC-Aircraft | DINOv2 | Stage-1 Image prototypes (baseline) | 23.11 ± 1.36 | 27.80 ± 1.17 | 34.41 |
| FGVC-Aircraft | DINOv2 | Standard FM, T = 4 | 33.91 ± 1.44 (Δ +10.80 ± 0.50) | 43.58 ± 0.54 (Δ +15.78 ± 0.84) | **57.59 ± 0.71 (Δ +23.17 ± 0.71)** |
| FGVC-Aircraft | DINOv2 | Standard FM, T = 12 | 33.50 ± 1.40 (Δ +10.39 ± 0.47) | 43.28 ± 0.53 (Δ +15.48 ± 0.74) | 57.28 ± 0.53 (Δ +22.86 ± 0.53) |
| FGVC-Aircraft | DINOv2 | Rolled-out FM, T = 4 | 33.87 ± 1.43 (Δ +10.76 ± 1.22) | 43.25 ± 1.06 (Δ +15.45 ± 1.05) | 55.37 ± 0.14 (Δ +20.95 ± 0.14) |
| FGVC-Aircraft | DINOv2 | Rolled-out FM, T = 12 | **34.02 ± 1.69 (Δ +10.91 ± 1.51)** | **43.73 ± 1.63 (Δ +15.93 ± 1.14)** | 54.28 ± 0.82 (Δ +19.86 ± 0.82) |

Top-1 accuracy (%) on the complete official test split; Δ = change vs the Stage-1 image-prototype baseline of the **identical** setting (same subset indices, seeds and prototypes). K ∈ {5, 10}: mean ± sample std over the 3 subset seeds, Δ paired per seed (mean ± std of per-seed differences). full: 3 FM initialization seeds against the single deterministic full-split baseline, so the Δ spread there measures FM training stochasticity only. The two Standard-FM rows of a setting share one trained network (standard FM training is independent of T; only inference differs). **Bold** marks the best value per column within each dataset–encoder block.

*Figures 1–3 — `results/figures/stage2_acc_{dtd_resnet18,fgvc_aircraft_resnet18,fgvc_aircraft_dinov2_vits14}_image_prototype.png`: accuracy vs training-set size with the Stage-1 baseline as the dashed reference; error bars = sample std over 3 runs.*

**FM gains track the Stage-1 headroom.** Stage 1 quantified the gap between the class-mean prototype and a trained linear boundary on identical features (validation headroom: 35.7 points on FGVC-Aircraft/DINOv2, moderate on FGVC-Aircraft/ResNet-18, 5.3 points on DTD/ResNet-18). The FM gains line up with that ordering — a consistent ordering across the three dataset–encoder settings available, suggestive rather than a statistical claim at n = 3 settings: +10.4 to +23.2 on FGVC/DINOv2 (3/3 seeds at every K; full split 34.4% → 57.6%), +1.6 to +6.7 for standard FM on FGVC/ResNet-18 (3/3 seeds), +0.5 to +0.8 at full on DTD — and consistent low-shot *losses* on DTD (0/3 seeds, down to −12.9). A trained nonlinear transport re-introduces discriminative structure that the class-mean estimator discards; where classes are heavily entangled (textures) and K is small, the learned basins misroute test features and transport creates errors instead.

*Table 2 — per-seed-paired deltas (`results/metrics/stage2_paired_delta_table.md`), the regime where pairing is genuine:*

| Dataset | Encoder | Head | K | Δ per seed (paired) | Seeds favouring FM |
|---|---|---|---|---|---|
| DTD | ResNet-18 | Standard FM, T = 4 | 5 | -2.70 ± 1.46 | 0 / 3 |
| DTD | ResNet-18 | Standard FM, T = 12 | 5 | -2.41 ± 1.36 | 0 / 3 |
| DTD | ResNet-18 | Rolled-out FM, T = 4 | 5 | -7.55 ± 0.75 | 0 / 3 |
| DTD | ResNet-18 | Rolled-out FM, T = 12 | 5 | -7.04 ± 0.74 | 0 / 3 |
| DTD | ResNet-18 | Standard FM, T = 4 | 10 | -4.68 ± 1.61 | 0 / 3 |
| DTD | ResNet-18 | Standard FM, T = 12 | 10 | -4.10 ± 1.34 | 0 / 3 |
| DTD | ResNet-18 | Rolled-out FM, T = 4 | 10 | -12.87 ± 0.45 | 0 / 3 |
| DTD | ResNet-18 | Rolled-out FM, T = 12 | 10 | -12.62 ± 0.53 | 0 / 3 |
| FGVC-Aircraft | ResNet-18 | Standard FM, T = 4 | 5 | +1.55 ± 0.18 | 3 / 3 |
| FGVC-Aircraft | ResNet-18 | Standard FM, T = 12 | 5 | +1.66 ± 0.06 | 3 / 3 |
| FGVC-Aircraft | ResNet-18 | Rolled-out FM, T = 4 | 5 | -0.17 ± 0.76 | 2 / 3 |
| FGVC-Aircraft | ResNet-18 | Rolled-out FM, T = 12 | 5 | -0.18 ± 0.92 | 2 / 3 |
| FGVC-Aircraft | ResNet-18 | Standard FM, T = 4 | 10 | +3.91 ± 0.77 | 3 / 3 |
| FGVC-Aircraft | ResNet-18 | Standard FM, T = 12 | 10 | +3.97 ± 0.63 | 3 / 3 |
| FGVC-Aircraft | ResNet-18 | Rolled-out FM, T = 4 | 10 | +0.78 ± 0.45 | 3 / 3 |
| FGVC-Aircraft | ResNet-18 | Rolled-out FM, T = 12 | 10 | +1.38 ± 1.04 | 3 / 3 |
| FGVC-Aircraft | DINOv2 | Standard FM, T = 4 | 5 | +10.80 ± 0.50 | 3 / 3 |
| FGVC-Aircraft | DINOv2 | Standard FM, T = 12 | 5 | +10.39 ± 0.47 | 3 / 3 |
| FGVC-Aircraft | DINOv2 | Rolled-out FM, T = 4 | 5 | +10.76 ± 1.22 | 3 / 3 |
| FGVC-Aircraft | DINOv2 | Rolled-out FM, T = 12 | 5 | +10.91 ± 1.51 | 3 / 3 |
| FGVC-Aircraft | DINOv2 | Standard FM, T = 4 | 10 | +15.78 ± 0.84 | 3 / 3 |
| FGVC-Aircraft | DINOv2 | Standard FM, T = 12 | 10 | +15.48 ± 0.74 | 3 / 3 |
| FGVC-Aircraft | DINOv2 | Rolled-out FM, T = 4 | 10 | +15.45 ± 1.05 | 3 / 3 |
| FGVC-Aircraft | DINOv2 | Rolled-out FM, T = 12 | 10 | +15.93 ± 1.14 | 3 / 3 |

Mean ± sample std (ddof = 1) of the per-seed difference FM − Stage-1 image-prototype baseline, both computed from identical committed subset indices and prototypes. Positive favours FM. n = 3 seeds; no confidence intervals are quoted at this sample size.

**Standard is the better or equal training mode on the spec branch.** In every image-branch setting (DTD: smaller losses; FGVC/ResNet-18: +1.6..+6.7 vs −0.2..+1.4; DINOv2: indistinguishable at K ∈ {5,10} — at matched T, rollout is nominally ahead in the two T = 12 cells and behind in the two T = 4 cells, all well within the paired spreads — and standard 2.2–3.0 points ahead at full), and all three genuine training divergences were rolled-out runs (§3). Root cause: the rolled-out objective constrains only the endpoint of the composed T-step map, so the field is free to become degenerate along the path; standard FM's per-point velocity supervision regularizes the entire trajectory.

**T barely matters for standard FM.** The two standard rows of a setting share one network; 4 → 12 Euler steps moves accuracy by less than 0.7 points in either direction — integration error is not the binding constraint.

### 2.2 Extension ‡ — FM on CLIP features toward text prototypes

*Table 3 — generated into `results/metrics/stage2_clip_text_table.md`:*

| Dataset | Encoder | Head | K = 5 | K = 10 | full train split |
|---|---|---|---|---|---|
| DTD ‡ | CLIP RN50 | Stage-1 Zero-shot CLIP (reference, K-independent) | 39.79 | 39.79 | 39.79 |
| DTD ‡ | CLIP RN50 | Standard FM, T = 4 | 50.80 ± 0.53 (Δ +11.01 ± 0.53) | 56.42 ± 0.84 (Δ +16.63 ± 0.84) | 63.03 ± 0.75 (Δ +23.24 ± 0.75) |
| DTD ‡ | CLIP RN50 | Standard FM, T = 12 | **50.89 ± 0.61 (Δ +11.10 ± 0.61)** | **56.70 ± 0.77 (Δ +16.91 ± 0.77)** | **63.67 ± 0.70 (Δ +23.88 ± 0.70)** |
| DTD ‡ | CLIP RN50 | Rolled-out FM, T = 4 | 47.64 ± 1.24 (Δ +7.85 ± 1.24) | 54.31 ± 0.78 (Δ +14.52 ± 0.78) | 63.21 ± 0.11 (Δ +23.42 ± 0.11) |
| DTD ‡ | CLIP RN50 | Rolled-out FM, T = 12 | 47.85 ± 1.01 (Δ +8.07 ± 1.01) | 54.73 ± 0.94 (Δ +14.95 ± 0.94) | 61.58 ± 1.23 (Δ +21.79 ± 1.23) |
| FGVC-Aircraft ‡ | CLIP RN50 | Stage-1 Zero-shot CLIP (reference, K-independent) | 17.04 | 17.04 | 17.04 |
| FGVC-Aircraft ‡ | CLIP RN50 | Standard FM, T = 4 | 19.20 ± 0.70 (Δ +2.16 ± 0.70) | 21.91 ± 0.62 (Δ +4.87 ± 0.62) | 23.93 ± 0.33 (Δ +6.89 ± 0.33) |
| FGVC-Aircraft ‡ | CLIP RN50 | Standard FM, T = 12 | **19.45 ± 0.82 (Δ +2.41 ± 0.82)** | 22.03 ± 0.36 (Δ +4.99 ± 0.36) | 24.55 ± 0.41 (Δ +7.51 ± 0.41) |
| FGVC-Aircraft ‡ | CLIP RN50 | Rolled-out FM, T = 4 | 18.50 ± 0.72 (Δ +1.46 ± 0.72) | **23.04 ± 0.38 (Δ +6.00 ± 0.38)** | **31.49 ± 1.03 (Δ +14.45 ± 1.03)** |
| FGVC-Aircraft ‡ | CLIP RN50 | Rolled-out FM, T = 12 | 18.68 ± 0.36 (Δ +1.64 ± 0.36) | 22.74 ± 0.45 (Δ +5.70 ± 0.45) | 28.64 ± 0.21 (Δ +11.60 ± 0.21) |

‡ Extension beyond the specification (the spec's Stage-2 branch is the image-prototype branch selected in Stage 1, ADR 0006). FM here transports **CLIP RN50 image features** toward the fixed CLIP **text** prototypes. Unlike the Stage-1 zero-shot reference, FM training consumes K labeled images per class — these rows are *supervised transport on CLIP features*, never "zero-shot". Δ therefore answers "does supervised transport toward text prototypes beat zero-shot classification?", **not** "does the FM layer help?" — only the image-prototype branch isolates the FM effect, because its baseline uses the identical labeled subset. The reference value is a single deterministic run, so Δ carries no pairing; at K ∈ {5, 10} the spread over the 3 subset seeds is reported on the accuracy.

Supervised transport exceeds the zero-shot reference in every cell (+1.5 to +23.9). As the caption states, these deltas bundle the value of K labels with the value of transport — only the image-prototype branch isolates the FM effect. The standout is FGVC-Aircraft at full split, where rolled-out T = 4 (+14.5) roughly doubles standard FM (+6.9): transporting across the image–text modality gap is the regime where the straight-line target velocity is most misspecified, and an endpoint-only objective is free to learn a better curved transport — the trajectory figures (§4) show that curvature directly.

## 3 · Training stability

Training-loss curves for every one of the 135 models are saved (`results/artifacts/curves_stage2/`), shown representatively in `results/figures/stage2_curves_{0,1}.png`, and evaluated against the pre-registered ADR 0007 §7 criterion (final-epoch loss ≤ 1.05× the running minimum). On the first grid the criterion fired: 3 genuine divergences — all rolled-out models at K = full (62× its minimum on FGVC-Aircraft/CLIP‡ T = 12 seed 1, test accuracy collapsed to ~6%, recorded at trigger time; 2.9× and 1.26× on FGVC-Aircraft/ResNet-18 T = 4) — alongside 20 marginal 1.05–1.13× cases (18 standard-FM runs, whose random-t resampling makes the last epoch noisy, and 2 rolled-out runs just past the threshold; the genuine-vs-marginal split is descriptive — the pre-registered criterion is binary at 1.05× and the uniform fallback applies regardless).

**Could the minimum-loss rule pick a "lucky" epoch?** The selection statistic uses training data only and the test set is evaluated once on the selected checkpoint, so no quantity involved in the selection is correlated with test noise — any winner's curse acts on the training loss (adding variance), not on test accuracy. Empirically the rule is a no-op except where it matters: the selected checkpoint epochs sit at the end of training for every stable run (standard FM: median 193 of 200, min 159; rolled-out: median 197, min 88 — the minimum belonging to a diverged run), and the superseded final-epoch grid's stable-run accuracies differed from the published ones only by tenths of a point in both directions (the first grid's DTD arrays are preserved at commit `cbceb53` and were compared during the adversarial review). As pre-registered, the uniform fallback was applied: every model uses its minimum-training-loss epoch, the grid was fully re-run, and the first grid's numbers were never published. Under the fallback all training is well-behaved: losses decrease smoothly (log-scale curves), and both training modes reach solutions whose loss curves are qualitatively similar across datasets and encoders — the spec's stability requirement is met by construction of the checkpoint rule rather than by hoping the last epoch is representative.

## 4 · Geometry

Figures 4–8 (`results/figures/stage2_features_*.png`) compare, per setting, the original features, the features after standard FM, and the features after rolled-out FM (T = 12, K = 10, seed 0 — fixed a priori), with the training prototypes; one PCA is fitted jointly to all three feature sets plus prototypes, so the three panels share a single plane. Figures 9–13 (`results/figures/stage2_traj_*.png`) show Euler trajectories of representative test examples in the same plane. Three observations: (i) transport visibly contracts each class toward its prototype — on FGVC-Aircraft/DINOv2 the contraction is dramatic and class-preserving, which is the geometric face of the +23-point gain; (ii) on DTD the same contraction merges entangled texture classes, the geometric face of the low-shot losses; (iii) standard-FM trajectories are nearly straight (as their supervision assumes), while rolled-out trajectories curve — most prominently on the CLIP‡ branch, where the straight-line velocity target is most misspecified and the endpoint-only objective exploits its freedom. A structural note for reading the standard-FM field near t → 1: with deterministic (z_i, p_{y_i}) pairing all class-c paths converge on p_c while their target velocities still differ, so the network there can only learn a conditional average — one more reason the two modes genuinely differ in late-time geometry.

## 5 · Limitations

- n = 3 runs per setting; spreads are sample standard deviations, no confidence intervals; per-seed pairing is reported with sign counts.
- At K = full the image-branch deltas are unpaired (three FM init seeds vs one deterministic baseline); CLIP‡ deltas are never paired (single zero-shot reference).
- The CLIP‡ branch cannot isolate the FM effect (its reference consumes zero labels; a labeled CLIP image-prototype baseline is excluded by the Stage-1 protocol). Its deltas conflate labels with transport, as stated in every caption.
- No hyperparameter search: the Stage-1 probe recipe was adopted verbatim a priori. Better FM numbers are plausibly reachable, particularly for rolled-out training, whose divergences suggest the shared lr is near its stability edge for backprop-through-T.
- The stability criterion (1.05×) proved tight for stochastic per-epoch losses: 20 of 23 flags (18 standard-FM, 2 rolled-out) were benign last-epoch noise. The uniform min-loss fallback absorbs both cases without discretion, but a future criterion should compare epoch-averaged losses.
- 2-D projections are qualitative; the joint-PCA plane explains part of the variance only, and off-plane trajectory geometry is invisible.
- FM models are selected at their minimum-training-loss epoch — a training-set-only criterion; no validation or test data enters any Stage-2 selection.

## 6 · Deviations from, and extensions beyond, the specification

| Item | Status | Note |
|---|---|---|
| FM on L2-normalized features | **disclosed deviation** | fixed a priori (ADR 0007 §3); the cosine classifier acts on the sphere; T = 0 reproduces Stage 1 exactly (integrity guard, §1) |
| Checkpoint = min-training-loss epoch | **pre-registered contingency, triggered** | ADR 0007 §7; uniform over all models after 3 rolled-out divergences; training-set-only criterion; first grid never published |
| Datasets | as specified | the Stage-1 selected pair DTD + FGVC-Aircraft (ADR 0006) — "same datasets as Stage 1" |
| Prototype branch | as specified + extension ‡ | spec branch = image prototypes (Stage-1 Option A selection); CLIP‡ text-prototype transport added, labeled supervised transport |
| Velocity network / T grid / losses / Euler | as specified | MLP 2×512 SiLU, scalar t concatenated; T ∈ {4, 12}; per-sample squared L2; ẑ_{k+1} = ẑ_k + (1/T)v(ẑ_k, k/T) |
| Training configuration | suggested-scope | Stage-1 probe recipe verbatim, no search |
| Runs / seeds / metric | as specified | Stage-1 repetition protocol mirrored exactly; top-1 on the complete official test split |
| Optional reverse-flow / intermediate-time exploration | not attempted | the spec encourages it as optional; out of scope for this deliverable |
| Raw-vs-normalized control | extension (beyond spec) | seed-0 slice isolating the §1 normalization decision — identical recipe/targets/subsets, only the input normalization differs (`results/metrics/stage2_raw_ablation.md`); main results untouched |

Reproduce with `tasks.ps1 run2 / tables2 / figures2 / notebook2`; `tasks.ps1 check` re-derives every summary number, every delta (against Stage-1 `runs.csv`), and every generated table from raw artifacts.

## References

- Lipman, Y., Chen, R. T. Q., Ben-Hamu, H., Nickel, M., & Le, M. (2023). *Flow Matching for Generative Modeling*. ICLR 2023.
- Radford, A., Kim, J. W., Hallacy, C., et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision* (CLIP). ICML 2021.
- Cimpoi, M., Maji, S., Kokkinos, I., Mohamed, S., & Vedaldi, A. (2014). *Describing Textures in the Wild* (DTD). CVPR 2014.
- Maji, S., Rahtu, E., Kannala, J., Blaschko, M., & Vedaldi, A. (2013). *Fine-Grained Visual Classification of Aircraft* (FGVC-Aircraft). arXiv:1306.5151.
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition* (ResNet). CVPR 2016.
- Oquab, M., Darcet, T., Moutakanni, T., et al. (2023). *DINOv2: Learning Robust Visual Features without Supervision*. TMLR 2024.
