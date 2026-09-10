# Stage 3 — FM Before a Linear Classifier

**CVLAB Summer Project · Stage 3 report**
Datasets: DTD (ResNet-18) · FGVC-Aircraft (DINOv2 ViT-S/14) — the Stage-1 validation-selected encoders
Specification: `_docs/stage_3.pdf` (source of truth); pre-registered decisions: `docs/adr/0008-stage3-design-decisions.md` (three external review rounds, R1–R28) · extended record: `docs/stage3_decisions.md`

## Abstract

Stage 3 inserts an FM transformation before the pretrained linear classifier — $z \to \mathrm{FM} \to \hat{z} \to$ frozen probe — testing whether FM can transform frozen encoder features into a representation the existing classifier handles better. The probe is trained first exactly as in Stage 1 and frozen (retrained per dataset × subset seed with the identical recipe; validation audit against the published Stage-1 numbers: 0.000 points difference and exact best-epoch agreement on all six probes); the FM is initialized to **exact** identity (zero final velocity layer), so the pipeline equals the probe at initialization — asserted at feature, logit, and prediction level. Two mandated strategies are compared against the pinned probe at K = 10, T = 4, three subset seeds, with per-seed-paired Δ: **Rolled strategy** (end-to-end rolled-out CE training with a relative displacement regularizer, λ validation-selected, λ = 0 always co-reported) and **Guided strategy** (classifier-guided targets — trust-region-projected, monotone-accepted, per-epoch cached — distilled into the FM by standard velocity regression; no CE gradient reaches the FM). The results split along the project-wide pattern: on FGVC-Aircraft/DINOv2, Guided strategy improves the probe by **+2.50 ± 0.50 points with all three seeds agreeing** (Rolled strategy: +0.88 ± 0.23, 3/3); on DTD/ResNet-18 every method sits within noise of the probe (Guided +0.16 ± 0.45; Rolled −0.69 ± 0.70). The most interesting finding is the ordering: the indirect strategy — no CE gradient ever reaches the FM — beats direct end-to-end CE training on both datasets, a pattern consistent with stronger regularization from bounded classifier-guided targets and path-regression training (Stage 3 does not isolate which of the Guided strategy's differing components is responsible; the resemblance to Stage 2's path-supervision lesson is suggestive, not isolated). All six probes reproduced Stage 1 — recorded validation and test accuracies within 10⁻¹² of the published values, with identical best epochs in the run log; no run triggered the failure ladder; test was evaluated once per locked checkpoint, in one final pass per pre-registered phase — no selection depended on any test read.

## 1 · Setup

**Pipeline and initialization.** Velocity network and Euler procedure are the Stage-2 design, per the spec: MLP $d{+}1 \to 512 \to 512 \to d$, SiLU, scalar $t$ concatenated; $\hat{z}_{k+1} = \hat{z}_k + \frac{1}{T} v_\theta(\hat{z}_k, k/T)$, $k = 0..T{-}1$, $\hat{z}_0 = z$ **raw** (no normalization anywhere: the probe was trained on raw features, and identity-at-initialization must reproduce it — the deliberate mirror image of Stage 2's normalized sphere, where the downstream classifier was cosine-based; the FM operates in the space its downstream classifier acts in). The final velocity layer starts at zero (weight and bias), so $\hat{z} = z$ exactly; an identity guard asserts bit-exact features and identical logits/predictions on validation data before any training.

**Pinned probes and the baseline.** Stage 1 did not save probe weights, so the probe is retrained per (dataset, subset seed) with the literal Stage-1 class and recipe, audited on validation against `runs.csv` (< 0.25 pts; observed 0.000 pts, exact best-epoch match), pinned, and shared by every method of that setting. The mathematical baseline for Δ is the exact pinned probe's own test accuracy — identically the pipeline-at-init accuracy; `runs.csv` serves as a reproduction audit only (its test values were compared post-hoc at the final pass).

**Training policy (uniform, pre-registered).** AdamW, lr $10^{-3}$, wd $10^{-4}$, batch 64, exactly 200 epochs, best-validation-accuracy checkpoint (ties → lowest validation CE → earliest epoch), validation every epoch. Validation-based selection is symmetric: the baseline probe itself was selected the same way. Failure policy — objective trigger (NaN/Inf loss only): restart with the next fallback of a cumulative ladder (grad-clip 1.0 → lr 3e-4 → internal input standardization, identity-preserving); finite late deterioration is absorbed by checkpointing; an exhausted ladder marks the run failed. Hyperparameters selected per dataset on **seed-0 validation** only (winners applied unchanged to seeds 1–2; seed 0 is disclosed as partly a development run); the **test split stayed sealed** during all training and selection, and was evaluated in one final pass per pre-registered phase (the mandatory grid; then the optional extension), each over its already-locked checkpoints.

**Strategies** (named after their training signal — the *Rolled* strategy is the specification's Strategy 1, the *Guided* strategy its Strategy 2)**.** Rolled: $\mathrm{CE}(W\hat{z}+b, y) + \lambda\,\mathrm{mean}_i \lVert\hat{z}_i - z_i\rVert^2 / (\lVert z_i\rVert^2 + \varepsilon)$ through the full rollout, FM parameters only; λ ∈ {0, 1, 10, 100}; the validation winner **and** λ = 0 are the pre-registered test pair. Guided (all quantities per sample, ε = 10⁻⁸): per epoch, snapshot the FM, set $u_0 = \Pi_{B(z,\rho)}(\hat{z})$ with $\rho = 0.1\lVert z\rVert$, take $m$ normalized CE-gradient steps of size $\eta = \beta\rho$ with re-projection after every step, select the lowest-CE projected iterate as the detached target, cache for all samples, then train one epoch of standard FM regression toward the cache; β ∈ {0.25, 0.5, 1} × m ∈ {1, 3}. The source-centered trust region prevents cumulative target drift across epochs; monotone acceptance guarantees targets never increase CE.

## 2 · Results

### 2.1 Main comparison

*Table 1 — generated into `results/metrics/stage3_main_table.md`:*

| Dataset / encoder | Head | Top-1 (%) | Δ vs pinned probe (pts, paired) |
|---|---|---|---|
| DTD / ResNet-18 | Pinned Stage-1 probe (baseline) | 54.31 ± 0.28 | — |
| DTD / ResNet-18 | Rolled strategy — end-to-end rolled-out CE training (λ = 1) | 53.62 ± 0.89 | -0.69 ± 0.70 (1/3 seeds > 0) |
| DTD / ResNet-18 | Guided strategy — classifier-guided targets (β = 0.25, m = 1) | 54.47 ± 0.70 | +0.16 ± 0.45 (2/3 seeds > 0) |
| FGVC-Aircraft / DINOv2 | Pinned Stage-1 probe (baseline) | 51.30 ± 0.95 | — |
| FGVC-Aircraft / DINOv2 | Rolled strategy — end-to-end rolled-out CE training (λ = 1) | 52.18 ± 1.11 | +0.88 ± 0.23 (3/3 seeds > 0) |
| FGVC-Aircraft / DINOv2 | Guided strategy — classifier-guided targets (β = 0.5, m = 3) | 53.80 ± 0.95 | +2.50 ± 0.50 (3/3 seeds > 0) |

Top-1 accuracy (%) on the complete official test split, K = 10, T = 4; mean ± sample std over the 3 subset seeds. Δ is **paired per seed** against the exact pinned probe of that seed — the identical frozen classifier inside the pipeline (the pipeline at initialization equals it exactly, by the zero-velocity init). Hyperparameters (λ; β, m) were selected per dataset on **seed-0 validation** only; seed 0 is therefore partly a development run, and the 3-seed mean is a summary, not an independent confirmatory estimate. n = 3 — no significance claims.

**Reading it.** The stage's question gets a split answer. On **FGVC-Aircraft/DINOv2** the transport helps: the Guided strategy gains +2.50 ± 0.50 points with per-seed deltas +2.76 / +1.92 / +2.82 — every seed positive, and larger than the probe's own seed spread; Rolled strategy gains +0.88 ± 0.23 (3/3). On **DTD/ResNet-18** nothing separates from the probe (Guided +0.16 ± 0.45, Rolled −0.69 ± 0.70). The same geography appeared in Stage 2 and, before it, in the Stage-1 headroom measurements: the fine-grained DINOv2 representation still holds structure a fixed linear boundary has not extracted; DTD showed no benefit at this operating point (K = 10, this encoder, this classifier — whether its representation is exhausted in general is not established here). The gains land inside the pre-registered expectation band ("low single digits, possibly ≈0 on DTD").

**The ordering Guided > Rolled holds on both datasets — and on validation before any test read.** The Guided strategy never passes a CE gradient into the FM; it distills conservative targets (trust-region-capped at 10% of each feature's norm, CE-monotone by acceptance) through standard velocity regression. Rolled strategy backpropagates CE directly through the rollout. Both memorize the K = 10 training set (§3 — train accuracy ≈100%), so the difference lies off the training points. **Why Guided wins is not isolated by this experiment**: it differs through several components at once (source-centred trust region, monotone target selection, classifier-guided targets, path-regression training), and the advantage is *consistent with* stronger regularization from bounded targets and path regression — everything that differs pushes Guided toward conservatism. The resemblance to Stage 2's standard-vs-rolled-out result is a suggestive cross-experiment pattern, not an isolated mechanism. (n = 3.)

### 2.2 Rolled strategy — the pre-registered regularization pair

*Table 2 — `results/metrics/stage3_lambda_ablation_table.md`:*

| Dataset / encoder | Variant | Top-1 (%) | Δ vs pinned probe (pts) |
|---|---|---|---|
| DTD / ResNet-18 | validation-selected λ (λ = 1) | 53.62 ± 0.89 | -0.69 ± 0.70 |
| DTD / ResNet-18 | λ = 0 | 53.51 ± 1.25 | -0.80 ± 0.98 |
| FGVC-Aircraft / DINOv2 | validation-selected λ (λ = 1) | 52.18 ± 1.11 | +0.88 ± 0.23 |
| FGVC-Aircraft / DINOv2 | λ = 0 | 51.80 ± 0.88 | +0.50 ± 0.95 |

Pre-registered pair: the validation winner AND λ = 0 both receive a test read-out, so the with/without-regularization comparison involves no post-hoc choice.

λ = 1 sits above λ = 0 on both datasets, and the three observed seeds show ≈4× lower delta variability with λ = 1 on FGVC (±0.23 vs ±0.95) — a small mean effect; three seeds cannot establish stabilization in general. Note that "less movement" is not the mechanism: at their selected checkpoints, Guided moves features *more* than regularized Rolled on FGVC (mean ‖ẑ−z‖ ≈ 4.6 vs 1.2 units) and generalizes better; displacement magnitude alone does not explain performance — the movement direction and/or the training construction may matter.

### 2.3 Selection transparency

*Table 3 — the complete seed-0 validation sweep (`results/metrics/stage3_sweep_table.md`):*

| Dataset | Strategy | Configuration | Val top-1 (%) | Checkpoint epoch | Fallback |
|---|---|---|---|---|---|
| DTD | Rolled | lam=0 | 49.79 | 1 | 0 |
| DTD | Rolled | lam=1 | 50.90 | 110 | 0 |
| DTD | Rolled | lam=10 | 50.59 | 103 | 0 |
| DTD | Rolled | lam=100 | 50.74 | 1 | 0 |
| DTD | Guided | beta=0.25,m=1 | 51.97 | 167 | 0 |
| DTD | Guided | beta=0.25,m=3 | 51.76 | 156 | 0 |
| DTD | Guided | beta=0.5,m=1 | 51.86 | 157 | 0 |
| DTD | Guided | beta=0.5,m=3 | 51.81 | 152 | 0 |
| DTD | Guided | beta=1.0,m=1 | 51.91 | 145 | 0 |
| DTD | Guided | beta=1.0,m=3 | 51.70 | 10 | 0 |
| FGVC-Aircraft | Rolled | lam=0 | 53.74 | 16 | 0 |
| FGVC-Aircraft | Rolled | lam=1 | 53.95 | 86 | 0 |
| FGVC-Aircraft | Rolled | lam=10 | 53.05 | 184 | 0 |
| FGVC-Aircraft | Rolled | lam=100 | 52.72 | 156 | 0 |
| FGVC-Aircraft | Guided | beta=0.25,m=1 | 55.06 | 42 | 0 |
| FGVC-Aircraft | Guided | beta=0.25,m=3 | 55.78 | 17 | 0 |
| FGVC-Aircraft | Guided | beta=0.5,m=1 | 55.54 | 62 | 0 |
| FGVC-Aircraft | Guided | beta=0.5,m=3 | 56.11 | 43 | 0 |
| FGVC-Aircraft | Guided | beta=1.0,m=1 | 55.78 | 34 | 0 |
| FGVC-Aircraft | Guided | beta=1.0,m=3 | 56.11 | 17 | 0 |

Seed-0 **validation** accuracy of the full pipeline per swept configuration (test untouched during selection). Winners per dataset by highest validation accuracy (ties → lowest validation CE → grid order). Fallback level 0 = the default recipe (ADR 0008 §7).

### 2.4 Optional extension — joint fine-tuning, with its attribution control

*Table 4 — `results/metrics/stage3_joint_table.md`:*

| Dataset / encoder | Variant | Top-1 (%) | Δ vs pinned probe (pts) |
|---|---|---|---|
| DTD / ResNet-18 | Joint FM + classifier fine-tuning | 53.88 ± 0.63 | -0.43 ± 0.37 |
| DTD / ResNet-18 | Control: classifier-only continued training | 54.26 ± 0.46 | -0.05 ± 0.19 |
| DTD / ResNet-18 | (reference) Guided strategy, frozen classifier | 54.47 ± 0.70 | +0.16 ± 0.45 |
| FGVC-Aircraft / DINOv2 | Joint FM + classifier fine-tuning | 51.86 ± 0.96 | +0.56 ± 0.98 |
| FGVC-Aircraft / DINOv2 | Control: classifier-only continued training | 51.94 ± 0.98 | +0.64 ± 0.08 |
| FGVC-Aircraft / DINOv2 | (reference) Guided strategy, frozen classifier | 53.80 ± 0.95 | +2.50 ± 0.50 |

The extension was pre-registered as an "upper reference"; it is not one. The joint variant does not outperform classifier-only continued training in these three seeds (FGVC: +0.56 ± 0.98 vs +0.64 ± 0.08; per-seed joint−control: +0.06 / −1.08 / +0.78; DTD: both ≈0/slightly negative), so **the results provide no evidence of an additional FM-attributable gain when the classifier is unfrozen** — and both sit far below the frozen-classifier Guided strategy (+2.50 ± 0.50). The early validation-selected checkpoints (epochs 0–18, single-digit in 5 of 6 runs) are consistent with rapid overfitting in the K = 10 regime, but this experiment does not isolate classifier freezing as the cause — particularly because the best frozen configuration uses Guided strategy while joint training uses a different objective; even the closest matched-objective comparison (frozen Rolled strategy vs joint, both CE-trained) does not separate the two (+0.88 ± 0.23 vs +0.56 ± 0.98). Empirically, frozen-classifier Guided strategy remains the best tested configuration. Without the control — added at external-review insistence — the +0.56 could have masqueraded as a small FM win.

## 3 · Training behaviour

Both strategies drive train-pipeline accuracy to ≈100% within the budget (K = 10 supplies 470–1000 images against 788K / 657K FM parameters on DTD / FGVC respectively) while validation plateaus — textbook memorization, absorbed by the best-validation-accuracy checkpoint exactly as in Stage 1. Winner-run checkpoints land mid-training across all seeds (Rolled: epochs 36–179; Guided: 25–185); the one family that checkpoints near the start is the unregularized λ = 0 variant (epochs 0–2) — the displacement penalty is what buys Rolled its useful training trajectory. The failure policy never fired: zero NaN/Inf events, every model completed at fallback level 0, so the entire pre-registered ladder (clipping → lr → standardization) remained unused — in contrast to Stage 2, where endpoint-only training diverged at full split; here the shallow T = 4 rollout, identity start, and validation checkpointing kept everything tame. Strategy-internal diagnostics (`stage3_diag_*.png`): Guided's trust-region hit rate and target-CE traces show the target construction working as designed — the CE at the selected target sits well below the CE at the projected start, and a substantial fraction of targets sit on the trust-region boundary, i.e. the cap binds and is doing its job.

## 4 · Feature-space geometry

The transports are deliberately small — roughly 5–13% of the mean feature norm (mean ‖ẑ−z‖ ≈ 1–5 units against mean norms of ≈24 on DTD and ≈50 on DINOv2) — and the joint-PCA panels (`stage3_features_*.png`) show it: unlike Stage 2's dramatic contraction onto prototypes, the global class layout is nearly unchanged, with mild boundary-relevant adjustments. The first two PCs change only subtly — especially on FGVC-Aircraft, despite its +2.50-point gain — suggesting the useful movement occurs along classifier-relevant high-dimensional directions that the joint 2-D PCA does not capture (it explains 15.6% / 33.0% of the variance on DTD / FGVC). That is the appropriate geometry for this stage: the FM's productive move is local corrections that flip marginal test points across an already-fixed boundary — large reorganizations could only destroy the alignment the probe was trained on. Gains here come from many small nudges, not from restructuring.

## 5 · Limitations

- n = 3 subset seeds; spreads are sample standard deviations; no significance claims. The spread measures subset sampling only (probe-init and FM-init fixed).
- Seed 0 is partly a development run (hyperparameters selected on its validation split); the 3-seed mean is a summary, not independent confirmation.
- One K, one T, one encoder per dataset — the spec's own scoping; conclusions are about this operating point.
- On DTD the λ-ablation contrast is intact but both variants are within noise of the probe; interpretation there is limited by the ceiling, not by the pair design.
- 2-D projections are qualitative; the joint-PCA plane explains 15.6% (DTD: 8.2 + 7.4) to 33.0% (FGVC: 20.6 + 12.4) of the variance.
- The Strategy-2 trust region was active for nearly all targets after the early epochs (hit rate ≈100%); performance may therefore depend materially on the fixed choice α = 0.1 — no α ablation was performed (future work, or a clearly-labelled validation-only exploration).
- Although the linear classifier is frozen, the FM adds nonlinear capacity by warping feature space — $W F_\theta(z) + b$ can represent nonlinear decision boundaries even with $W, b$ fixed. Conclusions are limited to the chosen FM architecture, K = 10, T = 4, and the fixed trust-region radius, not to "what a frozen classifier can do" in general.

## 6 · Deviations from, and extensions beyond, the specification

| Item | Status | Note |
|---|---|---|
| Probe trained first exactly as Stage 1, then frozen | as specified | retrained per (dataset, seed); validation audit 0.000 pts + exact best-epoch match on all six probes; weights pinned; Δ computed against the pinned probe itself, `runs.csv` as separate audit (also exact on test, post-hoc) |
| FM close to identity | as specified (strengthened) | zero final layer ⇒ exact identity, asserted at feature/logit/prediction level |
| Velocity net + Euler as Stage 2; single T; K = 10 | as specified | T = 4 pre-registered; K = the spec's suggested default |
| Both strategies + main comparison + deliverables | as specified | table + Δ; train/validation curves for both methods; joint-embedding feature visualization |
| Relative displacement penalty (Rolled); trust region / monotone acceptance / per-epoch cached targets (Guided) | within spec freedom | the spec invites regularization and constrained target updates; all details pre-registered (ADR 0008, three external review rounds) |
| Validation-accuracy checkpointing; 3-subset-seed repetition | our choice (spec silent) | symmetric to the baseline's own selection; keeps paired Δ ± std consistent with Stages 1–2 |
| Optional joint fine-tuning + classifier-only control | done (see §2.4) | control added so any joint gain is attributable to the FM rather than to longer classifier training |

Reproduce with `tasks.ps1 run3 / tables3 / figures3 / notebook3`; `tasks.ps1 check` re-derives every summary number, every paired Δ against the stored pinned-probe baselines, and every generated table from raw artifacts.

## References

- Lipman, Y., Chen, R. T. Q., Ben-Hamu, H., Nickel, M., & Le, M. (2023). *Flow Matching for Generative Modeling*. ICLR 2023.
- Cimpoi, M., Maji, S., Kokkinos, I., Mohamed, S., & Vedaldi, A. (2014). *Describing Textures in the Wild* (DTD). CVPR 2014.
- Maji, S., Rahtu, E., Kannala, J., Blaschko, M., & Vedaldi, A. (2013). *Fine-Grained Visual Classification of Aircraft* (FGVC-Aircraft). arXiv:1306.5151.
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition* (ResNet). CVPR 2016.
- Oquab, M., Darcet, T., Moutakanni, T., et al. (2023). *DINOv2: Learning Robust Visual Features without Supervision*. TMLR 2024.
