# ADR 0007 — Stage 2 design decisions, fixed a priori

**Status:** accepted (2026-08-09). Authored and smoke-tested before any full
training run; the infrastructure commit (`cbceb53`) landed mid-grid, but git
confirms the §7 contingency text and the `final_epoch` policy were committed
before any full-split (i.e. any subsequently-divergent) run had completed.
**§7 contingency TRIGGERED (2026-08-09, same day, after the first full grid):**
the stability sweep over all 135 training curves found 23 exceeding the 1.05×
criterion — 20 marginal (1.05–1.13×, last-epoch noise from standard FM's random-t
resampling) and 3 genuine divergences (all rolled-out at K = full: 62.4× on
FGVC-Aircraft/CLIP‡ T = 12 seed 1, whose test accuracy collapsed to 6%; 2.9× and
1.26× on FGVC-Aircraft/ResNet-18 T = 4). As pre-registered, the uniform fallback
was applied: **every** Stage-2 model — both modes, both branches — now uses the
minimum-training-loss epoch (`checkpoint_selection: "min_train_loss"` in config),
no validation or test data involved, and the entire grid was re-run under the
fallback. The first grid's numbers were never published; the criterion, fallback,
and trigger are reported in the report and notebook.

**Context.** `_docs/stage_2.pdf` adds a flow-matching layer to the Stage-1 prototype
classifier: standard FM (`z_t = (1−t)z_i + t·p_{y_i}`, target velocity `p − z`,
loss `‖v_θ(z_t,t) − u‖²`) versus rolled-out FM (run the T-step Euler inference
sequence during training, loss `‖ẑ_T − p_{y_i}‖²`, backprop through the sequence),
T ∈ {4, 12}, same datasets/encoders/prototypes/subsets/seeds/test sets as Stage 1.
The spec fixes the algorithm and protocol; the decisions below cover what it leaves
open. All are made **before** the first training run, so no choice can be influenced
by Stage-2 accuracy.

**Decisions.**

1. **Scope: DTD + FGVC-Aircraft** (the ADR 0006 pair; Flowers-102 stays Stage-1-only).
   Branch-A settings inherit Stage 1: ResNet-18 on both datasets, DINOv2 ViT-S/14 on
   FGVC-Aircraft — the three dataset–encoder configurations with image-prototype
   baselines.

2. **Both target branches, extension marked ‡** (user decision, mirroring ADR 0005):
   the spec-exact branch transports toward **image-derived prototypes** (Option A,
   the ADR 0006 selection); as an extension we also run the full grid on **CLIP RN50
   image features toward the fixed text prototypes**. The CLIP-branch reference
   remains the Stage-1 zero-shot number, but FM training consumes labeled K-shot
   images, so those rows are labeled as *supervised* transport on CLIP features —
   never as "zero-shot". Interest: the text targets sit across CLIP's modality gap.

3. **FM operates on L2-normalized features.** `ẑ_0 = z/‖z‖`; prototypes are unit-norm
   by construction. Rationale: the Stage-1 classifier rule is cosine similarity on
   normalized features, so the normalized sphere is the space in which "transport
   toward the prototype" is meaningful; interpolating raw features (norms ≈ 10–40)
   toward unit-norm prototypes would make the path mostly traverse scale, not class
   structure. Classification of `ẑ_T` uses the identical cosine rule as Stage 1.

4. **Prototypes per setting are identical to Stage 1**: recomputed with
   `PrototypeClassifier.fit` from the committed subset index files (K ∈ {5,10}, seeds
   {0,1,2}) or the full training split; text prototypes from the cached
   `clip_text_{ds}.pt`. Guard: for every setting, the T=0 classification (no
   transport) must reproduce the Stage-1 baseline accuracy exactly; the run script
   asserts this against `runs.csv` before training.

5. **Run/seed protocol mirrors Stage 1**: K ∈ {5,10} → 3 subset seeds, FM
   initialization seed fixed at 0 (spread = subset sampling); full → training set
   fixed, 3 FM initialization seeds {0,1,2} (spread = optimization stochasticity).
   Standard FM is trained **once** per (config, K, seed) — its training does not
   involve T — and evaluated at both T=4 and T=12. Rolled-out FM trains one model
   per T (the spec requires matching train/inference T). 27 + 54 = 81 branch-A
   trainings, 18 + 36 = 54 CLIP-branch trainings.

6. **Velocity network = the spec's suggestion, one architecture everywhere**: MLP
   `dim+1 → 512 → 512 → dim`, SiLU, scalar t concatenated to the input. No
   architecture search.

7. **Training configuration reuses the Stage-1 probe recipe verbatim** (AdamW,
   lr 1e-3, weight decay 1e-4, batch 64, 200 epochs) — a priori, consistent, and
   defensible as "no extensive hyperparameter optimization" (spec). One deliberate
   difference: **no validation-based checkpointing — the final-epoch model is used.**
   The spec specifies no selection rule for FM, a per-T selection would differ
   between standard (T-independent training) and rolled-out models, and the training
   objective (transport toward prototypes) is not the selection metric (accuracy);
   the fixed-budget final model keeps the standard-vs-rolled comparison clean.
   Training-loss curves for every model are saved and shown to verify stability
   (the spec's stated purpose for the curves). Contingency, fixed pre-launch
   (cv-expert review): a run counts as *unstable* only if its final-epoch training
   loss exceeds 1.05× its own running minimum (divergence, not plateau). In that
   case the uniform fallback for **all** Stage-2 models — both modes, both branches —
   is the minimum-training-loss epoch (still no validation or test involvement),
   and the change is reported. No other post-hoc adjustment is permitted.

8. **Determinism & artifacts.** Seeds control init, batch order, and t-sampling.
   Every trained model's state_dict is saved under `results/artifacts/fm_models/`
   (gitignored, regenerable — like feature caches) so figures (feature-space
   comparisons, flow trajectories) recompute transports without retraining; loss
   histories under `results/artifacts/curves_stage2/`; metrics in
   `results/metrics/runs_stage2.csv` / `summary_stage2.csv` with per-seed-paired
   ΔAcc against the Stage-1 baseline; raw arrays and run-0 predictions as in Stage 1.

9. **Euler inference exactly as specified**: `ẑ_{k+1} = ẑ_k + (1/T)·v_θ(ẑ_k, k/T)`,
   k = 0..T−1, starting from the normalized test feature; no renormalization between
   steps (the spec's update is plain Euler; `ẑ_T` is classified by cosine, which
   normalizes once at the end).

**Consequences.** The comparison triangle (Stage-1 prototype baseline, standard FM,
rolled-out FM) shares one feature cache, one prototype definition, one architecture,
one training recipe, and per-seed-paired subsets, so every ΔAcc is attributable to
the FM layer and its training mode. The CLIP‡ branch reuses the same machinery with
text targets. All tunables live in `config/config.json` under `"stage2"`.
