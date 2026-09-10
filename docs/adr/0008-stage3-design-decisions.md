# ADR 0008 — Stage 3 design decisions, fixed a priori

**Status:** accepted (2026-08-24), before any Stage-3 training run.

> **Naming note (added 2026-09-10, no other change):** in the notebook, figures, tables and report the two mandatory methods are called the **Rolled strategy** (this document's Strategy 1 — end-to-end rolled-out classification training) and the **Guided strategy** (Strategy 2 — classifier-guided targets with standard FM training). The numbering below is kept exactly as pre-registered.
Extended record with full rationale and the three-round external-review log:
`docs/stage3_decisions.md` (R1–R28, all resolved; reviewer verdict:
"methodologically sound and aligned with the PDF").

**Context.** `_docs/stage_3.pdf` inserts an FM transformation before the frozen
Stage-1 linear probe: `z → FM (T Euler steps) → ẑ → frozen classifier → logits`,
trains the probe first exactly as in Stage 1, initializes the FM close to
identity, and compares — per dataset — the Stage-1 probe, end-to-end rolled-out
classification training (Strategy 1), and classifier-guided targets with
standard FM training (Strategy 2). Both strategies are mandatory (Main
Comparison + "both Stage 3 methods" three times in Results to Present);
"Suggested" grants freedom in details only. The decisions below fill what the
PDF leaves open; all were fixed before the first training run.

**Decisions.**

1. **Raw integration and classifier space.** No input normalization anywhere:
   the probe was trained on raw features, and the pipeline must equal the probe
   at initialization. Internal conditioning inside the velocity net would not
   break identity but is not used initially (parsimony; per-dimension scale is
   already ~O(1)); it is fallback step (3) of decision 7.

2. **Exact identity initialization.** Weight AND bias of the velocity net's
   final layer are zero → ẑ = z exactly. Architecture and integration per the
   PDF ("same general velocity-network design and Euler integration procedure
   as in Stage 2"): MLP dim+1 → 512 → 512 → dim, SiLU, scalar t concatenated;
   `ẑ_{k+1} = ẑ_k + (1/T)·v_θ(ẑ_k, k/T)`, k = 0..T−1, ẑ₀ = z (raw).

3. **Scope:** DTD/ResNet-18 and FGVC-Aircraft/DINOv2 (the Stage-1
   validation-selected encoders), K = 10 (the PDF's suggested default),
   **T = 4** throughout (shallower backprop graph for Strategy 1; the PDF
   delegates the value). 3 runs per setting: subset seeds {0,1,2}, probe-init 0
   (as Stage 1 at K=10), FM-init 0 — the spread measures subset sampling.

4. **Pinned probes.** Stage 1 saved no weights, so the probe is retrained with
   the identical recipe/seed per (dataset, subset seed), saved, and pinned —
   every method in a setting receives the same checkpoint. The mathematical
   baseline for Δ is `Acc(C_s(z_test))` of the exact pinned probe (identically
   the pipeline-at-init accuracy); `runs.csv` is a reproduction audit only
   (pre-training on validation, |diff| < 0.25 pts hard assert; test compared
   post-hoc at the final pass).

5. **Sealed test.** All guards run on train/validation only: exact identity
   (`max‖ẑ−z‖ == 0`, logits allclose, predictions identical) and the probe
   validation audit. Test is evaluated once, at the end, over locked
   checkpoints. Hyperparameter winners are selected per dataset on seed-0
   validation; winners apply unchanged to seeds 1–2. Disclosed: seed 0 is
   partly a development run; the 3-seed mean is a summary, not independent
   confirmation. No "significant" language at n = 3.

6. **Strategy 1** — backprop CE through the full rollout, FM parameters only:
   `CE(Wẑ+b, y) + λ · mean_i ‖ẑᵢ−zᵢ‖² / (‖zᵢ‖² + ε)` (relative displacement
   penalty). λ grid {0, 1, 10, 100} on validation; the winner AND λ=0 are the
   pre-registered test pair (the with/without-regularization comparison).

7. **Training policy (both strategies, uniform).** AdamW lr 1e-3 / wd 1e-4 /
   batch 64 / exactly 200 epochs, best-validation-accuracy checkpoint retained
   (tie-breaks: highest val acc → lowest val CE → earliest epoch); validation
   every epoch; no scheduler; no clipping by default. Failure policy (objective
   trigger = NaN/Inf loss only): any NaN/Inf before completing 200 epochs →
   restart with the next pre-registered fallback — (1) grad clipping max-norm
   1.0, (2) lr 3e-4, (3) internal input standardization; **the ladder is
   cumulative** (level 2 = clipping + lr; level 3 = all three); finite late
   deterioration → retain best checkpoint, no restart; all fallbacks exhausted
   → run marked failed and reported (partial curves of failed attempts are
   saved). Hyperparameter selection across the sweep uses the same ordering
   (highest val acc → lowest val CE); an exact tie resolves to grid order —
   deterministic and stated here. If λ = 0 wins the S1 sweep, the pre-registered
   "winner AND λ=0" test pair degenerates to a single run and the
   with/without-regularization contrast is absent from the test table — the
   report must say so if it happens. Validation-based checkpointing is
   deliberate and symmetric: the baseline probe itself was selected the same
   way, and both strategies expose the same full-pipeline validation curve.

8. **Strategy 2** — classifier-guided targets, standard FM training; **no CE
   gradient ever reaches the FM** and no λ term (its constraint is the trust
   region). Per epoch (two-phase): snapshot the FM, build targets for all
   training samples in one pass, cache by row index, then train the epoch
   against the fixed cache. Target construction, all quantities per sample:
   `u₀ = Π_{B(z,ρ)}(ẑ)`; `u_{j+1} = Π_{B(z,ρ)}(u_j − η·∇CE/(‖∇CE‖+ε))`;
   ρ = 0.1·‖z‖; η = β·ρ; target = lowest-CE iterate among the projected
   {u₀..u_m} (monotone acceptance); CE(unprojected ẑ) recorded as a diagnostic
   only. Grid: β ∈ {0.25, 0.5, 1} × m ∈ {1, 3} on validation. ε = 1e-8.
   FM update: t ~ U(0,1), z_t = (1−t)z + t·ẑ′, velocity target ẑ′ − z,
   per-sample squared-L2 loss.

9. **Optional extension (after the mandatory comparison is complete):** joint
   fine-tuning — FM lr 1e-3, classifier lr 1e-4, no delay, same policy,
   3 seeds — plus a **classifier-only continued-training control** (same budget,
   no FM) so any joint gain is attributable. Framed as an upper reference.

10. **Deliverables & infrastructure.** Main table: pinned probe / S1 / S2
    (accuracies in %, Δ in percentage points, paired per seed); λ-ablation and
    joint experiment in separate tables. Curves: comparable end-to-end metrics
    (train/val accuracy and CE) for both strategies; strategy-internal
    quantities (S1 penalty, S2 FM loss, S2 CE before/after targets, mean
    displacement, trust-region hit rate) on separate axes. Feature viz: one PCA
    fitted jointly on [z, ẑ_S1, ẑ_S2] (raw space, `viz_selection` classes,
    house colors). Artifacts: `runs_stage3.csv` / `summary_stage3.csv`, raw
    arrays, `curves_stage3/`, pinned probes + FM models under
    `results/artifacts/stage3_models/` (gitignored, regenerable), run-0
    predictions, `stage3_*` tables/figures; tasks `run3 / smoke3 / tables3 /
    figures3 / notebook3`; `repro_check` gains `check_stage3` (Δ re-derived
    against the stored pinned-probe baseline; `runs.csv` audited separately);
    notebook `nb3_sections/` → `stage3_presentation.ipynb`, self-contained.

**Consequences.** The comparison triangle shares one feature cache, one pinned
classifier per setting, one architecture, one training policy, and per-seed
pairing, so every Δ is attributable to the FM layer and its training strategy.
Expectations are calibrated (probes at K=10: 54.3 / 51.3; small or null gains
are a legitimate outcome — the spec asks to "test whether", not "show that").
