# Stage 3 — Decision Summary (draft for approval, nothing implemented yet)

> Internal working document for team review, prior to being fixed in ADR 0008.
> Source of truth: `_docs/stage_3.pdf`. Date: 2026-08-24. Status: **awaiting approval — nothing has been implemented.**
>
> **Naming note (added 2026-09-10, no other change):** the deliverables call Strategy 1 the **Rolled strategy** and Strategy 2 the **Guided strategy**; this record keeps the spec's numbering.

## What is being built

```
z (frozen feature) → FM, T Euler steps → ẑ → frozen linear classifier (the Stage-1 probe) → logits
```

The stage's question: can FM transform the frozen features into a representation that the existing probe separates better.

**Architecture and integration (per the PDF: "the same general velocity-network design and Euler integration procedure as in Stage 2"):** velocity network v_θ(z, t) = MLP `dim+1 → 512 → 512 → dim`, SiLU activations, scalar time t concatenated to the input feature; Euler integration

`ẑ_{k+1} = ẑ_k + (1/T) · v_θ(ẑ_k, k/T),  k = 0, …, T−1,  ẑ_0 = z` (raw, unnormalized).

---

## Core decisions

| # | Decision | Detail and rationale |
|---|---|---|
| 1 | **Raw integration and classifier space — no input normalization** | The FM *state* and the classifier *input* live in raw feature space. The probe was trained on raw features (`s = Wz + b`, norms ≈10–40); normalizing on entry would break the spec's requirement that the system behaves like the original probe before training. Clarification: *internal* conditioning inside the velocity network (standardize-in, scale-out) would not break the zero-init identity and is not forbidden — it is simply not used initially (parsimony; per-dimension feature scale is already ~O(1)) and is available as fallback step (3) of item 11. |
| 2 | **Identity initialization: zeros in the last layer of the velocity network** | The spec requires "Initialize the FM close to identity" without specifying how; zero-init gives *exact* identity (ẑ = z), with an assertable guard: pipeline accuracy at init ≡ frozen-probe accuracy, to the last digit. |
| 3 | **The probe is retrained, saved, and pinned — one artifact per dataset × subset seed** | Stage 1 did not save weights. Retrain with the identical recipe and seed, save the weights (one pinned probe per dataset × seed, since each seed has its own training subset); every method within a setting receives the same file. Reproduction audited against `runs.csv` (see Guards). |
| 4 | **3 runs — subset seeds {0,1,2} at K=10; Δ against the pinned probe itself** | The spec is silent on repetitions; without this, Stage 3 would be the only stage without error bars. The baseline for Δ is `Acc(C_s(z_test))` — the **exact pinned checkpoint** `C_s` used by that FM model, evaluated on untransported features (identically the pipeline-at-init accuracy, by the exact-identity init). `runs.csv` serves as a reproduction **audit only**, never as the mathematical baseline. Cost: minutes. |
| 5 | **Checkpoint selection by validation accuracy** | Symmetric to how the baseline itself was selected in Stage 1; both methods expose the same full-pipeline validation curve; the spec explicitly asks for validation curves. A divergence monitor (final loss vs running minimum) is kept as a backstop. |
| 6 | **Both strategies — both are implemented** | See discussion below — the question was raised whether "Suggested" permits implementing only one. |
| 7 | **Strategy 1 with displacement regularization + a no-regularization comparison variant** | CE + relative displacement penalty, λ selected on validation. The λ=0 variant is **always reported separately** alongside the validation-selected variant (no "significance" gating — see statistical-language rule). |

### Discussion of item 6 — does "Suggested" exempt one of the methods? **[RESOLVED: both]**

Team question: the headings say "**Suggested** Strategy 1/2" — maybe one is enough?

What the spec actually says:

- Scope paragraph: *"The training strategies below are intended as structured starting points rather than fixed recipes. You are encouraged to modify their **details**, compare variants, and, if motivated, propose and evaluate **alternative** training schemes."* — the freedom is in the details, or in replacing a scheme with a justified alternative.
- **Main Comparison** (binding): *"For each of the two datasets, compare: Stage 1 linear probe; end-to-end rolled-out classification training; classifier-guided FM training."* — both methods named explicitly.
- **Results to Present**: "...for the Stage 1 linear probe and **both Stage 3 methods**" · "curves for **both Stage 3 methods**" · "ẑ for **the two Stage 3 methods**".

**Resolved (team agreement): implement both.** "Suggested" = freedom in each strategy's details, not in choosing between them; implementing one would leave three explicit deliverable bullets unfulfilled. Marginal cost is small: the infrastructure (network, rollout, training loop) is shared; Strategy 2 is roughly tens of additional lines.

---

## Strategy 2 details — fixed in advance (left open by the spec)

- **The target**: ẑ′ = one or more CE-gradient steps in feature space, **detached** from the graph.
- **Trust region**: a cap on ‖ẑ′ − z‖ — guards against adversarial target drift (CE can be driven to zero by walking deep into the classifier's decision cone; without a cap, test geometry is destroyed while training CE looks excellent).
- **Recompute cadence**: per epoch, not per batch (a per-batch moving target is the classic bootstrap instability).
- **η (step size) and m (number of steps)**: a small sweep on **validation only**, pre-registered.

## Delegated choices

| # | Choice | Value and rationale |
|---|---|---|
| 8 | Encoders | DTD → ResNet-18 (its only supervised encoder anyway), FGVC-Aircraft → DINOv2 — decided in Stage 1 on validation (handoff table). Zero new selection. |
| 9 | K | **10** — the default the PDF suggests ("a reasonable default is K = 10"); the main deliverable at this size only ("to keep this stage focused"). |
| 10 | T | The PDF delegates the value ("Choose a single number of Euler steps T and use it throughout"). **T = 4** — Strategy 1 backpropagates through the whole rollout; a small T = a shallower, more stable graph. Pre-registered. |
| 11 | Training recipe — **bounded policy, no open-ended freedom** | AdamW / lr 1e-3 / wd 1e-4 / batch 64 / **exactly 200 epochs, with the best validation checkpoint retained** (no early stopping); validation evaluated **every epoch**; no lr scheduler; no gradient clipping by default. Checkpoint tie-breaking: highest validation accuracy → lowest validation CE → earliest epoch. **Failure policy (single, objective):** **any NaN/Inf loss before completing the 200 epochs → restart with the next pre-registered fallback**: (1) gradient clipping (max-norm 1.0); (2) lr → 3e-4; (3) input standardization inside the velocity net. **Finite late deterioration → retain the best validation checkpoint, no restart** (that is what checkpointing is for). **All fallbacks exhausted → the run is marked failed and reported as such.** Each fallback applies uniformly to both strategies and is reported. Nothing else may change. |

## Gap closure — concrete values (pre-registered)

Everything that would otherwise be decided "silently" while writing code. Every sweep is on **validation only**, on **seed 0**; hyperparameter winners are selected **separately for each dataset**; the winning configuration is applied unchanged to seeds 1, 2; test is read once per (method, seed) at the end. Throughout this document **ε = 10⁻⁸**. All Strategy-2 per-sample quantities — CE values, gradient norms, projection radii ρᵢ, step sizes ηᵢ = β·ρᵢ, and best-iterate selection — are computed **per sample**, never aggregated across the batch.

### Strategy 1 — loss function and regularization

- Loss: `CE(Wẑ + b, y) + λ · mean_i ( ‖ẑᵢ − zᵢ‖² / (‖zᵢ‖² + ε) )` — a **relative** displacement penalty, so λ has the same meaning across encoders/datasets (feature norms vary 10–40) and is symmetric with S2's relative trust region.
- λ grid: **{0, 1, 10, 100}** on validation (re-centered for the relative penalty: relative displacement of ~0.1 → penalty term ~λ·0.01 against CE of ~ln C ≈ 3.9–4.6). λ=0 is exactly the "no-displacement variant" of item 7 — always reported next to the winner, so the with/without-regularization comparison is built into the table.

### Strategy 2 — classifier-guided targets

- **Target construction (source-centered trust region):** starting from the **projected** FM output

  `u₀ = Π_{B(z, ρ)}(ẑ)`,   then   `u_{j+1} = Π_{B(z, ρ)} ( u_j − η · ∇u CE / (‖∇u CE‖ + ε) )`

  i.e. a normalized gradient step, **projected after every step — including step zero — onto the ball of radius ρ centered at the ORIGINAL feature z**. Projecting u₀ closes the last drift hole: if the current FM output ẑ has itself left the trust region, an unprojected u₀ could be selected as the lowest-CE candidate and perpetuate the drift. Every eligible candidate therefore satisfies ‖u_j − z‖ ≤ ρ by construction; CE(ẑ) of the unprojected output is recorded for diagnostics only, never eligible as a target.
- **Radius:** relative, ρ = α·‖z‖ with **α = 0.1 fixed a priori** (≈1–4 feature-units against norms 10–40).
- **Step size — relative as well:** η_i = β·ρ_i per sample, so identical settings behave identically for features of norm 10 and 40 (a fixed absolute η would not). Grid: **β ∈ {0.25, 0.5, 1} × m ∈ {1, 3}** (6 combinations) on validation.
- **Monotone acceptance:** the target is the iterate with the **lowest CE** among the projected {u₀, …, u_m}; a step that increases CE can never be selected.
- Targets **detached**; recomputed **per epoch** as an explicit two-phase epoch: (1) snapshot the FM in eval mode, compute ẑ and targets for **all** training samples in one pass, cache by row index (our features are a fixed cached tensor, so sample identity is trivially stable); (2) train one epoch against the fixed cache with shuffled batches. FM update: standard FM — t∼U(0,1), z_t = (1−t)z + t·ẑ′, velocity target ẑ′ − z, squared-L2 loss. **No CE gradient ever reaches the FM in S2, and S2 has no λ displacement term** (its constraint is the trust region).

### Optional extension (unfreezing the classifier)

- **Postponed until the mandatory comparison is complete and clean.** One configuration, no sweep: FM lr 1e-3, classifier lr **1e-4** (10× lower), no delayed unfreezing, same epochs/checkpointing; 3 seeds; marked "optional — done"; framed as an upper reference only.
- **Attribution control (added after external review):** a **classifier-only continued-training** run — the pinned probe trained further alone (no FM, same budget, lr 1e-4). Without it, a joint-training gain cannot be attributed to the FM rather than to the classifier simply training longer.

### Guards (train/validation only — the test set stays sealed until the final pass)

1. **Exact identity (strengthened):** on a training/validation batch, assert `max_i ‖ẑᵢ − zᵢ‖ == 0` (bit-exact, by the zero init), plus logits allclose and predictions identical between the pipeline-at-init and the direct probe. Accuracy equality alone is too weak — two systems can agree on accuracy while disagreeing on many predictions.
2. **Probe reproduction (pre-training audit):** retrained probe's **validation** accuracy vs the per-seed `val_acc` in `runs.csv`, |diff| < **0.25 points** (hard assert; exact difference recorded).
3. **Final pass:** test is evaluated **once**, at the very end, over all already-locked checkpoints (pinned probes + selected FM models). At that point the retrained probes' test accuracies are also compared against `runs.csv` as a **post-hoc audit** — no decision depends on it.

### Final items (completed on second review)

- **Seed structure:** in each run s ∈ {0,1,2}: subset seed = s, probe-init = 0 (as Stage 1 did at K=10), FM-init = 0 — the spread measures subset sampling, as everywhere in the project.
- **Full-length sweeps:** the λ/β/m combinations train for **exactly 200 epochs** like the final run (item 11) — selection is not based on a different training regime.
- **S2 — complete separation from CE:** no λ·‖ẑ−z‖² term (S2's constraint is the trust region on the targets), and its FM trains exclusively on the velocity-regression loss — no CE gradient flows into the FM. This is the essential difference from S1 and keeps the comparison clean.
- **S1 test reads:** the validation winner **and** λ=0 — a pre-registered pair (not a post-hoc choice); the remaining combinations stay validation-only. For S2: the winner only.
- **Stability:** governed by the single failure policy in item 11 (any NaN/Inf before completing 200 epochs → restart with next fallback; finite late deterioration → retain best validation checkpoint; all fallbacks exhausted → run marked failed and reported). Curves are always saved; deterioration is reported descriptively.
- **Zero initialization:** weight **and bias** of the last layer = 0 (exact identity); the remaining layers use the standard initialization.
- **smoke3:** one dataset, one seed, a few epochs, one combination per strategy — a pipeline check only.

### Artifacts, reporting, and infrastructure

- **Files:** `runs_stage3.csv` / `summary_stage3.csv`, raw arrays with a prefix, curves in `curves_stage3/`, probe + FM weights in `results/artifacts/stage3_models/` (gitignored, regenerable), run-0 predictions, tables `stage3_*.md`, figures `stage3_*.png`.
- **Tasks:** `run3` / `smoke3` / `tables3` / `figures3` / `notebook3`; `repro_check` extended with `check_stage3` — re-deriving Δ against **the corresponding exact pinned probe** (whose accuracy is stored as `baseline_acc` in `runs_stage3.csv`); `runs.csv` is checked **separately** as a reproduction audit of the pinned probes.
- **Reporting:** per-seed-paired Δ against the **pinned probe** of the same seed (see item 4); mean ± sample std over 3 seeds; representative curves: seed 0 (house convention). **Statistical language:** no "significant" claims at n=3; the spread measures subset-sampling variability only (probe-init and FM-init are fixed); every per-seed paired Δ is shown. **Seed-0 caveat (disclosed):** hyperparameters are selected on seed-0 validation, so seed 0 is partly a development run — the three-seed mean is reported as the summary estimate, never presented as an independent confirmatory estimate.
- **Main table layout:** one clean table — rows = dataset/encoder, columns = pinned probe / S1 (acc + paired Δ) / S2 (acc + paired Δ). Accuracies are reported in **%**; only Δ values are in **percentage points**. The λ=0 ablation and the optional joint-training experiment go in **separate** tables.
- **Curves (comparable + diagnostics):** for both strategies, the *comparable* end-to-end metrics are plotted: training + validation classification accuracy and CE. Strategy-internal quantities are shown **separately, never on a shared axis** (they measure different things): S1's displacement penalty; S2's FM regression loss; S2's CE before-vs-after target improvement; mean displacement ‖ẑ−z‖; and S2's trust-region hit rate (fraction of samples whose target hit the projection).
- **Feature viz:** the same 10 `viz_selection` classes; one PCA fitted jointly on [z, ẑ_S1, ẑ_S2] in raw space; same examples/colors; PCA only (there are no prototypes at this stage — the classifier is W, b).
- **Subset identity:** already covered since Stage 1 — the K-shot subsets are committed index files with pool fingerprints asserted on every load; no additional hashing needed.

## Framing, deliverables, and scope

- **12 · Calibrated expectations:** probes at K=10: 54.3 (DTD/RN18), 51.3 (FGVC/DINOv2) — a discriminatively trained boundary on the same information. Honest expectation: low single-digit gains, possibly ≈0 on DTD. A small/negative result + clean geometry analysis is a legitimate deliverable; the spec says "test whether".
- **13 · The optional extension will be done:** unfreezing the classifier and joint FM+linear training — framed as an upper reference only (in practice ≈ a small residual MLP head; expected to win, which is shallow), marked "optional — done".
- **14 · Scope:** the main deliverable = exactly the PDF's comparison (probe / S1 / S2 × DTD, FGVC). Any extra — a separate ‡, only after the main story is clean.
- **Required deliverables from the PDF:** (a) top-1 table + Δ vs the probe; (b) training+validation curves for both methods; (c) feature viz of z vs ẑ for both methods — joint embedding, same examples/colors.

## Infrastructure and workflow

- **15 · New, separate notebook:** `nb3_sections/` → `stage3_presentation.ipynb`, self-contained, its own tasks (`run3`, `notebook3`, ...).
- **16 · Order:** ADR 0008 (fixing all items) → methodology review (cv-expert) → implementation → smoke → full runs → tables/figures/notebook → adversarial review → commit.

---

## External review — point-by-point resolution

An external methodological review of this document was received (2026-08-24). Resolution of each point:

| # | Review point | Resolution |
|---|---|---|
| R1 | Δ must be computed against the exact pinned probe, not `runs.csv`; one probe per dataset × seed | **Accepted** — items 3–4 rewritten; `runs.csv` demoted to reproduction audit |
| R2 | S2 trust region must be centered at the source z (per-recompute caps allow unlimited cumulative drift); add ε; verify CE actually decreases; relative radius | **Accepted in full** — source-centered projection Π_B(z, ρ), ρ = 0.1·‖z‖, ε in normalization, best-CE-iterate acceptance. This was a genuine inconsistency between our conceptual section and the concrete grid |
| R3 | No test access in pre-training guards | **Accepted** — guards moved to train/validation; single final test pass over locked checkpoints; `runs.csv` test comparison becomes post-hoc audit. (Nuance recorded: evaluating a pinned probe on test would not have been *selection* leakage — its test accuracy is already published — but the sealed-test protocol is adopted as the cleaner standard) |
| R4 | Identity guard: assert features/logits/predictions, not just accuracy | **Accepted** — `max‖ẑ−z‖ == 0` + logits allclose + identical predictions |
| R5 | Replace "full freedom to adapt" with a bounded policy | **Accepted** — item 11 rewritten: fixed budget, per-epoch validation, explicit tie-breaks, pre-registered two-step fallback ladder |
| R6 | Internal standardization inside the velocity net | **Not adopted initially (challenged on necessity, not correctness)** — per-*dimension* feature scale is already ~O(1) (norms 10–40 over 384–512 dims → rms ≈ 0.5–2), so the conditioning benefit is marginal; and as the reviewer correctly noted in round 2, it would *not* break the zero-init identity. Kept as step (3) of the fallback ladder; decision-1 wording clarified accordingly |
| R7 | Relative regularization scaling for S1 | **Accepted** — relative penalty; λ grid re-centered to {0, 1, 10, 100} |
| R8 | Comparable curves + per-strategy diagnostics (incl. trust-region hit rate) | **Accepted in full** |
| R9 | Hash sampled image IDs | **Already in place since Stage 1** — committed subset index files + pool fingerprints asserted on load |
| R10 | S2 needs an explicit two-phase epoch with cached targets by stable ID; "more involved than tens of lines" | **Structure accepted; difficulty estimate challenged** — features are a fixed cached tensor, so the stable ID is the row index and caching is trivial in this pipeline |
| R11 | Statistical language (no "significant" at n=3; spread = subset sampling only) | **Accepted** — was house style; now stated explicitly |
| R12 | Postpone optional joint fine-tuning; add classifier-only continued-training control | **Accepted** — postponement was already planned; the control is a genuinely good catch (attribution would otherwise be confounded) and is added |
| R13 | Main table clean; ablations/joint in separate tables; percentage points | **Accepted** |

### Round 2 (2026-08-24) — resolution

| # | Review point | Resolution |
|---|---|---|
| R14 | **u₀ must be projected**: an unprojected u₀ = ẑ outside B(z, ρ) could be selected as the lowest-CE target | **Accepted — genuine bug.** u₀ = Π_B(z,ρ)(ẑ); every eligible candidate now satisfies ‖u_j − z‖ ≤ ρ by construction; CE(ẑ) recorded for diagnostics only |
| R15 | "if the gap is significant" contradicts the n=3 statistical-language rule | **Accepted** — λ=0 is always reported separately; no significance gating |
| R16 | "Exactly ≤200 epochs" is contradictory | **Accepted** — "exactly 200 epochs, best validation checkpoint retained" |
| R17 | Infrastructure still said "re-deriving Δ against runs.csv" | **Accepted** — Δ re-derived against the exact pinned probe; `runs.csv` a separate reproduction audit |
| R18 | Fallback ladder vs "no hard contingency" conflict; define divergence objectively | **Accepted** — single policy in item 11: trigger = NaN/Inf only; restart-with-next-fallback only before any valid checkpoint exists; afterwards retain best checkpoint and report, no restart |
| R19 | Accuracy in %, only Δ in percentage points | **Accepted** — table-layout wording fixed |
| R20 | Make the S2 step relative: η = β·ρ | **Accepted** — β ∈ {0.25, 0.5, 1} × m ∈ {1, 3}; coherent with the relative radius (identical behavior across feature norms 10–40) |
| R21 | Decision-1 wording vs the fallback that may introduce internal standardization; standardization does not break identity | **Accepted** — decision 1 re-titled "raw integration and classifier space"; clarification added; R6 rationale corrected (challenged on necessity, not correctness) |
| R22 | Reference the exact Stage-2 architecture + Euler equation | **Accepted** — added to "What is being built" (also matches the PDF's "same general velocity-network design and Euler integration procedure as in Stage 2") |
| R23 | Seed-0 is partly a development run; don't present the 3-seed mean as independent confirmation | **Accepted** — disclosure added to Reporting |

### Round 3 (2026-08-24) — final corrections, approved for ADR

| # | Review point | Resolution |
|---|---|---|
| R24 | Sweeps still said "≤200 epochs" | **Accepted** — "exactly 200 epochs", matching item 11 |
| R25 | NaN/Inf loophole: a valid checkpoint exists after epoch 1, so an early NaN would retain a near-untrained model instead of triggering the fallback | **Accepted — genuine loophole.** New policy: any NaN/Inf before completing 200 epochs → restart with next fallback; finite late deterioration → retain best validation checkpoint; all fallbacks exhausted → run marked failed and reported |
| R26 | Define ε | **Accepted** — ε = 10⁻⁸ throughout |
| R27 | S2 quantities are per-sample, not batch-aggregated | **Accepted** — stated explicitly (CE, gradient norms, ρᵢ, ηᵢ, best-iterate selection) |
| R28 | Winners selected separately per dataset on seed-0 validation | **Accepted** — stated explicitly |

Reviewer verdict after round 3: *"methodologically sound and aligned with the PDF … approve moving to ADR + implementation."*

**Spec-drift check (round 2):** every adopted change was verified against `_docs/stage_3.pdf`; none contradicts an explicit instruction. The S2 mechanics fall under the PDF's explicit freedom ("for example, by taking one or more gradient steps in feature space"; "experiment with … whether the target updates should be normalized or otherwise constrained"); the training recipe and repetition protocol are left open by the PDF; Δ against the pinned probe *is* the PDF's "corresponding Stage 1 linear probe" baseline (the pinned probe is the faithful, audited reconstruction of it — and is literally the frozen classifier inside the pipeline); the architecture/Euler reference implements the PDF's own instruction to reuse the Stage-2 design.

*Status: **final** — amended after three external review rounds and approved by the reviewer for ADR + implementation. Awaiting team go-ahead; then fixed in ADR 0008 and implementation begins.*
