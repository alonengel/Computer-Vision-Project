CELLS = [
    ("markdown", """
## 6 · Discussion

Every observation below is read directly off §3–§5; the numbers appear in the generated tables and are re-derived from raw artifacts by the repro check.

**1 · The stage's question — "can FM transform frozen features into a representation the existing classifier handles better?" — gets a split answer that matches the project-wide pattern.** On **FGVC-Aircraft/DINOv2**, yes: Strategy 2 gains **+2.50 ± 0.50 points** over the pinned probe with **all three seeds agreeing** (per-seed +2.76 / +1.92 / +2.82), and Strategy 1 a smaller +0.88 ± 0.23 (3/3). On **DTD/ResNet-18**, no: every method sits within noise of the probe (S2 +0.16 ± 0.45, 2/3 seeds; S1 −0.69 ± 0.70, 1/3). This is the third stage in a row where the same geography appears — the fine-grained DINOv2 representation still holds structure a fixed linear boundary has not extracted, while DTD's texture features are already saturated for this classifier at K = 10. The gains land inside the range our pre-registered expectations called plausible ("low single digits, possibly ≈0 on DTD") — the honest headline is a *modest, consistent* improvement in one setting, not a breakthrough.

**2 · The most interesting finding: the indirect strategy beats the direct one, on both datasets.** Strategy 2 — which never lets a CE gradient touch the FM, and instead distills conservative, trust-region-bounded, CE-monotone targets through a standard velocity-regression loss — outperforms Strategy 1's direct end-to-end CE backprop everywhere (FGVC +2.50 vs +0.88; DTD +0.16 vs −0.69; and the same ordering already appeared on the validation sweeps of both datasets before any test data was seen). The training curves say why this is *consistent* (not proven, n = 3): both strategies memorize the training set (§4 — train accuracy reaches ~100% in both), so the difference is in what the objective does off the training points. S1's endpoint-CE objective is free to bend the field arbitrarily between the K = 10 anchors; S2's per-sample targets are capped at 10% of each feature's norm, can never increase CE (monotone acceptance), and the FM is fitted to them by *path-supervised* regression. This mirrors Stage 2's central lesson — velocity supervision along the path regularizes; endpoint-only objectives are freer and generalize worse — now reproduced in a different pipeline with a different classifier.

**3 · The displacement regularizer earns its place in Strategy 1, mildly.** The pre-registered pair puts λ = 1 above λ = 0 on both datasets (FGVC +0.88 ± 0.23 vs +0.50 ± 0.95; DTD −0.69 ± 0.70 vs −0.80 ± 0.98) — small mean differences, but the spreads tighten by 3–4× on FGVC, which is what a stabilizer is for. Note the geometry is not "smaller movement wins": S2 moves features *more* than regularized S1 on FGVC (mean ‖ẑ−z‖ ≈ 4.9 vs 1.1 units) yet generalizes better — it is the *kind* of movement (bounded, CE-monotone, path-distilled) that matters, not its magnitude.

**4 · The protocol held up end to end.** All six probes reproduced Stage 1 exactly — bit-identical validation and test accuracies (§2 displays the comparison; the run log additionally recorded identical best epochs); the identity guard passed at feature/logit/prediction level; no run ever hit NaN/Inf — every model trained at fallback level 0, so the entire ladder stayed unused; validation ordering transferred to test (the selected winners were also the better test performers in every setting); and test was evaluated once per locked checkpoint at the final pass of each phase — no selection depended on any test read.
"""),
    ("markdown", """
### Optional extension — unfreezing the classifier, with its attribution control

Run only after the mandatory comparison was complete (ADR 0008 §9): joint FM + classifier fine-tuning (classifier lr 1e-4, FM lr 1e-3, same policy), against the **classifier-only continued-training control** — the pinned probe trained further *alone* with the same budget.
"""),
    ("code", """
display(Markdown((REPO / "results" / "metrics" / "stage3_joint_table.md")
                 .read_text(encoding="utf-8")))
"""),
    ("markdown", """
**The control earns its place — and inverts the expected story.** The joint variant was pre-registered as an "upper reference"; it is not one. On FGVC-Aircraft, joint fine-tuning gains +0.56 ± 0.98 — consistent with no difference from the control's +0.64 ± 0.08 (per-seed joint − control: +0.06 / −1.08 / +0.78), i.e. **the data show no FM-attributable gain beyond what longer classifier training alone provides**. On DTD both are ≈0/slightly negative. Most strikingly, both sit far below the frozen-classifier Strategy 2 (+2.50 ± 0.50 on FGVC): with K = 10 images per class, unfreezing the classifier mostly buys near-immediate overfitting — the joint checkpoints land at epochs 0–18, single-digit in 5 of 6 runs — while the constrained frozen-classifier transport keeps improving for tens of epochs. The freeze is not a handicap here; it is the regularizer that makes the FM's contribution possible. (Without the control, the +0.56 could have looked like a small FM win — exactly the confound the external review flagged.)
"""),
    ("markdown", """
### Limitations

- **$n = 3$ subset seeds**; spreads are sample standard deviations; no significance claims. The spread measures subset-sampling variability only (probe-init and FM-init are fixed).
- **Seed 0 is partly a development run** — hyperparameters were selected on its validation split; the 3-seed mean is a summary, not an independent confirmatory estimate (pre-registered disclosure).
- **One K, one T, one encoder per dataset** — by the spec's own scoping ("to keep this stage focused"); conclusions are about this operating point.
- **The λ-ablation pair can degenerate**: where λ = 0 wins the sweep, the with/without-regularization contrast is absent from the test table by construction (stated in ADR 0008 §7).
- **2-D projections are qualitative**; the joint-PCA plane explains part of the variance only.
- The frozen classifier bounds what transport can achieve: FM re-arranges inputs to a fixed boundary; it cannot add capacity.

### Deviations from, and extensions beyond, the specification

| Item | Status | Note |
|---|---|---|
| Probe trained first exactly as Stage 1, then frozen | as specified | retrained per (dataset, seed) with the Stage-1 recipe; validation audit vs `runs.csv`: 0.000 pts and exact best-epoch match on all six probes; weights pinned |
| FM close to identity | as specified (strengthened) | zero final layer ⇒ **exact** identity, asserted at feature/logit/prediction level (§2) |
| Velocity net + Euler as Stage 2; single T; K = 10 | as specified | T = 4 pre-registered (shallower backprop for Strategy 1); K = the spec's suggested default |
| Both strategies + main comparison + deliverables | as specified | table + Δ; train/val curves both methods; joint-embedding feature viz |
| Relative displacement penalty (S1); trust region, monotone acceptance, per-epoch cached targets (S2) | within spec freedom | the spec explicitly invites regularization (S1) and constrained target updates (S2); all details pre-registered in ADR 0008 |
| Validation-accuracy checkpointing | our choice (spec silent) | symmetric to how the baseline probe itself was selected; identical rule for both strategies |
| Repetition protocol (3 subset seeds) | our choice (spec silent) | keeps paired Δ ± std, consistent with Stages 1–2 |
| Optional joint fine-tuning extension + classifier-only control | done (above) | run after the mandatory comparison, exactly per ADR 0008 §9; the control shows the joint gain is fully explained by longer classifier training |

### Artifact paths

| Artifact | Path |
|---|---|
| Per-run and summary results | `results/metrics/runs_stage3.csv`, `summary_stage3.csv`, `raw/stage3_*.npy` |
| Selection transparency | `results/metrics/stage3_sweep.csv` (+ generated `stage3_*_table.md`) |
| Training histories (every model, incl. failed attempts) | `results/artifacts/curves_stage3/*.json` |
| Pinned probes + trained FM models | `results/artifacts/stage3_models/*.pt` (gitignored, regenerable via `tasks.ps1 run3`) |
| Run-0 predictions | `results/artifacts/predictions/run3_*.npz` |
| Figures | `results/figures/stage3_*.png` |

Reproduce with `tasks.ps1 run3 / tables3 / figures3 / notebook3`; `tasks.ps1 check` re-derives every summary number, every paired Δ (against the stored pinned-probe baselines), and every generated table from raw artifacts.

**Provenance.** This notebook re-executes only inside its repository (`alonengel/Computer-Vision-Project`, private — available on request); as a standalone file it is fully readable (all outputs embedded) but not re-runnable — the repo, not the notebook, is the unit of reproduction.
"""),
]
