CELLS = [
    ("markdown", """
## 6 · Discussion

Every observation below is read directly off §3–§5; the numbers appear in the generated tables and are re-derived from raw artifacts by the repro check.

**1 · The stage's question — "can FM transform frozen features into a representation the existing classifier handles better?" — gets a split answer that matches the project-wide pattern.** On **FGVC-Aircraft/DINOv2**, yes: the Guided strategy gains **+2.50 ± 0.50 points** over the pinned probe with **all three seeds agreeing** (per-seed +2.76 / +1.92 / +2.82), and the Rolled strategy a smaller +0.88 ± 0.23 (3/3). On **DTD/ResNet-18**, no: every method sits within noise of the probe (Guided +0.16 ± 0.45, 2/3 seeds; Rolled −0.69 ± 0.70, 1/3). This is the third stage in a row where the same geography appears — the fine-grained DINOv2 representation still holds structure a fixed linear boundary has not extracted, while DTD showed no benefit at this operating point (K = 10, this encoder, this classifier — whether its representation is exhausted in general is not established here). The gains land inside the range our pre-registered expectations called plausible ("low single digits, possibly ≈0 on DTD") — the honest headline is a *modest, consistent* improvement in one setting, not a breakthrough.

**2 · The most interesting finding: the indirect strategy beats the direct one, on both datasets.** The Guided strategy — which never lets a CE gradient touch the FM, and instead distills conservative, trust-region-bounded, CE-monotone targets through a standard velocity-regression loss — outperforms the Rolled strategy's direct end-to-end CE backprop everywhere (FGVC +2.50 vs +0.88; DTD +0.16 vs −0.69; and the same ordering already appeared on the validation sweeps of both datasets before any test data was seen). Why remains open: the Guided strategy differs from the Rolled one through **several components at once** — the source-centred trust region, monotone target selection, classifier-guided target construction, and standard path-regression training — and this experiment does not isolate which is responsible. Its advantage is *consistent with* stronger regularization from bounded classifier-guided targets and path-regression training: both strategies memorize the training set (§4 — train accuracy ≈100% in both), so the difference lies in what each objective does off the training points, and everything that differs there pushes the Guided strategy toward conservatism. The resemblance to Stage 2's standard-vs-rolled-out result (path supervision generalizing where endpoint-only objectives overfit) is suggestive, but it is a pattern across experiments, not an isolated mechanism.

**3 · The displacement regularizer earns its place in the Rolled strategy, mildly.** The pre-registered pair puts λ = 1 above λ = 0 on both datasets (FGVC +0.88 ± 0.23 vs +0.50 ± 0.95; DTD −0.69 ± 0.70 vs −0.80 ± 0.98) — small mean differences; the three observed seeds show ≈4× lower variability with λ = 1 on FGVC (three seeds cannot establish stabilization in general). Note the geometry is not "smaller movement wins": at their selected checkpoints, the Guided strategy moves features *more* than the regularized Rolled strategy on FGVC (≈9% vs ≈2.4% of the feature norm; mean ‖ẑ−z‖ ≈ 4.6 vs 1.2 units) yet generalizes better — displacement magnitude alone does not explain performance; the movement direction and/or the training construction may matter.

**4 · The protocol held up end to end.** All six probes reproduced Stage 1 — recorded validation and test accuracies within $10^{-12}$ of the published values (§2 displays the comparison; the run log additionally recorded identical best epochs); the identity guard passed at feature/logit/prediction level; no run ever hit NaN/Inf — every model trained at fallback level 0, so the entire ladder stayed unused; every comparison that actually reached test preserved its validation ordering (Guided > Rolled on both datasets; λ = 1 > λ = 0 on both — non-winning Guided configurations were never test-evaluated, so nothing broader is claimed); and test was evaluated once per locked checkpoint, in one final pass per pre-registered phase (the mandatory grid; then the optional extension) — no selection depended on any test read.
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
**The control earns its place — and inverts the expected story.** The joint variant was pre-registered as an "upper reference"; it is not one. The joint variant does not outperform classifier-only continued training in these three seeds (FGVC: +0.56 ± 0.98 vs +0.64 ± 0.08; per-seed joint − control: +0.06 / −1.08 / +0.78; DTD: both ≈0/slightly negative), so **the results provide no evidence of an additional FM-attributable gain when the classifier is unfrozen**. The early validation-selected checkpoints (epochs 0–18, single-digit in 5 of 6 runs) are consistent with rapid overfitting in the K = 10 regime, but this experiment does not isolate classifier freezing as the cause — particularly because the best frozen configuration uses the Guided strategy while joint training uses a different objective. Even the closest matched-objective comparison (frozen Rolled strategy vs joint, both CE-trained) does not separate the two (+0.88 ± 0.23 vs +0.56 ± 0.98). **Empirically, the frozen-classifier Guided strategy remains the best tested configuration.** (Without the control, the +0.56 could have looked like a small FM win — exactly the confound the external review flagged.)
"""),
    ("markdown", """
**Training behaviour of the extension (seed 0; validation-selected checkpoints starred).** Same conventions as §4 — dashed = train, solid = validation; the control's train curves are the post-epoch full-train metrics recorded by `run_stage3_joint.py` (the extension was re-executed unchanged on 2026-09-10 to checkpoint its models; every recorded number reproduced byte-identically — lab notebook). Both variants reach ≈100% train accuracy within a few epochs; the difference is what happens afterwards. With the classifier unfrozen, the joint objective keeps pushing: validation cross-entropy climbs monotonically (≈4.6 on FGVC-Aircraft, ≈7 on DTD by epoch 200), validation accuracy drifts down (to ≈51.5% / ≈42.6%), and the FM's mean displacement grows from ≈11% / ≈5% of the mean feature norm at the checkpoint (5.6 / 1.3 units) to ≈135% / ≈200% by epoch 200 (≈67 / ≈48 units; all seeds 63–74 / 48–53) — larger than the features themselves (mean norms ≈50 on DINOv2, ≈24 on DTD). Nothing bounds the movement once CE is the only signal *and* the boundary can move too, so validation selects the earliest epochs (0–18). The classifier-only control changes slowly (validation CE ≈2.3 → 2.6 on FGVC-Aircraft, ≈1.8 → 1.95 on DTD) and checkpoints late on FGVC-Aircraft (epochs 171–197): the +0.64 it gains there is ordinary continued training — which is exactly why it accounts for the joint result.
"""),
    ("code", """
for ds, enc in cfg["stage3"]["settings"]:
    p = REPO / "results" / "figures" / f"stage3_curves_joint_{ds}_{enc}.png"
    if p.exists():
        display(Image(str(p), width=980))
"""),
    ("markdown", """
**Feature space of the extension (seed 0).** The shared ten `viz_selection` classes, same test examples and colours as §5; one PCA fitted jointly on $[z, \\hat{z}_{\\text{Guided}}, \\hat{z}_{\\text{joint}}]$ in raw space (a different joint fit than §5's, hence the slightly different plane: 32.6% / 15.8% of the variance on FGVC-Aircraft / DTD). At its validation-selected checkpoint the jointly trained FM has moved the features about as little as the Guided strategy (≈11% vs ≈9% of the mean feature norm on FGVC-Aircraft — 5.6 vs 4.7 units; ≈5% vs ≈10% on DTD — 1.3 vs 2.3 units), and the panels are correspondingly hard to tell apart — the checkpoint captures the model *before* the drift seen above. Qualitative only, as throughout.
"""),
    ("code", """
for ds, enc in cfg["stage3"]["settings"]:
    p = REPO / "results" / "figures" / f"stage3_features_joint_{ds}_{enc}.png"
    if p.exists():
        display(Image(str(p), width=980))
"""),
    ("markdown", """
**Overlay view of the extension.** Same construction as §5's overlay — original $z$ faded, transported $\\hat{z}$ solid, thin grey segments = per-example displacements — in the extension's joint-PCA plane. A difference in kind appears: the jointly trained FM's visible movement is *class-coherent* for a few classes (on FGVC-Aircraft the DHC-1 and Cessna 172 examples shift together in one direction; on DTD the knitted and bubbly examples do) while the remaining classes barely move, whereas the Guided strategy's segments stay short and directionally mixed. That is what one expects when the classifier boundary is itself trainable and CE is the only signal — whole classes get pushed rather than individual points nudged — but it is a qualitative observation only.
"""),
    ("code", """
for ds, enc in cfg["stage3"]["settings"]:
    p = REPO / "results" / "figures" / f"stage3_features_joint_overlay_{ds}_{enc}.png"
    if p.exists():
        display(Image(str(p), width=980))
"""),
    ("markdown", """
### Limitations

- **$n = 3$ subset seeds**; spreads are sample standard deviations; no significance claims. The spread measures subset-sampling variability only (probe-init and FM-init are fixed).
- **Seed 0 is partly a development run** — hyperparameters were selected on its validation split; the 3-seed mean is a summary, not an independent confirmatory estimate (pre-registered disclosure).
- **One K, one T, one encoder per dataset** — by the spec's own scoping ("to keep this stage focused"); conclusions are about this operating point.
- **The λ-ablation pair can degenerate**: where λ = 0 wins the sweep, the with/without-regularization contrast is absent from the test table by construction (stated in ADR 0008 §7).
- **2-D projections are qualitative**; the joint-PCA plane explains 15.6% (DTD) / 33.0% (FGVC) of the variance only.
- **The Strategy-2 trust region was active for nearly all targets after the early epochs** (hit rate ≈100% in the per-epoch training histories, `results/artifacts/curves_stage3/*.json`, drawn in `results/figures/stage3_diag_*.png`). Consequently, performance may depend materially on the fixed choice α = 0.1; no α ablation was performed (future work, or a clearly-labelled validation-only exploration).
- Although the linear classifier is frozen, **the FM adds nonlinear capacity** by warping feature space — $W F_\\theta(z) + b$ can represent nonlinear decision boundaries even with $W, b$ fixed. The conclusions are therefore limited to the chosen FM architecture, K = 10, T = 4, and the fixed trust-region radius — not to "what a frozen classifier can do" in general.

### Deviations from, and extensions beyond, the specification

| Item | Status | Note |
|---|---|---|
| Probe trained first exactly as Stage 1, then frozen | as specified | retrained per (dataset, seed) with the Stage-1 recipe; validation audit vs `runs.csv`: 0.000 pts and exact best-epoch match on all six probes; weights pinned |
| FM close to identity | as specified (strengthened) | zero final layer ⇒ **exact** identity, asserted at feature/logit/prediction level (§2) |
| Velocity net + Euler as Stage 2; single T; K = 10 | as specified | T = 4 pre-registered (shallower backprop for the Rolled strategy); K = the spec's suggested default |
| Both strategies + main comparison + deliverables | as specified | table + Δ; train/val curves both methods; joint-embedding feature viz |
| Relative displacement penalty (Rolled); trust region, monotone acceptance, per-epoch cached targets (Guided) | within spec freedom | the spec explicitly invites regularization (Rolled) and constrained target updates (Guided); all details pre-registered in ADR 0008 |
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
