# v2 talk — appendix: detailed evidence for questions (English only).

APPENDIX_META = {"jp-MarkdownHeadingCollapsed": True, "heading_collapsed": True}

A0 = r"""
# <span style="color:#1f3a5f;">Appendix — detailed evidence for questions</span>

Reference material moved out of the main flow. The complete scientific report remains `stage3_presentation.ipynb`.

- **A.1** complete environment and configuration, including the training and fallback policy
- **A.2** exact-identity audit (re-executed) · **A.3** pinned-probe reproduction audit
- **A.4** main table of record with its full caption · **A.5** full seed-0 validation sweep · **A.6** λ ablation · **A.7** per-seed paired deltas
- **A.8** figures of record and strategy-internal diagnostics · **A.9** checkpoint and displacement table
- **A.10** optional joint-training experiment · **A.11** classifier-only continued-training control
- **A.12** full limitations · **A.13** deviations from the specification · **A.14** artifact paths and reproduction commands
"""

A1_HEAD = r"""
## <span style="color:#1f3a5f;">A.1 · Complete environment and configuration</span>
"""

A1_CODE = r"""
# Runtime provenance and the Stage-3 configuration block, as recorded in the repository.
with open(REPO / "results" / "runtime_summary.json") as f:
    print("Runtime:", json.dumps(json.load(f), indent=2))
s3 = cfg["stage3"]
print("\nSettings:", s3["settings"], "| K =", s3["k_shot"], "| T =", s3["T"],
      "| feature space:", s3["feature_space"])
print("Training policy:", json.dumps(s3["training"], indent=2))
print("Strategy 1 lambda grid:", s3["strategy1"]["lambda_grid"],
      "| Strategy 2 grid: beta", s3["strategy2"]["beta_grid"], "x m", s3["strategy2"]["m_grid"],
      "| trust-region alpha =", s3["strategy2"]["alpha_trust_region"])
"""

A1_TEXT = r"""
**Provenance of every choice.** The FM operates in **raw feature space** — the probe was trained on raw features ($s = Wz + b$, norms ≈ 10–40), so normalizing on entry would break the required identity-at-initialization (note the deliberate contrast with Stage 2, where the downstream classifier was cosine-based and the FM therefore lived on the unit sphere: *the FM operates in the space its downstream classifier acts in*). The velocity network and Euler procedure are the Stage-2 design, as the spec instructs (MLP $d{+}1 \to 512 \to 512 \to d$, SiLU, scalar $t$ concatenated; $\hat{z}_{k+1} = \hat{z}_k + \frac{1}{T}v_\theta(\hat{z}_k, k/T)$). **Identity initialization is exact**: the final velocity layer (weight *and* bias) starts at zero, so $\hat{z} = z$ bit-for-bit — asserted, not assumed (A.2).

**Training policy (uniform for both strategies, fixed a priori):** AdamW, lr $10^{-3}$, weight decay $10^{-4}$, batch 64, exactly 200 epochs with the **best-validation-accuracy checkpoint** retained (ties → lowest validation CE → earliest epoch). Validation-based selection is deliberate and symmetric: the baseline probe itself was selected the same way in Stage 1, and both strategies expose the identical full-pipeline validation curve. Failure policy (objective trigger — NaN/Inf loss only): restart with the next pre-registered fallback (cumulative ladder: grad-clip 1.0 → lr 3e-4 → internal input standardization, which preserves the exact identity); finite late deterioration is absorbed by checkpointing; exhausted ladder ⇒ run marked failed. **Hyperparameters were selected per dataset on seed-0 validation only**; winners applied unchanged to seeds 1–2; the **test split stayed sealed** during all training and selection, and was evaluated in **one final pass per pre-registered phase** (the mandatory grid; then the optional extension), each over its already-locked checkpoints. Disclosed: seed 0 is therefore partly a development run, and the 3-seed mean is a summary, not an independent confirmatory estimate.
"""

A2_HEAD = r"""
## <span style="color:#1f3a5f;">A.2 · Exact-identity audit</span>

Transported features must be **bit-exact equal** to the inputs at initialization, with equal logits and identical predictions between the pipeline and the direct probe — stronger than accuracy equality (two systems can share accuracy while disagreeing on many predictions). Re-executed live below on validation data only.
"""

A2_CODE = r"""
import torch

from src.embeddings import load_features
from src.stage3 import Stage3FM, identity_guard, load_pinned_probe

print("Exact-identity guard (re-executed live, validation data only):")
for ds, enc in cfg["stage3"]["settings"]:
    probe = load_pinned_probe(MODELS / f"probe_{ds}_{enc}_{K_LABEL}_seed0.pt")
    fval = load_features(ds, "val", enc)
    fm0 = Stage3FM(probe, fval["dim"], seed=cfg["stage3"]["fm_init_seed"])
    identity_guard(fm0, fval["features"][:256])
    print(f"  [OK] {ds} / {enc}: transported features bit-exact, logits and "
          f"predictions identical to the direct probe")
"""

A3_HEAD = r"""
## <span style="color:#1f3a5f;">A.3 · Pinned-probe reproduction audit</span>

The probe is retrained with the identical Stage-1 recipe per (dataset, subset seed), its **validation** accuracy audited against `runs.csv` (|diff| < 0.25 pts; the run log shows 0.000 pts and identical best epochs on all six probes), and the weights pinned so every method receives the same frozen classifier. Δ is computed against **this exact pinned probe** — `runs.csv` serves as a reproduction audit only; the cell below displays the recorded accuracies reproducing the published values within $10^{-12}$ (test compared post-hoc at the final pass).
"""

A3_CODE = r"""
# Pinned probes' validation and test accuracies against the published Stage-1 values.
r1 = pd.read_csv(MET / "runs.csv")
r3 = pd.read_csv(MET / "runs_stage3.csv")
rows = []
for ds, enc in cfg["stage3"]["settings"]:
    for seed in cfg["stage3"]["subset_seeds"]:
        pin = r3[(r3["dataset"] == ds) & (r3["encoder"] == enc)
                 & (r3["head"] == "pinned_probe") & (r3["seed"] == seed)].iloc[0]
        ref = r1[(r1["dataset"] == ds) & (r1["encoder"] == enc)
                 & (r1["head"] == "linear_probe") & (r1["k_shot"] == K_LABEL)
                 & (r1["seed"] == seed)].iloc[0]
        rows.append({"dataset": ds, "seed": seed,
                     "pinned val (%)": round(100 * pin["val_acc"], 4),
                     "Stage-1 val (%)": round(100 * ref["val_acc"], 4),
                     "pinned test (%)": round(100 * pin["test_acc"], 4),
                     "Stage-1 test (%)": round(100 * ref["test_acc"], 4),
                     "within 1e-12": bool(np.isclose(pin["val_acc"], ref["val_acc"],
                                                     rtol=0, atol=1e-12)
                                          and np.isclose(pin["test_acc"],
                                                         ref["test_acc"],
                                                         rtol=0, atol=1e-12))})
audit = pd.DataFrame(rows)
assert audit["within 1e-12"].all()
print("Pinned-probe reproduction audit — recorded validation and test accuracies "
      "reproduce the published Stage-1 values within 1e-12 on all six probes "
      "(equal accuracies do not by themselves prove identical weights; the "
      "feature-level bit-exactness claim belongs to the identity guard above):")
display(audit)
"""

A4_HEAD = r"""
## <span style="color:#1f3a5f;">A.4 · Main table of record (with its full caption)</span>

Every number is generated programmatically from `results/metrics/` and re-derived by the repro check; raw per-run accuracies in `results/metrics/raw/stage3_*.npy`, the per-run table in `runs_stage3.csv`. Δ is **paired per seed** against the exact pinned probe of that seed.
"""

A4_CODE = r"""
display(Markdown((MET / "stage3_main_table.md").read_text(encoding="utf-8")))
"""

A5_HEAD = r"""
## <span style="color:#1f3a5f;">A.5 · Full seed-0 validation sweep</span>

Everything the winners were chosen from, so the selection involves no undisclosed freedom. Test was untouched during selection.
"""

A5_CODE = r"""
display(Markdown((MET / "stage3_sweep_table.md").read_text(encoding="utf-8")))
"""

A6_HEAD = r"""
## <span style="color:#1f3a5f;">A.6 · λ ablation — Strategy 1 with vs without the displacement regularizer (pre-registered pair)</span>
"""

A6_CODE = r"""
display(Markdown((MET / "stage3_lambda_ablation_table.md").read_text(encoding="utf-8")))
"""

A7_HEAD = r"""
## <span style="color:#1f3a5f;">A.7 · Per-seed paired deltas</span>
"""

A7_CODE = r"""
# Per-seed paired deltas (percentage points): the quantity every claim rests on.
HEAD_SHOW = {"fm_s1": "Strategy 1", "fm_s1_lambda0": "Strategy 1, λ=0",
             "fm_s2": "Strategy 2"}
r3 = pd.read_csv(MET / "runs_stage3.csv")
ok = r3[r3["status"] == "ok"]
print("Per-seed paired deltas (percentage points), the quantity every claim rests on:")
t = (ok[ok["head"] != "pinned_probe"]
     .assign(setting=lambda d: d["dataset"].map(dataset_label) + " / "
             + d["encoder"].map(lambda e: encoder_label(e, short=True)),
             method=lambda d: d["head"].map(HEAD_SHOW),
             delta_pts=lambda d: (100 * d["delta_acc"]).round(2))
     .pivot_table(index=["setting", "method"], columns="seed", values="delta_pts"))
display(t)
"""

A8_HEAD = r"""
## <span style="color:#1f3a5f;">A.8 · Figures of record and strategy-internal diagnostics</span>

**End-to-end curves of record** (Sections 7–8 re-render exactly these seed-0 histories): full-pipeline cross-entropy and top-1 accuracy of $z \to \mathrm{FM} \to$ frozen probe, on train and validation, every epoch. Dashed = train, solid = validation; blue = Strategy 1, orange = Strategy 2.
"""

A8_CURVES_CODE = r"""
for ds, enc in cfg["stage3"]["settings"]:
    display(Image(str(FIG / f"stage3_curves_{ds}_{enc}.png"), width=980))
"""

A8_DIAG_TEXT = r"""
**Strategy-internal diagnostics — separate axes, deliberately.** These quantities measure different things and are never drawn on a shared axis: Strategy 1's relative displacement penalty; Strategy 2's FM-regression loss; Strategy 2's target construction (CE of the unprojected snapshot output — diagnostic only; CE at the projected $u_0$; CE at the selected target) with the **trust-region hit rate** (fraction of samples whose target sits on the boundary $\lVert\hat{z}' - z\rVert = \rho$) on the twin axis; and the mean displacements. Note the S2 phase-1 values describe the *pre-update* snapshot of each epoch, while its validation values describe the post-update model — a one-phase offset inherent to the two-phase scheme.
"""

A8_DIAG_CODE = r"""
for ds, enc in cfg["stage3"]["settings"]:
    display(Image(str(FIG / f"stage3_diag_{ds}_{enc}.png"), width=1150))
"""

A8_PCA_TEXT = r"""
**Feature-space figures of record** (Sections 9–10 re-render the same selection, models and joint PCA with larger fonts; the explained-variance labels are asserted to match these):
"""

A8_PCA_CODE = r"""
for ds, enc in cfg["stage3"]["settings"]:
    display(Image(str(FIG / f"stage3_features_{ds}_{enc}.png"), width=1150))
"""

A9_HEAD = r"""
## <span style="color:#1f3a5f;">A.9 · Checkpoint placement and displacement</span>
"""

A9_CODE = r"""
# Checkpoint epoch from the table of record (runs_stage3.csv), which applies the
# full tie-break rule (validation accuracy -> validation CE -> earliest epoch).
HEADS = {"fm_s1": "Strategy 1", "fm_s2": "Strategy 2"}
r3 = pd.read_csv(MET / "runs_stage3.csv")
rows = []
for ds, enc in cfg["stage3"]["settings"]:
    for head in ("fm_s1", "fm_s2"):
        for seed in cfg["stage3"]["subset_seeds"]:
            with open(CURVES / f"{ds}_{enc}_{head}_seed{seed}.json") as f:
                h = json.load(f)
            ep = int(r3[(r3["dataset"] == ds) & (r3["encoder"] == enc)
                        & (r3["head"] == head)
                        & (r3["seed"] == seed)]["checkpoint_epoch"].iloc[0])
            rows.append({"dataset": dataset_label(ds), "strategy": HEADS[head],
                         "seed": seed,
                         "best val acc (%)": round(100 * max(h["val_acc"]), 2),
                         "checkpoint epoch": ep,
                         "‖ẑ−z‖ at checkpoint": round(h["mean_disp"][ep], 3)})
print("Checkpoint placement and displacement — the two mandatory strategies' "
      "final models (λ=0 and the optional-extension models are tabulated in "
      "A.6 / A.10):")
display(pd.DataFrame(rows))
failed = sorted(CURVES.glob("*_failed_lv*.json"))
print(f"Failed training attempts (partial curves saved): {len(failed)}"
      + (" — " + ", ".join(p.name for p in failed) if failed else
         " — no NaN/Inf occurred; every run completed at fallback level 0 "
         "(the default recipe)."))
"""

A10_HEAD = r"""
## <span style="color:#1f3a5f;">A.10 · Optional joint-training experiment</span>

Run only after the mandatory comparison was complete (ADR 0008 §9): joint FM + classifier fine-tuning (classifier lr 1e-4, FM lr 1e-3, same policy), against the **classifier-only continued-training control** — the pinned probe trained further *alone* with the same budget.
"""

A10_CODE = r"""
display(Markdown((MET / "stage3_joint_table.md").read_text(encoding="utf-8")))
"""

A11_TEXT = r"""
## <span style="color:#1f3a5f;">A.11 · Classifier-only continued-training control</span>

**The control earns its place — and inverts the expected story.** The joint variant was pre-registered as an "upper reference"; it is not one. The joint variant does not outperform classifier-only continued training in these three seeds (FGVC: +0.56 ± 0.98 vs +0.64 ± 0.08; per-seed joint − control: +0.06 / −1.08 / +0.78; DTD: both ≈0/slightly negative), so **the results provide no evidence of an additional FM-attributable gain when the classifier is unfrozen**. The early validation-selected checkpoints (epochs 0–18, single-digit in 5 of 6 runs) are consistent with rapid overfitting in the K = 10 regime, but this experiment does not isolate classifier freezing as the cause — particularly because the best frozen configuration uses Strategy 2 while joint training uses a different objective. Even the closest matched-objective comparison (frozen Strategy 1 vs joint, both CE-trained) does not separate the two (+0.88 ± 0.23 vs +0.56 ± 0.98). **Empirically, frozen-classifier Strategy 2 remains the best tested configuration.** (Without the control, the +0.56 could have looked like a small FM win — exactly the confound the external review flagged.)
"""

A12_TEXT = r"""
## <span style="color:#1f3a5f;">A.12 · Full limitations</span>

- **$n = 3$ subset seeds**; spreads are sample standard deviations; no significance claims. The spread measures subset-sampling variability only (probe-init and FM-init are fixed).
- **Seed 0 is partly a development run** — hyperparameters were selected on its validation split; the 3-seed mean is a summary, not an independent confirmatory estimate (pre-registered disclosure).
- **One K, one T, one encoder per dataset** — by the spec's own scoping ("to keep this stage focused"); conclusions are about this operating point.
- **The λ-ablation pair can degenerate**: where λ = 0 wins the sweep, the with/without-regularization contrast is absent from the test table by construction (stated in ADR 0008 §7).
- **2-D projections are qualitative**; the joint-PCA plane explains 15.6% (DTD) / 33.0% (FGVC) of the variance only.
- **The Strategy-2 trust region was active for nearly all targets after the early epochs** (hit rate ≈100%, A.8 diagnostics). Consequently, performance may depend materially on the fixed choice α = 0.1; no α ablation was performed (future work, or a clearly-labelled validation-only exploration).
- Although the linear classifier is frozen, **the FM adds nonlinear capacity** by warping feature space — $W F_\theta(z) + b$ can represent nonlinear decision boundaries even with $W, b$ fixed. The conclusions are therefore limited to the chosen FM architecture, K = 10, T = 4, and the fixed trust-region radius — not to "what a frozen classifier can do" in general.
"""

A13_TEXT = r"""
## <span style="color:#1f3a5f;">A.13 · Deviations from, and extensions beyond, the specification</span>

| Item | Status | Note |
|---|---|---|
| Probe trained first exactly as Stage 1, then frozen | as specified | retrained per (dataset, seed) with the Stage-1 recipe; validation audit vs `runs.csv`: 0.000 pts and exact best-epoch match on all six probes; weights pinned |
| FM close to identity | as specified (strengthened) | zero final layer ⇒ **exact** identity, asserted at feature/logit/prediction level (A.2) |
| Velocity net + Euler as Stage 2; single T; K = 10 | as specified | T = 4 pre-registered (shallower backprop for Strategy 1); K = the spec's suggested default |
| Both strategies + main comparison + deliverables | as specified | table + Δ; train/val curves both methods; joint-embedding feature viz |
| Relative displacement penalty (S1); trust region, monotone acceptance, per-epoch cached targets (S2) | within spec freedom | the spec explicitly invites regularization (S1) and constrained target updates (S2); all details pre-registered in ADR 0008 |
| Validation-accuracy checkpointing | our choice (spec silent) | symmetric to how the baseline probe itself was selected; identical rule for both strategies |
| Repetition protocol (3 subset seeds) | our choice (spec silent) | keeps paired Δ ± std, consistent with Stages 1–2 |
| Optional joint fine-tuning extension + classifier-only control | done (A.10–A.11) | run after the mandatory comparison, exactly per ADR 0008 §9; the control shows the joint gain is fully explained by longer classifier training |
"""

A14_TEXT = r"""
## <span style="color:#1f3a5f;">A.14 · Artifact paths and reproduction commands</span>

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

**How this presentation notebook was built.** Assembled from `notebooks/nb3_talk_v2_sections/s*.py` by `notebooks/build_talk_notebook.py v2` and executed with `jupyter nbconvert --execute`; it reads only `results/` (tables, histories and figures of record) and the pinned models, and writes nothing back. The scientific notebook `stage3_presentation.ipynb`, its sections, the source code and every artifact are unchanged.
"""

CELLS = [
    ("markdown", A0, APPENDIX_META),
    ("markdown", A1_HEAD),
    ("code", A1_CODE),
    ("markdown", A1_TEXT),
    ("markdown", A2_HEAD),
    ("code", A2_CODE),
    ("markdown", A3_HEAD),
    ("code", A3_CODE),
    ("markdown", A4_HEAD),
    ("code", A4_CODE),
    ("markdown", A5_HEAD),
    ("code", A5_CODE),
    ("markdown", A6_HEAD),
    ("code", A6_CODE),
    ("markdown", A7_HEAD),
    ("code", A7_CODE),
    ("markdown", A8_HEAD),
    ("code", A8_CURVES_CODE),
    ("markdown", A8_DIAG_TEXT),
    ("code", A8_DIAG_CODE),
    ("markdown", A8_PCA_TEXT),
    ("code", A8_PCA_CODE),
    ("markdown", A9_HEAD),
    ("code", A9_CODE),
    ("markdown", A10_HEAD),
    ("code", A10_CODE),
    ("markdown", A11_TEXT),
    ("markdown", A12_TEXT),
    ("markdown", A13_TEXT),
    ("markdown", A14_TEXT),
]
