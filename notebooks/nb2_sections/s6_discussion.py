CELLS = [
    ("markdown", """
## 6 · Discussion

(The observations here are read directly off §3–§5; the numbers are loaded from `results/metrics/`, never typed by hand. NARRATIVE-PLACEHOLDER — finalized from the completed grid.)
"""),
    ("markdown", """
### Limitations

- **$n = 3$ runs per setting.** Spreads are sample standard deviations; no confidence intervals are quoted. Per-seed pairing (where genuine) is stated with sign counts instead.
- **At K = full the image-branch deltas are unpaired** — all three FM initialization seeds are compared against the single deterministic full-split baseline, so their spread measures FM training stochasticity only. The CLIP‡ deltas are never paired (single zero-shot reference).
- **The CLIP‡ branch cannot isolate the FM effect.** Its reference consumes zero labeled images while FM consumes K per class; a labeled CLIP image-prototype baseline is excluded by the Stage-1 protocol (CLIP is restricted to the zero-shot branch). The ‡ deltas conflate "value of K labels" with "value of transport".
- **No hyperparameter search** — the Stage-1 probe recipe was adopted verbatim a priori. Better FM numbers are plausibly reachable; tuning was out of scope (spec) and would have to use validation data only.
- **Final-epoch models, no selection.** Justified in §1 and verified stable in §4; a validation-selected checkpoint could only be added symmetrically for both modes.
- **2-D projections are qualitative**; the joint-PCA plane explains only part of the variance and trajectory geometry off-plane is invisible.

### Deviations from, and extensions beyond, the specification

| Item | Status | Note |
|---|---|---|
| FM on $L_2$-normalized features | **disclosed deviation** | fixed a priori (ADR 0007 §3); the cosine classifier acts on the sphere; T = 0 reproduces Stage 1 exactly (§2 guard) |
| Datasets | as specified | the Stage-1 selected pair DTD + FGVC-Aircraft (ADR 0006); "same datasets as Stage 1" |
| Prototype branch | as specified + extension ‡ | spec branch = image prototypes (Stage-1 Option A selection); CLIP‡ text-prototype transport added as a marked extension |
| Velocity network / T grid / losses | as specified | MLP 2×512 SiLU, scalar t concatenated; T ∈ {4, 12}; per-sample squared $L_2$ losses |
| Training configuration | suggested-scope | Stage-1 probe recipe reused verbatim, no search; final-epoch model (no FM selection rule exists in the spec), stability verified over every curve |
| Runs / seeds / metric | as specified | Stage-1 repetition protocol mirrored exactly; top-1 on the complete official test split |

### Artifact paths

| Artifact | Path |
|---|---|
| Per-run and summary results | `results/metrics/runs_stage2.csv`, `summary_stage2.csv`, `raw/fm_*.npy` |
| Generated tables | `results/metrics/stage2_*_table.md` |
| FM training-loss histories (every model) | `results/artifacts/curves_stage2/*.json` |
| Trained velocity networks | `results/artifacts/fm_models/*.pt` (gitignored, regenerable via `tasks.ps1 run2`) |
| Run-0 predictions | `results/artifacts/predictions/run2_*.npz` |
| Figures | `results/figures/stage2_*.png` |

Reproduce with `tasks.ps1 run2` (grid), `tables2`, `figures2`, `notebook2`; `tasks.ps1 check` re-derives every summary number, every Δ, and every generated table from the raw artifacts and the Stage-1 `runs.csv`.
"""),
]
