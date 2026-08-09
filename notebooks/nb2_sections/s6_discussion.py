CELLS = [
    ("markdown", """
## 6 · Discussion

The observations below are read directly off §3–§5. Every Stage-2 number appears in the generated tables and is re-derived from raw artifacts by the repro check; the Stage-1 reference values used for context (probe accuracy, validation headroom) are computed live from the Stage-1 `runs.csv` in the cell below, so this notebook remains self-contained.

**1 · The FM layer pays where Stage 1 measured headroom, and only there.** Stage 1's handoff table quantified how much accuracy the class-mean prototype leaves on the table relative to a trained boundary on the same features (validation headroom). The Stage-2 gains line up with it: FGVC-Aircraft/DINOv2 — the setting with by far the largest headroom (35.7 validation points) — gains **+10.4 to +23.2 points** (3/3 seeds at every K), lifting the full-split prototype pipeline from 34.4% to 57.6%; FGVC-Aircraft/ResNet-18 (moderate headroom) gains +1.6 to +6.7 with standard FM (3/3 seeds); DTD/ResNet-18 (smallest headroom, 5.3 validation points) gains only +0.8 at full — and *loses* 2.4–12.9 points in the low-shot settings, 0/3 seeds in favour. The mechanism is the one §5 shows geometrically: a trained nonlinear transport re-introduces discriminative structure that the class-mean estimator discards. Where the representation is rich and classes separable (DINOv2 on fine-grained aircraft), there is a lot to re-introduce; where classes are heavily entangled (textures), the learned basins misroute test features and transport *creates* errors — increasingly so at small K, where the 5–10 images per class that define both the prototypes and the flow cannot pin down basin boundaries.

**2 · Standard FM is the better or equal training mode on the spec branch.** On all three image-prototype settings (DTD: standard loses less; FGVC/ResNet-18: standard +1.6 to +6.7 vs rollout −0.2 to +1.4; DINOv2: statistically indistinguishable at K ∈ {5,10} — rollout's means are nominally higher in three of four cells there, well within the paired spreads — and standard ahead by 2.2–3.0 points at full). Rolled-out training also produced all three genuine training divergences (§4). Both facts have one root: the rolled-out objective constrains only the *endpoint* of the composed T-step map and provides no supervision along the path, so the learned field is freer to become degenerate; standard FM's per-point velocity supervision acts as a regularizer along the entire trajectory. The one place rolled-out training wins decisively is the CLIP‡ branch at full split on FGVC-Aircraft (+14.5 vs +6.9 at T = 4) — transporting across the image–text modality gap is precisely the regime where the ideal straight-line velocity field is most misspecified, and an endpoint-only objective is free to find a better curved transport (§5's trajectory figures show the curvature directly).

**3 · T barely matters for standard FM.** The two standard rows of every setting share one network; going from 4 to 12 Euler steps changes accuracy by at most a few tenths of a point (both signs occur). The learned velocity field is smooth enough that integration error is not the binding constraint. For rolled-out models T is a training choice, and on the image branch T = 12 is never better than T = 4 by more than noise — deeper backprop-through-time buys no measurable benefit there (all three divergences were rolled-out runs at K = full: two at T = 4, one at T = 12).

**4 · The CLIP‡ extension: labels + transport beat zero-shot everywhere, with the caveat stated in §3.2.** Supervised transport toward text prototypes exceeds the zero-shot reference in every cell (from +1.5 at FGVC 5-shot to +23.9 on DTD full). These deltas bundle the value of K labels with the value of transport and therefore say nothing about the FM layer in isolation; they do show that a frozen CLIP image encoder plus a small trained transport is a viable low-shot classifier, and that the modality gap is learnable to cross.

**5 · What this hands Stage 3.** Stage 3 places the FM module *before* a linear classifier and trains both jointly with cross-entropy. Stage 2's lessons carry directly: (i) the transport map has real discriminative capacity (DINOv2 +23), but (ii) even the best FM-to-prototype pipeline (57.6%) remains 9.6 points below the Stage-1 linear probe on the same features (67.2%, verified below) — prototype-targeted transport alone does not recover a trained boundary; and (iii) per-path supervision (standard) is the stable optimization regime, endpoint-only objectives (which Stage 3's CE-through-the-module resembles) need the stability safeguards §4 established.
"""),
    ("code", """
# Stage-1 reference values quoted above, computed live from Stage-1 artifacts
# so this notebook is self-contained.
r1 = pd.read_csv(REPO / "results" / "metrics" / "runs.csv")
full = r1[r1["k_shot"] == "full"]
rows = []
for ds, enc in (("dtd", "resnet18"), ("fgvc_aircraft", "dinov2_vits14")):
    g = full[(full["dataset"] == ds) & (full["encoder"] == enc)]
    probe_val = g[g["head"] == "linear_probe"]["val_acc"].mean()
    proto_val = float(g[g["head"] == "image_prototype"]["val_acc"].iloc[0])
    probe_test = g[g["head"] == "linear_probe"]["test_acc"].mean()
    rows.append({"Dataset / encoder": f"{ds} / {enc}",
                 "Stage-1 probe, full (test %)": round(100 * probe_test, 2),
                 "Validation headroom probe − prototype (pts)":
                     round(100 * (probe_val - proto_val), 2)})
print("Stage-1 reference values used in the discussion (from runs.csv):")
display(pd.DataFrame(rows))

s2best = pd.read_csv(REPO / "results" / "metrics" / "summary_stage2.csv")
b = s2best[(s2best["dataset"] == "fgvc_aircraft") & (s2best["encoder"] == "dinov2_vits14")
           & (s2best["k_shot"] == "full")]["mean_acc"].max()
probe = full[(full["dataset"] == "fgvc_aircraft") & (full["encoder"] == "dinov2_vits14")
             & (full["head"] == "linear_probe")]["test_acc"].mean()
print(f"Best FM-to-prototype (FGVC/DINOv2, full): {100*b:.2f}%  vs  "
      f"Stage-1 probe: {100*probe:.2f}%  ->  gap {100*(probe-b):.2f} pts")
"""),
    ("markdown", """
### Limitations

- **$n = 3$ runs per setting.** Spreads are sample standard deviations; no confidence intervals are quoted. Per-seed pairing (where genuine) is stated with sign counts instead.
- **At K = full the image-branch deltas are unpaired** — all three FM initialization seeds are compared against the single deterministic full-split baseline, so their spread measures FM training stochasticity only. The CLIP‡ deltas are never paired (single zero-shot reference).
- **The CLIP‡ branch cannot isolate the FM effect.** Its reference consumes zero labeled images while FM consumes K per class; a labeled CLIP image-prototype baseline is excluded by the Stage-1 protocol (CLIP is restricted to the zero-shot branch). The ‡ deltas conflate "value of K labels" with "value of transport".
- **No hyperparameter search** — the Stage-1 probe recipe was adopted verbatim a priori. Better FM numbers are plausibly reachable — particularly for rolled-out training, whose divergences suggest the shared learning rate sits near its stability edge for backprop-through-T — but tuning was out of scope (spec) and would have to use validation data only.
- **Checkpoint = minimum-training-loss epoch, uniformly** (the §4 contingency; a training-set-only criterion — no validation or test data enters any Stage-2 selection). The 1.05× trigger threshold proved tight for standard FM's stochastic per-epoch loss (most flags were benign last-epoch noise); the uniform fallback absorbs both benign and genuine cases without discretion.
- **2-D projections are qualitative**; the joint-PCA plane explains only part of the variance and trajectory geometry off-plane is invisible.

### Deviations from, and extensions beyond, the specification

| Item | Status | Note |
|---|---|---|
| FM on $L_2$-normalized features | **disclosed deviation** | fixed a priori (ADR 0007 §3); the cosine classifier acts on the sphere; T = 0 reproduces Stage 1 exactly (§2 guard) |
| Datasets | as specified | the Stage-1 selected pair DTD + FGVC-Aircraft (ADR 0006); "same datasets as Stage 1" |
| Prototype branch | as specified + extension ‡ | spec branch = image prototypes (Stage-1 Option A selection); CLIP‡ text-prototype transport added as a marked extension |
| Velocity network / T grid / losses | as specified | MLP 2×512 SiLU, scalar t concatenated; T ∈ {4, 12}; per-sample squared $L_2$ losses |
| Training configuration | suggested-scope | Stage-1 probe recipe reused verbatim, no search |
| Checkpoint = min-training-loss epoch | **pre-registered contingency, triggered** | §4: uniform over all models after 3 rolled-out divergences on the first grid; training-set-only criterion; the first grid's numbers were never published |
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
