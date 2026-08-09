CELLS = [
    ("markdown", """
## 3 · Classification results

Every number below is loaded from `results/metrics/` (generated programmatically, never typed by hand); the raw per-run accuracies live in `results/metrics/raw/*.npy` and the per-run table in `runs_stage2.csv`, so every claim is independently checkable. $\\Delta$ = change vs the Stage-1 baseline of the **identical** setting.

### 3.1 Spec branch — FM toward image-derived prototypes
"""),
    ("code", """
display(Markdown((REPO / "results" / "metrics" / "stage2_image_prototype_table.md")
                 .read_text(encoding="utf-8")))
"""),
    ("markdown", """
**Per-seed-paired deltas.** At $K \\in \\{5, 10\\}$ the FM run and its baseline share the same committed subset indices and prototypes, so the per-seed difference is matched — subset-sampling noise cancels — and its spread is far tighter than the marginal spreads. (At full, the baseline is a single deterministic run; the three FM initialization seeds are compared against that one number, so those deltas are *not* paired and their spread measures FM training stochasticity only. $n = 3$ throughout — no confidence intervals are quoted at this sample size.)
"""),
    ("code", """
display(Markdown((REPO / "results" / "metrics" / "stage2_paired_delta_table.md")
                 .read_text(encoding="utf-8")))
"""),
    ("code", """
for name in ("stage2_acc_dtd_resnet18_image_prototype.png",
             "stage2_acc_fgvc_aircraft_resnet18_image_prototype.png",
             "stage2_acc_fgvc_aircraft_dinov2_vits14_image_prototype.png"):
    p = REPO / "results" / "figures" / name
    if p.exists():
        display(Image(str(p), width=880))
"""),
    ("markdown", """
### 3.2 Extension ‡ — FM on CLIP features toward text prototypes

FM here transports **CLIP RN50 image features** toward the fixed **text** prototypes. Read these rows with care: unlike the Stage-1 zero-shot reference, FM training consumes $K$ labeled images per class, so this is *supervised transport on CLIP features* — never "zero-shot". $\\Delta$ therefore answers *"does supervised transport toward text prototypes beat zero-shot classification?"*, **not** *"does the FM layer help?"* — only the image-prototype branch isolates the FM effect, because its baseline uses the identical labeled subset.
"""),
    ("code", """
display(Markdown((REPO / "results" / "metrics" / "stage2_clip_text_table.md")
                 .read_text(encoding="utf-8")))
"""),
    ("code", """
for name in ("stage2_acc_dtd_clip_rn50_clip_text.png",
             "stage2_acc_fgvc_aircraft_clip_rn50_clip_text.png"):
    p = REPO / "results" / "figures" / name
    if p.exists():
        display(Image(str(p), width=880))
"""),
    ("markdown", """
### 3.3 Reading the grid — computed, not asserted

The cell below reduces the full grid to the quantities the discussion rests on: per (branch, head, $T$) the number of settings with a positive mean delta, and the extreme deltas.
"""),
    ("code", """
r2 = pd.read_csv(REPO / "results" / "metrics" / "summary_stage2.csv")
g = (r2.assign(**{"Δ mean (pts)": (100 * r2["delta_mean"]).round(2)})
       .groupby(["target", "head", "T"])
       .agg(**{"settings": ("delta_mean", "size"),
               "Δ > 0": ("delta_mean", lambda d: int((d > 0).sum())),
               "best Δ (pts)": ("delta_mean", lambda d: round(100 * d.max(), 2)),
               "worst Δ (pts)": ("delta_mean", lambda d: round(100 * d.min(), 2)),
               "mean Δ (pts)": ("delta_mean", lambda d: round(100 * d.mean(), 2))}))
display(g)
"""),
]
