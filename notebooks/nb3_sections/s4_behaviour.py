CELLS = [
    ("markdown", """
## 4 · Training behaviour

**Comparable end-to-end metrics on shared axes** (the spec's required training + validation curves, both methods): the full-pipeline cross-entropy and top-1 accuracy of $z \\to \\mathrm{FM} \\to$ frozen probe, on train and validation, every epoch. Representative run: subset seed 0 (the validation-selected winners). Dashed = train, solid = validation; blue = Strategy 1, orange = Strategy 2.
"""),
    ("code", """
for ds, enc in cfg["stage3"]["settings"]:
    p = REPO / "results" / "figures" / f"stage3_curves_{ds}_{enc}.png"
    if p.exists():
        display(Image(str(p), width=980))
"""),
    ("markdown", """
**Strategy-internal diagnostics — separate axes, deliberately.** These quantities measure different things and are never drawn on a shared axis: Strategy 1's relative displacement penalty; Strategy 2's FM-regression loss; Strategy 2's target construction (CE of the unprojected snapshot output — diagnostic only; CE at the projected $u_0$; CE at the selected target) with the **trust-region hit rate** (fraction of samples whose target sits on the boundary $\\lVert\\hat{z}' - z\\rVert = \\rho$) on the twin axis; and the mean displacements. Note the S2 phase-1 values describe the *pre-update* snapshot of each epoch, while its validation values describe the post-update model — a one-phase offset inherent to the two-phase scheme.
"""),
    ("code", """
for ds, enc in cfg["stage3"]["settings"]:
    p = REPO / "results" / "figures" / f"stage3_diag_{ds}_{enc}.png"
    if p.exists():
        display(Image(str(p), width=1150))
"""),
    ("code", """
import json as _json

from src.visualize import dataset_label

HEADS = {"fm_s1": "Strategy 1", "fm_s2": "Strategy 2"}
cdir = REPO / "results" / "artifacts" / "curves_stage3"
r3 = pd.read_csv(REPO / "results" / "metrics" / "runs_stage3.csv")
rows = []
for ds, enc in cfg["stage3"]["settings"]:
    for head in ("fm_s1", "fm_s2"):
        for seed in cfg["stage3"]["subset_seeds"]:
            h = _json.load(open(cdir / f"{ds}_{enc}_{head}_seed{seed}.json"))
            # checkpoint epoch from the table of record (runs_stage3.csv), which
            # applies the full tie-break rule (val acc -> val CE -> earliest)
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
      "§3.2 / §6):")
display(pd.DataFrame(rows))
failed = sorted(cdir.glob("*_failed_lv*.json"))
print(f"Failed training attempts (partial curves saved): {len(failed)}"
      + (" — " + ", ".join(p.name for p in failed) if failed else
         " — no NaN/Inf occurred; every run completed at fallback level 0 "
         "(the default recipe)."))
"""),
]
