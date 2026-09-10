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
**Checkpoint placement and displacement.** Where validation selected each final model (epoch, of 200; label = epoch and best validation top-1) and how far that checkpointed model moves the training features (mean $\\lVert\\hat{z}-z\\rVert$, feature units) — both mandatory strategies, all three subset seeds. Checkpoint epochs come from `runs_stage3.csv` (the table of record, full tie-break rule), displacements from the training histories at that epoch. The λ = 0 and optional-extension models are covered in §3.2 / §6.
"""),
    ("code", """
display(Image(str(REPO / "results" / "figures" / "stage3_checkpoints.png"), width=1150))
cdir = REPO / "results" / "artifacts" / "curves_stage3"
failed = sorted(cdir.glob("*_failed_lv*.json"))
print(f"Failed training attempts (partial curves saved): {len(failed)}"
      + (" — " + ", ".join(p.name for p in failed) if failed else
         " — no NaN/Inf occurred; every run completed at fallback level 0 "
         "(the default recipe)."))
"""),
]
