CELLS = [
    ("markdown", """
## 4 · Training behaviour

**Comparable end-to-end metrics on shared axes** (the spec's required training + validation curves, both methods): the full-pipeline cross-entropy and top-1 accuracy of $z \\to \\mathrm{FM} \\to$ frozen probe, on train and validation, every epoch. Representative run: subset seed 0 (the validation-selected winners). Dashed = train, solid = validation; blue = Rolled strategy, orange = Guided strategy.
"""),
    ("code", """
for ds, enc in cfg["stage3"]["settings"]:
    p = REPO / "results" / "figures" / f"stage3_curves_{ds}_{enc}.png"
    if p.exists():
        display(Image(str(p), width=980))
"""),
    ("markdown", """
**Checkpoint placement and displacement.** Where validation selected each final model (epoch, of 200; label = epoch and best validation top-1) and how far that checkpointed model moves its training features, as the **relative displacement** $\\mathrm{mean}_i\\,\\lVert\\hat{z}_i-z_i\\rVert/\\lVert z_i\\rVert$ in % of the feature norm (label: % and, in parentheses, the absolute mean $\\lVert\\hat{z}-z\\rVert$ in feature units) — both mandatory strategies, all three subset seeds. Relative units make the two encoders comparable (mean feature norms ≈24 on ResNet-18 vs ≈50 on DINOv2) and are the quantity both methods are defined in: the Rolled strategy penalizes the squared relative displacement, and the Guided strategy's trust region is 10% of $\\lVert z\\rVert$ (dashed line). Checkpoint epochs come from `runs_stage3.csv` (the table of record, full tie-break rule); displacements are recomputed from the checkpointed models on their training subsets (the during-training history values agree within 0.1 units). The λ = 0 and optional-extension models are covered in §3.2 / §6.
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
    ("markdown", """
**Reading the right-hand panels.** The Guided strategy sits at its ≈10% trust-region cap on both datasets (per-sample maxima 11–14%: the FM regresses toward targets on the boundary but is itself unconstrained, so it can overshoot slightly), whereas the Rolled strategy moves ≈11–15% of the norm on DTD — where it does not help — and only ≈2.4% on FGVC-Aircraft, where it helps slightly. In absolute units the two encoders would not be comparable (mean feature norms ≈24 vs ≈50).
"""),
]
