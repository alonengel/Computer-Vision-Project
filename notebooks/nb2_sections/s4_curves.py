CELLS = [
    ("markdown", """
## 4 · Training curves and stability — and the one contingency that triggered

The spec's stated purpose for these curves is to verify that training is **stable** and that both approaches reach reasonable solutions. Stability is a load-bearing claim here (no validation checkpointing exists to absorb a bad ending), so it is not read off two representative plots: the cell below evaluates the a-priori criterion from ADR 0007 §7 over **every** saved training curve — a run counts as *unstable* if its final-epoch training loss exceeds $1.05\\times$ its own running minimum (divergence, not plateau).

**The criterion fired.** On the first (final-epoch) grid, 3 of 135 runs genuinely diverged — all rolled-out models at K = full, the worst at $62\\times$ its minimum with test accuracy collapsed to ~6% (recorded at trigger time; the first grid's artifacts were superseded by the re-run) — alongside 20 marginal 1.05–1.13× cases (18 standard-FM runs, where random-$t$ resampling makes the last epoch noisy, and 2 rolled-out runs just past the threshold; this 3-vs-20 split is our post-hoc description — the pre-registered criterion is binary at 1.05× and the uniform fallback applies regardless of it).

**Could min-training-loss pick a "lucky" epoch?** A fair objection: standard FM's per-epoch loss fluctuates with the random-$t$ draws, so the minimum could land on an epoch whose draws happened to be easy. Three answers, all checkable. First, the selection statistic uses training data only, and the test set is evaluated once on the selected checkpoint — no quantity involved in the selection is correlated with test noise, so the selection cannot be optimistic *toward test accuracy* (it is a winner's curse on the training loss, adding variance, not test bias). Second, the cell below prints where the selected epochs actually land: essentially at the end of training (medians ≈ 193–197 of 200) for every stable run — the rule effectively returns the late-training model everywhere except the divergences it exists to catch. Third, the superseded final-epoch grid's stable-run accuracies differed from the published ones only by tenths of a point in both directions (the first grid's DTD arrays are preserved at commit `cbceb53` and were compared during the adversarial review). As pre-registered, the uniform fallback was applied: **every model, both modes and both branches, was re-trained/selected at its minimum-training-loss epoch**, and the whole grid re-run. The first grid's numbers were never published. All numbers in this notebook come from the fallback grid; the cell below documents the divergence pattern that triggered it (final vs minimum loss on the *saved* curves — the checkpoint used is always the minimum).

Note the two objectives live on different scales (standard FM regresses velocities along the whole path; rolled-out FM penalizes only the final-state distance), hence the log axis; curve *shapes*, not absolute levels, are the comparable quantity.
"""),
    ("code", """
for name in ("stage2_curves_0.png", "stage2_curves_1.png"):
    p = REPO / "results" / "figures" / name
    if p.exists():
        display(Image(str(p), width=980))
"""),
    ("code", """
import json as _json

curve_dir = REPO / "results" / "artifacts" / "curves_stage2"
curves = [p for p in sorted(curve_dir.glob("*.json")) if "_smoke" not in p.name]
rows = []
for p in curves:
    with open(p) as f:
        h = _json.load(f)["train_loss"]
    ratio = h[-1] / min(h)
    if ratio > 1.05:
        rows.append({"curve": p.name, "final / min loss": round(ratio, 2)})
rows.sort(key=lambda r: -r["final / min loss"])
print(f"Stability criterion (ADR 0007 §7) over all {len(curves)} curves: "
      f"{len(rows)} exceed 1.05x (checkpoint used = min-loss epoch, so divergent "
      f"endings do not enter any reported number)")
display(pd.DataFrame(rows[:10]))

# Where do the selected (min-training-loss) checkpoints actually land?
r2 = pd.read_csv(REPO / "results" / "metrics" / "runs_stage2.csv")
ck = (r2.drop_duplicates(subset=["dataset", "encoder", "target", "head",
                                 "k_shot", "seed", "T"])
        .groupby("head")["checkpoint_epoch"].agg(["min", "median", "max"]))
print("\\nSelected checkpoint epoch (of 200) by training mode:")
display(ck)
print("=> the min-loss rule effectively selects the late-training model "
      "everywhere except the diverged runs it exists to catch.")
"""),
]
