CELLS = [
    ("markdown", """
## 4 · Training curves and stability

The spec's stated purpose for these curves is to verify that training is **stable** and that both approaches reach reasonable solutions. Since Stage-2 FM uses the final-epoch model (no checkpoint selection, §1), stability is a load-bearing claim — so it is not read off two representative plots only: the cell below re-evaluates the a-priori criterion from ADR 0007 §7 over **every** saved full-run curve. A run counts as *unstable* only if its final-epoch training loss exceeds $1.05\\times$ its own running minimum (divergence, not plateau); the fixed contingency — never triggered post-hoc — would be minimum-training-loss epoch selection for **all** models uniformly.

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
unstable = []
for p in curves:
    with open(p) as f:
        h = _json.load(f)["train_loss"]
    if h[-1] > 1.05 * min(h):
        unstable.append(p.name)
print(f"Stability criterion (ADR 0007 §7) over all {len(curves)} full-run curves: "
      f"{len(unstable)} unstable")
for n in unstable:
    print("  UNSTABLE:", n)
if not unstable:
    print("=> every model's final-epoch loss is within 5% of its running minimum; "
          "the final-epoch policy stands, no contingency triggered.")
"""),
]
