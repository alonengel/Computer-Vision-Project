CELLS = [
    ("markdown", """
## 2 · The two training strategies, and the guards that anchor the comparison

**Strategy 1 — end-to-end rolled-out classification training.** For each training feature $z$, run the complete $T$-step rollout to $\\hat{z}$, pass it through the frozen classifier, and minimize

$$\\mathcal{L}_{\\mathrm{cls}} = \\mathrm{CE}(W\\hat{z} + b,\\, y) \\;+\\; \\lambda\\,\\mathrm{mean}_i\\,\\frac{\\lVert\\hat{z}_i - z_i\\rVert^2}{\\lVert z_i\\rVert^2 + \\varepsilon},$$

backpropagating through the full rollout and updating **only** the FM parameters. The relative displacement penalty (the spec's suggested regularization, made scale-free) discourages unnecessarily large changes; $\\lambda$ is chosen on validation from {0, 1, 10, 100} and the $\\lambda = 0$ variant is always test-reported alongside the winner (pre-registered pair).

**Strategy 2 — classifier-guided targets and standard FM training.** Per epoch, in two phases. *Phase 1 (targets, snapshot):* run every training feature through the current FM to $\\hat{z}$; project onto the trust region, $u_0 = \\Pi_{B(z,\\rho)}(\\hat{z})$ with $\\rho = 0.1\\lVert z\\rVert$; take $m$ normalized CE-gradient steps of size $\\eta = \\beta\\rho$, re-projecting after every step,

$$u_{j+1} = \\Pi_{B(z,\\rho)}\\!\\left(u_j - \\eta\\,\\frac{\\nabla_{u_j}\\mathrm{CE}}{\\lVert\\nabla_{u_j}\\mathrm{CE}\\rVert + \\varepsilon}\\right);$$

the target $\\hat{z}'$ is the **lowest-CE iterate** among the projected $\\{u_0,\\dots,u_m\\}$ (monotone acceptance), detached, cached for all samples. *Phase 2 (FM update):* one epoch of **standard FM regression** against the fixed cache — $t \\sim U(0,1)$, $z_t = (1-t)z + t\\hat{z}'$, loss $\\lVert v_\\theta(z_t, t) - (\\hat{z}' - z)\\rVert^2$. All target quantities are per-sample; **no CE gradient ever reaches the FM** — that separation is the scientific contrast with Strategy 1. Grid: $\\beta \\in \\{0.25, 0.5, 1\\} \\times m \\in \\{1, 3\\}$ on validation. The source-centered trust region (not one centered at $\\hat{z}$) is what prevents cumulative target drift across epochs — without it, repeated recomputation could manufacture arbitrarily classifier-friendly features.

**Two guards, train/validation data only (the test set stayed sealed):**

1. **Exact identity at initialization** — transported features **bit-exact equal** to the inputs, logits equal, predictions identical between the pipeline and the direct probe. Stronger than accuracy equality (two systems can share accuracy while disagreeing on many predictions). The cell below re-executes it.
2. **Pinned-probe reproduction** — the probe is retrained with the identical Stage-1 recipe per (dataset, subset seed), its **validation** accuracy audited against `runs.csv` (|diff| < 0.25 pts; the run log shows 0.000 pts and exact best-epoch matches on all six probes), and the weights pinned so every method receives the same frozen classifier. Δ is computed against **this exact pinned probe** — `runs.csv` serves as a reproduction audit only (its test values were compared post-hoc at the final pass and matched exactly).
"""),
    ("code", """
import torch

from src.stage3 import Stage3FM, identity_guard, load_pinned_probe
from src.embeddings import load_features

mdir = REPO / "results" / "artifacts" / "stage3_models"
k = f"{cfg['stage3']['k_shot']}shot"
print("Exact-identity guard (re-executed live, validation data only):")
for ds, enc in cfg["stage3"]["settings"]:
    probe = load_pinned_probe(mdir / f"probe_{ds}_{enc}_{k}_seed0.pt")
    fval = load_features(ds, "val", enc)
    fm0 = Stage3FM(probe, fval["dim"], seed=cfg["stage3"]["fm_init_seed"])
    identity_guard(fm0, fval["features"][:256])
    print(f"  [OK] {ds} / {enc}: transported features bit-exact, logits and "
          f"predictions identical to the direct probe")
"""),
    ("code", """
# Probe-reproduction audit, displayed from the artifacts of record: the pinned
# probes' validation and test accuracies against the published Stage-1 values.
r1 = pd.read_csv(REPO / "results" / "metrics" / "runs.csv")
r3 = pd.read_csv(REPO / "results" / "metrics" / "runs_stage3.csv")
rows = []
for ds, enc in cfg["stage3"]["settings"]:
    for seed in cfg["stage3"]["subset_seeds"]:
        pin = r3[(r3["dataset"] == ds) & (r3["encoder"] == enc)
                 & (r3["head"] == "pinned_probe") & (r3["seed"] == seed)].iloc[0]
        ref = r1[(r1["dataset"] == ds) & (r1["encoder"] == enc)
                 & (r1["head"] == "linear_probe") & (r1["k_shot"] == k)
                 & (r1["seed"] == seed)].iloc[0]
        rows.append({"dataset": ds, "seed": seed,
                     "pinned val (%)": round(100 * pin["val_acc"], 4),
                     "Stage-1 val (%)": round(100 * ref["val_acc"], 4),
                     "pinned test (%)": round(100 * pin["test_acc"], 4),
                     "Stage-1 test (%)": round(100 * ref["test_acc"], 4),
                     "identical": bool(np.isclose(pin["val_acc"], ref["val_acc"],
                                                  atol=1e-12)
                                       and np.isclose(pin["test_acc"],
                                                      ref["test_acc"], atol=1e-12))})
audit = pd.DataFrame(rows)
assert audit["identical"].all()
print("Pinned-probe reproduction audit — bit-identical to the published "
      "Stage-1 numbers on all six probes:")
display(audit)
"""),
]
