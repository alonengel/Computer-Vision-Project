CELLS = [
    ("markdown", """
## 2 · Method — exactly as specified

**Standard FM training** (`_docs/stage_2.pdf`). For each training feature $z_i$ with class prototype $p_{y_i}$, sample $t \\sim \\mathcal{U}(0,1)$, form the interpolation point $z_t = (1-t)\\,z_i + t\\,p_{y_i}$, and supervise the velocity network on the ideal constant velocity $u_i = p_{y_i} - z_i$:

$$\\mathcal{L}_{\\mathrm{FM}} = \\lVert v_\\theta(z_t, t) - u_i \\rVert_2^2 .$$

**Inference** (both modes). Start from the test feature $\\hat{z}_0 = z$ and apply $T$ Euler steps,

$$\\hat{z}_{k+1} = \\hat{z}_k + \\tfrac{1}{T}\\, v_\\theta\\!\\left(\\hat{z}_k, \\tfrac{k}{T}\\right), \\qquad k = 0, \\dots, T-1, \\qquad T \\in \\{4, 12\\},$$

then classify $\\hat{z}_T$ with the Stage-1 rule $\\hat{y} = \\arg\\max_c \\cos(\\hat{z}_T, p_c)$.

**Rolled-out training.** Same network, same $T$-step Euler sequence starting from each *training* feature $\\hat{z}_0 = z_i$ — the network is exposed at training time to exactly the self-generated intermediate states it will see at inference — with the loss on the final state only:

$$\\mathcal{L}_{\\mathrm{roll}} = \\lVert \\hat{z}_T - p_{y_i} \\rVert_2^2 ,$$

backpropagated through the complete sequence of $T$ velocity evaluations. Rolled-out models use the **same $T$ at training and inference** (one model per $T$); standard FM training involves no $T$, so one standard model per setting is trained and evaluated at both $T = 4$ and $T = 12$ — those two rows share one set of weights.

Losses are per-sample squared $L_2$ norms (summed over feature dimensions, averaged over the batch), matching the spec's $\\lVert\\cdot\\rVert_2^2$ exactly.

**One disclosed deviation from the literal spec, fixed a priori (ADR 0007 §3): FM operates on $L_2$-normalized features.** $\\hat{z}_0 = z / \\lVert z \\rVert$; prototypes are unit-norm by construction. The Stage-1 classifier rule is cosine similarity — the normalized sphere is the space in which it acts — while raw frozen features have norms $\\approx$ 10–40, so interpolating raw features toward unit-norm prototypes would make the path traverse mostly *scale* rather than class structure. No renormalization is applied between Euler steps (the spec's update is plain Euler); cosine classification normalizes once at the end. With $T=0$ (no transport) the pipeline reduces exactly to Stage 1, which is asserted numerically below.

**Prototypes are identical to Stage 1.** For each $(K, \\text{seed})$ they are recomputed with the same formula $\\mu_c = \\mathrm{normalize}(\\tfrac{1}{|S_c|}\\sum_{i \\in S_c}\\mathrm{normalize}(z_i))$ from the same committed subset index files (fingerprint-checked); the CLIP‡ branch uses the cached text prototypes. **Integrity guard:** before any training, the run script asserts for *every* setting that classifying the untransported test features against these prototypes reproduces the Stage-1 baseline accuracy from `runs.csv` to within $10^{-6}$ — all 45 settings (27 image-branch, 18 CLIP‡) passed; the guard below re-executes that check for one setting per branch as a visible demonstration.
"""),
    ("code", """
import torch
import torch.nn.functional as F

from src.classifiers import PrototypeClassifier
from src.data import training_indices
from src.embeddings import clip_text_path, load_features

runs1 = pd.read_csv(REPO / "results" / "metrics" / "runs.csv")

def t0_check(ds, enc, target, k_shot, seed):
    f = {s: load_features(ds, s, enc) for s in ("train", "test")}
    if target == "image_prototype":
        idx = training_indices(ds, k_shot, seed, f["train"]["labels"].numpy(),
                               fingerprint=f["train"].get("pool_fingerprint"))
        protos = PrototypeClassifier(cfg["datasets"][ds]["n_classes"]).fit(
            f["train"]["features"][idx], f["train"]["labels"][idx].long()).prototypes
        b = runs1[(runs1.dataset == ds) & (runs1.encoder == enc)
                  & (runs1["head"] == "image_prototype")
                  & (runs1.k_shot == f"{k_shot}shot") & (runs1.seed == seed)]
    else:
        protos = torch.load(clip_text_path(ds), weights_only=True)["text_prototypes"].float()
        b = runs1[(runs1.dataset == ds) & (runs1["head"] == "zeroshot_clip")]
    z = F.normalize(f["test"]["features"].float(), dim=-1)
    acc0 = float(((z @ protos.T).argmax(-1) == f["test"]["labels"]).float().mean())
    expected = float(b["test_acc"].iloc[0])
    status = "OK" if abs(acc0 - expected) < 1e-6 else "MISMATCH"
    print(f"  [{status}] {ds} / {enc} / {target}: T=0 accuracy {100*acc0:.4f}% "
          f"vs Stage-1 baseline {100*expected:.4f}%")
    return status == "OK"

print("T = 0 integrity guard (no transport must reproduce Stage 1 exactly):")
assert t0_check("dtd", "resnet18", "image_prototype", 10, 0)
assert t0_check("fgvc_aircraft", "clip_rn50", "clip_text", 10, 0)
"""),
    ("markdown", """
#### The two versions ‡: the literal-spec raw-feature grid, run in full

The rationale above — raw-space interpolation toward unit-norm prototypes would spend the model's capacity on scale rather than class structure — is a claim about the learned problem, so we measure it rather than assert it: the **complete grid was run twice**. The published version operates on $L_2$-normalized inputs (ADR 0007 §3); the second version is the **literal-spec formulation** ($\\hat{z}_0 = z$, raw frozen features) under the otherwise-identical protocol — same velocity network, recipe, prototypes, committed subsets, all seeds, both T values, both branches, the same min-training-loss checkpoint rule, the same test split. The Stage-1 baselines are identical for both versions (cosine classification is scale-invariant), so the comparison isolates the input-normalization decision completely. Full raw-version artifacts: `runs_stage2_raw.csv`, `summary_stage2_raw.csv`, per-run curves and predictions with a `_raw` suffix — all covered by the repro check.
"""),
    ("code", """
for name in ("stage2_raw_image_prototype_table.md", "stage2_raw_clip_text_table.md"):
    display(Markdown((REPO / "results" / "metrics" / name).read_text(encoding="utf-8")))

s_raw = pd.read_csv(REPO / "results" / "metrics" / "summary_stage2_raw.csv")
s_norm = pd.read_csv(REPO / "results" / "metrics" / "summary_stage2.csv")
keys = ["dataset", "encoder", "target", "head", "T", "k_shot"]
m = s_raw.merge(s_norm, on=keys, suffixes=("_raw", "_norm"))
d = 100 * (m["mean_acc_norm"] - m["mean_acc_raw"])
n_norm = int((d >= 0).sum())
print(f"Normalized ≥ raw in {n_norm} / {len(m)} grid cells; "
      f"normalized − raw ranges {d.min():+.2f} to {d.max():+.2f} points "
      f"(median {d.median():+.2f}).")
below = int((m["mean_acc_raw"] < m["baseline_mean_raw"]).sum())
print(f"Raw-version cells below their own Stage-1 baseline: {below} / {len(m)}.")

import json as _json
raw_curves = sorted((REPO / "results" / "artifacts" / "curves_stage2").glob("*_raw.json"))
bad = sum(1 for p in raw_curves
          if (lambda h: h[-1] > 1.05 * min(h))(_json.load(open(p))["train_loss"]))
print(f"Raw-version stability (same ADR 0007 §7 criterion, same min-loss "
      f"checkpoint rule): {bad} of {len(raw_curves)} curves exceed 1.05x.")
"""),
    ("markdown", """
The two-version comparison replaces any need to argue the normalization choice from first principles. On the **spec branch the cost of the literal formulation is decisive** — the normalized version is better in every one of the 36 image-prototype cells, mostly by double digits, and raw training is also less stable (more curves trip the §4 criterion, and the raw rolled-out DINOv2 full-split runs carry multi-point spreads). On the **CLIP‡ branch the two versions are nearly equivalent** (differences small and mixed in sign, raw slightly ahead in most rolled-out cells) — consistent with CLIP image features having far more uniform norms, so raw and normalized inputs differ much less there. The normalized version remains the primary published result; the raw version is reported in full for transparency.
"""),
]
