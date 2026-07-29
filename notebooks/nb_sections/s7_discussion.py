CELLS = [
    ("markdown", """
## 6 · Discussion

The observations below are read directly off the tables and figures above; the numbers are loaded from `results/metrics/`, never typed by hand.

**The representation dominates the head.** On FGVC-Aircraft, swapping ResNet-18 for DINOv2 while holding everything else fixed moves the full-split linear probe from 36.62% to 67.21% — a 30.6-point gain, far larger than any difference between heads on a fixed encoder. Put sharply: the DINOv2 probe trained on **5 images per class** (36.47 ± 0.70) matches, within run-to-run noise, the ResNet-18 probe trained on the **entire** training split (36.62 ± 0.27).

**Prototypes win only in the lowest-data regime, and only once.** The image-prototype head beats the linear probe at exactly one cell of the grid — DTD at K = 5, paired +0.90 ± 0.14 with all three seeds agreeing (§5.1) — and at DTD K = 10 the two are statistically indistinguishable at n = 3. Everywhere else the probe wins decisively, by a margin that widens with training-set size. A class mean is a well-conditioned estimator when five examples are all you have, but it cannot exploit extra data the way a trained boundary can.

**Zero-shot CLIP is a different trade-off, not a weaker version of the same one.** It uses no labeled training images at all: 63.6% on Flowers-102, 39.8% on DTD, 17.0% on FGVC-Aircraft. The ordering tracks how describable the classes are in natural language — flowers and textures have names that carry visual meaning, aircraft *variants* essentially do not ("a photo of a 737-300 aircraft" says very little).

**Overfitting is real where it appears, and the protocol absorbs it** — see the training curves and the loss-versus-accuracy divergence discussed in §5.3 (three of the four saved 10-shot curves overfit; Flowers-102 does not).
"""),
    ("code", """
from src.visualize import dataset_label, encoder_label, head_label

s = pd.read_csv(REPO / "results" / "metrics" / "summary.csv")
s["Dataset"] = s["dataset"].map(dataset_label)
s["Encoder"] = s["encoder"].map(lambda e: encoder_label(e, short=True))
s["Baseline"] = s["head"].map(head_label)
s["Accuracy (%)"] = (100 * s["mean_acc"]).round(2).astype(str)
s.loc[s["n_runs"] > 1, "Accuracy (%)"] += " ± " + (100 * s.loc[s["n_runs"] > 1, "std_acc"]).round(2).astype(str)

print("Best configuration per dataset (all baselines, any training-set size):")
display(s.loc[s.groupby("dataset")["mean_acc"].idxmax()][
    ["Dataset", "Encoder", "Baseline", "k_shot", "Accuracy (%)"]].rename(
        columns={"k_shot": "Training-set size"}).reset_index(drop=True))

print("\\nAccuracy (%) by training-set size, and the gain from 5-shot to the full split:")
piv = s.pivot_table(index=["Dataset", "Encoder", "Baseline"], columns="k_shot",
                    values="mean_acc")
piv = (100 * piv).round(2)
piv["gain 5 → full (pts)"] = (piv["full"] - piv["5shot"]).round(2)
display(piv.dropna(subset=["gain 5 → full (pts)"])[
    ["5shot", "10shot", "full", "gain 5 → full (pts)"]])
"""),
    ("markdown", """
### Which prototype branch carries into Stage 2 — decision

**Selected: Option A, image-derived class prototypes.** The selection rests on methodological considerations and **validation data only — test accuracy plays no role in it.** Three of the four reasons are structural properties of the branches that need no accuracy numbers at all; the fourth is measured on the validation split:

1. **Experimental surface** *(methodological)* — image prototypes depend on both the encoder and K, so Stage 2 inherits a grid of transport targets (3 training sizes × the cached encoders) rather than one fixed text embedding per class.
2. **Encoder freedom** *(methodological)* — Option A can be built on any cached representation, including DINOv2 on the fine-grained task; Option B is locked to CLIP RN50 by construction.
3. **Continuity with Stage 3** *(methodological)* — Option A reuses exactly the caches the linear probe uses, so the Stage-2 and Stage-3 comparisons rest on the same representation.
4. **Headroom, measured on validation** — full-split prototype vs full-split probe **validation** accuracy on the same features (from `runs.csv`, `val_acc`; the prototype head is deterministic, so its validation accuracy involves no training or selection): the cell below computes the gaps.

Both branches are implemented and reported, so this selection can be revisited without re-running anything.
"""),
    ("code", """
runs = pd.read_csv(REPO / "results" / "metrics" / "runs.csv")
g = runs[runs["k_shot"] == "full"].copy()
rows = []
for ds, gd in g.groupby("dataset", sort=False):
    probe_val = gd[gd["head"] == "linear_probe"].groupby("encoder")["val_acc"].mean()
    enc = probe_val.idxmax()  # encoder selected on VALIDATION accuracy
    proto = gd[(gd["head"] == "image_prototype") & (gd["encoder"] == enc)]["val_acc"]
    rows.append({"Dataset": dataset_label(ds),
                 "Encoder (selected on val)": encoder_label(enc, short=True),
                 "Probe val acc (%)": round(100 * probe_val[enc], 2),
                 "Prototype val acc (%)": round(100 * float(proto.iloc[0]), 2),
                 "Validation headroom (pts)": round(100 * (probe_val[enc] - float(proto.iloc[0])), 2)})
print("Branch-selection evidence — validation split only (test accuracy unused):")
display(pd.DataFrame(rows))
"""),
    ("markdown", """
### Limitations

- Frozen encoders bound absolute accuracy by design; no adaptation of the representation is attempted at this stage.
- The linear-probe configuration is the specification's suggested baseline, adopted without search. Better per-dataset numbers are certainly reachable, but tuning was explicitly out of scope and would have to be done on the validation split.
- Flowers-102's official training split contains exactly 10 images per class, so its 10-shot and full settings coincide and its three 10-shot runs are identical by construction (hence the exact 0.00 spread). Its test split is also class-imbalanced, so top-1 and macro accuracy differ (§5.1).
- Two-dimensional feature projections are qualitative only; t-SNE in particular preserves local neighbourhoods, not global distances (§5.5).
- One seed governs both weight initialization and minibatch ordering, so the "full" spread measures optimization stochasticity as a whole rather than initialization in isolation.

### Deviations from, and extensions beyond, the specification

| Item | Status | Note |
|---|---|---|
| Third dataset (Flowers-102) | extension ‡ | spec allows any two; our selected pair (DTD + FGVC-Aircraft) is marked and readable on its own |
| Both prototype branches | extension | spec asks for one; implementing both let the Stage-2 branch be chosen on methodological + validation evidence |
| Linear-probe configuration | **as specified** | AdamW / 1e-3 / 1e-4 / 64 / ≤200 epochs / best-val-accuracy checkpoint, unchanged |
| DINOv2 coverage | **as specified** | one dataset (FGVC-Aircraft) |
| CLIP RN50 usage | **as specified** | zero-shot branch only; never a probe or prototype backbone |
| Feature visualizations | **as specified**, both projections | PCA (primary) *and* t-SNE (supplementary); the spec allows either |

## 7 · Stage-2/3 handoff

The concrete configuration each later stage builds on. Encoder selection uses **validation** accuracy of the full-split probe; the prototype target follows the branch decision above; the last column is the already-published one-time test read-out that Stage 2/3 must beat.
"""),
    ("code", """
from IPython.display import Markdown

display(Markdown((REPO / "results" / "metrics" / "handoff_table.md").read_text(encoding="utf-8")))
"""),
    ("markdown", """
### Artifact paths

| Artifact | Path |
|---|---|
| Cached frozen features (train/val/test) | `results/features/{dataset}_{split}_{encoder}.pt` |
| Balanced K-shot training-subset indices | `results/artifacts/subsets/{dataset}_k{K}_seed{s}.pt` |
| CLIP RN50 text prototypes | `results/artifacts/clip_text_{dataset}.pt` |
| Per-run and summary results | `results/metrics/runs.csv`, `summary.csv`, `raw/*.npy` |
| Linear-probe training histories | `results/artifacts/curves/*.json` |

Stage 2 keeps the encoder frozen, trains the Flow Matching model to transport embeddings toward the selected training subset's class prototypes, and is evaluated against the prototype baselines of §5.1 on the identical cached features and committed subset files. Stage 3 places the Flow Matching module between the frozen encoder and the linear probe, trained jointly with cross-entropy — the linear-probe baselines reported here are that comparison's reference.
"""),
]
