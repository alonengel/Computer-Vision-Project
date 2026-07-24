CELLS = [
    ("markdown", """
## 6 · Discussion

The observations below are read directly off the tables and figures above; the numbers are loaded from `results/metrics/`, never typed by hand.

**The representation dominates the head.** On FGVC-Aircraft, swapping ResNet-18 for DINOv2 while holding everything else fixed moves the full-split linear probe from 36.62% to 67.21% — a 30.6-point gain, far larger than any difference between heads on a fixed encoder. Put sharply: the DINOv2 probe trained on **5 images per class** (36.47%) already equals the ResNet-18 probe trained on the **entire** training split (36.62%).

**Prototypes win only in the lowest-data regime, and only once.** The image-prototype head beats the linear probe at exactly one cell of the grid — DTD at K = 5, paired +0.90 ± 0.14 — and loses everywhere else by a margin that widens with training-set size. A class mean is a well-conditioned estimator when five examples are all you have, but it cannot exploit extra data the way a trained boundary can, and it is blind to the fact that some feature directions separate classes better than others.

**Zero-shot CLIP is a different trade-off, not a weaker version of the same one.** It uses no labeled training images at all: 63.6% on Flowers-102, 39.8% on DTD, 17.0% on FGVC-Aircraft. The ordering tracks how describable the classes are in natural language — flowers and textures have names that carry visual meaning, aircraft *variants* essentially do not ("a photo of a 737-300 aircraft" says very little).

**Overfitting is real and the protocol absorbs it** — see the training curves and the loss-versus-accuracy divergence discussed in §5.3.
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

Stage 2 replaces the decision layer with a Flow Matching model that transports an embedding toward a class representation, so this choice determines what the FM model is trained to reach. **We select Option A, image-derived class prototypes**, for four reasons drawn from the results above:

1. **Headroom.** On the same frozen features the prototype head sits far below the linear probe (34.41 vs 67.21 on FGVC-Aircraft/DINOv2; 58.78 vs 62.84 on DTD). That gap is precisely the space a Flow Matching decision layer has to demonstrate a gain in.
2. **Experimental surface.** Image prototypes depend on both the encoder and K, so Stage 2 inherits a grid of targets (3 training sizes × the encoders already cached) rather than one fixed text embedding per class.
3. **Encoder freedom.** Option A can be built on the strongest available representation (DINOv2 on FGVC-Aircraft), whereas Option B is locked to CLIP RN50 — the weakest of the three encoders on the fine-grained task.
4. **Continuity with Stage 3.** Option A reuses exactly the caches the linear probe uses, so the Stage-2 and Stage-3 comparisons rest on the same representation.

Both branches are implemented and reported, so this decision can be revisited without re-running anything.

### Limitations

- Frozen encoders bound absolute accuracy by design; no adaptation of the representation is attempted at this stage.
- The linear-probe configuration is the specification's suggested baseline, adopted without search. Better per-dataset numbers are certainly reachable, but tuning was explicitly out of scope and would have to be done on the validation split.
- Flowers-102's official training split contains exactly 10 images per class, so its 10-shot and full settings coincide and its three 10-shot runs are identical by construction (hence the exact 0.00 spread). Its test split is also class-imbalanced, so top-1 and macro accuracy differ (§5.1).
- Two-dimensional feature projections are qualitative only: both PCA and t-SNE distort the geometry of the frozen feature spaces.
- One seed governs both weight initialization and minibatch ordering, so the "full" spread measures optimization stochasticity as a whole rather than initialization in isolation.

### Deviations from, and extensions beyond, the specification

| Item | Status | Note |
|---|---|---|
| Third dataset (Flowers-102) | extension ‡ | spec asks for two; the required pair (DTD + FGVC-Aircraft) is marked and readable on its own |
| Both prototype branches | extension | spec asks for one; implementing both let the Stage-2 branch be chosen on evidence |
| Linear-probe configuration | **as specified** | AdamW / 1e-3 / 1e-4 / 64 / ≤200 epochs / best-val-accuracy checkpoint, unchanged |
| DINOv2 coverage | **as specified** | one dataset (FGVC-Aircraft) |
| CLIP RN50 usage | **as specified** | zero-shot branch only; never a probe or prototype backbone |
| Feature visualizations | **as specified**, both projections | PCA *and* t-SNE shown; the spec allows either |

### Artifacts handed to Stage 2/3

| Artifact | Path |
|---|---|
| Cached frozen features (train/val/test) | `results/features/{dataset}_{split}_{encoder}.pt` |
| Balanced K-shot training-subset indices | `results/artifacts/subsets/{dataset}_k{K}_seed{s}.pt` |
| CLIP RN50 text prototypes | `results/artifacts/clip_text_{dataset}.pt` |
| Per-run and summary results | `results/metrics/runs.csv`, `summary.csv`, `raw/*.npy` |
| Linear-probe training histories | `results/artifacts/curves/*.json` |

Stage 2 keeps the encoder frozen and reuses these caches and subset files, so its comparison against these baselines is made on identical data. Stage 3 places the Flow Matching module between the frozen encoder and the linear probe, trained jointly with cross-entropy — the linear-probe configuration reported here is that comparison's baseline.
"""),
]
