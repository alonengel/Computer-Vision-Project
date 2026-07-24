CELLS = [
    ("markdown", """
## 6 · Discussion and handoff to Stage 2

The observations below are read directly off the tables and figures above; the numbers are loaded from `results/metrics/`, never typed by hand.
"""),
    ("code", """
s = pd.read_csv(REPO / "results" / "metrics" / "summary.csv")
s["accuracy"] = (100 * s["mean_acc"]).round(2).astype(str)
s.loc[s["n_runs"] > 1, "accuracy"] += " ± " + (100 * s.loc[s["n_runs"] > 1, "std_acc"]).round(2).astype(str)

print("Best configuration per dataset (all baselines, any training-set size):")
best = s.loc[s.groupby("dataset")["mean_acc"].idxmax()]
display(best[["dataset", "encoder", "head", "k_shot", "accuracy"]])

print("\\nGain from 5-shot to the full training split, per dataset/encoder/head:")
piv = s.pivot_table(index=["dataset", "encoder", "head"], columns="k_shot",
                    values="mean_acc")
if {"5shot", "full"}.issubset(piv.columns):
    piv["gain_5shot_to_full"] = (100 * (piv["full"] - piv["5shot"])).round(2)
    display(piv.dropna(subset=["gain_5shot_to_full"])[["5shot", "10shot", "full",
                                                       "gain_5shot_to_full"]].round(4))
"""),
    ("markdown", """
### Choosing the branch that continues into Stage 2

Stage 2 replaces the decision layer with a Flow Matching model that transports an embedding toward a class representation, so the branch chosen here determines what the FM model is trained to reach: **image-derived class prototypes** (which change with the training-set size and with the encoder) or **fixed CLIP text prototypes** (identical for every K, and available with no labeled images at all).

The evidence to weigh, from the results above: which branch is stronger on the spec-selected datasets, how each behaves as the training set grows, and how much headroom each leaves for Stage 2 to demonstrate a gain. A branch that already sits near the ceiling gives Flow Matching nothing to improve.

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
