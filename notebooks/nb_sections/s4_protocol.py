CELLS = [
    ("markdown", """
## 4 · Protocol integrity and provenance

Everything a reader needs to judge the numbers, in one place.

**Split discipline.** All classes, official train / validation / test splits, train and validation never merged. The validation split is used *only* for linear-probe checkpoint selection (highest validation accuracy). The complete official **test** split is used only for the final top-1 numbers reported below — no hyperparameter, encoder, subset or checkpoint was ever chosen by looking at test accuracy.

**Run and seed structure** (3 runs per training-set size, exactly as specified):

| Setting | What varies across the 3 runs | What is held fixed |
|---|---|---|
| 5-shot, 10-shot | balanced-subset seed ∈ {0, 1, 2} | classifier initialization (seed 0) |
| full — linear probe | classifier-initialization seed ∈ {0, 1, 2} | training set (the complete official split) |
| full — image prototypes | *nothing* — single deterministic run | — |
| zero-shot CLIP | *nothing* — single deterministic run, no training images | — |

Each setting therefore has a single interpretable source of variance: subset sampling at 5/10-shot, initialization at full. Single-run settings are reported without a standard deviation rather than as "± 0.00".

**Hyperparameter provenance.** The linear-probe configuration is the specification's suggested baseline, adopted unchanged and fixed before any result was observed: AdamW, lr 1e-3, weight decay 1e-4, batch size 64, ≤200 epochs, checkpoint on best validation accuracy. Feature extraction uses each checkpoint's own preprocessing. All values live in `config/config.json`; none was tuned.

**Reproducibility.** Balanced training subsets are saved as index files (`results/artifacts/subsets/`) carrying a fingerprint of the training-split labels, so every encoder and head trains on identical images and a dataset change fails loudly instead of silently shifting the data. Per-run accuracies are saved to `results/metrics/raw/`; `scripts/repro_check.py` re-derives every mean and standard deviation in the tables below from those raw arrays.
"""),
    ("code", """
runs = pd.read_csv(REPO / "results" / "metrics" / "runs.csv")
print("individual runs recorded:", len(runs))
display(runs.groupby(["head", "k_shot"], sort=False)
            .agg(runs=("test_acc", "size"), seed_type=("seed_type", "first"))
            .reset_index())
"""),
]
