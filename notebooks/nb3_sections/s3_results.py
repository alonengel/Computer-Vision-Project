CELLS = [
    ("markdown", """
## 3 · Classification results

Every number is generated programmatically from `results/metrics/` and re-derived by the repro check; raw per-run accuracies in `results/metrics/raw/stage3_*.npy`, the per-run table in `runs_stage3.csv`. Δ is **paired per seed** against the exact pinned probe of that seed.

### 3.1 Main comparison (the spec's required table)
"""),
    ("code", """
display(Markdown((REPO / "results" / "metrics" / "stage3_main_table.md")
                 .read_text(encoding="utf-8")))
"""),
    ("markdown", """
### 3.2 Rolled strategy — with vs without the displacement regularizer (pre-registered pair)
"""),
    ("code", """
display(Markdown((REPO / "results" / "metrics" / "stage3_lambda_ablation_table.md")
                 .read_text(encoding="utf-8")))
"""),
    ("markdown", """
### 3.3 Selection transparency — the full seed-0 validation sweep

Everything the winners were chosen from, so the selection involves no undisclosed freedom. Test was untouched during selection. One panel per (dataset, strategy) — the two strategies are never mixed on one axis: markers are the seed-0 **validation** top-1 of every swept configuration, ★ the validation-selected winner (highest validation accuracy; ties → lowest validation CE → grid order), "ep" the checkpoint epoch of that run, and the dashed line the pinned probe's seed-0 validation accuracy — the pipeline at its identity initialization. Every run completed at fallback level 0 (the default recipe). The generated table of record behind the figure is `results/metrics/stage3_sweep_table.md`.
"""),
    ("code", """
display(Image(str(REPO / "results" / "figures" / "stage3_sweep.png"), width=1150))
"""),
    ("code", """
from src.visualize import dataset_label, encoder_label

HEAD_SHOW = {"fm_s1": "Rolled strategy", "fm_s1_lambda0": "Rolled strategy, λ=0",
             "fm_s2": "Guided strategy"}
r3 = pd.read_csv(REPO / "results" / "metrics" / "runs_stage3.csv")
ok = r3[r3["status"] == "ok"]
print("Per-seed paired deltas (percentage points), the quantity every claim rests on:")
t = (ok[ok["head"] != "pinned_probe"]
     .assign(setting=lambda d: d["dataset"].map(dataset_label) + " / "
             + d["encoder"].map(lambda e: encoder_label(e, short=True)),
             method=lambda d: d["head"].map(HEAD_SHOW),
             delta_pts=lambda d: (100 * d["delta_acc"]).round(2))
     .pivot_table(index=["setting", "method"], columns="seed", values="delta_pts"))
display(t)
"""),
]
