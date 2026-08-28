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
### 3.2 Strategy 1 — with vs without the displacement regularizer (pre-registered pair)
"""),
    ("code", """
display(Markdown((REPO / "results" / "metrics" / "stage3_lambda_ablation_table.md")
                 .read_text(encoding="utf-8")))
"""),
    ("markdown", """
### 3.3 Selection transparency — the full seed-0 validation sweep

Everything the winners were chosen from, so the selection involves no undisclosed freedom. Test was untouched during selection.
"""),
    ("code", """
display(Markdown((REPO / "results" / "metrics" / "stage3_sweep_table.md")
                 .read_text(encoding="utf-8")))
"""),
    ("code", """
from src.visualize import dataset_label, encoder_label

HEAD_SHOW = {"fm_s1": "Strategy 1", "fm_s1_lambda0": "Strategy 1, λ=0",
             "fm_s2": "Strategy 2"}
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
