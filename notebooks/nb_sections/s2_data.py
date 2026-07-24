CELLS = [
    ("markdown", """
## 2 · Datasets, official splits and training subsets

All classes are used, with the **official** splits: DTD uses partition 1, FGVC-Aircraft uses the `variant` annotation level, Flowers-102 uses its published split. Training and validation splits are never merged.
"""),
    ("code", """
splits = pd.read_csv(REPO / "results" / "metrics" / "dataset_splits.csv")
display(splits)
"""),
    ("markdown", """
For the 5-shot and 10-shot settings a **balanced** subset of the official training split is sampled with seeds {0, 1, 2}; the chosen indices are saved under `results/artifacts/subsets/` so that every encoder and every classification head is trained on bit-identical images. The `full` setting is the complete official training split.

Note the structural quirk visible in the table: Flowers-102's official training split contains exactly 10 images per class, so for that dataset the 10-shot setting *is* the full setting (its three 10-shot "runs" therefore see identical data and the spread is exactly zero). DTD (40 per class) and FGVC-Aircraft (~33 per class) give three genuinely distinct settings — which is why they are the spec-selected pair.
"""),
    ("code", """
display(Image(str(REPO / "results" / "figures" / "samples_dtd.png"), width=760))
display(Image(str(REPO / "results" / "figures" / "samples_fgvc_aircraft.png"), width=760))
display(Image(str(REPO / "results" / "figures" / "samples_flowers102.png"), width=760))
"""),
]
