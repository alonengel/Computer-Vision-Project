"""Select the strongest baseline configuration per classifier head (Stage-2 references).

Stage 1's purpose is not to rank heads against each other but to find, for each
classifier function (prototype / linear probe / zero-shot CLIP), its best encoder
configuration per dataset — the baseline Stages 2-3 must beat. Selection is by
episodic accuracy from results/metrics/episodic.csv, per K, excluding the
ImageNet-label-contaminated ResNet-50 × Mini-ImageNet combination (ADR/report §4).

Writes results/artifacts/best_baselines.json (loaded by Stage 2) and prints the table.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.evaluation import metrics_dir
from src.utils import load_config, repo_path

HEAD_FAMILIES = {
    "prototype": ["proto_cos__", "proto_eucl__"],
    "linear": ["linear__"],
    "zeroshot_clip": ["clip_zeroshot__", "clip_zeroshot_ens__"],
}
CONTAMINATED = [("mini_imagenet", "resnet50")]


def main():
    ep = pd.read_csv(metrics_dir() / "episodic.csv")
    for ds, bb in CONTAMINATED:
        ep = ep[~((ep["dataset"] == ds) & (ep["backbone"] == bb))]

    cfg = load_config()
    selection = {}
    rows = []
    for ds in cfg["datasets"]:
        selection[ds] = {}
        for head, prefixes in HEAD_FAMILIES.items():
            selection[ds][head] = {}
            for k in cfg["episodic"]["shots"]:
                g = ep[(ep["dataset"] == ds) & (ep["k_shot"] == k)
                       & ep["classifier"].str.startswith(tuple(prefixes))]
                best = g.loc[g["acc"].idxmax()]
                runners = g[g["acc"] >= best["acc"] - best["ci95"]]
                selection[ds][head][f"{k}shot"] = {
                    "classifier": best["classifier"],
                    "backbone": best["backbone"],
                    "acc": round(float(best["acc"]), 4),
                    "ci95": round(float(best["ci95"]), 4),
                    "within_ci_alternatives": sorted(
                        c for c in runners["classifier"] if c != best["classifier"]),
                }
                rows.append({"dataset": ds, "head": head, "k_shot": k,
                             "best": best["classifier"],
                             "acc": f"{100 * best['acc']:.2f} ± {100 * best['ci95']:.2f}"})

    out = repo_path(cfg["paths"]["artifacts_dir"]) / "best_baselines.json"
    with open(out, "w") as f:
        json.dump({"note": "Strongest Stage-1 baseline per classifier head - the "
                           "reference Stages 2-3 must beat on the same episode files. "
                           "ResNet-50 x Mini-ImageNet excluded (label contamination).",
                   "selection_metric": "episodic accuracy, 600 fixed episodes",
                   "selection": selection}, f, indent=2)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
