"""Download all datasets, verify official splits, and build the balanced K-shot subsets."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.data import SPLITS, build_all_subsets, load_pool, subset_path
from src.utils import allow_insecure_downloads, load_config


def main():
    allow_insecure_downloads()
    cfg = load_config()

    train_pools, split_rows, pools = {}, [], {}
    for ds in cfg["datasets"]:
        print(f"\n=== {ds} ===", flush=True)
        for split in SPLITS:
            pool = load_pool(ds, split, download=True)
            counts = np.bincount(pool.labels, minlength=len(pool.class_names))
            print(f"  {split:5s}: {len(pool):6d} images | {len(pool.class_names)} classes "
                  f"| per-class min/median/max = {counts.min()}/{int(np.median(counts))}/{counts.max()}",
                  flush=True)
            split_rows.append({"dataset": ds, "split": split, "images": len(pool),
                               "classes": len(pool.class_names),
                               "per_class_min": int(counts.min()),
                               "per_class_median": int(np.median(counts)),
                               "per_class_max": int(counts.max()),
                               "spec_selected": cfg["datasets"][ds]["spec_selected"]})
            pools[(ds, split)] = pool
            if split == "train":
                train_pools[ds] = pool
                print(f"         example classes: {pool.class_names[:4]} ...")

    import pandas as pd

    from src.evaluation import metrics_dir
    from src.visualize import dataset_label

    df = pd.DataFrame(split_rows)
    df.insert(0, "Dataset", [dataset_label(d) + ("" if s else " ‡")
                             for d, s in zip(df["dataset"], df["spec_selected"])])
    df.to_csv(metrics_dir() / "dataset_splits.csv", index=False)
    print(f"\nwritten: {metrics_dir() / 'dataset_splits.csv'}")

    created, upgraded = build_all_subsets(train_pools)
    print(f"\nK-shot subset files created: {len(created)}; "
          f"fingerprints upgraded in place (indices unchanged): {len(upgraded)}")

    from src.embeddings import backfill_fingerprints

    filled = backfill_fingerprints(pools)
    if filled:
        print(f"feature caches stamped with the split fingerprint: {len(filled)}")
    for ds in cfg["datasets"]:
        for k in [s for s in cfg["shots"] if s != "full"]:
            for seed in cfg["subset_seeds"]:
                import torch

                d = torch.load(subset_path(ds, k, seed), weights_only=True)
                short = f" | short classes: {len(d['short_classes'])}" if d["short_classes"] else ""
                print(f"  {ds} K={k} seed={seed}: {len(d['indices'])} images{short}")
    print("\ndone.")


if __name__ == "__main__":
    main()
