"""Download all datasets, build the fixed episode/support index files, render sample episode grids."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.data import build_all_episode_files, episodic_path, load_pool
from src.utils import allow_insecure_downloads, load_config
from src.visualize import episode_grid


def main():
    allow_insecure_downloads()
    cfg = load_config()

    pools = {}
    for ds in ("mnist", "cifar10"):
        print(f"[{ds}] loading train/test splits ...", flush=True)
        pools[ds] = {"episodic": load_pool(ds, "test"), "simple_train": load_pool(ds, "train")}
        print(f"[{ds}] test={len(pools[ds]['episodic'])} train={len(pools[ds]['simple_train'])}")

    print("[mini_imagenet] loading (downloads ~several GB on first run) ...", flush=True)
    mi_test = load_pool("mini_imagenet", "test")
    pools["mini_imagenet"] = {"episodic": mi_test, "simple_train": None}
    counts = torch.bincount(torch.from_numpy(mi_test.labels))
    print(f"[mini_imagenet] test-classes pool: {len(mi_test)} images, "
          f"{len(mi_test.class_names)} classes, per-class min/max = "
          f"{counts.min().item()}/{counts.max().item()}")

    created = build_all_episode_files(pools)
    print(f"episode/support files created: {len(created)}")
    for n in created:
        print("  " + n)

    ep_cfg = cfg["episodic"]
    k_fig = max(ep_cfg["shots"])
    for ds, p in pools.items():
        episode = torch.load(episodic_path(ds, ep_cfg["n_way"], k_fig, ep_cfg["seed"]),
                             weights_only=True)
        path = episode_grid(p["episodic"], episode)
        print(f"figure written: {path}")


if __name__ == "__main__":
    main()
