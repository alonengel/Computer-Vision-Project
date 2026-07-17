"""Cache every needed dataset × backbone embedding + CLIP text embeddings + prototype artifacts.

Feature caches (skip-if-exists):
  mnist/cifar10: train + test splits × all backbones
  mini_imagenet: R&L test-class pool × all backbones
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.data import load_pool
from src.embeddings import (BACKBONES, build_backbone, cache_clip_text_embeddings,
                            cache_features, feat_path, load_features)
from src.evaluation import save_prototype_artifacts
from src.utils import allow_insecure_downloads, get_device


POOL_SPECS = [("mnist", "train"), ("mnist", "test"),
              ("cifar10", "train"), ("cifar10", "test"),
              ("mini_imagenet", "test")]


def main():
    allow_insecure_downloads()

    pools = {}

    def pool(ds, split):
        if (ds, split) not in pools:
            pools[(ds, split)] = load_pool(ds, split)
        return pools[(ds, split)]

    for bb in BACKBONES:
        todo = [(d, s) for d, s in POOL_SPECS if not feat_path(d, s, bb).exists()]
        if not todo:
            print(f"[skip] {bb}: all caches present")
            continue
        print(f"=== backbone: {bb} ===", flush=True)
        extract_fn, dim = build_backbone(bb)
        for ds, split in todo:
            path = cache_features(pool(ds, split), bb, extract_fn, dim)
            print(f"  cached {ds}/{split} -> {path.name}")
        del extract_fn
        if get_device() == "cuda":
            torch.cuda.empty_cache()

    for ds in ("mnist", "cifar10", "mini_imagenet"):
        split = "test" if ds == "mini_imagenet" else "train"
        names = load_features(ds, split, "clip_vitb32")["class_names"]
        print(f"CLIP text embeddings: {cache_clip_text_embeddings(ds, names).name}")

    # Class-mean prototypes from TRAIN splits only (ADR 0003): full-eval-split
    # statistics must never become stage-2 training targets — that would leak
    # test data into FM training and be unfair to the support-only baselines.
    for ds, split in [("mnist", "train"), ("cifar10", "train")]:
        for bb in BACKBONES:
            save_prototype_artifacts(load_features(ds, split, bb), ds, split, bb)
        print(f"prototypes saved for {ds}/{split} (all backbones)")
    print("done.")


if __name__ == "__main__":
    main()
