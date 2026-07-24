"""Cache train/validation/test features for every (dataset, encoder) pair, plus
CLIP RN50 text prototypes for the zero-shot branch.

Encoders stay frozen; extraction happens once and every classifier is trained and
evaluated on these caches (spec: `_docs/stage_1.pdf`).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.data import SPLITS, load_pool
from src.embeddings import (build_encoder, cache_clip_text_prototypes, cache_features,
                            encoders_for, feat_path)
from src.utils import allow_insecure_downloads, get_device, load_config


def main(only_datasets=None):
    allow_insecure_downloads()
    cfg = load_config()
    datasets = only_datasets or list(cfg["datasets"])

    # (encoder -> [(dataset, split)]) so each encoder is built at most once.
    todo = {}
    for ds in datasets:
        for enc in encoders_for(ds):
            for split in SPLITS:
                if not feat_path(ds, split, enc).exists():
                    todo.setdefault(enc, []).append((ds, split))

    pools = {}

    def pool(ds, split):
        if (ds, split) not in pools:
            pools[(ds, split)] = load_pool(ds, split)
        return pools[(ds, split)]

    for enc, items in todo.items():
        print(f"=== encoder: {enc} ===", flush=True)
        extract_fn, dim = build_encoder(enc)
        print(f"  dim = {dim}")
        for ds, split in items:
            path = cache_features(pool(ds, split), enc, extract_fn, dim)
            print(f"  cached {ds}/{split} -> {path.name}", flush=True)
        del extract_fn
        if get_device() == "cuda":
            torch.cuda.empty_cache()
    if not todo:
        print("[skip] all feature caches present")

    for ds in datasets:
        if "clip_rn50" not in encoders_for(ds):
            continue
        names = pool(ds, "test").class_names
        path = cache_clip_text_prototypes(ds, names)
        print(f"CLIP text prototypes: {path.name} "
              f"(prompt: {cfg['datasets'][ds]['prompt']})")
    print("done.")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    main(only_datasets=args or None)
