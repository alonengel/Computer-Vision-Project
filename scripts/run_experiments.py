"""Run the full Stage 1 experiment grid from cached features + saved episode files.

Episodic (5-way, K∈{1,5}, 600 episodes, mean ± 95% CI):
  prototype-cosine / prototype-euclidean on all 3 backbones,
  linear probe + zero-shot CLIP (primary & ensemble prompts) on clip_vitb32.
Simple (all classes, K∈{1,5,10}, 10 seeds, full test set, mean ± std) on mnist/cifar10:
  same classifier grid.

--smoke runs a reduced grid (20 episodes, 2 seeds) for sanity checking.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from src.classifiers import LinearProbe, PrototypeClassifier, ZeroShotCLIP
from src.data import episodic_path, simple_path
from src.embeddings import clip_text_path, load_features
from src.evaluation import ci95, run_episodic, run_simple, save_raw, save_table
from src.utils import load_config, set_seed

BACKBONE_GRID = ["clip_vitb32", "dinov2_vits14", "resnet50"]
PRIMARY = "clip_vitb32"


def load_text(dataset):
    from src.embeddings import PROMPT_TEMPLATES

    text = torch.load(clip_text_path(dataset), weights_only=True)
    assert text["templates"] == PROMPT_TEMPLATES[dataset], \
        f"stale CLIP text cache for {dataset}: templates changed — re-run extraction"
    return text


def support_classifiers():
    """(name, backbone, classifier) — heads that use the support set."""
    grid = []
    for bb in BACKBONE_GRID:
        grid.append((f"proto_cos__{bb}", bb, PrototypeClassifier("cosine")))
        grid.append((f"proto_eucl__{bb}", bb, PrototypeClassifier("euclidean")))
    grid.append((f"linear__{PRIMARY}", PRIMARY, LinearProbe()))
    return grid


def zeroshot_classifiers(dataset):
    """(name, backbone, classifier) — support-free heads (zero-shot CLIP)."""
    text = load_text(dataset)
    return [(f"clip_zeroshot__{PRIMARY}", PRIMARY, ZeroShotCLIP(text["primary"])),
            (f"clip_zeroshot_ens__{PRIMARY}", PRIMARY, ZeroShotCLIP(text["ensemble"]))]


def main(smoke=False):
    cfg = load_config()
    ep_cfg, si_cfg = cfg["episodic"], cfg["simple"]
    n_episodes = cfg["smoke"]["n_episodes"] if smoke else ep_cfg["n_episodes"]
    seeds = si_cfg["seeds"][:2] if smoke else si_cfg["seeds"]
    tag = "_smoke" if smoke else ""
    set_seed(ep_cfg["seed"])

    episodic_rows = []
    for ds in cfg["datasets"]:
        feats = {bb: load_features(ds, "test", bb) for bb in BACKBONE_GRID}
        for k in ep_cfg["shots"]:
            episode = torch.load(episodic_path(ds, ep_cfg["n_way"], k, ep_cfg["seed"]),
                                 weights_only=True)
            assert episode["n_way"] == ep_cfg["n_way"] and \
                episode["n_query"] == ep_cfg["n_query"] and \
                episode["classes"].shape[0] == ep_cfg["n_episodes"], \
                f"episode file for {ds} K={k} does not match config — regenerate"
            if smoke:
                episode = {**episode,
                           "classes": episode["classes"][:n_episodes],
                           "support_idx": episode["support_idx"][:n_episodes],
                           "query_idx": episode["query_idx"][:n_episodes]}
            grid = ([(n, b, c, False) for n, b, c in support_classifiers()] +
                    [(n, b, c, True) for n, b, c in zeroshot_classifiers(ds)])
            for name, bb, clf, subset_aware in grid:
                accs = run_episodic(feats[bb], episode, clf, class_subset_aware=subset_aware)
                save_raw(f"ep{tag}_{ds}_{ep_cfg['n_way']}w{k}s_{name}", accs)
                episodic_rows.append({
                    "dataset": ds, "classifier": name, "backbone": bb,
                    "n_way": ep_cfg["n_way"], "k_shot": k, "n_episodes": len(accs),
                    "acc": float(np.mean(accs)), "ci95": float(ci95(accs)),
                })
                print(f"[ep] {ds} {ep_cfg['n_way']}w{k}s {name}: "
                      f"{100 * np.mean(accs):.2f} ± {100 * ci95(accs):.2f}", flush=True)
    save_table(episodic_rows, f"episodic{tag}")

    simple_rows = []
    for ds in ("mnist", "cifar10"):
        train_f = {bb: load_features(ds, "train", bb) for bb in BACKBONE_GRID}
        test_f = {bb: load_features(ds, "test", bb) for bb in BACKBONE_GRID}
        for k in si_cfg["shots"]:
            for name, bb, clf in support_classifiers():
                accs = []
                for seed in seeds:
                    support = torch.load(simple_path(ds, k, seed), weights_only=True)
                    if isinstance(clf, LinearProbe):
                        clf = LinearProbe(seed=seed)  # fresh head per seed
                    acc, pred, true = run_simple(train_f[bb], test_f[bb], support, clf)
                    accs.append(acc)
                save_raw(f"simple{tag}_{ds}_{k}s_{name}", accs)
                simple_rows.append({
                    "dataset": ds, "classifier": name, "backbone": bb, "k_shot": k,
                    "n_seeds": len(accs),
                    "acc": float(np.mean(accs)), "std": float(np.std(accs, ddof=1)),
                })
                print(f"[simple] {ds} {k}s {name}: "
                      f"{100 * np.mean(accs):.2f} ± {100 * np.std(accs, ddof=1):.2f}", flush=True)
        # Zero-shot ignores support: K- and seed-independent, so report one row
        # per dataset (k_shot=0) instead of duplicated "± 0.00" rows.
        support0 = torch.load(simple_path(ds, si_cfg["shots"][0], seeds[0]), weights_only=True)
        for name, bb, clf in zeroshot_classifiers(ds):
            acc, pred, true = run_simple(train_f[bb], test_f[bb], support0, clf)
            save_raw(f"simple{tag}_{ds}_0s_{name}", [acc])
            simple_rows.append({
                "dataset": ds, "classifier": name, "backbone": bb, "k_shot": 0,
                "n_seeds": 1, "acc": float(acc), "std": 0.0,
            })
            print(f"[simple] {ds} zero-shot {name}: {100 * acc:.2f}", flush=True)
    save_table(simple_rows, f"simple{tag}")
    print("done.")


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
