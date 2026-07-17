"""Final embedding selection per classifier head — on VALIDATION data only.

Protocol (leakage-free):
  1. Selection episodes are sampled from data disjoint from all test evaluation:
     MNIST/CIFAR-10 -> train split; Mini-ImageNet -> the 16 R&L validation classes
     (their canonical purpose). Seed 123, 600 episodes per K.
  2. For each (dataset, head), ONE configuration is selected across all K, by mean
     validation accuracy over K in {1,5}; the winner must beat the runner-up in a
     paired per-episode comparison on the same validation episodes — if the paired
     95% CI includes zero, the tie goes to the smaller embedding.
  3. Zero-shot CLIP has no support, so its selection dimension is the prompt
     variant (single vs ensemble) per dataset — K is not part of its identity.
  4. The selected configuration is then read out ONCE from the (pre-existing)
     test tables as the Stage-2 reference numbers.
  ResNet-50 is excluded on Mini-ImageNet (val AND test classes are ImageNet-1k
  classes it was label-supervised on — selection would inherit the contamination).

Writes results/artifacts/best_baselines.json (loaded by Stage 2) and
results/metrics/selection_validation.csv (all validation accuracies).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch

from src.classifiers import LinearProbe, PrototypeClassifier, ZeroShotCLIP
from src.data import ensure_selection_episodes, selection_path
from src.embeddings import clip_text_path, load_features
from src.evaluation import ci95, metrics_dir, run_episodic
from src.utils import load_config, repo_path, set_seed

BACKBONES = ["clip_vitb32", "dinov2_vits14", "resnet50"]
DIMS = {"clip_vitb32": 512, "dinov2_vits14": 384, "resnet50": 2048}
SEL_SPLIT = {"mnist": "train", "cifar10": "train", "mini_imagenet": "val"}
TEXT_KEY = {"mnist": "mnist", "cifar10": "cifar10", "mini_imagenet": "mini_imagenet_val"}
EXCLUDE = {("mini_imagenet", "resnet50")}  # ImageNet-label contamination


def candidates(dataset, feats):
    """{head: [(config_name, backbone, classifier, subset_aware)]} on the selection pool."""
    text = torch.load(clip_text_path(TEXT_KEY[dataset]), weights_only=True)
    heads = {"prototype": [], "linear": [], "zeroshot_clip": []}
    for bb in BACKBONES:
        if (dataset, bb) in EXCLUDE:
            continue
        heads["prototype"].append((f"proto_cos__{bb}", bb, PrototypeClassifier("cosine"), False))
        heads["prototype"].append((f"proto_eucl__{bb}", bb, PrototypeClassifier("euclidean"), False))
        heads["linear"].append((f"linear__{bb}", bb, LinearProbe(), False))
    heads["zeroshot_clip"] = [
        ("clip_zeroshot__clip_vitb32", "clip_vitb32", ZeroShotCLIP(text["primary"]), True),
        ("clip_zeroshot_ens__clip_vitb32", "clip_vitb32", ZeroShotCLIP(text["ensemble"]), True),
    ]
    return heads


def main():
    cfg = load_config()
    ep_cfg, sel_cfg = cfg["episodic"], cfg["selection"]
    set_seed(sel_cfg["seed"])
    test_ep = pd.read_csv(metrics_dir() / "episodic.csv")

    val_rows, selection = [], {}
    for ds in cfg["datasets"]:
        split = SEL_SPLIT[ds]
        feats = {bb: load_features(ds, split, bb) for bb in BACKBONES}
        ensure_selection_episodes(ds, feats["clip_vitb32"]["labels"].numpy())
        episodes = {k: torch.load(selection_path(ds, ep_cfg["n_way"], k, sel_cfg["seed"]),
                                  weights_only=True) for k in ep_cfg["shots"]}

        selection[ds] = {}
        for head, configs in candidates(ds, feats).items():
            # per-config: per-episode val accuracies concatenated across K
            accs = {}
            for name, bb, clf, subset in configs:
                per_k = [run_episodic(feats[bb], episodes[k], clf, class_subset_aware=subset)
                         for k in ep_cfg["shots"]]
                accs[name] = {"by_k": per_k, "cat": np.concatenate(per_k), "backbone": bb}
                for k, a in zip(ep_cfg["shots"], per_k):
                    val_rows.append({"dataset": ds, "head": head, "config": name,
                                     "k_shot": k, "val_acc": float(a.mean()),
                                     "val_ci95": float(ci95(a))})
                print(f"[val] {ds} {head} {name}: "
                      + " ".join(f"{k}s={100 * a.mean():.2f}"
                                 for k, a in zip(ep_cfg["shots"], per_k)), flush=True)

            ranked = sorted(accs, key=lambda n: accs[n]["cat"].mean(), reverse=True)
            winner, runner = ranked[0], ranked[1]
            diff = accs[winner]["cat"] - accs[runner]["cat"]
            margin, margin_ci = float(diff.mean()), float(ci95(diff))
            tie = margin - margin_ci <= 0
            if tie:  # statistically tied -> smaller embedding wins
                pair = sorted(ranked[:2], key=lambda n: DIMS[accs[n]["backbone"]])
                chosen = pair[0]
            else:
                chosen = winner

            test_ref = {}
            for k in ep_cfg["shots"]:
                r = test_ep[(test_ep["dataset"] == ds) & (test_ep["k_shot"] == k)
                            & (test_ep["classifier"] == chosen)].iloc[0]
                test_ref[f"{k}shot"] = {"acc": round(float(r["acc"]), 4),
                                        "ci95": round(float(r["ci95"]), 4)}
            selection[ds][head] = {
                "classifier": chosen,
                "backbone": accs[chosen]["backbone"],
                "selected_on": f"validation ({split} split)" if split != "val"
                               else "validation (16 R&L val classes)",
                "val_acc_mean": round(float(accs[chosen]["cat"].mean()), 4),
                "paired_margin_vs_runner_up": f"{100 * margin:+.2f} ± {100 * margin_ci:.2f}",
                "tie_break_smaller_embedding": bool(tie),
                "test_reference": test_ref,
            }
            print(f"  -> {ds} {head}: {chosen}"
                  + (" (tie -> smaller embedding)" if tie else f" (margin {100 * margin:+.2f})"),
                  flush=True)

    pd.DataFrame(val_rows).to_csv(metrics_dir() / "selection_validation.csv", index=False)
    out = repo_path(cfg["paths"]["artifacts_dir"]) / "best_baselines.json"
    with open(out, "w") as f:
        json.dump({"note": "Final embedding per (dataset, head), selected on VALIDATION "
                           "data only (one config across all K; zero-shot selected by "
                           "prompt variant). test_reference = one-time test read-out; "
                           "the Stage-2 reference to beat. ResNet-50 excluded on "
                           "Mini-ImageNet (label contamination).",
                   "selection_protocol": {"episodes_seed": sel_cfg["seed"],
                                          "n_episodes_per_k": sel_cfg["n_episodes"],
                                          "pools": SEL_SPLIT,
                                          "rule": "mean val acc across K; paired CI vs "
                                                  "runner-up; tie -> smaller embedding"},
                   "selection": selection}, f, indent=2)
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
