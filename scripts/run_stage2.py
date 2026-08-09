"""Run the full Stage 2 flow-matching grid on cached frozen features.

Protocol (spec: `_docs/stage_2.pdf`; design decisions: docs/adr/0007):
  For every Stage-1 setting (dataset x encoder x K x seed, ADR 0006 pair only),
  train FM transport toward the identical class prototypes and compare against
  the Stage-1 prototype baseline:
    * standard FM  — trained once per setting (training is T-independent),
                     evaluated at T in {4, 12};
    * rolled-out FM — one model per T, trained through the same T-step Euler
                     sequence used at inference.
  Runs mirror Stage 1: K in {5, 10} -> 3 subset seeds (FM init fixed);
  full -> 3 FM init seeds. Top-1 on the complete official test split.
  Branches: image_prototype (spec) and clip_text (extension, marked in tables).

Guard: before any training, the T=0 classification must reproduce the Stage-1
baseline accuracy for that exact setting (asserted against runs.csv).

`--smoke` shortens training and shrinks the grid for a pipeline check.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import torch

from src.classifiers import PrototypeClassifier
from src.data import training_indices
from src.embeddings import clip_text_path, load_features
from src.evaluation import (artifacts_dir, metrics_dir, save_predictions,
                            save_raw, save_table, summarize, top1)
from src.flow_matching import FlowMatchingHead
from src.utils import load_config, set_seed


def k_label(k):
    return "full" if k == "full" else f"{k}shot"


def stage1_baseline(runs1, ds, enc, target, k, seed):
    """Per-seed-paired Stage-1 baseline accuracy for one Stage-2 setting."""
    if target == "clip_text":
        r = runs1[(runs1["dataset"] == ds) & (runs1["head"] == "zeroshot_clip")]
        assert len(r) == 1, f"no unique zero-shot baseline for {ds}"
        return float(r["test_acc"].iloc[0])
    r = runs1[(runs1["dataset"] == ds) & (runs1["encoder"] == enc)
              & (runs1["head"] == "image_prototype") & (runs1["k_shot"] == k_label(k))]
    if k != "full":
        r = r[r["seed"] == seed]
    assert len(r) == 1, f"no unique prototype baseline for {ds}/{enc}/{k_label(k)}/seed{seed}"
    return float(r["test_acc"].iloc[0])


def save_fm_curves(name, history):
    d = artifacts_dir("curves_stage2")
    with open(d / f"{name}.json", "w") as f:
        json.dump(history, f)


def fm_models_dir():
    d = artifacts_dir("fm_models")
    return d


def main(smoke=False):
    cfg = load_config()
    s2 = cfg["stage2"]
    overrides = {"max_epochs": cfg["smoke"]["max_epochs"]} if smoke else {}
    tag = "_smoke" if smoke else ""
    T_values = [s2["T_values"][0]] if smoke else s2["T_values"]
    shots = [5] if smoke else cfg["shots"]

    runs1 = pd.read_csv(metrics_dir() / "runs.csv")
    runs, summary = [], []

    for target, pairs in s2["branches"].items():
        if smoke:
            pairs = pairs[:1]
        for ds, enc in pairs:
            n_classes = cfg["datasets"][ds]["n_classes"]
            f = {s: load_features(ds, s, enc) for s in ("train", "test")}
            Xtr, ytr = f["train"]["features"], f["train"]["labels"].long()
            Xte, yte = f["test"]["features"], f["test"]["labels"].long()

            text_protos = None
            if target == "clip_text":
                text = torch.load(clip_text_path(ds), weights_only=True)
                assert text["class_names"] == f["train"]["class_names"], \
                    f"{ds}: text prototypes and features disagree on class order"
                text_protos = text["text_prototypes"].float()

            for k in shots:
                seeds = cfg["init_seeds"] if k == "full" else cfg["subset_seeds"]
                if smoke:
                    seeds = seeds[:1]
                accs = {("fm_standard", T): [] for T in T_values}
                accs.update({("fm_rollout", T): [] for T in T_values})
                deltas = {key: [] for key in accs}

                for run, seed in enumerate(seeds):
                    subset_seed = 0 if k == "full" else seed
                    init_seed = seed if k == "full" else 0
                    idx = training_indices(ds, k, subset_seed, f["train"]["labels"].numpy(),
                                           fingerprint=f["train"].get("pool_fingerprint"))

                    if target == "image_prototype":
                        protos = PrototypeClassifier(n_classes).fit(Xtr[idx], ytr[idx]).prototypes
                    else:
                        protos = text_protos

                    # T=0 guard: no-transport classification == Stage-1 baseline
                    base = stage1_baseline(runs1, ds, enc, target, k, seed if k != "full" else 0)
                    z = torch.nn.functional.normalize(Xte.float(), dim=-1)
                    acc0 = top1((z @ protos.T).argmax(-1), yte)
                    assert abs(acc0 - base) < 1e-6, \
                        f"{ds}/{enc}/{target}/{k_label(k)}/seed{seed}: T=0 acc " \
                        f"{acc0:.6f} != Stage-1 baseline {base:.6f}"

                    def train_one(mode, T_train):
                        set_seed(init_seed)
                        head = FlowMatchingHead(protos, mode, T=T_train,
                                                seed=init_seed, **overrides)
                        head.fit(Xtr[idx], ytr[idx])
                        suffix = f"_T{T_train}" if mode == "rollout" else ""
                        name = f"{ds}_{enc}_{target}_fm_{mode}{suffix}_{k_label(k)}_seed{seed}{tag}"
                        save_fm_curves(name, head.history)
                        if not smoke:
                            head.save(fm_models_dir() / f"{name}.pt")
                        return head

                    def record(head, head_name, T_eval):
                        pred = head.predict(Xte, T_eval)
                        acc = top1(pred, yte)
                        accs[(head_name, T_eval)].append(acc)
                        deltas[(head_name, T_eval)].append(acc - base)
                        runs.append({"dataset": ds, "encoder": enc, "target": target,
                                     "head": head_name, "T": T_eval,
                                     "k_shot": k_label(k), "run": run,
                                     "seed_type": "init" if k == "full" else "subset",
                                     "seed": seed, "n_train": len(idx),
                                     "test_acc": acc, "baseline_acc": base,
                                     "delta_acc": acc - base,
                                     "final_train_loss": head.history["train_loss"][-1],
                                     "checkpoint_epoch": head.best["epoch"],
                                     "checkpoint_train_loss": head.best["train_loss"]})
                        if run == 0:
                            save_predictions(f"run2{tag}_{ds}_{enc}_{target}_"
                                             f"{head_name}_T{T_eval}_{k_label(k)}", pred, yte)
                        print(f"[{head_name}] {ds}/{enc}/{target} {k_label(k)} run{run} "
                              f"T={T_eval}: test {100 * acc:.2f} "
                              f"(baseline {100 * base:.2f}, delta {100 * (acc - base):+.2f})",
                              flush=True)

                    # standard FM: one training, T-independent; evaluated at every T
                    std = train_one("standard", None)
                    for T in T_values:
                        record(std, "fm_standard", T)
                    # rolled-out FM: one training per T (train T == inference T)
                    for T in T_values:
                        record(train_one("rollout", T), "fm_rollout", T)

                for (head_name, T), a in accs.items():
                    save_raw(f"{head_name}{tag}_{ds}_{enc}_{target}_T{T}_{k_label(k)}", a)
                    m, s = summarize(a)
                    dm, dstd = summarize(deltas[(head_name, T)])
                    summary.append({"dataset": ds, "encoder": enc, "target": target,
                                    "head": head_name, "T": T, "k_shot": k_label(k),
                                    "n_runs": len(a), "mean_acc": m, "std_acc": s,
                                    "baseline_mean": m - dm,
                                    "delta_mean": dm, "delta_std": dstd})

    save_table(runs, f"runs_stage2{tag}")
    save_table(summary, f"summary_stage2{tag}")
    print("done.")


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
