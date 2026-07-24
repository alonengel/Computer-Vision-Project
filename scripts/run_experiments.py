"""Run the full Stage 1 experiment grid on cached frozen features.

Protocol (spec: `_docs/stage_1.pdf`):
  K in {5, 10, full} training images per class, from the official training split.
  * 5-shot / 10-shot : 3 runs, one per balanced-subset seed {0,1,2}
                       (linear-probe initialization fixed, so the spread measures
                        training-subset sampling).
  * full linear probe: 3 runs, one per classifier-initialization seed {0,1,2}
                       (training set fixed, so the spread measures initialization).
  * full image prototypes and zero-shot CLIP: a single run each (deterministic).
Model selection uses the validation split only (linear-probe checkpointing);
top-1 accuracy is measured once on the complete official test split.

`--smoke` shortens the probe to a few epochs for a pipeline check.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.classifiers import LinearProbe, PrototypeClassifier, ZeroShotCLIP
from src.data import training_indices
from src.embeddings import (clip_text_path, encoders_for, load_features,
                            supervised_encoders_for)
from src.evaluation import (save_curves, save_predictions, save_raw, save_table,
                            summarize, top1)
from src.utils import load_config, set_seed


def k_label(k):
    return "full" if k == "full" else f"{k}shot"


def main(smoke=False, only_datasets=None):
    cfg = load_config()
    overrides = {"max_epochs": cfg["smoke"]["max_epochs"]} if smoke else {}
    tag = "_smoke" if smoke else ""
    runs, summary = [], []

    for ds in (only_datasets or list(cfg["datasets"])):
        n_classes = cfg["datasets"][ds]["n_classes"]
        for enc in supervised_encoders_for(ds):
            f = {s: load_features(ds, s, enc) for s in ("train", "val", "test")}
            Xtr, ytr = f["train"]["features"], f["train"]["labels"].long()
            Xval, yval = f["val"]["features"], f["val"]["labels"].long()
            Xte, yte = f["test"]["features"], f["test"]["labels"].long()
            dim = f["train"]["dim"]

            for k in cfg["shots"]:
                # ---- linear probe (required baseline) ----
                accs = []
                for run, seed in enumerate(cfg["init_seeds"] if k == "full"
                                           else cfg["subset_seeds"]):
                    subset_seed = 0 if k == "full" else seed
                    init_seed = seed if k == "full" else 0
                    idx = training_indices(ds, k, subset_seed, f["train"]["labels"].numpy())
                    set_seed(init_seed)
                    probe = LinearProbe(n_classes, dim, seed=init_seed, **overrides)
                    probe.fit(Xtr[idx], ytr[idx], Xval, yval)
                    pred = probe.predict(Xte)
                    acc = top1(pred, yte)
                    accs.append(acc)
                    runs.append({"dataset": ds, "encoder": enc, "head": "linear_probe",
                                 "k_shot": k_label(k), "run": run,
                                 "seed_type": "init" if k == "full" else "subset",
                                 "seed": seed, "n_train": len(idx),
                                 "test_acc": acc, "val_acc": probe.best["val_acc"],
                                 "best_epoch": probe.best["epoch"]})
                    if run == 0:
                        save_predictions(f"{tag.lstrip('_') or 'run'}_{ds}_{enc}_"
                                         f"linear_probe_{k_label(k)}", pred, yte)
                    if k == 10 and run == 0 and not smoke:
                        # representative 10-shot run per dataset-encoder: loss curves
                        save_curves(f"{ds}_{enc}_10shot_seed0", probe.history)
                    print(f"[probe] {ds}/{enc} {k_label(k)} run{run}: "
                          f"test {100 * acc:.2f} (val {100 * probe.best['val_acc']:.2f} "
                          f"@ep{probe.best['epoch']})", flush=True)
                save_raw(f"linear_probe{tag}_{ds}_{enc}_{k_label(k)}", accs)
                m, s = summarize(accs)
                summary.append({"dataset": ds, "encoder": enc, "head": "linear_probe",
                                "k_shot": k_label(k), "n_runs": len(accs),
                                "mean_acc": m, "std_acc": s})

                # ---- branch A: image-derived class prototypes ----
                seeds = [0] if k == "full" else cfg["subset_seeds"]
                accs = []
                for run, seed in enumerate(seeds):
                    idx = training_indices(ds, k, seed, f["train"]["labels"].numpy())
                    proto = PrototypeClassifier(n_classes).fit(Xtr[idx], ytr[idx])
                    pred = proto.predict(Xte)
                    acc = top1(pred, yte)
                    accs.append(acc)
                    runs.append({"dataset": ds, "encoder": enc, "head": "image_prototype",
                                 "k_shot": k_label(k), "run": run, "seed_type": "subset",
                                 "seed": seed, "n_train": len(idx), "test_acc": acc,
                                 "val_acc": "", "best_epoch": ""})
                    if run == 0:
                        save_predictions(f"{tag.lstrip('_') or 'run'}_{ds}_{enc}_"
                                         f"image_prototype_{k_label(k)}", pred, yte)
                    print(f"[proto] {ds}/{enc} {k_label(k)} run{run}: "
                          f"test {100 * acc:.2f}", flush=True)
                save_raw(f"image_prototype{tag}_{ds}_{enc}_{k_label(k)}", accs)
                m, s = summarize(accs)
                summary.append({"dataset": ds, "encoder": enc, "head": "image_prototype",
                                "k_shot": k_label(k), "n_runs": len(accs),
                                "mean_acc": m, "std_acc": s})

        # ---- branch B: zero-shot CLIP (no labeled training images) ----
        if "clip_rn50" in encoders_for(ds):
            fte = load_features(ds, "test", "clip_rn50")
            text = torch.load(clip_text_path(ds), weights_only=True)
            assert text["class_names"] == fte["class_names"], \
                f"{ds}: text prototypes and features disagree on class order"
            zs = ZeroShotCLIP(text["text_prototypes"])
            yte_c = fte["labels"].long()
            pred = zs.predict(fte["features"])
            acc = top1(pred, yte_c)
            runs.append({"dataset": ds, "encoder": "clip_rn50", "head": "zeroshot_clip",
                         "k_shot": "none", "run": 0, "seed_type": "deterministic",
                         "seed": 0, "n_train": 0, "test_acc": acc,
                         "val_acc": "", "best_epoch": ""})
            save_predictions(f"{tag.lstrip('_') or 'run'}_{ds}_clip_rn50_zeroshot_clip_none",
                             pred, yte_c)
            save_raw(f"zeroshot_clip{tag}_{ds}_clip_rn50_none", [acc])
            summary.append({"dataset": ds, "encoder": "clip_rn50", "head": "zeroshot_clip",
                            "k_shot": "none", "n_runs": 1, "mean_acc": acc, "std_acc": 0.0})
            print(f"[zeroshot] {ds}: test {100 * acc:.2f}", flush=True)

    save_table(runs, f"runs{tag}")
    save_table(summary, f"summary{tag}")
    print("done.")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    main(smoke="--smoke" in sys.argv, only_datasets=args or None)
