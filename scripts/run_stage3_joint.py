"""Stage-3 OPTIONAL extension (ADR 0008 §9), run only after the mandatory
comparison completed and passed its checks: joint fine-tuning of FM +
classifier (upper reference), plus the classifier-only continued-training
CONTROL — the pinned probe trained further alone with the same budget and the
same classifier lr — without which a joint gain could not be attributed to the
FM rather than to the classifier simply training longer.

Same core policy as the mandatory runs (exactly 200 epochs, best validation
checkpoint, tie-breaks). The fallback LADDER is not wired here: a NaN/Inf loss
aborts the run loudly (fit_joint raises NonFiniteLoss; the control asserts
finiteness) rather than restarting — acceptable for the optional extension,
and no such event occurred. No sweep. Delta is against the same pinned probes
as the main table; test evaluated once at the end over locked checkpoints.
Outputs: runs_stage3_joint.csv / summary_stage3_joint.csv, curves, run-0
predictions.
"""
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F

from src.data import training_indices
from src.embeddings import load_features
from src.evaluation import (artifacts_dir, save_predictions, save_raw,
                            save_table, summarize, top1)
from src.stage3 import Stage3FM, load_pinned_probe
from src.utils import load_config


def continue_train_classifier(probe, Xtr, ytr, Xval, yval, lr, epochs, batch, seed):
    """The control: the pinned probe trained further ALONE (no FM), same budget,
    best-validation checkpoint. Operates on a copy; the pinned file is untouched."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    clf = copy.deepcopy(probe).to(dev).train()
    for p in clf.parameters():
        p.requires_grad_(True)
    Z, y = Xtr.float().to(dev), ytr.long().to(dev)
    Xv, yv = Xval.float().to(dev), yval.long().to(dev)
    opt = torch.optim.AdamW(clf.parameters(), lr=lr, weight_decay=1e-4)
    g = torch.Generator().manual_seed(seed)
    hist = {k: [] for k in ("epoch", "train_ce", "val_ce", "val_acc")}
    best, best_state = None, None
    n = len(Z)
    for epoch in range(epochs):
        clf.train()
        perm = torch.randperm(n, generator=g).to(dev)
        tot = 0.0
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            loss = F.cross_entropy(clf(Z[idx]), y[idx])
            assert torch.isfinite(loss), "non-finite loss in classifier-only control"
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        clf.eval()
        with torch.no_grad():
            logits = clf(Xv)
            val_ce = F.cross_entropy(logits, yv).item()
            val_acc = (logits.argmax(-1) == yv).float().mean().item()
        hist["epoch"].append(epoch)
        hist["train_ce"].append(tot / n)
        hist["val_ce"].append(val_ce)
        hist["val_acc"].append(val_acc)
        better = best is None or val_acc > best["val_acc"] or \
            (val_acc == best["val_acc"] and val_ce < best["val_ce"])
        if better:
            best = {"val_acc": val_acc, "val_ce": val_ce, "epoch": epoch}
            best_state = copy.deepcopy(clf.state_dict())
    clf.load_state_dict(best_state)
    clf.eval()
    return clf, best, hist


def main():
    cfg = load_config()
    s3 = cfg["stage3"]
    k = f"{s3['k_shot']}shot"
    mdir = artifacts_dir("stage3_models")
    clf_lr = s3["joint_extension"]["classifier_lr"]
    tr = s3["training"]

    trained = []
    for ds, enc in s3["settings"]:
        f = {s: load_features(ds, s, enc) for s in ("train", "val")}
        Xtr, ytr = f["train"]["features"], f["train"]["labels"].long()
        Xval, yval = f["val"]["features"], f["val"]["labels"].long()
        dim = f["train"]["dim"]
        for run, seed in enumerate(s3["subset_seeds"]):
            idx = training_indices(ds, s3["k_shot"], seed, f["train"]["labels"].numpy(),
                                   fingerprint=f["train"].get("pool_fingerprint"))
            probe = load_pinned_probe(mdir / f"probe_{ds}_{enc}_{k}_seed{seed}.pt")

            fm = Stage3FM(probe, dim, seed=s3["fm_init_seed"])
            fm.fit_joint(Xtr[idx], ytr[idx], Xval, yval, clf_lr)
            with open(artifacts_dir("curves_stage3") /
                      f"{ds}_{enc}_joint_seed{seed}.json", "w") as fp:
                json.dump(fm.history, fp)
            print(f"[joint] {ds}/{enc} seed{seed}: val {100*fm.best['val_acc']:.2f} "
                  f"@ep{fm.best['epoch']}", flush=True)

            ctrl, ctrl_best, ctrl_hist = continue_train_classifier(
                probe, Xtr[idx], ytr[idx], Xval, yval, clf_lr, tr["epochs"],
                tr["batch_size"], s3["fm_init_seed"])
            with open(artifacts_dir("curves_stage3") /
                      f"{ds}_{enc}_clf_only_seed{seed}.json", "w") as fp:
                json.dump(ctrl_hist, fp)
            print(f"[clf-only] {ds}/{enc} seed{seed}: val "
                  f"{100*ctrl_best['val_acc']:.2f} @ep{ctrl_best['epoch']}", flush=True)

            trained.append({"ds": ds, "enc": enc, "run": run, "seed": seed,
                            "probe": probe, "joint": fm, "ctrl": ctrl,
                            "joint_best": fm.best, "ctrl_best": ctrl_best,
                            "n_train": len(idx)})

    print("\n[final] extension locked - evaluating test once", flush=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    runs, summary, acc_by = [], [], {}
    for ds, enc in s3["settings"]:
        fte = load_features(ds, "test", enc)
        Xte, yte = fte["features"], fte["labels"].long()
        for row in [r for r in trained if r["ds"] == ds and r["enc"] == enc]:
            with torch.no_grad():
                base = top1(row["probe"].to(dev)(Xte.float().to(dev)).argmax(-1).cpu(), yte)
            for head, model_pred, best in (
                    ("fm_joint", lambda: row["joint"].predict(Xte), row["joint_best"]),
                    ("clf_only_continued",
                     lambda: row["ctrl"](Xte.float().to(dev)).argmax(-1).cpu(),
                     row["ctrl_best"])):
                with torch.no_grad():
                    pred = model_pred()
                acc = top1(pred, yte)
                if row["run"] == 0:
                    save_predictions(f"run3_{ds}_{enc}_{head}_{k}", pred, yte)
                runs.append({"dataset": ds, "encoder": enc, "head": head,
                             "k_shot": k, "run": row["run"], "seed": row["seed"],
                             "seed_type": "subset", "n_train": row["n_train"],
                             "status": "ok", "checkpoint_epoch": best["epoch"],
                             "val_acc": best["val_acc"], "test_acc": acc,
                             "baseline_acc": base, "delta_acc": acc - base})
                acc_by.setdefault((ds, enc, head), []).append((acc, acc - base))
                print(f"[test] {ds}/{enc} {head} seed{row['seed']}: {100*acc:.2f} "
                      f"(delta {100*(acc-base):+.2f})", flush=True)

    for (ds, enc, head), pairs in acc_by.items():
        accs = [a for a, _ in pairs]
        deltas = [d for _, d in pairs]
        m_, s_ = summarize(accs)
        dm, dstd = summarize(deltas)
        save_raw(f"stage3_{head}_{ds}_{enc}_{k}", accs)
        summary.append({"dataset": ds, "encoder": enc, "head": head, "k_shot": k,
                        "n_runs": len(accs), "mean_acc": m_, "std_acc": s_,
                        "delta_mean": dm, "delta_std": dstd})
    save_table(runs, "runs_stage3_joint")
    save_table(summary, "summary_stage3_joint")
    print("done.")


if __name__ == "__main__":
    main()
