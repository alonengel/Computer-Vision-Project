"""Run the Stage 3 grid: FM before the frozen Stage-1 linear probe.

Protocol (spec: `_docs/stage_3.pdf`; pre-registered decisions: docs/adr/0008):
  Per setting (DTD/ResNet-18, FGVC-Aircraft/DINOv2), K = 10, T = 4:
    1. Retrain the Stage-1 probe per subset seed {0,1,2} (identical recipe,
       probe-init 0), AUDIT its validation accuracy against runs.csv
       (< 0.25 pts), pin the weights — every method receives the same file.
    2. Exact-identity guard at FM initialization (features/logits/predictions),
       train/validation data only.
    3. Hyperparameter sweeps on seed-0 VALIDATION, per dataset:
       S1 lambda in {0,1,10,100}; S2 (beta, m) in {0.25,0.5,1} x {1,3}.
    4. Train the winners (and the pre-registered S1 lambda=0 pair) on seeds 1-2.
    5. SINGLE final test pass over all locked checkpoints; Delta is computed
       against the pinned probe's own test accuracy; runs.csv test values are
       compared post-hoc as a reproduction audit only.
  Failure policy per ADR 0008 §7: NaN/Inf -> restart with the next fallback
  (clip 1.0 -> lr 3e-4 -> input standardization); all exhausted -> run marked
  failed and reported.

`--smoke` shrinks everything for a pipeline check.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import torch

from src.classifiers import LinearProbe
from src.data import training_indices
from src.embeddings import load_features
from src.evaluation import (artifacts_dir, metrics_dir, save_predictions,
                            save_raw, save_table, summarize, top1)
from src.stage3 import NonFiniteLoss, Stage3FM, identity_guard
from src.utils import load_config

K = None   # set from config stage3.k_shot in main()


def k_label():
    return f"{K}shot"


def models_dir():
    return artifacts_dir("stage3_models")


def save_curves3(name, history):
    with open(artifacts_dir("curves_stage3") / f"{name}.json", "w") as f:
        json.dump(history, f)


def train_with_fallbacks(make_fm, fit, label, curve_name=None):
    """ADR 0008 §7: any NaN/Inf before completing the epochs -> restart with the
    next pre-registered fallback (the ladder is cumulative); all exhausted ->
    run marked failed and reported. Partial curves of failed attempts are saved."""
    for level in range(4):
        fm = make_fm(level)
        try:
            fit(fm)
            if level > 0:
                print(f"  [fallback] {label}: completed at fallback level {level}",
                      flush=True)
            return fm, level
        except NonFiniteLoss as err:
            print(f"  [fallback] {label}: {err} -> next fallback", flush=True)
            if curve_name and fm.history and fm.history.get("epoch"):
                save_curves3(f"{curve_name}_failed_lv{level}", fm.history)
    print(f"  [FAILED] {label}: all fallbacks exhausted", flush=True)
    return None, 4


def main(smoke=False):
    global K
    cfg = load_config()
    s3 = cfg["stage3"]
    K = s3["k_shot"]
    tag = "_smoke" if smoke else ""
    epochs_override = {"epochs": cfg["smoke"]["max_epochs"]} if smoke else {}
    settings = s3["settings"][:1] if smoke else s3["settings"]
    seeds = [0] if smoke else s3["subset_seeds"]
    lambda_grid = s3["strategy1"]["lambda_grid"][:1] if smoke \
        else s3["strategy1"]["lambda_grid"]
    s2_grid = [(s3["strategy2"]["beta_grid"][0], s3["strategy2"]["m_grid"][0])] if smoke \
        else [(b, m) for b in s3["strategy2"]["beta_grid"]
              for m in s3["strategy2"]["m_grid"]]

    runs1 = pd.read_csv(metrics_dir() / "runs.csv")
    sweep_rows, trained = [], []   # trained: everything locked, tested at the end
    probe_val = {}                 # (ds, enc, seed) -> pinned probe validation accuracy

    if epochs_override:
        s3 = {**s3, "training": {**s3["training"], **epochs_override}}

    def make_fm(classifier, dim, level, Xsub):
        stats = None
        if level >= 3:
            stats = (Xsub.float().mean(0), Xsub.float().std(0))
        fm = Stage3FM(classifier, dim, seed=s3["fm_init_seed"], fallback=level,
                      standardize_stats=stats)
        fm.cfg = s3   # smoke epoch override propagates
        return fm

    for ds, enc in settings:
        n_classes = cfg["datasets"][ds]["n_classes"]
        # test features are NOT loaded here — sealed until the final pass
        f = {s: load_features(ds, s, enc) for s in ("train", "val")}
        Xtr, ytr = f["train"]["features"], f["train"]["labels"].long()
        Xval, yval = f["val"]["features"], f["val"]["labels"].long()
        dim = f["train"]["dim"]

        # ---- 1: retrain + audit + pin one probe per subset seed ----
        probes, subsets = {}, {}
        for seed in seeds:
            idx = training_indices(ds, K, seed, f["train"]["labels"].numpy(),
                                   fingerprint=f["train"].get("pool_fingerprint"))
            subsets[seed] = idx
            probe = LinearProbe(n_classes, dim, seed=s3["probe_init_seed"],
                                **({"max_epochs": cfg["smoke"]["max_epochs"]} if smoke else {}))
            probe.fit(Xtr[idx], ytr[idx], Xval, yval)
            ref = runs1[(runs1["dataset"] == ds) & (runs1["encoder"] == enc)
                        & (runs1["head"] == "linear_probe")
                        & (runs1["k_shot"] == k_label()) & (runs1["seed"] == seed)]
            diff = abs(probe.best["val_acc"] - float(ref["val_acc"].iloc[0]))
            same_ep = probe.best["epoch"] == int(ref["best_epoch"].iloc[0])
            print(f"[probe] {ds}/{enc} seed{seed}: val {100*probe.best['val_acc']:.2f} "
                  f"@ep{probe.best['epoch']} (runs.csv "
                  f"{100*float(ref['val_acc'].iloc[0]):.2f} "
                  f"@ep{int(ref['best_epoch'].iloc[0])}; |diff| {100*diff:.3f} pts; "
                  f"epoch {'match' if same_ep else 'MISMATCH — recorded'})", flush=True)
            assert smoke or diff < 0.25 / 100, \
                f"probe reproduction audit failed for {ds}/{enc}/seed{seed}"
            torch.save({"state_dict": probe.model.state_dict(), "dim": dim,
                        "n_classes": n_classes, "seed": seed},
                       models_dir() / f"probe_{ds}_{enc}_{k_label()}_seed{seed}{tag}.pt")
            probes[seed] = probe.model
            probe_val[(ds, enc, seed)] = probe.best["val_acc"]

            # ---- 2: exact-identity guard (validation data only), covering both
            # the default path and the fallback-3 standardized construction ----
            identity_guard(make_fm(probes[seed], dim, 0, Xtr[idx]), Xval[:256])
            identity_guard(make_fm(probes[seed], dim, 3, Xtr[idx]), Xval[:256])
        print(f"[guard] {ds}/{enc}: identity + probe audits passed for "
              f"{len(seeds)} seeds", flush=True)

        # ---- 3: sweeps on seed-0 validation (winners per dataset) ----
        idx0 = subsets[seeds[0]]
        Z0, y0 = Xtr[idx0], ytr[idx0]
        cand = {"s1": {}, "s2": {}}

        for lam in lambda_grid:
            label = f"{ds}/{enc} S1 lam={lam} seed{seeds[0]}"
            fm, level = train_with_fallbacks(
                lambda lv: make_fm(probes[seeds[0]], dim, lv, Z0),
                lambda m: m.fit_strategy1(Z0, y0, Xval, yval, lam), label,
                curve_name=f"{ds}_{enc}_s1_lam{lam}_seed{seeds[0]}{tag}")
            if fm is None:
                sweep_rows.append({"dataset": ds, "encoder": enc, "strategy": "s1",
                                   "param": f"lam={lam}", "status": "failed"})
                continue
            save_curves3(f"{ds}_{enc}_s1_lam{lam}_seed{seeds[0]}{tag}", fm.history)
            cand["s1"][lam] = (fm, level)
            sweep_rows.append({"dataset": ds, "encoder": enc, "strategy": "s1",
                               "param": f"lam={lam}", "status": "ok",
                               "fallback": level, "val_acc": fm.best["val_acc"],
                               "val_ce": fm.best["val_ce"],
                               "checkpoint_epoch": fm.best["epoch"]})
            print(f"[sweep] {label}: val {100*fm.best['val_acc']:.2f} "
                  f"@ep{fm.best['epoch']}", flush=True)

        for beta, m_steps in s2_grid:
            label = f"{ds}/{enc} S2 beta={beta} m={m_steps} seed{seeds[0]}"
            fm, level = train_with_fallbacks(
                lambda lv: make_fm(probes[seeds[0]], dim, lv, Z0),
                lambda md: md.fit_strategy2(Z0, y0, Xval, yval, beta, m_steps), label,
                curve_name=f"{ds}_{enc}_s2_b{beta}_m{m_steps}_seed{seeds[0]}{tag}")
            if fm is None:
                sweep_rows.append({"dataset": ds, "encoder": enc, "strategy": "s2",
                                   "param": f"beta={beta},m={m_steps}", "status": "failed"})
                continue
            save_curves3(f"{ds}_{enc}_s2_b{beta}_m{m_steps}_seed{seeds[0]}{tag}", fm.history)
            cand["s2"][(beta, m_steps)] = (fm, level)
            sweep_rows.append({"dataset": ds, "encoder": enc, "strategy": "s2",
                               "param": f"beta={beta},m={m_steps}", "status": "ok",
                               "fallback": level, "val_acc": fm.best["val_acc"],
                               "val_ce": fm.best["val_ce"],
                               "checkpoint_epoch": fm.best["epoch"]})
            print(f"[sweep] {label}: val {100*fm.best['val_acc']:.2f} "
                  f"@ep{fm.best['epoch']}", flush=True)

        def pick(d):
            # ties resolve to grid order (first max) — deterministic, documented
            return max(d.items(), key=lambda kv: (kv[1][0].best["val_acc"],
                                                  -kv[1][0].best["val_ce"]))

        # ---- 4: heads to test = winners + pre-registered S1 lambda=0 pair ----
        head_specs = []
        if cand["s1"]:
            lam_win = pick(cand["s1"])[0]
            head_specs.append(("fm_s1", {"lam": lam_win}))
            if lam_win != 0 and 0 in cand["s1"]:
                head_specs.append(("fm_s1_lambda0", {"lam": 0}))
            print(f"[winners] {ds}/{enc}: S1 lam={lam_win}", flush=True)
        else:
            print(f"[FAILED] {ds}/{enc}: every S1 configuration failed", flush=True)
        if cand["s2"]:
            s2_win = pick(cand["s2"])[0]
            head_specs.append(("fm_s2", {"beta": s2_win[0], "m": s2_win[1]}))
            print(f"[winners] {ds}/{enc}: S2 beta={s2_win[0]}, m={s2_win[1]}", flush=True)
        else:
            print(f"[FAILED] {ds}/{enc}: every S2 configuration failed", flush=True)

        # duplicate the seed-0 winning histories under the head-based names, so
        # figure code addresses every seed uniformly
        for head, params in head_specs:
            key = params["lam"] if head.startswith("fm_s1") else (params["beta"], params["m"])
            fm0w, _ = cand["s1" if head.startswith("fm_s1") else "s2"][key]
            save_curves3(f"{ds}_{enc}_{head}_seed{seeds[0]}{tag}", fm0w.history)

        for head, params in head_specs:
            for run, seed in enumerate(seeds):
                if seed == seeds[0]:   # sweep model IS the seed-0 final model
                    key = params["lam"] if head.startswith("fm_s1") else (params["beta"], params["m"])
                    fm, level = cand["s1" if head.startswith("fm_s1") else "s2"][key]
                else:
                    idx = subsets[seed]
                    label = f"{ds}/{enc} {head} seed{seed}"
                    if head.startswith("fm_s1"):
                        fm, level = train_with_fallbacks(
                            lambda lv: make_fm(probes[seed], dim, lv, Xtr[idx]),
                            lambda md: md.fit_strategy1(Xtr[idx], ytr[idx], Xval, yval,
                                                        params["lam"]), label,
                            curve_name=f"{ds}_{enc}_{head}_seed{seed}{tag}")
                    else:
                        fm, level = train_with_fallbacks(
                            lambda lv: make_fm(probes[seed], dim, lv, Xtr[idx]),
                            lambda md: md.fit_strategy2(Xtr[idx], ytr[idx], Xval, yval,
                                                        params["beta"], params["m"]), label,
                            curve_name=f"{ds}_{enc}_{head}_seed{seed}{tag}")
                    if fm is not None:
                        save_curves3(f"{ds}_{enc}_{head}_seed{seed}{tag}", fm.history)
                name = f"{ds}_{enc}_{head}_{k_label()}_seed{seed}{tag}"
                if fm is not None and not smoke:
                    fm.save(models_dir() / f"{name}.pt")
                trained.append({"dataset": ds, "encoder": enc, "head": head,
                                "params": params, "run": run, "seed": seed,
                                "fm": fm, "fallback": level,
                                "n_train": len(subsets[seed]),
                                "probe": probes[seed]})

        for run, seed in enumerate(seeds):   # the baseline rows themselves
            trained.append({"dataset": ds, "encoder": enc, "head": "pinned_probe",
                            "params": {}, "run": run, "seed": seed, "fm": None,
                            "fallback": 0, "n_train": len(subsets[seed]),
                            "probe": probes[seed]})

    save_table(sweep_rows, f"stage3_sweep{tag}")

    # ---- 5: SINGLE final test pass over all locked checkpoints ----
    print("\n[final] all training and selection locked - evaluating test once",
          flush=True)
    runs, summary = [], []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    by_setting = {}
    for ds, enc in settings:
        fte = load_features(ds, "test", enc)
        Xte, yte = fte["features"], fte["labels"].long()
        base_acc = {}
        for row in [r for r in trained if r["dataset"] == ds and r["encoder"] == enc
                    and r["head"] == "pinned_probe"]:
            with torch.no_grad():
                pred = row["probe"](Xte.float().to(device)).argmax(-1).cpu()
            acc = top1(pred, yte)
            base_acc[row["seed"]] = acc
            ref = runs1[(runs1["dataset"] == ds) & (runs1["encoder"] == enc)
                        & (runs1["head"] == "linear_probe")
                        & (runs1["k_shot"] == k_label()) & (runs1["seed"] == row["seed"])]
            print(f"[audit] pinned probe {ds}/{enc} seed{row['seed']}: test "
                  f"{100*acc:.2f} vs runs.csv {100*float(ref['test_acc'].iloc[0]):.2f} "
                  f"(post-hoc, no decision depends on this)", flush=True)
            if row["run"] == 0:
                save_predictions(f"run3{tag}_{ds}_{enc}_pinned_probe_{k_label()}",
                                 pred, yte)
        for row in [r for r in trained if r["dataset"] == ds and r["encoder"] == enc]:
            if row["head"] == "pinned_probe":
                acc, delta = base_acc[row["seed"]], 0.0
            elif row["fm"] is None:
                runs.append({"dataset": ds, "encoder": enc, "head": row["head"],
                             **{k: v for k, v in row["params"].items()},
                             "k_shot": k_label(), "run": row["run"], "seed": row["seed"],
                             "status": "failed"})
                continue
            else:
                pred = row["fm"].predict(Xte)
                acc = top1(pred, yte)
                delta = acc - base_acc[row["seed"]]
                if row["run"] == 0:
                    save_predictions(f"run3{tag}_{ds}_{enc}_{row['head']}_{k_label()}",
                                     pred, yte)
            runs.append({"dataset": ds, "encoder": enc, "head": row["head"],
                         **{k: v for k, v in row["params"].items()},
                         "k_shot": k_label(), "run": row["run"], "seed": row["seed"],
                         "seed_type": "subset", "n_train": row["n_train"],
                         "status": "ok", "fallback": row["fallback"],
                         "checkpoint_epoch": "" if row["fm"] is None else row["fm"].best["epoch"],
                         "val_acc": probe_val[(ds, enc, row["seed"])] if row["fm"] is None
                         else row["fm"].best["val_acc"],
                         "test_acc": acc, "baseline_acc": base_acc[row["seed"]],
                         "delta_acc": delta})
            by_setting.setdefault((ds, enc, row["head"]), []).append((acc, delta))
            print(f"[test] {ds}/{enc} {row['head']} seed{row['seed']}: "
                  f"{100*acc:.2f} (delta {100*delta:+.2f})", flush=True)

    for (ds, enc, head), pairs in by_setting.items():
        accs = [a for a, _ in pairs]
        deltas = [d for _, d in pairs]
        m_, s_ = summarize(accs)
        dm, dstd = summarize(deltas)
        save_raw(f"stage3_{head}{tag}_{ds}_{enc}_{k_label()}", accs)
        summary.append({"dataset": ds, "encoder": enc, "head": head,
                        "k_shot": k_label(), "n_runs": len(accs), "mean_acc": m_,
                        "std_acc": s_, "delta_mean": dm, "delta_std": dstd})

    save_table(runs, f"runs_stage3{tag}")
    save_table(summary, f"summary_stage3{tag}")
    print("done.")


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
