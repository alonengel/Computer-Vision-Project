"""Beyond-spec control: raw-feature FM vs the published L2-normalized FM.

ADR 0007 §3 argues that transporting RAW frozen features (norms ~10-40) toward
unit-norm prototypes would spend the model's capacity on scale, not class
structure. The published Stage-2 results assert that rationale; this control
measures it (audit follow-up): for the seed-0 slice of the image-prototype
branch (3 dataset-encoder settings x K in {5,10,full} x {standard, rollout},
T = 12), train the identical velocity network with the identical recipe,
targets, subsets and seeds on raw features (normalize=False) and compare
against the already-published normalized run of the same setting.

Nothing here touches the main results: outputs go to
results/metrics/stage2_raw_ablation.{csv,md} and raw arrays with an
`ablation_` prefix. The published tables/figures are not rewritten.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import torch

from src.classifiers import PrototypeClassifier
from src.data import training_indices
from src.embeddings import load_features
from src.evaluation import metrics_dir, save_raw, top1
from src.flow_matching import FlowMatchingHead
from src.utils import load_config, set_seed
from src.visualize import dataset_label, encoder_label, head_label

T_ABL = 12
SEED = 0


def k_label(k):
    return "full" if k == "full" else f"{k}shot"


def main():
    cfg = load_config()
    runs2 = pd.read_csv(metrics_dir() / "runs_stage2.csv")
    rows = []

    for ds, enc in cfg["stage2"]["branches"]["image_prototype"]:
        n_classes = cfg["datasets"][ds]["n_classes"]
        f = {s: load_features(ds, s, enc) for s in ("train", "test")}
        Xtr, ytr = f["train"]["features"], f["train"]["labels"].long()
        Xte, yte = f["test"]["features"], f["test"]["labels"].long()

        for k in cfg["shots"]:
            idx = training_indices(ds, k, SEED, f["train"]["labels"].numpy(),
                                   fingerprint=f["train"].get("pool_fingerprint"))
            protos = PrototypeClassifier(n_classes).fit(Xtr[idx], ytr[idx]).prototypes

            for mode, head_name in (("standard", "fm_standard"), ("rollout", "fm_rollout")):
                # published normalized counterpart: run 0 of the same setting
                pub = runs2[(runs2["dataset"] == ds) & (runs2["encoder"] == enc)
                            & (runs2["target"] == "image_prototype")
                            & (runs2["head"] == head_name) & (runs2["T"] == T_ABL)
                            & (runs2["k_shot"] == k_label(k)) & (runs2["run"] == 0)]
                assert len(pub) == 1
                norm_acc = float(pub["test_acc"].iloc[0])
                base = float(pub["baseline_acc"].iloc[0])

                set_seed(SEED)
                head = FlowMatchingHead(protos, mode, T=T_ABL if mode == "rollout" else None,
                                        seed=SEED, normalize=False)
                head.fit(Xtr[idx], ytr[idx])
                raw_acc = top1(head.predict(Xte, T_ABL), yte)
                save_raw(f"ablation_raw_{head_name}_{ds}_{enc}_T{T_ABL}_{k_label(k)}",
                         [raw_acc])
                rows.append({"dataset": ds, "encoder": enc, "head": head_name,
                             "k_shot": k_label(k), "T": T_ABL, "seed": SEED,
                             "baseline_acc": base, "normalized_acc": norm_acc,
                             "raw_acc": raw_acc,
                             "raw_minus_normalized": raw_acc - norm_acc,
                             "raw_final_train_loss": head.history["train_loss"][-1],
                             "raw_checkpoint_epoch": head.best["epoch"]})
                print(f"[ablation] {ds}/{enc} {k_label(k)} {head_name}: "
                      f"raw {100 * raw_acc:.2f} vs normalized {100 * norm_acc:.2f} "
                      f"(baseline {100 * base:.2f})", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(metrics_dir() / "stage2_raw_ablation.csv", index=False)

    lines = ["| Dataset | Encoder | Head | K | Stage-1 baseline | FM on raw features "
             "| FM on normalized features (published) |",
             "|---|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        better = r["normalized_acc"] >= r["raw_acc"]
        raw_c = f"{100 * r['raw_acc']:.2f}"
        norm_c = f"{100 * r['normalized_acc']:.2f}"
        lines.append(f"| {dataset_label(r['dataset'])} | "
                     f"{encoder_label(r['encoder'], short=True)} | "
                     f"{head_label(r['head'])} | {r['k_shot'].replace('shot', '')} | "
                     f"{100 * r['baseline_acc']:.2f} | "
                     f"{raw_c} | {('**' + norm_c + '**') if better else norm_c} |")
    n_norm = int((df["normalized_acc"] >= df["raw_acc"]).sum())
    lines.append("")
    lines.append(f"Beyond-spec control (single run, seed 0, T = {T_ABL}; identical "
                 f"velocity network, recipe, prototypes, subsets and seeds — the only "
                 f"difference is whether the FM input features are L2-normalized). "
                 f"Normalized is better or equal in {n_norm} of {len(df)} settings. "
                 f"**Bold** marks the better FM variant. Top-1 (%) on the complete "
                 f"official test split. The Stage-1 baseline is unaffected by the "
                 f"choice (cosine classification is scale-invariant). Generated by "
                 f"`scripts/run_stage2_raw_ablation.py`; not part of the published "
                 f"grid.")
    out = metrics_dir() / "stage2_raw_ablation.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
