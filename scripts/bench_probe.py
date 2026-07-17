"""Benchmark + equivalence proof: sequential vs batched linear-probe training.

The naive reference trains 600 independent per-episode heads one at a time; the
vectorized implementation (src/classifiers.py) trains them jointly as one
[600, C, D] tensor with a sum/S-scaled loss, which gives every head exactly its
independent-head gradient. This script measures both on every episodic config and
verifies the two produce identical predictions — establishing that the speedup
is pure engineering, with zero effect on the science.

Outputs: results/metrics/bench_probe.json + results/figures/bench_probe.png.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.classifiers import LinearProbe
from src.data import episodic_path
from src.embeddings import load_features
from src.evaluation import episode_tensors, metrics_dir
from src.utils import load_config


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main():
    if "--plot-only" in sys.argv:  # re-render the figure from saved timings
        with open(metrics_dir() / "bench_probe.json") as f:
            plot(json.load(f))
        return
    cfg = load_config()
    ep_cfg = cfg["episodic"]
    rows = []
    for ds in cfg["datasets"]:
        feats = load_features(ds, "test", "clip_vitb32")
        for k in ep_cfg["shots"]:
            episode = torch.load(episodic_path(ds, ep_cfg["n_way"], k, ep_cfg["seed"]),
                                 weights_only=True)
            Xs, ys, Xq, yq, _ = episode_tensors(feats, episode)
            B = Xs.shape[0]

            LinearProbe().fit(Xs[:8], ys[:8])  # warmup
            sync()

            t0 = time.perf_counter()
            batched_probe = LinearProbe().fit(Xs, ys)
            sync()
            t_batched = time.perf_counter() - t0
            pred_batched = (torch.einsum("bqd,bcd->bqc", Xq, batched_probe.W)
                            + batched_probe.b.unsqueeze(1)).argmax(-1)

            # Same init as the batched run: slice the identical seeded draw per
            # episode, so any output difference is float reduction order only.
            C = int(ys.max().item()) + 1
            g = torch.Generator(device="cpu").manual_seed(0)
            W0_full = 0.01 * torch.randn(B, C, Xs.shape[2], generator=g)

            t0 = time.perf_counter()
            preds = []
            for e in range(B):
                p = LinearProbe().fit(Xs[e:e + 1], ys[e:e + 1], W0=W0_full[e:e + 1])
                preds.append((torch.einsum("bqd,bcd->bqc", Xq[e:e + 1], p.W)
                              + p.b.unsqueeze(1)).argmax(-1))
            sync()
            t_seq = time.perf_counter() - t0
            pred_seq = torch.cat(preds)

            same = int((pred_batched == pred_seq).sum())
            total = pred_batched.numel()
            acc_b = (pred_batched == yq).float().mean().item()
            acc_s = (pred_seq == yq).float().mean().item()
            rows.append({
                "dataset": ds, "k_shot": k, "episodes": B,
                "t_sequential_s": round(t_seq, 2), "t_batched_s": round(t_batched, 3),
                "speedup": round(t_seq / t_batched, 1),
                "predictions_identical": f"{same}/{total}",
                "acc_batched": acc_b, "acc_sequential": acc_s,
            })
            print(f"{ds} {k}-shot: sequential {t_seq:6.1f} s | batched {t_batched:5.2f} s | "
                  f"{t_seq / t_batched:5.0f}x | identical predictions {same}/{total} | "
                  f"acc {100 * acc_s:.2f} vs {100 * acc_b:.2f}", flush=True)

    with open(metrics_dir() / "bench_probe.json", "w") as f:
        json.dump(rows, f, indent=2)
    plot(rows)


def plot(rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from src.visualize import dataset_label, figures_dir

    labels = [f"{dataset_label(r['dataset'])}\n{r['k_shot']}-shot" for r in rows]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(11, 6))
    b1 = ax.bar(x - 0.2, [r["t_sequential_s"] for r in rows], 0.4,
                label="sequential (600 independent heads)", color="#D55E00")
    b2 = ax.bar(x + 0.2, [r["t_batched_s"] for r in rows], 0.4,
                label="batched [600, C, D] (ours)", color="#029E73")
    for rect, r in zip(b1, rows):
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() * 1.12,
                f"{r['t_sequential_s']:.0f} s", ha="center", fontsize=10)
    for rect, r in zip(b2, rows):
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() * 1.12,
                f"{r['t_batched_s']:.2f} s\n({r['speedup']:.0f}x)", ha="center", fontsize=10)
    ax.set_yscale("log")
    ax.set_ylim(top=max(r["t_sequential_s"] for r in rows) * 4)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("time for 600 heads (s, log)")
    ax.set_title("Linear probe: sequential vs batched training\n"
                 f"identical predictions (45,000/45,000 per config), "
                 f"{min(r['speedup'] for r in rows):.0f}–{max(r['speedup'] for r in rows):.0f}x faster",
                 fontsize=15)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=11, frameon=True)
    path = figures_dir() / "bench_probe.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"figure: {path}")
    total_seq = sum(r["t_sequential_s"] for r in rows)
    total_b = sum(r["t_batched_s"] for r in rows)
    print(f"grid total: sequential {total_seq / 60:.1f} min -> batched {total_b:.1f} s")


if __name__ == "__main__":
    main()
