CELLS = [
    ("markdown", """
### Optional exploration ‡ — reverse feature-space integration from the prototypes

The specification encourages exploring the learned flow **in the reverse direction**, starting from the class prototypes, and comparing samples and prototypes at intermediate flow times. We integrate the learned field backward from $t = 1$ to $t = 0$ with explicit Euler:

$$\\hat{z}_{k-1} = \\hat{z}_k - \\tfrac{1}{T}\\, v_\\theta\\!\\left(\\hat{z}_k, \\tfrac{k}{T}\\right), \\qquad k = T, T{-}1, \\dots, 1,$$

starting from each class prototype as $\\hat{z}_T$ (first velocity evaluation at $t = 1$, as canonical reverse integration requires). This is **approximate backward integration of the learned continuous field** — not the algebraic inverse of the discrete forward-Euler map — and it is fully deterministic: this FM head is not a generative model, so one prototype yields exactly one reverse trajectory (no noise is injected). We run it on **both branches** at the representative setting ($K = 10$, subset seed 0, $T = 12$, both training modes): the spec branch (FGVC-Aircraft / DINOv2, image prototypes) and the ‡ CLIP branch (FGVC-Aircraft / CLIP RN50, **text** prototypes), where backward integration additionally has to re-cross the image–text modality gap. Same classes, examples, colours and **the same jointly fitted PCA plane** as the forward figures (fitted once, never refit). No model was retrained and no published result changes.

**Why the reverse arrows do not retrace the forward arrows.** The reverse paths are *not* the forward paths played backwards — and should not be: (i) reverse integration starts *exactly at the prototype*, whereas forward trajectories start at individual test features and end only *near* it; (ii) reverse Euler evaluates the field at different states and times than the forward steps did; (iii) near $t = 1$ the standard-FM field is a *conditional average* over the whole class (§5), so its backward path follows the class-mean corridor rather than any single sample's path. To make the comparison visible rather than implicit, each panel also draws the **forward class-centroid path** (dashed, faint) in the same plane.
"""),
    ("code", """
from src.visualize import head_label

for name in ("stage2_reverse_flow_fgvc_aircraft_dinov2_vits14_image_prototype_T12.png",
             "stage2_reverse_flow_fgvc_aircraft_clip_rn50_clip_text_T12.png"):
    display(Image(str(REPO / "results" / "figures" / name), width=980))
"""),
    ("markdown", """
**Animated backward integration.** The static panels above show the whole reverse path at once; the two animations below play it **step by step** — all ten class prototypes travelling backward together, one Euler step per frame, with an arrow at each state showing the direction of the step about to be taken ($-\\tfrac{1}{T}v_\\theta$ projected into the plane) and the clock running $t = 1 \\to 0$. One animation per branch: the spec branch (image prototypes) and the ‡ CLIP branch, where the progression makes the backward crossing of the image–text modality gap directly visible — the prototypes start in their own isolated region and march into the image-feature cloud. Same trained checkpoints, same joint-PCA plane and the same verified trajectories as the static figure; GIFs are embedded as images so they play without JavaScript.

Both panels share axis limits set by the **data region and the Standard-FM paths**; rolled-out paths that escape the data region therefore run off-frame, which is the failure mode itself (including them would shrink everything else to an unreadable blob — the static figure above shows their full extent).
"""),
    ("code", """
import base64

import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython.display import HTML
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D

sys.path.insert(0, str(REPO / "scripts"))
from make_figures_stage2 import REP_T, _joint_pca, _rep_setting
from src.visualize import class_palette, dataset_label, encoder_label

def animate_reverse(ds, enc, target, subtitle):
    classes, idx, X, Xstd, Xroll, protos, y, sel_names, std, roll = \\
        _rep_setting(ds, enc, target)
    pca, (xy0, _, _, pxy) = _joint_pca([X.numpy(), Xstd.numpy(), Xroll.numpy(),
                                        protos.numpy()])
    series = []
    for head, label in ((std, "Standard FM"), (roll, "Rolled-out FM")):
        _, rtraj, times = head.reverse_transport(protos, REP_T, return_traj=True)
        assert torch.allclose(rtraj[0], protos.float()), "reverse must start at the prototype"
        assert float(times[0]) == 1.0 and float(times[-1]) == 0.0
        series.append((label, np.stack([pca.transform(rtraj[:, j].numpy())
                                        for j in range(len(classes))])))  # [C, T+1, 2]

    palette = class_palette(len(sel_names))
    # Limits are set by the data region (features + prototypes) and the
    # Standard-FM reverse paths, shared by both panels. Rolled-out paths that
    # escape the data region run off-frame — that divergence is the finding,
    # and including it would shrink everything else to an unreadable blob.
    allxy = np.concatenate([xy0, pxy, series[0][1].reshape(-1, 2)])
    pad = 0.08 * (allxy.max(0) - allxy.min(0))
    xlim = (allxy[:, 0].min() - pad[0], allxy[:, 0].max() + pad[0])
    ylim = (allxy[:, 1].min() - pad[1], allxy[:, 1].max() + pad[1])

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.4))
    fig.subplots_adjust(top=0.78, bottom=0.14, wspace=0.06)
    panels = []
    for ax, (label, S) in zip(axes, series):
        for j in range(len(classes)):
            m = y == j
            ax.scatter(xy0[m, 0], xy0[m, 1], s=10, alpha=0.13, color=palette[j])
            ax.scatter(pxy[j, 0], pxy[j, 1], marker="*", s=380, color=palette[j],
                       edgecolors="black", linewidths=1.2, zorder=5)
        arts = []
        for j in range(len(classes)):
            c = palette[j]
            trail, = ax.plot([], [], color=c, lw=1.8, alpha=0.95, zorder=4)
            cur, = ax.plot([], [], marker="o", ms=9, color=c, mec="black", mew=1.1, zorder=7)
            arrow = ax.quiver([S[j, 0, 0]], [S[j, 0, 1]], [0.0], [0.0], angles="xy",
                              scale_units="xy", scale=1.0, color="black",
                              width=0.006, zorder=8)
            fin = ax.scatter([], [], marker="s", s=110, color=c, edgecolors="black",
                             linewidths=1.1, zorder=9, visible=False)
            arts.append({"trail": trail, "cur": cur, "arrow": arrow, "fin": fin,
                         "xy": S[j]})
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        ax.set_title(f"{label} (T = {REP_T})", fontsize=12.5)
        panels.append(arts)

    step_txt = fig.text(0.5, 0.855, "", ha="center", fontsize=13)
    fig.legend(handles=[
        Line2D([], [], marker="*", ls="", ms=15, mfc="white", mec="black",
               label="class prototype (start, $t = 1$)"),
        Line2D([], [], marker="o", ls="", ms=9, color="#555555", mec="black",
               label="current reverse state"),
        Line2D([], [], color="#555555", lw=1.8, label="path travelled so far"),
        Line2D([], [], marker=r"$\\rightarrow$", ls="", ms=13, color="black",
               label=r"step direction $-\\frac{1}{T}v_\\theta$"),
        Line2D([], [], marker="s", ls="", ms=10, color="#555555", mec="black",
               label="reverse endpoint ($t = 0$)")],
        loc="lower center", bbox_to_anchor=(0.5, 0.005), ncol=5, fontsize=9.5,
        frameon=True)
    fig.suptitle(f"{dataset_label(ds)} — {encoder_label(enc, short=True)}: backward "
                 f"integration from the class prototypes, step by step\\n{subtitle}",
                 y=0.97, fontsize=13)

    def draw(k):
        for arts in panels:
            for a in arts:
                xy = a["xy"]
                a["trail"].set_data(xy[:k + 1, 0], xy[:k + 1, 1])
                a["cur"].set_data([xy[k, 0]], [xy[k, 1]])
                if k < REP_T:
                    a["arrow"].set_offsets(xy[k:k + 1])
                    a["arrow"].set_UVC([xy[k + 1, 0] - xy[k, 0]],
                                       [xy[k + 1, 1] - xy[k, 1]])
                    a["arrow"].set_visible(True)
                else:
                    a["arrow"].set_visible(False)
                a["fin"].set_visible(k == REP_T)
                if k == REP_T:
                    a["fin"].set_offsets(xy[-1:])
        step_txt.set_text(f"reverse step {k} / {REP_T},   t = {(REP_T - k) / REP_T:.3f}")
        return []

    frames = [0] * 3 + list(range(1, REP_T)) + [REP_T] * 4
    anim = FuncAnimation(fig, draw, frames=frames, interval=600, blit=False)
    path = (REPO / "results" / "figures" /
            f"stage2_reverse_anim_{ds}_{enc}_{target}_T{REP_T}.gif")
    anim.save(path, writer=PillowWriter(fps=2))
    plt.close(fig)
    print(f"[OK] {path.name}")
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    display(HTML(f'<img src="data:image/gif;base64,{b64}" width="940"/>'))

animate_reverse("fgvc_aircraft", "dinov2_vits14", "image_prototype",
                "spec branch: image-derived prototypes (K = 10, subset seed 0)")
animate_reverse("fgvc_aircraft", "clip_rn50", "clip_text",
                "‡ CLIP branch: text prototypes — watch the backward crossing "
                "of the image–text modality gap")
"""),
    ("markdown", """
**Samples vs prototypes at matching intermediate times.** For each selected class and every Euler time $t = k/T$: the centroid of that class's selected test samples along the **forward** flow, compared against the **reverse** trajectory that started at that class's prototype — cosine similarity ($L_2$ secondary), computed in the full feature space (384-d DINOv2 / 1024-d CLIP), *not* in the PCA plane. The curves show **mean ± sample std across the ten selected classes**; per-class raw values are in `results/metrics/stage2_reverse_intermediate_*.csv`.
"""),
    ("code", """
for name in ("stage2_reverse_intermediate_fgvc_aircraft_dinov2_vits14_image_prototype_T12.png",
             "stage2_reverse_intermediate_fgvc_aircraft_clip_rn50_clip_text_T12.png"):
    display(Image(str(REPO / "results" / "figures" / name), width=920))

rows = []
for tag, label in (("fgvc_aircraft_dinov2_vits14_image_prototype_T12", "DINOv2 → image prototypes"),
                   ("fgvc_aircraft_clip_rn50_clip_text_T12", "CLIP RN50 → text prototypes ‡")):
    d = pd.read_csv(REPO / "results" / "metrics" / f"stage2_reverse_intermediate_{tag}.csv")
    e = d[d["t"] == 0].groupby("head")[["cosine", "l2"]].agg(["mean", "std"])
    for head in e.index:
        rows.append({"setting": label, "head": head_label(head),
                     "cosine mean": round(e.loc[head, ("cosine", "mean")], 3),
                     "cosine std": round(e.loc[head, ("cosine", "std")], 3),
                     "L2 mean": round(e.loc[head, ("l2", "mean")], 3)})
print("At t = 0 — reverse endpoint vs the class's original-feature centroid "
      "(mean ± sample std over the 10 selected classes):")
display(pd.DataFrame(rows))
"""),
    ("markdown", """
**Reverse-flow Top-1 (%) — what exactly these percentages are.** Each number below is **top-1 accuracy on the complete official test split** (all 3,333 FGVC-Aircraft test images, all 100 classes), from a **single deterministic inference pass** on the representative $K = 10$, seed-0 model — so, unlike the accuracy tables in §3, these carry **no ± spread: they are not averaged over runs or seeds** (the only averaging anywhere in this section is the cosine/$L_2$ table above, which averages over the ten *selected classes*). Four columns per training mode:

| Column | What is measured |
|---|---|
| **Stage-1 prototype baseline** | untouched features classified against the prototypes — the reference the FM layer must beat |
| **Forward FM (published)** | the §3 grid number for this exact cell, recomputed here; the generator asserts it still matches `runs_stage2.csv` |
| **Round-trip** | transport every test feature forward to $t = 1$, integrate it back to $t = 0$, classify the reconstruction with the Stage-1 rule — *how much class information survives a full forward-then-backward traversal* |
| **Reverse-endpoint classifier** | integrate **all 100 class prototypes** backward to $t = 0$ and use those endpoints as class representatives for the **original** features — *whether the backward images of the prototypes are still usable class descriptions* |

Both reverse read-outs are label-free inference on already-trained models; no training, no selection, nothing published changes.
"""),
    ("code", """
frames = []
for tag, label in (("fgvc_aircraft_dinov2_vits14_image_prototype_T12", "DINOv2 → image prototypes"),
                   ("fgvc_aircraft_clip_rn50_clip_text_T12", "CLIP RN50 → text prototypes ‡")):
    d = pd.read_csv(REPO / "results" / "metrics" / f"stage2_reverse_top1_{tag}.csv")
    d.insert(0, "setting", label)
    d["head"] = d["head"].map(head_label)
    frames.append(d)
acc = pd.concat(frames, ignore_index=True)
for c in acc.columns[2:]:
    acc[c] = (100 * acc[c]).round(2)
acc.columns = ["setting", "head", "Stage-1 prototype baseline", "Forward FM (published)",
               "Round-trip (fwd → rev)", "Reverse-endpoint classifier"]
display(acc)
print("Top-1 (%) on the complete official test split; single deterministic runs "
      "(K = 10, seed 0, T = 12) — no averaging over runs or seeds.")
"""),
    ("markdown", """
**What the actual results show.**

*Spec branch (DINOv2 → image prototypes).* **Standard FM** integrates backward gracefully: reverse paths stay inside the data region and hug the forward class-centroid paths; the reverse state tracks the forward same-class centroid at every intermediate time (cosine 0.85 ± 0.06 at $t = 0$); the round trip *preserves* class information (31.17% vs the 29.13% untouched-feature baseline); and the backward images of the prototypes remain serviceable class representatives (25.68%). This is what per-path velocity supervision buys — the field carries meaningful values at every $(z, t)$, so it can be traversed in either direction. **Rolled-out FM** collapses in reverse: paths shoot outside the data, similarity falls to 0.24 ± 0.20, the round trip *loses* information (26.01%, below baseline) and the reverse-endpoint classifier drops to 11.07%. Its endpoint-only objective never constrains the field along the path — the composed forward map is accurate, but the field it is built from is not individually meaningful.

*CLIP branch ‡ (text prototypes).* The same split appears, more extremely, plus one genuinely striking result. Standard FM's reverse trajectories **cross the modality gap backward**: starting at the isolated text prototypes they travel into the image-feature cloud and land essentially on the class centroids (cosine **0.973 ± 0.006**, the tightest agreement anywhere in this section). And the reverse-endpoint classifier reaches **26.40%** — *above* the forward FM (21.99%) and far above zero-shot CLIP (17.04%). Reading it carefully: backward integration turns each text prototype into an **image-space** class representative, and classifying image features against those is markedly better than classifying them against the text prototypes directly — a concrete, measured demonstration that the learned flow encodes the modality gap in a usable way. (It is not a "better classifier" claim for the pipeline: the flow was trained with $K = 10$ labels per class, so this belongs beside the other supervised ‡ numbers, not beside zero-shot.) Rolled-out FM again fails in reverse — cosine −0.06 ± 0.57, round trip 6.27%, reverse endpoints 3.45%. Note also that the CLIP round trip is slightly *below* baseline even for standard FM (16.89% vs 17.04%): traversing there and back across the modality gap costs a little, while the one-way backward map is what carries the value.

Across both branches, the reverse direction is an independent, training-free confirmation of the §6 mechanism story: **standard FM learns a field that is meaningful everywhere; rolled-out FM learns only a good composed endpoint map.**

**Limitations (read before over-interpreting):** the network was trained in the **forward** direction only; rolled-out training never directly supervised any reverse trajectory; canonical reverse integration requires evaluating at $t = 1$, a time the forward Euler rollout never evaluates (it stops at $(T{-}1)/T$); reverse trajectories need **not** recover real samples — this is reverse *feature-space* exploration, not image generation; the learned map is deterministic and may be many-to-one, so nothing guarantees bijectivity; the Top-1 read-outs are single deterministic runs on one representative setting (no seed spread); and the trajectory plots are qualitative 2-D PCA projections (all metrics are computed in the full feature space).
"""),
]
