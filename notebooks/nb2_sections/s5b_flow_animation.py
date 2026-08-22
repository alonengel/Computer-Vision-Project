CELLS = [
    ("markdown", """
### Animated flow trajectories — one example at a time, zoomed

The static trajectory figures above show start, end, and intermediate Euler states at once; the animations below play the **same real transport** through time for the representative setting with the clearest Stage-2 improvement: **FGVC-Aircraft / DINOv2, image prototypes, K = 10, subset seed 0, T = 12**, Standard FM vs Rolled-out FM side by side. (This is deliberately the success case; the failure mode — DTD, where the same transport *loses* accuracy by merging entangled texture classes — is visible in the static DTD trajectory figure above and discussed in §6.) Each of the four representative examples gets its **own animation, zoomed to its trajectory**, and at every iteration an **arrow shows the direction of the current Euler step** — the learned velocity $\\tfrac{1}{T}v_\\theta(\\hat{z}_k, k/T)$, projected into the plot plane.

Everything is loaded from the saved Stage-2 artifacts — the trained velocity networks, the cached test features, and the training prototypes of this exact setting. The four examples, class colours, and the jointly-fitted PCA plane are **identical to the static trajectory figure by construction** (the same helper functions produce them), and the cell verifies programmatically that: each trajectory has exactly $T{+}1 = 13$ states; the first state equals the normalized feature the Stage-2 inference pipeline starts from; the final state matches the transported features used elsewhere in this notebook; both models start from identical features and share identical prototypes; every projected state comes from one fitted PCA; and no gradients flow during generation. Nothing is synthetic; the only new artifacts are the exported GIFs, which are embedded below as self-contained images (no JavaScript player needed).
"""),
    ("code", """
import base64

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
from IPython.display import HTML

sys.path.insert(0, str(REPO / "scripts"))
from make_figures_stage2 import REP_T, _joint_pca, _rep_setting
from src.visualize import class_palette, dataset_label, encoder_label

ANIM_DS, ANIM_ENC, ANIM_TARGET = "fgvc_aircraft", "dinov2_vits14", "image_prototype"

# Same selection code path as the static figures: classes, examples, features,
# transported sets, prototypes, and the two trained models of this setting.
(classes, idx, X, Xstd, Xroll, protos, y, sel_names,
 std_head, roll_head) = _rep_setting(ANIM_DS, ANIM_ENC, ANIM_TARGET)

# Same 4 examples as the static trajectory figure (same rng, same rule).
rng = np.random.default_rng(0)
picked_classes = rng.choice(len(classes), size=4, replace=False)
ex = [int(np.flatnonzero(y == c)[0]) for c in picked_classes]

# Real Euler states from the saved networks (transport() runs under no_grad).
_, traj_std = std_head.transport(X[ex], REP_T, return_traj=True)   # [T+1, 4, 384]
_, traj_roll = roll_head.transport(X[ex], REP_T, return_traj=True)

# ---- programmatic verification (all against the real pipeline) ----
z0 = F.normalize(X[ex].float(), dim=-1)
assert traj_std.shape[0] == REP_T + 1 and traj_roll.shape[0] == REP_T + 1
assert torch.allclose(traj_std[0], z0) and torch.allclose(traj_roll[0], z0)
assert torch.allclose(traj_std[0], traj_roll[0])
# final states match the transported features the feature-comparison figure uses
assert torch.allclose(traj_std[-1], Xstd[ex], atol=1e-5)
assert torch.allclose(traj_roll[-1], Xroll[ex], atol=1e-5)
# both models were trained toward identical prototypes; each example's target is
# its true class's prototype
assert torch.allclose(std_head.prototypes.cpu(), roll_head.prototypes.cpu(), atol=1e-6)
assert torch.allclose(protos, std_head.prototypes.cpu()[classes], atol=1e-6)
assert not traj_std.requires_grad and not traj_roll.requires_grad  # no gradients

# One PCA, fitted exactly as in the static figures (jointly to all compared
# sets + prototypes), transforms background, prototypes, and every frame.
pca, (bg_xy, _, _, proto_xy) = _joint_pca([X.numpy(), Xstd.numpy(), Xroll.numpy(),
                                           protos.numpy()])
txy_std = np.stack([pca.transform(traj_std[:, i].numpy()) for i in range(len(ex))])
txy_roll = np.stack([pca.transform(traj_roll[:, i].numpy()) for i in range(len(ex))])

print(f"[OK] {len(ex)} examples loaded (classes: "
      f"{', '.join(sel_names[int(y[e])] for e in ex)})")
print(f"[OK] {REP_T + 1} Euler states per trajectory")
print("[OK] shared PCA transform (fitted once, jointly, as in the static figures)")
print("[OK] first states = normalized pipeline inputs; final states match the "
      "transported features of the feature-comparison figure")
print("[OK] no gradients during trajectory generation")

palette = class_palette(len(sel_names))  # shared with every feature/traj figure
LEGEND = [
    Line2D([], [], marker="o", ls="", ms=10, mfc="none", mec="#555555", mew=2,
           label=r"original feature $\\hat{z}_0$"),
    Line2D([], [], marker="o", ls="", ms=10, color="#555555", mec="black",
           label=r"current state $\\hat{z}_k$"),
    Line2D([], [], color="#555555", lw=1.8, label="trajectory so far"),
    Line2D([], [], marker=r"$\\rightarrow$", ls="", ms=14, color="black",
           label=r"step direction $\\frac{1}{T}v_\\theta(\\hat{z}_k, k/T)$"),
    Line2D([], [], marker="X", ls="", ms=11, color="#555555", mec="black",
           label=r"final state $\\hat{z}_T$"),
    Line2D([], [], marker="*", ls="", ms=15, mfc="white", mec="black",
           label="class prototype")]


def animate_example(i):
    \"\"\"One zoomed two-panel animation (Standard | Rolled-out) for example i;
    returns the saved GIF path. Axis limits fixed to this example's trajectories
    + its prototype, identical in both panels and across all frames.\"\"\"
    e = ex[i]
    cls = int(y[e])
    c = palette[cls]
    pts = np.concatenate([txy_std[i], txy_roll[i], proto_xy[cls:cls + 1]])
    lo, hi = pts.min(0), pts.max(0)
    pad = 0.20 * max(hi - lo)
    xlim, ylim = (lo[0] - pad, hi[0] + pad), (lo[1] - pad, hi[1] + pad)

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 6.0))
    fig.subplots_adjust(top=0.76, bottom=0.16, wspace=0.06)
    panels = []
    for ax, title, txy in ((axes[0], f"Standard FM — T = {REP_T}", txy_std[i]),
                           (axes[1], f"Rolled-out FM — T = {REP_T}", txy_roll[i])):
        for j in sorted(np.unique(y)):   # background clipped by the zoom window
            m = y == j
            ax.scatter(bg_xy[m, 0], bg_xy[m, 1], s=16, alpha=0.15, color=palette[j])
            ax.scatter(proto_xy[j, 0], proto_xy[j, 1], marker="*", s=560,
                       color=palette[j], edgecolors="black", linewidths=1.4, zorder=5)
        ax.scatter(*txy[0], s=120, facecolors="none", edgecolors=c,
                   linewidths=2.2, zorder=6)                     # hollow z0
        trail, = ax.plot([], [], color=c, lw=2.0, alpha=0.95, zorder=4)
        cur, = ax.plot([], [], marker="o", ms=12, color=c, mec="black", mew=1.3,
                       zorder=7)
        arrow = ax.quiver([txy[0, 0]], [txy[0, 1]], [0.0], [0.0], angles="xy",
                          scale_units="xy", scale=1.0, color="black",
                          width=0.008, zorder=8)
        fin = ax.scatter([], [], marker="X", s=150, color=c, edgecolors="black",
                         linewidths=1.3, zorder=9, visible=False)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        ax.set_title(title, fontsize=12.5)
        panels.append({"trail": trail, "cur": cur, "arrow": arrow, "fin": fin,
                       "txy": txy})

    step_txt = fig.text(0.5, 0.845, "", ha="center", fontsize=13)
    fig.legend(handles=LEGEND, loc="lower center", bbox_to_anchor=(0.5, 0.005),
               ncol=6, fontsize=9, frameon=True)
    fig.suptitle(f"{dataset_label(ANIM_DS)} — {encoder_label(ANIM_ENC, short=True)}: "
                 f"one test example of class \\u201c{sel_names[cls]}\\u201d, zoomed"
                 f"\\n(real saved models, K = 10 subset seed 0; joint-PCA plane of "
                 f"the static figures; arrow = projected direction of the current "
                 f"Euler step)", y=0.97, fontsize=13)

    def draw(k):
        for p in panels:
            xy = p["txy"]
            p["trail"].set_data(xy[:k + 1, 0], xy[:k + 1, 1])
            p["cur"].set_data([xy[k, 0]], [xy[k, 1]])
            if k < REP_T:                 # the step about to be taken
                p["arrow"].set_offsets(xy[k:k + 1])
                p["arrow"].set_UVC([xy[k + 1, 0] - xy[k, 0]],
                                   [xy[k + 1, 1] - xy[k, 1]])
                p["arrow"].set_visible(True)
            else:
                p["arrow"].set_visible(False)
            p["fin"].set_visible(k == REP_T)
            if k == REP_T:
                p["fin"].set_offsets(xy[-1:])
        step_txt.set_text(f"Euler step {k} / {REP_T},  t = {k / REP_T:.3f}")
        return []

    frames = [0] * 3 + list(range(1, REP_T)) + [REP_T] * 4   # hold first & last
    anim = FuncAnimation(fig, draw, frames=frames, interval=600, blit=False)
    path = (REPO / "results" / "figures" /
            f"stage2_flow_anim_{ANIM_DS}_{ANIM_ENC}_T{REP_T}_ex{i}.gif")
    try:
        anim.save(path, writer=PillowWriter(fps=2))
    except Exception as err:
        print(f"[WARN] GIF export failed for example {i} ({err}); "
              f"falling back to the inline JS player")
        display(HTML(anim.to_jshtml()))
        path = None
    plt.close(fig)
    return path, cls


for i in range(len(ex)):
    gif, cls = animate_example(i)
    if gif is not None:
        print(f"[OK] example {i}: class \\u201c{sel_names[cls]}\\u201d -> "
              f"{gif.relative_to(REPO)}")
        # embed the GIF bytes directly, so the animation plays in any notebook
        # viewer (VS Code included) without JavaScript and without path issues
        b64 = base64.b64encode(gif.read_bytes()).decode("ascii")
        display(HTML(f'<img src="data:image/gif;base64,{b64}" width="920"/>'))
"""),
    ("markdown", """
**Reading the animations.** The markers follow **real states produced by the learned velocity networks** — nothing is a hand-drawn interpolation. In each pair of panels both models start from the identical normalized test feature; the difference is purely the training objective: **Standard FM** was supervised on velocities at random points along the ideal straight interpolation path (which is why its arrows stay nearly parallel and its path runs almost straight), while **Rolled-out FM** was trained through its own self-generated $T$-step sequence with a loss only on the final transported state — its velocity field is free to curve, and the arrows visibly change direction between steps.

Two caveats for honest reading: the actual transport happens in the original **384-dimensional DINOv2 feature space**; the zoomed PCA view is a qualitative 2-D projection of that motion, so projected path length, curvature, arrow direction, and distances must not be read as high-dimensional measurements (the drawn arrow is the projection of the true step $\\frac{1}{T}v_\\theta(\\hat{z}_k, k/T)$ into the plane). What the animations *are* meant to convey is mechanism: how $T$ small Euler updates $\\hat{z}_{k+1} = \\hat{z}_k + \\frac{1}{T}v_\\theta(\\hat{z}_k, k/T)$ accumulate into the final transported representation that the cosine/prototype rule then classifies.
"""),
]
