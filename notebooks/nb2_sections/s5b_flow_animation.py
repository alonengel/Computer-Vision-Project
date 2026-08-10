CELLS = [
    ("markdown", """
### Animated flow trajectories — the learned transport, step by step

The static trajectory figures above show start, end, and intermediate Euler states at once; the animation below plays the **same real transport** through time for the representative setting with the clearest Stage-2 improvement: **FGVC-Aircraft / DINOv2, image prototypes, K = 10, subset seed 0, T = 12**, Standard FM vs Rolled-out FM side by side.

Everything is loaded from the saved Stage-2 artifacts — the trained velocity networks, the cached test features, and the training prototypes of this exact setting. The four examples, class colours, and the jointly-fitted PCA plane are **identical to the static trajectory figure by construction** (the same helper functions produce them), and the cell verifies programmatically that: each trajectory has exactly $T{+}1 = 13$ states; the first state equals the normalized feature the Stage-2 inference pipeline starts from; the final state matches the transported features used elsewhere in this notebook; both models start from identical features and share identical prototypes; every projected state comes from one fitted PCA; and no gradients flow during generation. Nothing is synthetic and no existing artifact is overwritten (the only new file is the exported GIF).
"""),
    ("code", """
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import torch
import torch.nn.functional as F
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
from IPython.display import HTML

sys.path.insert(0, str(REPO / "scripts"))
from make_figures_stage2 import REP_T, _joint_pca, _rep_setting
from src.visualize import dataset_label, encoder_label

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

# ---- animation ----
matplotlib.rcParams["animation.embed_limit"] = 60  # MB, for the inline JS player
palette = sns.color_palette("colorblind", n_colors=max(10, len(sel_names)))
all_xy = np.concatenate([bg_xy, proto_xy, txy_std.reshape(-1, 2), txy_roll.reshape(-1, 2)])
pad = 0.06 * (all_xy.max(0) - all_xy.min(0))
xlim = (all_xy[:, 0].min() - pad[0], all_xy[:, 0].max() + pad[0])
ylim = (all_xy[:, 1].min() - pad[1], all_xy[:, 1].max() + pad[1])

fig, axes = plt.subplots(1, 2, figsize=(13.6, 7.4))
# animations are rendered without bbox_inches="tight", so every fixed text
# element must live inside the canvas
fig.subplots_adjust(top=0.78, bottom=0.15, wspace=0.06)
panels = []
for ax, title, txy in ((axes[0], f"Standard FM — T = {REP_T}", txy_std),
                       (axes[1], f"Rolled-out FM — T = {REP_T}", txy_roll)):
    for j in sorted(np.unique(y)):
        m = y == j
        ax.scatter(bg_xy[m, 0], bg_xy[m, 1], s=12, alpha=0.15, color=palette[j])
        ax.scatter(proto_xy[j, 0], proto_xy[j, 1], marker="*", s=430,
                   color=palette[j], edgecolors="black", linewidths=1.3, zorder=5)
    arts = []
    for i, e in enumerate(ex):
        c = palette[int(y[e])]
        ax.scatter(*txy[i, 0], s=95, facecolors="none", edgecolors=c,
                   linewidths=2.0, zorder=6)                      # ẑ₀, hollow
        guide, = ax.plot([], [], linestyle="--", color=c, lw=0.8, alpha=0.35, zorder=3)
        trail, = ax.plot([], [], color=c, lw=1.8, alpha=0.95, zorder=4)
        cur, = ax.plot([], [], marker="o", ms=11, color=c, mec="black", mew=1.2,
                       zorder=7)
        fin = ax.scatter([], [], marker="X", s=130, color=c, edgecolors="black",
                         linewidths=1.2, zorder=8, visible=False)
        arts.append({"trail": trail, "cur": cur, "fin": fin, "guide": guide,
                     "txy": txy[i], "pxy": proto_xy[int(y[e])]})
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    ax.set_title(title, fontsize=13)
    panels.append(arts)

step_txt = fig.text(0.5, 0.845, "", ha="center", fontsize=13)
fig.legend(handles=[
    Line2D([], [], marker="o", ls="", ms=10, mfc="none", mec="#555555", mew=2,
           label=r"original feature $\\hat{z}_0$"),
    Line2D([], [], marker="o", ls="", ms=10, color="#555555", mec="black",
           label=r"current state $\\hat{z}_k$"),
    Line2D([], [], color="#555555", lw=1.8, label="trajectory so far"),
    Line2D([], [], marker="X", ls="", ms=11, color="#555555", mec="black",
           label=r"final state $\\hat{z}_T$"),
    Line2D([], [], marker="*", ls="", ms=15, mfc="white", mec="black",
           label="class prototype"),
    Line2D([], [], color="#555555", lw=0.8, ls="--", alpha=0.5,
           label="guide to prototype")],
    loc="lower center", bbox_to_anchor=(0.5, 0.005), ncol=6, fontsize=9.5, frameon=True)
fig.suptitle(f"{dataset_label(ANIM_DS)} — {encoder_label(ANIM_ENC, short=True)}: the "
             f"learned FM transport, one Euler step at a time\\n(real saved models, "
             f"K = 10 subset seed 0; joint-PCA plane of the static figures)",
             y=0.97, fontsize=13.5)

def draw(k):
    for arts in panels:
        for a in arts:
            xy = a["txy"]
            a["trail"].set_data(xy[:k + 1, 0], xy[:k + 1, 1])
            a["cur"].set_data([xy[k, 0]], [xy[k, 1]])
            a["guide"].set_data([xy[k, 0], a["pxy"][0]], [xy[k, 1], a["pxy"][1]])
            a["fin"].set_visible(k == REP_T)
            if k == REP_T:
                a["fin"].set_offsets(xy[-1:])
    step_txt.set_text(f"Euler step {k} / {REP_T},  t = {k / REP_T:.3f}")
    return []

frames = [0] * 3 + list(range(1, REP_T)) + [REP_T] * 4   # hold first & last
anim = FuncAnimation(fig, draw, frames=frames, interval=600, blit=False)

gif_path = REPO / "results" / "figures" / "stage2_flow_animation_fgvc_dinov2_T12.gif"
try:
    anim.save(gif_path, writer=PillowWriter(fps=2))
    print(f"[OK] GIF exported: {gif_path.relative_to(REPO)}")
except Exception as err:  # inline player must still work without the GIF
    print(f"[WARN] GIF export skipped ({err}); inline animation still available")

plt.close(fig)
HTML(anim.to_jshtml())
"""),
    ("markdown", """
**Reading the animation.** The markers follow **real states produced by the learned velocity networks** — nothing is a hand-drawn interpolation. Both panels start from the identical normalized test features; the difference between them is purely the training objective: **Standard FM** was supervised on velocities at random points along the ideal straight interpolation path (which is why its paths run nearly straight at constant speed), while **Rolled-out FM** was trained through its own self-generated $T$-step sequence with a loss only on the final transported state — its field is free to curve, and does.

Two caveats for honest reading: the actual transport happens in the original **384-dimensional DINOv2 feature space**; the PCA plane is a qualitative 2-D projection of that motion, so projected path length, curvature, and distances must not be read as high-dimensional measurements. What the animation *is* meant to convey is mechanism: how $T$ small Euler updates $\\hat{z}_{k+1} = \\hat{z}_k + \\frac{1}{T}v_\\theta(\\hat{z}_k, k/T)$ accumulate into the final transported representation that the cosine/prototype rule then classifies.
"""),
]
