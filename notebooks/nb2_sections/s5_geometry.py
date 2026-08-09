CELLS = [
    ("markdown", """
## 5 · How the learned transformation changes the feature-space geometry

**Feature-space comparison** (spec item 3). For each setting: the same ten classes, test examples, and class colours as the Stage-1 visualizations; three views — original features, after standard FM, after rolled-out FM — plus the prototypes the models were actually trained toward (the 10-shot seed-0 prototypes, not the full-split ones). **One PCA is fitted jointly to all three feature sets and the prototypes**, so the three panels share a single plane and positions are directly comparable across panels (spec requirement). Representative models: $K = 10$, subset seed 0, $T = 12$, fixed a priori. As in Stage 1, 2-D projections are qualitative.
"""),
    ("code", """
for name in ("stage2_features_dtd_resnet18_image_prototype.png",
             "stage2_features_fgvc_aircraft_resnet18_image_prototype.png",
             "stage2_features_fgvc_aircraft_dinov2_vits14_image_prototype.png",
             "stage2_features_dtd_clip_rn50_clip_text.png",
             "stage2_features_fgvc_aircraft_clip_rn50_clip_text.png"):
    p = REPO / "results" / "figures" / name
    if p.exists():
        display(Image(str(p), width=980))
"""),
    ("markdown", """
**Flow trajectories** (spec item 4). A small number of representative test examples per setting, with every intermediate Euler state $\\hat{z}_0 \\to \\hat{z}_1 \\to \\dots \\to \\hat{z}_T$ drawn in the **same joint PCA plane** as the feature-comparison figure (background: original test features; stars: prototypes; ○ start, ✕ end). PCA is used because projected straight-line segments remain straight-line segments, so the trajectory geometry is interpretable — a t-SNE embedding would not preserve that.
"""),
    ("code", """
for name in ("stage2_traj_dtd_resnet18_image_prototype.png",
             "stage2_traj_fgvc_aircraft_resnet18_image_prototype.png",
             "stage2_traj_fgvc_aircraft_dinov2_vits14_image_prototype.png",
             "stage2_traj_dtd_clip_rn50_clip_text.png",
             "stage2_traj_fgvc_aircraft_clip_rn50_clip_text.png"):
    p = REPO / "results" / "figures" / name
    if p.exists():
        display(Image(str(p), width=980))
"""),
    ("markdown", """
**What the geometry says.** Two structural facts help read these figures:

1. **The trained flow is a contraction toward class regions, not a permutation-free identity.** Both objectives reward moving every training feature of class $c$ onto the single point $p_c$; a perfectly successful transport would collapse each class to one point. On *test* features the collapse is partial: features near a class's training region are pulled to its prototype, features in ambiguous regions are pulled toward whichever prototype's basin they fall in — which is exactly why transport can *create* errors as well as fix them, and why $\\Delta$Acc can be negative even when training loss is low. The classification rule only changes where basin boundaries differ from cosine-to-prototype boundaries.

2. **Near $t \\to 1$, standard FM learns a conditional average.** With deterministic $(z_i, p_{y_i})$ pairing, all class-$c$ paths converge on $p_c$ as $t \\to 1$ while their target velocities $p_c - z_i$ still differ — so at nearly identical inputs the network receives conflicting targets and can only learn their conditional mean. Rolled-out training does not have this consistency problem (it is judged only on where the composed map lands), which is the main mechanism by which the two modes can genuinely differ in late-time geometry — visible in the trajectory figures as the difference between smoothly converging paths and paths that still drift near the prototypes.
"""),
]
