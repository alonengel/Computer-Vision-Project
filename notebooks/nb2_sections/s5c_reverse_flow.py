CELLS = [
    ("markdown", """
### Optional exploration ‡ — reverse feature-space integration from the prototypes

The specification encourages exploring the learned flow **in the reverse direction**, starting from the class prototypes, and comparing samples and prototypes at intermediate flow times. We integrate the learned field backward from $t = 1$ to $t = 0$ with explicit Euler:

$$\\hat{z}_{k-1} = \\hat{z}_k - \\tfrac{1}{T}\\, v_\\theta\\!\\left(\\hat{z}_k, \\tfrac{k}{T}\\right), \\qquad k = T, T{-}1, \\dots, 1,$$

starting from each selected class prototype as $\\hat{z}_T$ (first velocity evaluation at $t = 1$, as canonical reverse integration requires). This is **approximate backward integration of the learned continuous field** — not the algebraic inverse of the discrete forward-Euler map — and it is fully deterministic: this FM head is not a generative model, so one prototype yields exactly one reverse trajectory (no noise is injected). Setting: the same representative models, classes, examples, colours and **the same jointly fitted PCA plane** as the forward figures (the projection is fitted once on the identical inputs, never refit for the reverse plots). No model was retrained and no published result changes.

**Why the reverse arrows do not retrace the forward arrows.** Comparing this figure with the forward trajectory figure above, the reverse paths are *not* the forward paths played backwards — and should not be: (i) the reverse integration starts *exactly at the prototype*, whereas forward trajectories start at individual test features and end only *near* it; (ii) reverse Euler evaluates the field at different states and times than the forward steps did, so even along the same corridor the discrete paths differ; (iii) near $t = 1$ the standard-FM field is a *conditional average* over the whole class (§5), so its backward path follows the class-mean corridor rather than any single sample's path. To make this comparison visible instead of implicit, each panel also draws the **forward class-centroid path** (dashed, faint) in the same plane: for Standard FM the reverse path hugs it almost exactly — for Rolled-out FM it departs immediately.
"""),
    ("code", """
display(Image(str(REPO / "results" / "figures" /
                  "stage2_reverse_flow_fgvc_aircraft_dinov2_vits14_T12.png"), width=980))
"""),
    ("markdown", """
**Samples vs prototypes at matching intermediate times.** For each selected class and every Euler time $t = k/T$: the centroid of the class's selected test samples along the **forward** flow is compared against the **reverse** trajectory that started at that class's prototype — cosine similarity ($L_2$ as a secondary metric), aggregated over the ten classes as mean ± sample std; per-class raw values in `results/metrics/stage2_reverse_intermediate_fgvc_aircraft_dinov2_vits14_T12.csv`. Both quantities live in the original 384-d feature space, not the PCA plane.
"""),
    ("code", """
display(Image(str(REPO / "results" / "figures" /
                  "stage2_reverse_intermediate_fgvc_aircraft_dinov2_vits14_T12.png"),
              width=920))
rev = pd.read_csv(REPO / "results" / "metrics" /
                  "stage2_reverse_intermediate_fgvc_aircraft_dinov2_vits14_T12.csv")
ends = rev[rev["t"] == 0].groupby("head")[["cosine", "l2"]].agg(["mean", "std"]).round(3)
print("At t = 0 (reverse endpoint vs original-feature class centroid), "
      "mean ± std over 10 classes:")
display(ends)
"""),
    ("markdown", """
**Reverse-flow Top-1 (%), complete official test split.** Two classification read-outs quantify the pictures (single deterministic runs on the representative 10-shot seed-0 models; generated into `results/metrics/stage2_reverse_top1_fgvc_aircraft_dinov2_vits14_T12.csv`): **round-trip** — transport every test feature forward to $t = 1$, integrate it back to $t = 0$, and classify the reconstruction with the Stage-1 cosine rule (measures how much class information survives forward-then-backward traversal); and **reverse-endpoint classifier** — integrate every one of the 100 class prototypes back to $t = 0$ and use the endpoints as class representatives for the *original* features (measures whether the backward images of the prototypes remain usable class descriptions). Both use only labeled-free inference on the already-trained models; the generation script asserts the forward accuracy still reproduces the published grid before writing these numbers.
"""),
    ("code", """
racc = pd.read_csv(REPO / "results" / "metrics" /
                   "stage2_reverse_top1_fgvc_aircraft_dinov2_vits14_T12.csv")
show = racc.copy()
for c in show.columns[1:]:
    show[c] = (100 * show[c]).round(2)
show.columns = ["head", "Stage-1 prototype baseline", "Forward FM (published)",
                "Round-trip (fwd → rev)", "Reverse-endpoint classifier"]
display(show)
"""),
    ("markdown", """
**What the actual results show.** The two training modes separate sharply in reverse. **Standard FM** integrates backward gracefully: reverse paths from the prototypes are short, stay inside the data region and hug the forward class-centroid paths; the reverse state tracks the forward same-class centroid at every intermediate time (cosine ≈ 0.85 ± 0.06 at $t = 0$); the round trip *preserves* class information (Top-1 31.2% vs the 29.1% baseline on untouched features); and even the backward images of the prototypes remain serviceable class representatives (25.7%). This is what per-path velocity supervision buys — the field carries meaningful values at every $(z, t)$, so it can be traversed in either direction. **Rolled-out FM** collapses in reverse: paths shoot far outside the data, similarity decays to ≈ 0.24 ± 0.20, the round trip *loses* information (26.0%, below the baseline), and the reverse-endpoint classifier drops to 11.1%. Its endpoint-only objective never constrains the field along the path — the composed forward map is accurate, but the field it is built from is not individually meaningful, and backward integration exposes exactly that. This is an independent, training-free confirmation of the §6 mechanism story (standard = regularized along the trajectory; rolled-out = free everywhere except the endpoint).

**Limitations (read before over-interpreting):** the network was trained in the **forward** direction only; rolled-out training never directly supervised any reverse trajectory; canonical reverse integration requires evaluating at $t = 1$, a time the forward Euler rollout never evaluates (it stops at $(T{-}1)/T$); reverse trajectories need **not** recover real samples — this is reverse *feature-space* exploration, not image generation; the learned map is deterministic and may be many-to-one, so nothing guarantees bijectivity; and the trajectory plot is a qualitative 2-D PCA projection (the similarity metrics are computed in the full 384-d space).
"""),
]
