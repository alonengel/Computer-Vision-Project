CELLS = [
    ("markdown", """
### Optional exploration ‡ — reverse feature-space integration from the prototypes

The specification encourages exploring the learned flow **in the reverse direction**, starting from the class prototypes, and comparing samples and prototypes at intermediate flow times. We integrate the learned field backward from $t = 1$ to $t = 0$ with explicit Euler:

$$\\hat{z}_{k-1} = \\hat{z}_k - \\tfrac{1}{T}\\, v_\\theta\\!\\left(\\hat{z}_k, \\tfrac{k}{T}\\right), \\qquad k = T, T{-}1, \\dots, 1,$$

starting from each selected class prototype as $\\hat{z}_T$ (first velocity evaluation at $t = 1$, as canonical reverse integration requires). This is **approximate backward integration of the learned continuous field** — not the algebraic inverse of the discrete forward-Euler map — and it is fully deterministic: this FM head is not a generative model, so one prototype yields exactly one reverse trajectory (no noise is injected). Setting: the same representative models, classes, examples, colours and **the same jointly fitted PCA plane** as the forward figures (the projection is fitted once on the identical inputs, never refit for the reverse plots). No model was retrained and no published result changes.
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
**What the actual results show.** The two training modes separate sharply in reverse. **Standard FM** integrates backward gracefully: reverse paths from the prototypes are short and stay inside the data region, and the reverse state tracks the forward same-class centroid closely at every intermediate time, ending at cosine ≈ 0.85 ± 0.06 to the original-feature centroid at $t = 0$. This is what per-path velocity supervision buys — the field carries meaningful values at every $(z, t)$ it was supervised on, so it can be traversed in either direction. **Rolled-out FM** collapses in reverse: several paths shoot far outside the data (visible in the right panel), similarity decays monotonically to ≈ 0.24 ± 0.20 at $t = 0$, and $L_2$ distances are ~4× larger. Its endpoint-only objective never constrains the field along the path — the composed forward map is accurate, but the field it is built from is not individually meaningful, and backward integration exposes exactly that. This is an independent, training-free confirmation of the §6 mechanism story (standard = regularized along the trajectory; rolled-out = free everywhere except the endpoint).

**Limitations (read before over-interpreting):** the network was trained in the **forward** direction only; rolled-out training never directly supervised any reverse trajectory; canonical reverse integration requires evaluating at $t = 1$, a time the forward Euler rollout never evaluates (it stops at $(T{-}1)/T$); reverse trajectories need **not** recover real samples — this is reverse *feature-space* exploration, not image generation; the learned map is deterministic and may be many-to-one, so nothing guarantees bijectivity; and the trajectory plot is a qualitative 2-D PCA projection (the similarity metrics are computed in the full 384-d space).
"""),
]
