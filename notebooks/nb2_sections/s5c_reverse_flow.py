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
