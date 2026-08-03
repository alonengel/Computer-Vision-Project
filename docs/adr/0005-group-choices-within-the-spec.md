# ADR 0005 — Group choices within the specification's degrees of freedom

**Status:** accepted (2026-07-24). The two selections this ADR left open on evidence
— which pair carries forward and which prototype branch — were confirmed post-results
in ADR 0006 (2026-08-03): DTD + FGVC-Aircraft, Option A.

**Context.** The Stage-1 specification leaves three decisions to the group: which two of the three datasets to use, which of the two prototype-based branches to implement alongside the required linear probe, and which single dataset gets the DINOv2 ViT-S/14 encoder.

**Decision.**

1. **Datasets — all three** (DTD, FGVC-Aircraft, Oxford Flowers-102), rather than two. The specification allows any two of the three; our selected pair is **DTD + FGVC-Aircraft** (marked `spec_selected: true` in `config/config.json`), and Flowers-102 is run as a clearly-marked superset (‡ in tables) so the choice of pair can be revisited on evidence. Rationale: DTD (40 train images/class) and FGVC-Aircraft (~33/class) give three genuinely distinct settings for K ∈ {5, 10, full}, whereas Flowers-102's official training split has exactly 10 images per class, so its 10-shot setting *is* its full setting — a degeneracy worth showing rather than hiding.

2. **Both prototype branches** — image-derived class prototypes (Option A) *and* zero-shot CLIP (Option B), rather than one. The spec asks for one; implementing both costs little (the image-prototype head reuses cached ResNet-18/DINOv2 features, and CLIP RN50 is one extra frozen encoder) and lets the branch that carries into Stage 2 be chosen from measured results rather than assumed. The branch selected for Stage 2 is recorded in the report once the results are in.

3. **DINOv2 ViT-S/14 on FGVC-Aircraft only** — the spec-minimal scope. FGVC-Aircraft is the fine-grained task, where the contrast between ImageNet-supervised ResNet-18 features and self-supervised DINOv2 features is most informative. ResNet-18 runs on all datasets; CLIP RN50 is used for the zero-shot branch only, exactly as the spec restricts it.

**Consequences.** The deliverable is a strict superset of what is required, and every beyond-spec cell is labeled as such, so a reader can read off the exactly-compliant subset (DTD + FGVC-Aircraft; linear probe + one prototype branch) at a glance. The cost is roughly 50% more runs and one extra dataset download; all experiments run on cached features in minutes.
