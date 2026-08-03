# ADR 0006 — Post-results confirmation: dataset pair DTD + FGVC-Aircraft, prototype branch Option A

**Status:** accepted (2026-08-03)

**Context.** ADR 0005 deliberately ran a superset of the specification — all three
datasets and both prototype branches — precisely so that the two selections the spec
leaves to the group (which two datasets; which one prototype branch) could be made
*after seeing evidence* instead of being locked in blind. The Stage-1 results are now
complete, so this ADR records the two selections and the evidence behind them.
Neither selection uses test accuracy: the dataset evidence is structural (split
properties that the results made concrete), and the branch evidence is methodological
plus validation-split accuracy only, as already laid out in the report §6.

**Decision 1 — the selected pair is DTD + FGVC-Aircraft (confirmed).** ADR 0005
provisionally marked this pair `spec_selected` on split-size grounds; the completed
runs confirm the rationale rather than overturning it:

- **Flowers-102's degeneracy materialized exactly as predicted.** Its official
  training split holds exactly 10 images per class, so 10-shot ≡ full: the three
  10-shot runs are identical by construction (spread exactly 0.00) and the
  accuracy-vs-training-size curve has only two distinct points instead of three.
  Since the training-set-size axis is the axis Stage 2 argues along, Flowers-102 is
  the weakest member to build on.
- **DTD and FGVC-Aircraft each earn their place.** DTD (40 train images/class) and
  FGVC-Aircraft (~33/class) give three genuinely distinct K settings; FGVC-Aircraft
  additionally carries the required DINOv2 encoder and, as the fine-grained
  low-accuracy task, the most headroom for a Flow-Matching layer to demonstrate a
  gain. Flowers-102's class-imbalanced test split (20–238 images/class) also makes
  its top-1 the least clean single number of the three.

Flowers-102 stays in the Stage-1 deliverable as the clearly-marked ‡ extension — the
degeneracy is instructive and the numbers are already published — but Stage 2 and
Stage 3 build on DTD + FGVC-Aircraft.

**Decision 2 — the selected prototype branch is Option A, image-derived class
prototypes.** Recorded here for the ADR trail; the full argument and the
validation-only headroom numbers live in the report §6 and notebook §6. In brief:
(1) source and target of the Stage-2 Flow-Matching transport live in the same
encoder feature space (Option B would transport across CLIP's image–text modality
gap); (2) Option A inherits the full K ∈ {5, 10, full} × encoder grid, whereas
zero-shot CLIP is one fixed target set and stops being zero-shot the moment an FM
layer is trained on labeled images; (3) Option A reuses exactly the cached features
the linear probe uses, so Stages 2 and 3 share one feature pipeline; (4) the
validation-split probe-vs-prototype gap (5.3–35.7 points depending on
dataset/encoder) is measurable room for FM to close. Zero-shot CLIP remains in the
tables and figures as the already-computed ‡ reference.

**Consequences.** Stage 2 trains Flow Matching on DTD + FGVC-Aircraft with
image-derived prototype targets (per ADR 0003's revised policy: prototypes from the
selected *training* subset only), compared against the Stage-1 baselines on the
identical cached features and committed subset files. `config/config.json`'s
`spec_selected` flags already encode Decision 1 and stay unchanged. Both selections
remain revisable without re-running anything, since the superset is fully reported.
