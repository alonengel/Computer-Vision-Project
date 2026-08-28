CELLS = [
    ("markdown", """
# CVLAB Summer Project — Stage 3: FM Before a Linear Classifier

**DTD · FGVC-Aircraft | frozen encoders | FM → frozen Stage-1 probe**

Stage 3 inserts an FM transformation **before** the pretrained linear classifier:

$$z \\;\\xrightarrow{\\;\\mathrm{FM},\\;T\\;\\text{Euler steps}\\;}\\; \\hat{z} \\;\\xrightarrow{\\;\\text{frozen linear classifier}\\;}\\; s$$

The linear classifier is trained first, **exactly as in Stage 1**, then frozen; the FM is initialized close to identity so that, before Stage-3 training, the complete system behaves like the original linear probe. The question: **can FM transform the frozen encoder features into a representation the existing classifier separates better?** Specification: `_docs/stage_3.pdf`; every decision it leaves open was fixed *before* the first training run in ADR 0008 (`docs/adr/0008-stage3-design-decisions.md`), which itself survived three rounds of an external methodological review (R1–R28, all resolved) plus a cv-expert pre-run review.

Three heads are compared per dataset, exactly as the spec's Main Comparison requires:

1. **Pinned Stage-1 linear probe** — the baseline, and literally the frozen classifier inside the pipeline;
2. **Strategy 1 — end-to-end rolled-out classification training**: backpropagate CE through the complete rollout, updating only the FM; with a relative displacement regularizer $\\lambda\\,\\overline{\\lVert\\hat{z}-z\\rVert^2/\\lVert z\\rVert^2}$ (the spec explicitly invites such regularization), $\\lambda$ selected on validation, and the pre-registered $\\lambda = 0$ variant always reported;
3. **Strategy 2 — classifier-guided targets + standard FM training**: the frozen classifier's feature-space CE gradient constructs a nearby improved target $\\hat{z}'$, and the FM is trained with the *standard* velocity-regression loss toward it — **no CE gradient ever reaches the FM**.

**Scope (per the spec):** one encoder per dataset — DTD → ResNet-18, FGVC-Aircraft → DINOv2 (the Stage-1 validation-selected encoders); one training-set size K = 10 (the spec's suggested default); one fixed T = 4 throughout. Three runs per setting (subset seeds {0, 1, 2}, probe-init and FM-init fixed at 0, as in Stage 1) so every Δ is **paired per seed** against the exact pinned probe of that seed.
"""),
]
