CELLS = [
    ("markdown", """
# CVLAB Summer Project — Stage 2: Flow Matching to Class Prototypes

**DTD · FGVC-Aircraft | frozen pretrained encoders | FM decision layer**

Stage 2 adds a **flow-matching (FM) layer** on top of the Stage-1 prototype classifier: given a frozen image feature $z$, a small velocity network transports it toward the fixed prototype of its class, and the transported feature is classified with the **identical** cosine/prototype rule as Stage 1. The specification is `_docs/stage_2.pdf`; every design decision left open by it was fixed *before* the first training run in `docs/adr/0007-stage2-design-decisions.md`.

Three heads are compared under identical conditions:

1. **Stage-1 prototype baseline** — no transport (the published Stage-1 numbers, reproduced exactly by this pipeline's $T=0$ integrity guard);
2. **Standard FM** — the velocity network is supervised at random interpolation points along the ideal path;
3. **Rolled-out FM** — the network is trained through the same $T$-step Euler sequence used at inference, with a loss on the final transported feature only.

**Scope and group choices.** Datasets: **DTD + FGVC-Aircraft** — the pair selected on evidence at the end of Stage 1 (ADR 0006). Spec branch: FM toward **image-derived prototypes** (the Stage-1 Option A selection) on ResNet-18 (both datasets) and DINOv2 ViT-S/14 (FGVC-Aircraft). As a clearly-marked **‡ extension** we additionally run the full grid on **CLIP RN50 image features toward the fixed CLIP text prototypes** — Stage 1 established both prototype branches, and the text targets sit across CLIP's image–text modality gap, which makes the transport problem qualitatively different. CLIP‡ FM rows consume labeled images and are therefore *supervised transport*, never "zero-shot"; their reference is the Stage-1 zero-shot number.

**Protocol (inherited from Stage 1, unchanged).** Same feature caches, same committed balanced-subset index files, same class-prototype formula, same seeds, same complete official test split. $K \\in \\{5, 10, \\text{full}\\}$; at $K \\in \\{5,10\\}$ the three runs are the three subset seeds (FM initialization fixed), at full they are three FM initialization seeds. Every reported change is $\\Delta\\mathrm{Acc} = \\mathrm{Acc}_{\\mathrm{FM}} - \\mathrm{Acc}_{\\mathrm{baseline}}$ against the Stage-1 baseline of the *identical* setting, paired per seed where pairing is genuine.
"""),
]
