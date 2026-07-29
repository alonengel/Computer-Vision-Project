CELLS = [
    ("markdown", """
# CVLAB Summer Project — Stage 1: Classification Baselines

**DTD · FGVC-Aircraft · Oxford Flowers-102 | frozen pretrained encoders**

The goal of Stage 1 is a reliable, reproducible classification pipeline built on **frozen** pretrained encoders. No flow-matching component appears at this stage: the selected prototype branch carries into Stage 2, and the linear-probe setting carries into Stage 3.

The specification (`_docs/stage_1.pdf`) lists three candidate baselines and asks each group to implement the linear probe plus **one** of the two prototype branches. We implement all three so the branch that continues into Stage 2 can be chosen from measured results:

1. **Linear probe** (required) — a multiclass linear classifier $s = Wz + b$ on frozen features, softmax cross-entropy, only $W$ and $b$ trained.
2. **Image-derived class prototypes** (branch A) — $\\mu_c = \\mathrm{normalize}\\left(\\frac{1}{|S_c|}\\sum_{i \\in S_c} \\mathrm{normalize}(z_i)\\right)$, classify by $\\hat{y} = \\arg\\max_c \\cos(z, \\mu_c)$.
3. **Zero-shot CLIP** (branch B) — text-derived class prototypes from CLIP RN50 prompts, $\\hat{y} = \\arg\\max_c \\cos(z, t_c)$; uses no labeled training images.

**Protocol.** All classes, official train / validation / test splits, never merging train and validation. Training-set sizes $K \\in \\{5, 10, \\text{full}\\}$ images per class; the 5- and 10-shot settings use balanced subsets of the official training split with seeds $\\{0,1,2\\}$. The validation split is used for model selection (linear-probe checkpointing) and the **complete official test split** only for the final top-1 accuracy.

**Group choices** (documented in `docs/adr/0005-group-choices-within-the-spec.md`): the specification allows any two of the three datasets; **our selected pair is DTD + FGVC-Aircraft** (their training splits make the three K settings genuinely distinct), and Flowers-102 is run additionally, marked ‡ throughout. DINOv2 ViT-S/14 is used on FGVC-Aircraft, the fine-grained task. ResNet-18 is used on all datasets. CLIP RN50 is used for the zero-shot branch only, as the spec restricts it.
"""),
]
