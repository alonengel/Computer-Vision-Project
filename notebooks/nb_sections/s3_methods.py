CELLS = [
    ("markdown", """
## 3 · Methods

**Linear probe (required baseline).** A multiclass linear classifier $s = Wz + b$ over the frozen feature $z$; only $W$ and $b$ are trained, with softmax cross-entropy. The configuration is the one suggested by the specification, adopted unchanged and fixed a priori — AdamW, learning rate $10^{-3}$, weight decay $10^{-4}$, batch size 64, at most 200 epochs, and **checkpoint selection by highest validation accuracy**. No hyperparameter search was performed; the objective is a reasonable, stable probe, since the later comparison of interest concerns the FM layer.

**Image-derived class prototypes (branch A).** Features are $L_2$-normalized first, then

$$\\mu_c = \\mathrm{normalize}\\left(\\frac{1}{|S_c|}\\sum_{i \\in S_c} \\mathrm{normalize}(z_i)\\right), \\qquad \\hat{y} = \\arg\\max_c \\cos(z, \\mu_c),$$

where $S_c$ is the selected **training** subset for class $c$. There is nothing to train and no initialization, so the only source of run-to-run variation is which training images the subset seed selected.

**Zero-shot CLIP (branch B).** One text prototype per class from the frozen CLIP RN50 text encoder, using the dataset-specific prompt from the spec — `a photo of a {class} texture` (DTD), `a photo of a {class} aircraft` (FGVC-Aircraft), `a photo of a {class} flower` (Flowers-102). Image and text embeddings are normalized and classified with $\\hat{y} = \\arg\\max_c \\cos(z, t_c)$. This branch uses **no labeled training images**, so it produces exactly one result per dataset and is drawn as a horizontal reference line rather than a curve.
"""),
]
