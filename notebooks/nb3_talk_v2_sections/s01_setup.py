# v2 talk — section 2: Stage 3 setup.

MAIN = r"""
## <span style="color:#1f3a5f;">1 · Stage 3 setup</span>

- **The encoder remains frozen** — the features $z$ are extracted once and never change.
- **The Stage-1 linear classifier remains frozen** — retrained exactly as in Stage 1, then pinned; it is both the baseline and the classifier inside the pipeline.
- **Only the FM velocity network is trained.**
- **The FM starts as an exact identity transformation** — its final layer is zero-initialized, so at epoch 0 the complete pipeline behaves exactly like the pinned Stage-1 probe.

$$\hat{z}_{k+1} = \hat{z}_k + \frac{1}{T}\, v_\theta\!\left(\hat{z}_k,\ \frac{k}{T}\right), \qquad T = 4$$

Stage 3 operates in the **raw feature space** used by the linear classifier ($s = Wz + b$): no normalization is applied on entry, so the identity initialization is exact.
"""

CELLS = [("markdown", MAIN)]
