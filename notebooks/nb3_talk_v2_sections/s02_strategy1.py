# v2 talk — section 3: Strategy 1.

MAIN = r"""
## <span style="color:#1f3a5f;">2 · Strategy 1 — rolled-out classification training</span>

1. Run the complete four-step FM rollout $z \to \hat{z}$.
2. Apply the frozen classifier to the transported feature $\hat{z}$.
3. Compute the cross-entropy against the true label.
4. Backpropagate through the whole rollout.
5. Update **only** the FM, with a relative displacement regularizer that discourages unnecessarily large moves.

$$\mathcal{L}_{\mathrm{S1}} = \mathrm{CE}(W\hat{z} + b,\ y) \;+\; \lambda\, \operatorname{mean}_i\, \frac{\lVert \hat{z}_i - z_i \rVert^2}{\lVert z_i \rVert^2 + \varepsilon}$$

Validation selected $\lambda = 1$ on both datasets (grid $\{0, 1, 10, 100\}$). The $\lambda = 0$ variant was also evaluated as a pre-registered pair; its complete table is in Appendix A.6.
"""

CELLS = [("markdown", MAIN)]
