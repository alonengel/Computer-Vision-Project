# v2 talk — section 4: Strategy 2.

MAIN = r"""
## <span style="color:#1f3a5f;">3 · Strategy 2 — classifier-guided targets, standard FM training</span>

Each epoch contains two phases.

**Phase 1 — construct classifier-guided targets** (the frozen classifier is used here only):

$$u_0 = \Pi_{B(z,\rho)}(\hat{z}), \qquad \rho = 0.1\,\lVert z \rVert$$

$$u_{j+1} = \Pi_{B(z,\rho)}\!\left(u_j - \beta\rho\, \frac{\nabla_{u_j}\mathrm{CE}}{\lVert \nabla_{u_j}\mathrm{CE} \rVert + \varepsilon}\right), \qquad j = 0, \dots, m-1$$

- The current FM output $\hat{z}$ is projected into a trust region centred on the original feature $z$.
- Every candidate $u_j$ remains within the trust region.
- The candidate with the lowest CE is selected as the target $\hat{z}'$.
- The target is detached and cached for the epoch.

**Phase 2 — train the FM with standard velocity regression toward the cached targets:**

$$z_t = (1-t)\,z + t\,\hat{z}', \qquad t \sim U(0,1)$$

$$\mathcal{L}_{\mathrm{S2}} = \bigl\lVert\, v_\theta(z_t, t) - (\hat{z}' - z) \,\bigr\rVert^2$$

<div style="border-left:6px solid #0b7a75; background:#f2f8f7; color:#1a1a1a; padding:10px 16px; margin:0.8em 0; font-size:1.15em;">The classifier CE constructs the targets, but <b>no CE gradient reaches the FM</b>.</div>
"""

CELLS = [("markdown", MAIN)]
