# v2 talk — section 6: experimental protocol and guard summary.

MAIN = r"""
## <span style="color:#1f3a5f;">5 · Experimental protocol</span>

| | DTD | FGVC-Aircraft |
|---|---|---|
| **Frozen encoder** | ResNet-18 (ImageNet-1K), 512-dimensional features | DINOv2 ViT-S/14, 384-dimensional features |
| **Training set** | K = 10 examples per class; subset seeds 0, 1, 2 | K = 10 examples per class; subset seeds 0, 1, 2 |

- **T = 4** Euler steps; **exactly 200 training epochs**; the **best validation-accuracy checkpoint** is retained.
- Hyperparameters ($\lambda$; $\beta$, $m$) selected on **seed-0 validation** only, then applied unchanged to seeds 1 and 2.
- **Test data sealed** during training and model selection; evaluated once per locked checkpoint.
- **All validation guards passed:** exact identity at initialization; pinned probes reproduced the Stage-1 accuracies; no NaN or Inf failures (full audits in Appendix A.2–A.3).
"""

CELLS = [("markdown", MAIN)]
