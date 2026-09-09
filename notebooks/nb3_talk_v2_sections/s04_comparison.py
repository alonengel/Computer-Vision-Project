# v2 talk — section 5: strategy comparison table.

MAIN = r"""
## <span style="color:#1f3a5f;">4 · Strategy comparison</span>

| | Strategy 1 — rolled-out CE | Strategy 2 — classifier-guided targets |
|---|---|---|
| **Training signal** | cross-entropy backpropagated through the full rollout, plus a relative displacement penalty | standard velocity regression toward cached classifier-guided targets |
| **Role of the frozen classifier** | inside the loss — its gradient flows through the rollout into the FM | builds the targets — its gradient never reaches the FM |
| **How feature movement is constrained** | soft penalty on $\lVert\hat{z} - z\rVert^2 / \lVert z\rVert^2$ with weight $\lambda$ | hard trust region of radius $0.1\,\lVert z\rVert$ around $z$; lowest-CE candidate accepted |
| **Selected DTD configuration** | $\lambda = 1$ | $\beta = 0.25,\ m = 1$ |
| **Selected FGVC-Aircraft configuration** | $\lambda = 1$ | $\beta = 0.5,\ m = 3$ |

Both strategies train the same FM architecture with the same optimizer, epoch budget and checkpoint rule; only the training signal differs.
"""

CELLS = [("markdown", MAIN)]
