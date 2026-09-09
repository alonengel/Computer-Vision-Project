# v2 talk — section 11: feature space, DTD (joint PCA).

HEAD = r"""
## <span style="color:#1f3a5f;">10 · Feature space — DTD / ResNet-18</span>

Same construction as the previous figure: the shared ten-class selection, one PCA fitted jointly over the three feature sets, seed-0 models.
"""

CODE = r"""
feature_figure("dtd", "resnet18", ("PC1 (8.2% var.)", "PC2 (7.4% var.)"))
"""

AFTER = r"""
- The first two components explain 15.6% of the variance.
- The projection changes little across the methods.
- This is consistent with the lack of a clear classification improvement.

Both PCA figures are qualitative two-dimensional views of transformations that take place in the full 384-dimensional (DINOv2) and 512-dimensional (ResNet-18) feature spaces.
"""

CELLS = [
    ("markdown", HEAD),
    ("code", CODE),
    ("markdown", AFTER),
]
