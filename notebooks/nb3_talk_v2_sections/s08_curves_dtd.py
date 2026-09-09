# v2 talk — section 9: training behaviour, DTD.

HEAD = r"""
## <span style="color:#1f3a5f;">8 · Training behaviour — DTD / ResNet-18</span>

Same layout and colours as the previous figure; validation-selected models, subset seed 0.
"""

CODE = r"""
training_figure("dtd", "resnet18")
"""

AFTER = r"""
- Training accuracy approaches 100%, while validation remains near 50–52%.
- The validation curves remain close, matching the near-zero test change.
"""

CELLS = [
    ("markdown", HEAD),
    ("code", CODE),
    ("markdown", AFTER),
]
