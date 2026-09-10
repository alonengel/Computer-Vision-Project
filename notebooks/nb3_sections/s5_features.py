CELLS = [
    ("markdown", """
## 5 · Feature-space visualization

The spec's requirement: for a readable subset of classes, visualize the original features $z$ and the transported features $\\hat{z}$ for the two Stage-3 methods, same test examples and class colours, embedding computed **jointly** over the compared sets. Below: the shared ten `viz_selection` classes per dataset (same classes, examples, and colours as Stages 1–2), one PCA fitted jointly on $[z, \\hat{z}_{\\text{Rolled}}, \\hat{z}_{\\text{Guided}}]$ in **raw** space, validation-selected models of seed 0. As throughout the project, 2-D projections are qualitative; the purpose is to see how each training strategy changes the class structure the frozen classifier receives.

**What the panels actually show:** the first two PCs change only subtly — especially on FGVC-Aircraft, despite its +2.50-point gain. This suggests the useful movement occurs along **classifier-relevant high-dimensional directions not captured by the joint two-dimensional PCA** (which explains 15.6% of the variance on DTD and 33.0% on FGVC). That is coherent with the measured displacements (≈5–13% of the mean feature norm): small, targeted adjustments rather than a visible reorganization.
"""),
    ("code", """
for ds, enc in cfg["stage3"]["settings"]:
    p = REPO / "results" / "figures" / f"stage3_features_{ds}_{enc}.png"
    if p.exists():
        display(Image(str(p), width=980))
"""),
    ("markdown", """
**Overlay view — before and after in the same plane.** The identical joint-PCA projection, one panel per strategy: the original features $z$ drawn faded, the transported features $\\hat{z}$ in full colour, and a thin grey segment joining each test example to its transported position. Most segments are shorter than the marker — the transported point sits on top of its original — and the visible ones are short and not class-coherent: no class shifts as a block. Consistent with the reading above, the movement that matters lies largely outside this plane.
"""),
    ("code", """
for ds, enc in cfg["stage3"]["settings"]:
    p = REPO / "results" / "figures" / f"stage3_features_overlay_{ds}_{enc}.png"
    if p.exists():
        display(Image(str(p), width=980))
"""),
]
