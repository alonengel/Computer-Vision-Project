CELLS = [
    ("markdown", """
## 5 · Feature-space visualization

The spec's requirement: for a readable subset of classes, visualize the original features $z$ and the transported features $\\hat{z}$ for the two Stage-3 methods, same test examples and class colours, embedding computed **jointly** over the compared sets. Below: the shared ten `viz_selection` classes per dataset (same classes, examples, and colours as Stages 1–2), one PCA fitted jointly on $[z, \\hat{z}_{S1}, \\hat{z}_{S2}]$ in **raw** space, validation-selected models of seed 0. As throughout the project, 2-D projections are qualitative; the purpose is to see how each training strategy changes the class structure the frozen classifier receives.
"""),
    ("code", """
for ds, enc in cfg["stage3"]["settings"]:
    p = REPO / "results" / "figures" / f"stage3_features_{ds}_{enc}.png"
    if p.exists():
        display(Image(str(p), width=980))
"""),
]
