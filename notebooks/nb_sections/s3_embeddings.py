CELLS = [
    ("markdown", """
## 3 · Embedding spaces

t-SNE of the frozen test embeddings per backbone (★ = class prototype). Class separation in these plots predicts few-shot performance: prototypes only work when frozen embeddings already cluster by class.
"""),
    ("code", """
for ds in ("mnist", "cifar10", "mini_imagenet"):
    display(Image(str(REPO / "results" / "figures" / f"tsne_{ds}.png"), width=980))
"""),
]
