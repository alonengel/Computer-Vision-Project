CELLS = [
    ("markdown", """
## 3 · Embedding spaces

t-SNE of the frozen test embeddings per backbone (black star = class prototype). These plots are a **qualitative** visualization — t-SNE distorts distances and neighborhood structure, so they are illustration, not evidence. Their visual class separation is, however, consistent with the quantitative metrics below and with the classification results in §5.
"""),
    ("code", """
for ds in ("mnist", "cifar10", "mini_imagenet"):
    display(Image(str(REPO / "results" / "figures" / f"tsne_{ds}.png"), width=980))
"""),
    ("markdown", """
**Quantitative embedding quality** (test features, 2000-point subsample, cosine metric): silhouette score and 1-NN accuracy higher = better; within/between-class distance ratio lower = better. The backbone ranking here predicts the prototype-classifier ranking in §5.
"""),
    ("code", """
eq = pd.read_csv(REPO / "results" / "metrics" / "embedding_quality.csv")
display(eq.pivot(index="backbone", columns="dataset",
                 values=["silhouette", "nn1_acc", "within_between_ratio"]).round(3))
"""),
]
