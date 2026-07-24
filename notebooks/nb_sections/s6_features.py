CELLS = [
    ("markdown", """
### 5.5 Feature visualizations

Ten classes per dataset (chosen once with a fixed seed) with up to 40 test images each. For a given dataset the **same classes, the same test examples and the same class colours** are used in every panel, so encoders and methods can be compared directly.

Each figure shows both a PCA and a t-SNE view. Stars are the class prototypes: image-derived prototypes for ResNet-18 and DINOv2, text prototypes for CLIP RN50. Features are $L_2$-normalized before projection — the space in which the cosine classifier actually operates — and, as the spec requires, **the projection is fitted jointly to the image features and the prototypes shown in the plot**, so a prototype's position is directly comparable to the points around it.

These are qualitative views of the representation, not measurements of classification performance: both projections distort distances, t-SNE especially so.

**What to look for.** On FGVC-Aircraft with DINOv2, distinctive light aircraft (Cessna 172, DHC-1, Fokker 50) form tight, well-separated clusters while the Boeing/Airbus narrow- and wide-bodies overlap heavily — a direct visual account of where the remaining ~33% of errors come from. The same ten classes under ResNet-18 show markedly weaker structure, matching the 30-point accuracy gap.

**In the CLIP panels**, the ten text prototypes cluster tightly together, away from the image cloud. This is the well-known CLIP **modality gap** — image and text embeddings occupy separate cones of the shared space — and is *not* a broken classifier: zero-shot classification depends only on the relative cosine ordering of a query against the text prototypes, never on absolute image–text proximity.
"""),
    ("code", """
for name in ("features_dtd_resnet18.png", "features_dtd_clip_rn50.png",
             "features_fgvc_aircraft_resnet18.png", "features_fgvc_aircraft_dinov2_vits14.png",
             "features_fgvc_aircraft_clip_rn50.png",
             "features_flowers102_resnet18.png", "features_flowers102_clip_rn50.png"):
    p = REPO / "results" / "figures" / name
    if p.exists():
        display(Image(str(p), width=980))
"""),
]
