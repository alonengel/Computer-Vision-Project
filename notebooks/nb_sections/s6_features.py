CELLS = [
    ("markdown", """
### 5.5 Feature visualizations

Ten classes per dataset (chosen once with a fixed seed) with up to 40 test images each. For a given dataset the **same classes, the same test examples and the same class colours** are used in every panel, so encoders and methods can be compared directly.

Each figure shows both a PCA and a t-SNE view. Stars are the class prototypes: image-derived prototypes for ResNet-18 and DINOv2, text prototypes for CLIP RN50. Features are $L_2$-normalized before projection — the space in which the cosine classifier actually operates — and, as the spec requires, **the projection is fitted jointly to the image features and the prototypes shown in the plot**, so a prototype's position is directly comparable to the points around it.

These are qualitative views of the representation, not measurements of classification performance: both projections distort distances, t-SNE especially so.
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
