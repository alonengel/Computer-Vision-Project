CELLS = [
    ("markdown", """
### 5.5 Feature visualizations

Ten classes per dataset (chosen once with a fixed seed) with up to 40 test images each. For a given dataset the **same classes, the same test examples and the same class colours** are used in every panel, so encoders and methods can be compared directly.

Each figure shows two two-dimensional projections of the $L_2$-normalized features — the space in which the cosine classifier actually operates. **PCA is the primary view**: it is deterministic, linear, and its axes and relative distances have a global meaning. **t-SNE is supplementary**: it is useful for revealing local cluster structure, but it preserves only local neighbourhoods — global distances, cluster sizes and absolute coordinates are not meaningful, and coordinates are not comparable across separately fitted panels. Every t-SNE panel uses the same fixed parameters (perplexity 30, PCA initialization, seed 0).

Stars are the class prototypes — image-derived for ResNet-18 and DINOv2, text-derived for CLIP RN50 — and, as the specification requires, **each projection is fitted jointly to the image features and the prototypes shown in it**, so a prototype's position is directly comparable to the points around it. The prototype/feature class order is asserted against the feature caches at figure-generation time.

**What to look for** (reading local neighbourhoods, not global geometry): on FGVC-Aircraft with DINOv2, distinctive light aircraft (Cessna 172, DHC-1, Fokker 50) form tight, well-separated clusters while the Boeing/Airbus narrow- and wide-bodies overlap heavily — a direct visual account of where the remaining ~33% of errors come from. The same ten classes under ResNet-18 show visibly weaker local structure, consistent with the accuracy gap between the two encoders.

**In the CLIP panels**, the ten text prototypes appear close to one another and separated from the image points in both projections. This is *consistent with* the modality gap reported for CLIP-style models in the literature (image and text embeddings concentrating in different regions of the joint space), but a 2-D projection — especially t-SNE — cannot establish that on its own; confirming it would require distance measurements in the original 1024-d space. Either way it does not indicate a broken classifier: zero-shot classification depends only on the *relative* cosine ordering of a query against the text prototypes, never on absolute image–text proximity.
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
