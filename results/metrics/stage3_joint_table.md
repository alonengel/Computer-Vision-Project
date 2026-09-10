| Dataset / encoder | Variant | Top-1 (%) | Δ vs pinned probe (pts) |
|---|---|---|---|
| DTD / ResNet-18 | Joint FM + classifier fine-tuning | 53.88 ± 0.63 | -0.43 ± 0.37 |
| DTD / ResNet-18 | Control: classifier-only continued training | 54.26 ± 0.46 | -0.05 ± 0.19 |
| DTD / ResNet-18 | (reference) Guided strategy, frozen classifier | 54.47 ± 0.70 | +0.16 ± 0.45 |
| FGVC-Aircraft / DINOv2 | Joint FM + classifier fine-tuning | 51.86 ± 0.96 | +0.56 ± 0.98 |
| FGVC-Aircraft / DINOv2 | Control: classifier-only continued training | 51.94 ± 0.98 | +0.64 ± 0.08 |
| FGVC-Aircraft / DINOv2 | (reference) Guided strategy, frozen classifier | 53.80 ± 0.95 | +2.50 ± 0.50 |

Optional extension (ADR 0008 §9), run after the mandatory comparison: the classifier is unfrozen (lr 1e-4; FM lr 1e-3; same policy, 3 seeds). The **control** trains the pinned classifier further *alone* with the same budget — without it, a joint gain could not be attributed to the FM rather than to the classifier simply training longer. Same pinned-probe baselines as the main table.