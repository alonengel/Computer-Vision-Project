| Dataset | Encoder | Head | K | Δ per seed (paired) | Seeds favouring FM |
|---|---|---|---|---|---|
| DTD | ResNet-18 | Standard FM, T = 4 | 5 | -2.70 ± 1.46 | 0 / 3 |
| DTD | ResNet-18 | Standard FM, T = 12 | 5 | -2.41 ± 1.36 | 0 / 3 |
| DTD | ResNet-18 | Rolled-out FM, T = 4 | 5 | -7.55 ± 0.75 | 0 / 3 |
| DTD | ResNet-18 | Rolled-out FM, T = 12 | 5 | -7.04 ± 0.74 | 0 / 3 |
| DTD | ResNet-18 | Standard FM, T = 4 | 10 | -4.68 ± 1.61 | 0 / 3 |
| DTD | ResNet-18 | Standard FM, T = 12 | 10 | -4.10 ± 1.34 | 0 / 3 |
| DTD | ResNet-18 | Rolled-out FM, T = 4 | 10 | -12.87 ± 0.45 | 0 / 3 |
| DTD | ResNet-18 | Rolled-out FM, T = 12 | 10 | -12.62 ± 0.53 | 0 / 3 |
| FGVC-Aircraft | ResNet-18 | Standard FM, T = 4 | 5 | +1.55 ± 0.18 | 3 / 3 |
| FGVC-Aircraft | ResNet-18 | Standard FM, T = 12 | 5 | +1.66 ± 0.06 | 3 / 3 |
| FGVC-Aircraft | ResNet-18 | Rolled-out FM, T = 4 | 5 | -0.17 ± 0.76 | 2 / 3 |
| FGVC-Aircraft | ResNet-18 | Rolled-out FM, T = 12 | 5 | -0.18 ± 0.92 | 2 / 3 |
| FGVC-Aircraft | ResNet-18 | Standard FM, T = 4 | 10 | +3.91 ± 0.77 | 3 / 3 |
| FGVC-Aircraft | ResNet-18 | Standard FM, T = 12 | 10 | +3.97 ± 0.63 | 3 / 3 |
| FGVC-Aircraft | ResNet-18 | Rolled-out FM, T = 4 | 10 | +0.78 ± 0.45 | 3 / 3 |
| FGVC-Aircraft | ResNet-18 | Rolled-out FM, T = 12 | 10 | +1.38 ± 1.04 | 3 / 3 |
| FGVC-Aircraft | DINOv2 | Standard FM, T = 4 | 5 | +10.80 ± 0.50 | 3 / 3 |
| FGVC-Aircraft | DINOv2 | Standard FM, T = 12 | 5 | +10.39 ± 0.47 | 3 / 3 |
| FGVC-Aircraft | DINOv2 | Rolled-out FM, T = 4 | 5 | +10.76 ± 1.22 | 3 / 3 |
| FGVC-Aircraft | DINOv2 | Rolled-out FM, T = 12 | 5 | +10.91 ± 1.51 | 3 / 3 |
| FGVC-Aircraft | DINOv2 | Standard FM, T = 4 | 10 | +15.78 ± 0.84 | 3 / 3 |
| FGVC-Aircraft | DINOv2 | Standard FM, T = 12 | 10 | +15.48 ± 0.74 | 3 / 3 |
| FGVC-Aircraft | DINOv2 | Rolled-out FM, T = 4 | 10 | +15.45 ± 1.05 | 3 / 3 |
| FGVC-Aircraft | DINOv2 | Rolled-out FM, T = 12 | 10 | +15.93 ± 1.14 | 3 / 3 |

Mean ± sample std (ddof = 1) of the per-seed difference FM − Stage-1 image-prototype baseline, both computed from identical committed subset indices and prototypes. Positive favours FM. n = 3 seeds; no confidence intervals are quoted at this sample size.