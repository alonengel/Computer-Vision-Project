| Dataset | Encoder | Head | K = 5 | K = 10 | full train split |
|---|---|---|---|---|---|
| DTD | ResNet-18 | Stage-1 Image prototypes (baseline) | **46.51 ± 0.64** | **53.37 ± 1.04** | 58.78 |
| DTD | ResNet-18 | Standard FM, T = 4 | 43.81 ± 1.35 (Δ -2.70 ± 1.46) | 48.69 ± 0.62 (Δ -4.68 ± 1.61) | 59.26 ± 0.32 (Δ +0.48 ± 0.32) |
| DTD | ResNet-18 | Standard FM, T = 12 | 44.10 ± 1.36 (Δ -2.41 ± 1.36) | 49.27 ± 0.52 (Δ -4.10 ± 1.34) | **59.56 ± 0.11 (Δ +0.78 ± 0.11)** |
| DTD | ResNet-18 | Rolled-out FM, T = 4 | 38.95 ± 0.94 (Δ -7.55 ± 0.75) | 40.50 ± 0.91 (Δ -12.87 ± 0.45) | 54.80 ± 1.17 (Δ -3.97 ± 1.17) |
| DTD | ResNet-18 | Rolled-out FM, T = 12 | 39.47 ± 0.97 (Δ -7.04 ± 0.74) | 40.74 ± 0.52 (Δ -12.62 ± 0.53) | 55.60 ± 0.21 (Δ -3.17 ± 0.21) |
| FGVC-Aircraft | ResNet-18 | Stage-1 Image prototypes (baseline) | 16.04 ± 0.83 | 19.85 ± 0.47 | 25.20 |
| FGVC-Aircraft | ResNet-18 | Standard FM, T = 4 | 17.59 ± 0.66 (Δ +1.55 ± 0.18) | 23.76 ± 0.61 (Δ +3.91 ± 0.77) | 31.67 ± 0.17 (Δ +6.47 ± 0.17) |
| FGVC-Aircraft | ResNet-18 | Standard FM, T = 12 | **17.70 ± 0.79 (Δ +1.66 ± 0.06)** | **23.82 ± 0.65 (Δ +3.97 ± 0.63)** | **31.87 ± 0.14 (Δ +6.67 ± 0.14)** |
| FGVC-Aircraft | ResNet-18 | Rolled-out FM, T = 4 | 15.87 ± 0.73 (Δ -0.17 ± 0.76) | 20.63 ± 0.31 (Δ +0.78 ± 0.45) | 25.16 ± 0.92 (Δ -0.04 ± 0.92) |
| FGVC-Aircraft | ResNet-18 | Rolled-out FM, T = 12 | 15.86 ± 0.72 (Δ -0.18 ± 0.92) | 21.23 ± 1.26 (Δ +1.38 ± 1.04) | 25.34 ± 0.80 (Δ +0.14 ± 0.80) |
| FGVC-Aircraft | DINOv2 | Stage-1 Image prototypes (baseline) | 23.11 ± 1.36 | 27.80 ± 1.17 | 34.41 |
| FGVC-Aircraft | DINOv2 | Standard FM, T = 4 | 33.91 ± 1.44 (Δ +10.80 ± 0.50) | 43.58 ± 0.54 (Δ +15.78 ± 0.84) | **57.59 ± 0.71 (Δ +23.17 ± 0.71)** |
| FGVC-Aircraft | DINOv2 | Standard FM, T = 12 | 33.50 ± 1.40 (Δ +10.39 ± 0.47) | 43.28 ± 0.53 (Δ +15.48 ± 0.74) | 57.28 ± 0.53 (Δ +22.86 ± 0.53) |
| FGVC-Aircraft | DINOv2 | Rolled-out FM, T = 4 | 33.87 ± 1.43 (Δ +10.76 ± 1.22) | 43.25 ± 1.06 (Δ +15.45 ± 1.05) | 55.37 ± 0.14 (Δ +20.95 ± 0.14) |
| FGVC-Aircraft | DINOv2 | Rolled-out FM, T = 12 | **34.02 ± 1.69 (Δ +10.91 ± 1.51)** | **43.73 ± 1.63 (Δ +15.93 ± 1.14)** | 54.28 ± 0.82 (Δ +19.86 ± 0.82) |

Top-1 accuracy (%) on the complete official test split; Δ = change vs the Stage-1 image-prototype baseline of the **identical** setting (same subset indices, seeds and prototypes). K ∈ {5, 10}: mean ± sample std over the 3 subset seeds, Δ paired per seed (mean ± std of per-seed differences). full: 3 FM initialization seeds against the single deterministic full-split baseline, so the Δ spread there measures FM training stochasticity only. The two Standard-FM rows of a setting share one trained network (standard FM training is independent of T; only inference differs). **Bold** marks the best value per column within each dataset–encoder block.