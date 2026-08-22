| Dataset | Encoder | Head | K = 5 | K = 10 | full train split |
|---|---|---|---|---|---|
| DTD | ResNet-18 | Standard FM, T = 4 | 29.93 ± 2.09 (norm -13.88) | 38.10 ± 0.99 (norm -10.59) | 39.80 ± 1.22 (norm -19.45) |
| DTD | ResNet-18 | Standard FM, T = 12 | 33.87 ± 1.44 (norm -10.23) | 40.48 ± 0.86 (norm -8.79) | 43.55 ± 0.97 (norm -16.01) |
| DTD | ResNet-18 | Rolled-out FM, T = 4 | 17.20 ± 0.99 (norm -21.76) | 21.38 ± 0.43 (norm -19.11) | 46.12 ± 0.52 (norm -8.69) |
| DTD | ResNet-18 | Rolled-out FM, T = 12 | 21.37 ± 1.21 (norm -18.10) | 27.22 ± 1.65 (norm -13.53) | 48.39 ± 1.00 (norm -7.22) |
| FGVC-Aircraft | ResNet-18 | Standard FM, T = 4 | 9.33 ± 0.12 (norm -8.26) | 12.23 ± 1.07 (norm -11.53) | 15.05 ± 0.93 (norm -16.62) |
| FGVC-Aircraft | ResNet-18 | Standard FM, T = 12 | 11.35 ± 0.66 (norm -6.35) | 13.01 ± 1.07 (norm -10.81) | 16.10 ± 0.36 (norm -15.77) |
| FGVC-Aircraft | ResNet-18 | Rolled-out FM, T = 4 | 5.46 ± 0.56 (norm -10.41) | 11.09 ± 0.47 (norm -9.54) | 13.44 ± 2.51 (norm -11.72) |
| FGVC-Aircraft | ResNet-18 | Rolled-out FM, T = 12 | 7.79 ± 0.15 (norm -8.07) | 15.73 ± 1.55 (norm -5.50) | 17.50 ± 1.40 (norm -7.84) |
| FGVC-Aircraft | DINOv2 | Standard FM, T = 4 | 24.10 ± 0.74 (norm -9.81) | 28.68 ± 0.94 (norm -14.90) | 34.78 ± 0.46 (norm -22.80) |
| FGVC-Aircraft | DINOv2 | Standard FM, T = 12 | 25.15 ± 0.94 (norm -8.35) | 29.58 ± 0.98 (norm -13.70) | 37.63 ± 0.40 (norm -19.64) |
| FGVC-Aircraft | DINOv2 | Rolled-out FM, T = 4 | 18.30 ± 0.82 (norm -15.57) | 33.54 ± 0.82 (norm -9.71) | 30.07 ± 10.40 (norm -25.29) |
| FGVC-Aircraft | DINOv2 | Rolled-out FM, T = 12 | 24.40 ± 0.27 (norm -9.62) | 40.49 ± 0.86 (norm -3.24) | 32.09 ± 6.12 (norm -22.18) |

**Raw-feature version** of the full Stage-2 grid — the literal-spec formulation (ẑ₀ = z, no input L2-normalization), run under the identical protocol: same velocity network, recipe, prototypes, committed subsets, seeds, checkpoint rule and test split as the published normalized version. Top-1 (%) on the complete official test split; mean ± sample std over the same 3 runs. The parenthesis gives the raw-minus-normalized difference of setting means (negative = the normalized version is better). The Stage-1 baselines are identical for both versions (cosine classification is scale-invariant). The normalized version remains the primary published result (ADR 0007 §3).