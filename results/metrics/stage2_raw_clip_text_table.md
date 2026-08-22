| Dataset | Encoder | Head | K = 5 | K = 10 | full train split |
|---|---|---|---|---|---|
| DTD ‡ | CLIP RN50 | Standard FM, T = 4 | 51.37 ± 0.83 (norm +0.57) | 56.05 ± 0.30 (norm -0.37) | 62.00 ± 1.65 (norm -1.03) |
| DTD ‡ | CLIP RN50 | Standard FM, T = 12 | 51.54 ± 0.90 (norm +0.66) | 56.54 ± 0.46 (norm -0.16) | 62.62 ± 1.35 (norm -1.05) |
| DTD ‡ | CLIP RN50 | Rolled-out FM, T = 4 | 48.35 ± 1.45 (norm +0.71) | 55.62 ± 0.44 (norm +1.31) | 64.54 ± 0.48 (norm +1.33) |
| DTD ‡ | CLIP RN50 | Rolled-out FM, T = 12 | 48.35 ± 0.93 (norm +0.50) | 54.88 ± 0.72 (norm +0.14) | 62.30 ± 0.90 (norm +0.73) |
| FGVC-Aircraft ‡ | CLIP RN50 | Standard FM, T = 4 | 18.38 ± 0.26 (norm -0.82) | 21.16 ± 0.79 (norm -0.75) | 24.47 ± 1.80 (norm +0.54) |
| FGVC-Aircraft ‡ | CLIP RN50 | Standard FM, T = 12 | 18.59 ± 0.32 (norm -0.86) | 21.39 ± 0.76 (norm -0.64) | 24.45 ± 1.79 (norm -0.10) |
| FGVC-Aircraft ‡ | CLIP RN50 | Rolled-out FM, T = 4 | 19.19 ± 0.29 (norm +0.69) | 25.29 ± 0.23 (norm +2.25) | 34.69 ± 0.69 (norm +3.20) |
| FGVC-Aircraft ‡ | CLIP RN50 | Rolled-out FM, T = 12 | 19.02 ± 0.92 (norm +0.34) | 24.57 ± 0.23 (norm +1.83) | 33.28 ± 0.38 (norm +4.64) |

**Raw-feature version** of the full Stage-2 grid — the literal-spec formulation (ẑ₀ = z, no input L2-normalization), run under the identical protocol: same velocity network, recipe, prototypes, committed subsets, seeds, checkpoint rule and test split as the published normalized version. Top-1 (%) on the complete official test split; mean ± sample std over the same 3 runs. The parenthesis gives the raw-minus-normalized difference of setting means (negative = the normalized version is better). The Stage-1 baselines are identical for both versions (cosine classification is scale-invariant). The normalized version remains the primary published result (ADR 0007 §3).