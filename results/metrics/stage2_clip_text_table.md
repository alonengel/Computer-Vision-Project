| Dataset | Encoder | Head | K = 5 | K = 10 | full train split |
|---|---|---|---|---|---|
| DTD ‡ | CLIP RN50 | Stage-1 Zero-shot CLIP (reference, K-independent) | 39.79 | 39.79 | 39.79 |
| DTD ‡ | CLIP RN50 | Standard FM, T = 4 | 50.80 ± 0.53 (Δ +11.01 ± 0.53) | 56.42 ± 0.84 (Δ +16.63 ± 0.84) | 63.03 ± 0.75 (Δ +23.24 ± 0.75) |
| DTD ‡ | CLIP RN50 | Standard FM, T = 12 | **50.89 ± 0.61 (Δ +11.10 ± 0.61)** | **56.70 ± 0.77 (Δ +16.91 ± 0.77)** | **63.67 ± 0.70 (Δ +23.88 ± 0.70)** |
| DTD ‡ | CLIP RN50 | Rolled-out FM, T = 4 | 47.64 ± 1.24 (Δ +7.85 ± 1.24) | 54.31 ± 0.78 (Δ +14.52 ± 0.78) | 63.21 ± 0.11 (Δ +23.42 ± 0.11) |
| DTD ‡ | CLIP RN50 | Rolled-out FM, T = 12 | 47.85 ± 1.01 (Δ +8.07 ± 1.01) | 54.73 ± 0.94 (Δ +14.95 ± 0.94) | 61.58 ± 1.23 (Δ +21.79 ± 1.23) |
| FGVC-Aircraft ‡ | CLIP RN50 | Stage-1 Zero-shot CLIP (reference, K-independent) | 17.04 | 17.04 | 17.04 |
| FGVC-Aircraft ‡ | CLIP RN50 | Standard FM, T = 4 | 19.20 ± 0.70 (Δ +2.16 ± 0.70) | 21.91 ± 0.62 (Δ +4.87 ± 0.62) | 23.93 ± 0.33 (Δ +6.89 ± 0.33) |
| FGVC-Aircraft ‡ | CLIP RN50 | Standard FM, T = 12 | **19.45 ± 0.82 (Δ +2.41 ± 0.82)** | 22.03 ± 0.36 (Δ +4.99 ± 0.36) | 24.55 ± 0.41 (Δ +7.51 ± 0.41) |
| FGVC-Aircraft ‡ | CLIP RN50 | Rolled-out FM, T = 4 | 18.50 ± 0.72 (Δ +1.46 ± 0.72) | **23.04 ± 0.38 (Δ +6.00 ± 0.38)** | **31.49 ± 1.03 (Δ +14.45 ± 1.03)** |
| FGVC-Aircraft ‡ | CLIP RN50 | Rolled-out FM, T = 12 | 18.68 ± 0.36 (Δ +1.64 ± 0.36) | 22.74 ± 0.45 (Δ +5.70 ± 0.45) | 28.64 ± 0.21 (Δ +11.60 ± 0.21) |

‡ Extension beyond the specification (the spec's Stage-2 branch is the image-prototype branch selected in Stage 1, ADR 0006). FM here transports **CLIP RN50 image features** toward the fixed CLIP **text** prototypes. Unlike the Stage-1 zero-shot reference, FM training consumes K labeled images per class — these rows are *supervised transport on CLIP features*, never "zero-shot". Δ therefore answers "does supervised transport toward text prototypes beat zero-shot classification?", **not** "does the FM layer help?" — only the image-prototype branch isolates the FM effect, because its baseline uses the identical labeled subset. The reference value is a single deterministic run, so Δ carries no pairing; at K ∈ {5, 10} the spread over the 3 subset seeds is reported on the accuracy.