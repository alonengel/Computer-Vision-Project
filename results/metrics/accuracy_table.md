| Dataset | Encoder | Baseline | K = 5 | K = 10 | full train split |
|---|---|---|---|---|---|
| DTD | ResNet-18 | Linear probe | 45.60 ± 0.64 | 54.31 ± 0.28 | 62.84 ± 0.45 |
| DTD | ResNet-18 | Image prototypes | 46.51 ± 0.64 | 53.37 ± 1.04 | 58.78 |
| DTD | CLIP RN50 | Zero-shot CLIP | — | — | 39.79 |

Top-1 accuracy (%) on the complete official test split. 5-shot / 10-shot: mean ± std over 3 training-subset seeds; full linear probe: mean ± std over 3 initialization seeds; full image prototypes and zero-shot CLIP are single deterministic runs. Zero-shot CLIP uses no labeled training images, so it has one value only. ‡ = beyond the spec's required pair of datasets.