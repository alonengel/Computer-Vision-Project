| Dataset | Encoder | Baseline | K = 5 | K = 10 | full train split | no training images |
|---|---|---|---|---|---|---|
| DTD | ResNet-18 | Linear probe | 45.60 ± 0.64 | **54.31 ± 0.28** | **62.84 ± 0.45** | — |
| DTD | ResNet-18 | Image prototypes | **46.51 ± 0.64** | 53.37 ± 1.04 | 58.78 | — |
| DTD | CLIP RN50 | Zero-shot CLIP | — | — | — | 39.79 |
| FGVC-Aircraft | ResNet-18 | Linear probe | 19.90 ± 1.72 | 27.36 ± 0.83 | 36.62 ± 0.27 | — |
| FGVC-Aircraft | ResNet-18 | Image prototypes | 16.04 ± 0.83 | 19.85 ± 0.47 | 25.20 | — |
| FGVC-Aircraft | DINOv2 | Linear probe | **36.47 ± 0.70** | **51.30 ± 0.95** | **67.21 ± 0.11** | — |
| FGVC-Aircraft | DINOv2 | Image prototypes | 23.11 ± 1.36 | 27.80 ± 1.17 | 34.41 | — |
| FGVC-Aircraft | CLIP RN50 | Zero-shot CLIP | — | — | — | 17.04 |
| Oxford Flowers-102 ‡ | ResNet-18 | Linear probe | **75.58 ± 0.84** | **83.22 ± 0.00** | **83.28 ± 0.11** | — |
| Oxford Flowers-102 ‡ | ResNet-18 | Image prototypes | 70.19 ± 0.14 | 75.22 ± 0.00 | 75.22 | — |
| Oxford Flowers-102 ‡ | CLIP RN50 | Zero-shot CLIP | — | — | — | 63.64 |

**Bold** marks the best supervised configuration in each (dataset, training-set size) column; zero-shot CLIP is excluded from that comparison because it uses no training images. Top-1 accuracy (%) on the complete official test split. 5-shot / 10-shot: mean ± std over 3 training-subset seeds; full linear probe: mean ± std over 3 initialization seeds; full image prototypes and zero-shot CLIP are single deterministic runs. Zero-shot CLIP uses no labeled training images, so it has one value only. ‡ = beyond the spec's required pair of datasets. A standard deviation of exactly 0.00 at Flowers-102 K = 10 is not a rounding artifact: that dataset's official training split holds exactly 10 images per class, so all three 10-shot subsets are the same set of images.