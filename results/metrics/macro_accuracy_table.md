| Dataset | Encoder | Top-1 (%) | Balanced / macro (%) |
|---|---|---|---|
| DTD | ResNet-18 | 63.35 | 63.35 |
| FGVC-Aircraft | DINOv2 | 67.09 | 67.07 |
| Oxford Flowers-102 ‡ | ResNet-18 | 83.22 | 85.38 |

Single full-split linear-probe run (initialization seed 0) on each dataset's best-by-validation encoder. The two columns coincide when the test split is balanced (DTD, FGVC-Aircraft) and diverge for Flowers-102, whose official test split is class-imbalanced.