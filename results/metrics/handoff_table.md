| Dataset | Selected encoder (by validation) | Stage-2 prototype target (branch A) | Validation headroom (probe − prototypes, full) | Stage-3 baseline: linear probe, full split (test) |
|---|---|---|---|---|
| DTD | ResNet-18 | class-mean prototypes $\mu_c$ of the selected training subset, ResNet-18 features | 60.71 − 55.37 = 5.34 pts | 62.84 ± 0.45 |
| FGVC-Aircraft | DINOv2 | class-mean prototypes $\mu_c$ of the selected training subset, DINOv2 features | 68.62 − 32.91 = 35.70 pts | 67.21 ± 0.11 |
| Oxford Flowers-102 ‡ | ResNet-18 | class-mean prototypes $\mu_c$ of the selected training subset, ResNet-18 features | 86.50 − 78.04 = 8.46 pts | 83.28 ± 0.11 |

Encoder selection: highest mean **validation** accuracy of the full-split linear probe (`runs.csv`, `val_acc`); test accuracy plays no role. Validation headroom: full-split probe validation accuracy minus full-split image-prototype validation accuracy on the same encoder — the gap a Stage-2 Flow-Matching decision layer has room to close, measured without touching the test split. The Stage-3 baseline column is the one-time test read-out published in the accuracy table. ‡ = beyond our selected pair.