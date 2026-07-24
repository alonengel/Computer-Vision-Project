| Dataset | Encoder | K | Prototypes − probe (paired) | Seeds favouring prototypes |
|---|---|---|---|---|
| DTD | ResNet-18 | 5 | +0.90 ± 0.14 | 3 / 3 |
| DTD | ResNet-18 | 10 | -0.94 ± 0.83 | 0 / 3 |
| FGVC-Aircraft | ResNet-18 | 5 | -3.86 ± 0.95 | 0 / 3 |
| FGVC-Aircraft | ResNet-18 | 10 | -7.51 ± 1.11 | 0 / 3 |
| FGVC-Aircraft | DINOv2 | 5 | -13.36 ± 0.81 | 0 / 3 |
| FGVC-Aircraft | DINOv2 | 10 | -23.49 ± 0.94 | 0 / 3 |
| Oxford Flowers-102 ‡ | ResNet-18 | 5 | -5.39 ± 0.95 | 0 / 3 |
| Oxford Flowers-102 ‡ | ResNet-18 | 10 | -8.00 ± 0.00 | 0 / 3 |

Mean ± sample standard deviation (ddof = 1) of the per-seed difference in top-1 accuracy, both heads trained on identical subset indices. Positive favours image prototypes. The `full` setting is omitted because the prototype head runs once there and the probe varies only by initialization, so the runs are not paired.