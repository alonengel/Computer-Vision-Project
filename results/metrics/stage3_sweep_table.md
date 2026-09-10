| Dataset | Strategy | Configuration | Val top-1 (%) | Checkpoint epoch | Fallback |
|---|---|---|---|---|---|
| DTD | Rolled | lam=0 | 49.79 | 1 | 0 |
| DTD | Rolled | lam=1 | 50.90 | 110 | 0 |
| DTD | Rolled | lam=10 | 50.59 | 103 | 0 |
| DTD | Rolled | lam=100 | 50.74 | 1 | 0 |
| DTD | Guided | beta=0.25,m=1 | 51.97 | 167 | 0 |
| DTD | Guided | beta=0.25,m=3 | 51.76 | 156 | 0 |
| DTD | Guided | beta=0.5,m=1 | 51.86 | 157 | 0 |
| DTD | Guided | beta=0.5,m=3 | 51.81 | 152 | 0 |
| DTD | Guided | beta=1.0,m=1 | 51.91 | 145 | 0 |
| DTD | Guided | beta=1.0,m=3 | 51.70 | 10 | 0 |
| FGVC-Aircraft | Rolled | lam=0 | 53.74 | 16 | 0 |
| FGVC-Aircraft | Rolled | lam=1 | 53.95 | 86 | 0 |
| FGVC-Aircraft | Rolled | lam=10 | 53.05 | 184 | 0 |
| FGVC-Aircraft | Rolled | lam=100 | 52.72 | 156 | 0 |
| FGVC-Aircraft | Guided | beta=0.25,m=1 | 55.06 | 42 | 0 |
| FGVC-Aircraft | Guided | beta=0.25,m=3 | 55.78 | 17 | 0 |
| FGVC-Aircraft | Guided | beta=0.5,m=1 | 55.54 | 62 | 0 |
| FGVC-Aircraft | Guided | beta=0.5,m=3 | 56.11 | 43 | 0 |
| FGVC-Aircraft | Guided | beta=1.0,m=1 | 55.78 | 34 | 0 |
| FGVC-Aircraft | Guided | beta=1.0,m=3 | 56.11 | 17 | 0 |

Seed-0 **validation** accuracy of the full pipeline per swept configuration (test untouched during selection). Winners per dataset by highest validation accuracy (ties → lowest validation CE → grid order). Fallback level 0 = the default recipe (ADR 0008 §7).