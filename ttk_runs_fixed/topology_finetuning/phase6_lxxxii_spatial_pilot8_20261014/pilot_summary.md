# Phase 6 locality pilot (8 predeclared samples)

Validated the original Phase 5E sparse-savings locality solver, with preserved feature coordinates and radius grid.
CNN/F1 old spatial-regression rows: 128; matched LXXIX unconstrained W22 cases: 40.
These are **pilot-only descriptive results**, not the all-168 spatial result.

| Model | r=4 mean constrained W22 | r=16 mean constrained W22 | r=inf mean W22 | r=16 mean GT coverage |
|---|---:|---:|---:|---:|
| cnn | 48.75961 | 36.31615 | 25.20037 | 0.61297 |
| grad | 47.64018 | 32.15893 | 19.26840 | 0.88809 |
| cv | 50.40156 | 32.99659 | 19.08425 | 0.75456 |
| pers | 51.17329 | 33.37535 | 18.75371 | 0.77340 |
| f1 | 50.53062 | 33.28444 | 18.96689 | 0.73128 |

**Caveats:** This pilot is not a statistical independence test; no kilometer calibration, no merge-tree comparison, and no F1 training-lineage closure.
