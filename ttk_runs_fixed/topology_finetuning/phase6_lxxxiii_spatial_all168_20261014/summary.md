# Phase 6 full 168-sample spatial-locality results

Original SHA-pinned Phase 5E sparse matcher; 168 same-sample comparisons, five models, D0/D1, all eight radii.
All infinity-radius values regress to frozen LXXIX corrected GUDHI W2,2.
CNN/F1 rows reproduce the frozen Phase 5E spatial analysis.
Smaller constrained W2,2 is better; GT coverage is persistence-mass coverage, not position error.

| Model | r=4 constrained W22 | r=16 constrained W22 | r=32 constrained W22 | r=inf W22 | r=16 GT coverage |
|---|---:|---:|---:|---:|---:|
| cnn | 46.057465 | 35.218315 | 29.250278 | 24.552151 | 0.591272 |
| grad | 45.844483 | 31.008340 | 23.893099 | 18.306367 | 0.867886 |
| cv | 47.750779 | 31.572720 | 23.812289 | 17.875645 | 0.740639 |
| pers | 48.570148 | 31.841850 | 23.707846 | 17.484939 | 0.747282 |
| f1 | 48.504946 | 31.947250 | 23.921293 | 17.831897 | 0.710923 |

## F1 versus PERS-only: paired cost at all radii

| Radius (grid cells) | Mean F1-PERS cost | F1 lower / PERS lower / ties |
|---|---:|---:|
| 4.0 | -0.065202 | 86 / 82 / 0 |
| 8.0 | -0.061573 | 85 / 83 / 0 |
| 16.0 | +0.105400 | 78 / 90 / 0 |
| 32.0 | +0.213447 | 74 / 94 / 0 |
| 64.0 | +0.313022 | 38 / 130 / 0 |
| 96.0 | +0.322591 | 38 / 130 / 0 |
| 128.0 | +0.343584 | 33 / 135 / 0 |
| inf | +0.346958 | 34 / 134 / 0 |

**Limitations:** descriptive consecutive-hour benchmark, no independent seed uncertainty; F1 optimizer/checkpoint lineage not independently closed; no geographical km calibration and no new Phase 6 merge-tree distances.
