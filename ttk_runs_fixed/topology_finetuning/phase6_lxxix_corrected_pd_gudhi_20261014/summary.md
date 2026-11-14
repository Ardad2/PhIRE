# Phase 6 full corrected persistence-diagram results

168 aligned benchmark fields; frozen canonical Phase 5E GT / CNN / F1 diagrams;
three new extracted SR arms. Recalculated with canonical TTK point parser and GUDHI.
Lower values are better. All statistics descriptive (consecutive hourly samples).

| Arm | dB mean | W2inf mean | W22 mean |
|---|---:|---:|---:|
| cnn | 3.124108 | 19.152145 | 24.552151 |
| grad | 2.562506 | 14.889700 | 18.306367 |
| cv | 2.188473 | 14.407073 | 17.875645 |
| pers | 2.196119 | 14.153230 | 17.484939 |
| f1 | 2.149744 | 14.325676 | 17.831897 |

## 2×2 factorial means

Negative main effects improve lower-is-better distance.
Interaction is signed and does not alone imply improvement.

| Metric | CV main effect | PERS main effect | Interaction |
|---|---:|---:|---:|
| dB | -0.210205 | -0.202558 | 0.327658 |
| W2inf | -0.155090 | -0.408934 | 0.655073 |
| W22 | -0.041882 | -0.432587 | 0.777680 |

## Caveats

- No merge-tree or spatial-locality metric is computed here.
- No formal independence assumption across consecutive hourly samples.
- Confirm F1 source continuity with the 2×2 experiment documentation before manuscript attribution.
