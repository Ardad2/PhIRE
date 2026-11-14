# Phase 7 XCIV — development-only canonical spatial locality

Original Phase5E sparse solver on four 160×160 top-left development fields.
The radii are **WTK grid cells**, not kilometers. Lower constrained W22 is better;
GT persistence-mass coverage is a separate higher-is-better statistic.

**Exploratory four-field numerical pilot, not a geographic-generalization estimate.**

| Arm | r4 W22 ↓ | r16 W22 ↓ | r32 W22 ↓ | r∞ W22 ↓ | GT coverage r16 ↑ |
|---|---:|---:|---:|---:|---:|
| cnn | 15.872628 | 13.065990 | 11.488728 | 9.560749 | 0.521896 |
| gan | 22.701018 | 16.753153 | 14.003156 | 11.388065 | 0.823593 |
| grad | 16.165423 | 11.531220 | 9.632986 | 7.412320 | 0.801454 |
| cv | 16.302017 | 12.256537 | 9.938167 | 7.806714 | 0.731901 |
| pers | 16.346778 | 11.956780 | 10.175161 | 7.919260 | 0.738989 |
| f1 | 16.568133 | 12.269268 | 10.029004 | 7.905475 | 0.709210 |

## Boundaries

- All six model checkpoints were fixed before this evaluation.
- No locked test-region model predictions or topology scores accessed.
- Coordinates validated against original corrected numeric persistence pairs.
- Same Phase5E optimizer and radius definitions as the frozen historical analysis.
- Infinity-radius aggregate values numerically anchored to the frozen XCIII W22.
- No terrain geodesic conversion, independence assumption across hours, or model selection.
- Four development fields alone cannot test robustness across complete seasonal weeks.

