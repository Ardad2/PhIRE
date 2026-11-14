# Phase 7 Part CI — full-week, six-model development inference

Frozen Gulf inland, 2010 February 1–7 UTC, exactly 168 continuous hours.
**One development region-week; descriptive engineering and within-week results, not independent geographic generalization.**

| Model | Mean hourly UV RMSE (m/s) ↓ | Mean hourly speed MAE (m/s) ↓ | XCVI anchor max difference (m/s) |
|---|---:|---:|---:|
| cnn | 0.538996 | 0.401217 | 0 |
| gan | 0.742781 | 0.555596 | 0 |
| grad | 0.387756 | 0.286138 | 0 |
| cv | 0.415355 | 0.307329 | 0 |
| pers | 0.421153 | 0.312106 | 0 |
| f1 | 0.426202 | 0.315816 | 0 |

Mean hourly RMSE and pooled whole-week RMSE are different statistics; all hour-level values and pooled metrics are retained in JSON.
The six original checkpoint prefixes, normalization, geographic split and XCI outcomes are unchanged.
The original PhIRE pretraining geography/time membership remains unknown.
No 160×160 TTK persistence diagrams or GUDHI/spatial-locality distances are computed in Part CI.
No external locked regions, model selection, hyperparameter tuning, or training.
