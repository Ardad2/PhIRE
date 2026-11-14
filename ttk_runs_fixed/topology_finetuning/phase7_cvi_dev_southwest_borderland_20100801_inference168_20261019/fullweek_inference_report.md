# Phase 7 Part CVI — southwest_borderland 2010-08-01 six-model development inference

Frozen region-week southwest_borderland, 2010-08-01T00:00:00, exactly 168 consecutive UTC hours.
**One development region-week; do not interpret hours as independent statistical replicates.**

| Model | Mean hourly UV RMSE (m/s) ↓ | Mean hourly speed MAE (m/s) ↓ | XCVI anchor max difference (m/s) |
|---|---:|---:|---:|
| cnn | 1.010869 | 0.753298 | 0 |
| gan | 1.291964 | 0.938344 | 0 |
| grad | 0.754845 | 0.527164 | 0 |
| cv | 0.816829 | 0.582699 | 0 |
| pers | 0.826490 | 0.594136 | 0 |
| f1 | 0.844202 | 0.607390 | 0 |

Mean hourly metrics differ from pooled metrics. Hourly and pooled values are recorded in the sealed manifest.
The six pretrained/fine-tuned checkpoints and normalization are unchanged.
Topology, training, model selection, external locked geographic testing and 2012 replication were not performed.
Original pretrained PhIRE WTK geographical and temporal exposure remains unverified.
