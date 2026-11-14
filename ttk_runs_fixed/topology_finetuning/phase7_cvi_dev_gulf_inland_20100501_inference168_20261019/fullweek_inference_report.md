# Phase 7 Part CVI — gulf_inland 2010-05-01 six-model development inference

Frozen region-week gulf_inland, 2010-05-01T00:00:00, exactly 168 consecutive UTC hours.
**One development region-week; do not interpret hours as independent statistical replicates.**

| Model | Mean hourly UV RMSE (m/s) ↓ | Mean hourly speed MAE (m/s) ↓ | XCVI anchor max difference (m/s) |
|---|---:|---:|---:|
| cnn | 0.629918 | 0.466572 | 0 |
| gan | 0.835749 | 0.640025 | 0 |
| grad | 0.459957 | 0.341700 | 0 |
| cv | 0.488481 | 0.364388 | 0 |
| pers | 0.492405 | 0.367549 | 0 |
| f1 | 0.500307 | 0.374144 | 0 |

Mean hourly metrics differ from pooled metrics. Hourly and pooled values are recorded in the sealed manifest.
The six pretrained/fine-tuned checkpoints and normalization are unchanged.
Topology, training, model selection, external locked geographic testing and 2012 replication were not performed.
Original pretrained PhIRE WTK geographical and temporal exposure remains unverified.
