# Phase 7 Part CVI — southwest_borderland 2010-11-01 six-model development inference

Frozen region-week southwest_borderland, 2010-11-01T00:00:00, exactly 168 consecutive UTC hours.
**One development region-week; do not interpret hours as independent statistical replicates.**

| Model | Mean hourly UV RMSE (m/s) ↓ | Mean hourly speed MAE (m/s) ↓ | XCVI anchor max difference (m/s) |
|---|---:|---:|---:|
| cnn | 1.260917 | 0.988358 | 0 |
| gan | 1.575665 | 1.194236 | 0 |
| grad | 0.980426 | 0.721097 | 0 |
| cv | 1.089592 | 0.815928 | 0 |
| pers | 1.114296 | 0.841241 | 0 |
| f1 | 1.134609 | 0.854566 | 0 |

Mean hourly metrics differ from pooled metrics. Hourly and pooled values are recorded in the sealed manifest.
The six pretrained/fine-tuned checkpoints and normalization are unchanged.
Topology, training, model selection, external locked geographic testing and 2012 replication were not performed.
Original pretrained PhIRE WTK geographical and temporal exposure remains unverified.
