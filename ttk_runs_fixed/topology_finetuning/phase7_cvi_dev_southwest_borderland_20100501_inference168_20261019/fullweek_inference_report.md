# Phase 7 Part CVI — southwest_borderland 2010-05-01 six-model development inference

Frozen region-week southwest_borderland, 2010-05-01T00:00:00, exactly 168 consecutive UTC hours.
**One development region-week; do not interpret hours as independent statistical replicates.**

| Model | Mean hourly UV RMSE (m/s) ↓ | Mean hourly speed MAE (m/s) ↓ | XCVI anchor max difference (m/s) |
|---|---:|---:|---:|
| cnn | 1.169758 | 0.900178 | 0 |
| gan | 1.498365 | 1.154610 | 0 |
| grad | 0.896735 | 0.660963 | 0 |
| cv | 0.974496 | 0.733498 | 0 |
| pers | 0.989912 | 0.752140 | 0 |
| f1 | 1.010148 | 0.767211 | 0 |

Mean hourly metrics differ from pooled metrics. Hourly and pooled values are recorded in the sealed manifest.
The six pretrained/fine-tuned checkpoints and normalization are unchanged.
Topology, training, model selection, external locked geographic testing and 2012 replication were not performed.
Original pretrained PhIRE WTK geographical and temporal exposure remains unverified.
