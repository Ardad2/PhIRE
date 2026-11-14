# Phase 7 Part CVI — gulf_inland 2010-11-01 six-model development inference

Frozen region-week gulf_inland, 2010-11-01T00:00:00, exactly 168 consecutive UTC hours.
**One development region-week; do not interpret hours as independent statistical replicates.**

| Model | Mean hourly UV RMSE (m/s) ↓ | Mean hourly speed MAE (m/s) ↓ | XCVI anchor max difference (m/s) |
|---|---:|---:|---:|
| cnn | 0.623674 | 0.463298 | 0 |
| gan | 0.831226 | 0.635914 | 0 |
| grad | 0.465435 | 0.339534 | 0 |
| cv | 0.495974 | 0.364344 | 0 |
| pers | 0.502537 | 0.370119 | 0 |
| f1 | 0.508138 | 0.375152 | 0 |

Mean hourly metrics differ from pooled metrics. Hourly and pooled values are recorded in the sealed manifest.
The six pretrained/fine-tuned checkpoints and normalization are unchanged.
Topology, training, model selection, external locked geographic testing and 2012 replication were not performed.
Original pretrained PhIRE WTK geographical and temporal exposure remains unverified.
