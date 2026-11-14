# Phase 7 Part CVI — gulf_inland 2010-08-01 six-model development inference

Frozen region-week gulf_inland, 2010-08-01T00:00:00, exactly 168 consecutive UTC hours.
**One development region-week; do not interpret hours as independent statistical replicates.**

| Model | Mean hourly UV RMSE (m/s) ↓ | Mean hourly speed MAE (m/s) ↓ | XCVI anchor max difference (m/s) |
|---|---:|---:|---:|
| cnn | 0.633128 | 0.435966 | 0 |
| gan | 0.853928 | 0.611698 | 0 |
| grad | 0.436242 | 0.293447 | 0 |
| cv | 0.468839 | 0.319747 | 0 |
| pers | 0.473966 | 0.325034 | 0 |
| f1 | 0.481414 | 0.330441 | 0 |

Mean hourly metrics differ from pooled metrics. Hourly and pooled values are recorded in the sealed manifest.
The six pretrained/fine-tuned checkpoints and normalization are unchanged.
Topology, training, model selection, external locked geographic testing and 2012 replication were not performed.
Original pretrained PhIRE WTK geographical and temporal exposure remains unverified.
