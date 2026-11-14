# Phase 7 Part CVI — southwest_borderland 2010-02-01 six-model development inference

Frozen region-week southwest_borderland, 2010-02-01T00:00:00, exactly 168 consecutive UTC hours.
**One development region-week; do not interpret hours as independent statistical replicates.**

| Model | Mean hourly UV RMSE (m/s) ↓ | Mean hourly speed MAE (m/s) ↓ | XCVI anchor max difference (m/s) |
|---|---:|---:|---:|
| cnn | 1.066926 | 0.826463 | 0 |
| gan | 1.389807 | 1.052985 | 0 |
| grad | 0.815951 | 0.598653 | 0 |
| cv | 0.889036 | 0.664562 | 0 |
| pers | 0.902097 | 0.679437 | 0 |
| f1 | 0.923134 | 0.695386 | 0 |

Mean hourly metrics differ from pooled metrics. Hourly and pooled values are recorded in the sealed manifest.
The six pretrained/fine-tuned checkpoints and normalization are unchanged.
Topology, training, model selection, external locked geographic testing and 2012 replication were not performed.
Original pretrained PhIRE WTK geographical and temporal exposure remains unverified.
