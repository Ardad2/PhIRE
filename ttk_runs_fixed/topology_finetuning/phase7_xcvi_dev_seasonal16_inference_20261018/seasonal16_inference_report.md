# Phase 7 XCVI — Six-model 2010 development seasonal16 inference

**Descriptive engineering/seasonal pilot (16 fields, two per region-week), not a generalization benchmark.**

No locked external-test region predictions, hyperparameter tuning, or topology calculations.

| Model | 16-field UV RMSE (m/s) ↓ | 16-field speed MAE (m/s) ↓ | Max XC four-field prediction difference (m/s) |
|---|---:|---:|---:|
| cnn | 1.005078 | 0.694239 | 0 |
| gan | 1.260284 | 0.889043 | 0 |
| grad | 0.767223 | 0.505430 | 0 |
| cv | 0.836607 | 0.556942 | 0 |
| pers | 0.851250 | 0.569289 | 0 |
| f1 | 0.866208 | 0.579233 | 0 |

| Region | Season start UTC | Fields | Mean wind-speed GT (m/s) |
|---|---|---:|---:|
| gulf_inland | 2010-02-01 | 2 | 7.242058 |
| gulf_inland | 2010-05-01 | 2 | 7.444985 |
| gulf_inland | 2010-08-01 | 2 | 4.031404 |
| gulf_inland | 2010-11-01 | 2 | 8.643776 |
| southwest_borderland | 2010-02-01 | 2 | 4.177628 |
| southwest_borderland | 2010-05-01 | 2 | 7.121749 |
| southwest_borderland | 2010-08-01 | 2 | 4.616359 |
| southwest_borderland | 2010-11-01 | 2 | 7.285376 |

Metrics are computed over full 500×500 fields; 2010 two-hour region-week snippets are not independent hourly replications.
XC four-field checkpoint-output regression is enforced before publication, without modifying original XC outputs.
The original PhIRE pretrained-model WTK geography/time membership remains unknown.
