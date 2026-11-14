# Phase 7 — eight development region-week summary, 2010

Two development regions × four frozen 168-hour seasonal windows; eight equal-weighted region-week means.
One year and only two geographic regions: **development validation, not external generalization**.
Original top-left 160×160 speed crop and frozen eight grid-cell radii; no held-out geographic tests.

| Model | UV RMSE ↓ | Speed MAE ↓ | PD W22 ↓ | r4 W22 ↓ | r16 W22 ↓ | r∞ W22 ↓ | r16 GT coverage ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| cnn | 0.866773 | 0.654419 | 10.006731 | 17.676943 | 14.552387 | 10.006731 | 0.454517 |
| gan | 1.127436 | 0.847926 | 11.012592 | 24.211405 | 18.429210 | 11.012592 | 0.809042 |
| grad | 0.649668 | 0.471087 | 7.915321 | 17.408841 | 12.760236 | 7.915321 | 0.794140 |
| cv | 0.704825 | 0.519062 | 8.011084 | 17.765592 | 13.144159 | 8.011084 | 0.730265 |
| pers | 0.715357 | 0.530220 | 8.015590 | 17.816061 | 13.192121 | 8.015590 | 0.742817 |
| f1 | 0.728519 | 0.540013 | 8.107298 | 17.865450 | 13.276252 | 8.107298 | 0.715569 |

Consecutive hourly fields within each week are correlated.
No pretrained-model WTK exposure assertions, independent solver, or sample-level significance claims.
