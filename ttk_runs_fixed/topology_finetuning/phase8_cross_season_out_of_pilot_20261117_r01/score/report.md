# Phase 8 — Cross-season, out-of-pilot development evaluation


Training steps: 0,1,2,3,0,1,2,3; same original checkpoint, two TF1 Adam sessions.


**Not an independent test:** the 4 evaluation examples remain in the 2688-field training pool.


| Index (season) | CNN W22 | Control W22 | Dynamic W22 | Dynamic–control |
|---|---:|---:|---:|---:|
| 672 (spring_1) | 14.454486 | 14.364098 | 14.204610 | -0.159488 |
| 1344 (summer_1) | 13.128504 | 13.055755 | 13.018161 | -0.037594 |
| 2016 (fall_1) | 18.535270 | 18.466405 | 18.360566 | -0.105839 |
| 2520 (fall_4) | 22.115530 | 22.025126 | 21.820440 | -0.204686 |


Mean metrics (lower is better):

| Arm | W22 | L_uv | L_grad | Speed crop MAE (m/s) |
|---|---:|---:|---:|---:|
| cnn | 17.058448 | 0.02119180 | 0.355866 | 0.806571 |
| control | 16.977846 | 0.02109498 | 0.354791 | 0.806240 |
| dynamic | 16.850944 | 0.02120016 | 0.353994 | 0.808665 |


Dynamic W22 better on 4/4 fields.

Audit: replayed all 8 original steps with predictions compared to frozen artifacts; no model checkpoint written; all diagrams recomputed from outputs.
