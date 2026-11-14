# Phase 7 Part CIV — 168-hour development spatial-locality analysis

Southwest Borderland, 2010-11 1–7, 2010 UTC; 168 contiguous hourly fields × six frozen models.
**One development region-week. Consecutive hours/days are temporally correlated, not independent generalization replications.**

Unchanged original Phase5E sparse matching on TTK canonical top-left 160×160 speed persistence diagrams.
Radii **4, 8, 16, 32, 64, 96, 128, ∞ are WTK grid cells, not kilometres**.

Lower constrained W₂,₂ means better persistence matching; GT persistence-mass coverage is separate (higher is better).

| Model | r4 W22 ↓ | r16 W22 ↓ | r32 W22 ↓ | r∞ W22 ↓ | GT coverage r16 ↑ |
|---|---:|---:|---:|---:|---:|
| cnn | 25.704016 | 20.428265 | 16.990502 | 12.311039 | 0.531256 |
| gan | 34.575616 | 26.502420 | 21.663822 | 15.668255 | 0.821151 |
| grad | 24.514412 | 18.039750 | 14.599421 | 10.145199 | 0.759831 |
| cv | 25.583152 | 18.756379 | 14.983563 | 10.047159 | 0.685720 |
| pers | 25.684342 | 18.900377 | 15.087287 | 10.083989 | 0.702724 |
| f1 | 25.845294 | 18.938477 | 15.042700 | 10.073751 | 0.677498 |

## Daily descriptive summaries (24 consecutive hours per UTC date)

| UTC day | Arm | N | r4 W22 ↓ | r16 W22 ↓ | r∞ W22 ↓ | GT coverage r16 ↑ |
|---|---|---:|---:|---:|---:|---:|
| 2010-11-01 | cnn | 24 | 31.553070 | 24.985834 | 14.635893 | 0.566651 |
| 2010-11-01 | gan | 24 | 41.092114 | 31.637379 | 17.609065 | 0.809566 |
| 2010-11-01 | grad | 24 | 29.992471 | 22.078224 | 12.923442 | 0.734131 |
| 2010-11-01 | cv | 24 | 31.555148 | 22.873768 | 12.466202 | 0.677090 |
| 2010-11-01 | pers | 24 | 31.590640 | 22.715591 | 12.494942 | 0.693312 |
| 2010-11-01 | f1 | 24 | 31.713530 | 22.587611 | 12.436433 | 0.682844 |
| 2010-11-02 | cnn | 24 | 32.147418 | 25.468258 | 14.405962 | 0.555072 |
| 2010-11-02 | gan | 24 | 41.308366 | 31.438769 | 17.505127 | 0.808863 |
| 2010-11-02 | grad | 24 | 30.052483 | 22.158363 | 12.093672 | 0.758629 |
| 2010-11-02 | cv | 24 | 31.420153 | 23.104257 | 11.581252 | 0.682181 |
| 2010-11-02 | pers | 24 | 31.808420 | 23.454438 | 11.626405 | 0.715208 |
| 2010-11-02 | f1 | 24 | 32.031326 | 23.350913 | 11.466820 | 0.686217 |
| 2010-11-03 | cnn | 24 | 32.017550 | 24.642589 | 13.918234 | 0.556092 |
| 2010-11-03 | gan | 24 | 41.795661 | 31.993562 | 18.747469 | 0.806226 |
| 2010-11-03 | grad | 24 | 31.087540 | 22.650292 | 12.047355 | 0.754437 |
| 2010-11-03 | cv | 24 | 32.093532 | 23.345196 | 11.995443 | 0.668316 |
| 2010-11-03 | pers | 24 | 32.137552 | 23.503638 | 11.685620 | 0.689024 |
| 2010-11-03 | f1 | 24 | 32.406969 | 23.977773 | 11.954472 | 0.662395 |
| 2010-11-04 | cnn | 24 | 28.924226 | 23.090026 | 12.835768 | 0.538471 |
| 2010-11-04 | gan | 24 | 37.865601 | 29.732310 | 16.713884 | 0.798707 |
| 2010-11-04 | grad | 24 | 27.262681 | 20.839020 | 10.758295 | 0.739010 |
| 2010-11-04 | cv | 24 | 29.003461 | 22.205494 | 10.631281 | 0.649546 |
| 2010-11-04 | pers | 24 | 29.259592 | 22.326768 | 10.833094 | 0.656794 |
| 2010-11-04 | f1 | 24 | 29.351031 | 22.386598 | 10.778228 | 0.638021 |
| 2010-11-05 | cnn | 24 | 20.064915 | 15.678739 | 10.600610 | 0.544476 |
| 2010-11-05 | gan | 24 | 30.210358 | 23.417317 | 16.451080 | 0.855518 |
| 2010-11-05 | grad | 24 | 18.752438 | 13.155672 | 7.797881 | 0.789791 |
| 2010-11-05 | cv | 24 | 19.755735 | 13.832198 | 8.076092 | 0.713807 |
| 2010-11-05 | pers | 24 | 19.907278 | 14.068620 | 8.263881 | 0.723164 |
| 2010-11-05 | f1 | 24 | 19.892445 | 14.094488 | 8.221402 | 0.691004 |
| 2010-11-06 | cnn | 24 | 16.405202 | 13.463361 | 9.573531 | 0.477279 |
| 2010-11-06 | gan | 24 | 23.672773 | 17.539185 | 10.928192 | 0.840195 |
| 2010-11-06 | grad | 24 | 15.982247 | 11.430130 | 7.324874 | 0.774453 |
| 2010-11-06 | cv | 24 | 16.391207 | 11.618924 | 7.461230 | 0.714176 |
| 2010-11-06 | pers | 24 | 16.277643 | 11.854004 | 7.516318 | 0.723096 |
| 2010-11-06 | f1 | 24 | 16.570281 | 11.798150 | 7.525697 | 0.691892 |
| 2010-11-07 | cnn | 24 | 18.815729 | 15.669045 | 10.207273 | 0.480748 |
| 2010-11-07 | gan | 24 | 26.084443 | 19.758415 | 11.722967 | 0.828982 |
| 2010-11-07 | grad | 24 | 18.471024 | 13.966546 | 8.070872 | 0.768365 |
| 2010-11-07 | cv | 24 | 18.862829 | 14.314816 | 8.118612 | 0.694921 |
| 2010-11-07 | pers | 24 | 18.809273 | 14.379578 | 8.167663 | 0.718468 |
| 2010-11-07 | f1 | 24 | 18.951474 | 14.373804 | 8.133206 | 0.690111 |

## Scientific and reproducibility boundaries

- All 1,008 infinite-radius W22 model-to-GT values must match the sealed Part CIII corrected GUDHI output.
- All 96 aggregate + 192 dimension rows for historical week hours 0 and 84 must reproduce Part XCIX.
- Canonical zero/near-zero persistence pairs use the frozen original Phase5E tolerance 1e-10 with excluded count validation.
- All 1,176 diagram coordinate sets are validated against their corrected numerical birth/death features.
- GT persistence-mass coverage measures matched mass, not correct spatial positions or physical wind structure.
- This is not topology on full 500×500 fields and does not include merge trees or training.
- No external locked-region predictions or new hyperparameter/model selection. Original pretrained WTK exposure unknown.
- Verifier repeats the same original Phase5E algorithm; it is not an independently implemented solver.

