# Phase 8 — Original Mid-Atlantic 168-field benchmark: eight-step pilot

**DEVELOPMENT COMPARISON, NOT MANUSCRIPT-SCALE TRAINING OR EXTERNAL TEST.**

Two independent Adam arms were restored and replayed from the verified pilot:
`L_uv + 0.05 L_grad` versus `L_uv + 0.05 L_grad + 1e-5 * W22²`.
Only 4 training fields, 8 optimizer updates per arm. All 168 ORIGINAL benchmark fields were evaluation-only.

| Arm | PD W22 | W2inf | dB | L_uv | L_grad | Crop speed MAE m/s |
|---|---:|---:|---:|---:|---:|---:|
| cnn | 24.552151 | 19.152145 | 3.124109 | 0.031759 | 0.575292 | 1.039206 |
| control | 24.447153 | 19.081044 | 3.107711 | 0.031607 | 0.573553 | 1.037166 |
| dynamic | 24.258077 | 18.952863 | 3.094715 | 0.031746 | 0.571685 | 1.039384 |

| Arm | r4 W22 | r16 W22 | r32 W22 | r∞ W22 | r16 GT coverage | r16 SR coverage |
|---|---:|---:|---:|---:|---:|---:|
| cnn | 46.055699 | 35.215055 | 29.248896 | 24.552151 | 0.591364 | 0.842619 |
| control | 46.087170 | 35.122493 | 29.182348 | 24.447153 | 0.593823 | 0.843389 |
| dynamic | 46.175792 | 35.086606 | 29.024586 | 24.258077 | 0.598228 | 0.843550 |

Dynamic improved unconstrained W22 on 168 of 168 fields.
Mean dynamic-control W22 difference: -0.189076.

| Consecutive 24-hour block | Mean dynamic-control W22 | Dynamic improved hours |
|---|---:|---:|
| 1 | -0.169016 | 24/24 |
| 2 | -0.092096 | 24/24 |
| 3 | -0.167285 | 24/24 |
| 4 | -0.275246 | 24/24 |
| 5 | -0.304122 | 24/24 |
| 6 | -0.223511 | 24/24 |
| 7 | -0.092257 | 24/24 |

## Scope cautions

- NOT comparable as matched-training-scale experiment to 2688-sample/3-epoch manuscript arms
- Eight updates on FOUR training fields only
- Benchmark used in prior model/hyperparameter selection; not an external held-out test
- Time-adjacent hours are correlated; seven daily blocks, not 168 independent events
- Top-left 160x160 scalar-speed crop for PD and locality, not full 500x500
- L_uv and L_grad refer to normalized model loss; speed_crop_MAE_mps is 160-crop
- No Phase7 regional test fields were loaded; locked geographies untouched
- Pretraining independence not verified
