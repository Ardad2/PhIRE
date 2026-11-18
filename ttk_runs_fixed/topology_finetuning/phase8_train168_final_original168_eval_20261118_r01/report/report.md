# Phase 8 — Three-arm 168-training-field model comparison on original 168 benchmark

**Development comparison, not an independent geographic test.**

All three fine-tuning arms trained on the SAME separate 168 selected fields from the 2688 pool,
three epochs, batch size 1, 504 optimizer updates, identical pretrained CNN initialization.

| Arm | PD W22 ↓ | W2inf ↓ | dB ↓ | L_uv ↓ | L_grad ↓ | Crop speed MAE m/s ↓ |
|---|---:|---:|---:|---:|---:|---:|
| cnn | 24.552151 | 19.152145 | 3.124109 | 0.031759 | 0.575292 | 1.039206 |
| grad | 24.781492 | 19.339899 | 3.120108 | 0.027496 | 0.555334 | 0.974094 |
| fixed | 23.482851 | 18.451870 | 3.016185 | 0.029381 | 0.551447 | 1.006403 |
| dynamic | 21.772240 | 17.305430 | 2.944325 | 0.029072 | 0.537125 | 1.005016 |

| Arm | r4 W22 ↓ | r16 W22 ↓ | r32 W22 ↓ | r∞ W22 ↓ | r16 GT coverage ↑ | r16 SR coverage ↑ |
|---|---:|---:|---:|---:|---:|---:|
| cnn | 46.055699 | 35.215055 | 29.248896 | 24.552151 | 0.591364 | 0.842619 |
| grad | 45.440198 | 34.843391 | 29.161108 | 24.781492 | 0.639673 | 0.804767 |
| fixed | 46.170854 | 34.514739 | 28.351015 | 23.482851 | 0.652285 | 0.806129 |
| dynamic | 46.547853 | 33.578672 | 27.005997 | 21.772240 | 0.708153 | 0.792104 |

## Paired unconstrained W22 comparisons

| Comparison (a − b) | Mean difference ↓ | a better hours /168 | a worse hours /168 |
|---|---:|---:|---:|
| fixed_minus_grad | -1.298642 | 166/168 | 2/168 |
| dynamic_minus_grad | -3.009252 | 168/168 | 0/168 |
| dynamic_minus_fixed | -1.710611 | 161/168 | 7/168 |

## Limitations

- Development benchmark previously used for model and hyperparameter selection; not independent test
- Training uses batch size 1, not manuscript-scale batch size 4 / 2688 fields
- Fixed pair means initial pretrained CNN correspondence and critical vertex locations held frozen
- Dynamic pairs updated each step; evaluation always recomputes fresh diagrams
- Time-adjacent hourly fields are correlated (seven 24-hour blocks)
- PD and spatial locality use top-left 160x160 speed crop, not full 500x500
- Pretraining exclusion cannot be demonstrated
- Upper Midwest and Pacific inland geographic tests not accessed
