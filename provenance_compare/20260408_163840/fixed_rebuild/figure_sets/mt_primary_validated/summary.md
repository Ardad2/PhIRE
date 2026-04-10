# Wind SR diagnostic summary
Selected sample IDs: 27, 31, 37, 39, 29, 32.
Comparison crop: full frame.
Direct scalar CNN available: no.

## Per-sample speed metrics
| sample_id | vector_cnn_speed_mae | vector_gan_speed_mae |
|---|---|---|
| 27 | 1.0111 | 1.2366 |
| 29 | 0.9719 | 1.2266 |
| 31 | 0.9299 | 1.1454 |
| 32 | 0.8988 | 1.1301 |
| 37 | 0.8446 | 1.0289 |
| 39 | 0.6204 | 0.8319 |

## Aggregate notes
- Mean vector_cnn_speed_mae: 0.8794
- Mean vector_gan_speed_mae: 1.0999
- Use the per-sample speed and velocity figures in this directory to judge whether structural mismatch persists across methods and whether it is already present in the vector components.
