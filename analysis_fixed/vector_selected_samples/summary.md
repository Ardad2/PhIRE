# Wind SR diagnostic summary
Selected sample IDs: 0, 2, 4, 11, 23, 29, 101, 134, 162, 165, 166.
Comparison crop: full frame.
Direct scalar CNN available: no.

## Per-sample speed metrics
| sample_id | vector_cnn_speed_mae | vector_gan_speed_mae |
|---|---|---|
| 0 | 0.7135 | 0.8243 |
| 2 | 0.7464 | 0.8645 |
| 4 | 0.7779 | 0.9201 |
| 11 | 0.7542 | 0.9330 |
| 23 | 0.8076 | 1.0245 |
| 29 | 0.9719 | 1.2266 |
| 101 | 0.8214 | 1.0388 |
| 134 | 0.8395 | 1.0832 |
| 162 | 0.3764 | 0.5288 |
| 165 | 0.4972 | 0.6419 |
| 166 | 0.5568 | 0.6753 |

## Aggregate notes
- Mean vector_cnn_speed_mae: 0.7148
- Mean vector_gan_speed_mae: 0.8874
- Use the per-sample speed and velocity figures in this directory to judge whether structural mismatch persists across methods and whether it is already present in the vector components.
