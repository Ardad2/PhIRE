# Wind SR diagnostic summary
Selected sample IDs: 8, 12, 25.
Comparison crop: full frame.
Direct scalar CNN available: no.

## Per-sample speed metrics
| sample_id | vector_cnn_speed_mae | vector_gan_speed_mae |
|---|---|---|
| 8 | 0.7798 | 0.9186 |
| 12 | 0.7538 | 0.9281 |
| 25 | 0.9166 | 1.1334 |

## Aggregate notes
- Mean vector_cnn_speed_mae: 0.8167
- Mean vector_gan_speed_mae: 0.9934
- Use the per-sample speed and velocity figures in this directory to judge whether structural mismatch persists across methods and whether it is already present in the vector components.
