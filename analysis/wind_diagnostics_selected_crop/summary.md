# Wind SR diagnostic summary
Selected sample IDs: 165, 166, 2, 11, 4, 29, 23, 0.
Repo reference/default sample IDs included when present: 0, 2, 165, 166.
Comparison crop: patch=160, x0=0, y0=0.
Direct scalar CNN available: yes.

## Per-sample speed metrics
| sample_id | vector_cnn_speed_mae | vector_gan_speed_mae | scalar_cnn_speed_mae |
|---|---|---|---|
| 0 | 3.6621 | 3.6433 | 3.7532 |
| 2 | 4.0012 | 3.9321 | 4.0869 |
| 4 | 3.3320 | 3.4685 | 3.3245 |
| 11 | 4.9138 | 5.4379 | 1.9778 |
| 23 | 4.2607 | 4.3991 | 3.6701 |
| 29 | 6.4137 | 6.0773 | 4.0500 |
| 165 | 3.3227 | 3.1688 | 3.2863 |
| 166 | 3.0741 | 2.9269 | 3.1405 |

## Aggregate notes
- Mean vector_cnn_speed_mae: 4.1225
- Mean vector_gan_speed_mae: 4.1317
- Mean scalar_cnn_speed_mae: 3.4112
- Use the per-sample speed and velocity figures in this directory to judge whether structural mismatch persists across methods and whether it is already present in the vector components.
