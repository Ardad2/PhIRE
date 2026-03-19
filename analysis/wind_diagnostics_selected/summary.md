# Wind SR diagnostic summary
Selected sample IDs: 165, 166, 2, 11, 4, 29, 23, 0.
Repo reference/default sample IDs included when present: 0, 2, 165, 166.
Comparison crop: full frame.
Direct scalar CNN available: yes.

## Per-sample speed metrics
| sample_id | vector_cnn_speed_mae | vector_gan_speed_mae | scalar_cnn_speed_mae |
|---|---|---|---|
| 0 | 3.1864 | 3.1952 | 3.1985 |
| 2 | 3.8221 | 3.7815 | 3.3285 |
| 4 | 4.1131 | 4.1018 | 3.1753 |
| 11 | 4.6574 | 4.9549 | 2.5695 |
| 23 | 7.2572 | 7.7461 | 2.1306 |
| 29 | 6.5486 | 6.5740 | 3.5500 |
| 165 | 2.7817 | 2.7473 | 2.5176 |
| 166 | 2.9483 | 2.8742 | 2.3115 |

## Aggregate notes
- Mean vector_cnn_speed_mae: 4.4143
- Mean vector_gan_speed_mae: 4.4969
- Mean scalar_cnn_speed_mae: 2.8477
- Use the per-sample speed and velocity figures in this directory to judge whether structural mismatch persists across methods and whether it is already present in the vector components.
