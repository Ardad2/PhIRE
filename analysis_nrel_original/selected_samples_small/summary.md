# Wind SR diagnostic summary
Selected sample IDs: 0, 1, 2, 3, 4.
Comparison crop: full frame.
Direct scalar CNN available: yes.

## Per-sample speed metrics
| sample_id | vector_cnn_speed_mae | vector_gan_speed_mae | scalar_cnn_speed_mae |
|---|---|---|---|
| 0 | 0.6757 | 0.8861 | 1.1886 |
| 1 | 0.4035 | 0.5832 | 1.0221 |
| 2 | 0.4268 | 0.6547 | 1.1010 |
| 3 | 1.5930 | 1.6726 | 2.4583 |
| 4 | 0.6708 | 0.7915 | 1.5579 |

## Aggregate notes
- Mean vector_cnn_speed_mae: 0.7539
- Mean vector_gan_speed_mae: 0.9176
- Mean scalar_cnn_speed_mae: 1.4656
- Use the per-sample speed and velocity figures in this directory to judge whether structural mismatch persists across methods and whether it is already present in the vector components.
