# Wind SR diagnostic summary
Selected sample IDs: 0, 2.
Comparison crop: full frame.
Direct scalar CNN available: yes.

## Per-sample speed metrics
| sample_id | vector_cnn_speed_mae | vector_gan_speed_mae | scalar_cnn_speed_mae |
|---|---|---|---|
| 0 | 0.6757 | 0.8861 | 1.1886 |
| 2 | 0.4268 | 0.6547 | 1.1010 |

## Aggregate notes
- Mean vector_cnn_speed_mae: 0.5512
- Mean vector_gan_speed_mae: 0.7704
- Mean scalar_cnn_speed_mae: 1.1448
- Use the per-sample speed and velocity figures in this directory to judge whether structural mismatch persists across methods and whether it is already present in the vector components.
