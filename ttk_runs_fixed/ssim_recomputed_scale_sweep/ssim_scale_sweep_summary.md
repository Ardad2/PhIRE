# Recomputed SSIM scale sweep

Computed in an isolated environment to avoid the Spark scikit-image/NumPy incompatibility.

Definitions:

- `ssim_speed`: SSIM on scalar wind-speed magnitude fields.
- `ssim_uv_mean`: average of separate SSIM on u and v components.

## Summary by method

| method           | family      | training_size   |   n |   ssim_speed_mean |   ssim_uv_mean_mean |   ssim_speed_std |   ssim_uv_mean_std |
|:-----------------|:------------|:----------------|----:|------------------:|--------------------:|-----------------:|-------------------:|
| cnn              | baseline    | baseline        | 168 |          0.741175 |            0.771031 |        0.0433509 |          0.0385659 |
| gan              | baseline    | baseline        | 168 |          0.677418 |            0.698833 |        0.0554873 |          0.0494062 |
| candidateC_168   | candidateC  | 168             | 168 |          0.777743 |            0.809682 |        0.0345476 |          0.0308111 |
| candidateC_672   | candidateC  | 672             | 168 |          0.794862 |            0.830481 |        0.0349605 |          0.0317998 |
| candidateC_1344  | candidateC  | 1344            | 168 |          0.804386 |            0.840138 |        0.0340974 |          0.0309555 |
| candidateC_2688  | candidateC  | 2688            | 168 |          0.812622 |            0.84811  |        0.0336225 |          0.0303147 |
| candidateUV_168  | candidateUV | 168             | 168 |          0.77518  |            0.811189 |        0.0359265 |          0.0317725 |
| candidateUV_672  | candidateUV | 672             | 168 |          0.795236 |            0.835095 |        0.0361187 |          0.0319504 |
| candidateUV_1344 | candidateUV | 1344            | 168 |          0.805511 |            0.845448 |        0.034735  |          0.030995  |
| candidateUV_2688 | candidateUV | 2688            | 168 |          0.813443 |            0.853551 |        0.0337025 |          0.0301418 |

## Pairwise wins vs CNN

| method           | family      | training_size   |   n |   speed_ssim_gt_cnn_count |   uv_ssim_gt_cnn_count |   mean_delta_ssim_speed_vs_cnn |   mean_delta_ssim_uv_mean_vs_cnn |
|:-----------------|:------------|:----------------|----:|--------------------------:|-----------------------:|-------------------------------:|---------------------------------:|
| candidateC_1344  | candidateC  | 1344            | 168 |                       168 |                    168 |                      0.0632107 |                        0.0691076 |
| candidateC_168   | candidateC  | 168             | 168 |                       166 |                    168 |                      0.0365682 |                        0.0386513 |
| candidateC_2688  | candidateC  | 2688            | 168 |                       168 |                    168 |                      0.0714469 |                        0.0770798 |
| candidateC_672   | candidateC  | 672             | 168 |                       168 |                    168 |                      0.0536864 |                        0.0594508 |
| candidateUV_1344 | candidateUV | 1344            | 168 |                       168 |                    168 |                      0.0643359 |                        0.074417  |
| candidateUV_168  | candidateUV | 168             | 168 |                       166 |                    168 |                      0.0340048 |                        0.0401586 |
| candidateUV_2688 | candidateUV | 2688            | 168 |                       168 |                    168 |                      0.0722677 |                        0.0825199 |
| candidateUV_672  | candidateUV | 672             | 168 |                       168 |                    168 |                      0.0540605 |                        0.0640648 |
| gan              | baseline    | baseline        | 168 |                         3 |                      0 |                     -0.0637569 |                       -0.0721975 |

## Candidate C vs UV by training size

|   training_size |   n |   c_speed_ssim_gt_uv_count |   c_uv_ssim_gt_uv_count |   mean_delta_c_minus_uv_ssim_speed |   mean_delta_c_minus_uv_ssim_uv_mean |
|----------------:|----:|---------------------------:|------------------------:|-----------------------------------:|-------------------------------------:|
|             168 | 168 |                        135 |                      38 |                        0.00256335  |                          -0.00150729 |
|             672 | 168 |                         81 |                       3 |                       -0.000374047 |                          -0.00461401 |
|            1344 | 168 |                         47 |                       1 |                       -0.00112518  |                          -0.00530943 |
|            2688 | 168 |                         54 |                       2 |                       -0.000820727 |                          -0.00544007 |
