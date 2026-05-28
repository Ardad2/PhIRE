# candidateUV_expanded672 topology evaluation

**Generated:** 2026-05-26

**Important:** This report covers **true topology metrics** (PD bottleneck distance, MT Wasserstein distance) computed by TTK. These are distinct from the component-count *proxy* metrics in `topology_finetuning_candidateUV_expanded672_eval.md`.

**Note:** Lower PD/MT distance = SR is topologically closer to GT. **Lower is better for both metrics.**

**candidateUV_expanded672:** fine-tuned from the pretrained CNN checkpoint with auxiliary physics/topology losses. See the corresponding fine-tuning pilot script for configuration details.

**Samples evaluated:** 168

**MT-GAN baseline wins (before candidateUV_expanded672):** 20 samples — [6, 8, 12, 16, 17, 18, 19, 20, 25, 48, 62, 63, 65, 68, 77, 79, 80, 82, 92, 154]

## Summary statistics

| Metric | CNN (baseline) | GAN (baseline) | candidateUV_expanded672 |
|---|---|---|---|
| PD distance (mean) | 27.4063 | 20.8641 | 29.8747 |
| MT distance (mean) | 5.8678 | 8.3481 | 6.2891 |
| PD wins (vs CNN) | — | 166 | 6 |
| MT wins (vs CNN) | — | 20 | 43 |
| PD beats GAN | — | — | 0 |
| MT beats GAN | — | — | 137 |

## 6 Key evaluation questions

### Q1. Does candidateUV_expanded672 improve PD distance relative to baseline CNN?

- Mean PD: CNN=27.4063, candidateUV_expanded672=29.8747, Δ=2.4684 (▲ worse)
- candidateUV_expanded672 has lower PD on **6/168** samples.

### Q2. Does candidateUV_expanded672 ever beat GAN on PD distance?

- candidateUV_expanded672 beats GAN on PD for **0/168** samples.
- Mean PD: GAN=20.8641, candidateUV_expanded672=29.8747. Δ=9.0106 (▲ worse than GAN on average)

### Q3. Does candidateUV_expanded672 improve MT distance relative to baseline CNN?

- Mean MT: CNN=5.8678, candidateUV_expanded672=6.2891, Δ=0.4213 (▲ worse)
- candidateUV_expanded672 has lower MT on **43/168** samples.

### Q4. Does candidateUV_expanded672 change any of the original 20 MT-GAN cases?

The 20 samples where GAN originally beat CNN on MT distance: [6, 8, 12, 16, 17, 18, 19, 20, 25, 48, 62, 63, 65, 68, 77, 79, 80, 82, 92, 154]

| Outcome | Count |
|---|---|
| MT winner stays GAN | 17 |
| MT winner changes to candidateUV_expanded672 | 3 |
| MT winner changes to CNN | 0 |

Per-sample detail for changed cases:

| sample_idx | MT CNN | MT GAN | MT candidateUV_expanded672 | winner before | winner after |
|---|---|---|---|---|---|
| 25 | 11.3716 | 10.5015 | 7.2904 | gan | candidateUV_expanded672 |
| 63 | 7.6606 | 7.5776 | 7.2497 | gan | candidateUV_expanded672 |
| 68 | 6.0743 | 5.6709 | 5.6597 | gan | candidateUV_expanded672 |

### Q5. What happens in adjacent clusters (samples 10–13 and 90–93)?

| sample_idx | PD CNN | PD GAN | PD candidateUV_expanded672 | MT CNN | MT GAN | MT candidateUV_expanded672 | PD winner → | MT winner → |
|---|---|---|---|---|---|---|---|---|
| 10 | 29.6970 | 24.7100 | 31.4030 | 6.6345 | 7.4238 | 6.0848 | gan → gan | cnn → candidateUV_expanded672 |
| 11 | 28.6873 | 24.9699 | 31.1419 | 5.9107 | 7.8317 | 6.1698 | gan → gan | cnn → cnn |
| 12 | 30.1045 | 24.2206 | 31.6952 | 6.6164 | 5.9137 | 6.7448 | gan → gan | gan → gan |
| 13 | 29.2737 | 24.3237 | 30.8284 | 6.0738 | 7.4023 | 6.5295 | gan → gan | cnn → cnn |
| 90 | 22.2068 | 17.3268 | 23.9892 | 5.2579 | 5.6635 | 5.6863 | gan → gan | cnn → cnn |
| 91 | 22.3094 | 16.2661 | 24.2815 | 5.5083 | 5.5198 | 6.1218 | gan → gan | cnn → cnn |
| 92 | 22.0793 | 16.8077 | 25.0180 | 5.7358 | 5.6779 | 5.7365 | gan → gan | gan → gan |
| 93 | 24.4029 | 18.1355 | 28.5357 | 6.5070 | 6.8160 | 6.3279 | gan → gan | cnn → candidateUV_expanded672 |

Note: samples [12, 92] are in the MT-GAN-wins set (GAN beat CNN on MT in baseline).

### Q6. What happens in rare topology-CNN controls (samples 162 and 163)?

These samples were originally topology-CNN controls (CNN wins on topology metrics). candidateUV_expanded672 should not reverse this.

| sample_idx | PD CNN | PD GAN | PD candidateUV_expanded672 | MT CNN | MT GAN | MT candidateUV_expanded672 | PD winner → | MT winner → |
|---|---|---|---|---|---|---|---|---|
| 162 | 18.2711 | 19.4579 | 23.2716 | 4.5406 | 9.6957 | 4.4987 | cnn → cnn | cnn → candidateUV_expanded672 |
| 163 | 19.3704 | 20.4464 | 24.7756 | 5.6989 | 6.7890 | 4.9968 | cnn → cnn | cnn → candidateUV_expanded672 |

## Winner distribution

### PD distance winners

| Method | Wins before candidateUV_expanded672 | Wins after candidateUV_expanded672 |
|---|---|---|
| cnn | 2 | 2 |
| gan | 166 | 166 |

### MT distance winners

| Method | Wins before candidateUV_expanded672 | Wins after candidateUV_expanded672 |
|---|---|---|
| candidateUV_expanded672 | 0 | 38 |
| cnn | 148 | 113 |
| gan | 20 | 17 |

## Full per-sample detail: focus samples

Includes adjacent-cluster samples, rare controls, and the 20 MT-GAN baseline wins.

| sample | PD CNN | PD GAN | PD candidateUV_expanded672 | MT CNN | MT GAN | MT candidateUV_expanded672 | MT-GAN-win? | PD after | MT after |
|---|---|---|---|---|---|---|---|---|---|
| 6 | 28.0690 | 24.0451 | 31.7620 | 6.1440 | 6.0106 | 6.4833 | ✓ | gan | gan |
| 8 | 29.6351 | 23.2821 | 31.8457 | 7.5096 | 6.0550 | 6.8425 | ✓ | gan | gan |
| 10 | 29.6970 | 24.7100 | 31.4030 | 6.6345 | 7.4238 | 6.0848 |  | gan | candidateUV_expanded672 |
| 11 | 28.6873 | 24.9699 | 31.1419 | 5.9107 | 7.8317 | 6.1698 |  | gan | cnn |
| 12 | 30.1045 | 24.2206 | 31.6952 | 6.6164 | 5.9137 | 6.7448 | ✓ | gan | gan |
| 13 | 29.2737 | 24.3237 | 30.8284 | 6.0738 | 7.4023 | 6.5295 |  | gan | cnn |
| 16 | 23.7560 | 17.3245 | 24.9913 | 5.4642 | 5.2552 | 6.2468 | ✓ | gan | gan |
| 17 | 20.9004 | 17.1181 | 22.7967 | 5.2897 | 4.8487 | 5.4939 | ✓ | gan | gan |
| 18 | 20.3949 | 18.6102 | 22.1424 | 6.2586 | 5.5544 | 5.8613 | ✓ | gan | gan |
| 19 | 21.0641 | 19.0867 | 21.8260 | 6.9280 | 5.4429 | 6.3823 | ✓ | gan | gan |
| 20 | 21.9792 | 18.2663 | 23.4903 | 6.2541 | 5.6891 | 6.9931 | ✓ | gan | gan |
| 25 | 36.5841 | 30.1235 | 39.3127 | 11.3716 | 10.5015 | 7.2904 | ✓ | gan | candidateUV_expanded672 |
| 48 | 22.8325 | 13.1070 | 21.7651 | 7.0815 | 6.6830 | 7.7160 | ✓ | gan | gan |
| 62 | 27.4635 | 17.9937 | 29.3562 | 6.7203 | 6.6502 | 7.3830 | ✓ | gan | gan |
| 63 | 28.5246 | 18.1998 | 29.9466 | 7.6606 | 7.5776 | 7.2497 | ✓ | gan | candidateUV_expanded672 |
| 65 | 22.1489 | 16.2395 | 23.4760 | 5.8208 | 5.5768 | 5.8723 | ✓ | gan | gan |
| 68 | 21.7713 | 15.5009 | 22.3193 | 6.0743 | 5.6709 | 5.6597 | ✓ | gan | candidateUV_expanded672 |
| 77 | 29.9447 | 20.3533 | 32.2884 | 6.1320 | 6.0000 | 6.8894 | ✓ | gan | gan |
| 79 | 30.1481 | 20.3212 | 31.8804 | 6.8565 | 6.4670 | 7.4692 | ✓ | gan | gan |
| 80 | 30.0863 | 20.7449 | 31.6269 | 6.9827 | 6.1767 | 6.5700 | ✓ | gan | gan |
| 82 | 28.9009 | 20.8163 | 31.2042 | 7.1841 | 6.3036 | 6.9989 | ✓ | gan | gan |
| 90 | 22.2068 | 17.3268 | 23.9892 | 5.2579 | 5.6635 | 5.6863 |  | gan | cnn |
| 91 | 22.3094 | 16.2661 | 24.2815 | 5.5083 | 5.5198 | 6.1218 |  | gan | cnn |
| 92 | 22.0793 | 16.8077 | 25.0180 | 5.7358 | 5.6779 | 5.7365 | ✓ | gan | gan |
| 93 | 24.4029 | 18.1355 | 28.5357 | 6.5070 | 6.8160 | 6.3279 |  | gan | candidateUV_expanded672 |
| 154 | 25.3102 | 16.8345 | 25.2758 | 5.8898 | 5.7368 | 6.4535 | ✓ | gan | gan |
| 162 | 18.2711 | 19.4579 | 23.2716 | 4.5406 | 9.6957 | 4.4987 |  | cnn | candidateUV_expanded672 |
| 163 | 19.3704 | 20.4464 | 24.7756 | 5.6989 | 6.7890 | 4.9968 |  | cnn | candidateUV_expanded672 |

## Output files

- `ttk_runs_fixed/topology_finetuning/candidateUV_expanded672_topology/candidateUV_expanded672_pd_mt_distances.csv` — candidateUV_expanded672 PD/MT per sample
- `ttk_runs_fixed/topology_finetuning/candidateUV_expanded672_topology/candidateUV_expanded672_topology_comparison.csv` — three-way comparison (CNN, GAN, candidateUV_expanded672)
- `docs/topology_finetuning_candidateUV_expanded672_topology_eval.md` — this report

## Notes

- PD distance = bottleneck distance between persistence diagrams (computed by `ttkBottleneckDistance`).
- MT distance = Wasserstein-type merge tree distance (computed by `ttkMergeTreeDistanceMatrix`).
- Both use 160×160 speed patches at (x0=0, y0=0), consistent with the CNN/GAN baseline.
- **Lower distance = SR is topologically closer to GT. Lower is better for both metrics.**
- **Component-count proxy metrics** (counts of connected components in speed superlevel sets) are reported separately in `topology_finetuning_candidateUV_expanded672_eval.md` and do not require TTK.
- candidateUV_expanded672 was fine-tuned from the pretrained CNN checkpoint. Results may change with different lambda settings or more epochs.
- Existing CNN/GAN and Candidate B topology outputs were not overwritten.
