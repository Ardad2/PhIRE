# candidateD topology evaluation

**Generated:** 2026-05-18

**Important:** This report covers **true topology metrics** (PD bottleneck distance, MT Wasserstein distance) computed by TTK. These are distinct from the component-count *proxy* metrics in `topology_finetuning_candidateD_eval.md`.

**Note:** Lower PD/MT distance = SR is topologically closer to GT. **Lower is better for both metrics.**

**candidateD:** fine-tuned from the pretrained CNN checkpoint with auxiliary physics/topology losses. See the corresponding fine-tuning pilot script for configuration details.

**Samples evaluated:** 168

**MT-GAN baseline wins (before candidateD):** 20 samples — [6, 8, 12, 16, 17, 18, 19, 20, 25, 48, 62, 63, 65, 68, 77, 79, 80, 82, 92, 154]

## Summary statistics

| Metric | CNN (baseline) | GAN (baseline) | candidateD |
|---|---|---|---|
| PD distance (mean) | 27.4063 | 20.8641 | 27.9510 |
| MT distance (mean) | 5.8678 | 8.3481 | 5.9837 |
| PD wins (vs CNN) | — | 166 | 1 |
| MT wins (vs CNN) | — | 20 | 50 |
| PD beats GAN | — | — | 2 |
| MT beats GAN | — | — | 146 |

## 6 Key evaluation questions

### Q1. Does candidateD improve PD distance relative to baseline CNN?

- Mean PD: CNN=27.4063, candidateD=27.9510, Δ=0.5447 (▲ worse)
- candidateD has lower PD on **1/168** samples.

### Q2. Does candidateD ever beat GAN on PD distance?

- candidateD beats GAN on PD for **2/168** samples.
- Mean PD: GAN=20.8641, candidateD=27.9510. Δ=7.0869 (▲ worse than GAN on average)

### Q3. Does candidateD improve MT distance relative to baseline CNN?

- Mean MT: CNN=5.8678, candidateD=5.9837, Δ=0.1159 (▲ worse)
- candidateD has lower MT on **50/168** samples.

### Q4. Does candidateD change any of the original 20 MT-GAN cases?

The 20 samples where GAN originally beat CNN on MT distance: [6, 8, 12, 16, 17, 18, 19, 20, 25, 48, 62, 63, 65, 68, 77, 79, 80, 82, 92, 154]

| Outcome | Count |
|---|---|
| MT winner stays GAN | 19 |
| MT winner changes to candidateD | 1 |
| MT winner changes to CNN | 0 |

Per-sample detail for changed cases:

| sample_idx | MT CNN | MT GAN | MT candidateD | winner before | winner after |
|---|---|---|---|---|---|
| 25 | 11.3716 | 10.5015 | 8.3649 | gan | candidateD |

### Q5. What happens in adjacent clusters (samples 10–13 and 90–93)?

| sample_idx | PD CNN | PD GAN | PD candidateD | MT CNN | MT GAN | MT candidateD | PD winner → | MT winner → |
|---|---|---|---|---|---|---|---|---|
| 10 | 29.6970 | 24.7100 | 30.4724 | 6.6345 | 7.4238 | 6.8246 | gan → gan | cnn → cnn |
| 11 | 28.6873 | 24.9699 | 29.6452 | 5.9107 | 7.8317 | 5.8405 | gan → gan | cnn → candidateD |
| 12 | 30.1045 | 24.2206 | 31.1261 | 6.6164 | 5.9137 | 6.9601 | gan → gan | gan → gan |
| 13 | 29.2737 | 24.3237 | 30.0189 | 6.0738 | 7.4023 | 6.4373 | gan → gan | cnn → cnn |
| 90 | 22.2068 | 17.3268 | 22.3730 | 5.2579 | 5.6635 | 5.3654 | gan → gan | cnn → cnn |
| 91 | 22.3094 | 16.2661 | 22.4924 | 5.5083 | 5.5198 | 5.4993 | gan → gan | cnn → candidateD |
| 92 | 22.0793 | 16.8077 | 22.2362 | 5.7358 | 5.6779 | 5.7051 | gan → gan | gan → gan |
| 93 | 24.4029 | 18.1355 | 24.6374 | 6.5070 | 6.8160 | 6.5468 | gan → gan | cnn → cnn |

Note: samples [12, 92] are in the MT-GAN-wins set (GAN beat CNN on MT in baseline).

### Q6. What happens in rare topology-CNN controls (samples 162 and 163)?

These samples were originally topology-CNN controls (CNN wins on topology metrics). candidateD should not reverse this.

| sample_idx | PD CNN | PD GAN | PD candidateD | MT CNN | MT GAN | MT candidateD | PD winner → | MT winner → |
|---|---|---|---|---|---|---|---|---|
| 162 | 18.2711 | 19.4579 | 18.2750 | 4.5406 | 9.6957 | 4.4006 | cnn → cnn | cnn → candidateD |
| 163 | 19.3704 | 20.4464 | 19.2642 | 5.6989 | 6.7890 | 5.3723 | cnn → candidateD | cnn → candidateD |

## Winner distribution

### PD distance winners

| Method | Wins before candidateD | Wins after candidateD |
|---|---|---|
| candidateD | 0 | 1 |
| cnn | 2 | 1 |
| gan | 166 | 166 |

### MT distance winners

| Method | Wins before candidateD | Wins after candidateD |
|---|---|---|
| candidateD | 0 | 47 |
| cnn | 148 | 102 |
| gan | 20 | 19 |

## Full per-sample detail: focus samples

Includes adjacent-cluster samples, rare controls, and the 20 MT-GAN baseline wins.

| sample | PD CNN | PD GAN | PD candidateD | MT CNN | MT GAN | MT candidateD | MT-GAN-win? | PD after | MT after |
|---|---|---|---|---|---|---|---|---|---|
| 6 | 28.0690 | 24.0451 | 28.5900 | 6.1440 | 6.0106 | 6.2721 | ✓ | gan | gan |
| 8 | 29.6351 | 23.2821 | 30.1306 | 7.5096 | 6.0550 | 7.5583 | ✓ | gan | gan |
| 10 | 29.6970 | 24.7100 | 30.4724 | 6.6345 | 7.4238 | 6.8246 |  | gan | cnn |
| 11 | 28.6873 | 24.9699 | 29.6452 | 5.9107 | 7.8317 | 5.8405 |  | gan | candidateD |
| 12 | 30.1045 | 24.2206 | 31.1261 | 6.6164 | 5.9137 | 6.9601 | ✓ | gan | gan |
| 13 | 29.2737 | 24.3237 | 30.0189 | 6.0738 | 7.4023 | 6.4373 |  | gan | cnn |
| 16 | 23.7560 | 17.3245 | 24.3362 | 5.4642 | 5.2552 | 6.2495 | ✓ | gan | gan |
| 17 | 20.9004 | 17.1181 | 21.5620 | 5.2897 | 4.8487 | 5.4105 | ✓ | gan | gan |
| 18 | 20.3949 | 18.6102 | 21.1941 | 6.2586 | 5.5544 | 6.2933 | ✓ | gan | gan |
| 19 | 21.0641 | 19.0867 | 21.6700 | 6.9280 | 5.4429 | 7.1069 | ✓ | gan | gan |
| 20 | 21.9792 | 18.2663 | 22.7951 | 6.2541 | 5.6891 | 6.7331 | ✓ | gan | gan |
| 25 | 36.5841 | 30.1235 | 38.0650 | 11.3716 | 10.5015 | 8.3649 | ✓ | gan | candidateD |
| 48 | 22.8325 | 13.1070 | 23.0239 | 7.0815 | 6.6830 | 6.8673 | ✓ | gan | gan |
| 62 | 27.4635 | 17.9937 | 27.7838 | 6.7203 | 6.6502 | 7.2076 | ✓ | gan | gan |
| 63 | 28.5246 | 18.1998 | 28.7412 | 7.6606 | 7.5776 | 7.9172 | ✓ | gan | gan |
| 65 | 22.1489 | 16.2395 | 22.3186 | 5.8208 | 5.5768 | 6.0010 | ✓ | gan | gan |
| 68 | 21.7713 | 15.5009 | 21.8757 | 6.0743 | 5.6709 | 6.2957 | ✓ | gan | gan |
| 77 | 29.9447 | 20.3533 | 30.1048 | 6.1320 | 6.0000 | 6.2968 | ✓ | gan | gan |
| 79 | 30.1481 | 20.3212 | 30.4674 | 6.8565 | 6.4670 | 7.1805 | ✓ | gan | gan |
| 80 | 30.0863 | 20.7449 | 30.3464 | 6.9827 | 6.1767 | 6.8043 | ✓ | gan | gan |
| 82 | 28.9009 | 20.8163 | 29.3553 | 7.1841 | 6.3036 | 7.4707 | ✓ | gan | gan |
| 90 | 22.2068 | 17.3268 | 22.3730 | 5.2579 | 5.6635 | 5.3654 |  | gan | cnn |
| 91 | 22.3094 | 16.2661 | 22.4924 | 5.5083 | 5.5198 | 5.4993 |  | gan | candidateD |
| 92 | 22.0793 | 16.8077 | 22.2362 | 5.7358 | 5.6779 | 5.7051 | ✓ | gan | gan |
| 93 | 24.4029 | 18.1355 | 24.6374 | 6.5070 | 6.8160 | 6.5468 |  | gan | cnn |
| 154 | 25.3102 | 16.8345 | 25.5394 | 5.8898 | 5.7368 | 5.9712 | ✓ | gan | gan |
| 162 | 18.2711 | 19.4579 | 18.2750 | 4.5406 | 9.6957 | 4.4006 |  | cnn | candidateD |
| 163 | 19.3704 | 20.4464 | 19.2642 | 5.6989 | 6.7890 | 5.3723 |  | candidateD | candidateD |

## Output files

- `ttk_runs_fixed/topology_finetuning/candidateD_topology/candidateD_pd_mt_distances.csv` — candidateD PD/MT per sample
- `ttk_runs_fixed/topology_finetuning/candidateD_topology/candidateD_topology_comparison.csv` — three-way comparison (CNN, GAN, candidateD)
- `docs/topology_finetuning_candidateD_topology_eval.md` — this report

## Notes

- PD distance = bottleneck distance between persistence diagrams (computed by `ttkBottleneckDistance`).
- MT distance = Wasserstein-type merge tree distance (computed by `ttkMergeTreeDistanceMatrix`).
- Both use 160×160 speed patches at (x0=0, y0=0), consistent with the CNN/GAN baseline.
- **Lower distance = SR is topologically closer to GT. Lower is better for both metrics.**
- **Component-count proxy metrics** (counts of connected components in speed superlevel sets) are reported separately in `topology_finetuning_candidateD_eval.md` and do not require TTK.
- candidateD was fine-tuned from the pretrained CNN checkpoint. Results may change with different lambda settings or more epochs.
- Existing CNN/GAN and Candidate B topology outputs were not overwritten.
