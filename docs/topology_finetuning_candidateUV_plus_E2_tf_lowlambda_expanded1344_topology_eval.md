# candidateUV_plus_E2_tf_lowlambda_expanded1344 topology evaluation

**Generated:** 2026-07-11

**Important:** This report covers **true topology metrics** (PD bottleneck distance, MT Wasserstein distance) computed by TTK. These are distinct from the component-count *proxy* metrics in `topology_finetuning_candidateUV_plus_E2_tf_lowlambda_expanded1344_eval.md`.

**Note:** Lower PD/MT distance = SR is topologically closer to GT. **Lower is better for both metrics.**

**candidateUV_plus_E2_tf_lowlambda_expanded1344:** fine-tuned from the pretrained CNN checkpoint with auxiliary physics/topology losses. See the corresponding fine-tuning pilot script for configuration details.

**Samples evaluated:** 168

**MT-GAN baseline wins (before candidateUV_plus_E2_tf_lowlambda_expanded1344):** 20 samples — [6, 8, 12, 16, 17, 18, 19, 20, 25, 48, 62, 63, 65, 68, 77, 79, 80, 82, 92, 154]

## Summary statistics

| Metric | CNN (baseline) | GAN (baseline) | candidateUV_plus_E2_tf_lowlambda_expanded1344 |
|---|---|---|---|
| PD distance (mean) | 27.4063 | 20.8641 | 25.1566 |
| MT distance (mean) | 5.8678 | 8.3481 | 5.6300 |
| PD wins (vs CNN) | — | 166 | 158 |
| MT wins (vs CNN) | — | 20 | 105 |
| PD beats GAN | — | — | 2 |
| MT beats GAN | — | — | 165 |

## 6 Key evaluation questions

### Q1. Does candidateUV_plus_E2_tf_lowlambda_expanded1344 improve PD distance relative to baseline CNN?

- Mean PD: CNN=27.4063, candidateUV_plus_E2_tf_lowlambda_expanded1344=25.1566, Δ=-2.2497 (▼ better)
- candidateUV_plus_E2_tf_lowlambda_expanded1344 has lower PD on **158/168** samples.

### Q2. Does candidateUV_plus_E2_tf_lowlambda_expanded1344 ever beat GAN on PD distance?

- candidateUV_plus_E2_tf_lowlambda_expanded1344 beats GAN on PD for **2/168** samples.
- Mean PD: GAN=20.8641, candidateUV_plus_E2_tf_lowlambda_expanded1344=25.1566. Δ=4.2925 (▲ worse than GAN on average)

### Q3. Does candidateUV_plus_E2_tf_lowlambda_expanded1344 improve MT distance relative to baseline CNN?

- Mean MT: CNN=5.8678, candidateUV_plus_E2_tf_lowlambda_expanded1344=5.6300, Δ=-0.2378 (▼ better)
- candidateUV_plus_E2_tf_lowlambda_expanded1344 has lower MT on **105/168** samples.

### Q4. Does candidateUV_plus_E2_tf_lowlambda_expanded1344 change any of the original 20 MT-GAN cases?

The 20 samples where GAN originally beat CNN on MT distance: [6, 8, 12, 16, 17, 18, 19, 20, 25, 48, 62, 63, 65, 68, 77, 79, 80, 82, 92, 154]

| Outcome | Count |
|---|---|
| MT winner stays GAN | 2 |
| MT winner changes to candidateUV_plus_E2_tf_lowlambda_expanded1344 | 18 |
| MT winner changes to CNN | 0 |

Per-sample detail for changed cases:

| sample_idx | MT CNN | MT GAN | MT candidateUV_plus_E2_tf_lowlambda_expanded1344 | winner before | winner after |
|---|---|---|---|---|---|
| 6 | 6.1440 | 6.0106 | 5.3329 | gan | candidateUV_plus_E2_tf_lowlambda_expanded1344 |
| 12 | 6.6164 | 5.9137 | 5.7734 | gan | candidateUV_plus_E2_tf_lowlambda_expanded1344 |
| 16 | 5.4642 | 5.2552 | 4.7770 | gan | candidateUV_plus_E2_tf_lowlambda_expanded1344 |
| 17 | 5.2897 | 4.8487 | 4.0703 | gan | candidateUV_plus_E2_tf_lowlambda_expanded1344 |
| 18 | 6.2586 | 5.5544 | 4.3897 | gan | candidateUV_plus_E2_tf_lowlambda_expanded1344 |
| 19 | 6.9280 | 5.4429 | 4.8731 | gan | candidateUV_plus_E2_tf_lowlambda_expanded1344 |
| 20 | 6.2541 | 5.6891 | 5.4769 | gan | candidateUV_plus_E2_tf_lowlambda_expanded1344 |
| 25 | 11.3716 | 10.5015 | 7.0673 | gan | candidateUV_plus_E2_tf_lowlambda_expanded1344 |
| 48 | 7.0815 | 6.6830 | 6.5217 | gan | candidateUV_plus_E2_tf_lowlambda_expanded1344 |
| 62 | 6.7203 | 6.6502 | 5.9449 | gan | candidateUV_plus_E2_tf_lowlambda_expanded1344 |
| 63 | 7.6606 | 7.5776 | 6.4846 | gan | candidateUV_plus_E2_tf_lowlambda_expanded1344 |
| 65 | 5.8208 | 5.5768 | 5.0407 | gan | candidateUV_plus_E2_tf_lowlambda_expanded1344 |
| 68 | 6.0743 | 5.6709 | 4.8175 | gan | candidateUV_plus_E2_tf_lowlambda_expanded1344 |
| 77 | 6.1320 | 6.0000 | 4.8748 | gan | candidateUV_plus_E2_tf_lowlambda_expanded1344 |
| 79 | 6.8565 | 6.4670 | 5.5560 | gan | candidateUV_plus_E2_tf_lowlambda_expanded1344 |
| 80 | 6.9827 | 6.1767 | 4.6783 | gan | candidateUV_plus_E2_tf_lowlambda_expanded1344 |
| 82 | 7.1841 | 6.3036 | 5.2591 | gan | candidateUV_plus_E2_tf_lowlambda_expanded1344 |
| 92 | 5.7358 | 5.6779 | 5.4798 | gan | candidateUV_plus_E2_tf_lowlambda_expanded1344 |

### Q5. What happens in adjacent clusters (samples 10–13 and 90–93)?

| sample_idx | PD CNN | PD GAN | PD candidateUV_plus_E2_tf_lowlambda_expanded1344 | MT CNN | MT GAN | MT candidateUV_plus_E2_tf_lowlambda_expanded1344 | PD winner → | MT winner → |
|---|---|---|---|---|---|---|---|---|
| 10 | 29.6970 | 24.7100 | 26.7794 | 6.6345 | 7.4238 | 5.6304 | gan → gan | cnn → candidateUV_plus_E2_tf_lowlambda_expanded1344 |
| 11 | 28.6873 | 24.9699 | 25.5401 | 5.9107 | 7.8317 | 5.8477 | gan → gan | cnn → candidateUV_plus_E2_tf_lowlambda_expanded1344 |
| 12 | 30.1045 | 24.2206 | 26.9539 | 6.6164 | 5.9137 | 5.7734 | gan → gan | gan → candidateUV_plus_E2_tf_lowlambda_expanded1344 |
| 13 | 29.2737 | 24.3237 | 26.6634 | 6.0738 | 7.4023 | 6.1289 | gan → gan | cnn → cnn |
| 90 | 22.2068 | 17.3268 | 20.2672 | 5.2579 | 5.6635 | 4.9213 | gan → gan | cnn → candidateUV_plus_E2_tf_lowlambda_expanded1344 |
| 91 | 22.3094 | 16.2661 | 20.2546 | 5.5083 | 5.5198 | 5.2208 | gan → gan | cnn → candidateUV_plus_E2_tf_lowlambda_expanded1344 |
| 92 | 22.0793 | 16.8077 | 20.5491 | 5.7358 | 5.6779 | 5.4798 | gan → gan | gan → candidateUV_plus_E2_tf_lowlambda_expanded1344 |
| 93 | 24.4029 | 18.1355 | 23.2500 | 6.5070 | 6.8160 | 6.4603 | gan → gan | cnn → candidateUV_plus_E2_tf_lowlambda_expanded1344 |

Note: samples [12, 92] are in the MT-GAN-wins set (GAN beat CNN on MT in baseline).

### Q6. What happens in rare topology-CNN controls (samples 162 and 163)?

These samples were originally topology-CNN controls (CNN wins on topology metrics). candidateUV_plus_E2_tf_lowlambda_expanded1344 should not reverse this.

| sample_idx | PD CNN | PD GAN | PD candidateUV_plus_E2_tf_lowlambda_expanded1344 | MT CNN | MT GAN | MT candidateUV_plus_E2_tf_lowlambda_expanded1344 | PD winner → | MT winner → |
|---|---|---|---|---|---|---|---|---|
| 162 | 18.2711 | 19.4579 | 19.3901 | 4.5406 | 9.6957 | 4.5669 | cnn → cnn | cnn → cnn |
| 163 | 19.3704 | 20.4464 | 20.5597 | 5.6989 | 6.7890 | 5.0447 | cnn → cnn | cnn → candidateUV_plus_E2_tf_lowlambda_expanded1344 |

## Winner distribution

### PD distance winners

| Method | Wins before candidateUV_plus_E2_tf_lowlambda_expanded1344 | Wins after candidateUV_plus_E2_tf_lowlambda_expanded1344 |
|---|---|---|
| candidateUV_plus_E2_tf_lowlambda_expanded1344 | 0 | 1 |
| cnn | 2 | 2 |
| gan | 166 | 165 |

### MT distance winners

| Method | Wins before candidateUV_plus_E2_tf_lowlambda_expanded1344 | Wins after candidateUV_plus_E2_tf_lowlambda_expanded1344 |
|---|---|---|
| candidateUV_plus_E2_tf_lowlambda_expanded1344 | 0 | 104 |
| cnn | 148 | 62 |
| gan | 20 | 2 |

## Full per-sample detail: focus samples

Includes adjacent-cluster samples, rare controls, and the 20 MT-GAN baseline wins.

| sample | PD CNN | PD GAN | PD candidateUV_plus_E2_tf_lowlambda_expanded1344 | MT CNN | MT GAN | MT candidateUV_plus_E2_tf_lowlambda_expanded1344 | MT-GAN-win? | PD after | MT after |
|---|---|---|---|---|---|---|---|---|---|
| 6 | 28.0690 | 24.0451 | 26.1709 | 6.1440 | 6.0106 | 5.3329 | ✓ | gan | candidateUV_plus_E2_tf_lowlambda_expanded1344 |
| 8 | 29.6351 | 23.2821 | 27.4463 | 7.5096 | 6.0550 | 6.2544 | ✓ | gan | gan |
| 10 | 29.6970 | 24.7100 | 26.7794 | 6.6345 | 7.4238 | 5.6304 |  | gan | candidateUV_plus_E2_tf_lowlambda_expanded1344 |
| 11 | 28.6873 | 24.9699 | 25.5401 | 5.9107 | 7.8317 | 5.8477 |  | gan | candidateUV_plus_E2_tf_lowlambda_expanded1344 |
| 12 | 30.1045 | 24.2206 | 26.9539 | 6.6164 | 5.9137 | 5.7734 | ✓ | gan | candidateUV_plus_E2_tf_lowlambda_expanded1344 |
| 13 | 29.2737 | 24.3237 | 26.6634 | 6.0738 | 7.4023 | 6.1289 |  | gan | cnn |
| 16 | 23.7560 | 17.3245 | 21.0397 | 5.4642 | 5.2552 | 4.7770 | ✓ | gan | candidateUV_plus_E2_tf_lowlambda_expanded1344 |
| 17 | 20.9004 | 17.1181 | 19.2201 | 5.2897 | 4.8487 | 4.0703 | ✓ | gan | candidateUV_plus_E2_tf_lowlambda_expanded1344 |
| 18 | 20.3949 | 18.6102 | 18.9777 | 6.2586 | 5.5544 | 4.3897 | ✓ | gan | candidateUV_plus_E2_tf_lowlambda_expanded1344 |
| 19 | 21.0641 | 19.0867 | 18.8473 | 6.9280 | 5.4429 | 4.8731 | ✓ | candidateUV_plus_E2_tf_lowlambda_expanded1344 | candidateUV_plus_E2_tf_lowlambda_expanded1344 |
| 20 | 21.9792 | 18.2663 | 20.1178 | 6.2541 | 5.6891 | 5.4769 | ✓ | gan | candidateUV_plus_E2_tf_lowlambda_expanded1344 |
| 25 | 36.5841 | 30.1235 | 32.8625 | 11.3716 | 10.5015 | 7.0673 | ✓ | gan | candidateUV_plus_E2_tf_lowlambda_expanded1344 |
| 48 | 22.8325 | 13.1070 | 20.0421 | 7.0815 | 6.6830 | 6.5217 | ✓ | gan | candidateUV_plus_E2_tf_lowlambda_expanded1344 |
| 62 | 27.4635 | 17.9937 | 24.4335 | 6.7203 | 6.6502 | 5.9449 | ✓ | gan | candidateUV_plus_E2_tf_lowlambda_expanded1344 |
| 63 | 28.5246 | 18.1998 | 24.6988 | 7.6606 | 7.5776 | 6.4846 | ✓ | gan | candidateUV_plus_E2_tf_lowlambda_expanded1344 |
| 65 | 22.1489 | 16.2395 | 19.6475 | 5.8208 | 5.5768 | 5.0407 | ✓ | gan | candidateUV_plus_E2_tf_lowlambda_expanded1344 |
| 68 | 21.7713 | 15.5009 | 18.9075 | 6.0743 | 5.6709 | 4.8175 | ✓ | gan | candidateUV_plus_E2_tf_lowlambda_expanded1344 |
| 77 | 29.9447 | 20.3533 | 26.2845 | 6.1320 | 6.0000 | 4.8748 | ✓ | gan | candidateUV_plus_E2_tf_lowlambda_expanded1344 |
| 79 | 30.1481 | 20.3212 | 27.0803 | 6.8565 | 6.4670 | 5.5560 | ✓ | gan | candidateUV_plus_E2_tf_lowlambda_expanded1344 |
| 80 | 30.0863 | 20.7449 | 26.5904 | 6.9827 | 6.1767 | 4.6783 | ✓ | gan | candidateUV_plus_E2_tf_lowlambda_expanded1344 |
| 82 | 28.9009 | 20.8163 | 25.2972 | 7.1841 | 6.3036 | 5.2591 | ✓ | gan | candidateUV_plus_E2_tf_lowlambda_expanded1344 |
| 90 | 22.2068 | 17.3268 | 20.2672 | 5.2579 | 5.6635 | 4.9213 |  | gan | candidateUV_plus_E2_tf_lowlambda_expanded1344 |
| 91 | 22.3094 | 16.2661 | 20.2546 | 5.5083 | 5.5198 | 5.2208 |  | gan | candidateUV_plus_E2_tf_lowlambda_expanded1344 |
| 92 | 22.0793 | 16.8077 | 20.5491 | 5.7358 | 5.6779 | 5.4798 | ✓ | gan | candidateUV_plus_E2_tf_lowlambda_expanded1344 |
| 93 | 24.4029 | 18.1355 | 23.2500 | 6.5070 | 6.8160 | 6.4603 |  | gan | candidateUV_plus_E2_tf_lowlambda_expanded1344 |
| 154 | 25.3102 | 16.8345 | 22.5501 | 5.8898 | 5.7368 | 6.5250 | ✓ | gan | gan |
| 162 | 18.2711 | 19.4579 | 19.3901 | 4.5406 | 9.6957 | 4.5669 |  | cnn | cnn |
| 163 | 19.3704 | 20.4464 | 20.5597 | 5.6989 | 6.7890 | 5.0447 |  | cnn | candidateUV_plus_E2_tf_lowlambda_expanded1344 |

## Output files

- `ttk_runs_fixed/topology_finetuning/candidateUV_plus_E2_tf_lowlambda_expanded1344_topology/candidateUV_plus_E2_tf_lowlambda_expanded1344_pd_mt_distances.csv` — candidateUV_plus_E2_tf_lowlambda_expanded1344 PD/MT per sample
- `ttk_runs_fixed/topology_finetuning/candidateUV_plus_E2_tf_lowlambda_expanded1344_topology/candidateUV_plus_E2_tf_lowlambda_expanded1344_topology_comparison.csv` — three-way comparison (CNN, GAN, candidateUV_plus_E2_tf_lowlambda_expanded1344)
- `docs/topology_finetuning_candidateUV_plus_E2_tf_lowlambda_expanded1344_topology_eval.md` — this report

## Notes

- PD distance = bottleneck distance between persistence diagrams (computed by `ttkBottleneckDistance`).
- MT distance = Wasserstein-type merge tree distance (computed by `ttkMergeTreeDistanceMatrix`).
- Both use 160×160 speed patches at (x0=0, y0=0), consistent with the CNN/GAN baseline.
- **Lower distance = SR is topologically closer to GT. Lower is better for both metrics.**
- **Component-count proxy metrics** (counts of connected components in speed superlevel sets) are reported separately in `topology_finetuning_candidateUV_plus_E2_tf_lowlambda_expanded1344_eval.md` and do not require TTK.
- candidateUV_plus_E2_tf_lowlambda_expanded1344 was fine-tuned from the pretrained CNN checkpoint. Results may change with different lambda settings or more epochs.
- Existing CNN/GAN and Candidate B topology outputs were not overwritten.
