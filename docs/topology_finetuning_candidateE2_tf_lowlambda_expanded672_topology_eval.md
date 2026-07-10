# candidateE2_tf_lowlambda_expanded672 topology evaluation

**Generated:** 2026-07-09

**Important:** This report covers **true topology metrics** (PD bottleneck distance, MT Wasserstein distance) computed by TTK. These are distinct from the component-count *proxy* metrics in `topology_finetuning_candidateE2_tf_lowlambda_expanded672_eval.md`.

**Note:** Lower PD/MT distance = SR is topologically closer to GT. **Lower is better for both metrics.**

**candidateE2_tf_lowlambda_expanded672:** fine-tuned from the pretrained CNN checkpoint with auxiliary physics/topology losses. See the corresponding fine-tuning pilot script for configuration details.

**Samples evaluated:** 168

**MT-GAN baseline wins (before candidateE2_tf_lowlambda_expanded672):** 20 samples — [6, 8, 12, 16, 17, 18, 19, 20, 25, 48, 62, 63, 65, 68, 77, 79, 80, 82, 92, 154]

## Summary statistics

| Metric | CNN (baseline) | GAN (baseline) | candidateE2_tf_lowlambda_expanded672 |
|---|---|---|---|
| PD distance (mean) | 27.4063 | 20.8641 | 24.9844 |
| MT distance (mean) | 5.8678 | 8.3481 | 5.7076 |
| PD wins (vs CNN) | — | 166 | 162 |
| MT wins (vs CNN) | — | 20 | 96 |
| PD beats GAN | — | — | 4 |
| MT beats GAN | — | — | 160 |

## 6 Key evaluation questions

### Q1. Does candidateE2_tf_lowlambda_expanded672 improve PD distance relative to baseline CNN?

- Mean PD: CNN=27.4063, candidateE2_tf_lowlambda_expanded672=24.9844, Δ=-2.4219 (▼ better)
- candidateE2_tf_lowlambda_expanded672 has lower PD on **162/168** samples.

### Q2. Does candidateE2_tf_lowlambda_expanded672 ever beat GAN on PD distance?

- candidateE2_tf_lowlambda_expanded672 beats GAN on PD for **4/168** samples.
- Mean PD: GAN=20.8641, candidateE2_tf_lowlambda_expanded672=24.9844. Δ=4.1203 (▲ worse than GAN on average)

### Q3. Does candidateE2_tf_lowlambda_expanded672 improve MT distance relative to baseline CNN?

- Mean MT: CNN=5.8678, candidateE2_tf_lowlambda_expanded672=5.7076, Δ=-0.1602 (▼ better)
- candidateE2_tf_lowlambda_expanded672 has lower MT on **96/168** samples.

### Q4. Does candidateE2_tf_lowlambda_expanded672 change any of the original 20 MT-GAN cases?

The 20 samples where GAN originally beat CNN on MT distance: [6, 8, 12, 16, 17, 18, 19, 20, 25, 48, 62, 63, 65, 68, 77, 79, 80, 82, 92, 154]

| Outcome | Count |
|---|---|
| MT winner stays GAN | 6 |
| MT winner changes to candidateE2_tf_lowlambda_expanded672 | 14 |
| MT winner changes to CNN | 0 |

Per-sample detail for changed cases:

| sample_idx | MT CNN | MT GAN | MT candidateE2_tf_lowlambda_expanded672 | winner before | winner after |
|---|---|---|---|---|---|
| 16 | 5.4642 | 5.2552 | 4.8143 | gan | candidateE2_tf_lowlambda_expanded672 |
| 17 | 5.2897 | 4.8487 | 4.3087 | gan | candidateE2_tf_lowlambda_expanded672 |
| 18 | 6.2586 | 5.5544 | 4.6770 | gan | candidateE2_tf_lowlambda_expanded672 |
| 25 | 11.3716 | 10.5015 | 7.7876 | gan | candidateE2_tf_lowlambda_expanded672 |
| 48 | 7.0815 | 6.6830 | 6.6124 | gan | candidateE2_tf_lowlambda_expanded672 |
| 62 | 6.7203 | 6.6502 | 6.1064 | gan | candidateE2_tf_lowlambda_expanded672 |
| 63 | 7.6606 | 7.5776 | 6.2031 | gan | candidateE2_tf_lowlambda_expanded672 |
| 65 | 5.8208 | 5.5768 | 4.6336 | gan | candidateE2_tf_lowlambda_expanded672 |
| 68 | 6.0743 | 5.6709 | 4.7991 | gan | candidateE2_tf_lowlambda_expanded672 |
| 77 | 6.1320 | 6.0000 | 5.5826 | gan | candidateE2_tf_lowlambda_expanded672 |
| 79 | 6.8565 | 6.4670 | 5.8018 | gan | candidateE2_tf_lowlambda_expanded672 |
| 80 | 6.9827 | 6.1767 | 5.0462 | gan | candidateE2_tf_lowlambda_expanded672 |
| 82 | 7.1841 | 6.3036 | 5.7384 | gan | candidateE2_tf_lowlambda_expanded672 |
| 92 | 5.7358 | 5.6779 | 5.6612 | gan | candidateE2_tf_lowlambda_expanded672 |

### Q5. What happens in adjacent clusters (samples 10–13 and 90–93)?

| sample_idx | PD CNN | PD GAN | PD candidateE2_tf_lowlambda_expanded672 | MT CNN | MT GAN | MT candidateE2_tf_lowlambda_expanded672 | PD winner → | MT winner → |
|---|---|---|---|---|---|---|---|---|
| 10 | 29.6970 | 24.7100 | 26.8889 | 6.6345 | 7.4238 | 5.8768 | gan → gan | cnn → candidateE2_tf_lowlambda_expanded672 |
| 11 | 28.6873 | 24.9699 | 25.6334 | 5.9107 | 7.8317 | 5.9130 | gan → gan | cnn → cnn |
| 12 | 30.1045 | 24.2206 | 26.9259 | 6.6164 | 5.9137 | 6.1441 | gan → gan | gan → gan |
| 13 | 29.2737 | 24.3237 | 26.6505 | 6.0738 | 7.4023 | 6.1800 | gan → gan | cnn → cnn |
| 90 | 22.2068 | 17.3268 | 19.6113 | 5.2579 | 5.6635 | 4.9208 | gan → gan | cnn → candidateE2_tf_lowlambda_expanded672 |
| 91 | 22.3094 | 16.2661 | 19.5785 | 5.5083 | 5.5198 | 5.5519 | gan → gan | cnn → cnn |
| 92 | 22.0793 | 16.8077 | 19.7510 | 5.7358 | 5.6779 | 5.6612 | gan → gan | gan → candidateE2_tf_lowlambda_expanded672 |
| 93 | 24.4029 | 18.1355 | 22.2233 | 6.5070 | 6.8160 | 6.2226 | gan → gan | cnn → candidateE2_tf_lowlambda_expanded672 |

Note: samples [12, 92] are in the MT-GAN-wins set (GAN beat CNN on MT in baseline).

### Q6. What happens in rare topology-CNN controls (samples 162 and 163)?

These samples were originally topology-CNN controls (CNN wins on topology metrics). candidateE2_tf_lowlambda_expanded672 should not reverse this.

| sample_idx | PD CNN | PD GAN | PD candidateE2_tf_lowlambda_expanded672 | MT CNN | MT GAN | MT candidateE2_tf_lowlambda_expanded672 | PD winner → | MT winner → |
|---|---|---|---|---|---|---|---|---|
| 162 | 18.2711 | 19.4579 | 19.0390 | 4.5406 | 9.6957 | 5.5033 | cnn → cnn | cnn → cnn |
| 163 | 19.3704 | 20.4464 | 20.2104 | 5.6989 | 6.7890 | 5.0420 | cnn → cnn | cnn → candidateE2_tf_lowlambda_expanded672 |

## Winner distribution

### PD distance winners

| Method | Wins before candidateE2_tf_lowlambda_expanded672 | Wins after candidateE2_tf_lowlambda_expanded672 |
|---|---|---|
| candidateE2_tf_lowlambda_expanded672 | 0 | 2 |
| cnn | 2 | 2 |
| gan | 166 | 164 |

### MT distance winners

| Method | Wins before candidateE2_tf_lowlambda_expanded672 | Wins after candidateE2_tf_lowlambda_expanded672 |
|---|---|---|
| candidateE2_tf_lowlambda_expanded672 | 0 | 92 |
| cnn | 148 | 70 |
| gan | 20 | 6 |

## Full per-sample detail: focus samples

Includes adjacent-cluster samples, rare controls, and the 20 MT-GAN baseline wins.

| sample | PD CNN | PD GAN | PD candidateE2_tf_lowlambda_expanded672 | MT CNN | MT GAN | MT candidateE2_tf_lowlambda_expanded672 | MT-GAN-win? | PD after | MT after |
|---|---|---|---|---|---|---|---|---|---|
| 6 | 28.0690 | 24.0451 | 25.8209 | 6.1440 | 6.0106 | 6.1556 | ✓ | gan | gan |
| 8 | 29.6351 | 23.2821 | 27.3382 | 7.5096 | 6.0550 | 6.6444 | ✓ | gan | gan |
| 10 | 29.6970 | 24.7100 | 26.8889 | 6.6345 | 7.4238 | 5.8768 |  | gan | candidateE2_tf_lowlambda_expanded672 |
| 11 | 28.6873 | 24.9699 | 25.6334 | 5.9107 | 7.8317 | 5.9130 |  | gan | cnn |
| 12 | 30.1045 | 24.2206 | 26.9259 | 6.6164 | 5.9137 | 6.1441 | ✓ | gan | gan |
| 13 | 29.2737 | 24.3237 | 26.6505 | 6.0738 | 7.4023 | 6.1800 |  | gan | cnn |
| 16 | 23.7560 | 17.3245 | 21.4824 | 5.4642 | 5.2552 | 4.8143 | ✓ | gan | candidateE2_tf_lowlambda_expanded672 |
| 17 | 20.9004 | 17.1181 | 19.2329 | 5.2897 | 4.8487 | 4.3087 | ✓ | gan | candidateE2_tf_lowlambda_expanded672 |
| 18 | 20.3949 | 18.6102 | 18.5078 | 6.2586 | 5.5544 | 4.6770 | ✓ | candidateE2_tf_lowlambda_expanded672 | candidateE2_tf_lowlambda_expanded672 |
| 19 | 21.0641 | 19.0867 | 18.4937 | 6.9280 | 5.4429 | 5.4430 | ✓ | candidateE2_tf_lowlambda_expanded672 | gan |
| 20 | 21.9792 | 18.2663 | 19.5098 | 6.2541 | 5.6891 | 5.7873 | ✓ | gan | gan |
| 25 | 36.5841 | 30.1235 | 33.3459 | 11.3716 | 10.5015 | 7.7876 | ✓ | gan | candidateE2_tf_lowlambda_expanded672 |
| 48 | 22.8325 | 13.1070 | 18.5068 | 7.0815 | 6.6830 | 6.6124 | ✓ | gan | candidateE2_tf_lowlambda_expanded672 |
| 62 | 27.4635 | 17.9937 | 24.3533 | 6.7203 | 6.6502 | 6.1064 | ✓ | gan | candidateE2_tf_lowlambda_expanded672 |
| 63 | 28.5246 | 18.1998 | 24.8923 | 7.6606 | 7.5776 | 6.2031 | ✓ | gan | candidateE2_tf_lowlambda_expanded672 |
| 65 | 22.1489 | 16.2395 | 19.4216 | 5.8208 | 5.5768 | 4.6336 | ✓ | gan | candidateE2_tf_lowlambda_expanded672 |
| 68 | 21.7713 | 15.5009 | 18.5216 | 6.0743 | 5.6709 | 4.7991 | ✓ | gan | candidateE2_tf_lowlambda_expanded672 |
| 77 | 29.9447 | 20.3533 | 26.1917 | 6.1320 | 6.0000 | 5.5826 | ✓ | gan | candidateE2_tf_lowlambda_expanded672 |
| 79 | 30.1481 | 20.3212 | 26.9384 | 6.8565 | 6.4670 | 5.8018 | ✓ | gan | candidateE2_tf_lowlambda_expanded672 |
| 80 | 30.0863 | 20.7449 | 26.4730 | 6.9827 | 6.1767 | 5.0462 | ✓ | gan | candidateE2_tf_lowlambda_expanded672 |
| 82 | 28.9009 | 20.8163 | 25.4770 | 7.1841 | 6.3036 | 5.7384 | ✓ | gan | candidateE2_tf_lowlambda_expanded672 |
| 90 | 22.2068 | 17.3268 | 19.6113 | 5.2579 | 5.6635 | 4.9208 |  | gan | candidateE2_tf_lowlambda_expanded672 |
| 91 | 22.3094 | 16.2661 | 19.5785 | 5.5083 | 5.5198 | 5.5519 |  | gan | cnn |
| 92 | 22.0793 | 16.8077 | 19.7510 | 5.7358 | 5.6779 | 5.6612 | ✓ | gan | candidateE2_tf_lowlambda_expanded672 |
| 93 | 24.4029 | 18.1355 | 22.2233 | 6.5070 | 6.8160 | 6.2226 |  | gan | candidateE2_tf_lowlambda_expanded672 |
| 154 | 25.3102 | 16.8345 | 22.0549 | 5.8898 | 5.7368 | 6.5947 | ✓ | gan | gan |
| 162 | 18.2711 | 19.4579 | 19.0390 | 4.5406 | 9.6957 | 5.5033 |  | cnn | cnn |
| 163 | 19.3704 | 20.4464 | 20.2104 | 5.6989 | 6.7890 | 5.0420 |  | cnn | candidateE2_tf_lowlambda_expanded672 |

## Output files

- `ttk_runs_fixed/topology_finetuning/candidateE2_tf_lowlambda_expanded672_topology/candidateE2_tf_lowlambda_expanded672_pd_mt_distances.csv` — candidateE2_tf_lowlambda_expanded672 PD/MT per sample
- `ttk_runs_fixed/topology_finetuning/candidateE2_tf_lowlambda_expanded672_topology/candidateE2_tf_lowlambda_expanded672_topology_comparison.csv` — three-way comparison (CNN, GAN, candidateE2_tf_lowlambda_expanded672)
- `docs/topology_finetuning_candidateE2_tf_lowlambda_expanded672_topology_eval.md` — this report

## Notes

- PD distance = bottleneck distance between persistence diagrams (computed by `ttkBottleneckDistance`).
- MT distance = Wasserstein-type merge tree distance (computed by `ttkMergeTreeDistanceMatrix`).
- Both use 160×160 speed patches at (x0=0, y0=0), consistent with the CNN/GAN baseline.
- **Lower distance = SR is topologically closer to GT. Lower is better for both metrics.**
- **Component-count proxy metrics** (counts of connected components in speed superlevel sets) are reported separately in `topology_finetuning_candidateE2_tf_lowlambda_expanded672_eval.md` and do not require TTK.
- candidateE2_tf_lowlambda_expanded672 was fine-tuned from the pretrained CNN checkpoint. Results may change with different lambda settings or more epochs.
- Existing CNN/GAN and Candidate B topology outputs were not overwritten.
