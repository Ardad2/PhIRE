# candidateE2_fixed_lowlambda_expanded2688 topology evaluation

**Generated:** 2026-07-08

**Important:** This report covers **true topology metrics** (PD bottleneck distance, MT Wasserstein distance) computed by TTK. These are distinct from the component-count *proxy* metrics in `topology_finetuning_candidateE2_fixed_lowlambda_expanded2688_eval.md`.

**Note:** Lower PD/MT distance = SR is topologically closer to GT. **Lower is better for both metrics.**

**candidateE2_fixed_lowlambda_expanded2688:** fine-tuned from the pretrained CNN checkpoint with auxiliary physics/topology losses. See the corresponding fine-tuning pilot script for configuration details.

**Samples evaluated:** 168

**MT-GAN baseline wins (before candidateE2_fixed_lowlambda_expanded2688):** 20 samples — [6, 8, 12, 16, 17, 18, 19, 20, 25, 48, 62, 63, 65, 68, 77, 79, 80, 82, 92, 154]

## Summary statistics

| Metric | CNN (baseline) | GAN (baseline) | candidateE2_fixed_lowlambda_expanded2688 |
|---|---|---|---|
| PD distance (mean) | 27.4063 | 20.8641 | 26.4934 |
| MT distance (mean) | 5.8678 | 8.3481 | 5.6989 |
| PD wins (vs CNN) | — | 166 | 142 |
| MT wins (vs CNN) | — | 20 | 118 |
| PD beats GAN | — | — | 2 |
| MT beats GAN | — | — | 159 |

## 6 Key evaluation questions

### Q1. Does candidateE2_fixed_lowlambda_expanded2688 improve PD distance relative to baseline CNN?

- Mean PD: CNN=27.4063, candidateE2_fixed_lowlambda_expanded2688=26.4934, Δ=-0.9129 (▼ better)
- candidateE2_fixed_lowlambda_expanded2688 has lower PD on **142/168** samples.

### Q2. Does candidateE2_fixed_lowlambda_expanded2688 ever beat GAN on PD distance?

- candidateE2_fixed_lowlambda_expanded2688 beats GAN on PD for **2/168** samples.
- Mean PD: GAN=20.8641, candidateE2_fixed_lowlambda_expanded2688=26.4934. Δ=5.6293 (▲ worse than GAN on average)

### Q3. Does candidateE2_fixed_lowlambda_expanded2688 improve MT distance relative to baseline CNN?

- Mean MT: CNN=5.8678, candidateE2_fixed_lowlambda_expanded2688=5.6989, Δ=-0.1689 (▼ better)
- candidateE2_fixed_lowlambda_expanded2688 has lower MT on **118/168** samples.

### Q4. Does candidateE2_fixed_lowlambda_expanded2688 change any of the original 20 MT-GAN cases?

The 20 samples where GAN originally beat CNN on MT distance: [6, 8, 12, 16, 17, 18, 19, 20, 25, 48, 62, 63, 65, 68, 77, 79, 80, 82, 92, 154]

| Outcome | Count |
|---|---|
| MT winner stays GAN | 9 |
| MT winner changes to candidateE2_fixed_lowlambda_expanded2688 | 11 |
| MT winner changes to CNN | 0 |

Per-sample detail for changed cases:

| sample_idx | MT CNN | MT GAN | MT candidateE2_fixed_lowlambda_expanded2688 | winner before | winner after |
|---|---|---|---|---|---|
| 6 | 6.1440 | 6.0106 | 5.3417 | gan | candidateE2_fixed_lowlambda_expanded2688 |
| 17 | 5.2897 | 4.8487 | 4.8482 | gan | candidateE2_fixed_lowlambda_expanded2688 |
| 25 | 11.3716 | 10.5015 | 10.4875 | gan | candidateE2_fixed_lowlambda_expanded2688 |
| 62 | 6.7203 | 6.6502 | 6.5290 | gan | candidateE2_fixed_lowlambda_expanded2688 |
| 63 | 7.6606 | 7.5776 | 7.0652 | gan | candidateE2_fixed_lowlambda_expanded2688 |
| 65 | 5.8208 | 5.5768 | 5.3812 | gan | candidateE2_fixed_lowlambda_expanded2688 |
| 77 | 6.1320 | 6.0000 | 5.3983 | gan | candidateE2_fixed_lowlambda_expanded2688 |
| 79 | 6.8565 | 6.4670 | 6.0836 | gan | candidateE2_fixed_lowlambda_expanded2688 |
| 80 | 6.9827 | 6.1767 | 5.6156 | gan | candidateE2_fixed_lowlambda_expanded2688 |
| 82 | 7.1841 | 6.3036 | 6.2903 | gan | candidateE2_fixed_lowlambda_expanded2688 |
| 92 | 5.7358 | 5.6779 | 5.1302 | gan | candidateE2_fixed_lowlambda_expanded2688 |

### Q5. What happens in adjacent clusters (samples 10–13 and 90–93)?

| sample_idx | PD CNN | PD GAN | PD candidateE2_fixed_lowlambda_expanded2688 | MT CNN | MT GAN | MT candidateE2_fixed_lowlambda_expanded2688 | PD winner → | MT winner → |
|---|---|---|---|---|---|---|---|---|
| 10 | 29.6970 | 24.7100 | 28.5100 | 6.6345 | 7.4238 | 6.1900 | gan → gan | cnn → candidateE2_fixed_lowlambda_expanded2688 |
| 11 | 28.6873 | 24.9699 | 27.3228 | 5.9107 | 7.8317 | 6.1706 | gan → gan | cnn → cnn |
| 12 | 30.1045 | 24.2206 | 28.8227 | 6.6164 | 5.9137 | 6.4627 | gan → gan | gan → gan |
| 13 | 29.2737 | 24.3237 | 27.9391 | 6.0738 | 7.4023 | 5.9380 | gan → gan | cnn → candidateE2_fixed_lowlambda_expanded2688 |
| 90 | 22.2068 | 17.3268 | 21.3299 | 5.2579 | 5.6635 | 5.0530 | gan → gan | cnn → candidateE2_fixed_lowlambda_expanded2688 |
| 91 | 22.3094 | 16.2661 | 21.2763 | 5.5083 | 5.5198 | 5.2492 | gan → gan | cnn → candidateE2_fixed_lowlambda_expanded2688 |
| 92 | 22.0793 | 16.8077 | 21.1016 | 5.7358 | 5.6779 | 5.1302 | gan → gan | gan → candidateE2_fixed_lowlambda_expanded2688 |
| 93 | 24.4029 | 18.1355 | 23.6910 | 6.5070 | 6.8160 | 6.2238 | gan → gan | cnn → candidateE2_fixed_lowlambda_expanded2688 |

Note: samples [12, 92] are in the MT-GAN-wins set (GAN beat CNN on MT in baseline).

### Q6. What happens in rare topology-CNN controls (samples 162 and 163)?

These samples were originally topology-CNN controls (CNN wins on topology metrics). candidateE2_fixed_lowlambda_expanded2688 should not reverse this.

| sample_idx | PD CNN | PD GAN | PD candidateE2_fixed_lowlambda_expanded2688 | MT CNN | MT GAN | MT candidateE2_fixed_lowlambda_expanded2688 | PD winner → | MT winner → |
|---|---|---|---|---|---|---|---|---|
| 162 | 18.2711 | 19.4579 | 18.8386 | 4.5406 | 9.6957 | 4.6824 | cnn → cnn | cnn → cnn |
| 163 | 19.3704 | 20.4464 | 19.7477 | 5.6989 | 6.7890 | 5.5210 | cnn → cnn | cnn → candidateE2_fixed_lowlambda_expanded2688 |

## Winner distribution

### PD distance winners

| Method | Wins before candidateE2_fixed_lowlambda_expanded2688 | Wins after candidateE2_fixed_lowlambda_expanded2688 |
|---|---|---|
| cnn | 2 | 2 |
| gan | 166 | 166 |

### MT distance winners

| Method | Wins before candidateE2_fixed_lowlambda_expanded2688 | Wins after candidateE2_fixed_lowlambda_expanded2688 |
|---|---|---|
| candidateE2_fixed_lowlambda_expanded2688 | 0 | 110 |
| cnn | 148 | 49 |
| gan | 20 | 9 |

## Full per-sample detail: focus samples

Includes adjacent-cluster samples, rare controls, and the 20 MT-GAN baseline wins.

| sample | PD CNN | PD GAN | PD candidateE2_fixed_lowlambda_expanded2688 | MT CNN | MT GAN | MT candidateE2_fixed_lowlambda_expanded2688 | MT-GAN-win? | PD after | MT after |
|---|---|---|---|---|---|---|---|---|---|
| 6 | 28.0690 | 24.0451 | 27.5530 | 6.1440 | 6.0106 | 5.3417 | ✓ | gan | candidateE2_fixed_lowlambda_expanded2688 |
| 8 | 29.6351 | 23.2821 | 28.4918 | 7.5096 | 6.0550 | 7.1844 | ✓ | gan | gan |
| 10 | 29.6970 | 24.7100 | 28.5100 | 6.6345 | 7.4238 | 6.1900 |  | gan | candidateE2_fixed_lowlambda_expanded2688 |
| 11 | 28.6873 | 24.9699 | 27.3228 | 5.9107 | 7.8317 | 6.1706 |  | gan | cnn |
| 12 | 30.1045 | 24.2206 | 28.8227 | 6.6164 | 5.9137 | 6.4627 | ✓ | gan | gan |
| 13 | 29.2737 | 24.3237 | 27.9391 | 6.0738 | 7.4023 | 5.9380 |  | gan | candidateE2_fixed_lowlambda_expanded2688 |
| 16 | 23.7560 | 17.3245 | 22.3895 | 5.4642 | 5.2552 | 5.3833 | ✓ | gan | gan |
| 17 | 20.9004 | 17.1181 | 19.2896 | 5.2897 | 4.8487 | 4.8482 | ✓ | gan | candidateE2_fixed_lowlambda_expanded2688 |
| 18 | 20.3949 | 18.6102 | 18.7651 | 6.2586 | 5.5544 | 5.5816 | ✓ | gan | gan |
| 19 | 21.0641 | 19.0867 | 19.2403 | 6.9280 | 5.4429 | 6.2191 | ✓ | gan | gan |
| 20 | 21.9792 | 18.2663 | 20.1800 | 6.2541 | 5.6891 | 5.7153 | ✓ | gan | gan |
| 25 | 36.5841 | 30.1235 | 34.4574 | 11.3716 | 10.5015 | 10.4875 | ✓ | gan | candidateE2_fixed_lowlambda_expanded2688 |
| 48 | 22.8325 | 13.1070 | 22.7496 | 7.0815 | 6.6830 | 6.7406 | ✓ | gan | gan |
| 62 | 27.4635 | 17.9937 | 26.4845 | 6.7203 | 6.6502 | 6.5290 | ✓ | gan | candidateE2_fixed_lowlambda_expanded2688 |
| 63 | 28.5246 | 18.1998 | 27.4396 | 7.6606 | 7.5776 | 7.0652 | ✓ | gan | candidateE2_fixed_lowlambda_expanded2688 |
| 65 | 22.1489 | 16.2395 | 21.3796 | 5.8208 | 5.5768 | 5.3812 | ✓ | gan | candidateE2_fixed_lowlambda_expanded2688 |
| 68 | 21.7713 | 15.5009 | 21.1163 | 6.0743 | 5.6709 | 5.7114 | ✓ | gan | gan |
| 77 | 29.9447 | 20.3533 | 28.3385 | 6.1320 | 6.0000 | 5.3983 | ✓ | gan | candidateE2_fixed_lowlambda_expanded2688 |
| 79 | 30.1481 | 20.3212 | 28.6828 | 6.8565 | 6.4670 | 6.0836 | ✓ | gan | candidateE2_fixed_lowlambda_expanded2688 |
| 80 | 30.0863 | 20.7449 | 28.6674 | 6.9827 | 6.1767 | 5.6156 | ✓ | gan | candidateE2_fixed_lowlambda_expanded2688 |
| 82 | 28.9009 | 20.8163 | 27.5559 | 7.1841 | 6.3036 | 6.2903 | ✓ | gan | candidateE2_fixed_lowlambda_expanded2688 |
| 90 | 22.2068 | 17.3268 | 21.3299 | 5.2579 | 5.6635 | 5.0530 |  | gan | candidateE2_fixed_lowlambda_expanded2688 |
| 91 | 22.3094 | 16.2661 | 21.2763 | 5.5083 | 5.5198 | 5.2492 |  | gan | candidateE2_fixed_lowlambda_expanded2688 |
| 92 | 22.0793 | 16.8077 | 21.1016 | 5.7358 | 5.6779 | 5.1302 | ✓ | gan | candidateE2_fixed_lowlambda_expanded2688 |
| 93 | 24.4029 | 18.1355 | 23.6910 | 6.5070 | 6.8160 | 6.2238 |  | gan | candidateE2_fixed_lowlambda_expanded2688 |
| 154 | 25.3102 | 16.8345 | 25.4515 | 5.8898 | 5.7368 | 6.3117 | ✓ | gan | gan |
| 162 | 18.2711 | 19.4579 | 18.8386 | 4.5406 | 9.6957 | 4.6824 |  | cnn | cnn |
| 163 | 19.3704 | 20.4464 | 19.7477 | 5.6989 | 6.7890 | 5.5210 |  | cnn | candidateE2_fixed_lowlambda_expanded2688 |

## Output files

- `ttk_runs_fixed/topology_finetuning/candidateE2_fixed_lowlambda_expanded2688_topology/candidateE2_fixed_lowlambda_expanded2688_pd_mt_distances.csv` — candidateE2_fixed_lowlambda_expanded2688 PD/MT per sample
- `ttk_runs_fixed/topology_finetuning/candidateE2_fixed_lowlambda_expanded2688_topology/candidateE2_fixed_lowlambda_expanded2688_topology_comparison.csv` — three-way comparison (CNN, GAN, candidateE2_fixed_lowlambda_expanded2688)
- `docs/topology_finetuning_candidateE2_fixed_lowlambda_expanded2688_topology_eval.md` — this report

## Notes

- PD distance = bottleneck distance between persistence diagrams (computed by `ttkBottleneckDistance`).
- MT distance = Wasserstein-type merge tree distance (computed by `ttkMergeTreeDistanceMatrix`).
- Both use 160×160 speed patches at (x0=0, y0=0), consistent with the CNN/GAN baseline.
- **Lower distance = SR is topologically closer to GT. Lower is better for both metrics.**
- **Component-count proxy metrics** (counts of connected components in speed superlevel sets) are reported separately in `topology_finetuning_candidateE2_fixed_lowlambda_expanded2688_eval.md` and do not require TTK.
- candidateE2_fixed_lowlambda_expanded2688 was fine-tuned from the pretrained CNN checkpoint. Results may change with different lambda settings or more epochs.
- Existing CNN/GAN and Candidate B topology outputs were not overwritten.
