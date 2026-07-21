# candidateE2_fixed_lowlambda topology evaluation

**Generated:** 2026-07-07

**Important:** This report covers **true topology metrics** (PD bottleneck distance, MT Wasserstein distance) computed by TTK. These are distinct from the component-count *proxy* metrics in `topology_finetuning_candidateE2_fixed_lowlambda_eval.md`.

**Note:** Lower PD/MT distance = SR is topologically closer to GT. **Lower is better for both metrics.**

**candidateE2_fixed_lowlambda:** fine-tuned from the pretrained CNN checkpoint with auxiliary physics/topology losses. See the corresponding fine-tuning pilot script for configuration details.

**Samples evaluated:** 168

**MT-GAN baseline wins (before candidateE2_fixed_lowlambda):** 20 samples — [6, 8, 12, 16, 17, 18, 19, 20, 25, 48, 62, 63, 65, 68, 77, 79, 80, 82, 92, 154]

## Summary statistics

| Metric | CNN (baseline) | GAN (baseline) | candidateE2_fixed_lowlambda |
|---|---|---|---|
| PD distance (mean) | 27.4063 | 20.8641 | 27.1011 |
| MT distance (mean) | 5.8678 | 8.3481 | 5.7522 |
| PD wins (vs CNN) | — | 166 | 130 |
| MT wins (vs CNN) | — | 20 | 113 |
| PD beats GAN | — | — | 2 |
| MT beats GAN | — | — | 153 |

## 6 Key evaluation questions

### Q1. Does candidateE2_fixed_lowlambda improve PD distance relative to baseline CNN?

- Mean PD: CNN=27.4063, candidateE2_fixed_lowlambda=27.1011, Δ=-0.3052 (▼ better)
- candidateE2_fixed_lowlambda has lower PD on **130/168** samples.

### Q2. Does candidateE2_fixed_lowlambda ever beat GAN on PD distance?

- candidateE2_fixed_lowlambda beats GAN on PD for **2/168** samples.
- Mean PD: GAN=20.8641, candidateE2_fixed_lowlambda=27.1011. Δ=6.2370 (▲ worse than GAN on average)

### Q3. Does candidateE2_fixed_lowlambda improve MT distance relative to baseline CNN?

- Mean MT: CNN=5.8678, candidateE2_fixed_lowlambda=5.7522, Δ=-0.1156 (▼ better)
- candidateE2_fixed_lowlambda has lower MT on **113/168** samples.

### Q4. Does candidateE2_fixed_lowlambda change any of the original 20 MT-GAN cases?

The 20 samples where GAN originally beat CNN on MT distance: [6, 8, 12, 16, 17, 18, 19, 20, 25, 48, 62, 63, 65, 68, 77, 79, 80, 82, 92, 154]

| Outcome | Count |
|---|---|
| MT winner stays GAN | 14 |
| MT winner changes to candidateE2_fixed_lowlambda | 6 |
| MT winner changes to CNN | 0 |

Per-sample detail for changed cases:

| sample_idx | MT CNN | MT GAN | MT candidateE2_fixed_lowlambda | winner before | winner after |
|---|---|---|---|---|---|
| 62 | 6.7203 | 6.6502 | 6.6016 | gan | candidateE2_fixed_lowlambda |
| 63 | 7.6606 | 7.5776 | 6.8442 | gan | candidateE2_fixed_lowlambda |
| 77 | 6.1320 | 6.0000 | 5.6729 | gan | candidateE2_fixed_lowlambda |
| 79 | 6.8565 | 6.4670 | 6.4529 | gan | candidateE2_fixed_lowlambda |
| 82 | 7.1841 | 6.3036 | 6.2027 | gan | candidateE2_fixed_lowlambda |
| 92 | 5.7358 | 5.6779 | 5.6751 | gan | candidateE2_fixed_lowlambda |

### Q5. What happens in adjacent clusters (samples 10–13 and 90–93)?

| sample_idx | PD CNN | PD GAN | PD candidateE2_fixed_lowlambda | MT CNN | MT GAN | MT candidateE2_fixed_lowlambda | PD winner → | MT winner → |
|---|---|---|---|---|---|---|---|---|
| 10 | 29.6970 | 24.7100 | 28.8450 | 6.6345 | 7.4238 | 6.2342 | gan → gan | cnn → candidateE2_fixed_lowlambda |
| 11 | 28.6873 | 24.9699 | 27.6204 | 5.9107 | 7.8317 | 5.8793 | gan → gan | cnn → candidateE2_fixed_lowlambda |
| 12 | 30.1045 | 24.2206 | 29.1357 | 6.6164 | 5.9137 | 6.2749 | gan → gan | gan → gan |
| 13 | 29.2737 | 24.3237 | 28.2472 | 6.0738 | 7.4023 | 6.2030 | gan → gan | cnn → cnn |
| 90 | 22.2068 | 17.3268 | 22.0570 | 5.2579 | 5.6635 | 5.3094 | gan → gan | cnn → cnn |
| 91 | 22.3094 | 16.2661 | 22.1984 | 5.5083 | 5.5198 | 5.4756 | gan → gan | cnn → candidateE2_fixed_lowlambda |
| 92 | 22.0793 | 16.8077 | 21.9905 | 5.7358 | 5.6779 | 5.6751 | gan → gan | gan → candidateE2_fixed_lowlambda |
| 93 | 24.4029 | 18.1355 | 24.4547 | 6.5070 | 6.8160 | 6.3231 | gan → gan | cnn → candidateE2_fixed_lowlambda |

Note: samples [12, 92] are in the MT-GAN-wins set (GAN beat CNN on MT in baseline).

### Q6. What happens in rare topology-CNN controls (samples 162 and 163)?

These samples were originally topology-CNN controls (CNN wins on topology metrics). candidateE2_fixed_lowlambda should not reverse this.

| sample_idx | PD CNN | PD GAN | PD candidateE2_fixed_lowlambda | MT CNN | MT GAN | MT candidateE2_fixed_lowlambda | PD winner → | MT winner → |
|---|---|---|---|---|---|---|---|---|
| 162 | 18.2711 | 19.4579 | 18.7087 | 4.5406 | 9.6957 | 4.6073 | cnn → cnn | cnn → cnn |
| 163 | 19.3704 | 20.4464 | 19.8056 | 5.6989 | 6.7890 | 5.2034 | cnn → cnn | cnn → candidateE2_fixed_lowlambda |

## Winner distribution

### PD distance winners

| Method | Wins before candidateE2_fixed_lowlambda | Wins after candidateE2_fixed_lowlambda |
|---|---|---|
| cnn | 2 | 2 |
| gan | 166 | 166 |

### MT distance winners

| Method | Wins before candidateE2_fixed_lowlambda | Wins after candidateE2_fixed_lowlambda |
|---|---|---|
| candidateE2_fixed_lowlambda | 0 | 101 |
| cnn | 148 | 53 |
| gan | 20 | 14 |

## Full per-sample detail: focus samples

Includes adjacent-cluster samples, rare controls, and the 20 MT-GAN baseline wins.

| sample | PD CNN | PD GAN | PD candidateE2_fixed_lowlambda | MT CNN | MT GAN | MT candidateE2_fixed_lowlambda | MT-GAN-win? | PD after | MT after |
|---|---|---|---|---|---|---|---|---|---|
| 6 | 28.0690 | 24.0451 | 28.1661 | 6.1440 | 6.0106 | 6.1420 | ✓ | gan | gan |
| 8 | 29.6351 | 23.2821 | 29.1657 | 7.5096 | 6.0550 | 7.4740 | ✓ | gan | gan |
| 10 | 29.6970 | 24.7100 | 28.8450 | 6.6345 | 7.4238 | 6.2342 |  | gan | candidateE2_fixed_lowlambda |
| 11 | 28.6873 | 24.9699 | 27.6204 | 5.9107 | 7.8317 | 5.8793 |  | gan | candidateE2_fixed_lowlambda |
| 12 | 30.1045 | 24.2206 | 29.1357 | 6.6164 | 5.9137 | 6.2749 | ✓ | gan | gan |
| 13 | 29.2737 | 24.3237 | 28.2472 | 6.0738 | 7.4023 | 6.2030 |  | gan | cnn |
| 16 | 23.7560 | 17.3245 | 22.8815 | 5.4642 | 5.2552 | 5.2626 | ✓ | gan | gan |
| 17 | 20.9004 | 17.1181 | 19.9150 | 5.2897 | 4.8487 | 5.2797 | ✓ | gan | gan |
| 18 | 20.3949 | 18.6102 | 19.4712 | 6.2586 | 5.5544 | 5.9136 | ✓ | gan | gan |
| 19 | 21.0641 | 19.0867 | 20.0640 | 6.9280 | 5.4429 | 6.5348 | ✓ | gan | gan |
| 20 | 21.9792 | 18.2663 | 20.7004 | 6.2541 | 5.6891 | 5.7708 | ✓ | gan | gan |
| 25 | 36.5841 | 30.1235 | 35.5579 | 11.3716 | 10.5015 | 11.3849 | ✓ | gan | gan |
| 48 | 22.8325 | 13.1070 | 22.8092 | 7.0815 | 6.6830 | 7.0039 | ✓ | gan | gan |
| 62 | 27.4635 | 17.9937 | 27.2907 | 6.7203 | 6.6502 | 6.6016 | ✓ | gan | candidateE2_fixed_lowlambda |
| 63 | 28.5246 | 18.1998 | 28.2108 | 7.6606 | 7.5776 | 6.8442 | ✓ | gan | candidateE2_fixed_lowlambda |
| 65 | 22.1489 | 16.2395 | 22.0292 | 5.8208 | 5.5768 | 5.5946 | ✓ | gan | gan |
| 68 | 21.7713 | 15.5009 | 21.6391 | 6.0743 | 5.6709 | 5.7470 | ✓ | gan | gan |
| 77 | 29.9447 | 20.3533 | 29.5506 | 6.1320 | 6.0000 | 5.6729 | ✓ | gan | candidateE2_fixed_lowlambda |
| 79 | 30.1481 | 20.3212 | 29.8420 | 6.8565 | 6.4670 | 6.4529 | ✓ | gan | candidateE2_fixed_lowlambda |
| 80 | 30.0863 | 20.7449 | 29.6724 | 6.9827 | 6.1767 | 6.5758 | ✓ | gan | gan |
| 82 | 28.9009 | 20.8163 | 28.5167 | 7.1841 | 6.3036 | 6.2027 | ✓ | gan | candidateE2_fixed_lowlambda |
| 90 | 22.2068 | 17.3268 | 22.0570 | 5.2579 | 5.6635 | 5.3094 |  | gan | cnn |
| 91 | 22.3094 | 16.2661 | 22.1984 | 5.5083 | 5.5198 | 5.4756 |  | gan | candidateE2_fixed_lowlambda |
| 92 | 22.0793 | 16.8077 | 21.9905 | 5.7358 | 5.6779 | 5.6751 | ✓ | gan | candidateE2_fixed_lowlambda |
| 93 | 24.4029 | 18.1355 | 24.4547 | 6.5070 | 6.8160 | 6.3231 |  | gan | candidateE2_fixed_lowlambda |
| 154 | 25.3102 | 16.8345 | 25.3454 | 5.8898 | 5.7368 | 6.2615 | ✓ | gan | gan |
| 162 | 18.2711 | 19.4579 | 18.7087 | 4.5406 | 9.6957 | 4.6073 |  | cnn | cnn |
| 163 | 19.3704 | 20.4464 | 19.8056 | 5.6989 | 6.7890 | 5.2034 |  | cnn | candidateE2_fixed_lowlambda |

## Output files

- `ttk_runs_fixed/topology_finetuning/candidateE2_fixed_lowlambda_topology/candidateE2_fixed_lowlambda_pd_mt_distances.csv` — candidateE2_fixed_lowlambda PD/MT per sample
- `ttk_runs_fixed/topology_finetuning/candidateE2_fixed_lowlambda_topology/candidateE2_fixed_lowlambda_topology_comparison.csv` — three-way comparison (CNN, GAN, candidateE2_fixed_lowlambda)
- `docs/topology_finetuning_candidateE2_fixed_lowlambda_topology_eval.md` — this report

## Notes

- PD distance = bottleneck distance between persistence diagrams (computed by `ttkBottleneckDistance`).
- MT distance = Wasserstein-type merge tree distance (computed by `ttkMergeTreeDistanceMatrix`).
- Both use 160×160 speed patches at (x0=0, y0=0), consistent with the CNN/GAN baseline.
- **Lower distance = SR is topologically closer to GT. Lower is better for both metrics.**
- **Component-count proxy metrics** (counts of connected components in speed superlevel sets) are reported separately in `topology_finetuning_candidateE2_fixed_lowlambda_eval.md` and do not require TTK.
- candidateE2_fixed_lowlambda was fine-tuned from the pretrained CNN checkpoint. Results may change with different lambda settings or more epochs.
- Existing CNN/GAN and Candidate B topology outputs were not overwritten.
