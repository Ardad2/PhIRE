# candidateF_grad_crit_expanded2688 topology evaluation

**Generated:** 2026-07-19

**Important:** This report covers **true topology metrics** (PD bottleneck distance, MT Wasserstein distance) computed by TTK. These are distinct from the component-count *proxy* metrics in `topology_finetuning_candidateF_grad_crit_expanded2688_eval.md`.

**Note:** Lower PD/MT distance = SR is topologically closer to GT. **Lower is better for both metrics.**

**candidateF_grad_crit_expanded2688:** fine-tuned from the pretrained CNN checkpoint with auxiliary physics/topology losses. See the corresponding fine-tuning pilot script for configuration details.

**Samples evaluated:** 168

**MT-GAN baseline wins (before candidateF_grad_crit_expanded2688):** 20 samples — [6, 8, 12, 16, 17, 18, 19, 20, 25, 48, 62, 63, 65, 68, 77, 79, 80, 82, 92, 154]

## Summary statistics

| Metric | CNN (baseline) | GAN (baseline) | candidateF_grad_crit_expanded2688 |
|---|---|---|---|
| PD distance (mean) | 27.4063 | 20.8641 | 22.0179 |
| MT distance (mean) | 5.8678 | 8.3481 | 5.9840 |
| PD wins (vs CNN) | — | 166 | 168 |
| MT wins (vs CNN) | — | 20 | 61 |
| PD beats GAN | — | — | 35 |
| MT beats GAN | — | — | 152 |

## 6 Key evaluation questions

### Q1. Does candidateF_grad_crit_expanded2688 improve PD distance relative to baseline CNN?

- Mean PD: CNN=27.4063, candidateF_grad_crit_expanded2688=22.0179, Δ=-5.3884 (▼ better)
- candidateF_grad_crit_expanded2688 has lower PD on **168/168** samples.

### Q2. Does candidateF_grad_crit_expanded2688 ever beat GAN on PD distance?

- candidateF_grad_crit_expanded2688 beats GAN on PD for **35/168** samples.
- Mean PD: GAN=20.8641, candidateF_grad_crit_expanded2688=22.0179. Δ=1.1538 (▲ worse than GAN on average)

### Q3. Does candidateF_grad_crit_expanded2688 improve MT distance relative to baseline CNN?

- Mean MT: CNN=5.8678, candidateF_grad_crit_expanded2688=5.9840, Δ=0.1162 (▲ worse)
- candidateF_grad_crit_expanded2688 has lower MT on **61/168** samples.

### Q4. Does candidateF_grad_crit_expanded2688 change any of the original 20 MT-GAN cases?

The 20 samples where GAN originally beat CNN on MT distance: [6, 8, 12, 16, 17, 18, 19, 20, 25, 48, 62, 63, 65, 68, 77, 79, 80, 82, 92, 154]

| Outcome | Count |
|---|---|
| MT winner stays GAN | 8 |
| MT winner changes to candidateF_grad_crit_expanded2688 | 12 |
| MT winner changes to CNN | 0 |

Per-sample detail for changed cases:

| sample_idx | MT CNN | MT GAN | MT candidateF_grad_crit_expanded2688 | winner before | winner after |
|---|---|---|---|---|---|
| 18 | 6.2586 | 5.5544 | 5.0972 | gan | candidateF_grad_crit_expanded2688 |
| 19 | 6.9280 | 5.4429 | 5.2407 | gan | candidateF_grad_crit_expanded2688 |
| 20 | 6.2541 | 5.6891 | 5.3770 | gan | candidateF_grad_crit_expanded2688 |
| 25 | 11.3716 | 10.5015 | 7.2374 | gan | candidateF_grad_crit_expanded2688 |
| 62 | 6.7203 | 6.6502 | 6.4454 | gan | candidateF_grad_crit_expanded2688 |
| 63 | 7.6606 | 7.5776 | 6.7521 | gan | candidateF_grad_crit_expanded2688 |
| 68 | 6.0743 | 5.6709 | 5.4392 | gan | candidateF_grad_crit_expanded2688 |
| 77 | 6.1320 | 6.0000 | 5.2626 | gan | candidateF_grad_crit_expanded2688 |
| 79 | 6.8565 | 6.4670 | 5.7180 | gan | candidateF_grad_crit_expanded2688 |
| 80 | 6.9827 | 6.1767 | 5.4402 | gan | candidateF_grad_crit_expanded2688 |
| 82 | 7.1841 | 6.3036 | 5.6944 | gan | candidateF_grad_crit_expanded2688 |
| 154 | 5.8898 | 5.7368 | 5.5770 | gan | candidateF_grad_crit_expanded2688 |

### Q5. What happens in adjacent clusters (samples 10–13 and 90–93)?

| sample_idx | PD CNN | PD GAN | PD candidateF_grad_crit_expanded2688 | MT CNN | MT GAN | MT candidateF_grad_crit_expanded2688 | PD winner → | MT winner → |
|---|---|---|---|---|---|---|---|---|
| 10 | 29.6970 | 24.7100 | 23.7217 | 6.6345 | 7.4238 | 6.2023 | gan → candidateF_grad_crit_expanded2688 | cnn → candidateF_grad_crit_expanded2688 |
| 11 | 28.6873 | 24.9699 | 23.6653 | 5.9107 | 7.8317 | 6.6390 | gan → candidateF_grad_crit_expanded2688 | cnn → cnn |
| 12 | 30.1045 | 24.2206 | 23.8430 | 6.6164 | 5.9137 | 6.3821 | gan → candidateF_grad_crit_expanded2688 | gan → gan |
| 13 | 29.2737 | 24.3237 | 23.2200 | 6.0738 | 7.4023 | 6.1887 | gan → candidateF_grad_crit_expanded2688 | cnn → cnn |
| 90 | 22.2068 | 17.3268 | 17.3119 | 5.2579 | 5.6635 | 5.7377 | gan → candidateF_grad_crit_expanded2688 | cnn → cnn |
| 91 | 22.3094 | 16.2661 | 16.9997 | 5.5083 | 5.5198 | 6.4169 | gan → gan | cnn → cnn |
| 92 | 22.0793 | 16.8077 | 17.4856 | 5.7358 | 5.6779 | 5.9874 | gan → gan | gan → gan |
| 93 | 24.4029 | 18.1355 | 20.0698 | 6.5070 | 6.8160 | 6.4994 | gan → gan | cnn → candidateF_grad_crit_expanded2688 |

Note: samples [12, 92] are in the MT-GAN-wins set (GAN beat CNN on MT in baseline).

### Q6. What happens in rare topology-CNN controls (samples 162 and 163)?

These samples were originally topology-CNN controls (CNN wins on topology metrics). candidateF_grad_crit_expanded2688 should not reverse this.

| sample_idx | PD CNN | PD GAN | PD candidateF_grad_crit_expanded2688 | MT CNN | MT GAN | MT candidateF_grad_crit_expanded2688 | PD winner → | MT winner → |
|---|---|---|---|---|---|---|---|---|
| 162 | 18.2711 | 19.4579 | 17.6822 | 4.5406 | 9.6957 | 4.9154 | cnn → candidateF_grad_crit_expanded2688 | cnn → cnn |
| 163 | 19.3704 | 20.4464 | 18.7545 | 5.6989 | 6.7890 | 5.3746 | cnn → candidateF_grad_crit_expanded2688 | cnn → candidateF_grad_crit_expanded2688 |

## Winner distribution

### PD distance winners

| Method | Wins before candidateF_grad_crit_expanded2688 | Wins after candidateF_grad_crit_expanded2688 |
|---|---|---|
| candidateF_grad_crit_expanded2688 | 0 | 35 |
| cnn | 2 | 0 |
| gan | 166 | 133 |

### MT distance winners

| Method | Wins before candidateF_grad_crit_expanded2688 | Wins after candidateF_grad_crit_expanded2688 |
|---|---|---|
| candidateF_grad_crit_expanded2688 | 0 | 54 |
| cnn | 148 | 106 |
| gan | 20 | 8 |

## Full per-sample detail: focus samples

Includes adjacent-cluster samples, rare controls, and the 20 MT-GAN baseline wins.

| sample | PD CNN | PD GAN | PD candidateF_grad_crit_expanded2688 | MT CNN | MT GAN | MT candidateF_grad_crit_expanded2688 | MT-GAN-win? | PD after | MT after |
|---|---|---|---|---|---|---|---|---|---|
| 6 | 28.0690 | 24.0451 | 23.5062 | 6.1440 | 6.0106 | 6.1303 | ✓ | candidateF_grad_crit_expanded2688 | gan |
| 8 | 29.6351 | 23.2821 | 23.3542 | 7.5096 | 6.0550 | 6.1602 | ✓ | gan | gan |
| 10 | 29.6970 | 24.7100 | 23.7217 | 6.6345 | 7.4238 | 6.2023 |  | candidateF_grad_crit_expanded2688 | candidateF_grad_crit_expanded2688 |
| 11 | 28.6873 | 24.9699 | 23.6653 | 5.9107 | 7.8317 | 6.6390 |  | candidateF_grad_crit_expanded2688 | cnn |
| 12 | 30.1045 | 24.2206 | 23.8430 | 6.6164 | 5.9137 | 6.3821 | ✓ | candidateF_grad_crit_expanded2688 | gan |
| 13 | 29.2737 | 24.3237 | 23.2200 | 6.0738 | 7.4023 | 6.1887 |  | candidateF_grad_crit_expanded2688 | cnn |
| 16 | 23.7560 | 17.3245 | 18.4402 | 5.4642 | 5.2552 | 5.3069 | ✓ | gan | gan |
| 17 | 20.9004 | 17.1181 | 16.4953 | 5.2897 | 4.8487 | 4.8650 | ✓ | candidateF_grad_crit_expanded2688 | gan |
| 18 | 20.3949 | 18.6102 | 15.9938 | 6.2586 | 5.5544 | 5.0972 | ✓ | candidateF_grad_crit_expanded2688 | candidateF_grad_crit_expanded2688 |
| 19 | 21.0641 | 19.0867 | 15.6027 | 6.9280 | 5.4429 | 5.2407 | ✓ | candidateF_grad_crit_expanded2688 | candidateF_grad_crit_expanded2688 |
| 20 | 21.9792 | 18.2663 | 17.5465 | 6.2541 | 5.6891 | 5.3770 | ✓ | candidateF_grad_crit_expanded2688 | candidateF_grad_crit_expanded2688 |
| 25 | 36.5841 | 30.1235 | 29.5073 | 11.3716 | 10.5015 | 7.2374 | ✓ | candidateF_grad_crit_expanded2688 | candidateF_grad_crit_expanded2688 |
| 48 | 22.8325 | 13.1070 | 16.7048 | 7.0815 | 6.6830 | 7.0802 | ✓ | gan | gan |
| 62 | 27.4635 | 17.9937 | 21.6513 | 6.7203 | 6.6502 | 6.4454 | ✓ | gan | candidateF_grad_crit_expanded2688 |
| 63 | 28.5246 | 18.1998 | 22.3546 | 7.6606 | 7.5776 | 6.7521 | ✓ | gan | candidateF_grad_crit_expanded2688 |
| 65 | 22.1489 | 16.2395 | 17.4859 | 5.8208 | 5.5768 | 5.7049 | ✓ | gan | gan |
| 68 | 21.7713 | 15.5009 | 16.1139 | 6.0743 | 5.6709 | 5.4392 | ✓ | gan | candidateF_grad_crit_expanded2688 |
| 77 | 29.9447 | 20.3533 | 23.1304 | 6.1320 | 6.0000 | 5.2626 | ✓ | gan | candidateF_grad_crit_expanded2688 |
| 79 | 30.1481 | 20.3212 | 23.6062 | 6.8565 | 6.4670 | 5.7180 | ✓ | gan | candidateF_grad_crit_expanded2688 |
| 80 | 30.0863 | 20.7449 | 23.3866 | 6.9827 | 6.1767 | 5.4402 | ✓ | gan | candidateF_grad_crit_expanded2688 |
| 82 | 28.9009 | 20.8163 | 23.4552 | 7.1841 | 6.3036 | 5.6944 | ✓ | gan | candidateF_grad_crit_expanded2688 |
| 90 | 22.2068 | 17.3268 | 17.3119 | 5.2579 | 5.6635 | 5.7377 |  | candidateF_grad_crit_expanded2688 | cnn |
| 91 | 22.3094 | 16.2661 | 16.9997 | 5.5083 | 5.5198 | 6.4169 |  | gan | cnn |
| 92 | 22.0793 | 16.8077 | 17.4856 | 5.7358 | 5.6779 | 5.9874 | ✓ | gan | gan |
| 93 | 24.4029 | 18.1355 | 20.0698 | 6.5070 | 6.8160 | 6.4994 |  | gan | candidateF_grad_crit_expanded2688 |
| 154 | 25.3102 | 16.8345 | 18.9302 | 5.8898 | 5.7368 | 5.5770 | ✓ | gan | candidateF_grad_crit_expanded2688 |
| 162 | 18.2711 | 19.4579 | 17.6822 | 4.5406 | 9.6957 | 4.9154 |  | candidateF_grad_crit_expanded2688 | cnn |
| 163 | 19.3704 | 20.4464 | 18.7545 | 5.6989 | 6.7890 | 5.3746 |  | candidateF_grad_crit_expanded2688 | candidateF_grad_crit_expanded2688 |

## Output files

- `ttk_runs_fixed/topology_finetuning/candidateF_grad_crit_expanded2688_topology/candidateF_grad_crit_expanded2688_pd_mt_distances.csv` — candidateF_grad_crit_expanded2688 PD/MT per sample
- `ttk_runs_fixed/topology_finetuning/candidateF_grad_crit_expanded2688_topology/candidateF_grad_crit_expanded2688_topology_comparison.csv` — three-way comparison (CNN, GAN, candidateF_grad_crit_expanded2688)
- `docs/topology_finetuning_candidateF_grad_crit_expanded2688_topology_eval.md` — this report

## Notes

- PD distance = bottleneck distance between persistence diagrams (computed by `ttkBottleneckDistance`).
- MT distance = Wasserstein-type merge tree distance (computed by `ttkMergeTreeDistanceMatrix`).
- Both use 160×160 speed patches at (x0=0, y0=0), consistent with the CNN/GAN baseline.
- **Lower distance = SR is topologically closer to GT. Lower is better for both metrics.**
- **Component-count proxy metrics** (counts of connected components in speed superlevel sets) are reported separately in `topology_finetuning_candidateF_grad_crit_expanded2688_eval.md` and do not require TTK.
- candidateF_grad_crit_expanded2688 was fine-tuned from the pretrained CNN checkpoint. Results may change with different lambda settings or more epochs.
- Existing CNN/GAN and Candidate B topology outputs were not overwritten.
