# candidateC_expanded672 topology evaluation

**Generated:** 2026-05-24

**Important:** This report covers **true topology metrics** (PD bottleneck distance, MT Wasserstein distance) computed by TTK. These are distinct from the component-count *proxy* metrics in `topology_finetuning_candidateC_expanded672_eval.md`.

**Note:** Lower PD/MT distance = SR is topologically closer to GT. **Lower is better for both metrics.**

**candidateC_expanded672:** fine-tuned from the pretrained CNN checkpoint with auxiliary physics/topology losses. See the corresponding fine-tuning pilot script for configuration details.

**Candidate C = Candidate B + critical-value/topological-extrema proxy loss** (`lambda_crit=0.001`). Candidate B adds `lambda_speed=0.01, lambda_grad=0.05, lambda_levelset=0.25` on top of the baseline CNN MSE loss.

**Samples evaluated:** 168

**MT-GAN baseline wins (before candidateC_expanded672):** 20 samples — [6, 8, 12, 16, 17, 18, 19, 20, 25, 48, 62, 63, 65, 68, 77, 79, 80, 82, 92, 154]

## Summary statistics

| Metric | CNN (baseline) | GAN (baseline) | candidateC_expanded672 |
|---|---|---|---|
| PD distance (mean) | 27.4063 | 20.8641 | 23.9580 |
| MT distance (mean) | 5.8678 | 8.3481 | 6.0765 |
| PD wins (vs CNN) | — | 166 | 168 |
| MT wins (vs CNN) | — | 20 | 54 |
| PD beats GAN | — | — | 9 |
| MT beats GAN | — | — | 148 |

## 6 Key evaluation questions

### Q1. Does candidateC_expanded672 improve PD distance relative to baseline CNN?

- Mean PD: CNN=27.4063, candidateC_expanded672=23.9580, Δ=-3.4483 (▼ better)
- candidateC_expanded672 has lower PD on **168/168** samples.

### Q2. Does candidateC_expanded672 ever beat GAN on PD distance?

- candidateC_expanded672 beats GAN on PD for **9/168** samples.
- Mean PD: GAN=20.8641, candidateC_expanded672=23.9580. Δ=3.0939 (▲ worse than GAN on average)

### Q3. Does candidateC_expanded672 improve MT distance relative to baseline CNN?

- Mean MT: CNN=5.8678, candidateC_expanded672=6.0765, Δ=0.2087 (▲ worse)
- candidateC_expanded672 has lower MT on **54/168** samples.

### Q4. Does candidateC_expanded672 change any of the original 20 MT-GAN cases?

The 20 samples where GAN originally beat CNN on MT distance: [6, 8, 12, 16, 17, 18, 19, 20, 25, 48, 62, 63, 65, 68, 77, 79, 80, 82, 92, 154]

| Outcome | Count |
|---|---|
| MT winner stays GAN | 13 |
| MT winner changes to candidateC_expanded672 | 7 |
| MT winner changes to CNN | 0 |

Per-sample detail for changed cases:

| sample_idx | MT CNN | MT GAN | MT candidateC_expanded672 | winner before | winner after |
|---|---|---|---|---|---|
| 6 | 6.1440 | 6.0106 | 5.8455 | gan | candidateC_expanded672 |
| 25 | 11.3716 | 10.5015 | 6.8249 | gan | candidateC_expanded672 |
| 63 | 7.6606 | 7.5776 | 6.8043 | gan | candidateC_expanded672 |
| 65 | 5.8208 | 5.5768 | 5.3767 | gan | candidateC_expanded672 |
| 68 | 6.0743 | 5.6709 | 5.0383 | gan | candidateC_expanded672 |
| 80 | 6.9827 | 6.1767 | 6.0845 | gan | candidateC_expanded672 |
| 82 | 7.1841 | 6.3036 | 6.2192 | gan | candidateC_expanded672 |

### Q5. What happens in adjacent clusters (samples 10–13 and 90–93)?

| sample_idx | PD CNN | PD GAN | PD candidateC_expanded672 | MT CNN | MT GAN | MT candidateC_expanded672 | PD winner → | MT winner → |
|---|---|---|---|---|---|---|---|---|
| 10 | 29.6970 | 24.7100 | 25.5068 | 6.6345 | 7.4238 | 6.0416 | gan → gan | cnn → candidateC_expanded672 |
| 11 | 28.6873 | 24.9699 | 24.4656 | 5.9107 | 7.8317 | 6.4353 | gan → candidateC_expanded672 | cnn → cnn |
| 12 | 30.1045 | 24.2206 | 25.1333 | 6.6164 | 5.9137 | 6.6007 | gan → gan | gan → gan |
| 13 | 29.2737 | 24.3237 | 24.9176 | 6.0738 | 7.4023 | 6.8907 | gan → gan | cnn → cnn |
| 90 | 22.2068 | 17.3268 | 19.0367 | 5.2579 | 5.6635 | 5.3563 | gan → gan | cnn → cnn |
| 91 | 22.3094 | 16.2661 | 18.9232 | 5.5083 | 5.5198 | 6.0844 | gan → gan | cnn → cnn |
| 92 | 22.0793 | 16.8077 | 19.3578 | 5.7358 | 5.6779 | 5.7236 | gan → gan | gan → gan |
| 93 | 24.4029 | 18.1355 | 21.8943 | 6.5070 | 6.8160 | 6.0376 | gan → gan | cnn → candidateC_expanded672 |

Note: samples [12, 92] are in the MT-GAN-wins set (GAN beat CNN on MT in baseline).

### Q6. What happens in rare topology-CNN controls (samples 162 and 163)?

These samples were originally topology-CNN controls (CNN wins on topology metrics). candidateC_expanded672 should not reverse this.

| sample_idx | PD CNN | PD GAN | PD candidateC_expanded672 | MT CNN | MT GAN | MT candidateC_expanded672 | PD winner → | MT winner → |
|---|---|---|---|---|---|---|---|---|
| 162 | 18.2711 | 19.4579 | 17.9713 | 4.5406 | 9.6957 | 4.7727 | cnn → candidateC_expanded672 | cnn → cnn |
| 163 | 19.3704 | 20.4464 | 19.0889 | 5.6989 | 6.7890 | 5.5793 | cnn → candidateC_expanded672 | cnn → candidateC_expanded672 |

## Winner distribution

### PD distance winners

| Method | Wins before candidateC_expanded672 | Wins after candidateC_expanded672 |
|---|---|---|
| candidateC_expanded672 | 0 | 9 |
| cnn | 2 | 0 |
| gan | 166 | 159 |

### MT distance winners

| Method | Wins before candidateC_expanded672 | Wins after candidateC_expanded672 |
|---|---|---|
| candidateC_expanded672 | 0 | 47 |
| cnn | 148 | 108 |
| gan | 20 | 13 |

## Full per-sample detail: focus samples

Includes adjacent-cluster samples, rare controls, and the 20 MT-GAN baseline wins.

| sample | PD CNN | PD GAN | PD candidateC_expanded672 | MT CNN | MT GAN | MT candidateC_expanded672 | MT-GAN-win? | PD after | MT after |
|---|---|---|---|---|---|---|---|---|---|
| 6 | 28.0690 | 24.0451 | 24.8726 | 6.1440 | 6.0106 | 5.8455 | ✓ | gan | candidateC_expanded672 |
| 8 | 29.6351 | 23.2821 | 26.1900 | 7.5096 | 6.0550 | 6.2528 | ✓ | gan | gan |
| 10 | 29.6970 | 24.7100 | 25.5068 | 6.6345 | 7.4238 | 6.0416 |  | gan | candidateC_expanded672 |
| 11 | 28.6873 | 24.9699 | 24.4656 | 5.9107 | 7.8317 | 6.4353 |  | candidateC_expanded672 | cnn |
| 12 | 30.1045 | 24.2206 | 25.1333 | 6.6164 | 5.9137 | 6.6007 | ✓ | gan | gan |
| 13 | 29.2737 | 24.3237 | 24.9176 | 6.0738 | 7.4023 | 6.8907 |  | gan | cnn |
| 16 | 23.7560 | 17.3245 | 21.3713 | 5.4642 | 5.2552 | 6.0348 | ✓ | gan | gan |
| 17 | 20.9004 | 17.1181 | 18.8462 | 5.2897 | 4.8487 | 5.3049 | ✓ | gan | gan |
| 18 | 20.3949 | 18.6102 | 18.3174 | 6.2586 | 5.5544 | 5.7461 | ✓ | candidateC_expanded672 | gan |
| 19 | 21.0641 | 19.0867 | 17.8463 | 6.9280 | 5.4429 | 6.2723 | ✓ | candidateC_expanded672 | gan |
| 20 | 21.9792 | 18.2663 | 18.8086 | 6.2541 | 5.6891 | 6.6531 | ✓ | gan | gan |
| 25 | 36.5841 | 30.1235 | 31.5832 | 11.3716 | 10.5015 | 6.8249 | ✓ | gan | candidateC_expanded672 |
| 48 | 22.8325 | 13.1070 | 16.9197 | 7.0815 | 6.6830 | 6.9371 | ✓ | gan | gan |
| 62 | 27.4635 | 17.9937 | 24.4485 | 6.7203 | 6.6502 | 7.0212 | ✓ | gan | gan |
| 63 | 28.5246 | 18.1998 | 25.4458 | 7.6606 | 7.5776 | 6.8043 | ✓ | gan | candidateC_expanded672 |
| 65 | 22.1489 | 16.2395 | 19.4622 | 5.8208 | 5.5768 | 5.3767 | ✓ | gan | candidateC_expanded672 |
| 68 | 21.7713 | 15.5009 | 17.6924 | 6.0743 | 5.6709 | 5.0383 | ✓ | gan | candidateC_expanded672 |
| 77 | 29.9447 | 20.3533 | 26.7595 | 6.1320 | 6.0000 | 6.6805 | ✓ | gan | gan |
| 79 | 30.1481 | 20.3212 | 27.3118 | 6.8565 | 6.4670 | 6.6035 | ✓ | gan | gan |
| 80 | 30.0863 | 20.7449 | 26.7103 | 6.9827 | 6.1767 | 6.0845 | ✓ | gan | candidateC_expanded672 |
| 82 | 28.9009 | 20.8163 | 26.2397 | 7.1841 | 6.3036 | 6.2192 | ✓ | gan | candidateC_expanded672 |
| 90 | 22.2068 | 17.3268 | 19.0367 | 5.2579 | 5.6635 | 5.3563 |  | gan | cnn |
| 91 | 22.3094 | 16.2661 | 18.9232 | 5.5083 | 5.5198 | 6.0844 |  | gan | cnn |
| 92 | 22.0793 | 16.8077 | 19.3578 | 5.7358 | 5.6779 | 5.7236 | ✓ | gan | gan |
| 93 | 24.4029 | 18.1355 | 21.8943 | 6.5070 | 6.8160 | 6.0376 |  | gan | candidateC_expanded672 |
| 154 | 25.3102 | 16.8345 | 20.3179 | 5.8898 | 5.7368 | 6.2828 | ✓ | gan | gan |
| 162 | 18.2711 | 19.4579 | 17.9713 | 4.5406 | 9.6957 | 4.7727 |  | candidateC_expanded672 | cnn |
| 163 | 19.3704 | 20.4464 | 19.0889 | 5.6989 | 6.7890 | 5.5793 |  | candidateC_expanded672 | candidateC_expanded672 |

## Output files

- `ttk_runs_fixed/topology_finetuning/candidateC_expanded672_topology/candidateC_expanded672_pd_mt_distances.csv` — candidateC_expanded672 PD/MT per sample
- `ttk_runs_fixed/topology_finetuning/candidateC_expanded672_topology/candidateC_expanded672_topology_comparison.csv` — three-way comparison (CNN, GAN, candidateC_expanded672)
- `docs/topology_finetuning_candidateC_expanded672_topology_eval.md` — this report

## Notes

- PD distance = bottleneck distance between persistence diagrams (computed by `ttkBottleneckDistance`).
- MT distance = Wasserstein-type merge tree distance (computed by `ttkMergeTreeDistanceMatrix`).
- Both use 160×160 speed patches at (x0=0, y0=0), consistent with the CNN/GAN baseline.
- **Lower distance = SR is topologically closer to GT. Lower is better for both metrics.**
- **Component-count proxy metrics** (counts of connected components in speed superlevel sets) are reported separately in `topology_finetuning_candidateC_expanded672_eval.md` and do not require TTK.
- candidateC_expanded672 was fine-tuned from the pretrained CNN checkpoint. Results may change with different lambda settings or more epochs.
- Existing CNN/GAN and Candidate B topology outputs were not overwritten.
