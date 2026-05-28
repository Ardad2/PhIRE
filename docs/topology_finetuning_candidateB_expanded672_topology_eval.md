# candidateB_expanded672 topology evaluation

**Generated:** 2026-05-26

**Important:** This report covers **true topology metrics** (PD bottleneck distance, MT Wasserstein distance) computed by TTK. These are distinct from the component-count *proxy* metrics in `topology_finetuning_candidateB_expanded672_eval.md`.

**Note:** Lower PD/MT distance = SR is topologically closer to GT. **Lower is better for both metrics.**

**candidateB_expanded672:** fine-tuned from the pretrained CNN checkpoint with auxiliary physics/topology losses. See the corresponding fine-tuning pilot script for configuration details.

**Samples evaluated:** 168

**MT-GAN baseline wins (before candidateB_expanded672):** 20 samples — [6, 8, 12, 16, 17, 18, 19, 20, 25, 48, 62, 63, 65, 68, 77, 79, 80, 82, 92, 154]

## Summary statistics

| Metric | CNN (baseline) | GAN (baseline) | candidateB_expanded672 |
|---|---|---|---|
| PD distance (mean) | 27.4063 | 20.8641 | 23.7094 |
| MT distance (mean) | 5.8678 | 8.3481 | 6.3124 |
| PD wins (vs CNN) | — | 166 | 167 |
| MT wins (vs CNN) | — | 20 | 36 |
| PD beats GAN | — | — | 9 |
| MT beats GAN | — | — | 137 |

## 6 Key evaluation questions

### Q1. Does candidateB_expanded672 improve PD distance relative to baseline CNN?

- Mean PD: CNN=27.4063, candidateB_expanded672=23.7094, Δ=-3.6969 (▼ better)
- candidateB_expanded672 has lower PD on **167/168** samples.

### Q2. Does candidateB_expanded672 ever beat GAN on PD distance?

- candidateB_expanded672 beats GAN on PD for **9/168** samples.
- Mean PD: GAN=20.8641, candidateB_expanded672=23.7094. Δ=2.8453 (▲ worse than GAN on average)

### Q3. Does candidateB_expanded672 improve MT distance relative to baseline CNN?

- Mean MT: CNN=5.8678, candidateB_expanded672=6.3124, Δ=0.4446 (▲ worse)
- candidateB_expanded672 has lower MT on **36/168** samples.

### Q4. Does candidateB_expanded672 change any of the original 20 MT-GAN cases?

The 20 samples where GAN originally beat CNN on MT distance: [6, 8, 12, 16, 17, 18, 19, 20, 25, 48, 62, 63, 65, 68, 77, 79, 80, 82, 92, 154]

| Outcome | Count |
|---|---|
| MT winner stays GAN | 15 |
| MT winner changes to candidateB_expanded672 | 5 |
| MT winner changes to CNN | 0 |

Per-sample detail for changed cases:

| sample_idx | MT CNN | MT GAN | MT candidateB_expanded672 | winner before | winner after |
|---|---|---|---|---|---|
| 6 | 6.1440 | 6.0106 | 5.9685 | gan | candidateB_expanded672 |
| 25 | 11.3716 | 10.5015 | 6.8100 | gan | candidateB_expanded672 |
| 63 | 7.6606 | 7.5776 | 6.7730 | gan | candidateB_expanded672 |
| 68 | 6.0743 | 5.6709 | 5.2605 | gan | candidateB_expanded672 |
| 80 | 6.9827 | 6.1767 | 5.7764 | gan | candidateB_expanded672 |

### Q5. What happens in adjacent clusters (samples 10–13 and 90–93)?

| sample_idx | PD CNN | PD GAN | PD candidateB_expanded672 | MT CNN | MT GAN | MT candidateB_expanded672 | PD winner → | MT winner → |
|---|---|---|---|---|---|---|---|---|
| 10 | 29.6970 | 24.7100 | 25.8487 | 6.6345 | 7.4238 | 6.4093 | gan → gan | cnn → candidateB_expanded672 |
| 11 | 28.6873 | 24.9699 | 25.4716 | 5.9107 | 7.8317 | 6.6574 | gan → gan | cnn → cnn |
| 12 | 30.1045 | 24.2206 | 26.0610 | 6.6164 | 5.9137 | 6.7147 | gan → gan | gan → gan |
| 13 | 29.2737 | 24.3237 | 25.5291 | 6.0738 | 7.4023 | 6.8773 | gan → gan | cnn → cnn |
| 90 | 22.2068 | 17.3268 | 18.5151 | 5.2579 | 5.6635 | 5.6765 | gan → gan | cnn → cnn |
| 91 | 22.3094 | 16.2661 | 18.5627 | 5.5083 | 5.5198 | 6.5642 | gan → gan | cnn → cnn |
| 92 | 22.0793 | 16.8077 | 19.1999 | 5.7358 | 5.6779 | 6.0819 | gan → gan | gan → gan |
| 93 | 24.4029 | 18.1355 | 22.7818 | 6.5070 | 6.8160 | 6.9726 | gan → gan | cnn → cnn |

Note: samples [12, 92] are in the MT-GAN-wins set (GAN beat CNN on MT in baseline).

### Q6. What happens in rare topology-CNN controls (samples 162 and 163)?

These samples were originally topology-CNN controls (CNN wins on topology metrics). candidateB_expanded672 should not reverse this.

| sample_idx | PD CNN | PD GAN | PD candidateB_expanded672 | MT CNN | MT GAN | MT candidateB_expanded672 | PD winner → | MT winner → |
|---|---|---|---|---|---|---|---|---|
| 162 | 18.2711 | 19.4579 | 18.1880 | 4.5406 | 9.6957 | 5.2756 | cnn → candidateB_expanded672 | cnn → cnn |
| 163 | 19.3704 | 20.4464 | 19.5110 | 5.6989 | 6.7890 | 5.5239 | cnn → cnn | cnn → candidateB_expanded672 |

## Winner distribution

### PD distance winners

| Method | Wins before candidateB_expanded672 | Wins after candidateB_expanded672 |
|---|---|---|
| candidateB_expanded672 | 0 | 8 |
| cnn | 2 | 1 |
| gan | 166 | 159 |

### MT distance winners

| Method | Wins before candidateB_expanded672 | Wins after candidateB_expanded672 |
|---|---|---|
| candidateB_expanded672 | 0 | 31 |
| cnn | 148 | 122 |
| gan | 20 | 15 |

## Full per-sample detail: focus samples

Includes adjacent-cluster samples, rare controls, and the 20 MT-GAN baseline wins.

| sample | PD CNN | PD GAN | PD candidateB_expanded672 | MT CNN | MT GAN | MT candidateB_expanded672 | MT-GAN-win? | PD after | MT after |
|---|---|---|---|---|---|---|---|---|---|
| 6 | 28.0690 | 24.0451 | 23.8400 | 6.1440 | 6.0106 | 5.9685 | ✓ | candidateB_expanded672 | candidateB_expanded672 |
| 8 | 29.6351 | 23.2821 | 25.5819 | 7.5096 | 6.0550 | 6.5588 | ✓ | gan | gan |
| 10 | 29.6970 | 24.7100 | 25.8487 | 6.6345 | 7.4238 | 6.4093 |  | gan | candidateB_expanded672 |
| 11 | 28.6873 | 24.9699 | 25.4716 | 5.9107 | 7.8317 | 6.6574 |  | gan | cnn |
| 12 | 30.1045 | 24.2206 | 26.0610 | 6.6164 | 5.9137 | 6.7147 | ✓ | gan | gan |
| 13 | 29.2737 | 24.3237 | 25.5291 | 6.0738 | 7.4023 | 6.8773 |  | gan | cnn |
| 16 | 23.7560 | 17.3245 | 21.6984 | 5.4642 | 5.2552 | 6.1908 | ✓ | gan | gan |
| 17 | 20.9004 | 17.1181 | 19.0731 | 5.2897 | 4.8487 | 5.3873 | ✓ | gan | gan |
| 18 | 20.3949 | 18.6102 | 18.1493 | 6.2586 | 5.5544 | 5.5702 | ✓ | candidateB_expanded672 | gan |
| 19 | 21.0641 | 19.0867 | 17.9025 | 6.9280 | 5.4429 | 6.3483 | ✓ | candidateB_expanded672 | gan |
| 20 | 21.9792 | 18.2663 | 18.7547 | 6.2541 | 5.6891 | 6.7932 | ✓ | gan | gan |
| 25 | 36.5841 | 30.1235 | 31.7353 | 11.3716 | 10.5015 | 6.8100 | ✓ | gan | candidateB_expanded672 |
| 48 | 22.8325 | 13.1070 | 17.0084 | 7.0815 | 6.6830 | 6.9792 | ✓ | gan | gan |
| 62 | 27.4635 | 17.9937 | 23.9404 | 6.7203 | 6.6502 | 6.9897 | ✓ | gan | gan |
| 63 | 28.5246 | 18.1998 | 24.3351 | 7.6606 | 7.5776 | 6.7730 | ✓ | gan | candidateB_expanded672 |
| 65 | 22.1489 | 16.2395 | 18.9727 | 5.8208 | 5.5768 | 5.8710 | ✓ | gan | gan |
| 68 | 21.7713 | 15.5009 | 17.1523 | 6.0743 | 5.6709 | 5.2605 | ✓ | gan | candidateB_expanded672 |
| 77 | 29.9447 | 20.3533 | 25.8321 | 6.1320 | 6.0000 | 6.6268 | ✓ | gan | gan |
| 79 | 30.1481 | 20.3212 | 26.8852 | 6.8565 | 6.4670 | 6.8821 | ✓ | gan | gan |
| 80 | 30.0863 | 20.7449 | 26.3771 | 6.9827 | 6.1767 | 5.7764 | ✓ | gan | candidateB_expanded672 |
| 82 | 28.9009 | 20.8163 | 25.9113 | 7.1841 | 6.3036 | 6.3659 | ✓ | gan | gan |
| 90 | 22.2068 | 17.3268 | 18.5151 | 5.2579 | 5.6635 | 5.6765 |  | gan | cnn |
| 91 | 22.3094 | 16.2661 | 18.5627 | 5.5083 | 5.5198 | 6.5642 |  | gan | cnn |
| 92 | 22.0793 | 16.8077 | 19.1999 | 5.7358 | 5.6779 | 6.0819 | ✓ | gan | gan |
| 93 | 24.4029 | 18.1355 | 22.7818 | 6.5070 | 6.8160 | 6.9726 |  | gan | cnn |
| 154 | 25.3102 | 16.8345 | 19.2876 | 5.8898 | 5.7368 | 6.2056 | ✓ | gan | gan |
| 162 | 18.2711 | 19.4579 | 18.1880 | 4.5406 | 9.6957 | 5.2756 |  | candidateB_expanded672 | cnn |
| 163 | 19.3704 | 20.4464 | 19.5110 | 5.6989 | 6.7890 | 5.5239 |  | cnn | candidateB_expanded672 |

## Output files

- `ttk_runs_fixed/topology_finetuning/candidateB_expanded672_topology/candidateB_expanded672_pd_mt_distances.csv` — candidateB_expanded672 PD/MT per sample
- `ttk_runs_fixed/topology_finetuning/candidateB_expanded672_topology/candidateB_expanded672_topology_comparison.csv` — three-way comparison (CNN, GAN, candidateB_expanded672)
- `docs/topology_finetuning_candidateB_expanded672_topology_eval.md` — this report

## Notes

- PD distance = bottleneck distance between persistence diagrams (computed by `ttkBottleneckDistance`).
- MT distance = Wasserstein-type merge tree distance (computed by `ttkMergeTreeDistanceMatrix`).
- Both use 160×160 speed patches at (x0=0, y0=0), consistent with the CNN/GAN baseline.
- **Lower distance = SR is topologically closer to GT. Lower is better for both metrics.**
- **Component-count proxy metrics** (counts of connected components in speed superlevel sets) are reported separately in `topology_finetuning_candidateB_expanded672_eval.md` and do not require TTK.
- candidateB_expanded672 was fine-tuned from the pretrained CNN checkpoint. Results may change with different lambda settings or more epochs.
- Existing CNN/GAN and Candidate B topology outputs were not overwritten.
