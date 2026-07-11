# candidateE2_tf_lowlambda_expanded2688 topology evaluation

**Generated:** 2026-07-10

**Important:** This report covers **true topology metrics** (PD bottleneck distance, MT Wasserstein distance) computed by TTK. These are distinct from the component-count *proxy* metrics in `topology_finetuning_candidateE2_tf_lowlambda_expanded2688_eval.md`.

**Note:** Lower PD/MT distance = SR is topologically closer to GT. **Lower is better for both metrics.**

**candidateE2_tf_lowlambda_expanded2688:** fine-tuned from the pretrained CNN checkpoint with auxiliary physics/topology losses. See the corresponding fine-tuning pilot script for configuration details.

**Samples evaluated:** 168

**MT-GAN baseline wins (before candidateE2_tf_lowlambda_expanded2688):** 20 samples — [6, 8, 12, 16, 17, 18, 19, 20, 25, 48, 62, 63, 65, 68, 77, 79, 80, 82, 92, 154]

## Summary statistics

| Metric | CNN (baseline) | GAN (baseline) | candidateE2_tf_lowlambda_expanded2688 |
|---|---|---|---|
| PD distance (mean) | 27.4063 | 20.8641 | 24.2686 |
| MT distance (mean) | 5.8678 | 8.3481 | 5.6628 |
| PD wins (vs CNN) | — | 166 | 165 |
| MT wins (vs CNN) | — | 20 | 94 |
| PD beats GAN | — | — | 7 |
| MT beats GAN | — | — | 167 |

## 6 Key evaluation questions

### Q1. Does candidateE2_tf_lowlambda_expanded2688 improve PD distance relative to baseline CNN?

- Mean PD: CNN=27.4063, candidateE2_tf_lowlambda_expanded2688=24.2686, Δ=-3.1377 (▼ better)
- candidateE2_tf_lowlambda_expanded2688 has lower PD on **165/168** samples.

### Q2. Does candidateE2_tf_lowlambda_expanded2688 ever beat GAN on PD distance?

- candidateE2_tf_lowlambda_expanded2688 beats GAN on PD for **7/168** samples.
- Mean PD: GAN=20.8641, candidateE2_tf_lowlambda_expanded2688=24.2686. Δ=3.4045 (▲ worse than GAN on average)

### Q3. Does candidateE2_tf_lowlambda_expanded2688 improve MT distance relative to baseline CNN?

- Mean MT: CNN=5.8678, candidateE2_tf_lowlambda_expanded2688=5.6628, Δ=-0.2050 (▼ better)
- candidateE2_tf_lowlambda_expanded2688 has lower MT on **94/168** samples.

### Q4. Does candidateE2_tf_lowlambda_expanded2688 change any of the original 20 MT-GAN cases?

The 20 samples where GAN originally beat CNN on MT distance: [6, 8, 12, 16, 17, 18, 19, 20, 25, 48, 62, 63, 65, 68, 77, 79, 80, 82, 92, 154]

| Outcome | Count |
|---|---|
| MT winner stays GAN | 1 |
| MT winner changes to candidateE2_tf_lowlambda_expanded2688 | 19 |
| MT winner changes to CNN | 0 |

Per-sample detail for changed cases:

| sample_idx | MT CNN | MT GAN | MT candidateE2_tf_lowlambda_expanded2688 | winner before | winner after |
|---|---|---|---|---|---|
| 6 | 6.1440 | 6.0106 | 5.5025 | gan | candidateE2_tf_lowlambda_expanded2688 |
| 8 | 7.5096 | 6.0550 | 5.9762 | gan | candidateE2_tf_lowlambda_expanded2688 |
| 16 | 5.4642 | 5.2552 | 4.5317 | gan | candidateE2_tf_lowlambda_expanded2688 |
| 17 | 5.2897 | 4.8487 | 4.1503 | gan | candidateE2_tf_lowlambda_expanded2688 |
| 18 | 6.2586 | 5.5544 | 4.8221 | gan | candidateE2_tf_lowlambda_expanded2688 |
| 19 | 6.9280 | 5.4429 | 4.9152 | gan | candidateE2_tf_lowlambda_expanded2688 |
| 20 | 6.2541 | 5.6891 | 5.3072 | gan | candidateE2_tf_lowlambda_expanded2688 |
| 25 | 11.3716 | 10.5015 | 7.1026 | gan | candidateE2_tf_lowlambda_expanded2688 |
| 48 | 7.0815 | 6.6830 | 6.2410 | gan | candidateE2_tf_lowlambda_expanded2688 |
| 62 | 6.7203 | 6.6502 | 6.2927 | gan | candidateE2_tf_lowlambda_expanded2688 |
| 63 | 7.6606 | 7.5776 | 5.9699 | gan | candidateE2_tf_lowlambda_expanded2688 |
| 65 | 5.8208 | 5.5768 | 5.1578 | gan | candidateE2_tf_lowlambda_expanded2688 |
| 68 | 6.0743 | 5.6709 | 5.4809 | gan | candidateE2_tf_lowlambda_expanded2688 |
| 77 | 6.1320 | 6.0000 | 4.7583 | gan | candidateE2_tf_lowlambda_expanded2688 |
| 79 | 6.8565 | 6.4670 | 5.5990 | gan | candidateE2_tf_lowlambda_expanded2688 |
| 80 | 6.9827 | 6.1767 | 4.9332 | gan | candidateE2_tf_lowlambda_expanded2688 |
| 82 | 7.1841 | 6.3036 | 5.6262 | gan | candidateE2_tf_lowlambda_expanded2688 |
| 92 | 5.7358 | 5.6779 | 5.0705 | gan | candidateE2_tf_lowlambda_expanded2688 |
| 154 | 5.8898 | 5.7368 | 5.6417 | gan | candidateE2_tf_lowlambda_expanded2688 |

### Q5. What happens in adjacent clusters (samples 10–13 and 90–93)?

| sample_idx | PD CNN | PD GAN | PD candidateE2_tf_lowlambda_expanded2688 | MT CNN | MT GAN | MT candidateE2_tf_lowlambda_expanded2688 | PD winner → | MT winner → |
|---|---|---|---|---|---|---|---|---|
| 10 | 29.6970 | 24.7100 | 25.7336 | 6.6345 | 7.4238 | 5.6949 | gan → gan | cnn → candidateE2_tf_lowlambda_expanded2688 |
| 11 | 28.6873 | 24.9699 | 24.6026 | 5.9107 | 7.8317 | 6.2256 | gan → candidateE2_tf_lowlambda_expanded2688 | cnn → cnn |
| 12 | 30.1045 | 24.2206 | 25.9471 | 6.6164 | 5.9137 | 6.1600 | gan → gan | gan → gan |
| 13 | 29.2737 | 24.3237 | 25.8795 | 6.0738 | 7.4023 | 6.4638 | gan → gan | cnn → cnn |
| 90 | 22.2068 | 17.3268 | 19.6522 | 5.2579 | 5.6635 | 4.8234 | gan → gan | cnn → candidateE2_tf_lowlambda_expanded2688 |
| 91 | 22.3094 | 16.2661 | 19.3167 | 5.5083 | 5.5198 | 5.0637 | gan → gan | cnn → candidateE2_tf_lowlambda_expanded2688 |
| 92 | 22.0793 | 16.8077 | 19.5366 | 5.7358 | 5.6779 | 5.0705 | gan → gan | gan → candidateE2_tf_lowlambda_expanded2688 |
| 93 | 24.4029 | 18.1355 | 21.8486 | 6.5070 | 6.8160 | 6.0232 | gan → gan | cnn → candidateE2_tf_lowlambda_expanded2688 |

Note: samples [12, 92] are in the MT-GAN-wins set (GAN beat CNN on MT in baseline).

### Q6. What happens in rare topology-CNN controls (samples 162 and 163)?

These samples were originally topology-CNN controls (CNN wins on topology metrics). candidateE2_tf_lowlambda_expanded2688 should not reverse this.

| sample_idx | PD CNN | PD GAN | PD candidateE2_tf_lowlambda_expanded2688 | MT CNN | MT GAN | MT candidateE2_tf_lowlambda_expanded2688 | PD winner → | MT winner → |
|---|---|---|---|---|---|---|---|---|
| 162 | 18.2711 | 19.4579 | 18.8290 | 4.5406 | 9.6957 | 5.1843 | cnn → cnn | cnn → cnn |
| 163 | 19.3704 | 20.4464 | 19.6862 | 5.6989 | 6.7890 | 4.7177 | cnn → cnn | cnn → candidateE2_tf_lowlambda_expanded2688 |

## Winner distribution

### PD distance winners

| Method | Wins before candidateE2_tf_lowlambda_expanded2688 | Wins after candidateE2_tf_lowlambda_expanded2688 |
|---|---|---|
| candidateE2_tf_lowlambda_expanded2688 | 0 | 5 |
| cnn | 2 | 2 |
| gan | 166 | 161 |

### MT distance winners

| Method | Wins before candidateE2_tf_lowlambda_expanded2688 | Wins after candidateE2_tf_lowlambda_expanded2688 |
|---|---|---|
| candidateE2_tf_lowlambda_expanded2688 | 0 | 93 |
| cnn | 148 | 74 |
| gan | 20 | 1 |

## Full per-sample detail: focus samples

Includes adjacent-cluster samples, rare controls, and the 20 MT-GAN baseline wins.

| sample | PD CNN | PD GAN | PD candidateE2_tf_lowlambda_expanded2688 | MT CNN | MT GAN | MT candidateE2_tf_lowlambda_expanded2688 | MT-GAN-win? | PD after | MT after |
|---|---|---|---|---|---|---|---|---|---|
| 6 | 28.0690 | 24.0451 | 26.0434 | 6.1440 | 6.0106 | 5.5025 | ✓ | gan | candidateE2_tf_lowlambda_expanded2688 |
| 8 | 29.6351 | 23.2821 | 26.6025 | 7.5096 | 6.0550 | 5.9762 | ✓ | gan | candidateE2_tf_lowlambda_expanded2688 |
| 10 | 29.6970 | 24.7100 | 25.7336 | 6.6345 | 7.4238 | 5.6949 |  | gan | candidateE2_tf_lowlambda_expanded2688 |
| 11 | 28.6873 | 24.9699 | 24.6026 | 5.9107 | 7.8317 | 6.2256 |  | candidateE2_tf_lowlambda_expanded2688 | cnn |
| 12 | 30.1045 | 24.2206 | 25.9471 | 6.6164 | 5.9137 | 6.1600 | ✓ | gan | gan |
| 13 | 29.2737 | 24.3237 | 25.8795 | 6.0738 | 7.4023 | 6.4638 |  | gan | cnn |
| 16 | 23.7560 | 17.3245 | 20.7221 | 5.4642 | 5.2552 | 4.5317 | ✓ | gan | candidateE2_tf_lowlambda_expanded2688 |
| 17 | 20.9004 | 17.1181 | 18.7097 | 5.2897 | 4.8487 | 4.1503 | ✓ | gan | candidateE2_tf_lowlambda_expanded2688 |
| 18 | 20.3949 | 18.6102 | 18.0835 | 6.2586 | 5.5544 | 4.8221 | ✓ | candidateE2_tf_lowlambda_expanded2688 | candidateE2_tf_lowlambda_expanded2688 |
| 19 | 21.0641 | 19.0867 | 17.7169 | 6.9280 | 5.4429 | 4.9152 | ✓ | candidateE2_tf_lowlambda_expanded2688 | candidateE2_tf_lowlambda_expanded2688 |
| 20 | 21.9792 | 18.2663 | 19.2712 | 6.2541 | 5.6891 | 5.3072 | ✓ | gan | candidateE2_tf_lowlambda_expanded2688 |
| 25 | 36.5841 | 30.1235 | 32.3192 | 11.3716 | 10.5015 | 7.1026 | ✓ | gan | candidateE2_tf_lowlambda_expanded2688 |
| 48 | 22.8325 | 13.1070 | 17.6528 | 7.0815 | 6.6830 | 6.2410 | ✓ | gan | candidateE2_tf_lowlambda_expanded2688 |
| 62 | 27.4635 | 17.9937 | 24.1590 | 6.7203 | 6.6502 | 6.2927 | ✓ | gan | candidateE2_tf_lowlambda_expanded2688 |
| 63 | 28.5246 | 18.1998 | 24.4240 | 7.6606 | 7.5776 | 5.9699 | ✓ | gan | candidateE2_tf_lowlambda_expanded2688 |
| 65 | 22.1489 | 16.2395 | 19.3172 | 5.8208 | 5.5768 | 5.1578 | ✓ | gan | candidateE2_tf_lowlambda_expanded2688 |
| 68 | 21.7713 | 15.5009 | 17.8936 | 6.0743 | 5.6709 | 5.4809 | ✓ | gan | candidateE2_tf_lowlambda_expanded2688 |
| 77 | 29.9447 | 20.3533 | 26.0926 | 6.1320 | 6.0000 | 4.7583 | ✓ | gan | candidateE2_tf_lowlambda_expanded2688 |
| 79 | 30.1481 | 20.3212 | 26.4520 | 6.8565 | 6.4670 | 5.5990 | ✓ | gan | candidateE2_tf_lowlambda_expanded2688 |
| 80 | 30.0863 | 20.7449 | 26.2299 | 6.9827 | 6.1767 | 4.9332 | ✓ | gan | candidateE2_tf_lowlambda_expanded2688 |
| 82 | 28.9009 | 20.8163 | 25.6669 | 7.1841 | 6.3036 | 5.6262 | ✓ | gan | candidateE2_tf_lowlambda_expanded2688 |
| 90 | 22.2068 | 17.3268 | 19.6522 | 5.2579 | 5.6635 | 4.8234 |  | gan | candidateE2_tf_lowlambda_expanded2688 |
| 91 | 22.3094 | 16.2661 | 19.3167 | 5.5083 | 5.5198 | 5.0637 |  | gan | candidateE2_tf_lowlambda_expanded2688 |
| 92 | 22.0793 | 16.8077 | 19.5366 | 5.7358 | 5.6779 | 5.0705 | ✓ | gan | candidateE2_tf_lowlambda_expanded2688 |
| 93 | 24.4029 | 18.1355 | 21.8486 | 6.5070 | 6.8160 | 6.0232 |  | gan | candidateE2_tf_lowlambda_expanded2688 |
| 154 | 25.3102 | 16.8345 | 20.3161 | 5.8898 | 5.7368 | 5.6417 | ✓ | gan | candidateE2_tf_lowlambda_expanded2688 |
| 162 | 18.2711 | 19.4579 | 18.8290 | 4.5406 | 9.6957 | 5.1843 |  | cnn | cnn |
| 163 | 19.3704 | 20.4464 | 19.6862 | 5.6989 | 6.7890 | 4.7177 |  | cnn | candidateE2_tf_lowlambda_expanded2688 |

## Output files

- `ttk_runs_fixed/topology_finetuning/candidateE2_tf_lowlambda_expanded2688_topology/candidateE2_tf_lowlambda_expanded2688_pd_mt_distances.csv` — candidateE2_tf_lowlambda_expanded2688 PD/MT per sample
- `ttk_runs_fixed/topology_finetuning/candidateE2_tf_lowlambda_expanded2688_topology/candidateE2_tf_lowlambda_expanded2688_topology_comparison.csv` — three-way comparison (CNN, GAN, candidateE2_tf_lowlambda_expanded2688)
- `docs/topology_finetuning_candidateE2_tf_lowlambda_expanded2688_topology_eval.md` — this report

## Notes

- PD distance = bottleneck distance between persistence diagrams (computed by `ttkBottleneckDistance`).
- MT distance = Wasserstein-type merge tree distance (computed by `ttkMergeTreeDistanceMatrix`).
- Both use 160×160 speed patches at (x0=0, y0=0), consistent with the CNN/GAN baseline.
- **Lower distance = SR is topologically closer to GT. Lower is better for both metrics.**
- **Component-count proxy metrics** (counts of connected components in speed superlevel sets) are reported separately in `topology_finetuning_candidateE2_tf_lowlambda_expanded2688_eval.md` and do not require TTK.
- candidateE2_tf_lowlambda_expanded2688 was fine-tuned from the pretrained CNN checkpoint. Results may change with different lambda settings or more epochs.
- Existing CNN/GAN and Candidate B topology outputs were not overwritten.
