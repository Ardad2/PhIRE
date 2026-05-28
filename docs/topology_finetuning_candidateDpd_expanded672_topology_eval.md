# candidateDpd_expanded672 topology evaluation

**Generated:** 2026-05-27

**Important:** This report covers **true topology metrics** (PD bottleneck distance, MT Wasserstein distance) computed by TTK. These are distinct from the component-count *proxy* metrics in `topology_finetuning_candidateDpd_expanded672_eval.md`.

**Note:** Lower PD/MT distance = SR is topologically closer to GT. **Lower is better for both metrics.**

**candidateDpd_expanded672:** fine-tuned from the pretrained CNN checkpoint with auxiliary physics/topology losses. See the corresponding fine-tuning pilot script for configuration details.

**Samples evaluated:** 168

**MT-GAN baseline wins (before candidateDpd_expanded672):** 20 samples — [6, 8, 12, 16, 17, 18, 19, 20, 25, 48, 62, 63, 65, 68, 77, 79, 80, 82, 92, 154]

## Summary statistics

| Metric | CNN (baseline) | GAN (baseline) | candidateDpd_expanded672 |
|---|---|---|---|
| PD distance (mean) | 27.4063 | 20.8641 | 28.7656 |
| MT distance (mean) | 5.8678 | 8.3481 | 6.2210 |
| PD wins (vs CNN) | — | 166 | 0 |
| MT wins (vs CNN) | — | 20 | 23 |
| PD beats GAN | — | — | 0 |
| MT beats GAN | — | — | 138 |

## 6 Key evaluation questions

### Q1. Does candidateDpd_expanded672 improve PD distance relative to baseline CNN?

- Mean PD: CNN=27.4063, candidateDpd_expanded672=28.7656, Δ=1.3594 (▲ worse)
- candidateDpd_expanded672 has lower PD on **0/168** samples.

### Q2. Does candidateDpd_expanded672 ever beat GAN on PD distance?

- candidateDpd_expanded672 beats GAN on PD for **0/168** samples.
- Mean PD: GAN=20.8641, candidateDpd_expanded672=28.7656. Δ=7.9015 (▲ worse than GAN on average)

### Q3. Does candidateDpd_expanded672 improve MT distance relative to baseline CNN?

- Mean MT: CNN=5.8678, candidateDpd_expanded672=6.2210, Δ=0.3532 (▲ worse)
- candidateDpd_expanded672 has lower MT on **23/168** samples.

### Q4. Does candidateDpd_expanded672 change any of the original 20 MT-GAN cases?

The 20 samples where GAN originally beat CNN on MT distance: [6, 8, 12, 16, 17, 18, 19, 20, 25, 48, 62, 63, 65, 68, 77, 79, 80, 82, 92, 154]

| Outcome | Count |
|---|---|
| MT winner stays GAN | 19 |
| MT winner changes to candidateDpd_expanded672 | 1 |
| MT winner changes to CNN | 0 |

Per-sample detail for changed cases:

| sample_idx | MT CNN | MT GAN | MT candidateDpd_expanded672 | winner before | winner after |
|---|---|---|---|---|---|
| 25 | 11.3716 | 10.5015 | 6.6562 | gan | candidateDpd_expanded672 |

### Q5. What happens in adjacent clusters (samples 10–13 and 90–93)?

| sample_idx | PD CNN | PD GAN | PD candidateDpd_expanded672 | MT CNN | MT GAN | MT candidateDpd_expanded672 | PD winner → | MT winner → |
|---|---|---|---|---|---|---|---|---|
| 10 | 29.6970 | 24.7100 | 30.8589 | 6.6345 | 7.4238 | 7.0526 | gan → gan | cnn → cnn |
| 11 | 28.6873 | 24.9699 | 30.1006 | 5.9107 | 7.8317 | 6.4341 | gan → gan | cnn → cnn |
| 12 | 30.1045 | 24.2206 | 31.4899 | 6.6164 | 5.9137 | 7.0096 | gan → gan | gan → gan |
| 13 | 29.2737 | 24.3237 | 30.4886 | 6.0738 | 7.4023 | 6.6798 | gan → gan | cnn → cnn |
| 90 | 22.2068 | 17.3268 | 22.9358 | 5.2579 | 5.6635 | 5.4466 | gan → gan | cnn → cnn |
| 91 | 22.3094 | 16.2661 | 23.1632 | 5.5083 | 5.5198 | 5.8265 | gan → gan | cnn → cnn |
| 92 | 22.0793 | 16.8077 | 23.0035 | 5.7358 | 5.6779 | 5.6887 | gan → gan | gan → gan |
| 93 | 24.4029 | 18.1355 | 25.8074 | 6.5070 | 6.8160 | 6.8781 | gan → gan | cnn → cnn |

Note: samples [12, 92] are in the MT-GAN-wins set (GAN beat CNN on MT in baseline).

### Q6. What happens in rare topology-CNN controls (samples 162 and 163)?

These samples were originally topology-CNN controls (CNN wins on topology metrics). candidateDpd_expanded672 should not reverse this.

| sample_idx | PD CNN | PD GAN | PD candidateDpd_expanded672 | MT CNN | MT GAN | MT candidateDpd_expanded672 | PD winner → | MT winner → |
|---|---|---|---|---|---|---|---|---|
| 162 | 18.2711 | 19.4579 | 20.1609 | 4.5406 | 9.6957 | 4.4906 | cnn → cnn | cnn → candidateDpd_expanded672 |
| 163 | 19.3704 | 20.4464 | 21.2794 | 5.6989 | 6.7890 | 4.7374 | cnn → cnn | cnn → candidateDpd_expanded672 |

## Winner distribution

### PD distance winners

| Method | Wins before candidateDpd_expanded672 | Wins after candidateDpd_expanded672 |
|---|---|---|
| cnn | 2 | 2 |
| gan | 166 | 166 |

### MT distance winners

| Method | Wins before candidateDpd_expanded672 | Wins after candidateDpd_expanded672 |
|---|---|---|
| candidateDpd_expanded672 | 0 | 21 |
| cnn | 148 | 128 |
| gan | 20 | 19 |

## Full per-sample detail: focus samples

Includes adjacent-cluster samples, rare controls, and the 20 MT-GAN baseline wins.

| sample | PD CNN | PD GAN | PD candidateDpd_expanded672 | MT CNN | MT GAN | MT candidateDpd_expanded672 | MT-GAN-win? | PD after | MT after |
|---|---|---|---|---|---|---|---|---|---|
| 6 | 28.0690 | 24.0451 | 29.8821 | 6.1440 | 6.0106 | 7.6568 | ✓ | gan | gan |
| 8 | 29.6351 | 23.2821 | 30.8505 | 7.5096 | 6.0550 | 7.9138 | ✓ | gan | gan |
| 10 | 29.6970 | 24.7100 | 30.8589 | 6.6345 | 7.4238 | 7.0526 |  | gan | cnn |
| 11 | 28.6873 | 24.9699 | 30.1006 | 5.9107 | 7.8317 | 6.4341 |  | gan | cnn |
| 12 | 30.1045 | 24.2206 | 31.4899 | 6.6164 | 5.9137 | 7.0096 | ✓ | gan | gan |
| 13 | 29.2737 | 24.3237 | 30.4886 | 6.0738 | 7.4023 | 6.6798 |  | gan | cnn |
| 16 | 23.7560 | 17.3245 | 24.6246 | 5.4642 | 5.2552 | 6.0811 | ✓ | gan | gan |
| 17 | 20.9004 | 17.1181 | 21.8228 | 5.2897 | 4.8487 | 5.4600 | ✓ | gan | gan |
| 18 | 20.3949 | 18.6102 | 21.5724 | 6.2586 | 5.5544 | 6.3945 | ✓ | gan | gan |
| 19 | 21.0641 | 19.0867 | 22.0097 | 6.9280 | 5.4429 | 7.2624 | ✓ | gan | gan |
| 20 | 21.9792 | 18.2663 | 23.1775 | 6.2541 | 5.6891 | 7.0473 | ✓ | gan | gan |
| 25 | 36.5841 | 30.1235 | 38.9881 | 11.3716 | 10.5015 | 6.6562 | ✓ | gan | candidateDpd_expanded672 |
| 48 | 22.8325 | 13.1070 | 23.3052 | 7.0815 | 6.6830 | 7.0499 | ✓ | gan | gan |
| 62 | 27.4635 | 17.9937 | 28.0604 | 6.7203 | 6.6502 | 7.4137 | ✓ | gan | gan |
| 63 | 28.5246 | 18.1998 | 29.0243 | 7.6606 | 7.5776 | 7.8437 | ✓ | gan | gan |
| 65 | 22.1489 | 16.2395 | 22.5821 | 5.8208 | 5.5768 | 5.9003 | ✓ | gan | gan |
| 68 | 21.7713 | 15.5009 | 22.2335 | 6.0743 | 5.6709 | 6.4388 | ✓ | gan | gan |
| 77 | 29.9447 | 20.3533 | 30.4445 | 6.1320 | 6.0000 | 6.4068 | ✓ | gan | gan |
| 79 | 30.1481 | 20.3212 | 30.7794 | 6.8565 | 6.4670 | 7.3073 | ✓ | gan | gan |
| 80 | 30.0863 | 20.7449 | 30.6178 | 6.9827 | 6.1767 | 7.0604 | ✓ | gan | gan |
| 82 | 28.9009 | 20.8163 | 29.7252 | 7.1841 | 6.3036 | 7.8478 | ✓ | gan | gan |
| 90 | 22.2068 | 17.3268 | 22.9358 | 5.2579 | 5.6635 | 5.4466 |  | gan | cnn |
| 91 | 22.3094 | 16.2661 | 23.1632 | 5.5083 | 5.5198 | 5.8265 |  | gan | cnn |
| 92 | 22.0793 | 16.8077 | 23.0035 | 5.7358 | 5.6779 | 5.6887 | ✓ | gan | gan |
| 93 | 24.4029 | 18.1355 | 25.8074 | 6.5070 | 6.8160 | 6.8781 |  | gan | cnn |
| 154 | 25.3102 | 16.8345 | 26.0382 | 5.8898 | 5.7368 | 6.2963 | ✓ | gan | gan |
| 162 | 18.2711 | 19.4579 | 20.1609 | 4.5406 | 9.6957 | 4.4906 |  | cnn | candidateDpd_expanded672 |
| 163 | 19.3704 | 20.4464 | 21.2794 | 5.6989 | 6.7890 | 4.7374 |  | cnn | candidateDpd_expanded672 |

## Output files

- `ttk_runs_fixed/topology_finetuning/candidateDpd_expanded672_topology/candidateDpd_expanded672_pd_mt_distances.csv` — candidateDpd_expanded672 PD/MT per sample
- `ttk_runs_fixed/topology_finetuning/candidateDpd_expanded672_topology/candidateDpd_expanded672_topology_comparison.csv` — three-way comparison (CNN, GAN, candidateDpd_expanded672)
- `docs/topology_finetuning_candidateDpd_expanded672_topology_eval.md` — this report

## Notes

- PD distance = bottleneck distance between persistence diagrams (computed by `ttkBottleneckDistance`).
- MT distance = Wasserstein-type merge tree distance (computed by `ttkMergeTreeDistanceMatrix`).
- Both use 160×160 speed patches at (x0=0, y0=0), consistent with the CNN/GAN baseline.
- **Lower distance = SR is topologically closer to GT. Lower is better for both metrics.**
- **Component-count proxy metrics** (counts of connected components in speed superlevel sets) are reported separately in `topology_finetuning_candidateDpd_expanded672_eval.md` and do not require TTK.
- candidateDpd_expanded672 was fine-tuned from the pretrained CNN checkpoint. Results may change with different lambda settings or more epochs.
- Existing CNN/GAN and Candidate B topology outputs were not overwritten.
