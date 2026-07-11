# candidateUV_plus_crit_expanded2688 topology evaluation

**Generated:** 2026-07-11

**Important:** This report covers **true topology metrics** (PD bottleneck distance, MT Wasserstein distance) computed by TTK. These are distinct from the component-count *proxy* metrics in `topology_finetuning_candidateUV_plus_crit_expanded2688_eval.md`.

**Note:** Lower PD/MT distance = SR is topologically closer to GT. **Lower is better for both metrics.**

**candidateUV_plus_crit_expanded2688:** fine-tuned from the pretrained CNN checkpoint with auxiliary physics/topology losses. See the corresponding fine-tuning pilot script for configuration details.

**Samples evaluated:** 168

**MT-GAN baseline wins (before candidateUV_plus_crit_expanded2688):** 20 samples — [6, 8, 12, 16, 17, 18, 19, 20, 25, 48, 62, 63, 65, 68, 77, 79, 80, 82, 92, 154]

## Summary statistics

| Metric | CNN (baseline) | GAN (baseline) | candidateUV_plus_crit_expanded2688 |
|---|---|---|---|
| PD distance (mean) | 27.4063 | 20.8641 | 29.1143 |
| MT distance (mean) | 5.8678 | 8.3481 | 5.6899 |
| PD wins (vs CNN) | — | 166 | 12 |
| MT wins (vs CNN) | — | 20 | 94 |
| PD beats GAN | — | — | 0 |
| MT beats GAN | — | — | 160 |

## 6 Key evaluation questions

### Q1. Does candidateUV_plus_crit_expanded2688 improve PD distance relative to baseline CNN?

- Mean PD: CNN=27.4063, candidateUV_plus_crit_expanded2688=29.1143, Δ=1.7081 (▲ worse)
- candidateUV_plus_crit_expanded2688 has lower PD on **12/168** samples.

### Q2. Does candidateUV_plus_crit_expanded2688 ever beat GAN on PD distance?

- candidateUV_plus_crit_expanded2688 beats GAN on PD for **0/168** samples.
- Mean PD: GAN=20.8641, candidateUV_plus_crit_expanded2688=29.1143. Δ=8.2502 (▲ worse than GAN on average)

### Q3. Does candidateUV_plus_crit_expanded2688 improve MT distance relative to baseline CNN?

- Mean MT: CNN=5.8678, candidateUV_plus_crit_expanded2688=5.6899, Δ=-0.1779 (▼ better)
- candidateUV_plus_crit_expanded2688 has lower MT on **94/168** samples.

### Q4. Does candidateUV_plus_crit_expanded2688 change any of the original 20 MT-GAN cases?

The 20 samples where GAN originally beat CNN on MT distance: [6, 8, 12, 16, 17, 18, 19, 20, 25, 48, 62, 63, 65, 68, 77, 79, 80, 82, 92, 154]

| Outcome | Count |
|---|---|
| MT winner stays GAN | 7 |
| MT winner changes to candidateUV_plus_crit_expanded2688 | 13 |
| MT winner changes to CNN | 0 |

Per-sample detail for changed cases:

| sample_idx | MT CNN | MT GAN | MT candidateUV_plus_crit_expanded2688 | winner before | winner after |
|---|---|---|---|---|---|
| 6 | 6.1440 | 6.0106 | 5.3980 | gan | candidateUV_plus_crit_expanded2688 |
| 8 | 7.5096 | 6.0550 | 5.9562 | gan | candidateUV_plus_crit_expanded2688 |
| 18 | 6.2586 | 5.5544 | 5.2499 | gan | candidateUV_plus_crit_expanded2688 |
| 19 | 6.9280 | 5.4429 | 5.4157 | gan | candidateUV_plus_crit_expanded2688 |
| 25 | 11.3716 | 10.5015 | 6.2469 | gan | candidateUV_plus_crit_expanded2688 |
| 63 | 7.6606 | 7.5776 | 7.2329 | gan | candidateUV_plus_crit_expanded2688 |
| 65 | 5.8208 | 5.5768 | 5.4403 | gan | candidateUV_plus_crit_expanded2688 |
| 68 | 6.0743 | 5.6709 | 5.3599 | gan | candidateUV_plus_crit_expanded2688 |
| 77 | 6.1320 | 6.0000 | 5.5216 | gan | candidateUV_plus_crit_expanded2688 |
| 79 | 6.8565 | 6.4670 | 6.1100 | gan | candidateUV_plus_crit_expanded2688 |
| 80 | 6.9827 | 6.1767 | 5.8517 | gan | candidateUV_plus_crit_expanded2688 |
| 82 | 7.1841 | 6.3036 | 5.8293 | gan | candidateUV_plus_crit_expanded2688 |
| 92 | 5.7358 | 5.6779 | 5.2092 | gan | candidateUV_plus_crit_expanded2688 |

### Q5. What happens in adjacent clusters (samples 10–13 and 90–93)?

| sample_idx | PD CNN | PD GAN | PD candidateUV_plus_crit_expanded2688 | MT CNN | MT GAN | MT candidateUV_plus_crit_expanded2688 | PD winner → | MT winner → |
|---|---|---|---|---|---|---|---|---|
| 10 | 29.6970 | 24.7100 | 30.5115 | 6.6345 | 7.4238 | 5.7085 | gan → gan | cnn → candidateUV_plus_crit_expanded2688 |
| 11 | 28.6873 | 24.9699 | 30.2572 | 5.9107 | 7.8317 | 6.1841 | gan → gan | cnn → cnn |
| 12 | 30.1045 | 24.2206 | 30.6175 | 6.6164 | 5.9137 | 6.2554 | gan → gan | gan → gan |
| 13 | 29.2737 | 24.3237 | 29.8255 | 6.0738 | 7.4023 | 6.2396 | gan → gan | cnn → cnn |
| 90 | 22.2068 | 17.3268 | 23.7019 | 5.2579 | 5.6635 | 4.9962 | gan → gan | cnn → candidateUV_plus_crit_expanded2688 |
| 91 | 22.3094 | 16.2661 | 24.0258 | 5.5083 | 5.5198 | 5.4201 | gan → gan | cnn → candidateUV_plus_crit_expanded2688 |
| 92 | 22.0793 | 16.8077 | 24.6703 | 5.7358 | 5.6779 | 5.2092 | gan → gan | gan → candidateUV_plus_crit_expanded2688 |
| 93 | 24.4029 | 18.1355 | 28.0981 | 6.5070 | 6.8160 | 5.8195 | gan → gan | cnn → candidateUV_plus_crit_expanded2688 |

Note: samples [12, 92] are in the MT-GAN-wins set (GAN beat CNN on MT in baseline).

### Q6. What happens in rare topology-CNN controls (samples 162 and 163)?

These samples were originally topology-CNN controls (CNN wins on topology metrics). candidateUV_plus_crit_expanded2688 should not reverse this.

| sample_idx | PD CNN | PD GAN | PD candidateUV_plus_crit_expanded2688 | MT CNN | MT GAN | MT candidateUV_plus_crit_expanded2688 | PD winner → | MT winner → |
|---|---|---|---|---|---|---|---|---|
| 162 | 18.2711 | 19.4579 | 22.8100 | 4.5406 | 9.6957 | 4.2876 | cnn → cnn | cnn → candidateUV_plus_crit_expanded2688 |
| 163 | 19.3704 | 20.4464 | 24.0081 | 5.6989 | 6.7890 | 4.7612 | cnn → cnn | cnn → candidateUV_plus_crit_expanded2688 |

## Winner distribution

### PD distance winners

| Method | Wins before candidateUV_plus_crit_expanded2688 | Wins after candidateUV_plus_crit_expanded2688 |
|---|---|---|
| cnn | 2 | 2 |
| gan | 166 | 166 |

### MT distance winners

| Method | Wins before candidateUV_plus_crit_expanded2688 | Wins after candidateUV_plus_crit_expanded2688 |
|---|---|---|
| candidateUV_plus_crit_expanded2688 | 0 | 90 |
| cnn | 148 | 71 |
| gan | 20 | 7 |

## Full per-sample detail: focus samples

Includes adjacent-cluster samples, rare controls, and the 20 MT-GAN baseline wins.

| sample | PD CNN | PD GAN | PD candidateUV_plus_crit_expanded2688 | MT CNN | MT GAN | MT candidateUV_plus_crit_expanded2688 | MT-GAN-win? | PD after | MT after |
|---|---|---|---|---|---|---|---|---|---|
| 6 | 28.0690 | 24.0451 | 31.0323 | 6.1440 | 6.0106 | 5.3980 | ✓ | gan | candidateUV_plus_crit_expanded2688 |
| 8 | 29.6351 | 23.2821 | 30.7377 | 7.5096 | 6.0550 | 5.9562 | ✓ | gan | candidateUV_plus_crit_expanded2688 |
| 10 | 29.6970 | 24.7100 | 30.5115 | 6.6345 | 7.4238 | 5.7085 |  | gan | candidateUV_plus_crit_expanded2688 |
| 11 | 28.6873 | 24.9699 | 30.2572 | 5.9107 | 7.8317 | 6.1841 |  | gan | cnn |
| 12 | 30.1045 | 24.2206 | 30.6175 | 6.6164 | 5.9137 | 6.2554 | ✓ | gan | gan |
| 13 | 29.2737 | 24.3237 | 29.8255 | 6.0738 | 7.4023 | 6.2396 |  | gan | cnn |
| 16 | 23.7560 | 17.3245 | 24.2730 | 5.4642 | 5.2552 | 5.7745 | ✓ | gan | gan |
| 17 | 20.9004 | 17.1181 | 21.9267 | 5.2897 | 4.8487 | 5.1215 | ✓ | gan | gan |
| 18 | 20.3949 | 18.6102 | 21.4555 | 6.2586 | 5.5544 | 5.2499 | ✓ | gan | candidateUV_plus_crit_expanded2688 |
| 19 | 21.0641 | 19.0867 | 21.0061 | 6.9280 | 5.4429 | 5.4157 | ✓ | gan | candidateUV_plus_crit_expanded2688 |
| 20 | 21.9792 | 18.2663 | 22.5877 | 6.2541 | 5.6891 | 5.8533 | ✓ | gan | gan |
| 25 | 36.5841 | 30.1235 | 38.1875 | 11.3716 | 10.5015 | 6.2469 | ✓ | gan | candidateUV_plus_crit_expanded2688 |
| 48 | 22.8325 | 13.1070 | 21.7839 | 7.0815 | 6.6830 | 6.9537 | ✓ | gan | gan |
| 62 | 27.4635 | 17.9937 | 28.6625 | 6.7203 | 6.6502 | 6.8004 | ✓ | gan | gan |
| 63 | 28.5246 | 18.1998 | 29.4840 | 7.6606 | 7.5776 | 7.2329 | ✓ | gan | candidateUV_plus_crit_expanded2688 |
| 65 | 22.1489 | 16.2395 | 23.0308 | 5.8208 | 5.5768 | 5.4403 | ✓ | gan | candidateUV_plus_crit_expanded2688 |
| 68 | 21.7713 | 15.5009 | 21.9389 | 6.0743 | 5.6709 | 5.3599 | ✓ | gan | candidateUV_plus_crit_expanded2688 |
| 77 | 29.9447 | 20.3533 | 31.2988 | 6.1320 | 6.0000 | 5.5216 | ✓ | gan | candidateUV_plus_crit_expanded2688 |
| 79 | 30.1481 | 20.3212 | 30.9462 | 6.8565 | 6.4670 | 6.1100 | ✓ | gan | candidateUV_plus_crit_expanded2688 |
| 80 | 30.0863 | 20.7449 | 31.1126 | 6.9827 | 6.1767 | 5.8517 | ✓ | gan | candidateUV_plus_crit_expanded2688 |
| 82 | 28.9009 | 20.8163 | 30.7364 | 7.1841 | 6.3036 | 5.8293 | ✓ | gan | candidateUV_plus_crit_expanded2688 |
| 90 | 22.2068 | 17.3268 | 23.7019 | 5.2579 | 5.6635 | 4.9962 |  | gan | candidateUV_plus_crit_expanded2688 |
| 91 | 22.3094 | 16.2661 | 24.0258 | 5.5083 | 5.5198 | 5.4201 |  | gan | candidateUV_plus_crit_expanded2688 |
| 92 | 22.0793 | 16.8077 | 24.6703 | 5.7358 | 5.6779 | 5.2092 | ✓ | gan | candidateUV_plus_crit_expanded2688 |
| 93 | 24.4029 | 18.1355 | 28.0981 | 6.5070 | 6.8160 | 5.8195 |  | gan | candidateUV_plus_crit_expanded2688 |
| 154 | 25.3102 | 16.8345 | 24.6471 | 5.8898 | 5.7368 | 6.6648 | ✓ | gan | gan |
| 162 | 18.2711 | 19.4579 | 22.8100 | 4.5406 | 9.6957 | 4.2876 |  | cnn | candidateUV_plus_crit_expanded2688 |
| 163 | 19.3704 | 20.4464 | 24.0081 | 5.6989 | 6.7890 | 4.7612 |  | cnn | candidateUV_plus_crit_expanded2688 |

## Output files

- `ttk_runs_fixed/topology_finetuning/candidateUV_plus_crit_expanded2688_topology/candidateUV_plus_crit_expanded2688_pd_mt_distances.csv` — candidateUV_plus_crit_expanded2688 PD/MT per sample
- `ttk_runs_fixed/topology_finetuning/candidateUV_plus_crit_expanded2688_topology/candidateUV_plus_crit_expanded2688_topology_comparison.csv` — three-way comparison (CNN, GAN, candidateUV_plus_crit_expanded2688)
- `docs/topology_finetuning_candidateUV_plus_crit_expanded2688_topology_eval.md` — this report

## Notes

- PD distance = bottleneck distance between persistence diagrams (computed by `ttkBottleneckDistance`).
- MT distance = Wasserstein-type merge tree distance (computed by `ttkMergeTreeDistanceMatrix`).
- Both use 160×160 speed patches at (x0=0, y0=0), consistent with the CNN/GAN baseline.
- **Lower distance = SR is topologically closer to GT. Lower is better for both metrics.**
- **Component-count proxy metrics** (counts of connected components in speed superlevel sets) are reported separately in `topology_finetuning_candidateUV_plus_crit_expanded2688_eval.md` and do not require TTK.
- candidateUV_plus_crit_expanded2688 was fine-tuned from the pretrained CNN checkpoint. Results may change with different lambda settings or more epochs.
- Existing CNN/GAN and Candidate B topology outputs were not overwritten.
