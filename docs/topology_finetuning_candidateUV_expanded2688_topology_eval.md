# candidateUV_expanded2688 topology evaluation

**Generated:** 2026-05-29

**Important:** This report covers **true topology metrics** (PD bottleneck distance, MT Wasserstein distance) computed by TTK. These are distinct from the component-count *proxy* metrics in `topology_finetuning_candidateUV_expanded2688_eval.md`.

**Note:** Lower PD/MT distance = SR is topologically closer to GT. **Lower is better for both metrics.**

**candidateUV_expanded2688:** fine-tuned from the pretrained CNN checkpoint with auxiliary physics/topology losses. See the corresponding fine-tuning pilot script for configuration details.

**Samples evaluated:** 168

**MT-GAN baseline wins (before candidateUV_expanded2688):** 20 samples — [6, 8, 12, 16, 17, 18, 19, 20, 25, 48, 62, 63, 65, 68, 77, 79, 80, 82, 92, 154]

## Summary statistics

| Metric | CNN (baseline) | GAN (baseline) | candidateUV_expanded2688 |
|---|---|---|---|
| PD distance (mean) | 27.4063 | 20.8641 | 29.6121 |
| MT distance (mean) | 5.8678 | 8.3481 | 6.0119 |
| PD wins (vs CNN) | — | 166 | 9 |
| MT wins (vs CNN) | — | 20 | 64 |
| PD beats GAN | — | — | 0 |
| MT beats GAN | — | — | 147 |

## 6 Key evaluation questions

### Q1. Does candidateUV_expanded2688 improve PD distance relative to baseline CNN?

- Mean PD: CNN=27.4063, candidateUV_expanded2688=29.6121, Δ=2.2058 (▲ worse)
- candidateUV_expanded2688 has lower PD on **9/168** samples.

### Q2. Does candidateUV_expanded2688 ever beat GAN on PD distance?

- candidateUV_expanded2688 beats GAN on PD for **0/168** samples.
- Mean PD: GAN=20.8641, candidateUV_expanded2688=29.6121. Δ=8.7480 (▲ worse than GAN on average)

### Q3. Does candidateUV_expanded2688 improve MT distance relative to baseline CNN?

- Mean MT: CNN=5.8678, candidateUV_expanded2688=6.0119, Δ=0.1441 (▲ worse)
- candidateUV_expanded2688 has lower MT on **64/168** samples.

### Q4. Does candidateUV_expanded2688 change any of the original 20 MT-GAN cases?

The 20 samples where GAN originally beat CNN on MT distance: [6, 8, 12, 16, 17, 18, 19, 20, 25, 48, 62, 63, 65, 68, 77, 79, 80, 82, 92, 154]

| Outcome | Count |
|---|---|
| MT winner stays GAN | 15 |
| MT winner changes to candidateUV_expanded2688 | 5 |
| MT winner changes to CNN | 0 |

Per-sample detail for changed cases:

| sample_idx | MT CNN | MT GAN | MT candidateUV_expanded2688 | winner before | winner after |
|---|---|---|---|---|---|
| 18 | 6.2586 | 5.5544 | 5.5428 | gan | candidateUV_expanded2688 |
| 25 | 11.3716 | 10.5015 | 7.0109 | gan | candidateUV_expanded2688 |
| 63 | 7.6606 | 7.5776 | 7.4979 | gan | candidateUV_expanded2688 |
| 68 | 6.0743 | 5.6709 | 5.5097 | gan | candidateUV_expanded2688 |
| 92 | 5.7358 | 5.6779 | 5.6358 | gan | candidateUV_expanded2688 |

### Q5. What happens in adjacent clusters (samples 10–13 and 90–93)?

| sample_idx | PD CNN | PD GAN | PD candidateUV_expanded2688 | MT CNN | MT GAN | MT candidateUV_expanded2688 | PD winner → | MT winner → |
|---|---|---|---|---|---|---|---|---|
| 10 | 29.6970 | 24.7100 | 30.8282 | 6.6345 | 7.4238 | 5.8184 | gan → gan | cnn → candidateUV_expanded2688 |
| 11 | 28.6873 | 24.9699 | 30.7860 | 5.9107 | 7.8317 | 6.0474 | gan → gan | cnn → cnn |
| 12 | 30.1045 | 24.2206 | 31.0968 | 6.6164 | 5.9137 | 6.4362 | gan → gan | gan → gan |
| 13 | 29.2737 | 24.3237 | 30.1385 | 6.0738 | 7.4023 | 6.5902 | gan → gan | cnn → cnn |
| 90 | 22.2068 | 17.3268 | 24.1833 | 5.2579 | 5.6635 | 5.5267 | gan → gan | cnn → cnn |
| 91 | 22.3094 | 16.2661 | 24.4404 | 5.5083 | 5.5198 | 5.8402 | gan → gan | cnn → cnn |
| 92 | 22.0793 | 16.8077 | 25.1250 | 5.7358 | 5.6779 | 5.6358 | gan → gan | gan → candidateUV_expanded2688 |
| 93 | 24.4029 | 18.1355 | 28.5879 | 6.5070 | 6.8160 | 6.0119 | gan → gan | cnn → candidateUV_expanded2688 |

Note: samples [12, 92] are in the MT-GAN-wins set (GAN beat CNN on MT in baseline).

### Q6. What happens in rare topology-CNN controls (samples 162 and 163)?

These samples were originally topology-CNN controls (CNN wins on topology metrics). candidateUV_expanded2688 should not reverse this.

| sample_idx | PD CNN | PD GAN | PD candidateUV_expanded2688 | MT CNN | MT GAN | MT candidateUV_expanded2688 | PD winner → | MT winner → |
|---|---|---|---|---|---|---|---|---|
| 162 | 18.2711 | 19.4579 | 23.3273 | 4.5406 | 9.6957 | 4.4565 | cnn → cnn | cnn → candidateUV_expanded2688 |
| 163 | 19.3704 | 20.4464 | 24.3422 | 5.6989 | 6.7890 | 5.0286 | cnn → cnn | cnn → candidateUV_expanded2688 |

## Winner distribution

### PD distance winners

| Method | Wins before candidateUV_expanded2688 | Wins after candidateUV_expanded2688 |
|---|---|---|
| cnn | 2 | 2 |
| gan | 166 | 166 |

### MT distance winners

| Method | Wins before candidateUV_expanded2688 | Wins after candidateUV_expanded2688 |
|---|---|---|
| candidateUV_expanded2688 | 0 | 56 |
| cnn | 148 | 97 |
| gan | 20 | 15 |

## Full per-sample detail: focus samples

Includes adjacent-cluster samples, rare controls, and the 20 MT-GAN baseline wins.

| sample | PD CNN | PD GAN | PD candidateUV_expanded2688 | MT CNN | MT GAN | MT candidateUV_expanded2688 | MT-GAN-win? | PD after | MT after |
|---|---|---|---|---|---|---|---|---|---|
| 6 | 28.0690 | 24.0451 | 31.5320 | 6.1440 | 6.0106 | 6.1235 | ✓ | gan | gan |
| 8 | 29.6351 | 23.2821 | 31.2299 | 7.5096 | 6.0550 | 6.5754 | ✓ | gan | gan |
| 10 | 29.6970 | 24.7100 | 30.8282 | 6.6345 | 7.4238 | 5.8184 |  | gan | candidateUV_expanded2688 |
| 11 | 28.6873 | 24.9699 | 30.7860 | 5.9107 | 7.8317 | 6.0474 |  | gan | cnn |
| 12 | 30.1045 | 24.2206 | 31.0968 | 6.6164 | 5.9137 | 6.4362 | ✓ | gan | gan |
| 13 | 29.2737 | 24.3237 | 30.1385 | 6.0738 | 7.4023 | 6.5902 |  | gan | cnn |
| 16 | 23.7560 | 17.3245 | 24.6256 | 5.4642 | 5.2552 | 6.1676 | ✓ | gan | gan |
| 17 | 20.9004 | 17.1181 | 22.3107 | 5.2897 | 4.8487 | 5.3730 | ✓ | gan | gan |
| 18 | 20.3949 | 18.6102 | 21.8212 | 6.2586 | 5.5544 | 5.5428 | ✓ | gan | candidateUV_expanded2688 |
| 19 | 21.0641 | 19.0867 | 21.5403 | 6.9280 | 5.4429 | 5.7450 | ✓ | gan | gan |
| 20 | 21.9792 | 18.2663 | 22.9531 | 6.2541 | 5.6891 | 6.2874 | ✓ | gan | gan |
| 25 | 36.5841 | 30.1235 | 38.8185 | 11.3716 | 10.5015 | 7.0109 | ✓ | gan | candidateUV_expanded2688 |
| 48 | 22.8325 | 13.1070 | 21.8707 | 7.0815 | 6.6830 | 7.1122 | ✓ | gan | gan |
| 62 | 27.4635 | 17.9937 | 29.0997 | 6.7203 | 6.6502 | 7.1866 | ✓ | gan | gan |
| 63 | 28.5246 | 18.1998 | 29.8026 | 7.6606 | 7.5776 | 7.4979 | ✓ | gan | candidateUV_expanded2688 |
| 65 | 22.1489 | 16.2395 | 23.2505 | 5.8208 | 5.5768 | 5.8542 | ✓ | gan | gan |
| 68 | 21.7713 | 15.5009 | 22.2635 | 6.0743 | 5.6709 | 5.5097 | ✓ | gan | candidateUV_expanded2688 |
| 77 | 29.9447 | 20.3533 | 31.9501 | 6.1320 | 6.0000 | 6.0612 | ✓ | gan | gan |
| 79 | 30.1481 | 20.3212 | 31.6107 | 6.8565 | 6.4670 | 6.6791 | ✓ | gan | gan |
| 80 | 30.0863 | 20.7449 | 31.5361 | 6.9827 | 6.1767 | 6.6164 | ✓ | gan | gan |
| 82 | 28.9009 | 20.8163 | 31.2836 | 7.1841 | 6.3036 | 6.3209 | ✓ | gan | gan |
| 90 | 22.2068 | 17.3268 | 24.1833 | 5.2579 | 5.6635 | 5.5267 |  | gan | cnn |
| 91 | 22.3094 | 16.2661 | 24.4404 | 5.5083 | 5.5198 | 5.8402 |  | gan | cnn |
| 92 | 22.0793 | 16.8077 | 25.1250 | 5.7358 | 5.6779 | 5.6358 | ✓ | gan | candidateUV_expanded2688 |
| 93 | 24.4029 | 18.1355 | 28.5879 | 6.5070 | 6.8160 | 6.0119 |  | gan | candidateUV_expanded2688 |
| 154 | 25.3102 | 16.8345 | 24.6978 | 5.8898 | 5.7368 | 6.0027 | ✓ | gan | gan |
| 162 | 18.2711 | 19.4579 | 23.3273 | 4.5406 | 9.6957 | 4.4565 |  | cnn | candidateUV_expanded2688 |
| 163 | 19.3704 | 20.4464 | 24.3422 | 5.6989 | 6.7890 | 5.0286 |  | cnn | candidateUV_expanded2688 |

## Output files

- `ttk_runs_fixed/topology_finetuning/candidateUV_expanded2688_topology/candidateUV_expanded2688_pd_mt_distances.csv` — candidateUV_expanded2688 PD/MT per sample
- `ttk_runs_fixed/topology_finetuning/candidateUV_expanded2688_topology/candidateUV_expanded2688_topology_comparison.csv` — three-way comparison (CNN, GAN, candidateUV_expanded2688)
- `docs/topology_finetuning_candidateUV_expanded2688_topology_eval.md` — this report

## Notes

- PD distance = bottleneck distance between persistence diagrams (computed by `ttkBottleneckDistance`).
- MT distance = Wasserstein-type merge tree distance (computed by `ttkMergeTreeDistanceMatrix`).
- Both use 160×160 speed patches at (x0=0, y0=0), consistent with the CNN/GAN baseline.
- **Lower distance = SR is topologically closer to GT. Lower is better for both metrics.**
- **Component-count proxy metrics** (counts of connected components in speed superlevel sets) are reported separately in `topology_finetuning_candidateUV_expanded2688_eval.md` and do not require TTK.
- candidateUV_expanded2688 was fine-tuned from the pretrained CNN checkpoint. Results may change with different lambda settings or more epochs.
- Existing CNN/GAN and Candidate B topology outputs were not overwritten.
