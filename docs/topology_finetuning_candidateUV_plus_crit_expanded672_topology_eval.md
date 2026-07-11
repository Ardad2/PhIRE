# candidateUV_plus_crit_expanded672 topology evaluation

**Generated:** 2026-07-11

**Important:** This report covers **true topology metrics** (PD bottleneck distance, MT Wasserstein distance) computed by TTK. These are distinct from the component-count *proxy* metrics in `topology_finetuning_candidateUV_plus_crit_expanded672_eval.md`.

**Note:** Lower PD/MT distance = SR is topologically closer to GT. **Lower is better for both metrics.**

**candidateUV_plus_crit_expanded672:** fine-tuned from the pretrained CNN checkpoint with auxiliary physics/topology losses. See the corresponding fine-tuning pilot script for configuration details.

**Samples evaluated:** 168

**MT-GAN baseline wins (before candidateUV_plus_crit_expanded672):** 20 samples — [6, 8, 12, 16, 17, 18, 19, 20, 25, 48, 62, 63, 65, 68, 77, 79, 80, 82, 92, 154]

## Summary statistics

| Metric | CNN (baseline) | GAN (baseline) | candidateUV_plus_crit_expanded672 |
|---|---|---|---|
| PD distance (mean) | 27.4063 | 20.8641 | 29.4764 |
| MT distance (mean) | 5.8678 | 8.3481 | 5.9217 |
| PD wins (vs CNN) | — | 166 | 8 |
| MT wins (vs CNN) | — | 20 | 65 |
| PD beats GAN | — | — | 0 |
| MT beats GAN | — | — | 149 |

## 6 Key evaluation questions

### Q1. Does candidateUV_plus_crit_expanded672 improve PD distance relative to baseline CNN?

- Mean PD: CNN=27.4063, candidateUV_plus_crit_expanded672=29.4764, Δ=2.0702 (▲ worse)
- candidateUV_plus_crit_expanded672 has lower PD on **8/168** samples.

### Q2. Does candidateUV_plus_crit_expanded672 ever beat GAN on PD distance?

- candidateUV_plus_crit_expanded672 beats GAN on PD for **0/168** samples.
- Mean PD: GAN=20.8641, candidateUV_plus_crit_expanded672=29.4764. Δ=8.6123 (▲ worse than GAN on average)

### Q3. Does candidateUV_plus_crit_expanded672 improve MT distance relative to baseline CNN?

- Mean MT: CNN=5.8678, candidateUV_plus_crit_expanded672=5.9217, Δ=0.0539 (▲ worse)
- candidateUV_plus_crit_expanded672 has lower MT on **65/168** samples.

### Q4. Does candidateUV_plus_crit_expanded672 change any of the original 20 MT-GAN cases?

The 20 samples where GAN originally beat CNN on MT distance: [6, 8, 12, 16, 17, 18, 19, 20, 25, 48, 62, 63, 65, 68, 77, 79, 80, 82, 92, 154]

| Outcome | Count |
|---|---|
| MT winner stays GAN | 13 |
| MT winner changes to candidateUV_plus_crit_expanded672 | 7 |
| MT winner changes to CNN | 0 |

Per-sample detail for changed cases:

| sample_idx | MT CNN | MT GAN | MT candidateUV_plus_crit_expanded672 | winner before | winner after |
|---|---|---|---|---|---|
| 6 | 6.1440 | 6.0106 | 5.9035 | gan | candidateUV_plus_crit_expanded672 |
| 25 | 11.3716 | 10.5015 | 6.5120 | gan | candidateUV_plus_crit_expanded672 |
| 63 | 7.6606 | 7.5776 | 7.0269 | gan | candidateUV_plus_crit_expanded672 |
| 65 | 5.8208 | 5.5768 | 5.5520 | gan | candidateUV_plus_crit_expanded672 |
| 68 | 6.0743 | 5.6709 | 5.4792 | gan | candidateUV_plus_crit_expanded672 |
| 80 | 6.9827 | 6.1767 | 6.0774 | gan | candidateUV_plus_crit_expanded672 |
| 92 | 5.7358 | 5.6779 | 5.1622 | gan | candidateUV_plus_crit_expanded672 |

### Q5. What happens in adjacent clusters (samples 10–13 and 90–93)?

| sample_idx | PD CNN | PD GAN | PD candidateUV_plus_crit_expanded672 | MT CNN | MT GAN | MT candidateUV_plus_crit_expanded672 | PD winner → | MT winner → |
|---|---|---|---|---|---|---|---|---|
| 10 | 29.6970 | 24.7100 | 31.0687 | 6.6345 | 7.4238 | 5.8770 | gan → gan | cnn → candidateUV_plus_crit_expanded672 |
| 11 | 28.6873 | 24.9699 | 30.8139 | 5.9107 | 7.8317 | 6.0462 | gan → gan | cnn → cnn |
| 12 | 30.1045 | 24.2206 | 31.4205 | 6.6164 | 5.9137 | 6.4888 | gan → gan | gan → gan |
| 13 | 29.2737 | 24.3237 | 30.5402 | 6.0738 | 7.4023 | 6.4669 | gan → gan | cnn → cnn |
| 90 | 22.2068 | 17.3268 | 23.7229 | 5.2579 | 5.6635 | 5.3258 | gan → gan | cnn → cnn |
| 91 | 22.3094 | 16.2661 | 24.0976 | 5.5083 | 5.5198 | 5.9114 | gan → gan | cnn → cnn |
| 92 | 22.0793 | 16.8077 | 24.6962 | 5.7358 | 5.6779 | 5.1622 | gan → gan | gan → candidateUV_plus_crit_expanded672 |
| 93 | 24.4029 | 18.1355 | 28.0934 | 6.5070 | 6.8160 | 6.0011 | gan → gan | cnn → candidateUV_plus_crit_expanded672 |

Note: samples [12, 92] are in the MT-GAN-wins set (GAN beat CNN on MT in baseline).

### Q6. What happens in rare topology-CNN controls (samples 162 and 163)?

These samples were originally topology-CNN controls (CNN wins on topology metrics). candidateUV_plus_crit_expanded672 should not reverse this.

| sample_idx | PD CNN | PD GAN | PD candidateUV_plus_crit_expanded672 | MT CNN | MT GAN | MT candidateUV_plus_crit_expanded672 | PD winner → | MT winner → |
|---|---|---|---|---|---|---|---|---|
| 162 | 18.2711 | 19.4579 | 22.9580 | 4.5406 | 9.6957 | 4.2404 | cnn → cnn | cnn → candidateUV_plus_crit_expanded672 |
| 163 | 19.3704 | 20.4464 | 24.3444 | 5.6989 | 6.7890 | 4.8452 | cnn → cnn | cnn → candidateUV_plus_crit_expanded672 |

## Winner distribution

### PD distance winners

| Method | Wins before candidateUV_plus_crit_expanded672 | Wins after candidateUV_plus_crit_expanded672 |
|---|---|---|
| cnn | 2 | 2 |
| gan | 166 | 166 |

### MT distance winners

| Method | Wins before candidateUV_plus_crit_expanded672 | Wins after candidateUV_plus_crit_expanded672 |
|---|---|---|
| candidateUV_plus_crit_expanded672 | 0 | 58 |
| cnn | 148 | 97 |
| gan | 20 | 13 |

## Full per-sample detail: focus samples

Includes adjacent-cluster samples, rare controls, and the 20 MT-GAN baseline wins.

| sample | PD CNN | PD GAN | PD candidateUV_plus_crit_expanded672 | MT CNN | MT GAN | MT candidateUV_plus_crit_expanded672 | MT-GAN-win? | PD after | MT after |
|---|---|---|---|---|---|---|---|---|---|
| 6 | 28.0690 | 24.0451 | 31.1019 | 6.1440 | 6.0106 | 5.9035 | ✓ | gan | candidateUV_plus_crit_expanded672 |
| 8 | 29.6351 | 23.2821 | 31.3965 | 7.5096 | 6.0550 | 6.3307 | ✓ | gan | gan |
| 10 | 29.6970 | 24.7100 | 31.0687 | 6.6345 | 7.4238 | 5.8770 |  | gan | candidateUV_plus_crit_expanded672 |
| 11 | 28.6873 | 24.9699 | 30.8139 | 5.9107 | 7.8317 | 6.0462 |  | gan | cnn |
| 12 | 30.1045 | 24.2206 | 31.4205 | 6.6164 | 5.9137 | 6.4888 | ✓ | gan | gan |
| 13 | 29.2737 | 24.3237 | 30.5402 | 6.0738 | 7.4023 | 6.4669 |  | gan | cnn |
| 16 | 23.7560 | 17.3245 | 24.7560 | 5.4642 | 5.2552 | 6.1464 | ✓ | gan | gan |
| 17 | 20.9004 | 17.1181 | 22.5460 | 5.2897 | 4.8487 | 5.3665 | ✓ | gan | gan |
| 18 | 20.3949 | 18.6102 | 21.8473 | 6.2586 | 5.5544 | 5.6969 | ✓ | gan | gan |
| 19 | 21.0641 | 19.0867 | 21.5870 | 6.9280 | 5.4429 | 6.3172 | ✓ | gan | gan |
| 20 | 21.9792 | 18.2663 | 23.2616 | 6.2541 | 5.6891 | 6.7360 | ✓ | gan | gan |
| 25 | 36.5841 | 30.1235 | 38.7108 | 11.3716 | 10.5015 | 6.5120 | ✓ | gan | candidateUV_plus_crit_expanded672 |
| 48 | 22.8325 | 13.1070 | 21.7809 | 7.0815 | 6.6830 | 6.7843 | ✓ | gan | gan |
| 62 | 27.4635 | 17.9937 | 29.0516 | 6.7203 | 6.6502 | 7.1729 | ✓ | gan | gan |
| 63 | 28.5246 | 18.1998 | 29.5449 | 7.6606 | 7.5776 | 7.0269 | ✓ | gan | candidateUV_plus_crit_expanded672 |
| 65 | 22.1489 | 16.2395 | 23.3709 | 5.8208 | 5.5768 | 5.5520 | ✓ | gan | candidateUV_plus_crit_expanded672 |
| 68 | 21.7713 | 15.5009 | 22.0854 | 6.0743 | 5.6709 | 5.4792 | ✓ | gan | candidateUV_plus_crit_expanded672 |
| 77 | 29.9447 | 20.3533 | 31.5553 | 6.1320 | 6.0000 | 6.4424 | ✓ | gan | gan |
| 79 | 30.1481 | 20.3212 | 31.4267 | 6.8565 | 6.4670 | 6.7390 | ✓ | gan | gan |
| 80 | 30.0863 | 20.7449 | 31.0951 | 6.9827 | 6.1767 | 6.0774 | ✓ | gan | candidateUV_plus_crit_expanded672 |
| 82 | 28.9009 | 20.8163 | 30.7963 | 7.1841 | 6.3036 | 6.3466 | ✓ | gan | gan |
| 90 | 22.2068 | 17.3268 | 23.7229 | 5.2579 | 5.6635 | 5.3258 |  | gan | cnn |
| 91 | 22.3094 | 16.2661 | 24.0976 | 5.5083 | 5.5198 | 5.9114 |  | gan | cnn |
| 92 | 22.0793 | 16.8077 | 24.6962 | 5.7358 | 5.6779 | 5.1622 | ✓ | gan | candidateUV_plus_crit_expanded672 |
| 93 | 24.4029 | 18.1355 | 28.0934 | 6.5070 | 6.8160 | 6.0011 |  | gan | candidateUV_plus_crit_expanded672 |
| 154 | 25.3102 | 16.8345 | 25.2855 | 5.8898 | 5.7368 | 6.0058 | ✓ | gan | gan |
| 162 | 18.2711 | 19.4579 | 22.9580 | 4.5406 | 9.6957 | 4.2404 |  | cnn | candidateUV_plus_crit_expanded672 |
| 163 | 19.3704 | 20.4464 | 24.3444 | 5.6989 | 6.7890 | 4.8452 |  | cnn | candidateUV_plus_crit_expanded672 |

## Output files

- `ttk_runs_fixed/topology_finetuning/candidateUV_plus_crit_expanded672_topology/candidateUV_plus_crit_expanded672_pd_mt_distances.csv` — candidateUV_plus_crit_expanded672 PD/MT per sample
- `ttk_runs_fixed/topology_finetuning/candidateUV_plus_crit_expanded672_topology/candidateUV_plus_crit_expanded672_topology_comparison.csv` — three-way comparison (CNN, GAN, candidateUV_plus_crit_expanded672)
- `docs/topology_finetuning_candidateUV_plus_crit_expanded672_topology_eval.md` — this report

## Notes

- PD distance = bottleneck distance between persistence diagrams (computed by `ttkBottleneckDistance`).
- MT distance = Wasserstein-type merge tree distance (computed by `ttkMergeTreeDistanceMatrix`).
- Both use 160×160 speed patches at (x0=0, y0=0), consistent with the CNN/GAN baseline.
- **Lower distance = SR is topologically closer to GT. Lower is better for both metrics.**
- **Component-count proxy metrics** (counts of connected components in speed superlevel sets) are reported separately in `topology_finetuning_candidateUV_plus_crit_expanded672_eval.md` and do not require TTK.
- candidateUV_plus_crit_expanded672 was fine-tuned from the pretrained CNN checkpoint. Results may change with different lambda settings or more epochs.
- Existing CNN/GAN and Candidate B topology outputs were not overwritten.
