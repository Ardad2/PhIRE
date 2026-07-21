# candidateE2_expanded672 topology evaluation

**Generated:** 2026-05-28

**Important:** This report covers **true topology metrics** (PD bottleneck distance, MT Wasserstein distance) computed by TTK. These are distinct from the component-count *proxy* metrics in `topology_finetuning_candidateE2_expanded672_eval.md`.

**Note:** Lower PD/MT distance = SR is topologically closer to GT. **Lower is better for both metrics.**

**candidateE2_expanded672:** fine-tuned from the pretrained CNN checkpoint with auxiliary physics/topology losses. See the corresponding fine-tuning pilot script for configuration details.

**Samples evaluated:** 168

**MT-GAN baseline wins (before candidateE2_expanded672):** 20 samples — [6, 8, 12, 16, 17, 18, 19, 20, 25, 48, 62, 63, 65, 68, 77, 79, 80, 82, 92, 154]

## Summary statistics

| Metric | CNN (baseline) | GAN (baseline) | candidateE2_expanded672 |
|---|---|---|---|
| PD distance (mean) | 27.4063 | 20.8641 | 28.6697 |
| MT distance (mean) | 5.8678 | 8.3481 | 6.1622 |
| PD wins (vs CNN) | — | 166 | 0 |
| MT wins (vs CNN) | — | 20 | 24 |
| PD beats GAN | — | — | 0 |
| MT beats GAN | — | — | 142 |

## 6 Key evaluation questions

### Q1. Does candidateE2_expanded672 improve PD distance relative to baseline CNN?

- Mean PD: CNN=27.4063, candidateE2_expanded672=28.6697, Δ=1.2634 (▲ worse)
- candidateE2_expanded672 has lower PD on **0/168** samples.

### Q2. Does candidateE2_expanded672 ever beat GAN on PD distance?

- candidateE2_expanded672 beats GAN on PD for **0/168** samples.
- Mean PD: GAN=20.8641, candidateE2_expanded672=28.6697. Δ=7.8056 (▲ worse than GAN on average)

### Q3. Does candidateE2_expanded672 improve MT distance relative to baseline CNN?

- Mean MT: CNN=5.8678, candidateE2_expanded672=6.1622, Δ=0.2944 (▲ worse)
- candidateE2_expanded672 has lower MT on **24/168** samples.

### Q4. Does candidateE2_expanded672 change any of the original 20 MT-GAN cases?

The 20 samples where GAN originally beat CNN on MT distance: [6, 8, 12, 16, 17, 18, 19, 20, 25, 48, 62, 63, 65, 68, 77, 79, 80, 82, 92, 154]

| Outcome | Count |
|---|---|
| MT winner stays GAN | 18 |
| MT winner changes to candidateE2_expanded672 | 2 |
| MT winner changes to CNN | 0 |

Per-sample detail for changed cases:

| sample_idx | MT CNN | MT GAN | MT candidateE2_expanded672 | winner before | winner after |
|---|---|---|---|---|---|
| 25 | 11.3716 | 10.5015 | 6.6751 | gan | candidateE2_expanded672 |
| 92 | 5.7358 | 5.6779 | 5.6222 | gan | candidateE2_expanded672 |

### Q5. What happens in adjacent clusters (samples 10–13 and 90–93)?

| sample_idx | PD CNN | PD GAN | PD candidateE2_expanded672 | MT CNN | MT GAN | MT candidateE2_expanded672 | PD winner → | MT winner → |
|---|---|---|---|---|---|---|---|---|
| 10 | 29.6970 | 24.7100 | 30.7716 | 6.6345 | 7.4238 | 7.0026 | gan → gan | cnn → cnn |
| 11 | 28.6873 | 24.9699 | 29.9865 | 5.9107 | 7.8317 | 6.1904 | gan → gan | cnn → cnn |
| 12 | 30.1045 | 24.2206 | 31.4387 | 6.6164 | 5.9137 | 7.0462 | gan → gan | gan → gan |
| 13 | 29.2737 | 24.3237 | 30.4106 | 6.0738 | 7.4023 | 6.6337 | gan → gan | cnn → cnn |
| 90 | 22.2068 | 17.3268 | 22.7763 | 5.2579 | 5.6635 | 5.3885 | gan → gan | cnn → cnn |
| 91 | 22.3094 | 16.2661 | 23.0255 | 5.5083 | 5.5198 | 5.7914 | gan → gan | cnn → cnn |
| 92 | 22.0793 | 16.8077 | 22.8988 | 5.7358 | 5.6779 | 5.6222 | gan → gan | gan → candidateE2_expanded672 |
| 93 | 24.4029 | 18.1355 | 25.6420 | 6.5070 | 6.8160 | 6.6958 | gan → gan | cnn → cnn |

Note: samples [12, 92] are in the MT-GAN-wins set (GAN beat CNN on MT in baseline).

### Q6. What happens in rare topology-CNN controls (samples 162 and 163)?

These samples were originally topology-CNN controls (CNN wins on topology metrics). candidateE2_expanded672 should not reverse this.

| sample_idx | PD CNN | PD GAN | PD candidateE2_expanded672 | MT CNN | MT GAN | MT candidateE2_expanded672 | PD winner → | MT winner → |
|---|---|---|---|---|---|---|---|---|
| 162 | 18.2711 | 19.4579 | 20.2006 | 4.5406 | 9.6957 | 4.5324 | cnn → cnn | cnn → candidateE2_expanded672 |
| 163 | 19.3704 | 20.4464 | 21.2994 | 5.6989 | 6.7890 | 4.7598 | cnn → cnn | cnn → candidateE2_expanded672 |

## Winner distribution

### PD distance winners

| Method | Wins before candidateE2_expanded672 | Wins after candidateE2_expanded672 |
|---|---|---|
| cnn | 2 | 2 |
| gan | 166 | 166 |

### MT distance winners

| Method | Wins before candidateE2_expanded672 | Wins after candidateE2_expanded672 |
|---|---|---|
| candidateE2_expanded672 | 0 | 22 |
| cnn | 148 | 128 |
| gan | 20 | 18 |

## Full per-sample detail: focus samples

Includes adjacent-cluster samples, rare controls, and the 20 MT-GAN baseline wins.

| sample | PD CNN | PD GAN | PD candidateE2_expanded672 | MT CNN | MT GAN | MT candidateE2_expanded672 | MT-GAN-win? | PD after | MT after |
|---|---|---|---|---|---|---|---|---|---|
| 6 | 28.0690 | 24.0451 | 29.7041 | 6.1440 | 6.0106 | 7.4410 | ✓ | gan | gan |
| 8 | 29.6351 | 23.2821 | 30.7225 | 7.5096 | 6.0550 | 7.9639 | ✓ | gan | gan |
| 10 | 29.6970 | 24.7100 | 30.7716 | 6.6345 | 7.4238 | 7.0026 |  | gan | cnn |
| 11 | 28.6873 | 24.9699 | 29.9865 | 5.9107 | 7.8317 | 6.1904 |  | gan | cnn |
| 12 | 30.1045 | 24.2206 | 31.4387 | 6.6164 | 5.9137 | 7.0462 | ✓ | gan | gan |
| 13 | 29.2737 | 24.3237 | 30.4106 | 6.0738 | 7.4023 | 6.6337 |  | gan | cnn |
| 16 | 23.7560 | 17.3245 | 24.5842 | 5.4642 | 5.2552 | 6.1402 | ✓ | gan | gan |
| 17 | 20.9004 | 17.1181 | 21.6896 | 5.2897 | 4.8487 | 5.4279 | ✓ | gan | gan |
| 18 | 20.3949 | 18.6102 | 21.4174 | 6.2586 | 5.5544 | 6.3659 | ✓ | gan | gan |
| 19 | 21.0641 | 19.0867 | 21.9273 | 6.9280 | 5.4429 | 7.2419 | ✓ | gan | gan |
| 20 | 21.9792 | 18.2663 | 23.1182 | 6.2541 | 5.6891 | 7.0173 | ✓ | gan | gan |
| 25 | 36.5841 | 30.1235 | 39.0800 | 11.3716 | 10.5015 | 6.6751 | ✓ | gan | candidateE2_expanded672 |
| 48 | 22.8325 | 13.1070 | 23.2718 | 7.0815 | 6.6830 | 7.0021 | ✓ | gan | gan |
| 62 | 27.4635 | 17.9937 | 27.9461 | 6.7203 | 6.6502 | 7.1721 | ✓ | gan | gan |
| 63 | 28.5246 | 18.1998 | 28.8764 | 7.6606 | 7.5776 | 7.7139 | ✓ | gan | gan |
| 65 | 22.1489 | 16.2395 | 22.4717 | 5.8208 | 5.5768 | 5.7764 | ✓ | gan | gan |
| 68 | 21.7713 | 15.5009 | 22.1481 | 6.0743 | 5.6709 | 6.3933 | ✓ | gan | gan |
| 77 | 29.9447 | 20.3533 | 30.2751 | 6.1320 | 6.0000 | 6.3315 | ✓ | gan | gan |
| 79 | 30.1481 | 20.3212 | 30.5878 | 6.8565 | 6.4670 | 7.1589 | ✓ | gan | gan |
| 80 | 30.0863 | 20.7449 | 30.4913 | 6.9827 | 6.1767 | 7.1658 | ✓ | gan | gan |
| 82 | 28.9009 | 20.8163 | 29.4701 | 7.1841 | 6.3036 | 7.6580 | ✓ | gan | gan |
| 90 | 22.2068 | 17.3268 | 22.7763 | 5.2579 | 5.6635 | 5.3885 |  | gan | cnn |
| 91 | 22.3094 | 16.2661 | 23.0255 | 5.5083 | 5.5198 | 5.7914 |  | gan | cnn |
| 92 | 22.0793 | 16.8077 | 22.8988 | 5.7358 | 5.6779 | 5.6222 | ✓ | gan | candidateE2_expanded672 |
| 93 | 24.4029 | 18.1355 | 25.6420 | 6.5070 | 6.8160 | 6.6958 |  | gan | cnn |
| 154 | 25.3102 | 16.8345 | 26.0017 | 5.8898 | 5.7368 | 6.1137 | ✓ | gan | gan |
| 162 | 18.2711 | 19.4579 | 20.2006 | 4.5406 | 9.6957 | 4.5324 |  | cnn | candidateE2_expanded672 |
| 163 | 19.3704 | 20.4464 | 21.2994 | 5.6989 | 6.7890 | 4.7598 |  | cnn | candidateE2_expanded672 |

## Output files

- `ttk_runs_fixed/topology_finetuning/candidateE2_expanded672_topology/candidateE2_expanded672_pd_mt_distances.csv` — candidateE2_expanded672 PD/MT per sample
- `ttk_runs_fixed/topology_finetuning/candidateE2_expanded672_topology/candidateE2_expanded672_topology_comparison.csv` — three-way comparison (CNN, GAN, candidateE2_expanded672)
- `docs/topology_finetuning_candidateE2_expanded672_topology_eval.md` — this report

## Notes

- PD distance = bottleneck distance between persistence diagrams (computed by `ttkBottleneckDistance`).
- MT distance = Wasserstein-type merge tree distance (computed by `ttkMergeTreeDistanceMatrix`).
- Both use 160×160 speed patches at (x0=0, y0=0), consistent with the CNN/GAN baseline.
- **Lower distance = SR is topologically closer to GT. Lower is better for both metrics.**
- **Component-count proxy metrics** (counts of connected components in speed superlevel sets) are reported separately in `topology_finetuning_candidateE2_expanded672_eval.md` and do not require TTK.
- candidateE2_expanded672 was fine-tuned from the pretrained CNN checkpoint. Results may change with different lambda settings or more epochs.
- Existing CNN/GAN and Candidate B topology outputs were not overwritten.
