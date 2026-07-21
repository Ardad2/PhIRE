# candidateUV topology evaluation

**Generated:** 2026-05-22

**Important:** This report covers **true topology metrics** (PD bottleneck distance, MT Wasserstein distance) computed by TTK. These are distinct from the component-count *proxy* metrics in `topology_finetuning_candidateUV_eval.md`.

**Note:** Lower PD/MT distance = SR is topologically closer to GT. **Lower is better for both metrics.**

**candidateUV:** fine-tuned from the pretrained CNN checkpoint with auxiliary physics/topology losses. See the corresponding fine-tuning pilot script for configuration details.

**Samples evaluated:** 168

**MT-GAN baseline wins (before candidateUV):** 20 samples — [6, 8, 12, 16, 17, 18, 19, 20, 25, 48, 62, 63, 65, 68, 77, 79, 80, 82, 92, 154]

## Summary statistics

| Metric | CNN (baseline) | GAN (baseline) | candidateUV |
|---|---|---|---|
| PD distance (mean) | 27.4063 | 20.8641 | 30.5098 |
| MT distance (mean) | 5.8678 | 8.3481 | 6.3862 |
| PD wins (vs CNN) | — | 166 | 0 |
| MT wins (vs CNN) | — | 20 | 26 |
| PD beats GAN | — | — | 0 |
| MT beats GAN | — | — | 129 |

## 6 Key evaluation questions

### Q1. Does candidateUV improve PD distance relative to baseline CNN?

- Mean PD: CNN=27.4063, candidateUV=30.5098, Δ=3.1035 (▲ worse)
- candidateUV has lower PD on **0/168** samples.

### Q2. Does candidateUV ever beat GAN on PD distance?

- candidateUV beats GAN on PD for **0/168** samples.
- Mean PD: GAN=20.8641, candidateUV=30.5098. Δ=9.6457 (▲ worse than GAN on average)

### Q3. Does candidateUV improve MT distance relative to baseline CNN?

- Mean MT: CNN=5.8678, candidateUV=6.3862, Δ=0.5184 (▲ worse)
- candidateUV has lower MT on **26/168** samples.

### Q4. Does candidateUV change any of the original 20 MT-GAN cases?

The 20 samples where GAN originally beat CNN on MT distance: [6, 8, 12, 16, 17, 18, 19, 20, 25, 48, 62, 63, 65, 68, 77, 79, 80, 82, 92, 154]

| Outcome | Count |
|---|---|
| MT winner stays GAN | 19 |
| MT winner changes to candidateUV | 1 |
| MT winner changes to CNN | 0 |

Per-sample detail for changed cases:

| sample_idx | MT CNN | MT GAN | MT candidateUV | winner before | winner after |
|---|---|---|---|---|---|
| 25 | 11.3716 | 10.5015 | 8.0850 | gan | candidateUV |

### Q5. What happens in adjacent clusters (samples 10–13 and 90–93)?

| sample_idx | PD CNN | PD GAN | PD candidateUV | MT CNN | MT GAN | MT candidateUV | PD winner → | MT winner → |
|---|---|---|---|---|---|---|---|---|
| 10 | 29.6970 | 24.7100 | 32.1076 | 6.6345 | 7.4238 | 6.4003 | gan → gan | cnn → candidateUV |
| 11 | 28.6873 | 24.9699 | 31.8788 | 5.9107 | 7.8317 | 5.8311 | gan → gan | cnn → candidateUV |
| 12 | 30.1045 | 24.2206 | 32.5674 | 6.6164 | 5.9137 | 6.6568 | gan → gan | gan → gan |
| 13 | 29.2737 | 24.3237 | 31.6091 | 6.0738 | 7.4023 | 6.7589 | gan → gan | cnn → cnn |
| 90 | 22.2068 | 17.3268 | 24.5423 | 5.2579 | 5.6635 | 5.7431 | gan → gan | cnn → cnn |
| 91 | 22.3094 | 16.2661 | 24.6914 | 5.5083 | 5.5198 | 6.2309 | gan → gan | cnn → cnn |
| 92 | 22.0793 | 16.8077 | 25.3450 | 5.7358 | 5.6779 | 5.7060 | gan → gan | gan → gan |
| 93 | 24.4029 | 18.1355 | 28.8161 | 6.5070 | 6.8160 | 6.4922 | gan → gan | cnn → candidateUV |

Note: samples [12, 92] are in the MT-GAN-wins set (GAN beat CNN on MT in baseline).

### Q6. What happens in rare topology-CNN controls (samples 162 and 163)?

These samples were originally topology-CNN controls (CNN wins on topology metrics). candidateUV should not reverse this.

| sample_idx | PD CNN | PD GAN | PD candidateUV | MT CNN | MT GAN | MT candidateUV | PD winner → | MT winner → |
|---|---|---|---|---|---|---|---|---|
| 162 | 18.2711 | 19.4579 | 23.4305 | 4.5406 | 9.6957 | 4.4615 | cnn → cnn | cnn → candidateUV |
| 163 | 19.3704 | 20.4464 | 24.5835 | 5.6989 | 6.7890 | 4.8489 | cnn → cnn | cnn → candidateUV |

## Winner distribution

### PD distance winners

| Method | Wins before candidateUV | Wins after candidateUV |
|---|---|---|
| cnn | 2 | 2 |
| gan | 166 | 166 |

### MT distance winners

| Method | Wins before candidateUV | Wins after candidateUV |
|---|---|---|
| candidateUV | 0 | 21 |
| cnn | 148 | 128 |
| gan | 20 | 19 |

## Full per-sample detail: focus samples

Includes adjacent-cluster samples, rare controls, and the 20 MT-GAN baseline wins.

| sample | PD CNN | PD GAN | PD candidateUV | MT CNN | MT GAN | MT candidateUV | MT-GAN-win? | PD after | MT after |
|---|---|---|---|---|---|---|---|---|---|
| 6 | 28.0690 | 24.0451 | 32.0537 | 6.1440 | 6.0106 | 6.6661 | ✓ | gan | gan |
| 8 | 29.6351 | 23.2821 | 32.6786 | 7.5096 | 6.0550 | 7.4128 | ✓ | gan | gan |
| 10 | 29.6970 | 24.7100 | 32.1076 | 6.6345 | 7.4238 | 6.4003 |  | gan | candidateUV |
| 11 | 28.6873 | 24.9699 | 31.8788 | 5.9107 | 7.8317 | 5.8311 |  | gan | candidateUV |
| 12 | 30.1045 | 24.2206 | 32.5674 | 6.6164 | 5.9137 | 6.6568 | ✓ | gan | gan |
| 13 | 29.2737 | 24.3237 | 31.6091 | 6.0738 | 7.4023 | 6.7589 |  | gan | cnn |
| 16 | 23.7560 | 17.3245 | 25.5648 | 5.4642 | 5.2552 | 6.2894 | ✓ | gan | gan |
| 17 | 20.9004 | 17.1181 | 23.3626 | 5.2897 | 4.8487 | 5.8154 | ✓ | gan | gan |
| 18 | 20.3949 | 18.6102 | 22.9025 | 6.2586 | 5.5544 | 6.3585 | ✓ | gan | gan |
| 19 | 21.0641 | 19.0867 | 22.5661 | 6.9280 | 5.4429 | 6.5459 | ✓ | gan | gan |
| 20 | 21.9792 | 18.2663 | 23.9551 | 6.2541 | 5.6891 | 6.4892 | ✓ | gan | gan |
| 25 | 36.5841 | 30.1235 | 40.0137 | 11.3716 | 10.5015 | 8.0850 | ✓ | gan | candidateUV |
| 48 | 22.8325 | 13.1070 | 23.2757 | 7.0815 | 6.6830 | 7.4637 | ✓ | gan | gan |
| 62 | 27.4635 | 17.9937 | 29.7573 | 6.7203 | 6.6502 | 7.2540 | ✓ | gan | gan |
| 63 | 28.5246 | 18.1998 | 30.5368 | 7.6606 | 7.5776 | 7.6689 | ✓ | gan | gan |
| 65 | 22.1489 | 16.2395 | 23.8001 | 5.8208 | 5.5768 | 5.7441 | ✓ | gan | gan |
| 68 | 21.7713 | 15.5009 | 23.1531 | 6.0743 | 5.6709 | 6.1433 | ✓ | gan | gan |
| 77 | 29.9447 | 20.3533 | 33.0546 | 6.1320 | 6.0000 | 6.4444 | ✓ | gan | gan |
| 79 | 30.1481 | 20.3212 | 32.7252 | 6.8565 | 6.4670 | 7.8227 | ✓ | gan | gan |
| 80 | 30.0863 | 20.7449 | 32.5164 | 6.9827 | 6.1767 | 6.7089 | ✓ | gan | gan |
| 82 | 28.9009 | 20.8163 | 32.1341 | 7.1841 | 6.3036 | 7.5839 | ✓ | gan | gan |
| 90 | 22.2068 | 17.3268 | 24.5423 | 5.2579 | 5.6635 | 5.7431 |  | gan | cnn |
| 91 | 22.3094 | 16.2661 | 24.6914 | 5.5083 | 5.5198 | 6.2309 |  | gan | cnn |
| 92 | 22.0793 | 16.8077 | 25.3450 | 5.7358 | 5.6779 | 5.7060 | ✓ | gan | gan |
| 93 | 24.4029 | 18.1355 | 28.8161 | 6.5070 | 6.8160 | 6.4922 |  | gan | candidateUV |
| 154 | 25.3102 | 16.8345 | 26.6710 | 5.8898 | 5.7368 | 6.1746 | ✓ | gan | gan |
| 162 | 18.2711 | 19.4579 | 23.4305 | 4.5406 | 9.6957 | 4.4615 |  | cnn | candidateUV |
| 163 | 19.3704 | 20.4464 | 24.5835 | 5.6989 | 6.7890 | 4.8489 |  | cnn | candidateUV |

## Output files

- `ttk_runs_fixed/topology_finetuning/candidateUV_topology/candidateUV_pd_mt_distances.csv` — candidateUV PD/MT per sample
- `ttk_runs_fixed/topology_finetuning/candidateUV_topology/candidateUV_topology_comparison.csv` — three-way comparison (CNN, GAN, candidateUV)
- `docs/topology_finetuning_candidateUV_topology_eval.md` — this report

## Notes

- PD distance = bottleneck distance between persistence diagrams (computed by `ttkBottleneckDistance`).
- MT distance = Wasserstein-type merge tree distance (computed by `ttkMergeTreeDistanceMatrix`).
- Both use 160×160 speed patches at (x0=0, y0=0), consistent with the CNN/GAN baseline.
- **Lower distance = SR is topologically closer to GT. Lower is better for both metrics.**
- **Component-count proxy metrics** (counts of connected components in speed superlevel sets) are reported separately in `topology_finetuning_candidateUV_eval.md` and do not require TTK.
- candidateUV was fine-tuned from the pretrained CNN checkpoint. Results may change with different lambda settings or more epochs.
- Existing CNN/GAN and Candidate B topology outputs were not overwritten.
