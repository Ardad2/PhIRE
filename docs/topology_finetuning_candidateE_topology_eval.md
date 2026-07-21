# candidateE topology evaluation

**Generated:** 2026-05-20

**Important:** This report covers **true topology metrics** (PD bottleneck distance, MT Wasserstein distance) computed by TTK. These are distinct from the component-count *proxy* metrics in `topology_finetuning_candidateE_eval.md`.

**Note:** Lower PD/MT distance = SR is topologically closer to GT. **Lower is better for both metrics.**

**candidateE:** fine-tuned from the pretrained CNN checkpoint with auxiliary physics/topology losses. See the corresponding fine-tuning pilot script for configuration details.

**Samples evaluated:** 168

**MT-GAN baseline wins (before candidateE):** 20 samples — [6, 8, 12, 16, 17, 18, 19, 20, 25, 48, 62, 63, 65, 68, 77, 79, 80, 82, 92, 154]

## Summary statistics

| Metric | CNN (baseline) | GAN (baseline) | candidateE |
|---|---|---|---|
| PD distance (mean) | 27.4063 | 20.8641 | 28.2522 |
| MT distance (mean) | 5.8678 | 8.3481 | 6.0788 |
| PD wins (vs CNN) | — | 166 | 0 |
| MT wins (vs CNN) | — | 20 | 35 |
| PD beats GAN | — | — | 2 |
| MT beats GAN | — | — | 144 |

## 6 Key evaluation questions

### Q1. Does candidateE improve PD distance relative to baseline CNN?

- Mean PD: CNN=27.4063, candidateE=28.2522, Δ=0.8459 (▲ worse)
- candidateE has lower PD on **0/168** samples.

### Q2. Does candidateE ever beat GAN on PD distance?

- candidateE beats GAN on PD for **2/168** samples.
- Mean PD: GAN=20.8641, candidateE=28.2522. Δ=7.3881 (▲ worse than GAN on average)

### Q3. Does candidateE improve MT distance relative to baseline CNN?

- Mean MT: CNN=5.8678, candidateE=6.0788, Δ=0.2110 (▲ worse)
- candidateE has lower MT on **35/168** samples.

### Q4. Does candidateE change any of the original 20 MT-GAN cases?

The 20 samples where GAN originally beat CNN on MT distance: [6, 8, 12, 16, 17, 18, 19, 20, 25, 48, 62, 63, 65, 68, 77, 79, 80, 82, 92, 154]

| Outcome | Count |
|---|---|
| MT winner stays GAN | 19 |
| MT winner changes to candidateE | 1 |
| MT winner changes to CNN | 0 |

Per-sample detail for changed cases:

| sample_idx | MT CNN | MT GAN | MT candidateE | winner before | winner after |
|---|---|---|---|---|---|
| 25 | 11.3716 | 10.5015 | 7.4313 | gan | candidateE |

### Q5. What happens in adjacent clusters (samples 10–13 and 90–93)?

| sample_idx | PD CNN | PD GAN | PD candidateE | MT CNN | MT GAN | MT candidateE | PD winner → | MT winner → |
|---|---|---|---|---|---|---|---|---|
| 10 | 29.6970 | 24.7100 | 30.7179 | 6.6345 | 7.4238 | 6.8296 | gan → gan | cnn → cnn |
| 11 | 28.6873 | 24.9699 | 30.0162 | 5.9107 | 7.8317 | 5.9136 | gan → gan | cnn → cnn |
| 12 | 30.1045 | 24.2206 | 31.4389 | 6.6164 | 5.9137 | 6.9581 | gan → gan | gan → gan |
| 13 | 29.2737 | 24.3237 | 30.4099 | 6.0738 | 7.4023 | 6.6930 | gan → gan | cnn → cnn |
| 90 | 22.2068 | 17.3268 | 22.5417 | 5.2579 | 5.6635 | 5.4135 | gan → gan | cnn → cnn |
| 91 | 22.3094 | 16.2661 | 22.6712 | 5.5083 | 5.5198 | 5.5350 | gan → gan | cnn → cnn |
| 92 | 22.0793 | 16.8077 | 22.3903 | 5.7358 | 5.6779 | 5.7631 | gan → gan | gan → gan |
| 93 | 24.4029 | 18.1355 | 24.8935 | 6.5070 | 6.8160 | 6.6135 | gan → gan | cnn → cnn |

Note: samples [12, 92] are in the MT-GAN-wins set (GAN beat CNN on MT in baseline).

### Q6. What happens in rare topology-CNN controls (samples 162 and 163)?

These samples were originally topology-CNN controls (CNN wins on topology metrics). candidateE should not reverse this.

| sample_idx | PD CNN | PD GAN | PD candidateE | MT CNN | MT GAN | MT candidateE | PD winner → | MT winner → |
|---|---|---|---|---|---|---|---|---|
| 162 | 18.2711 | 19.4579 | 18.4935 | 4.5406 | 9.6957 | 4.6497 | cnn → cnn | cnn → cnn |
| 163 | 19.3704 | 20.4464 | 19.5869 | 5.6989 | 6.7890 | 5.2257 | cnn → cnn | cnn → candidateE |

## Winner distribution

### PD distance winners

| Method | Wins before candidateE | Wins after candidateE |
|---|---|---|
| cnn | 2 | 2 |
| gan | 166 | 166 |

### MT distance winners

| Method | Wins before candidateE | Wins after candidateE |
|---|---|---|
| candidateE | 0 | 33 |
| cnn | 148 | 116 |
| gan | 20 | 19 |

## Full per-sample detail: focus samples

Includes adjacent-cluster samples, rare controls, and the 20 MT-GAN baseline wins.

| sample | PD CNN | PD GAN | PD candidateE | MT CNN | MT GAN | MT candidateE | MT-GAN-win? | PD after | MT after |
|---|---|---|---|---|---|---|---|---|---|
| 6 | 28.0690 | 24.0451 | 28.8664 | 6.1440 | 6.0106 | 6.4785 | ✓ | gan | gan |
| 8 | 29.6351 | 23.2821 | 30.4052 | 7.5096 | 6.0550 | 7.5925 | ✓ | gan | gan |
| 10 | 29.6970 | 24.7100 | 30.7179 | 6.6345 | 7.4238 | 6.8296 |  | gan | cnn |
| 11 | 28.6873 | 24.9699 | 30.0162 | 5.9107 | 7.8317 | 5.9136 |  | gan | cnn |
| 12 | 30.1045 | 24.2206 | 31.4389 | 6.6164 | 5.9137 | 6.9581 | ✓ | gan | gan |
| 13 | 29.2737 | 24.3237 | 30.4099 | 6.0738 | 7.4023 | 6.6930 |  | gan | cnn |
| 16 | 23.7560 | 17.3245 | 24.6367 | 5.4642 | 5.2552 | 6.1791 | ✓ | gan | gan |
| 17 | 20.9004 | 17.1181 | 21.7682 | 5.2897 | 4.8487 | 5.4065 | ✓ | gan | gan |
| 18 | 20.3949 | 18.6102 | 21.4355 | 6.2586 | 5.5544 | 6.3342 | ✓ | gan | gan |
| 19 | 21.0641 | 19.0867 | 21.9549 | 6.9280 | 5.4429 | 7.1693 | ✓ | gan | gan |
| 20 | 21.9792 | 18.2663 | 23.1477 | 6.2541 | 5.6891 | 6.8558 | ✓ | gan | gan |
| 25 | 36.5841 | 30.1235 | 38.6674 | 11.3716 | 10.5015 | 7.4313 | ✓ | gan | candidateE |
| 48 | 22.8325 | 13.1070 | 23.0791 | 7.0815 | 6.6830 | 6.8813 | ✓ | gan | gan |
| 62 | 27.4635 | 17.9937 | 27.9172 | 6.7203 | 6.6502 | 7.3882 | ✓ | gan | gan |
| 63 | 28.5246 | 18.1998 | 28.9062 | 7.6606 | 7.5776 | 7.7284 | ✓ | gan | gan |
| 65 | 22.1489 | 16.2395 | 22.4386 | 5.8208 | 5.5768 | 6.1146 | ✓ | gan | gan |
| 68 | 21.7713 | 15.5009 | 21.9706 | 6.0743 | 5.6709 | 6.4579 | ✓ | gan | gan |
| 77 | 29.9447 | 20.3533 | 30.2705 | 6.1320 | 6.0000 | 6.3991 | ✓ | gan | gan |
| 79 | 30.1481 | 20.3212 | 30.6157 | 6.8565 | 6.4670 | 7.3588 | ✓ | gan | gan |
| 80 | 30.0863 | 20.7449 | 30.5071 | 6.9827 | 6.1767 | 6.8980 | ✓ | gan | gan |
| 82 | 28.9009 | 20.8163 | 29.6226 | 7.1841 | 6.3036 | 8.0248 | ✓ | gan | gan |
| 90 | 22.2068 | 17.3268 | 22.5417 | 5.2579 | 5.6635 | 5.4135 |  | gan | cnn |
| 91 | 22.3094 | 16.2661 | 22.6712 | 5.5083 | 5.5198 | 5.5350 |  | gan | cnn |
| 92 | 22.0793 | 16.8077 | 22.3903 | 5.7358 | 5.6779 | 5.7631 | ✓ | gan | gan |
| 93 | 24.4029 | 18.1355 | 24.8935 | 6.5070 | 6.8160 | 6.6135 |  | gan | cnn |
| 154 | 25.3102 | 16.8345 | 25.6237 | 5.8898 | 5.7368 | 6.0211 | ✓ | gan | gan |
| 162 | 18.2711 | 19.4579 | 18.4935 | 4.5406 | 9.6957 | 4.6497 |  | cnn | cnn |
| 163 | 19.3704 | 20.4464 | 19.5869 | 5.6989 | 6.7890 | 5.2257 |  | cnn | candidateE |

## Output files

- `ttk_runs_fixed/topology_finetuning/candidateE_topology/candidateE_pd_mt_distances.csv` — candidateE PD/MT per sample
- `ttk_runs_fixed/topology_finetuning/candidateE_topology/candidateE_topology_comparison.csv` — three-way comparison (CNN, GAN, candidateE)
- `docs/topology_finetuning_candidateE_topology_eval.md` — this report

## Notes

- PD distance = bottleneck distance between persistence diagrams (computed by `ttkBottleneckDistance`).
- MT distance = Wasserstein-type merge tree distance (computed by `ttkMergeTreeDistanceMatrix`).
- Both use 160×160 speed patches at (x0=0, y0=0), consistent with the CNN/GAN baseline.
- **Lower distance = SR is topologically closer to GT. Lower is better for both metrics.**
- **Component-count proxy metrics** (counts of connected components in speed superlevel sets) are reported separately in `topology_finetuning_candidateE_eval.md` and do not require TTK.
- candidateE was fine-tuned from the pretrained CNN checkpoint. Results may change with different lambda settings or more epochs.
- Existing CNN/GAN and Candidate B topology outputs were not overwritten.
