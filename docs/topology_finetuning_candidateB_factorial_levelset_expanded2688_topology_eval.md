# candidateB_factorial_levelset_expanded2688 topology evaluation

**Generated:** 2026-07-15

**Important:** This report covers **true topology metrics** (PD bottleneck distance, MT Wasserstein distance) computed by TTK. These are distinct from the component-count *proxy* metrics in `topology_finetuning_candidateB_factorial_levelset_expanded2688_eval.md`.

**Note:** Lower PD/MT distance = SR is topologically closer to GT. **Lower is better for both metrics.**

**candidateB_factorial_levelset_expanded2688:** fine-tuned from the pretrained CNN checkpoint with auxiliary physics/topology losses. See the corresponding fine-tuning pilot script for configuration details.

**Samples evaluated:** 168

**MT-GAN baseline wins (before candidateB_factorial_levelset_expanded2688):** 20 samples — [6, 8, 12, 16, 17, 18, 19, 20, 25, 48, 62, 63, 65, 68, 77, 79, 80, 82, 92, 154]

## Summary statistics

| Metric | CNN (baseline) | GAN (baseline) | candidateB_factorial_levelset_expanded2688 |
|---|---|---|---|
| PD distance (mean) | 27.4063 | 20.8641 | 29.5953 |
| MT distance (mean) | 5.8678 | 8.3481 | 6.0076 |
| PD wins (vs CNN) | — | 166 | 10 |
| MT wins (vs CNN) | — | 20 | 64 |
| PD beats GAN | — | — | 0 |
| MT beats GAN | — | — | 151 |

## 6 Key evaluation questions

### Q1. Does candidateB_factorial_levelset_expanded2688 improve PD distance relative to baseline CNN?

- Mean PD: CNN=27.4063, candidateB_factorial_levelset_expanded2688=29.5953, Δ=2.1890 (▲ worse)
- candidateB_factorial_levelset_expanded2688 has lower PD on **10/168** samples.

### Q2. Does candidateB_factorial_levelset_expanded2688 ever beat GAN on PD distance?

- candidateB_factorial_levelset_expanded2688 beats GAN on PD for **0/168** samples.
- Mean PD: GAN=20.8641, candidateB_factorial_levelset_expanded2688=29.5953. Δ=8.7312 (▲ worse than GAN on average)

### Q3. Does candidateB_factorial_levelset_expanded2688 improve MT distance relative to baseline CNN?

- Mean MT: CNN=5.8678, candidateB_factorial_levelset_expanded2688=6.0076, Δ=0.1398 (▲ worse)
- candidateB_factorial_levelset_expanded2688 has lower MT on **64/168** samples.

### Q4. Does candidateB_factorial_levelset_expanded2688 change any of the original 20 MT-GAN cases?

The 20 samples where GAN originally beat CNN on MT distance: [6, 8, 12, 16, 17, 18, 19, 20, 25, 48, 62, 63, 65, 68, 77, 79, 80, 82, 92, 154]

| Outcome | Count |
|---|---|
| MT winner stays GAN | 12 |
| MT winner changes to candidateB_factorial_levelset_expanded2688 | 8 |
| MT winner changes to CNN | 0 |

Per-sample detail for changed cases:

| sample_idx | MT CNN | MT GAN | MT candidateB_factorial_levelset_expanded2688 | winner before | winner after |
|---|---|---|---|---|---|
| 6 | 6.1440 | 6.0106 | 6.0013 | gan | candidateB_factorial_levelset_expanded2688 |
| 18 | 6.2586 | 5.5544 | 5.5339 | gan | candidateB_factorial_levelset_expanded2688 |
| 25 | 11.3716 | 10.5015 | 7.1045 | gan | candidateB_factorial_levelset_expanded2688 |
| 63 | 7.6606 | 7.5776 | 7.5398 | gan | candidateB_factorial_levelset_expanded2688 |
| 68 | 6.0743 | 5.6709 | 5.5914 | gan | candidateB_factorial_levelset_expanded2688 |
| 77 | 6.1320 | 6.0000 | 5.9805 | gan | candidateB_factorial_levelset_expanded2688 |
| 82 | 7.1841 | 6.3036 | 6.2746 | gan | candidateB_factorial_levelset_expanded2688 |
| 92 | 5.7358 | 5.6779 | 5.3820 | gan | candidateB_factorial_levelset_expanded2688 |

### Q5. What happens in adjacent clusters (samples 10–13 and 90–93)?

| sample_idx | PD CNN | PD GAN | PD candidateB_factorial_levelset_expanded2688 | MT CNN | MT GAN | MT candidateB_factorial_levelset_expanded2688 | PD winner → | MT winner → |
|---|---|---|---|---|---|---|---|---|
| 10 | 29.6970 | 24.7100 | 30.9223 | 6.6345 | 7.4238 | 5.9031 | gan → gan | cnn → candidateB_factorial_levelset_expanded2688 |
| 11 | 28.6873 | 24.9699 | 30.8535 | 5.9107 | 7.8317 | 5.9923 | gan → gan | cnn → cnn |
| 12 | 30.1045 | 24.2206 | 31.2004 | 6.6164 | 5.9137 | 6.4604 | gan → gan | gan → gan |
| 13 | 29.2737 | 24.3237 | 30.2222 | 6.0738 | 7.4023 | 6.5683 | gan → gan | cnn → cnn |
| 90 | 22.2068 | 17.3268 | 24.1329 | 5.2579 | 5.6635 | 5.4807 | gan → gan | cnn → cnn |
| 91 | 22.3094 | 16.2661 | 24.4595 | 5.5083 | 5.5198 | 5.7771 | gan → gan | cnn → cnn |
| 92 | 22.0793 | 16.8077 | 25.1134 | 5.7358 | 5.6779 | 5.3820 | gan → gan | gan → candidateB_factorial_levelset_expanded2688 |
| 93 | 24.4029 | 18.1355 | 28.6427 | 6.5070 | 6.8160 | 6.0761 | gan → gan | cnn → candidateB_factorial_levelset_expanded2688 |

Note: samples [12, 92] are in the MT-GAN-wins set (GAN beat CNN on MT in baseline).

### Q6. What happens in rare topology-CNN controls (samples 162 and 163)?

These samples were originally topology-CNN controls (CNN wins on topology metrics). candidateB_factorial_levelset_expanded2688 should not reverse this.

| sample_idx | PD CNN | PD GAN | PD candidateB_factorial_levelset_expanded2688 | MT CNN | MT GAN | MT candidateB_factorial_levelset_expanded2688 | PD winner → | MT winner → |
|---|---|---|---|---|---|---|---|---|
| 162 | 18.2711 | 19.4579 | 23.3082 | 4.5406 | 9.6957 | 4.5317 | cnn → cnn | cnn → candidateB_factorial_levelset_expanded2688 |
| 163 | 19.3704 | 20.4464 | 24.4548 | 5.6989 | 6.7890 | 5.1335 | cnn → cnn | cnn → candidateB_factorial_levelset_expanded2688 |

## Winner distribution

### PD distance winners

| Method | Wins before candidateB_factorial_levelset_expanded2688 | Wins after candidateB_factorial_levelset_expanded2688 |
|---|---|---|
| cnn | 2 | 2 |
| gan | 166 | 166 |

### MT distance winners

| Method | Wins before candidateB_factorial_levelset_expanded2688 | Wins after candidateB_factorial_levelset_expanded2688 |
|---|---|---|
| candidateB_factorial_levelset_expanded2688 | 0 | 57 |
| cnn | 148 | 99 |
| gan | 20 | 12 |

## Full per-sample detail: focus samples

Includes adjacent-cluster samples, rare controls, and the 20 MT-GAN baseline wins.

| sample | PD CNN | PD GAN | PD candidateB_factorial_levelset_expanded2688 | MT CNN | MT GAN | MT candidateB_factorial_levelset_expanded2688 | MT-GAN-win? | PD after | MT after |
|---|---|---|---|---|---|---|---|---|---|
| 6 | 28.0690 | 24.0451 | 31.4838 | 6.1440 | 6.0106 | 6.0013 | ✓ | gan | candidateB_factorial_levelset_expanded2688 |
| 8 | 29.6351 | 23.2821 | 31.2766 | 7.5096 | 6.0550 | 6.5163 | ✓ | gan | gan |
| 10 | 29.6970 | 24.7100 | 30.9223 | 6.6345 | 7.4238 | 5.9031 |  | gan | candidateB_factorial_levelset_expanded2688 |
| 11 | 28.6873 | 24.9699 | 30.8535 | 5.9107 | 7.8317 | 5.9923 |  | gan | cnn |
| 12 | 30.1045 | 24.2206 | 31.2004 | 6.6164 | 5.9137 | 6.4604 | ✓ | gan | gan |
| 13 | 29.2737 | 24.3237 | 30.2222 | 6.0738 | 7.4023 | 6.5683 |  | gan | cnn |
| 16 | 23.7560 | 17.3245 | 24.5681 | 5.4642 | 5.2552 | 6.1634 | ✓ | gan | gan |
| 17 | 20.9004 | 17.1181 | 22.2776 | 5.2897 | 4.8487 | 5.3329 | ✓ | gan | gan |
| 18 | 20.3949 | 18.6102 | 21.7174 | 6.2586 | 5.5544 | 5.5339 | ✓ | gan | candidateB_factorial_levelset_expanded2688 |
| 19 | 21.0641 | 19.0867 | 21.5670 | 6.9280 | 5.4429 | 5.8555 | ✓ | gan | gan |
| 20 | 21.9792 | 18.2663 | 22.9527 | 6.2541 | 5.6891 | 6.3373 | ✓ | gan | gan |
| 25 | 36.5841 | 30.1235 | 38.5990 | 11.3716 | 10.5015 | 7.1045 | ✓ | gan | candidateB_factorial_levelset_expanded2688 |
| 48 | 22.8325 | 13.1070 | 21.8691 | 7.0815 | 6.6830 | 6.9348 | ✓ | gan | gan |
| 62 | 27.4635 | 17.9937 | 29.0628 | 6.7203 | 6.6502 | 7.2124 | ✓ | gan | gan |
| 63 | 28.5246 | 18.1998 | 29.8659 | 7.6606 | 7.5776 | 7.5398 | ✓ | gan | candidateB_factorial_levelset_expanded2688 |
| 65 | 22.1489 | 16.2395 | 23.2337 | 5.8208 | 5.5768 | 5.7969 | ✓ | gan | gan |
| 68 | 21.7713 | 15.5009 | 22.2386 | 6.0743 | 5.6709 | 5.5914 | ✓ | gan | candidateB_factorial_levelset_expanded2688 |
| 77 | 29.9447 | 20.3533 | 31.9048 | 6.1320 | 6.0000 | 5.9805 | ✓ | gan | candidateB_factorial_levelset_expanded2688 |
| 79 | 30.1481 | 20.3212 | 31.5582 | 6.8565 | 6.4670 | 6.6807 | ✓ | gan | gan |
| 80 | 30.0863 | 20.7449 | 31.5255 | 6.9827 | 6.1767 | 6.5562 | ✓ | gan | gan |
| 82 | 28.9009 | 20.8163 | 31.2110 | 7.1841 | 6.3036 | 6.2746 | ✓ | gan | candidateB_factorial_levelset_expanded2688 |
| 90 | 22.2068 | 17.3268 | 24.1329 | 5.2579 | 5.6635 | 5.4807 |  | gan | cnn |
| 91 | 22.3094 | 16.2661 | 24.4595 | 5.5083 | 5.5198 | 5.7771 |  | gan | cnn |
| 92 | 22.0793 | 16.8077 | 25.1134 | 5.7358 | 5.6779 | 5.3820 | ✓ | gan | candidateB_factorial_levelset_expanded2688 |
| 93 | 24.4029 | 18.1355 | 28.6427 | 6.5070 | 6.8160 | 6.0761 |  | gan | candidateB_factorial_levelset_expanded2688 |
| 154 | 25.3102 | 16.8345 | 24.6630 | 5.8898 | 5.7368 | 6.3909 | ✓ | gan | gan |
| 162 | 18.2711 | 19.4579 | 23.3082 | 4.5406 | 9.6957 | 4.5317 |  | cnn | candidateB_factorial_levelset_expanded2688 |
| 163 | 19.3704 | 20.4464 | 24.4548 | 5.6989 | 6.7890 | 5.1335 |  | cnn | candidateB_factorial_levelset_expanded2688 |

## Output files

- `ttk_runs_fixed/topology_finetuning/candidateB_factorial_levelset_expanded2688_topology/candidateB_factorial_levelset_expanded2688_pd_mt_distances.csv` — candidateB_factorial_levelset_expanded2688 PD/MT per sample
- `ttk_runs_fixed/topology_finetuning/candidateB_factorial_levelset_expanded2688_topology/candidateB_factorial_levelset_expanded2688_topology_comparison.csv` — three-way comparison (CNN, GAN, candidateB_factorial_levelset_expanded2688)
- `docs/topology_finetuning_candidateB_factorial_levelset_expanded2688_topology_eval.md` — this report

## Notes

- PD distance = bottleneck distance between persistence diagrams (computed by `ttkBottleneckDistance`).
- MT distance = Wasserstein-type merge tree distance (computed by `ttkMergeTreeDistanceMatrix`).
- Both use 160×160 speed patches at (x0=0, y0=0), consistent with the CNN/GAN baseline.
- **Lower distance = SR is topologically closer to GT. Lower is better for both metrics.**
- **Component-count proxy metrics** (counts of connected components in speed superlevel sets) are reported separately in `topology_finetuning_candidateB_factorial_levelset_expanded2688_eval.md` and do not require TTK.
- candidateB_factorial_levelset_expanded2688 was fine-tuned from the pretrained CNN checkpoint. Results may change with different lambda settings or more epochs.
- Existing CNN/GAN and Candidate B topology outputs were not overwritten.
