# candidateF_grad_levelset_E2_low_expanded2688 topology evaluation

**Generated:** 2026-07-19

**Important:** This report covers **true topology metrics** (PD bottleneck distance, MT Wasserstein distance) computed by TTK. These are distinct from the component-count *proxy* metrics in `topology_finetuning_candidateF_grad_levelset_E2_low_expanded2688_eval.md`.

**Note:** Lower PD/MT distance = SR is topologically closer to GT. **Lower is better for both metrics.**

**candidateF_grad_levelset_E2_low_expanded2688:** fine-tuned from the pretrained CNN checkpoint with auxiliary physics/topology losses. See the corresponding fine-tuning pilot script for configuration details.

**Samples evaluated:** 168

**MT-GAN baseline wins (before candidateF_grad_levelset_E2_low_expanded2688):** 20 samples — [6, 8, 12, 16, 17, 18, 19, 20, 25, 48, 62, 63, 65, 68, 77, 79, 80, 82, 92, 154]

## Summary statistics

| Metric | CNN (baseline) | GAN (baseline) | candidateF_grad_levelset_E2_low_expanded2688 |
|---|---|---|---|
| PD distance (mean) | 27.4063 | 20.8641 | 23.7481 |
| MT distance (mean) | 5.8678 | 8.3481 | 5.6742 |
| PD wins (vs CNN) | — | 166 | 166 |
| MT wins (vs CNN) | — | 20 | 102 |
| PD beats GAN | — | — | 12 |
| MT beats GAN | — | — | 162 |

## 6 Key evaluation questions

### Q1. Does candidateF_grad_levelset_E2_low_expanded2688 improve PD distance relative to baseline CNN?

- Mean PD: CNN=27.4063, candidateF_grad_levelset_E2_low_expanded2688=23.7481, Δ=-3.6582 (▼ better)
- candidateF_grad_levelset_E2_low_expanded2688 has lower PD on **166/168** samples.

### Q2. Does candidateF_grad_levelset_E2_low_expanded2688 ever beat GAN on PD distance?

- candidateF_grad_levelset_E2_low_expanded2688 beats GAN on PD for **12/168** samples.
- Mean PD: GAN=20.8641, candidateF_grad_levelset_E2_low_expanded2688=23.7481. Δ=2.8840 (▲ worse than GAN on average)

### Q3. Does candidateF_grad_levelset_E2_low_expanded2688 improve MT distance relative to baseline CNN?

- Mean MT: CNN=5.8678, candidateF_grad_levelset_E2_low_expanded2688=5.6742, Δ=-0.1936 (▼ better)
- candidateF_grad_levelset_E2_low_expanded2688 has lower MT on **102/168** samples.

### Q4. Does candidateF_grad_levelset_E2_low_expanded2688 change any of the original 20 MT-GAN cases?

The 20 samples where GAN originally beat CNN on MT distance: [6, 8, 12, 16, 17, 18, 19, 20, 25, 48, 62, 63, 65, 68, 77, 79, 80, 82, 92, 154]

| Outcome | Count |
|---|---|
| MT winner stays GAN | 4 |
| MT winner changes to candidateF_grad_levelset_E2_low_expanded2688 | 16 |
| MT winner changes to CNN | 0 |

Per-sample detail for changed cases:

| sample_idx | MT CNN | MT GAN | MT candidateF_grad_levelset_E2_low_expanded2688 | winner before | winner after |
|---|---|---|---|---|---|
| 6 | 6.1440 | 6.0106 | 5.0085 | gan | candidateF_grad_levelset_E2_low_expanded2688 |
| 16 | 5.4642 | 5.2552 | 4.6546 | gan | candidateF_grad_levelset_E2_low_expanded2688 |
| 17 | 5.2897 | 4.8487 | 4.0334 | gan | candidateF_grad_levelset_E2_low_expanded2688 |
| 18 | 6.2586 | 5.5544 | 4.3716 | gan | candidateF_grad_levelset_E2_low_expanded2688 |
| 19 | 6.9280 | 5.4429 | 5.3123 | gan | candidateF_grad_levelset_E2_low_expanded2688 |
| 20 | 6.2541 | 5.6891 | 5.2453 | gan | candidateF_grad_levelset_E2_low_expanded2688 |
| 25 | 11.3716 | 10.5015 | 6.6388 | gan | candidateF_grad_levelset_E2_low_expanded2688 |
| 62 | 6.7203 | 6.6502 | 5.6356 | gan | candidateF_grad_levelset_E2_low_expanded2688 |
| 63 | 7.6606 | 7.5776 | 6.1309 | gan | candidateF_grad_levelset_E2_low_expanded2688 |
| 65 | 5.8208 | 5.5768 | 5.4165 | gan | candidateF_grad_levelset_E2_low_expanded2688 |
| 68 | 6.0743 | 5.6709 | 5.0282 | gan | candidateF_grad_levelset_E2_low_expanded2688 |
| 77 | 6.1320 | 6.0000 | 5.0930 | gan | candidateF_grad_levelset_E2_low_expanded2688 |
| 79 | 6.8565 | 6.4670 | 5.4241 | gan | candidateF_grad_levelset_E2_low_expanded2688 |
| 80 | 6.9827 | 6.1767 | 5.2464 | gan | candidateF_grad_levelset_E2_low_expanded2688 |
| 82 | 7.1841 | 6.3036 | 5.5453 | gan | candidateF_grad_levelset_E2_low_expanded2688 |
| 92 | 5.7358 | 5.6779 | 5.0129 | gan | candidateF_grad_levelset_E2_low_expanded2688 |

### Q5. What happens in adjacent clusters (samples 10–13 and 90–93)?

| sample_idx | PD CNN | PD GAN | PD candidateF_grad_levelset_E2_low_expanded2688 | MT CNN | MT GAN | MT candidateF_grad_levelset_E2_low_expanded2688 | PD winner → | MT winner → |
|---|---|---|---|---|---|---|---|---|
| 10 | 29.6970 | 24.7100 | 25.4602 | 6.6345 | 7.4238 | 5.7957 | gan → gan | cnn → candidateF_grad_levelset_E2_low_expanded2688 |
| 11 | 28.6873 | 24.9699 | 24.2285 | 5.9107 | 7.8317 | 6.0552 | gan → candidateF_grad_levelset_E2_low_expanded2688 | cnn → cnn |
| 12 | 30.1045 | 24.2206 | 25.5062 | 6.6164 | 5.9137 | 6.0404 | gan → gan | gan → gan |
| 13 | 29.2737 | 24.3237 | 25.5214 | 6.0738 | 7.4023 | 6.8162 | gan → gan | cnn → cnn |
| 90 | 22.2068 | 17.3268 | 19.0785 | 5.2579 | 5.6635 | 4.8256 | gan → gan | cnn → candidateF_grad_levelset_E2_low_expanded2688 |
| 91 | 22.3094 | 16.2661 | 18.7301 | 5.5083 | 5.5198 | 5.4759 | gan → gan | cnn → candidateF_grad_levelset_E2_low_expanded2688 |
| 92 | 22.0793 | 16.8077 | 18.8840 | 5.7358 | 5.6779 | 5.0129 | gan → gan | gan → candidateF_grad_levelset_E2_low_expanded2688 |
| 93 | 24.4029 | 18.1355 | 21.1414 | 6.5070 | 6.8160 | 6.7043 | gan → gan | cnn → cnn |

Note: samples [12, 92] are in the MT-GAN-wins set (GAN beat CNN on MT in baseline).

### Q6. What happens in rare topology-CNN controls (samples 162 and 163)?

These samples were originally topology-CNN controls (CNN wins on topology metrics). candidateF_grad_levelset_E2_low_expanded2688 should not reverse this.

| sample_idx | PD CNN | PD GAN | PD candidateF_grad_levelset_E2_low_expanded2688 | MT CNN | MT GAN | MT candidateF_grad_levelset_E2_low_expanded2688 | PD winner → | MT winner → |
|---|---|---|---|---|---|---|---|---|
| 162 | 18.2711 | 19.4579 | 18.5500 | 4.5406 | 9.6957 | 4.4856 | cnn → cnn | cnn → candidateF_grad_levelset_E2_low_expanded2688 |
| 163 | 19.3704 | 20.4464 | 19.6533 | 5.6989 | 6.7890 | 4.9738 | cnn → cnn | cnn → candidateF_grad_levelset_E2_low_expanded2688 |

## Winner distribution

### PD distance winners

| Method | Wins before candidateF_grad_levelset_E2_low_expanded2688 | Wins after candidateF_grad_levelset_E2_low_expanded2688 |
|---|---|---|
| candidateF_grad_levelset_E2_low_expanded2688 | 0 | 10 |
| cnn | 2 | 2 |
| gan | 166 | 156 |

### MT distance winners

| Method | Wins before candidateF_grad_levelset_E2_low_expanded2688 | Wins after candidateF_grad_levelset_E2_low_expanded2688 |
|---|---|---|
| candidateF_grad_levelset_E2_low_expanded2688 | 0 | 100 |
| cnn | 148 | 64 |
| gan | 20 | 4 |

## Full per-sample detail: focus samples

Includes adjacent-cluster samples, rare controls, and the 20 MT-GAN baseline wins.

| sample | PD CNN | PD GAN | PD candidateF_grad_levelset_E2_low_expanded2688 | MT CNN | MT GAN | MT candidateF_grad_levelset_E2_low_expanded2688 | MT-GAN-win? | PD after | MT after |
|---|---|---|---|---|---|---|---|---|---|
| 6 | 28.0690 | 24.0451 | 25.3062 | 6.1440 | 6.0106 | 5.0085 | ✓ | gan | candidateF_grad_levelset_E2_low_expanded2688 |
| 8 | 29.6351 | 23.2821 | 25.8523 | 7.5096 | 6.0550 | 6.0610 | ✓ | gan | gan |
| 10 | 29.6970 | 24.7100 | 25.4602 | 6.6345 | 7.4238 | 5.7957 |  | gan | candidateF_grad_levelset_E2_low_expanded2688 |
| 11 | 28.6873 | 24.9699 | 24.2285 | 5.9107 | 7.8317 | 6.0552 |  | candidateF_grad_levelset_E2_low_expanded2688 | cnn |
| 12 | 30.1045 | 24.2206 | 25.5062 | 6.6164 | 5.9137 | 6.0404 | ✓ | gan | gan |
| 13 | 29.2737 | 24.3237 | 25.5214 | 6.0738 | 7.4023 | 6.8162 |  | gan | cnn |
| 16 | 23.7560 | 17.3245 | 20.2053 | 5.4642 | 5.2552 | 4.6546 | ✓ | gan | candidateF_grad_levelset_E2_low_expanded2688 |
| 17 | 20.9004 | 17.1181 | 18.2422 | 5.2897 | 4.8487 | 4.0334 | ✓ | gan | candidateF_grad_levelset_E2_low_expanded2688 |
| 18 | 20.3949 | 18.6102 | 17.7847 | 6.2586 | 5.5544 | 4.3716 | ✓ | candidateF_grad_levelset_E2_low_expanded2688 | candidateF_grad_levelset_E2_low_expanded2688 |
| 19 | 21.0641 | 19.0867 | 17.5144 | 6.9280 | 5.4429 | 5.3123 | ✓ | candidateF_grad_levelset_E2_low_expanded2688 | candidateF_grad_levelset_E2_low_expanded2688 |
| 20 | 21.9792 | 18.2663 | 18.8511 | 6.2541 | 5.6891 | 5.2453 | ✓ | gan | candidateF_grad_levelset_E2_low_expanded2688 |
| 25 | 36.5841 | 30.1235 | 31.4780 | 11.3716 | 10.5015 | 6.6388 | ✓ | gan | candidateF_grad_levelset_E2_low_expanded2688 |
| 48 | 22.8325 | 13.1070 | 16.8979 | 7.0815 | 6.6830 | 8.4088 | ✓ | gan | gan |
| 62 | 27.4635 | 17.9937 | 23.7173 | 6.7203 | 6.6502 | 5.6356 | ✓ | gan | candidateF_grad_levelset_E2_low_expanded2688 |
| 63 | 28.5246 | 18.1998 | 24.0193 | 7.6606 | 7.5776 | 6.1309 | ✓ | gan | candidateF_grad_levelset_E2_low_expanded2688 |
| 65 | 22.1489 | 16.2395 | 18.8694 | 5.8208 | 5.5768 | 5.4165 | ✓ | gan | candidateF_grad_levelset_E2_low_expanded2688 |
| 68 | 21.7713 | 15.5009 | 17.5381 | 6.0743 | 5.6709 | 5.0282 | ✓ | gan | candidateF_grad_levelset_E2_low_expanded2688 |
| 77 | 29.9447 | 20.3533 | 25.4350 | 6.1320 | 6.0000 | 5.0930 | ✓ | gan | candidateF_grad_levelset_E2_low_expanded2688 |
| 79 | 30.1481 | 20.3212 | 26.0477 | 6.8565 | 6.4670 | 5.4241 | ✓ | gan | candidateF_grad_levelset_E2_low_expanded2688 |
| 80 | 30.0863 | 20.7449 | 25.7768 | 6.9827 | 6.1767 | 5.2464 | ✓ | gan | candidateF_grad_levelset_E2_low_expanded2688 |
| 82 | 28.9009 | 20.8163 | 25.1997 | 7.1841 | 6.3036 | 5.5453 | ✓ | gan | candidateF_grad_levelset_E2_low_expanded2688 |
| 90 | 22.2068 | 17.3268 | 19.0785 | 5.2579 | 5.6635 | 4.8256 |  | gan | candidateF_grad_levelset_E2_low_expanded2688 |
| 91 | 22.3094 | 16.2661 | 18.7301 | 5.5083 | 5.5198 | 5.4759 |  | gan | candidateF_grad_levelset_E2_low_expanded2688 |
| 92 | 22.0793 | 16.8077 | 18.8840 | 5.7358 | 5.6779 | 5.0129 | ✓ | gan | candidateF_grad_levelset_E2_low_expanded2688 |
| 93 | 24.4029 | 18.1355 | 21.1414 | 6.5070 | 6.8160 | 6.7043 |  | gan | cnn |
| 154 | 25.3102 | 16.8345 | 19.7844 | 5.8898 | 5.7368 | 5.8994 | ✓ | gan | gan |
| 162 | 18.2711 | 19.4579 | 18.5500 | 4.5406 | 9.6957 | 4.4856 |  | cnn | candidateF_grad_levelset_E2_low_expanded2688 |
| 163 | 19.3704 | 20.4464 | 19.6533 | 5.6989 | 6.7890 | 4.9738 |  | cnn | candidateF_grad_levelset_E2_low_expanded2688 |

## Output files

- `ttk_runs_fixed/topology_finetuning/candidateF_grad_levelset_E2_low_expanded2688_topology/candidateF_grad_levelset_E2_low_expanded2688_pd_mt_distances.csv` — candidateF_grad_levelset_E2_low_expanded2688 PD/MT per sample
- `ttk_runs_fixed/topology_finetuning/candidateF_grad_levelset_E2_low_expanded2688_topology/candidateF_grad_levelset_E2_low_expanded2688_topology_comparison.csv` — three-way comparison (CNN, GAN, candidateF_grad_levelset_E2_low_expanded2688)
- `docs/topology_finetuning_candidateF_grad_levelset_E2_low_expanded2688_topology_eval.md` — this report

## Notes

- PD distance = bottleneck distance between persistence diagrams (computed by `ttkBottleneckDistance`).
- MT distance = Wasserstein-type merge tree distance (computed by `ttkMergeTreeDistanceMatrix`).
- Both use 160×160 speed patches at (x0=0, y0=0), consistent with the CNN/GAN baseline.
- **Lower distance = SR is topologically closer to GT. Lower is better for both metrics.**
- **Component-count proxy metrics** (counts of connected components in speed superlevel sets) are reported separately in `topology_finetuning_candidateF_grad_levelset_E2_low_expanded2688_eval.md` and do not require TTK.
- candidateF_grad_levelset_E2_low_expanded2688 was fine-tuned from the pretrained CNN checkpoint. Results may change with different lambda settings or more epochs.
- Existing CNN/GAN and Candidate B topology outputs were not overwritten.
