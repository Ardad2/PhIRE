# candidateUV_plus_E2_tf_lowlambda_expanded2688 topology evaluation

**Generated:** 2026-07-11

**Important:** This report covers **true topology metrics** (PD bottleneck distance, MT Wasserstein distance) computed by TTK. These are distinct from the component-count *proxy* metrics in `topology_finetuning_candidateUV_plus_E2_tf_lowlambda_expanded2688_eval.md`.

**Note:** Lower PD/MT distance = SR is topologically closer to GT. **Lower is better for both metrics.**

**candidateUV_plus_E2_tf_lowlambda_expanded2688:** fine-tuned from the pretrained CNN checkpoint with auxiliary physics/topology losses. See the corresponding fine-tuning pilot script for configuration details.

**Samples evaluated:** 168

**MT-GAN baseline wins (before candidateUV_plus_E2_tf_lowlambda_expanded2688):** 20 samples — [6, 8, 12, 16, 17, 18, 19, 20, 25, 48, 62, 63, 65, 68, 77, 79, 80, 82, 92, 154]

## Summary statistics

| Metric | CNN (baseline) | GAN (baseline) | candidateUV_plus_E2_tf_lowlambda_expanded2688 |
|---|---|---|---|
| PD distance (mean) | 27.4063 | 20.8641 | 25.0721 |
| MT distance (mean) | 5.8678 | 8.3481 | 5.5940 |
| PD wins (vs CNN) | — | 166 | 160 |
| MT wins (vs CNN) | — | 20 | 104 |
| PD beats GAN | — | — | 2 |
| MT beats GAN | — | — | 166 |

## 6 Key evaluation questions

### Q1. Does candidateUV_plus_E2_tf_lowlambda_expanded2688 improve PD distance relative to baseline CNN?

- Mean PD: CNN=27.4063, candidateUV_plus_E2_tf_lowlambda_expanded2688=25.0721, Δ=-2.3342 (▼ better)
- candidateUV_plus_E2_tf_lowlambda_expanded2688 has lower PD on **160/168** samples.

### Q2. Does candidateUV_plus_E2_tf_lowlambda_expanded2688 ever beat GAN on PD distance?

- candidateUV_plus_E2_tf_lowlambda_expanded2688 beats GAN on PD for **2/168** samples.
- Mean PD: GAN=20.8641, candidateUV_plus_E2_tf_lowlambda_expanded2688=25.0721. Δ=4.2080 (▲ worse than GAN on average)

### Q3. Does candidateUV_plus_E2_tf_lowlambda_expanded2688 improve MT distance relative to baseline CNN?

- Mean MT: CNN=5.8678, candidateUV_plus_E2_tf_lowlambda_expanded2688=5.5940, Δ=-0.2738 (▼ better)
- candidateUV_plus_E2_tf_lowlambda_expanded2688 has lower MT on **104/168** samples.

### Q4. Does candidateUV_plus_E2_tf_lowlambda_expanded2688 change any of the original 20 MT-GAN cases?

The 20 samples where GAN originally beat CNN on MT distance: [6, 8, 12, 16, 17, 18, 19, 20, 25, 48, 62, 63, 65, 68, 77, 79, 80, 82, 92, 154]

| Outcome | Count |
|---|---|
| MT winner stays GAN | 2 |
| MT winner changes to candidateUV_plus_E2_tf_lowlambda_expanded2688 | 18 |
| MT winner changes to CNN | 0 |

Per-sample detail for changed cases:

| sample_idx | MT CNN | MT GAN | MT candidateUV_plus_E2_tf_lowlambda_expanded2688 | winner before | winner after |
|---|---|---|---|---|---|
| 6 | 6.1440 | 6.0106 | 5.1087 | gan | candidateUV_plus_E2_tf_lowlambda_expanded2688 |
| 16 | 5.4642 | 5.2552 | 4.7919 | gan | candidateUV_plus_E2_tf_lowlambda_expanded2688 |
| 17 | 5.2897 | 4.8487 | 4.1197 | gan | candidateUV_plus_E2_tf_lowlambda_expanded2688 |
| 18 | 6.2586 | 5.5544 | 4.5040 | gan | candidateUV_plus_E2_tf_lowlambda_expanded2688 |
| 19 | 6.9280 | 5.4429 | 4.9873 | gan | candidateUV_plus_E2_tf_lowlambda_expanded2688 |
| 20 | 6.2541 | 5.6891 | 5.2393 | gan | candidateUV_plus_E2_tf_lowlambda_expanded2688 |
| 25 | 11.3716 | 10.5015 | 7.1139 | gan | candidateUV_plus_E2_tf_lowlambda_expanded2688 |
| 48 | 7.0815 | 6.6830 | 6.2773 | gan | candidateUV_plus_E2_tf_lowlambda_expanded2688 |
| 62 | 6.7203 | 6.6502 | 5.9619 | gan | candidateUV_plus_E2_tf_lowlambda_expanded2688 |
| 63 | 7.6606 | 7.5776 | 6.4119 | gan | candidateUV_plus_E2_tf_lowlambda_expanded2688 |
| 65 | 5.8208 | 5.5768 | 4.5451 | gan | candidateUV_plus_E2_tf_lowlambda_expanded2688 |
| 68 | 6.0743 | 5.6709 | 4.8568 | gan | candidateUV_plus_E2_tf_lowlambda_expanded2688 |
| 77 | 6.1320 | 6.0000 | 4.9744 | gan | candidateUV_plus_E2_tf_lowlambda_expanded2688 |
| 79 | 6.8565 | 6.4670 | 5.5869 | gan | candidateUV_plus_E2_tf_lowlambda_expanded2688 |
| 80 | 6.9827 | 6.1767 | 5.2942 | gan | candidateUV_plus_E2_tf_lowlambda_expanded2688 |
| 82 | 7.1841 | 6.3036 | 5.4369 | gan | candidateUV_plus_E2_tf_lowlambda_expanded2688 |
| 92 | 5.7358 | 5.6779 | 4.8137 | gan | candidateUV_plus_E2_tf_lowlambda_expanded2688 |
| 154 | 5.8898 | 5.7368 | 5.5100 | gan | candidateUV_plus_E2_tf_lowlambda_expanded2688 |

### Q5. What happens in adjacent clusters (samples 10–13 and 90–93)?

| sample_idx | PD CNN | PD GAN | PD candidateUV_plus_E2_tf_lowlambda_expanded2688 | MT CNN | MT GAN | MT candidateUV_plus_E2_tf_lowlambda_expanded2688 | PD winner → | MT winner → |
|---|---|---|---|---|---|---|---|---|
| 10 | 29.6970 | 24.7100 | 26.8346 | 6.6345 | 7.4238 | 5.3291 | gan → gan | cnn → candidateUV_plus_E2_tf_lowlambda_expanded2688 |
| 11 | 28.6873 | 24.9699 | 25.7832 | 5.9107 | 7.8317 | 6.9597 | gan → gan | cnn → cnn |
| 12 | 30.1045 | 24.2206 | 26.9446 | 6.6164 | 5.9137 | 5.9467 | gan → gan | gan → gan |
| 13 | 29.2737 | 24.3237 | 26.7278 | 6.0738 | 7.4023 | 6.5368 | gan → gan | cnn → cnn |
| 90 | 22.2068 | 17.3268 | 20.0844 | 5.2579 | 5.6635 | 4.9645 | gan → gan | cnn → candidateUV_plus_E2_tf_lowlambda_expanded2688 |
| 91 | 22.3094 | 16.2661 | 19.9851 | 5.5083 | 5.5198 | 5.0156 | gan → gan | cnn → candidateUV_plus_E2_tf_lowlambda_expanded2688 |
| 92 | 22.0793 | 16.8077 | 20.4982 | 5.7358 | 5.6779 | 4.8137 | gan → gan | gan → candidateUV_plus_E2_tf_lowlambda_expanded2688 |
| 93 | 24.4029 | 18.1355 | 23.0757 | 6.5070 | 6.8160 | 6.6908 | gan → gan | cnn → cnn |

Note: samples [12, 92] are in the MT-GAN-wins set (GAN beat CNN on MT in baseline).

### Q6. What happens in rare topology-CNN controls (samples 162 and 163)?

These samples were originally topology-CNN controls (CNN wins on topology metrics). candidateUV_plus_E2_tf_lowlambda_expanded2688 should not reverse this.

| sample_idx | PD CNN | PD GAN | PD candidateUV_plus_E2_tf_lowlambda_expanded2688 | MT CNN | MT GAN | MT candidateUV_plus_E2_tf_lowlambda_expanded2688 | PD winner → | MT winner → |
|---|---|---|---|---|---|---|---|---|
| 162 | 18.2711 | 19.4579 | 19.5478 | 4.5406 | 9.6957 | 4.8230 | cnn → cnn | cnn → cnn |
| 163 | 19.3704 | 20.4464 | 20.3336 | 5.6989 | 6.7890 | 4.9024 | cnn → cnn | cnn → candidateUV_plus_E2_tf_lowlambda_expanded2688 |

## Winner distribution

### PD distance winners

| Method | Wins before candidateUV_plus_E2_tf_lowlambda_expanded2688 | Wins after candidateUV_plus_E2_tf_lowlambda_expanded2688 |
|---|---|---|
| candidateUV_plus_E2_tf_lowlambda_expanded2688 | 0 | 1 |
| cnn | 2 | 2 |
| gan | 166 | 165 |

### MT distance winners

| Method | Wins before candidateUV_plus_E2_tf_lowlambda_expanded2688 | Wins after candidateUV_plus_E2_tf_lowlambda_expanded2688 |
|---|---|---|
| candidateUV_plus_E2_tf_lowlambda_expanded2688 | 0 | 102 |
| cnn | 148 | 64 |
| gan | 20 | 2 |

## Full per-sample detail: focus samples

Includes adjacent-cluster samples, rare controls, and the 20 MT-GAN baseline wins.

| sample | PD CNN | PD GAN | PD candidateUV_plus_E2_tf_lowlambda_expanded2688 | MT CNN | MT GAN | MT candidateUV_plus_E2_tf_lowlambda_expanded2688 | MT-GAN-win? | PD after | MT after |
|---|---|---|---|---|---|---|---|---|---|
| 6 | 28.0690 | 24.0451 | 26.5826 | 6.1440 | 6.0106 | 5.1087 | ✓ | gan | candidateUV_plus_E2_tf_lowlambda_expanded2688 |
| 8 | 29.6351 | 23.2821 | 27.4629 | 7.5096 | 6.0550 | 6.0882 | ✓ | gan | gan |
| 10 | 29.6970 | 24.7100 | 26.8346 | 6.6345 | 7.4238 | 5.3291 |  | gan | candidateUV_plus_E2_tf_lowlambda_expanded2688 |
| 11 | 28.6873 | 24.9699 | 25.7832 | 5.9107 | 7.8317 | 6.9597 |  | gan | cnn |
| 12 | 30.1045 | 24.2206 | 26.9446 | 6.6164 | 5.9137 | 5.9467 | ✓ | gan | gan |
| 13 | 29.2737 | 24.3237 | 26.7278 | 6.0738 | 7.4023 | 6.5368 |  | gan | cnn |
| 16 | 23.7560 | 17.3245 | 21.2498 | 5.4642 | 5.2552 | 4.7919 | ✓ | gan | candidateUV_plus_E2_tf_lowlambda_expanded2688 |
| 17 | 20.9004 | 17.1181 | 19.3687 | 5.2897 | 4.8487 | 4.1197 | ✓ | gan | candidateUV_plus_E2_tf_lowlambda_expanded2688 |
| 18 | 20.3949 | 18.6102 | 19.1385 | 6.2586 | 5.5544 | 4.5040 | ✓ | gan | candidateUV_plus_E2_tf_lowlambda_expanded2688 |
| 19 | 21.0641 | 19.0867 | 19.0611 | 6.9280 | 5.4429 | 4.9873 | ✓ | candidateUV_plus_E2_tf_lowlambda_expanded2688 | candidateUV_plus_E2_tf_lowlambda_expanded2688 |
| 20 | 21.9792 | 18.2663 | 20.6434 | 6.2541 | 5.6891 | 5.2393 | ✓ | gan | candidateUV_plus_E2_tf_lowlambda_expanded2688 |
| 25 | 36.5841 | 30.1235 | 33.4037 | 11.3716 | 10.5015 | 7.1139 | ✓ | gan | candidateUV_plus_E2_tf_lowlambda_expanded2688 |
| 48 | 22.8325 | 13.1070 | 19.2205 | 7.0815 | 6.6830 | 6.2773 | ✓ | gan | candidateUV_plus_E2_tf_lowlambda_expanded2688 |
| 62 | 27.4635 | 17.9937 | 24.5529 | 6.7203 | 6.6502 | 5.9619 | ✓ | gan | candidateUV_plus_E2_tf_lowlambda_expanded2688 |
| 63 | 28.5246 | 18.1998 | 24.7775 | 7.6606 | 7.5776 | 6.4119 | ✓ | gan | candidateUV_plus_E2_tf_lowlambda_expanded2688 |
| 65 | 22.1489 | 16.2395 | 19.5879 | 5.8208 | 5.5768 | 4.5451 | ✓ | gan | candidateUV_plus_E2_tf_lowlambda_expanded2688 |
| 68 | 21.7713 | 15.5009 | 18.4915 | 6.0743 | 5.6709 | 4.8568 | ✓ | gan | candidateUV_plus_E2_tf_lowlambda_expanded2688 |
| 77 | 29.9447 | 20.3533 | 26.2864 | 6.1320 | 6.0000 | 4.9744 | ✓ | gan | candidateUV_plus_E2_tf_lowlambda_expanded2688 |
| 79 | 30.1481 | 20.3212 | 26.8846 | 6.8565 | 6.4670 | 5.5869 | ✓ | gan | candidateUV_plus_E2_tf_lowlambda_expanded2688 |
| 80 | 30.0863 | 20.7449 | 26.6535 | 6.9827 | 6.1767 | 5.2942 | ✓ | gan | candidateUV_plus_E2_tf_lowlambda_expanded2688 |
| 82 | 28.9009 | 20.8163 | 25.7546 | 7.1841 | 6.3036 | 5.4369 | ✓ | gan | candidateUV_plus_E2_tf_lowlambda_expanded2688 |
| 90 | 22.2068 | 17.3268 | 20.0844 | 5.2579 | 5.6635 | 4.9645 |  | gan | candidateUV_plus_E2_tf_lowlambda_expanded2688 |
| 91 | 22.3094 | 16.2661 | 19.9851 | 5.5083 | 5.5198 | 5.0156 |  | gan | candidateUV_plus_E2_tf_lowlambda_expanded2688 |
| 92 | 22.0793 | 16.8077 | 20.4982 | 5.7358 | 5.6779 | 4.8137 | ✓ | gan | candidateUV_plus_E2_tf_lowlambda_expanded2688 |
| 93 | 24.4029 | 18.1355 | 23.0757 | 6.5070 | 6.8160 | 6.6908 |  | gan | cnn |
| 154 | 25.3102 | 16.8345 | 21.5941 | 5.8898 | 5.7368 | 5.5100 | ✓ | gan | candidateUV_plus_E2_tf_lowlambda_expanded2688 |
| 162 | 18.2711 | 19.4579 | 19.5478 | 4.5406 | 9.6957 | 4.8230 |  | cnn | cnn |
| 163 | 19.3704 | 20.4464 | 20.3336 | 5.6989 | 6.7890 | 4.9024 |  | cnn | candidateUV_plus_E2_tf_lowlambda_expanded2688 |

## Output files

- `ttk_runs_fixed/topology_finetuning/candidateUV_plus_E2_tf_lowlambda_expanded2688_topology/candidateUV_plus_E2_tf_lowlambda_expanded2688_pd_mt_distances.csv` — candidateUV_plus_E2_tf_lowlambda_expanded2688 PD/MT per sample
- `ttk_runs_fixed/topology_finetuning/candidateUV_plus_E2_tf_lowlambda_expanded2688_topology/candidateUV_plus_E2_tf_lowlambda_expanded2688_topology_comparison.csv` — three-way comparison (CNN, GAN, candidateUV_plus_E2_tf_lowlambda_expanded2688)
- `docs/topology_finetuning_candidateUV_plus_E2_tf_lowlambda_expanded2688_topology_eval.md` — this report

## Notes

- PD distance = bottleneck distance between persistence diagrams (computed by `ttkBottleneckDistance`).
- MT distance = Wasserstein-type merge tree distance (computed by `ttkMergeTreeDistanceMatrix`).
- Both use 160×160 speed patches at (x0=0, y0=0), consistent with the CNN/GAN baseline.
- **Lower distance = SR is topologically closer to GT. Lower is better for both metrics.**
- **Component-count proxy metrics** (counts of connected components in speed superlevel sets) are reported separately in `topology_finetuning_candidateUV_plus_E2_tf_lowlambda_expanded2688_eval.md` and do not require TTK.
- candidateUV_plus_E2_tf_lowlambda_expanded2688 was fine-tuned from the pretrained CNN checkpoint. Results may change with different lambda settings or more epochs.
- Existing CNN/GAN and Candidate B topology outputs were not overwritten.
