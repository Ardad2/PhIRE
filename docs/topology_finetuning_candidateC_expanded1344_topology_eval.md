# candidateC_expanded1344 topology evaluation

**Generated:** 2026-05-29

**Important:** This report covers **true topology metrics** (PD bottleneck distance, MT Wasserstein distance) computed by TTK. These are distinct from the component-count *proxy* metrics in `topology_finetuning_candidateC_expanded1344_eval.md`.

**Note:** Lower PD/MT distance = SR is topologically closer to GT. **Lower is better for both metrics.**

**candidateC_expanded1344:** fine-tuned from the pretrained CNN checkpoint with auxiliary physics/topology losses. See the corresponding fine-tuning pilot script for configuration details.

**Candidate C = Candidate B + critical-value/topological-extrema proxy loss** (`lambda_crit=0.001`). Candidate B adds `lambda_speed=0.01, lambda_grad=0.05, lambda_levelset=0.25` on top of the baseline CNN MSE loss.

**Samples evaluated:** 168

**MT-GAN baseline wins (before candidateC_expanded1344):** 20 samples — [6, 8, 12, 16, 17, 18, 19, 20, 25, 48, 62, 63, 65, 68, 77, 79, 80, 82, 92, 154]

## Summary statistics

| Metric | CNN (baseline) | GAN (baseline) | candidateC_expanded1344 |
|---|---|---|---|
| PD distance (mean) | 27.4063 | 20.8641 | 22.8623 |
| MT distance (mean) | 5.8678 | 8.3481 | 6.1236 |
| PD wins (vs CNN) | — | 166 | 168 |
| MT wins (vs CNN) | — | 20 | 49 |
| PD beats GAN | — | — | 17 |
| MT beats GAN | — | — | 149 |

## 6 Key evaluation questions

### Q1. Does candidateC_expanded1344 improve PD distance relative to baseline CNN?

- Mean PD: CNN=27.4063, candidateC_expanded1344=22.8623, Δ=-4.5440 (▼ better)
- candidateC_expanded1344 has lower PD on **168/168** samples.

### Q2. Does candidateC_expanded1344 ever beat GAN on PD distance?

- candidateC_expanded1344 beats GAN on PD for **17/168** samples.
- Mean PD: GAN=20.8641, candidateC_expanded1344=22.8623. Δ=1.9982 (▲ worse than GAN on average)

### Q3. Does candidateC_expanded1344 improve MT distance relative to baseline CNN?

- Mean MT: CNN=5.8678, candidateC_expanded1344=6.1236, Δ=0.2558 (▲ worse)
- candidateC_expanded1344 has lower MT on **49/168** samples.

### Q4. Does candidateC_expanded1344 change any of the original 20 MT-GAN cases?

The 20 samples where GAN originally beat CNN on MT distance: [6, 8, 12, 16, 17, 18, 19, 20, 25, 48, 62, 63, 65, 68, 77, 79, 80, 82, 92, 154]

| Outcome | Count |
|---|---|
| MT winner stays GAN | 11 |
| MT winner changes to candidateC_expanded1344 | 9 |
| MT winner changes to CNN | 0 |

Per-sample detail for changed cases:

| sample_idx | MT CNN | MT GAN | MT candidateC_expanded1344 | winner before | winner after |
|---|---|---|---|---|---|
| 6 | 6.1440 | 6.0106 | 5.7978 | gan | candidateC_expanded1344 |
| 18 | 6.2586 | 5.5544 | 4.7988 | gan | candidateC_expanded1344 |
| 25 | 11.3716 | 10.5015 | 7.1624 | gan | candidateC_expanded1344 |
| 62 | 6.7203 | 6.6502 | 6.6479 | gan | candidateC_expanded1344 |
| 68 | 6.0743 | 5.6709 | 5.4036 | gan | candidateC_expanded1344 |
| 77 | 6.1320 | 6.0000 | 5.5411 | gan | candidateC_expanded1344 |
| 79 | 6.8565 | 6.4670 | 6.0244 | gan | candidateC_expanded1344 |
| 80 | 6.9827 | 6.1767 | 5.8025 | gan | candidateC_expanded1344 |
| 82 | 7.1841 | 6.3036 | 5.9932 | gan | candidateC_expanded1344 |

### Q5. What happens in adjacent clusters (samples 10–13 and 90–93)?

| sample_idx | PD CNN | PD GAN | PD candidateC_expanded1344 | MT CNN | MT GAN | MT candidateC_expanded1344 | PD winner → | MT winner → |
|---|---|---|---|---|---|---|---|---|
| 10 | 29.6970 | 24.7100 | 24.1715 | 6.6345 | 7.4238 | 6.3030 | gan → candidateC_expanded1344 | cnn → candidateC_expanded1344 |
| 11 | 28.6873 | 24.9699 | 24.0780 | 5.9107 | 7.8317 | 6.6358 | gan → candidateC_expanded1344 | cnn → cnn |
| 12 | 30.1045 | 24.2206 | 24.0603 | 6.6164 | 5.9137 | 6.5719 | gan → candidateC_expanded1344 | gan → gan |
| 13 | 29.2737 | 24.3237 | 23.7866 | 6.0738 | 7.4023 | 6.8676 | gan → candidateC_expanded1344 | cnn → cnn |
| 90 | 22.2068 | 17.3268 | 18.1088 | 5.2579 | 5.6635 | 5.6002 | gan → gan | cnn → cnn |
| 91 | 22.3094 | 16.2661 | 17.3555 | 5.5083 | 5.5198 | 6.6357 | gan → gan | cnn → cnn |
| 92 | 22.0793 | 16.8077 | 17.6515 | 5.7358 | 5.6779 | 5.7426 | gan → gan | gan → gan |
| 93 | 24.4029 | 18.1355 | 20.1678 | 6.5070 | 6.8160 | 7.4779 | gan → gan | cnn → cnn |

Note: samples [12, 92] are in the MT-GAN-wins set (GAN beat CNN on MT in baseline).

### Q6. What happens in rare topology-CNN controls (samples 162 and 163)?

These samples were originally topology-CNN controls (CNN wins on topology metrics). candidateC_expanded1344 should not reverse this.

| sample_idx | PD CNN | PD GAN | PD candidateC_expanded1344 | MT CNN | MT GAN | MT candidateC_expanded1344 | PD winner → | MT winner → |
|---|---|---|---|---|---|---|---|---|
| 162 | 18.2711 | 19.4579 | 17.6570 | 4.5406 | 9.6957 | 4.8777 | cnn → candidateC_expanded1344 | cnn → cnn |
| 163 | 19.3704 | 20.4464 | 19.0264 | 5.6989 | 6.7890 | 5.4062 | cnn → candidateC_expanded1344 | cnn → candidateC_expanded1344 |

## Winner distribution

### PD distance winners

| Method | Wins before candidateC_expanded1344 | Wins after candidateC_expanded1344 |
|---|---|---|
| candidateC_expanded1344 | 0 | 17 |
| cnn | 2 | 0 |
| gan | 166 | 151 |

### MT distance winners

| Method | Wins before candidateC_expanded1344 | Wins after candidateC_expanded1344 |
|---|---|---|
| candidateC_expanded1344 | 0 | 43 |
| cnn | 148 | 114 |
| gan | 20 | 11 |

## Full per-sample detail: focus samples

Includes adjacent-cluster samples, rare controls, and the 20 MT-GAN baseline wins.

| sample | PD CNN | PD GAN | PD candidateC_expanded1344 | MT CNN | MT GAN | MT candidateC_expanded1344 | MT-GAN-win? | PD after | MT after |
|---|---|---|---|---|---|---|---|---|---|
| 6 | 28.0690 | 24.0451 | 24.1830 | 6.1440 | 6.0106 | 5.7978 | ✓ | gan | candidateC_expanded1344 |
| 8 | 29.6351 | 23.2821 | 24.6378 | 7.5096 | 6.0550 | 6.4821 | ✓ | gan | gan |
| 10 | 29.6970 | 24.7100 | 24.1715 | 6.6345 | 7.4238 | 6.3030 |  | candidateC_expanded1344 | candidateC_expanded1344 |
| 11 | 28.6873 | 24.9699 | 24.0780 | 5.9107 | 7.8317 | 6.6358 |  | candidateC_expanded1344 | cnn |
| 12 | 30.1045 | 24.2206 | 24.0603 | 6.6164 | 5.9137 | 6.5719 | ✓ | candidateC_expanded1344 | gan |
| 13 | 29.2737 | 24.3237 | 23.7866 | 6.0738 | 7.4023 | 6.8676 |  | candidateC_expanded1344 | cnn |
| 16 | 23.7560 | 17.3245 | 20.0794 | 5.4642 | 5.2552 | 5.8626 | ✓ | gan | gan |
| 17 | 20.9004 | 17.1181 | 17.3634 | 5.2897 | 4.8487 | 5.0556 | ✓ | gan | gan |
| 18 | 20.3949 | 18.6102 | 16.9872 | 6.2586 | 5.5544 | 4.7988 | ✓ | candidateC_expanded1344 | candidateC_expanded1344 |
| 19 | 21.0641 | 19.0867 | 16.5221 | 6.9280 | 5.4429 | 5.8209 | ✓ | candidateC_expanded1344 | gan |
| 20 | 21.9792 | 18.2663 | 17.5254 | 6.2541 | 5.6891 | 6.3278 | ✓ | candidateC_expanded1344 | gan |
| 25 | 36.5841 | 30.1235 | 30.8199 | 11.3716 | 10.5015 | 7.1624 | ✓ | gan | candidateC_expanded1344 |
| 48 | 22.8325 | 13.1070 | 16.2665 | 7.0815 | 6.6830 | 6.8468 | ✓ | gan | gan |
| 62 | 27.4635 | 17.9937 | 23.4813 | 6.7203 | 6.6502 | 6.6479 | ✓ | gan | candidateC_expanded1344 |
| 63 | 28.5246 | 18.1998 | 24.3063 | 7.6606 | 7.5776 | 7.6514 | ✓ | gan | gan |
| 65 | 22.1489 | 16.2395 | 18.7484 | 5.8208 | 5.5768 | 5.9487 | ✓ | gan | gan |
| 68 | 21.7713 | 15.5009 | 16.6349 | 6.0743 | 5.6709 | 5.4036 | ✓ | gan | candidateC_expanded1344 |
| 77 | 29.9447 | 20.3533 | 25.2449 | 6.1320 | 6.0000 | 5.5411 | ✓ | gan | candidateC_expanded1344 |
| 79 | 30.1481 | 20.3212 | 26.0665 | 6.8565 | 6.4670 | 6.0244 | ✓ | gan | candidateC_expanded1344 |
| 80 | 30.0863 | 20.7449 | 25.6654 | 6.9827 | 6.1767 | 5.8025 | ✓ | gan | candidateC_expanded1344 |
| 82 | 28.9009 | 20.8163 | 25.4618 | 7.1841 | 6.3036 | 5.9932 | ✓ | gan | candidateC_expanded1344 |
| 90 | 22.2068 | 17.3268 | 18.1088 | 5.2579 | 5.6635 | 5.6002 |  | gan | cnn |
| 91 | 22.3094 | 16.2661 | 17.3555 | 5.5083 | 5.5198 | 6.6357 |  | gan | cnn |
| 92 | 22.0793 | 16.8077 | 17.6515 | 5.7358 | 5.6779 | 5.7426 | ✓ | gan | gan |
| 93 | 24.4029 | 18.1355 | 20.1678 | 6.5070 | 6.8160 | 7.4779 |  | gan | cnn |
| 154 | 25.3102 | 16.8345 | 19.5640 | 5.8898 | 5.7368 | 6.6427 | ✓ | gan | gan |
| 162 | 18.2711 | 19.4579 | 17.6570 | 4.5406 | 9.6957 | 4.8777 |  | candidateC_expanded1344 | cnn |
| 163 | 19.3704 | 20.4464 | 19.0264 | 5.6989 | 6.7890 | 5.4062 |  | candidateC_expanded1344 | candidateC_expanded1344 |

## Output files

- `ttk_runs_fixed/topology_finetuning/candidateC_expanded1344_topology/candidateC_expanded1344_pd_mt_distances.csv` — candidateC_expanded1344 PD/MT per sample
- `ttk_runs_fixed/topology_finetuning/candidateC_expanded1344_topology/candidateC_expanded1344_topology_comparison.csv` — three-way comparison (CNN, GAN, candidateC_expanded1344)
- `docs/topology_finetuning_candidateC_expanded1344_topology_eval.md` — this report

## Notes

- PD distance = bottleneck distance between persistence diagrams (computed by `ttkBottleneckDistance`).
- MT distance = Wasserstein-type merge tree distance (computed by `ttkMergeTreeDistanceMatrix`).
- Both use 160×160 speed patches at (x0=0, y0=0), consistent with the CNN/GAN baseline.
- **Lower distance = SR is topologically closer to GT. Lower is better for both metrics.**
- **Component-count proxy metrics** (counts of connected components in speed superlevel sets) are reported separately in `topology_finetuning_candidateC_expanded1344_eval.md` and do not require TTK.
- candidateC_expanded1344 was fine-tuned from the pretrained CNN checkpoint. Results may change with different lambda settings or more epochs.
- Existing CNN/GAN and Candidate B topology outputs were not overwritten.
