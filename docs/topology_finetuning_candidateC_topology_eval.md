# candidateC topology evaluation

**Generated:** 2026-05-12

**Important:** This report covers **true topology metrics** (PD bottleneck distance, MT Wasserstein distance) computed by TTK. These are distinct from the component-count *proxy* metrics in `topology_finetuning_candidateC_eval.md`.

**Note:** Lower PD/MT distance = SR is topologically closer to GT. **Lower is better for both metrics.**

**candidateC:** fine-tuned from the pretrained CNN checkpoint with auxiliary physics/topology losses. See the corresponding fine-tuning pilot script for configuration details.

**Candidate C = Candidate B + critical-value/topological-extrema proxy loss** (`lambda_crit=0.001`). Candidate B adds `lambda_speed=0.01, lambda_grad=0.05, lambda_levelset=0.25` on top of the baseline CNN MSE loss.

**Samples evaluated:** 168

**MT-GAN baseline wins (before candidateC):** 20 samples — [6, 8, 12, 16, 17, 18, 19, 20, 25, 48, 62, 63, 65, 68, 77, 79, 80, 82, 92, 154]

## Summary statistics

| Metric | CNN (baseline) | GAN (baseline) | candidateC |
|---|---|---|---|
| PD distance (mean) | 27.4063 | 20.8641 | 27.0021 |
| MT distance (mean) | 5.8678 | 8.3481 | 5.7141 |
| PD wins (vs CNN) | — | 166 | 120 |
| MT wins (vs CNN) | — | 20 | 102 |
| PD beats GAN | — | — | 0 |
| MT beats GAN | — | — | 157 |

## 6 Key evaluation questions

### Q1. Does candidateC improve PD distance relative to baseline CNN?

- Mean PD: CNN=27.4063, candidateC=27.0021, Δ=-0.4042 (▼ better)
- candidateC has lower PD on **120/168** samples.

### Q2. Does candidateC ever beat GAN on PD distance?

- candidateC beats GAN on PD for **0/168** samples.
- Mean PD: GAN=20.8641, candidateC=27.0021. Δ=6.1380 (▲ worse than GAN on average)

### Q3. Does candidateC improve MT distance relative to baseline CNN?

- Mean MT: CNN=5.8678, candidateC=5.7141, Δ=-0.1537 (▼ better)
- candidateC has lower MT on **102/168** samples.

### Q4. Does candidateC change any of the original 20 MT-GAN cases?

The 20 samples where GAN originally beat CNN on MT distance: [6, 8, 12, 16, 17, 18, 19, 20, 25, 48, 62, 63, 65, 68, 77, 79, 80, 82, 92, 154]

| Outcome | Count |
|---|---|
| MT winner stays GAN | 9 |
| MT winner changes to candidateC | 11 |
| MT winner changes to CNN | 0 |

Per-sample detail for changed cases:

| sample_idx | MT CNN | MT GAN | MT candidateC | winner before | winner after |
|---|---|---|---|---|---|
| 6 | 6.1440 | 6.0106 | 5.7881 | gan | candidateC |
| 18 | 6.2586 | 5.5544 | 5.2225 | gan | candidateC |
| 20 | 6.2541 | 5.6891 | 5.4852 | gan | candidateC |
| 25 | 11.3716 | 10.5015 | 6.5417 | gan | candidateC |
| 62 | 6.7203 | 6.6502 | 6.5421 | gan | candidateC |
| 63 | 7.6606 | 7.5776 | 6.5270 | gan | candidateC |
| 65 | 5.8208 | 5.5768 | 5.2142 | gan | candidateC |
| 68 | 6.0743 | 5.6709 | 5.5099 | gan | candidateC |
| 79 | 6.8565 | 6.4670 | 6.2650 | gan | candidateC |
| 80 | 6.9827 | 6.1767 | 5.2586 | gan | candidateC |
| 92 | 5.7358 | 5.6779 | 5.2720 | gan | candidateC |

### Q5. What happens in adjacent clusters (samples 10–13 and 90–93)?

| sample_idx | PD CNN | PD GAN | PD candidateC | MT CNN | MT GAN | MT candidateC | PD winner → | MT winner → |
|---|---|---|---|---|---|---|---|---|
| 10 | 29.6970 | 24.7100 | 28.8943 | 6.6345 | 7.4238 | 6.0843 | gan → gan | cnn → candidateC |
| 11 | 28.6873 | 24.9699 | 28.3733 | 5.9107 | 7.8317 | 5.8242 | gan → gan | cnn → candidateC |
| 12 | 30.1045 | 24.2206 | 29.3527 | 6.6164 | 5.9137 | 6.1039 | gan → gan | gan → gan |
| 13 | 29.2737 | 24.3237 | 28.7161 | 6.0738 | 7.4023 | 6.6037 | gan → gan | cnn → cnn |
| 90 | 22.2068 | 17.3268 | 20.9458 | 5.2579 | 5.6635 | 5.0751 | gan → gan | cnn → candidateC |
| 91 | 22.3094 | 16.2661 | 21.1217 | 5.5083 | 5.5198 | 5.7392 | gan → gan | cnn → cnn |
| 92 | 22.0793 | 16.8077 | 21.0387 | 5.7358 | 5.6779 | 5.2720 | gan → gan | gan → candidateC |
| 93 | 24.4029 | 18.1355 | 24.0283 | 6.5070 | 6.8160 | 6.1437 | gan → gan | cnn → candidateC |

Note: samples [12, 92] are in the MT-GAN-wins set (GAN beat CNN on MT in baseline).

### Q6. What happens in rare topology-CNN controls (samples 162 and 163)?

These samples were originally topology-CNN controls (CNN wins on topology metrics). candidateC should not reverse this.

| sample_idx | PD CNN | PD GAN | PD candidateC | MT CNN | MT GAN | MT candidateC | PD winner → | MT winner → |
|---|---|---|---|---|---|---|---|---|
| 162 | 18.2711 | 19.4579 | 21.4457 | 4.5406 | 9.6957 | 5.1857 | cnn → cnn | cnn → cnn |
| 163 | 19.3704 | 20.4464 | 22.1821 | 5.6989 | 6.7890 | 4.9161 | cnn → cnn | cnn → candidateC |

## Winner distribution

### PD distance winners

| Method | Wins before candidateC | Wins after candidateC |
|---|---|---|
| cnn | 2 | 2 |
| gan | 166 | 166 |

### MT distance winners

| Method | Wins before candidateC | Wins after candidateC |
|---|---|---|
| candidateC | 0 | 94 |
| cnn | 148 | 65 |
| gan | 20 | 9 |

## Full per-sample detail: focus samples

Includes adjacent-cluster samples, rare controls, and the 20 MT-GAN baseline wins.

| sample | PD CNN | PD GAN | PD candidateC | MT CNN | MT GAN | MT candidateC | MT-GAN-win? | PD after | MT after |
|---|---|---|---|---|---|---|---|---|---|
| 6 | 28.0690 | 24.0451 | 28.0635 | 6.1440 | 6.0106 | 5.7881 | ✓ | gan | candidateC |
| 8 | 29.6351 | 23.2821 | 28.9463 | 7.5096 | 6.0550 | 7.0123 | ✓ | gan | gan |
| 10 | 29.6970 | 24.7100 | 28.8943 | 6.6345 | 7.4238 | 6.0843 |  | gan | candidateC |
| 11 | 28.6873 | 24.9699 | 28.3733 | 5.9107 | 7.8317 | 5.8242 |  | gan | candidateC |
| 12 | 30.1045 | 24.2206 | 29.3527 | 6.6164 | 5.9137 | 6.1039 | ✓ | gan | gan |
| 13 | 29.2737 | 24.3237 | 28.7161 | 6.0738 | 7.4023 | 6.6037 |  | gan | cnn |
| 16 | 23.7560 | 17.3245 | 22.9363 | 5.4642 | 5.2552 | 5.3338 | ✓ | gan | gan |
| 17 | 20.9004 | 17.1181 | 20.6481 | 5.2897 | 4.8487 | 5.3671 | ✓ | gan | gan |
| 18 | 20.3949 | 18.6102 | 20.0227 | 6.2586 | 5.5544 | 5.2225 | ✓ | gan | candidateC |
| 19 | 21.0641 | 19.0867 | 19.7804 | 6.9280 | 5.4429 | 5.8876 | ✓ | gan | gan |
| 20 | 21.9792 | 18.2663 | 21.2869 | 6.2541 | 5.6891 | 5.4852 | ✓ | gan | candidateC |
| 25 | 36.5841 | 30.1235 | 36.0593 | 11.3716 | 10.5015 | 6.5417 | ✓ | gan | candidateC |
| 48 | 22.8325 | 13.1070 | 20.1040 | 7.0815 | 6.6830 | 7.0206 | ✓ | gan | gan |
| 62 | 27.4635 | 17.9937 | 26.1244 | 6.7203 | 6.6502 | 6.5421 | ✓ | gan | candidateC |
| 63 | 28.5246 | 18.1998 | 27.1630 | 7.6606 | 7.5776 | 6.5270 | ✓ | gan | candidateC |
| 65 | 22.1489 | 16.2395 | 20.9500 | 5.8208 | 5.5768 | 5.2142 | ✓ | gan | candidateC |
| 68 | 21.7713 | 15.5009 | 20.3339 | 6.0743 | 5.6709 | 5.5099 | ✓ | gan | candidateC |
| 77 | 29.9447 | 20.3533 | 28.5126 | 6.1320 | 6.0000 | 6.0790 | ✓ | gan | gan |
| 79 | 30.1481 | 20.3212 | 28.9610 | 6.8565 | 6.4670 | 6.2650 | ✓ | gan | candidateC |
| 80 | 30.0863 | 20.7449 | 28.5088 | 6.9827 | 6.1767 | 5.2586 | ✓ | gan | candidateC |
| 82 | 28.9009 | 20.8163 | 27.8805 | 7.1841 | 6.3036 | 6.6031 | ✓ | gan | gan |
| 90 | 22.2068 | 17.3268 | 20.9458 | 5.2579 | 5.6635 | 5.0751 |  | gan | candidateC |
| 91 | 22.3094 | 16.2661 | 21.1217 | 5.5083 | 5.5198 | 5.7392 |  | gan | cnn |
| 92 | 22.0793 | 16.8077 | 21.0387 | 5.7358 | 5.6779 | 5.2720 | ✓ | gan | candidateC |
| 93 | 24.4029 | 18.1355 | 24.0283 | 6.5070 | 6.8160 | 6.1437 |  | gan | candidateC |
| 154 | 25.3102 | 16.8345 | 24.5896 | 5.8898 | 5.7368 | 5.8193 | ✓ | gan | gan |
| 162 | 18.2711 | 19.4579 | 21.4457 | 4.5406 | 9.6957 | 5.1857 |  | cnn | cnn |
| 163 | 19.3704 | 20.4464 | 22.1821 | 5.6989 | 6.7890 | 4.9161 |  | cnn | candidateC |

## Output files

- `ttk_runs_fixed/topology_finetuning/candidateC_topology/candidateC_pd_mt_distances.csv` — candidateC PD/MT per sample
- `ttk_runs_fixed/topology_finetuning/candidateC_topology/candidateC_topology_comparison.csv` — three-way comparison (CNN, GAN, candidateC)
- `docs/topology_finetuning_candidateC_topology_eval.md` — this report

## Notes

- PD distance = bottleneck distance between persistence diagrams (computed by `ttkBottleneckDistance`).
- MT distance = Wasserstein-type merge tree distance (computed by `ttkMergeTreeDistanceMatrix`).
- Both use 160×160 speed patches at (x0=0, y0=0), consistent with the CNN/GAN baseline.
- **Lower distance = SR is topologically closer to GT. Lower is better for both metrics.**
- **Component-count proxy metrics** (counts of connected components in speed superlevel sets) are reported separately in `topology_finetuning_candidateC_eval.md` and do not require TTK.
- candidateC was fine-tuned from the pretrained CNN checkpoint. Results may change with different lambda settings or more epochs.
- Existing CNN/GAN and Candidate B topology outputs were not overwritten.
