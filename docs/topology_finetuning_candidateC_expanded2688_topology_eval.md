# candidateC_expanded2688 topology evaluation

**Generated:** 2026-05-29

**Important:** This report covers **true topology metrics** (PD bottleneck distance, MT Wasserstein distance) computed by TTK. These are distinct from the component-count *proxy* metrics in `topology_finetuning_candidateC_expanded2688_eval.md`.

**Note:** Lower PD/MT distance = SR is topologically closer to GT. **Lower is better for both metrics.**

**candidateC_expanded2688:** fine-tuned from the pretrained CNN checkpoint with auxiliary physics/topology losses. See the corresponding fine-tuning pilot script for configuration details.

**Candidate C = Candidate B + critical-value/topological-extrema proxy loss** (`lambda_crit=0.001`). Candidate B adds `lambda_speed=0.01, lambda_grad=0.05, lambda_levelset=0.25` on top of the baseline CNN MSE loss.

**Samples evaluated:** 168

**MT-GAN baseline wins (before candidateC_expanded2688):** 20 samples — [6, 8, 12, 16, 17, 18, 19, 20, 25, 48, 62, 63, 65, 68, 77, 79, 80, 82, 92, 154]

## Summary statistics

| Metric | CNN (baseline) | GAN (baseline) | candidateC_expanded2688 |
|---|---|---|---|
| PD distance (mean) | 27.4063 | 20.8641 | 22.4944 |
| MT distance (mean) | 5.8678 | 8.3481 | 6.0803 |
| PD wins (vs CNN) | — | 166 | 168 |
| MT wins (vs CNN) | — | 20 | 52 |
| PD beats GAN | — | — | 20 |
| MT beats GAN | — | — | 147 |

## 6 Key evaluation questions

### Q1. Does candidateC_expanded2688 improve PD distance relative to baseline CNN?

- Mean PD: CNN=27.4063, candidateC_expanded2688=22.4944, Δ=-4.9119 (▼ better)
- candidateC_expanded2688 has lower PD on **168/168** samples.

### Q2. Does candidateC_expanded2688 ever beat GAN on PD distance?

- candidateC_expanded2688 beats GAN on PD for **20/168** samples.
- Mean PD: GAN=20.8641, candidateC_expanded2688=22.4944. Δ=1.6303 (▲ worse than GAN on average)

### Q3. Does candidateC_expanded2688 improve MT distance relative to baseline CNN?

- Mean MT: CNN=5.8678, candidateC_expanded2688=6.0803, Δ=0.2125 (▲ worse)
- candidateC_expanded2688 has lower MT on **52/168** samples.

### Q4. Does candidateC_expanded2688 change any of the original 20 MT-GAN cases?

The 20 samples where GAN originally beat CNN on MT distance: [6, 8, 12, 16, 17, 18, 19, 20, 25, 48, 62, 63, 65, 68, 77, 79, 80, 82, 92, 154]

| Outcome | Count |
|---|---|
| MT winner stays GAN | 10 |
| MT winner changes to candidateC_expanded2688 | 10 |
| MT winner changes to CNN | 0 |

Per-sample detail for changed cases:

| sample_idx | MT CNN | MT GAN | MT candidateC_expanded2688 | winner before | winner after |
|---|---|---|---|---|---|
| 6 | 6.1440 | 6.0106 | 5.7433 | gan | candidateC_expanded2688 |
| 18 | 6.2586 | 5.5544 | 5.1673 | gan | candidateC_expanded2688 |
| 25 | 11.3716 | 10.5015 | 7.4621 | gan | candidateC_expanded2688 |
| 63 | 7.6606 | 7.5776 | 6.9989 | gan | candidateC_expanded2688 |
| 68 | 6.0743 | 5.6709 | 5.4428 | gan | candidateC_expanded2688 |
| 79 | 6.8565 | 6.4670 | 6.2152 | gan | candidateC_expanded2688 |
| 80 | 6.9827 | 6.1767 | 6.1106 | gan | candidateC_expanded2688 |
| 82 | 7.1841 | 6.3036 | 6.0884 | gan | candidateC_expanded2688 |
| 92 | 5.7358 | 5.6779 | 5.6635 | gan | candidateC_expanded2688 |
| 154 | 5.8898 | 5.7368 | 5.4092 | gan | candidateC_expanded2688 |

### Q5. What happens in adjacent clusters (samples 10–13 and 90–93)?

| sample_idx | PD CNN | PD GAN | PD candidateC_expanded2688 | MT CNN | MT GAN | MT candidateC_expanded2688 | PD winner → | MT winner → |
|---|---|---|---|---|---|---|---|---|
| 10 | 29.6970 | 24.7100 | 24.2662 | 6.6345 | 7.4238 | 6.1357 | gan → candidateC_expanded2688 | cnn → candidateC_expanded2688 |
| 11 | 28.6873 | 24.9699 | 24.3359 | 5.9107 | 7.8317 | 6.6922 | gan → candidateC_expanded2688 | cnn → cnn |
| 12 | 30.1045 | 24.2206 | 24.3050 | 6.6164 | 5.9137 | 6.6338 | gan → gan | gan → gan |
| 13 | 29.2737 | 24.3237 | 23.8631 | 6.0738 | 7.4023 | 6.4857 | gan → candidateC_expanded2688 | cnn → cnn |
| 90 | 22.2068 | 17.3268 | 17.4807 | 5.2579 | 5.6635 | 5.7815 | gan → gan | cnn → cnn |
| 91 | 22.3094 | 16.2661 | 16.9812 | 5.5083 | 5.5198 | 5.8139 | gan → gan | cnn → cnn |
| 92 | 22.0793 | 16.8077 | 17.7520 | 5.7358 | 5.6779 | 5.6635 | gan → gan | gan → candidateC_expanded2688 |
| 93 | 24.4029 | 18.1355 | 20.1519 | 6.5070 | 6.8160 | 7.3064 | gan → gan | cnn → cnn |

Note: samples [12, 92] are in the MT-GAN-wins set (GAN beat CNN on MT in baseline).

### Q6. What happens in rare topology-CNN controls (samples 162 and 163)?

These samples were originally topology-CNN controls (CNN wins on topology metrics). candidateC_expanded2688 should not reverse this.

| sample_idx | PD CNN | PD GAN | PD candidateC_expanded2688 | MT CNN | MT GAN | MT candidateC_expanded2688 | PD winner → | MT winner → |
|---|---|---|---|---|---|---|---|---|
| 162 | 18.2711 | 19.4579 | 17.7221 | 4.5406 | 9.6957 | 4.9298 | cnn → candidateC_expanded2688 | cnn → cnn |
| 163 | 19.3704 | 20.4464 | 19.0215 | 5.6989 | 6.7890 | 5.3675 | cnn → candidateC_expanded2688 | cnn → candidateC_expanded2688 |

## Winner distribution

### PD distance winners

| Method | Wins before candidateC_expanded2688 | Wins after candidateC_expanded2688 |
|---|---|---|
| candidateC_expanded2688 | 0 | 20 |
| cnn | 2 | 0 |
| gan | 166 | 148 |

### MT distance winners

| Method | Wins before candidateC_expanded2688 | Wins after candidateC_expanded2688 |
|---|---|---|
| candidateC_expanded2688 | 0 | 46 |
| cnn | 148 | 112 |
| gan | 20 | 10 |

## Full per-sample detail: focus samples

Includes adjacent-cluster samples, rare controls, and the 20 MT-GAN baseline wins.

| sample | PD CNN | PD GAN | PD candidateC_expanded2688 | MT CNN | MT GAN | MT candidateC_expanded2688 | MT-GAN-win? | PD after | MT after |
|---|---|---|---|---|---|---|---|---|---|
| 6 | 28.0690 | 24.0451 | 23.6230 | 6.1440 | 6.0106 | 5.7433 | ✓ | candidateC_expanded2688 | candidateC_expanded2688 |
| 8 | 29.6351 | 23.2821 | 24.1196 | 7.5096 | 6.0550 | 6.2696 | ✓ | gan | gan |
| 10 | 29.6970 | 24.7100 | 24.2662 | 6.6345 | 7.4238 | 6.1357 |  | candidateC_expanded2688 | candidateC_expanded2688 |
| 11 | 28.6873 | 24.9699 | 24.3359 | 5.9107 | 7.8317 | 6.6922 |  | candidateC_expanded2688 | cnn |
| 12 | 30.1045 | 24.2206 | 24.3050 | 6.6164 | 5.9137 | 6.6338 | ✓ | gan | gan |
| 13 | 29.2737 | 24.3237 | 23.8631 | 6.0738 | 7.4023 | 6.4857 |  | candidateC_expanded2688 | cnn |
| 16 | 23.7560 | 17.3245 | 19.4866 | 5.4642 | 5.2552 | 5.5117 | ✓ | gan | gan |
| 17 | 20.9004 | 17.1181 | 17.4575 | 5.2897 | 4.8487 | 5.1043 | ✓ | gan | gan |
| 18 | 20.3949 | 18.6102 | 16.7747 | 6.2586 | 5.5544 | 5.1673 | ✓ | candidateC_expanded2688 | candidateC_expanded2688 |
| 19 | 21.0641 | 19.0867 | 16.0161 | 6.9280 | 5.4429 | 5.4994 | ✓ | candidateC_expanded2688 | gan |
| 20 | 21.9792 | 18.2663 | 17.4421 | 6.2541 | 5.6891 | 5.8528 | ✓ | candidateC_expanded2688 | gan |
| 25 | 36.5841 | 30.1235 | 30.7946 | 11.3716 | 10.5015 | 7.4621 | ✓ | gan | candidateC_expanded2688 |
| 48 | 22.8325 | 13.1070 | 17.1610 | 7.0815 | 6.6830 | 7.0595 | ✓ | gan | gan |
| 62 | 27.4635 | 17.9937 | 22.1303 | 6.7203 | 6.6502 | 6.6682 | ✓ | gan | gan |
| 63 | 28.5246 | 18.1998 | 22.6494 | 7.6606 | 7.5776 | 6.9989 | ✓ | gan | candidateC_expanded2688 |
| 65 | 22.1489 | 16.2395 | 17.7036 | 5.8208 | 5.5768 | 6.1381 | ✓ | gan | gan |
| 68 | 21.7713 | 15.5009 | 16.0891 | 6.0743 | 5.6709 | 5.4428 | ✓ | gan | candidateC_expanded2688 |
| 77 | 29.9447 | 20.3533 | 23.5632 | 6.1320 | 6.0000 | 6.5723 | ✓ | gan | gan |
| 79 | 30.1481 | 20.3212 | 24.1981 | 6.8565 | 6.4670 | 6.2152 | ✓ | gan | candidateC_expanded2688 |
| 80 | 30.0863 | 20.7449 | 24.0500 | 6.9827 | 6.1767 | 6.1106 | ✓ | gan | candidateC_expanded2688 |
| 82 | 28.9009 | 20.8163 | 24.2360 | 7.1841 | 6.3036 | 6.0884 | ✓ | gan | candidateC_expanded2688 |
| 90 | 22.2068 | 17.3268 | 17.4807 | 5.2579 | 5.6635 | 5.7815 |  | gan | cnn |
| 91 | 22.3094 | 16.2661 | 16.9812 | 5.5083 | 5.5198 | 5.8139 |  | gan | cnn |
| 92 | 22.0793 | 16.8077 | 17.7520 | 5.7358 | 5.6779 | 5.6635 | ✓ | gan | candidateC_expanded2688 |
| 93 | 24.4029 | 18.1355 | 20.1519 | 6.5070 | 6.8160 | 7.3064 |  | gan | cnn |
| 154 | 25.3102 | 16.8345 | 19.3197 | 5.8898 | 5.7368 | 5.4092 | ✓ | gan | candidateC_expanded2688 |
| 162 | 18.2711 | 19.4579 | 17.7221 | 4.5406 | 9.6957 | 4.9298 |  | candidateC_expanded2688 | cnn |
| 163 | 19.3704 | 20.4464 | 19.0215 | 5.6989 | 6.7890 | 5.3675 |  | candidateC_expanded2688 | candidateC_expanded2688 |

## Output files

- `ttk_runs_fixed/topology_finetuning/candidateC_expanded2688_topology/candidateC_expanded2688_pd_mt_distances.csv` — candidateC_expanded2688 PD/MT per sample
- `ttk_runs_fixed/topology_finetuning/candidateC_expanded2688_topology/candidateC_expanded2688_topology_comparison.csv` — three-way comparison (CNN, GAN, candidateC_expanded2688)
- `docs/topology_finetuning_candidateC_expanded2688_topology_eval.md` — this report

## Notes

- PD distance = bottleneck distance between persistence diagrams (computed by `ttkBottleneckDistance`).
- MT distance = Wasserstein-type merge tree distance (computed by `ttkMergeTreeDistanceMatrix`).
- Both use 160×160 speed patches at (x0=0, y0=0), consistent with the CNN/GAN baseline.
- **Lower distance = SR is topologically closer to GT. Lower is better for both metrics.**
- **Component-count proxy metrics** (counts of connected components in speed superlevel sets) are reported separately in `topology_finetuning_candidateC_expanded2688_eval.md` and do not require TTK.
- candidateC_expanded2688 was fine-tuned from the pretrained CNN checkpoint. Results may change with different lambda settings or more epochs.
- Existing CNN/GAN and Candidate B topology outputs were not overwritten.
