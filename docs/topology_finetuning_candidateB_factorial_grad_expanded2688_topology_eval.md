# candidateB_factorial_grad_expanded2688 topology evaluation

**Generated:** 2026-07-15

**Important:** This report covers **true topology metrics** (PD bottleneck distance, MT Wasserstein distance) computed by TTK. These are distinct from the component-count *proxy* metrics in `topology_finetuning_candidateB_factorial_grad_expanded2688_eval.md`.

**Note:** Lower PD/MT distance = SR is topologically closer to GT. **Lower is better for both metrics.**

**candidateB_factorial_grad_expanded2688:** fine-tuned from the pretrained CNN checkpoint with auxiliary physics/topology losses. See the corresponding fine-tuning pilot script for configuration details.

**Samples evaluated:** 168

**MT-GAN baseline wins (before candidateB_factorial_grad_expanded2688):** 20 samples — [6, 8, 12, 16, 17, 18, 19, 20, 25, 48, 62, 63, 65, 68, 77, 79, 80, 82, 92, 154]

## Summary statistics

| Metric | CNN (baseline) | GAN (baseline) | candidateB_factorial_grad_expanded2688 |
|---|---|---|---|
| PD distance (mean) | 27.4063 | 20.8641 | 22.9326 |
| MT distance (mean) | 5.8678 | 8.3481 | 6.0560 |
| PD wins (vs CNN) | — | 166 | 157 |
| MT wins (vs CNN) | — | 20 | 66 |
| PD beats GAN | — | — | 22 |
| MT beats GAN | — | — | 154 |

## 6 Key evaluation questions

### Q1. Does candidateB_factorial_grad_expanded2688 improve PD distance relative to baseline CNN?

- Mean PD: CNN=27.4063, candidateB_factorial_grad_expanded2688=22.9326, Δ=-4.4737 (▼ better)
- candidateB_factorial_grad_expanded2688 has lower PD on **157/168** samples.

### Q2. Does candidateB_factorial_grad_expanded2688 ever beat GAN on PD distance?

- candidateB_factorial_grad_expanded2688 beats GAN on PD for **22/168** samples.
- Mean PD: GAN=20.8641, candidateB_factorial_grad_expanded2688=22.9326. Δ=2.0685 (▲ worse than GAN on average)

### Q3. Does candidateB_factorial_grad_expanded2688 improve MT distance relative to baseline CNN?

- Mean MT: CNN=5.8678, candidateB_factorial_grad_expanded2688=6.0560, Δ=0.1882 (▲ worse)
- candidateB_factorial_grad_expanded2688 has lower MT on **66/168** samples.

### Q4. Does candidateB_factorial_grad_expanded2688 change any of the original 20 MT-GAN cases?

The 20 samples where GAN originally beat CNN on MT distance: [6, 8, 12, 16, 17, 18, 19, 20, 25, 48, 62, 63, 65, 68, 77, 79, 80, 82, 92, 154]

| Outcome | Count |
|---|---|
| MT winner stays GAN | 10 |
| MT winner changes to candidateB_factorial_grad_expanded2688 | 10 |
| MT winner changes to CNN | 0 |

Per-sample detail for changed cases:

| sample_idx | MT CNN | MT GAN | MT candidateB_factorial_grad_expanded2688 | winner before | winner after |
|---|---|---|---|---|---|
| 18 | 6.2586 | 5.5544 | 5.4659 | gan | candidateB_factorial_grad_expanded2688 |
| 25 | 11.3716 | 10.5015 | 6.5836 | gan | candidateB_factorial_grad_expanded2688 |
| 62 | 6.7203 | 6.6502 | 6.4504 | gan | candidateB_factorial_grad_expanded2688 |
| 63 | 7.6606 | 7.5776 | 6.8958 | gan | candidateB_factorial_grad_expanded2688 |
| 65 | 5.8208 | 5.5768 | 5.0304 | gan | candidateB_factorial_grad_expanded2688 |
| 68 | 6.0743 | 5.6709 | 4.9140 | gan | candidateB_factorial_grad_expanded2688 |
| 77 | 6.1320 | 6.0000 | 5.1747 | gan | candidateB_factorial_grad_expanded2688 |
| 79 | 6.8565 | 6.4670 | 5.1572 | gan | candidateB_factorial_grad_expanded2688 |
| 80 | 6.9827 | 6.1767 | 6.0892 | gan | candidateB_factorial_grad_expanded2688 |
| 82 | 7.1841 | 6.3036 | 6.1279 | gan | candidateB_factorial_grad_expanded2688 |

### Q5. What happens in adjacent clusters (samples 10–13 and 90–93)?

| sample_idx | PD CNN | PD GAN | PD candidateB_factorial_grad_expanded2688 | MT CNN | MT GAN | MT candidateB_factorial_grad_expanded2688 | PD winner → | MT winner → |
|---|---|---|---|---|---|---|---|---|
| 10 | 29.6970 | 24.7100 | 24.1424 | 6.6345 | 7.4238 | 5.9917 | gan → candidateB_factorial_grad_expanded2688 | cnn → candidateB_factorial_grad_expanded2688 |
| 11 | 28.6873 | 24.9699 | 25.3118 | 5.9107 | 7.8317 | 6.8819 | gan → gan | cnn → cnn |
| 12 | 30.1045 | 24.2206 | 25.6922 | 6.6164 | 5.9137 | 6.1735 | gan → gan | gan → gan |
| 13 | 29.2737 | 24.3237 | 24.7341 | 6.0738 | 7.4023 | 6.8928 | gan → gan | cnn → cnn |
| 90 | 22.2068 | 17.3268 | 17.6969 | 5.2579 | 5.6635 | 5.1240 | gan → gan | cnn → candidateB_factorial_grad_expanded2688 |
| 91 | 22.3094 | 16.2661 | 17.3829 | 5.5083 | 5.5198 | 6.1678 | gan → gan | cnn → cnn |
| 92 | 22.0793 | 16.8077 | 18.1257 | 5.7358 | 5.6779 | 5.7880 | gan → gan | gan → gan |
| 93 | 24.4029 | 18.1355 | 21.9977 | 6.5070 | 6.8160 | 6.4372 | gan → gan | cnn → candidateB_factorial_grad_expanded2688 |

Note: samples [12, 92] are in the MT-GAN-wins set (GAN beat CNN on MT in baseline).

### Q6. What happens in rare topology-CNN controls (samples 162 and 163)?

These samples were originally topology-CNN controls (CNN wins on topology metrics). candidateB_factorial_grad_expanded2688 should not reverse this.

| sample_idx | PD CNN | PD GAN | PD candidateB_factorial_grad_expanded2688 | MT CNN | MT GAN | MT candidateB_factorial_grad_expanded2688 | PD winner → | MT winner → |
|---|---|---|---|---|---|---|---|---|
| 162 | 18.2711 | 19.4579 | 19.2441 | 4.5406 | 9.6957 | 6.0054 | cnn → cnn | cnn → cnn |
| 163 | 19.3704 | 20.4464 | 20.0693 | 5.6989 | 6.7890 | 6.6250 | cnn → cnn | cnn → cnn |

## Winner distribution

### PD distance winners

| Method | Wins before candidateB_factorial_grad_expanded2688 | Wins after candidateB_factorial_grad_expanded2688 |
|---|---|---|
| candidateB_factorial_grad_expanded2688 | 0 | 20 |
| cnn | 2 | 2 |
| gan | 166 | 146 |

### MT distance winners

| Method | Wins before candidateB_factorial_grad_expanded2688 | Wins after candidateB_factorial_grad_expanded2688 |
|---|---|---|
| candidateB_factorial_grad_expanded2688 | 0 | 62 |
| cnn | 148 | 96 |
| gan | 20 | 10 |

## Full per-sample detail: focus samples

Includes adjacent-cluster samples, rare controls, and the 20 MT-GAN baseline wins.

| sample | PD CNN | PD GAN | PD candidateB_factorial_grad_expanded2688 | MT CNN | MT GAN | MT candidateB_factorial_grad_expanded2688 | MT-GAN-win? | PD after | MT after |
|---|---|---|---|---|---|---|---|---|---|
| 6 | 28.0690 | 24.0451 | 22.6922 | 6.1440 | 6.0106 | 6.4990 | ✓ | candidateB_factorial_grad_expanded2688 | gan |
| 8 | 29.6351 | 23.2821 | 23.4842 | 7.5096 | 6.0550 | 6.0976 | ✓ | gan | gan |
| 10 | 29.6970 | 24.7100 | 24.1424 | 6.6345 | 7.4238 | 5.9917 |  | candidateB_factorial_grad_expanded2688 | candidateB_factorial_grad_expanded2688 |
| 11 | 28.6873 | 24.9699 | 25.3118 | 5.9107 | 7.8317 | 6.8819 |  | gan | cnn |
| 12 | 30.1045 | 24.2206 | 25.6922 | 6.6164 | 5.9137 | 6.1735 | ✓ | gan | gan |
| 13 | 29.2737 | 24.3237 | 24.7341 | 6.0738 | 7.4023 | 6.8928 |  | gan | cnn |
| 16 | 23.7560 | 17.3245 | 18.3448 | 5.4642 | 5.2552 | 5.6019 | ✓ | gan | gan |
| 17 | 20.9004 | 17.1181 | 17.0303 | 5.2897 | 4.8487 | 5.3673 | ✓ | candidateB_factorial_grad_expanded2688 | gan |
| 18 | 20.3949 | 18.6102 | 16.8002 | 6.2586 | 5.5544 | 5.4659 | ✓ | candidateB_factorial_grad_expanded2688 | candidateB_factorial_grad_expanded2688 |
| 19 | 21.0641 | 19.0867 | 18.1202 | 6.9280 | 5.4429 | 5.6326 | ✓ | candidateB_factorial_grad_expanded2688 | gan |
| 20 | 21.9792 | 18.2663 | 20.7210 | 6.2541 | 5.6891 | 6.3951 | ✓ | gan | gan |
| 25 | 36.5841 | 30.1235 | 35.6417 | 11.3716 | 10.5015 | 6.5836 | ✓ | gan | candidateB_factorial_grad_expanded2688 |
| 48 | 22.8325 | 13.1070 | 15.8992 | 7.0815 | 6.6830 | 7.7019 | ✓ | gan | gan |
| 62 | 27.4635 | 17.9937 | 20.7521 | 6.7203 | 6.6502 | 6.4504 | ✓ | gan | candidateB_factorial_grad_expanded2688 |
| 63 | 28.5246 | 18.1998 | 21.1056 | 7.6606 | 7.5776 | 6.8958 | ✓ | gan | candidateB_factorial_grad_expanded2688 |
| 65 | 22.1489 | 16.2395 | 17.0979 | 5.8208 | 5.5768 | 5.0304 | ✓ | gan | candidateB_factorial_grad_expanded2688 |
| 68 | 21.7713 | 15.5009 | 15.9998 | 6.0743 | 5.6709 | 4.9140 | ✓ | gan | candidateB_factorial_grad_expanded2688 |
| 77 | 29.9447 | 20.3533 | 22.6195 | 6.1320 | 6.0000 | 5.1747 | ✓ | gan | candidateB_factorial_grad_expanded2688 |
| 79 | 30.1481 | 20.3212 | 22.4549 | 6.8565 | 6.4670 | 5.1572 | ✓ | gan | candidateB_factorial_grad_expanded2688 |
| 80 | 30.0863 | 20.7449 | 22.0218 | 6.9827 | 6.1767 | 6.0892 | ✓ | gan | candidateB_factorial_grad_expanded2688 |
| 82 | 28.9009 | 20.8163 | 21.9452 | 7.1841 | 6.3036 | 6.1279 | ✓ | gan | candidateB_factorial_grad_expanded2688 |
| 90 | 22.2068 | 17.3268 | 17.6969 | 5.2579 | 5.6635 | 5.1240 |  | gan | candidateB_factorial_grad_expanded2688 |
| 91 | 22.3094 | 16.2661 | 17.3829 | 5.5083 | 5.5198 | 6.1678 |  | gan | cnn |
| 92 | 22.0793 | 16.8077 | 18.1257 | 5.7358 | 5.6779 | 5.7880 | ✓ | gan | gan |
| 93 | 24.4029 | 18.1355 | 21.9977 | 6.5070 | 6.8160 | 6.4372 |  | gan | candidateB_factorial_grad_expanded2688 |
| 154 | 25.3102 | 16.8345 | 18.6121 | 5.8898 | 5.7368 | 5.8772 | ✓ | gan | gan |
| 162 | 18.2711 | 19.4579 | 19.2441 | 4.5406 | 9.6957 | 6.0054 |  | cnn | cnn |
| 163 | 19.3704 | 20.4464 | 20.0693 | 5.6989 | 6.7890 | 6.6250 |  | cnn | cnn |

## Output files

- `ttk_runs_fixed/topology_finetuning/candidateB_factorial_grad_expanded2688_topology/candidateB_factorial_grad_expanded2688_pd_mt_distances.csv` — candidateB_factorial_grad_expanded2688 PD/MT per sample
- `ttk_runs_fixed/topology_finetuning/candidateB_factorial_grad_expanded2688_topology/candidateB_factorial_grad_expanded2688_topology_comparison.csv` — three-way comparison (CNN, GAN, candidateB_factorial_grad_expanded2688)
- `docs/topology_finetuning_candidateB_factorial_grad_expanded2688_topology_eval.md` — this report

## Notes

- PD distance = bottleneck distance between persistence diagrams (computed by `ttkBottleneckDistance`).
- MT distance = Wasserstein-type merge tree distance (computed by `ttkMergeTreeDistanceMatrix`).
- Both use 160×160 speed patches at (x0=0, y0=0), consistent with the CNN/GAN baseline.
- **Lower distance = SR is topologically closer to GT. Lower is better for both metrics.**
- **Component-count proxy metrics** (counts of connected components in speed superlevel sets) are reported separately in `topology_finetuning_candidateB_factorial_grad_expanded2688_eval.md` and do not require TTK.
- candidateB_factorial_grad_expanded2688 was fine-tuned from the pretrained CNN checkpoint. Results may change with different lambda settings or more epochs.
- Existing CNN/GAN and Candidate B topology outputs were not overwritten.
