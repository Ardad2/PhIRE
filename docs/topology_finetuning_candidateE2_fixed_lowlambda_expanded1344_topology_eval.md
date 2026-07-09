# candidateE2_fixed_lowlambda_expanded1344 topology evaluation

**Generated:** 2026-07-08

**Important:** This report covers **true topology metrics** (PD bottleneck distance, MT Wasserstein distance) computed by TTK. These are distinct from the component-count *proxy* metrics in `topology_finetuning_candidateE2_fixed_lowlambda_expanded1344_eval.md`.

**Note:** Lower PD/MT distance = SR is topologically closer to GT. **Lower is better for both metrics.**

**candidateE2_fixed_lowlambda_expanded1344:** fine-tuned from the pretrained CNN checkpoint with auxiliary physics/topology losses. See the corresponding fine-tuning pilot script for configuration details.

**Samples evaluated:** 168

**MT-GAN baseline wins (before candidateE2_fixed_lowlambda_expanded1344):** 20 samples — [6, 8, 12, 16, 17, 18, 19, 20, 25, 48, 62, 63, 65, 68, 77, 79, 80, 82, 92, 154]

## Summary statistics

| Metric | CNN (baseline) | GAN (baseline) | candidateE2_fixed_lowlambda_expanded1344 |
|---|---|---|---|
| PD distance (mean) | 27.4063 | 20.8641 | 26.9905 |
| MT distance (mean) | 5.8678 | 8.3481 | 5.7102 |
| PD wins (vs CNN) | — | 166 | 128 |
| MT wins (vs CNN) | — | 20 | 120 |
| PD beats GAN | — | — | 2 |
| MT beats GAN | — | — | 159 |

## 6 Key evaluation questions

### Q1. Does candidateE2_fixed_lowlambda_expanded1344 improve PD distance relative to baseline CNN?

- Mean PD: CNN=27.4063, candidateE2_fixed_lowlambda_expanded1344=26.9905, Δ=-0.4158 (▼ better)
- candidateE2_fixed_lowlambda_expanded1344 has lower PD on **128/168** samples.

### Q2. Does candidateE2_fixed_lowlambda_expanded1344 ever beat GAN on PD distance?

- candidateE2_fixed_lowlambda_expanded1344 beats GAN on PD for **2/168** samples.
- Mean PD: GAN=20.8641, candidateE2_fixed_lowlambda_expanded1344=26.9905. Δ=6.1264 (▲ worse than GAN on average)

### Q3. Does candidateE2_fixed_lowlambda_expanded1344 improve MT distance relative to baseline CNN?

- Mean MT: CNN=5.8678, candidateE2_fixed_lowlambda_expanded1344=5.7102, Δ=-0.1576 (▼ better)
- candidateE2_fixed_lowlambda_expanded1344 has lower MT on **120/168** samples.

### Q4. Does candidateE2_fixed_lowlambda_expanded1344 change any of the original 20 MT-GAN cases?

The 20 samples where GAN originally beat CNN on MT distance: [6, 8, 12, 16, 17, 18, 19, 20, 25, 48, 62, 63, 65, 68, 77, 79, 80, 82, 92, 154]

| Outcome | Count |
|---|---|
| MT winner stays GAN | 9 |
| MT winner changes to candidateE2_fixed_lowlambda_expanded1344 | 11 |
| MT winner changes to CNN | 0 |

Per-sample detail for changed cases:

| sample_idx | MT CNN | MT GAN | MT candidateE2_fixed_lowlambda_expanded1344 | winner before | winner after |
|---|---|---|---|---|---|
| 6 | 6.1440 | 6.0106 | 5.9640 | gan | candidateE2_fixed_lowlambda_expanded1344 |
| 16 | 5.4642 | 5.2552 | 5.0587 | gan | candidateE2_fixed_lowlambda_expanded1344 |
| 20 | 6.2541 | 5.6891 | 5.4843 | gan | candidateE2_fixed_lowlambda_expanded1344 |
| 25 | 11.3716 | 10.5015 | 9.9861 | gan | candidateE2_fixed_lowlambda_expanded1344 |
| 62 | 6.7203 | 6.6502 | 6.3360 | gan | candidateE2_fixed_lowlambda_expanded1344 |
| 63 | 7.6606 | 7.5776 | 6.7259 | gan | candidateE2_fixed_lowlambda_expanded1344 |
| 65 | 5.8208 | 5.5768 | 5.3544 | gan | candidateE2_fixed_lowlambda_expanded1344 |
| 77 | 6.1320 | 6.0000 | 5.6828 | gan | candidateE2_fixed_lowlambda_expanded1344 |
| 79 | 6.8565 | 6.4670 | 6.2432 | gan | candidateE2_fixed_lowlambda_expanded1344 |
| 80 | 6.9827 | 6.1767 | 6.0892 | gan | candidateE2_fixed_lowlambda_expanded1344 |
| 92 | 5.7358 | 5.6779 | 5.5337 | gan | candidateE2_fixed_lowlambda_expanded1344 |

### Q5. What happens in adjacent clusters (samples 10–13 and 90–93)?

| sample_idx | PD CNN | PD GAN | PD candidateE2_fixed_lowlambda_expanded1344 | MT CNN | MT GAN | MT candidateE2_fixed_lowlambda_expanded1344 | PD winner → | MT winner → |
|---|---|---|---|---|---|---|---|---|
| 10 | 29.6970 | 24.7100 | 28.8255 | 6.6345 | 7.4238 | 5.9860 | gan → gan | cnn → candidateE2_fixed_lowlambda_expanded1344 |
| 11 | 28.6873 | 24.9699 | 27.5590 | 5.9107 | 7.8317 | 6.7331 | gan → gan | cnn → cnn |
| 12 | 30.1045 | 24.2206 | 29.1344 | 6.6164 | 5.9137 | 6.3473 | gan → gan | gan → gan |
| 13 | 29.2737 | 24.3237 | 28.0974 | 6.0738 | 7.4023 | 6.1040 | gan → gan | cnn → cnn |
| 90 | 22.2068 | 17.3268 | 21.9003 | 5.2579 | 5.6635 | 5.2478 | gan → gan | cnn → candidateE2_fixed_lowlambda_expanded1344 |
| 91 | 22.3094 | 16.2661 | 21.9798 | 5.5083 | 5.5198 | 5.3523 | gan → gan | cnn → candidateE2_fixed_lowlambda_expanded1344 |
| 92 | 22.0793 | 16.8077 | 21.8185 | 5.7358 | 5.6779 | 5.5337 | gan → gan | gan → candidateE2_fixed_lowlambda_expanded1344 |
| 93 | 24.4029 | 18.1355 | 24.3229 | 6.5070 | 6.8160 | 6.4197 | gan → gan | cnn → candidateE2_fixed_lowlambda_expanded1344 |

Note: samples [12, 92] are in the MT-GAN-wins set (GAN beat CNN on MT in baseline).

### Q6. What happens in rare topology-CNN controls (samples 162 and 163)?

These samples were originally topology-CNN controls (CNN wins on topology metrics). candidateE2_fixed_lowlambda_expanded1344 should not reverse this.

| sample_idx | PD CNN | PD GAN | PD candidateE2_fixed_lowlambda_expanded1344 | MT CNN | MT GAN | MT candidateE2_fixed_lowlambda_expanded1344 | PD winner → | MT winner → |
|---|---|---|---|---|---|---|---|---|
| 162 | 18.2711 | 19.4579 | 18.8031 | 4.5406 | 9.6957 | 4.5369 | cnn → cnn | cnn → candidateE2_fixed_lowlambda_expanded1344 |
| 163 | 19.3704 | 20.4464 | 19.7352 | 5.6989 | 6.7890 | 5.3362 | cnn → cnn | cnn → candidateE2_fixed_lowlambda_expanded1344 |

## Winner distribution

### PD distance winners

| Method | Wins before candidateE2_fixed_lowlambda_expanded1344 | Wins after candidateE2_fixed_lowlambda_expanded1344 |
|---|---|---|
| cnn | 2 | 2 |
| gan | 166 | 166 |

### MT distance winners

| Method | Wins before candidateE2_fixed_lowlambda_expanded1344 | Wins after candidateE2_fixed_lowlambda_expanded1344 |
|---|---|---|
| candidateE2_fixed_lowlambda_expanded1344 | 0 | 112 |
| cnn | 148 | 47 |
| gan | 20 | 9 |

## Full per-sample detail: focus samples

Includes adjacent-cluster samples, rare controls, and the 20 MT-GAN baseline wins.

| sample | PD CNN | PD GAN | PD candidateE2_fixed_lowlambda_expanded1344 | MT CNN | MT GAN | MT candidateE2_fixed_lowlambda_expanded1344 | MT-GAN-win? | PD after | MT after |
|---|---|---|---|---|---|---|---|---|---|
| 6 | 28.0690 | 24.0451 | 28.0068 | 6.1440 | 6.0106 | 5.9640 | ✓ | gan | candidateE2_fixed_lowlambda_expanded1344 |
| 8 | 29.6351 | 23.2821 | 29.1140 | 7.5096 | 6.0550 | 7.4867 | ✓ | gan | gan |
| 10 | 29.6970 | 24.7100 | 28.8255 | 6.6345 | 7.4238 | 5.9860 |  | gan | candidateE2_fixed_lowlambda_expanded1344 |
| 11 | 28.6873 | 24.9699 | 27.5590 | 5.9107 | 7.8317 | 6.7331 |  | gan | cnn |
| 12 | 30.1045 | 24.2206 | 29.1344 | 6.6164 | 5.9137 | 6.3473 | ✓ | gan | gan |
| 13 | 29.2737 | 24.3237 | 28.0974 | 6.0738 | 7.4023 | 6.1040 |  | gan | cnn |
| 16 | 23.7560 | 17.3245 | 22.6238 | 5.4642 | 5.2552 | 5.0587 | ✓ | gan | candidateE2_fixed_lowlambda_expanded1344 |
| 17 | 20.9004 | 17.1181 | 19.5799 | 5.2897 | 4.8487 | 5.1502 | ✓ | gan | gan |
| 18 | 20.3949 | 18.6102 | 19.1080 | 6.2586 | 5.5544 | 5.5792 | ✓ | gan | gan |
| 19 | 21.0641 | 19.0867 | 19.5724 | 6.9280 | 5.4429 | 6.2229 | ✓ | gan | gan |
| 20 | 21.9792 | 18.2663 | 20.2696 | 6.2541 | 5.6891 | 5.4843 | ✓ | gan | candidateE2_fixed_lowlambda_expanded1344 |
| 25 | 36.5841 | 30.1235 | 35.0582 | 11.3716 | 10.5015 | 9.9861 | ✓ | gan | candidateE2_fixed_lowlambda_expanded1344 |
| 48 | 22.8325 | 13.1070 | 22.8766 | 7.0815 | 6.6830 | 6.7062 | ✓ | gan | gan |
| 62 | 27.4635 | 17.9937 | 27.0893 | 6.7203 | 6.6502 | 6.3360 | ✓ | gan | candidateE2_fixed_lowlambda_expanded1344 |
| 63 | 28.5246 | 18.1998 | 27.9953 | 7.6606 | 7.5776 | 6.7259 | ✓ | gan | candidateE2_fixed_lowlambda_expanded1344 |
| 65 | 22.1489 | 16.2395 | 21.7870 | 5.8208 | 5.5768 | 5.3544 | ✓ | gan | candidateE2_fixed_lowlambda_expanded1344 |
| 68 | 21.7713 | 15.5009 | 21.5432 | 6.0743 | 5.6709 | 5.7871 | ✓ | gan | gan |
| 77 | 29.9447 | 20.3533 | 29.2722 | 6.1320 | 6.0000 | 5.6828 | ✓ | gan | candidateE2_fixed_lowlambda_expanded1344 |
| 79 | 30.1481 | 20.3212 | 29.5007 | 6.8565 | 6.4670 | 6.2432 | ✓ | gan | candidateE2_fixed_lowlambda_expanded1344 |
| 80 | 30.0863 | 20.7449 | 29.3935 | 6.9827 | 6.1767 | 6.0892 | ✓ | gan | candidateE2_fixed_lowlambda_expanded1344 |
| 82 | 28.9009 | 20.8163 | 28.3244 | 7.1841 | 6.3036 | 6.4721 | ✓ | gan | gan |
| 90 | 22.2068 | 17.3268 | 21.9003 | 5.2579 | 5.6635 | 5.2478 |  | gan | candidateE2_fixed_lowlambda_expanded1344 |
| 91 | 22.3094 | 16.2661 | 21.9798 | 5.5083 | 5.5198 | 5.3523 |  | gan | candidateE2_fixed_lowlambda_expanded1344 |
| 92 | 22.0793 | 16.8077 | 21.8185 | 5.7358 | 5.6779 | 5.5337 | ✓ | gan | candidateE2_fixed_lowlambda_expanded1344 |
| 93 | 24.4029 | 18.1355 | 24.3229 | 6.5070 | 6.8160 | 6.4197 |  | gan | candidateE2_fixed_lowlambda_expanded1344 |
| 154 | 25.3102 | 16.8345 | 25.5118 | 5.8898 | 5.7368 | 5.9980 | ✓ | gan | gan |
| 162 | 18.2711 | 19.4579 | 18.8031 | 4.5406 | 9.6957 | 4.5369 |  | cnn | candidateE2_fixed_lowlambda_expanded1344 |
| 163 | 19.3704 | 20.4464 | 19.7352 | 5.6989 | 6.7890 | 5.3362 |  | cnn | candidateE2_fixed_lowlambda_expanded1344 |

## Output files

- `ttk_runs_fixed/topology_finetuning/candidateE2_fixed_lowlambda_expanded1344_topology/candidateE2_fixed_lowlambda_expanded1344_pd_mt_distances.csv` — candidateE2_fixed_lowlambda_expanded1344 PD/MT per sample
- `ttk_runs_fixed/topology_finetuning/candidateE2_fixed_lowlambda_expanded1344_topology/candidateE2_fixed_lowlambda_expanded1344_topology_comparison.csv` — three-way comparison (CNN, GAN, candidateE2_fixed_lowlambda_expanded1344)
- `docs/topology_finetuning_candidateE2_fixed_lowlambda_expanded1344_topology_eval.md` — this report

## Notes

- PD distance = bottleneck distance between persistence diagrams (computed by `ttkBottleneckDistance`).
- MT distance = Wasserstein-type merge tree distance (computed by `ttkMergeTreeDistanceMatrix`).
- Both use 160×160 speed patches at (x0=0, y0=0), consistent with the CNN/GAN baseline.
- **Lower distance = SR is topologically closer to GT. Lower is better for both metrics.**
- **Component-count proxy metrics** (counts of connected components in speed superlevel sets) are reported separately in `topology_finetuning_candidateE2_fixed_lowlambda_expanded1344_eval.md` and do not require TTK.
- candidateE2_fixed_lowlambda_expanded1344 was fine-tuned from the pretrained CNN checkpoint. Results may change with different lambda settings or more epochs.
- Existing CNN/GAN and Candidate B topology outputs were not overwritten.
