# Candidate B topology evaluation

**Generated:** 2026-05-12

**Important:** This report covers **true topology metrics** (PD bottleneck distance, MT Wasserstein distance) computed by TTK. These are distinct from the component-count *proxy* metrics in `topology_finetuning_candidateB_eval.md`.

**Candidate B:** `lambda_speed=0.01, lambda_grad=0.05, lambda_wpd=0.0, lambda_levelset=0.25, lr=1e-5, 3 epochs`

**Samples evaluated:** 168

**MT-GAN baseline wins (before Candidate B):** 20 samples — [6, 8, 12, 16, 17, 18, 19, 20, 25, 48, 62, 63, 65, 68, 77, 79, 80, 82, 92, 154]

## Summary statistics

| Metric | CNN (baseline) | GAN (baseline) | Candidate B |
|---|---|---|---|
| PD distance (mean) | 27.4063 | 20.8641 | 26.1794 |
| MT distance (mean) | 5.8678 | 8.3481 | 6.0186 |
| PD wins (vs CNN) | — | 166 | 136 |
| MT wins (vs CNN) | — | 20 | 66 |
| PD beats GAN | — | — | 1 |
| MT beats GAN | — | — | 145 |

## 6 Key evaluation questions

### Q1. Does Candidate B improve PD distance relative to baseline CNN?

- Mean PD: CNN=27.4063, CandidateB=26.1794, Δ=-1.2269 (▼ better)
- Candidate B has lower PD on **136/168** samples.

### Q2. Does Candidate B ever beat GAN on PD distance?

- Candidate B beats GAN on PD for **1/168** samples.
- Mean PD: GAN=20.8641, CandidateB=26.1794. Δ=5.3153 (▲ worse than GAN on average)

### Q3. Does Candidate B improve MT distance relative to baseline CNN?

- Mean MT: CNN=5.8678, CandidateB=6.0186, Δ=0.1508 (▲ worse)
- Candidate B has lower MT on **66/168** samples.

### Q4. Does Candidate B change any of the original 20 MT-GAN cases?

The 20 samples where GAN originally beat CNN on MT distance: [6, 8, 12, 16, 17, 18, 19, 20, 25, 48, 62, 63, 65, 68, 77, 79, 80, 82, 92, 154]

| Outcome | Count |
|---|---|
| MT winner stays GAN | 16 |
| MT winner changes to CandidateB | 4 |
| MT winner changes to CNN | 0 |

Per-sample detail for changed cases:

| sample_idx | MT CNN | MT GAN | MT CandB | winner before | winner after |
|---|---|---|---|---|---|
| 18 | 6.2586 | 5.5544 | 5.5258 | gan | candidateB |
| 25 | 11.3716 | 10.5015 | 6.5420 | gan | candidateB |
| 63 | 7.6606 | 7.5776 | 6.9816 | gan | candidateB |
| 80 | 6.9827 | 6.1767 | 5.7105 | gan | candidateB |

### Q5. What happens in adjacent clusters (samples 10–13 and 90–93)?

| sample_idx | PD CNN | PD GAN | PD CandB | MT CNN | MT GAN | MT CandB | PD winner → | MT winner → |
|---|---|---|---|---|---|---|---|---|
| 10 | 29.6970 | 24.7100 | 27.7473 | 6.6345 | 7.4238 | 6.3437 | gan → gan | cnn → candidateB |
| 11 | 28.6873 | 24.9699 | 27.1888 | 5.9107 | 7.8317 | 6.7322 | gan → gan | cnn → cnn |
| 12 | 30.1045 | 24.2206 | 27.9032 | 6.6164 | 5.9137 | 6.0934 | gan → gan | gan → gan |
| 13 | 29.2737 | 24.3237 | 27.4926 | 6.0738 | 7.4023 | 6.8512 | gan → gan | cnn → cnn |
| 90 | 22.2068 | 17.3268 | 19.2843 | 5.2579 | 5.6635 | 5.4967 | gan → gan | cnn → cnn |
| 91 | 22.3094 | 16.2661 | 19.7203 | 5.5083 | 5.5198 | 6.5752 | gan → gan | cnn → cnn |
| 92 | 22.0793 | 16.8077 | 19.8349 | 5.7358 | 5.6779 | 5.8324 | gan → gan | gan → gan |
| 93 | 24.4029 | 18.1355 | 23.0513 | 6.5070 | 6.8160 | 5.8948 | gan → gan | cnn → candidateB |

Note: samples [12, 92] are in the MT-GAN-wins set (GAN beat CNN on MT in baseline).

### Q6. What happens in rare topology-CNN controls (samples 162 and 163)?

These samples were originally topology-CNN controls (CNN wins on topology metrics). Candidate B should not reverse this.

| sample_idx | PD CNN | PD GAN | PD CandB | MT CNN | MT GAN | MT CandB | PD winner → | MT winner → |
|---|---|---|---|---|---|---|---|---|
| 162 | 18.2711 | 19.4579 | 21.2772 | 4.5406 | 9.6957 | 6.3513 | cnn → cnn | cnn → cnn |
| 163 | 19.3704 | 20.4464 | 22.0634 | 5.6989 | 6.7890 | 5.1843 | cnn → cnn | cnn → candidateB |

## Winner distribution

### PD distance winners

| Method | Wins before CandB | Wins after CandB |
|---|---|---|
| candidateB | 0 | 1 |
| cnn | 2 | 2 |
| gan | 166 | 165 |

### MT distance winners

| Method | Wins before CandB | Wins after CandB |
|---|---|---|
| candidateB | 0 | 60 |
| cnn | 148 | 92 |
| gan | 20 | 16 |

## Full per-sample detail: focus samples

Includes adjacent-cluster samples, rare controls, and the 20 MT-GAN baseline wins.

| sample | PD CNN | PD GAN | PD CandB | MT CNN | MT GAN | MT CandB | MT-GAN-win? | PD after | MT after |
|---|---|---|---|---|---|---|---|---|---|
| 6 | 28.0690 | 24.0451 | 27.5618 | 6.1440 | 6.0106 | 6.2005 | ✓ | gan | gan |
| 8 | 29.6351 | 23.2821 | 27.9042 | 7.5096 | 6.0550 | 7.1545 | ✓ | gan | gan |
| 10 | 29.6970 | 24.7100 | 27.7473 | 6.6345 | 7.4238 | 6.3437 |  | gan | candidateB |
| 11 | 28.6873 | 24.9699 | 27.1888 | 5.9107 | 7.8317 | 6.7322 |  | gan | cnn |
| 12 | 30.1045 | 24.2206 | 27.9032 | 6.6164 | 5.9137 | 6.0934 | ✓ | gan | gan |
| 13 | 29.2737 | 24.3237 | 27.4926 | 6.0738 | 7.4023 | 6.8512 |  | gan | cnn |
| 16 | 23.7560 | 17.3245 | 21.8585 | 5.4642 | 5.2552 | 5.8597 | ✓ | gan | gan |
| 17 | 20.9004 | 17.1181 | 19.5392 | 5.2897 | 4.8487 | 5.6739 | ✓ | gan | gan |
| 18 | 20.3949 | 18.6102 | 18.8100 | 6.2586 | 5.5544 | 5.5258 | ✓ | gan | candidateB |
| 19 | 21.0641 | 19.0867 | 18.9337 | 6.9280 | 5.4429 | 6.2049 | ✓ | candidateB | gan |
| 20 | 21.9792 | 18.2663 | 20.4050 | 6.2541 | 5.6891 | 5.7956 | ✓ | gan | gan |
| 25 | 36.5841 | 30.1235 | 35.7248 | 11.3716 | 10.5015 | 6.5420 | ✓ | gan | candidateB |
| 48 | 22.8325 | 13.1070 | 19.0389 | 7.0815 | 6.6830 | 7.8532 | ✓ | gan | gan |
| 62 | 27.4635 | 17.9937 | 25.0197 | 6.7203 | 6.6502 | 7.0125 | ✓ | gan | gan |
| 63 | 28.5246 | 18.1998 | 25.9714 | 7.6606 | 7.5776 | 6.9816 | ✓ | gan | candidateB |
| 65 | 22.1489 | 16.2395 | 19.6739 | 5.8208 | 5.5768 | 5.9211 | ✓ | gan | gan |
| 68 | 21.7713 | 15.5009 | 18.7825 | 6.0743 | 5.6709 | 5.8812 | ✓ | gan | gan |
| 77 | 29.9447 | 20.3533 | 27.5169 | 6.1320 | 6.0000 | 6.5012 | ✓ | gan | gan |
| 79 | 30.1481 | 20.3212 | 28.0204 | 6.8565 | 6.4670 | 7.0591 | ✓ | gan | gan |
| 80 | 30.0863 | 20.7449 | 27.6699 | 6.9827 | 6.1767 | 5.7105 | ✓ | gan | candidateB |
| 82 | 28.9009 | 20.8163 | 26.6911 | 7.1841 | 6.3036 | 6.7961 | ✓ | gan | gan |
| 90 | 22.2068 | 17.3268 | 19.2843 | 5.2579 | 5.6635 | 5.4967 |  | gan | cnn |
| 91 | 22.3094 | 16.2661 | 19.7203 | 5.5083 | 5.5198 | 6.5752 |  | gan | cnn |
| 92 | 22.0793 | 16.8077 | 19.8349 | 5.7358 | 5.6779 | 5.8324 | ✓ | gan | gan |
| 93 | 24.4029 | 18.1355 | 23.0513 | 6.5070 | 6.8160 | 5.8948 |  | gan | candidateB |
| 154 | 25.3102 | 16.8345 | 23.8734 | 5.8898 | 5.7368 | 6.1549 | ✓ | gan | gan |
| 162 | 18.2711 | 19.4579 | 21.2772 | 4.5406 | 9.6957 | 6.3513 |  | cnn | cnn |
| 163 | 19.3704 | 20.4464 | 22.0634 | 5.6989 | 6.7890 | 5.1843 |  | cnn | candidateB |

## Output files

- `ttk_runs_fixed/topology_finetuning/candidateB_topology/candidateB_pd_mt_distances.csv` — Candidate B PD/MT per sample
- `ttk_runs_fixed/topology_finetuning/candidateB_topology/candidateB_topology_comparison.csv` — three-way comparison (CNN, GAN, CandidateB)
- `docs/topology_finetuning_candidateB_topology_eval.md` — this report

## Notes

- PD distance = bottleneck distance between persistence diagrams (computed by `ttkBottleneckDistance`).
- MT distance = Wasserstein-type merge tree distance (computed by `ttkMergeTreeDistanceMatrix`).
- Both use 160×160 speed patches at (x0=0, y0=0), consistent with the CNN/GAN baseline.
- Lower distance = SR is topologically closer to GT. Lower is better for both metrics.
- **Component-count proxy metrics** (counts of connected components in speed superlevel sets) are reported separately in `topology_finetuning_candidateB_eval.md` and do not require TTK.
- Candidate B was fine-tuned for 3 epochs from the pretrained CNN checkpoint. Results may improve with longer training.
