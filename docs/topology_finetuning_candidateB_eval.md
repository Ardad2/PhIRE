# Candidate B fine-tuning evaluation

**Generated:** 2026-05-12

**Candidate B:** `lambda_speed=0.01, lambda_grad=0.05, lambda_wpd=0.0, lambda_levelset=0.25, lr=1e-5, 3 epochs`

**Samples evaluated:** 168

**Methods compared:** bicubic, cnn, gan, candidateB

> **SSIM note:** SSIM was not computed because skimage could not be imported in the current NumPy environment (`ValueError('numpy.dtype size changed, may indicate binary incompatibility. Expected 96 from C header, got 88 from PyObject')`). All SSIM values are NaN.

> Note: PD/MT distances for Candidate B are N/A — TTK not re-run on pilot outputs. Values for CNN/GAN are loaded from the pre-computed merged CSV.

## Summary metrics table

| Metric | Bicubic | Cnn | Gan | Candidateb |
|---|---|---|---|---|
| PSNRuv (dB) | 32.2202 | 31.1925 | 29.1380 | 32.5394 |
| SSIM on speed | N/A | N/A | N/A | N/A |
| Speed MAE (m/s) | 0.5963 | 0.6941 | 0.9026 | 0.5885 |
| Speed RMSE (m/s) | 1.0156 | 1.1078 | 1.3775 | 0.9641 |
| WPD MAE (m³/s³) | 193.5403 | 231.6709 | 310.7328 | 191.4653 |
| WPD Wasserstein-1 | 55.5103 | 45.2713 | 85.6191 | 30.4994 |
| WPD bias abs (m³/s³) | 41.2244 | 35.3439 | 78.7236 | 21.5107 |
| Gradient MAE | 0.3696 | 0.3491 | 0.3806 | 0.3272 |
| Gradient W1 | 0.3282 | 0.2329 | 0.0564 | 0.2109 |
| Gradient kurtosis abs Δ | 6.6163 | 3.7004 | 4.2010 | 2.7542 |
| PSD log-L2 | 1.3625 | 0.8335 | 0.5139 | 0.9465 |
| PSD slope abs Δ | 1.6684 | 0.9150 | 0.9482 | 1.2079 |
| Exceedance abs Δ s>5 | 0.0095 | 0.0042 | 0.0082 | 0.0044 |
| Exceedance abs Δ s>10 | 0.0085 | 0.0066 | 0.0243 | 0.0038 |
| Exceedance abs Δ s>15 | 0.0074 | 0.0062 | 0.0096 | 0.0043 |
| Exceedance abs Δ p90 | 0.0109 | 0.0103 | 0.0123 | 0.0060 |
| Component-count curve L1 | 173.1855 | 115.5278 | 124.6567 | 105.8363 |
| PD distance (TTK) | N/A | 27.4063 | 20.8641 | N/A |
| MT distance (TTK) | N/A | 5.8678 | 8.3481 | N/A |

## 7 Key evaluation questions

### Q1. Did Candidate B preserve CNN's direct-fidelity advantage?

- **PSNRuv**: CNN=31.1925, GAN=29.1380, CandidateB=32.5394. Δ(CandidateB−CNN)=+1.3469 (▲ better), improved on 168/168 samples.
- **SSIM**: CNN=N/A, GAN=N/A, CandidateB=N/A. Δ(CandidateB−CNN)=N/A, improved on 0/0 samples.
- **Speed MAE**: CNN=0.6941, GAN=0.9026, CandidateB=0.5885. Δ(CandidateB−CNN)=-0.1056 (▲ better), improved on 168/168 samples.
- **Speed RMSE**: CNN=1.1078, GAN=1.3775, CandidateB=0.9641. Δ(CandidateB−CNN)=-0.1437 (▲ better), improved on 168/168 samples.

### Q2. Did Candidate B improve scalar speed or WPD metrics?

- **WPD MAE**: CNN=231.6709, CandidateB=191.4653. Δ=-40.2056 (▲ better), improved on 168/168 samples.
- **WPD W1**: CNN=45.2713, CandidateB=30.4994. Δ=-14.7718 (▲ better), improved on 125/168 samples.
- **WPD bias abs**: CNN=35.3439, CandidateB=21.5107. Δ=-13.8332 (▲ better), improved on 119/168 samples.

### Q3. Did Candidate B improve gradient metrics?

- **Gradient MAE**: CNN=0.3491, CandidateB=0.3272. Δ=-0.0218 (▲ better), improved on 168/168 samples.
- **Gradient W1**: CNN=0.2329, CandidateB=0.2109. Δ=-0.0221 (▲ better), improved on 143/168 samples.
- **Gradient kurtosis abs Δ**: CNN=3.7004, CandidateB=2.7542. Δ=-0.9462 (▲ better), improved on 87/168 samples.

### Q4. Did Candidate B improve exceedance metrics, especially s>5?

- **Exceedance abs Δ s>5**: CNN=0.00424, CandidateB=0.00443. Δ=+0.0002 (▼ worse), improved on 89/168 samples.
- **Exceedance abs Δ s>10**: CNN=0.00658, CandidateB=0.00382. Δ=-0.0028 (▲ better), improved on 118/168 samples.
- **Exceedance abs Δ s>15**: CNN=0.00621, CandidateB=0.00426. Δ=-0.0019 (▲ better), improved on 104/168 samples.
- **Exceedance abs Δ p90**: CNN=0.01025, CandidateB=0.00599. Δ=-0.0043 (▲ better), improved on 131/168 samples.

### Q5. Did Candidate B move component-count behavior toward GT?

- **Component-count curve L1**: CNN=115.5278, CandidateB=105.8363. Δ=-9.6915 (▲ better), improved on 147/168 samples.
- **Comp-count abs Δ t5**: CNN=64.3810, CandidateB=57.2202. Δ=-7.1607 (▲ better), improved on 111/168 samples.
- **Comp-count abs Δ t10**: CNN=187.1667, CandidateB=179.6607. Δ=-7.5060 (▲ better), improved on 99/168 samples.
- **Comp-count abs Δ t15**: CNN=116.8095, CandidateB=104.0714. Δ=-12.7381 (▲ better), improved on 122/168 samples.

### Q6. Did Candidate B improve PD or MT distances?

PD and MT distances for Candidate B are not available in this evaluation (TTK was not re-run on pilot outputs). Pre-computed values for CNN and GAN:

- **PD distance**: CNN=27.4063, GAN=20.8641, CandidateB=N/A (requires TTK run)
- **MT distance**: CNN=5.8678, GAN=8.3481, CandidateB=N/A (requires TTK run)

To compute PD/MT for Candidate B: convert outputs to VTI via `scripts/convert_phire_to_vti.py`, then run `scripts/compute_composite_tree_distance.py`.

### Q7. Did Candidate B change the adjacent clusters (samples 10–13, 90–93, 162–163)?

See `adjacent_cluster_table.csv` for full detail. Summary of PSNRuv, speed MAE, and gradient MAE changes:

| Sample | Δ PSNRuv | Δ Speed MAE | Δ Grad MAE | Δ Exceed s>5 |
|---|---|---|---|---|
| 10 | 1.514 | -0.1210 | -0.0215 | -0.00010 |
| 11 | 1.384 | -0.1136 | -0.0207 | -0.00180 |
| 12 | 1.419 | -0.1130 | -0.0238 | -0.00226 |
| 13 | 1.332 | -0.1040 | -0.0196 | -0.00415 |
| 90 | 1.194 | -0.0763 | -0.0182 | -0.00019 |
| 91 | 1.267 | -0.0827 | -0.0182 | 0.00450 |
| 92 | 1.292 | -0.0939 | -0.0208 | 0.00826 |
| 93 | 1.255 | -0.1003 | -0.0211 | 0.00509 |
| 162 | 1.964 | -0.0733 | -0.0134 | -0.00065 |
| 163 | 1.960 | -0.0742 | -0.0151 | 0.00524 |

## Pairwise summary: CNN vs Candidate B

| Metric | Mean CNN | Mean CandB | Δ | N improved | N worsened |
|---|---|---|---|---|---|
| psnruv | 31.1925 | 32.5394 | 1.3469 | 168 | 0 |
| ssim | N/A | N/A | N/A | 0 | 0 |
| speed_mae | 0.6941 | 0.5885 | -0.1056 | 168 | 0 |
| speed_rmse | 1.1078 | 0.9641 | -0.1437 | 168 | 0 |
| wpd_bias_abs | 35.3439 | 21.5107 | -13.8332 | 119 | 49 |
| wpd_mae | 231.6709 | 191.4653 | -40.2056 | 168 | 0 |
| wpd_w1 | 45.2713 | 30.4994 | -14.7718 | 125 | 43 |
| psd_log_l2 | 0.8335 | 0.9465 | 0.1130 | 4 | 164 |
| psd_slope_abs_delta | 0.9150 | 1.2079 | 0.2929 | 0 | 168 |
| grad_mae | 0.3491 | 0.3272 | -0.0218 | 168 | 0 |
| grad_w1 | 0.2329 | 0.2109 | -0.0221 | 143 | 25 |
| grad_kurtosis_abs_delta | 3.7004 | 2.7542 | -0.9462 | 87 | 81 |
| exceed_abs_t5 | 0.0042 | 0.0044 | 0.0002 | 89 | 79 |
| exceed_abs_t10 | 0.0066 | 0.0038 | -0.0028 | 118 | 50 |
| exceed_abs_t15 | 0.0062 | 0.0043 | -0.0019 | 104 | 63 |
| exceed_abs_p90 | 0.0103 | 0.0060 | -0.0043 | 131 | 37 |
| comp_curve_l1 | 115.5278 | 105.8363 | -9.6915 | 147 | 21 |
| pd_distance | N/A | N/A | N/A | 0 | 0 |
| mt_distance | N/A | N/A | N/A | 0 | 0 |

## Output files

- `ttk_runs_fixed/topology_finetuning/candidateB_eval/all_sample_metrics_candidateB.csv` — per-sample metrics for all methods
- `ttk_runs_fixed/topology_finetuning/candidateB_eval/pairwise_cnn_vs_candidateB.csv` — per-metric delta table
- `ttk_runs_fixed/topology_finetuning/candidateB_eval/winner_counts.csv` — win counts per method
- `ttk_runs_fixed/topology_finetuning/candidateB_eval/adjacent_cluster_table.csv` — adjacent-cluster detail
- `docs/topology_finetuning_candidateB_eval.md` — this report

## Notes

- PSNRuv is computed on physical [u,v] arrays using per-sample dynamic data range (GT max − GT min).
- All speed-based metrics use scalar speed = sqrt(u² + v²) in physical units (m/s).
- Exceedance fractions use GT percentile thresholds (p90/p95/p99) computed per sample.
- Component counts use 8-connectivity on speed superlevel sets.
- PD/MT distances for CNN and GAN are loaded from pre-computed merged CSV; Candidate B requires a fresh TTK run.
- Candidate B was trained for 3 epochs from the pretrained CNN checkpoint — results may improve with longer training.
