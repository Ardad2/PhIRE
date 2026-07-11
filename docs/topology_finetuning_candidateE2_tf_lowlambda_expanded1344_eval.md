# candidateE2_tf_lowlambda_expanded1344 fine-tuning evaluation

**Generated:** 2026-07-10

**Candidate:** `candidateE2_tf_lowlambda_expanded1344`

**Samples evaluated:** 168

**Methods compared:** bicubic, cnn, gan, candidateE2_tf_lowlambda_expanded1344

> **SSIM note:** SSIM was not computed because skimage could not be imported in the current NumPy environment (`ValueError('numpy.dtype size changed, may indicate binary incompatibility. Expected 96 from C header, got 88 from PyObject')`). All SSIM values are NaN.

> Note: PD/MT distances for `candidateE2_tf_lowlambda_expanded1344` are N/A — TTK has not been re-run on these outputs. Values for CNN/GAN are loaded from the pre-computed merged CSV.

## Summary metrics table

| Metric | Bicubic | Cnn | Gan | Candidatee2_tf_lowlambda_expanded1344 |
|---|---|---|---|---|
| PSNRuv (dB) | 32.2202 | 31.1925 | 29.1380 | 32.3659 |
| SSIM on speed | N/A | N/A | N/A | N/A |
| Speed MAE (m/s) | 0.5963 | 0.6941 | 0.9026 | 0.5925 |
| Speed RMSE (m/s) | 1.0156 | 1.1078 | 1.3775 | 0.9947 |
| WPD MAE (m³/s³) | 193.5403 | 231.6709 | 310.7328 | 193.1359 |
| WPD Wasserstein-1 | 55.5103 | 45.2713 | 85.6191 | 14.7417 |
| WPD bias abs (m³/s³) | 41.2244 | 35.3439 | 78.7236 | 4.7898 |
| Gradient MAE | 0.3696 | 0.3491 | 0.3806 | 0.3173 |
| Gradient W1 | 0.3282 | 0.2329 | 0.0564 | 0.1530 |
| Gradient kurtosis abs Δ | 6.6163 | 3.7004 | 4.2010 | 2.8428 |
| PSD log-L2 | 1.3625 | 0.8335 | 0.5139 | 0.8302 |
| PSD slope abs Δ | 1.6684 | 0.9150 | 0.9482 | 1.1317 |
| Exceedance abs Δ s>5 | 0.0095 | 0.0042 | 0.0082 | 0.0030 |
| Exceedance abs Δ s>10 | 0.0085 | 0.0066 | 0.0243 | 0.0030 |
| Exceedance abs Δ s>15 | 0.0074 | 0.0062 | 0.0096 | 0.0013 |
| Exceedance abs Δ p90 | 0.0109 | 0.0103 | 0.0123 | 0.0018 |
| Component-count curve L1 | 173.1855 | 115.5278 | 124.6567 | 88.8919 |
| PD distance (TTK) | N/A | 27.4063 | 20.8641 | N/A |
| MT distance (TTK) | N/A | 5.8678 | 8.3481 | N/A |

## 7 Key evaluation questions

### Q1. Did candidateE2_tf_lowlambda_expanded1344 preserve CNN's direct-fidelity advantage?

- **PSNRuv**: CNN=31.1925, GAN=29.1380, candidateE2_tf_lowlambda_expanded1344=32.3659. Δ(candidateE2_tf_lowlambda_expanded1344−CNN)=+1.1734 (▲ better), improved on 168/168 samples.
- **SSIM**: CNN=N/A, GAN=N/A, candidateE2_tf_lowlambda_expanded1344=N/A. Δ(candidateE2_tf_lowlambda_expanded1344−CNN)=N/A, improved on 0/0 samples.
- **Speed MAE**: CNN=0.6941, GAN=0.9026, candidateE2_tf_lowlambda_expanded1344=0.5925. Δ(candidateE2_tf_lowlambda_expanded1344−CNN)=-0.1015 (▲ better), improved on 168/168 samples.
- **Speed RMSE**: CNN=1.1078, GAN=1.3775, candidateE2_tf_lowlambda_expanded1344=0.9947. Δ(candidateE2_tf_lowlambda_expanded1344−CNN)=-0.1131 (▲ better), improved on 168/168 samples.

### Q2. Did candidateE2_tf_lowlambda_expanded1344 improve scalar speed or WPD metrics?

- **WPD MAE**: CNN=231.6709, candidateE2_tf_lowlambda_expanded1344=193.1359. Δ=-38.5350 (▲ better), improved on 168/168 samples.
- **WPD W1**: CNN=45.2713, candidateE2_tf_lowlambda_expanded1344=14.7417. Δ=-30.5296 (▲ better), improved on 165/168 samples.
- **WPD bias abs**: CNN=35.3439, candidateE2_tf_lowlambda_expanded1344=4.7898. Δ=-30.5541 (▲ better), improved on 155/168 samples.

### Q3. Did candidateE2_tf_lowlambda_expanded1344 improve gradient metrics?

- **Gradient MAE**: CNN=0.3491, candidateE2_tf_lowlambda_expanded1344=0.3173. Δ=-0.0318 (▲ better), improved on 168/168 samples.
- **Gradient W1**: CNN=0.2329, candidateE2_tf_lowlambda_expanded1344=0.1530. Δ=-0.0799 (▲ better), improved on 165/168 samples.
- **Gradient kurtosis abs Δ**: CNN=3.7004, candidateE2_tf_lowlambda_expanded1344=2.8428. Δ=-0.8576 (▲ better), improved on 79/168 samples.

### Q4. Did candidateE2_tf_lowlambda_expanded1344 improve exceedance metrics, especially s>5?

- **Exceedance abs Δ s>5**: CNN=0.00424, candidateE2_tf_lowlambda_expanded1344=0.00298. Δ=-0.0013 (▲ better), improved on 117/168 samples.
- **Exceedance abs Δ s>10**: CNN=0.00658, candidateE2_tf_lowlambda_expanded1344=0.00296. Δ=-0.0036 (▲ better), improved on 123/168 samples.
- **Exceedance abs Δ s>15**: CNN=0.00621, candidateE2_tf_lowlambda_expanded1344=0.00126. Δ=-0.0049 (▲ better), improved on 141/168 samples.
- **Exceedance abs Δ p90**: CNN=0.01025, candidateE2_tf_lowlambda_expanded1344=0.00177. Δ=-0.0085 (▲ better), improved on 160/168 samples.

### Q5. Did candidateE2_tf_lowlambda_expanded1344 move component-count behavior toward GT?

- **Component-count curve L1**: CNN=115.5278, candidateE2_tf_lowlambda_expanded1344=88.8919. Δ=-26.6359 (▲ better), improved on 156/168 samples.
- **Comp-count abs Δ t5**: CNN=64.3810, candidateE2_tf_lowlambda_expanded1344=56.4702. Δ=-7.9107 (▲ better), improved on 131/168 samples.
- **Comp-count abs Δ t10**: CNN=187.1667, candidateE2_tf_lowlambda_expanded1344=169.2440. Δ=-17.9226 (▲ better), improved on 127/168 samples.
- **Comp-count abs Δ t15**: CNN=116.8095, candidateE2_tf_lowlambda_expanded1344=77.1012. Δ=-39.7083 (▲ better), improved on 151/168 samples.

### Q6. Did candidateE2_tf_lowlambda_expanded1344 improve PD or MT distances?

PD and MT distances for `candidateE2_tf_lowlambda_expanded1344` are not available in this evaluation (TTK was not re-run on these outputs). Pre-computed values for CNN and GAN:

- **PD distance**: CNN=27.4063, GAN=20.8641, candidateE2_tf_lowlambda_expanded1344=N/A (requires TTK run)
- **MT distance**: CNN=5.8678, GAN=8.3481, candidateE2_tf_lowlambda_expanded1344=N/A (requires TTK run)

To compute PD/MT for `candidateE2_tf_lowlambda_expanded1344`: convert outputs to VTI via `scripts/convert_phire_to_vti.py`, then run `scripts/compute_composite_tree_distance.py`.

### Q7. Did candidateE2_tf_lowlambda_expanded1344 change the adjacent clusters (samples 10–13, 90–93, 162–163)?

See `adjacent_cluster_table_candidateE2_tf_lowlambda_expanded1344.csv` for full detail. Summary of PSNRuv, speed MAE, and gradient MAE changes:

| Sample | Δ PSNRuv | Δ Speed MAE | Δ Grad MAE | Δ Exceed s>5 |
|---|---|---|---|---|
| 10 | 1.433 | -0.1241 | -0.0320 | -0.00248 |
| 11 | 1.315 | -0.1185 | -0.0317 | -0.00358 |
| 12 | 1.362 | -0.1197 | -0.0364 | -0.00297 |
| 13 | 1.210 | -0.1047 | -0.0314 | -0.00276 |
| 90 | 1.175 | -0.0834 | -0.0252 | 0.00170 |
| 91 | 1.223 | -0.0885 | -0.0246 | 0.00752 |
| 92 | 1.129 | -0.0932 | -0.0289 | 0.00918 |
| 93 | 0.994 | -0.0913 | -0.0304 | 0.00205 |
| 162 | 2.363 | -0.1009 | -0.0201 | 0.00126 |
| 163 | 2.180 | -0.0973 | -0.0209 | 0.00722 |

## Pairwise summary: CNN vs candidateE2_tf_lowlambda_expanded1344

| Metric | Mean CNN | Mean candidateE2_tf_lowlambda_expanded1344 | Δ | N improved | N worsened |
|---|---|---|---|---|---|
| psnruv | 31.1925 | 32.3659 | 1.1734 | 168 | 0 |
| ssim | N/A | N/A | N/A | 0 | 0 |
| speed_mae | 0.6941 | 0.5925 | -0.1015 | 168 | 0 |
| speed_rmse | 1.1078 | 0.9947 | -0.1131 | 168 | 0 |
| wpd_bias_abs | 35.3439 | 4.7898 | -30.5541 | 155 | 13 |
| wpd_mae | 231.6709 | 193.1359 | -38.5350 | 168 | 0 |
| wpd_w1 | 45.2713 | 14.7417 | -30.5296 | 165 | 3 |
| psd_log_l2 | 0.8335 | 0.8302 | -0.0033 | 98 | 70 |
| psd_slope_abs_delta | 0.9150 | 1.1317 | 0.2167 | 3 | 165 |
| grad_mae | 0.3491 | 0.3173 | -0.0318 | 168 | 0 |
| grad_w1 | 0.2329 | 0.1530 | -0.0799 | 165 | 3 |
| grad_kurtosis_abs_delta | 3.7004 | 2.8428 | -0.8576 | 79 | 89 |
| exceed_abs_t5 | 0.0042 | 0.0030 | -0.0013 | 117 | 51 |
| exceed_abs_t10 | 0.0066 | 0.0030 | -0.0036 | 123 | 45 |
| exceed_abs_t15 | 0.0062 | 0.0013 | -0.0049 | 141 | 26 |
| exceed_abs_p90 | 0.0103 | 0.0018 | -0.0085 | 160 | 8 |
| comp_curve_l1 | 115.5278 | 88.8919 | -26.6359 | 156 | 12 |
| pd_distance | N/A | N/A | N/A | 0 | 0 |
| mt_distance | N/A | N/A | N/A | 0 | 0 |

## Output files

- `ttk_runs_fixed/topology_finetuning/candidateE2_tf_lowlambda_expanded1344_eval/all_sample_metrics_candidateE2_tf_lowlambda_expanded1344.csv` — per-sample metrics for all methods
- `ttk_runs_fixed/topology_finetuning/candidateE2_tf_lowlambda_expanded1344_eval/pairwise_cnn_vs_candidateE2_tf_lowlambda_expanded1344.csv` — per-metric delta table
- `ttk_runs_fixed/topology_finetuning/candidateE2_tf_lowlambda_expanded1344_eval/winner_counts_candidateE2_tf_lowlambda_expanded1344.csv` — win counts per method
- `ttk_runs_fixed/topology_finetuning/candidateE2_tf_lowlambda_expanded1344_eval/adjacent_cluster_table_candidateE2_tf_lowlambda_expanded1344.csv` — adjacent-cluster detail
- `docs/topology_finetuning_candidateE2_tf_lowlambda_expanded1344_eval.md` — this report

## Notes

- PSNRuv is computed on physical [u,v] arrays using per-sample dynamic data range (GT max − GT min).
- All speed-based metrics use scalar speed = sqrt(u² + v²) in physical units (m/s).
- Exceedance fractions use GT percentile thresholds (p90/p95/p99) computed per sample.
- Component counts use 8-connectivity on speed superlevel sets.
- PD/MT distances for CNN and GAN are loaded from pre-computed merged CSV; `candidateE2_tf_lowlambda_expanded1344` requires a fresh TTK run.
- `candidateE2_tf_lowlambda_expanded1344` was fine-tuned from the pretrained CNN checkpoint. Results may change with different lambda settings or more epochs.
