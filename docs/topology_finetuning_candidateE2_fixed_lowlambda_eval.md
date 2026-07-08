# candidateE2_fixed_lowlambda fine-tuning evaluation

**Generated:** 2026-07-07

**Candidate:** `candidateE2_fixed_lowlambda`

**Samples evaluated:** 168

**Methods compared:** bicubic, cnn, gan, candidateE2_fixed_lowlambda

> **SSIM note:** SSIM was not computed because skimage could not be imported in the current NumPy environment (`ValueError('numpy.dtype size changed, may indicate binary incompatibility. Expected 96 from C header, got 88 from PyObject')`). All SSIM values are NaN.

> Note: PD/MT distances for `candidateE2_fixed_lowlambda` are N/A — TTK has not been re-run on these outputs. Values for CNN/GAN are loaded from the pre-computed merged CSV.

## Summary metrics table

| Metric | Bicubic | Cnn | Gan | Candidatee2_fixed_lowlambda |
|---|---|---|---|---|
| PSNRuv (dB) | 32.2202 | 31.1925 | 29.1380 | 31.1573 |
| SSIM on speed | N/A | N/A | N/A | N/A |
| Speed MAE (m/s) | 0.5963 | 0.6941 | 0.9026 | 0.6952 |
| Speed RMSE (m/s) | 1.0156 | 1.1078 | 1.3775 | 1.1099 |
| WPD MAE (m³/s³) | 193.5403 | 231.6709 | 310.7328 | 232.1962 |
| WPD Wasserstein-1 | 55.5103 | 45.2713 | 85.6191 | 29.8082 |
| WPD bias abs (m³/s³) | 41.2244 | 35.3439 | 78.7236 | 12.5397 |
| Gradient MAE | 0.3696 | 0.3491 | 0.3806 | 0.3457 |
| Gradient W1 | 0.3282 | 0.2329 | 0.0564 | 0.2163 |
| Gradient kurtosis abs Δ | 6.6163 | 3.7004 | 4.2010 | 3.9368 |
| PSD log-L2 | 1.3625 | 0.8335 | 0.5139 | 0.8130 |
| PSD slope abs Δ | 1.6684 | 0.9150 | 0.9482 | 0.8806 |
| Exceedance abs Δ s>5 | 0.0095 | 0.0042 | 0.0082 | 0.0044 |
| Exceedance abs Δ s>10 | 0.0085 | 0.0066 | 0.0243 | 0.0050 |
| Exceedance abs Δ s>15 | 0.0074 | 0.0062 | 0.0096 | 0.0037 |
| Exceedance abs Δ p90 | 0.0109 | 0.0103 | 0.0123 | 0.0054 |
| Component-count curve L1 | 173.1855 | 115.5278 | 124.6567 | 109.4405 |
| PD distance (TTK) | N/A | 27.4063 | 20.8641 | N/A |
| MT distance (TTK) | N/A | 5.8678 | 8.3481 | N/A |

## 7 Key evaluation questions

### Q1. Did candidateE2_fixed_lowlambda preserve CNN's direct-fidelity advantage?

- **PSNRuv**: CNN=31.1925, GAN=29.1380, candidateE2_fixed_lowlambda=31.1573. Δ(candidateE2_fixed_lowlambda−CNN)=-0.0352 (▼ worse), improved on 32/168 samples.
- **SSIM**: CNN=N/A, GAN=N/A, candidateE2_fixed_lowlambda=N/A. Δ(candidateE2_fixed_lowlambda−CNN)=N/A, improved on 0/0 samples.
- **Speed MAE**: CNN=0.6941, GAN=0.9026, candidateE2_fixed_lowlambda=0.6952. Δ(candidateE2_fixed_lowlambda−CNN)=+0.0011 (▼ worse), improved on 88/168 samples.
- **Speed RMSE**: CNN=1.1078, GAN=1.3775, candidateE2_fixed_lowlambda=1.1099. Δ(candidateE2_fixed_lowlambda−CNN)=+0.0022 (▼ worse), improved on 71/168 samples.

### Q2. Did candidateE2_fixed_lowlambda improve scalar speed or WPD metrics?

- **WPD MAE**: CNN=231.6709, candidateE2_fixed_lowlambda=232.1962. Δ=+0.5253 (▼ worse), improved on 78/168 samples.
- **WPD W1**: CNN=45.2713, candidateE2_fixed_lowlambda=29.8082. Δ=-15.4631 (▲ better), improved on 138/168 samples.
- **WPD bias abs**: CNN=35.3439, candidateE2_fixed_lowlambda=12.5397. Δ=-22.8043 (▲ better), improved on 152/168 samples.

### Q3. Did candidateE2_fixed_lowlambda improve gradient metrics?

- **Gradient MAE**: CNN=0.3491, candidateE2_fixed_lowlambda=0.3457. Δ=-0.0034 (▲ better), improved on 161/168 samples.
- **Gradient W1**: CNN=0.2329, candidateE2_fixed_lowlambda=0.2163. Δ=-0.0167 (▲ better), improved on 146/168 samples.
- **Gradient kurtosis abs Δ**: CNN=3.7004, candidateE2_fixed_lowlambda=3.9368. Δ=+0.2364 (▼ worse), improved on 71/168 samples.

### Q4. Did candidateE2_fixed_lowlambda improve exceedance metrics, especially s>5?

- **Exceedance abs Δ s>5**: CNN=0.00424, candidateE2_fixed_lowlambda=0.00444. Δ=+0.0002 (▼ worse), improved on 78/168 samples.
- **Exceedance abs Δ s>10**: CNN=0.00658, candidateE2_fixed_lowlambda=0.00501. Δ=-0.0016 (▲ better), improved on 109/168 samples.
- **Exceedance abs Δ s>15**: CNN=0.00621, candidateE2_fixed_lowlambda=0.00369. Δ=-0.0025 (▲ better), improved on 132/168 samples.
- **Exceedance abs Δ p90**: CNN=0.01025, candidateE2_fixed_lowlambda=0.00538. Δ=-0.0049 (▲ better), improved on 143/168 samples.

### Q5. Did candidateE2_fixed_lowlambda move component-count behavior toward GT?

- **Component-count curve L1**: CNN=115.5278, candidateE2_fixed_lowlambda=109.4405. Δ=-6.0873 (▲ better), improved on 140/168 samples.
- **Comp-count abs Δ t5**: CNN=64.3810, candidateE2_fixed_lowlambda=64.3571. Δ=-0.0238 (▲ better), improved on 71/168 samples.
- **Comp-count abs Δ t10**: CNN=187.1667, candidateE2_fixed_lowlambda=184.0357. Δ=-3.1310 (▲ better), improved on 99/168 samples.
- **Comp-count abs Δ t15**: CNN=116.8095, candidateE2_fixed_lowlambda=107.0298. Δ=-9.7798 (▲ better), improved on 137/168 samples.

### Q6. Did candidateE2_fixed_lowlambda improve PD or MT distances?

PD and MT distances for `candidateE2_fixed_lowlambda` are not available in this evaluation (TTK was not re-run on these outputs). Pre-computed values for CNN and GAN:

- **PD distance**: CNN=27.4063, GAN=20.8641, candidateE2_fixed_lowlambda=N/A (requires TTK run)
- **MT distance**: CNN=5.8678, GAN=8.3481, candidateE2_fixed_lowlambda=N/A (requires TTK run)

To compute PD/MT for `candidateE2_fixed_lowlambda`: convert outputs to VTI via `scripts/convert_phire_to_vti.py`, then run `scripts/compute_composite_tree_distance.py`.

### Q7. Did candidateE2_fixed_lowlambda change the adjacent clusters (samples 10–13, 90–93, 162–163)?

See `adjacent_cluster_table_candidateE2_fixed_lowlambda.csv` for full detail. Summary of PSNRuv, speed MAE, and gradient MAE changes:

| Sample | Δ PSNRuv | Δ Speed MAE | Δ Grad MAE | Δ Exceed s>5 |
|---|---|---|---|---|
| 10 | 0.107 | -0.0118 | -0.0040 | 0.00075 |
| 11 | 0.086 | -0.0099 | -0.0032 | -0.00001 |
| 12 | 0.060 | -0.0076 | -0.0032 | -0.00003 |
| 13 | 0.026 | -0.0034 | -0.0025 | -0.00076 |
| 90 | -0.070 | 0.0013 | -0.0017 | 0.00238 |
| 91 | -0.064 | 0.0009 | -0.0018 | 0.00234 |
| 92 | -0.044 | -0.0013 | -0.0020 | 0.00166 |
| 93 | -0.011 | -0.0048 | -0.0025 | 0.00172 |
| 162 | 0.277 | -0.0094 | -0.0050 | 0.00703 |
| 163 | 0.263 | -0.0093 | -0.0048 | 0.00965 |

## Pairwise summary: CNN vs candidateE2_fixed_lowlambda

| Metric | Mean CNN | Mean candidateE2_fixed_lowlambda | Δ | N improved | N worsened |
|---|---|---|---|---|---|
| psnruv | 31.1925 | 31.1573 | -0.0352 | 32 | 136 |
| ssim | N/A | N/A | N/A | 0 | 0 |
| speed_mae | 0.6941 | 0.6952 | 0.0011 | 88 | 80 |
| speed_rmse | 1.1078 | 1.1099 | 0.0022 | 71 | 97 |
| wpd_bias_abs | 35.3439 | 12.5397 | -22.8043 | 152 | 16 |
| wpd_mae | 231.6709 | 232.1962 | 0.5253 | 78 | 90 |
| wpd_w1 | 45.2713 | 29.8082 | -15.4631 | 138 | 30 |
| psd_log_l2 | 0.8335 | 0.8130 | -0.0205 | 114 | 54 |
| psd_slope_abs_delta | 0.9150 | 0.8806 | -0.0344 | 131 | 37 |
| grad_mae | 0.3491 | 0.3457 | -0.0034 | 161 | 7 |
| grad_w1 | 0.2329 | 0.2163 | -0.0167 | 146 | 22 |
| grad_kurtosis_abs_delta | 3.7004 | 3.9368 | 0.2364 | 71 | 97 |
| exceed_abs_t5 | 0.0042 | 0.0044 | 0.0002 | 78 | 90 |
| exceed_abs_t10 | 0.0066 | 0.0050 | -0.0016 | 109 | 59 |
| exceed_abs_t15 | 0.0062 | 0.0037 | -0.0025 | 132 | 36 |
| exceed_abs_p90 | 0.0103 | 0.0054 | -0.0049 | 143 | 25 |
| comp_curve_l1 | 115.5278 | 109.4405 | -6.0873 | 140 | 27 |
| pd_distance | N/A | N/A | N/A | 0 | 0 |
| mt_distance | N/A | N/A | N/A | 0 | 0 |

## Output files

- `ttk_runs_fixed/topology_finetuning/candidateE2_fixed_lowlambda_eval/all_sample_metrics_candidateE2_fixed_lowlambda.csv` — per-sample metrics for all methods
- `ttk_runs_fixed/topology_finetuning/candidateE2_fixed_lowlambda_eval/pairwise_cnn_vs_candidateE2_fixed_lowlambda.csv` — per-metric delta table
- `ttk_runs_fixed/topology_finetuning/candidateE2_fixed_lowlambda_eval/winner_counts_candidateE2_fixed_lowlambda.csv` — win counts per method
- `ttk_runs_fixed/topology_finetuning/candidateE2_fixed_lowlambda_eval/adjacent_cluster_table_candidateE2_fixed_lowlambda.csv` — adjacent-cluster detail
- `docs/topology_finetuning_candidateE2_fixed_lowlambda_eval.md` — this report

## Notes

- PSNRuv is computed on physical [u,v] arrays using per-sample dynamic data range (GT max − GT min).
- All speed-based metrics use scalar speed = sqrt(u² + v²) in physical units (m/s).
- Exceedance fractions use GT percentile thresholds (p90/p95/p99) computed per sample.
- Component counts use 8-connectivity on speed superlevel sets.
- PD/MT distances for CNN and GAN are loaded from pre-computed merged CSV; `candidateE2_fixed_lowlambda` requires a fresh TTK run.
- `candidateE2_fixed_lowlambda` was fine-tuned from the pretrained CNN checkpoint. Results may change with different lambda settings or more epochs.
