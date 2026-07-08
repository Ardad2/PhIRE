# candidateE2_fixed fine-tuning evaluation

**Generated:** 2026-07-07

**Candidate:** `candidateE2_fixed`

**Samples evaluated:** 168

**Methods compared:** bicubic, cnn, gan, candidateE2_fixed

> **SSIM note:** SSIM was not computed because skimage could not be imported in the current NumPy environment (`ValueError('numpy.dtype size changed, may indicate binary incompatibility. Expected 96 from C header, got 88 from PyObject')`). All SSIM values are NaN.

> Note: PD/MT distances for `candidateE2_fixed` are N/A — TTK has not been re-run on these outputs. Values for CNN/GAN are loaded from the pre-computed merged CSV.

## Summary metrics table

| Metric | Bicubic | Cnn | Gan | Candidatee2_fixed |
|---|---|---|---|---|
| PSNRuv (dB) | 32.2202 | 31.1925 | 29.1380 | 30.5559 |
| SSIM on speed | N/A | N/A | N/A | N/A |
| Speed MAE (m/s) | 0.5963 | 0.6941 | 0.9026 | 0.7506 |
| Speed RMSE (m/s) | 1.0156 | 1.1078 | 1.3775 | 1.1809 |
| WPD MAE (m³/s³) | 193.5403 | 231.6709 | 310.7328 | 254.3562 |
| WPD Wasserstein-1 | 55.5103 | 45.2713 | 85.6191 | 69.6564 |
| WPD bias abs (m³/s³) | 41.2244 | 35.3439 | 78.7236 | 66.2821 |
| Gradient MAE | 0.3696 | 0.3491 | 0.3806 | 0.3476 |
| Gradient W1 | 0.3282 | 0.2329 | 0.0564 | 0.1624 |
| Gradient kurtosis abs Δ | 6.6163 | 3.7004 | 4.2010 | 4.4804 |
| PSD log-L2 | 1.3625 | 0.8335 | 0.5139 | 0.7668 |
| PSD slope abs Δ | 1.6684 | 0.9150 | 0.9482 | 0.8037 |
| Exceedance abs Δ s>5 | 0.0095 | 0.0042 | 0.0082 | 0.0050 |
| Exceedance abs Δ s>10 | 0.0085 | 0.0066 | 0.0243 | 0.0135 |
| Exceedance abs Δ s>15 | 0.0074 | 0.0062 | 0.0096 | 0.0093 |
| Exceedance abs Δ p90 | 0.0109 | 0.0103 | 0.0123 | 0.0156 |
| Component-count curve L1 | 173.1855 | 115.5278 | 124.6567 | 94.6429 |
| PD distance (TTK) | N/A | 27.4063 | 20.8641 | N/A |
| MT distance (TTK) | N/A | 5.8678 | 8.3481 | N/A |

## 7 Key evaluation questions

### Q1. Did candidateE2_fixed preserve CNN's direct-fidelity advantage?

- **PSNRuv**: CNN=31.1925, GAN=29.1380, candidateE2_fixed=30.5559. Δ(candidateE2_fixed−CNN)=-0.6366 (▼ worse), improved on 0/168 samples.
- **SSIM**: CNN=N/A, GAN=N/A, candidateE2_fixed=N/A. Δ(candidateE2_fixed−CNN)=N/A, improved on 0/0 samples.
- **Speed MAE**: CNN=0.6941, GAN=0.9026, candidateE2_fixed=0.7506. Δ(candidateE2_fixed−CNN)=+0.0566 (▼ worse), improved on 0/168 samples.
- **Speed RMSE**: CNN=1.1078, GAN=1.3775, candidateE2_fixed=1.1809. Δ(candidateE2_fixed−CNN)=+0.0731 (▼ worse), improved on 0/168 samples.

### Q2. Did candidateE2_fixed improve scalar speed or WPD metrics?

- **WPD MAE**: CNN=231.6709, candidateE2_fixed=254.3562. Δ=+22.6853 (▼ worse), improved on 0/168 samples.
- **WPD W1**: CNN=45.2713, candidateE2_fixed=69.6564. Δ=+24.3851 (▼ worse), improved on 34/168 samples.
- **WPD bias abs**: CNN=35.3439, candidateE2_fixed=66.2821. Δ=+30.9382 (▼ worse), improved on 31/168 samples.

### Q3. Did candidateE2_fixed improve gradient metrics?

- **Gradient MAE**: CNN=0.3491, candidateE2_fixed=0.3476. Δ=-0.0015 (▲ better), improved on 105/168 samples.
- **Gradient W1**: CNN=0.2329, candidateE2_fixed=0.1624. Δ=-0.0705 (▲ better), improved on 168/168 samples.
- **Gradient kurtosis abs Δ**: CNN=3.7004, candidateE2_fixed=4.4804. Δ=+0.7801 (▼ worse), improved on 60/168 samples.

### Q4. Did candidateE2_fixed improve exceedance metrics, especially s>5?

- **Exceedance abs Δ s>5**: CNN=0.00424, candidateE2_fixed=0.00499. Δ=+0.0008 (▼ worse), improved on 89/168 samples.
- **Exceedance abs Δ s>10**: CNN=0.00658, candidateE2_fixed=0.01347. Δ=+0.0069 (▼ worse), improved on 32/168 samples.
- **Exceedance abs Δ s>15**: CNN=0.00621, candidateE2_fixed=0.00928. Δ=+0.0031 (▼ worse), improved on 78/168 samples.
- **Exceedance abs Δ p90**: CNN=0.01025, candidateE2_fixed=0.01560. Δ=+0.0053 (▼ worse), improved on 43/168 samples.

### Q5. Did candidateE2_fixed move component-count behavior toward GT?

- **Component-count curve L1**: CNN=115.5278, candidateE2_fixed=94.6429. Δ=-20.8849 (▲ better), improved on 168/168 samples.
- **Comp-count abs Δ t5**: CNN=64.3810, candidateE2_fixed=59.7679. Δ=-4.6131 (▲ better), improved on 135/168 samples.
- **Comp-count abs Δ t10**: CNN=187.1667, candidateE2_fixed=163.4702. Δ=-23.6964 (▲ better), improved on 159/168 samples.
- **Comp-count abs Δ t15**: CNN=116.8095, candidateE2_fixed=87.7857. Δ=-29.0238 (▲ better), improved on 160/168 samples.

### Q6. Did candidateE2_fixed improve PD or MT distances?

PD and MT distances for `candidateE2_fixed` are not available in this evaluation (TTK was not re-run on these outputs). Pre-computed values for CNN and GAN:

- **PD distance**: CNN=27.4063, GAN=20.8641, candidateE2_fixed=N/A (requires TTK run)
- **MT distance**: CNN=5.8678, GAN=8.3481, candidateE2_fixed=N/A (requires TTK run)

To compute PD/MT for `candidateE2_fixed`: convert outputs to VTI via `scripts/convert_phire_to_vti.py`, then run `scripts/compute_composite_tree_distance.py`.

### Q7. Did candidateE2_fixed change the adjacent clusters (samples 10–13, 90–93, 162–163)?

See `adjacent_cluster_table_candidateE2_fixed.csv` for full detail. Summary of PSNRuv, speed MAE, and gradient MAE changes:

| Sample | Δ PSNRuv | Δ Speed MAE | Δ Grad MAE | Δ Exceed s>5 |
|---|---|---|---|---|
| 10 | -0.502 | 0.0875 | -0.0011 | 0.00220 |
| 11 | -0.499 | 0.0862 | -0.0012 | -0.00026 |
| 12 | -0.534 | 0.0867 | -0.0018 | 0.00091 |
| 13 | -0.554 | 0.0903 | -0.0026 | -0.00101 |
| 90 | -0.702 | 0.0415 | -0.0020 | -0.00574 |
| 91 | -0.664 | 0.0412 | -0.0019 | 0.00234 |
| 92 | -0.619 | 0.0385 | -0.0030 | 0.01221 |
| 93 | -0.550 | 0.0365 | -0.0048 | -0.00017 |
| 162 | -0.566 | 0.0196 | 0.0046 | -0.00400 |
| 163 | -0.532 | 0.0182 | 0.0043 | -0.00248 |

## Pairwise summary: CNN vs candidateE2_fixed

| Metric | Mean CNN | Mean candidateE2_fixed | Δ | N improved | N worsened |
|---|---|---|---|---|---|
| psnruv | 31.1925 | 30.5559 | -0.6366 | 0 | 168 |
| ssim | N/A | N/A | N/A | 0 | 0 |
| speed_mae | 0.6941 | 0.7506 | 0.0566 | 0 | 168 |
| speed_rmse | 1.1078 | 1.1809 | 0.0731 | 0 | 168 |
| wpd_bias_abs | 35.3439 | 66.2821 | 30.9382 | 31 | 137 |
| wpd_mae | 231.6709 | 254.3562 | 22.6853 | 0 | 168 |
| wpd_w1 | 45.2713 | 69.6564 | 24.3851 | 34 | 134 |
| psd_log_l2 | 0.8335 | 0.7668 | -0.0667 | 136 | 32 |
| psd_slope_abs_delta | 0.9150 | 0.8037 | -0.1113 | 168 | 0 |
| grad_mae | 0.3491 | 0.3476 | -0.0015 | 105 | 63 |
| grad_w1 | 0.2329 | 0.1624 | -0.0705 | 168 | 0 |
| grad_kurtosis_abs_delta | 3.7004 | 4.4804 | 0.7801 | 60 | 108 |
| exceed_abs_t5 | 0.0042 | 0.0050 | 0.0008 | 89 | 79 |
| exceed_abs_t10 | 0.0066 | 0.0135 | 0.0069 | 32 | 136 |
| exceed_abs_t15 | 0.0062 | 0.0093 | 0.0031 | 78 | 90 |
| exceed_abs_p90 | 0.0103 | 0.0156 | 0.0053 | 43 | 125 |
| comp_curve_l1 | 115.5278 | 94.6429 | -20.8849 | 168 | 0 |
| pd_distance | N/A | N/A | N/A | 0 | 0 |
| mt_distance | N/A | N/A | N/A | 0 | 0 |

## Output files

- `ttk_runs_fixed/topology_finetuning/candidateE2_fixed_eval/all_sample_metrics_candidateE2_fixed.csv` — per-sample metrics for all methods
- `ttk_runs_fixed/topology_finetuning/candidateE2_fixed_eval/pairwise_cnn_vs_candidateE2_fixed.csv` — per-metric delta table
- `ttk_runs_fixed/topology_finetuning/candidateE2_fixed_eval/winner_counts_candidateE2_fixed.csv` — win counts per method
- `ttk_runs_fixed/topology_finetuning/candidateE2_fixed_eval/adjacent_cluster_table_candidateE2_fixed.csv` — adjacent-cluster detail
- `docs/topology_finetuning_candidateE2_fixed_eval.md` — this report

## Notes

- PSNRuv is computed on physical [u,v] arrays using per-sample dynamic data range (GT max − GT min).
- All speed-based metrics use scalar speed = sqrt(u² + v²) in physical units (m/s).
- Exceedance fractions use GT percentile thresholds (p90/p95/p99) computed per sample.
- Component counts use 8-connectivity on speed superlevel sets.
- PD/MT distances for CNN and GAN are loaded from pre-computed merged CSV; `candidateE2_fixed` requires a fresh TTK run.
- `candidateE2_fixed` was fine-tuned from the pretrained CNN checkpoint. Results may change with different lambda settings or more epochs.
