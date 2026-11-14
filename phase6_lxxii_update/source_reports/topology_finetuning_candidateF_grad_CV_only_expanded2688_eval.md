# candidateF_grad_CV_only_expanded2688 fine-tuning evaluation

**Generated:** 2026-10-13

**Candidate:** `candidateF_grad_CV_only_expanded2688`

**Samples evaluated:** 168

**Methods compared:** bicubic, cnn, gan, candidateF_grad_CV_only_expanded2688

> **SSIM note:** SSIM was not computed because skimage could not be imported in the current NumPy environment (`ValueError('numpy.dtype size changed, may indicate binary incompatibility. Expected 96 from C header, got 88 from PyObject')`). All SSIM values are NaN.

> Note: PD/MT distances for `candidateF_grad_CV_only_expanded2688` are N/A — TTK has not been re-run on these outputs. Values for CNN/GAN are loaded from the pre-computed merged CSV.

## Summary metrics table

| Metric | Bicubic | Cnn | Gan | Candidatef_grad_cv_only_expanded2688 |
|---|---|---|---|---|
| PSNRuv (dB) | 32.2202 | 31.1925 | 29.1380 | 32.7620 |
| SSIM on speed | N/A | N/A | N/A | N/A |
| Speed MAE (m/s) | 0.5963 | 0.6941 | 0.9026 | 0.5714 |
| Speed RMSE (m/s) | 1.0156 | 1.1078 | 1.3775 | 0.9592 |
| WPD MAE (m³/s³) | 193.5403 | 231.6709 | 310.7328 | 184.5553 |
| WPD Wasserstein-1 | 55.5103 | 45.2713 | 85.6191 | 22.6663 |
| WPD bias abs (m³/s³) | 41.2244 | 35.3439 | 78.7236 | 20.6194 |
| Gradient MAE | 0.3696 | 0.3491 | 0.3806 | 0.3111 |
| Gradient W1 | 0.3282 | 0.2329 | 0.0564 | 0.1494 |
| Gradient kurtosis abs Δ | 6.6163 | 3.7004 | 4.2010 | 3.1057 |
| PSD log-L2 | 1.3625 | 0.8335 | 0.5139 | 0.8433 |
| PSD slope abs Δ | 1.6684 | 0.9150 | 0.9482 | 1.3487 |
| Exceedance abs Δ s>5 | 0.0095 | 0.0042 | 0.0082 | 0.0022 |
| Exceedance abs Δ s>10 | 0.0085 | 0.0066 | 0.0243 | 0.0052 |
| Exceedance abs Δ s>15 | 0.0074 | 0.0062 | 0.0096 | 0.0025 |
| Exceedance abs Δ p90 | 0.0109 | 0.0103 | 0.0123 | 0.0039 |
| Component-count curve L1 | 173.1855 | 115.5278 | 124.6567 | 77.8165 |
| PD distance (TTK) | N/A | 27.4063 | 20.8641 | N/A |
| MT distance (TTK) | N/A | 5.8678 | 8.3481 | N/A |

## 7 Key evaluation questions

### Q1. Did candidateF_grad_CV_only_expanded2688 preserve CNN's direct-fidelity advantage?

- **PSNRuv**: CNN=31.1925, GAN=29.1380, candidateF_grad_CV_only_expanded2688=32.7620. Δ(candidateF_grad_CV_only_expanded2688−CNN)=+1.5695 (▲ better), improved on 168/168 samples.
- **SSIM**: CNN=N/A, GAN=N/A, candidateF_grad_CV_only_expanded2688=N/A. Δ(candidateF_grad_CV_only_expanded2688−CNN)=N/A, improved on 0/0 samples.
- **Speed MAE**: CNN=0.6941, GAN=0.9026, candidateF_grad_CV_only_expanded2688=0.5714. Δ(candidateF_grad_CV_only_expanded2688−CNN)=-0.1227 (▲ better), improved on 168/168 samples.
- **Speed RMSE**: CNN=1.1078, GAN=1.3775, candidateF_grad_CV_only_expanded2688=0.9592. Δ(candidateF_grad_CV_only_expanded2688−CNN)=-0.1485 (▲ better), improved on 168/168 samples.

### Q2. Did candidateF_grad_CV_only_expanded2688 improve scalar speed or WPD metrics?

- **WPD MAE**: CNN=231.6709, candidateF_grad_CV_only_expanded2688=184.5553. Δ=-47.1156 (▲ better), improved on 168/168 samples.
- **WPD W1**: CNN=45.2713, candidateF_grad_CV_only_expanded2688=22.6663. Δ=-22.6049 (▲ better), improved on 155/168 samples.
- **WPD bias abs**: CNN=35.3439, candidateF_grad_CV_only_expanded2688=20.6194. Δ=-14.7245 (▲ better), improved on 146/168 samples.

### Q3. Did candidateF_grad_CV_only_expanded2688 improve gradient metrics?

- **Gradient MAE**: CNN=0.3491, candidateF_grad_CV_only_expanded2688=0.3111. Δ=-0.0380 (▲ better), improved on 168/168 samples.
- **Gradient W1**: CNN=0.2329, candidateF_grad_CV_only_expanded2688=0.1494. Δ=-0.0836 (▲ better), improved on 167/168 samples.
- **Gradient kurtosis abs Δ**: CNN=3.7004, candidateF_grad_CV_only_expanded2688=3.1057. Δ=-0.5947 (▲ better), improved on 72/168 samples.

### Q4. Did candidateF_grad_CV_only_expanded2688 improve exceedance metrics, especially s>5?

- **Exceedance abs Δ s>5**: CNN=0.00424, candidateF_grad_CV_only_expanded2688=0.00217. Δ=-0.0021 (▲ better), improved on 128/168 samples.
- **Exceedance abs Δ s>10**: CNN=0.00658, candidateF_grad_CV_only_expanded2688=0.00516. Δ=-0.0014 (▲ better), improved on 105/168 samples.
- **Exceedance abs Δ s>15**: CNN=0.00621, candidateF_grad_CV_only_expanded2688=0.00247. Δ=-0.0037 (▲ better), improved on 140/168 samples.
- **Exceedance abs Δ p90**: CNN=0.01025, candidateF_grad_CV_only_expanded2688=0.00395. Δ=-0.0063 (▲ better), improved on 145/168 samples.

### Q5. Did candidateF_grad_CV_only_expanded2688 move component-count behavior toward GT?

- **Component-count curve L1**: CNN=115.5278, candidateF_grad_CV_only_expanded2688=77.8165. Δ=-37.7113 (▲ better), improved on 162/168 samples.
- **Comp-count abs Δ t5**: CNN=64.3810, candidateF_grad_CV_only_expanded2688=50.6071. Δ=-13.7738 (▲ better), improved on 146/168 samples.
- **Comp-count abs Δ t10**: CNN=187.1667, candidateF_grad_CV_only_expanded2688=145.8512. Δ=-41.3155 (▲ better), improved on 159/168 samples.
- **Comp-count abs Δ t15**: CNN=116.8095, candidateF_grad_CV_only_expanded2688=68.0476. Δ=-48.7619 (▲ better), improved on 154/168 samples.

### Q6. Did candidateF_grad_CV_only_expanded2688 improve PD or MT distances?

PD and MT distances for `candidateF_grad_CV_only_expanded2688` are not available in this evaluation (TTK was not re-run on these outputs). Pre-computed values for CNN and GAN:

- **PD distance**: CNN=27.4063, GAN=20.8641, candidateF_grad_CV_only_expanded2688=N/A (requires TTK run)
- **MT distance**: CNN=5.8678, GAN=8.3481, candidateF_grad_CV_only_expanded2688=N/A (requires TTK run)

To compute PD/MT for `candidateF_grad_CV_only_expanded2688`: convert outputs to VTI via `scripts/convert_phire_to_vti.py`, then run `scripts/compute_composite_tree_distance.py`.

### Q7. Did candidateF_grad_CV_only_expanded2688 change the adjacent clusters (samples 10–13, 90–93, 162–163)?

See `adjacent_cluster_table_candidateF_grad_CV_only_expanded2688.csv` for full detail. Summary of PSNRuv, speed MAE, and gradient MAE changes:

| Sample | Δ PSNRuv | Δ Speed MAE | Δ Grad MAE | Δ Exceed s>5 |
|---|---|---|---|---|
| 10 | 1.728 | -0.1373 | -0.0393 | -0.00246 |
| 11 | 1.655 | -0.1353 | -0.0397 | -0.00156 |
| 12 | 1.695 | -0.1367 | -0.0434 | -0.00109 |
| 13 | 1.583 | -0.1259 | -0.0385 | -0.00132 |
| 90 | 1.530 | -0.0974 | -0.0299 | -0.00405 |
| 91 | 1.548 | -0.1027 | -0.0304 | 0.00036 |
| 92 | 1.485 | -0.1095 | -0.0342 | 0.00199 |
| 93 | 1.340 | -0.1062 | -0.0362 | -0.00479 |
| 162 | 2.689 | -0.1137 | -0.0222 | -0.00093 |
| 163 | 2.497 | -0.1114 | -0.0240 | 0.00132 |

## Pairwise summary: CNN vs candidateF_grad_CV_only_expanded2688

| Metric | Mean CNN | Mean candidateF_grad_CV_only_expanded2688 | Δ | N improved | N worsened |
|---|---|---|---|---|---|
| psnruv | 31.1925 | 32.7620 | 1.5695 | 168 | 0 |
| ssim | N/A | N/A | N/A | 0 | 0 |
| speed_mae | 0.6941 | 0.5714 | -0.1227 | 168 | 0 |
| speed_rmse | 1.1078 | 0.9592 | -0.1485 | 168 | 0 |
| wpd_bias_abs | 35.3439 | 20.6194 | -14.7245 | 146 | 22 |
| wpd_mae | 231.6709 | 184.5553 | -47.1156 | 168 | 0 |
| wpd_w1 | 45.2713 | 22.6663 | -22.6049 | 155 | 13 |
| psd_log_l2 | 0.8335 | 0.8433 | 0.0098 | 94 | 74 |
| psd_slope_abs_delta | 0.9150 | 1.3487 | 0.4337 | 0 | 168 |
| grad_mae | 0.3491 | 0.3111 | -0.0380 | 168 | 0 |
| grad_w1 | 0.2329 | 0.1494 | -0.0836 | 167 | 1 |
| grad_kurtosis_abs_delta | 3.7004 | 3.1057 | -0.5947 | 72 | 96 |
| exceed_abs_t5 | 0.0042 | 0.0022 | -0.0021 | 128 | 40 |
| exceed_abs_t10 | 0.0066 | 0.0052 | -0.0014 | 105 | 63 |
| exceed_abs_t15 | 0.0062 | 0.0025 | -0.0037 | 140 | 28 |
| exceed_abs_p90 | 0.0103 | 0.0039 | -0.0063 | 145 | 23 |
| comp_curve_l1 | 115.5278 | 77.8165 | -37.7113 | 162 | 6 |
| pd_distance | N/A | N/A | N/A | 0 | 0 |
| mt_distance | N/A | N/A | N/A | 0 | 0 |

## Output files

- `ttk_runs_fixed/topology_finetuning/candidateF_grad_CV_only_expanded2688_eval/all_sample_metrics_candidateF_grad_CV_only_expanded2688.csv` — per-sample metrics for all methods
- `ttk_runs_fixed/topology_finetuning/candidateF_grad_CV_only_expanded2688_eval/pairwise_cnn_vs_candidateF_grad_CV_only_expanded2688.csv` — per-metric delta table
- `ttk_runs_fixed/topology_finetuning/candidateF_grad_CV_only_expanded2688_eval/winner_counts_candidateF_grad_CV_only_expanded2688.csv` — win counts per method
- `ttk_runs_fixed/topology_finetuning/candidateF_grad_CV_only_expanded2688_eval/adjacent_cluster_table_candidateF_grad_CV_only_expanded2688.csv` — adjacent-cluster detail
- `docs/topology_finetuning_candidateF_grad_CV_only_expanded2688_eval.md` — this report

## Notes

- PSNRuv is computed on physical [u,v] arrays using per-sample dynamic data range (GT max − GT min).
- All speed-based metrics use scalar speed = sqrt(u² + v²) in physical units (m/s).
- Exceedance fractions use GT percentile thresholds (p90/p95/p99) computed per sample.
- Component counts use 8-connectivity on speed superlevel sets.
- PD/MT distances for CNN and GAN are loaded from pre-computed merged CSV; `candidateF_grad_CV_only_expanded2688` requires a fresh TTK run.
- `candidateF_grad_CV_only_expanded2688` was fine-tuned from the pretrained CNN checkpoint. Results may change with different lambda settings or more epochs.
