# candidateE2_tf_lowlambda_expanded2688 fine-tuning evaluation

**Generated:** 2026-07-10

**Candidate:** `candidateE2_tf_lowlambda_expanded2688`

**Samples evaluated:** 168

**Methods compared:** bicubic, cnn, gan, candidateE2_tf_lowlambda_expanded2688

> **SSIM note:** SSIM was not computed because skimage could not be imported in the current NumPy environment (`ValueError('numpy.dtype size changed, may indicate binary incompatibility. Expected 96 from C header, got 88 from PyObject')`). All SSIM values are NaN.

> Note: PD/MT distances for `candidateE2_tf_lowlambda_expanded2688` are N/A — TTK has not been re-run on these outputs. Values for CNN/GAN are loaded from the pre-computed merged CSV.

## Summary metrics table

| Metric | Bicubic | Cnn | Gan | Candidatee2_tf_lowlambda_expanded2688 |
|---|---|---|---|---|
| PSNRuv (dB) | 32.2202 | 31.1925 | 29.1380 | 32.7329 |
| SSIM on speed | N/A | N/A | N/A | N/A |
| Speed MAE (m/s) | 0.5963 | 0.6941 | 0.9026 | 0.5649 |
| Speed RMSE (m/s) | 1.0156 | 1.1078 | 1.3775 | 0.9566 |
| WPD MAE (m³/s³) | 193.5403 | 231.6709 | 310.7328 | 182.9106 |
| WPD Wasserstein-1 | 55.5103 | 45.2713 | 85.6191 | 12.9098 |
| WPD bias abs (m³/s³) | 41.2244 | 35.3439 | 78.7236 | 6.7038 |
| Gradient MAE | 0.3696 | 0.3491 | 0.3806 | 0.3116 |
| Gradient W1 | 0.3282 | 0.2329 | 0.0564 | 0.1582 |
| Gradient kurtosis abs Δ | 6.6163 | 3.7004 | 4.2010 | 2.9537 |
| PSD log-L2 | 1.3625 | 0.8335 | 0.5139 | 0.8589 |
| PSD slope abs Δ | 1.6684 | 0.9150 | 0.9482 | 1.2509 |
| Exceedance abs Δ s>5 | 0.0095 | 0.0042 | 0.0082 | 0.0022 |
| Exceedance abs Δ s>10 | 0.0085 | 0.0066 | 0.0243 | 0.0024 |
| Exceedance abs Δ s>15 | 0.0074 | 0.0062 | 0.0096 | 0.0015 |
| Exceedance abs Δ p90 | 0.0109 | 0.0103 | 0.0123 | 0.0017 |
| Component-count curve L1 | 173.1855 | 115.5278 | 124.6567 | 91.8690 |
| PD distance (TTK) | N/A | 27.4063 | 20.8641 | N/A |
| MT distance (TTK) | N/A | 5.8678 | 8.3481 | N/A |

## 7 Key evaluation questions

### Q1. Did candidateE2_tf_lowlambda_expanded2688 preserve CNN's direct-fidelity advantage?

- **PSNRuv**: CNN=31.1925, GAN=29.1380, candidateE2_tf_lowlambda_expanded2688=32.7329. Δ(candidateE2_tf_lowlambda_expanded2688−CNN)=+1.5404 (▲ better), improved on 168/168 samples.
- **SSIM**: CNN=N/A, GAN=N/A, candidateE2_tf_lowlambda_expanded2688=N/A. Δ(candidateE2_tf_lowlambda_expanded2688−CNN)=N/A, improved on 0/0 samples.
- **Speed MAE**: CNN=0.6941, GAN=0.9026, candidateE2_tf_lowlambda_expanded2688=0.5649. Δ(candidateE2_tf_lowlambda_expanded2688−CNN)=-0.1292 (▲ better), improved on 168/168 samples.
- **Speed RMSE**: CNN=1.1078, GAN=1.3775, candidateE2_tf_lowlambda_expanded2688=0.9566. Δ(candidateE2_tf_lowlambda_expanded2688−CNN)=-0.1511 (▲ better), improved on 168/168 samples.

### Q2. Did candidateE2_tf_lowlambda_expanded2688 improve scalar speed or WPD metrics?

- **WPD MAE**: CNN=231.6709, candidateE2_tf_lowlambda_expanded2688=182.9106. Δ=-48.7603 (▲ better), improved on 168/168 samples.
- **WPD W1**: CNN=45.2713, candidateE2_tf_lowlambda_expanded2688=12.9098. Δ=-32.3615 (▲ better), improved on 166/168 samples.
- **WPD bias abs**: CNN=35.3439, candidateE2_tf_lowlambda_expanded2688=6.7038. Δ=-28.6401 (▲ better), improved on 148/168 samples.

### Q3. Did candidateE2_tf_lowlambda_expanded2688 improve gradient metrics?

- **Gradient MAE**: CNN=0.3491, candidateE2_tf_lowlambda_expanded2688=0.3116. Δ=-0.0375 (▲ better), improved on 168/168 samples.
- **Gradient W1**: CNN=0.2329, candidateE2_tf_lowlambda_expanded2688=0.1582. Δ=-0.0748 (▲ better), improved on 165/168 samples.
- **Gradient kurtosis abs Δ**: CNN=3.7004, candidateE2_tf_lowlambda_expanded2688=2.9537. Δ=-0.7467 (▲ better), improved on 80/168 samples.

### Q4. Did candidateE2_tf_lowlambda_expanded2688 improve exceedance metrics, especially s>5?

- **Exceedance abs Δ s>5**: CNN=0.00424, candidateE2_tf_lowlambda_expanded2688=0.00222. Δ=-0.0020 (▲ better), improved on 131/168 samples.
- **Exceedance abs Δ s>10**: CNN=0.00658, candidateE2_tf_lowlambda_expanded2688=0.00241. Δ=-0.0042 (▲ better), improved on 126/168 samples.
- **Exceedance abs Δ s>15**: CNN=0.00621, candidateE2_tf_lowlambda_expanded2688=0.00151. Δ=-0.0047 (▲ better), improved on 140/168 samples.
- **Exceedance abs Δ p90**: CNN=0.01025, candidateE2_tf_lowlambda_expanded2688=0.00165. Δ=-0.0086 (▲ better), improved on 155/168 samples.

### Q5. Did candidateE2_tf_lowlambda_expanded2688 move component-count behavior toward GT?

- **Component-count curve L1**: CNN=115.5278, candidateE2_tf_lowlambda_expanded2688=91.8690. Δ=-23.6587 (▲ better), improved on 154/168 samples.
- **Comp-count abs Δ t5**: CNN=64.3810, candidateE2_tf_lowlambda_expanded2688=56.5536. Δ=-7.8274 (▲ better), improved on 127/168 samples.
- **Comp-count abs Δ t10**: CNN=187.1667, candidateE2_tf_lowlambda_expanded2688=175.9821. Δ=-11.1845 (▲ better), improved on 114/168 samples.
- **Comp-count abs Δ t15**: CNN=116.8095, candidateE2_tf_lowlambda_expanded2688=82.2619. Δ=-34.5476 (▲ better), improved on 144/168 samples.

### Q6. Did candidateE2_tf_lowlambda_expanded2688 improve PD or MT distances?

PD and MT distances for `candidateE2_tf_lowlambda_expanded2688` are not available in this evaluation (TTK was not re-run on these outputs). Pre-computed values for CNN and GAN:

- **PD distance**: CNN=27.4063, GAN=20.8641, candidateE2_tf_lowlambda_expanded2688=N/A (requires TTK run)
- **MT distance**: CNN=5.8678, GAN=8.3481, candidateE2_tf_lowlambda_expanded2688=N/A (requires TTK run)

To compute PD/MT for `candidateE2_tf_lowlambda_expanded2688`: convert outputs to VTI via `scripts/convert_phire_to_vti.py`, then run `scripts/compute_composite_tree_distance.py`.

### Q7. Did candidateE2_tf_lowlambda_expanded2688 change the adjacent clusters (samples 10–13, 90–93, 162–163)?

See `adjacent_cluster_table_candidateE2_tf_lowlambda_expanded2688.csv` for full detail. Summary of PSNRuv, speed MAE, and gradient MAE changes:

| Sample | Δ PSNRuv | Δ Speed MAE | Δ Grad MAE | Δ Exceed s>5 |
|---|---|---|---|---|
| 10 | 1.767 | -0.1495 | -0.0384 | -0.00254 |
| 11 | 1.675 | -0.1463 | -0.0390 | -0.00368 |
| 12 | 1.696 | -0.1461 | -0.0425 | -0.00317 |
| 13 | 1.562 | -0.1331 | -0.0377 | -0.00346 |
| 90 | 1.491 | -0.1021 | -0.0292 | 0.00012 |
| 91 | 1.517 | -0.1078 | -0.0296 | 0.00510 |
| 92 | 1.462 | -0.1148 | -0.0339 | 0.00633 |
| 93 | 1.318 | -0.1133 | -0.0356 | -0.00018 |
| 162 | 2.595 | -0.1175 | -0.0222 | -0.00250 |
| 163 | 2.431 | -0.1150 | -0.0237 | 0.00208 |

## Pairwise summary: CNN vs candidateE2_tf_lowlambda_expanded2688

| Metric | Mean CNN | Mean candidateE2_tf_lowlambda_expanded2688 | Δ | N improved | N worsened |
|---|---|---|---|---|---|
| psnruv | 31.1925 | 32.7329 | 1.5404 | 168 | 0 |
| ssim | N/A | N/A | N/A | 0 | 0 |
| speed_mae | 0.6941 | 0.5649 | -0.1292 | 168 | 0 |
| speed_rmse | 1.1078 | 0.9566 | -0.1511 | 168 | 0 |
| wpd_bias_abs | 35.3439 | 6.7038 | -28.6401 | 148 | 20 |
| wpd_mae | 231.6709 | 182.9106 | -48.7603 | 168 | 0 |
| wpd_w1 | 45.2713 | 12.9098 | -32.3615 | 166 | 2 |
| psd_log_l2 | 0.8335 | 0.8589 | 0.0253 | 79 | 89 |
| psd_slope_abs_delta | 0.9150 | 1.2509 | 0.3358 | 0 | 168 |
| grad_mae | 0.3491 | 0.3116 | -0.0375 | 168 | 0 |
| grad_w1 | 0.2329 | 0.1582 | -0.0748 | 165 | 3 |
| grad_kurtosis_abs_delta | 3.7004 | 2.9537 | -0.7467 | 80 | 88 |
| exceed_abs_t5 | 0.0042 | 0.0022 | -0.0020 | 131 | 37 |
| exceed_abs_t10 | 0.0066 | 0.0024 | -0.0042 | 126 | 42 |
| exceed_abs_t15 | 0.0062 | 0.0015 | -0.0047 | 140 | 28 |
| exceed_abs_p90 | 0.0103 | 0.0017 | -0.0086 | 155 | 13 |
| comp_curve_l1 | 115.5278 | 91.8690 | -23.6587 | 154 | 14 |
| pd_distance | N/A | N/A | N/A | 0 | 0 |
| mt_distance | N/A | N/A | N/A | 0 | 0 |

## Output files

- `ttk_runs_fixed/topology_finetuning/candidateE2_tf_lowlambda_expanded2688_eval/all_sample_metrics_candidateE2_tf_lowlambda_expanded2688.csv` — per-sample metrics for all methods
- `ttk_runs_fixed/topology_finetuning/candidateE2_tf_lowlambda_expanded2688_eval/pairwise_cnn_vs_candidateE2_tf_lowlambda_expanded2688.csv` — per-metric delta table
- `ttk_runs_fixed/topology_finetuning/candidateE2_tf_lowlambda_expanded2688_eval/winner_counts_candidateE2_tf_lowlambda_expanded2688.csv` — win counts per method
- `ttk_runs_fixed/topology_finetuning/candidateE2_tf_lowlambda_expanded2688_eval/adjacent_cluster_table_candidateE2_tf_lowlambda_expanded2688.csv` — adjacent-cluster detail
- `docs/topology_finetuning_candidateE2_tf_lowlambda_expanded2688_eval.md` — this report

## Notes

- PSNRuv is computed on physical [u,v] arrays using per-sample dynamic data range (GT max − GT min).
- All speed-based metrics use scalar speed = sqrt(u² + v²) in physical units (m/s).
- Exceedance fractions use GT percentile thresholds (p90/p95/p99) computed per sample.
- Component counts use 8-connectivity on speed superlevel sets.
- PD/MT distances for CNN and GAN are loaded from pre-computed merged CSV; `candidateE2_tf_lowlambda_expanded2688` requires a fresh TTK run.
- `candidateE2_tf_lowlambda_expanded2688` was fine-tuned from the pretrained CNN checkpoint. Results may change with different lambda settings or more epochs.
