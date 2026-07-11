# candidateUV_plus_E2_tf_lowlambda_expanded1344 fine-tuning evaluation

**Generated:** 2026-07-11

**Candidate:** `candidateUV_plus_E2_tf_lowlambda_expanded1344`

**Samples evaluated:** 168

**Methods compared:** bicubic, cnn, gan, candidateUV_plus_E2_tf_lowlambda_expanded1344

> **SSIM note:** SSIM was not computed because skimage could not be imported in the current NumPy environment (`ValueError('numpy.dtype size changed, may indicate binary incompatibility. Expected 96 from C header, got 88 from PyObject')`). All SSIM values are NaN.

> Note: PD/MT distances for `candidateUV_plus_E2_tf_lowlambda_expanded1344` are N/A — TTK has not been re-run on these outputs. Values for CNN/GAN are loaded from the pre-computed merged CSV.

## Summary metrics table

| Metric | Bicubic | Cnn | Gan | Candidateuv_plus_e2_tf_lowlambda_expanded1344 |
|---|---|---|---|---|
| PSNRuv (dB) | 32.2202 | 31.1925 | 29.1380 | 32.1573 |
| SSIM on speed | N/A | N/A | N/A | N/A |
| Speed MAE (m/s) | 0.5963 | 0.6941 | 0.9026 | 0.6118 |
| Speed RMSE (m/s) | 1.0156 | 1.1078 | 1.3775 | 1.0235 |
| WPD MAE (m³/s³) | 193.5403 | 231.6709 | 310.7328 | 200.2107 |
| WPD Wasserstein-1 | 55.5103 | 45.2713 | 85.6191 | 22.7899 |
| WPD bias abs (m³/s³) | 41.2244 | 35.3439 | 78.7236 | 18.5849 |
| Gradient MAE | 0.3696 | 0.3491 | 0.3806 | 0.3226 |
| Gradient W1 | 0.3282 | 0.2329 | 0.0564 | 0.1572 |
| Gradient kurtosis abs Δ | 6.6163 | 3.7004 | 4.2010 | 2.8830 |
| PSD log-L2 | 1.3625 | 0.8335 | 0.5139 | 0.8099 |
| PSD slope abs Δ | 1.6684 | 0.9150 | 0.9482 | 1.0799 |
| Exceedance abs Δ s>5 | 0.0095 | 0.0042 | 0.0082 | 0.0025 |
| Exceedance abs Δ s>10 | 0.0085 | 0.0066 | 0.0243 | 0.0040 |
| Exceedance abs Δ s>15 | 0.0074 | 0.0062 | 0.0096 | 0.0024 |
| Exceedance abs Δ p90 | 0.0109 | 0.0103 | 0.0123 | 0.0044 |
| Component-count curve L1 | 173.1855 | 115.5278 | 124.6567 | 88.8442 |
| PD distance (TTK) | N/A | 27.4063 | 20.8641 | N/A |
| MT distance (TTK) | N/A | 5.8678 | 8.3481 | N/A |

## 7 Key evaluation questions

### Q1. Did candidateUV_plus_E2_tf_lowlambda_expanded1344 preserve CNN's direct-fidelity advantage?

- **PSNRuv**: CNN=31.1925, GAN=29.1380, candidateUV_plus_E2_tf_lowlambda_expanded1344=32.1573. Δ(candidateUV_plus_E2_tf_lowlambda_expanded1344−CNN)=+0.9648 (▲ better), improved on 168/168 samples.
- **SSIM**: CNN=N/A, GAN=N/A, candidateUV_plus_E2_tf_lowlambda_expanded1344=N/A. Δ(candidateUV_plus_E2_tf_lowlambda_expanded1344−CNN)=N/A, improved on 0/0 samples.
- **Speed MAE**: CNN=0.6941, GAN=0.9026, candidateUV_plus_E2_tf_lowlambda_expanded1344=0.6118. Δ(candidateUV_plus_E2_tf_lowlambda_expanded1344−CNN)=-0.0823 (▲ better), improved on 168/168 samples.
- **Speed RMSE**: CNN=1.1078, GAN=1.3775, candidateUV_plus_E2_tf_lowlambda_expanded1344=1.0235. Δ(candidateUV_plus_E2_tf_lowlambda_expanded1344−CNN)=-0.0843 (▲ better), improved on 154/168 samples.

### Q2. Did candidateUV_plus_E2_tf_lowlambda_expanded1344 improve scalar speed or WPD metrics?

- **WPD MAE**: CNN=231.6709, candidateUV_plus_E2_tf_lowlambda_expanded1344=200.2107. Δ=-31.4602 (▲ better), improved on 168/168 samples.
- **WPD W1**: CNN=45.2713, candidateUV_plus_E2_tf_lowlambda_expanded1344=22.7899. Δ=-22.4813 (▲ better), improved on 140/168 samples.
- **WPD bias abs**: CNN=35.3439, candidateUV_plus_E2_tf_lowlambda_expanded1344=18.5849. Δ=-16.7590 (▲ better), improved on 124/168 samples.

### Q3. Did candidateUV_plus_E2_tf_lowlambda_expanded1344 improve gradient metrics?

- **Gradient MAE**: CNN=0.3491, candidateUV_plus_E2_tf_lowlambda_expanded1344=0.3226. Δ=-0.0265 (▲ better), improved on 168/168 samples.
- **Gradient W1**: CNN=0.2329, candidateUV_plus_E2_tf_lowlambda_expanded1344=0.1572. Δ=-0.0758 (▲ better), improved on 161/168 samples.
- **Gradient kurtosis abs Δ**: CNN=3.7004, candidateUV_plus_E2_tf_lowlambda_expanded1344=2.8830. Δ=-0.8173 (▲ better), improved on 83/168 samples.

### Q4. Did candidateUV_plus_E2_tf_lowlambda_expanded1344 improve exceedance metrics, especially s>5?

- **Exceedance abs Δ s>5**: CNN=0.00424, candidateUV_plus_E2_tf_lowlambda_expanded1344=0.00250. Δ=-0.0017 (▲ better), improved on 122/168 samples.
- **Exceedance abs Δ s>10**: CNN=0.00658, candidateUV_plus_E2_tf_lowlambda_expanded1344=0.00404. Δ=-0.0025 (▲ better), improved on 106/168 samples.
- **Exceedance abs Δ s>15**: CNN=0.00621, candidateUV_plus_E2_tf_lowlambda_expanded1344=0.00243. Δ=-0.0038 (▲ better), improved on 127/168 samples.
- **Exceedance abs Δ p90**: CNN=0.01025, candidateUV_plus_E2_tf_lowlambda_expanded1344=0.00435. Δ=-0.0059 (▲ better), improved on 134/168 samples.

### Q5. Did candidateUV_plus_E2_tf_lowlambda_expanded1344 move component-count behavior toward GT?

- **Component-count curve L1**: CNN=115.5278, candidateUV_plus_E2_tf_lowlambda_expanded1344=88.8442. Δ=-26.6835 (▲ better), improved on 156/168 samples.
- **Comp-count abs Δ t5**: CNN=64.3810, candidateUV_plus_E2_tf_lowlambda_expanded1344=58.9643. Δ=-5.4167 (▲ better), improved on 117/168 samples.
- **Comp-count abs Δ t10**: CNN=187.1667, candidateUV_plus_E2_tf_lowlambda_expanded1344=169.2560. Δ=-17.9107 (▲ better), improved on 119/168 samples.
- **Comp-count abs Δ t15**: CNN=116.8095, candidateUV_plus_E2_tf_lowlambda_expanded1344=76.1905. Δ=-40.6190 (▲ better), improved on 154/168 samples.

### Q6. Did candidateUV_plus_E2_tf_lowlambda_expanded1344 improve PD or MT distances?

PD and MT distances for `candidateUV_plus_E2_tf_lowlambda_expanded1344` are not available in this evaluation (TTK was not re-run on these outputs). Pre-computed values for CNN and GAN:

- **PD distance**: CNN=27.4063, GAN=20.8641, candidateUV_plus_E2_tf_lowlambda_expanded1344=N/A (requires TTK run)
- **MT distance**: CNN=5.8678, GAN=8.3481, candidateUV_plus_E2_tf_lowlambda_expanded1344=N/A (requires TTK run)

To compute PD/MT for `candidateUV_plus_E2_tf_lowlambda_expanded1344`: convert outputs to VTI via `scripts/convert_phire_to_vti.py`, then run `scripts/compute_composite_tree_distance.py`.

### Q7. Did candidateUV_plus_E2_tf_lowlambda_expanded1344 change the adjacent clusters (samples 10–13, 90–93, 162–163)?

See `adjacent_cluster_table_candidateUV_plus_E2_tf_lowlambda_expanded1344.csv` for full detail. Summary of PSNRuv, speed MAE, and gradient MAE changes:

| Sample | Δ PSNRuv | Δ Speed MAE | Δ Grad MAE | Δ Exceed s>5 |
|---|---|---|---|---|
| 10 | 1.225 | -0.1040 | -0.0246 | -0.00297 |
| 11 | 1.107 | -0.0981 | -0.0249 | -0.00170 |
| 12 | 1.178 | -0.0999 | -0.0298 | -0.00126 |
| 13 | 1.034 | -0.0845 | -0.0243 | -0.00076 |
| 90 | 0.954 | -0.0664 | -0.0220 | 0.00022 |
| 91 | 1.021 | -0.0716 | -0.0211 | 0.00528 |
| 92 | 0.906 | -0.0748 | -0.0250 | 0.00646 |
| 93 | 0.776 | -0.0709 | -0.0256 | -0.00081 |
| 162 | 2.163 | -0.0941 | -0.0192 | -0.00354 |
| 163 | 1.950 | -0.0899 | -0.0193 | 0.00422 |

## Pairwise summary: CNN vs candidateUV_plus_E2_tf_lowlambda_expanded1344

| Metric | Mean CNN | Mean candidateUV_plus_E2_tf_lowlambda_expanded1344 | Δ | N improved | N worsened |
|---|---|---|---|---|---|
| psnruv | 31.1925 | 32.1573 | 0.9648 | 168 | 0 |
| ssim | N/A | N/A | N/A | 0 | 0 |
| speed_mae | 0.6941 | 0.6118 | -0.0823 | 168 | 0 |
| speed_rmse | 1.1078 | 1.0235 | -0.0843 | 154 | 14 |
| wpd_bias_abs | 35.3439 | 18.5849 | -16.7590 | 124 | 44 |
| wpd_mae | 231.6709 | 200.2107 | -31.4602 | 168 | 0 |
| wpd_w1 | 45.2713 | 22.7899 | -22.4813 | 140 | 28 |
| psd_log_l2 | 0.8335 | 0.8099 | -0.0236 | 107 | 61 |
| psd_slope_abs_delta | 0.9150 | 1.0799 | 0.1649 | 33 | 135 |
| grad_mae | 0.3491 | 0.3226 | -0.0265 | 168 | 0 |
| grad_w1 | 0.2329 | 0.1572 | -0.0758 | 161 | 7 |
| grad_kurtosis_abs_delta | 3.7004 | 2.8830 | -0.8173 | 83 | 85 |
| exceed_abs_t5 | 0.0042 | 0.0025 | -0.0017 | 122 | 46 |
| exceed_abs_t10 | 0.0066 | 0.0040 | -0.0025 | 106 | 62 |
| exceed_abs_t15 | 0.0062 | 0.0024 | -0.0038 | 127 | 41 |
| exceed_abs_p90 | 0.0103 | 0.0044 | -0.0059 | 134 | 34 |
| comp_curve_l1 | 115.5278 | 88.8442 | -26.6835 | 156 | 12 |
| pd_distance | N/A | N/A | N/A | 0 | 0 |
| mt_distance | N/A | N/A | N/A | 0 | 0 |

## Output files

- `ttk_runs_fixed/topology_finetuning/candidateUV_plus_E2_tf_lowlambda_expanded1344_eval/all_sample_metrics_candidateUV_plus_E2_tf_lowlambda_expanded1344.csv` — per-sample metrics for all methods
- `ttk_runs_fixed/topology_finetuning/candidateUV_plus_E2_tf_lowlambda_expanded1344_eval/pairwise_cnn_vs_candidateUV_plus_E2_tf_lowlambda_expanded1344.csv` — per-metric delta table
- `ttk_runs_fixed/topology_finetuning/candidateUV_plus_E2_tf_lowlambda_expanded1344_eval/winner_counts_candidateUV_plus_E2_tf_lowlambda_expanded1344.csv` — win counts per method
- `ttk_runs_fixed/topology_finetuning/candidateUV_plus_E2_tf_lowlambda_expanded1344_eval/adjacent_cluster_table_candidateUV_plus_E2_tf_lowlambda_expanded1344.csv` — adjacent-cluster detail
- `docs/topology_finetuning_candidateUV_plus_E2_tf_lowlambda_expanded1344_eval.md` — this report

## Notes

- PSNRuv is computed on physical [u,v] arrays using per-sample dynamic data range (GT max − GT min).
- All speed-based metrics use scalar speed = sqrt(u² + v²) in physical units (m/s).
- Exceedance fractions use GT percentile thresholds (p90/p95/p99) computed per sample.
- Component counts use 8-connectivity on speed superlevel sets.
- PD/MT distances for CNN and GAN are loaded from pre-computed merged CSV; `candidateUV_plus_E2_tf_lowlambda_expanded1344` requires a fresh TTK run.
- `candidateUV_plus_E2_tf_lowlambda_expanded1344` was fine-tuned from the pretrained CNN checkpoint. Results may change with different lambda settings or more epochs.
