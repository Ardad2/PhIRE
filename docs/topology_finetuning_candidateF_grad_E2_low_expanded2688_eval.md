# candidateF_grad_E2_low_expanded2688 fine-tuning evaluation

**Generated:** 2026-07-16

**Candidate:** `candidateF_grad_E2_low_expanded2688`

**Samples evaluated:** 168

**Methods compared:** bicubic, cnn, gan, candidateF_grad_E2_low_expanded2688

> **SSIM note:** SSIM was not computed because skimage could not be imported in the current NumPy environment (`ValueError('numpy.dtype size changed, may indicate binary incompatibility. Expected 96 from C header, got 88 from PyObject')`). All SSIM values are NaN.

> Note: PD/MT distances for `candidateF_grad_E2_low_expanded2688` are N/A — TTK has not been re-run on these outputs. Values for CNN/GAN are loaded from the pre-computed merged CSV.

## Summary metrics table

| Metric | Bicubic | Cnn | Gan | Candidatef_grad_e2_low_expanded2688 |
|---|---|---|---|---|
| PSNRuv (dB) | 32.2202 | 31.1925 | 29.1380 | 32.4949 |
| SSIM on speed | N/A | N/A | N/A | N/A |
| Speed MAE (m/s) | 0.5963 | 0.6941 | 0.9026 | 0.5899 |
| Speed RMSE (m/s) | 1.0156 | 1.1078 | 1.3775 | 0.9891 |
| WPD MAE (m³/s³) | 193.5403 | 231.6709 | 310.7328 | 190.6897 |
| WPD Wasserstein-1 | 55.5103 | 45.2713 | 85.6191 | 24.7322 |
| WPD bias abs (m³/s³) | 41.2244 | 35.3439 | 78.7236 | 22.6377 |
| Gradient MAE | 0.3696 | 0.3491 | 0.3806 | 0.3125 |
| Gradient W1 | 0.3282 | 0.2329 | 0.0564 | 0.1381 |
| Gradient kurtosis abs Δ | 6.6163 | 3.7004 | 4.2010 | 3.2601 |
| PSD log-L2 | 1.3625 | 0.8335 | 0.5139 | 0.8147 |
| PSD slope abs Δ | 1.6684 | 0.9150 | 0.9482 | 1.1385 |
| Exceedance abs Δ s>5 | 0.0095 | 0.0042 | 0.0082 | 0.0025 |
| Exceedance abs Δ s>10 | 0.0085 | 0.0066 | 0.0243 | 0.0052 |
| Exceedance abs Δ s>15 | 0.0074 | 0.0062 | 0.0096 | 0.0027 |
| Exceedance abs Δ p90 | 0.0109 | 0.0103 | 0.0123 | 0.0047 |
| Component-count curve L1 | 173.1855 | 115.5278 | 124.6567 | 81.1815 |
| PD distance (TTK) | N/A | 27.4063 | 20.8641 | N/A |
| MT distance (TTK) | N/A | 5.8678 | 8.3481 | N/A |

## 7 Key evaluation questions

### Q1. Did candidateF_grad_E2_low_expanded2688 preserve CNN's direct-fidelity advantage?

- **PSNRuv**: CNN=31.1925, GAN=29.1380, candidateF_grad_E2_low_expanded2688=32.4949. Δ(candidateF_grad_E2_low_expanded2688−CNN)=+1.3024 (▲ better), improved on 168/168 samples.
- **SSIM**: CNN=N/A, GAN=N/A, candidateF_grad_E2_low_expanded2688=N/A. Δ(candidateF_grad_E2_low_expanded2688−CNN)=N/A, improved on 0/0 samples.
- **Speed MAE**: CNN=0.6941, GAN=0.9026, candidateF_grad_E2_low_expanded2688=0.5899. Δ(candidateF_grad_E2_low_expanded2688−CNN)=-0.1042 (▲ better), improved on 168/168 samples.
- **Speed RMSE**: CNN=1.1078, GAN=1.3775, candidateF_grad_E2_low_expanded2688=0.9891. Δ(candidateF_grad_E2_low_expanded2688−CNN)=-0.1187 (▲ better), improved on 168/168 samples.

### Q2. Did candidateF_grad_E2_low_expanded2688 improve scalar speed or WPD metrics?

- **WPD MAE**: CNN=231.6709, candidateF_grad_E2_low_expanded2688=190.6897. Δ=-40.9813 (▲ better), improved on 168/168 samples.
- **WPD W1**: CNN=45.2713, candidateF_grad_E2_low_expanded2688=24.7322. Δ=-20.5391 (▲ better), improved on 156/168 samples.
- **WPD bias abs**: CNN=35.3439, candidateF_grad_E2_low_expanded2688=22.6377. Δ=-12.7062 (▲ better), improved on 141/168 samples.

### Q3. Did candidateF_grad_E2_low_expanded2688 improve gradient metrics?

- **Gradient MAE**: CNN=0.3491, candidateF_grad_E2_low_expanded2688=0.3125. Δ=-0.0365 (▲ better), improved on 168/168 samples.
- **Gradient W1**: CNN=0.2329, candidateF_grad_E2_low_expanded2688=0.1381. Δ=-0.0948 (▲ better), improved on 167/168 samples.
- **Gradient kurtosis abs Δ**: CNN=3.7004, candidateF_grad_E2_low_expanded2688=3.2601. Δ=-0.4403 (▲ better), improved on 68/168 samples.

### Q4. Did candidateF_grad_E2_low_expanded2688 improve exceedance metrics, especially s>5?

- **Exceedance abs Δ s>5**: CNN=0.00424, candidateF_grad_E2_low_expanded2688=0.00254. Δ=-0.0017 (▲ better), improved on 123/168 samples.
- **Exceedance abs Δ s>10**: CNN=0.00658, candidateF_grad_E2_low_expanded2688=0.00515. Δ=-0.0014 (▲ better), improved on 117/168 samples.
- **Exceedance abs Δ s>15**: CNN=0.00621, candidateF_grad_E2_low_expanded2688=0.00267. Δ=-0.0035 (▲ better), improved on 143/168 samples.
- **Exceedance abs Δ p90**: CNN=0.01025, candidateF_grad_E2_low_expanded2688=0.00466. Δ=-0.0056 (▲ better), improved on 141/168 samples.

### Q5. Did candidateF_grad_E2_low_expanded2688 move component-count behavior toward GT?

- **Component-count curve L1**: CNN=115.5278, candidateF_grad_E2_low_expanded2688=81.1815. Δ=-34.3462 (▲ better), improved on 166/168 samples.
- **Comp-count abs Δ t5**: CNN=64.3810, candidateF_grad_E2_low_expanded2688=50.7262. Δ=-13.6548 (▲ better), improved on 142/168 samples.
- **Comp-count abs Δ t10**: CNN=187.1667, candidateF_grad_E2_low_expanded2688=150.7440. Δ=-36.4226 (▲ better), improved on 154/168 samples.
- **Comp-count abs Δ t15**: CNN=116.8095, candidateF_grad_E2_low_expanded2688=71.9286. Δ=-44.8810 (▲ better), improved on 156/168 samples.

### Q6. Did candidateF_grad_E2_low_expanded2688 improve PD or MT distances?

PD and MT distances for `candidateF_grad_E2_low_expanded2688` are not available in this evaluation (TTK was not re-run on these outputs). Pre-computed values for CNN and GAN:

- **PD distance**: CNN=27.4063, GAN=20.8641, candidateF_grad_E2_low_expanded2688=N/A (requires TTK run)
- **MT distance**: CNN=5.8678, GAN=8.3481, candidateF_grad_E2_low_expanded2688=N/A (requires TTK run)

To compute PD/MT for `candidateF_grad_E2_low_expanded2688`: convert outputs to VTI via `scripts/convert_phire_to_vti.py`, then run `scripts/compute_composite_tree_distance.py`.

### Q7. Did candidateF_grad_E2_low_expanded2688 change the adjacent clusters (samples 10–13, 90–93, 162–163)?

See `adjacent_cluster_table_candidateF_grad_E2_low_expanded2688.csv` for full detail. Summary of PSNRuv, speed MAE, and gradient MAE changes:

| Sample | Δ PSNRuv | Δ Speed MAE | Δ Grad MAE | Δ Exceed s>5 |
|---|---|---|---|---|
| 10 | 1.515 | -0.1207 | -0.0370 | -0.00160 |
| 11 | 1.431 | -0.1171 | -0.0372 | -0.00072 |
| 12 | 1.457 | -0.1177 | -0.0403 | -0.00053 |
| 13 | 1.334 | -0.1051 | -0.0352 | -0.00075 |
| 90 | 1.289 | -0.0840 | -0.0294 | -0.00371 |
| 91 | 1.311 | -0.0890 | -0.0296 | 0.00085 |
| 92 | 1.229 | -0.0929 | -0.0338 | 0.00262 |
| 93 | 1.058 | -0.0867 | -0.0355 | -0.00572 |
| 162 | 2.343 | -0.1016 | -0.0203 | 0.00006 |
| 163 | 2.172 | -0.0985 | -0.0216 | 0.00414 |

## Pairwise summary: CNN vs candidateF_grad_E2_low_expanded2688

| Metric | Mean CNN | Mean candidateF_grad_E2_low_expanded2688 | Δ | N improved | N worsened |
|---|---|---|---|---|---|
| psnruv | 31.1925 | 32.4949 | 1.3024 | 168 | 0 |
| ssim | N/A | N/A | N/A | 0 | 0 |
| speed_mae | 0.6941 | 0.5899 | -0.1042 | 168 | 0 |
| speed_rmse | 1.1078 | 0.9891 | -0.1187 | 168 | 0 |
| wpd_bias_abs | 35.3439 | 22.6377 | -12.7062 | 141 | 27 |
| wpd_mae | 231.6709 | 190.6897 | -40.9813 | 168 | 0 |
| wpd_w1 | 45.2713 | 24.7322 | -20.5391 | 156 | 12 |
| psd_log_l2 | 0.8335 | 0.8147 | -0.0188 | 119 | 49 |
| psd_slope_abs_delta | 0.9150 | 1.1385 | 0.2235 | 3 | 165 |
| grad_mae | 0.3491 | 0.3125 | -0.0365 | 168 | 0 |
| grad_w1 | 0.2329 | 0.1381 | -0.0948 | 167 | 1 |
| grad_kurtosis_abs_delta | 3.7004 | 3.2601 | -0.4403 | 68 | 100 |
| exceed_abs_t5 | 0.0042 | 0.0025 | -0.0017 | 123 | 45 |
| exceed_abs_t10 | 0.0066 | 0.0052 | -0.0014 | 117 | 51 |
| exceed_abs_t15 | 0.0062 | 0.0027 | -0.0035 | 143 | 25 |
| exceed_abs_p90 | 0.0103 | 0.0047 | -0.0056 | 141 | 27 |
| comp_curve_l1 | 115.5278 | 81.1815 | -34.3462 | 166 | 2 |
| pd_distance | N/A | N/A | N/A | 0 | 0 |
| mt_distance | N/A | N/A | N/A | 0 | 0 |

## Output files

- `ttk_runs_fixed/topology_finetuning/candidateF_grad_E2_low_expanded2688_eval/all_sample_metrics_candidateF_grad_E2_low_expanded2688.csv` — per-sample metrics for all methods
- `ttk_runs_fixed/topology_finetuning/candidateF_grad_E2_low_expanded2688_eval/pairwise_cnn_vs_candidateF_grad_E2_low_expanded2688.csv` — per-metric delta table
- `ttk_runs_fixed/topology_finetuning/candidateF_grad_E2_low_expanded2688_eval/winner_counts_candidateF_grad_E2_low_expanded2688.csv` — win counts per method
- `ttk_runs_fixed/topology_finetuning/candidateF_grad_E2_low_expanded2688_eval/adjacent_cluster_table_candidateF_grad_E2_low_expanded2688.csv` — adjacent-cluster detail
- `docs/topology_finetuning_candidateF_grad_E2_low_expanded2688_eval.md` — this report

## Notes

- PSNRuv is computed on physical [u,v] arrays using per-sample dynamic data range (GT max − GT min).
- All speed-based metrics use scalar speed = sqrt(u² + v²) in physical units (m/s).
- Exceedance fractions use GT percentile thresholds (p90/p95/p99) computed per sample.
- Component counts use 8-connectivity on speed superlevel sets.
- PD/MT distances for CNN and GAN are loaded from pre-computed merged CSV; `candidateF_grad_E2_low_expanded2688` requires a fresh TTK run.
- `candidateF_grad_E2_low_expanded2688` was fine-tuned from the pretrained CNN checkpoint. Results may change with different lambda settings or more epochs.
