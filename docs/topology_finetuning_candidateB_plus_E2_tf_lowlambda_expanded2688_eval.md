# candidateB_plus_E2_tf_lowlambda_expanded2688 fine-tuning evaluation

**Generated:** 2026-07-10

**Candidate:** `candidateB_plus_E2_tf_lowlambda_expanded2688`

**Samples evaluated:** 168

**Methods compared:** bicubic, cnn, gan, candidateB_plus_E2_tf_lowlambda_expanded2688

> **SSIM note:** SSIM was not computed because skimage could not be imported in the current NumPy environment (`ValueError('numpy.dtype size changed, may indicate binary incompatibility. Expected 96 from C header, got 88 from PyObject')`). All SSIM values are NaN.

> Note: PD/MT distances for `candidateB_plus_E2_tf_lowlambda_expanded2688` are N/A — TTK has not been re-run on these outputs. Values for CNN/GAN are loaded from the pre-computed merged CSV.

## Summary metrics table

| Metric | Bicubic | Cnn | Gan | Candidateb_plus_e2_tf_lowlambda_expanded2688 |
|---|---|---|---|---|
| PSNRuv (dB) | 32.2202 | 31.1925 | 29.1380 | 32.6889 |
| SSIM on speed | N/A | N/A | N/A | N/A |
| Speed MAE (m/s) | 0.5963 | 0.6941 | 0.9026 | 0.5684 |
| Speed RMSE (m/s) | 1.0156 | 1.1078 | 1.3775 | 0.9629 |
| WPD MAE (m³/s³) | 193.5403 | 231.6709 | 310.7328 | 183.4170 |
| WPD Wasserstein-1 | 55.5103 | 45.2713 | 85.6191 | 16.0868 |
| WPD bias abs (m³/s³) | 41.2244 | 35.3439 | 78.7236 | 12.6851 |
| Gradient MAE | 0.3696 | 0.3491 | 0.3806 | 0.3120 |
| Gradient W1 | 0.3282 | 0.2329 | 0.0564 | 0.1543 |
| Gradient kurtosis abs Δ | 6.6163 | 3.7004 | 4.2010 | 2.9747 |
| PSD log-L2 | 1.3625 | 0.8335 | 0.5139 | 0.8509 |
| PSD slope abs Δ | 1.6684 | 0.9150 | 0.9482 | 1.2029 |
| Exceedance abs Δ s>5 | 0.0095 | 0.0042 | 0.0082 | 0.0020 |
| Exceedance abs Δ s>10 | 0.0085 | 0.0066 | 0.0243 | 0.0026 |
| Exceedance abs Δ s>15 | 0.0074 | 0.0062 | 0.0096 | 0.0018 |
| Exceedance abs Δ p90 | 0.0109 | 0.0103 | 0.0123 | 0.0028 |
| Component-count curve L1 | 173.1855 | 115.5278 | 124.6567 | 88.0060 |
| PD distance (TTK) | N/A | 27.4063 | 20.8641 | N/A |
| MT distance (TTK) | N/A | 5.8678 | 8.3481 | N/A |

## 7 Key evaluation questions

### Q1. Did candidateB_plus_E2_tf_lowlambda_expanded2688 preserve CNN's direct-fidelity advantage?

- **PSNRuv**: CNN=31.1925, GAN=29.1380, candidateB_plus_E2_tf_lowlambda_expanded2688=32.6889. Δ(candidateB_plus_E2_tf_lowlambda_expanded2688−CNN)=+1.4964 (▲ better), improved on 168/168 samples.
- **SSIM**: CNN=N/A, GAN=N/A, candidateB_plus_E2_tf_lowlambda_expanded2688=N/A. Δ(candidateB_plus_E2_tf_lowlambda_expanded2688−CNN)=N/A, improved on 0/0 samples.
- **Speed MAE**: CNN=0.6941, GAN=0.9026, candidateB_plus_E2_tf_lowlambda_expanded2688=0.5684. Δ(candidateB_plus_E2_tf_lowlambda_expanded2688−CNN)=-0.1256 (▲ better), improved on 168/168 samples.
- **Speed RMSE**: CNN=1.1078, GAN=1.3775, candidateB_plus_E2_tf_lowlambda_expanded2688=0.9629. Δ(candidateB_plus_E2_tf_lowlambda_expanded2688−CNN)=-0.1448 (▲ better), improved on 168/168 samples.

### Q2. Did candidateB_plus_E2_tf_lowlambda_expanded2688 improve scalar speed or WPD metrics?

- **WPD MAE**: CNN=231.6709, candidateB_plus_E2_tf_lowlambda_expanded2688=183.4170. Δ=-48.2539 (▲ better), improved on 168/168 samples.
- **WPD W1**: CNN=45.2713, candidateB_plus_E2_tf_lowlambda_expanded2688=16.0868. Δ=-29.1844 (▲ better), improved on 157/168 samples.
- **WPD bias abs**: CNN=35.3439, candidateB_plus_E2_tf_lowlambda_expanded2688=12.6851. Δ=-22.6588 (▲ better), improved on 140/168 samples.

### Q3. Did candidateB_plus_E2_tf_lowlambda_expanded2688 improve gradient metrics?

- **Gradient MAE**: CNN=0.3491, candidateB_plus_E2_tf_lowlambda_expanded2688=0.3120. Δ=-0.0371 (▲ better), improved on 168/168 samples.
- **Gradient W1**: CNN=0.2329, candidateB_plus_E2_tf_lowlambda_expanded2688=0.1543. Δ=-0.0787 (▲ better), improved on 162/168 samples.
- **Gradient kurtosis abs Δ**: CNN=3.7004, candidateB_plus_E2_tf_lowlambda_expanded2688=2.9747. Δ=-0.7257 (▲ better), improved on 72/168 samples.

### Q4. Did candidateB_plus_E2_tf_lowlambda_expanded2688 improve exceedance metrics, especially s>5?

- **Exceedance abs Δ s>5**: CNN=0.00424, candidateB_plus_E2_tf_lowlambda_expanded2688=0.00203. Δ=-0.0022 (▲ better), improved on 132/168 samples.
- **Exceedance abs Δ s>10**: CNN=0.00658, candidateB_plus_E2_tf_lowlambda_expanded2688=0.00264. Δ=-0.0039 (▲ better), improved on 134/168 samples.
- **Exceedance abs Δ s>15**: CNN=0.00621, candidateB_plus_E2_tf_lowlambda_expanded2688=0.00181. Δ=-0.0044 (▲ better), improved on 141/168 samples.
- **Exceedance abs Δ p90**: CNN=0.01025, candidateB_plus_E2_tf_lowlambda_expanded2688=0.00279. Δ=-0.0075 (▲ better), improved on 148/168 samples.

### Q5. Did candidateB_plus_E2_tf_lowlambda_expanded2688 move component-count behavior toward GT?

- **Component-count curve L1**: CNN=115.5278, candidateB_plus_E2_tf_lowlambda_expanded2688=88.0060. Δ=-27.5218 (▲ better), improved on 155/168 samples.
- **Comp-count abs Δ t5**: CNN=64.3810, candidateB_plus_E2_tf_lowlambda_expanded2688=56.1667. Δ=-8.2143 (▲ better), improved on 135/168 samples.
- **Comp-count abs Δ t10**: CNN=187.1667, candidateB_plus_E2_tf_lowlambda_expanded2688=167.5893. Δ=-19.5774 (▲ better), improved on 134/168 samples.
- **Comp-count abs Δ t15**: CNN=116.8095, candidateB_plus_E2_tf_lowlambda_expanded2688=77.4702. Δ=-39.3393 (▲ better), improved on 150/168 samples.

### Q6. Did candidateB_plus_E2_tf_lowlambda_expanded2688 improve PD or MT distances?

PD and MT distances for `candidateB_plus_E2_tf_lowlambda_expanded2688` are not available in this evaluation (TTK was not re-run on these outputs). Pre-computed values for CNN and GAN:

- **PD distance**: CNN=27.4063, GAN=20.8641, candidateB_plus_E2_tf_lowlambda_expanded2688=N/A (requires TTK run)
- **MT distance**: CNN=5.8678, GAN=8.3481, candidateB_plus_E2_tf_lowlambda_expanded2688=N/A (requires TTK run)

To compute PD/MT for `candidateB_plus_E2_tf_lowlambda_expanded2688`: convert outputs to VTI via `scripts/convert_phire_to_vti.py`, then run `scripts/compute_composite_tree_distance.py`.

### Q7. Did candidateB_plus_E2_tf_lowlambda_expanded2688 change the adjacent clusters (samples 10–13, 90–93, 162–163)?

See `adjacent_cluster_table_candidateB_plus_E2_tf_lowlambda_expanded2688.csv` for full detail. Summary of PSNRuv, speed MAE, and gradient MAE changes:

| Sample | Δ PSNRuv | Δ Speed MAE | Δ Grad MAE | Δ Exceed s>5 |
|---|---|---|---|---|
| 10 | 1.737 | -0.1469 | -0.0373 | -0.00322 |
| 11 | 1.640 | -0.1434 | -0.0378 | -0.00287 |
| 12 | 1.660 | -0.1433 | -0.0412 | -0.00232 |
| 13 | 1.522 | -0.1307 | -0.0364 | -0.00278 |
| 90 | 1.465 | -0.1001 | -0.0291 | -0.00087 |
| 91 | 1.493 | -0.1061 | -0.0295 | 0.00376 |
| 92 | 1.433 | -0.1127 | -0.0337 | 0.00489 |
| 93 | 1.270 | -0.1100 | -0.0351 | -0.00188 |
| 162 | 2.573 | -0.1162 | -0.0217 | -0.00488 |
| 163 | 2.414 | -0.1144 | -0.0234 | -0.00205 |

## Pairwise summary: CNN vs candidateB_plus_E2_tf_lowlambda_expanded2688

| Metric | Mean CNN | Mean candidateB_plus_E2_tf_lowlambda_expanded2688 | Δ | N improved | N worsened |
|---|---|---|---|---|---|
| psnruv | 31.1925 | 32.6889 | 1.4964 | 168 | 0 |
| ssim | N/A | N/A | N/A | 0 | 0 |
| speed_mae | 0.6941 | 0.5684 | -0.1256 | 168 | 0 |
| speed_rmse | 1.1078 | 0.9629 | -0.1448 | 168 | 0 |
| wpd_bias_abs | 35.3439 | 12.6851 | -22.6588 | 140 | 28 |
| wpd_mae | 231.6709 | 183.4170 | -48.2539 | 168 | 0 |
| wpd_w1 | 45.2713 | 16.0868 | -29.1844 | 157 | 11 |
| psd_log_l2 | 0.8335 | 0.8509 | 0.0173 | 86 | 82 |
| psd_slope_abs_delta | 0.9150 | 1.2029 | 0.2879 | 0 | 168 |
| grad_mae | 0.3491 | 0.3120 | -0.0371 | 168 | 0 |
| grad_w1 | 0.2329 | 0.1543 | -0.0787 | 162 | 6 |
| grad_kurtosis_abs_delta | 3.7004 | 2.9747 | -0.7257 | 72 | 96 |
| exceed_abs_t5 | 0.0042 | 0.0020 | -0.0022 | 132 | 36 |
| exceed_abs_t10 | 0.0066 | 0.0026 | -0.0039 | 134 | 34 |
| exceed_abs_t15 | 0.0062 | 0.0018 | -0.0044 | 141 | 27 |
| exceed_abs_p90 | 0.0103 | 0.0028 | -0.0075 | 148 | 20 |
| comp_curve_l1 | 115.5278 | 88.0060 | -27.5218 | 155 | 13 |
| pd_distance | N/A | N/A | N/A | 0 | 0 |
| mt_distance | N/A | N/A | N/A | 0 | 0 |

## Output files

- `ttk_runs_fixed/topology_finetuning/candidateB_plus_E2_tf_lowlambda_expanded2688_eval/all_sample_metrics_candidateB_plus_E2_tf_lowlambda_expanded2688.csv` — per-sample metrics for all methods
- `ttk_runs_fixed/topology_finetuning/candidateB_plus_E2_tf_lowlambda_expanded2688_eval/pairwise_cnn_vs_candidateB_plus_E2_tf_lowlambda_expanded2688.csv` — per-metric delta table
- `ttk_runs_fixed/topology_finetuning/candidateB_plus_E2_tf_lowlambda_expanded2688_eval/winner_counts_candidateB_plus_E2_tf_lowlambda_expanded2688.csv` — win counts per method
- `ttk_runs_fixed/topology_finetuning/candidateB_plus_E2_tf_lowlambda_expanded2688_eval/adjacent_cluster_table_candidateB_plus_E2_tf_lowlambda_expanded2688.csv` — adjacent-cluster detail
- `docs/topology_finetuning_candidateB_plus_E2_tf_lowlambda_expanded2688_eval.md` — this report

## Notes

- PSNRuv is computed on physical [u,v] arrays using per-sample dynamic data range (GT max − GT min).
- All speed-based metrics use scalar speed = sqrt(u² + v²) in physical units (m/s).
- Exceedance fractions use GT percentile thresholds (p90/p95/p99) computed per sample.
- Component counts use 8-connectivity on speed superlevel sets.
- PD/MT distances for CNN and GAN are loaded from pre-computed merged CSV; `candidateB_plus_E2_tf_lowlambda_expanded2688` requires a fresh TTK run.
- `candidateB_plus_E2_tf_lowlambda_expanded2688` was fine-tuned from the pretrained CNN checkpoint. Results may change with different lambda settings or more epochs.
