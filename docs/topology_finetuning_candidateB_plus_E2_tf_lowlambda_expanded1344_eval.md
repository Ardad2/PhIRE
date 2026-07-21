# candidateB_plus_E2_tf_lowlambda_expanded1344 fine-tuning evaluation

**Generated:** 2026-07-10

**Candidate:** `candidateB_plus_E2_tf_lowlambda_expanded1344`

**Samples evaluated:** 168

**Methods compared:** bicubic, cnn, gan, candidateB_plus_E2_tf_lowlambda_expanded1344

> **SSIM note:** SSIM was not computed because skimage could not be imported in the current NumPy environment (`ValueError('numpy.dtype size changed, may indicate binary incompatibility. Expected 96 from C header, got 88 from PyObject')`). All SSIM values are NaN.

> Note: PD/MT distances for `candidateB_plus_E2_tf_lowlambda_expanded1344` are N/A — TTK has not been re-run on these outputs. Values for CNN/GAN are loaded from the pre-computed merged CSV.

## Summary metrics table

| Metric | Bicubic | Cnn | Gan | Candidateb_plus_e2_tf_lowlambda_expanded1344 |
|---|---|---|---|---|
| PSNRuv (dB) | 32.2202 | 31.1925 | 29.1380 | 32.4480 |
| SSIM on speed | N/A | N/A | N/A | N/A |
| Speed MAE (m/s) | 0.5963 | 0.6941 | 0.9026 | 0.5884 |
| Speed RMSE (m/s) | 1.0156 | 1.1078 | 1.3775 | 0.9852 |
| WPD MAE (m³/s³) | 193.5403 | 231.6709 | 310.7328 | 191.8304 |
| WPD Wasserstein-1 | 55.5103 | 45.2713 | 85.6191 | 19.3128 |
| WPD bias abs (m³/s³) | 41.2244 | 35.3439 | 78.7236 | 11.5541 |
| Gradient MAE | 0.3696 | 0.3491 | 0.3806 | 0.3180 |
| Gradient W1 | 0.3282 | 0.2329 | 0.0564 | 0.1654 |
| Gradient kurtosis abs Δ | 6.6163 | 3.7004 | 4.2010 | 2.9293 |
| PSD log-L2 | 1.3625 | 0.8335 | 0.5139 | 0.8560 |
| PSD slope abs Δ | 1.6684 | 0.9150 | 0.9482 | 1.1891 |
| Exceedance abs Δ s>5 | 0.0095 | 0.0042 | 0.0082 | 0.0029 |
| Exceedance abs Δ s>10 | 0.0085 | 0.0066 | 0.0243 | 0.0027 |
| Exceedance abs Δ s>15 | 0.0074 | 0.0062 | 0.0096 | 0.0023 |
| Exceedance abs Δ p90 | 0.0109 | 0.0103 | 0.0123 | 0.0031 |
| Component-count curve L1 | 173.1855 | 115.5278 | 124.6567 | 91.1538 |
| PD distance (TTK) | N/A | 27.4063 | 20.8641 | N/A |
| MT distance (TTK) | N/A | 5.8678 | 8.3481 | N/A |

## 7 Key evaluation questions

### Q1. Did candidateB_plus_E2_tf_lowlambda_expanded1344 preserve CNN's direct-fidelity advantage?

- **PSNRuv**: CNN=31.1925, GAN=29.1380, candidateB_plus_E2_tf_lowlambda_expanded1344=32.4480. Δ(candidateB_plus_E2_tf_lowlambda_expanded1344−CNN)=+1.2555 (▲ better), improved on 168/168 samples.
- **SSIM**: CNN=N/A, GAN=N/A, candidateB_plus_E2_tf_lowlambda_expanded1344=N/A. Δ(candidateB_plus_E2_tf_lowlambda_expanded1344−CNN)=N/A, improved on 0/0 samples.
- **Speed MAE**: CNN=0.6941, GAN=0.9026, candidateB_plus_E2_tf_lowlambda_expanded1344=0.5884. Δ(candidateB_plus_E2_tf_lowlambda_expanded1344−CNN)=-0.1057 (▲ better), improved on 168/168 samples.
- **Speed RMSE**: CNN=1.1078, GAN=1.3775, candidateB_plus_E2_tf_lowlambda_expanded1344=0.9852. Δ(candidateB_plus_E2_tf_lowlambda_expanded1344−CNN)=-0.1225 (▲ better), improved on 168/168 samples.

### Q2. Did candidateB_plus_E2_tf_lowlambda_expanded1344 improve scalar speed or WPD metrics?

- **WPD MAE**: CNN=231.6709, candidateB_plus_E2_tf_lowlambda_expanded1344=191.8304. Δ=-39.8405 (▲ better), improved on 168/168 samples.
- **WPD W1**: CNN=45.2713, candidateB_plus_E2_tf_lowlambda_expanded1344=19.3128. Δ=-25.9585 (▲ better), improved on 139/168 samples.
- **WPD bias abs**: CNN=35.3439, candidateB_plus_E2_tf_lowlambda_expanded1344=11.5541. Δ=-23.7898 (▲ better), improved on 124/168 samples.

### Q3. Did candidateB_plus_E2_tf_lowlambda_expanded1344 improve gradient metrics?

- **Gradient MAE**: CNN=0.3491, candidateB_plus_E2_tf_lowlambda_expanded1344=0.3180. Δ=-0.0310 (▲ better), improved on 168/168 samples.
- **Gradient W1**: CNN=0.2329, candidateB_plus_E2_tf_lowlambda_expanded1344=0.1654. Δ=-0.0676 (▲ better), improved on 155/168 samples.
- **Gradient kurtosis abs Δ**: CNN=3.7004, candidateB_plus_E2_tf_lowlambda_expanded1344=2.9293. Δ=-0.7711 (▲ better), improved on 73/168 samples.

### Q4. Did candidateB_plus_E2_tf_lowlambda_expanded1344 improve exceedance metrics, especially s>5?

- **Exceedance abs Δ s>5**: CNN=0.00424, candidateB_plus_E2_tf_lowlambda_expanded1344=0.00290. Δ=-0.0013 (▲ better), improved on 118/168 samples.
- **Exceedance abs Δ s>10**: CNN=0.00658, candidateB_plus_E2_tf_lowlambda_expanded1344=0.00274. Δ=-0.0038 (▲ better), improved on 123/168 samples.
- **Exceedance abs Δ s>15**: CNN=0.00621, candidateB_plus_E2_tf_lowlambda_expanded1344=0.00230. Δ=-0.0039 (▲ better), improved on 124/168 samples.
- **Exceedance abs Δ p90**: CNN=0.01025, candidateB_plus_E2_tf_lowlambda_expanded1344=0.00315. Δ=-0.0071 (▲ better), improved on 136/168 samples.

### Q5. Did candidateB_plus_E2_tf_lowlambda_expanded1344 move component-count behavior toward GT?

- **Component-count curve L1**: CNN=115.5278, candidateB_plus_E2_tf_lowlambda_expanded1344=91.1538. Δ=-24.3740 (▲ better), improved on 153/168 samples.
- **Comp-count abs Δ t5**: CNN=64.3810, candidateB_plus_E2_tf_lowlambda_expanded1344=56.8929. Δ=-7.4881 (▲ better), improved on 127/168 samples.
- **Comp-count abs Δ t10**: CNN=187.1667, candidateB_plus_E2_tf_lowlambda_expanded1344=171.1250. Δ=-16.0417 (▲ better), improved on 122/168 samples.
- **Comp-count abs Δ t15**: CNN=116.8095, candidateB_plus_E2_tf_lowlambda_expanded1344=80.4107. Δ=-36.3988 (▲ better), improved on 149/168 samples.

### Q6. Did candidateB_plus_E2_tf_lowlambda_expanded1344 improve PD or MT distances?

PD and MT distances for `candidateB_plus_E2_tf_lowlambda_expanded1344` are not available in this evaluation (TTK was not re-run on these outputs). Pre-computed values for CNN and GAN:

- **PD distance**: CNN=27.4063, GAN=20.8641, candidateB_plus_E2_tf_lowlambda_expanded1344=N/A (requires TTK run)
- **MT distance**: CNN=5.8678, GAN=8.3481, candidateB_plus_E2_tf_lowlambda_expanded1344=N/A (requires TTK run)

To compute PD/MT for `candidateB_plus_E2_tf_lowlambda_expanded1344`: convert outputs to VTI via `scripts/convert_phire_to_vti.py`, then run `scripts/compute_composite_tree_distance.py`.

### Q7. Did candidateB_plus_E2_tf_lowlambda_expanded1344 change the adjacent clusters (samples 10–13, 90–93, 162–163)?

See `adjacent_cluster_table_candidateB_plus_E2_tf_lowlambda_expanded1344.csv` for full detail. Summary of PSNRuv, speed MAE, and gradient MAE changes:

| Sample | Δ PSNRuv | Δ Speed MAE | Δ Grad MAE | Δ Exceed s>5 |
|---|---|---|---|---|
| 10 | 1.524 | -0.1289 | -0.0300 | -0.00230 |
| 11 | 1.400 | -0.1235 | -0.0306 | -0.00358 |
| 12 | 1.437 | -0.1237 | -0.0351 | -0.00310 |
| 13 | 1.294 | -0.1088 | -0.0303 | -0.00296 |
| 90 | 1.228 | -0.0842 | -0.0241 | 0.00375 |
| 91 | 1.279 | -0.0893 | -0.0237 | 0.00927 |
| 92 | 1.208 | -0.0957 | -0.0279 | 0.01065 |
| 93 | 1.079 | -0.0954 | -0.0295 | 0.00306 |
| 162 | 2.484 | -0.1048 | -0.0201 | -0.00469 |
| 163 | 2.319 | -0.1023 | -0.0212 | 0.00158 |

## Pairwise summary: CNN vs candidateB_plus_E2_tf_lowlambda_expanded1344

| Metric | Mean CNN | Mean candidateB_plus_E2_tf_lowlambda_expanded1344 | Δ | N improved | N worsened |
|---|---|---|---|---|---|
| psnruv | 31.1925 | 32.4480 | 1.2555 | 168 | 0 |
| ssim | N/A | N/A | N/A | 0 | 0 |
| speed_mae | 0.6941 | 0.5884 | -0.1057 | 168 | 0 |
| speed_rmse | 1.1078 | 0.9852 | -0.1225 | 168 | 0 |
| wpd_bias_abs | 35.3439 | 11.5541 | -23.7898 | 124 | 44 |
| wpd_mae | 231.6709 | 191.8304 | -39.8405 | 168 | 0 |
| wpd_w1 | 45.2713 | 19.3128 | -25.9585 | 139 | 29 |
| psd_log_l2 | 0.8335 | 0.8560 | 0.0225 | 76 | 92 |
| psd_slope_abs_delta | 0.9150 | 1.1891 | 0.2741 | 0 | 168 |
| grad_mae | 0.3491 | 0.3180 | -0.0310 | 168 | 0 |
| grad_w1 | 0.2329 | 0.1654 | -0.0676 | 155 | 13 |
| grad_kurtosis_abs_delta | 3.7004 | 2.9293 | -0.7711 | 73 | 95 |
| exceed_abs_t5 | 0.0042 | 0.0029 | -0.0013 | 118 | 50 |
| exceed_abs_t10 | 0.0066 | 0.0027 | -0.0038 | 123 | 45 |
| exceed_abs_t15 | 0.0062 | 0.0023 | -0.0039 | 124 | 44 |
| exceed_abs_p90 | 0.0103 | 0.0031 | -0.0071 | 136 | 32 |
| comp_curve_l1 | 115.5278 | 91.1538 | -24.3740 | 153 | 15 |
| pd_distance | N/A | N/A | N/A | 0 | 0 |
| mt_distance | N/A | N/A | N/A | 0 | 0 |

## Output files

- `ttk_runs_fixed/topology_finetuning/candidateB_plus_E2_tf_lowlambda_expanded1344_eval/all_sample_metrics_candidateB_plus_E2_tf_lowlambda_expanded1344.csv` — per-sample metrics for all methods
- `ttk_runs_fixed/topology_finetuning/candidateB_plus_E2_tf_lowlambda_expanded1344_eval/pairwise_cnn_vs_candidateB_plus_E2_tf_lowlambda_expanded1344.csv` — per-metric delta table
- `ttk_runs_fixed/topology_finetuning/candidateB_plus_E2_tf_lowlambda_expanded1344_eval/winner_counts_candidateB_plus_E2_tf_lowlambda_expanded1344.csv` — win counts per method
- `ttk_runs_fixed/topology_finetuning/candidateB_plus_E2_tf_lowlambda_expanded1344_eval/adjacent_cluster_table_candidateB_plus_E2_tf_lowlambda_expanded1344.csv` — adjacent-cluster detail
- `docs/topology_finetuning_candidateB_plus_E2_tf_lowlambda_expanded1344_eval.md` — this report

## Notes

- PSNRuv is computed on physical [u,v] arrays using per-sample dynamic data range (GT max − GT min).
- All speed-based metrics use scalar speed = sqrt(u² + v²) in physical units (m/s).
- Exceedance fractions use GT percentile thresholds (p90/p95/p99) computed per sample.
- Component counts use 8-connectivity on speed superlevel sets.
- PD/MT distances for CNN and GAN are loaded from pre-computed merged CSV; `candidateB_plus_E2_tf_lowlambda_expanded1344` requires a fresh TTK run.
- `candidateB_plus_E2_tf_lowlambda_expanded1344` was fine-tuned from the pretrained CNN checkpoint. Results may change with different lambda settings or more epochs.
