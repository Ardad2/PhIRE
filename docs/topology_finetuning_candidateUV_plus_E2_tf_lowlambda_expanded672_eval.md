# candidateUV_plus_E2_tf_lowlambda_expanded672 fine-tuning evaluation

**Generated:** 2026-07-10

**Candidate:** `candidateUV_plus_E2_tf_lowlambda_expanded672`

**Samples evaluated:** 168

**Methods compared:** bicubic, cnn, gan, candidateUV_plus_E2_tf_lowlambda_expanded672

> **SSIM note:** SSIM was not computed because skimage could not be imported in the current NumPy environment (`ValueError('numpy.dtype size changed, may indicate binary incompatibility. Expected 96 from C header, got 88 from PyObject')`). All SSIM values are NaN.

> Note: PD/MT distances for `candidateUV_plus_E2_tf_lowlambda_expanded672` are N/A — TTK has not been re-run on these outputs. Values for CNN/GAN are loaded from the pre-computed merged CSV.

## Summary metrics table

| Metric | Bicubic | Cnn | Gan | Candidateuv_plus_e2_tf_lowlambda_expanded672 |
|---|---|---|---|---|
| PSNRuv (dB) | 32.2202 | 31.1925 | 29.1380 | 31.8657 |
| SSIM on speed | N/A | N/A | N/A | N/A |
| Speed MAE (m/s) | 0.5963 | 0.6941 | 0.9026 | 0.6386 |
| Speed RMSE (m/s) | 1.0156 | 1.1078 | 1.3775 | 1.0574 |
| WPD MAE (m³/s³) | 193.5403 | 231.6709 | 310.7328 | 210.1936 |
| WPD Wasserstein-1 | 55.5103 | 45.2713 | 85.6191 | 20.2967 |
| WPD bias abs (m³/s³) | 41.2244 | 35.3439 | 78.7236 | 12.1275 |
| Gradient MAE | 0.3696 | 0.3491 | 0.3806 | 0.3260 |
| Gradient W1 | 0.3282 | 0.2329 | 0.0564 | 0.1466 |
| Gradient kurtosis abs Δ | 6.6163 | 3.7004 | 4.2010 | 2.9402 |
| PSD log-L2 | 1.3625 | 0.8335 | 0.5139 | 0.7724 |
| PSD slope abs Δ | 1.6684 | 0.9150 | 0.9482 | 0.9683 |
| Exceedance abs Δ s>5 | 0.0095 | 0.0042 | 0.0082 | 0.0029 |
| Exceedance abs Δ s>10 | 0.0085 | 0.0066 | 0.0243 | 0.0040 |
| Exceedance abs Δ s>15 | 0.0074 | 0.0062 | 0.0096 | 0.0019 |
| Exceedance abs Δ p90 | 0.0109 | 0.0103 | 0.0123 | 0.0034 |
| Component-count curve L1 | 173.1855 | 115.5278 | 124.6567 | 83.8552 |
| PD distance (TTK) | N/A | 27.4063 | 20.8641 | N/A |
| MT distance (TTK) | N/A | 5.8678 | 8.3481 | N/A |

## 7 Key evaluation questions

### Q1. Did candidateUV_plus_E2_tf_lowlambda_expanded672 preserve CNN's direct-fidelity advantage?

- **PSNRuv**: CNN=31.1925, GAN=29.1380, candidateUV_plus_E2_tf_lowlambda_expanded672=31.8657. Δ(candidateUV_plus_E2_tf_lowlambda_expanded672−CNN)=+0.6732 (▲ better), improved on 155/168 samples.
- **SSIM**: CNN=N/A, GAN=N/A, candidateUV_plus_E2_tf_lowlambda_expanded672=N/A. Δ(candidateUV_plus_E2_tf_lowlambda_expanded672−CNN)=N/A, improved on 0/0 samples.
- **Speed MAE**: CNN=0.6941, GAN=0.9026, candidateUV_plus_E2_tf_lowlambda_expanded672=0.6386. Δ(candidateUV_plus_E2_tf_lowlambda_expanded672−CNN)=-0.0555 (▲ better), improved on 168/168 samples.
- **Speed RMSE**: CNN=1.1078, GAN=1.3775, candidateUV_plus_E2_tf_lowlambda_expanded672=1.0574. Δ(candidateUV_plus_E2_tf_lowlambda_expanded672−CNN)=-0.0503 (▲ better), improved on 129/168 samples.

### Q2. Did candidateUV_plus_E2_tf_lowlambda_expanded672 improve scalar speed or WPD metrics?

- **WPD MAE**: CNN=231.6709, candidateUV_plus_E2_tf_lowlambda_expanded672=210.1936. Δ=-21.4773 (▲ better), improved on 163/168 samples.
- **WPD W1**: CNN=45.2713, candidateUV_plus_E2_tf_lowlambda_expanded672=20.2967. Δ=-24.9745 (▲ better), improved on 154/168 samples.
- **WPD bias abs**: CNN=35.3439, candidateUV_plus_E2_tf_lowlambda_expanded672=12.1275. Δ=-23.2165 (▲ better), improved on 144/168 samples.

### Q3. Did candidateUV_plus_E2_tf_lowlambda_expanded672 improve gradient metrics?

- **Gradient MAE**: CNN=0.3491, candidateUV_plus_E2_tf_lowlambda_expanded672=0.3260. Δ=-0.0230 (▲ better), improved on 168/168 samples.
- **Gradient W1**: CNN=0.2329, candidateUV_plus_E2_tf_lowlambda_expanded672=0.1466. Δ=-0.0863 (▲ better), improved on 159/168 samples.
- **Gradient kurtosis abs Δ**: CNN=3.7004, candidateUV_plus_E2_tf_lowlambda_expanded672=2.9402. Δ=-0.7602 (▲ better), improved on 86/168 samples.

### Q4. Did candidateUV_plus_E2_tf_lowlambda_expanded672 improve exceedance metrics, especially s>5?

- **Exceedance abs Δ s>5**: CNN=0.00424, candidateUV_plus_E2_tf_lowlambda_expanded672=0.00292. Δ=-0.0013 (▲ better), improved on 115/168 samples.
- **Exceedance abs Δ s>10**: CNN=0.00658, candidateUV_plus_E2_tf_lowlambda_expanded672=0.00398. Δ=-0.0026 (▲ better), improved on 119/168 samples.
- **Exceedance abs Δ s>15**: CNN=0.00621, candidateUV_plus_E2_tf_lowlambda_expanded672=0.00195. Δ=-0.0043 (▲ better), improved on 131/168 samples.
- **Exceedance abs Δ p90**: CNN=0.01025, candidateUV_plus_E2_tf_lowlambda_expanded672=0.00338. Δ=-0.0069 (▲ better), improved on 145/168 samples.

### Q5. Did candidateUV_plus_E2_tf_lowlambda_expanded672 move component-count behavior toward GT?

- **Component-count curve L1**: CNN=115.5278, candidateUV_plus_E2_tf_lowlambda_expanded672=83.8552. Δ=-31.6726 (▲ better), improved on 154/168 samples.
- **Comp-count abs Δ t5**: CNN=64.3810, candidateUV_plus_E2_tf_lowlambda_expanded672=59.3690. Δ=-5.0119 (▲ better), improved on 117/168 samples.
- **Comp-count abs Δ t10**: CNN=187.1667, candidateUV_plus_E2_tf_lowlambda_expanded672=164.7440. Δ=-22.4226 (▲ better), improved on 140/168 samples.
- **Comp-count abs Δ t15**: CNN=116.8095, candidateUV_plus_E2_tf_lowlambda_expanded672=68.2917. Δ=-48.5179 (▲ better), improved on 150/168 samples.

### Q6. Did candidateUV_plus_E2_tf_lowlambda_expanded672 improve PD or MT distances?

PD and MT distances for `candidateUV_plus_E2_tf_lowlambda_expanded672` are not available in this evaluation (TTK was not re-run on these outputs). Pre-computed values for CNN and GAN:

- **PD distance**: CNN=27.4063, GAN=20.8641, candidateUV_plus_E2_tf_lowlambda_expanded672=N/A (requires TTK run)
- **MT distance**: CNN=5.8678, GAN=8.3481, candidateUV_plus_E2_tf_lowlambda_expanded672=N/A (requires TTK run)

To compute PD/MT for `candidateUV_plus_E2_tf_lowlambda_expanded672`: convert outputs to VTI via `scripts/convert_phire_to_vti.py`, then run `scripts/compute_composite_tree_distance.py`.

### Q7. Did candidateUV_plus_E2_tf_lowlambda_expanded672 change the adjacent clusters (samples 10–13, 90–93, 162–163)?

See `adjacent_cluster_table_candidateUV_plus_E2_tf_lowlambda_expanded672.csv` for full detail. Summary of PSNRuv, speed MAE, and gradient MAE changes:

| Sample | Δ PSNRuv | Δ Speed MAE | Δ Grad MAE | Δ Exceed s>5 |
|---|---|---|---|---|
| 10 | 1.016 | -0.0789 | -0.0228 | -0.00311 |
| 11 | 0.871 | -0.0707 | -0.0218 | -0.00126 |
| 12 | 0.955 | -0.0734 | -0.0267 | -0.00003 |
| 13 | 0.806 | -0.0583 | -0.0211 | -0.00015 |
| 90 | 0.648 | -0.0443 | -0.0200 | -0.00202 |
| 91 | 0.711 | -0.0484 | -0.0187 | 0.00263 |
| 92 | 0.611 | -0.0518 | -0.0230 | 0.00455 |
| 93 | 0.511 | -0.0454 | -0.0232 | -0.00238 |
| 162 | 1.777 | -0.0769 | -0.0160 | -0.00110 |
| 163 | 1.583 | -0.0747 | -0.0160 | 0.00646 |

## Pairwise summary: CNN vs candidateUV_plus_E2_tf_lowlambda_expanded672

| Metric | Mean CNN | Mean candidateUV_plus_E2_tf_lowlambda_expanded672 | Δ | N improved | N worsened |
|---|---|---|---|---|---|
| psnruv | 31.1925 | 31.8657 | 0.6732 | 155 | 13 |
| ssim | N/A | N/A | N/A | 0 | 0 |
| speed_mae | 0.6941 | 0.6386 | -0.0555 | 168 | 0 |
| speed_rmse | 1.1078 | 1.0574 | -0.0503 | 129 | 39 |
| wpd_bias_abs | 35.3439 | 12.1275 | -23.2165 | 144 | 24 |
| wpd_mae | 231.6709 | 210.1936 | -21.4773 | 163 | 5 |
| wpd_w1 | 45.2713 | 20.2967 | -24.9745 | 154 | 14 |
| psd_log_l2 | 0.8335 | 0.7724 | -0.0611 | 128 | 40 |
| psd_slope_abs_delta | 0.9150 | 0.9683 | 0.0533 | 55 | 113 |
| grad_mae | 0.3491 | 0.3260 | -0.0230 | 168 | 0 |
| grad_w1 | 0.2329 | 0.1466 | -0.0863 | 159 | 9 |
| grad_kurtosis_abs_delta | 3.7004 | 2.9402 | -0.7602 | 86 | 82 |
| exceed_abs_t5 | 0.0042 | 0.0029 | -0.0013 | 115 | 53 |
| exceed_abs_t10 | 0.0066 | 0.0040 | -0.0026 | 119 | 48 |
| exceed_abs_t15 | 0.0062 | 0.0019 | -0.0043 | 131 | 37 |
| exceed_abs_p90 | 0.0103 | 0.0034 | -0.0069 | 145 | 23 |
| comp_curve_l1 | 115.5278 | 83.8552 | -31.6726 | 154 | 13 |
| pd_distance | N/A | N/A | N/A | 0 | 0 |
| mt_distance | N/A | N/A | N/A | 0 | 0 |

## Output files

- `ttk_runs_fixed/topology_finetuning/candidateUV_plus_E2_tf_lowlambda_expanded672_eval/all_sample_metrics_candidateUV_plus_E2_tf_lowlambda_expanded672.csv` — per-sample metrics for all methods
- `ttk_runs_fixed/topology_finetuning/candidateUV_plus_E2_tf_lowlambda_expanded672_eval/pairwise_cnn_vs_candidateUV_plus_E2_tf_lowlambda_expanded672.csv` — per-metric delta table
- `ttk_runs_fixed/topology_finetuning/candidateUV_plus_E2_tf_lowlambda_expanded672_eval/winner_counts_candidateUV_plus_E2_tf_lowlambda_expanded672.csv` — win counts per method
- `ttk_runs_fixed/topology_finetuning/candidateUV_plus_E2_tf_lowlambda_expanded672_eval/adjacent_cluster_table_candidateUV_plus_E2_tf_lowlambda_expanded672.csv` — adjacent-cluster detail
- `docs/topology_finetuning_candidateUV_plus_E2_tf_lowlambda_expanded672_eval.md` — this report

## Notes

- PSNRuv is computed on physical [u,v] arrays using per-sample dynamic data range (GT max − GT min).
- All speed-based metrics use scalar speed = sqrt(u² + v²) in physical units (m/s).
- Exceedance fractions use GT percentile thresholds (p90/p95/p99) computed per sample.
- Component counts use 8-connectivity on speed superlevel sets.
- PD/MT distances for CNN and GAN are loaded from pre-computed merged CSV; `candidateUV_plus_E2_tf_lowlambda_expanded672` requires a fresh TTK run.
- `candidateUV_plus_E2_tf_lowlambda_expanded672` was fine-tuned from the pretrained CNN checkpoint. Results may change with different lambda settings or more epochs.
