# candidateC_expanded672 fine-tuning evaluation

**Generated:** 2026-05-23

**Candidate:** `candidateC_expanded672`

**Samples evaluated:** 168

**Methods compared:** bicubic, cnn, gan, candidateC_expanded672

> **SSIM note:** SSIM was not computed because skimage could not be imported in the current NumPy environment (`ValueError('numpy.dtype size changed, may indicate binary incompatibility. Expected 96 from C header, got 88 from PyObject')`). All SSIM values are NaN.

> Note: PD/MT distances for `candidateC_expanded672` are N/A — TTK has not been re-run on these outputs. Values for CNN/GAN are loaded from the pre-computed merged CSV.

## Summary metrics table

| Metric | Bicubic | Cnn | Gan | Candidatec_expanded672 |
|---|---|---|---|---|
| PSNRuv (dB) | 32.2202 | 31.1925 | 29.1380 | 32.9969 |
| SSIM on speed | N/A | N/A | N/A | N/A |
| Speed MAE (m/s) | 0.5963 | 0.6941 | 0.9026 | 0.5488 |
| Speed RMSE (m/s) | 1.0156 | 1.1078 | 1.3775 | 0.9218 |
| WPD MAE (m³/s³) | 193.5403 | 231.6709 | 310.7328 | 178.8435 |
| WPD Wasserstein-1 | 55.5103 | 45.2713 | 85.6191 | 20.9637 |
| WPD bias abs (m³/s³) | 41.2244 | 35.3439 | 78.7236 | 7.7277 |
| Gradient MAE | 0.3696 | 0.3491 | 0.3806 | 0.3200 |
| Gradient W1 | 0.3282 | 0.2329 | 0.0564 | 0.2061 |
| Gradient kurtosis abs Δ | 6.6163 | 3.7004 | 4.2010 | 2.1936 |
| PSD log-L2 | 1.3625 | 0.8335 | 0.5139 | 0.9804 |
| PSD slope abs Δ | 1.6684 | 0.9150 | 0.9482 | 1.2715 |
| Exceedance abs Δ s>5 | 0.0095 | 0.0042 | 0.0082 | 0.0064 |
| Exceedance abs Δ s>10 | 0.0085 | 0.0066 | 0.0243 | 0.0051 |
| Exceedance abs Δ s>15 | 0.0074 | 0.0062 | 0.0096 | 0.0020 |
| Exceedance abs Δ p90 | 0.0109 | 0.0103 | 0.0123 | 0.0025 |
| Component-count curve L1 | 173.1855 | 115.5278 | 124.6567 | 109.5446 |
| PD distance (TTK) | N/A | 27.4063 | 20.8641 | N/A |
| MT distance (TTK) | N/A | 5.8678 | 8.3481 | N/A |

## 7 Key evaluation questions

### Q1. Did candidateC_expanded672 preserve CNN's direct-fidelity advantage?

- **PSNRuv**: CNN=31.1925, GAN=29.1380, candidateC_expanded672=32.9969. Δ(candidateC_expanded672−CNN)=+1.8044 (▲ better), improved on 168/168 samples.
- **SSIM**: CNN=N/A, GAN=N/A, candidateC_expanded672=N/A. Δ(candidateC_expanded672−CNN)=N/A, improved on 0/0 samples.
- **Speed MAE**: CNN=0.6941, GAN=0.9026, candidateC_expanded672=0.5488. Δ(candidateC_expanded672−CNN)=-0.1453 (▲ better), improved on 168/168 samples.
- **Speed RMSE**: CNN=1.1078, GAN=1.3775, candidateC_expanded672=0.9218. Δ(candidateC_expanded672−CNN)=-0.1860 (▲ better), improved on 168/168 samples.

### Q2. Did candidateC_expanded672 improve scalar speed or WPD metrics?

- **WPD MAE**: CNN=231.6709, candidateC_expanded672=178.8435. Δ=-52.8275 (▲ better), improved on 168/168 samples.
- **WPD W1**: CNN=45.2713, candidateC_expanded672=20.9637. Δ=-24.3076 (▲ better), improved on 148/168 samples.
- **WPD bias abs**: CNN=35.3439, candidateC_expanded672=7.7277. Δ=-27.6162 (▲ better), improved on 138/168 samples.

### Q3. Did candidateC_expanded672 improve gradient metrics?

- **Gradient MAE**: CNN=0.3491, candidateC_expanded672=0.3200. Δ=-0.0291 (▲ better), improved on 168/168 samples.
- **Gradient W1**: CNN=0.2329, candidateC_expanded672=0.2061. Δ=-0.0269 (▲ better), improved on 142/168 samples.
- **Gradient kurtosis abs Δ**: CNN=3.7004, candidateC_expanded672=2.1936. Δ=-1.5068 (▲ better), improved on 94/168 samples.

### Q4. Did candidateC_expanded672 improve exceedance metrics, especially s>5?

- **Exceedance abs Δ s>5**: CNN=0.00424, candidateC_expanded672=0.00639. Δ=+0.0021 (▼ worse), improved on 34/168 samples.
- **Exceedance abs Δ s>10**: CNN=0.00658, candidateC_expanded672=0.00508. Δ=-0.0015 (▲ better), improved on 99/168 samples.
- **Exceedance abs Δ s>15**: CNN=0.00621, candidateC_expanded672=0.00204. Δ=-0.0042 (▲ better), improved on 129/168 samples.
- **Exceedance abs Δ p90**: CNN=0.01025, candidateC_expanded672=0.00253. Δ=-0.0077 (▲ better), improved on 148/168 samples.

### Q5. Did candidateC_expanded672 move component-count behavior toward GT?

- **Component-count curve L1**: CNN=115.5278, candidateC_expanded672=109.5446. Δ=-5.9831 (▲ better), improved on 122/168 samples.
- **Comp-count abs Δ t5**: CNN=64.3810, candidateC_expanded672=61.8333. Δ=-2.5476 (▲ better), improved on 90/168 samples.
- **Comp-count abs Δ t10**: CNN=187.1667, candidateC_expanded672=193.0417. Δ=+5.8750 (▼ worse), improved on 64/168 samples.
- **Comp-count abs Δ t15**: CNN=116.8095, candidateC_expanded672=103.6131. Δ=-13.1964 (▲ better), improved on 120/168 samples.

### Q6. Did candidateC_expanded672 improve PD or MT distances?

PD and MT distances for `candidateC_expanded672` are not available in this evaluation (TTK was not re-run on these outputs). Pre-computed values for CNN and GAN:

- **PD distance**: CNN=27.4063, GAN=20.8641, candidateC_expanded672=N/A (requires TTK run)
- **MT distance**: CNN=5.8678, GAN=8.3481, candidateC_expanded672=N/A (requires TTK run)

To compute PD/MT for `candidateC_expanded672`: convert outputs to VTI via `scripts/convert_phire_to_vti.py`, then run `scripts/compute_composite_tree_distance.py`.

### Q7. Did candidateC_expanded672 change the adjacent clusters (samples 10–13, 90–93, 162–163)?

See `adjacent_cluster_table_candidateC_expanded672.csv` for full detail. Summary of PSNRuv, speed MAE, and gradient MAE changes:

| Sample | Δ PSNRuv | Δ Speed MAE | Δ Grad MAE | Δ Exceed s>5 |
|---|---|---|---|---|
| 10 | 2.032 | -0.1652 | -0.0289 | 0.00324 |
| 11 | 1.916 | -0.1607 | -0.0291 | 0.00236 |
| 12 | 1.936 | -0.1612 | -0.0338 | 0.00230 |
| 13 | 1.824 | -0.1503 | -0.0298 | -0.00058 |
| 90 | 1.676 | -0.1106 | -0.0228 | 0.00561 |
| 91 | 1.702 | -0.1151 | -0.0226 | 0.01080 |
| 92 | 1.669 | -0.1240 | -0.0261 | 0.01422 |
| 93 | 1.622 | -0.1319 | -0.0272 | 0.00926 |
| 162 | 2.948 | -0.1221 | -0.0207 | -0.00319 |
| 163 | 2.915 | -0.1223 | -0.0227 | 0.00212 |

## Pairwise summary: CNN vs candidateC_expanded672

| Metric | Mean CNN | Mean candidateC_expanded672 | Δ | N improved | N worsened |
|---|---|---|---|---|---|
| psnruv | 31.1925 | 32.9969 | 1.8044 | 168 | 0 |
| ssim | N/A | N/A | N/A | 0 | 0 |
| speed_mae | 0.6941 | 0.5488 | -0.1453 | 168 | 0 |
| speed_rmse | 1.1078 | 0.9218 | -0.1860 | 168 | 0 |
| wpd_bias_abs | 35.3439 | 7.7277 | -27.6162 | 138 | 30 |
| wpd_mae | 231.6709 | 178.8435 | -52.8275 | 168 | 0 |
| wpd_w1 | 45.2713 | 20.9637 | -24.3076 | 148 | 20 |
| psd_log_l2 | 0.8335 | 0.9804 | 0.1469 | 1 | 167 |
| psd_slope_abs_delta | 0.9150 | 1.2715 | 0.3565 | 0 | 168 |
| grad_mae | 0.3491 | 0.3200 | -0.0291 | 168 | 0 |
| grad_w1 | 0.2329 | 0.2061 | -0.0269 | 142 | 26 |
| grad_kurtosis_abs_delta | 3.7004 | 2.1936 | -1.5068 | 94 | 74 |
| exceed_abs_t5 | 0.0042 | 0.0064 | 0.0021 | 34 | 134 |
| exceed_abs_t10 | 0.0066 | 0.0051 | -0.0015 | 99 | 69 |
| exceed_abs_t15 | 0.0062 | 0.0020 | -0.0042 | 129 | 39 |
| exceed_abs_p90 | 0.0103 | 0.0025 | -0.0077 | 148 | 20 |
| comp_curve_l1 | 115.5278 | 109.5446 | -5.9831 | 122 | 46 |
| pd_distance | N/A | N/A | N/A | 0 | 0 |
| mt_distance | N/A | N/A | N/A | 0 | 0 |

## Output files

- `ttk_runs_fixed/topology_finetuning/candidateC_expanded672_eval/all_sample_metrics_candidateC_expanded672.csv` — per-sample metrics for all methods
- `ttk_runs_fixed/topology_finetuning/candidateC_expanded672_eval/pairwise_cnn_vs_candidateC_expanded672.csv` — per-metric delta table
- `ttk_runs_fixed/topology_finetuning/candidateC_expanded672_eval/winner_counts_candidateC_expanded672.csv` — win counts per method
- `ttk_runs_fixed/topology_finetuning/candidateC_expanded672_eval/adjacent_cluster_table_candidateC_expanded672.csv` — adjacent-cluster detail
- `docs/topology_finetuning_candidateC_expanded672_eval.md` — this report

## Notes

- PSNRuv is computed on physical [u,v] arrays using per-sample dynamic data range (GT max − GT min).
- All speed-based metrics use scalar speed = sqrt(u² + v²) in physical units (m/s).
- Exceedance fractions use GT percentile thresholds (p90/p95/p99) computed per sample.
- Component counts use 8-connectivity on speed superlevel sets.
- PD/MT distances for CNN and GAN are loaded from pre-computed merged CSV; `candidateC_expanded672` requires a fresh TTK run.
- `candidateC_expanded672` was fine-tuned from the pretrained CNN checkpoint. Results may change with different lambda settings or more epochs.
