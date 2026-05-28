# candidateDpd_expanded672 fine-tuning evaluation

**Generated:** 2026-05-27

**Candidate:** `candidateDpd_expanded672`

**Samples evaluated:** 168

**Methods compared:** bicubic, cnn, gan, candidateDpd_expanded672

> **SSIM note:** SSIM was not computed because skimage could not be imported in the current NumPy environment (`ValueError('numpy.dtype size changed, may indicate binary incompatibility. Expected 96 from C header, got 88 from PyObject')`). All SSIM values are NaN.

> Note: PD/MT distances for `candidateDpd_expanded672` are N/A — TTK has not been re-run on these outputs. Values for CNN/GAN are loaded from the pre-computed merged CSV.

## Summary metrics table

| Metric | Bicubic | Cnn | Gan | Candidatedpd_expanded672 |
|---|---|---|---|---|
| PSNRuv (dB) | 32.2202 | 31.1925 | 29.1380 | 31.4555 |
| SSIM on speed | N/A | N/A | N/A | N/A |
| Speed MAE (m/s) | 0.5963 | 0.6941 | 0.9026 | 0.6777 |
| Speed RMSE (m/s) | 1.0156 | 1.1078 | 1.3775 | 1.0730 |
| WPD MAE (m³/s³) | 193.5403 | 231.6709 | 310.7328 | 225.6084 |
| WPD Wasserstein-1 | 55.5103 | 45.2713 | 85.6191 | 51.1586 |
| WPD bias abs (m³/s³) | 41.2244 | 35.3439 | 78.7236 | 35.1255 |
| Gradient MAE | 0.3696 | 0.3491 | 0.3806 | 0.3583 |
| Gradient W1 | 0.3282 | 0.2329 | 0.0564 | 0.2763 |
| Gradient kurtosis abs Δ | 6.6163 | 3.7004 | 4.2010 | 4.2908 |
| PSD log-L2 | 1.3625 | 0.8335 | 0.5139 | 0.9390 |
| PSD slope abs Δ | 1.6684 | 0.9150 | 0.9482 | 1.1152 |
| Exceedance abs Δ s>5 | 0.0095 | 0.0042 | 0.0082 | 0.0067 |
| Exceedance abs Δ s>10 | 0.0085 | 0.0066 | 0.0243 | 0.0077 |
| Exceedance abs Δ s>15 | 0.0074 | 0.0062 | 0.0096 | 0.0072 |
| Exceedance abs Δ p90 | 0.0109 | 0.0103 | 0.0123 | 0.0099 |
| Component-count curve L1 | 173.1855 | 115.5278 | 124.6567 | 130.9821 |
| PD distance (TTK) | N/A | 27.4063 | 20.8641 | N/A |
| MT distance (TTK) | N/A | 5.8678 | 8.3481 | N/A |

## 7 Key evaluation questions

### Q1. Did candidateDpd_expanded672 preserve CNN's direct-fidelity advantage?

- **PSNRuv**: CNN=31.1925, GAN=29.1380, candidateDpd_expanded672=31.4555. Δ(candidateDpd_expanded672−CNN)=+0.2630 (▲ better), improved on 168/168 samples.
- **SSIM**: CNN=N/A, GAN=N/A, candidateDpd_expanded672=N/A. Δ(candidateDpd_expanded672−CNN)=N/A, improved on 0/0 samples.
- **Speed MAE**: CNN=0.6941, GAN=0.9026, candidateDpd_expanded672=0.6777. Δ(candidateDpd_expanded672−CNN)=-0.0164 (▲ better), improved on 167/168 samples.
- **Speed RMSE**: CNN=1.1078, GAN=1.3775, candidateDpd_expanded672=1.0730. Δ(candidateDpd_expanded672−CNN)=-0.0347 (▲ better), improved on 168/168 samples.

### Q2. Did candidateDpd_expanded672 improve scalar speed or WPD metrics?

- **WPD MAE**: CNN=231.6709, candidateDpd_expanded672=225.6084. Δ=-6.0625 (▲ better), improved on 161/168 samples.
- **WPD W1**: CNN=45.2713, candidateDpd_expanded672=51.1586. Δ=+5.8873 (▼ worse), improved on 72/168 samples.
- **WPD bias abs**: CNN=35.3439, candidateDpd_expanded672=35.1255. Δ=-0.2184 (▲ better), improved on 99/168 samples.

### Q3. Did candidateDpd_expanded672 improve gradient metrics?

- **Gradient MAE**: CNN=0.3491, candidateDpd_expanded672=0.3583. Δ=+0.0093 (▼ worse), improved on 8/168 samples.
- **Gradient W1**: CNN=0.2329, candidateDpd_expanded672=0.2763. Δ=+0.0433 (▼ worse), improved on 0/168 samples.
- **Gradient kurtosis abs Δ**: CNN=3.7004, candidateDpd_expanded672=4.2908. Δ=+0.5904 (▼ worse), improved on 78/168 samples.

### Q4. Did candidateDpd_expanded672 improve exceedance metrics, especially s>5?

- **Exceedance abs Δ s>5**: CNN=0.00424, candidateDpd_expanded672=0.00669. Δ=+0.0025 (▼ worse), improved on 17/168 samples.
- **Exceedance abs Δ s>10**: CNN=0.00658, candidateDpd_expanded672=0.00770. Δ=+0.0011 (▼ worse), improved on 75/168 samples.
- **Exceedance abs Δ s>15**: CNN=0.00621, candidateDpd_expanded672=0.00718. Δ=+0.0010 (▼ worse), improved on 49/168 samples.
- **Exceedance abs Δ p90**: CNN=0.01025, candidateDpd_expanded672=0.00992. Δ=-0.0003 (▲ better), improved on 99/168 samples.

### Q5. Did candidateDpd_expanded672 move component-count behavior toward GT?

- **Component-count curve L1**: CNN=115.5278, candidateDpd_expanded672=130.9821. Δ=+15.4544 (▼ worse), improved on 1/168 samples.
- **Comp-count abs Δ t5**: CNN=64.3810, candidateDpd_expanded672=66.7262. Δ=+2.3452 (▼ worse), improved on 44/168 samples.
- **Comp-count abs Δ t10**: CNN=187.1667, candidateDpd_expanded672=202.6845. Δ=+15.5179 (▼ worse), improved on 16/168 samples.
- **Comp-count abs Δ t15**: CNN=116.8095, candidateDpd_expanded672=138.1667. Δ=+21.3571 (▼ worse), improved on 7/168 samples.

### Q6. Did candidateDpd_expanded672 improve PD or MT distances?

PD and MT distances for `candidateDpd_expanded672` are not available in this evaluation (TTK was not re-run on these outputs). Pre-computed values for CNN and GAN:

- **PD distance**: CNN=27.4063, GAN=20.8641, candidateDpd_expanded672=N/A (requires TTK run)
- **MT distance**: CNN=5.8678, GAN=8.3481, candidateDpd_expanded672=N/A (requires TTK run)

To compute PD/MT for `candidateDpd_expanded672`: convert outputs to VTI via `scripts/convert_phire_to_vti.py`, then run `scripts/compute_composite_tree_distance.py`.

### Q7. Did candidateDpd_expanded672 change the adjacent clusters (samples 10–13, 90–93, 162–163)?

See `adjacent_cluster_table_candidateDpd_expanded672.csv` for full detail. Summary of PSNRuv, speed MAE, and gradient MAE changes:

| Sample | Δ PSNRuv | Δ Speed MAE | Δ Grad MAE | Δ Exceed s>5 |
|---|---|---|---|---|
| 10 | 0.374 | -0.0314 | 0.0106 | 0.00499 |
| 11 | 0.307 | -0.0250 | 0.0113 | 0.00393 |
| 12 | 0.269 | -0.0208 | 0.0116 | 0.00359 |
| 13 | 0.217 | -0.0132 | 0.0132 | 0.00310 |
| 90 | 0.123 | -0.0062 | 0.0049 | 0.00549 |
| 91 | 0.122 | -0.0075 | 0.0049 | 0.00610 |
| 92 | 0.154 | -0.0101 | 0.0061 | 0.00546 |
| 93 | 0.229 | -0.0175 | 0.0079 | 0.00637 |
| 162 | 0.697 | -0.0267 | -0.0062 | 0.00071 |
| 163 | 0.625 | -0.0276 | -0.0057 | 0.00494 |

## Pairwise summary: CNN vs candidateDpd_expanded672

| Metric | Mean CNN | Mean candidateDpd_expanded672 | Δ | N improved | N worsened |
|---|---|---|---|---|---|
| psnruv | 31.1925 | 31.4555 | 0.2630 | 168 | 0 |
| ssim | N/A | N/A | N/A | 0 | 0 |
| speed_mae | 0.6941 | 0.6777 | -0.0164 | 167 | 1 |
| speed_rmse | 1.1078 | 1.0730 | -0.0347 | 168 | 0 |
| wpd_bias_abs | 35.3439 | 35.1255 | -0.2184 | 99 | 69 |
| wpd_mae | 231.6709 | 225.6084 | -6.0625 | 161 | 7 |
| wpd_w1 | 45.2713 | 51.1586 | 5.8873 | 72 | 96 |
| psd_log_l2 | 0.8335 | 0.9390 | 0.1055 | 13 | 155 |
| psd_slope_abs_delta | 0.9150 | 1.1152 | 0.2002 | 44 | 124 |
| grad_mae | 0.3491 | 0.3583 | 0.0093 | 8 | 160 |
| grad_w1 | 0.2329 | 0.2763 | 0.0433 | 0 | 168 |
| grad_kurtosis_abs_delta | 3.7004 | 4.2908 | 0.5904 | 78 | 90 |
| exceed_abs_t5 | 0.0042 | 0.0067 | 0.0025 | 17 | 151 |
| exceed_abs_t10 | 0.0066 | 0.0077 | 0.0011 | 75 | 93 |
| exceed_abs_t15 | 0.0062 | 0.0072 | 0.0010 | 49 | 119 |
| exceed_abs_p90 | 0.0103 | 0.0099 | -0.0003 | 99 | 69 |
| comp_curve_l1 | 115.5278 | 130.9821 | 15.4544 | 1 | 167 |
| pd_distance | N/A | N/A | N/A | 0 | 0 |
| mt_distance | N/A | N/A | N/A | 0 | 0 |

## Output files

- `ttk_runs_fixed/topology_finetuning/candidateDpd_expanded672_eval/all_sample_metrics_candidateDpd_expanded672.csv` — per-sample metrics for all methods
- `ttk_runs_fixed/topology_finetuning/candidateDpd_expanded672_eval/pairwise_cnn_vs_candidateDpd_expanded672.csv` — per-metric delta table
- `ttk_runs_fixed/topology_finetuning/candidateDpd_expanded672_eval/winner_counts_candidateDpd_expanded672.csv` — win counts per method
- `ttk_runs_fixed/topology_finetuning/candidateDpd_expanded672_eval/adjacent_cluster_table_candidateDpd_expanded672.csv` — adjacent-cluster detail
- `docs/topology_finetuning_candidateDpd_expanded672_eval.md` — this report

## Notes

- PSNRuv is computed on physical [u,v] arrays using per-sample dynamic data range (GT max − GT min).
- All speed-based metrics use scalar speed = sqrt(u² + v²) in physical units (m/s).
- Exceedance fractions use GT percentile thresholds (p90/p95/p99) computed per sample.
- Component counts use 8-connectivity on speed superlevel sets.
- PD/MT distances for CNN and GAN are loaded from pre-computed merged CSV; `candidateDpd_expanded672` requires a fresh TTK run.
- `candidateDpd_expanded672` was fine-tuned from the pretrained CNN checkpoint. Results may change with different lambda settings or more epochs.
