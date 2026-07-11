# candidateUV_plus_crit_expanded672 fine-tuning evaluation

**Generated:** 2026-07-11

**Candidate:** `candidateUV_plus_crit_expanded672`

**Samples evaluated:** 168

**Methods compared:** bicubic, cnn, gan, candidateUV_plus_crit_expanded672

> **SSIM note:** SSIM was not computed because skimage could not be imported in the current NumPy environment (`ValueError('numpy.dtype size changed, may indicate binary incompatibility. Expected 96 from C header, got 88 from PyObject')`). All SSIM values are NaN.

> Note: PD/MT distances for `candidateUV_plus_crit_expanded672` are N/A — TTK has not been re-run on these outputs. Values for CNN/GAN are loaded from the pre-computed merged CSV.

## Summary metrics table

| Metric | Bicubic | Cnn | Gan | Candidateuv_plus_crit_expanded672 |
|---|---|---|---|---|
| PSNRuv (dB) | 32.2202 | 31.1925 | 29.1380 | 33.1536 |
| SSIM on speed | N/A | N/A | N/A | N/A |
| Speed MAE (m/s) | 0.5963 | 0.6941 | 0.9026 | 0.5385 |
| Speed RMSE (m/s) | 1.0156 | 1.1078 | 1.3775 | 0.9093 |
| WPD MAE (m³/s³) | 193.5403 | 231.6709 | 310.7328 | 176.0295 |
| WPD Wasserstein-1 | 55.5103 | 45.2713 | 85.6191 | 27.2570 |
| WPD bias abs (m³/s³) | 41.2244 | 35.3439 | 78.7236 | 12.0790 |
| Gradient MAE | 0.3696 | 0.3491 | 0.3806 | 0.3296 |
| Gradient W1 | 0.3282 | 0.2329 | 0.0564 | 0.2439 |
| Gradient kurtosis abs Δ | 6.6163 | 3.7004 | 4.2010 | 2.9691 |
| PSD log-L2 | 1.3625 | 0.8335 | 0.5139 | 1.0670 |
| PSD slope abs Δ | 1.6684 | 0.9150 | 0.9482 | 1.3543 |
| Exceedance abs Δ s>5 | 0.0095 | 0.0042 | 0.0082 | 0.0091 |
| Exceedance abs Δ s>10 | 0.0085 | 0.0066 | 0.0243 | 0.0081 |
| Exceedance abs Δ s>15 | 0.0074 | 0.0062 | 0.0096 | 0.0024 |
| Exceedance abs Δ p90 | 0.0109 | 0.0103 | 0.0123 | 0.0027 |
| Component-count curve L1 | 173.1855 | 115.5278 | 124.6567 | 133.9722 |
| PD distance (TTK) | N/A | 27.4063 | 20.8641 | N/A |
| MT distance (TTK) | N/A | 5.8678 | 8.3481 | N/A |

## 7 Key evaluation questions

### Q1. Did candidateUV_plus_crit_expanded672 preserve CNN's direct-fidelity advantage?

- **PSNRuv**: CNN=31.1925, GAN=29.1380, candidateUV_plus_crit_expanded672=33.1536. Δ(candidateUV_plus_crit_expanded672−CNN)=+1.9611 (▲ better), improved on 168/168 samples.
- **SSIM**: CNN=N/A, GAN=N/A, candidateUV_plus_crit_expanded672=N/A. Δ(candidateUV_plus_crit_expanded672−CNN)=N/A, improved on 0/0 samples.
- **Speed MAE**: CNN=0.6941, GAN=0.9026, candidateUV_plus_crit_expanded672=0.5385. Δ(candidateUV_plus_crit_expanded672−CNN)=-0.1556 (▲ better), improved on 168/168 samples.
- **Speed RMSE**: CNN=1.1078, GAN=1.3775, candidateUV_plus_crit_expanded672=0.9093. Δ(candidateUV_plus_crit_expanded672−CNN)=-0.1985 (▲ better), improved on 168/168 samples.

### Q2. Did candidateUV_plus_crit_expanded672 improve scalar speed or WPD metrics?

- **WPD MAE**: CNN=231.6709, candidateUV_plus_crit_expanded672=176.0295. Δ=-55.6414 (▲ better), improved on 168/168 samples.
- **WPD W1**: CNN=45.2713, candidateUV_plus_crit_expanded672=27.2570. Δ=-18.0143 (▲ better), improved on 150/168 samples.
- **WPD bias abs**: CNN=35.3439, candidateUV_plus_crit_expanded672=12.0790. Δ=-23.2649 (▲ better), improved on 151/168 samples.

### Q3. Did candidateUV_plus_crit_expanded672 improve gradient metrics?

- **Gradient MAE**: CNN=0.3491, candidateUV_plus_crit_expanded672=0.3296. Δ=-0.0194 (▲ better), improved on 168/168 samples.
- **Gradient W1**: CNN=0.2329, candidateUV_plus_crit_expanded672=0.2439. Δ=+0.0109 (▼ worse), improved on 56/168 samples.
- **Gradient kurtosis abs Δ**: CNN=3.7004, candidateUV_plus_crit_expanded672=2.9691. Δ=-0.7313 (▲ better), improved on 87/168 samples.

### Q4. Did candidateUV_plus_crit_expanded672 improve exceedance metrics, especially s>5?

- **Exceedance abs Δ s>5**: CNN=0.00424, candidateUV_plus_crit_expanded672=0.00911. Δ=+0.0049 (▼ worse), improved on 5/168 samples.
- **Exceedance abs Δ s>10**: CNN=0.00658, candidateUV_plus_crit_expanded672=0.00810. Δ=+0.0015 (▼ worse), improved on 81/168 samples.
- **Exceedance abs Δ s>15**: CNN=0.00621, candidateUV_plus_crit_expanded672=0.00236. Δ=-0.0039 (▲ better), improved on 133/168 samples.
- **Exceedance abs Δ p90**: CNN=0.01025, candidateUV_plus_crit_expanded672=0.00274. Δ=-0.0075 (▲ better), improved on 147/168 samples.

### Q5. Did candidateUV_plus_crit_expanded672 move component-count behavior toward GT?

- **Component-count curve L1**: CNN=115.5278, candidateUV_plus_crit_expanded672=133.9722. Δ=+18.4444 (▼ worse), improved on 9/168 samples.
- **Comp-count abs Δ t5**: CNN=64.3810, candidateUV_plus_crit_expanded672=70.6488. Δ=+6.2679 (▼ worse), improved on 30/168 samples.
- **Comp-count abs Δ t10**: CNN=187.1667, candidateUV_plus_crit_expanded672=230.0952. Δ=+42.9286 (▼ worse), improved on 2/168 samples.
- **Comp-count abs Δ t15**: CNN=116.8095, candidateUV_plus_crit_expanded672=129.8988. Δ=+13.0893 (▼ worse), improved on 55/168 samples.

### Q6. Did candidateUV_plus_crit_expanded672 improve PD or MT distances?

PD and MT distances for `candidateUV_plus_crit_expanded672` are not available in this evaluation (TTK was not re-run on these outputs). Pre-computed values for CNN and GAN:

- **PD distance**: CNN=27.4063, GAN=20.8641, candidateUV_plus_crit_expanded672=N/A (requires TTK run)
- **MT distance**: CNN=5.8678, GAN=8.3481, candidateUV_plus_crit_expanded672=N/A (requires TTK run)

To compute PD/MT for `candidateUV_plus_crit_expanded672`: convert outputs to VTI via `scripts/convert_phire_to_vti.py`, then run `scripts/compute_composite_tree_distance.py`.

### Q7. Did candidateUV_plus_crit_expanded672 change the adjacent clusters (samples 10–13, 90–93, 162–163)?

See `adjacent_cluster_table_candidateUV_plus_crit_expanded672.csv` for full detail. Summary of PSNRuv, speed MAE, and gradient MAE changes:

| Sample | Δ PSNRuv | Δ Speed MAE | Δ Grad MAE | Δ Exceed s>5 |
|---|---|---|---|---|
| 10 | 2.124 | -0.1729 | -0.0187 | 0.00641 |
| 11 | 2.028 | -0.1687 | -0.0185 | 0.00532 |
| 12 | 2.064 | -0.1684 | -0.0233 | 0.00514 |
| 13 | 1.966 | -0.1567 | -0.0193 | 0.00201 |
| 90 | 1.829 | -0.1186 | -0.0161 | 0.00749 |
| 91 | 1.855 | -0.1233 | -0.0157 | 0.01248 |
| 92 | 1.835 | -0.1332 | -0.0173 | 0.01662 |
| 93 | 1.793 | -0.1429 | -0.0151 | 0.01361 |
| 162 | 3.176 | -0.1340 | -0.0213 | -0.00070 |
| 163 | 3.103 | -0.1339 | -0.0231 | 0.00477 |

## Pairwise summary: CNN vs candidateUV_plus_crit_expanded672

| Metric | Mean CNN | Mean candidateUV_plus_crit_expanded672 | Δ | N improved | N worsened |
|---|---|---|---|---|---|
| psnruv | 31.1925 | 33.1536 | 1.9611 | 168 | 0 |
| ssim | N/A | N/A | N/A | 0 | 0 |
| speed_mae | 0.6941 | 0.5385 | -0.1556 | 168 | 0 |
| speed_rmse | 1.1078 | 0.9093 | -0.1985 | 168 | 0 |
| wpd_bias_abs | 35.3439 | 12.0790 | -23.2649 | 151 | 17 |
| wpd_mae | 231.6709 | 176.0295 | -55.6414 | 168 | 0 |
| wpd_w1 | 45.2713 | 27.2570 | -18.0143 | 150 | 18 |
| psd_log_l2 | 0.8335 | 1.0670 | 0.2335 | 0 | 168 |
| psd_slope_abs_delta | 0.9150 | 1.3543 | 0.4393 | 0 | 168 |
| grad_mae | 0.3491 | 0.3296 | -0.0194 | 168 | 0 |
| grad_w1 | 0.2329 | 0.2439 | 0.0109 | 56 | 112 |
| grad_kurtosis_abs_delta | 3.7004 | 2.9691 | -0.7313 | 87 | 81 |
| exceed_abs_t5 | 0.0042 | 0.0091 | 0.0049 | 5 | 163 |
| exceed_abs_t10 | 0.0066 | 0.0081 | 0.0015 | 81 | 86 |
| exceed_abs_t15 | 0.0062 | 0.0024 | -0.0039 | 133 | 35 |
| exceed_abs_p90 | 0.0103 | 0.0027 | -0.0075 | 147 | 21 |
| comp_curve_l1 | 115.5278 | 133.9722 | 18.4444 | 9 | 158 |
| pd_distance | N/A | N/A | N/A | 0 | 0 |
| mt_distance | N/A | N/A | N/A | 0 | 0 |

## Output files

- `ttk_runs_fixed/topology_finetuning/candidateUV_plus_crit_expanded672_eval/all_sample_metrics_candidateUV_plus_crit_expanded672.csv` — per-sample metrics for all methods
- `ttk_runs_fixed/topology_finetuning/candidateUV_plus_crit_expanded672_eval/pairwise_cnn_vs_candidateUV_plus_crit_expanded672.csv` — per-metric delta table
- `ttk_runs_fixed/topology_finetuning/candidateUV_plus_crit_expanded672_eval/winner_counts_candidateUV_plus_crit_expanded672.csv` — win counts per method
- `ttk_runs_fixed/topology_finetuning/candidateUV_plus_crit_expanded672_eval/adjacent_cluster_table_candidateUV_plus_crit_expanded672.csv` — adjacent-cluster detail
- `docs/topology_finetuning_candidateUV_plus_crit_expanded672_eval.md` — this report

## Notes

- PSNRuv is computed on physical [u,v] arrays using per-sample dynamic data range (GT max − GT min).
- All speed-based metrics use scalar speed = sqrt(u² + v²) in physical units (m/s).
- Exceedance fractions use GT percentile thresholds (p90/p95/p99) computed per sample.
- Component counts use 8-connectivity on speed superlevel sets.
- PD/MT distances for CNN and GAN are loaded from pre-computed merged CSV; `candidateUV_plus_crit_expanded672` requires a fresh TTK run.
- `candidateUV_plus_crit_expanded672` was fine-tuned from the pretrained CNN checkpoint. Results may change with different lambda settings or more epochs.
