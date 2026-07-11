# candidateUV_plus_crit_expanded1344 fine-tuning evaluation

**Generated:** 2026-07-11

**Candidate:** `candidateUV_plus_crit_expanded1344`

**Samples evaluated:** 168

**Methods compared:** bicubic, cnn, gan, candidateUV_plus_crit_expanded1344

> **SSIM note:** SSIM was not computed because skimage could not be imported in the current NumPy environment (`ValueError('numpy.dtype size changed, may indicate binary incompatibility. Expected 96 from C header, got 88 from PyObject')`). All SSIM values are NaN.

> Note: PD/MT distances for `candidateUV_plus_crit_expanded1344` are N/A — TTK has not been re-run on these outputs. Values for CNN/GAN are loaded from the pre-computed merged CSV.

## Summary metrics table

| Metric | Bicubic | Cnn | Gan | Candidateuv_plus_crit_expanded1344 |
|---|---|---|---|---|
| PSNRuv (dB) | 32.2202 | 31.1925 | 29.1380 | 33.3888 |
| SSIM on speed | N/A | N/A | N/A | N/A |
| Speed MAE (m/s) | 0.5963 | 0.6941 | 0.9026 | 0.5218 |
| Speed RMSE (m/s) | 1.0156 | 1.1078 | 1.3775 | 0.8902 |
| WPD MAE (m³/s³) | 193.5403 | 231.6709 | 310.7328 | 170.1947 |
| WPD Wasserstein-1 | 55.5103 | 45.2713 | 85.6191 | 25.8500 |
| WPD bias abs (m³/s³) | 41.2244 | 35.3439 | 78.7236 | 17.8387 |
| Gradient MAE | 0.3696 | 0.3491 | 0.3806 | 0.3229 |
| Gradient W1 | 0.3282 | 0.2329 | 0.0564 | 0.2335 |
| Gradient kurtosis abs Δ | 6.6163 | 3.7004 | 4.2010 | 4.6949 |
| PSD log-L2 | 1.3625 | 0.8335 | 0.5139 | 1.0568 |
| PSD slope abs Δ | 1.6684 | 0.9150 | 0.9482 | 1.3842 |
| Exceedance abs Δ s>5 | 0.0095 | 0.0042 | 0.0082 | 0.0081 |
| Exceedance abs Δ s>10 | 0.0085 | 0.0066 | 0.0243 | 0.0082 |
| Exceedance abs Δ s>15 | 0.0074 | 0.0062 | 0.0096 | 0.0020 |
| Exceedance abs Δ p90 | 0.0109 | 0.0103 | 0.0123 | 0.0024 |
| Component-count curve L1 | 173.1855 | 115.5278 | 124.6567 | 132.1756 |
| PD distance (TTK) | N/A | 27.4063 | 20.8641 | N/A |
| MT distance (TTK) | N/A | 5.8678 | 8.3481 | N/A |

## 7 Key evaluation questions

### Q1. Did candidateUV_plus_crit_expanded1344 preserve CNN's direct-fidelity advantage?

- **PSNRuv**: CNN=31.1925, GAN=29.1380, candidateUV_plus_crit_expanded1344=33.3888. Δ(candidateUV_plus_crit_expanded1344−CNN)=+2.1963 (▲ better), improved on 168/168 samples.
- **SSIM**: CNN=N/A, GAN=N/A, candidateUV_plus_crit_expanded1344=N/A. Δ(candidateUV_plus_crit_expanded1344−CNN)=N/A, improved on 0/0 samples.
- **Speed MAE**: CNN=0.6941, GAN=0.9026, candidateUV_plus_crit_expanded1344=0.5218. Δ(candidateUV_plus_crit_expanded1344−CNN)=-0.1722 (▲ better), improved on 168/168 samples.
- **Speed RMSE**: CNN=1.1078, GAN=1.3775, candidateUV_plus_crit_expanded1344=0.8902. Δ(candidateUV_plus_crit_expanded1344−CNN)=-0.2176 (▲ better), improved on 168/168 samples.

### Q2. Did candidateUV_plus_crit_expanded1344 improve scalar speed or WPD metrics?

- **WPD MAE**: CNN=231.6709, candidateUV_plus_crit_expanded1344=170.1947. Δ=-61.4762 (▲ better), improved on 168/168 samples.
- **WPD W1**: CNN=45.2713, candidateUV_plus_crit_expanded1344=25.8500. Δ=-19.4213 (▲ better), improved on 161/168 samples.
- **WPD bias abs**: CNN=35.3439, candidateUV_plus_crit_expanded1344=17.8387. Δ=-17.5052 (▲ better), improved on 141/168 samples.

### Q3. Did candidateUV_plus_crit_expanded1344 improve gradient metrics?

- **Gradient MAE**: CNN=0.3491, candidateUV_plus_crit_expanded1344=0.3229. Δ=-0.0261 (▲ better), improved on 168/168 samples.
- **Gradient W1**: CNN=0.2329, candidateUV_plus_crit_expanded1344=0.2335. Δ=+0.0006 (▼ worse), improved on 111/168 samples.
- **Gradient kurtosis abs Δ**: CNN=3.7004, candidateUV_plus_crit_expanded1344=4.6949. Δ=+0.9945 (▼ worse), improved on 68/168 samples.

### Q4. Did candidateUV_plus_crit_expanded1344 improve exceedance metrics, especially s>5?

- **Exceedance abs Δ s>5**: CNN=0.00424, candidateUV_plus_crit_expanded1344=0.00810. Δ=+0.0039 (▼ worse), improved on 13/168 samples.
- **Exceedance abs Δ s>10**: CNN=0.00658, candidateUV_plus_crit_expanded1344=0.00816. Δ=+0.0016 (▼ worse), improved on 79/168 samples.
- **Exceedance abs Δ s>15**: CNN=0.00621, candidateUV_plus_crit_expanded1344=0.00205. Δ=-0.0042 (▲ better), improved on 135/168 samples.
- **Exceedance abs Δ p90**: CNN=0.01025, candidateUV_plus_crit_expanded1344=0.00242. Δ=-0.0078 (▲ better), improved on 151/168 samples.

### Q5. Did candidateUV_plus_crit_expanded1344 move component-count behavior toward GT?

- **Component-count curve L1**: CNN=115.5278, candidateUV_plus_crit_expanded1344=132.1756. Δ=+16.6478 (▼ worse), improved on 13/168 samples.
- **Comp-count abs Δ t5**: CNN=64.3810, candidateUV_plus_crit_expanded1344=69.1250. Δ=+4.7440 (▼ worse), improved on 35/168 samples.
- **Comp-count abs Δ t10**: CNN=187.1667, candidateUV_plus_crit_expanded1344=228.4226. Δ=+41.2560 (▼ worse), improved on 3/168 samples.
- **Comp-count abs Δ t15**: CNN=116.8095, candidateUV_plus_crit_expanded1344=128.1607. Δ=+11.3512 (▼ worse), improved on 65/168 samples.

### Q6. Did candidateUV_plus_crit_expanded1344 improve PD or MT distances?

PD and MT distances for `candidateUV_plus_crit_expanded1344` are not available in this evaluation (TTK was not re-run on these outputs). Pre-computed values for CNN and GAN:

- **PD distance**: CNN=27.4063, GAN=20.8641, candidateUV_plus_crit_expanded1344=N/A (requires TTK run)
- **MT distance**: CNN=5.8678, GAN=8.3481, candidateUV_plus_crit_expanded1344=N/A (requires TTK run)

To compute PD/MT for `candidateUV_plus_crit_expanded1344`: convert outputs to VTI via `scripts/convert_phire_to_vti.py`, then run `scripts/compute_composite_tree_distance.py`.

### Q7. Did candidateUV_plus_crit_expanded1344 change the adjacent clusters (samples 10–13, 90–93, 162–163)?

See `adjacent_cluster_table_candidateUV_plus_crit_expanded1344.csv` for full detail. Summary of PSNRuv, speed MAE, and gradient MAE changes:

| Sample | Δ PSNRuv | Δ Speed MAE | Δ Grad MAE | Δ Exceed s>5 |
|---|---|---|---|---|
| 10 | 2.355 | -0.1915 | -0.0260 | 0.00552 |
| 11 | 2.255 | -0.1884 | -0.0266 | 0.00500 |
| 12 | 2.300 | -0.1889 | -0.0310 | 0.00552 |
| 13 | 2.226 | -0.1803 | -0.0282 | 0.00258 |
| 90 | 2.087 | -0.1343 | -0.0202 | 0.00403 |
| 91 | 2.091 | -0.1384 | -0.0198 | 0.00805 |
| 92 | 2.066 | -0.1487 | -0.0213 | 0.01202 |
| 93 | 2.006 | -0.1590 | -0.0194 | 0.00928 |
| 162 | 3.494 | -0.1454 | -0.0241 | -0.00512 |
| 163 | 3.414 | -0.1455 | -0.0259 | -0.00388 |

## Pairwise summary: CNN vs candidateUV_plus_crit_expanded1344

| Metric | Mean CNN | Mean candidateUV_plus_crit_expanded1344 | Δ | N improved | N worsened |
|---|---|---|---|---|---|
| psnruv | 31.1925 | 33.3888 | 2.1963 | 168 | 0 |
| ssim | N/A | N/A | N/A | 0 | 0 |
| speed_mae | 0.6941 | 0.5218 | -0.1722 | 168 | 0 |
| speed_rmse | 1.1078 | 0.8902 | -0.2176 | 168 | 0 |
| wpd_bias_abs | 35.3439 | 17.8387 | -17.5052 | 141 | 27 |
| wpd_mae | 231.6709 | 170.1947 | -61.4762 | 168 | 0 |
| wpd_w1 | 45.2713 | 25.8500 | -19.4213 | 161 | 7 |
| psd_log_l2 | 0.8335 | 1.0568 | 0.2233 | 0 | 168 |
| psd_slope_abs_delta | 0.9150 | 1.3842 | 0.4692 | 0 | 168 |
| grad_mae | 0.3491 | 0.3229 | -0.0261 | 168 | 0 |
| grad_w1 | 0.2329 | 0.2335 | 0.0006 | 111 | 57 |
| grad_kurtosis_abs_delta | 3.7004 | 4.6949 | 0.9945 | 68 | 100 |
| exceed_abs_t5 | 0.0042 | 0.0081 | 0.0039 | 13 | 155 |
| exceed_abs_t10 | 0.0066 | 0.0082 | 0.0016 | 79 | 89 |
| exceed_abs_t15 | 0.0062 | 0.0020 | -0.0042 | 135 | 33 |
| exceed_abs_p90 | 0.0103 | 0.0024 | -0.0078 | 151 | 17 |
| comp_curve_l1 | 115.5278 | 132.1756 | 16.6478 | 13 | 155 |
| pd_distance | N/A | N/A | N/A | 0 | 0 |
| mt_distance | N/A | N/A | N/A | 0 | 0 |

## Output files

- `ttk_runs_fixed/topology_finetuning/candidateUV_plus_crit_expanded1344_eval/all_sample_metrics_candidateUV_plus_crit_expanded1344.csv` — per-sample metrics for all methods
- `ttk_runs_fixed/topology_finetuning/candidateUV_plus_crit_expanded1344_eval/pairwise_cnn_vs_candidateUV_plus_crit_expanded1344.csv` — per-metric delta table
- `ttk_runs_fixed/topology_finetuning/candidateUV_plus_crit_expanded1344_eval/winner_counts_candidateUV_plus_crit_expanded1344.csv` — win counts per method
- `ttk_runs_fixed/topology_finetuning/candidateUV_plus_crit_expanded1344_eval/adjacent_cluster_table_candidateUV_plus_crit_expanded1344.csv` — adjacent-cluster detail
- `docs/topology_finetuning_candidateUV_plus_crit_expanded1344_eval.md` — this report

## Notes

- PSNRuv is computed on physical [u,v] arrays using per-sample dynamic data range (GT max − GT min).
- All speed-based metrics use scalar speed = sqrt(u² + v²) in physical units (m/s).
- Exceedance fractions use GT percentile thresholds (p90/p95/p99) computed per sample.
- Component counts use 8-connectivity on speed superlevel sets.
- PD/MT distances for CNN and GAN are loaded from pre-computed merged CSV; `candidateUV_plus_crit_expanded1344` requires a fresh TTK run.
- `candidateUV_plus_crit_expanded1344` was fine-tuned from the pretrained CNN checkpoint. Results may change with different lambda settings or more epochs.
