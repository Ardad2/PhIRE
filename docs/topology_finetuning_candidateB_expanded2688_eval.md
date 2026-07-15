# candidateB_expanded2688 fine-tuning evaluation

**Generated:** 2026-07-14

**Candidate:** `candidateB_expanded2688`

**Samples evaluated:** 168

**Methods compared:** bicubic, cnn, gan, candidateB_expanded2688

> **SSIM note:** SSIM was not computed because skimage could not be imported in the current NumPy environment (`ValueError('numpy.dtype size changed, may indicate binary incompatibility. Expected 96 from C header, got 88 from PyObject')`). All SSIM values are NaN.

> Note: PD/MT distances for `candidateB_expanded2688` are N/A — TTK has not been re-run on these outputs. Values for CNN/GAN are loaded from the pre-computed merged CSV.

## Summary metrics table

| Metric | Bicubic | Cnn | Gan | Candidateb_expanded2688 |
|---|---|---|---|---|
| PSNRuv (dB) | 32.2202 | 31.1925 | 29.1380 | 33.5198 |
| SSIM on speed | N/A | N/A | N/A | N/A |
| Speed MAE (m/s) | 0.5963 | 0.6941 | 0.9026 | 0.5141 |
| Speed RMSE (m/s) | 1.0156 | 1.1078 | 1.3775 | 0.8763 |
| WPD MAE (m³/s³) | 193.5403 | 231.6709 | 310.7328 | 164.5935 |
| WPD Wasserstein-1 | 55.5103 | 45.2713 | 85.6191 | 26.6869 |
| WPD bias abs (m³/s³) | 41.2244 | 35.3439 | 78.7236 | 19.6891 |
| Gradient MAE | 0.3696 | 0.3491 | 0.3806 | 0.3124 |
| Gradient W1 | 0.3282 | 0.2329 | 0.0564 | 0.1996 |
| Gradient kurtosis abs Δ | 6.6163 | 3.7004 | 4.2010 | 2.9374 |
| PSD log-L2 | 1.3625 | 0.8335 | 0.5139 | 0.9754 |
| PSD slope abs Δ | 1.6684 | 0.9150 | 0.9482 | 1.3177 |
| Exceedance abs Δ s>5 | 0.0095 | 0.0042 | 0.0082 | 0.0049 |
| Exceedance abs Δ s>10 | 0.0085 | 0.0066 | 0.0243 | 0.0036 |
| Exceedance abs Δ s>15 | 0.0074 | 0.0062 | 0.0096 | 0.0035 |
| Exceedance abs Δ p90 | 0.0109 | 0.0103 | 0.0123 | 0.0051 |
| Component-count curve L1 | 173.1855 | 115.5278 | 124.6567 | 112.3254 |
| PD distance (TTK) | N/A | 27.4063 | 20.8641 | N/A |
| MT distance (TTK) | N/A | 5.8678 | 8.3481 | N/A |

## 7 Key evaluation questions

### Q1. Did candidateB_expanded2688 preserve CNN's direct-fidelity advantage?

- **PSNRuv**: CNN=31.1925, GAN=29.1380, candidateB_expanded2688=33.5198. Δ(candidateB_expanded2688−CNN)=+2.3273 (▲ better), improved on 168/168 samples.
- **SSIM**: CNN=N/A, GAN=N/A, candidateB_expanded2688=N/A. Δ(candidateB_expanded2688−CNN)=N/A, improved on 0/0 samples.
- **Speed MAE**: CNN=0.6941, GAN=0.9026, candidateB_expanded2688=0.5141. Δ(candidateB_expanded2688−CNN)=-0.1800 (▲ better), improved on 168/168 samples.
- **Speed RMSE**: CNN=1.1078, GAN=1.3775, candidateB_expanded2688=0.8763. Δ(candidateB_expanded2688−CNN)=-0.2315 (▲ better), improved on 168/168 samples.

### Q2. Did candidateB_expanded2688 improve scalar speed or WPD metrics?

- **WPD MAE**: CNN=231.6709, candidateB_expanded2688=164.5935. Δ=-67.0774 (▲ better), improved on 168/168 samples.
- **WPD W1**: CNN=45.2713, candidateB_expanded2688=26.6869. Δ=-18.5843 (▲ better), improved on 138/168 samples.
- **WPD bias abs**: CNN=35.3439, candidateB_expanded2688=19.6891. Δ=-15.6549 (▲ better), improved on 123/168 samples.

### Q3. Did candidateB_expanded2688 improve gradient metrics?

- **Gradient MAE**: CNN=0.3491, candidateB_expanded2688=0.3124. Δ=-0.0367 (▲ better), improved on 168/168 samples.
- **Gradient W1**: CNN=0.2329, candidateB_expanded2688=0.1996. Δ=-0.0333 (▲ better), improved on 150/168 samples.
- **Gradient kurtosis abs Δ**: CNN=3.7004, candidateB_expanded2688=2.9374. Δ=-0.7630 (▲ better), improved on 97/168 samples.

### Q4. Did candidateB_expanded2688 improve exceedance metrics, especially s>5?

- **Exceedance abs Δ s>5**: CNN=0.00424, candidateB_expanded2688=0.00489. Δ=+0.0006 (▼ worse), improved on 63/168 samples.
- **Exceedance abs Δ s>10**: CNN=0.00658, candidateB_expanded2688=0.00365. Δ=-0.0029 (▲ better), improved on 129/168 samples.
- **Exceedance abs Δ s>15**: CNN=0.00621, candidateB_expanded2688=0.00347. Δ=-0.0027 (▲ better), improved on 126/168 samples.
- **Exceedance abs Δ p90**: CNN=0.01025, candidateB_expanded2688=0.00509. Δ=-0.0052 (▲ better), improved on 140/168 samples.

### Q5. Did candidateB_expanded2688 move component-count behavior toward GT?

- **Component-count curve L1**: CNN=115.5278, candidateB_expanded2688=112.3254. Δ=-3.2024 (▲ better), improved on 112/168 samples.
- **Comp-count abs Δ t5**: CNN=64.3810, candidateB_expanded2688=58.7143. Δ=-5.6667 (▲ better), improved on 119/168 samples.
- **Comp-count abs Δ t10**: CNN=187.1667, candidateB_expanded2688=189.2083. Δ=+2.0417 (▼ worse), improved on 76/168 samples.
- **Comp-count abs Δ t15**: CNN=116.8095, candidateB_expanded2688=110.2500. Δ=-6.5595 (▲ better), improved on 114/168 samples.

### Q6. Did candidateB_expanded2688 improve PD or MT distances?

PD and MT distances for `candidateB_expanded2688` are not available in this evaluation (TTK was not re-run on these outputs). Pre-computed values for CNN and GAN:

- **PD distance**: CNN=27.4063, GAN=20.8641, candidateB_expanded2688=N/A (requires TTK run)
- **MT distance**: CNN=5.8678, GAN=8.3481, candidateB_expanded2688=N/A (requires TTK run)

To compute PD/MT for `candidateB_expanded2688`: convert outputs to VTI via `scripts/convert_phire_to_vti.py`, then run `scripts/compute_composite_tree_distance.py`.

### Q7. Did candidateB_expanded2688 change the adjacent clusters (samples 10–13, 90–93, 162–163)?

See `adjacent_cluster_table_candidateB_expanded2688.csv` for full detail. Summary of PSNRuv, speed MAE, and gradient MAE changes:

| Sample | Δ PSNRuv | Δ Speed MAE | Δ Grad MAE | Δ Exceed s>5 |
|---|---|---|---|---|
| 10 | 2.505 | -0.1983 | -0.0371 | 0.00268 |
| 11 | 2.408 | -0.1954 | -0.0380 | 0.00162 |
| 12 | 2.430 | -0.1960 | -0.0421 | 0.00177 |
| 13 | 2.339 | -0.1886 | -0.0394 | 0.00006 |
| 90 | 2.145 | -0.1396 | -0.0292 | -0.00212 |
| 91 | 2.180 | -0.1443 | -0.0289 | 0.00201 |
| 92 | 2.181 | -0.1561 | -0.0314 | 0.00618 |
| 93 | 2.150 | -0.1664 | -0.0323 | 0.00343 |
| 162 | 3.622 | -0.1476 | -0.0260 | -0.00581 |
| 163 | 3.548 | -0.1475 | -0.0282 | -0.00349 |

## Pairwise summary: CNN vs candidateB_expanded2688

| Metric | Mean CNN | Mean candidateB_expanded2688 | Δ | N improved | N worsened |
|---|---|---|---|---|---|
| psnruv | 31.1925 | 33.5198 | 2.3273 | 168 | 0 |
| ssim | N/A | N/A | N/A | 0 | 0 |
| speed_mae | 0.6941 | 0.5141 | -0.1800 | 168 | 0 |
| speed_rmse | 1.1078 | 0.8763 | -0.2315 | 168 | 0 |
| wpd_bias_abs | 35.3439 | 19.6891 | -15.6549 | 123 | 45 |
| wpd_mae | 231.6709 | 164.5935 | -67.0774 | 168 | 0 |
| wpd_w1 | 45.2713 | 26.6869 | -18.5843 | 138 | 30 |
| psd_log_l2 | 0.8335 | 0.9754 | 0.1419 | 2 | 166 |
| psd_slope_abs_delta | 0.9150 | 1.3177 | 0.4027 | 0 | 168 |
| grad_mae | 0.3491 | 0.3124 | -0.0367 | 168 | 0 |
| grad_w1 | 0.2329 | 0.1996 | -0.0333 | 150 | 18 |
| grad_kurtosis_abs_delta | 3.7004 | 2.9374 | -0.7630 | 97 | 71 |
| exceed_abs_t5 | 0.0042 | 0.0049 | 0.0006 | 63 | 105 |
| exceed_abs_t10 | 0.0066 | 0.0036 | -0.0029 | 129 | 39 |
| exceed_abs_t15 | 0.0062 | 0.0035 | -0.0027 | 126 | 42 |
| exceed_abs_p90 | 0.0103 | 0.0051 | -0.0052 | 140 | 28 |
| comp_curve_l1 | 115.5278 | 112.3254 | -3.2024 | 112 | 56 |
| pd_distance | N/A | N/A | N/A | 0 | 0 |
| mt_distance | N/A | N/A | N/A | 0 | 0 |

## Output files

- `ttk_runs_fixed/topology_finetuning/candidateB_expanded2688_eval/all_sample_metrics_candidateB_expanded2688.csv` — per-sample metrics for all methods
- `ttk_runs_fixed/topology_finetuning/candidateB_expanded2688_eval/pairwise_cnn_vs_candidateB_expanded2688.csv` — per-metric delta table
- `ttk_runs_fixed/topology_finetuning/candidateB_expanded2688_eval/winner_counts_candidateB_expanded2688.csv` — win counts per method
- `ttk_runs_fixed/topology_finetuning/candidateB_expanded2688_eval/adjacent_cluster_table_candidateB_expanded2688.csv` — adjacent-cluster detail
- `docs/topology_finetuning_candidateB_expanded2688_eval.md` — this report

## Notes

- PSNRuv is computed on physical [u,v] arrays using per-sample dynamic data range (GT max − GT min).
- All speed-based metrics use scalar speed = sqrt(u² + v²) in physical units (m/s).
- Exceedance fractions use GT percentile thresholds (p90/p95/p99) computed per sample.
- Component counts use 8-connectivity on speed superlevel sets.
- PD/MT distances for CNN and GAN are loaded from pre-computed merged CSV; `candidateB_expanded2688` requires a fresh TTK run.
- `candidateB_expanded2688` was fine-tuned from the pretrained CNN checkpoint. Results may change with different lambda settings or more epochs.
