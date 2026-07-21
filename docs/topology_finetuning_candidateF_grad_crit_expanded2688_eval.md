# candidateF_grad_crit_expanded2688 fine-tuning evaluation

**Generated:** 2026-07-19

**Candidate:** `candidateF_grad_crit_expanded2688`

**Samples evaluated:** 168

**Methods compared:** bicubic, cnn, gan, candidateF_grad_crit_expanded2688

> **SSIM note:** SSIM was not computed because skimage could not be imported in the current NumPy environment (`ValueError('numpy.dtype size changed, may indicate binary incompatibility. Expected 96 from C header, got 88 from PyObject')`). All SSIM values are NaN.

> Note: PD/MT distances for `candidateF_grad_crit_expanded2688` are N/A — TTK has not been re-run on these outputs. Values for CNN/GAN are loaded from the pre-computed merged CSV.

## Summary metrics table

| Metric | Bicubic | Cnn | Gan | Candidatef_grad_crit_expanded2688 |
|---|---|---|---|---|
| PSNRuv (dB) | 32.2202 | 31.1925 | 29.1380 | 33.3042 |
| SSIM on speed | N/A | N/A | N/A | N/A |
| Speed MAE (m/s) | 0.5963 | 0.6941 | 0.9026 | 0.5326 |
| Speed RMSE (m/s) | 1.0156 | 1.1078 | 1.3775 | 0.9026 |
| WPD MAE (m³/s³) | 193.5403 | 231.6709 | 310.7328 | 174.1976 |
| WPD Wasserstein-1 | 55.5103 | 45.2713 | 85.6191 | 32.0205 |
| WPD bias abs (m³/s³) | 41.2244 | 35.3439 | 78.7236 | 28.3295 |
| Gradient MAE | 0.3696 | 0.3491 | 0.3806 | 0.3087 |
| Gradient W1 | 0.3282 | 0.2329 | 0.0564 | 0.1755 |
| Gradient kurtosis abs Δ | 6.6163 | 3.7004 | 4.2010 | 2.5256 |
| PSD log-L2 | 1.3625 | 0.8335 | 0.5139 | 0.9029 |
| PSD slope abs Δ | 1.6684 | 0.9150 | 0.9482 | 1.6221 |
| Exceedance abs Δ s>5 | 0.0095 | 0.0042 | 0.0082 | 0.0063 |
| Exceedance abs Δ s>10 | 0.0085 | 0.0066 | 0.0243 | 0.0086 |
| Exceedance abs Δ s>15 | 0.0074 | 0.0062 | 0.0096 | 0.0033 |
| Exceedance abs Δ p90 | 0.0109 | 0.0103 | 0.0123 | 0.0046 |
| Component-count curve L1 | 173.1855 | 115.5278 | 124.6567 | 96.7510 |
| PD distance (TTK) | N/A | 27.4063 | 20.8641 | N/A |
| MT distance (TTK) | N/A | 5.8678 | 8.3481 | N/A |

## 7 Key evaluation questions

### Q1. Did candidateF_grad_crit_expanded2688 preserve CNN's direct-fidelity advantage?

- **PSNRuv**: CNN=31.1925, GAN=29.1380, candidateF_grad_crit_expanded2688=33.3042. Δ(candidateF_grad_crit_expanded2688−CNN)=+2.1117 (▲ better), improved on 168/168 samples.
- **SSIM**: CNN=N/A, GAN=N/A, candidateF_grad_crit_expanded2688=N/A. Δ(candidateF_grad_crit_expanded2688−CNN)=N/A, improved on 0/0 samples.
- **Speed MAE**: CNN=0.6941, GAN=0.9026, candidateF_grad_crit_expanded2688=0.5326. Δ(candidateF_grad_crit_expanded2688−CNN)=-0.1615 (▲ better), improved on 168/168 samples.
- **Speed RMSE**: CNN=1.1078, GAN=1.3775, candidateF_grad_crit_expanded2688=0.9026. Δ(candidateF_grad_crit_expanded2688−CNN)=-0.2052 (▲ better), improved on 168/168 samples.

### Q2. Did candidateF_grad_crit_expanded2688 improve scalar speed or WPD metrics?

- **WPD MAE**: CNN=231.6709, candidateF_grad_crit_expanded2688=174.1976. Δ=-57.4733 (▲ better), improved on 168/168 samples.
- **WPD W1**: CNN=45.2713, candidateF_grad_crit_expanded2688=32.0205. Δ=-13.2508 (▲ better), improved on 150/168 samples.
- **WPD bias abs**: CNN=35.3439, candidateF_grad_crit_expanded2688=28.3295. Δ=-7.0144 (▲ better), improved on 133/168 samples.

### Q3. Did candidateF_grad_crit_expanded2688 improve gradient metrics?

- **Gradient MAE**: CNN=0.3491, candidateF_grad_crit_expanded2688=0.3087. Δ=-0.0404 (▲ better), improved on 168/168 samples.
- **Gradient W1**: CNN=0.2329, candidateF_grad_crit_expanded2688=0.1755. Δ=-0.0574 (▲ better), improved on 154/168 samples.
- **Gradient kurtosis abs Δ**: CNN=3.7004, candidateF_grad_crit_expanded2688=2.5256. Δ=-1.1748 (▲ better), improved on 103/168 samples.

### Q4. Did candidateF_grad_crit_expanded2688 improve exceedance metrics, especially s>5?

- **Exceedance abs Δ s>5**: CNN=0.00424, candidateF_grad_crit_expanded2688=0.00627. Δ=+0.0020 (▼ worse), improved on 38/168 samples.
- **Exceedance abs Δ s>10**: CNN=0.00658, candidateF_grad_crit_expanded2688=0.00863. Δ=+0.0021 (▼ worse), improved on 77/168 samples.
- **Exceedance abs Δ s>15**: CNN=0.00621, candidateF_grad_crit_expanded2688=0.00326. Δ=-0.0030 (▲ better), improved on 130/168 samples.
- **Exceedance abs Δ p90**: CNN=0.01025, candidateF_grad_crit_expanded2688=0.00458. Δ=-0.0057 (▲ better), improved on 135/168 samples.

### Q5. Did candidateF_grad_crit_expanded2688 move component-count behavior toward GT?

- **Component-count curve L1**: CNN=115.5278, candidateF_grad_crit_expanded2688=96.7510. Δ=-18.7768 (▲ better), improved on 150/168 samples.
- **Comp-count abs Δ t5**: CNN=64.3810, candidateF_grad_crit_expanded2688=57.1726. Δ=-7.2083 (▲ better), improved on 121/168 samples.
- **Comp-count abs Δ t10**: CNN=187.1667, candidateF_grad_crit_expanded2688=179.1667. Δ=-8.0000 (▲ better), improved on 102/168 samples.
- **Comp-count abs Δ t15**: CNN=116.8095, candidateF_grad_crit_expanded2688=86.1429. Δ=-30.6667 (▲ better), improved on 143/168 samples.

### Q6. Did candidateF_grad_crit_expanded2688 improve PD or MT distances?

PD and MT distances for `candidateF_grad_crit_expanded2688` are not available in this evaluation (TTK was not re-run on these outputs). Pre-computed values for CNN and GAN:

- **PD distance**: CNN=27.4063, GAN=20.8641, candidateF_grad_crit_expanded2688=N/A (requires TTK run)
- **MT distance**: CNN=5.8678, GAN=8.3481, candidateF_grad_crit_expanded2688=N/A (requires TTK run)

To compute PD/MT for `candidateF_grad_crit_expanded2688`: convert outputs to VTI via `scripts/convert_phire_to_vti.py`, then run `scripts/compute_composite_tree_distance.py`.

### Q7. Did candidateF_grad_crit_expanded2688 change the adjacent clusters (samples 10–13, 90–93, 162–163)?

See `adjacent_cluster_table_candidateF_grad_crit_expanded2688.csv` for full detail. Summary of PSNRuv, speed MAE, and gradient MAE changes:

| Sample | Δ PSNRuv | Δ Speed MAE | Δ Grad MAE | Δ Exceed s>5 |
|---|---|---|---|---|
| 10 | 2.232 | -0.1734 | -0.0399 | 0.00402 |
| 11 | 2.162 | -0.1715 | -0.0407 | 0.00254 |
| 12 | 2.180 | -0.1714 | -0.0448 | 0.00305 |
| 13 | 2.100 | -0.1640 | -0.0418 | 0.00110 |
| 90 | 1.972 | -0.1254 | -0.0310 | 0.00368 |
| 91 | 1.997 | -0.1302 | -0.0309 | 0.00821 |
| 92 | 1.964 | -0.1391 | -0.0340 | 0.01215 |
| 93 | 1.892 | -0.1474 | -0.0358 | 0.00851 |
| 162 | 3.454 | -0.1422 | -0.0265 | -0.00317 |
| 163 | 3.378 | -0.1421 | -0.0289 | 0.00090 |

## Pairwise summary: CNN vs candidateF_grad_crit_expanded2688

| Metric | Mean CNN | Mean candidateF_grad_crit_expanded2688 | Δ | N improved | N worsened |
|---|---|---|---|---|---|
| psnruv | 31.1925 | 33.3042 | 2.1117 | 168 | 0 |
| ssim | N/A | N/A | N/A | 0 | 0 |
| speed_mae | 0.6941 | 0.5326 | -0.1615 | 168 | 0 |
| speed_rmse | 1.1078 | 0.9026 | -0.2052 | 168 | 0 |
| wpd_bias_abs | 35.3439 | 28.3295 | -7.0144 | 133 | 35 |
| wpd_mae | 231.6709 | 174.1976 | -57.4733 | 168 | 0 |
| wpd_w1 | 45.2713 | 32.0205 | -13.2508 | 150 | 18 |
| psd_log_l2 | 0.8335 | 0.9029 | 0.0693 | 35 | 133 |
| psd_slope_abs_delta | 0.9150 | 1.6221 | 0.7071 | 0 | 168 |
| grad_mae | 0.3491 | 0.3087 | -0.0404 | 168 | 0 |
| grad_w1 | 0.2329 | 0.1755 | -0.0574 | 154 | 14 |
| grad_kurtosis_abs_delta | 3.7004 | 2.5256 | -1.1748 | 103 | 65 |
| exceed_abs_t5 | 0.0042 | 0.0063 | 0.0020 | 38 | 130 |
| exceed_abs_t10 | 0.0066 | 0.0086 | 0.0021 | 77 | 91 |
| exceed_abs_t15 | 0.0062 | 0.0033 | -0.0030 | 130 | 38 |
| exceed_abs_p90 | 0.0103 | 0.0046 | -0.0057 | 135 | 33 |
| comp_curve_l1 | 115.5278 | 96.7510 | -18.7768 | 150 | 18 |
| pd_distance | N/A | N/A | N/A | 0 | 0 |
| mt_distance | N/A | N/A | N/A | 0 | 0 |

## Output files

- `ttk_runs_fixed/topology_finetuning/candidateF_grad_crit_expanded2688_eval/all_sample_metrics_candidateF_grad_crit_expanded2688.csv` — per-sample metrics for all methods
- `ttk_runs_fixed/topology_finetuning/candidateF_grad_crit_expanded2688_eval/pairwise_cnn_vs_candidateF_grad_crit_expanded2688.csv` — per-metric delta table
- `ttk_runs_fixed/topology_finetuning/candidateF_grad_crit_expanded2688_eval/winner_counts_candidateF_grad_crit_expanded2688.csv` — win counts per method
- `ttk_runs_fixed/topology_finetuning/candidateF_grad_crit_expanded2688_eval/adjacent_cluster_table_candidateF_grad_crit_expanded2688.csv` — adjacent-cluster detail
- `docs/topology_finetuning_candidateF_grad_crit_expanded2688_eval.md` — this report

## Notes

- PSNRuv is computed on physical [u,v] arrays using per-sample dynamic data range (GT max − GT min).
- All speed-based metrics use scalar speed = sqrt(u² + v²) in physical units (m/s).
- Exceedance fractions use GT percentile thresholds (p90/p95/p99) computed per sample.
- Component counts use 8-connectivity on speed superlevel sets.
- PD/MT distances for CNN and GAN are loaded from pre-computed merged CSV; `candidateF_grad_crit_expanded2688` requires a fresh TTK run.
- `candidateF_grad_crit_expanded2688` was fine-tuned from the pretrained CNN checkpoint. Results may change with different lambda settings or more epochs.
