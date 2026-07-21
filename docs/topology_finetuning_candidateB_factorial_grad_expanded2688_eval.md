# candidateB_factorial_grad_expanded2688 fine-tuning evaluation

**Generated:** 2026-07-15

**Candidate:** `candidateB_factorial_grad_expanded2688`

**Samples evaluated:** 168

**Methods compared:** bicubic, cnn, gan, candidateB_factorial_grad_expanded2688

> **SSIM note:** SSIM was not computed because skimage could not be imported in the current NumPy environment (`ValueError('numpy.dtype size changed, may indicate binary incompatibility. Expected 96 from C header, got 88 from PyObject')`). All SSIM values are NaN.

> Note: PD/MT distances for `candidateB_factorial_grad_expanded2688` are N/A — TTK has not been re-run on these outputs. Values for CNN/GAN are loaded from the pre-computed merged CSV.

## Summary metrics table

| Metric | Bicubic | Cnn | Gan | Candidateb_factorial_grad_expanded2688 |
|---|---|---|---|---|
| PSNRuv (dB) | 32.2202 | 31.1925 | 29.1380 | 33.3698 |
| SSIM on speed | N/A | N/A | N/A | N/A |
| Speed MAE (m/s) | 0.5963 | 0.6941 | 0.9026 | 0.5286 |
| Speed RMSE (m/s) | 1.0156 | 1.1078 | 1.3775 | 0.8963 |
| WPD MAE (m³/s³) | 193.5403 | 231.6709 | 310.7328 | 169.5510 |
| WPD Wasserstein-1 | 55.5103 | 45.2713 | 85.6191 | 22.8114 |
| WPD bias abs (m³/s³) | 41.2244 | 35.3439 | 78.7236 | 17.5039 |
| Gradient MAE | 0.3696 | 0.3491 | 0.3806 | 0.3088 |
| Gradient W1 | 0.3282 | 0.2329 | 0.0564 | 0.1751 |
| Gradient kurtosis abs Δ | 6.6163 | 3.7004 | 4.2010 | 2.5098 |
| PSD log-L2 | 1.3625 | 0.8335 | 0.5139 | 0.8930 |
| PSD slope abs Δ | 1.6684 | 0.9150 | 0.9482 | 1.3907 |
| Exceedance abs Δ s>5 | 0.0095 | 0.0042 | 0.0082 | 0.0036 |
| Exceedance abs Δ s>10 | 0.0085 | 0.0066 | 0.0243 | 0.0036 |
| Exceedance abs Δ s>15 | 0.0074 | 0.0062 | 0.0096 | 0.0031 |
| Exceedance abs Δ p90 | 0.0109 | 0.0103 | 0.0123 | 0.0041 |
| Component-count curve L1 | 173.1855 | 115.5278 | 124.6567 | 97.1349 |
| PD distance (TTK) | N/A | 27.4063 | 20.8641 | N/A |
| MT distance (TTK) | N/A | 5.8678 | 8.3481 | N/A |

## 7 Key evaluation questions

### Q1. Did candidateB_factorial_grad_expanded2688 preserve CNN's direct-fidelity advantage?

- **PSNRuv**: CNN=31.1925, GAN=29.1380, candidateB_factorial_grad_expanded2688=33.3698. Δ(candidateB_factorial_grad_expanded2688−CNN)=+2.1773 (▲ better), improved on 168/168 samples.
- **SSIM**: CNN=N/A, GAN=N/A, candidateB_factorial_grad_expanded2688=N/A. Δ(candidateB_factorial_grad_expanded2688−CNN)=N/A, improved on 0/0 samples.
- **Speed MAE**: CNN=0.6941, GAN=0.9026, candidateB_factorial_grad_expanded2688=0.5286. Δ(candidateB_factorial_grad_expanded2688−CNN)=-0.1655 (▲ better), improved on 168/168 samples.
- **Speed RMSE**: CNN=1.1078, GAN=1.3775, candidateB_factorial_grad_expanded2688=0.8963. Δ(candidateB_factorial_grad_expanded2688−CNN)=-0.2115 (▲ better), improved on 168/168 samples.

### Q2. Did candidateB_factorial_grad_expanded2688 improve scalar speed or WPD metrics?

- **WPD MAE**: CNN=231.6709, candidateB_factorial_grad_expanded2688=169.5510. Δ=-62.1199 (▲ better), improved on 168/168 samples.
- **WPD W1**: CNN=45.2713, candidateB_factorial_grad_expanded2688=22.8114. Δ=-22.4599 (▲ better), improved on 136/168 samples.
- **WPD bias abs**: CNN=35.3439, candidateB_factorial_grad_expanded2688=17.5039. Δ=-17.8401 (▲ better), improved on 118/168 samples.

### Q3. Did candidateB_factorial_grad_expanded2688 improve gradient metrics?

- **Gradient MAE**: CNN=0.3491, candidateB_factorial_grad_expanded2688=0.3088. Δ=-0.0402 (▲ better), improved on 168/168 samples.
- **Gradient W1**: CNN=0.2329, candidateB_factorial_grad_expanded2688=0.1751. Δ=-0.0578 (▲ better), improved on 154/168 samples.
- **Gradient kurtosis abs Δ**: CNN=3.7004, candidateB_factorial_grad_expanded2688=2.5098. Δ=-1.1906 (▲ better), improved on 97/168 samples.

### Q4. Did candidateB_factorial_grad_expanded2688 improve exceedance metrics, especially s>5?

- **Exceedance abs Δ s>5**: CNN=0.00424, candidateB_factorial_grad_expanded2688=0.00361. Δ=-0.0006 (▲ better), improved on 105/168 samples.
- **Exceedance abs Δ s>10**: CNN=0.00658, candidateB_factorial_grad_expanded2688=0.00364. Δ=-0.0029 (▲ better), improved on 117/168 samples.
- **Exceedance abs Δ s>15**: CNN=0.00621, candidateB_factorial_grad_expanded2688=0.00306. Δ=-0.0031 (▲ better), improved on 123/168 samples.
- **Exceedance abs Δ p90**: CNN=0.01025, candidateB_factorial_grad_expanded2688=0.00412. Δ=-0.0061 (▲ better), improved on 137/168 samples.

### Q5. Did candidateB_factorial_grad_expanded2688 move component-count behavior toward GT?

- **Component-count curve L1**: CNN=115.5278, candidateB_factorial_grad_expanded2688=97.1349. Δ=-18.3929 (▲ better), improved on 146/168 samples.
- **Comp-count abs Δ t5**: CNN=64.3810, candidateB_factorial_grad_expanded2688=54.0060. Δ=-10.3750 (▲ better), improved on 135/168 samples.
- **Comp-count abs Δ t10**: CNN=187.1667, candidateB_factorial_grad_expanded2688=167.7321. Δ=-19.4345 (▲ better), improved on 129/168 samples.
- **Comp-count abs Δ t15**: CNN=116.8095, candidateB_factorial_grad_expanded2688=93.8750. Δ=-22.9345 (▲ better), improved on 137/168 samples.

### Q6. Did candidateB_factorial_grad_expanded2688 improve PD or MT distances?

PD and MT distances for `candidateB_factorial_grad_expanded2688` are not available in this evaluation (TTK was not re-run on these outputs). Pre-computed values for CNN and GAN:

- **PD distance**: CNN=27.4063, GAN=20.8641, candidateB_factorial_grad_expanded2688=N/A (requires TTK run)
- **MT distance**: CNN=5.8678, GAN=8.3481, candidateB_factorial_grad_expanded2688=N/A (requires TTK run)

To compute PD/MT for `candidateB_factorial_grad_expanded2688`: convert outputs to VTI via `scripts/convert_phire_to_vti.py`, then run `scripts/compute_composite_tree_distance.py`.

### Q7. Did candidateB_factorial_grad_expanded2688 change the adjacent clusters (samples 10–13, 90–93, 162–163)?

See `adjacent_cluster_table_candidateB_factorial_grad_expanded2688.csv` for full detail. Summary of PSNRuv, speed MAE, and gradient MAE changes:

| Sample | Δ PSNRuv | Δ Speed MAE | Δ Grad MAE | Δ Exceed s>5 |
|---|---|---|---|---|
| 10 | 2.360 | -0.1851 | -0.0392 | 0.00153 |
| 11 | 2.260 | -0.1813 | -0.0400 | 0.00013 |
| 12 | 2.284 | -0.1820 | -0.0445 | 0.00037 |
| 13 | 2.188 | -0.1744 | -0.0413 | -0.00150 |
| 90 | 1.990 | -0.1266 | -0.0331 | -0.00341 |
| 91 | 2.025 | -0.1310 | -0.0334 | 0.00068 |
| 92 | 2.017 | -0.1414 | -0.0363 | 0.00432 |
| 93 | 1.968 | -0.1496 | -0.0384 | 0.00118 |
| 162 | 3.514 | -0.1420 | -0.0255 | -0.00464 |
| 163 | 3.446 | -0.1427 | -0.0276 | -0.00178 |

## Pairwise summary: CNN vs candidateB_factorial_grad_expanded2688

| Metric | Mean CNN | Mean candidateB_factorial_grad_expanded2688 | Δ | N improved | N worsened |
|---|---|---|---|---|---|
| psnruv | 31.1925 | 33.3698 | 2.1773 | 168 | 0 |
| ssim | N/A | N/A | N/A | 0 | 0 |
| speed_mae | 0.6941 | 0.5286 | -0.1655 | 168 | 0 |
| speed_rmse | 1.1078 | 0.8963 | -0.2115 | 168 | 0 |
| wpd_bias_abs | 35.3439 | 17.5039 | -17.8401 | 118 | 50 |
| wpd_mae | 231.6709 | 169.5510 | -62.1199 | 168 | 0 |
| wpd_w1 | 45.2713 | 22.8114 | -22.4599 | 136 | 32 |
| psd_log_l2 | 0.8335 | 0.8930 | 0.0595 | 72 | 96 |
| psd_slope_abs_delta | 0.9150 | 1.3907 | 0.4757 | 0 | 168 |
| grad_mae | 0.3491 | 0.3088 | -0.0402 | 168 | 0 |
| grad_w1 | 0.2329 | 0.1751 | -0.0578 | 154 | 14 |
| grad_kurtosis_abs_delta | 3.7004 | 2.5098 | -1.1906 | 97 | 71 |
| exceed_abs_t5 | 0.0042 | 0.0036 | -0.0006 | 105 | 63 |
| exceed_abs_t10 | 0.0066 | 0.0036 | -0.0029 | 117 | 51 |
| exceed_abs_t15 | 0.0062 | 0.0031 | -0.0031 | 123 | 45 |
| exceed_abs_p90 | 0.0103 | 0.0041 | -0.0061 | 137 | 31 |
| comp_curve_l1 | 115.5278 | 97.1349 | -18.3929 | 146 | 22 |
| pd_distance | N/A | N/A | N/A | 0 | 0 |
| mt_distance | N/A | N/A | N/A | 0 | 0 |

## Output files

- `ttk_runs_fixed/topology_finetuning/candidateB_factorial_grad_expanded2688_eval/all_sample_metrics_candidateB_factorial_grad_expanded2688.csv` — per-sample metrics for all methods
- `ttk_runs_fixed/topology_finetuning/candidateB_factorial_grad_expanded2688_eval/pairwise_cnn_vs_candidateB_factorial_grad_expanded2688.csv` — per-metric delta table
- `ttk_runs_fixed/topology_finetuning/candidateB_factorial_grad_expanded2688_eval/winner_counts_candidateB_factorial_grad_expanded2688.csv` — win counts per method
- `ttk_runs_fixed/topology_finetuning/candidateB_factorial_grad_expanded2688_eval/adjacent_cluster_table_candidateB_factorial_grad_expanded2688.csv` — adjacent-cluster detail
- `docs/topology_finetuning_candidateB_factorial_grad_expanded2688_eval.md` — this report

## Notes

- PSNRuv is computed on physical [u,v] arrays using per-sample dynamic data range (GT max − GT min).
- All speed-based metrics use scalar speed = sqrt(u² + v²) in physical units (m/s).
- Exceedance fractions use GT percentile thresholds (p90/p95/p99) computed per sample.
- Component counts use 8-connectivity on speed superlevel sets.
- PD/MT distances for CNN and GAN are loaded from pre-computed merged CSV; `candidateB_factorial_grad_expanded2688` requires a fresh TTK run.
- `candidateB_factorial_grad_expanded2688` was fine-tuned from the pretrained CNN checkpoint. Results may change with different lambda settings or more epochs.
