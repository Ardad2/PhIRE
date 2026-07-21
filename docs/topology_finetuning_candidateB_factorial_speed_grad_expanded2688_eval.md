# candidateB_factorial_speed_grad_expanded2688 fine-tuning evaluation

**Generated:** 2026-07-15

**Candidate:** `candidateB_factorial_speed_grad_expanded2688`

**Samples evaluated:** 168

**Methods compared:** bicubic, cnn, gan, candidateB_factorial_speed_grad_expanded2688

> **SSIM note:** SSIM was not computed because skimage could not be imported in the current NumPy environment (`ValueError('numpy.dtype size changed, may indicate binary incompatibility. Expected 96 from C header, got 88 from PyObject')`). All SSIM values are NaN.

> Note: PD/MT distances for `candidateB_factorial_speed_grad_expanded2688` are N/A — TTK has not been re-run on these outputs. Values for CNN/GAN are loaded from the pre-computed merged CSV.

## Summary metrics table

| Metric | Bicubic | Cnn | Gan | Candidateb_factorial_speed_grad_expanded2688 |
|---|---|---|---|---|
| PSNRuv (dB) | 32.2202 | 31.1925 | 29.1380 | 33.4726 |
| SSIM on speed | N/A | N/A | N/A | N/A |
| Speed MAE (m/s) | 0.5963 | 0.6941 | 0.9026 | 0.5179 |
| Speed RMSE (m/s) | 1.0156 | 1.1078 | 1.3775 | 0.8813 |
| WPD MAE (m³/s³) | 193.5403 | 231.6709 | 310.7328 | 166.0587 |
| WPD Wasserstein-1 | 55.5103 | 45.2713 | 85.6191 | 23.6460 |
| WPD bias abs (m³/s³) | 41.2244 | 35.3439 | 78.7236 | 15.6753 |
| Gradient MAE | 0.3696 | 0.3491 | 0.3806 | 0.3123 |
| Gradient W1 | 0.3282 | 0.2329 | 0.0564 | 0.1959 |
| Gradient kurtosis abs Δ | 6.6163 | 3.7004 | 4.2010 | 2.9340 |
| PSD log-L2 | 1.3625 | 0.8335 | 0.5139 | 0.9694 |
| PSD slope abs Δ | 1.6684 | 0.9150 | 0.9482 | 1.2875 |
| Exceedance abs Δ s>5 | 0.0095 | 0.0042 | 0.0082 | 0.0050 |
| Exceedance abs Δ s>10 | 0.0085 | 0.0066 | 0.0243 | 0.0033 |
| Exceedance abs Δ s>15 | 0.0074 | 0.0062 | 0.0096 | 0.0031 |
| Exceedance abs Δ p90 | 0.0109 | 0.0103 | 0.0123 | 0.0042 |
| Component-count curve L1 | 173.1855 | 115.5278 | 124.6567 | 111.3294 |
| PD distance (TTK) | N/A | 27.4063 | 20.8641 | N/A |
| MT distance (TTK) | N/A | 5.8678 | 8.3481 | N/A |

## 7 Key evaluation questions

### Q1. Did candidateB_factorial_speed_grad_expanded2688 preserve CNN's direct-fidelity advantage?

- **PSNRuv**: CNN=31.1925, GAN=29.1380, candidateB_factorial_speed_grad_expanded2688=33.4726. Δ(candidateB_factorial_speed_grad_expanded2688−CNN)=+2.2801 (▲ better), improved on 168/168 samples.
- **SSIM**: CNN=N/A, GAN=N/A, candidateB_factorial_speed_grad_expanded2688=N/A. Δ(candidateB_factorial_speed_grad_expanded2688−CNN)=N/A, improved on 0/0 samples.
- **Speed MAE**: CNN=0.6941, GAN=0.9026, candidateB_factorial_speed_grad_expanded2688=0.5179. Δ(candidateB_factorial_speed_grad_expanded2688−CNN)=-0.1762 (▲ better), improved on 168/168 samples.
- **Speed RMSE**: CNN=1.1078, GAN=1.3775, candidateB_factorial_speed_grad_expanded2688=0.8813. Δ(candidateB_factorial_speed_grad_expanded2688−CNN)=-0.2265 (▲ better), improved on 168/168 samples.

### Q2. Did candidateB_factorial_speed_grad_expanded2688 improve scalar speed or WPD metrics?

- **WPD MAE**: CNN=231.6709, candidateB_factorial_speed_grad_expanded2688=166.0587. Δ=-65.6123 (▲ better), improved on 168/168 samples.
- **WPD W1**: CNN=45.2713, candidateB_factorial_speed_grad_expanded2688=23.6460. Δ=-21.6253 (▲ better), improved on 135/168 samples.
- **WPD bias abs**: CNN=35.3439, candidateB_factorial_speed_grad_expanded2688=15.6753. Δ=-19.6686 (▲ better), improved on 122/168 samples.

### Q3. Did candidateB_factorial_speed_grad_expanded2688 improve gradient metrics?

- **Gradient MAE**: CNN=0.3491, candidateB_factorial_speed_grad_expanded2688=0.3123. Δ=-0.0368 (▲ better), improved on 168/168 samples.
- **Gradient W1**: CNN=0.2329, candidateB_factorial_speed_grad_expanded2688=0.1959. Δ=-0.0371 (▲ better), improved on 153/168 samples.
- **Gradient kurtosis abs Δ**: CNN=3.7004, candidateB_factorial_speed_grad_expanded2688=2.9340. Δ=-0.7663 (▲ better), improved on 94/168 samples.

### Q4. Did candidateB_factorial_speed_grad_expanded2688 improve exceedance metrics, especially s>5?

- **Exceedance abs Δ s>5**: CNN=0.00424, candidateB_factorial_speed_grad_expanded2688=0.00501. Δ=+0.0008 (▼ worse), improved on 67/168 samples.
- **Exceedance abs Δ s>10**: CNN=0.00658, candidateB_factorial_speed_grad_expanded2688=0.00330. Δ=-0.0033 (▲ better), improved on 118/168 samples.
- **Exceedance abs Δ s>15**: CNN=0.00621, candidateB_factorial_speed_grad_expanded2688=0.00312. Δ=-0.0031 (▲ better), improved on 120/168 samples.
- **Exceedance abs Δ p90**: CNN=0.01025, candidateB_factorial_speed_grad_expanded2688=0.00423. Δ=-0.0060 (▲ better), improved on 139/168 samples.

### Q5. Did candidateB_factorial_speed_grad_expanded2688 move component-count behavior toward GT?

- **Component-count curve L1**: CNN=115.5278, candidateB_factorial_speed_grad_expanded2688=111.3294. Δ=-4.1984 (▲ better), improved on 120/168 samples.
- **Comp-count abs Δ t5**: CNN=64.3810, candidateB_factorial_speed_grad_expanded2688=58.8095. Δ=-5.5714 (▲ better), improved on 112/168 samples.
- **Comp-count abs Δ t10**: CNN=187.1667, candidateB_factorial_speed_grad_expanded2688=188.0000. Δ=+0.8333 (▼ worse), improved on 79/168 samples.
- **Comp-count abs Δ t15**: CNN=116.8095, candidateB_factorial_speed_grad_expanded2688=109.1250. Δ=-7.6845 (▲ better), improved on 114/168 samples.

### Q6. Did candidateB_factorial_speed_grad_expanded2688 improve PD or MT distances?

PD and MT distances for `candidateB_factorial_speed_grad_expanded2688` are not available in this evaluation (TTK was not re-run on these outputs). Pre-computed values for CNN and GAN:

- **PD distance**: CNN=27.4063, GAN=20.8641, candidateB_factorial_speed_grad_expanded2688=N/A (requires TTK run)
- **MT distance**: CNN=5.8678, GAN=8.3481, candidateB_factorial_speed_grad_expanded2688=N/A (requires TTK run)

To compute PD/MT for `candidateB_factorial_speed_grad_expanded2688`: convert outputs to VTI via `scripts/convert_phire_to_vti.py`, then run `scripts/compute_composite_tree_distance.py`.

### Q7. Did candidateB_factorial_speed_grad_expanded2688 change the adjacent clusters (samples 10–13, 90–93, 162–163)?

See `adjacent_cluster_table_candidateB_factorial_speed_grad_expanded2688.csv` for full detail. Summary of PSNRuv, speed MAE, and gradient MAE changes:

| Sample | Δ PSNRuv | Δ Speed MAE | Δ Grad MAE | Δ Exceed s>5 |
|---|---|---|---|---|
| 10 | 2.455 | -0.1944 | -0.0371 | 0.00284 |
| 11 | 2.360 | -0.1914 | -0.0379 | 0.00165 |
| 12 | 2.375 | -0.1920 | -0.0420 | 0.00178 |
| 13 | 2.286 | -0.1847 | -0.0390 | 0.00018 |
| 90 | 2.097 | -0.1359 | -0.0294 | -0.00080 |
| 91 | 2.128 | -0.1404 | -0.0291 | 0.00336 |
| 92 | 2.128 | -0.1519 | -0.0317 | 0.00771 |
| 93 | 2.097 | -0.1623 | -0.0326 | 0.00424 |
| 162 | 3.586 | -0.1464 | -0.0259 | -0.00504 |
| 163 | 3.510 | -0.1468 | -0.0283 | -0.00184 |

## Pairwise summary: CNN vs candidateB_factorial_speed_grad_expanded2688

| Metric | Mean CNN | Mean candidateB_factorial_speed_grad_expanded2688 | Δ | N improved | N worsened |
|---|---|---|---|---|---|
| psnruv | 31.1925 | 33.4726 | 2.2801 | 168 | 0 |
| ssim | N/A | N/A | N/A | 0 | 0 |
| speed_mae | 0.6941 | 0.5179 | -0.1762 | 168 | 0 |
| speed_rmse | 1.1078 | 0.8813 | -0.2265 | 168 | 0 |
| wpd_bias_abs | 35.3439 | 15.6753 | -19.6686 | 122 | 46 |
| wpd_mae | 231.6709 | 166.0587 | -65.6123 | 168 | 0 |
| wpd_w1 | 45.2713 | 23.6460 | -21.6253 | 135 | 33 |
| psd_log_l2 | 0.8335 | 0.9694 | 0.1358 | 2 | 166 |
| psd_slope_abs_delta | 0.9150 | 1.2875 | 0.3725 | 0 | 168 |
| grad_mae | 0.3491 | 0.3123 | -0.0368 | 168 | 0 |
| grad_w1 | 0.2329 | 0.1959 | -0.0371 | 153 | 15 |
| grad_kurtosis_abs_delta | 3.7004 | 2.9340 | -0.7663 | 94 | 74 |
| exceed_abs_t5 | 0.0042 | 0.0050 | 0.0008 | 67 | 101 |
| exceed_abs_t10 | 0.0066 | 0.0033 | -0.0033 | 118 | 50 |
| exceed_abs_t15 | 0.0062 | 0.0031 | -0.0031 | 120 | 48 |
| exceed_abs_p90 | 0.0103 | 0.0042 | -0.0060 | 139 | 29 |
| comp_curve_l1 | 115.5278 | 111.3294 | -4.1984 | 120 | 48 |
| pd_distance | N/A | N/A | N/A | 0 | 0 |
| mt_distance | N/A | N/A | N/A | 0 | 0 |

## Output files

- `ttk_runs_fixed/topology_finetuning/candidateB_factorial_speed_grad_expanded2688_eval/all_sample_metrics_candidateB_factorial_speed_grad_expanded2688.csv` — per-sample metrics for all methods
- `ttk_runs_fixed/topology_finetuning/candidateB_factorial_speed_grad_expanded2688_eval/pairwise_cnn_vs_candidateB_factorial_speed_grad_expanded2688.csv` — per-metric delta table
- `ttk_runs_fixed/topology_finetuning/candidateB_factorial_speed_grad_expanded2688_eval/winner_counts_candidateB_factorial_speed_grad_expanded2688.csv` — win counts per method
- `ttk_runs_fixed/topology_finetuning/candidateB_factorial_speed_grad_expanded2688_eval/adjacent_cluster_table_candidateB_factorial_speed_grad_expanded2688.csv` — adjacent-cluster detail
- `docs/topology_finetuning_candidateB_factorial_speed_grad_expanded2688_eval.md` — this report

## Notes

- PSNRuv is computed on physical [u,v] arrays using per-sample dynamic data range (GT max − GT min).
- All speed-based metrics use scalar speed = sqrt(u² + v²) in physical units (m/s).
- Exceedance fractions use GT percentile thresholds (p90/p95/p99) computed per sample.
- Component counts use 8-connectivity on speed superlevel sets.
- PD/MT distances for CNN and GAN are loaded from pre-computed merged CSV; `candidateB_factorial_speed_grad_expanded2688` requires a fresh TTK run.
- `candidateB_factorial_speed_grad_expanded2688` was fine-tuned from the pretrained CNN checkpoint. Results may change with different lambda settings or more epochs.
