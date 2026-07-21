# candidateE2_fixed_lowlambda_expanded1344 fine-tuning evaluation

**Generated:** 2026-07-08

**Candidate:** `candidateE2_fixed_lowlambda_expanded1344`

**Samples evaluated:** 168

**Methods compared:** bicubic, cnn, gan, candidateE2_fixed_lowlambda_expanded1344

> **SSIM note:** SSIM was not computed because skimage could not be imported in the current NumPy environment (`ValueError('numpy.dtype size changed, may indicate binary incompatibility. Expected 96 from C header, got 88 from PyObject')`). All SSIM values are NaN.

> Note: PD/MT distances for `candidateE2_fixed_lowlambda_expanded1344` are N/A — TTK has not been re-run on these outputs. Values for CNN/GAN are loaded from the pre-computed merged CSV.

## Summary metrics table

| Metric | Bicubic | Cnn | Gan | Candidatee2_fixed_lowlambda_expanded1344 |
|---|---|---|---|---|
| PSNRuv (dB) | 32.2202 | 31.1925 | 29.1380 | 31.1643 |
| SSIM on speed | N/A | N/A | N/A | N/A |
| Speed MAE (m/s) | 0.5963 | 0.6941 | 0.9026 | 0.6980 |
| Speed RMSE (m/s) | 1.0156 | 1.1078 | 1.3775 | 1.1142 |
| WPD MAE (m³/s³) | 193.5403 | 231.6709 | 310.7328 | 233.6817 |
| WPD Wasserstein-1 | 55.5103 | 45.2713 | 85.6191 | 26.4206 |
| WPD bias abs (m³/s³) | 41.2244 | 35.3439 | 78.7236 | 8.4876 |
| Gradient MAE | 0.3696 | 0.3491 | 0.3806 | 0.3441 |
| Gradient W1 | 0.3282 | 0.2329 | 0.0564 | 0.2071 |
| Gradient kurtosis abs Δ | 6.6163 | 3.7004 | 4.2010 | 3.3142 |
| PSD log-L2 | 1.3625 | 0.8335 | 0.5139 | 0.7878 |
| PSD slope abs Δ | 1.6684 | 0.9150 | 0.9482 | 0.8790 |
| Exceedance abs Δ s>5 | 0.0095 | 0.0042 | 0.0082 | 0.0043 |
| Exceedance abs Δ s>10 | 0.0085 | 0.0066 | 0.0243 | 0.0046 |
| Exceedance abs Δ s>15 | 0.0074 | 0.0062 | 0.0096 | 0.0032 |
| Exceedance abs Δ p90 | 0.0109 | 0.0103 | 0.0123 | 0.0041 |
| Component-count curve L1 | 173.1855 | 115.5278 | 124.6567 | 107.1022 |
| PD distance (TTK) | N/A | 27.4063 | 20.8641 | N/A |
| MT distance (TTK) | N/A | 5.8678 | 8.3481 | N/A |

## 7 Key evaluation questions

### Q1. Did candidateE2_fixed_lowlambda_expanded1344 preserve CNN's direct-fidelity advantage?

- **PSNRuv**: CNN=31.1925, GAN=29.1380, candidateE2_fixed_lowlambda_expanded1344=31.1643. Δ(candidateE2_fixed_lowlambda_expanded1344−CNN)=-0.0282 (▼ worse), improved on 32/168 samples.
- **SSIM**: CNN=N/A, GAN=N/A, candidateE2_fixed_lowlambda_expanded1344=N/A. Δ(candidateE2_fixed_lowlambda_expanded1344−CNN)=N/A, improved on 0/0 samples.
- **Speed MAE**: CNN=0.6941, GAN=0.9026, candidateE2_fixed_lowlambda_expanded1344=0.6980. Δ(candidateE2_fixed_lowlambda_expanded1344−CNN)=+0.0040 (▼ worse), improved on 38/168 samples.
- **Speed RMSE**: CNN=1.1078, GAN=1.3775, candidateE2_fixed_lowlambda_expanded1344=1.1142. Δ(candidateE2_fixed_lowlambda_expanded1344−CNN)=+0.0064 (▼ worse), improved on 31/168 samples.

### Q2. Did candidateE2_fixed_lowlambda_expanded1344 improve scalar speed or WPD metrics?

- **WPD MAE**: CNN=231.6709, candidateE2_fixed_lowlambda_expanded1344=233.6817. Δ=+2.0108 (▼ worse), improved on 29/168 samples.
- **WPD W1**: CNN=45.2713, candidateE2_fixed_lowlambda_expanded1344=26.4206. Δ=-18.8507 (▲ better), improved on 153/168 samples.
- **WPD bias abs**: CNN=35.3439, candidateE2_fixed_lowlambda_expanded1344=8.4876. Δ=-26.8563 (▲ better), improved on 154/168 samples.

### Q3. Did candidateE2_fixed_lowlambda_expanded1344 improve gradient metrics?

- **Gradient MAE**: CNN=0.3491, candidateE2_fixed_lowlambda_expanded1344=0.3441. Δ=-0.0050 (▲ better), improved on 162/168 samples.
- **Gradient W1**: CNN=0.2329, candidateE2_fixed_lowlambda_expanded1344=0.2071. Δ=-0.0258 (▲ better), improved on 149/168 samples.
- **Gradient kurtosis abs Δ**: CNN=3.7004, candidateE2_fixed_lowlambda_expanded1344=3.3142. Δ=-0.3861 (▲ better), improved on 111/168 samples.

### Q4. Did candidateE2_fixed_lowlambda_expanded1344 improve exceedance metrics, especially s>5?

- **Exceedance abs Δ s>5**: CNN=0.00424, candidateE2_fixed_lowlambda_expanded1344=0.00428. Δ=+0.0000 (▼ worse), improved on 62/168 samples.
- **Exceedance abs Δ s>10**: CNN=0.00658, candidateE2_fixed_lowlambda_expanded1344=0.00462. Δ=-0.0020 (▲ better), improved on 106/168 samples.
- **Exceedance abs Δ s>15**: CNN=0.00621, candidateE2_fixed_lowlambda_expanded1344=0.00323. Δ=-0.0030 (▲ better), improved on 138/168 samples.
- **Exceedance abs Δ p90**: CNN=0.01025, candidateE2_fixed_lowlambda_expanded1344=0.00412. Δ=-0.0061 (▲ better), improved on 151/168 samples.

### Q5. Did candidateE2_fixed_lowlambda_expanded1344 move component-count behavior toward GT?

- **Component-count curve L1**: CNN=115.5278, candidateE2_fixed_lowlambda_expanded1344=107.1022. Δ=-8.4256 (▲ better), improved on 140/168 samples.
- **Comp-count abs Δ t5**: CNN=64.3810, candidateE2_fixed_lowlambda_expanded1344=64.4167. Δ=+0.0357 (▼ worse), improved on 72/168 samples.
- **Comp-count abs Δ t10**: CNN=187.1667, candidateE2_fixed_lowlambda_expanded1344=182.6667. Δ=-4.5000 (▲ better), improved on 92/168 samples.
- **Comp-count abs Δ t15**: CNN=116.8095, candidateE2_fixed_lowlambda_expanded1344=103.8929. Δ=-12.9167 (▲ better), improved on 139/168 samples.

### Q6. Did candidateE2_fixed_lowlambda_expanded1344 improve PD or MT distances?

PD and MT distances for `candidateE2_fixed_lowlambda_expanded1344` are not available in this evaluation (TTK was not re-run on these outputs). Pre-computed values for CNN and GAN:

- **PD distance**: CNN=27.4063, GAN=20.8641, candidateE2_fixed_lowlambda_expanded1344=N/A (requires TTK run)
- **MT distance**: CNN=5.8678, GAN=8.3481, candidateE2_fixed_lowlambda_expanded1344=N/A (requires TTK run)

To compute PD/MT for `candidateE2_fixed_lowlambda_expanded1344`: convert outputs to VTI via `scripts/convert_phire_to_vti.py`, then run `scripts/compute_composite_tree_distance.py`.

### Q7. Did candidateE2_fixed_lowlambda_expanded1344 change the adjacent clusters (samples 10–13, 90–93, 162–163)?

See `adjacent_cluster_table_candidateE2_fixed_lowlambda_expanded1344.csv` for full detail. Summary of PSNRuv, speed MAE, and gradient MAE changes:

| Sample | Δ PSNRuv | Δ Speed MAE | Δ Grad MAE | Δ Exceed s>5 |
|---|---|---|---|---|
| 10 | 0.143 | -0.0114 | -0.0056 | 0.00031 |
| 11 | 0.117 | -0.0093 | -0.0050 | 0.00002 |
| 12 | 0.088 | -0.0068 | -0.0050 | 0.00044 |
| 13 | 0.042 | -0.0016 | -0.0042 | -0.00023 |
| 90 | -0.041 | 0.0029 | -0.0030 | 0.00460 |
| 91 | -0.042 | 0.0024 | -0.0030 | 0.00467 |
| 92 | -0.037 | 0.0009 | -0.0032 | 0.00309 |
| 93 | -0.021 | -0.0026 | -0.0040 | 0.00226 |
| 162 | 0.329 | -0.0134 | -0.0046 | -0.00053 |
| 163 | 0.310 | -0.0138 | -0.0045 | 0.00068 |

## Pairwise summary: CNN vs candidateE2_fixed_lowlambda_expanded1344

| Metric | Mean CNN | Mean candidateE2_fixed_lowlambda_expanded1344 | Δ | N improved | N worsened |
|---|---|---|---|---|---|
| psnruv | 31.1925 | 31.1643 | -0.0282 | 32 | 136 |
| ssim | N/A | N/A | N/A | 0 | 0 |
| speed_mae | 0.6941 | 0.6980 | 0.0040 | 38 | 130 |
| speed_rmse | 1.1078 | 1.1142 | 0.0064 | 31 | 137 |
| wpd_bias_abs | 35.3439 | 8.4876 | -26.8563 | 154 | 14 |
| wpd_mae | 231.6709 | 233.6817 | 2.0108 | 29 | 139 |
| wpd_w1 | 45.2713 | 26.4206 | -18.8507 | 153 | 15 |
| psd_log_l2 | 0.8335 | 0.7878 | -0.0458 | 140 | 28 |
| psd_slope_abs_delta | 0.9150 | 0.8790 | -0.0360 | 109 | 59 |
| grad_mae | 0.3491 | 0.3441 | -0.0050 | 162 | 6 |
| grad_w1 | 0.2329 | 0.2071 | -0.0258 | 149 | 19 |
| grad_kurtosis_abs_delta | 3.7004 | 3.3142 | -0.3861 | 111 | 57 |
| exceed_abs_t5 | 0.0042 | 0.0043 | 0.0000 | 62 | 106 |
| exceed_abs_t10 | 0.0066 | 0.0046 | -0.0020 | 106 | 62 |
| exceed_abs_t15 | 0.0062 | 0.0032 | -0.0030 | 138 | 30 |
| exceed_abs_p90 | 0.0103 | 0.0041 | -0.0061 | 151 | 17 |
| comp_curve_l1 | 115.5278 | 107.1022 | -8.4256 | 140 | 26 |
| pd_distance | N/A | N/A | N/A | 0 | 0 |
| mt_distance | N/A | N/A | N/A | 0 | 0 |

## Output files

- `ttk_runs_fixed/topology_finetuning/candidateE2_fixed_lowlambda_expanded1344_eval/all_sample_metrics_candidateE2_fixed_lowlambda_expanded1344.csv` — per-sample metrics for all methods
- `ttk_runs_fixed/topology_finetuning/candidateE2_fixed_lowlambda_expanded1344_eval/pairwise_cnn_vs_candidateE2_fixed_lowlambda_expanded1344.csv` — per-metric delta table
- `ttk_runs_fixed/topology_finetuning/candidateE2_fixed_lowlambda_expanded1344_eval/winner_counts_candidateE2_fixed_lowlambda_expanded1344.csv` — win counts per method
- `ttk_runs_fixed/topology_finetuning/candidateE2_fixed_lowlambda_expanded1344_eval/adjacent_cluster_table_candidateE2_fixed_lowlambda_expanded1344.csv` — adjacent-cluster detail
- `docs/topology_finetuning_candidateE2_fixed_lowlambda_expanded1344_eval.md` — this report

## Notes

- PSNRuv is computed on physical [u,v] arrays using per-sample dynamic data range (GT max − GT min).
- All speed-based metrics use scalar speed = sqrt(u² + v²) in physical units (m/s).
- Exceedance fractions use GT percentile thresholds (p90/p95/p99) computed per sample.
- Component counts use 8-connectivity on speed superlevel sets.
- PD/MT distances for CNN and GAN are loaded from pre-computed merged CSV; `candidateE2_fixed_lowlambda_expanded1344` requires a fresh TTK run.
- `candidateE2_fixed_lowlambda_expanded1344` was fine-tuned from the pretrained CNN checkpoint. Results may change with different lambda settings or more epochs.
