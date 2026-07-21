# candidateE2 fine-tuning evaluation

**Generated:** 2026-05-20

**Candidate:** `candidateE2`

**Samples evaluated:** 168

**Methods compared:** bicubic, cnn, gan, candidateE2

> **SSIM note:** SSIM was not computed because skimage could not be imported in the current NumPy environment (`ValueError('numpy.dtype size changed, may indicate binary incompatibility. Expected 96 from C header, got 88 from PyObject')`). All SSIM values are NaN.

> Note: PD/MT distances for `candidateE2` are N/A — TTK has not been re-run on these outputs. Values for CNN/GAN are loaded from the pre-computed merged CSV.

## Summary metrics table

| Metric | Bicubic | Cnn | Gan | Candidatee2 |
|---|---|---|---|---|
| PSNRuv (dB) | 32.2202 | 31.1925 | 29.1380 | 31.3621 |
| SSIM on speed | N/A | N/A | N/A | N/A |
| Speed MAE (m/s) | 0.5963 | 0.6941 | 0.9026 | 0.6839 |
| Speed RMSE (m/s) | 1.0156 | 1.1078 | 1.3775 | 1.0835 |
| WPD MAE (m³/s³) | 193.5403 | 231.6709 | 310.7328 | 227.8378 |
| WPD Wasserstein-1 | 55.5103 | 45.2713 | 85.6191 | 54.7098 |
| WPD bias abs (m³/s³) | 41.2244 | 35.3439 | 78.7236 | 44.1480 |
| Gradient MAE | 0.3696 | 0.3491 | 0.3806 | 0.3540 |
| Gradient W1 | 0.3282 | 0.2329 | 0.0564 | 0.2589 |
| Gradient kurtosis abs Δ | 6.6163 | 3.7004 | 4.2010 | 3.8736 |
| PSD log-L2 | 1.3625 | 0.8335 | 0.5139 | 0.8734 |
| PSD slope abs Δ | 1.6684 | 0.9150 | 0.9482 | 1.0145 |
| Exceedance abs Δ s>5 | 0.0095 | 0.0042 | 0.0082 | 0.0052 |
| Exceedance abs Δ s>10 | 0.0085 | 0.0066 | 0.0243 | 0.0073 |
| Exceedance abs Δ s>15 | 0.0074 | 0.0062 | 0.0096 | 0.0083 |
| Exceedance abs Δ p90 | 0.0109 | 0.0103 | 0.0123 | 0.0125 |
| Component-count curve L1 | 173.1855 | 115.5278 | 124.6567 | 125.0208 |
| PD distance (TTK) | N/A | 27.4063 | 20.8641 | N/A |
| MT distance (TTK) | N/A | 5.8678 | 8.3481 | N/A |

## 7 Key evaluation questions

### Q1. Did candidateE2 preserve CNN's direct-fidelity advantage?

- **PSNRuv**: CNN=31.1925, GAN=29.1380, candidateE2=31.3621. Δ(candidateE2−CNN)=+0.1696 (▲ better), improved on 167/168 samples.
- **SSIM**: CNN=N/A, GAN=N/A, candidateE2=N/A. Δ(candidateE2−CNN)=N/A, improved on 0/0 samples.
- **Speed MAE**: CNN=0.6941, GAN=0.9026, candidateE2=0.6839. Δ(candidateE2−CNN)=-0.0102 (▲ better), improved on 161/168 samples.
- **Speed RMSE**: CNN=1.1078, GAN=1.3775, candidateE2=1.0835. Δ(candidateE2−CNN)=-0.0243 (▲ better), improved on 167/168 samples.

### Q2. Did candidateE2 improve scalar speed or WPD metrics?

- **WPD MAE**: CNN=231.6709, candidateE2=227.8378. Δ=-3.8331 (▲ better), improved on 155/168 samples.
- **WPD W1**: CNN=45.2713, candidateE2=54.7098. Δ=+9.4385 (▼ worse), improved on 75/168 samples.
- **WPD bias abs**: CNN=35.3439, candidateE2=44.1480. Δ=+8.8041 (▼ worse), improved on 90/168 samples.

### Q3. Did candidateE2 improve gradient metrics?

- **Gradient MAE**: CNN=0.3491, candidateE2=0.3540. Δ=+0.0050 (▼ worse), improved on 16/168 samples.
- **Gradient W1**: CNN=0.2329, candidateE2=0.2589. Δ=+0.0260 (▼ worse), improved on 0/168 samples.
- **Gradient kurtosis abs Δ**: CNN=3.7004, candidateE2=3.8736. Δ=+0.1732 (▼ worse), improved on 103/168 samples.

### Q4. Did candidateE2 improve exceedance metrics, especially s>5?

- **Exceedance abs Δ s>5**: CNN=0.00424, candidateE2=0.00518. Δ=+0.0009 (▼ worse), improved on 27/168 samples.
- **Exceedance abs Δ s>10**: CNN=0.00658, candidateE2=0.00730. Δ=+0.0007 (▼ worse), improved on 83/168 samples.
- **Exceedance abs Δ s>15**: CNN=0.00621, candidateE2=0.00831. Δ=+0.0021 (▼ worse), improved on 39/168 samples.
- **Exceedance abs Δ p90**: CNN=0.01025, candidateE2=0.01254. Δ=+0.0023 (▼ worse), improved on 86/168 samples.

### Q5. Did candidateE2 move component-count behavior toward GT?

- **Component-count curve L1**: CNN=115.5278, candidateE2=125.0208. Δ=+9.4931 (▼ worse), improved on 7/168 samples.
- **Comp-count abs Δ t5**: CNN=64.3810, candidateE2=65.0893. Δ=+0.7083 (▼ worse), improved on 61/168 samples.
- **Comp-count abs Δ t10**: CNN=187.1667, candidateE2=195.5774. Δ=+8.4107 (▼ worse), improved on 26/168 samples.
- **Comp-count abs Δ t15**: CNN=116.8095, candidateE2=130.2619. Δ=+13.4524 (▼ worse), improved on 16/168 samples.

### Q6. Did candidateE2 improve PD or MT distances?

PD and MT distances for `candidateE2` are not available in this evaluation (TTK was not re-run on these outputs). Pre-computed values for CNN and GAN:

- **PD distance**: CNN=27.4063, GAN=20.8641, candidateE2=N/A (requires TTK run)
- **MT distance**: CNN=5.8678, GAN=8.3481, candidateE2=N/A (requires TTK run)

To compute PD/MT for `candidateE2`: convert outputs to VTI via `scripts/convert_phire_to_vti.py`, then run `scripts/compute_composite_tree_distance.py`.

### Q7. Did candidateE2 change the adjacent clusters (samples 10–13, 90–93, 162–163)?

See `adjacent_cluster_table_candidateE2.csv` for full detail. Summary of PSNRuv, speed MAE, and gradient MAE changes:

| Sample | Δ PSNRuv | Δ Speed MAE | Δ Grad MAE | Δ Exceed s>5 |
|---|---|---|---|---|
| 10 | 0.188 | -0.0140 | 0.0009 | 0.00136 |
| 11 | 0.152 | -0.0111 | 0.0017 | 0.00129 |
| 12 | 0.140 | -0.0098 | 0.0024 | 0.00108 |
| 13 | 0.121 | -0.0075 | 0.0038 | 0.00104 |
| 90 | 0.071 | -0.0033 | 0.0019 | 0.00200 |
| 91 | 0.067 | -0.0040 | 0.0017 | 0.00211 |
| 92 | 0.078 | -0.0056 | 0.0016 | 0.00131 |
| 93 | 0.099 | -0.0084 | 0.0019 | 0.00164 |
| 162 | 0.329 | -0.0116 | -0.0032 | -0.00438 |
| 163 | 0.308 | -0.0114 | -0.0031 | -0.00359 |

## Pairwise summary: CNN vs candidateE2

| Metric | Mean CNN | Mean candidateE2 | Δ | N improved | N worsened |
|---|---|---|---|---|---|
| psnruv | 31.1925 | 31.3621 | 0.1696 | 167 | 1 |
| ssim | N/A | N/A | N/A | 0 | 0 |
| speed_mae | 0.6941 | 0.6839 | -0.0102 | 161 | 7 |
| speed_rmse | 1.1078 | 1.0835 | -0.0243 | 167 | 1 |
| wpd_bias_abs | 35.3439 | 44.1480 | 8.8041 | 90 | 78 |
| wpd_mae | 231.6709 | 227.8378 | -3.8331 | 155 | 13 |
| wpd_w1 | 45.2713 | 54.7098 | 9.4385 | 75 | 93 |
| psd_log_l2 | 0.8335 | 0.8734 | 0.0399 | 17 | 151 |
| psd_slope_abs_delta | 0.9150 | 1.0145 | 0.0995 | 39 | 129 |
| grad_mae | 0.3491 | 0.3540 | 0.0050 | 16 | 152 |
| grad_w1 | 0.2329 | 0.2589 | 0.0260 | 0 | 168 |
| grad_kurtosis_abs_delta | 3.7004 | 3.8736 | 0.1732 | 103 | 65 |
| exceed_abs_t5 | 0.0042 | 0.0052 | 0.0009 | 27 | 141 |
| exceed_abs_t10 | 0.0066 | 0.0073 | 0.0007 | 83 | 85 |
| exceed_abs_t15 | 0.0062 | 0.0083 | 0.0021 | 39 | 127 |
| exceed_abs_p90 | 0.0103 | 0.0125 | 0.0023 | 86 | 82 |
| comp_curve_l1 | 115.5278 | 125.0208 | 9.4931 | 7 | 160 |
| pd_distance | N/A | N/A | N/A | 0 | 0 |
| mt_distance | N/A | N/A | N/A | 0 | 0 |

## Output files

- `ttk_runs_fixed/topology_finetuning/candidateE2_eval/all_sample_metrics_candidateE2.csv` — per-sample metrics for all methods
- `ttk_runs_fixed/topology_finetuning/candidateE2_eval/pairwise_cnn_vs_candidateE2.csv` — per-metric delta table
- `ttk_runs_fixed/topology_finetuning/candidateE2_eval/winner_counts_candidateE2.csv` — win counts per method
- `ttk_runs_fixed/topology_finetuning/candidateE2_eval/adjacent_cluster_table_candidateE2.csv` — adjacent-cluster detail
- `docs/topology_finetuning_candidateE2_eval.md` — this report

## Notes

- PSNRuv is computed on physical [u,v] arrays using per-sample dynamic data range (GT max − GT min).
- All speed-based metrics use scalar speed = sqrt(u² + v²) in physical units (m/s).
- Exceedance fractions use GT percentile thresholds (p90/p95/p99) computed per sample.
- Component counts use 8-connectivity on speed superlevel sets.
- PD/MT distances for CNN and GAN are loaded from pre-computed merged CSV; `candidateE2` requires a fresh TTK run.
- `candidateE2` was fine-tuned from the pretrained CNN checkpoint. Results may change with different lambda settings or more epochs.
