# candidateD fine-tuning evaluation

**Generated:** 2026-05-18

**Candidate:** `candidateD`

**Samples evaluated:** 168

**Methods compared:** bicubic, cnn, gan, candidateD

> **SSIM note:** SSIM was not computed because skimage could not be imported in the current NumPy environment (`ValueError('numpy.dtype size changed, may indicate binary incompatibility. Expected 96 from C header, got 88 from PyObject')`). All SSIM values are NaN.

> Note: PD/MT distances for `candidateD` are N/A — TTK has not been re-run on these outputs. Values for CNN/GAN are loaded from the pre-computed merged CSV.

## Summary metrics table

| Metric | Bicubic | Cnn | Gan | Candidated |
|---|---|---|---|---|
| PSNRuv (dB) | 32.2202 | 31.1925 | 29.1380 | 31.3402 |
| SSIM on speed | N/A | N/A | N/A | N/A |
| Speed MAE (m/s) | 0.5963 | 0.6941 | 0.9026 | 0.6839 |
| Speed RMSE (m/s) | 1.0156 | 1.1078 | 1.3775 | 1.0860 |
| WPD MAE (m³/s³) | 193.5403 | 231.6709 | 310.7328 | 227.6354 |
| WPD Wasserstein-1 | 55.5103 | 45.2713 | 85.6191 | 43.5633 |
| WPD bias abs (m³/s³) | 41.2244 | 35.3439 | 78.7236 | 30.2191 |
| Gradient MAE | 0.3696 | 0.3491 | 0.3806 | 0.3517 |
| Gradient W1 | 0.3282 | 0.2329 | 0.0564 | 0.2509 |
| Gradient kurtosis abs Δ | 6.6163 | 3.7004 | 4.2010 | 3.4580 |
| PSD log-L2 | 1.3625 | 0.8335 | 0.5139 | 0.8592 |
| PSD slope abs Δ | 1.6684 | 0.9150 | 0.9482 | 0.9693 |
| Exceedance abs Δ s>5 | 0.0095 | 0.0042 | 0.0082 | 0.0053 |
| Exceedance abs Δ s>10 | 0.0085 | 0.0066 | 0.0243 | 0.0062 |
| Exceedance abs Δ s>15 | 0.0074 | 0.0062 | 0.0096 | 0.0065 |
| Exceedance abs Δ p90 | 0.0109 | 0.0103 | 0.0123 | 0.0093 |
| Component-count curve L1 | 173.1855 | 115.5278 | 124.6567 | 121.8849 |
| PD distance (TTK) | N/A | 27.4063 | 20.8641 | N/A |
| MT distance (TTK) | N/A | 5.8678 | 8.3481 | N/A |

## 7 Key evaluation questions

### Q1. Did candidateD preserve CNN's direct-fidelity advantage?

- **PSNRuv**: CNN=31.1925, GAN=29.1380, candidateD=31.3402. Δ(candidateD−CNN)=+0.1477 (▲ better), improved on 167/168 samples.
- **SSIM**: CNN=N/A, GAN=N/A, candidateD=N/A. Δ(candidateD−CNN)=N/A, improved on 0/0 samples.
- **Speed MAE**: CNN=0.6941, GAN=0.9026, candidateD=0.6839. Δ(candidateD−CNN)=-0.0102 (▲ better), improved on 163/168 samples.
- **Speed RMSE**: CNN=1.1078, GAN=1.3775, candidateD=1.0860. Δ(candidateD−CNN)=-0.0217 (▲ better), improved on 167/168 samples.

### Q2. Did candidateD improve scalar speed or WPD metrics?

- **WPD MAE**: CNN=231.6709, candidateD=227.6354. Δ=-4.0355 (▲ better), improved on 161/168 samples.
- **WPD W1**: CNN=45.2713, candidateD=43.5633. Δ=-1.7080 (▲ better), improved on 105/168 samples.
- **WPD bias abs**: CNN=35.3439, candidateD=30.2191. Δ=-5.1248 (▲ better), improved on 108/168 samples.

### Q3. Did candidateD improve gradient metrics?

- **Gradient MAE**: CNN=0.3491, candidateD=0.3517. Δ=+0.0027 (▼ worse), improved on 20/168 samples.
- **Gradient W1**: CNN=0.2329, candidateD=0.2509. Δ=+0.0180 (▼ worse), improved on 0/168 samples.
- **Gradient kurtosis abs Δ**: CNN=3.7004, candidateD=3.4580. Δ=-0.2424 (▲ better), improved on 107/168 samples.

### Q4. Did candidateD improve exceedance metrics, especially s>5?

- **Exceedance abs Δ s>5**: CNN=0.00424, candidateD=0.00531. Δ=+0.0011 (▼ worse), improved on 26/168 samples.
- **Exceedance abs Δ s>10**: CNN=0.00658, candidateD=0.00622. Δ=-0.0004 (▲ better), improved on 94/168 samples.
- **Exceedance abs Δ s>15**: CNN=0.00621, candidateD=0.00651. Δ=+0.0003 (▼ worse), improved on 77/168 samples.
- **Exceedance abs Δ p90**: CNN=0.01025, candidateD=0.00929. Δ=-0.0010 (▲ better), improved on 107/168 samples.

### Q5. Did candidateD move component-count behavior toward GT?

- **Component-count curve L1**: CNN=115.5278, candidateD=121.8849. Δ=+6.3571 (▼ worse), improved on 21/168 samples.
- **Comp-count abs Δ t5**: CNN=64.3810, candidateD=64.8214. Δ=+0.4405 (▼ worse), improved on 70/168 samples.
- **Comp-count abs Δ t10**: CNN=187.1667, candidateD=194.4405. Δ=+7.2738 (▼ worse), improved on 40/168 samples.
- **Comp-count abs Δ t15**: CNN=116.8095, candidateD=125.5714. Δ=+8.7619 (▼ worse), improved on 39/168 samples.

### Q6. Did candidateD improve PD or MT distances?

PD and MT distances for `candidateD` are not available in this evaluation (TTK was not re-run on these outputs). Pre-computed values for CNN and GAN:

- **PD distance**: CNN=27.4063, GAN=20.8641, candidateD=N/A (requires TTK run)
- **MT distance**: CNN=5.8678, GAN=8.3481, candidateD=N/A (requires TTK run)

To compute PD/MT for `candidateD`: convert outputs to VTI via `scripts/convert_phire_to_vti.py`, then run `scripts/compute_composite_tree_distance.py`.

### Q7. Did candidateD change the adjacent clusters (samples 10–13, 90–93, 162–163)?

See `adjacent_cluster_table_candidateD.csv` for full detail. Summary of PSNRuv, speed MAE, and gradient MAE changes:

| Sample | Δ PSNRuv | Δ Speed MAE | Δ Grad MAE | Δ Exceed s>5 |
|---|---|---|---|---|
| 10 | 0.155 | -0.0114 | 0.0000 | 0.00080 |
| 11 | 0.125 | -0.0092 | 0.0006 | 0.00096 |
| 12 | 0.115 | -0.0082 | 0.0011 | 0.00096 |
| 13 | 0.100 | -0.0064 | 0.0022 | 0.00084 |
| 90 | 0.054 | -0.0031 | 0.0008 | 0.00324 |
| 91 | 0.051 | -0.0036 | 0.0007 | 0.00360 |
| 92 | 0.062 | -0.0057 | 0.0006 | 0.00227 |
| 93 | 0.081 | -0.0093 | 0.0008 | 0.00218 |
| 162 | 0.274 | -0.0098 | -0.0027 | -0.00360 |
| 163 | 0.256 | -0.0095 | -0.0026 | -0.00340 |

## Pairwise summary: CNN vs candidateD

| Metric | Mean CNN | Mean candidateD | Δ | N improved | N worsened |
|---|---|---|---|---|---|
| psnruv | 31.1925 | 31.3402 | 0.1477 | 167 | 1 |
| ssim | N/A | N/A | N/A | 0 | 0 |
| speed_mae | 0.6941 | 0.6839 | -0.0102 | 163 | 5 |
| speed_rmse | 1.1078 | 1.0860 | -0.0217 | 167 | 1 |
| wpd_bias_abs | 35.3439 | 30.2191 | -5.1248 | 108 | 60 |
| wpd_mae | 231.6709 | 227.6354 | -4.0355 | 161 | 7 |
| wpd_w1 | 45.2713 | 43.5633 | -1.7080 | 105 | 63 |
| psd_log_l2 | 0.8335 | 0.8592 | 0.0257 | 44 | 124 |
| psd_slope_abs_delta | 0.9150 | 0.9693 | 0.0542 | 57 | 111 |
| grad_mae | 0.3491 | 0.3517 | 0.0027 | 20 | 148 |
| grad_w1 | 0.2329 | 0.2509 | 0.0180 | 0 | 168 |
| grad_kurtosis_abs_delta | 3.7004 | 3.4580 | -0.2424 | 107 | 61 |
| exceed_abs_t5 | 0.0042 | 0.0053 | 0.0011 | 26 | 142 |
| exceed_abs_t10 | 0.0066 | 0.0062 | -0.0004 | 94 | 74 |
| exceed_abs_t15 | 0.0062 | 0.0065 | 0.0003 | 77 | 90 |
| exceed_abs_p90 | 0.0103 | 0.0093 | -0.0010 | 107 | 61 |
| comp_curve_l1 | 115.5278 | 121.8849 | 6.3571 | 21 | 146 |
| pd_distance | N/A | N/A | N/A | 0 | 0 |
| mt_distance | N/A | N/A | N/A | 0 | 0 |

## Output files

- `ttk_runs_fixed/topology_finetuning/candidateD_eval/all_sample_metrics_candidateD.csv` — per-sample metrics for all methods
- `ttk_runs_fixed/topology_finetuning/candidateD_eval/pairwise_cnn_vs_candidateD.csv` — per-metric delta table
- `ttk_runs_fixed/topology_finetuning/candidateD_eval/winner_counts_candidateD.csv` — win counts per method
- `ttk_runs_fixed/topology_finetuning/candidateD_eval/adjacent_cluster_table_candidateD.csv` — adjacent-cluster detail
- `docs/topology_finetuning_candidateD_eval.md` — this report

## Notes

- PSNRuv is computed on physical [u,v] arrays using per-sample dynamic data range (GT max − GT min).
- All speed-based metrics use scalar speed = sqrt(u² + v²) in physical units (m/s).
- Exceedance fractions use GT percentile thresholds (p90/p95/p99) computed per sample.
- Component counts use 8-connectivity on speed superlevel sets.
- PD/MT distances for CNN and GAN are loaded from pre-computed merged CSV; `candidateD` requires a fresh TTK run.
- `candidateD` was fine-tuned from the pretrained CNN checkpoint. Results may change with different lambda settings or more epochs.
