# candidateUV fine-tuning evaluation

**Generated:** 2026-05-22

**Candidate:** `candidateUV`

**Samples evaluated:** 168

**Methods compared:** bicubic, cnn, gan, candidateUV

> **SSIM note:** SSIM was not computed because skimage could not be imported in the current NumPy environment (`ValueError('numpy.dtype size changed, may indicate binary incompatibility. Expected 96 from C header, got 88 from PyObject')`). All SSIM values are NaN.

> Note: PD/MT distances for `candidateUV` are N/A — TTK has not been re-run on these outputs. Values for CNN/GAN are loaded from the pre-computed merged CSV.

## Summary metrics table

| Metric | Bicubic | Cnn | Gan | Candidateuv |
|---|---|---|---|---|
| PSNRuv (dB) | 32.2202 | 31.1925 | 29.1380 | 32.6692 |
| SSIM on speed | N/A | N/A | N/A | N/A |
| Speed MAE (m/s) | 0.5963 | 0.6941 | 0.9026 | 0.5775 |
| Speed RMSE (m/s) | 1.0156 | 1.1078 | 1.3775 | 0.9501 |
| WPD MAE (m³/s³) | 193.5403 | 231.6709 | 310.7328 | 187.9271 |
| WPD Wasserstein-1 | 55.5103 | 45.2713 | 85.6191 | 45.1163 |
| WPD bias abs (m³/s³) | 41.2244 | 35.3439 | 78.7236 | 33.8267 |
| Gradient MAE | 0.3696 | 0.3491 | 0.3806 | 0.3467 |
| Gradient W1 | 0.3282 | 0.2329 | 0.0564 | 0.2770 |
| Gradient kurtosis abs Δ | 6.6163 | 3.7004 | 4.2010 | 3.0806 |
| PSD log-L2 | 1.3625 | 0.8335 | 0.5139 | 1.1135 |
| PSD slope abs Δ | 1.6684 | 0.9150 | 0.9482 | 1.3547 |
| Exceedance abs Δ s>5 | 0.0095 | 0.0042 | 0.0082 | 0.0064 |
| Exceedance abs Δ s>10 | 0.0085 | 0.0066 | 0.0243 | 0.0059 |
| Exceedance abs Δ s>15 | 0.0074 | 0.0062 | 0.0096 | 0.0065 |
| Exceedance abs Δ p90 | 0.0109 | 0.0103 | 0.0123 | 0.0088 |
| Component-count curve L1 | 173.1855 | 115.5278 | 124.6567 | 146.3879 |
| PD distance (TTK) | N/A | 27.4063 | 20.8641 | N/A |
| MT distance (TTK) | N/A | 5.8678 | 8.3481 | N/A |

## 7 Key evaluation questions

### Q1. Did candidateUV preserve CNN's direct-fidelity advantage?

- **PSNRuv**: CNN=31.1925, GAN=29.1380, candidateUV=32.6692. Δ(candidateUV−CNN)=+1.4767 (▲ better), improved on 168/168 samples.
- **SSIM**: CNN=N/A, GAN=N/A, candidateUV=N/A. Δ(candidateUV−CNN)=N/A, improved on 0/0 samples.
- **Speed MAE**: CNN=0.6941, GAN=0.9026, candidateUV=0.5775. Δ(candidateUV−CNN)=-0.1166 (▲ better), improved on 168/168 samples.
- **Speed RMSE**: CNN=1.1078, GAN=1.3775, candidateUV=0.9501. Δ(candidateUV−CNN)=-0.1577 (▲ better), improved on 168/168 samples.

### Q2. Did candidateUV improve scalar speed or WPD metrics?

- **WPD MAE**: CNN=231.6709, candidateUV=187.9271. Δ=-43.7438 (▲ better), improved on 168/168 samples.
- **WPD W1**: CNN=45.2713, candidateUV=45.1163. Δ=-0.1550 (▲ better), improved on 96/168 samples.
- **WPD bias abs**: CNN=35.3439, candidateUV=33.8267. Δ=-1.5172 (▲ better), improved on 97/168 samples.

### Q3. Did candidateUV improve gradient metrics?

- **Gradient MAE**: CNN=0.3491, candidateUV=0.3467. Δ=-0.0023 (▲ better), improved on 106/168 samples.
- **Gradient W1**: CNN=0.2329, candidateUV=0.2770. Δ=+0.0441 (▼ worse), improved on 2/168 samples.
- **Gradient kurtosis abs Δ**: CNN=3.7004, candidateUV=3.0806. Δ=-0.6198 (▲ better), improved on 108/168 samples.

### Q4. Did candidateUV improve exceedance metrics, especially s>5?

- **Exceedance abs Δ s>5**: CNN=0.00424, candidateUV=0.00635. Δ=+0.0021 (▼ worse), improved on 29/168 samples.
- **Exceedance abs Δ s>10**: CNN=0.00658, candidateUV=0.00593. Δ=-0.0007 (▲ better), improved on 89/168 samples.
- **Exceedance abs Δ s>15**: CNN=0.00621, candidateUV=0.00651. Δ=+0.0003 (▼ worse), improved on 71/168 samples.
- **Exceedance abs Δ p90**: CNN=0.01025, candidateUV=0.00882. Δ=-0.0014 (▲ better), improved on 103/168 samples.

### Q5. Did candidateUV move component-count behavior toward GT?

- **Component-count curve L1**: CNN=115.5278, candidateUV=146.3879. Δ=+30.8601 (▼ worse), improved on 1/168 samples.
- **Comp-count abs Δ t5**: CNN=64.3810, candidateUV=72.9940. Δ=+8.6131 (▼ worse), improved on 19/168 samples.
- **Comp-count abs Δ t10**: CNN=187.1667, candidateUV=233.9464. Δ=+46.7798 (▼ worse), improved on 0/168 samples.
- **Comp-count abs Δ t15**: CNN=116.8095, candidateUV=148.3452. Δ=+31.5357 (▼ worse), improved on 5/168 samples.

### Q6. Did candidateUV improve PD or MT distances?

PD and MT distances for `candidateUV` are not available in this evaluation (TTK was not re-run on these outputs). Pre-computed values for CNN and GAN:

- **PD distance**: CNN=27.4063, GAN=20.8641, candidateUV=N/A (requires TTK run)
- **MT distance**: CNN=5.8678, GAN=8.3481, candidateUV=N/A (requires TTK run)

To compute PD/MT for `candidateUV`: convert outputs to VTI via `scripts/convert_phire_to_vti.py`, then run `scripts/compute_composite_tree_distance.py`.

### Q7. Did candidateUV change the adjacent clusters (samples 10–13, 90–93, 162–163)?

See `adjacent_cluster_table_candidateUV.csv` for full detail. Summary of PSNRuv, speed MAE, and gradient MAE changes:

| Sample | Δ PSNRuv | Δ Speed MAE | Δ Grad MAE | Δ Exceed s>5 |
|---|---|---|---|---|
| 10 | 1.611 | -0.1335 | -0.0016 | 0.00278 |
| 11 | 1.502 | -0.1269 | -0.0018 | 0.00122 |
| 12 | 1.501 | -0.1237 | -0.0043 | -0.00020 |
| 13 | 1.428 | -0.1144 | -0.0017 | -0.00207 |
| 90 | 1.309 | -0.0865 | -0.0039 | 0.00187 |
| 91 | 1.384 | -0.0934 | -0.0034 | 0.00525 |
| 92 | 1.436 | -0.1053 | -0.0034 | 0.01022 |
| 93 | 1.426 | -0.1138 | 0.0007 | 0.00882 |
| 162 | 2.180 | -0.0827 | -0.0126 | -0.00198 |
| 163 | 2.163 | -0.0832 | -0.0142 | 0.00479 |

## Pairwise summary: CNN vs candidateUV

| Metric | Mean CNN | Mean candidateUV | Δ | N improved | N worsened |
|---|---|---|---|---|---|
| psnruv | 31.1925 | 32.6692 | 1.4767 | 168 | 0 |
| ssim | N/A | N/A | N/A | 0 | 0 |
| speed_mae | 0.6941 | 0.5775 | -0.1166 | 168 | 0 |
| speed_rmse | 1.1078 | 0.9501 | -0.1577 | 168 | 0 |
| wpd_bias_abs | 35.3439 | 33.8267 | -1.5172 | 97 | 71 |
| wpd_mae | 231.6709 | 187.9271 | -43.7438 | 168 | 0 |
| wpd_w1 | 45.2713 | 45.1163 | -0.1550 | 96 | 72 |
| psd_log_l2 | 0.8335 | 1.1135 | 0.2800 | 0 | 168 |
| psd_slope_abs_delta | 0.9150 | 1.3547 | 0.4397 | 0 | 168 |
| grad_mae | 0.3491 | 0.3467 | -0.0023 | 106 | 62 |
| grad_w1 | 0.2329 | 0.2770 | 0.0441 | 2 | 166 |
| grad_kurtosis_abs_delta | 3.7004 | 3.0806 | -0.6198 | 108 | 60 |
| exceed_abs_t5 | 0.0042 | 0.0064 | 0.0021 | 29 | 139 |
| exceed_abs_t10 | 0.0066 | 0.0059 | -0.0007 | 89 | 79 |
| exceed_abs_t15 | 0.0062 | 0.0065 | 0.0003 | 71 | 96 |
| exceed_abs_p90 | 0.0103 | 0.0088 | -0.0014 | 103 | 65 |
| comp_curve_l1 | 115.5278 | 146.3879 | 30.8601 | 1 | 167 |
| pd_distance | N/A | N/A | N/A | 0 | 0 |
| mt_distance | N/A | N/A | N/A | 0 | 0 |

## Output files

- `ttk_runs_fixed/topology_finetuning/candidateUV_eval/all_sample_metrics_candidateUV.csv` — per-sample metrics for all methods
- `ttk_runs_fixed/topology_finetuning/candidateUV_eval/pairwise_cnn_vs_candidateUV.csv` — per-metric delta table
- `ttk_runs_fixed/topology_finetuning/candidateUV_eval/winner_counts_candidateUV.csv` — win counts per method
- `ttk_runs_fixed/topology_finetuning/candidateUV_eval/adjacent_cluster_table_candidateUV.csv` — adjacent-cluster detail
- `docs/topology_finetuning_candidateUV_eval.md` — this report

## Notes

- PSNRuv is computed on physical [u,v] arrays using per-sample dynamic data range (GT max − GT min).
- All speed-based metrics use scalar speed = sqrt(u² + v²) in physical units (m/s).
- Exceedance fractions use GT percentile thresholds (p90/p95/p99) computed per sample.
- Component counts use 8-connectivity on speed superlevel sets.
- PD/MT distances for CNN and GAN are loaded from pre-computed merged CSV; `candidateUV` requires a fresh TTK run.
- `candidateUV` was fine-tuned from the pretrained CNN checkpoint. Results may change with different lambda settings or more epochs.
