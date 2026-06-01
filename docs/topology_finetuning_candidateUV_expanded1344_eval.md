# candidateUV_expanded1344 fine-tuning evaluation

**Generated:** 2026-05-29

**Candidate:** `candidateUV_expanded1344`

**Samples evaluated:** 168

**Methods compared:** bicubic, cnn, gan, candidateUV_expanded1344

> **SSIM note:** SSIM was not computed because skimage could not be imported in the current NumPy environment (`ValueError('numpy.dtype size changed, may indicate binary incompatibility. Expected 96 from C header, got 88 from PyObject')`). All SSIM values are NaN.

> Note: PD/MT distances for `candidateUV_expanded1344` are N/A — TTK has not been re-run on these outputs. Values for CNN/GAN are loaded from the pre-computed merged CSV.

## Summary metrics table

| Metric | Bicubic | Cnn | Gan | Candidateuv_expanded1344 |
|---|---|---|---|---|
| PSNRuv (dB) | 32.2202 | 31.1925 | 29.1380 | 33.5246 |
| SSIM on speed | N/A | N/A | N/A | N/A |
| Speed MAE (m/s) | 0.5963 | 0.6941 | 0.9026 | 0.5121 |
| Speed RMSE (m/s) | 1.0156 | 1.1078 | 1.3775 | 0.8770 |
| WPD MAE (m³/s³) | 193.5403 | 231.6709 | 310.7328 | 165.3896 |
| WPD Wasserstein-1 | 55.5103 | 45.2713 | 85.6191 | 34.4141 |
| WPD bias abs (m³/s³) | 41.2244 | 35.3439 | 78.7236 | 22.4428 |
| Gradient MAE | 0.3696 | 0.3491 | 0.3806 | 0.3279 |
| Gradient W1 | 0.3282 | 0.2329 | 0.0564 | 0.2498 |
| Gradient kurtosis abs Δ | 6.6163 | 3.7004 | 4.2010 | 3.8761 |
| PSD log-L2 | 1.3625 | 0.8335 | 0.5139 | 1.1253 |
| PSD slope abs Δ | 1.6684 | 0.9150 | 0.9482 | 1.4343 |
| Exceedance abs Δ s>5 | 0.0095 | 0.0042 | 0.0082 | 0.0068 |
| Exceedance abs Δ s>10 | 0.0085 | 0.0066 | 0.0243 | 0.0050 |
| Exceedance abs Δ s>15 | 0.0074 | 0.0062 | 0.0096 | 0.0046 |
| Exceedance abs Δ p90 | 0.0109 | 0.0103 | 0.0123 | 0.0062 |
| Component-count curve L1 | 173.1855 | 115.5278 | 124.6567 | 139.8145 |
| PD distance (TTK) | N/A | 27.4063 | 20.8641 | N/A |
| MT distance (TTK) | N/A | 5.8678 | 8.3481 | N/A |

## 7 Key evaluation questions

### Q1. Did candidateUV_expanded1344 preserve CNN's direct-fidelity advantage?

- **PSNRuv**: CNN=31.1925, GAN=29.1380, candidateUV_expanded1344=33.5246. Δ(candidateUV_expanded1344−CNN)=+2.3321 (▲ better), improved on 168/168 samples.
- **SSIM**: CNN=N/A, GAN=N/A, candidateUV_expanded1344=N/A. Δ(candidateUV_expanded1344−CNN)=N/A, improved on 0/0 samples.
- **Speed MAE**: CNN=0.6941, GAN=0.9026, candidateUV_expanded1344=0.5121. Δ(candidateUV_expanded1344−CNN)=-0.1820 (▲ better), improved on 168/168 samples.
- **Speed RMSE**: CNN=1.1078, GAN=1.3775, candidateUV_expanded1344=0.8770. Δ(candidateUV_expanded1344−CNN)=-0.2307 (▲ better), improved on 168/168 samples.

### Q2. Did candidateUV_expanded1344 improve scalar speed or WPD metrics?

- **WPD MAE**: CNN=231.6709, candidateUV_expanded1344=165.3896. Δ=-66.2813 (▲ better), improved on 168/168 samples.
- **WPD W1**: CNN=45.2713, candidateUV_expanded1344=34.4141. Δ=-10.8572 (▲ better), improved on 110/168 samples.
- **WPD bias abs**: CNN=35.3439, candidateUV_expanded1344=22.4428. Δ=-12.9011 (▲ better), improved on 109/168 samples.

### Q3. Did candidateUV_expanded1344 improve gradient metrics?

- **Gradient MAE**: CNN=0.3491, candidateUV_expanded1344=0.3279. Δ=-0.0211 (▲ better), improved on 168/168 samples.
- **Gradient W1**: CNN=0.2329, candidateUV_expanded1344=0.2498. Δ=+0.0168 (▼ worse), improved on 32/168 samples.
- **Gradient kurtosis abs Δ**: CNN=3.7004, candidateUV_expanded1344=3.8761. Δ=+0.1758 (▼ worse), improved on 74/168 samples.

### Q4. Did candidateUV_expanded1344 improve exceedance metrics, especially s>5?

- **Exceedance abs Δ s>5**: CNN=0.00424, candidateUV_expanded1344=0.00681. Δ=+0.0026 (▼ worse), improved on 24/168 samples.
- **Exceedance abs Δ s>10**: CNN=0.00658, candidateUV_expanded1344=0.00503. Δ=-0.0015 (▲ better), improved on 96/168 samples.
- **Exceedance abs Δ s>15**: CNN=0.00621, candidateUV_expanded1344=0.00458. Δ=-0.0016 (▲ better), improved on 103/168 samples.
- **Exceedance abs Δ p90**: CNN=0.01025, candidateUV_expanded1344=0.00625. Δ=-0.0040 (▲ better), improved on 122/168 samples.

### Q5. Did candidateUV_expanded1344 move component-count behavior toward GT?

- **Component-count curve L1**: CNN=115.5278, candidateUV_expanded1344=139.8145. Δ=+24.2867 (▼ worse), improved on 5/168 samples.
- **Comp-count abs Δ t5**: CNN=64.3810, candidateUV_expanded1344=68.8155. Δ=+4.4345 (▼ worse), improved on 40/168 samples.
- **Comp-count abs Δ t10**: CNN=187.1667, candidateUV_expanded1344=226.7202. Δ=+39.5536 (▼ worse), improved on 4/168 samples.
- **Comp-count abs Δ t15**: CNN=116.8095, candidateUV_expanded1344=140.6250. Δ=+23.8155 (▼ worse), improved on 23/168 samples.

### Q6. Did candidateUV_expanded1344 improve PD or MT distances?

PD and MT distances for `candidateUV_expanded1344` are not available in this evaluation (TTK was not re-run on these outputs). Pre-computed values for CNN and GAN:

- **PD distance**: CNN=27.4063, GAN=20.8641, candidateUV_expanded1344=N/A (requires TTK run)
- **MT distance**: CNN=5.8678, GAN=8.3481, candidateUV_expanded1344=N/A (requires TTK run)

To compute PD/MT for `candidateUV_expanded1344`: convert outputs to VTI via `scripts/convert_phire_to_vti.py`, then run `scripts/compute_composite_tree_distance.py`.

### Q7. Did candidateUV_expanded1344 change the adjacent clusters (samples 10–13, 90–93, 162–163)?

See `adjacent_cluster_table_candidateUV_expanded1344.csv` for full detail. Summary of PSNRuv, speed MAE, and gradient MAE changes:

| Sample | Δ PSNRuv | Δ Speed MAE | Δ Grad MAE | Δ Exceed s>5 |
|---|---|---|---|---|
| 10 | 2.514 | -0.2035 | -0.0192 | 0.00478 |
| 11 | 2.389 | -0.1988 | -0.0198 | 0.00302 |
| 12 | 2.424 | -0.1981 | -0.0247 | 0.00310 |
| 13 | 2.352 | -0.1896 | -0.0222 | 0.00019 |
| 90 | 2.162 | -0.1406 | -0.0175 | 0.00314 |
| 91 | 2.182 | -0.1450 | -0.0171 | 0.00684 |
| 92 | 2.184 | -0.1569 | -0.0175 | 0.01160 |
| 93 | 2.158 | -0.1680 | -0.0143 | 0.00810 |
| 162 | 3.636 | -0.1497 | -0.0238 | -0.00528 |
| 163 | 3.541 | -0.1500 | -0.0254 | -0.00104 |

## Pairwise summary: CNN vs candidateUV_expanded1344

| Metric | Mean CNN | Mean candidateUV_expanded1344 | Δ | N improved | N worsened |
|---|---|---|---|---|---|
| psnruv | 31.1925 | 33.5246 | 2.3321 | 168 | 0 |
| ssim | N/A | N/A | N/A | 0 | 0 |
| speed_mae | 0.6941 | 0.5121 | -0.1820 | 168 | 0 |
| speed_rmse | 1.1078 | 0.8770 | -0.2307 | 168 | 0 |
| wpd_bias_abs | 35.3439 | 22.4428 | -12.9011 | 109 | 59 |
| wpd_mae | 231.6709 | 165.3896 | -66.2813 | 168 | 0 |
| wpd_w1 | 45.2713 | 34.4141 | -10.8572 | 110 | 58 |
| psd_log_l2 | 0.8335 | 1.1253 | 0.2918 | 0 | 168 |
| psd_slope_abs_delta | 0.9150 | 1.4343 | 0.5193 | 0 | 168 |
| grad_mae | 0.3491 | 0.3279 | -0.0211 | 168 | 0 |
| grad_w1 | 0.2329 | 0.2498 | 0.0168 | 32 | 136 |
| grad_kurtosis_abs_delta | 3.7004 | 3.8761 | 0.1758 | 74 | 94 |
| exceed_abs_t5 | 0.0042 | 0.0068 | 0.0026 | 24 | 144 |
| exceed_abs_t10 | 0.0066 | 0.0050 | -0.0015 | 96 | 72 |
| exceed_abs_t15 | 0.0062 | 0.0046 | -0.0016 | 103 | 65 |
| exceed_abs_p90 | 0.0103 | 0.0062 | -0.0040 | 122 | 46 |
| comp_curve_l1 | 115.5278 | 139.8145 | 24.2867 | 5 | 163 |
| pd_distance | N/A | N/A | N/A | 0 | 0 |
| mt_distance | N/A | N/A | N/A | 0 | 0 |

## Output files

- `ttk_runs_fixed/topology_finetuning/candidateUV_expanded1344_eval/all_sample_metrics_candidateUV_expanded1344.csv` — per-sample metrics for all methods
- `ttk_runs_fixed/topology_finetuning/candidateUV_expanded1344_eval/pairwise_cnn_vs_candidateUV_expanded1344.csv` — per-metric delta table
- `ttk_runs_fixed/topology_finetuning/candidateUV_expanded1344_eval/winner_counts_candidateUV_expanded1344.csv` — win counts per method
- `ttk_runs_fixed/topology_finetuning/candidateUV_expanded1344_eval/adjacent_cluster_table_candidateUV_expanded1344.csv` — adjacent-cluster detail
- `docs/topology_finetuning_candidateUV_expanded1344_eval.md` — this report

## Notes

- PSNRuv is computed on physical [u,v] arrays using per-sample dynamic data range (GT max − GT min).
- All speed-based metrics use scalar speed = sqrt(u² + v²) in physical units (m/s).
- Exceedance fractions use GT percentile thresholds (p90/p95/p99) computed per sample.
- Component counts use 8-connectivity on speed superlevel sets.
- PD/MT distances for CNN and GAN are loaded from pre-computed merged CSV; `candidateUV_expanded1344` requires a fresh TTK run.
- `candidateUV_expanded1344` was fine-tuned from the pretrained CNN checkpoint. Results may change with different lambda settings or more epochs.
