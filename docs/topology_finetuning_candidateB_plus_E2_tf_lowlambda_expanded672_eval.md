# candidateB_plus_E2_tf_lowlambda_expanded672 fine-tuning evaluation

**Generated:** 2026-07-09

**Candidate:** `candidateB_plus_E2_tf_lowlambda_expanded672`

**Samples evaluated:** 168

**Methods compared:** bicubic, cnn, gan, candidateB_plus_E2_tf_lowlambda_expanded672

> **SSIM note:** SSIM was not computed because skimage could not be imported in the current NumPy environment (`ValueError('numpy.dtype size changed, may indicate binary incompatibility. Expected 96 from C header, got 88 from PyObject')`). All SSIM values are NaN.

> Note: PD/MT distances for `candidateB_plus_E2_tf_lowlambda_expanded672` are N/A — TTK has not been re-run on these outputs. Values for CNN/GAN are loaded from the pre-computed merged CSV.

## Summary metrics table

| Metric | Bicubic | Cnn | Gan | Candidateb_plus_e2_tf_lowlambda_expanded672 |
|---|---|---|---|---|
| PSNRuv (dB) | 32.2202 | 31.1925 | 29.1380 | 32.1146 |
| SSIM on speed | N/A | N/A | N/A | N/A |
| Speed MAE (m/s) | 0.5963 | 0.6941 | 0.9026 | 0.6178 |
| Speed RMSE (m/s) | 1.0156 | 1.1078 | 1.3775 | 1.0240 |
| WPD MAE (m³/s³) | 193.5403 | 231.6709 | 310.7328 | 202.2449 |
| WPD Wasserstein-1 | 55.5103 | 45.2713 | 85.6191 | 19.5757 |
| WPD bias abs (m³/s³) | 41.2244 | 35.3439 | 78.7236 | 13.1372 |
| Gradient MAE | 0.3696 | 0.3491 | 0.3806 | 0.3212 |
| Gradient W1 | 0.3282 | 0.2329 | 0.0564 | 0.1492 |
| Gradient kurtosis abs Δ | 6.6163 | 3.7004 | 4.2010 | 3.0060 |
| PSD log-L2 | 1.3625 | 0.8335 | 0.5139 | 0.8096 |
| PSD slope abs Δ | 1.6684 | 0.9150 | 0.9482 | 1.0791 |
| Exceedance abs Δ s>5 | 0.0095 | 0.0042 | 0.0082 | 0.0027 |
| Exceedance abs Δ s>10 | 0.0085 | 0.0066 | 0.0243 | 0.0037 |
| Exceedance abs Δ s>15 | 0.0074 | 0.0062 | 0.0096 | 0.0018 |
| Exceedance abs Δ p90 | 0.0109 | 0.0103 | 0.0123 | 0.0032 |
| Component-count curve L1 | 173.1855 | 115.5278 | 124.6567 | 84.1935 |
| PD distance (TTK) | N/A | 27.4063 | 20.8641 | N/A |
| MT distance (TTK) | N/A | 5.8678 | 8.3481 | N/A |

## 7 Key evaluation questions

### Q1. Did candidateB_plus_E2_tf_lowlambda_expanded672 preserve CNN's direct-fidelity advantage?

- **PSNRuv**: CNN=31.1925, GAN=29.1380, candidateB_plus_E2_tf_lowlambda_expanded672=32.1146. Δ(candidateB_plus_E2_tf_lowlambda_expanded672−CNN)=+0.9221 (▲ better), improved on 168/168 samples.
- **SSIM**: CNN=N/A, GAN=N/A, candidateB_plus_E2_tf_lowlambda_expanded672=N/A. Δ(candidateB_plus_E2_tf_lowlambda_expanded672−CNN)=N/A, improved on 0/0 samples.
- **Speed MAE**: CNN=0.6941, GAN=0.9026, candidateB_plus_E2_tf_lowlambda_expanded672=0.6178. Δ(candidateB_plus_E2_tf_lowlambda_expanded672−CNN)=-0.0762 (▲ better), improved on 168/168 samples.
- **Speed RMSE**: CNN=1.1078, GAN=1.3775, candidateB_plus_E2_tf_lowlambda_expanded672=1.0240. Δ(candidateB_plus_E2_tf_lowlambda_expanded672−CNN)=-0.0838 (▲ better), improved on 168/168 samples.

### Q2. Did candidateB_plus_E2_tf_lowlambda_expanded672 improve scalar speed or WPD metrics?

- **WPD MAE**: CNN=231.6709, candidateB_plus_E2_tf_lowlambda_expanded672=202.2449. Δ=-29.4260 (▲ better), improved on 168/168 samples.
- **WPD W1**: CNN=45.2713, candidateB_plus_E2_tf_lowlambda_expanded672=19.5757. Δ=-25.6956 (▲ better), improved on 156/168 samples.
- **WPD bias abs**: CNN=35.3439, candidateB_plus_E2_tf_lowlambda_expanded672=13.1372. Δ=-22.2067 (▲ better), improved on 152/168 samples.

### Q3. Did candidateB_plus_E2_tf_lowlambda_expanded672 improve gradient metrics?

- **Gradient MAE**: CNN=0.3491, candidateB_plus_E2_tf_lowlambda_expanded672=0.3212. Δ=-0.0278 (▲ better), improved on 168/168 samples.
- **Gradient W1**: CNN=0.2329, candidateB_plus_E2_tf_lowlambda_expanded672=0.1492. Δ=-0.0838 (▲ better), improved on 162/168 samples.
- **Gradient kurtosis abs Δ**: CNN=3.7004, candidateB_plus_E2_tf_lowlambda_expanded672=3.0060. Δ=-0.6944 (▲ better), improved on 83/168 samples.

### Q4. Did candidateB_plus_E2_tf_lowlambda_expanded672 improve exceedance metrics, especially s>5?

- **Exceedance abs Δ s>5**: CNN=0.00424, candidateB_plus_E2_tf_lowlambda_expanded672=0.00274. Δ=-0.0015 (▲ better), improved on 116/168 samples.
- **Exceedance abs Δ s>10**: CNN=0.00658, candidateB_plus_E2_tf_lowlambda_expanded672=0.00368. Δ=-0.0029 (▲ better), improved on 117/168 samples.
- **Exceedance abs Δ s>15**: CNN=0.00621, candidateB_plus_E2_tf_lowlambda_expanded672=0.00183. Δ=-0.0044 (▲ better), improved on 137/168 samples.
- **Exceedance abs Δ p90**: CNN=0.01025, candidateB_plus_E2_tf_lowlambda_expanded672=0.00317. Δ=-0.0071 (▲ better), improved on 155/168 samples.

### Q5. Did candidateB_plus_E2_tf_lowlambda_expanded672 move component-count behavior toward GT?

- **Component-count curve L1**: CNN=115.5278, candidateB_plus_E2_tf_lowlambda_expanded672=84.1935. Δ=-31.3343 (▲ better), improved on 155/168 samples.
- **Comp-count abs Δ t5**: CNN=64.3810, candidateB_plus_E2_tf_lowlambda_expanded672=56.2440. Δ=-8.1369 (▲ better), improved on 128/168 samples.
- **Comp-count abs Δ t10**: CNN=187.1667, candidateB_plus_E2_tf_lowlambda_expanded672=164.4821. Δ=-22.6845 (▲ better), improved on 139/168 samples.
- **Comp-count abs Δ t15**: CNN=116.8095, candidateB_plus_E2_tf_lowlambda_expanded672=70.9345. Δ=-45.8750 (▲ better), improved on 154/168 samples.

### Q6. Did candidateB_plus_E2_tf_lowlambda_expanded672 improve PD or MT distances?

PD and MT distances for `candidateB_plus_E2_tf_lowlambda_expanded672` are not available in this evaluation (TTK was not re-run on these outputs). Pre-computed values for CNN and GAN:

- **PD distance**: CNN=27.4063, GAN=20.8641, candidateB_plus_E2_tf_lowlambda_expanded672=N/A (requires TTK run)
- **MT distance**: CNN=5.8678, GAN=8.3481, candidateB_plus_E2_tf_lowlambda_expanded672=N/A (requires TTK run)

To compute PD/MT for `candidateB_plus_E2_tf_lowlambda_expanded672`: convert outputs to VTI via `scripts/convert_phire_to_vti.py`, then run `scripts/compute_composite_tree_distance.py`.

### Q7. Did candidateB_plus_E2_tf_lowlambda_expanded672 change the adjacent clusters (samples 10–13, 90–93, 162–163)?

See `adjacent_cluster_table_candidateB_plus_E2_tf_lowlambda_expanded672.csv` for full detail. Summary of PSNRuv, speed MAE, and gradient MAE changes:

| Sample | Δ PSNRuv | Δ Speed MAE | Δ Grad MAE | Δ Exceed s>5 |
|---|---|---|---|---|
| 10 | 1.247 | -0.0982 | -0.0275 | -0.00315 |
| 11 | 1.097 | -0.0902 | -0.0262 | -0.00227 |
| 12 | 1.163 | -0.0917 | -0.0310 | -0.00127 |
| 13 | 1.035 | -0.0782 | -0.0257 | -0.00084 |
| 90 | 0.895 | -0.0616 | -0.0232 | -0.00110 |
| 91 | 0.951 | -0.0657 | -0.0225 | 0.00325 |
| 92 | 0.860 | -0.0705 | -0.0268 | 0.00624 |
| 93 | 0.757 | -0.0662 | -0.0280 | -0.00090 |
| 162 | 2.005 | -0.0829 | -0.0159 | -0.00040 |
| 163 | 1.836 | -0.0803 | -0.0167 | 0.00710 |

## Pairwise summary: CNN vs candidateB_plus_E2_tf_lowlambda_expanded672

| Metric | Mean CNN | Mean candidateB_plus_E2_tf_lowlambda_expanded672 | Δ | N improved | N worsened |
|---|---|---|---|---|---|
| psnruv | 31.1925 | 32.1146 | 0.9221 | 168 | 0 |
| ssim | N/A | N/A | N/A | 0 | 0 |
| speed_mae | 0.6941 | 0.6178 | -0.0762 | 168 | 0 |
| speed_rmse | 1.1078 | 1.0240 | -0.0838 | 168 | 0 |
| wpd_bias_abs | 35.3439 | 13.1372 | -22.2067 | 152 | 16 |
| wpd_mae | 231.6709 | 202.2449 | -29.4260 | 168 | 0 |
| wpd_w1 | 45.2713 | 19.5757 | -25.6956 | 156 | 12 |
| psd_log_l2 | 0.8335 | 0.8096 | -0.0239 | 113 | 55 |
| psd_slope_abs_delta | 0.9150 | 1.0791 | 0.1641 | 26 | 142 |
| grad_mae | 0.3491 | 0.3212 | -0.0278 | 168 | 0 |
| grad_w1 | 0.2329 | 0.1492 | -0.0838 | 162 | 6 |
| grad_kurtosis_abs_delta | 3.7004 | 3.0060 | -0.6944 | 83 | 85 |
| exceed_abs_t5 | 0.0042 | 0.0027 | -0.0015 | 116 | 52 |
| exceed_abs_t10 | 0.0066 | 0.0037 | -0.0029 | 117 | 51 |
| exceed_abs_t15 | 0.0062 | 0.0018 | -0.0044 | 137 | 31 |
| exceed_abs_p90 | 0.0103 | 0.0032 | -0.0071 | 155 | 12 |
| comp_curve_l1 | 115.5278 | 84.1935 | -31.3343 | 155 | 13 |
| pd_distance | N/A | N/A | N/A | 0 | 0 |
| mt_distance | N/A | N/A | N/A | 0 | 0 |

## Output files

- `ttk_runs_fixed/topology_finetuning/candidateB_plus_E2_tf_lowlambda_expanded672_eval/all_sample_metrics_candidateB_plus_E2_tf_lowlambda_expanded672.csv` — per-sample metrics for all methods
- `ttk_runs_fixed/topology_finetuning/candidateB_plus_E2_tf_lowlambda_expanded672_eval/pairwise_cnn_vs_candidateB_plus_E2_tf_lowlambda_expanded672.csv` — per-metric delta table
- `ttk_runs_fixed/topology_finetuning/candidateB_plus_E2_tf_lowlambda_expanded672_eval/winner_counts_candidateB_plus_E2_tf_lowlambda_expanded672.csv` — win counts per method
- `ttk_runs_fixed/topology_finetuning/candidateB_plus_E2_tf_lowlambda_expanded672_eval/adjacent_cluster_table_candidateB_plus_E2_tf_lowlambda_expanded672.csv` — adjacent-cluster detail
- `docs/topology_finetuning_candidateB_plus_E2_tf_lowlambda_expanded672_eval.md` — this report

## Notes

- PSNRuv is computed on physical [u,v] arrays using per-sample dynamic data range (GT max − GT min).
- All speed-based metrics use scalar speed = sqrt(u² + v²) in physical units (m/s).
- Exceedance fractions use GT percentile thresholds (p90/p95/p99) computed per sample.
- Component counts use 8-connectivity on speed superlevel sets.
- PD/MT distances for CNN and GAN are loaded from pre-computed merged CSV; `candidateB_plus_E2_tf_lowlambda_expanded672` requires a fresh TTK run.
- `candidateB_plus_E2_tf_lowlambda_expanded672` was fine-tuned from the pretrained CNN checkpoint. Results may change with different lambda settings or more epochs.
