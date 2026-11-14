# candidateF_grad_PERS_only_expanded2688 fine-tuning evaluation

**Generated:** 2026-10-14

**Candidate:** `candidateF_grad_PERS_only_expanded2688`

**Samples evaluated:** 168

**Methods compared:** bicubic, cnn, gan, candidateF_grad_PERS_only_expanded2688

> **SSIM note:** SSIM was not computed because skimage could not be imported in the current NumPy environment (`ValueError('numpy.dtype size changed, may indicate binary incompatibility. Expected 96 from C header, got 88 from PyObject')`). All SSIM values are NaN.

> Note: PD/MT distances for `candidateF_grad_PERS_only_expanded2688` are N/A — TTK has not been re-run on these outputs. Values for CNN/GAN are loaded from the pre-computed merged CSV.

## Summary metrics table

| Metric | Bicubic | Cnn | Gan | Candidatef_grad_pers_only_expanded2688 |
|---|---|---|---|---|
| PSNRuv (dB) | 32.2202 | 31.1925 | 29.1380 | 32.6119 |
| SSIM on speed | N/A | N/A | N/A | N/A |
| Speed MAE (m/s) | 0.5963 | 0.6941 | 0.9026 | 0.5805 |
| Speed RMSE (m/s) | 1.0156 | 1.1078 | 1.3775 | 0.9781 |
| WPD MAE (m³/s³) | 193.5403 | 231.6709 | 310.7328 | 188.9087 |
| WPD Wasserstein-1 | 55.5103 | 45.2713 | 85.6191 | 16.4613 |
| WPD bias abs (m³/s³) | 41.2244 | 35.3439 | 78.7236 | 10.5507 |
| Gradient MAE | 0.3696 | 0.3491 | 0.3806 | 0.3116 |
| Gradient W1 | 0.3282 | 0.2329 | 0.0564 | 0.1412 |
| Gradient kurtosis abs Δ | 6.6163 | 3.7004 | 4.2010 | 3.1732 |
| PSD log-L2 | 1.3625 | 0.8335 | 0.5139 | 0.8699 |
| PSD slope abs Δ | 1.6684 | 0.9150 | 0.9482 | 1.4574 |
| Exceedance abs Δ s>5 | 0.0095 | 0.0042 | 0.0082 | 0.0021 |
| Exceedance abs Δ s>10 | 0.0085 | 0.0066 | 0.0243 | 0.0020 |
| Exceedance abs Δ s>15 | 0.0074 | 0.0062 | 0.0096 | 0.0019 |
| Exceedance abs Δ p90 | 0.0109 | 0.0103 | 0.0123 | 0.0033 |
| Component-count curve L1 | 173.1855 | 115.5278 | 124.6567 | 77.0109 |
| PD distance (TTK) | N/A | 27.4063 | 20.8641 | N/A |
| MT distance (TTK) | N/A | 5.8678 | 8.3481 | N/A |

## 7 Key evaluation questions

### Q1. Did candidateF_grad_PERS_only_expanded2688 preserve CNN's direct-fidelity advantage?

- **PSNRuv**: CNN=31.1925, GAN=29.1380, candidateF_grad_PERS_only_expanded2688=32.6119. Δ(candidateF_grad_PERS_only_expanded2688−CNN)=+1.4194 (▲ better), improved on 168/168 samples.
- **SSIM**: CNN=N/A, GAN=N/A, candidateF_grad_PERS_only_expanded2688=N/A. Δ(candidateF_grad_PERS_only_expanded2688−CNN)=N/A, improved on 0/0 samples.
- **Speed MAE**: CNN=0.6941, GAN=0.9026, candidateF_grad_PERS_only_expanded2688=0.5805. Δ(candidateF_grad_PERS_only_expanded2688−CNN)=-0.1136 (▲ better), improved on 168/168 samples.
- **Speed RMSE**: CNN=1.1078, GAN=1.3775, candidateF_grad_PERS_only_expanded2688=0.9781. Δ(candidateF_grad_PERS_only_expanded2688−CNN)=-0.1297 (▲ better), improved on 168/168 samples.

### Q2. Did candidateF_grad_PERS_only_expanded2688 improve scalar speed or WPD metrics?

- **WPD MAE**: CNN=231.6709, candidateF_grad_PERS_only_expanded2688=188.9087. Δ=-42.7622 (▲ better), improved on 168/168 samples.
- **WPD W1**: CNN=45.2713, candidateF_grad_PERS_only_expanded2688=16.4613. Δ=-28.8100 (▲ better), improved on 158/168 samples.
- **WPD bias abs**: CNN=35.3439, candidateF_grad_PERS_only_expanded2688=10.5507. Δ=-24.7932 (▲ better), improved on 147/168 samples.

### Q3. Did candidateF_grad_PERS_only_expanded2688 improve gradient metrics?

- **Gradient MAE**: CNN=0.3491, candidateF_grad_PERS_only_expanded2688=0.3116. Δ=-0.0375 (▲ better), improved on 168/168 samples.
- **Gradient W1**: CNN=0.2329, candidateF_grad_PERS_only_expanded2688=0.1412. Δ=-0.0917 (▲ better), improved on 167/168 samples.
- **Gradient kurtosis abs Δ**: CNN=3.7004, candidateF_grad_PERS_only_expanded2688=3.1732. Δ=-0.5272 (▲ better), improved on 68/168 samples.

### Q4. Did candidateF_grad_PERS_only_expanded2688 improve exceedance metrics, especially s>5?

- **Exceedance abs Δ s>5**: CNN=0.00424, candidateF_grad_PERS_only_expanded2688=0.00208. Δ=-0.0022 (▲ better), improved on 131/168 samples.
- **Exceedance abs Δ s>10**: CNN=0.00658, candidateF_grad_PERS_only_expanded2688=0.00204. Δ=-0.0045 (▲ better), improved on 139/168 samples.
- **Exceedance abs Δ s>15**: CNN=0.00621, candidateF_grad_PERS_only_expanded2688=0.00194. Δ=-0.0043 (▲ better), improved on 140/168 samples.
- **Exceedance abs Δ p90**: CNN=0.01025, candidateF_grad_PERS_only_expanded2688=0.00326. Δ=-0.0070 (▲ better), improved on 146/168 samples.

### Q5. Did candidateF_grad_PERS_only_expanded2688 move component-count behavior toward GT?

- **Component-count curve L1**: CNN=115.5278, candidateF_grad_PERS_only_expanded2688=77.0109. Δ=-38.5169 (▲ better), improved on 159/168 samples.
- **Comp-count abs Δ t5**: CNN=64.3810, candidateF_grad_PERS_only_expanded2688=53.8333. Δ=-10.5476 (▲ better), improved on 136/168 samples.
- **Comp-count abs Δ t10**: CNN=187.1667, candidateF_grad_PERS_only_expanded2688=156.4107. Δ=-30.7560 (▲ better), improved on 142/168 samples.
- **Comp-count abs Δ t15**: CNN=116.8095, candidateF_grad_PERS_only_expanded2688=61.3869. Δ=-55.4226 (▲ better), improved on 151/168 samples.

### Q6. Did candidateF_grad_PERS_only_expanded2688 improve PD or MT distances?

PD and MT distances for `candidateF_grad_PERS_only_expanded2688` are not available in this evaluation (TTK was not re-run on these outputs). Pre-computed values for CNN and GAN:

- **PD distance**: CNN=27.4063, GAN=20.8641, candidateF_grad_PERS_only_expanded2688=N/A (requires TTK run)
- **MT distance**: CNN=5.8678, GAN=8.3481, candidateF_grad_PERS_only_expanded2688=N/A (requires TTK run)

To compute PD/MT for `candidateF_grad_PERS_only_expanded2688`: convert outputs to VTI via `scripts/convert_phire_to_vti.py`, then run `scripts/compute_composite_tree_distance.py`.

### Q7. Did candidateF_grad_PERS_only_expanded2688 change the adjacent clusters (samples 10–13, 90–93, 162–163)?

See `adjacent_cluster_table_candidateF_grad_PERS_only_expanded2688.csv` for full detail. Summary of PSNRuv, speed MAE, and gradient MAE changes:

| Sample | Δ PSNRuv | Δ Speed MAE | Δ Grad MAE | Δ Exceed s>5 |
|---|---|---|---|---|
| 10 | 1.675 | -0.1343 | -0.0373 | -0.00347 |
| 11 | 1.564 | -0.1289 | -0.0380 | -0.00229 |
| 12 | 1.598 | -0.1298 | -0.0419 | -0.00261 |
| 13 | 1.459 | -0.1166 | -0.0367 | -0.00326 |
| 90 | 1.409 | -0.0921 | -0.0298 | 0.00036 |
| 91 | 1.452 | -0.0977 | -0.0303 | 0.00559 |
| 92 | 1.385 | -0.1034 | -0.0345 | 0.00716 |
| 93 | 1.208 | -0.0989 | -0.0364 | -0.00033 |
| 162 | 2.344 | -0.1072 | -0.0206 | -0.00285 |
| 163 | 2.322 | -0.1075 | -0.0227 | 0.00140 |

## Pairwise summary: CNN vs candidateF_grad_PERS_only_expanded2688

| Metric | Mean CNN | Mean candidateF_grad_PERS_only_expanded2688 | Δ | N improved | N worsened |
|---|---|---|---|---|---|
| psnruv | 31.1925 | 32.6119 | 1.4194 | 168 | 0 |
| ssim | N/A | N/A | N/A | 0 | 0 |
| speed_mae | 0.6941 | 0.5805 | -0.1136 | 168 | 0 |
| speed_rmse | 1.1078 | 0.9781 | -0.1297 | 168 | 0 |
| wpd_bias_abs | 35.3439 | 10.5507 | -24.7932 | 147 | 21 |
| wpd_mae | 231.6709 | 188.9087 | -42.7622 | 168 | 0 |
| wpd_w1 | 45.2713 | 16.4613 | -28.8100 | 158 | 10 |
| psd_log_l2 | 0.8335 | 0.8699 | 0.0364 | 72 | 96 |
| psd_slope_abs_delta | 0.9150 | 1.4574 | 0.5424 | 0 | 168 |
| grad_mae | 0.3491 | 0.3116 | -0.0375 | 168 | 0 |
| grad_w1 | 0.2329 | 0.1412 | -0.0917 | 167 | 1 |
| grad_kurtosis_abs_delta | 3.7004 | 3.1732 | -0.5272 | 68 | 100 |
| exceed_abs_t5 | 0.0042 | 0.0021 | -0.0022 | 131 | 37 |
| exceed_abs_t10 | 0.0066 | 0.0020 | -0.0045 | 139 | 29 |
| exceed_abs_t15 | 0.0062 | 0.0019 | -0.0043 | 140 | 28 |
| exceed_abs_p90 | 0.0103 | 0.0033 | -0.0070 | 146 | 22 |
| comp_curve_l1 | 115.5278 | 77.0109 | -38.5169 | 159 | 9 |
| pd_distance | N/A | N/A | N/A | 0 | 0 |
| mt_distance | N/A | N/A | N/A | 0 | 0 |

## Output files

- `ttk_runs_fixed/topology_finetuning/candidateF_grad_PERS_only_expanded2688_eval/all_sample_metrics_candidateF_grad_PERS_only_expanded2688.csv` — per-sample metrics for all methods
- `ttk_runs_fixed/topology_finetuning/candidateF_grad_PERS_only_expanded2688_eval/pairwise_cnn_vs_candidateF_grad_PERS_only_expanded2688.csv` — per-metric delta table
- `ttk_runs_fixed/topology_finetuning/candidateF_grad_PERS_only_expanded2688_eval/winner_counts_candidateF_grad_PERS_only_expanded2688.csv` — win counts per method
- `ttk_runs_fixed/topology_finetuning/candidateF_grad_PERS_only_expanded2688_eval/adjacent_cluster_table_candidateF_grad_PERS_only_expanded2688.csv` — adjacent-cluster detail
- `docs/topology_finetuning_candidateF_grad_PERS_only_expanded2688_eval.md` — this report

## Notes

- PSNRuv is computed on physical [u,v] arrays using per-sample dynamic data range (GT max − GT min).
- All speed-based metrics use scalar speed = sqrt(u² + v²) in physical units (m/s).
- Exceedance fractions use GT percentile thresholds (p90/p95/p99) computed per sample.
- Component counts use 8-connectivity on speed superlevel sets.
- PD/MT distances for CNN and GAN are loaded from pre-computed merged CSV; `candidateF_grad_PERS_only_expanded2688` requires a fresh TTK run.
- `candidateF_grad_PERS_only_expanded2688` was fine-tuned from the pretrained CNN checkpoint. Results may change with different lambda settings or more epochs.
