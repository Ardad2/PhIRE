# candidateF_grad_levelset_E2_low_expanded2688 fine-tuning evaluation

**Generated:** 2026-07-19

**Candidate:** `candidateF_grad_levelset_E2_low_expanded2688`

**Samples evaluated:** 168

**Methods compared:** bicubic, cnn, gan, candidateF_grad_levelset_E2_low_expanded2688

> **SSIM note:** SSIM was not computed because skimage could not be imported in the current NumPy environment (`ValueError('numpy.dtype size changed, may indicate binary incompatibility. Expected 96 from C header, got 88 from PyObject')`). All SSIM values are NaN.

> Note: PD/MT distances for `candidateF_grad_levelset_E2_low_expanded2688` are N/A — TTK has not been re-run on these outputs. Values for CNN/GAN are loaded from the pre-computed merged CSV.

## Summary metrics table

| Metric | Bicubic | Cnn | Gan | Candidatef_grad_levelset_e2_low_expanded2688 |
|---|---|---|---|---|
| PSNRuv (dB) | 32.2202 | 31.1925 | 29.1380 | 32.5762 |
| SSIM on speed | N/A | N/A | N/A | N/A |
| Speed MAE (m/s) | 0.5963 | 0.6941 | 0.9026 | 0.5788 |
| Speed RMSE (m/s) | 1.0156 | 1.1078 | 1.3775 | 0.9776 |
| WPD MAE (m³/s³) | 193.5403 | 231.6709 | 310.7328 | 186.8230 |
| WPD Wasserstein-1 | 55.5103 | 45.2713 | 85.6191 | 20.1529 |
| WPD bias abs (m³/s³) | 41.2244 | 35.3439 | 78.7236 | 17.4210 |
| Gradient MAE | 0.3696 | 0.3491 | 0.3806 | 0.3122 |
| Gradient W1 | 0.3282 | 0.2329 | 0.0564 | 0.1459 |
| Gradient kurtosis abs Δ | 6.6163 | 3.7004 | 4.2010 | 3.1483 |
| PSD log-L2 | 1.3625 | 0.8335 | 0.5139 | 0.8292 |
| PSD slope abs Δ | 1.6684 | 0.9150 | 0.9482 | 1.1478 |
| Exceedance abs Δ s>5 | 0.0095 | 0.0042 | 0.0082 | 0.0021 |
| Exceedance abs Δ s>10 | 0.0085 | 0.0066 | 0.0243 | 0.0035 |
| Exceedance abs Δ s>15 | 0.0074 | 0.0062 | 0.0096 | 0.0023 |
| Exceedance abs Δ p90 | 0.0109 | 0.0103 | 0.0123 | 0.0038 |
| Component-count curve L1 | 173.1855 | 115.5278 | 124.6567 | 83.0784 |
| PD distance (TTK) | N/A | 27.4063 | 20.8641 | N/A |
| MT distance (TTK) | N/A | 5.8678 | 8.3481 | N/A |

## 7 Key evaluation questions

### Q1. Did candidateF_grad_levelset_E2_low_expanded2688 preserve CNN's direct-fidelity advantage?

- **PSNRuv**: CNN=31.1925, GAN=29.1380, candidateF_grad_levelset_E2_low_expanded2688=32.5762. Δ(candidateF_grad_levelset_E2_low_expanded2688−CNN)=+1.3837 (▲ better), improved on 168/168 samples.
- **SSIM**: CNN=N/A, GAN=N/A, candidateF_grad_levelset_E2_low_expanded2688=N/A. Δ(candidateF_grad_levelset_E2_low_expanded2688−CNN)=N/A, improved on 0/0 samples.
- **Speed MAE**: CNN=0.6941, GAN=0.9026, candidateF_grad_levelset_E2_low_expanded2688=0.5788. Δ(candidateF_grad_levelset_E2_low_expanded2688−CNN)=-0.1153 (▲ better), improved on 168/168 samples.
- **Speed RMSE**: CNN=1.1078, GAN=1.3775, candidateF_grad_levelset_E2_low_expanded2688=0.9776. Δ(candidateF_grad_levelset_E2_low_expanded2688−CNN)=-0.1301 (▲ better), improved on 168/168 samples.

### Q2. Did candidateF_grad_levelset_E2_low_expanded2688 improve scalar speed or WPD metrics?

- **WPD MAE**: CNN=231.6709, candidateF_grad_levelset_E2_low_expanded2688=186.8230. Δ=-44.8479 (▲ better), improved on 168/168 samples.
- **WPD W1**: CNN=45.2713, candidateF_grad_levelset_E2_low_expanded2688=20.1529. Δ=-25.1184 (▲ better), improved on 158/168 samples.
- **WPD bias abs**: CNN=35.3439, candidateF_grad_levelset_E2_low_expanded2688=17.4210. Δ=-17.9229 (▲ better), improved on 151/168 samples.

### Q3. Did candidateF_grad_levelset_E2_low_expanded2688 improve gradient metrics?

- **Gradient MAE**: CNN=0.3491, candidateF_grad_levelset_E2_low_expanded2688=0.3122. Δ=-0.0369 (▲ better), improved on 168/168 samples.
- **Gradient W1**: CNN=0.2329, candidateF_grad_levelset_E2_low_expanded2688=0.1459. Δ=-0.0871 (▲ better), improved on 167/168 samples.
- **Gradient kurtosis abs Δ**: CNN=3.7004, candidateF_grad_levelset_E2_low_expanded2688=3.1483. Δ=-0.5521 (▲ better), improved on 68/168 samples.

### Q4. Did candidateF_grad_levelset_E2_low_expanded2688 improve exceedance metrics, especially s>5?

- **Exceedance abs Δ s>5**: CNN=0.00424, candidateF_grad_levelset_E2_low_expanded2688=0.00212. Δ=-0.0021 (▲ better), improved on 132/168 samples.
- **Exceedance abs Δ s>10**: CNN=0.00658, candidateF_grad_levelset_E2_low_expanded2688=0.00348. Δ=-0.0031 (▲ better), improved on 126/168 samples.
- **Exceedance abs Δ s>15**: CNN=0.00621, candidateF_grad_levelset_E2_low_expanded2688=0.00226. Δ=-0.0039 (▲ better), improved on 143/168 samples.
- **Exceedance abs Δ p90**: CNN=0.01025, candidateF_grad_levelset_E2_low_expanded2688=0.00385. Δ=-0.0064 (▲ better), improved on 150/168 samples.

### Q5. Did candidateF_grad_levelset_E2_low_expanded2688 move component-count behavior toward GT?

- **Component-count curve L1**: CNN=115.5278, candidateF_grad_levelset_E2_low_expanded2688=83.0784. Δ=-32.4494 (▲ better), improved on 159/168 samples.
- **Comp-count abs Δ t5**: CNN=64.3810, candidateF_grad_levelset_E2_low_expanded2688=53.1726. Δ=-11.2083 (▲ better), improved on 139/168 samples.
- **Comp-count abs Δ t10**: CNN=187.1667, candidateF_grad_levelset_E2_low_expanded2688=158.2024. Δ=-28.9643 (▲ better), improved on 147/168 samples.
- **Comp-count abs Δ t15**: CNN=116.8095, candidateF_grad_levelset_E2_low_expanded2688=71.9048. Δ=-44.9048 (▲ better), improved on 153/168 samples.

### Q6. Did candidateF_grad_levelset_E2_low_expanded2688 improve PD or MT distances?

PD and MT distances for `candidateF_grad_levelset_E2_low_expanded2688` are not available in this evaluation (TTK was not re-run on these outputs). Pre-computed values for CNN and GAN:

- **PD distance**: CNN=27.4063, GAN=20.8641, candidateF_grad_levelset_E2_low_expanded2688=N/A (requires TTK run)
- **MT distance**: CNN=5.8678, GAN=8.3481, candidateF_grad_levelset_E2_low_expanded2688=N/A (requires TTK run)

To compute PD/MT for `candidateF_grad_levelset_E2_low_expanded2688`: convert outputs to VTI via `scripts/convert_phire_to_vti.py`, then run `scripts/compute_composite_tree_distance.py`.

### Q7. Did candidateF_grad_levelset_E2_low_expanded2688 change the adjacent clusters (samples 10–13, 90–93, 162–163)?

See `adjacent_cluster_table_candidateF_grad_levelset_E2_low_expanded2688.csv` for full detail. Summary of PSNRuv, speed MAE, and gradient MAE changes:

| Sample | Δ PSNRuv | Δ Speed MAE | Δ Grad MAE | Δ Exceed s>5 |
|---|---|---|---|---|
| 10 | 1.602 | -0.1342 | -0.0370 | -0.00284 |
| 11 | 1.514 | -0.1310 | -0.0376 | -0.00202 |
| 12 | 1.542 | -0.1318 | -0.0408 | -0.00173 |
| 13 | 1.418 | -0.1195 | -0.0360 | -0.00231 |
| 90 | 1.381 | -0.0930 | -0.0294 | -0.00156 |
| 91 | 1.408 | -0.0985 | -0.0297 | 0.00298 |
| 92 | 1.333 | -0.1039 | -0.0340 | 0.00457 |
| 93 | 1.160 | -0.0989 | -0.0354 | -0.00348 |
| 162 | 2.426 | -0.1097 | -0.0216 | -0.00252 |
| 163 | 2.266 | -0.1068 | -0.0227 | 0.00135 |

## Pairwise summary: CNN vs candidateF_grad_levelset_E2_low_expanded2688

| Metric | Mean CNN | Mean candidateF_grad_levelset_E2_low_expanded2688 | Δ | N improved | N worsened |
|---|---|---|---|---|---|
| psnruv | 31.1925 | 32.5762 | 1.3837 | 168 | 0 |
| ssim | N/A | N/A | N/A | 0 | 0 |
| speed_mae | 0.6941 | 0.5788 | -0.1153 | 168 | 0 |
| speed_rmse | 1.1078 | 0.9776 | -0.1301 | 168 | 0 |
| wpd_bias_abs | 35.3439 | 17.4210 | -17.9229 | 151 | 17 |
| wpd_mae | 231.6709 | 186.8230 | -44.8479 | 168 | 0 |
| wpd_w1 | 45.2713 | 20.1529 | -25.1184 | 158 | 10 |
| psd_log_l2 | 0.8335 | 0.8292 | -0.0044 | 105 | 63 |
| psd_slope_abs_delta | 0.9150 | 1.1478 | 0.2327 | 2 | 166 |
| grad_mae | 0.3491 | 0.3122 | -0.0369 | 168 | 0 |
| grad_w1 | 0.2329 | 0.1459 | -0.0871 | 167 | 1 |
| grad_kurtosis_abs_delta | 3.7004 | 3.1483 | -0.5521 | 68 | 100 |
| exceed_abs_t5 | 0.0042 | 0.0021 | -0.0021 | 132 | 36 |
| exceed_abs_t10 | 0.0066 | 0.0035 | -0.0031 | 126 | 42 |
| exceed_abs_t15 | 0.0062 | 0.0023 | -0.0039 | 143 | 25 |
| exceed_abs_p90 | 0.0103 | 0.0038 | -0.0064 | 150 | 18 |
| comp_curve_l1 | 115.5278 | 83.0784 | -32.4494 | 159 | 8 |
| pd_distance | N/A | N/A | N/A | 0 | 0 |
| mt_distance | N/A | N/A | N/A | 0 | 0 |

## Output files

- `ttk_runs_fixed/topology_finetuning/candidateF_grad_levelset_E2_low_expanded2688_eval/all_sample_metrics_candidateF_grad_levelset_E2_low_expanded2688.csv` — per-sample metrics for all methods
- `ttk_runs_fixed/topology_finetuning/candidateF_grad_levelset_E2_low_expanded2688_eval/pairwise_cnn_vs_candidateF_grad_levelset_E2_low_expanded2688.csv` — per-metric delta table
- `ttk_runs_fixed/topology_finetuning/candidateF_grad_levelset_E2_low_expanded2688_eval/winner_counts_candidateF_grad_levelset_E2_low_expanded2688.csv` — win counts per method
- `ttk_runs_fixed/topology_finetuning/candidateF_grad_levelset_E2_low_expanded2688_eval/adjacent_cluster_table_candidateF_grad_levelset_E2_low_expanded2688.csv` — adjacent-cluster detail
- `docs/topology_finetuning_candidateF_grad_levelset_E2_low_expanded2688_eval.md` — this report

## Notes

- PSNRuv is computed on physical [u,v] arrays using per-sample dynamic data range (GT max − GT min).
- All speed-based metrics use scalar speed = sqrt(u² + v²) in physical units (m/s).
- Exceedance fractions use GT percentile thresholds (p90/p95/p99) computed per sample.
- Component counts use 8-connectivity on speed superlevel sets.
- PD/MT distances for CNN and GAN are loaded from pre-computed merged CSV; `candidateF_grad_levelset_E2_low_expanded2688` requires a fresh TTK run.
- `candidateF_grad_levelset_E2_low_expanded2688` was fine-tuned from the pretrained CNN checkpoint. Results may change with different lambda settings or more epochs.
