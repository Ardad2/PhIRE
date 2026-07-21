# candidateE2_tf_lowlambda_expanded672 fine-tuning evaluation

**Generated:** 2026-07-09

**Candidate:** `candidateE2_tf_lowlambda_expanded672`

**Samples evaluated:** 168

**Methods compared:** bicubic, cnn, gan, candidateE2_tf_lowlambda_expanded672

> **SSIM note:** SSIM was not computed because skimage could not be imported in the current NumPy environment (`ValueError('numpy.dtype size changed, may indicate binary incompatibility. Expected 96 from C header, got 88 from PyObject')`). All SSIM values are NaN.

> Note: PD/MT distances for `candidateE2_tf_lowlambda_expanded672` are N/A — TTK has not been re-run on these outputs. Values for CNN/GAN are loaded from the pre-computed merged CSV.

## Summary metrics table

| Metric | Bicubic | Cnn | Gan | Candidatee2_tf_lowlambda_expanded672 |
|---|---|---|---|---|
| PSNRuv (dB) | 32.2202 | 31.1925 | 29.1380 | 32.1651 |
| SSIM on speed | N/A | N/A | N/A | N/A |
| Speed MAE (m/s) | 0.5963 | 0.6941 | 0.9026 | 0.6133 |
| Speed RMSE (m/s) | 1.0156 | 1.1078 | 1.3775 | 1.0170 |
| WPD MAE (m³/s³) | 193.5403 | 231.6709 | 310.7328 | 201.4855 |
| WPD Wasserstein-1 | 55.5103 | 45.2713 | 85.6191 | 16.3995 |
| WPD bias abs (m³/s³) | 41.2244 | 35.3439 | 78.7236 | 4.7166 |
| Gradient MAE | 0.3696 | 0.3491 | 0.3806 | 0.3210 |
| Gradient W1 | 0.3282 | 0.2329 | 0.0564 | 0.1529 |
| Gradient kurtosis abs Δ | 6.6163 | 3.7004 | 4.2010 | 2.8789 |
| PSD log-L2 | 1.3625 | 0.8335 | 0.5139 | 0.8141 |
| PSD slope abs Δ | 1.6684 | 0.9150 | 0.9482 | 1.0531 |
| Exceedance abs Δ s>5 | 0.0095 | 0.0042 | 0.0082 | 0.0029 |
| Exceedance abs Δ s>10 | 0.0085 | 0.0066 | 0.0243 | 0.0033 |
| Exceedance abs Δ s>15 | 0.0074 | 0.0062 | 0.0096 | 0.0016 |
| Exceedance abs Δ p90 | 0.0109 | 0.0103 | 0.0123 | 0.0018 |
| Component-count curve L1 | 173.1855 | 115.5278 | 124.6567 | 87.3800 |
| PD distance (TTK) | N/A | 27.4063 | 20.8641 | N/A |
| MT distance (TTK) | N/A | 5.8678 | 8.3481 | N/A |

## 7 Key evaluation questions

### Q1. Did candidateE2_tf_lowlambda_expanded672 preserve CNN's direct-fidelity advantage?

- **PSNRuv**: CNN=31.1925, GAN=29.1380, candidateE2_tf_lowlambda_expanded672=32.1651. Δ(candidateE2_tf_lowlambda_expanded672−CNN)=+0.9726 (▲ better), improved on 168/168 samples.
- **SSIM**: CNN=N/A, GAN=N/A, candidateE2_tf_lowlambda_expanded672=N/A. Δ(candidateE2_tf_lowlambda_expanded672−CNN)=N/A, improved on 0/0 samples.
- **Speed MAE**: CNN=0.6941, GAN=0.9026, candidateE2_tf_lowlambda_expanded672=0.6133. Δ(candidateE2_tf_lowlambda_expanded672−CNN)=-0.0807 (▲ better), improved on 168/168 samples.
- **Speed RMSE**: CNN=1.1078, GAN=1.3775, candidateE2_tf_lowlambda_expanded672=1.0170. Δ(candidateE2_tf_lowlambda_expanded672−CNN)=-0.0907 (▲ better), improved on 168/168 samples.

### Q2. Did candidateE2_tf_lowlambda_expanded672 improve scalar speed or WPD metrics?

- **WPD MAE**: CNN=231.6709, candidateE2_tf_lowlambda_expanded672=201.4855. Δ=-30.1854 (▲ better), improved on 168/168 samples.
- **WPD W1**: CNN=45.2713, candidateE2_tf_lowlambda_expanded672=16.3995. Δ=-28.8718 (▲ better), improved on 161/168 samples.
- **WPD bias abs**: CNN=35.3439, candidateE2_tf_lowlambda_expanded672=4.7166. Δ=-30.6273 (▲ better), improved on 155/168 samples.

### Q3. Did candidateE2_tf_lowlambda_expanded672 improve gradient metrics?

- **Gradient MAE**: CNN=0.3491, candidateE2_tf_lowlambda_expanded672=0.3210. Δ=-0.0281 (▲ better), improved on 168/168 samples.
- **Gradient W1**: CNN=0.2329, candidateE2_tf_lowlambda_expanded672=0.1529. Δ=-0.0800 (▲ better), improved on 163/168 samples.
- **Gradient kurtosis abs Δ**: CNN=3.7004, candidateE2_tf_lowlambda_expanded672=2.8789. Δ=-0.8215 (▲ better), improved on 86/168 samples.

### Q4. Did candidateE2_tf_lowlambda_expanded672 improve exceedance metrics, especially s>5?

- **Exceedance abs Δ s>5**: CNN=0.00424, candidateE2_tf_lowlambda_expanded672=0.00288. Δ=-0.0014 (▲ better), improved on 118/168 samples.
- **Exceedance abs Δ s>10**: CNN=0.00658, candidateE2_tf_lowlambda_expanded672=0.00328. Δ=-0.0033 (▲ better), improved on 115/168 samples.
- **Exceedance abs Δ s>15**: CNN=0.00621, candidateE2_tf_lowlambda_expanded672=0.00157. Δ=-0.0046 (▲ better), improved on 139/168 samples.
- **Exceedance abs Δ p90**: CNN=0.01025, candidateE2_tf_lowlambda_expanded672=0.00179. Δ=-0.0085 (▲ better), improved on 159/168 samples.

### Q5. Did candidateE2_tf_lowlambda_expanded672 move component-count behavior toward GT?

- **Component-count curve L1**: CNN=115.5278, candidateE2_tf_lowlambda_expanded672=87.3800. Δ=-28.1478 (▲ better), improved on 158/168 samples.
- **Comp-count abs Δ t5**: CNN=64.3810, candidateE2_tf_lowlambda_expanded672=55.9643. Δ=-8.4167 (▲ better), improved on 124/168 samples.
- **Comp-count abs Δ t10**: CNN=187.1667, candidateE2_tf_lowlambda_expanded672=167.7857. Δ=-19.3810 (▲ better), improved on 132/168 samples.
- **Comp-count abs Δ t15**: CNN=116.8095, candidateE2_tf_lowlambda_expanded672=74.7262. Δ=-42.0833 (▲ better), improved on 153/168 samples.

### Q6. Did candidateE2_tf_lowlambda_expanded672 improve PD or MT distances?

PD and MT distances for `candidateE2_tf_lowlambda_expanded672` are not available in this evaluation (TTK was not re-run on these outputs). Pre-computed values for CNN and GAN:

- **PD distance**: CNN=27.4063, GAN=20.8641, candidateE2_tf_lowlambda_expanded672=N/A (requires TTK run)
- **MT distance**: CNN=5.8678, GAN=8.3481, candidateE2_tf_lowlambda_expanded672=N/A (requires TTK run)

To compute PD/MT for `candidateE2_tf_lowlambda_expanded672`: convert outputs to VTI via `scripts/convert_phire_to_vti.py`, then run `scripts/compute_composite_tree_distance.py`.

### Q7. Did candidateE2_tf_lowlambda_expanded672 change the adjacent clusters (samples 10–13, 90–93, 162–163)?

See `adjacent_cluster_table_candidateE2_tf_lowlambda_expanded672.csv` for full detail. Summary of PSNRuv, speed MAE, and gradient MAE changes:

| Sample | Δ PSNRuv | Δ Speed MAE | Δ Grad MAE | Δ Exceed s>5 |
|---|---|---|---|---|
| 10 | 1.277 | -0.1009 | -0.0269 | -0.00190 |
| 11 | 1.129 | -0.0930 | -0.0259 | -0.00367 |
| 12 | 1.190 | -0.0946 | -0.0306 | -0.00263 |
| 13 | 1.058 | -0.0808 | -0.0251 | -0.00228 |
| 90 | 0.925 | -0.0637 | -0.0233 | 0.00203 |
| 91 | 0.980 | -0.0680 | -0.0226 | 0.00606 |
| 92 | 0.893 | -0.0731 | -0.0268 | 0.00884 |
| 93 | 0.800 | -0.0709 | -0.0282 | 0.00191 |
| 162 | 2.023 | -0.0854 | -0.0167 | -0.00259 |
| 163 | 1.854 | -0.0829 | -0.0176 | 0.00423 |

## Pairwise summary: CNN vs candidateE2_tf_lowlambda_expanded672

| Metric | Mean CNN | Mean candidateE2_tf_lowlambda_expanded672 | Δ | N improved | N worsened |
|---|---|---|---|---|---|
| psnruv | 31.1925 | 32.1651 | 0.9726 | 168 | 0 |
| ssim | N/A | N/A | N/A | 0 | 0 |
| speed_mae | 0.6941 | 0.6133 | -0.0807 | 168 | 0 |
| speed_rmse | 1.1078 | 1.0170 | -0.0907 | 168 | 0 |
| wpd_bias_abs | 35.3439 | 4.7166 | -30.6273 | 155 | 13 |
| wpd_mae | 231.6709 | 201.4855 | -30.1854 | 168 | 0 |
| wpd_w1 | 45.2713 | 16.3995 | -28.8718 | 161 | 7 |
| psd_log_l2 | 0.8335 | 0.8141 | -0.0194 | 108 | 60 |
| psd_slope_abs_delta | 0.9150 | 1.0531 | 0.1381 | 20 | 148 |
| grad_mae | 0.3491 | 0.3210 | -0.0281 | 168 | 0 |
| grad_w1 | 0.2329 | 0.1529 | -0.0800 | 163 | 5 |
| grad_kurtosis_abs_delta | 3.7004 | 2.8789 | -0.8215 | 86 | 82 |
| exceed_abs_t5 | 0.0042 | 0.0029 | -0.0014 | 118 | 50 |
| exceed_abs_t10 | 0.0066 | 0.0033 | -0.0033 | 115 | 53 |
| exceed_abs_t15 | 0.0062 | 0.0016 | -0.0046 | 139 | 29 |
| exceed_abs_p90 | 0.0103 | 0.0018 | -0.0085 | 159 | 9 |
| comp_curve_l1 | 115.5278 | 87.3800 | -28.1478 | 158 | 10 |
| pd_distance | N/A | N/A | N/A | 0 | 0 |
| mt_distance | N/A | N/A | N/A | 0 | 0 |

## Output files

- `ttk_runs_fixed/topology_finetuning/candidateE2_tf_lowlambda_expanded672_eval/all_sample_metrics_candidateE2_tf_lowlambda_expanded672.csv` — per-sample metrics for all methods
- `ttk_runs_fixed/topology_finetuning/candidateE2_tf_lowlambda_expanded672_eval/pairwise_cnn_vs_candidateE2_tf_lowlambda_expanded672.csv` — per-metric delta table
- `ttk_runs_fixed/topology_finetuning/candidateE2_tf_lowlambda_expanded672_eval/winner_counts_candidateE2_tf_lowlambda_expanded672.csv` — win counts per method
- `ttk_runs_fixed/topology_finetuning/candidateE2_tf_lowlambda_expanded672_eval/adjacent_cluster_table_candidateE2_tf_lowlambda_expanded672.csv` — adjacent-cluster detail
- `docs/topology_finetuning_candidateE2_tf_lowlambda_expanded672_eval.md` — this report

## Notes

- PSNRuv is computed on physical [u,v] arrays using per-sample dynamic data range (GT max − GT min).
- All speed-based metrics use scalar speed = sqrt(u² + v²) in physical units (m/s).
- Exceedance fractions use GT percentile thresholds (p90/p95/p99) computed per sample.
- Component counts use 8-connectivity on speed superlevel sets.
- PD/MT distances for CNN and GAN are loaded from pre-computed merged CSV; `candidateE2_tf_lowlambda_expanded672` requires a fresh TTK run.
- `candidateE2_tf_lowlambda_expanded672` was fine-tuned from the pretrained CNN checkpoint. Results may change with different lambda settings or more epochs.
