# candidateUV_plus_E2_tf_lowlambda_expanded2688 fine-tuning evaluation

**Generated:** 2026-07-11

**Candidate:** `candidateUV_plus_E2_tf_lowlambda_expanded2688`

**Samples evaluated:** 168

**Methods compared:** bicubic, cnn, gan, candidateUV_plus_E2_tf_lowlambda_expanded2688

> **SSIM note:** SSIM was not computed because skimage could not be imported in the current NumPy environment (`ValueError('numpy.dtype size changed, may indicate binary incompatibility. Expected 96 from C header, got 88 from PyObject')`). All SSIM values are NaN.

> Note: PD/MT distances for `candidateUV_plus_E2_tf_lowlambda_expanded2688` are N/A — TTK has not been re-run on these outputs. Values for CNN/GAN are loaded from the pre-computed merged CSV.

## Summary metrics table

| Metric | Bicubic | Cnn | Gan | Candidateuv_plus_e2_tf_lowlambda_expanded2688 |
|---|---|---|---|---|
| PSNRuv (dB) | 32.2202 | 31.1925 | 29.1380 | 32.5752 |
| SSIM on speed | N/A | N/A | N/A | N/A |
| Speed MAE (m/s) | 0.5963 | 0.6941 | 0.9026 | 0.5789 |
| Speed RMSE (m/s) | 1.0156 | 1.1078 | 1.3775 | 0.9773 |
| WPD MAE (m³/s³) | 193.5403 | 231.6709 | 310.7328 | 187.6210 |
| WPD Wasserstein-1 | 55.5103 | 45.2713 | 85.6191 | 21.2812 |
| WPD bias abs (m³/s³) | 41.2244 | 35.3439 | 78.7236 | 17.1101 |
| Gradient MAE | 0.3696 | 0.3491 | 0.3806 | 0.3181 |
| Gradient W1 | 0.3282 | 0.2329 | 0.0564 | 0.1694 |
| Gradient kurtosis abs Δ | 6.6163 | 3.7004 | 4.2010 | 3.2223 |
| PSD log-L2 | 1.3625 | 0.8335 | 0.5139 | 0.8378 |
| PSD slope abs Δ | 1.6684 | 0.9150 | 0.9482 | 1.1206 |
| Exceedance abs Δ s>5 | 0.0095 | 0.0042 | 0.0082 | 0.0025 |
| Exceedance abs Δ s>10 | 0.0085 | 0.0066 | 0.0243 | 0.0035 |
| Exceedance abs Δ s>15 | 0.0074 | 0.0062 | 0.0096 | 0.0025 |
| Exceedance abs Δ p90 | 0.0109 | 0.0103 | 0.0123 | 0.0037 |
| Component-count curve L1 | 173.1855 | 115.5278 | 124.6567 | 97.1538 |
| PD distance (TTK) | N/A | 27.4063 | 20.8641 | N/A |
| MT distance (TTK) | N/A | 5.8678 | 8.3481 | N/A |

## 7 Key evaluation questions

### Q1. Did candidateUV_plus_E2_tf_lowlambda_expanded2688 preserve CNN's direct-fidelity advantage?

- **PSNRuv**: CNN=31.1925, GAN=29.1380, candidateUV_plus_E2_tf_lowlambda_expanded2688=32.5752. Δ(candidateUV_plus_E2_tf_lowlambda_expanded2688−CNN)=+1.3827 (▲ better), improved on 168/168 samples.
- **SSIM**: CNN=N/A, GAN=N/A, candidateUV_plus_E2_tf_lowlambda_expanded2688=N/A. Δ(candidateUV_plus_E2_tf_lowlambda_expanded2688−CNN)=N/A, improved on 0/0 samples.
- **Speed MAE**: CNN=0.6941, GAN=0.9026, candidateUV_plus_E2_tf_lowlambda_expanded2688=0.5789. Δ(candidateUV_plus_E2_tf_lowlambda_expanded2688−CNN)=-0.1152 (▲ better), improved on 168/168 samples.
- **Speed RMSE**: CNN=1.1078, GAN=1.3775, candidateUV_plus_E2_tf_lowlambda_expanded2688=0.9773. Δ(candidateUV_plus_E2_tf_lowlambda_expanded2688−CNN)=-0.1304 (▲ better), improved on 168/168 samples.

### Q2. Did candidateUV_plus_E2_tf_lowlambda_expanded2688 improve scalar speed or WPD metrics?

- **WPD MAE**: CNN=231.6709, candidateUV_plus_E2_tf_lowlambda_expanded2688=187.6210. Δ=-44.0499 (▲ better), improved on 168/168 samples.
- **WPD W1**: CNN=45.2713, candidateUV_plus_E2_tf_lowlambda_expanded2688=21.2812. Δ=-23.9900 (▲ better), improved on 152/168 samples.
- **WPD bias abs**: CNN=35.3439, candidateUV_plus_E2_tf_lowlambda_expanded2688=17.1101. Δ=-18.2338 (▲ better), improved on 143/168 samples.

### Q3. Did candidateUV_plus_E2_tf_lowlambda_expanded2688 improve gradient metrics?

- **Gradient MAE**: CNN=0.3491, candidateUV_plus_E2_tf_lowlambda_expanded2688=0.3181. Δ=-0.0310 (▲ better), improved on 168/168 samples.
- **Gradient W1**: CNN=0.2329, candidateUV_plus_E2_tf_lowlambda_expanded2688=0.1694. Δ=-0.0635 (▲ better), improved on 159/168 samples.
- **Gradient kurtosis abs Δ**: CNN=3.7004, candidateUV_plus_E2_tf_lowlambda_expanded2688=3.2223. Δ=-0.4780 (▲ better), improved on 80/168 samples.

### Q4. Did candidateUV_plus_E2_tf_lowlambda_expanded2688 improve exceedance metrics, especially s>5?

- **Exceedance abs Δ s>5**: CNN=0.00424, candidateUV_plus_E2_tf_lowlambda_expanded2688=0.00254. Δ=-0.0017 (▲ better), improved on 121/168 samples.
- **Exceedance abs Δ s>10**: CNN=0.00658, candidateUV_plus_E2_tf_lowlambda_expanded2688=0.00351. Δ=-0.0031 (▲ better), improved on 132/168 samples.
- **Exceedance abs Δ s>15**: CNN=0.00621, candidateUV_plus_E2_tf_lowlambda_expanded2688=0.00249. Δ=-0.0037 (▲ better), improved on 141/168 samples.
- **Exceedance abs Δ p90**: CNN=0.01025, candidateUV_plus_E2_tf_lowlambda_expanded2688=0.00373. Δ=-0.0065 (▲ better), improved on 144/168 samples.

### Q5. Did candidateUV_plus_E2_tf_lowlambda_expanded2688 move component-count behavior toward GT?

- **Component-count curve L1**: CNN=115.5278, candidateUV_plus_E2_tf_lowlambda_expanded2688=97.1538. Δ=-18.3740 (▲ better), improved on 154/168 samples.
- **Comp-count abs Δ t5**: CNN=64.3810, candidateUV_plus_E2_tf_lowlambda_expanded2688=58.2321. Δ=-6.1488 (▲ better), improved on 117/168 samples.
- **Comp-count abs Δ t10**: CNN=187.1667, candidateUV_plus_E2_tf_lowlambda_expanded2688=178.1131. Δ=-9.0536 (▲ better), improved on 107/168 samples.
- **Comp-count abs Δ t15**: CNN=116.8095, candidateUV_plus_E2_tf_lowlambda_expanded2688=89.8571. Δ=-26.9524 (▲ better), improved on 143/168 samples.

### Q6. Did candidateUV_plus_E2_tf_lowlambda_expanded2688 improve PD or MT distances?

PD and MT distances for `candidateUV_plus_E2_tf_lowlambda_expanded2688` are not available in this evaluation (TTK was not re-run on these outputs). Pre-computed values for CNN and GAN:

- **PD distance**: CNN=27.4063, GAN=20.8641, candidateUV_plus_E2_tf_lowlambda_expanded2688=N/A (requires TTK run)
- **MT distance**: CNN=5.8678, GAN=8.3481, candidateUV_plus_E2_tf_lowlambda_expanded2688=N/A (requires TTK run)

To compute PD/MT for `candidateUV_plus_E2_tf_lowlambda_expanded2688`: convert outputs to VTI via `scripts/convert_phire_to_vti.py`, then run `scripts/compute_composite_tree_distance.py`.

### Q7. Did candidateUV_plus_E2_tf_lowlambda_expanded2688 change the adjacent clusters (samples 10–13, 90–93, 162–163)?

See `adjacent_cluster_table_candidateUV_plus_E2_tf_lowlambda_expanded2688.csv` for full detail. Summary of PSNRuv, speed MAE, and gradient MAE changes:

| Sample | Δ PSNRuv | Δ Speed MAE | Δ Grad MAE | Δ Exceed s>5 |
|---|---|---|---|---|
| 10 | 1.594 | -0.1352 | -0.0308 | -0.00260 |
| 11 | 1.509 | -0.1316 | -0.0317 | -0.00163 |
| 12 | 1.536 | -0.1316 | -0.0350 | -0.00139 |
| 13 | 1.409 | -0.1185 | -0.0301 | -0.00179 |
| 90 | 1.362 | -0.0916 | -0.0238 | 0.00063 |
| 91 | 1.396 | -0.0972 | -0.0240 | 0.00536 |
| 92 | 1.339 | -0.1035 | -0.0285 | 0.00640 |
| 93 | 1.181 | -0.1012 | -0.0292 | -0.00158 |
| 162 | 2.420 | -0.1103 | -0.0212 | -0.00212 |
| 163 | 2.235 | -0.1070 | -0.0219 | 0.00220 |

## Pairwise summary: CNN vs candidateUV_plus_E2_tf_lowlambda_expanded2688

| Metric | Mean CNN | Mean candidateUV_plus_E2_tf_lowlambda_expanded2688 | Δ | N improved | N worsened |
|---|---|---|---|---|---|
| psnruv | 31.1925 | 32.5752 | 1.3827 | 168 | 0 |
| ssim | N/A | N/A | N/A | 0 | 0 |
| speed_mae | 0.6941 | 0.5789 | -0.1152 | 168 | 0 |
| speed_rmse | 1.1078 | 0.9773 | -0.1304 | 168 | 0 |
| wpd_bias_abs | 35.3439 | 17.1101 | -18.2338 | 143 | 25 |
| wpd_mae | 231.6709 | 187.6210 | -44.0499 | 168 | 0 |
| wpd_w1 | 45.2713 | 21.2812 | -23.9900 | 152 | 16 |
| psd_log_l2 | 0.8335 | 0.8378 | 0.0043 | 95 | 73 |
| psd_slope_abs_delta | 0.9150 | 1.1206 | 0.2056 | 3 | 165 |
| grad_mae | 0.3491 | 0.3181 | -0.0310 | 168 | 0 |
| grad_w1 | 0.2329 | 0.1694 | -0.0635 | 159 | 9 |
| grad_kurtosis_abs_delta | 3.7004 | 3.2223 | -0.4780 | 80 | 88 |
| exceed_abs_t5 | 0.0042 | 0.0025 | -0.0017 | 121 | 47 |
| exceed_abs_t10 | 0.0066 | 0.0035 | -0.0031 | 132 | 36 |
| exceed_abs_t15 | 0.0062 | 0.0025 | -0.0037 | 141 | 27 |
| exceed_abs_p90 | 0.0103 | 0.0037 | -0.0065 | 144 | 24 |
| comp_curve_l1 | 115.5278 | 97.1538 | -18.3740 | 154 | 14 |
| pd_distance | N/A | N/A | N/A | 0 | 0 |
| mt_distance | N/A | N/A | N/A | 0 | 0 |

## Output files

- `ttk_runs_fixed/topology_finetuning/candidateUV_plus_E2_tf_lowlambda_expanded2688_eval/all_sample_metrics_candidateUV_plus_E2_tf_lowlambda_expanded2688.csv` — per-sample metrics for all methods
- `ttk_runs_fixed/topology_finetuning/candidateUV_plus_E2_tf_lowlambda_expanded2688_eval/pairwise_cnn_vs_candidateUV_plus_E2_tf_lowlambda_expanded2688.csv` — per-metric delta table
- `ttk_runs_fixed/topology_finetuning/candidateUV_plus_E2_tf_lowlambda_expanded2688_eval/winner_counts_candidateUV_plus_E2_tf_lowlambda_expanded2688.csv` — win counts per method
- `ttk_runs_fixed/topology_finetuning/candidateUV_plus_E2_tf_lowlambda_expanded2688_eval/adjacent_cluster_table_candidateUV_plus_E2_tf_lowlambda_expanded2688.csv` — adjacent-cluster detail
- `docs/topology_finetuning_candidateUV_plus_E2_tf_lowlambda_expanded2688_eval.md` — this report

## Notes

- PSNRuv is computed on physical [u,v] arrays using per-sample dynamic data range (GT max − GT min).
- All speed-based metrics use scalar speed = sqrt(u² + v²) in physical units (m/s).
- Exceedance fractions use GT percentile thresholds (p90/p95/p99) computed per sample.
- Component counts use 8-connectivity on speed superlevel sets.
- PD/MT distances for CNN and GAN are loaded from pre-computed merged CSV; `candidateUV_plus_E2_tf_lowlambda_expanded2688` requires a fresh TTK run.
- `candidateUV_plus_E2_tf_lowlambda_expanded2688` was fine-tuned from the pretrained CNN checkpoint. Results may change with different lambda settings or more epochs.
