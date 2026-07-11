# candidateUV_plus_crit_expanded2688 fine-tuning evaluation

**Generated:** 2026-07-11

**Candidate:** `candidateUV_plus_crit_expanded2688`

**Samples evaluated:** 168

**Methods compared:** bicubic, cnn, gan, candidateUV_plus_crit_expanded2688

> **SSIM note:** SSIM was not computed because skimage could not be imported in the current NumPy environment (`ValueError('numpy.dtype size changed, may indicate binary incompatibility. Expected 96 from C header, got 88 from PyObject')`). All SSIM values are NaN.

> Note: PD/MT distances for `candidateUV_plus_crit_expanded2688` are N/A — TTK has not been re-run on these outputs. Values for CNN/GAN are loaded from the pre-computed merged CSV.

## Summary metrics table

| Metric | Bicubic | Cnn | Gan | Candidateuv_plus_crit_expanded2688 |
|---|---|---|---|---|
| PSNRuv (dB) | 32.2202 | 31.1925 | 29.1380 | 33.6552 |
| SSIM on speed | N/A | N/A | N/A | N/A |
| Speed MAE (m/s) | 0.5963 | 0.6941 | 0.9026 | 0.5038 |
| Speed RMSE (m/s) | 1.0156 | 1.1078 | 1.3775 | 0.8667 |
| WPD MAE (m³/s³) | 193.5403 | 231.6709 | 310.7328 | 163.0417 |
| WPD Wasserstein-1 | 55.5103 | 45.2713 | 85.6191 | 21.8276 |
| WPD bias abs (m³/s³) | 41.2244 | 35.3439 | 78.7236 | 14.0303 |
| Gradient MAE | 0.3696 | 0.3491 | 0.3806 | 0.3184 |
| Gradient W1 | 0.3282 | 0.2329 | 0.0564 | 0.2295 |
| Gradient kurtosis abs Δ | 6.6163 | 3.7004 | 4.2010 | 6.3798 |
| PSD log-L2 | 1.3625 | 0.8335 | 0.5139 | 1.0504 |
| PSD slope abs Δ | 1.6684 | 0.9150 | 0.9482 | 1.3937 |
| Exceedance abs Δ s>5 | 0.0095 | 0.0042 | 0.0082 | 0.0077 |
| Exceedance abs Δ s>10 | 0.0085 | 0.0066 | 0.0243 | 0.0066 |
| Exceedance abs Δ s>15 | 0.0074 | 0.0062 | 0.0096 | 0.0017 |
| Exceedance abs Δ p90 | 0.0109 | 0.0103 | 0.0123 | 0.0022 |
| Component-count curve L1 | 173.1855 | 115.5278 | 124.6567 | 133.6042 |
| PD distance (TTK) | N/A | 27.4063 | 20.8641 | N/A |
| MT distance (TTK) | N/A | 5.8678 | 8.3481 | N/A |

## 7 Key evaluation questions

### Q1. Did candidateUV_plus_crit_expanded2688 preserve CNN's direct-fidelity advantage?

- **PSNRuv**: CNN=31.1925, GAN=29.1380, candidateUV_plus_crit_expanded2688=33.6552. Δ(candidateUV_plus_crit_expanded2688−CNN)=+2.4627 (▲ better), improved on 168/168 samples.
- **SSIM**: CNN=N/A, GAN=N/A, candidateUV_plus_crit_expanded2688=N/A. Δ(candidateUV_plus_crit_expanded2688−CNN)=N/A, improved on 0/0 samples.
- **Speed MAE**: CNN=0.6941, GAN=0.9026, candidateUV_plus_crit_expanded2688=0.5038. Δ(candidateUV_plus_crit_expanded2688−CNN)=-0.1903 (▲ better), improved on 168/168 samples.
- **Speed RMSE**: CNN=1.1078, GAN=1.3775, candidateUV_plus_crit_expanded2688=0.8667. Δ(candidateUV_plus_crit_expanded2688−CNN)=-0.2410 (▲ better), improved on 168/168 samples.

### Q2. Did candidateUV_plus_crit_expanded2688 improve scalar speed or WPD metrics?

- **WPD MAE**: CNN=231.6709, candidateUV_plus_crit_expanded2688=163.0417. Δ=-68.6292 (▲ better), improved on 168/168 samples.
- **WPD W1**: CNN=45.2713, candidateUV_plus_crit_expanded2688=21.8276. Δ=-23.4437 (▲ better), improved on 160/168 samples.
- **WPD bias abs**: CNN=35.3439, candidateUV_plus_crit_expanded2688=14.0303. Δ=-21.3136 (▲ better), improved on 147/168 samples.

### Q3. Did candidateUV_plus_crit_expanded2688 improve gradient metrics?

- **Gradient MAE**: CNN=0.3491, candidateUV_plus_crit_expanded2688=0.3184. Δ=-0.0306 (▲ better), improved on 168/168 samples.
- **Gradient W1**: CNN=0.2329, candidateUV_plus_crit_expanded2688=0.2295. Δ=-0.0034 (▲ better), improved on 119/168 samples.
- **Gradient kurtosis abs Δ**: CNN=3.7004, candidateUV_plus_crit_expanded2688=6.3798. Δ=+2.6794 (▼ worse), improved on 43/168 samples.

### Q4. Did candidateUV_plus_crit_expanded2688 improve exceedance metrics, especially s>5?

- **Exceedance abs Δ s>5**: CNN=0.00424, candidateUV_plus_crit_expanded2688=0.00773. Δ=+0.0035 (▼ worse), improved on 10/168 samples.
- **Exceedance abs Δ s>10**: CNN=0.00658, candidateUV_plus_crit_expanded2688=0.00655. Δ=-0.0000 (▲ better), improved on 84/168 samples.
- **Exceedance abs Δ s>15**: CNN=0.00621, candidateUV_plus_crit_expanded2688=0.00172. Δ=-0.0045 (▲ better), improved on 141/168 samples.
- **Exceedance abs Δ p90**: CNN=0.01025, candidateUV_plus_crit_expanded2688=0.00224. Δ=-0.0080 (▲ better), improved on 151/168 samples.

### Q5. Did candidateUV_plus_crit_expanded2688 move component-count behavior toward GT?

- **Component-count curve L1**: CNN=115.5278, candidateUV_plus_crit_expanded2688=133.6042. Δ=+18.0764 (▼ worse), improved on 12/168 samples.
- **Comp-count abs Δ t5**: CNN=64.3810, candidateUV_plus_crit_expanded2688=68.9464. Δ=+4.5655 (▼ worse), improved on 39/168 samples.
- **Comp-count abs Δ t10**: CNN=187.1667, candidateUV_plus_crit_expanded2688=230.0536. Δ=+42.8869 (▼ worse), improved on 3/168 samples.
- **Comp-count abs Δ t15**: CNN=116.8095, candidateUV_plus_crit_expanded2688=129.9286. Δ=+13.1190 (▼ worse), improved on 65/168 samples.

### Q6. Did candidateUV_plus_crit_expanded2688 improve PD or MT distances?

PD and MT distances for `candidateUV_plus_crit_expanded2688` are not available in this evaluation (TTK was not re-run on these outputs). Pre-computed values for CNN and GAN:

- **PD distance**: CNN=27.4063, GAN=20.8641, candidateUV_plus_crit_expanded2688=N/A (requires TTK run)
- **MT distance**: CNN=5.8678, GAN=8.3481, candidateUV_plus_crit_expanded2688=N/A (requires TTK run)

To compute PD/MT for `candidateUV_plus_crit_expanded2688`: convert outputs to VTI via `scripts/convert_phire_to_vti.py`, then run `scripts/compute_composite_tree_distance.py`.

### Q7. Did candidateUV_plus_crit_expanded2688 change the adjacent clusters (samples 10–13, 90–93, 162–163)?

See `adjacent_cluster_table_candidateUV_plus_crit_expanded2688.csv` for full detail. Summary of PSNRuv, speed MAE, and gradient MAE changes:

| Sample | Δ PSNRuv | Δ Speed MAE | Δ Grad MAE | Δ Exceed s>5 |
|---|---|---|---|---|
| 10 | 2.590 | -0.2076 | -0.0316 | 0.00571 |
| 11 | 2.501 | -0.2052 | -0.0326 | 0.00470 |
| 12 | 2.549 | -0.2061 | -0.0372 | 0.00482 |
| 13 | 2.475 | -0.1990 | -0.0347 | 0.00228 |
| 90 | 2.313 | -0.1480 | -0.0242 | 0.00186 |
| 91 | 2.334 | -0.1529 | -0.0240 | 0.00604 |
| 92 | 2.336 | -0.1652 | -0.0248 | 0.01074 |
| 93 | 2.294 | -0.1768 | -0.0236 | 0.00911 |
| 162 | 3.776 | -0.1561 | -0.0265 | -0.00400 |
| 163 | 3.724 | -0.1572 | -0.0291 | 0.00087 |

## Pairwise summary: CNN vs candidateUV_plus_crit_expanded2688

| Metric | Mean CNN | Mean candidateUV_plus_crit_expanded2688 | Δ | N improved | N worsened |
|---|---|---|---|---|---|
| psnruv | 31.1925 | 33.6552 | 2.4627 | 168 | 0 |
| ssim | N/A | N/A | N/A | 0 | 0 |
| speed_mae | 0.6941 | 0.5038 | -0.1903 | 168 | 0 |
| speed_rmse | 1.1078 | 0.8667 | -0.2410 | 168 | 0 |
| wpd_bias_abs | 35.3439 | 14.0303 | -21.3136 | 147 | 21 |
| wpd_mae | 231.6709 | 163.0417 | -68.6292 | 168 | 0 |
| wpd_w1 | 45.2713 | 21.8276 | -23.4437 | 160 | 8 |
| psd_log_l2 | 0.8335 | 1.0504 | 0.2169 | 0 | 168 |
| psd_slope_abs_delta | 0.9150 | 1.3937 | 0.4786 | 0 | 168 |
| grad_mae | 0.3491 | 0.3184 | -0.0306 | 168 | 0 |
| grad_w1 | 0.2329 | 0.2295 | -0.0034 | 119 | 49 |
| grad_kurtosis_abs_delta | 3.7004 | 6.3798 | 2.6794 | 43 | 125 |
| exceed_abs_t5 | 0.0042 | 0.0077 | 0.0035 | 10 | 158 |
| exceed_abs_t10 | 0.0066 | 0.0066 | -0.0000 | 84 | 84 |
| exceed_abs_t15 | 0.0062 | 0.0017 | -0.0045 | 141 | 27 |
| exceed_abs_p90 | 0.0103 | 0.0022 | -0.0080 | 151 | 17 |
| comp_curve_l1 | 115.5278 | 133.6042 | 18.0764 | 12 | 154 |
| pd_distance | N/A | N/A | N/A | 0 | 0 |
| mt_distance | N/A | N/A | N/A | 0 | 0 |

## Output files

- `ttk_runs_fixed/topology_finetuning/candidateUV_plus_crit_expanded2688_eval/all_sample_metrics_candidateUV_plus_crit_expanded2688.csv` — per-sample metrics for all methods
- `ttk_runs_fixed/topology_finetuning/candidateUV_plus_crit_expanded2688_eval/pairwise_cnn_vs_candidateUV_plus_crit_expanded2688.csv` — per-metric delta table
- `ttk_runs_fixed/topology_finetuning/candidateUV_plus_crit_expanded2688_eval/winner_counts_candidateUV_plus_crit_expanded2688.csv` — win counts per method
- `ttk_runs_fixed/topology_finetuning/candidateUV_plus_crit_expanded2688_eval/adjacent_cluster_table_candidateUV_plus_crit_expanded2688.csv` — adjacent-cluster detail
- `docs/topology_finetuning_candidateUV_plus_crit_expanded2688_eval.md` — this report

## Notes

- PSNRuv is computed on physical [u,v] arrays using per-sample dynamic data range (GT max − GT min).
- All speed-based metrics use scalar speed = sqrt(u² + v²) in physical units (m/s).
- Exceedance fractions use GT percentile thresholds (p90/p95/p99) computed per sample.
- Component counts use 8-connectivity on speed superlevel sets.
- PD/MT distances for CNN and GAN are loaded from pre-computed merged CSV; `candidateUV_plus_crit_expanded2688` requires a fresh TTK run.
- `candidateUV_plus_crit_expanded2688` was fine-tuned from the pretrained CNN checkpoint. Results may change with different lambda settings or more epochs.
