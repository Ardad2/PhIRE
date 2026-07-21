# candidateB_factorial_speed_expanded2688 fine-tuning evaluation

**Generated:** 2026-07-15

**Candidate:** `candidateB_factorial_speed_expanded2688`

**Samples evaluated:** 168

**Methods compared:** bicubic, cnn, gan, candidateB_factorial_speed_expanded2688

> **SSIM note:** SSIM was not computed because skimage could not be imported in the current NumPy environment (`ValueError('numpy.dtype size changed, may indicate binary incompatibility. Expected 96 from C header, got 88 from PyObject')`). All SSIM values are NaN.

> Note: PD/MT distances for `candidateB_factorial_speed_expanded2688` are N/A — TTK has not been re-run on these outputs. Values for CNN/GAN are loaded from the pre-computed merged CSV.

## Summary metrics table

| Metric | Bicubic | Cnn | Gan | Candidateb_factorial_speed_expanded2688 |
|---|---|---|---|---|
| PSNRuv (dB) | 32.2202 | 31.1925 | 29.1380 | 33.7894 |
| SSIM on speed | N/A | N/A | N/A | N/A |
| Speed MAE (m/s) | 0.5963 | 0.6941 | 0.9026 | 0.4943 |
| Speed RMSE (m/s) | 1.0156 | 1.1078 | 1.3775 | 0.8532 |
| WPD MAE (m³/s³) | 193.5403 | 231.6709 | 310.7328 | 158.3337 |
| WPD Wasserstein-1 | 55.5103 | 45.2713 | 85.6191 | 35.8761 |
| WPD bias abs (m³/s³) | 41.2244 | 35.3439 | 78.7236 | 26.4431 |
| Gradient MAE | 0.3696 | 0.3491 | 0.3806 | 0.3242 |
| Gradient W1 | 0.3282 | 0.2329 | 0.0564 | 0.2491 |
| Gradient kurtosis abs Δ | 6.6163 | 3.7004 | 4.2010 | 5.1312 |
| PSD log-L2 | 1.3625 | 0.8335 | 0.5139 | 1.1311 |
| PSD slope abs Δ | 1.6684 | 0.9150 | 0.9482 | 1.4548 |
| Exceedance abs Δ s>5 | 0.0095 | 0.0042 | 0.0082 | 0.0066 |
| Exceedance abs Δ s>10 | 0.0085 | 0.0066 | 0.0243 | 0.0049 |
| Exceedance abs Δ s>15 | 0.0074 | 0.0062 | 0.0096 | 0.0048 |
| Exceedance abs Δ p90 | 0.0109 | 0.0103 | 0.0123 | 0.0070 |
| Component-count curve L1 | 173.1855 | 115.5278 | 124.6567 | 142.1994 |
| PD distance (TTK) | N/A | 27.4063 | 20.8641 | N/A |
| MT distance (TTK) | N/A | 5.8678 | 8.3481 | N/A |

## 7 Key evaluation questions

### Q1. Did candidateB_factorial_speed_expanded2688 preserve CNN's direct-fidelity advantage?

- **PSNRuv**: CNN=31.1925, GAN=29.1380, candidateB_factorial_speed_expanded2688=33.7894. Δ(candidateB_factorial_speed_expanded2688−CNN)=+2.5969 (▲ better), improved on 168/168 samples.
- **SSIM**: CNN=N/A, GAN=N/A, candidateB_factorial_speed_expanded2688=N/A. Δ(candidateB_factorial_speed_expanded2688−CNN)=N/A, improved on 0/0 samples.
- **Speed MAE**: CNN=0.6941, GAN=0.9026, candidateB_factorial_speed_expanded2688=0.4943. Δ(candidateB_factorial_speed_expanded2688−CNN)=-0.1998 (▲ better), improved on 168/168 samples.
- **Speed RMSE**: CNN=1.1078, GAN=1.3775, candidateB_factorial_speed_expanded2688=0.8532. Δ(candidateB_factorial_speed_expanded2688−CNN)=-0.2546 (▲ better), improved on 168/168 samples.

### Q2. Did candidateB_factorial_speed_expanded2688 improve scalar speed or WPD metrics?

- **WPD MAE**: CNN=231.6709, candidateB_factorial_speed_expanded2688=158.3337. Δ=-73.3372 (▲ better), improved on 168/168 samples.
- **WPD W1**: CNN=45.2713, candidateB_factorial_speed_expanded2688=35.8761. Δ=-9.3951 (▲ better), improved on 120/168 samples.
- **WPD bias abs**: CNN=35.3439, candidateB_factorial_speed_expanded2688=26.4431. Δ=-8.9008 (▲ better), improved on 113/168 samples.

### Q3. Did candidateB_factorial_speed_expanded2688 improve gradient metrics?

- **Gradient MAE**: CNN=0.3491, candidateB_factorial_speed_expanded2688=0.3242. Δ=-0.0248 (▲ better), improved on 168/168 samples.
- **Gradient W1**: CNN=0.2329, candidateB_factorial_speed_expanded2688=0.2491. Δ=+0.0162 (▼ worse), improved on 35/168 samples.
- **Gradient kurtosis abs Δ**: CNN=3.7004, candidateB_factorial_speed_expanded2688=5.1312. Δ=+1.4308 (▼ worse), improved on 59/168 samples.

### Q4. Did candidateB_factorial_speed_expanded2688 improve exceedance metrics, especially s>5?

- **Exceedance abs Δ s>5**: CNN=0.00424, candidateB_factorial_speed_expanded2688=0.00656. Δ=+0.0023 (▼ worse), improved on 21/168 samples.
- **Exceedance abs Δ s>10**: CNN=0.00658, candidateB_factorial_speed_expanded2688=0.00489. Δ=-0.0017 (▲ better), improved on 115/168 samples.
- **Exceedance abs Δ s>15**: CNN=0.00621, candidateB_factorial_speed_expanded2688=0.00475. Δ=-0.0015 (▲ better), improved on 112/168 samples.
- **Exceedance abs Δ p90**: CNN=0.01025, candidateB_factorial_speed_expanded2688=0.00695. Δ=-0.0033 (▲ better), improved on 131/168 samples.

### Q5. Did candidateB_factorial_speed_expanded2688 move component-count behavior toward GT?

- **Component-count curve L1**: CNN=115.5278, candidateB_factorial_speed_expanded2688=142.1994. Δ=+26.6716 (▼ worse), improved on 6/168 samples.
- **Comp-count abs Δ t5**: CNN=64.3810, candidateB_factorial_speed_expanded2688=69.2143. Δ=+4.8333 (▼ worse), improved on 45/168 samples.
- **Comp-count abs Δ t10**: CNN=187.1667, candidateB_factorial_speed_expanded2688=230.0060. Δ=+42.8393 (▼ worse), improved on 5/168 samples.
- **Comp-count abs Δ t15**: CNN=116.8095, candidateB_factorial_speed_expanded2688=143.5714. Δ=+26.7619 (▼ worse), improved on 24/168 samples.

### Q6. Did candidateB_factorial_speed_expanded2688 improve PD or MT distances?

PD and MT distances for `candidateB_factorial_speed_expanded2688` are not available in this evaluation (TTK was not re-run on these outputs). Pre-computed values for CNN and GAN:

- **PD distance**: CNN=27.4063, GAN=20.8641, candidateB_factorial_speed_expanded2688=N/A (requires TTK run)
- **MT distance**: CNN=5.8678, GAN=8.3481, candidateB_factorial_speed_expanded2688=N/A (requires TTK run)

To compute PD/MT for `candidateB_factorial_speed_expanded2688`: convert outputs to VTI via `scripts/convert_phire_to_vti.py`, then run `scripts/compute_composite_tree_distance.py`.

### Q7. Did candidateB_factorial_speed_expanded2688 change the adjacent clusters (samples 10–13, 90–93, 162–163)?

See `adjacent_cluster_table_candidateB_factorial_speed_expanded2688.csv` for full detail. Summary of PSNRuv, speed MAE, and gradient MAE changes:

| Sample | Δ PSNRuv | Δ Speed MAE | Δ Grad MAE | Δ Exceed s>5 |
|---|---|---|---|---|
| 10 | 2.773 | -0.2220 | -0.0243 | 0.00483 |
| 11 | 2.665 | -0.2187 | -0.0258 | 0.00388 |
| 12 | 2.698 | -0.2183 | -0.0306 | 0.00375 |
| 13 | 2.624 | -0.2110 | -0.0284 | 0.00131 |
| 90 | 2.394 | -0.1547 | -0.0204 | -0.00123 |
| 91 | 2.423 | -0.1596 | -0.0200 | 0.00214 |
| 92 | 2.443 | -0.1727 | -0.0200 | 0.00727 |
| 93 | 2.427 | -0.1849 | -0.0175 | 0.00619 |
| 162 | 3.907 | -0.1598 | -0.0256 | -0.00567 |
| 163 | 3.848 | -0.1603 | -0.0277 | -0.00145 |

## Pairwise summary: CNN vs candidateB_factorial_speed_expanded2688

| Metric | Mean CNN | Mean candidateB_factorial_speed_expanded2688 | Δ | N improved | N worsened |
|---|---|---|---|---|---|
| psnruv | 31.1925 | 33.7894 | 2.5969 | 168 | 0 |
| ssim | N/A | N/A | N/A | 0 | 0 |
| speed_mae | 0.6941 | 0.4943 | -0.1998 | 168 | 0 |
| speed_rmse | 1.1078 | 0.8532 | -0.2546 | 168 | 0 |
| wpd_bias_abs | 35.3439 | 26.4431 | -8.9008 | 113 | 55 |
| wpd_mae | 231.6709 | 158.3337 | -73.3372 | 168 | 0 |
| wpd_w1 | 45.2713 | 35.8761 | -9.3951 | 120 | 48 |
| psd_log_l2 | 0.8335 | 1.1311 | 0.2975 | 0 | 168 |
| psd_slope_abs_delta | 0.9150 | 1.4548 | 0.5397 | 0 | 168 |
| grad_mae | 0.3491 | 0.3242 | -0.0248 | 168 | 0 |
| grad_w1 | 0.2329 | 0.2491 | 0.0162 | 35 | 133 |
| grad_kurtosis_abs_delta | 3.7004 | 5.1312 | 1.4308 | 59 | 109 |
| exceed_abs_t5 | 0.0042 | 0.0066 | 0.0023 | 21 | 147 |
| exceed_abs_t10 | 0.0066 | 0.0049 | -0.0017 | 115 | 53 |
| exceed_abs_t15 | 0.0062 | 0.0048 | -0.0015 | 112 | 56 |
| exceed_abs_p90 | 0.0103 | 0.0070 | -0.0033 | 131 | 37 |
| comp_curve_l1 | 115.5278 | 142.1994 | 26.6716 | 6 | 161 |
| pd_distance | N/A | N/A | N/A | 0 | 0 |
| mt_distance | N/A | N/A | N/A | 0 | 0 |

## Output files

- `ttk_runs_fixed/topology_finetuning/candidateB_factorial_speed_expanded2688_eval/all_sample_metrics_candidateB_factorial_speed_expanded2688.csv` — per-sample metrics for all methods
- `ttk_runs_fixed/topology_finetuning/candidateB_factorial_speed_expanded2688_eval/pairwise_cnn_vs_candidateB_factorial_speed_expanded2688.csv` — per-metric delta table
- `ttk_runs_fixed/topology_finetuning/candidateB_factorial_speed_expanded2688_eval/winner_counts_candidateB_factorial_speed_expanded2688.csv` — win counts per method
- `ttk_runs_fixed/topology_finetuning/candidateB_factorial_speed_expanded2688_eval/adjacent_cluster_table_candidateB_factorial_speed_expanded2688.csv` — adjacent-cluster detail
- `docs/topology_finetuning_candidateB_factorial_speed_expanded2688_eval.md` — this report

## Notes

- PSNRuv is computed on physical [u,v] arrays using per-sample dynamic data range (GT max − GT min).
- All speed-based metrics use scalar speed = sqrt(u² + v²) in physical units (m/s).
- Exceedance fractions use GT percentile thresholds (p90/p95/p99) computed per sample.
- Component counts use 8-connectivity on speed superlevel sets.
- PD/MT distances for CNN and GAN are loaded from pre-computed merged CSV; `candidateB_factorial_speed_expanded2688` requires a fresh TTK run.
- `candidateB_factorial_speed_expanded2688` was fine-tuned from the pretrained CNN checkpoint. Results may change with different lambda settings or more epochs.
