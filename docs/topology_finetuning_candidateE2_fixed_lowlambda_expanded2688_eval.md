# candidateE2_fixed_lowlambda_expanded2688 fine-tuning evaluation

**Generated:** 2026-07-08

**Candidate:** `candidateE2_fixed_lowlambda_expanded2688`

**Samples evaluated:** 168

**Methods compared:** bicubic, cnn, gan, candidateE2_fixed_lowlambda_expanded2688

> **SSIM note:** SSIM was not computed because skimage could not be imported in the current NumPy environment (`ValueError('numpy.dtype size changed, may indicate binary incompatibility. Expected 96 from C header, got 88 from PyObject')`). All SSIM values are NaN.

> Note: PD/MT distances for `candidateE2_fixed_lowlambda_expanded2688` are N/A — TTK has not been re-run on these outputs. Values for CNN/GAN are loaded from the pre-computed merged CSV.

## Summary metrics table

| Metric | Bicubic | Cnn | Gan | Candidatee2_fixed_lowlambda_expanded2688 |
|---|---|---|---|---|
| PSNRuv (dB) | 32.2202 | 31.1925 | 29.1380 | 31.0929 |
| SSIM on speed | N/A | N/A | N/A | N/A |
| Speed MAE (m/s) | 0.5963 | 0.6941 | 0.9026 | 0.7148 |
| Speed RMSE (m/s) | 1.0156 | 1.1078 | 1.3775 | 1.1341 |
| WPD MAE (m³/s³) | 193.5403 | 231.6709 | 310.7328 | 239.5881 |
| WPD Wasserstein-1 | 55.5103 | 45.2713 | 85.6191 | 32.7018 |
| WPD bias abs (m³/s³) | 41.2244 | 35.3439 | 78.7236 | 20.0523 |
| Gradient MAE | 0.3696 | 0.3491 | 0.3806 | 0.3413 |
| Gradient W1 | 0.3282 | 0.2329 | 0.0564 | 0.1813 |
| Gradient kurtosis abs Δ | 6.6163 | 3.7004 | 4.2010 | 3.0188 |
| PSD log-L2 | 1.3625 | 0.8335 | 0.5139 | 0.7484 |
| PSD slope abs Δ | 1.6684 | 0.9150 | 0.9482 | 0.7654 |
| Exceedance abs Δ s>5 | 0.0095 | 0.0042 | 0.0082 | 0.0041 |
| Exceedance abs Δ s>10 | 0.0085 | 0.0066 | 0.0243 | 0.0070 |
| Exceedance abs Δ s>15 | 0.0074 | 0.0062 | 0.0096 | 0.0046 |
| Exceedance abs Δ p90 | 0.0109 | 0.0103 | 0.0123 | 0.0066 |
| Component-count curve L1 | 173.1855 | 115.5278 | 124.6567 | 98.9534 |
| PD distance (TTK) | N/A | 27.4063 | 20.8641 | N/A |
| MT distance (TTK) | N/A | 5.8678 | 8.3481 | N/A |

## 7 Key evaluation questions

### Q1. Did candidateE2_fixed_lowlambda_expanded2688 preserve CNN's direct-fidelity advantage?

- **PSNRuv**: CNN=31.1925, GAN=29.1380, candidateE2_fixed_lowlambda_expanded2688=31.0929. Δ(candidateE2_fixed_lowlambda_expanded2688−CNN)=-0.0996 (▼ worse), improved on 31/168 samples.
- **SSIM**: CNN=N/A, GAN=N/A, candidateE2_fixed_lowlambda_expanded2688=N/A. Δ(candidateE2_fixed_lowlambda_expanded2688−CNN)=N/A, improved on 0/0 samples.
- **Speed MAE**: CNN=0.6941, GAN=0.9026, candidateE2_fixed_lowlambda_expanded2688=0.7148. Δ(candidateE2_fixed_lowlambda_expanded2688−CNN)=+0.0208 (▼ worse), improved on 21/168 samples.
- **Speed RMSE**: CNN=1.1078, GAN=1.3775, candidateE2_fixed_lowlambda_expanded2688=1.1341. Δ(candidateE2_fixed_lowlambda_expanded2688−CNN)=+0.0264 (▼ worse), improved on 22/168 samples.

### Q2. Did candidateE2_fixed_lowlambda_expanded2688 improve scalar speed or WPD metrics?

- **WPD MAE**: CNN=231.6709, candidateE2_fixed_lowlambda_expanded2688=239.5881. Δ=+7.9172 (▼ worse), improved on 20/168 samples.
- **WPD W1**: CNN=45.2713, candidateE2_fixed_lowlambda_expanded2688=32.7018. Δ=-12.5695 (▲ better), improved on 121/168 samples.
- **WPD bias abs**: CNN=35.3439, candidateE2_fixed_lowlambda_expanded2688=20.0523. Δ=-15.2916 (▲ better), improved on 123/168 samples.

### Q3. Did candidateE2_fixed_lowlambda_expanded2688 improve gradient metrics?

- **Gradient MAE**: CNN=0.3491, candidateE2_fixed_lowlambda_expanded2688=0.3413. Δ=-0.0078 (▲ better), improved on 164/168 samples.
- **Gradient W1**: CNN=0.2329, candidateE2_fixed_lowlambda_expanded2688=0.1813. Δ=-0.0517 (▲ better), improved on 154/168 samples.
- **Gradient kurtosis abs Δ**: CNN=3.7004, candidateE2_fixed_lowlambda_expanded2688=3.0188. Δ=-0.6816 (▲ better), improved on 88/168 samples.

### Q4. Did candidateE2_fixed_lowlambda_expanded2688 improve exceedance metrics, especially s>5?

- **Exceedance abs Δ s>5**: CNN=0.00424, candidateE2_fixed_lowlambda_expanded2688=0.00408. Δ=-0.0002 (▲ better), improved on 107/168 samples.
- **Exceedance abs Δ s>10**: CNN=0.00658, candidateE2_fixed_lowlambda_expanded2688=0.00702. Δ=+0.0004 (▼ worse), improved on 75/168 samples.
- **Exceedance abs Δ s>15**: CNN=0.00621, candidateE2_fixed_lowlambda_expanded2688=0.00464. Δ=-0.0016 (▲ better), improved on 111/168 samples.
- **Exceedance abs Δ p90**: CNN=0.01025, candidateE2_fixed_lowlambda_expanded2688=0.00659. Δ=-0.0037 (▲ better), improved on 121/168 samples.

### Q5. Did candidateE2_fixed_lowlambda_expanded2688 move component-count behavior toward GT?

- **Component-count curve L1**: CNN=115.5278, candidateE2_fixed_lowlambda_expanded2688=98.9534. Δ=-16.5744 (▲ better), improved on 152/168 samples.
- **Comp-count abs Δ t5**: CNN=64.3810, candidateE2_fixed_lowlambda_expanded2688=64.0060. Δ=-0.3750 (▲ better), improved on 82/168 samples.
- **Comp-count abs Δ t10**: CNN=187.1667, candidateE2_fixed_lowlambda_expanded2688=176.2798. Δ=-10.8869 (▲ better), improved on 118/168 samples.
- **Comp-count abs Δ t15**: CNN=116.8095, candidateE2_fixed_lowlambda_expanded2688=92.9940. Δ=-23.8155 (▲ better), improved on 144/168 samples.

### Q6. Did candidateE2_fixed_lowlambda_expanded2688 improve PD or MT distances?

PD and MT distances for `candidateE2_fixed_lowlambda_expanded2688` are not available in this evaluation (TTK was not re-run on these outputs). Pre-computed values for CNN and GAN:

- **PD distance**: CNN=27.4063, GAN=20.8641, candidateE2_fixed_lowlambda_expanded2688=N/A (requires TTK run)
- **MT distance**: CNN=5.8678, GAN=8.3481, candidateE2_fixed_lowlambda_expanded2688=N/A (requires TTK run)

To compute PD/MT for `candidateE2_fixed_lowlambda_expanded2688`: convert outputs to VTI via `scripts/convert_phire_to_vti.py`, then run `scripts/compute_composite_tree_distance.py`.

### Q7. Did candidateE2_fixed_lowlambda_expanded2688 change the adjacent clusters (samples 10–13, 90–93, 162–163)?

See `adjacent_cluster_table_candidateE2_fixed_lowlambda_expanded2688.csv` for full detail. Summary of PSNRuv, speed MAE, and gradient MAE changes:

| Sample | Δ PSNRuv | Δ Speed MAE | Δ Grad MAE | Δ Exceed s>5 |
|---|---|---|---|---|
| 10 | 0.093 | -0.0028 | -0.0072 | -0.00125 |
| 11 | 0.074 | -0.0017 | -0.0063 | -0.00146 |
| 12 | 0.065 | -0.0018 | -0.0069 | -0.00043 |
| 13 | 0.005 | 0.0037 | -0.0058 | -0.00134 |
| 90 | -0.120 | 0.0184 | -0.0069 | 0.00636 |
| 91 | -0.136 | 0.0179 | -0.0064 | 0.00650 |
| 92 | -0.133 | 0.0172 | -0.0070 | 0.00332 |
| 93 | -0.120 | 0.0146 | -0.0094 | -0.00006 |
| 162 | 0.325 | -0.0104 | -0.0049 | -0.00100 |
| 163 | 0.307 | -0.0119 | -0.0049 | 0.00014 |

## Pairwise summary: CNN vs candidateE2_fixed_lowlambda_expanded2688

| Metric | Mean CNN | Mean candidateE2_fixed_lowlambda_expanded2688 | Δ | N improved | N worsened |
|---|---|---|---|---|---|
| psnruv | 31.1925 | 31.0929 | -0.0996 | 31 | 137 |
| ssim | N/A | N/A | N/A | 0 | 0 |
| speed_mae | 0.6941 | 0.7148 | 0.0208 | 21 | 147 |
| speed_rmse | 1.1078 | 1.1341 | 0.0264 | 22 | 146 |
| wpd_bias_abs | 35.3439 | 20.0523 | -15.2916 | 123 | 45 |
| wpd_mae | 231.6709 | 239.5881 | 7.9172 | 20 | 148 |
| wpd_w1 | 45.2713 | 32.7018 | -12.5695 | 121 | 47 |
| psd_log_l2 | 0.8335 | 0.7484 | -0.0852 | 146 | 22 |
| psd_slope_abs_delta | 0.9150 | 0.7654 | -0.1496 | 111 | 57 |
| grad_mae | 0.3491 | 0.3413 | -0.0078 | 164 | 4 |
| grad_w1 | 0.2329 | 0.1813 | -0.0517 | 154 | 14 |
| grad_kurtosis_abs_delta | 3.7004 | 3.0188 | -0.6816 | 88 | 80 |
| exceed_abs_t5 | 0.0042 | 0.0041 | -0.0002 | 107 | 61 |
| exceed_abs_t10 | 0.0066 | 0.0070 | 0.0004 | 75 | 93 |
| exceed_abs_t15 | 0.0062 | 0.0046 | -0.0016 | 111 | 57 |
| exceed_abs_p90 | 0.0103 | 0.0066 | -0.0037 | 121 | 47 |
| comp_curve_l1 | 115.5278 | 98.9534 | -16.5744 | 152 | 16 |
| pd_distance | N/A | N/A | N/A | 0 | 0 |
| mt_distance | N/A | N/A | N/A | 0 | 0 |

## Output files

- `ttk_runs_fixed/topology_finetuning/candidateE2_fixed_lowlambda_expanded2688_eval/all_sample_metrics_candidateE2_fixed_lowlambda_expanded2688.csv` — per-sample metrics for all methods
- `ttk_runs_fixed/topology_finetuning/candidateE2_fixed_lowlambda_expanded2688_eval/pairwise_cnn_vs_candidateE2_fixed_lowlambda_expanded2688.csv` — per-metric delta table
- `ttk_runs_fixed/topology_finetuning/candidateE2_fixed_lowlambda_expanded2688_eval/winner_counts_candidateE2_fixed_lowlambda_expanded2688.csv` — win counts per method
- `ttk_runs_fixed/topology_finetuning/candidateE2_fixed_lowlambda_expanded2688_eval/adjacent_cluster_table_candidateE2_fixed_lowlambda_expanded2688.csv` — adjacent-cluster detail
- `docs/topology_finetuning_candidateE2_fixed_lowlambda_expanded2688_eval.md` — this report

## Notes

- PSNRuv is computed on physical [u,v] arrays using per-sample dynamic data range (GT max − GT min).
- All speed-based metrics use scalar speed = sqrt(u² + v²) in physical units (m/s).
- Exceedance fractions use GT percentile thresholds (p90/p95/p99) computed per sample.
- Component counts use 8-connectivity on speed superlevel sets.
- PD/MT distances for CNN and GAN are loaded from pre-computed merged CSV; `candidateE2_fixed_lowlambda_expanded2688` requires a fresh TTK run.
- `candidateE2_fixed_lowlambda_expanded2688` was fine-tuned from the pretrained CNN checkpoint. Results may change with different lambda settings or more epochs.
