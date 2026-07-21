# candidateUV_expanded2688 fine-tuning evaluation

**Generated:** 2026-05-29

**Candidate:** `candidateUV_expanded2688`

**Samples evaluated:** 168

**Methods compared:** bicubic, cnn, gan, candidateUV_expanded2688

> **SSIM note:** SSIM was not computed because skimage could not be imported in the current NumPy environment (`ValueError('numpy.dtype size changed, may indicate binary incompatibility. Expected 96 from C header, got 88 from PyObject')`). All SSIM values are NaN.

> Note: PD/MT distances for `candidateUV_expanded2688` are N/A — TTK has not been re-run on these outputs. Values for CNN/GAN are loaded from the pre-computed merged CSV.

## Summary metrics table

| Metric | Bicubic | Cnn | Gan | Candidateuv_expanded2688 |
|---|---|---|---|---|
| PSNRuv (dB) | 32.2202 | 31.1925 | 29.1380 | 33.7892 |
| SSIM on speed | N/A | N/A | N/A | N/A |
| Speed MAE (m/s) | 0.5963 | 0.6941 | 0.9026 | 0.4958 |
| Speed RMSE (m/s) | 1.0156 | 1.1078 | 1.3775 | 0.8555 |
| WPD MAE (m³/s³) | 193.5403 | 231.6709 | 310.7328 | 158.9769 |
| WPD Wasserstein-1 | 55.5103 | 45.2713 | 85.6191 | 39.3927 |
| WPD bias abs (m³/s³) | 41.2244 | 35.3439 | 78.7236 | 31.3033 |
| Gradient MAE | 0.3696 | 0.3491 | 0.3806 | 0.3247 |
| Gradient W1 | 0.3282 | 0.2329 | 0.0564 | 0.2496 |
| Gradient kurtosis abs Δ | 6.6163 | 3.7004 | 4.2010 | 5.5793 |
| PSD log-L2 | 1.3625 | 0.8335 | 0.5139 | 1.1338 |
| PSD slope abs Δ | 1.6684 | 0.9150 | 0.9482 | 1.4560 |
| Exceedance abs Δ s>5 | 0.0095 | 0.0042 | 0.0082 | 0.0059 |
| Exceedance abs Δ s>10 | 0.0085 | 0.0066 | 0.0243 | 0.0055 |
| Exceedance abs Δ s>15 | 0.0074 | 0.0062 | 0.0096 | 0.0053 |
| Exceedance abs Δ p90 | 0.0109 | 0.0103 | 0.0123 | 0.0079 |
| Component-count curve L1 | 173.1855 | 115.5278 | 124.6567 | 141.9157 |
| PD distance (TTK) | N/A | 27.4063 | 20.8641 | N/A |
| MT distance (TTK) | N/A | 5.8678 | 8.3481 | N/A |

## 7 Key evaluation questions

### Q1. Did candidateUV_expanded2688 preserve CNN's direct-fidelity advantage?

- **PSNRuv**: CNN=31.1925, GAN=29.1380, candidateUV_expanded2688=33.7892. Δ(candidateUV_expanded2688−CNN)=+2.5967 (▲ better), improved on 168/168 samples.
- **SSIM**: CNN=N/A, GAN=N/A, candidateUV_expanded2688=N/A. Δ(candidateUV_expanded2688−CNN)=N/A, improved on 0/0 samples.
- **Speed MAE**: CNN=0.6941, GAN=0.9026, candidateUV_expanded2688=0.4958. Δ(candidateUV_expanded2688−CNN)=-0.1983 (▲ better), improved on 168/168 samples.
- **Speed RMSE**: CNN=1.1078, GAN=1.3775, candidateUV_expanded2688=0.8555. Δ(candidateUV_expanded2688−CNN)=-0.2523 (▲ better), improved on 168/168 samples.

### Q2. Did candidateUV_expanded2688 improve scalar speed or WPD metrics?

- **WPD MAE**: CNN=231.6709, candidateUV_expanded2688=158.9769. Δ=-72.6940 (▲ better), improved on 168/168 samples.
- **WPD W1**: CNN=45.2713, candidateUV_expanded2688=39.3927. Δ=-5.8786 (▲ better), improved on 111/168 samples.
- **WPD bias abs**: CNN=35.3439, candidateUV_expanded2688=31.3033. Δ=-4.0406 (▲ better), improved on 104/168 samples.

### Q3. Did candidateUV_expanded2688 improve gradient metrics?

- **Gradient MAE**: CNN=0.3491, candidateUV_expanded2688=0.3247. Δ=-0.0243 (▲ better), improved on 168/168 samples.
- **Gradient W1**: CNN=0.2329, candidateUV_expanded2688=0.2496. Δ=+0.0167 (▼ worse), improved on 32/168 samples.
- **Gradient kurtosis abs Δ**: CNN=3.7004, candidateUV_expanded2688=5.5793. Δ=+1.8789 (▼ worse), improved on 55/168 samples.

### Q4. Did candidateUV_expanded2688 improve exceedance metrics, especially s>5?

- **Exceedance abs Δ s>5**: CNN=0.00424, candidateUV_expanded2688=0.00588. Δ=+0.0016 (▼ worse), improved on 41/168 samples.
- **Exceedance abs Δ s>10**: CNN=0.00658, candidateUV_expanded2688=0.00547. Δ=-0.0011 (▲ better), improved on 109/168 samples.
- **Exceedance abs Δ s>15**: CNN=0.00621, candidateUV_expanded2688=0.00534. Δ=-0.0009 (▲ better), improved on 100/168 samples.
- **Exceedance abs Δ p90**: CNN=0.01025, candidateUV_expanded2688=0.00787. Δ=-0.0024 (▲ better), improved on 116/168 samples.

### Q5. Did candidateUV_expanded2688 move component-count behavior toward GT?

- **Component-count curve L1**: CNN=115.5278, candidateUV_expanded2688=141.9157. Δ=+26.3879 (▼ worse), improved on 4/168 samples.
- **Comp-count abs Δ t5**: CNN=64.3810, candidateUV_expanded2688=68.8810. Δ=+4.5000 (▼ worse), improved on 44/168 samples.
- **Comp-count abs Δ t10**: CNN=187.1667, candidateUV_expanded2688=228.8095. Δ=+41.6429 (▼ worse), improved on 3/168 samples.
- **Comp-count abs Δ t15**: CNN=116.8095, candidateUV_expanded2688=143.5655. Δ=+26.7560 (▼ worse), improved on 28/168 samples.

### Q6. Did candidateUV_expanded2688 improve PD or MT distances?

PD and MT distances for `candidateUV_expanded2688` are not available in this evaluation (TTK was not re-run on these outputs). Pre-computed values for CNN and GAN:

- **PD distance**: CNN=27.4063, GAN=20.8641, candidateUV_expanded2688=N/A (requires TTK run)
- **MT distance**: CNN=5.8678, GAN=8.3481, candidateUV_expanded2688=N/A (requires TTK run)

To compute PD/MT for `candidateUV_expanded2688`: convert outputs to VTI via `scripts/convert_phire_to_vti.py`, then run `scripts/compute_composite_tree_distance.py`.

### Q7. Did candidateUV_expanded2688 change the adjacent clusters (samples 10–13, 90–93, 162–163)?

See `adjacent_cluster_table_candidateUV_expanded2688.csv` for full detail. Summary of PSNRuv, speed MAE, and gradient MAE changes:

| Sample | Δ PSNRuv | Δ Speed MAE | Δ Grad MAE | Δ Exceed s>5 |
|---|---|---|---|---|
| 10 | 2.771 | -0.2208 | -0.0237 | 0.00422 |
| 11 | 2.664 | -0.2176 | -0.0252 | 0.00307 |
| 12 | 2.699 | -0.2170 | -0.0301 | 0.00265 |
| 13 | 2.626 | -0.2095 | -0.0279 | 0.00017 |
| 90 | 2.395 | -0.1539 | -0.0200 | -0.00262 |
| 91 | 2.426 | -0.1589 | -0.0198 | 0.00112 |
| 92 | 2.448 | -0.1717 | -0.0196 | 0.00587 |
| 93 | 2.430 | -0.1834 | -0.0168 | 0.00521 |
| 162 | 3.912 | -0.1596 | -0.0255 | -0.00555 |
| 163 | 3.851 | -0.1600 | -0.0278 | -0.00080 |

## Pairwise summary: CNN vs candidateUV_expanded2688

| Metric | Mean CNN | Mean candidateUV_expanded2688 | Δ | N improved | N worsened |
|---|---|---|---|---|---|
| psnruv | 31.1925 | 33.7892 | 2.5967 | 168 | 0 |
| ssim | N/A | N/A | N/A | 0 | 0 |
| speed_mae | 0.6941 | 0.4958 | -0.1983 | 168 | 0 |
| speed_rmse | 1.1078 | 0.8555 | -0.2523 | 168 | 0 |
| wpd_bias_abs | 35.3439 | 31.3033 | -4.0406 | 104 | 64 |
| wpd_mae | 231.6709 | 158.9769 | -72.6940 | 168 | 0 |
| wpd_w1 | 45.2713 | 39.3927 | -5.8786 | 111 | 57 |
| psd_log_l2 | 0.8335 | 1.1338 | 0.3003 | 0 | 168 |
| psd_slope_abs_delta | 0.9150 | 1.4560 | 0.5410 | 0 | 168 |
| grad_mae | 0.3491 | 0.3247 | -0.0243 | 168 | 0 |
| grad_w1 | 0.2329 | 0.2496 | 0.0167 | 32 | 136 |
| grad_kurtosis_abs_delta | 3.7004 | 5.5793 | 1.8789 | 55 | 113 |
| exceed_abs_t5 | 0.0042 | 0.0059 | 0.0016 | 41 | 127 |
| exceed_abs_t10 | 0.0066 | 0.0055 | -0.0011 | 109 | 59 |
| exceed_abs_t15 | 0.0062 | 0.0053 | -0.0009 | 100 | 68 |
| exceed_abs_p90 | 0.0103 | 0.0079 | -0.0024 | 116 | 52 |
| comp_curve_l1 | 115.5278 | 141.9157 | 26.3879 | 4 | 163 |
| pd_distance | N/A | N/A | N/A | 0 | 0 |
| mt_distance | N/A | N/A | N/A | 0 | 0 |

## Output files

- `ttk_runs_fixed/topology_finetuning/candidateUV_expanded2688_eval/all_sample_metrics_candidateUV_expanded2688.csv` — per-sample metrics for all methods
- `ttk_runs_fixed/topology_finetuning/candidateUV_expanded2688_eval/pairwise_cnn_vs_candidateUV_expanded2688.csv` — per-metric delta table
- `ttk_runs_fixed/topology_finetuning/candidateUV_expanded2688_eval/winner_counts_candidateUV_expanded2688.csv` — win counts per method
- `ttk_runs_fixed/topology_finetuning/candidateUV_expanded2688_eval/adjacent_cluster_table_candidateUV_expanded2688.csv` — adjacent-cluster detail
- `docs/topology_finetuning_candidateUV_expanded2688_eval.md` — this report

## Notes

- PSNRuv is computed on physical [u,v] arrays using per-sample dynamic data range (GT max − GT min).
- All speed-based metrics use scalar speed = sqrt(u² + v²) in physical units (m/s).
- Exceedance fractions use GT percentile thresholds (p90/p95/p99) computed per sample.
- Component counts use 8-connectivity on speed superlevel sets.
- PD/MT distances for CNN and GAN are loaded from pre-computed merged CSV; `candidateUV_expanded2688` requires a fresh TTK run.
- `candidateUV_expanded2688` was fine-tuned from the pretrained CNN checkpoint. Results may change with different lambda settings or more epochs.
