# candidateE2_expanded672 fine-tuning evaluation

**Generated:** 2026-05-28

**Candidate:** `candidateE2_expanded672`

**Samples evaluated:** 168

**Methods compared:** bicubic, cnn, gan, candidateE2_expanded672

> **SSIM note:** SSIM was not computed because skimage could not be imported in the current NumPy environment (`ValueError('numpy.dtype size changed, may indicate binary incompatibility. Expected 96 from C header, got 88 from PyObject')`). All SSIM values are NaN.

> Note: PD/MT distances for `candidateE2_expanded672` are N/A — TTK has not been re-run on these outputs. Values for CNN/GAN are loaded from the pre-computed merged CSV.

## Summary metrics table

| Metric | Bicubic | Cnn | Gan | Candidatee2_expanded672 |
|---|---|---|---|---|
| PSNRuv (dB) | 32.2202 | 31.1925 | 29.1380 | 31.4334 |
| SSIM on speed | N/A | N/A | N/A | N/A |
| Speed MAE (m/s) | 0.5963 | 0.6941 | 0.9026 | 0.6788 |
| Speed RMSE (m/s) | 1.0156 | 1.1078 | 1.3775 | 1.0751 |
| WPD MAE (m³/s³) | 193.5403 | 231.6709 | 310.7328 | 225.9703 |
| WPD Wasserstein-1 | 55.5103 | 45.2713 | 85.6191 | 49.9461 |
| WPD bias abs (m³/s³) | 41.2244 | 35.3439 | 78.7236 | 34.0484 |
| Gradient MAE | 0.3696 | 0.3491 | 0.3806 | 0.3569 |
| Gradient W1 | 0.3282 | 0.2329 | 0.0564 | 0.2726 |
| Gradient kurtosis abs Δ | 6.6163 | 3.7004 | 4.2010 | 4.4419 |
| PSD log-L2 | 1.3625 | 0.8335 | 0.5139 | 0.9299 |
| PSD slope abs Δ | 1.6684 | 0.9150 | 0.9482 | 1.1167 |
| Exceedance abs Δ s>5 | 0.0095 | 0.0042 | 0.0082 | 0.0066 |
| Exceedance abs Δ s>10 | 0.0085 | 0.0066 | 0.0243 | 0.0077 |
| Exceedance abs Δ s>15 | 0.0074 | 0.0062 | 0.0096 | 0.0071 |
| Exceedance abs Δ p90 | 0.0109 | 0.0103 | 0.0123 | 0.0097 |
| Component-count curve L1 | 173.1855 | 115.5278 | 124.6567 | 129.3770 |
| PD distance (TTK) | N/A | 27.4063 | 20.8641 | N/A |
| MT distance (TTK) | N/A | 5.8678 | 8.3481 | N/A |

## 7 Key evaluation questions

### Q1. Did candidateE2_expanded672 preserve CNN's direct-fidelity advantage?

- **PSNRuv**: CNN=31.1925, GAN=29.1380, candidateE2_expanded672=31.4334. Δ(candidateE2_expanded672−CNN)=+0.2409 (▲ better), improved on 168/168 samples.
- **SSIM**: CNN=N/A, GAN=N/A, candidateE2_expanded672=N/A. Δ(candidateE2_expanded672−CNN)=N/A, improved on 0/0 samples.
- **Speed MAE**: CNN=0.6941, GAN=0.9026, candidateE2_expanded672=0.6788. Δ(candidateE2_expanded672−CNN)=-0.0152 (▲ better), improved on 166/168 samples.
- **Speed RMSE**: CNN=1.1078, GAN=1.3775, candidateE2_expanded672=1.0751. Δ(candidateE2_expanded672−CNN)=-0.0326 (▲ better), improved on 168/168 samples.

### Q2. Did candidateE2_expanded672 improve scalar speed or WPD metrics?

- **WPD MAE**: CNN=231.6709, candidateE2_expanded672=225.9703. Δ=-5.7006 (▲ better), improved on 159/168 samples.
- **WPD W1**: CNN=45.2713, candidateE2_expanded672=49.9461. Δ=+4.6748 (▼ worse), improved on 93/168 samples.
- **WPD bias abs**: CNN=35.3439, candidateE2_expanded672=34.0484. Δ=-1.2955 (▲ better), improved on 103/168 samples.

### Q3. Did candidateE2_expanded672 improve gradient metrics?

- **Gradient MAE**: CNN=0.3491, candidateE2_expanded672=0.3569. Δ=+0.0079 (▼ worse), improved on 9/168 samples.
- **Gradient W1**: CNN=0.2329, candidateE2_expanded672=0.2726. Δ=+0.0397 (▼ worse), improved on 0/168 samples.
- **Gradient kurtosis abs Δ**: CNN=3.7004, candidateE2_expanded672=4.4419. Δ=+0.7416 (▼ worse), improved on 81/168 samples.

### Q4. Did candidateE2_expanded672 improve exceedance metrics, especially s>5?

- **Exceedance abs Δ s>5**: CNN=0.00424, candidateE2_expanded672=0.00656. Δ=+0.0023 (▼ worse), improved on 20/168 samples.
- **Exceedance abs Δ s>10**: CNN=0.00658, candidateE2_expanded672=0.00772. Δ=+0.0011 (▼ worse), improved on 76/168 samples.
- **Exceedance abs Δ s>15**: CNN=0.00621, candidateE2_expanded672=0.00708. Δ=+0.0009 (▼ worse), improved on 55/168 samples.
- **Exceedance abs Δ p90**: CNN=0.01025, candidateE2_expanded672=0.00968. Δ=-0.0006 (▲ better), improved on 107/168 samples.

### Q5. Did candidateE2_expanded672 move component-count behavior toward GT?

- **Component-count curve L1**: CNN=115.5278, candidateE2_expanded672=129.3770. Δ=+13.8492 (▼ worse), improved on 2/168 samples.
- **Comp-count abs Δ t5**: CNN=64.3810, candidateE2_expanded672=66.5238. Δ=+2.1429 (▼ worse), improved on 43/168 samples.
- **Comp-count abs Δ t10**: CNN=187.1667, candidateE2_expanded672=201.1845. Δ=+14.0179 (▼ worse), improved on 19/168 samples.
- **Comp-count abs Δ t15**: CNN=116.8095, candidateE2_expanded672=136.5298. Δ=+19.7202 (▼ worse), improved on 7/168 samples.

### Q6. Did candidateE2_expanded672 improve PD or MT distances?

PD and MT distances for `candidateE2_expanded672` are not available in this evaluation (TTK was not re-run on these outputs). Pre-computed values for CNN and GAN:

- **PD distance**: CNN=27.4063, GAN=20.8641, candidateE2_expanded672=N/A (requires TTK run)
- **MT distance**: CNN=5.8678, GAN=8.3481, candidateE2_expanded672=N/A (requires TTK run)

To compute PD/MT for `candidateE2_expanded672`: convert outputs to VTI via `scripts/convert_phire_to_vti.py`, then run `scripts/compute_composite_tree_distance.py`.

### Q7. Did candidateE2_expanded672 change the adjacent clusters (samples 10–13, 90–93, 162–163)?

See `adjacent_cluster_table_candidateE2_expanded672.csv` for full detail. Summary of PSNRuv, speed MAE, and gradient MAE changes:

| Sample | Δ PSNRuv | Δ Speed MAE | Δ Grad MAE | Δ Exceed s>5 |
|---|---|---|---|---|
| 10 | 0.356 | -0.0296 | 0.0082 | 0.00495 |
| 11 | 0.289 | -0.0234 | 0.0092 | 0.00381 |
| 12 | 0.256 | -0.0196 | 0.0095 | 0.00350 |
| 13 | 0.204 | -0.0121 | 0.0111 | 0.00309 |
| 90 | 0.108 | -0.0058 | 0.0031 | 0.00460 |
| 91 | 0.103 | -0.0067 | 0.0031 | 0.00496 |
| 92 | 0.132 | -0.0092 | 0.0041 | 0.00432 |
| 93 | 0.197 | -0.0158 | 0.0054 | 0.00545 |
| 162 | 0.676 | -0.0257 | -0.0063 | 0.00150 |
| 163 | 0.605 | -0.0269 | -0.0059 | 0.00632 |

## Pairwise summary: CNN vs candidateE2_expanded672

| Metric | Mean CNN | Mean candidateE2_expanded672 | Δ | N improved | N worsened |
|---|---|---|---|---|---|
| psnruv | 31.1925 | 31.4334 | 0.2409 | 168 | 0 |
| ssim | N/A | N/A | N/A | 0 | 0 |
| speed_mae | 0.6941 | 0.6788 | -0.0152 | 166 | 2 |
| speed_rmse | 1.1078 | 1.0751 | -0.0326 | 168 | 0 |
| wpd_bias_abs | 35.3439 | 34.0484 | -1.2955 | 103 | 65 |
| wpd_mae | 231.6709 | 225.9703 | -5.7006 | 159 | 9 |
| wpd_w1 | 45.2713 | 49.9461 | 4.6748 | 93 | 75 |
| psd_log_l2 | 0.8335 | 0.9299 | 0.0964 | 12 | 156 |
| psd_slope_abs_delta | 0.9150 | 1.1167 | 0.2017 | 43 | 125 |
| grad_mae | 0.3491 | 0.3569 | 0.0079 | 9 | 159 |
| grad_w1 | 0.2329 | 0.2726 | 0.0397 | 0 | 168 |
| grad_kurtosis_abs_delta | 3.7004 | 4.4419 | 0.7416 | 81 | 87 |
| exceed_abs_t5 | 0.0042 | 0.0066 | 0.0023 | 20 | 148 |
| exceed_abs_t10 | 0.0066 | 0.0077 | 0.0011 | 76 | 92 |
| exceed_abs_t15 | 0.0062 | 0.0071 | 0.0009 | 55 | 111 |
| exceed_abs_p90 | 0.0103 | 0.0097 | -0.0006 | 107 | 61 |
| comp_curve_l1 | 115.5278 | 129.3770 | 13.8492 | 2 | 166 |
| pd_distance | N/A | N/A | N/A | 0 | 0 |
| mt_distance | N/A | N/A | N/A | 0 | 0 |

## Output files

- `ttk_runs_fixed/topology_finetuning/candidateE2_expanded672_eval/all_sample_metrics_candidateE2_expanded672.csv` — per-sample metrics for all methods
- `ttk_runs_fixed/topology_finetuning/candidateE2_expanded672_eval/pairwise_cnn_vs_candidateE2_expanded672.csv` — per-metric delta table
- `ttk_runs_fixed/topology_finetuning/candidateE2_expanded672_eval/winner_counts_candidateE2_expanded672.csv` — win counts per method
- `ttk_runs_fixed/topology_finetuning/candidateE2_expanded672_eval/adjacent_cluster_table_candidateE2_expanded672.csv` — adjacent-cluster detail
- `docs/topology_finetuning_candidateE2_expanded672_eval.md` — this report

## Notes

- PSNRuv is computed on physical [u,v] arrays using per-sample dynamic data range (GT max − GT min).
- All speed-based metrics use scalar speed = sqrt(u² + v²) in physical units (m/s).
- Exceedance fractions use GT percentile thresholds (p90/p95/p99) computed per sample.
- Component counts use 8-connectivity on speed superlevel sets.
- PD/MT distances for CNN and GAN are loaded from pre-computed merged CSV; `candidateE2_expanded672` requires a fresh TTK run.
- `candidateE2_expanded672` was fine-tuned from the pretrained CNN checkpoint. Results may change with different lambda settings or more epochs.
