# candidateB_expanded672 fine-tuning evaluation

**Generated:** 2026-05-26

**Candidate:** `candidateB_expanded672`

**Samples evaluated:** 168

**Methods compared:** bicubic, cnn, gan, candidateB_expanded672

> **SSIM note:** SSIM was not computed because skimage could not be imported in the current NumPy environment (`ValueError('numpy.dtype size changed, may indicate binary incompatibility. Expected 96 from C header, got 88 from PyObject')`). All SSIM values are NaN.

> Note: PD/MT distances for `candidateB_expanded672` are N/A — TTK has not been re-run on these outputs. Values for CNN/GAN are loaded from the pre-computed merged CSV.

## Summary metrics table

| Metric | Bicubic | Cnn | Gan | Candidateb_expanded672 |
|---|---|---|---|---|
| PSNRuv (dB) | 32.2202 | 31.1925 | 29.1380 | 33.0574 |
| SSIM on speed | N/A | N/A | N/A | N/A |
| Speed MAE (m/s) | 0.5963 | 0.6941 | 0.9026 | 0.5443 |
| Speed RMSE (m/s) | 1.0156 | 1.1078 | 1.3775 | 0.9153 |
| WPD MAE (m³/s³) | 193.5403 | 231.6709 | 310.7328 | 176.5828 |
| WPD Wasserstein-1 | 55.5103 | 45.2713 | 85.6191 | 30.7988 |
| WPD bias abs (m³/s³) | 41.2244 | 35.3439 | 78.7236 | 20.6078 |
| Gradient MAE | 0.3696 | 0.3491 | 0.3806 | 0.3226 |
| Gradient W1 | 0.3282 | 0.2329 | 0.0564 | 0.2177 |
| Gradient kurtosis abs Δ | 6.6163 | 3.7004 | 4.2010 | 2.1621 |
| PSD log-L2 | 1.3625 | 0.8335 | 0.5139 | 1.0040 |
| PSD slope abs Δ | 1.6684 | 0.9150 | 0.9482 | 1.2821 |
| Exceedance abs Δ s>5 | 0.0095 | 0.0042 | 0.0082 | 0.0059 |
| Exceedance abs Δ s>10 | 0.0085 | 0.0066 | 0.0243 | 0.0036 |
| Exceedance abs Δ s>15 | 0.0074 | 0.0062 | 0.0096 | 0.0042 |
| Exceedance abs Δ p90 | 0.0109 | 0.0103 | 0.0123 | 0.0059 |
| Component-count curve L1 | 173.1855 | 115.5278 | 124.6567 | 113.9841 |
| PD distance (TTK) | N/A | 27.4063 | 20.8641 | N/A |
| MT distance (TTK) | N/A | 5.8678 | 8.3481 | N/A |

## 7 Key evaluation questions

### Q1. Did candidateB_expanded672 preserve CNN's direct-fidelity advantage?

- **PSNRuv**: CNN=31.1925, GAN=29.1380, candidateB_expanded672=33.0574. Δ(candidateB_expanded672−CNN)=+1.8649 (▲ better), improved on 168/168 samples.
- **SSIM**: CNN=N/A, GAN=N/A, candidateB_expanded672=N/A. Δ(candidateB_expanded672−CNN)=N/A, improved on 0/0 samples.
- **Speed MAE**: CNN=0.6941, GAN=0.9026, candidateB_expanded672=0.5443. Δ(candidateB_expanded672−CNN)=-0.1498 (▲ better), improved on 168/168 samples.
- **Speed RMSE**: CNN=1.1078, GAN=1.3775, candidateB_expanded672=0.9153. Δ(candidateB_expanded672−CNN)=-0.1925 (▲ better), improved on 168/168 samples.

### Q2. Did candidateB_expanded672 improve scalar speed or WPD metrics?

- **WPD MAE**: CNN=231.6709, candidateB_expanded672=176.5828. Δ=-55.0881 (▲ better), improved on 168/168 samples.
- **WPD W1**: CNN=45.2713, candidateB_expanded672=30.7988. Δ=-14.4724 (▲ better), improved on 123/168 samples.
- **WPD bias abs**: CNN=35.3439, candidateB_expanded672=20.6078. Δ=-14.7361 (▲ better), improved on 117/168 samples.

### Q3. Did candidateB_expanded672 improve gradient metrics?

- **Gradient MAE**: CNN=0.3491, candidateB_expanded672=0.3226. Δ=-0.0264 (▲ better), improved on 168/168 samples.
- **Gradient W1**: CNN=0.2329, candidateB_expanded672=0.2177. Δ=-0.0153 (▲ better), improved on 131/168 samples.
- **Gradient kurtosis abs Δ**: CNN=3.7004, candidateB_expanded672=2.1621. Δ=-1.5383 (▲ better), improved on 97/168 samples.

### Q4. Did candidateB_expanded672 improve exceedance metrics, especially s>5?

- **Exceedance abs Δ s>5**: CNN=0.00424, candidateB_expanded672=0.00586. Δ=+0.0016 (▼ worse), improved on 43/168 samples.
- **Exceedance abs Δ s>10**: CNN=0.00658, candidateB_expanded672=0.00365. Δ=-0.0029 (▲ better), improved on 119/168 samples.
- **Exceedance abs Δ s>15**: CNN=0.00621, candidateB_expanded672=0.00418. Δ=-0.0020 (▲ better), improved on 111/168 samples.
- **Exceedance abs Δ p90**: CNN=0.01025, candidateB_expanded672=0.00590. Δ=-0.0044 (▲ better), improved on 132/168 samples.

### Q5. Did candidateB_expanded672 move component-count behavior toward GT?

- **Component-count curve L1**: CNN=115.5278, candidateB_expanded672=113.9841. Δ=-1.5437 (▲ better), improved on 105/168 samples.
- **Comp-count abs Δ t5**: CNN=64.3810, candidateB_expanded672=61.5655. Δ=-2.8155 (▲ better), improved on 95/168 samples.
- **Comp-count abs Δ t10**: CNN=187.1667, candidateB_expanded672=192.1190. Δ=+4.9524 (▼ worse), improved on 68/168 samples.
- **Comp-count abs Δ t15**: CNN=116.8095, candidateB_expanded672=109.4702. Δ=-7.3393 (▲ better), improved on 109/168 samples.

### Q6. Did candidateB_expanded672 improve PD or MT distances?

PD and MT distances for `candidateB_expanded672` are not available in this evaluation (TTK was not re-run on these outputs). Pre-computed values for CNN and GAN:

- **PD distance**: CNN=27.4063, GAN=20.8641, candidateB_expanded672=N/A (requires TTK run)
- **MT distance**: CNN=5.8678, GAN=8.3481, candidateB_expanded672=N/A (requires TTK run)

To compute PD/MT for `candidateB_expanded672`: convert outputs to VTI via `scripts/convert_phire_to_vti.py`, then run `scripts/compute_composite_tree_distance.py`.

### Q7. Did candidateB_expanded672 change the adjacent clusters (samples 10–13, 90–93, 162–163)?

See `adjacent_cluster_table_candidateB_expanded672.csv` for full detail. Summary of PSNRuv, speed MAE, and gradient MAE changes:

| Sample | Δ PSNRuv | Δ Speed MAE | Δ Grad MAE | Δ Exceed s>5 |
|---|---|---|---|---|
| 10 | 2.108 | -0.1719 | -0.0245 | 0.00269 |
| 11 | 1.973 | -0.1660 | -0.0246 | 0.00136 |
| 12 | 1.988 | -0.1659 | -0.0292 | 0.00126 |
| 13 | 1.879 | -0.1552 | -0.0259 | -0.00142 |
| 90 | 1.724 | -0.1141 | -0.0203 | 0.00389 |
| 91 | 1.757 | -0.1189 | -0.0200 | 0.00884 |
| 92 | 1.742 | -0.1289 | -0.0228 | 0.01179 |
| 93 | 1.713 | -0.1375 | -0.0232 | 0.00757 |
| 162 | 3.035 | -0.1250 | -0.0206 | -0.00282 |
| 163 | 2.995 | -0.1257 | -0.0221 | 0.00272 |

## Pairwise summary: CNN vs candidateB_expanded672

| Metric | Mean CNN | Mean candidateB_expanded672 | Δ | N improved | N worsened |
|---|---|---|---|---|---|
| psnruv | 31.1925 | 33.0574 | 1.8649 | 168 | 0 |
| ssim | N/A | N/A | N/A | 0 | 0 |
| speed_mae | 0.6941 | 0.5443 | -0.1498 | 168 | 0 |
| speed_rmse | 1.1078 | 0.9153 | -0.1925 | 168 | 0 |
| wpd_bias_abs | 35.3439 | 20.6078 | -14.7361 | 117 | 51 |
| wpd_mae | 231.6709 | 176.5828 | -55.0881 | 168 | 0 |
| wpd_w1 | 45.2713 | 30.7988 | -14.4724 | 123 | 45 |
| psd_log_l2 | 0.8335 | 1.0040 | 0.1705 | 0 | 168 |
| psd_slope_abs_delta | 0.9150 | 1.2821 | 0.3671 | 0 | 168 |
| grad_mae | 0.3491 | 0.3226 | -0.0264 | 168 | 0 |
| grad_w1 | 0.2329 | 0.2177 | -0.0153 | 131 | 37 |
| grad_kurtosis_abs_delta | 3.7004 | 2.1621 | -1.5383 | 97 | 71 |
| exceed_abs_t5 | 0.0042 | 0.0059 | 0.0016 | 43 | 125 |
| exceed_abs_t10 | 0.0066 | 0.0036 | -0.0029 | 119 | 48 |
| exceed_abs_t15 | 0.0062 | 0.0042 | -0.0020 | 111 | 56 |
| exceed_abs_p90 | 0.0103 | 0.0059 | -0.0044 | 132 | 36 |
| comp_curve_l1 | 115.5278 | 113.9841 | -1.5437 | 105 | 63 |
| pd_distance | N/A | N/A | N/A | 0 | 0 |
| mt_distance | N/A | N/A | N/A | 0 | 0 |

## Output files

- `ttk_runs_fixed/topology_finetuning/candidateB_expanded672_eval/all_sample_metrics_candidateB_expanded672.csv` — per-sample metrics for all methods
- `ttk_runs_fixed/topology_finetuning/candidateB_expanded672_eval/pairwise_cnn_vs_candidateB_expanded672.csv` — per-metric delta table
- `ttk_runs_fixed/topology_finetuning/candidateB_expanded672_eval/winner_counts_candidateB_expanded672.csv` — win counts per method
- `ttk_runs_fixed/topology_finetuning/candidateB_expanded672_eval/adjacent_cluster_table_candidateB_expanded672.csv` — adjacent-cluster detail
- `docs/topology_finetuning_candidateB_expanded672_eval.md` — this report

## Notes

- PSNRuv is computed on physical [u,v] arrays using per-sample dynamic data range (GT max − GT min).
- All speed-based metrics use scalar speed = sqrt(u² + v²) in physical units (m/s).
- Exceedance fractions use GT percentile thresholds (p90/p95/p99) computed per sample.
- Component counts use 8-connectivity on speed superlevel sets.
- PD/MT distances for CNN and GAN are loaded from pre-computed merged CSV; `candidateB_expanded672` requires a fresh TTK run.
- `candidateB_expanded672` was fine-tuned from the pretrained CNN checkpoint. Results may change with different lambda settings or more epochs.
