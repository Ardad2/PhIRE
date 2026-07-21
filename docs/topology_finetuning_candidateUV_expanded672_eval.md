# candidateUV_expanded672 fine-tuning evaluation

**Generated:** 2026-05-26

**Candidate:** `candidateUV_expanded672`

**Samples evaluated:** 168

**Methods compared:** bicubic, cnn, gan, candidateUV_expanded672

> **SSIM note:** SSIM was not computed because skimage could not be imported in the current NumPy environment (`ValueError('numpy.dtype size changed, may indicate binary incompatibility. Expected 96 from C header, got 88 from PyObject')`). All SSIM values are NaN.

> Note: PD/MT distances for `candidateUV_expanded672` are N/A — TTK has not been re-run on these outputs. Values for CNN/GAN are loaded from the pre-computed merged CSV.

## Summary metrics table

| Metric | Bicubic | Cnn | Gan | Candidateuv_expanded672 |
|---|---|---|---|---|
| PSNRuv (dB) | 32.2202 | 31.1925 | 29.1380 | 33.2549 |
| SSIM on speed | N/A | N/A | N/A | N/A |
| Speed MAE (m/s) | 0.5963 | 0.6941 | 0.9026 | 0.5323 |
| Speed RMSE (m/s) | 1.0156 | 1.1078 | 1.3775 | 0.9003 |
| WPD MAE (m³/s³) | 193.5403 | 231.6709 | 310.7328 | 173.4316 |
| WPD Wasserstein-1 | 55.5103 | 45.2713 | 85.6191 | 43.2818 |
| WPD bias abs (m³/s³) | 41.2244 | 35.3439 | 78.7236 | 31.8010 |
| Gradient MAE | 0.3696 | 0.3491 | 0.3806 | 0.3347 |
| Gradient W1 | 0.3282 | 0.2329 | 0.0564 | 0.2593 |
| Gradient kurtosis abs Δ | 6.6163 | 3.7004 | 4.2010 | 2.7841 |
| PSD log-L2 | 1.3625 | 0.8335 | 0.5139 | 1.1226 |
| PSD slope abs Δ | 1.6684 | 0.9150 | 0.9482 | 1.4023 |
| Exceedance abs Δ s>5 | 0.0095 | 0.0042 | 0.0082 | 0.0072 |
| Exceedance abs Δ s>10 | 0.0085 | 0.0066 | 0.0243 | 0.0052 |
| Exceedance abs Δ s>15 | 0.0074 | 0.0062 | 0.0096 | 0.0060 |
| Exceedance abs Δ p90 | 0.0109 | 0.0103 | 0.0123 | 0.0084 |
| Component-count curve L1 | 173.1855 | 115.5278 | 124.6567 | 140.7183 |
| PD distance (TTK) | N/A | 27.4063 | 20.8641 | N/A |
| MT distance (TTK) | N/A | 5.8678 | 8.3481 | N/A |

## 7 Key evaluation questions

### Q1. Did candidateUV_expanded672 preserve CNN's direct-fidelity advantage?

- **PSNRuv**: CNN=31.1925, GAN=29.1380, candidateUV_expanded672=33.2549. Δ(candidateUV_expanded672−CNN)=+2.0624 (▲ better), improved on 168/168 samples.
- **SSIM**: CNN=N/A, GAN=N/A, candidateUV_expanded672=N/A. Δ(candidateUV_expanded672−CNN)=N/A, improved on 0/0 samples.
- **Speed MAE**: CNN=0.6941, GAN=0.9026, candidateUV_expanded672=0.5323. Δ(candidateUV_expanded672−CNN)=-0.1618 (▲ better), improved on 168/168 samples.
- **Speed RMSE**: CNN=1.1078, GAN=1.3775, candidateUV_expanded672=0.9003. Δ(candidateUV_expanded672−CNN)=-0.2075 (▲ better), improved on 168/168 samples.

### Q2. Did candidateUV_expanded672 improve scalar speed or WPD metrics?

- **WPD MAE**: CNN=231.6709, candidateUV_expanded672=173.4316. Δ=-58.2393 (▲ better), improved on 168/168 samples.
- **WPD W1**: CNN=45.2713, candidateUV_expanded672=43.2818. Δ=-1.9895 (▲ better), improved on 98/168 samples.
- **WPD bias abs**: CNN=35.3439, candidateUV_expanded672=31.8010. Δ=-3.5429 (▲ better), improved on 103/168 samples.

### Q3. Did candidateUV_expanded672 improve gradient metrics?

- **Gradient MAE**: CNN=0.3491, candidateUV_expanded672=0.3347. Δ=-0.0144 (▲ better), improved on 168/168 samples.
- **Gradient W1**: CNN=0.2329, candidateUV_expanded672=0.2593. Δ=+0.0264 (▼ worse), improved on 13/168 samples.
- **Gradient kurtosis abs Δ**: CNN=3.7004, candidateUV_expanded672=2.7841. Δ=-0.9163 (▲ better), improved on 84/168 samples.

### Q4. Did candidateUV_expanded672 improve exceedance metrics, especially s>5?

- **Exceedance abs Δ s>5**: CNN=0.00424, candidateUV_expanded672=0.00720. Δ=+0.0030 (▼ worse), improved on 21/168 samples.
- **Exceedance abs Δ s>10**: CNN=0.00658, candidateUV_expanded672=0.00525. Δ=-0.0013 (▲ better), improved on 99/168 samples.
- **Exceedance abs Δ s>15**: CNN=0.00621, candidateUV_expanded672=0.00604. Δ=-0.0002 (▲ better), improved on 81/168 samples.
- **Exceedance abs Δ p90**: CNN=0.01025, candidateUV_expanded672=0.00837. Δ=-0.0019 (▲ better), improved on 103/168 samples.

### Q5. Did candidateUV_expanded672 move component-count behavior toward GT?

- **Component-count curve L1**: CNN=115.5278, candidateUV_expanded672=140.7183. Δ=+25.1905 (▼ worse), improved on 9/168 samples.
- **Comp-count abs Δ t5**: CNN=64.3810, candidateUV_expanded672=69.9464. Δ=+5.5655 (▼ worse), improved on 31/168 samples.
- **Comp-count abs Δ t10**: CNN=187.1667, candidateUV_expanded672=227.3393. Δ=+40.1726 (▼ worse), improved on 2/168 samples.
- **Comp-count abs Δ t15**: CNN=116.8095, candidateUV_expanded672=141.1429. Δ=+24.3333 (▼ worse), improved on 23/168 samples.

### Q6. Did candidateUV_expanded672 improve PD or MT distances?

PD and MT distances for `candidateUV_expanded672` are not available in this evaluation (TTK was not re-run on these outputs). Pre-computed values for CNN and GAN:

- **PD distance**: CNN=27.4063, GAN=20.8641, candidateUV_expanded672=N/A (requires TTK run)
- **MT distance**: CNN=5.8678, GAN=8.3481, candidateUV_expanded672=N/A (requires TTK run)

To compute PD/MT for `candidateUV_expanded672`: convert outputs to VTI via `scripts/convert_phire_to_vti.py`, then run `scripts/compute_composite_tree_distance.py`.

### Q7. Did candidateUV_expanded672 change the adjacent clusters (samples 10–13, 90–93, 162–163)?

See `adjacent_cluster_table_candidateUV_expanded672.csv` for full detail. Summary of PSNRuv, speed MAE, and gradient MAE changes:

| Sample | Δ PSNRuv | Δ Speed MAE | Δ Grad MAE | Δ Exceed s>5 |
|---|---|---|---|---|
| 10 | 2.268 | -0.1824 | -0.0116 | 0.00466 |
| 11 | 2.141 | -0.1759 | -0.0116 | 0.00332 |
| 12 | 2.162 | -0.1748 | -0.0167 | 0.00282 |
| 13 | 2.067 | -0.1636 | -0.0134 | -0.00030 |
| 90 | 1.909 | -0.1241 | -0.0134 | 0.00306 |
| 91 | 1.941 | -0.1287 | -0.0128 | 0.00728 |
| 92 | 1.953 | -0.1403 | -0.0134 | 0.01154 |
| 93 | 1.938 | -0.1498 | -0.0096 | 0.00906 |
| 162 | 3.315 | -0.1380 | -0.0209 | -0.00323 |
| 163 | 3.233 | -0.1377 | -0.0222 | 0.00218 |

## Pairwise summary: CNN vs candidateUV_expanded672

| Metric | Mean CNN | Mean candidateUV_expanded672 | Δ | N improved | N worsened |
|---|---|---|---|---|---|
| psnruv | 31.1925 | 33.2549 | 2.0624 | 168 | 0 |
| ssim | N/A | N/A | N/A | 0 | 0 |
| speed_mae | 0.6941 | 0.5323 | -0.1618 | 168 | 0 |
| speed_rmse | 1.1078 | 0.9003 | -0.2075 | 168 | 0 |
| wpd_bias_abs | 35.3439 | 31.8010 | -3.5429 | 103 | 65 |
| wpd_mae | 231.6709 | 173.4316 | -58.2393 | 168 | 0 |
| wpd_w1 | 45.2713 | 43.2818 | -1.9895 | 98 | 70 |
| psd_log_l2 | 0.8335 | 1.1226 | 0.2891 | 0 | 168 |
| psd_slope_abs_delta | 0.9150 | 1.4023 | 0.4873 | 0 | 168 |
| grad_mae | 0.3491 | 0.3347 | -0.0144 | 168 | 0 |
| grad_w1 | 0.2329 | 0.2593 | 0.0264 | 13 | 155 |
| grad_kurtosis_abs_delta | 3.7004 | 2.7841 | -0.9163 | 84 | 84 |
| exceed_abs_t5 | 0.0042 | 0.0072 | 0.0030 | 21 | 147 |
| exceed_abs_t10 | 0.0066 | 0.0052 | -0.0013 | 99 | 69 |
| exceed_abs_t15 | 0.0062 | 0.0060 | -0.0002 | 81 | 86 |
| exceed_abs_p90 | 0.0103 | 0.0084 | -0.0019 | 103 | 65 |
| comp_curve_l1 | 115.5278 | 140.7183 | 25.1905 | 9 | 159 |
| pd_distance | N/A | N/A | N/A | 0 | 0 |
| mt_distance | N/A | N/A | N/A | 0 | 0 |

## Output files

- `ttk_runs_fixed/topology_finetuning/candidateUV_expanded672_eval/all_sample_metrics_candidateUV_expanded672.csv` — per-sample metrics for all methods
- `ttk_runs_fixed/topology_finetuning/candidateUV_expanded672_eval/pairwise_cnn_vs_candidateUV_expanded672.csv` — per-metric delta table
- `ttk_runs_fixed/topology_finetuning/candidateUV_expanded672_eval/winner_counts_candidateUV_expanded672.csv` — win counts per method
- `ttk_runs_fixed/topology_finetuning/candidateUV_expanded672_eval/adjacent_cluster_table_candidateUV_expanded672.csv` — adjacent-cluster detail
- `docs/topology_finetuning_candidateUV_expanded672_eval.md` — this report

## Notes

- PSNRuv is computed on physical [u,v] arrays using per-sample dynamic data range (GT max − GT min).
- All speed-based metrics use scalar speed = sqrt(u² + v²) in physical units (m/s).
- Exceedance fractions use GT percentile thresholds (p90/p95/p99) computed per sample.
- Component counts use 8-connectivity on speed superlevel sets.
- PD/MT distances for CNN and GAN are loaded from pre-computed merged CSV; `candidateUV_expanded672` requires a fresh TTK run.
- `candidateUV_expanded672` was fine-tuned from the pretrained CNN checkpoint. Results may change with different lambda settings or more epochs.
