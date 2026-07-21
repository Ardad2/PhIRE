# candidateB_factorial_speed_levelset_expanded2688 fine-tuning evaluation

**Generated:** 2026-07-15

**Candidate:** `candidateB_factorial_speed_levelset_expanded2688`

**Samples evaluated:** 168

**Methods compared:** bicubic, cnn, gan, candidateB_factorial_speed_levelset_expanded2688

> **SSIM note:** SSIM was not computed because skimage could not be imported in the current NumPy environment (`ValueError('numpy.dtype size changed, may indicate binary incompatibility. Expected 96 from C header, got 88 from PyObject')`). All SSIM values are NaN.

> Note: PD/MT distances for `candidateB_factorial_speed_levelset_expanded2688` are N/A — TTK has not been re-run on these outputs. Values for CNN/GAN are loaded from the pre-computed merged CSV.

## Summary metrics table

| Metric | Bicubic | Cnn | Gan | Candidateb_factorial_speed_levelset_expanded2688 |
|---|---|---|---|---|
| PSNRuv (dB) | 32.2202 | 31.1925 | 29.1380 | 33.7789 |
| SSIM on speed | N/A | N/A | N/A | N/A |
| Speed MAE (m/s) | 0.5963 | 0.6941 | 0.9026 | 0.4934 |
| Speed RMSE (m/s) | 1.0156 | 1.1078 | 1.3775 | 0.8531 |
| WPD MAE (m³/s³) | 193.5403 | 231.6709 | 310.7328 | 157.9165 |
| WPD Wasserstein-1 | 55.5103 | 45.2713 | 85.6191 | 31.8966 |
| WPD bias abs (m³/s³) | 41.2244 | 35.3439 | 78.7236 | 21.2398 |
| Gradient MAE | 0.3696 | 0.3491 | 0.3806 | 0.3228 |
| Gradient W1 | 0.3282 | 0.2329 | 0.0564 | 0.2457 |
| Gradient kurtosis abs Δ | 6.6163 | 3.7004 | 4.2010 | 5.2715 |
| PSD log-L2 | 1.3625 | 0.8335 | 0.5139 | 1.1219 |
| PSD slope abs Δ | 1.6684 | 0.9150 | 0.9482 | 1.4492 |
| Exceedance abs Δ s>5 | 0.0095 | 0.0042 | 0.0082 | 0.0069 |
| Exceedance abs Δ s>10 | 0.0085 | 0.0066 | 0.0243 | 0.0045 |
| Exceedance abs Δ s>15 | 0.0074 | 0.0062 | 0.0096 | 0.0041 |
| Exceedance abs Δ p90 | 0.0109 | 0.0103 | 0.0123 | 0.0060 |
| Component-count curve L1 | 173.1855 | 115.5278 | 124.6567 | 141.4871 |
| PD distance (TTK) | N/A | 27.4063 | 20.8641 | N/A |
| MT distance (TTK) | N/A | 5.8678 | 8.3481 | N/A |

## 7 Key evaluation questions

### Q1. Did candidateB_factorial_speed_levelset_expanded2688 preserve CNN's direct-fidelity advantage?

- **PSNRuv**: CNN=31.1925, GAN=29.1380, candidateB_factorial_speed_levelset_expanded2688=33.7789. Δ(candidateB_factorial_speed_levelset_expanded2688−CNN)=+2.5864 (▲ better), improved on 168/168 samples.
- **SSIM**: CNN=N/A, GAN=N/A, candidateB_factorial_speed_levelset_expanded2688=N/A. Δ(candidateB_factorial_speed_levelset_expanded2688−CNN)=N/A, improved on 0/0 samples.
- **Speed MAE**: CNN=0.6941, GAN=0.9026, candidateB_factorial_speed_levelset_expanded2688=0.4934. Δ(candidateB_factorial_speed_levelset_expanded2688−CNN)=-0.2007 (▲ better), improved on 168/168 samples.
- **Speed RMSE**: CNN=1.1078, GAN=1.3775, candidateB_factorial_speed_levelset_expanded2688=0.8531. Δ(candidateB_factorial_speed_levelset_expanded2688−CNN)=-0.2547 (▲ better), improved on 168/168 samples.

### Q2. Did candidateB_factorial_speed_levelset_expanded2688 improve scalar speed or WPD metrics?

- **WPD MAE**: CNN=231.6709, candidateB_factorial_speed_levelset_expanded2688=157.9165. Δ=-73.7544 (▲ better), improved on 168/168 samples.
- **WPD W1**: CNN=45.2713, candidateB_factorial_speed_levelset_expanded2688=31.8966. Δ=-13.3747 (▲ better), improved on 137/168 samples.
- **WPD bias abs**: CNN=35.3439, candidateB_factorial_speed_levelset_expanded2688=21.2398. Δ=-14.1041 (▲ better), improved on 122/168 samples.

### Q3. Did candidateB_factorial_speed_levelset_expanded2688 improve gradient metrics?

- **Gradient MAE**: CNN=0.3491, candidateB_factorial_speed_levelset_expanded2688=0.3228. Δ=-0.0263 (▲ better), improved on 168/168 samples.
- **Gradient W1**: CNN=0.2329, candidateB_factorial_speed_levelset_expanded2688=0.2457. Δ=+0.0128 (▼ worse), improved on 46/168 samples.
- **Gradient kurtosis abs Δ**: CNN=3.7004, candidateB_factorial_speed_levelset_expanded2688=5.2715. Δ=+1.5711 (▼ worse), improved on 55/168 samples.

### Q4. Did candidateB_factorial_speed_levelset_expanded2688 improve exceedance metrics, especially s>5?

- **Exceedance abs Δ s>5**: CNN=0.00424, candidateB_factorial_speed_levelset_expanded2688=0.00686. Δ=+0.0026 (▼ worse), improved on 16/168 samples.
- **Exceedance abs Δ s>10**: CNN=0.00658, candidateB_factorial_speed_levelset_expanded2688=0.00448. Δ=-0.0021 (▲ better), improved on 112/168 samples.
- **Exceedance abs Δ s>15**: CNN=0.00621, candidateB_factorial_speed_levelset_expanded2688=0.00411. Δ=-0.0021 (▲ better), improved on 119/168 samples.
- **Exceedance abs Δ p90**: CNN=0.01025, candidateB_factorial_speed_levelset_expanded2688=0.00599. Δ=-0.0043 (▲ better), improved on 135/168 samples.

### Q5. Did candidateB_factorial_speed_levelset_expanded2688 move component-count behavior toward GT?

- **Component-count curve L1**: CNN=115.5278, candidateB_factorial_speed_levelset_expanded2688=141.4871. Δ=+25.9593 (▼ worse), improved on 5/168 samples.
- **Comp-count abs Δ t5**: CNN=64.3810, candidateB_factorial_speed_levelset_expanded2688=69.4048. Δ=+5.0238 (▼ worse), improved on 40/168 samples.
- **Comp-count abs Δ t10**: CNN=187.1667, candidateB_factorial_speed_levelset_expanded2688=230.2202. Δ=+43.0536 (▼ worse), improved on 2/168 samples.
- **Comp-count abs Δ t15**: CNN=116.8095, candidateB_factorial_speed_levelset_expanded2688=142.0714. Δ=+25.2619 (▼ worse), improved on 24/168 samples.

### Q6. Did candidateB_factorial_speed_levelset_expanded2688 improve PD or MT distances?

PD and MT distances for `candidateB_factorial_speed_levelset_expanded2688` are not available in this evaluation (TTK was not re-run on these outputs). Pre-computed values for CNN and GAN:

- **PD distance**: CNN=27.4063, GAN=20.8641, candidateB_factorial_speed_levelset_expanded2688=N/A (requires TTK run)
- **MT distance**: CNN=5.8678, GAN=8.3481, candidateB_factorial_speed_levelset_expanded2688=N/A (requires TTK run)

To compute PD/MT for `candidateB_factorial_speed_levelset_expanded2688`: convert outputs to VTI via `scripts/convert_phire_to_vti.py`, then run `scripts/compute_composite_tree_distance.py`.

### Q7. Did candidateB_factorial_speed_levelset_expanded2688 change the adjacent clusters (samples 10–13, 90–93, 162–163)?

See `adjacent_cluster_table_candidateB_factorial_speed_levelset_expanded2688.csv` for full detail. Summary of PSNRuv, speed MAE, and gradient MAE changes:

| Sample | Δ PSNRuv | Δ Speed MAE | Δ Grad MAE | Δ Exceed s>5 |
|---|---|---|---|---|
| 10 | 2.770 | -0.2231 | -0.0258 | 0.00484 |
| 11 | 2.659 | -0.2196 | -0.0272 | 0.00420 |
| 12 | 2.692 | -0.2194 | -0.0323 | 0.00410 |
| 13 | 2.625 | -0.2123 | -0.0300 | 0.00186 |
| 90 | 2.388 | -0.1553 | -0.0210 | 0.00003 |
| 91 | 2.418 | -0.1605 | -0.0206 | 0.00349 |
| 92 | 2.439 | -0.1738 | -0.0209 | 0.00842 |
| 93 | 2.420 | -0.1863 | -0.0185 | 0.00680 |
| 162 | 3.893 | -0.1604 | -0.0264 | -0.00488 |
| 163 | 3.836 | -0.1610 | -0.0286 | -0.00073 |

## Pairwise summary: CNN vs candidateB_factorial_speed_levelset_expanded2688

| Metric | Mean CNN | Mean candidateB_factorial_speed_levelset_expanded2688 | Δ | N improved | N worsened |
|---|---|---|---|---|---|
| psnruv | 31.1925 | 33.7789 | 2.5864 | 168 | 0 |
| ssim | N/A | N/A | N/A | 0 | 0 |
| speed_mae | 0.6941 | 0.4934 | -0.2007 | 168 | 0 |
| speed_rmse | 1.1078 | 0.8531 | -0.2547 | 168 | 0 |
| wpd_bias_abs | 35.3439 | 21.2398 | -14.1041 | 122 | 46 |
| wpd_mae | 231.6709 | 157.9165 | -73.7544 | 168 | 0 |
| wpd_w1 | 45.2713 | 31.8966 | -13.3747 | 137 | 31 |
| psd_log_l2 | 0.8335 | 1.1219 | 0.2884 | 0 | 168 |
| psd_slope_abs_delta | 0.9150 | 1.4492 | 0.5342 | 0 | 168 |
| grad_mae | 0.3491 | 0.3228 | -0.0263 | 168 | 0 |
| grad_w1 | 0.2329 | 0.2457 | 0.0128 | 46 | 122 |
| grad_kurtosis_abs_delta | 3.7004 | 5.2715 | 1.5711 | 55 | 113 |
| exceed_abs_t5 | 0.0042 | 0.0069 | 0.0026 | 16 | 152 |
| exceed_abs_t10 | 0.0066 | 0.0045 | -0.0021 | 112 | 56 |
| exceed_abs_t15 | 0.0062 | 0.0041 | -0.0021 | 119 | 49 |
| exceed_abs_p90 | 0.0103 | 0.0060 | -0.0043 | 135 | 33 |
| comp_curve_l1 | 115.5278 | 141.4871 | 25.9593 | 5 | 163 |
| pd_distance | N/A | N/A | N/A | 0 | 0 |
| mt_distance | N/A | N/A | N/A | 0 | 0 |

## Output files

- `ttk_runs_fixed/topology_finetuning/candidateB_factorial_speed_levelset_expanded2688_eval/all_sample_metrics_candidateB_factorial_speed_levelset_expanded2688.csv` — per-sample metrics for all methods
- `ttk_runs_fixed/topology_finetuning/candidateB_factorial_speed_levelset_expanded2688_eval/pairwise_cnn_vs_candidateB_factorial_speed_levelset_expanded2688.csv` — per-metric delta table
- `ttk_runs_fixed/topology_finetuning/candidateB_factorial_speed_levelset_expanded2688_eval/winner_counts_candidateB_factorial_speed_levelset_expanded2688.csv` — win counts per method
- `ttk_runs_fixed/topology_finetuning/candidateB_factorial_speed_levelset_expanded2688_eval/adjacent_cluster_table_candidateB_factorial_speed_levelset_expanded2688.csv` — adjacent-cluster detail
- `docs/topology_finetuning_candidateB_factorial_speed_levelset_expanded2688_eval.md` — this report

## Notes

- PSNRuv is computed on physical [u,v] arrays using per-sample dynamic data range (GT max − GT min).
- All speed-based metrics use scalar speed = sqrt(u² + v²) in physical units (m/s).
- Exceedance fractions use GT percentile thresholds (p90/p95/p99) computed per sample.
- Component counts use 8-connectivity on speed superlevel sets.
- PD/MT distances for CNN and GAN are loaded from pre-computed merged CSV; `candidateB_factorial_speed_levelset_expanded2688` requires a fresh TTK run.
- `candidateB_factorial_speed_levelset_expanded2688` was fine-tuned from the pretrained CNN checkpoint. Results may change with different lambda settings or more epochs.
