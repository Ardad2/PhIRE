# candidateB_factorial_levelset_expanded2688 fine-tuning evaluation

**Generated:** 2026-07-15

**Candidate:** `candidateB_factorial_levelset_expanded2688`

**Samples evaluated:** 168

**Methods compared:** bicubic, cnn, gan, candidateB_factorial_levelset_expanded2688

> **SSIM note:** SSIM was not computed because skimage could not be imported in the current NumPy environment (`ValueError('numpy.dtype size changed, may indicate binary incompatibility. Expected 96 from C header, got 88 from PyObject')`). All SSIM values are NaN.

> Note: PD/MT distances for `candidateB_factorial_levelset_expanded2688` are N/A — TTK has not been re-run on these outputs. Values for CNN/GAN are loaded from the pre-computed merged CSV.

## Summary metrics table

| Metric | Bicubic | Cnn | Gan | Candidateb_factorial_levelset_expanded2688 |
|---|---|---|---|---|
| PSNRuv (dB) | 32.2202 | 31.1925 | 29.1380 | 33.7886 |
| SSIM on speed | N/A | N/A | N/A | N/A |
| Speed MAE (m/s) | 0.5963 | 0.6941 | 0.9026 | 0.4937 |
| Speed RMSE (m/s) | 1.0156 | 1.1078 | 1.3775 | 0.8542 |
| WPD MAE (m³/s³) | 193.5403 | 231.6709 | 310.7328 | 158.0550 |
| WPD Wasserstein-1 | 55.5103 | 45.2713 | 85.6191 | 33.1943 |
| WPD bias abs (m³/s³) | 41.2244 | 35.3439 | 78.7236 | 23.2017 |
| Gradient MAE | 0.3696 | 0.3491 | 0.3806 | 0.3240 |
| Gradient W1 | 0.3282 | 0.2329 | 0.0564 | 0.2492 |
| Gradient kurtosis abs Δ | 6.6163 | 3.7004 | 4.2010 | 5.2613 |
| PSD log-L2 | 1.3625 | 0.8335 | 0.5139 | 1.1331 |
| PSD slope abs Δ | 1.6684 | 0.9150 | 0.9482 | 1.4633 |
| Exceedance abs Δ s>5 | 0.0095 | 0.0042 | 0.0082 | 0.0065 |
| Exceedance abs Δ s>10 | 0.0085 | 0.0066 | 0.0243 | 0.0045 |
| Exceedance abs Δ s>15 | 0.0074 | 0.0062 | 0.0096 | 0.0043 |
| Exceedance abs Δ p90 | 0.0109 | 0.0103 | 0.0123 | 0.0062 |
| Component-count curve L1 | 173.1855 | 115.5278 | 124.6567 | 143.0357 |
| PD distance (TTK) | N/A | 27.4063 | 20.8641 | N/A |
| MT distance (TTK) | N/A | 5.8678 | 8.3481 | N/A |

## 7 Key evaluation questions

### Q1. Did candidateB_factorial_levelset_expanded2688 preserve CNN's direct-fidelity advantage?

- **PSNRuv**: CNN=31.1925, GAN=29.1380, candidateB_factorial_levelset_expanded2688=33.7886. Δ(candidateB_factorial_levelset_expanded2688−CNN)=+2.5961 (▲ better), improved on 168/168 samples.
- **SSIM**: CNN=N/A, GAN=N/A, candidateB_factorial_levelset_expanded2688=N/A. Δ(candidateB_factorial_levelset_expanded2688−CNN)=N/A, improved on 0/0 samples.
- **Speed MAE**: CNN=0.6941, GAN=0.9026, candidateB_factorial_levelset_expanded2688=0.4937. Δ(candidateB_factorial_levelset_expanded2688−CNN)=-0.2004 (▲ better), improved on 168/168 samples.
- **Speed RMSE**: CNN=1.1078, GAN=1.3775, candidateB_factorial_levelset_expanded2688=0.8542. Δ(candidateB_factorial_levelset_expanded2688−CNN)=-0.2536 (▲ better), improved on 168/168 samples.

### Q2. Did candidateB_factorial_levelset_expanded2688 improve scalar speed or WPD metrics?

- **WPD MAE**: CNN=231.6709, candidateB_factorial_levelset_expanded2688=158.0550. Δ=-73.6159 (▲ better), improved on 168/168 samples.
- **WPD W1**: CNN=45.2713, candidateB_factorial_levelset_expanded2688=33.1943. Δ=-12.0770 (▲ better), improved on 130/168 samples.
- **WPD bias abs**: CNN=35.3439, candidateB_factorial_levelset_expanded2688=23.2017. Δ=-12.1422 (▲ better), improved on 120/168 samples.

### Q3. Did candidateB_factorial_levelset_expanded2688 improve gradient metrics?

- **Gradient MAE**: CNN=0.3491, candidateB_factorial_levelset_expanded2688=0.3240. Δ=-0.0250 (▲ better), improved on 168/168 samples.
- **Gradient W1**: CNN=0.2329, candidateB_factorial_levelset_expanded2688=0.2492. Δ=+0.0162 (▼ worse), improved on 33/168 samples.
- **Gradient kurtosis abs Δ**: CNN=3.7004, candidateB_factorial_levelset_expanded2688=5.2613. Δ=+1.5610 (▼ worse), improved on 57/168 samples.

### Q4. Did candidateB_factorial_levelset_expanded2688 improve exceedance metrics, especially s>5?

- **Exceedance abs Δ s>5**: CNN=0.00424, candidateB_factorial_levelset_expanded2688=0.00652. Δ=+0.0023 (▼ worse), improved on 22/168 samples.
- **Exceedance abs Δ s>10**: CNN=0.00658, candidateB_factorial_levelset_expanded2688=0.00447. Δ=-0.0021 (▲ better), improved on 112/168 samples.
- **Exceedance abs Δ s>15**: CNN=0.00621, candidateB_factorial_levelset_expanded2688=0.00433. Δ=-0.0019 (▲ better), improved on 117/168 samples.
- **Exceedance abs Δ p90**: CNN=0.01025, candidateB_factorial_levelset_expanded2688=0.00620. Δ=-0.0041 (▲ better), improved on 136/168 samples.

### Q5. Did candidateB_factorial_levelset_expanded2688 move component-count behavior toward GT?

- **Component-count curve L1**: CNN=115.5278, candidateB_factorial_levelset_expanded2688=143.0357. Δ=+27.5079 (▼ worse), improved on 4/168 samples.
- **Comp-count abs Δ t5**: CNN=64.3810, candidateB_factorial_levelset_expanded2688=69.6845. Δ=+5.3036 (▼ worse), improved on 37/168 samples.
- **Comp-count abs Δ t10**: CNN=187.1667, candidateB_factorial_levelset_expanded2688=232.0655. Δ=+44.8988 (▼ worse), improved on 2/168 samples.
- **Comp-count abs Δ t15**: CNN=116.8095, candidateB_factorial_levelset_expanded2688=144.1071. Δ=+27.2976 (▼ worse), improved on 23/168 samples.

### Q6. Did candidateB_factorial_levelset_expanded2688 improve PD or MT distances?

PD and MT distances for `candidateB_factorial_levelset_expanded2688` are not available in this evaluation (TTK was not re-run on these outputs). Pre-computed values for CNN and GAN:

- **PD distance**: CNN=27.4063, GAN=20.8641, candidateB_factorial_levelset_expanded2688=N/A (requires TTK run)
- **MT distance**: CNN=5.8678, GAN=8.3481, candidateB_factorial_levelset_expanded2688=N/A (requires TTK run)

To compute PD/MT for `candidateB_factorial_levelset_expanded2688`: convert outputs to VTI via `scripts/convert_phire_to_vti.py`, then run `scripts/compute_composite_tree_distance.py`.

### Q7. Did candidateB_factorial_levelset_expanded2688 change the adjacent clusters (samples 10–13, 90–93, 162–163)?

See `adjacent_cluster_table_candidateB_factorial_levelset_expanded2688.csv` for full detail. Summary of PSNRuv, speed MAE, and gradient MAE changes:

| Sample | Δ PSNRuv | Δ Speed MAE | Δ Grad MAE | Δ Exceed s>5 |
|---|---|---|---|---|
| 10 | 2.777 | -0.2227 | -0.0234 | 0.00454 |
| 11 | 2.670 | -0.2193 | -0.0251 | 0.00384 |
| 12 | 2.707 | -0.2191 | -0.0302 | 0.00340 |
| 13 | 2.642 | -0.2119 | -0.0281 | 0.00107 |
| 90 | 2.394 | -0.1552 | -0.0203 | -0.00037 |
| 91 | 2.425 | -0.1603 | -0.0200 | 0.00336 |
| 92 | 2.448 | -0.1735 | -0.0200 | 0.00798 |
| 93 | 2.430 | -0.1858 | -0.0173 | 0.00648 |
| 162 | 3.915 | -0.1608 | -0.0257 | -0.00526 |
| 163 | 3.849 | -0.1610 | -0.0277 | -0.00126 |

## Pairwise summary: CNN vs candidateB_factorial_levelset_expanded2688

| Metric | Mean CNN | Mean candidateB_factorial_levelset_expanded2688 | Δ | N improved | N worsened |
|---|---|---|---|---|---|
| psnruv | 31.1925 | 33.7886 | 2.5961 | 168 | 0 |
| ssim | N/A | N/A | N/A | 0 | 0 |
| speed_mae | 0.6941 | 0.4937 | -0.2004 | 168 | 0 |
| speed_rmse | 1.1078 | 0.8542 | -0.2536 | 168 | 0 |
| wpd_bias_abs | 35.3439 | 23.2017 | -12.1422 | 120 | 48 |
| wpd_mae | 231.6709 | 158.0550 | -73.6159 | 168 | 0 |
| wpd_w1 | 45.2713 | 33.1943 | -12.0770 | 130 | 38 |
| psd_log_l2 | 0.8335 | 1.1331 | 0.2995 | 0 | 168 |
| psd_slope_abs_delta | 0.9150 | 1.4633 | 0.5483 | 0 | 168 |
| grad_mae | 0.3491 | 0.3240 | -0.0250 | 168 | 0 |
| grad_w1 | 0.2329 | 0.2492 | 0.0162 | 33 | 135 |
| grad_kurtosis_abs_delta | 3.7004 | 5.2613 | 1.5610 | 57 | 111 |
| exceed_abs_t5 | 0.0042 | 0.0065 | 0.0023 | 22 | 146 |
| exceed_abs_t10 | 0.0066 | 0.0045 | -0.0021 | 112 | 56 |
| exceed_abs_t15 | 0.0062 | 0.0043 | -0.0019 | 117 | 51 |
| exceed_abs_p90 | 0.0103 | 0.0062 | -0.0041 | 136 | 32 |
| comp_curve_l1 | 115.5278 | 143.0357 | 27.5079 | 4 | 164 |
| pd_distance | N/A | N/A | N/A | 0 | 0 |
| mt_distance | N/A | N/A | N/A | 0 | 0 |

## Output files

- `ttk_runs_fixed/topology_finetuning/candidateB_factorial_levelset_expanded2688_eval/all_sample_metrics_candidateB_factorial_levelset_expanded2688.csv` — per-sample metrics for all methods
- `ttk_runs_fixed/topology_finetuning/candidateB_factorial_levelset_expanded2688_eval/pairwise_cnn_vs_candidateB_factorial_levelset_expanded2688.csv` — per-metric delta table
- `ttk_runs_fixed/topology_finetuning/candidateB_factorial_levelset_expanded2688_eval/winner_counts_candidateB_factorial_levelset_expanded2688.csv` — win counts per method
- `ttk_runs_fixed/topology_finetuning/candidateB_factorial_levelset_expanded2688_eval/adjacent_cluster_table_candidateB_factorial_levelset_expanded2688.csv` — adjacent-cluster detail
- `docs/topology_finetuning_candidateB_factorial_levelset_expanded2688_eval.md` — this report

## Notes

- PSNRuv is computed on physical [u,v] arrays using per-sample dynamic data range (GT max − GT min).
- All speed-based metrics use scalar speed = sqrt(u² + v²) in physical units (m/s).
- Exceedance fractions use GT percentile thresholds (p90/p95/p99) computed per sample.
- Component counts use 8-connectivity on speed superlevel sets.
- PD/MT distances for CNN and GAN are loaded from pre-computed merged CSV; `candidateB_factorial_levelset_expanded2688` requires a fresh TTK run.
- `candidateB_factorial_levelset_expanded2688` was fine-tuned from the pretrained CNN checkpoint. Results may change with different lambda settings or more epochs.
