# candidateB_factorial_grad_levelset_expanded2688 fine-tuning evaluation

**Generated:** 2026-07-15

**Candidate:** `candidateB_factorial_grad_levelset_expanded2688`

**Samples evaluated:** 168

**Methods compared:** bicubic, cnn, gan, candidateB_factorial_grad_levelset_expanded2688

> **SSIM note:** SSIM was not computed because skimage could not be imported in the current NumPy environment (`ValueError('numpy.dtype size changed, may indicate binary incompatibility. Expected 96 from C header, got 88 from PyObject')`). All SSIM values are NaN.

> Note: PD/MT distances for `candidateB_factorial_grad_levelset_expanded2688` are N/A — TTK has not been re-run on these outputs. Values for CNN/GAN are loaded from the pre-computed merged CSV.

## Summary metrics table

| Metric | Bicubic | Cnn | Gan | Candidateb_factorial_grad_levelset_expanded2688 |
|---|---|---|---|---|
| PSNRuv (dB) | 32.2202 | 31.1925 | 29.1380 | 33.4634 |
| SSIM on speed | N/A | N/A | N/A | N/A |
| Speed MAE (m/s) | 0.5963 | 0.6941 | 0.9026 | 0.5185 |
| Speed RMSE (m/s) | 1.0156 | 1.1078 | 1.3775 | 0.8844 |
| WPD MAE (m³/s³) | 193.5403 | 231.6709 | 310.7328 | 166.1751 |
| WPD Wasserstein-1 | 55.5103 | 45.2713 | 85.6191 | 27.9677 |
| WPD bias abs (m³/s³) | 41.2244 | 35.3439 | 78.7236 | 22.3050 |
| Gradient MAE | 0.3696 | 0.3491 | 0.3806 | 0.3116 |
| Gradient W1 | 0.3282 | 0.2329 | 0.0564 | 0.1943 |
| Gradient kurtosis abs Δ | 6.6163 | 3.7004 | 4.2010 | 2.8487 |
| PSD log-L2 | 1.3625 | 0.8335 | 0.5139 | 0.9425 |
| PSD slope abs Δ | 1.6684 | 0.9150 | 0.9482 | 1.3018 |
| Exceedance abs Δ s>5 | 0.0095 | 0.0042 | 0.0082 | 0.0042 |
| Exceedance abs Δ s>10 | 0.0085 | 0.0066 | 0.0243 | 0.0041 |
| Exceedance abs Δ s>15 | 0.0074 | 0.0062 | 0.0096 | 0.0037 |
| Exceedance abs Δ p90 | 0.0109 | 0.0103 | 0.0123 | 0.0055 |
| Component-count curve L1 | 173.1855 | 115.5278 | 124.6567 | 109.3185 |
| PD distance (TTK) | N/A | 27.4063 | 20.8641 | N/A |
| MT distance (TTK) | N/A | 5.8678 | 8.3481 | N/A |

## 7 Key evaluation questions

### Q1. Did candidateB_factorial_grad_levelset_expanded2688 preserve CNN's direct-fidelity advantage?

- **PSNRuv**: CNN=31.1925, GAN=29.1380, candidateB_factorial_grad_levelset_expanded2688=33.4634. Δ(candidateB_factorial_grad_levelset_expanded2688−CNN)=+2.2709 (▲ better), improved on 168/168 samples.
- **SSIM**: CNN=N/A, GAN=N/A, candidateB_factorial_grad_levelset_expanded2688=N/A. Δ(candidateB_factorial_grad_levelset_expanded2688−CNN)=N/A, improved on 0/0 samples.
- **Speed MAE**: CNN=0.6941, GAN=0.9026, candidateB_factorial_grad_levelset_expanded2688=0.5185. Δ(candidateB_factorial_grad_levelset_expanded2688−CNN)=-0.1755 (▲ better), improved on 168/168 samples.
- **Speed RMSE**: CNN=1.1078, GAN=1.3775, candidateB_factorial_grad_levelset_expanded2688=0.8844. Δ(candidateB_factorial_grad_levelset_expanded2688−CNN)=-0.2234 (▲ better), improved on 168/168 samples.

### Q2. Did candidateB_factorial_grad_levelset_expanded2688 improve scalar speed or WPD metrics?

- **WPD MAE**: CNN=231.6709, candidateB_factorial_grad_levelset_expanded2688=166.1751. Δ=-65.4959 (▲ better), improved on 168/168 samples.
- **WPD W1**: CNN=45.2713, candidateB_factorial_grad_levelset_expanded2688=27.9677. Δ=-17.3036 (▲ better), improved on 128/168 samples.
- **WPD bias abs**: CNN=35.3439, candidateB_factorial_grad_levelset_expanded2688=22.3050. Δ=-13.0389 (▲ better), improved on 114/168 samples.

### Q3. Did candidateB_factorial_grad_levelset_expanded2688 improve gradient metrics?

- **Gradient MAE**: CNN=0.3491, candidateB_factorial_grad_levelset_expanded2688=0.3116. Δ=-0.0375 (▲ better), improved on 168/168 samples.
- **Gradient W1**: CNN=0.2329, candidateB_factorial_grad_levelset_expanded2688=0.1943. Δ=-0.0386 (▲ better), improved on 150/168 samples.
- **Gradient kurtosis abs Δ**: CNN=3.7004, candidateB_factorial_grad_levelset_expanded2688=2.8487. Δ=-0.8517 (▲ better), improved on 97/168 samples.

### Q4. Did candidateB_factorial_grad_levelset_expanded2688 improve exceedance metrics, especially s>5?

- **Exceedance abs Δ s>5**: CNN=0.00424, candidateB_factorial_grad_levelset_expanded2688=0.00416. Δ=-0.0001 (▲ better), improved on 94/168 samples.
- **Exceedance abs Δ s>10**: CNN=0.00658, candidateB_factorial_grad_levelset_expanded2688=0.00406. Δ=-0.0025 (▲ better), improved on 116/168 samples.
- **Exceedance abs Δ s>15**: CNN=0.00621, candidateB_factorial_grad_levelset_expanded2688=0.00374. Δ=-0.0025 (▲ better), improved on 117/168 samples.
- **Exceedance abs Δ p90**: CNN=0.01025, candidateB_factorial_grad_levelset_expanded2688=0.00549. Δ=-0.0048 (▲ better), improved on 131/168 samples.

### Q5. Did candidateB_factorial_grad_levelset_expanded2688 move component-count behavior toward GT?

- **Component-count curve L1**: CNN=115.5278, candidateB_factorial_grad_levelset_expanded2688=109.3185. Δ=-6.2093 (▲ better), improved on 121/168 samples.
- **Comp-count abs Δ t5**: CNN=64.3810, candidateB_factorial_grad_levelset_expanded2688=58.5952. Δ=-5.7857 (▲ better), improved on 119/168 samples.
- **Comp-count abs Δ t10**: CNN=187.1667, candidateB_factorial_grad_levelset_expanded2688=184.8036. Δ=-2.3631 (▲ better), improved on 89/168 samples.
- **Comp-count abs Δ t15**: CNN=116.8095, candidateB_factorial_grad_levelset_expanded2688=106.6845. Δ=-10.1250 (▲ better), improved on 122/168 samples.

### Q6. Did candidateB_factorial_grad_levelset_expanded2688 improve PD or MT distances?

PD and MT distances for `candidateB_factorial_grad_levelset_expanded2688` are not available in this evaluation (TTK was not re-run on these outputs). Pre-computed values for CNN and GAN:

- **PD distance**: CNN=27.4063, GAN=20.8641, candidateB_factorial_grad_levelset_expanded2688=N/A (requires TTK run)
- **MT distance**: CNN=5.8678, GAN=8.3481, candidateB_factorial_grad_levelset_expanded2688=N/A (requires TTK run)

To compute PD/MT for `candidateB_factorial_grad_levelset_expanded2688`: convert outputs to VTI via `scripts/convert_phire_to_vti.py`, then run `scripts/compute_composite_tree_distance.py`.

### Q7. Did candidateB_factorial_grad_levelset_expanded2688 change the adjacent clusters (samples 10–13, 90–93, 162–163)?

See `adjacent_cluster_table_candidateB_factorial_grad_levelset_expanded2688.csv` for full detail. Summary of PSNRuv, speed MAE, and gradient MAE changes:

| Sample | Δ PSNRuv | Δ Speed MAE | Δ Grad MAE | Δ Exceed s>5 |
|---|---|---|---|---|
| 10 | 2.444 | -0.1939 | -0.0377 | 0.00219 |
| 11 | 2.347 | -0.1907 | -0.0388 | 0.00102 |
| 12 | 2.364 | -0.1910 | -0.0429 | 0.00089 |
| 13 | 2.281 | -0.1839 | -0.0401 | -0.00072 |
| 90 | 2.100 | -0.1366 | -0.0298 | -0.00272 |
| 91 | 2.135 | -0.1411 | -0.0296 | 0.00143 |
| 92 | 2.129 | -0.1523 | -0.0323 | 0.00540 |
| 93 | 2.092 | -0.1618 | -0.0337 | 0.00251 |
| 162 | 3.588 | -0.1474 | -0.0260 | -0.00557 |
| 163 | 3.519 | -0.1480 | -0.0283 | -0.00285 |

## Pairwise summary: CNN vs candidateB_factorial_grad_levelset_expanded2688

| Metric | Mean CNN | Mean candidateB_factorial_grad_levelset_expanded2688 | Δ | N improved | N worsened |
|---|---|---|---|---|---|
| psnruv | 31.1925 | 33.4634 | 2.2709 | 168 | 0 |
| ssim | N/A | N/A | N/A | 0 | 0 |
| speed_mae | 0.6941 | 0.5185 | -0.1755 | 168 | 0 |
| speed_rmse | 1.1078 | 0.8844 | -0.2234 | 168 | 0 |
| wpd_bias_abs | 35.3439 | 22.3050 | -13.0389 | 114 | 54 |
| wpd_mae | 231.6709 | 166.1751 | -65.4959 | 168 | 0 |
| wpd_w1 | 45.2713 | 27.9677 | -17.3036 | 128 | 40 |
| psd_log_l2 | 0.8335 | 0.9425 | 0.1090 | 10 | 158 |
| psd_slope_abs_delta | 0.9150 | 1.3018 | 0.3868 | 0 | 168 |
| grad_mae | 0.3491 | 0.3116 | -0.0375 | 168 | 0 |
| grad_w1 | 0.2329 | 0.1943 | -0.0386 | 150 | 18 |
| grad_kurtosis_abs_delta | 3.7004 | 2.8487 | -0.8517 | 97 | 71 |
| exceed_abs_t5 | 0.0042 | 0.0042 | -0.0001 | 94 | 74 |
| exceed_abs_t10 | 0.0066 | 0.0041 | -0.0025 | 116 | 52 |
| exceed_abs_t15 | 0.0062 | 0.0037 | -0.0025 | 117 | 51 |
| exceed_abs_p90 | 0.0103 | 0.0055 | -0.0048 | 131 | 37 |
| comp_curve_l1 | 115.5278 | 109.3185 | -6.2093 | 121 | 46 |
| pd_distance | N/A | N/A | N/A | 0 | 0 |
| mt_distance | N/A | N/A | N/A | 0 | 0 |

## Output files

- `ttk_runs_fixed/topology_finetuning/candidateB_factorial_grad_levelset_expanded2688_eval/all_sample_metrics_candidateB_factorial_grad_levelset_expanded2688.csv` — per-sample metrics for all methods
- `ttk_runs_fixed/topology_finetuning/candidateB_factorial_grad_levelset_expanded2688_eval/pairwise_cnn_vs_candidateB_factorial_grad_levelset_expanded2688.csv` — per-metric delta table
- `ttk_runs_fixed/topology_finetuning/candidateB_factorial_grad_levelset_expanded2688_eval/winner_counts_candidateB_factorial_grad_levelset_expanded2688.csv` — win counts per method
- `ttk_runs_fixed/topology_finetuning/candidateB_factorial_grad_levelset_expanded2688_eval/adjacent_cluster_table_candidateB_factorial_grad_levelset_expanded2688.csv` — adjacent-cluster detail
- `docs/topology_finetuning_candidateB_factorial_grad_levelset_expanded2688_eval.md` — this report

## Notes

- PSNRuv is computed on physical [u,v] arrays using per-sample dynamic data range (GT max − GT min).
- All speed-based metrics use scalar speed = sqrt(u² + v²) in physical units (m/s).
- Exceedance fractions use GT percentile thresholds (p90/p95/p99) computed per sample.
- Component counts use 8-connectivity on speed superlevel sets.
- PD/MT distances for CNN and GAN are loaded from pre-computed merged CSV; `candidateB_factorial_grad_levelset_expanded2688` requires a fresh TTK run.
- `candidateB_factorial_grad_levelset_expanded2688` was fine-tuned from the pretrained CNN checkpoint. Results may change with different lambda settings or more epochs.
