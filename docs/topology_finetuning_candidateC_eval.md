# candidateC fine-tuning evaluation

**Generated:** 2026-05-12

**Candidate:** `candidateC`

**Samples evaluated:** 168

**Methods compared:** bicubic, cnn, gan, candidateC

> **SSIM note:** SSIM was not computed because skimage could not be imported in the current NumPy environment (`ValueError('numpy.dtype size changed, may indicate binary incompatibility. Expected 96 from C header, got 88 from PyObject')`). All SSIM values are NaN.

> Note: PD/MT distances for `candidateC` are N/A — TTK has not been re-run on these outputs. Values for CNN/GAN are loaded from the pre-computed merged CSV.

## Summary metrics table

| Metric | Bicubic | Cnn | Gan | Candidatec |
|---|---|---|---|---|
| PSNRuv (dB) | 32.2202 | 31.1925 | 29.1380 | 32.4344 |
| SSIM on speed | N/A | N/A | N/A | N/A |
| Speed MAE (m/s) | 0.5963 | 0.6941 | 0.9026 | 0.5951 |
| Speed RMSE (m/s) | 1.0156 | 1.1078 | 1.3775 | 0.9755 |
| WPD MAE (m³/s³) | 193.5403 | 231.6709 | 310.7328 | 195.1476 |
| WPD Wasserstein-1 | 55.5103 | 45.2713 | 85.6191 | 21.9732 |
| WPD bias abs (m³/s³) | 41.2244 | 35.3439 | 78.7236 | 9.7407 |
| Gradient MAE | 0.3696 | 0.3491 | 0.3806 | 0.3255 |
| Gradient W1 | 0.3282 | 0.2329 | 0.0564 | 0.2028 |
| Gradient kurtosis abs Δ | 6.6163 | 3.7004 | 4.2010 | 2.6852 |
| PSD log-L2 | 1.3625 | 0.8335 | 0.5139 | 0.9095 |
| PSD slope abs Δ | 1.6684 | 0.9150 | 0.9482 | 1.2550 |
| Exceedance abs Δ s>5 | 0.0095 | 0.0042 | 0.0082 | 0.0053 |
| Exceedance abs Δ s>10 | 0.0085 | 0.0066 | 0.0243 | 0.0063 |
| Exceedance abs Δ s>15 | 0.0074 | 0.0062 | 0.0096 | 0.0021 |
| Exceedance abs Δ p90 | 0.0109 | 0.0103 | 0.0123 | 0.0022 |
| Component-count curve L1 | 173.1855 | 115.5278 | 124.6567 | 99.6786 |
| PD distance (TTK) | N/A | 27.4063 | 20.8641 | N/A |
| MT distance (TTK) | N/A | 5.8678 | 8.3481 | N/A |

## 7 Key evaluation questions

### Q1. Did candidateC preserve CNN's direct-fidelity advantage?

- **PSNRuv**: CNN=31.1925, GAN=29.1380, candidateC=32.4344. Δ(candidateC−CNN)=+1.2419 (▲ better), improved on 168/168 samples.
- **SSIM**: CNN=N/A, GAN=N/A, candidateC=N/A. Δ(candidateC−CNN)=N/A, improved on 0/0 samples.
- **Speed MAE**: CNN=0.6941, GAN=0.9026, candidateC=0.5951. Δ(candidateC−CNN)=-0.0989 (▲ better), improved on 168/168 samples.
- **Speed RMSE**: CNN=1.1078, GAN=1.3775, candidateC=0.9755. Δ(candidateC−CNN)=-0.1323 (▲ better), improved on 168/168 samples.

### Q2. Did candidateC improve scalar speed or WPD metrics?

- **WPD MAE**: CNN=231.6709, candidateC=195.1476. Δ=-36.5233 (▲ better), improved on 168/168 samples.
- **WPD W1**: CNN=45.2713, candidateC=21.9732. Δ=-23.2981 (▲ better), improved on 160/168 samples.
- **WPD bias abs**: CNN=35.3439, candidateC=9.7407. Δ=-25.6033 (▲ better), improved on 153/168 samples.

### Q3. Did candidateC improve gradient metrics?

- **Gradient MAE**: CNN=0.3491, candidateC=0.3255. Δ=-0.0235 (▲ better), improved on 168/168 samples.
- **Gradient W1**: CNN=0.2329, candidateC=0.2028. Δ=-0.0301 (▲ better), improved on 150/168 samples.
- **Gradient kurtosis abs Δ**: CNN=3.7004, candidateC=2.6852. Δ=-1.0152 (▲ better), improved on 88/168 samples.

### Q4. Did candidateC improve exceedance metrics, especially s>5?

- **Exceedance abs Δ s>5**: CNN=0.00424, candidateC=0.00530. Δ=+0.0011 (▼ worse), improved on 61/168 samples.
- **Exceedance abs Δ s>10**: CNN=0.00658, candidateC=0.00635. Δ=-0.0002 (▲ better), improved on 80/168 samples.
- **Exceedance abs Δ s>15**: CNN=0.00621, candidateC=0.00206. Δ=-0.0042 (▲ better), improved on 136/168 samples.
- **Exceedance abs Δ p90**: CNN=0.01025, candidateC=0.00218. Δ=-0.0081 (▲ better), improved on 146/168 samples.

### Q5. Did candidateC move component-count behavior toward GT?

- **Component-count curve L1**: CNN=115.5278, candidateC=99.6786. Δ=-15.8492 (▲ better), improved on 151/168 samples.
- **Comp-count abs Δ t5**: CNN=64.3810, candidateC=56.5595. Δ=-7.8214 (▲ better), improved on 112/168 samples.
- **Comp-count abs Δ t10**: CNN=187.1667, candidateC=180.5833. Δ=-6.5833 (▲ better), improved on 101/168 samples.
- **Comp-count abs Δ t15**: CNN=116.8095, candidateC=93.7083. Δ=-23.1012 (▲ better), improved on 144/168 samples.

### Q6. Did candidateC improve PD or MT distances?

PD and MT distances for `candidateC` are not available in this evaluation (TTK was not re-run on these outputs). Pre-computed values for CNN and GAN:

- **PD distance**: CNN=27.4063, GAN=20.8641, candidateC=N/A (requires TTK run)
- **MT distance**: CNN=5.8678, GAN=8.3481, candidateC=N/A (requires TTK run)

To compute PD/MT for `candidateC`: convert outputs to VTI via `scripts/convert_phire_to_vti.py`, then run `scripts/compute_composite_tree_distance.py`.

### Q7. Did candidateC change the adjacent clusters (samples 10–13, 90–93, 162–163)?

See `adjacent_cluster_table_candidateC.csv` for full detail. Summary of PSNRuv, speed MAE, and gradient MAE changes:

| Sample | Δ PSNRuv | Δ Speed MAE | Δ Grad MAE | Δ Exceed s>5 |
|---|---|---|---|---|
| 10 | 1.391 | -0.1114 | -0.0235 | 0.00135 |
| 11 | 1.269 | -0.1050 | -0.0227 | 0.00003 |
| 12 | 1.316 | -0.1053 | -0.0260 | -0.00037 |
| 13 | 1.239 | -0.0970 | -0.0219 | -0.00216 |
| 90 | 1.123 | -0.0734 | -0.0195 | 0.00202 |
| 91 | 1.189 | -0.0790 | -0.0192 | 0.00620 |
| 92 | 1.200 | -0.0889 | -0.0221 | 0.00984 |
| 93 | 1.121 | -0.0925 | -0.0230 | 0.00646 |
| 162 | 1.833 | -0.0685 | -0.0127 | 0.00202 |
| 163 | 1.828 | -0.0693 | -0.0147 | 0.00796 |

## Pairwise summary: CNN vs candidateC

| Metric | Mean CNN | Mean candidateC | Δ | N improved | N worsened |
|---|---|---|---|---|---|
| psnruv | 31.1925 | 32.4344 | 1.2419 | 168 | 0 |
| ssim | N/A | N/A | N/A | 0 | 0 |
| speed_mae | 0.6941 | 0.5951 | -0.0989 | 168 | 0 |
| speed_rmse | 1.1078 | 0.9755 | -0.1323 | 168 | 0 |
| wpd_bias_abs | 35.3439 | 9.7407 | -25.6033 | 153 | 15 |
| wpd_mae | 231.6709 | 195.1476 | -36.5233 | 168 | 0 |
| wpd_w1 | 45.2713 | 21.9732 | -23.2981 | 160 | 8 |
| psd_log_l2 | 0.8335 | 0.9095 | 0.0760 | 35 | 133 |
| psd_slope_abs_delta | 0.9150 | 1.2550 | 0.3400 | 0 | 168 |
| grad_mae | 0.3491 | 0.3255 | -0.0235 | 168 | 0 |
| grad_w1 | 0.2329 | 0.2028 | -0.0301 | 150 | 18 |
| grad_kurtosis_abs_delta | 3.7004 | 2.6852 | -1.0152 | 88 | 80 |
| exceed_abs_t5 | 0.0042 | 0.0053 | 0.0011 | 61 | 107 |
| exceed_abs_t10 | 0.0066 | 0.0063 | -0.0002 | 80 | 88 |
| exceed_abs_t15 | 0.0062 | 0.0021 | -0.0042 | 136 | 32 |
| exceed_abs_p90 | 0.0103 | 0.0022 | -0.0081 | 146 | 21 |
| comp_curve_l1 | 115.5278 | 99.6786 | -15.8492 | 151 | 15 |
| pd_distance | N/A | N/A | N/A | 0 | 0 |
| mt_distance | N/A | N/A | N/A | 0 | 0 |

## Output files

- `ttk_runs_fixed/topology_finetuning/candidateC_eval/all_sample_metrics_candidateC.csv` — per-sample metrics for all methods
- `ttk_runs_fixed/topology_finetuning/candidateC_eval/pairwise_cnn_vs_candidateC.csv` — per-metric delta table
- `ttk_runs_fixed/topology_finetuning/candidateC_eval/winner_counts_candidateC.csv` — win counts per method
- `ttk_runs_fixed/topology_finetuning/candidateC_eval/adjacent_cluster_table_candidateC.csv` — adjacent-cluster detail
- `docs/topology_finetuning_candidateC_eval.md` — this report

## Notes

- PSNRuv is computed on physical [u,v] arrays using per-sample dynamic data range (GT max − GT min).
- All speed-based metrics use scalar speed = sqrt(u² + v²) in physical units (m/s).
- Exceedance fractions use GT percentile thresholds (p90/p95/p99) computed per sample.
- Component counts use 8-connectivity on speed superlevel sets.
- PD/MT distances for CNN and GAN are loaded from pre-computed merged CSV; `candidateC` requires a fresh TTK run.
- `candidateC` was fine-tuned from the pretrained CNN checkpoint. Results may change with different lambda settings or more epochs.
