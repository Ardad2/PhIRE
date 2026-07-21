# candidateC_expanded1344 fine-tuning evaluation

**Generated:** 2026-05-29

**Candidate:** `candidateC_expanded1344`

**Samples evaluated:** 168

**Methods compared:** bicubic, cnn, gan, candidateC_expanded1344

> **SSIM note:** SSIM was not computed because skimage could not be imported in the current NumPy environment (`ValueError('numpy.dtype size changed, may indicate binary incompatibility. Expected 96 from C header, got 88 from PyObject')`). All SSIM values are NaN.

> Note: PD/MT distances for `candidateC_expanded1344` are N/A — TTK has not been re-run on these outputs. Values for CNN/GAN are loaded from the pre-computed merged CSV.

## Summary metrics table

| Metric | Bicubic | Cnn | Gan | Candidatec_expanded1344 |
|---|---|---|---|---|
| PSNRuv (dB) | 32.2202 | 31.1925 | 29.1380 | 33.2128 |
| SSIM on speed | N/A | N/A | N/A | N/A |
| Speed MAE (m/s) | 0.5963 | 0.6941 | 0.9026 | 0.5330 |
| Speed RMSE (m/s) | 1.0156 | 1.1078 | 1.3775 | 0.9036 |
| WPD MAE (m³/s³) | 193.5403 | 231.6709 | 310.7328 | 173.2167 |
| WPD Wasserstein-1 | 55.5103 | 45.2713 | 85.6191 | 16.9219 |
| WPD bias abs (m³/s³) | 41.2244 | 35.3439 | 78.7236 | 7.6499 |
| Gradient MAE | 0.3696 | 0.3491 | 0.3806 | 0.3139 |
| Gradient W1 | 0.3282 | 0.2329 | 0.0564 | 0.1934 |
| Gradient kurtosis abs Δ | 6.6163 | 3.7004 | 4.2010 | 2.2152 |
| PSD log-L2 | 1.3625 | 0.8335 | 0.5139 | 0.9574 |
| PSD slope abs Δ | 1.6684 | 0.9150 | 0.9482 | 1.3600 |
| Exceedance abs Δ s>5 | 0.0095 | 0.0042 | 0.0082 | 0.0059 |
| Exceedance abs Δ s>10 | 0.0085 | 0.0066 | 0.0243 | 0.0047 |
| Exceedance abs Δ s>15 | 0.0074 | 0.0062 | 0.0096 | 0.0013 |
| Exceedance abs Δ p90 | 0.0109 | 0.0103 | 0.0123 | 0.0016 |
| Component-count curve L1 | 173.1855 | 115.5278 | 124.6567 | 107.5437 |
| PD distance (TTK) | N/A | 27.4063 | 20.8641 | N/A |
| MT distance (TTK) | N/A | 5.8678 | 8.3481 | N/A |

## 7 Key evaluation questions

### Q1. Did candidateC_expanded1344 preserve CNN's direct-fidelity advantage?

- **PSNRuv**: CNN=31.1925, GAN=29.1380, candidateC_expanded1344=33.2128. Δ(candidateC_expanded1344−CNN)=+2.0203 (▲ better), improved on 168/168 samples.
- **SSIM**: CNN=N/A, GAN=N/A, candidateC_expanded1344=N/A. Δ(candidateC_expanded1344−CNN)=N/A, improved on 0/0 samples.
- **Speed MAE**: CNN=0.6941, GAN=0.9026, candidateC_expanded1344=0.5330. Δ(candidateC_expanded1344−CNN)=-0.1611 (▲ better), improved on 168/168 samples.
- **Speed RMSE**: CNN=1.1078, GAN=1.3775, candidateC_expanded1344=0.9036. Δ(candidateC_expanded1344−CNN)=-0.2042 (▲ better), improved on 168/168 samples.

### Q2. Did candidateC_expanded1344 improve scalar speed or WPD metrics?

- **WPD MAE**: CNN=231.6709, candidateC_expanded1344=173.2167. Δ=-58.4542 (▲ better), improved on 168/168 samples.
- **WPD W1**: CNN=45.2713, candidateC_expanded1344=16.9219. Δ=-28.3494 (▲ better), improved on 164/168 samples.
- **WPD bias abs**: CNN=35.3439, candidateC_expanded1344=7.6499. Δ=-27.6940 (▲ better), improved on 156/168 samples.

### Q3. Did candidateC_expanded1344 improve gradient metrics?

- **Gradient MAE**: CNN=0.3491, candidateC_expanded1344=0.3139. Δ=-0.0352 (▲ better), improved on 168/168 samples.
- **Gradient W1**: CNN=0.2329, candidateC_expanded1344=0.1934. Δ=-0.0395 (▲ better), improved on 153/168 samples.
- **Gradient kurtosis abs Δ**: CNN=3.7004, candidateC_expanded1344=2.2152. Δ=-1.4852 (▲ better), improved on 101/168 samples.

### Q4. Did candidateC_expanded1344 improve exceedance metrics, especially s>5?

- **Exceedance abs Δ s>5**: CNN=0.00424, candidateC_expanded1344=0.00586. Δ=+0.0016 (▼ worse), improved on 39/168 samples.
- **Exceedance abs Δ s>10**: CNN=0.00658, candidateC_expanded1344=0.00473. Δ=-0.0018 (▲ better), improved on 94/168 samples.
- **Exceedance abs Δ s>15**: CNN=0.00621, candidateC_expanded1344=0.00129. Δ=-0.0049 (▲ better), improved on 138/168 samples.
- **Exceedance abs Δ p90**: CNN=0.01025, candidateC_expanded1344=0.00158. Δ=-0.0087 (▲ better), improved on 154/168 samples.

### Q5. Did candidateC_expanded1344 move component-count behavior toward GT?

- **Component-count curve L1**: CNN=115.5278, candidateC_expanded1344=107.5437. Δ=-7.9841 (▲ better), improved on 131/168 samples.
- **Comp-count abs Δ t5**: CNN=64.3810, candidateC_expanded1344=61.3036. Δ=-3.0774 (▲ better), improved on 99/168 samples.
- **Comp-count abs Δ t10**: CNN=187.1667, candidateC_expanded1344=193.8155. Δ=+6.6488 (▼ worse), improved on 64/168 samples.
- **Comp-count abs Δ t15**: CNN=116.8095, candidateC_expanded1344=99.7321. Δ=-17.0774 (▲ better), improved on 128/168 samples.

### Q6. Did candidateC_expanded1344 improve PD or MT distances?

PD and MT distances for `candidateC_expanded1344` are not available in this evaluation (TTK was not re-run on these outputs). Pre-computed values for CNN and GAN:

- **PD distance**: CNN=27.4063, GAN=20.8641, candidateC_expanded1344=N/A (requires TTK run)
- **MT distance**: CNN=5.8678, GAN=8.3481, candidateC_expanded1344=N/A (requires TTK run)

To compute PD/MT for `candidateC_expanded1344`: convert outputs to VTI via `scripts/convert_phire_to_vti.py`, then run `scripts/compute_composite_tree_distance.py`.

### Q7. Did candidateC_expanded1344 change the adjacent clusters (samples 10–13, 90–93, 162–163)?

See `adjacent_cluster_table_candidateC_expanded1344.csv` for full detail. Summary of PSNRuv, speed MAE, and gradient MAE changes:

| Sample | Δ PSNRuv | Δ Speed MAE | Δ Grad MAE | Δ Exceed s>5 |
|---|---|---|---|---|
| 10 | 2.205 | -0.1764 | -0.0333 | 0.00268 |
| 11 | 2.093 | -0.1722 | -0.0339 | 0.00227 |
| 12 | 2.119 | -0.1732 | -0.0380 | 0.00300 |
| 13 | 2.025 | -0.1648 | -0.0348 | 0.00048 |
| 90 | 1.899 | -0.1249 | -0.0269 | 0.00171 |
| 91 | 1.912 | -0.1293 | -0.0266 | 0.00618 |
| 92 | 1.871 | -0.1379 | -0.0296 | 0.00971 |
| 93 | 1.798 | -0.1463 | -0.0305 | 0.00546 |
| 162 | 3.293 | -0.1360 | -0.0241 | -0.00547 |
| 163 | 3.222 | -0.1366 | -0.0263 | -0.00210 |

## Pairwise summary: CNN vs candidateC_expanded1344

| Metric | Mean CNN | Mean candidateC_expanded1344 | Δ | N improved | N worsened |
|---|---|---|---|---|---|
| psnruv | 31.1925 | 33.2128 | 2.0203 | 168 | 0 |
| ssim | N/A | N/A | N/A | 0 | 0 |
| speed_mae | 0.6941 | 0.5330 | -0.1611 | 168 | 0 |
| speed_rmse | 1.1078 | 0.9036 | -0.2042 | 168 | 0 |
| wpd_bias_abs | 35.3439 | 7.6499 | -27.6940 | 156 | 12 |
| wpd_mae | 231.6709 | 173.2167 | -58.4542 | 168 | 0 |
| wpd_w1 | 45.2713 | 16.9219 | -28.3494 | 164 | 4 |
| psd_log_l2 | 0.8335 | 0.9574 | 0.1239 | 7 | 161 |
| psd_slope_abs_delta | 0.9150 | 1.3600 | 0.4450 | 0 | 168 |
| grad_mae | 0.3491 | 0.3139 | -0.0352 | 168 | 0 |
| grad_w1 | 0.2329 | 0.1934 | -0.0395 | 153 | 15 |
| grad_kurtosis_abs_delta | 3.7004 | 2.2152 | -1.4852 | 101 | 67 |
| exceed_abs_t5 | 0.0042 | 0.0059 | 0.0016 | 39 | 129 |
| exceed_abs_t10 | 0.0066 | 0.0047 | -0.0018 | 94 | 74 |
| exceed_abs_t15 | 0.0062 | 0.0013 | -0.0049 | 138 | 30 |
| exceed_abs_p90 | 0.0103 | 0.0016 | -0.0087 | 154 | 14 |
| comp_curve_l1 | 115.5278 | 107.5437 | -7.9841 | 131 | 36 |
| pd_distance | N/A | N/A | N/A | 0 | 0 |
| mt_distance | N/A | N/A | N/A | 0 | 0 |

## Output files

- `ttk_runs_fixed/topology_finetuning/candidateC_expanded1344_eval/all_sample_metrics_candidateC_expanded1344.csv` — per-sample metrics for all methods
- `ttk_runs_fixed/topology_finetuning/candidateC_expanded1344_eval/pairwise_cnn_vs_candidateC_expanded1344.csv` — per-metric delta table
- `ttk_runs_fixed/topology_finetuning/candidateC_expanded1344_eval/winner_counts_candidateC_expanded1344.csv` — win counts per method
- `ttk_runs_fixed/topology_finetuning/candidateC_expanded1344_eval/adjacent_cluster_table_candidateC_expanded1344.csv` — adjacent-cluster detail
- `docs/topology_finetuning_candidateC_expanded1344_eval.md` — this report

## Notes

- PSNRuv is computed on physical [u,v] arrays using per-sample dynamic data range (GT max − GT min).
- All speed-based metrics use scalar speed = sqrt(u² + v²) in physical units (m/s).
- Exceedance fractions use GT percentile thresholds (p90/p95/p99) computed per sample.
- Component counts use 8-connectivity on speed superlevel sets.
- PD/MT distances for CNN and GAN are loaded from pre-computed merged CSV; `candidateC_expanded1344` requires a fresh TTK run.
- `candidateC_expanded1344` was fine-tuned from the pretrained CNN checkpoint. Results may change with different lambda settings or more epochs.
