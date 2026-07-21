# candidateE fine-tuning evaluation

**Generated:** 2026-05-20

**Candidate:** `candidateE`

**Samples evaluated:** 168

**Methods compared:** bicubic, cnn, gan, candidateE

> **SSIM note:** SSIM was not computed because skimage could not be imported in the current NumPy environment (`ValueError('numpy.dtype size changed, may indicate binary incompatibility. Expected 96 from C header, got 88 from PyObject')`). All SSIM values are NaN.

> Note: PD/MT distances for `candidateE` are N/A — TTK has not been re-run on these outputs. Values for CNN/GAN are loaded from the pre-computed merged CSV.

## Summary metrics table

| Metric | Bicubic | Cnn | Gan | Candidatee |
|---|---|---|---|---|
| PSNRuv (dB) | 32.2202 | 31.1925 | 29.1380 | 31.3765 |
| SSIM on speed | N/A | N/A | N/A | N/A |
| Speed MAE (m/s) | 0.5963 | 0.6941 | 0.9026 | 0.6816 |
| Speed RMSE (m/s) | 1.0156 | 1.1078 | 1.3775 | 1.0816 |
| WPD MAE (m³/s³) | 193.5403 | 231.6709 | 310.7328 | 226.8054 |
| WPD Wasserstein-1 | 55.5103 | 45.2713 | 85.6191 | 50.5029 |
| WPD bias abs (m³/s³) | 41.2244 | 35.3439 | 78.7236 | 37.7964 |
| Gradient MAE | 0.3696 | 0.3491 | 0.3806 | 0.3544 |
| Gradient W1 | 0.3282 | 0.2329 | 0.0564 | 0.2607 |
| Gradient kurtosis abs Δ | 6.6163 | 3.7004 | 4.2010 | 3.7305 |
| PSD log-L2 | 1.3625 | 0.8335 | 0.5139 | 0.8786 |
| PSD slope abs Δ | 1.6684 | 0.9150 | 0.9482 | 1.0210 |
| Exceedance abs Δ s>5 | 0.0095 | 0.0042 | 0.0082 | 0.0057 |
| Exceedance abs Δ s>10 | 0.0085 | 0.0066 | 0.0243 | 0.0067 |
| Exceedance abs Δ s>15 | 0.0074 | 0.0062 | 0.0096 | 0.0074 |
| Exceedance abs Δ p90 | 0.0109 | 0.0103 | 0.0123 | 0.0109 |
| Component-count curve L1 | 173.1855 | 115.5278 | 124.6567 | 125.6260 |
| PD distance (TTK) | N/A | 27.4063 | 20.8641 | N/A |
| MT distance (TTK) | N/A | 5.8678 | 8.3481 | N/A |

## 7 Key evaluation questions

### Q1. Did candidateE preserve CNN's direct-fidelity advantage?

- **PSNRuv**: CNN=31.1925, GAN=29.1380, candidateE=31.3765. Δ(candidateE−CNN)=+0.1840 (▲ better), improved on 167/168 samples.
- **SSIM**: CNN=N/A, GAN=N/A, candidateE=N/A. Δ(candidateE−CNN)=N/A, improved on 0/0 samples.
- **Speed MAE**: CNN=0.6941, GAN=0.9026, candidateE=0.6816. Δ(candidateE−CNN)=-0.0125 (▲ better), improved on 164/168 samples.
- **Speed RMSE**: CNN=1.1078, GAN=1.3775, candidateE=1.0816. Δ(candidateE−CNN)=-0.0261 (▲ better), improved on 167/168 samples.

### Q2. Did candidateE improve scalar speed or WPD metrics?

- **WPD MAE**: CNN=231.6709, candidateE=226.8054. Δ=-4.8655 (▲ better), improved on 161/168 samples.
- **WPD W1**: CNN=45.2713, candidateE=50.5029. Δ=+5.2316 (▼ worse), improved on 92/168 samples.
- **WPD bias abs**: CNN=35.3439, candidateE=37.7964. Δ=+2.4525 (▼ worse), improved on 102/168 samples.

### Q3. Did candidateE improve gradient metrics?

- **Gradient MAE**: CNN=0.3491, candidateE=0.3544. Δ=+0.0053 (▼ worse), improved on 16/168 samples.
- **Gradient W1**: CNN=0.2329, candidateE=0.2607. Δ=+0.0278 (▼ worse), improved on 0/168 samples.
- **Gradient kurtosis abs Δ**: CNN=3.7004, candidateE=3.7305. Δ=+0.0301 (▼ worse), improved on 106/168 samples.

### Q4. Did candidateE improve exceedance metrics, especially s>5?

- **Exceedance abs Δ s>5**: CNN=0.00424, candidateE=0.00569. Δ=+0.0014 (▼ worse), improved on 20/168 samples.
- **Exceedance abs Δ s>10**: CNN=0.00658, candidateE=0.00672. Δ=+0.0001 (▼ worse), improved on 86/168 samples.
- **Exceedance abs Δ s>15**: CNN=0.00621, candidateE=0.00736. Δ=+0.0011 (▼ worse), improved on 52/168 samples.
- **Exceedance abs Δ p90**: CNN=0.01025, candidateE=0.01088. Δ=+0.0006 (▼ worse), improved on 108/168 samples.

### Q5. Did candidateE move component-count behavior toward GT?

- **Component-count curve L1**: CNN=115.5278, candidateE=125.6260. Δ=+10.0982 (▼ worse), improved on 5/168 samples.
- **Comp-count abs Δ t5**: CNN=64.3810, candidateE=65.3869. Δ=+1.0060 (▼ worse), improved on 56/168 samples.
- **Comp-count abs Δ t10**: CNN=187.1667, candidateE=197.3274. Δ=+10.1607 (▼ worse), improved on 26/168 samples.
- **Comp-count abs Δ t15**: CNN=116.8095, candidateE=130.6726. Δ=+13.8631 (▼ worse), improved on 19/168 samples.

### Q6. Did candidateE improve PD or MT distances?

PD and MT distances for `candidateE` are not available in this evaluation (TTK was not re-run on these outputs). Pre-computed values for CNN and GAN:

- **PD distance**: CNN=27.4063, GAN=20.8641, candidateE=N/A (requires TTK run)
- **MT distance**: CNN=5.8678, GAN=8.3481, candidateE=N/A (requires TTK run)

To compute PD/MT for `candidateE`: convert outputs to VTI via `scripts/convert_phire_to_vti.py`, then run `scripts/compute_composite_tree_distance.py`.

### Q7. Did candidateE change the adjacent clusters (samples 10–13, 90–93, 162–163)?

See `adjacent_cluster_table_candidateE.csv` for full detail. Summary of PSNRuv, speed MAE, and gradient MAE changes:

| Sample | Δ PSNRuv | Δ Speed MAE | Δ Grad MAE | Δ Exceed s>5 |
|---|---|---|---|---|
| 10 | 0.212 | -0.0172 | 0.0011 | 0.00206 |
| 11 | 0.171 | -0.0137 | 0.0020 | 0.00170 |
| 12 | 0.156 | -0.0118 | 0.0027 | 0.00147 |
| 13 | 0.132 | -0.0089 | 0.0043 | 0.00147 |
| 90 | 0.078 | -0.0038 | 0.0022 | 0.00238 |
| 91 | 0.076 | -0.0045 | 0.0019 | 0.00249 |
| 92 | 0.093 | -0.0065 | 0.0020 | 0.00177 |
| 93 | 0.124 | -0.0100 | 0.0023 | 0.00227 |
| 162 | 0.365 | -0.0137 | -0.0038 | 0.00171 |
| 163 | 0.346 | -0.0138 | -0.0038 | 0.00283 |

## Pairwise summary: CNN vs candidateE

| Metric | Mean CNN | Mean candidateE | Δ | N improved | N worsened |
|---|---|---|---|---|---|
| psnruv | 31.1925 | 31.3765 | 0.1840 | 167 | 1 |
| ssim | N/A | N/A | N/A | 0 | 0 |
| speed_mae | 0.6941 | 0.6816 | -0.0125 | 164 | 4 |
| speed_rmse | 1.1078 | 1.0816 | -0.0261 | 167 | 1 |
| wpd_bias_abs | 35.3439 | 37.7964 | 2.4525 | 102 | 66 |
| wpd_mae | 231.6709 | 226.8054 | -4.8655 | 161 | 7 |
| wpd_w1 | 45.2713 | 50.5029 | 5.2316 | 92 | 76 |
| psd_log_l2 | 0.8335 | 0.8786 | 0.0451 | 16 | 152 |
| psd_slope_abs_delta | 0.9150 | 1.0210 | 0.1060 | 36 | 132 |
| grad_mae | 0.3491 | 0.3544 | 0.0053 | 16 | 152 |
| grad_w1 | 0.2329 | 0.2607 | 0.0278 | 0 | 168 |
| grad_kurtosis_abs_delta | 3.7004 | 3.7305 | 0.0301 | 106 | 62 |
| exceed_abs_t5 | 0.0042 | 0.0057 | 0.0014 | 20 | 148 |
| exceed_abs_t10 | 0.0066 | 0.0067 | 0.0001 | 86 | 82 |
| exceed_abs_t15 | 0.0062 | 0.0074 | 0.0011 | 52 | 115 |
| exceed_abs_p90 | 0.0103 | 0.0109 | 0.0006 | 108 | 59 |
| comp_curve_l1 | 115.5278 | 125.6260 | 10.0982 | 5 | 162 |
| pd_distance | N/A | N/A | N/A | 0 | 0 |
| mt_distance | N/A | N/A | N/A | 0 | 0 |

## Output files

- `ttk_runs_fixed/topology_finetuning/candidateE_eval/all_sample_metrics_candidateE.csv` — per-sample metrics for all methods
- `ttk_runs_fixed/topology_finetuning/candidateE_eval/pairwise_cnn_vs_candidateE.csv` — per-metric delta table
- `ttk_runs_fixed/topology_finetuning/candidateE_eval/winner_counts_candidateE.csv` — win counts per method
- `ttk_runs_fixed/topology_finetuning/candidateE_eval/adjacent_cluster_table_candidateE.csv` — adjacent-cluster detail
- `docs/topology_finetuning_candidateE_eval.md` — this report

## Notes

- PSNRuv is computed on physical [u,v] arrays using per-sample dynamic data range (GT max − GT min).
- All speed-based metrics use scalar speed = sqrt(u² + v²) in physical units (m/s).
- Exceedance fractions use GT percentile thresholds (p90/p95/p99) computed per sample.
- Component counts use 8-connectivity on speed superlevel sets.
- PD/MT distances for CNN and GAN are loaded from pre-computed merged CSV; `candidateE` requires a fresh TTK run.
- `candidateE` was fine-tuned from the pretrained CNN checkpoint. Results may change with different lambda settings or more epochs.
