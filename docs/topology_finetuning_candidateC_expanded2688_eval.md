# candidateC_expanded2688 fine-tuning evaluation

**Generated:** 2026-05-29

**Candidate:** `candidateC_expanded2688`

**Samples evaluated:** 168

**Methods compared:** bicubic, cnn, gan, candidateC_expanded2688

> **SSIM note:** SSIM was not computed because skimage could not be imported in the current NumPy environment (`ValueError('numpy.dtype size changed, may indicate binary incompatibility. Expected 96 from C header, got 88 from PyObject')`). All SSIM values are NaN.

> Note: PD/MT distances for `candidateC_expanded2688` are N/A — TTK has not been re-run on these outputs. Values for CNN/GAN are loaded from the pre-computed merged CSV.

## Summary metrics table

| Metric | Bicubic | Cnn | Gan | Candidatec_expanded2688 |
|---|---|---|---|---|
| PSNRuv (dB) | 32.2202 | 31.1925 | 29.1380 | 33.4807 |
| SSIM on speed | N/A | N/A | N/A | N/A |
| Speed MAE (m/s) | 0.5963 | 0.6941 | 0.9026 | 0.5147 |
| Speed RMSE (m/s) | 1.0156 | 1.1078 | 1.3775 | 0.8796 |
| WPD MAE (m³/s³) | 193.5403 | 231.6709 | 310.7328 | 165.9447 |
| WPD Wasserstein-1 | 55.5103 | 45.2713 | 85.6191 | 16.1544 |
| WPD bias abs (m³/s³) | 41.2244 | 35.3439 | 78.7236 | 7.9858 |
| Gradient MAE | 0.3696 | 0.3491 | 0.3806 | 0.3105 |
| Gradient W1 | 0.3282 | 0.2329 | 0.0564 | 0.1942 |
| Gradient kurtosis abs Δ | 6.6163 | 3.7004 | 4.2010 | 3.2762 |
| PSD log-L2 | 1.3625 | 0.8335 | 0.5139 | 0.9527 |
| PSD slope abs Δ | 1.6684 | 0.9150 | 0.9482 | 1.2719 |
| Exceedance abs Δ s>5 | 0.0095 | 0.0042 | 0.0082 | 0.0055 |
| Exceedance abs Δ s>10 | 0.0085 | 0.0066 | 0.0243 | 0.0036 |
| Exceedance abs Δ s>15 | 0.0074 | 0.0062 | 0.0096 | 0.0015 |
| Exceedance abs Δ p90 | 0.0109 | 0.0103 | 0.0123 | 0.0023 |
| Component-count curve L1 | 173.1855 | 115.5278 | 124.6567 | 112.7014 |
| PD distance (TTK) | N/A | 27.4063 | 20.8641 | N/A |
| MT distance (TTK) | N/A | 5.8678 | 8.3481 | N/A |

## 7 Key evaluation questions

### Q1. Did candidateC_expanded2688 preserve CNN's direct-fidelity advantage?

- **PSNRuv**: CNN=31.1925, GAN=29.1380, candidateC_expanded2688=33.4807. Δ(candidateC_expanded2688−CNN)=+2.2882 (▲ better), improved on 168/168 samples.
- **SSIM**: CNN=N/A, GAN=N/A, candidateC_expanded2688=N/A. Δ(candidateC_expanded2688−CNN)=N/A, improved on 0/0 samples.
- **Speed MAE**: CNN=0.6941, GAN=0.9026, candidateC_expanded2688=0.5147. Δ(candidateC_expanded2688−CNN)=-0.1794 (▲ better), improved on 168/168 samples.
- **Speed RMSE**: CNN=1.1078, GAN=1.3775, candidateC_expanded2688=0.8796. Δ(candidateC_expanded2688−CNN)=-0.2281 (▲ better), improved on 168/168 samples.

### Q2. Did candidateC_expanded2688 improve scalar speed or WPD metrics?

- **WPD MAE**: CNN=231.6709, candidateC_expanded2688=165.9447. Δ=-65.7262 (▲ better), improved on 168/168 samples.
- **WPD W1**: CNN=45.2713, candidateC_expanded2688=16.1544. Δ=-29.1169 (▲ better), improved on 162/168 samples.
- **WPD bias abs**: CNN=35.3439, candidateC_expanded2688=7.9858. Δ=-27.3581 (▲ better), improved on 156/168 samples.

### Q3. Did candidateC_expanded2688 improve gradient metrics?

- **Gradient MAE**: CNN=0.3491, candidateC_expanded2688=0.3105. Δ=-0.0386 (▲ better), improved on 168/168 samples.
- **Gradient W1**: CNN=0.2329, candidateC_expanded2688=0.1942. Δ=-0.0388 (▲ better), improved on 153/168 samples.
- **Gradient kurtosis abs Δ**: CNN=3.7004, candidateC_expanded2688=3.2762. Δ=-0.4242 (▲ better), improved on 94/168 samples.

### Q4. Did candidateC_expanded2688 improve exceedance metrics, especially s>5?

- **Exceedance abs Δ s>5**: CNN=0.00424, candidateC_expanded2688=0.00555. Δ=+0.0013 (▼ worse), improved on 48/168 samples.
- **Exceedance abs Δ s>10**: CNN=0.00658, candidateC_expanded2688=0.00358. Δ=-0.0030 (▲ better), improved on 112/168 samples.
- **Exceedance abs Δ s>15**: CNN=0.00621, candidateC_expanded2688=0.00155. Δ=-0.0047 (▲ better), improved on 148/168 samples.
- **Exceedance abs Δ p90**: CNN=0.01025, candidateC_expanded2688=0.00230. Δ=-0.0080 (▲ better), improved on 156/168 samples.

### Q5. Did candidateC_expanded2688 move component-count behavior toward GT?

- **Component-count curve L1**: CNN=115.5278, candidateC_expanded2688=112.7014. Δ=-2.8264 (▲ better), improved on 109/168 samples.
- **Comp-count abs Δ t5**: CNN=64.3810, candidateC_expanded2688=61.9048. Δ=-2.4762 (▲ better), improved on 91/168 samples.
- **Comp-count abs Δ t10**: CNN=187.1667, candidateC_expanded2688=200.0000. Δ=+12.8333 (▼ worse), improved on 46/168 samples.
- **Comp-count abs Δ t15**: CNN=116.8095, candidateC_expanded2688=105.8095. Δ=-11.0000 (▲ better), improved on 122/168 samples.

### Q6. Did candidateC_expanded2688 improve PD or MT distances?

PD and MT distances for `candidateC_expanded2688` are not available in this evaluation (TTK was not re-run on these outputs). Pre-computed values for CNN and GAN:

- **PD distance**: CNN=27.4063, GAN=20.8641, candidateC_expanded2688=N/A (requires TTK run)
- **MT distance**: CNN=5.8678, GAN=8.3481, candidateC_expanded2688=N/A (requires TTK run)

To compute PD/MT for `candidateC_expanded2688`: convert outputs to VTI via `scripts/convert_phire_to_vti.py`, then run `scripts/compute_composite_tree_distance.py`.

### Q7. Did candidateC_expanded2688 change the adjacent clusters (samples 10–13, 90–93, 162–163)?

See `adjacent_cluster_table_candidateC_expanded2688.csv` for full detail. Summary of PSNRuv, speed MAE, and gradient MAE changes:

| Sample | Δ PSNRuv | Δ Speed MAE | Δ Grad MAE | Δ Exceed s>5 |
|---|---|---|---|---|
| 10 | 2.425 | -0.1942 | -0.0387 | 0.00328 |
| 11 | 2.349 | -0.1923 | -0.0398 | 0.00222 |
| 12 | 2.363 | -0.1927 | -0.0437 | 0.00280 |
| 13 | 2.280 | -0.1855 | -0.0409 | 0.00090 |
| 90 | 2.134 | -0.1396 | -0.0293 | -0.00154 |
| 91 | 2.161 | -0.1443 | -0.0291 | 0.00285 |
| 92 | 2.149 | -0.1554 | -0.0318 | 0.00735 |
| 93 | 2.105 | -0.1660 | -0.0329 | 0.00519 |
| 162 | 3.579 | -0.1491 | -0.0271 | -0.00300 |
| 163 | 3.510 | -0.1493 | -0.0296 | 0.00106 |

## Pairwise summary: CNN vs candidateC_expanded2688

| Metric | Mean CNN | Mean candidateC_expanded2688 | Δ | N improved | N worsened |
|---|---|---|---|---|---|
| psnruv | 31.1925 | 33.4807 | 2.2882 | 168 | 0 |
| ssim | N/A | N/A | N/A | 0 | 0 |
| speed_mae | 0.6941 | 0.5147 | -0.1794 | 168 | 0 |
| speed_rmse | 1.1078 | 0.8796 | -0.2281 | 168 | 0 |
| wpd_bias_abs | 35.3439 | 7.9858 | -27.3581 | 156 | 12 |
| wpd_mae | 231.6709 | 165.9447 | -65.7262 | 168 | 0 |
| wpd_w1 | 45.2713 | 16.1544 | -29.1169 | 162 | 6 |
| psd_log_l2 | 0.8335 | 0.9527 | 0.1192 | 8 | 160 |
| psd_slope_abs_delta | 0.9150 | 1.2719 | 0.3569 | 0 | 168 |
| grad_mae | 0.3491 | 0.3105 | -0.0386 | 168 | 0 |
| grad_w1 | 0.2329 | 0.1942 | -0.0388 | 153 | 15 |
| grad_kurtosis_abs_delta | 3.7004 | 3.2762 | -0.4242 | 94 | 74 |
| exceed_abs_t5 | 0.0042 | 0.0055 | 0.0013 | 48 | 119 |
| exceed_abs_t10 | 0.0066 | 0.0036 | -0.0030 | 112 | 56 |
| exceed_abs_t15 | 0.0062 | 0.0015 | -0.0047 | 148 | 19 |
| exceed_abs_p90 | 0.0103 | 0.0023 | -0.0080 | 156 | 12 |
| comp_curve_l1 | 115.5278 | 112.7014 | -2.8264 | 109 | 58 |
| pd_distance | N/A | N/A | N/A | 0 | 0 |
| mt_distance | N/A | N/A | N/A | 0 | 0 |

## Output files

- `ttk_runs_fixed/topology_finetuning/candidateC_expanded2688_eval/all_sample_metrics_candidateC_expanded2688.csv` — per-sample metrics for all methods
- `ttk_runs_fixed/topology_finetuning/candidateC_expanded2688_eval/pairwise_cnn_vs_candidateC_expanded2688.csv` — per-metric delta table
- `ttk_runs_fixed/topology_finetuning/candidateC_expanded2688_eval/winner_counts_candidateC_expanded2688.csv` — win counts per method
- `ttk_runs_fixed/topology_finetuning/candidateC_expanded2688_eval/adjacent_cluster_table_candidateC_expanded2688.csv` — adjacent-cluster detail
- `docs/topology_finetuning_candidateC_expanded2688_eval.md` — this report

## Notes

- PSNRuv is computed on physical [u,v] arrays using per-sample dynamic data range (GT max − GT min).
- All speed-based metrics use scalar speed = sqrt(u² + v²) in physical units (m/s).
- Exceedance fractions use GT percentile thresholds (p90/p95/p99) computed per sample.
- Component counts use 8-connectivity on speed superlevel sets.
- PD/MT distances for CNN and GAN are loaded from pre-computed merged CSV; `candidateC_expanded2688` requires a fresh TTK run.
- `candidateC_expanded2688` was fine-tuned from the pretrained CNN checkpoint. Results may change with different lambda settings or more epochs.
