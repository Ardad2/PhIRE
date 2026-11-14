# Phase 6 PERS-only and full 2×2 ablation: reproducibility record

## Phase 6 PERS-only: verified numerical evaluation and full four-arm CV×PERS ablation (October 14, 2026)

## LXXII.1 Status, evidence, and chronology

**Authoritative new finding:** The researcher supplied the PERS-only training-validation terminal output, a successful cheap evaluator transcript (`Evaluation exit status: 0`), and the generated report `docs/topology_finetuning_candidateF_grad_PERS_only_expanded2688_eval.md` (generated 2026-10-14). **The new 2×2 study now has complete training, paired inference, and scalar/domain evaluation for all four arms.** Section LXXI is retained as a historical, timestamped *pre-evaluation* snapshot; its statement that PERS-only evaluation was pending is superseded by this LXXII entry. Do not change LXXI retrospectively.

A separate Git patch attempt failed because the user selected the `phase6_notes_LXVII_to_LXXI.patch` file even though `grep '^# Part LX' ... | tail` showed the local file **already ended at Part LXX**. The failed `git apply --check` rejected the wrong-base patch, so the LXXI entry was **not** added to the Spark copy in that attempt. This update includes a **single verified LXX→LXXII** patch to add both LXXI and LXXII; no manual deletion/reset is required. The earlier worktree includes unrelated uncommitted figure and code changes; neither those nor the frozen `topoinvis-2026-arxiv-v1.1` tag should be modified by the docs-only patch.

New arm: `grad_pers_only`, method `candidateF_grad_PERS_only_expanded2688`. This is fine-tuning of the original pretrained PhIRE CNN, **not a new SR architecture**. The loss configuration is normalized `L_uv` (coefficient 1.0), scalar-speed gradient `L_grad` (0.05), fixed-index TTK persistence-gap supervision `L_TTKpers` (0.002), with `L_TTKCV=0.0`, `L_crit=0.0`, `L_speed=0.0`, `L_levelset=0.0` and `L_wpd=0.0`. See `scripts/run_candidateF_cv_pers_ablation.py` and `scripts/run_candidateF_expanded2688_finetune.py`. Uses the repaired paired-constraint file and exactly 64 constraints per training field, as validated in Part LXV.

The fixed datasets and experiment outputs are:

```text
Training: example_data_topology_expanded_2688/wind_MR-HR.tfrecord          # 2,688 fields
Benchmark: example_data_fixed/wind_MR-HR.tfrecord                        # 168 fields
E2 targets: ttk_runs_fixed/topology_finetuning/candidateE2_fixed_lowlambda_expanded2688_constraints/ttk_pd_critical_pairs_gtvalues.npz
Training script: scripts/run_candidateF_cv_pers_ablation.py --variant grad_pers_only
Primary log: logs/wind_finetune_candidateF_grad_PERS_only_expanded2688.log
Console log: logs/phase6_grad_pers_only_console.log
Checkpoint root: models_fixed/topology_finetuning/wind_finetune_candidateF_grad_PERS_only_expanded2688/
  cnn00001/cnn, cnn00002/cnn, cnn00003/cnn, cnn/cnn
Inference: data_out/wind_finetune_candidateF_grad_PERS_only_expanded2688/
  idx.npy, dataIN.npy, dataGT.npy, dataSR.npy
Cheap-eval script: scripts/evaluate_finetune_candidate.py
Cheap-eval log: logs/phase6_candidateF_grad_PERS_only_expanded2688_cheap_eval.log
Cheap-eval results: ttk_runs_fixed/topology_finetuning/candidateF_grad_PERS_only_expanded2688_eval/
  all_sample_metrics_candidateF_grad_PERS_only_expanded2688.csv
  pairwise_cnn_vs_candidateF_grad_PERS_only_expanded2688.csv
  winner_counts_candidateF_grad_PERS_only_expanded2688.csv
  adjacent_cluster_table_candidateF_grad_PERS_only_expanded2688.csv
Report: docs/topology_finetuning_candidateF_grad_PERS_only_expanded2688_eval.md
```

**Validated runtime results (user's Spark output):** `Training pipeline exit status: 0`; three epoch checkpoints and final `cnn/cnn` checkpoint present; `idx.npy` shape `(168,)` exactly ordered `0..167`, `dataIN.npy` `(168,100,100,2)`, `dataGT.npy` and `dataSR.npy` `(168,500,500,2)` each, all finite; `dataGT.npy` and `dataIN.npy` **exactly equal** to fixed CNN benchmark (`np.array_equal=True`, max absolute difference `0`). GT wind components: `u=[-16.1737,30.8122]`, `v=[-30.4676,28.1310]`; PERS-only SR `u=[-17.6968,28.4375]`, `v=[-29.2251,28.2943]`. GT scalar-speed range `[0.0122,33.9885] m/s`, PERS-only SR `[0.0006,37.9284] m/s`; mean speed MAE **0.5805 m/s**. PERS-only maximum exceeds this particular fixed benchmark's GT maximum by **3.9399 m/s (~11.6%)**. This merits per-field high-speed checks but is not, by itself, proof of physically invalid winds; similarly CV-only has a larger +5.3446 m/s (~15.7%) maximum overshoot. Gradient-only's max was 33.4720 m/s.

The cheap evaluator confirmed `Common samples: 168`, benchmark `GT alignment ... max_abs_diff=0.00e+00`, 336 existing CNN/GAN topology lookup entries, **672 output rows** (bicubic/CNN/GAN/PERS-only × 168 fields), and `Evaluation exit status: 0`. **SSIM was not computed for any candidate** because of the `skimage`/NumPy binary ABI mismatch (`Expected 96 from C header, got 88 from PyObject`); NaN/0-valid comparisons are *missing*, not measured zeros. Avoid changing the working TF1-compat installation merely to silence these errors; use an isolated clean SSIM replay with the paper's validated metric convention later.

## LXXII.2 Exact PERS-only evaluation command and terminal checkpoints

```bash
cd ~/PhIRE
mkdir -p logs
set -o pipefail
METHOD=candidateF_grad_PERS_only_expanded2688

env -u PYTHONNOUSERSITE /usr/bin/python3 \
  scripts/evaluate_finetune_candidate.py \
  --candidate-name "$METHOD" \
  --candidate-dir "data_out/wind_finetune_${METHOD}" \
  --cnn-dir data_out_fixed/wind_mrhr_cnn \
  --gan-dir data_out_fixed/wind_mrhr_gan \
  --merged-csv ttk_runs_fixed/combined/psnr_topology_physics_merged.csv \
  --out-dir "ttk_runs_fixed/topology_finetuning/${METHOD}_eval" \
  2>&1 | tee "logs/phase6_${METHOD}_cheap_eval.log"

echo "Evaluation exit status: $?"
```

Key observed lines, selectively reproduced from user-supplied terminal output:

```text
[validate] Post-inference validation complete: shapes, exact idx order, finiteness, and ... GT/IN alignment all passed.
Run complete.
GT scalar-speed range : [0.0122, 33.9885] m/s
SR scalar-speed range : [0.0006, 37.9284] m/s
Mean absolute scalar-speed error (SR vs GT): 0.5805 m/s
Common samples: 168
Checking GT alignment … OK (max_abs_diff=0.00e+00)
Total metric rows: 672
Written: docs/topology_finetuning_candidateF_grad_PERS_only_expanded2688_eval.md
Evaluation complete.
Evaluation exit status: 0
```

## LXXII.3 Complete four-cell matched scalar/domain comparison

All candidate values below are arithmetic means **as rounded to four decimals in the respective Markdown evaluation reports**. The three new reports were supplied in chat. The fourth is the archived `docs/topology_finetuning_candidateF_grad_E2_low_expanded2688_eval.md` (F1, generated 2026-07-16). All four use the same fixed 168-sample benchmark, evaluator conventions, pretrained CNN reference and gradient coefficient `0.05`, at 2,688 training samples. The four arms are: `(CV=0,PERS=0)=grad_pair_zero`; `(1,0)=grad_cv_only`; `(0,1)=grad_pers_only`; `(1,1)=grad_e2_low` (F1). ↑ is higher is better; ↓ is lower is better. The baseline CNN is **not** the factorial zero cell, which is the fine-tuned gradient-only control.

| Metric | CNN | Grad-only (00) | Grad+CV (10) | Grad+PERS (01) | Full F1 (11) |
|---|---:|---:|---:|---:|---:|
| PSNRuv (dB) ↑ | 31.1925 | 33.3872 | 32.7620 | 32.6119 | 32.4949 |
| Speed MAE (m/s) ↓ | 0.6941 | 0.5275 | 0.5714 | 0.5805 | 0.5899 |
| Speed RMSE (m/s) ↓ | 1.1078 | 0.8939 | 0.9592 | 0.9781 | 0.9891 |
| WPD MAE (m³/s³) ↓ | 231.6709 | 169.3772 | 184.5553 | 188.9087 | 190.6897 |
| WPD Wasserstein-1 ↓ | 45.2713 | 22.3588 | 22.6663 | 16.4613 | 24.7322 |
| WPD bias abs (m³/s³) ↓ | 35.3439 | 16.4208 | 20.6194 | 10.5507 | 22.6377 |
| Gradient MAE ↓ | 0.3491 | 0.3107 | 0.3111 | 0.3116 | 0.3125 |
| Gradient W1 ↓ | 0.2329 | 0.1809 | 0.1494 | 0.1412 | 0.1381 |
| Gradient kurtosis abs Δ ↓ | 3.7004 | 2.5295 | 3.1057 | 3.1732 | 3.2601 |
| PSD log-L2 ↓ | 0.8335 | 0.8992 | 0.8433 | 0.8699 | 0.8147 |
| PSD slope abs Δ ↓ | 0.9150 | 1.4924 | 1.3487 | 1.4574 | 1.1385 |
| Exceedance abs Δ s>5 ↓ | 0.0042 | 0.0039 | 0.0022 | 0.0021 | 0.0025 |
| Exceedance abs Δ s>10 ↓ | 0.0066 | 0.0032 | 0.0052 | 0.0020 | 0.0052 |
| Exceedance abs Δ s>15 ↓ | 0.0062 | 0.0030 | 0.0025 | 0.0019 | 0.0027 |
| Exceedance abs Δ p90 ↓ | 0.0103 | 0.0040 | 0.0039 | 0.0033 | 0.0047 |
| Component-count curve L1 ↓ | 115.5278 | 97.6548 | 77.8165 | 77.0109 | 81.1815 |
| SSIM (speed) | N/A | N/A | N/A | N/A | N/A |

Interpretation, with important limitations:

- **Fidelity:** Gradient-only is best of the four fine-tuned cells on PSNRuv (`33.3872`) and speed MAE (`0.5275 m/s`), with all **168/168** benchmark fields improved versus the pretrained CNN for both measures. This does not establish a direct per-field win count against the other three fine-tuned cells; that would require comparing their saved per-field CSVs by sample ID.
- **CV-only structural proxy:** Component-count curve L1 drops from `97.6548` (gradient-only) to `77.8165` (CV-only), even below the historical F1 value `81.1815`. **PERS-only improves it still further to `77.0109`**, the lowest of the four. These are connected-component curve proxies, **not** persistence-diagram/merge-tree distances and **not** proof of spatial placement correctness.
- **PERS-only wind-power/distribution agreement:** WPD Wasserstein-1 is the best of the four at `16.4613` (gradient-only `22.3588`, CV-only `22.6663`, F1 `24.7322`), and WPD bias absolute error is lowest at `10.5507` (gradient-only `16.4208`, CV-only `20.6194`, F1 `22.6377`). PERS-only also leads these four on exceedance errors at `s>10`, `s>15` and p90 in the rounded reports. These are substantive *scalar/domain* advantages, not yet verified true topological gains.
- **Full F1 strengths:** It has the lowest gradient W1 `0.1381`, PSD log-L2 `0.8147`, and PSD slope deviation `1.1385` *among the four fine-tuned cells*. The pretrained CNN PSD slope deviation `0.9150` remains **better than all four fine-tuned cells**; compared with CNN, the three new single-auxiliary arms worsen PSD slope on `168/168` benchmark fields. Do not describe F1's PSD-slope result as a universal improvement over the unmodified CNN.
- **Wind extremes and SSIM:** PERS-only and CV-only each reach a larger scalar-speed maximum than the maximum observed in this benchmark GT; inspect extremes by sample and space. SSIM is missing for the whole table due to the local broken dependency.

## LXXII.4 Descriptive factorial effects from rounded aggregate means

For a metric `Y` with the four cells `Y_00` (gradient-only), `Y_10` (CV-only), `Y_01` (PERS-only), and `Y_11` (full F1), use:

```text
CV averaged main effect    = ((Y_10 - Y_00) + (Y_11 - Y_01))/2
PERS averaged main effect  = ((Y_01 - Y_00) + (Y_11 - Y_10))/2
CV×PERS interaction       = Y_11 - Y_10 - Y_01 + Y_00
```

| Metric | CV averaged main effect | PERS averaged main effect | CV×PERS interaction |
|---|---:|---:|---:|
| PSNRuv (dB) | -0.3711 | -0.5212 | +0.5082 |
| Speed MAE (m/s) | +0.0267 | +0.0358 | -0.0345 |
| Gradient W1 | -0.0173 | -0.0255 | +0.0284 |
| WPD Wasserstein-1 | +4.2892 | -1.9158 | +7.9634 |
| WPD bias abs (m³/s³) | +8.1428 | -1.9259 | +7.8884 |
| Component-count curve L1 | -7.8338 | -8.6394 | +24.0089 |
| PSD log-L2 | -0.0555 | -0.0290 | +0.0007 |
| PSD slope abs Δ | -0.2313 | -0.1226 | -0.1752 |
| Exceedance abs Δ s>10 | +0.0026 | -0.0006 | +0.0012 |

The effects are **descriptive mean contrasts from rounded source reports**; the underlying unrounded sample-level means may change the final reported digits slightly. For PSNR, positive differences are better; for every other listed error metric, negative differences are better. The interaction is a difference-in-differences in the metric's own units, **not a p-value**, and is only meaningful conditional on the fixed gradient loss/training setup. These 168 fields include temporal dependence and there is only one completed training run per arm: uncertainty, seed sensitivity, statistical significance, and generalizability **have not been established**. A robust follow-on should calculate paired per-field contrasts from matching `sample_idx` across the generated CSVs and, if warranted, repeat training with controlled seeds and temporally aware inference.

The optional standard-library-only reproducibility script supplied alongside this notes update is `scripts/compare_phase6_2x2_reports.py`. It reads four existing report Markdown files from the Spark repository, validates that CNN anchor values match, prints the 2×2 summary, averaged effects and interactions, and does not write files or import TensorFlow/NumPy/scikit-image. Run it without copying it into the repository:

```bash
cd ~/PhIRE
/usr/bin/python3 phase6_lxxii_update/scripts/compare_phase6_2x2_reports.py --repo-root "$PWD"
```

## LXXII.5 What **has not** been measured, and next reproducibility checkpoints

- The new gradient-only, CV-only and PERS-only arms have **no newly computed PD bottleneck, corrected W₂, merge-tree, or Phase 5E locality-constrained matching metrics**. The 336 historical lookup entries in the cheap evaluator come from the CNN/GAN baseline CSV only, not from candidate topology. Treat all new-cell candidate PD/MT values as **pending**.
- The old TTK `pd_distance` printed for historical CNN/GAN and the historical F1 topology report are **not substitutes** for the accepted corrected/GUDHI-audited PD-distance procedure. To compare topology, first run each candidate's TTK VTI/PD/MT generation consistently, with frozen 160×160 speed patch conventions, then apply the **corrected** PD distance audit and matching merge-tree method across the same 168 sample indices. Match sample order and version/hash all inputs; then run the same Phase 5E radii `r ∈ {4,8,16,32,64,∞}` on the feature-anchor localities, explicitly separating ground-truth feature coverage from locality-constrained matching cost. Record the pixel-to-geographic distance calibration before referring to physical kilometers.
- Preserve training checkpoints, per-field arrays and all original reports unchanged. Optionally create a manifest with `sha256sum` for critical data/CSV/checkpoint artifacts; store logs and package/version manifests separately, including the TF 2.20.0 `compat.v1` runtime and NumPy ABI limitation. Do not commit large ignored `.npy` arrays to Git by accident.
- Inspect high-speed extremes: the PERS-only `37.9284 m/s` and CV-only `39.3331 m/s` maxima are *dataset-wide maxima*, not evidence that a particular sample/position has been located. A per-field maximum and absolute-error audit is needed to identify overshoot locations before scientific interpretation.
- For a future manuscript figure, characterize the trade-off honestly: gradient-only = best pointwise fidelity; CV-only and PERS-only = stronger selected structural proxies, with PERS-only leading WPD distribution/bias; F1 = strongest gradient W1 and PSD among these four fine-tuned cells. No claim of individual CV/PERS contributions to true topology can be made until the matched corrected topological results exist.

**Status at the close of LXXII:** Three newly trained arms, plus archived full F1, are now available with **all four scalar/domain evaluations**. SSIM, new-arm corrected PD/MT, spatially constrained persistence and uncertainty analysis remain incomplete. The failed wrong-base LXVII→LXXI Git patch did not update the repository; use the bundled **LXX→LXXII cumulative patch** only after its `git apply --check` succeeds.
