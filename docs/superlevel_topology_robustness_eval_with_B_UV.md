# Superlevel topology robustness evaluation

**Generated:** 2026-07-14

**Method:** superlevel topology of speed `s` is computed as sublevel topology of `-s`, using the unmodified TTK `ttkPersistenceDiagramCmd`/`ttkMergeTreeCmd` commands on VTI files whose `wind_speed` array stores `-sqrt(u^2+v^2)` (negated speed), not speed. No existing sublevel/default topology output was read, modified, or overwritten, and no model was retrained. All outputs live under `ttk_runs_fixed/superlevel_topology/`.

**Note: lower PD/MT distance is better for both metrics, in both the superlevel and sublevel evaluations.**

**Samples evaluated per method:** 168

**Methods completed:** cnn, gan, candidateC_expanded2688, candidateB_plus_E2_tf_lowlambda_expanded2688, candidateE2_tf_lowlambda_expanded2688, candidateUV_plus_E2_tf_lowlambda_expanded2688, candidateUV_plus_crit_expanded2688

**Methods skipped/incomplete:** candidateUV_expanded2688, candidateB_expanded2688 (see console log / PASS-FAIL checklist for reasons)

**Superlevel MT-GAN baseline wins (GAN beats CNN on superlevel MT):** 23 samples — [6, 8, 16, 17, 18, 19, 20, 22, 25, 62, 63, 65, 66, 70, 77, 79, 80, 82, 83, 88, 89, 90, 122]

## Mean PD / MT by method (superlevel)

| Method | PD mean (superlevel) | MT mean (superlevel) | PD mean (sublevel, known) | MT mean (sublevel, known) |
|---|---:|---:|---:|---:|
| cnn | 27.3762 | 5.3231 | 27.4063 | 5.8678 |
| gan | 20.7168 | 7.8397 | 20.8641 | 8.3481 |
| candidateC_expanded2688 | 22.4417 | 5.3578 | 22.4944 | 6.0803 |
| candidateB_plus_E2_tf_lowlambda_expanded2688 | 24.1042 | 4.9998 | 23.9876 | 5.6774 |
| candidateE2_tf_lowlambda_expanded2688 | 24.3811 | 4.9796 | 24.2686 | 5.6628 |
| candidateUV_plus_E2_tf_lowlambda_expanded2688 | 25.1923 | 4.9522 | 25.0721 | 5.5940 |
| candidateUV_plus_crit_expanded2688 | 29.0752 | 5.0875 | 29.1143 | 5.6899 |

## PD / MT wins vs CNN, beats-GAN counts (superlevel)

| Method | PD < CNN | MT < CNN | PD beats GAN | MT beats GAN | MT-GAN recovered (/23) |
|---|---:|---:|---:|---:|---:|
| cnn | --/168 | --/168 | --/168 | --/168 | -- |
| gan | 166/168 | 23/168 | --/168 | --/168 | -- |
| candidateC_expanded2688 | 168/168 | 72/168 | 19/168 | 151/168 | 14 |
| candidateB_plus_E2_tf_lowlambda_expanded2688 | 165/168 | 110/168 | 6/168 | 165/168 | 23 |
| candidateE2_tf_lowlambda_expanded2688 | 164/168 | 112/168 | 5/168 | 166/168 | 23 |
| candidateUV_plus_E2_tf_lowlambda_expanded2688 | 160/168 | 120/168 | 0/168 | 168/168 | 23 |
| candidateUV_plus_crit_expanded2688 | 13/168 | 104/168 | 0/168 | 157/168 | 13 |

## Winner distribution (superlevel)

### PD distance winners

| Method | Wins |
|---|---:|
| gan | 149 |
| candidateC_expanded2688 | 19 |

### MT distance winners

| Method | Wins |
|---|---:|
| candidateUV_plus_E2_tf_lowlambda_expanded2688 | 43 |
| candidateE2_tf_lowlambda_expanded2688 | 34 |
| candidateUV_plus_crit_expanded2688 | 31 |
| candidateB_plus_E2_tf_lowlambda_expanded2688 | 30 |
| cnn | 27 |
| candidateC_expanded2688 | 3 |

## MT-GAN recovery detail (superlevel)

Samples where GAN's superlevel MT distance beat CNN's superlevel MT distance: [6, 8, 16, 17, 18, 19, 20, 22, 25, 62, 63, 65, 66, 70, 77, 79, 80, 82, 83, 88, 89, 90, 122]

| Method | Recovered (winner becomes method) | Still GAN | Now CNN |
|---|---:|---:|---:|
| candidateC_expanded2688 | 14 | 9 | 0 |
| candidateB_plus_E2_tf_lowlambda_expanded2688 | 23 | 0 | 0 |
| candidateE2_tf_lowlambda_expanded2688 | 23 | 0 | 0 |
| candidateUV_plus_E2_tf_lowlambda_expanded2688 | 23 | 0 | 0 |
| candidateUV_plus_crit_expanded2688 | 13 | 10 | 0 |

## Superlevel vs sublevel side-by-side (known values)

| Method | PD superlevel | PD sublevel (known) | MT superlevel | MT sublevel (known) |
|---|---:|---:|---:|---:|
| cnn | 27.3762 | 27.4063 | 5.3231 | 5.8678 |
| gan | 20.7168 | 20.8641 | 7.8397 | 8.3481 |
| candidateC_expanded2688 | 22.4417 | 22.4944 | 5.3578 | 6.0803 |
| candidateB_plus_E2_tf_lowlambda_expanded2688 | 24.1042 | 23.9876 | 4.9998 | 5.6774 |
| candidateE2_tf_lowlambda_expanded2688 | 24.3811 | 24.2686 | 4.9796 | 5.6628 |
| candidateUV_plus_E2_tf_lowlambda_expanded2688 | 25.1923 | 25.0721 | 4.9522 | 5.5940 |
| candidateUV_plus_crit_expanded2688 | 29.0752 | 29.1143 | 5.0875 | 5.6899 |

## Output files

- `ttk_runs_fixed/superlevel_topology/superlevel_pd_mt_per_sample_with_B_UV.csv` — per-sample PD/MT distances, one row per (method, sample)
- `ttk_runs_fixed/superlevel_topology/superlevel_summary_by_method_with_B_UV.csv` — per-method summary statistics
- `ttk_runs_fixed/superlevel_topology/superlevel_winner_comparison_with_B_UV.csv` — per-sample PD/MT winner across all completed methods
- `docs/superlevel_topology_robustness_eval_with_B_UV.md` — this report

## Notes

- PD distance = bottleneck distance between persistence diagrams (`ttkBottleneckDistance`), computed on the negated (`-speed`) scalar field.
- MT distance = Wasserstein-type merge tree distance (`ttkMergeTreeDistanceMatrix`), computed on the negated (`-speed`) scalar field.
- Both use 160x160 speed patches at (x0=0, y0=0), matching the sublevel baseline convention.
- **Lower distance is better for both metrics, in both the superlevel and sublevel domains.**
- Existing sublevel/default topology outputs and all candidate data_out*/models*/ outputs were never read for writing, modified, or overwritten by this script.
- No model was retrained to produce this evaluation.
