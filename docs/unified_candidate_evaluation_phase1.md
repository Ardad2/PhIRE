# Unified candidate evaluation -- Phase 1 report

## 0. Run mode

**STRICT MODE (`--strict-primary`): PASSED.** Every primary method met the completeness/validation criteria (168 cheap rows with sample_idx exactly 0..167, finite nonnegative PD/MT for the 18 learned/baseline-with-topology methods, PD/MT mean reproduction within 0.0001). The tables below are fully authoritative -- no placeholder rows.

## 1. Scope and primary/secondary method distinction

Primary set: 19 methods evaluated on the fixed 168-sample benchmark (3 baselines: bicubic, cnn, gan; 9 B-term-factorial variants incl. full Candidate B and C; 1 critical-proxy-only ablation; 3 repaired low-lambda E2 ablations; 3 Candidate F recombinations). Secondary set: 168-sample pilot runs, 672/1344-sample scale-study duplicates of primary objectives, PyTorch residual-refiner E2 variants (architecture confound), and deprecated pre-Phase-C legacy archives -- see `docs/unified_candidate_evaluation_inventory.md` for the full secondary listing.

## 2. Complete artifact inventory

See `ttk_runs_fixed/unified_candidate_evaluation/method_inventory.csv`, `docs/unified_candidate_evaluation_inventory.md`, and `docs/primary_candidate_artifact_reference.md`.

## 3. Exact table dimensions

- `unified_primary_per_sample_long.csv`: 19 methods x 168 samples = 3192 rows, one row per (method_id, sample_idx), no duplicates.
- Of these 19 primary methods, **19 have real per-sample data** in this repository checkout (bicubic, cnn, gan, uv, speed_only, levelset_only, speed_levelset, grad_only, speed_grad, grad_levelset, candidate_b, candidate_c, uv_crit, uv_e2, b_e2, c_e2, f1_grad_e2, f2_grad_levelset_e2, f3_grad_crit), and **0 have zero real data** (none).

## 4. Metric families and representations

See `column_mapping.csv`. Families: `vector_uv` (psnruv), `scalar_speed` (ssim_speed, speed_mae, speed_rmse), `wind_power_distribution` (wpd_*), `gradient_distribution` (grad_*), `frequency_domain` (psd_*), `threshold_geometry` (exceed_*, comp_*), `topology_pd` (pd_distance), `topology_mt` (mt_distance). PSNR is vector-field PSNR on physical [u,v] (`psnruv`), never scalar-speed PSNR; SSIM, speed errors, WPD, PD, and MT are all computed on scalar wind speed.

## 5. Validation results

- Topology PD/MT distance columns are resolved per-file via `resolve_topology_distance_columns()` (generic `pd_distance`/`mt_distance`, exact `pd_distance_<method>`/`mt_distance_<method>`, or a unique `pd_distance_*`/`mt_distance_*` prefix match) rather than assumed to be the generic name -- a topology CSV whose distance columns cannot be unambiguously resolved hard-fails the whole run immediately rather than silently reporting `row_count_topology=168` with unpopulated distances. See `topology_pd_source_column`/`topology_mt_source_column`/`topology_schema_status` in `method_inventory.csv`.
- Topology-mean reproduction: **18 PASS**, **0 FAIL**, **0 NO_DATA** out of 18 primary methods with an expected value.
- Cheap-metric completeness (168 rows, sample_idx exactly 0..167, no duplicates): checked for every primary method with any discovered source (baseline-harvested, legacy-combined, or a resolved candidate all_sample_metrics CSV).
- Join (cheap metrics <-> true topology, one-to-one on sample_idx): every per-sample record is a single merged dict keyed by sample_idx, so a missing cheap or topology value for a given sample_idx shows up directly as a non-finite cell rather than a silent row-count mismatch.

## 6. Baseline duplicate-consistency audit

Discovered 41 candidate `all_sample_metrics_*.csv` file(s) under `ttk_runs_fixed/topology_finetuning/*_eval/`. Every discovered file was checked for bicubic/cnn/gan rows; every baseline method present in more than one file had its rows compared pairwise, metric by metric, against a 1e-06 tolerance (would hard-fail the whole run on disagreement):
  - **bicubic**: 41 source(s) with data; canonical source (required metrics) = `/home/adadhwal/PhIRE/ttk_runs_fixed/topology_finetuning/candidateB_eval/all_sample_metrics_candidateB.csv`; ssim_availability=`unavailable`, canonical ssim source = `(none -- all-NaN)`.
  - **cnn**: 41 source(s) with data; canonical source (required metrics) = `/home/adadhwal/PhIRE/ttk_runs_fixed/topology_finetuning/candidateB_eval/all_sample_metrics_candidateB.csv`; ssim_availability=`unavailable`, canonical ssim source = `(none -- all-NaN)`.
  - **gan**: 41 source(s) with data; canonical source (required metrics) = `/home/adadhwal/PhIRE/ttk_runs_fixed/topology_finetuning/candidateB_eval/all_sample_metrics_candidateB.csv`; ssim_availability=`unavailable`, canonical ssim source = `(none -- all-NaN)`.

Canonical cnn/gan rows were additionally cross-checked against the older `ttk_runs_fixed/combined/psnr_topology_physics_merged.csv` pipeline for overlapping columns (tolerance 0.001):
  - **cnn**: worst overlapping-column disagreement `wpd_mae` = 1.705e-13; ssim_status=`availability_mismatch`.
  - **gan**: worst overlapping-column disagreement `wpd_mae` = 1.705e-13; ssim_status=`availability_mismatch`.

The independent `ttk_runs_fixed/combined/phase_c_results.csv` PD/MT cross-check remains as before:
  - **cnn**: max |Δpd| = 0.000e+00, max |Δmt| = 0.000e+00 across 168 cross-checked samples.
  - **gan**: max |Δpd| = 0.000e+00, max |Δmt| = 0.000e+00 across 168 cross-checked samples.

## 7. PD/MT mean reproduction table

| method_id | observed_pd_mean | expected_pd_mean | pd_pass | observed_mt_mean | expected_mt_mean | mt_pass |
|---|---:|---:|---|---:|---:|---|
| cnn | 27.4063 | 27.4063 | True | 5.8678 | 5.8678 | True |
| gan | 20.8641 | 20.8641 | True | 8.3481 | 8.3481 | True |
| uv | 29.6121 | 29.6121 | True | 6.0119 | 6.0119 | True |
| speed_only | 29.5783 | 29.5783 | True | 5.9996 | 5.9996 | True |
| levelset_only | 29.5953 | 29.5953 | True | 6.0076 | 6.0076 | True |
| speed_levelset | 29.4363 | 29.4363 | True | 5.9441 | 5.9441 | True |
| grad_only | 22.9326 | 22.9326 | True | 6.0560 | 6.056 | True |
| speed_grad | 22.9706 | 22.9706 | True | 6.2905 | 6.2905 | True |
| grad_levelset | 22.6194 | 22.6194 | True | 6.1996 | 6.1996 | True |
| candidate_b | 22.7070 | 22.707 | True | 6.1612 | 6.1612 | True |
| candidate_c | 22.4944 | 22.4944 | True | 6.0803 | 6.0803 | True |
| uv_crit | 29.1143 | 29.1143 | True | 5.6899 | 5.6899 | True |
| uv_e2 | 25.0721 | 25.0721 | True | 5.5940 | 5.594 | True |
| b_e2 | 23.9876 | 23.9876 | True | 5.6774 | 5.6774 | True |
| c_e2 | 24.2686 | 24.2686 | True | 5.6628 | 5.6628 | True |
| f1_grad_e2 | 23.8382 | 23.8382 | True | 5.6566 | 5.6566 | True |
| f2_grad_levelset_e2 | 23.7481 | 23.7481 | True | 5.6742 | 5.6742 | True |
| f3_grad_crit | 22.0179 | 22.0179 | True | 5.9840 | 5.984 | True |

## 8. Missingness, especially SSIM

Every (method, metric) cell in `unified_primary_missingness.csv` is classified into exactly one of three `missing_reason` categories: `no_source_artifact` (no file provides this metric for this method at all), `unavailable_global_dependency` (SSIM specifically, 0/168 finite -- consistent with the documented NumPy/scikit-image ABI incompatibility, not a data-quality bug), or `partial_source_coverage` (1..167/168 finite -- inconsistent coverage, always treated as a strict-mode failure since it indicates a real problem rather than a known benign gap).

- SSIM (`ssim_speed`) is in `OPTIONAL_CHEAP_METRIC_COLUMNS`: strict mode accepts either full (168/168) or fully-unavailable (0/168) coverage, and hard-fails only on partial coverage.
  - Fully available (168/168): none
  - Globally unavailable, accepted (0/168, `unavailable_global_dependency`): ['bicubic', 'cnn', 'gan', 'uv', 'speed_only', 'levelset_only', 'speed_levelset', 'grad_only', 'speed_grad', 'grad_levelset', 'candidate_b', 'candidate_c', 'uv_crit', 'uv_e2', 'b_e2', 'c_e2', 'f1_grad_e2', 'f2_grad_levelset_e2', 'f3_grad_crit']
  - No source at all for this method (`no_source_artifact`): none
  - Partial coverage, would strict-fail (`partial_source_coverage`): none
- SSIM is never filled, copied from a legacy row into a candidate row, or recomputed -- missing SSIM stays an empty cell in the unified table exactly as found in its source.
- Pairwise-vs-CNN summaries report `n_valid=0` for SSIM (and every other metric) when the candidate/CNN intersection of finite samples is empty, rather than fabricating a comparison.
- See `unified_primary_missingness.csv` for the full total/finite/missing breakdown per (method_id, metric).
- No missing value was filled with zero or inferred; all gaps are empty cells in the CSVs.

## 8b. Raw benchmark/array validation

For every primary method, `idx.npy`/`dataIN.npy`/`dataGT.npy`/`dataSR.npy` under its `data_out_dir` (or `data_out_fixed/wind_mrhr_<method>/` for the three baselines) are validated against the canonical CNN benchmark arrays at `data_out_fixed/wind_mrhr_cnn/{idx,dataIN,dataGT}.npy` -- loaded via `np.load(mmap_mode="r", allow_pickle=False)` and compared in 16-sample chunks so a full (168, 500, 500, 2) array is never fully materialized in memory. Checks: `idx.npy` shape `(168,)` and exactly `np.arange(168)`; `dataIN.npy` shape `(168, 100, 100, 2)` and exactly equal to the canonical `dataIN.npy`; `dataGT.npy` shape `(168, 500, 500, 2)` and exactly equal to the canonical `dataGT.npy`; `dataSR.npy` shape `(168, 500, 500, 2)` and entirely finite. `idx.npy` is not required for bicubic (its generator script does not produce one); every other file is required for every primary method. No `.npy` file is ever written or modified by this script.

| method_id | idx_validation_status | input_alignment_status | gt_alignment_status | sr_shape_status | sr_finiteness_status |
|---|---|---|---|---|---|
| bicubic | exact_0_167 | exact | exact | exact | all_finite |
| cnn | exact_0_167 | exact | exact | exact | all_finite |
| gan | exact_0_167 | exact | exact | exact | all_finite |
| uv | exact_0_167 | exact | exact | exact | all_finite |
| speed_only | exact_0_167 | exact | exact | exact | all_finite |
| levelset_only | exact_0_167 | exact | exact | exact | all_finite |
| speed_levelset | exact_0_167 | exact | exact | exact | all_finite |
| grad_only | exact_0_167 | exact | exact | exact | all_finite |
| speed_grad | exact_0_167 | exact | exact | exact | all_finite |
| grad_levelset | exact_0_167 | exact | exact | exact | all_finite |
| candidate_b | exact_0_167 | exact | exact | exact | all_finite |
| candidate_c | exact_0_167 | exact | exact | exact | all_finite |
| uv_crit | exact_0_167 | exact | exact | exact | all_finite |
| uv_e2 | exact_0_167 | exact | exact | exact | all_finite |
| b_e2 | exact_0_167 | exact | exact | exact | all_finite |
| c_e2 | exact_0_167 | exact | exact | exact | all_finite |
| f1_grad_e2 | exact_0_167 | exact | exact | exact | all_finite |
| f2_grad_levelset_e2 | exact_0_167 | exact | exact | exact | all_finite |
| f3_grad_crit | exact_0_167 | exact | exact | exact | all_finite |

## 9. Candidates that could not be included and why

All primary methods have real data in this checkout.

## 10. No training or TTK was rerun

This script and this audit performed zero training runs, zero TTK invocations, and zero cheap-evaluation runs. It only read pre-existing CSV files already present in the repository. No existing artifact was modified or deleted.

## 11. Generated file paths

- `ttk_runs_fixed/unified_candidate_evaluation/method_inventory.csv`
- `ttk_runs_fixed/unified_candidate_evaluation/unified_primary_per_sample_long.csv`
- `ttk_runs_fixed/unified_candidate_evaluation/column_mapping.csv`
- `ttk_runs_fixed/unified_candidate_evaluation/unified_primary_method_summary.csv`
- `ttk_runs_fixed/unified_candidate_evaluation/unified_primary_topology_validation.csv`
- `ttk_runs_fixed/unified_candidate_evaluation/unified_primary_pairwise_vs_cnn.csv`
- `ttk_runs_fixed/unified_candidate_evaluation/unified_primary_missingness.csv`
- `ttk_runs_fixed/unified_candidate_evaluation/unified_primary_wide.csv`
- `docs/unified_candidate_evaluation_inventory.md`
- `docs/primary_candidate_artifact_reference.md`
- `docs/unified_candidate_evaluation_phase1.md` (this file)
- `logs/build_unified_candidate_evaluation.log`

## 12. Recommended next step

The primary set is complete and strict-validated in this run. Per the task instructions, no correlation, factorial-model, Pareto-front, or visualization-selection work was performed in this Phase-1 pass -- that is the recommended next step.
