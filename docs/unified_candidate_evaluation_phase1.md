# Unified candidate evaluation -- Phase 1 report

## 0. Run mode

**AUDIT MODE (`--audit-allow-missing`).** This is a permissive inventory pass: the unified table below intentionally includes empty cells for any method/metric with no source artifact in this checkout. Do not treat it as authoritative -- see section 9.

## 1. Scope and primary/secondary method distinction

Primary set: 19 methods evaluated on the fixed 168-sample benchmark (3 baselines: bicubic, cnn, gan; 9 B-term-factorial variants incl. full Candidate B and C; 1 critical-proxy-only ablation; 3 repaired low-lambda E2 ablations; 3 Candidate F recombinations). Secondary set: 168-sample pilot runs, 672/1344-sample scale-study duplicates of primary objectives, PyTorch residual-refiner E2 variants (architecture confound), and deprecated pre-Phase-C legacy archives -- see `docs/unified_candidate_evaluation_inventory.md` for the full secondary listing.

## 2. Complete artifact inventory

See `ttk_runs_fixed/unified_candidate_evaluation/method_inventory.csv`, `docs/unified_candidate_evaluation_inventory.md`, and `docs/primary_candidate_artifact_reference.md`.

## 3. Exact table dimensions

- `unified_primary_per_sample_long.csv`: 19 methods x 168 samples = 3192 rows, one row per (method_id, sample_idx), no duplicates.
- Of these 19 primary methods, **2 have real per-sample data** in this repository checkout (cnn, gan), and **17 have zero real data** (bicubic, uv, speed_only, levelset_only, speed_levelset, grad_only, speed_grad, grad_levelset, candidate_b, candidate_c, uv_crit, uv_e2, b_e2, c_e2, f1_grad_e2, f2_grad_levelset_e2, f3_grad_crit).

## 4. Metric families and representations

See `column_mapping.csv`. Families: `vector_uv` (psnruv), `scalar_speed` (ssim_speed, speed_mae, speed_rmse), `wind_power_distribution` (wpd_*), `gradient_distribution` (grad_*), `frequency_domain` (psd_*), `threshold_geometry` (exceed_*, comp_*), `topology_pd` (pd_distance), `topology_mt` (mt_distance). PSNR is vector-field PSNR on physical [u,v] (`psnruv`), never scalar-speed PSNR; SSIM, speed errors, WPD, PD, and MT are all computed on scalar wind speed.

## 5. Validation results

- Topology-mean reproduction: **2 PASS**, **0 FAIL**, **16 NO_DATA** out of 18 primary methods with an expected value.
- Cheap-metric completeness (168 rows, sample_idx exactly 0..167, no duplicates): checked for every primary method with any discovered source (baseline-harvested, legacy-combined, or a resolved candidate all_sample_metrics CSV).
- Join (cheap metrics <-> true topology, one-to-one on sample_idx): every per-sample record is a single merged dict keyed by sample_idx, so a missing cheap or topology value for a given sample_idx shows up directly as a non-finite cell rather than a silent row-count mismatch.

## 6. Baseline duplicate-consistency audit

Discovered 0 candidate `all_sample_metrics_*.csv` file(s) under `ttk_runs_fixed/topology_finetuning/*_eval/`. Every discovered file was checked for bicubic/cnn/gan rows; every baseline method present in more than one file had its rows compared pairwise, metric by metric, against a 1e-06 tolerance (would hard-fail the whole run on disagreement):
  - **bicubic**: 0 source(s) with data; canonical source = `(none found)`.
  - **cnn**: 0 source(s) with data; canonical source = `(none found)`.
  - **gan**: 0 source(s) with data; canonical source = `(none found)`.

Canonical cnn/gan rows were additionally cross-checked against the older `ttk_runs_fixed/combined/psnr_topology_physics_merged.csv` pipeline for overlapping columns (tolerance 0.001):
  - **cnn**: skipped (harvested candidate-eval rows not found for this method in this checkout).
  - **gan**: skipped (harvested candidate-eval rows not found for this method in this checkout).

The independent `ttk_runs_fixed/combined/phase_c_results.csv` PD/MT cross-check remains as before:
  - **cnn**: max |Δpd| = 0.000e+00, max |Δmt| = 0.000e+00 across 168 cross-checked samples.
  - **gan**: max |Δpd| = 0.000e+00, max |Δmt| = 0.000e+00 across 168 cross-checked samples.

## 7. PD/MT mean reproduction table

| method_id | observed_pd_mean | expected_pd_mean | pd_pass | observed_mt_mean | expected_mt_mean | mt_pass |
|---|---:|---:|---|---:|---:|---|
| cnn | 27.4063 | 27.4063 | True | 5.8678 | 5.8678 | True |
| gan | 20.8641 | 20.8641 | True | 8.3481 | 8.3481 | True |
| uv | n/a | 29.6121 | NO_DATA | n/a | 6.0119 | NO_DATA |
| speed_only | n/a | 29.5783 | NO_DATA | n/a | 5.9996 | NO_DATA |
| levelset_only | n/a | 29.5953 | NO_DATA | n/a | 6.0076 | NO_DATA |
| speed_levelset | n/a | 29.4363 | NO_DATA | n/a | 5.9441 | NO_DATA |
| grad_only | n/a | 22.9326 | NO_DATA | n/a | 6.056 | NO_DATA |
| speed_grad | n/a | 22.9706 | NO_DATA | n/a | 6.2905 | NO_DATA |
| grad_levelset | n/a | 22.6194 | NO_DATA | n/a | 6.1996 | NO_DATA |
| candidate_b | n/a | 22.707 | NO_DATA | n/a | 6.1612 | NO_DATA |
| candidate_c | n/a | 22.4944 | NO_DATA | n/a | 6.0803 | NO_DATA |
| uv_crit | n/a | 29.1143 | NO_DATA | n/a | 5.6899 | NO_DATA |
| uv_e2 | n/a | 25.0721 | NO_DATA | n/a | 5.594 | NO_DATA |
| b_e2 | n/a | 23.9876 | NO_DATA | n/a | 5.6774 | NO_DATA |
| c_e2 | n/a | 24.2686 | NO_DATA | n/a | 5.6628 | NO_DATA |
| f1_grad_e2 | n/a | 23.8382 | NO_DATA | n/a | 5.6566 | NO_DATA |
| f2_grad_levelset_e2 | n/a | 23.7481 | NO_DATA | n/a | 5.6742 | NO_DATA |
| f3_grad_crit | n/a | 22.0179 | NO_DATA | n/a | 5.984 | NO_DATA |

## 8. Missingness, especially SSIM

- SSIM (`ssim_speed`): finite for ['cnn', 'gan']. Entirely missing (no source at all, not the known NumPy/scikit-image ABI issue) for the other 17 primary methods in this checkout.
- See `unified_primary_missingness.csv` for the full total/finite/missing breakdown per (method_id, metric).
- No missing value was filled with zero or inferred; all gaps are empty cells in the CSVs.

## 9. Candidates that could not be included and why

17 of 19 primary methods (bicubic, uv, speed_only, levelset_only, speed_levelset, grad_only, speed_grad, grad_levelset, candidate_b, candidate_c, uv_crit, uv_e2, b_e2, c_e2, f1_grad_e2, f2_grad_levelset_e2, f3_grad_crit) have **zero** real per-sample cheap-evaluation or true-topology artifacts anywhere in this git checkout. Root cause: this repository's `.gitignore` excludes `*.npy`, `*.npz`, `data_out/`, and `ttk_runs_fixed/topology_finetuning/*` (tracked exceptions are only `candidateE_constraints` and the cnn/gan `combined`/`phase_c_final` summary artifacts); large experiment outputs for the loss-ablation candidates are produced only on the separate training machine and were never committed.

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

Before any factorial-effect analysis, paired contrasts, correlations, or Pareto-front work can be performed on the full primary set, the 17 missing methods' cheap-evaluation and true-topology artifacts need to be synced from the training machine into this checkout (or this script re-run there with `--strict-primary`). Until then, any such analysis is only valid for the 2 methods with real data (cnn, gan).
