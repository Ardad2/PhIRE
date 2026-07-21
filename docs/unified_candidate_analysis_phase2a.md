# Unified candidate analysis -- Phase 2A report

## 1. Scope and authoritative inputs

Phase 2A is a deterministic descriptive and paired multi-metric analysis of all 19 primary methods, built exclusively from the immutable Phase-1 outputs under `ttk_runs_fixed/unified_candidate_evaluation/` (`unified_primary_per_sample_long.csv` as the source of truth; `unified_primary_method_summary.csv`, `unified_primary_pairwise_vs_cnn.csv`, `unified_primary_topology_validation.csv`, `unified_primary_missingness.csv`, `method_inventory.csv`, and `column_mapping.csv` as validation references). No Phase-1 file was modified, regenerated, or overwritten. No training, inference, cheap evaluation, or TTK was run.

Explicitly deferred to later Phase-2 stages (not performed here): speed x gradient x level-set factorial decomposition, targeted E2/critical-proxy contrasts, metric-correlation analysis, Pareto-front analysis, sample selection, visualization generation, and composite/weighted ranking.

## 2. Validation results

All 1027 independent validation checks passed: exact row/method/sample counts, no duplicate (method_id, sample_idx) keys, constant per-method metadata, SSIM at either 168/168 or 0/168 per method, finite/nonnegative PD and MT wherever topology data exists, exact reproduction of every `unified_primary_method_summary.csv` mean, and exact reproduction of every `unified_primary_topology_validation.csv` PD/MT mean. See `phase2a_validation.csv` for the full per-check ledger. Any failure here would have hard-failed the run before any other output was written.
Additionally, all 3564 field-level comparisons against `unified_primary_pairwise_vs_cnn.csv` (an independent re-derivation using Phase-1's exact algorithm, not a copy) passed -- see `phase1_pairwise_reproduction.csv` and section 5.

## 3. Metric coverage and SSIM status

22 metrics x 19 methods = 418 (method, metric) coverage rows in `metric_coverage.csv`. SSIM (`ssim_speed`) is the only metric allowed to be globally unavailable (the documented NumPy/scikit-image ABI issue): fully available (168/168) for []; globally unavailable (0/168) for ['b_e2', 'bicubic', 'c_e2', 'candidate_b', 'candidate_c', 'cnn', 'f1_grad_e2', 'f2_grad_levelset_e2', 'f3_grad_crit', 'gan', 'grad_levelset', 'grad_only', 'levelset_only', 'speed_grad', 'speed_levelset', 'speed_only', 'uv', 'uv_crit', 'uv_e2']. No value was ever imputed -- missing cells stay empty in every generated CSV.

## 4. Descriptive method-level results

`method_descriptive_summary.csv` has 418 rows (one per method x metric): mean, sample standard deviation, standard error, median, quartiles, min/max, and a 95% bootstrap CI of the mean (10,000 resamples, seed 20260721, sample-axis resampling of the 168 benchmark indices). Rows with `n_valid=0` (e.g. bicubic PD/MT, or a method whose SSIM is globally unavailable) leave every numeric field empty rather than reporting a fabricated statistic.

## 5. Paired comparison methodology

For every non-CNN method x metric, `paired_vs_cnn_detailed.csv` restricts to samples where BOTH CNN and the method have a finite value (`n_valid_pairs`), then computes the direction-aware `improvement` (positive always means better than CNN; `raw_delta = method - cnn` is preserved separately), win/tie/loss counts using a 1e-12 tie tolerance, an exact two-sided sign test over non-tied pairs only (no SciPy required), the paired effect size `dz = mean(improvement) / std(improvement)` (empty when the paired standard deviation is zero or undefined), and the same deterministic bootstrap for the improvement 95% CI. A two-sided paired Wilcoxon signed-rank test (SciPy) was also computed. 396 method x metric rows were produced; 376 had at least one valid pair. `paired_vs_cnn_adjusted.csv` adds Holm-corrected p-values, computed both once across all valid comparisons (`_holm_global`) and separately within each metric across methods (`_holm_within_metric`), for both the sign test and (when available) Wilcoxon. No effect is labeled "significant" from these p-values alone -- the adjusted values are preserved for later interpretation. `method_mean_improvement_matrix.csv` and `method_win_rate_matrix.csv` pivot the mean improvement and win rate into a method x metric matrix; no aggregate ranking or weighted score was computed.

## 6. Benchmark-sample uncertainty caveat

The 168 samples are paired benchmark observations, not independent training runs -- every model here was trained exactly once. The bootstrap confidence intervals and sign-test/Wilcoxon p-values in this report quantify variability **across the 168 benchmark samples only**. They do not, and cannot, quantify variation across independent training seeds or reruns. Results here therefore support benchmark-level comparisons (does this trained model do better than CNN on this fixed evaluation set?) but not claims about training-run robustness (would a differently-seeded retraining of the same objective reproduce this result?).

A second, independent caveat concerns the samples themselves: the 168 benchmark fields are consecutive hourly wind samples, and are therefore likely temporally correlated rather than independent and identically distributed. The ordinary sample-level bootstrap used throughout this report, the exact sign-test p-values, and any Wilcoxon p-values all rely on an independence-across-samples approximation; under real temporal autocorrelation they may understate the true sampling uncertainty (i.e. be anti-conservative -- confidence intervals narrower, and p-values smaller, than an analysis that properly accounted for the autocorrelation would produce). This does not affect the descriptive quantities themselves: means, medians, paired deltas, and win rates in this report remain valid, exact descriptive summaries of this fixed 168-sample benchmark regardless of any temporal correlation. What it affects is the *inferential* quantities layered on top of those summaries -- the bootstrap CIs and the sign-test/Wilcoxon p-values should not be read as fully calibrated population-level inference, and should be treated as approximate, sample-independence-assuming quantities rather than exact ones. A temporal block-bootstrap sensitivity analysis (resampling contiguous runs of samples rather than single samples independently, to see whether the CIs/p-values widen materially) may be worth considering in a later phase, but is explicitly not performed here -- this patch documents the caveat only and adds no new calculation to Phase 2A.

## 7. Topology tradeoff summary

`topology_tradeoff_summary.csv` has one row per method (19 total); quadrant counts: {'improves_both': 5, 'topology_unavailable': 1, 'improves_pd_only': 7, 'cnn_reference': 1, 'improves_neither': 4, 'improves_mt_only': 1}. `bicubic` is marked `topology_unavailable` (it has no PD/MT source, consistent with Phase-1); `cnn` is marked `cnn_reference`. `topology_tradeoff_summary_sorted.csv` sorts by (topology_quadrant, pd_mean, mt_mean) -- this is a display ordering only, not a claimed total ranking.

## 8. Strongest descriptive patterns

Methods whose mean PD **and** mean MT both improve over CNN on this benchmark: ['b_e2', 'c_e2', 'f1_grad_e2', 'f2_grad_levelset_e2', 'uv_e2']. Methods whose mean PD and mean MT both fail to improve over CNN: ['levelset_only', 'speed_levelset', 'speed_only', 'uv']. This is a purely descriptive observation from the paired means above (section 7) -- it is not a causal claim about which loss term drove the result and not a ranking; per-metric win rates and confidence intervals in `paired_vs_cnn_detailed.csv` should be consulted before drawing conclusions about any individual method.

## 9. Deferred analyses

Causal loss-term attribution (e.g. isolating the effect of `L_grad` via the B-factorial ablation), metric correlation analysis, Pareto-front analysis, and sample-level visualization selection are all explicitly deferred to later Phase-2 stages. Nothing in this report should be read as performing that attribution.

## 10. Generated file list

- `ttk_runs_fixed/unified_candidate_analysis/phase2a/phase2a_validation.csv`
- `ttk_runs_fixed/unified_candidate_analysis/phase2a/metric_coverage.csv`
- `ttk_runs_fixed/unified_candidate_analysis/phase2a/method_descriptive_summary.csv`
- `ttk_runs_fixed/unified_candidate_analysis/phase2a/paired_vs_cnn_detailed.csv`
- `ttk_runs_fixed/unified_candidate_analysis/phase2a/paired_vs_cnn_adjusted.csv`
- `ttk_runs_fixed/unified_candidate_analysis/phase2a/method_mean_improvement_matrix.csv`
- `ttk_runs_fixed/unified_candidate_analysis/phase2a/method_win_rate_matrix.csv`
- `ttk_runs_fixed/unified_candidate_analysis/phase2a/topology_tradeoff_summary.csv`
- `ttk_runs_fixed/unified_candidate_analysis/phase2a/topology_tradeoff_summary_sorted.csv`
- `ttk_runs_fixed/unified_candidate_analysis/phase2a/phase1_pairwise_reproduction.csv`
- `ttk_runs_fixed/unified_candidate_analysis/phase2a/phase1_immutability_check.csv`
- `docs/unified_candidate_analysis_phase2a.md` (this file)
- `logs/unified_candidate_analysis_phase2a.log`

