# Phase 2C: Unified Wind-SR Candidate Relationship and Pareto-Tradeoff Analysis

## 1. Scope

Phase 2C analyzes relationships among the 21 non-SSIM standardized metrics (SSIM is globally unavailable across all 19 methods and is excluded from every correlation and Pareto computation in this phase) and characterizes multi-objective tradeoffs among the 19 fixed, designed candidate methods evaluated in Phase 1. It is read-only with respect to every Phase-1, Phase-2A, and Phase-2B artifact: no training, inference, cheap evaluation, or TTK is run, and no prior-phase file is modified, regenerated, or overwritten. Sample selection and figure generation remain deferred to Phase 2D, which has not begun.

**No weighted aggregate score or total method ranking is computed anywhere in this phase.** Every summary table in this document is a machine-readable cross-reference of independently-defined quantities, not a leaderboard.

## 2. Why pooled 3,192-row correlations are not used

The Phase-1 long table has 3,192 rows (19 methods x 168 samples). A naive Pearson/Spearman correlation computed directly across all 3,192 rows would conflate two structurally different sources of covariation: (a) systematic between-method differences (some methods are simply better or worse on average across every metric) and (b) within-method sample-to-sample variation (which samples are easy or hard for a given method). These two sources can have different, even opposite, signs, and a pooled correlation reports an uninterpretable mixture of both. Phase 2C instead reports four analysis levels that isolate specific, clearly-labeled sources of covariation (Sections 4-7), plus a two-way-centered residual analysis (Section 7) that explicitly removes the additive method-mean and sample-mean effects before correlating -- this is the closest Phase 2C comes to a "pooled" correlation, and even that is on residuals, not raw pooled values.

## 3. Metric orientation convention

Every metric direction (`higher_is_better` / `lower_is_better`) is read exclusively from `column_mapping.csv` and never inferred from the metric name. The oriented value is `z = y` if higher-is-better, else `z = -y`, so that **larger oriented values always mean better performance** for every metric, and a positive oriented correlation always means "these two metrics agree about which methods/samples are better." SSIM is excluded from all orientation, correlation, and Pareto computations in this phase because it is 0/168 available for every method (globally unavailable, not merely missing for some methods).

## 4. Analysis A -- method-mean relationships (between_method_means)

For each of the 19 methods, the mean of every available metric is computed independently from the Phase-1 long table (`method_mean_oriented_values.csv`). Metric-pair correlations across these 19 method means answer: **"across the realized set of 19 designed methods, do two metrics favor the same methods on average?"** This says nothing about within-method sample-to-sample behavior. There are C(21,2) = 210 non-SSIM metric pairs; PD/MT-involving pairs use the 18 topology-bearing methods (bicubic has no topology data), all other pairs use all 19 methods. A leave-one-method-out (LOO) sensitivity bound is reported per pair: this is a descriptive perturbation diagnostic, not an inferential confidence interval -- the 19 methods are a fixed designed set, not a random sample, so no sampling-distribution interpretation is intended.

**PD/MT at the method-mean level:** oriented Pearson r = -0.3725, oriented Spearman rho = -0.4283 (n=18 common methods). Across the realized method means, oriented PD and MT association is non-positive -- i.e. methods with better (lower) mean PD distance tend to also have not consistently better mean MT distance.

## 5. Analysis B -- within-method sample relationships (within_method_across_samples)

For each of the 19 methods independently, metric-pair correlations are computed across the 168 samples (fields) that method was evaluated on. This answers: **"within a single fixed method, do two metrics identify the same easy/difficult samples?"** Every (method, pair) combination has either exactly 168 valid paired samples (`status=available`) or is explicitly marked `status=unavailable` (bicubic has no PD/MT data) -- partial coverage is never silently accepted; it hard-fails the run. `within_method_correlation_summary.csv` aggregates the per-method correlations into median/quartile/min/max/sign-count statistics across the methods for which that pair is available.

**PD/MT within individual methods:** median oriented Pearson r = 0.3883, median oriented Spearman rho = 0.2991 across 18 methods. Within individual methods across the 168 fields, the median association is positive.

## 6. Analysis C -- per-sample cross-method relationships (within_sample_across_methods)

For each of the 168 samples independently, metric-pair correlations are computed across the methods evaluated on that sample. This answers: **"for a fixed field, do different metrics rank the 19 (or 18, for PD/MT pairs) methods similarly?"** PD/MT-involving pairs use exactly the 18 topology-bearing methods; all other pairs use all 19 methods -- these counts are enforced exactly, never partially. `samplewise_correlation_summary.csv` aggregates across the 168 samples.

**PD/MT per sample:** median oriented Pearson r = -0.2623, median oriented Spearman rho = -0.1538 across 168 samples. For a fixed field, PD and MT agree on how they rank the methods at a rate reflected by this median association; see Section 8 for the direct preference-agreement rates, which are a more literal reading of "agree/disagree."

## 7. Analysis D -- two-way-centered residual relationships (two_way_centered_residual)

For each metric, the method main effect and sample main effect are removed by additive demeaning: `residual[m,s] = z[m,s] - mean_over_samples(z[m,*]) - mean_over_methods(z[*,s]) + grand_mean(z)`. By construction every row mean, column mean, and the grand mean of the residual matrix is numerically zero (verified to within 1e-09; see `phase2c_validation.csv`). Correlating the residuals of two metrics (pooled over the common method x sample cells) answers: **"once the obvious method-level and sample-level main effects are removed, do two metrics still move together?"** This is explicitly **additive demeaning, not a fitted causal mixed-effects model** -- no variance components, random effects, or significance testing are involved, and no causal claim is intended or supported.

**PD/MT two-way-residual association:** Pearson r = 0.0526, Spearman rho = 0.1027 (n_cells=3024).

## 8. PD/MT direct disagreement analysis

Beyond correlation, Phase 2C directly quantifies how often PD (persistence-diagram distance) and MT (merge-tree distance) *disagree about which of two methods is better*, at both a per-method-pair level (aggregated over the 168 fields, `topology_pairwise_preference_agreement.csv`) and a per-field level (aggregated over the C(18,2)=153 topology-bearing method pairs, `topology_sample_preference_agreement.csv`). PD and MT are both legitimate, differently-scoped topological descriptors (PD captures persistence-pair geometry, MT captures merge-tree structure); disagreement between them is evidence of genuinely different geometric sensitivity, and is not evidence that either descriptor is invalid.

- **between_method_means** (n=18): oriented_pearson=-0.3725, oriented_spearman=-0.4283
- **within_method_across_samples_median** (n=18): median_oriented_pearson=0.3883, median_oriented_spearman=0.2991
- **within_sample_across_methods_median** (n=168): median_oriented_pearson=-0.2623, median_oriented_spearman=-0.1538
- **pairwise_preference_agreement** (n=153): mean_agreement_rate=0.4384, median_agreement_rate=0.4464
- **sample_preference_agreement** (n=168): mean_agreement_rate=0.4384, median_agreement_rate=0.4542

## 9. Pareto front definition

All Pareto analysis operates on **oriented method means** (Section 3), where higher is always better. Method A strictly dominates method B on a given objective set when A >= B - 1e-12 on every objective in the set AND A > B + 1e-12 on at least one objective. The Pareto front of an objective set is the set of methods dominated by no other eligible method. No metric normalization is applied or needed: strict dominance under this definition is invariant to any monotonic per-objective rescaling.

## 10. Pareto objective-set choices

Six objective sets are defined, each built from three metric groups: **fidelity** (`psnruv`, `speed_mae`), **physics** (`wpd_mae`, `grad_mae`, `psd_log_l2`, `exceed_abs_p90`, `comp_curve_l1`), and **topology** (`pd_distance`, `mt_distance`). Any objective set that includes a topology objective is restricted to the 18 topology-bearing methods (bicubic has no PD/MT data); sets without a topology objective use all 19 methods.

- **topology_only**: 2 objectives = ['pd_distance', 'mt_distance']
- **fidelity_physics_compact**: 7 objectives = ['psnruv', 'speed_mae', 'wpd_mae', 'grad_mae', 'psd_log_l2', 'exceed_abs_p90', 'comp_curve_l1']
- **fidelity_topology**: 4 objectives = ['psnruv', 'speed_mae', 'pd_distance', 'mt_distance']
- **physics_topology**: 7 objectives = ['wpd_mae', 'grad_mae', 'psd_log_l2', 'exceed_abs_p90', 'comp_curve_l1', 'pd_distance', 'mt_distance']
- **cross_family_compact**: 9 objectives = ['psnruv', 'speed_mae', 'wpd_mae', 'grad_mae', 'psd_log_l2', 'exceed_abs_p90', 'comp_curve_l1', 'pd_distance', 'mt_distance']
- **all_available_non_ssim** (labeled sensitivity analysis -- all 21 non-SSIM objectives at once): 21 objectives = ['psnruv', 'speed_mae', 'speed_rmse', 'wpd_mae', 'wpd_w1', 'wpd_bias_abs', 'grad_mae', 'grad_w1', 'grad_kurtosis_abs_delta', 'psd_log_l2', 'psd_slope_abs_delta', 'exceed_abs_t5', 'exceed_abs_t10', 'exceed_abs_t15', 'exceed_abs_p90', 'comp_curve_l1', 'comp_abs_t5', 'comp_abs_t10', 'comp_abs_t15', 'pd_distance', 'mt_distance']

**A Pareto front depends on the chosen objective set and is not a universal ranking.** A method can be on the front for one objective set and dominated under another; Section 11 makes this concrete.

## 11. Deterministic Pareto fronts

- **topology_only**: front = ['f1_grad_e2', 'f2_grad_levelset_e2', 'f3_grad_crit', 'gan', 'uv_e2']
- **fidelity_physics_compact**: front = ['b_e2', 'c_e2', 'candidate_b', 'candidate_c', 'f1_grad_e2', 'f2_grad_levelset_e2', 'f3_grad_crit', 'gan', 'grad_levelset', 'grad_only', 'levelset_only', 'speed_grad', 'speed_levelset', 'speed_only', 'uv', 'uv_crit', 'uv_e2']
- **fidelity_topology**: front = ['b_e2', 'c_e2', 'candidate_b', 'candidate_c', 'f1_grad_e2', 'f2_grad_levelset_e2', 'f3_grad_crit', 'gan', 'grad_only', 'levelset_only', 'speed_levelset', 'speed_only', 'uv_crit', 'uv_e2']
- **physics_topology**: front = ['b_e2', 'c_e2', 'candidate_b', 'candidate_c', 'f1_grad_e2', 'f2_grad_levelset_e2', 'f3_grad_crit', 'gan', 'grad_levelset', 'grad_only', 'speed_grad', 'speed_levelset', 'uv_crit', 'uv_e2']
- **cross_family_compact**: front = ['b_e2', 'c_e2', 'candidate_b', 'candidate_c', 'f1_grad_e2', 'f2_grad_levelset_e2', 'f3_grad_crit', 'gan', 'grad_levelset', 'grad_only', 'levelset_only', 'speed_grad', 'speed_levelset', 'speed_only', 'uv', 'uv_crit', 'uv_e2']
- **all_available_non_ssim**: front = ['b_e2', 'c_e2', 'candidate_b', 'candidate_c', 'cnn', 'f1_grad_e2', 'f2_grad_levelset_e2', 'f3_grad_crit', 'gan', 'grad_levelset', 'grad_only', 'levelset_only', 'speed_grad', 'speed_levelset', 'speed_only', 'uv', 'uv_crit', 'uv_e2']

**Topology-only sanity check:** the deterministic `topology_only` front was independently required to equal exactly `['f1_grad_e2', 'f2_grad_levelset_e2', 'f3_grad_crit', 'gan', 'uv_e2']`, derived directly from the Phase-1 long-table method means; the run hard-fails if it does not. It matched exactly.

## 12. Pareto bootstrap stability

For every objective set, front membership is recomputed under 10000 resamples per scheme, across four resampling schemes: ordinary i.i.d. resampling of the 168 fields, and circular moving-block bootstrap with block lengths 6, 12, and 24 (to probe robustness to the fact that the 168 fields are consecutive hourly observations and are likely temporally dependent). Within a single replicate, the identical resampled set of field indices is used for every method and every objective in the objective set, so that all method means being compared in that replicate are built from the same synthetic sample. The resulting **front-membership rate is a descriptive stability diagnostic under resampling, explicitly not a posterior probability** -- no prior or likelihood model is specified.

For `topology_only` under i.i.d. resampling: always-on-front = ['f2_grad_levelset_e2', 'f3_grad_crit', 'gan'], never-on-front = ['candidate_b', 'cnn', 'grad_levelset', 'levelset_only', 'speed_grad', 'speed_levelset', 'speed_only', 'uv']. See `pareto_bootstrap_stability.csv` for the full objective-set x scheme x method table, and `pareto_bootstrap_front_size.csv` for how front size itself varies under resampling.

## 13. Findings that are consistent across analysis levels

See `metric_relationship_summary.csv` for the full per-pair cross-level comparison. A pair is labeled `consistent_nonzero_sign=True` when the method-level, within-method-median, samplewise-median, and two-way-residual oriented Pearson correlations all share the same nonzero sign -- this is a statement about **sign** agreement across genuinely different analysis levels, not about equal magnitude, and not a claim that any one of the four coefficients is more "correct" than another.

## 14. Findings that reverse or weaken across analysis levels

Metric pairs NOT flagged `consistent_nonzero_sign` in `metric_relationship_summary.csv` have a sign that differs, or a coefficient that is exactly zero or undefined, across at least one of the four analysis levels. This is expected and scientifically meaningful: a pair can, for instance, favor the same methods on average (positive method-level correlation) while showing no consistent within-method sample-to-sample relationship (near-zero within-method median), because these two levels answer different questions (Sections 4 and 5).

## 15. Caveats

- All 19 methods were trained exactly once; none of the correlation or Pareto results here should be read as capturing training-run-to-training-run variance.
- The 19 methods are a fixed, designed candidate set, not a random sample from a broader population of possible architectures -- method-level (Analysis A) statistics, including the LOO sensitivity bounds, are descriptive, not inferential.
- The 168 benchmark fields are consecutive hourly observations and are likely temporally dependent; this is why every bootstrap procedure in this phase includes circular moving-block resampling alongside ordinary i.i.d. resampling.
- Correlation is not causation anywhere in this document, including the two-way-residual analysis (Section 7), which removes additive main effects but does not fit a causal model.
- Analysis levels are not interchangeable: method-mean, within-method, cross-method, and two-way-residual correlations answer different questions and can legitimately disagree (Section 14).

## 16. No weighted score, no total ranking

Phase 2C never computes a weighted combination of metrics, never produces an overall method score, and never produces a total ranking of the 19 methods. `pareto_layers.csv` reports iterative non-domination layers (an "onion peeling" of the dominance structure), which is explicitly **not** a total ranking: methods within the same layer are mutually non-dominated, and layer order reflects dominance depth under one specific objective set, not overall quality. `topology_relationship_and_pareto_summary.csv` is a machine-readable cross-reference table, also explicitly not a leaderboard.

## 17. Sample selection and figures deferred to Phase 2D

This phase performs no sample selection and generates no figures. Both remain deferred to Phase 2D, which has not begun.

## 18. Validation summary

970 total validation checks were run; 970 passed and 0 failed (a run with any failure hard-fails before this document is written, so `n_fail` is always 0 in a completed report). See `phase2c_validation.csv` for the full check list.

## 19. Generated files

- `ttk_runs_fixed/unified_candidate_analysis/phase2c/phase2c_validation.csv`
- `ttk_runs_fixed/unified_candidate_analysis/phase2c/method_mean_oriented_values.csv`
- `ttk_runs_fixed/unified_candidate_analysis/phase2c/method_level_metric_correlations.csv`
- `ttk_runs_fixed/unified_candidate_analysis/phase2c/method_level_oriented_pearson_matrix.csv`
- `ttk_runs_fixed/unified_candidate_analysis/phase2c/method_level_oriented_spearman_matrix.csv`
- `ttk_runs_fixed/unified_candidate_analysis/phase2c/within_method_metric_correlations.csv`
- `ttk_runs_fixed/unified_candidate_analysis/phase2c/within_method_correlation_summary.csv`
- `ttk_runs_fixed/unified_candidate_analysis/phase2c/samplewise_cross_method_correlations.csv`
- `ttk_runs_fixed/unified_candidate_analysis/phase2c/samplewise_correlation_summary.csv`
- `ttk_runs_fixed/unified_candidate_analysis/phase2c/two_way_residual_correlations.csv`
- `ttk_runs_fixed/unified_candidate_analysis/phase2c/two_way_residual_pearson_matrix.csv`
- `ttk_runs_fixed/unified_candidate_analysis/phase2c/two_way_residual_spearman_matrix.csv`
- `ttk_runs_fixed/unified_candidate_analysis/phase2c/focal_topology_correlation_bootstrap.csv`
- `ttk_runs_fixed/unified_candidate_analysis/phase2c/focal_topology_relationship_summary.csv`
- `ttk_runs_fixed/unified_candidate_analysis/phase2c/topology_rank_by_method.csv`
- `ttk_runs_fixed/unified_candidate_analysis/phase2c/topology_pairwise_preference_agreement.csv`
- `ttk_runs_fixed/unified_candidate_analysis/phase2c/topology_sample_preference_agreement.csv`
- `ttk_runs_fixed/unified_candidate_analysis/phase2c/topology_descriptor_disagreement_summary.csv`
- `ttk_runs_fixed/unified_candidate_analysis/phase2c/pareto_objective_manifest.csv`
- `ttk_runs_fixed/unified_candidate_analysis/phase2c/pareto_front_membership.csv`
- `ttk_runs_fixed/unified_candidate_analysis/phase2c/pareto_dominance_edges.csv`
- `ttk_runs_fixed/unified_candidate_analysis/phase2c/pareto_layers.csv`
- `ttk_runs_fixed/unified_candidate_analysis/phase2c/topology_pareto_sanity_check.csv`
- `ttk_runs_fixed/unified_candidate_analysis/phase2c/pareto_bootstrap_stability.csv`
- `ttk_runs_fixed/unified_candidate_analysis/phase2c/pareto_bootstrap_front_size.csv`
- `ttk_runs_fixed/unified_candidate_analysis/phase2c/metric_relationship_summary.csv`
- `ttk_runs_fixed/unified_candidate_analysis/phase2c/topology_relationship_and_pareto_summary.csv`
- `ttk_runs_fixed/unified_candidate_analysis/phase2c/prior_phase_immutability_check.csv`
- `docs/unified_candidate_analysis_phase2c.md` (this file)
- `logs/unified_candidate_analysis_phase2c.log`

