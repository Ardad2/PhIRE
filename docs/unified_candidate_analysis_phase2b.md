# Unified candidate analysis -- Phase 2B report

## 1. Scope and frozen inputs

Phase 2B performs controlled factorial and targeted matched-pair analyses on top of the now-immutable Phase-1 (`ttk_runs_fixed/unified_candidate_evaluation/`) and Phase-2A (`ttk_runs_fixed/unified_candidate_analysis/phase2a/`) outputs. `unified_primary_per_sample_long.csv` remains the numeric source of truth throughout; every Phase-2A file is a validation reference only. 26 prior-phase files (12 Phase-1 + 14 Phase-2A) were required to exist, checksummed before and after this run, and confirmed byte-for-byte unchanged -- see `prior_phase_immutability_check.csv`. No training, inference, cheap evaluation, or TTK was run. Metric correlations, Pareto-front analysis, sample selection, and visualization remain deferred to Phase 2C/2D.

## 2. Factor coding and exact factorial-effect convention

Factor levels are coded disabled=-1, enabled=+1. For a raw metric value y with direction read from `column_mapping.csv` (never inferred from the metric name), the oriented value is z=y when higher-is-better and z=-y when lower-is-better, so a positive oriented effect always means improvement. For a complete, balanced 2^k design, the saturated coded regression coefficient for any nonempty factor subset J is beta_J = mean over the 2^k cells of (value * product of coded x_j for j in J) -- exact because the coded design columns are mutually orthogonal with squared norm 2^k, so this is exactly the ordinary-least-squares solution. The reported `factorial_effect` is `2 * beta_J`: a main effect is the average high-minus-low difference, and an interaction is the standard balanced factorial interaction (half of the unaveraged two-cell difference-of-differences in a simple 2^2 design). Every sample-level effect and every targeted contrast reports both `raw_*` (untransformed) and `oriented_*` (direction-applied, positive=improvement) versions side by side; nothing is ever reported in only one form.

## 3. Full B 2^3 design

Cells (uv, speed_only, levelset_only, speed_levelset, grad_only, speed_grad, grad_levelset, candidate_b) span the complete `speed x grad x levelset` design; method metadata was verified to match the declared coding exactly for every cell before any statistic was computed. Seven effects were estimated: speed, grad, levelset, speed:grad, speed:levelset, grad:levelset, speed:grad:levelset. `b_factorial_reconstruction_check.csv` has 29,568 rows (168 samples x 22 metrics x 8 cells); of these, 28,224 finite values were exactly reconstructed from their 8 saturated coefficients (max abs error <= 1e-12), and 1,344 are globally-unavailable SSIM entries retained as `no_data` rather than reconstructed or fabricated. Every effect's task-provided PD/MT sanity-check target was reproduced from the validated method means within 0.0001.

## 4. B-scaffold critical x E2 2^2 design

Cells (candidate_b, candidate_c, b_e2, c_e2) hold the speed+grad+levelset scaffold fixed and vary only the critical-maxima proxy and repaired E2. Effects estimated: crit, e2, crit:e2. `b_scaffold_crit_e2_reconstruction_check.csv` has 14,784 rows (168 x 22 x 4 cells); 14,112 finite values reconstructed within 1e-12, 672 SSIM entries retained as `no_data`. Same reconstruction gate, sanity-check reproduction, and bootstrap methodology as Analysis A.

## 5. Gradient-scaffold level-set x E2 2^2 design

Cells (grad_only, grad_levelset, f1_grad_e2, f2_grad_levelset_e2) hold the gradient term fixed and vary level-set and repaired E2. Effects estimated: levelset, e2, levelset:e2. `grad_scaffold_levelset_e2_reconstruction_check.csv` has 14,784 rows (168 x 22 x 4 cells); 14,112 finite values reconstructed within 1e-12, 672 SSIM entries retained as `no_data`. Same gates and methodology as Analyses A and B.

## 6. Targeted matched contrasts

15 matched-pair contrasts across four families: adding the critical-maxima proxy (4, all single-term), adding repaired E2 (5, all single-term), E2 versus the critical proxy on a matched scaffold (3, a two-flag substitution, not single-term), and minimality/scaffold-pruning (3, explicitly labeled composite regardless of how many flags literally differ, per the requested convention for this family). A positive oriented improvement always means the comparison method is better than the base method. See `targeted_contrast_manifest.csv` for the full base/comparison/term/interpretation table.

## 7. Ordinary and temporal-block bootstrap methodology

Every sample-level factorial effect and every targeted contrast reports four 95% confidence intervals of its mean: an ordinary sample-axis (iid) bootstrap, and three deterministic circular moving-block bootstraps with block lengths 6, 12, and 24 hours. All four use 10,000 resamples with seeds derived from 20260721 (the block bootstraps use `20260721 * 1000 + block_length`); each resample draws block start positions uniformly from 0..167 with replacement, appends consecutive circular blocks (wrapping past sample 167 back to sample 0), and truncates the concatenated index sequence to exactly 168. The 168 benchmark samples are consecutive hourly wind fields and are likely temporally correlated; the block-length sensitivity comparison in section 11 is exploratory evidence about how much that correlation might matter for a given effect, not proof of a correct dependence model or a formally validated block length.

## 8. Multiple-testing correction

`phase2b_multiple_testing_adjusted.csv` has 616 rows (every effect/contrast x metric combination across all four analyses); 588 carry a valid exact sign-test p-value and 0 carry a valid Wilcoxon p-value (SciPy was not importable in this environment, so every Wilcoxon field is empty and the run completed without it, as required). Holm step-down correction was applied three ways: once globally across every valid comparison, once within each metric across all effects/contrasts, and once within each analysis family. No binary "significant" field was created, and the adjusted values retain the temporal-independence caveat from section 7 even after correction.

## 9. PD and MT findings

Within the realized B 2^3 factorial, enabling gradient has the largest positive mean oriented effect on PD distance (6.7481, win rate 1.0000); within the realized B-scaffold crit x E2 design, adding repaired E2 has the largest positive mean oriented effect on MT distance (0.4506, win rate 0.8631). These are factorial contrasts among the realized trained models in each specific design on this fixed benchmark, not general causal statements about the loss terms -- see `b_factorial_effect_summary.csv`, `b_scaffold_crit_e2_effect_summary.csv`, and `grad_scaffold_levelset_e2_effect_summary.csv` for every effect on every metric, and `topology_factorial_and_contrast_summary.csv` for the PD/MT-only extract across all four analyses.

## 10. Patterns across metric families

`b_factorial_oriented_effect_matrix.csv`, `b_scaffold_crit_e2_oriented_effect_matrix.csv`, `grad_scaffold_levelset_e2_oriented_effect_matrix.csv`, and `targeted_contrast_oriented_effect_matrix.csv` give the mean oriented effect for every effect/contrast against every metric family (`vector_uv`, `scalar_speed`, `wind_power_distribution`, `gradient_distribution`, `frequency_domain`, `threshold_geometry`, `topology_pd`, `topology_mt`; SSIM left empty). No weighted aggregate score or total ranking was computed across metrics or methods -- reading these matrices column-by-column, alongside `method_descriptive_summary.csv`'s `metric_family` field, is the intended way to see which families move together for a given effect.

## 11. Findings consistent across block lengths

Of the 56 PD/MT effect/contrast rows in `topology_factorial_and_contrast_summary.csv` with all four bootstrap CIs available, 42 have the iid, block-6, block-12, and block-24 confidence intervals agreeing on sign (all excluding zero on the same side, or all including zero) -- see that file's `*_ci95_low`/`*_ci95_high` columns directly for the per-row detail. This is a simple sign-agreement heuristic reported for convenience, not a formal test of block-length robustness.

## 12. Small, context-dependent, or sensitive findings

Effects whose mean oriented value is small relative to its bootstrap CI width, or whose sign flips between block lengths per section 11, should be described with cautious language ("slight", "small", "not clearly distinguishable from zero at these block lengths") rather than confident directional claims. In particular, near-tied contrasts such as F1 versus F2 (`f1_grad_e2` vs `f2_grad_levelset_e2`, not a direct entry in `targeted_contrast_summary.csv` but visible by comparing their respective rows against shared base methods) should only be described with stronger wording than "slight" if both the paired distribution (win/tie/loss counts) and the block-bootstrap intervals in this report actually support it.

## 13. Training-seed and temporal-dependence caveats

Every model in this benchmark was trained exactly once. Every factorial effect and every targeted contrast in this report describes a relationship AMONG THE REALIZED TRAINED MODELS on this fixed 168-sample benchmark -- it does not establish that a differently-seeded retraining of the same objective would reproduce the same effect, and it is not a universal causal claim about a loss term in general. Use language like "within this realized 2^3 factorial, enabling gradient has the largest positive mean oriented effect on PD distance," never "gradient loss universally causes better topology." Separately, because the 168 samples are consecutive hourly fields and likely temporally correlated, the ordinary bootstrap, exact sign-test, and any Wilcoxon p-values all rely on an independence-across-samples approximation and may be anti-conservative; the block-bootstrap comparison in section 11 is exploratory sensitivity evidence, not proof of a correct dependence model. Means, medians, raw/oriented deltas, and win rates remain valid exact descriptive summaries of this fixed benchmark regardless of either caveat.

## 14. Deferred analyses

Metric-correlation analysis, Pareto-front analysis, sample-level selection, and visualization generation are explicitly deferred to Phase 2C/2D and were not performed here.

## 15. Generated file list

- `ttk_runs_fixed/unified_candidate_analysis/phase2b/phase2b_validation.csv`
- `ttk_runs_fixed/unified_candidate_analysis/phase2b/b_factorial_design.csv`
- `ttk_runs_fixed/unified_candidate_analysis/phase2b/b_factorial_cell_summary.csv`
- `ttk_runs_fixed/unified_candidate_analysis/phase2b/b_factorial_sample_effects.csv`
- `ttk_runs_fixed/unified_candidate_analysis/phase2b/b_factorial_effect_summary.csv`
- `ttk_runs_fixed/unified_candidate_analysis/phase2b/b_factorial_reconstruction_check.csv`
- `ttk_runs_fixed/unified_candidate_analysis/phase2b/b_scaffold_crit_e2_design.csv`
- `ttk_runs_fixed/unified_candidate_analysis/phase2b/b_scaffold_crit_e2_sample_effects.csv`
- `ttk_runs_fixed/unified_candidate_analysis/phase2b/b_scaffold_crit_e2_effect_summary.csv`
- `ttk_runs_fixed/unified_candidate_analysis/phase2b/b_scaffold_crit_e2_reconstruction_check.csv`
- `ttk_runs_fixed/unified_candidate_analysis/phase2b/grad_scaffold_levelset_e2_design.csv`
- `ttk_runs_fixed/unified_candidate_analysis/phase2b/grad_scaffold_levelset_e2_sample_effects.csv`
- `ttk_runs_fixed/unified_candidate_analysis/phase2b/grad_scaffold_levelset_e2_effect_summary.csv`
- `ttk_runs_fixed/unified_candidate_analysis/phase2b/grad_scaffold_levelset_e2_reconstruction_check.csv`
- `ttk_runs_fixed/unified_candidate_analysis/phase2b/targeted_contrast_manifest.csv`
- `ttk_runs_fixed/unified_candidate_analysis/phase2b/targeted_contrast_per_sample.csv`
- `ttk_runs_fixed/unified_candidate_analysis/phase2b/targeted_contrast_summary.csv`
- `ttk_runs_fixed/unified_candidate_analysis/phase2b/phase2b_multiple_testing_adjusted.csv`
- `ttk_runs_fixed/unified_candidate_analysis/phase2b/b_factorial_oriented_effect_matrix.csv`
- `ttk_runs_fixed/unified_candidate_analysis/phase2b/b_scaffold_crit_e2_oriented_effect_matrix.csv`
- `ttk_runs_fixed/unified_candidate_analysis/phase2b/grad_scaffold_levelset_e2_oriented_effect_matrix.csv`
- `ttk_runs_fixed/unified_candidate_analysis/phase2b/targeted_contrast_oriented_effect_matrix.csv`
- `ttk_runs_fixed/unified_candidate_analysis/phase2b/topology_factorial_and_contrast_summary.csv`
- `ttk_runs_fixed/unified_candidate_analysis/phase2b/prior_phase_immutability_check.csv`
- `docs/unified_candidate_analysis_phase2b.md` (this file)
- `logs/unified_candidate_analysis_phase2b.log`

