# Phase 2D-A: Deterministic Archetype Selection and Raw-Artifact Preview Plan

```
Phase 2D-A selection stage complete.
Raw-array audit and preview rendering pending authoritative Spark run.
```

## 1. Scope and frozen inputs

This document reflects a `--selection-only` run in a lightweight checkout. It reads exclusively the 86 frozen Phase-1/2A/2B/2C CSV and Markdown artifacts (checksummed before and after this stage) and the plain-text raw topology CSVs (`*_pd_mt_distances.csv`, `phase_c_results.csv`) that are ordinary git-tracked files, not gitignored raw arrays. It never touches `data_out/` or `data_out_fixed/`.

## 2. Why selection is algorithmic rather than manual

Every archetype is defined by an explicit eligibility rule and a closed-form score computed from the frozen Phase-1 long table and Phase-2C relationship tables. No sample was chosen by looking at an image: raw wind-field arrays are not available to (and are never read by) the selection stage, so the chosen samples cannot reflect appearance-based cherry-picking even in principle. The same script run twice on the same frozen inputs produces byte-identical output.

## 3. Robust-z scoring convention

`robust_z(x) = (x - median(x)) / (1.4826 * MAD(x))`, computed over the eligible-sample population for each archetype (the full 168 samples for the two global archetypes, the eligible subset otherwise). If MAD is zero the score falls back to `(x - mean(x)) / std(x)`; if both MAD and std are zero the component contributes exactly zero to every sample's score (see `archetype_selection_diagnostics.csv`, `event_type=robust_z_fallback`, for any such occurrence in this run). Scores are used ONLY to select samples deterministically -- they are never a method ranking.

## 4. Archetype definitions

### 1. `global_descriptor_disagreement` -- PD/MT descriptor disagreement across methods

- Primary methods: all 18 topology-bearing methods (cross-method preference agreement)
- Metrics used: pd_distance, mt_distance (via Phase-2C topology_sample_preference_agreement.csv and samplewise_cross_method_correlations.csv)
- Score formula: `mean(robust_z(-agreement_rate), robust_z(-oriented_spearman), robust_z(-oriented_pearson))`
- Eligible samples: 168 / 168
- Tie-break: score desc, then sample_idx asc

### 2. `gan_pd_vs_cnn_mt_conflict` -- The GAN-versus-CNN topology tradeoff

- Primary methods: gan, cnn
- Metrics used: pd_distance, mt_distance
- Score formula: `mean(robust_z(gan_pd_improvement_vs_cnn), robust_z(cnn_mt_improvement_vs_gan))`
- Eligible samples: 146 / 168
- Tie-break: score desc, then sample_idx asc

### 3. `f3_pd_vs_uv_e2_mt_tradeoff` -- The F3-versus-UV+E2 PD/MT tradeoff

- Primary methods: f3_grad_crit, uv_e2
- Metrics used: pd_distance, mt_distance
- Score formula: `mean(robust_z(f3_pd_improvement_vs_uv_e2), robust_z(uv_e2_mt_improvement_vs_f3))`
- Eligible samples: 141 / 168
- Tie-break: score desc, then sample_idx asc

### 4. `f2_balanced_vs_cnn` -- A balanced F2 improvement over CNN

- Primary methods: f2_grad_levelset_e2, cnn
- Metrics used: pd_distance, mt_distance, psnruv, speed_mae
- Score formula: `min(robust_z(pd_improvement), robust_z(mt_improvement)) + 0.25*mean(robust_z(psnr_improvement), robust_z(speed_mae_improvement))`
- Eligible samples: 100 / 168
- Tie-break: score desc, then sample_idx asc

### 5. `candidate_c_continuity` -- Candidate C as continuity with the submitted topology-inspired model

- Primary methods: candidate_c, cnn
- Metrics used: pd_distance, mt_distance, psnruv, speed_mae
- Score formula: `0.50*robust_z(pd_improvement) + 0.20*robust_z(mt_improvement) + 0.15*robust_z(psnr_improvement) + 0.15*robust_z(speed_mae_improvement)`
- Eligible samples: 168 / 168
- Tie-break: score desc, then sample_idx asc

### 6. `global_descriptor_agreement` -- PD/MT descriptor agreement across methods

- Primary methods: all 18 topology-bearing methods (cross-method preference agreement)
- Metrics used: pd_distance, mt_distance (via Phase-2C topology_sample_preference_agreement.csv and samplewise_cross_method_correlations.csv)
- Score formula: `mean(robust_z(agreement_rate), robust_z(oriented_spearman), robust_z(oriented_pearson))`
- Eligible samples: 168 / 168
- Tie-break: score desc, then sample_idx asc

## 5. Ranked top-10 candidates per archetype

See `selection/archetype_score_table.csv` for the full table (rank, sample_idx, score, and every named score component with its raw value, robust-z value, and the robust-z fallback method used).

## 6. Duplicate-resolution decisions

- `f3_pd_vs_uv_e2_mt_tradeoff`: sample_idx=120 (rank 4 before de-duplication) was skipped -- already selected by a higher-priority archetype.
- `candidate_c_continuity`: sample_idx=25 (rank 1 before de-duplication) was skipped -- already selected by a higher-priority archetype.

## 7. Selected sample IDs and alternates

- **global_descriptor_disagreement**: selected sample_idx=**120** (score=1.8691); alternates: 41 (score=1.8378), 143 (score=1.8273), 26 (score=1.8022)
- **gan_pd_vs_cnn_mt_conflict**: selected sample_idx=**34** (score=3.4279); alternates: 33 (score=3.3429), 32 (score=2.9393), 27 (score=2.8922)
- **f3_pd_vs_uv_e2_mt_tradeoff**: selected sample_idx=**119** (score=2.2605); alternates: 71 (score=1.7286), 114 (score=1.6787), 115 (score=1.4117)
- **f2_balanced_vs_cnn**: selected sample_idx=**25** (score=1.9198); alternates: 135 (score=1.5959), 24 (score=1.4944), 22 (score=1.1224)
- **candidate_c_continuity**: selected sample_idx=**30** (score=1.4228); alternates: 134 (score=1.2916), 107 (score=1.0878), 155 (score=1.0315)
- **global_descriptor_agreement**: selected sample_idx=**19** (score=2.7107); alternates: 20 (score=2.4124), 45 (score=2.2208), 79 (score=2.1532)

## 8. Raw-array and topology-artifact validation

Topology CSVs (plain text, git-tracked) were read directly and cross-checked: the independently recomputed PD/MT method means from the raw `*_pd_mt_distances.csv` / `phase_c_results.csv` sources match both the Phase-1 long table and Phase-1 `unified_primary_topology_validation.csv` within tolerance for every topology-bearing `full_selected_story` method (see `selection_validation.csv`).

Raw `.npy` array auditing (idx/dataIN/dataGT/dataSR existence, shape, alignment, finiteness) was **not** performed in this run, because `data_out/` and `data_out_fixed/` are gitignored and absent from this lightweight checkout by design. `selection/raw_artifact_requirements.csv` enumerates the exact repository-relative paths, expected shapes, and resolution convention required to complete this audit on a machine where those directories are present.

## 9. Preview inventory

**No preview PNGs were generated in this run.** `selection/preview_plan.csv` records what will be rendered (one combined speed+error review PNG per selected sample, plus one contact sheet), each currently `status=pending_raw_artifacts`.

## 10. Figure plan for Phase 2D-B

Phase 2D-A produces only review-only audit previews, never final publication figures. `selection/preview_method_manifest.csv` records the three method groups (`baseline_story`, `descriptor_tradeoff_story`, `full_selected_story`) that Phase 2D-B's final figures will draw from. Final rendering remains explicitly deferred to Phase 2D-B.

## 11. Caveat: illustrative, not a population estimate

The six selected samples are drawn from this fixed 168-sample benchmark using a fixed, designed set of 19 candidate methods. They are illustrative examples chosen to make specific, pre-registered archetypes concrete -- they are not a random sample, and no population-level or generalization claim should be inferred from any single selected field.

## 12. Exact command to complete Phase 2D-A on Spark

```
python3 scripts/select_and_preview_unified_candidates_phase2d.py --full
```

This reads the already-written `selection/archetype_selected_samples.csv` (unchanged from this run, since selection is purely CSV-driven and deterministic), performs the full raw-artifact audit against the real `data_out/`/`data_out_fixed/` arrays, and -- only if every audit check passes -- renders the review-only preview PNGs. `--render-previews` alone performs the same audit-and-render step without re-running selection, provided `selection/archetype_selected_samples.csv` already exists.

## 13. Generated files

Selection-stage outputs (`ttk_runs_fixed/unified_candidate_analysis/phase2d/selection/`):
- `ttk_runs_fixed/unified_candidate_analysis/phase2d/selection/archetype_score_table.csv`
- `ttk_runs_fixed/unified_candidate_analysis/phase2d/selection/archetype_selected_samples.csv`
- `ttk_runs_fixed/unified_candidate_analysis/phase2d/selection/archetype_alternates.csv`
- `ttk_runs_fixed/unified_candidate_analysis/phase2d/selection/archetype_selection_diagnostics.csv`
- `ttk_runs_fixed/unified_candidate_analysis/phase2d/selection/selected_sample_metric_context.csv`
- `ttk_runs_fixed/unified_candidate_analysis/phase2d/selection/selected_sample_method_values.csv`
- `ttk_runs_fixed/unified_candidate_analysis/phase2d/selection/selected_sample_pairwise_preferences.csv`
- `ttk_runs_fixed/unified_candidate_analysis/phase2d/selection/preview_method_manifest.csv`
- `ttk_runs_fixed/unified_candidate_analysis/phase2d/selection/preview_plan.csv`
- `ttk_runs_fixed/unified_candidate_analysis/phase2d/selection/raw_artifact_requirements.csv`
- `ttk_runs_fixed/unified_candidate_analysis/phase2d/selection/selection_validation.csv`
- `ttk_runs_fixed/unified_candidate_analysis/phase2d/selection/prior_phase_immutability_check.csv`
- `docs/unified_candidate_analysis_phase2d.md` (this file)
- `logs/unified_candidate_analysis_phase2d_selection.log`

Not yet generated (pending Spark): `preview_audit/*.csv`, `previews/**/*.png`.

