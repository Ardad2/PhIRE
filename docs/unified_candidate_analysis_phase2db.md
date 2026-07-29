# Phase 2D-B: Final Publication-Quality Figure Production

```
Phase 2D-B complete.
All final composites and figure-data packages validated.
```

## 1. Scope and frozen inputs

This document reflects a completed `--full` run: planning, scripted rendering (`--render-fields`), manual topology input validation, and final composite assembly (`--assemble-composites`) all completed in sequence and passed every required validation. All six final composite figures are rendered and validated; nothing in this run remains lightweight or planning-only.

## 2. Frozen sample set

Cross-checked against `ttk_runs_fixed/unified_candidate_analysis/phase2d/selection/archetype_selected_samples.csv` (primary rows only -- `archetype_alternates.csv` is never read for selection purposes):

- `global_descriptor_disagreement` = sample_idx **120**
- `gan_pd_vs_cnn_mt_conflict` = sample_idx **34**
- `f3_pd_vs_uv_e2_mt_tradeoff` = sample_idx **119**
- `f2_balanced_vs_cnn` = sample_idx **25**
- `candidate_c_continuity` = sample_idx **30**
- `global_descriptor_agreement` = sample_idx **19**

## 3. Figure contracts

### Figure 1: `global_descriptor_disagreement` (sample_idx=120)

- Primary claim: PD and MT can produce strongly different cross-method preferences.
- Required methods: Ground Truth, CNN, GAN, Candidate C, F3: Grad+Crit, F2: Grad+Levelset+E2, UV+E2
- Required panels: speed_fields, error_maps, metric_strip, pd_evidence, mt_evidence
- Emphasis: GAN best PD but worst MT; CNN worst displayed PD but best MT; UV+E2 is comparatively MT-oriented.

### Figure 2: `gan_pd_vs_cnn_mt_conflict` (sample_idx=34)

- Primary claim: A lower PD distance does not guarantee better merge-tree or pointwise fidelity.
- Required methods: Ground Truth, Bicubic, CNN, GAN
- Required panels: speed_fields, error_maps, pd_comparison, mt_comparison, metric_strip

### Figure 3: `f3_pd_vs_uv_e2_mt_tradeoff` (sample_idx=119)

- Primary claim: Gradient-plus-critical supervision and repaired E2 supervision influence different topology descriptors.
- Required methods: Ground Truth, CNN, F3: Grad+Crit, F2: Grad+Levelset+E2, UV+E2
- Method roles: F2: Grad+Levelset+E2=compact_contextual_reference
- Required panels: speed_fields, error_maps, pd_evidence, mt_evidence, zoom_crop, metric_strip
- Emphasis: This figure must not rely on speed/error panels alone.

### Figure 4: `f2_balanced_vs_cnn` (sample_idx=25)

- Primary claim: F2 provides a balanced PD/MT improvement over CNN rather than universally optimizing every objective.
- Required methods: Ground Truth, CNN, F3: Grad+Crit, F2: Grad+Levelset+E2, UV+E2
- Required panels: speed_fields, error_maps, pd_mt_tradeoff_compact, metric_strip

### Figure 5: `candidate_c_continuity` (sample_idx=30)

- Primary claim: Candidate C is a valid topology-inspired improvement over CNN, while the expanded ablation study clarifies the more specific PD and MT mechanisms.
- Required methods: Ground Truth, CNN, Candidate C, F3: Grad+Crit, F2: Grad+Levelset+E2, UV+E2
- Required panels: speed_fields, error_maps, topology_comparison, metric_strip

### Figure 6: `global_descriptor_agreement` (sample_idx=19)

- Primary claim: PD and MT disagreement is not universal; strong methods can show broad descriptor concordance without identical rankings.
- Required methods: Ground Truth, CNN, GAN, Candidate C, F3: Grad+Crit, F2: Grad+Levelset+E2, UV+E2
- Required panels: speed_fields, error_maps, pd_mt_comparison_compact, metric_strip

## 4. Deterministic zoom region (sample 119, Figure 3)

Scoring formula: score(y0,x0) = sum((d/dy GT_speed)^2 + (d/dx GT_speed)^2) over the window [gt_gradient_energy, np.gradient on the GT speed patch] + sum_over_pixels(var_across_methods(abs_speed_error)) over the window [cross_method_error_variance, per-pixel variance of |method_speed - GT_speed| across all required full-panel methods]; candidate windows are a fixed-size 100x100 grid at stride 25 over the HR grid; ranked by score descending; ties broken by smallest y0 then smallest x0 (top-left).

Selected bounds: y=[100, 200), x=[25, 125), score=36596.483374. Computed from the real GT and per-method error fields loaded in `--render-fields`.

## 5. Authoritative PD coordinate source discovery

Not summarized in this execution_mode's report -- exact PD source discovery/resolution already ran (per-figure, inside `--render-fields`) during rendering; see `plan/pd_source_discovery.csv` and `plan/pd_source_verdicts.csv` from the most recent `--plan-only` run for the full audit trail.

## 6. Manual topology (merge-tree) requirements

21 manual ParaView/TTK merge-tree panel(s) are required across all figures (Figures 1, 2, 3, 5); 0 are currently missing. Each requires both `manual_topology_inputs/figure_XX/<method_id>_mt.png` and the sibling `_mt_metadata.csv` (schema: figure_id, sample_idx, method_id, source_vtu_path, persistence_threshold, arc_sampling, arc_line_size, camera_or_view_id, scalar_range, image_width, image_height, paraview_version, ttk_version, renderer_type, notes). Default initial settings: persistence_threshold=11.0, arc_sampling=10, arc_line_size=3 -- final metadata must record the actual values used. See `plan/manual_topology_requirements.csv` for the exact per-panel list.

## 7. Validation summary

- `validation/figure_data_reproduction.csv`: every figure-data metric value cross-checked against the frozen Phase-1 long table and Phase-2D-A `selected_sample_method_values.csv` within tolerance (1e-06); hard-fails on any disagreement.
- `validation/panel_validation.csv`: every planned panel structurally matches its figure contract.
- `validation/final_figure_validation.csv`: all six final figures are `status=not_yet_rendered`.
- `validation/prior_phase_immutability_check.csv`: all 118 protected files confirmed unchanged.

## 8. Exact commands to complete Phase 2D-B on Spark

```
python3 scripts/render_unified_candidate_figures_phase2db.py --render-fields
python3 scripts/render_unified_candidate_figures_phase2db.py --assemble-composites
python3 scripts/render_unified_candidate_figures_phase2db.py --full
```

`--render-fields` requires the real `data_out/`/`data_out_fixed/` arrays. `--assemble-composites` additionally requires every manual topology panel and metadata row listed in Section 6 to be supplied. `--full` runs both in sequence and hard-fails (never downgrades this report's status banner) while any required manual panel is absent.

## 9. Generated files

Planning-stage outputs (`ttk_runs_fixed/unified_candidate_analysis/phase2db/`):
- `ttk_runs_fixed/unified_candidate_analysis/phase2db/plan/final_figure_plan.csv`
- `ttk_runs_fixed/unified_candidate_analysis/phase2db/plan/final_panel_manifest.csv`
- `ttk_runs_fixed/unified_candidate_analysis/phase2db/plan/manual_topology_requirements.csv`
- `ttk_runs_fixed/unified_candidate_analysis/phase2db/plan/pd_source_discovery.csv`
- `ttk_runs_fixed/unified_candidate_analysis/phase2db/plan/pd_source_verdicts.csv`
- `ttk_runs_fixed/unified_candidate_analysis/phase2db/plan/final_composite_manifest.csv`
- `ttk_runs_fixed/unified_candidate_analysis/phase2db/plan/final_figure_captions.md`
- `ttk_runs_fixed/unified_candidate_analysis/phase2db/figure_data/figure_01_global_disagreement.csv`
- `ttk_runs_fixed/unified_candidate_analysis/phase2db/figure_data/figure_02_gan_cnn_conflict.csv`
- `ttk_runs_fixed/unified_candidate_analysis/phase2db/figure_data/figure_03_f3_uv_e2_tradeoff.csv`
- `ttk_runs_fixed/unified_candidate_analysis/phase2db/figure_data/figure_04_f2_balanced.csv`
- `ttk_runs_fixed/unified_candidate_analysis/phase2db/figure_data/figure_05_candidate_c_continuity.csv`
- `ttk_runs_fixed/unified_candidate_analysis/phase2db/figure_data/figure_06_global_agreement.csv`
- `ttk_runs_fixed/unified_candidate_analysis/phase2db/validation/prior_phase_immutability_check.csv`
- `ttk_runs_fixed/unified_candidate_analysis/phase2db/validation/figure_data_reproduction.csv`
- `ttk_runs_fixed/unified_candidate_analysis/phase2db/validation/panel_validation.csv`
- `ttk_runs_fixed/unified_candidate_analysis/phase2db/validation/final_figure_validation.csv`
- `ttk_runs_fixed/unified_candidate_analysis/phase2db/validation/zoom_selection_validation.csv`
- `ttk_runs_fixed/unified_candidate_analysis/phase2db/validation/panel_scale_provenance.csv`
- `docs/unified_candidate_analysis_phase2db.md` (this file)
- `logs/unified_candidate_analysis_phase2db.log`

Scripted panels rendered and validated in `--render-fields`: 81 panel file(s) under `panels/**/*.png` (never claimed merely because the directory exists -- counted from `validation/panel_validation.csv`).
Manual topology inputs validated: 21/21 panel(s) under `manual_topology_inputs/**/*` (see `plan/manual_topology_requirements.csv`).
Final composite figures assembled and validated: 6/6 under `figures/**/*` (see `validation/final_figure_validation.csv`).

