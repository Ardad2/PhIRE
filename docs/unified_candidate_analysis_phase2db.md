# Phase 2D-B: Final Publication-Quality Figure Production

```
Phase 2D-B planning complete (authoritative Spark checkout).
Raw arrays were intentionally not loaded in --plan-only; run --render-fields to continue.
```

## 1. Scope and frozen inputs

This document reflects a `--plan-only` run on the authoritative Spark machine. It reads exclusively frozen Phase-1 through Phase-2D-A artifacts (118 files, checksummed before and after this stage) and never touches `data_out/`, `data_out_fixed/`, or reruns any training/inference/TTK step. The raw Spark arrays may exist on this machine, but `--plan-only` intentionally does not load or render them in this mode -- exact PD VTU sources are still audited directly on disk (Section 5), and no method-level PD verdict may remain pending after this run. Phase 2D-A is treated as complete and authoritative; no sample is re-selected and no alternate is activated here.

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

**Not yet computed.** The raw GT and per-method error fields required by the zoom-window score exist on this authoritative Spark machine, but `--plan-only` intentionally does not load or render them; `select_deterministic_zoom()` will run in `--render-fields`.

## 5. Authoritative PD coordinate source discovery

An exact, repository-relative filesystem search (`plan/pd_source_discovery.csv`, 26 row(s); reduced per-`(figure, sample, method)` verdicts in `plan/pd_source_verdicts.csv`, 15 row(s)) was performed for every (figure, method) requiring a `pd_evidence`/`pd_comparison` panel -- GT, CNN, GAN, and Bicubic included, none assumed found without a concrete `selected_candidate_path`. This process is running in the authoritative Spark machine: 9 method(s) available_validated, 0 pending_authoritative_spark_source_discovery, 6 unavailable_after_authoritative_spark_audit. On the authoritative Spark machine, no method-level verdict is ever left pending (enforced in code). Only the exact TTK PD VTU convention is matched: `<artifact_alias>_topology/pd/<GT|SR>/<artifact_alias>_<GT|SR>_s<sample_idx>_..._pd_port_0.vtu`, with `mt/` paths, `_mt_port_*.vtu`, and `_pd_port_1.vtu` always excluded, and GT cross-checked for coordinate agreement across every candidate topology tree before being accepted as canonical. Method-to-artifact aliases: {'candidate_c': 'candidateC_expanded2688', 'f3_grad_crit': 'candidateF_grad_crit_expanded2688', 'f2_grad_levelset_e2': 'candidateF_grad_levelset_E2_low_expanded2688', 'uv_e2': 'candidateUV_plus_E2_tf_lowlambda_expanded2688'}; CNN/GAN/bicubic use their own method_id (no `<alias>_topology` directory exists for them anywhere in this repository) and are searched under {'cnn': ['ttk_runs_fixed/cnn'], 'gan': ['ttk_runs_fixed/gan'], 'bicubic': ['ttk_runs_fixed/bicubic']}. Existing PD-overlay-related scripts already in this repository were also inventoried for provenance.

**Filtration convention.** All main-figure PD coordinate panels use the `default_sublevel` filtration convention exclusively -- TTK's `ttkPersistenceDiagramCmd`/`ttkMergeTreeCmd` default (sublevel-set) behavior on the raw, non-negated speed field, which is what the main reported PD distances and the paper's primary TTK evaluation use. A separate `superlevel_negated_speed` robustness evaluation (`scripts/run_superlevel_topology_robustness.py`) computes the SAME sublevel TTK call on the NEGATED speed field -- mathematically the superlevel-set topology of the true speed field -- and is written to a disjoint root, `ttk_runs_fixed/superlevel_topology/`. cnn/gan sources under that root were audited for real (searched under {'cnn': ['ttk_runs_fixed/superlevel_topology/cnn/topology'], 'gan': ['ttk_runs_fixed/superlevel_topology/gan/topology']}) and are recorded in `plan/pd_source_discovery.csv` as excluded provenance (11 row(s), `eligible_for_main_figure=false`, `exclusion_reason=filtration_convention_mismatch`) -- they are never mixed into a default_sublevel coordinate panel and never compared against the default_sublevel GT family. Default_sublevel and superlevel_negated_speed diagrams are NOT expected to share raw coordinates (different filtration, generally different pair counts and coordinate ranges); this is a source-availability decision, not a change to any frozen metric. Methods with no exact default_sublevel coordinate artifact use the explicit scalar PD-evidence fallback (clearly labeled, never presented as a real persistence diagram) instead of a superlevel substitute.

## 6. Manual topology (merge-tree) requirements

21 manual ParaView/TTK merge-tree panel(s) are required across all figures (Figures 1, 2, 3, 5); 21 are currently missing. Each requires both `manual_topology_inputs/figure_XX/<method_id>_mt.png` and the sibling `_mt_metadata.csv` (schema: figure_id, sample_idx, method_id, source_vtu_path, persistence_threshold, arc_sampling, arc_line_size, camera_or_view_id, scalar_range, image_width, image_height, paraview_version, ttk_version, renderer_type, notes). Default initial settings: persistence_threshold=11.0, arc_sampling=10, arc_line_size=3 -- final metadata must record the actual values used. See `plan/manual_topology_requirements.csv` for the exact per-panel list.

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

Not yet generated (pending `--render-fields`, manual topology export, and `--assemble-composites`): `panels/**/*.png`, `manual_topology_inputs/**/*`, `figures/**/*`.

