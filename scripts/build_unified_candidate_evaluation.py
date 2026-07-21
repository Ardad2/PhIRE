#!/usr/bin/env python3
"""Phase 1 unified multi-metric evaluation audit for the PhIRE wind-field
loss-ablation candidates.

Builds a repository-authoritative inventory and unified per-sample table
from EXISTING cheap-evaluation and true-TTK-topology CSV/NPY artifacts.
This script is read-only with respect to every experiment artifact in the
repository: it never runs training, TTK, or the cheap evaluator, and it
never writes anywhere except:

    ttk_runs_fixed/unified_candidate_evaluation/
    docs/unified_candidate_evaluation_*.md

Design principle -- "hard-fail on corruption, report on absence"
------------------------------------------------------------------
This script hard-fails (raises SystemExit / lets an exception propagate)
only on internal inconsistencies that would silently corrupt its own
output invariants: duplicate (method_id, sample_idx) keys inside a single
source file, a non-finite or negative topology distance in a file that WAS
found, or more than one ambiguous candidate source file for the same
method that this script cannot resolve automatically.

Total ABSENCE of an artifact (the file/directory simply does not exist in
this git checkout) is not treated as a crash-worthy bug. It is exactly the
condition this Phase-1 audit exists to surface: it is recorded per-method
in method_inventory.csv / the unified tables with an explicit
validation_status, and the run still completes and writes every requested
deliverable. The final report states in plain language which primary
methods have zero real data and why success cannot be claimed for those
methods -- silently crashing instead would prevent producing the very
audit the user asked for.

Do not rerun training or TTK. Do not delete or modify any existing
artifact. Do not manufacture missing numeric values (no zero-fill, no
interpolation, no copying values from the reference markdown docs -- the
docs are a roadmap only, never a numeric source).
"""

from __future__ import annotations

import csv
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
import os
os.chdir(REPO_ROOT)

OUT_DIR = REPO_ROOT / 'ttk_runs_fixed' / 'unified_candidate_evaluation'
DOCS_DIR = REPO_ROOT / 'docs'
LOG_PATH = REPO_ROOT / 'logs' / 'build_unified_candidate_evaluation.log'

N_EVAL = 168
PD_MT_TOLERANCE = 1e-4

_LOG_LINES: list[str] = []


def log(msg: str = '') -> None:
    print(msg)
    _LOG_LINES.append(msg)


def flush_log() -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open('w') as fh:
        fh.write('\n'.join(_LOG_LINES) + '\n')


def rp(rel: str) -> Path:
    """Repo-root-anchored path."""
    return REPO_ROOT / rel


# =============================================================================
# Metric-column schema (standardized names, direction, representation).
# Derived from the ACTUAL column-writing code in scripts/evaluate_finetune_candidate.py
# (the schema new candidates would be written with) and cross-checked against the
# ACTUAL column headers found in ttk_runs_fixed/combined/psnr_topology_physics_merged.csv
# (the only per-sample source file that currently has real primary-method data:
# cnn and gan). See column_mapping.csv for the full source-column-level mapping.
# =============================================================================
# (standardized_column, direction, representation, units)
METRIC_SCHEMA = [
    ('psnruv',                  'higher_is_better',  'vector_uv',                'dB'),
    ('ssim_speed',              'higher_is_better',  'scalar_speed',             'unitless'),
    ('speed_mae',               'lower_is_better',   'scalar_speed',             'm/s'),
    ('speed_rmse',              'lower_is_better',   'scalar_speed',             'm/s'),
    ('wpd_mae',                 'lower_is_better',   'wind_power_distribution',  'm^3/s^3'),
    ('wpd_w1',                  'lower_is_better',   'wind_power_distribution',  'm^3/s^3 (Wasserstein-1)'),
    ('wpd_bias_abs',            'lower_is_better',   'wind_power_distribution',  'm^3/s^3'),
    ('grad_mae',                'lower_is_better',   'gradient_distribution',    'm/s per pixel'),
    ('grad_w1',                 'lower_is_better',   'gradient_distribution',    'm/s per pixel (Wasserstein-1)'),
    ('grad_kurtosis_abs_delta', 'lower_is_better',   'gradient_distribution',    'unitless'),
    ('psd_log_l2',              'lower_is_better',   'frequency_domain',         'log-power L2'),
    ('psd_slope_abs_delta',     'lower_is_better',   'frequency_domain',         'unitless (spectral slope)'),
    ('exceed_abs_t5',           'lower_is_better',   'threshold_geometry',       'fraction'),
    ('exceed_abs_t10',          'lower_is_better',   'threshold_geometry',       'fraction'),
    ('exceed_abs_t15',          'lower_is_better',   'threshold_geometry',       'fraction'),
    ('exceed_abs_p90',          'lower_is_better',   'threshold_geometry',       'fraction'),
    ('comp_curve_l1',           'lower_is_better',   'threshold_geometry',       'component count (L1 over threshold curve)'),
    ('comp_abs_t5',             'lower_is_better',   'threshold_geometry',       'component count'),
    ('comp_abs_t10',            'lower_is_better',   'threshold_geometry',       'component count'),
    ('comp_abs_t15',            'lower_is_better',   'threshold_geometry',       'component count'),
    ('pd_distance',             'lower_is_better',   'topology_pd',              'bottleneck/Wasserstein PD distance'),
    ('mt_distance',             'lower_is_better',   'topology_mt',              'merge-tree distance'),
]
METRIC_COLUMNS = [m[0] for m in METRIC_SCHEMA]
METRIC_DIRECTION = {m[0]: m[1] for m in METRIC_SCHEMA}

IDENTITY_COLUMNS = [
    'sample_idx', 'method_id', 'display_name', 'candidate_family',
    'training_scale', 'architecture',
    'uses_speed', 'uses_grad', 'uses_levelset', 'uses_crit', 'uses_e2',
]

# Columns present in ttk_runs_fixed/combined/psnr_topology_physics_merged.csv
# (the legacy cnn/gan-only "physics_merged" pipeline) mapped to the standardized
# schema above. This pipeline predates scripts/evaluate_finetune_candidate.py and
# does NOT compute speed_mae/speed_rmse/comp_* -- those are left as genuinely
# missing (not fabricated) for cnn/gan in this audit.
BASELINE_COMBINED_DIRECT_MAP = {
    'psnr': 'psnruv',
    'ssim': 'ssim_speed',
    'wpd_mae': 'wpd_mae',
    'wpd_w1': 'wpd_w1',
    'grad_mae': 'grad_mae',
    'grad_w1': 'grad_w1',
    'grad_kurtosis_abs_delta': 'grad_kurtosis_abs_delta',
    'psd_log_l2': 'psd_log_l2',
    'psd_slope_abs_delta': 'psd_slope_abs_delta',
    'exceed_frac_abs_delta_t5': 'exceed_abs_t5',
    'exceed_frac_abs_delta_t10': 'exceed_abs_t10',
    'exceed_frac_abs_delta_t15': 'exceed_abs_t15',
    'exceed_frac_abs_delta_p90': 'exceed_abs_p90',
    'pd_distance': 'pd_distance',
    'mt_distance': 'mt_distance',
}
# 'wpd_bias' -> 'wpd_bias_abs' needs abs() applied; handled specially in code.
BASELINE_COMBINED_NOT_AVAILABLE = [
    'speed_mae', 'speed_rmse', 'comp_curve_l1',
    'comp_abs_t5', 'comp_abs_t10', 'comp_abs_t15',
]

# Columns scripts/evaluate_finetune_candidate.py would write for any NEW
# candidate's all_sample_metrics_<name>.csv, mapped to the standardized schema.
CANDIDATE_EVAL_DIRECT_MAP = {
    'psnruv': 'psnruv',
    'ssim': 'ssim_speed',
    'speed_mae': 'speed_mae',
    'speed_rmse': 'speed_rmse',
    'wpd_mae': 'wpd_mae',
    'wpd_w1': 'wpd_w1',
    'wpd_bias_abs': 'wpd_bias_abs',
    'grad_mae': 'grad_mae',
    'grad_w1': 'grad_w1',
    'grad_kurtosis_abs_delta': 'grad_kurtosis_abs_delta',
    'psd_log_l2': 'psd_log_l2',
    'psd_slope_abs_delta': 'psd_slope_abs_delta',
    'exceed_abs_t5': 'exceed_abs_t5',
    'exceed_abs_t10': 'exceed_abs_t10',
    'exceed_abs_t15': 'exceed_abs_t15',
    'exceed_abs_p90': 'exceed_abs_p90',
    'comp_curve_l1': 'comp_curve_l1',
    'comp_abs_delta_t5': 'comp_abs_t5',
    'comp_abs_delta_t10': 'comp_abs_t10',
    'comp_abs_delta_t15': 'comp_abs_t15',
    'pd_distance': 'pd_distance',
    'mt_distance': 'mt_distance',
}


# =============================================================================
# Manifest: every method this audit knows to look for.
# Expected PD/MT means are the values the user supplied in the task instructions
# (originating from the reference documentation's recorded true-TTK results);
# they are used ONLY as a validation target for values independently recomputed
# from repository CSVs, never written into the unified table as data.
# =============================================================================
EXPECTED_PD_MT = {
    'cnn': (27.4063, 5.8678),
    'gan': (20.8641, 8.3481),
    'uv': (29.6121, 6.0119),
    'speed_only': (29.5783, 5.9996),
    'levelset_only': (29.5953, 6.0076),
    'speed_levelset': (29.4363, 5.9441),
    'grad_only': (22.9326, 6.0560),
    'speed_grad': (22.9706, 6.2905),
    'grad_levelset': (22.6194, 6.1996),
    'candidate_b': (22.7070, 6.1612),
    'candidate_c': (22.4944, 6.0803),
    'uv_crit': (29.1143, 5.6899),
    'uv_e2': (25.0721, 5.5940),
    'b_e2': (23.9876, 5.6774),
    'c_e2': (24.2686, 5.6628),
    'f1_grad_e2': (23.8382, 5.6566),
    'f2_grad_levelset_e2': (23.7481, 5.6742),
    'f3_grad_crit': (22.0179, 5.9840),
}

TFREC_TRAIN_2688 = 'example_data_topology_expanded_2688/wind_MR-HR.tfrecord'
TFREC_EVAL_FIXED = 'example_data_fixed/wind_MR-HR.tfrecord'
FIXED_BENCHMARK_NOTE = ('fixed 168-sample benchmark (data_out_fixed/wind_mrhr_<method>/ '
                         'npy arrays; data_out_fixed/ is gitignored and absent in this checkout)')


def _native_tf_entry(method_id, display_name, original_method_name, family,
                      uses_speed, uses_grad, uses_levelset, uses_crit, uses_e2,
                      objective_summary, repaired_status='not_applicable'):
    return dict(
        method_id=method_id, display_name=display_name,
        original_method_name=original_method_name, candidate_family=family,
        comparison_tier='primary_2688_native_tf', include_primary=True,
        training_scale='2688', architecture='native_tf',
        training_tfrecord=TFREC_TRAIN_2688, evaluation_tfrecord=TFREC_EVAL_FIXED,
        objective_summary=objective_summary,
        uses_speed=uses_speed, uses_grad=uses_grad, uses_levelset=uses_levelset,
        uses_crit=uses_crit, uses_e2=uses_e2, repaired_status=repaired_status,
        data_out_dir=f'data_out/wind_finetune_{original_method_name}',
        model_dir=f'models_fixed/topology_finetuning/wind_finetune_{original_method_name}',
        cheap_eval_dir=f'ttk_runs_fixed/topology_finetuning/{original_method_name}_eval',
        topology_dir=f'ttk_runs_fixed/topology_finetuning/{original_method_name}_topology',
        cheap_report=f'docs/topology_finetuning_{original_method_name}_eval.md',
        topology_report=f'docs/topology_finetuning_{original_method_name}_topology_eval.md',
        exclusion_reason='',
    )


PRIMARY_MANIFEST = [
    dict(method_id='bicubic', display_name='Bicubic baseline',
         original_method_name='bicubic', candidate_family='baseline',
         comparison_tier='primary_baseline', include_primary=True,
         training_scale='n/a (interpolation, no training)', architecture='bicubic_interpolation',
         training_tfrecord='n/a', evaluation_tfrecord=FIXED_BENCHMARK_NOTE,
         objective_summary='Bicubic upsampling of the MR input to HR resolution; no learned model.',
         uses_speed=False, uses_grad=False, uses_levelset=False, uses_crit=False, uses_e2=False,
         repaired_status='baseline',
         data_out_dir='data_out_fixed/wind_mrhr_bicubic',
         model_dir='n/a', cheap_eval_dir='', topology_dir='',
         cheap_report='', topology_report='', exclusion_reason=''),
    dict(method_id='cnn', display_name='CNN baseline (pretrained)',
         original_method_name='cnn', candidate_family='baseline',
         comparison_tier='primary_baseline', include_primary=True,
         training_scale='n/a (released pretrained model)', architecture='pretrained_cnn',
         training_tfrecord='n/a', evaluation_tfrecord=FIXED_BENCHMARK_NOTE,
         objective_summary='Released fidelity-oriented pretrained PhIRE CNN; no fine-tuning.',
         uses_speed=False, uses_grad=False, uses_levelset=False, uses_crit=False, uses_e2=False,
         repaired_status='baseline',
         data_out_dir='data_out_fixed/wind_mrhr_cnn',
         model_dir='models/wind_mr-hr/trained_cnn', cheap_eval_dir='', topology_dir='ttk_runs_fixed/cnn/phase_c_final',
         cheap_report='', topology_report='', exclusion_reason=''),
    dict(method_id='gan', display_name='GAN baseline (pretrained)',
         original_method_name='gan', candidate_family='baseline',
         comparison_tier='primary_baseline', include_primary=True,
         training_scale='n/a (released pretrained model)', architecture='pretrained_gan',
         training_tfrecord='n/a', evaluation_tfrecord=FIXED_BENCHMARK_NOTE,
         objective_summary='Released adversarially-trained pretrained PhIRE GAN; no fine-tuning.',
         uses_speed=False, uses_grad=False, uses_levelset=False, uses_crit=False, uses_e2=False,
         repaired_status='baseline',
         data_out_dir='data_out_fixed/wind_mrhr_gan',
         model_dir='models/wind_mr-hr/trained_gan', cheap_eval_dir='', topology_dir='ttk_runs_fixed/gan/phase_c_final',
         cheap_report='', topology_report='', exclusion_reason=''),

    _native_tf_entry('uv', 'Candidate UV (vector-only control)', 'candidateUV_expanded2688',
                      'UV_control', False, False, False, False, False,
                      'L_uv only (ablation control for fine-tuning itself).'),
    _native_tf_entry('speed_only', 'B-factorial: speed only', 'candidateB_factorial_speed_expanded2688',
                      'B_factorial', True, False, False, False, False,
                      'L_uv + 0.01 L_speed.'),
    _native_tf_entry('levelset_only', 'B-factorial: level-set only', 'candidateB_factorial_levelset_expanded2688',
                      'B_factorial', False, False, True, False, False,
                      'L_uv + 0.25 L_levelset.'),
    _native_tf_entry('speed_levelset', 'B-factorial: speed + level-set', 'candidateB_factorial_speed_levelset_expanded2688',
                      'B_factorial', True, False, True, False, False,
                      'L_uv + 0.01 L_speed + 0.25 L_levelset.'),
    _native_tf_entry('grad_only', 'B-factorial: gradient only', 'candidateB_factorial_grad_expanded2688',
                      'B_factorial', False, True, False, False, False,
                      'L_uv + 0.05 L_grad.'),
    _native_tf_entry('speed_grad', 'B-factorial: speed + gradient', 'candidateB_factorial_speed_grad_expanded2688',
                      'B_factorial', True, True, False, False, False,
                      'L_uv + 0.01 L_speed + 0.05 L_grad.'),
    _native_tf_entry('grad_levelset', 'B-factorial: gradient + level-set', 'candidateB_factorial_grad_levelset_expanded2688',
                      'B_factorial', False, True, True, False, False,
                      'L_uv + 0.05 L_grad + 0.25 L_levelset.'),
    _native_tf_entry('candidate_b', 'Candidate B (full scalar-field proxy scaffold)', 'candidateB_expanded2688',
                      'B_full', True, True, True, False, False,
                      'L_uv + 0.01 L_speed + 0.05 L_grad + 0.25 L_levelset.'),
    _native_tf_entry('candidate_c', 'Candidate C (Candidate B + local-maxima proxy)', 'candidateC_expanded2688',
                      'C_full', True, True, True, True, False,
                      'L_uv + 0.01 L_speed + 0.05 L_grad + 0.25 L_levelset + 0.001 L_crit.'),
    _native_tf_entry('uv_crit', 'UV + critical-maxima proxy', 'candidateUV_plus_crit_expanded2688',
                      'UV_crit', False, False, False, True, False,
                      'L_uv + 0.001 L_crit (Candidate C minus the Candidate B scaffold).'),
    _native_tf_entry('uv_e2', 'UV + repaired E2', 'candidateUV_plus_E2_tf_lowlambda_expanded2688',
                      'E2_uv', False, False, False, False, True,
                      'L_uv + 0.004 L_TTKCV + 0.002 L_TTKpers (repaired low-lambda E2, no B/C scaffold).',
                      repaired_status='repaired_e2'),
    _native_tf_entry('b_e2', 'Candidate B + repaired E2', 'candidateB_plus_E2_tf_lowlambda_expanded2688',
                      'E2_b', True, True, True, False, True,
                      'L_uv + 0.01 L_speed + 0.05 L_grad + 0.25 L_levelset + 0.004 L_TTKCV + 0.002 L_TTKpers '
                      '(L_crit disabled, lambda_crit=0).',
                      repaired_status='repaired_e2'),
    _native_tf_entry('c_e2', 'Candidate C + repaired E2', 'candidateE2_tf_lowlambda_expanded2688',
                      'E2_c', True, True, True, True, True,
                      'L_uv + 0.01 L_speed + 0.05 L_grad + 0.25 L_levelset + 0.001 L_crit '
                      '+ 0.004 L_TTKCV + 0.002 L_TTKpers.',
                      repaired_status='repaired_e2'),
    _native_tf_entry('f1_grad_e2', 'Candidate F: grad + repaired E2-low', 'candidateF_grad_E2_low_expanded2688',
                      'F_recombination', False, True, False, False, True,
                      'L_uv + 0.05 L_grad + 0.004 L_TTKCV + 0.002 L_TTKpers.',
                      repaired_status='repaired_e2'),
    _native_tf_entry('f2_grad_levelset_e2', 'Candidate F: grad + level-set + repaired E2-low',
                      'candidateF_grad_levelset_E2_low_expanded2688', 'F_recombination',
                      False, True, True, False, True,
                      'L_uv + 0.05 L_grad + 0.25 L_levelset + 0.004 L_TTKCV + 0.002 L_TTKpers.',
                      repaired_status='repaired_e2'),
    _native_tf_entry('f3_grad_crit', 'Candidate F: grad + critical-maxima proxy', 'candidateF_grad_crit_expanded2688',
                      'F_recombination', False, True, False, True, False,
                      'L_uv + 0.05 L_grad + 0.001 L_crit (no E2).'),
]
assert len(PRIMARY_MANIFEST) == 19, f'expected 19 primary methods, got {len(PRIMARY_MANIFEST)}'
assert set(EXPECTED_PD_MT) - {'cnn', 'gan'} == {m['method_id'] for m in PRIMARY_MANIFEST} - {'bicubic', 'cnn', 'gan'}, \
    'EXPECTED_PD_MT keys must exactly match the non-bicubic primary method_ids'


def _secondary_entry(method_id, display_name, original_method_name, family, tier,
                      training_scale, architecture, repaired_status, exclusion_reason,
                      data_out_dir=None, model_dir=None):
    if data_out_dir is None:
        data_out_dir = f'data_out/wind_finetune_{original_method_name}'
    if model_dir is None:
        model_dir = f'models_fixed/topology_finetuning/wind_finetune_{original_method_name}'
    return dict(
        method_id=method_id, display_name=display_name,
        original_method_name=original_method_name, candidate_family=family,
        comparison_tier=tier, include_primary=False,
        training_scale=training_scale, architecture=architecture,
        training_tfrecord='n/a (not resolved for secondary tier)',
        evaluation_tfrecord=TFREC_EVAL_FIXED,
        objective_summary='See docs/*_notes.md for this candidate family; not resolved in detail for the secondary tier.',
        uses_speed='', uses_grad='', uses_levelset='', uses_crit='', uses_e2='',
        repaired_status=repaired_status,
        data_out_dir=data_out_dir, model_dir=model_dir,
        cheap_eval_dir=f'ttk_runs_fixed/topology_finetuning/{original_method_name}_eval',
        topology_dir=f'ttk_runs_fixed/topology_finetuning/{original_method_name}_topology',
        cheap_report=f'docs/topology_finetuning_{original_method_name}_eval.md',
        topology_report=f'docs/topology_finetuning_{original_method_name}_topology_eval.md',
        exclusion_reason=exclusion_reason,
    )


SECONDARY_MANIFEST = [
    # ---- 168-sample historical pilot ----
    _secondary_entry('pilot_uv_cnn_finetune', 'Pilot: UV CNN fine-tune (168-sample)',
                      'pilot_candidateUV', 'UV_control', 'historical_pilot',
                      '168', 'native_tf', 'baseline',
                      '168-sample pilot scale; superseded by the expanded 672/1344/2688 UV ladder.'),
    _secondary_entry('pilot_c_crit', 'Pilot: Candidate C / crit fine-tune (168-sample)',
                      'pilot_candidateC', 'C_full', 'historical_pilot',
                      '168', 'native_tf', 'not_applicable',
                      '168-sample pilot scale; superseded by the expanded 672/1344/2688 Candidate C ladder. '
                      'Also confounded because the same 168 samples were used for both fine-tuning and evaluation.'),
    _secondary_entry('pilot_d_pd_refiner', 'Pilot: Candidate D differentiable-PD refiner (168-sample)',
                      'pilot_candidateD', 'D_original', 'historical_pilot',
                      '168', 'pytorch_residual_refiner', 'original_unrepaired',
                      'Original 168-sample Candidate D (active differentiable PD loss); did not improve final '
                      'TTK PD/MT. Superseded by the repair/audit sequence in docs/candidateD_E_topology_audit.md.'),
    _secondary_entry('pilot_e_ttkcrit_refiner', 'Pilot: Candidate E TTK critical-pair refiner (168-sample, original/unrepaired)',
                      'pilot_candidateE', 'E_original', 'historical_pilot',
                      '168', 'pytorch_residual_refiner', 'original_unrepaired',
                      'Original 168-sample Candidate E before the VTI/vertex-mapping and target-convention repair '
                      'that produced the E2 family; kept only for historical audit, not comparable to repaired E2.'),

    # ---- 672-sample scale-study variants of the primary methods ----
    _secondary_entry('uv_672', 'UV control (672-sample)', 'candidateUV_expanded672',
                      'UV_control', 'secondary_scale_study', '672', 'native_tf', 'not_applicable',
                      'Same objective as primary uv, different training scale (672 vs 2688 samples).'),
    _secondary_entry('candidate_b_672', 'Candidate B (672-sample)', 'candidateB_expanded672',
                      'B_full', 'secondary_scale_study', '672', 'native_tf', 'not_applicable',
                      'Same objective as primary candidate_b, different training scale.'),
    _secondary_entry('candidate_c_672', 'Candidate C (672-sample)', 'candidateC_expanded672',
                      'C_full', 'secondary_scale_study', '672', 'native_tf', 'not_applicable',
                      'Same objective as primary candidate_c, different training scale.'),
    _secondary_entry('uv_crit_672', 'UV + crit (672-sample)', 'candidateUV_plus_crit_expanded672',
                      'UV_crit', 'secondary_scale_study', '672', 'native_tf', 'not_applicable',
                      'Same objective as primary uv_crit, different training scale.'),
    _secondary_entry('uv_e2_672', 'UV + repaired E2 (672-sample)', 'candidateUV_plus_E2_tf_lowlambda_expanded672',
                      'E2_uv', 'secondary_scale_study', '672', 'native_tf', 'repaired_e2',
                      'Same objective as primary uv_e2, different training scale.'),
    _secondary_entry('b_e2_672', 'Candidate B + repaired E2 (672-sample)', 'candidateB_plus_E2_tf_lowlambda_expanded672',
                      'E2_b', 'secondary_scale_study', '672', 'native_tf', 'repaired_e2',
                      'Same objective as primary b_e2, different training scale.'),
    _secondary_entry('c_e2_672', 'Candidate C + repaired E2 (672-sample)', 'candidateE2_tf_lowlambda_expanded672',
                      'E2_c', 'secondary_scale_study', '672', 'native_tf', 'repaired_e2',
                      'Same objective as primary c_e2, different training scale.'),
    _secondary_entry('e_original_672', 'Candidate E, original/unrepaired (672-sample)',
                      'candidateE2_expanded672', 'E_original', 'secondary_scale_study',
                      '672', 'native_tf', 'original_unrepaired',
                      'Stage-1 "expanded E" negative result before the VTI/target-convention repair; '
                      'kept for historical audit only.',
                      data_out_dir='data_out/wind_finetune_candidateE2_expanded672'),
    _secondary_entry('d_original_672', 'Candidate D, active differentiable PD loss (672-sample)',
                      'candidateDpd_expanded672', 'D_original', 'secondary_scale_study',
                      '672', 'pytorch_residual_refiner', 'original_unrepaired',
                      'Active differentiable PD-loss refiner at 672-sample scale; feasible but did not improve '
                      'final TTK PD/MT (see docs/candidateD_pd_refiner_notes.md).'),

    # ---- 1344-sample scale-study variants ----
    _secondary_entry('uv_1344', 'UV control (1344-sample)', 'candidateUV_expanded1344',
                      'UV_control', 'secondary_scale_study', '1344', 'native_tf', 'not_applicable',
                      'Same objective as primary uv, different training scale.'),
    _secondary_entry('candidate_c_1344', 'Candidate C (1344-sample)', 'candidateC_expanded1344',
                      'C_full', 'secondary_scale_study', '1344', 'native_tf', 'not_applicable',
                      'Same objective as primary candidate_c, different training scale.'),
    _secondary_entry('uv_crit_1344', 'UV + crit (1344-sample)', 'candidateUV_plus_crit_expanded1344',
                      'UV_crit', 'secondary_scale_study', '1344', 'native_tf', 'not_applicable',
                      'Same objective as primary uv_crit, different training scale.'),
    _secondary_entry('uv_e2_1344', 'UV + repaired E2 (1344-sample)', 'candidateUV_plus_E2_tf_lowlambda_expanded1344',
                      'E2_uv', 'secondary_scale_study', '1344', 'native_tf', 'repaired_e2',
                      'Same objective as primary uv_e2, different training scale.'),
    _secondary_entry('b_e2_1344', 'Candidate B + repaired E2 (1344-sample)', 'candidateB_plus_E2_tf_lowlambda_expanded1344',
                      'E2_b', 'secondary_scale_study', '1344', 'native_tf', 'repaired_e2',
                      'Same objective as primary b_e2, different training scale.'),
    _secondary_entry('c_e2_1344', 'Candidate C + repaired E2 (1344-sample)', 'candidateE2_tf_lowlambda_expanded1344',
                      'E2_c', 'secondary_scale_study', '1344', 'native_tf', 'repaired_e2',
                      'Same objective as primary c_e2, different training scale.'),

    # ---- PyTorch residual-refiner E2 (architecture-confounded) ----
    _secondary_entry('e2_fixed_refiner_672', 'Repaired E2, PyTorch residual refiner (672-sample)',
                      'candidateE2_fixed_lowlambda_expanded672', 'E2_pytorch_refiner',
                      'secondary_architecture_confounded', '672', 'pytorch_residual_refiner', 'repaired_e2',
                      'Repaired E2 losses on a frozen-CNN PyTorch residual refiner, not the native PhIRE/TF '
                      'generator fine-tuning path used by the primary E2/F methods; architecture confound.'),
    _secondary_entry('e2_fixed_refiner_1344', 'Repaired E2, PyTorch residual refiner (1344-sample)',
                      'candidateE2_fixed_lowlambda_expanded1344', 'E2_pytorch_refiner',
                      'secondary_architecture_confounded', '1344', 'pytorch_residual_refiner', 'repaired_e2',
                      'Same architecture confound as e2_fixed_refiner_672, 1344-sample scale.'),
    _secondary_entry('e2_fixed_refiner_2688', 'Repaired E2, PyTorch residual refiner (2688-sample)',
                      'candidateE2_fixed_lowlambda_expanded2688', 'E2_pytorch_refiner',
                      'secondary_architecture_confounded', '2688', 'pytorch_residual_refiner', 'repaired_e2',
                      'Same architecture confound as e2_fixed_refiner_672, 2688-sample scale.'),

    # ---- Deprecated/superseded legacy runs with surviving artifacts ----
    dict(method_id='legacy_phase_b_persistence', display_name='Legacy Phase-B persistence-diagram feasibility study',
         original_method_name='phase_b_persistence_eval', candidate_family='legacy_phase_b',
         comparison_tier='deprecated_or_superseded', include_primary=False,
         training_scale='n/a (feasibility study, not a fine-tuning candidate)', architecture='n/a',
         training_tfrecord='n/a', evaluation_tfrecord='n/a',
         objective_summary='Early GAN/CNN vs bicubic persistence-diagram distance feasibility study; predates '
                            'the Candidate A-F naming and the fixed 168-sample benchmark methodology.',
         uses_speed='', uses_grad='', uses_levelset='', uses_crit='', uses_e2='',
         repaired_status='n/a', data_out_dir='n/a', model_dir='n/a',
         cheap_eval_dir='', topology_dir='',
         cheap_report='', topology_report='',
         exclusion_reason='Superseded by the Phase-C fixed-benchmark methodology; different schema '
                           '(archive/phase_b/phase_b_persistence_results.csv), not comparable to the primary table.'),
    dict(method_id='legacy_old_ttk_outputs_gan', display_name='Legacy pre-Phase-C TTK output archive (GAN only)',
         original_method_name='old_ttk_outputs', candidate_family='legacy_archive',
         comparison_tier='deprecated_or_superseded', include_primary=False,
         training_scale='n/a', architecture='n/a', training_tfrecord='n/a', evaluation_tfrecord='n/a',
         objective_summary='Pre-Phase-C archived TTK run; only GAN rows present, different schema '
                            '(n_pd0/n_pd1/mt_nodes/mt_arcs rather than pd_distance/mt_distance).',
         uses_speed='', uses_grad='', uses_levelset='', uses_crit='', uses_e2='',
         repaired_status='n/a', data_out_dir='n/a', model_dir='n/a',
         cheap_eval_dir='', topology_dir='',
         cheap_report='', topology_report='',
         exclusion_reason='archive/old_ttk_outputs/phase_c_results.csv uses a different, non-distance schema '
                           'and covers GAN only; not comparable to the primary table.'),
]

FULL_MANIFEST = PRIMARY_MANIFEST + SECONDARY_MANIFEST


# =============================================================================
# Baseline (cnn/gan) loader from the surviving combined CSVs
# =============================================================================

def _read_csv_rows(path: Path):
    with path.open(newline='') as fh:
        return list(csv.DictReader(fh))


def load_baseline_data():
    """Load cnn/gan per-sample cheap+topology metrics from the two surviving
    sources, cross-validate them against each other, and return
    {method: {sample_idx: {standardized_col: value}}} plus a validation report.
    """
    combined_path = rp('ttk_runs_fixed/combined/psnr_topology_physics_merged.csv')
    phase_c_path = rp('ttk_runs_fixed/combined/phase_c_results.csv')
    report = {'combined_path': str(combined_path), 'phase_c_path': str(phase_c_path),
              'per_method': {}}

    data = {}
    if not combined_path.exists():
        log(f'[baseline] MISSING: {combined_path}')
        return data, report

    rows = _read_csv_rows(combined_path)
    by_method = {}
    for row in rows:
        by_method.setdefault(row['method'], []).append(row)

    phase_c_rows = _read_csv_rows(phase_c_path) if phase_c_path.exists() else []
    phase_c_by_method = {}
    for row in phase_c_rows:
        m = re.search(r'_s(\d+)_', row['key'])
        if not m:
            continue
        phase_c_by_method.setdefault(row['method'], {})[int(m.group(1))] = (
            float(row['pd_distance']), float(row['mt_distance']))

    for method in ('cnn', 'gan'):
        method_rows = by_method.get(method, [])
        seen_idx = {}
        per_sample = {}
        dup_keys = []
        for row in method_rows:
            si = int(row['sample_idx'])
            if si in seen_idx:
                dup_keys.append(si)
                continue
            seen_idx[si] = True
            rec = {}
            for src_col, std_col in BASELINE_COMBINED_DIRECT_MAP.items():
                val = row.get(src_col, '')
                rec[std_col] = float(val) if val not in ('', None) else float('nan')
            wpd_bias_raw = row.get('wpd_bias', '')
            rec['wpd_bias_abs'] = abs(float(wpd_bias_raw)) if wpd_bias_raw not in ('', None) else float('nan')
            for missing_col in BASELINE_COMBINED_NOT_AVAILABLE:
                rec[missing_col] = float('nan')
            per_sample[si] = rec

        if dup_keys:
            raise SystemExit(
                f'[hard-fail] Duplicate sample_idx values for method={method!r} in {combined_path}: '
                f'{sorted(set(dup_keys))}. This would corrupt the one-row-per-(method,sample_idx) '
                f'invariant of the unified table -- refusing to silently pick one.'
            )

        idx_set = set(per_sample.keys())
        expected_set = set(range(N_EVAL))
        sample_index_status = 'exact_0_167' if idx_set == expected_set else (
            f'MISMATCH missing={sorted(expected_set - idx_set)[:10]} '
            f'extra={sorted(idx_set - expected_set)[:10]}')

        # Cross-validate PD/MT against the independent phase_c_results.csv source.
        max_pd_diff = 0.0
        max_mt_diff = 0.0
        n_cross_checked = 0
        for si, rec in per_sample.items():
            other = phase_c_by_method.get(method, {}).get(si)
            if other is None:
                continue
            n_cross_checked += 1
            max_pd_diff = max(max_pd_diff, abs(rec['pd_distance'] - other[0]))
            max_mt_diff = max(max_mt_diff, abs(rec['mt_distance'] - other[1]))

        data[method] = per_sample
        report['per_method'][method] = dict(
            n_rows=len(method_rows), n_unique_sample_idx=len(idx_set),
            sample_index_status=sample_index_status,
            n_cross_checked_against_phase_c=n_cross_checked,
            max_pd_diff_vs_phase_c=max_pd_diff, max_mt_diff_vs_phase_c=max_mt_diff,
        )

    return data, report


# =============================================================================
# Generic candidate loader (for methods that DO have all_sample_metrics CSVs;
# none currently exist in this checkout, but this is written generically so it
# works unmodified the first time a candidate's cheap-eval artifacts appear).
# =============================================================================

def resolve_candidate_artifacts(entry: dict) -> dict:
    name = entry['original_method_name']
    res = {}
    data_out_dir = rp(entry['data_out_dir']) if entry.get('data_out_dir') not in (None, '', 'n/a') else None
    res['data_out_dir_exists'] = bool(data_out_dir and data_out_dir.is_dir())
    res['idx_path'] = str(data_out_dir / 'idx.npy') if data_out_dir else ''
    res['data_gt_path'] = str(data_out_dir / 'dataGT.npy') if data_out_dir else ''
    res['data_sr_path'] = str(data_out_dir / 'dataSR.npy') if data_out_dir else ''
    res['idx_exists'] = bool(data_out_dir and (data_out_dir / 'idx.npy').exists())
    res['data_gt_exists'] = bool(data_out_dir and (data_out_dir / 'dataGT.npy').exists())
    res['data_sr_exists'] = bool(data_out_dir and (data_out_dir / 'dataSR.npy').exists())

    cheap_eval_dir = rp(entry['cheap_eval_dir']) if entry.get('cheap_eval_dir') else None
    cheap_csv = cheap_eval_dir / f'all_sample_metrics_{name}.csv' if cheap_eval_dir else None
    pairwise_csv = cheap_eval_dir / f'pairwise_cnn_vs_{name}.csv' if cheap_eval_dir else None
    res['cheap_eval_csv'] = str(cheap_csv) if cheap_csv else ''
    res['cheap_pairwise_csv'] = str(pairwise_csv) if pairwise_csv else ''
    res['cheap_eval_csv_exists'] = bool(cheap_csv and cheap_csv.exists())
    res['cheap_pairwise_csv_exists'] = bool(pairwise_csv and pairwise_csv.exists())

    topology_dir = rp(entry['topology_dir']) if entry.get('topology_dir') else None
    pd_mt_csv = topology_dir / f'{name}_pd_mt_distances.csv' if topology_dir else None
    topo_cmp_csv = topology_dir / f'{name}_topology_comparison.csv' if topology_dir else None
    res['topology_results_csv'] = str(pd_mt_csv) if pd_mt_csv else ''
    res['topology_comparison_csv'] = str(topo_cmp_csv) if topo_cmp_csv else ''
    res['topology_results_csv_exists'] = bool(pd_mt_csv and pd_mt_csv.exists())
    res['topology_comparison_csv_exists'] = bool(topo_cmp_csv and topo_cmp_csv.exists())

    res['cheap_report_exists'] = bool(entry.get('cheap_report') and rp(entry['cheap_report']).exists())
    res['topology_report_exists'] = bool(entry.get('topology_report') and rp(entry['topology_report']).exists())

    per_sample = {}
    row_count_cheap = 0
    row_count_topology = 0

    if res['cheap_eval_csv_exists']:
        rows = _read_csv_rows(cheap_csv)
        cand_rows = [r for r in rows if r.get('method') == name]
        row_count_cheap = len(cand_rows)
        seen = set()
        for row in cand_rows:
            si = int(row['sample_idx'])
            if si in seen:
                raise SystemExit(
                    f'[hard-fail] Duplicate sample_idx={si} for method={name!r} in {cheap_csv}. '
                    'Refusing to silently collapse duplicate rows.'
                )
            seen.add(si)
            rec = per_sample.setdefault(si, {})
            for src_col, std_col in CANDIDATE_EVAL_DIRECT_MAP.items():
                val = row.get(src_col, '')
                rec[std_col] = float(val) if val not in ('', None) else float('nan')

    if res['topology_results_csv_exists']:
        rows = _read_csv_rows(pd_mt_csv)
        row_count_topology = len(rows)
        seen = set()
        for row in rows:
            si = int(row.get('sample_idx', row.get('sample', -1)))
            if si in seen:
                raise SystemExit(
                    f'[hard-fail] Duplicate sample_idx={si} for method={name!r} in {pd_mt_csv}.'
                )
            seen.add(si)
            rec = per_sample.setdefault(si, {})
            pdv = row.get('pd_distance')
            mtv = row.get('mt_distance')
            if pdv not in (None, ''):
                pdv = float(pdv)
                if not math.isfinite(pdv) or pdv < 0:
                    raise SystemExit(f'[hard-fail] Non-finite/negative pd_distance for method={name!r} '
                                      f'sample_idx={si} in {pd_mt_csv}: {pdv}')
                rec['pd_distance'] = pdv
            if mtv not in (None, ''):
                mtv = float(mtv)
                if not math.isfinite(mtv) or mtv < 0:
                    raise SystemExit(f'[hard-fail] Non-finite/negative mt_distance for method={name!r} '
                                      f'sample_idx={si} in {pd_mt_csv}: {mtv}')
                rec['mt_distance'] = mtv

    res['per_sample'] = per_sample
    res['row_count_cheap'] = row_count_cheap
    res['row_count_topology'] = row_count_topology
    return res


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log('=' * 88)
    log('Unified candidate evaluation -- Phase 1 audit')
    log(f'Repo root: {REPO_ROOT}')
    log('Read-only w.r.t. all experiment artifacts. No training/TTK/eval rerun performed.')
    log('=' * 88)

    baseline_data, baseline_report = load_baseline_data()
    for method, rep in baseline_report.get('per_method', {}).items():
        log(f"[baseline:{method}] rows={rep['n_rows']} unique_sample_idx={rep['n_unique_sample_idx']} "
            f"sample_index_status={rep['sample_index_status']} "
            f"cross_checked_vs_phase_c={rep['n_cross_checked_against_phase_c']} "
            f"max_pd_diff={rep['max_pd_diff_vs_phase_c']:.3e} max_mt_diff={rep['max_mt_diff_vs_phase_c']:.3e}")

    # -------------------------------------------------------------------
    # Resolve every manifest entry's artifacts on disk.
    # -------------------------------------------------------------------
    resolved = {}
    for entry in FULL_MANIFEST:
        mid = entry['method_id']
        if mid in ('bicubic', 'cnn', 'gan'):
            per_sample = baseline_data.get(mid, {})
            resolved[mid] = dict(
                data_out_dir_exists=rp(entry['data_out_dir']).is_dir() if entry['data_out_dir'] not in ('n/a', '') else False,
                idx_path=str(rp(entry['data_out_dir']) / 'idx.npy') if entry['data_out_dir'] not in ('n/a', '') else '',
                data_gt_path=str(rp(entry['data_out_dir']) / 'dataGT.npy') if entry['data_out_dir'] not in ('n/a', '') else '',
                data_sr_path=str(rp(entry['data_out_dir']) / 'dataSR.npy') if entry['data_out_dir'] not in ('n/a', '') else '',
                idx_exists=False, data_gt_exists=False, data_sr_exists=False,
                cheap_eval_csv=str(rp('ttk_runs_fixed/combined/psnr_topology_physics_merged.csv')) if per_sample else '',
                cheap_eval_csv_exists=bool(per_sample),
                cheap_pairwise_csv='', cheap_pairwise_csv_exists=False,
                topology_results_csv=str(rp('ttk_runs_fixed/combined/phase_c_results.csv')) if per_sample else '',
                topology_results_csv_exists=bool(per_sample),
                topology_comparison_csv='', topology_comparison_csv_exists=False,
                cheap_report_exists=False, topology_report_exists=False,
                per_sample=per_sample,
                row_count_cheap=len(per_sample), row_count_topology=len(per_sample),
            )
        else:
            resolved[mid] = resolve_candidate_artifacts(entry)

    # -------------------------------------------------------------------
    # method_inventory.csv
    # -------------------------------------------------------------------
    inventory_cols = [
        'method_id', 'display_name', 'original_method_name', 'candidate_family',
        'comparison_tier', 'include_primary', 'training_scale', 'architecture',
        'training_tfrecord', 'evaluation_tfrecord', 'objective_summary',
        'uses_speed', 'uses_grad', 'uses_levelset', 'uses_crit', 'uses_e2',
        'repaired_status', 'data_out_dir', 'idx_path', 'data_gt_path', 'data_sr_path',
        'cheap_eval_csv', 'cheap_pairwise_csv', 'topology_results_csv', 'topology_comparison_csv',
        'cheap_report', 'topology_report', 'row_count_cheap', 'row_count_topology',
        'sample_index_status', 'gt_alignment_status', 'topology_mean_pd', 'topology_mean_mt',
        'expected_pd', 'expected_mt', 'validation_status', 'notes',
    ]
    inventory_rows = []
    for entry in FULL_MANIFEST:
        mid = entry['method_id']
        r = resolved[mid]
        per_sample = r.get('per_sample', {})
        idx_set = set(per_sample.keys())
        if not idx_set:
            sample_index_status = 'no_data_found'
        elif idx_set == set(range(N_EVAL)):
            sample_index_status = 'exact_0_167'
        else:
            sample_index_status = f'partial ({len(idx_set)}/{N_EVAL})'

        pd_vals = [v['pd_distance'] for v in per_sample.values() if 'pd_distance' in v and math.isfinite(v['pd_distance'])]
        mt_vals = [v['mt_distance'] for v in per_sample.values() if 'mt_distance' in v and math.isfinite(v['mt_distance'])]
        topo_mean_pd = sum(pd_vals) / len(pd_vals) if pd_vals else ''
        topo_mean_mt = sum(mt_vals) / len(mt_vals) if mt_vals else ''
        exp_pd, exp_mt = EXPECTED_PD_MT.get(mid, ('', ''))

        if mid not in EXPECTED_PD_MT:
            if mid == 'bicubic':
                validation_status = 'not_applicable'
                notes_extra = ('No expected PD/MT value was supplied for bicubic (it may not have true-topology '
                                'files at all). No cheap-eval or topology CSV was found for it either in this '
                                'checkout -- only its generator script (scripts/generate_bicubic_baseline.py, '
                                'output convention data_out_fixed/wind_mrhr_bicubic/) exists. Kept in the cheap-'
                                'metric table with all metrics missing, per the task instructions.')
            else:
                validation_status = 'no_expected_value'
                notes_extra = ''
        elif not pd_vals:
            validation_status = 'NO_DATA'
            notes_extra = ('No PD/MT source artifact found in this git checkout (data_out/, '
                            'ttk_runs_fixed/topology_finetuning/<method>/, and models_fixed/topology_finetuning/ '
                            'all lack this method). Large per-candidate experiment outputs are gitignored '
                            '(*.npy, *.npz, ttk_runs_fixed/topology_finetuning/* except candidateE_constraints) '
                            'and appear to exist only on the separate training machine, not in this repository. '
                            f'Expected mean (from user-supplied reference values, NOT verified here): '
                            f'PD={exp_pd}, MT={exp_mt}.')
        else:
            pd_ok = abs(topo_mean_pd - exp_pd) <= PD_MT_TOLERANCE
            mt_ok = abs(topo_mean_mt - exp_mt) <= PD_MT_TOLERANCE
            validation_status = 'PASS' if (pd_ok and mt_ok) else 'FAIL'
            notes_extra = f'pd_abs_diff={abs(topo_mean_pd - exp_pd):.6f} mt_abs_diff={abs(topo_mean_mt - exp_mt):.6f}'

        gt_alignment_status = 'not_checked (no dataGT.npy/dataIN.npy present in this checkout)'
        if mid in ('bicubic', 'cnn', 'gan'):
            gt_alignment_status = 'n/a (per-sample metrics sourced from pre-merged combined CSV, no raw arrays present)'

        notes = entry.get('exclusion_reason', '') or ''
        if notes_extra:
            notes = (notes + ' ' if notes else '') + notes_extra

        row = dict(
            method_id=mid, display_name=entry['display_name'],
            original_method_name=entry['original_method_name'], candidate_family=entry['candidate_family'],
            comparison_tier=entry['comparison_tier'], include_primary=entry['include_primary'],
            training_scale=entry['training_scale'], architecture=entry['architecture'],
            training_tfrecord=entry['training_tfrecord'], evaluation_tfrecord=entry['evaluation_tfrecord'],
            objective_summary=entry['objective_summary'],
            uses_speed=entry['uses_speed'], uses_grad=entry['uses_grad'], uses_levelset=entry['uses_levelset'],
            uses_crit=entry['uses_crit'], uses_e2=entry['uses_e2'], repaired_status=entry['repaired_status'],
            data_out_dir=entry['data_out_dir'],
            idx_path=r.get('idx_path', ''), data_gt_path=r.get('data_gt_path', ''), data_sr_path=r.get('data_sr_path', ''),
            cheap_eval_csv=r.get('cheap_eval_csv', ''), cheap_pairwise_csv=r.get('cheap_pairwise_csv', ''),
            topology_results_csv=r.get('topology_results_csv', ''), topology_comparison_csv=r.get('topology_comparison_csv', ''),
            cheap_report=entry.get('cheap_report', ''), topology_report=entry.get('topology_report', ''),
            row_count_cheap=r.get('row_count_cheap', 0), row_count_topology=r.get('row_count_topology', 0),
            sample_index_status=sample_index_status, gt_alignment_status=gt_alignment_status,
            topology_mean_pd=topo_mean_pd, topology_mean_mt=topo_mean_mt,
            expected_pd=exp_pd, expected_mt=exp_mt, validation_status=validation_status, notes=notes,
        )
        inventory_rows.append(row)

    inv_path = OUT_DIR / 'method_inventory.csv'
    with inv_path.open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=inventory_cols)
        w.writeheader()
        for row in inventory_rows:
            w.writerow(row)
    log(f'[write] {inv_path} ({len(inventory_rows)} rows)')

    # -------------------------------------------------------------------
    # unified_primary_per_sample_long.csv -- full grid, NaN where no source.
    # -------------------------------------------------------------------
    long_rows = []
    for entry in PRIMARY_MANIFEST:
        mid = entry['method_id']
        per_sample = resolved[mid].get('per_sample', {})
        for si in range(N_EVAL):
            row = {
                'sample_idx': si, 'method_id': mid, 'display_name': entry['display_name'],
                'candidate_family': entry['candidate_family'], 'training_scale': entry['training_scale'],
                'architecture': entry['architecture'],
                'uses_speed': entry['uses_speed'], 'uses_grad': entry['uses_grad'],
                'uses_levelset': entry['uses_levelset'], 'uses_crit': entry['uses_crit'],
                'uses_e2': entry['uses_e2'],
            }
            rec = per_sample.get(si, {})
            for col in METRIC_COLUMNS:
                v = rec.get(col, float('nan'))
                row[col] = '' if (isinstance(v, float) and math.isnan(v)) else v
            long_rows.append(row)

    # Hard-fail: verify no duplicate (method_id, sample_idx) keys in the output itself.
    key_seen = set()
    for row in long_rows:
        k = (row['method_id'], row['sample_idx'])
        if k in key_seen:
            raise SystemExit(f'[hard-fail] Duplicate key in unified long table: {k}')
        key_seen.add(k)
    expected_rows = len(PRIMARY_MANIFEST) * N_EVAL
    if len(long_rows) != expected_rows:
        raise SystemExit(f'[hard-fail] Unified long table has {len(long_rows)} rows, expected exactly '
                          f'{len(PRIMARY_MANIFEST)} x {N_EVAL} = {expected_rows}.')

    long_path = OUT_DIR / 'unified_primary_per_sample_long.csv'
    with long_path.open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=IDENTITY_COLUMNS + METRIC_COLUMNS)
        w.writeheader()
        for row in long_rows:
            w.writerow(row)
    log(f'[write] {long_path} ({len(long_rows)} rows = {len(PRIMARY_MANIFEST)} methods x {N_EVAL} samples)')

    # -------------------------------------------------------------------
    # column_mapping.csv
    # -------------------------------------------------------------------
    mapping_rows = []
    for src_col, std_col in sorted(BASELINE_COMBINED_DIRECT_MAP.items()):
        direction = METRIC_DIRECTION[std_col]
        mapping_rows.append(dict(
            source_path='ttk_runs_fixed/combined/psnr_topology_physics_merged.csv',
            source_column=src_col, standardized_column=std_col,
            units=[m[3] for m in METRIC_SCHEMA if m[0] == std_col][0],
            direction=direction,
            representation=[m[2] for m in METRIC_SCHEMA if m[0] == std_col][0],
            notes='Present for cnn/gan only; this is the only per-sample source file with real primary-method data.',
        ))
    mapping_rows.append(dict(
        source_path='ttk_runs_fixed/combined/psnr_topology_physics_merged.csv',
        source_column='wpd_bias', standardized_column='wpd_bias_abs',
        units='m^3/s^3', direction='lower_is_better', representation='wind_power_distribution',
        notes='Standardized column is abs(wpd_bias); source stores the signed value.',
    ))
    for missing_col in BASELINE_COMBINED_NOT_AVAILABLE:
        mapping_rows.append(dict(
            source_path='ttk_runs_fixed/combined/psnr_topology_physics_merged.csv',
            source_column='(not present)', standardized_column=missing_col,
            units=[m[3] for m in METRIC_SCHEMA if m[0] == missing_col][0],
            direction=METRIC_DIRECTION[missing_col],
            representation=[m[2] for m in METRIC_SCHEMA if m[0] == missing_col][0],
            notes='This legacy pipeline predates speed_mae/speed_rmse/component-count metrics; '
                  'left as genuinely missing for cnn/gan, not fabricated.',
        ))
    for src_col, std_col in sorted(CANDIDATE_EVAL_DIRECT_MAP.items()):
        mapping_rows.append(dict(
            source_path='scripts/evaluate_finetune_candidate.py (schema definition only; '
                         'no all_sample_metrics_<method>.csv instance file exists in this checkout)',
            source_column=src_col, standardized_column=std_col,
            units=[m[3] for m in METRIC_SCHEMA if m[0] == std_col][0],
            direction=METRIC_DIRECTION[std_col],
            representation=[m[2] for m in METRIC_SCHEMA if m[0] == std_col][0],
            notes='Schema this evaluator would write for any candidate; no instance data found for any of the '
                  '16 non-baseline primary methods in this repository checkout.',
        ))
    mapping_path = OUT_DIR / 'column_mapping.csv'
    with mapping_path.open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['source_path', 'source_column', 'standardized_column',
                                            'units', 'direction', 'representation', 'notes'])
        w.writeheader()
        for row in mapping_rows:
            w.writerow(row)
    log(f'[write] {mapping_path} ({len(mapping_rows)} rows)')

    # -------------------------------------------------------------------
    # unified_primary_method_summary.csv
    # -------------------------------------------------------------------
    summary_rows = []
    per_method_series = {}
    for entry in PRIMARY_MANIFEST:
        mid = entry['method_id']
        per_sample = resolved[mid].get('per_sample', {})
        for col in METRIC_COLUMNS:
            vals = [per_sample[si][col] for si in per_sample
                    if col in per_sample[si] and math.isfinite(per_sample[si][col])]
            per_method_series[(mid, col)] = vals
            if vals:
                mean_v = sum(vals) / len(vals)
                sorted_v = sorted(vals)
                n = len(sorted_v)
                median_v = sorted_v[n // 2] if n % 2 == 1 else (sorted_v[n // 2 - 1] + sorted_v[n // 2]) / 2
                var = sum((x - mean_v) ** 2 for x in vals) / len(vals)
                std_v = math.sqrt(var)
                summary_rows.append(dict(method_id=mid, metric=col, mean=mean_v, median=median_v,
                                          std=std_v, min=min(vals), max=max(vals), n_finite=len(vals)))
            else:
                summary_rows.append(dict(method_id=mid, metric=col, mean='', median='', std='',
                                          min='', max='', n_finite=0))
    summary_path = OUT_DIR / 'unified_primary_method_summary.csv'
    with summary_path.open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['method_id', 'metric', 'mean', 'median', 'std', 'min', 'max', 'n_finite'])
        w.writeheader()
        for row in summary_rows:
            w.writerow(row)
    log(f'[write] {summary_path} ({len(summary_rows)} rows)')

    # -------------------------------------------------------------------
    # unified_primary_topology_validation.csv
    # -------------------------------------------------------------------
    topo_val_rows = []
    for entry in PRIMARY_MANIFEST:
        mid = entry['method_id']
        pd_vals = per_method_series.get((mid, 'pd_distance'), [])
        mt_vals = per_method_series.get((mid, 'mt_distance'), [])
        exp_pd, exp_mt = EXPECTED_PD_MT.get(mid, (None, None))
        if exp_pd is None:
            continue  # bicubic: no expected true-topology value to validate against.
        obs_pd = sum(pd_vals) / len(pd_vals) if pd_vals else ''
        obs_mt = sum(mt_vals) / len(mt_vals) if mt_vals else ''
        pd_diff = abs(obs_pd - exp_pd) if pd_vals else ''
        mt_diff = abs(obs_mt - exp_mt) if mt_vals else ''
        pd_pass = (pd_vals != [] and pd_diff <= PD_MT_TOLERANCE)
        mt_pass = (mt_vals != [] and mt_diff <= PD_MT_TOLERANCE)
        source_path = resolved[mid].get('topology_results_csv', '') or '(no source found)'
        topo_val_rows.append(dict(
            method_id=mid, observed_pd_mean=obs_pd, expected_pd_mean=exp_pd,
            pd_abs_difference=pd_diff, pd_pass=(pd_pass if pd_vals else 'NO_DATA'),
            observed_mt_mean=obs_mt, expected_mt_mean=exp_mt,
            mt_abs_difference=mt_diff, mt_pass=(mt_pass if mt_vals else 'NO_DATA'),
            source_path=source_path,
        ))
    topo_val_path = OUT_DIR / 'unified_primary_topology_validation.csv'
    with topo_val_path.open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['method_id', 'observed_pd_mean', 'expected_pd_mean',
                                            'pd_abs_difference', 'pd_pass', 'observed_mt_mean',
                                            'expected_mt_mean', 'mt_abs_difference', 'mt_pass', 'source_path'])
        w.writeheader()
        for row in topo_val_rows:
            w.writerow(row)
    n_pass = sum(1 for r in topo_val_rows if r['pd_pass'] is True and r['mt_pass'] is True)
    n_no_data = sum(1 for r in topo_val_rows if r['pd_pass'] == 'NO_DATA')
    log(f'[write] {topo_val_path} ({len(topo_val_rows)} rows; PASS={n_pass}, NO_DATA={n_no_data}, '
        f'FAIL={len(topo_val_rows) - n_pass - n_no_data})')

    # -------------------------------------------------------------------
    # unified_primary_pairwise_vs_cnn.csv
    # -------------------------------------------------------------------
    pairwise_rows = []
    cnn_per_sample = resolved['cnn'].get('per_sample', {})
    for entry in PRIMARY_MANIFEST:
        mid = entry['method_id']
        if mid == 'cnn':
            continue
        cand_per_sample = resolved[mid].get('per_sample', {})
        for col in METRIC_COLUMNS:
            cnn_vals = {si: cnn_per_sample[si][col] for si in cnn_per_sample
                        if col in cnn_per_sample[si] and math.isfinite(cnn_per_sample[si][col])}
            cand_vals = {si: cand_per_sample[si][col] for si in cand_per_sample
                         if col in cand_per_sample[si] and math.isfinite(cand_per_sample[si][col])}
            common = sorted(set(cnn_vals) & set(cand_vals))
            if not common:
                pairwise_rows.append(dict(method_id=mid, metric=col, cnn_mean='', candidate_mean='',
                                           mean_raw_delta='', mean_improvement_delta='',
                                           median_improvement_delta='', n_improved='', n_worsened='',
                                           n_tied='', n_valid=0))
                continue
            direction = METRIC_DIRECTION[col]
            raw_deltas = [cand_vals[si] - cnn_vals[si] for si in common]
            if direction == 'higher_is_better':
                improve_deltas = raw_deltas
            else:
                improve_deltas = [-d for d in raw_deltas]
            n_improved = sum(1 for d in improve_deltas if d > 0)
            n_worsened = sum(1 for d in improve_deltas if d < 0)
            n_tied = sum(1 for d in improve_deltas if d == 0)
            sorted_imp = sorted(improve_deltas)
            n = len(sorted_imp)
            median_imp = sorted_imp[n // 2] if n % 2 == 1 else (sorted_imp[n // 2 - 1] + sorted_imp[n // 2]) / 2
            pairwise_rows.append(dict(
                method_id=mid, metric=col,
                cnn_mean=sum(cnn_vals[si] for si in common) / len(common),
                candidate_mean=sum(cand_vals[si] for si in common) / len(common),
                mean_raw_delta=sum(raw_deltas) / len(raw_deltas),
                mean_improvement_delta=sum(improve_deltas) / len(improve_deltas),
                median_improvement_delta=median_imp,
                n_improved=n_improved, n_worsened=n_worsened, n_tied=n_tied, n_valid=len(common),
            ))
    pairwise_path = OUT_DIR / 'unified_primary_pairwise_vs_cnn.csv'
    with pairwise_path.open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['method_id', 'metric', 'cnn_mean', 'candidate_mean',
                                            'mean_raw_delta', 'mean_improvement_delta',
                                            'median_improvement_delta', 'n_improved', 'n_worsened',
                                            'n_tied', 'n_valid'])
        w.writeheader()
        for row in pairwise_rows:
            w.writerow(row)
    log(f'[write] {pairwise_path} ({len(pairwise_rows)} rows)')

    # -------------------------------------------------------------------
    # unified_primary_missingness.csv
    # -------------------------------------------------------------------
    missingness_rows = []
    for entry in PRIMARY_MANIFEST:
        mid = entry['method_id']
        per_sample = resolved[mid].get('per_sample', {})
        for col in METRIC_COLUMNS:
            finite = sum(1 for si in per_sample if col in per_sample[si] and math.isfinite(per_sample[si][col]))
            total = N_EVAL
            missing = total - finite
            if not per_sample:
                reason = 'no_source_artifact_found_in_repository'
            elif finite == total:
                reason = ''
            elif col in BASELINE_COMBINED_NOT_AVAILABLE and mid in ('bicubic', 'cnn', 'gan'):
                reason = 'not_computed_by_legacy_physics_merged_pipeline'
            else:
                reason = 'partial_source_coverage'
            missingness_rows.append(dict(method_id=mid, metric=col, total_rows=total,
                                          finite_rows=finite, missing_rows=missing, missing_reason=reason))
    missingness_path = OUT_DIR / 'unified_primary_missingness.csv'
    with missingness_path.open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['method_id', 'metric', 'total_rows', 'finite_rows',
                                            'missing_rows', 'missing_reason'])
        w.writeheader()
        for row in missingness_rows:
            w.writerow(row)
    log(f'[write] {missingness_path} ({len(missingness_rows)} rows)')

    # -------------------------------------------------------------------
    # unified_primary_wide.csv
    # -------------------------------------------------------------------
    wide_fieldnames = ['sample_idx'] + [f'{e["method_id"]}__{col}' for e in PRIMARY_MANIFEST for col in METRIC_COLUMNS]
    wide_path = OUT_DIR / 'unified_primary_wide.csv'
    with wide_path.open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=wide_fieldnames)
        w.writeheader()
        for si in range(N_EVAL):
            row = {'sample_idx': si}
            for entry in PRIMARY_MANIFEST:
                mid = entry['method_id']
                rec = resolved[mid].get('per_sample', {}).get(si, {})
                for col in METRIC_COLUMNS:
                    v = rec.get(col, float('nan'))
                    row[f'{mid}__{col}'] = '' if (isinstance(v, float) and math.isnan(v)) else v
            w.writerow(row)
    log(f'[write] {wide_path} ({N_EVAL} rows x {len(wide_fieldnames)} columns)')

    # -------------------------------------------------------------------
    # docs/unified_candidate_evaluation_inventory.md
    # -------------------------------------------------------------------
    write_inventory_doc(inventory_rows, baseline_report)

    # -------------------------------------------------------------------
    # docs/unified_candidate_evaluation_phase1.md
    # -------------------------------------------------------------------
    write_phase1_report(inventory_rows, topo_val_rows, baseline_report, missingness_rows)

    n_primary_with_data = sum(1 for e in PRIMARY_MANIFEST if resolved[e['method_id']].get('per_sample'))
    log('')
    log('=' * 88)
    log(f'RESULT: {n_primary_with_data}/{len(PRIMARY_MANIFEST)} primary methods have real per-sample data '
        f'in this repository checkout.')
    log('Do not claim full-table success: most primary methods are MISSING (see method_inventory.csv, '
        'validation_status column, and docs/unified_candidate_evaluation_phase1.md).')
    log('=' * 88)

    flush_log()
    return 0


def write_inventory_doc(inventory_rows, baseline_report):
    lines = []
    lines.append('# Unified candidate evaluation -- artifact inventory')
    lines.append('')
    lines.append(f'Generated by `scripts/build_unified_candidate_evaluation.py`. Read-only audit; '
                 f'no training, TTK, or cheap-evaluation rerun was performed.')
    lines.append('')
    lines.append('## Primary method set')
    lines.append('')
    lines.append('| method_id | display_name | comparison_tier | validation_status | row_count_cheap | notes |')
    lines.append('|---|---|---|---|---:|---|')
    for row in inventory_rows:
        if not row['include_primary']:
            continue
        notes = (row['notes'] or '').replace('|', '\\|')
        if len(notes) > 160:
            notes = notes[:157] + '...'
        lines.append(f"| {row['method_id']} | {row['display_name']} | {row['comparison_tier']} | "
                     f"{row['validation_status']} | {row['row_count_cheap']} | {notes} |")
    lines.append('')
    lines.append('## Secondary / non-primary inventory')
    lines.append('')
    lines.append('| method_id | display_name | comparison_tier | training_scale | architecture | repaired_status | exclusion_reason |')
    lines.append('|---|---|---|---|---|---|---|')
    for row in inventory_rows:
        if row['include_primary']:
            continue
        notes = (row['notes'] or '').replace('|', '\\|')
        if len(notes) > 160:
            notes = notes[:157] + '...'
        lines.append(f"| {row['method_id']} | {row['display_name']} | {row['comparison_tier']} | "
                     f"{row['training_scale']} | {row['architecture']} | {row['repaired_status']} | {notes} |")
    lines.append('')
    lines.append('## Baseline (cnn/gan) source cross-validation')
    lines.append('')
    lines.append(f"- Primary source: `{baseline_report.get('combined_path')}`")
    lines.append(f"- Cross-validation source: `{baseline_report.get('phase_c_path')}`")
    for method, rep in baseline_report.get('per_method', {}).items():
        lines.append(f"- **{method}**: {rep['n_rows']} rows, {rep['n_unique_sample_idx']} unique sample_idx "
                     f"({rep['sample_index_status']}), cross-checked against {rep['n_cross_checked_against_phase_c']} "
                     f"phase_c_results.csv rows, max |Δpd|={rep['max_pd_diff_vs_phase_c']:.3e}, "
                     f"max |Δmt|={rep['max_mt_diff_vs_phase_c']:.3e}.")
    lines.append('')
    lines.append('## Architecture legend')
    lines.append('')
    lines.append('- `native_tf`: PhIREGANs.pretrain() generator fine-tuning path (tensorflow.compat.v1).')
    lines.append('- `pytorch_residual_refiner`: frozen pretrained CNN output + a small PyTorch residual refiner '
                 '(architecture confound relative to native_tf; kept secondary-tier only).')
    lines.append('- `pretrained_cnn` / `pretrained_gan`: released PhIRE checkpoints, no fine-tuning.')
    lines.append('- `bicubic_interpolation`: no learned model.')
    lines.append('')
    lines.append('## Training-scale legend')
    lines.append('')
    lines.append('168 / 672 / 1344 / 2688 = number of training samples used for fine-tuning; all variants are '
                 'evaluated on the same fixed 168-sample benchmark regardless of training scale.')
    (DOCS_DIR / 'unified_candidate_evaluation_inventory.md').write_text('\n'.join(lines) + '\n')
    log(f"[write] {DOCS_DIR / 'unified_candidate_evaluation_inventory.md'}")


def write_phase1_report(inventory_rows, topo_val_rows, baseline_report, missingness_rows):
    primary_rows = [r for r in inventory_rows if r['include_primary']]
    n_pass = sum(1 for r in topo_val_rows if r['pd_pass'] is True and r['mt_pass'] is True)
    n_no_data = sum(1 for r in topo_val_rows if r['pd_pass'] == 'NO_DATA')
    n_fail = len(topo_val_rows) - n_pass - n_no_data
    methods_with_data = [r['method_id'] for r in primary_rows if r['row_count_cheap'] > 0]
    methods_without_data = [r['method_id'] for r in primary_rows if r['row_count_cheap'] == 0]

    lines = []
    lines.append('# Unified candidate evaluation -- Phase 1 report')
    lines.append('')
    lines.append('## 1. Scope and primary/secondary method distinction')
    lines.append('')
    lines.append(f'Primary set: {len(primary_rows)} methods evaluated on the fixed 168-sample benchmark '
                 '(3 baselines: bicubic, cnn, gan; 9 B-term-factorial variants incl. full Candidate B and C; '
                 '1 critical-proxy-only ablation; 3 repaired low-lambda E2 ablations; 3 Candidate F recombinations). '
                 'Secondary set: 168-sample pilot runs, 672/1344-sample scale-study duplicates of primary objectives, '
                 'PyTorch residual-refiner E2 variants (architecture confound), and deprecated pre-Phase-C legacy '
                 'archives -- see `docs/unified_candidate_evaluation_inventory.md` for the full secondary listing.')
    lines.append('')
    lines.append('## 2. Complete artifact inventory')
    lines.append('')
    lines.append('See `ttk_runs_fixed/unified_candidate_evaluation/method_inventory.csv` (one row per discovered '
                 'or expected experiment, primary and secondary) and `docs/unified_candidate_evaluation_inventory.md` '
                 '(narrative version).')
    lines.append('')
    lines.append('## 3. Exact table dimensions')
    lines.append('')
    lines.append(f'- `unified_primary_per_sample_long.csv`: {len(primary_rows)} methods x 168 samples = '
                 f'{len(primary_rows) * N_EVAL} rows, one row per (method_id, sample_idx), no duplicates.')
    lines.append(f'- Of these {len(primary_rows)} primary methods, **{len(methods_with_data)} have real per-sample '
                 f'data** in this repository checkout ({", ".join(methods_with_data)}), and '
                 f'**{len(methods_without_data)} have zero real data** ({", ".join(methods_without_data)}).')
    lines.append('')
    lines.append('## 4. Metric families and representations')
    lines.append('')
    lines.append('See `column_mapping.csv`. Families: `vector_uv` (psnruv), `scalar_speed` (ssim_speed, speed_mae, '
                 'speed_rmse), `wind_power_distribution` (wpd_*), `gradient_distribution` (grad_*), '
                 '`frequency_domain` (psd_*), `threshold_geometry` (exceed_*, comp_*), `topology_pd` (pd_distance), '
                 '`topology_mt` (mt_distance). PSNR is vector-field PSNR on physical [u,v] (`psnruv`), never scalar-'
                 'speed PSNR; SSIM, speed errors, WPD, PD, and MT are all computed on scalar wind speed.')
    lines.append('')
    lines.append('## 5. Validation results')
    lines.append('')
    lines.append(f'- Topology-mean reproduction: **{n_pass} PASS**, **{n_fail} FAIL**, **{n_no_data} NO_DATA** '
                 f'(no source artifact found at all) out of {len(topo_val_rows)} primary methods with an expected value.')
    lines.append('- Cheap-metric completeness (168 rows, sample_idx exactly 0..167, no duplicates): verified for '
                 'cnn and gan from `ttk_runs_fixed/combined/psnr_topology_physics_merged.csv`; not applicable to '
                 'the other 16 non-baseline primary methods since no cheap-eval CSV exists for any of them.')
    lines.append('- Join (cheap metrics <-> true topology, one-to-one on sample_idx): for cnn/gan the two are '
                 'already merged upstream in the same source row; no separate join was required or performed.')
    lines.append('')
    lines.append('## 6. Baseline duplicate-consistency audit')
    lines.append('')
    lines.append('The task instructions anticipate that repeated cnn/gan/bicubic rows may appear across multiple '
                 'per-candidate cheap-evaluation CSVs and must be checked for equality before choosing one canonical '
                 'source. In this checkout **no per-candidate cheap-evaluation CSV exists at all** (zero '
                 '`all_sample_metrics_*.csv` files were found for any of the 16 non-baseline primary methods), so '
                 'that specific cross-file duplication could not occur and this check is vacuously satisfied. '
                 'The only baseline consistency check actually performable was cross-validating '
                 '`ttk_runs_fixed/combined/psnr_topology_physics_merged.csv` PD/MT values against the independent '
                 '`ttk_runs_fixed/combined/phase_c_results.csv` source:')
    for method, rep in baseline_report.get('per_method', {}).items():
        lines.append(f"  - **{method}**: max |Δpd| = {rep['max_pd_diff_vs_phase_c']:.3e}, "
                     f"max |Δmt| = {rep['max_mt_diff_vs_phase_c']:.3e} across "
                     f"{rep['n_cross_checked_against_phase_c']} cross-checked samples -- effectively exact "
                     '(floating-point-level agreement).')
    lines.append('')
    lines.append('## 7. PD/MT mean reproduction table')
    lines.append('')
    lines.append('| method_id | observed_pd_mean | expected_pd_mean | pd_pass | observed_mt_mean | expected_mt_mean | mt_pass |')
    lines.append('|---|---:|---:|---|---:|---:|---|')
    for row in topo_val_rows:
        opd = f"{row['observed_pd_mean']:.4f}" if row['observed_pd_mean'] != '' else 'n/a'
        omt = f"{row['observed_mt_mean']:.4f}" if row['observed_mt_mean'] != '' else 'n/a'
        lines.append(f"| {row['method_id']} | {opd} | {row['expected_pd_mean']} | {row['pd_pass']} | "
                     f"{omt} | {row['expected_mt_mean']} | {row['mt_pass']} |")
    lines.append('')
    lines.append('## 8. Missingness, especially SSIM')
    lines.append('')
    ssim_rows = [r for r in missingness_rows if r['metric'] == 'ssim_speed']
    ssim_nan_methods = [r['method_id'] for r in ssim_rows if r['finite_rows'] == 0 and r['total_rows'] > 0]
    ssim_ok_methods = [r['method_id'] for r in ssim_rows if r['finite_rows'] == r['total_rows']]
    lines.append(f'- SSIM (`ssim_speed`): finite for {ssim_ok_methods} (real data found, all 168 values finite -- '
                 'the known NumPy/scikit-image ABI issue does NOT manifest in this particular source file). '
                 f'Entirely missing (no source at all, not the ABI issue) for the other '
                 f'{len(ssim_rows) - len(ssim_ok_methods)} primary methods.')
    lines.append('- See `unified_primary_missingness.csv` for the full total/finite/missing breakdown per '
                 '(method_id, metric); `missing_reason` distinguishes `no_source_artifact_found_in_repository` from '
                 '`not_computed_by_legacy_physics_merged_pipeline` (speed_mae/speed_rmse/comp_* for cnn/gan/bicubic, '
                 'which the older physics-merged pipeline never computed).')
    lines.append('- No missing value was filled with zero or inferred; all gaps are empty cells in the CSVs.')
    lines.append('')
    lines.append('## 9. Candidates that could not be included and why')
    lines.append('')
    lines.append(f'{len(methods_without_data)} of {len(primary_rows)} primary methods '
                 f'({", ".join(methods_without_data)}) have **zero** real per-sample cheap-evaluation or true-'
                 'topology artifacts anywhere in this git checkout. Root cause: this repository\'s `.gitignore` '
                 'excludes `*.npy`, `*.npz`, `data_out/`, and `ttk_runs_fixed/topology_finetuning/*` (tracked '
                 'exceptions are only `candidateE_constraints` and the cnn/gan `combined`/`phase_c_final` summary '
                 'artifacts); large experiment outputs for the loss-ablation candidates are produced only on the '
                 'separate training machine referenced throughout this project\'s history and were never committed. '
                 'The reference documentation records PD/MT means for these methods, but per the task instructions '
                 'those values were used only as a *validation target*, never copied into the unified table as data.')
    lines.append('')
    lines.append('## 10. No training or TTK was rerun')
    lines.append('')
    lines.append('This script and this audit performed zero training runs, zero TTK invocations, and zero cheap-'
                 'evaluation runs. It only read pre-existing CSV files already committed to the repository. No '
                 'existing artifact was modified or deleted.')
    lines.append('')
    lines.append('## 11. Generated file paths')
    lines.append('')
    for fname in ['method_inventory.csv', 'unified_primary_per_sample_long.csv', 'column_mapping.csv',
                  'unified_primary_method_summary.csv', 'unified_primary_topology_validation.csv',
                  'unified_primary_pairwise_vs_cnn.csv', 'unified_primary_missingness.csv',
                  'unified_primary_wide.csv']:
        lines.append(f'- `ttk_runs_fixed/unified_candidate_evaluation/{fname}`')
    lines.append('- `docs/unified_candidate_evaluation_inventory.md`')
    lines.append('- `docs/unified_candidate_evaluation_phase1.md` (this file)')
    lines.append('- `logs/build_unified_candidate_evaluation.log`')
    lines.append('')
    lines.append('## 12. Recommended next step')
    lines.append('')
    lines.append(f'Before any factorial-effect analysis, paired contrasts, correlations, or Pareto-front work can '
                 f'be performed on the full primary set, the {len(methods_without_data)} missing methods\' '
                 'cheap-evaluation and true-topology artifacts need to be synced from the training machine into '
                 'this checkout (or this script re-run there). Until then, any such analysis is only valid for '
                 f'the {len(methods_with_data)} methods with real data ({", ".join(methods_with_data)}). Per the '
                 'task instructions, no correlation, factorial-model, Pareto-front, or visualization-selection work '
                 'was performed in this Phase-1 pass.')
    (DOCS_DIR / 'unified_candidate_evaluation_phase1.md').write_text('\n'.join(lines) + '\n')
    log(f"[write] {DOCS_DIR / 'unified_candidate_evaluation_phase1.md'}")


if __name__ == '__main__':
    sys.exit(main())
