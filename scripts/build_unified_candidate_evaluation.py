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
    docs/primary_candidate_artifact_reference.md

Two run modes
-------------
--strict-primary (default): the authoritative Spark-machine mode. Every
primary method (bicubic included) must have complete, cross-validated data
or the script prints exactly which methods/criteria failed and exits
nonzero WITHOUT writing the six `unified_primary_*` tables -- strict mode
never produces a table that looks complete but secretly contains empty
placeholder rows. The inventory, column-mapping, and reference docs are
still written in this case since they are diagnostic, not a completeness
claim.

--audit-allow-missing: the permissive mode used for auditing an incomplete
checkout (e.g. this lightweight sandbox clone). Builds the full method x
sample grid with empty cells for anything genuinely missing, and always
exits 0 unless a genuine data-integrity problem (duplicate keys, corrupt
values, ambiguous file resolution) is hit.

Design principle -- "hard-fail on corruption, report on absence"
------------------------------------------------------------------
This script hard-fails (raises SystemExit) on internal inconsistencies that
would silently corrupt its own output invariants: duplicate
(method_id, sample_idx) keys inside a single source file, a non-finite or
negative topology distance in a file that WAS found, repeated
bicubic/cnn/gan baseline rows that disagree beyond a small numeric
tolerance across candidate cheap-evaluation CSVs, or more than one
ambiguous fallback-discovered source file for the same method.

Total ABSENCE of an artifact (the file/directory simply does not exist)
is not a crash-worthy bug by itself -- in --audit-allow-missing mode it is
recorded and reported; in --strict-primary mode (the default) it is a
reason the run exits nonzero, but the run still finishes and reports
exactly what is missing rather than crashing uninformatively.

Do not rerun training, cheap evaluation, or TTK. Do not delete or modify
any existing artifact. Do not manufacture missing numeric values (no
zero-fill, no interpolation, no copying values from the reference markdown
docs -- the docs are a roadmap only, never a numeric source).
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
os.chdir(REPO_ROOT)

OUT_DIR = REPO_ROOT / 'ttk_runs_fixed' / 'unified_candidate_evaluation'
DOCS_DIR = REPO_ROOT / 'docs'
LOG_PATH = REPO_ROOT / 'logs' / 'build_unified_candidate_evaluation.log'

N_EVAL = 168
PD_MT_TOLERANCE = 1e-4
# Repeated bicubic/cnn/gan rows across different candidate all_sample_metrics
# CSVs are recomputed from the SAME fixed dataGT/dataSR/dataIN arrays each
# time evaluate_finetune_candidate.py runs, so they should be identical up to
# floating-point roundoff -- a tight tolerance is appropriate here.
BASELINE_CROSS_SOURCE_TOLERANCE = 1e-6
# Comparing the harvested (new-schema) cnn/gan rows against the older,
# independently-implemented psnr_topology_physics_merged.csv pipeline is a
# cross-pipeline check, not a same-computation repeat -- consistent with the
# +/-1e-3 threshold already used elsewhere in this repo as a non-fatal
# warning bound (scripts/evaluate_finetune_candidate.py), promoted here to a
# hard-fail bound per this task's explicit request.
OLDER_SOURCE_CROSS_CHECK_TOLERANCE = 1e-3

BASELINE_METHODS = ('bicubic', 'cnn', 'gan')

# -----------------------------------------------------------------------
# Raw benchmark/array validation (requirement 2 of the second patch).
# -----------------------------------------------------------------------
CANONICAL_CNN_DIR = 'data_out_fixed/wind_mrhr_cnn'
EXPECTED_IDX_SHAPE = (N_EVAL,)
EXPECTED_IN_SHAPE = (N_EVAL, 100, 100, 2)
EXPECTED_HR_SHAPE = (N_EVAL, 500, 500, 2)
# Compare/scan large (N_EVAL, 500, 500, 2) arrays in sample-axis chunks so a
# full candidate GT/SR array is never materialized in memory at once.
ARRAY_COMPARE_CHUNK_SAMPLES = 16

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


def _worst_numeric_diff(max_diff: dict):
    """max_diff values may legitimately be None (e.g. SSIM availability
    mismatch/both-unavailable, where no numeric comparison happened) --
    never let max() choke on comparing None to a float, and never report a
    None entry as if it were the worst (or a zero) numeric difference."""
    numeric = {k: v for k, v in max_diff.items() if v is not None}
    if not numeric:
        return None, 0.0
    worst_metric = max(numeric, key=numeric.get)
    return worst_metric, numeric[worst_metric]


def _relpath(p) -> str:
    if not p:
        return ''
    try:
        return str(Path(p).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


# =============================================================================
# Metric-column schema (standardized names, direction, representation).
# =============================================================================
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
TOPOLOGY_METRIC_COLUMNS = ['pd_distance', 'mt_distance']
CHEAP_METRIC_COLUMNS = [c for c in METRIC_COLUMNS if c not in TOPOLOGY_METRIC_COLUMNS]
# ssim_speed is documented to be legitimately, globally NaN across an entire
# evaluation run because of the known NumPy/scikit-image ABI incompatibility
# -- that is not a data-quality bug and must not block strict validation.
# Every other cheap metric has no such known-benign global failure mode and
# stays required.
OPTIONAL_CHEAP_METRIC_COLUMNS = {'ssim_speed'}
REQUIRED_CHEAP_METRIC_COLUMNS = [c for c in CHEAP_METRIC_COLUMNS if c not in OPTIONAL_CHEAP_METRIC_COLUMNS]

IDENTITY_COLUMNS = [
    'sample_idx', 'method_id', 'display_name', 'candidate_family',
    'training_scale', 'architecture',
    'uses_speed', 'uses_grad', 'uses_levelset', 'uses_crit', 'uses_e2',
]

# Columns present in ttk_runs_fixed/combined/psnr_topology_physics_merged.csv
# (the legacy cnn/gan-only "physics_merged" pipeline) mapped to the standardized
# schema. Predates scripts/evaluate_finetune_candidate.py; does NOT compute
# speed_mae/speed_rmse/comp_* -- those are left genuinely missing here unless
# the newer harvested baseline rows (see harvest_baseline_rows()) supply them.
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
BASELINE_COMBINED_NOT_AVAILABLE = [
    'speed_mae', 'speed_rmse', 'comp_curve_l1',
    'comp_abs_t5', 'comp_abs_t10', 'comp_abs_t15',
]

# Columns scripts/evaluate_finetune_candidate.py writes for bicubic/cnn/gan/
# the candidate in every all_sample_metrics_<name>.csv, mapped to the
# standardized schema. This is both the schema used to harvest repeated
# baseline rows (requirement 1) and the schema for a genuine new candidate.
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
                      objective_summary, repaired_status='not_applicable', historical_alias=''):
    return dict(
        method_id=method_id, display_name=display_name,
        original_method_name=original_method_name, candidate_family=family,
        comparison_tier='primary_2688_native_tf', include_primary=True,
        training_scale='2688', architecture='native_tf',
        training_tfrecord=TFREC_TRAIN_2688, evaluation_tfrecord=TFREC_EVAL_FIXED,
        objective_summary=objective_summary,
        uses_speed=uses_speed, uses_grad=uses_grad, uses_levelset=uses_levelset,
        uses_crit=uses_crit, uses_e2=uses_e2, repaired_status=repaired_status,
        historical_alias=historical_alias,
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
         repaired_status='baseline', historical_alias='',
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
         repaired_status='baseline', historical_alias='',
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
         repaired_status='baseline', historical_alias='',
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
                      'L_uv + 0.001 L_crit (Candidate C minus the Candidate B scaffold).',
                      historical_alias='C-(B scaffold): "Candidate C minus the Candidate B scaffold"'),
    _native_tf_entry('uv_e2', 'UV + repaired E2', 'candidateUV_plus_E2_tf_lowlambda_expanded2688',
                      'E2_uv', False, False, False, False, True,
                      'L_uv + 0.004 L_TTKCV + 0.002 L_TTKpers (repaired low-lambda E2, no B/C scaffold).',
                      repaired_status='repaired_e2',
                      historical_alias='E-(C+B); also referenced as "UV+E2-low"'),
    _native_tf_entry('b_e2', 'Candidate B + repaired E2', 'candidateB_plus_E2_tf_lowlambda_expanded2688',
                      'E2_b', True, True, True, False, True,
                      'L_uv + 0.01 L_speed + 0.05 L_grad + 0.25 L_levelset + 0.004 L_TTKCV + 0.002 L_TTKpers '
                      '(L_crit disabled, lambda_crit=0).',
                      repaired_status='repaired_e2',
                      historical_alias='E-C; also referenced as "TF B+E2-low"'),
    _native_tf_entry('c_e2', 'Candidate C + repaired E2', 'candidateE2_tf_lowlambda_expanded2688',
                      'E2_c', True, True, True, True, True,
                      'L_uv + 0.01 L_speed + 0.05 L_grad + 0.25 L_levelset + 0.001 L_crit '
                      '+ 0.004 L_TTKCV + 0.002 L_TTKpers.',
                      repaired_status='repaired_e2',
                      historical_alias='E; also referenced as "TF C+E2-low"'),
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
        repaired_status=repaired_status, historical_alias='',
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
         repaired_status='n/a', historical_alias='',
         data_out_dir='n/a', model_dir='n/a',
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
         repaired_status='n/a', historical_alias='',
         data_out_dir='n/a', model_dir='n/a',
         cheap_eval_dir='', topology_dir='',
         cheap_report='', topology_report='',
         exclusion_reason='archive/old_ttk_outputs/phase_c_results.csv uses a different, non-distance schema '
                           'and covers GAN only; not comparable to the primary table.'),
]

FULL_MANIFEST = PRIMARY_MANIFEST + SECONDARY_MANIFEST


# =============================================================================
# CSV helpers
# =============================================================================

def _read_csv_rows(path: Path):
    with path.open(newline='') as fh:
        return list(csv.DictReader(fh))


# =============================================================================
# Requirement 1: harvest repeated bicubic/cnn/gan rows from every candidate
# all_sample_metrics_<name>.csv found anywhere in the tree.
# =============================================================================

def discover_candidate_eval_csvs() -> list:
    root = rp('ttk_runs_fixed/topology_finetuning')
    if not root.is_dir():
        return []
    found = []
    for eval_dir in sorted(root.glob('*_eval')):
        if not eval_dir.is_dir():
            continue
        for csv_path in sorted(eval_dir.glob('all_sample_metrics_*.csv')):
            found.append(csv_path)
    return sorted(found, key=str)


def _finite_count(rec_by_si: dict, col: str) -> int:
    return sum(1 for si in range(N_EVAL) if math.isfinite(rec_by_si.get(si, {}).get(col, float('nan'))))


def harvest_baseline_rows(csv_paths: list):
    """Extract bicubic/cnn/gan rows from every discovered candidate
    all_sample_metrics CSV, validate each source, cross-compare sources, and
    select canonical baseline data PER METRIC rather than one whole file per
    baseline method (patch requirement: 'small baseline-coverage correction').

    Required cheap metrics (REQUIRED_CHEAP_METRIC_COLUMNS): every source must
    have exactly 168/168 finite values or the whole run hard-fails -- a
    source with 0-167 finite values for a required metric is a data-
    integrity problem, not something to silently work around by picking a
    different source. Once every source passes this gate, required-metric
    values are compared pairwise across sources (tolerance
    BASELINE_CROSS_SOURCE_TOLERANCE) and the lexicographically-first source
    is used as the canonical value (they are validated equal anyway).

    SSIM (ssim_speed) is handled differently: each source must independently
    be 168/168 or 0/168 finite (partial coverage within one source still
    hard-fails), but 168/168-in-one-source-and-0/168-in-another across
    different sources is NOT a disagreement -- it is recorded as
    'mixed_global_availability', and the canonical SSIM value is taken from
    the lexicographically-first fully-finite source when one exists (so a
    real SSIM value is never discarded just because another, unrelated
    source happens to sort first), or preserved as all-NaN when no source
    has it. Any two fully-finite SSIM sources must still numerically agree.

    Returns (baseline_data, report) where baseline_data is
    {baseline_method: {sample_idx: {standardized_col: value}}}.
    """
    per_source: dict = {}
    ssim_status_by_source: dict = {}  # (path, bm) -> ('full'|'empty', finite_count)
    for path in csv_paths:
        rows = _read_csv_rows(path)
        by_method: dict = {}
        for row in rows:
            by_method.setdefault(row.get('method', ''), []).append(row)
        per_source[path] = {}
        for bm in BASELINE_METHODS:
            method_rows = by_method.get(bm, [])
            if not method_rows:
                continue
            seen: dict = {}
            dup = []
            for row in method_rows:
                si = int(row['sample_idx'])
                if si in seen:
                    dup.append(si)
                    continue
                seen[si] = row
            if dup:
                raise SystemExit(
                    f'[hard-fail] Duplicate sample_idx for baseline={bm!r} in {path}: {sorted(set(dup))}.'
                )
            idx_set = set(seen.keys())
            if len(method_rows) != N_EVAL or idx_set != set(range(N_EVAL)):
                raise SystemExit(
                    f'[hard-fail] Baseline {bm!r} rows in {path} are not exactly {N_EVAL} rows with '
                    f'sample_idx exactly 0..{N_EVAL - 1} (found {len(method_rows)} rows, '
                    f'{len(idx_set)} unique indices).'
                )
            rec_by_si = {}
            for si, row in seen.items():
                rec = {}
                for src_col, std_col in CANDIDATE_EVAL_DIRECT_MAP.items():
                    val = row.get(src_col, '')
                    rec[std_col] = float(val) if val not in ('', None) else float('nan')
                rec_by_si[si] = rec

            # Requirement 1: every required cheap metric must be exactly 168/168
            # finite in THIS source, regardless of what other sources look like.
            for col in REQUIRED_CHEAP_METRIC_COLUMNS:
                finite = _finite_count(rec_by_si, col)
                if finite != N_EVAL:
                    raise SystemExit(
                        f'[hard-fail] Baseline {bm!r} required cheap metric {col!r} in {path} has '
                        f'{finite}/{N_EVAL} finite values (must be exactly {N_EVAL}). A required metric '
                        'being partially or fully missing indicates a corrupted or incomplete evaluation '
                        'source, even if another source is complete.'
                    )

            # Requirement 2: SSIM must independently be 168/168 or 0/168 in THIS source.
            ssim_finite = _finite_count(rec_by_si, 'ssim_speed')
            if ssim_finite not in (0, N_EVAL):
                raise SystemExit(
                    f'[hard-fail] Baseline {bm!r} ssim_speed in {path} has {ssim_finite}/{N_EVAL} finite '
                    f'values -- must be either 0/{N_EVAL} (globally unavailable) or {N_EVAL}/{N_EVAL} '
                    '(fully available); partial coverage indicates inconsistent evaluation coverage.'
                )

            per_source[path][bm] = rec_by_si
            ssim_status_by_source[(path, bm)] = ('full' if ssim_finite == N_EVAL else 'empty', ssim_finite)

    cross_report: dict = {}
    for bm in BASELINE_METHODS:
        sources_with_bm = sorted([p for p in csv_paths if bm in per_source.get(p, {})], key=str)
        cross_report[bm] = dict(n_sources=len(sources_with_bm),
                                 sources=[str(p) for p in sources_with_bm],
                                 max_diff_per_metric={}, worst_pair_per_metric={},
                                 ssim_coverage={str(p): ssim_status_by_source[(p, bm)] for p in sources_with_bm},
                                 ssim_availability='n/a', ssim_canonical_source='')
        if not sources_with_bm:
            continue
        full_sources = [p for p in sources_with_bm if ssim_status_by_source[(p, bm)][0] == 'full']
        empty_sources = [p for p in sources_with_bm if ssim_status_by_source[(p, bm)][0] == 'empty']
        if full_sources and empty_sources:
            cross_report[bm]['ssim_availability'] = 'mixed_global_availability'
        elif full_sources:
            cross_report[bm]['ssim_availability'] = 'full'
        elif empty_sources:
            cross_report[bm]['ssim_availability'] = 'unavailable'
        cross_report[bm]['ssim_canonical_source'] = str(full_sources[0]) if full_sources else ''

        if len(sources_with_bm) < 2:
            continue

        # Required cheap metrics: by this point every source is validated
        # exactly 168/168 finite, so pairwise comparison needs no NaN-skip.
        for col in REQUIRED_CHEAP_METRIC_COLUMNS:
            overall_max = 0.0
            worst_pair = None
            for i in range(len(sources_with_bm)):
                for j in range(i + 1, len(sources_with_bm)):
                    pi, pj = sources_with_bm[i], sources_with_bm[j]
                    pair_max = 0.0
                    for si in range(N_EVAL):
                        a = per_source[pi][bm][si][col]
                        b = per_source[pj][bm][si][col]
                        pair_max = max(pair_max, abs(a - b))
                    if pair_max > overall_max:
                        overall_max = pair_max
                        worst_pair = (str(pi), str(pj))
            cross_report[bm]['max_diff_per_metric'][col] = overall_max
            cross_report[bm]['worst_pair_per_metric'][col] = worst_pair
            if overall_max > BASELINE_CROSS_SOURCE_TOLERANCE:
                raise SystemExit(
                    f'[hard-fail] Repeated baseline {bm!r} rows for required metric {col!r} disagree by '
                    f'{overall_max:.6g} (> tolerance {BASELINE_CROSS_SOURCE_TOLERANCE:g}) between '
                    f'{worst_pair[0]} and {worst_pair[1]}. Repeated baseline rows must be identical '
                    'across every candidate evaluation run.'
                )

        # SSIM: only fully-finite sources can be numerically compared; a
        # full-vs-empty pairing is availability, not disagreement.
        if len(full_sources) >= 2:
            overall_max = 0.0
            worst_pair = None
            for i in range(len(full_sources)):
                for j in range(i + 1, len(full_sources)):
                    pi, pj = full_sources[i], full_sources[j]
                    pair_max = 0.0
                    for si in range(N_EVAL):
                        a = per_source[pi][bm][si]['ssim_speed']
                        b = per_source[pj][bm][si]['ssim_speed']
                        pair_max = max(pair_max, abs(a - b))
                    if pair_max > overall_max:
                        overall_max = pair_max
                        worst_pair = (str(pi), str(pj))
            cross_report[bm]['max_diff_per_metric']['ssim_speed'] = overall_max
            cross_report[bm]['worst_pair_per_metric']['ssim_speed'] = worst_pair
            if overall_max > BASELINE_CROSS_SOURCE_TOLERANCE:
                raise SystemExit(
                    f'[hard-fail] Repeated baseline {bm!r} ssim_speed disagrees by {overall_max:.6g} '
                    f'(> tolerance {BASELINE_CROSS_SOURCE_TOLERANCE:g}) between fully-finite sources '
                    f'{worst_pair[0]} and {worst_pair[1]}.'
                )

    # Per-metric canonical selection (never one whole file for every metric):
    # required metrics from the lexicographically-first source (all sources
    # are validated equal for these); SSIM from the lexicographically-first
    # FULLY-FINITE source when one exists, else preserved as all-NaN.
    canonical_required = {}
    canonical_ssim = {}
    baseline_data = {}
    for bm in BASELINE_METHODS:
        sources_with_bm = sorted([p for p in csv_paths if bm in per_source.get(p, {})], key=str)
        if not sources_with_bm:
            baseline_data[bm] = {}
            canonical_required[bm] = None
            canonical_ssim[bm] = None
            continue

        chosen_required = sources_with_bm[0]
        canonical_required[bm] = chosen_required
        full_sources = [p for p in sources_with_bm if ssim_status_by_source[(p, bm)][0] == 'full']
        chosen_ssim = full_sources[0] if full_sources else None
        canonical_ssim[bm] = chosen_ssim

        per_sample = {}
        for si in range(N_EVAL):
            rec = dict(per_source[chosen_required][bm][si])
            rec['ssim_speed'] = per_source[chosen_ssim][bm][si]['ssim_speed'] if chosen_ssim is not None else float('nan')
            per_sample[si] = rec
        baseline_data[bm] = per_sample

    report = dict(
        discovered_csvs=[str(p) for p in csv_paths],
        cross_report=cross_report,
        canonical={bm: (str(canonical_required[bm]) if canonical_required[bm] else '') for bm in BASELINE_METHODS},
        canonical_ssim={bm: (str(canonical_ssim[bm]) if canonical_ssim.get(bm) else '') for bm in BASELINE_METHODS},
    )
    return baseline_data, report


# =============================================================================
# Legacy combined-CSV loader (cnn/gan only) -- remains the topology (PD/MT)
# source of record; also a cheap-metric fallback when no harvested rows exist.
# =============================================================================

def load_legacy_baseline_data():
    combined_path = rp('ttk_runs_fixed/combined/psnr_topology_physics_merged.csv')
    phase_c_path = rp('ttk_runs_fixed/combined/phase_c_results.csv')
    report = {'combined_path': str(combined_path), 'phase_c_path': str(phase_c_path), 'per_method': {}}

    data = {}
    if not combined_path.exists():
        log(f'[legacy-baseline] MISSING: {combined_path}')
        return data, report

    rows = _read_csv_rows(combined_path)
    by_method: dict = {}
    for row in rows:
        by_method.setdefault(row['method'], []).append(row)

    phase_c_rows = _read_csv_rows(phase_c_path) if phase_c_path.exists() else []
    phase_c_by_method: dict = {}
    for row in phase_c_rows:
        m = re.search(r'_s(\d+)_', row['key'])
        if not m:
            continue
        phase_c_by_method.setdefault(row['method'], {})[int(m.group(1))] = (
            float(row['pd_distance']), float(row['mt_distance']))

    for method in ('cnn', 'gan'):
        method_rows = by_method.get(method, [])
        seen_idx: dict = {}
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
                f'{sorted(set(dup_keys))}.'
            )

        idx_set = set(per_sample.keys())
        expected_set = set(range(N_EVAL))
        sample_index_status = 'exact_0_167' if idx_set == expected_set else (
            f'MISMATCH missing={sorted(expected_set - idx_set)[:10]} '
            f'extra={sorted(idx_set - expected_set)[:10]}')

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
# Requirement 2: cross-check harvested (new-schema) cnn/gan rows against the
# older combined pipeline for overlapping columns.
# =============================================================================

def cross_check_baseline_vs_legacy(harvested_baseline_data: dict, legacy_baseline_data: dict) -> dict:
    """Cross-check canonical harvested cnn/gan rows against the older
    psnr_topology_physics_merged.csv pipeline for overlapping columns.

    Required overlap columns: both sides must have exactly 168/168 finite
    values before comparing -- if either side has 0-167, that is a hard
    failure, not something to silently skip.

    SSIM: compared numerically only when BOTH sides have 168/168 finite. A
    168/168-vs-0/168 split is reported as an availability mismatch, not a
    zero-difference "pass" -- max_diff['ssim_speed'] is left as None (never
    fabricated) in that case and in the both-unavailable case, since no
    numeric comparison actually happened.
    """
    overlap_cols = sorted(set(BASELINE_COMBINED_DIRECT_MAP.values()) | {'wpd_bias_abs'})
    required_overlap_cols = [c for c in overlap_cols if c != 'ssim_speed']
    report = {}
    for method in ('cnn', 'gan'):
        harvested = harvested_baseline_data.get(method, {})
        legacy = legacy_baseline_data.get(method, {})
        if not harvested or not legacy:
            report[method] = dict(skipped=True,
                                   reason='harvested candidate-eval rows not found for this method in this checkout'
                                   if not harvested else 'legacy combined source not found',
                                   max_diff={}, ssim_status='n/a')
            continue

        max_diff = {}
        for col in required_overlap_cols:
            h_finite = _finite_count(harvested, col)
            l_finite = _finite_count(legacy, col)
            if h_finite != N_EVAL or l_finite != N_EVAL:
                raise SystemExit(
                    f'[hard-fail] {method} overlapping required metric {col!r}: harvested has '
                    f'{h_finite}/{N_EVAL} finite values, legacy has {l_finite}/{N_EVAL} -- both must be '
                    f'exactly {N_EVAL}/{N_EVAL} before they can be compared.'
                )
            d = max(abs(harvested[si][col] - legacy[si][col]) for si in range(N_EVAL))
            max_diff[col] = d
            if d > OLDER_SOURCE_CROSS_CHECK_TOLERANCE:
                raise SystemExit(
                    f'[hard-fail] {method} metric {col!r} disagrees between the harvested candidate-eval '
                    f'baseline rows and the legacy combined source ({d:.6g} > tolerance '
                    f'{OLDER_SOURCE_CROSS_CHECK_TOLERANCE:g}).'
                )

        h_ssim_finite = _finite_count(harvested, 'ssim_speed')
        l_ssim_finite = _finite_count(legacy, 'ssim_speed')
        if h_ssim_finite == N_EVAL and l_ssim_finite == N_EVAL:
            d = max(abs(harvested[si]['ssim_speed'] - legacy[si]['ssim_speed']) for si in range(N_EVAL))
            max_diff['ssim_speed'] = d
            ssim_status = 'compared'
            if d > OLDER_SOURCE_CROSS_CHECK_TOLERANCE:
                raise SystemExit(
                    f'[hard-fail] {method} ssim_speed disagrees between the harvested candidate-eval '
                    f'baseline rows and the legacy combined source ({d:.6g} > tolerance '
                    f'{OLDER_SOURCE_CROSS_CHECK_TOLERANCE:g}).'
                )
        elif h_ssim_finite == 0 and l_ssim_finite == 0:
            ssim_status = 'both_unavailable'
            max_diff['ssim_speed'] = None
        elif h_ssim_finite in (0, N_EVAL) and l_ssim_finite in (0, N_EVAL):
            ssim_status = 'availability_mismatch'
            max_diff['ssim_speed'] = None
        else:
            # Partial coverage on either side should already be impossible
            # (harvested is gated by requirement 2's per-source hard-fail;
            # legacy is loaded as either fully complete or fully empty per
            # column by construction) -- defensive hard-fail if it ever occurs.
            raise SystemExit(
                f'[hard-fail] {method} ssim_speed has unexpected partial coverage: harvested='
                f'{h_ssim_finite}/{N_EVAL}, legacy={l_ssim_finite}/{N_EVAL}.'
            )
        report[method] = dict(skipped=False, max_diff=max_diff, ssim_status=ssim_status)
    return report


def build_baseline_per_sample(harvested_baseline_data: dict, legacy_baseline_data: dict) -> dict:
    """Merge harvested + legacy baseline sources: PD/MT stay sourced from the
    legacy/phase_c cross-validated pipeline (per requirement 2, it remains
    the topology source of record); every other (cheap) metric prefers the
    harvested candidate-eval rows and falls back to the legacy subset."""
    merged = {}
    for method in BASELINE_METHODS:
        harvested = harvested_baseline_data.get(method, {})
        legacy = legacy_baseline_data.get(method, {})
        per_sample = {}
        for si in set(harvested) | set(legacy):
            legacy_rec = legacy.get(si, {})
            harvested_rec = harvested.get(si, {})
            rec = {}
            for col in METRIC_COLUMNS:
                if col in TOPOLOGY_METRIC_COLUMNS:
                    v = legacy_rec.get(col, harvested_rec.get(col, float('nan')))
                else:
                    v = harvested_rec.get(col, legacy_rec.get(col, float('nan')))
                rec[col] = v
            per_sample[si] = rec
        merged[method] = per_sample
    return merged


# =============================================================================
# Requirement 4: path-discovery fallback.
# =============================================================================

FALLBACK_SEARCH_ROOTS = ['ttk_runs_fixed/topology_finetuning', 'data_out', 'docs']


def resolve_with_fallback(exact_path, expected_basename: str):
    """Try the exact manifest path first; if absent, search the fallback
    roots for a file with the exact expected basename. Accept the fallback
    only if exactly one match is found; hard-fail on ambiguity."""
    if exact_path is not None and exact_path.exists():
        return exact_path, 'exact_path'
    candidates = []
    for root in FALLBACK_SEARCH_ROOTS:
        root_path = rp(root)
        if not root_path.is_dir():
            continue
        candidates.extend(p for p in root_path.rglob(expected_basename) if p.is_file())
    candidates = sorted(set(candidates), key=str)
    if not candidates:
        return None, 'not_found'
    if len(candidates) > 1:
        raise SystemExit(
            f'[hard-fail] Ambiguous fallback discovery for {expected_basename!r}: found '
            f'{len(candidates)} candidates under {FALLBACK_SEARCH_ROOTS}: {[str(c) for c in candidates]}. '
            'Refusing to guess which one is authoritative.'
        )
    return candidates[0], 'fallback_discovery'


# =============================================================================
# Generic candidate loader (path-discovery fallback + full schema load).
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
    exact_cheap = cheap_eval_dir / f'all_sample_metrics_{name}.csv' if cheap_eval_dir else None
    exact_pairwise = cheap_eval_dir / f'pairwise_cnn_vs_{name}.csv' if cheap_eval_dir else None
    cheap_csv, cheap_resolution = resolve_with_fallback(exact_cheap, f'all_sample_metrics_{name}.csv')
    pairwise_csv, pairwise_resolution = resolve_with_fallback(exact_pairwise, f'pairwise_cnn_vs_{name}.csv')
    res['cheap_eval_csv'] = str(cheap_csv) if cheap_csv else ''
    res['cheap_pairwise_csv'] = str(pairwise_csv) if pairwise_csv else ''
    res['cheap_eval_csv_exists'] = bool(cheap_csv)
    res['cheap_pairwise_csv_exists'] = bool(pairwise_csv)
    res['cheap_eval_resolution'] = cheap_resolution
    res['cheap_pairwise_resolution'] = pairwise_resolution

    topology_dir = rp(entry['topology_dir']) if entry.get('topology_dir') else None
    exact_pd_mt = topology_dir / f'{name}_pd_mt_distances.csv' if topology_dir else None
    exact_topo_cmp = topology_dir / f'{name}_topology_comparison.csv' if topology_dir else None
    pd_mt_csv, topo_resolution = resolve_with_fallback(exact_pd_mt, f'{name}_pd_mt_distances.csv')
    topo_cmp_csv, topo_cmp_resolution = resolve_with_fallback(exact_topo_cmp, f'{name}_topology_comparison.csv')
    res['topology_results_csv'] = str(pd_mt_csv) if pd_mt_csv else ''
    res['topology_comparison_csv'] = str(topo_cmp_csv) if topo_cmp_csv else ''
    res['topology_results_csv_exists'] = bool(pd_mt_csv)
    res['topology_comparison_csv_exists'] = bool(topo_cmp_csv)
    res['topology_results_resolution'] = topo_resolution
    res['topology_comparison_resolution'] = topo_cmp_resolution

    res['cheap_report_exists'] = bool(entry.get('cheap_report') and rp(entry['cheap_report']).exists())
    res['topology_report_exists'] = bool(entry.get('topology_report') and rp(entry['topology_report']).exists())

    per_sample: dict = {}
    row_count_cheap = 0
    row_count_topology = 0

    if cheap_csv:
        rows = _read_csv_rows(cheap_csv)
        cand_rows = [r for r in rows if r.get('method') == name]
        row_count_cheap = len(cand_rows)
        seen = set()
        for row in cand_rows:
            si = int(row['sample_idx'])
            if si in seen:
                raise SystemExit(
                    f'[hard-fail] Duplicate sample_idx={si} for method={name!r} in {cheap_csv}.'
                )
            seen.add(si)
            rec = per_sample.setdefault(si, {})
            for src_col, std_col in CANDIDATE_EVAL_DIRECT_MAP.items():
                val = row.get(src_col, '')
                rec[std_col] = float(val) if val not in ('', None) else float('nan')

    if pd_mt_csv:
        rows = _read_csv_rows(pd_mt_csv)
        row_count_topology = len(rows)
        seen = set()
        for row in rows:
            si = int(row.get('sample_idx', row.get('sample', -1)))
            if si in seen:
                raise SystemExit(f'[hard-fail] Duplicate sample_idx={si} for method={name!r} in {pd_mt_csv}.')
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
# SSIM optional-but-audited classification (patch requirement 1).
# =============================================================================

def classify_ssim_status(per_sample: dict):
    """Returns (status, n_finite) where status is one of:
      'full'                        -- 168/168 finite (fully available)
      'unavailable_global_dependency' -- 0/168 finite (documented ABI issue or
                                          simply never computed for this method)
      'partial_source_coverage'     -- 1..167/168 finite (inconsistent
                                        evaluation coverage -- always a strict
                                        failure, never accepted)
      'no_data'                     -- per_sample itself is empty
    """
    if not per_sample:
        return 'no_data', 0
    n_finite = sum(1 for si in range(N_EVAL)
                   if math.isfinite(per_sample.get(si, {}).get('ssim_speed', float('nan'))))
    if n_finite == N_EVAL:
        return 'full', n_finite
    if n_finite == 0:
        return 'unavailable_global_dependency', n_finite
    return 'partial_source_coverage', n_finite


def classify_missing_reason(col: str, per_sample: dict, finite: int, total: int) -> str:
    """Three-way missingness classification shared by the missingness table
    and the Phase-1 report (patch requirement 1)."""
    if not per_sample:
        return 'no_source_artifact'
    if finite == total:
        return ''
    if col == 'ssim_speed':
        return 'unavailable_global_dependency' if finite == 0 else 'partial_source_coverage'
    if finite == 0:
        return 'no_source_artifact'
    return 'partial_source_coverage'


# =============================================================================
# Raw benchmark/array validation (patch requirement 2).
# =============================================================================

def _load_npy_mmap(path: Path):
    return np.load(str(path), mmap_mode='r', allow_pickle=False)


def _arrays_exact_equal_chunked(a, b, chunk: int = ARRAY_COMPARE_CHUNK_SAMPLES):
    """Compare two (N, ...) array-likes (e.g. memory-mapped) sample-chunk by
    sample-chunk so neither is ever fully materialized in memory at once.
    Returns (exact: bool, max_abs_diff: float)."""
    if a.shape != b.shape:
        return False, float('inf')
    n = a.shape[0]
    exact = True
    max_diff = 0.0
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        ca = np.asarray(a[start:end])
        cb = np.asarray(b[start:end])
        if not np.array_equal(ca, cb):
            exact = False
            max_diff = max(max_diff, float(np.max(np.abs(ca.astype(np.float64) - cb.astype(np.float64)))))
    return exact, max_diff


def _array_all_finite_chunked(a, chunk: int = ARRAY_COMPARE_CHUNK_SAMPLES) -> bool:
    n = a.shape[0]
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        c = np.asarray(a[start:end])
        if not np.isfinite(c).all():
            return False
    return True


def validate_raw_arrays(data_out_dir: Path, canonical_in, canonical_gt, require_idx: bool = True):
    """Validate idx/dataIN/dataGT/dataSR .npy files under data_out_dir against
    the expected shapes and the canonical CNN benchmark arrays (when supplied).
    Never writes or modifies any file. Returns (status_dict, reasons_list).

    status_dict keys: idx_validation_status, input_alignment_status,
    gt_alignment_status, sr_shape_status, sr_finiteness_status.
    """
    status = {}
    reasons = []

    idx_path = data_out_dir / 'idx.npy'
    in_path = data_out_dir / 'dataIN.npy'
    gt_path = data_out_dir / 'dataGT.npy'
    sr_path = data_out_dir / 'dataSR.npy'

    # idx.npy
    if not idx_path.exists():
        status['idx_validation_status'] = 'missing' if require_idx else 'not_applicable_no_idx_file'
        if require_idx:
            reasons.append(f'idx.npy missing at {idx_path}')
    else:
        idx = np.asarray(np.load(str(idx_path), allow_pickle=False))
        if idx.shape != EXPECTED_IDX_SHAPE:
            status['idx_validation_status'] = f'bad_shape {idx.shape}'
            reasons.append(f'idx.npy shape {idx.shape} != {EXPECTED_IDX_SHAPE} at {idx_path}')
        else:
            # Never validate through integer truncation: a float array like
            # [0.5, 1.0, 2.0, ...] must NOT pass just because .astype(int64)
            # would truncate 0.5 down to 0. Accept either an integer dtype
            # compared exactly to arange(168), or an exact (uncast) numeric
            # comparison against arange(168) in the array's own dtype.
            if np.issubdtype(idx.dtype, np.integer):
                idx_ok = np.array_equal(idx.astype(np.int64), np.arange(N_EVAL, dtype=np.int64))
            else:
                idx_ok = np.array_equal(idx, np.arange(N_EVAL, dtype=idx.dtype))
            if not idx_ok:
                status['idx_validation_status'] = 'not_exact_0_167'
                reasons.append(f'idx.npy is not exactly ordered 0..{N_EVAL - 1} at {idx_path}')
            else:
                status['idx_validation_status'] = 'exact_0_167'

    # dataIN.npy
    if not in_path.exists():
        status['input_alignment_status'] = 'missing'
        reasons.append(f'dataIN.npy missing at {in_path}')
    else:
        din = _load_npy_mmap(in_path)
        if din.shape != EXPECTED_IN_SHAPE:
            status['input_alignment_status'] = f'bad_shape {din.shape}'
            reasons.append(f'dataIN.npy shape {din.shape} != {EXPECTED_IN_SHAPE} at {in_path}')
        elif canonical_in is None:
            status['input_alignment_status'] = 'no_canonical_reference_to_compare'
            reasons.append(f'cannot validate dataIN.npy alignment at {in_path}: canonical CNN benchmark '
                            f'dataIN.npy not available')
        else:
            exact, max_diff = _arrays_exact_equal_chunked(din, canonical_in)
            status['input_alignment_status'] = 'exact' if exact else f'MISMATCH max_abs_diff={max_diff:.4e}'
            if not exact:
                reasons.append(f'dataIN.npy not exactly aligned with canonical CNN dataIN at {in_path} '
                                f'(max_abs_diff={max_diff:.4e})')

    # dataGT.npy
    if not gt_path.exists():
        status['gt_alignment_status'] = 'missing'
        reasons.append(f'dataGT.npy missing at {gt_path}')
    else:
        dgt = _load_npy_mmap(gt_path)
        if dgt.shape != EXPECTED_HR_SHAPE:
            status['gt_alignment_status'] = f'bad_shape {dgt.shape}'
            reasons.append(f'dataGT.npy shape {dgt.shape} != {EXPECTED_HR_SHAPE} at {gt_path}')
        elif canonical_gt is None:
            status['gt_alignment_status'] = 'no_canonical_reference_to_compare'
            reasons.append(f'cannot validate dataGT.npy alignment at {gt_path}: canonical CNN benchmark '
                            f'dataGT.npy not available')
        else:
            exact, max_diff = _arrays_exact_equal_chunked(dgt, canonical_gt)
            status['gt_alignment_status'] = 'exact' if exact else f'MISMATCH max_abs_diff={max_diff:.4e}'
            if not exact:
                reasons.append(f'dataGT.npy not exactly aligned with canonical CNN dataGT at {gt_path} '
                                f'(max_abs_diff={max_diff:.4e})')

    # dataSR.npy
    if not sr_path.exists():
        status['sr_shape_status'] = 'missing'
        status['sr_finiteness_status'] = 'missing'
        reasons.append(f'dataSR.npy missing at {sr_path}')
    else:
        dsr = _load_npy_mmap(sr_path)
        if dsr.shape != EXPECTED_HR_SHAPE:
            status['sr_shape_status'] = f'bad_shape {dsr.shape}'
            status['sr_finiteness_status'] = 'not_checked_bad_shape'
            reasons.append(f'dataSR.npy shape {dsr.shape} != {EXPECTED_HR_SHAPE} at {sr_path}')
        else:
            status['sr_shape_status'] = 'exact'
            finite_ok = _array_all_finite_chunked(dsr)
            status['sr_finiteness_status'] = 'all_finite' if finite_ok else 'NONFINITE_VALUES_PRESENT'
            if not finite_ok:
                reasons.append(f'dataSR.npy contains non-finite values at {sr_path}')

    return status, reasons


ARRAY_STATUS_FIELDS = ['idx_validation_status', 'input_alignment_status', 'gt_alignment_status',
                        'sr_shape_status', 'sr_finiteness_status']
ARRAY_STATUS_NOT_APPLICABLE = {f: 'not_checked_secondary_tier' for f in ARRAY_STATUS_FIELDS}


# =============================================================================
# Requirement 3: strict-mode per-method completeness check.
# =============================================================================

def check_strict_primary(mid, per_sample, is_bicubic, expected_pd, expected_mt, array_reasons=()):
    # Raw-array validation reasons apply unconditionally -- a method with
    # missing/misaligned/non-finite raw arrays cannot strict-pass even if its
    # CSV-derived cheap/topology metrics look complete.
    reasons = list(array_reasons)

    idx_set = set(per_sample.keys())
    if idx_set != set(range(N_EVAL)):
        reasons.append(f'cheap/topology rows not exactly {N_EVAL} with sample_idx 0..{N_EVAL - 1} '
                        f'(found {len(idx_set)} rows)')
        return (len(reasons) == 0), reasons

    cheap_bad = [c for c in REQUIRED_CHEAP_METRIC_COLUMNS
                 if not all(math.isfinite(per_sample[si].get(c, float('nan'))) for si in range(N_EVAL))]
    if cheap_bad:
        reasons.append(f'required cheap metric column(s) not finite for all {N_EVAL} samples: {cheap_bad}')

    ssim_status, ssim_finite = classify_ssim_status(per_sample)
    if ssim_status == 'partial_source_coverage':
        reasons.append(f'ssim_speed has partial coverage ({ssim_finite}/{N_EVAL} finite) -- must be either '
                        f'0/{N_EVAL} (globally unavailable, e.g. the documented ABI issue) or {N_EVAL}/{N_EVAL} '
                        f'(fully available); partial coverage indicates inconsistent evaluation coverage')
    # 'full' and 'unavailable_global_dependency' are both acceptable in strict mode.

    if is_bicubic:
        return (len(reasons) == 0), reasons

    pd_vals = [per_sample[si].get('pd_distance', float('nan')) for si in range(N_EVAL)]
    mt_vals = [per_sample[si].get('mt_distance', float('nan')) for si in range(N_EVAL)]
    if not all(math.isfinite(v) and v >= 0 for v in pd_vals):
        reasons.append('pd_distance not finite/nonnegative for all 168 samples')
    if not all(math.isfinite(v) and v >= 0 for v in mt_vals):
        reasons.append('mt_distance not finite/nonnegative for all 168 samples')

    if all(math.isfinite(v) and v >= 0 for v in pd_vals) and all(math.isfinite(v) and v >= 0 for v in mt_vals):
        obs_pd = sum(pd_vals) / len(pd_vals)
        obs_mt = sum(mt_vals) / len(mt_vals)
        if expected_pd is not None and abs(obs_pd - expected_pd) > PD_MT_TOLERANCE:
            reasons.append(f'pd mean {obs_pd:.6f} differs from expected {expected_pd} by more than '
                            f'{PD_MT_TOLERANCE:g}')
        if expected_mt is not None and abs(obs_mt - expected_mt) > PD_MT_TOLERANCE:
            reasons.append(f'mt mean {obs_mt:.6f} differs from expected {expected_mt} by more than '
                            f'{PD_MT_TOLERANCE:g}')

    return (len(reasons) == 0), reasons


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--strict-primary', action='store_true',
                       help='(default) Authoritative mode: require every primary method to have complete, '
                            'cross-validated data; exit nonzero and skip writing the unified tables otherwise.')
    mode.add_argument('--audit-allow-missing', action='store_true',
                       help='Permissive audit mode: build the full grid with empty cells for missing methods '
                            'and always exit 0 unless a genuine data-integrity error occurs.')
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    strict = not args.audit_allow_missing

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log('=' * 88)
    log('Unified candidate evaluation -- Phase 1 audit')
    log(f'Repo root: {REPO_ROOT}')
    log(f"Mode: {'STRICT (--strict-primary, default)' if strict else 'AUDIT (--audit-allow-missing)'}")
    log('Read-only w.r.t. all experiment artifacts. No training/TTK/eval rerun performed.')
    log('=' * 88)

    # -------------------------------------------------------------------
    # Requirement 1: harvest repeated baseline rows from every discovered
    # candidate all_sample_metrics CSV.
    # -------------------------------------------------------------------
    eval_csvs = discover_candidate_eval_csvs()
    log(f'[harvest] Discovered {len(eval_csvs)} candidate all_sample_metrics CSV(s) under '
        f'ttk_runs_fixed/topology_finetuning/*_eval/.')
    for p in eval_csvs:
        log(f'[harvest]   - {p}')
    harvested_baseline_data, harvest_report = harvest_baseline_rows(eval_csvs)
    for bm in BASELINE_METHODS:
        cr = harvest_report['cross_report'][bm]
        canon = harvest_report['canonical'][bm] or '(none found)'
        log(f"[harvest:{bm}] sources_with_data={cr['n_sources']} canonical_source={canon}")

    # -------------------------------------------------------------------
    # Legacy combined source: remains the PD/MT topology source of record
    # for cnn/gan, and a cheap-metric fallback when no harvested rows exist.
    # -------------------------------------------------------------------
    legacy_baseline_data, legacy_report = load_legacy_baseline_data()
    for method, rep in legacy_report.get('per_method', {}).items():
        log(f"[legacy:{method}] rows={rep['n_rows']} unique_sample_idx={rep['n_unique_sample_idx']} "
            f"sample_index_status={rep['sample_index_status']} "
            f"cross_checked_vs_phase_c={rep['n_cross_checked_against_phase_c']} "
            f"max_pd_diff={rep['max_pd_diff_vs_phase_c']:.3e} max_mt_diff={rep['max_mt_diff_vs_phase_c']:.3e}")

    # -------------------------------------------------------------------
    # Requirement 2: cross-check harvested vs legacy for overlapping columns.
    # -------------------------------------------------------------------
    cross_check_report = cross_check_baseline_vs_legacy(harvested_baseline_data, legacy_baseline_data)
    for method, rep in cross_check_report.items():
        if rep.get('skipped'):
            log(f"[cross-check:{method}] skipped ({rep['reason']})")
        else:
            worst_metric, worst_val = _worst_numeric_diff(rep['max_diff'])
            log(f"[cross-check:{method}] max diff vs legacy combined source across "
                f"{len(rep['max_diff'])} overlapping metrics: {worst_metric}={worst_val:.3e}; "
                f"ssim_status={rep.get('ssim_status')}")

    baseline_data = build_baseline_per_sample(harvested_baseline_data, legacy_baseline_data)

    # -------------------------------------------------------------------
    # Resolve every manifest entry's artifacts on disk.
    # -------------------------------------------------------------------
    resolved = {}
    for entry in FULL_MANIFEST:
        mid = entry['method_id']
        if mid in BASELINE_METHODS:
            per_sample = baseline_data.get(mid, {})
            harvested_source = harvest_report['canonical'].get(mid) or ''
            cheap_resolution = ('harvested_baseline_row' if harvested_source
                                 else ('legacy_combined' if per_sample else 'not_found'))
            cheap_source = harvested_source or (legacy_report['combined_path'] if per_sample else '')
            has_topology = (mid != 'bicubic') and bool(per_sample)
            resolved[mid] = dict(
                data_out_dir_exists=rp(entry['data_out_dir']).is_dir() if entry['data_out_dir'] not in ('n/a', '') else False,
                idx_path=str(rp(entry['data_out_dir']) / 'idx.npy') if entry['data_out_dir'] not in ('n/a', '') else '',
                data_gt_path=str(rp(entry['data_out_dir']) / 'dataGT.npy') if entry['data_out_dir'] not in ('n/a', '') else '',
                data_sr_path=str(rp(entry['data_out_dir']) / 'dataSR.npy') if entry['data_out_dir'] not in ('n/a', '') else '',
                idx_exists=False, data_gt_exists=False, data_sr_exists=False,
                cheap_eval_csv=cheap_source, cheap_eval_csv_exists=bool(per_sample),
                cheap_eval_resolution=cheap_resolution,
                cheap_pairwise_csv='', cheap_pairwise_csv_exists=False, cheap_pairwise_resolution='not_found',
                topology_results_csv=legacy_report['phase_c_path'] if has_topology else '',
                topology_results_csv_exists=has_topology,
                topology_results_resolution='legacy_combined' if has_topology else 'not_found',
                topology_comparison_csv='', topology_comparison_csv_exists=False,
                topology_comparison_resolution='not_found',
                cheap_report_exists=False, topology_report_exists=False,
                per_sample=per_sample,
                row_count_cheap=len(per_sample), row_count_topology=(len(per_sample) if has_topology else 0),
            )
        else:
            resolved[mid] = resolve_candidate_artifacts(entry)

    # -------------------------------------------------------------------
    # Requirement 2 (raw array validation): load the canonical CNN benchmark
    # arrays once (mmap'd, never fully materialized), then validate every
    # primary method's idx/dataIN/dataGT/dataSR .npy files against it.
    # -------------------------------------------------------------------
    canonical_dir = rp(CANONICAL_CNN_DIR)
    canonical_in = canonical_gt = None
    if (canonical_dir / 'dataIN.npy').exists() and (canonical_dir / 'dataGT.npy').exists():
        try:
            canonical_in = _load_npy_mmap(canonical_dir / 'dataIN.npy')
            canonical_gt = _load_npy_mmap(canonical_dir / 'dataGT.npy')
            log(f'[arrays] Loaded canonical CNN benchmark arrays (mmap) from {canonical_dir}: '
                f'dataIN.shape={canonical_in.shape}, dataGT.shape={canonical_gt.shape}')
        except Exception as e:
            log(f'[arrays] Failed to load canonical CNN benchmark arrays from {canonical_dir}: {e}')
    else:
        log(f'[arrays] Canonical CNN benchmark arrays not found at {canonical_dir} -- alignment cannot be '
            f'validated for any method until they are present.')

    array_validation = {}
    for entry in PRIMARY_MANIFEST:
        mid = entry['method_id']
        data_out_dir = rp(entry['data_out_dir']) if entry.get('data_out_dir') not in (None, '', 'n/a') else None
        require_idx = (mid != 'bicubic')
        if data_out_dir is None:
            status = dict(ARRAY_STATUS_NOT_APPLICABLE)
            status['idx_validation_status'] = status['input_alignment_status'] = 'missing'
            status['gt_alignment_status'] = status['sr_shape_status'] = status['sr_finiteness_status'] = 'missing'
            reasons = [f'no data_out_dir configured for {mid!r}']
        else:
            status, reasons = validate_raw_arrays(data_out_dir, canonical_in, canonical_gt, require_idx=require_idx)
        array_validation[mid] = (status, reasons)
        worst = '; '.join(reasons) if reasons else 'OK'
        log(f'[arrays:{mid}] idx={status.get("idx_validation_status")} '
            f'in={status.get("input_alignment_status")} gt={status.get("gt_alignment_status")} '
            f'sr_shape={status.get("sr_shape_status")} sr_finite={status.get("sr_finiteness_status")} '
            f'-- {worst}')

    # -------------------------------------------------------------------
    # method_inventory.csv (always written -- diagnostic, not a completeness claim)
    # -------------------------------------------------------------------
    inventory_cols = [
        'method_id', 'display_name', 'original_method_name', 'candidate_family',
        'comparison_tier', 'include_primary', 'training_scale', 'architecture',
        'training_tfrecord', 'evaluation_tfrecord', 'objective_summary',
        'uses_speed', 'uses_grad', 'uses_levelset', 'uses_crit', 'uses_e2',
        'repaired_status', 'historical_alias', 'data_out_dir', 'idx_path', 'data_gt_path', 'data_sr_path',
        'cheap_eval_csv', 'cheap_eval_resolution', 'cheap_pairwise_csv',
        'topology_results_csv', 'topology_results_resolution', 'topology_comparison_csv',
        'cheap_report', 'topology_report', 'row_count_cheap', 'row_count_topology',
        'sample_index_status', 'ssim_status',
        'idx_validation_status', 'input_alignment_status', 'gt_alignment_status',
        'sr_shape_status', 'sr_finiteness_status',
        'topology_mean_pd', 'topology_mean_mt',
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
                                'files at all). Cheap metrics are sourced from harvested candidate-eval baseline '
                                'rows when available (scripts/generate_bicubic_baseline.py, output convention '
                                'data_out_fixed/wind_mrhr_bicubic/); missing entirely otherwise.')
            else:
                validation_status = 'no_expected_value'
                notes_extra = ''
        elif not pd_vals:
            validation_status = 'NO_DATA'
            notes_extra = ('No PD/MT source artifact found in this git checkout. Large per-candidate experiment '
                            'outputs are gitignored (*.npy, *.npz, ttk_runs_fixed/topology_finetuning/* except '
                            'candidateE_constraints) and appear to exist only on the separate training machine. '
                            f'Expected mean (from user-supplied reference values, NOT verified here): '
                            f'PD={exp_pd}, MT={exp_mt}.')
        else:
            pd_ok = abs(topo_mean_pd - exp_pd) <= PD_MT_TOLERANCE
            mt_ok = abs(topo_mean_mt - exp_mt) <= PD_MT_TOLERANCE
            validation_status = 'PASS' if (pd_ok and mt_ok) else 'FAIL'
            notes_extra = f'pd_abs_diff={abs(topo_mean_pd - exp_pd):.6f} mt_abs_diff={abs(topo_mean_mt - exp_mt):.6f}'

        if mid in array_validation:
            arr_status, arr_reasons = array_validation[mid]
        else:
            arr_status, arr_reasons = dict(ARRAY_STATUS_NOT_APPLICABLE), []

        ssim_status_val, ssim_finite_val = classify_ssim_status(per_sample)

        notes = entry.get('exclusion_reason', '') or ''
        if notes_extra:
            notes = (notes + ' ' if notes else '') + notes_extra
        if arr_reasons:
            notes = (notes + ' ' if notes else '') + 'Array validation: ' + '; '.join(arr_reasons)

        row = dict(
            method_id=mid, display_name=entry['display_name'],
            original_method_name=entry['original_method_name'], candidate_family=entry['candidate_family'],
            comparison_tier=entry['comparison_tier'], include_primary=entry['include_primary'],
            training_scale=entry['training_scale'], architecture=entry['architecture'],
            training_tfrecord=entry['training_tfrecord'], evaluation_tfrecord=entry['evaluation_tfrecord'],
            objective_summary=entry['objective_summary'],
            uses_speed=entry['uses_speed'], uses_grad=entry['uses_grad'], uses_levelset=entry['uses_levelset'],
            uses_crit=entry['uses_crit'], uses_e2=entry['uses_e2'], repaired_status=entry['repaired_status'],
            historical_alias=entry.get('historical_alias', ''),
            data_out_dir=entry['data_out_dir'],
            idx_path=r.get('idx_path', ''), data_gt_path=r.get('data_gt_path', ''), data_sr_path=r.get('data_sr_path', ''),
            cheap_eval_csv=r.get('cheap_eval_csv', ''), cheap_eval_resolution=r.get('cheap_eval_resolution', ''),
            cheap_pairwise_csv=r.get('cheap_pairwise_csv', ''),
            topology_results_csv=r.get('topology_results_csv', ''),
            topology_results_resolution=r.get('topology_results_resolution', ''),
            topology_comparison_csv=r.get('topology_comparison_csv', ''),
            cheap_report=entry.get('cheap_report', ''), topology_report=entry.get('topology_report', ''),
            row_count_cheap=r.get('row_count_cheap', 0), row_count_topology=r.get('row_count_topology', 0),
            sample_index_status=sample_index_status,
            ssim_status=f'{ssim_status_val} ({ssim_finite_val}/{N_EVAL})',
            idx_validation_status=arr_status.get('idx_validation_status', 'not_checked_secondary_tier'),
            input_alignment_status=arr_status.get('input_alignment_status', 'not_checked_secondary_tier'),
            gt_alignment_status=arr_status.get('gt_alignment_status', 'not_checked_secondary_tier'),
            sr_shape_status=arr_status.get('sr_shape_status', 'not_checked_secondary_tier'),
            sr_finiteness_status=arr_status.get('sr_finiteness_status', 'not_checked_secondary_tier'),
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
    # column_mapping.csv (always written)
    # -------------------------------------------------------------------
    write_column_mapping()

    # -------------------------------------------------------------------
    # docs/unified_candidate_evaluation_inventory.md (always written)
    # docs/primary_candidate_artifact_reference.md (always written, requirement 5)
    # -------------------------------------------------------------------
    write_inventory_doc(inventory_rows, legacy_report, harvest_report, cross_check_report)
    write_primary_reference_doc(inventory_rows)

    # -------------------------------------------------------------------
    # Requirement 3: strict-mode completeness gate.
    # -------------------------------------------------------------------
    primary_strict_results = {}
    for entry in PRIMARY_MANIFEST:
        mid = entry['method_id']
        per_sample = resolved[mid].get('per_sample', {})
        exp_pd, exp_mt = EXPECTED_PD_MT.get(mid, (None, None))
        _, arr_reasons = array_validation.get(mid, ({}, []))
        passed, reasons = check_strict_primary(mid, per_sample, is_bicubic=(mid == 'bicubic'),
                                                expected_pd=exp_pd, expected_mt=exp_mt,
                                                array_reasons=arr_reasons)
        primary_strict_results[mid] = (passed, reasons)
        log(f"[strict-check:{mid}] {'PASS' if passed else 'FAIL: ' + '; '.join(reasons)}")

    all_strict_pass = all(p for p, _ in primary_strict_results.values())

    if strict and not all_strict_pass:
        log('')
        log('=' * 88)
        log('[STRICT MODE FAILURE] Not every primary method meets the strict completeness/validation criteria.')
        for mid, (passed, reasons) in primary_strict_results.items():
            if not passed:
                log(f'  - {mid}: {"; ".join(reasons)}')
        log('Refusing to write unified_primary_per_sample_long.csv / unified_primary_method_summary.csv / '
            'unified_primary_topology_validation.csv / unified_primary_pairwise_vs_cnn.csv / '
            'unified_primary_missingness.csv / unified_primary_wide.csv: doing so in strict mode would produce '
            'an apparently complete authoritative table containing empty placeholder rows.')
        stale = [f.name for f in OUT_DIR.glob('unified_primary_*.csv') if f.exists()]
        if stale:
            log(f'[warning] Stale file(s) from a previous (permissive) run still exist and were NOT regenerated '
                f'or deleted by this strict run -- do not treat them as current: {stale}')
        log('Re-run with --audit-allow-missing for a permissive inventory-only pass, or fix the missing '
            'artifacts and re-run --strict-primary.')
        log('=' * 88)
        write_phase1_report_strict_failure(inventory_rows, primary_strict_results, harvest_report,
                                            legacy_report, cross_check_report)
        flush_log()
        return 1

    # -------------------------------------------------------------------
    # unified_primary_per_sample_long.csv -- full grid. In audit mode this
    # may contain empty cells for missing methods; in strict mode (reached
    # only if every check above passed) every cell is real.
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
            continue
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
            improve_deltas = raw_deltas if direction == 'higher_is_better' else [-d for d in raw_deltas]
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
            reason = classify_missing_reason(col, per_sample, finite, total)
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

    write_phase1_report(inventory_rows, topo_val_rows, legacy_report, missingness_rows,
                         strict, all_strict_pass, harvest_report, cross_check_report)

    n_primary_with_data = sum(1 for e in PRIMARY_MANIFEST if resolved[e['method_id']].get('per_sample'))
    log('')
    log('=' * 88)
    log(f'RESULT: {n_primary_with_data}/{len(PRIMARY_MANIFEST)} primary methods have real per-sample data '
        f'in this repository checkout. Strict criteria all-pass: {all_strict_pass}.')
    if not all_strict_pass:
        log('Do not claim full-table success: some primary methods are MISSING or incomplete (see '
            'method_inventory.csv, validation_status column, and docs/unified_candidate_evaluation_phase1.md).')
    log('=' * 88)

    flush_log()
    return 0


def write_column_mapping():
    mapping_rows = []
    for src_col, std_col in sorted(BASELINE_COMBINED_DIRECT_MAP.items()):
        mapping_rows.append(dict(
            source_path='ttk_runs_fixed/combined/psnr_topology_physics_merged.csv',
            source_column=src_col, standardized_column=std_col,
            units=[m[3] for m in METRIC_SCHEMA if m[0] == std_col][0],
            direction=METRIC_DIRECTION[std_col],
            representation=[m[2] for m in METRIC_SCHEMA if m[0] == std_col][0],
            notes='Legacy cnn/gan-only pipeline; used as the topology (PD/MT) source of record and as a cheap-'
                  'metric fallback when no harvested candidate-eval baseline row is available.',
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
            notes='This legacy pipeline predates speed_mae/speed_rmse/component-count metrics; sourced instead '
                  'from harvested ttk_runs_fixed/topology_finetuning/*_eval/all_sample_metrics_*.csv baseline '
                  'rows when at least one such file exists (see harvest_baseline_rows()).',
        ))
    for src_col, std_col in sorted(CANDIDATE_EVAL_DIRECT_MAP.items()):
        mapping_rows.append(dict(
            source_path='ttk_runs_fixed/topology_finetuning/<method>_eval/all_sample_metrics_<method>.csv '
                         '(scripts/evaluate_finetune_candidate.py schema; also the source harvested for '
                         'repeated bicubic/cnn/gan baseline rows)',
            source_column=src_col, standardized_column=std_col,
            units=[m[3] for m in METRIC_SCHEMA if m[0] == std_col][0],
            direction=METRIC_DIRECTION[std_col],
            representation=[m[2] for m in METRIC_SCHEMA if m[0] == std_col][0],
            notes='Full candidate/baseline schema written by evaluate_finetune_candidate.py.',
        ))
    mapping_path = OUT_DIR / 'column_mapping.csv'
    with mapping_path.open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['source_path', 'source_column', 'standardized_column',
                                            'units', 'direction', 'representation', 'notes'])
        w.writeheader()
        for row in mapping_rows:
            w.writerow(row)
    log(f'[write] {mapping_path} ({len(mapping_rows)} rows)')


def write_inventory_doc(inventory_rows, legacy_report, harvest_report, cross_check_report):
    lines = []
    lines.append('# Unified candidate evaluation -- artifact inventory')
    lines.append('')
    lines.append('Generated by `scripts/build_unified_candidate_evaluation.py`. Read-only audit; '
                 'no training, TTK, or cheap-evaluation rerun was performed.')
    lines.append('')
    lines.append('## Primary method set')
    lines.append('')
    lines.append('| method_id | display_name | comparison_tier | validation_status | row_count_cheap | '
                 'cheap_eval_resolution | notes |')
    lines.append('|---|---|---|---|---:|---|---|')
    for row in inventory_rows:
        if not row['include_primary']:
            continue
        notes = (row['notes'] or '').replace('|', '\\|')
        if len(notes) > 160:
            notes = notes[:157] + '...'
        lines.append(f"| {row['method_id']} | {row['display_name']} | {row['comparison_tier']} | "
                     f"{row['validation_status']} | {row['row_count_cheap']} | {row['cheap_eval_resolution']} | "
                     f"{notes} |")
    lines.append('')
    lines.append('## Secondary / non-primary inventory')
    lines.append('')
    lines.append('| method_id | display_name | comparison_tier | training_scale | architecture | '
                 'repaired_status | exclusion_reason |')
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
    lines.append('## Baseline harvesting (requirement 1)')
    lines.append('')
    lines.append(f"- Discovered candidate all_sample_metrics CSVs: {len(harvest_report['discovered_csvs'])}")
    for p in harvest_report['discovered_csvs']:
        lines.append(f'  - `{p}`')
    for bm in BASELINE_METHODS:
        cr = harvest_report['cross_report'][bm]
        canon = harvest_report['canonical'][bm] or '(none found)'
        ssim_canon = harvest_report['canonical_ssim'].get(bm) or '(none -- all-NaN)'
        lines.append(f"- **{bm}**: {cr['n_sources']} source(s) with data, canonical source (required metrics) = "
                     f"`{canon}`; ssim_availability=`{cr['ssim_availability']}`, canonical ssim source = "
                     f"`{ssim_canon}`")
    lines.append('')
    lines.append('## Baseline (cnn/gan) legacy-source cross-validation')
    lines.append('')
    lines.append(f"- Legacy combined source: `{legacy_report.get('combined_path')}`")
    lines.append(f"- Legacy cross-validation source: `{legacy_report.get('phase_c_path')}`")
    for method, rep in legacy_report.get('per_method', {}).items():
        lines.append(f"- **{method}**: {rep['n_rows']} rows, {rep['n_unique_sample_idx']} unique sample_idx "
                     f"({rep['sample_index_status']}), cross-checked against {rep['n_cross_checked_against_phase_c']} "
                     f"phase_c_results.csv rows, max |Δpd|={rep['max_pd_diff_vs_phase_c']:.3e}, "
                     f"max |Δmt|={rep['max_mt_diff_vs_phase_c']:.3e}.")
    lines.append('')
    lines.append('## Harvested-vs-legacy cross-check (requirement 2)')
    lines.append('')
    for method, rep in cross_check_report.items():
        if rep.get('skipped'):
            lines.append(f"- **{method}**: skipped ({rep['reason']})")
        else:
            worst_metric, worst_val = _worst_numeric_diff(rep['max_diff'])
            lines.append(f"- **{method}**: max diff across {len(rep['max_diff'])} overlapping metrics: "
                         f"`{worst_metric}` = {worst_val:.3e} (tolerance {OLDER_SOURCE_CROSS_CHECK_TOLERANCE:g}); "
                         f"ssim_status=`{rep.get('ssim_status')}`")
    lines.append('')
    lines.append('## Architecture legend')
    lines.append('')
    lines.append('- `native_tf`: PhIREGANs.pretrain() generator fine-tuning path (tensorflow.compat.v1).')
    lines.append('- `pytorch_residual_refiner`: frozen pretrained CNN output + a small PyTorch residual refiner '
                 '(architecture confound relative to native_tf; kept secondary-tier only).')
    lines.append('- `pretrained_cnn` / `pretrained_gan`: released PhIRE checkpoints, no fine-tuning.')
    lines.append('- `bicubic_interpolation`: no learned model.')
    lines.append('')
    lines.append('## Path-resolution legend (requirement 4)')
    lines.append('')
    lines.append('- `exact_path`: found at the manifest-derived expected path.')
    lines.append('- `fallback_discovery`: exact path absent; found via a unique filename match under '
                 f'{FALLBACK_SEARCH_ROOTS}.')
    lines.append('- `not_found`: absent at the exact path and no unique fallback match exists.')
    lines.append('- `harvested_baseline_row` / `legacy_combined`: bicubic/cnn/gan-specific sources (see above).')
    (DOCS_DIR / 'unified_candidate_evaluation_inventory.md').write_text('\n'.join(lines) + '\n')
    log(f"[write] {DOCS_DIR / 'unified_candidate_evaluation_inventory.md'}")


def write_primary_reference_doc(inventory_rows):
    """Requirement 5: human-readable candidate reference, repository-relative paths."""
    by_id = {r['method_id']: r for r in inventory_rows}
    lines = []
    lines.append('# Primary candidate artifact reference')
    lines.append('')
    lines.append('Human-readable reference for every primary method in the Phase-1 unified evaluation. '
                 'All paths are repository-relative. Generated by `scripts/build_unified_candidate_evaluation.py`; '
                 'read-only, no training/TTK/eval rerun.')
    lines.append('')
    for entry in PRIMARY_MANIFEST:
        mid = entry['method_id']
        inv = by_id[mid]
        lines.append(f"## `{mid}` -- {entry['display_name']}")
        lines.append('')
        lines.append(f"- **original_method_name**: `{entry['original_method_name']}`")
        lines.append(f"- **weighted loss objective**: {entry['objective_summary']}")
        lines.append(f"- **candidate_family**: {entry['candidate_family']}")
        lines.append(f"- **historical_alias**: {entry.get('historical_alias') or 'n/a'}")
        lines.append(f"- **training_scale**: {entry['training_scale']}")
        lines.append(f"- **architecture**: {entry['architecture']}")
        lines.append(f"- **data_out directory**: `{_relpath(rp(entry['data_out_dir'])) if entry['data_out_dir'] not in ('n/a', '') else 'n/a'}`")
        lines.append(f"- **model directory**: `{entry.get('model_dir', '') or 'n/a'}`")
        lines.append(f"- **cheap-metric CSV**: `{_relpath(inv['cheap_eval_csv']) or '(not resolved)'}` "
                     f"[{inv['cheap_eval_resolution'] or 'not_found'}]")
        lines.append(f"- **pairwise-vs-CNN CSV**: `{_relpath(inv['cheap_pairwise_csv']) or '(not resolved)'}`")
        lines.append(f"- **PD/MT per-sample CSV**: `{_relpath(inv['topology_results_csv']) or '(not resolved)'}` "
                     f"[{inv['topology_results_resolution'] or 'not_found'}]")
        lines.append(f"- **topology comparison CSV**: `{_relpath(inv['topology_comparison_csv']) or '(not resolved)'}`")
        lines.append(f"- **cheap report**: `{entry.get('cheap_report') or '(not resolved)'}`")
        lines.append(f"- **topology report**: `{entry.get('topology_report') or '(not resolved)'}`")
        exp_pd, exp_mt = EXPECTED_PD_MT.get(mid, ('n/a', 'n/a'))
        lines.append(f"- **expected validation PD mean**: {exp_pd}")
        lines.append(f"- **expected validation MT mean**: {exp_mt}")
        lines.append('')
    (DOCS_DIR / 'primary_candidate_artifact_reference.md').write_text('\n'.join(lines) + '\n')
    log(f"[write] {DOCS_DIR / 'primary_candidate_artifact_reference.md'}")


def write_phase1_report_strict_failure(inventory_rows, primary_strict_results, harvest_report,
                                        legacy_report, cross_check_report):
    primary_rows = [r for r in inventory_rows if r['include_primary']]
    failing = [(mid, reasons) for mid, (passed, reasons) in primary_strict_results.items() if not passed]
    lines = []
    lines.append('# Unified candidate evaluation -- Phase 1 report')
    lines.append('')
    lines.append('## STRICT MODE: FAILED')
    lines.append('')
    lines.append(f'{len(failing)} of {len(primary_rows)} primary methods did not meet the `--strict-primary` '
                 'completeness/validation criteria. The authoritative unified tables '
                 '(`unified_primary_per_sample_long.csv` and its derived summary/validation/pairwise/'
                 'missingness/wide tables) were **not written** in this run, to avoid producing a table that '
                 'looks complete but silently contains empty placeholder rows. `method_inventory.csv`, '
                 '`column_mapping.csv`, `docs/unified_candidate_evaluation_inventory.md`, and '
                 '`docs/primary_candidate_artifact_reference.md` were written -- they are diagnostic and do not '
                 'claim table completeness.')
    lines.append('')
    lines.append('## Failing methods and reasons')
    lines.append('')
    for mid, reasons in failing:
        lines.append(f'- **{mid}**: {"; ".join(reasons)}')
    lines.append('')
    lines.append('## Baseline harvesting summary')
    lines.append('')
    lines.append(f"Discovered {len(harvest_report['discovered_csvs'])} candidate all_sample_metrics CSV(s).")
    for bm in BASELINE_METHODS:
        cr = harvest_report['cross_report'][bm]
        lines.append(f"- {bm}: {cr['n_sources']} source(s), canonical = `{harvest_report['canonical'][bm] or '(none)'}`")
    lines.append('')
    lines.append('## No training or TTK was rerun')
    lines.append('')
    lines.append('This script performed zero training runs, zero TTK invocations, and zero cheap-evaluation '
                 'runs. It only read pre-existing CSV files already present in the repository. No existing '
                 'artifact was modified or deleted.')
    lines.append('')
    lines.append('## Next step')
    lines.append('')
    lines.append('Sync the missing artifacts listed above from the training machine, or investigate why the '
                 'PD/MT means or row counts disagree, then re-run `python3 scripts/build_unified_candidate_'
                 'evaluation.py --strict-primary`.')
    (DOCS_DIR / 'unified_candidate_evaluation_phase1.md').write_text('\n'.join(lines) + '\n')
    log(f"[write] {DOCS_DIR / 'unified_candidate_evaluation_phase1.md'} (strict-failure variant)")


def write_phase1_report(inventory_rows, topo_val_rows, legacy_report, missingness_rows,
                         strict, all_strict_pass, harvest_report, cross_check_report):
    primary_rows = [r for r in inventory_rows if r['include_primary']]
    n_pass = sum(1 for r in topo_val_rows if r['pd_pass'] is True and r['mt_pass'] is True)
    n_no_data = sum(1 for r in topo_val_rows if r['pd_pass'] == 'NO_DATA')
    n_fail = len(topo_val_rows) - n_pass - n_no_data
    methods_with_data = [r['method_id'] for r in primary_rows if r['row_count_cheap'] > 0]
    methods_without_data = [r['method_id'] for r in primary_rows if r['row_count_cheap'] == 0]

    lines = []
    lines.append('# Unified candidate evaluation -- Phase 1 report')
    lines.append('')
    lines.append('## 0. Run mode')
    lines.append('')
    if strict:
        lines.append('**STRICT MODE (`--strict-primary`): PASSED.** Every primary method met the completeness/'
                     'validation criteria (168 cheap rows with sample_idx exactly 0..167, finite nonnegative '
                     'PD/MT for the 18 learned/baseline-with-topology methods, PD/MT mean reproduction within '
                     f'{PD_MT_TOLERANCE:g}). The tables below are fully authoritative -- no placeholder rows.')
    else:
        lines.append('**AUDIT MODE (`--audit-allow-missing`).** This is a permissive inventory pass: the '
                     'unified table below intentionally includes empty cells for any method/metric with no '
                     'source artifact in this checkout. Do not treat it as authoritative -- see section 9.')
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
    lines.append('See `ttk_runs_fixed/unified_candidate_evaluation/method_inventory.csv`, '
                 '`docs/unified_candidate_evaluation_inventory.md`, and `docs/primary_candidate_artifact_reference.md`.')
    lines.append('')
    lines.append('## 3. Exact table dimensions')
    lines.append('')
    lines.append(f'- `unified_primary_per_sample_long.csv`: {len(primary_rows)} methods x 168 samples = '
                 f'{len(primary_rows) * N_EVAL} rows, one row per (method_id, sample_idx), no duplicates.')
    lines.append(f'- Of these {len(primary_rows)} primary methods, **{len(methods_with_data)} have real per-sample '
                 f'data** in this repository checkout ({", ".join(methods_with_data) or "none"}), and '
                 f'**{len(methods_without_data)} have zero real data** ({", ".join(methods_without_data) or "none"}).')
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
                 f'out of {len(topo_val_rows)} primary methods with an expected value.')
    lines.append('- Cheap-metric completeness (168 rows, sample_idx exactly 0..167, no duplicates): checked for '
                 'every primary method with any discovered source (baseline-harvested, legacy-combined, or a '
                 'resolved candidate all_sample_metrics CSV).')
    lines.append('- Join (cheap metrics <-> true topology, one-to-one on sample_idx): every per-sample record is '
                 'a single merged dict keyed by sample_idx, so a missing cheap or topology value for a given '
                 'sample_idx shows up directly as a non-finite cell rather than a silent row-count mismatch.')
    lines.append('')
    lines.append('## 6. Baseline duplicate-consistency audit')
    lines.append('')
    lines.append(f"Discovered {len(harvest_report['discovered_csvs'])} candidate `all_sample_metrics_*.csv` file(s) "
                 'under `ttk_runs_fixed/topology_finetuning/*_eval/`. Every discovered file was checked for '
                 'bicubic/cnn/gan rows; every baseline method present in more than one file had its rows compared '
                 f'pairwise, metric by metric, against a {BASELINE_CROSS_SOURCE_TOLERANCE:g} tolerance (would '
                 'hard-fail the whole run on disagreement):')
    for bm in BASELINE_METHODS:
        cr = harvest_report['cross_report'][bm]
        ssim_canon = harvest_report['canonical_ssim'].get(bm) or '(none -- all-NaN)'
        lines.append(f"  - **{bm}**: {cr['n_sources']} source(s) with data; canonical source (required metrics) = "
                     f"`{harvest_report['canonical'][bm] or '(none found)'}`; "
                     f"ssim_availability=`{cr['ssim_availability']}`, canonical ssim source = `{ssim_canon}`.")
    lines.append('')
    lines.append('Canonical cnn/gan rows were additionally cross-checked against the older '
                 '`ttk_runs_fixed/combined/psnr_topology_physics_merged.csv` pipeline for overlapping columns '
                 f'(tolerance {OLDER_SOURCE_CROSS_CHECK_TOLERANCE:g}):')
    for method, rep in cross_check_report.items():
        if rep.get('skipped'):
            lines.append(f"  - **{method}**: skipped ({rep['reason']}).")
        else:
            worst_metric, worst_val = _worst_numeric_diff(rep['max_diff'])
            lines.append(f"  - **{method}**: worst overlapping-column disagreement `{worst_metric}` = "
                         f"{worst_val:.3e}; ssim_status=`{rep.get('ssim_status')}`.")
    lines.append('')
    lines.append('The independent `ttk_runs_fixed/combined/phase_c_results.csv` PD/MT cross-check remains as before:')
    for method, rep in legacy_report.get('per_method', {}).items():
        lines.append(f"  - **{method}**: max |Δpd| = {rep['max_pd_diff_vs_phase_c']:.3e}, "
                     f"max |Δmt| = {rep['max_mt_diff_vs_phase_c']:.3e} across "
                     f"{rep['n_cross_checked_against_phase_c']} cross-checked samples.")
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
    lines.append('Every (method, metric) cell in `unified_primary_missingness.csv` is classified into exactly '
                 'one of three `missing_reason` categories: `no_source_artifact` (no file provides this metric '
                 'for this method at all), `unavailable_global_dependency` (SSIM specifically, 0/168 finite -- '
                 'consistent with the documented NumPy/scikit-image ABI incompatibility, not a data-quality bug), '
                 'or `partial_source_coverage` (1..167/168 finite -- inconsistent coverage, always treated as a '
                 'strict-mode failure since it indicates a real problem rather than a known benign gap).')
    lines.append('')
    ssim_rows = [r for r in missingness_rows if r['metric'] == 'ssim_speed']
    ssim_full = [r['method_id'] for r in ssim_rows if r['missing_reason'] == '']
    ssim_unavailable = [r['method_id'] for r in ssim_rows if r['missing_reason'] == 'unavailable_global_dependency']
    ssim_no_source = [r['method_id'] for r in ssim_rows if r['missing_reason'] == 'no_source_artifact']
    ssim_partial = [r['method_id'] for r in ssim_rows if r['missing_reason'] == 'partial_source_coverage']
    lines.append(f'- SSIM (`ssim_speed`) is in `OPTIONAL_CHEAP_METRIC_COLUMNS`: strict mode accepts either full '
                 '(168/168) or fully-unavailable (0/168) coverage, and hard-fails only on partial coverage.')
    lines.append(f'  - Fully available (168/168): {ssim_full or "none"}')
    lines.append(f'  - Globally unavailable, accepted (0/168, `unavailable_global_dependency`): {ssim_unavailable or "none"}')
    lines.append(f'  - No source at all for this method (`no_source_artifact`): {ssim_no_source or "none"}')
    lines.append(f'  - Partial coverage, would strict-fail (`partial_source_coverage`): {ssim_partial or "none"}')
    lines.append('- SSIM is never filled, copied from a legacy row into a candidate row, or recomputed -- missing '
                 'SSIM stays an empty cell in the unified table exactly as found in its source.')
    lines.append('- Pairwise-vs-CNN summaries report `n_valid=0` for SSIM (and every other metric) when the '
                 'candidate/CNN intersection of finite samples is empty, rather than fabricating a comparison.')
    lines.append('- See `unified_primary_missingness.csv` for the full total/finite/missing breakdown per '
                 '(method_id, metric).')
    lines.append('- No missing value was filled with zero or inferred; all gaps are empty cells in the CSVs.')
    lines.append('')
    lines.append('## 8b. Raw benchmark/array validation')
    lines.append('')
    lines.append('For every primary method, `idx.npy`/`dataIN.npy`/`dataGT.npy`/`dataSR.npy` under its '
                 '`data_out_dir` (or `data_out_fixed/wind_mrhr_<method>/` for the three baselines) are validated '
                 'against the canonical CNN benchmark arrays at '
                 f'`{CANONICAL_CNN_DIR}/{{idx,dataIN,dataGT}}.npy` -- loaded via `np.load(mmap_mode="r", '
                 'allow_pickle=False)` and compared in '
                 f'{ARRAY_COMPARE_CHUNK_SAMPLES}-sample chunks so a full (168, 500, 500, 2) array is never fully '
                 'materialized in memory. Checks: `idx.npy` shape `(168,)` and exactly `np.arange(168)`; '
                 '`dataIN.npy` shape `(168, 100, 100, 2)` and exactly equal to the canonical `dataIN.npy`; '
                 '`dataGT.npy` shape `(168, 500, 500, 2)` and exactly equal to the canonical `dataGT.npy`; '
                 '`dataSR.npy` shape `(168, 500, 500, 2)` and entirely finite. `idx.npy` is not required for '
                 'bicubic (its generator script does not produce one); every other file is required for every '
                 'primary method. No `.npy` file is ever written or modified by this script.')
    lines.append('')
    lines.append('| method_id | idx_validation_status | input_alignment_status | gt_alignment_status | '
                 'sr_shape_status | sr_finiteness_status |')
    lines.append('|---|---|---|---|---|---|')
    for row in inventory_rows:
        if not row['include_primary']:
            continue
        lines.append(f"| {row['method_id']} | {row['idx_validation_status']} | {row['input_alignment_status']} | "
                     f"{row['gt_alignment_status']} | {row['sr_shape_status']} | {row['sr_finiteness_status']} |")
    lines.append('')
    lines.append('## 9. Candidates that could not be included and why')
    lines.append('')
    if methods_without_data:
        lines.append(f'{len(methods_without_data)} of {len(primary_rows)} primary methods '
                     f'({", ".join(methods_without_data)}) have **zero** real per-sample cheap-evaluation or '
                     'true-topology artifacts anywhere in this git checkout. Root cause: this repository\'s '
                     '`.gitignore` excludes `*.npy`, `*.npz`, `data_out/`, and '
                     '`ttk_runs_fixed/topology_finetuning/*` (tracked exceptions are only `candidateE_constraints` '
                     'and the cnn/gan `combined`/`phase_c_final` summary artifacts); large experiment outputs for '
                     'the loss-ablation candidates are produced only on the separate training machine and were '
                     'never committed.')
    else:
        lines.append('All primary methods have real data in this checkout.')
    lines.append('')
    lines.append('## 10. No training or TTK was rerun')
    lines.append('')
    lines.append('This script and this audit performed zero training runs, zero TTK invocations, and zero cheap-'
                 'evaluation runs. It only read pre-existing CSV files already present in the repository. No '
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
    lines.append('- `docs/primary_candidate_artifact_reference.md`')
    lines.append('- `docs/unified_candidate_evaluation_phase1.md` (this file)')
    lines.append('- `logs/build_unified_candidate_evaluation.log`')
    lines.append('')
    lines.append('## 12. Recommended next step')
    lines.append('')
    if methods_without_data:
        lines.append(f'Before any factorial-effect analysis, paired contrasts, correlations, or Pareto-front work '
                     f'can be performed on the full primary set, the {len(methods_without_data)} missing methods\' '
                     'cheap-evaluation and true-topology artifacts need to be synced from the training machine '
                     'into this checkout (or this script re-run there with `--strict-primary`). Until then, any '
                     f'such analysis is only valid for the {len(methods_with_data)} methods with real data '
                     f'({", ".join(methods_with_data)}).')
    else:
        lines.append('The primary set is complete and strict-validated in this run. Per the task instructions, '
                     'no correlation, factorial-model, Pareto-front, or visualization-selection work was performed '
                     'in this Phase-1 pass -- that is the recommended next step.')
    (DOCS_DIR / 'unified_candidate_evaluation_phase1.md').write_text('\n'.join(lines) + '\n')
    log(f"[write] {DOCS_DIR / 'unified_candidate_evaluation_phase1.md'}")


if __name__ == '__main__':
    sys.exit(main())
