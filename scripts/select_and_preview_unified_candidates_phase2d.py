#!/usr/bin/env python3
"""Phase 2D-A: deterministic archetype sample selection and raw-artifact
audit/preview rendering for the unified wind-SR candidate benchmark.

Split execution architecture (two independently runnable stages):

  --selection-only   Reads only frozen Phase-1/2A/2B/2C CSV and Markdown
                      inputs. Computes archetype scores, selected samples,
                      alternates, diagnostics, and the preview plan. Must
                      not touch or require raw .npy arrays. This is the
                      mode that can run in a lightweight checkout where
                      data_out/ and data_out_fixed/ (gitignored) are absent.

  --render-previews   Reads and validates the deterministic selection
                      manifest written by --selection-only. Requires the
                      raw Spark arrays (data_out/, data_out_fixed/).
                      Performs the full raw-artifact audit and generates
                      review-only preview PNGs. Hard-fails if any required
                      raw artifact is unavailable or invalid.

  --full              Runs selection then raw auditing/rendering in one
                      process.

The authoritative Phase-2D-A completion state is reached only when --full
or --render-previews succeeds (on a machine where the raw Spark arrays are
present). A --selection-only run in a lightweight checkout is reported
honestly as `selection_complete_preview_render_pending_raw_artifacts` and
never claims full completion.

Read-only with respect to every Phase-1, Phase-2A, Phase-2B, and Phase-2C
artifact (86 files total: 12 + 14 + 28 + 32). Never runs training,
inference, cheap evaluation, or TTK. Never writes outside:

    ttk_runs_fixed/unified_candidate_analysis/phase2d/{selection,preview_audit,previews}/
    docs/unified_candidate_analysis_phase2d.md
    logs/unified_candidate_analysis_phase2d_selection.log
    logs/unified_candidate_analysis_phase2d_render.log

Absolute paths recorded in method_inventory.csv (e.g. /home/adadhwal/PhIRE/...)
are historical provenance only and are never used as executable paths. Raw
artifacts are resolved purely from repository-relative conventions rooted
at REPO_ROOT: `data_out/wind_finetune_<original_method_name>/` for the four
learned finetuned candidates, and the canonical fixed `data_out_fixed/
wind_mrhr_{cnn,gan}/` locations for the two pretrained baselines. Bicubic
has no saved dataSR.npy of its own: it is reconstructed in memory from
CNN's canonical dataIN.npy via `scipy.ndimage.zoom(order=3, mode='reflect',
prefilter=True)`, applied independently per (u, v) channel, and is never
written back to any earlier-phase or new baseline-dataset location.

The saved dataIN/dataGT/dataSR arrays are already physical (denormalized)
[u, v] wind-vector fields. Scalar wind speed is `sqrt(u**2 + v**2)`. The
MU_SIG normalization constants are recorded here only for provenance/tests
against raw normalized network tensors; they are never applied to the
saved on-disk arrays.

No sample selection is ever influenced by image appearance -- the
selection stage is purely CSV-driven and produces identical output whether
or not raw arrays are available.

Determinism: no wall-clock time, hostname, or environment-dependent value
is ever written to a generated file. Running --selection-only twice
produces byte-identical CSVs, Markdown, and log.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import math
import os
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
os.chdir(REPO_ROOT)

# -----------------------------------------------------------------------
# Frozen-input directories (Phase 1 / 2A / 2B / 2C)
# -----------------------------------------------------------------------
PHASE1_DIR = REPO_ROOT / 'ttk_runs_fixed' / 'unified_candidate_evaluation'
PHASE2A_DIR = REPO_ROOT / 'ttk_runs_fixed' / 'unified_candidate_analysis' / 'phase2a'
PHASE2B_DIR = REPO_ROOT / 'ttk_runs_fixed' / 'unified_candidate_analysis' / 'phase2b'
PHASE2C_DIR = REPO_ROOT / 'ttk_runs_fixed' / 'unified_candidate_analysis' / 'phase2c'

# -----------------------------------------------------------------------
# Phase-2D-A output locations (the only locations this script may write)
# -----------------------------------------------------------------------
OUT_DIR = REPO_ROOT / 'ttk_runs_fixed' / 'unified_candidate_analysis' / 'phase2d'
SELECTION_DIR = OUT_DIR / 'selection'
PREVIEW_AUDIT_DIR = OUT_DIR / 'preview_audit'
PREVIEWS_DIR = OUT_DIR / 'previews'
DOCS_DIR = REPO_ROOT / 'docs'
DOC_PATH = DOCS_DIR / 'unified_candidate_analysis_phase2d.md'
SELECTION_LOG_PATH = REPO_ROOT / 'logs' / 'unified_candidate_analysis_phase2d_selection.log'
RENDER_LOG_PATH = REPO_ROOT / 'logs' / 'unified_candidate_analysis_phase2d_render.log'

# -----------------------------------------------------------------------
# Explicit protected-file sets -- never a glob. 12 + 14 + 28 + 32 = 86.
# -----------------------------------------------------------------------
PHASE1_PROTECTED_CSV_NAMES = [
    'column_mapping.csv', 'method_inventory.csv', 'unified_primary_method_summary.csv',
    'unified_primary_missingness.csv', 'unified_primary_pairwise_vs_cnn.csv',
    'unified_primary_per_sample_long.csv', 'unified_primary_topology_validation.csv',
    'unified_primary_wide.csv',
]
PHASE1_PROTECTED_CSVS = [PHASE1_DIR / n for n in PHASE1_PROTECTED_CSV_NAMES]
PHASE1_PROTECTED_DOCS = [
    REPO_ROOT / 'docs' / 'unified_candidate_evaluation_phase1.md',
    REPO_ROOT / 'docs' / 'unified_candidate_evaluation_inventory.md',
    REPO_ROOT / 'docs' / 'primary_candidate_artifact_reference.md',
    REPO_ROOT / 'logs' / 'build_unified_candidate_evaluation.log',
]
PHASE1_PROTECTED_FILES = PHASE1_PROTECTED_CSVS + PHASE1_PROTECTED_DOCS  # exactly 12

PHASE2A_PROTECTED_CSV_NAMES = [
    'phase2a_validation.csv', 'metric_coverage.csv', 'method_descriptive_summary.csv',
    'paired_vs_cnn_detailed.csv', 'paired_vs_cnn_adjusted.csv', 'method_mean_improvement_matrix.csv',
    'method_win_rate_matrix.csv', 'topology_tradeoff_summary.csv', 'topology_tradeoff_summary_sorted.csv',
    'phase1_pairwise_reproduction.csv', 'phase1_immutability_check.csv',
]
PHASE2A_PROTECTED_CSVS = [PHASE2A_DIR / n for n in PHASE2A_PROTECTED_CSV_NAMES]
PHASE2A_PROTECTED_OTHER = [
    REPO_ROOT / 'docs' / 'unified_candidate_analysis_phase2a.md',
    REPO_ROOT / 'logs' / 'unified_candidate_analysis_phase2a.log',
    REPO_ROOT / 'scripts' / 'analyze_unified_candidate_metrics_phase2a.py',
]
PHASE2A_PROTECTED_FILES = PHASE2A_PROTECTED_CSVS + PHASE2A_PROTECTED_OTHER  # exactly 14

PHASE2B_PROTECTED_CSV_NAMES = [
    'phase2b_validation.csv', 'prior_phase_immutability_check.csv',
    'b_factorial_design.csv', 'b_factorial_cell_summary.csv', 'b_factorial_sample_effects.csv',
    'b_factorial_effect_summary.csv', 'b_factorial_reconstruction_check.csv',
    'b_factorial_oriented_effect_matrix.csv',
    'b_scaffold_crit_e2_design.csv', 'b_scaffold_crit_e2_sample_effects.csv',
    'b_scaffold_crit_e2_effect_summary.csv', 'b_scaffold_crit_e2_reconstruction_check.csv',
    'b_scaffold_crit_e2_oriented_effect_matrix.csv',
    'grad_scaffold_levelset_e2_design.csv', 'grad_scaffold_levelset_e2_sample_effects.csv',
    'grad_scaffold_levelset_e2_effect_summary.csv', 'grad_scaffold_levelset_e2_reconstruction_check.csv',
    'grad_scaffold_levelset_e2_oriented_effect_matrix.csv',
    'targeted_contrast_manifest.csv', 'targeted_contrast_per_sample.csv', 'targeted_contrast_summary.csv',
    'targeted_contrast_oriented_effect_matrix.csv',
    'phase2b_multiple_testing_adjusted.csv', 'topology_factorial_and_contrast_summary.csv',
]
PHASE2B_PROTECTED_CSVS = [PHASE2B_DIR / n for n in PHASE2B_PROTECTED_CSV_NAMES]
PHASE2B_PROTECTED_OTHER = [
    REPO_ROOT / 'docs' / 'unified_candidate_analysis_phase2b.md',
    REPO_ROOT / 'logs' / 'unified_candidate_analysis_phase2b.log',
    REPO_ROOT / 'scripts' / 'analyze_unified_candidate_factors_phase2b.py',
    REPO_ROOT / 'scripts' / 'test_analyze_unified_candidate_factors_phase2b.py',
]
PHASE2B_PROTECTED_FILES = PHASE2B_PROTECTED_CSVS + PHASE2B_PROTECTED_OTHER  # exactly 28

PHASE2C_PROTECTED_CSV_NAMES = [
    'focal_topology_correlation_bootstrap.csv', 'focal_topology_relationship_summary.csv',
    'method_level_metric_correlations.csv', 'method_level_oriented_pearson_matrix.csv',
    'method_level_oriented_spearman_matrix.csv', 'method_mean_oriented_values.csv',
    'metric_relationship_summary.csv', 'pareto_bootstrap_front_size.csv',
    'pareto_bootstrap_stability.csv', 'pareto_dominance_edges.csv', 'pareto_front_membership.csv',
    'pareto_layers.csv', 'pareto_objective_manifest.csv', 'phase2c_validation.csv',
    'prior_phase_immutability_check.csv', 'samplewise_correlation_summary.csv',
    'samplewise_cross_method_correlations.csv', 'topology_descriptor_disagreement_summary.csv',
    'topology_pairwise_preference_agreement.csv', 'topology_pareto_sanity_check.csv',
    'topology_rank_by_method.csv', 'topology_relationship_and_pareto_summary.csv',
    'topology_sample_preference_agreement.csv', 'two_way_residual_correlations.csv',
    'two_way_residual_pearson_matrix.csv', 'two_way_residual_spearman_matrix.csv',
    'within_method_correlation_summary.csv', 'within_method_metric_correlations.csv',
]
PHASE2C_PROTECTED_CSVS = [PHASE2C_DIR / n for n in PHASE2C_PROTECTED_CSV_NAMES]
PHASE2C_PROTECTED_OTHER = [
    REPO_ROOT / 'docs' / 'unified_candidate_analysis_phase2c.md',
    REPO_ROOT / 'logs' / 'unified_candidate_analysis_phase2c.log',
    REPO_ROOT / 'scripts' / 'analyze_unified_candidate_relationships_phase2c.py',
    REPO_ROOT / 'scripts' / 'test_analyze_unified_candidate_relationships_phase2c.py',
]
PHASE2C_PROTECTED_FILES = PHASE2C_PROTECTED_CSVS + PHASE2C_PROTECTED_OTHER  # exactly 32

ALL_PROTECTED_FILES = (PHASE1_PROTECTED_FILES + PHASE2A_PROTECTED_FILES +
                         PHASE2B_PROTECTED_FILES + PHASE2C_PROTECTED_FILES)  # exactly 86
assert len(PHASE1_PROTECTED_FILES) == 12
assert len(PHASE2A_PROTECTED_FILES) == 14
assert len(PHASE2B_PROTECTED_FILES) == 28
assert len(PHASE2C_PROTECTED_FILES) == 32
assert len(ALL_PROTECTED_FILES) == 86

PROTECTED_DIRS_AND_CSVS = (
    (PHASE1_DIR, set(PHASE1_PROTECTED_CSVS)),
    (PHASE2A_DIR, set(PHASE2A_PROTECTED_CSVS)),
    (PHASE2B_DIR, set(PHASE2B_PROTECTED_CSVS)),
    (PHASE2C_DIR, set(PHASE2C_PROTECTED_CSVS)),
)

N_EVAL = 168
N_METHODS = 19
CNN_METHOD = 'cnn'
GAN_METHOD = 'gan'
BICUBIC_METHOD = 'bicubic'
CANDIDATE_C_METHOD = 'candidate_c'
F2_METHOD = 'f2_grad_levelset_e2'
F3_METHOD = 'f3_grad_crit'
UV_E2_METHOD = 'uv_e2'
TIE_TOLERANCE = 1e-12
TOPOLOGY_METRICS = ('pd_distance', 'mt_distance')

# Raw-array expected shapes (documented convention; enforced only in
# --render-previews / --full).
EXPECTED_IN_SHAPE = (N_EVAL, 100, 100, 2)
EXPECTED_HR_SHAPE = (N_EVAL, 500, 500, 2)

# MU_SIG: provenance-only normalization constants for raw normalized network
# tensors (mu=[u_mean, v_mean], sig=[u_std, v_std]). NEVER applied to the
# saved on-disk dataIN/dataGT/dataSR arrays, which are already physical.
MU_SIG = ([0.7684, -0.4575], [5.02455, 5.9017])

# Method groups for Phase-2D-B figure/preview planning.
BASELINE_STORY = [BICUBIC_METHOD, CNN_METHOD, GAN_METHOD, CANDIDATE_C_METHOD]
DESCRIPTOR_TRADEOFF_STORY = [CNN_METHOD, F3_METHOD, F2_METHOD, UV_E2_METHOD]
FULL_SELECTED_STORY = [BICUBIC_METHOD, CNN_METHOD, GAN_METHOD, CANDIDATE_C_METHOD,
                         F3_METHOD, F2_METHOD, UV_E2_METHOD]
assert len(FULL_SELECTED_STORY) == 7

_LOG_LINES: list = []


def log(msg: str = '') -> None:
    print(msg)
    _LOG_LINES.append(msg)


def flush_log(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as fh:
        fh.write('\n'.join(_LOG_LINES) + '\n')


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def require_protected_files() -> None:
    missing = [str(p) for p in ALL_PROTECTED_FILES if not p.exists()]
    if missing:
        raise SystemExit(
            f'[hard-fail] Missing required prior-phase protected file(s) (expected exactly '
            f'{len(ALL_PROTECTED_FILES)}: 12 Phase-1 + 14 Phase-2A + 28 Phase-2B + 32 Phase-2C):\n' +
            '\n'.join(f'  - {m}' for m in missing)
        )
    for directory, expected_csvs in PROTECTED_DIRS_AND_CSVS:
        actual_csvs = sorted(directory.glob('*.csv'), key=str)
        unexpected = [str(p) for p in actual_csvs if p not in expected_csvs]
        if unexpected:
            raise SystemExit(
                f'[hard-fail] Unexpected extra CSV(s) found in frozen directory {directory} '
                f'(schema is intended to be immutable): {unexpected}'
            )


def checksum_all(files: list) -> dict:
    result = {}
    for p in files:
        rel = p.resolve().relative_to(REPO_ROOT).as_posix()
        result[rel] = sha256_file(p) if p.exists() else None
    return result


def preflight_immutability():
    require_protected_files()
    file_to_phase = {p.resolve().relative_to(REPO_ROOT).as_posix(): 'phase1' for p in PHASE1_PROTECTED_FILES}
    file_to_phase.update({p.resolve().relative_to(REPO_ROOT).as_posix(): 'phase2a'
                            for p in PHASE2A_PROTECTED_FILES})
    file_to_phase.update({p.resolve().relative_to(REPO_ROOT).as_posix(): 'phase2b'
                            for p in PHASE2B_PROTECTED_FILES})
    file_to_phase.update({p.resolve().relative_to(REPO_ROOT).as_posix(): 'phase2c'
                            for p in PHASE2C_PROTECTED_FILES})
    checksums_before = checksum_all(ALL_PROTECTED_FILES)
    log(f'[immutability] Checksummed {len(checksums_before)} prior-phase file(s) before this stage '
        f'(12 Phase-1 + 14 Phase-2A + 28 Phase-2B + 32 Phase-2C = 86 exactly).')
    return checksums_before, file_to_phase


def postflight_immutability(checksums_before, file_to_phase, out_path: Path):
    require_protected_files()
    checksums_after = checksum_all(ALL_PROTECTED_FILES)
    immut_rows = []
    changed = []
    for path_str, before in sorted(checksums_before.items()):
        after = checksums_after.get(path_str)
        status = 'unchanged' if after == before else 'CHANGED'
        if after is None:
            status = 'MISSING_AFTER_RUN'
        if status != 'unchanged':
            changed.append(path_str)
        immut_rows.append(dict(phase=file_to_phase.get(path_str, 'unknown'), file_path=path_str,
                                  sha256_before=(before or ''), sha256_after=(after or ''), status=status))
    for path_str in sorted(set(checksums_after) - set(checksums_before)):
        immut_rows.append(dict(phase=file_to_phase.get(path_str, 'unknown'), file_path=path_str,
                                  sha256_before='', sha256_after=checksums_after[path_str],
                                  status='NEW_FILE_APPEARED'))
        changed.append(path_str)
    write_csv(out_path, ['phase', 'file_path', 'sha256_before', 'sha256_after', 'status'], immut_rows)
    if changed:
        raise SystemExit(f'[hard-fail] Prior-phase immutability violated: {changed}')
    log(f'[immutability] Confirmed all {len(immut_rows)} prior-phase file(s) unchanged.')
    return immut_rows


def read_csv_dicts(path: Path) -> list:
    with path.open(newline='') as fh:
        return list(csv.DictReader(fh))


def _f(val):
    if val in (None, ''):
        return float('nan')
    return float(val)


def nfmt(v):
    return '' if v is None else v


def _rel_posix(path: Path, base_dir) -> str:
    return Path(path).resolve().relative_to(Path(base_dir).resolve()).as_posix()


def _require_no_absolute_csv_paths(path: Path, fieldnames: list, rows: list) -> None:
    """Every path written to a Phase-2D CSV must be repository-relative POSIX
    text; absolute paths may appear only in console diagnostics, never in a
    generated scientific artifact. Applies to every field whose name
    contains 'path' (case-insensitive) -- e.g. `path`, `output_path`,
    `expected_repo_relative_path`, `topology_csv_path`, `source_file`."""
    path_like_fields = [f for f in fieldnames if 'path' in f.lower()]
    if not path_like_fields:
        return
    for row in rows:
        for f in path_like_fields:
            v = row.get(f, '')
            if isinstance(v, str) and v.startswith('/'):
                raise SystemExit(
                    f'[hard-fail] Absolute path found in generated CSV field {f!r} of {path}: {v!r}. '
                    f'All generated-artifact path fields must be repository-relative POSIX text.'
                )


def write_csv(path: Path, fieldnames: list, rows: list) -> None:
    _require_no_absolute_csv_paths(path, fieldnames, rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    log(f'[write] {path} ({len(rows)} rows)')


# =============================================================================
# Loaders (CSV-only; never touch raw .npy arrays)
# =============================================================================

def load_long_table():
    path = PHASE1_DIR / 'unified_primary_per_sample_long.csv'
    with path.open(newline='') as fh:
        header = next(csv.reader(fh))
    identity_cols = ['sample_idx', 'method_id', 'display_name', 'candidate_family', 'training_scale',
                      'architecture', 'uses_speed', 'uses_grad', 'uses_levelset', 'uses_crit', 'uses_e2']
    metric_cols = [c for c in header if c not in identity_cols]
    rows = read_csv_dicts(path)
    per_sample = {}
    dup_keys = []
    seen_keys = set()
    for row in rows:
        mid = row['method_id']
        si = int(row['sample_idx'])
        key = (mid, si)
        if key in seen_keys:
            dup_keys.append(key)
            continue
        seen_keys.add(key)
        per_sample.setdefault(mid, {})[si] = {c: _f(row.get(c, '')) for c in metric_cols}
    return dict(rows=rows, metric_cols=metric_cols, per_sample=per_sample, dup_keys=dup_keys, n_rows=len(rows))


def load_column_mapping():
    rows = read_csv_dicts(PHASE1_DIR / 'column_mapping.csv')
    direction = {}
    for row in rows:
        sc = row['standardized_column']
        d = row['direction']
        if sc in direction and direction[sc] != d:
            raise SystemExit(f'[hard-fail] column_mapping.csv gives conflicting directions for {sc!r}.')
        direction[sc] = d
    return direction


def load_method_inventory():
    rows = read_csv_dicts(PHASE1_DIR / 'method_inventory.csv')
    by_id = {}
    for row in rows:
        by_id[row['method_id']] = row
    return by_id


def load_phase1_refs():
    return dict(
        method_summary=read_csv_dicts(PHASE1_DIR / 'unified_primary_method_summary.csv'),
        topo_val=read_csv_dicts(PHASE1_DIR / 'unified_primary_topology_validation.csv'),
    )


def load_phase2a_refs():
    return dict(validation=read_csv_dicts(PHASE2A_DIR / 'phase2a_validation.csv'))


def load_phase2b_refs():
    return dict(validation=read_csv_dicts(PHASE2B_DIR / 'phase2b_validation.csv'))


def load_phase2c_refs():
    return dict(
        validation=read_csv_dicts(PHASE2C_DIR / 'phase2c_validation.csv'),
        immutability=read_csv_dicts(PHASE2C_DIR / 'prior_phase_immutability_check.csv'),
        samplewise=read_csv_dicts(PHASE2C_DIR / 'samplewise_cross_method_correlations.csv'),
        sample_pref=read_csv_dicts(PHASE2C_DIR / 'topology_sample_preference_agreement.csv'),
        pairwise_pref=read_csv_dicts(PHASE2C_DIR / 'topology_pairwise_preference_agreement.csv'),
        rank_by_method=read_csv_dicts(PHASE2C_DIR / 'topology_rank_by_method.csv'),
        front_membership=read_csv_dicts(PHASE2C_DIR / 'pareto_front_membership.csv'),
        bootstrap_stability=read_csv_dicts(PHASE2C_DIR / 'pareto_bootstrap_stability.csv'),
        relationship_and_pareto_summary=read_csv_dicts(PHASE2C_DIR / 'topology_relationship_and_pareto_summary.csv'),
    )


def load_phase2a2b_immutability_ledgers():
    return dict(
        phase2a=read_csv_dicts(PHASE2A_DIR / 'phase1_immutability_check.csv'),
        phase2b=read_csv_dicts(PHASE2B_DIR / 'prior_phase_immutability_check.csv'),
    )


def load_column_mapping_rows():
    return read_csv_dicts(PHASE1_DIR / 'column_mapping.csv')


# =============================================================================
# Topology-CSV source resolution. These are ordinary git-tracked CSV files
# (NOT gitignored raw .npy arrays) and are read here as plain provenance-
# verification text files -- this does not violate the "must not touch raw
# .npy arrays" constraint on --selection-only. Absolute paths recorded in
# column_mapping.csv/method_inventory.csv are historical-machine provenance
# only; the portable repo-relative suffix is recovered by splitting on the
# repo directory-name marker, never by assuming any specific machine's home
# directory.
# =============================================================================

def repo_relative_from_provenance_path(abs_path_str: str) -> str:
    marker = '/PhIRE/'
    if marker in abs_path_str:
        return abs_path_str.split(marker, 1)[1]
    return abs_path_str


TOPOLOGY_BEARING_FULL_STORY = [CNN_METHOD, GAN_METHOD, CANDIDATE_C_METHOD, F3_METHOD, F2_METHOD, UV_E2_METHOD]


def build_topology_source_map(column_mapping_rows):
    """One entry per topology-bearing full_selected_story method_id:
    dict(path=Path, pd_column=str, mt_column=str, is_shared_combined=bool)."""
    source_map = {}
    for mid in TOPOLOGY_BEARING_FULL_STORY:
        marker = f"method_id='{mid}'"
        pd_row = next((r for r in column_mapping_rows
                        if r['standardized_column'] == 'pd_distance' and marker in r['notes']), None)
        mt_row = next((r for r in column_mapping_rows
                        if r['standardized_column'] == 'mt_distance' and marker in r['notes']), None)
        if pd_row is None or mt_row is None:
            raise SystemExit(f'[hard-fail] column_mapping.csv has no topology row for method_id={mid!r}.')
        rel = repo_relative_from_provenance_path(pd_row['source_path'])
        path = REPO_ROOT / rel
        source_map[mid] = dict(
            path=path, pd_column=pd_row['source_column'], mt_column=mt_row['source_column'],
            is_shared_combined=(path.name == 'phase_c_results.csv'),
        )
    return source_map


def read_topology_pd_mt_for_method(method_id: str, source_map: dict):
    """Returns {sample_idx: (pd, mt)} read directly from the raw topology
    CSV for method_id (not from any Phase-2C derived artifact)."""
    entry = source_map[method_id]
    path = entry['path']
    if not path.exists():
        raise SystemExit(f'[hard-fail] Topology source CSV missing for method_id={method_id!r}: {path}')
    result = {}
    if entry['is_shared_combined']:
        import re
        with path.open(newline='') as fh:
            for row in csv.DictReader(fh):
                if row.get('method') != method_id:
                    continue
                m = re.search(r'_s(\d+)_', row['key'])
                if not m:
                    continue
                si = int(m.group(1))
                result[si] = (_f(row['pd_distance']), _f(row['mt_distance']))
    else:
        with path.open(newline='') as fh:
            for row in csv.DictReader(fh):
                si = int(row['sample_idx'])
                result[si] = (_f(row[entry['pd_column']]), _f(row[entry['mt_column']]))
    return result


# =============================================================================
# Robust-z scoring (median / 1.4826*MAD, falling back to std, then zero)
# =============================================================================

def robust_z(values):
    """Returns (z: np.ndarray, method: 'mad'|'std_fallback'|'zero_contribution').
    If both MAD and std are zero (a degenerate constant component), the
    component contributes exactly zero to every sample's score, and this is
    reported via the returned method tag for the audit trail."""
    x = np.asarray(values, dtype=np.float64)
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    scale = 1.4826 * mad
    if scale > 0:
        return (x - med) / scale, 'mad'
    std = float(np.std(x))
    if std > 0:
        return (x - med) / std, 'std_fallback'
    return np.zeros_like(x), 'zero_contribution'


def orient(y, direction: str):
    return y if direction == 'higher_is_better' else -y


def paired_improvement(base_val, comparison_val, direction: str):
    """Positive always means comparison is better than base."""
    if direction == 'higher_is_better':
        return comparison_val - base_val
    return base_val - comparison_val


# =============================================================================
# Six archetype definitions (exact priority order controls deterministic
# de-duplication). Scores are used ONLY for deterministic sample selection,
# never as a method ranking.
# =============================================================================

ARCHETYPE_PRIORITY = [
    'global_descriptor_disagreement',
    'gan_pd_vs_cnn_mt_conflict',
    'f3_pd_vs_uv_e2_mt_tradeoff',
    'f2_balanced_vs_cnn',
    'candidate_c_continuity',
    'global_descriptor_agreement',
]


def _make_result(archetype_id, priority, eligible_idxs, components, score, score_formula,
                   primary_methods, narrative_scope, metrics_used, methods_required):
    return dict(
        archetype_id=archetype_id, priority=priority, eligible_idxs=list(eligible_idxs),
        components=components, score=np.asarray(score, dtype=np.float64),
        score_formula=score_formula, primary_methods=primary_methods, narrative_scope=narrative_scope,
        metrics_used=metrics_used, methods_required=methods_required,
        tie_break_fields='score desc, then sample_idx asc',
    )


def archetype_a1_global_disagreement(per_sample, sample_pref_rows, samplewise_rows):
    eligible = list(range(N_EVAL))
    agree_by_idx = {int(r['sample_idx']): float(r['agreement_rate']) for r in sample_pref_rows}
    pdmt_by_idx = {}
    for r in samplewise_rows:
        if {r['metric_a'], r['metric_b']} == {'pd_distance', 'mt_distance'}:
            pdmt_by_idx[int(r['sample_idx'])] = (float(r['oriented_pearson']), float(r['oriented_spearman']))
    raw_agree = np.array([-agree_by_idx[si] for si in eligible])
    raw_spearman = np.array([-pdmt_by_idx[si][1] for si in eligible])
    raw_pearson = np.array([-pdmt_by_idx[si][0] for si in eligible])
    z_agree, m_agree = robust_z(raw_agree)
    z_spearman, m_spearman = robust_z(raw_spearman)
    z_pearson, m_pearson = robust_z(raw_pearson)
    score = (z_agree + z_spearman + z_pearson) / 3.0
    components = [
        dict(name='neg_agreement_rate', raw=raw_agree, z=z_agree, z_method=m_agree),
        dict(name='neg_oriented_pd_mt_spearman', raw=raw_spearman, z=z_spearman, z_method=m_spearman),
        dict(name='neg_oriented_pd_mt_pearson', raw=raw_pearson, z=z_pearson, z_method=m_pearson),
    ]
    return _make_result(
        'global_descriptor_disagreement', 1, eligible, components, score,
        score_formula='mean(robust_z(-agreement_rate), robust_z(-oriented_spearman), robust_z(-oriented_pearson))',
        primary_methods='all 18 topology-bearing methods (cross-method preference agreement)',
        narrative_scope='PD/MT descriptor disagreement across methods',
        metrics_used='pd_distance, mt_distance (via Phase-2C topology_sample_preference_agreement.csv and '
                      'samplewise_cross_method_correlations.csv)',
        methods_required=','.join(FULL_SELECTED_STORY),
    )


def archetype_a6_global_agreement(per_sample, sample_pref_rows, samplewise_rows):
    eligible = list(range(N_EVAL))
    agree_by_idx = {int(r['sample_idx']): float(r['agreement_rate']) for r in sample_pref_rows}
    pdmt_by_idx = {}
    for r in samplewise_rows:
        if {r['metric_a'], r['metric_b']} == {'pd_distance', 'mt_distance'}:
            pdmt_by_idx[int(r['sample_idx'])] = (float(r['oriented_pearson']), float(r['oriented_spearman']))
    raw_agree = np.array([agree_by_idx[si] for si in eligible])
    raw_spearman = np.array([pdmt_by_idx[si][1] for si in eligible])
    raw_pearson = np.array([pdmt_by_idx[si][0] for si in eligible])
    z_agree, m_agree = robust_z(raw_agree)
    z_spearman, m_spearman = robust_z(raw_spearman)
    z_pearson, m_pearson = robust_z(raw_pearson)
    score = (z_agree + z_spearman + z_pearson) / 3.0
    components = [
        dict(name='agreement_rate', raw=raw_agree, z=z_agree, z_method=m_agree),
        dict(name='oriented_pd_mt_spearman', raw=raw_spearman, z=z_spearman, z_method=m_spearman),
        dict(name='oriented_pd_mt_pearson', raw=raw_pearson, z=z_pearson, z_method=m_pearson),
    ]
    return _make_result(
        'global_descriptor_agreement', 6, eligible, components, score,
        score_formula='mean(robust_z(agreement_rate), robust_z(oriented_spearman), robust_z(oriented_pearson))',
        primary_methods='all 18 topology-bearing methods (cross-method preference agreement)',
        narrative_scope='PD/MT descriptor agreement across methods',
        metrics_used='pd_distance, mt_distance (via Phase-2C topology_sample_preference_agreement.csv and '
                      'samplewise_cross_method_correlations.csv)',
        methods_required=','.join(FULL_SELECTED_STORY),
    )


def archetype_a2_gan_cnn_conflict(per_sample):
    gan_pd = np.array([per_sample[GAN_METHOD][si]['pd_distance'] for si in range(N_EVAL)])
    cnn_pd = np.array([per_sample[CNN_METHOD][si]['pd_distance'] for si in range(N_EVAL)])
    gan_mt = np.array([per_sample[GAN_METHOD][si]['mt_distance'] for si in range(N_EVAL)])
    cnn_mt = np.array([per_sample[CNN_METHOD][si]['mt_distance'] for si in range(N_EVAL)])
    eligible = [si for si in range(N_EVAL) if gan_pd[si] < cnn_pd[si] and cnn_mt[si] < gan_mt[si]]
    if not eligible:
        raise SystemExit('[hard-fail] archetype gan_pd_vs_cnn_mt_conflict has zero eligible samples.')
    idx = np.array(eligible)
    raw_pd_imp = cnn_pd[idx] - gan_pd[idx]
    raw_mt_imp = gan_mt[idx] - cnn_mt[idx]
    z_pd, m_pd = robust_z(raw_pd_imp)
    z_mt, m_mt = robust_z(raw_mt_imp)
    score = (z_pd + z_mt) / 2.0
    components = [
        dict(name='gan_pd_improvement_vs_cnn', raw=raw_pd_imp, z=z_pd, z_method=m_pd),
        dict(name='cnn_mt_improvement_vs_gan', raw=raw_mt_imp, z=z_mt, z_method=m_mt),
    ]
    return _make_result(
        'gan_pd_vs_cnn_mt_conflict', 2, eligible, components, score,
        score_formula='mean(robust_z(gan_pd_improvement_vs_cnn), robust_z(cnn_mt_improvement_vs_gan))',
        primary_methods='gan, cnn', narrative_scope='The GAN-versus-CNN topology tradeoff',
        metrics_used='pd_distance, mt_distance', methods_required='gan,cnn',
    )


def archetype_a3_f3_uv_e2_tradeoff(per_sample):
    f3_pd = np.array([per_sample[F3_METHOD][si]['pd_distance'] for si in range(N_EVAL)])
    uv_pd = np.array([per_sample[UV_E2_METHOD][si]['pd_distance'] for si in range(N_EVAL)])
    f3_mt = np.array([per_sample[F3_METHOD][si]['mt_distance'] for si in range(N_EVAL)])
    uv_mt = np.array([per_sample[UV_E2_METHOD][si]['mt_distance'] for si in range(N_EVAL)])
    eligible = [si for si in range(N_EVAL) if f3_pd[si] < uv_pd[si] and uv_mt[si] < f3_mt[si]]
    if not eligible:
        raise SystemExit('[hard-fail] archetype f3_pd_vs_uv_e2_mt_tradeoff has zero eligible samples.')
    idx = np.array(eligible)
    raw_pd_imp = uv_pd[idx] - f3_pd[idx]
    raw_mt_imp = f3_mt[idx] - uv_mt[idx]
    z_pd, m_pd = robust_z(raw_pd_imp)
    z_mt, m_mt = robust_z(raw_mt_imp)
    score = (z_pd + z_mt) / 2.0
    components = [
        dict(name='f3_pd_improvement_vs_uv_e2', raw=raw_pd_imp, z=z_pd, z_method=m_pd),
        dict(name='uv_e2_mt_improvement_vs_f3', raw=raw_mt_imp, z=z_mt, z_method=m_mt),
    ]
    return _make_result(
        'f3_pd_vs_uv_e2_mt_tradeoff', 3, eligible, components, score,
        score_formula='mean(robust_z(f3_pd_improvement_vs_uv_e2), robust_z(uv_e2_mt_improvement_vs_f3))',
        primary_methods='f3_grad_crit, uv_e2', narrative_scope='The F3-versus-UV+E2 PD/MT tradeoff',
        metrics_used='pd_distance, mt_distance', methods_required='f3_grad_crit,uv_e2',
    )


def archetype_a4_f2_balanced(per_sample):
    f2_pd = np.array([per_sample[F2_METHOD][si]['pd_distance'] for si in range(N_EVAL)])
    cnn_pd = np.array([per_sample[CNN_METHOD][si]['pd_distance'] for si in range(N_EVAL)])
    f2_mt = np.array([per_sample[F2_METHOD][si]['mt_distance'] for si in range(N_EVAL)])
    cnn_mt = np.array([per_sample[CNN_METHOD][si]['mt_distance'] for si in range(N_EVAL)])
    f2_psnr = np.array([per_sample[F2_METHOD][si]['psnruv'] for si in range(N_EVAL)])
    cnn_psnr = np.array([per_sample[CNN_METHOD][si]['psnruv'] for si in range(N_EVAL)])
    f2_speed = np.array([per_sample[F2_METHOD][si]['speed_mae'] for si in range(N_EVAL)])
    cnn_speed = np.array([per_sample[CNN_METHOD][si]['speed_mae'] for si in range(N_EVAL)])
    eligible = [si for si in range(N_EVAL) if f2_pd[si] < cnn_pd[si] and f2_mt[si] < cnn_mt[si]]
    if not eligible:
        raise SystemExit('[hard-fail] archetype f2_balanced_vs_cnn has zero eligible samples.')
    idx = np.array(eligible)
    raw_pd_imp = cnn_pd[idx] - f2_pd[idx]
    raw_mt_imp = cnn_mt[idx] - f2_mt[idx]
    raw_psnr_imp = f2_psnr[idx] - cnn_psnr[idx]
    raw_speed_imp = cnn_speed[idx] - f2_speed[idx]
    z_pd, m_pd = robust_z(raw_pd_imp)
    z_mt, m_mt = robust_z(raw_mt_imp)
    z_psnr, m_psnr = robust_z(raw_psnr_imp)
    z_speed, m_speed = robust_z(raw_speed_imp)
    primary = np.minimum(z_pd, z_mt)
    secondary = (z_psnr + z_speed) / 2.0
    score = primary + 0.25 * secondary
    components = [
        dict(name='f2_pd_improvement_vs_cnn', raw=raw_pd_imp, z=z_pd, z_method=m_pd),
        dict(name='f2_mt_improvement_vs_cnn', raw=raw_mt_imp, z=z_mt, z_method=m_mt),
        dict(name='f2_psnruv_improvement_vs_cnn', raw=raw_psnr_imp, z=z_psnr, z_method=m_psnr),
        dict(name='f2_speed_mae_improvement_vs_cnn', raw=raw_speed_imp, z=z_speed, z_method=m_speed),
    ]
    return _make_result(
        'f2_balanced_vs_cnn', 4, eligible, components, score,
        score_formula='min(robust_z(pd_improvement), robust_z(mt_improvement)) '
                       '+ 0.25*mean(robust_z(psnr_improvement), robust_z(speed_mae_improvement))',
        primary_methods='f2_grad_levelset_e2, cnn', narrative_scope='A balanced F2 improvement over CNN',
        metrics_used='pd_distance, mt_distance, psnruv, speed_mae', methods_required='f2_grad_levelset_e2,cnn',
    )


def archetype_a5_candidate_c_continuity(per_sample):
    c_pd = np.array([per_sample[CANDIDATE_C_METHOD][si]['pd_distance'] for si in range(N_EVAL)])
    cnn_pd = np.array([per_sample[CNN_METHOD][si]['pd_distance'] for si in range(N_EVAL)])
    c_mt = np.array([per_sample[CANDIDATE_C_METHOD][si]['mt_distance'] for si in range(N_EVAL)])
    cnn_mt = np.array([per_sample[CNN_METHOD][si]['mt_distance'] for si in range(N_EVAL)])
    c_psnr = np.array([per_sample[CANDIDATE_C_METHOD][si]['psnruv'] for si in range(N_EVAL)])
    cnn_psnr = np.array([per_sample[CNN_METHOD][si]['psnruv'] for si in range(N_EVAL)])
    c_speed = np.array([per_sample[CANDIDATE_C_METHOD][si]['speed_mae'] for si in range(N_EVAL)])
    cnn_speed = np.array([per_sample[CNN_METHOD][si]['speed_mae'] for si in range(N_EVAL)])
    eligible = [si for si in range(N_EVAL)
                if c_pd[si] < cnn_pd[si]
                and math.isfinite(c_mt[si]) and math.isfinite(c_psnr[si]) and math.isfinite(c_speed[si])]
    if not eligible:
        raise SystemExit('[hard-fail] archetype candidate_c_continuity has zero eligible samples.')
    idx = np.array(eligible)
    raw_pd_imp = cnn_pd[idx] - c_pd[idx]
    raw_mt_imp = cnn_mt[idx] - c_mt[idx]
    raw_psnr_imp = c_psnr[idx] - cnn_psnr[idx]
    raw_speed_imp = cnn_speed[idx] - c_speed[idx]
    z_pd, m_pd = robust_z(raw_pd_imp)
    z_mt, m_mt = robust_z(raw_mt_imp)
    z_psnr, m_psnr = robust_z(raw_psnr_imp)
    z_speed, m_speed = robust_z(raw_speed_imp)
    score = 0.50 * z_pd + 0.20 * z_mt + 0.15 * z_psnr + 0.15 * z_speed
    components = [
        dict(name='candidate_c_pd_improvement_vs_cnn', raw=raw_pd_imp, z=z_pd, z_method=m_pd),
        dict(name='candidate_c_mt_improvement_vs_cnn', raw=raw_mt_imp, z=z_mt, z_method=m_mt),
        dict(name='candidate_c_psnruv_improvement_vs_cnn', raw=raw_psnr_imp, z=z_psnr, z_method=m_psnr),
        dict(name='candidate_c_speed_mae_improvement_vs_cnn', raw=raw_speed_imp, z=z_speed, z_method=m_speed),
    ]
    return _make_result(
        'candidate_c_continuity', 5, eligible, components, score,
        score_formula='0.50*robust_z(pd_improvement) + 0.20*robust_z(mt_improvement) '
                       '+ 0.15*robust_z(psnr_improvement) + 0.15*robust_z(speed_mae_improvement)',
        primary_methods='candidate_c, cnn',
        narrative_scope='Candidate C as continuity with the submitted topology-inspired model',
        metrics_used='pd_distance, mt_distance, psnruv, speed_mae', methods_required='candidate_c,cnn',
    )


def compute_all_archetypes(per_sample, sample_pref_rows, samplewise_rows):
    results = {}
    results['global_descriptor_disagreement'] = archetype_a1_global_disagreement(
        per_sample, sample_pref_rows, samplewise_rows)
    results['gan_pd_vs_cnn_mt_conflict'] = archetype_a2_gan_cnn_conflict(per_sample)
    results['f3_pd_vs_uv_e2_mt_tradeoff'] = archetype_a3_f3_uv_e2_tradeoff(per_sample)
    results['f2_balanced_vs_cnn'] = archetype_a4_f2_balanced(per_sample)
    results['candidate_c_continuity'] = archetype_a5_candidate_c_continuity(per_sample)
    results['global_descriptor_agreement'] = archetype_a6_global_agreement(
        per_sample, sample_pref_rows, samplewise_rows)
    return results


# =============================================================================
# Ranking, unique greedy de-duplication, alternates
# =============================================================================

def rank_eligible(result):
    idxs = result['eligible_idxs']
    scores = result['score']
    score_by_idx = {si: float(scores[i]) for i, si in enumerate(idxs)}
    ranked = sorted(idxs, key=lambda si: (-score_by_idx[si], si))
    return ranked, score_by_idx


def method_metric_mean(per_sample, mid, metric):
    vals = [per_sample[mid][si][metric] for si in range(N_EVAL)
            if math.isfinite(per_sample[mid][si].get(metric, float('nan')))]
    if not vals:
        return None
    return sum(vals) / len(vals)


# =============================================================================
# --selection-only base validation (CSV-only; no raw .npy arrays touched)
# =============================================================================

def run_selection_base_validation(long_table, phase1_refs, phase2a_refs, phase2b_refs, phase2c_refs,
                                     immutability_ledgers, topology_source_map, metric_direction):
    checks = []
    failures = []

    def add(name, observed, expected, ok, notes=''):
        status = 'PASS' if ok else 'FAIL'
        checks.append(dict(check_name=name, observed=observed, expected=expected, status=status, notes=notes))
        if not ok:
            failures.append(f'{name}: observed={observed!r} expected={expected!r} notes={notes}')

    per_sample = long_table['per_sample']
    metric_cols = long_table['metric_cols']

    add('long_table_row_count', long_table['n_rows'], N_METHODS * N_EVAL, long_table['n_rows'] == N_METHODS * N_EVAL)
    add('duplicate_keys', len(long_table['dup_keys']), 0, len(long_table['dup_keys']) == 0)
    add('method_count', len(per_sample), N_METHODS, len(per_sample) == N_METHODS)
    add('metric_count', len(metric_cols), 22, len(metric_cols) == 22)
    for mid in sorted(per_sample):
        n = len(per_sample[mid])
        add(f'samples_per_method[{mid}]', n, N_EVAL, n == N_EVAL)
        idx_set = set(per_sample[mid].keys())
        add(f'sample_idx_exact_0_167[{mid}]', len(idx_set), N_EVAL, idx_set == set(range(N_EVAL)))

    ssim_total_finite = sum(1 for mid in per_sample for si in range(N_EVAL)
                              if math.isfinite(per_sample[mid][si].get('ssim_speed', float('nan'))))
    add('ssim_globally_unavailable', ssim_total_finite, 0, ssim_total_finite == 0)

    n_p2a_fail = sum(1 for r in phase2a_refs['validation'] if r['status'] != 'PASS')
    add('phase2a_validation_all_pass', n_p2a_fail, 0, n_p2a_fail == 0)
    n_p2b_fail = sum(1 for r in phase2b_refs['validation'] if r['status'] != 'PASS')
    add('phase2b_validation_all_pass', n_p2b_fail, 0, n_p2b_fail == 0)
    n_p2c_fail = sum(1 for r in phase2c_refs['validation'] if r['status'] != 'PASS')
    add('phase2c_validation_all_pass', n_p2c_fail, 0, n_p2c_fail == 0)

    n_p2a_immut_bad = sum(1 for r in immutability_ledgers['phase2a'] if r['status'] != 'unchanged')
    add('phase2a_immutability_ledger_all_unchanged', n_p2a_immut_bad, 0, n_p2a_immut_bad == 0)
    n_p2b_immut_bad = sum(1 for r in immutability_ledgers['phase2b'] if r['status'] != 'unchanged')
    add('phase2b_immutability_ledger_all_unchanged', n_p2b_immut_bad, 0, n_p2b_immut_bad == 0)
    n_p2c_immut_bad = sum(1 for r in phase2c_refs['immutability'] if r['status'] != 'unchanged')
    add('phase2c_immutability_ledger_all_unchanged', n_p2c_immut_bad, 0, n_p2c_immut_bad == 0)

    add('topology_sample_preference_table_has_168_rows', len(phase2c_refs['sample_pref']), N_EVAL,
        len(phase2c_refs['sample_pref']) == N_EVAL)
    bad_pair_count = [r for r in phase2c_refs['sample_pref'] if int(r['method_pair_count']) != 153]
    add('topology_sample_rows_have_153_method_pairs', len(bad_pair_count), 0, len(bad_pair_count) == 0,
        notes=str([r['sample_idx'] for r in bad_pair_count][:5]))

    pdmt_lookup = {}
    for r in phase2c_refs['samplewise']:
        if {r['metric_a'], r['metric_b']} == {'pd_distance', 'mt_distance'}:
            pdmt_lookup[int(r['sample_idx'])] = (r['oriented_pearson'], r['oriented_spearman'])
    bad_match = 0
    for r in phase2c_refs['sample_pref']:
        si = int(r['sample_idx'])
        exp_p, exp_s = pdmt_lookup.get(si, ('', ''))
        if r['pd_mt_cross_method_pearson'] != exp_p or r['pd_mt_cross_method_spearman'] != exp_s:
            bad_match += 1
    add('samplewise_pd_mt_matches_phase2c_sample_preference_table', bad_match, 0, bad_match == 0)

    for mid in TOPOLOGY_BEARING_FULL_STORY:
        try:
            rows = read_topology_pd_mt_for_method(mid, topology_source_map)
            n_rows = len(rows)
            has_all = set(rows.keys()) == set(range(N_EVAL))
        except SystemExit as e:
            n_rows = 0
            has_all = False
        add(f'topology_csv_has_168_sample_rows[{mid}]', n_rows, N_EVAL, n_rows == N_EVAL and has_all)

    for mid in TOPOLOGY_BEARING_FULL_STORY:
        rows = read_topology_pd_mt_for_method(mid, topology_source_map)
        observed_pd = sum(v[0] for v in rows.values()) / len(rows)
        observed_mt = sum(v[1] for v in rows.values()) / len(rows)
        lt_pd = method_metric_mean(per_sample, mid, 'pd_distance')
        lt_mt = method_metric_mean(per_sample, mid, 'mt_distance')
        add(f'topology_csv_pd_mean_matches_long_table[{mid}]', observed_pd, lt_pd,
            lt_pd is not None and abs(observed_pd - lt_pd) <= 1e-4, notes=f'diff={abs(observed_pd - (lt_pd or 0)):.3e}')
        add(f'topology_csv_mt_mean_matches_long_table[{mid}]', observed_mt, lt_mt,
            lt_mt is not None and abs(observed_mt - lt_mt) <= 1e-4, notes=f'diff={abs(observed_mt - (lt_mt or 0)):.3e}')

    missing_direction = [c for c in metric_cols if c not in metric_direction]
    add('metric_directions_resolved_all_22', len(missing_direction), 0, len(missing_direction) == 0,
        notes=str(missing_direction))

    add('no_imputation_by_design', True, True, True,
        notes='Selection reads finite values directly from the long table and topology CSVs; no missing '
              'value is ever filled in. Candidate_C eligibility explicitly requires finite MT/PSNR/speed_mae '
              'rather than imputing a value for non-finite cases.')

    return checks, failures


def select_with_dedup(archetype_id, ranked, score_by_idx, already_selected):
    diagnostics = []
    selected = None
    alternates = []
    rank_after = 0
    for rank_before, si in enumerate(ranked, start=1):
        if si in already_selected:
            diagnostics.append(dict(
                archetype_id=archetype_id, event_type='duplicate_skip', sample_idx=si,
                rank_before_dedup=rank_before, rank_after_dedup='',
                detail='already selected by a higher-priority archetype',
            ))
            continue
        rank_after += 1
        if selected is None:
            selected = dict(sample_idx=si, rank_before_dedup=rank_before, rank_after_dedup=rank_after,
                              score=score_by_idx[si])
        else:
            alternates.append(dict(sample_idx=si, rank_before_dedup=rank_before, rank_after_dedup=rank_after,
                                      score=score_by_idx[si]))
            if len(alternates) == 3:
                break
    return selected, alternates, diagnostics


def pref3(v1, v2):
    if abs(v1 - v2) <= TIE_TOLERANCE:
        return 'tie'
    return 'a' if v1 < v2 else 'b'


def run_all_selections(results):
    """Processes archetypes in ARCHETYPE_PRIORITY order, applying unique
    greedy de-duplication. Returns (selection_by_archetype, all_diagnostics,
    ranked_by_archetype, score_by_idx_by_archetype)."""
    already_selected = set()
    selection_by_archetype = {}
    ranked_by_archetype = {}
    score_by_idx_by_archetype = {}
    all_diagnostics = []
    for archetype_id in ARCHETYPE_PRIORITY:
        result = results[archetype_id]
        if not result['eligible_idxs']:
            raise SystemExit(f'[hard-fail] archetype {archetype_id!r} has zero eligible samples.')
        ranked, score_by_idx = rank_eligible(result)
        ranked_by_archetype[archetype_id] = ranked
        score_by_idx_by_archetype[archetype_id] = score_by_idx
        all_diagnostics.append(dict(archetype_id=archetype_id, event_type='eligibility_count', sample_idx='',
                                       rank_before_dedup='', rank_after_dedup='',
                                       detail=f'{len(result["eligible_idxs"])} eligible samples'))
        selected, alternates, diag = select_with_dedup(archetype_id, ranked, score_by_idx, already_selected)
        all_diagnostics.extend(diag)
        global_archetypes = ('global_descriptor_disagreement', 'global_descriptor_agreement')
        population = 'full 168-sample set' if archetype_id in global_archetypes else 'eligible subset'
        for comp in result['components']:
            if comp['z_method'] != 'mad':
                all_diagnostics.append(dict(
                    archetype_id=archetype_id, event_type='robust_z_fallback', sample_idx='',
                    rank_before_dedup='', rank_after_dedup='',
                    detail=(f"component {comp['name']!r} used z_method={comp['z_method']!r} "
                             f"(MAD was zero across the {population})"),
                ))
        selection_by_archetype[archetype_id] = dict(selected=selected, alternates=alternates)
        already_selected.add(selected['sample_idx'])
    return selection_by_archetype, all_diagnostics, ranked_by_archetype, score_by_idx_by_archetype


# =============================================================================
# Selection-only output builders (12 CSVs under phase2d/selection/)
# =============================================================================

_COMPONENT_SLOTS = 4


def build_archetype_score_table(results, ranked_by_archetype, score_by_idx_by_archetype, top_n=10):
    rows = []
    for archetype_id in ARCHETYPE_PRIORITY:
        result = results[archetype_id]
        ranked = ranked_by_archetype[archetype_id]
        score_by_idx = score_by_idx_by_archetype[archetype_id]
        idx_pos = {si: i for i, si in enumerate(result['eligible_idxs'])}
        for rank, si in enumerate(ranked[:top_n], start=1):
            pos = idx_pos[si]
            row = dict(archetype_id=archetype_id, priority=result['priority'], rank=rank, sample_idx=si,
                        score=score_by_idx[si], eligibility_count=len(result['eligible_idxs']),
                        score_formula=result['score_formula'])
            for slot in range(1, _COMPONENT_SLOTS + 1):
                if slot <= len(result['components']):
                    comp = result['components'][slot - 1]
                    row[f'c{slot}_name'] = comp['name']
                    row[f'c{slot}_raw'] = float(comp['raw'][pos])
                    row[f'c{slot}_z'] = float(comp['z'][pos])
                    row[f'c{slot}_z_method'] = comp['z_method']
                else:
                    row[f'c{slot}_name'] = ''
                    row[f'c{slot}_raw'] = ''
                    row[f'c{slot}_z'] = ''
                    row[f'c{slot}_z_method'] = ''
            rows.append(row)
    return rows


ARCHETYPE_SELECTED_FIELDS = [
    'archetype_id', 'selected_sample_idx', 'selection_rank', 'primary_or_alternate', 'score',
    'eligibility_status', 'selection_reason', 'methods_required', 'metrics_used', 'tie_break_fields',
]


def build_selected_and_alternates(results, selection_by_archetype):
    selected_rows = []
    alternate_rows = []
    for archetype_id in ARCHETYPE_PRIORITY:
        result = results[archetype_id]
        sel = selection_by_archetype[archetype_id]['selected']
        alts = selection_by_archetype[archetype_id]['alternates']
        selected_rows.append(dict(
            archetype_id=archetype_id, selected_sample_idx=sel['sample_idx'], selection_rank=1,
            primary_or_alternate='primary', score=sel['score'], eligibility_status='eligible',
            selection_reason=(f'Top-ranked eligible sample (rank {sel["rank_before_dedup"]} before '
                                f'de-duplication) not already selected by a higher-priority archetype.'),
            methods_required=result['methods_required'], metrics_used=result['metrics_used'],
            tie_break_fields=result['tie_break_fields'],
        ))
        for alt in alts:
            alternate_rows.append(dict(
                archetype_id=archetype_id, selected_sample_idx=alt['sample_idx'],
                selection_rank=alt['rank_after_dedup'], primary_or_alternate='alternate', score=alt['score'],
                eligibility_status='eligible',
                selection_reason=(f'Non-selected eligible alternate (rank {alt["rank_before_dedup"]} before '
                                    f'de-duplication, rank {alt["rank_after_dedup"]} after) retained for review.'),
                methods_required=result['methods_required'], metrics_used=result['metrics_used'],
                tie_break_fields=result['tie_break_fields'],
            ))
    return selected_rows, alternate_rows


def build_diagnostics_rows(all_diagnostics):
    return [dict(archetype_id=d['archetype_id'], event_type=d['event_type'], sample_idx=d['sample_idx'],
                   rank_before_dedup=d['rank_before_dedup'], rank_after_dedup=d['rank_after_dedup'],
                   detail=d['detail'])
            for d in all_diagnostics]


def build_metric_context_rows(selected_sample_idx_by_archetype, phase2c_refs):
    sample_pref_by_idx = {int(r['sample_idx']): r for r in phase2c_refs['sample_pref']}
    rows = []
    for archetype_id, si in selected_sample_idx_by_archetype.items():
        r = sample_pref_by_idx[si]
        rows.append(dict(
            sample_idx=si, archetype_id=archetype_id, agreement_rate=r['agreement_rate'],
            disagreement_rate=r['disagreement_rate'], tie_or_undefined_count=r['tie_or_undefined_count'],
            method_pair_count=r['method_pair_count'],
            pd_mt_cross_method_pearson=r['pd_mt_cross_method_pearson'],
            pd_mt_cross_method_spearman=r['pd_mt_cross_method_spearman'],
        ))
    rows.sort(key=lambda row: row['sample_idx'])
    return rows


def build_method_values_rows(per_sample, metric_cols, metric_direction, selected_sample_idx_by_archetype,
                                phase2c_refs):
    rank_by_method = {r['method_id']: r for r in phase2c_refs['rank_by_method']}
    topo_summary = {r['method_id']: r for r in phase2c_refs['relationship_and_pareto_summary']}
    front_by_method = {r['method_id']: r['on_front'] for r in phase2c_refs['front_membership']
                         if r['objective_set'] == 'topology_only'}
    rows = []
    for archetype_id in ARCHETYPE_PRIORITY:
        si = selected_sample_idx_by_archetype[archetype_id]
        for mid in FULL_SELECTED_STORY:
            row = dict(sample_idx=si, archetype_id=archetype_id, method_id=mid)
            for metric in metric_cols:
                raw_val = per_sample[mid][si].get(metric, float('nan'))
                row[f'raw__{metric}'] = nfmt(raw_val) if math.isfinite(raw_val) else ''
                if mid == CNN_METHOD:
                    row[f'oriented_improvement_vs_cnn__{metric}'] = ''
                    continue
                cnn_val = per_sample[CNN_METHOD][si].get(metric, float('nan'))
                if math.isfinite(raw_val) and math.isfinite(cnn_val) and metric in metric_direction:
                    row[f'oriented_improvement_vs_cnn__{metric}'] = paired_improvement(
                        cnn_val, raw_val, metric_direction[metric])
                else:
                    row[f'oriented_improvement_vs_cnn__{metric}'] = ''
            rbm = rank_by_method.get(mid)
            row['method_level_pd_rank'] = rbm['pd_rank'] if rbm else ''
            row['method_level_mt_rank'] = rbm['mt_rank'] if rbm else ''
            row['on_topology_only_deterministic_front'] = front_by_method.get(mid, '')
            ts = topo_summary.get(mid)
            row['topology_only_bootstrap_rate_iid'] = ts['topology_only_bootstrap_rate_iid'] if ts else ''
            row['topology_only_bootstrap_rate_block6'] = ts['topology_only_bootstrap_rate_block6'] if ts else ''
            row['topology_only_bootstrap_rate_block12'] = ts['topology_only_bootstrap_rate_block12'] if ts else ''
            row['topology_only_bootstrap_rate_block24'] = ts['topology_only_bootstrap_rate_block24'] if ts else ''
            rows.append(row)
    return rows


def build_method_values_fieldnames(metric_cols):
    fieldnames = ['sample_idx', 'archetype_id', 'method_id']
    for metric in metric_cols:
        fieldnames += [f'raw__{metric}', f'oriented_improvement_vs_cnn__{metric}']
    fieldnames += ['method_level_pd_rank', 'method_level_mt_rank', 'on_topology_only_deterministic_front',
                    'topology_only_bootstrap_rate_iid', 'topology_only_bootstrap_rate_block6',
                    'topology_only_bootstrap_rate_block12', 'topology_only_bootstrap_rate_block24']
    return fieldnames


def build_pairwise_preferences_rows(per_sample, selected_sample_idx_by_archetype):
    rows = []
    for archetype_id in ARCHETYPE_PRIORITY:
        si = selected_sample_idx_by_archetype[archetype_id]
        for m1, m2 in itertools.combinations(TOPOLOGY_BEARING_FULL_STORY, 2):
            pd1, pd2 = per_sample[m1][si]['pd_distance'], per_sample[m2][si]['pd_distance']
            mt1, mt2 = per_sample[m1][si]['mt_distance'], per_sample[m2][si]['mt_distance']
            pd_pref = pref3(pd1, pd2)
            mt_pref = pref3(mt1, mt2)
            agree = (pd_pref == mt_pref) and pd_pref != 'tie'
            rows.append(dict(
                sample_idx=si, archetype_id=archetype_id, method_a=m1, method_b=m2,
                pd_value_a=pd1, pd_value_b=pd2, mt_value_a=mt1, mt_value_b=mt2,
                pd_preferred=({'a': m1, 'b': m2, 'tie': 'tie'}[pd_pref]),
                mt_preferred=({'a': m1, 'b': m2, 'tie': 'tie'}[mt_pref]),
                descriptors_agree=agree,
            ))
    return rows


def build_preview_method_manifest():
    rows = []
    for group_name, methods in (('baseline_story', BASELINE_STORY),
                                  ('descriptor_tradeoff_story', DESCRIPTOR_TRADEOFF_STORY),
                                  ('full_selected_story', FULL_SELECTED_STORY)):
        for position, mid in enumerate(methods, start=1):
            rows.append(dict(group_name=group_name, position=position, method_id=mid,
                                includes_gt_reference=True))
    return rows


def build_preview_plan_rows(selected_sample_idx_by_archetype):
    rows = []
    for archetype_id in ARCHETYPE_PRIORITY:
        si = selected_sample_idx_by_archetype[archetype_id]
        rows.append(dict(
            archetype_id=archetype_id, selected_sample_idx=si, preview_type='combined_speed_and_error_review',
            output_path=f'ttk_runs_fixed/unified_candidate_analysis/phase2d/previews/{archetype_id}/'
                        f'selected_sample_{si}_review.png',
            methods_included='GT,' + ','.join(FULL_SELECTED_STORY),
            panel_count=1 + len(FULL_SELECTED_STORY) + len(FULL_SELECTED_STORY),
            planned_annotations='PD/MT values where available; sample_idx; archetype title; '
                                  '"AUDIT PREVIEW -- NOT FINAL FIGURE" watermark',
            status='pending_raw_artifacts',
        ))
    rows.append(dict(
        archetype_id='all', selected_sample_idx='', preview_type='contact_sheet',
        output_path='ttk_runs_fixed/unified_candidate_analysis/phase2d/previews/'
                    'phase2d_selected_archetypes_contact_sheet.png',
        methods_included='GT,' + ','.join(FULL_SELECTED_STORY),
        panel_count=len(ARCHETYPE_PRIORITY),
        planned_annotations='One row per archetype; archetype id + sample_idx as row label',
        status='pending_raw_artifacts',
    ))
    return rows


def build_raw_artifact_requirements_rows(method_inventory):
    rows = []
    for mid in FULL_SELECTED_STORY:
        inv = method_inventory.get(mid, {})
        original = inv.get('original_method_name', mid)
        if mid == BICUBIC_METHOD:
            rows.append(dict(method_id=mid, original_method_name=original, array_role='idx',
                               expected_repo_relative_path='data_out_fixed/wind_mrhr_cnn/idx.npy',
                               expected_shape=f'({N_EVAL},)', expected_dtype='integer',
                               resolution_convention='reused_from_cnn_canonical_mr_input',
                               notes='Bicubic has no independent inference run; reuses CNN idx/dataIN/dataGT.'))
            rows.append(dict(method_id=mid, original_method_name=original, array_role='dataIN',
                               expected_repo_relative_path='data_out_fixed/wind_mrhr_cnn/dataIN.npy',
                               expected_shape=str(EXPECTED_IN_SHAPE), expected_dtype='float32/float64',
                               resolution_convention='reused_from_cnn_canonical_mr_input', notes=''))
            rows.append(dict(method_id=mid, original_method_name=original, array_role='dataGT',
                               expected_repo_relative_path='data_out_fixed/wind_mrhr_cnn/dataGT.npy',
                               expected_shape=str(EXPECTED_HR_SHAPE), expected_dtype='float32/float64',
                               resolution_convention='reused_from_cnn_canonical_mr_input', notes=''))
            rows.append(dict(method_id=mid, original_method_name=original, array_role='dataSR',
                               expected_repo_relative_path='(reconstructed in memory, not read from disk)',
                               expected_shape=str(EXPECTED_HR_SHAPE), expected_dtype='float32',
                               resolution_convention='scipy.ndimage.zoom(order=3, mode=reflect, prefilter=True) '
                                                       'per channel, applied to CNN dataIN',
                               notes='Never written back to any earlier-phase or new baseline-dataset location.'))
            continue
        if mid in (CNN_METHOD, GAN_METHOD):
            base = f'data_out_fixed/wind_mrhr_{mid}'
        else:
            base = f'data_out/wind_finetune_{original}'
        for role, shape in (('idx', f'({N_EVAL},)'), ('dataIN', str(EXPECTED_IN_SHAPE)),
                              ('dataGT', str(EXPECTED_HR_SHAPE)), ('dataSR', str(EXPECTED_HR_SHAPE))):
            rows.append(dict(method_id=mid, original_method_name=original, array_role=role,
                               expected_repo_relative_path=f'{base}/{role}.npy', expected_shape=shape,
                               expected_dtype='integer' if role == 'idx' else 'float32/float64',
                               resolution_convention=('canonical_fixed_pretrained_baseline'
                                                        if mid in (CNN_METHOD, GAN_METHOD)
                                                        else 'learned_finetuned_candidate'),
                               notes=''))
    return rows


# One planned Phase-2D-B figure per archetype, mapped to whichever existing
# method group (baseline_story / descriptor_tradeoff_story / full_selected_story)
# actually contains that archetype's primary methods. Planning only -- no
# automated TTK/ParaView rendering exists anywhere in this pipeline.
FIGURE_METHOD_GROUP_BY_ARCHETYPE = {
    'global_descriptor_disagreement': 'full_selected_story',
    'gan_pd_vs_cnn_mt_conflict': 'baseline_story',
    'f3_pd_vs_uv_e2_mt_tradeoff': 'descriptor_tradeoff_story',
    'f2_balanced_vs_cnn': 'descriptor_tradeoff_story',
    'candidate_c_continuity': 'baseline_story',
    'global_descriptor_agreement': 'full_selected_story',
}
METHOD_GROUPS_BY_NAME = {
    'baseline_story': BASELINE_STORY,
    'descriptor_tradeoff_story': DESCRIPTOR_TRADEOFF_STORY,
    'full_selected_story': FULL_SELECTED_STORY,
}
assert set(FIGURE_METHOD_GROUP_BY_ARCHETYPE) == set(ARCHETYPE_PRIORITY)


def build_figure_plan_rows(selected_sample_idx_by_archetype, results):
    """Plans (does not render) the six Phase-2D-B figures, one per
    archetype. Rendering itself is out of scope for Phase 2D-A and is never
    claimed here."""
    rows = []
    for i, archetype_id in enumerate(ARCHETYPE_PRIORITY, start=1):
        si = selected_sample_idx_by_archetype[archetype_id]
        r = results[archetype_id]
        group_name = FIGURE_METHOD_GROUP_BY_ARCHETYPE[archetype_id]
        methods = METHOD_GROUPS_BY_NAME[group_name]
        rows.append(dict(
            figure_id=i, figure_title=f'Phase 2D-B figure {i}: {r["narrative_scope"]}',
            archetype_id=archetype_id, selected_sample_idx=si, method_group=group_name,
            methods_included='GT,' + ','.join(methods), narrative_scope=r['narrative_scope'],
            status='planned_not_rendered',
            rendering_note='Figure planning only -- no automated TTK/ParaView rendering exists in this '
                             'pipeline; final publication-quality figure production is manual and explicitly '
                             'deferred to Phase 2D-B.',
        ))
    return rows


def run_selection(long_table, metric_direction, phase1_refs, phase2a_refs, phase2b_refs, phase2c_refs,
                    immutability_ledgers, topology_source_map, method_inventory):
    per_sample = long_table['per_sample']
    metric_cols = long_table['metric_cols']

    base_checks, base_failures = run_selection_base_validation(
        long_table, phase1_refs, phase2a_refs, phase2b_refs, phase2c_refs, immutability_ledgers,
        topology_source_map, metric_direction)
    if base_failures:
        log('')
        log('[SELECTION BASE VALIDATION FAILURE]')
        for f in base_failures:
            log(f'  - {f}')
        raise SystemExit(f'[hard-fail] {len(base_failures)} Phase-2D-A selection base validation check(s) failed.')
    log(f'[validate] All {len(base_checks)} selection base validation checks PASSED.')

    results = compute_all_archetypes(per_sample, phase2c_refs['sample_pref'], phase2c_refs['samplewise'])
    for archetype_id in ARCHETYPE_PRIORITY:
        log(f'[archetype] {archetype_id}: {len(results[archetype_id]["eligible_idxs"])} eligible samples.')

    selection_by_archetype, all_diagnostics, ranked_by_archetype, score_by_idx_by_archetype = \
        run_all_selections(results)

    selected_sample_idx_by_archetype = {a: selection_by_archetype[a]['selected']['sample_idx']
                                          for a in ARCHETYPE_PRIORITY}
    selected_idxs = list(selected_sample_idx_by_archetype.values())

    extra_checks = []

    def add_extra(name, ok, notes=''):
        extra_checks.append(dict(check_name=name, observed='', expected='', status=('PASS' if ok else 'FAIL'),
                                    notes=notes))

    add_extra('exactly_six_archetypes_selected', len(selected_idxs) == 6, notes=f'n={len(selected_idxs)}')
    add_extra('selected_sample_indices_unique', len(set(selected_idxs)) == len(selected_idxs),
               notes=str(selected_idxs))
    for archetype_id in ARCHETYPE_PRIORITY:
        add_extra(f'archetype_has_eligible_samples[{archetype_id}]',
                   len(results[archetype_id]['eligible_idxs']) > 0)
    if extra_checks and any(c['status'] != 'PASS' for c in extra_checks):
        failed = [c['check_name'] for c in extra_checks if c['status'] != 'PASS']
        raise SystemExit(f'[hard-fail] post-selection validation failed: {failed}')
    log(f'[validate] All {len(extra_checks)} post-selection validation checks PASSED.')

    score_table_rows = build_archetype_score_table(results, ranked_by_archetype, score_by_idx_by_archetype)
    write_csv(SELECTION_DIR / 'archetype_score_table.csv',
               ['archetype_id', 'priority', 'rank', 'sample_idx', 'score', 'eligibility_count',
                'c1_name', 'c1_raw', 'c1_z', 'c1_z_method', 'c2_name', 'c2_raw', 'c2_z', 'c2_z_method',
                'c3_name', 'c3_raw', 'c3_z', 'c3_z_method', 'c4_name', 'c4_raw', 'c4_z', 'c4_z_method',
                'score_formula'], score_table_rows)

    selected_rows, alternate_rows = build_selected_and_alternates(results, selection_by_archetype)
    write_csv(SELECTION_DIR / 'archetype_selected_samples.csv', ARCHETYPE_SELECTED_FIELDS, selected_rows)
    write_csv(SELECTION_DIR / 'archetype_alternates.csv', ARCHETYPE_SELECTED_FIELDS, alternate_rows)

    diagnostics_rows = build_diagnostics_rows(all_diagnostics)
    write_csv(SELECTION_DIR / 'archetype_selection_diagnostics.csv',
               ['archetype_id', 'event_type', 'sample_idx', 'rank_before_dedup', 'rank_after_dedup', 'detail'],
               diagnostics_rows)

    metric_context_rows = build_metric_context_rows(selected_sample_idx_by_archetype, phase2c_refs)
    write_csv(SELECTION_DIR / 'selected_sample_metric_context.csv',
               ['sample_idx', 'archetype_id', 'agreement_rate', 'disagreement_rate', 'tie_or_undefined_count',
                'method_pair_count', 'pd_mt_cross_method_pearson', 'pd_mt_cross_method_spearman'],
               metric_context_rows)

    method_values_rows = build_method_values_rows(per_sample, metric_cols, metric_direction,
                                                     selected_sample_idx_by_archetype, phase2c_refs)
    write_csv(SELECTION_DIR / 'selected_sample_method_values.csv',
               build_method_values_fieldnames(metric_cols), method_values_rows)

    pairwise_rows = build_pairwise_preferences_rows(per_sample, selected_sample_idx_by_archetype)
    write_csv(SELECTION_DIR / 'selected_sample_pairwise_preferences.csv',
               ['sample_idx', 'archetype_id', 'method_a', 'method_b', 'pd_value_a', 'pd_value_b',
                'mt_value_a', 'mt_value_b', 'pd_preferred', 'mt_preferred', 'descriptors_agree'],
               pairwise_rows)

    manifest_rows = build_preview_method_manifest()
    write_csv(SELECTION_DIR / 'preview_method_manifest.csv',
               ['group_name', 'position', 'method_id', 'includes_gt_reference'], manifest_rows)

    plan_rows = build_preview_plan_rows(selected_sample_idx_by_archetype)
    write_csv(SELECTION_DIR / 'preview_plan.csv',
               ['archetype_id', 'selected_sample_idx', 'preview_type', 'output_path', 'methods_included',
                'panel_count', 'planned_annotations', 'status'], plan_rows)

    raw_req_rows = build_raw_artifact_requirements_rows(method_inventory)
    write_csv(SELECTION_DIR / 'raw_artifact_requirements.csv',
               ['method_id', 'original_method_name', 'array_role', 'expected_repo_relative_path',
                'expected_shape', 'expected_dtype', 'resolution_convention', 'notes'], raw_req_rows)

    figure_plan_rows = build_figure_plan_rows(selected_sample_idx_by_archetype, results)
    write_csv(SELECTION_DIR / 'figure_plan.csv',
               ['figure_id', 'figure_title', 'archetype_id', 'selected_sample_idx', 'method_group',
                'methods_included', 'narrative_scope', 'status', 'rendering_note'], figure_plan_rows)

    all_validation_rows = base_checks + extra_checks
    write_csv(SELECTION_DIR / 'selection_validation.csv',
               ['check_name', 'observed', 'expected', 'status', 'notes'], all_validation_rows)

    return dict(
        results=results, selection_by_archetype=selection_by_archetype,
        selected_sample_idx_by_archetype=selected_sample_idx_by_archetype,
        all_diagnostics=all_diagnostics, all_validation_rows=all_validation_rows,
    )


# =============================================================================
# Raw-artifact path resolution (--render-previews / --full only). Absolute
# paths recorded in method_inventory.csv are historical-machine provenance
# only and are NEVER used as executable paths here -- every path is built
# from a repository-relative convention rooted at base_dir (REPO_ROOT by
# default; overridable so tests can point at a synthetic temp tree).
# =============================================================================

def resolve_raw_paths(method_inventory, base_dir=None):
    base_dir = base_dir or REPO_ROOT
    paths = {}
    for mid in FULL_SELECTED_STORY:
        inv = method_inventory.get(mid, {})
        original = inv.get('original_method_name', mid)
        if mid == BICUBIC_METHOD:
            cnn_dir = base_dir / 'data_out_fixed' / 'wind_mrhr_cnn'
            paths[mid] = dict(dir=cnn_dir, idx=cnn_dir / 'idx.npy', dataIN=cnn_dir / 'dataIN.npy',
                                dataGT=cnn_dir / 'dataGT.npy', dataSR=None, reconstructed=True)
        elif mid in (CNN_METHOD, GAN_METHOD):
            d = base_dir / 'data_out_fixed' / f'wind_mrhr_{mid}'
            paths[mid] = dict(dir=d, idx=d / 'idx.npy', dataIN=d / 'dataIN.npy', dataGT=d / 'dataGT.npy',
                                dataSR=d / 'dataSR.npy', reconstructed=False)
        else:
            d = base_dir / 'data_out' / f'wind_finetune_{original}'
            paths[mid] = dict(dir=d, idx=d / 'idx.npy', dataIN=d / 'dataIN.npy', dataGT=d / 'dataGT.npy',
                                dataSR=d / 'dataSR.npy', reconstructed=False)
    return paths


def require_raw_artifacts_exist(paths_by_method):
    missing = []
    for mid, p in paths_by_method.items():
        for role in ('idx', 'dataIN', 'dataGT', 'dataSR'):
            path = p[role]
            if path is None:
                continue
            if not path.exists():
                missing.append(f'{mid}:{role}:{path}')
    if missing:
        raise SystemExit(
            '[hard-fail] Missing raw artifact(s) required for --render-previews (these are expected on the '
            'authoritative Spark run, and are absent by design in a lightweight checkout since data_out/ and '
            'data_out_fixed/ are gitignored):\n' + '\n'.join(f'  - {m}' for m in missing)
        )


# =============================================================================
# Physical-unit helpers. Saved dataIN/dataGT/dataSR arrays are already
# physical (denormalized) [u, v]; scalar speed is sqrt(u**2+v**2) directly.
# MU_SIG denorm is provenance/test-only and is NEVER applied on this path.
# =============================================================================

def speed_from_uv(uv):
    """uv: array with the vector-component axis LAST -- either a single
    sample (H, W, C) or a batch (N, H, W, C). Channel is always the final
    axis in every call site in this module (never an implicit (N, H, W)
    channel-less batch), so no ndim-based heuristic is needed or applied."""
    uv = np.asarray(uv)
    if uv.shape[-1] >= 2:
        return np.sqrt(uv[..., 0] ** 2 + uv[..., 1] ** 2)
    return np.abs(uv[..., 0])


def _denorm_from_normalized(x_norm, mu=None, sig=None):
    """Provenance/test-only helper for RAW NORMALIZED network tensors.
    Never called on the authoritative saved-array path (those are already
    physical). physical = sigma * normalized + mu."""
    mu = np.asarray(mu if mu is not None else MU_SIG[0], dtype=np.float64)
    sig = np.asarray(sig if sig is not None else MU_SIG[1], dtype=np.float64)
    return sig * np.asarray(x_norm, dtype=np.float64) + mu


def bicubic_reconstruct_selected(data_in_selected: np.ndarray) -> np.ndarray:
    """Reconstructs bicubic dataSR for exactly the given selected-sample MR
    inputs, using the established project convention (scipy.ndimage.zoom,
    order=3, mode='reflect', prefilter=True, applied independently per
    channel). Never writes any output to disk."""
    from scipy.ndimage import zoom
    n, h, w, c = data_in_selected.shape
    target_h, target_w = EXPECTED_HR_SHAPE[1], EXPECTED_HR_SHAPE[2]
    zoom_h = target_h / h
    zoom_w = target_w / w
    out = np.empty((n, target_h, target_w, c), dtype=np.float32)
    for i in range(n):
        for ch in range(c):
            out[i, :, :, ch] = zoom(
                data_in_selected[i, :, :, ch].astype(np.float64), (zoom_h, zoom_w),
                order=3, mode='reflect', prefilter=True,
            )
    return out


def _idx_position_map(idx_array: np.ndarray) -> dict:
    return {int(v): i for i, v in enumerate(np.asarray(idx_array).ravel())}


def validate_idx_array(idx_arr: np.ndarray, mid: str):
    """Exact hard-fail idx-array gate. Never uses set equality (which would
    silently accept a reordered/permuted idx array as if it were sorted).
    Returns (status, detail) where status is 'PASS' or 'FAIL' and detail is
    a human-readable explanation (empty on PASS)."""
    if tuple(idx_arr.shape) != (N_EVAL,):
        return 'FAIL', f'{mid}: idx.npy shape {tuple(idx_arr.shape)} != expected ({N_EVAL},)'
    if not np.issubdtype(idx_arr.dtype, np.integer):
        return 'FAIL', f'{mid}: idx.npy dtype {idx_arr.dtype} is not an integer dtype'
    if not np.array_equal(idx_arr, np.arange(N_EVAL, dtype=idx_arr.dtype)):
        return 'FAIL', (f'{mid}: idx.npy is not exactly [0, 1, ..., {N_EVAL - 1}] in order '
                          f'(a permutation, duplicate, or missing index was detected)')
    return 'PASS', ''


# Bounded-memory chunk size (rows) for full-168-row raw-array audits.
CHUNK_SIZE = 16


def _chunk_ranges(n: int, chunk_size: int = CHUNK_SIZE):
    for start in range(0, n, chunk_size):
        yield start, min(start + chunk_size, n)


def _full_array_shape_finite_aligned(arr_mmap, expected_shape, canonical_mmap=None, chunk_size=CHUNK_SIZE):
    """Validates an ENTIRE memory-mapped raw array (all N_EVAL rows, never
    only a selected subset) in bounded-memory chunks: exact shape, all-rows
    finiteness, and (when canonical_mmap is given) exact row-for-row equality
    against the canonical CNN array. Returns (shape_ok, finite_ok,
    aligned_ok), where aligned_ok is None iff canonical_mmap is None (no
    alignment check requested, e.g. dataSR, which differs per method by
    design)."""
    shape_ok = tuple(arr_mmap.shape) == expected_shape
    if not shape_ok:
        return False, False, (None if canonical_mmap is None else False)
    finite_ok = True
    aligned_ok = None if canonical_mmap is None else True
    for start, end in _chunk_ranges(arr_mmap.shape[0], chunk_size):
        chunk = np.asarray(arr_mmap[start:end])
        if not np.isfinite(chunk).all():
            finite_ok = False
        if canonical_mmap is not None:
            canon_chunk = np.asarray(canonical_mmap[start:end])
            if not np.array_equal(chunk, canon_chunk):
                aligned_ok = False
    return shape_ok, finite_ok, aligned_ok


RAW_ALIGNMENT_FIELDS = [
    'method_id', 'idx_validation_status', 'dataIN_shape_status', 'dataIN_finiteness_status',
    'input_alignment_status', 'dataGT_shape_status', 'dataGT_finiteness_status', 'gt_alignment_status',
    'dataSR_shape_status', 'dataSR_finiteness_status', 'overall_status',
]


# =============================================================================
# Raw-artifact audit (--render-previews / --full only). Validates the FULL
# N_EVAL-row array for every method (idx ordering, shape, finiteness, and --
# for dataIN/dataGT -- exact alignment against the canonical CNN arrays)
# using bounded-memory chunked reads over NumPy memory maps, so corruption in
# any row (selected or not) is caught. Only after a method's full-array
# audit passes are its selected-sample rows extracted into memory for
# per-sample statistics, metric reproduction, and preview rendering.
# =============================================================================

def audit_raw_artifacts(paths_by_method, selected_sample_idxs, per_sample, base_dir=None):
    base_dir = base_dir or REPO_ROOT
    inventory_rows = []
    alignment_rows = []
    stats_rows = []
    repro_rows = []
    failures = []
    selected_data = {}  # method_id -> dict(gt=array, sr=array)

    def add_fail(msg):
        failures.append(msg)

    canonical = paths_by_method[CNN_METHOD]
    canonical_idx = np.load(canonical['idx'])
    canonical_idx_status, canonical_idx_detail = validate_idx_array(canonical_idx, CNN_METHOD)
    if canonical_idx_status != 'PASS':
        # Malformed canonical data must produce a controlled SystemExit here,
        # not a KeyError further down when every other method's selected-row
        # positions are resolved against this array.
        raise SystemExit(f'[hard-fail] Canonical CNN idx.npy failed validation: {canonical_idx_detail}')
    canonical_pos = _idx_position_map(canonical_idx)
    canonical_in = np.load(canonical['dataIN'], mmap_mode='r')
    canonical_gt = np.load(canonical['dataGT'], mmap_mode='r')

    for mid, p in paths_by_method.items():
        for role in ('idx', 'dataIN', 'dataGT', 'dataSR'):
            path = p[role]
            if path is None:
                inventory_rows.append(dict(method_id=mid, array_role=role, path='(reconstructed_in_memory)',
                                              exists=True, shape='', dtype='', size_bytes=''))
                continue
            exists = path.exists()
            shape = dtype = size_bytes = ''
            if exists:
                arr = np.load(path, mmap_mode='r')
                shape = str(arr.shape)
                dtype = str(arr.dtype)
                size_bytes = path.stat().st_size
            inventory_rows.append(dict(method_id=mid, array_role=role, path=_rel_posix(path, base_dir),
                                          exists=exists, shape=shape, dtype=dtype, size_bytes=size_bytes))

        idx_arr = np.load(p['idx'])
        idx_status, idx_detail = validate_idx_array(idx_arr, mid)
        if idx_status != 'PASS':
            add_fail(idx_detail)

        data_in = np.load(p['dataIN'], mmap_mode='r')
        data_gt = np.load(p['dataGT'], mmap_mode='r')

        in_shape_ok, in_finite_ok, in_aligned_ok = _full_array_shape_finite_aligned(
            data_in, EXPECTED_IN_SHAPE, canonical_mmap=(None if mid == CNN_METHOD else canonical_in))
        gt_shape_ok, gt_finite_ok, gt_aligned_ok = _full_array_shape_finite_aligned(
            data_gt, EXPECTED_HR_SHAPE, canonical_mmap=(None if mid == CNN_METHOD else canonical_gt))

        if not in_shape_ok:
            add_fail(f'{mid}: dataIN shape {tuple(data_in.shape)} != expected {EXPECTED_IN_SHAPE} '
                       f'(checked over all {N_EVAL} rows)')
        if not in_finite_ok:
            add_fail(f'{mid}: dataIN contains non-finite value(s) somewhere in the full {N_EVAL}-row array')
        if in_aligned_ok is False:
            add_fail(f'{mid}: dataIN does not exactly match canonical CNN dataIN over the full {N_EVAL}-row array')
        if not gt_shape_ok:
            add_fail(f'{mid}: dataGT shape {tuple(data_gt.shape)} != expected {EXPECTED_HR_SHAPE} '
                       f'(checked over all {N_EVAL} rows)')
        if not gt_finite_ok:
            add_fail(f'{mid}: dataGT contains non-finite value(s) somewhere in the full {N_EVAL}-row array')
        if gt_aligned_ok is False:
            add_fail(f'{mid}: dataGT does not exactly match canonical CNN dataGT over the full {N_EVAL}-row array')

        if p['reconstructed']:
            # Bicubic has no saved on-disk dataSR: only the selected fields are
            # ever reconstructed in memory (never all 168 rows), so a full-array
            # audit does not apply.
            sr_shape_status = sr_finite_status = 'N/A'
        else:
            data_sr = np.load(p['dataSR'], mmap_mode='r')
            sr_shape_ok, sr_finite_ok, _ = _full_array_shape_finite_aligned(data_sr, EXPECTED_HR_SHAPE)
            if not sr_shape_ok:
                add_fail(f'{mid}: dataSR shape {tuple(data_sr.shape)} != expected {EXPECTED_HR_SHAPE} '
                           f'(checked over all {N_EVAL} rows)')
            if not sr_finite_ok:
                add_fail(f'{mid}: dataSR contains non-finite value(s) somewhere in the full {N_EVAL}-row array')
            sr_shape_status = 'PASS' if sr_shape_ok else 'FAIL'
            sr_finite_status = 'PASS' if sr_finite_ok else 'FAIL'

        in_alignment_status = 'PASS' if (mid == CNN_METHOD or in_aligned_ok) else 'FAIL'
        gt_alignment_status = 'PASS' if (mid == CNN_METHOD or gt_aligned_ok) else 'FAIL'
        status_fields = [
            idx_status, ('PASS' if in_shape_ok else 'FAIL'), ('PASS' if in_finite_ok else 'FAIL'),
            in_alignment_status, ('PASS' if gt_shape_ok else 'FAIL'), ('PASS' if gt_finite_ok else 'FAIL'),
            gt_alignment_status, sr_shape_status, sr_finite_status,
        ]
        overall_status = 'PASS' if all(s in ('PASS', 'N/A') for s in status_fields) else 'FAIL'

        alignment_rows.append(dict(
            method_id=mid, idx_validation_status=idx_status,
            dataIN_shape_status=('PASS' if in_shape_ok else 'FAIL'),
            dataIN_finiteness_status=('PASS' if in_finite_ok else 'FAIL'),
            input_alignment_status=in_alignment_status,
            dataGT_shape_status=('PASS' if gt_shape_ok else 'FAIL'),
            dataGT_finiteness_status=('PASS' if gt_finite_ok else 'FAIL'),
            gt_alignment_status=gt_alignment_status,
            dataSR_shape_status=sr_shape_status, dataSR_finiteness_status=sr_finite_status,
            overall_status=overall_status,
        ))

        if overall_status != 'PASS':
            # The full-array audit already failed for this method (recorded
            # above); skip selected-row extraction rather than risk operating
            # on malformed, misaligned, or non-finite data.
            continue

        pos_map = canonical_pos if mid == CNN_METHOD else _idx_position_map(idx_arr)
        positions = [pos_map[si] for si in selected_sample_idxs]
        in_sel = np.asarray(data_in[positions])
        gt_sel = np.asarray(data_gt[positions])

        if p['reconstructed']:
            sr_sel = bicubic_reconstruct_selected(in_sel)
        else:
            data_sr = np.load(p['dataSR'], mmap_mode='r')
            sr_sel = np.asarray(data_sr[positions])

        for i, si in enumerate(selected_sample_idxs):
            speed_gt = speed_from_uv(gt_sel[i])
            speed_sr = speed_from_uv(sr_sel[i])
            stats_rows.append(dict(
                method_id=mid, sample_idx=si,
                gt_speed_min=float(speed_gt.min()), gt_speed_max=float(speed_gt.max()),
                gt_speed_mean=float(speed_gt.mean()),
                sr_speed_min=float(speed_sr.min()), sr_speed_max=float(speed_sr.max()),
                sr_speed_mean=float(speed_sr.mean()),
            ))
            recomputed_speed_mae = float(np.mean(np.abs(speed_sr - speed_gt)))
            lt_speed_mae = per_sample.get(mid, {}).get(si, {}).get('speed_mae', float('nan'))
            diff = abs(recomputed_speed_mae - lt_speed_mae) if math.isfinite(lt_speed_mae) else float('nan')
            within_tol = math.isfinite(diff) and diff <= 1e-3
            repro_rows.append(dict(
                method_id=mid, sample_idx=si, recomputed_speed_mae=recomputed_speed_mae,
                long_table_speed_mae=nfmt(lt_speed_mae if math.isfinite(lt_speed_mae) else None),
                abs_diff=nfmt(diff if math.isfinite(diff) else None), within_tolerance=within_tol,
            ))
            # No method (including bicubic) is exempt: whenever the Phase-1
            # long-table speed_mae is finite, the in-memory recomputation must
            # match it within tolerance.
            if math.isfinite(lt_speed_mae) and not within_tol:
                add_fail(f'{mid} sample={si}: recomputed speed_mae {recomputed_speed_mae:.6f} does not '
                          f'match long-table value {lt_speed_mae!r} within tolerance')

        selected_data[mid] = dict(gt=gt_sel, sr=sr_sel)

    return dict(
        inventory_rows=inventory_rows, alignment_rows=alignment_rows, stats_rows=stats_rows,
        repro_rows=repro_rows, failures=failures, selected_data=selected_data,
    )


# =============================================================================
# Review-only preview rendering (--render-previews / --full only). Never
# writes placeholder/blank images; only called after the raw audit passes.
# =============================================================================

# Meaningful panel count for the detailed per-sample review preview:
# GT speed + 7 method speed panels + 7 method error panels = 15. The figure
# actually allocates 16 axes (2 rows x 8 cols); the bottom-left axis under
# GT is intentionally blank (there is no "GT error" panel) and is turned
# off. preview_plan.csv and preview_render_validation.csv both report the
# same 15-meaningful-panel convention.
DETAILED_PREVIEW_PANEL_COUNT = 1 + len(FULL_SELECTED_STORY) + len(FULL_SELECTED_STORY)


def compute_preview_panel_data(gt_uv, sr_uv_by_method):
    """Pure (no matplotlib) computation shared by the per-sample review
    preview and each contact-sheet row. The shared speed vmin/vmax are
    derived from GT + every displayed method's SR speed field (not GT
    alone), and that single (vmin, vmax) pair is the one applied to every
    speed panel -- GT and all seven methods alike. The shared error vmax is
    likewise one value (0..max) applied to every error panel."""
    gt_speed = speed_from_uv(gt_uv)
    method_speeds = {mid: speed_from_uv(uv) for mid, uv in sr_uv_by_method.items()}
    all_speed_arrays = [gt_speed] + list(method_speeds.values())
    speed_vmin = float(min(float(a.min()) for a in all_speed_arrays))
    speed_vmax = float(max(float(a.max()) for a in all_speed_arrays))
    errors = {mid: np.abs(method_speeds[mid] - gt_speed) for mid in method_speeds}
    error_vmin = 0.0
    error_vmax = float(max(float(e.max()) for e in errors.values())) if errors else 0.0
    return dict(gt_speed=gt_speed, method_speeds=method_speeds, speed_vmin=speed_vmin, speed_vmax=speed_vmax,
                errors=errors, error_vmin=error_vmin, error_vmax=error_vmax)


def render_selected_previews(selected_data, selected_sample_idx_by_archetype, per_sample):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    render_rows = []
    panel_methods = FULL_SELECTED_STORY
    # selected_sample_idx_by_archetype preserves insertion order (archetype priority
    # order); build one stable ordered list of the (up to 6) distinct selected
    # sample_idx values, used to index into each method's selected_data arrays.
    ordered_selected = list(dict.fromkeys(selected_sample_idx_by_archetype.values()))

    for archetype_id in ARCHETYPE_PRIORITY:
        si = selected_sample_idx_by_archetype[archetype_id]
        out_dir = PREVIEWS_DIR / archetype_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f'selected_sample_{si}_review.png'
        pos = ordered_selected.index(si)

        gt_uv = selected_data[CNN_METHOD]['gt'][pos]
        sr_uv_by_method = {mid: selected_data[mid]['sr'][pos] for mid in panel_methods}
        panel = compute_preview_panel_data(gt_uv, sr_uv_by_method)
        gt_speed = panel['gt_speed']
        method_speeds = panel['method_speeds']
        errors = panel['errors']
        vmin, vmax = panel['speed_vmin'], panel['speed_vmax']
        err_max = panel['error_vmax']

        n_cols = 1 + len(panel_methods)
        fig, axes = plt.subplots(2, n_cols, figsize=(2.4 * n_cols, 5.2), dpi=150)
        fig.suptitle(f'AUDIT PREVIEW -- NOT FINAL FIGURE  |  archetype={archetype_id}  sample_idx={si}',
                      fontsize=9)

        ax = axes[0, 0]
        im0 = ax.imshow(gt_speed, cmap='viridis', vmin=vmin, vmax=vmax, origin='lower', aspect='equal')
        ax.set_title('GT', fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        axes[1, 0].axis('off')  # no "GT error" panel; the 16th axis is intentionally blank

        for j, mid in enumerate(panel_methods, start=1):
            ax = axes[0, j]
            ax.imshow(method_speeds[mid], cmap='viridis', vmin=vmin, vmax=vmax, origin='lower', aspect='equal')
            pd_val = per_sample.get(mid, {}).get(si, {}).get('pd_distance', float('nan'))
            mt_val = per_sample.get(mid, {}).get(si, {}).get('mt_distance', float('nan'))
            annot = ''
            if math.isfinite(pd_val) and math.isfinite(mt_val):
                annot = f'\nPD={pd_val:.2f} MT={mt_val:.2f}'
            ax.set_title(f'{mid}{annot}', fontsize=7)
            ax.set_xticks([])
            ax.set_yticks([])

            axe = axes[1, j]
            im_err = axe.imshow(errors[mid], cmap='magma', vmin=0.0, vmax=err_max, origin='lower', aspect='equal')
            axe.set_title(f'|{mid} - GT|', fontsize=7)
            axe.set_xticks([])
            axe.set_yticks([])

        fig.colorbar(im0, ax=list(axes[0, :]), fraction=0.02, pad=0.02, label='Wind speed (m/s)')
        fig.colorbar(im_err, ax=list(axes[1, 1:]), fraction=0.02, pad=0.02, label='Abs speed error (m/s)')
        fig.savefig(out_path, dpi=150, metadata={'Software': ''})
        plt.close(fig)
        render_rows.append(dict(
            archetype_id=archetype_id, sample_idx=si, output_path=str(out_path.relative_to(REPO_ROOT)),
            panel_count=DETAILED_PREVIEW_PANEL_COUNT, shared_speed_vmin=vmin, shared_speed_vmax=vmax,
            shared_error_vmin=panel['error_vmin'], shared_error_vmax=err_max, status='rendered',
        ))
        log(f'[write] {out_path}')

    # Contact sheet: one row per archetype, GT + 7 method speed panels only.
    contact_path = PREVIEWS_DIR / 'phase2d_selected_archetypes_contact_sheet.png'
    n_cols = 1 + len(panel_methods)
    n_rows = len(ARCHETYPE_PRIORITY)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.2 * n_cols, 2.2 * n_rows), dpi=150)
    fig.suptitle('AUDIT PREVIEW -- NOT FINAL FIGURE  |  Phase 2D-A selected archetypes contact sheet', fontsize=10)
    for r, archetype_id in enumerate(ARCHETYPE_PRIORITY):
        si = selected_sample_idx_by_archetype[archetype_id]
        pos = ordered_selected.index(si)
        gt_uv = selected_data[CNN_METHOD]['gt'][pos]
        sr_uv_by_method = {mid: selected_data[mid]['sr'][pos] for mid in panel_methods}
        # Same common-speed-limit rule applied per row: GT + all 7 method SR fields.
        panel = compute_preview_panel_data(gt_uv, sr_uv_by_method)
        vmin, vmax = panel['speed_vmin'], panel['speed_vmax']
        axes[r, 0].imshow(panel['gt_speed'], cmap='viridis', vmin=vmin, vmax=vmax, origin='lower', aspect='equal')
        axes[r, 0].set_ylabel(f'{archetype_id}\nidx={si}', fontsize=6)
        axes[r, 0].set_xticks([])
        axes[r, 0].set_yticks([])
        for c, mid in enumerate(panel_methods, start=1):
            axes[r, c].imshow(panel['method_speeds'][mid], cmap='viridis', vmin=vmin, vmax=vmax,
                                origin='lower', aspect='equal')
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            if r == 0:
                axes[r, c].set_title(mid, fontsize=7)
        if r == 0:
            axes[r, 0].set_title('GT', fontsize=7)
    fig.savefig(contact_path, dpi=150, metadata={'Software': ''})
    plt.close(fig)
    log(f'[write] {contact_path}')
    render_rows.append(dict(
        archetype_id='all', sample_idx='', output_path=str(contact_path.relative_to(REPO_ROOT)),
        panel_count=n_cols * n_rows, shared_speed_vmin='', shared_speed_vmax='',
        shared_error_vmin='', shared_error_vmax='', status='rendered',
    ))
    return render_rows


ARTIFACT_MANIFEST_FIELDS = [
    'archetype_id', 'sample_idx', 'method_id', 'original_method_name', 'array_role',
    'source_array_directory', 'source_file', 'sample_position', 'shape', 'dtype', 'finite_status',
    'input_alignment_status', 'gt_alignment_status', 'topology_csv_path', 'topology_pd_column',
    'topology_mt_column', 'topology_row_found_status',
]


def build_artifact_manifest_rows(audit, selected_sample_idx_by_archetype, ordered_selected, method_inventory,
                                    topology_source_map, paths_by_method, base_dir=None):
    """One row per selected sample x full_selected_story method, plus one GT
    row per selected sample. Produced only after the Spark raw audit has
    passed (audit['selected_data'] is populated for every method at that
    point). All paths are repository-relative POSIX text."""
    base_dir = base_dir or REPO_ROOT
    archetype_by_sample = {si: a for a, si in selected_sample_idx_by_archetype.items()}
    alignment_by_method = {r['method_id']: r for r in audit['alignment_rows']}
    topo_rows_by_method = {mid: read_topology_pd_mt_for_method(mid, topology_source_map)
                             for mid in topology_source_map}
    rows = []
    for pos, si in enumerate(ordered_selected):
        archetype_id = archetype_by_sample.get(si, '')

        gt_field = audit['selected_data'][CNN_METHOD]['gt'][pos]
        cnn_dir = paths_by_method[CNN_METHOD]['dir']
        rows.append(dict(
            archetype_id=archetype_id, sample_idx=si, method_id='GT', original_method_name='ground_truth',
            array_role='dataGT', source_array_directory=_rel_posix(cnn_dir, base_dir), source_file='dataGT.npy',
            sample_position=si, shape=str(tuple(gt_field.shape)), dtype=str(gt_field.dtype),
            finite_status=('PASS' if np.isfinite(gt_field).all() else 'FAIL'),
            input_alignment_status='N/A', gt_alignment_status='PASS',
            topology_csv_path='', topology_pd_column='', topology_mt_column='',
            topology_row_found_status='not_applicable',
        ))

        for mid in FULL_SELECTED_STORY:
            inv = method_inventory.get(mid, {})
            original = inv.get('original_method_name', mid)
            p = paths_by_method[mid]
            sr_field = audit['selected_data'][mid]['sr'][pos]
            align = alignment_by_method.get(mid, {})
            if mid == BICUBIC_METHOD:
                source_dir, source_file = '(reconstructed_in_memory)', '(reconstructed_in_memory)'
            else:
                source_dir, source_file = _rel_posix(p['dir'], base_dir), 'dataSR.npy'
            if mid in topology_source_map:
                entry = topology_source_map[mid]
                # Topology CSVs are always real, git-tracked provenance files
                # rooted at REPO_ROOT, independent of `base_dir` (which may
                # point at a synthetic array tree in tests).
                topo_path = _rel_posix(entry['path'], REPO_ROOT)
                found = 'found' if si in topo_rows_by_method[mid] else 'missing'
                pd_col, mt_col = entry['pd_column'], entry['mt_column']
            else:
                topo_path, pd_col, mt_col, found = '', '', '', 'not_applicable'
            rows.append(dict(
                archetype_id=archetype_id, sample_idx=si, method_id=mid, original_method_name=original,
                array_role='dataSR', source_array_directory=source_dir, source_file=source_file,
                sample_position=si, shape=str(tuple(sr_field.shape)), dtype=str(sr_field.dtype),
                finite_status=('PASS' if np.isfinite(sr_field).all() else 'FAIL'),
                input_alignment_status=align.get('input_alignment_status', ''),
                gt_alignment_status=align.get('gt_alignment_status', ''),
                topology_csv_path=topo_path, topology_pd_column=pd_col, topology_mt_column=mt_col,
                topology_row_found_status=found,
            ))
    return rows


def run_render(selected_sample_idx_by_archetype, per_sample, method_inventory, topology_source_map):
    paths_by_method = resolve_raw_paths(method_inventory)
    require_raw_artifacts_exist(paths_by_method)

    ordered_selected = list(dict.fromkeys(selected_sample_idx_by_archetype.values()))
    audit = audit_raw_artifacts(paths_by_method, ordered_selected, per_sample)

    write_csv(PREVIEW_AUDIT_DIR / 'raw_artifact_inventory.csv',
               ['method_id', 'array_role', 'path', 'exists', 'shape', 'dtype', 'size_bytes'],
               audit['inventory_rows'])
    write_csv(PREVIEW_AUDIT_DIR / 'raw_alignment_validation.csv', RAW_ALIGNMENT_FIELDS, audit['alignment_rows'])
    write_csv(PREVIEW_AUDIT_DIR / 'selected_sample_array_statistics.csv',
               ['method_id', 'sample_idx', 'gt_speed_min', 'gt_speed_max', 'gt_speed_mean', 'sr_speed_min',
                'sr_speed_max', 'sr_speed_mean'], audit['stats_rows'])
    write_csv(PREVIEW_AUDIT_DIR / 'selected_sample_metric_reproduction.csv',
               ['method_id', 'sample_idx', 'recomputed_speed_mae', 'long_table_speed_mae', 'abs_diff',
                'within_tolerance'], audit['repro_rows'])

    if audit['failures']:
        write_csv(PREVIEW_AUDIT_DIR / 'preview_render_validation.csv',
                   ['archetype_id', 'sample_idx', 'output_path', 'panel_count', 'shared_speed_vmin',
                    'shared_speed_vmax', 'shared_error_vmin', 'shared_error_vmax', 'status'], [])
        write_csv(PREVIEW_AUDIT_DIR / 'selected_sample_artifact_manifest.csv', ARTIFACT_MANIFEST_FIELDS, [])
        log('[RAW-ARTIFACT AUDIT FAILURE]')
        for f in audit['failures']:
            log(f'  - {f}')
        raise SystemExit(f'[hard-fail] {len(audit["failures"])} raw-artifact audit check(s) failed; '
                           f'no preview was rendered.')
    log(f'[audit] All raw-artifact checks PASSED for {len(paths_by_method)} methods x '
        f'{len(ordered_selected)} selected samples.')

    manifest_rows = build_artifact_manifest_rows(audit, selected_sample_idx_by_archetype, ordered_selected,
                                                    method_inventory, topology_source_map, paths_by_method)
    write_csv(PREVIEW_AUDIT_DIR / 'selected_sample_artifact_manifest.csv', ARTIFACT_MANIFEST_FIELDS, manifest_rows)

    render_rows = render_selected_previews(audit['selected_data'], selected_sample_idx_by_archetype, per_sample)
    write_csv(PREVIEW_AUDIT_DIR / 'preview_render_validation.csv',
               ['archetype_id', 'sample_idx', 'output_path', 'panel_count', 'shared_speed_vmin',
                'shared_speed_vmax', 'shared_error_vmin', 'shared_error_vmax', 'status'], render_rows)
    return dict(audit=audit, render_rows=render_rows, manifest_rows=manifest_rows)


SPARK_COMMAND = 'python3 scripts/select_and_preview_unified_candidates_phase2d.py --full'


def validate_selected_samples_manifest_rows(all_rows, path):
    """Strong validation of archetype_selected_samples.csv before it is used
    to drive --render-previews. Returns the list of errors found (empty if
    valid). Never raises; the caller decides whether/how to hard-fail."""
    errors = []
    missing_cols = [f for f in ARCHETYPE_SELECTED_FIELDS if all_rows and f not in all_rows[0]]
    if missing_cols:
        return [f'missing required column(s): {missing_cols}']

    primary_rows = [r for r in all_rows if r.get('primary_or_alternate') == 'primary']
    if len(primary_rows) != 6:
        errors.append(f'expected exactly 6 primary rows, found {len(primary_rows)}')

    primary_ids_in_order = [r.get('archetype_id', '') for r in primary_rows]
    if primary_ids_in_order != ARCHETYPE_PRIORITY:
        errors.append(f'primary-row archetype_id order/identity mismatch: expected {ARCHETYPE_PRIORITY}, '
                        f'got {primary_ids_in_order}')

    seen_archetypes = set()
    selected_idxs = []
    for r in primary_rows:
        aid = r.get('archetype_id', '(missing)')
        if aid in seen_archetypes:
            errors.append(f'{aid}: duplicate primary row for this archetype_id')
        seen_archetypes.add(aid)

        raw_idx = r.get('selected_sample_idx', '')
        try:
            si = int(str(raw_idx).strip())
        except (TypeError, ValueError):
            errors.append(f'{aid}: selected_sample_idx {raw_idx!r} is not an integer')
        else:
            if not (0 <= si <= N_EVAL - 1):
                errors.append(f'{aid}: selected_sample_idx {si} is out of range [0, {N_EVAL - 1}]')
            selected_idxs.append(si)

        if r.get('primary_or_alternate') != 'primary':
            errors.append(f'{aid}: primary_or_alternate != "primary" (got {r.get("primary_or_alternate")!r})')
        if r.get('eligibility_status') != 'eligible':
            errors.append(f'{aid}: eligibility_status != "eligible" (got {r.get("eligibility_status")!r})')
        for f in ARCHETYPE_SELECTED_FIELDS:
            if str(r.get(f, '')).strip() == '':
                errors.append(f'{aid}: required field {f!r} is empty')

    if len(set(selected_idxs)) != len(selected_idxs):
        errors.append(f'selected_sample_idx values across the 6 primary rows are not all unique: {selected_idxs}')

    return errors


def read_selected_samples_manifest():
    path = SELECTION_DIR / 'archetype_selected_samples.csv'
    if not path.exists():
        raise SystemExit(
            f'[hard-fail] {path} does not exist. --render-previews requires the deterministic selection '
            f'manifest written by --selection-only (or by an earlier stage of --full) to already be present. '
            f'Run `python3 scripts/select_and_preview_unified_candidates_phase2d.py --selection-only` first.'
        )
    all_rows = read_csv_dicts(path)
    errors = validate_selected_samples_manifest_rows(all_rows, path)
    if errors:
        raise SystemExit(
            f'[hard-fail] {path} failed manifest validation ({len(errors)} issue(s)) before use in '
            f'--render-previews:\n' + '\n'.join(f'  - {e}' for e in errors)
        )
    primary_rows = {r['archetype_id']: r for r in all_rows if r['primary_or_alternate'] == 'primary'}
    return {a: int(primary_rows[a]['selected_sample_idx']) for a in ARCHETYPE_PRIORITY}


def _write_doc(lines):
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    DOC_PATH.write_text('\n'.join(lines) + '\n')
    log(f'[write] {DOC_PATH}')


def build_phase2d_doc_lines(selection, render_result=None):
    """Single shared report builder (state parameter: render_result is None
    in the --selection-only state, populated in the render-complete state).
    A successful render extends this report with actual audit/preview
    results; it never replaces it with a shorter, minimal document -- every
    section below is present in both states."""
    lines = []
    a = lines.append
    results = selection['results']
    is_rendered = render_result is not None

    a('# Phase 2D-A: Deterministic Archetype Selection and Raw-Artifact Preview Plan')
    a('')
    a('```')
    if is_rendered:
        a('Phase 2D-A complete.')
        a('Selection, raw audit, and review-preview generation all passed.')
    else:
        a('Phase 2D-A selection stage complete.')
        a('Raw-array audit and preview rendering pending authoritative Spark run.')
    a('```')
    a('')

    a('## 1. Scope and frozen inputs')
    a('')
    if is_rendered:
        a('This document reflects a completed raw-artifact audit and preview render (`--full`, or '
          '`--render-previews` following an earlier `--selection-only`). It reads the 86 frozen '
          'Phase-1/2A/2B/2C CSV and Markdown artifacts (checksummed before and after this stage), the '
          'plain-text raw topology CSVs (`*_pd_mt_distances.csv`, `phase_c_results.csv`), and -- for this '
          'stage only -- the raw `data_out/`/`data_out_fixed/` Spark arrays required to audit and preview '
          'the six selected samples.')
    else:
        a('This document reflects a `--selection-only` run in a lightweight checkout. It reads exclusively the '
          '86 frozen Phase-1/2A/2B/2C CSV and Markdown artifacts (checksummed before and after this stage) and '
          'the plain-text raw topology CSVs (`*_pd_mt_distances.csv`, `phase_c_results.csv`) that are ordinary '
          'git-tracked files, not gitignored raw arrays. It never touches `data_out/` or `data_out_fixed/`.')
    a('')

    a('## 2. Why selection is algorithmic rather than manual')
    a('')
    a('Every archetype is defined by an explicit eligibility rule and a closed-form score computed from the '
      'frozen Phase-1 long table and Phase-2C relationship tables. No sample was chosen by looking at an '
      'image: sample selection is computed entirely from the frozen CSV inputs, before any raw wind-field '
      'array is ever read, so the chosen samples cannot reflect appearance-based cherry-picking even in '
      'principle. The same script run twice on the same frozen inputs produces byte-identical selection '
      'output.')
    a('')

    a('## 3. Robust-z scoring convention')
    a('')
    a('`robust_z(x) = (x - median(x)) / (1.4826 * MAD(x))`, computed over the eligible-sample population for '
      'each archetype (the full 168 samples for the two global archetypes, the eligible subset otherwise). '
      'If MAD is zero the score falls back to `(x - mean(x)) / std(x)`; if both MAD and std are zero the '
      'component contributes exactly zero to every sample\'s score (see `archetype_selection_diagnostics.csv`, '
      '`event_type=robust_z_fallback`, for any such occurrence in this run). Scores are used ONLY to select '
      'samples deterministically -- they are never a method ranking.')
    a('')

    a('## 4. Archetype definitions')
    a('')
    for archetype_id in ARCHETYPE_PRIORITY:
        r = results[archetype_id]
        a(f'### {r["priority"]}. `{archetype_id}` -- {r["narrative_scope"]}')
        a('')
        a(f'- Primary methods: {r["primary_methods"]}')
        a(f'- Metrics used: {r["metrics_used"]}')
        a(f'- Score formula: `{r["score_formula"]}`')
        a(f'- Eligible samples: {len(r["eligible_idxs"])} / 168')
        a(f'- Tie-break: {r["tie_break_fields"]}')
        a('')

    a('## 5. Ranked top-10 candidates per archetype')
    a('')
    a('See `selection/archetype_score_table.csv` for the full table (rank, sample_idx, score, and every '
      'named score component with its raw value, robust-z value, and the robust-z fallback method used).')
    a('')

    a('## 6. Duplicate-resolution decisions')
    a('')
    dup_events = [d for d in selection['all_diagnostics'] if d['event_type'] == 'duplicate_skip']
    if dup_events:
        for d in dup_events:
            a(f'- `{d["archetype_id"]}`: sample_idx={d["sample_idx"]} (rank {d["rank_before_dedup"]} before '
              f'de-duplication) was skipped -- {d["detail"]}.')
    else:
        a('No duplicate-resolution events occurred in this run: every archetype\'s top-ranked eligible sample '
          'was still available when its (priority-ordered) turn came.')
    a('')

    a('## 7. Selected sample IDs and alternates')
    a('')
    for archetype_id in ARCHETYPE_PRIORITY:
        sel = selection['selection_by_archetype'][archetype_id]['selected']
        alts = selection['selection_by_archetype'][archetype_id]['alternates']
        alt_str = ', '.join(f'{alt["sample_idx"]} (score={alt["score"]:.4f})' for alt in alts)
        a(f'- **{archetype_id}**: selected sample_idx=**{sel["sample_idx"]}** (score={sel["score"]:.4f}); '
          f'alternates: {alt_str}')
    a('')

    a('## 8. Metric package')
    a('')
    a('Archetype scoring draws on four Phase-1 per-sample metrics: `pd_distance` and `mt_distance` '
      '(persistence-diagram / merge-tree topological distances, used by every archetype), plus `psnruv` and '
      '`speed_mae` (used only by `f2_balanced_vs_cnn` and `candidate_c_continuity`). These are 4 of the 22 '
      'metrics recorded in the Phase-1 long table (`unified_primary_per_sample_long.csv`); all 22, for every '
      'selected sample and `full_selected_story` method, are recorded in `selected_sample_method_values.csv` '
      'regardless of whether they drove selection.')
    a('')

    a('## 9. Raw-artifact requirements and validation')
    a('')
    a('Topology CSVs (plain text, git-tracked) were read directly and cross-checked: the independently '
      'recomputed PD/MT method means from the raw `*_pd_mt_distances.csv` / `phase_c_results.csv` sources '
      'match both the Phase-1 long table and Phase-1 `unified_primary_topology_validation.csv` within '
      'tolerance for every topology-bearing `full_selected_story` method (see `selection_validation.csv`). '
      '`selection/raw_artifact_requirements.csv` enumerates the exact repository-relative paths, expected '
      'shapes, and resolution convention for every method\'s idx/dataIN/dataGT/dataSR array.')
    a('')
    if is_rendered:
        n_methods = len(FULL_SELECTED_STORY)
        n_samples = len(set(selection['selected_sample_idx_by_archetype'].values()))
        a(f'Raw `.npy` array auditing was performed against the real `data_out/`/`data_out_fixed/` Spark '
          f'arrays and PASSED with 0 failures across all {n_methods} `full_selected_story` methods. The audit '
          f'validated: exact idx ordering (`idx == arange(168)`, never set equality); full-168-row '
          f'dataIN/dataGT/dataSR shape and finiteness (not only the {n_samples} selected rows); full-168-row '
          f'dataIN/dataGT exact alignment against the canonical CNN arrays; and per-selected-sample speed_mae '
          f'reproduction against the Phase-1 long table for every method with no exemption, bicubic included. '
          f'See `preview_audit/raw_artifact_inventory.csv`, `preview_audit/raw_alignment_validation.csv`, '
          f'`preview_audit/selected_sample_array_statistics.csv`, '
          f'`preview_audit/selected_sample_metric_reproduction.csv`, and '
          f'`preview_audit/selected_sample_artifact_manifest.csv` for the full evidence trail.')
    else:
        a('Raw `.npy` array auditing (idx validation, full-168-row dataIN/dataGT/dataSR shape and finiteness, '
          'full-168-row dataIN/dataGT alignment against the canonical CNN arrays, and per-selected-sample '
          'speed_mae reproduction with no method exempted) was **not** performed in this run, because '
          '`data_out/` and `data_out_fixed/` are gitignored and absent from this lightweight checkout by '
          'design.')
    a('')

    a('## 10. Preview inventory')
    a('')
    if is_rendered:
        a('The following review-only preview PNGs were rendered (each titled "AUDIT PREVIEW -- NOT FINAL '
          'FIGURE"):')
        a('')
        for r in render_result['render_rows']:
            a(f'- `{r["output_path"]}` (archetype={r["archetype_id"]}, sample_idx={r["sample_idx"]}, '
              f'status={r["status"]})')
    else:
        a('**No preview PNGs were generated in this run.** `selection/preview_plan.csv` records what will be '
          'rendered (one combined speed+error review PNG per selected sample, plus one contact sheet), each '
          'currently `status=pending_raw_artifacts`.')
    a('')

    a('## 11. Figure plan for Phase 2D-B')
    a('')
    a('Phase 2D-A produces only review-only audit previews, never final publication figures. '
      '`selection/preview_method_manifest.csv` records the three method groups (`baseline_story`, '
      '`descriptor_tradeoff_story`, `full_selected_story`) and `selection/figure_plan.csv` plans (does not '
      'render) the six Phase-2D-B figures, one per archetype, each `status=planned_not_rendered`. No '
      'automated TTK/ParaView rendering exists anywhere in this pipeline. Final figure production remains '
      'explicitly deferred to Phase 2D-B.')
    a('')

    a('## 12. Caveat: illustrative, not a population estimate')
    a('')
    a('The six selected samples are drawn from this fixed 168-sample benchmark using a fixed, designed set '
      'of 19 candidate methods. They are illustrative examples chosen to make specific, pre-registered '
      'archetypes concrete -- they are not a random sample, and no population-level or generalization claim '
      'should be inferred from any single selected field. These are review-only audit previews, not final '
      'publication figures; final figure production remains explicitly deferred to Phase 2D-B.')
    a('')

    a('## 13. Exact command to complete Phase 2D-A on Spark')
    a('')
    a('```')
    a(SPARK_COMMAND)
    a('```')
    a('')
    if is_rendered:
        a('This command (or `--render-previews` alone, given an existing `selection/archetype_selected_samples.csv`) '
          'is what produced this document\'s render-complete state: it read the already-written deterministic '
          'selection manifest, performed the full raw-artifact audit against the real `data_out/`/'
          '`data_out_fixed/` arrays, and -- because every audit check passed -- rendered the review-only '
          'preview PNGs listed above.')
    else:
        a('This reads the already-written `selection/archetype_selected_samples.csv` (unchanged from this run, '
          'since selection is purely CSV-driven and deterministic), performs the full raw-artifact audit against '
          'the real `data_out/`/`data_out_fixed/` arrays, and -- only if every audit check passes -- renders the '
          'review-only preview PNGs. `--render-previews` alone performs the same audit-and-render step without '
          're-running selection, provided `selection/archetype_selected_samples.csv` already exists.')
    a('')

    a('## 14. Generated files')
    a('')
    a('Selection-stage outputs (`ttk_runs_fixed/unified_candidate_analysis/phase2d/selection/`):')
    for fname in [
        'archetype_score_table.csv', 'archetype_selected_samples.csv', 'archetype_alternates.csv',
        'archetype_selection_diagnostics.csv', 'selected_sample_metric_context.csv',
        'selected_sample_method_values.csv', 'selected_sample_pairwise_preferences.csv',
        'preview_method_manifest.csv', 'preview_plan.csv', 'raw_artifact_requirements.csv', 'figure_plan.csv',
        'selection_validation.csv', 'prior_phase_immutability_check.csv',
    ]:
        a(f'- `ttk_runs_fixed/unified_candidate_analysis/phase2d/selection/{fname}`')
    a('- `docs/unified_candidate_analysis_phase2d.md` (this file)')
    a('- `logs/unified_candidate_analysis_phase2d_selection.log`')
    if is_rendered:
        a('- `logs/unified_candidate_analysis_phase2d_render.log`')
        a('')
        a('Raw-audit-and-preview outputs (`ttk_runs_fixed/unified_candidate_analysis/phase2d/preview_audit/`):')
        for fname in [
            'raw_artifact_inventory.csv', 'raw_alignment_validation.csv', 'selected_sample_array_statistics.csv',
            'selected_sample_metric_reproduction.csv', 'selected_sample_artifact_manifest.csv',
            'preview_render_validation.csv',
        ]:
            a(f'- `ttk_runs_fixed/unified_candidate_analysis/phase2d/preview_audit/{fname}`')
        a('')
        a('Preview PNGs (`ttk_runs_fixed/unified_candidate_analysis/phase2d/previews/`):')
        for r in render_result['render_rows']:
            a(f'- `{r["output_path"]}`')
    else:
        a('')
        a('Not yet generated (pending Spark): `preview_audit/*.csv`, `previews/**/*.png`.')
    a('')

    return lines


def write_phase2d_doc(selection, render_result=None):
    _write_doc(build_phase2d_doc_lines(selection, render_result))


# =============================================================================
# main()
# =============================================================================

def cmd_selection_only() -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log('=' * 88)
    log('Unified candidate analysis -- Phase 2D-A (--selection-only)')
    log(f'Repo root: {REPO_ROOT}')
    log('Read-only w.r.t. Phase-1/2A/2B/2C artifacts (86 files). Reads only frozen CSV/Markdown inputs; '
        'touches no raw .npy array.')
    log('=' * 88)

    checksums_before, file_to_phase = preflight_immutability()

    long_table = load_long_table()
    metric_direction = load_column_mapping()
    method_inventory = load_method_inventory()
    phase1_refs = load_phase1_refs()
    phase2a_refs = load_phase2a_refs()
    phase2b_refs = load_phase2b_refs()
    phase2c_refs = load_phase2c_refs()
    immutability_ledgers = load_phase2a2b_immutability_ledgers()
    column_mapping_rows = load_column_mapping_rows()
    topology_source_map = build_topology_source_map(column_mapping_rows)
    log(f'[load] Long table: {long_table["n_rows"]} rows, {len(long_table["metric_cols"])} metrics, '
        f'{len(long_table["per_sample"])} methods.')

    selection = run_selection(long_table, metric_direction, phase1_refs, phase2a_refs, phase2b_refs,
                                 phase2c_refs, immutability_ledgers, topology_source_map, method_inventory)

    write_phase2d_doc(selection)

    postflight_immutability(checksums_before, file_to_phase, SELECTION_DIR / 'prior_phase_immutability_check.csv')

    log('')
    log('=' * 88)
    six = ', '.join(f"{a}={selection['selected_sample_idx_by_archetype'][a]}" for a in ARCHETYPE_PRIORITY)
    log(f'RESULT: selection_complete_preview_render_pending_raw_artifacts. Selected samples: {six}. '
        f'{len(selection["all_validation_rows"])} validation checks PASSED. Raw-array audit and preview '
        f'rendering pending an authoritative Spark run (--render-previews or --full).')
    log('=' * 88)
    flush_log(SELECTION_LOG_PATH)
    return selection


def cmd_render_previews(selection=None) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log('=' * 88)
    log('Unified candidate analysis -- Phase 2D-A (--render-previews)')
    log(f'Repo root: {REPO_ROOT}')
    log('Requires the raw Spark arrays (data_out/, data_out_fixed/). Hard-fails if any required raw '
        'artifact is unavailable or invalid.')
    log('=' * 88)

    checksums_before, file_to_phase = preflight_immutability()

    method_inventory = load_method_inventory()
    long_table = load_long_table()
    per_sample = long_table['per_sample']
    column_mapping_rows = load_column_mapping_rows()
    topology_source_map = build_topology_source_map(column_mapping_rows)

    if selection is not None:
        selected_sample_idx_by_archetype = selection['selected_sample_idx_by_archetype']
    else:
        # Strong manifest validation (Section 6): the on-disk manifest is
        # validated for row count, archetype identity/order, unique in-range
        # indices, and required fields before it is trusted at all.
        manifest_selected = read_selected_samples_manifest()
        # A standalone --render-previews invocation has no in-memory `results`
        # (per-archetype eligibility, score formula, alternates, etc.) needed
        # for the complete Phase-2D-A report. Selection is purely CSV-driven
        # and deterministic, so the same pure (no-CSV-write) computation used
        # by --selection-only is re-run here in memory only, and cross-checked
        # against the already-validated on-disk manifest before being trusted.
        phase2c_refs = load_phase2c_refs()
        results = compute_all_archetypes(per_sample, phase2c_refs['sample_pref'], phase2c_refs['samplewise'])
        selection_by_archetype, all_diagnostics, _, _ = run_all_selections(results)
        recomputed_selected = {a: selection_by_archetype[a]['selected']['sample_idx'] for a in ARCHETYPE_PRIORITY}
        if recomputed_selected != manifest_selected:
            raise SystemExit(
                f'[hard-fail] In-memory recomputed selection does not match the validated on-disk manifest '
                f'{SELECTION_DIR / "archetype_selected_samples.csv"}: recomputed='
                f'{recomputed_selected!r} manifest={manifest_selected!r}'
            )
        selection = dict(results=results, selection_by_archetype=selection_by_archetype,
                           selected_sample_idx_by_archetype=recomputed_selected, all_diagnostics=all_diagnostics)
        selected_sample_idx_by_archetype = recomputed_selected
    log(f'[selection] Using selected samples: {selected_sample_idx_by_archetype}')

    render_result = run_render(selected_sample_idx_by_archetype, per_sample, method_inventory,
                                  topology_source_map)

    write_phase2d_doc(selection, render_result)

    postflight_immutability(checksums_before, file_to_phase, SELECTION_DIR / 'prior_phase_immutability_check.csv')

    log('')
    log('=' * 88)
    log(f'RESULT: Phase 2D-A complete. Selection, raw audit, and review-preview generation all passed. '
        f'{len(render_result["render_rows"])} preview file(s) written.')
    log('=' * 88)
    flush_log(RENDER_LOG_PATH)
    return render_result


def cmd_full():
    selection = cmd_selection_only()
    render_result = cmd_render_previews(selection=selection)
    return selection, render_result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--selection-only', action='store_true')
    mode.add_argument('--render-previews', action='store_true')
    mode.add_argument('--full', action='store_true')
    args = ap.parse_args()

    if args.selection_only:
        cmd_selection_only()
        return 0
    if args.render_previews:
        cmd_render_previews()
        return 0
    if args.full:
        cmd_full()
        return 0
    return 1


if __name__ == '__main__':
    sys.exit(main())
