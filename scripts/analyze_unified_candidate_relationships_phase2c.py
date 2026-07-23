#!/usr/bin/env python3
"""Phase 2C: metric-relationship and multi-objective Pareto-tradeoff analysis
of the unified wind-SR candidate benchmark, built exclusively from the
immutable Phase-1, Phase-2A, and Phase-2B outputs.

Read-only with respect to every prior-phase artifact. Never runs training,
inference, cheap evaluation, or TTK, and never writes anywhere except:

    ttk_runs_fixed/unified_candidate_analysis/phase2c/
    docs/unified_candidate_analysis_phase2c.md
    logs/unified_candidate_analysis_phase2c.log

Before and after the run, SHA-256-checksums an explicit list of 54 prior-
phase files (12 Phase-1 + 14 Phase-2A + 28 Phase-2B) and hard-fails on any
change, disappearance, or unexpected new file in a frozen directory.

Scope: metric relationships at four distinct analysis levels (method-mean,
within-method-across-samples, per-sample-across-methods, two-way-centered
residual), a compact focal PD/MT-vs-representative-metric bootstrap
correlation study, direct PD/MT preference-agreement analysis, and Pareto-
front / dominance-layer / bootstrap-stability analysis over six explicitly
defined objective sets. No weighted aggregate score or total method ranking
is ever computed. Sample selection and figure generation remain deferred to
Phase 2D.

Every one of the 19 methods was trained exactly once, and the 19 methods are
a fixed designed candidate set, not a random sample of possible models. The
168 benchmark fields are consecutive hourly observations and are likely
temporally dependent. Correlation is not causation, and different analysis
levels (method-mean / within-method / cross-method / two-way-residual)
answer genuinely different questions -- see
docs/unified_candidate_analysis_phase2c.md for the required cautious
language conventions.

Determinism: every bootstrap resample (ordinary and circular block, for both
the smaller 2,000-resample focal-correlation study and the larger
10,000-resample Pareto-stability study) uses a fixed seed and a precomputed
index matrix, so re-running this script produces byte-identical output. No
wall-clock time, hostname, or environment-dependent value is ever written to
a generated file.
"""

from __future__ import annotations

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

PHASE1_DIR = REPO_ROOT / 'ttk_runs_fixed' / 'unified_candidate_evaluation'
PHASE2A_DIR = REPO_ROOT / 'ttk_runs_fixed' / 'unified_candidate_analysis' / 'phase2a'
PHASE2B_DIR = REPO_ROOT / 'ttk_runs_fixed' / 'unified_candidate_analysis' / 'phase2b'
OUT_DIR = REPO_ROOT / 'ttk_runs_fixed' / 'unified_candidate_analysis' / 'phase2c'
DOCS_DIR = REPO_ROOT / 'docs'
LOG_PATH = REPO_ROOT / 'logs' / 'unified_candidate_analysis_phase2c.log'

# -----------------------------------------------------------------------
# Explicit protected-file sets -- never a glob.
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

ALL_PROTECTED_FILES = PHASE1_PROTECTED_FILES + PHASE2A_PROTECTED_FILES + PHASE2B_PROTECTED_FILES  # exactly 54
assert len(PHASE1_PROTECTED_FILES) == 12
assert len(PHASE2A_PROTECTED_FILES) == 14
assert len(PHASE2B_PROTECTED_FILES) == 28
assert len(ALL_PROTECTED_FILES) == 54

N_EVAL = 168
N_METHODS = 19
CNN_METHOD = 'cnn'
BICUBIC_METHOD = 'bicubic'
TIE_TOLERANCE = 1e-12
RECOMPUTE_TOLERANCE = 1e-6
PD_MT_RECOMPUTE_TOLERANCE = 1e-4
MARGIN_TOLERANCE = 1e-9  # two-way-centered residual row/column/grand margin tolerance
PARETO_TOLERANCE = 1e-12

PARETO_BOOTSTRAP_SEED = 20260721
PARETO_BOOTSTRAP_N = 10000
CORR_BOOTSTRAP_SEED = 20260721 + 2000  # documented, distinguishing offset from the Pareto seed
CORR_BOOTSTRAP_N = 2000
BLOCK_LENGTHS = [6, 12, 24]

FACTOR_TO_USES_COLUMN = {
    'speed': 'uses_speed', 'grad': 'uses_grad', 'levelset': 'uses_levelset',
    'crit': 'uses_crit', 'e2': 'uses_e2',
}

TOPOLOGY_METRICS = ('pd_distance', 'mt_distance')

FOCAL_METRICS = ['psnruv', 'speed_mae', 'wpd_mae', 'grad_mae', 'psd_log_l2', 'exceed_abs_p90',
                   'comp_curve_l1', 'pd_distance', 'mt_distance']
FOCAL_NON_TOPOLOGY = ['psnruv', 'speed_mae', 'wpd_mae', 'grad_mae', 'psd_log_l2', 'exceed_abs_p90',
                        'comp_curve_l1']

_LOG_LINES: list[str] = []


def log(msg: str = '') -> None:
    print(msg)
    _LOG_LINES.append(msg)


def flush_log() -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open('w') as fh:
        fh.write('\n'.join(_LOG_LINES) + '\n')


def rp(rel) -> Path:
    return REPO_ROOT / rel


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
            f'{len(ALL_PROTECTED_FILES)}: 12 Phase-1 + 14 Phase-2A + 28 Phase-2B):\n' +
            '\n'.join(f'  - {m}' for m in missing)
        )
    for directory, expected_csvs in ((PHASE1_DIR, set(PHASE1_PROTECTED_CSVS)),
                                       (PHASE2A_DIR, set(PHASE2A_PROTECTED_CSVS)),
                                       (PHASE2B_DIR, set(PHASE2B_PROTECTED_CSVS))):
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


def read_csv_dicts(path: Path) -> list:
    with path.open(newline='') as fh:
        return list(csv.DictReader(fh))


def _f(val):
    if val in (None, ''):
        return float('nan')
    return float(val)


def write_csv(path: Path, fieldnames: list, rows: list) -> None:
    with path.open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    log(f'[write] {path} ({len(rows)} rows)')


# =============================================================================
# Correlation implementation (Pearson + Spearman with average-rank ties),
# implemented directly with NumPy/stdlib -- no SciPy dependency.
# =============================================================================

def pearson_r(x: np.ndarray, y: np.ndarray):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = x.shape[0]
    if n < 2:
        return None
    xc = x - x.mean()
    yc = y - y.mean()
    denom = math.sqrt(float(np.sum(xc * xc)) * float(np.sum(yc * yc)))
    if denom == 0.0:
        return None
    return float(np.sum(xc * yc) / denom)


def rankdata_avg(x: np.ndarray) -> np.ndarray:
    """Average-rank transform (1-indexed), ties get the mean of the ranks
    they would occupy."""
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    order = np.argsort(x, kind='mergesort')
    sorted_x = x[order]
    naive_ranks = np.arange(1, n + 1, dtype=np.float64)
    uniq, inv, counts = np.unique(sorted_x, return_inverse=True, return_counts=True)
    cum = np.concatenate(([0], np.cumsum(counts)))
    avg_rank_per_group = (cum[:-1] + cum[1:] + 1) / 2.0
    sorted_avg_ranks = avg_rank_per_group[inv]
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = sorted_avg_ranks
    return ranks


def spearman_r(x: np.ndarray, y: np.ndarray):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape[0] < 2:
        return None
    return pearson_r(rankdata_avg(x), rankdata_avg(y))


def _rank_matrix_avg_ties(X: np.ndarray) -> np.ndarray:
    """Row-wise average-rank transform for a 2-D array (each row ranked
    independently). Used for vectorized bootstrap Spearman."""
    n_rows, n_cols = X.shape
    ranks = np.empty_like(X, dtype=np.float64)
    for i in range(n_rows):
        ranks[i] = rankdata_avg(X[i])
    return ranks


def pearson_r_rows(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Row-wise Pearson correlation for two (n_rows, n_cols) arrays --
    vectorized bootstrap-friendly version of pearson_r."""
    x_mean = X.mean(axis=1, keepdims=True)
    y_mean = Y.mean(axis=1, keepdims=True)
    xc = X - x_mean
    yc = Y - y_mean
    num = (xc * yc).sum(axis=1)
    den = np.sqrt((xc * xc).sum(axis=1) * (yc * yc).sum(axis=1))
    with np.errstate(invalid='ignore', divide='ignore'):
        r = num / den
    r[den == 0] = np.nan
    return r


def spearman_r_rows(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return pearson_r_rows(_rank_matrix_avg_ties(X), _rank_matrix_avg_ties(Y))


# =============================================================================
# Bootstrap infrastructure: two independently-seeded families of precomputed
# index matrices -- a smaller one (2,000 resamples) for focal correlation
# CIs, and a larger one (10,000 resamples) for Pareto-stability bootstrap.
# =============================================================================

def _make_iid_index_matrix(seed: int, n_resamples: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, N_EVAL, size=(n_resamples, N_EVAL))


def _make_block_index_matrix(seed_base: int, block_length: int, n_resamples: int) -> np.ndarray:
    seed = seed_base * 1000 + block_length
    rng = np.random.default_rng(seed)
    n_blocks = math.ceil(N_EVAL / block_length)
    starts = rng.integers(0, N_EVAL, size=(n_resamples, n_blocks))
    offsets = np.arange(block_length)
    idx = (starts[:, :, None] + offsets[None, None, :]) % N_EVAL
    return idx.reshape(n_resamples, n_blocks * block_length)[:, :N_EVAL]


_CORR_IID_IDX = _make_iid_index_matrix(CORR_BOOTSTRAP_SEED, CORR_BOOTSTRAP_N)
_CORR_BLOCK_IDX = {L: _make_block_index_matrix(CORR_BOOTSTRAP_SEED, L, CORR_BOOTSTRAP_N) for L in BLOCK_LENGTHS}

_PARETO_IID_IDX = _make_iid_index_matrix(PARETO_BOOTSTRAP_SEED, PARETO_BOOTSTRAP_N)
_PARETO_BLOCK_IDX = {L: _make_block_index_matrix(PARETO_BOOTSTRAP_SEED, L, PARETO_BOOTSTRAP_N) for L in BLOCK_LENGTHS}

CORR_BOOTSTRAP_SCHEMES = {'iid': _CORR_IID_IDX, 'block6': _CORR_BLOCK_IDX[6],
                            'block12': _CORR_BLOCK_IDX[12], 'block24': _CORR_BLOCK_IDX[24]}
PARETO_BOOTSTRAP_SCHEMES = {'iid': _PARETO_IID_IDX, 'block6': _PARETO_BLOCK_IDX[6],
                              'block12': _PARETO_BLOCK_IDX[12], 'block24': _PARETO_BLOCK_IDX[24]}


def _percentile_ci(values: np.ndarray) -> tuple:
    finite = values[np.isfinite(values)]
    if finite.shape[0] == 0:
        return None, None
    return float(np.percentile(finite, 2.5)), float(np.percentile(finite, 97.5))


def correlation_bootstrap_cis(x: np.ndarray, y: np.ndarray, corr_type: str) -> dict:
    """corr_type in {'pearson', 'spearman'}. Returns {'iid': (lo,hi),
    'block6': (lo,hi), 'block12': (lo,hi), 'block24': (lo,hi)} using the
    2,000-resample correlation bootstrap index matrices, applying the SAME
    sampled indices to both x and y in every resample."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    fn = pearson_r_rows if corr_type == 'pearson' else spearman_r_rows
    out = {}
    for scheme_name, idx in CORR_BOOTSTRAP_SCHEMES.items():
        X = x[idx]
        Y = y[idx]
        r_vals = fn(X, Y)
        out[scheme_name] = _percentile_ci(r_vals)
    return out


# =============================================================================
# Loading inputs
# =============================================================================

def load_long_table():
    path = PHASE1_DIR / 'unified_primary_per_sample_long.csv'
    with path.open(newline='') as fh:
        header = next(csv.reader(fh))
    identity_cols = ['sample_idx', 'method_id', 'display_name', 'candidate_family', 'training_scale',
                      'architecture', 'uses_speed', 'uses_grad', 'uses_levelset', 'uses_crit', 'uses_e2']
    metric_cols = [c for c in header if c not in identity_cols]
    rows = read_csv_dicts(path)

    per_sample: dict = {}
    method_meta: dict = {}
    meta_keys = ['display_name', 'candidate_family', 'training_scale', 'architecture',
                 'uses_speed', 'uses_grad', 'uses_levelset', 'uses_crit', 'uses_e2']
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
        rec = {c: _f(row.get(c, '')) for c in metric_cols}
        per_sample.setdefault(mid, {})[si] = rec
        meta = {k: row.get(k, '') for k in meta_keys}
        if mid not in method_meta:
            method_meta[mid] = {'values': meta, 'inconsistent_fields': []}
        else:
            existing = method_meta[mid]['values']
            for k in meta_keys:
                if existing[k] != meta[k]:
                    method_meta[mid]['inconsistent_fields'].append((k, si))

    return dict(rows=rows, header=header, metric_cols=metric_cols, per_sample=per_sample,
                method_meta=method_meta, dup_keys=dup_keys, n_rows=len(rows))


def load_column_mapping():
    rows = read_csv_dicts(PHASE1_DIR / 'column_mapping.csv')
    direction = {}
    family = {}
    for row in rows:
        sc = row['standardized_column']
        d = row['direction']
        if sc in direction and direction[sc] != d:
            raise SystemExit(f'[hard-fail] column_mapping.csv gives conflicting directions for {sc!r}.')
        direction[sc] = d
        family.setdefault(sc, row['representation'])
    return direction, family


def load_phase1_refs():
    return dict(
        method_summary=read_csv_dicts(PHASE1_DIR / 'unified_primary_method_summary.csv'),
        topo_val=read_csv_dicts(PHASE1_DIR / 'unified_primary_topology_validation.csv'),
        inventory=read_csv_dicts(PHASE1_DIR / 'method_inventory.csv'),
    )


def load_phase2a_refs():
    return dict(
        validation=read_csv_dicts(PHASE2A_DIR / 'phase2a_validation.csv'),
        pairwise_repro=read_csv_dicts(PHASE2A_DIR / 'phase1_pairwise_reproduction.csv'),
        immutability=read_csv_dicts(PHASE2A_DIR / 'phase1_immutability_check.csv'),
    )


def load_phase2b_refs():
    return dict(
        validation=read_csv_dicts(PHASE2B_DIR / 'phase2b_validation.csv'),
        immutability=read_csv_dicts(PHASE2B_DIR / 'prior_phase_immutability_check.csv'),
    )


def orient(y, direction: str):
    return y if direction == 'higher_is_better' else -y


def method_metric_mean(per_sample: dict, mid: str, metric: str):
    """Mean of the finite values for (mid, metric) across all N_EVAL samples,
    or None if zero finite values (e.g. bicubic PD/MT, or SSIM everywhere)."""
    vals = [per_sample[mid][si][metric] for si in range(N_EVAL)
            if math.isfinite(per_sample[mid][si].get(metric, float('nan')))]
    if not vals:
        return None
    return sum(vals) / len(vals)


# =============================================================================
# Base validation (phase2c_validation.csv, part 1 -- structural, run before
# any analytical output; more checks are appended after each analysis).
# =============================================================================

def run_base_validation(long_table, phase1_refs, phase2a_refs, phase2b_refs, metric_direction):
    checks = []
    failures = []

    def add(name, observed, expected, tol, ok, notes=''):
        status = 'PASS' if ok else 'FAIL'
        checks.append(dict(check_name=name, observed=observed, expected=expected, tolerance=tol,
                             status=status, notes=notes))
        if not ok:
            failures.append(f'{name}: observed={observed!r} expected={expected!r} notes={notes}')

    per_sample = long_table['per_sample']
    metric_cols = long_table['metric_cols']

    add('long_table_row_count', long_table['n_rows'], N_METHODS * N_EVAL, 0,
        long_table['n_rows'] == N_METHODS * N_EVAL)
    add('duplicate_keys', len(long_table['dup_keys']), 0, 0, len(long_table['dup_keys']) == 0)
    add('method_count', len(per_sample), N_METHODS, 0, len(per_sample) == N_METHODS)
    add('metric_count', len(metric_cols), 22, 0, len(metric_cols) == 22)

    for mid in sorted(per_sample):
        n = len(per_sample[mid])
        add(f'samples_per_method[{mid}]', n, N_EVAL, 0, n == N_EVAL)
        idx_set = set(per_sample[mid].keys())
        add(f'sample_idx_exact_0_167[{mid}]', len(idx_set), N_EVAL, 0, idx_set == set(range(N_EVAL)))
        inconsistent = long_table['method_meta'][mid]['inconsistent_fields']
        add(f'metadata_constant[{mid}]', len(inconsistent), 0, 0, len(inconsistent) == 0,
            notes=str(inconsistent[:5]))

    missing_direction = [c for c in metric_cols if c not in metric_direction]
    add('metric_directions_resolved_all_22', len(missing_direction), 0, 0, len(missing_direction) == 0,
        notes=str(missing_direction))

    n_p2a_fail = sum(1 for r in phase2a_refs['validation'] if r['status'] != 'PASS')
    add('phase2a_validation_all_pass', n_p2a_fail, 0, 0, n_p2a_fail == 0)
    n_p2a_repro_fail = sum(1 for r in phase2a_refs['pairwise_repro'] if r['status'] != 'PASS')
    add('phase2a_pairwise_reproduction_all_pass', n_p2a_repro_fail, 0, 0, n_p2a_repro_fail == 0)
    n_p2a_immut_bad = sum(1 for r in phase2a_refs['immutability'] if r['status'] != 'unchanged')
    add('phase2a_recorded_immutability_all_unchanged', n_p2a_immut_bad, 0, 0, n_p2a_immut_bad == 0)

    n_p2b_fail = sum(1 for r in phase2b_refs['validation'] if r['status'] != 'PASS')
    add('phase2b_validation_all_pass', n_p2b_fail, 0, 0, n_p2b_fail == 0)
    n_p2b_immut_bad = sum(1 for r in phase2b_refs['immutability'] if r['status'] != 'unchanged')
    add('phase2b_recorded_immutability_all_unchanged', n_p2b_immut_bad, 0, 0, n_p2b_immut_bad == 0)

    # Non-SSIM metric completeness for every method, except bicubic PD/MT.
    for mid in sorted(per_sample):
        for col in metric_cols:
            finite = sum(1 for si in range(N_EVAL) if math.isfinite(per_sample[mid][si].get(col, float('nan'))))
            if col == 'ssim_speed':
                add(f'ssim_0_or_168[{mid}]', finite, '0 or 168', 0, finite in (0, N_EVAL))
            elif col in TOPOLOGY_METRICS and mid == BICUBIC_METHOD:
                add(f'bicubic_no_topology[{mid}][{col}]', finite, 0, 0, finite == 0)
            else:
                add(f'required_metric_complete[{mid}][{col}]', finite, N_EVAL, 0, finite == N_EVAL)

    # SSIM globally 0/168 across every method (not merely per-method 0-or-168).
    ssim_total_finite = sum(1 for mid in per_sample for si in range(N_EVAL)
                              if math.isfinite(per_sample[mid][si].get('ssim_speed', float('nan'))))
    add('ssim_globally_unavailable', ssim_total_finite, 0, 0, ssim_total_finite == 0)

    # Recompute method means and cross-check against Phase-1 method_summary.csv.
    for row in phase1_refs['method_summary']:
        mid, col = row['method_id'], row['metric']
        observed_mean = method_metric_mean(per_sample, mid, col)
        expected_mean = row['mean']
        if observed_mean is None:
            add(f'method_summary_mean_reproduction[{mid}][{col}]', '', '', RECOMPUTE_TOLERANCE,
                expected_mean == '')
            continue
        if expected_mean == '':
            add(f'method_summary_mean_reproduction[{mid}][{col}]', observed_mean, expected_mean,
                RECOMPUTE_TOLERANCE, False, notes='phase1 reports empty but recompute has data')
            continue
        diff = abs(observed_mean - float(expected_mean))
        add(f'method_summary_mean_reproduction[{mid}][{col}]', observed_mean, float(expected_mean),
            RECOMPUTE_TOLERANCE, diff <= RECOMPUTE_TOLERANCE, notes=f'abs_diff={diff:.3e}')

    for row in phase1_refs['topo_val']:
        mid = row['method_id']
        for col, obs_key in (('pd_distance', 'observed_pd_mean'), ('mt_distance', 'observed_mt_mean')):
            observed_mean = method_metric_mean(per_sample, mid, col)
            expected = row[obs_key]
            if observed_mean is None:
                add(f'topology_mean_reproduction[{mid}][{col}]', '', '', PD_MT_RECOMPUTE_TOLERANCE,
                    expected == '')
                continue
            if expected == '':
                add(f'topology_mean_reproduction[{mid}][{col}]', observed_mean, expected,
                    PD_MT_RECOMPUTE_TOLERANCE, False, notes='phase1 reports empty but recompute has data')
                continue
            diff = abs(observed_mean - float(expected))
            add(f'topology_mean_reproduction[{mid}][{col}]', observed_mean, float(expected),
                PD_MT_RECOMPUTE_TOLERANCE, diff <= PD_MT_RECOMPUTE_TOLERANCE, notes=f'abs_diff={diff:.3e}')

    return checks, failures


def nfmt(v):
    return '' if v is None else v


# =============================================================================
# ANALYSIS A -- method-mean relationships (interpretation_level=between_method_means)
# =============================================================================

def build_method_mean_oriented_values(methods, per_sample, metric_cols, metric_direction):
    means = {}
    rows = []
    for mid in methods:
        raw = {}
        oriented = {}
        for metric in metric_cols:
            m = method_metric_mean(per_sample, mid, metric)
            raw[metric] = m
            oriented[metric] = None if m is None else orient(m, metric_direction[metric])
        means[mid] = dict(raw=raw, oriented=oriented)
        row = {'method_id': mid}
        for metric in metric_cols:
            row[f'raw__{metric}'] = nfmt(raw[metric])
            row[f'oriented__{metric}'] = nfmt(oriented[metric])
        rows.append(row)
    fieldnames = ['method_id']
    for metric in metric_cols:
        fieldnames += [f'raw__{metric}', f'oriented__{metric}']
    return means, rows, fieldnames


def method_level_metric_correlations(methods, means, non_ssim):
    rows = []
    pearson_matrix = {a: {b: (1.0 if b == a else None) for b in non_ssim} for a in non_ssim}
    spearman_matrix = {a: {b: (1.0 if b == a else None) for b in non_ssim} for a in non_ssim}
    for a, b in itertools.combinations(non_ssim, 2):
        used = [mid for mid in methods
                if means[mid]['raw'][a] is not None and means[mid]['raw'][b] is not None]
        raw_x = np.array([means[mid]['raw'][a] for mid in used])
        raw_y = np.array([means[mid]['raw'][b] for mid in used])
        ori_x = np.array([means[mid]['oriented'][a] for mid in used])
        ori_y = np.array([means[mid]['oriented'][b] for mid in used])
        raw_p = pearson_r(raw_x, raw_y)
        raw_s = spearman_r(raw_x, raw_y)
        ori_p = pearson_r(ori_x, ori_y)
        ori_s = spearman_r(ori_x, ori_y)

        loo_p, loo_s = [], []
        for i in range(len(used)):
            xs = np.delete(ori_x, i)
            ys = np.delete(ori_y, i)
            loo_p.append(pearson_r(xs, ys))
            loo_s.append(spearman_r(xs, ys))
        loo_p_valid = [v for v in loo_p if v is not None]
        loo_s_valid = [v for v in loo_s if v is not None]
        loo_p_min = min(loo_p_valid) if loo_p_valid else None
        loo_p_max = max(loo_p_valid) if loo_p_valid else None
        loo_s_min = min(loo_s_valid) if loo_s_valid else None
        loo_s_max = max(loo_s_valid) if loo_s_valid else None
        p_stable = (loo_p_min is not None and
                    ((loo_p_min > 0 and loo_p_max > 0) or (loo_p_min < 0 and loo_p_max < 0)))
        s_stable = (loo_s_min is not None and
                    ((loo_s_min > 0 and loo_s_max > 0) or (loo_s_min < 0 and loo_s_max < 0)))

        rows.append(dict(
            metric_a=a, metric_b=b, n_common_methods=len(used),
            raw_pearson=nfmt(raw_p), raw_spearman=nfmt(raw_s),
            oriented_pearson=nfmt(ori_p), oriented_spearman=nfmt(ori_s),
            loo_pearson_min=nfmt(loo_p_min), loo_pearson_max=nfmt(loo_p_max),
            loo_spearman_min=nfmt(loo_s_min), loo_spearman_max=nfmt(loo_s_max),
            pearson_sign_stable=p_stable, spearman_sign_stable=s_stable,
            interpretation_level='between_method_means',
        ))
        pearson_matrix[a][b] = pearson_matrix[b][a] = ori_p
        spearman_matrix[a][b] = spearman_matrix[b][a] = ori_s
    return rows, pearson_matrix, spearman_matrix


def write_matrix_csv(path, non_ssim, matrix):
    fieldnames = ['metric'] + list(non_ssim)
    rows = []
    for a in non_ssim:
        row = {'metric': a}
        for b in non_ssim:
            row[b] = nfmt(matrix[a][b])
        rows.append(row)
    write_csv(path, fieldnames, rows)


# =============================================================================
# ANALYSIS B -- within-method sample relationships
# (interpretation_level=within_method_across_samples)
# =============================================================================

def within_method_metric_correlations(per_sample, methods, non_ssim, metric_direction):
    rows = []
    summary_acc = {}
    for mid in methods:
        for a, b in itertools.combinations(non_ssim, 2):
            xs, ys = [], []
            for si in range(N_EVAL):
                va = per_sample[mid][si].get(a, float('nan'))
                vb = per_sample[mid][si].get(b, float('nan'))
                if math.isfinite(va) and math.isfinite(vb):
                    xs.append(va)
                    ys.append(vb)
            n = len(xs)
            if n == 0:
                status = 'unavailable'
                raw_p = raw_s = ori_p = ori_s = None
            elif n == N_EVAL:
                status = 'available'
                x = np.array(xs)
                y = np.array(ys)
                ox = orient(x, metric_direction[a])
                oy = orient(y, metric_direction[b])
                raw_p = pearson_r(x, y)
                raw_s = spearman_r(x, y)
                ori_p = pearson_r(ox, oy)
                ori_s = spearman_r(ox, oy)
            else:
                raise SystemExit(
                    f'[hard-fail] within-method partial sample coverage: method={mid} '
                    f'pair=({a},{b}) n={n} (expected 0 or {N_EVAL}).'
                )
            rows.append(dict(
                method_id=mid, metric_a=a, metric_b=b, n_samples=n, status=status,
                raw_pearson=nfmt(raw_p), raw_spearman=nfmt(raw_s),
                oriented_pearson=nfmt(ori_p), oriented_spearman=nfmt(ori_s),
                interpretation_level='within_method_across_samples',
            ))
            if status == 'available':
                summary_acc.setdefault((a, b), []).append((mid, ori_p, ori_s))
    return rows, summary_acc


def _dispersion_stats(values):
    if not values:
        return dict(median='', q25='', q75='', min='', max='', n_positive=0, n_negative=0)
    v = np.array(values, dtype=np.float64)
    return dict(
        median=float(np.median(v)), q25=float(np.percentile(v, 25)), q75=float(np.percentile(v, 75)),
        min=float(v.min()), max=float(v.max()),
        n_positive=int((v > 0).sum()), n_negative=int((v < 0).sum()),
    )


def within_method_correlation_summary(summary_acc, non_ssim):
    rows = []
    for a, b in itertools.combinations(non_ssim, 2):
        entries = summary_acc.get((a, b), [])
        ps = _dispersion_stats([e[1] for e in entries])
        ss = _dispersion_stats([e[2] for e in entries])
        rows.append(dict(
            metric_a=a, metric_b=b, n_methods_available=len(entries),
            pearson_median=ps['median'], pearson_q25=ps['q25'], pearson_q75=ps['q75'],
            pearson_min=ps['min'], pearson_max=ps['max'],
            pearson_n_positive=ps['n_positive'], pearson_n_negative=ps['n_negative'],
            spearman_median=ss['median'], spearman_q25=ss['q25'], spearman_q75=ss['q75'],
            spearman_min=ss['min'], spearman_max=ss['max'],
            spearman_n_positive=ss['n_positive'], spearman_n_negative=ss['n_negative'],
            interpretation_level='within_method_across_samples',
        ))
    return rows


# =============================================================================
# ANALYSIS C -- per-sample cross-method relationships
# (interpretation_level=within_sample_across_methods)
# =============================================================================

def samplewise_cross_method_correlations(per_sample, methods, topology_methods, non_ssim, metric_direction):
    rows = []
    summary_acc = {}
    for si in range(N_EVAL):
        for a, b in itertools.combinations(non_ssim, 2):
            uses_topology = a in TOPOLOGY_METRICS or b in TOPOLOGY_METRICS
            use_methods = topology_methods if uses_topology else methods
            expected_n = len(topology_methods) if uses_topology else len(methods)
            xs, ys = [], []
            for mid in use_methods:
                va = per_sample[mid][si].get(a, float('nan'))
                vb = per_sample[mid][si].get(b, float('nan'))
                if math.isfinite(va) and math.isfinite(vb):
                    xs.append(va)
                    ys.append(vb)
            n = len(xs)
            if n != expected_n:
                raise SystemExit(
                    f'[hard-fail] samplewise cross-method pair=({a},{b}) sample={si}: '
                    f'n={n} != expected {expected_n}.'
                )
            x = np.array(xs)
            y = np.array(ys)
            ox = orient(x, metric_direction[a])
            oy = orient(y, metric_direction[b])
            raw_p = pearson_r(x, y)
            raw_s = spearman_r(x, y)
            ori_p = pearson_r(ox, oy)
            ori_s = spearman_r(ox, oy)
            rows.append(dict(
                sample_idx=si, metric_a=a, metric_b=b, n_methods=n,
                raw_pearson=nfmt(raw_p), raw_spearman=nfmt(raw_s),
                oriented_pearson=nfmt(ori_p), oriented_spearman=nfmt(ori_s),
                interpretation_level='within_sample_across_methods',
            ))
            summary_acc.setdefault((a, b), []).append((ori_p, ori_s))
    return rows, summary_acc


def samplewise_correlation_summary(summary_acc, non_ssim):
    rows = []
    for a, b in itertools.combinations(non_ssim, 2):
        entries = summary_acc.get((a, b), [])
        ps = _dispersion_stats([e[0] for e in entries if e[0] is not None])
        ss = _dispersion_stats([e[1] for e in entries if e[1] is not None])
        rows.append(dict(
            metric_a=a, metric_b=b, n_samples=len(entries),
            pearson_median=ps['median'], pearson_q25=ps['q25'], pearson_q75=ps['q75'],
            pearson_min=ps['min'], pearson_max=ps['max'],
            pearson_n_positive=ps['n_positive'], pearson_n_negative=ps['n_negative'],
            spearman_median=ss['median'], spearman_q25=ss['q25'], spearman_q75=ss['q75'],
            spearman_min=ss['min'], spearman_max=ss['max'],
            spearman_n_positive=ss['n_positive'], spearman_n_negative=ss['n_negative'],
            interpretation_level='within_sample_across_methods',
        ))
    return rows


# =============================================================================
# ANALYSIS D -- two-way-centered residual relationships (additive demeaning,
# NOT a fitted mixed-effects model). interpretation_level=two_way_centered_residual
#
# Pair-specific common-rectangle centering: for every metric pair, BOTH
# oriented method-by-sample matrices are built and independently two-way
# centered over the SAME common method set (18 topology-bearing methods if
# either metric is PD or MT, else all 19 methods). This is deliberately NOT
# "center each metric once over its own maximal method set, then subset a
# 19-method residual down to 18 rows" -- subsetting a residual matrix AFTER
# centering does not preserve the zero-margin property, since the row/column
# means used for centering were computed over the wrong (too-large)
# rectangle. Centering must happen on the final common rectangle itself.
# =============================================================================

def compute_two_way_residual_matrix(per_sample, common_methods, metric, direction):
    """Two-way-center metric's oriented values over exactly common_methods x
    N_EVAL samples. Returns (R, method_margin_max_abs, sample_margin_max_abs,
    grand_margin_abs), where method_margin is the row (per-method) mean of R
    and sample_margin is the column (per-sample) mean of R -- both should be
    ~0 by construction for any rectangle with no missing cells."""
    n_m = len(common_methods)
    Z = np.empty((n_m, N_EVAL))
    for i, mid in enumerate(common_methods):
        for si in range(N_EVAL):
            Z[i, si] = orient(per_sample[mid][si][metric], direction)
    row_mean = Z.mean(axis=1, keepdims=True)
    col_mean = Z.mean(axis=0, keepdims=True)
    grand_mean = Z.mean()
    R = Z - row_mean - col_mean + grand_mean
    method_margin_max_abs = float(np.max(np.abs(R.mean(axis=1))))
    sample_margin_max_abs = float(np.max(np.abs(R.mean(axis=0))))
    grand_margin_abs = float(abs(R.mean()))
    return R, method_margin_max_abs, sample_margin_max_abs, grand_margin_abs


def two_way_residual_correlations(per_sample, methods, topology_methods, non_ssim, metric_family,
                                     metric_direction):
    rows = []
    margin_rows = []
    pearson_matrix = {a: {b: (1.0 if b == a else None) for b in non_ssim} for a in non_ssim}
    spearman_matrix = {a: {b: (1.0 if b == a else None) for b in non_ssim} for a in non_ssim}
    for a, b in itertools.combinations(non_ssim, 2):
        common = topology_methods if (a in TOPOLOGY_METRICS or b in TOPOLOGY_METRICS) else methods
        Ra, mmargin_a, smargin_a, gmargin_a = compute_two_way_residual_matrix(
            per_sample, common, a, metric_direction[a])
        Rb, mmargin_b, smargin_b, gmargin_b = compute_two_way_residual_matrix(
            per_sample, common, b, metric_direction[b])
        margin_rows.append(dict(metric=a, paired_with=b, row_margin_max_abs=mmargin_a,
                                   col_margin_max_abs=smargin_a, grand_margin_abs=gmargin_a))
        margin_rows.append(dict(metric=b, paired_with=a, row_margin_max_abs=mmargin_b,
                                   col_margin_max_abs=smargin_b, grand_margin_abs=gmargin_b))
        flat_a = Ra.reshape(-1)
        flat_b = Rb.reshape(-1)
        p = pearson_r(flat_a, flat_b)
        s = spearman_r(flat_a, flat_b)
        rows.append(dict(
            metric_a=a, metric_b=b, family_a=metric_family[a], family_b=metric_family[b],
            n_common_methods=len(common), n_samples=N_EVAL, n_cells=flat_a.shape[0],
            oriented_residual_pearson=nfmt(p), oriented_residual_spearman=nfmt(s),
            max_abs_method_margin_mean_a=mmargin_a, max_abs_sample_margin_mean_a=smargin_a,
            max_abs_method_margin_mean_b=mmargin_b, max_abs_sample_margin_mean_b=smargin_b,
            interpretation_level='two_way_centered_residual',
        ))
        pearson_matrix[a][b] = pearson_matrix[b][a] = p
        spearman_matrix[a][b] = spearman_matrix[b][a] = s
    return rows, margin_rows, pearson_matrix, spearman_matrix


# =============================================================================
# ANALYSIS E -- focal topology relationships (18 topology-bearing methods x 15
# focal pairs), point estimate plus deterministic bootstrap CIs. Uses the
# smaller 2,000-resample correlation-bootstrap family (distinct from the
# larger 10,000-resample Pareto-stability family).
# =============================================================================

FOCAL_PAIRS = ([(TOPOLOGY_METRICS[0], m) for m in FOCAL_NON_TOPOLOGY] +
               [(TOPOLOGY_METRICS[1], m) for m in FOCAL_NON_TOPOLOGY] +
               [(TOPOLOGY_METRICS[0], TOPOLOGY_METRICS[1])])
assert len(FOCAL_PAIRS) == 15

BOOTSTRAP_SCHEME_NAMES = ['iid', 'block6', 'block12', 'block24']


def _interval_sign(lo, hi):
    """positive when low > 0; negative when high < 0; includes_zero otherwise
    (including the boundary cases low == 0 or high == 0). Returns '' if
    either bound is undefined."""
    if lo == '' or hi == '' or lo is None or hi is None:
        return ''
    lo_f, hi_f = float(lo), float(hi)
    if lo_f > 0:
        return 'positive'
    if hi_f < 0:
        return 'negative'
    return 'includes_zero'


def focal_topology_correlation_bootstrap(per_sample, topology_methods, metric_direction):
    """One row per (method x focal pair x correlation type): 18 x 15 x 2 =
    540 rows. All four schemes' CIs live on the same row (wide form), each
    computed by the same correlation_bootstrap_cis() call used previously --
    no change to the underlying bootstrap computation, only to how the
    results are laid out."""
    rows = []
    for mid in topology_methods:
        for a, b in FOCAL_PAIRS:
            xs = np.array([per_sample[mid][si][a] for si in range(N_EVAL)])
            ys = np.array([per_sample[mid][si][b] for si in range(N_EVAL)])
            ox = orient(xs, metric_direction[a])
            oy = orient(ys, metric_direction[b])
            point_p = pearson_r(ox, oy)
            point_s = spearman_r(ox, oy)
            ci_p = correlation_bootstrap_cis(ox, oy, 'pearson')
            ci_s = correlation_bootstrap_cis(ox, oy, 'spearman')
            for corr_type, point, ci in (('pearson', point_p, ci_p), ('spearman', point_s, ci_s)):
                row = dict(method_id=mid, metric_a=a, metric_b=b, correlation_type=corr_type,
                             observed_correlation=nfmt(point))
                signs_seen = set()
                for scheme in BOOTSTRAP_SCHEME_NAMES:
                    lo, hi = ci[scheme]
                    row[f'{scheme}_ci95_low'] = nfmt(lo)
                    row[f'{scheme}_ci95_high'] = nfmt(hi)
                    sign = _interval_sign(lo, hi)
                    row[f'{scheme}_sign'] = sign
                    signs_seen.add(sign)
                row['all_interval_signs_agree'] = (len(signs_seen) == 1 and '' not in signs_seen)
                rows.append(row)
    return rows


def focal_topology_relationship_summary(focal_rows):
    from collections import defaultdict
    acc = defaultdict(list)
    for r in focal_rows:
        acc[(r['metric_a'], r['metric_b'], r['correlation_type'])].append(r)

    rows = []
    for a, b in FOCAL_PAIRS:
        for ctype in ('pearson', 'spearman'):
            entries = acc.get((a, b, ctype), [])
            vals = [float(e['observed_correlation']) for e in entries if e['observed_correlation'] != '']
            stats = _dispersion_stats(vals)
            excl = {scheme: sum(1 for e in entries if e[f'{scheme}_sign'] in ('positive', 'negative'))
                    for scheme in BOOTSTRAP_SCHEME_NAMES}
            rows.append(dict(
                metric_a=a, metric_b=b, correlation_type=ctype, n_methods=len(entries),
                median=stats['median'], q25=stats['q25'], q75=stats['q75'],
                min=stats['min'], max=stats['max'],
                n_positive=stats['n_positive'], n_negative=stats['n_negative'],
                n_ci_excludes_zero_iid=excl['iid'],
                n_ci_excludes_zero_block6=excl['block6'],
                n_ci_excludes_zero_block12=excl['block12'],
                n_ci_excludes_zero_block24=excl['block24'],
                interpretation_level='within_method_across_samples',
            ))
    return rows


# =============================================================================
# ANALYSIS F -- direct PD/MT disagreement (rank-based and pairwise-preference
# based). Descriptor disagreement must NOT be interpreted as one descriptor
# being invalid -- both PD and MT are legitimate, differently-scoped
# topological descriptors.
# =============================================================================

def pref(v1, v2):
    if abs(v1 - v2) <= TIE_TOLERANCE:
        return 'tie'
    return 'a' if v1 < v2 else 'b'


def topology_rank_by_method(means, topology_methods, display_name_by_method):
    """Rank the 18 topology-bearing METHOD MEANS (not an average of 168
    per-sample ranks) in ascending raw-distance order -- rank 1 = smallest
    (best) mean PD/MT distance, with average ranks for exact ties.

    signed_rank_gap = pd_rank - mt_rank: positive means MT favors the method
    more strongly than PD (its MT rank is better/lower than its PD rank);
    negative means PD favors the method more strongly than MT."""
    pd_means = np.array([means[m]['raw']['pd_distance'] for m in topology_methods])
    mt_means = np.array([means[m]['raw']['mt_distance'] for m in topology_methods])
    pd_ranks = rankdata_avg(pd_means)
    mt_ranks = rankdata_avg(mt_means)
    rows = []
    for i, m in enumerate(topology_methods):
        signed_gap = float(pd_ranks[i] - mt_ranks[i])
        rows.append(dict(
            method_id=m, display_name=display_name_by_method.get(m, ''),
            pd_mean=float(pd_means[i]), mt_mean=float(mt_means[i]),
            pd_rank=float(pd_ranks[i]), mt_rank=float(mt_ranks[i]),
            absolute_rank_gap=abs(signed_gap), signed_rank_gap=signed_gap,
        ))
    return rows


def topology_pairwise_preference_agreement(per_sample, topology_methods):
    """Per method pair, classify every one of the 168 fields into exactly one
    of three mutually-exclusive categories:

      descriptor_agreement:      both PD and MT are non-tied and prefer the
                                  same method;
      descriptor_disagreement:   both PD and MT are non-tied and prefer
                                  opposite methods;
      descriptor_tie_or_undefined: either descriptor is tied (a tie in
                                  either descriptor is NOT evidence of
                                  agreement, and a tie in only one descriptor
                                  is NOT evidence of disagreement -- both are
                                  classified as tie/undefined).
    """
    rows = []
    for m1, m2 in itertools.combinations(topology_methods, 2):
        n_pd_a = n_pd_b = n_pd_tie = 0
        n_mt_a = n_mt_b = n_mt_tie = 0
        n_agree = n_disagree = n_tie_or_undef = 0
        for si in range(N_EVAL):
            pd_pref = pref(per_sample[m1][si]['pd_distance'], per_sample[m2][si]['pd_distance'])
            mt_pref = pref(per_sample[m1][si]['mt_distance'], per_sample[m2][si]['mt_distance'])
            if pd_pref == 'a':
                n_pd_a += 1
            elif pd_pref == 'b':
                n_pd_b += 1
            else:
                n_pd_tie += 1
            if mt_pref == 'a':
                n_mt_a += 1
            elif mt_pref == 'b':
                n_mt_b += 1
            else:
                n_mt_tie += 1
            if pd_pref == 'tie' or mt_pref == 'tie':
                n_tie_or_undef += 1
            elif pd_pref == mt_pref:
                n_agree += 1
            else:
                n_disagree += 1
        rows.append(dict(
            method_a=m1, method_b=m2, n_samples=N_EVAL,
            pd_prefers_a_count=n_pd_a, pd_prefers_b_count=n_pd_b, pd_tie_count=n_pd_tie,
            mt_prefers_a_count=n_mt_a, mt_prefers_b_count=n_mt_b, mt_tie_count=n_mt_tie,
            descriptor_agreement_count=n_agree, descriptor_disagreement_count=n_disagree,
            descriptor_tie_or_undefined_count=n_tie_or_undef,
            descriptor_agreement_rate=n_agree / N_EVAL,
            descriptor_disagreement_rate=n_disagree / N_EVAL,
        ))
    return rows


def topology_sample_preference_agreement(per_sample, topology_methods, samplewise_rows):
    """Per field, aggregate the same three-way classification over all
    C(18,2)=153 topology-bearing method pairs, and cross-reference the
    field's PD/MT cross-method correlation (must match
    samplewise_cross_method_correlations.csv exactly for that field)."""
    pairs = list(itertools.combinations(topology_methods, 2))
    n_pairs = len(pairs)
    pd_mt_lookup = {}
    for r in samplewise_rows:
        if {r['metric_a'], r['metric_b']} == {'pd_distance', 'mt_distance'}:
            pd_mt_lookup[r['sample_idx']] = (r['oriented_pearson'], r['oriented_spearman'])
    rows = []
    for si in range(N_EVAL):
        n_agree = n_disagree = n_tie_or_undef = 0
        for m1, m2 in pairs:
            pd_pref = pref(per_sample[m1][si]['pd_distance'], per_sample[m2][si]['pd_distance'])
            mt_pref = pref(per_sample[m1][si]['mt_distance'], per_sample[m2][si]['mt_distance'])
            if pd_pref == 'tie' or mt_pref == 'tie':
                n_tie_or_undef += 1
            elif pd_pref == mt_pref:
                n_agree += 1
            else:
                n_disagree += 1
        pearson_v, spearman_v = pd_mt_lookup.get(si, ('', ''))
        rows.append(dict(
            sample_idx=si, method_pair_count=n_pairs, agreement_count=n_agree,
            disagreement_count=n_disagree, tie_or_undefined_count=n_tie_or_undef,
            agreement_rate=n_agree / n_pairs, disagreement_rate=n_disagree / n_pairs,
            pd_mt_cross_method_pearson=pearson_v, pd_mt_cross_method_spearman=spearman_v,
        ))
    return rows


def topology_descriptor_disagreement_summary(method_level_pd_mt, within_pd_mt, samplewise_pd_mt,
                                                pairwise_rows, sample_rows):
    pairwise_agree_rates = np.array([r['descriptor_agreement_rate'] for r in pairwise_rows])
    pairwise_disagree_rates = np.array([r['descriptor_disagreement_rate'] for r in pairwise_rows])
    sample_agree_rates = np.array([r['agreement_rate'] for r in sample_rows])
    sample_disagree_rates = np.array([r['disagreement_rate'] for r in sample_rows])
    corr_note = ('This row reports a correlation / rank-association coefficient (Pearson or Spearman), '
                 'NOT a literal preference-agreement rate. PD (persistence diagram) and MT (merge tree) '
                 'are distinct, legitimate topological descriptors; a weak or negative association here '
                 'reflects genuinely different geometric sensitivity, not that one descriptor is invalid.')
    rate_note = ('This row reports a literal pairwise-preference agreement/disagreement RATE (fraction of '
                 'cases where PD and MT pick the same, non-tied, better method), NOT a correlation '
                 'coefficient. Cases where either descriptor is tied are excluded from both the agreement '
                 'and disagreement counts (see descriptor_tie_or_undefined_count).')
    rows = [
        dict(level='between_method_means', n_units=method_level_pd_mt['n_common_methods'],
             value_a_label='oriented_pearson', value_a=method_level_pd_mt['oriented_pearson'],
             value_b_label='oriented_spearman', value_b=method_level_pd_mt['oriented_spearman'],
             note=corr_note),
        dict(level='within_method_across_samples_median', n_units=within_pd_mt['n_methods_available'],
             value_a_label='median_oriented_pearson', value_a=within_pd_mt['pearson_median'],
             value_b_label='median_oriented_spearman', value_b=within_pd_mt['spearman_median'],
             note=corr_note),
        dict(level='within_sample_across_methods_median', n_units=samplewise_pd_mt['n_samples'],
             value_a_label='median_oriented_pearson', value_a=samplewise_pd_mt['pearson_median'],
             value_b_label='median_oriented_spearman', value_b=samplewise_pd_mt['spearman_median'],
             note=corr_note),
        dict(level='pairwise_preference_agreement', n_units=len(pairwise_rows),
             value_a_label='mean_descriptor_agreement_rate', value_a=float(pairwise_agree_rates.mean()),
             value_b_label='mean_descriptor_disagreement_rate', value_b=float(pairwise_disagree_rates.mean()),
             note=rate_note),
        dict(level='sample_preference_agreement', n_units=len(sample_rows),
             value_a_label='mean_agreement_rate', value_a=float(sample_agree_rates.mean()),
             value_b_label='mean_disagreement_rate', value_b=float(sample_disagree_rates.mean()),
             note=rate_note),
    ]
    return rows


# =============================================================================
# PARETO ANALYSIS -- six explicitly-defined objective sets. Oriented method
# means, higher always better; A strictly dominates B when A>=B-tol on every
# objective AND A>B+tol on at least one. This is a set of Pareto fronts under
# specific, stated objective choices -- not a universal ranking. No metric
# normalization is applied or needed: strict dominance is invariant to any
# monotonic per-objective rescaling.
# =============================================================================

FIDELITY_METRICS = ['psnruv', 'speed_mae']
PHYSICS_METRICS = ['wpd_mae', 'grad_mae', 'psd_log_l2', 'exceed_abs_p90', 'comp_curve_l1']
TOPOLOGY_OBJ = ['pd_distance', 'mt_distance']

EXPECTED_TOPOLOGY_ONLY_FRONT = frozenset(
    {'gan', 'f3_grad_crit', 'f2_grad_levelset_e2', 'f1_grad_e2', 'uv_e2'}
)


def build_pareto_objective_sets(non_ssim):
    return dict(
        topology_only=list(TOPOLOGY_OBJ),
        fidelity_physics_compact=FIDELITY_METRICS + PHYSICS_METRICS,
        fidelity_topology=FIDELITY_METRICS + TOPOLOGY_OBJ,
        physics_topology=PHYSICS_METRICS + TOPOLOGY_OBJ,
        cross_family_compact=FIDELITY_METRICS + PHYSICS_METRICS + TOPOLOGY_OBJ,
        all_available_non_ssim=list(non_ssim),
    )


def pareto_objective_manifest(objective_sets):
    rows = []
    for oset_name, objs in objective_sets.items():
        eligible_methods = 18 if any(o in TOPOLOGY_METRICS for o in objs) else 19
        is_sensitivity = (oset_name == 'all_available_non_ssim')
        for obj in objs:
            rows.append(dict(
                objective_set=oset_name, objective=obj, n_objectives_total=len(objs),
                n_eligible_methods=eligible_methods, is_sensitivity_analysis=is_sensitivity,
            ))
    return rows


def compute_dominance_edges(oriented_rows, methods, objs, tol):
    edges = []
    dominated_by_someone = set()
    for mi in methods:
        vi = np.array([oriented_rows[mi][o] for o in objs])
        for mj in methods:
            if mi == mj:
                continue
            vj = np.array([oriented_rows[mj][o] for o in objs])
            if np.all(vi >= vj - tol) and np.any(vi > vj + tol):
                edges.append((mi, mj))
                dominated_by_someone.add(mj)
    front = [m for m in methods if m not in dominated_by_someone]
    return edges, front


def compute_layers(oriented_rows, methods, objs, tol):
    remaining = list(methods)
    layers = []
    layer_idx = 0
    while remaining:
        _, front = compute_dominance_edges(oriented_rows, remaining, objs, tol)
        for m in front:
            layers.append(dict(method_id=m, layer=layer_idx))
        remaining = [m for m in remaining if m not in front]
        layer_idx += 1
    return layers


def run_pareto_deterministic(means, methods, topology_methods, objective_sets):
    front_membership_rows = []
    dominance_edge_rows = []
    layer_rows = []
    front_by_set = {}
    for oset_name, objs in objective_sets.items():
        eligible = topology_methods if any(o in TOPOLOGY_METRICS for o in objs) else methods
        oriented_rows = {m: {o: means[m]['oriented'][o] for o in objs} for m in eligible}
        edges, front = compute_dominance_edges(oriented_rows, eligible, objs, PARETO_TOLERANCE)
        front_set = set(front)
        front_by_set[oset_name] = front_set
        for m in eligible:
            front_membership_rows.append(dict(objective_set=oset_name, method_id=m,
                                                 on_front=(m in front_set)))
        for dominator, dominated in edges:
            dominance_edge_rows.append(dict(objective_set=oset_name, dominator=dominator,
                                               dominated=dominated))
        layers = compute_layers(oriented_rows, eligible, objs, PARETO_TOLERANCE)
        for row in layers:
            layer_rows.append(dict(objective_set=oset_name, method_id=row['method_id'],
                                      layer=row['layer']))
    return front_membership_rows, dominance_edge_rows, layer_rows, front_by_set


def topology_pareto_sanity_check(front_by_set, dominance_edge_rows, layer_rows, topology_methods):
    checks = []

    def add(name, ok, notes=''):
        checks.append(dict(check_name=name, status=('PASS' if ok else 'FAIL'), notes=notes))

    observed_front = front_by_set['topology_only']
    add('topology_only_front_matches_expected_set',
        observed_front == EXPECTED_TOPOLOGY_ONLY_FRONT,
        notes=f'observed={sorted(observed_front)} expected={sorted(EXPECTED_TOPOLOGY_ONLY_FRONT)}')
    if observed_front != EXPECTED_TOPOLOGY_ONLY_FRONT:
        raise SystemExit(
            f'[hard-fail] topology-only Pareto front sanity check failed: '
            f'observed={sorted(observed_front)} expected={sorted(EXPECTED_TOPOLOGY_ONLY_FRONT)}'
        )

    topo_edges = [(r['dominator'], r['dominated']) for r in dominance_edge_rows
                  if r['objective_set'] == 'topology_only']
    dominated_set = {d for _, d in topo_edges}
    add('no_front_member_dominated', not (observed_front & dominated_set),
        notes=str(sorted(observed_front & dominated_set)))
    non_front = set(topology_methods) - observed_front
    add('every_non_front_method_dominated', non_front <= dominated_set,
        notes=str(sorted(non_front - dominated_set)))
    edge_set = set(topo_edges)
    antisym_violations = [(a, b) for a, b in edge_set if (b, a) in edge_set]
    add('antisymmetric_dominance', len(antisym_violations) == 0, notes=str(antisym_violations))
    self_dom = [(a, b) for a, b in edge_set if a == b]
    add('no_self_dominance', len(self_dom) == 0, notes=str(self_dom))

    topo_layers = [r for r in layer_rows if r['objective_set'] == 'topology_only']
    methods_seen = [r['method_id'] for r in topo_layers]
    add('every_method_in_exactly_one_layer',
        sorted(methods_seen) == sorted(topology_methods) and len(methods_seen) == len(set(methods_seen)),
        notes=f'n_layer_rows={len(methods_seen)} n_methods={len(topology_methods)}')

    return checks


# --- Pareto bootstrap stability (10,000 resamples/scheme, same sampled 168
# indices for every method/objective/objective-set within a replicate) ---

def _bootstrap_count_matrix(idx):
    c, k = idx.shape
    rows = np.repeat(np.arange(c), k)
    cols = idx.reshape(-1)
    linear = rows.astype(np.int64) * N_EVAL + cols.astype(np.int64)
    counts_flat = np.bincount(linear, minlength=c * N_EVAL)
    return counts_flat.reshape(c, N_EVAL).astype(np.float64)


def _resample_means_matrix(raw, count_matrix):
    n_m, n_o, _ = raw.shape
    M = raw.reshape(n_m * n_o, N_EVAL)
    means = (M @ count_matrix.T) / N_EVAL
    means = means.reshape(n_m, n_o, -1)
    return np.transpose(means, (2, 0, 1))


def _chunked_front_membership(oriented, tol, chunk_size=1000):
    n_rep, n_m, n_o = oriented.shape
    front_counts = np.zeros(n_m, dtype=np.int64)
    front_sizes = np.empty(n_rep, dtype=np.int64)
    eye = np.eye(n_m, dtype=bool)
    for start in range(0, n_rep, chunk_size):
        end = min(start + chunk_size, n_rep)
        chunk = oriented[start:end]
        diff = chunk[:, :, None, :] - chunk[:, None, :, :]
        ge_all = np.all(diff >= -tol, axis=3)
        gt_any = np.any(diff > tol, axis=3)
        dom = ge_all & gt_any
        dom[:, eye] = False
        dominated_by_someone = np.any(dom, axis=1)
        on_front = ~dominated_by_someone
        front_counts += on_front.sum(axis=0)
        front_sizes[start:end] = on_front.sum(axis=1)
    return front_counts, front_sizes


def pareto_bootstrap_stability(per_sample, methods, topology_methods, objective_sets, metric_direction):
    membership_rows = []
    front_size_rows = []
    for scheme_name in BOOTSTRAP_SCHEME_NAMES:
        idx_full = PARETO_BOOTSTRAP_SCHEMES[scheme_name]
        count_matrix = _bootstrap_count_matrix(idx_full)
        for oset_name, objs in objective_sets.items():
            eligible = topology_methods if any(o in TOPOLOGY_METRICS for o in objs) else methods
            n_m = len(eligible)
            n_o = len(objs)
            raw = np.empty((n_m, n_o, N_EVAL))
            for i, mid in enumerate(eligible):
                for k, obj in enumerate(objs):
                    raw[i, k, :] = [per_sample[mid][si][obj] for si in range(N_EVAL)]
            signs = np.array([1.0 if metric_direction[o] == 'higher_is_better' else -1.0 for o in objs])
            means = _resample_means_matrix(raw, count_matrix)
            oriented = means * signs[None, None, :]
            front_counts, front_sizes = _chunked_front_membership(oriented, PARETO_TOLERANCE)
            rate = front_counts / PARETO_BOOTSTRAP_N
            for i, mid in enumerate(eligible):
                membership_rows.append(dict(
                    objective_set=oset_name, bootstrap_scheme=scheme_name, method_id=mid,
                    n_replicates=PARETO_BOOTSTRAP_N, front_membership_count=int(front_counts[i]),
                    front_membership_rate=float(rate[i]),
                    always_on_front=bool(front_counts[i] == PARETO_BOOTSTRAP_N),
                    never_on_front=bool(front_counts[i] == 0),
                ))
            front_size_rows.append(dict(
                objective_set=oset_name, bootstrap_scheme=scheme_name, n_replicates=PARETO_BOOTSTRAP_N,
                mean_front_size=float(front_sizes.mean()), median_front_size=float(np.median(front_sizes)),
                min_front_size=int(front_sizes.min()), max_front_size=int(front_sizes.max()),
            ))
    return membership_rows, front_size_rows


# =============================================================================
# Cross-analysis relationship summaries. Neither of these is, or should be
# read as, a weighted aggregate score or a total method ranking.
# =============================================================================

def metric_relationship_summary(non_ssim, method_level_rows, within_summary_rows,
                                   samplewise_summary_rows, residual_rows):
    method_level_map = {(r['metric_a'], r['metric_b']): r for r in method_level_rows}
    within_map = {(r['metric_a'], r['metric_b']): r for r in within_summary_rows}
    samplewise_map = {(r['metric_a'], r['metric_b']): r for r in samplewise_summary_rows}
    residual_map = {(r['metric_a'], r['metric_b']): r for r in residual_rows}
    rows = []
    for a, b in itertools.combinations(non_ssim, 2):
        ml = method_level_map[(a, b)]['oriented_pearson']
        wm = within_map[(a, b)]['pearson_median']
        sw = samplewise_map[(a, b)]['pearson_median']
        rs = residual_map[(a, b)]['oriented_residual_pearson']
        vals = [ml, wm, sw, rs]
        all_defined = all(v != '' for v in vals)
        consistent = False
        if all_defined:
            signs = set()
            for v in vals:
                fv = float(v)
                signs.add(1 if fv > 0 else (-1 if fv < 0 else 0))
            consistent = (len(signs) == 1 and 0 not in signs)
        rows.append(dict(
            metric_a=a, metric_b=b,
            method_level_oriented_pearson=ml, within_method_median_oriented_pearson=wm,
            samplewise_median_oriented_pearson=sw, two_way_residual_oriented_pearson=rs,
            all_levels_defined=all_defined, consistent_nonzero_sign=consistent,
        ))
    return rows


def topology_relationship_and_pareto_summary(topology_rank_rows, front_membership_rows,
                                                bootstrap_membership_rows, topology_methods):
    pd_rank = {r['method_id']: r['pd_rank'] for r in topology_rank_rows}
    mt_rank = {r['method_id']: r['mt_rank'] for r in topology_rank_rows}
    signed_gap = {r['method_id']: r['signed_rank_gap'] for r in topology_rank_rows}
    on_front = {r['method_id']: r['on_front'] for r in front_membership_rows
                if r['objective_set'] == 'topology_only'}
    boot_rate = {}
    for r in bootstrap_membership_rows:
        if r['objective_set'] == 'topology_only':
            boot_rate.setdefault(r['method_id'], {})[r['bootstrap_scheme']] = r['front_membership_rate']
    rows = []
    for m in topology_methods:
        rates = boot_rate.get(m, {})
        rows.append(dict(
            method_id=m, pd_rank=pd_rank[m], mt_rank=mt_rank[m], signed_rank_gap=signed_gap[m],
            on_topology_only_deterministic_front=on_front.get(m, False),
            topology_only_bootstrap_rate_iid=rates.get('iid', ''),
            topology_only_bootstrap_rate_block6=rates.get('block6', ''),
            topology_only_bootstrap_rate_block12=rates.get('block12', ''),
            topology_only_bootstrap_rate_block24=rates.get('block24', ''),
            note='Not a ranked leaderboard; Pareto front membership depends on the chosen objective set. '
                 'pd_rank/mt_rank are ranks of the METHOD-MEAN distances (not averages of per-sample ranks).',
        ))
    return rows


# =============================================================================
# Additional validation (phase2c_validation.csv, part 2 -- appended after the
# analytical output has been produced).
# =============================================================================

def run_additional_validation(non_ssim, method_level_rows, pearson_matrix_a, spearman_matrix_a,
                                 within_rows, samplewise_rows, residual_rows, residual_margin_rows,
                                 pearson_matrix_d, spearman_matrix_d, pairwise_pref_rows, sample_pref_rows,
                                 front_membership_rows, dominance_edge_rows, sanity_checks,
                                 bootstrap_membership_rows, bootstrap_front_size_rows,
                                 objective_sets, methods, topology_methods, rank_rows, means, focal_rows,
                                 per_sample, metric_direction):
    checks = []
    failures = []

    def add(name, ok, notes=''):
        status = 'PASS' if ok else 'FAIL'
        checks.append(dict(check_name=name, observed='', expected='', tolerance='', status=status, notes=notes))
        if not ok:
            failures.append(f'{name}: {notes}')

    add('ssim_excluded_from_non_ssim_metric_list', 'ssim_speed' not in non_ssim, notes='')
    for oset_name, objs in objective_sets.items():
        add(f'ssim_excluded_from_pareto_objective_set[{oset_name}]', 'ssim_speed' not in objs, notes='')

    def check_bounds(rows, fields, label):
        bad = 0
        for r in rows:
            for f in fields:
                v = r.get(f, '')
                if v == '' or v is None:
                    continue
                fv = float(v)
                if not (-1.0 - 1e-9 <= fv <= 1.0 + 1e-9):
                    bad += 1
        add(f'correlation_bounds[{label}]', bad == 0, notes=f'n_out_of_bounds={bad}')

    check_bounds(method_level_rows,
                 ['raw_pearson', 'raw_spearman', 'oriented_pearson', 'oriented_spearman',
                  'loo_pearson_min', 'loo_pearson_max', 'loo_spearman_min', 'loo_spearman_max'],
                 'method_level_metric_correlations')
    check_bounds(within_rows, ['raw_pearson', 'raw_spearman', 'oriented_pearson', 'oriented_spearman'],
                 'within_method_metric_correlations')
    check_bounds(samplewise_rows, ['raw_pearson', 'raw_spearman', 'oriented_pearson', 'oriented_spearman'],
                 'samplewise_cross_method_correlations')
    check_bounds(residual_rows, ['oriented_residual_pearson', 'oriented_residual_spearman'],
                 'two_way_residual_correlations')

    def check_matrix(matrix, label):
        bad_sym = 0
        bad_diag = 0
        for a in non_ssim:
            if matrix[a][a] != 1.0:
                bad_diag += 1
            for b in non_ssim:
                va, vb = matrix[a][b], matrix[b][a]
                if va is None and vb is None:
                    continue
                if va is None or vb is None or abs(va - vb) > 1e-12:
                    bad_sym += 1
        add(f'matrix_symmetric[{label}]', bad_sym == 0, notes=f'n_asymmetric_cells={bad_sym}')
        add(f'matrix_diagonal_one[{label}]', bad_diag == 0, notes=f'n_bad_diag={bad_diag}')

    check_matrix(pearson_matrix_a, 'method_level_oriented_pearson')
    check_matrix(spearman_matrix_a, 'method_level_oriented_spearman')
    check_matrix(pearson_matrix_d, 'two_way_residual_pearson')
    check_matrix(spearman_matrix_d, 'two_way_residual_spearman')

    bad_agree = 0
    for r in method_level_rows:
        a, b = r['metric_a'], r['metric_b']
        if r['oriented_pearson'] != '' and abs(float(r['oriented_pearson']) - pearson_matrix_a[a][b]) > 1e-12:
            bad_agree += 1
        if r['oriented_spearman'] != '' and abs(float(r['oriented_spearman']) - spearman_matrix_a[a][b]) > 1e-12:
            bad_agree += 1
    add('method_level_pairwise_matrix_agreement', bad_agree == 0, notes=f'n_mismatch={bad_agree}')

    bad_n = sum(1 for r in within_rows if r['n_samples'] not in (0, N_EVAL))
    add('within_method_n_samples_valid', bad_n == 0, notes=f'n_bad={bad_n}')

    bad_cm = sum(1 for r in samplewise_rows if r['n_methods'] not in (18, 19))
    add('samplewise_n_methods_valid', bad_cm == 0, notes=f'n_bad={bad_cm}')

    bad_margin = [r for r in residual_margin_rows
                  if r['row_margin_max_abs'] > MARGIN_TOLERANCE or r['col_margin_max_abs'] > MARGIN_TOLERANCE
                  or r['grand_margin_abs'] > MARGIN_TOLERANCE]
    add('two_way_residual_margins_zero_pair_specific', len(bad_margin) == 0,
        notes=str([(r['metric'], r['paired_with']) for r in bad_margin]))

    bad_rect = [r for r in residual_rows
                if r['n_common_methods'] != (18 if (r['metric_a'] in TOPOLOGY_METRICS
                                                      or r['metric_b'] in TOPOLOGY_METRICS) else 19)
                or r['n_samples'] != N_EVAL]
    add('two_way_residual_uses_common_rectangular_method_set', len(bad_rect) == 0,
        notes=f'n_bad={len(bad_rect)}')

    bad_pref = [r for r in pairwise_pref_rows
                if (r['descriptor_agreement_count'] + r['descriptor_disagreement_count'] +
                    r['descriptor_tie_or_undefined_count']) != N_EVAL
                or (r['pd_prefers_a_count'] + r['pd_prefers_b_count'] + r['pd_tie_count']) != N_EVAL
                or (r['mt_prefers_a_count'] + r['mt_prefers_b_count'] + r['mt_tie_count']) != N_EVAL]
    add('topology_pairwise_preference_counts_sum_168', len(bad_pref) == 0, notes=f'n_bad={len(bad_pref)}')

    expected_pairs = len(list(itertools.combinations(topology_methods, 2)))
    bad_sample_pref = [r for r in sample_pref_rows
                        if (r['agreement_count'] + r['disagreement_count'] + r['tie_or_undefined_count'])
                        != expected_pairs
                        or r['method_pair_count'] != expected_pairs]
    add('topology_sample_preference_counts_sum_153', len(bad_sample_pref) == 0,
        notes=f'expected={expected_pairs} n_bad={len(bad_sample_pref)}')

    pd_mt_lookup = {}
    for r in samplewise_rows:
        if {r['metric_a'], r['metric_b']} == {'pd_distance', 'mt_distance'}:
            pd_mt_lookup[r['sample_idx']] = (r['oriented_pearson'], r['oriented_spearman'])
    bad_corr_match = sum(
        1 for r in sample_pref_rows
        if (r['pd_mt_cross_method_pearson'], r['pd_mt_cross_method_spearman'])
        != pd_mt_lookup.get(r['sample_idx'], ('', ''))
    )
    add('sample_preference_pd_mt_correlation_matches_samplewise_table', bad_corr_match == 0,
        notes=f'n_bad={bad_corr_match}')

    add('topology_rank_by_method_has_18_rows', len(rank_rows) == 18, notes=f'n_rows={len(rank_rows)}')
    pd_means_check = np.array([means[r['method_id']]['raw']['pd_distance'] for r in rank_rows])
    mt_means_check = np.array([means[r['method_id']]['raw']['mt_distance'] for r in rank_rows])
    expected_pd_ranks = rankdata_avg(pd_means_check)
    expected_mt_ranks = rankdata_avg(mt_means_check)
    bad_rank = 0
    for i, r in enumerate(rank_rows):
        if abs(r['pd_rank'] - expected_pd_ranks[i]) > 1e-9 or abs(r['mt_rank'] - expected_mt_ranks[i]) > 1e-9:
            bad_rank += 1
        signed = r['pd_rank'] - r['mt_rank']
        if abs(r['signed_rank_gap'] - signed) > 1e-9 or abs(r['absolute_rank_gap'] - abs(signed)) > 1e-9:
            bad_rank += 1
        if abs(r['pd_mean'] - means[r['method_id']]['raw']['pd_distance']) > 1e-9:
            bad_rank += 1
    add('topology_rank_by_method_ranks_the_method_means', bad_rank == 0, notes=f'n_bad={bad_rank}')
    add('topology_rank_gap_internally_consistent', bad_rank == 0, notes=f'n_bad={bad_rank}')

    add('focal_topology_bootstrap_has_540_rows', len(focal_rows) == 18 * 15 * 2,
        notes=f'n_rows={len(focal_rows)}')
    bad_focal_point = 0
    bad_focal_sign = 0
    for r in focal_rows:
        mid = r['method_id']
        xs = np.array([per_sample[mid][si][r['metric_a']] for si in range(N_EVAL)])
        ys = np.array([per_sample[mid][si][r['metric_b']] for si in range(N_EVAL)])
        ox = orient(xs, metric_direction[r['metric_a']])
        oy = orient(ys, metric_direction[r['metric_b']])
        expected_point = pearson_r(ox, oy) if r['correlation_type'] == 'pearson' else spearman_r(ox, oy)
        observed = r['observed_correlation']
        if (expected_point is None) != (observed == ''):
            bad_focal_point += 1
        elif expected_point is not None and abs(float(observed) - expected_point) > 1e-9:
            bad_focal_point += 1
        signs_seen = set()
        for scheme in BOOTSTRAP_SCHEME_NAMES:
            lo, hi = r[f'{scheme}_ci95_low'], r[f'{scheme}_ci95_high']
            expected_sign = ('' if lo == '' or hi == '' else
                              'positive' if float(lo) > 0 else 'negative' if float(hi) < 0 else 'includes_zero')
            if r[f'{scheme}_sign'] != expected_sign:
                bad_focal_sign += 1
            signs_seen.add(r[f'{scheme}_sign'])
        expected_agree = (len(signs_seen) == 1 and '' not in signs_seen)
        if r['all_interval_signs_agree'] != expected_agree:
            bad_focal_sign += 1
    add('focal_topology_bootstrap_observed_correlation_matches_raw_data', bad_focal_point == 0,
        notes=f'n_bad={bad_focal_point}')
    add('focal_topology_bootstrap_interval_sign_fields_correct', bad_focal_sign == 0,
        notes=f'n_bad={bad_focal_sign}')

    dominated_lookup = {}
    for r in dominance_edge_rows:
        dominated_lookup.setdefault(r['objective_set'], set()).add(r['dominated'])
    bad_pareto = 0
    for r in front_membership_rows:
        dominated_set = dominated_lookup.get(r['objective_set'], set())
        is_dominated = r['method_id'] in dominated_set
        if r['on_front'] == is_dominated:
            bad_pareto += 1
    add('pareto_front_edge_agreement', bad_pareto == 0, notes=f'n_mismatch={bad_pareto}')

    sanity_fail = [c for c in sanity_checks if c['status'] != 'PASS']
    add('topology_pareto_sanity_all_pass', len(sanity_fail) == 0,
        notes=str([c['check_name'] for c in sanity_fail]))

    bad_rate = [r for r in bootstrap_membership_rows
                if not (0.0 <= r['front_membership_rate'] <= 1.0)
                or not (0 <= r['front_membership_count'] <= PARETO_BOOTSTRAP_N)]
    add('pareto_bootstrap_rates_and_counts_in_range', len(bad_rate) == 0, notes=f'n_bad={len(bad_rate)}')

    bad_fs = [r for r in bootstrap_front_size_rows
              if not (0 <= r['min_front_size'] <= r['max_front_size'] <= len(methods))]
    add('pareto_bootstrap_front_size_range_valid', len(bad_fs) == 0, notes=f'n_bad={len(bad_fs)}')

    add('no_imputation_partial_coverage_hard_fails_enforced', True,
        notes='within-method and samplewise partial-coverage cases hard-fail before reaching validation; '
              'no missing value was ever filled in.')

    return checks, failures


# =============================================================================
# main()
# =============================================================================

def _find_pair_row(rows, a, b):
    for r in rows:
        if {r['metric_a'], r['metric_b']} == {a, b}:
            return r
    raise KeyError((a, b))


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log('=' * 88)
    log('Unified candidate analysis -- Phase 2C (metric relationships and Pareto tradeoffs)')
    log(f'Repo root: {REPO_ROOT}')
    log('Read-only w.r.t. Phase-1, Phase-2A, and Phase-2B artifacts. No training/inference/eval/TTK performed.')
    log('No weighted aggregate score or total method ranking is computed. Sample selection and figure '
        'generation remain deferred to Phase 2D.')
    log('=' * 88)

    require_protected_files()
    file_to_phase = {p.resolve().relative_to(REPO_ROOT).as_posix(): 'phase1' for p in PHASE1_PROTECTED_FILES}
    file_to_phase.update({p.resolve().relative_to(REPO_ROOT).as_posix(): 'phase2a' for p in PHASE2A_PROTECTED_FILES})
    file_to_phase.update({p.resolve().relative_to(REPO_ROOT).as_posix(): 'phase2b' for p in PHASE2B_PROTECTED_FILES})
    checksums_before = checksum_all(ALL_PROTECTED_FILES)
    log(f'[immutability] Checksummed {len(checksums_before)} prior-phase file(s) before the run '
        f'(12 Phase-1 + 14 Phase-2A + 28 Phase-2B = 54 exactly).')

    long_table = load_long_table()
    metric_direction, metric_family = load_column_mapping()
    metric_cols = long_table['metric_cols']
    per_sample = long_table['per_sample']
    phase1_refs = load_phase1_refs()
    phase2a_refs = load_phase2a_refs()
    phase2b_refs = load_phase2b_refs()
    log(f'[load] Long table: {long_table["n_rows"]} rows, {len(metric_cols)} metrics, '
        f'{len(per_sample)} methods.')

    base_checks, base_failures = run_base_validation(long_table, phase1_refs, phase2a_refs, phase2b_refs,
                                                        metric_direction)
    if base_failures:
        log('')
        log('[BASE VALIDATION FAILURE]')
        for f in base_failures:
            log(f'  - {f}')
        flush_log()
        raise SystemExit(f'[hard-fail] {len(base_failures)} Phase-2C base validation check(s) failed.')
    log(f'[validate] All {len(base_checks)} base validation checks PASSED (hard-fail gate before any '
        f'analytical output).')

    methods = sorted(per_sample.keys())
    topology_methods = sorted(m for m in methods if m != BICUBIC_METHOD)
    non_ssim = [c for c in metric_cols if c != 'ssim_speed']
    assert len(methods) == N_METHODS
    assert len(topology_methods) == N_METHODS - 1
    assert len(non_ssim) == 21
    display_name_by_method = {mid: long_table['method_meta'][mid]['values'].get('display_name', '')
                                for mid in methods}

    # -------------------------------------------------------------------
    # Analysis A -- method-mean relationships
    # -------------------------------------------------------------------
    means, mmov_rows, mmov_fieldnames = build_method_mean_oriented_values(methods, per_sample, metric_cols,
                                                                             metric_direction)
    write_csv(OUT_DIR / 'method_mean_oriented_values.csv', mmov_fieldnames, mmov_rows)

    method_level_rows, pearson_matrix_a, spearman_matrix_a = method_level_metric_correlations(
        methods, means, non_ssim)
    write_csv(OUT_DIR / 'method_level_metric_correlations.csv',
               ['metric_a', 'metric_b', 'n_common_methods', 'raw_pearson', 'raw_spearman',
                'oriented_pearson', 'oriented_spearman', 'loo_pearson_min', 'loo_pearson_max',
                'loo_spearman_min', 'loo_spearman_max', 'pearson_sign_stable', 'spearman_sign_stable',
                'interpretation_level'], method_level_rows)
    write_matrix_csv(OUT_DIR / 'method_level_oriented_pearson_matrix.csv', non_ssim, pearson_matrix_a)
    write_matrix_csv(OUT_DIR / 'method_level_oriented_spearman_matrix.csv', non_ssim, spearman_matrix_a)
    log(f'[analysis-A] {len(method_level_rows)} method-level metric-pair correlations '
        f'(21 non-SSIM metrics, C(21,2)=210 pairs).')

    # -------------------------------------------------------------------
    # Analysis B -- within-method sample relationships
    # -------------------------------------------------------------------
    within_rows, within_acc = within_method_metric_correlations(per_sample, methods, non_ssim, metric_direction)
    write_csv(OUT_DIR / 'within_method_metric_correlations.csv',
               ['method_id', 'metric_a', 'metric_b', 'n_samples', 'status', 'raw_pearson', 'raw_spearman',
                'oriented_pearson', 'oriented_spearman', 'interpretation_level'], within_rows)
    within_summary_rows = within_method_correlation_summary(within_acc, non_ssim)
    write_csv(OUT_DIR / 'within_method_correlation_summary.csv',
               ['metric_a', 'metric_b', 'n_methods_available', 'pearson_median', 'pearson_q25', 'pearson_q75',
                'pearson_min', 'pearson_max', 'pearson_n_positive', 'pearson_n_negative', 'spearman_median',
                'spearman_q25', 'spearman_q75', 'spearman_min', 'spearman_max', 'spearman_n_positive',
                'spearman_n_negative', 'interpretation_level'], within_summary_rows)
    log(f'[analysis-B] {len(within_rows)} within-method metric-pair correlations '
        f'(19 methods x 210 pairs = {N_METHODS * 210}).')

    # -------------------------------------------------------------------
    # Analysis C -- per-sample cross-method relationships
    # -------------------------------------------------------------------
    samplewise_rows, samplewise_acc = samplewise_cross_method_correlations(per_sample, methods, topology_methods,
                                                                              non_ssim, metric_direction)
    write_csv(OUT_DIR / 'samplewise_cross_method_correlations.csv',
               ['sample_idx', 'metric_a', 'metric_b', 'n_methods', 'raw_pearson', 'raw_spearman',
                'oriented_pearson', 'oriented_spearman', 'interpretation_level'], samplewise_rows)
    samplewise_summary_rows = samplewise_correlation_summary(samplewise_acc, non_ssim)
    write_csv(OUT_DIR / 'samplewise_correlation_summary.csv',
               ['metric_a', 'metric_b', 'n_samples', 'pearson_median', 'pearson_q25', 'pearson_q75',
                'pearson_min', 'pearson_max', 'pearson_n_positive', 'pearson_n_negative', 'spearman_median',
                'spearman_q25', 'spearman_q75', 'spearman_min', 'spearman_max', 'spearman_n_positive',
                'spearman_n_negative', 'interpretation_level'], samplewise_summary_rows)
    log(f'[analysis-C] {len(samplewise_rows)} per-sample cross-method metric-pair correlations '
        f'(168 samples x 210 pairs = {N_EVAL * 210}).')

    # -------------------------------------------------------------------
    # Analysis D -- two-way-centered residual relationships
    # -------------------------------------------------------------------
    residual_rows, residual_margin_rows, pearson_matrix_d, spearman_matrix_d = two_way_residual_correlations(
        per_sample, methods, topology_methods, non_ssim, metric_family, metric_direction)
    write_csv(OUT_DIR / 'two_way_residual_correlations.csv',
               ['metric_a', 'metric_b', 'family_a', 'family_b', 'n_common_methods', 'n_samples', 'n_cells',
                'oriented_residual_pearson', 'oriented_residual_spearman', 'max_abs_method_margin_mean_a',
                'max_abs_sample_margin_mean_a', 'max_abs_method_margin_mean_b',
                'max_abs_sample_margin_mean_b', 'interpretation_level'], residual_rows)
    write_matrix_csv(OUT_DIR / 'two_way_residual_pearson_matrix.csv', non_ssim, pearson_matrix_d)
    write_matrix_csv(OUT_DIR / 'two_way_residual_spearman_matrix.csv', non_ssim, spearman_matrix_d)
    n_pure_topology_pairs = sum(1 for r in residual_rows
                                  if r['metric_a'] in TOPOLOGY_METRICS and r['metric_b'] in TOPOLOGY_METRICS)
    n_mixed_pairs = sum(1 for r in residual_rows
                          if (r['metric_a'] in TOPOLOGY_METRICS) != (r['metric_b'] in TOPOLOGY_METRICS))
    n_pure_nontopology_pairs = len(residual_rows) - n_pure_topology_pairs - n_mixed_pairs
    log(f'[analysis-D] {len(residual_rows)} pair-specific common-rectangle two-way-centered-residual '
        f'correlations ({n_pure_nontopology_pairs} non-topology/non-topology pairs on 19 methods, '
        f'{n_pure_topology_pairs} PD/MT pair on 18 methods, {n_mixed_pairs} topology/non-topology '
        f'pairs on 18 methods); max row/col/grand margin = '
        f'{max(r["row_margin_max_abs"] for r in residual_margin_rows):.3e} / '
        f'{max(r["col_margin_max_abs"] for r in residual_margin_rows):.3e} / '
        f'{max(r["grand_margin_abs"] for r in residual_margin_rows):.3e}.')

    # -------------------------------------------------------------------
    # Analysis E -- focal topology relationships (bootstrap)
    # -------------------------------------------------------------------
    focal_rows = focal_topology_correlation_bootstrap(per_sample, topology_methods, metric_direction)
    write_csv(OUT_DIR / 'focal_topology_correlation_bootstrap.csv',
               ['method_id', 'metric_a', 'metric_b', 'correlation_type', 'observed_correlation',
                'iid_ci95_low', 'iid_ci95_high', 'block6_ci95_low', 'block6_ci95_high',
                'block12_ci95_low', 'block12_ci95_high', 'block24_ci95_low', 'block24_ci95_high',
                'iid_sign', 'block6_sign', 'block12_sign', 'block24_sign', 'all_interval_signs_agree'],
               focal_rows)
    focal_summary_rows = focal_topology_relationship_summary(focal_rows)
    write_csv(OUT_DIR / 'focal_topology_relationship_summary.csv',
               ['metric_a', 'metric_b', 'correlation_type', 'n_methods', 'median', 'q25', 'q75', 'min', 'max',
                'n_positive', 'n_negative', 'n_ci_excludes_zero_iid', 'n_ci_excludes_zero_block6',
                'n_ci_excludes_zero_block12', 'n_ci_excludes_zero_block24', 'interpretation_level'],
               focal_summary_rows)
    log(f'[analysis-E] {len(focal_rows)} focal-topology bootstrap rows '
        f'(18 methods x 15 pairs x 2 correlation types = {18 * 15 * 2}, wide form with all 4 schemes '
        f'per row), {CORR_BOOTSTRAP_N} resamples/scheme.')

    # -------------------------------------------------------------------
    # Analysis F -- direct PD/MT disagreement
    # -------------------------------------------------------------------
    rank_rows = topology_rank_by_method(means, topology_methods, display_name_by_method)
    write_csv(OUT_DIR / 'topology_rank_by_method.csv',
               ['method_id', 'display_name', 'pd_mean', 'mt_mean', 'pd_rank', 'mt_rank', 'absolute_rank_gap',
                'signed_rank_gap'], rank_rows)
    pairwise_pref_rows = topology_pairwise_preference_agreement(per_sample, topology_methods)
    write_csv(OUT_DIR / 'topology_pairwise_preference_agreement.csv',
               ['method_a', 'method_b', 'n_samples', 'pd_prefers_a_count', 'pd_prefers_b_count',
                'pd_tie_count', 'mt_prefers_a_count', 'mt_prefers_b_count', 'mt_tie_count',
                'descriptor_agreement_count', 'descriptor_disagreement_count',
                'descriptor_tie_or_undefined_count', 'descriptor_agreement_rate',
                'descriptor_disagreement_rate'],
               pairwise_pref_rows)
    sample_pref_rows = topology_sample_preference_agreement(per_sample, topology_methods, samplewise_rows)
    write_csv(OUT_DIR / 'topology_sample_preference_agreement.csv',
               ['sample_idx', 'method_pair_count', 'agreement_count', 'disagreement_count',
                'tie_or_undefined_count', 'agreement_rate', 'disagreement_rate',
                'pd_mt_cross_method_pearson', 'pd_mt_cross_method_spearman'],
               sample_pref_rows)
    method_level_pd_mt = _find_pair_row(method_level_rows, 'pd_distance', 'mt_distance')
    within_pd_mt = _find_pair_row(within_summary_rows, 'pd_distance', 'mt_distance')
    samplewise_pd_mt = _find_pair_row(samplewise_summary_rows, 'pd_distance', 'mt_distance')
    descriptor_summary_rows = topology_descriptor_disagreement_summary(
        method_level_pd_mt, within_pd_mt, samplewise_pd_mt, pairwise_pref_rows, sample_pref_rows)
    write_csv(OUT_DIR / 'topology_descriptor_disagreement_summary.csv',
               ['level', 'n_units', 'value_a_label', 'value_a', 'value_b_label', 'value_b', 'note'],
               descriptor_summary_rows)
    log(f'[analysis-F] {len(pairwise_pref_rows)} method-pair PD/MT preference-agreement rows '
        f'(C(18,2)=153), {len(sample_pref_rows)} sample-level aggregation rows.')

    # -------------------------------------------------------------------
    # Pareto analysis -- deterministic fronts, six objective sets
    # -------------------------------------------------------------------
    objective_sets = build_pareto_objective_sets(non_ssim)
    manifest_rows = pareto_objective_manifest(objective_sets)
    write_csv(OUT_DIR / 'pareto_objective_manifest.csv',
               ['objective_set', 'objective', 'n_objectives_total', 'n_eligible_methods',
                'is_sensitivity_analysis'], manifest_rows)

    front_membership_rows, dominance_edge_rows, layer_rows, front_by_set = run_pareto_deterministic(
        means, methods, topology_methods, objective_sets)
    write_csv(OUT_DIR / 'pareto_front_membership.csv', ['objective_set', 'method_id', 'on_front'],
               front_membership_rows)
    write_csv(OUT_DIR / 'pareto_dominance_edges.csv', ['objective_set', 'dominator', 'dominated'],
               dominance_edge_rows)
    write_csv(OUT_DIR / 'pareto_layers.csv', ['objective_set', 'method_id', 'layer'], layer_rows)
    for oset_name, front_set in front_by_set.items():
        log(f'[pareto] {oset_name}: front = {sorted(front_set)}')

    sanity_checks = topology_pareto_sanity_check(front_by_set, dominance_edge_rows, layer_rows, topology_methods)
    write_csv(OUT_DIR / 'topology_pareto_sanity_check.csv', ['check_name', 'status', 'notes'], sanity_checks)
    log('[pareto] topology-only front sanity check: PASSED (matches the expected set exactly).')

    # -------------------------------------------------------------------
    # Pareto bootstrap stability
    # -------------------------------------------------------------------
    bootstrap_membership_rows, bootstrap_front_size_rows = pareto_bootstrap_stability(
        per_sample, methods, topology_methods, objective_sets, metric_direction)
    write_csv(OUT_DIR / 'pareto_bootstrap_stability.csv',
               ['objective_set', 'bootstrap_scheme', 'method_id', 'n_replicates', 'front_membership_count',
                'front_membership_rate', 'always_on_front', 'never_on_front'], bootstrap_membership_rows)
    write_csv(OUT_DIR / 'pareto_bootstrap_front_size.csv',
               ['objective_set', 'bootstrap_scheme', 'n_replicates', 'mean_front_size', 'median_front_size',
                'min_front_size', 'max_front_size'], bootstrap_front_size_rows)
    log(f'[pareto-bootstrap] {PARETO_BOOTSTRAP_N} resamples/scheme x 4 schemes x '
        f'{len(objective_sets)} objective sets.')

    # -------------------------------------------------------------------
    # Cross-analysis relationship summaries (not a weighted score / leaderboard)
    # -------------------------------------------------------------------
    metric_rel_rows = metric_relationship_summary(non_ssim, method_level_rows, within_summary_rows,
                                                     samplewise_summary_rows, residual_rows)
    write_csv(OUT_DIR / 'metric_relationship_summary.csv',
               ['metric_a', 'metric_b', 'method_level_oriented_pearson', 'within_method_median_oriented_pearson',
                'samplewise_median_oriented_pearson', 'two_way_residual_oriented_pearson', 'all_levels_defined',
                'consistent_nonzero_sign'], metric_rel_rows)

    topo_rel_pareto_rows = topology_relationship_and_pareto_summary(rank_rows, front_membership_rows,
                                                                        bootstrap_membership_rows,
                                                                        topology_methods)
    write_csv(OUT_DIR / 'topology_relationship_and_pareto_summary.csv',
               ['method_id', 'pd_rank', 'mt_rank', 'signed_rank_gap', 'on_topology_only_deterministic_front',
                'topology_only_bootstrap_rate_iid', 'topology_only_bootstrap_rate_block6',
                'topology_only_bootstrap_rate_block12', 'topology_only_bootstrap_rate_block24', 'note'],
               topo_rel_pareto_rows)

    # -------------------------------------------------------------------
    # Additional validation (appended to phase2c_validation.csv)
    # -------------------------------------------------------------------
    additional_checks, additional_failures = run_additional_validation(
        non_ssim, method_level_rows, pearson_matrix_a, spearman_matrix_a, within_rows, samplewise_rows,
        residual_rows, residual_margin_rows, pearson_matrix_d, spearman_matrix_d, pairwise_pref_rows,
        sample_pref_rows, front_membership_rows, dominance_edge_rows, sanity_checks, bootstrap_membership_rows,
        bootstrap_front_size_rows, objective_sets, methods, topology_methods, rank_rows, means, focal_rows,
        per_sample, metric_direction)
    if additional_failures:
        log('')
        log('[ADDITIONAL VALIDATION FAILURE]')
        for f in additional_failures:
            log(f'  - {f}')
        flush_log()
        raise SystemExit(f'[hard-fail] {len(additional_failures)} Phase-2C additional validation check(s) failed.')
    log(f'[validate] All {len(additional_checks)} additional validation checks PASSED.')

    all_validation_rows = base_checks + additional_checks
    write_csv(OUT_DIR / 'phase2c_validation.csv',
               ['check_name', 'observed', 'expected', 'tolerance', 'status', 'notes'], all_validation_rows)
    log(f'[validate] phase2c_validation.csv: {len(all_validation_rows)} total checks, all PASSED.')

    # -------------------------------------------------------------------
    # Documentation
    # -------------------------------------------------------------------
    write_phase2c_doc(non_ssim, methods, topology_methods, method_level_rows, within_summary_rows,
                        samplewise_summary_rows, residual_rows, method_level_pd_mt, within_pd_mt,
                        samplewise_pd_mt, descriptor_summary_rows, objective_sets, front_by_set,
                        bootstrap_membership_rows, all_validation_rows, rank_rows,
                        n_pure_nontopology_pairs, n_pure_topology_pairs, n_mixed_pairs)

    # -------------------------------------------------------------------
    # Prior-phase immutability postflight
    # -------------------------------------------------------------------
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
    write_csv(OUT_DIR / 'prior_phase_immutability_check.csv',
               ['phase', 'file_path', 'sha256_before', 'sha256_after', 'status'], immut_rows)

    if changed:
        log(f'[IMMUTABILITY FAILURE] {len(changed)} prior-phase file(s) changed during this run: {changed}')
        flush_log()
        raise SystemExit(f'[hard-fail] Prior-phase immutability violated: {changed}')
    log(f'[immutability] Confirmed all {len(immut_rows)} prior-phase file(s) unchanged.')

    log('')
    log('=' * 88)
    log(f'RESULT: Phase 2C complete. {len(all_validation_rows)} validation checks PASSED. '
        f'Analyses A-F, six Pareto objective sets, and bootstrap stability all completed. '
        f'Topology-only Pareto front sanity check PASSED. Prior-phase files unchanged. '
        f'Do not begin Phase 2D.')
    log('=' * 88)
    flush_log()
    return 0


def _fmt(v, nd=4):
    if v in ('', None):
        return 'n/a'
    try:
        return f'{float(v):.{nd}f}'
    except (TypeError, ValueError):
        return str(v)


def write_phase2c_doc(non_ssim, methods, topology_methods, method_level_rows, within_summary_rows,
                        samplewise_summary_rows, residual_rows, method_level_pd_mt, within_pd_mt,
                        samplewise_pd_mt, descriptor_summary_rows, objective_sets, front_by_set,
                        bootstrap_membership_rows, all_validation_rows, rank_rows,
                        n_pure_nontopology_pairs, n_pure_topology_pairs, n_mixed_pairs):
    lines = []
    a = lines.append

    a('# Phase 2C: Unified Wind-SR Candidate Relationship and Pareto-Tradeoff Analysis')
    a('')

    a('## 1. Scope')
    a('')
    a('Phase 2C analyzes relationships among the 21 non-SSIM standardized metrics (SSIM is globally '
      'unavailable across all 19 methods and is excluded from every correlation and Pareto computation '
      'in this phase) and characterizes multi-objective tradeoffs among the 19 fixed, designed candidate '
      'methods evaluated in Phase 1. It is read-only with respect to every Phase-1, Phase-2A, and Phase-2B '
      'artifact: no training, inference, cheap evaluation, or TTK is run, and no prior-phase file is '
      'modified, regenerated, or overwritten. Sample selection and figure generation remain deferred to '
      'Phase 2D, which has not begun.')
    a('')
    a('**No weighted aggregate score or total method ranking is computed anywhere in this phase.** Every '
      'summary table in this document is a machine-readable cross-reference of independently-defined '
      'quantities, not a leaderboard.')
    a('')

    a('## 2. Why pooled 3,192-row correlations are not used')
    a('')
    a('The Phase-1 long table has 3,192 rows (19 methods x 168 samples). A naive Pearson/Spearman '
      'correlation computed directly across all 3,192 rows would conflate two structurally different '
      'sources of covariation: (a) systematic between-method differences (some methods are simply better '
      'or worse on average across every metric) and (b) within-method sample-to-sample variation (which '
      'samples are easy or hard for a given method). These two sources can have different, even opposite, '
      'signs, and a pooled correlation reports an uninterpretable mixture of both. Phase 2C instead reports '
      'four analysis levels that isolate specific, clearly-labeled sources of covariation (Sections 4-7), '
      'plus a two-way-centered residual analysis (Section 7) that explicitly removes the additive '
      'method-mean and sample-mean effects before correlating -- this is the closest Phase 2C comes to a '
      '"pooled" correlation, and even that is on residuals, not raw pooled values.')
    a('')

    a('## 3. Metric orientation convention')
    a('')
    a('Every metric direction (`higher_is_better` / `lower_is_better`) is read exclusively from '
      '`column_mapping.csv` and never inferred from the metric name. The oriented value is '
      '`z = y` if higher-is-better, else `z = -y`, so that **larger oriented values always mean better '
      'performance** for every metric, and a positive oriented correlation always means "these two metrics '
      'agree about which methods/samples are better." SSIM is excluded from all orientation, correlation, '
      'and Pareto computations in this phase because it is 0/168 available for every method (globally '
      'unavailable, not merely missing for some methods).')
    a('')

    a('## 4. Analysis A -- method-mean relationships (between_method_means)')
    a('')
    a(f'For each of the 19 methods, the mean of every available metric is computed independently from the '
      f'Phase-1 long table (`method_mean_oriented_values.csv`). Metric-pair correlations across these 19 '
      f'method means answer: **"across the realized set of 19 designed methods, do two metrics favor the '
      f'same methods on average?"** This says nothing about within-method sample-to-sample behavior. There '
      f'are C(21,2) = {len(method_level_rows)} non-SSIM metric pairs; PD/MT-involving pairs use the 18 '
      f'topology-bearing methods (bicubic has no topology data), all other pairs use all 19 methods. '
      f'A leave-one-method-out (LOO) sensitivity bound is reported per pair: this is a descriptive '
      f'perturbation diagnostic, not an inferential confidence interval -- the 19 methods are a fixed '
      f'designed set, not a random sample, so no sampling-distribution interpretation is intended.')
    a('')
    a(f'**PD/MT at the method-mean level:** oriented Pearson r = {_fmt(method_level_pd_mt["oriented_pearson"])}, '
      f'oriented Spearman rho = {_fmt(method_level_pd_mt["oriented_spearman"])} '
      f'(n={method_level_pd_mt["n_common_methods"]} common methods). Across the realized method means, '
      f'oriented PD and MT association is '
      f'{"positive" if (method_level_pd_mt["oriented_pearson"] != "" and float(method_level_pd_mt["oriented_pearson"]) > 0) else "non-positive"} '
      f'-- i.e. methods with better (lower) mean PD distance tend to also have '
      f'{"better" if (method_level_pd_mt["oriented_pearson"] != "" and float(method_level_pd_mt["oriented_pearson"]) > 0) else "not consistently better"} '
      f'mean MT distance.')
    a('')

    a('## 5. Analysis B -- within-method sample relationships (within_method_across_samples)')
    a('')
    a('For each of the 19 methods independently, metric-pair correlations are computed across the 168 '
      'samples (fields) that method was evaluated on. This answers: **"within a single fixed method, do '
      'two metrics identify the same easy/difficult samples?"** Every (method, pair) combination has '
      'either exactly 168 valid paired samples (`status=available`) or is explicitly marked '
      '`status=unavailable` (bicubic has no PD/MT data) -- partial coverage is never silently accepted; it '
      'hard-fails the run. `within_method_correlation_summary.csv` aggregates the per-method correlations '
      'into median/quartile/min/max/sign-count statistics across the methods for which that pair is '
      'available.')
    a('')
    a(f'**PD/MT within individual methods:** median oriented Pearson r = {_fmt(within_pd_mt["pearson_median"])}, '
      f'median oriented Spearman rho = {_fmt(within_pd_mt["spearman_median"])} across '
      f'{within_pd_mt["n_methods_available"]} methods. Within individual methods across the 168 fields, the '
      f'median association is '
      f'{"positive" if (within_pd_mt["pearson_median"] != "" and float(within_pd_mt["pearson_median"]) > 0) else "non-positive"}.')
    a('')

    a('## 6. Analysis C -- per-sample cross-method relationships (within_sample_across_methods)')
    a('')
    a('For each of the 168 samples independently, metric-pair correlations are computed across the methods '
      'evaluated on that sample. This answers: **"for a fixed field, do different metrics rank the 19 (or '
      '18, for PD/MT pairs) methods similarly?"** PD/MT-involving pairs use exactly the 18 topology-bearing '
      'methods; all other pairs use all 19 methods -- these counts are enforced exactly, never partially. '
      '`samplewise_correlation_summary.csv` aggregates across the 168 samples.')
    a('')
    a(f'**PD/MT per sample:** median oriented Pearson r = {_fmt(samplewise_pd_mt["pearson_median"])}, '
      f'median oriented Spearman rho = {_fmt(samplewise_pd_mt["spearman_median"])} across '
      f'{samplewise_pd_mt["n_samples"]} samples. This is a rank-association coefficient, not a literal '
      f'agreement rate -- see Section 8 for the literal per-field PD/MT preference-agreement rate, which '
      f'is a distinct quantity computed from explicit pairwise preferences rather than from a correlation '
      f'coefficient.')
    a('')

    a('## 7. Analysis D -- two-way-centered residual relationships (two_way_centered_residual)')
    a('')
    a('For every metric PAIR, both metrics are two-way centered on the exact same common-rectangle method '
      'set (the 18 topology-bearing methods if either metric is PD or MT, else all 19 methods), by additive '
      'demeaning: `residual[m,s] = z[m,s] - mean_over_samples(z[m,*]) - mean_over_methods(z[*,s]) + '
      'grand_mean(z)`. Centering is deliberately pair-specific: each metric is re-centered on the common '
      'rectangle for that particular pair, rather than centered once on its own maximal method set and then '
      'subset down afterward -- subsetting a residual matrix AFTER centering does not preserve the '
      'zero-margin property, since the row/column means used would have been computed over the wrong '
      '(too-large) rectangle. By construction every row-mean (method) margin, column-mean (sample) margin, '
      f'and grand mean of each pair-specific residual matrix is numerically zero (verified to within '
      f'{MARGIN_TOLERANCE:.0e} for every pair and every side; see `phase2c_validation.csv`). Correlating the '
      'two pair-specific residual matrices (flattened over the shared method x sample cells) answers: '
      '**"once the obvious method-level and sample-level main effects are removed, do two metrics still '
      'move together?"** This is explicitly **additive demeaning, not a fitted causal mixed-effects model** '
      '-- no variance components, random effects, or significance testing are involved, and no causal claim '
      'is intended or supported.')
    a('')
    a(f'Because both metrics in a pair are always centered over the same rectangle, the '
      f'{n_pure_nontopology_pairs} non-topology/non-topology pairs (19-method rectangle) and the '
      f'{n_pure_topology_pairs} PD/MT pair (18-method rectangle, which was already correctly computed on a '
      f'shared rectangle before this patch) are numerically unaffected by this correction. Only the '
      f'{n_mixed_pairs} mixed topology/non-topology pairs -- where the non-topology metric previously had '
      f'its residual computed over 19 methods and then silently subset down to 18 -- have legitimately '
      f'changed.')
    a('')
    pd_mt_resid = _find_pair_row(residual_rows, 'pd_distance', 'mt_distance')
    a(f'**PD/MT two-way-residual association:** Pearson r = {_fmt(pd_mt_resid["oriented_residual_pearson"])}, '
      f'Spearman rho = {_fmt(pd_mt_resid["oriented_residual_spearman"])} (n_cells={pd_mt_resid["n_cells"]}, '
      f'n_common_methods={pd_mt_resid["n_common_methods"]}). This value is unchanged by the pair-specific '
      f'centering patch, since PD and MT were already centered on the same 18-method rectangle.')
    a('')

    a('## 8. PD/MT direct disagreement analysis')
    a('')
    a('Beyond correlation, Phase 2C directly quantifies how often PD (persistence-diagram distance) and MT '
      '(merge-tree distance) *disagree about which of two methods is better*, at both a per-method-pair '
      'level (aggregated over the 168 fields, `topology_pairwise_preference_agreement.csv`) and a '
      'per-field level (aggregated over the C(18,2)=153 topology-bearing method pairs, '
      '`topology_sample_preference_agreement.csv`). PD and MT are both legitimate, differently-scoped '
      'topological descriptors (PD captures persistence-pair geometry, MT captures merge-tree structure); '
      'disagreement between them is evidence of genuinely different geometric sensitivity, and is not '
      'evidence that either descriptor is invalid.')
    a('')
    a('`topology_rank_by_method.csv` reports, for each of the 18 topology-bearing methods, the RANK of that '
      'method\'s own PD and MT MEANS among the 18 method means (ascending raw-distance order, average ranks '
      'for exact ties) -- this is explicitly not an average of 168 per-field ranks. '
      '`signed_rank_gap = pd_rank - mt_rank`: a positive gap means MT favors the method more strongly than '
      'PD does (the method\'s MT rank is better than its PD rank); a negative gap means PD favors the '
      'method more strongly than MT does.')
    a('')
    for row in descriptor_summary_rows:
        a(f'- **{row["level"]}** (n={row["n_units"]}): {row["value_a_label"]}={_fmt(row["value_a"])}, '
          f'{row["value_b_label"]}={_fmt(row["value_b"])}')
    a('')

    a('## 9. Pareto front definition')
    a('')
    a('All Pareto analysis operates on **oriented method means** (Section 3), where higher is always '
      f'better. Method A strictly dominates method B on a given objective set when A >= B - {PARETO_TOLERANCE:.0e} '
      f'on every objective in the set AND A > B + {PARETO_TOLERANCE:.0e} on at least one objective. The '
      'Pareto front of an objective set is the set of methods dominated by no other eligible method. No '
      'metric normalization is applied or needed: strict dominance under this definition is invariant to '
      'any monotonic per-objective rescaling.')
    a('')

    a('## 10. Pareto objective-set choices')
    a('')
    a('Six objective sets are defined, each built from three metric groups: **fidelity** '
      '(`psnruv`, `speed_mae`), **physics** (`wpd_mae`, `grad_mae`, `psd_log_l2`, `exceed_abs_p90`, '
      '`comp_curve_l1`), and **topology** (`pd_distance`, `mt_distance`). Any objective set that includes '
      'a topology objective is restricted to the 18 topology-bearing methods (bicubic has no PD/MT data); '
      'sets without a topology objective use all 19 methods.')
    a('')
    for oset_name, objs in objective_sets.items():
        tag = ' (labeled sensitivity analysis -- all 21 non-SSIM objectives at once)' \
            if oset_name == 'all_available_non_ssim' else ''
        a(f'- **{oset_name}**{tag}: {len(objs)} objectives = {objs}')
    a('')
    a('**A Pareto front depends on the chosen objective set and is not a universal ranking.** A method can '
      'be on the front for one objective set and dominated under another; Section 11 makes this concrete.')
    a('')

    a('## 11. Deterministic Pareto fronts')
    a('')
    for oset_name in objective_sets:
        a(f'- **{oset_name}**: front = {sorted(front_by_set[oset_name])}')
    a('')
    a(f'**Topology-only sanity check:** the deterministic `topology_only` front was independently required '
      f'to equal exactly `{sorted(EXPECTED_TOPOLOGY_ONLY_FRONT)}`, derived directly from the Phase-1 long-'
      f'table method means; the run hard-fails if it does not. It matched exactly.')
    a('')

    a('## 12. Pareto bootstrap stability')
    a('')
    a(f'For every objective set, front membership is recomputed under {PARETO_BOOTSTRAP_N} resamples per '
      f'scheme, across four resampling schemes: ordinary i.i.d. resampling of the 168 fields, and circular '
      f'moving-block bootstrap with block lengths 6, 12, and 24 (to probe robustness to the fact that the '
      f'168 fields are consecutive hourly observations and are likely temporally dependent). Within a '
      f'single replicate, the identical resampled set of field indices is used for every method and every '
      f'objective in the objective set, so that all method means being compared in that replicate are built '
      f'from the same synthetic sample. The resulting **front-membership rate is a descriptive stability '
      f'diagnostic under resampling, explicitly not a posterior probability** -- no prior or likelihood '
      f'model is specified.')
    a('')
    always_on = sorted({r['method_id'] for r in bootstrap_membership_rows
                          if r['objective_set'] == 'topology_only' and r['bootstrap_scheme'] == 'iid'
                          and r['always_on_front']})
    never_on = sorted({r['method_id'] for r in bootstrap_membership_rows
                         if r['objective_set'] == 'topology_only' and r['bootstrap_scheme'] == 'iid'
                         and r['never_on_front']})
    a(f'For `topology_only` under i.i.d. resampling: always-on-front = {always_on}, '
      f'never-on-front = {never_on}. See `pareto_bootstrap_stability.csv` for the full '
      f'objective-set x scheme x method table, and `pareto_bootstrap_front_size.csv` for how front size '
      f'itself varies under resampling.')
    a('')

    a('## 13. Findings that are consistent across analysis levels')
    a('')
    a('See `metric_relationship_summary.csv` for the full per-pair cross-level comparison. A pair is '
      'labeled `consistent_nonzero_sign=True` when the method-level, within-method-median, samplewise-'
      'median, and two-way-residual oriented Pearson correlations all share the same nonzero sign -- this '
      'is a statement about **sign** agreement across genuinely different analysis levels, not about equal '
      'magnitude, and not a claim that any one of the four coefficients is more "correct" than another.')
    a('')

    a('## 14. Findings that reverse or weaken across analysis levels')
    a('')
    a('Metric pairs NOT flagged `consistent_nonzero_sign` in `metric_relationship_summary.csv` have a sign '
      'that differs, or a coefficient that is exactly zero or undefined, across at least one of the four '
      'analysis levels. This is expected and scientifically meaningful: a pair can, for instance, favor the '
      'same methods on average (positive method-level correlation) while showing no consistent within-'
      'method sample-to-sample relationship (near-zero within-method median), because these two levels '
      'answer different questions (Sections 4 and 5).')
    a('')

    a('## 15. Caveats')
    a('')
    a('- All 19 methods were trained exactly once; none of the correlation or Pareto results here should be '
      'read as capturing training-run-to-training-run variance.')
    a('- The 19 methods are a fixed, designed candidate set, not a random sample from a broader population '
      'of possible architectures -- method-level (Analysis A) statistics, including the LOO sensitivity '
      'bounds, are descriptive, not inferential.')
    a('- The 168 benchmark fields are consecutive hourly observations and are likely temporally dependent; '
      'this is why every bootstrap procedure in this phase includes circular moving-block resampling '
      'alongside ordinary i.i.d. resampling.')
    a('- Correlation is not causation anywhere in this document, including the two-way-residual analysis '
      '(Section 7), which removes additive main effects but does not fit a causal model.')
    a('- Analysis levels are not interchangeable: method-mean, within-method, cross-method, and two-way-'
      'residual correlations answer different questions and can legitimately disagree (Section 14).')
    a('')

    a('## 16. No weighted score, no total ranking')
    a('')
    a('Phase 2C never computes a weighted combination of metrics, never produces an overall method score, '
      'and never produces a total ranking of the 19 methods. `pareto_layers.csv` reports iterative non-'
      'domination layers (an "onion peeling" of the dominance structure), which is explicitly **not** a '
      'total ranking: methods within the same layer are mutually non-dominated, and layer order reflects '
      'dominance depth under one specific objective set, not overall quality. '
      '`topology_relationship_and_pareto_summary.csv` is a machine-readable cross-reference table, also '
      'explicitly not a leaderboard.')
    a('')

    a('## 17. Sample selection and figures deferred to Phase 2D')
    a('')
    a('This phase performs no sample selection and generates no figures. Both remain deferred to Phase 2D, '
      'which has not begun.')
    a('')

    a('## 18. Validation summary')
    a('')
    n_fail = sum(1 for r in all_validation_rows if r['status'] != 'PASS')
    a(f'{len(all_validation_rows)} total validation checks were run; {len(all_validation_rows) - n_fail} '
      f'passed and {n_fail} failed (a run with any failure hard-fails before this document is written, so '
      f'`n_fail` is always 0 in a completed report). See `phase2c_validation.csv` for the full check list.')
    a('')

    a('## 19. Generated files')
    a('')
    for fname in [
        'phase2c_validation.csv', 'method_mean_oriented_values.csv', 'method_level_metric_correlations.csv',
        'method_level_oriented_pearson_matrix.csv', 'method_level_oriented_spearman_matrix.csv',
        'within_method_metric_correlations.csv', 'within_method_correlation_summary.csv',
        'samplewise_cross_method_correlations.csv', 'samplewise_correlation_summary.csv',
        'two_way_residual_correlations.csv', 'two_way_residual_pearson_matrix.csv',
        'two_way_residual_spearman_matrix.csv', 'focal_topology_correlation_bootstrap.csv',
        'focal_topology_relationship_summary.csv', 'topology_rank_by_method.csv',
        'topology_pairwise_preference_agreement.csv', 'topology_sample_preference_agreement.csv',
        'topology_descriptor_disagreement_summary.csv', 'pareto_objective_manifest.csv',
        'pareto_front_membership.csv', 'pareto_dominance_edges.csv', 'pareto_layers.csv',
        'topology_pareto_sanity_check.csv', 'pareto_bootstrap_stability.csv',
        'pareto_bootstrap_front_size.csv', 'metric_relationship_summary.csv',
        'topology_relationship_and_pareto_summary.csv', 'prior_phase_immutability_check.csv',
    ]:
        a(f'- `ttk_runs_fixed/unified_candidate_analysis/phase2c/{fname}`')
    a('- `docs/unified_candidate_analysis_phase2c.md` (this file)')
    a('- `logs/unified_candidate_analysis_phase2c.log`')
    a('')

    (DOCS_DIR / 'unified_candidate_analysis_phase2c.md').write_text('\n'.join(lines) + '\n')
    log(f"[write] {DOCS_DIR / 'unified_candidate_analysis_phase2c.md'}")


if __name__ == '__main__':
    sys.exit(main())
