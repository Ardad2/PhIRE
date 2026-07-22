#!/usr/bin/env python3
"""Phase 2B: controlled factorial and targeted loss-term analyses of the
unified wind-SR candidate benchmark, built exclusively from the immutable
Phase-1 and Phase-2A outputs.

This script is read-only with respect to every Phase-1 and Phase-2A
artifact. It never runs training, inference, cheap evaluation, or TTK, and
never writes anywhere except:

    ttk_runs_fixed/unified_candidate_analysis/phase2b/
    docs/unified_candidate_analysis_phase2b.md
    logs/unified_candidate_analysis_phase2b.log

Before and after the run it SHA-256-checksums every protected Phase-1 file
(12) and every protected Phase-2A file (14) -- 26 total, all identified by
an explicit list, never a glob -- and hard-fails if any changed, disappeared,
or if an unexpected new file appears in either frozen output directory.

Scope: (A) the complete Candidate-B 2^3 factorial (speed x grad x levelset),
(B) critical-maxima-proxy x repaired-E2 2^2 on the B scaffold, (C) level-set
x repaired-E2 2^2 on the grad scaffold, (D) 12 targeted matched-pair
contrasts (adding crit, adding E2, E2-vs-crit, scaffold-pruning). Explicitly
OUT of scope (deferred to Phase 2C/2D): metric correlations, Pareto-front
analysis, sample selection, visualization.

Every model here was trained exactly once. Every effect and contrast in
this script describes a factorial/paired-difference relationship AMONG THE
REALIZED TRAINED MODELS on this fixed 168-sample benchmark. None of it
establishes training-seed robustness or a universal causal claim about a
loss term -- see docs/unified_candidate_analysis_phase2b.md section 13.

Determinism: every bootstrap resample (ordinary and circular block) uses a
fixed seed derived from 20260721 and a precomputed index matrix, so
re-running this script produces byte-identical output. No wall-clock time,
hostname, or other non-deterministic value is ever written to a generated
file.
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

try:
    import scipy.stats as _scipy_stats
    HAVE_SCIPY = True
except Exception:
    _scipy_stats = None
    HAVE_SCIPY = False

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
os.chdir(REPO_ROOT)

PHASE1_DIR = REPO_ROOT / 'ttk_runs_fixed' / 'unified_candidate_evaluation'
PHASE2A_DIR = REPO_ROOT / 'ttk_runs_fixed' / 'unified_candidate_analysis' / 'phase2a'
OUT_DIR = REPO_ROOT / 'ttk_runs_fixed' / 'unified_candidate_analysis' / 'phase2b'
DOCS_DIR = REPO_ROOT / 'docs'
LOG_PATH = REPO_ROOT / 'logs' / 'unified_candidate_analysis_phase2b.log'

# -----------------------------------------------------------------------
# Explicit protected-file sets (never a glob) -- patch philosophy carried
# forward from the hardened Phase-2A script.
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

N_EVAL = 168
CNN_METHOD = 'cnn'
TIE_TOLERANCE = 1e-12
SANITY_TOLERANCE = 1e-4          # tolerance for the task-provided PD/MT sanity-check constants
LINEARITY_TOLERANCE = 1e-10      # mean-of-sample-effects == method-mean effect; mean-contrast == mean diff
RECONSTRUCTION_TOLERANCE = 1e-12  # saturated factorial reconstruction
BOOTSTRAP_SEED = 20260721
BOOTSTRAP_N = 10000
BLOCK_LENGTHS = [6, 12, 24]

FACTOR_TO_USES_COLUMN = {
    'speed': 'uses_speed', 'grad': 'uses_grad', 'levelset': 'uses_levelset',
    'crit': 'uses_crit', 'e2': 'uses_e2',
}

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
    """Requires the exact expected Phase-1 (12) + Phase-2A (14) protected
    file set to exist -- hard-fails with the complete missing-file list if
    any are absent, and hard-fails if an unexpected extra CSV appears in
    either frozen directory (their schemas are intended to be immutable)."""
    all_required = PHASE1_PROTECTED_FILES + PHASE2A_PROTECTED_FILES
    missing = [str(p) for p in all_required if not p.exists()]
    if missing:
        raise SystemExit(
            f'[hard-fail] Missing required prior-phase protected file(s) (expected exactly '
            f'{len(all_required)}: 12 Phase-1 + 14 Phase-2A):\n' + '\n'.join(f'  - {m}' for m in missing)
        )
    for directory, expected_csvs in ((PHASE1_DIR, set(PHASE1_PROTECTED_CSVS)),
                                       (PHASE2A_DIR, set(PHASE2A_PROTECTED_CSVS))):
        actual_csvs = sorted(directory.glob('*.csv'), key=str)
        unexpected = [str(p) for p in actual_csvs if p not in expected_csvs]
        if unexpected:
            raise SystemExit(
                f'[hard-fail] Unexpected extra CSV(s) found in frozen directory {directory} '
                f'(schema is intended to be immutable): {unexpected}'
            )


def checksum_all(files: list) -> dict:
    """Returns {repo-relative POSIX path: sha256 or None}."""
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


# =============================================================================
# Bootstrap infrastructure (ordinary iid + circular moving-block, all
# precomputed once for the common n=168 case).
# =============================================================================

_IID_IDX_168 = np.random.default_rng(BOOTSTRAP_SEED).integers(0, N_EVAL, size=(BOOTSTRAP_N, N_EVAL))


def _make_block_index_matrix(block_length: int) -> np.ndarray:
    """Deterministic circular moving-block bootstrap index matrix, shape
    (BOOTSTRAP_N, N_EVAL). Start positions are drawn uniformly from
    0..N_EVAL-1 with replacement (seeded deterministically from
    BOOTSTRAP_SEED and block_length); consecutive circular blocks of
    block_length are appended and the concatenated index sequence is
    truncated to exactly N_EVAL."""
    seed = BOOTSTRAP_SEED * 1000 + block_length
    rng = np.random.default_rng(seed)
    n_blocks = math.ceil(N_EVAL / block_length)
    starts = rng.integers(0, N_EVAL, size=(BOOTSTRAP_N, n_blocks))
    offsets = np.arange(block_length)
    idx = (starts[:, :, None] + offsets[None, None, :]) % N_EVAL
    idx = idx.reshape(BOOTSTRAP_N, n_blocks * block_length)[:, :N_EVAL]
    return idx


_BLOCK_IDX = {L: _make_block_index_matrix(L) for L in BLOCK_LENGTHS}


def bootstrap_ci_all(values) -> dict:
    """Returns {'iid': (lo, hi), 'block6': (lo, hi), 'block12': (lo, hi),
    'block24': (lo, hi)}, each (None, None) if values is empty or its length
    isn't N_EVAL (the only case that occurs in this dataset: 0 for SSIM)."""
    values = np.asarray(values, dtype=np.float64)
    n = values.shape[0]
    out = {}
    if n == N_EVAL:
        out['iid'] = _percentile_ci(values[_IID_IDX_168].mean(axis=1))
        for L in BLOCK_LENGTHS:
            out[f'block{L}'] = _percentile_ci(values[_BLOCK_IDX[L]].mean(axis=1))
    else:
        out['iid'] = (None, None)
        for L in BLOCK_LENGTHS:
            out[f'block{L}'] = (None, None)
    return out


def _percentile_ci(resampled_means: np.ndarray) -> tuple:
    return float(np.percentile(resampled_means, 2.5)), float(np.percentile(resampled_means, 97.5))


# =============================================================================
# Exact two-sided sign test (p=0.5) and Holm step-down correction
# =============================================================================

def exact_sign_test_pvalue(n_pos: int, n_neg: int):
    n = n_pos + n_neg
    if n == 0:
        return None
    k = min(n_pos, n_neg)
    tail = sum(math.comb(n, i) for i in range(0, k + 1))
    return min(1.0, 2 * (tail / (2 ** n)))


def holm_correction(items: list) -> dict:
    valid = [(k, p) for k, p in items if p is not None]
    m = len(valid)
    if m == 0:
        return {}
    valid.sort(key=lambda kv: kv[1])
    adjusted = {}
    running_max = 0.0
    for i, (key, p) in enumerate(valid):
        adj = (m - i) * p
        running_max = max(running_max, adj)
        adjusted[key] = min(1.0, running_max)
    return adjusted


# =============================================================================
# Oriented-series summary statistics (shared by factorial effect summaries
# and targeted-contrast summaries -- identical statistical machinery).
# =============================================================================

def summarize_oriented_series(raw_vals: np.ndarray, oriented_vals: np.ndarray) -> dict:
    """raw_vals/oriented_vals: 1-D arrays, same finite mask (NaN in the same
    positions in both). Returns every field needed by an effect-summary or
    contrast-summary row except the identity columns."""
    finite_mask = np.isfinite(oriented_vals)
    n_valid = int(finite_mask.sum())
    if n_valid == 0:
        return dict(n_valid=0, mean_raw='', median_raw='', mean_oriented='', median_oriented='',
                     std='', se='', ci=dict(iid=('', ''), block6=('', ''), block12=('', ''), block24=('', '')),
                     pos=0, zero=0, neg=0, pos_rate='', zero_rate='', neg_rate='',
                     dz='', sign_p='', wilcoxon_p='', test_status='no_valid_data')

    raw_f = raw_vals[finite_mask]
    or_f = oriented_vals[finite_mask]
    mean_raw = float(raw_f.mean())
    median_raw = float(np.median(raw_f))
    mean_or = float(or_f.mean())
    median_or = float(np.median(or_f))
    if n_valid >= 2:
        std = float(or_f.std(ddof=1))
        se = std / math.sqrt(n_valid)
    else:
        std = None
        se = None
    ci = bootstrap_ci_all(or_f) if n_valid == N_EVAL else dict(
        iid=('', ''), block6=('', ''), block12=('', ''), block24=('', ''))

    pos = int(np.sum(or_f > TIE_TOLERANCE))
    neg = int(np.sum(or_f < -TIE_TOLERANCE))
    zero = n_valid - pos - neg
    dz = (mean_or / std) if (std is not None and std != 0) else ''

    n_nontied = pos + neg
    sign_p = exact_sign_test_pvalue(pos, neg) if n_nontied > 0 else None
    sign_note = 'ok' if sign_p is not None else 'sign_test_undefined_zero_nontied_pairs'

    wilcoxon_p = None
    wilcoxon_note = 'scipy_unavailable'
    if HAVE_SCIPY:
        try:
            if np.all(or_f == 0):
                wilcoxon_note = 'wilcoxon_undefined_all_zero_differences'
            else:
                res = _scipy_stats.wilcoxon(or_f, alternative='two-sided', zero_method='wilcox', mode='auto')
                wilcoxon_p = float(res.pvalue)
                wilcoxon_note = 'ok'
        except Exception as e:
            wilcoxon_note = f'wilcoxon_failed:{type(e).__name__}'

    return dict(
        n_valid=n_valid, mean_raw=mean_raw, median_raw=median_raw, mean_oriented=mean_or,
        median_oriented=median_or, std=(std if std is not None else ''), se=(se if se is not None else ''),
        ci=ci, pos=pos, zero=zero, neg=neg,
        pos_rate=pos / n_valid, zero_rate=zero / n_valid, neg_rate=neg / n_valid,
        dz=dz, sign_p=(sign_p if sign_p is not None else ''),
        wilcoxon_p=(wilcoxon_p if wilcoxon_p is not None else ''),
        test_status=f'sign_test={sign_note}; wilcoxon={wilcoxon_note}',
    )


# =============================================================================
# Factorial design machinery
# =============================================================================

def factorial_subsets(factor_names: tuple) -> list:
    """All subsets of factor_names in canonical order: () first (intercept),
    then increasing order, matching itertools.combinations' natural order --
    which for ('speed','grad','levelset') gives exactly the task's requested
    effect order: speed, grad, levelset, speed:grad, speed:levelset,
    grad:levelset, speed:grad:levelset."""
    subsets = []
    for r in range(0, len(factor_names) + 1):
        subsets.extend(itertools.combinations(factor_names, r))
    return subsets


def build_coded_matrix(cells: list, factor_names: tuple, subsets: list) -> np.ndarray:
    """cells: list of (method_id, level_dict) where level_dict[factor] in {0,1}.
    Returns an (n_cells, n_subsets) matrix of +-1 products, column 0 = all 1s
    (intercept)."""
    n_cells = len(cells)
    n_subsets = len(subsets)
    coded = np.zeros((n_cells, n_subsets), dtype=np.float64)
    for ci, (_, levels) in enumerate(cells):
        coded_levels = {f: (1.0 if levels[f] == 1 else -1.0) for f in factor_names}
        for si, subset in enumerate(subsets):
            prod = 1.0
            for f in subset:
                prod *= coded_levels[f]
            coded[ci, si] = prod
    return coded


class FactorialResult:
    __slots__ = ('design_id', 'factor_names', 'cells', 'subsets', 'coded_matrix', 'n_cells')

    def __init__(self, design_id, factor_names, cells):
        self.design_id = design_id
        self.factor_names = factor_names
        self.cells = cells
        self.subsets = factorial_subsets(factor_names)
        self.coded_matrix = build_coded_matrix(cells, factor_names, self.subsets)
        self.n_cells = len(cells)


def compute_beta_matrix(coded_matrix: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """coded_matrix: (n_cells, n_subsets). Y: (n_cells, n_samples) raw or
    oriented values. Returns beta: (n_subsets, n_samples) = coded_matrix.T @
    Y / n_cells (the exact OLS solution for an orthogonal saturated design)."""
    n_cells = coded_matrix.shape[0]
    return (coded_matrix.T @ Y) / n_cells


def effect_label(subset: tuple) -> str:
    return ':'.join(subset)


def effect_factors_str(subset: tuple) -> str:
    return ','.join(subset)


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


def orient(y: np.ndarray, direction: str) -> np.ndarray:
    return y if direction == 'higher_is_better' else -y


# =============================================================================
# Base data validation (phase2b_validation.csv)
# =============================================================================

def run_base_validation(long_table, phase1_refs, phase2a_refs, metric_direction, methods_used_in_designs):
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

    add('long_table_row_count', long_table['n_rows'], 19 * N_EVAL, 0, long_table['n_rows'] == 19 * N_EVAL)
    add('duplicate_keys', len(long_table['dup_keys']), 0, 0, len(long_table['dup_keys']) == 0)
    add('method_count', len(per_sample), 19, 0, len(per_sample) == 19)
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
    add('metric_directions_resolved', len(missing_direction), 0, 0, len(missing_direction) == 0,
        notes=str(missing_direction))

    # Phase-2A validation must contain no failures
    n_p2a_fail = sum(1 for r in phase2a_refs['validation'] if r['status'] != 'PASS')
    add('phase2a_validation_no_failures', n_p2a_fail, 0, 0, n_p2a_fail == 0)

    # Phase-2A pairwise reproduction must contain no failures
    n_repro_fail = sum(1 for r in phase2a_refs['pairwise_repro'] if r['status'] != 'PASS')
    add('phase2a_pairwise_reproduction_no_failures', n_repro_fail, 0, 0, n_repro_fail == 0)

    # Phase-2A's own immutability check (of Phase-1) must show only 'unchanged'
    n_immut_bad = sum(1 for r in phase2a_refs['immutability'] if r['status'] != 'unchanged')
    add('phase2a_recorded_immutability_all_unchanged', n_immut_bad, 0, 0, n_immut_bad == 0)

    # Every non-SSIM metric complete (168/168) for every method used by any design/contrast
    for mid in sorted(methods_used_in_designs):
        for col in metric_cols:
            finite = sum(1 for si in range(N_EVAL) if math.isfinite(per_sample[mid][si].get(col, float('nan'))))
            if col == 'ssim_speed':
                add(f'ssim_0_or_168[{mid}]', finite, '0 or 168', 0, finite in (0, N_EVAL))
            else:
                add(f'required_metric_complete[{mid}][{col}]', finite, N_EVAL, 0, finite == N_EVAL)

    return checks, failures


# =============================================================================
# Design specifications
# =============================================================================

DESIGN_A_CELLS = [
    ('uv', {'speed': 0, 'grad': 0, 'levelset': 0}),
    ('speed_only', {'speed': 1, 'grad': 0, 'levelset': 0}),
    ('levelset_only', {'speed': 0, 'grad': 0, 'levelset': 1}),
    ('speed_levelset', {'speed': 1, 'grad': 0, 'levelset': 1}),
    ('grad_only', {'speed': 0, 'grad': 1, 'levelset': 0}),
    ('speed_grad', {'speed': 1, 'grad': 1, 'levelset': 0}),
    ('grad_levelset', {'speed': 0, 'grad': 1, 'levelset': 1}),
    ('candidate_b', {'speed': 1, 'grad': 1, 'levelset': 1}),
]
DESIGN_A_FACTORS = ('speed', 'grad', 'levelset')
DESIGN_A_SANITY_PD = {'speed': 0.0168, 'grad': 6.7481, 'levelset': 0.1839, 'speed:grad': -0.0796,
                        'speed:levelset': 0.0189, 'grad:levelset': 0.1045, 'speed:grad:levelset': -0.0437}
DESIGN_A_SANITY_MT = {'speed': -0.030075, 'grad': -0.186025, 'levelset': 0.011375, 'speed:grad': -0.067975,
                        'speed:levelset': 0.081025, 'grad:levelset': -0.018525, 'speed:grad:levelset': 0.055425}

DESIGN_B_CELLS = [
    ('candidate_b', {'crit': 0, 'e2': 0}),
    ('candidate_c', {'crit': 1, 'e2': 0}),
    ('b_e2', {'crit': 0, 'e2': 1}),
    ('c_e2', {'crit': 1, 'e2': 1}),
]
DESIGN_B_FACTORS = ('crit', 'e2')
DESIGN_B_SANITY_PD = {'crit': -0.0342, 'e2': -1.5274, 'crit:e2': -0.2468}
DESIGN_B_SANITY_MT = {'crit': 0.04775, 'e2': 0.45065, 'crit:e2': -0.03315}

DESIGN_C_CELLS = [
    ('grad_only', {'levelset': 0, 'e2': 0}),
    ('grad_levelset', {'levelset': 1, 'e2': 0}),
    ('f1_grad_e2', {'levelset': 0, 'e2': 1}),
    ('f2_grad_levelset_e2', {'levelset': 1, 'e2': 1}),
]
DESIGN_C_FACTORS = ('levelset', 'e2')
DESIGN_C_SANITY_PD = {'levelset': 0.20165, 'e2': -1.01715, 'levelset:e2': -0.11155}
DESIGN_C_SANITY_MT = {'levelset': -0.0806, 'e2': 0.4624, 'levelset:e2': 0.0630}


def verify_design_metadata(design_id, cells, factor_names, method_meta):
    """Hard-fails if a cell's method metadata (uses_speed/uses_grad/...)
    doesn't match its declared factor coding, or if two cells share
    identical factor coding (duplicated coding)."""
    coded_seen = {}
    for method_id, levels in cells:
        if method_id not in method_meta:
            raise SystemExit(f'[hard-fail] Design {design_id!r}: method {method_id!r} not found in the long table.')
        meta_values = method_meta[method_id]['values']
        for factor in factor_names:
            expected_bool = 'True' if levels[factor] == 1 else 'False'
            uses_col = FACTOR_TO_USES_COLUMN[factor]
            actual = meta_values.get(uses_col, '')
            if actual != expected_bool:
                raise SystemExit(
                    f'[hard-fail] Design {design_id!r}: method {method_id!r} has {uses_col}={actual!r} in '
                    f'the long table, but the design declares {factor}={levels[factor]} (expected '
                    f'{uses_col}={expected_bool!r}).'
                )
        coded_key = tuple(levels[f] for f in factor_names)
        if coded_key in coded_seen:
            raise SystemExit(
                f'[hard-fail] Design {design_id!r}: methods {coded_seen[coded_key]!r} and {method_id!r} '
                f'share identical factor coding {coded_key} -- a complete factorial design requires each '
                'cell to be unique.'
            )
        coded_seen[coded_key] = method_id
    expected_n_cells = 2 ** len(factor_names)
    if len(cells) != expected_n_cells:
        raise SystemExit(f'[hard-fail] Design {design_id!r}: expected exactly {expected_n_cells} cells for a '
                          f'complete 2^{len(factor_names)} design, got {len(cells)}.')
    if len(coded_seen) != expected_n_cells:
        raise SystemExit(f'[hard-fail] Design {design_id!r}: expected {expected_n_cells} distinct factor-coding '
                          f'combinations, found {len(coded_seen)} -- the design is missing a cell.')


def _metric_matrix(per_sample, method_order, metric) -> np.ndarray:
    return np.array([[per_sample[mid][si].get(metric, float('nan')) for si in range(N_EVAL)]
                      for mid in method_order], dtype=np.float64)


def run_factorial_design(design_id, factor_names, cells, metric_cols, metric_direction, metric_family,
                           per_sample, method_meta, sanity_pd, sanity_mt):
    """Runs one complete 2^k factorial design end-to-end. Returns a dict with
    design_rows, cell_summary_rows, sample_effect_rows, effect_summary_rows,
    reconstruction_rows -- all fully hard-fail-validated (metadata match,
    exact reconstruction, sanity-check targets, linearity of the mean)."""
    verify_design_metadata(design_id, cells, factor_names, method_meta)
    fr = FactorialResult(design_id, factor_names, cells)
    method_order = [c[0] for c in cells]
    nonempty_subsets = [s for s in fr.subsets if len(s) > 0]

    design_rows = []
    for method_id, levels in cells:
        row = dict(design_id=design_id, method_id=method_id,
                    display_name=method_meta[method_id]['values']['display_name'])
        for f in factor_names:
            row[f] = levels[f]
        for f in factor_names:
            row[f'coded_{f}'] = 1 if levels[f] == 1 else -1
        design_rows.append(row)

    cell_summary_rows = []
    sample_effect_rows = []
    effect_summary_rows = []
    reconstruction_rows = []

    for metric in metric_cols:
        direction = metric_direction[metric]
        family = metric_family.get(metric, '')
        Y_raw = _metric_matrix(per_sample, method_order, metric)         # (n_cells, 168)
        Y_oriented = orient(Y_raw, direction)

        beta_raw = compute_beta_matrix(fr.coded_matrix, Y_raw)           # (n_subsets, 168)
        beta_oriented = compute_beta_matrix(fr.coded_matrix, Y_oriented)

        # --- Reconstruction check (raw scale) ---
        Y_hat_raw = fr.coded_matrix @ beta_raw                           # (n_cells, 168)
        for ci, method_id in enumerate(method_order):
            for si in range(N_EVAL):
                obs = Y_raw[ci, si]
                if not math.isfinite(obs):
                    reconstruction_rows.append(dict(sample_idx=si, metric=metric, method_id=method_id,
                                                       observed_value='', reconstructed_value='',
                                                       absolute_difference='', status='no_data'))
                    continue
                rec = Y_hat_raw[ci, si]
                diff = abs(obs - rec)
                status = 'PASS' if diff <= RECONSTRUCTION_TOLERANCE else 'FAIL'
                if status == 'FAIL':
                    raise SystemExit(
                        f'[hard-fail] Design {design_id!r} reconstruction error for method={method_id!r} '
                        f'metric={metric!r} sample_idx={si}: observed={obs} reconstructed={rec} '
                        f'abs_diff={diff:.3e} > {RECONSTRUCTION_TOLERANCE:g}.'
                    )
                reconstruction_rows.append(dict(sample_idx=si, metric=metric, method_id=method_id,
                                                   observed_value=obs, reconstructed_value=rec,
                                                   absolute_difference=diff, status=status))

        # --- Cell summary (per method x metric) ---
        for ci, method_id in enumerate(method_order):
            vals = Y_raw[ci, :]
            finite = vals[np.isfinite(vals)]
            n_valid = finite.shape[0]
            if n_valid == 0:
                cell_summary_rows.append(dict(design_id=design_id, method_id=method_id, metric=metric,
                                                 metric_family=family, direction=direction, n_valid=0,
                                                 mean='', median='', standard_deviation='', standard_error='',
                                                 iid_bootstrap_ci95_low='', iid_bootstrap_ci95_high=''))
                continue
            mean_v = float(finite.mean())
            median_v = float(np.median(finite))
            std_v = float(finite.std(ddof=1)) if n_valid >= 2 else ''
            se_v = (std_v / math.sqrt(n_valid)) if std_v != '' else ''
            ci_lo, ci_hi = bootstrap_ci_all(finite)['iid'] if n_valid == N_EVAL else ('', '')
            cell_summary_rows.append(dict(design_id=design_id, method_id=method_id, metric=metric,
                                             metric_family=family, direction=direction, n_valid=n_valid,
                                             mean=mean_v, median=median_v, standard_deviation=std_v,
                                             standard_error=se_v, iid_bootstrap_ci95_low=ci_lo,
                                             iid_bootstrap_ci95_high=ci_hi))

        # --- Sample effects + effect summary (per nonempty subset) ---
        for subset in nonempty_subsets:
            subset_idx = fr.subsets.index(subset)
            eid = effect_label(subset)
            factors_str = effect_factors_str(subset)
            order = len(subset)

            raw_coef = beta_raw[subset_idx, :]        # (168,)
            raw_eff = 2.0 * raw_coef
            or_coef = beta_oriented[subset_idx, :]
            or_eff = 2.0 * or_coef

            for si in range(N_EVAL):
                if not math.isfinite(raw_eff[si]):
                    sample_effect_rows.append(dict(sample_idx=si, metric=metric, metric_family=family,
                                                       direction=direction, effect_id=eid, effect_order=order,
                                                       factors=factors_str, raw_regression_coefficient='',
                                                       raw_factorial_effect='', oriented_regression_coefficient='',
                                                       oriented_factorial_effect=''))
                else:
                    sample_effect_rows.append(dict(sample_idx=si, metric=metric, metric_family=family,
                                                       direction=direction, effect_id=eid, effect_order=order,
                                                       factors=factors_str,
                                                       raw_regression_coefficient=raw_coef[si],
                                                       raw_factorial_effect=raw_eff[si],
                                                       oriented_regression_coefficient=or_coef[si],
                                                       oriented_factorial_effect=or_eff[si]))

            summ = summarize_oriented_series(raw_eff, or_eff)

            # Linearity self-check: mean of the per-sample raw factorial
            # effects must equal the same effect computed directly from the
            # cell MEANS (method means), within LINEARITY_TOLERANCE.
            if summ['n_valid'] > 0:
                cell_means = np.array([np.nanmean(Y_raw[ci, :]) for ci in range(fr.n_cells)])
                beta_from_means = float((fr.coded_matrix[:, subset_idx] @ cell_means) / fr.n_cells)
                effect_from_means = 2.0 * beta_from_means
                lin_diff = abs(summ['mean_raw'] - effect_from_means)
                if lin_diff > LINEARITY_TOLERANCE:
                    raise SystemExit(
                        f'[hard-fail] Design {design_id!r} linearity check failed for metric={metric!r} '
                        f'effect={eid!r}: mean-of-sample-effects={summ["mean_raw"]!r} vs '
                        f'effect-from-method-means={effect_from_means!r} (abs_diff={lin_diff:.3e} > '
                        f'{LINEARITY_TOLERANCE:g}).'
                    )

            # Sanity-check targets (PD/MT only), tolerance SANITY_TOLERANCE.
            if metric == 'pd_distance' and eid in sanity_pd:
                target = sanity_pd[eid]
                observed = summ['mean_oriented']
                if observed == '' or abs(observed - target) > SANITY_TOLERANCE:
                    raise SystemExit(
                        f'[hard-fail] Design {design_id!r} PD sanity check failed for effect={eid!r}: '
                        f'observed mean_oriented_factorial_effect={observed!r}, expected {target} '
                        f'(tolerance {SANITY_TOLERANCE:g}).'
                    )
            if metric == 'mt_distance' and eid in sanity_mt:
                target = sanity_mt[eid]
                observed = summ['mean_oriented']
                if observed == '' or abs(observed - target) > SANITY_TOLERANCE:
                    raise SystemExit(
                        f'[hard-fail] Design {design_id!r} MT sanity check failed for effect={eid!r}: '
                        f'observed mean_oriented_factorial_effect={observed!r}, expected {target} '
                        f'(tolerance {SANITY_TOLERANCE:g}).'
                    )

            ci = summ['ci']
            effect_summary_rows.append(dict(
                metric=metric, metric_family=family, direction=direction, effect_id=eid, effect_order=order,
                factors=factors_str, n_valid=summ['n_valid'],
                mean_raw_factorial_effect=summ['mean_raw'], median_raw_factorial_effect=summ['median_raw'],
                mean_oriented_factorial_effect=summ['mean_oriented'],
                median_oriented_factorial_effect=summ['median_oriented'],
                oriented_effect_standard_deviation=summ['std'], oriented_effect_standard_error=summ['se'],
                iid_bootstrap_ci95_low=ci['iid'][0], iid_bootstrap_ci95_high=ci['iid'][1],
                block6_bootstrap_ci95_low=ci['block6'][0], block6_bootstrap_ci95_high=ci['block6'][1],
                block12_bootstrap_ci95_low=ci['block12'][0], block12_bootstrap_ci95_high=ci['block12'][1],
                block24_bootstrap_ci95_low=ci['block24'][0], block24_bootstrap_ci95_high=ci['block24'][1],
                positive_count=summ['pos'], zero_count=summ['zero'], negative_count=summ['neg'],
                positive_rate=summ['pos_rate'], zero_rate=summ['zero_rate'], negative_rate=summ['neg_rate'],
                paired_effect_size_dz=summ['dz'], sign_test_p_raw=summ['sign_p'],
                wilcoxon_p_raw=summ['wilcoxon_p'], test_status=summ['test_status'],
            ))

    return dict(design_id=design_id, factor_names=factor_names, method_order=method_order,
                 design_rows=design_rows, cell_summary_rows=cell_summary_rows,
                 sample_effect_rows=sample_effect_rows, effect_summary_rows=effect_summary_rows,
                 reconstruction_rows=reconstruction_rows)


# =============================================================================
# Analysis D: targeted matched contrasts
# =============================================================================

TARGETED_CONTRASTS = [
    # --- A. Adding the critical-maxima proxy (all single-term: only uses_crit differs) ---
    dict(contrast_id='add_crit_to_uv', contrast_family='add_critical_proxy',
          base_method='uv', comparison_method='uv_crit', added_or_changed_term='crit',
          single_term_contrast=True,
          interpretation='Effect of adding the critical-maxima proxy (L_crit) to the vector-only UV control.'),
    dict(contrast_id='add_crit_to_b', contrast_family='add_critical_proxy',
          base_method='candidate_b', comparison_method='candidate_c', added_or_changed_term='crit',
          single_term_contrast=True,
          interpretation='Effect of adding the critical-maxima proxy to the full Candidate B scaffold '
                          '(the original Candidate B -> Candidate C step).'),
    dict(contrast_id='add_crit_to_grad', contrast_family='add_critical_proxy',
          base_method='grad_only', comparison_method='f3_grad_crit', added_or_changed_term='crit',
          single_term_contrast=True,
          interpretation='Effect of adding the critical-maxima proxy to the gradient-only objective.'),
    dict(contrast_id='add_crit_to_b_e2', contrast_family='add_critical_proxy',
          base_method='b_e2', comparison_method='c_e2', added_or_changed_term='crit',
          single_term_contrast=True,
          interpretation='Effect of adding the critical-maxima proxy on top of the B+repaired-E2 objective.'),

    # --- B. Adding repaired E2 (all single-term: only uses_e2 differs) ---
    dict(contrast_id='add_e2_to_uv', contrast_family='add_repaired_e2',
          base_method='uv', comparison_method='uv_e2', added_or_changed_term='e2', single_term_contrast=True,
          interpretation='Effect of adding repaired low-lambda E2 to the vector-only UV control.'),
    dict(contrast_id='add_e2_to_b', contrast_family='add_repaired_e2',
          base_method='candidate_b', comparison_method='b_e2', added_or_changed_term='e2',
          single_term_contrast=True,
          interpretation='Effect of adding repaired E2 to the full Candidate B scaffold.'),
    dict(contrast_id='add_e2_to_c', contrast_family='add_repaired_e2',
          base_method='candidate_c', comparison_method='c_e2', added_or_changed_term='e2',
          single_term_contrast=True,
          interpretation='Effect of adding repaired E2 to Candidate C (B scaffold + critical proxy).'),
    dict(contrast_id='add_e2_to_grad', contrast_family='add_repaired_e2',
          base_method='grad_only', comparison_method='f1_grad_e2', added_or_changed_term='e2',
          single_term_contrast=True,
          interpretation='Effect of adding repaired E2 to the gradient-only objective.'),
    dict(contrast_id='add_e2_to_grad_levelset', contrast_family='add_repaired_e2',
          base_method='grad_levelset', comparison_method='f2_grad_levelset_e2', added_or_changed_term='e2',
          single_term_contrast=True,
          interpretation='Effect of adding repaired E2 to the gradient+level-set objective.'),

    # --- C. E2 versus critical proxy on a matched scaffold (both crit and e2
    #     flip simultaneously -- a substitution, not a single-term addition) ---
    dict(contrast_id='e2_vs_crit_uv', contrast_family='e2_vs_crit_matched_scaffold',
          base_method='uv_crit', comparison_method='uv_e2', added_or_changed_term='crit_replaced_by_e2',
          single_term_contrast=False,
          interpretation='Substitution contrast on the UV scaffold: replaces the critical-maxima proxy with '
                          'repaired E2 (both uses_crit and uses_e2 flip simultaneously; not a single-term effect).'),
    dict(contrast_id='e2_vs_crit_b', contrast_family='e2_vs_crit_matched_scaffold',
          base_method='candidate_c', comparison_method='b_e2', added_or_changed_term='crit_replaced_by_e2',
          single_term_contrast=False,
          interpretation='Substitution contrast on the full B scaffold: replaces the critical-maxima proxy '
                          'with repaired E2 (both uses_crit and uses_e2 flip simultaneously).'),
    dict(contrast_id='e2_vs_crit_grad', contrast_family='e2_vs_crit_matched_scaffold',
          base_method='f3_grad_crit', comparison_method='f1_grad_e2', added_or_changed_term='crit_replaced_by_e2',
          single_term_contrast=False,
          interpretation='Substitution contrast on the gradient-only scaffold: replaces the critical-maxima '
                          'proxy with repaired E2 (both uses_crit and uses_e2 flip simultaneously).'),

    # --- D. Minimality / scaffold-pruning contrasts (composite; labeled as
    #     such regardless of how many flags literally differ) ---
    dict(contrast_id='remove_speed_from_b_e2', contrast_family='scaffold_pruning',
          base_method='b_e2', comparison_method='f2_grad_levelset_e2', added_or_changed_term='speed',
          single_term_contrast=False,
          interpretation='Composite scaffold-pruning contrast (not a single-term causal effect, per the '
                          'requested labeling for this family): removes the speed-loss term from B+E2 while '
                          'keeping grad+levelset+E2. (Flag-level note: this particular pair differs only in '
                          'uses_speed, but is still reported under the composite scaffold_pruning family.)'),
    dict(contrast_id='remove_speed_levelset_from_b_e2', contrast_family='scaffold_pruning',
          base_method='b_e2', comparison_method='f1_grad_e2', added_or_changed_term='speed+levelset',
          single_term_contrast=False,
          interpretation='Composite scaffold-pruning contrast: removes both the speed- and level-set-loss '
                          'terms from B+E2, keeping only grad+E2. Differs from base in two flags '
                          '(uses_speed and uses_levelset) -- not a single-term effect.'),
    dict(contrast_id='remove_speed_levelset_from_c', contrast_family='scaffold_pruning',
          base_method='candidate_c', comparison_method='f3_grad_crit', added_or_changed_term='speed+levelset',
          single_term_contrast=False,
          interpretation='Composite scaffold-pruning contrast: removes both the speed- and level-set-loss '
                          'terms from Candidate C, keeping only grad+crit. Differs from base in two flags '
                          '(uses_speed and uses_levelset) -- not a single-term effect.'),
]


def run_targeted_contrasts(metric_cols, metric_direction, metric_family, per_sample, method_meta):
    per_sample_rows = []
    summary_rows = []
    n_valid_tests = 0

    for spec in TARGETED_CONTRASTS:
        base, comp = spec['base_method'], spec['comparison_method']
        for method_id in (base, comp):
            if method_id not in per_sample:
                raise SystemExit(f"[hard-fail] Targeted contrast {spec['contrast_id']!r}: method "
                                  f'{method_id!r} not found in the long table.')
        for metric in metric_cols:
            direction = metric_direction[metric]
            family = metric_family.get(metric, '')
            base_vals = {si: per_sample[base][si].get(metric, float('nan')) for si in range(N_EVAL)
                          if math.isfinite(per_sample[base][si].get(metric, float('nan')))}
            comp_vals = {si: per_sample[comp][si].get(metric, float('nan')) for si in range(N_EVAL)
                          if math.isfinite(per_sample[comp][si].get(metric, float('nan')))}
            common = sorted(set(base_vals) & set(comp_vals))
            n_pairs = len(common)

            for si in range(N_EVAL):
                if si in common:
                    bv, cv = base_vals[si], comp_vals[si]
                    raw_delta = cv - bv
                    oriented_improvement = raw_delta if direction == 'higher_is_better' else -raw_delta
                    per_sample_rows.append(dict(
                        contrast_id=spec['contrast_id'], contrast_family=spec['contrast_family'],
                        base_method=base, comparison_method=comp, sample_idx=si, metric=metric,
                        metric_family=family, direction=direction, base_value=bv, comparison_value=cv,
                        raw_delta=raw_delta, oriented_improvement=oriented_improvement))
                else:
                    per_sample_rows.append(dict(
                        contrast_id=spec['contrast_id'], contrast_family=spec['contrast_family'],
                        base_method=base, comparison_method=comp, sample_idx=si, metric=metric,
                        metric_family=family, direction=direction, base_value='', comparison_value='',
                        raw_delta='', oriented_improvement=''))

            if n_pairs == 0:
                summary_rows.append(dict(
                    contrast_id=spec['contrast_id'], contrast_family=spec['contrast_family'],
                    base_method=base, comparison_method=comp, metric=metric, metric_family=family,
                    direction=direction, n_valid_pairs=0, base_mean='', comparison_mean='', mean_raw_delta='',
                    median_raw_delta='', mean_oriented_improvement='', median_oriented_improvement='',
                    oriented_standard_deviation='', oriented_standard_error='',
                    iid_bootstrap_ci95_low='', iid_bootstrap_ci95_high='',
                    block6_bootstrap_ci95_low='', block6_bootstrap_ci95_high='',
                    block12_bootstrap_ci95_low='', block12_bootstrap_ci95_high='',
                    block24_bootstrap_ci95_low='', block24_bootstrap_ci95_high='',
                    win_count=0, tie_count=0, loss_count=0, win_rate='', tie_rate='', loss_rate='',
                    paired_effect_size_dz='', sign_test_p_raw='', wilcoxon_p_raw='',
                    test_status='no_valid_pairs'))
                continue

            n_valid_tests += 1
            base_arr = np.array([base_vals[si] for si in common], dtype=np.float64)
            comp_arr = np.array([comp_vals[si] for si in common], dtype=np.float64)
            raw_delta_arr = comp_arr - base_arr
            oriented_arr = raw_delta_arr if direction == 'higher_is_better' else -raw_delta_arr

            summ = summarize_oriented_series(raw_delta_arr, oriented_arr)

            # Linearity check: mean raw delta must equal comparison-method
            # mean minus base-method mean, within LINEARITY_TOLERANCE.
            comp_full_mean = float(np.nanmean([per_sample[comp][si].get(metric, float('nan'))
                                                  for si in range(N_EVAL)]))
            base_full_mean = float(np.nanmean([per_sample[base][si].get(metric, float('nan'))
                                                  for si in range(N_EVAL)]))
            expected_mean_delta = comp_full_mean - base_full_mean
            lin_diff = abs(summ['mean_raw'] - expected_mean_delta)
            if lin_diff > LINEARITY_TOLERANCE:
                raise SystemExit(
                    f"[hard-fail] Targeted contrast {spec['contrast_id']!r} metric={metric!r}: mean_raw_delta="
                    f'{summ["mean_raw"]!r} vs (comparison_mean - base_mean)={expected_mean_delta!r} '
                    f'(abs_diff={lin_diff:.3e} > {LINEARITY_TOLERANCE:g}).'
                )

            win = int(np.sum(oriented_arr > TIE_TOLERANCE))
            loss = int(np.sum(oriented_arr < -TIE_TOLERANCE))
            tie = n_pairs - win - loss
            ci = summ['ci']
            summary_rows.append(dict(
                contrast_id=spec['contrast_id'], contrast_family=spec['contrast_family'],
                base_method=base, comparison_method=comp, metric=metric, metric_family=family,
                direction=direction, n_valid_pairs=n_pairs,
                base_mean=float(base_arr.mean()), comparison_mean=float(comp_arr.mean()),
                mean_raw_delta=summ['mean_raw'], median_raw_delta=summ['median_raw'],
                mean_oriented_improvement=summ['mean_oriented'],
                median_oriented_improvement=summ['median_oriented'],
                oriented_standard_deviation=summ['std'], oriented_standard_error=summ['se'],
                iid_bootstrap_ci95_low=ci['iid'][0], iid_bootstrap_ci95_high=ci['iid'][1],
                block6_bootstrap_ci95_low=ci['block6'][0], block6_bootstrap_ci95_high=ci['block6'][1],
                block12_bootstrap_ci95_low=ci['block12'][0], block12_bootstrap_ci95_high=ci['block12'][1],
                block24_bootstrap_ci95_low=ci['block24'][0], block24_bootstrap_ci95_high=ci['block24'][1],
                win_count=win, tie_count=tie, loss_count=loss,
                win_rate=win / n_pairs, tie_rate=tie / n_pairs, loss_rate=loss / n_pairs,
                paired_effect_size_dz=summ['dz'], sign_test_p_raw=summ['sign_p'],
                wilcoxon_p_raw=summ['wilcoxon_p'], test_status=summ['test_status'],
            ))

    return dict(per_sample_rows=per_sample_rows, summary_rows=summary_rows, n_valid_tests=n_valid_tests)


# =============================================================================
# Combined test table, multiple-testing correction
# =============================================================================

def build_combined_test_rows(design_results: dict, contrasts_result: dict) -> list:
    """One row per (analysis_family, design_or_contrast_id, metric) across all
    four analyses, carrying sign_test_p_raw / wilcoxon_p_raw (possibly empty)."""
    rows = []
    for analysis_family, res in design_results.items():
        for r in res['effect_summary_rows']:
            rows.append(dict(analysis_family=analysis_family, design_or_contrast_id=r['effect_id'],
                               metric=r['metric'], sign_test_p_raw=r['sign_test_p_raw'],
                               wilcoxon_p_raw=r['wilcoxon_p_raw']))
    for r in contrasts_result['summary_rows']:
        rows.append(dict(analysis_family='TARGETED_CONTRAST', design_or_contrast_id=r['contrast_id'],
                           metric=r['metric'], sign_test_p_raw=r['sign_test_p_raw'],
                           wilcoxon_p_raw=r['wilcoxon_p_raw']))
    return rows


def write_multiple_testing_adjusted(combined_rows: list, out_path: Path) -> None:
    def key_of(r):
        return (r['analysis_family'], r['design_or_contrast_id'], r['metric'])

    sign_global = holm_correction([(key_of(r), r['sign_test_p_raw']) for r in combined_rows
                                     if r['sign_test_p_raw'] not in ('', None)])
    wil_global = holm_correction([(key_of(r), r['wilcoxon_p_raw']) for r in combined_rows
                                    if r['wilcoxon_p_raw'] not in ('', None)])

    sign_within_metric = {}
    wil_within_metric = {}
    metrics = sorted(set(r['metric'] for r in combined_rows))
    for m in metrics:
        subset = [r for r in combined_rows if r['metric'] == m]
        sign_within_metric.update(holm_correction([(key_of(r), r['sign_test_p_raw']) for r in subset
                                                      if r['sign_test_p_raw'] not in ('', None)]))
        wil_within_metric.update(holm_correction([(key_of(r), r['wilcoxon_p_raw']) for r in subset
                                                     if r['wilcoxon_p_raw'] not in ('', None)]))

    sign_within_family = {}
    wil_within_family = {}
    families = sorted(set(r['analysis_family'] for r in combined_rows))
    for fam in families:
        subset = [r for r in combined_rows if r['analysis_family'] == fam]
        sign_within_family.update(holm_correction([(key_of(r), r['sign_test_p_raw']) for r in subset
                                                      if r['sign_test_p_raw'] not in ('', None)]))
        wil_within_family.update(holm_correction([(key_of(r), r['wilcoxon_p_raw']) for r in subset
                                                     if r['wilcoxon_p_raw'] not in ('', None)]))

    fieldnames = ['analysis_family', 'design_or_contrast_id', 'metric', 'sign_test_p_raw',
                   'sign_test_p_holm_global', 'sign_test_p_holm_within_metric',
                   'sign_test_p_holm_within_analysis_family', 'wilcoxon_p_raw', 'wilcoxon_p_holm_global',
                   'wilcoxon_p_holm_within_metric', 'wilcoxon_p_holm_within_analysis_family']
    with out_path.open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in combined_rows:
            k = key_of(r)
            w.writerow(dict(
                analysis_family=r['analysis_family'], design_or_contrast_id=r['design_or_contrast_id'],
                metric=r['metric'], sign_test_p_raw=r['sign_test_p_raw'],
                sign_test_p_holm_global=sign_global.get(k, ''),
                sign_test_p_holm_within_metric=sign_within_metric.get(k, ''),
                sign_test_p_holm_within_analysis_family=sign_within_family.get(k, ''),
                wilcoxon_p_raw=r['wilcoxon_p_raw'], wilcoxon_p_holm_global=wil_global.get(k, ''),
                wilcoxon_p_holm_within_metric=wil_within_metric.get(k, ''),
                wilcoxon_p_holm_within_analysis_family=wil_within_family.get(k, ''),
            ))


# =============================================================================
# Matrix outputs
# =============================================================================

def write_effect_matrix(effect_summary_rows: list, metric_cols: list, id_field: str, value_field: str,
                          out_path: Path) -> None:
    ids = []
    seen = set()
    for r in effect_summary_rows:
        if r[id_field] not in seen:
            seen.add(r[id_field])
            ids.append(r[id_field])
    lookup = {(r[id_field], r['metric']): r[value_field] for r in effect_summary_rows}
    with out_path.open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=[id_field] + metric_cols)
        w.writeheader()
        for eid in ids:
            row = {id_field: eid}
            for m in metric_cols:
                row[m] = lookup.get((eid, m), '')
            w.writerow(row)


# =============================================================================
# Topology-focused summary
# =============================================================================

ANALYSIS_FAMILY_ORDER = ['B_FACTORIAL_2x3', 'B_SCAFFOLD_CRIT_E2_2x2', 'GRAD_SCAFFOLD_LEVELSET_E2_2x2',
                           'TARGETED_CONTRAST']


def build_topology_summary(design_results: dict, contrasts_result: dict) -> list:
    rows = []
    effect_order_lookup = {}
    for analysis_family, res in design_results.items():
        for r in res['effect_summary_rows']:
            if r['metric'] not in ('pd_distance', 'mt_distance'):
                continue
            scope = ('Factorial contrast among the realized trained models in this design on the fixed '
                      '168-sample benchmark; not a universal causal claim about the loss term(s) involved.')
            rows.append(dict(
                analysis_family=analysis_family, design_or_contrast_id=r['effect_id'],
                effect_or_contrast_label=r['effect_id'], metric=r['metric'],
                mean_raw_effect=r['mean_raw_factorial_effect'], mean_oriented_effect=r['mean_oriented_factorial_effect'],
                median_oriented_effect=r['median_oriented_factorial_effect'],
                iid_ci95_low=r['iid_bootstrap_ci95_low'], iid_ci95_high=r['iid_bootstrap_ci95_high'],
                block6_ci95_low=r['block6_bootstrap_ci95_low'], block6_ci95_high=r['block6_bootstrap_ci95_high'],
                block12_ci95_low=r['block12_bootstrap_ci95_low'], block12_ci95_high=r['block12_bootstrap_ci95_high'],
                block24_ci95_low=r['block24_bootstrap_ci95_low'], block24_ci95_high=r['block24_bootstrap_ci95_high'],
                positive_rate=r['positive_rate'], interpretation_scope=scope,
                _family_rank=ANALYSIS_FAMILY_ORDER.index(analysis_family), _order_rank=r['effect_order'],
            ))
    for r in contrasts_result['summary_rows']:
        if r['metric'] not in ('pd_distance', 'mt_distance'):
            continue
        spec = next(s for s in TARGETED_CONTRASTS if s['contrast_id'] == r['contrast_id'])
        term_kind = 'single-term' if spec['single_term_contrast'] else 'composite/substitution'
        scope = (f'Matched-pair contrast ({term_kind}) among the realized trained models on the fixed '
                  '168-sample benchmark; not a universal causal claim.')
        rows.append(dict(
            analysis_family='TARGETED_CONTRAST', design_or_contrast_id=r['contrast_id'],
            effect_or_contrast_label=f"{r['base_method']}->{r['comparison_method']}", metric=r['metric'],
            mean_raw_effect=r['mean_raw_delta'], mean_oriented_effect=r['mean_oriented_improvement'],
            median_oriented_effect=r['median_oriented_improvement'],
            iid_ci95_low=r['iid_bootstrap_ci95_low'], iid_ci95_high=r['iid_bootstrap_ci95_high'],
            block6_ci95_low=r['block6_bootstrap_ci95_low'], block6_ci95_high=r['block6_bootstrap_ci95_high'],
            block12_ci95_low=r['block12_bootstrap_ci95_low'], block12_ci95_high=r['block12_bootstrap_ci95_high'],
            block24_ci95_low=r['block24_bootstrap_ci95_low'], block24_ci95_high=r['block24_bootstrap_ci95_high'],
            positive_rate=(r['win_rate'] if r['win_rate'] != '' else ''), interpretation_scope=scope,
            _family_rank=ANALYSIS_FAMILY_ORDER.index('TARGETED_CONTRAST'),
            _order_rank=TARGETED_CONTRASTS.index(spec),
        ))

    rows.sort(key=lambda r: (r['metric'], r['_family_rank'], r['_order_rank']))
    for r in rows:
        del r['_family_rank']
        del r['_order_rank']
    return rows


# =============================================================================
# Generic CSV writer
# =============================================================================

def write_csv(path: Path, fieldnames: list, rows: list) -> None:
    with path.open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    log(f'[write] {path} ({len(rows)} rows)')


DESIGN_FIELDNAMES = {
    'B_FACTORIAL_2x3': ['design_id', 'method_id', 'display_name', 'speed', 'grad', 'levelset',
                          'coded_speed', 'coded_grad', 'coded_levelset'],
    'B_SCAFFOLD_CRIT_E2_2x2': ['design_id', 'method_id', 'display_name', 'crit', 'e2', 'coded_crit', 'coded_e2'],
    'GRAD_SCAFFOLD_LEVELSET_E2_2x2': ['design_id', 'method_id', 'display_name', 'levelset', 'e2',
                                        'coded_levelset', 'coded_e2'],
}
CELL_SUMMARY_FIELDNAMES = ['design_id', 'method_id', 'metric', 'metric_family', 'direction', 'n_valid',
                             'mean', 'median', 'standard_deviation', 'standard_error',
                             'iid_bootstrap_ci95_low', 'iid_bootstrap_ci95_high']
SAMPLE_EFFECT_FIELDNAMES = ['sample_idx', 'metric', 'metric_family', 'direction', 'effect_id', 'effect_order',
                              'factors', 'raw_regression_coefficient', 'raw_factorial_effect',
                              'oriented_regression_coefficient', 'oriented_factorial_effect']
EFFECT_SUMMARY_FIELDNAMES = ['metric', 'metric_family', 'direction', 'effect_id', 'effect_order', 'factors',
                               'n_valid', 'mean_raw_factorial_effect', 'median_raw_factorial_effect',
                               'mean_oriented_factorial_effect', 'median_oriented_factorial_effect',
                               'oriented_effect_standard_deviation', 'oriented_effect_standard_error',
                               'iid_bootstrap_ci95_low', 'iid_bootstrap_ci95_high',
                               'block6_bootstrap_ci95_low', 'block6_bootstrap_ci95_high',
                               'block12_bootstrap_ci95_low', 'block12_bootstrap_ci95_high',
                               'block24_bootstrap_ci95_low', 'block24_bootstrap_ci95_high',
                               'positive_count', 'zero_count', 'negative_count', 'positive_rate', 'zero_rate',
                               'negative_rate', 'paired_effect_size_dz', 'sign_test_p_raw', 'wilcoxon_p_raw',
                               'test_status']
RECONSTRUCTION_FIELDNAMES = ['sample_idx', 'metric', 'method_id', 'observed_value', 'reconstructed_value',
                                'absolute_difference', 'status']


def write_design_outputs(prefix: str, res: dict, write_cell_summary: bool) -> None:
    write_csv(OUT_DIR / f'{prefix}_design.csv', DESIGN_FIELDNAMES[res['design_id']], res['design_rows'])
    if write_cell_summary:
        write_csv(OUT_DIR / f'{prefix}_cell_summary.csv', CELL_SUMMARY_FIELDNAMES, res['cell_summary_rows'])
    write_csv(OUT_DIR / f'{prefix}_sample_effects.csv', SAMPLE_EFFECT_FIELDNAMES, res['sample_effect_rows'])
    write_csv(OUT_DIR / f'{prefix}_effect_summary.csv', EFFECT_SUMMARY_FIELDNAMES, res['effect_summary_rows'])
    write_csv(OUT_DIR / f'{prefix}_reconstruction_check.csv', RECONSTRUCTION_FIELDNAMES, res['reconstruction_rows'])


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log('=' * 88)
    log('Unified candidate analysis -- Phase 2B')
    log(f'Repo root: {REPO_ROOT}')
    log(f'SciPy available: {HAVE_SCIPY}')
    log('Read-only w.r.t. Phase-1 and Phase-2A artifacts. No training/inference/eval/TTK performed.')
    log('=' * 88)

    require_protected_files()
    file_to_phase = {p.resolve().relative_to(REPO_ROOT).as_posix(): 'phase1' for p in PHASE1_PROTECTED_FILES}
    file_to_phase.update({p.resolve().relative_to(REPO_ROOT).as_posix(): 'phase2a' for p in PHASE2A_PROTECTED_FILES})
    all_protected = PHASE1_PROTECTED_FILES + PHASE2A_PROTECTED_FILES
    checksums_before = checksum_all(all_protected)
    log(f'[immutability] Checksummed {len(checksums_before)} prior-phase file(s) before the run '
        f'(12 Phase-1 + 14 Phase-2A = 26 exactly).')

    long_table = load_long_table()
    metric_direction, metric_family = load_column_mapping()
    metric_cols = long_table['metric_cols']
    phase1_refs = load_phase1_refs()
    phase2a_refs = load_phase2a_refs()
    log(f'[load] Long table: {long_table["n_rows"]} rows, {len(metric_cols)} metrics, '
        f'{len(long_table["per_sample"])} methods. Phase-2A validation={len(phase2a_refs["validation"])} rows, '
        f'pairwise_repro={len(phase2a_refs["pairwise_repro"])} rows.')

    methods_used_in_designs = set()
    for cells in (DESIGN_A_CELLS, DESIGN_B_CELLS, DESIGN_C_CELLS):
        methods_used_in_designs.update(c[0] for c in cells)
    for spec in TARGETED_CONTRASTS:
        methods_used_in_designs.add(spec['base_method'])
        methods_used_in_designs.add(spec['comparison_method'])

    # -------------------------------------------------------------------
    # Base validation
    # -------------------------------------------------------------------
    validation_rows, failures = run_base_validation(long_table, phase1_refs, phase2a_refs, metric_direction,
                                                       methods_used_in_designs)
    if failures:
        log('')
        log('[VALIDATION FAILURE]')
        for f in failures:
            log(f'  - {f}')
        flush_log()
        raise SystemExit(f'[hard-fail] {len(failures)} Phase-2B base validation check(s) failed.')
    log(f'[validate] All {len(validation_rows)} base validation checks PASSED.')
    write_csv(OUT_DIR / 'phase2b_validation.csv',
               ['check_name', 'observed', 'expected', 'tolerance', 'status', 'notes'], validation_rows)

    per_sample = long_table['per_sample']
    method_meta = long_table['method_meta']

    # -------------------------------------------------------------------
    # Analysis A/B/C: complete factorial designs
    # -------------------------------------------------------------------
    res_a = run_factorial_design('B_FACTORIAL_2x3', DESIGN_A_FACTORS, DESIGN_A_CELLS, metric_cols,
                                    metric_direction, metric_family, per_sample, method_meta,
                                    DESIGN_A_SANITY_PD, DESIGN_A_SANITY_MT)
    write_design_outputs('b_factorial', res_a, write_cell_summary=True)
    log(f"[analysis-A] B 2^3 factorial: {len(res_a['design_rows'])} cells, "
        f"{len(res_a['effect_summary_rows'])} effect-summary rows, all sanity checks + reconstruction PASSED.")

    res_b = run_factorial_design('B_SCAFFOLD_CRIT_E2_2x2', DESIGN_B_FACTORS, DESIGN_B_CELLS, metric_cols,
                                    metric_direction, metric_family, per_sample, method_meta,
                                    DESIGN_B_SANITY_PD, DESIGN_B_SANITY_MT)
    write_design_outputs('b_scaffold_crit_e2', res_b, write_cell_summary=False)
    log(f"[analysis-B] B-scaffold crit x E2 2^2: {len(res_b['design_rows'])} cells, sanity + reconstruction PASSED.")

    res_c = run_factorial_design('GRAD_SCAFFOLD_LEVELSET_E2_2x2', DESIGN_C_FACTORS, DESIGN_C_CELLS, metric_cols,
                                    metric_direction, metric_family, per_sample, method_meta,
                                    DESIGN_C_SANITY_PD, DESIGN_C_SANITY_MT)
    write_design_outputs('grad_scaffold_levelset_e2', res_c, write_cell_summary=False)
    log(f"[analysis-C] Grad-scaffold levelset x E2 2^2: {len(res_c['design_rows'])} cells, "
        f"sanity + reconstruction PASSED.")

    design_results = {'B_FACTORIAL_2x3': res_a, 'B_SCAFFOLD_CRIT_E2_2x2': res_b,
                        'GRAD_SCAFFOLD_LEVELSET_E2_2x2': res_c}

    # -------------------------------------------------------------------
    # Analysis D: targeted matched contrasts
    # -------------------------------------------------------------------
    manifest_rows = [dict(contrast_id=s['contrast_id'], contrast_family=s['contrast_family'],
                             base_method=s['base_method'], comparison_method=s['comparison_method'],
                             added_or_changed_term=s['added_or_changed_term'],
                             single_term_contrast=s['single_term_contrast'], interpretation=s['interpretation'])
                       for s in TARGETED_CONTRASTS]
    write_csv(OUT_DIR / 'targeted_contrast_manifest.csv',
               ['contrast_id', 'contrast_family', 'base_method', 'comparison_method', 'added_or_changed_term',
                'single_term_contrast', 'interpretation'], manifest_rows)

    contrasts_result = run_targeted_contrasts(metric_cols, metric_direction, metric_family, per_sample, method_meta)
    write_csv(OUT_DIR / 'targeted_contrast_per_sample.csv',
               ['contrast_id', 'contrast_family', 'base_method', 'comparison_method', 'sample_idx', 'metric',
                'metric_family', 'direction', 'base_value', 'comparison_value', 'raw_delta',
                'oriented_improvement'], contrasts_result['per_sample_rows'])
    write_csv(OUT_DIR / 'targeted_contrast_summary.csv',
               ['contrast_id', 'contrast_family', 'base_method', 'comparison_method', 'metric', 'metric_family',
                'direction', 'n_valid_pairs', 'base_mean', 'comparison_mean', 'mean_raw_delta',
                'median_raw_delta', 'mean_oriented_improvement', 'median_oriented_improvement',
                'oriented_standard_deviation', 'oriented_standard_error', 'iid_bootstrap_ci95_low',
                'iid_bootstrap_ci95_high', 'block6_bootstrap_ci95_low', 'block6_bootstrap_ci95_high',
                'block12_bootstrap_ci95_low', 'block12_bootstrap_ci95_high', 'block24_bootstrap_ci95_low',
                'block24_bootstrap_ci95_high', 'win_count', 'tie_count', 'loss_count', 'win_rate', 'tie_rate',
                'loss_rate', 'paired_effect_size_dz', 'sign_test_p_raw', 'wilcoxon_p_raw', 'test_status'],
               contrasts_result['summary_rows'])
    log(f"[analysis-D] {len(TARGETED_CONTRASTS)} targeted contrasts x {len(metric_cols)} metrics, "
        f"{contrasts_result['n_valid_tests']} with >=1 valid pair. Linearity checks PASSED.")

    # -------------------------------------------------------------------
    # Multiple-testing correction
    # -------------------------------------------------------------------
    combined_rows = build_combined_test_rows(design_results, contrasts_result)
    write_multiple_testing_adjusted(combined_rows, OUT_DIR / 'phase2b_multiple_testing_adjusted.csv')
    n_sign_valid = sum(1 for r in combined_rows if r['sign_test_p_raw'] not in ('', None))
    log(f'[write] {OUT_DIR / "phase2b_multiple_testing_adjusted.csv"} ({len(combined_rows)} rows; '
        f'{n_sign_valid} with a valid sign-test p-value)')

    # -------------------------------------------------------------------
    # Matrix outputs
    # -------------------------------------------------------------------
    write_effect_matrix(res_a['effect_summary_rows'], metric_cols, 'effect_id', 'mean_oriented_factorial_effect',
                          OUT_DIR / 'b_factorial_oriented_effect_matrix.csv')
    write_effect_matrix(res_b['effect_summary_rows'], metric_cols, 'effect_id', 'mean_oriented_factorial_effect',
                          OUT_DIR / 'b_scaffold_crit_e2_oriented_effect_matrix.csv')
    write_effect_matrix(res_c['effect_summary_rows'], metric_cols, 'effect_id', 'mean_oriented_factorial_effect',
                          OUT_DIR / 'grad_scaffold_levelset_e2_oriented_effect_matrix.csv')
    write_effect_matrix(contrasts_result['summary_rows'], metric_cols, 'contrast_id',
                          'mean_oriented_improvement', OUT_DIR / 'targeted_contrast_oriented_effect_matrix.csv')
    log('[write] 4 oriented-effect matrices (rows=effects/contrasts, columns=22 metrics, SSIM empty).')

    # -------------------------------------------------------------------
    # Topology-focused summary
    # -------------------------------------------------------------------
    topo_rows = build_topology_summary(design_results, contrasts_result)
    write_csv(OUT_DIR / 'topology_factorial_and_contrast_summary.csv',
               ['analysis_family', 'design_or_contrast_id', 'effect_or_contrast_label', 'metric',
                'mean_raw_effect', 'mean_oriented_effect', 'median_oriented_effect', 'iid_ci95_low',
                'iid_ci95_high', 'block6_ci95_low', 'block6_ci95_high', 'block12_ci95_low', 'block12_ci95_high',
                'block24_ci95_low', 'block24_ci95_high', 'positive_rate', 'interpretation_scope'], topo_rows)

    # -------------------------------------------------------------------
    # Documentation
    # -------------------------------------------------------------------
    write_phase2b_doc(design_results, contrasts_result, validation_rows, combined_rows, topo_rows)

    # -------------------------------------------------------------------
    # Prior-phase immutability postflight
    # -------------------------------------------------------------------
    checksums_after = checksum_all(all_protected)
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
    log(f'RESULT: Phase 2B complete. Designs: A(2^3)={len(res_a["design_rows"])} cells, '
        f'B(2^2)={len(res_b["design_rows"])} cells, C(2^2)={len(res_c["design_rows"])} cells, '
        f'{len(TARGETED_CONTRASTS)} targeted contrasts. All sanity checks, reconstructions, and linearity '
        f'checks passed. Prior-phase files unchanged.')
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


def write_phase2b_doc(design_results, contrasts_result, validation_rows, combined_rows, topo_rows):
    res_a = design_results['B_FACTORIAL_2x3']
    res_b = design_results['B_SCAFFOLD_CRIT_E2_2x2']
    res_c = design_results['GRAD_SCAFFOLD_LEVELSET_E2_2x2']

    def effect_row(res, effect_id, metric):
        return next(r for r in res['effect_summary_rows'] if r['effect_id'] == effect_id and r['metric'] == metric)

    def block_consistency(row):
        """True if the IID and all three block-bootstrap CIs agree on sign
        of the mean oriented effect (all exclude 0 on the same side, or all
        include 0) -- a simple, documented consistency heuristic, not a
        formal test."""
        cis = [(row['iid_ci95_low'], row['iid_ci95_high']),
                (row['block6_ci95_low'], row['block6_ci95_high']),
                (row['block12_ci95_low'], row['block12_ci95_high']),
                (row['block24_ci95_low'], row['block24_ci95_high'])]
        if any(lo == '' or hi == '' for lo, hi in cis):
            return None
        signs = []
        for lo, hi in cis:
            if lo > 0:
                signs.append('positive')
            elif hi < 0:
                signs.append('negative')
            else:
                signs.append('includes_zero')
        return len(set(signs)) == 1

    pd_grad_a = effect_row(res_a, 'grad', 'pd_distance')
    mt_e2_b = effect_row(res_b, 'e2', 'mt_distance')

    n_valid_sign = sum(1 for r in combined_rows if r['sign_test_p_raw'] not in ('', None))
    n_valid_wilcoxon = sum(1 for r in combined_rows if r['wilcoxon_p_raw'] not in ('', None))

    lines = []
    lines.append('# Unified candidate analysis -- Phase 2B report')
    lines.append('')
    lines.append('## 1. Scope and frozen inputs')
    lines.append('')
    lines.append('Phase 2B performs controlled factorial and targeted matched-pair analyses on top of the '
                 'now-immutable Phase-1 (`ttk_runs_fixed/unified_candidate_evaluation/`) and Phase-2A '
                 '(`ttk_runs_fixed/unified_candidate_analysis/phase2a/`) outputs. `unified_primary_per_sample_'
                 'long.csv` remains the numeric source of truth throughout; every Phase-2A file is a validation '
                 'reference only. 26 prior-phase files (12 Phase-1 + 14 Phase-2A) were required to exist, '
                 'checksummed before and after this run, and confirmed byte-for-byte unchanged -- see '
                 '`prior_phase_immutability_check.csv`. No training, inference, cheap evaluation, or TTK was run. '
                 'Metric correlations, Pareto-front analysis, sample selection, and visualization remain '
                 'deferred to Phase 2C/2D.')
    lines.append('')
    lines.append('## 2. Factor coding and exact factorial-effect convention')
    lines.append('')
    lines.append('Factor levels are coded disabled=-1, enabled=+1. For a raw metric value y with direction '
                 'read from `column_mapping.csv` (never inferred from the metric name), the oriented value is '
                 'z=y when higher-is-better and z=-y when lower-is-better, so a positive oriented effect always '
                 'means improvement. For a complete, balanced 2^k design, the saturated coded regression '
                 'coefficient for any nonempty factor subset J is beta_J = mean over the 2^k cells of '
                 '(value * product of coded x_j for j in J) -- exact because the coded design columns are '
                 'mutually orthogonal with squared norm 2^k, so this is exactly the ordinary-least-squares '
                 'solution. The reported `factorial_effect` is `2 * beta_J`: a main effect is the average '
                 'high-minus-low difference, and an interaction is the standard balanced factorial interaction '
                 '(half of the unaveraged two-cell difference-of-differences in a simple 2^2 design). Every '
                 'sample-level effect and every targeted contrast reports both `raw_*` (untransformed) and '
                 '`oriented_*` (direction-applied, positive=improvement) versions side by side; nothing is ever '
                 'reported in only one form.')
    lines.append('')
    lines.append('## 3. Full B 2^3 design')
    lines.append('')
    lines.append('Cells (uv, speed_only, levelset_only, speed_levelset, grad_only, speed_grad, grad_levelset, '
                 'candidate_b) span the complete `speed x grad x levelset` design; method metadata was verified '
                 'to match the declared coding exactly for every cell before any statistic was computed. Seven '
                 'effects were estimated: speed, grad, levelset, speed:grad, speed:levelset, grad:levelset, '
                 'speed:grad:levelset. Every one of the 168 x 22 x 8 = 29,568 cell values was exactly '
                 f'reconstructed from its 8 saturated coefficients (max abs error <= {RECONSTRUCTION_TOLERANCE:g}), '
                 'and every effect\'s task-provided PD/MT sanity-check target was reproduced from the validated '
                 f'method means within {SANITY_TOLERANCE:g}.')
    lines.append('')
    lines.append('## 4. B-scaffold critical x E2 2^2 design')
    lines.append('')
    lines.append('Cells (candidate_b, candidate_c, b_e2, c_e2) hold the speed+grad+levelset scaffold fixed and '
                 'vary only the critical-maxima proxy and repaired E2. Effects estimated: crit, e2, crit:e2. '
                 'Same reconstruction gate, sanity-check reproduction, and bootstrap methodology as Analysis A.')
    lines.append('')
    lines.append('## 5. Gradient-scaffold level-set x E2 2^2 design')
    lines.append('')
    lines.append('Cells (grad_only, grad_levelset, f1_grad_e2, f2_grad_levelset_e2) hold the gradient term fixed '
                 'and vary level-set and repaired E2. Effects estimated: levelset, e2, levelset:e2. Same gates '
                 'and methodology as Analyses A and B.')
    lines.append('')
    lines.append('## 6. Targeted matched contrasts')
    lines.append('')
    lines.append(f'{len(TARGETED_CONTRASTS)} matched-pair contrasts across four families: adding the critical-'
                 'maxima proxy (4, all single-term), adding repaired E2 (5, all single-term), E2 versus the '
                 'critical proxy on a matched scaffold (3, a two-flag substitution, not single-term), and '
                 'minimality/scaffold-pruning (3, explicitly labeled composite regardless of how many flags '
                 'literally differ, per the requested convention for this family). A positive oriented '
                 'improvement always means the comparison method is better than the base method. See '
                 '`targeted_contrast_manifest.csv` for the full base/comparison/term/interpretation table.')
    lines.append('')
    lines.append('## 7. Ordinary and temporal-block bootstrap methodology')
    lines.append('')
    lines.append(f'Every sample-level factorial effect and every targeted contrast reports four 95% confidence '
                 f'intervals of its mean: an ordinary sample-axis (iid) bootstrap, and three deterministic '
                 f'circular moving-block bootstraps with block lengths 6, 12, and 24 hours. All four use '
                 f'{BOOTSTRAP_N:,} resamples with seeds derived from {BOOTSTRAP_SEED} (the block bootstraps use '
                 f'`{BOOTSTRAP_SEED} * 1000 + block_length`); each resample draws block start positions '
                 'uniformly from 0..167 with replacement, appends consecutive circular blocks (wrapping past '
                 'sample 167 back to sample 0), and truncates the concatenated index sequence to exactly 168. '
                 'The 168 benchmark samples are consecutive hourly wind fields and are likely temporally '
                 'correlated; the block-length sensitivity comparison in section 11 is exploratory evidence '
                 'about how much that correlation might matter for a given effect, not proof of a correct '
                 'dependence model or a formally validated block length.')
    lines.append('')
    lines.append('## 8. Multiple-testing correction')
    lines.append('')
    lines.append(f'`phase2b_multiple_testing_adjusted.csv` has {len(combined_rows)} rows (every effect/contrast '
                 f'x metric combination across all four analyses); {n_valid_sign} carry a valid exact sign-test '
                 f'p-value and {n_valid_wilcoxon} carry a valid Wilcoxon p-value '
                 f'({"SciPy was available" if HAVE_SCIPY else "SciPy was not importable in this environment, so every Wilcoxon field is empty and the run completed without it, as required"}). '
                 'Holm step-down correction was applied three ways: once globally across every valid comparison, '
                 'once within each metric across all effects/contrasts, and once within each analysis family. '
                 'No binary "significant" field was created, and the adjusted values retain the temporal-'
                 'independence caveat from section 7 even after correction.')
    lines.append('')
    lines.append('## 9. PD and MT findings')
    lines.append('')
    lines.append(f'Within the realized B 2^3 factorial, enabling gradient has the largest positive mean '
                 f'oriented effect on PD distance ({_fmt(pd_grad_a["mean_oriented_factorial_effect"])}, win rate '
                 f'{_fmt(pd_grad_a["positive_rate"])}); within the realized B-scaffold crit x E2 design, adding '
                 f'repaired E2 has the largest positive mean oriented effect on MT distance '
                 f'({_fmt(mt_e2_b["mean_oriented_factorial_effect"])}, win rate {_fmt(mt_e2_b["positive_rate"])}). '
                 'These are factorial contrasts among the realized trained models in each specific design on '
                 'this fixed benchmark, not general causal statements about the loss terms -- see '
                 '`b_factorial_effect_summary.csv`, `b_scaffold_crit_e2_effect_summary.csv`, and '
                 '`grad_scaffold_levelset_e2_effect_summary.csv` for every effect on every metric, and '
                 '`topology_factorial_and_contrast_summary.csv` for the PD/MT-only extract across all four '
                 'analyses.')
    lines.append('')
    lines.append('## 10. Patterns across metric families')
    lines.append('')
    lines.append('`b_factorial_oriented_effect_matrix.csv`, `b_scaffold_crit_e2_oriented_effect_matrix.csv`, '
                 '`grad_scaffold_levelset_e2_oriented_effect_matrix.csv`, and `targeted_contrast_oriented_'
                 'effect_matrix.csv` give the mean oriented effect for every effect/contrast against every '
                 'metric family (`vector_uv`, `scalar_speed`, `wind_power_distribution`, `gradient_distribution`, '
                 '`frequency_domain`, `threshold_geometry`, `topology_pd`, `topology_mt`; SSIM left empty). No '
                 'weighted aggregate score or total ranking was computed across metrics or methods -- reading '
                 'these matrices column-by-column, alongside `method_descriptive_summary.csv`\'s `metric_family` '
                 'field, is the intended way to see which families move together for a given effect.')
    lines.append('')
    lines.append('## 11. Findings consistent across block lengths')
    lines.append('')
    n_consistent = n_total_checked = 0
    for r in topo_rows:
        c = block_consistency(r)
        if c is not None:
            n_total_checked += 1
            if c:
                n_consistent += 1
    lines.append(f'Of the {n_total_checked} PD/MT effect/contrast rows in `topology_factorial_and_contrast_'
                 f'summary.csv` with all four bootstrap CIs available, {n_consistent} have the iid, block-6, '
                 'block-12, and block-24 confidence intervals agreeing on sign (all excluding zero on the same '
                 'side, or all including zero) -- see that file\'s `*_ci95_low`/`*_ci95_high` columns directly '
                 'for the per-row detail. This is a simple sign-agreement heuristic reported for convenience, '
                 'not a formal test of block-length robustness.')
    lines.append('')
    lines.append('## 12. Small, context-dependent, or sensitive findings')
    lines.append('')
    lines.append('Effects whose mean oriented value is small relative to its bootstrap CI width, or whose sign '
                 'flips between block lengths per section 11, should be described with cautious language '
                 '("slight", "small", "not clearly distinguishable from zero at these block lengths") rather '
                 'than confident directional claims. In particular, near-tied contrasts such as F1 versus F2 '
                 '(`f1_grad_e2` vs `f2_grad_levelset_e2`, not a direct entry in `targeted_contrast_summary.csv` '
                 'but visible by comparing their respective rows against shared base methods) should only be '
                 'described with stronger wording than "slight" if both the paired distribution (win/tie/loss '
                 'counts) and the block-bootstrap intervals in this report actually support it.')
    lines.append('')
    lines.append('## 13. Training-seed and temporal-dependence caveats')
    lines.append('')
    lines.append('Every model in this benchmark was trained exactly once. Every factorial effect and every '
                 'targeted contrast in this report describes a relationship AMONG THE REALIZED TRAINED MODELS on '
                 'this fixed 168-sample benchmark -- it does not establish that a differently-seeded retraining '
                 'of the same objective would reproduce the same effect, and it is not a universal causal claim '
                 'about a loss term in general. Use language like "within this realized 2^3 factorial, enabling '
                 'gradient has the largest positive mean oriented effect on PD distance," never "gradient loss '
                 'universally causes better topology." Separately, because the 168 samples are consecutive '
                 'hourly fields and likely temporally correlated, the ordinary bootstrap, exact sign-test, and '
                 'any Wilcoxon p-values all rely on an independence-across-samples approximation and may be '
                 'anti-conservative; the block-bootstrap comparison in section 11 is exploratory sensitivity '
                 'evidence, not proof of a correct dependence model. Means, medians, raw/oriented deltas, and '
                 'win rates remain valid exact descriptive summaries of this fixed benchmark regardless of '
                 'either caveat.')
    lines.append('')
    lines.append('## 14. Deferred analyses')
    lines.append('')
    lines.append('Metric-correlation analysis, Pareto-front analysis, sample-level selection, and '
                 'visualization generation are explicitly deferred to Phase 2C/2D and were not performed here.')
    lines.append('')
    lines.append('## 15. Generated file list')
    lines.append('')
    for fname in [
        'phase2b_validation.csv',
        'b_factorial_design.csv', 'b_factorial_cell_summary.csv', 'b_factorial_sample_effects.csv',
        'b_factorial_effect_summary.csv', 'b_factorial_reconstruction_check.csv',
        'b_scaffold_crit_e2_design.csv', 'b_scaffold_crit_e2_sample_effects.csv',
        'b_scaffold_crit_e2_effect_summary.csv', 'b_scaffold_crit_e2_reconstruction_check.csv',
        'grad_scaffold_levelset_e2_design.csv', 'grad_scaffold_levelset_e2_sample_effects.csv',
        'grad_scaffold_levelset_e2_effect_summary.csv', 'grad_scaffold_levelset_e2_reconstruction_check.csv',
        'targeted_contrast_manifest.csv', 'targeted_contrast_per_sample.csv', 'targeted_contrast_summary.csv',
        'phase2b_multiple_testing_adjusted.csv',
        'b_factorial_oriented_effect_matrix.csv', 'b_scaffold_crit_e2_oriented_effect_matrix.csv',
        'grad_scaffold_levelset_e2_oriented_effect_matrix.csv', 'targeted_contrast_oriented_effect_matrix.csv',
        'topology_factorial_and_contrast_summary.csv', 'prior_phase_immutability_check.csv',
    ]:
        lines.append(f'- `ttk_runs_fixed/unified_candidate_analysis/phase2b/{fname}`')
    lines.append('- `docs/unified_candidate_analysis_phase2b.md` (this file)')
    lines.append('- `logs/unified_candidate_analysis_phase2b.log`')
    lines.append('')
    (DOCS_DIR / 'unified_candidate_analysis_phase2b.md').write_text('\n'.join(lines) + '\n')
    log(f"[write] {DOCS_DIR / 'unified_candidate_analysis_phase2b.md'}")


if __name__ == '__main__':
    sys.exit(main())
