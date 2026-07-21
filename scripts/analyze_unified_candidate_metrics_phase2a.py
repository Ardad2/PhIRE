#!/usr/bin/env python3
"""Phase 2A: deterministic descriptive and paired multi-metric analysis of
the 19 primary wind-SR candidates, built exclusively from the immutable
Phase-1 unified-evaluation outputs.

This script is read-only with respect to every Phase-1 artifact. It never
runs training, inference, cheap evaluation, or TTK, and it never writes
anywhere except:

    ttk_runs_fixed/unified_candidate_analysis/phase2a/
    docs/unified_candidate_analysis_phase2a.md
    logs/unified_candidate_analysis_phase2a.log

Before and after the run it SHA-256-checksums every Phase-1 CSV plus the
four protected Phase-1 docs/log, and hard-fails if any of them changed.

Scope (Phase 2A only):
    - per-method descriptive statistics for every metric (mean/median/
      spread/bootstrap CI),
    - paired comparison of every method against CNN on the same 168
      samples (raw delta, direction-aware improvement, win/tie/loss,
      exact sign test, optional Wilcoxon, paired bootstrap CI),
    - Holm multiple-testing correction (global and within-metric),
    - a topology (PD/MT) tradeoff summary,
    - an independent reproduction of the Phase-1 pairwise-vs-CNN table.

Explicitly OUT of scope for this script (deferred to later Phase-2 stages):
    factorial decomposition, targeted E2/critical-proxy contrasts, metric
    correlation analysis, Pareto-front analysis, sample selection,
    visualization generation, composite/weighted ranking.

Determinism: every bootstrap resample uses a fixed seed (20260721) and a
precomputed index matrix, so re-running this script produces byte-identical
CSV/Markdown output. No wall-clock time, hostname, or other non-deterministic
value is ever written to a generated file.
"""

from __future__ import annotations

import csv
import hashlib
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
OUT_DIR = REPO_ROOT / 'ttk_runs_fixed' / 'unified_candidate_analysis' / 'phase2a'
DOCS_DIR = REPO_ROOT / 'docs'
LOG_PATH = REPO_ROOT / 'logs' / 'unified_candidate_analysis_phase2a.log'

# Phase-1 files this script must never modify -- checksummed before and
# after the run; any change is a hard failure.
PHASE1_PROTECTED_DOCS = [
    REPO_ROOT / 'docs' / 'unified_candidate_evaluation_phase1.md',
    REPO_ROOT / 'docs' / 'unified_candidate_evaluation_inventory.md',
    REPO_ROOT / 'docs' / 'primary_candidate_artifact_reference.md',
    REPO_ROOT / 'logs' / 'build_unified_candidate_evaluation.log',
]

N_EVAL = 168
N_PRIMARY_METHODS = 19
CNN_METHOD = 'cnn'
TIE_TOLERANCE = 1e-12  # documented paired win/tie/loss tolerance for Phase-2A's own analysis
RECOMPUTE_TOLERANCE = 1e-6  # strict tolerance for reproducing Phase-1 cheap-metric summaries
PD_MT_RECOMPUTE_TOLERANCE = 1e-4  # matches the tolerance Phase-1 itself used for PD/MT validation
BOOTSTRAP_SEED = 20260721
BOOTSTRAP_N = 10000

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


def phase1_protected_files() -> list:
    files = sorted(PHASE1_DIR.glob('*.csv'), key=str)
    files += [p for p in PHASE1_PROTECTED_DOCS if p.exists()]
    return files


def checksum_all(files: list) -> dict:
    return {str(p): sha256_file(p) for p in files if p.exists()}


def read_csv_dicts(path: Path) -> list:
    with path.open(newline='') as fh:
        return list(csv.DictReader(fh))


def _f(val):
    """Parse a CSV cell into float('nan') for empty/missing, else float."""
    if val in (None, ''):
        return float('nan')
    return float(val)


# =============================================================================
# Bootstrap (deterministic, precomputed index matrix for the common n=168 case)
# =============================================================================

_BOOTSTRAP_IDX_168 = np.random.default_rng(BOOTSTRAP_SEED).integers(0, N_EVAL, size=(BOOTSTRAP_N, N_EVAL))
_bootstrap_idx_cache: dict = {}


def _bootstrap_index_matrix(n: int) -> np.ndarray:
    if n == N_EVAL:
        return _BOOTSTRAP_IDX_168
    if n not in _bootstrap_idx_cache:
        _bootstrap_idx_cache[n] = np.random.default_rng(BOOTSTRAP_SEED).integers(0, n, size=(BOOTSTRAP_N, n))
    return _bootstrap_idx_cache[n]


def bootstrap_ci95(values) -> tuple:
    """Deterministic sample-axis bootstrap 95% CI of the mean. Returns
    (ci_low, ci_high) as floats, or (None, None) if values is empty."""
    values = np.asarray(values, dtype=np.float64)
    n = values.shape[0]
    if n == 0:
        return None, None
    idx = _bootstrap_index_matrix(n)
    resampled_means = values[idx].mean(axis=1)
    lo = float(np.percentile(resampled_means, 2.5))
    hi = float(np.percentile(resampled_means, 97.5))
    return lo, hi


# =============================================================================
# Exact two-sided sign test (p=0.5), no SciPy required
# =============================================================================

def exact_sign_test_pvalue(n_pos: int, n_neg: int):
    n = n_pos + n_neg
    if n == 0:
        return None
    k = min(n_pos, n_neg)
    tail = sum(math.comb(n, i) for i in range(0, k + 1))
    p_one_side = tail / (2 ** n)
    return min(1.0, 2 * p_one_side)


# =============================================================================
# Holm step-down correction
# =============================================================================

def holm_correction(items: list) -> dict:
    """items: list of (key, p). Returns {key: holm-adjusted p} for every key
    with a non-None p. Standard Holm step-down: adjusted p_(i) =
    max_{j<=i} (m-j+1) * p_(j), capped at 1.0."""
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
# Descriptive-stat helpers (NumPy only)
# =============================================================================

def describe(values) -> dict:
    values = np.asarray(values, dtype=np.float64)
    n = values.shape[0]
    if n == 0:
        return dict(n_valid=0, mean=None, std=None, se=None, median=None,
                     q25=None, q75=None, minimum=None, maximum=None,
                     ci_low=None, ci_high=None)
    mean = float(values.mean())
    median = float(np.median(values))
    q25 = float(np.percentile(values, 25))
    q75 = float(np.percentile(values, 75))
    minimum = float(values.min())
    maximum = float(values.max())
    if n >= 2:
        std = float(values.std(ddof=1))
        se = std / math.sqrt(n)
    else:
        std = None
        se = None
    ci_low, ci_high = bootstrap_ci95(values)
    return dict(n_valid=n, mean=mean, std=std, se=se, median=median,
                 q25=q25, q75=q75, minimum=minimum, maximum=maximum,
                 ci_low=ci_low, ci_high=ci_high)


# =============================================================================
# Loading Phase-1 inputs
# =============================================================================

def load_long_table():
    path = PHASE1_DIR / 'unified_primary_per_sample_long.csv'
    rows = read_csv_dicts(path)
    metric_cols = None
    with path.open(newline='') as fh:
        header = next(csv.reader(fh))
    identity_cols = ['sample_idx', 'method_id', 'display_name', 'candidate_family',
                      'training_scale', 'architecture', 'uses_speed', 'uses_grad',
                      'uses_levelset', 'uses_crit', 'uses_e2']
    metric_cols = [c for c in header if c not in identity_cols]

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

    return dict(rows=rows, header=header, metric_cols=metric_cols,
                per_sample=per_sample, method_meta=method_meta,
                dup_keys=dup_keys, n_rows=len(rows))


def load_column_mapping():
    path = PHASE1_DIR / 'column_mapping.csv'
    rows = read_csv_dicts(path)
    direction = {}
    family = {}
    for row in rows:
        sc = row['standardized_column']
        d = row['direction']
        rep = row['representation']
        if sc in direction and direction[sc] != d:
            raise SystemExit(f'[hard-fail] column_mapping.csv gives conflicting directions for '
                              f'{sc!r}: {direction[sc]!r} vs {d!r}.')
        direction[sc] = d
        family.setdefault(sc, rep)
    return direction, family


def load_reference_tables():
    method_summary = read_csv_dicts(PHASE1_DIR / 'unified_primary_method_summary.csv')
    pairwise = read_csv_dicts(PHASE1_DIR / 'unified_primary_pairwise_vs_cnn.csv')
    topo_val = read_csv_dicts(PHASE1_DIR / 'unified_primary_topology_validation.csv')
    missingness = read_csv_dicts(PHASE1_DIR / 'unified_primary_missingness.csv')
    inventory = read_csv_dicts(PHASE1_DIR / 'method_inventory.csv')
    return dict(method_summary=method_summary, pairwise=pairwise, topo_val=topo_val,
                missingness=missingness, inventory=inventory)


# =============================================================================
# Data validation
# =============================================================================

def run_validation(long_table, refs, metric_direction):
    checks = []
    failures = []

    def add(name, observed, expected, tol, ok, notes=''):
        status = 'PASS' if ok else 'FAIL'
        checks.append(dict(check_name=name, observed=observed, expected=expected,
                             tolerance=tol, status=status, notes=notes))
        if not ok:
            failures.append(f'{name}: observed={observed!r} expected={expected!r} notes={notes}')

    per_sample = long_table['per_sample']
    metric_cols = long_table['metric_cols']

    add('long_table_row_count', long_table['n_rows'], N_PRIMARY_METHODS * N_EVAL, 0,
        long_table['n_rows'] == N_PRIMARY_METHODS * N_EVAL)
    add('duplicate_keys', len(long_table['dup_keys']), 0, 0, len(long_table['dup_keys']) == 0,
        notes=str(long_table['dup_keys'][:5]))
    add('method_count', len(per_sample), N_PRIMARY_METHODS, 0, len(per_sample) == N_PRIMARY_METHODS)

    for mid in sorted(per_sample):
        n = len(per_sample[mid])
        add(f'samples_per_method[{mid}]', n, N_EVAL, 0, n == N_EVAL)
        idx_set = set(per_sample[mid].keys())
        add(f'sample_idx_exact_0_167[{mid}]', sorted(idx_set)[:1] + sorted(idx_set)[-1:] if idx_set else [],
            [0, N_EVAL - 1], 0, idx_set == set(range(N_EVAL)))
        inconsistent = long_table['method_meta'][mid]['inconsistent_fields']
        add(f'metadata_constant[{mid}]', len(inconsistent), 0, 0, len(inconsistent) == 0,
            notes=str(inconsistent[:5]))

    # SSIM 168/168-or-0/168 per method
    for mid in sorted(per_sample):
        finite = sum(1 for si in per_sample[mid] if math.isfinite(per_sample[mid][si].get('ssim_speed', float('nan'))))
        add(f'ssim_168_or_0[{mid}]', finite, '0 or 168', 0, finite in (0, N_EVAL))

    # PD/MT finite and nonnegative for all topology-bearing methods
    for mid in sorted(per_sample):
        for col in ('pd_distance', 'mt_distance'):
            vals = [per_sample[mid][si][col] for si in per_sample[mid] if math.isfinite(per_sample[mid][si].get(col, float('nan')))]
            neg = [v for v in vals if v < 0]
            add(f'{col}_nonnegative[{mid}]', len(neg), 0, 0, len(neg) == 0, notes=str(neg[:3]))

    # Metric coverage cross-check against unified_primary_missingness.csv
    missingness_lookup = {(r['method_id'], r['metric']): r for r in refs['missingness']}
    for mid in sorted(per_sample):
        for col in metric_cols:
            finite = sum(1 for si in per_sample[mid] if math.isfinite(per_sample[mid][si].get(col, float('nan'))))
            ref = missingness_lookup.get((mid, col))
            if ref is None:
                add(f'missingness_ref_present[{mid}][{col}]', 'missing', 'present', 0, False)
                continue
            ref_finite = int(ref['finite_rows'])
            add(f'metric_coverage_vs_phase1[{mid}][{col}]', finite, ref_finite, 0, finite == ref_finite)

    # Recompute method means vs unified_primary_method_summary.csv
    for row in refs['method_summary']:
        mid, col = row['method_id'], row['metric']
        vals = [per_sample[mid][si][col] for si in per_sample.get(mid, {}) if math.isfinite(per_sample[mid][si].get(col, float('nan')))]
        expected_mean = row['mean']
        if not vals:
            add(f'method_summary_mean_reproduction[{mid}][{col}]', '', '', RECOMPUTE_TOLERANCE,
                expected_mean == '')
            continue
        observed_mean = sum(vals) / len(vals)
        if expected_mean == '':
            add(f'method_summary_mean_reproduction[{mid}][{col}]', observed_mean, expected_mean,
                RECOMPUTE_TOLERANCE, False, notes='phase1 reports empty but recompute has data')
            continue
        diff = abs(observed_mean - float(expected_mean))
        add(f'method_summary_mean_reproduction[{mid}][{col}]', observed_mean, float(expected_mean),
            RECOMPUTE_TOLERANCE, diff <= RECOMPUTE_TOLERANCE, notes=f'abs_diff={diff:.3e}')

    # Recompute PD/MT means vs unified_primary_topology_validation.csv
    for row in refs['topo_val']:
        mid = row['method_id']
        for col, obs_key in (('pd_distance', 'observed_pd_mean'), ('mt_distance', 'observed_mt_mean')):
            vals = [per_sample[mid][si][col] for si in per_sample.get(mid, {}) if math.isfinite(per_sample[mid][si].get(col, float('nan')))]
            expected = row[obs_key]
            if not vals:
                add(f'topology_mean_reproduction[{mid}][{col}]', '', '', PD_MT_RECOMPUTE_TOLERANCE,
                    expected == '')
                continue
            observed_mean = sum(vals) / len(vals)
            if expected == '':
                add(f'topology_mean_reproduction[{mid}][{col}]', observed_mean, expected,
                    PD_MT_RECOMPUTE_TOLERANCE, False, notes='phase1 reports empty but recompute has data')
                continue
            diff = abs(observed_mean - float(expected))
            add(f'topology_mean_reproduction[{mid}][{col}]', observed_mean, float(expected),
                PD_MT_RECOMPUTE_TOLERANCE, diff <= PD_MT_RECOMPUTE_TOLERANCE, notes=f'abs_diff={diff:.3e}')

    return checks, failures


# =============================================================================
# Phase-1 pairwise reproduction (exact algorithmic reconstruction)
# =============================================================================

def phase1_style_pairwise(per_sample, metric_direction, method_id, metric):
    """Reproduces scripts/build_unified_candidate_evaluation.py's exact
    pairwise-vs-cnn algorithm (same tie definition ==0, same mean/median
    formula) so Output 8 can genuinely re-derive Phase-1's numbers rather
    than just re-reading them."""
    cnn_vals = {si: per_sample[CNN_METHOD][si][metric] for si in per_sample[CNN_METHOD]
                if math.isfinite(per_sample[CNN_METHOD][si].get(metric, float('nan')))}
    cand_vals = {si: per_sample[method_id][si][metric] for si in per_sample[method_id]
                 if math.isfinite(per_sample[method_id][si].get(metric, float('nan')))}
    common = sorted(set(cnn_vals) & set(cand_vals))
    if not common:
        # Matches build_unified_candidate_evaluation.py's exact empty-pair
        # branch: n_improved/n_worsened/n_tied are left as empty strings
        # (only n_valid is an actual int 0).
        return dict(cnn_mean='', candidate_mean='', mean_raw_delta='', mean_improvement_delta='',
                     median_improvement_delta='', n_improved='', n_worsened='', n_tied='', n_valid=0)
    direction = metric_direction[metric]
    raw_deltas = [cand_vals[si] - cnn_vals[si] for si in common]
    improve_deltas = raw_deltas if direction == 'higher_is_better' else [-d for d in raw_deltas]
    n_improved = sum(1 for d in improve_deltas if d > 0)
    n_worsened = sum(1 for d in improve_deltas if d < 0)
    n_tied = sum(1 for d in improve_deltas if d == 0)
    sorted_imp = sorted(improve_deltas)
    n = len(sorted_imp)
    median_imp = sorted_imp[n // 2] if n % 2 == 1 else (sorted_imp[n // 2 - 1] + sorted_imp[n // 2]) / 2
    return dict(
        cnn_mean=sum(cnn_vals[si] for si in common) / len(common),
        candidate_mean=sum(cand_vals[si] for si in common) / len(common),
        mean_raw_delta=sum(raw_deltas) / len(raw_deltas),
        mean_improvement_delta=sum(improve_deltas) / len(improve_deltas),
        median_improvement_delta=median_imp,
        n_improved=n_improved, n_worsened=n_worsened, n_tied=n_tied, n_valid=len(common),
    )


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log('=' * 88)
    log('Unified candidate analysis -- Phase 2A')
    log(f'Repo root: {REPO_ROOT}')
    log(f'SciPy available: {HAVE_SCIPY}')
    log('Read-only w.r.t. Phase-1 artifacts. No training/inference/eval/TTK performed.')
    log('=' * 88)

    protected_files = phase1_protected_files()
    checksums_before = checksum_all(protected_files)
    log(f'[immutability] Checksummed {len(checksums_before)} Phase-1 file(s) before the run.')

    log_step = '[load] '
    long_table = load_long_table()
    log(f'{log_step}Long table: {long_table["n_rows"]} rows, {len(long_table["metric_cols"])} metric columns, '
        f'{len(long_table["per_sample"])} methods.')
    metric_direction, metric_family = load_column_mapping()
    metric_cols = long_table['metric_cols']
    missing_direction = [c for c in metric_cols if c not in metric_direction]
    if missing_direction:
        raise SystemExit(f'[hard-fail] column_mapping.csv has no direction for metric(s): {missing_direction}')
    refs = load_reference_tables()
    log(f'{log_step}method_summary={len(refs["method_summary"])} rows, pairwise={len(refs["pairwise"])} rows, '
        f'topo_val={len(refs["topo_val"])} rows, missingness={len(refs["missingness"])} rows, '
        f'inventory={len(refs["inventory"])} rows.')

    inv_lookup = {r['method_id']: r for r in refs['inventory']}

    # -------------------------------------------------------------------
    # Data validation (hard-fail on any disagreement)
    # -------------------------------------------------------------------
    validation_rows, failures = run_validation(long_table, refs, metric_direction)
    if failures:
        log('')
        log('[VALIDATION FAILURE] The following checks failed:')
        for f in failures:
            log(f'  - {f}')
        flush_log()
        raise SystemExit(f'[hard-fail] {len(failures)} Phase-2A data-validation check(s) failed; refusing to '
                          'proceed. See log above for details.')
    log(f'[validate] All {len(validation_rows)} validation checks PASSED.')

    val_path = OUT_DIR / 'phase2a_validation.csv'
    with val_path.open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['check_name', 'observed', 'expected', 'tolerance', 'status', 'notes'])
        w.writeheader()
        for row in validation_rows:
            w.writerow(row)
    log(f'[write] {val_path} ({len(validation_rows)} rows)')

    per_sample = long_table['per_sample']
    methods = sorted(per_sample)
    non_cnn_methods = [m for m in methods if m != CNN_METHOD]

    def meta(mid):
        return long_table['method_meta'][mid]['values']

    # -------------------------------------------------------------------
    # OUTPUT 2: metric_coverage.csv
    # -------------------------------------------------------------------
    missingness_lookup = {(r['method_id'], r['metric']): r for r in refs['missingness']}
    coverage_rows = []
    for mid in methods:
        for col in metric_cols:
            finite = sum(1 for si in per_sample[mid] if math.isfinite(per_sample[mid][si].get(col, float('nan'))))
            total = N_EVAL
            missing = total - finite
            if finite == total:
                status = 'complete'
            elif col == 'ssim_speed' and finite == 0:
                status = 'globally_unavailable_optional'
            elif finite == 0:
                status = 'absent'
            else:
                status = 'partial_invalid'
            ref = missingness_lookup.get((mid, col), {})
            coverage_rows.append(dict(
                method_id=mid, display_name=meta(mid)['display_name'], metric=col,
                metric_family=metric_family.get(col, ''), direction=metric_direction[col],
                n_total=total, n_finite=finite, n_missing=missing, coverage_status=status,
                missing_reason=ref.get('missing_reason', ''),
            ))
    cov_path = OUT_DIR / 'metric_coverage.csv'
    with cov_path.open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['method_id', 'display_name', 'metric', 'metric_family', 'direction',
                                            'n_total', 'n_finite', 'n_missing', 'coverage_status', 'missing_reason'])
        w.writeheader()
        for row in coverage_rows:
            w.writerow(row)
    log(f'[write] {cov_path} ({len(coverage_rows)} rows)')

    # -------------------------------------------------------------------
    # OUTPUT 3: method_descriptive_summary.csv
    # -------------------------------------------------------------------
    desc_rows = []
    method_metric_values = {}  # (mid, col) -> list of finite values, cached for reuse
    for mid in methods:
        for col in metric_cols:
            vals = [per_sample[mid][si][col] for si in range(N_EVAL)
                    if math.isfinite(per_sample[mid][si].get(col, float('nan')))]
            method_metric_values[(mid, col)] = vals
            d = describe(vals)
            desc_rows.append(dict(
                method_id=mid, display_name=meta(mid)['display_name'],
                candidate_family=meta(mid)['candidate_family'], metric=col,
                metric_family=metric_family.get(col, ''), direction=metric_direction[col],
                n_valid=d['n_valid'],
                mean=d['mean'], standard_deviation=d['std'], standard_error=d['se'],
                median=d['median'], q25=d['q25'], q75=d['q75'],
                minimum=d['minimum'], maximum=d['maximum'],
                bootstrap_ci95_low=d['ci_low'], bootstrap_ci95_high=d['ci_high'],
            ))
    desc_path = OUT_DIR / 'method_descriptive_summary.csv'
    with desc_path.open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['method_id', 'display_name', 'candidate_family', 'metric',
                                            'metric_family', 'direction', 'n_valid', 'mean',
                                            'standard_deviation', 'standard_error', 'median', 'q25', 'q75',
                                            'minimum', 'maximum', 'bootstrap_ci95_low', 'bootstrap_ci95_high'])
        w.writeheader()
        for row in desc_rows:
            w.writerow(row)
    log(f'[write] {desc_path} ({len(desc_rows)} rows)')

    # -------------------------------------------------------------------
    # OUTPUT 4: paired_vs_cnn_detailed.csv
    # -------------------------------------------------------------------
    paired_rows = []
    n_valid_paired_tests = 0
    for mid in non_cnn_methods:
        for col in metric_cols:
            direction = metric_direction[col]
            cnn_vals = {si: per_sample[CNN_METHOD][si][col] for si in range(N_EVAL)
                        if math.isfinite(per_sample[CNN_METHOD][si].get(col, float('nan')))}
            m_vals = {si: per_sample[mid][si][col] for si in range(N_EVAL)
                      if math.isfinite(per_sample[mid][si].get(col, float('nan')))}
            common = sorted(set(cnn_vals) & set(m_vals))
            n_pairs = len(common)

            row = dict(method_id=mid, display_name=meta(mid)['display_name'],
                        candidate_family=meta(mid)['candidate_family'], metric=col,
                        metric_family=metric_family.get(col, ''), direction=direction,
                        n_valid_pairs=n_pairs)
            if n_pairs == 0:
                row.update(cnn_mean_on_valid_pairs='', method_mean_on_valid_pairs='',
                            mean_raw_delta='', median_raw_delta='', mean_improvement='',
                            median_improvement='', improvement_standard_deviation='',
                            improvement_standard_error='', improvement_ci95_low='', improvement_ci95_high='',
                            win_count=0, tie_count=0, loss_count=0, win_rate='', tie_rate='', loss_rate='',
                            paired_effect_size_dz='', sign_test_p_raw='', wilcoxon_p_raw='',
                            test_status='no_valid_pairs')
                paired_rows.append(row)
                continue

            n_valid_paired_tests += 1
            cnn_arr = np.array([cnn_vals[si] for si in common], dtype=np.float64)
            m_arr = np.array([m_vals[si] for si in common], dtype=np.float64)
            raw_delta = m_arr - cnn_arr
            improvement = raw_delta if direction == 'higher_is_better' else -raw_delta

            win_count = int(np.sum(improvement > TIE_TOLERANCE))
            loss_count = int(np.sum(improvement < -TIE_TOLERANCE))
            tie_count = n_pairs - win_count - loss_count

            imp_mean = float(improvement.mean())
            imp_median = float(np.median(improvement))
            if n_pairs >= 2:
                imp_std = float(improvement.std(ddof=1))
                imp_se = imp_std / math.sqrt(n_pairs)
            else:
                imp_std = None
                imp_se = None
            ci_lo, ci_hi = bootstrap_ci95(improvement)

            dz = (imp_mean / imp_std) if (imp_std is not None and imp_std != 0) else ''

            n_nontied = win_count + loss_count
            sign_p = exact_sign_test_pvalue(win_count, loss_count) if n_nontied > 0 else None

            wilcoxon_p = None
            wilcoxon_note = 'scipy_unavailable'
            if HAVE_SCIPY:
                try:
                    if np.all(improvement == 0):
                        wilcoxon_note = 'wilcoxon_undefined_all_zero_differences'
                    else:
                        res = _scipy_stats.wilcoxon(improvement, alternative='two-sided',
                                                      zero_method='wilcox', mode='auto')
                        wilcoxon_p = float(res.pvalue)
                        wilcoxon_note = 'ok'
                except Exception as e:
                    wilcoxon_note = f'wilcoxon_failed:{type(e).__name__}'

            sign_note = 'ok' if sign_p is not None else 'sign_test_undefined_zero_nontied_pairs'
            test_status = f'sign_test={sign_note}; wilcoxon={wilcoxon_note}'

            row.update(
                cnn_mean_on_valid_pairs=float(cnn_arr.mean()),
                method_mean_on_valid_pairs=float(m_arr.mean()),
                mean_raw_delta=float(raw_delta.mean()), median_raw_delta=float(np.median(raw_delta)),
                mean_improvement=imp_mean, median_improvement=imp_median,
                improvement_standard_deviation=imp_std, improvement_standard_error=imp_se,
                improvement_ci95_low=ci_lo, improvement_ci95_high=ci_hi,
                win_count=win_count, tie_count=tie_count, loss_count=loss_count,
                win_rate=win_count / n_pairs, tie_rate=tie_count / n_pairs, loss_rate=loss_count / n_pairs,
                paired_effect_size_dz=dz, sign_test_p_raw=sign_p,
                wilcoxon_p_raw=(wilcoxon_p if wilcoxon_p is not None else ''),
                test_status=test_status,
            )
            paired_rows.append(row)

    paired_fields = ['method_id', 'display_name', 'candidate_family', 'metric', 'metric_family', 'direction',
                      'n_valid_pairs', 'cnn_mean_on_valid_pairs', 'method_mean_on_valid_pairs', 'mean_raw_delta',
                      'median_raw_delta', 'mean_improvement', 'median_improvement',
                      'improvement_standard_deviation', 'improvement_standard_error', 'improvement_ci95_low',
                      'improvement_ci95_high', 'win_count', 'tie_count', 'loss_count', 'win_rate', 'tie_rate',
                      'loss_rate', 'paired_effect_size_dz', 'sign_test_p_raw', 'wilcoxon_p_raw', 'test_status']
    paired_path = OUT_DIR / 'paired_vs_cnn_detailed.csv'
    with paired_path.open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=paired_fields)
        w.writeheader()
        for row in paired_rows:
            w.writerow(row)
    log(f'[write] {paired_path} ({len(paired_rows)} rows; {n_valid_paired_tests} with >=1 valid pair)')

    # -------------------------------------------------------------------
    # OUTPUT 5: paired_vs_cnn_adjusted.csv (Holm correction)
    # -------------------------------------------------------------------
    sign_items_global = [((r['method_id'], r['metric']), r['sign_test_p_raw']) for r in paired_rows
                          if r['sign_test_p_raw'] not in ('', None)]
    wilcoxon_items_global = [((r['method_id'], r['metric']), r['wilcoxon_p_raw']) for r in paired_rows
                              if r['wilcoxon_p_raw'] not in ('', None)]
    sign_holm_global = holm_correction(sign_items_global)
    wilcoxon_holm_global = holm_correction(wilcoxon_items_global)

    sign_holm_within = {}
    wilcoxon_holm_within = {}
    for col in metric_cols:
        sign_items = [((r['method_id'], r['metric']), r['sign_test_p_raw']) for r in paired_rows
                      if r['metric'] == col and r['sign_test_p_raw'] not in ('', None)]
        wilcoxon_items = [((r['method_id'], r['metric']), r['wilcoxon_p_raw']) for r in paired_rows
                           if r['metric'] == col and r['wilcoxon_p_raw'] not in ('', None)]
        sign_holm_within.update(holm_correction(sign_items))
        wilcoxon_holm_within.update(holm_correction(wilcoxon_items))

    adjusted_rows = []
    for r in paired_rows:
        key = (r['method_id'], r['metric'])
        adj = dict(r)
        adj['sign_test_p_holm_global'] = sign_holm_global.get(key, '')
        adj['sign_test_p_holm_within_metric'] = sign_holm_within.get(key, '')
        adj['wilcoxon_p_holm_global'] = wilcoxon_holm_global.get(key, '')
        adj['wilcoxon_p_holm_within_metric'] = wilcoxon_holm_within.get(key, '')
        adjusted_rows.append(adj)
    adjusted_fields = paired_fields + ['sign_test_p_holm_global', 'sign_test_p_holm_within_metric',
                                        'wilcoxon_p_holm_global', 'wilcoxon_p_holm_within_metric']
    adjusted_path = OUT_DIR / 'paired_vs_cnn_adjusted.csv'
    with adjusted_path.open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=adjusted_fields)
        w.writeheader()
        for row in adjusted_rows:
            w.writerow(row)
    log(f'[write] {adjusted_path} ({len(adjusted_rows)} rows; '
        f'{len(sign_items_global)} sign-test p-values Holm-corrected globally)')

    # -------------------------------------------------------------------
    # OUTPUT 6: method_mean_improvement_matrix.csv / method_win_rate_matrix.csv
    # -------------------------------------------------------------------
    paired_lookup = {(r['method_id'], r['metric']): r for r in paired_rows}
    imp_matrix_path = OUT_DIR / 'method_mean_improvement_matrix.csv'
    with imp_matrix_path.open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['method_id', 'display_name'] + metric_cols)
        w.writeheader()
        for mid in non_cnn_methods:
            row = {'method_id': mid, 'display_name': meta(mid)['display_name']}
            for col in metric_cols:
                row[col] = paired_lookup[(mid, col)]['mean_improvement']
            w.writerow(row)
    log(f'[write] {imp_matrix_path} ({len(non_cnn_methods)} rows x {len(metric_cols)} metric columns)')

    win_matrix_path = OUT_DIR / 'method_win_rate_matrix.csv'
    with win_matrix_path.open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['method_id', 'display_name'] + metric_cols)
        w.writeheader()
        for mid in non_cnn_methods:
            row = {'method_id': mid, 'display_name': meta(mid)['display_name']}
            for col in metric_cols:
                row[col] = paired_lookup[(mid, col)]['win_rate']
            w.writerow(row)
    log(f'[write] {win_matrix_path} ({len(non_cnn_methods)} rows x {len(metric_cols)} metric columns)')

    # -------------------------------------------------------------------
    # OUTPUT 7: topology_tradeoff_summary.csv (+ sorted variant)
    # -------------------------------------------------------------------
    topo_rows = []
    quadrant_counts: dict = {}
    for mid in methods:
        pd_vals = method_metric_values[(mid, 'pd_distance')]
        mt_vals = method_metric_values[(mid, 'mt_distance')]
        has_topology = bool(pd_vals) and bool(mt_vals)
        display_name = meta(mid)['display_name']

        if mid == CNN_METHOD:
            quadrant = 'cnn_reference'
            row = dict(method_id=mid, display_name=display_name,
                        pd_mean=sum(pd_vals) / len(pd_vals), mt_mean=sum(mt_vals) / len(mt_vals),
                        pd_mean_improvement_vs_cnn=0.0, pd_median_improvement_vs_cnn=0.0,
                        pd_win_rate_vs_cnn='', mt_mean_improvement_vs_cnn=0.0,
                        mt_median_improvement_vs_cnn=0.0, mt_win_rate_vs_cnn='',
                        improves_pd_mean='', improves_mt_mean='', topology_quadrant=quadrant)
        elif not has_topology:
            quadrant = 'topology_unavailable'
            row = dict(method_id=mid, display_name=display_name, pd_mean='', mt_mean='',
                        pd_mean_improvement_vs_cnn='', pd_median_improvement_vs_cnn='', pd_win_rate_vs_cnn='',
                        mt_mean_improvement_vs_cnn='', mt_median_improvement_vs_cnn='', mt_win_rate_vs_cnn='',
                        improves_pd_mean='', improves_mt_mean='', topology_quadrant=quadrant)
        else:
            pd_pair = paired_lookup[(mid, 'pd_distance')]
            mt_pair = paired_lookup[(mid, 'mt_distance')]
            improves_pd = pd_pair['mean_improvement'] not in ('', None) and pd_pair['mean_improvement'] > 0
            improves_mt = mt_pair['mean_improvement'] not in ('', None) and mt_pair['mean_improvement'] > 0
            if improves_pd and improves_mt:
                quadrant = 'improves_both'
            elif improves_pd:
                quadrant = 'improves_pd_only'
            elif improves_mt:
                quadrant = 'improves_mt_only'
            else:
                quadrant = 'improves_neither'
            row = dict(
                method_id=mid, display_name=display_name,
                pd_mean=sum(pd_vals) / len(pd_vals), mt_mean=sum(mt_vals) / len(mt_vals),
                pd_mean_improvement_vs_cnn=pd_pair['mean_improvement'],
                pd_median_improvement_vs_cnn=pd_pair['median_improvement'],
                pd_win_rate_vs_cnn=pd_pair['win_rate'],
                mt_mean_improvement_vs_cnn=mt_pair['mean_improvement'],
                mt_median_improvement_vs_cnn=mt_pair['median_improvement'],
                mt_win_rate_vs_cnn=mt_pair['win_rate'],
                improves_pd_mean=improves_pd, improves_mt_mean=improves_mt,
                topology_quadrant=quadrant,
            )
        quadrant_counts[quadrant] = quadrant_counts.get(quadrant, 0) + 1
        topo_rows.append(row)

    topo_fields = ['method_id', 'display_name', 'pd_mean', 'pd_mean_improvement_vs_cnn',
                    'pd_median_improvement_vs_cnn', 'pd_win_rate_vs_cnn', 'mt_mean',
                    'mt_mean_improvement_vs_cnn', 'mt_median_improvement_vs_cnn', 'mt_win_rate_vs_cnn',
                    'improves_pd_mean', 'improves_mt_mean', 'topology_quadrant']
    topo_path = OUT_DIR / 'topology_tradeoff_summary.csv'
    with topo_path.open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=topo_fields)
        w.writeheader()
        for row in topo_rows:
            w.writerow(row)
    log(f'[write] {topo_path} ({len(topo_rows)} rows); quadrant counts: {quadrant_counts}')

    def sort_key(row):
        pd_m = row['pd_mean'] if row['pd_mean'] != '' else float('inf')
        mt_m = row['mt_mean'] if row['mt_mean'] != '' else float('inf')
        return (row['topology_quadrant'], pd_m, mt_m)

    topo_sorted_path = OUT_DIR / 'topology_tradeoff_summary_sorted.csv'
    with topo_sorted_path.open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=topo_fields)
        w.writeheader()
        for row in sorted(topo_rows, key=sort_key):
            w.writerow(row)
    log(f'[write] {topo_sorted_path} (sorted by topology_quadrant, pd_mean, mt_mean -- not a total ranking)')

    # -------------------------------------------------------------------
    # OUTPUT 8: phase1_pairwise_reproduction.csv
    # -------------------------------------------------------------------
    repro_rows = []
    repro_failures = []
    phase1_pairwise_lookup = {(r['method_id'], r['metric']): r for r in refs['pairwise']}
    compare_fields = ['cnn_mean', 'candidate_mean', 'mean_raw_delta', 'mean_improvement_delta',
                        'median_improvement_delta', 'n_improved', 'n_worsened', 'n_tied', 'n_valid']
    for mid in non_cnn_methods:
        for col in metric_cols:
            phase1_row = phase1_pairwise_lookup.get((mid, col))
            if phase1_row is None:
                repro_failures.append(f'{mid}/{col}: no Phase-1 pairwise row found')
                continue
            recomputed = phase1_style_pairwise(per_sample, metric_direction, mid, col)
            for field in compare_fields:
                phase1_val = phase1_row[field]
                recomputed_val = recomputed[field]
                if phase1_val in ('', None) and recomputed_val in ('', None):
                    status = 'PASS'
                    abs_diff = ''
                elif phase1_val in ('', None) or recomputed_val in ('', None):
                    status = 'FAIL'
                    abs_diff = ''
                    repro_failures.append(f'{mid}/{col}/{field}: phase1={phase1_val!r} recomputed={recomputed_val!r}')
                else:
                    tol = 0 if field in ('n_improved', 'n_worsened', 'n_tied', 'n_valid') else RECOMPUTE_TOLERANCE
                    abs_diff = abs(float(phase1_val) - float(recomputed_val))
                    status = 'PASS' if abs_diff <= tol else 'FAIL'
                    if status == 'FAIL':
                        repro_failures.append(f'{mid}/{col}/{field}: phase1={phase1_val} recomputed={recomputed_val} '
                                                f'abs_diff={abs_diff}')
                repro_rows.append(dict(method_id=mid, metric=col, field=field,
                                         phase1_value=phase1_val, phase2a_recomputed_value=recomputed_val,
                                         absolute_difference=abs_diff, status=status))
    repro_path = OUT_DIR / 'phase1_pairwise_reproduction.csv'
    with repro_path.open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['method_id', 'metric', 'field', 'phase1_value',
                                            'phase2a_recomputed_value', 'absolute_difference', 'status'])
        w.writeheader()
        for row in repro_rows:
            w.writerow(row)
    log(f'[write] {repro_path} ({len(repro_rows)} rows)')
    if repro_failures:
        log('[REPRODUCTION FAILURE]')
        for f in repro_failures[:20]:
            log(f'  - {f}')
        flush_log()
        raise SystemExit(f'[hard-fail] {len(repro_failures)} Phase-1 pairwise field(s) failed to reproduce within '
                          'tolerance; see phase1_pairwise_reproduction.csv and the log above.')
    log(f'[validate] All {len(repro_rows)} Phase-1 pairwise field comparisons PASSED.')

    # -------------------------------------------------------------------
    # Documentation
    # -------------------------------------------------------------------
    write_phase2a_doc(methods, non_cnn_methods, metric_cols, coverage_rows, desc_rows, paired_rows,
                        topo_rows, quadrant_counts, n_valid_paired_tests, validation_rows, repro_rows)

    # -------------------------------------------------------------------
    # Phase-1 immutability check (after)
    # -------------------------------------------------------------------
    checksums_after = checksum_all(phase1_protected_files())
    immut_rows = []
    changed = []
    for path_str, before in sorted(checksums_before.items()):
        after = checksums_after.get(path_str)
        status = 'unchanged' if after == before else 'CHANGED'
        if after is None:
            status = 'MISSING_AFTER_RUN'
        if status != 'unchanged':
            changed.append(path_str)
        immut_rows.append(dict(file_path=path_str, sha256_before=before, sha256_after=(after or ''),
                                 status=status))
    for path_str in sorted(set(checksums_after) - set(checksums_before)):
        immut_rows.append(dict(file_path=path_str, sha256_before='', sha256_after=checksums_after[path_str],
                                 status='NEW_FILE_APPEARED'))
        changed.append(path_str)
    immut_path = OUT_DIR / 'phase1_immutability_check.csv'
    with immut_path.open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['file_path', 'sha256_before', 'sha256_after', 'status'])
        w.writeheader()
        for row in immut_rows:
            w.writerow(row)
    log(f'[write] {immut_path} ({len(immut_rows)} files checked)')

    if changed:
        log(f'[IMMUTABILITY FAILURE] {len(changed)} Phase-1 file(s) changed during this run: {changed}')
        flush_log()
        raise SystemExit(f'[hard-fail] Phase-1 immutability violated: {changed}')
    log(f'[immutability] Confirmed all {len(immut_rows)} Phase-1 file(s) unchanged.')

    log('')
    log('=' * 88)
    log(f'RESULT: Phase 2A complete. {len(methods)} methods, {n_valid_paired_tests} valid method x metric paired '
        f'tests, topology quadrant counts: {quadrant_counts}. Phase-1 files unchanged.')
    log('=' * 88)
    flush_log()
    return 0


def write_phase2a_doc(methods, non_cnn_methods, metric_cols, coverage_rows, desc_rows, paired_rows,
                        topo_rows, quadrant_counts, n_valid_paired_tests, validation_rows, repro_rows):
    ssim_rows = [r for r in coverage_rows if r['metric'] == 'ssim_speed']
    ssim_full = sorted(r['method_id'] for r in ssim_rows if r['coverage_status'] == 'complete')
    ssim_unavail = sorted(r['method_id'] for r in ssim_rows if r['coverage_status'] == 'globally_unavailable_optional')

    lines = []
    lines.append('# Unified candidate analysis -- Phase 2A report')
    lines.append('')
    lines.append('## 1. Scope and authoritative inputs')
    lines.append('')
    lines.append('Phase 2A is a deterministic descriptive and paired multi-metric analysis of all 19 primary '
                 'methods, built exclusively from the immutable Phase-1 outputs under '
                 '`ttk_runs_fixed/unified_candidate_evaluation/` (`unified_primary_per_sample_long.csv` as the '
                 'source of truth; `unified_primary_method_summary.csv`, `unified_primary_pairwise_vs_cnn.csv`, '
                 '`unified_primary_topology_validation.csv`, `unified_primary_missingness.csv`, '
                 '`method_inventory.csv`, and `column_mapping.csv` as validation references). No Phase-1 file was '
                 'modified, regenerated, or overwritten. No training, inference, cheap evaluation, or TTK was run.')
    lines.append('')
    lines.append('Explicitly deferred to later Phase-2 stages (not performed here): speed x gradient x level-set '
                 'factorial decomposition, targeted E2/critical-proxy contrasts, metric-correlation analysis, '
                 'Pareto-front analysis, sample selection, visualization generation, and composite/weighted ranking.')
    lines.append('')
    lines.append('## 2. Validation results')
    lines.append('')
    lines.append(f'All {len(validation_rows)} independent validation checks passed: exact row/method/sample '
                 'counts, no duplicate (method_id, sample_idx) keys, constant per-method metadata, SSIM at either '
                 '168/168 or 0/168 per method, finite/nonnegative PD and MT wherever topology data exists, exact '
                 'reproduction of every `unified_primary_method_summary.csv` mean, and exact reproduction of every '
                 '`unified_primary_topology_validation.csv` PD/MT mean. See `phase2a_validation.csv` for the full '
                 'per-check ledger. Any failure here would have hard-failed the run before any other output was '
                 'written.')
    lines.append(f'Additionally, all {len(repro_rows)} field-level comparisons against '
                 '`unified_primary_pairwise_vs_cnn.csv` (an independent re-derivation using Phase-1\'s exact '
                 'algorithm, not a copy) passed -- see `phase1_pairwise_reproduction.csv` and section 5.')
    lines.append('')
    lines.append('## 3. Metric coverage and SSIM status')
    lines.append('')
    lines.append(f'22 metrics x 19 methods = {len(coverage_rows)} (method, metric) coverage rows in '
                 '`metric_coverage.csv`. SSIM (`ssim_speed`) is the only metric allowed to be globally unavailable '
                 '(the documented NumPy/scikit-image ABI issue): fully available (168/168) for '
                 f'{ssim_full}; globally unavailable (0/168) for {ssim_unavail}. No value was ever imputed -- '
                 'missing cells stay empty in every generated CSV.')
    lines.append('')
    lines.append('## 4. Descriptive method-level results')
    lines.append('')
    lines.append(f'`method_descriptive_summary.csv` has {len(desc_rows)} rows (one per method x metric): mean, '
                 'sample standard deviation, standard error, median, quartiles, min/max, and a 95% bootstrap CI '
                 'of the mean (10,000 resamples, seed 20260721, sample-axis resampling of the 168 benchmark '
                 'indices). Rows with `n_valid=0` (e.g. bicubic PD/MT, or a method whose SSIM is globally '
                 'unavailable) leave every numeric field empty rather than reporting a fabricated statistic.')
    lines.append('')
    lines.append('## 5. Paired comparison methodology')
    lines.append('')
    lines.append('For every non-CNN method x metric, `paired_vs_cnn_detailed.csv` restricts to samples where '
                 'BOTH CNN and the method have a finite value (`n_valid_pairs`), then computes the direction-aware '
                 '`improvement` (positive always means better than CNN; `raw_delta = method - cnn` is preserved '
                 'separately), win/tie/loss counts using a '
                 f'{TIE_TOLERANCE:g} tie tolerance, an exact two-sided sign test over non-tied pairs only (no '
                 'SciPy required), the paired effect size `dz = mean(improvement) / std(improvement)` (empty when '
                 'the paired standard deviation is zero or undefined), and the same deterministic bootstrap for '
                 'the improvement 95% CI. '
                 f'{"A two-sided paired Wilcoxon signed-rank test (SciPy) was also computed." if HAVE_SCIPY else "SciPy was not importable in this environment, so the Wilcoxon column is left empty for every row and `test_status` records `wilcoxon=scipy_unavailable` -- the analysis completed successfully without it, as required."} '
                 f'{len(paired_rows)} method x metric rows were produced; {n_valid_paired_tests} had at least one '
                 'valid pair. `paired_vs_cnn_adjusted.csv` adds Holm-corrected p-values, computed both once across '
                 'all valid comparisons (`_holm_global`) and separately within each metric across methods '
                 '(`_holm_within_metric`), for both the sign test and (when available) Wilcoxon. No effect is '
                 'labeled "significant" from these p-values alone -- the adjusted values are preserved for later '
                 'interpretation. `method_mean_improvement_matrix.csv` and `method_win_rate_matrix.csv` pivot the '
                 'mean improvement and win rate into a method x metric matrix; no aggregate ranking or weighted '
                 'score was computed.')
    lines.append('')
    lines.append('## 6. Benchmark-sample uncertainty caveat')
    lines.append('')
    lines.append('The 168 samples are paired benchmark observations, not independent training runs -- every '
                 'model here was trained exactly once. The bootstrap confidence intervals and sign-test/Wilcoxon '
                 'p-values in this report quantify variability **across the 168 benchmark samples only**. They do '
                 'not, and cannot, quantify variation across independent training seeds or reruns. Results here '
                 'therefore support benchmark-level comparisons (does this trained model do better than CNN on '
                 'this fixed evaluation set?) but not claims about training-run robustness (would a differently-'
                 'seeded retraining of the same objective reproduce this result?).')
    lines.append('')
    lines.append('## 7. Topology tradeoff summary')
    lines.append('')
    lines.append(f'`topology_tradeoff_summary.csv` has one row per method ({len(topo_rows)} total); quadrant '
                 f'counts: {quadrant_counts}. `bicubic` is marked `topology_unavailable` (it has no PD/MT source, '
                 'consistent with Phase-1); `cnn` is marked `cnn_reference`. '
                 '`topology_tradeoff_summary_sorted.csv` sorts by (topology_quadrant, pd_mean, mt_mean) -- this '
                 'is a display ordering only, not a claimed total ranking.')
    lines.append('')
    lines.append('## 8. Strongest descriptive patterns')
    lines.append('')
    both = sorted(r['method_id'] for r in topo_rows if r['topology_quadrant'] == 'improves_both')
    neither = sorted(r['method_id'] for r in topo_rows if r['topology_quadrant'] == 'improves_neither')
    lines.append(f'Methods whose mean PD **and** mean MT both improve over CNN on this benchmark: {both or "none"}. '
                 f'Methods whose mean PD and mean MT both fail to improve over CNN: {neither or "none"}. '
                 'This is a purely descriptive observation from the paired means above (section 7) -- it is not a '
                 'causal claim about which loss term drove the result and not a ranking; per-metric win rates and '
                 'confidence intervals in `paired_vs_cnn_detailed.csv` should be consulted before drawing '
                 'conclusions about any individual method.')
    lines.append('')
    lines.append('## 9. Deferred analyses')
    lines.append('')
    lines.append('Causal loss-term attribution (e.g. isolating the effect of `L_grad` via the B-factorial '
                 'ablation), metric correlation analysis, Pareto-front analysis, and sample-level visualization '
                 'selection are all explicitly deferred to later Phase-2 stages. Nothing in this report should be '
                 'read as performing that attribution.')
    lines.append('')
    lines.append('## 10. Generated file list')
    lines.append('')
    for fname in ['phase2a_validation.csv', 'metric_coverage.csv', 'method_descriptive_summary.csv',
                  'paired_vs_cnn_detailed.csv', 'paired_vs_cnn_adjusted.csv', 'method_mean_improvement_matrix.csv',
                  'method_win_rate_matrix.csv', 'topology_tradeoff_summary.csv',
                  'topology_tradeoff_summary_sorted.csv', 'phase1_pairwise_reproduction.csv',
                  'phase1_immutability_check.csv']:
        lines.append(f'- `ttk_runs_fixed/unified_candidate_analysis/phase2a/{fname}`')
    lines.append('- `docs/unified_candidate_analysis_phase2a.md` (this file)')
    lines.append('- `logs/unified_candidate_analysis_phase2a.log`')
    lines.append('')
    (DOCS_DIR / 'unified_candidate_analysis_phase2a.md').write_text('\n'.join(lines) + '\n')
    log(f"[write] {DOCS_DIR / 'unified_candidate_analysis_phase2a.md'}")


if __name__ == '__main__':
    sys.exit(main())
