#!/usr/bin/env python3
"""Synthetic tests for scripts/analyze_unified_candidate_relationships_phase2c.py.

Exercises the real shipped functions directly (from-scratch Pearson/Spearman,
orientation, within/cross-method dimension enforcement, two-way-centered
residual demeaning, PD/MT preference agreement, Pareto dominance/front/layer
computation, deterministic bootstrap index construction, and prior-phase
immutability hard-fail gates). Any test that temporarily perturbs a real
Phase-1/Phase-2A/Phase-2B protected file restores it immediately in a
`finally` block, and the suite verifies the restoration succeeded before
continuing.

Run directly:
    python3 scripts/test_analyze_unified_candidate_relationships_phase2c.py
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_unified_candidate_relationships_phase2c as m
import numpy as np

failures = []


def check(name, cond):
    status = 'PASS' if cond else 'FAIL'
    print(f'[{status}] {name}')
    if not cond:
        failures.append(name)


def make_synthetic_per_sample(method_specs):
    """method_specs: {method_id: {metric: array-like of length N_EVAL or None for all-NaN}}"""
    per_sample = {}
    for mid, metrics in method_specs.items():
        per_sample[mid] = {}
        for si in range(m.N_EVAL):
            rec = {}
            for metric, arr in metrics.items():
                rec[metric] = float('nan') if arr is None else float(arr[si])
            per_sample[mid][si] = rec
    return per_sample


print('=== 1. Pearson identity / reversal ===')
x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
check('pearson_r(x, x) == 1', abs(m.pearson_r(x, x) - 1.0) < 1e-12)
check('pearson_r(x, -x) == -1', abs(m.pearson_r(x, -x) - (-1.0)) < 1e-12)
check('pearson_r(x, 2x+3) == 1 (affine invariance)', abs(m.pearson_r(x, 2 * x + 3) - 1.0) < 1e-12)
const = np.array([5.0, 5.0, 5.0])
check('pearson_r with zero variance -> None', m.pearson_r(const, x[:3]) is None)

print()
print('=== 2. Spearman identity / reversal / tied ranks ===')
check('spearman_r(x, x) == 1', abs(m.spearman_r(x, x) - 1.0) < 1e-12)
check('spearman_r(x, -x) == -1', abs(m.spearman_r(x, -x) - (-1.0)) < 1e-12)
# Non-monotonic but positively-correlated-in-rank transform still gives rho=1
# as long as ranks agree: use a strictly increasing nonlinear transform.
check('spearman_r(x, x**3) == 1 (rank-preserving nonlinear)', abs(m.spearman_r(x, x ** 3) - 1.0) < 1e-12)

tied = np.array([1.0, 2.0, 2.0, 2.0, 5.0])
ranks = m.rankdata_avg(tied)
# positions 2,3,4 (0-indexed) are tied for ranks 2,3,4 -> average rank 3.0 each
expected_ranks = np.array([1.0, 3.0, 3.0, 3.0, 5.0])
check('rankdata_avg average-rank ties correct', np.allclose(ranks, expected_ranks))
check('rankdata_avg ranks sum to n(n+1)/2', abs(ranks.sum() - 5 * 6 / 2) < 1e-9)

print()
print('=== 3. Vectorized row-wise Pearson/Spearman match scalar versions ===')
rng = np.random.default_rng(12345)
X = rng.normal(size=(6, 20))
Y = rng.normal(size=(6, 20))
row_p = m.pearson_r_rows(X, Y)
row_s = m.spearman_r_rows(X, Y)
for i in range(6):
    check(f'pearson_r_rows matches pearson_r (row {i})',
          abs(row_p[i] - m.pearson_r(X[i], Y[i])) < 1e-9)
    check(f'spearman_r_rows matches spearman_r (row {i})',
          abs(row_s[i] - m.spearman_r(X[i], Y[i])) < 1e-9)

print()
print('=== 4. Oriented sign reversal ===')
raw = np.array([1.0, 2.0, 3.0])
check('orient(higher_is_better) == raw', np.array_equal(m.orient(raw, 'higher_is_better'), raw))
check('orient(lower_is_better) == -raw', np.array_equal(m.orient(raw, 'lower_is_better'), -raw))

print()
print('=== 5. Method-level correlation with LOO bounds (hand-computed) ===')
# 4 methods; metric 'a' perfectly increasing, metric 'b' perfectly increasing ->
# oriented correlation should be exactly 1.0 for every leave-one-out subset too.
methods_5 = ['m1', 'm2', 'm3', 'm4']
means_5 = {
    'm1': dict(raw={'a': 1.0, 'b': 10.0}, oriented={'a': 1.0, 'b': 10.0}),
    'm2': dict(raw={'a': 2.0, 'b': 20.0}, oriented={'a': 2.0, 'b': 20.0}),
    'm3': dict(raw={'a': 3.0, 'b': 30.0}, oriented={'a': 3.0, 'b': 30.0}),
    'm4': dict(raw={'a': 4.0, 'b': 40.0}, oriented={'a': 4.0, 'b': 40.0}),
}
rows_5, pm_5, sm_5 = m.method_level_metric_correlations(methods_5, means_5, ['a', 'b'])
row_ab = rows_5[0]
check('perfectly linear pair: oriented_pearson == 1', abs(row_ab['oriented_pearson'] - 1.0) < 1e-9)
check('perfectly linear pair: loo_pearson_min == loo_pearson_max == 1',
      abs(row_ab['loo_pearson_min'] - 1.0) < 1e-9 and abs(row_ab['loo_pearson_max'] - 1.0) < 1e-9)
check('perfectly linear pair: pearson_sign_stable is True', row_ab['pearson_sign_stable'] is True)
check('method_level matrix agrees with pairwise row', abs(pm_5['a']['b'] - row_ab['oriented_pearson']) < 1e-12)
check('method_level matrix diagonal is 1.0', pm_5['a']['a'] == 1.0 and pm_5['b']['b'] == 1.0)

# One method with a reversed value should break sign stability under LOO.
means_5b = dict(means_5)
means_5b['m4'] = dict(raw={'a': 4.0, 'b': -100.0}, oriented={'a': 4.0, 'b': -100.0})
rows_5b, _, _ = m.method_level_metric_correlations(methods_5, means_5b, ['a', 'b'])
row_ab_b = rows_5b[0]
check('LOO bounds straddle sign after one method reversed -> sign_stable False',
      row_ab_b['pearson_sign_stable'] is False)

print()
print('=== 6. Within-method correlation: dimension enforcement ===')
xvals = np.linspace(0, 1, m.N_EVAL)
yvals = 2.0 * xvals + 1.0
per_sample_6 = make_synthetic_per_sample({
    'm1': {'x': xvals, 'y': yvals},
    'm2': {'x': None, 'y': None},  # fully unavailable, like bicubic PD/MT
})
within_rows, within_acc = m.within_method_metric_correlations(
    per_sample_6, ['m1', 'm2'], ['x', 'y'], {'x': 'higher_is_better', 'y': 'higher_is_better'})
row_m1 = next(r for r in within_rows if r['method_id'] == 'm1')
row_m2 = next(r for r in within_rows if r['method_id'] == 'm2')
check('within-method available case has exactly N_EVAL samples', row_m1['n_samples'] == m.N_EVAL)
check('within-method available case status', row_m1['status'] == 'available')
check('within-method available case oriented_pearson == 1', abs(row_m1['oriented_pearson'] - 1.0) < 1e-9)
check('within-method fully-missing case has 0 samples, status unavailable',
      row_m2['n_samples'] == 0 and row_m2['status'] == 'unavailable')

# Partial coverage must hard-fail, never silently produce a partial correlation.
xvals_partial = xvals.copy()
xvals_partial[:50] = np.nan
per_sample_6b = make_synthetic_per_sample({'m1': {'x': xvals_partial, 'y': yvals}})
try:
    m.within_method_metric_correlations(per_sample_6b, ['m1'], ['x', 'y'],
                                          {'x': 'higher_is_better', 'y': 'higher_is_better'})
    check('within-method partial coverage -> SystemExit', False)
except SystemExit as e:
    check('within-method partial coverage -> SystemExit', 'partial sample coverage' in str(e))

print()
print('=== 7. Samplewise cross-method correlation: dimension enforcement ===')
xarr = np.linspace(0, 1, m.N_EVAL)
per_sample_7 = make_synthetic_per_sample({
    'bicubic': {'x': xarr, 'y': xarr * 2, 'pd_distance': None},
    'ma': {'x': xarr + 1, 'y': xarr * 2 + 1, 'pd_distance': xarr},
    'mb': {'x': xarr + 2, 'y': xarr * 2 + 2, 'pd_distance': xarr + 1},
})
topology_methods_7 = ['ma', 'mb']
methods_7 = ['bicubic', 'ma', 'mb']
sw_rows, sw_acc = m.samplewise_cross_method_correlations(
    per_sample_7, methods_7, topology_methods_7, ['x', 'y', 'pd_distance'],
    {'x': 'higher_is_better', 'y': 'higher_is_better', 'pd_distance': 'lower_is_better'})
n_methods_xy = {r['n_methods'] for r in sw_rows if {r['metric_a'], r['metric_b']} == {'x', 'y'}}
n_methods_xpd = {r['n_methods'] for r in sw_rows if 'pd_distance' in (r['metric_a'], r['metric_b'])}
check('non-topology pair uses all 3 methods every sample', n_methods_xy == {3})
check('topology-involving pair uses exactly the 2 topology-bearing methods every sample', n_methods_xpd == {2})

# Injecting a NaN into a required cell must hard-fail (no partial cross-method sets).
per_sample_7b = make_synthetic_per_sample({
    'bicubic': {'x': xarr, 'y': xarr * 2},
    'ma': {'x': xarr + 1, 'y': xarr * 2 + 1},
    'mb': {'x': np.concatenate([[np.nan], (xarr + 2)[1:]]), 'y': xarr * 2 + 2},
})
try:
    m.samplewise_cross_method_correlations(per_sample_7b, methods_7[:0] + ['bicubic', 'ma', 'mb'], [],
                                              ['x', 'y'], {'x': 'higher_is_better', 'y': 'higher_is_better'})
    check('samplewise missing cell -> SystemExit', False)
except SystemExit as e:
    check('samplewise missing cell -> SystemExit', 'n=' in str(e))

print()
print('=== 8. Two-way-centered residual: exact demeaning, zero margins ===')
rng2 = np.random.default_rng(999)
method_effect = {'ma': 2.0, 'mb': -1.0, 'mc': 0.5}
sample_effect = rng2.normal(size=m.N_EVAL) * 0.01
specs_8 = {}
for mid, eff in method_effect.items():
    specs_8[mid] = {'z': eff + sample_effect}
per_sample_8 = make_synthetic_per_sample(specs_8)
R8, mmargin_8, smargin_8, gmargin_8 = m.compute_two_way_residual_matrix(
    per_sample_8, ['ma', 'mb', 'mc'], 'z', 'higher_is_better')
check('two-way residual method (row) margin ~0', mmargin_8 < 1e-9)
check('two-way residual sample (col) margin ~0', smargin_8 < 1e-9)
check('two-way residual grand margin ~0', gmargin_8 < 1e-9)
# Since z = method_effect + sample_effect (purely additive, no interaction),
# the residual should be exactly (numerically) zero everywhere.
check('purely-additive synthetic signal -> residual ~0 everywhere', np.max(np.abs(R8)) < 1e-9)

print()
print('=== 8b. Pair-specific common-rectangle centering (bug demo + fix) ===')
# Metric A is defined for 4 methods; metric B is only defined for 3 of them
# (mirrors bicubic-style total absence, e.g. PD/MT). The bug being fixed:
# centering A over all 4 methods and THEN subsetting the residual down to the
# 3 common methods leaves nonzero margins, because the row/column means used
# for centering were computed over the wrong (too-large) rectangle.
methods4 = ['m1', 'm2', 'm3', 'm4']
common3 = ['m1', 'm2', 'm3']
rng4 = np.random.default_rng(777)
interaction = rng4.normal(size=(3, m.N_EVAL))
method_effect_a = np.array([1.0, 2.0, -1.0])
method_effect_b = np.array([0.5, -0.5, 2.0])
sample_effect_a = rng4.normal(size=m.N_EVAL) * 0.1
sample_effect_b = rng4.normal(size=m.N_EVAL) * 0.1
k = -3.0
Z_a_common = method_effect_a[:, None] + sample_effect_a[None, :] + interaction
Z_b_common = method_effect_b[:, None] + sample_effect_b[None, :] + k * interaction
Z_a_4th = np.full(m.N_EVAL, 9.0)  # method m4 has metric A but not metric B

specs_res = {}
for i, mid in enumerate(common3):
    specs_res[mid] = {'aM2': Z_a_common[i], 'bM2': Z_b_common[i]}
specs_res['m4'] = {'aM2': Z_a_4th, 'bM2': None}
per_sample_res = make_synthetic_per_sample(specs_res)

# BUGGY approach (what the old code effectively did): center metric A over
# all 4 methods, then subset the residual rows down to the 3 common methods.
# NOTE: the per-method (row) margin of the subset is unaffected by row
# subsetting (each row's own mean was already subtracted exactly), so the
# bug specifically shows up as a nonzero PER-SAMPLE (column) margin once a
# row is dropped -- the column means baked into the full-4-method residual
# no longer match the 3-method subset they are being applied to.
R_a_full, _, _, _ = m.compute_two_way_residual_matrix(per_sample_res, methods4, 'aM2', 'higher_is_better')
R_a_subset_after = R_a_full[:3, :]
row_margin_after_subset = float(np.max(np.abs(R_a_subset_after.mean(axis=1))))
sample_margin_after_subset = float(np.max(np.abs(R_a_subset_after.mean(axis=0))))
check('row (per-method) margin is unaffected by post-hoc row subsetting (each row was already exactly demeaned)',
      row_margin_after_subset < 1e-9)
check('centering over 4 then subsetting to 3 produces a NONzero per-sample (column) margin '
      '(demonstrates the bug: the baked-in column means no longer match the 3-method subset)',
      sample_margin_after_subset > 1e-3)

# CORRECT approach: pair-specific centering, both metrics over the SAME
# common 3 methods, computed directly on that rectangle.
R_a, mmargin_a, smargin_a, gmargin_a = m.compute_two_way_residual_matrix(
    per_sample_res, common3, 'aM2', 'higher_is_better')
R_b, mmargin_b, smargin_b, gmargin_b = m.compute_two_way_residual_matrix(
    per_sample_res, common3, 'bM2', 'higher_is_better')
check('pair-specific centering over the common 3 methods: metric A margins ~0',
      mmargin_a < 1e-9 and smargin_a < 1e-9 and gmargin_a < 1e-9)
check('pair-specific centering over the common 3 methods: metric B margins ~0',
      mmargin_b < 1e-9 and smargin_b < 1e-9 and gmargin_b < 1e-9)
resid_corr = m.pearson_r(R_a.reshape(-1), R_b.reshape(-1))
check('pair-specific residual correlation is strongly negative as designed (shared interaction term, k=-3)',
      resid_corr < -0.9)

print()
print('=== 9. PD/MT pairwise and per-sample preference classification ===')
n_ev = m.N_EVAL
pd_a = np.linspace(0, 1, n_ev)
pd_b = pd_a + 1.0  # a always has smaller (better) pd
mt_a = np.linspace(0, 1, n_ev)
mt_b = mt_a + 1.0  # a always has smaller (better) mt too -> full agreement
per_sample_9a = make_synthetic_per_sample({
    'ta': {'pd_distance': pd_a, 'mt_distance': mt_a},
    'tb': {'pd_distance': pd_b, 'mt_distance': mt_b},
})
pref_rows_agree = m.topology_pairwise_preference_agreement(per_sample_9a, ['ta', 'tb'])
check('same non-tied preference every field -> descriptor_agreement_rate == 1.0',
      abs(pref_rows_agree[0]['descriptor_agreement_rate'] - 1.0) < 1e-12)
check('same non-tied preference every field -> descriptor_disagreement_rate == 0.0',
      abs(pref_rows_agree[0]['descriptor_disagreement_rate'] - 0.0) < 1e-12)
check('same non-tied preference every field -> descriptor_tie_or_undefined_count == 0',
      pref_rows_agree[0]['descriptor_tie_or_undefined_count'] == 0)

per_sample_9b = make_synthetic_per_sample({
    'ta': {'pd_distance': pd_a, 'mt_distance': mt_b},  # a better on PD, worse on MT
    'tb': {'pd_distance': pd_b, 'mt_distance': mt_a},
})
pref_rows_disagree = m.topology_pairwise_preference_agreement(per_sample_9b, ['ta', 'tb'])
check('opposite non-tied preference every field -> descriptor_disagreement_rate == 1.0',
      abs(pref_rows_disagree[0]['descriptor_disagreement_rate'] - 1.0) < 1e-12)
check('opposite non-tied preference every field -> descriptor_agreement_rate == 0.0',
      abs(pref_rows_disagree[0]['descriptor_agreement_rate'] - 0.0) < 1e-12)

# PD tie only (MT non-tied): must land in tie_or_undefined, NOT disagreement,
# even though PD itself has no preference to compare.
pd_tied = np.zeros(n_ev)  # ta == tb on PD -> tie every field
per_sample_9c = make_synthetic_per_sample({
    'ta': {'pd_distance': pd_tied, 'mt_distance': mt_a},
    'tb': {'pd_distance': pd_tied, 'mt_distance': mt_b},
})
pref_rows_pd_tie = m.topology_pairwise_preference_agreement(per_sample_9c, ['ta', 'tb'])
check('PD tie only -> all fields classified tie_or_undefined (not disagreement)',
      pref_rows_pd_tie[0]['descriptor_tie_or_undefined_count'] == n_ev
      and pref_rows_pd_tie[0]['descriptor_disagreement_count'] == 0
      and pref_rows_pd_tie[0]['descriptor_agreement_count'] == 0)
check('PD tie only -> pd_tie_count == N_EVAL', pref_rows_pd_tie[0]['pd_tie_count'] == n_ev)

# MT tie only (PD non-tied): same requirement, other descriptor tied.
mt_tied = np.zeros(n_ev)
per_sample_9d = make_synthetic_per_sample({
    'ta': {'pd_distance': pd_a, 'mt_distance': mt_tied},
    'tb': {'pd_distance': pd_b, 'mt_distance': mt_tied},
})
pref_rows_mt_tie = m.topology_pairwise_preference_agreement(per_sample_9d, ['ta', 'tb'])
check('MT tie only -> all fields classified tie_or_undefined (not disagreement)',
      pref_rows_mt_tie[0]['descriptor_tie_or_undefined_count'] == n_ev
      and pref_rows_mt_tie[0]['descriptor_disagreement_count'] == 0
      and pref_rows_mt_tie[0]['descriptor_agreement_count'] == 0)
check('MT tie only -> mt_tie_count == N_EVAL', pref_rows_mt_tie[0]['mt_tie_count'] == n_ev)

# Both descriptors tied: must land in tie_or_undefined, NOT agreement.
per_sample_9e = make_synthetic_per_sample({
    'ta': {'pd_distance': pd_tied, 'mt_distance': mt_tied},
    'tb': {'pd_distance': pd_tied, 'mt_distance': mt_tied},
})
pref_rows_both_tie = m.topology_pairwise_preference_agreement(per_sample_9e, ['ta', 'tb'])
check('both descriptors tied -> all fields classified tie_or_undefined (NOT agreement)',
      pref_rows_both_tie[0]['descriptor_tie_or_undefined_count'] == n_ev
      and pref_rows_both_tie[0]['descriptor_agreement_count'] == 0)

check('preference counts sum to N_EVAL',
      pref_rows_agree[0]['pd_prefers_a_count'] + pref_rows_agree[0]['pd_prefers_b_count'] +
      pref_rows_agree[0]['pd_tie_count'] == m.N_EVAL)
for rows in (pref_rows_agree, pref_rows_disagree, pref_rows_pd_tie, pref_rows_mt_tie, pref_rows_both_tie):
    r = rows[0]
    check('three-way classification counts sum to N_EVAL',
          r['descriptor_agreement_count'] + r['descriptor_disagreement_count'] +
          r['descriptor_tie_or_undefined_count'] == m.N_EVAL)

# Per-sample aggregation, plus the cross-reference to samplewise_cross_method_correlations.
samplewise_rows_9, _ = m.samplewise_cross_method_correlations(
    per_sample_9a, ['ta', 'tb'], ['ta', 'tb'], ['pd_distance', 'mt_distance'],
    {'pd_distance': 'lower_is_better', 'mt_distance': 'lower_is_better'})
sample_pref_rows = m.topology_sample_preference_agreement(per_sample_9a, ['ta', 'tb'], samplewise_rows_9)
check('sample-level rows use method_pair_count == C(2,2) == 1',
      all(r['method_pair_count'] == 1 for r in sample_pref_rows))
check('sample-level fully-agreeing case: every sample agreement_rate == 1.0',
      all(abs(r['agreement_rate'] - 1.0) < 1e-12 for r in sample_pref_rows))
pd_mt_lookup_9 = {r['sample_idx']: (r['oriented_pearson'], r['oriented_spearman']) for r in samplewise_rows_9}
check('sample-level pd_mt_cross_method_pearson/spearman exactly match samplewise_cross_method_correlations',
      all((r['pd_mt_cross_method_pearson'], r['pd_mt_cross_method_spearman']) ==
          pd_mt_lookup_9[r['sample_idx']] for r in sample_pref_rows))

print()
print('=== 10. Pareto dominance edges (hand case) ===')
# 3 methods, 2 objectives (oriented, higher-is-better). A dominates C; B and C
# are non-dominated w.r.t. each other (tradeoff); A does not dominate B.
oriented_10 = {
    'A': {'o1': 3.0, 'o2': 3.0},
    'B': {'o1': 1.0, 'o2': 5.0},
    'C': {'o1': 2.0, 'o2': 2.0},
}
edges_10, front_10 = m.compute_dominance_edges(oriented_10, ['A', 'B', 'C'], ['o1', 'o2'], m.PARETO_TOLERANCE)
check('A dominates C', ('A', 'C') in edges_10)
check('A does not dominate B (tradeoff)', ('A', 'B') not in edges_10)
check('B does not dominate A (tradeoff)', ('B', 'A') not in edges_10)
check('front is {A, B}', set(front_10) == {'A', 'B'})

print()
print('=== 11. Iterative non-domination layers ===')
# Linear chain: D dominates C dominates B dominates A on a single objective.
oriented_11 = {'A': {'o1': 1.0}, 'B': {'o1': 2.0}, 'C': {'o1': 3.0}, 'D': {'o1': 4.0}}
layers_11 = m.compute_layers(oriented_11, ['A', 'B', 'C', 'D'], ['o1'], m.PARETO_TOLERANCE)
layer_of = {r['method_id']: r['layer'] for r in layers_11}
check('single-objective chain: 4 distinct layers', len(set(layer_of.values())) == 4)
check('single-objective chain: D (best) is layer 0', layer_of['D'] == 0)
check('single-objective chain: A (worst) is layer 3', layer_of['A'] == 3)
check('every method appears in exactly one layer', sorted(layer_of) == ['A', 'B', 'C', 'D'])

print()
print('=== 12. Tolerance handling in dominance ===')
tol = 1e-6
oriented_12 = {'X': {'o1': 1.0}, 'Y': {'o1': 1.0 + tol / 10}}  # within tolerance -> tie, no dominance
edges_12, front_12 = m.compute_dominance_edges(oriented_12, ['X', 'Y'], ['o1'], tol)
check('near-tie within tolerance -> no dominance edges', len(edges_12) == 0)
check('near-tie within tolerance -> both on front', set(front_12) == {'X', 'Y'})
oriented_12b = {'X': {'o1': 1.0}, 'Y': {'o1': 1.0 + tol * 10}}  # clearly outside tolerance
edges_12b, front_12b = m.compute_dominance_edges(oriented_12b, ['X', 'Y'], ['o1'], tol)
check('difference clearly outside tolerance -> dominance detected', ('Y', 'X') in edges_12b)

print()
print('=== 13. Missing-objective hard fail ===')
means_13 = {'m1': dict(raw={'x': 1.0}, oriented={'x': 1.0}), 'm2': dict(raw={'x': 2.0}, oriented={'x': 2.0})}
bogus_objective_sets = {'bogus_set': ['x', 'z_not_present']}
try:
    m.run_pareto_deterministic(means_13, ['m1', 'm2'], [], bogus_objective_sets)
    check('missing objective -> exception', False)
except KeyError:
    check('missing objective -> exception', True)

print()
print('=== 14. Duplicate-method detection in run_base_validation ===')
empty_refs = dict(validation=[], pairwise_repro=[], immutability=[])
empty_p2b_refs = dict(validation=[], immutability=[])
empty_p1_refs = dict(method_summary=[], topo_val=[], inventory=[])
long_table_dup = dict(n_rows=0, per_sample={}, metric_cols=[], method_meta={}, dup_keys=[('m1', 0)])
checks_dup, failures_dup = m.run_base_validation(long_table_dup, empty_p1_refs, empty_refs, empty_p2b_refs, {})
dup_check = next(c for c in checks_dup if c['check_name'] == 'duplicate_keys')
check('nonzero dup_keys -> duplicate_keys check FAILs', dup_check['status'] == 'FAIL')
long_table_nodup = dict(n_rows=0, per_sample={}, metric_cols=[], method_meta={}, dup_keys=[])
checks_nodup, _ = m.run_base_validation(long_table_nodup, empty_p1_refs, empty_refs, empty_p2b_refs, {})
dup_check2 = next(c for c in checks_nodup if c['check_name'] == 'duplicate_keys')
check('zero dup_keys -> duplicate_keys check PASSes', dup_check2['status'] == 'PASS')

print()
print('=== 15. Topology-only expected-front hard fail ===')
try:
    m.topology_pareto_sanity_check({'topology_only': frozenset({'wrong', 'set'})}, [], [], ['a', 'b'])
    check('wrong topology-only front -> SystemExit', False)
except SystemExit as e:
    check('wrong topology-only front -> SystemExit', 'sanity check failed' in str(e))

correct_front = m.EXPECTED_TOPOLOGY_ONLY_FRONT
all_topo_methods = sorted(correct_front) + ['other1', 'other2']
edges_correct = [dict(objective_set='topology_only', dominator=d, dominated=o)
                  for d in correct_front for o in ('other1', 'other2')]
layers_correct = [dict(objective_set='topology_only', method_id=mid, layer=(0 if mid in correct_front else 1))
                   for mid in all_topo_methods]
sanity_checks = m.topology_pareto_sanity_check({'topology_only': correct_front}, edges_correct, layers_correct,
                                                  all_topo_methods)
check('correct topology-only front -> returns without raising, all structural checks PASS',
      all(c['status'] == 'PASS' for c in sanity_checks))

print()
print('=== 16. Deterministic iid/block bootstrap index reproducibility ===')
idx_a = m._make_iid_index_matrix(42, 100)
idx_b = m._make_iid_index_matrix(42, 100)
check('_make_iid_index_matrix reproducible across calls', np.array_equal(idx_a, idx_b))
check('_make_iid_index_matrix indices within [0, N_EVAL)', idx_a.min() >= 0 and idx_a.max() < m.N_EVAL)
blk_a = m._make_block_index_matrix(42, 6, 100)
blk_b = m._make_block_index_matrix(42, 6, 100)
check('_make_block_index_matrix reproducible across calls', np.array_equal(blk_a, blk_b))
check('_make_block_index_matrix indices within [0, N_EVAL)', blk_a.min() >= 0 and blk_a.max() < m.N_EVAL)
check('module-level CORR and PARETO bootstrap index families use distinct seeds',
      m.CORR_BOOTSTRAP_SEED != m.PARETO_BOOTSTRAP_SEED)
# Manually verify circular wrapping for a known start position.
offsets = np.arange(6)
wrapped = (165 + offsets) % m.N_EVAL
check('circular wrap formula for start=165, block_length=6 gives [165,166,167,0,1,2]',
      list(wrapped) == [165, 166, 167, 0, 1, 2])

print()
print('=== 17. Paired bootstrap indices shared across methods/objectives ===')
count_matrix_test = m._bootstrap_count_matrix(m._make_iid_index_matrix(7, 50))
check('count matrix rows sum to N_EVAL (each replicate resamples N_EVAL indices)',
      np.allclose(count_matrix_test.sum(axis=1), m.N_EVAL))
raw_shared_metric = np.linspace(0, 1, m.N_EVAL)
raw_1obj = np.zeros((1, 1, m.N_EVAL))
raw_1obj[0, 0, :] = raw_shared_metric
raw_2obj = np.zeros((1, 2, m.N_EVAL))
raw_2obj[0, 0, :] = raw_shared_metric
raw_2obj[0, 1, :] = np.linspace(5, 10, m.N_EVAL)  # unrelated second objective
means_1obj = m._resample_means_matrix(raw_1obj, count_matrix_test)
means_2obj = m._resample_means_matrix(raw_2obj, count_matrix_test)
check('same metric bootstrapped with the same count matrix gives identical per-replicate means '
      'regardless of which other objectives share the objective set',
      np.allclose(means_1obj[:, 0, 0], means_2obj[:, 0, 0]))

print()
print('=== 18. Pareto bootstrap front-membership counting ===')
# 5 replicates, 3 methods, 1 objective: replicate r's winner is method (r % 3).
n_rep, n_m = 6, 3
oriented_18 = np.zeros((n_rep, n_m, 1))
winners = []
for r in range(n_rep):
    winner = r % n_m
    winners.append(winner)
    for mm in range(n_m):
        oriented_18[r, mm, 0] = 10.0 if mm == winner else float(mm)
front_counts_18, front_sizes_18 = m._chunked_front_membership(oriented_18, m.PARETO_TOLERANCE, chunk_size=4)
check('single-objective bootstrap: every replicate has front_size == 1',
      np.all(front_sizes_18 == 1))
check('single-objective bootstrap: front_counts sum to n_replicates',
      front_counts_18.sum() == n_rep)
expected_counts = np.array([winners.count(mm) for mm in range(n_m)])
check('single-objective bootstrap: front_counts match the known per-replicate winner',
      np.array_equal(front_counts_18, expected_counts))

print()
print('=== 19. Missing prior protected file -> hard fail ===')
target = m.PHASE1_PROTECTED_CSVS[0]
backup = target.with_suffix('.csv.bak_test2c')
try:
    target.rename(backup)
    try:
        m.require_protected_files()
        check('missing prior protected file -> SystemExit', False)
    except SystemExit as e:
        check('missing prior protected file -> SystemExit', str(target) in str(e))
finally:
    if backup.exists():
        backup.rename(target)
check('file restored', target.exists())

print()
print('=== 20. Unexpected extra CSV in a frozen directory -> hard fail ===')
extra_path = m.PHASE2B_DIR / '__phase2c_test_unexpected_extra.csv'
try:
    extra_path.write_text('a,b\n1,2\n')
    try:
        m.require_protected_files()
        check('unexpected extra CSV -> SystemExit', False)
    except SystemExit as e:
        check('unexpected extra CSV -> SystemExit', 'Unexpected extra CSV' in str(e))
finally:
    if extra_path.exists():
        extra_path.unlink()
check('extra test file removed', not extra_path.exists())

print()
print('=== 21. Altered prior checksum -> hard fail (detected via checksum_all comparison) ===')
target2 = m.PHASE2A_PROTECTED_CSVS[0]
original_bytes = target2.read_bytes()
before = m.checksum_all([target2])
try:
    target2.write_bytes(original_bytes + b'\n# tampered\n')
    after = m.checksum_all([target2])
    check('altered file -> checksum differs from before', before != after)
finally:
    target2.write_bytes(original_bytes)
restored = m.checksum_all([target2])
check('file restored to original checksum', restored == before)

print()
print('=== 22. Protected-file-list counts ===')
check('Phase-1 protected files == 12', len(m.PHASE1_PROTECTED_FILES) == 12)
check('Phase-2A protected files == 14', len(m.PHASE2A_PROTECTED_FILES) == 14)
check('Phase-2B protected files == 28', len(m.PHASE2B_PROTECTED_FILES) == 28)
check('All protected files == 54', len(m.ALL_PROTECTED_FILES) == 54)

print()
print('=== 23. topology_rank_by_method: hand-computed method-mean ranks with ties ===')
means_rank_test = {
    'ma': dict(raw={'pd_distance': 1.0, 'mt_distance': 5.0}, oriented={}),
    'mb': dict(raw={'pd_distance': 2.0, 'mt_distance': 2.0}, oriented={}),
    'mc': dict(raw={'pd_distance': 2.0, 'mt_distance': 2.0}, oriented={}),  # tied with mb on both
    'md': dict(raw={'pd_distance': 4.0, 'mt_distance': 1.0}, oriented={}),
}
rank_rows_test = m.topology_rank_by_method(means_rank_test, ['ma', 'mb', 'mc', 'md'],
                                              {'ma': 'Method A'})
check('topology_rank_by_method returns exactly len(topology_methods) rows', len(rank_rows_test) == 4)
by_id = {r['method_id']: r for r in rank_rows_test}
check('display_name is looked up (present) or empty (absent) rather than erroring',
      by_id['ma']['display_name'] == 'Method A' and by_id['mb']['display_name'] == '')
# pd ascending: ma=1(rank1), mb=2,mc=2 (tie -> average rank 2.5 each), md=4(rank4)
check('pd rank for ma (smallest distance) is 1', abs(by_id['ma']['pd_rank'] - 1.0) < 1e-9)
check('pd tie ranks (mb, mc) average to 2.5', abs(by_id['mb']['pd_rank'] - 2.5) < 1e-9
      and abs(by_id['mc']['pd_rank'] - 2.5) < 1e-9)
check('pd rank for md (largest distance) is 4', abs(by_id['md']['pd_rank'] - 4.0) < 1e-9)
# mt ascending: md=1(rank1), mb=2,mc=2(tie->2.5), ma=5(rank4)
check('mt rank for md (smallest distance) is 1', abs(by_id['md']['mt_rank'] - 1.0) < 1e-9)
check('mt tie ranks (mb, mc) average to 2.5', abs(by_id['mb']['mt_rank'] - 2.5) < 1e-9
      and abs(by_id['mc']['mt_rank'] - 2.5) < 1e-9)
check('mt rank for ma (largest distance) is 4', abs(by_id['ma']['mt_rank'] - 4.0) < 1e-9)
check('signed_rank_gap = pd_rank - mt_rank for ma (1 - 4 = -3, PD favors ma much more)',
      abs(by_id['ma']['signed_rank_gap'] - (-3.0)) < 1e-9)
check('signed_rank_gap for md (4 - 1 = 3, MT favors md much more)',
      abs(by_id['md']['signed_rank_gap'] - 3.0) < 1e-9)
check('absolute_rank_gap is abs(signed_rank_gap) for every row',
      all(abs(r['absolute_rank_gap'] - abs(r['signed_rank_gap'])) < 1e-9 for r in rank_rows_test))
check('pd_mean/mt_mean pass through the raw method means unchanged',
      abs(by_id['ma']['pd_mean'] - 1.0) < 1e-12 and abs(by_id['ma']['mt_mean'] - 5.0) < 1e-12)

print()
print('=== 24. Focal bootstrap wide-form: row shape, CI reproduction, interval signs ===')
rng5 = np.random.default_rng(2024)
focal_specs = {'tm': {fm: rng5.normal(size=m.N_EVAL) for fm in m.FOCAL_METRICS}}
per_sample_focal = make_synthetic_per_sample(focal_specs)
directions_focal = {fm: ('higher_is_better' if fm == 'psnruv' else 'lower_is_better') for fm in m.FOCAL_METRICS}
focal_rows_test = m.focal_topology_correlation_bootstrap(per_sample_focal, ['tm'], directions_focal)
check('wide-form row count scales as 1 method x 15 pairs x 2 types == 30', len(focal_rows_test) == 30)

check('_interval_sign: low > 0 -> positive', m._interval_sign(0.1, 0.5) == 'positive')
check('_interval_sign: high < 0 -> negative', m._interval_sign(-0.5, -0.1) == 'negative')
check('_interval_sign: straddles zero -> includes_zero', m._interval_sign(-0.1, 0.1) == 'includes_zero')
check('_interval_sign: low == 0 (boundary) -> includes_zero, not positive',
      m._interval_sign(0.0, 0.5) == 'includes_zero')
check('_interval_sign: high == 0 (boundary) -> includes_zero, not negative',
      m._interval_sign(-0.5, 0.0) == 'includes_zero')

for r in focal_rows_test:
    signs = {r['iid_sign'], r['block6_sign'], r['block12_sign'], r['block24_sign']}
    expected_agree = (len(signs) == 1 and '' not in signs)
    check(f"all_interval_signs_agree correct for {r['metric_a']}/{r['metric_b']}/{r['correlation_type']}",
          r['all_interval_signs_agree'] == expected_agree)
    for scheme in m.BOOTSTRAP_SCHEME_NAMES:
        lo, hi = r[f'{scheme}_ci95_low'], r[f'{scheme}_ci95_high']
        check(f"{scheme}_sign matches _interval_sign(lo, hi) for {r['metric_a']}/{r['metric_b']}",
              r[f'{scheme}_sign'] == m._interval_sign(lo, hi))

row_check = next(r for r in focal_rows_test
                  if r['metric_a'] == 'pd_distance' and r['metric_b'] == 'psnruv'
                  and r['correlation_type'] == 'pearson')
ox_check = m.orient(np.array([per_sample_focal['tm'][si]['pd_distance'] for si in range(m.N_EVAL)]),
                      'lower_is_better')
oy_check = m.orient(np.array([per_sample_focal['tm'][si]['psnruv'] for si in range(m.N_EVAL)]),
                      'higher_is_better')
ci_direct = m.correlation_bootstrap_cis(ox_check, oy_check, 'pearson')
check('wide-form observed_correlation matches a direct pearson_r call',
      abs(row_check['observed_correlation'] - m.pearson_r(ox_check, oy_check)) < 1e-12)
for scheme in m.BOOTSTRAP_SCHEME_NAMES:
    lo_direct, hi_direct = ci_direct[scheme]
    check(f'wide-form {scheme} CI reproduces a direct correlation_bootstrap_cis() call exactly',
          abs(row_check[f'{scheme}_ci95_low'] - lo_direct) < 1e-12
          and abs(row_check[f'{scheme}_ci95_high'] - hi_direct) < 1e-12)

print()
if failures:
    print(f'{len(failures)} FAILURE(S): {failures}')
    sys.exit(1)
else:
    print('ALL TESTS PASSED')
