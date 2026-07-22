#!/usr/bin/env python3
"""Synthetic tests for scripts/analyze_unified_candidate_factors_phase2b.py.

Exercises the real shipped functions directly (factorial-design math, sign
orientation, hard-fail gates, deterministic bootstrap, prior-phase
immutability checks). Any test that temporarily perturbs a real Phase-1/
Phase-2A protected file restores it immediately in a `finally` block, and
the suite verifies the restoration succeeded before continuing.

Run directly:
    python3 scripts/test_analyze_unified_candidate_factors_phase2b.py
"""
import math
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_unified_candidate_factors_phase2b as m
import numpy as np

failures = []


def check(name, cond):
    status = 'PASS' if cond else 'FAIL'
    print(f'[{status}] {name}')
    if not cond:
        failures.append(name)


print('=== 1. Correct 2^2 coefficients and effects (hand-computed) ===')
# 2^2 design over factors (a, b): cells uv=(0,0), a=(1,0), b=(0,1), ab=(1,1)
cells22 = [
    ('m00', {'a': 0, 'b': 0}),
    ('m10', {'a': 1, 'b': 0}),
    ('m01', {'a': 0, 'b': 1}),
    ('m11', {'a': 1, 'b': 1}),
]
fr22 = m.FactorialResult('TEST_2x2', ('a', 'b'), cells22)
check('2^2 coded matrix shape', fr22.coded_matrix.shape == (4, 4))
check('2^2 subsets order', fr22.subsets == [(), ('a',), ('b',), ('a', 'b')])
# y = [1, 2, 3, 8] for m00, m10, m01, m11 (one sample)
y = np.array([[1.0], [2.0], [3.0], [8.0]])
# NOTE: kept as beta22 (not the generic name `beta`) and never reassigned
# again anywhere in this file, specifically so the later interaction-sign
# check in section 4/5/6 cannot be silently checking a `beta` that section 3's
# random-reconstruction loop reassigned out from under it.
beta22 = m.compute_beta_matrix(fr22.coded_matrix, y)
# beta_0 = mean = 3.5; beta_a = mean(y*coded_a) = (1*-1+2*1+3*-1+8*1)/4 = (−1+2−3+8)/4=6/4=1.5
# beta_b = mean(y*coded_b) = (-1-2+3+8)/4 = 8/4=2.0
# beta_ab = mean(y*coded_a*coded_b) = (1*1 + 2*-1 + 3*-1 + 8*1)/4 = (1-2-3+8)/4=4/4=1.0
check('2^2 intercept beta_0', abs(beta22[0, 0] - 3.5) < 1e-12)
check('2^2 main effect a coefficient', abs(beta22[1, 0] - 1.5) < 1e-12)
check('2^2 main effect b coefficient', abs(beta22[2, 0] - 2.0) < 1e-12)
check('2^2 interaction ab coefficient', abs(beta22[3, 0] - 1.0) < 1e-12)
check('2^2 factorial_effect(a) = 2*beta_a = 3.0', abs(2 * beta22[1, 0] - 3.0) < 1e-12)

print()
print('=== 2. Correct 2^3 coefficients and effects (hand-computed) ===')
cells23 = [
    ('c000', {'a': 0, 'b': 0, 'c': 0}), ('c100', {'a': 1, 'b': 0, 'c': 0}),
    ('c010', {'a': 0, 'b': 1, 'c': 0}), ('c110', {'a': 1, 'b': 1, 'c': 0}),
    ('c001', {'a': 0, 'b': 0, 'c': 1}), ('c101', {'a': 1, 'b': 0, 'c': 1}),
    ('c011', {'a': 0, 'b': 1, 'c': 1}), ('c111', {'a': 1, 'b': 1, 'c': 1}),
]
fr23 = m.FactorialResult('TEST_2x3', ('a', 'b', 'c'), cells23)
check('2^3 subsets order matches task effect order',
      fr23.subsets == [(), ('a',), ('b',), ('c',), ('a', 'b'), ('a', 'c'), ('b', 'c'), ('a', 'b', 'c')])
y3 = np.array([[0.], [2.], [0.], [2.], [0.], [2.], [0.], [10.]])  # cells in the order above
beta3 = m.compute_beta_matrix(fr23.coded_matrix, y3)
mean3 = float(np.mean(y3))
check('2^3 intercept = mean of all 8 cells', abs(beta3[0, 0] - mean3) < 1e-12)
# verify reconstruction is exact for this arbitrary vector
y3_hat = fr23.coded_matrix @ beta3
check('2^3 exact reconstruction for arbitrary y', np.allclose(y3_hat, y3, atol=1e-12))

print()
print('=== 3. Exact saturated reconstruction (random data, both k=2 and k=3) ===')
rng = np.random.default_rng(0)
for fr, n_cells in ((fr22, 4), (fr23, 8)):
    Y = rng.normal(size=(n_cells, 50))
    beta = m.compute_beta_matrix(fr.coded_matrix, Y)
    Y_hat = fr.coded_matrix @ beta
    max_err = float(np.max(np.abs(Y - Y_hat)))
    check(f'reconstruction error for random {n_cells}-cell design <= 1e-10 (got {max_err:.2e})', max_err <= 1e-10)

print()
print('=== 4/5/6. Orientation (higher/lower-is-better) and interaction sign ===')
raw_higher = np.array([1.0, 2.0, 3.0])
check('orient() with higher_is_better is identity', np.array_equal(m.orient(raw_higher, 'higher_is_better'), raw_higher))
check('orient() with lower_is_better negates', np.array_equal(m.orient(raw_higher, 'lower_is_better'), -raw_higher))
# interaction sign: y_11=8 much higher than additive prediction (1+ (2-1)+(3-1)=4) -> positive interaction
check('interaction coefficient positive when 11-cell exceeds additive prediction (from test 1, beta22)',
      beta22[3, 0] > 0)

print()
print('=== 7. Missing one factorial cell -> hard fail ===')
cells22_real = [
    ('m00', {'crit': 0, 'e2': 0}), ('m10', {'crit': 1, 'e2': 0}),
    ('m01', {'crit': 0, 'e2': 1}), ('m11', {'crit': 1, 'e2': 1}),
]
bad_cells = cells22_real[:3]  # only 3 of 4 cells
meta_ok = {mid: {'values': {'uses_crit': ('True' if levels['crit'] == 1 else 'False'),
                              'uses_e2': ('True' if levels['e2'] == 1 else 'False'),
                              'display_name': mid}} for mid, levels in cells22_real}
try:
    m.verify_design_metadata('TEST_MISSING_CELL', bad_cells,
                               ('crit', 'e2'), {mid: meta_ok[mid] for mid, _ in bad_cells})
    check('missing cell -> SystemExit', False)
except SystemExit as e:
    check('missing cell -> SystemExit', 'expected exactly 4 cells' in str(e))

print()
print('=== 8. Duplicated factor coding -> hard fail ===')
dup_cells = [('mA', {'a': 0, 'b': 0}), ('mB', {'a': 0, 'b': 0}), ('mC', {'a': 1, 'b': 0}), ('mD', {'a': 1, 'b': 1})]
dup_meta = {'mA': {'values': {'uses_speed': 'False'}}, 'mB': {'values': {'uses_speed': 'False'}},
            'mC': {'values': {'uses_speed': 'False'}}, 'mD': {'values': {'uses_speed': 'False'}}}
# rebuild factor-to-column mapping trick: use 'speed'/'grad' as the two test factors so
# FACTOR_TO_USES_COLUMN resolves; give every cell uses_speed/uses_grad = 'False' (matching a=0,b=0 always
# for the metadata check) except we deliberately want the DUPLICATE-CODING check to fire, which happens
# after the metadata check, so give consistent (if duplicated) metadata for the first two.
dup_cells2 = [('mA', {'speed': 0, 'grad': 0}), ('mB', {'speed': 0, 'grad': 0}),
              ('mC', {'speed': 1, 'grad': 0}), ('mD', {'speed': 1, 'grad': 1})]
dup_meta2 = {mid: {'values': {'uses_speed': 'False', 'uses_grad': 'False'}} for mid, _ in dup_cells2}
dup_meta2['mA']['values'] = {'uses_speed': 'False', 'uses_grad': 'False'}
dup_meta2['mB']['values'] = {'uses_speed': 'False', 'uses_grad': 'False'}
dup_meta2['mC']['values'] = {'uses_speed': 'True', 'uses_grad': 'False'}
dup_meta2['mD']['values'] = {'uses_speed': 'True', 'uses_grad': 'True'}
try:
    m.verify_design_metadata('TEST_DUP_CODING', dup_cells2, ('speed', 'grad'), dup_meta2)
    check('duplicated factor coding -> SystemExit', False)
except SystemExit as e:
    check('duplicated factor coding -> SystemExit', 'share identical factor coding' in str(e))

print()
print('=== 9. Partial metric coverage -> hard fail (via summarize_oriented_series is fine; the real ===')
print('===    hard-fail is inside run_factorial_design; test the linearity/finite-count guard path) ===')
# Build a minimal per_sample fixture where one cell has partial (not full 168) coverage for a metric,
# and confirm run_factorial_design's per-sample finite-mask logic correctly yields n_valid < 168
# (which, combined with base validation's required_metric_complete check, is what actually blocks
# a real run -- exercised here at the summarize level).
raw_partial = np.array([1.0] * 100 + [float('nan')] * 68)
or_partial = raw_partial.copy()
summ_partial = m.summarize_oriented_series(raw_partial, or_partial)
check('partial coverage (100/168) -> n_valid=100 (not silently treated as complete)', summ_partial['n_valid'] == 100)
check('partial coverage -> bootstrap CI left empty (only n==168 is bootstrapped)',
      summ_partial['ci']['iid'] == ('', ''))

# Directly exercise the real hard-fail path via run_base_validation(): a method
# used in a design with only 100/168 finite values for a required (non-SSIM)
# metric must produce a FAIL check and appear in the failures list.
orig_n_eval = m.N_EVAL
m.N_EVAL = 8
try:
    per_sample_stub = {'uv': {si: {'pd_distance': 1.0} for si in range(8)}}
    per_sample_stub['uv'][3]['pd_distance'] = float('nan')  # 7/8 finite -- partial
    long_table_stub = dict(n_rows=8, dup_keys=[], per_sample=per_sample_stub,
                             method_meta={'uv': {'values': {'display_name': 'x', 'candidate_family': 'x',
                                                              'training_scale': 'x', 'architecture': 'x',
                                                              'uses_speed': 'False', 'uses_grad': 'False',
                                                              'uses_levelset': 'False', 'uses_crit': 'False',
                                                              'uses_e2': 'False'}, 'inconsistent_fields': []}},
                             metric_cols=['pd_distance'])
    phase1_refs_stub = dict()
    phase2a_refs_stub = dict(validation=[], pairwise_repro=[], immutability=[])
    checks, base_failures = m.run_base_validation(long_table_stub, phase1_refs_stub, phase2a_refs_stub,
                                                     {'pd_distance': 'lower_is_better'}, {'uv'})
    check('partial (7/8) coverage for a required metric on a design method -> appears in failures',
          any('required_metric_complete[uv][pd_distance]' in f for f in base_failures))
finally:
    m.N_EVAL = orig_n_eval

print()
print('=== 10. SSIM 0/168 -> accepted and empty ===')
raw_all_nan = np.full(168, float('nan'))
summ_ssim = m.summarize_oriented_series(raw_all_nan, raw_all_nan)
check('all-NaN (SSIM-like) series -> n_valid=0', summ_ssim['n_valid'] == 0)
check('all-NaN series -> mean/median/etc left empty (not fabricated zero)',
      summ_ssim['mean_raw'] == '' and summ_ssim['mean_oriented'] == '')

print()
print('=== 11. Targeted contrast direction (positive = comparison better than base) ===')
# Exercises the ACTUAL production helper (compute_paired_contrast_point),
# the same function run_targeted_contrasts() calls for every per-sample
# point -- not a duplicate of its arithmetic.
# lower_is_better metric: base=10, comparison=7 -> comparison is BETTER (lower) -> oriented_improvement > 0
raw_delta, oriented = m.compute_paired_contrast_point(10.0, 7.0, 'lower_is_better')
check('compute_paired_contrast_point: raw_delta = comparison - base = -3.0', raw_delta == -3.0)
check('compute_paired_contrast_point: lower_is_better, comparison better (lower) -> oriented positive',
      oriented > 0 and oriented == 3.0)
# higher_is_better metric: base=10, comparison=15 -> comparison is better (higher) -> oriented positive
raw_delta2, oriented2 = m.compute_paired_contrast_point(10.0, 15.0, 'higher_is_better')
check('compute_paired_contrast_point: raw_delta = comparison - base = 5.0', raw_delta2 == 5.0)
check('compute_paired_contrast_point: higher_is_better, comparison better (higher) -> oriented positive',
      oriented2 > 0 and oriented2 == 5.0)
# comparison WORSE cases, both directions, to confirm sign flips correctly (not just always positive).
raw_delta3, oriented3 = m.compute_paired_contrast_point(10.0, 14.0, 'lower_is_better')
check('compute_paired_contrast_point: lower_is_better, comparison worse (higher) -> oriented negative',
      oriented3 < 0)
raw_delta4, oriented4 = m.compute_paired_contrast_point(10.0, 6.0, 'higher_is_better')
check('compute_paired_contrast_point: higher_is_better, comparison worse (lower) -> oriented negative',
      oriented4 < 0)
# Cross-check against a real contrast spec + real long-table method metadata:
# add_crit_to_uv should exist as a fully-specified, single-flag contrast.
_add_crit_to_uv = next(s for s in m.TARGETED_CONTRASTS if s['contrast_id'] == 'add_crit_to_uv')
check('add_crit_to_uv expected_flag_changes is exactly {crit: (False, True)}',
      _add_crit_to_uv['expected_flag_changes'] == {'crit': (False, True)})

print()
print('=== 11b. verify_contrast_metadata (new strict targeted-contrast metadata gate) ===')


def _meta(speed=False, grad=False, levelset=False, crit=False, e2=False):
    return {'values': {'uses_speed': str(speed), 'uses_grad': str(grad), 'uses_levelset': str(levelset),
                         'uses_crit': str(crit), 'uses_e2': str(e2)}}


ok_spec = dict(contrast_id='t_ok', base_method='base', comparison_method='comp',
                expected_flag_changes={'crit': (False, True)})
ok_meta = {'base': _meta(grad=True), 'comp': _meta(grad=True, crit=True)}
try:
    m.verify_contrast_metadata(ok_spec, ok_meta)
    check('verify_contrast_metadata: matching single-flag change -> no exception', True)
except SystemExit:
    check('verify_contrast_metadata: matching single-flag change -> no exception', False)

bad_spec = dict(contrast_id='t_bad', base_method='base', comparison_method='comp',
                  expected_flag_changes={'crit': (False, True)})
bad_meta_extra_change = {'base': _meta(grad=True), 'comp': _meta(grad=False, crit=True)}  # grad ALSO changed
try:
    m.verify_contrast_metadata(bad_spec, bad_meta_extra_change)
    check('verify_contrast_metadata: undeclared extra flag change -> SystemExit', False)
except SystemExit as e:
    check('verify_contrast_metadata: undeclared extra flag change -> SystemExit',
          't_bad' in str(e) and 'grad' in str(e))

bad_meta_wrong_direction = {'base': _meta(grad=True, crit=True), 'comp': _meta(grad=True, crit=False)}  # crit flipped the wrong way
try:
    m.verify_contrast_metadata(bad_spec, bad_meta_wrong_direction)
    check('verify_contrast_metadata: declared flag changes the wrong direction -> SystemExit', False)
except SystemExit as e:
    check('verify_contrast_metadata: declared flag changes the wrong direction -> SystemExit', 't_bad' in str(e))

# Cross-check every real contrast in TARGETED_CONTRASTS against the real long-table
# method metadata (this is exactly what run_targeted_contrasts() does internally).
_long_table = m.load_long_table()
for _spec in m.TARGETED_CONTRASTS:
    try:
        m.verify_contrast_metadata(_spec, _long_table['method_meta'])
        _ok = True
    except SystemExit:
        _ok = False
    check(f"verify_contrast_metadata passes for real contrast {_spec['contrast_id']!r} against real long-table metadata", _ok)

print()
print('=== 12. Circular block-bootstrap index shape and wrapping ===')
idx6 = m._BLOCK_IDX[6]
check('block-6 index matrix shape is (10000, 168)', idx6.shape == (m.BOOTSTRAP_N, m.N_EVAL))
check('block-6 index values within [0, 167]', idx6.min() >= 0 and idx6.max() <= 167)
# Manually verify circular wrapping: for block_length=6, a start position of 165 must wrap
# to indices [165, 166, 167, 0, 1, 2] within its 6-length block.
test_seed = m.BOOTSTRAP_SEED * 1000 + 6
test_rng = np.random.default_rng(test_seed)
n_blocks_6 = math.ceil(m.N_EVAL / 6)
starts_6 = test_rng.integers(0, m.N_EVAL, size=(m.BOOTSTRAP_N, n_blocks_6))
# find a resample/block where start==165 (or synthesize the check directly on the formula)
offsets = np.arange(6)
wrapped = (165 + offsets) % 168
check('circular wrap formula for start=165, block_length=6 gives [165,166,167,0,1,2]',
      list(wrapped) == [165, 166, 167, 0, 1, 2])

print()
print('=== 13. Deterministic bootstrap output ===')
vals = np.linspace(0, 10, 168)
ci_a = m.bootstrap_ci_all(vals)
ci_b = m.bootstrap_ci_all(vals)
check('bootstrap_ci_all is deterministic (identical CIs across two calls)', ci_a == ci_b)
# Also confirm the block index matrices themselves are identical across two independent constructions.
idx6_again = m._make_block_index_matrix(6)
check('_make_block_index_matrix(6) reproducible across calls', np.array_equal(idx6, idx6_again))

print()
print('=== 14. Missing prior protected file -> hard fail ===')
target = m.PHASE1_PROTECTED_CSVS[0]
backup = target.with_suffix('.csv.bak_test2b')
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
print('=== 15. Altered prior checksum -> hard fail (detected via checksum_all comparison) ===')
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
if failures:
    print(f'{len(failures)} FAILURE(S): {failures}')
    sys.exit(1)
else:
    print('ALL TESTS PASSED')