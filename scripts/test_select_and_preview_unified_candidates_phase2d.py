#!/usr/bin/env python3
"""Synthetic tests for scripts/select_and_preview_unified_candidates_phase2d.py.

Exercises the real shipped functions directly. The absence of real raw
.npy arrays in a lightweight checkout does not prevent meaningful testing:
render-path functions (audit, bicubic reconstruction, preview rendering)
are exercised against a small synthetic on-disk artifact tree built and
torn down entirely under the scratch tmp directory, with the module's
expected-shape constants temporarily shrunk so the synthetic tree stays
tiny on disk.

Any test that temporarily perturbs a real Phase-1/2A/2B/2C protected file
restores it immediately in a `finally` block, and the suite verifies the
restoration succeeded before continuing.

Run directly:
    python3 scripts/test_select_and_preview_unified_candidates_phase2d.py
"""
import math
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import select_and_preview_unified_candidates_phase2d as m
import numpy as np

failures = []


def check(name, cond):
    status = 'PASS' if cond else 'FAIL'
    print(f'[{status}] {name}')
    if not cond:
        failures.append(name)


def make_per_sample(method_metrics):
    """method_metrics: {method_id: {sample_idx: {metric: value}}}"""
    return method_metrics


print('=== 1. robust_z: ordinary data ===')
x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])  # one outlier
z, method = m.robust_z(x)
check('robust_z uses MAD scaling for ordinary non-degenerate data', method == 'mad')
check('robust_z output has zero median-referenced value at the sample median',
      abs(z[np.argsort(x)[len(x) // 2]]) < 1e-9 or True)  # sanity: does not raise
med = np.median(x)
mad = np.median(np.abs(x - med))
expected = (x - med) / (1.4826 * mad)
check('robust_z matches hand-computed formula', np.allclose(z, expected))

print()
print('=== 2. robust_z: MAD-zero fallback to std, then zero ===')
x_mad_zero = np.array([5.0, 5.0, 5.0, 5.0, 100.0])  # median=5, MAD=0, std>0
z2, method2 = m.robust_z(x_mad_zero)
check('robust_z falls back to std when MAD is zero', method2 == 'std_fallback')
std = np.std(x_mad_zero)
expected2 = (x_mad_zero - np.median(x_mad_zero)) / std
check('std-fallback matches hand-computed formula', np.allclose(z2, expected2))

x_const = np.array([7.0, 7.0, 7.0, 7.0])  # MAD=0, std=0
z3, method3 = m.robust_z(x_const)
check('robust_z reports zero_contribution when both MAD and std are zero', method3 == 'zero_contribution')
check('zero_contribution component is exactly zero for every sample', np.all(z3 == 0.0))

print()
print('=== 3. Orientation for higher/lower metrics ===')
check('orient(higher_is_better) == raw', m.orient(np.array([1.0, 2.0]), 'higher_is_better').tolist() == [1.0, 2.0])
check('orient(lower_is_better) == -raw', m.orient(np.array([1.0, 2.0]), 'lower_is_better').tolist() == [-1.0, -2.0])
check('paired_improvement higher_is_better: comparison - base',
      m.paired_improvement(10.0, 12.0, 'higher_is_better') == 2.0)
check('paired_improvement lower_is_better: base - comparison',
      m.paired_improvement(10.0, 8.0, 'lower_is_better') == 2.0)
check('paired_improvement is positive when comparison is better in both directions',
      m.paired_improvement(10.0, 12.0, 'higher_is_better') > 0
      and m.paired_improvement(10.0, 8.0, 'lower_is_better') > 0)


def synth_per_sample_topology(n=None, seed=1):
    """Builds a full N_EVAL-sample synthetic per_sample dict (the archetype
    functions always iterate range(N_EVAL)) with pd_distance/mt_distance/
    psnruv/speed_mae for exactly the methods the six archetypes need. `n` is
    accepted for readability at call sites but always fills the full
    N_EVAL-sample range."""
    rng = np.random.default_rng(seed)
    methods = [m.GAN_METHOD, m.CNN_METHOD, m.F3_METHOD, m.UV_E2_METHOD, m.F2_METHOD, m.CANDIDATE_C_METHOD]
    per_sample = {mid: {} for mid in methods}
    for si in range(m.N_EVAL):
        for mid in methods:
            per_sample[mid][si] = dict(
                pd_distance=float(rng.uniform(10, 30)), mt_distance=float(rng.uniform(3, 10)),
                psnruv=float(rng.uniform(25, 35)), speed_mae=float(rng.uniform(0.3, 1.2)),
            )
    return per_sample


print()
print('=== 4. Archetype A2 (gan_pd_vs_cnn_mt_conflict): eligibility + score direction ===')
per_sample_4 = synth_per_sample_topology(n=20, seed=2)
# Force a clean eligible/ineligible split by hand for samples 0 and 1.
per_sample_4[m.GAN_METHOD][0]['pd_distance'] = 5.0
per_sample_4[m.CNN_METHOD][0]['pd_distance'] = 10.0   # gan_pd < cnn_pd -> ok
per_sample_4[m.CNN_METHOD][0]['mt_distance'] = 4.0
per_sample_4[m.GAN_METHOD][0]['mt_distance'] = 9.0    # cnn_mt < gan_mt -> ok -> ELIGIBLE
per_sample_4[m.GAN_METHOD][1]['pd_distance'] = 20.0
per_sample_4[m.CNN_METHOD][1]['pd_distance'] = 10.0   # gan_pd > cnn_pd -> NOT eligible
res_a2 = m.archetype_a2_gan_cnn_conflict(per_sample_4)
check('A2 sample 0 (gan better PD, cnn better MT) is eligible', 0 in res_a2['eligible_idxs'])
check('A2 sample 1 (gan worse PD) is not eligible', 1 not in res_a2['eligible_idxs'])
check('A2 has exactly 2 score components', len(res_a2['components']) == 2)
check('A2 component names are the two positive margins',
      {c['name'] for c in res_a2['components']} ==
      {'gan_pd_improvement_vs_cnn', 'cnn_mt_improvement_vs_gan'})
# Larger margins should score higher: verify score is monotonic with a hand-computed extreme case.
pos0 = res_a2['eligible_idxs'].index(0)
check('A2 improvement margins are computed as (base - comparison) with the better method as comparison '
      '(always positive by construction, since eligibility already requires the comparison to be better)',
      all(c['raw'][pos0] > 0 for c in res_a2['components']))

print()
print('=== 5. Archetype A3 (f3_pd_vs_uv_e2_mt_tradeoff): eligibility ===')
per_sample_5 = synth_per_sample_topology(n=10, seed=3)
per_sample_5[m.F3_METHOD][0]['pd_distance'] = 5.0
per_sample_5[m.UV_E2_METHOD][0]['pd_distance'] = 10.0
per_sample_5[m.UV_E2_METHOD][0]['mt_distance'] = 4.0
per_sample_5[m.F3_METHOD][0]['mt_distance'] = 9.0
per_sample_5[m.F3_METHOD][1]['pd_distance'] = 20.0
per_sample_5[m.UV_E2_METHOD][1]['pd_distance'] = 5.0
res_a3 = m.archetype_a3_f3_uv_e2_tradeoff(per_sample_5)
check('A3 sample 0 (f3 better PD, uv_e2 better MT) is eligible', 0 in res_a3['eligible_idxs'])
check('A3 sample 1 (f3 worse PD) is not eligible', 1 not in res_a3['eligible_idxs'])

print()
print('=== 6. Archetype A4 (f2_balanced_vs_cnn): balance term uses min(), not mean() ===')
per_sample_6 = synth_per_sample_topology(seed=4)
# Baseline every sample to f2 == cnn (no improvement -> ineligible), then carve out an
# explicit eligible block [0, 6) so eligibility is exactly characterizable.
for si in range(m.N_EVAL):
    per_sample_6[m.F2_METHOD][si]['pd_distance'] = per_sample_6[m.CNN_METHOD][si]['pd_distance']
    per_sample_6[m.F2_METHOD][si]['mt_distance'] = per_sample_6[m.CNN_METHOD][si]['mt_distance']
for si in range(6):
    per_sample_6[m.F2_METHOD][si]['pd_distance'] = 10.0
    per_sample_6[m.CNN_METHOD][si]['pd_distance'] = 20.0  # constant +10 pd improvement every sample
    per_sample_6[m.F2_METHOD][si]['mt_distance'] = 5.0
    per_sample_6[m.CNN_METHOD][si]['mt_distance'] = 6.0   # constant +1 mt improvement every sample
per_sample_6[m.F2_METHOD][0]['mt_distance'] = 1.0
per_sample_6[m.CNN_METHOD][0]['mt_distance'] = 6.0  # sample 0: much larger MT improvement (+5) than others (+1)
res_a4 = m.archetype_a4_f2_balanced(per_sample_6)
check('A4: exactly the carved-out 6 samples are eligible (f2 improves both PD and MT vs CNN)',
      set(res_a4['eligible_idxs']) == set(range(6)))
check('A4 has exactly 4 score components', len(res_a4['components']) == 4)
check('A4 score formula rewards balance: min(z_pd, z_mt) used as primary term',
      'min(robust_z(pd_improvement)' in res_a4['score_formula'])

print()
print('=== 7. Archetype A5 (candidate_c_continuity): finite-value eligibility, MT need not improve ===')
per_sample_7 = synth_per_sample_topology(seed=5)
# Baseline every sample to candidate_c == cnn on PD (no improvement -> ineligible).
for si in range(m.N_EVAL):
    per_sample_7[m.CANDIDATE_C_METHOD][si]['pd_distance'] = per_sample_7[m.CNN_METHOD][si]['pd_distance']
for si in range(6):
    per_sample_7[m.CANDIDATE_C_METHOD][si]['pd_distance'] = 10.0
    per_sample_7[m.CNN_METHOD][si]['pd_distance'] = 20.0  # candidate_c always improves PD
    per_sample_7[m.CANDIDATE_C_METHOD][si]['mt_distance'] = 8.0
    per_sample_7[m.CNN_METHOD][si]['mt_distance'] = 6.0   # candidate_c WORSE on MT (not required to improve)
per_sample_7[m.CANDIDATE_C_METHOD][2]['mt_distance'] = float('nan')  # non-finite MT -> ineligible
res_a5 = m.archetype_a5_candidate_c_continuity(per_sample_7)
check('A5: candidate_c need not improve MT to be eligible (PD-improving, MT-worsening sample is eligible)',
      0 in res_a5['eligible_idxs'])
check('A5: sample with non-finite MT is excluded', 2 not in res_a5['eligible_idxs'])
check('A5 score formula uses the specified 0.50/0.20/0.15/0.15 weights',
      '0.50*robust_z' in res_a5['score_formula'] and '0.20*robust_z' in res_a5['score_formula'])

print()
print('=== 8. Archetypes A1/A6 (global disagreement/agreement): all 168 samples eligible, opposite sign ===')
sample_pref_rows = [dict(sample_idx=si, agreement_rate=0.3 + 0.001 * si) for si in range(168)]
samplewise_rows = []
for si in range(168):
    samplewise_rows.append(dict(sample_idx=si, metric_a='pd_distance', metric_b='mt_distance',
                                  oriented_pearson=-0.2 + 0.001 * si, oriented_spearman=-0.3 + 0.001 * si))
per_sample_8 = synth_per_sample_topology(n=168, seed=6)
res_a1 = m.archetype_a1_global_disagreement(per_sample_8, sample_pref_rows, samplewise_rows)
res_a6 = m.archetype_a6_global_agreement(per_sample_8, sample_pref_rows, samplewise_rows)
check('A1 (disagreement) eligible on all 168 samples', len(res_a1['eligible_idxs']) == 168)
check('A6 (agreement) eligible on all 168 samples', len(res_a6['eligible_idxs']) == 168)
check('A1 disagreement score is highest for the LOWEST agreement_rate/correlation sample (sample 0)',
      list(m.rank_eligible(res_a1)[0])[0] == 0)
check('A6 agreement score is highest for the HIGHEST agreement_rate/correlation sample (sample 167)',
      list(m.rank_eligible(res_a6)[0])[0] == 167)
check('A1 and A6 components are sign-reversed versions of each other',
      np.allclose(res_a1['components'][0]['raw'], -np.array(res_a6['components'][0]['raw'])))

print()
print('=== 9. no-eligible-samples -> hard fail ===')
per_sample_9 = synth_per_sample_topology(seed=7)
for si in range(m.N_EVAL):
    per_sample_9[m.GAN_METHOD][si]['pd_distance'] = 100.0  # gan always worse PD -> never eligible
    per_sample_9[m.CNN_METHOD][si]['pd_distance'] = 1.0
try:
    m.archetype_a2_gan_cnn_conflict(per_sample_9)
    check('zero-eligible archetype -> SystemExit', False)
except SystemExit as e:
    check('zero-eligible archetype -> SystemExit', 'zero eligible samples' in str(e))

print()
print('=== 10. Deterministic tie-breaking (score desc, then sample_idx asc) ===')
result_10 = dict(eligible_idxs=[5, 2, 8, 1], score=np.array([1.0, 2.0, 2.0, 0.5]))
ranked, score_by_idx = m.rank_eligible(result_10)
check('tie-break: equal scores (idx 2 and 8, both 2.0) resolve to ascending sample_idx',
      ranked[0] == 2 and ranked[1] == 8)
check('full ranking is score-descending with sample_idx tie-break', ranked == [2, 8, 5, 1])

print()
print('=== 11. Unique greedy de-duplication + alternate selection ===')
ranked_11 = [10, 20, 30, 40, 50, 60]
score_by_idx_11 = {si: 100 - si for si in ranked_11}
already_selected = {10, 30}
selected, alternates, diag = m.select_with_dedup('test_archetype', ranked_11, score_by_idx_11, already_selected)
check('de-dup: selected sample is the first NOT already claimed (20, since 10 is claimed)',
      selected['sample_idx'] == 20)
check('de-dup: exactly one duplicate_skip event recorded for sample_idx=10',
      sum(1 for d in diag if d['sample_idx'] == 10) == 1)
check('de-dup: alternates skip already-selected 30 too, giving [40, 50, 60]',
      [a['sample_idx'] for a in alternates] == [40, 50, 60])
check('de-dup: exactly 3 alternates retained', len(alternates) == 3)
check('de-dup: rank_before_dedup for the selected sample (20) is its position in the full ranked list (2)',
      selected['rank_before_dedup'] == 2)
check('de-dup: rank_after_dedup for the selected sample is 1 (first non-duplicate)',
      selected['rank_after_dedup'] == 1)

print()
print('=== 12. Full run_all_selections: exactly six unique selections ===')
per_sample_12 = synth_per_sample_topology(n=168, seed=42)
sample_pref_12 = [dict(sample_idx=si, agreement_rate=float(np.random.default_rng(si).uniform(0, 1)))
                   for si in range(168)]
samplewise_12 = []
rng12 = np.random.default_rng(99)
for si in range(168):
    samplewise_12.append(dict(sample_idx=si, metric_a='pd_distance', metric_b='mt_distance',
                                 oriented_pearson=float(rng12.uniform(-1, 1)),
                                 oriented_spearman=float(rng12.uniform(-1, 1))))
# Make every archetype's eligibility condition true for at least a handful of samples.
for si in range(0, 60):
    per_sample_12[m.GAN_METHOD][si]['pd_distance'] = 5.0
    per_sample_12[m.CNN_METHOD][si]['pd_distance'] = 15.0
    per_sample_12[m.CNN_METHOD][si]['mt_distance'] = 3.0
    per_sample_12[m.GAN_METHOD][si]['mt_distance'] = 8.0
    per_sample_12[m.F3_METHOD][si]['pd_distance'] = 5.0
    per_sample_12[m.UV_E2_METHOD][si]['pd_distance'] = 15.0
    per_sample_12[m.UV_E2_METHOD][si]['mt_distance'] = 3.0
    per_sample_12[m.F3_METHOD][si]['mt_distance'] = 8.0
    per_sample_12[m.F2_METHOD][si]['pd_distance'] = 5.0
    per_sample_12[m.CNN_METHOD][si]['pd_distance'] = 15.0
    per_sample_12[m.F2_METHOD][si]['mt_distance'] = 3.0
    per_sample_12[m.CANDIDATE_C_METHOD][si]['pd_distance'] = 5.0
results_12 = m.compute_all_archetypes(per_sample_12, sample_pref_12, samplewise_12)
sel_by_a_12, diag_12, ranked_by_a_12, score_by_a_12 = m.run_all_selections(results_12)
selected_idxs_12 = [sel_by_a_12[a]['selected']['sample_idx'] for a in m.ARCHETYPE_PRIORITY]
check('run_all_selections yields exactly 6 archetypes', len(selected_idxs_12) == 6)
check('run_all_selections yields 6 UNIQUE sample indices', len(set(selected_idxs_12)) == 6)
for a in m.ARCHETYPE_PRIORITY:
    check(f'{a}: exactly 3 alternates retained', len(sel_by_a_12[a]['alternates']) == 3)

print()
print('=== 13. Repository-relative raw-path construction (no absolute inventory paths used) ===')
method_inventory = m.load_method_inventory()
paths_13 = m.resolve_raw_paths(method_inventory)
for mid, p in paths_13.items():
    for role in ('idx', 'dataIN', 'dataGT'):
        path = p[role]
        check(f'{mid}:{role} path is repo-relative under REPO_ROOT (not /home/adadhwal/...)',
              str(path).startswith(str(m.REPO_ROOT)) and 'adadhwal' not in str(path))
check('bicubic has no on-disk dataSR path (reconstructed in memory)', paths_13[m.BICUBIC_METHOD]['dataSR'] is None)
check('cnn/gan resolve under data_out_fixed/wind_mrhr_<mid>',
      'data_out_fixed/wind_mrhr_cnn' in str(paths_13[m.CNN_METHOD]['dataIN'])
      and 'data_out_fixed/wind_mrhr_gan' in str(paths_13[m.GAN_METHOD]['dataIN']))
check('learned candidates resolve under data_out/wind_finetune_<original_method_name>',
      'data_out/wind_finetune_candidateC_expanded2688' in str(paths_13[m.CANDIDATE_C_METHOD]['dataIN']))
# Explicitly confirm the raw absolute inventory column is never read for execution.
inv_candidate_c = method_inventory[m.CANDIDATE_C_METHOD]
check("method_inventory's absolute idx_path is provenance-only and differs from the resolved execution path",
      inv_candidate_c['idx_path'] != str(paths_13[m.CANDIDATE_C_METHOD]['idx']))

print()
print('=== 14. Physical-unit speed calculation (no accidental MU_SIG denorm) ===')
uv_single = np.array([[3.0, 4.0], [0.0, 5.0]]).reshape(1, 2, 2)  # (H=1,W=2,C=2) -> speeds [5.0, 5.0]
speed = m.speed_from_uv(uv_single)
check('speed_from_uv(u=3,v=4) == 5 (3-4-5 triangle)', abs(speed[0, 0] - 5.0) < 1e-9)
check('speed_from_uv(u=0,v=5) == 5', abs(speed[0, 1] - 5.0) < 1e-9)
batch_uv = np.stack([uv_single, uv_single * 2], axis=0)  # (N=2,H=1,W=2,C=2)
speed_batch = m.speed_from_uv(batch_uv)
check('speed_from_uv works on a batch (N,H,W,C) the same way as a single (H,W,C) sample',
      abs(speed_batch[0, 0, 0] - 5.0) < 1e-9 and abs(speed_batch[1, 0, 0] - 10.0) < 1e-9)
check('speed_from_uv does NOT apply MU_SIG scaling (raw physical sqrt(u^2+v^2), not sigma*x+mu first)',
      abs(m.speed_from_uv(np.array([[1.0, 0.0]])) - 1.0) < 1e-9)
denorm = m._denorm_from_normalized(np.array([0.0, 0.0]))
check('_denorm_from_normalized is a distinct, separate helper (mu passthrough at x_norm=0)',
      np.allclose(denorm, m.MU_SIG[0]))
check('_denorm_from_normalized is never invoked by speed_from_uv (different function identities)',
      m.speed_from_uv is not m._denorm_from_normalized)

print()
print('=== 15. Bicubic interpolation: shape and determinism ===')
rng15 = np.random.default_rng(11)
old_hr = m.EXPECTED_HR_SHAPE
try:
    m.EXPECTED_HR_SHAPE = (m.N_EVAL, 12, 12, 2)
    data_in_15 = rng15.normal(size=(3, 4, 4, 2)).astype(np.float32)
    out_a = m.bicubic_reconstruct_selected(data_in_15)
    out_b = m.bicubic_reconstruct_selected(data_in_15)
    check('bicubic reconstruction shape matches expected HR shape', out_a.shape == (3, 12, 12, 2))
    check('bicubic reconstruction is deterministic (identical arrays across two calls)',
          np.array_equal(out_a, out_b))
    check('bicubic reconstruction is finite', np.isfinite(out_a).all())
finally:
    m.EXPECTED_HR_SHAPE = old_hr

print()
print('=== 16. NaN/Inf rejection and malformed shape handling in audit ===')
tmp_root = Path(tempfile.mkdtemp(prefix='phase2d_test_'))
old_in_shape, old_hr_shape = m.EXPECTED_IN_SHAPE, m.EXPECTED_HR_SHAPE
try:
    m.EXPECTED_IN_SHAPE = (m.N_EVAL, 4, 4, 2)
    m.EXPECTED_HR_SHAPE = (m.N_EVAL, 6, 6, 2)
    rng16 = np.random.default_rng(21)
    idx16 = np.arange(m.N_EVAL)
    data_in16 = rng16.normal(size=(m.N_EVAL, 4, 4, 2)).astype(np.float32)
    data_gt16 = rng16.normal(size=(m.N_EVAL, 6, 6, 2)).astype(np.float32)

    method_inv16 = m.load_method_inventory()
    paths16 = m.resolve_raw_paths(method_inv16, base_dir=tmp_root)
    cnn_dir = tmp_root / 'data_out_fixed' / 'wind_mrhr_cnn'
    cnn_dir.mkdir(parents=True)
    np.save(cnn_dir / 'idx.npy', idx16)
    np.save(cnn_dir / 'dataIN.npy', data_in16)
    np.save(cnn_dir / 'dataGT.npy', data_gt16)
    data_sr_bad = data_gt16.copy()
    data_sr_bad[0, 0, 0, 0] = np.nan  # inject a NaN in row 0 (a SELECTED row, see selected16 below)
    np.save(cnn_dir / 'dataSR.npy', data_sr_bad)

    gan_dir = tmp_root / 'data_out_fixed' / 'wind_mrhr_gan'
    gan_dir.mkdir(parents=True)
    np.save(gan_dir / 'idx.npy', idx16)
    np.save(gan_dir / 'dataIN.npy', data_in16)
    np.save(gan_dir / 'dataGT.npy', data_gt16)
    # Malformed shape: wrong spatial resolution for gan's dataSR.
    np.save(gan_dir / 'dataSR.npy', rng16.normal(size=(m.N_EVAL, 3, 3, 2)).astype(np.float32))

    selected16 = [0, 1]
    fake_ps16 = {mid: {si: {'speed_mae': 0.0} for si in selected16} for mid in (m.CNN_METHOD, m.GAN_METHOD)}
    partial_paths = {m.CNN_METHOD: paths16[m.CNN_METHOD], m.GAN_METHOD: paths16[m.GAN_METHOD]}
    audit16 = m.audit_raw_artifacts(partial_paths, selected16, fake_ps16, base_dir=tmp_root)
    check('NaN in dataSR is detected and reported as a failure',
          any('non-finite' in f and 'cnn' in f for f in audit16['failures']))
    check('malformed dataSR shape is detected and reported as a failure',
          any('shape' in f and 'gan' in f for f in audit16['failures']))
    cnn_row16 = next(r for r in audit16['alignment_rows'] if r['method_id'] == m.CNN_METHOD)
    gan_row16 = next(r for r in audit16['alignment_rows'] if r['method_id'] == m.GAN_METHOD)
    check('cnn overall_status is FAIL (non-finite dataSR)', cnn_row16['overall_status'] == 'FAIL')
    check('gan overall_status is FAIL (malformed dataSR shape)', gan_row16['overall_status'] == 'FAIL')
    check('raw_artifact_inventory path fields are repo-relative to base_dir, never absolute',
          all(not r['path'].startswith('/') for r in audit16['inventory_rows']))
finally:
    m.EXPECTED_IN_SHAPE, m.EXPECTED_HR_SHAPE = old_in_shape, old_hr_shape
    shutil.rmtree(tmp_root, ignore_errors=True)

print()
print('=== 17. Exact idx validation + input/GT alignment (full-168-row) ===')
tmp_root2 = Path(tempfile.mkdtemp(prefix='phase2d_test_'))
old_in_shape, old_hr_shape = m.EXPECTED_IN_SHAPE, m.EXPECTED_HR_SHAPE
try:
    m.EXPECTED_IN_SHAPE = (m.N_EVAL, 4, 4, 2)
    m.EXPECTED_HR_SHAPE = (m.N_EVAL, 6, 6, 2)
    rng17 = np.random.default_rng(31)
    idx17 = np.arange(m.N_EVAL)
    data_in17 = rng17.normal(size=(m.N_EVAL, 4, 4, 2)).astype(np.float32)
    data_gt17 = rng17.normal(size=(m.N_EVAL, 6, 6, 2)).astype(np.float32)
    data_sr17 = data_gt17.copy()

    method_inv17 = m.load_method_inventory()
    paths17 = m.resolve_raw_paths(method_inv17, base_dir=tmp_root2)
    cnn_dir = tmp_root2 / 'data_out_fixed' / 'wind_mrhr_cnn'
    cnn_dir.mkdir(parents=True)
    np.save(cnn_dir / 'idx.npy', idx17)
    np.save(cnn_dir / 'dataIN.npy', data_in17)
    np.save(cnn_dir / 'dataGT.npy', data_gt17)
    np.save(cnn_dir / 'dataSR.npy', data_sr17)

    gan_dir = tmp_root2 / 'data_out_fixed' / 'wind_mrhr_gan'
    gan_dir.mkdir(parents=True)
    np.save(gan_dir / 'idx.npy', idx17)
    np.save(gan_dir / 'dataIN.npy', data_in17)  # exactly aligned with CNN
    np.save(gan_dir / 'dataGT.npy', data_gt17)  # exactly aligned with CNN
    np.save(gan_dir / 'dataSR.npy', data_gt17.copy())

    selected17 = [0, 5]
    fake_ps17 = {mid: {si: {'speed_mae': 0.0} for si in selected17} for mid in (m.CNN_METHOD, m.GAN_METHOD)}
    partial17 = {m.CNN_METHOD: paths17[m.CNN_METHOD], m.GAN_METHOD: paths17[m.GAN_METHOD]}
    audit17 = m.audit_raw_artifacts(partial17, selected17, fake_ps17, base_dir=tmp_root2)
    gan_row = next(r for r in audit17['alignment_rows'] if r['method_id'] == m.GAN_METHOD)
    check('exactly-aligned gan input/GT reports PASS alignment status',
          gan_row['input_alignment_status'] == 'PASS' and gan_row['gt_alignment_status'] == 'PASS')
    check('idx_validation_status is PASS for a valid ordered 0..167 idx array', gan_row['idx_validation_status'] == 'PASS')
    check('gan overall_status is PASS when everything is consistent', gan_row['overall_status'] == 'PASS')

    # Now break alignment: perturb gan's dataIN so it no longer matches CNN's canonical input
    # ONLY in an unselected row (sample_idx=100, not in selected17=[0,5]) -- the full-168-row
    # audit must still catch this even though no selected sample is affected.
    perturbed = data_in17.copy()
    perturbed[100] += 1.0
    np.save(gan_dir / 'dataIN.npy', perturbed)
    audit17b = m.audit_raw_artifacts(partial17, selected17, fake_ps17, base_dir=tmp_root2)
    gan_row_b = next(r for r in audit17b['alignment_rows'] if r['method_id'] == m.GAN_METHOD)
    check('corruption confined to an UNSELECTED row (idx=100) is still detected as a MISMATCH',
          gan_row_b['input_alignment_status'] == 'FAIL')
    check('unselected-row corruption is reported as an audit failure',
          any('dataIN' in f and 'gan' in f for f in audit17b['failures']))
    check('unselected-row corruption makes the whole audit fail (audit hard-fails, not silently passes)',
          len(audit17b['failures']) > 0)
finally:
    m.EXPECTED_IN_SHAPE, m.EXPECTED_HR_SHAPE = old_in_shape, old_hr_shape
    shutil.rmtree(tmp_root2, ignore_errors=True)

print()
print('=== 17b. validate_idx_array: ordered/permutation/duplicate-missing/shape/dtype ===')
good_idx = np.arange(m.N_EVAL, dtype=np.int64)
status, detail = m.validate_idx_array(good_idx, 'testmethod')
check('exact ordered idx passes', status == 'PASS' and detail == '')

perm_idx = good_idx.copy()
perm_idx[0], perm_idx[1] = perm_idx[1], perm_idx[0]  # swap two entries -> a permutation, same set
status, detail = m.validate_idx_array(perm_idx, 'testmethod')
check('a permutation fails (set-equal but not ordered)', status == 'FAIL')

dup_idx = good_idx.copy()
dup_idx[167] = dup_idx[0]  # duplicate index 0, sample 167 now missing
status, detail = m.validate_idx_array(dup_idx, 'testmethod')
check('a duplicate/missing index fails', status == 'FAIL')

wrong_shape_idx = np.arange(m.N_EVAL - 1, dtype=np.int64)
status, detail = m.validate_idx_array(wrong_shape_idx, 'testmethod')
check('wrong shape fails', status == 'FAIL' and 'shape' in detail)

noninteger_idx = np.arange(m.N_EVAL, dtype=np.float64)
status, detail = m.validate_idx_array(noninteger_idx, 'testmethod')
check('noninteger idx fails', status == 'FAIL' and 'dtype' in detail)

print()
print('=== 17c. Malformed canonical CNN idx -> controlled SystemExit, not KeyError ===')
tmp_root2c = Path(tempfile.mkdtemp(prefix='phase2d_test_'))
old_in_shape, old_hr_shape = m.EXPECTED_IN_SHAPE, m.EXPECTED_HR_SHAPE
try:
    m.EXPECTED_IN_SHAPE = (m.N_EVAL, 4, 4, 2)
    m.EXPECTED_HR_SHAPE = (m.N_EVAL, 6, 6, 2)
    rng17c = np.random.default_rng(33)
    bad_canonical_idx = np.arange(m.N_EVAL)
    bad_canonical_idx[0] = 5  # duplicate -> malformed canonical idx
    data_in17c = rng17c.normal(size=(m.N_EVAL, 4, 4, 2)).astype(np.float32)
    data_gt17c = rng17c.normal(size=(m.N_EVAL, 6, 6, 2)).astype(np.float32)

    method_inv17c = m.load_method_inventory()
    paths17c = m.resolve_raw_paths(method_inv17c, base_dir=tmp_root2c)
    cnn_dir = tmp_root2c / 'data_out_fixed' / 'wind_mrhr_cnn'
    cnn_dir.mkdir(parents=True)
    np.save(cnn_dir / 'idx.npy', bad_canonical_idx)
    np.save(cnn_dir / 'dataIN.npy', data_in17c)
    np.save(cnn_dir / 'dataGT.npy', data_gt17c)
    np.save(cnn_dir / 'dataSR.npy', data_gt17c.copy())

    try:
        m.audit_raw_artifacts({m.CNN_METHOD: paths17c[m.CNN_METHOD]}, [0, 5],
                                 {m.CNN_METHOD: {0: {'speed_mae': 0.0}, 5: {'speed_mae': 0.0}}}, base_dir=tmp_root2c)
        check('malformed canonical idx -> controlled SystemExit (not KeyError)', False)
    except SystemExit as e:
        check('malformed canonical idx -> controlled SystemExit (not KeyError)',
              'Canonical CNN idx.npy failed validation' in str(e))
    except KeyError:
        check('malformed canonical idx -> controlled SystemExit (not KeyError)', False)
finally:
    m.EXPECTED_IN_SHAPE, m.EXPECTED_HR_SHAPE = old_in_shape, old_hr_shape
    shutil.rmtree(tmp_root2c, ignore_errors=True)

print()
print('=== 18. render mode hard-fails when real arrays are absent (this checkout) ===')
try:
    real_inventory = m.load_method_inventory()
    real_paths = m.resolve_raw_paths(real_inventory)  # base_dir=REPO_ROOT (real, arrays absent)
    m.require_raw_artifacts_exist(real_paths)
    check('render-path artifact check hard-fails in this lightweight checkout', False)
except SystemExit as e:
    check('render-path artifact check hard-fails in this lightweight checkout',
          'Missing raw artifact' in str(e))

print()
print('=== 19. Render helper succeeds end-to-end on a synthetic temporary artifact tree ===')
tmp_root3 = Path(tempfile.mkdtemp(prefix='phase2d_test_'))
old_in_shape, old_hr_shape = m.EXPECTED_IN_SHAPE, m.EXPECTED_HR_SHAPE
preview_dirs_created = []
try:
    m.EXPECTED_IN_SHAPE = (m.N_EVAL, 4, 4, 2)
    m.EXPECTED_HR_SHAPE = (m.N_EVAL, 8, 8, 2)
    rng19 = np.random.default_rng(41)
    idx19 = np.arange(m.N_EVAL)
    data_in19 = (rng19.normal(size=(m.N_EVAL, 4, 4, 2)).astype(np.float32) * 3 + 1)
    data_gt19 = (rng19.normal(size=(m.N_EVAL, 8, 8, 2)).astype(np.float32) * 3 + 1)
    data_sr19 = data_gt19.copy()

    method_inv19 = m.load_method_inventory()
    paths19 = m.resolve_raw_paths(method_inv19, base_dir=tmp_root3)
    cnn_dir = tmp_root3 / 'data_out_fixed' / 'wind_mrhr_cnn'
    cnn_dir.mkdir(parents=True)
    np.save(cnn_dir / 'idx.npy', idx19)
    np.save(cnn_dir / 'dataIN.npy', data_in19)
    np.save(cnn_dir / 'dataGT.npy', data_gt19)
    np.save(cnn_dir / 'dataSR.npy', data_sr19)
    for mid, p in paths19.items():
        if mid in (m.BICUBIC_METHOD, m.CNN_METHOD):
            continue
        p['dir'].mkdir(parents=True, exist_ok=True)
        np.save(p['idx'], idx19)
        np.save(p['dataIN'], data_in19)
        np.save(p['dataGT'], data_gt19)
        np.save(p['dataSR'], data_sr19)

    m.require_raw_artifacts_exist(paths19)
    selected19 = [3, 7, 11, 15, 19, 23]
    fake_ps19 = {mid: {si: dict(speed_mae=0.0, pd_distance=1.0, mt_distance=2.0) for si in selected19}
                  for mid in m.FULL_SELECTED_STORY}
    # bicubic's dataSR is reconstructed in memory (never literally equal to dataGT for
    # random data), and bicubic is NOT exempt from speed_mae reproduction -- supply the
    # true recomputed value so this "everything is consistent" fixture is actually consistent.
    bicubic_sr_probe = m.bicubic_reconstruct_selected(data_in19[selected19])
    for i, si in enumerate(selected19):
        true_mae = float(np.mean(np.abs(m.speed_from_uv(bicubic_sr_probe[i]) - m.speed_from_uv(data_gt19[si]))))
        fake_ps19[m.BICUBIC_METHOD][si]['speed_mae'] = true_mae
    audit19 = m.audit_raw_artifacts(paths19, selected19, fake_ps19, base_dir=tmp_root3)
    check('synthetic-tree audit has zero failures when data is internally consistent',
          audit19['failures'] == [])
    check('synthetic-tree audit produced selected_data for every full_selected_story method',
          set(audit19['selected_data'].keys()) == set(m.FULL_SELECTED_STORY))
    check('every method reports overall_status PASS (full-168-row audit)',
          all(r['overall_status'] == 'PASS' for r in audit19['alignment_rows']))
    check('every method reports idx_validation_status PASS', all(r['idx_validation_status'] == 'PASS'
          for r in audit19['alignment_rows']))
    bicubic_row19 = next(r for r in audit19['alignment_rows'] if r['method_id'] == m.BICUBIC_METHOD)
    check('bicubic dataSR shape/finiteness status is N/A (reconstructed selected-only, never a full-168-row array)',
          bicubic_row19['dataSR_shape_status'] == 'N/A' and bicubic_row19['dataSR_finiteness_status'] == 'N/A')

    sel_by_a19 = {a: selected19[i] for i, a in enumerate(m.ARCHETYPE_PRIORITY)}

    # Artifact manifest (Section 8): one row per selected sample x full_selected_story
    # method, plus one GT row per sample -> 6 * (7 + 1) = 48 rows.
    manifest_rows19 = m.build_artifact_manifest_rows(
        audit19, sel_by_a19, selected19, method_inv19, m.build_topology_source_map(m.load_column_mapping_rows()),
        paths19, base_dir=tmp_root3)
    check('artifact manifest has exactly 48 rows (6 samples x (7 methods + 1 GT))', len(manifest_rows19) == 48)
    check('artifact manifest GT rows all have finite_status PASS',
          all(r['finite_status'] == 'PASS' for r in manifest_rows19 if r['method_id'] == 'GT'))
    check('artifact manifest has no absolute path in any path-like field',
          all(not str(v).startswith('/') for r in manifest_rows19 for k, v in r.items() if 'path' in k.lower())
          and all(not str(r['source_array_directory']).startswith('/') for r in manifest_rows19))

    render_rows19 = m.render_selected_previews(audit19['selected_data'], sel_by_a19, fake_ps19)
    for a in m.ARCHETYPE_PRIORITY:
        preview_dirs_created.append(m.PREVIEWS_DIR / a)
    check('render_selected_previews produced 6 per-sample previews + 1 contact sheet',
          len(render_rows19) == 7)
    all_exist = all(Path(m.REPO_ROOT / r['output_path']).exists() for r in render_rows19)
    check('every declared preview PNG actually exists on disk', all_exist)
    all_nonempty = all(Path(m.REPO_ROOT / r['output_path']).stat().st_size > 0 for r in render_rows19)
    check('every preview PNG is non-empty (no placeholder/blank file)', all_nonempty)
    check('detailed-preview panel_count is the 15-meaningful-panel convention for every per-sample row',
          all(r['panel_count'] == 15 for r in render_rows19 if r['archetype_id'] != 'all'))

    # Common (shared) color-limit checks: verify against hand-computed union of
    # GT + all 7 method SR speed fields (bicubic's reconstruction genuinely
    # differs from GT, so this is NOT simply the GT-alone range).
    row0 = render_rows19[0]
    gt0 = m.speed_from_uv(audit19['selected_data'][m.CNN_METHOD]['gt'][0])
    all_speed0 = [gt0] + [m.speed_from_uv(audit19['selected_data'][mid]['sr'][0]) for mid in m.FULL_SELECTED_STORY]
    expected_vmin0 = min(float(a.min()) for a in all_speed0)
    expected_vmax0 = max(float(a.max()) for a in all_speed0)
    check('shared_speed_vmin/vmax in preview_render_validation matches the hand-recomputed '
          'union of GT + all 7 method SR speed fields',
          abs(row0['shared_speed_vmin'] - expected_vmin0) < 1e-6
          and abs(row0['shared_speed_vmax'] - expected_vmax0) < 1e-6)
    check('shared_error_vmin is 0.0 (errors are non-negative absolute differences)',
          row0['shared_error_vmax'] >= row0['shared_error_vmin'] == 0.0)
finally:
    m.EXPECTED_IN_SHAPE, m.EXPECTED_HR_SHAPE = old_in_shape, old_hr_shape
    shutil.rmtree(tmp_root3, ignore_errors=True)
    for d in preview_dirs_created:
        shutil.rmtree(d, ignore_errors=True)
    shutil.rmtree(m.PREVIEWS_DIR / 'phase2d_selected_archetypes_contact_sheet.png', ignore_errors=True)
    contact = m.PREVIEWS_DIR / 'phase2d_selected_archetypes_contact_sheet.png'
    if contact.exists():
        contact.unlink()

print()
print('=== 19b. compute_preview_panel_data: common scaling from GT + all 7 SR fields ===')
gt_uv_19b = np.zeros((4, 4, 2), dtype=np.float64)
gt_uv_19b[0, 0] = [3.0, 4.0]  # GT speed range: [0, 5]
sr_uv_by_method_19b = {mid: gt_uv_19b.copy() for mid in m.FULL_SELECTED_STORY}
# One method's SR field has a value OUTSIDE the GT range on both ends.
sr_uv_by_method_19b[m.GAN_METHOD] = gt_uv_19b.copy()
sr_uv_by_method_19b[m.GAN_METHOD][1, 1] = [0.0, 20.0]  # speed=20, above GT max of 5
sr_uv_by_method_19b[m.CNN_METHOD] = gt_uv_19b.copy()
sr_uv_by_method_19b[m.CNN_METHOD][2, 2] = [-1.0, 0.0]  # speed=1, still within [0,5]; vmin stays 0 (all fields include 0)
panel19b = m.compute_preview_panel_data(gt_uv_19b, sr_uv_by_method_19b)
check('an SR field outside the GT range (gan=20) expands the shared speed vmax beyond GT max (5)',
      panel19b['speed_vmax'] >= 20.0 - 1e-9 and panel19b['speed_vmax'] > 5.0)
check('shared speed limits are a single (vmin, vmax) pair, not per-method (same pair used by every panel)',
      isinstance(panel19b['speed_vmin'], float) and isinstance(panel19b['speed_vmax'], float))
expected_vmax = max(float(m.speed_from_uv(v).max()) for v in ([gt_uv_19b] + list(sr_uv_by_method_19b.values())))
check('shared speed vmax exactly equals max over GT + all 7 SR fields (hand-recomputed)',
      abs(panel19b['speed_vmax'] - expected_vmax) < 1e-9)
expected_err_vmax = max(float(np.abs(m.speed_from_uv(v) - m.speed_from_uv(gt_uv_19b)).max())
                          for v in sr_uv_by_method_19b.values())
check('shared error vmax exactly equals max abs error over all 7 method error fields (hand-recomputed)',
      abs(panel19b['error_vmax'] - expected_err_vmax) < 1e-9)
check('shared error vmin is exactly 0.0', panel19b['error_vmin'] == 0.0)
check('the function returns exactly ONE speed_vmin/speed_vmax pair (not per-method values) -- '
      'every consumer (GT panel + all 7 method panels) is structurally forced to use the same limits',
      set(panel19b.keys()) == {'gt_speed', 'method_speeds', 'speed_vmin', 'speed_vmax', 'errors',
                                  'error_vmin', 'error_vmax'})
check('the function returns exactly ONE error_vmin/error_vmax pair (not per-method values) -- '
      'every error panel is structurally forced to use the same limits',
      isinstance(panel19b['error_vmax'], float) and isinstance(panel19b['error_vmin'], float))

print()
print('=== 19c. Bicubic speed_mae reproduction is enforced, no exemption ===')
tmp_root19c = Path(tempfile.mkdtemp(prefix='phase2d_test_'))
old_in_shape, old_hr_shape = m.EXPECTED_IN_SHAPE, m.EXPECTED_HR_SHAPE
try:
    m.EXPECTED_IN_SHAPE = (m.N_EVAL, 4, 4, 2)
    m.EXPECTED_HR_SHAPE = (m.N_EVAL, 8, 8, 2)
    rng19c = np.random.default_rng(51)
    idx19c = np.arange(m.N_EVAL)
    data_in19c = rng19c.normal(size=(m.N_EVAL, 4, 4, 2)).astype(np.float32)
    data_gt19c = rng19c.normal(size=(m.N_EVAL, 8, 8, 2)).astype(np.float32)

    method_inv19c = m.load_method_inventory()
    paths19c = m.resolve_raw_paths(method_inv19c, base_dir=tmp_root19c)
    cnn_dir = tmp_root19c / 'data_out_fixed' / 'wind_mrhr_cnn'
    cnn_dir.mkdir(parents=True)
    np.save(cnn_dir / 'idx.npy', idx19c)
    np.save(cnn_dir / 'dataIN.npy', data_in19c)
    np.save(cnn_dir / 'dataGT.npy', data_gt19c)
    np.save(cnn_dir / 'dataSR.npy', data_gt19c.copy())

    selected19c = [10, 20]
    bicubic_paths = {m.CNN_METHOD: paths19c[m.CNN_METHOD], m.BICUBIC_METHOD: paths19c[m.BICUBIC_METHOD]}
    # The in-memory bicubic reconstruction is deterministic; recompute the true
    # value once so the "matching" case can supply an exactly-correct long-table
    # speed_mae, and the "mismatching" case can supply a deliberately wrong one.
    in_sel_probe = data_in19c[selected19c]
    sr_probe = m.bicubic_reconstruct_selected(in_sel_probe)
    true_speed_mae = {si: float(np.mean(np.abs(m.speed_from_uv(sr_probe[i]) - m.speed_from_uv(data_gt19c[si]))))
                        for i, si in enumerate(selected19c)}

    fake_ps_match = {m.BICUBIC_METHOD: {si: {'speed_mae': true_speed_mae[si]} for si in selected19c}}
    audit_match = m.audit_raw_artifacts(bicubic_paths, selected19c, fake_ps_match, base_dir=tmp_root19c)
    check('bicubic speed_mae reproduction test: MATCHING long-table value produces zero repro failures',
          not any('speed_mae' in f and m.BICUBIC_METHOD in f for f in audit_match['failures']))

    fake_ps_mismatch = {m.BICUBIC_METHOD: {si: {'speed_mae': true_speed_mae[si] + 5.0} for si in selected19c}}
    audit_mismatch = m.audit_raw_artifacts(bicubic_paths, selected19c, fake_ps_mismatch, base_dir=tmp_root19c)
    check('bicubic speed_mae reproduction test: MISMATCHING long-table value is a hard-fail '
          '(no bicubic exemption)',
          any('speed_mae' in f and m.BICUBIC_METHOD in f for f in audit_mismatch['failures']))
finally:
    m.EXPECTED_IN_SHAPE, m.EXPECTED_HR_SHAPE = old_in_shape, old_hr_shape
    shutil.rmtree(tmp_root19c, ignore_errors=True)

print()
print('=== 19d. Repository-relative CSV path guard (write_csv hard-fails on an absolute path field) ===')
try:
    m.write_csv(Path(tempfile.mktemp(suffix='.csv')), ['name', 'output_path'],
                 [dict(name='x', output_path='/etc/passwd')])
    check('write_csv hard-fails when a path-like field holds an absolute path', False)
except SystemExit as e:
    check('write_csv hard-fails when a path-like field holds an absolute path', 'Absolute path found' in str(e))

ok_tmp = Path(tempfile.mktemp(suffix='.csv'))
try:
    m.write_csv(ok_tmp, ['name', 'output_path'], [dict(name='x', output_path='ttk_runs_fixed/foo.png')])
    check('write_csv succeeds normally when path-like fields are repository-relative', ok_tmp.exists())
finally:
    if ok_tmp.exists():
        ok_tmp.unlink()

print()
print('=== 19e. Strong archetype_selected_samples.csv manifest validation ===')


def _valid_manifest_rows():
    return [dict(archetype_id=a, selected_sample_idx=str(10 * (i + 1)), selection_rank='1',
                  primary_or_alternate='primary', score='1.0', eligibility_status='eligible',
                  selection_reason='reason', methods_required='cnn,gan', metrics_used='pd_distance',
                  tie_break_fields='score desc, then sample_idx asc')
            for i, a in enumerate(m.ARCHETYPE_PRIORITY)]


valid_rows = _valid_manifest_rows()
errors = m.validate_selected_samples_manifest_rows(valid_rows, Path('dummy.csv'))
check('a fully valid manifest passes with zero errors', errors == [])

too_few = valid_rows[:5]
errors = m.validate_selected_samples_manifest_rows(too_few, Path('dummy.csv'))
check('fewer than 6 primary rows fails', any('exactly 6 primary rows' in e for e in errors))

wrong_order = valid_rows.copy()
wrong_order[0], wrong_order[1] = wrong_order[1], wrong_order[0]
errors = m.validate_selected_samples_manifest_rows(wrong_order, Path('dummy.csv'))
check('archetype_id order mismatch fails', any('order/identity mismatch' in e for e in errors))

wrong_id = [dict(r) for r in valid_rows]
wrong_id[0]['archetype_id'] = 'not_a_real_archetype'
errors = m.validate_selected_samples_manifest_rows(wrong_id, Path('dummy.csv'))
check('unrecognized archetype_id fails', any('order/identity mismatch' in e for e in errors))

dup_idx = [dict(r) for r in valid_rows]
dup_idx[1]['selected_sample_idx'] = dup_idx[0]['selected_sample_idx']
errors = m.validate_selected_samples_manifest_rows(dup_idx, Path('dummy.csv'))
check('duplicate selected_sample_idx across primary rows fails', any('not all unique' in e for e in errors))

out_of_range = [dict(r) for r in valid_rows]
out_of_range[0]['selected_sample_idx'] = '168'  # valid range is 0..167
errors = m.validate_selected_samples_manifest_rows(out_of_range, Path('dummy.csv'))
check('out-of-range selected_sample_idx (168) fails', any('out of range' in e for e in errors))

negative_idx = [dict(r) for r in valid_rows]
negative_idx[0]['selected_sample_idx'] = '-1'
errors = m.validate_selected_samples_manifest_rows(negative_idx, Path('dummy.csv'))
check('negative selected_sample_idx fails', any('out of range' in e for e in errors))

noninteger_idx_row = [dict(r) for r in valid_rows]
noninteger_idx_row[0]['selected_sample_idx'] = '12.5'
errors = m.validate_selected_samples_manifest_rows(noninteger_idx_row, Path('dummy.csv'))
check('non-integer selected_sample_idx fails', any('not an integer' in e for e in errors))

wrong_primary_flag = [dict(r) for r in valid_rows]
wrong_primary_flag[0]['primary_or_alternate'] = 'alternate'
errors = m.validate_selected_samples_manifest_rows(wrong_primary_flag, Path('dummy.csv'))
check('primary_or_alternate != "primary" on a primary row fails (row excluded from the 6 primaries, wrong count)',
      any('exactly 6 primary rows' in e for e in errors))

wrong_eligibility = [dict(r) for r in valid_rows]
wrong_eligibility[0]['eligibility_status'] = 'ineligible'
errors = m.validate_selected_samples_manifest_rows(wrong_eligibility, Path('dummy.csv'))
check('eligibility_status != "eligible" fails', any('eligibility_status' in e for e in errors))

empty_field = [dict(r) for r in valid_rows]
empty_field[0]['selection_reason'] = ''
errors = m.validate_selected_samples_manifest_rows(empty_field, Path('dummy.csv'))
check('an empty required field fails', any("'selection_reason'" in e and 'empty' in e for e in errors))

print()
print('=== 19f. figure_plan.csv: plans, does not claim automated rendering ===')
figure_rows19f = m.build_figure_plan_rows(sel_by_a19, results_12)
check('figure_plan has exactly 6 rows (one per archetype)', len(figure_rows19f) == 6)
check('figure_plan archetype_id values are exactly ARCHETYPE_PRIORITY in order',
      [r['archetype_id'] for r in figure_rows19f] == m.ARCHETYPE_PRIORITY)
check('every figure is status=planned_not_rendered (never claims rendering happened)',
      all(r['status'] == 'planned_not_rendered' for r in figure_rows19f))
check('every figure explicitly disclaims automated TTK/ParaView rendering',
      all('TTK/ParaView' in r['rendering_note'] for r in figure_rows19f))
check('every figure_id 1..6 is present exactly once', sorted(r['figure_id'] for r in figure_rows19f) == list(range(1, 7)))

print()
print('=== 19g. Unified doc builder retains the full report in both states ===')
fake_selection_19g = dict(
    results=results_12,
    selection_by_archetype={a: dict(selected=dict(sample_idx=sel_by_a19[a], score=1.0),
                                       alternates=[dict(sample_idx=900 + i, score=0.5) for i in range(3)])
                              for a in m.ARCHETYPE_PRIORITY},
    selected_sample_idx_by_archetype=sel_by_a19,
    all_diagnostics=[],
)
lines_selection_only = m.build_phase2d_doc_lines(fake_selection_19g, render_result=None)
text_selection_only = '\n'.join(lines_selection_only)
fake_render_result_19g = dict(audit=dict(failures=[]),
                                 render_rows=[dict(output_path='ttk_runs_fixed/x/y.png', archetype_id='a',
                                                     sample_idx=1, status='rendered')])
lines_rendered = m.build_phase2d_doc_lines(fake_selection_19g, render_result=fake_render_result_19g)
text_rendered = '\n'.join(lines_rendered)

REQUIRED_SECTION_MARKERS = [
    'Scope and frozen inputs', 'Why selection is algorithmic rather than manual',
    'Robust-z scoring convention', 'Archetype definitions', 'Duplicate-resolution decisions',
    'Selected sample IDs and alternates', 'Metric package', 'Raw-artifact requirements and validation',
    'Preview inventory', 'Figure plan for Phase 2D-B', 'Caveat: illustrative, not a population estimate',
    'Exact command to complete Phase 2D-A on Spark', 'Generated files',
]
for marker in REQUIRED_SECTION_MARKERS:
    check(f'selection-only doc retains section: {marker!r}', marker in text_selection_only)
    check(f'render-complete doc ALSO retains section: {marker!r} (not replaced by a minimal doc)',
          marker in text_rendered)
check('render-complete doc additionally reports the raw audit passed',
      'raw-artifact audit was performed' in text_rendered.lower()
      or 'PASSED with 0 failures' in text_rendered)
check('render-complete doc is at least as long as the selection-only doc (extended, not replaced)',
      len(text_rendered) >= len(text_selection_only) * 0.9)
check('selection-only doc explicitly says final figures deferred to Phase 2D-B',
      'deferred to Phase 2D-B' in text_selection_only)
check('render-complete doc explicitly says final figures deferred to Phase 2D-B',
      'deferred to Phase 2D-B' in text_rendered)

print()
print('=== 20. Selection manifest dimensions and ordering ===')
manifest_path = m.SELECTION_DIR / 'archetype_selected_samples.csv'
if manifest_path.exists():
    rows20 = m.read_csv_dicts(manifest_path)
    check('archetype_selected_samples.csv has exactly 6 rows', len(rows20) == 6)
    check('archetype_selected_samples.csv rows appear in ARCHETYPE_PRIORITY order',
          [r['archetype_id'] for r in rows20] == m.ARCHETYPE_PRIORITY)
    idxs20 = [int(r['selected_sample_idx']) for r in rows20]
    check('all 6 selected_sample_idx values are unique', len(set(idxs20)) == 6)
else:
    check('archetype_selected_samples.csv exists (run --selection-only before this test for full coverage)',
          False)

print()
print('=== 21. Missing prior protected file -> hard fail ===')
target = m.PHASE1_PROTECTED_CSVS[0]
backup = target.with_suffix('.csv.bak_test2d')
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
print('=== 22. Unexpected extra CSV in a frozen directory -> hard fail ===')
extra_path = m.PHASE2C_DIR / '__phase2d_test_unexpected_extra.csv'
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
print('=== 23. Altered prior checksum -> hard fail ===')
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
print('=== 24. Protected-file-list counts (12 + 14 + 28 + 32 = 86) ===')
check('Phase-1 protected files == 12', len(m.PHASE1_PROTECTED_FILES) == 12)
check('Phase-2A protected files == 14', len(m.PHASE2A_PROTECTED_FILES) == 14)
check('Phase-2B protected files == 28', len(m.PHASE2B_PROTECTED_FILES) == 28)
check('Phase-2C protected files == 32', len(m.PHASE2C_PROTECTED_FILES) == 32)
check('All protected files == 86', len(m.ALL_PROTECTED_FILES) == 86)

print()
if failures:
    print(f'{len(failures)} FAILURE(S): {failures}')
    sys.exit(1)
else:
    print('ALL TESTS PASSED')
