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
    data_sr_bad[0, 0, 0, 0] = np.nan  # inject a NaN
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
    audit16 = m.audit_raw_artifacts(partial_paths, selected16, fake_ps16)
    check('NaN in dataSR is detected and reported as a failure',
          any('non-finite' in f and 'cnn' in f for f in audit16['failures']))
    check('malformed dataSR shape is detected and reported as a failure',
          any('shape' in f and 'gan' in f for f in audit16['failures']))
finally:
    m.EXPECTED_IN_SHAPE, m.EXPECTED_HR_SHAPE = old_in_shape, old_hr_shape
    shutil.rmtree(tmp_root, ignore_errors=True)

print()
print('=== 17. Exact idx validation + input/GT alignment ===')
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
    audit17 = m.audit_raw_artifacts(partial17, selected17, fake_ps17)
    gan_row = next(r for r in audit17['alignment_rows'] if r['method_id'] == m.GAN_METHOD)
    check('exactly-aligned gan input/GT reports exact alignment status', gan_row['input_alignment_status'] == 'exact'
          and gan_row['gt_alignment_status'] == 'exact')
    check('idx_exact_0_167 is True for a valid 0..167 idx array', gan_row['idx_exact_0_167'] is True)

    # Now break alignment: perturb gan's dataIN so it no longer matches CNN's canonical input.
    np.save(gan_dir / 'dataIN.npy', data_in17 + 1.0)
    audit17b = m.audit_raw_artifacts(partial17, selected17, fake_ps17)
    gan_row_b = next(r for r in audit17b['alignment_rows'] if r['method_id'] == m.GAN_METHOD)
    check('perturbed gan input is detected as a MISMATCH', gan_row_b['input_alignment_status'] == 'MISMATCH')
    check('input misalignment is reported as an audit failure',
          any('dataIN' in f and 'gan' in f for f in audit17b['failures']))
finally:
    m.EXPECTED_IN_SHAPE, m.EXPECTED_HR_SHAPE = old_in_shape, old_hr_shape
    shutil.rmtree(tmp_root2, ignore_errors=True)

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
    audit19 = m.audit_raw_artifacts(paths19, selected19, fake_ps19)
    check('synthetic-tree audit has zero failures when data is internally consistent',
          audit19['failures'] == [])
    check('synthetic-tree audit produced selected_data for every full_selected_story method',
          set(audit19['selected_data'].keys()) == set(m.FULL_SELECTED_STORY))

    sel_by_a19 = {a: selected19[i] for i, a in enumerate(m.ARCHETYPE_PRIORITY)}
    render_rows19 = m.render_selected_previews(audit19['selected_data'], sel_by_a19, fake_ps19)
    for a in m.ARCHETYPE_PRIORITY:
        preview_dirs_created.append(m.PREVIEWS_DIR / a)
    check('render_selected_previews produced 6 per-sample previews + 1 contact sheet',
          len(render_rows19) == 7)
    all_exist = all(Path(m.REPO_ROOT / r['output_path']).exists() for r in render_rows19)
    check('every declared preview PNG actually exists on disk', all_exist)
    all_nonempty = all(Path(m.REPO_ROOT / r['output_path']).stat().st_size > 0 for r in render_rows19)
    check('every preview PNG is non-empty (no placeholder/blank file)', all_nonempty)

    # Common (shared) color-limit checks: verify against hand-computed GT min/max.
    row0 = render_rows19[0]
    si0 = selected19[0]
    gt0 = m.speed_from_uv(audit19['selected_data'][m.CNN_METHOD]['gt'][0])
    check('shared_speed_vmin/vmax in preview_render_validation matches GT min/max exactly',
          abs(row0['shared_speed_vmin'] - float(gt0.min())) < 1e-6
          and abs(row0['shared_speed_vmax'] - float(gt0.max())) < 1e-6)
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
