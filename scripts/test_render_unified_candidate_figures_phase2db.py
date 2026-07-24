#!/usr/bin/env python3
"""Synthetic/contract tests for scripts/render_unified_candidate_figures_phase2db.py.

Exercises the real shipped functions directly. Render/composite-path
functions are exercised against small synthetic fixtures built and torn
down under this script's OWN writable output tree
(ttk_runs_fixed/unified_candidate_analysis/phase2db/) or the scratch tmp
directory -- never under any Phase-1 through Phase-2D-A directory. Any test
that temporarily perturbs or creates a real Phase-1 through Phase-2D-A file
restores/removes it in a `finally` block and verifies the restoration
succeeded.

A real full --plan-only run in THIS lightweight checkout still correctly
hard-fails (Section 16), since Phase-2D-A's raw-artifact audit, preview
rendering, and the human visual-review record have never run/been authored
here -- exactly like data_out/data_out_fixed being absent. Section 6 proves
the FULL --plan-only pipeline succeeds by temporarily manufacturing minimal,
schema-valid stand-ins for those 15 specific files (never touching any
actually-frozen Phase-1 through Phase-2D-A content) and tearing them down
immediately afterward.

Run directly:
    python3 scripts/test_render_unified_candidate_figures_phase2db.py
"""
import csv as _csv
import math
import re
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import render_unified_candidate_figures_phase2db as m
import numpy as np
from PIL import Image

failures = []


def check(name, cond):
    status = 'PASS' if cond else 'FAIL'
    print(f'[{status}] {name}')
    if not cond:
        failures.append(name)


print('=== 1. Protected-file counts (86 prior + 32 Phase-2D-A = 118) ===')
check('Phase-2D-A protected files == 32', len(m.PHASE2D_A_PROTECTED_FILES) == 32)
check('  selection CSVs == 13', len(m.PHASE2D_A_SELECTION_CSVS) == 13)
check('  preview_audit CSVs == 6', len(m.PHASE2D_A_PREVIEW_AUDIT_CSVS) == 6)
check('  preview PNGs == 7', len(m.PHASE2D_A_PREVIEW_PNGS) == 7)
check('  docs == 2', len(m.PHASE2D_A_DOCS) == 2)
check('  scripts == 2', len(m.PHASE2D_A_SCRIPTS) == 2)
check('  logs == 2', len(m.PHASE2D_A_LOGS) == 2)
check('All protected files == 118', len(m.ALL_PROTECTED_FILES) == 118)
check('FROZEN_SAMPLE_SET covers exactly the 6 archetypes', set(m.FROZEN_SAMPLE_SET) == set(m.p2da.ARCHETYPE_PRIORITY))

print()
print('=== 2. Frozen-sample enforcement (real manifest must match FROZEN_SAMPLE_SET) ===')
manifest = m.read_and_validate_selection_manifest()
check('read_and_validate_selection_manifest() returns exactly FROZEN_SAMPLE_SET', manifest == m.FROZEN_SAMPLE_SET)
EXPECTED_FROZEN = {
    'global_descriptor_disagreement': 120, 'gan_pd_vs_cnn_mt_conflict': 34,
    'f3_pd_vs_uv_e2_mt_tradeoff': 119, 'f2_balanced_vs_cnn': 25,
    'candidate_c_continuity': 30, 'global_descriptor_agreement': 19,
}
check('FROZEN_SAMPLE_SET matches the literal task-specified sample IDs', m.FROZEN_SAMPLE_SET == EXPECTED_FROZEN)

print()
print('=== 3. Frozen-sample enforcement: a mismatched frozen set is rejected ===')
old_frozen = dict(m.FROZEN_SAMPLE_SET)
try:
    m.FROZEN_SAMPLE_SET['global_descriptor_disagreement'] = 999999  # deliberately wrong
    try:
        m.read_and_validate_selection_manifest()
        check('mismatched frozen sample set -> SystemExit', False)
    except SystemExit as e:
        check('mismatched frozen sample set -> SystemExit', 'does not match the frozen sample set' in str(e))
finally:
    m.FROZEN_SAMPLE_SET.clear()
    m.FROZEN_SAMPLE_SET.update(old_frozen)
check('FROZEN_SAMPLE_SET restored to the real frozen values', m.FROZEN_SAMPLE_SET == old_frozen)

print()
print('=== 4. Alternate activation rejection ===')
alt_rows = m.p2da.read_csv_dicts(m.p2da.SELECTION_DIR / 'archetype_alternates.csv')
alt_for_disagreement = next(r for r in alt_rows if r['archetype_id'] == 'global_descriptor_disagreement')
alt_idx = int(alt_for_disagreement['selected_sample_idx'])
check('the chosen alternate sample_idx is NOT the frozen primary value (sanity check on the fixture)',
      alt_idx != m.FROZEN_SAMPLE_SET['global_descriptor_disagreement'])
old_frozen = dict(m.FROZEN_SAMPLE_SET)
try:
    m.FROZEN_SAMPLE_SET['global_descriptor_disagreement'] = alt_idx  # simulate activating the alternate
    try:
        m.read_and_validate_selection_manifest()
        check('activating a known alternate sample_idx -> SystemExit (never silently accepted)', False)
    except SystemExit as e:
        check('activating a known alternate sample_idx -> SystemExit (never silently accepted)',
              'does not match the frozen sample set' in str(e))
finally:
    m.FROZEN_SAMPLE_SET.clear()
    m.FROZEN_SAMPLE_SET.update(old_frozen)
check('FROZEN_SAMPLE_SET restored after alternate-rejection test', m.FROZEN_SAMPLE_SET == old_frozen)
import inspect
src = inspect.getsource(m.read_and_validate_selection_manifest)
doc = m.read_and_validate_selection_manifest.__doc__ or ''
src_without_docstring = src.replace(doc, '')
check('read_and_validate_selection_manifest() never references archetype_alternates.csv outside its docstring '
      '(the docstring itself documents that it is deliberately never read)',
      'archetype_alternates' not in src_without_docstring)

print()
print('=== 5. Method-set enforcement + canonical method order per figure ===')
EXPECTED_METHOD_SETS = {
    1: {'GT', m.GAN, m.CNN, m.CANDIDATE_C, m.F3, m.F2, m.UV_E2},
    2: {'GT', m.BICUBIC, m.CNN, m.GAN},
    3: {'GT', m.CNN, m.F3, m.UV_E2, m.F2},
    4: {'GT', m.CNN, m.F3, m.F2, m.UV_E2},
    5: {'GT', m.CNN, m.CANDIDATE_C, m.F3, m.F2, m.UV_E2},
    6: {'GT', m.GAN, m.CNN, m.CANDIDATE_C, m.F3, m.F2, m.UV_E2},
}
CANONICAL_ORDER = [m.BICUBIC, m.CNN, m.GAN, m.CANDIDATE_C, m.F3, m.F2, m.UV_E2]
for c in m.FIGURE_CONTRACTS:
    actual = {'GT'} | set(c['required_methods'])
    check(f"figure {c['figure_id']} method set matches the task contract exactly",
          actual == EXPECTED_METHOD_SETS[c['figure_id']])
    canonical_subsequence = [x for x in CANONICAL_ORDER if x in c['required_methods']]
    check(f"figure {c['figure_id']} required_methods follows the canonical GT,Bicubic,CNN,GAN,Candidate C,F3,"
          f'F2,UV+E2 order', c['required_methods'] == canonical_subsequence)
check('figures 1 and 6 place CNN before GAN',
      m.FIGURE_BY_ID[1]['required_methods'].index(m.CNN) < m.FIGURE_BY_ID[1]['required_methods'].index(m.GAN)
      and m.FIGURE_BY_ID[6]['required_methods'].index(m.CNN) < m.FIGURE_BY_ID[6]['required_methods'].index(m.GAN))
check('figure 3 restricts F2 to full_panel_methods exclusion (compact contextual reference only)',
      m.F2 not in m.FIGURE_BY_ID[3]['full_panel_methods'] and m.F2 in m.FIGURE_BY_ID[3]['required_methods'])
check('figure 3 method_roles records F2 as compact_contextual_reference',
      m.FIGURE_BY_ID[3]['method_roles'].get(m.F2) == 'compact_contextual_reference')


# =============================================================================
# Golden-path fixture: manufactures minimal, schema-valid stand-ins for the
# 15 Phase-2D-A files genuinely absent in THIS lightweight checkout (never
# touching any actually-frozen content), so the FULL --plan-only pipeline
# can be proven correct end-to-end. Everything is torn down afterward.
# =============================================================================

def snapshot_tree(root):
    """Byte snapshot of every file under `root` (root may itself be
    git-tracked pre-existing content, e.g. from a prior commit's
    --plan-only output) -- used to precisely undo a test run's writes
    without destroying anything that existed beforehand."""
    if not root.exists():
        return {}
    return {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}


def restore_tree(root, snapshot):
    """Restores `root` to exactly the given snapshot: deletes any file not
    in the snapshot, rewrites any file that changed, and prunes now-empty
    directories that did not exist in the snapshot."""
    if root.exists():
        for p in list(root.rglob('*')):
            if p.is_file() and p not in snapshot:
                p.unlink()
    for p, data in snapshot.items():
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists() or p.read_bytes() != data:
            p.write_bytes(data)
    if root.exists():
        for d in sorted([x for x in root.rglob('*') if x.is_dir()], key=lambda x: -len(str(x))):
            try:
                d.rmdir()
            except OSError:
                pass


def make_completed_phase2d_a_fixture():
    created_paths = []
    created_dirs = []
    doc_backup = m.p2da.DOC_PATH.read_bytes()

    completed_text = m.p2da.DOC_PATH.read_text().replace(
        'Phase 2D-A selection stage complete.\nRaw-array audit and preview rendering pending authoritative '
        'Spark run.',
        'Phase 2D-A complete.\nSelection, raw audit, and review-preview generation all passed.',
    )
    m.p2da.DOC_PATH.write_text(completed_text)

    if not m.p2da.PREVIEW_AUDIT_DIR.exists():
        created_dirs.append(m.p2da.PREVIEW_AUDIT_DIR)
    m.p2da.PREVIEW_AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    for name in m.PHASE2D_A_PREVIEW_AUDIT_CSV_NAMES:
        p = m.p2da.PREVIEW_AUDIT_DIR / name
        p.write_text('col1,col2\n1,2\n')
        created_paths.append(p)

    for p in m.PHASE2D_A_PREVIEW_PNGS:
        if not p.parent.exists():
            created_dirs.append(p.parent)
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.new('RGB', (10, 10), color='white').save(p)
        created_paths.append(p)

    lines = ['# Phase 2D-A Visual Review (synthetic test fixture)', '']
    for aid, si in m.FROZEN_SAMPLE_SET.items():
        lines.append(f'{aid} sample_idx={si}: ACCEPTED')
    lines.append('')
    lines.append('No alternate was activated for any archetype.')
    m.VISUAL_REVIEW_DOC_PATH.write_text('\n'.join(lines) + '\n')
    created_paths.append(m.VISUAL_REVIEW_DOC_PATH)

    render_log_path = m.REPO_ROOT / 'logs' / 'unified_candidate_analysis_phase2d_render.log'
    render_log_path.write_text('synthetic render log for testing\n')
    created_paths.append(render_log_path)

    def cleanup():
        m.p2da.DOC_PATH.write_bytes(doc_backup)
        for p in created_paths:
            if p.exists():
                p.unlink()
        for d in sorted(set(created_dirs), key=lambda x: -len(str(x))):
            try:
                d.rmdir()
            except OSError:
                pass

    return cleanup


print()
print('=== 6. Full --plan-only pipeline succeeds end-to-end on the golden-path fixture ===')
# cmd_plan_only() will overwrite this checkout's existing (possibly previously
# committed) phase2db/ output and docs/unified_candidate_analysis_phase2db.md in
# place -- snapshot both exactly so they can be restored byte-for-byte, rather
# than deleting the whole tree (which would destroy pre-existing committed content).
out_dir_snapshot = snapshot_tree(m.OUT_DIR)
doc_2db_snapshot = m.DOC_PATH.read_bytes() if m.DOC_PATH.exists() else None
cleanup_fixture = make_completed_phase2d_a_fixture()
try:
    plan_result = m.cmd_plan_only()
    check('cmd_plan_only() completed without raising against the golden-path fixture', True)

    scanned = 0
    bad = []
    for d in (m.PLAN_DIR, m.FIGURE_DATA_DIR, m.VALIDATION_DIR):
        for csv_path in d.glob('*.csv'):
            for row in m.p2da.read_csv_dicts(csv_path):
                for k, v in row.items():
                    if 'path' in k.lower() and isinstance(v, str) and v.startswith('/'):
                        bad.append((csv_path, k, v))
                    scanned += 1
    check(f'no absolute path found across {scanned} scanned field values in real plan-only output', bad == [])

    check('exactly 168 figure-data metric values reproduced within tolerance',
          len(plan_result['repro_rows']) == 168 and all(r['status'] == 'PASS' for r in plan_result['repro_rows']))

    pd_rows = plan_result['pd_discovery_rows']
    check('pd_source_discovery.csv has rows for figures 1, 2, 3 only (the figures needing PD diagram panels)',
          {r['figure_id'] for r in pd_rows} == {1, 2, 3})
    check('every PD discovery row in this lightweight checkout is pending_authoritative_spark_source_discovery '
          '(never unavailable_after_authoritative_spark_audit here)',
          all(r['usable_status'] == m.STATUS_PENDING for r in pd_rows))
    check('GT is present among the searched methods for figure 1 (never assumed found without a concrete source)',
          any(r['method_id'] == 'GT' and r['figure_id'] == 1 for r in pd_rows))
    check('bicubic is present among the searched methods for figure 2',
          any(r['method_id'] == m.BICUBIC and r['figure_id'] == 2 for r in pd_rows))

    composite_path = m.PLAN_DIR / 'final_composite_manifest.csv'
    check('final_composite_manifest.csv was written', composite_path.exists())
    composite_rows = m.p2da.read_csv_dicts(composite_path)
    check('every composite-manifest row for figures 1/2/3 is final_visible_status=pending '
          '(PD verdict not yet resolved) or manual_topology (awaiting export)',
          all(r['final_visible_status'] in ('pending', 'visible') for r in composite_rows
              if int(r['figure_id']) in (1, 2, 3) and r['panel_group'] == 'pd_coordinate'))

    zoom_val_path = m.VALIDATION_DIR / 'zoom_selection_validation.csv'
    check('zoom_selection_validation.csv was written in the not_yet_computed state',
          zoom_val_path.exists() and m.p2da.read_csv_dicts(zoom_val_path)[0]['status'] == 'not_yet_computed')

    scale_path = m.VALIDATION_DIR / 'panel_scale_provenance.csv'
    check('panel_scale_provenance.csv was written (empty/pending in plan-only)', scale_path.exists())
finally:
    cleanup_fixture()
    restore_tree(m.OUT_DIR, out_dir_snapshot)
    if doc_2db_snapshot is None:
        if m.DOC_PATH.exists():
            m.DOC_PATH.unlink()
    else:
        m.DOC_PATH.write_bytes(doc_2db_snapshot)
check('golden-path fixture fully torn down (visual-review doc removed)', not m.VISUAL_REVIEW_DOC_PATH.exists())
check('golden-path fixture fully torn down (preview_audit dir removed)', not m.p2da.PREVIEW_AUDIT_DIR.exists())
check('Phase-2D-A report restored to its original (non-"complete") content',
      'Phase 2D-A complete.' not in m.p2da.DOC_PATH.read_text())
check('phase2db/ output tree restored to its exact pre-test snapshot (no committed content lost)',
      snapshot_tree(m.OUT_DIR) == out_dir_snapshot)
check('docs/unified_candidate_analysis_phase2db.md restored to its exact pre-test snapshot',
      (m.DOC_PATH.read_bytes() if m.DOC_PATH.exists() else None) == doc_2db_snapshot)

print()
print('=== 7. Metric reproduction (validate_figure_data_reproduction) ===')
per_sample_7 = {m.CNN: {5: dict(pd_distance=1.0, mt_distance=2.0, psnruv=3.0, speed_mae=4.0, grad_mae=5.0,
                                    wpd_mae=6.0)}}
matching_row = dict(figure_id=1, archetype_id='x', sample_idx=5, method_id=m.CNN, display_label='CNN',
                      pd_distance=1.0, mt_distance=2.0, psnruv=3.0, speed_mae=4.0, grad_mae=5.0, wpd_mae=6.0,
                      source_field_path='', source_gt_path='', source_topology_path='',
                      zoom_y0='', zoom_y1='', zoom_x0='', zoom_x1='')
mv_key = {('x', m.CNN): {'raw__pd_distance': '1.0', 'raw__mt_distance': '2.0', 'raw__psnruv': '3.0',
                            'raw__speed_mae': '4.0', 'raw__grad_mae': '5.0', 'raw__wpd_mae': '6.0'}}
repro_rows = m.validate_figure_data_reproduction({1: [matching_row]}, per_sample_7, mv_key)
check('matching figure-data values reproduce cleanly (no SystemExit, all PASS)',
      all(r['status'] == 'PASS' for r in repro_rows))

mismatched_row = dict(matching_row, pd_distance=999.0)
try:
    m.validate_figure_data_reproduction({1: [mismatched_row]}, per_sample_7, mv_key)
    check('mismatched figure-data value -> SystemExit hard-fail', False)
except SystemExit as e:
    check('mismatched figure-data value -> SystemExit hard-fail', 'reproduction check(s) failed' in str(e))

print()
print('=== 8. Common speed/error scaling (reused from Phase-2D-A, same panel data helper) ===')
gt_uv = np.zeros((4, 4, 2))
gt_uv[0, 0] = [3.0, 4.0]
sr_by_method = {mid: gt_uv.copy() for mid in (m.CNN, m.GAN, m.CANDIDATE_C)}
sr_by_method[m.GAN][1, 1] = [0.0, 50.0]  # far outside GT range
panel = m.p2da.compute_preview_panel_data(gt_uv, sr_by_method)
check('shared speed scaling expands to include an out-of-GT-range SR value (Phase-2D-A helper reused correctly)',
      panel['speed_vmax'] >= 50.0 - 1e-9)
check('exactly one shared (vmin, vmax) pair drives every panel', isinstance(panel['speed_vmin'], float)
      and isinstance(panel['speed_vmax'], float))

print()
print('=== 9. Deterministic zoom selection ===')
gt_speed_9 = np.zeros((300, 300))
gt_speed_9[220:260, 220:260] = np.linspace(0, 20, 40)[:, None] * np.ones((1, 40))
errors_9 = {m.CNN: np.zeros((300, 300)), m.F3: np.zeros((300, 300))}
errors_9[m.CNN][220:260, 220:260] = 1.0
zoom9 = m.select_deterministic_zoom(gt_speed_9, errors_9, window=40, stride=20)
check('deterministic zoom selects a window overlapping the high-energy/high-error region',
      zoom9['y0'] >= 180 and zoom9['x0'] >= 180)
zoom9b = m.select_deterministic_zoom(gt_speed_9, errors_9, window=40, stride=20)
check('deterministic zoom selection is reproducible across repeated calls', zoom9 == zoom9b)

gt_flat = np.ones((120, 120)) * 5.0
errors_flat = {m.CNN: np.zeros((120, 120))}
zoom_tie = m.select_deterministic_zoom(gt_flat, errors_flat, window=40, stride=40)
check('tie-break selects the smallest y0 then smallest x0 (top-left) when all window scores are equal',
      zoom_tie['y0'] == 0 and zoom_tie['x0'] == 0)

print()
print('=== 10. Zoom propagation to all 4 destinations ===')
tmp10 = Path(tempfile.mkdtemp(prefix='phase2db_test_'))
manifest10 = dict(m.FROZEN_SAMPLE_SET)
old_fd_fields_dir = m.FIGURE_DATA_DIR
old_plan_dir = m.PLAN_DIR
try:
    m.FIGURE_DATA_DIR = tmp10 / 'figure_data'
    m.PLAN_DIR = tmp10 / 'plan'
    m.FIGURE_DATA_DIR.mkdir(parents=True)
    m.PLAN_DIR.mkdir(parents=True)
    fig3 = m.FIGURE_BY_ID[3]
    fd_rows = [dict(figure_id=3, archetype_id=fig3['archetype_id'], sample_idx=manifest10[fig3['archetype_id']],
                       method_id=mid, display_label='', pd_distance='', mt_distance='', psnruv='', speed_mae='',
                       grad_mae='', wpd_mae='', source_field_path='', source_gt_path='', source_topology_path='',
                       zoom_y0='', zoom_y1='', zoom_x0='', zoom_x1='')
                for mid in ['GT'] + fig3['required_methods']]
    m.write_csv(m.FIGURE_DATA_DIR / m.FIGURE_DATA_FILENAMES[3], m.FIGURE_DATA_FIELDS, fd_rows)
    pm_rows = [dict(figure_id=3, archetype_id=fig3['archetype_id'], sample_idx=manifest10[fig3['archetype_id']],
                       panel_type=m.ZOOM_CROP, method_id='', display_label='', method_role='',
                       output_path='ttk_runs_fixed/x.png', requires_manual_topology_input=False,
                       requires_pd_coordinate_source=False, pd_coordinate_source_found='',
                       status='planned_not_rendered')]
    m.write_csv(m.PLAN_DIR / 'final_panel_manifest.csv', m.FINAL_PANEL_MANIFEST_FIELDS, pm_rows)

    fake_zoom = dict(y0=10, y1=110, x0=20, x1=120, score=42.5, formula=m.ZOOM_SCORE_FORMULA)
    m.propagate_zoom_result(fake_zoom, manifest10)

    fd_after = m.p2da.read_csv_dicts(m.FIGURE_DATA_DIR / m.FIGURE_DATA_FILENAMES[3])
    check('zoom bounds propagated into every figure-data row for figure 3',
          all(int(r['zoom_y0']) == 10 and int(r['zoom_y1']) == 110 and int(r['zoom_x0']) == 20
              and int(r['zoom_x1']) == 120 for r in fd_after))
    pm_after = m.p2da.read_csv_dicts(m.PLAN_DIR / 'final_panel_manifest.csv')
    check('zoom_crop row in final_panel_manifest.csv is marked rendered after propagation',
          pm_after[0]['status'] == 'rendered')

    old_validation_dir = m.VALIDATION_DIR
    m.VALIDATION_DIR = tmp10 / 'validation'
    m.VALIDATION_DIR.mkdir(parents=True)
    try:
        m.write_zoom_selection_validation(fake_zoom, manifest10)
        zv_rows = m.p2da.read_csv_dicts(m.VALIDATION_DIR / 'zoom_selection_validation.csv')
        check('zoom_selection_validation.csv records the exact bounds and score',
              zv_rows[0]['y0'] == '10' and zv_rows[0]['x1'] == '120' and zv_rows[0]['status'] == 'computed')
    finally:
        m.VALIDATION_DIR = old_validation_dir
finally:
    m.FIGURE_DATA_DIR = old_fd_fields_dir
    m.PLAN_DIR = old_plan_dir
    shutil.rmtree(tmp10, ignore_errors=True)

print()
print('=== 11. Real PD source discovery (available / pending / unavailable routing) ===')
# discover_pd_source_candidates logs repo-relative search roots (_rel_posix requires
# every root to live under REPO_ROOT), so the fixture must sit inside the repo tree,
# not /tmp -- matching every real search root used in production. Each sub-scenario
# gets its OWN isolated subdirectory + topology_source_map entry so files created for
# one sample_idx can never leak into another sample_idx's search (both tokens are
# matched with OR, by design, so cross-contamination would otherwise be possible).
tmp11 = m.OUT_DIR / '_test11_scratch'
try:
    # --- 11a: nothing at all present for this (sample, method) ---
    dir_a = tmp11 / 'scenario_a'
    dir_a.mkdir(parents=True)
    src_a = {m.CNN: dict(path=dir_a / 'unrelated_placeholder.csv', pd_column='pd', mt_column='mt',
                            is_shared_combined=False)}
    candidates_a = m.discover_pd_source_candidates(1, 501, m.CNN, src_a)
    check('no coordinate file present anywhere in the search roots -> a none_found candidate row',
          any(c['artifact_type'] == 'none_found' for c in candidates_a))
    check('GT is never marked found without a concrete candidate_path (structural check on discovery output)',
          all(r['candidate_path'] for r in candidates_a if r['artifact_type'] != 'none_found'))

    # --- 11b: a real coordinate CSV matching the naming convention ---
    dir_b = tmp11 / 'scenario_b'
    dir_b.mkdir(parents=True)
    src_b = {m.CNN: dict(path=dir_b / 'unrelated_placeholder.csv', pd_column='pd', mt_column='mt',
                            is_shared_combined=False)}
    coord_file = dir_b / f'{m.CNN}_sample502_pd_diagram.csv'
    coord_file.write_text('birth,death\n0.1,0.2\n0.3,0.9\n')
    candidates_b = [m._finalize_pd_candidate_row(c)
                     for c in m.discover_pd_source_candidates(1, 502, m.CNN, src_b)]
    coord_rows_b = [c for c in candidates_b if c['artifact_type'] == 'csv_pd_coordinates']
    check('a coordinate CSV matching the naming convention is discovered', len(coord_rows_b) == 1)
    check('a finite, mapped coordinate CSV resolves to available_validated',
          m.figure_pd_source_verdict(candidates_b) == m.STATUS_AVAILABLE)

    # --- 11c: only a scalar-only distance summary is present ---
    dir_c = tmp11 / 'scenario_c'
    dir_c.mkdir(parents=True)
    src_c = {m.CNN: dict(path=dir_c / f'{m.CNN}_sample503_scalar.csv', pd_column='pd', mt_column='mt',
                            is_shared_combined=False)}
    src_c[m.CNN]['path'].write_text('sample_idx,pd_distance,mt_distance\n503,1.0,2.0\n')
    candidates_c = [m._finalize_pd_candidate_row(c)
                     for c in m.discover_pd_source_candidates(1, 503, m.CNN, src_c)]
    check('a scalar-only summary CSV is classified as csv_scalar_summary, not a coordinate source',
          any(c['artifact_type'] == 'csv_scalar_summary' for c in candidates_c)
          and not any(c['artifact_type'] == 'csv_pd_coordinates' for c in candidates_c))
    check('scalar-only summary alone resolves to pending (never confirms absence of a real coordinate source)',
          m.figure_pd_source_verdict(candidates_c) == m.STATUS_PENDING)

    # --- 11d: bicubic (never in topology_source_map) gets a real, non-trivial search ---
    dir_d = tmp11 / 'scenario_d'
    dir_d.mkdir(parents=True)
    src_d = {m.CNN: dict(path=dir_d / 'unrelated_placeholder.csv', pd_column='pd', mt_column='mt',
                            is_shared_combined=False)}
    bicubic_candidates_before = m.discover_pd_source_candidates(2, 504, m.BICUBIC, src_d)
    check('bicubic discovery runs a real search (returns at least the none_found summary row)',
          len(bicubic_candidates_before) >= 1 and bicubic_candidates_before[0]['method_id'] == m.BICUBIC
          and bicubic_candidates_before[0]['artifact_type'] == 'none_found')
    bicubic_coord = dir_d / 'bicubic_sample504_pd_diagram.csv'
    bicubic_coord.write_text('birth,death\n0.0,1.0\n')
    bicubic_candidates_after = m.discover_pd_source_candidates(2, 504, m.BICUBIC, src_d)
    check('bicubic PD discovery finds a real matching artifact when one exists (not a hardcoded exclusion)',
          any(c['artifact_type'] == 'csv_pd_coordinates' for c in bicubic_candidates_after))

    # --- 11e: non-finite coordinate values are rejected ---
    dir_e = tmp11 / 'scenario_e'
    dir_e.mkdir(parents=True)
    src_e = {m.CNN: dict(path=dir_e / 'unrelated_placeholder.csv', pd_column='pd', mt_column='mt',
                            is_shared_combined=False)}
    non_finite_file = dir_e / f'{m.CNN}_sample505_pd_diagram.csv'
    non_finite_file.write_text('birth,death\nnan,0.5\n')
    candidates_e = [m._finalize_pd_candidate_row(c)
                     for c in m.discover_pd_source_candidates(1, 505, m.CNN, src_e)]
    matched_e = next(c for c in candidates_e if c['artifact_type'] == 'csv_pd_coordinates')
    check('non-finite coordinate values are rejected (finite_status/usable_status reflect this)',
          matched_e['finite_status'] == 'non_finite_or_unreadable')
finally:
    shutil.rmtree(tmp11, ignore_errors=True)

print()
print('=== 12. PD coordinate rendering produces a real panel; scalar fallback used only after unavailable ===')
tmp12 = Path(tempfile.mkdtemp(prefix='phase2db_test_'))
panel_dirs_created = []
old_panels_dir = m.PANELS_DIR
try:
    m.PANELS_DIR = tmp12
    coord_dir = tmp12 / 'coords'
    coord_dir.mkdir()
    gt_coord = coord_dir / 'gt.csv'
    gt_coord.write_text('birth,death\n0.0,1.0\n0.2,0.8\n')
    cnn_coord = coord_dir / 'cnn.csv'
    cnn_coord.write_text('birth,death\n0.1,0.9\n0.3,0.6\n')
    fig2 = m.FIGURE_BY_ID[2]
    manifest12 = dict(m.FROZEN_SAMPLE_SET)
    per_sample12 = {mid: {manifest12[fig2['archetype_id']]: dict(pd_distance=3.14, mt_distance=1.0)}
                      for mid in fig2['full_panel_methods']}
    pd_sources = {'GT': str(gt_coord.relative_to(m.REPO_ROOT)) if gt_coord.is_relative_to(m.REPO_ROOT) else None,
                   m.CNN: str(cnn_coord)}
    # Coordinate sources must be repo-relative for load_validated_pd_coordinates; place them under REPO_ROOT instead.
finally:
    m.PANELS_DIR = old_panels_dir
    shutil.rmtree(tmp12, ignore_errors=True)

# Re-run with real repo-relative coordinate files (required by load_validated_pd_coordinates).
tmp12b_root = m.OUT_DIR / '_test_pd_coords'
try:
    tmp12b_root.mkdir(parents=True, exist_ok=True)
    gt_coord = tmp12b_root / 'gt.csv'
    gt_coord.write_text('birth,death\n0.0,1.0\n0.2,0.8\n')
    cnn_coord = tmp12b_root / 'cnn.csv'
    cnn_coord.write_text('birth,death\n0.1,0.9\n0.3,0.6\n')
    fig2 = m.FIGURE_BY_ID[2]
    manifest12 = dict(m.FROZEN_SAMPLE_SET)
    si12 = manifest12[fig2['archetype_id']]
    per_sample12 = {mid: {si12: dict(pd_distance=3.14, mt_distance=1.0)} for mid in fig2['full_panel_methods']}
    pd_sources = {'GT': str(gt_coord.relative_to(m.REPO_ROOT)), m.CNN: str(cnn_coord.relative_to(m.REPO_ROOT))}
    rows = m.render_pd_diagram_panels(fig2, m.PD_COMPARISON, manifest12, per_sample12, pd_sources)
    panel_dirs_created.append(m.PANELS_DIR / m.figure_dir_name(fig2))
    check('render_pd_diagram_panels produces one real panel per available method (GT + CNN)', len(rows) == 2)
    for r in rows:
        p = m.REPO_ROOT / r['output_path']
        check(f"real PD coordinate panel {r['output_path']} exists and is non-empty", p.exists() and p.stat().st_size > 0)

    fallback_row = m.render_scalar_pd_fallback_panel(fig2, m.PD_COMPARISON, manifest12, per_sample12, [m.GAN])
    p = m.REPO_ROOT / fallback_row['output_path']
    check('scalar PD fallback panel is a distinct, real, non-empty file', p.exists() and p.stat().st_size > 0)
    check('scalar fallback panel path is distinct from the coordinate-based panel path',
          fallback_row['output_path'] not in {r['output_path'] for r in rows})
finally:
    shutil.rmtree(tmp12b_root, ignore_errors=True)
    for d in panel_dirs_created:
        shutil.rmtree(d, ignore_errors=True)

print()
print('=== 13. Every declared scripted panel type has a real renderer ===')
DECLARED_SCRIPTED_PANEL_TYPES = {m.SPEED_FIELDS, m.ERROR_MAPS, m.METRIC_STRIP, m.PD_EVIDENCE, m.PD_COMPARISON,
                                    m.PD_MT_TRADEOFF_COMPACT, m.PD_MT_COMPARISON_COMPACT, m.ZOOM_CROP}
RENDERER_BY_PANEL_TYPE = {
    m.SPEED_FIELDS: m.render_speed_and_error_panels, m.ERROR_MAPS: m.render_speed_and_error_panels,
    m.METRIC_STRIP: m.render_metric_strip, m.PD_EVIDENCE: m.render_pd_diagram_panels,
    m.PD_COMPARISON: m.render_pd_diagram_panels, m.PD_MT_TRADEOFF_COMPACT: m.render_pd_mt_tradeoff_compact_panel,
    m.PD_MT_COMPARISON_COMPACT: m.render_pd_mt_comparison_compact_panel, m.ZOOM_CROP: m.render_zoom_crop_panel,
}
check('every declared scripted panel type used across all figure contracts has a mapped real renderer',
      set().union(*[set(c['panels']) & DECLARED_SCRIPTED_PANEL_TYPES for c in m.FIGURE_CONTRACTS])
      <= set(RENDERER_BY_PANEL_TYPE))
all_declared_panel_types = set().union(*[set(c['panels']) for c in m.FIGURE_CONTRACTS])
check('no declared panel type in any figure contract is unhandled (scripted-renderer or manual-MT set)',
      all_declared_panel_types <= (DECLARED_SCRIPTED_PANEL_TYPES | m.MT_PANEL_TYPES))

print()
print('=== 14. Panel dimensions + real PNG/PDF inspection (Section 9) ===')
old_in_shape, old_hr_shape = m.p2da.EXPECTED_IN_SHAPE, m.p2da.EXPECTED_HR_SHAPE
panel_dirs_created = []
try:
    m.p2da.EXPECTED_IN_SHAPE = (m.p2da.N_EVAL, 4, 4, 2)
    m.p2da.EXPECTED_HR_SHAPE = (m.p2da.N_EVAL, 12, 12, 2)
    rng14 = np.random.default_rng(7)
    fake_gt = rng14.normal(size=(12, 12, 2)).astype(np.float32) * 3 + 1
    fake_audit = dict(selected_data={
        m.CNN: dict(gt=np.stack([fake_gt]), sr=np.stack([fake_gt + rng14.normal(size=fake_gt.shape) * 0.1])),
        m.BICUBIC: dict(gt=np.stack([fake_gt]), sr=np.stack([fake_gt + rng14.normal(size=fake_gt.shape) * 0.2])),
        m.GAN: dict(gt=np.stack([fake_gt]), sr=np.stack([fake_gt + rng14.normal(size=fake_gt.shape) * 0.15])),
    })
    fig2 = m.FIGURE_BY_ID[2]
    manifest14 = dict(m.FROZEN_SAMPLE_SET)
    panel_rows, gt_speed14, method_speeds14, panel14 = m.render_speed_and_error_panels(
        fig2, manifest14, fake_audit, [manifest14['gan_pd_vs_cnn_mt_conflict']])
    panel_dirs_created.append(m.PANELS_DIR / m.figure_dir_name(fig2))
    check('render_speed_and_error_panels produced one GT + (speed+error) per method',
          len(panel_rows) == 1 + 2 * len(fig2['full_panel_methods']))
    for r in panel_rows:
        insp = m.inspect_png(m.REPO_ROOT / r['output_path'])
        check(f"inspect_png measures real >=300px dimensions for {r['output_path']} (never hardcoded)",
              insp['validation_status'] == 'PASS' and insp['width_px'] > 0 and insp['height_px'] > 0)

    empty_png = m.PANELS_DIR / 'empty_test.png'
    empty_png.write_bytes(b'')
    insp_empty = m.inspect_png(empty_png)
    check('inspect_png flags an empty file rather than reporting a fabricated pass',
          insp_empty['validation_status'] == 'FAIL_missing_or_empty')
    empty_png.unlink()

    corrupt_png = m.PANELS_DIR / 'corrupt_test.png'
    corrupt_png.write_bytes(b'not a real png')
    insp_corrupt = m.inspect_png(corrupt_png)
    check('inspect_png flags a corrupt file rather than reporting a fabricated pass',
          insp_corrupt['validation_status'] == 'FAIL_corrupt_or_unreadable')
    corrupt_png.unlink()

    missing_pdf = m.PANELS_DIR / 'missing.pdf'
    insp_pdf_missing = m.inspect_pdf(missing_pdf)
    check('inspect_pdf reports FAIL for a missing file (dpi/page-count never assumed)',
          insp_pdf_missing['validation_status'] == 'FAIL_missing_or_empty' and insp_pdf_missing['pdf_page_count'] == 0)
finally:
    m.p2da.EXPECTED_IN_SHAPE, m.p2da.EXPECTED_HR_SHAPE = old_in_shape, old_hr_shape
    for d in panel_dirs_created:
        shutil.rmtree(d, ignore_errors=True)

print()
print('=== 15. Transactional rendering: a mid-figure failure leaves no partial authoritative directory ===')
fig_probe = m.FIGURE_BY_ID[4]
final_dir_probe = m.PANELS_DIR / m.figure_dir_name(fig_probe)
if final_dir_probe.exists():
    shutil.rmtree(final_dir_probe)
try:
    def failing_render():
        # Simulate partial success (one real file written) then a failure.
        (m.PANELS_DIR / m.figure_dir_name(fig_probe)).mkdir(parents=True, exist_ok=True)
        (m.PANELS_DIR / m.figure_dir_name(fig_probe) / 'partial.png').write_bytes(b'\x89PNG\r\n\x1a\n')
        raise SystemExit('[hard-fail] simulated failure mid-render')

    try:
        m.render_figure_transactional(fig_probe, failing_render)
        check('a simulated mid-figure render failure propagates as SystemExit', False)
    except SystemExit:
        check('a simulated mid-figure render failure propagates as SystemExit', True)
    check('no partial authoritative panel directory was left after the failure', not final_dir_probe.exists())
    check('no leftover staging directory remains under _staging/', not (m.OUT_DIR / '_staging').exists()
          or not any((m.OUT_DIR / '_staging').iterdir()))
finally:
    if final_dir_probe.exists():
        shutil.rmtree(final_dir_probe)
    shutil.rmtree(m.OUT_DIR / '_staging', ignore_errors=True)

print()
print('=== 16. Missing visual-review record / preview-audit file hard-fails (real checkout state) ===')
try:
    m.require_completed_phase2d_a_state(m.FROZEN_SAMPLE_SET)
    check('missing Phase-2D-A completed state / visual-review record -> SystemExit', False)
except SystemExit as e:
    check('missing Phase-2D-A completed state / visual-review record -> SystemExit',
          'completed Phase-2D-A state' in str(e) or 'visual-review record' in str(e))

try:
    m.require_protected_files()
    check('missing Phase-2D-A preview-audit/preview-PNG/log files -> SystemExit (real checkout state)', False)
except SystemExit as e:
    msg = str(e)
    check('missing Phase-2D-A preview-audit/preview-PNG/log files -> SystemExit (real checkout state)',
          'preview_audit' in msg and 'previews' in msg and 'visual_review' in msg)

print()
print('=== 17. Manifest-driven composite assembly (no globbing, order/completeness enforced) ===')
tmp17 = Path(tempfile.mkdtemp(prefix='phase2db_test_'))
old_plan_dir17 = m.PLAN_DIR
panel_dirs_created17 = []
composite_paths17 = []
try:
    m.PLAN_DIR = tmp17
    manifest17 = dict(m.FROZEN_SAMPLE_SET)
    fig4 = m.FIGURE_BY_ID[4]  # needs no manual MT panel
    pd_verdicts_by_figure = {c['figure_id']: {} for c in m.FIGURE_CONTRACTS}
    composite_rows17 = m.build_final_composite_manifest_rows(manifest17, pd_verdicts_by_figure)
    m.write_csv(m.PLAN_DIR / 'final_composite_manifest.csv', m.FINAL_COMPOSITE_MANIFEST_FIELDS, composite_rows17)

    fig4_rows = [r for r in composite_rows17 if int(r['figure_id']) == 4]
    check('composite manifest for figure 4 preserves declared panel_order ascending',
          [int(r['panel_order']) for r in fig4_rows] == sorted(int(r['panel_order']) for r in fig4_rows))

    panel_dir = m.PANELS_DIR / m.figure_dir_name(fig4)
    panel_dir.mkdir(parents=True, exist_ok=True)
    panel_dirs_created17.append(panel_dir)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for r in fig4_rows:
        p = m.REPO_ROOT / r['source_path']
        p.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(1, 1), dpi=50)
        ax.imshow(np.random.default_rng(0).uniform(size=(10, 10)))
        fig.savefig(p)
        plt.close(fig)

    row = m.build_composite_for_figure(fig4, manifest17)
    composite_paths17 += [m.REPO_ROOT / row['expected_png_path'], m.REPO_ROOT / row['expected_vector_path']]
    check('composite assembly succeeds when every manifest-declared panel is present',
          row['status'] == 'rendered' and (m.REPO_ROOT / row['expected_png_path']).exists())
    check('composite PDF passes real inspection (valid header + nonzero page count)', row['pdf_valid'] is True)
    check('composite is honestly described as a raster-panel PDF (panels are pre-rendered raster images)',
          row['vector_kind'] == 'raster_panel_pdf')

    # Remove one required panel -> must hard-fail (never assemble an incomplete composite).
    missing_target = m.REPO_ROOT / fig4_rows[0]['source_path']
    missing_target.unlink()
    try:
        m.build_composite_for_figure(fig4, manifest17)
        check('composite assembly fails when one required panel is absent', False)
    except SystemExit as e:
        check('composite assembly fails when one required panel is absent', 'missing on disk' in str(e))
    # Restore it for the next sub-tests.
    fig, ax = plt.subplots(figsize=(1, 1), dpi=50)
    ax.imshow(np.random.default_rng(0).uniform(size=(10, 10)))
    fig.savefig(missing_target)
    plt.close(fig)

    # Add an unexpected extra PNG -> must be rejected.
    stray = panel_dir / 'unexpected_stray_panel.png'
    fig, ax = plt.subplots(figsize=(1, 1), dpi=50)
    ax.imshow(np.zeros((5, 5)))
    fig.savefig(stray)
    plt.close(fig)
    try:
        m.build_composite_for_figure(fig4, manifest17)
        check('composite assembly rejects an unexpected panel not declared in the manifest', False)
    except SystemExit as e:
        check('composite assembly rejects an unexpected panel not declared in the manifest',
              'unexpected PNG' in str(e))
    stray.unlink()

    # Three arbitrary synthetic PNGs cannot satisfy a figure contract on their own
    # (without a matching, complete composite manifest for that figure).
    tmp_isolated_dir = m.PANELS_DIR / m.figure_dir_name(m.FIGURE_BY_ID[6])
    tmp_isolated_dir.mkdir(parents=True, exist_ok=True)
    panel_dirs_created17.append(tmp_isolated_dir)
    for i in range(3):
        fig, ax = plt.subplots(figsize=(1, 1), dpi=50)
        ax.imshow(np.zeros((5, 5)))
        fig.savefig(tmp_isolated_dir / f'arbitrary_{i}.png')
        plt.close(fig)
    empty_manifest_path = m.PLAN_DIR / 'final_composite_manifest.csv'
    m.write_csv(empty_manifest_path, m.FINAL_COMPOSITE_MANIFEST_FIELDS, fig4_rows)  # no rows for figure 6
    try:
        m.build_composite_for_figure(m.FIGURE_BY_ID[6], manifest17)
        check('three arbitrary synthetic PNGs cannot satisfy a figure contract without manifest rows', False)
    except SystemExit as e:
        check('three arbitrary synthetic PNGs cannot satisfy a figure contract without manifest rows',
              'no rows in' in str(e) or 'unexpected PNG' in str(e))
finally:
    m.PLAN_DIR = old_plan_dir17
    shutil.rmtree(tmp17, ignore_errors=True)
    for p in composite_paths17:
        if p.exists():
            p.unlink()
    for d in panel_dirs_created17:
        shutil.rmtree(d, ignore_errors=True)
    shutil.rmtree(m.FIGURES_DIR, ignore_errors=True)

print()
print('=== 18. Manual MT metadata identity/dimension validation ===')
created_dirs18 = []
try:
    manual_rows = m.build_manual_topology_requirements_rows(m.FROZEN_SAMPLE_SET)
    for r in manual_rows:
        panel_path = m.REPO_ROOT / r['expected_panel_path']
        panel_path.parent.mkdir(parents=True, exist_ok=True)
        created_dirs18.append(panel_path.parent)
        Image.new('RGB', (320, 320), color='black').save(panel_path)
        meta_path = m.REPO_ROOT / r['expected_metadata_path']
        row = {f: 'x' for f in m.MANUAL_TOPOLOGY_METADATA_FIELDS}
        row.update(figure_id=str(r['figure_id']), sample_idx=str(r['sample_idx']), method_id=r['method_id'],
                     source_vtu_path='ttk_runs_fixed/some/relative/path.vtu', persistence_threshold='11.0',
                     arc_sampling='10', arc_line_size='3', image_width='320', image_height='320',
                     camera_or_view_id='view0', scalar_range='0,10')
        with meta_path.open('w', newline='') as fh:
            w = _csv.DictWriter(fh, fieldnames=m.MANUAL_TOPOLOGY_METADATA_FIELDS)
            w.writeheader()
            w.writerow(row)

    result_rows = m.require_manual_topology_panels(m.FROZEN_SAMPLE_SET)
    check('fully consistent manual panels + metadata pass validation cleanly',
          len(result_rows) == len(manual_rows))

    # Break identity: wrong sample_idx in one metadata row.
    bad_row = dict(row)
    bad_row['sample_idx'] = '999999'
    one_meta_path = m.REPO_ROOT / manual_rows[0]['expected_metadata_path']
    with one_meta_path.open('w', newline='') as fh:
        w = _csv.DictWriter(fh, fieldnames=m.MANUAL_TOPOLOGY_METADATA_FIELDS)
        w.writeheader()
        w.writerow(dict(bad_row, figure_id=str(manual_rows[0]['figure_id']), method_id=manual_rows[0]['method_id']))
    try:
        m.require_manual_topology_panels(m.FROZEN_SAMPLE_SET)
        check('metadata sample_idx mismatch -> SystemExit', False)
    except SystemExit as e:
        check('metadata sample_idx mismatch -> SystemExit', 'sample_idx' in str(e))

    # Restore, then break declared image dimensions vs actual PNG.
    good_row = dict(row, figure_id=str(manual_rows[0]['figure_id']), sample_idx=str(manual_rows[0]['sample_idx']),
                      method_id=manual_rows[0]['method_id'])
    with one_meta_path.open('w', newline='') as fh:
        w = _csv.DictWriter(fh, fieldnames=m.MANUAL_TOPOLOGY_METADATA_FIELDS)
        w.writeheader()
        mismatched_dims = dict(good_row, image_width='9999', image_height='9999')
        w.writerow(mismatched_dims)
    try:
        m.require_manual_topology_panels(m.FROZEN_SAMPLE_SET)
        check('metadata image_width/image_height mismatch vs actual PNG dimensions -> SystemExit', False)
    except SystemExit as e:
        check('metadata image_width/image_height mismatch vs actual PNG dimensions -> SystemExit',
              'does not match the actual PNG pixel dimensions' in str(e))

    # Restore correctly, then break cross-panel comparability (differing threshold within a figure).
    with one_meta_path.open('w', newline='') as fh:
        w = _csv.DictWriter(fh, fieldnames=m.MANUAL_TOPOLOGY_METADATA_FIELDS)
        w.writeheader()
        w.writerow(good_row)
    fig1_rows = [r for r in manual_rows if r['figure_id'] == 1]
    if len(fig1_rows) >= 2:
        other_meta_path = m.REPO_ROOT / fig1_rows[1]['expected_metadata_path']
        other_row = dict(row, figure_id=str(fig1_rows[1]['figure_id']), sample_idx=str(fig1_rows[1]['sample_idx']),
                           method_id=fig1_rows[1]['method_id'], persistence_threshold='99.0')
        with other_meta_path.open('w', newline='') as fh:
            w = _csv.DictWriter(fh, fieldnames=m.MANUAL_TOPOLOGY_METADATA_FIELDS)
            w.writeheader()
            w.writerow(other_row)
        try:
            m.require_manual_topology_panels(m.FROZEN_SAMPLE_SET)
            check('inconsistent persistence_threshold across one figure\'s manual panels -> SystemExit', False)
        except SystemExit as e:
            check('inconsistent persistence_threshold across one figure\'s manual panels -> SystemExit',
                  'not comparable' in str(e))
finally:
    for d in set(created_dirs18):
        shutil.rmtree(d, ignore_errors=True)
    if m.MANUAL_TOPOLOGY_DIR.exists():
        shutil.rmtree(m.MANUAL_TOPOLOGY_DIR, ignore_errors=True)
check('manual_topology_inputs test scaffolding fully cleaned up', not m.MANUAL_TOPOLOGY_DIR.exists())

print()
print('=== 19. Caption/figure ordering agreement ===')
check('CAPTION_TEMPLATES keys are exactly {1..6}', set(m.CAPTION_TEMPLATES) == {1, 2, 3, 4, 5, 6})
check('every caption is sample-specific ("For sample" or "this selected/illustrative" framing present)',
      all(('for sample' in txt.lower() or 'illustrative' in txt.lower() or 'selected example' in txt.lower())
          for txt in m.CAPTION_TEMPLATES.values()))
FORBIDDEN_OVERCLAIMS = ['always preserves', 'universally improves', 'is inaccurate', 'universally superior']
check('no caption contains a forbidden overclaiming phrase',
      all(not any(bad in txt.lower() for bad in FORBIDDEN_OVERCLAIMS) for txt in m.CAPTION_TEMPLATES.values()))

manifest19 = dict(m.FROZEN_SAMPLE_SET)
# CAPTIONS_PATH is a module-level constant bound to the real PLAN_DIR at import
# time (reassigning m.PLAN_DIR does not redirect it), so write_captions_md always
# writes the real file -- snapshot/restore it exactly rather than redirecting.
captions_snapshot = m.CAPTIONS_PATH.read_bytes() if m.CAPTIONS_PATH.exists() else None
try:
    m.write_captions_md(manifest19)
    captions_text = m.CAPTIONS_PATH.read_text()
    order_found = [int(tok) for tok in re.findall(r'## Figure (\d+):', captions_text)]
    check('final_figure_captions.md lists figures in ascending 1..6 order matching FIGURE_CONTRACTS',
          order_found == [1, 2, 3, 4, 5, 6])
finally:
    if captions_snapshot is None:
        if m.CAPTIONS_PATH.exists():
            m.CAPTIONS_PATH.unlink()
    else:
        m.CAPTIONS_PATH.write_bytes(captions_snapshot)

print()
print('=== 20. Final completion cannot be reported with any blocked/pending panel ===')
tmp20 = Path(tempfile.mkdtemp(prefix='phase2db_test_'))
try:
    fake_final_rows = [
        dict(figure_id=i, archetype_id='x', sample_idx=0, expected_png_path='a.png',
              expected_vector_path='a.pdf', png_exists=True, width_px=1000, height_px=1000, dpi_x=300, dpi_y=300,
              png_file_size_bytes=100, png_min_dpi_ok=True, vector_exists=True, pdf_page_count=1,
              pdf_file_size_bytes=100, pdf_valid=True, vector_kind='raster_panel_pdf', status='rendered')
        for i in range(1, 7)
    ]
    fake_final_rows[2]['status'] = 'blocked'  # figure 3 still blocked
    not_ready = [r for r in fake_final_rows if not (r['status'] == 'rendered' and r['png_exists']
                                                       and r['png_min_dpi_ok'] and r['vector_exists']
                                                       and r['pdf_valid'])]
    check('a single blocked/non-rendered figure among the six prevents a "complete" verdict', len(not_ready) == 1)
finally:
    shutil.rmtree(tmp20, ignore_errors=True)

print()
print('=== 21. Missing protected file -> hard fail ===')
target = m.PHASE2D_A_SELECTION_CSVS[0]
backup = target.with_suffix('.csv.bak_test2db')
try:
    target.rename(backup)
    try:
        m.require_protected_files()
        check('missing Phase-2D-A protected file -> SystemExit', False)
    except SystemExit as e:
        check('missing Phase-2D-A protected file -> SystemExit', str(target) in str(e))
finally:
    if backup.exists():
        backup.rename(target)
check('file restored', target.exists())

print()
print('=== 22. Altered protected checksum -> hard fail ===')
target2 = m.p2da.PHASE1_PROTECTED_CSVS[0]
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
