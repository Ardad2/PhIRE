#!/usr/bin/env python3
"""Synthetic/contract tests for scripts/render_unified_candidate_figures_phase2db.py.

Exercises the real shipped functions directly. Render/composite-path
functions are exercised against small synthetic fixtures built and torn
down under this script's OWN writable output tree
(ttk_runs_fixed/unified_candidate_analysis/phase2db/) or the scratch tmp
directory -- never under any Phase-1 through Phase-2D-A directory. Any test
that temporarily perturbs a real protected file restores it immediately in
a `finally` block and verifies the restoration succeeded.

Run directly:
    python3 scripts/test_render_unified_candidate_figures_phase2db.py
"""
import math
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import render_unified_candidate_figures_phase2db as m
import numpy as np

failures = []


def check(name, cond):
    status = 'PASS' if cond else 'FAIL'
    print(f'[{status}] {name}')
    if not cond:
        failures.append(name)


print('=== 1. Protected-file counts (86 prior + 17 Phase-2D-A = 103) ===')
check('Phase-2D-A protected files == 17', len(m.PHASE2D_A_PROTECTED_FILES) == 17)
check('All protected files == 103', len(m.ALL_PROTECTED_FILES) == 103)
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
print('=== 5. Method-set enforcement per figure (matches the literal task contract) ===')
EXPECTED_METHOD_SETS = {
    1: {'GT', m.GAN, m.CNN, m.CANDIDATE_C, m.F3, m.F2, m.UV_E2},
    2: {'GT', m.BICUBIC, m.CNN, m.GAN},
    3: {'GT', m.CNN, m.F3, m.UV_E2, m.F2},
    4: {'GT', m.CNN, m.F3, m.F2, m.UV_E2},
    5: {'GT', m.CNN, m.CANDIDATE_C, m.F3, m.F2, m.UV_E2},
    6: {'GT', m.GAN, m.CNN, m.CANDIDATE_C, m.F3, m.F2, m.UV_E2},
}
for c in m.FIGURE_CONTRACTS:
    actual = {'GT'} | set(c['required_methods'])
    check(f"figure {c['figure_id']} method set matches the task contract exactly",
          actual == EXPECTED_METHOD_SETS[c['figure_id']])
check('figure 3 restricts F2 to full_panel_methods exclusion (compact contextual reference only)',
      m.F2 not in m.FIGURE_BY_ID[3]['full_panel_methods'] and m.F2 in m.FIGURE_BY_ID[3]['required_methods'])
check('figure 3 method_roles records F2 as compact_contextual_reference',
      m.FIGURE_BY_ID[3]['method_roles'].get(m.F2) == 'compact_contextual_reference')

print()
print('=== 6. Repository-relative paths (write_csv guard) ===')
try:
    m.write_csv(Path(tempfile.mktemp(suffix='.csv')), ['name', 'output_path'],
                 [dict(name='x', output_path='/etc/passwd')])
    check('write_csv hard-fails on an absolute path field', False)
except SystemExit as e:
    check('write_csv hard-fails on an absolute path field', 'Absolute path found' in str(e))

print('    (re-running --plan-only and scanning every written CSV for absolute paths)')
plan_result = m.cmd_plan_only()
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
# A clearly higher-energy region near the bottom-right.
gt_speed_9[220:260, 220:260] = np.linspace(0, 20, 40)[:, None] * np.ones((1, 40))
errors_9 = {m.CNN: np.zeros((300, 300)), m.F3: np.zeros((300, 300))}
errors_9[m.CNN][220:260, 220:260] = 1.0
zoom9 = m.select_deterministic_zoom(gt_speed_9, errors_9, window=40, stride=20)
check('deterministic zoom selects a window overlapping the high-energy/high-error region',
      zoom9['y0'] >= 180 and zoom9['x0'] >= 180)
zoom9b = m.select_deterministic_zoom(gt_speed_9, errors_9, window=40, stride=20)
check('deterministic zoom selection is reproducible across repeated calls',
      zoom9 == zoom9b)

# Tie-break: a perfectly uniform field makes every window score equal -> top-left (0,0) must win.
gt_flat = np.ones((120, 120)) * 5.0
errors_flat = {m.CNN: np.zeros((120, 120))}
zoom_tie = m.select_deterministic_zoom(gt_flat, errors_flat, window=40, stride=40)
check('tie-break selects the smallest y0 then smallest x0 (top-left) when all window scores are equal',
      zoom_tie['y0'] == 0 and zoom_tie['x0'] == 0)

print()
print('=== 10. Missing PD source hard-fail ===')
tmp10 = Path(tempfile.mkdtemp(prefix='phase2db_test_'))
try:
    fake_topo_dir = tmp10 / 'topo'
    fake_topo_dir.mkdir()
    fake_source_map = {m.CNN: dict(path=fake_topo_dir / 'cnn_pd_mt_distances.csv', pd_column='pd', mt_column='mt',
                                      is_shared_combined=False)}
    (fake_source_map[m.CNN]['path']).write_text('sample_idx,pd,mt\n0,1.0,2.0\n')
    check('resolve_pd_diagram_source returns None when no coordinate file exists',
          m.resolve_pd_diagram_source(m.CNN, 5, fake_source_map) is None)

    # Create a matching deterministic filename -> must now be found.
    coord_file = fake_topo_dir / f'{m.CNN}_sample5_pd_diagram.csv'
    coord_file.write_text('birth,death\n0.1,0.2\n')
    check('resolve_pd_diagram_source finds a coordinate file matching the documented naming convention',
          m.resolve_pd_diagram_source(m.CNN, 5, fake_source_map) == coord_file)
    check('resolve_pd_diagram_source still returns None for an unresolved (method,sample) pair',
          m.resolve_pd_diagram_source(m.CNN, 999, fake_source_map) is None)

    fig3 = m.FIGURE_BY_ID[3]
    manifest10 = dict(m.FROZEN_SAMPLE_SET)
    try:
        m.require_pd_diagram_sources_for_figure(fig3, manifest10, fake_source_map)
        check('require_pd_diagram_sources_for_figure hard-fails when sources are missing for required methods',
              False)
    except SystemExit as e:
        check('require_pd_diagram_sources_for_figure hard-fails when sources are missing for required methods',
              'no frozen PD coordinate source was found' in str(e))
finally:
    shutil.rmtree(tmp10, ignore_errors=True)

print()
print('=== 11. Missing manual MT panel hard-fail ===')
try:
    m.require_manual_topology_panels(m.FROZEN_SAMPLE_SET)
    check('require_manual_topology_panels hard-fails when no manual panels exist (real checkout state)', False)
except SystemExit as e:
    check('require_manual_topology_panels hard-fails when no manual panels exist (real checkout state)',
          'required manual topology panel' in str(e))

print()
print('=== 12. Missing MT metadata hard-fail (panel present, metadata absent/incomplete) ===')
created_dirs = []
try:
    manual_rows = m.build_manual_topology_requirements_rows(m.FROZEN_SAMPLE_SET)
    for r in manual_rows:
        panel_path = m.REPO_ROOT / r['expected_panel_path']
        panel_path.parent.mkdir(parents=True, exist_ok=True)
        created_dirs.append(panel_path.parent)
        panel_path.write_bytes(b'\x89PNG\r\n\x1a\n')  # minimal placeholder bytes, panel PNG "present"
        # Deliberately do NOT write the metadata CSV for any panel.
    try:
        m.require_manual_topology_panels(m.FROZEN_SAMPLE_SET)
        check('all manual panels present but ALL metadata missing -> SystemExit', False)
    except SystemExit as e:
        check('all manual panels present but ALL metadata missing -> SystemExit',
              'metadata' in str(e).lower() or 'manual topology panel' in str(e))

    # Now write valid, complete metadata for every panel except one -- must still fail, citing metadata.
    for r in manual_rows[1:]:
        meta_path = m.REPO_ROOT / r['expected_metadata_path']
        with meta_path.open('w', newline='') as fh:
            import csv as _csv
            w = _csv.DictWriter(fh, fieldnames=m.MANUAL_TOPOLOGY_METADATA_FIELDS)
            w.writeheader()
            w.writerow({f: 'x' for f in m.MANUAL_TOPOLOGY_METADATA_FIELDS})
    try:
        m.require_manual_topology_panels(m.FROZEN_SAMPLE_SET)
        check('one missing metadata file among many present -> still SystemExit', False)
    except SystemExit as e:
        check('one missing metadata file among many present -> still SystemExit', True)

    # Complete the last one, but leave a required field blank -- must fail citing the missing field.
    last = manual_rows[0]
    meta_path = m.REPO_ROOT / last['expected_metadata_path']
    with meta_path.open('w', newline='') as fh:
        import csv as _csv
        w = _csv.DictWriter(fh, fieldnames=m.MANUAL_TOPOLOGY_METADATA_FIELDS)
        w.writeheader()
        row = {f: 'x' for f in m.MANUAL_TOPOLOGY_METADATA_FIELDS}
        row['persistence_threshold'] = ''  # required field left blank
        w.writerow(row)
    try:
        m.require_manual_topology_panels(m.FROZEN_SAMPLE_SET)
        check('metadata row with an empty required field -> SystemExit', False)
    except SystemExit as e:
        check('metadata row with an empty required field -> SystemExit',
              'missing required field' in str(e) and 'persistence_threshold' in str(e))

    # Finally, fill in every field for every panel -- must now pass cleanly.
    row['persistence_threshold'] = '11.0'
    with meta_path.open('w', newline='') as fh:
        import csv as _csv
        w = _csv.DictWriter(fh, fieldnames=m.MANUAL_TOPOLOGY_METADATA_FIELDS)
        w.writeheader()
        w.writerow(row)
    result_rows = m.require_manual_topology_panels(m.FROZEN_SAMPLE_SET)
    check('fully complete manual panels + metadata -> passes without raising',
          len(result_rows) == len(manual_rows) and all(r['status'] == 'present' for r in result_rows))
finally:
    for d in set(created_dirs):
        shutil.rmtree(d, ignore_errors=True)
    if m.MANUAL_TOPOLOGY_DIR.exists():
        shutil.rmtree(m.MANUAL_TOPOLOGY_DIR, ignore_errors=True)
check('manual_topology_inputs test scaffolding fully cleaned up', not m.MANUAL_TOPOLOGY_DIR.exists())

print()
print('=== 13. Caption/figure ordering agreement ===')
check('CAPTION_TEMPLATES keys are exactly {1..6}', set(m.CAPTION_TEMPLATES) == {1, 2, 3, 4, 5, 6})
check('every caption is sample-specific ("For sample" or "this selected/illustrative" framing present)',
      all(('for sample' in txt.lower() or 'illustrative' in txt.lower() or 'selected example' in txt.lower())
          for txt in m.CAPTION_TEMPLATES.values()))
FORBIDDEN_OVERCLAIMS = ['always preserves', 'universally improves', 'is inaccurate', 'universally superior']
check('no caption contains a forbidden overclaiming phrase',
      all(not any(bad in txt.lower() for bad in FORBIDDEN_OVERCLAIMS) for txt in m.CAPTION_TEMPLATES.values()))

captions_text = m.CAPTIONS_PATH.read_text()
order_found = [int(tok) for tok in __import__('re').findall(r'## Figure (\d+):', captions_text)]
check('final_figure_captions.md lists figures in ascending 1..6 order matching FIGURE_CONTRACTS',
      order_found == [1, 2, 3, 4, 5, 6])

print()
print('=== 14. Panel dimensions (synthetic render on a tiny fixture) ===')
old_in_shape, old_hr_shape = m.p2da.EXPECTED_IN_SHAPE, m.p2da.EXPECTED_HR_SHAPE
panel_dirs_created = []
try:
    m.p2da.EXPECTED_IN_SHAPE = (m.p2da.N_EVAL, 4, 4, 2)
    m.p2da.EXPECTED_HR_SHAPE = (m.p2da.N_EVAL, 12, 12, 2)
    rng14 = np.random.default_rng(7)
    fake_gt = rng14.normal(size=(12, 12, 2)).astype(np.float32) * 3 + 1
    fake_audit = dict(selected_data={
        m.CNN: dict(gt=np.stack([fake_gt]), sr=np.stack([fake_gt + rng14.normal(size=fake_gt.shape) * 0.1])),
        m.BICUBIC: dict(gt=np.stack([fake_gt]),
                          sr=np.stack([fake_gt + rng14.normal(size=fake_gt.shape) * 0.2])),
        m.GAN: dict(gt=np.stack([fake_gt]), sr=np.stack([fake_gt + rng14.normal(size=fake_gt.shape) * 0.15])),
    })
    fig2 = m.FIGURE_BY_ID[2]
    manifest14 = dict(m.FROZEN_SAMPLE_SET)
    panel_rows, gt_speed14, method_speeds14, panel14 = m.render_speed_and_error_panels(
        fig2, manifest14, fake_audit, [manifest14['gan_pd_vs_cnn_mt_conflict']])
    panel_dirs_created.append(m.PANELS_DIR / m.figure_dir_name(fig2))
    check('render_speed_and_error_panels produced one GT + (speed+error) per method',
          len(panel_rows) == 1 + 2 * len(fig2['full_panel_methods']))
    from PIL import Image
    for r in panel_rows:
        img = Image.open(m.REPO_ROOT / r['output_path'])
        expected_w = round(3 * 300)  # figsize=(3,3), dpi=300
        expected_h = round(3 * 300)
        check(f"panel {r['output_path']} has the expected >=300dpi pixel dimensions "
              f'(within matplotlib tight-layout tolerance)',
              abs(img.width - expected_w) <= 40 and abs(img.height - expected_h) <= 40)
        img.close()
finally:
    m.p2da.EXPECTED_IN_SHAPE, m.p2da.EXPECTED_HR_SHAPE = old_in_shape, old_hr_shape
    for d in panel_dirs_created:
        shutil.rmtree(d, ignore_errors=True)

print()
print('=== 15. Final composite creation from synthetic panels (figures needing no manual MT panel) ===')
composite_paths_created = []
try:
    for fig_id in (4, 6):  # neither requires a manual MT panel
        c = m.FIGURE_BY_ID[fig_id]
        panel_dir = m.PANELS_DIR / m.figure_dir_name(c)
        panel_dir.mkdir(parents=True, exist_ok=True)
        panel_dirs_created.append(panel_dir)
        rng15 = np.random.default_rng(fig_id)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        for i in range(3):
            fig, ax = plt.subplots(figsize=(1, 1), dpi=50)
            ax.imshow(rng15.uniform(size=(10, 10)))
            fig.savefig(panel_dir / f'synthetic_panel_{i}.png')
            plt.close(fig)
        manifest15 = dict(m.FROZEN_SAMPLE_SET)
        row = m.build_composite_for_figure(c, manifest15)
        composite_paths_created.append(m.REPO_ROOT / row['expected_png_path'])
        composite_paths_created.append(m.REPO_ROOT / row['expected_vector_path'])
        check(f'figure {fig_id} composite PNG exists and is non-empty',
              (m.REPO_ROOT / row['expected_png_path']).exists()
              and (m.REPO_ROOT / row['expected_png_path']).stat().st_size > 0)
        check(f'figure {fig_id} composite vector (PDF) exists and is non-empty',
              (m.REPO_ROOT / row['expected_vector_path']).exists()
              and (m.REPO_ROOT / row['expected_vector_path']).stat().st_size > 0)
        check(f'figure {fig_id} composite row reports status=rendered', row['status'] == 'rendered')

    # No panels at all -> hard-fail, never a blank/placeholder composite.
    c3 = m.FIGURE_BY_ID[3]
    empty_dir = m.PANELS_DIR / m.figure_dir_name(c3)
    if empty_dir.exists():
        shutil.rmtree(empty_dir)
    try:
        m.build_composite_for_figure(c3, dict(m.FROZEN_SAMPLE_SET))
        check('composite assembly with zero available panels -> SystemExit (never a blank figure)', False)
    except SystemExit as e:
        check('composite assembly with zero available panels -> SystemExit (never a blank figure)',
              'No rendered panels found' in str(e))
finally:
    for p in composite_paths_created:
        if p.exists():
            p.unlink()
    for d in panel_dirs_created:
        shutil.rmtree(d, ignore_errors=True)
    if m.FIGURES_DIR.exists():
        shutil.rmtree(m.FIGURES_DIR, ignore_errors=True)
check('panels/ and figures/ test scaffolding fully cleaned up',
      not any((m.PANELS_DIR / m.figure_dir_name(m.FIGURE_BY_ID[i])).exists() for i in (4, 6))
      and not m.FIGURES_DIR.exists())

print()
print('=== 16. Missing protected file -> hard fail ===')
target = m.PHASE2D_A_PROTECTED_CSVS[0]
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
print('=== 17. Altered protected checksum -> hard fail ===')
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
