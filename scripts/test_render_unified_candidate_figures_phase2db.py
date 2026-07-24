#!/usr/bin/env python3
"""Synthetic/contract tests for scripts/render_unified_candidate_figures_phase2db.py.

Exercises the real shipped functions directly. Safe and deterministic in
BOTH:

  1. a lightweight development checkout, where the Phase-2D-A preview_audit
     package, preview images, visual-review record, and render log are
     genuinely absent; and
  2. the authoritative Spark checkout, where the completed Phase-2D-A
     package already exists.

No test ever overwrites, deletes, or renames an existing authoritative
Phase-1 through Phase-2D-A file without restoring it byte-identically in a
`finally` block (checked, not assumed). The golden-path fixture in Section 6
uses real authoritative files directly when present and only manufactures a
synthetic stand-in for a path that is genuinely absent -- it never
overwrites existing content merely to make the test easier. A top-level
snapshot/restore wrapper (guarding against BaseException, including
KeyboardInterrupt) provides a second, independent safety net around the
entire suite.

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


# =============================================================================
# Generic snapshot/restore helpers used throughout this suite and by the
# top-level safety wrapper at the bottom of the file.
# =============================================================================

def snapshot_tree(root):
    """Byte snapshot of every file under `root` (root may itself be
    git-tracked pre-existing content) -- used to precisely undo a test
    run's writes without destroying anything that existed beforehand."""
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


def snapshot_paths(paths):
    """Exact per-path snapshot: None if the path did not exist, else its
    bytes. Never assumes a common parent directory structure."""
    return {p: (p.read_bytes() if p.exists() else None) for p in paths}


def restore_paths(snapshot):
    """Restores each path exactly: deletes it if it did not exist before,
    (re)writes it if it existed with possibly-different bytes, and prunes
    every directory the restoration touched -- and that directory's
    ancestors up to (not including) REPO_ROOT -- bottom-up. Each rmdir is a
    safe no-op (OSError, caught) whenever the directory still holds other
    content or was never created by this restoration."""
    touched_dirs = set()
    for p, data in snapshot.items():
        touched_dirs.add(p.parent)
        if data is None:
            if p.exists():
                p.unlink()
        else:
            p.parent.mkdir(parents=True, exist_ok=True)
            if not p.exists() or p.read_bytes() != data:
                p.write_bytes(data)
    all_dirs = set()
    for d in touched_dirs:
        cur = d
        while cur != m.REPO_ROOT and m.REPO_ROOT in cur.parents:
            all_dirs.add(cur)
            cur = cur.parent
    for d in sorted(all_dirs, key=lambda x: -len(str(x))):
        try:
            d.rmdir()
        except OSError:
            pass


def paths_match_snapshot(paths, snapshot):
    return snapshot_paths(paths) == snapshot


# =============================================================================
# Top-level safety wrapper: snapshots everything the whole suite could ever
# touch BEFORE running a single test, and restores it in a `finally` that
# also fires on KeyboardInterrupt/SystemExit/BaseException. This is an
# independent second safety net on top of every individual section's own
# try/finally cleanup.
# =============================================================================

PHASE2D_A_FIXTURE_PATHS = None  # populated once m is imported; see below


def _init_fixture_paths():
    global PHASE2D_A_FIXTURE_PATHS
    PHASE2D_A_FIXTURE_PATHS = (
        [m.p2da.DOC_PATH, m.VISUAL_REVIEW_DOC_PATH,
         m.REPO_ROOT / 'logs' / 'unified_candidate_analysis_phase2d_render.log']
        + [m.p2da.PREVIEW_AUDIT_DIR / name for name in m.PHASE2D_A_PREVIEW_AUDIT_CSV_NAMES]
        + list(m.PHASE2D_A_PREVIEW_PNGS)
    )


_init_fixture_paths()

TOP_LEVEL_TARGETS = [m.OUT_DIR, ]  # tree targets handled via snapshot_tree
TOP_LEVEL_FILE_TARGETS = [m.DOC_PATH, m.LOG_PATH]  # single-file targets


def _top_level_snapshot():
    return dict(
        out_dir=snapshot_tree(m.OUT_DIR),
        doc=(m.DOC_PATH.read_bytes() if m.DOC_PATH.exists() else None),
        log=(m.LOG_PATH.read_bytes() if m.LOG_PATH.exists() else None),
        phase2d_a=snapshot_paths(PHASE2D_A_FIXTURE_PATHS),
    )


def _top_level_restore(snap):
    restore_tree(m.OUT_DIR, snap['out_dir'])
    if snap['doc'] is None:
        if m.DOC_PATH.exists():
            m.DOC_PATH.unlink()
    else:
        m.DOC_PATH.write_bytes(snap['doc'])
    if snap['log'] is None:
        if m.LOG_PATH.exists():
            m.LOG_PATH.unlink()
    else:
        m.LOG_PATH.write_bytes(snap['log'])
    restore_paths(snap['phase2d_a'])


# =============================================================================
# Golden-path fixture (Section 6 helper): non-destructive by construction.
# For every path Phase-2D-B's authoritative-input gate requires, if it
# ALREADY exists (e.g. the authoritative Spark package), it is left
# completely untouched and used directly; a synthetic stand-in is written
# ONLY for a path that is genuinely absent.
# =============================================================================

def make_completed_phase2d_a_fixture():
    """Returns (cleanup, pre_snapshot). cleanup() restores every path in
    PHASE2D_A_FIXTURE_PATHS to its exact prior state (including
    non-existence); pre_snapshot is exposed so callers can independently
    verify byte-identical restoration afterward."""
    pre_snapshot = snapshot_paths(PHASE2D_A_FIXTURE_PATHS)

    doc_bytes = pre_snapshot[m.p2da.DOC_PATH]
    if doc_bytes is None:
        raise SystemExit(
            f'[test-fixture-error] {m.p2da.DOC_PATH} does not exist at all; Phase-2D-A must at least have run '
            f'--selection-only for this test to proceed. This is not one of the two supported test states.'
        )
    if 'Phase 2D-A complete.' not in doc_bytes.decode():
        # Lightweight checkout: --selection-only banner only. Synthesize the
        # completed banner for this test run; restored byte-identically after.
        completed_text = doc_bytes.decode().replace(
            'Phase 2D-A selection stage complete.\nRaw-array audit and preview rendering pending authoritative '
            'Spark run.',
            'Phase 2D-A complete.\nSelection, raw audit, and review-preview generation all passed.',
        )
        m.p2da.DOC_PATH.write_text(completed_text)
    # else: already reports the completed state (authoritative Spark) -- leave untouched, use directly.

    if pre_snapshot[m.VISUAL_REVIEW_DOC_PATH] is None:
        lines = ['# Phase 2D-A Visual Review (synthetic test fixture)', '']
        for aid, si in m.FROZEN_SAMPLE_SET.items():
            lines.append(f'{aid} sample_idx={si}: ACCEPTED')
        lines.append('')
        lines.append('No alternate was activated for any archetype.')
        m.VISUAL_REVIEW_DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
        m.VISUAL_REVIEW_DOC_PATH.write_text('\n'.join(lines) + '\n')
    # else: a real, human-authored visual-review record already exists -- leave untouched.

    render_log_path = m.REPO_ROOT / 'logs' / 'unified_candidate_analysis_phase2d_render.log'
    if pre_snapshot[render_log_path] is None:
        render_log_path.parent.mkdir(parents=True, exist_ok=True)
        render_log_path.write_text('synthetic render log for testing\n')

    for name in m.PHASE2D_A_PREVIEW_AUDIT_CSV_NAMES:
        p = m.p2da.PREVIEW_AUDIT_DIR / name
        if pre_snapshot[p] is None:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text('col1,col2\n1,2\n')

    for p in m.PHASE2D_A_PREVIEW_PNGS:
        if pre_snapshot[p] is None:
            p.parent.mkdir(parents=True, exist_ok=True)
            Image.new('RGB', (10, 10), color='white').save(p)

    def cleanup():
        restore_paths(pre_snapshot)

    return cleanup, pre_snapshot


# =============================================================================
# Synthetic TTK PD VTU builder (Section 8): a minimal but schema-correct
# ASCII-inline VTU matching the REAL TTK ttkPersistenceDiagram output --
# CellData = PairIdentifier, PairType, Persistence, Birth, IsFinite;
# PointData = ttkVertexScalarField; Cells = connectivity/offsets/types.
# Point (mesh) geometry is deliberately set to large, obviously-bogus values
# far from any birth/death value, so any test that accidentally used raw
# point coordinates as birth/death would fail loudly.
# =============================================================================

def _write_synthetic_pd_vtu(path, pair_id, pair_type, persistence, birth, is_finite, omit_arrays=()):
    n_cells = len(pair_id)
    n_points = n_cells * 2
    vertex_ids = list(range(2000, 2000 + n_points))
    connectivity = list(range(n_points))
    offsets = [2 * (i + 1) for i in range(n_cells)]
    types = [3] * n_cells
    points = []
    for i in range(n_points):
        points += [8888.0 + i, -8888.0 - i, 0.0]

    def _fmt(vals):
        return ' '.join(str(v) for v in vals)

    cell_arrays = [
        ('PairIdentifier', 'Int32', pair_id), ('PairType', 'Int32', pair_type),
        ('Persistence', 'Float32', persistence), ('Birth', 'Float32', birth),
        ('IsFinite', 'UInt8', is_finite),
    ]
    cell_data_xml = '\n'.join(
        f'        <DataArray type="{dtype}" Name="{name}" format="ascii">{_fmt(vals)}</DataArray>'
        for name, dtype, vals in cell_arrays if name not in omit_arrays)
    point_data_xml = (
        '' if 'ttkVertexScalarField' in omit_arrays else
        f'        <DataArray type="Int32" Name="ttkVertexScalarField" format="ascii">{_fmt(vertex_ids)}</DataArray>')
    connectivity_xml = (
        '' if 'connectivity' in omit_arrays else
        f'        <DataArray type="Int64" Name="connectivity" format="ascii">{_fmt(connectivity)}</DataArray>')
    xml = f'''<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian" header_type="UInt32">
  <UnstructuredGrid>
    <Piece NumberOfPoints="{n_points}" NumberOfCells="{n_cells}">
      <PointData>
{point_data_xml}
      </PointData>
      <CellData>
{cell_data_xml}
      </CellData>
      <Points>
        <DataArray type="Float32" Name="Points" NumberOfComponents="3" format="ascii">{_fmt(points)}</DataArray>
      </Points>
      <Cells>
{connectivity_xml}
        <DataArray type="Int64" Name="offsets" format="ascii">{_fmt(offsets)}</DataArray>
        <DataArray type="UInt8" Name="types" format="ascii">{_fmt(types)}</DataArray>
      </Cells>
    </Piece>
  </UnstructuredGrid>
</VTKFile>
'''
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(xml)


def _pd_vtu_path(tmp_root, alias, role, sample_idx, suffix='v0'):
    return tmp_root / f'{alias}_topology' / 'pd' / role / f'{alias}_{role}_s{sample_idx}_{suffix}_pd_port_0.vtu'


# =============================================================================
# Run the whole suite inside a top-level try/finally so that even an
# unexpected mid-suite exception (or Ctrl-C) cannot leave any pre-existing
# tracked or untracked Phase-2D-B/Phase-2D-A artifact modified.
# =============================================================================

_top_snapshot = _top_level_snapshot()

try:
    print('=== 1. Protected-file counts (86 prior + 32 Phase-2D-A = 118) ===')
    check('Phase-2D-A protected files == 32', len(m.PHASE2D_A_PROTECTED_FILES) == 32)
    check('  selection CSVs == 13', len(m.PHASE2D_A_SELECTION_CSVS) == 13)
    check('  preview_audit CSVs == 6', len(m.PHASE2D_A_PREVIEW_AUDIT_CSVS) == 6)
    check('  preview PNGs == 7', len(m.PHASE2D_A_PREVIEW_PNGS) == 7)
    check('  docs == 2', len(m.PHASE2D_A_DOCS) == 2)
    check('  scripts == 2', len(m.PHASE2D_A_SCRIPTS) == 2)
    check('  logs == 2', len(m.PHASE2D_A_LOGS) == 2)
    check('All protected files == 118', len(m.ALL_PROTECTED_FILES) == 118)
    check('FROZEN_SAMPLE_SET covers exactly the 6 archetypes',
          set(m.FROZEN_SAMPLE_SET) == set(m.p2da.ARCHETYPE_PRIORITY))

    print()
    print('=== 2. Frozen-sample enforcement (real manifest must match FROZEN_SAMPLE_SET) ===')
    manifest = m.read_and_validate_selection_manifest()
    check('read_and_validate_selection_manifest() returns exactly FROZEN_SAMPLE_SET',
          manifest == m.FROZEN_SAMPLE_SET)
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
    check('read_and_validate_selection_manifest() never references archetype_alternates.csv outside its '
          'docstring (the docstring itself documents that it is deliberately never read)',
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
        check(f"figure {c['figure_id']} required_methods follows the canonical GT,Bicubic,CNN,GAN,Candidate C,"
              f'F3,F2,UV+E2 order', c['required_methods'] == canonical_subsequence)
    check('figures 1 and 6 place CNN before GAN',
          m.FIGURE_BY_ID[1]['required_methods'].index(m.CNN) < m.FIGURE_BY_ID[1]['required_methods'].index(m.GAN)
          and m.FIGURE_BY_ID[6]['required_methods'].index(m.CNN) < m.FIGURE_BY_ID[6]['required_methods'].index(m.GAN))
    check('figure 3 restricts F2 to full_panel_methods exclusion (compact contextual reference only)',
          m.F2 not in m.FIGURE_BY_ID[3]['full_panel_methods'] and m.F2 in m.FIGURE_BY_ID[3]['required_methods'])
    check('figure 3 method_roles records F2 as compact_contextual_reference',
          m.FIGURE_BY_ID[3]['method_roles'].get(m.F2) == 'compact_contextual_reference')

    print()
    print('=== 6. Full --plan-only pipeline succeeds end-to-end (real Phase-2D-A package used directly '
          'when present; synthetic stand-ins only for genuinely-absent paths) ===')
    print(f'    Environment: IS_LIGHTWEIGHT_CHECKOUT={m.IS_LIGHTWEIGHT_CHECKOUT}')
    out_dir_snapshot = snapshot_tree(m.OUT_DIR)
    doc_2db_snapshot = m.DOC_PATH.read_bytes() if m.DOC_PATH.exists() else None
    cleanup_fixture, fixture_pre_snapshot = make_completed_phase2d_a_fixture()
    try:
        plan_result = m.cmd_plan_only()
        check('cmd_plan_only() completed without raising', True)

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
              len(plan_result['repro_rows']) == 168
              and all(r['status'] == 'PASS' for r in plan_result['repro_rows']))

        pd_rows = plan_result['pd_discovery_rows']
        check('pd_source_discovery.csv has rows for figures 1, 2, 3 only (the figures needing PD diagram panels)',
              {r['figure_id'] for r in pd_rows} == {1, 2, 3})
        check('GT is present among the searched methods for figure 1 (never assumed found without a concrete '
              'source)', any(r['method_id'] == 'GT' and r['figure_id'] == 1 for r in pd_rows))
        check('bicubic is present among the searched methods for figure 2 (real search, not a hardcoded '
              'exclusion)', any(r['method_id'] == m.BICUBIC and r['figure_id'] == 2 for r in pd_rows))

        allowed_statuses = ({m.STATUS_PENDING} if m.IS_LIGHTWEIGHT_CHECKOUT
                              else {m.STATUS_AVAILABLE, m.STATUS_PENDING, m.STATUS_UNAVAILABLE})
        check('every PD discovery row status is within the environment-appropriate vocabulary',
              all(r['usable_status'] in allowed_statuses for r in pd_rows))
        if m.IS_LIGHTWEIGHT_CHECKOUT:
            check('every PD discovery row is pending_authoritative_spark_source_discovery in a lightweight '
                  'checkout (never unavailable_after_authoritative_spark_audit here)',
                  all(r['usable_status'] == m.STATUS_PENDING for r in pd_rows))
        else:
            check('available_validated rows have a concrete candidate_path',
                  all(r['candidate_path'] for r in pd_rows if r['usable_status'] == m.STATUS_AVAILABLE))
            check('GT is never marked available_validated without a concrete candidate_path',
                  all(r['candidate_path'] for r in pd_rows
                      if r['method_id'] == 'GT' and r['usable_status'] == m.STATUS_AVAILABLE))
            check('unavailable_after_authoritative_spark_audit is legal here only because '
                  'IS_LIGHTWEIGHT_CHECKOUT is False', not m.IS_LIGHTWEIGHT_CHECKOUT
                  or not any(r['usable_status'] == m.STATUS_UNAVAILABLE for r in pd_rows))

        composite_path = m.PLAN_DIR / 'final_composite_manifest.csv'
        check('final_composite_manifest.csv was written', composite_path.exists())
        composite_rows = m.p2da.read_csv_dicts(composite_path)
        pd_group_rows = [r for r in composite_rows if int(r['figure_id']) in (1, 2, 3)
                           and r['panel_group'] in ('pd_coordinate', 'pd_scalar_fallback')]
        check('every PD-related composite-manifest row for figures 1/2/3 has a status consistent with its '
              'discovery verdict (pending, visible, or scalar_fallback)',
              all(r['final_visible_status'] in ('pending', 'visible', 'scalar_fallback') for r in pd_group_rows))

        zoom_val_path = m.VALIDATION_DIR / 'zoom_selection_validation.csv'
        check('zoom_selection_validation.csv was written in the not_yet_computed state (requires real GT/error '
              'arrays, out of scope for --plan-only in either environment)',
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
    check('Phase-2D-A report (docs/unified_candidate_analysis_phase2d.md) restored to its exact pre-test '
          'content (never asserted to be non-complete -- the Spark report is legitimately already complete)',
          paths_match_snapshot([m.p2da.DOC_PATH], {m.p2da.DOC_PATH: fixture_pre_snapshot[m.p2da.DOC_PATH]}))
    check('the visual-review record was restored to its exact pre-test state (present-with-identical-bytes, '
          'or absent, exactly as before)',
          paths_match_snapshot([m.VISUAL_REVIEW_DOC_PATH],
                                 {m.VISUAL_REVIEW_DOC_PATH: fixture_pre_snapshot[m.VISUAL_REVIEW_DOC_PATH]}))
    check('the entire preview_audit path tree is byte-identical to its pre-test snapshot',
          paths_match_snapshot(
              [m.p2da.PREVIEW_AUDIT_DIR / n for n in m.PHASE2D_A_PREVIEW_AUDIT_CSV_NAMES],
              {p: fixture_pre_snapshot[p] for p in [m.p2da.PREVIEW_AUDIT_DIR / n
                                                       for n in m.PHASE2D_A_PREVIEW_AUDIT_CSV_NAMES]}))
    check('every Phase-2D-A preview PNG is byte-identical to its pre-test snapshot',
          paths_match_snapshot(m.PHASE2D_A_PREVIEW_PNGS,
                                 {p: fixture_pre_snapshot[p] for p in m.PHASE2D_A_PREVIEW_PNGS}))
    check('the render log is byte-identical to its pre-test snapshot',
          paths_match_snapshot([m.REPO_ROOT / 'logs' / 'unified_candidate_analysis_phase2d_render.log'],
                                 {m.REPO_ROOT / 'logs' / 'unified_candidate_analysis_phase2d_render.log':
                                      fixture_pre_snapshot[m.REPO_ROOT / 'logs' /
                                                             'unified_candidate_analysis_phase2d_render.log']}))
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
    check('shared speed scaling expands to include an out-of-GT-range SR value (Phase-2D-A helper reused '
          'correctly)', panel['speed_vmax'] >= 50.0 - 1e-9)
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
        fd_rows = [dict(figure_id=3, archetype_id=fig3['archetype_id'],
                           sample_idx=manifest10[fig3['archetype_id']], method_id=mid, display_label='',
                           pd_distance='', mt_distance='', psnruv='', speed_mae='', grad_mae='', wpd_mae='',
                           source_field_path='', source_gt_path='', source_topology_path='', zoom_y0='',
                           zoom_y1='', zoom_x0='', zoom_x1='')
                    for mid in ['GT'] + fig3['required_methods']]
        m.write_csv(m.FIGURE_DATA_DIR / m.FIGURE_DATA_FILENAMES[3], m.FIGURE_DATA_FIELDS, fd_rows)
        pm_rows = [dict(figure_id=3, archetype_id=fig3['archetype_id'],
                           sample_idx=manifest10[fig3['archetype_id']], panel_type=m.ZOOM_CROP, method_id='',
                           display_label='', method_role='', output_path='ttk_runs_fixed/x.png',
                           requires_manual_topology_input=False, requires_pd_coordinate_source=False,
                           pd_coordinate_source_found='', status='planned_not_rendered')]
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
    print('=== 11. Authoritative PD VTU discovery, parsing, and validation (available / pending / unavailable '
          'routing; exact role/alias/sample mapping; hard-fail-on-malformed) ===')
    tmp11 = m.OUT_DIR / '_test11_scratch'
    old_topology_tree_root = m._topology_tree_root
    old_cnn_gan_bicubic_roots = dict(m.CNN_GAN_BICUBIC_SEARCH_ROOTS)
    old_lightweight_flag = m.IS_LIGHTWEIGHT_CHECKOUT
    ALIAS_C = m.METHOD_ARTIFACT_ALIASES[m.CANDIDATE_C]
    ALIAS_F3 = m.METHOD_ARTIFACT_ALIASES[m.F3]
    try:
        tmp11.mkdir(parents=True, exist_ok=True)
        m._topology_tree_root = lambda alias: tmp11 / f'{alias}_topology'
        m.CNN_GAN_BICUBIC_SEARCH_ROOTS = {
            m.CNN: [tmp11 / 'cnn_root'], m.GAN: [tmp11 / 'gan_root'], m.BICUBIC: [tmp11 / 'bicubic_root'],
        }

        check('Section 2 method-to-artifact aliases match the task specification exactly',
              m.METHOD_ARTIFACT_ALIASES == {
                  m.CANDIDATE_C: 'candidateC_expanded2688', m.F3: 'candidateF_grad_crit_expanded2688',
                  m.F2: 'candidateF_grad_levelset_E2_low_expanded2688',
                  m.UV_E2: 'candidateUV_plus_E2_tf_lowlambda_expanded2688',
              })
        check('CNN/GAN/bicubic are never given an invented <alias>_topology directory (no entry in '
              'METHOD_ARTIFACT_ALIASES)',
              not ({m.CNN, m.GAN, m.BICUBIC} & set(m.METHOD_ARTIFACT_ALIASES)))

        # --- 11a: birth/death are derived from Birth/Persistence CellData, never
        # from raw VTU point geometry; infinite pairs, pair_type==-1, and
        # zero/negative-persistence pairs are all excluded from the mask. ---
        mixed_path = _pd_vtu_path(tmp11, ALIAS_C, 'SR', 601)
        _write_synthetic_pd_vtu(
            mixed_path,
            pair_id=[0, 1, 2, 3, 4, 5], pair_type=[0, 1, 0, -1, 0, 0],
            persistence=[2.0, 0.5, 5.0, 5.0, 0.0, -1.0], birth=[1.0, 4.0, 0.0, 0.0, 0.0, 0.0],
            is_finite=[1, 1, 0, 1, 1, 1])
        result_11a = m.parse_and_validate_pd_vtu(mixed_path, 601, 'SR', ALIAS_C)
        check('exactly the two finite/positive-persistence/non-infinite pairs survive masking',
              result_11a['pair_count'] == 2)
        check('birth/death are derived from Birth+Persistence CellData (never raw point geometry, which was '
              'deliberately set to ~8888/-8888 in this fixture)',
              sorted(result_11a['birth'].tolist()) == [1.0, 4.0]
              and sorted(result_11a['death'].tolist()) == [3.0, 4.5]
              and not any(abs(v) > 100 for v in list(result_11a['birth']) + list(result_11a['death'])))

        # --- 11b: missing required array -> hard-fail (present-but-malformed, never
        # silently reclassified as pending/unavailable). ---
        missing_arr_path = _pd_vtu_path(tmp11, ALIAS_C, 'SR', 602)
        _write_synthetic_pd_vtu(missing_arr_path, pair_id=[0], pair_type=[0], persistence=[1.0], birth=[0.0],
                                  is_finite=[1], omit_arrays=('IsFinite',))
        try:
            m.parse_and_validate_pd_vtu(missing_arr_path, 602, 'SR', ALIAS_C)
            check('a PD VTU missing a required CellData array hard-fails', False)
        except SystemExit:
            check('a PD VTU missing a required CellData array hard-fails', True)

        # --- 11c: non-finite value in a mask-passing pair -> hard-fail. ---
        nonfinite_path = _pd_vtu_path(tmp11, ALIAS_C, 'SR', 603)
        _write_synthetic_pd_vtu(nonfinite_path, pair_id=[0], pair_type=[0], persistence=[1.0], birth=['nan'],
                                  is_finite=[1])
        try:
            m.parse_and_validate_pd_vtu(nonfinite_path, 603, 'SR', ALIAS_C)
            check('a non-finite birth/death value in a mask-passing pair hard-fails', False)
        except SystemExit:
            check('a non-finite birth/death value in a mask-passing pair hard-fails', True)

        # --- 11d: exact sample_idx mapping -- a correctly-formed file for sample
        # 601 is REJECTED when the caller expects a different sample. ---
        try:
            m.parse_and_validate_pd_vtu(mixed_path, 999999, 'SR', ALIAS_C)
            check('a PD VTU whose filename sample_idx does not exactly match the expected sample_idx '
                  'hard-fails (never a partial/prefix match)', False)
        except SystemExit:
            check('a PD VTU whose filename sample_idx does not exactly match the expected sample_idx '
                  'hard-fails (never a partial/prefix match)', True)
        wrong_prefix_dir = tmp11 / f'{ALIAS_C}_topology' / 'pd' / 'SR'
        (wrong_prefix_dir / f'{ALIAS_C}_SR_s6010_extra_pd_port_0.vtu').write_bytes(mixed_path.read_bytes())
        found_601 = m.find_exact_pd_vtu_candidates(tmp11 / f'{ALIAS_C}_topology', 'SR', ALIAS_C, 601)
        check('exact sample-index search for s601 never matches a file for s6010 (boundary-safe prefix, not '
              'a raw substring match)', {p.name for p in found_601} == {mixed_path.name})

        # --- 11e: MT paths are never mapped as a PD source, even if nested inside
        # an otherwise-plausible pd/<role>/ tree; _mt_port_*/_pd_port_1 filenames
        # are rejected outright. ---
        check('_is_valid_pd_vtu_path rejects _mt_port_0.vtu', not m._is_valid_pd_vtu_path(
            tmp11 / 'x_mt_port_0.vtu'))
        check('_is_valid_pd_vtu_path rejects _mt_port_1.vtu', not m._is_valid_pd_vtu_path(
            tmp11 / 'x_mt_port_1.vtu'))
        check('_is_valid_pd_vtu_path rejects _pd_port_1.vtu', not m._is_valid_pd_vtu_path(
            tmp11 / 'x_pd_port_1.vtu'))
        check('_is_valid_pd_vtu_path rejects any path with an mt/ directory component',
              not m._is_valid_pd_vtu_path(tmp11 / f'{ALIAS_C}_topology' / 'mt' / 'x_pd_port_0.vtu'))
        check('_is_valid_pd_vtu_path accepts a real, correctly-suffixed pd_port_0.vtu path',
              m._is_valid_pd_vtu_path(mixed_path))
        stray_mt_dir = tmp11 / f'{ALIAS_C}_topology' / 'pd' / 'SR' / 'mt'
        stray_mt_dir.mkdir(parents=True, exist_ok=True)
        (stray_mt_dir / f'{ALIAS_C}_SR_s601_stray_pd_port_0.vtu').write_bytes(mixed_path.read_bytes())
        found_after_stray = m.find_exact_pd_vtu_candidates(tmp11 / f'{ALIAS_C}_topology', 'SR', ALIAS_C, 601)
        check('a file nested under an mt/ subdirectory is never returned as a PD candidate, even though it '
              'matches the filename convention (only the real, non-mt/ file is found)',
              [p.name for p in found_after_stray] == [mixed_path.name])
        shutil.rmtree(stray_mt_dir)
        (wrong_prefix_dir / f'{ALIAS_C}_SR_s6010_extra_pd_port_0.vtu').unlink()

        # --- 11f: duplicate resolved paths (e.g. the same directory reachable via
        # two overlapping search roots) collapse to ONE candidate, never a
        # duplicate row / spurious conflict. ---
        old_cnn_roots_for_dedup = m.CNN_GAN_BICUBIC_SEARCH_ROOTS
        cnn_root = tmp11 / 'cnn_root'
        cnn_sr_path = cnn_root / 'pd' / 'SR' / 'cnn_SR_s604_v0_pd_port_0.vtu'
        cnn_sr_path.parent.mkdir(parents=True, exist_ok=True)
        _write_synthetic_pd_vtu(cnn_sr_path, pair_id=[0], pair_type=[0], persistence=[1.0], birth=[0.5],
                                  is_finite=[1])
        try:
            m.CNN_GAN_BICUBIC_SEARCH_ROOTS = {m.CNN: [cnn_root, cnn_root], m.GAN: [], m.BICUBIC: []}
            verdict_dedup = m.resolve_pd_source_verdict(99, 604, m.CNN)
            check('the exact same file reachable via a duplicated search root resolves to a single '
                  'available_validated candidate (not a spurious conflict)',
                  verdict_dedup['verdict'] == m.STATUS_AVAILABLE and verdict_dedup['parsed_pair_count'] == 1)
        finally:
            m.CNN_GAN_BICUBIC_SEARCH_ROOTS = old_cnn_roots_for_dedup

        # --- 11g: role-aware GT vs SR mapping; artifact-alias mapping end to end
        # via resolve_pd_source_verdict(). ---
        gt_path_c = _pd_vtu_path(tmp11, ALIAS_C, 'GT', 605)
        sr_path_c = _pd_vtu_path(tmp11, ALIAS_C, 'SR', 605)
        _write_synthetic_pd_vtu(gt_path_c, pair_id=[0, 1], pair_type=[0, 0], persistence=[1.0, 2.0],
                                  birth=[0.0, 1.0], is_finite=[1, 1])
        _write_synthetic_pd_vtu(sr_path_c, pair_id=[0, 1], pair_type=[0, 0], persistence=[1.5, 1.0],
                                  birth=[0.1, 0.9], is_finite=[1, 1])
        gt_verdict = m.resolve_pd_source_verdict(99, 605, 'GT')
        sr_verdict = m.resolve_pd_source_verdict(99, 605, m.CANDIDATE_C)
        check('GT role resolves to the pd/GT/ file, never the pd/SR/ file',
              gt_verdict['verdict'] == m.STATUS_AVAILABLE and gt_verdict['source_role'] == 'GT'
              and gt_verdict['selected_candidate_path'] == m._rel(gt_path_c))
        check('SR (learned-method) role resolves to the pd/SR/ file, never pd/GT/, with the correct resolved '
              'artifact alias', sr_verdict['verdict'] == m.STATUS_AVAILABLE and sr_verdict['source_role'] == 'SR'
              and sr_verdict['selected_candidate_path'] == m._rel(sr_path_c)
              and sr_verdict['artifact_alias'] == ALIAS_C)
        check('GT and SR verdicts for the same sample never cross-map to each other\'s file',
              gt_verdict['selected_candidate_path'] != sr_verdict['selected_candidate_path'])

        # --- 11h: canonical GT selection -- a single exact GT copy is accepted
        # directly; two agreeing copies (different pair order, same content) also
        # resolve cleanly to the highest-priority alias; two DISAGREEING copies
        # hard-fail, listing every conflicting path. ---
        single_gt = m.resolve_canonical_gt_pd_source(605)
        check('a single exact GT copy resolves directly as canonical',
              single_gt is not None and single_gt['n_copies'] == 1 and single_gt['alias'] == ALIAS_C)

        gt_path_c2 = _pd_vtu_path(tmp11, ALIAS_C, 'GT', 606)
        gt_path_f3 = _pd_vtu_path(tmp11, ALIAS_F3, 'GT', 606)
        _write_synthetic_pd_vtu(gt_path_c2, pair_id=[0, 1], pair_type=[0, 0], persistence=[1.0, 2.0],
                                  birth=[0.0, 5.0], is_finite=[1, 1])
        _write_synthetic_pd_vtu(gt_path_f3, pair_id=[1, 0], pair_type=[0, 0], persistence=[2.0, 1.0],
                                  birth=[5.0, 0.0], is_finite=[1, 1])  # same pairs, different order
        agreeing = m.resolve_canonical_gt_pd_source(606)
        check('two exact GT copies with identical birth/death content (in a different pair order) agree and '
              'resolve to the highest-priority alias (candidate_c before f3_grad_crit)',
              agreeing is not None and agreeing['n_copies'] == 2 and agreeing['alias'] == ALIAS_C)

        gt_path_f3.write_text(gt_path_f3.read_text().replace('5.0', '9.0'))
        try:
            m.resolve_canonical_gt_pd_source(606)
            check('two exact GT copies that disagree on birth/death hard-fail, never an arbitrary pick', False)
        except SystemExit as e:
            check('two exact GT copies that disagree on birth/death hard-fail, never an arbitrary pick',
                  m._rel(gt_path_c2) in str(e) and m._rel(gt_path_f3) in str(e))

        # --- 11i: available/pending/unavailable routing in both environments, and
        # "no pending verdicts in simulated authoritative mode after complete
        # discovery" for the methods that DO have a real exact source. ---
        nothing_verdict_lightweight = m.resolve_pd_source_verdict(99, 607, m.CANDIDATE_C)
        check('a genuinely-not-found search resolves to pending_authoritative_spark_source_discovery in the '
              'real (lightweight) checkout', nothing_verdict_lightweight['verdict'] ==
              (m.STATUS_PENDING if old_lightweight_flag else nothing_verdict_lightweight['verdict']))
        try:
            m.IS_LIGHTWEIGHT_CHECKOUT = False
            nothing_verdict_spark = m.resolve_pd_source_verdict(99, 607, m.CANDIDATE_C)
            check('with IS_LIGHTWEIGHT_CHECKOUT forced False, the SAME genuinely-not-found search resolves to '
                  'unavailable_after_authoritative_spark_audit (never fabricated as pending)',
                  nothing_verdict_spark['verdict'] == m.STATUS_UNAVAILABLE)

            complete_gt = m.resolve_pd_source_verdict(99, 605, 'GT')
            complete_sr = m.resolve_pd_source_verdict(99, 605, m.CANDIDATE_C)
            check('in simulated authoritative (Spark) mode, every method with a real exact source resolves to '
                  'available_validated -- never left pending -- after a complete discovery pass',
                  complete_gt['verdict'] == m.STATUS_AVAILABLE and complete_sr['verdict'] == m.STATUS_AVAILABLE)
        finally:
            m.IS_LIGHTWEIGHT_CHECKOUT = old_lightweight_flag
        check('IS_LIGHTWEIGHT_CHECKOUT restored to its real value after the forced-False sub-tests',
              m.IS_LIGHTWEIGHT_CHECKOUT == old_lightweight_flag)
        restored_verdict = m.resolve_pd_source_verdict(99, 607, m.CANDIDATE_C)
        check('with the real (lightweight) IS_LIGHTWEIGHT_CHECKOUT restored, the SAME not-found search '
              'resolves back to pending_authoritative_spark_source_discovery',
              restored_verdict['verdict'] == m.STATUS_PENDING or not m.IS_LIGHTWEIGHT_CHECKOUT)
    finally:
        m._topology_tree_root = old_topology_tree_root
        m.CNN_GAN_BICUBIC_SEARCH_ROOTS = old_cnn_gan_bicubic_roots
        m.IS_LIGHTWEIGHT_CHECKOUT = old_lightweight_flag
        shutil.rmtree(tmp11, ignore_errors=True)

    print()
    print('=== 12. Mixed real-coordinate / scalar-fallback PD panel rendering (Section 7) ===')
    tmp12_root = m.OUT_DIR / '_test_pd_coords'
    panel_dirs_created = []
    try:
        tmp12_root.mkdir(parents=True, exist_ok=True)
        fig2 = m.FIGURE_BY_ID[2]
        manifest12 = dict(m.FROZEN_SAMPLE_SET)
        si12 = manifest12[fig2['archetype_id']]
        per_sample12 = {mid: {si12: dict(pd_distance=3.14, mt_distance=1.0)} for mid in fig2['full_panel_methods']}

        def _fake_verdict(mid, birth, death):
            return dict(figure_id=2, sample_idx=si12, method_id=mid, source_role=('GT' if mid == 'GT' else 'SR'),
                          artifact_alias=(mid if mid != 'GT' else ''), verdict=m.STATUS_AVAILABLE,
                          selected_candidate_path='synthetic/test/only.vtu', parsed_pair_count=len(birth),
                          fallback_required='False', notes='', birth=np.asarray(birth, dtype=np.float64),
                          death=np.asarray(death, dtype=np.float64))

        pd_verdicts_available = {
            'GT': _fake_verdict('GT', [0.0, 0.2], [1.0, 0.8]),
            m.CNN: _fake_verdict(m.CNN, [0.1, 0.3], [0.9, 0.6]),
        }
        rows = m.render_pd_diagram_panels(fig2, m.PD_COMPARISON, manifest12, per_sample12, pd_verdicts_available)
        panel_dirs_created.append(m.PANELS_DIR / m.figure_dir_name(fig2))
        check('render_pd_diagram_panels produces one real panel per available_validated method (GT + CNN), '
              'reading coordinates directly from the resolved verdict (never re-parsing point geometry)',
              len(rows) == 2)
        for r in rows:
            p = m.REPO_ROOT / r['output_path']
            check(f"real PD coordinate panel {r['output_path']} exists and is non-empty",
                  p.exists() and p.stat().st_size > 0)

        fallback_row = m.render_scalar_pd_fallback_panel(fig2, m.PD_COMPARISON, manifest12, per_sample12, [m.GAN])
        p = m.REPO_ROOT / fallback_row['output_path']
        check('scalar PD fallback panel is a distinct, real, non-empty file', p.exists() and p.stat().st_size > 0)
        check('scalar fallback panel path is distinct from the coordinate-based panel path',
              fallback_row['output_path'] not in {r['output_path'] for r in rows})

        # Mixed figure: GT + one available method get real coordinate panels, one
        # unavailable method gets the explicit scalar fallback -- both routed by
        # render_all_panels_for_figure() from a single pd_verdicts mapping, and the
        # composite manifest must keep the two panel groups distinct (Section 7).
        mixed_pd_verdicts = {
            'GT': _fake_verdict('GT', [0.0, 0.2], [1.0, 0.8]),
            m.CNN: _fake_verdict(m.CNN, [0.1, 0.3], [0.9, 0.6]),
            m.GAN: dict(figure_id=2, sample_idx=si12, method_id=m.GAN, source_role='SR', artifact_alias='gan',
                          verdict=m.STATUS_UNAVAILABLE, selected_candidate_path='', parsed_pair_count='',
                          fallback_required='True', notes='not found', birth=None, death=None),
        }
        pending_methods = [mid for mid in ('GT', m.CNN, m.GAN) if mixed_pd_verdicts[mid]['verdict'] ==
                            m.STATUS_PENDING]
        check('no method in the mixed-rendering fixture is left pending (a precondition for rendering)',
              pending_methods == [])
        available_mixed = {mid: mixed_pd_verdicts[mid] for mid in ('GT', m.CNN)
                             if mixed_pd_verdicts[mid]['verdict'] == m.STATUS_AVAILABLE}
        unavailable_mixed = [mid for mid in (m.GAN,) if mixed_pd_verdicts[mid]['verdict'] == m.STATUS_UNAVAILABLE]
        mixed_rows = m.render_pd_diagram_panels(fig2, m.PD_COMPARISON, manifest12, per_sample12, available_mixed)
        mixed_fallback = m.render_scalar_pd_fallback_panel(fig2, m.PD_COMPARISON, manifest12, per_sample12,
                                                               unavailable_mixed)
        check('mixed rendering produces real coordinate panels for GT + available methods AND one explicit '
              'scalar-fallback panel for the unavailable method, never a fabricated coordinate panel for it',
              len(mixed_rows) == 2 and (m.REPO_ROOT / mixed_fallback['output_path']).exists())

        composite_pd_verdicts_by_figure = {2: {mid: v['verdict'] for mid, v in mixed_pd_verdicts.items()}}
        mixed_composite_rows = [r for r in m.build_final_composite_manifest_rows(manifest12,
                                                                                    composite_pd_verdicts_by_figure)
                                  if r['figure_id'] == 2 and r['panel_type'] == m.PD_COMPARISON]
        check('the composite manifest keeps pd_coordinate and pd_scalar_fallback as distinct panel groups for '
              'the same figure/panel_type (Section 7)',
              {r['panel_group'] for r in mixed_composite_rows if r['method_id'] in ('GT', m.CNN)} == {'pd_coordinate'}
              and {r['panel_group'] for r in mixed_composite_rows if r['method_id'] == m.GAN} ==
              {'pd_scalar_fallback'})
        check('the composite manifest never labels the scalar-fallback panel as a real coordinate diagram',
              all(r['final_visible_status'] != 'visible' or r['panel_group'] != 'pd_scalar_fallback'
                  for r in mixed_composite_rows if r['method_id'] == m.GAN))
    finally:
        shutil.rmtree(tmp12_root, ignore_errors=True)
        for d in panel_dirs_created:
            shutil.rmtree(d, ignore_errors=True)

    print()
    print('=== 13. Every declared scripted panel type has a real renderer ===')
    DECLARED_SCRIPTED_PANEL_TYPES = {m.SPEED_FIELDS, m.ERROR_MAPS, m.METRIC_STRIP, m.PD_EVIDENCE, m.PD_COMPARISON,
                                        m.PD_MT_TRADEOFF_COMPACT, m.PD_MT_COMPARISON_COMPACT, m.ZOOM_CROP}
    RENDERER_BY_PANEL_TYPE = {
        m.SPEED_FIELDS: m.render_speed_and_error_panels, m.ERROR_MAPS: m.render_speed_and_error_panels,
        m.METRIC_STRIP: m.render_metric_strip, m.PD_EVIDENCE: m.render_pd_diagram_panels,
        m.PD_COMPARISON: m.render_pd_diagram_panels,
        m.PD_MT_TRADEOFF_COMPACT: m.render_pd_mt_tradeoff_compact_panel,
        m.PD_MT_COMPARISON_COMPACT: m.render_pd_mt_comparison_compact_panel, m.ZOOM_CROP: m.render_zoom_crop_panel,
    }
    check('every declared scripted panel type used across all figure contracts has a mapped real renderer',
          set().union(*[set(c['panels']) & DECLARED_SCRIPTED_PANEL_TYPES for c in m.FIGURE_CONTRACTS])
          <= set(RENDERER_BY_PANEL_TYPE))
    all_declared_panel_types = set().union(*[set(c['panels']) for c in m.FIGURE_CONTRACTS])
    check('no declared panel type in any figure contract is unhandled (scripted-renderer or manual-MT set)',
          all_declared_panel_types <= (DECLARED_SCRIPTED_PANEL_TYPES | m.MT_PANEL_TYPES))
    check('every PD panel type has an explicit, validated scalar fallback function distinct from the '
          'coordinate renderer', m.render_scalar_pd_fallback_panel is not None
          and m.render_scalar_pd_fallback_panel is not m.render_pd_diagram_panels)

    print()
    print('=== 14. Panel dimensions + real PNG/PDF inspection (measured, never hardcoded) ===')
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

        # Deterministic corruption fixture: a REAL PNG (valid signature + header),
        # truncated deep enough into the pixel data to force a fast, deterministic
        # decode failure -- never an arbitrary byte string (which forces Pillow to
        # probe every registered format plugin trying to identify the file).
        import time
        valid_src = m.PANELS_DIR / 'corrupt_source.png'
        Image.fromarray((np.random.default_rng(1).uniform(0, 255, size=(64, 64, 3))).astype('uint8')).save(
            valid_src)
        valid_bytes = valid_src.read_bytes()
        check('corruption fixture source PNG is large enough to truncate meaningfully past its header',
              len(valid_bytes) > 300)
        corrupt_png = m.PANELS_DIR / 'corrupt_test.png'
        corrupt_png.write_bytes(valid_bytes[:120])  # valid signature+IHDR, truncated mid-IDAT
        t0 = time.time()
        insp_corrupt = m.inspect_png(corrupt_png)
        elapsed = time.time() - t0
        check('inspect_png flags a truncated-but-recognizable PNG rather than reporting a fabricated pass',
              insp_corrupt['validation_status'] == 'FAIL_corrupt_or_unreadable')
        check('inspect_png returns quickly on corrupt input (deterministic, never hangs)', elapsed < 5.0)
        valid_src.unlink()
        corrupt_png.unlink()

        missing_pdf = m.PANELS_DIR / 'missing.pdf'
        insp_pdf_missing = m.inspect_pdf(missing_pdf)
        check('inspect_pdf reports FAIL for a missing file (dpi/page-count never assumed)',
              insp_pdf_missing['validation_status'] == 'FAIL_missing_or_empty'
              and insp_pdf_missing['pdf_page_count'] == 0)
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
    print('=== 16. Authoritative-input gate: completed-package and simulated-absence cases ===')
    print(f'    Environment: IS_LIGHTWEIGHT_CHECKOUT={m.IS_LIGHTWEIGHT_CHECKOUT}')
    real_state_ok = True
    try:
        m.require_completed_phase2d_a_state(m.FROZEN_SAMPLE_SET)
    except SystemExit:
        real_state_ok = False
    try:
        m.require_protected_files()
        real_files_ok = True
    except SystemExit:
        real_files_ok = False
    if not m.IS_LIGHTWEIGHT_CHECKOUT:
        check('completed authoritative package: require_completed_phase2d_a_state() does not raise', real_state_ok)
        check('completed authoritative package: require_protected_files() does not raise', real_files_ok)
    else:
        check('lightweight checkout: require_completed_phase2d_a_state() correctly raises (real state is '
              'incomplete)', not real_state_ok)
        check('lightweight checkout: require_protected_files() correctly raises (real files are genuinely '
              'absent)', not real_files_ok)

    # Simulated-absence case: ALWAYS exercised, in both environments, via isolated
    # module-constant overrides that point at paths guaranteed not to exist --
    # never by touching the real files.
    fake_doc_path = m.OUT_DIR / '_test16_nonexistent_report.md'
    fake_review_path = m.OUT_DIR / '_test16_nonexistent_review.md'
    old_doc_path, old_review_path = m.p2da.DOC_PATH, m.VISUAL_REVIEW_DOC_PATH
    try:
        m.p2da.DOC_PATH = fake_doc_path
        m.VISUAL_REVIEW_DOC_PATH = fake_review_path
        try:
            m.require_completed_phase2d_a_state(m.FROZEN_SAMPLE_SET)
            check('simulated-absence (isolated fake paths): require_completed_phase2d_a_state() -> SystemExit',
                  False)
        except SystemExit as e:
            check('simulated-absence (isolated fake paths): require_completed_phase2d_a_state() -> SystemExit',
                  'Phase-2D-A report is missing' in str(e) or str(fake_doc_path) in str(e))
    finally:
        m.p2da.DOC_PATH = old_doc_path
        m.VISUAL_REVIEW_DOC_PATH = old_review_path
    check('module constants restored after the simulated-absence gate test',
          m.p2da.DOC_PATH == old_doc_path and m.VISUAL_REVIEW_DOC_PATH == old_review_path)

    fake_missing_protected = m.OUT_DIR / '_test16_nonexistent_protected_file.csv'
    old_all_protected = list(m.ALL_PROTECTED_FILES)
    try:
        m.ALL_PROTECTED_FILES = old_all_protected + [fake_missing_protected]
        try:
            m.require_protected_files()
            check('simulated-absence (isolated fake protected-file entry): require_protected_files() -> '
                  'SystemExit', False)
        except SystemExit as e:
            check('simulated-absence (isolated fake protected-file entry): require_protected_files() -> '
                  'SystemExit', str(fake_missing_protected) in str(e))
    finally:
        m.ALL_PROTECTED_FILES = old_all_protected
    check('ALL_PROTECTED_FILES restored to its original list after the simulated-absence test',
          m.ALL_PROTECTED_FILES == old_all_protected)

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
        m.write_csv(m.PLAN_DIR / 'final_composite_manifest.csv', m.FINAL_COMPOSITE_MANIFEST_FIELDS,
                     composite_rows17)

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
            w.writerow(dict(bad_row, figure_id=str(manual_rows[0]['figure_id']),
                              method_id=manual_rows[0]['method_id']))
        try:
            m.require_manual_topology_panels(m.FROZEN_SAMPLE_SET)
            check('metadata sample_idx mismatch -> SystemExit', False)
        except SystemExit as e:
            check('metadata sample_idx mismatch -> SystemExit', 'sample_idx' in str(e))

        # Restore, then break declared image dimensions vs actual PNG.
        good_row = dict(row, figure_id=str(manual_rows[0]['figure_id']),
                          sample_idx=str(manual_rows[0]['sample_idx']), method_id=manual_rows[0]['method_id'])
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
            other_row = dict(row, figure_id=str(fig1_rows[1]['figure_id']),
                               sample_idx=str(fig1_rows[1]['sample_idx']), method_id=fig1_rows[1]['method_id'],
                               persistence_threshold='99.0')
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

    print()
    print('=== 21. Missing protected file -> hard fail (real file, exact rename/restore) ===')
    target = m.PHASE2D_A_SELECTION_CSVS[0]
    backup = target.with_suffix('.csv.bak_test2db')
    target_bytes_before = target.read_bytes()
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
    check('file restored (exists)', target.exists())
    check('file restored byte-identically', target.exists() and target.read_bytes() == target_bytes_before)

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
    check('file restored byte-identically', target2.read_bytes() == original_bytes)

finally:
    # Top-level safety net: restores every path the entire suite could ever
    # have touched, regardless of which section (if any) raised or was
    # interrupted. Runs on normal completion too, as a final verification.
    _top_level_restore(_top_snapshot)

print()
print('=== 23. Top-level artifact-safety verification ===')
_final_snapshot = _top_level_snapshot()
check('phase2db/ output tree is byte-identical to its pre-suite snapshot',
      _final_snapshot['out_dir'] == _top_snapshot['out_dir'])
check('docs/unified_candidate_analysis_phase2db.md is byte-identical to its pre-suite snapshot',
      _final_snapshot['doc'] == _top_snapshot['doc'])
check('logs/unified_candidate_analysis_phase2db.log is byte-identical to its pre-suite snapshot',
      _final_snapshot['log'] == _top_snapshot['log'])
check('every Phase-2D-A path the golden-path fixture could touch is byte-identical to its pre-suite snapshot',
      _final_snapshot['phase2d_a'] == _top_snapshot['phase2d_a'])
check('no unexpected new files remain under phase2db/',
      set(_final_snapshot['out_dir']) == set(_top_snapshot['out_dir']))

print()
if failures:
    print(f'{len(failures)} FAILURE(S): {failures}')
    sys.exit(1)
else:
    print('ALL TESTS PASSED')
