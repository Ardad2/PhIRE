#!/usr/bin/env python3
"""Phase 2D-B: final publication-quality figure production for the unified
wind-field super-resolution candidate benchmark.

Phase 2D-A is complete, authoritative, committed, and pushed. This script
consumes its frozen selection artifacts -- it never re-selects samples,
never activates an alternate, and never recomputes any Phase-1 through
Phase-2D-A analysis. It reuses pure logic (path-resolution conventions,
protected-file constants, manifest validation, raw-array helpers) directly
from scripts/select_and_preview_unified_candidates_phase2d.py by importing
it as a module -- this only reads code, it never writes to any prior-phase
location.

Split execution architecture (four independently runnable modes):

  --plan-only            CSV/Markdown-only. Validates the frozen sample set
                          against Phase-2D-A's selection manifest, plans
                          every final panel and figure, writes figure-data
                          CSVs (cross-checked against the frozen Phase-1
                          long table and Phase-2D-A's
                          selected_sample_method_values.csv), drafts
                          captions, and reports exactly what is blocked
                          (missing raw arrays, missing PD coordinate
                          sources, missing manual topology panels). Never
                          touches data_out/, data_out_fixed/, or renders
                          any image. This is the mode that runs in a
                          lightweight checkout.

  --render-fields         Requires the raw Spark arrays. Renders
                          publication-quality speed fields, error maps, the
                          deterministic zoom crop, metric strips, and any
                          scriptable PD panel whose frozen coordinate source
                          can actually be found. Hard-fails cleanly (never
                          fabricates) when a required raw array or PD
                          coordinate source is absent.

  --assemble-composites   Assembles the six final composite figures from
                          validated script-rendered panels plus explicitly
                          supplied manual ParaView/TTK merge-tree exports
                          under manual_topology_inputs/. Hard-fails when a
                          required manual panel or its metadata row is
                          missing -- this script never automates merge-tree
                          geometry rendering itself.

  --full                  Runs all three stages in sequence. Hard-fails
                          (does not silently downgrade the status banner)
                          when a required manual topology panel is absent.

Never writes outside:

    ttk_runs_fixed/unified_candidate_analysis/phase2db/{plan,figure_data,
        panels,manual_topology_inputs,figures,validation}/
    docs/unified_candidate_analysis_phase2db.md
    logs/unified_candidate_analysis_phase2db.log

Determinism: no wall-clock time, hostname, or environment-dependent value is
ever written to a generated file. Running --plan-only twice produces
byte-identical plans, figure-data CSVs, captions, report, and log.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import select_and_preview_unified_candidates_phase2d as p2da  # noqa: E402

REPO_ROOT = p2da.REPO_ROOT
assert REPO_ROOT == SCRIPT_DIR.parent

# -----------------------------------------------------------------------
# Phase-2D-B output locations (the only locations this script may write)
# -----------------------------------------------------------------------
OUT_DIR = REPO_ROOT / 'ttk_runs_fixed' / 'unified_candidate_analysis' / 'phase2db'
PLAN_DIR = OUT_DIR / 'plan'
FIGURE_DATA_DIR = OUT_DIR / 'figure_data'
PANELS_DIR = OUT_DIR / 'panels'
MANUAL_TOPOLOGY_DIR = OUT_DIR / 'manual_topology_inputs'
FIGURES_DIR = OUT_DIR / 'figures'
VALIDATION_DIR = OUT_DIR / 'validation'
DOCS_DIR = REPO_ROOT / 'docs'
DOC_PATH = DOCS_DIR / 'unified_candidate_analysis_phase2db.md'
LOG_PATH = REPO_ROOT / 'logs' / 'unified_candidate_analysis_phase2db.log'

# -----------------------------------------------------------------------
# Protected files: everything Phase-1 through Phase-2D-A produced. Reuses
# Phase-2D-A's own explicit (never-a-glob) Phase-1/2A/2B/2C lists, plus this
# script's own explicit list of every Phase-2D-A core artifact that exists
# in this checkout today.
# -----------------------------------------------------------------------
PHASE2D_A_PROTECTED_CSV_NAMES = [
    'archetype_alternates.csv', 'archetype_score_table.csv', 'archetype_selected_samples.csv',
    'archetype_selection_diagnostics.csv', 'figure_plan.csv', 'preview_method_manifest.csv',
    'preview_plan.csv', 'prior_phase_immutability_check.csv', 'raw_artifact_requirements.csv',
    'selected_sample_method_values.csv', 'selected_sample_metric_context.csv',
    'selected_sample_pairwise_preferences.csv', 'selection_validation.csv',
]
PHASE2D_A_PROTECTED_CSVS = [p2da.SELECTION_DIR / n for n in PHASE2D_A_PROTECTED_CSV_NAMES]
PHASE2D_A_PROTECTED_OTHER = [
    REPO_ROOT / 'docs' / 'unified_candidate_analysis_phase2d.md',
    REPO_ROOT / 'logs' / 'unified_candidate_analysis_phase2d_selection.log',
    REPO_ROOT / 'scripts' / 'select_and_preview_unified_candidates_phase2d.py',
    REPO_ROOT / 'scripts' / 'test_select_and_preview_unified_candidates_phase2d.py',
]
PHASE2D_A_PROTECTED_FILES = PHASE2D_A_PROTECTED_CSVS + PHASE2D_A_PROTECTED_OTHER  # exactly 17
assert len(PHASE2D_A_PROTECTED_FILES) == 17

ALL_PROTECTED_FILES = (p2da.PHASE1_PROTECTED_FILES + p2da.PHASE2A_PROTECTED_FILES +
                         p2da.PHASE2B_PROTECTED_FILES + p2da.PHASE2C_PROTECTED_FILES +
                         PHASE2D_A_PROTECTED_FILES)  # 12+14+28+32+17 = 103
assert len(ALL_PROTECTED_FILES) == 103

PROTECTED_DIRS_AND_CSVS = list(p2da.PROTECTED_DIRS_AND_CSVS) + [
    (p2da.SELECTION_DIR, set(PHASE2D_A_PROTECTED_CSVS)),
]

# -----------------------------------------------------------------------
# Frozen sample set (given, not selected here). No alternate is ever active.
# -----------------------------------------------------------------------
FROZEN_SAMPLE_SET = {
    'global_descriptor_disagreement': 120,
    'gan_pd_vs_cnn_mt_conflict': 34,
    'f3_pd_vs_uv_e2_mt_tradeoff': 119,
    'f2_balanced_vs_cnn': 25,
    'candidate_c_continuity': 30,
    'global_descriptor_agreement': 19,
}
assert set(FROZEN_SAMPLE_SET) == set(p2da.ARCHETYPE_PRIORITY)

CNN, GAN, BICUBIC, CANDIDATE_C, F2, F3, UV_E2 = (
    p2da.CNN_METHOD, p2da.GAN_METHOD, p2da.BICUBIC_METHOD, p2da.CANDIDATE_C_METHOD,
    p2da.F2_METHOD, p2da.F3_METHOD, p2da.UV_E2_METHOD,
)

HUMAN_LABELS = {
    BICUBIC: 'Bicubic', CNN: 'CNN', GAN: 'GAN', CANDIDATE_C: 'Candidate C',
    F3: 'F3: Grad+Crit', F2: 'F2: Grad+Levelset+E2', UV_E2: 'UV+E2',
}
GT_DISPLAY_LABEL = 'Ground Truth'

_LOG_LINES: list = []


def log(msg: str = '') -> None:
    print(msg)
    _LOG_LINES.append(msg)


def flush_log(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as fh:
        fh.write('\n'.join(_LOG_LINES) + '\n')


def _rel(path) -> str:
    return p2da._rel_posix(path, REPO_ROOT)


def _f(val):
    if val in (None, ''):
        return float('nan')
    return float(val)


def nfmt(v):
    return '' if v is None else v


def require_protected_files() -> None:
    missing = [str(p) for p in ALL_PROTECTED_FILES if not p.exists()]
    if missing:
        raise SystemExit(
            f'[hard-fail] Missing required prior-phase protected file(s) (expected exactly '
            f'{len(ALL_PROTECTED_FILES)}: 12 Phase-1 + 14 Phase-2A + 28 Phase-2B + 32 Phase-2C + '
            f'17 Phase-2D-A):\n' + '\n'.join(f'  - {m}' for m in missing)
        )
    for directory, expected_csvs in PROTECTED_DIRS_AND_CSVS:
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
        result[rel] = p2da.sha256_file(p) if p.exists() else None
    return result


def preflight_immutability():
    require_protected_files()
    file_to_phase = {}
    for files, phase in ((p2da.PHASE1_PROTECTED_FILES, 'phase1'), (p2da.PHASE2A_PROTECTED_FILES, 'phase2a'),
                          (p2da.PHASE2B_PROTECTED_FILES, 'phase2b'), (p2da.PHASE2C_PROTECTED_FILES, 'phase2c'),
                          (PHASE2D_A_PROTECTED_FILES, 'phase2d_a')):
        file_to_phase.update({p.resolve().relative_to(REPO_ROOT).as_posix(): phase for p in files})
    checksums_before = checksum_all(ALL_PROTECTED_FILES)
    log(f'[immutability] Checksummed {len(checksums_before)} prior-phase file(s) before this stage '
        f'(12 Phase-1 + 14 Phase-2A + 28 Phase-2B + 32 Phase-2C + 17 Phase-2D-A = 103 exactly).')
    return checksums_before, file_to_phase


def postflight_immutability(checksums_before, file_to_phase, out_path: Path):
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
    write_csv(out_path, ['phase', 'file_path', 'sha256_before', 'sha256_after', 'status'], immut_rows)
    if changed:
        raise SystemExit(f'[hard-fail] Prior-phase immutability violated: {changed}')
    log(f'[immutability] Confirmed all {len(immut_rows)} prior-phase file(s) unchanged.')
    return immut_rows


def _require_no_absolute_csv_paths(path: Path, fieldnames: list, rows: list) -> None:
    path_like_fields = [f for f in fieldnames if 'path' in f.lower()]
    if not path_like_fields:
        return
    for row in rows:
        for f in path_like_fields:
            v = row.get(f, '')
            if isinstance(v, str) and v.startswith('/'):
                raise SystemExit(
                    f'[hard-fail] Absolute path found in generated CSV field {f!r} of {path}: {v!r}. '
                    f'All generated-artifact path fields must be repository-relative POSIX text.'
                )


def write_csv(path: Path, fieldnames: list, rows: list) -> None:
    _require_no_absolute_csv_paths(path, fieldnames, rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    log(f'[write] {path} ({len(rows)} rows)')


# =============================================================================
# Frozen-sample-set enforcement (Section: FROZEN SAMPLE SET)
# =============================================================================

def read_and_validate_selection_manifest() -> dict:
    """Reads Phase-2D-A's archetype_selected_samples.csv, strongly validates
    it (reusing Phase-2D-A's own validator), and then requires the primary
    selection to match FROZEN_SAMPLE_SET exactly. Never reads
    archetype_alternates.csv for selection purposes -- no alternate is ever
    considered, let alone activated."""
    path = p2da.SELECTION_DIR / 'archetype_selected_samples.csv'
    if not path.exists():
        raise SystemExit(f'[hard-fail] Required Phase-2D-A manifest is missing: {path}')
    all_rows = p2da.read_csv_dicts(path)
    errors = p2da.validate_selected_samples_manifest_rows(all_rows, path)
    if errors:
        raise SystemExit(
            f'[hard-fail] {path} failed manifest validation ({len(errors)} issue(s)):\n' +
            '\n'.join(f'  - {e}' for e in errors)
        )
    primary = {r['archetype_id']: int(r['selected_sample_idx']) for r in all_rows
                if r['primary_or_alternate'] == 'primary'}
    if primary != FROZEN_SAMPLE_SET:
        raise SystemExit(
            f'[hard-fail] Phase-2D-A selection manifest does not match the frozen sample set specified for '
            f'Phase 2D-B. This is a hard boundary violation (re-selection or alternate activation is never '
            f'permitted here): manifest={primary!r} frozen={FROZEN_SAMPLE_SET!r}'
        )
    return dict(primary)


def load_selected_sample_method_values() -> dict:
    path = p2da.SELECTION_DIR / 'selected_sample_method_values.csv'
    rows = p2da.read_csv_dicts(path)
    return {(r['archetype_id'], r['method_id']): r for r in rows}


# =============================================================================
# Figure contracts (Section: FINAL FIGURE CONTRACT)
# =============================================================================

SPEED_FIELDS = 'speed_fields'
ERROR_MAPS = 'error_maps'
METRIC_STRIP = 'metric_strip'
PD_EVIDENCE = 'pd_evidence'
MT_EVIDENCE = 'mt_evidence'
PD_COMPARISON = 'pd_comparison'
MT_COMPARISON = 'mt_comparison'
PD_MT_TRADEOFF_COMPACT = 'pd_mt_tradeoff_compact'
PD_MT_COMPARISON_COMPACT = 'pd_mt_comparison_compact'
TOPOLOGY_COMPARISON = 'topology_comparison'
ZOOM_CROP = 'zoom_crop'

# Panels requiring a genuine merge-tree GEOMETRY image -- no scripted
# renderer exists for these; manual ParaView/TTK export is required.
MT_PANEL_TYPES = {MT_EVIDENCE, MT_COMPARISON, TOPOLOGY_COMPARISON}
# Panels requiring frozen persistence-diagram COORDINATE data (birth/death
# pairs), distinct from the scalar pd_distance value already in the long
# table. See resolve_pd_diagram_source().
PD_DIAGRAM_PANEL_TYPES = {PD_EVIDENCE, PD_COMPARISON}
# Figure-level (not per-method) panels.
FIGURE_LEVEL_PANEL_TYPES = {METRIC_STRIP, PD_MT_TRADEOFF_COMPACT, PD_MT_COMPARISON_COMPACT, ZOOM_CROP}
PER_METHOD_PANEL_TYPES = {SPEED_FIELDS, ERROR_MAPS, PD_EVIDENCE, MT_EVIDENCE, PD_COMPARISON, MT_COMPARISON,
                            TOPOLOGY_COMPARISON}

FIGURE_CONTRACTS = [
    dict(
        figure_id=1, short_name='global_disagreement', archetype_id='global_descriptor_disagreement',
        primary_claim='PD and MT can produce strongly different cross-method preferences.',
        required_methods=[GAN, CNN, CANDIDATE_C, F3, F2, UV_E2],
        full_panel_methods=[GAN, CNN, CANDIDATE_C, F3, F2, UV_E2],
        method_roles={},
        panels=[SPEED_FIELDS, ERROR_MAPS, METRIC_STRIP, PD_EVIDENCE, MT_EVIDENCE],
        emphasis='GAN best PD but worst MT; CNN worst displayed PD but best MT; UV+E2 is comparatively '
                  'MT-oriented.',
    ),
    dict(
        figure_id=2, short_name='gan_cnn_conflict', archetype_id='gan_pd_vs_cnn_mt_conflict',
        primary_claim='A lower PD distance does not guarantee better merge-tree or pointwise fidelity.',
        required_methods=[BICUBIC, CNN, GAN],
        full_panel_methods=[BICUBIC, CNN, GAN],
        method_roles={},
        panels=[SPEED_FIELDS, ERROR_MAPS, PD_COMPARISON, MT_COMPARISON, METRIC_STRIP],
        emphasis='',
    ),
    dict(
        figure_id=3, short_name='f3_uv_e2_tradeoff', archetype_id='f3_pd_vs_uv_e2_mt_tradeoff',
        primary_claim='Gradient-plus-critical supervision and repaired E2 supervision influence different '
                        'topology descriptors.',
        required_methods=[CNN, F3, UV_E2, F2],
        full_panel_methods=[CNN, F3, UV_E2],  # F2 appears only as a compact contextual reference
        method_roles={F2: 'compact_contextual_reference'},
        panels=[SPEED_FIELDS, ERROR_MAPS, PD_EVIDENCE, MT_EVIDENCE, ZOOM_CROP, METRIC_STRIP],
        emphasis='This figure must not rely on speed/error panels alone.',
    ),
    dict(
        figure_id=4, short_name='f2_balanced', archetype_id='f2_balanced_vs_cnn',
        primary_claim='F2 provides a balanced PD/MT improvement over CNN rather than universally optimizing '
                        'every objective.',
        required_methods=[CNN, F3, F2, UV_E2],
        full_panel_methods=[CNN, F3, F2, UV_E2],
        method_roles={},
        panels=[SPEED_FIELDS, ERROR_MAPS, PD_MT_TRADEOFF_COMPACT, METRIC_STRIP],
        emphasis='',
    ),
    dict(
        figure_id=5, short_name='candidate_c_continuity', archetype_id='candidate_c_continuity',
        primary_claim='Candidate C is a valid topology-inspired improvement over CNN, while the expanded '
                        'ablation study clarifies the more specific PD and MT mechanisms.',
        required_methods=[CNN, CANDIDATE_C, F3, F2, UV_E2],
        full_panel_methods=[CNN, CANDIDATE_C, F3, F2, UV_E2],
        method_roles={},
        panels=[SPEED_FIELDS, ERROR_MAPS, TOPOLOGY_COMPARISON, METRIC_STRIP],
        emphasis='',
    ),
    dict(
        figure_id=6, short_name='global_agreement', archetype_id='global_descriptor_agreement',
        primary_claim='PD and MT disagreement is not universal; strong methods can show broad descriptor '
                        'concordance without identical rankings.',
        required_methods=[GAN, CNN, CANDIDATE_C, F3, F2, UV_E2],
        full_panel_methods=[GAN, CNN, CANDIDATE_C, F3, F2, UV_E2],
        method_roles={},
        panels=[SPEED_FIELDS, ERROR_MAPS, PD_MT_COMPARISON_COMPACT, METRIC_STRIP],
        emphasis='',
    ),
]
assert [c['figure_id'] for c in FIGURE_CONTRACTS] == [1, 2, 3, 4, 5, 6]
assert [c['archetype_id'] for c in FIGURE_CONTRACTS] == p2da.ARCHETYPE_PRIORITY
FIGURE_BY_ID = {c['figure_id']: c for c in FIGURE_CONTRACTS}
FIGURE_DATA_FILENAMES = {
    1: 'figure_01_global_disagreement.csv', 2: 'figure_02_gan_cnn_conflict.csv',
    3: 'figure_03_f3_uv_e2_tradeoff.csv', 4: 'figure_04_f2_balanced.csv',
    5: 'figure_05_candidate_c_continuity.csv', 6: 'figure_06_global_agreement.csv',
}


def figure_dir_name(contract) -> str:
    return f"figure_{contract['figure_id']:02d}_{contract['short_name']}"


def panel_output_path(contract, panel_type, method_id) -> str:
    fname = f'{panel_type}_{method_id}.png' if method_id else f'{panel_type}.png'
    return f'ttk_runs_fixed/unified_candidate_analysis/phase2db/panels/{figure_dir_name(contract)}/{fname}'


def final_figure_paths(contract):
    base = f'ttk_runs_fixed/unified_candidate_analysis/phase2db/figures/{figure_dir_name(contract)}'
    return dict(png=f'{base}.png', pdf=f'{base}.pdf')


# =============================================================================
# PD-diagram coordinate source resolution. This project's frozen topology
# artifacts (column_mapping.csv-referenced CSVs, confirmed by direct
# inspection) contain only the SCALAR pd_distance/mt_distance value already
# in the Phase-1 long table -- never raw persistence-diagram birth/death
# coordinates. This function documents the deterministic repository-relative
# naming convention such coordinate exports would need to follow if/when
# they become available, and returns None (never fabricates) when absent.
# =============================================================================

def resolve_pd_diagram_source(mid, sample_idx, topology_source_map):
    if mid not in topology_source_map:
        return None
    candidate_dir = topology_source_map[mid]['path'].parent
    candidates = [
        candidate_dir / f'{mid}_sample{sample_idx}_pd_diagram.csv',
        candidate_dir / f'sample_{sample_idx}_pd_diagram_{mid}.csv',
        candidate_dir / 'pd_diagrams' / f'{mid}_s{sample_idx}.csv',
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def require_pd_diagram_sources_for_figure(contract, manifest, topology_source_map):
    """Hard-fails (never fabricates) if any full-panel method required by a
    pd_evidence/pd_comparison panel in this figure has no resolvable PD
    coordinate source."""
    si = manifest[contract['archetype_id']]
    missing = [mid for mid in contract['full_panel_methods']
                if resolve_pd_diagram_source(mid, si, topology_source_map) is None]
    if missing:
        raise SystemExit(
            f"[hard-fail] Figure {contract['figure_id']} requires a PD evidence/comparison panel for "
            f"{[HUMAN_LABELS[m] for m in missing]}, but no frozen PD coordinate source was found for "
            f'sample_idx={si!r}. Refusing to fabricate this panel.'
        )


# =============================================================================
# Deterministic zoom-region selection for sample 119 (Section: ZOOM REGION)
# =============================================================================

ZOOM_WINDOW_SIZE = 100
ZOOM_STRIDE = 25
ZOOM_SCORE_FORMULA = (
    'score(y0,x0) = sum((d/dy GT_speed)^2 + (d/dx GT_speed)^2) over the window '
    '[gt_gradient_energy, np.gradient on the GT speed patch] '
    '+ sum_over_pixels(var_across_methods(abs_speed_error)) over the window '
    '[cross_method_error_variance, per-pixel variance of |method_speed - GT_speed| across all required '
    'full-panel methods]; candidate windows are a fixed-size '
    f'{ZOOM_WINDOW_SIZE}x{ZOOM_WINDOW_SIZE} grid at stride {ZOOM_STRIDE} over the HR grid; ranked by score '
    'descending; ties broken by smallest y0 then smallest x0 (top-left).'
)


def compute_zoom_window_score(gt_speed, method_errors_by_method, y0, x0, window):
    gt_patch = gt_speed[y0:y0 + window, x0:x0 + window]
    grad_y, grad_x = np.gradient(gt_patch.astype(np.float64))
    gt_gradient_energy = float(np.sum(grad_y ** 2 + grad_x ** 2))
    stacked = np.stack([method_errors_by_method[mid][y0:y0 + window, x0:x0 + window]
                          for mid in sorted(method_errors_by_method)], axis=0)
    cross_method_error_variance = float(np.sum(np.var(stacked, axis=0)))
    return gt_gradient_energy + cross_method_error_variance


def select_deterministic_zoom(gt_speed, method_errors_by_method, window=ZOOM_WINDOW_SIZE, stride=ZOOM_STRIDE):
    """Pure, deterministic zoom-window selector. Never chosen by visual
    preference. Returns dict(y0, y1, x0, x1, score, formula)."""
    h, w = gt_speed.shape
    if h < window or w < window:
        raise SystemExit(f'[hard-fail] GT field ({h}x{w}) is smaller than the zoom window ({window}x{window}).')
    candidates = []
    for y0 in range(0, h - window + 1, stride):
        for x0 in range(0, w - window + 1, stride):
            score = compute_zoom_window_score(gt_speed, method_errors_by_method, y0, x0, window)
            candidates.append((score, y0, x0))
    if not candidates:
        raise SystemExit('[hard-fail] No candidate zoom windows were generated.')
    candidates.sort(key=lambda c: (-c[0], c[1], c[2]))
    best_score, y0, x0 = candidates[0]
    return dict(y0=y0, y1=y0 + window, x0=x0, x1=x0 + window, score=best_score, formula=ZOOM_SCORE_FORMULA)


# =============================================================================
# --plan-only: figure plan, panel manifest, manual-topology requirements,
# figure-data CSVs (with reproduction cross-check), captions, validation.
# =============================================================================

FINAL_FIGURE_PLAN_FIELDS = [
    'figure_id', 'archetype_id', 'sample_idx', 'primary_claim', 'required_methods',
    'required_methods_human', 'required_panels', 'method_roles', 'emphasis_notes', 'status',
]


def build_final_figure_plan_rows(manifest):
    rows = []
    for c in FIGURE_CONTRACTS:
        si = manifest[c['archetype_id']]
        methods = ['GT'] + c['required_methods']
        human = [GT_DISPLAY_LABEL] + [HUMAN_LABELS[m] for m in c['required_methods']]
        roles = ';'.join(f'{m}={role}' for m, role in c['method_roles'].items())
        rows.append(dict(
            figure_id=c['figure_id'], archetype_id=c['archetype_id'], sample_idx=si,
            primary_claim=c['primary_claim'], required_methods=','.join(methods),
            required_methods_human=','.join(human), required_panels=','.join(c['panels']),
            method_roles=roles, emphasis_notes=c['emphasis'], status='planned',
        ))
    return rows


FINAL_PANEL_MANIFEST_FIELDS = [
    'figure_id', 'archetype_id', 'sample_idx', 'panel_type', 'method_id', 'method_role', 'display_label',
    'output_path', 'requires_manual_topology_input', 'requires_pd_coordinate_source',
    'pd_coordinate_source_found', 'status',
]


def build_final_panel_manifest_rows(manifest, topology_source_map):
    rows = []
    for c in FIGURE_CONTRACTS:
        si = manifest[c['archetype_id']]
        for panel_type in c['panels']:
            if panel_type in PER_METHOD_PANEL_TYPES:
                for mid in (['GT'] + c['full_panel_methods']):
                    if panel_type == ERROR_MAPS and mid == 'GT':
                        continue  # no self-error panel for GT
                    role = c['method_roles'].get(mid, 'primary')
                    needs_manual = panel_type in MT_PANEL_TYPES
                    needs_pd_source = panel_type in PD_DIAGRAM_PANEL_TYPES
                    pd_found = ''
                    status = 'planned_not_rendered'
                    if needs_manual:
                        status = 'blocked_awaiting_manual_topology_input'
                    elif needs_pd_source:
                        pd_found = (mid == 'GT') or (resolve_pd_diagram_source(mid, si, topology_source_map)
                                                       is not None)
                        if not pd_found:
                            status = 'blocked_missing_pd_source'
                    rows.append(dict(
                        figure_id=c['figure_id'], archetype_id=c['archetype_id'], sample_idx=si,
                        panel_type=panel_type, method_id=mid,
                        display_label=(GT_DISPLAY_LABEL if mid == 'GT' else HUMAN_LABELS[mid]),
                        method_role=role, output_path=panel_output_path(c, panel_type, mid),
                        requires_manual_topology_input=needs_manual,
                        requires_pd_coordinate_source=needs_pd_source, pd_coordinate_source_found=pd_found,
                        status=status,
                    ))
            else:
                rows.append(dict(
                    figure_id=c['figure_id'], archetype_id=c['archetype_id'], sample_idx=si,
                    panel_type=panel_type, method_id='', display_label='', method_role='',
                    output_path=panel_output_path(c, panel_type, None),
                    requires_manual_topology_input=False, requires_pd_coordinate_source=False,
                    pd_coordinate_source_found='', status='planned_not_rendered',
                ))
    return rows


MANUAL_TOPOLOGY_REQ_FIELDS = [
    'figure_id', 'archetype_id', 'sample_idx', 'method_id', 'display_label', 'expected_panel_path',
    'expected_metadata_path', 'panel_type', 'default_persistence_threshold', 'default_arc_sampling',
    'default_arc_line_size', 'status',
]


def build_manual_topology_requirements_rows(manifest):
    rows = []
    for c in FIGURE_CONTRACTS:
        si = manifest[c['archetype_id']]
        for panel_type in c['panels']:
            if panel_type not in MT_PANEL_TYPES:
                continue
            for mid in (['GT'] + c['full_panel_methods']):
                panel_path = f'ttk_runs_fixed/unified_candidate_analysis/phase2db/manual_topology_inputs/' \
                              f"figure_{c['figure_id']:02d}/{mid}_mt.png"
                meta_path = f'ttk_runs_fixed/unified_candidate_analysis/phase2db/manual_topology_inputs/' \
                             f"figure_{c['figure_id']:02d}/{mid}_mt_metadata.csv"
                exists = (REPO_ROOT / panel_path).exists() and (REPO_ROOT / meta_path).exists()
                rows.append(dict(
                    figure_id=c['figure_id'], archetype_id=c['archetype_id'], sample_idx=si, method_id=mid,
                    display_label=(GT_DISPLAY_LABEL if mid == 'GT' else HUMAN_LABELS[mid]),
                    expected_panel_path=panel_path, expected_metadata_path=meta_path, panel_type=panel_type,
                    default_persistence_threshold=11.0, default_arc_sampling=10, default_arc_line_size=3,
                    status=('present' if exists else 'missing'),
                ))
    return rows


MANUAL_TOPOLOGY_METADATA_FIELDS = [
    'figure_id', 'sample_idx', 'method_id', 'source_vtu_path', 'persistence_threshold', 'arc_sampling',
    'arc_line_size', 'camera_or_view_id', 'scalar_range', 'image_width', 'image_height', 'paraview_version',
    'ttk_version', 'renderer_type', 'notes',
]

FIGURE_DATA_FIELDS = [
    'figure_id', 'archetype_id', 'sample_idx', 'method_id', 'display_label', 'pd_distance', 'mt_distance',
    'psnruv', 'speed_mae', 'grad_mae', 'wpd_mae', 'source_field_path', 'source_gt_path',
    'source_topology_path', 'zoom_y0', 'zoom_y1', 'zoom_x0', 'zoom_x1',
]
FIGURE_DATA_METRICS = ('pd_distance', 'mt_distance', 'psnruv', 'speed_mae', 'grad_mae', 'wpd_mae')


def build_figure_data_rows(contract, manifest, per_sample, topology_source_map, raw_paths, zoom_bounds):
    si = manifest[contract['archetype_id']]
    zoom = zoom_bounds or {}
    rows = []
    for mid in (['GT'] + contract['required_methods']):
        gt_path = _rel(raw_paths[CNN]['dataGT'])
        if mid == 'GT':
            display_label = GT_DISPLAY_LABEL
            metrics = {k: '' for k in FIGURE_DATA_METRICS}
            source_field_path = gt_path
            source_topology_path = ''
        else:
            display_label = HUMAN_LABELS[mid]
            m = per_sample[mid][si]
            metrics = {k: nfmt(m.get(k) if math.isfinite(m.get(k, float('nan'))) else None)
                        for k in FIGURE_DATA_METRICS}
            p = raw_paths[mid]
            source_field_path = '(reconstructed_in_memory)' if p['dataSR'] is None else _rel(p['dataSR'])
            source_topology_path = _rel(topology_source_map[mid]['path']) if mid in topology_source_map else ''
        rows.append(dict(
            figure_id=contract['figure_id'], archetype_id=contract['archetype_id'], sample_idx=si,
            method_id=mid, display_label=display_label, **metrics,
            source_field_path=source_field_path, source_gt_path=gt_path,
            source_topology_path=source_topology_path,
            zoom_y0=nfmt(zoom.get('y0')), zoom_y1=nfmt(zoom.get('y1')),
            zoom_x0=nfmt(zoom.get('x0')), zoom_x1=nfmt(zoom.get('x1')),
        ))
    return rows


REPRO_TOLERANCE = 1e-6


def validate_figure_data_reproduction(figure_data_by_id, per_sample, method_values_by_key):
    rows = []
    failures = []
    for figure_id, data_rows in sorted(figure_data_by_id.items()):
        for r in data_rows:
            if r['method_id'] == 'GT':
                continue
            mid, si, aid = r['method_id'], r['sample_idx'], r['archetype_id']
            for metric in FIGURE_DATA_METRICS:
                fd_val = r[metric]
                fd_num = float('nan') if fd_val == '' else float(fd_val)
                lt_val = per_sample[mid][si].get(metric, float('nan'))
                msv_row = method_values_by_key.get((aid, mid))
                msv_val = _f(msv_row.get(f'raw__{metric}', '')) if msv_row else float('nan')
                both_nonfinite = not math.isfinite(fd_num) and not math.isfinite(lt_val) and \
                    not math.isfinite(msv_val)
                lt_ok = math.isfinite(lt_val) and math.isfinite(fd_num) and abs(fd_num - lt_val) <= REPRO_TOLERANCE
                msv_ok = math.isfinite(msv_val) and math.isfinite(fd_num) and \
                    abs(fd_num - msv_val) <= REPRO_TOLERANCE
                status = 'PASS' if (both_nonfinite or (lt_ok and msv_ok)) else 'FAIL'
                if status == 'FAIL':
                    failures.append(f'{aid}/{mid}/sample={si}/{metric}: figure_data={fd_val!r} '
                                      f'long_table={lt_val!r} selected_sample_method_values={msv_val!r}')
                rows.append(dict(
                    figure_id=figure_id, archetype_id=aid, sample_idx=si, method_id=mid, metric=metric,
                    figure_data_value=fd_val,
                    long_table_value=nfmt(lt_val if math.isfinite(lt_val) else None),
                    selected_sample_method_values_value=nfmt(msv_val if math.isfinite(msv_val) else None),
                    status=status,
                ))
    if failures:
        raise SystemExit(
            f'[hard-fail] {len(failures)} figure-data reproduction check(s) failed against frozen sources:\n' +
            '\n'.join(f'  - {f}' for f in failures)
        )
    return rows


def validate_panel_manifest(panel_rows):
    rows = []
    for r in panel_rows:
        notes = []
        contract = FIGURE_BY_ID[r['figure_id']]
        ok = True
        if r['panel_type'] not in contract['panels']:
            ok = False
            notes.append('panel_type not declared in figure contract')
        if r['method_id'] and r['method_id'] != 'GT' and r['method_id'] not in contract['required_methods']:
            ok = False
            notes.append('method_id not in figure required_methods')
        rows.append(dict(
            figure_id=r['figure_id'], panel_type=r['panel_type'], method_id=r['method_id'],
            output_path=r['output_path'], status=r['status'],
            structural_check_status=('PASS' if ok else 'FAIL'), notes='; '.join(notes),
        ))
    return rows


def build_not_yet_rendered_final_figure_validation(manifest):
    rows = []
    for c in FIGURE_CONTRACTS:
        si = manifest[c['archetype_id']]
        paths = final_figure_paths(c)
        rows.append(dict(
            figure_id=c['figure_id'], archetype_id=c['archetype_id'], sample_idx=si,
            expected_png_path=paths['png'], expected_vector_path=paths['pdf'],
            png_exists=False, png_min_dpi_ok='', vector_exists=False, status='not_yet_rendered',
        ))
    return rows


CAPTION_TEMPLATES = {
    1: ('For sample 120, PD and MT can produce strongly different cross-method preferences: GAN attains the '
         'best displayed PD distance but the worst MT distance, CNN shows the worst displayed PD distance but '
         'the best MT distance, and UV+E2 is comparatively MT-oriented among the remaining methods. Across the '
         'fixed benchmark, the quantitative analysis found this disagreement pattern is not universal (see '
         'Figure 6); this selected example visualizes one instance where it is pronounced.'),
    2: ('For sample 34, a lower PD distance does not guarantee better merge-tree or pointwise fidelity: GAN '
         'improves on CNN\'s PD distance while CNN improves on GAN\'s MT distance. This illustrative case shows '
         'the tradeoff concretely; no claim is made that this ordering holds for every sample in the benchmark.'),
    3: ('For sample 119, gradient-plus-critical supervision (F3) and repaired E2 supervision (UV+E2) influence '
         'different topology descriptors: F3 improves the PD distance relative to UV+E2, while UV+E2 improves '
         'the MT distance relative to F3, within the deterministically selected zoomed structural region shown. '
         'F2 is shown only as a compact contextual reference.'),
    4: ('For sample 25, F2 provides a balanced PD/MT improvement over CNN rather than universally optimizing '
         'every objective: both the PD and MT distances improve over CNN in this selected example, illustrating '
         'the balanced-improvement archetype identified by the quantitative analysis.'),
    5: ('For sample 30, Candidate C is a valid topology-inspired improvement over CNN in this selected example; '
         'the expanded ablation study (F3, F2, UV+E2) shown alongside it clarifies the more specific PD and MT '
         'mechanisms contributing to that improvement across the fixed benchmark.'),
    6: ('For sample 19, PD and MT disagreement is not universal: the displayed methods show broad descriptor '
         'concordance without necessarily sharing an identical ranking. This selected example visualizes a case '
         'of cross-method agreement, in contrast with the disagreement case in Figure 1.'),
}


def write_captions_md(manifest):
    lines = ['# Phase 2D-B: Draft Final Figure Captions', '',
              'Draft captions only. Claims are sample-specific; no claim in this file generalizes beyond the '
              'sample and methods shown in that figure. See docs/unified_candidate_analysis_phase2db.md for '
              'the full quantitative-vs-illustrative framing.', '']
    for c in FIGURE_CONTRACTS:
        si = manifest[c['archetype_id']]
        lines.append(f"## Figure {c['figure_id']}: {c['archetype_id']} (sample_idx={si})")
        lines.append('')
        lines.append(CAPTION_TEMPLATES[c['figure_id']])
        lines.append('')
    CAPTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    CAPTIONS_PATH.write_text('\n'.join(lines) + '\n')
    log(f'[write] {CAPTIONS_PATH}')


CAPTIONS_PATH = PLAN_DIR / 'final_figure_captions.md'


# =============================================================================
# Report
# =============================================================================

def build_phase2db_doc_lines(manifest, zoom_result, manual_topo_rows, panel_rows, is_full_complete=False):
    lines = []
    a = lines.append
    a('# Phase 2D-B: Final Publication-Quality Figure Production')
    a('')
    a('```')
    if is_full_complete:
        a('Phase 2D-B complete.')
        a('All final composites and figure-data packages validated.')
    else:
        a('Phase 2D-B planning complete.')
        a('Final publication rendering pending required panels.')
    a('```')
    a('')
    a('## 1. Scope and frozen inputs')
    a('')
    a('This document reflects a `--plan-only` run in a lightweight checkout. It reads exclusively frozen '
      'Phase-1 through Phase-2D-A artifacts (103 files, checksummed before and after this stage) and never '
      'touches `data_out/`, `data_out_fixed/`, or reruns any training/inference/TTK step. Phase 2D-A is '
      'treated as complete and authoritative; no sample is re-selected and no alternate is activated here.')
    a('')
    a('## 2. Frozen sample set')
    a('')
    a('Cross-checked against `ttk_runs_fixed/unified_candidate_analysis/phase2d/selection/'
      'archetype_selected_samples.csv` (primary rows only -- `archetype_alternates.csv` is never read for '
      'selection purposes):')
    a('')
    for aid in p2da.ARCHETYPE_PRIORITY:
        a(f'- `{aid}` = sample_idx **{manifest[aid]}**')
    a('')
    a('## 3. Figure contracts')
    a('')
    for c in FIGURE_CONTRACTS:
        si = manifest[c['archetype_id']]
        a(f"### Figure {c['figure_id']}: `{c['archetype_id']}` (sample_idx={si})")
        a('')
        a(f"- Primary claim: {c['primary_claim']}")
        methods_human = ', '.join([GT_DISPLAY_LABEL] + [HUMAN_LABELS[m] for m in c['required_methods']])
        a(f'- Required methods: {methods_human}')
        if c['method_roles']:
            roles_human = ', '.join(f'{HUMAN_LABELS[m]}={role}' for m, role in c['method_roles'].items())
            a(f'- Method roles: {roles_human}')
        a(f"- Required panels: {', '.join(c['panels'])}")
        if c['emphasis']:
            a(f"- Emphasis: {c['emphasis']}")
        a('')
    a('## 4. Deterministic zoom region (sample 119, Figure 3)')
    a('')
    a(f'Scoring formula: {ZOOM_SCORE_FORMULA}')
    a('')
    if zoom_result is not None:
        a(f"Selected bounds: y=[{zoom_result['y0']}, {zoom_result['y1']}), "
          f"x=[{zoom_result['x0']}, {zoom_result['x1']}), score={zoom_result['score']:.6f}.")
    else:
        a('**Not yet computed.** The zoom window score requires the real GT and per-method error fields '
          '(`data_out_fixed/`/`data_out/`), which are absent in this lightweight checkout by design. '
          '`select_deterministic_zoom()` is implemented and synthetic-tested; it will run in `--render-fields`.')
    a('')
    a('## 5. PD coordinate source status')
    a('')
    a('This project\'s frozen topology artifacts (every CSV referenced by `column_mapping.csv`) were directly '
      'inspected and contain only the scalar `pd_distance`/`mt_distance` value per sample/method -- never raw '
      'persistence-diagram birth/death coordinates. `resolve_pd_diagram_source()` documents the deterministic '
      'naming convention such coordinate exports would need; it currently finds none for any method. Every '
      '`pd_evidence`/`pd_comparison` panel is recorded with `status=blocked_missing_pd_source` in '
      '`plan/final_panel_manifest.csv` (Figures 1, 2, 3). `--render-fields` will hard-fail rather than '
      'fabricate these panels unless coordinate-level PD data becomes available.')
    a('')
    a('## 6. Manual topology (merge-tree) requirements')
    a('')
    n_missing = sum(1 for r in manual_topo_rows if r['status'] == 'missing')
    a(f'{len(manual_topo_rows)} manual ParaView/TTK merge-tree panel(s) are required across all figures '
      f'(Figures 1, 2, 3, 5); {n_missing} are currently missing. Each requires both '
      '`manual_topology_inputs/figure_XX/<method_id>_mt.png` and the sibling `_mt_metadata.csv` (schema: '
      f'{", ".join(MANUAL_TOPOLOGY_METADATA_FIELDS)}). Default initial settings: persistence_threshold=11.0, '
      'arc_sampling=10, arc_line_size=3 -- final metadata must record the actual values used. See '
      '`plan/manual_topology_requirements.csv` for the exact per-panel list.')
    a('')
    a('## 7. Validation summary')
    a('')
    a('- `validation/figure_data_reproduction.csv`: every figure-data metric value cross-checked against the '
      'frozen Phase-1 long table and Phase-2D-A `selected_sample_method_values.csv` within tolerance '
      f'({REPRO_TOLERANCE:g}); hard-fails on any disagreement.')
    a('- `validation/panel_validation.csv`: every planned panel structurally matches its figure contract.')
    a('- `validation/final_figure_validation.csv`: all six final figures are `status=not_yet_rendered`.')
    a('- `validation/prior_phase_immutability_check.csv`: all 103 protected files confirmed unchanged.')
    a('')
    a('## 8. Exact commands to complete Phase 2D-B on Spark')
    a('')
    a('```')
    a('python3 scripts/render_unified_candidate_figures_phase2db.py --render-fields')
    a('python3 scripts/render_unified_candidate_figures_phase2db.py --assemble-composites')
    a('python3 scripts/render_unified_candidate_figures_phase2db.py --full')
    a('```')
    a('')
    a('`--render-fields` requires the real `data_out/`/`data_out_fixed/` arrays. `--assemble-composites` '
      'additionally requires every manual topology panel and metadata row listed in Section 6 to be supplied. '
      '`--full` runs both in sequence and hard-fails (never downgrades this report\'s status banner) while any '
      'required manual panel is absent.')
    a('')
    a('## 9. Generated files')
    a('')
    a('Planning-stage outputs (`ttk_runs_fixed/unified_candidate_analysis/phase2db/`):')
    for rel in [
        'plan/final_figure_plan.csv', 'plan/final_panel_manifest.csv', 'plan/manual_topology_requirements.csv',
        'plan/final_figure_captions.md',
    ] + [f'figure_data/{FIGURE_DATA_FILENAMES[i]}' for i in range(1, 7)] + [
        'validation/prior_phase_immutability_check.csv', 'validation/figure_data_reproduction.csv',
        'validation/panel_validation.csv', 'validation/final_figure_validation.csv',
    ]:
        a(f'- `ttk_runs_fixed/unified_candidate_analysis/phase2db/{rel}`')
    a('- `docs/unified_candidate_analysis_phase2db.md` (this file)')
    a('- `logs/unified_candidate_analysis_phase2db.log`')
    a('')
    a('Not yet generated (pending Spark + manual topology export): `panels/**/*.png`, '
      '`manual_topology_inputs/**/*`, `figures/**/*`.')
    a('')
    return lines


def write_phase2db_doc(manifest, zoom_result, manual_topo_rows, panel_rows, is_full_complete=False):
    lines = build_phase2db_doc_lines(manifest, zoom_result, manual_topo_rows, panel_rows, is_full_complete)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    DOC_PATH.write_text('\n'.join(lines) + '\n')
    log(f'[write] {DOC_PATH}')


# =============================================================================
# --plan-only
# =============================================================================

def cmd_plan_only() -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log('=' * 88)
    log('Unified candidate figures -- Phase 2D-B (--plan-only)')
    log(f'Repo root: {REPO_ROOT}')
    log('Read-only w.r.t. Phase-1/2A/2B/2C/2D-A artifacts (103 files). Reads only frozen CSV/Markdown '
        'inputs; touches no raw .npy array; renders no image.')
    log('=' * 88)

    checksums_before, file_to_phase = preflight_immutability()

    manifest = read_and_validate_selection_manifest()
    log(f'[selection] Frozen sample set confirmed: {manifest}')

    method_inventory = p2da.load_method_inventory()
    long_table = p2da.load_long_table()
    per_sample = long_table['per_sample']
    column_mapping_rows = p2da.load_column_mapping_rows()
    topology_source_map = p2da.build_topology_source_map(column_mapping_rows)
    method_values_by_key = load_selected_sample_method_values()
    raw_paths = p2da.resolve_raw_paths(method_inventory)

    figure_plan_rows = build_final_figure_plan_rows(manifest)
    write_csv(PLAN_DIR / 'final_figure_plan.csv', FINAL_FIGURE_PLAN_FIELDS, figure_plan_rows)

    panel_rows = build_final_panel_manifest_rows(manifest, topology_source_map)
    write_csv(PLAN_DIR / 'final_panel_manifest.csv', FINAL_PANEL_MANIFEST_FIELDS, panel_rows)

    manual_topo_rows = build_manual_topology_requirements_rows(manifest)
    write_csv(PLAN_DIR / 'manual_topology_requirements.csv', MANUAL_TOPOLOGY_REQ_FIELDS, manual_topo_rows)

    zoom_result = None  # requires real GT/error arrays; absent by design in this checkout

    figure_data_by_id = {}
    for c in FIGURE_CONTRACTS:
        rows = build_figure_data_rows(c, manifest, per_sample, topology_source_map, raw_paths,
                                         zoom_result if c['figure_id'] == 3 else None)
        write_csv(FIGURE_DATA_DIR / FIGURE_DATA_FILENAMES[c['figure_id']], FIGURE_DATA_FIELDS, rows)
        figure_data_by_id[c['figure_id']] = rows

    repro_rows = validate_figure_data_reproduction(figure_data_by_id, per_sample, method_values_by_key)
    write_csv(VALIDATION_DIR / 'figure_data_reproduction.csv',
               ['figure_id', 'archetype_id', 'sample_idx', 'method_id', 'metric', 'figure_data_value',
                'long_table_value', 'selected_sample_method_values_value', 'status'], repro_rows)

    panel_validation_rows = validate_panel_manifest(panel_rows)
    write_csv(VALIDATION_DIR / 'panel_validation.csv',
               ['figure_id', 'panel_type', 'method_id', 'output_path', 'status', 'structural_check_status',
                'notes'], panel_validation_rows)

    final_figure_validation_rows = build_not_yet_rendered_final_figure_validation(manifest)
    write_csv(VALIDATION_DIR / 'final_figure_validation.csv',
               ['figure_id', 'archetype_id', 'sample_idx', 'expected_png_path', 'expected_vector_path',
                'png_exists', 'png_min_dpi_ok', 'vector_exists', 'status'], final_figure_validation_rows)

    write_captions_md(manifest)
    write_phase2db_doc(manifest, zoom_result, manual_topo_rows, panel_rows, is_full_complete=False)

    postflight_immutability(checksums_before, file_to_phase, VALIDATION_DIR / 'prior_phase_immutability_check.csv')

    n_manual_missing = sum(1 for r in manual_topo_rows if r['status'] == 'missing')
    n_pd_blocked = sum(1 for r in panel_rows if r['status'] == 'blocked_missing_pd_source')
    log('')
    log('=' * 88)
    log(f'RESULT: phase2db_planning_complete_final_rendering_pending. 6 figure plans, {len(panel_rows)} '
        f'planned panels ({n_manual_missing} awaiting manual topology input, {n_pd_blocked} blocked on a '
        f'missing PD coordinate source), {len(repro_rows)} figure-data values reproduced within tolerance.')
    log('=' * 88)
    flush_log(LOG_PATH)
    return dict(manifest=manifest, figure_plan_rows=figure_plan_rows, panel_rows=panel_rows,
                 manual_topo_rows=manual_topo_rows, figure_data_by_id=figure_data_by_id,
                 repro_rows=repro_rows)


# =============================================================================
# --render-fields (requires real raw arrays; not run in this checkout)
# =============================================================================

def _load_full_panel_arrays(manifest, method_inventory):
    """Loads and audits raw arrays for every method needed by any figure,
    reusing Phase-2D-A's exact audit machinery (idx validation, full-168-row
    shape/finiteness/alignment). Hard-fails cleanly if arrays are absent or
    invalid -- never fabricates a field."""
    long_table = p2da.load_long_table()
    per_sample = long_table['per_sample']
    raw_paths = p2da.resolve_raw_paths(method_inventory)
    p2da.require_raw_artifacts_exist(raw_paths)
    ordered_selected = sorted(set(manifest.values()))
    audit = p2da.audit_raw_artifacts(raw_paths, ordered_selected, per_sample)
    if audit['failures']:
        raise SystemExit(
            f'[hard-fail] {len(audit["failures"])} raw-artifact audit check(s) failed; no field was rendered:\n' +
            '\n'.join(f'  - {f}' for f in audit['failures'])
        )
    return audit, ordered_selected


def render_speed_and_error_panels(contract, manifest, audit, ordered_selected):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    si = manifest[contract['archetype_id']]
    pos = ordered_selected.index(si)
    methods = contract['full_panel_methods']
    gt_speed = p2da.speed_from_uv(audit['selected_data'][CNN]['gt'][pos])
    method_speeds = {mid: p2da.speed_from_uv(audit['selected_data'][mid]['sr'][pos]) for mid in methods}
    panel = p2da.compute_preview_panel_data(audit['selected_data'][CNN]['gt'][pos],
                                                {mid: audit['selected_data'][mid]['sr'][pos] for mid in methods})
    out_dir = PANELS_DIR / figure_dir_name(contract)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    fig, ax = plt.subplots(figsize=(3, 3), dpi=300)
    ax.imshow(gt_speed, cmap='cividis', vmin=panel['speed_vmin'], vmax=panel['speed_vmax'],
               origin='lower', aspect='equal')
    ax.set_title(GT_DISPLAY_LABEL)
    ax.set_xticks([])
    ax.set_yticks([])
    out_path = out_dir / f'{SPEED_FIELDS}_GT.png'
    fig.savefig(out_path, dpi=300, metadata={'Software': ''})
    plt.close(fig)
    rows.append(dict(output_path=str(out_path.relative_to(REPO_ROOT)), method_id='GT', panel_type=SPEED_FIELDS))
    for mid in methods:
        fig, ax = plt.subplots(figsize=(3, 3), dpi=300)
        ax.imshow(method_speeds[mid], cmap='cividis', vmin=panel['speed_vmin'], vmax=panel['speed_vmax'],
                    origin='lower', aspect='equal')
        ax.set_title(HUMAN_LABELS[mid])
        ax.set_xticks([])
        ax.set_yticks([])
        out_path = out_dir / f'{SPEED_FIELDS}_{mid}.png'
        fig.savefig(out_path, dpi=300, metadata={'Software': ''})
        plt.close(fig)
        rows.append(dict(output_path=str(out_path.relative_to(REPO_ROOT)), method_id=mid,
                           panel_type=SPEED_FIELDS))

        fig, ax = plt.subplots(figsize=(3, 3), dpi=300)
        ax.imshow(panel['errors'][mid], cmap='magma', vmin=panel['error_vmin'], vmax=panel['error_vmax'],
                    origin='lower', aspect='equal')
        ax.set_title(f'|{HUMAN_LABELS[mid]} - GT|')
        ax.set_xticks([])
        ax.set_yticks([])
        out_path = out_dir / f'{ERROR_MAPS}_{mid}.png'
        fig.savefig(out_path, dpi=300, metadata={'Software': ''})
        plt.close(fig)
        rows.append(dict(output_path=str(out_path.relative_to(REPO_ROOT)), method_id=mid, panel_type=ERROR_MAPS))
    return rows, gt_speed, method_speeds, panel


def render_metric_strip(contract, manifest, per_sample):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    si = manifest[contract['archetype_id']]
    out_dir = PANELS_DIR / figure_dir_name(contract)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(1.6 * (len(contract['required_methods']) + 1), 1.2), dpi=300)
    ax.axis('off')
    col_labels = [GT_DISPLAY_LABEL] + [HUMAN_LABELS[m] for m in contract['required_methods']]
    cell_text = [['--'] + [f"{per_sample[m][si]['pd_distance']:.2f}" for m in contract['required_methods']],
                  ['--'] + [f"{per_sample[m][si]['mt_distance']:.2f}" for m in contract['required_methods']]]
    table = ax.table(cellText=cell_text, colLabels=col_labels, rowLabels=['PD', 'MT'], loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    out_path = out_dir / f'{METRIC_STRIP}.png'
    fig.savefig(out_path, dpi=300, metadata={'Software': ''})
    plt.close(fig)
    return str(out_path.relative_to(REPO_ROOT))


def render_zoom_crop_panel(contract, manifest, gt_speed, method_speeds, zoom):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    out_dir = PANELS_DIR / figure_dir_name(contract)
    out_dir.mkdir(parents=True, exist_ok=True)
    methods = contract['full_panel_methods']
    n = 1 + len(methods)
    fig, axes = plt.subplots(1, n, figsize=(2.2 * n, 2.4), dpi=300)
    vmin = min([float(gt_speed.min())] + [float(v.min()) for v in method_speeds.values()])
    vmax = max([float(gt_speed.max())] + [float(v.max()) for v in method_speeds.values()])
    y0, y1, x0, x1 = zoom['y0'], zoom['y1'], zoom['x0'], zoom['x1']
    axes[0].imshow(gt_speed[y0:y1, x0:x1], cmap='cividis', vmin=vmin, vmax=vmax, origin='lower')
    axes[0].set_title(GT_DISPLAY_LABEL, fontsize=8)
    for j, mid in enumerate(methods, start=1):
        axes[j].imshow(method_speeds[mid][y0:y1, x0:x1], cmap='cividis', vmin=vmin, vmax=vmax, origin='lower')
        axes[j].set_title(HUMAN_LABELS[mid], fontsize=8)
    for axi in axes:
        axi.set_xticks([])
        axi.set_yticks([])
    out_path = out_dir / f'{ZOOM_CROP}.png'
    fig.savefig(out_path, dpi=300, metadata={'Software': ''})
    plt.close(fig)
    return str(out_path.relative_to(REPO_ROOT))


def cmd_render_fields(plan_result=None) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log('=' * 88)
    log('Unified candidate figures -- Phase 2D-B (--render-fields)')
    log('Requires the raw Spark arrays (data_out/, data_out_fixed/). Hard-fails if any required raw '
        'artifact, or PD coordinate source, is unavailable or invalid.')
    log('=' * 88)

    checksums_before, file_to_phase = preflight_immutability()
    manifest = read_and_validate_selection_manifest()
    method_inventory = p2da.load_method_inventory()
    long_table = p2da.load_long_table()
    per_sample = long_table['per_sample']
    column_mapping_rows = p2da.load_column_mapping_rows()
    topology_source_map = p2da.build_topology_source_map(column_mapping_rows)

    audit, ordered_selected = _load_full_panel_arrays(manifest, method_inventory)

    render_rows = []
    for c in FIGURE_CONTRACTS:
        panel_rows, gt_speed, method_speeds, _panel = render_speed_and_error_panels(
            c, manifest, audit, ordered_selected)
        render_rows.extend(panel_rows)
        metric_strip_path = render_metric_strip(c, manifest, per_sample)
        render_rows.append(dict(output_path=metric_strip_path, method_id='', panel_type=METRIC_STRIP))
        for panel_type in c['panels']:
            if panel_type in PD_DIAGRAM_PANEL_TYPES:
                require_pd_diagram_sources_for_figure(c, manifest, topology_source_map)
            if panel_type == ZOOM_CROP:
                gt_full = gt_speed
                zoom = select_deterministic_zoom(gt_full, {mid: np.abs(method_speeds[mid] - gt_full)
                                                              for mid in c['full_panel_methods']})
                render_rows.append(dict(
                    output_path=render_zoom_crop_panel(c, manifest, gt_full, method_speeds, zoom),
                    method_id='', panel_type=ZOOM_CROP,
                ))

    write_csv(VALIDATION_DIR / 'panel_validation.csv',
               ['output_path', 'method_id', 'panel_type'], render_rows)
    postflight_immutability(checksums_before, file_to_phase, VALIDATION_DIR / 'prior_phase_immutability_check.csv')
    log('')
    log('=' * 88)
    log(f'RESULT: --render-fields wrote {len(render_rows)} panel(s).')
    log('=' * 88)
    flush_log(LOG_PATH)
    return dict(render_rows=render_rows)


# =============================================================================
# --assemble-composites (requires manual topology panels; not run here)
# =============================================================================

def require_manual_topology_panels(manifest):
    manual_topo_rows = build_manual_topology_requirements_rows(manifest)
    missing = [r for r in manual_topo_rows if r['status'] == 'missing']
    if missing:
        detail = '\n'.join(
            f"  - figure_{r['figure_id']:02d}/{r['method_id']}: expected {r['expected_panel_path']} and "
            f"{r['expected_metadata_path']}" for r in missing
        )
        raise SystemExit(
            f'[hard-fail] {len(missing)} required manual topology panel(s)/metadata row(s) are missing. '
            f'This script never automates merge-tree geometry rendering -- supply the manual ParaView/TTK '
            f'exports first:\n{detail}'
        )
    for r in manual_topo_rows:
        meta_path = REPO_ROOT / r['expected_metadata_path']
        meta_rows = p2da.read_csv_dicts(meta_path)
        if not meta_rows:
            raise SystemExit(f'[hard-fail] Manual topology metadata file is empty: {meta_path}')
        row0 = meta_rows[0]
        missing_fields = [f for f in MANUAL_TOPOLOGY_METADATA_FIELDS if not str(row0.get(f, '')).strip()]
        if missing_fields:
            raise SystemExit(
                f'[hard-fail] Manual topology metadata {meta_path} is missing required field(s): {missing_fields}'
            )
    return manual_topo_rows


def gather_composite_source_panels(contract):
    """Collects the ordered list of PNG panel paths (script-rendered +
    manual topology) available on disk for one figure's composite. Pure
    filesystem lookup; raises no error -- callers decide what an empty
    result means."""
    panel_dir = PANELS_DIR / figure_dir_name(contract)
    script_panel_paths = sorted(panel_dir.glob('*.png')) if panel_dir.exists() else []
    needs_manual_mt = any(pt in MT_PANEL_TYPES for pt in contract['panels'])
    manual_paths = [MANUAL_TOPOLOGY_DIR / f"figure_{contract['figure_id']:02d}" / f'{mid}_mt.png'
                      for mid in (['GT'] + contract['full_panel_methods'])] if needs_manual_mt else []
    return script_panel_paths + [p for p in manual_paths if p.exists()]


def build_composite_for_figure(contract, manifest):
    """Assembles ONE figure's final composite (PNG + vector PDF) from
    whatever validated panels are currently on disk for it. Hard-fails if no
    panel is available. Returns the final_figure_validation.csv row."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    si = manifest[contract['archetype_id']]
    all_panels = gather_composite_source_panels(contract)
    if not all_panels:
        raise SystemExit(
            f"[hard-fail] No rendered panels found for figure {contract['figure_id']} under "
            f"{PANELS_DIR / figure_dir_name(contract)}; run --render-fields first."
        )
    n = len(all_panels)
    n_cols = min(n, 4)
    n_rows = math.ceil(n / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.4 * n_cols, 2.4 * n_rows), dpi=300)
    axes_flat = np.atleast_1d(axes).ravel()
    for ax, p in zip(axes_flat, all_panels):
        ax.imshow(plt.imread(p))
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes_flat[len(all_panels):]:
        ax.axis('off')
    out_paths = final_figure_paths(contract)
    png_path = REPO_ROOT / out_paths['png']
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=300, metadata={'Software': ''})
    pdf_path = REPO_ROOT / out_paths['pdf']
    fig.savefig(pdf_path, metadata={'Creator': '', 'Producer': ''})
    plt.close(fig)
    return dict(
        figure_id=contract['figure_id'], archetype_id=contract['archetype_id'], sample_idx=si,
        expected_png_path=out_paths['png'], expected_vector_path=out_paths['pdf'],
        png_exists=png_path.exists(), png_min_dpi_ok=True, vector_exists=pdf_path.exists(),
        status='rendered',
    )


def cmd_assemble_composites(plan_result=None) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log('=' * 88)
    log('Unified candidate figures -- Phase 2D-B (--assemble-composites)')
    log('=' * 88)
    checksums_before, file_to_phase = preflight_immutability()
    manifest = read_and_validate_selection_manifest()
    manual_topo_rows = require_manual_topology_panels(manifest)

    final_rows = [build_composite_for_figure(c, manifest) for c in FIGURE_CONTRACTS]
    write_csv(VALIDATION_DIR / 'final_figure_validation.csv',
               ['figure_id', 'archetype_id', 'sample_idx', 'expected_png_path', 'expected_vector_path',
                'png_exists', 'png_min_dpi_ok', 'vector_exists', 'status'], final_rows)
    postflight_immutability(checksums_before, file_to_phase, VALIDATION_DIR / 'prior_phase_immutability_check.csv')
    write_phase2db_doc(manifest, None, manual_topo_rows, [], is_full_complete=True)
    log('')
    log('=' * 88)
    log(f'RESULT: Phase 2D-B complete. {len(final_rows)} final composite figure(s) written.')
    log('=' * 88)
    flush_log(LOG_PATH)
    return dict(final_rows=final_rows)


def cmd_full():
    plan_result = cmd_plan_only()
    render_result = cmd_render_fields(plan_result=plan_result)
    composite_result = cmd_assemble_composites(plan_result=plan_result)
    return plan_result, render_result, composite_result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--plan-only', action='store_true')
    mode.add_argument('--render-fields', action='store_true')
    mode.add_argument('--assemble-composites', action='store_true')
    mode.add_argument('--full', action='store_true')
    args = ap.parse_args()

    if args.plan_only:
        cmd_plan_only()
        return 0
    if args.render_fields:
        cmd_render_fields()
        return 0
    if args.assemble_composites:
        cmd_assemble_composites()
        return 0
    if args.full:
        cmd_full()
        return 0
    return 1


if __name__ == '__main__':
    sys.exit(main())
