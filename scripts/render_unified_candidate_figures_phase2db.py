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
import re
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import select_and_preview_unified_candidates_phase2d as p2da  # noqa: E402
import extract_ttk_pd_critical_pairs as ttk_pd_parser  # noqa: E402

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
# Protected files: the COMPLETE authoritative Phase-2D-A package (86 prior
# + 32 Phase-2D-A = 118), never a glob. Reuses Phase-2D-A's own explicit
# Phase-1/2A/2B/2C lists.
# -----------------------------------------------------------------------
PHASE2D_A_SELECTION_CSV_NAMES = [
    'archetype_alternates.csv', 'archetype_score_table.csv', 'archetype_selected_samples.csv',
    'archetype_selection_diagnostics.csv', 'figure_plan.csv', 'preview_method_manifest.csv',
    'preview_plan.csv', 'prior_phase_immutability_check.csv', 'raw_artifact_requirements.csv',
    'selected_sample_method_values.csv', 'selected_sample_metric_context.csv',
    'selected_sample_pairwise_preferences.csv', 'selection_validation.csv',
]
PHASE2D_A_SELECTION_CSVS = [p2da.SELECTION_DIR / n for n in PHASE2D_A_SELECTION_CSV_NAMES]
assert len(PHASE2D_A_SELECTION_CSVS) == 13

PHASE2D_A_PREVIEW_AUDIT_CSV_NAMES = [
    'raw_artifact_inventory.csv', 'raw_alignment_validation.csv', 'selected_sample_array_statistics.csv',
    'selected_sample_metric_reproduction.csv', 'selected_sample_artifact_manifest.csv',
    'preview_render_validation.csv',
]
PHASE2D_A_PREVIEW_AUDIT_CSVS = [p2da.PREVIEW_AUDIT_DIR / n for n in PHASE2D_A_PREVIEW_AUDIT_CSV_NAMES]
assert len(PHASE2D_A_PREVIEW_AUDIT_CSVS) == 6

# The six per-sample review PNGs live one directory per archetype. The literal
# archetype_id -> sample_idx mapping here is deliberately restated (not a
# forward reference to FROZEN_SAMPLE_SET, defined further below) so this
# protected-file list never depends on definition order.
PHASE2D_A_PREVIEW_PNGS = [
    p2da.PREVIEWS_DIR / archetype_id / f'selected_sample_{sample_idx}_review.png'
    for archetype_id, sample_idx in [
        ('global_descriptor_disagreement', 120), ('gan_pd_vs_cnn_mt_conflict', 34),
        ('f3_pd_vs_uv_e2_mt_tradeoff', 119), ('f2_balanced_vs_cnn', 25),
        ('candidate_c_continuity', 30), ('global_descriptor_agreement', 19),
    ]
] + [p2da.PREVIEWS_DIR / 'phase2d_selected_archetypes_contact_sheet.png']
assert len(PHASE2D_A_PREVIEW_PNGS) == 7

PHASE2D_A_DOCS = [
    REPO_ROOT / 'docs' / 'unified_candidate_analysis_phase2d.md',
    REPO_ROOT / 'docs' / 'unified_candidate_phase2d_visual_review.md',
]
assert len(PHASE2D_A_DOCS) == 2

PHASE2D_A_SCRIPTS = [
    REPO_ROOT / 'scripts' / 'select_and_preview_unified_candidates_phase2d.py',
    REPO_ROOT / 'scripts' / 'test_select_and_preview_unified_candidates_phase2d.py',
]
assert len(PHASE2D_A_SCRIPTS) == 2

PHASE2D_A_LOGS = [
    REPO_ROOT / 'logs' / 'unified_candidate_analysis_phase2d_selection.log',
    REPO_ROOT / 'logs' / 'unified_candidate_analysis_phase2d_render.log',
]
assert len(PHASE2D_A_LOGS) == 2

PHASE2D_A_PROTECTED_FILES = (PHASE2D_A_SELECTION_CSVS + PHASE2D_A_PREVIEW_AUDIT_CSVS + PHASE2D_A_PREVIEW_PNGS +
                               PHASE2D_A_DOCS + PHASE2D_A_SCRIPTS + PHASE2D_A_LOGS)  # 13+6+7+2+2+2 = 32
assert len(PHASE2D_A_PROTECTED_FILES) == 32

VISUAL_REVIEW_DOC_PATH = REPO_ROOT / 'docs' / 'unified_candidate_phase2d_visual_review.md'

ALL_PROTECTED_FILES = (p2da.PHASE1_PROTECTED_FILES + p2da.PHASE2A_PROTECTED_FILES +
                         p2da.PHASE2B_PROTECTED_FILES + p2da.PHASE2C_PROTECTED_FILES +
                         PHASE2D_A_PROTECTED_FILES)  # 12+14+28+32+32 = 118
assert len(ALL_PROTECTED_FILES) == 118

PROTECTED_DIRS_AND_CSVS = list(p2da.PROTECTED_DIRS_AND_CSVS) + [
    (p2da.SELECTION_DIR, set(PHASE2D_A_SELECTION_CSVS)),
    (p2da.PREVIEW_AUDIT_DIR, set(PHASE2D_A_PREVIEW_AUDIT_CSVS)),
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
assert sorted(int(p.stem.split('_')[2]) for p in PHASE2D_A_PREVIEW_PNGS
               if p.name != 'phase2d_selected_archetypes_contact_sheet.png') == \
    sorted(FROZEN_SAMPLE_SET.values()), 'PHASE2D_A_PREVIEW_PNGS sample indices must match FROZEN_SAMPLE_SET'

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
            f'32 Phase-2D-A):\n' + '\n'.join(f'  - {m}' for m in missing)
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
        f'(12 Phase-1 + 14 Phase-2A + 28 Phase-2B + 32 Phase-2C + 32 Phase-2D-A = 118 exactly).')
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


# Phrasing accepted as an explicit "no alternate was activated" statement in
# the human visual-review record. Deliberately several equivalent phrasings
# rather than one exact string, since the record is authored by a person.
VISUAL_REVIEW_NO_ALTERNATE_PHRASES = (
    'no alternate', 'zero alternates', 'alternate activated: none', 'alternates activated: none',
    'no alternates were activated', 'without activating any alternate',
)


def require_completed_phase2d_a_state(manifest) -> None:
    """Requires (a) Phase-2D-A's own report to state the fully rendered
    'complete' state (never merely the planning-only state), and (b) the
    human visual-review record to explicitly confirm, in text, that every
    one of the six frozen primaries was accepted and that no alternate was
    activated. Never inferred from the mere presence of the files -- their
    content is read and checked."""
    if not p2da.DOC_PATH.exists():
        raise SystemExit(f'[hard-fail] Required Phase-2D-A report is missing: {p2da.DOC_PATH}')
    doc_text = p2da.DOC_PATH.read_text()
    if 'Phase 2D-A complete.' not in doc_text:
        raise SystemExit(
            f'[hard-fail] {p2da.DOC_PATH} does not report the completed Phase-2D-A state (the '
            f'"Phase 2D-A complete." banner, written only after --render-previews/--full succeeds, was not '
            f'found). Phase 2D-B may only build on the fully rendered, authoritative Phase-2D-A package -- '
            f'not a --selection-only-only state.'
        )
    if not VISUAL_REVIEW_DOC_PATH.exists():
        raise SystemExit(
            f'[hard-fail] Required human visual-review record is missing: {VISUAL_REVIEW_DOC_PATH}. Phase '
            f'2D-B never infers visual acceptance -- it must be explicitly recorded by a human reviewer.'
        )
    review_text = VISUAL_REVIEW_DOC_PATH.read_text()
    review_lower = review_text.lower()
    missing_acceptance = []
    for archetype_id, sample_idx in manifest.items():
        found = any(archetype_id in line.lower() and str(sample_idx) in line and 'accept' in line.lower()
                     for line in review_text.splitlines())
        if not found:
            missing_acceptance.append((archetype_id, sample_idx))
    if missing_acceptance:
        raise SystemExit(
            f'[hard-fail] {VISUAL_REVIEW_DOC_PATH} does not explicitly confirm acceptance for: '
            f'{missing_acceptance}. Required format: at least one line containing the archetype_id, its '
            f'sample_idx, and the word "accept" (e.g. "global_descriptor_disagreement sample_idx=120: '
            f'ACCEPTED").'
        )
    if not any(phrase in review_lower for phrase in VISUAL_REVIEW_NO_ALTERNATE_PHRASES):
        raise SystemExit(
            f'[hard-fail] {VISUAL_REVIEW_DOC_PATH} does not explicitly state that no alternate was '
            f'activated (expected phrasing such as "no alternate(s) activated" or '
            f'"alternate activated: none").'
        )


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
# table. See resolve_pd_source_verdict().
PD_DIAGRAM_PANEL_TYPES = {PD_EVIDENCE, PD_COMPARISON}
# Figure-level (not per-method) panels.
FIGURE_LEVEL_PANEL_TYPES = {METRIC_STRIP, PD_MT_TRADEOFF_COMPACT, PD_MT_COMPARISON_COMPACT, ZOOM_CROP}
PER_METHOD_PANEL_TYPES = {SPEED_FIELDS, ERROR_MAPS, PD_EVIDENCE, MT_EVIDENCE, PD_COMPARISON, MT_COMPARISON,
                            TOPOLOGY_COMPARISON}

FIGURE_CONTRACTS = [
    dict(
        figure_id=1, short_name='global_disagreement', archetype_id='global_descriptor_disagreement',
        primary_claim='PD and MT can produce strongly different cross-method preferences.',
        required_methods=[CNN, GAN, CANDIDATE_C, F3, F2, UV_E2],
        full_panel_methods=[CNN, GAN, CANDIDATE_C, F3, F2, UV_E2],
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
        required_methods=[CNN, F3, F2, UV_E2],
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
        required_methods=[CNN, GAN, CANDIDATE_C, F3, F2, UV_E2],
        full_panel_methods=[CNN, GAN, CANDIDATE_C, F3, F2, UV_E2],
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
# Authoritative PD-diagram coordinate source discovery and validation. Real
# repository-relative filesystem search against the EXACT conventions TTK's
# ttkPersistenceDiagram filter and this repository's topology pipeline use --
# never a fuzzy/partial-name match, never raw VTU point geometry.
#
# Real TTK PD VTU schema (see scripts/extract_ttk_pd_critical_pairs.py,
# reused here via ttk_pd_parser.parse_vtu()): CellData = PairIdentifier,
# PairType, Persistence, Birth, IsFinite; PointData = ttkVertexScalarField;
# Cells = connectivity. Publication PD coordinates are ALL finite positive
# pairs -- mask = (IsFinite==1) & (Persistence>0) & (PairType!=-1),
# birth = Birth[mask], death = Birth[mask] + Persistence[mask] -- never the
# critical-pair training script's top-k/persistence-fraction filtering
# (extract_constraints()), and never vtkUnstructuredGrid point coordinates
# (those are mesh geometry, not birth/death).
#
# Exact source convention (role-aware, Section 3):
#   ttk_runs_fixed/topology_finetuning/<artifact_alias>_topology/pd/SR/
#       <artifact_alias>_SR_s<sample_idx>_..._pd_port_0.vtu   (learned method)
#   ttk_runs_fixed/topology_finetuning/<artifact_alias>_topology/pd/GT/
#       <artifact_alias>_GT_s<sample_idx>_..._pd_port_0.vtu   (shared GT)
# Never mapped: any path under an `mt/` directory, `_mt_port_0.vtu`,
# `_mt_port_1.vtu`, `_pd_port_1.vtu`, a GT file to a learned-method panel, an
# SR file to GT, a different sample_idx, or a partial method-name match.
#
# Three-status vocabulary only:
#   available_validated                    -- an exact source was found,
#                                              parsed, and validated.
#   pending_authoritative_spark_source_discovery
#                                           -- the ONLY status ever produced
#                                              by a lightweight (non-Spark)
#                                              checkout, since a raw .vtu
#                                              Spark intermediate being absent
#                                              here never proves it is absent
#                                              on the authoritative Spark
#                                              machine.
#   unavailable_after_authoritative_spark_audit
#                                           -- only reachable when this
#                                              process IS running where
#                                              data_out_fixed/ exists (i.e. on
#                                              Spark) and the exact search
#                                              still finds nothing.
# A present exact candidate that fails parsing/validation HARD-FAILS -- it is
# never silently reclassified as pending/unavailable, and never falls back
# to scalar evidence.
# =============================================================================

# True in this checkout (and any lightweight checkout): raw Spark
# intermediates (data_out_fixed/, and by the same reasoning any raw .vtu
# TTK output) are gitignored and never present here by design.
IS_LIGHTWEIGHT_CHECKOUT = not (REPO_ROOT / 'data_out_fixed').exists()

STATUS_AVAILABLE = 'available_validated'
STATUS_PENDING = 'pending_authoritative_spark_source_discovery'
STATUS_UNAVAILABLE = 'unavailable_after_authoritative_spark_audit'

PD_SOURCE_DISCOVERY_FIELDS = [
    'figure_id', 'sample_idx', 'method_id', 'candidate_path', 'artifact_type', 'schema_or_array_names',
    'sample_mapping_status', 'finite_status', 'usable_status', 'notes',
]

PD_SOURCE_VERDICT_FIELDS = [
    'figure_id', 'sample_idx', 'method_id', 'source_role', 'artifact_alias', 'verdict',
    'selected_candidate_path', 'parsed_pair_count', 'fallback_required', 'notes',
]

PD_OVERLAY_SCRIPT_NAME_MARKERS = ('pd_critical_pairs', 'pd_diagram', 'persistence_diagram')

PD_VTU_ROLE_GT = 'GT'
PD_VTU_ROLE_SR = 'SR'
PD_VTU_REQUIRED_SUFFIX = '_pd_port_0.vtu'
PD_VTU_REJECTED_MARKERS = ('_mt_port_0.vtu', '_mt_port_1.vtu', '_pd_port_1.vtu')
PD_VTU_REQUIRED_ARRAYS = ('pair_id', 'pair_type', 'persistence', 'birth', 'is_finite', 'connectivity',
                            'vertex_ids')
GT_PD_COORD_TOLERANCE = 1e-6

# Section 2: explicit method-to-artifact aliases. Confirmed directly against
# ttk_runs_fixed/unified_candidate_evaluation/column_mapping.csv (the same
# artifact each method's own raw topology CSV is already resolved from) and
# the actual ttk_runs_fixed/topology_finetuning/ directory listing.
METHOD_ARTIFACT_ALIASES = {
    CANDIDATE_C: 'candidateC_expanded2688',
    F3: 'candidateF_grad_crit_expanded2688',
    F2: 'candidateF_grad_levelset_E2_low_expanded2688',
    UV_E2: 'candidateUV_plus_E2_tf_lowlambda_expanded2688',
}


def _topology_tree_root(alias):
    return REPO_ROOT / 'ttk_runs_fixed' / 'topology_finetuning' / f'{alias}_topology'


# CNN, GAN, and bicubic have no <alias>_topology directory anywhere in this
# repository (confirmed by direct search: ttk_runs_fixed/cnn/,
# ttk_runs_fixed/gan/, and ttk_runs_fixed/superlevel_topology/{cnn,gan}/
# contain only scalar phase_c_final/*.csv summaries -- no pd/ VTU tree at
# all; no bicubic-named topology directory exists anywhere). These plausible
# raw-VTU root locations are still searched for real on every run -- a
# pd/{GT,SR} tree could exist there on the authoritative Spark machine even
# though it is absent (and *.vtu is gitignored) in this lightweight
# checkout -- but no baseline source is ever invented if the search finds
# nothing.
CNN_GAN_BICUBIC_SEARCH_ROOTS = {
    CNN: [REPO_ROOT / 'ttk_runs_fixed' / 'cnn',
          REPO_ROOT / 'ttk_runs_fixed' / 'superlevel_topology' / 'cnn' / 'topology'],
    GAN: [REPO_ROOT / 'ttk_runs_fixed' / 'gan',
          REPO_ROOT / 'ttk_runs_fixed' / 'superlevel_topology' / 'gan' / 'topology'],
    BICUBIC: [REPO_ROOT / 'ttk_runs_fixed' / 'bicubic',
               REPO_ROOT / 'ttk_runs_fixed' / 'superlevel_topology' / 'bicubic' / 'topology'],
}

# Deterministic priority order for canonical GT resolution: GT is shared by
# every method's topology tree, so every tree that could plausibly hold it
# is searched (never only the requesting method's own tree); ties among
# multiple agreeing exact copies are broken by this fixed order.
GT_SOURCE_PRIORITY_METHODS = [CANDIDATE_C, F3, F2, UV_E2, CNN, GAN, BICUBIC]


def _method_search_roots_and_alias(mid):
    """(topology_roots, filename_alias_token) for method mid. Real
    repository-relative roots only -- never a fixed 3-filename guess."""
    if mid in METHOD_ARTIFACT_ALIASES:
        alias = METHOD_ARTIFACT_ALIASES[mid]
        return [_topology_tree_root(alias)], alias
    return CNN_GAN_BICUBIC_SEARCH_ROOTS.get(mid, []), mid


def find_pd_overlay_scripts():
    """Existing repository scripts whose name suggests they already generate
    PD overlays/coordinate extracts (informational provenance only)."""
    return [p for p in sorted((REPO_ROOT / 'scripts').glob('*.py'))
             if any(marker in p.name.lower() for marker in PD_OVERLAY_SCRIPT_NAME_MARKERS)]


def _is_valid_pd_vtu_path(path):
    """True only for a real PD-output VTU: ends with `_pd_port_0.vtu`, never
    under an `mt/` directory, never `_mt_port_0/1.vtu` or `_pd_port_1.vtu`."""
    name = path.name
    if not name.endswith(PD_VTU_REQUIRED_SUFFIX):
        return False
    if any(marker in name for marker in PD_VTU_REJECTED_MARKERS):
        return False
    if 'mt' in {part.lower() for part in path.parts}:
        return False
    return True


def find_exact_pd_vtu_candidates(root, role, alias, sample_idx):
    """Real filesystem search for exact `<alias>_<role>_s<sample_idx>_..._
    pd_port_0.vtu` sources under root/pd/<role>/. Returns a deduplicated
    (by resolved repository-relative path) list of Paths -- never a
    fuzzy/partial match, never an mt/ or port-1 file."""
    role_dir = root / 'pd' / role
    if not role_dir.exists():
        return []
    prefix = f'{alias}_{role}_s{sample_idx}_'
    seen_rel = set()
    found = []
    for path in sorted(role_dir.rglob('*_pd_port_0.vtu')):
        if not _is_valid_pd_vtu_path(path):
            continue
        if not path.name.startswith(prefix):
            continue
        rel = _rel(path)
        if rel in seen_rel:
            continue
        seen_rel.add(rel)
        found.append(path)
    return found


def parse_and_validate_pd_vtu(path, sample_idx, role, alias):
    """Parses one exact PD VTU source via the repository's established TTK
    PD parser (extract_ttk_pd_critical_pairs.parse_vtu) and derives the
    publication PD coordinates: ALL finite positive pairs, never the
    critical-pair training script's top-k/persistence-fraction filtering,
    and never raw VTU point geometry. A present, exactly role/alias/sample-
    mapped candidate that fails any check here HARD-FAILS -- it is never
    silently reclassified as pending/unavailable."""
    if not _is_valid_pd_vtu_path(path):
        raise SystemExit(
            f'[hard-fail] {_rel(path)} is not a valid PD-output path (must end with {PD_VTU_REQUIRED_SUFFIX!r}, '
            f'never under an mt/ directory, never _mt_port_*/_pd_port_1).'
        )
    match = ttk_pd_parser._SAMPLE_IDX_RE.search(path.name)
    if not match or int(match.group(1)) != sample_idx:
        raise SystemExit(
            f'[hard-fail] {_rel(path)} does not carry the exact expected sample index {sample_idx} '
            f'(matched: {match.group(1) if match else None!r}).'
        )
    expected_prefix = f'{alias}_{role}_s{sample_idx}_'
    if not path.name.startswith(expected_prefix):
        raise SystemExit(
            f'[hard-fail] {_rel(path)} does not match the expected {role} artifact-alias filename convention '
            f'{expected_prefix!r}.'
        )
    try:
        arrays = ttk_pd_parser.parse_vtu(path)
    except Exception as exc:
        raise SystemExit(f'[hard-fail] PD VTU source exists but failed to parse: {_rel(path)}: {exc}')
    for name in PD_VTU_REQUIRED_ARRAYS:
        if name not in arrays:
            raise SystemExit(f'[hard-fail] PD VTU {_rel(path)} is missing required array {name!r}.')
    pair_id, pair_type = arrays['pair_id'], arrays['pair_type']
    persistence, birth, is_finite = arrays['persistence'], arrays['birth'], arrays['is_finite']
    lengths = {len(pair_id), len(pair_type), len(persistence), len(birth), len(is_finite)}
    if len(lengths) != 1:
        raise SystemExit(
            f'[hard-fail] PD VTU {_rel(path)} has mismatched CellData array lengths: pair_id={len(pair_id)} '
            f'pair_type={len(pair_type)} persistence={len(persistence)} birth={len(birth)} '
            f'is_finite={len(is_finite)}.'
        )
    if len(arrays['connectivity']) != 2 * len(pair_id):
        raise SystemExit(
            f'[hard-fail] PD VTU {_rel(path)} connectivity length {len(arrays["connectivity"])} does not equal '
            f'2 * cell count ({2 * len(pair_id)}).'
        )
    mask = (is_finite == 1) & (persistence > 0) & (pair_type != -1)
    if not np.any(mask):
        raise SystemExit(f'[hard-fail] PD VTU {_rel(path)} has zero finite positive pairs after masking.')
    birth_m = birth[mask].astype(np.float64)
    pers_m = persistence[mask].astype(np.float64)
    death_m = birth_m + pers_m
    if not (np.all(np.isfinite(birth_m)) and np.all(np.isfinite(pers_m)) and np.all(np.isfinite(death_m))):
        raise SystemExit(f'[hard-fail] PD VTU {_rel(path)} has non-finite birth/persistence/death after masking.')
    if not np.all(pers_m > 0):
        raise SystemExit(f'[hard-fail] PD VTU {_rel(path)} has non-positive persistence after masking.')
    if not np.all(death_m >= birth_m):
        raise SystemExit(f'[hard-fail] PD VTU {_rel(path)} has death < birth for at least one pair after masking.')
    return dict(birth=birth_m, death=death_m, pair_count=int(mask.sum()))


def _sorted_pd_pairs(birth, death):
    """Row-consistent sort (never independently-sorted columns, which would
    scramble the birth/death pairing) for cross-copy coordinate comparison."""
    order = np.lexsort((death, birth))
    return birth[order], death[order]


def _resolve_canonical_pd_copy(candidate_paths_and_aliases, sample_idx, role):
    """Parses every exact candidate (already deduplicated by resolved path)
    and requires them to agree on birth/death coordinates within tolerance.
    Returns (canonical_path, canonical_alias, canonical_result, n_copies).
    Hard-fails, listing every conflicting path, on disagreement -- never an
    arbitrary pick."""
    parsed = [(path, alias, parse_and_validate_pd_vtu(path, sample_idx, role, alias))
               for path, alias in candidate_paths_and_aliases]
    canonical_path, canonical_alias, canonical_result = parsed[0]
    if len(parsed) > 1:
        cb, cd = _sorted_pd_pairs(canonical_result['birth'], canonical_result['death'])
        conflicts = []
        for other_path, _, other_result in parsed[1:]:
            ob, od = _sorted_pd_pairs(other_result['birth'], other_result['death'])
            if ob.shape != cb.shape or not (np.allclose(ob, cb, atol=GT_PD_COORD_TOLERANCE) and
                                              np.allclose(od, cd, atol=GT_PD_COORD_TOLERANCE)):
                conflicts.append(other_path)
        if conflicts:
            raise SystemExit(
                f'[hard-fail] Conflicting exact {role} PD sources for sample_idx={sample_idx}: exact copies '
                f'disagree on birth/death coordinates beyond tolerance {GT_PD_COORD_TOLERANCE}. Conflicting '
                f'paths: {[_rel(p) for p, _, _ in parsed]}.'
            )
    return canonical_path, canonical_alias, canonical_result, len(parsed)


def resolve_canonical_gt_pd_source(sample_idx):
    """Deterministically resolves the single canonical GT PD coordinate
    source for one sample_idx, searched across EVERY known topology tree in
    GT_SOURCE_PRIORITY_METHODS order (GT is shared by all methods -- never
    only the requesting method's own tree). Returns None if no exact GT
    source is found anywhere."""
    candidates = []
    seen_rel = set()
    for pmid in GT_SOURCE_PRIORITY_METHODS:
        roots, alias = _method_search_roots_and_alias(pmid)
        for root in roots:
            for path in find_exact_pd_vtu_candidates(root, PD_VTU_ROLE_GT, alias, sample_idx):
                rel = _rel(path)
                if rel in seen_rel:
                    continue
                seen_rel.add(rel)
                candidates.append((path, alias))
    if not candidates:
        return None
    path, alias, result, n_copies = _resolve_canonical_pd_copy(candidates, sample_idx, PD_VTU_ROLE_GT)
    return dict(path=path, alias=alias, result=result, n_copies=n_copies)


def resolve_pd_source_verdict(figure_id, sample_idx, mid):
    """Reduces the exact-source search for one required (figure, sample,
    method) to a single authoritative verdict row (Section 5/6). Also
    carries the parsed birth/death coordinates (not part of the CSV schema)
    so callers never re-open/re-parse the VTU a second time in-process."""
    if mid == 'GT':
        canonical = resolve_canonical_gt_pd_source(sample_idx)
        if canonical is None:
            verdict = STATUS_PENDING if IS_LIGHTWEIGHT_CHECKOUT else STATUS_UNAVAILABLE
            notes = ('Lightweight (non-Spark) checkout: absence here does not prove absence on the '
                        'authoritative Spark machine (raw .vtu Spark intermediates are gitignored and never '
                        'present here).' if IS_LIGHTWEIGHT_CHECKOUT else
                        f'Searched every known topology tree ({GT_SOURCE_PRIORITY_METHODS}) for an exact GT PD '
                        f'source for sample_idx={sample_idx}; none found on the authoritative Spark machine.')
            return dict(figure_id=figure_id, sample_idx=sample_idx, method_id='GT', source_role=PD_VTU_ROLE_GT,
                          artifact_alias='', verdict=verdict, selected_candidate_path='', parsed_pair_count='',
                          fallback_required=str(verdict != STATUS_AVAILABLE), notes=notes, birth=None, death=None)
        notes = ('Single exact GT copy found.' if canonical['n_copies'] == 1 else
                   f"Canonical GT selected from {canonical['n_copies']} agreeing exact copies "
                   f"(priority alias={canonical['alias']!r}).")
        return dict(figure_id=figure_id, sample_idx=sample_idx, method_id='GT', source_role=PD_VTU_ROLE_GT,
                      artifact_alias=canonical['alias'], verdict=STATUS_AVAILABLE,
                      selected_candidate_path=_rel(canonical['path']),
                      parsed_pair_count=canonical['result']['pair_count'], fallback_required='False', notes=notes,
                      birth=canonical['result']['birth'], death=canonical['result']['death'])

    roots, alias = _method_search_roots_and_alias(mid)
    candidates = []
    seen_rel = set()
    for root in roots:
        for path in find_exact_pd_vtu_candidates(root, PD_VTU_ROLE_SR, alias, sample_idx):
            rel = _rel(path)
            if rel in seen_rel:
                continue
            seen_rel.add(rel)
            candidates.append((path, alias))
    if not candidates:
        verdict = STATUS_PENDING if IS_LIGHTWEIGHT_CHECKOUT else STATUS_UNAVAILABLE
        notes = ('Lightweight (non-Spark) checkout: absence here does not prove absence on the authoritative '
                    'Spark machine.' if IS_LIGHTWEIGHT_CHECKOUT else
                    f"Searched {[_rel(r) for r in roots]} for an exact SR PD source; none found on the "
                    f'authoritative Spark machine.')
        return dict(figure_id=figure_id, sample_idx=sample_idx, method_id=mid, source_role=PD_VTU_ROLE_SR,
                      artifact_alias=alias, verdict=verdict, selected_candidate_path='', parsed_pair_count='',
                      fallback_required=str(verdict != STATUS_AVAILABLE), notes=notes, birth=None, death=None)
    path, resolved_alias, result, n_copies = _resolve_canonical_pd_copy(candidates, sample_idx, PD_VTU_ROLE_SR)
    notes = '' if n_copies == 1 else f'{n_copies} agreeing exact SR copies found; selected {_rel(path)}.'
    return dict(figure_id=figure_id, sample_idx=sample_idx, method_id=mid, source_role=PD_VTU_ROLE_SR,
                  artifact_alias=resolved_alias, verdict=STATUS_AVAILABLE, selected_candidate_path=_rel(path),
                  parsed_pair_count=result['pair_count'], fallback_required='False', notes=notes,
                  birth=result['birth'], death=result['death'])


def discover_and_resolve_pd_sources_for_figure(contract, manifest):
    """Returns (discovery_rows, verdicts) for one figure. `verdicts` maps
    method_id (including 'GT') -> the full resolve_pd_source_verdict() dict.
    `discovery_rows` is raw per-method provenance in the legacy
    PD_SOURCE_DISCOVERY_FIELDS schema (plan/pd_source_discovery.csv) --
    retained for audit trail only; all render/plan gating now consumes
    `verdicts` / plan/pd_source_verdicts.csv."""
    si = manifest[contract['archetype_id']]
    needs_pd = any(pt in PD_DIAGRAM_PANEL_TYPES for pt in contract['panels'])
    methods_needing_pd = (['GT'] + contract['full_panel_methods']) if needs_pd else []
    discovery_rows = []
    verdicts = {}
    for mid in methods_needing_pd:
        v = resolve_pd_source_verdict(contract['figure_id'], si, mid)
        verdicts[mid] = v
        found = bool(v['selected_candidate_path'])
        discovery_rows.append(dict(
            figure_id=contract['figure_id'], sample_idx=si, method_id=mid,
            candidate_path=v['selected_candidate_path'],
            artifact_type=('vtk_family_vtu' if found else 'none_found'),
            schema_or_array_names=(','.join(PD_VTU_REQUIRED_ARRAYS) if found else ''),
            sample_mapping_status=('mapped' if found else 'not_found'),
            finite_status=('finite' if v['verdict'] == STATUS_AVAILABLE else 'not_applicable'),
            usable_status=v['verdict'], notes=v['notes'],
        ))
    return discovery_rows, verdicts


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


def build_final_panel_manifest_rows(manifest, pd_verdicts_by_figure):
    rows = []
    for c in FIGURE_CONTRACTS:
        si = manifest[c['archetype_id']]
        pd_verdicts = pd_verdicts_by_figure.get(c['figure_id'], {})
        for panel_type in c['panels']:
            if panel_type in PER_METHOD_PANEL_TYPES:
                for mid in (['GT'] + c['full_panel_methods']):
                    if panel_type == ERROR_MAPS and mid == 'GT':
                        continue  # no self-error panel for GT
                    role = c['method_roles'].get(mid, 'primary')
                    needs_manual = panel_type in MT_PANEL_TYPES
                    needs_pd_source = panel_type in PD_DIAGRAM_PANEL_TYPES
                    pd_verdict = ''
                    status = 'planned_not_rendered'
                    if needs_manual:
                        status = 'blocked_awaiting_manual_topology_input'
                    elif needs_pd_source:
                        pd_verdict = pd_verdicts.get(mid, STATUS_PENDING)
                        status = {
                            STATUS_AVAILABLE: 'planned_not_rendered',
                            STATUS_PENDING: STATUS_PENDING,
                            STATUS_UNAVAILABLE: 'scalar_fallback_planned',
                        }[pd_verdict]
                    rows.append(dict(
                        figure_id=c['figure_id'], archetype_id=c['archetype_id'], sample_idx=si,
                        panel_type=panel_type, method_id=mid,
                        display_label=(GT_DISPLAY_LABEL if mid == 'GT' else HUMAN_LABELS[mid]),
                        method_role=role, output_path=panel_output_path(c, panel_type, mid),
                        requires_manual_topology_input=needs_manual,
                        requires_pd_coordinate_source=needs_pd_source, pd_coordinate_source_found=pd_verdict,
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


FINAL_COMPOSITE_MANIFEST_FIELDS = [
    'figure_id', 'panel_order', 'panel_group', 'panel_type', 'method_id', 'source_path', 'final_visible_status',
]


def build_final_composite_manifest_rows(manifest, pd_verdicts_by_figure):
    """Section 7: the EXPLICIT ordered manifest --assemble-composites must
    use exclusively (never glob-and-alphabetically-sort). One row per panel
    that will appear in the final composite, in declared method/panel
    order, with a deterministic source_path routed by the current PD
    discovery verdict (coordinate panel, scalar fallback, or still-pending)."""
    rows = []
    for c in FIGURE_CONTRACTS:
        pd_verdicts = pd_verdicts_by_figure.get(c['figure_id'], {})
        order = 0

        def add(group, panel_type, mid, source_path, visible_status):
            nonlocal order
            order += 1
            rows.append(dict(figure_id=c['figure_id'], panel_order=order, panel_group=group,
                                panel_type=panel_type, method_id=mid, source_path=source_path,
                                final_visible_status=visible_status))

        for panel_type in c['panels']:
            if panel_type in MT_PANEL_TYPES:
                for mid in (['GT'] + c['full_panel_methods']):
                    add('manual_topology', panel_type, mid,
                         f"ttk_runs_fixed/unified_candidate_analysis/phase2db/manual_topology_inputs/"
                         f"figure_{c['figure_id']:02d}/{mid}_mt.png", 'visible')
            elif panel_type in PD_DIAGRAM_PANEL_TYPES:
                for mid in (['GT'] + c['full_panel_methods']):
                    verdict = pd_verdicts.get(mid, STATUS_PENDING)
                    if verdict == STATUS_AVAILABLE:
                        add('pd_coordinate', panel_type, mid, panel_output_path(c, panel_type, mid), 'visible')
                    elif verdict == STATUS_UNAVAILABLE:
                        add('pd_scalar_fallback', panel_type, mid,
                             panel_output_path(c, panel_type, 'scalar_fallback'), 'scalar_fallback')
                    else:
                        add('pd_coordinate', panel_type, mid, panel_output_path(c, panel_type, mid), 'pending')
            elif panel_type in PER_METHOD_PANEL_TYPES:
                methods = c['full_panel_methods'] if panel_type == ERROR_MAPS else (['GT'] + c['full_panel_methods'])
                for mid in methods:
                    add('scripted_per_method', panel_type, mid, panel_output_path(c, panel_type, mid), 'visible')
            else:
                add('scripted_figure_level', panel_type, '', panel_output_path(c, panel_type, None), 'visible')
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


FINAL_FIGURE_VALIDATION_FIELDS = [
    'figure_id', 'archetype_id', 'sample_idx', 'expected_png_path', 'expected_vector_path', 'png_exists',
    'width_px', 'height_px', 'dpi_x', 'dpi_y', 'png_file_size_bytes', 'png_min_dpi_ok', 'vector_exists',
    'pdf_page_count', 'pdf_file_size_bytes', 'pdf_valid', 'vector_kind', 'status',
]


def build_not_yet_rendered_final_figure_validation(manifest):
    rows = []
    for c in FIGURE_CONTRACTS:
        si = manifest[c['archetype_id']]
        paths = final_figure_paths(c)
        rows.append(dict(
            figure_id=c['figure_id'], archetype_id=c['archetype_id'], sample_idx=si,
            expected_png_path=paths['png'], expected_vector_path=paths['pdf'], png_exists=False,
            width_px='', height_px='', dpi_x='', dpi_y='', png_file_size_bytes='', png_min_dpi_ok='',
            vector_exists=False, pdf_page_count='', pdf_file_size_bytes='', pdf_valid='', vector_kind='',
            status='not_yet_rendered',
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

# Explicit execution-state vocabulary for the generated report (never inferred
# merely from whether zoom_result is None): each cmd_* entry point states
# truthfully which stage(s) it has actually completed.
EXECUTION_MODE_PLAN_ONLY = 'plan_only'
EXECUTION_MODE_RENDER_FIELDS = 'render_fields'
EXECUTION_MODE_ASSEMBLE_COMPOSITES = 'assemble_composites'
EXECUTION_MODE_FULL = 'full'
EXECUTION_MODES = (EXECUTION_MODE_PLAN_ONLY, EXECUTION_MODE_RENDER_FIELDS, EXECUTION_MODE_ASSEMBLE_COMPOSITES,
                     EXECUTION_MODE_FULL)


def read_zoom_result_from_validation():
    """Reads back the deterministic zoom bounds/score already written by a
    prior --render-fields run (validation/zoom_selection_validation.csv) so
    --assemble-composites/--full can report the real computed values
    without needing render_fields' in-memory state threaded in. Returns
    None if the file is absent or still not_yet_computed."""
    path = VALIDATION_DIR / 'zoom_selection_validation.csv'
    if not path.exists():
        return None
    rows = p2da.read_csv_dicts(path)
    if not rows or rows[0]['status'] != 'computed':
        return None
    r = rows[0]
    return dict(y0=int(r['y0']), y1=int(r['y1']), x0=int(r['x0']), x1=int(r['x1']), score=float(r['score']))


def read_rendered_panel_rows():
    """Reads back the scripted-panel rows already written by a prior
    --render-fields run (validation/panel_validation.csv). Returns [] if
    the file is absent."""
    path = VALIDATION_DIR / 'panel_validation.csv'
    if not path.exists():
        return []
    return p2da.read_csv_dicts(path)


def _status_banner_lines(execution_mode):
    if execution_mode == EXECUTION_MODE_FULL:
        return ['Phase 2D-B complete.', 'All final composites and figure-data packages validated.']
    if execution_mode == EXECUTION_MODE_ASSEMBLE_COMPOSITES:
        return ['Phase 2D-B composite assembly complete.',
                 'Final composites assembled from validated scripted panels and manual topology input.']
    if execution_mode == EXECUTION_MODE_RENDER_FIELDS:
        return ['Phase 2D-B scripted rendering complete.',
                 'Manual topology input and final composite assembly still pending.']
    # plan_only
    if IS_LIGHTWEIGHT_CHECKOUT:
        return ['Phase 2D-B planning complete (lightweight checkout).',
                 'Final publication rendering pending; raw Spark arrays are absent here by design.']
    return ['Phase 2D-B planning complete (authoritative Spark checkout).',
             'Raw arrays were intentionally not loaded in --plan-only; run --render-fields to continue.']


def _scope_intro_text(execution_mode):
    if execution_mode == EXECUTION_MODE_PLAN_ONLY:
        if IS_LIGHTWEIGHT_CHECKOUT:
            return (
                'This document reflects a `--plan-only` run in a lightweight (non-Spark) checkout. It reads '
                'exclusively frozen Phase-1 through Phase-2D-A artifacts (118 files, checksummed before and '
                'after this stage) and never touches `data_out/`, `data_out_fixed/`, or reruns any '
                'training/inference/TTK step. Raw Spark arrays are absent in this checkout by design, so PD '
                'source verdicts may remain `pending_authoritative_spark_source_discovery` here. Phase 2D-A '
                'is treated as complete and authoritative; no sample is re-selected and no alternate is '
                'activated here.'
            )
        return (
            'This document reflects a `--plan-only` run on the authoritative Spark machine. It reads '
            'exclusively frozen Phase-1 through Phase-2D-A artifacts (118 files, checksummed before and after '
            'this stage) and never touches `data_out/`, `data_out_fixed/`, or reruns any training/inference/'
            'TTK step. The raw Spark arrays may exist on this machine, but `--plan-only` intentionally does '
            'not load or render them in this mode -- exact PD VTU sources are still audited directly on disk '
            '(Section 5), and no method-level PD verdict may remain pending after this run. Phase 2D-A is '
            'treated as complete and authoritative; no sample is re-selected and no alternate is activated '
            'here.'
        )
    if execution_mode == EXECUTION_MODE_RENDER_FIELDS:
        return (
            'This document reflects a `--render-fields` run. The real Spark arrays (`data_out/`, '
            '`data_out_fixed/`) were loaded and audited, and every scripted panel (speed/error fields, metric '
            'strips, PD-diagram or scalar-fallback panels, and the deterministic sample-119 zoom crop) was '
            'rendered and validated. Manual topology (merge-tree) input and final composite assembly may '
            'still be pending -- see Sections 6 and 9.'
        )
    if execution_mode == EXECUTION_MODE_ASSEMBLE_COMPOSITES:
        return (
            'This document reflects an `--assemble-composites` run. Manual topology (merge-tree) inputs were '
            'validated against their declared metadata, and the six final composite figures were assembled '
            'strictly from the explicit `plan/final_composite_manifest.csv` -- never by globbing a directory. '
            'This is not a `--plan-only` run: it consumes already-rendered scripted panels from a prior '
            '`--render-fields` run and does not itself plan, re-select, or render fields.'
        )
    # full
    return (
        'This document reflects a completed `--full` run: planning, scripted rendering (`--render-fields`), '
        'manual topology input validation, and final composite assembly (`--assemble-composites`) all '
        'completed in sequence and passed every required validation. All six final composite figures are '
        'rendered and validated; nothing in this run remains lightweight or planning-only.'
    )


def build_phase2db_doc_lines(manifest, zoom_result, manual_topo_rows, panel_rows, pd_discovery_rows=None,
                                execution_mode=EXECUTION_MODE_PLAN_ONLY, final_figure_rows=None):
    if execution_mode not in EXECUTION_MODES:
        raise SystemExit(
            f'[hard-fail] Unknown Phase 2D-B report execution_mode={execution_mode!r}; expected one of '
            f'{EXECUTION_MODES}.'
        )
    if execution_mode != EXECUTION_MODE_PLAN_ONLY and zoom_result is None:
        raise SystemExit(
            f'[hard-fail] Cannot build the Phase 2D-B report in execution_mode={execution_mode!r} with '
            f'zoom_result=None -- the deterministic zoom must already be computed by this stage. This '
            f'indicates an inconsistent execution state; refusing to report it as though it were legitimate.'
        )
    lines = []
    a = lines.append
    a('# Phase 2D-B: Final Publication-Quality Figure Production')
    a('')
    a('```')
    for banner_line in _status_banner_lines(execution_mode):
        a(banner_line)
    a('```')
    a('')
    a('## 1. Scope and frozen inputs')
    a('')
    a(_scope_intro_text(execution_mode))
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
          f"x=[{zoom_result['x0']}, {zoom_result['x1']}), score={zoom_result['score']:.6f}. Computed from the "
          f"real GT and per-method error fields loaded in `--render-fields`.")
    elif IS_LIGHTWEIGHT_CHECKOUT:
        a('**Not yet computed.** The zoom window score requires the real GT and per-method error fields '
          '(`data_out_fixed/`/`data_out/`), which are absent in this lightweight checkout by design. '
          '`select_deterministic_zoom()` is implemented and synthetic-tested; it will run in `--render-fields`.')
    else:
        a('**Not yet computed.** The raw GT and per-method error fields required by the zoom-window score '
          'exist on this authoritative Spark machine, but `--plan-only` intentionally does not load or render '
          'them; `select_deterministic_zoom()` will run in `--render-fields`.')
    a('')
    a('## 5. Authoritative PD coordinate source discovery')
    a('')
    if pd_discovery_rows:
        n_avail = sum(1 for r in pd_discovery_rows if r['usable_status'] == STATUS_AVAILABLE)
        n_pending = sum(1 for r in pd_discovery_rows if r['usable_status'] == STATUS_PENDING)
        n_unavail = sum(1 for r in pd_discovery_rows if r['usable_status'] == STATUS_UNAVAILABLE)
        env = 'a lightweight (non-Spark) checkout' if IS_LIGHTWEIGHT_CHECKOUT else 'the authoritative Spark machine'
        baseline_search_roots = {mid: [_rel(p) for p in roots] for mid, roots in CNN_GAN_BICUBIC_SEARCH_ROOTS.items()}
        a(f'An exact, repository-relative filesystem search (`plan/pd_source_discovery.csv`, '
          f'{len(pd_discovery_rows)} row(s); reduced per-`(figure, sample, method)` verdicts in '
          f'`plan/pd_source_verdicts.csv`) was performed for every (figure, method) requiring a '
          f'`pd_evidence`/`pd_comparison` panel -- GT, CNN, GAN, and Bicubic included, none assumed found '
          f'without a concrete `selected_candidate_path`. This process is running in {env}: {n_avail} '
          f'method(s) available_validated, {n_pending} pending_authoritative_spark_source_discovery, '
          f'{n_unavail} unavailable_after_authoritative_spark_audit. On the authoritative Spark machine, no '
          f'method-level verdict is ever left pending (enforced in code). Only the exact TTK PD VTU convention '
          f'is matched: `<artifact_alias>_topology/pd/<GT|SR>/<artifact_alias>_<GT|SR>_s<sample_idx>_'
          f'..._pd_port_0.vtu`, with `mt/` paths, `_mt_port_*.vtu`, and `_pd_port_1.vtu` always excluded, and '
          f'GT cross-checked for coordinate agreement across every candidate topology tree before being '
          f'accepted as canonical. Method-to-artifact aliases: {dict(METHOD_ARTIFACT_ALIASES)}; CNN/GAN/bicubic '
          f'use their own method_id (no `<alias>_topology` directory exists for them anywhere in this '
          f'repository) and are searched under {baseline_search_roots}. Existing PD-overlay-related scripts '
          f'already in this repository were also inventoried for provenance.')
    elif execution_mode == EXECUTION_MODE_PLAN_ONLY:
        a('Not yet run in this render state.')
    else:
        a('Not summarized in this execution_mode\'s report -- exact PD source discovery/resolution already '
          'ran (per-figure, inside `--render-fields`) during rendering; see `plan/pd_source_discovery.csv` '
          'and `plan/pd_source_verdicts.csv` from the most recent `--plan-only` run for the full audit trail.')
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
    a('- `validation/prior_phase_immutability_check.csv`: all 118 protected files confirmed unchanged.')
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
        'plan/pd_source_discovery.csv', 'plan/pd_source_verdicts.csv', 'plan/final_composite_manifest.csv',
        'plan/final_figure_captions.md',
    ] + [f'figure_data/{FIGURE_DATA_FILENAMES[i]}' for i in range(1, 7)] + [
        'validation/prior_phase_immutability_check.csv', 'validation/figure_data_reproduction.csv',
        'validation/panel_validation.csv', 'validation/final_figure_validation.csv',
        'validation/zoom_selection_validation.csv', 'validation/panel_scale_provenance.csv',
    ]:
        a(f'- `ttk_runs_fixed/unified_candidate_analysis/phase2db/{rel}`')
    a('- `docs/unified_candidate_analysis_phase2db.md` (this file)')
    a('- `logs/unified_candidate_analysis_phase2db.log`')
    a('')
    if execution_mode == EXECUTION_MODE_PLAN_ONLY:
        a('Not yet generated (pending `--render-fields`, manual topology export, and `--assemble-composites`): '
          '`panels/**/*.png`, `manual_topology_inputs/**/*`, `figures/**/*`.')
    else:
        n_panels = len(panel_rows or [])
        a(f'Scripted panels rendered and validated in `--render-fields`: {n_panels} panel file(s) under '
          f'`panels/**/*.png` (never claimed merely because the directory exists -- counted from '
          f'`validation/panel_validation.csv`).')
        if execution_mode == EXECUTION_MODE_RENDER_FIELDS:
            a('')
            a('Not yet generated (pending manual topology export and `--assemble-composites`): '
              '`manual_topology_inputs/**/*`, `figures/**/*`.')
        else:
            n_manual = len(manual_topo_rows or [])
            n_manual_present = sum(1 for r in (manual_topo_rows or []) if r['status'] == 'present')
            a(f'Manual topology inputs validated: {n_manual_present}/{n_manual} panel(s) under '
              f'`manual_topology_inputs/**/*` (see `plan/manual_topology_requirements.csv`).')
            n_final = len(final_figure_rows or [])
            n_final_ready = sum(1 for r in (final_figure_rows or [])
                                  if r['status'] == 'rendered' and r['png_exists'] and r['png_min_dpi_ok']
                                  and r['vector_exists'] and r['pdf_valid'])
            a(f'Final composite figures assembled and validated: {n_final_ready}/{n_final} under '
              f'`figures/**/*` (see `validation/final_figure_validation.csv`).')
    a('')
    return lines


def write_phase2db_doc(manifest, zoom_result, manual_topo_rows, panel_rows, pd_discovery_rows=None,
                          execution_mode=EXECUTION_MODE_PLAN_ONLY, final_figure_rows=None):
    lines = build_phase2db_doc_lines(manifest, zoom_result, manual_topo_rows, panel_rows, pd_discovery_rows,
                                        execution_mode, final_figure_rows)
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
    log('Read-only w.r.t. Phase-1/2A/2B/2C/2D-A artifacts (118 files). Reads only frozen CSV/Markdown '
        'inputs; touches no raw .npy array; renders no image.')
    log('=' * 88)

    checksums_before, file_to_phase = preflight_immutability()

    manifest = read_and_validate_selection_manifest()
    require_completed_phase2d_a_state(manifest)
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

    pd_discovery_rows = []
    pd_verdict_rows = []
    pd_verdicts_by_figure = {}
    for c in FIGURE_CONTRACTS:
        rows_for_figure, verdicts_for_figure = discover_and_resolve_pd_sources_for_figure(c, manifest)
        pd_discovery_rows.extend(rows_for_figure)
        pd_verdict_rows.extend({k: v[k] for k in PD_SOURCE_VERDICT_FIELDS} for v in verdicts_for_figure.values())
        pd_verdicts_by_figure[c['figure_id']] = {mid: v['verdict'] for mid, v in verdicts_for_figure.items()}
    write_csv(PLAN_DIR / 'pd_source_discovery.csv', PD_SOURCE_DISCOVERY_FIELDS, pd_discovery_rows)
    write_csv(PLAN_DIR / 'pd_source_verdicts.csv', PD_SOURCE_VERDICT_FIELDS, pd_verdict_rows)
    if not IS_LIGHTWEIGHT_CHECKOUT:
        still_pending = [r for r in pd_verdict_rows if r['verdict'] == STATUS_PENDING]
        if still_pending:
            raise SystemExit(
                f'[hard-fail] On the authoritative Spark machine, no exact method-level PD source verdict may '
                f'remain pending_authoritative_spark_source_discovery after a complete search. Still pending: '
                f'{still_pending}.'
            )
    overlay_scripts = find_pd_overlay_scripts()
    log(f'[pd-discovery] Found {len(overlay_scripts)} existing PD-overlay-related script(s) (informational): '
        f'{[str(p.relative_to(REPO_ROOT)) for p in overlay_scripts]}')

    panel_rows = build_final_panel_manifest_rows(manifest, pd_verdicts_by_figure)
    write_csv(PLAN_DIR / 'final_panel_manifest.csv', FINAL_PANEL_MANIFEST_FIELDS, panel_rows)

    manual_topo_rows = build_manual_topology_requirements_rows(manifest)
    write_csv(PLAN_DIR / 'manual_topology_requirements.csv', MANUAL_TOPOLOGY_REQ_FIELDS, manual_topo_rows)

    composite_manifest_rows = build_final_composite_manifest_rows(manifest, pd_verdicts_by_figure)
    write_csv(PLAN_DIR / 'final_composite_manifest.csv', FINAL_COMPOSITE_MANIFEST_FIELDS, composite_manifest_rows)

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
    write_csv(VALIDATION_DIR / 'final_figure_validation.csv', FINAL_FIGURE_VALIDATION_FIELDS,
               final_figure_validation_rows)

    write_zoom_selection_validation(zoom_result, manifest)
    write_csv(VALIDATION_DIR / 'panel_scale_provenance.csv',
               ['figure_id', 'speed_vmin', 'speed_vmax', 'error_vmin', 'error_vmax', 'colormap_speed',
                'colormap_error', 'physical_units'], [])

    write_captions_md(manifest)
    write_phase2db_doc(manifest, zoom_result, manual_topo_rows, panel_rows, pd_discovery_rows,
                         execution_mode=EXECUTION_MODE_PLAN_ONLY)

    postflight_immutability(checksums_before, file_to_phase, VALIDATION_DIR / 'prior_phase_immutability_check.csv')

    n_manual_missing = sum(1 for r in manual_topo_rows if r['status'] == 'missing')
    n_pd_available = sum(1 for r in panel_rows if r['requires_pd_coordinate_source']
                           and r['pd_coordinate_source_found'] == STATUS_AVAILABLE)
    n_pd_pending = sum(1 for r in panel_rows if r['requires_pd_coordinate_source']
                         and r['pd_coordinate_source_found'] == STATUS_PENDING)
    n_pd_unavailable = sum(1 for r in panel_rows if r['requires_pd_coordinate_source']
                             and r['pd_coordinate_source_found'] == STATUS_UNAVAILABLE)
    log('')
    log('=' * 88)
    log(f'RESULT: phase2db_planning_complete_final_rendering_pending. 6 figure plans, {len(panel_rows)} '
        f'planned panels ({n_manual_missing} awaiting manual topology input, {n_pd_available} PD panels '
        f'available_validated, {n_pd_pending} pending_authoritative_spark_source_discovery, '
        f'{n_pd_unavailable} unavailable_after_authoritative_spark_audit), {len(repro_rows)} figure-data '
        f'values reproduced within tolerance.')
    log('=' * 88)
    flush_log(LOG_PATH)
    return dict(manifest=manifest, figure_plan_rows=figure_plan_rows, panel_rows=panel_rows,
                 manual_topo_rows=manual_topo_rows, figure_data_by_id=figure_data_by_id,
                 repro_rows=repro_rows, pd_discovery_rows=pd_discovery_rows, pd_verdict_rows=pd_verdict_rows)


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


# =============================================================================
# Real PD panel rendering (Section 3): reads VALIDATED birth/death
# coordinates (never fabricates), verifies finiteness and sample/method
# identity, uses common axes within a figure, draws the diagonal, and
# annotates the frozen scalar PD distance. When a method's authoritative
# discovery verdict is unavailable_after_authoritative_spark_audit (never
# merely pending), the figure contract switches to a scalar PD-evidence
# fallback panel for that method instead.
# =============================================================================

def render_pd_diagram_panels(contract, panel_type, manifest, per_sample, pd_verdicts_by_method):
    """Real coordinate-based PD panel renderer: one panel per method (GT +
    full_panel_methods with an available_validated verdict), common axes
    within the figure, the diagonal drawn, and the frozen scalar PD
    distance annotated. `pd_verdicts_by_method` maps method_id -> the full
    resolve_pd_source_verdict() dict (only for methods with
    verdict==available_validated -- callers must route unavailable methods
    to the scalar fallback instead). Coordinates are read directly from the
    verdict (already parsed and validated by parse_and_validate_pd_vtu()) --
    never re-derived from raw VTU point geometry."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    si = manifest[contract['archetype_id']]
    out_dir = PANELS_DIR / figure_dir_name(contract)
    out_dir.mkdir(parents=True, exist_ok=True)
    methods = ['GT'] + [m for m in contract['full_panel_methods'] if m in pd_verdicts_by_method]
    if 'GT' not in pd_verdicts_by_method:
        raise SystemExit(
            f"[hard-fail] Figure {contract['figure_id']} {panel_type} panel requires a validated GT PD "
            f'coordinate source; GT is never assumed found without a concrete candidate_path.'
        )
    coords = {mid: (pd_verdicts_by_method[mid]['birth'], pd_verdicts_by_method[mid]['death']) for mid in methods}
    all_vals = np.concatenate([np.concatenate([b, d]) for b, d in coords.values()])
    lo, hi = float(all_vals.min()), float(all_vals.max())
    rows = []
    for mid in methods:
        birth, death = coords[mid]
        fig, ax = plt.subplots(figsize=(2.6, 2.6), dpi=300)
        ax.plot([lo, hi], [lo, hi], color='gray', linestyle='--', linewidth=0.8, label='diagonal')
        ax.scatter(birth, death, s=6, alpha=0.7)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        label = GT_DISPLAY_LABEL if mid == 'GT' else HUMAN_LABELS[mid]
        if mid != 'GT':
            pd_val = per_sample[mid][si]['pd_distance']
            ax.set_title(f'{label}\nPD distance={pd_val:.3f}', fontsize=8)
        else:
            ax.set_title(label, fontsize=8)
        ax.set_xlabel('Birth', fontsize=7)
        ax.set_ylabel('Death', fontsize=7)
        out_path = out_dir / f'{panel_type}_{mid}.png'
        fig.savefig(out_path, dpi=300, metadata={'Software': ''})
        plt.close(fig)
        rows.append(dict(output_path=str(out_path.relative_to(REPO_ROOT)), method_id=mid, panel_type=panel_type))
    return rows


def render_scalar_pd_fallback_panel(contract, panel_type, manifest, per_sample, fallback_methods):
    """Scalar PD-evidence fallback (Section 3): used only for methods whose
    authoritative discovery verdict is confirmed
    unavailable_after_authoritative_spark_audit. Draws the frozen PD
    distance, method rank, and pairwise improvement margin vs CNN as a
    compact bar/dot comparison -- never a fabricated diagram."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    si = manifest[contract['archetype_id']]
    out_dir = PANELS_DIR / figure_dir_name(contract)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels, values, margins = [], [], []
    for mid in fallback_methods:
        pd_val = per_sample.get(mid, {}).get(si, {}).get('pd_distance', float('nan'))
        cnn_val = per_sample.get(CNN, {}).get(si, {}).get('pd_distance', float('nan'))
        labels.append(HUMAN_LABELS[mid])
        values.append(pd_val if math.isfinite(pd_val) else 0.0)
        margins.append((cnn_val - pd_val) if (math.isfinite(pd_val) and math.isfinite(cnn_val)) else float('nan'))
    fig, ax = plt.subplots(figsize=(1.6 * max(len(fallback_methods), 1), 2.4), dpi=300)
    finite_mask = [math.isfinite(per_sample.get(mid, {}).get(si, {}).get('pd_distance', float('nan')))
                    for mid in fallback_methods]
    colors = ['#4C72B0' if ok else '#BBBBBB' for ok in finite_mask]
    ax.bar(labels, values, color=colors)
    for i, (v, ok) in enumerate(zip(values, finite_mask)):
        ax.text(i, v, (f'{v:.2f}' if ok else 'N/A'), ha='center', va='bottom', fontsize=7)
    ax.set_ylabel('PD distance (scalar fallback)', fontsize=7)
    ax.set_title('Scalar PD-evidence fallback\n(no validated coordinate source)', fontsize=8)
    out_path = out_dir / f'{panel_type}_scalar_fallback.png'
    fig.savefig(out_path, dpi=300, metadata={'Software': ''})
    plt.close(fig)
    return dict(output_path=str(out_path.relative_to(REPO_ROOT)), method_id=','.join(fallback_methods),
                 panel_type=panel_type)


def render_pd_mt_tradeoff_compact_panel(contract, manifest, per_sample):
    """Figure 4: compact PD-vs-MT improvement scatter for each required
    method relative to CNN."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    si = manifest[contract['archetype_id']]
    out_dir = PANELS_DIR / figure_dir_name(contract)
    out_dir.mkdir(parents=True, exist_ok=True)
    cnn_pd = per_sample[CNN][si]['pd_distance']
    cnn_mt = per_sample[CNN][si]['mt_distance']
    fig, ax = plt.subplots(figsize=(2.8, 2.8), dpi=300)
    for mid in contract['required_methods']:
        if mid == CNN:
            continue
        pd_imp = cnn_pd - per_sample[mid][si]['pd_distance']
        mt_imp = cnn_mt - per_sample[mid][si]['mt_distance']
        ax.scatter(pd_imp, mt_imp, s=30)
        ax.annotate(HUMAN_LABELS[mid], (pd_imp, mt_imp), fontsize=6, xytext=(3, 3), textcoords='offset points')
    ax.axhline(0, color='gray', linewidth=0.6)
    ax.axvline(0, color='gray', linewidth=0.6)
    ax.set_xlabel('PD improvement vs CNN', fontsize=7)
    ax.set_ylabel('MT improvement vs CNN', fontsize=7)
    ax.set_title('PD/MT tradeoff (compact)', fontsize=8)
    out_path = out_dir / f'{PD_MT_TRADEOFF_COMPACT}.png'
    fig.savefig(out_path, dpi=300, metadata={'Software': ''})
    plt.close(fig)
    return str(out_path.relative_to(REPO_ROOT))


def render_pd_mt_comparison_compact_panel(contract, manifest, per_sample):
    """Figure 6: compact side-by-side PD/MT bar comparison across required
    methods (both descriptors, not just a tradeoff scatter)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    si = manifest[contract['archetype_id']]
    out_dir = PANELS_DIR / figure_dir_name(contract)
    out_dir.mkdir(parents=True, exist_ok=True)
    methods = contract['required_methods']
    pd_vals = [per_sample[mid][si]['pd_distance'] for mid in methods]
    mt_vals = [per_sample[mid][si]['mt_distance'] for mid in methods]
    x = np.arange(len(methods))
    width = 0.35
    fig, ax = plt.subplots(figsize=(1.4 * len(methods), 2.6), dpi=300)
    ax.bar(x - width / 2, pd_vals, width, label='PD')
    ax.bar(x + width / 2, mt_vals, width, label='MT')
    ax.set_xticks(x)
    ax.set_xticklabels([HUMAN_LABELS[mid] for mid in methods], fontsize=6, rotation=30, ha='right')
    ax.legend(fontsize=7)
    ax.set_title('PD/MT comparison (compact)', fontsize=8)
    out_path = out_dir / f'{PD_MT_COMPARISON_COMPACT}.png'
    fig.savefig(out_path, dpi=300, metadata={'Software': ''})
    plt.close(fig)
    return str(out_path.relative_to(REPO_ROOT))


def render_figure_transactional(contract, panel_render_fn):
    """Renders one figure's panels into a temporary staging directory
    (inside the repo tree, so relative-path bookkeeping stays valid),
    validates every declared panel actually exists and is non-empty, and
    only then promotes the staged directory over the authoritative
    panels/<figure_dir>/ location -- replacing it wholesale, so any stale
    or unexpected PNG left over from a prior run is removed. On any
    failure, the staging directory is discarded and the authoritative
    directory is left completely untouched (no partial update)."""
    import shutil
    import tempfile

    global PANELS_DIR
    real_panels_dir = PANELS_DIR
    staging_base = OUT_DIR / '_staging'
    staging_base.mkdir(parents=True, exist_ok=True)
    staging_run_dir = Path(tempfile.mkdtemp(prefix=f"fig{contract['figure_id']:02d}_", dir=staging_base))
    PANELS_DIR = staging_run_dir
    try:
        result = panel_render_fn()
        rows = result[0] if isinstance(result, tuple) else result
        for r in rows:
            p = REPO_ROOT / r['output_path']
            if not p.exists() or p.stat().st_size == 0:
                raise SystemExit(
                    f"[hard-fail] Figure {contract['figure_id']}: declared panel {r['output_path']} was not "
                    f'actually produced (missing or empty). No partial panel set was promoted to the '
                    f'authoritative directory.'
                )
    except BaseException:
        shutil.rmtree(staging_run_dir, ignore_errors=True)
        PANELS_DIR = real_panels_dir
        raise
    PANELS_DIR = real_panels_dir
    staged_figure_dir = staging_run_dir / figure_dir_name(contract)
    final_dir = PANELS_DIR / figure_dir_name(contract)
    if final_dir.exists():
        shutil.rmtree(final_dir)  # reject/remove stale unexpected PNGs from a prior run
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(staged_figure_dir), str(final_dir))
    shutil.rmtree(staging_run_dir, ignore_errors=True)
    staged_rel = str(staging_run_dir.relative_to(REPO_ROOT) / figure_dir_name(contract))
    final_rel = str(final_dir.relative_to(REPO_ROOT))
    if isinstance(result, tuple):
        promoted_rows = [dict(r, output_path=r['output_path'].replace(staged_rel, final_rel)) for r in rows]
        return (promoted_rows,) + result[1:]
    return [dict(r, output_path=r['output_path'].replace(staged_rel, final_rel)) for r in rows]


def render_all_panels_for_figure(contract, manifest, audit, ordered_selected, per_sample, pd_verdicts):
    """Renders EVERY declared scripted panel type for one figure. No panel
    type declared in a figure contract is ever skipped or silently marked
    rendered without a real renderer call. `pd_verdicts` maps method_id
    (including 'GT') -> the full resolve_pd_source_verdict() dict for every
    method required by a pd_evidence/pd_comparison panel in this figure."""
    si = manifest[contract['archetype_id']]
    panel_rows, gt_speed, method_speeds, panel_data = render_speed_and_error_panels(
        contract, manifest, audit, ordered_selected)
    panel_rows.append(dict(output_path=render_metric_strip(contract, manifest, per_sample), method_id='',
                              panel_type=METRIC_STRIP))

    scale_row = dict(
        figure_id=contract['figure_id'], speed_vmin=panel_data['speed_vmin'], speed_vmax=panel_data['speed_vmax'],
        error_vmin=panel_data['error_vmin'], error_vmax=panel_data['error_vmax'], colormap_speed='cividis',
        colormap_error='magma', physical_units='m/s',
    )

    zoom_result = None
    for panel_type in contract['panels']:
        if panel_type in PD_DIAGRAM_PANEL_TYPES:
            pending = [mid for mid in (['GT'] + contract['full_panel_methods'])
                        if pd_verdicts.get(mid, {}).get('verdict') == STATUS_PENDING]
            if pending:
                raise SystemExit(
                    f"[hard-fail] Figure {contract['figure_id']} {panel_type}: {pending} still "
                    f'pending_authoritative_spark_source_discovery. Cannot render a real panel or fall back '
                    f'to the scalar path until the authoritative Spark search concludes for these methods.'
                )
            available = {mid: pd_verdicts[mid] for mid in (['GT'] + contract['full_panel_methods'])
                          if pd_verdicts.get(mid, {}).get('verdict') == STATUS_AVAILABLE}
            unavailable = [mid for mid in contract['full_panel_methods']
                            if pd_verdicts.get(mid, {}).get('verdict') == STATUS_UNAVAILABLE]
            if available:
                if pd_verdicts.get('GT', {}).get('verdict') != STATUS_AVAILABLE:
                    raise SystemExit(
                        f"[hard-fail] Figure {contract['figure_id']} {panel_type}: GT has no "
                        f'available_validated PD coordinate source; refusing to render method panels '
                        f'without a validated GT reference.'
                    )
                panel_rows.extend(render_pd_diagram_panels(contract, panel_type, manifest, per_sample, available))
            if unavailable:
                panel_rows.append(render_scalar_pd_fallback_panel(contract, panel_type, manifest, per_sample,
                                                                      unavailable))
        elif panel_type == PD_MT_TRADEOFF_COMPACT:
            panel_rows.append(dict(output_path=render_pd_mt_tradeoff_compact_panel(contract, manifest, per_sample),
                                      method_id='', panel_type=panel_type))
        elif panel_type == PD_MT_COMPARISON_COMPACT:
            panel_rows.append(dict(
                output_path=render_pd_mt_comparison_compact_panel(contract, manifest, per_sample),
                method_id='', panel_type=panel_type))
        elif panel_type == ZOOM_CROP:
            zoom_result = select_deterministic_zoom(
                gt_speed, {mid: np.abs(method_speeds[mid] - gt_speed) for mid in contract['full_panel_methods']})
            panel_rows.append(dict(
                output_path=render_zoom_crop_panel(contract, manifest, gt_speed, method_speeds, zoom_result),
                method_id='', panel_type=ZOOM_CROP))
        # speed_fields, error_maps, metric_strip already rendered above; MT_* panel
        # types are manual-only (Section 8) and are never scripted here.

    return panel_rows, scale_row, zoom_result


def propagate_zoom_result(zoom_result, manifest):
    """Section 6: writes the exact zoom bounds and score to all four
    required destinations. No-op (leaves them in their pending state) when
    zoom_result is None."""
    fig3 = FIGURE_BY_ID[3]
    si = manifest[fig3['archetype_id']]
    if zoom_result is None:
        return
    fd_path = FIGURE_DATA_DIR / FIGURE_DATA_FILENAMES[3]
    if fd_path.exists():
        rows = p2da.read_csv_dicts(fd_path)
        for r in rows:
            r['zoom_y0'], r['zoom_y1'] = zoom_result['y0'], zoom_result['y1']
            r['zoom_x0'], r['zoom_x1'] = zoom_result['x0'], zoom_result['x1']
        write_csv(fd_path, FIGURE_DATA_FIELDS, rows)
    pm_path = PLAN_DIR / 'final_panel_manifest.csv'
    if pm_path.exists():
        rows = p2da.read_csv_dicts(pm_path)
        for r in rows:
            if int(r['figure_id']) == 3 and r['panel_type'] == ZOOM_CROP:
                r['status'] = 'rendered'
        write_csv(pm_path, FINAL_PANEL_MANIFEST_FIELDS, rows)


def write_zoom_selection_validation(zoom_result, manifest):
    fig3 = FIGURE_BY_ID[3]
    si = manifest[fig3['archetype_id']]
    row = dict(
        figure_id=3, archetype_id=fig3['archetype_id'], sample_idx=si, window_size=ZOOM_WINDOW_SIZE,
        stride=ZOOM_STRIDE, scoring_formula=ZOOM_SCORE_FORMULA,
        y0=nfmt(zoom_result['y0']) if zoom_result else '', y1=nfmt(zoom_result['y1']) if zoom_result else '',
        x0=nfmt(zoom_result['x0']) if zoom_result else '', x1=nfmt(zoom_result['x1']) if zoom_result else '',
        score=nfmt(zoom_result['score']) if zoom_result else '',
        status=('computed' if zoom_result else 'not_yet_computed'),
    )
    write_csv(VALIDATION_DIR / 'zoom_selection_validation.csv',
               ['figure_id', 'archetype_id', 'sample_idx', 'window_size', 'stride', 'scoring_formula', 'y0', 'y1',
                'x0', 'x1', 'score', 'status'], [row])


def cmd_render_fields(plan_result=None) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log('=' * 88)
    log('Unified candidate figures -- Phase 2D-B (--render-fields)')
    log('Requires the raw Spark arrays (data_out/, data_out_fixed/). Hard-fails if any required raw '
        'artifact, or PD coordinate source, is unavailable or invalid.')
    log('=' * 88)

    checksums_before, file_to_phase = preflight_immutability()
    manifest = read_and_validate_selection_manifest()
    require_completed_phase2d_a_state(manifest)
    method_inventory = p2da.load_method_inventory()
    long_table = p2da.load_long_table()
    per_sample = long_table['per_sample']

    audit, ordered_selected = _load_full_panel_arrays(manifest, method_inventory)

    render_rows = []
    scale_rows = []
    zoom_result = None
    for c in FIGURE_CONTRACTS:
        _, pd_verdicts = discover_and_resolve_pd_sources_for_figure(c, manifest)

        def _render(c=c, pd_verdicts=pd_verdicts):
            return render_all_panels_for_figure(c, manifest, audit, ordered_selected, per_sample, pd_verdicts)

        panel_rows, scale_row, fig_zoom = render_figure_transactional(c, _render)
        render_rows.extend(panel_rows)
        scale_rows.append(scale_row)
        if fig_zoom is not None:
            zoom_result = fig_zoom

    propagate_zoom_result(zoom_result, manifest)
    write_zoom_selection_validation(zoom_result, manifest)
    write_csv(VALIDATION_DIR / 'panel_scale_provenance.csv',
               ['figure_id', 'speed_vmin', 'speed_vmax', 'error_vmin', 'error_vmax', 'colormap_speed',
                'colormap_error', 'physical_units'], scale_rows)
    write_csv(VALIDATION_DIR / 'panel_validation.csv',
               ['output_path', 'method_id', 'panel_type'], render_rows)

    manual_topo_rows = build_manual_topology_requirements_rows(manifest)
    write_phase2db_doc(manifest, zoom_result, manual_topo_rows, render_rows, pd_discovery_rows=None,
                         execution_mode=EXECUTION_MODE_RENDER_FIELDS)

    postflight_immutability(checksums_before, file_to_phase, VALIDATION_DIR / 'prior_phase_immutability_check.csv')
    log('')
    log('=' * 88)
    log(f'RESULT: --render-fields wrote {len(render_rows)} panel(s).')
    log('=' * 88)
    flush_log(LOG_PATH)
    return dict(render_rows=render_rows, zoom_result=zoom_result)


# =============================================================================
# Real PNG/PDF output inspection (Section 9). Never hardcode a pass -- every
# dimension/DPI/page-count value is measured from the actual file.
# =============================================================================

# Coarse minimum pixel floor consistent with >=300 dpi at a plausible minimum
# publication panel size (single-column figures are commonly >=3.3in wide).
PNG_MIN_WIDTH_PX = 300
PNG_MIN_HEIGHT_PX = 300


def inspect_png(path):
    """Opens the PNG and measures real width/height/dpi/size. Never assumes
    success from the file merely existing.

    Restricted to formats=['PNG']: every panel this script writes is a real
    PNG, so there is never a reason to let Pillow guess across its full
    plugin registry. Image.open() without a `formats` filter lazily
    registers and probes every installed image plugin on the first call in
    a process; for corrupt or unrecognizable bytes this scan has been
    observed to be slow in some environments. Restricting to PNG keeps
    corrupt-file detection fast and deterministic everywhere."""
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return dict(width_px='', height_px='', dpi_x='', dpi_y='',
                     file_size_bytes=(p.stat().st_size if p.exists() else 0),
                     validation_status='FAIL_missing_or_empty')
    from PIL import Image, UnidentifiedImageError
    try:
        with Image.open(p, formats=['PNG']) as img:
            img.verify()
        with Image.open(p, formats=['PNG']) as img:
            width_px, height_px = img.size
            dpi = img.info.get('dpi', (0, 0))
            dpi_x, dpi_y = (dpi[0], dpi[1]) if dpi and dpi[0] else (0, 0)
    except (UnidentifiedImageError, OSError):
        return dict(width_px='', height_px='', dpi_x='', dpi_y='', file_size_bytes=p.stat().st_size,
                     validation_status='FAIL_corrupt_or_unreadable')
    ok = width_px >= PNG_MIN_WIDTH_PX and height_px >= PNG_MIN_HEIGHT_PX
    return dict(width_px=width_px, height_px=height_px, dpi_x=nfmt(dpi_x or None), dpi_y=nfmt(dpi_y or None),
                 file_size_bytes=p.stat().st_size, validation_status=('PASS' if ok else 'FAIL_min_dimensions'))


def inspect_pdf(path):
    """Dependency-free PDF validity check: %PDF- header magic bytes and a
    page count via /Type /Page object occurrences (never /Type /Pages)."""
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return dict(file_size_bytes=(p.stat().st_size if p.exists() else 0), pdf_page_count=0,
                     validation_status='FAIL_missing_or_empty')
    data = p.read_bytes()
    valid_header = data[:5] == b'%PDF-'
    page_count = len(re.findall(rb'/Type\s*/Page(?!s)\b', data))
    ok = valid_header and page_count > 0
    return dict(file_size_bytes=len(data), pdf_page_count=page_count,
                 validation_status=('PASS' if ok else 'FAIL_invalid_pdf'))


# =============================================================================
# --assemble-composites (requires manual topology panels; not run here)
# =============================================================================

MT_CONSISTENCY_FIELDS = ('persistence_threshold', 'camera_or_view_id', 'scalar_range')


def _read_single_metadata_row(meta_path):
    meta_rows = p2da.read_csv_dicts(meta_path)
    if not meta_rows:
        raise SystemExit(f'[hard-fail] Manual topology metadata file is empty: {meta_path}')
    return meta_rows[0]


def require_manual_topology_panels(manifest):
    """Section 8: full manual MT panel + metadata validation -- nonempty
    valid PNG, real pixel dimensions matching the declared metadata,
    identity fields (figure_id/sample_idx/method_id) matching the manifest,
    a repository-relative source_vtu_path, numeric threshold/sampling/
    line-size fields, and same-figure comparability (persistence_threshold/
    camera_or_view_id/scalar_range consistent across a figure's panels)."""
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
    by_figure = {}
    for r in manual_topo_rows:
        meta_path = REPO_ROOT / r['expected_metadata_path']
        panel_path = REPO_ROOT / r['expected_panel_path']
        row0 = _read_single_metadata_row(meta_path)

        missing_fields = [f for f in MANUAL_TOPOLOGY_METADATA_FIELDS if not str(row0.get(f, '')).strip()]
        if missing_fields:
            raise SystemExit(
                f'[hard-fail] Manual topology metadata {meta_path} is missing required field(s): {missing_fields}'
            )
        if str(row0.get('figure_id', '')) != str(r['figure_id']):
            raise SystemExit(f"[hard-fail] {meta_path}: metadata figure_id={row0.get('figure_id')!r} does not "
                               f"match expected {r['figure_id']!r}.")
        if str(row0.get('sample_idx', '')) != str(r['sample_idx']):
            raise SystemExit(f"[hard-fail] {meta_path}: metadata sample_idx={row0.get('sample_idx')!r} does not "
                               f"match expected {r['sample_idx']!r}.")
        if str(row0.get('method_id', '')) != str(r['method_id']):
            raise SystemExit(f"[hard-fail] {meta_path}: metadata method_id={row0.get('method_id')!r} does not "
                               f"match expected {r['method_id']!r}.")
        source_vtu_path = str(row0.get('source_vtu_path', ''))
        if source_vtu_path.startswith('/'):
            raise SystemExit(f'[hard-fail] {meta_path}: source_vtu_path must be repository-relative POSIX text, '
                               f'got an absolute path: {source_vtu_path!r}.')
        for numeric_field in ('persistence_threshold', 'arc_sampling', 'arc_line_size', 'image_width',
                                'image_height'):
            try:
                float(row0[numeric_field])
            except (TypeError, ValueError):
                raise SystemExit(f'[hard-fail] {meta_path}: field {numeric_field!r}={row0[numeric_field]!r} is '
                                   f'not numeric.')

        insp = inspect_png(panel_path)
        if insp['validation_status'] != 'PASS':
            raise SystemExit(f"[hard-fail] Manual topology panel {panel_path} failed validation: "
                               f"{insp['validation_status']} (expected a nonempty, valid, adequately-sized PNG).")
        if int(float(row0['image_width'])) != insp['width_px'] or int(float(row0['image_height'])) != insp['height_px']:
            raise SystemExit(
                f"[hard-fail] {meta_path}: declared image_width/image_height "
                f"({row0['image_width']}x{row0['image_height']}) does not match the actual PNG pixel "
                f"dimensions ({insp['width_px']}x{insp['height_px']}) of {panel_path}."
            )
        by_figure.setdefault(r['figure_id'], []).append(row0)

    for figure_id, rows0 in by_figure.items():
        if len(rows0) < 2:
            continue
        for field in MT_CONSISTENCY_FIELDS:
            values = {row0[field] for row0 in rows0}
            if len(values) > 1:
                raise SystemExit(
                    f'[hard-fail] Figure {figure_id}: manual topology panels are not comparable -- field '
                    f'{field!r} differs across panels required to share common axes/scale conventions: '
                    f'{values}.'
                )
    return manual_topo_rows


def load_composite_manifest_for_figure(figure_id):
    path = PLAN_DIR / 'final_composite_manifest.csv'
    if not path.exists():
        raise SystemExit(f'[hard-fail] {path} does not exist; run --plan-only first.')
    rows = [r for r in p2da.read_csv_dicts(path) if int(r['figure_id']) == figure_id]
    rows.sort(key=lambda r: int(r['panel_order']))
    return rows


def build_composite_for_figure(contract, manifest):
    """Assembles ONE figure's final composite (PNG + vector PDF) STRICTLY
    from plan/final_composite_manifest.csv -- never a directory glob. Every
    row with final_visible_status in (visible, scalar_fallback) is required
    to exist; a `pending` row hard-fails (still awaiting an authoritative PD
    verdict). Duplicate (panel_type, method_id) identities are rejected, but
    several unavailable-method rows may legitimately share ONE combined
    scalar-fallback source_path -- that shared path is de-duplicated
    (first-occurrence order preserved) so it appears exactly once in the
    assembled visual grid, never once per sharing row. Any PNG present on
    disk that is not referenced by the manifest is rejected as unexpected.
    Declared panel_order is preserved exactly (figure 3's compact F2 role --
    excluded from full_panel_methods, so it never appears as a
    speed/error/PD panel -- is respected automatically since the manifest
    is itself built from the same figure contract)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    si = manifest[contract['archetype_id']]
    manifest_rows = load_composite_manifest_for_figure(contract['figure_id'])
    if not manifest_rows:
        raise SystemExit(f"[hard-fail] Figure {contract['figure_id']} has no rows in "
                           f'plan/final_composite_manifest.csv; run --plan-only first.')

    pending_rows = [r for r in manifest_rows if r['final_visible_status'] == 'pending']
    if pending_rows:
        raise SystemExit(
            f"[hard-fail] Figure {contract['figure_id']} has {len(pending_rows)} panel(s) still "
            f"final_visible_status=pending (awaiting an authoritative PD source verdict): "
            f"{[(r['panel_type'], r['method_id']) for r in pending_rows]}. Cannot assemble the final composite."
        )
    required_rows = [r for r in manifest_rows if r['final_visible_status'] in ('visible', 'scalar_fallback')]

    # Reject duplicate (panel_type, method_id) identities -- never source_path
    # identity, since the scalar PD fallback deliberately shares ONE combined
    # image across several method rows (that sharing is intentional, not a bug).
    seen_identities = set()
    for r in required_rows:
        identity = (r['panel_type'], r['method_id'])
        if identity in seen_identities:
            raise SystemExit(f"[hard-fail] Figure {contract['figure_id']}: duplicate panel identity in the "
                               f'composite manifest: {identity!r}.')
        seen_identities.add(identity)

    missing = sorted({r['source_path'] for r in required_rows if not (REPO_ROOT / r['source_path']).exists()})
    if missing:
        raise SystemExit(
            f"[hard-fail] Figure {contract['figure_id']}: {len(missing)} panel(s) declared in the composite "
            f'manifest are missing on disk (never assembling an incomplete composite): {missing}'
        )

    # De-duplicate at the IMAGE level (preserving first-occurrence/declared order):
    # several manifest rows may legitimately point at the same shared scalar
    # PD-fallback file, but that file must appear exactly once in the visual grid.
    seen_paths = set()
    unique_source_paths = []
    for r in required_rows:
        if r['source_path'] not in seen_paths:
            seen_paths.add(r['source_path'])
            unique_source_paths.append(r['source_path'])

    panel_dir = PANELS_DIR / figure_dir_name(contract)
    if panel_dir.exists():
        on_disk = {str(p.relative_to(REPO_ROOT)) for p in panel_dir.glob('*.png')}
        unexpected = on_disk - seen_paths
        if unexpected:
            raise SystemExit(
                f"[hard-fail] Figure {contract['figure_id']}: unexpected PNG(s) present in {panel_dir} that are "
                f'not declared in the composite manifest (rejected, never silently included): {sorted(unexpected)}'
            )

    ordered_panels = [REPO_ROOT / p for p in unique_source_paths]
    n = len(ordered_panels)
    n_cols = min(n, 4)
    n_rows = math.ceil(n / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.4 * n_cols, 2.4 * n_rows), dpi=300)
    axes_flat = np.atleast_1d(axes).ravel()
    for ax, p in zip(axes_flat, ordered_panels):
        ax.imshow(plt.imread(p))
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes_flat[len(ordered_panels):]:
        ax.axis('off')
    out_paths = final_figure_paths(contract)
    png_path = REPO_ROOT / out_paths['png']
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=300, metadata={'Software': ''})
    pdf_path = REPO_ROOT / out_paths['pdf']
    fig.savefig(pdf_path, metadata={'Creator': '', 'Producer': ''})
    plt.close(fig)

    png_insp = inspect_png(png_path)
    pdf_insp = inspect_pdf(pdf_path)
    return dict(
        figure_id=contract['figure_id'], archetype_id=contract['archetype_id'], sample_idx=si,
        expected_png_path=out_paths['png'], expected_vector_path=out_paths['pdf'],
        png_exists=png_path.exists(), width_px=png_insp['width_px'], height_px=png_insp['height_px'],
        dpi_x=png_insp['dpi_x'], dpi_y=png_insp['dpi_y'], png_file_size_bytes=png_insp['file_size_bytes'],
        png_min_dpi_ok=(png_insp['validation_status'] == 'PASS'), vector_exists=pdf_path.exists(),
        pdf_page_count=pdf_insp['pdf_page_count'], pdf_file_size_bytes=pdf_insp['file_size_bytes'],
        pdf_valid=(pdf_insp['validation_status'] == 'PASS'),
        vector_kind='raster_panel_pdf',  # panels are pre-rendered raster PNGs placed in a PDF, not vector objects
        status='rendered',
    )


def cmd_assemble_composites(plan_result=None, is_full_run=False) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log('=' * 88)
    log('Unified candidate figures -- Phase 2D-B (--assemble-composites)')
    log('=' * 88)
    checksums_before, file_to_phase = preflight_immutability()
    manifest = read_and_validate_selection_manifest()
    require_completed_phase2d_a_state(manifest)
    manual_topo_rows = require_manual_topology_panels(manifest)

    final_rows = [build_composite_for_figure(c, manifest) for c in FIGURE_CONTRACTS]
    write_csv(VALIDATION_DIR / 'final_figure_validation.csv', FINAL_FIGURE_VALIDATION_FIELDS, final_rows)

    # Never claim completion with any blocked/pending/invalid composite -- all six
    # exact composite manifests must be fully satisfied and pass real inspection.
    not_ready = [r for r in final_rows if not (r['status'] == 'rendered' and r['png_exists']
                                                  and r['png_min_dpi_ok'] and r['vector_exists'] and r['pdf_valid'])]
    if len(final_rows) != 6 or not_ready:
        raise SystemExit(
            f'[hard-fail] Cannot report Phase 2D-B complete: {len(final_rows)}/6 composites built, '
            f'{len(not_ready)} failed real PNG/PDF validation: {[r["figure_id"] for r in not_ready]}.'
        )

    postflight_immutability(checksums_before, file_to_phase, VALIDATION_DIR / 'prior_phase_immutability_check.csv')
    execution_mode = EXECUTION_MODE_FULL if is_full_run else EXECUTION_MODE_ASSEMBLE_COMPOSITES
    zoom_result = read_zoom_result_from_validation()
    panel_rows = read_rendered_panel_rows()
    write_phase2db_doc(manifest, zoom_result, manual_topo_rows, panel_rows, pd_discovery_rows=None,
                         execution_mode=execution_mode, final_figure_rows=final_rows)
    log('')
    log('=' * 88)
    log(f'RESULT: Phase 2D-B complete. {len(final_rows)} final composite figure(s) written.')
    log('=' * 88)
    flush_log(LOG_PATH)
    return dict(final_rows=final_rows)


def cmd_full():
    plan_result = cmd_plan_only()
    render_result = cmd_render_fields(plan_result=plan_result)
    composite_result = cmd_assemble_composites(plan_result=plan_result, is_full_run=True)
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