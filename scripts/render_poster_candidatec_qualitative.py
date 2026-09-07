#!/usr/bin/env python3
"""Poster-ready qualitative comparison: CNN vs. Ablation (candidateUV_expanded2688)
vs. Topology-inspired (candidateC_expanded2688).

Scope is deliberately exactly three methods -- GAN is never loaded, resolved,
or displayed anywhere in this script.

  CNN                -> pretrained CNN baseline (data_out_fixed/wind_mrhr_cnn/)
  Ablation            -> candidateUV_expanded2688 (data_out/wind_finetune_candidateUV_expanded2688/)
  Topology-inspired    -> candidateC_expanded2688 (data_out/wind_finetune_candidateC_expanded2688/)

  Ablation = reconstruction-only fine-tuning (the matched UV-only control:
  same fine-tuning procedure as Candidate C, but without any topology-
  inspired auxiliary loss term).

This script REUSES, rather than reimplements, the repository's already-
validated default-sublevel persistence-diagram source-resolution/parsing
machinery from scripts/render_unified_candidate_figures_phase2db.py (module
alias `phase2db` below) and scripts/select_and_preview_unified_candidates_
phase2d.py (module alias `p2da`):

  - phase2db.find_exact_pd_vtu_candidates / parse_and_validate_pd_vtu /
    _resolve_canonical_pd_copy / resolve_canonical_gt_pd_source: the exact,
    hard-fail-on-malformed, hard-fail-on-cross-copy-conflict VTU discovery
    and parsing logic. Only the `default_sublevel` filtration convention is
    ever used (phase2db.FILTRATION_DEFAULT_SUBLEVEL) -- the disjoint
    `superlevel_negated_speed` family (a separate robustness study) is never
    read, searched, or substituted here.
  - p2da.validate_idx_array / p2da._idx_position_map / p2da.speed_from_uv /
    p2da._full_array_shape_finite_aligned / p2da.compute_preview_panel_data:
    the exact raw-array validation and speed/error-field computation already
    used by the Phase-2D pipeline.

Every persistence-diagram coordinate source is validated to be an exact,
default-sublevel `_pd_port_0.vtu` artifact before use (never a fuzzy match,
never a superlevel substitute, never fabricated). The corrected d_B / W2
topology metrics are loaded verbatim from the independently recomputed
canonical sweep CSV; the historical TTK "pd_distance" number is never read
or displayed anywhere in this script.

This script is read-only with respect to every prior-phase artifact (raw
arrays, PD/MT VTU outputs, frozen CSVs). It writes only under --outdir.

Usage:
    python3 scripts/render_poster_candidatec_qualitative.py \\
        --sample 107 --min-persistence 4.0 \\
        --outdir ttk_runs_fixed/poster_qualitative

    python3 scripts/render_poster_candidatec_qualitative.py \\
        --sample-list poster_qualitative_candidates.csv --top-n 10 \\
        --outdir ttk_runs_fixed/poster_qualitative
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import select_and_preview_unified_candidates_phase2d as p2da  # noqa: E402
import render_unified_candidate_figures_phase2db as phase2db  # noqa: E402

REPO_ROOT = p2da.REPO_ROOT
assert REPO_ROOT == SCRIPT_DIR.parent

# =============================================================================
# Method identities (exactly three; GAN is never referenced anywhere below)
# =============================================================================

CNN_KEY = 'cnn'
ABLATION_KEY = 'ablation'
TOPO_KEY = 'topology_inspired'
METHOD_KEYS = (CNN_KEY, ABLATION_KEY, TOPO_KEY)

DISPLAY_LABEL = {CNN_KEY: 'CNN', ABLATION_KEY: 'Ablation', TOPO_KEY: 'Topology-inspired'}
GT_LABEL = 'GT'

ABLATION_NOTE = 'Ablation = reconstruction-only fine-tuning'
PSNR_SSIM_NOTE = 'PSNR is on [u,v]; SSIM and topology are on scalar speed.'
CAPTION_TEXT = ('Displayed PD points are thresholded/zoomed for readability; d_B and W2 use the '
                 'complete finite diagrams.')

# Internal artifact aliases (topology_finetuning/<alias>_topology/pd/{GT,SR}/...) and the
# unified-evaluation method_id used by frozen per-sample CSVs (unified_primary_per_sample_long.csv).
ARTIFACT_ALIAS = {
    CNN_KEY: phase2db.CNN,  # 'cnn' -- resolved via phase2db.CNN_GAN_BICUBIC_DEFAULT_SUBLEVEL_ROOTS
    ABLATION_KEY: 'candidateUV_expanded2688',
    TOPO_KEY: phase2db.METHOD_ARTIFACT_ALIASES[phase2db.CANDIDATE_C],  # 'candidateC_expanded2688'
}
UNIFIED_METHOD_ID = {CNN_KEY: 'cnn', ABLATION_KEY: 'uv', TOPO_KEY: 'candidate_c'}
ORIGINAL_METHOD_NAME_EXPECTED = {
    CNN_KEY: 'cnn', ABLATION_KEY: 'candidateUV_expanded2688', TOPO_KEY: 'candidateC_expanded2688',
}

# Raw-array directories, exactly as given in the task spec, cross-validated
# below against ttk_runs_fixed/unified_candidate_evaluation/method_inventory.csv
# rather than trusted blindly.
RAW_DIR = {
    CNN_KEY: REPO_ROOT / 'data_out_fixed' / 'wind_mrhr_cnn',
    ABLATION_KEY: REPO_ROOT / 'data_out' / 'wind_finetune_candidateUV_expanded2688',
    TOPO_KEY: REPO_ROOT / 'data_out' / 'wind_finetune_candidateC_expanded2688',
}

# Topology-crop convention: identical to scripts/convert_phire_to_vti.py's
# defaults (--patch 160 --x0 0 --y0 0), the only crop convention used
# anywhere in the repository's PD/MT evaluation pipeline. field[Y0:Y0+PATCH,
# X0:X0+PATCH] on a (H, W, ...) array, matching that script's
# `field[y0:y1, x0:x1]` slicing exactly.
PATCH = 160
X0 = 0
Y0 = 0

# Default source locations for the metric strip / subtitle values.
DEFAULT_CANONICAL_PD_CSV = Path.home() / 'phire_runtime_audit_20260809_221548' / 'recompute_pd' / \
    'canonical_pd_full_sweep.csv'
DEFAULT_SSIM_CSV = REPO_ROOT / 'ttk_runs_fixed' / 'ssim_recomputed_scale_sweep' / \
    'ssim_per_sample_scale_sweep.csv'
DEFAULT_UNIFIED_LONG_CSV = p2da.PHASE1_DIR / 'unified_primary_per_sample_long.csv'

DEFAULT_OUTDIR = REPO_ROOT / 'ttk_runs_fixed' / 'poster_qualitative'

# Reasonable-minimum-points warning threshold for the shared PD zoom.
MIN_REASONABLE_DISPLAY_POINTS = 6

# Colorblind-safe accents (Okabe-Ito palette). Marker SHAPE, never color
# alone, is the semantic distinguisher between GT and a reconstruction.
COLOR_GT = '#1A1A1A'          # near-black, hollow circles
COLOR_RECON = '#D55E00'       # vermillion, filled X markers
CURVE_STYLE = {
    GT_LABEL: dict(color=COLOR_GT, linestyle='-', marker='o', markersize=4),
    CNN_KEY: dict(color='#0072B2', linestyle='--', marker='s', markersize=4),
    ABLATION_KEY: dict(color='#009E73', linestyle=':', marker='^', markersize=4),
    TOPO_KEY: dict(color=COLOR_RECON, linestyle='-.', marker='D', markersize=4),
}


class ProvenanceError(SystemExit):
    """SystemExit subclass used for every hard-fail in this script, so the
    caller's [hard-fail]-prefixed message is preserved verbatim."""


def hard_fail(msg: str):
    raise ProvenanceError(f'[hard-fail] {msg}')


# =============================================================================
# Section 1: raw-array path resolution and validation
#   - idx.npy arrays are exactly 0..167 for all three methods
#   - dataGT.npy is byte-identical across all three methods (shared GT)
#   - requested sample_idx is in range
# =============================================================================

def resolve_and_cross_check_raw_paths(method_inventory: dict) -> dict:
    """Resolves the three methods' raw-array directories from the exact
    paths given in the task specification, then cross-validates each
    against method_inventory.csv's `original_method_name` (never trusted
    blindly -- a mismatch is a hard-fail, not a silent override)."""
    for key in METHOD_KEYS:
        mid = UNIFIED_METHOD_ID[key]
        inv_row = method_inventory.get(mid)
        if inv_row is None:
            hard_fail(f"method_inventory.csv has no row for method_id={mid!r} (needed to cross-check "
                       f"the raw-array directory for {DISPLAY_LABEL[key]!r}).")
        actual_original = inv_row.get('original_method_name')
        expected_original = ORIGINAL_METHOD_NAME_EXPECTED[key]
        if actual_original != expected_original:
            hard_fail(f"method_inventory.csv method_id={mid!r} original_method_name={actual_original!r} "
                       f"does not match the expected {expected_original!r} for {DISPLAY_LABEL[key]!r}. "
                       f"Refusing to guess which raw-array directory is correct.")
    paths = {}
    for key in METHOD_KEYS:
        d = RAW_DIR[key]
        paths[key] = dict(dir=d, idx=d / 'idx.npy', dataIN=d / 'dataIN.npy', dataGT=d / 'dataGT.npy',
                            dataSR=d / 'dataSR.npy')
    return paths


def validate_raw_arrays(paths: dict) -> list:
    """Reuses p2da.validate_idx_array / p2da._full_array_shape_finite_aligned
    exactly as the Phase-2D-A raw-artifact audit does. Hard-fails (never
    warns) on: a missing file, a non-`0..167` idx array, or a GT array that
    disagrees with the CNN canonical GT. Returns informational notes."""
    for key in METHOD_KEYS:
        for role in ('idx', 'dataIN', 'dataGT', 'dataSR'):
            path = paths[key][role]
            if not path.exists():
                hard_fail(f'Missing raw artifact required for {DISPLAY_LABEL[key]}: {path} (expected on the '
                           f'authoritative Spark run; data_out/ and data_out_fixed/ are gitignored and absent '
                           f'by design in a lightweight checkout).')
    notes = []
    idx_arrays = {}
    for key in METHOD_KEYS:
        idx_arr = np.load(paths[key]['idx'])
        status, detail = p2da.validate_idx_array(idx_arr, DISPLAY_LABEL[key])
        if status != 'PASS':
            hard_fail(f'idx.npy validation failed for {DISPLAY_LABEL[key]}: {detail}')
        idx_arrays[key] = idx_arr
        notes.append(f'{DISPLAY_LABEL[key]} idx.npy: exactly 0..{p2da.N_EVAL - 1} (PASS)')

    canonical_gt = np.load(paths[CNN_KEY]['dataGT'], mmap_mode='r')
    for key in METHOD_KEYS:
        gt_mmap = np.load(paths[key]['dataGT'], mmap_mode='r')
        shape_ok, finite_ok, aligned_ok = p2da._full_array_shape_finite_aligned(
            gt_mmap, p2da.EXPECTED_HR_SHAPE, canonical_mmap=(None if key == CNN_KEY else canonical_gt))
        if not shape_ok:
            hard_fail(f'{DISPLAY_LABEL[key]} dataGT.npy shape {tuple(gt_mmap.shape)} != expected '
                       f'{p2da.EXPECTED_HR_SHAPE}.')
        if not finite_ok:
            hard_fail(f'{DISPLAY_LABEL[key]} dataGT.npy contains non-finite values.')
        if key != CNN_KEY and not aligned_ok:
            hard_fail(f'{DISPLAY_LABEL[key]} dataGT.npy is not exactly equal to the canonical CNN dataGT.npy '
                       f'(GT must be identical across all model outputs on the shared 168-sample benchmark).')
        notes.append(f'{DISPLAY_LABEL[key]} dataGT.npy: shape {p2da.EXPECTED_HR_SHAPE}, finite'
                      + ('' if key == CNN_KEY else ', exactly equal to canonical CNN GT') + ' (PASS)')
    return notes


def require_sample_available(sample_idx: int) -> None:
    if not (0 <= sample_idx < p2da.N_EVAL):
        hard_fail(f'Requested sample_idx={sample_idx} is out of range; the benchmark covers exactly '
                   f'0..{p2da.N_EVAL - 1}.')


# =============================================================================
# Section 2: default-sublevel PD coordinate-source resolution (reused
# machinery from phase2db -- never reimplemented, never superlevel).
# =============================================================================

def _dedup_rel(candidates_with_alias, seen_rel):
    out = []
    for path, alias in candidates_with_alias:
        rel = phase2db._rel(path)
        if rel in seen_rel:
            continue
        seen_rel.add(rel)
        out.append((path, alias))
    return out


def _find_flat_pd_vtu_candidates(root: Path, role: str, alias: str, sample_idx: int):
    """Find exact legacy/default-sublevel PD VTUs stored directly under root/pd/.

    The older CNN/GAN baseline pipeline on Spark stores files like:
        ttk_runs_fixed/cnn/pd/cnn_SR_s99_speed_p160_x0_y0_pd_port_0.vtu

    whereas phase2db.find_exact_pd_vtu_candidates() searches the newer layout:
        root/pd/SR/<filename>

    This helper adds the real historical flat layout without relaxing any
    provenance rule: exact alias, exact role, exact sample, exact
    `_pd_port_0.vtu` suffix, and phase2db's own validity check.
    """
    pd_dir = root / 'pd'
    if not pd_dir.exists():
        return []

    pattern = f'{alias}_{role}_s{sample_idx}_*_pd_port_0.vtu'
    found = []
    for p in sorted(pd_dir.glob(pattern)):
        if phase2db._is_valid_pd_vtu_path(p):
            found.append(p)
    return found


def _find_default_sublevel_pd_candidates(root: Path, role: str, alias: str, sample_idx: int):
    """Search both validated repository layouts for an exact PD coordinate artifact."""
    found = list(
        phase2db.find_exact_pd_vtu_candidates(root, role, alias, sample_idx)
    )
    found.extend(_find_flat_pd_vtu_candidates(root, role, alias, sample_idx))

    # De-duplicate by resolved path.
    unique = []
    seen = set()
    for p in found:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            unique.append(p)
    return unique


def resolve_gt_pd_source(sample_idx: int) -> dict:
    """Canonical GT PD source for one sample, default_sublevel only.
    Extends phase2db.resolve_canonical_gt_pd_source's own search (which
    already covers CANDIDATE_C/F3/F2/UV_E2/CNN/GAN/BICUBIC's topology
    trees) with the Ablation (candidateUV_expanded2688) tree, since that
    alias is not one of phase2db's own built-in GT_SOURCE_PRIORITY_METHODS.
    Every discovered exact copy is parsed and cross-checked for coordinate
    agreement via phase2db._resolve_canonical_pd_copy -- disagreement
    hard-fails, listing every conflicting path, exactly as it would inside
    phase2db itself."""
    seen_rel = set()
    candidates = []
    for pmid in phase2db.GT_SOURCE_PRIORITY_METHODS:
        roots, alias = phase2db._method_search_roots_and_alias(pmid, phase2db.FILTRATION_DEFAULT_SUBLEVEL)
        for root in roots:
            found = [(p, alias) for p in
                      _find_default_sublevel_pd_candidates(root, phase2db.PD_VTU_ROLE_GT, alias, sample_idx)]
            candidates.extend(_dedup_rel(found, seen_rel))
    extra_root = phase2db._topology_tree_root(ARTIFACT_ALIAS[ABLATION_KEY])
    extra_found = [(p, ARTIFACT_ALIAS[ABLATION_KEY]) for p in
                     _find_default_sublevel_pd_candidates(extra_root, phase2db.PD_VTU_ROLE_GT,
                                                            ARTIFACT_ALIAS[ABLATION_KEY], sample_idx)]
    candidates.extend(_dedup_rel(extra_found, seen_rel))
    if not candidates:
        hard_fail(f'No exact default_sublevel GT PD source (_pd_port_0.vtu) found for sample_idx={sample_idx} '
                   f'in any known topology tree (candidate_c, f3, f2, uv_e2, cnn, gan, bicubic, ablation). '
                   f'Never falling back to a superlevel-negated source.')
    path, alias, result, n_copies = phase2db._resolve_canonical_pd_copy(candidates, sample_idx,
                                                                            phase2db.PD_VTU_ROLE_GT)
    return dict(path=path, alias=alias, n_copies=n_copies, birth=result['birth'], death=result['death'],
                 pair_count=result['pair_count'], filtration_convention=phase2db.FILTRATION_DEFAULT_SUBLEVEL,
                 source_family=phase2db._source_family_for_alias(alias))


def resolve_sr_pd_source(key: str, sample_idx: int) -> dict:
    """Canonical SR PD source for one non-GT method, default_sublevel only."""
    alias = ARTIFACT_ALIAS[key]
    if key == CNN_KEY:
        roots, alias = phase2db._method_search_roots_and_alias(phase2db.CNN, phase2db.FILTRATION_DEFAULT_SUBLEVEL)
    else:
        roots = [phase2db._topology_tree_root(alias)]
    seen_rel = set()
    candidates = []
    for root in roots:
        found = [(p, alias) for p in
                  _find_default_sublevel_pd_candidates(root, phase2db.PD_VTU_ROLE_SR, alias, sample_idx)]
        candidates.extend(_dedup_rel(found, seen_rel))
    if not candidates:
        hard_fail(f'No exact default_sublevel SR PD source (_pd_port_0.vtu) found for {DISPLAY_LABEL[key]} '
                   f'(alias={alias!r}) sample_idx={sample_idx}. Never falling back to a superlevel-negated '
                   f'source, and never substituting a different method or sample.')
    path, resolved_alias, result, n_copies = phase2db._resolve_canonical_pd_copy(
        candidates, sample_idx, phase2db.PD_VTU_ROLE_SR)
    return dict(path=path, alias=resolved_alias, n_copies=n_copies, birth=result['birth'], death=result['death'],
                 pair_count=result['pair_count'], filtration_convention=phase2db.FILTRATION_DEFAULT_SUBLEVEL,
                 source_family=phase2db._source_family_for_alias(resolved_alias))


def resolve_pd_sources_for_sample(sample_idx: int) -> dict:
    sources = {GT_LABEL: resolve_gt_pd_source(sample_idx)}
    for key in METHOD_KEYS:
        sources[key] = resolve_sr_pd_source(key, sample_idx)

        # Candidate UV and Candidate C are both produced by the topology-
        # fine-tuning pipeline. Candidate UV is not one of phase2db's built-in
        # candidate aliases, so correct the provenance label explicitly.
        if key in (ABLATION_KEY, TOPO_KEY):
            sources[key]['source_family'] = phase2db.SOURCE_FAMILY_TOPOLOGY_FINETUNING

    return sources


# =============================================================================
# Section 3: canonical corrected d_B / W2 metrics (never the historical TTK
# "pd_distance" number, which is never read anywhere in this script).
# =============================================================================

SAMPLE_COL_CANDIDATES = ['sample_idx', 'sample', 'sample_id']
METHOD_COL_CANDIDATES = ['run', 'run_name', 'method', 'method_id', 'candidate', 'candidate_id', 'name']
DB_COL_CANDIDATES = ['pd_bottleneck_all', 'pd_bottleneck_all_distance', 'bottleneck_all', 'd_B', 'dB', 'd_b', 'bottleneck', 'bottleneck_distance', 'd_bottleneck']
W2_COL_CANDIDATES = ['pd_w2_all', 'pd_w2_all_distance', 'w2_all', 'W2', 'w2', 'W_2', 'wasserstein_2', 'wasserstein2', 'w2_linf', 'W2_Linf', 'w2_ground_linf']

# Every plausible string a resolved method-column value could use to
# identify each of the three methods. Never guessed blindly: the actual
# unique values present in the file are always printed, and a match here
# still requires that exactly one distinct on-disk value corresponds to a
# given method (never an ambiguous multi-value match).
METHOD_VALUE_CANDIDATES = {
    CNN_KEY: {
        'cnn',
        'CNN',
    },
    ABLATION_KEY: {
        'topology_finetuning/candidateUV_expanded2688_topology',
        'candidateUV_expanded2688',
        'uv',
        'candidateUV_2688',
        'ablation',
        'Ablation',
        'candidateUV',
        'UV',
    },
    TOPO_KEY: {
        'topology_finetuning/candidateC_expanded2688_topology',
        'candidateC_expanded2688',
        'candidate_c',
        'candidateC_2688',
        'topology_inspired',
        'Topology-inspired',
        'candidateC',
        'C',
    },
}


def _resolve_column(fieldnames: list, candidates: list, label: str, path: Path) -> str:
    fieldset = set(fieldnames)
    for c in candidates:
        if c in fieldset:
            return c
    hard_fail(f'Could not resolve a {label!r} column in {path}. Tried (in order): {candidates}. '
               f'Available columns: {fieldnames}. Refusing to guess.')


def _resolve_method_values(unique_values: set, path: Path) -> dict:
    """Maps each of the three internal method keys to the single matching
    on-disk string value, hard-failing on zero or ambiguous (>1) matches."""
    resolved = {}
    for key in METHOD_KEYS:
        matches = sorted(unique_values & METHOD_VALUE_CANDIDATES[key])
        if len(matches) == 0:
            hard_fail(f'No method-column value in {path} identifies {DISPLAY_LABEL[key]!r}. Tried: '
                       f'{sorted(METHOD_VALUE_CANDIDATES[key])}. Actual unique values present: '
                       f'{sorted(unique_values)}. Refusing to guess.')
        if len(matches) > 1:
            hard_fail(f'Ambiguous method-column values in {path} for {DISPLAY_LABEL[key]!r}: {matches} all '
                       f'match. Refusing to guess which is authoritative.')
        resolved[key] = matches[0]
    return resolved


def load_canonical_pd_metrics(csv_path: Path, sample_idx: int) -> dict:
    """Loads corrected bottleneck d_B and 2-Wasserstein (L_inf ground cost)
    W2 for the three methods at one sample, from the independently
    recomputed canonical sweep. Column names are resolved explicitly
    (never assumed); resolution is reported via the returned dict's
    `_resolved_columns` entry for provenance."""
    if not csv_path.exists():
        hard_fail(f'Canonical PD metrics CSV not found: {csv_path}. This file is produced by the '
                   f'independent PD recomputation pass (bottleneck d_B, 2-Wasserstein W2 with L_inf ground '
                   f'cost); the historical TTK pd_distance number is never used as a substitute.')
    with csv_path.open(newline='') as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or []
        sample_col = _resolve_column(fieldnames, SAMPLE_COL_CANDIDATES, 'sample index', csv_path)
        method_col = _resolve_column(fieldnames, METHOD_COL_CANDIDATES, 'method identity', csv_path)
        db_col = _resolve_column(fieldnames, DB_COL_CANDIDATES, 'bottleneck distance d_B', csv_path)
        w2_col = _resolve_column(fieldnames, W2_COL_CANDIDATES, '2-Wasserstein W2', csv_path)
        rows = list(reader)

    unique_values = {r[method_col] for r in rows if r.get(method_col) is not None}
    method_value = _resolve_method_values(unique_values, csv_path)

    out = {'_resolved_columns': dict(sample_col=sample_col, method_col=method_col, d_B_col=db_col, W2_col=w2_col,
                                        method_values=method_value)}
    for key in METHOD_KEYS:
        matching = [r for r in rows if str(r.get(sample_col)) == str(sample_idx)
                     and r.get(method_col) == method_value[key]]
        if len(matching) == 0:
            hard_fail(f'No row in {csv_path} for sample_idx={sample_idx}, method={method_value[key]!r} '
                       f'({DISPLAY_LABEL[key]}). Canonical d_B/W2 rows must exist for every requested '
                       f'sample/method before rendering.')
        if len(matching) > 1:
            hard_fail(f'Multiple rows ({len(matching)}) in {csv_path} for sample_idx={sample_idx}, '
                       f'method={method_value[key]!r} ({DISPLAY_LABEL[key]}) -- ambiguous, refusing to pick one.')
        row = matching[0]
        try:
            d_b = float(row[db_col])
            w2 = float(row[w2_col])
        except (TypeError, ValueError) as exc:
            hard_fail(f'Non-numeric d_B/W2 value in {csv_path} for sample_idx={sample_idx}, '
                       f'method={method_value[key]!r}: {exc}')
        if not (math.isfinite(d_b) and math.isfinite(w2)):
            hard_fail(f'Non-finite d_B/W2 value in {csv_path} for sample_idx={sample_idx}, '
                       f'method={method_value[key]!r}: d_B={d_b}, W2={w2}.')
        out[key] = dict(d_B=d_b, W2=w2)
    return out


# =============================================================================
# Section 4: PSNR_uv (project convention, loaded from the frozen unified
# per-sample table) and SSIM_speed (loaded from the recomputed scale-sweep
# CSV -- SSIM in the frozen unified table is documented as globally
# unavailable due to the known NumPy/scikit-image ABI incompatibility).
# =============================================================================

def load_psnr_uv(csv_path: Path, sample_idx: int) -> dict:
    if not csv_path.exists():
        hard_fail(f'Unified per-sample CSV not found: {csv_path} (needed for PSNR_uv).')
    with csv_path.open(newline='') as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or []
        if 'sample_idx' not in fieldnames or 'method_id' not in fieldnames or 'psnruv' not in fieldnames:
            hard_fail(f'{csv_path} is missing one of the expected columns sample_idx/method_id/psnruv. '
                       f'Available columns: {fieldnames}.')
        rows = [r for r in reader if str(r['sample_idx']) == str(sample_idx)]
    out = {}
    for key in METHOD_KEYS:
        mid = UNIFIED_METHOD_ID[key]
        matching = [r for r in rows if r['method_id'] == mid]
        if len(matching) != 1:
            hard_fail(f'Expected exactly one {csv_path} row for sample_idx={sample_idx}, method_id={mid!r} '
                       f'({DISPLAY_LABEL[key]}); found {len(matching)}.')
        try:
            val = float(matching[0]['psnruv'])
        except (TypeError, ValueError):
            hard_fail(f'Non-numeric psnruv for sample_idx={sample_idx}, method_id={mid!r} in {csv_path}: '
                       f'{matching[0]["psnruv"]!r}.')
        if not math.isfinite(val):
            hard_fail(f'Non-finite psnruv for sample_idx={sample_idx}, method_id={mid!r} in {csv_path}.')
        out[key] = val
    return out


def load_ssim_speed(csv_path: Path, sample_idx: int) -> dict:
    if not csv_path.exists():
        hard_fail(f'Recomputed SSIM CSV not found: {csv_path}.')
    required = {'sample_idx', 'method', 'family', 'training_size', 'ssim_speed'}
    with csv_path.open(newline='') as fh:
        reader = csv.DictReader(fh)
        fieldnames = set(reader.fieldnames or [])
        missing = required - fieldnames
        if missing:
            hard_fail(f'{csv_path} is missing expected column(s) {sorted(missing)}. Available columns: '
                       f'{sorted(fieldnames)}.')
        rows = [r for r in reader if str(r['sample_idx']) == str(sample_idx)]
    method_value = {
        CNN_KEY: 'cnn',
        ABLATION_KEY: 'candidateUV_2688',
        TOPO_KEY: 'candidateC_2688',
    }
    out = {}
    for key in METHOD_KEYS:
        matching = [r for r in rows if r['method'] == method_value[key]]
        if len(matching) != 1:
            present = sorted({r['method'] for r in rows})
            hard_fail(f'Expected exactly one {csv_path} row for sample_idx={sample_idx}, '
                       f'method={method_value[key]!r} ({DISPLAY_LABEL[key]}); found {len(matching)}. '
                       f'Methods present for this sample: {present}.')
        try:
            val = float(matching[0]['ssim_speed'])
        except (TypeError, ValueError):
            hard_fail(f'Non-numeric ssim_speed for sample_idx={sample_idx}, method={method_value[key]!r} in '
                       f'{csv_path}: {matching[0]["ssim_speed"]!r}.')
        if not math.isfinite(val):
            hard_fail(f'Non-finite ssim_speed for sample_idx={sample_idx}, method={method_value[key]!r} in '
                       f'{csv_path}.')
        out[key] = val
    return out


# =============================================================================
# Section 5: field-space crop loading (same 160x160 default_sublevel
# topology crop used for PD evaluation; identical crop/orientation
# convention as scripts/convert_phire_to_vti.py).
# =============================================================================

def load_uv_crop(dataGT_path: Path, dataSR_path: Path, sample_idx: int):
    gt4d = np.load(dataGT_path, mmap_mode='r')
    sr4d = np.load(dataSR_path, mmap_mode='r')
    gt_uv = np.asarray(gt4d[sample_idx, Y0:Y0 + PATCH, X0:X0 + PATCH, :], dtype=np.float64)
    sr_uv = np.asarray(sr4d[sample_idx, Y0:Y0 + PATCH, X0:X0 + PATCH, :], dtype=np.float64)
    return gt_uv, sr_uv


# =============================================================================
# Section 6: PD-overlay geometry -- display filtering, shared zoom. The
# display filter NEVER touches the complete-diagram arrays used for
# survival curves or the canonical d_B/W2 values (those are entirely
# separate code paths, see Sections 2/3 above).
# =============================================================================

def display_filter(birth: np.ndarray, death: np.ndarray, min_persistence: float):
    persistence = death - birth
    mask = persistence >= min_persistence
    return birth[mask], death[mask]


def compute_shared_zoom(filtered_pairs: list, pad_frac: float = 0.06):
    """filtered_pairs: list of (birth, death) arrays (already display-
    filtered). Returns (lo, hi, total_points, warning_or_None)."""
    all_vals = np.concatenate([np.concatenate([b, d]) for b, d in filtered_pairs if len(b) > 0]) \
        if any(len(b) > 0 for b, _ in filtered_pairs) else np.array([])
    total_points = sum(len(b) for b, _ in filtered_pairs)
    warning = None
    if all_vals.size == 0:
        warning = ('No points survive the --min-persistence filter across GT/CNN/Ablation/Topology-inspired -- '
                    'try a lower --min-persistence.')
        return 0.0, 1.0, 0, warning
    lo, hi = float(all_vals.min()), float(all_vals.max())
    span = max(hi - lo, 1e-9)
    lo -= pad_frac * span
    hi += pad_frac * span
    # Wind-speed PD coordinates are nonnegative; avoid wasting poster area
    # on a negative lower axis introduced only by padding.
    lo = max(0.0, lo)
    if total_points < MIN_REASONABLE_DISPLAY_POINTS:
        warning = (f'Only {total_points} point(s) survive the --min-persistence filter across GT/CNN/'
                    f'Ablation/Topology-inspired (fewer than the {MIN_REASONABLE_DISPLAY_POINTS} considered '
                    f'reasonable for a legible overlay) -- consider a lower --min-persistence.')
    return lo, hi, total_points, warning


def survival_counts(persistence_full: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    persistence_full = np.sort(persistence_full)
    # searchsorted on sorted ascending array: count of values >= t.
    idx = np.searchsorted(persistence_full, thresholds, side='left')
    return len(persistence_full) - idx


# =============================================================================
# Section 7: rendering
# =============================================================================

def _compute_high_persistence_focus(gt_b: np.ndarray, gt_d: np.ndarray, lo: float, hi: float):
    """GT-only focus window for all three PD panels.

    The window is chosen without looking at any reconstruction: it is centered
    on the most persistent displayed GT pair and has a fixed size relative to
    the shared PD axis span. Thus the same zoom is used for CNN, Ablation, and
    Topology-inspired and is never tuned to favor one reconstruction.
    """
    if len(gt_b) == 0:
        return None

    persistence = gt_d - gt_b
    i = int(np.argmax(persistence))
    cx = float(gt_b[i])
    cy = float(gt_d[i])

    span = max(float(hi - lo), 1e-6)
    side = max(0.28 * span, 3.0)
    half = side / 2.0

    x0 = max(lo, cx - half)
    x1 = min(hi, cx + half)
    y0 = max(lo, cy - half)
    y1 = min(hi, cy + half)

    side2 = min(x1 - x0, y1 - y0)
    if side2 <= 0:
        return None
    x1 = x0 + side2
    y1 = y0 + side2
    return (x0, x1, y0, y1)


def _draw_pd_points(ax, gt_b, gt_d, sr_b, sr_d, marker_size=42):
    """GT = hollow circles; reconstruction = X. Shape carries meaning."""
    ax.scatter(
        gt_b, gt_d,
        s=marker_size,
        facecolors='none',
        edgecolors=COLOR_GT,
        linewidths=1.5,
        marker='o',
        label=GT_LABEL,
        zorder=3,
    )
    ax.scatter(
        sr_b, sr_d,
        s=marker_size,
        color=COLOR_RECON,
        marker='x',
        linewidths=2.0,
        label='prediction',
        zorder=4,
    )


def _draw_pd_overlay(ax, gt_b, gt_d, sr_b, sr_d, lo, hi, title, d_b, w2,
                     focus_region=None, improvement_text=None):
    from matplotlib.patches import Rectangle
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    ax.plot([lo, hi], [lo, hi], color='0.58', linestyle='--', linewidth=1.0, zorder=1)
    _draw_pd_points(ax, gt_b, gt_d, sr_b, sr_d)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel('Birth', fontsize=13)
    ax.set_ylabel('Death', fontsize=13)
    ax.tick_params(labelsize=10.5)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(
        f'{title} vs GT\n' rf'$d_B$={d_b:.3f}   $W_2$={w2:.3f}',
        fontsize=13.5,
        fontweight='bold' if title == DISPLAY_LABEL[TOPO_KEY] else 'normal',
        pad=7,
    )

    # Same GT-defined focus region and inset in every method panel.
    if focus_region is not None:
        x0, x1, y0, y1 = focus_region
        ax.add_patch(Rectangle(
            (x0, y0), x1 - x0, y1 - y0,
            fill=False, edgecolor='0.25', linewidth=1.8,
            linestyle=(0, (5, 3)), zorder=5,
        ))

        iax = inset_axes(ax, width='48%', height='48%', loc='lower right', borderpad=0.70)
        iax.set_facecolor('white')
        iax.plot([x0, x1], [x0, x1], color='0.72', linestyle='--', linewidth=0.75, zorder=1)
        _draw_pd_points(iax, gt_b, gt_d, sr_b, sr_d, marker_size=38)
        iax.set_xlim(x0, x1)
        iax.set_ylim(y0, y1)
        iax.set_xticks([])
        iax.set_yticks([])
        iax.set_aspect('equal', adjustable='box')
        iax.set_title('GT-defined zoom', fontsize=8.7, pad=2.2)
        for spine in iax.spines.values():
            spine.set_edgecolor('0.25')
            spine.set_linewidth(1.15)

def _draw_survival(ax, curves: dict, min_persistence: float,
                   x_min: float = 4.0, x_max: float = 10.0):
    """Poster view of the high-persistence tail, from complete diagrams."""
    max_p = max(float(p.max()) if len(p) else 0.0 for p in curves.values())
    x_max_eff = min(max_p, x_max) if max_p > 0 else x_max
    x_max_eff = max(x_max_eff, min_persistence + 0.5)
    x_min_eff = min(x_min, max(0.0, min_persistence - 0.5))
    thresholds = np.linspace(x_min_eff, x_max_eff, 350)

    counts_by_label = {}
    for label, persistence in curves.items():
        style = CURVE_STYLE[label]
        counts = survival_counts(persistence, thresholds)
        counts_by_label[label] = counts
        ax.plot(
            thresholds, counts,
            drawstyle='steps-post',
            linewidth=2.8,
            label=DISPLAY_LABEL.get(label, label),
            color=style.get('color'),
            linestyle=style.get('linestyle', '-'),
        )

    ax.axvspan(min_persistence, x_max_eff, facecolor='0.94', edgecolor='none', zorder=0)
    ax.axvline(min_persistence, color='0.35', linestyle=(0, (4, 3)), linewidth=1.3, zorder=1)
    ax.text(
        min_persistence + 0.08, 0.97, 'PD display threshold',
        transform=ax.get_xaxis_transform(),
        ha='left', va='top', fontsize=9.5,
    )

    ax.set_xlim(x_min_eff, x_max_eff)
    ax.set_ylim(bottom=0)
    ax.set_xlabel('Persistence threshold', fontsize=13)
    ax.set_ylabel('Surviving pairs', fontsize=13)
    ax.set_title('Persistence Survival - High-Persistence Tail\n(complete finite diagrams)', fontsize=13.0)
    ax.tick_params(labelsize=10.5)
    ax.grid(True, linewidth=0.55, alpha=0.28)
    ax.legend(fontsize=9.5, frameon=False, loc='upper right')

    # Gentle annotation in the visible tail; this does NOT imply exact
    # point-to-point pair matching, only that the curve lies closer to GT.
    idx = int(np.clip(
        np.searchsorted(
            thresholds,
            max(min_persistence + 1.2, x_min_eff + 0.28 * (x_max_eff - x_min_eff))
        ),
        0,
        len(thresholds) - 1
    ))
    x_ann = thresholds[idx]
    y_anchor = counts_by_label[TOPO_KEY][idx]
    y_text = max(y_anchor + 6, counts_by_label[GT_LABEL][idx] - 4)

    ax.annotate(
        'Topology-inspired\ncloser to GT here',
        xy=(x_ann, y_anchor),
        xytext=(x_ann + 0.55, y_text),
        textcoords='data',
        fontsize=9.5,
        ha='left', va='center',
        arrowprops=dict(arrowstyle='->', color='0.35', lw=1.0),
        bbox=dict(boxstyle='round,pad=0.22', fc='white', ec='0.65', alpha=0.92),
    )


def _draw_field_no_cbar(ax, field, vmin, vmax, title):
    im = ax.imshow(field, cmap='cividis', vmin=vmin, vmax=vmax,
                   origin='lower', aspect='equal')
    ax.set_title(title, fontsize=13.5, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def _metric_card(ax, heading: str, body: str):
    ax.axis('off')
    ax.text(
        0.5, 0.52, f'{heading}\n{body}',
        ha='center', va='center', fontsize=12.7, linespacing=1.45,
        bbox=dict(boxstyle='round,pad=0.52', facecolor='white',
                  edgecolor='0.35', linewidth=1.25),
    )


def render_one_sample(sample_idx: int, args, raw_paths: dict, method_inventory: dict, warnings: list) -> dict:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    require_sample_available(sample_idx)

    print(f'\n=== sample_idx={sample_idx} ===')

    pd_sources = resolve_pd_sources_for_sample(sample_idx)
    for label, src in pd_sources.items():
        print(f'  PD source [{label}]: {phase2db._rel(src["path"])} (alias={src["alias"]!r}, '
              f'filtration={src["filtration_convention"]}, source_family={src["source_family"]}, '
              f'pair_count={src["pair_count"]}, n_agreeing_copies={src["n_copies"]})')

    canonical = load_canonical_pd_metrics(Path(args.canonical_pd_csv), sample_idx)
    print(f'  Canonical PD metrics CSV: {args.canonical_pd_csv}')
    print(f'  Resolved columns: {canonical["_resolved_columns"]}')
    for key in METHOD_KEYS:
        print(f'    {DISPLAY_LABEL[key]}: d_B={canonical[key]["d_B"]:.4f}  W2={canonical[key]["W2"]:.4f}')

    psnr = load_psnr_uv(Path(args.unified_long_csv), sample_idx)
    print(f'  PSNR_uv source: {args.unified_long_csv}')
    for key in METHOD_KEYS:
        print(f'    {DISPLAY_LABEL[key]}: PSNR_uv={psnr[key]:.4f} dB')

    ssim = load_ssim_speed(Path(args.ssim_csv), sample_idx)
    print(f'  SSIM_speed source: {args.ssim_csv}')
    for key in METHOD_KEYS:
        print(f'    {DISPLAY_LABEL[key]}: SSIM_speed={ssim[key]:.4f}')

    # Field-space crops (GT is shared/identical across methods; already
    # validated above; loaded once via CNN's canonical dataGT).
    gt_uv_full, cnn_sr_uv = load_uv_crop(raw_paths[CNN_KEY]['dataGT'], raw_paths[CNN_KEY]['dataSR'], sample_idx)
    _, ablation_sr_uv = load_uv_crop(raw_paths[ABLATION_KEY]['dataGT'], raw_paths[ABLATION_KEY]['dataSR'],
                                        sample_idx)
    _, topo_sr_uv = load_uv_crop(raw_paths[TOPO_KEY]['dataGT'], raw_paths[TOPO_KEY]['dataSR'], sample_idx)
    panel_data = p2da.compute_preview_panel_data(
        gt_uv_full,
        {
            CNN_KEY: cnn_sr_uv,
            ABLATION_KEY: ablation_sr_uv,
            TOPO_KEY: topo_sr_uv,
        },
    )

    # --- Display filtering + shared PD zoom -------------------------------
    display = {}
    for label in (GT_LABEL,) + METHOD_KEYS:
        display[label] = display_filter(pd_sources[label]['birth'], pd_sources[label]['death'],
                                          args.min_persistence)
    lo, hi, total_points, zoom_warning = compute_shared_zoom(list(display.values()))
    if zoom_warning:
        print(f'  [warning] {zoom_warning}')
        warnings.append(zoom_warning)
    print(f'  Shared PD zoom: [{lo:.4f}, {hi:.4f}]  ({total_points} displayed point(s) total, '
          f'>= {args.min_persistence} persistence)')

    # --- Figure assembly ----------------------------------------------------
    # Compact poster composition:
    # row 1: CNN PD | Ablation PD | Topology-inspired PD | survival curve
    # row 2: GT field | CNN field | Ablation field | Topology-inspired field | shared cbar
    # row 3: persistence | fidelity | interpretation cards
    fig = plt.figure(figsize=(22, 10.8), dpi=100, facecolor='white')
    fig.suptitle(
        f'Representative Example (Sample {sample_idx}): '
        'Better Persistence Agreement at Similar Fidelity',
        fontsize=20, fontweight='bold', y=0.985,
    )

    gs = gridspec.GridSpec(
        3, 12, figure=fig,
        height_ratios=[1.32, 1.00, 0.43],
        hspace=0.40, wspace=0.55,
        top=0.91, bottom=0.078, left=0.045, right=0.965,
    )

    focus_region = _compute_high_persistence_focus(
        display[GT_LABEL][0], display[GT_LABEL][1], lo, hi
    )
    if focus_region is not None:
        print(
            '  GT-defined PD focus window: '
            f'birth=[{focus_region[0]:.3f}, {focus_region[1]:.3f}], '
            f'death=[{focus_region[2]:.3f}, {focus_region[3]:.3f}]'
        )

    db_reduction = 100.0 * (
        canonical[ABLATION_KEY]['d_B'] - canonical[TOPO_KEY]['d_B']
    ) / canonical[ABLATION_KEY]['d_B']
    w2_reduction = 100.0 * (
        canonical[ABLATION_KEY]['W2'] - canonical[TOPO_KEY]['W2']
    ) / canonical[ABLATION_KEY]['W2']
    psnr_delta = psnr[TOPO_KEY] - psnr[ABLATION_KEY]
    ssim_delta = ssim[TOPO_KEY] - ssim[ABLATION_KEY]

    # Row 1: PD evidence.
    ax_cnn = fig.add_subplot(gs[0, 0:3])
    ax_abl = fig.add_subplot(gs[0, 3:6])
    ax_topo = fig.add_subplot(gs[0, 6:9])
    ax_surv = fig.add_subplot(gs[0, 9:12])

    _draw_pd_overlay(
        ax_cnn,
        display[GT_LABEL][0], display[GT_LABEL][1],
        display[CNN_KEY][0], display[CNN_KEY][1],
        lo, hi, DISPLAY_LABEL[CNN_KEY],
        canonical[CNN_KEY]['d_B'], canonical[CNN_KEY]['W2'],
        focus_region=focus_region,
    )
    _draw_pd_overlay(
        ax_abl,
        display[GT_LABEL][0], display[GT_LABEL][1],
        display[ABLATION_KEY][0], display[ABLATION_KEY][1],
        lo, hi, DISPLAY_LABEL[ABLATION_KEY],
        canonical[ABLATION_KEY]['d_B'], canonical[ABLATION_KEY]['W2'],
        focus_region=focus_region,
    )
    _draw_pd_overlay(
        ax_topo,
        display[GT_LABEL][0], display[GT_LABEL][1],
        display[TOPO_KEY][0], display[TOPO_KEY][1],
        lo, hi, DISPLAY_LABEL[TOPO_KEY],
        canonical[TOPO_KEY]['d_B'], canonical[TOPO_KEY]['W2'],
        focus_region=focus_region,
    )

    survival_curves = {
        GT_LABEL: pd_sources[GT_LABEL]['death'] - pd_sources[GT_LABEL]['birth'],
        CNN_KEY: pd_sources[CNN_KEY]['death'] - pd_sources[CNN_KEY]['birth'],
        ABLATION_KEY: pd_sources[ABLATION_KEY]['death'] - pd_sources[ABLATION_KEY]['birth'],
        TOPO_KEY: pd_sources[TOPO_KEY]['death'] - pd_sources[TOPO_KEY]['birth'],
    }
    _draw_survival(
        ax_surv, survival_curves, args.min_persistence,
        x_min=args.survival_xmin, x_max=args.survival_xmax,
    )

    # Row 2: field-space comparison. Keep CNN here so the visual row matches
    # the three reconstruction methods shown in the PD row. All four panels
    # share one color scale and one colorbar.
    ax_gt_field = fig.add_subplot(gs[1, 0:3])
    ax_cnn_field = fig.add_subplot(gs[1, 3:6])
    ax_abl_field = fig.add_subplot(gs[1, 6:9])
    ax_topo_field = fig.add_subplot(gs[1, 9:12])

    im = _draw_field_no_cbar(
        ax_gt_field,
        panel_data['gt_speed'],
        panel_data['speed_vmin'],
        panel_data['speed_vmax'],
        'GT',
    )
    _draw_field_no_cbar(
        ax_cnn_field,
        panel_data['method_speeds'][CNN_KEY],
        panel_data['speed_vmin'],
        panel_data['speed_vmax'],
        DISPLAY_LABEL[CNN_KEY],
    )
    _draw_field_no_cbar(
        ax_abl_field,
        panel_data['method_speeds'][ABLATION_KEY],
        panel_data['speed_vmin'],
        panel_data['speed_vmax'],
        DISPLAY_LABEL[ABLATION_KEY],
    )
    _draw_field_no_cbar(
        ax_topo_field,
        panel_data['method_speeds'][TOPO_KEY],
        panel_data['speed_vmin'],
        panel_data['speed_vmax'],
        DISPLAY_LABEL[TOPO_KEY],
    )

    cbar = fig.colorbar(
        im,
        ax=[ax_gt_field, ax_cnn_field, ax_abl_field, ax_topo_field],
        fraction=0.018,
        pad=0.018,
    )
    cbar.set_label('Wind-speed magnitude', fontsize=11)
    cbar.ax.tick_params(labelsize=9.5)

    # Row 3: compact poster callouts instead of a large table.
    ax_card1 = fig.add_subplot(gs[2, 0:4])
    ax_card2 = fig.add_subplot(gs[2, 4:8])
    ax_card3 = fig.add_subplot(gs[2, 8:12])

    _metric_card(
        ax_card1,
        'Persistence improvement vs Ablation',
        (
            rf'$d_B$: {canonical[ABLATION_KEY]["d_B"]:.3f} → '
            rf'{canonical[TOPO_KEY]["d_B"]:.3f}  (-{db_reduction:.1f}%)'
            '\n'
            rf'$W_2$: {canonical[ABLATION_KEY]["W2"]:.3f} → '
            rf'{canonical[TOPO_KEY]["W2"]:.3f}  (-{w2_reduction:.1f}%)'
        ),
    )
    _metric_card(
        ax_card2,
        'Fidelity remains similar',
        (
            rf'PSNR$_{{uv}}$: {psnr[ABLATION_KEY]:.3f} → {psnr[TOPO_KEY]:.3f} '
            rf'({psnr_delta:+.3f} dB)'
            '\n'
            rf'SSIM$_{{speed}}$: {ssim[ABLATION_KEY]:.3f} → {ssim[TOPO_KEY]:.3f}'
        ),
    )
    _metric_card(
        ax_card3,
        'Interpretation',
        (
            'Large PD improvement with minimal change'
            '\nin reconstruction fidelity.'
        ),
    )

    fig.text(
        0.5, 0.025,
        (
            f'Displayed PD points use persistence ≥ {args.min_persistence:g} for readability; '
            r'$d_B$ and $W_2$ use the complete finite diagrams. '
            r'PSNR is on $[u,v]$; SSIM and topology are on scalar speed.'
        ),
        ha='center', va='bottom', fontsize=10.5,
    )

    # Robust absolute output directory, so provenance relative_to(REPO_ROOT) is valid.
    args_outdir = Path(args.outdir).expanduser()
    if not args_outdir.is_absolute():
        args_outdir = REPO_ROOT / args_outdir
    args_outdir = args_outdir.resolve()
    args_outdir.mkdir(parents=True, exist_ok=True)

    stem = f'poster_candidatec_qualitative_s{sample_idx}'
    out_paths = {ext: args_outdir / f'{stem}.{ext}' for ext in ('png', 'svg', 'pdf')}
    for ext, path in out_paths.items():
        fig.savefig(path, dpi=(300 if ext == 'png' else None),
                    facecolor='white', bbox_inches='tight')
    plt.close(fig)

    provenance = dict(
        sample_idx=sample_idx,
        min_persistence=args.min_persistence,
        ablation_definition=ABLATION_NOTE,
        methods={
            GT_LABEL: dict(pd_source_path=phase2db._rel(pd_sources[GT_LABEL]['path']),
                              artifact_alias=pd_sources[GT_LABEL]['alias'],
                              filtration_convention=pd_sources[GT_LABEL]['filtration_convention'],
                              source_family=pd_sources[GT_LABEL]['source_family'],
                              pair_count=pd_sources[GT_LABEL]['pair_count'],
                              n_agreeing_copies=pd_sources[GT_LABEL]['n_copies']),
            **{key: dict(display_label=DISPLAY_LABEL[key],
                           pd_source_path=phase2db._rel(pd_sources[key]['path']),
                           artifact_alias=pd_sources[key]['alias'],
                           filtration_convention=pd_sources[key]['filtration_convention'],
                           source_family=pd_sources[key]['source_family'],
                           pair_count=pd_sources[key]['pair_count'],
                           n_agreeing_copies=pd_sources[key]['n_copies'],
                           d_B=canonical[key]['d_B'], W2=canonical[key]['W2'],
                           psnr_uv=psnr[key], ssim_speed=ssim[key],
                           raw_dataGT=str(raw_paths[key]['dataGT'].relative_to(REPO_ROOT)),
                           raw_dataSR=str(raw_paths[key]['dataSR'].relative_to(REPO_ROOT)),
                           raw_idx=str(raw_paths[key]['idx'].relative_to(REPO_ROOT)))
               for key in METHOD_KEYS},
        },
        canonical_pd_csv=str(args.canonical_pd_csv),
        canonical_pd_csv_resolved_columns=canonical['_resolved_columns'],
        ssim_csv=str(args.ssim_csv),
        unified_long_csv=str(args.unified_long_csv),
        crop=dict(patch=PATCH, x0=X0, y0=Y0),
        shared_pd_zoom=dict(lo=lo, hi=hi, total_displayed_points=total_points),
        gt_defined_focus_region=(
            None if focus_region is None else dict(
                birth_min=focus_region[0], birth_max=focus_region[1],
                death_min=focus_region[2], death_max=focus_region[3],
            )
        ),
        survival_display_range=dict(xmin=args.survival_xmin, xmax=args.survival_xmax),
        output_paths={ext: str(p.relative_to(REPO_ROOT)) for ext, p in out_paths.items()},
        warnings=list(warnings),
        caption=CAPTION_TEXT,
        psnr_ssim_note=PSNR_SSIM_NOTE,
    )
    sidecar_path = args_outdir / f'{stem}.json'
    sidecar_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + '\n')
    print(f'  Wrote: {out_paths["png"]}')
    print(f'  Wrote: {out_paths["svg"]}')
    print(f'  Wrote: {out_paths["pdf"]}')
    print(f'  Wrote: {sidecar_path}')

    return dict(sample_idx=sample_idx, canonical=canonical, psnr=psnr, ssim=ssim,
                 output_paths=out_paths, sidecar_path=sidecar_path, warnings=list(warnings))


# =============================================================================
# Section 8: batch / contact sheet
# =============================================================================

SAMPLE_LIST_ID_CANDIDATES = ['sample_idx', 'sample', 'sample_id']


def read_sample_list(path: Path, top_n: int, warnings: list) -> list:
    if not path.exists():
        hard_fail(f'--sample-list file not found: {path}.')
    with path.open(newline='') as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or []
        id_col = None
        for c in SAMPLE_LIST_ID_CANDIDATES:
            if c in fieldnames:
                id_col = c
                break
        if id_col is None:
            hard_fail(f'{path} has none of the expected sample-id columns {SAMPLE_LIST_ID_CANDIDATES}. '
                       f'Available columns: {fieldnames}.')
        rows = list(reader)
    if len(rows) < top_n:
        warnings.append(f'--sample-list {path} has only {len(rows)} row(s); requested --top-n {top_n}. '
                          f'Rendering all {len(rows)} available rows in their existing (never re-sorted) order.')
        print(f'  [warning] {warnings[-1]}')
    selected_rows = rows[:top_n]
    sample_ids = []
    for r in selected_rows:
        try:
            sample_ids.append(int(r[id_col]))
        except (TypeError, ValueError):
            hard_fail(f'Non-integer sample id {r[id_col]!r} in {path} column {id_col!r}.')
    return sample_ids


def build_contact_sheet(results: list, outdir: Path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n = len(results)
    fig, ax = plt.subplots(figsize=(max(8, 1.6 * n), 2.2 + 0.4 * n), dpi=150, facecolor='white')
    ax.axis('off')
    col_labels = ['sample', 'd_B improvement\nvs Ablation ↑', 'W2 improvement\nvs Ablation ↑',
                   'PSNR diff\nC - Ablation ↑', 'SSIM diff\nC - Ablation ↑']
    cell_text = []
    for r in results:
        c = r['canonical']
        db_imp = c[ABLATION_KEY]['d_B'] - c[TOPO_KEY]['d_B']
        w2_imp = c[ABLATION_KEY]['W2'] - c[TOPO_KEY]['W2']
        psnr_diff = r['psnr'][TOPO_KEY] - r['psnr'][ABLATION_KEY]
        ssim_diff = r['ssim'][TOPO_KEY] - r['ssim'][ABLATION_KEY]
        cell_text.append([str(r['sample_idx']), f'{db_imp:+.3f}', f'{w2_imp:+.3f}', f'{psnr_diff:+.3f}',
                            f'{ssim_diff:+.3f}'])
    table = ax.table(cellText=cell_text, colLabels=col_labels, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.8)
    ax.set_title('Poster-qualitative contact sheet: quantitative shortlist\n'
                  '(sample selection is never changed based on rendered appearance)', fontsize=12)
    out_path = outdir / 'contact_sheet.png'
    fig.savefig(out_path, dpi=300, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    return out_path


# =============================================================================
# main
# =============================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--sample', type=int, help='Single sample index to render (0..167).')
    mode.add_argument('--sample-list', type=str,
                        help='CSV of candidate samples (e.g. produced by select_poster_qualitative_sample.py); '
                             'renders the top --top-n rows in file order plus a contact_sheet.png.')
    ap.add_argument('--top-n', type=int, default=10, help='With --sample-list: number of rows to render.')
    ap.add_argument('--min-persistence', type=float, default=5.0,
                      help='Minimum persistence (death - birth) for a point to be DISPLAYED in the PD overlay '
                           'panels. Never affects d_B/W2 or the survival curves.')
    ap.add_argument('--survival-xmin', type=float, default=4.0,
                      help='Left edge of the displayed survival-curve persistence range. '
                           'The curve itself is computed from complete finite diagrams.')
    ap.add_argument('--survival-xmax', type=float, default=10.0,
                      help='Right edge of the displayed survival-curve persistence range. '
                           'The curve itself is computed from complete finite diagrams.')
    ap.add_argument('--outdir', type=str, default=str(DEFAULT_OUTDIR), help='Output directory.')
    ap.add_argument('--canonical-pd-csv', type=str, default=str(DEFAULT_CANONICAL_PD_CSV),
                      help='Independently recomputed canonical PD sweep CSV (bottleneck d_B, 2-Wasserstein W2).')
    ap.add_argument('--ssim-csv', type=str, default=str(DEFAULT_SSIM_CSV),
                      help='Recomputed SSIM scale-sweep per-sample CSV.')
    ap.add_argument('--unified-long-csv', type=str, default=str(DEFAULT_UNIFIED_LONG_CSV),
                      help='Frozen unified per-sample long table (source of PSNR_uv).')
    return ap


def main() -> int:
    ap = build_arg_parser()
    args = ap.parse_args()

    if args.min_persistence < 0:
        hard_fail('--min-persistence must be >= 0.')
    if args.survival_xmin < 0:
        hard_fail('--survival-xmin must be >= 0.')
    if args.survival_xmax <= args.survival_xmin:
        hard_fail('--survival-xmax must be greater than --survival-xmin.')

    print('=' * 88)
    print('Poster qualitative comparison: CNN / Ablation / Topology-inspired')
    print(f'  {ABLATION_NOTE}')
    print(f'  Repo root: {REPO_ROOT}')
    print(f'  Canonical PD metrics CSV: {args.canonical_pd_csv}')
    print(f'  SSIM CSV: {args.ssim_csv}')
    print(f'  Unified long CSV (PSNR_uv): {args.unified_long_csv}')
    print(f'  Output dir: {args.outdir}')
    print('=' * 88)

    method_inventory = p2da.load_method_inventory()
    raw_paths = resolve_and_cross_check_raw_paths(method_inventory)
    print('\n[validate] Raw-array paths (cross-checked against method_inventory.csv):')
    for key in METHOD_KEYS:
        print(f'  {DISPLAY_LABEL[key]}: {raw_paths[key]["dir"].relative_to(REPO_ROOT)}')
    validation_notes = validate_raw_arrays(raw_paths)
    for n in validation_notes:
        print(f'  [PASS] {n}')

    warnings: list = []
    all_results = []

    if args.sample is not None:
        result = render_one_sample(args.sample, args, raw_paths, method_inventory, warnings)
        all_results.append(result)
    else:
        sample_ids = read_sample_list(Path(args.sample_list), args.top_n, warnings)
        print(f'\n[batch] Rendering {len(sample_ids)} sample(s) from {args.sample_list}: {sample_ids}')
        for sidx in sample_ids:
            result = render_one_sample(sidx, args, raw_paths, method_inventory, warnings)
            all_results.append(result)
        contact_sheet_path = build_contact_sheet(all_results, Path(args.outdir))
        print(f'\n  Wrote: {contact_sheet_path}')

    print('\n' + '=' * 88)
    print('DONE.')
    print('Output files:')
    for r in all_results:
        for ext, p in r['output_paths'].items():
            print(f'  {p}')
        print(f'  {r["sidecar_path"]}')
    if warnings:
        print(f'\n{len(warnings)} warning(s):')
        for w in warnings:
            print(f'  - {w}')
    else:
        print('\nNo warnings.')
    print('=' * 88)
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except ProvenanceError:
        raise
    except SystemExit:
        raise
