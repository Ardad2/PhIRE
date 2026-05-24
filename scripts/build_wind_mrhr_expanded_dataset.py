#!/usr/bin/env python3
r"""
Build an expanded PhIRE-compatible wind [u,v] MR-HR dataset from NREL WIND Toolkit.

Extracts four 168-hour seasonal windows from /nrel/wtk-us.h5 via HSDS/h5pyd,
using the same 500×500 spatial crop centered on 39.5°N 75°W as the 168-sample
benchmark.  Produces 672 samples that are strictly non-overlapping with the
original benchmark (WTK indices 0..167).

HSDS endpoint and API key are never hardcoded.  They are read (in priority order)
from environment variables HS_ENDPOINT / HS_API_KEY, or from ~/.hscfg.  The
developer.nlr.gov endpoint can be set via HS_ENDPOINT without modifying this file.

Required output (in --out-dir):
  wind_LR-MR.tfrecord   PhIRE-native TFRecord (LR 10×10  → HR 100×100)
  wind_MR-HR.tfrecord   PhIRE-native TFRecord (LR 100×100 → HR 500×500)
  hr_stack.npy          (672, 500, 500, 2) float64
  mr_stack.npy          (672, 100, 100, 2) float64
  lr_stack.npy          (672,  10,  10, 2) float64
  manifest.csv          per-sample metadata
  stats.json            channel statistics and normalization compatibility

Usage (from repo root, on Spark):
  # Dry run — connect, find crop, print config, exit without downloading data:
  python3 scripts/build_wind_mrhr_expanded_dataset.py \
    --out-dir example_data_topology_expanded_672 --dry-run

  # Full 672-sample build:
  python3 scripts/build_wind_mrhr_expanded_dataset.py \
    --out-dir example_data_topology_expanded_672
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT   = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HR_SIZE  = 500
MR_SIZE  = 100
LR_SIZE  = 10
HEIGHT   = 100        # hub height (metres)
CHANNELS = 2          # [u, v]

# PhIRE pretrained normalization (unchanged — only compatibility is assessed)
PHIRE_MU  = [0.7684, -0.4575]
PHIRE_SIG = [5.02455, 5.9017]

# Original benchmark WTK indices (0..167 must not appear in expanded set)
_BENCHMARK_SET = frozenset(range(168))

# Paths that must never be overwritten
_PROTECTED_PATHS = [
    'example_data_fixed',
    'data_out_fixed',
    'example_data/wind_MR-HR.tfrecord',
    'example_data/wind_LR-MR.tfrecord',
]

_DEFAULT_WTK_PATH = '/nrel/wtk-us.h5'
_DEFAULT_WINDOWS  = 'winter:336:168,spring:2160:168,summer:4344:168,fall:6552:168'

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument('--out-dir', required=True,
                    help='Output directory (created if needed).')
    ap.add_argument('--wtk-path', default=_DEFAULT_WTK_PATH,
                    help=f'HDF5 path in the HSDS server (default: {_DEFAULT_WTK_PATH}).')
    ap.add_argument('--center-lat', type=float, default=39.5,
                    help='Center latitude for spatial crop (default: 39.5).')
    ap.add_argument('--center-lon', type=float, default=-75.0,
                    help='Center longitude for spatial crop (default: -75.0).')
    ap.add_argument('--windows', default=_DEFAULT_WINDOWS,
                    help='Comma-separated windows as name:start_idx:length '
                         f'(default: {_DEFAULT_WINDOWS}).')
    ap.add_argument('--overwrite', action='store_true',
                    help='Replace the output directory if it already exists.')
    ap.add_argument('--save-npy', default=True,
                    action=argparse.BooleanOptionalAction,
                    help='Save hr/mr/lr_stack.npy (default: on; use --no-save-npy to skip).')
    ap.add_argument('--dry-run', action='store_true',
                    help='Connect to HSDS, verify spatial crop, print plan, '
                         'then exit without downloading wind data or writing files.')
    return ap.parse_args()

# ---------------------------------------------------------------------------
# Window parsing
# ---------------------------------------------------------------------------

def _parse_windows(s: str) -> List[Tuple[str, int, int]]:
    """Parse "winter:336:168,spring:2160:168" → [(name, start, length), ...]."""
    result = []
    for part in s.split(','):
        name, start, length = part.strip().split(':')
        result.append((name, int(start), int(length)))
    return result

# ---------------------------------------------------------------------------
# Safety
# ---------------------------------------------------------------------------

def _safety_check(out_dir: Path) -> None:
    for protected in _PROTECTED_PATHS:
        p = (REPO_ROOT / protected).resolve()
        c = out_dir.resolve()
        if c == p or str(c).startswith(str(p) + os.sep):
            sys.exit(
                f'[error] --out-dir {out_dir!r} overlaps protected path '
                f'{protected!r}. Aborting to protect baselines.'
            )
    print('[safety] Output path does not overlap any protected baseline. OK.')

# ---------------------------------------------------------------------------
# HSDS connection
# ---------------------------------------------------------------------------

def _open_wtk(wtk_path: str):
    import h5pyd
    kwargs: Dict = {}
    endpoint = (os.environ.get('HS_ENDPOINT')
                or os.environ.get('HSDS_ENDPOINT'))
    api_key  = (os.environ.get('HS_API_KEY')
                or os.environ.get('HSDS_API_KEY'))
    if endpoint:
        kwargs['endpoint'] = endpoint
        print(f'  HSDS endpoint (env): {endpoint}')
    else:
        print('  HSDS endpoint: ~/.hscfg')
    if api_key:
        kwargs['api_key'] = api_key
    return h5pyd.File(wtk_path, 'r', **kwargs)

# ---------------------------------------------------------------------------
# Spatial crop
# ---------------------------------------------------------------------------

def _find_crop(
        f,
        center_lat: float,
        center_lon: float,
        hr_size: int,
) -> Tuple[int, int, int, int, float, float]:
    """Return (y0, y1, x0, x1, actual_lat, actual_lon)."""
    coords = f['coordinates'][...]
    if coords.dtype.names:
        lats = coords['lat']
        lons = coords['lon']
    else:
        lats = coords[0]
        lons = coords[1]
    H, W = lats.shape
    dist2 = (lats - center_lat) ** 2 + (lons - center_lon) ** 2
    cy, cx = np.unravel_index(np.argmin(dist2), dist2.shape)
    cy, cx = int(cy), int(cx)
    actual_lat = float(lats[cy, cx])
    actual_lon = float(lons[cy, cx])
    half = hr_size // 2
    y0, y1 = cy - half, cy - half + hr_size
    x0, x1 = cx - half, cx - half + hr_size
    if y0 < 0 or x0 < 0 or y1 > H or x1 > W:
        sys.exit(
            f'[error] {hr_size}×{hr_size} crop at ({center_lat}, {center_lon}) '
            f'out of bounds (grid {H}×{W}): rows {y0}:{y1}, cols {x0}:{x1}.'
        )
    return y0, y1, x0, x1, actual_lat, actual_lon

# ---------------------------------------------------------------------------
# Wind-component conversion
# ---------------------------------------------------------------------------

def _compute_uv(ws: np.ndarray, wd: np.ndarray) -> np.ndarray:
    """Meteorological convention: u = -speed·sin(θ), v = -speed·cos(θ)."""
    theta = np.radians(wd)
    u = (-ws * np.sin(theta)).astype(np.float64)
    v = (-ws * np.cos(theta)).astype(np.float64)
    return np.stack([u, v], axis=-1)

# ---------------------------------------------------------------------------
# Timestamp decoding (WTK stores bytes like b'2007-01-01 00:00:00')
# ---------------------------------------------------------------------------

def _decode_ts(raw) -> str:
    if isinstance(raw, (bytes, np.bytes_)):
        return raw.decode('ascii', errors='replace').strip()
    return str(raw).strip()

# ---------------------------------------------------------------------------
# HR extraction
# ---------------------------------------------------------------------------

def _extract_hr_stack(
        f,
        y0: int, y1: int, x0: int, x1: int,
        windows: List[Tuple[str, int, int]],
) -> Tuple[np.ndarray, List[Dict]]:
    """Stream HR [u,v] from WTK for all windows; return stack + metadata list."""
    n_total = sum(l for _, _, l in windows)
    hr_stack   = np.empty((n_total, HR_SIZE, HR_SIZE, CHANNELS), dtype=np.float64)
    sample_meta: List[Dict] = []
    out_idx = 0

    for season, start, length in windows:
        print(f'  [{season}] t={start}..{start+length-1} ({length} samples) …',
              flush=True)
        ws = f[f'windspeed_{HEIGHT}m'][start:start + length, y0:y1, x0:x1]
        wd = f[f'winddirection_{HEIGHT}m'][start:start + length, y0:y1, x0:x1]

        # Try to read WTK timestamps (contiguous slice avoids h5pyd fancy-index POST)
        try:
            dt_raw  = f['datetime'][start:start + length]
            timestamps = [_decode_ts(dt_raw[i]) for i in range(length)]
        except Exception:
            timestamps = [f'index_{start + i}' for i in range(length)]

        for i in range(length):
            uv = _compute_uv(ws[i], wd[i])
            hr_stack[out_idx] = uv
            u_ch = uv[..., 0]
            v_ch = uv[..., 1]
            spd  = np.sqrt(u_ch ** 2 + v_ch ** 2)
            sample_meta.append({
                'sample_idx':    out_idx,
                'wtk_time_index': start + i,
                'timestamp':     timestamps[i],
                'season':        season,
                'u_min':   float(u_ch.min()),  'u_max':   float(u_ch.max()),
                'u_mean':  float(u_ch.mean()), 'u_std':   float(u_ch.std()),
                'v_min':   float(v_ch.min()),  'v_max':   float(v_ch.max()),
                'v_mean':  float(v_ch.mean()), 'v_std':   float(v_ch.std()),
                'speed_min':  float(spd.min()),  'speed_max':  float(spd.max()),
                'speed_mean': float(spd.mean()), 'speed_std':  float(spd.std()),
            })
            out_idx += 1

        block = hr_stack[out_idx - length:out_idx]
        print(f'    u=[{block[...,0].min():.2f}, {block[...,0].max():.2f}]  '
              f'v=[{block[...,1].min():.2f}, {block[...,1].max():.2f}]')

    return hr_stack, sample_meta

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate(
        hr_stack: np.ndarray,
        mr_stack: np.ndarray,
        lr_stack: np.ndarray,
        sample_meta: List[Dict],
        n_expected: int,
) -> None:
    assert hr_stack.shape == (n_expected, HR_SIZE, HR_SIZE, CHANNELS), \
        f'HR shape {hr_stack.shape} ≠ ({n_expected}, {HR_SIZE}, {HR_SIZE}, {CHANNELS})'
    assert mr_stack.shape == (n_expected, MR_SIZE, MR_SIZE, CHANNELS), \
        f'MR shape {mr_stack.shape} ≠ ({n_expected}, {MR_SIZE}, {MR_SIZE}, {CHANNELS})'
    assert lr_stack.shape == (n_expected, LR_SIZE, LR_SIZE, CHANNELS), \
        f'LR shape {lr_stack.shape} ≠ ({n_expected}, {LR_SIZE}, {LR_SIZE}, {CHANNELS})'

    for tag, arr in [('HR', hr_stack), ('MR', mr_stack), ('LR', lr_stack)]:
        if not np.all(np.isfinite(arr)):
            bad = int(np.sum(~np.isfinite(arr)))
            raise AssertionError(f'{tag} stack contains {bad} NaN/Inf values')

    overlaps = [m['wtk_time_index'] for m in sample_meta
                if m['wtk_time_index'] in _BENCHMARK_SET]
    assert not overlaps, \
        f'WTK indices {overlaps} overlap benchmark 0..167'

    speed = np.sqrt(hr_stack[..., 0] ** 2 + hr_stack[..., 1] ** 2)
    print(f'[validate] HR {hr_stack.shape}  OK')
    print(f'[validate] MR {mr_stack.shape}  OK')
    print(f'[validate] LR {lr_stack.shape}  OK')
    print(f'[validate] No NaN/Inf in any stack.  OK')
    print(f'[validate] No WTK overlap with benchmark 0..167.  OK')
    print(f'[validate] u  range: [{hr_stack[...,0].min():.3f}, {hr_stack[...,0].max():.3f}]')
    print(f'[validate] v  range: [{hr_stack[...,1].min():.3f}, {hr_stack[...,1].max():.3f}]')
    print(f'[validate] spd range: [{speed.min():.3f}, {speed.max():.3f}]')

# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def _write_manifest(
        out_dir: Path,
        sample_meta: List[Dict],
        crop: Tuple[int, int, int, int],
        actual_lat: float,
        actual_lon: float,
) -> None:
    y0, y1, x0, x1 = crop
    fieldnames = [
        'sample_idx', 'wtk_time_index', 'timestamp', 'season',
        'center_lat', 'center_lon',
        'row_start', 'row_end', 'col_start', 'col_end',
        'u_min', 'u_max', 'u_mean', 'u_std',
        'v_min', 'v_max', 'v_mean', 'v_std',
        'speed_min', 'speed_max', 'speed_mean', 'speed_std',
    ]
    path = out_dir / 'manifest.csv'
    with path.open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for m in sample_meta:
            row = dict(m)
            row.update({
                'center_lat': actual_lat, 'center_lon': actual_lon,
                'row_start': y0, 'row_end': y1,
                'col_start': x0, 'col_end': x1,
            })
            w.writerow(row)
    print(f'Written: {path}  ({len(sample_meta)} rows)')

# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def _write_stats(out_dir: Path, hr_stack: np.ndarray) -> Dict:
    u     = hr_stack[..., 0].ravel()
    v     = hr_stack[..., 1].ravel()
    speed = np.sqrt(hr_stack[..., 0] ** 2 + hr_stack[..., 1] ** 2).ravel()

    u_z = float((u.mean() - PHIRE_MU[0]) / PHIRE_SIG[0])
    v_z = float((v.mean() - PHIRE_MU[1]) / PHIRE_SIG[1])

    stats: Dict = {
        'n_samples': int(hr_stack.shape[0]),
        'hr_shape':  list(hr_stack.shape),
        'u': {
            'mean': float(u.mean()), 'std': float(u.std()),
            'min':  float(u.min()),  'max': float(u.max()),
        },
        'v': {
            'mean': float(v.mean()), 'std': float(v.std()),
            'min':  float(v.min()),  'max': float(v.max()),
        },
        'speed': {
            'mean': float(speed.mean()), 'std':  float(speed.std()),
            'min':  float(speed.min()),  'max':  float(speed.max()),
            'p50':  float(np.percentile(speed, 50)),
            'p90':  float(np.percentile(speed, 90)),
            'p95':  float(np.percentile(speed, 95)),
            'p99':  float(np.percentile(speed, 99)),
        },
        'phire_pretrained_normalization': {
            'mu_u':  PHIRE_MU[0],  'mu_v':  PHIRE_MU[1],
            'sig_u': PHIRE_SIG[0], 'sig_v': PHIRE_SIG[1],
            'note': 'These constants are unchanged; only compatibility is assessed here.',
        },
        'normalization_compatibility': {
            'u_z_of_expanded_mean': u_z,
            'v_z_of_expanded_mean': v_z,
            'compatible': bool(abs(u_z) < 3.0 and abs(v_z) < 3.0),
            'note': (
                'compatible=True when both channel means are within 3σ of the '
                'pretrained normalization center.  If True, the pretrained CNN can '
                'be applied directly.  If False, consider recomputing mu/sig or '
                'fine-tuning from the expanded dataset.'
            ),
        },
    }

    path = out_dir / 'stats.json'
    with path.open('w') as fh:
        json.dump(stats, fh, indent=2)
    print(f'Written: {path}')
    return stats

# ---------------------------------------------------------------------------
# Notes document
# ---------------------------------------------------------------------------

def _write_notes(
        out_dir: Path,
        stats: Dict,
        windows: List[Tuple[str, int, int]],
        actual_lat: float,
        actual_lon: float,
        crop: Tuple[int, int, int, int],
        center_lat: float,
        center_lon: float,
) -> None:
    import datetime
    y0, y1, x0, x1 = crop
    n    = stats['n_samples']
    compat = stats['normalization_compatibility']['compatible']
    u_z    = stats['normalization_compatibility']['u_z_of_expanded_mean']
    v_z    = stats['normalization_compatibility']['v_z_of_expanded_mean']

    notes_path = REPO_ROOT / 'docs' / 'expanded_dataset_672_notes.md'
    notes_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        '# Expanded Dataset 672 Notes',
        '',
        f'**Generated:** {datetime.date.today().isoformat()}',
        '',
        '## Summary',
        '',
        f'- Total samples: {n} (4 windows × 168 hours)',
        f'- Output directory: `{out_dir}`',
        f'- HR shape: {stats["hr_shape"]}',
        '',
        '## Spatial crop',
        '',
        f'- Target center: {center_lat}°N, {center_lon}°W',
        f'- Actual grid center: {actual_lat:.4f}°N, {actual_lon:.4f}°W',
        f'- WTK native rows: {y0}:{y1}, cols: {x0}:{x1}',
        f'- HR patch: {HR_SIZE}×{HR_SIZE} (same as 168-sample benchmark)',
        '',
        '## Seasonal windows',
        '',
        '| Season | WTK start | Length | WTK end |',
        '|--------|-----------|--------|---------|',
    ]
    for name, start, length in windows:
        lines.append(f'| {name} | {start} | {length} | {start + length - 1} |')

    lines += [
        '',
        '## Resolution hierarchy',
        '',
        f'- HR: {HR_SIZE}×{HR_SIZE}×2 — native WTK crop',
        f'- MR: {MR_SIZE}×{MR_SIZE}×2 — 5× block-average from HR via `utils.downscale_image`',
        f'- LR: {LR_SIZE}×{LR_SIZE}×2 — 10× block-average from MR via `utils.downscale_image`',
        '',
        '## Statistics over expanded HR stack',
        '',
        f'- u: mean={stats["u"]["mean"]:.4f}  std={stats["u"]["std"]:.4f}'
        f'  min={stats["u"]["min"]:.3f}  max={stats["u"]["max"]:.3f}',
        f'- v: mean={stats["v"]["mean"]:.4f}  std={stats["v"]["std"]:.4f}'
        f'  min={stats["v"]["min"]:.3f}  max={stats["v"]["max"]:.3f}',
        f'- speed: mean={stats["speed"]["mean"]:.4f}  std={stats["speed"]["std"]:.4f}'
        f'  p50={stats["speed"]["p50"]:.3f}  p90={stats["speed"]["p90"]:.3f}'
        f'  p95={stats["speed"]["p95"]:.3f}  p99={stats["speed"]["p99"]:.3f}',
        '',
        '## Normalization compatibility with pretrained PhIRE CNN',
        '',
        f'Pretrained: mu=[{PHIRE_MU[0]}, {PHIRE_MU[1]}]  sig=[{PHIRE_SIG[0]}, {PHIRE_SIG[1]}]',
        '',
        f'- u mean z-score vs pretrained: {u_z:.3f}',
        f'- v mean z-score vs pretrained: {v_z:.3f}',
        f'- **compatible={compat}** (criterion: |z| < 3 for both channels)',
        '',
    ]
    if compat:
        lines.append(
            'The expanded dataset mean falls within 3σ of the pretrained normalization '
            'center.  The pretrained CNN checkpoint can be applied to this data without '
            'recomputing mu/sig.'
        )
    else:
        lines.append(
            'The expanded dataset mean falls OUTSIDE 3σ of the pretrained normalization '
            'center.  Consider recomputing mu/sig from the expanded set or fine-tuning '
            'the pretrained CNN before applying it to this data.'
        )

    min_start = min(s for _, s, _ in windows)
    lines += [
        '',
        '## Benchmark non-overlap',
        '',
        f'Minimum WTK index in the expanded set: {min_start}  (benchmark ends at 167).',
        'None of the 672 samples overlap the original benchmark (WTK 0..167 = '
        '2007-01-01 00:00 to 2007-01-07 23:00 UTC).',
        '',
        '## Output files',
        '',
        f'| File | Description |',
        f'|------|-------------|',
        f'| `{out_dir}/wind_LR-MR.tfrecord` | PhIRE TFRecord LR=10×10→MR=100×100 |',
        f'| `{out_dir}/wind_MR-HR.tfrecord` | PhIRE TFRecord MR=100×100→HR=500×500 |',
        f'| `{out_dir}/hr_stack.npy` | (672, 500, 500, 2) float64 |',
        f'| `{out_dir}/mr_stack.npy` | (672, 100, 100, 2) float64 |',
        f'| `{out_dir}/lr_stack.npy` | (672, 10, 10, 2) float64 |',
        f'| `{out_dir}/manifest.csv` | Per-sample metadata (672 rows) |',
        f'| `{out_dir}/stats.json` | Channel stats and normalization comparison |',
        '',
    ]

    notes_path.write_text('\n'.join(lines), encoding='utf-8')
    print(f'Written: {notes_path}')

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args    = _parse_args()
    out_dir = Path(args.out_dir)
    windows = _parse_windows(args.windows)
    n_total = sum(l for _, _, l in windows)

    print('=' * 64)
    print('PhIRE expanded dataset builder')
    print('=' * 64)
    print(f'  Out dir      : {out_dir}')
    print(f'  WTK path     : {args.wtk_path}')
    print(f'  Center       : {args.center_lat}°N  {args.center_lon}°W')
    print(f'  Dry run      : {args.dry_run}')
    print(f'  Overwrite    : {args.overwrite}')
    print(f'  Save NPY     : {args.save_npy}')
    print()
    for name, start, length in windows:
        print(f'  [{name:8s}] t={start}..{start + length - 1}  ({length} samples)')
    print(f'  Total        : {n_total} samples')
    print()

    # Guard: no window may touch the benchmark indices
    for name, start, length in windows:
        overlap = _BENCHMARK_SET.intersection(range(start, start + length))
        if overlap:
            sys.exit(
                f'[error] Window [{name}] WTK indices {sorted(overlap)} '
                'overlap the benchmark range 0..167.'
            )

    _safety_check(out_dir)

    # Check output directory
    if not args.dry_run:
        if out_dir.exists() and any(out_dir.iterdir()):
            if not args.overwrite:
                sys.exit(
                    f'[error] Output directory {out_dir!r} exists and is non-empty. '
                    'Use --overwrite to replace it.'
                )
            import shutil
            shutil.rmtree(out_dir)

    # Open WTK and locate crop
    print('Opening WTK via h5pyd …')
    f = _open_wtk(args.wtk_path)
    ws_shape = f[f'windspeed_{HEIGHT}m'].shape
    print(f'  windspeed_{HEIGHT}m shape: {ws_shape}')

    y0, y1, x0, x1, actual_lat, actual_lon = _find_crop(
        f, args.center_lat, args.center_lon, HR_SIZE,
    )
    print(f'  Nearest grid center: ({actual_lat:.4f}°N, {actual_lon:.4f}°W)')
    print(f'  Crop               : rows {y0}:{y1}, cols {x0}:{x1}')

    if args.dry_run:
        f.close()
        print()
        print('[dry-run] Config valid. Would extract:')
        for name, start, length in windows:
            print(f'  [{name}] t={start}..{start + length - 1}  '
                  f'rows {y0}:{y1}  cols {x0}:{x1}  →  {length} HR samples')
        print(f'  Total: {n_total} samples → {out_dir}')
        print('[dry-run] No data downloaded, no files written. Done.')
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    # Extract HR stack
    print()
    print('Extracting HR [u,v] stack from WTK …')
    hr_stack, sample_meta = _extract_hr_stack(f, y0, y1, x0, x1, windows)
    f.close()
    print(f'HR stack extracted: {hr_stack.shape}  dtype={hr_stack.dtype}')

    # Downscale
    print()
    print('Downscaling HR→MR (factor 5) using utils.downscale_image …')
    import utils
    mr_stack = utils.downscale_image(hr_stack, 5)
    print(f'MR stack: {mr_stack.shape}')

    print('Downscaling MR→LR (factor 10) using utils.downscale_image …')
    lr_stack = utils.downscale_image(mr_stack, 10)
    print(f'LR stack: {lr_stack.shape}')

    # Validate
    print()
    print('Validating …')
    _validate(hr_stack, mr_stack, lr_stack, sample_meta, n_total)

    # Save NPY stacks
    if args.save_npy:
        print()
        print('Saving NPY stacks …')
        np.save(out_dir / 'hr_stack.npy', hr_stack)
        print(f'Written: {out_dir}/hr_stack.npy')
        np.save(out_dir / 'mr_stack.npy', mr_stack)
        print(f'Written: {out_dir}/mr_stack.npy')
        np.save(out_dir / 'lr_stack.npy', lr_stack)
        print(f'Written: {out_dir}/lr_stack.npy')

    # Write PhIRE-native TFRecords
    print()
    print('Writing PhIRE-native TFRecords via utils.generate_TFRecords …')
    lr_mr_path = out_dir / 'wind_LR-MR.tfrecord'
    mr_hr_path = out_dir / 'wind_MR-HR.tfrecord'
    # LR-MR: pass MR stack (100×100) with K=10 → TFRecord stores LR=10×10, HR=100×100
    utils.generate_TFRecords(
        str(lr_mr_path), mr_stack.astype(np.float64), mode='train', K=10,
    )
    print(f'Written: {lr_mr_path}')
    # MR-HR: pass HR stack (500×500) with K=5  → TFRecord stores LR=100×100, HR=500×500
    utils.generate_TFRecords(
        str(mr_hr_path), hr_stack.astype(np.float64), mode='train', K=5,
    )
    print(f'Written: {mr_hr_path}')

    # Write manifest
    print()
    crop = (y0, y1, x0, x1)
    _write_manifest(out_dir, sample_meta, crop, actual_lat, actual_lon)

    # Write stats
    stats = _write_stats(out_dir, hr_stack)

    # Write notes doc
    _write_notes(out_dir, stats, windows, actual_lat, actual_lon, crop,
                 args.center_lat, args.center_lon)

    # Summary
    print()
    print('=' * 64)
    print('Expanded dataset build complete.')
    print(f'  Output    : {out_dir}')
    print(f'  Samples   : {n_total}')
    u_s = stats['u']
    v_s = stats['v']
    print(f'  u mean/std: {u_s["mean"]:.4f} / {u_s["std"]:.4f}')
    print(f'  v mean/std: {v_s["mean"]:.4f} / {v_s["std"]:.4f}')
    compat = stats['normalization_compatibility']['compatible']
    u_z = stats['normalization_compatibility']['u_z_of_expanded_mean']
    v_z = stats['normalization_compatibility']['v_z_of_expanded_mean']
    print(f'  Normalization compatible: {compat}  (u_z={u_z:.2f}, v_z={v_z:.2f})')
    print('=' * 64)


if __name__ == '__main__':
    main()
