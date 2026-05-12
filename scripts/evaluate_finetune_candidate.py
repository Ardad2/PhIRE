#!/usr/bin/env python3
"""Evaluate any fine-tuned CNN candidate against baselines.

Computes per-sample metrics for: bicubic (optional), baseline CNN, baseline GAN,
and the named fine-tuned candidate.  Writes:

  <out-dir>/all_sample_metrics_<name>.csv   — one row per (sample, method)
  <out-dir>/pairwise_cnn_vs_<name>.csv      — per-metric delta CNN vs candidate
  <out-dir>/winner_counts_<name>.csv        — wins per method per metric
  <out-dir>/adjacent_cluster_table_<name>.csv — detail for adjacent-cluster samples
  docs/topology_finetuning_<name>_eval.md   — markdown report (7 questions answered)

Baseline outputs are NEVER overwritten.

PD/MT topology distances for the new candidate are NOT computed or claimed here;
they require a dedicated TTK run.  Existing pre-computed values for CNN and GAN
are loaded from the merged CSV if available.

Usage (on Spark, from repo root):
    python3 scripts/evaluate_finetune_candidate.py \\
        --candidate-name candidateC \\
        --candidate-dir  data_out/wind_finetune_pilot_candidateC \\
        --cnn-dir        data_out_fixed/wind_mrhr_cnn \\
        --gan-dir        data_out_fixed/wind_mrhr_gan \\
        --out-dir        ttk_runs_fixed/topology_finetuning/candidateC_eval

Optional:
    --bic-dir    data_out_fixed/wind_mrhr_bicubic  (generated inline if absent)
    --merged-csv ttk_runs_fixed/combined/psnr_topology_physics_merged.csv
                 (used to import pre-computed PD/MT distances for CNN and GAN)
"""

from __future__ import annotations

import argparse
import csv
import io
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import label as _nd_label
from scipy.ndimage import zoom as _nd_zoom

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT   = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

_SKIMAGE_IMPORT_ERROR: str = ''
try:
    from skimage.metrics import structural_similarity as _skssim
    _HAS_SKIMAGE = True
except Exception as e:
    _HAS_SKIMAGE = False
    _SKIMAGE_IMPORT_ERROR = repr(e)
    print(f'[warning] skimage SSIM unavailable — SSIM will be NaN. '
          f'Reason: {_SKIMAGE_IMPORT_ERROR}')

from metrics_physics import compute_physics_metrics

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ADJACENT_CLUSTER_SAMPLES = [10, 11, 12, 13, 90, 91, 92, 93, 162, 163]
FIXED_THRESHOLDS: List[float] = [5.0, 10.0, 15.0]
PERCENTILES: List[float] = [90.0, 95.0, 99.0]
_STRUCT8 = np.ones((3, 3), dtype=np.int32)

# Metrics where LOWER is better; everything else is assumed higher-is-better.
_LOWER_IS_BETTER = {
    'speed_mae', 'speed_rmse',
    'wpd_mae', 'wpd_rmse', 'wpd_bias_abs', 'wpd_w1',
    'grad_mae', 'grad_w1', 'grad_kurtosis_abs_delta',
    'psd_log_l2', 'psd_slope_abs_delta',
    'pd_distance', 'mt_distance',
    'comp_curve_l1',
}
# Exceedance and component-count abs-deltas added dynamically below.

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--candidate-name', required=True,
                    help='Short identifier for this candidate (e.g. candidateB, candidateC). '
                         'Used in output filenames and the markdown report.')
    ap.add_argument('--candidate-dir', required=True,
                    help='Directory with candidate outputs (dataSR/GT/IN/idx.npy)')
    ap.add_argument('--cnn-dir', required=True,
                    help='Baseline CNN output directory (data_out_fixed/wind_mrhr_cnn)')
    ap.add_argument('--gan-dir', required=True,
                    help='Baseline GAN output directory (data_out_fixed/wind_mrhr_gan)')
    ap.add_argument('--bic-dir', default=None,
                    help='Bicubic baseline dir (generated inline from CNN dataIN if absent)')
    ap.add_argument('--merged-csv', default=None,
                    help='Existing psnr_topology_physics_merged.csv for PD/MT distances')
    ap.add_argument('--out-dir', default=None,
                    help='Output directory (default: ttk_runs_fixed/topology_finetuning/<name>_eval)')
    ap.add_argument('--report-path', default=None,
                    help='Markdown report output path '
                         '(default: docs/topology_finetuning_<name>_eval.md)')
    return ap.parse_args()

# ---------------------------------------------------------------------------
# Array helpers
# ---------------------------------------------------------------------------

def _load(path: Path, mmap: bool = True) -> np.ndarray:
    if not path.exists():
        sys.exit(f'[error] Required file not found: {path}')
    return np.load(path, mmap_mode='r' if mmap else None)


def _speed(uv: np.ndarray) -> np.ndarray:
    """Convert (N,H,W,C) physical [u,v] to (N,H,W) speed."""
    uv = np.asarray(uv)
    if uv.ndim == 3:
        uv = uv[..., None]
    if uv.shape[-1] >= 2:
        return np.sqrt(uv[..., 0] ** 2 + uv[..., 1] ** 2)
    return np.abs(uv[..., 0])


def _psnruv(sr4d: np.ndarray, gt4d: np.ndarray) -> np.ndarray:
    """Per-sample PSNRuv (dB) on physical [u,v] arrays, dynamic range."""
    d = sr4d.astype(np.float64) - gt4d.astype(np.float64)
    mse = np.mean(d * d, axis=(1, 2, 3))
    lo = gt4d.min(axis=(1, 2, 3)).astype(np.float64)
    hi = gt4d.max(axis=(1, 2, 3)).astype(np.float64)
    dr = hi - lo
    out = np.empty(len(mse), dtype=np.float64)
    for i, m in enumerate(mse):
        out[i] = (float('inf') if m == 0.0
                  else 20.0 * math.log10(max(dr[i], 1e-12)) - 10.0 * math.log10(m))
    return out


def _speed_mae_rmse(speed_sr: np.ndarray,
                    speed_gt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per-sample speed MAE and RMSE."""
    d = speed_sr.astype(np.float64) - speed_gt.astype(np.float64)
    mae  = np.mean(np.abs(d), axis=(1, 2))
    rmse = np.sqrt(np.mean(d * d, axis=(1, 2)))
    return mae, rmse


def _ssim_per_sample(speed_sr: np.ndarray,
                     speed_gt: np.ndarray) -> np.ndarray:
    """Per-sample SSIM on scalar speed (NaN if skimage unavailable)."""
    N = speed_gt.shape[0]
    out = np.full(N, float('nan'), dtype=np.float64)
    if not _HAS_SKIMAGE:
        return out
    for i in range(N):
        g = speed_gt[i].astype(np.float64)
        s = speed_sr[i].astype(np.float64)
        dr = float(g.max() - g.min())
        if dr < 1e-12:
            out[i] = 1.0
        else:
            out[i] = _skssim(g, s, data_range=dr)
    return out


def _count_components(speed_2d: np.ndarray, threshold: float) -> int:
    mask = speed_2d >= threshold
    if not np.any(mask):
        return 0
    _, n = _nd_label(mask, _STRUCT8)
    return n


def _component_rows(speed_sr: np.ndarray,
                    speed_gt: np.ndarray,
                    fixed_thresholds: Sequence[float],
                    percentiles: Sequence[float]) -> List[Dict]:
    N = speed_gt.shape[0]
    rows = []
    for i in range(N):
        sg = speed_gt[i].astype(np.float64)
        ss = speed_sr[i].astype(np.float64)
        row: Dict[str, float] = {}

        for t in fixed_thresholds:
            tag = f't{t:g}'
            cgt = _count_components(sg, t)
            csr = _count_components(ss, t)
            row[f'comp_gt_{tag}']        = float(cgt)
            row[f'comp_sr_{tag}']        = float(csr)
            row[f'comp_delta_{tag}']     = float(csr - cgt)
            row[f'comp_abs_delta_{tag}'] = float(abs(csr - cgt))

        for p in percentiles:
            thr = float(np.percentile(sg, p))
            tag = f'p{int(p)}'
            cgt = _count_components(sg, thr)
            csr = _count_components(ss, thr)
            row[f'comp_gt_{tag}']        = float(cgt)
            row[f'comp_sr_{tag}']        = float(csr)
            row[f'comp_delta_{tag}']     = float(csr - cgt)
            row[f'comp_abs_delta_{tag}'] = float(abs(csr - cgt))

        # Dense curve L1: mean abs-delta across all 6 thresholds
        all_abs_deltas = [row[f'comp_abs_delta_{f"t{t:g}"}'] for t in fixed_thresholds] + \
                         [row[f'comp_abs_delta_{f"p{int(p)}"}'] for p in percentiles]
        row['comp_curve_l1'] = float(np.mean(all_abs_deltas))

        rows.append(row)
    return rows


def _bicubic_upsample(data_in: np.ndarray,
                      target_h: int, target_w: int) -> np.ndarray:
    n, h, w, c = data_in.shape
    if h == target_h and w == target_w:
        return data_in.astype(np.float32)
    zh, zw = target_h / h, target_w / w
    out = np.empty((n, target_h, target_w, c), dtype=np.float32)
    for i in range(n):
        for ch in range(c):
            out[i, :, :, ch] = _nd_zoom(
                data_in[i, :, :, ch].astype(np.float64),
                (zh, zw), order=3, mode='reflect', prefilter=True,
            )
        if (i + 1) % 20 == 0 or i == n - 1:
            print(f'  bicubic {i + 1}/{n}', flush=True)
    return out


def _load_topology_from_merged_csv(
        csv_path: Optional[Path],
) -> Dict[Tuple[str, int], Tuple[float, float]]:
    """Return {(method, sample_idx): (pd_distance, mt_distance)} from merged CSV."""
    result: Dict[Tuple[str, int], Tuple[float, float]] = {}
    if csv_path is None or not csv_path.exists():
        return result
    with csv_path.open() as fh:
        for row in csv.DictReader(fh):
            try:
                method = row['method'].strip().lower()
                si = int(float(row['sample_idx']))
                pd = float(row.get('pd_distance') or 'nan')
                mt = float(row.get('mt_distance') or 'nan')
                result[(method, si)] = (pd, mt)
            except (KeyError, ValueError):
                continue
    return result

# ---------------------------------------------------------------------------
# CSV writing helper
# ---------------------------------------------------------------------------

def _write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)

# ---------------------------------------------------------------------------
# Alignment: match all methods to a common sorted index list
# ---------------------------------------------------------------------------

def _align_by_idx(
        idx_arrays: Dict[str, np.ndarray]
) -> Tuple[List[int], Dict[str, np.ndarray]]:
    """Find common sample indices across all methods; return sorted list and
    per-method position maps {method: array_of_positions}."""
    sets = [set(arr.astype(int).ravel()) for arr in idx_arrays.values()]
    common = sorted(sets[0].intersection(*sets[1:]))
    pos_maps: Dict[str, np.ndarray] = {}
    for name, arr in idx_arrays.items():
        flat = arr.astype(int).ravel()
        lookup = {int(v): i for i, v in enumerate(flat)}
        pos_maps[name] = np.array([lookup[c] for c in common], dtype=int)
    return common, pos_maps

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    cand_name = args.candidate_name

    # Derive output-path defaults from candidate name.
    if args.out_dir is None:
        args.out_dir = (f'ttk_runs_fixed/topology_finetuning/{cand_name}_eval')
    if args.report_path is None:
        args.report_path = f'docs/topology_finetuning_{cand_name}_eval.md'

    cand_dir = Path(args.candidate_dir)
    cnn_dir  = Path(args.cnn_dir)
    gan_dir  = Path(args.gan_dir)
    bic_dir  = Path(args.bic_dir) if args.bic_dir else None
    out_dir  = Path(args.out_dir)
    report_p = Path(args.report_path)
    merged_csv = Path(args.merged_csv) if args.merged_csv else \
                 REPO_ROOT / 'ttk_runs_fixed' / 'combined' / \
                 'psnr_topology_physics_merged.csv'

    out_dir.mkdir(parents=True, exist_ok=True)
    report_p.parent.mkdir(parents=True, exist_ok=True)

    print('=' * 64)
    print(f'{cand_name} evaluation')
    print('=' * 64)

    # -----------------------------------------------------------------------
    # Load arrays — GT from CNN (authoritative), GT from cand (verify match)
    # -----------------------------------------------------------------------
    print('Loading arrays …')
    cnn_idx  = _load(cnn_dir  / 'idx.npy', mmap=False).astype(int).ravel()
    gan_idx  = _load(gan_dir  / 'idx.npy', mmap=False).astype(int).ravel() \
               if (gan_dir / 'idx.npy').exists() else cnn_idx.copy()
    cand_idx = _load(cand_dir / 'idx.npy', mmap=False).astype(int).ravel()

    idx_arrays = {'cnn': cnn_idx, 'gan': gan_idx, cand_name: cand_idx}

    # -----------------------------------------------------------------------
    # Handle bicubic
    # -----------------------------------------------------------------------
    bic_sr: Optional[np.ndarray] = None

    def _try_bic_dir(d: Optional[Path]) -> Optional[np.ndarray]:
        if d is not None and (d / 'dataSR.npy').exists():
            print(f'  Loading bicubic from {d}')
            return _load(d / 'dataSR.npy', mmap=True)
        return None

    bic_sr = _try_bic_dir(bic_dir)
    if bic_sr is None:
        bic_sr = _try_bic_dir(REPO_ROOT / 'data_out_fixed' / 'wind_mrhr_bicubic')

    if bic_sr is None and (cnn_dir / 'dataIN.npy').exists():
        print('  Bicubic outputs absent — generating inline …')
        data_in = _load(cnn_dir / 'dataIN.npy', mmap=False)
        data_gt = _load(cnn_dir / 'dataGT.npy', mmap=True)
        _, h_hr, w_hr, _ = data_gt.shape
        bic_sr = _bicubic_upsample(data_in, h_hr, w_hr)
        del data_in

    bic_available = bic_sr is not None
    if bic_available:
        idx_arrays['bicubic'] = cnn_idx  # bicubic shares CNN's index order

    # -----------------------------------------------------------------------
    # Align all methods to a common sorted index list
    # -----------------------------------------------------------------------
    common_idx, pos_maps = _align_by_idx(idx_arrays)
    N = len(common_idx)
    print(f'  Common samples: {N}')

    # Load large arrays (mmap)
    cnn_gt  = _load(cnn_dir  / 'dataGT.npy', mmap=True)
    cnn_sr  = _load(cnn_dir  / 'dataSR.npy', mmap=True)
    gan_sr  = _load(gan_dir  / 'dataSR.npy', mmap=True)
    cand_sr = _load(cand_dir / 'dataSR.npy', mmap=True)
    cand_gt = _load(cand_dir / 'dataGT.npy', mmap=True)

    # GT alignment check: candidate GT must match CNN GT for common samples
    print('  Checking GT alignment …', end=' ', flush=True)
    cnn_pos_c  = pos_maps['cnn']
    cand_pos_c = pos_maps[cand_name]
    cgt_check  = np.asarray(cnn_gt[cnn_pos_c[:4]])
    cdgt_check = np.asarray(cand_gt[cand_pos_c[:4]])
    max_gt_diff = float(np.abs(cgt_check.astype(np.float64) -
                               cdgt_check.astype(np.float64)).max())
    if max_gt_diff > 1e-3:
        print(f'\n[warning] GT arrays differ (max={max_gt_diff:.4e}) — '
              'candidate may have used different normalization.')
    else:
        print(f'OK (max_abs_diff={max_gt_diff:.2e})')

    # -----------------------------------------------------------------------
    # Helper: extract N-sample slices in common-index order
    # -----------------------------------------------------------------------
    def _slice(arr: np.ndarray, method: str) -> np.ndarray:
        return np.asarray(arr[pos_maps[method]])

    # -----------------------------------------------------------------------
    # Pre-load topology distances from merged CSV
    # -----------------------------------------------------------------------
    topo_lookup = _load_topology_from_merged_csv(merged_csv)
    print(f'  Topology lookup loaded: {len(topo_lookup)} entries '
          f'(from {merged_csv.name if merged_csv.exists() else "not found"})')

    def _get_pd_mt(method: str, sample_idx: int) -> Tuple[float, float]:
        return topo_lookup.get((method, sample_idx), (float('nan'), float('nan')))

    # -----------------------------------------------------------------------
    # Method definitions: (label, gt4d_slice, sr4d_slice)
    # -----------------------------------------------------------------------
    methods_to_run: List[Tuple[str, np.ndarray, np.ndarray]] = []

    if bic_available:
        methods_to_run.append(('bicubic',
                               _slice(cnn_gt, 'cnn'),
                               _slice(bic_sr, 'bicubic')))
    methods_to_run.append(('cnn',
                           _slice(cnn_gt, 'cnn'),
                           _slice(cnn_sr, 'cnn')))
    methods_to_run.append(('gan',
                           _slice(cnn_gt, 'cnn'),
                           _slice(gan_sr, 'gan')))
    methods_to_run.append((cand_name,
                           _slice(cnn_gt, 'cnn'),
                           _slice(cand_sr, cand_name)))

    # -----------------------------------------------------------------------
    # Compute all metrics per method
    # -----------------------------------------------------------------------
    all_rows: List[Dict] = []

    for method, gt4d, sr4d in methods_to_run:
        print(f'\nComputing metrics for [{method}] …')

        sp_gt = _speed(gt4d)       # (N,H,W)
        sp_sr = _speed(sr4d)

        psnruv         = _psnruv(sr4d, gt4d)
        speed_mae, speed_rmse = _speed_mae_rmse(sp_sr, sp_gt)
        ssim_vals      = _ssim_per_sample(sp_sr, sp_gt)

        print('  physics …', end=' ', flush=True)
        phys_rows = compute_physics_metrics(gt4d, sr4d,
                                            exceedance_thresholds=FIXED_THRESHOLDS,
                                            percentile_thresholds=PERCENTILES)
        print('done')

        print('  components …', end=' ', flush=True)
        comp_rows = _component_rows(sp_sr, sp_gt, FIXED_THRESHOLDS, PERCENTILES)
        print('done')

        for i, si in enumerate(common_idx):
            pd_dist, mt_dist = _get_pd_mt(method, si)
            phys = phys_rows[i]
            comp = comp_rows[i]

            row: Dict = {
                'method':           method,
                'sample_idx':       si,
                # Direct / vector
                'psnruv':           float(psnruv[i]),
                'ssim':             float(ssim_vals[i]),
                # Scalar speed
                'speed_mae':        float(speed_mae[i]),
                'speed_rmse':       float(speed_rmse[i]),
                # WPD
                'wpd_bias':         phys['wpd_bias'],
                'wpd_bias_abs':     abs(phys['wpd_bias']),
                'wpd_mae':          phys['wpd_mae'],
                'wpd_rmse':         phys['wpd_rmse'],
                'wpd_w1':           phys['wpd_w1'],
                # PSD
                'psd_log_l2':       phys['psd_log_l2'],
                'psd_slope_abs_delta': phys['psd_slope_abs_delta'],
                # Gradient
                'grad_mae':         phys['grad_mae'],
                'grad_w1':          phys['grad_w1'],
                'grad_kurtosis_abs_delta': phys['grad_kurtosis_abs_delta'],
                # Exceedance abs deltas
                'exceed_abs_t5':    phys['exceed_frac_abs_delta_t5'],
                'exceed_abs_t10':   phys['exceed_frac_abs_delta_t10'],
                'exceed_abs_t15':   phys['exceed_frac_abs_delta_t15'],
                'exceed_abs_p90':   phys['exceed_frac_abs_delta_p90'],
                'exceed_abs_p95':   phys['exceed_frac_abs_delta_p95'],
                'exceed_abs_p99':   phys['exceed_frac_abs_delta_p99'],
                # Exceedance SR fractions (raw)
                'exceed_sr_t5':     phys['exceed_frac_sr_t5'],
                'exceed_sr_t10':    phys['exceed_frac_sr_t10'],
                'exceed_sr_t15':    phys['exceed_frac_sr_t15'],
                # Component counts
                'comp_abs_delta_t5':  comp['comp_abs_delta_t5'],
                'comp_abs_delta_t10': comp['comp_abs_delta_t10'],
                'comp_abs_delta_t15': comp['comp_abs_delta_t15'],
                'comp_abs_delta_p90': comp['comp_abs_delta_p90'],
                'comp_abs_delta_p95': comp['comp_abs_delta_p95'],
                'comp_abs_delta_p99': comp['comp_abs_delta_p99'],
                'comp_sr_t5':         comp['comp_sr_t5'],
                'comp_sr_t10':        comp['comp_sr_t10'],
                'comp_sr_t15':        comp['comp_sr_t15'],
                'comp_sr_p90':        comp['comp_sr_p90'],
                'comp_sr_p95':        comp['comp_sr_p95'],
                'comp_sr_p99':        comp['comp_sr_p99'],
                'comp_gt_t5':         comp['comp_gt_t5'],
                'comp_gt_t10':        comp['comp_gt_t10'],
                'comp_gt_t15':        comp['comp_gt_t15'],
                'comp_gt_p90':        comp['comp_gt_p90'],
                'comp_gt_p95':        comp['comp_gt_p95'],
                'comp_gt_p99':        comp['comp_gt_p99'],
                'comp_curve_l1':      comp['comp_curve_l1'],
                # Topology distances (from merged CSV; NaN for new candidate/bicubic)
                'pd_distance':        pd_dist,
                'mt_distance':        mt_dist,
            }
            all_rows.append(row)

    print(f'\nTotal metric rows: {len(all_rows)}')

    # -----------------------------------------------------------------------
    # Write per-sample CSV
    # -----------------------------------------------------------------------
    csv_path = out_dir / f'all_sample_metrics_{cand_name}.csv'
    _write_csv(csv_path, all_rows)
    print(f'Written: {csv_path}')

    # -----------------------------------------------------------------------
    # Build per-method lookups: {method: {sample_idx: metric_dict}}
    # -----------------------------------------------------------------------
    method_data: Dict[str, Dict[int, Dict]] = {}
    for r in all_rows:
        method_data.setdefault(r['method'], {})[r['sample_idx']] = r

    # Scalar metric columns (exclude identifiers and raw fractions)
    _ID_COLS = {'method', 'sample_idx', 'wpd_bias',
                'exceed_sr_t5', 'exceed_sr_t10', 'exceed_sr_t15',
                'comp_sr_t5', 'comp_sr_t10', 'comp_sr_t15',
                'comp_sr_p90', 'comp_sr_p95', 'comp_sr_p99',
                'comp_gt_t5', 'comp_gt_t10', 'comp_gt_t15',
                'comp_gt_p90', 'comp_gt_p95', 'comp_gt_p99'}

    metric_cols = [k for k in (all_rows[0].keys() if all_rows else [])
                   if k not in _ID_COLS]

    # -----------------------------------------------------------------------
    # Determine winner direction per metric
    # -----------------------------------------------------------------------
    _lower_keys = _LOWER_IS_BETTER | {
        f'exceed_abs_{t}' for t in
        ['t5', 't10', 't15', 'p90', 'p95', 'p99']
    } | {
        f'comp_abs_delta_{t}' for t in
        ['t5', 't10', 't15', 'p90', 'p95', 'p99']
    }

    def _is_lower_better(m: str) -> bool:
        return m in _lower_keys

    # -----------------------------------------------------------------------
    # Winner-count summary
    # -----------------------------------------------------------------------
    present_methods = list(method_data.keys())
    winner_rows: List[Dict] = []

    for metric in metric_cols:
        wins = {m: 0 for m in present_methods}
        ties = 0
        n_valid = 0
        lower_better = _is_lower_better(metric)

        for si in common_idx:
            vals = {}
            for m in present_methods:
                v = method_data[m].get(si, {}).get(metric, float('nan'))
                if not math.isnan(v):
                    vals[m] = v
            if not vals:
                continue
            n_valid += 1
            best_val = min(vals.values()) if lower_better else max(vals.values())
            winners = [m for m, v in vals.items()
                       if abs(v - best_val) < 1e-12 * max(1.0, abs(best_val))]
            if len(winners) > 1:
                ties += 1
            else:
                wins[winners[0]] += 1

        row: Dict = {'metric': metric, 'lower_is_better': lower_better,
                     'n_valid': n_valid, 'n_ties': ties}
        for m in present_methods:
            row[f'wins_{m}'] = wins[m]
        winner_rows.append(row)

    winner_csv = out_dir / f'winner_counts_{cand_name}.csv'
    _write_csv(winner_csv, winner_rows)
    print(f'Written: {winner_csv}')

    # -----------------------------------------------------------------------
    # Pairwise comparison: baseline CNN vs candidate
    # -----------------------------------------------------------------------
    pairwise_rows: List[Dict] = []
    cnn_d  = method_data.get('cnn', {})
    cand_d = method_data.get(cand_name, {})

    for metric in metric_cols:
        lower_better = _is_lower_better(metric)
        vals_cnn, vals_cand = [], []
        n_improved = n_worsened = 0

        for si in common_idx:
            vc = cnn_d.get(si, {}).get(metric, float('nan'))
            vb = cand_d.get(si, {}).get(metric, float('nan'))
            if math.isnan(vc) or math.isnan(vb):
                continue
            vals_cnn.append(vc)
            vals_cand.append(vb)
            if lower_better:
                if vb < vc:
                    n_improved += 1
                elif vb > vc:
                    n_worsened += 1
            else:
                if vb > vc:
                    n_improved += 1
                elif vb < vc:
                    n_worsened += 1

        def _fmt_mean(lst):
            return float(np.mean(lst)) if lst else float('nan')

        mu_cnn  = _fmt_mean(vals_cnn)
        mu_cand = _fmt_mean(vals_cand)
        delta   = mu_cand - mu_cnn

        pairwise_rows.append({
            'metric':              metric,
            'lower_is_better':     lower_better,
            'mean_baseline_cnn':   mu_cnn,
            f'mean_{cand_name}':   mu_cand,
            'delta':               delta,
            'n_improved':          n_improved,
            'n_worsened':          n_worsened,
            'n_valid':             len(vals_cnn),
        })

    pairwise_csv = out_dir / f'pairwise_cnn_vs_{cand_name}.csv'
    _write_csv(pairwise_csv, pairwise_rows)
    print(f'Written: {pairwise_csv}')

    # -----------------------------------------------------------------------
    # Adjacent-cluster table
    # -----------------------------------------------------------------------
    adj_rows: List[Dict] = []
    for si in ADJACENT_CLUSTER_SAMPLES:
        if si not in set(common_idx):
            print(f'  [warning] adjacent-cluster sample {si} not in common set')
            continue
        row: Dict = {'sample_idx': si}
        for metric in metric_cols:
            for m in present_methods:
                v = method_data[m].get(si, {}).get(metric, float('nan'))
                row[f'{m}__{metric}'] = v

        # Derived: delta fields (candidate vs CNN) for key metrics
        def _delta_field(metric: str) -> float:
            vc = cnn_d.get(si, {}).get(metric, float('nan'))
            vb = cand_d.get(si, {}).get(metric, float('nan'))
            return vb - vc if not (math.isnan(vc) or math.isnan(vb)) else float('nan')

        row['delta_psnruv']        = _delta_field('psnruv')
        row['delta_speed_mae']     = _delta_field('speed_mae')
        row['delta_grad_mae']      = _delta_field('grad_mae')
        row['delta_exceed_abs_t5'] = _delta_field('exceed_abs_t5')

        # Winner columns per key metric
        for metric in ['psnruv', 'speed_mae', 'grad_mae',
                       'comp_curve_l1', 'pd_distance', 'mt_distance']:
            lower_better = _is_lower_better(metric)
            best = None
            best_val = None
            for m in present_methods:
                v = method_data[m].get(si, {}).get(metric, float('nan'))
                if math.isnan(v):
                    continue
                if best_val is None:
                    best, best_val = m, v
                elif (lower_better and v < best_val) or (not lower_better and v > best_val):
                    best, best_val = m, v
            row[f'winner_{metric}'] = best or 'N/A'

        # Comp-count winners per threshold
        for tag in ['t5', 't10', 't15', 'p90', 'p95', 'p99']:
            metric = f'comp_abs_delta_{tag}'
            best, best_val = None, None
            for m in present_methods:
                v = method_data[m].get(si, {}).get(metric, float('nan'))
                if math.isnan(v):
                    continue
                if best_val is None or v < best_val:
                    best, best_val = m, v
            row[f'comp_winner_{tag}'] = best or 'N/A'

        adj_rows.append(row)

    adj_csv = out_dir / f'adjacent_cluster_table_{cand_name}.csv'
    _write_csv(adj_csv, adj_rows)
    print(f'Written: {adj_csv}')

    # -----------------------------------------------------------------------
    # Build summary stats for markdown report
    # -----------------------------------------------------------------------
    def _mean_for(method: str, metric: str) -> float:
        vals = [method_data[method][si][metric]
                for si in common_idx
                if si in method_data.get(method, {}) and
                   not math.isnan(method_data[method][si].get(metric, float('nan')))]
        return float(np.mean(vals)) if vals else float('nan')

    def _relpath_for_report(path: Path) -> str:
        try:
            return str(path.resolve().relative_to(REPO_ROOT.resolve()))
        except ValueError:
            return str(path)

    def _fmt(v: float, decimals: int = 4) -> str:
        return f'{v:.{decimals}f}' if not math.isnan(v) else 'N/A'

    def _delta_str(method_a: str, method_b: str, metric: str,
                   lower_better: bool) -> str:
        va = _mean_for(method_a, metric)
        vb = _mean_for(method_b, metric)
        if math.isnan(va) or math.isnan(vb):
            return 'N/A'
        d = vb - va
        sign = '+' if d >= 0 else ''
        better = (d < 0) if lower_better else (d > 0)
        flag = ' (▲ better)' if better else ' (▼ worse)'
        return f'{sign}{d:.4f}{flag}'

    def _n_improved(metric: str) -> str:
        for r in pairwise_rows:
            if r['metric'] == metric:
                return f'{r["n_improved"]}/{r["n_valid"]}'
        return 'N/A'

    # Collect per-method mean for key metrics table
    key_metrics = [
        ('psnruv',                    False, 'PSNRuv (dB)'),
        ('ssim',                      False, 'SSIM on speed'),
        ('speed_mae',                 True,  'Speed MAE (m/s)'),
        ('speed_rmse',                True,  'Speed RMSE (m/s)'),
        ('wpd_mae',                   True,  'WPD MAE (m³/s³)'),
        ('wpd_w1',                    True,  'WPD Wasserstein-1'),
        ('wpd_bias_abs',              True,  'WPD bias abs (m³/s³)'),
        ('grad_mae',                  True,  'Gradient MAE'),
        ('grad_w1',                   True,  'Gradient W1'),
        ('grad_kurtosis_abs_delta',   True,  'Gradient kurtosis abs Δ'),
        ('psd_log_l2',                True,  'PSD log-L2'),
        ('psd_slope_abs_delta',       True,  'PSD slope abs Δ'),
        ('exceed_abs_t5',             True,  'Exceedance abs Δ s>5'),
        ('exceed_abs_t10',            True,  'Exceedance abs Δ s>10'),
        ('exceed_abs_t15',            True,  'Exceedance abs Δ s>15'),
        ('exceed_abs_p90',            True,  'Exceedance abs Δ p90'),
        ('comp_curve_l1',             True,  'Component-count curve L1'),
        ('pd_distance',               True,  'PD distance (TTK)'),
        ('mt_distance',               True,  'MT distance (TTK)'),
    ]

    # -----------------------------------------------------------------------
    # Markdown report
    # -----------------------------------------------------------------------
    md = io.StringIO()
    md.write(f'# {cand_name} fine-tuning evaluation\n\n')

    import datetime
    md.write(f'**Generated:** {datetime.date.today().isoformat()}\n\n')
    md.write(f'**Candidate:** `{cand_name}`\n\n')
    md.write(f'**Samples evaluated:** {N}\n\n')
    md.write(f'**Methods compared:** {", ".join(present_methods)}\n\n')
    if not _HAS_SKIMAGE:
        md.write('> **SSIM note:** SSIM was not computed because skimage could not '
                 'be imported in the current NumPy environment '
                 f'(`{_SKIMAGE_IMPORT_ERROR}`). All SSIM values are NaN.\n\n')
    md.write(f'> Note: PD/MT distances for `{cand_name}` are N/A — '
             'TTK has not been re-run on these outputs. '
             'Values for CNN/GAN are loaded from the pre-computed merged CSV.\n\n')

    md.write('## Summary metrics table\n\n')
    cols_here = ['Metric'] + [m.capitalize() for m in present_methods]
    md.write('| ' + ' | '.join(cols_here) + ' |\n')
    md.write('|' + '---|' * len(cols_here) + '\n')
    for metric, lower_better, label in key_metrics:
        row_vals = [label]
        for m in present_methods:
            v = _mean_for(m, metric)
            row_vals.append(_fmt(v, 4))
        md.write('| ' + ' | '.join(row_vals) + ' |\n')

    md.write('\n## 7 Key evaluation questions\n\n')

    # Q1: Direct fidelity
    md.write(f'### Q1. Did {cand_name} preserve CNN\'s direct-fidelity advantage?\n\n')
    for metric, lower_better, label in [
            ('psnruv', False, 'PSNRuv'),
            ('ssim',   False, 'SSIM'),
            ('speed_mae', True, 'Speed MAE'),
            ('speed_rmse', True, 'Speed RMSE')]:
        v_cnn  = _mean_for('cnn', metric)
        v_cand = _mean_for(cand_name, metric)
        if 'gan' in method_data:
            v_gan = _mean_for('gan', metric)
            md.write(f'- **{label}**: CNN={_fmt(v_cnn)}, GAN={_fmt(v_gan)}, '
                     f'{cand_name}={_fmt(v_cand)}. '
                     f'Δ({cand_name}−CNN)={_delta_str("cnn", cand_name, metric, lower_better)}, '
                     f'improved on {_n_improved(metric)} samples.\n')
        else:
            md.write(f'- **{label}**: CNN={_fmt(v_cnn)}, '
                     f'{cand_name}={_fmt(v_cand)}. '
                     f'Δ={_delta_str("cnn", cand_name, metric, lower_better)}.\n')
    md.write('\n')

    # Q2: Speed/WPD
    md.write(f'### Q2. Did {cand_name} improve scalar speed or WPD metrics?\n\n')
    for metric, lower_better, label in [
            ('wpd_mae', True, 'WPD MAE'),
            ('wpd_w1',  True, 'WPD W1'),
            ('wpd_bias_abs', True, 'WPD bias abs')]:
        v_cnn  = _mean_for('cnn', metric)
        v_cand = _mean_for(cand_name, metric)
        md.write(f'- **{label}**: CNN={_fmt(v_cnn)}, {cand_name}={_fmt(v_cand)}. '
                 f'Δ={_delta_str("cnn", cand_name, metric, lower_better)}, '
                 f'improved on {_n_improved(metric)} samples.\n')
    md.write('\n')

    # Q3: Gradient
    md.write(f'### Q3. Did {cand_name} improve gradient metrics?\n\n')
    for metric, lower_better, label in [
            ('grad_mae', True, 'Gradient MAE'),
            ('grad_w1',  True, 'Gradient W1'),
            ('grad_kurtosis_abs_delta', True, 'Gradient kurtosis abs Δ')]:
        v_cnn  = _mean_for('cnn', metric)
        v_cand = _mean_for(cand_name, metric)
        md.write(f'- **{label}**: CNN={_fmt(v_cnn)}, {cand_name}={_fmt(v_cand)}. '
                 f'Δ={_delta_str("cnn", cand_name, metric, lower_better)}, '
                 f'improved on {_n_improved(metric)} samples.\n')
    md.write('\n')

    # Q4: Exceedance
    md.write(f'### Q4. Did {cand_name} improve exceedance metrics, especially s>5?\n\n')
    for metric, lower_better, label in [
            ('exceed_abs_t5',  True, 'Exceedance abs Δ s>5'),
            ('exceed_abs_t10', True, 'Exceedance abs Δ s>10'),
            ('exceed_abs_t15', True, 'Exceedance abs Δ s>15'),
            ('exceed_abs_p90', True, 'Exceedance abs Δ p90')]:
        v_cnn  = _mean_for('cnn', metric)
        v_cand = _mean_for(cand_name, metric)
        md.write(f'- **{label}**: CNN={_fmt(v_cnn,5)}, {cand_name}={_fmt(v_cand,5)}. '
                 f'Δ={_delta_str("cnn", cand_name, metric, lower_better)}, '
                 f'improved on {_n_improved(metric)} samples.\n')
    md.write('\n')

    # Q5: Component counts
    md.write(f'### Q5. Did {cand_name} move component-count behavior toward GT?\n\n')
    for metric, lower_better, label in [
            ('comp_curve_l1', True, 'Component-count curve L1'),
            ('comp_abs_delta_t5',  True, 'Comp-count abs Δ t5'),
            ('comp_abs_delta_t10', True, 'Comp-count abs Δ t10'),
            ('comp_abs_delta_t15', True, 'Comp-count abs Δ t15')]:
        v_cnn  = _mean_for('cnn', metric)
        v_cand = _mean_for(cand_name, metric)
        md.write(f'- **{label}**: CNN={_fmt(v_cnn)}, {cand_name}={_fmt(v_cand)}. '
                 f'Δ={_delta_str("cnn", cand_name, metric, lower_better)}, '
                 f'improved on {_n_improved(metric)} samples.\n')
    md.write('\n')

    # Q6: PD/MT
    md.write(f'### Q6. Did {cand_name} improve PD or MT distances?\n\n')
    md.write(f'PD and MT distances for `{cand_name}` are not available in this '
             'evaluation (TTK was not re-run on these outputs). '
             'Pre-computed values for CNN and GAN:\n\n')
    for metric, lower_better, label in [
            ('pd_distance', True, 'PD distance'),
            ('mt_distance', True, 'MT distance')]:
        v_cnn = _mean_for('cnn', metric)
        v_gan = _mean_for('gan', metric) if 'gan' in method_data else float('nan')
        md.write(f'- **{label}**: CNN={_fmt(v_cnn)}, GAN={_fmt(v_gan)}, '
                 f'{cand_name}=N/A (requires TTK run)\n')
    md.write(f'\nTo compute PD/MT for `{cand_name}`: convert outputs to VTI via '
             '`scripts/convert_phire_to_vti.py`, then run '
             '`scripts/compute_composite_tree_distance.py`.\n\n')

    # Q7: Adjacent clusters
    md.write(f'### Q7. Did {cand_name} change the adjacent clusters '
             '(samples 10–13, 90–93, 162–163)?\n\n')
    md.write(f'See `{adj_csv.name}` for full detail. '
             'Summary of PSNRuv, speed MAE, and gradient MAE changes:\n\n')
    md.write('| Sample | Δ PSNRuv | Δ Speed MAE | Δ Grad MAE | Δ Exceed s>5 |\n')
    md.write('|---|---|---|---|---|\n')
    for ar in adj_rows:
        si = ar['sample_idx']
        md.write(f'| {si} '
                 f'| {_fmt(ar["delta_psnruv"],3)} '
                 f'| {_fmt(ar["delta_speed_mae"],4)} '
                 f'| {_fmt(ar["delta_grad_mae"],4)} '
                 f'| {_fmt(ar["delta_exceed_abs_t5"],5)} |\n')
    md.write('\n')

    # Pairwise summary table
    md.write(f'## Pairwise summary: CNN vs {cand_name}\n\n')
    md.write(f'| Metric | Mean CNN | Mean {cand_name} | Δ | N improved | N worsened |\n')
    md.write('|---|---|---|---|---|---|\n')
    mean_cand_col = f'mean_{cand_name}'
    for pr in pairwise_rows:
        if any(pr['metric'] == m for m, _, _ in key_metrics):
            md.write(f'| {pr["metric"]} '
                     f'| {_fmt(pr["mean_baseline_cnn"],4)} '
                     f'| {_fmt(pr[mean_cand_col],4)} '
                     f'| {_fmt(pr["delta"],4)} '
                     f'| {pr["n_improved"]} '
                     f'| {pr["n_worsened"]} |\n')
    md.write('\n')

    md.write('## Output files\n\n')
    md.write(f'- `{_relpath_for_report(csv_path)}` — per-sample metrics for all methods\n')
    md.write(f'- `{_relpath_for_report(pairwise_csv)}` — per-metric delta table\n')
    md.write(f'- `{_relpath_for_report(winner_csv)}` — win counts per method\n')
    md.write(f'- `{_relpath_for_report(adj_csv)}` — adjacent-cluster detail\n')
    md.write(f'- `{_relpath_for_report(report_p)}` — this report\n\n')

    md.write('## Notes\n\n')
    md.write('- PSNRuv is computed on physical [u,v] arrays using per-sample '
             'dynamic data range (GT max − GT min).\n')
    md.write('- All speed-based metrics use scalar speed = sqrt(u² + v²) '
             'in physical units (m/s).\n')
    md.write('- Exceedance fractions use GT percentile thresholds (p90/p95/p99) '
             'computed per sample.\n')
    md.write('- Component counts use 8-connectivity on speed superlevel sets.\n')
    md.write('- PD/MT distances for CNN and GAN are loaded from pre-computed '
             f'merged CSV; `{cand_name}` requires a fresh TTK run.\n')
    md.write(f'- `{cand_name}` was fine-tuned from the pretrained CNN checkpoint. '
             'Results may change with different lambda settings or more epochs.\n')

    report_p.write_text(md.getvalue(), encoding='utf-8')
    print(f'Written: {report_p}')

    print('\n' + '=' * 64)
    print('Evaluation complete.')
    print(f'  Outputs: {out_dir}')
    print(f'  Report:  {report_p}')
    print('=' * 64)


if __name__ == '__main__':
    main()
