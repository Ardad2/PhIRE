#!/usr/bin/env python3
"""
APPROXIMATE constraint builder for CandidateE2-expanded-672.

*** DO NOT USE FOR THE MAIN CANDIDATEE2-EXPANDED RESULT ***

This script uses scipy local maxima + global minimum as birth/death vertex
approximations instead of actual TTK persistence-diagram vertex IDs.  It was
an early prototype written before TTK was available on the expanded dataset.

For the faithful, scientifically-correct CandidateE2-expanded-672 constraints
that use actual TTK persistence birth/death vertices, use:
  scripts/build_candidateE2_expanded672_ttk_constraints.py

This approximate helper is preserved here only as a development reference.
It must NOT be used to generate the NPZ that feeds run_candidateE2_expanded672_ttkcrit_refiner.py.

Approximation used:
  Birth vertices = local maxima of GT speed in 160×160 crop
                   (scipy maximum_filter, kernel 5×5)
  Death vertex   = global minimum of the same 160×160 crop (one per sample)
  Persistence    = birth_val - death_val (>= 0.01 × speed_range)

Input:
  data_out/wind_mrhr_cnn_expanded672/dataGT.npy  (672, 500, 500, 2)
  data_out/wind_mrhr_cnn_expanded672/idx.npy      (672,)

Output:
  ttk_runs_fixed/topology_finetuning/candidateE2approx_expanded672_constraints/
    ttk_pd_critical_pairs_gtvalues.npz

  NOTE: this helper writes to candidateE2APPROX_expanded672_constraints/ (not
  candidateE2_expanded672_constraints/).  The real E2-expanded training script
  (run_candidateE2_expanded672_ttkcrit_refiner.py) expects the faithful TTK
  constraints at candidateE2_expanded672_constraints/.  This approximate output
  must NOT be used as input to that training script.

Usage (development/reference only):
  python3 scripts/build_candidateE2approx_expanded672_constraints.py

Requires numpy and scipy (available in .mamba_candidateD_pd / .venv_candidateD_pd).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent

GT_PATH  = REPO_ROOT / 'data_out' / 'wind_mrhr_cnn_expanded672' / 'dataGT.npy'
IDX_PATH = REPO_ROOT / 'data_out' / 'wind_mrhr_cnn_expanded672' / 'idx.npy'

OUT_DIR = REPO_ROOT / 'ttk_runs_fixed' / 'topology_finetuning' / 'candidateE2approx_expanded672_constraints'
OUT_NPZ = OUT_DIR / 'ttk_pd_critical_pairs_gtvalues.npz'

# ---------------------------------------------------------------------------
# Parameters (matching E pilot for comparability)
# ---------------------------------------------------------------------------
PATCH            = 160   # crop size; vertex IDs: vid = row * PATCH + col
TOP_K            = 64    # max pairs per sample
PERSISTENCE_FRAC = 0.01  # minimum persistence as fraction of local speed range
MAX_FILTER_SIZE  = 5     # neighborhood radius for local maxima detection

N_EXPECTED = 672

_EPS = 1e-8


# ---------------------------------------------------------------------------
# Speed computation
# ---------------------------------------------------------------------------

def _speed(uv: np.ndarray) -> np.ndarray:
    """(H, W, 2) or (H, W, 2) float → (H, W) scalar wind speed."""
    return np.sqrt(uv[..., 0] ** 2 + uv[..., 1] ** 2)


# ---------------------------------------------------------------------------
# Local extrema detection
# ---------------------------------------------------------------------------

def _find_local_maxima(arr: np.ndarray, size: int) -> np.ndarray:
    """
    Return boolean mask of local maxima using scipy if available, numpy otherwise.

    A pixel is a local maximum if its value equals the maximum of all pixels
    in its (size × size) neighbourhood.
    """
    try:
        from scipy.ndimage import maximum_filter
        return arr == maximum_filter(arr, size=size)
    except ImportError:
        pass

    # Numpy fallback using sliding_window_view (numpy >= 1.20)
    try:
        from numpy.lib.stride_tricks import sliding_window_view
        H, W = arr.shape
        pad = size // 2
        padded = np.pad(arr, pad, mode='edge')
        windows = sliding_window_view(padded, (size, size))  # (H, W, size, size)
        neighbourhood_max = windows.max(axis=(-2, -1))
        return arr == neighbourhood_max
    except Exception:
        pass

    # Last-resort manual implementation
    H, W = arr.shape
    pad = size // 2
    result = np.zeros((H, W), dtype=bool)
    for r in range(H):
        for c in range(W):
            r0, r1 = max(0, r - pad), min(H, r + pad + 1)
            c0, c1 = max(0, c - pad), min(W, c + pad + 1)
            if arr[r, c] >= arr[r0:r1, c0:c1].max():
                result[r, c] = True
    return result


def _find_persistent_pairs(
    speed_crop: np.ndarray,
    top_k: int,
    persistence_frac: float,
    max_filter_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Find simplified persistence-diagram birth/death pairs from a speed field.

    Superlevel-set convention (matching TTK):
      - Birth vertex = local maximum of speed (high-speed peak)
      - Death vertex = global minimum of crop (last saddle when all components merge)
      - Persistence = birth_val - death_val

    Returns (birth_vid, death_vid, birth_val, death_val, persistence) as 1D arrays.
    All vertex IDs are flat indices: vid = row * PATCH + col.

    Returns empty arrays if no pairs survive the persistence threshold.
    """
    H, W = speed_crop.shape  # both == PATCH

    s_min = float(speed_crop.min())
    s_max = float(speed_crop.max())
    s_range = s_max - s_min
    pers_threshold = persistence_frac * max(s_range, _EPS)

    # ── Birth vertices: local maxima ──────────────────────────────────────
    is_lmax = _find_local_maxima(speed_crop, size=max_filter_size)
    birth_ys, birth_xs = np.where(is_lmax)

    if len(birth_ys) == 0:
        empty = np.array([], dtype=np.int64)
        emptyf = np.array([], dtype=np.float32)
        return empty, empty, emptyf, emptyf, emptyf

    birth_vals = speed_crop[birth_ys, birth_xs].astype(np.float32)

    # ── Death vertex: global minimum of the crop ──────────────────────────
    flat_min_idx = int(np.argmin(speed_crop))
    death_y = flat_min_idx // W
    death_x = flat_min_idx % W
    death_val_scalar = float(speed_crop[death_y, death_x])

    # ── Filter by persistence ─────────────────────────────────────────────
    persistences = birth_vals - death_val_scalar
    valid = persistences >= pers_threshold

    birth_ys  = birth_ys[valid]
    birth_xs  = birth_xs[valid]
    birth_vals = birth_vals[valid]
    persistences = persistences[valid]

    if len(birth_ys) == 0:
        empty = np.array([], dtype=np.int64)
        emptyf = np.array([], dtype=np.float32)
        return empty, empty, emptyf, emptyf, emptyf

    # ── Sort by persistence descending, take top_k ────────────────────────
    order = np.argsort(persistences)[::-1][:top_k]
    birth_ys  = birth_ys[order]
    birth_xs  = birth_xs[order]
    birth_vals = birth_vals[order]
    persistences = persistences[order]

    n = len(birth_ys)
    death_vids = np.full(n, death_y * W + death_x, dtype=np.int64)
    death_vals = np.full(n, death_val_scalar,       dtype=np.float32)

    return (
        (birth_ys * W + birth_xs).astype(np.int64),
        death_vids,
        birth_vals,
        death_vals,
        persistences.astype(np.float32),
    )


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

def _sanity_check(
    gt: np.ndarray,
    sample_idx: np.ndarray,
    sample_start: np.ndarray,
    sample_count: np.ndarray,
    birth_vid: np.ndarray,
    death_vid: np.ndarray,
    birth_val: np.ndarray,
    death_val: np.ndarray,
    patch: int,
) -> None:
    """
    Verify that every stored birth_val/death_val matches the actual GT speed at
    the corresponding vertex.  Aborts with details if max error >= 1e-5.
    """
    print('Sanity check: verifying stored values match GT speed at each vertex …')
    max_b_err = 0.0
    max_d_err = 0.0
    W = patch

    for row_i in range(len(sample_idx)):
        sid  = int(sample_idx[row_i])
        # Find the position in gt: we need the original array index, not the WTK index.
        # We use row_i directly since gt is indexed 0..671 in the same order as sample_idx.
        start = int(sample_start[row_i])
        count = int(sample_count[row_i])
        if count == 0:
            continue

        spd = _speed(gt[row_i, :patch, :patch, :])  # (patch, patch)

        bvids = birth_vid[start : start + count]
        dvids = death_vid[start : start + count]
        bvals = birth_val[start : start + count]
        dvals = death_val[start : start + count]

        b_ys, b_xs = bvids // W, bvids % W
        d_ys, d_xs = dvids // W, dvids % W

        computed_bval = spd[b_ys, b_xs].astype(np.float32)
        computed_dval = spd[d_ys, d_xs].astype(np.float32)

        b_err = float(np.abs(computed_bval - bvals).max())
        d_err = float(np.abs(computed_dval - dvals).max())

        max_b_err = max(max_b_err, b_err)
        max_d_err = max(max_d_err, d_err)

        if b_err > 1e-4 or d_err > 1e-4:
            print(f'  [WARN] sample_idx={sid}: birth_err={b_err:.2e}  death_err={d_err:.2e}')

    print(f'  max birth_val error: {max_b_err:.2e}')
    print(f'  max death_val error: {max_d_err:.2e}')

    if max_b_err >= 1e-4 or max_d_err >= 1e-4:
        sys.exit(
            f'[error] Sanity check FAILED: birth_err={max_b_err:.2e}, death_err={max_d_err:.2e}.\n'
            'Stored values do not match GT speed at the stored vertex positions.\n'
            'This indicates a bug in the constraint-generation code.'
        )
    print('  Sanity check PASSED.')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print('=' * 64)
    print('Build CandidateE2-expanded-672 constraints')
    print('=' * 64)
    print(f'  GT input  : {GT_PATH}')
    print(f'  IDX input : {IDX_PATH}')
    print(f'  Output    : {OUT_NPZ}')
    print(f'  PATCH     : {PATCH}')
    print(f'  TOP_K     : {TOP_K}')
    print(f'  PERS_FRAC : {PERSISTENCE_FRAC}')
    print(f'  FILTER    : {MAX_FILTER_SIZE}×{MAX_FILTER_SIZE}')
    print()

    # ── Pre-condition checks ──────────────────────────────────────────────
    if not GT_PATH.exists():
        sys.exit(
            f'[error] Expanded CNN GT arrays not found: {GT_PATH}\n'
            '  Generate them first:\n'
            '    python3 - <<\'PY\'\n'
            '    import sys; sys.path.insert(0, \'.\')\n'
            '    import tensorflow.compat.v1 as tf; tf.disable_v2_behavior()\n'
            '    from PhIREGANs import PhIREGANs\n'
            '    phire = PhIREGANs(\n'
            '        data_type=\'wind_mrhr_cnn_expanded672\',\n'
            '        mu_sig=[[0.7684, -0.4575], [5.02455, 5.9017]],\n'
            '    )\n'
            '    phire.set_data_out_path(\'data_out/wind_mrhr_cnn_expanded672\')\n'
            '    phire.test_paired(\n'
            '        r=[5],\n'
            '        data_path=\'example_data_topology_expanded_672/wind_MR-HR.tfrecord\',\n'
            '        model_path=\'models/wind_mr-hr/trained_cnn/cnn\',\n'
            '        batch_size=1, save_inputs=True,\n'
            '    )\n'
            '    PY'
        )

    # ── Load GT ──────────────────────────────────────────────────────────
    print('Loading GT arrays (mmap) …')
    gt = np.load(GT_PATH, mmap_mode='r')   # (672, 500, 500, 2)
    print(f'  GT shape: {gt.shape}  dtype: {gt.dtype}')

    if gt.shape[0] != N_EXPECTED:
        sys.exit(
            f'[error] GT array has {gt.shape[0]} samples; expected {N_EXPECTED}.\n'
            f'  Regenerate: {GT_PATH}'
        )
    if gt.shape[1] < PATCH or gt.shape[2] < PATCH:
        sys.exit(
            f'[error] GT spatial size {gt.shape[1]}×{gt.shape[2]} < PATCH={PATCH}.'
        )

    if IDX_PATH.exists():
        idx_all = np.load(IDX_PATH).astype(np.int64)
        if len(idx_all) != N_EXPECTED:
            sys.exit(
                f'[error] idx.npy has {len(idx_all)} entries; expected {N_EXPECTED}.'
            )
        print(f'  idx range: [{int(idx_all.min())}, {int(idx_all.max())}]')
    else:
        print(f'  [warn] idx.npy not found; using 0..{N_EXPECTED - 1}')
        idx_all = np.arange(N_EXPECTED, dtype=np.int64)

    # ── Generate per-sample constraints ──────────────────────────────────
    print()
    print('Extracting constraints …')

    all_birth_vid: list[np.ndarray] = []
    all_death_vid: list[np.ndarray] = []
    all_birth_val: list[np.ndarray] = []
    all_death_val: list[np.ndarray] = []
    all_persistence: list[np.ndarray] = []
    sample_start: list[int] = []
    sample_count: list[int] = []

    pair_counts = []
    offset = 0

    for i in range(N_EXPECTED):
        # 160×160 crop of GT speed (read as float64, downcast to float32 for speed)
        spd_crop = _speed(np.asarray(gt[i, :PATCH, :PATCH, :], dtype=np.float64)).astype(np.float32)

        bvid, dvid, bval, dval, pers = _find_persistent_pairs(
            spd_crop, TOP_K, PERSISTENCE_FRAC, MAX_FILTER_SIZE,
        )

        n = len(bvid)
        sample_start.append(offset)
        sample_count.append(n)
        offset += n
        pair_counts.append(n)

        all_birth_vid.append(bvid)
        all_death_vid.append(dvid)
        all_birth_val.append(bval)
        all_death_val.append(dval)
        all_persistence.append(pers)

        if (i + 1) % 100 == 0 or i == 0:
            print(f'  [{i + 1:4d}/{N_EXPECTED}]  sample_idx={int(idx_all[i]):6d}  pairs={n}')

    print()
    pair_counts_arr = np.array(pair_counts)
    print(f'  Total pairs : {int(pair_counts_arr.sum())}')
    print(f'  Pairs/sample: min={pair_counts_arr.min()}  '
          f'mean={pair_counts_arr.mean():.1f}  max={pair_counts_arr.max()}')
    zero_pair_samples = int((pair_counts_arr == 0).sum())
    if zero_pair_samples > 0:
        print(f'  [warn] {zero_pair_samples} samples have 0 pairs '
              f'(persistence threshold may be too high).')

    birth_vid_all  = np.concatenate(all_birth_vid)  if all_birth_vid  else np.array([], dtype=np.int64)
    death_vid_all  = np.concatenate(all_death_vid)  if all_death_vid  else np.array([], dtype=np.int64)
    birth_val_all  = np.concatenate(all_birth_val)  if all_birth_val  else np.array([], dtype=np.float32)
    death_val_all  = np.concatenate(all_death_val)  if all_death_val  else np.array([], dtype=np.float32)
    persistence_all = np.concatenate(all_persistence) if all_persistence else np.array([], dtype=np.float32)

    sample_start_arr = np.array(sample_start, dtype=np.int64)
    sample_count_arr = np.array(sample_count, dtype=np.int64)

    # ── Sanity check ────────────────────────────────────────────────────
    print()
    _sanity_check(
        gt=np.asarray(gt),  # load into memory for sanity check
        sample_idx=idx_all,
        sample_start=sample_start_arr,
        sample_count=sample_count_arr,
        birth_vid=birth_vid_all,
        death_vid=death_vid_all,
        birth_val=birth_val_all,
        death_val=death_val_all,
        patch=PATCH,
    )

    # ── Write NPZ ────────────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print()
    print(f'Writing: {OUT_NPZ}')
    np.savez(
        str(OUT_NPZ),
        n_samples   = np.int64(N_EXPECTED),
        sample_idx  = idx_all,
        sample_start= sample_start_arr,
        sample_count= sample_count_arr,
        birth_vid   = birth_vid_all,
        death_vid   = death_vid_all,
        birth_val   = birth_val_all,
        death_val   = death_val_all,
        persistence = persistence_all,
    )

    # ── Verify written NPZ ──────────────────────────────────────────────
    print('Verifying written NPZ …')
    check = np.load(str(OUT_NPZ), allow_pickle=True)
    required = {'n_samples', 'sample_idx', 'sample_start', 'sample_count',
                'birth_vid', 'death_vid', 'birth_val', 'death_val', 'persistence'}
    missing  = required - set(check.files)
    if missing:
        sys.exit(f'[error] Written NPZ is missing keys: {missing}')

    assert int(check['n_samples']) == N_EXPECTED
    assert len(check['sample_idx']) == N_EXPECTED
    assert len(check['sample_start']) == N_EXPECTED
    assert len(check['sample_count']) == N_EXPECTED
    print(f'  n_samples   : {int(check["n_samples"])}')
    print(f'  total pairs : {len(check["birth_vid"])}')
    print(f'  sample_idx range: [{int(check["sample_idx"].min())}, {int(check["sample_idx"].max())}]')

    print()
    print('=' * 64)
    print('Done.')
    print(f'  Constraints: {OUT_NPZ}')
    print()
    print('Next step: train CandidateE2-expanded-672:')
    print('  micromamba run -p /home/adadhwal/PhIRE/.mamba_candidateD_pd \\')
    print('    python scripts/run_candidateE2_expanded672_ttkcrit_refiner.py')
    print('=' * 64)


if __name__ == '__main__':
    main()
