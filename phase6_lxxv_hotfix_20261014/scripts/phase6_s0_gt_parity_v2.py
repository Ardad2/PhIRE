#!/usr/bin/env python3
"""Phase 6 sample-0, read-only GT parity (LXXV hotfix).

Historical CNN diagrams are archived under ttk_runs_fixed/cnn/pd, but its
original VTI is not required or assumed to exist. Never modifies artifacts.

Modes:
  inventory  - locate and hash source .npy, new VTI, and both archived PDs
  arrays-vti - compare benchmark source arrays, new canonical VTI against array
  pd        - compare historical and newly extracted finite GT PD multisets

Optional --historical-vti can supply an archived original VTI from elsewhere.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib
from pathlib import Path
import sys

METHOD = 'candidateF_grad_pairzero_expanded2688'
PILOT = METHOD + '_s0_extraction_pilot_20261014'
NEW_STEM = METHOD + '_GT_s0_speed_p160_x0_y0'
OLD_STEM = 'cnn_GT_s0_speed_p160_x0_y0'


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def one(folder: Path, exact_name: str) -> Path:
    path = folder / exact_name
    require(path.is_file() and path.stat().st_size > 0,
            f'Missing or empty required file: {path}')
    return path.resolve()


def source_paths(repo: Path, supplied_old_vti: Path | None):
    historical = repo / 'ttk_runs_fixed' / 'cnn'
    pilot = repo / 'ttk_runs_fixed' / 'topology_finetuning' / PILOT
    require(historical.is_dir(), f'Missing historical CNN tree: {historical}')
    require(pilot.is_dir(), f'Missing new pilot tree: {pilot}')
    result = {
        'hist_pd': one(historical / 'pd', OLD_STEM + '_pd_port_0.vtu'),
        'new_pd': one(pilot / 'pd', NEW_STEM + '_pd_port_0.vtu'),
        'new_vti': one(pilot / 'vti', NEW_STEM + '.vti'),
        'hist_gt_array': one(repo / 'data_out_fixed' / 'wind_mrhr_cnn', 'dataGT.npy'),
        'new_gt_array': one(repo / 'data_out' / ('wind_finetune_' + METHOD), 'dataGT.npy'),
        'hist_idx': one(repo / 'data_out_fixed' / 'wind_mrhr_cnn', 'idx.npy'),
        'new_idx': one(repo / 'data_out' / ('wind_finetune_' + METHOD), 'idx.npy'),
    }
    if supplied_old_vti is not None:
        candidate = supplied_old_vti.expanduser().resolve()
        require(candidate.is_file() and candidate.stat().st_size > 0,
                f'--historical-vti is missing or empty: {candidate}')
        result['hist_vti'] = candidate
    else:
        # Do not recursively guess a historical VTI's identity. Archive may
        # intentionally preserve only TTK PD/MT VTUs, not original VTI.
        candidate = historical / 'vti' / (OLD_STEM + '.vti')
        if candidate.is_file() and candidate.stat().st_size > 0:
            result['hist_vti'] = candidate.resolve()
        else:
            print('[HIST VTI] Absent from the conventional historical vti/ folder; '
                  'this is not an error. Historical PD is available.', flush=True)
    return result


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def inventory(paths):
    for name, path in paths.items():
        print(f'[FILE] {name} bytes={path.stat().st_size} sha256={sha256(path)} path={path}', flush=True)
    print('[INVENTORY] PASS: required arrays, new GT VTI and both GT PD VTUs located', flush=True)


def vti_field(path):
    import numpy as np
    from vtkmodules.vtkIOXML import vtkXMLImageDataReader
    from vtkmodules.util.numpy_support import vtk_to_numpy
    rdr = vtkXMLImageDataReader()
    rdr.SetFileName(str(path))
    rdr.Update()
    o = rdr.GetOutput()
    require(o is not None and tuple(o.GetDimensions()) == (160, 160, 1),
            f'Unexpected VTI dimensions in {path}: {o.GetDimensions() if o else None}')
    point = o.GetPointData()
    arr = point.GetArray('wind_speed')
    require(arr is not None, f'Missing wind_speed array in {path}')
    active = point.GetScalars()
    require(active is not None and active.GetName() == 'wind_speed',
            f'Incorrect active scalars in {path}')
    raw = np.asarray(vtk_to_numpy(arr))
    require(raw.size == 160 * 160 and np.isfinite(raw).all(),
            f'Invalid or nonfinite wind_speed data in {path}')
    return raw.reshape((160, 160))


def max_abs(a, b):
    import numpy as np
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def arrays_vti(paths):
    import numpy as np
    ids = {}
    fields = {}
    for kind in ('hist', 'new'):
        ids[kind] = np.load(paths[kind + '_idx'], allow_pickle=False)
        fields[kind] = np.load(paths[kind + '_gt_array'], allow_pickle=False, mmap_mode='r')
        require(ids[kind].ndim == 1 and len(ids[kind]) >= 1 and int(ids[kind][0]) == 0,
                f'{kind}: first idx not equal to 0')
        require(fields[kind].ndim == 4 and fields[kind].shape[-1] == 2,
                f'{kind}: unexpected GT tensor shape {fields[kind].shape}')
        print(f'[ARRAY] {kind} shape={fields[kind].shape} dtype={fields[kind].dtype} '
              f'sample0_idx={ids[kind][0]}', flush=True)
    a = np.asarray(fields['hist'][0]); b = np.asarray(fields['new'][0])
    require(a.shape == b.shape, f'Sample-0 GT shape mismatch {a.shape} vs {b.shape}')
    equal_gt = bool(np.array_equal(a, b))
    equal_idx = bool(np.array_equal(ids['hist'], ids['new']))
    print(f'[ARRAY] sample0 GT vectors equal={equal_gt} max_abs_diff={max_abs(a,b):.12g}', flush=True)
    print(f'[ARRAY] all benchmark idx vectors equal={equal_idx}', flush=True)

    actual = vti_field(paths['new_vti'])
    sample = b[:160, :160]
    u = sample[..., 0].astype(np.float32)
    v = sample[..., 1].astype(np.float32)
    expected = np.sqrt(u * u + v * v)
    eq = bool(np.array_equal(actual, expected))
    close = bool(np.allclose(actual, expected, rtol=0, atol=1e-5))
    transposed = bool(np.array_equal(actual, expected.T))
    print(f'[VTI] new GT vs canonical NumPy speed exact={eq} '
          f'allclose_atol_1e-5={close} max_abs_diff={max_abs(actual,expected):.12g} '
          f'exact_transpose={transposed}', flush=True)

    if 'hist_vti' in paths:
        old = vti_field(paths['hist_vti'])
        print(f'[HIST VTI] new vs historical raw exact={bool(np.array_equal(actual,old))} '
              f'max_abs_diff={max_abs(actual,old):.12g}', flush=True)
        print(f'[HIST VTI] new vs transpose(historical) exact={bool(np.array_equal(actual,old.T))} '
              f'max_abs_diff={max_abs(actual,old.T):.12g}', flush=True)
    else:
        print('[HIST VTI] Comparison skipped: original historical VTI not located '
              '(historical PD comparison remains available)', flush=True)

    if equal_gt and equal_idx and close:
        print('[GT FIELD] PASS: matched sample-0 GT arrays/indices and canonical new VTI speed', flush=True)
    else:
        print('[GT FIELD] FAIL: source-array/index or VTI scalar mismatch; pause further topology work', flush=True)
        raise RuntimeError('GT field parity gate failed')


def pd_compare(paths, audit_root):
    import numpy as np
    parser = audit_root / 'recompute_pd' / 'canonical_pd_pilot.py'
    require(parser.is_file(), f'Archived canonical parser missing: {parser}')
    sys.path.insert(0, str(parser.parent))
    canonical = importlib.import_module('canonical_pd_pilot')
    old, _ = canonical.read_pd(paths['hist_pd'])
    new, _ = canonical.read_pd(paths['new_pd'])
    parity = True
    for dim in (0, 1):
        aa = np.asarray(old[dim], dtype=float)
        bb = np.asarray(new[dim], dtype=float)
        require(aa.ndim == bb.ndim == 2 and aa.shape[1] == bb.shape[1] == 2,
                f'D{dim} unexpected shapes {aa.shape} / {bb.shape}')
        require(np.isfinite(aa).all() and np.isfinite(bb).all(), 'Non-finite PD birth/death pairs')
        sa = aa[np.lexsort((aa[:, 1], aa[:, 0]))]
        sb = bb[np.lexsort((bb[:, 1], bb[:, 0]))]
        same = bool(np.array_equal(sa, sb))
        parity &= same
        z_old = int(np.count_nonzero(aa[:,1] == aa[:,0]))
        z_new = int(np.count_nonzero(bb[:,1] == bb[:,0]))
        print(f'[PD] D{dim} historical={len(aa)} new={len(bb)} '
              f'exact_birth_death_multiset={same} zero_persistence_hist/new={z_old}/{z_new}', flush=True)
        if len(aa) == len(bb):
            print(f'[PD] D{dim} sorted_pair_max_abs_diff={max_abs(sa,sb):.12g}', flush=True)
        if not same:
            ca = Counter(map(tuple,aa)); cb = Counter(map(tuple,bb))
            print(f'[PD] D{dim} unmatched_hist_example={list((ca-cb).items())[:6]}', flush=True)
            print(f'[PD] D{dim} unmatched_new_example={list((cb-ca).items())[:6]}', flush=True)
    if parity:
        print('[GT DIAGRAM] MATCH: identical finite GT persistence pairs', flush=True)
    else:
        print('[GT DIAGRAM] DIFFER: GT PD multisets are not identical. '
              'Historical orientation differences were already documented in Phase 5A; '
              'do not attribute this solely to TTK thread count.', flush=True)
    print('[NOTE] This run compares diagrams; it does not compute new model distances.', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo-root', type=Path, default=Path.home() / 'PhIRE')
    ap.add_argument('--audit-root', type=Path, default=Path.home() / 'phire_runtime_audit_20260809_221548')
    ap.add_argument('--historical-vti', type=Path, default=None,
                    help='Optional known actual historical CNN GT VTI path (not required)')
    ap.add_argument('--mode', choices=('inventory', 'arrays-vti', 'pd'), required=True)
    args = ap.parse_args()
    paths = source_paths(args.repo_root.expanduser().resolve(), args.historical_vti)
    if args.mode == 'inventory':
        inventory(paths)
    elif args.mode == 'arrays-vti':
        arrays_vti(paths)
    else:
        pd_compare(paths, args.audit_root.expanduser().resolve())
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f'[PARITY] ERROR {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        raise SystemExit(1)
