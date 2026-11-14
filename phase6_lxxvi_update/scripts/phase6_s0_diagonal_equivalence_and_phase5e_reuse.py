#!/usr/bin/env python3
"""Phase 6 LXXVI: read-only metric equivalence and existing Phase-5E archive gate.

No model, TTK, VTI, PD, or audit artifacts are written or modified.
Usage: --mode metric in the frozen gudhi-audit environment; --mode phase5e
in the same environment (only parser/numpy are needed for phase5e).
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import importlib.util
import math
from pathlib import Path
import re
import sys

METHOD = 'candidateF_grad_pairzero_expanded2688'
PILOT = METHOD + '_s0_extraction_pilot_20261014'
GT_TAIL = '_s0_speed_p160_x0_y0_pd_port_0.vtu'

def require(ok, detail):
    if not ok:
        raise RuntimeError(detail)

def existing(path):
    require(path.is_file() and path.stat().st_size > 0, f'Missing/empty {path}')
    return path

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()

def reader(audit_root):
    path = existing(audit_root / 'recompute_pd' / 'canonical_pd_pilot.py')
    # Load exact archived implementation under a private module name.
    spec = importlib.util.spec_from_file_location('phase6_readonly_canonical_parser', path)
    require(spec is not None and spec.loader is not None, f'Cannot load {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.read_pd

def parse(read_pd, path):
    import numpy as np
    d, _ = read_pd(existing(path))
    result = {}
    for dim in (0, 1):
        a = np.asarray(d[dim], dtype=float)
        require(a.ndim == 2 and a.shape[1] == 2 and np.isfinite(a).all(),
                f'Invalid D{dim} in {path}: {a.shape}')
        require((a[:, 1] >= a[:, 0]).all(), f'Death < birth for D{dim}: {path}')
        result[dim] = a
    return result

def counter(a, *, positive=False):
    if positive:
        a = a[a[:, 1] > a[:, 0]]
    return Counter((float(b), float(d)) for b, d in a)

def parity(a, b, label):
    import numpy as np
    for d in (0, 1):
        x, y = a[d], b[d]
        cx, cy = counter(x), counter(y)
        pos_same = counter(x, positive=True) == counter(y, positive=True)
        zx = int(np.count_nonzero(x[:, 0] == x[:, 1]))
        zy = int(np.count_nonzero(y[:, 0] == y[:, 1]))
        print(f'[{label}] D{d} total={len(x)}/{len(y)} zero={zx}/{zy} '
              f'positive_multiset_equal={pos_same} all_multiset_equal={cx==cy}', flush=True)
        if not pos_same:
            print(f'[{label}] D{d} unmatched_positive_A={list((counter(x,positive=True)-counter(y,positive=True)).items())[:5]}')
            print(f'[{label}] D{d} unmatched_positive_B={list((counter(y,positive=True)-counter(x,positive=True)).items())[:5]}')
            raise RuntimeError(f'{label}: NONZERO D{d} discrepancy: cannot claim metric-equivalence')
    print(f'[{label}] PASS: positive finite D0/D1 pair multisets identical', flush=True)

def gudhi_dist(gt, sr):
    import numpy as np
    import gudhi
    from gudhi.wasserstein import wasserstein_distance
    db = {}; wi = {}; we = {}
    for d in (0, 1):
        a, b = gt[d], sr[d]
        db[d] = float(gudhi.bottleneck_distance(a, b, e=0.0))
        wi[d] = float(wasserstein_distance(a, b, matching=False, order=2.,
                   internal_p=np.inf, keep_essential_parts=False))
        we[d] = float(wasserstein_distance(a, b, matching=False, order=2.,
                   internal_p=2., keep_essential_parts=False))
    return dict(dB=max(db.values()), W2inf=math.hypot(wi[0], wi[1]),
                W22=math.hypot(we[0], we[1]))

def metric(repo, read_pd):
    root = repo / 'ttk_runs_fixed'
    prior = root / 'cnn' / 'pd'
    pilot = root / 'topology_finetuning' / PILOT / 'pd'
    hist_gt = parse(read_pd, prior / ('cnn_GT' + GT_TAIL))
    hist_sr = parse(read_pd, prior / ('cnn_SR' + GT_TAIL))
    new_gt = parse(read_pd, pilot / (METHOD + '_GT' + GT_TAIL))
    new_sr = parse(read_pd, pilot / (METHOD + '_SR' + GT_TAIL))
    parity(hist_gt, new_gt, 'HIST-NEW-GT')
    import numpy as np
    zero_clean_gt = {d: a[a[:,1]>a[:,0]] for d,a in new_gt.items()}
    parity(hist_gt, zero_clean_gt, 'POSITIVE-GT')
    comparisons = {
        'historical-CNN': gudhi_dist(hist_gt, hist_sr),
        'new-GT-vs-gradient': gudhi_dist(new_gt, new_sr),
        'historical-GT-vs-gradient': gudhi_dist(hist_gt, new_sr),
        'positive-only-GT-vs-gradient': gudhi_dist(zero_clean_gt, new_sr),
    }
    for label, vals in comparisons.items():
        print(f'[METRIC] {label}: ' + ' '.join(f'{k}={v:.12g}' for k,v in vals.items()), flush=True)
    require(math.isclose(comparisons['historical-CNN']['dB'], 3.0914716720581055, abs_tol=1e-10, rel_tol=1e-12),
            'Frozen historical CNN dB regression failed')
    require(math.isclose(comparisons['historical-CNN']['W2inf'],17.467813434330861, abs_tol=1e-10, rel_tol=1e-12),
            'Frozen historical CNN W2inf regression failed')
    for label in ('historical-GT-vs-gradient', 'positive-only-GT-vs-gradient'):
        for name, value in comparisons['new-GT-vs-gradient'].items():
            diff = abs(value - comparisons[label][name])
            print(f'[INVARIANT] {label} {name} abs_diff={diff:.12g}', flush=True)
            require(math.isclose(value, comparisons[label][name], abs_tol=1e-10, rel_tol=1e-12),
                    f'Numerical invariant failed: {label} {name} diff={diff}')
    print('[METRIC-GATE] PASS: all three corrected distances invariant under observed zero-persistence GT discrepancy', flush=True)

def phase5e(repo, read_pd, override):
    root = override or repo / 'spatial_pd' / 'phase5e_all168_canonical_pd'
    root = root.expanduser().resolve()
    require(root.is_dir(), f'Phase-5E canonical archive not found: {root}')
    records = sorted(root.rglob('*_pd_port_0.vtu'))
    print(f'[PHASE5E] root={root} count_pd_port0={len(records)}', flush=True)
    labels = ('GT', 'CNN_SR', 'UV_SR', 'F1_SR')
    rx = re.compile(r'(?:^|_)(GT|CNN_SR|UV_SR|F1_SR)_s(\d+)_speed_p160_x0_y0_pd_port_0\.vtu$')
    groups = {x: {} for x in labels}
    unrecognized = []
    for p in records:
        m = rx.search(p.name)
        if not m:
            unrecognized.append(str(p.relative_to(root)))
            continue
        label, idx = m.group(1), int(m.group(2))
        require(idx not in groups[label], f'Duplicate Phase-5E {label} sample {idx}')
        groups[label][idx] = p
    for label in labels:
        found = sorted(groups[label])
        missing = sorted(set(range(168)) - set(found))
        extras = sorted(set(found) - set(range(168)))
        print(f'[PHASE5E] {label}: found={len(found)} missing={missing[:12]} extras={extras[:12]}', flush=True)
    if unrecognized:
        print(f'[PHASE5E] unrecognized PD names (first 5)={unrecognized[:5]}', flush=True)
    require(all(set(groups[x]) == set(range(168)) for x in labels) and not unrecognized,
            'Phase-5E inventory not a verified 4x168 archive under expected naming; do not reuse blindly')
    phase5_gt = groups['GT'][0]
    pilot_gt = repo / 'ttk_runs_fixed/topology_finetuning' / PILOT / 'pd' / (METHOD+'_GT'+GT_TAIL)
    for label,p in [('phase5e_gt',phase5_gt), ('new_gt',pilot_gt), ('phase5e_CNN_s0',groups['CNN_SR'][0]), ('phase5e_F1_s0',groups['F1_SR'][0])]:
        print(f'[FILE] {label} bytes={p.stat().st_size} sha256={digest(p)} path={p}', flush=True)
    parity(parse(read_pd, phase5_gt), parse(read_pd, pilot_gt), 'PHASE5E-NEW-GT')
    print('[PHASE5E-GATE] PASS: 672-file archive coverage and sample-0 canonical GT positive-pair parity', flush=True)
    print('[PHASE5E-GATE] Existing CNN/F1 diagrams are candidates for reuse; full source/config continuity must also be recorded.', flush=True)

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo-root', type=Path, default=Path.home()/'PhIRE')
    ap.add_argument('--audit-root', type=Path, default=Path.home()/'phire_runtime_audit_20260809_221548')
    ap.add_argument('--phase5e-root', type=Path, default=None)
    ap.add_argument('--mode', required=True, choices=('metric', 'phase5e'))
    args = ap.parse_args()
    root = args.repo_root.expanduser().resolve()
    require(root.is_dir(), f'Not a repo folder: {root}')
    read_pd = reader(args.audit_root.expanduser().resolve())
    if args.mode == 'metric':
        metric(root, read_pd)
    else:
        phase5e(root, read_pd, args.phase5e_root)
    return 0

if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f'[LXXVI] FAIL {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        raise SystemExit(1)
