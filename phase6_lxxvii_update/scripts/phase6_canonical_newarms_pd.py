#!/usr/bin/env python3
"""Phase 6 LXXVII. Isolated, resumable SR-only canonical PD extraction.

Run inside the existing phire-ttk:latest Docker image, with ~/PhIRE mounted
at /work. Does not modify source arrays, Phase-5E PDs, CNN/F1 or legacy metrics.
Stages: preflight (read only), pilot (s0, all arms), full (s1..167), verify.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ARM = {
    'grad': 'candidateF_grad_pairzero_expanded2688',
    'cv': 'candidateF_grad_CV_only_expanded2688',
    'pers': 'candidateF_grad_PERS_only_expanded2688',
}
N = 168
CANONICAL_GT_S0_SHA = '85db754ceac029a8deeea85896c03af9993fc953a323a1000d8b9c1b310638af'
ROOT_REL = 'ttk_runs_fixed/topology_finetuning/phase6_lxxvii_canonical_pd_newarms_20261014'
GT_REFERENCE = 'spatial_pd/phase5e_all168_canonical_pd/pd'


def ensure(ok: bool, message: str):
    if not ok:
        raise RuntimeError(message)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def file_ok(path: Path) -> Path:
    ensure(path.is_file() and path.stat().st_size > 0, f'Missing/empty file: {path}')
    return path


def get_paths(repo: Path, method: str):
    inp = repo / 'data_out' / ('wind_finetune_' + method)
    return {kind: file_ok(inp / fn) for kind, fn in (
        ('GT', 'dataGT.npy'), ('SR', 'dataSR.npy'), ('idx', 'idx.npy'))}


def root_for(repo: Path, arm: str) -> Path:
    return repo / ROOT_REL / arm


def stem(method: str, i: int) -> str:
    return f'{method}_SR_s{i}_speed_p160_x0_y0'


def vti_file(base: Path, method: str, i: int) -> Path:
    return base / 'vti' / (stem(method, i) + '.vti')


def pd_file(base: Path, method: str, i: int) -> Path:
    return base / 'pd' / (stem(method, i) + '_pd_port_0.vtu')


def reference_gate(repo: Path):
    reference = repo / GT_REFERENCE
    ensure(reference.is_dir(), f'Missing Phase-5E PD directory: {reference}')
    gt = [reference / f'phase5_GT_s{i}_speed_p160_x0_y0_pd_port_0.vtu' for i in range(N)]
    for p in gt:
        file_ok(p)
    ensure(sha(gt[0]) == CANONICAL_GT_S0_SHA,
           'Frozen Phase5E GT sample-0 SHA differs; stop and inspect source continuity')
    print(f'[REFERENCE] PASS: {N} canonical Phase5E GT PD VTUs; frozen s0 SHA matches', flush=True)


def validate_arrays(repo: Path, method: str):
    import numpy as np
    fixed = get_paths(repo, method)
    benchmark = repo / 'data_out_fixed' / 'wind_mrhr_cnn'
    base_gt = file_ok(benchmark / 'dataGT.npy')
    base_idx = file_ok(benchmark / 'idx.npy')
    ensure(sha(fixed['GT']) == sha(base_gt), f'{method}: GT file SHA differs from benchmark')
    ensure(sha(fixed['idx']) == sha(base_idx), f'{method}: idx file SHA differs from benchmark')
    idx = np.load(fixed['idx'])
    ensure(idx.shape == (N,) and np.array_equal(idx, np.arange(N)),
           f'{method}: indices not exact ordered 0..167: {idx}')
    gt = np.load(fixed['GT'], mmap_mode='r')
    sr = np.load(fixed['SR'], mmap_mode='r')
    ensure(gt.shape == (N, 500, 500, 2) and sr.shape == gt.shape,
           f'{method}: GT/SR shape mismatch {gt.shape} vs {sr.shape}')
    ensure(gt.dtype == np.float64 and sr.dtype == np.float64,
           f'{method}: unexpected dtype GT/SR: {gt.dtype}/{sr.dtype}')
    for i in range(N):
        ensure(np.isfinite(sr[i]).all(), f'{method}: nonfinite SR at index {i}')
    fingerprints = {k: sha(path) for k, path in fixed.items()}
    print(f'[INPUT] PASS {method}: {N} ordered, finite, aligned fields; source SHA256 captured', flush=True)
    return fingerprints


def manifest_payload(repo: Path, method: str, hashes):
    converter = file_ok(repo / 'scripts' / 'convert_phire_to_vti.py')
    this = file_ok(repo / 'phase6_lxxvii_update' / 'scripts' / 'phase6_canonical_newarms_pd.py')
    return {
        'format': 'phase6-lxxvii-sr-pd-v1', 'method': method, 'sample_positions': list(range(N)),
        'reference_gt': GT_REFERENCE, 'patch': 160, 'x0': 0, 'y0': 0,
        'scalar': 'speed', 'ttk_cmd': 'ttkPersistenceDiagramCmd', 'threads': 1,
        'source_sha256': hashes, 'converter_sha256': sha(converter), 'driver_sha256': sha(this),
    }


def write_manifest_or_check(base: Path, payload):
    manifest = base / 'input_manifest.json'
    if manifest.is_file():
        got = json.loads(manifest.read_text())
        ensure(got == payload, f'Manifest conflict in {manifest}; do not resume with changed inputs/scripts')
    else:
        ensure(not base.exists() or not any(base.iterdir()),
               f'Output directory exists without manifest: {base}. Do not overwrite')
        base.mkdir(parents=True, exist_ok=True)
        tmp = base / '.input_manifest.tmp'
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
        tmp.rename(manifest)
    (base / 'vti').mkdir(exist_ok=True)
    (base / 'pd').mkdir(exist_ok=True)


def verify_vti(path: Path, sr, i: int):
    import numpy as np
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy
    file_ok(path)
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(str(path)); reader.Update()
    image = reader.GetOutput()
    ensure(image is not None and image.GetDimensions() == (160, 160, 1), f'Wrong VTI dimensions {path}')
    arr = image.GetPointData().GetArray('wind_speed')
    ensure(arr is not None, f'Missing wind_speed in {path}')
    a = vtk_to_numpy(arr).reshape(160, 160)
    patch = np.asarray(sr[i, :160, :160, :])
    expected = np.sqrt(patch[...,0].astype(np.float32)**2 + patch[...,1].astype(np.float32)**2)
    ensure(np.array_equal(a, expected),
           f'VTI data mismatch/orientation at {path}; max abs diff={np.max(np.abs(a-expected))}')


def verify_pd(path: Path):
    import vtk
    file_ok(path)
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(path)); reader.Update()
    grid = reader.GetOutput()
    ensure(grid is not None and grid.GetNumberOfCells() > 0,
           f'Unreadable/empty PD unstructured grid: {path}')


def choose_methods(arm: str):
    return ARM.items() if arm == 'all' else [(arm, ARM[arm])]


def process(repo: Path, mode: str, arm: str):
    ensure((repo / 'scripts' / 'convert_phire_to_vti.py').is_file(),
           'Canonical VTI writer missing')
    reference_gate(repo)
    for name, method in choose_methods(arm):
        hashes = validate_arrays(repo, method)
        base = root_for(repo, name)
        if mode == 'preflight':
            print(f'[PREFLIGHT] {name}: output target={base} (not created)', flush=True)
            continue
        if mode in ('pilot', 'full'):
            write_manifest_or_check(base, manifest_payload(repo, method, hashes))
        else:
            ensure((base / 'input_manifest.json').is_file(), f'Missing {name} manifest')
            ensure(json.loads((base / 'input_manifest.json').read_text()) == manifest_payload(repo, method, hashes),
                   f'Manifest mismatch for {name}')
        sr = __import__('numpy').load(get_paths(repo, method)['SR'], mmap_mode='r')
        samples = [0] if mode == 'pilot' else list(range(N))
        if mode == 'full':
            gate = repo / ROOT_REL / 'pilot_verified.json'
            ensure(gate.is_file(), 'Missing GUDHI pilot_verified.json. Run pilot gate first')
            checked = json.loads(gate.read_text())
            for n, other in ARM.items():
                p0 = file_ok(pd_file(root_for(repo, n), other, 0))
                ensure(checked['pd_sha256'].get(n) == sha(p0),
                       f'Pilot validation mismatch for {n}; redo GUDHI pilot gate')
            print('[CONTINUITY] PASS: all three PD sample-0 hashes match the independent GUDHI pilot gate', flush=True)
        missing_vti = []
        for i in samples:
            p = vti_file(base, method, i)
            if p.exists():
                verify_vti(p, sr, i)
            else:
                missing_vti.append(i)
        if mode in ('pilot', 'full') and missing_vti:
            converter = [sys.executable, str(repo / 'scripts' / 'convert_phire_to_vti.py'),
                         '--input', str(get_paths(repo, method)['SR']),
                         '--outdir', str(base / 'vti'), '--label', method + '_SR',
                         '--scalar', 'speed', '--patch', '160', '--x0', '0', '--y0', '0',
                         '--samples', *map(str, missing_vti)]
            print(f'[VTI] {name}: generating {len(missing_vti)} explicitly selected samples', flush=True)
            subprocess.run(converter, check=True)
        for i in samples:
            verify_vti(vti_file(base, method, i), sr, i)
        print(f'[VTI] {name} PASS count={len(samples)}', flush=True)
        for j, i in enumerate(samples, 1):
            out = pd_file(base, method, i)
            if not out.is_file():
                ensure(mode in ('pilot', 'full'), f'Missing output during verify: {out}')
                prefix = base / 'pd' / (stem(method, i) + '_pd')
                cmd = ['ttkPersistenceDiagramCmd', '-t', '1', '-i', str(vti_file(base, method, i)),
                       '-a', 'wind_speed', '-o', str(prefix)]
                subprocess.run(cmd, check=True)
            verify_pd(out)
            if j % 10 == 0 or j == len(samples):
                print(f'[PD] {name} validated {j}/{len(samples)}', flush=True)
        print(f'[{mode.upper()}] PASS {name}: {len(samples)} SR VTIs and PD port-0 VTUs', flush=True)
    print(f'[LXXVII] PASS mode={mode} arm={arm}. No legacy distances computed.', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo-root', type=Path, default=Path('/work'))
    ap.add_argument('--mode', choices=['preflight', 'pilot', 'full', 'verify'], required=True)
    ap.add_argument('--arm', choices=['all', *ARM], default='all')
    args = ap.parse_args()
    ensure(args.repo_root.is_dir(), f'No repository: {args.repo_root}')
    process(args.repo_root.resolve(), args.mode, args.arm)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f'[LXXVII] ERROR {type(e).__name__}: {e}', file=sys.stderr, flush=True)
        raise SystemExit(1)
