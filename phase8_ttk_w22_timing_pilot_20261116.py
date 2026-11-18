#!/usr/bin/env python3
"""Phase 8: non-training TTK + canonical GUDHI W2,2 matching/timing pilot.

Read-only with respect to PhIRE datasets, checkpoints, and historical metrics.
Writes only a new, isolated --out-dir in benchmark mode.  All numerical
results are produced on the user's Spark machine, not by this source file.

Conventions: native 500x500 [u,v] -> physical scalar speed (m/s) -> top-left
160x160; C-order VTK points; TTK PD -t 1; canonical Phase 6 parser; finite
H0/H1; GUDHI Wasserstein order=2, internal_p=2, keep_essential_parts=False;
D0/D1 combined by hypot; diagonal Euclidean squared cost = (d-b)^2/2.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET

import numpy as np

GT_DEFAULT = 'data_out/wind_mrhr_cnn_expanded2688/dataGT.npy'
SR_DEFAULT = 'data_out/wind_mrhr_cnn_expanded2688/dataSR.npy'
CANONICAL_REL = Path.home() / 'phire_runtime_audit_20260809_221548/recompute_pd/canonical_pd_pilot.py'
PILOT_REL = Path('phase6_lxxvii_update/scripts/phase6_pilot_pd_gudhi_gate.py')
OUT_DEFAULT = 'ttk_runs_fixed/topology_finetuning/phase8_timing_s000_20261116'


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda: f.read(1024*1024), b''):
            h.update(data)
    return h.hexdigest()


def module_from_path(path, name):
    require(path.is_file(), f'Required canonical source missing: {path}')
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, f'Cannot load {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def speed_crop(sample):
    """Physical-speed convention from the existing E2 constraint builder."""
    a = np.asarray(sample, dtype=np.float32)
    require(a.shape == (500, 500, 2), f'Expected one 500x500x2 wind field, got {a.shape}')
    s = np.sqrt(a[..., 0]**2 + a[..., 1]**2)
    crop = np.ascontiguousarray(s[:160, :160], dtype=np.float32)
    require(np.isfinite(crop).all(), 'Nonfinite wind-speed values')
    return crop


def write_vti(path, field):
    """Write ASCII VTI with the same point indexing as repaired E2 code."""
    h,w = field.shape
    flat = np.asarray(field, dtype=np.float32).ravel(order='C')
    data = '\n          '.join(' '.join(f'{v:.8g}' for v in flat[k:k+8])
                              for k in range(0, flat.size, 8))
    path.write_text(
        '<?xml version="1.0"?>\n'
        '<VTKFile type="ImageData" version="0.1" byte_order="LittleEndian">\n'
        f'  <ImageData WholeExtent="0 {w-1} 0 {h-1} 0 0" Origin="0 0 0" Spacing="1 1 1">\n'
        f'    <Piece Extent="0 {w-1} 0 {h-1} 0 0">\n'
        '      <PointData Scalars="wind_speed">\n'
        '        <DataArray type="Float32" Name="wind_speed" format="ascii" NumberOfComponents="1">\n'
        f'          {data}\n'
        '        </DataArray>\n'
        '      </PointData>\n'
        '    </Piece>\n'
        '  </ImageData>\n'
        '</VTKFile>\n'
    )
    # Independent XML numeric readback: detect historical Fortran/C-order error.
    root = ET.parse(path).getroot()
    da = root.find('.//DataArray[@Name="wind_speed"]')
    require(da is not None, f'VTI point array missing: {path}')
    recovered = np.fromstring(da.text or '', sep=' ', dtype=np.float32)
    require(recovered.size == h*w, 'VTI point count mismatch')
    recovered = recovered.reshape(h,w,order='C')
    # ASCII uses 8 significant digits, so roundtrip tolerance is needed.
    require(np.allclose(recovered, field, atol=1e-5, rtol=1e-6),
            f'VTI coordinate/numerical readback mismatch: maxerr={np.max(np.abs(recovered-field))}')


def diag_cost2(p):
    return float((float(p[1]) - float(p[0]))**2 / 2)


def manual_match_squared(gt, sr, pairs):
    """Independently reconstruct matching cost and validate full coverage."""
    m = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    seen_gt, seen_sr = set(), set()
    total = 0.
    counts = {'matched': 0, 'gt_to_diagonal': 0, 'sr_to_diagonal': 0}
    for i,j in m:
        i,j = int(i),int(j)
        require(i >= 0 or j >= 0, 'Invalid diagonal-to-diagonal edge')
        if i >= 0:
            require(i < len(gt) and i not in seen_gt, f'Invalid/repeated GT point {i}')
            seen_gt.add(i)
        if j >= 0:
            require(j < len(sr) and j not in seen_sr, f'Invalid/repeated SR point {j}')
            seen_sr.add(j)
        if i >= 0 and j >= 0:
            total += float(np.sum((gt[i]-sr[j])**2))
            counts['matched'] += 1
        elif i >= 0:
            total += diag_cost2(gt[i]);counts['gt_to_diagonal'] += 1
        else:
            total += diag_cost2(sr[j]);counts['sr_to_diagonal'] += 1
    require(len(seen_gt)==len(gt) and len(seen_sr)==len(sr),
            f'Matching is incomplete: {len(seen_gt)}/{len(gt)} GT and {len(seen_sr)}/{len(sr)} SR')
    return total, counts


def tiny_diagram_tests():
    from gudhi.wasserstein import wasserstein_distance
    x = np.array([[0., 2.]], dtype=np.float64)
    empty = np.empty((0, 2), dtype=np.float64)
    displaced = np.array([[.2, 2.1]], dtype=np.float64)
    spurious = np.array([[0., 2.], [3.,5.]], dtype=np.float64)
    cases = [('identity',x,x,0.),
             ('displaced',x,displaced,math.sqrt(.2**2+.1**2)),
             ('extra-feature',x,spurious,math.sqrt(2.)),
             ('unmatched-to-diagonal',x,empty,math.sqrt(2.))]
    for name, g, s, expected in cases:
        d, matching = wasserstein_distance(g,s,order=2.,internal_p=2.,
            matching=True,keep_essential_parts=False)
        cost2, counts = manual_match_squared(g,s,matching)
        require(math.isclose(d, expected,abs_tol=1e-8,rel_tol=1e-8),
                f'{name}: expected W22 {expected}, received {d}')
        require(math.isclose(d*d,cost2,abs_tol=1e-8,rel_tol=1e-8),
                f'{name}: matching squared cost mismatch {cost2} vs {d*d}')
        print(f'[SYNTHETIC] {name}: W22={d:.8g}, {counts} PASS',flush=True)
    # Non-symmetric orientation regression: original F-order bug must be caught.
    with tempfile.TemporaryDirectory() as d:
        field = np.arange(35,dtype=np.float32).reshape(5,7)
        field[1,5] = 1000
        write_vti(Path(d)/'orientation.vti',field)
    print('[SELFTEST] PASS: diagonal cost, matching coverage, displacement, spurious feature and C-order',flush=True)


def check_backend(backend):
    if backend == 'host':
        require(shutil.which('ttkPersistenceDiagramCmd'),
                'Host ttkPersistenceDiagramCmd unavailable; use --ttk-backend docker')
    else:
        require(shutil.which('docker'), 'Docker unavailable; use --ttk-backend host if installed')
        p = subprocess.run(['docker','image','inspect','phire-ttk:latest'],
                           capture_output=True,text=True,timeout=30)
        require(p.returncode == 0,
                'Local Docker image phire-ttk:latest not found; do not silently pull another version')


def run_ttk(backend, base, name, timeout):
    inp = base / 'vti' / f'{name}.vti'
    prefix = base / 'pd' / f'{name}_pd'
    dest = base / 'pd' / f'{name}_pd_port_0.vtu'
    if backend=='host':
        cmd = ['ttkPersistenceDiagramCmd','-t','1','-i',str(inp),
               '-a','wind_speed','-o',str(prefix)]
    else:
        cmd = ['docker','run','--rm','--user',f'{os.getuid()}:{os.getgid()}',
               '-v',f'{base.resolve()}:/work','phire-ttk:latest',
               'ttkPersistenceDiagramCmd','-t','1','-i',f'/work/vti/{name}.vti',
               '-a','wind_speed','-o',f'/work/pd/{name}_pd']
    start=time.perf_counter()
    p = subprocess.run(cmd,capture_output=True,text=True,timeout=timeout)
    elapsed=time.perf_counter()-start
    require(p.returncode == 0,
            f'TTK failed ({name}), exit={p.returncode}: {(p.stderr+p.stdout)[-1600:]}')
    require(dest.is_file() and dest.stat().st_size>0, f'Missing TTK port-0 diagram {dest}')
    print(f'[TTK] {name}: {elapsed:.3f}s, {dest.stat().st_size} bytes',flush=True)
    return {'seconds':round(elapsed,6), 'vtu_bytes':dest.stat().st_size,
            'backend':backend, 'stdout_tail':p.stdout[-500:], 'stderr_tail':p.stderr[-500:]}


def worker_match(repo, base, max_matching_points):
    """Child process: matching time can be bounded via subprocess timeout."""
    import gudhi
    from gudhi.wasserstein import wasserstein_distance as w2
    pilot = module_from_path(repo/PILOT_REL,'phase8_trusted_pilot')
    canonical = module_from_path(CANONICAL_REL,'phase8_canonical_decoder')
    paths = {name:base/'pd'/f'{name}_pd_port_0.vtu' for name in ('training_gt','training_sr')}
    diagrams = {key:pilot.parse(canonical.read_pd,p) for key,p in paths.items()}
    ref_all = pilot.distance(diagrams['training_gt'],diagrams['training_sr'])
    scores, dims = [], {}
    for dim in (0,1):
        raw_gt,raw_sr = diagrams['training_gt'][dim],diagrams['training_sr'][dim]
        # Removing exactly zero-persistence pairs is W2-equivalent to matching them
        # to the diagonal at zero cost; other positive finite pairs are retained.
        g = np.ascontiguousarray(raw_gt[raw_gt[:,1] > raw_gt[:,0]],dtype=np.float64)
        s = np.ascontiguousarray(raw_sr[raw_sr[:,1] > raw_sr[:,0]],dtype=np.float64)
        require(len(g)+len(s)<=max_matching_points,
                f'D{dim}: {len(g)+len(s)} positive pairs across GT+SR exceed safety cap '+
                f'{max_matching_points}. No pruning performed; benchmark stopped to avoid unbounded POT allocation.')
        t0=time.perf_counter()
        val,match=w2(g,s,order=2.,internal_p=2.,matching=True,keep_essential_parts=False)
        dt=time.perf_counter()-t0
        require(math.isfinite(val), f'Non-finite GUDHI D{dim} W22')
        squared,counts=manual_match_squared(g,s,match)
        require(math.isclose(squared,val*val,rel_tol=1e-6,abs_tol=1e-6),
                f'D{dim}: matching reconstruction {squared} != W22^2 {val*val}')
        dims[f'D{dim}']={'all_finite_gt':len(raw_gt),'all_finite_sr':len(raw_sr),
                         'positive_gt':len(g),'positive_sr':len(s),
                         'matching_seconds':round(dt,6), 'w22':float(val),
                         'matched_cost_squared':float(squared), **counts}
        scores.append(float(val))
    combined = math.hypot(*scores)
    require(math.isclose(combined,ref_all['W22'],rel_tol=1e-6,abs_tol=1e-6),
            f'Corrected Phase6 metric regression failed: {combined} vs {ref_all["W22"]}')
    result={'format':'phase8-matching-worker-v1','dimensions':dims,
            'canonical_metrics':ref_all,'matching_w22':combined,
            'matching_seconds_sum':sum(x['matching_seconds'] for x in dims.values()),
            'versions':{'numpy':np.__version__,
                        'gudhi':str(getattr(gudhi,'__version__','unavailable'))},
            'canonical_parser_sha256':sha256(CANONICAL_REL),
            'phase6_pilot_sha256':sha256(repo/PILOT_REL)}
    (base/'matching.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(f'[MATCH] PASS: W22={combined:.8g}, D0/D1 decomposed; '
          f'matching_seconds={result["matching_seconds_sum"]:.3f}',flush=True)


def arrays_preflight(repo,args):
    gt=repo/args.gt_npy
    sr=repo/args.sr_npy
    require(gt.is_file() and sr.is_file(),
            f'Training prediction arrays missing: {gt} / {sr}. '
            'Supply --gt-npy and --sr-npy from an expanded TRAINING set, not 168 benchmark.')
    a=np.load(gt,mmap_mode='r',allow_pickle=False)
    b=np.load(sr,mmap_mode='r',allow_pickle=False)
    require(a.ndim==4 and a.shape[1:]==(500,500,2) and b.shape==a.shape,
            f'Unexpected training arrays: GT={a.shape}, SR={b.shape}')
    require(len(a)==2688, f'Expected the 2,688-field training source, found {len(a)}; do not use the 168-field benchmark')
    require(0<=args.sample_index<len(a),'sample-index out of bounds')
    require('expanded' in str(gt) and 'expanded' in str(sr),
            'Paths must refer to expanded TRAINING arrays; no 168-field held-out data')
    return gt,sr,a,b


def preflight(repo,args):
    require((repo/PILOT_REL).is_file(),f'Phase6 canonical metric source missing: {repo/PILOT_REL}')
    require(CANONICAL_REL.is_file(),f'Canonical VTU decoder missing: {CANONICAL_REL}')
    check_backend(args.ttk_backend)
    gt,sr,a,b=arrays_preflight(repo,args)
    speed_crop(a[args.sample_index]);speed_crop(b[args.sample_index])
    from gudhi.wasserstein import wasserstein_distance as unused  # noqa: F401
    print(f'[PREFLIGHT] PASS: training arrays={a.shape}, sample={args.sample_index}, '
          f'TTK backend={args.ttk_backend}, canonical parser and GUDHI available',flush=True)
    print(f'[SOURCE] GT={gt} SR={sr}',flush=True)
    return gt,sr,a,b


def benchmark(repo,args):
    gt,sr,a,b=preflight(repo,args)
    output=(repo/args.out_dir).resolve()
    require(not output.exists(),f'Output already exists; refusing overwrite: {output}')
    output.parent.mkdir(parents=True,exist_ok=True)
    base=Path(tempfile.mkdtemp(prefix='.'+output.name+'.staging-',dir=output.parent))
    try:
        (base/'vti').mkdir();(base/'pd').mkdir()
        samples={'training_gt':speed_crop(a[args.sample_index]),
                 'training_sr':speed_crop(b[args.sample_index])}
        for name,crop in samples.items():
            write_vti(base/'vti'/f'{name}.vti',crop)
        ttk={name:run_ttk(args.ttk_backend,base,name,args.ttk_timeout)
             for name in ('training_gt','training_sr')}
        cmd=[sys.executable,str(Path(__file__).resolve()),'--mode','_match-worker',
             '--repo-root',str(repo),'--working-dir',str(base),
             '--max-matching-points',str(args.max_matching_points)]
        t0=time.perf_counter()
        p=subprocess.run(cmd,capture_output=True,text=True,timeout=args.match_timeout)
        worker_wall=time.perf_counter()-t0
        require(p.returncode==0,
                f'GUDHI matching worker failed ({p.returncode}): {(p.stdout+p.stderr)[-2500:]}')
        print(p.stdout.strip(),flush=True)
        matching=json.loads((base/'matching.json').read_text())
        record={'format':'phase8-ttk-w22-timing-pilot-v1',
                'status':'PASS_NONTRAINING_TIMING_ONLY',
                'sample_index':args.sample_index, 'crop':'top-left 160x160 scalar wind speed m/s',
                'no_training':True,'no_held_out_168':True,
                'ttk':ttk,'matching':matching,'matching_worker_wall_seconds':round(worker_wall,6),
                'end_to_end_seconds':round(sum(v['seconds'] for v in ttk.values())+worker_wall,6),
                'input_paths':{'gt':str(gt),'sr':str(sr)},
                'input_shapes':list(a.shape),
                'script_sha256':sha256(Path(__file__).resolve()),
                'input_sample_sha256':{
                    'gt':hashlib.sha256(np.ascontiguousarray(a[args.sample_index]).tobytes()).hexdigest(),
                    'sr':hashlib.sha256(np.ascontiguousarray(b[args.sample_index]).tobytes()).hexdigest()},
                'input_array_sha256':'NOT_HASHED_FULL_2688_ARRAYS_TO_AVOID_GB_READ',
                'source_code_sha256':{'phase6_pilot':sha256(repo/PILOT_REL),
                                      'canonical_decoder':sha256(CANONICAL_REL)}}
        (base/'benchmark.json').write_text(json.dumps(record,sort_keys=True,indent=2)+'\n')
        members=['vti/training_gt.vti','vti/training_sr.vti',
                 'pd/training_gt_pd_port_0.vtu','pd/training_sr_pd_port_0.vtu',
                 'matching.json','benchmark.json']
        manifest=''.join(f'{sha256(base/rel)}  {rel}\n' for rel in members)
        (base/'SHA256SUMS').write_text(manifest)
        base.rename(output)
        print(f'[BENCHMARK] PASS. New immutable report at {output}/benchmark.json',flush=True)
        print(f'[BENCHMARK] end-to-end={record["end_to_end_seconds"]:.3f}s; '
              f'matching-only={matching["matching_seconds_sum"]:.3f}s',flush=True)
    finally:
        if base.exists():
            shutil.rmtree(base)


def verify(repo,args):
    out=(repo/args.out_dir).resolve()
    require(out.is_dir(),f'Missing benchmark directory: {out}')
    lines=(out/'SHA256SUMS').read_text().splitlines()
    require(len(lines)==6,'Manifest must contain exactly 6 files')
    for line in lines:
        digest,rel=line.split('  ',1)
        require(sha256(out/rel)==digest,f'SHA drift: {rel}')
    rec=json.loads((out/'benchmark.json').read_text())
    require(rec['status']=='PASS_NONTRAINING_TIMING_ONLY' and rec['no_training'] is True,
            'Benchmark metadata changed')
    require(rec['source_code_sha256']['canonical_decoder']==sha256(CANONICAL_REL),
            'Canonical decoder drift since this benchmark')
    require(rec['source_code_sha256']['phase6_pilot']==sha256(repo/PILOT_REL),
            'Phase6 metric source drift since this benchmark')
    require(rec['script_sha256']==sha256(Path(__file__).resolve()),
            'Benchmark script drift since this benchmark')
    print(f'[VERIFY] PASS: 6 output artifacts, canonical decoder SHA; {out}',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['selftest','preflight','benchmark','verify','_match-worker'],required=True)
    p.add_argument('--repo-root',type=Path,default=Path.cwd())
    p.add_argument('--sample-index',type=int,default=0)
    p.add_argument('--gt-npy',default=GT_DEFAULT)
    p.add_argument('--sr-npy',default=SR_DEFAULT)
    p.add_argument('--ttk-backend',choices=['docker','host'],default='docker')
    p.add_argument('--ttk-timeout',type=int,default=180)
    p.add_argument('--match-timeout',type=int,default=180)
    p.add_argument('--max-matching-points',type=int,default=5000)
    p.add_argument('--out-dir',default=OUT_DEFAULT)
    p.add_argument('--working-dir',type=Path,help=argparse.SUPPRESS)
    args=p.parse_args()
    root=args.repo_root.resolve()
    if args.mode=='selftest':tiny_diagram_tests()
    elif args.mode=='preflight':preflight(root,args)
    elif args.mode=='benchmark':benchmark(root,args)
    elif args.mode=='verify':verify(root,args)
    elif args.mode=='_match-worker':
        require(args.working_dir is not None,'Missing internal working directory')
        worker_match(root,args.working_dir.resolve(),args.max_matching_points)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        print(f'[PHASE8] FAIL {type(exc).__name__}: {exc}',file=sys.stderr)
        sys.exit(1)
