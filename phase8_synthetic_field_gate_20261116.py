#!/usr/bin/env python3
"""Phase 8 synthetic-field gate: unchanged, spurious, and relocated wind-speed peaks.

Non-training, independent of the 168-hour benchmark. Uses the *same* native
TTK executable, canonical Phase-6 numerical decoder, GUDHI W2,2 implementation,
C-order VTI writer and matching-cost reconstruction as the accepted Phase-8
TTK timing pilot. Writes only a new, immutable output directory.

`displaced` demonstrates the location-blindness of ordinary persistence W2;
we intentionally do NOT require displaced W2 to be large. The `spurious`
case is an explicit falsification test: it should generate an unmatched SR
feature while leaving the GT fixed-pair sample positions unchanged.
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
import sys
import tempfile
import time

import numpy as np

PILOT_NAME = 'phase8_ttk_w22_timing_pilot_20261116.py'
CANONICAL = Path.home() / 'phire_runtime_audit_20260809_221548/recompute_pd/canonical_pd_pilot.py'
PHASE6 = Path('phase6_lxxvii_update/scripts/phase6_pilot_pd_gudhi_gate.py')
EXTRACT = Path('scripts/extract_ttk_pd_critical_pairs.py')
OUT_DEFAULT = 'ttk_runs_fixed/topology_finetuning/phase8_synthetic_fields_20261116'
CROP = 160
NAMES = ('gt', 'identity', 'spurious', 'displaced')


def need(flag, msg):
    if not flag:
        raise RuntimeError(msg)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def load(path, name):
    need(Path(path).is_file(), f'Missing source: {path}')
    spec = importlib.util.spec_from_file_location(name, path)
    need(spec is not None and spec.loader is not None, f'Cannot import {path}')
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def bump(y, x, cy, cx, rad, height):
    q = (y-cy)**2 / rad**2 + (x-cx)**2 / rad**2
    return height * np.maximum(1.0-q, 0.0)**3


def fields():
    y, x = np.mgrid[0:CROP, 0:CROP].astype(np.float64)
    # Small non-symmetric slope breaks ties and catches x/y inversion.
    background = 2.5 + 0.00091*x + 0.00173*y
    original = bump(y,x,52,56,16,12)
    extra = bump(y,x,117,110,12,9)
    moved = bump(y,x,83,95,16,12)
    results = {
        'gt': background + original,
        'identity': background + original,
        'spurious': background + original + extra,
        'displaced': background + moved,
    }
    return {k:np.ascontiguousarray(v,dtype=np.float32) for k,v in results.items()}


def fixed_pair_proxy(gt, sr, gt_pairs):
    """Candidate F CV/PERS *at fixed GT vertices*, without loss weights."""
    flat_gt = gt.ravel(order='C')
    flat_sr = sr.ravel(order='C')
    b = gt_pairs['birth_vid'].astype(int)
    d = gt_pairs['death_vid'].astype(int)
    need(len(b)>0 and len(d)==len(b), 'No valid GT persistence anchors')
    need(np.max(b)<gt.size and np.max(d)<gt.size and np.min(b)>=0 and np.min(d)>=0,
         'GT anchor point IDs out of bounds')
    gb,gd = flat_gt[b].astype(float),flat_gt[d].astype(float)
    sb,sd = flat_sr[b].astype(float),flat_sr[d].astype(float)
    cv=.5*(np.mean((sb-gb)**2)+np.mean((sd-gd)**2))
    pers=np.mean((np.abs(sd-sb)-np.abs(gd-gb))**2)
    return {'cv_mse':float(cv),'pers_mse':float(pers),'selected_gt_pairs':len(b),
            'gt_endpoint_changed_count':int(np.count_nonzero(sb!=gb)+np.count_nonzero(sd!=gd))}


def compare(gt, pred, w2_impl, manual):
    from gudhi.wasserstein import wasserstein_distance
    canonical=w2_impl.distance(gt,pred)
    dims={}
    components=[]
    for dim in (0,1):
        g_all=np.asarray(gt[dim],dtype=np.float64)
        p_all=np.asarray(pred[dim],dtype=np.float64)
        g=np.ascontiguousarray(g_all[g_all[:,1]>g_all[:,0]],dtype=np.float64)
        p=np.ascontiguousarray(p_all[p_all[:,1]>p_all[:,0]],dtype=np.float64)
        need(len(g)+len(p)<=20000, f'D{dim} pairs too numerous: {len(g)} + {len(p)}')
        t0=time.perf_counter()
        distance,matches=wasserstein_distance(g,p,order=2.,internal_p=2.,
                         matching=True,keep_essential_parts=False)
        dt=time.perf_counter()-t0
        sq,counts=manual(g,p,matches)
        need(math.isfinite(distance), f'D{dim} non-finite distance')
        need(math.isclose(sq,float(distance)**2,rel_tol=1e-6,abs_tol=1e-6),
             f'D{dim}: reconstructed matching cost mismatch')
        dims[f'D{dim}']={'gt_finite_positive':len(g),'sr_finite_positive':len(p),
                          'w22':float(distance),'squared_cost':float(sq),
                          'matching_seconds':float(dt),**counts}
        components.append(float(distance))
    d=math.hypot(*components)
    need(math.isclose(d,canonical['W22'],rel_tol=1e-6,abs_tol=1e-6),
         f'Canonical W22 regression: {d} vs {canonical["W22"]}')
    return {'W22':d,'canonical_metrics':canonical,'dimensions':dims,
            'matching_seconds':sum(v['matching_seconds'] for v in dims.values())}


def prerequisites(repo, pilot, backend, check_runtime=True):
    need((repo/PILOT_NAME).is_file(),f'Missing previous Phase 8 script {repo/PILOT_NAME}')
    need((repo/PHASE6).is_file(),f'Missing validated Phase-6 GUDHI pilot {repo/PHASE6}')
    need((repo/EXTRACT).is_file(),f'Missing TTK pair parser {repo/EXTRACT}')
    need(CANONICAL.is_file(),f'Missing canonical parser {CANONICAL}')
    if check_runtime:
        import gudhi  # noqa: F401
        pilot.check_backend(backend)


def synthetic_selftest(pilot):
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        fs=fields()
        need(all(v.shape==(160,160) for v in fs.values()),'Shape error')
        need(all(np.isfinite(v).all() and v.min()>0 for v in fs.values()),'Bad scalar speeds')
        need(np.array_equal(fs['identity'],fs['gt']),'Identity differs from GT')
        need(np.count_nonzero(fs['spurious']!=fs['gt'])>0,'Spurious bump absent')
        need(np.count_nonzero(fs['spurious']!=fs['gt'])<1000,'Spurious bump not local')
        need(not np.array_equal(fs['displaced'],fs['gt']),'Moved bump absent')
        a=Path(d)/'test.vti'
        pilot.write_vti(a,fs['spurious'])
        for k,v in fs.items():
            need(v.dtype==np.float32,f'{k}: incorrect dtype')
    print('[SELFTEST] PASS: scalar fields 160x160, compact spurious bump, C-order VTI readback; no TTK executed',flush=True)


def benchmark(repo, out, backend, timeout, pilot):
    import gudhi
    prerequisites(repo,pilot,backend,True)
    synthetic_selftest(pilot)
    need(not out.exists(),f'Refusing existing output directory: {out}')
    out.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='.'+out.name+'.staging-',dir=out.parent))
    try:
        (stage/'vti').mkdir();(stage/'pd').mkdir();(stage/'fields').mkdir()
        fs=fields()
        source_hashes={str(p.relative_to(repo)):sha(p) for p in
               (repo/PILOT_NAME,repo/PHASE6,repo/EXTRACT)}
        source_hashes[str(CANONICAL)]=sha(CANONICAL)
        ttk={}
        for name in NAMES:
            np.save(stage/'fields'/f'{name}.npy',fs[name])
            pilot.write_vti(stage/'vti'/f'{name}.vti',fs[name])
            ttk[name]=pilot.run_ttk(backend,stage,name,timeout)
        phase6=load(repo/PHASE6,'phase8_field_phase6')
        decoder=load(CANONICAL,'phase8_field_canonical')
        extraction=load(repo/EXTRACT,'phase8_field_constraint_parser')
        diags={name:phase6.parse(decoder.read_pd,
                         stage/'pd'/f'{name}_pd_port_0.vtu') for name in NAMES}
        gt_raw=extraction.parse_vtu(stage/'pd'/'gt_pd_port_0.vtu')
        gt_pairs=extraction.extract_constraints(gt_raw,0,'synthetic_gt',0.01,64,CROP)
        need(gt_pairs is not None,'No finite GT TTK pairs passed 1% persistence criterion')
        comparisons={}
        for name in NAMES[1:]:
            match=compare(diags['gt'],diags[name],phase6,pilot.manual_match_squared)
            fixed=fixed_pair_proxy(fs['gt'],fs[name],gt_pairs)
            comparisons[name]={**match,'fixed_gt_vertex_proxy':fixed}
            c=sum(x['sr_to_diagonal'] for x in match['dimensions'].values())
            print(f'[FIELD] {name:9s}: W22={match["W22"]:.8g}, '
                  f'SR->diagonal={c}, fixed-CV={fixed["cv_mse"]:.6g}, '
                  f'fixed-PERS={fixed["pers_mse"]:.6g}',flush=True)
        identity=comparisons['identity']
        require_zero=math.isclose(identity['W22'],0.,abs_tol=1e-7)
        need(require_zero,'Identical GT/SR fields produced nonzero W22')
        sp=comparisons['spurious']
        nd=sum(v['sr_to_diagonal'] for v in sp['dimensions'].values())
        need(nd>0,'Spurious peak did not generate an unmatched SR diagram point; synthetic construction must be revised')
        need(sp['W22']>1e-4,'Spurious peak did not raise diagram W22')
        need(sp['fixed_gt_vertex_proxy']['gt_endpoint_changed_count']==0,
             'Spurious peak touched GT fixed-pair anchor(s); synthetic construction must be revised')
        need(sp['fixed_gt_vertex_proxy']['cv_mse']==0.0 and
             sp['fixed_gt_vertex_proxy']['pers_mse']==0.0,
             'Spurious peak unexpectedly changes fixed GT-pair loss')
        # Displacement may be topology-equivalent: W22 can be small or zero!
        displaced=comparisons['displaced']
        need(displaced['fixed_gt_vertex_proxy']['cv_mse']>0,
             'Displaced peak was not detected by fixed GT critical-value proxy')
        expected=[*('fields/'+name+'.npy' for name in NAMES),
                  *('vti/'+name+'.vti' for name in NAMES),
                  *('pd/'+name+'_pd_port_0.vtu' for name in NAMES)]
        manifest={'schema':'phase8-synthetic-fields-v1','source_hashes':source_hashes,
                  'ttk_backend':backend,'ttk_timing':ttk,
                  'gt_selected_fixed_pairs':len(gt_pairs['birth_vid']),
                  'comparisons':comparisons,
                  'interpretation':{
                     'spurious':'The extra peak is outside fixed GT endpoint locations, but incurs a predicted-PD matching/diagonal penalty.',
                     'displaced':'Pure PD W22 is location-blind; a displaced peak can have a small PD distance even when GT-vertex supervision detects displacement.',
                     'important':'No gradient or TensorFlow optimizer step was tested; matching assignments must be translated to prediction-dependent critical-vertex targets next.'},
                  'gudhi_version':str(getattr(gudhi,'__version__','installed')),
                  'files_sha256':{f:sha(stage/f) for f in expected}}
        (stage/'results.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
        need(not out.exists(),f'Destination unexpectedly appeared: {out}')
        stage.rename(out)
        print(f'[SYNTHETIC-FIELD-GATE] PASS: spurious features detected with dynamic matching but invisible to fixed GT anchors; output={out}',flush=True)
        print(f'[SPATIAL-CAVEAT] displaced W22={displaced["W22"]:.8g}; W22 does not encode geographic location',flush=True)
    finally:
        if stage.is_dir():shutil.rmtree(stage)


def verify(repo,out,pilot):
    need(out.is_dir(),f'Missing output {out}')
    m=json.loads((out/'results.json').read_text())
    need(m['schema']=='phase8-synthetic-fields-v1','Wrong report format')
    for rel,expected in m['files_sha256'].items():
        need((out/rel).is_file() and sha(out/rel)==expected, f'File hash mismatch {rel}')
    for path,expected in m['source_hashes'].items():
        source=Path(path) if Path(path).is_absolute() else repo/path
        need(source.is_file() and sha(source)==expected, f'Source drift {source}')
    phase6=load(repo/PHASE6,'phase8_field_phase6_verify')
    decoder=load(CANONICAL,'phase8_field_decoder_verify')
    diags={name:phase6.parse(decoder.read_pd,
                     out/'pd'/f'{name}_pd_port_0.vtu') for name in NAMES}
    for name in NAMES[1:]:
        calc=compare(diags['gt'],diags[name],phase6,pilot.manual_match_squared)
        old=m['comparisons'][name]
        need(math.isclose(calc['W22'],old['W22'],rel_tol=1e-8,abs_tol=1e-8),
             f'{name}: W22 changed')
        for d in ('D0','D1'):
            for key in ('matched','sr_to_diagonal','gt_to_diagonal'):
                need(calc['dimensions'][d][key]==old['dimensions'][d][key],
                     f'{name}: {d} {key} matching count changed')
    print(f'[VERIFY] PASS: hashed 12 field/VTI/PD artifacts, source code hashes and all canonical W22/matching counts; {out}',flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode',choices=('selftest','preflight','benchmark','verify'),required=True)
    ap.add_argument('--repo-root',type=Path,default=Path('.'))
    ap.add_argument('--ttk-backend',choices=('docker','host'),default='docker')
    ap.add_argument('--ttk-timeout',type=int,default=180)
    ap.add_argument('--out-dir',default=OUT_DEFAULT)
    args=ap.parse_args()
    repo=args.repo_root.resolve()
    pilot=load(repo/PILOT_NAME,'phase8_field_prior_pilot')
    if args.mode=='selftest':
        synthetic_selftest(pilot)
    elif args.mode=='preflight':
        prerequisites(repo,pilot,args.ttk_backend,True)
        print(f'[PREFLIGHT] PASS: TTK={args.ttk_backend}, canonical Phase-6 H0/H1 decoder and GUDHI available, no historical output modified',flush=True)
    elif args.mode=='benchmark':
        benchmark(repo,(repo/args.out_dir).resolve(),args.ttk_backend,args.ttk_timeout,pilot)
    else:
        verify(repo,(repo/args.out_dir).resolve(),pilot)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        print(f'[PHASE8-FIELDS] ERROR {type(exc).__name__}: {exc}',file=sys.stderr)
        sys.exit(1)
