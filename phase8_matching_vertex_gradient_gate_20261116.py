#!/usr/bin/env python3
"""Phase 8 read-only matching -> predicted critical vertex -> gradient gate.

Uses the ACCEPTED Phase 8 synthetic scalar fields and TTK VTUs. Checks that
canonical GUDHI PD points can be connected to predicted TTK vertex indices,
that a local fixed-assignment squared-W22 surrogate reproduces canonical W22^2,
and that its analytic field gradients agree with numerical finite differences.

This does NOT train a model, does NOT backpropagate through TTK/matching, and
is not yet a TensorFlow gradient or optimizer test. On ambiguities we fail closed.
"""
from __future__ import annotations

import argparse
from collections import Counter
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

SYNTHETIC = Path('ttk_runs_fixed/topology_finetuning/phase8_synthetic_fields_20261116')
DEFAULT_OUT = Path('ttk_runs_fixed/topology_finetuning/phase8_matching_vertex_gradient_gate_20261116')
SYNTHETIC_SCRIPT = Path('phase8_synthetic_field_gate_20261116.py')
PILOT = Path('phase8_ttk_w22_timing_pilot_20261116.py')
PHASE6 = Path('phase6_lxxvii_update/scripts/phase6_pilot_pd_gudhi_gate.py')
EXTRACT = Path('scripts/extract_ttk_pd_critical_pairs.py')
CANONICAL = Path.home() / 'phire_runtime_audit_20260809_221548/recompute_pd/canonical_pd_pilot.py'
NAMES = ('gt', 'identity', 'spurious', 'displaced')
CROP=160
# VTI is written as ASCII float with 8 significant decimal digits.
PAIR_COORD_ATOL=5.e-4


def need(c, msg):
    if not c: raise RuntimeError(msg)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for p in iter(lambda:f.read(1024*1024),b''):h.update(p)
    return h.hexdigest()


def load(path, name):
    need(path.is_file(), f'Missing expected source: {path}')
    spec=importlib.util.spec_from_file_location(name,path)
    need(spec is not None and spec.loader is not None,f'Cannot load {path}')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m


def positive(diag):
    a=np.asarray(diag,dtype=np.float64)
    need(a.ndim==2 and a.shape[1]==2 and np.all(np.isfinite(a)),'Invalid canonical diagram')
    return np.ascontiguousarray(a[a[:,1]>a[:,0]],dtype=np.float64)


def raw_pairs(vtu,field,extract):
    """Gather finite TTK pair endpoints using repaired C-order VTK IDs."""
    r=extract.parse_vtu(vtu)
    ids=np.asarray(r['vertex_ids'],dtype=np.int64)
    edges=np.asarray(r['connectivity'],dtype=np.int64)
    n=len(r['persistence'])
    need(len(edges)==2*n, f'Unexpected non-2-endpoint TTK connectivity: {vtu}')
    need(len(r['is_finite'])==len(r['pair_type'])==len(r['birth'])==n,
         'Raw TTK arrays have incompatible lengths')
    flat=field.ravel(order='C')
    records=[]
    for k in range(n):
        if int(r['is_finite'][k])!=1 or int(r['pair_type'][k])==-1 or float(r['persistence'][k])<=0:
            continue
        p0,p1=map(int,edges[2*k:2*k+2])
        need(0<=p0<len(ids) and 0<=p1<len(ids),'TTK pair point index out of bounds')
        v0,v1=int(ids[p0]),int(ids[p1])
        need(0<=v0<flat.size and 0<=v1<flat.size,'TTK critical vertex out of 160x160 crop')
        a,b=float(flat[v0]),float(flat[v1])
        if a<=b:
            birth,death,bvid,dvid=a,b,v0,v1
        else:
            birth,death,bvid,dvid=b,a,v1,v0
        if death<=birth:continue
        raw_b,raw_d=float(r['birth'][k]),float(r['birth'][k])+float(r['persistence'][k])
        # Coordinates in the canonical diagram are in the same scalar units;
        # check, but do not silently transpose/adjust vertex IDs to make them fit.
        need(max(abs(raw_b-birth),abs(raw_d-death))<=PAIR_COORD_ATOL,
             f'Pair {k} raw birth/death versus field-indexed vertices disagree: '
             f'raw=({raw_b:.9g},{raw_d:.9g}) indexed=({birth:.9g},{death:.9g}) '
             f'v=({bvid},{dvid}); investigate TTK/VTU conventions')
        records.append({'pair_cell':int(k),'pair_type':int(r['pair_type'][k]),
                        'birth':birth,'death':death,'birth_vid':bvid,'death_vid':dvid})
    return records


def align(diagrams,records):
    """Uniquely associate every positive canonical H0/H1 point with one raw pair.

    We intentionally infer dimension through numerical pair identity rather than
    assuming TTK PairType labels map directly to H0/H1. A future training gate
    should use verified pair identifiers and handle duplicate scalar values.
    """
    assignments={}; used=set()
    for dim in (0,1):
        arr=positive(diagrams[dim]); dim_records=[]
        for i,pt in enumerate(arr):
            candidates=[]
            for j,r in enumerate(records):
                if j in used:continue
                err=max(abs(r['birth']-pt[0]), abs(r['death']-pt[1]))
                if err <= PAIR_COORD_ATOL:
                    candidates.append((err,j))
            need(candidates,f'D{dim} canonical PD[{i}]={pt.tolist()} cannot be '
                            'mapped to any TTK critical-vertex pair')
            candidates.sort()
            # Do not silently pick a spatial location for tied diagrams.
            if len(candidates)>1:
                need(candidates[1][0]-candidates[0][0]>1e-7,
                     f'D{dim} PD[{i}] ambiguous critical-vertex pair; '
                     f'{len(candidates)} compatible TTK edges, best errors '
                     f'{candidates[0][0]:.3g}/{candidates[1][0]:.3g}')
            used.add(candidates[0][1]); dim_records.append(records[candidates[0][1]])
        assignments[dim]=dim_records
    return assignments, len(records)-len(used)


def matching_obj_and_grad(field,ref,mapped,matches):
    """Exact squared W22 *at this fixed matching and these critical vertices*.

    Unmatched predicted features contribute half squared persistence; unmatched
    GT features are constant with respect to the current predicted field.
    """
    from collections import Counter
    flat=np.asarray(field,dtype=np.float64).ravel(order='C')
    grad=np.zeros_like(flat)
    loss=0.
    c=Counter()
    for i,j in np.asarray(matches,dtype=np.int64).reshape((-1,2)):
        i,j=int(i),int(j)
        if i>=0 and j>=0:
            r=mapped[j];bb,dd=r['birth_vid'],r['death_vid']
            x,y=float(flat[bb]),float(flat[dd])
            a,b=map(float,ref[i])
            loss+=(x-a)**2+(y-b)**2
            grad[bb]+=2*(x-a);grad[dd]+=2*(y-b)
            c['matched']+=1
        elif i==-1 and j>=0:
            r=mapped[j];bb,dd=r['birth_vid'],r['death_vid']
            x,y=float(flat[bb]),float(flat[dd]);diff=y-x
            loss+=.5*diff*diff
            grad[bb]-=diff;grad[dd]+=diff
            c['sr_to_diagonal']+=1
        elif i>=0 and j==-1:
            a,b=map(float,ref[i]);loss+=.5*(b-a)**2
            c['gt_to_diagonal']+=1
        else:
            raise RuntimeError(f'Invalid matching edge {(i,j)}')
    return float(loss), grad.reshape(field.shape),dict(c)


def finite_difference(field,ref,mapped,matches,gradient):
    """Perturb only occupied predicted TTK vertex IDs; no recomputation of TTK.

    We check the local fixed-assignment surrogate, not differentiability through
    discrete TTK or changes in GUDHI optimal assignment.
    """
    ids=sorted({int(r[t]) for r in mapped for t in ('birth_vid','death_vid')})
    need(ids,'No predicted critical pair vertices to differentiate')
    flat=field.astype(np.float64,copy=True).reshape(-1)
    eps=1e-4; max_abs=0.;worst_vid=None
    for vid in ids:
        flat[vid]+=eps
        lp,_,_=matching_obj_and_grad(flat.reshape(field.shape),ref,mapped,matches)
        flat[vid]-=2*eps
        lm,_,_=matching_obj_and_grad(flat.reshape(field.shape),ref,mapped,matches)
        flat[vid]+=eps
        numerical=(lp-lm)/(2*eps)
        analytic=float(gradient.flat[vid])
        err=abs(numerical-analytic)
        if err>max_abs:max_abs=err;worst_vid=vid
        need(math.isclose(numerical,analytic,rel_tol=5e-5,abs_tol=1e-5),
             f'Finite difference failed at vid={vid} numerical={numerical} '
             f'analytic={analytic} absdiff={err}')
    return {'unique_predicted_critical_vertices':len(ids),
            'eps':eps,'max_abs_gradient_error':float(max_abs),'worst_vid':worst_vid,
            'nonzero_field_gradient_entries':int(np.count_nonzero(gradient))}


def sources(repo):
    return [repo/p for p in (SYNTHETIC_SCRIPT,PILOT,PHASE6,EXTRACT)] + [CANONICAL]


def preflight(repo,syn):
    need(syn.is_dir(),f'Synthetic-field gate outputs missing: {syn}')
    for p in sources(repo):need(p.is_file(),f'Missing {p}')
    manifest=json.loads((syn/'results.json').read_text())
    need(manifest['schema']=='phase8-synthetic-fields-v1','Unexpected synthetic manifest schema')
    for rel,expected in manifest['files_sha256'].items():
        p=syn/rel
        need(p.is_file() and sha(p)==expected,f'Synthetic input artifact changed: {p}')
    for s,expected in manifest['source_hashes'].items():
        p=Path(s) if Path(s).is_absolute() else repo/s
        need(p.is_file() and sha(p)==expected,f'Synthetic source changed: {p}')
    print(f'[PREFLIGHT] PASS: immutable synthetic field/VTU/PD outputs, source hashes; {syn}',flush=True)
    return manifest


def compute(repo,syn,verbose=False):
    import gudhi
    from gudhi.wasserstein import wasserstein_distance
    p6=load(repo/PHASE6,'gradient_gate_phase6')
    decoder=load(CANONICAL,'gradient_gate_canonical')
    extract=load(repo/EXTRACT,'gradient_gate_raw_ttk')
    fields={n:np.load(syn/'fields'/f'{n}.npy',allow_pickle=False) for n in NAMES}
    diagrams={n:p6.parse(decoder.read_pd,syn/'pd'/f'{n}_pd_port_0.vtu') for n in NAMES}
    maps={};unmapped={}
    for name in NAMES:
        need(fields[name].shape==(CROP,CROP) and np.isfinite(fields[name]).all(),
             f'Bad physical scalar field shape/values: {name}')
        raw=raw_pairs(syn/'pd'/f'{name}_pd_port_0.vtu',fields[name],extract)
        maps[name],unmapped[name]=align(diagrams[name],raw)
    result={};
    for name in NAMES[1:]:
        loss=0.;grad=np.zeros((CROP,CROP),dtype=np.float64)
        dim_out={};
        for dim in (0,1):
            gt=positive(diagrams['gt'][dim]);pred=positive(diagrams[name][dim])
            need(len(pred)==len(maps[name][dim]),f'{name} H{dim} mapping count mismatch')
            distance,matches=wasserstein_distance(gt,pred,order=2.,internal_p=2.,
                matching=True,keep_essential_parts=False)
            val,g,c=matching_obj_and_grad(fields[name],gt,maps[name][dim],matches)
            need(math.isclose(val,float(distance)**2,rel_tol=1e-5,abs_tol=1e-4),
                 f'{name} D{dim}: vertex-indexed local surrogate J={val} '
                 f'!= GUDHI W22^2={float(distance)**2}')
            fd=finite_difference(fields[name],gt,maps[name][dim],matches,g)
            loss+=val;grad+=g
            dim_out[f'D{dim}']={'n_gt_positive':len(gt),'n_pred_positive':len(pred),
                                 'W22':float(distance),'vertex_surrogate_squared':val,
                                 'counts':c,'finite_difference':fd,
                                 'pair_types':dict(Counter(str(r['pair_type']) for r in maps[name][dim]))}
        W22=math.sqrt(loss)
        need(math.isclose(W22,float(p6.distance(diagrams['gt'],diagrams[name])['W22']),
                          rel_tol=1e-5,abs_tol=1e-4),f'{name}: total canonical W22 mismatch')
        result[name]={'W22':float(W22),'vertex_surrogate_squared':float(loss),
                      'gradient_max_abs':float(np.abs(grad).max()),
                      'gradient_nonzero_count':int(np.count_nonzero(grad)),
                      'dimensions':dim_out,'unmapped_finite_raw_pairs':unmapped[name]}
        if verbose:
            print(f'[VERTEX-GRAD] {name}: W22={W22:.9g} J={loss:.9g} '
                  f'|grad|_max={np.abs(grad).max():.6g} '
                  f'nonzero={np.count_nonzero(grad)} PASS',flush=True)
    # The spurious extra feature has a gradient direction toward the diagonal.
    need(result['spurious']['gradient_nonzero_count']>0,
         'Spurious feature has zero local diagram gradient')
    need(result['identity']['gradient_max_abs']<1e-5,
         'Identical prediction has nonzero local diagram gradient')
    return result


def selftest():
    # Matched, predicted-diagonal, GT-diagonal. Shared endpoints accumulate grads.
    f=np.array([[2.,4.,5.,7.]],dtype=np.float64)
    ref=np.array([[1.,4.],[2.,4.]],dtype=np.float64)
    m=[{'birth_vid':0,'death_vid':1},{'birth_vid':1,'death_vid':2}]
    edges=np.array([[0,0],[-1,1],[1,-1]],dtype=np.int64)
    obj,grad,c=matching_obj_and_grad(f,ref,m,edges)
    need(math.isclose(obj,(2-1)**2+(4-4)**2+.5*(5-4)**2+.5*(4-2)**2),
         'Synthetic matching surrogate squared cost mismatch')
    need(c=={'matched':1,'sr_to_diagonal':1,'gt_to_diagonal':1},'Synthetic coverage counts')
    fd=finite_difference(f,ref,m,edges,grad)
    need(fd['max_abs_gradient_error']<1e-6,'Analytic/numeric gradient failed')
    # Diagonal-only cost leads to gradient zero for an unmatched GT feature.
    f2=np.array([[1.,2.]],dtype=float)
    obj2,g2,c2=matching_obj_and_grad(f2,np.array([[0.,2.]]),[],np.array([[0,-1]]))
    need(np.count_nonzero(g2)==0 and math.isclose(obj2,2.),'GT diagonal gradient should be zero')
    print(f'[SELFTEST] PASS: matched predicted pair, predicted->diagonal, '
          f'GT->diagonal constant, shared-vertex accumulation, finite differences '
          f'(max abs err {fd["max_abs_gradient_error"]:.3g}); no TTK/TF run',flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode',choices=('selftest','preflight','benchmark','verify'),required=True)
    ap.add_argument('--repo-root',type=Path,default=Path('.'))
    ap.add_argument('--synthetic-dir',type=Path,default=SYNTHETIC)
    ap.add_argument('--out-dir',type=Path,default=DEFAULT_OUT)
    a=ap.parse_args();repo=a.repo_root.resolve();syn=(repo/a.synthetic_dir).resolve();out=(repo/a.out_dir).resolve()
    if a.mode=='selftest':selftest();return
    preflight(repo,syn)
    if a.mode=='preflight':
        import gudhi
        print('[PREFLIGHT] PASS: GUDHI present; data are read-only; no training or artifact writes',flush=True)
        return
    if a.mode=='benchmark':
        need(not out.exists(),f'Refusing existing output: {out}')
        results=compute(repo,syn,verbose=True)
        payload={'schema':'phase8-matching-vertex-gradient-v1',
                 'status':'nontraining-local-fixed-assignment-gradient-only',
                 'sources_sha256':{str(p if p.is_absolute() else p.relative_to(repo)):sha(p) for p in sources(repo)},
                 'input_results_sha256':sha(syn/'results.json'),
                 'synthetic_dir':str(a.synthetic_dir),'results':results,
                 'caveats':['Pair-critical-vertex mapping may be ambiguous on real fields; this test fails closed.',
                            'This is a fixed-assignment local surrogate derivative, not a derivative of the TTK/GUDHI algorithms.',
                            'Unmatched GT diagonal costs are constant; a separate GT reconstruction term may be required.',
                            'No TF1 training, model weights, or held-out benchmark data were used.']}
        need(not out.exists(),f'Output appeared during benchmark: {out}')
        out.parent.mkdir(parents=True,exist_ok=True)
        stage=Path(tempfile.mkdtemp(prefix='.'+out.name+'.staging-',dir=out.parent))
        try:
            (stage/'result.json').write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
            need(not out.exists(),f'Output appeared during stage: {out}')
            stage.rename(out)
        finally:
            if stage.is_dir():shutil.rmtree(stage)
        print(f'[VERTEX-GRAD-GATE] PASS: immutable report={out/"result.json"}',flush=True)
    else:
        need(out.is_dir(),f'No output at {out}')
        old=json.loads((out/'result.json').read_text())
        need(old['schema']=='phase8-matching-vertex-gradient-v1','Invalid output schema')
        need(sha(syn/'results.json')==old['input_results_sha256'],'Upstream manifest hash changed')
        for path,hash_ in old['sources_sha256'].items():
            p=Path(path) if Path(path).is_absolute() else repo/path
            need(p.is_file() and sha(p)==hash_,f'Gate source changed: {p}')
        new=compute(repo,syn,verbose=False)
        for name in NAMES[1:]:
            for key in ('W22','vertex_surrogate_squared','gradient_max_abs','gradient_nonzero_count'):
                v,w=new[name][key],old['results'][name][key]
                need(math.isclose(v,w,rel_tol=1e-8,abs_tol=1e-7),f'Verification mismatch: {name}/{key}')
        print(f'[VERIFY] PASS: sources, upstream immutable manifest, canonical W22, '
              f'TTK critical-vertex surrogate and finite-difference gradients; {out}',flush=True)


if __name__=='__main__':
    try: main()
    except Exception as e:
        print(f'[PHASE8-VERTEX-GRAD] ERROR {type(e).__name__}: {e}',file=sys.stderr)
        sys.exit(1)
