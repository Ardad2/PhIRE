#!/usr/bin/env python3
"""LXXVII: sample-0 canonical continuity + GUDHI audit of three new arms.

Read-only TTK/Phase5E inputs. Writes only the new pilot_verified.json gate.
Run in the existing gudhi-audit micromamba environment on Spark host.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import math
import sys
from collections import Counter

METHODS = {
    'grad': 'candidateF_grad_pairzero_expanded2688',
    'cv': 'candidateF_grad_CV_only_expanded2688',
    'pers': 'candidateF_grad_PERS_only_expanded2688',
}
ROOT = Path('ttk_runs_fixed/topology_finetuning/phase6_lxxvii_canonical_pd_newarms_20261014')
FROZEN = Path('spatial_pd/phase5e_all168_canonical_pd/pd/phase5_GT_s0_speed_p160_x0_y0_pd_port_0.vtu')
AUDIT = Path.home() / 'phire_runtime_audit_20260809_221548/recompute_pd/canonical_pd_pilot.py'
PILOT = Path('ttk_runs_fixed/topology_finetuning/candidateF_grad_pairzero_expanded2688_s0_extraction_pilot_20261014/pd/candidateF_grad_pairzero_expanded2688_SR_s0_speed_p160_x0_y0_pd_port_0.vtu')
REF_GRAD = dict(dB=2.5805740356445312, W2inf=15.6249480697, W22=18.6702768797)

def need(x, msg):
    if not x: raise RuntimeError(msg)

def hash_file(path):
    need(path.is_file() and path.stat().st_size > 0, f'Missing {path}')
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def parse(read_pd,p):
    import numpy as np
    hash_file(p)
    diagrams,_=read_pd(p)
    result={}
    for dim in (0,1):
        a=np.asarray(diagrams[dim],dtype=float)
        need(a.ndim==2 and a.shape[1]==2 and np.isfinite(a).all(),f'Bad D{dim} PD: {p}')
        need(np.all(a[:,1]>=a[:,0]), f'Negative persistence D{dim}: {p}')
        result[dim]=a
    return result

def pos_multiset(a):
    return Counter(map(tuple,a[a[:,1]>a[:,0]]))

def distance(gt,sr):
    import numpy as np
    import gudhi
    from gudhi.wasserstein import wasserstein_distance as wasserstein
    dB=[]; wi=[]; we=[]
    for dim in (0,1):
        g,s=gt[dim],sr[dim]
        dB.append(float(gudhi.bottleneck_distance(g,s,e=0.)))
        wi.append(float(wasserstein(g,s,order=2.,internal_p=np.inf,matching=False,keep_essential_parts=False)))
        we.append(float(wasserstein(g,s,order=2.,internal_p=2.,matching=False,keep_essential_parts=False)))
    return dict(dB=max(dB), W2inf=math.hypot(*wi), W22=math.hypot(*we))

def main():
    import numpy as np
    spec=importlib.util.spec_from_file_location('canonical_pd',AUDIT)
    need(spec is not None and spec.loader is not None, f'Canonical parser unavailable: {AUDIT}')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    gt=parse(module.read_pd,FROZEN)
    print('[REFERENCE] Frozen Phase5E canonical GT s0 loaded',flush=True)
    records={};hashes={}
    for arm,method in METHODS.items():
        p=ROOT/arm/'pd'/f'{method}_SR_s0_speed_p160_x0_y0_pd_port_0.vtu'
        hashes[arm]=hash_file(p)
        sr=parse(module.read_pd,p)
        records[arm]=distance(gt,sr)
        print('[PILOT-METRIC]',arm,' '.join(f'{k}={v:.12g}' for k,v in records[arm].items()),flush=True)
        need(all(math.isfinite(v) and v>=0 for v in records[arm].values()), f'Invalid {arm} metric')
    old=parse(module.read_pd,PILOT)
    new=parse(module.read_pd,ROOT/'grad'/'pd'/'candidateF_grad_pairzero_expanded2688_SR_s0_speed_p160_x0_y0_pd_port_0.vtu')
    for dim in (0,1):
        need(pos_multiset(old[dim])==pos_multiset(new[dim]),
             f'Gradient sample-0 continuity: nonzero D{dim} mismatch')
    for key,ref in REF_GRAD.items():
        delta=abs(records['grad'][key]-ref)
        need(math.isclose(records['grad'][key],ref,rel_tol=1e-10,abs_tol=1e-8),
             f'Gradient historical metric regression failed {key}, absdiff={delta}')
    print('[GRAD-CONTINUITY] PASS: positive D0/D1 GT/SR conventions and all three corrected GUDHI metrics',flush=True)
    payload={'format':'phase6-lxxvii-pilot-verified-v1','gt_sha256':hash_file(FROZEN),
             'pd_sha256':hashes,'metrics_s0':records,'parser_sha256':hash_file(AUDIT)}
    target=ROOT/'pilot_verified.json'
    if target.exists():
        need(json.loads(target.read_text())==payload, f'Existing pilot gate has different contents: {target}')
    else:
        target.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
    print(f'[PILOT-GATE] PASS: immutable check recorded at {target}',flush=True)

if __name__=='__main__':
    try: main()
    except Exception as exc:
        print(f'[PILOT-GATE] ERROR {type(exc).__name__}: {exc}',file=sys.stderr)
        sys.exit(1)
