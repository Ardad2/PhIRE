#!/usr/bin/env python3
"""Phase 6 / LXXXII: matched 8-sample locality pilot via original Phase 5E sparse solver.

Does not alter the Phase 5E solver or historical/new persistence diagrams.
Requires Spark micromamba 'gudhi-audit' (VTK, GUDHI, NumPy and SciPy).
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from collections import Counter

SAMPLES=(0,24,48,69,96,120,144,167)
ARMS=('cnn','grad','cv','pers','f1')
RADII=(4.0,8.0,16.0,32.0,64.0,96.0,128.0,math.inf)
NEW={'grad':'candidateF_grad_pairzero_expanded2688','cv':'candidateF_grad_CV_only_expanded2688','pers':'candidateF_grad_PERS_only_expanded2688'}
SOLVER=Path('spatial_pd/phase5e_sparse_locality_solver.py')
SOLVER_SHA='47c73decca4ba151fd8b250cae2911b4c2307471fe9c881fc42bfc968e211c3a'
REF_PD=Path('spatial_pd/phase5e_all168_canonical_pd/pd')
REF_CSV=Path('spatial_pd/phase5e_all168_locality_analysis/all168_locality_aggregate_by_sample.csv')
LXXIX=Path('ttk_runs_fixed/topology_finetuning/phase6_lxxix_corrected_pd_gudhi_20261014/per_sample_corrected_pd.csv')
LXXIX_SHA='3d53189ec69cc2f89c5561159d0768c973351d7b9998deceeaef42ebb6b379ea'
NEWROOT=Path('ttk_runs_fixed/topology_finetuning/phase6_lxxvii_canonical_pd_newarms_20261014')
OUT=Path('ttk_runs_fixed/topology_finetuning/phase6_lxxxii_spatial_pilot8_20261014')
SELF=Path('phase6_lxxxii_update/scripts/phase6_lxxxii_locality_pilot.py')

def need(ok,msg):
    if not ok: raise RuntimeError(msg)

def sha(p):
    need(p.is_file() and p.stat().st_size>0,f'Missing/empty file: {p}')
    h=hashlib.sha256()
    with p.open('rb') as f:
        for x in iter(lambda:f.read(2**20),b''):h.update(x)
    return h.hexdigest()

def path_for(root,arm,s):
    stem=f's{s}_speed_p160_x0_y0_pd_port_0.vtu'
    if arm=='gt':return root/REF_PD/f'phase5_GT_{stem}'
    if arm in ('cnn','f1'):
        label='CNN_SR' if arm=='cnn' else 'F1_SR'
        return root/REF_PD/f'phase5_{label}_{stem}'
    return root/NEWROOT/arm/'pd'/f'{NEW[arm]}_SR_{stem}'

def get_tables(root):
    need(sha(root/LXXIX)==LXXIX_SHA,'Frozen LXXIX CSV checksum mismatch')
    with (root/LXXIX).open(newline='') as f: data=list(csv.DictReader(f))
    need(len(data)==840,'Expected 840 frozen LXXIX rows')
    known={}
    for row in data:
        k=(int(row['sample']),row['arm'])
        need(k not in known,f'Duplicate LXXIX row {k}')
        known[k]=row
    need(set(known)=={(s,a) for s in range(168) for a in ARMS},'Incomplete LXXIX sample/method keys')
    for arm in ARMS:
        for s in SAMPLES:
            r=known[s,arm]
            need(r['gt_sha256']==sha(path_for(root,'gt',s)),f'LXXIX GT artifact drift {s}')
            need(r['sr_sha256']==sha(path_for(root,arm,s)),f'LXXIX SR artifact drift {arm}:{s}')
    need((root/REF_CSV).is_file(),'Missing historical Phase5E aggregate locality CSV')
    with (root/REF_CSV).open(newline='') as f: refs=list(csv.DictReader(f))
    need(len(refs)==168*3*8, f'Historical Phase5E row count mismatch: {len(refs)}')
    hist={}
    for r in refs:
        k=(int(r['sample']),r['method'],r['radius'])
        need(k not in hist,f'Duplicate historical locality key {k}')
        hist[k]=r
    need(set(hist)=={(s,a,fmt_radius(rad)) for s in range(168) for a in ('CNN','UV','F1') for rad in RADII},'Phase5E locality reference sample/method/radius mismatch')
    return known,hist

def fmt_radius(r):return 'inf' if math.isinf(r) else str(r)

def import_solver(root):
    spec=importlib.util.spec_from_file_location('frozen_phase5e_sparse_solver',root/SOLVER)
    need(spec is not None and spec.loader is not None,'Could not load Phase5E solver')
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    need(tuple(mod.DIMS)==(0,1),'Phase5E dimensions changed')
    need(tuple(mod.RADII)==RADII,'Phase5E radius grid changed')
    need(tuple(mod.PILOT_SAMPLES)==SAMPLES,'Phase5E pilot sample grid changed')
    need(callable(mod.read_features) and callable(mod.SparsePrepared),'Phase5E solver interface changed')
    return mod

def check_near(a,b,what,atol=1e-8,rtol=1e-10):
    need(math.isfinite(a) and math.isfinite(b),f'Nonfinite comparison: {what}')
    need(math.isclose(a,b,abs_tol=atol,rel_tol=rtol),f'{what}: {a} vs {b} diff={abs(a-b)}')

def preflight(root):
    need(sha(root/SOLVER)==SOLVER_SHA,'Phase5E solver differs from discovered/verified source SHA; do not substitute')
    need((root/SELF).is_file(),'LXXXII wrapper not installed at expected path')
    known,hist=get_tables(root)
    src={}
    for s in SAMPLES:
        for arm in ('gt',*ARMS):
            p=path_for(root,arm,s)
            src[f'{arm}:{s}']={'sha256':sha(p),'path':str(p.relative_to(root))}
    print(f'[SOLVER] PASS pinned Phase5E sparse code SHA256={SOLVER_SHA}',flush=True)
    print(f'[REFERENCE] PASS frozen LXXIX 840 rows, historical Phase5E 4032 rows, {len(src)} pilot PDs',flush=True)
    print('[F1-LINEAGE] NOT CLOSED: this pilot checks numeric/locality continuity, not F1 training-run provenance',flush=True)
    return known,hist,src

def pilot(root):
    need(not (root/OUT).exists(),f'Pilot output already exists; refusing overwrite: {root/OUT}')
    known,hist,source=preflight(root)
    solver=import_solver(root)
    print(f'[ENV] VTK={solver.vtk.vtkVersion.GetVTKVersion()} GUDHI={solver.gudhi.__version__}',flush=True)
    dim_rows=[]; aggs=[];count_old=0;count_inf=0
    for s in SAMPLES:
        gt_features,gt_zero=solver.read_features(path_for(root,'gt',s))
        need(all(len(gt_features[d])>0 for d in (0,1)),f'GT missing D0/D1 positive features s={s}')
        for arm in ARMS:
            sr_features,sr_zero=solver.read_features(path_for(root,arm,s))
            need(all(len(sr_features[d])>0 for d in (0,1)),f'{arm} missing D0/D1 positive features s={s}')
            by_radius={key:{} for key in map(fmt_radius,RADII)}
            for d in (0,1):
                prep=solver.SparsePrepared(gt_features[d],sr_features[d])
                independent=solver.gudhi_w22(gt_features[d],sr_features[d])
                inf=prep.solve(math.inf)
                check_near(inf['w22'],independent,f'GUDHI D{d} s{s} {arm}',atol=1e-8)
                for rad in RADII:
                    key=fmt_radius(rad)
                    v=prep.solve(rad)
                    need(v['w22']+1e-8>=inf['w22'],f'Locality monotonicity lower than inf {s} {arm} D{d} r={key}')
                    by_radius[key][d]=v
                    dim_rows.append({'sample':s,'arm':arm,'dimension':d,'radius':key,'unconstrained_w22':independent,**v,'locality_gap':v['w22']-independent})
            for rad in RADII:
                key=fmt_radius(rad)
                a,b=by_radius[key][0],by_radius[key][1]
                total_sq=a['total_sq']+b['total_sq']
                w22=math.sqrt(total_sq)
                ref_inf=float(known[s,arm]['W22'])
                if math.isinf(rad):
                    check_near(w22,ref_inf,f'LXXIX r=inf {s} {arm}',atol=1e-7)
                    count_inf+=1
                gt_total=a['gt_total_persistence']+b['gt_total_persistence']
                sr_total=a['sr_total_persistence']+b['sr_total_persistence']
                gt_cov=(a['gt_matched_persistence']+b['gt_matched_persistence'])/gt_total if gt_total else None
                sr_cov=(a['sr_matched_persistence']+b['sr_matched_persistence'])/sr_total if sr_total else None
                need(gt_cov is not None and sr_cov is not None,'Zero denominator in pilot coverage')
                row={'sample':s,'arm':arm,'radius':key,'unconstrained_w22_all':ref_inf,
                     'constrained_w22_all':w22,'locality_gap_all':w22-ref_inf,
                     'gt_persistence_coverage_all':gt_cov,'sr_persistence_coverage_all':sr_cov,
                     'total_sq':total_sq,'rr_sq':a['rr_sq']+b['rr_sq'],
                     'gt_diag_sq':a['gt_diag_sq']+b['gt_diag_sq'],
                     'sr_diag_sq':a['sr_diag_sq']+b['sr_diag_sq'],
                     'real_real_all':a['real_real']+b['real_real'],
                     'gt_to_diagonal_all':a['gt_to_diagonal']+b['gt_to_diagonal'],
                     'diagonal_to_sr_all':a['diagonal_to_sr']+b['diagonal_to_sr']}
                if arm in ('cnn','f1'):
                    old=hist[(s,'CNN' if arm=='cnn' else 'F1',key)]
                    for f in ('constrained_w22_all','gt_persistence_coverage_all','sr_persistence_coverage_all','locality_gap_all'):
                        check_near(float(row[f]),float(old[f]),f'Phase5E regression s{s} {arm} r={key} {f}',atol=1e-7)
                    for f in ('real_real_all','gt_to_diagonal_all','diagonal_to_sr_all'):
                        need(row[f]==int(old[f]),f'Phase5E count mismatch s{s} {arm} r={key} {f}')
                    count_old+=1
                aggs.append(row)
        print(f'[PILOT] PASS sample={s}: 5 models × 2 dimensions × 8 radii, LXXIX inf regression and frozen CNN/F1 parity',flush=True)
    need(len(aggs)==8*5*8 and len(dim_rows)==8*5*2*8,'Unexpected pilot output sizes')
    need(count_old==8*2*8 and count_inf==8*5,'Incomplete reference continuity comparisons')
    # All validation completed before creating any output directory.
    dst=root/OUT
    dst.mkdir(parents=True,exist_ok=False)
    write_csv(dst/'pilot_locality_aggregate.csv',aggs)
    write_csv(dst/'pilot_locality_dimension.csv',dim_rows)
    report={'format':'phase6-lxxxii-pilot8-locality-v1','samples':list(SAMPLES),'arms':list(ARMS),
            'radius_keys':[fmt_radius(r) for r in RADII],
            'solver_sha256':SOLVER_SHA,'script_sha256':sha(root/SELF),
            'lxxix_csv_sha256':LXXIX_SHA,'phase5e_locality_csv_sha256':sha(root/REF_CSV),
            'pd_inputs':source, 'n_aggregate_rows':len(aggs),'n_dimension_rows':len(dim_rows),
            'infinity_w22_matches':count_inf,'phase5e_cnn_f1_regression_rows':count_old,
            'f1_training_lineage':'not closed',
            'package_note':'Uses unchanged on-Spark frozen phase5e_sparse_locality_solver.py functions: read_features, SparsePrepared, gudhi_w22'}
    (dst/'pilot_manifest.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    md=['# Phase 6 locality pilot (8 predeclared samples)','',
        'Validated the original Phase 5E sparse-savings locality solver, with preserved feature coordinates and radius grid.',
        f'CNN/F1 old spatial-regression rows: {count_old}; matched LXXIX unconstrained W22 cases: {count_inf}.',
        'These are **pilot-only descriptive results**, not the all-168 spatial result.','',
        '| Model | r=4 mean constrained W22 | r=16 mean constrained W22 | r=inf mean W22 | r=16 mean GT coverage |',
        '|---|---:|---:|---:|---:|']
    for arm in ARMS:
        rs={r:[x for x in aggs if x['arm']==arm and x['radius']==r] for r in ('4.0','16.0','inf')}
        avg=lambda vs,k:sum(x[k] for x in vs)/len(vs)
        md.append(f"| {arm} | {avg(rs['4.0'],'constrained_w22_all'):.5f} | {avg(rs['16.0'],'constrained_w22_all'):.5f} | {avg(rs['inf'],'constrained_w22_all'):.5f} | {avg(rs['16.0'],'gt_persistence_coverage_all'):.5f} |")
    md.extend(['','**Caveats:** This pilot is not a statistical independence test; no kilometer calibration, no merge-tree comparison, and no F1 training-lineage closure.',''])
    (dst/'pilot_summary.md').write_text('\n'.join(md))
    tohash=('pilot_locality_aggregate.csv','pilot_locality_dimension.csv','pilot_manifest.json','pilot_summary.md')
    (dst/'SHA256SUMS').write_text(''.join(f'{sha(dst/name)}  {name}\n' for name in tohash))
    print(f'[PHASE5E-REGRESSION] PASS: {count_old} CNN/F1 sample×radius rows with cost, coverage, gap and matching counts',flush=True)
    print(f'[INFINITY-GATE] PASS: {count_inf} sample×model values reproduce LXXIX W22 to tolerance',flush=True)
    print(f'[PILOT-GATE] PASS: {len(dim_rows)} dim rows, {len(aggs)} aggregate rows; output={dst}',flush=True)

def write_csv(path,rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0].keys()))
        writer.writeheader();writer.writerows(rows)

def verify(root):
    preflight(root)
    d=root/OUT
    need(d.is_dir(),'Pilot output absent')
    for line in (d/'SHA256SUMS').read_text().splitlines():
        expected,name=line.split('  ',1)
        need(sha(d/name)==expected,f'Output checksum mismatch {name}')
    manifest=json.loads((d/'pilot_manifest.json').read_text())
    need(manifest['solver_sha256']==SOLVER_SHA and manifest['lxxix_csv_sha256']==LXXIX_SHA,'Wrong pilot provenance')
    need(manifest['n_aggregate_rows']==320 and manifest['n_dimension_rows']==640,'Wrong pilot row count')
    for key,item in manifest['pd_inputs'].items():
        need(sha(root/item['path'])==item['sha256'],f'Changed pilot PD source {key}')
    print('[VERIFY-GATE] PASS: output manifests, input hashes and expected pilot row counts',flush=True)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode',required=True,choices=['preflight','pilot','verify'])
    ap.add_argument('--repo-root',type=Path,default=Path.cwd())
    args=ap.parse_args();root=args.repo_root.resolve()
    if args.mode=='pilot':pilot(root)
    elif args.mode=='verify':verify(root)
    else:
        preflight(root)
        print('[PREFLIGHT] PASS: source hashes/inventories only; no locality solver run or files written',flush=True)
if __name__=='__main__':
    try:main()
    except Exception as e:
        print(f'[LXXXII] ERROR {type(e).__name__}: {e}',file=sys.stderr)
        sys.exit(1)
