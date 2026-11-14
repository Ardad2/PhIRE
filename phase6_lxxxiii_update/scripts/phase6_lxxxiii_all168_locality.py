#!/usr/bin/env python3
"""Phase 6 LXXXIII: resumable, reference-checked full-168 spatial W2,2 study.

Reuses the *unchanged* original Phase 5E sparse solver through the audited LXXXII
pilot wrapper. Does not modify model outputs, PD files, Phase 5E or LXXIX data.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import sys

BASE = Path('phase6_lxxxii_update/scripts/phase6_lxxxii_locality_pilot.py')
BASE_SHA = '09225302d04e70a8259983ab5b127d0f1180d4a0d0f1fa448b5dd7b00094f18a'
SELF = Path('phase6_lxxxiii_update/scripts/phase6_lxxxiii_all168_locality.py')
OUT = Path('ttk_runs_fixed/topology_finetuning/phase6_lxxxiii_spatial_all168_20261014')
PILOT_AGG = Path('ttk_runs_fixed/topology_finetuning/phase6_lxxxii_spatial_pilot8_20261014/pilot_locality_aggregate.csv')
PILOT_DIM = Path('ttk_runs_fixed/topology_finetuning/phase6_lxxxii_spatial_pilot8_20261014/pilot_locality_dimension.csv')
ARMS = ('cnn','grad','cv','pers','f1')
SAMPLES = tuple(range(168))


def need(cond,msg):
    if not cond: raise RuntimeError(msg)


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for buf in iter(lambda:f.read(1<<20),b''): h.update(buf)
    return h.hexdigest()


def bmod(root):
    p=root/BASE
    need(p.is_file(),f'LXXXII pilot script not installed: {p}')
    spec=importlib.util.spec_from_file_location('frozen_lxxxii_pilot',p)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def near(a,b,what,atol=1e-7,rtol=1e-10):
    aa,bb=float(a),float(b)
    need(math.isfinite(aa) and math.isfinite(bb),f'{what}: nonfinite')
    need(math.isclose(aa,bb,abs_tol=atol,rel_tol=rtol),f'{what}: {aa} vs {bb} delta={aa-bb}')


def rows_csv(path):
    with path.open(newline='') as f: return list(csv.DictReader(f))


def row_key(r):
    return (int(r['sample']),r['arm'],r['radius'])


def dim_key(r):
    return (int(r['sample']),r['arm'],int(r['dimension']),r['radius'])


def index_unique(rows,key,label):
    ans={}
    for r in rows:
        k=key(r)
        need(k not in ans,f'Duplicate {label} {k}')
        ans[k]=r
    return ans


def compare_row(actual,reference,what):
    # Require every field emitted by the original Phase 5E solver/old pilot.
    for name,expected in reference.items():
        need(name in actual,f'{what}: missing {name}')
        observed=actual[name]
        if expected is None or expected=='':
            need(observed is None or observed=='',f'{what}: null {name} differs')
        elif name in ('sample','arm','radius','dimension'):
            need(str(observed)==str(expected),f'{what}: {name} mismatch {observed}/{expected}')
        elif name in ('real_real','gt_to_diagonal','diagonal_to_sr',
                       'real_real_all','gt_to_diagonal_all','diagonal_to_sr_all','candidate_real_edges'):
            need(int(observed)==int(expected),f'{what}: count {name} mismatch')
        else:
            near(observed,expected,f'{what} {name}',atol=1e-7)


def preflight(root,verbose=True):
    b=bmod(root)
    need((root/SELF).is_file(),'LXXXIII source not at packaged path')
    need(sha(root/BASE)==BASE_SHA,'LXXXII trusted wrapper changed; refusing to port source')
    b.verify(root)  # LXXXII pilot hashes + sample-0 origin continuity
    known,hist=b.get_tables(root)
    prior=json.loads((root/b.OUT/'pilot_manifest.json').read_text())
    need(prior['phase5e_locality_csv_sha256']==sha(root/b.REF_CSV),
         'Historical Phase5E locality reference CSV changed since validated LXXXII pilot')
    need(prior['script_sha256']==BASE_SHA,'Saved LXXXII pilot is not from trusted source code')
    pilot_a=rows_csv(root/PILOT_AGG)
    pilot_d=rows_csv(root/PILOT_DIM)
    pa=index_unique(pilot_a,row_key,'pilot-aggregate')
    pd=index_unique(pilot_d,dim_key,'pilot-dimension')
    rkeys={b.fmt_radius(r) for r in b.RADII}
    need(set(pa)=={(s,a,r) for s in b.SAMPLES for a in ARMS for r in rkeys},'Pilot aggregate inventory mismatch')
    need(set(pd)=={(s,a,d,r) for s in b.SAMPLES for a in ARMS for d in (0,1) for r in rkeys},'Pilot dimension inventory mismatch')
    # Ensure all 1008 PD files match the frozen LXXIX per-sample input hashes.
    sources={}
    for s in SAMPLES:
        gt=b.path_for(root,'gt',s)
        gt_digest=sha(gt)
        for a in ARMS:
            row=known[s,a]
            need(gt_digest==row['gt_sha256'],f'GT hash mismatch LXXIX s{s} {a}')
            sr=b.path_for(root,a,s)
            sr_digest=sha(sr)
            need(sr_digest==row['sr_sha256'],f'SR hash mismatch LXXIX s{s} {a}')
            sources[f'{a}:{s}']={'gt_sha256':gt_digest,'sr_sha256':sr_digest}
    expected={'format':'phase6-lxxxiii-spatial-all168-v1','n_samples':168,'arms':list(ARMS),
        'radii':[b.fmt_radius(r) for r in b.RADII],
        'original_solver_sha256':b.SOLVER_SHA, 'pilot_driver_sha256':sha(root/BASE),
        'full_driver_sha256':sha(root/SELF),
        'lxxix_csv_sha256':b.LXXIX_SHA,'phase5e_csv_sha256':sha(root/b.REF_CSV),
        'pilot_aggregate_sha256':sha(root/PILOT_AGG),'pilot_dimension_sha256':sha(root/PILOT_DIM),
        'sources':sources,'f1_lineage':'not independently closed'}
    if verbose:
        print('[REFERENCE] PASS: LXXXII pilot checksum/invariants and all 1008 input PD file hashes match LXXIX',flush=True)
        print('[SOLVER] PASS original Phase5E sparse solver unchanged SHA256='+b.SOLVER_SHA,flush=True)
        print('[EXPECTED] 168 samples × 5 arms × 8 radii = 6720 aggregate; 13440 dimension rows',flush=True)
        print('[F1-LINEAGE] NOT CLOSED: source/checkpoint training-provenance check separate',flush=True)
    return b,known,hist,pa,pd,expected


def run_one(root,b,known,hist,pa,pd,s,solver):
    gt,_=solver.read_features(b.path_for(root,'gt',s))
    need(all(len(gt[d])>0 for d in (0,1)),f'GT D0/D1 empty s{s}')
    dim_rows=[];aggs=[]
    old_count=0;inf_count=0;pilot_count=0
    for a in ARMS:
        sr,_=solver.read_features(b.path_for(root,a,s))
        need(all(len(sr[d])>0 for d in (0,1)),f'{a} D0/D1 empty s{s}')
        accum={b.fmt_radius(r):{} for r in b.RADII}
        for d in (0,1):
            prep=solver.SparsePrepared(gt[d],sr[d])
            independent=solver.gudhi_w22(gt[d],sr[d])
            ref_inf=prep.solve(math.inf)
            near(ref_inf['w22'],independent,f'GUDHI D{d} s{s} {a}',atol=1e-8)
            previous=-math.inf
            for r in b.RADII:
                key=b.fmt_radius(r)
                result=prep.solve(r)
                need(result['w22']+1e-8>=ref_inf['w22'],f'Constrained below infinity D{d} s{s} {a} r={key}')
                # Increasing radius cannot make optimal cost larger.
                need(result['w22']<=previous+1e-8 or previous==-math.inf,
                     f'Non-monotone local matching cost D{d} s{s} {a} r={key}')
                previous=result['w22']
                accum[key][d]=result
                row={'sample':s,'arm':a,'dimension':d,'radius':key,'unconstrained_w22':independent,
                     **result,'locality_gap':result['w22']-independent}
                if s in b.SAMPLES:
                    compare_row(row,pd[s,a,d,key],f'Pilot dimension {s} {a} D{d} r={key}')
                dim_rows.append(row)
        for r in b.RADII:
            key=b.fmt_radius(r)
            x,y=accum[key][0],accum[key][1]
            total=x['total_sq']+y['total_sq']
            w=math.sqrt(max(total,0))
            reference=float(known[s,a]['W22'])
            if math.isinf(r):
                near(w,reference,f'All-168 LXXIX infinity s{s} {a}',atol=1e-7)
                inf_count+=1
            gp=x['gt_total_persistence']+y['gt_total_persistence']
            sp=x['sr_total_persistence']+y['sr_total_persistence']
            need(gp>0 and sp>0,f'Invalid persistence mass s{s} {a}')
            row={'sample':s,'arm':a,'radius':key,'unconstrained_w22_all':reference,
                'constrained_w22_all':w,'locality_gap_all':w-reference,
                'gt_persistence_coverage_all':(x['gt_matched_persistence']+y['gt_matched_persistence'])/gp,
                'sr_persistence_coverage_all':(x['sr_matched_persistence']+y['sr_matched_persistence'])/sp,
                'total_sq':total,'rr_sq':x['rr_sq']+y['rr_sq'],
                'gt_diag_sq':x['gt_diag_sq']+y['gt_diag_sq'],
                'sr_diag_sq':x['sr_diag_sq']+y['sr_diag_sq'],
                'real_real_all':x['real_real']+y['real_real'],
                'gt_to_diagonal_all':x['gt_to_diagonal']+y['gt_to_diagonal'],
                'diagonal_to_sr_all':x['diagonal_to_sr']+y['diagonal_to_sr']}
            near(row['total_sq'],row['rr_sq']+row['gt_diag_sq']+row['sr_diag_sq'],f'Cost decomposition {s} {a} r={key}',atol=1e-7)
            need(-1e-9<=row['gt_persistence_coverage_all']<=1+1e-9,'GT coverage out of bounds')
            need(-1e-9<=row['sr_persistence_coverage_all']<=1+1e-9,'SR coverage out of bounds')
            if a in ('cnn','f1'):
                old=hist[s,'CNN' if a=='cnn' else 'F1',key]
                for name in ('constrained_w22_all','gt_persistence_coverage_all','sr_persistence_coverage_all','locality_gap_all'):
                    near(row[name],old[name],f'Phase5E regression s{s} {a} r={key} {name}')
                for name in ('real_real_all','gt_to_diagonal_all','diagonal_to_sr_all'):
                    need(row[name]==int(old[name]),f'Phase5E matching count mismatch s{s} {a} r={key} {name}')
                old_count+=1
            if s in b.SAMPLES:
                compare_row(row,pa[s,a,key],f'Pilot aggregate s{s} {a} r={key}')
                pilot_count+=1
            aggs.append(row)
    need(len(dim_rows)==80 and len(aggs)==40 and inf_count==5 and old_count==16,'Incorrect per-sample row counts')
    return {'sample':s,'dimension_rows':dim_rows,'aggregate_rows':aggs,
            'infinity_regressions':inf_count,'historical_regressions':old_count,
            'pilot_regressions':pilot_count}


def atomic_json(path,obj):
    tmp=path.with_suffix('.tmp')
    with tmp.open('x') as f:
        json.dump(obj,f,sort_keys=True,separators=(',',':'))
        f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def source_match(root,out,expected):
    manifest=out/'run_manifest.json'
    need(manifest.is_file(),f'Missing manifest {manifest}')
    actual=json.loads(manifest.read_text())
    need(actual==expected,'Input/script checksum manifest changed since full run; refusing resume')


def validate_partial(obj,s,b,known,hist,pa,pd):
    need(obj['sample']==s,'Incorrect partial sample ID')
    need(len(obj['aggregate_rows'])==40 and len(obj['dimension_rows'])==80,'Partial rows incomplete')
    agg=index_unique(obj['aggregate_rows'],row_key,'partial-aggregate')
    dims=index_unique(obj['dimension_rows'],dim_key,'partial-dimension')
    need(set(agg)=={(s,a,b.fmt_radius(r)) for a in ARMS for r in b.RADII},'Incorrect aggregate sample keys')
    need(set(dims)=={(s,a,d,b.fmt_radius(r)) for a in ARMS for d in (0,1) for r in b.RADII},'Incorrect dimension sample keys')
    for a in ARMS:
        last=float('inf')
        for r in b.RADII:
            key=b.fmt_radius(r)
            v=agg[s,a,key]
            w=float(v['constrained_w22_all'])
            need(math.isfinite(w) and w>=0,'Nonfinite or negative saved cost')
            need(w<=last+1e-8,'Nonmonotonic saved aggregate W22')
            last=w
            need(0<=float(v['gt_persistence_coverage_all'])<=1+1e-9,'GT coverage out of range')
            need(0<=float(v['sr_persistence_coverage_all'])<=1+1e-9,'SR coverage out of range')
            near(float(v['total_sq']),w*w,f'Saved cost-square identity {s} {a} r={key}')
            near(float(v['total_sq']),sum(float(v[k]) for k in ('rr_sq','gt_diag_sq','sr_diag_sq')),f'Saved cost decomposition {s} {a} r={key}')
            if a in ('cnn','f1'):
                old=hist[s,'CNN' if a=='cnn' else 'F1',key]
                for name in ('constrained_w22_all','gt_persistence_coverage_all','sr_persistence_coverage_all','locality_gap_all'):
                    near(v[name],old[name],f'Saved Phase5E {s} {a} r={key} {name}')
                for name in ('real_real_all','gt_to_diagonal_all','diagonal_to_sr_all'):
                    need(int(v[name])==int(old[name]),f'Saved Phase5E {name} mismatch')
            if s in b.SAMPLES:
                compare_row(v,pa[s,a,key],f'Saved pilot aggregate {s} {a} r={key}')
            if math.isinf(r): near(w,known[s,a]['W22'],f'Saved infinity {s} {a}')
            for d in (0,1):
                dim=dims[s,a,d,key]
                near(dim['w22'],math.sqrt(float(dim['total_sq'])),f'Saved dimensional cost {s} {a} D{d} r={key}')
                if s in b.SAMPLES: compare_row(dim,pd[s,a,d,key],f'Saved pilot dim {s} {a} D{d} r={key}')
            near(float(dims[s,a,0,key]['total_sq'])+float(dims[s,a,1,key]['total_sq']),v['total_sq'],f'Saved D0+D1 sum {s} {a} r={key}')


def save_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def finish(out,partials,b,known,hist,pa,pd):
    aggregate=[];dimension=[]
    for s in SAMPLES:
        obj=json.loads((partials/f's{s:03d}.json').read_text())
        validate_partial(obj,s,b,known,hist,pa,pd)
        aggregate.extend(obj['aggregate_rows']);dimension.extend(obj['dimension_rows'])
    need(len(aggregate)==len(SAMPLES)*len(ARMS)*len(b.RADII) and len(dimension)==len(SAMPLES)*len(ARMS)*len(b.RADII)*2,'Full output cardinality mismatch')
    agg=index_unique(aggregate,row_key,'full-aggregate')
    summaries=[];pairs=[];factorials=[]
    for r in b.RADII:
        key=b.fmt_radius(r)
        for a in ARMS:
            sub=[agg[s,a,key] for s in SAMPLES]
            cols=('constrained_w22_all','gt_persistence_coverage_all','sr_persistence_coverage_all','locality_gap_all')
            stats={}
            for name in cols:
                vals=[float(x[name]) for x in sub]
                stats['mean_'+name]=statistics.mean(vals)
                stats['median_'+name]=statistics.median(vals)
            summaries.append({'radius':key,'arm':a,'n':168,**stats})
        for a,c in [('grad','cnn'),('cv','grad'),('pers','grad'),('f1','grad'),('f1','pers'),('pers','cv'),('f1','cv'),('f1','cnn')]:
            differences=[float(agg[s,a,key]['constrained_w22_all'])-float(agg[s,c,key]['constrained_w22_all']) for s in SAMPLES]
            coverage=[float(agg[s,a,key]['gt_persistence_coverage_all'])-float(agg[s,c,key]['gt_persistence_coverage_all']) for s in SAMPLES]
            pairs.append({'radius':key,'contrast':a+'-minus-'+c,'mean_cost_difference':statistics.mean(differences),
                         'median_cost_difference':statistics.median(differences),
                         'cost_A_better_count':sum(v<0 for v in differences),
                         'cost_A_worse_count':sum(v>0 for v in differences),
                         'cost_tie_count':sum(v==0 for v in differences),
                         'mean_GTcoverage_difference':statistics.mean(coverage),
                         'gtcoverage_A_higher_count':sum(v>0 for v in coverage)})
        for metric in ('constrained_w22_all','gt_persistence_coverage_all'):
            contrast={}
            for s in SAMPLES:
                g,c,p,f=[float(agg[s,a,key][metric]) for a in ('grad','cv','pers','f1')]
                for name,value in [('cv',((c-g)+(f-p))/2),('pers',((p-g)+(f-c))/2),('interaction',f-c-p+g)]:
                    contrast.setdefault(name,[]).append(value)
            for name,vals in contrast.items():
                factorials.append({'radius':key,'metric':metric,'effect':name,'mean':statistics.mean(vals),
                                  'median':statistics.median(vals),'negative_fields':sum(v<0 for v in vals),
                                  'positive_fields':sum(v>0 for v in vals)})
    save_csv(out/'all168_locality_aggregate.csv',aggregate)
    save_csv(out/'all168_locality_dimension.csv',dimension)
    save_csv(out/'all168_macro_summary.csv',summaries)
    save_csv(out/'all168_paired_by_radius.csv',pairs)
    save_csv(out/'all168_factorial_by_radius.csv',factorials)
    ms={(x['arm'],x['radius']):x for x in summaries}
    lines=['# Phase 6 full 168-sample spatial-locality results','',
        'Original SHA-pinned Phase 5E sparse matcher; 168 same-sample comparisons, five models, D0/D1, all eight radii.',
        'All infinity-radius values regress to frozen LXXIX corrected GUDHI W2,2.',
        'CNN/F1 rows reproduce the frozen Phase 5E spatial analysis.',
        'Smaller constrained W2,2 is better; GT coverage is persistence-mass coverage, not position error.','',
        '| Model | r=4 constrained W22 | r=16 constrained W22 | r=32 constrained W22 | r=inf W22 | r=16 GT coverage |',
        '|---|---:|---:|---:|---:|---:|']
    for a in ARMS:
        cost=lambda rad:ms[a,rad]['mean_constrained_w22_all']
        lines.append(f"| {a} | {cost('4.0'):.6f} | {cost('16.0'):.6f} | {cost('32.0'):.6f} | {cost('inf'):.6f} | {ms[a,'16.0']['mean_gt_persistence_coverage_all']:.6f} |")
    lines += ['','## F1 versus PERS-only: paired cost at all radii','',
        '| Radius (grid cells) | Mean F1-PERS cost | F1 lower / PERS lower / ties |',
        '|---|---:|---:|']
    for x in pairs:
        if x['contrast']=='f1-minus-pers':
            lines.append(f"| {x['radius']} | {x['mean_cost_difference']:+.6f} | {x['cost_A_better_count']} / {x['cost_A_worse_count']} / {x['cost_tie_count']} |")
    lines+=['','**Limitations:** descriptive consecutive-hour benchmark, no independent seed uncertainty; F1 optimizer/checkpoint lineage not independently closed; no geographical km calibration and no new Phase 6 merge-tree distances.','']
    (out/'summary.md').write_text('\n'.join(lines))
    names=('run_manifest.json','all168_locality_aggregate.csv','all168_locality_dimension.csv',
           'all168_macro_summary.csv','all168_paired_by_radius.csv','all168_factorial_by_radius.csv','summary.md')
    (out/'SHA256SUMS').write_text(''.join(f'{sha(out/n)}  {n}\n' for n in names))


def full(root):
    b,known,hist,pa,pd,expected=preflight(root)
    out=root/OUT
    if out.exists():
        source_match(root,out,expected)
        need(not (out/'SHA256SUMS').exists(),f'Final results already exist: {out}. Use --mode verify; do not overwrite.')
    else:
        out.mkdir(parents=True,exist_ok=False)
        atomic_json(out/'run_manifest.json',expected)
    partials=out/'checkpoints';partials.mkdir(exist_ok=True)
    solver=b.import_solver(root)
    print('[ENV] VTK='+solver.vtk.vtkVersion.GetVTKVersion()+' GUDHI='+solver.gudhi.__version__,flush=True)
    processed=0
    for s in SAMPLES:
        dest=partials/f's{s:03d}.json'
        if dest.exists():
            obj=json.loads(dest.read_text())
            validate_partial(obj,s,b,known,hist,pa,pd)
            print(f'[RESUME] PASS sample={s} checkpoint reused',flush=True)
        else:
            obj=run_one(root,b,known,hist,pa,pd,s,solver)
            validate_partial(obj,s,b,known,hist,pa,pd)
            atomic_json(dest,obj)
            print(f'[FULL] PASS sample={s}; 40 aggregate rows, 80 dimensional rows',flush=True)
        processed+=1
        if processed%16==0 or processed==len(SAMPLES):
            print(f'[PROGRESS] PASS {processed}/{len(SAMPLES)} complete five-arm samples',flush=True)
    finish(out,partials,b,known,hist,pa,pd)
    print(f'[HIST-REGRESSION] PASS: {len(SAMPLES)*2*8} CNN/F1 sample×radius comparisons',flush=True)
    print(f'[INFINITY-GATE] PASS: {len(SAMPLES)*5} LXXIX exact-sample W22 comparisons',flush=True)
    print(f'[PILOT-CONTINUITY] PASS: {len(b.SAMPLES)*5*8} pilot aggregate comparisons plus dimensional checks',flush=True)
    print('[FULL-GATE] PASS: 13440 dimension rows, 6720 aggregate rows, isolated output with checksums',flush=True)
    print('[OUTPUT] '+str(out),flush=True)


def verify(root):
    b,known,hist,pa,pd,expected=preflight(root)
    out=root/OUT;source_match(root,out,expected)
    sums=out/'SHA256SUMS';need(sums.is_file(),'Final SHA256SUMS missing: full sweep not completed')
    observed=[]
    for line in sums.read_text().splitlines():
        digest,name=line.split('  ',1)
        need(sha(out/name)==digest,'Modified/missing output '+name)
        observed.append(name)
    need(len(observed)==7,'Final manifest requires exactly seven outputs')
    aggregate=rows_csv(out/'all168_locality_aggregate.csv')
    dimension=rows_csv(out/'all168_locality_dimension.csv')
    need(len(aggregate)==6720 and len(dimension)==13440,'Unexpected final CSV count')
    ai=index_unique(aggregate,row_key,'verify-aggregate')
    di=index_unique(dimension,dim_key,'verify-dimension')
    need(len(ai)==6720 and len(di)==13440,'Full key uniqueness mismatch')
    for s in SAMPLES:
        p=out/'checkpoints'/f's{s:03d}.json'
        need(p.is_file(),'Missing final checkpoint '+str(p))
        obj=json.loads(p.read_text())
        validate_partial(obj,s,b,known,hist,pa,pd)
        for v in obj['aggregate_rows']:
            compare_row(ai[row_key(v)],v,'Final aggregate checkpoint')
        for v in obj['dimension_rows']:
            compare_row(di[dim_key(v)],v,'Final dimension checkpoint')
    summary=rows_csv(out/'all168_macro_summary.csv')
    need(len(summary)==5*8,'Wrong macro summary count')
    for row in summary:
        subset=[float(ai[s,row['arm'],row['radius']]['constrained_w22_all']) for s in SAMPLES]
        near(row['mean_constrained_w22_all'],statistics.mean(subset),'Macro mean mismatch')
    print('[VERIFY-GATE] PASS: all 1008 input hashes, 168 checkpoint validations, 6720 aggregate and 13440 dimension rows, final SHA256 and mean recomputation',flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode',required=True,choices=['preflight','full','verify'])
    ap.add_argument('--repo-root',type=Path,default=Path.cwd())
    args=ap.parse_args();root=args.repo_root.resolve()
    if args.mode=='full':full(root)
    elif args.mode=='verify':verify(root)
    else:
        preflight(root)
        print('[PREFLIGHT] PASS: source/pilot/provenance gate only, no locality matching or outputs written',flush=True)
if __name__=='__main__':
    try:main()
    except Exception as exc:
        print(f'[LXXXIII] ERROR {type(exc).__name__}: {exc}',file=sys.stderr,flush=True)
        raise SystemExit(1)
