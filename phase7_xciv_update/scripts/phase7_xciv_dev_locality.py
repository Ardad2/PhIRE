#!/usr/bin/env python3
"""Phase 7 XCIV: original Phase5E locality solver on four development-only XCII PDs.

Fail closed if the original VTK reader cannot reconstruct critical endpoints or
if its persistence pairs do not match the corrected XCIII GUDHI parser.
Never reads/extracts locked-test fields, retrains models, or edits old artifacts.
"""
from __future__ import annotations
import argparse, csv, hashlib, importlib.util, json, math, os, re, shutil, sys, tempfile
from pathlib import Path
from statistics import mean

XCIII_SCRIPT=Path('phase7_xciii_update/scripts/phase7_xciii_dev_corrected_pd.py')
XCIII=Path('ttk_runs_fixed/topology_finetuning/phase7_xciii_dev_corrected_pd_20261016')
XCII=Path('ttk_runs_fixed/topology_finetuning/phase7_xcii_dev_canonical_pd_20261015')
SOLVER=Path('spatial_pd/phase5e_sparse_locality_solver.py')
SOLVER_SHA='47c73decca4ba151fd8b250cae2911b4c2307471fe9c881fc42bfc968e211c3a'
BASE=Path('ttk_runs_fixed/topology_finetuning')
PHASE6=BASE/'phase6_lxxxiii_spatial_all168_20261014'
SELF=Path('phase7_xciv_update/scripts/phase7_xciv_dev_locality.py')
OUT=BASE/'phase7_xciv_dev_locality_20261017'
ARMS=('cnn','gan','grad','cv','pers','f1')
ALL=('gt',*ARMS)
N=4
RADII=(4.,8.,16.,32.,64.,96.,128.,math.inf)
A_FIELDS=('sample','region','preview','arm','radius','unconstrained_w22_all',
    'constrained_w22_all','locality_gap_all','gt_persistence_coverage_all',
    'sr_persistence_coverage_all','total_sq','rr_sq','gt_diag_sq','sr_diag_sq',
    'real_real_all','gt_to_diagonal_all','diagonal_to_sr_all')
D_FIELDS=('sample','region','arm','dimension','radius','w22','total_sq',
    'gt_persistence_coverage','sr_persistence_coverage','real_real','gt_to_diagonal','diagonal_to_sr')


def need(ok,msg):
    if not ok: raise RuntimeError(msg)

def sha(path):
    need(path.is_file() and path.stat().st_size>0,f'Missing/empty file: {path}')
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()

def at(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    need(spec is not None and spec.loader is not None,f'Cannot import {path}')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def rk(r):return 'inf' if math.isinf(r) else str(r)

def near(a,b,label,atol=2e-6,rtol=1e-9):
    a,b=float(a),float(b)
    need(math.isfinite(a) and math.isfinite(b) and math.isclose(a,b,abs_tol=atol,rel_tol=rtol),
         f'{label}: {a} != {b}')

def read_csv(path):
    with path.open(newline='') as f:return list(csv.DictReader(f))

def source(root):
    need(sha(root/SELF) and (root/XCIII_SCRIPT).is_file(),'XCIV/XCIII script absent')
    xc=at(root/XCIII_SCRIPT,'pinned_xciii')
    pilot,decoder,fields,src=xc.source_gate(root,regression=True)
    xc.verified_hashes(root/XCIII,['per_field_corrected_pd.csv','numeric_manifest.json','numeric_report.md'])
    report=json.loads((root/XCIII/'numeric_manifest.json').read_text())
    need(report['format']=='phase7-xciii-corrected-pd-v1' and report['status']=='DEVELOPMENT_ONLY',
         'XCIII numerical source is not frozen development output')
    need(report['sources']==src and report['fields']==fields and report['arms']==list(ARMS),
         'XCIII source lineage differs from active inputs')
    scores=read_csv(root/XCIII/'per_field_corrected_pd.csv')
    xc.validate_rows(scores,fields,src)
    score_by={(int(r['sample']),r['arm']):r for r in scores}
    need(len(score_by)==N*len(ARMS),'Duplicate numeric score rows')
    # Original Phase5E implementation and previously completed full-168 reference.
    need(sha(root/SOLVER)==SOLVER_SHA,
         'Original Phase5E solver source changed. Stop; do not substitute a new optimizer.')
    full=root/PHASE6
    need(full.is_dir(),'Missing previously verified Phase6 full-168 locality reference')
    names=['run_manifest.json','all168_locality_aggregate.csv','all168_locality_dimension.csv',
           'all168_macro_summary.csv','all168_paired_by_radius.csv','all168_factorial_by_radius.csv','summary.md']
    xc.verified_hashes(full,names)
    old=json.loads((full/'run_manifest.json').read_text())
    need(old['original_solver_sha256']==SOLVER_SHA,'Phase6 reference used different solver SHA')
    solver=at(root/SOLVER,'pinned_phase5e_sparse_solver')
    need(tuple(solver.RADII)==RADII and tuple(solver.DIMS)==(0,1) and
         callable(solver.read_features) and callable(solver.SparsePrepared) and
         callable(solver.gudhi_w22),'Original locality implementation API changed')
    print('[SOURCE] PASS: XCII/XCIII hashes and corrected numerical scores; frozen XCI source and roles',flush=True)
    print('[SOLVER] PASS: original Phase5E SHA256 and audited full-168 reference; radii 4..128,inf',flush=True)
    return xc,pilot,decoder,solver,fields,score_by,src


def pairs(features,expected,path):
    """Strict coordinate/persistence check against canonical *numeric* parser.

    Cannot silently interpret persistence-diagram coordinates as spatial ones.
    """
    import numpy as np
    observed={}
    for dim in (0,1):
        fs=features[dim]
        need(len(fs)>0,f'Empty D{dim} finite positive locality features: {path}')
        vals=[]
        for f in fs:
            need(set(('birth','death','persistence','birth_x','birth_y','death_x','death_y'))<=set(f),
                 f'Missing birth/death critical coordinates: {path}')
            a,b,p=(float(f[k]) for k in ('birth','death','persistence'))
            need(all(math.isfinite(float(f[k])) for k in
                 ('birth','death','persistence','birth_x','birth_y','death_x','death_y')),
                 f'Nonfinite feature or coordinate: {path}')
            need(p>0 and b>a and math.isclose(b-a,p,abs_tol=3e-5,rel_tol=2e-6),
                 f'Invalid persistence feature: {path}')
            for k in ('birth_x','birth_y','death_x','death_y'):
                v=float(f[k]);need(-1e-4<=v<=159+1e-4,f'Critical position outside 160 crop {k}={v}: {path}')
            vals.append((a,b))
        expected_dim=np.asarray(expected[dim],dtype=float)
        observed_dim=np.asarray(vals,dtype=float)
        need(expected_dim.shape==observed_dim.shape,
             f'VTK locality feature counts differ from corrected canonical diagram: D{dim} {path}: '
             f'{observed_dim.shape} vs {expected_dim.shape}')
        obs=observed_dim[np.lexsort((observed_dim[:,1],observed_dim[:,0]))]
        exp=expected_dim[np.lexsort((expected_dim[:,1],expected_dim[:,0]))]
        need(np.allclose(obs,exp,rtol=2e-6,atol=3e-5),
             f'VTK locality birth/death values differ from corrected parser: D{dim} {path}')
        observed[dim]=len(fs)
    return observed


def readiness(root,xc,pilot,decoder,solver,fields):
    decoded={}
    counts={}
    for s in range(N):
        for arm in ALL:
            path=root/XCII/xc.pd_name(arm,s)
            # vtkXMLUnstructuredGridReader may emit ERR and return empty.
            # We always refuse zero or invalid content, never accept shell exit alone.
            f,z=solver.read_features(path)
            expected=pilot.parse(decoder.read_pd,path)
            counts[f'{s}:{arm}']=pairs(f,expected,path)
            decoded[(s,arm)]=f
        print(f'[COORDINATES] PASS {s+1}/{N} field={fields[s]["name"]} 7/7 sources; corrected birth/death agreement',flush=True)
    return decoded,counts


def compute_rows(solver,decoded,fields,scores,progress=True):
    dim_rows,aggregate=[],[]
    for s in range(N):
        gt=decoded[(s,'gt')]
        for arm in ARMS:
            pred=decoded[(s,arm)]
            results={}
            independent={}
            for dim in (0,1):
                prep=solver.SparsePrepared(gt[dim],pred[dim])
                gudhi=float(solver.gudhi_w22(gt[dim],pred[dim]))
                independent[dim]=gudhi
                per_dim=[]
                prior=math.inf
                for radius in RADII:
                    item=prep.solve(radius)
                    v=float(item['w22'])
                    need(math.isfinite(v) and v>=0 and v<=prior+1e-7,
                         f'Invalid/nonmonotone W22 field={s} model={arm} D{dim} r={rk(radius)}')
                    prior=v
                    near(v*v,float(item['total_sq']),f'Cost square field={s} arm={arm} D{dim} r={rk(radius)}',atol=1e-6)
                    need(float(item['gt_total_persistence'])>0 and float(item['sr_total_persistence'])>0,
                         'Missing total persistence mass')
                    if math.isinf(radius):near(v,gudhi,f'Phase5E/GUDHI D{dim} field={s} arm={arm}',atol=1e-7)
                    for c in ('gt_persistence_coverage','sr_persistence_coverage'):
                        cov=item[c]
                        need(cov is not None and -1e-9<=float(cov)<=1+1e-9,
                             f'Invalid {c}: field={s} arm={arm} D{dim} r={rk(radius)}')
                    per_dim.append(item)
                    dim_rows.append({'sample':s,'region':fields[s]['region'],'arm':arm,
                                     'dimension':dim,'radius':rk(radius),
                                     **{k:item[k] for k in
                                     ('w22','total_sq','gt_persistence_coverage',
                                      'sr_persistence_coverage','real_real','gt_to_diagonal','diagonal_to_sr')}})
                results[dim]=per_dim
            expected=float(scores[s,arm]['W22'])
            near(math.sqrt(sum(independent[d]**2 for d in (0,1))),expected,
                 f'Frozen XCIII W22 field={s} arm={arm}',atol=1e-6)
            for idx,radius in enumerate(RADII):
                a,b=results[0][idx],results[1][idx]
                total=float(a['total_sq'])+float(b['total_sq'])
                actual=math.sqrt(max(0.,total))
                cov_gt=(float(a['gt_matched_persistence'])+float(b['gt_matched_persistence']))/(
                        float(a['gt_total_persistence'])+float(b['gt_total_persistence']))
                cov_sr=(float(a['sr_matched_persistence'])+float(b['sr_matched_persistence']))/(
                        float(a['sr_total_persistence'])+float(b['sr_total_persistence']))
                need(0<=cov_gt<=1+1e-9 and 0<=cov_sr<=1+1e-9,'Coverage outside [0,1]')
                near(total,sum(float(t[k]) for t in (a,b)
                               for k in ('rr_sq','gt_diag_sq','sr_diag_sq')),
                     f'Aggregate decomposition field={s} arm={arm} r={rk(radius)}',atol=1e-6)
                if math.isinf(radius):near(actual,expected,f'Infinite radius equals XCIII W22 s={s} {arm}',atol=1e-6)
                aggregate.append({'sample':s,'region':fields[s]['region'],'preview':fields[s]['name'],
                    'arm':arm,'radius':rk(radius),'unconstrained_w22_all':expected,
                    'constrained_w22_all':actual,'locality_gap_all':actual-expected,
                    'gt_persistence_coverage_all':cov_gt,'sr_persistence_coverage_all':cov_sr,
                    'total_sq':total,'rr_sq':float(a['rr_sq'])+float(b['rr_sq']),
                    'gt_diag_sq':float(a['gt_diag_sq'])+float(b['gt_diag_sq']),
                    'sr_diag_sq':float(a['sr_diag_sq'])+float(b['sr_diag_sq']),
                    'real_real_all':int(a['real_real'])+int(b['real_real']),
                    'gt_to_diagonal_all':int(a['gt_to_diagonal'])+int(b['gt_to_diagonal']),
                    'diagonal_to_sr_all':int(a['diagonal_to_sr'])+int(b['diagonal_to_sr'])})
            if progress:print(f'[LOCALITY] {s*len(ARMS)+ARMS.index(arm)+1:02d}/24 field={s} arm={arm} '
                               f'r4={aggregate[-8]["constrained_w22_all"]:.6f} '
                               f'rinf={aggregate[-1]["constrained_w22_all"]:.6f} '
                               f'GTcover_r16={aggregate[-6]["gt_persistence_coverage_all"]:.4f}',flush=True)
    need(len(aggregate)==N*len(ARMS)*len(RADII) and
         len(dim_rows)==N*len(ARMS)*len(RADII)*2,'Incorrect locality output row count')
    return aggregate,dim_rows


def report(rows):
    lines=['# Phase 7 XCIV — development-only canonical spatial locality','',
        'Original Phase5E sparse solver on four 160×160 top-left development fields.',
        'The radii are **WTK grid cells**, not kilometers. Lower constrained W22 is better;',
        'GT persistence-mass coverage is a separate higher-is-better statistic.','',
        '**Exploratory four-field numerical pilot, not a geographic-generalization estimate.**','',
        '| Arm | r4 W22 ↓ | r16 W22 ↓ | r32 W22 ↓ | r∞ W22 ↓ | GT coverage r16 ↑ |',
        '|---|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        vals={rad:[r for r in rows if r['arm']==arm and r['radius']==rad]
              for rad in ('4.0','16.0','32.0','inf')}
        need(all(len(x)==N for x in vals.values()),'Incorrect macro membership')
        c=lambda rad:mean(float(x['constrained_w22_all']) for x in vals[rad])
        cover=mean(float(x['gt_persistence_coverage_all']) for x in vals['16.0'])
        lines.append(f'| {arm} | {c("4.0"):.6f} | {c("16.0"):.6f} | {c("32.0"):.6f} | {c("inf"):.6f} | {cover:.6f} |')
    lines.extend(['','## Boundaries','',
        '- All six model checkpoints were fixed before this evaluation.',
        '- No locked test-region model predictions or topology scores accessed.',
        '- Coordinates validated against original corrected numeric persistence pairs.',
        '- Same Phase5E optimizer and radius definitions as the frozen historical analysis.',
        '- Infinity-radius aggregate values numerically anchored to the frozen XCIII W22.',
        '- No terrain geodesic conversion, independence assumption across hours, or model selection.',
        '- Four development fields alone cannot test robustness across complete seasonal weeks.',''])
    return '\n'.join(lines)+'\n'


def write_csv(path,fields,rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader();writer.writerows(rows)


def save(root,aggs,dims,counts,fields,src):
    target=root/OUT
    need(not target.exists(),f'XCIV output already exists; never overwrite: {target}')
    target.parent.mkdir(parents=True,exist_ok=True)
    staging=Path(tempfile.mkdtemp(prefix='.'+target.name+'.staging-',dir=target.parent))
    try:
        write_csv(staging/'locality_aggregate.csv',A_FIELDS,aggs)
        write_csv(staging/'locality_dimension.csv',D_FIELDS,dims)
        (staging/'locality_report.md').write_text(report(aggs))
        manifest={'format':'phase7-xciv-locality-pilot-v1','scope':'DEVELOPMENT_ONLY',
          'n_fields':N,'arms':list(ARMS),'radii':[rk(r) for r in RADII],
          'source_xciii_sha256':sha(root/XCIII/'numeric_manifest.json'),
          'xcII_source_sha256':sha(root/XCII/'extraction_manifest.json'),
          'xciii_verified_source':src,'fields':fields,'coord_feature_counts':counts,
          'phase5e_solver_sha256':SOLVER_SHA,'this_script_sha256':sha(root/SELF),
          'n_aggregate_rows':len(aggs),'n_dimension_rows':len(dims),
          'output_semantics':'Grid-cell endpoint constrained sparse assignment; not kilometer physical distance'}
        (staging/'locality_manifest.json').write_text(json.dumps(manifest,sort_keys=True,indent=2,allow_nan=False)+'\n')
        items=['locality_aggregate.csv','locality_dimension.csv','locality_report.md','locality_manifest.json']
        (staging/'SHA256SUMS').write_text(''.join(f'{sha(staging/x)}  {x}\n' for x in items))
        need(not target.exists(),'XCIV output appeared while staging')
        staging.rename(target)
    except BaseException:
        shutil.rmtree(staging,ignore_errors=True)
        raise
    print(f'[LOCALITY-GATE] PASS: {len(aggs)} aggregate + {len(dims)} dimension rows; '
          f'24 infinite-radius matches XCIII; published={target}',flush=True)


def verify(root,xc,solver,fields,scores,src,pilot,decoder):
    target=root/OUT
    need(target.is_dir(),f'XCIV output missing: {target}')
    xc.verified_hashes(target,['locality_aggregate.csv','locality_dimension.csv','locality_report.md','locality_manifest.json'])
    saved=json.loads((target/'locality_manifest.json').read_text())
    need(saved['format']=='phase7-xciv-locality-pilot-v1' and saved['scope']=='DEVELOPMENT_ONLY',
         'Manifest scope/format mismatch')
    need(saved['source_xciii_sha256']==sha(root/XCIII/'numeric_manifest.json') and
         saved['xcII_source_sha256']==sha(root/XCII/'extraction_manifest.json') and
         saved['xciii_verified_source']==src and saved['fields']==fields and
         saved['phase5e_solver_sha256']==SOLVER_SHA and saved['this_script_sha256']==sha(root/SELF),
         'XCIV provenance drift')
    decoded,counts=readiness(root,xc,pilot,decoder,solver,fields)
    need(counts==saved['coord_feature_counts'],'Coordinate feature counts changed')
    aggs,dims=compute_rows(solver,decoded,fields,scores,progress=False)
    need(saved['n_aggregate_rows']==192 and saved['n_dimension_rows']==384,'Saved cardinalities differ')
    for name,columns,reference in [('locality_aggregate.csv',A_FIELDS,aggs),
                                   ('locality_dimension.csv',D_FIELDS,dims)]:
        rows=read_csv(target/name)
        need(len(rows)==len(reference),f'{name} row count differs')
        for i,(actual,expected) in enumerate(zip(rows,reference)):
            need(tuple(actual)==columns,f'{name} column keys/order differ')
            for col in columns:
                if isinstance(expected[col],float): near(float(actual[col]),expected[col],f'{name}:{i}:{col}',atol=1e-6)
                else: need(str(actual[col])==str(expected[col]),f'{name}:{i}:{col} differs')
    need((target/'locality_report.md').read_text()==report(aggs),'Locality report differs from recomputation')
    print('[VERIFY-GATE] PASS: unchanged XCIII/XCII/XCI lineage, independent full locality recomputation, '
          '192 aggregate and 384 dimension rows, four SHA256 results, zero external samples',flush=True)


def selftest():
    need([rk(x) for x in RADII]==['4.0','8.0','16.0','32.0','64.0','96.0','128.0','inf'],
         'Radius serialization drift')
    near(0.0,0.0,'identity')
    try:near(1.0,2.0,'intentional error');raise AssertionError('near failed open')
    except RuntimeError:pass
    need(len(A_FIELDS)==17 and len(D_FIELDS)==12,'CSV schema drift')
    print('[SELFTEST] PASS: frozen radius labels, strict numerical checks, CSV schemas (no data or TF)',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',required=True,choices=('selftest','preflight','compute','verify'))
    parser.add_argument('--repo-root',type=Path,default=Path('.'))
    arg=parser.parse_args()
    if arg.mode=='selftest':selftest();return
    root=arg.repo_root.resolve()
    xc,pilot,decoder,solver,fields,scores,src=source(root)
    if arg.mode!='verify':need(not (root/OUT).exists(),f'XCIV already exists; verify, never overwrite: {root/OUT}')
    if arg.mode=='preflight':
        readiness(root,xc,pilot,decoder,solver,fields)
        print('[PREFLIGHT] PASS: 28 feature/position arrays agree with corrected PD parser; '
              'no outputs, no locked-region access',flush=True)
    elif arg.mode=='compute':
        decoded,counts=readiness(root,xc,pilot,decoder,solver,fields)
        aggs,dims=compute_rows(solver,decoded,fields,scores)
        save(root,aggs,dims,counts,fields,src)
    else:verify(root,xc,solver,fields,scores,src,pilot,decoder)

if __name__=='__main__':
    try:main()
    except Exception as e:
        print(f'[XCIV] ERROR {type(e).__name__}: {e}',file=sys.stderr,flush=True)
        sys.exit(1)
