#!/usr/bin/env python3
"""Phase 7 XCIX — seasonal16 spatial locality on development-only canonical PDs.

Delegates ALL assignment/matching logic to unchanged, pinned Phase7 XCIV
compute_rows and original Phase5E SparsePrepared. Reconciles canonical zero/
near-zero persistence with original reader's skip counts, requires 96 infinity
W2,2 values equal XCVIII and all 24 original XCIV comparisons reproduce.
Does not touch model checkpoints, locked regions, or frozen protocol.
"""
from __future__ import annotations
import argparse, csv, hashlib, importlib.util, json, math, shutil, sys, tempfile
from pathlib import Path
from statistics import mean

BASE=Path('ttk_runs_fixed/topology_finetuning')
XCVII=BASE/'phase7_xcvii_dev_seasonal16_pd_20261018'
XCVIII=BASE/'phase7_xcviii_dev_seasonal16_corrected_pd_20261018'
XCIV=BASE/'phase7_xciv_dev_locality_20261017'
OUT=BASE/'phase7_xcix_dev_seasonal16_locality_20261018'
SELF=Path('phase7_xcix_update/scripts/phase7_xcix_dev_seasonal16_locality.py')
ORIG=Path('phase7_xciv_update/scripts/phase7_xciv_dev_locality.py')
COMPAT=Path('phase7_xciv_zero_persistence_compat_20261017/scripts/phase7_xciv_zero_compat.py')
NUMERIC=Path('phase7_xcviii_update/scripts/phase7_xcviii_dev_seasonal16_numeric_pd.py')
ORIG_SHA='71168242dbac580936250122385315c153999e4d5ef27e83b4a7ad6c93da63ad'
COMPAT_SHA='90724b801b7a430109cdecc327addc07a9eef395e07e884a7590c05e2f2e014e'
NUMERIC_SHA='5f2522cf5fb170138634bae21643dc32f9404f7ed9efad226a467fc5c0db9ff4'
SOLVER_SHA='47c73decca4ba151fd8b250cae2911b4c2307471fe9c881fc42bfc968e211c3a'
ARMS=('cnn','gan','grad','cv','pers','f1')
ALL=('gt',*ARMS)
N=16
MAPPING={0:0,1:4,2:8,3:12}
OUTPUTS=('locality_aggregate.csv','locality_dimension.csv','region_week_locality.csv','locality_report.md','locality_manifest.json')
BLOCK_COLUMNS=('region','season_utc','arm','n_fields','r4_w22','r16_w22','r32_w22','rinf_w22','r16_gt_persistence_coverage')


def need(ok,msg):
    if not ok:raise RuntimeError(msg)


def sha(p):
    p=Path(p);need(p.is_file() and p.stat().st_size>0,f'Missing/empty file {p}')
    h=hashlib.sha256()
    with p.open('rb') as f:
        for c in iter(lambda:f.read(1<<20),b''):h.update(c)
    return h.hexdigest()


def load(p,name):
    spec=importlib.util.spec_from_file_location(name,p)
    need(spec is not None and spec.loader is not None,f'Cannot load {p}')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod


def near(a,b,label,atol=1e-6,rtol=1e-9):
    a=float(a);b=float(b)
    need(math.isfinite(a) and math.isfinite(b) and math.isclose(a,b,abs_tol=atol,rel_tol=rtol),
         f'{label}: {a} != {b}')


def read_rows(p,cols):
    with p.open(newline='') as f:
        reader=csv.DictReader(f)
        need(tuple(reader.fieldnames or ())==cols,f'Schema mismatch {p}')
        return list(reader)


def sources(root):
    # Keep every source stable and reject substitutions of the archived algorithms.
    for p,expected in ((ORIG,ORIG_SHA),(COMPAT,COMPAT_SHA),(NUMERIC,NUMERIC_SHA)):
        need(sha(root/p)==expected,f'Pinned script hash mismatch: {p}')
    orig=load(root/ORIG,'xcix_pinned_original_xciv')
    compat=load(root/COMPAT,'xcix_pinned_original_compat')
    numeric=load(root/NUMERIC,'xcix_pinned_numeric_xcviii')
    old,pilot,decoder,fields,numsrc,old_lookup=numeric.verified_sources(root)
    numeric.verify(root,pilot,decoder,fields,numsrc,old_lookup)
    # XCVIII verifier fully recalculates all 96 GUDHI distances.
    scored=read_rows(root/XCVIII/'per_field_corrected_pd.csv',numeric.COLS)
    numeric.check_rows(scored,fields,numsrc)
    scores={(int(x['sample']),x['arm']):x for x in scored}
    need(len(scores)==96,'Expected 96 unique numerical comparisons')
    # Read and verify original XCIV four-field locality outputs and compatibility schema.
    oldhash=old.verified_hashes(root/XCIV,['locality_aggregate.csv','locality_dimension.csv',
                                         'locality_report.md','locality_manifest.json'])
    oldmanifest=json.loads((root/XCIV/'locality_manifest.json').read_text())
    need(oldmanifest['format']=='phase7-xciv-locality-pilot-v1' and
         oldmanifest['scope']=='DEVELOPMENT_ONLY' and
         oldmanifest['phase5e_solver_sha256']==SOLVER_SHA and
         oldmanifest['coordinate_compatibility']['launcher_sha256']==COMPAT_SHA and
         oldmanifest['coordinate_compatibility']['original_driver_sha256']==ORIG_SHA and
         oldmanifest['n_aggregate_rows']==192 and oldmanifest['n_dimension_rows']==384,
         'Original XCIV locality reference/compatibility provenance changed')
    old_aggs=read_rows(root/XCIV/'locality_aggregate.csv',orig.A_FIELDS)
    old_dims=read_rows(root/XCIV/'locality_dimension.csv',orig.D_FIELDS)
    need(len(old_aggs)==192 and len(old_dims)==384,'Original XCIV locality row count mismatch')
    need(sha(root/orig.SOLVER)==SOLVER_SHA,'Original Phase5E solver source changed')
    phase6=root/orig.PHASE6
    old.verified_hashes(phase6,['run_manifest.json','all168_locality_aggregate.csv',
        'all168_locality_dimension.csv','all168_macro_summary.csv',
        'all168_paired_by_radius.csv','all168_factorial_by_radius.csv','summary.md'])
    need(json.loads((phase6/'run_manifest.json').read_text())['original_solver_sha256']==SOLVER_SHA,
         'Frozen Phase6 168-field reference used different solver')
    solver=load(root/orig.SOLVER,'xcix_original_phase5e')
    need(tuple(solver.RADII)==orig.RADII and solver.TOL==1e-10 and tuple(solver.DIMS)==(0,1),
         'Frozen radius grid or filtering tolerance differs')
    # The original four XCIV fields map exactly to seasonal indices 0,4,8,12.
    need(tuple(fields[i]['region'] for i in range(N))==('gulf_inland',)*8+('southwest_borderland',)*8,
         'Out of scope geographical region')
    need(tuple(fields[i]['wtk_index'] for i in range(N))==
         tuple([27048,27132,29184,29268,31392,31476,33600,33684]*2),
         'Region/time index differs from frozen development fixture')
    need(tuple(fields[i]['offset_hours'] for i in range(N))==tuple([0,84]*8),
         'Unapproved hour offset')
    orig.N=N
    orig.XCII=XCVII
    orig.XCIII=XCVIII
    # Original compute_rows() uses fields[s]['name'] only as metadata; no changed math.
    augmented=[dict(f,name=f['filename']) for f in fields]
    source={
        'format':'phase7-xcix-seasonal16-locality-source-v1',
        'XCIV_manifest_sha256':sha(root/XCIV/'locality_manifest.json'),
        'XCIV_SHA256SUMS_sha256':sha(root/XCIV/'SHA256SUMS'),
        'XCVIII_manifest_sha256':sha(root/XCVIII/'numeric_manifest.json'),
        'XCVIII_SHA256SUMS_sha256':sha(root/XCVIII/'SHA256SUMS'),
        'XCVII_manifest_sha256':sha(root/XCVII/'extraction_manifest.json'),
        'XCVIII_verified_source_sha256':numsrc,
        'Phase5E_solver_sha256':SOLVER_SHA,
        'XCIV_script_sha256':ORIG_SHA,
        'XCIV_compat_sha256':COMPAT_SHA,
        'XCVIII_script_sha256':NUMERIC_SHA,
        'XCIX_script_sha256':sha(root/SELF),
    }
    need(len(oldhash)==4,'Incorrect original locality file hash cardinality')
    print('[SOURCE] PASS: frozen XCIV/XCVIII/XCVII/XCI lineage, Phase6 regression, 96 numerical GUDHI scores',flush=True)
    print('[SOLVER] PASS: original Phase5E sparse assignment, eight radii, 168-field reference, SHA256 pinned',flush=True)
    print('[SCOPE] PASS: 16 development-only fields; no locked geographic model outputs',flush=True)
    return orig,compat,solver,pilot,decoder,fields,augmented,scores,old_aggs,old_dims,source


def readiness(root,orig,compat,solver,pilot,decoder,fields):
    import numpy as np
    decoded={};counts={};zeros={}
    for i in range(N):
        for arm in ALL:
            path=root/XCVII/f'pd/dev_{arm}_s{i}_speed_p160_x0_y0_pd_port_0.vtu'
            features,skipped=solver.read_features(path)
            expected=pilot.parse(decoder.read_pd,path)
            kept={};removed={}
            for dim in (0,1):
                subset,excluded=compat.filtered_pairs(expected[dim],float(solver.TOL),np)
                need(excluded==int(skipped[dim]),f'Zero persistence mismatch {arm} s{i} H{dim}: {excluded} vs {skipped[dim]}')
                kept[dim]=subset
                removed[str(dim)]=excluded
            # Original, unmodified XCIV endpoint-bounds/birth-death consistency check.
            matched=orig.pairs(features,kept,path)
            decoded[(i,arm)]=features
            counts[f'{i}:{arm}']={str(dim):int(matched[dim]) for dim in (0,1)}
            zeros[f'{i}:{arm}']=removed
        print(f'[COORDINATES] PASS {i+1:02d}/16 region={fields[i]["region"]} '
              f'WTK={fields[i]["wtk_index"]}: all seven H0/H1 coordinate sets match',flush=True)
    need(len(decoded)==112 and len(zeros)==112 and len(counts)==112,'Readiness completeness mismatch')
    return decoded,counts,zeros


def compare_old4(orig,aggregate,dimensions,old_aggs,old_dims):
    ag_lookup={(int(r['sample']),r['arm'],r['radius']):r for r in old_aggs}
    d_lookup={(int(r['sample']),r['arm'],r['radius'],int(r['dimension'])):r for r in old_dims}
    need(len(ag_lookup)==192 and len(d_lookup)==384,'Duplicate XCIV old rows')
    checked_a=0;checked_d=0
    for r in aggregate:
        i=int(r['sample'])
        if i not in MAPPING.values():continue
        oldi=next(oi for oi,ni in MAPPING.items() if ni==i)
        prev=ag_lookup[(oldi,r['arm'],r['radius'])]
        for col in orig.A_FIELDS:
            if col in ('sample','preview'):continue
            if col in ('region','arm','radius'):
                need(str(r[col])==str(prev[col]),f'Old XCIV metadata mismatch {i} {col}')
            else:near(r[col],prev[col],f'Old XCIV aggregate {i}:{r["arm"]}:{r["radius"]}:{col}',atol=2e-6)
        checked_a+=1
    for r in dimensions:
        i=int(r['sample'])
        if i not in MAPPING.values():continue
        oldi=next(oi for oi,ni in MAPPING.items() if ni==i)
        prev=d_lookup[(oldi,r['arm'],r['radius'],int(r['dimension']))]
        for col in orig.D_FIELDS:
            if col=='sample':continue
            if col in ('region','arm','radius','dimension'):
                need(str(r[col])==str(prev[col]),f'Old XCIV dimension metadata mismatch {i} {col}')
            else:near(r[col],prev[col],f'Old XCIV dimension {i}:{r["arm"]}:{r["radius"]}:{col}',atol=2e-6)
        checked_d+=1
    need(checked_a==192 and checked_d==384,'Four-field locality regression incomplete')
    print('[OLD-4] PASS: 192 archived aggregate and 384 H0/H1 rows reproduce across original four fields',flush=True)


def group(aggs,fields):
    blocks=[]
    for region in ('gulf_inland','southwest_borderland'):
        for date in ('2010-02-01','2010-05-01','2010-08-01','2010-11-01'):
            sampleids={i for i,f in enumerate(fields) if f['region']==region and f['anchor_utc'].startswith(date)}
            need(len(sampleids)==2 and {fields[i]['offset_hours'] for i in sampleids}=={0,84},
                 'Region-week block requires exactly 0h and 84h')
            for arm in ARMS:
                matches={(int(r['sample']),r['radius']):r for r in aggs
                         if int(r['sample']) in sampleids and r['arm']==arm}
                need(len(matches)==16,'Missing/duplicate region-week locality comparison')
                def av(key,rad):return mean(float(matches[i,rad][key]) for i in sorted(sampleids))
                blocks.append({'region':region,'season_utc':date,'arm':arm,'n_fields':'2',
                    'r4_w22':f'{av("constrained_w22_all","4.0"):.17g}',
                    'r16_w22':f'{av("constrained_w22_all","16.0"):.17g}',
                    'r32_w22':f'{av("constrained_w22_all","32.0"):.17g}',
                    'rinf_w22':f'{av("constrained_w22_all","inf"):.17g}',
                    'r16_gt_persistence_coverage':f'{av("gt_persistence_coverage_all","16.0"):.17g}'})
    need(len(blocks)==48,'Region/week rows mismatch')
    return blocks


def report(aggs,blocks):
    lines=['# Phase 7 XCIX — spatial locality for 16 development-only wind fields','',
           'Original, unchanged Phase5E sparse locality matching, canonical top-left 160×160 speed crop.',
           '16 predeclared 2010 development snapshots, two per region-week in two regions.',
           '**Descriptive engineering data, NOT 168-hour regional/seasonal generalization statistics.**','',
           'Lower spatially constrained W₂,₂ is better; GT persistence-mass coverage is separate and higher is better.',
           'All radii are **WTK grid cells**, not kilometers.','',
           '| Model | r4 W22 ↓ | r16 W22 ↓ | r32 W22 ↓ | r∞ W22 ↓ | r16 GT coverage ↑ |',
           '|---|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        def m(key,radius):
            v=[float(r[key]) for r in aggs if r['arm']==arm and r['radius']==radius]
            need(len(v)==N,'Incorrect overall mean membership')
            return mean(v)
        lines.append(f'| {arm} | {m("constrained_w22_all","4.0"):.6f} | '
                     f'{m("constrained_w22_all","16.0"):.6f} | '
                     f'{m("constrained_w22_all","32.0"):.6f} | '
                     f'{m("constrained_w22_all","inf"):.6f} | '
                     f'{m("gt_persistence_coverage_all","16.0"):.6f} |')
    lines+=['','## Region-week means (two fields each)','',
            '| Region | 2010 week | Model | r4 W22 ↓ | r16 W22 ↓ | r∞ W22 ↓ | GT coverage r16 ↑ |',
            '|---|---|---|---:|---:|---:|---:|']
    for r in blocks:
        lines.append(f'| {r["region"]} | {r["season_utc"]} | {r["arm"]} | '
            f'{float(r["r4_w22"]):.5f} | {float(r["r16_w22"]):.5f} | '
            f'{float(r["rinf_w22"]):.5f} | {float(r["r16_gt_persistence_coverage"]):.5f} |')
    lines+=['','## Method and evidence boundaries','',
            '- Reuses the original Phase5E feature-coordinate reader and SparsePrepared assignment optimizer without source edits.',
            '- Canonical zero/near-zero persistence pairs excluded using the original Phase5E tolerance, and counts compared in all 112 H0/H1 diagrams.',
            '- All 96 infinity-radius W22 distances must reproduce frozen XCVIII corrected GUDHI values.',
            '- All 192 aggregate and 384 dimension-level rows for the original four development fields must reproduce XCIV.',
            '- Persistence-mass coverage is not equivalent to accurate physical placement of features.',
            '- Geographical roles, all checkpoint parameters, and all eight locality radii remain frozen by XCI.',
            '- These 16 fields are two chosen hours per week, not independent full-week meteorology or an external benchmark.',
            '- Original PhIRE pretrained-model WTK exposure remains unverified; locked regions were not evaluated.','']
    return '\n'.join(lines)+'\n'


def write_rows(path,cols,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=cols)
        w.writeheader();w.writerows(rows)


def seal(root,orig,aggs,dims,counts,zeros,fields,source):
    out=root/OUT;need(not out.exists(),f'Output exists; refuse overwrite: {out}')
    blocks=group(aggs,fields)
    rep=report(aggs,blocks)
    out.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='.'+out.name+'.staging-',dir=out.parent))
    try:
        write_rows(stage/OUTPUTS[0],orig.A_FIELDS,aggs)
        write_rows(stage/OUTPUTS[1],orig.D_FIELDS,dims)
        write_rows(stage/OUTPUTS[2],BLOCK_COLUMNS,blocks)
        (stage/OUTPUTS[3]).write_text(rep)
        meta={'format':'phase7-xcix-seasonal16-locality-v1','scope':'DEVELOPMENT_ONLY',
              'n_fields':N,'models':list(ARMS),'radii':[orig.rk(x) for x in orig.RADII],
              'n_aggregate_rows':len(aggs),'n_dimension_rows':len(dims),'n_region_week_rows':len(blocks),
              'fields':fields,'source_sha256':source,'coord_feature_counts':counts,
              'skipped_zero_or_nearzero_pairs_by_file':zeros,'filter_tolerance':1e-10,
              'matching':'unchanged Phase5E original sparse solver through original XCIV compute_rows',
              'no_locked_external_region':True}
        (stage/OUTPUTS[4]).write_text(json.dumps(meta,indent=2,sort_keys=True,allow_nan=False)+'\n')
        (stage/'SHA256SUMS').write_text(''.join(f'{sha(stage/f)}  {f}\n' for f in OUTPUTS))
        need(not out.exists(),'Output appeared during atomic staging')
        stage.rename(out)
    except BaseException:
        shutil.rmtree(stage,ignore_errors=True)
        raise
    print(f'[LOCALITY-GATE] PASS: {len(aggs)} aggregate + {len(dims)} H0/H1 rows; '
          f'48 region-week rows; 96 infinity-radius XCVIII anchors; 24 XCIV four-field anchors; out={out}',flush=True)


def recompute(root,orig,compat,solver,pilot,decoder,fields,augmented,scores,old_aggs,old_dims,progress):
    decoded,counts,zeros=readiness(root,orig,compat,solver,pilot,decoder,fields)
    aggs,dims=orig.compute_rows(solver,decoded,augmented,scores,progress=progress)
    need(len(aggs)==768 and len(dims)==1536,'Incorrect computed locality output row counts')
    compare_old4(orig,aggs,dims,old_aggs,old_dims)
    return aggs,dims,counts,zeros


def verify(root,orig,compat,solver,pilot,decoder,fields,augmented,scores,old_aggs,old_dims,source):
    out=root/OUT;need(out.is_dir(),f'XCIX output directory missing: {out}')
    # Original verified hash routine checks exact member set and each file's contents.
    numeric=load(root/NUMERIC,'xcix_verify_numeric')
    old=load(root/numeric.ORIGINAL_XCIII,'xcix_hash_verifier')
    old.verified_hashes(out,OUTPUTS)
    obj=json.loads((out/OUTPUTS[4]).read_text())
    need(obj['format']=='phase7-xcix-seasonal16-locality-v1' and obj['scope']=='DEVELOPMENT_ONLY' and
         obj['n_fields']==16 and obj['n_aggregate_rows']==768 and obj['n_dimension_rows']==1536 and
         obj['n_region_week_rows']==48 and obj['models']==list(ARMS) and
         obj['radii']==[orig.rk(r) for r in orig.RADII] and obj['fields']==fields and
         obj['source_sha256']==source and obj['filter_tolerance']==1e-10 and
         obj['no_locked_external_region'] is True,'XCIX manifest provenance drift')
    aggs,dims,counts,zeros=recompute(root,orig,compat,solver,pilot,decoder,fields,augmented,scores,old_aggs,old_dims,False)
    need(counts==obj['coord_feature_counts'] and zeros==obj['skipped_zero_or_nearzero_pairs_by_file'],
         'Coordinate/zero filtering inventory changed')
    for rel,columns,expected in ((OUTPUTS[0],orig.A_FIELDS,aggs),(OUTPUTS[1],orig.D_FIELDS,dims)):
        saved=read_rows(out/rel,columns)
        need(len(saved)==len(expected),f'{rel}: wrong number of rows')
        for i,(s,e) in enumerate(zip(saved,expected)):
            for col in columns:
                if isinstance(e[col],(int,float)) and not isinstance(e[col],bool):
                    near(s[col],e[col],f'{rel} row {i} col {col}',atol=1e-6)
                else:need(s[col]==str(e[col]),f'{rel} row {i} col {col} changed')
    blocks=group(aggs,fields)
    need(read_rows(out/OUTPUTS[2],BLOCK_COLUMNS)==blocks,'Region-week means differ from full recomputation')
    need((out/OUTPUTS[3]).read_text()==report(aggs,blocks),'Research report was changed')
    print('[VERIFY-GATE] PASS: all 112 coordinate/zero feature checks, 768+1536 locality values '
          'recomputed with original Phase5E, 96 XCVIII infinity anchors, XCIV regression, 5 SHA256 files; '
          'no external scores',flush=True)


def selftest():
    import numpy as np
    need(MAPPING=={0:0,1:4,2:8,3:12} and len(ARMS)==6 and len(OUTPUTS)==5,'Frozen schemas changed')
    need(len({(i,a,r) for i in range(N) for a in ARMS for r in (4,8,16,32,64,96,128,'inf')})==768,
         'Incorrect aggregate cardinality')
    near(0.0,0.0,'identity')
    try:near(1.0,2.0,'intentional mismatch');raise AssertionError('Fail-open near')
    except RuntimeError:pass
    p=np.asarray([[0.,0.],[0.,2.],[1.,1.+1e-12]],float)
    kept=p[(p[:,1]-p[:,0])>1e-10]
    need(kept.shape==(1,2),'Zero filtering selftest failed')
    print('[SELFTEST] PASS: 16×6×8 bookkeeping, original radius grid, regression mapping, zero-filter guard',flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode',required=True,choices=('selftest','preflight','compute','verify'))
    ap.add_argument('--repo-root',type=Path,default=Path('.'))
    args=ap.parse_args()
    if args.mode=='selftest':selftest();return
    root=args.repo_root.resolve()
    orig,compat,solver,pilot,decoder,fields,augmented,scores,old_aggs,old_dims,source=sources(root)
    out=root/OUT
    if args.mode!='verify':need(not out.exists(),f'XCIX result exists; verify rather than overwrite: {out}')
    if args.mode=='preflight':
        _,_,_=readiness(root,orig,compat,solver,pilot,decoder,fields)
        print('[PREFLIGHT] PASS: 112 coordinate-valid H0/H1 diagrams, original filter counts, all source scores '
              'and historical regression; zero output writes',flush=True)
    elif args.mode=='compute':
        aggs,dims,counts,zeros=recompute(root,orig,compat,solver,pilot,decoder,fields,augmented,scores,old_aggs,old_dims,True)
        seal(root,orig,aggs,dims,counts,zeros,fields,source)
    else:
        verify(root,orig,compat,solver,pilot,decoder,fields,augmented,scores,old_aggs,old_dims,source)

if __name__=='__main__':
    try:main()
    except Exception as exc:
        print(f'[XCIX] ERROR {type(exc).__name__}: {exc}',file=sys.stderr,flush=True)
        sys.exit(1)
