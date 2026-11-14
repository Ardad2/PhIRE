#!/usr/bin/env python3
"""Part CIV: staged 168-hour development-only canonical spatial locality.

Uses unchanged SHA-pinned XCIV compute_rows + original Phase5E SparsePrepared,
with original XCIV zero-persistence compatibility semantics (as in XCIX).
Requires CIII numerical distances and XCIX hour-0/84 numerical regressions.
No locked regions, models, new distance algorithms, TTK/TF, or HSDS.
"""
from __future__ import annotations
import argparse, csv, fcntl, hashlib, importlib.util, json, math, os, re, shutil, sys, tempfile
from datetime import datetime, timedelta
from pathlib import Path
from statistics import mean

BASE=Path('ttk_runs_fixed/topology_finetuning')
CII=BASE/'phase7_cii_dev_gulf_inland_feb2010_pd168_20261018'
CIII=BASE/'phase7_ciii_dev_gulf_inland_feb2010_corrected_pd168_20261019'
XCIX=BASE/'phase7_xcix_dev_seasonal16_locality_20261018'
OUT=BASE/'phase7_civ_dev_gulf_inland_feb2010_locality168_20261019'
SELF=Path('phase7_civ_update/scripts/phase7_civ_dev_fullweek_locality.py')
CIII_SCRIPT=Path('phase7_ciii_update/scripts/phase7_ciii_dev_fullweek_numeric_pd.py')
XCIX_SCRIPT=Path('phase7_xcix_update/scripts/phase7_xcix_dev_seasonal16_locality.py')
PINS={'ciii':'59518de2eed0c5ff24e00a5e75c9889226bd1d456f252d6be7449f72afbce632',
      'xcix':'5540c485f7ffa32449dba1e9c5747694b64d089b2021d8f3ec6d70724ce87f52'}
N=168
ARMS=('cnn','gan','grad','cv','pers','f1')
ALL=('gt',*ARMS)
ANCHORS={0:0,84:1}
RADIUS_LABELS=('4.0','8.0','16.0','32.0','64.0','96.0','128.0','inf')
SCHEMA='phase7-civ-gulf-inland-feb2010-locality168-v1'
DAILY_COLS=('day_start_utc','arm','n_hours','r4_w22','r16_w22','r32_w22','rinf_w22','r16_gt_persistence_coverage')
OUTPUTS=('locality_aggregate.csv','locality_dimension.csv','daily_locality.csv','locality_report.md','locality_manifest.json')


def need(ok,msg):
    if not ok:raise RuntimeError(msg)

def sha(p):
    p=Path(p);need(p.is_file() and p.stat().st_size>0,f'Missing/empty file: {p}')
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def stable(x):return json.dumps(x,sort_keys=True,allow_nan=False,separators=(',',':'))
def jd(x):return json.dumps(x,sort_keys=True,indent=2,allow_nan=False)+'\n'
def txtsha(x):return hashlib.sha256(x.encode()).hexdigest()

def atomic_text(p,s):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix='.civ-atomic-',dir=p.parent)
    try:
        with os.fdopen(fd,'w') as f:f.write(s);f.flush();os.fsync(f.fileno())
        os.replace(tmp,p)
    finally:Path(tmp).unlink(missing_ok=True)

def load(p,name):
    spec=importlib.util.spec_from_file_location(name,p)
    need(spec and spec.loader,f'Cannot import source {p}')
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module

def read_csv(p,cols):
    with Path(p).open(newline='') as f:
        r=csv.DictReader(f);need(tuple(r.fieldnames or ())==tuple(cols),f'CSV columns drifted: {p}')
        return list(r)

def write_csv(p,rows,cols):
    with Path(p).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=cols);writer.writeheader();writer.writerows(rows)

def anchored(a,b,label,atol=2e-6):
    fa,fb=float(a),float(b)
    need(math.isfinite(fa) and math.isfinite(fb) and math.isclose(fa,fb,abs_tol=atol,rel_tol=1e-9),
         f'{label} mismatch: {fa} != {fb}')

def source_gate(root):
    for name,p in [('ciii',CIII_SCRIPT),('xcix',XCIX_SCRIPT)]:
        need(sha(root/p)==PINS[name],f'Archived {name} driver SHA mismatch')
    ciii=load(root/CIII_SCRIPT,'civ_pinned_ciii')
    xcix=load(root/XCIX_SCRIPT,'civ_pinned_xcix')
    pilot,decoder,bypair,lookups,line=ciii.source_gate(root)
    cfg3=ciii.config(line)
    ciii.verify(root,pilot,decoder,bypair,lookups,cfg3)
    old=xcix.sources(root)
    orig,compat,solver,*_=old
    need(sha(root/orig.SELF)==xcix.ORIG_SHA and sha(root/xcix.COMPAT)==xcix.COMPAT_SHA,
         'Archived coordinate reader or zero-persistence compatibility changed')
    # Require verified, sealed XCIX output; 0h and 84h must be those original development fields.
    verified=load(root/xcix.NUMERIC,'civ_old_numeric')
    verifier=load(root/verified.ORIGINAL_XCIII,'civ_old_sha_verifier')
    verifier.verified_hashes(root/XCIX,xcix.OUTPUTS)
    oldmanifest=json.loads((root/XCIX/'locality_manifest.json').read_text())
    need(oldmanifest['format']=='phase7-xcix-seasonal16-locality-v1' and
         oldmanifest['n_fields']==16 and oldmanifest['n_aggregate_rows']==768 and
         oldmanifest['n_dimension_rows']==1536 and oldmanifest['no_locked_external_region'] is True,
         'XCIX source manifest/schema or scope drift')
    olda=read_csv(root/XCIX/'locality_aggregate.csv',orig.A_FIELDS)
    oldd=read_csv(root/XCIX/'locality_dimension.csv',orig.D_FIELDS)
    oldagg={(int(r['sample']),r['arm'],r['radius']):r for r in olda}
    olddim={(int(r['sample']),r['arm'],r['radius'],int(r['dimension'])):r for r in oldd}
    need(len(oldagg)==768 and len(olddim)==1536,'Duplicate archived XCIX keys')
    rows=read_csv(root/CIII/'per_hour_corrected_pd.csv',ciii.COLS)
    need(len(rows)==1008,'CIII requires 1008 numeric source rows')
    score={(int(r['hour']),r['arm']):r for r in rows}
    need(len(score)==1008 and all(set(score[h,a]['region'] for a in ARMS)=={'gulf_inland'} for h in range(N)),
         'Duplicate or out-of-region CIII scores')
    for h,i in ANCHORS.items():
        for a in ARMS:
            anchored(score[h,a]['W22'],old[7][(i,a)]['W22'],f'CIII/XCVIII global anchor {h} {a}',atol=1e-7)
    need([orig.rk(v) for v in orig.RADII]==list(RADIUS_LABELS) and
         tuple(solver.RADII)==orig.RADII and solver.TOL==1e-10 and orig.SOLVER_SHA==xcix.SOLVER_SHA,
         'Frozen solver radii/tolerance/source changed')
    lineage={'CIV_script_sha256':sha(root/SELF),
             'CIII_script_sha256':PINS['ciii'],'XCIX_script_sha256':PINS['xcix'],
             'CIII_SHA256SUMS_sha256':sha(root/CIII/'SHA256SUMS'),
             'CIII_manifest_sha256':sha(root/CIII/'numeric_manifest.json'),
             'CII_SHA256SUMS_sha256':sha(root/CII/'SHA256SUMS'),
             'CII_manifest_sha256':sha(root/CII/'extraction_manifest.json'),
             'XCIX_SHA256SUMS_sha256':sha(root/XCIX/'SHA256SUMS'),
             'XCIX_manifest_sha256':sha(root/XCIX/'locality_manifest.json'),
             'Phase5E_solver_sha256':xcix.SOLVER_SHA,
             'XCIV_driver_sha256':xcix.ORIG_SHA,'XCIV_zero_compat_sha256':xcix.COMPAT_SHA}
    print('[SOURCES] PASS: CII 2354 / CIII 4 / XCIX 5 checksums, historical GUDHI and XCIV/Phase5E provenance',flush=True)
    print('[SCOPE] PASS: one Gulf inland February 2010 development week; original six models, eight grid-cell radii; no locked test',flush=True)
    return ciii,xcix,orig,compat,solver,pilot,decoder,bypair,score,oldagg,olddim,lineage

def config(lineage):
    return {'schema':SCHEMA,'scope':'DEVELOPMENT_ONLY','region':'gulf_inland',
            'start_utc':'2010-02-01T00:00:00','n_hours':168,
            'wtk_start_index':27048,'wtk_last_index':27215,
            'models':list(ARMS),'radii':list(RADIUS_LABELS),'anchor_hours':{'0':0,'84':1},
            'zero_tolerance':1e-10,'primary_crop':{'x0':0,'y0':0,'height':160,'width':160},
            'lineage_sha256':lineage,'stage_policy':'atomic complete six-arm hour',
            'locked_test_access':False}

def expected_utc(h):return (datetime(2010,2,1)+timedelta(hours=h)).isoformat()

def pd_path(root,h,a):return root/CII/'pd'/f'dev_{a}_s{h}_speed_p160_x0_y0_pd_port_0.vtu'

def anchor_rows(h,aggs,dims,oldagg,olddim,orig):
    if h not in ANCHORS:return
    i=ANCHORS[h]
    need(len(aggs)==48 and len(dims)==96,'Anchor row cardinality changed')
    for r in aggs:
        older=oldagg[(i,r['arm'],r['radius'])]
        for k in orig.A_FIELDS:
            if k in ('sample','preview'):continue
            if k in ('region','arm','radius'):need(str(r[k])==str(older[k]),f'XCIX anchor metadata {h} {k}')
            else:anchored(r[k],older[k],f'XCIX aggregate hour {h} {r["arm"]} r{r["radius"]} {k}')
    for r in dims:
        older=olddim[(i,r['arm'],r['radius'],int(r['dimension']))]
        for k in orig.D_FIELDS:
            if k=='sample':continue
            if k in ('region','arm','dimension','radius'):
                need(str(r[k])==str(older[k]),f'XCIX dimension anchor metadata {h} {k}')
            else:anchored(r[k],older[k],f'XCIX dimension hour {h} {r["arm"]} {k}')


def one_hour(root,h,orig,compat,solver,pilot,decoder,score,oldagg,olddim):
    import numpy as np
    need(0<=h<N,'Hour outside frozen scope')
    decoded={};counts={};zeros={}
    for a in ALL:
        path=pd_path(root,h,a)
        feat,skipped=solver.read_features(path)
        expected=pilot.parse(decoder.read_pd,path)
        filtered={};excluded={}
        for dim in (0,1):
            kept,num=compat.filtered_pairs(expected[dim],float(solver.TOL),np)
            need(int(skipped[dim])==num,f'Zero-persistence reader/parser mismatch h{h} {a} H{dim}')
            filtered[dim]=kept;excluded[str(dim)]=num
        matched=orig.pairs(feat,filtered,path)
        decoded[(0,a)]=feat
        counts[a]={str(d):int(matched[d]) for d in (0,1)}
        zeros[a]=excluded
        for m in ARMS:
            src=score[h,m]
            if a=='gt':
                need(len(expected[0])==int(src['D0_gt']) and len(expected[1])==int(src['D1_gt']),
                     f'CIII GT canonical count changed h{h} {m}')
            elif a==m:
                need(len(expected[0])==int(src['D0_sr']) and len(expected[1])==int(src['D1_sr']),
                     f'CIII model canonical count changed h{h} {a}')
    # The unchanged original XCIV compute_rows only expects local sample=0.
    prev_n=orig.N
    try:
        orig.N=1
        group=[{'region':'gulf_inland','name':f'dev2010_gulf_inland_20100201_h{h:03d}'}]
        metrics={(0,a):score[h,a] for a in ARMS}
        aggs,dims=orig.compute_rows(solver,decoded,group,metrics,progress=False)
    finally:orig.N=prev_n
    need(len(aggs)==48 and len(dims)==96,'Incorrect per-hour locality row count')
    for r in aggs:r['sample']=h
    for r in dims:r['sample']=h
    anchor_rows(h,aggs,dims,oldagg,olddim,orig)
    need(all(r['region']=='gulf_inland' for r in [*aggs,*dims]),'Nondevelopment region emitted')
    return {'hour':h,'utc':expected_utc(h),'wtk_index':27048+h,
            'aggregate':aggs,'dimension':dims,'positive_feature_counts':counts,'zero_pair_counts':zeros}

def validate_hour(payload,h,orig,score,oldagg,olddim):
    need(set(payload)=={'hour','utc','wtk_index','aggregate','dimension','positive_feature_counts','zero_pair_counts'},
         'Unexpected/missing hour payload fields')
    need(payload['hour']==h and payload['utc']==expected_utc(h) and payload['wtk_index']==27048+h,
         f'Hour ordering/identity drift {h}')
    aggs=payload['aggregate'];dims=payload['dimension']
    need(len(aggs)==48 and len(dims)==96,f'Bad locality cardinality h{h}')
    expected_a=[(a,r) for a in ARMS for r in RADIUS_LABELS]
    expected_d=[(a,d,r) for a in ARMS for d in (0,1) for r in RADIUS_LABELS]
    need([(x['arm'],x['radius']) for x in aggs]==expected_a,
         f'Aggregate arm/radius order drift h{h}')
    need([(x['arm'],x['dimension'],x['radius']) for x in dims]==expected_d,
         f'Dimension arm/H/radius order drift h{h}')
    for x in aggs:
        need(set(x)==set(orig.A_FIELDS) and x['sample']==h and x['region']=='gulf_inland' and
             x['preview']==f'dev2010_gulf_inland_20100201_h{h:03d}',f'Aggregate source drift h{h}')
        actual=float(x['constrained_w22_all']);base=float(score[h,x['arm']]['W22'])
        need(math.isfinite(actual) and actual>=0 and actual+1e-7>=base,'Bad constrained W22')
        anchored(x['unconstrained_w22_all'],base,f'CIII score drift h{h}',atol=1e-7)
        if x['radius']=='inf':anchored(actual,base,f'CIII infinity radius h{h}',atol=1e-6)
        for cov in ('gt_persistence_coverage_all','sr_persistence_coverage_all'):
            need(-1e-9<=float(x[cov])<=1+1e-9,f'Invalid persistence coverage h{h}')
    for x in dims:
        need(set(x)==set(orig.D_FIELDS) and x['sample']==h and x['region']=='gulf_inland',
             f'Dimension source drift h{h}')
        need(math.isfinite(float(x['w22'])) and float(x['w22'])>=0,f'Invalid dimension cost h{h}')
    need(set(payload['positive_feature_counts'])==set(ALL) and
         set(payload['zero_pair_counts'])==set(ALL) and
         all(set(e)=={'0','1'} and all(isinstance(v,int) and v>=0 for v in e.values())
             for pack in ('positive_feature_counts','zero_pair_counts')
             for e in payload[pack].values()), 'Coordinate feature/zero-pair schema drift')
    anchor_rows(h,aggs,dims,oldagg,olddim,orig)

def stage_rows(stage,cfg,orig,score,oldagg,olddim):
    need((stage/'run_config.json').is_file() and
         json.loads((stage/'run_config.json').read_text())==cfg,'Stage configuration/source lineage changed')
    d=stage/'hours';need(d.is_dir(),'Stage hours directory missing')
    seen={}
    for p in d.iterdir():
        need(p.is_file() and re.fullmatch(r'h\d{3}\.json',p.name),f'Unexpected stage file {p}')
        h=int(p.stem[1:]);need(0<=h<N and h not in seen,f'Invalid/duplicate staged hour {h}')
        content=json.loads(p.read_text())
        need(set(content)=={'hour','payload','payload_sha256'} and content['hour']==h and
             txtsha(stable(content['payload']))==content['payload_sha256'],
             f'Corrupt staged hour checksum h{h}')
        validate_hour(content['payload'],h,orig,score,oldagg,olddim)
        seen[h]=content['payload']
    need(set(seen)==set(range(len(seen))),'Staged hours are not contiguous from zero')
    return seen

def stage_hour(stage,h,payload):
    p=stage/'hours'/f'h{h:03d}.json';need(not p.exists(),f'Hour {h} already exists')
    atomic_text(p,jd({'hour':h,'payload':payload,'payload_sha256':txtsha(stable(payload))}))

def daily_stats(aggs):
    lookup={(int(x['sample']),x['arm'],x['radius']):x for x in aggs}
    need(len(lookup)==8064,'Duplicate/missing 168-hour locality aggregate keys')
    out=[];hourly={}
    for d in range(7):
        for a in ARMS:
            hrs=range(d*24,(d+1)*24)
            def avg(r,k):return mean(float(lookup[h,a,r][k]) for h in hrs)
            out.append({'day_start_utc':f'2010-02-0{d+1}T00:00:00',
                        'arm':a,'n_hours':'24',
                        'r4_w22':format(avg('4.0','constrained_w22_all'),'.17g'),
                        'r16_w22':format(avg('16.0','constrained_w22_all'),'.17g'),
                        'r32_w22':format(avg('32.0','constrained_w22_all'),'.17g'),
                        'rinf_w22':format(avg('inf','constrained_w22_all'),'.17g'),
                        'r16_gt_persistence_coverage':format(avg('16.0','gt_persistence_coverage_all'),'.17g')})
    for a in ARMS:
        hourly[a]={}
        for r,label in (('4.0','r4_w22'),('16.0','r16_w22'),('32.0','r32_w22'),('inf','rinf_w22')):
            hourly[a][label]=mean(float(lookup[h,a,r]['constrained_w22_all']) for h in range(N))
        hourly[a]['r16_gt_persistence_coverage']=mean(float(lookup[h,a,'16.0']['gt_persistence_coverage_all']) for h in range(N))
    return out,hourly

def make_report(days,means):
    s=['# Phase 7 Part CIV — 168-hour development spatial-locality analysis','',
       'Gulf inland, February 1–7, 2010 UTC; 168 contiguous hourly fields × six frozen models.',
       '**One development region-week. Consecutive hours/days are temporally correlated, not independent generalization replications.**','',
       'Unchanged original Phase5E sparse matching on TTK canonical top-left 160×160 speed persistence diagrams.',
       'Radii **4, 8, 16, 32, 64, 96, 128, ∞ are WTK grid cells, not kilometres**.','',
       'Lower constrained W₂,₂ means better persistence matching; GT persistence-mass coverage is separate (higher is better).','',
       '| Model | r4 W22 ↓ | r16 W22 ↓ | r32 W22 ↓ | r∞ W22 ↓ | GT coverage r16 ↑ |',
       '|---|---:|---:|---:|---:|---:|']
    for a in ARMS:
        x=means[a]
        s.append(f'| {a} | {x["r4_w22"]:.6f} | {x["r16_w22"]:.6f} | {x["r32_w22"]:.6f} | '
                 f'{x["rinf_w22"]:.6f} | {x["r16_gt_persistence_coverage"]:.6f} |')
    s+=['','## Daily descriptive summaries (24 consecutive hours per UTC date)','',
        '| UTC day | Arm | N | r4 W22 ↓ | r16 W22 ↓ | r∞ W22 ↓ | GT coverage r16 ↑ |',
        '|---|---|---:|---:|---:|---:|---:|']
    for d in days:
        s.append(f'| {d["day_start_utc"][:10]} | {d["arm"]} | 24 | '
                 f'{float(d["r4_w22"]):.6f} | {float(d["r16_w22"]):.6f} | '
                 f'{float(d["rinf_w22"]):.6f} | {float(d["r16_gt_persistence_coverage"]):.6f} |')
    s+=['','## Scientific and reproducibility boundaries','',
        '- All 1,008 infinite-radius W22 model-to-GT values must match the sealed Part CIII corrected GUDHI output.',
        '- All 96 aggregate + 192 dimension rows for historical week hours 0 and 84 must reproduce Part XCIX.',
        '- Canonical zero/near-zero persistence pairs use the frozen original Phase5E tolerance 1e-10 with excluded count validation.',
        '- All 1,176 diagram coordinate sets are validated against their corrected numerical birth/death features.',
        '- GT persistence-mass coverage measures matched mass, not correct spatial positions or physical wind structure.',
        '- This is not topology on full 500×500 fields and does not include merge trees or training.',
        '- No external locked-region predictions or new hyperparameter/model selection. Original pretrained WTK exposure unknown.',
        '- Verifier repeats the same original Phase5E algorithm; it is not an independently implemented solver.','']
    return '\n'.join(s)+'\n'

def checkhashes(out):
    records={}
    for s in (out/'SHA256SUMS').read_text().splitlines():
        m=re.fullmatch(r'([0-9a-f]{64})  ([^\n]+)',s)
        need(m is not None,f'Invalid checksum line {s}')
        digest,name=m.groups();need(name in OUTPUTS and name not in records,'Extra/duplicate output SHA256 entry')
        need(sha(out/name)==digest,f'Output hash mismatch {name}')
        records[name]=digest
    need(set(records)==set(OUTPUTS),'Missing output checksums')

def publish(root,stage,cfg,orig,score,oldagg,olddim):
    out=root/OUT;need(not out.exists(),'Immutable completed CIV output exists')
    hrs=stage_rows(stage,cfg,orig,score,oldagg,olddim)
    need(len(hrs)==N,'Cannot publish incomplete region-week')
    aggs=[r for h in range(N) for r in hrs[h]['aggregate']]
    dims=[r for h in range(N) for r in hrs[h]['dimension']]
    need(len(aggs)==8064 and len(dims)==16128,'Incorrect final row totals')
    days,means=daily_stats(aggs)
    out.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.civ-publish-',dir=out.parent) as temp:
        tmp=Path(temp)
        write_csv(tmp/OUTPUTS[0],aggs,orig.A_FIELDS)
        write_csv(tmp/OUTPUTS[1],dims,orig.D_FIELDS)
        write_csv(tmp/OUTPUTS[2],days,DAILY_COLS)
        (tmp/OUTPUTS[3]).write_text(make_report(days,means))
        mani={'schema':SCHEMA,'run_config':cfg,'n_hours':N,'n_diagrams':1176,
              'n_comparisons':1008,'n_aggregate_rows':8064,'n_dimension_rows':16128,
              'n_daily_rows':42,'n_legacy_anchor_aggregate_rows':96,'n_legacy_anchor_dimension_rows':192,
              'n_ciii_infinity_anchors':1008,'mean_hourly':means,
              'stage_hour_sha256':{f'h{h:03d}':sha(stage/'hours'/f'h{h:03d}.json') for h in range(N)},
              'positive_feature_counts':{f'h{h:03d}':hrs[h]['positive_feature_counts'] for h in range(N)},
              'excluded_zero_pairs':{f'h{h:03d}':hrs[h]['zero_pair_counts'] for h in range(N)},
              'no_locked_external_test':True,'verification':'original pinned Phase5E; independent same-method recomputation'}
        (tmp/OUTPUTS[4]).write_text(jd(mani))
        (tmp/'SHA256SUMS').write_text(''.join(f'{sha(tmp/f)}  {f}\n' for f in OUTPUTS))
        checkhashes(tmp)
        need(not out.exists(),'Output collision before immutable publication')
        tmp.rename(out)
    print(f'[LOCALITY-GATE] PASS: 168 hours, 1008 comparisons, 8064 aggregate + 16128 H0/H1 rows, '
          f'1008 CIII infinity anchors, 96 XCIX anchor rows, 42 daily summaries; published={out}',flush=True)

def verify(root,cfg,orig,compat,solver,pilot,decoder,score,oldagg,olddim):
    out=root/OUT;need(out.is_dir(),'No completed CIV output to verify')
    checkhashes(out)
    obj=json.loads((out/OUTPUTS[4]).read_text())
    need(obj['schema']==SCHEMA and obj['run_config']==cfg and obj['n_hours']==N and
         obj['n_diagrams']==1176 and obj['n_comparisons']==1008 and
         obj['n_aggregate_rows']==8064 and obj['n_dimension_rows']==16128 and
         obj['n_daily_rows']==42 and obj['n_legacy_anchor_aggregate_rows']==96 and
         obj['n_legacy_anchor_dimension_rows']==192 and obj['n_ciii_infinity_anchors']==1008 and
         obj['no_locked_external_test'] is True,'CIV manifest schema/lineage differs')
    saved_a=read_csv(out/OUTPUTS[0],orig.A_FIELDS)
    saved_d=read_csv(out/OUTPUTS[1],orig.D_FIELDS)
    need(len(saved_a)==8064 and len(saved_d)==16128,'CIV CSV row counts differ')
    aa=[];dd=[];counts={};zeros={}
    for h in range(N):
        x=one_hour(root,h,orig,compat,solver,pilot,decoder,score,oldagg,olddim)
        validate_hour(x,h,orig,score,oldagg,olddim)
        counts[f'h{h:03d}']=x['positive_feature_counts'];zeros[f'h{h:03d}']=x['zero_pair_counts']
        for saved,new,fields in ((saved_a[h*48:(h+1)*48],x['aggregate'],orig.A_FIELDS),
                                 (saved_d[h*96:(h+1)*96],x['dimension'],orig.D_FIELDS)):
            need(len(saved)==len(new),'Incomplete local source rows')
            for old,actual in zip(saved,new):
                for k in fields:
                    if isinstance(actual[k],(float,int)) and k not in ('sample','dimension','real_real_all','gt_to_diagonal_all','diagonal_to_sr_all','real_real','gt_to_diagonal','diagonal_to_sr'):
                        anchored(old[k],actual[k],f'Independent numerical recomputation h{h} {k}',atol=1e-6)
                    else:need(str(old[k])==str(actual[k]),f'CIV data changed hour {h} {k}')
        aa.extend(x['aggregate']);dd.extend(x['dimension'])
        if h%24==23:print(f'[VERIFY-HOURS] PASS {(h+1):03d}/168 complete hours',flush=True)
    need(obj['positive_feature_counts']==counts and obj['excluded_zero_pairs']==zeros,
         'Coordinate count/zero-filter inventories differ')
    days,means=daily_stats(aa)
    need(read_csv(out/OUTPUTS[2],DAILY_COLS)==days and obj['mean_hourly']==means and
         (out/OUTPUTS[3]).read_text()==make_report(days,means),
         'Daily summaries, mean values, or report modified')
    stage=out.parent/('.'+out.name+'.staging')
    if stage.is_dir():
        earlier=stage_rows(stage,cfg,orig,score,oldagg,olddim)
        need(len(earlier)==N and obj['stage_hour_sha256']=={
             f'h{h:03d}':sha(stage/'hours'/f'h{h:03d}.json') for h in range(N)},
             'CIV published stage digests changed')
    print('[VERIFY-GATE] PASS: all 1176 numeric H0/H1 coordinate sets, 8064+16128 unchanged Phase5E rows '
          'independently recomputed, 1008 CIII infinity anchors, XCIX 0/84 regression, 5 SHA256 files; '
          'no locked regions',flush=True)

def selftest():
    need(len(ARMS)==6 and len(ALL)==7 and N*len(ARMS)*8==8064 and
         N*len(ARMS)*8*2==16128 and len(DAILY_COLS)==8 and len(OUTPUTS)==5,
         'Frozen matrix cardinalities changed')
    need(list(ANCHORS)==[0,84] and len(RADIUS_LABELS)==8,'Frozen anchors/radii changed')
    anchored(0,0,'identity')
    try:anchored(1,2,'negative control');raise AssertionError('Mismatch silently accepted')
    except RuntimeError:pass
    payload={'hour':0,'aggregate':[{'sample':0}], 'dimension':[]}
    payload_digest=txtsha(stable(payload));payload['aggregate'][0]['sample']=1
    need(txtsha(stable(payload))!=payload_digest,'Stage checksum tamper rejection failed')
    print('[SELFTEST] PASS: 168×6×8 locality bookkeeping, original radius grid, two anchors, stage checksum guard',flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',required=True,choices=('selftest','preflight','compute','verify'))
    p.add_argument('--repo-root',type=Path,default=Path.cwd())
    p.add_argument('--max-new-hours',type=int)
    a=p.parse_args()
    if a.mode=='selftest':selftest();return
    need(a.max_new_hours is None or (a.mode=='compute' and a.max_new_hours>=1),
         '--max-new-hours is positive and only valid for compute')
    root=a.repo_root.resolve();os.chdir(root)
    ciii,xcix,orig,compat,solver,pilot,decoder,bypair,score,oldagg,olddim,lineage=source_gate(root)
    cfg=config(lineage);out=root/OUT;stage=out.parent/('.'+out.name+'.staging')
    if a.mode=='verify':verify(root,cfg,orig,compat,solver,pilot,decoder,score,oldagg,olddim);return
    need(not out.exists(),'Immutable CIV output exists; verify instead of overwrite')
    if a.mode=='preflight':
        if stage.is_dir():
            previous=stage_rows(stage,cfg,orig,score,oldagg,olddim)
            print(f'[PREFLIGHT] PASS: valid resumable stage {len(previous)}/168; frozen lineage, no output writes',flush=True)
        else:
            for h in ANCHORS:
                check=one_hour(root,h,orig,compat,solver,pilot,decoder,score,oldagg,olddim)
                validate_hour(check,h,orig,score,oldagg,olddim)
                print(f'[ANCHOR] PASS h{h:03d} 48 aggregate + 96 dimension rows equal XCIX',flush=True)
            print('[PREFLIGHT] PASS: source hashes and original two XCIX locality anchors; no staging files written',flush=True)
        return
    out.parent.mkdir(parents=True,exist_ok=True)
    lockfile=out.parent/('.'+out.name+'.lock')
    with lockfile.open('a+') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        need(not out.exists(),'Immutable CIV result appeared while locking')
        if not stage.exists():
            stage.mkdir();(stage/'hours').mkdir();atomic_text(stage/'run_config.json',jd(cfg))
        hrs=stage_rows(stage,cfg,orig,score,oldagg,olddim)
        print(f'[RESUME] PASS: {len(hrs)}/168 sealed locality hours',flush=True)
        produced=0
        for h in range(N):
            if h in hrs:continue
            if a.max_new_hours is not None and produced>=a.max_new_hours:
                print(f'[PARTIAL-GATE] PASS: {len(hrs)}/168 sealed hours; resume same --mode compute',flush=True)
                return
            payload=one_hour(root,h,orig,compat,solver,pilot,decoder,score,oldagg,olddim)
            validate_hour(payload,h,orig,score,oldagg,olddim)
            stage_hour(stage,h,payload);hrs[h]=payload;produced+=1
            r4=next(x for x in payload['aggregate'] if x['arm']=='grad' and x['radius']=='4.0')
            r16=next(x for x in payload['aggregate'] if x['arm']=='grad' and x['radius']=='16.0')
            print(f'[HOUR-LOCALITY] {h+1:03d}/168 six arms, eight radii PASS '
                  f'grad_r4={float(r4["constrained_w22_all"]):.6f} '
                  f'grad_r16={float(r16["constrained_w22_all"]):.6f}',flush=True)
        publish(root,stage,cfg,orig,score,oldagg,olddim)

if __name__=='__main__':
    try:main()
    except Exception as exc:
        print(f'[CIV] ERROR {type(exc).__name__}: {exc}',file=sys.stderr,flush=True)
        sys.exit(1)
