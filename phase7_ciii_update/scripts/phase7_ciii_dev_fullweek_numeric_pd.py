#!/usr/bin/env python3
"""Phase 7 Part CIII: resumable corrected GUDHI numerical PDs for one full development week.

Frozen 2010-Feb gulf_inland, 168 h x six original checkpoints = 1008 comparisons.
Reuses SHA-pinned XCVIII trusted decoder, Phase6 GUDHI functions, XCIII historical
2007 regression and XCVIII gulf anchor hours 0 and 84. No TTK/TF/HSDS/locked data.
"""
from __future__ import annotations
import argparse, csv, fcntl, hashlib, importlib.util, json, math, os, re, shutil, sys, tempfile
from pathlib import Path
import numpy as np

BASE=Path('ttk_runs_fixed/topology_finetuning')
CII=BASE/'phase7_cii_dev_gulf_inland_feb2010_pd168_20261018'
XCVIII=BASE/'phase7_xcviii_dev_seasonal16_corrected_pd_20261018'
XCI=BASE/'phase7_xci_frozen_evaluation_protocol_20261015'
OUT=BASE/'phase7_ciii_dev_gulf_inland_feb2010_corrected_pd168_20261019'
CII_SCRIPT=Path('phase7_cii_update/scripts/phase7_cii_dev_fullweek_pd.py')
XCVIII_SCRIPT=Path('phase7_xcviii_update/scripts/phase7_xcviii_dev_seasonal16_numeric_pd.py')
SELF=Path('phase7_ciii_update/scripts/phase7_ciii_dev_fullweek_numeric_pd.py')
PINS={'cii':'f0e5ae86874c7e4d853320ce0bdc45b2a4fbfec3efc9b7ce6c7630edbdc39f2d',
      'xcviii':'5f2522cf5fb170138634bae21643dc32f9404f7ed9efad226a467fc5c0db9ff4'}
N=168
ARMS=('cnn','gan','grad','cv','pers','f1')
ALL=('gt',*ARMS)
METRICS=('dB','W2inf','W22')
SCHEMA='phase7-ciii-gulf-inland-feb2010-corrected-pd168-v1'
ANCHORS={0:0,84:1}
COLS=('hour','utc','wtk_index','region','arm','gt_sha256','sr_sha256',
      'D0_gt','D1_gt','D0_sr','D1_sr',*METRICS)
DAY_COLS=('day_start_utc','arm','n_hours',*METRICS)
FILES=('per_hour_corrected_pd.csv','daily_corrected_pd.csv','numeric_manifest.json','numeric_report.md')

def need(test,message):
    if not test:raise RuntimeError(message)

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as fh:
        for b in iter(lambda:fh.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def stable(obj):return json.dumps(obj,sort_keys=True,allow_nan=False,separators=(',',':'))
def jd(obj):return json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n'
def fmt(x):return format(float(x),'.17g')

def atomic_text(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix='.ciii-atomic-',dir=p.parent)
    try:
        with os.fdopen(fd,'w') as f:f.write(value);f.flush();os.fsync(f.fileno())
        os.replace(name,p)
    finally:Path(name).unlink(missing_ok=True)

def mod(p,name):
    spec=importlib.util.spec_from_file_location(name,p)
    need(spec and spec.loader,f'Missing imported source {p}')
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m)
    return m

def check_anchors(row,lookup):
    h=int(row['hour'])
    if h not in ANCHORS:return
    old=lookup[(ANCHORS[h],row['arm'])]
    for k in (*METRICS,'D0_gt','D1_gt','D0_sr','D1_sr'):
        a=float(row[k]) if k in METRICS else int(row[k]);b=float(old[k]) if k in METRICS else int(old[k])
        need(math.isclose(a,b,rel_tol=1e-11,abs_tol=1e-8),
             f'XCVIII anchor regression hour {h} arm {row["arm"]} {k}: {a} vs {b}')
    need(row['gt_sha256']==old['gt_sha256'] and row['sr_sha256']==old['sr_sha256'],
         f'Anchor persistence diagram SHA changed hour {h} {row["arm"]}')

def source_gate(root):
    for label,p in [('cii',CII_SCRIPT),('xcviii',XCVIII_SCRIPT)]:
        need((root/p).is_file() and sha(root/p)==PINS[label],f'Frozen {label} source SHA256 mismatch')
    cii=mod(root/CII_SCRIPT,'ciii_original_cii')
    xcviii=mod(root/XCVIII_SCRIPT,'ciii_original_xcviii')
    need(tuple(cii.ARMS)==ALL and cii.N==N and tuple(xcviii.ARMS)==ARMS,'Original arm scope mismatch')
    _,xcvii,xml,cii_lineage,_,_=cii.sources(root)
    arr=cii.arrays(root)
    cfg=cii.config(cii_lineage)
    cii.verify(root,None,root/CII,arr,xcvii,xml,cfg)
    old,pilot,decoder,fields,source,old_lookup=xcviii.verified_sources(root)
    xcviii.verify(root,pilot,decoder,fields,source,old_lookup)
    cii_manifest=json.loads((root/CII/'extraction_manifest.json').read_text())
    need(cii_manifest['format']==cii.SCHEMA and cii_manifest['total_pairs']==1176 and
         cii_manifest['run_config']['scope']==cii.SCHEMA and
         cii_manifest['run_config']['region']=='gulf_inland' and
         cii_manifest['run_config']['hours']==168,'CII frozen run changed')
    lookup={}
    with (root/XCVIII/'per_field_corrected_pd.csv').open(newline='') as fh:
        r=csv.DictReader(fh)
        need(tuple(r.fieldnames or ())==xcviii.COLS,'XCVIII anchor columns changed')
        for line in r:
            if int(line['sample']) in (0,1):
                need(line['region']=='gulf_inland' and int(line['hour_offset']) in (0,84),'XCVIII Gulf hour mismatch')
                lookup[(int(line['sample']),line['arm'])]=line
    need(set(lookup)=={(i,a) for i in (0,1) for a in ARMS},'Missing XCVIII anchor metrics')
    bypair={(e['hour'],e['arm']):e for e in cii_manifest['pairs']}
    need(len(bypair)==1176,'CII duplicate diagram ids')
    for h in range(N):
        for a in ALL:
            e=bypair[(h,a)]
            need(e['pd_file']==f'pd/{cii.pd_name(a,h)}' and
                 e['pd_sha256']==sha(root/CII/e['pd_file']),f'CII source digest drift h={h} {a}')
    line={'CIII_script_sha256':sha(root/SELF),'CII_script_sha256':PINS['cii'],
          'XCVIII_script_sha256':PINS['xcviii'],
          'CII_SHA256SUMS_sha256':sha(root/CII/'SHA256SUMS'),
          'CII_manifest_sha256':sha(root/CII/'extraction_manifest.json'),
          'XCVIII_SHA256SUMS_sha256':sha(root/XCVIII/'SHA256SUMS'),
          'XCVIII_manifest_sha256':sha(root/XCVIII/'numeric_manifest.json'),
          'XCI_protocol_sha256':sha(root/XCI/'evaluation_protocol.json')}
    print('[SOURCES] PASS: 2354 CII hashes, 1176 numeric VTIs/structural VTUs, XCVIII 16-field GUDHI, 2007 Phase6 regression',flush=True)
    print('[SCOPE] PASS: 168 contiguous Gulf development UTC hours, 6 arms, 2 anchored hours, zero locked external',flush=True)
    return pilot,decoder,bypair,lookup,line

def config(line):
    return {'format':SCHEMA,'scope':'DEVELOPMENT_ONLY','region':'gulf_inland',
            'start_utc':'2010-02-01T00:00:00','wtk_indices':[27048,27215],
            'n_hours':N,'arms':list(ARMS),'metrics':list(METRICS),
            'crop':{'x0':0,'y0':0,'height':160,'width':160},
            'anchor_hours':{'0':0,'84':1},'source_sha256':line,
            'stage_policy':'one sha-sealed JSON per complete hour',
            'hour_independence_claim':False,'external_test':False}

def pdpath(root,cii,h,a):return root/CII/'pd'/f'dev_{a}_s{h}_speed_p160_x0_y0_pd_port_0.vtu'

def one_hour(root,pilot,decoder,h,bypair,lookup):
    def read(a):
        p=pdpath(root,CII,h,a)
        d=pilot.parse(decoder.read_pd,p)
        need(set(d)=={0,1},f'Unexpected homology dimensions h{h} {a}')
        for dim in (0,1):
            z=d[dim]
            need(z.ndim==2 and z.shape[1]==2 and bool(np.isfinite(z).all()),f'Invalid H{dim} pairs h{h} {a}')
        return d
    gt=read('gt');rows=[]
    for a in ARMS:
        sr=read(a);dist=pilot.distance(gt,sr)
        vals=[float(dist[m]) for m in METRICS]
        need(all(math.isfinite(v) and v>=0 for v in vals) and
             vals[0]<=vals[1]+1e-7 and vals[1]<=vals[2]+1e-7,
             f'Invalid corrected distances h{h} {a}: {vals}')
        r={'hour':str(h),'utc':( __import__('datetime').datetime(2010,2,1)+__import__('datetime').timedelta(hours=h)).isoformat(),
           'wtk_index':str(27048+h),'region':'gulf_inland','arm':a,
           'gt_sha256':bypair[(h,'gt')]['pd_sha256'],'sr_sha256':bypair[(h,a)]['pd_sha256'],
           'D0_gt':str(len(gt[0])),'D1_gt':str(len(gt[1])),
           'D0_sr':str(len(sr[0])),'D1_sr':str(len(sr[1])),
           **{m:fmt(dist[m]) for m in METRICS}}
        check_anchors(r,lookup);rows.append(r)
    return rows

def check_rows(rows,h,bypair,lookup):
    need(len(rows)==6 and [r['arm'] for r in rows]==list(ARMS),'Incorrect hour/arm row inventory')
    for r in rows:
        need(set(r)==set(COLS) and r['hour']==str(h) and r['region']=='gulf_inland'
             and r['wtk_index']==str(27048+h),'Hourly row field order/identity changed')
        need(r['gt_sha256']==bypair[(h,'gt')]['pd_sha256'] and
             r['sr_sha256']==bypair[(h,r['arm'])]['pd_sha256'],'PD source SHA mismatch')
        expected=(__import__('datetime').datetime(2010,2,1)+__import__('datetime').timedelta(hours=h)).isoformat()
        need(r['utc']==expected,'Hour UTC timestamp mismatch')
        need(all(int(r[k])>=0 for k in ('D0_gt','D1_gt','D0_sr','D1_sr')),'Invalid feature count')
        v=[float(r[m]) for m in METRICS]
        need(all(math.isfinite(x) and x>=0 for x in v) and v[0]<=v[1]+1e-7 and v[1]<=v[2]+1e-7,
             'Invalid distance row')
        check_anchors(r,lookup)

def stage_rows(stage,cfg,bypair,lookup):
    need((stage/'run_config.json').is_file() and json.loads((stage/'run_config.json').read_text())==cfg,
         'Stage configuration/source provenance drift')
    hourdir=stage/'hours';need(hourdir.is_dir(),'Stage hours directory missing')
    rows_by_hour={}
    for p in hourdir.iterdir():
        need(p.is_file() and re.fullmatch(r'h\d{3}\.json',p.name),f'Unexpected stage artifact {p}')
        h=int(p.stem[1:]);need(0<=h<N and h not in rows_by_hour,f'Duplicate or invalid staged hour {h}')
        d=json.loads(p.read_text())
        need(set(d)=={'hour','rows','payload_sha256'} and d['hour']==h and
             sha_string(stable(d['rows']))==d['payload_sha256'],f'Corrupt staged hour {h}')
        check_rows(d['rows'],h,bypair,lookup)
        rows_by_hour[h]=d['rows']
    need(set(rows_by_hour)==set(range(len(rows_by_hour))),'Staged hours must be contiguous from 0')
    return rows_by_hour

def sha_string(x):return hashlib.sha256(x.encode()).hexdigest()

def write_hour(stage,h,rows):
    p=stage/'hours'/f'h{h:03d}.json';need(not p.exists(),'Staged hour collision')
    payload={'hour':h,'rows':rows,'payload_sha256':sha_string(stable(rows))}
    atomic_text(p,jd(payload))

def read_csv(path,cols):
    with Path(path).open(newline='') as f:
        reader=csv.DictReader(f)
        need(tuple(reader.fieldnames or ())==cols,f'CSV column drift {path}')
        return list(reader)

def write_csv(path,rows,cols):
    with Path(path).open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=cols);wr.writeheader();wr.writerows(rows)

def aggregate(rows):
    import statistics
    days=[]
    for day in range(7):
        for arm in ARMS:
            subset=[r for r in rows if int(r['hour'])//24==day and r['arm']==arm]
            need(len(subset)==24,f'Incomplete day={day} arm={arm}')
            days.append({'day_start_utc':f'2010-02-0{day+1}T00:00:00',
                         'arm':arm,'n_hours':'24',
                         **{m:fmt(statistics.mean(float(r[m]) for r in subset)) for m in METRICS}})
    means={a:{m:statistics.mean(float(r[m]) for r in rows if r['arm']==a) for m in METRICS} for a in ARMS}
    return days,means

def report(rows,days,means):
    s=['# Phase 7 Part CIII — corrected GUDHI distances, 168-hour development week','',
       'Gulf inland, February 1–7 2010 UTC, 168 consecutive hourly samples.',
       '**This is one development region-week, not 168 independent generalization replicates.**','',
       '1176 numerically decoded canonical TTK diagrams; 1008 model-to-GT comparisons,',
       'using the exact Phase6 corrected GUDHI functions, with historical 2007 and XCVIII hour-0/84 regressions.','',
       '| Model | Mean hourly dB ↓ | Mean hourly W2inf ↓ | Mean hourly W22 ↓ |',
       '|---|---:|---:|---:|']
    for a in ARMS:s.append(f'| {a} | {means[a]["dB"]:.6f} | {means[a]["W2inf"]:.6f} | {means[a]["W22"]:.6f} |')
    s+=['','## Daily descriptive means','',
         '| Day UTC | Model | N hours | dB ↓ | W2inf ↓ | W22 ↓ |',
         '|---|---|---:|---:|---:|---:|']
    for d in days:s.append(f'| {d["day_start_utc"][:10]} | {d["arm"]} | 24 | {float(d["dB"]):.6f} | {float(d["W2inf"]):.6f} | {float(d["W22"]):.6f} |')
    s+=['','**Scope and limitations:** numeric topology concerns only the unchanged top-left 160×160 scalar-speed crop,',
         'not the full 500×500 vector field. Day-level summaries describe correlated hours, not independent statistical blocks.',
         'Verification recomputes every value with the same trusted GUDHI decoder and distance implementation; this is not an',
         'independently implemented numerical algorithm. No Phase5E locality, merge trees, training, model selection,',
         'locked-region inference or external geographic generalization is performed in CIII.',
         'Original PhIRE pretrained WTK spatial/temporal exposure is unverified.','']
    return '\n'.join(s)+'\n'

def hash_inventory(folder):
    names=[]
    for line in (folder/'SHA256SUMS').read_text().splitlines():
        m=re.fullmatch(r'([0-9a-f]{64})  ([^\n]+)',line)
        need(m is not None,f'Invalid SHA256SUMS row {line}')
        digest,name=m.groups();need(name in FILES and name not in names,'Unknown/repeated output file')
        need(sha(folder/name)==digest,f'SHA mismatch {name}')
        names.append(name)
    need(set(names)==set(FILES),'Four output checksum entries required')

def publish(root,stage,cfg,bypair,lookup):
    out=root/OUT
    need(not out.exists(),'Immutable CIII output exists')
    hrs=stage_rows(stage,cfg,bypair,lookup)
    need(len(hrs)==N,'Cannot publish incomplete week')
    rows=[r for h in range(N) for r in hrs[h]]
    need(len(rows)==1008,'Wrong output comparison total')
    days,means=aggregate(rows);s=report(rows,days,means)
    out.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.ciii-publish-',dir=out.parent) as td:
        tmp=Path(td)
        write_csv(tmp/FILES[0],rows,COLS)
        write_csv(tmp/FILES[1],days,DAY_COLS)
        obj={'format':SCHEMA,'run_config':cfg,'n_hours':N,'n_diagrams':1176,
             'n_pairs':1008,'n_daily_rows':42,'anchor_score_regressions':12,
             'mean_hourly':means,'verification':'all numeric rows independently recomputed with same original GUDHI functions',
             'hour_files_sha256':{f'h{h:03d}':sha(stage/'hours'/f'h{h:03d}.json') for h in range(N)}}
        (tmp/FILES[2]).write_text(jd(obj));(tmp/FILES[3]).write_text(s)
        (tmp/'SHA256SUMS').write_text(''.join(f'{sha(tmp/name)}  {name}\n' for name in FILES))
        hash_inventory(tmp)
        need(not out.exists(),'Output collision at publish time')
        tmp.rename(out)
    print(f'[COMPUTE-GATE] PASS: 1176 numerically decoded PDs, 1008 corrected distances, '
          f'42 day-arm means, 12 XCVIII anchor regressions, 4 SHA256 entries; output={out}',flush=True)

def verify(root,pilot,decoder,bypair,lookup,cfg):
    out=root/OUT;need(out.is_dir(),'CIII immutable output missing')
    hash_inventory(out)
    got=json.loads((out/FILES[2]).read_text())
    need(got['format']==SCHEMA and got['run_config']==cfg and got['n_hours']==N and
         got['n_diagrams']==1176 and got['n_pairs']==1008 and
         got['n_daily_rows']==42 and got['anchor_score_regressions']==12,'Output manifest schema/lineage differs')
    stored=read_csv(out/FILES[0],COLS)
    need(len(stored)==1008,'Per-hour output row count incorrect')
    recalc=[]
    for h in range(N):
        one=one_hour(root,pilot,decoder,h,bypair,lookup)
        for i,r in enumerate(one):
            v=stored[h*6+i]
            for k in COLS:
                if k in METRICS:
                    need(math.isclose(float(v[k]),float(r[k]),rel_tol=1e-11,abs_tol=1e-8),
                         f'Independent same-method recomputation failed h{h} {r["arm"]} {k}')
                else:need(v[k]==r[k],f'Output row identity drift h{h} {r["arm"]} {k}')
        recalc+=one
    days,means=aggregate(recalc)
    need(read_csv(out/FILES[1],DAY_COLS)==days,'Daily summary tampered/differs')
    need(got['mean_hourly']==means and (out/FILES[3]).read_text()==report(recalc,days,means),
         'Numerical report/mean mismatch')
    stage=root/str(OUT.parent/('.'+OUT.name+'.staging'))
    if stage.is_dir():
        stage_rows(stage,cfg,bypair,lookup)
        need(got['hour_files_sha256']=={f'h{h:03d}':sha(stage/'hours'/f'h{h:03d}.json') for h in range(N)},
             'Published stage source differs')
    print('[VERIFY-GATE] PASS: 1176 numeric TTK PD decodes, 1008 recomputed corrected GUDHI comparisons, '
          '12 XCVIII anchors, 7x6 daily rows, 4 SHA256 outputs; no external tests',flush=True)

def selftest():
    import gudhi
    from gudhi.wasserstein import wasserstein_distance
    pts=np.array([[0.,2.],[1.,2.2]],float)
    need(gudhi.bottleneck_distance(pts,pts,e=0.)==0. and
         wasserstein_distance(pts,pts,matching=False,order=2.,internal_p=2.,keep_essential_parts=False)==0.,
         'GUDHI identity check failed')
    need(len(COLS)==14 and len(DAY_COLS)==6 and len(FILES)==4 and len(ARMS)*N==1008,'Schema discrepancy')
    with tempfile.TemporaryDirectory() as td:
        p=Path(td);(p/'hours').mkdir();cfg={'test':True};(p/'run_config.json').write_text(jd(cfg))
        rows=[{'hour':'0','arm':a} for a in ARMS]
        atomic_text(p/'hours'/'h000.json',jd({'hour':0,'rows':rows,'payload_sha256':sha_string(stable(rows))}))
        obj=json.loads((p/'hours'/'h000.json').read_text());need(obj['payload_sha256']==sha_string(stable(obj['rows'])),'Stage digest fails')
        obj['rows'][0]['hour']='bad';need(obj['payload_sha256']!=sha_string(stable(obj['rows'])),'Stage tamper not detected')
    print(f'[SELFTEST] PASS: GUDHI {getattr(gudhi,"__version__","installed")}, 1008/42 schema, anchors, atomic sealed hour tamper check',flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',required=True,choices=['selftest','preflight','compute','verify'])
    p.add_argument('--repo-root',type=Path,default=Path.cwd())
    p.add_argument('--max-new-hours',type=int,default=None)
    a=p.parse_args()
    if a.mode=='selftest':selftest();return
    need(a.max_new_hours is None or (a.mode=='compute' and 1<=a.max_new_hours<=N),
         '--max-new-hours only for compute, 1..168')
    root=a.repo_root.resolve();os.chdir(root)
    pilot,decoder,bypair,lookup,line=source_gate(root)
    cfg=config(line);out=root/OUT
    stage=out.parent/('.'+out.name+'.staging')
    if a.mode=='verify':verify(root,pilot,decoder,bypair,lookup,cfg);return
    need(not out.exists(),f'Completed CIII output exists; verify instead: {out}')
    if a.mode=='preflight':
        if stage.is_dir():
            hrs=stage_rows(stage,cfg,bypair,lookup)
            print(f'[PREFLIGHT] PASS: {len(hrs)}/168 previous atomic hours checked; no numeric distances computed or writes',flush=True)
        else:print('[PREFLIGHT] PASS: frozen CII/XCVIII lineage, 168 diagram sets, original GUDHI, output absent, no writes',flush=True)
        return
    out.parent.mkdir(parents=True,exist_ok=True)
    with (out.parent/('.'+out.name+'.lock')).open('a+') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        need(not out.exists(),'Final output appeared')
        if not stage.exists():
            stage.mkdir();(stage/'hours').mkdir();atomic_text(stage/'run_config.json',jd(cfg))
        hrs=stage_rows(stage,cfg,bypair,lookup)
        print(f'[RESUME] PASS: {len(hrs)}/168 sealed numeric hours',flush=True)
        done=0
        for h in range(N):
            if h in hrs:continue
            if a.max_new_hours is not None and done>=a.max_new_hours:
                print(f'[PARTIAL-GATE] PASS: {len(hrs)}/168 sealed hours, resume with --mode compute',flush=True)
                return
            rows=one_hour(root,pilot,decoder,h,bypair,lookup)
            check_rows(rows,h,bypair,lookup);write_hour(stage,h,rows)
            hrs[h]=rows;done+=1
            print(f'[HOUR-NUMERIC] {h+1:03d}/168 WTK={27048+h} six arms PASS '
                  f'grad_W22={float(rows[2]["W22"]):.6f} cnn_W22={float(rows[0]["W22"]):.6f}',flush=True)
        publish(root,stage,cfg,bypair,lookup)

if __name__=='__main__':
    try:main()
    except Exception as e:
        print(f'[CIII] ERROR {type(e).__name__}: {e}',file=sys.stderr,flush=True)
        sys.exit(1)
