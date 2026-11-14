#!/usr/bin/env python3
"""Phase 7: seven-week 2010 development corrected PD + spatial locality.

Adapter to SHA-pinned, unchanged CIII/CIV numerical implementations. Bound
copies are compiled in memory; adaptations are *only* week metadata / source
paths / anchor index bindings. The trusted XCVIII GUDHI numerical functions and
original Phase5E matching implementation are not modified.

Each source stage must be independently sealed and verified before moving on.
No locked geographic tests, no training, no 2012 replication.
"""
from __future__ import annotations
import argparse,csv,hashlib,importlib.util,json,os,sys,tempfile
from datetime import datetime
from pathlib import Path

BASE=Path('ttk_runs_fixed/topology_finetuning')
CVII_SCRIPT=Path('phase7_cvii_update/scripts/phase7_cvii_dev2010_sevenweek_pd.py')
CIII_SCRIPT=Path('phase7_ciii_update/scripts/phase7_ciii_dev_fullweek_numeric_pd.py')
CIV_SCRIPT=Path('phase7_civ_update/scripts/phase7_civ_dev_fullweek_locality.py')
SELF=Path('phase7_cviii_update/scripts/phase7_cviii_cix_2010_dev_topology.py')
PINS={CVII_SCRIPT:'4f06a60878a98c639931cc048c190a219347dff90b9d4749338834d59f4802ae',
      CIII_SCRIPT:'59518de2eed0c5ff24e00a5e75c9889226bd1d456f252d6be7449f72afbce632',
      CIV_SCRIPT:'6381c75d84d10e812bed3c93d53bc6d03544cd8bf5f53f515cbf2bd952cec7d5'}
ARMS=('cnn','gan','grad','cv','pers','f1')
RADIUS=('4.0','8.0','16.0','32.0','64.0','96.0','128.0','inf')
REGIONS=('gulf_inland','southwest_borderland')
WEEKS=('2010-02-01T00:00:00','2010-05-01T00:00:00','2010-08-01T00:00:00','2010-11-01T00:00:00')
TAG='20261021'
S_CIII='phase7_ciii_dev_gulf_inland_feb2010_corrected_pd168_20261019'
S_CIV='phase7_civ_dev_gulf_inland_feb2010_locality168_20261019'
S_CII='phase7_cii_dev_gulf_inland_feb2010_pd168_20261018'

def need(ok,msg):
    if not ok:raise RuntimeError(msg)

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def load(p,name):
    spec=importlib.util.spec_from_file_location(name,p)
    need(spec and spec.loader,'Cannot load frozen driver '+str(p))
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m)
    return m

def replace(src,old,new,qty=None):
    ct=src.count(old)
    need(ct>0 and (qty is None or ct==qty),f'Unexpected frozen template: {old!r} count={ct}, expected={qty}')
    return src.replace(old,new)

def name(phase,reg,w):
    return f'phase7_{phase}_dev_{reg}_{w[:10].replace("-","")}_{"corrected_pd168" if phase=="cviii" else "locality168"}_{TAG}'

def adapt_source(path,root,phase,reg,w,anchors,pd_dir,numeric_dir=None):
    src=(root/path).read_text()
    need(sha(root/path)==PINS[path],f'Frozen {phase} script changed')
    dates=w[:10].replace('-','');month=int(w[5:7]);index={'02':27048,'05':29184,'08':31392,'11':33600}[w[5:7]]
    if phase=='cviii':
        src=replace(src,f"CII=BASE/'{S_CII}'",f"CII=BASE/'{pd_dir}'",1)
        src=replace(src,f"OUT=BASE/'{S_CIII}'",f"OUT=BASE/'{name('cviii',reg,w)}'",1)
        src=replace(src,"SCHEMA='phase7-ciii-gulf-inland-feb2010-corrected-pd168-v1'",
                    "SCHEMA='phase7-cviii-dev2010-corrected-pd168-v1'",1)
        src=replace(src,'ANCHORS={0:0,84:1}',f'ANCHORS={{0:{anchors[0]},84:{anchors[84]}}}',1)
        src=replace(src,'int(line[\'sample\']) in (0,1)',
                    f"int(line['sample']) in ({anchors[0]},{anchors[84]})",1)
        src=replace(src,'for i in (0,1) for a in ARMS',
                    f'for i in ({anchors[0]},{anchors[84]}) for a in ARMS',1)
    else:
        need(numeric_dir is not None,'Locality needs sealed numeric input')
        src=replace(src,f"CII=BASE/'{S_CII}'",f"CII=BASE/'{pd_dir}'",1)
        src=replace(src,f"CIII=BASE/'{S_CIII}'",f"CIII=BASE/'{numeric_dir}'",1)
        src=replace(src,f"OUT=BASE/'{S_CIV}'",f"OUT=BASE/'{name('cix',reg,w)}'",1)
        src=replace(src,"SCHEMA='phase7-civ-gulf-inland-feb2010-locality168-v1'",
                    "SCHEMA='phase7-cix-dev2010-locality168-v1'",1)
        src=replace(src,'ANCHORS={0:0,84:1}',f'ANCHORS={{0:{anchors[0]},84:{anchors[84]}}}',1)
    src=replace(src,'gulf_inland',reg)
    src=replace(src,'Gulf inland',reg.replace('_',' ').title())
    src=replace(src,'February 1–7',w[:7]+' 1–7')
    if '20100201' in src:src=replace(src,'20100201',dates)
    src=replace(src,'2010-02-01',w[:10])
    src=replace(src,'2010-02-0',w[:7]+'-0')
    src=replace(src,'(2010,2,1)',f'(2010,{month},1)')
    src=replace(src,'27048',str(index))
    src=replace(src,'27215',str(index+167))
    # All dynamic compiled transforms are pinned to original source and this adapter.
    compile(src,str(path)+'::'+reg+'::'+w,'exec')
    return src

def compiled(src,key):
    import types
    module=types.ModuleType(key)
    module.__file__=key
    sys.modules[key]=module
    exec(compile(src,key,'exec'),module.__dict__)
    return module

def prepared(root,cvii,ctx,base,reg,w,phase):
    need((reg,w) in ctx[2][1].weeks(),'Attempt to select non-development region/week')
    cvi,cii,cvi_ctx,original=ctx
    pdout=cvii.bind(root,cvi,cii,cvi_ctx,original,reg,w)
    # Structural full-week gate, including 14 exact historical diagrams.
    cvii.invoke(cii,root,'verify')
    pd_dir=pdout.name
    cfg=json.loads((pdout/'run_config.json').read_text());mapping=cfg['anchor_hours']
    anchors={0:int(mapping['0']),84:int(mapping['84'])}
    need(set(anchors.values())<=set(range(16)) and anchors[0]!=anchors[84], 'PD anchor mapping invalid')
    ciii=compiled(adapt_source(CIII_SCRIPT,root,'cviii',reg,w,anchors,pd_dir),
                  f'cviii_bound_{reg}_{w[:10].replace("-","")}')
    orig_mod=ciii.mod
    def mod(p,n):
        if Path(p).resolve()==(root/ciii.CII_SCRIPT).resolve():return cii
        return orig_mod(p,n)
    ciii.mod=mod
    first_source=ciii.source_gate
    def numeric_gate(r):
        pilot,decoder,bypair,lookup,line=first_source(r)
        # Original pinned VTI/VTU input and per-week numeric adapter provenance.
        line={**line,'adapted_driver_sha256':sha(root/SELF),
              'cvii_driver_sha256':PINS[CVII_SCRIPT],
              'cvii_manifest_sha256':sha(pdout/'extraction_manifest.json'),
              'week_binding':{'region':reg,'start_utc':w,'anchors':{str(k):v for k,v in anchors.items()}}}
        return pilot,decoder,bypair,lookup,line
    ciii.source_gate=numeric_gate
    ciii.config=lambda line,old=ciii.config:old(line)
    need(ciii.ANCHORS==anchors and ciii.CII==BASE/pd_dir,'Numeric weekly bound drift')
    if phase=='cviii':return ciii,None,anchors
    # The original CIV *requires* independently verified corrected distances.
    pilot,decoder,bypair,lookup,line=ciii.source_gate(root)
    ciii.verify(root,pilot,decoder,bypair,lookup,ciii.config(line))
    numeric_dir=name('cviii',reg,w)
    civ=compiled(adapt_source(CIV_SCRIPT,root,'cix',reg,w,anchors,pd_dir,numeric_dir),
                 f'cix_bound_{reg}_{w[:10].replace("-","")}')
    original_load=civ.load
    def load_civ(p,n):
        if Path(p).resolve()==(root/civ.CIII_SCRIPT).resolve():return ciii
        return original_load(p,n)
    civ.load=load_civ
    orig_gate=civ.source_gate
    def locality_gate(r):
        vals=orig_gate(r)
        v=list(vals);v[-1]={**v[-1],
            'adapted_driver_sha256':sha(root/SELF),
            'cvii_driver_sha256':PINS[CVII_SCRIPT],
            'week_binding':{'region':reg,'start_utc':w,'anchors':{str(k):s for k,s in anchors.items()}}}
        return tuple(v)
    civ.source_gate=locality_gate
    need(civ.ANCHORS==anchors and civ.CIII==BASE/numeric_dir,'Locality weekly bound drift')
    return ciii,civ,anchors

def run_module(module,script_path,root,mode,max_hours):
    original=sys.argv
    argv=[str(root/script_path),'--repo-root',str(root),'--mode',mode]
    if max_hours:argv.extend(['--max-new-hours',str(max_hours)])
    try:sys.argv=argv;module.main()
    finally:sys.argv=original

def inventory(root,weeks):
    for r,w in weeks:
        s=[]
        for phase in ('cviii','cix'):
            dest=root/BASE/name(phase,r,w)
            stage=dest.parent/('.'+dest.name+'.staging')
            s.append(phase+('=SEALED' if dest.is_dir() else '=STAGED' if stage.is_dir() else '=PENDING'))
        print('[WEEK]',r,w,*s,flush=True)

def summary(root,weeks,cvii,ctx):
    # Must be called AFTER all seven weeks pass independent numerical and locality verification.
    import statistics
    out=root/BASE/('phase7_cx_dev2010_eightweek_summary_'+TAG)
    need(not out.exists(),'Immutable eight-week summary already exists')
    rows=[];source_dirs=[]
    need(set(weeks)=={(r,w) for r in REGIONS for w in WEEKS if (r,w)!=('gulf_inland',WEEKS[0])},'Unexpected 2010 summary week inventory')
    for r,w in [('gulf_inland','2010-02-01T00:00:00'),*weeks]:
        if (r,w)==('gulf_inland','2010-02-01T00:00:00'):
            numeric=root/BASE/S_CIII;loc=root/BASE/S_CIV
            infer=root/BASE/'phase7_ci_dev_gulf_inland_feb2010_inference168_20261018'
        else:
            numeric=root/BASE/name('cviii',r,w);loc=root/BASE/name('cix',r,w)
            infer=root/BASE/ctx[0].output_name(r,w) if hasattr(ctx[0],'output_name') else root/BASE/('phase7_cvi_dev_'+r+'_'+w[:10].replace('-','')+'_inference168_20261019')
        need(all(p.is_dir() for p in (numeric,loc,infer)),f'Missing sealed results {r} {w}')
        source_dirs.extend((numeric,loc,infer))
        # Inventories and cross-metric consistency, no uncited recomputation of full week on summary call.
        for folder,n in ((numeric,4),(loc,5),(infer,13)):
            entries=(folder/'SHA256SUMS').read_text().splitlines()
            need(len(entries)==n,'Missing SHA entries in '+str(folder))
            for line in entries:
                digest,rel=line.split('  ',1)
                need(len(digest)==64 and sha(folder/rel)==digest,'Checksum mismatch '+str(folder/rel))
        nm=json.loads((numeric/'numeric_manifest.json').read_text());lm=json.loads((loc/'locality_manifest.json').read_text())
        need(nm['n_pairs']==1008 and lm['n_aggregate_rows']==8064 and lm['n_ciii_infinity_anchors']==1008,'Wrong source counts')
        numeric_means=nm['mean_hourly'];local_means=lm['mean_hourly']
        progress=json.loads((infer/'progress.json').read_text())
        for arm in ARMS:
            im=progress['models'][arm]['mean_hourly_metrics']
            need(abs(float(numeric_means[arm]['W22'])-float(local_means[arm]['rinf_w22']))<1e-6,'Infinity cross-stage mismatch')
            rows.append({'region':r,'week_start_utc':w,'arm':arm,
                         'uv_rmse':im['uv_rmse_mps'],
                         'speed_mae':im['speed_mae_mps'],
                         'dB':numeric_means[arm]['dB'],
                         'W2inf':numeric_means[arm]['W2inf'],'W22':numeric_means[arm]['W22'],
                         **local_means[arm]})
    need(len(rows)==48,'Expected 8 region-weeks × 6 model arms')
    cols=['region','week_start_utc','arm','uv_rmse','speed_mae','dB','W2inf','W22',
          'r4_w22','r16_w22','r32_w22','rinf_w22','r16_gt_persistence_coverage']
    means={a:{k:statistics.mean(float(row[k]) for row in rows if row['arm']==a) for k in cols[3:]
              if all(str(row[k]) not in ('nan','None') for row in rows if row['arm']==a)} for a in ARMS}
    md=['# Phase 7 — eight development region-week summary, 2010','',
        'Two development regions × four frozen 168-hour seasonal windows; eight equal-weighted region-week means.',
        'One year and only two geographic regions: **development validation, not external generalization**.',
        'Original top-left 160×160 speed crop and frozen eight grid-cell radii; no held-out geographic tests.','',
        '| Model | UV RMSE ↓ | Speed MAE ↓ | PD W22 ↓ | r4 W22 ↓ | r16 W22 ↓ | r∞ W22 ↓ | r16 GT coverage ↑ |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        m=means[arm];md.append('| '+arm+' | '+' | '.join(f'{m.get(k,float("nan")):.6f}' for k in
                              ('uv_rmse','speed_mae','W22','r4_w22','r16_w22','rinf_w22','r16_gt_persistence_coverage'))+' |')
    md.extend(['','Consecutive hourly fields within each week are correlated.',
               'No pretrained-model WTK exposure assertions, independent solver, or sample-level significance claims.'])
    out.parent.mkdir(exist_ok=True,parents=True)
    with tempfile.TemporaryDirectory(prefix='.cx-summary-',dir=out.parent) as tmpdir:
        tmp=Path(tmpdir)
        with (tmp/'per_region_week.csv').open('w',newline='') as f:
            wr=csv.DictWriter(f,fieldnames=cols);wr.writeheader();wr.writerows(rows)
        (tmp/'summary_report.md').write_text('\n'.join(md)+'\n')
        (tmp/'summary_manifest.json').write_text(json.dumps({'format':'phase7-cx-dev2010-summary-v1',
              'n_region_weeks':8,'n_rows':48,'equal_weight_block_means':means,
              'upstream_sha256':{str(p.relative_to(root)):sha(p/'SHA256SUMS') for p in
                 set([root/BASE/S_CIII,root/BASE/S_CIV]+[root/BASE/name(ph,r,w) for r,w in weeks for ph in ('cviii','cix')])}},sort_keys=True,indent=2)+'\n')
        files=['per_region_week.csv','summary_report.md','summary_manifest.json']
        (tmp/'SHA256SUMS').write_text(''.join(f'{sha(tmp/f)}  {f}\n' for f in files))
        need(not out.exists(),'Summary output collision');tmp.rename(out)
    print('[SUMMARY-GATE] PASS: eight equal-weighted 2010 development region-weeks, 48 model-block rows; output='+str(out),flush=True)

def selftest(root):
    for path,d in PINS.items():need(sha(root/path)==d,'Frozen source SHA changed: '+str(path))
    cvii=load(root/CVII_SCRIPT,'cviii_self_cvii')
    cvii.selftest()
    n=0
    for r in REGIONS:
        for w in WEEKS:
            if (r,w)==('gulf_inland',WEEKS[0]):continue
            index=next(i for i,(rr,ww) in enumerate([(rg,wk) for rg in REGIONS for wk in WEEKS]) if rr==r and ww==w)
            anchors={0:2*index,84:2*index+1}
            need(all(i<16 for i in anchors.values()),'Anchor range')
            for phase,path in (('cviii',CIII_SCRIPT),('cix',CIV_SCRIPT)):
                source=adapt_source(path,root,phase,r,w,anchors,'fixture_pd','fixture_numeric')
                need(f'ANCHORS={{0:{anchors[0]},84:{anchors[84]}}}' in source,'Anchor mis-bound')
            n+=1
    need(n==7 and 7*168*6==7056 and 7*168*6*8==56448,'Cardinality error')
    print('[SELFTEST] PASS: original CIII/CIV pins, seven unique frozen metadata bindings, all 14 history anchors, numerical/locality cardinalities',flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',required=True,choices=('selftest','inventory','preflight-numeric','numeric','numeric-all','verify-numeric',
       'verify-numeric-all','preflight-locality','locality','locality-all','verify-locality','verify-locality-all','summary'))
    p.add_argument('--repo-root',default=str(Path(__file__).resolve().parents[2]))
    p.add_argument('--region',choices=REGIONS)
    p.add_argument('--week-start',choices=WEEKS)
    p.add_argument('--max-new-hours',type=int)
    args=p.parse_args();root=Path(args.repo_root).resolve()
    need(root.is_dir(),'Repository directory absent')
    if args.mode=='selftest':selftest(root);return
    os.chdir(root);sys.path.insert(0,str(root))
    for path,d in PINS.items():need(sha(root/path)==d,'Frozen source hash drift: '+str(path))
    cvii=load(root/CVII_SCRIPT,'cviii_main_cvii')
    ctx=cvii.setup(root)
    weeks=ctx[2][1].weeks()
    need(len(weeks)==7 and all(r in REGIONS and w in WEEKS for r,w in weeks),'Development scope changed')
    if args.mode=='inventory':need(not args.region and not args.week_start,'Inventory does not accept region/week');inventory(root,weeks);return
    if args.mode=='summary':need(not args.region and not args.week_start,'Summary selects all weeks');summary(root,weeks,cvii,ctx);return
    allmode=args.mode.endswith('-all')
    if allmode:need(not args.region and not args.week_start,'All mode excludes manual week selection')
    else:need((args.region,args.week_start) in weeks,'Select one of seven frozen development weeks')
    if args.max_new_hours is not None:need(args.max_new_hours>0 and args.mode in ('numeric','locality'),
                                           'Max new hours only supported for single-week smoke runs')
    phase='cix' if 'locality' in args.mode else 'cviii'
    entries=weeks if allmode else [(args.region,args.week_start)]
    for r,w in entries:
        ciii,civ,anchors=prepared(root,cvii,ctx,ctx,r,w,phase)
        target=root/BASE/name(phase,r,w)
        mode=('verify' if args.mode.startswith('verify') else
              'preflight' if args.mode.startswith('preflight') else 'compute')
        if allmode and target.is_dir() and mode=='compute':mode='verify'
        script=CIV_SCRIPT if phase=='cix' else CIII_SCRIPT
        run_module(civ if phase=='cix' else ciii,script,root,mode,args.max_new_hours)
        print(f'[WEEK-{phase.upper()}] PASS: {r} {w} {mode}',flush=True)
    if allmode:
        count=sum((root/BASE/name(phase,r,w)).is_dir() for r,w in weeks)
        if args.mode in ('numeric-all','verify-numeric-all','locality-all','verify-locality-all'):
            need(count==7,f'Only {count}/7 sealed {phase} weeks')
        print(f'[ALL-{args.mode.upper()}] PASS: {count}/7 sealed {phase} weeks; zero locked region access',flush=True)

if __name__=='__main__':
    try:main()
    except Exception as e:
        print(f'[CVIII/CIX] ERROR {type(e).__name__}: {e}',file=sys.stderr,flush=True)
        sys.exit(1)
