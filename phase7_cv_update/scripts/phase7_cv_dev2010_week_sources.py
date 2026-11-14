#!/usr/bin/env python3
"""Part CV: read-only planning and resumable WTK acquisition for 7 remaining 2010 DEV weeks.

Reuses *unmodified* SHA-pinned Part C source code. Binds only those frozen
2010 region/week identities already represented by XCV hour-0/84 arrays.
Never evaluates model predictions or accesses locked external regions.
"""
from __future__ import annotations
import argparse, contextlib, hashlib, importlib.util, io, json, re, sys
from datetime import datetime
from pathlib import Path

BASE = Path('ttk_runs_fixed/topology_finetuning')
XCV = BASE/'phase7_xcv_development_seasonal16_20261018'
CIV = BASE/'phase7_civ_dev_gulf_inland_feb2010_locality168_20261019'
XCI = BASE/'phase7_xci_frozen_evaluation_protocol_20261015'
C_SCRIPT=Path('phase7_c_update/scripts/phase7_c_dev_fullweek_source.py')
C_SHA='dfb073530be638a87d51d90916af1e5b9293cb837a4cde8663b4dc7c4778b54b'
REGIONS={'gulf_inland':(205,705,1471,1971),'southwest_borderland':(173,673,572,1072)}
WEEKS=('2010-02-01T00:00:00','2010-05-01T00:00:00','2010-08-01T00:00:00','2010-11-01T00:00:00')
COMPLETED=('gulf_inland','2010-02-01T00:00:00')
FIELDS_PER_WEEK=168
DATE_TAG='20261019'

def need(cond,msg):
    if not cond: raise RuntimeError(msg)

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()

def import_c(path):
    spec=importlib.util.spec_from_file_location('phase7_cv_immutable_part_c',path)
    need(spec is not None and spec.loader is not None,'Missing original Part C module')
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
    return m

def weeks():
    return [(region,anchor) for region in REGIONS for anchor in WEEKS
            if (region,anchor)!=COMPLETED]

def week_name(region,anchor):
    return f'phase7_cv_dev_{region}_{anchor[:10].replace("-","")}_week168_{DATE_TAG}'

def inventory(root):
    cfile=root/C_SCRIPT
    need(cfile.is_file() and sha(cfile)==C_SHA,'Original Part C script missing or modified')
    c=import_c(cfile)
    c.hashed_files(root/XCI,['evaluation_protocol.json','evaluation_protocol.md'])
    xcv_hash=c.hashed_files(root/XCV)
    need(len(xcv_hash)==18,'Source pilot XCV does not have exactly 18 hashes')
    protocol=json.loads((root/XCI/'evaluation_protocol.json').read_text())['protocol']
    need(tuple(protocol['geographic_roles']['development'])==tuple(REGIONS),'Development region identities changed')
    need(tuple(protocol['geographic_roles']['locked_external_test'])==('upper_midwest','pacific_inland'),'Locked region identities changed')
    need(tuple(protocol['times']['primary_week_start_utc'])==WEEKS,'Frozen primary week starts changed')
    need(protocol['times']['hourly_fields_per_week']==FIELDS_PER_WEEK,'Frozen 168-hour week specification changed')
    manifest=json.loads((root/XCV/'seasonal16_manifest.json').read_text())
    need(manifest['n_fields']==16 and len(manifest['fields'])==16,'Source pilot manifest count changed')
    lookup={(e['region'],e['anchor_utc'],e['offset_hours']):e for e in manifest['fields']}
    need(len(lookup)==16,'Duplicated XCV anchor identity')
    for region in REGIONS:
        for anchor in WEEKS:
            a=lookup.get((region,anchor,0));b=lookup.get((region,anchor,84))
            need(a is not None and b is not None,'One of the frozen XCV hour 0/84 anchors is missing')
            need(tuple(a['rect'])==REGIONS[region] and tuple(b['rect'])==REGIONS[region],'XCV rectangle changed')
            need(b['wtk_index']==a['wtk_index']+84,'XCV WTK index spacing changed')
            need(a['timestamp_utc']==anchor,'XCV hour-zero date changed')
            need(a['wtk_index']>7223 and a['wtk_index']+167<61368,'WTK selected week outside dataset')
            for e in (a,b):
                need(e['filename'] in xcv_hash, 'XCV anchor filename is not SHA-indexed')
    civ=root/CIV
    c.hashed_files(civ,('locality_aggregate.csv','locality_dimension.csv','daily_locality.csv','locality_report.md','locality_manifest.json'))
    print('[CIV-CLOSEOUT] PASS: completed first week locality report and five SHA256 entries',flush=True)
    print('[FROZEN] PASS: XCI development regions/weeks, XCV 16 anchors and original Part C script SHA256',flush=True)
    print('[LOCKED] PASS: no external-test region is selectable',flush=True)
    return c,lookup

def bind_week(root,base,lookup,region,anchor):
    need(region in REGIONS and anchor in WEEKS,'Non-development or non-primary week refused')
    need((region,anchor)!=COMPLETED,'Original completed Gulf February week is immutable; use Part C verifier')
    start=lookup[(region,anchor,0)]['wtk_index']
    tag=anchor[:10].replace('-','');src=week_name(region,anchor)
    # This loaded module is an in-memory copy. The archived Part C file remains untouched.
    base.REGION=region
    base.RECT=REGIONS[region]
    base.ANCHOR=anchor
    base.DATE=tag
    base.START_IDX=start
    base.EXPECT_XCV={f'dev2010_{region}_{tag}_h{i:03d}.npz' for i in (0,84)}
    base.SRC=src
    base.OUT=str(BASE/src)
    base.STAGING=str(BASE/('.'+src+'.staging'))
    base.SCHEMA='phase7_cv_frozen_dev2010_regionweek_sources_v1'
    need(base.N==168 and base.DATA_KEYS=={'hr_uv','mr_uv','grid_rect','wtk_index','timestamp_utc','region'},'Original Part C source schema changed')
    need(base.START_IDX+167<61368,'WTK index overflow')
    need(base.timestamp(0)==anchor and base.timestamp(84)==lookup[(region,anchor,84)]['timestamp_utc'],'Hour identity changed')
    return src

def bound_sources(root,c):
    # Silence Part C's hardcoded February diagnostic lines, then print the actual bound region.
    with contextlib.redirect_stdout(io.StringIO()):fp,anchors=c.frozen(root)
    fp=dict(fp)
    fp['part_cv_controller_sha256']=sha(Path(__file__).resolve())
    fp['part_c_archived_sha256']=C_SHA
    need({0,84}==set(anchors),'Expected exactly two preserved XCV anchors')
    print(f'[SCOPE] PASS: {c.REGION} {c.ANCHOR} 168h, WTK {c.START_IDX}..{c.START_IDX+167}; 2 SHA-pinned anchors, no locked external regions',flush=True)
    return fp,anchors

def run_one(root,c,lookup,region,anchor,mode,max_new):
    src=bind_week(root,c,lookup,region,anchor)
    fp,cached=bound_sources(root,c)
    out=root/c.OUT;stage=root/c.STAGING
    if mode=='preflight':
        need(not out.exists(),f'Already published: {out}; use --mode verify')
        c.budget(root,stage)
        if stage.exists():
            import numpy as np
            c.read_progress(stage,fp,np,cached,full_check=True)
        print(f'[PREFLIGHT] PASS: {src}; no HSDS requests or output writes',flush=True)
    elif mode=='probe':
        import h5pyd
        orig=c.load_module(root/c.ORIGINAL,'phase7_cv_pinned_orig')
        c.remote_probe(orig,h5pyd)
    elif mode=='extract':
        c.extract(root,fp,cached,max_new)
    elif mode=='verify':
        need(not stage.exists(),f'Partial source stage remains: {stage}')
        c.verify(root,fp,cached)
    else:raise RuntimeError('Invalid execution mode')

def selftest():
    need(len(weeks())==7 and len(set(weeks()))==7,'Expected exactly seven remaining primary development weeks')
    need(all(r in REGIONS and a in WEEKS and (r,a)!=COMPLETED for r,a in weeks()),'Unexpected selected region/week')
    need(len(set(week_name(r,a) for r,a in weeks()))==7,'Source output collision')
    import numpy as np
    need(int(np.prod((500,500,2)))==500000,'Reference dimensional sanity failed')
    # Static consistency: archived source module file and its SHA remain pinned separately.
    print('[SELFTEST] PASS: seven frozen primary 2010 weeks, distinct immutable destinations, no locked region',flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',required=True,choices=('selftest','inventory','preflight','probe','extract','verify','preflight-all','extract-all','verify-all'))
    parser.add_argument('--repo-root',default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument('--region',choices=tuple(REGIONS))
    parser.add_argument('--week-start',choices=WEEKS)
    parser.add_argument('--max-new-hours',type=int)
    args=parser.parse_args()
    if args.mode=='selftest':selftest();return
    need(args.max_new_hours is None or (args.mode in ('extract','extract-all') and args.max_new_hours>=1),'--max-new-hours only for extract, >=1')
    if args.mode.endswith('-all') or args.mode=='inventory':
        need(args.region is None and args.week_start is None,'Region/week not permitted for all-week and inventory modes')
    else:
        need(args.region is not None and args.week_start is not None,'Select --region and --week-start explicitly')
    root=Path(args.repo_root).expanduser().resolve();need(root.is_dir(),f'Not a repo: {root}')
    c,lookup=inventory(root)
    if args.mode=='inventory':
        for region,anchor in weeks():
            target=root/BASE/week_name(region,anchor)
            stage=root/BASE/('.'+week_name(region,anchor)+'.staging')
            state='SEALED' if target.exists() else 'STAGED' if stage.exists() else 'PENDING'
            print(f'[WEEK] {region} {anchor} WTK={lookup[(region,anchor,0)]["wtk_index"]} status={state}',flush=True)
        return
    if args.mode.endswith('-all'):
        unit=args.mode.removesuffix('-all')
        for region,anchor in weeks():
            target=root/BASE/week_name(region,anchor)
            if unit in ('extract','preflight') and target.exists():
                print(f'[SKIP-SEALED] {region} {anchor}: independently verifying immutable output',flush=True)
                run_one(root,c,lookup,region,anchor,'verify',None)
            else:run_one(root,c,lookup,region,anchor,unit,args.max_new_hours)
        sealed=sum((root/BASE/week_name(r,a)).is_dir() for r,a in weeks())
        if unit=='verify':need(sealed==7,'Expected all seven development weeks sealed after verify-all')
        if unit=='extract' and args.max_new_hours is None:
            need(sealed==7,'One or more development source weeks not fully extracted')
        label='ALL-PARTIAL' if unit=='extract' and sealed<7 else 'ALL-'+unit.upper()
        print(f'[{label}] PASS: {sealed}/7 sealed source weeks; seven development-only region-weeks checked; zero locked regions',flush=True)
    else:run_one(root,c,lookup,args.region,args.week_start,args.mode,args.max_new_hours)

if __name__=='__main__':
    try:main()
    except Exception as exc:
        print('[PART-CV] ERROR',type(exc).__name__+':',str(exc),file=sys.stderr,flush=True)
        sys.exit(1)
