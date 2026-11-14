#!/usr/bin/env python3
"""Phase7 Part C: guarded, restartable, DEV ONLY first 168h WTK source week.

All output is source u/v data only; never performs model inference or opens
locked-region field arrays. In preflight, no HSDS or output writes are made.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time

RELBASE='ttk_runs_fixed/topology_finetuning'
XCV=f'{RELBASE}/phase7_xcv_development_seasonal16_20261018'
XCIX=f'{RELBASE}/phase7_xcix_dev_seasonal16_locality_20261018'
SPLIT=f'{RELBASE}/phase7_lxxxix_locked_region_split_20261015'
XCI=f'{RELBASE}/phase7_xci_frozen_evaluation_protocol_20261015'
SOURCE='phase7_xcv_update/scripts/phase7_xcv_dev_seasonal16.py'
ORIGINAL='phase7_lxxxvii_update/scripts/phase7_wtk_wind_screen.py'
DATE='20100201'
ANCHOR='2010-02-01T00:00:00'
REGION='gulf_inland'
RECT=(205,705,1471,1971)
START_IDX=27048
N=168
SRC='phase7_c_dev_gulf_inland_feb2010_week168_20261018'
OUT=f'{RELBASE}/{SRC}'
STAGING=f'{RELBASE}/.{SRC}.staging'
SCHEMA='phase7_c_dev_first_fullweek_source_v1'
DATA_KEYS={'hr_uv','mr_uv','grid_rect','wtk_index','timestamp_utc','region'}
RAW_HR=500*500*2*8
RAW_MR=100*100*2*8
RAW_WEEK=(RAW_HR+RAW_MR)*N
MIN_FREE=max(2*1024**3, 3*RAW_WEEK)
EXPECT_XCV={f'dev2010_{REGION}_{DATE}_h{i:03d}.npz' for i in (0,84)}


def require(v,msg):
    if not v: raise RuntimeError(msg)

def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1<<20),b''): h.update(block)
    return h.hexdigest()

def hashed_files(d, expected=None):
    d=Path(d); p=d/'SHA256SUMS'
    require(p.is_file(),f'Missing manifest: {p}')
    got={}
    for line in p.read_text().splitlines():
        m=re.fullmatch(r'([0-9a-f]{64})  ([^\n]+)',line)
        require(m is not None,f'Malformed SHA256 line in {p}')
        sha,rel=m.groups()
        path=(d/rel).resolve()
        require(path.is_relative_to(d.resolve()) and path.is_file(),f'Missing/unsafe file: {rel}')
        require(rel not in got and digest(path)==sha,f'SHA256 failed or duplicate: {rel}')
        got[rel]=sha
    require(bool(got),f'Empty SHA manifest: {p}')
    if expected is not None: require(set(got)==set(expected),f'Unexpected manifest entries: {p}')
    return got

def load_module(file, name):
    spec=importlib.util.spec_from_file_location(name,str(file))
    require(spec is not None and spec.loader is not None,f'Module not available: {file}')
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module

def filename(i):
    return f'dev2010_{REGION}_{DATE}_h{i:03d}.npz'

def timestamp(i):
    return (datetime.fromisoformat(ANCHOR)+timedelta(hours=i)).isoformat()

def expected_hour(i):
    return {'hour':i,'wtk_index':START_IDX+i,'timestamp_utc':timestamp(i),
            'region':REGION,'grid_rect':list(RECT),'filename':filename(i),
            'source':'XCV_CACHED_BITWISE' if i in (0,84) else 'HSDS_WTK_100M'}

def validate_npz(path,i,np):
    with np.load(path,allow_pickle=False) as z:
        require(set(z.files)==DATA_KEYS,f'NPZ schema changed: {path}')
        hr=z['hr_uv'];mr=z['mr_uv']
        require(hr.shape==(500,500,2) and mr.shape==(100,100,2),f'Wind array shapes incorrect: {path}')
        require(hr.dtype==mr.dtype==np.float64,f'Wind array dtype incorrect: {path}')
        require(bool(np.isfinite(hr).all() and np.isfinite(mr).all()),f'Nonfinite wind array: {path}')
        require(tuple(z['grid_rect'].tolist())==RECT,f'Grid rectangle mismatch: {path}')
        require(str(z['timestamp_utc'])==timestamp(i) and int(z['wtk_index'])==START_IDX+i
            and str(z['region'])==REGION,f'WTK metadata mismatch: {path}')
        pooled=hr.reshape(100,5,100,5,2).mean(axis=(1,3))
        require(float(np.max(np.abs(pooled-mr)))<=1e-10,f'MR pooling mismatch: {path}')
        speed=np.hypot(hr[...,0],hr[...,1])
        return {'speed_mean_mps':float(speed.mean()),'speed_p99_mps':float(np.percentile(speed,99)),
                'hr_uv_bytes':int(hr.nbytes),'mr_uv_bytes':int(mr.nbytes)}

def frozen(root, check_all_xcv=True):
    import numpy as np
    sd=root/SPLIT; pd=root/XCI; xcv=root/XCV; xcix=root/XCIX
    hashed_files(sd,('locked_region_split.json','locked_region_split.md'))
    hashed_files(pd,('evaluation_protocol.json','evaluation_protocol.md'))
    hashed_files(xcix,('locality_aggregate.csv','locality_dimension.csv',
                'region_week_locality.csv','locality_report.md','locality_manifest.json'))
    hashes=hashed_files(xcv)
    require(len(hashes)==18,'Expected 16 XCV fields + 2 result documents')
    js=json.loads((sd/'locked_region_split.json').read_text())
    proto=json.loads((pd/'evaluation_protocol.json').read_text())['protocol']
    require(js['status']=='FROZEN_BEFORE_MODEL_SCORES','Geographic split unfrozen')
    split=js['frozen_split']
    require(tuple(split['development_regions'])==('gulf_inland','southwest_borderland'),'Dev role drift')
    require(tuple(split['locked_external_test_regions'])==('upper_midwest','pacific_inland'),'Locked roles drift')
    require(tuple(js['region_metadata'][REGION]['grid_rect'])==RECT,'Frozen rectangle drift')
    require(ANCHOR in split['primary_windows_utc'] and split['hours_per_window']==168,'Frozen seven-day window drift')
    require(ANCHOR in proto['times']['primary_week_start_utc'] and proto['times']['hourly_fields_per_week']==168,'XCI window drift')
    require(tuple(proto['geographic_roles']['development'])==('gulf_inland','southwest_borderland'),'XCI dev role drift')
    require(tuple(proto['geographic_roles']['locked_external_test'])==('upper_midwest','pacific_inland'),'XCI test role drift')
    xcv_code=root/SOURCE; original_code=root/ORIGINAL
    require(xcv_code.is_file() and original_code.is_file(),'Archived XCV or LXXXVII source missing')
    xcv_manifest=json.loads((xcv/'seasonal16_manifest.json').read_text())
    require(xcv_manifest['source_sha256']['XCV_script_sha256']==digest(xcv_code),'XCV script source changed')
    xcv_mod=load_module(xcv_code,'phase7_c_import_xcv_verified')
    if check_all_xcv:
        items,lookup,fprints=xcv_mod.sources(root)
        xcv_mod.verify(root,xcv,items,lookup,fprints)
    cached={}
    by_name={x['filename']:x for x in xcv_manifest['fields']}
    for i in (0,84):
        name=filename(i);require(name in hashes and name in by_name,f'Missing XCV week anchor {name}')
        require(by_name[name]['wtk_index']==START_IDX+i and by_name[name]['timestamp_utc']==timestamp(i),'XCV source index changed')
        p=xcv/name;validate_npz(p,i,np)
        cached[i]={'file':p,'sha256':digest(p)}
    fp={'lxxxix_split_sha256':digest(sd/'locked_region_split.json'),
        'xci_protocol_sha256':digest(pd/'evaluation_protocol.json'),
        'xcv_manifest_sha256':digest(xcv/'seasonal16_manifest.json'),
        'xcix_manifest_sha256':digest(xcix/'locality_manifest.json'),
        'lxxxvii_script_sha256':digest(original_code),'xcv_script_sha256':digest(xcv_code),
        'part_c_script_sha256':digest(Path(__file__).resolve())}
    print('[FROZEN] PASS: XCI protocol, LXXXIX geographic split, verified XCV fields, final XCIX lineage',flush=True)
    print('[SCOPE] PASS: gulf_inland 2010-02-01 UTC hours 0..167; 168 dev-only; 2 XCV anchors; zero locked access',flush=True)
    return fp,cached

def budget(root, staging):
    usage=shutil.disk_usage((root/STAGING).parent)
    mib=1024**2;gib=1024**3
    predicted=RAW_HR*N*6
    print(f'[BUDGET] HR+MR raw week={RAW_WEEK/mib:.2f} MiB; per-hour={ (RAW_HR+RAW_MR)/mib:.2f} MiB',flush=True)
    print(f'[BUDGET] six full-resolution model output arrays (future, uncompressed)={predicted/gib:.2f} GiB',flush=True)
    print(f'[BUDGET] free_disk={usage.free/gib:.2f} GiB; required_free_for_source={MIN_FREE/gib:.2f} GiB (conservative)',flush=True)
    print('[BUDGET] PD/TTK, checkpoints, TFRecords, runtime and network costs are NOT measured here; avoid inferring a wall time.',flush=True)
    require(usage.free>=MIN_FREE,f'Insufficient free disk for conservative source pilot: {usage.free} bytes')
    with open('/proc/meminfo') as f: info=f.read()
    m=re.search(r'^MemAvailable:\s+(\d+) kB',info,re.M)
    if m:
        avail=int(m.group(1))*1024
        print(f'[BUDGET] Linux MemAvailable={avail/gib:.2f} GiB; streaming one hour at a time',flush=True)
        require(avail>=2*gib,'Insufficient available memory (2GiB minimum)')
    print('[BUDGET] PASS: source-only storage and memory readiness; downstream model/PD feasibility NOT yet established',flush=True)

def config(fp):
    return {'schema_version':1,'scope':SCHEMA,'source_sha256':fp,'region':REGION,
        'year':2010,'week_start_utc':ANCHOR,'start_wtk_index':START_IDX,
        'region_rect':list(RECT),'hours':N,'reuse_existing_hours':[0,84],
        'dataset':'/nrel/wtk-us.h5','wind_datasets':['windspeed_100m','winddirection_100m']}

def atomic_text(path,s):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w',dir=path.parent,prefix='.tmp-',delete=False) as f:
        tmp=Path(f.name);f.write(s);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

def atomic_npz(dest, i, hr, mr, np):
    dest.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=dest.parent,prefix='.tmp-',suffix='.npz',delete=False) as f:
        tmp=Path(f.name)
    try:
        np.savez_compressed(tmp,hr_uv=hr,mr_uv=mr,grid_rect=np.asarray(RECT),wtk_index=np.asarray(START_IDX+i),
                            timestamp_utc=np.asarray(timestamp(i)),region=np.asarray(REGION))
        with tmp.open('rb') as f:os.fsync(f.fileno())
        os.replace(tmp,dest)
    finally:
        tmp.unlink(missing_ok=True)

def read_progress(stage,fp,np,cached,full_check=False):
    cfg=stage/'run_config.json'; ledger=stage/'progress.json'; fields=stage/'fields'
    require(cfg.is_file() and ledger.is_file(),'Staging config/ledger incomplete, stop for inspection')
    require(json.loads(cfg.read_text())==config(fp),'Staging source code/frozen source/version changed; no resume allowed')
    log=json.loads(ledger.read_text());require(isinstance(log,dict),'Progress must be object')
    names=set()
    for index,rec in log.items():
        require(index.isdigit() and str(int(index))==index and 0<=int(index)<N,'Invalid progress hour key')
        i=int(index);expected=expected_hour(i)
        require({k:rec[k] for k in expected}==expected,f'Hour provenance mismatch: {i}')
        file=fields/filename(i)
        require(file.is_file() and digest(file)==rec['sha256'],f'Incomplete/altered recorded hour: {i}')
        if full_check:
            stats=validate_npz(file,i,np)
            require(rec['stats']==stats,f'Numerical metadata drift at hour {i}')
        if i in cached:require(rec['sha256']==cached[i]['sha256'],f'Cached-XCV anchor drift at {i}')
        names.add(filename(i))
    require(set(x.name for x in fields.iterdir())==names,
        'Orphan/unrecorded hour file or incomplete temporary file; fail closed and inspect staging before resume')
    print(f'[RESUME] PASS: {len(log)}/{N} recorded hours, immutable source/progress hashes',flush=True)
    return log

def init_stage(stage,fp):
    stage.mkdir(parents=True,exist_ok=False);(stage/'fields').mkdir()
    atomic_text(stage/'run_config.json',json.dumps(config(fp),indent=2,sort_keys=True)+'\n')
    atomic_text(stage/'progress.json','{}\n')

def remote_probe(original, h5pyd):
    with h5pyd.File('/nrel/wtk-us.h5','r') as h:
        ws=h['windspeed_100m'];wd=h['winddirection_100m'];clock=h['datetime']
        require(tuple(ws.shape)==(61368,1602,2976) and tuple(wd.shape)==tuple(ws.shape), 'WTK full size changed')
        ts=clock[START_IDX:START_IDX+N]
        require(len(ts)==N,'Unable to read all 168 WTK timestamps')
        for i,stamp in enumerate(ts):
            require(original.parse_time(stamp)==datetime.fromisoformat(timestamp(i)),
                f'WTK datetime continuity failed: hour={i}, index={START_IDX+i}')
        print(f'[HSDS-PROBE] PASS: 168/168 UTC clock entries, index [{START_IDX},{START_IDX+N-1}], 500x500 region bound; no wind arrays read',flush=True)

def write_outputs(stage,fp,log,write_hashes=True):
    require(set(log)=={str(i) for i in range(N)},'Cannot seal incomplete week')
    if write_hashes:
        for i in range(N): require((stage/'fields'/filename(i)).is_file(),'Missing hour field')
    rows=[dict(log[str(i)]) for i in range(N)]
    manifest={'schema_version':1,'scope':SCHEMA,'source_sha256':fp,'run_config':config(fp),
              'field_count':N,'cached':2,'remote':N-2,'fields':rows,
              'limitations':['development-only source week','no model predictions or topological evaluation',
                             'original pretrained PhIRE membership unknown','not external generalization']}
    atomic_text(stage/'fullweek_manifest.json',json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    cols=['hour','wtk_index','timestamp_utc','region','filename','source','sha256','speed_mean_mps','speed_p99_mps']
    import io
    stream=io.StringIO();w=csv.DictWriter(stream,fieldnames=cols,lineterminator='\n');w.writeheader()
    for item in rows:
        rec={k:item[k] for k in ('hour','wtk_index','timestamp_utc','region','filename','source','sha256')}
        rec.update({k:item['stats'][k] for k in ('speed_mean_mps','speed_p99_mps')})
        w.writerow(rec)
    atomic_text(stage/'hourly_index.csv',stream.getvalue())
    report=(f'# Phase 7 Part C — complete development-only WTK source week\n\n'
        f'- Frozen region: `{REGION}`, rectangle `{RECT}` (WTK rows/columns).\n'
        f'- UTC: `{ANCHOR}` through `{timestamp(N-1)}` inclusive; WTK indices `{START_IDX}` to `{START_IDX+N-1}`.\n'
        f'- 168 verified native HR `500×500×2` and factor-five MR `100×100×2` `[u,v]` float64 fields.\n'
        f'- Original XCV hours 0 and 84 copied bit-for-bit; 166 additional hourly WTK slices.\n'
        f'- Raw uncompressed HR+MR payload {RAW_WEEK:,} bytes ({RAW_WEEK/1024**2:.2f} MiB).\n'
        f'- Wind-speed and pooling validations applied to each hour; precise SHA256 provenance in manifest.\n'
        f'- **Source-data only.** No six-model predictions, PD, locality, training or locked external-region access.\n'
        f'- No conclusions about pretrained PhIRE data exposure, seasonal robustness, or independent hourly inference.\n')
    atomic_text(stage/'fullweek_report.md',report)
    files=[f'fields/{filename(i)}' for i in range(N)]+[
        'run_config.json','progress.json','fullweek_manifest.json','hourly_index.csv','fullweek_report.md']
    if write_hashes:
        atomic_text(stage/'SHA256SUMS',''.join(f'{digest(stage/p)}  {p}\n' for p in files))
    return files

def extract(root,fp,cached,max_new):
    import numpy as np
    import h5pyd
    out=root/OUT;stage=root/STAGING
    require(not out.exists(),f'Completed output already exists: {out}; use --mode verify')
    budget(root,stage)
    original=load_module(root/ORIGINAL,'phase7_c_original_extract')
    remote_probe(original,h5pyd)
    lock=(root/OUT).parent/('.'+SRC+'.lock')
    lock.parent.mkdir(parents=True,exist_ok=True)
    with lock.open('a+') as l:
        fcntl.flock(l.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        require(not out.exists(),'Output appeared concurrently; refusal')
        if not stage.exists():init_stage(stage,fp)
        log=read_progress(stage,fp,np,cached,full_check=True)
        remaining=[i for i in range(N) if str(i) not in log]
        pending_remote=[i for i in remaining if i not in cached]
        count_new=0;start=time.monotonic()
        with h5pyd.File('/nrel/wtk-us.h5','r') as h:
            ws=h['windspeed_100m'];wd=h['winddirection_100m'];clock=h['datetime']
            require(tuple(ws.shape)==(61368,1602,2976) and tuple(wd.shape)==tuple(ws.shape),'WTK wind dimensions changed')
            for i in remaining:
                if i in cached:
                    shutil.copy2(cached[i]['file'],stage/'fields'/filename(i));origin='XCV_CACHED_BITWISE'
                else:
                    if max_new is not None and count_new>=max_new:break
                    require(original.parse_time(clock[START_IDX+i])==datetime.fromisoformat(timestamp(i)),f'WTK time mismatch at {i}')
                    y0,y1,x0,x1=RECT
                    speed=np.asarray(ws[START_IDX+i,y0:y1,x0:x1],dtype=np.float64)
                    direction=np.asarray(wd[START_IDX+i,y0:y1,x0:x1],dtype=np.float64)
                    require(speed.shape==direction.shape==(500,500),'WTK slice dimension mismatch')
                    require(bool(np.isfinite(speed).all() and np.isfinite(direction).all() and
                         (speed>=0).all() and (direction>=0).all() and (direction<=360).all()),f'Invalid WTK slice hour {i}')
                    u,v=original.wind_components(speed,direction,np)
                    hr=np.stack((u,v),axis=-1)
                    mr=hr.reshape(100,5,100,5,2).mean(axis=(1,3))
                    atomic_npz(stage/'fields'/filename(i),i,hr,mr,np)
                    origin='HSDS_WTK_100M';count_new+=1
                f=stage/'fields'/filename(i)
                stats=validate_npz(f,i,np)
                expected=expected_hour(i)
                require(origin==expected['source'],'Source designation changed')
                record={**expected,'sha256':digest(f),'stats':stats}
                if i in cached:require(record['sha256']==cached[i]['sha256'],'XCV cached bytes drift')
                log[str(i)]=record
                atomic_text(stage/'progress.json',json.dumps(log,indent=2,sort_keys=True)+'\n')
                print(f'[HOUR] {len(log):03d}/{N} UTC={timestamp(i)} WTK={START_IDX+i} {origin} SHA256_OK elapsed_s={time.monotonic()-start:.1f}',flush=True)
        if len(log)<N:
            print(f'[PARTIAL-GATE] PASS: {len(log)}/{N} safely recorded; {N-len(log)} remain; resume with same --mode extract (no overwrite)',flush=True)
            return
        read_progress(stage,fp,np,cached,full_check=True)
        write_outputs(stage,fp,log)
        require(not out.exists(),'Output appeared before publish; refusing overwrite')
        stage.rename(out)
        print(f'[SOURCE-WEEK-GATE] PASS: {N} valid hours, two bitwise XCV anchors, 166 WTK source slices; immutable result {out}',flush=True)

def verify(root,fp,cached):
    import numpy as np
    out=root/OUT
    require(out.is_dir(),'Completed output absent; extraction may still be in staging')
    files=[f'fields/{filename(i)}' for i in range(N)]+[
        'run_config.json','progress.json','fullweek_manifest.json','hourly_index.csv','fullweek_report.md']
    hashed_files(out,files)
    require(json.loads((out/'run_config.json').read_text())==config(fp),'Frozen run config altered')
    manifest=json.loads((out/'fullweek_manifest.json').read_text())
    require(manifest['schema_version']==1 and manifest['scope']==SCHEMA and manifest['source_sha256']==fp,'Source lineage mismatch')
    require(manifest['field_count']==N and manifest['cached']==2 and manifest['remote']==N-2,'Count mismatch')
    require(manifest['run_config']==config(fp) and len(manifest['fields'])==N,'Plan mismatch')
    rows=manifest['fields']
    progress=json.loads((out/'progress.json').read_text())
    require(progress=={str(i):rows[i] for i in range(N)},'Saved progress ledger differs from fullweek manifest')
    for i,rec in enumerate(rows):
        exp=expected_hour(i)
        require({k:rec[k] for k in exp}==exp,f'Wrong field origin/time: {i}')
        path=out/'fields'/filename(i)
        require(rec['sha256']==digest(path),f'Incorrect field checksum: {i}')
        require(rec['stats']==validate_npz(path,i,np),f'Numerical field validation failed: {i}')
        if i in cached:require(rec['sha256']==cached[i]['sha256'],f'XCV anchor changed: {i}')
    # Independently rerender all source-only sidecar metadata into a temporary directory.
    with tempfile.TemporaryDirectory() as td:
        d=Path(td)
        write_outputs(d,fp,{str(i):rows[i] for i in range(N)},write_hashes=False)
        for name in ['fullweek_manifest.json','hourly_index.csv','fullweek_report.md']:
            require((d/name).read_bytes()==(out/name).read_bytes(),f'Sidecar report changed: {name}')
    print(f'[VERIFY-GATE] PASS: {N} hourly field arrays and metadata; XCV bitwise anchors; {len(files)} SHA256 files; no HSDS reads or locked regions',flush=True)

def selftest():
    import numpy as np
    require(timestamp(0)==ANCHOR and timestamp(167)=='2010-02-07T23:00:00','Week endpoint mismatch')
    require(START_IDX+N-1==27215,'WTK week end mismatch')
    require(len({filename(i) for i in range(N)})==N,'Duplicate hour filename')
    uv=np.zeros((500,500,2),dtype=np.float64);uv[...,0]=3;uv[...,1]=4
    mr=uv.reshape(100,5,100,5,2).mean(axis=(1,3))
    with tempfile.TemporaryDirectory() as td:
        p=Path(td)/filename(0)
        atomic_npz(p,0,uv,mr,np)
        a=validate_npz(p,0,np)
        require(abs(a['speed_mean_mps']-5)<1e-12,'Synthetic speed incorrect')
        d=Path(td);atomic_text(d/'SHA256SUMS',f'{digest(p)}  {p.name}\n')
        hashed_files(d,[p.name]);p.open('ab').write(b'tamper')
        try:hashed_files(d,[p.name])
        except RuntimeError:pass
        else:raise RuntimeError('Tampering undetected')
    print('[SELFTEST] PASS: 168 monotone hour identities, MR pooling, atomic writer, checksum-tamper gate',flush=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--mode',choices=('selftest','preflight','probe','extract','verify'),required=True)
    ap.add_argument('--repo-root',default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument('--max-new-hours',type=int,default=None,help='Optional run limit on NEW HSDS hours for a safe resumable first batch; omit to finish all hours.')
    args=ap.parse_args()
    if args.mode=='selftest':selftest();return
    root=Path(args.repo_root).expanduser().resolve()
    require(root.is_dir(),f'Missing repo: {root}')
    if args.max_new_hours is not None:
        require(args.mode=='extract' and args.max_new_hours>=1,'--max-new-hours permitted only for extract, >=1')
    fp,cached=frozen(root)
    stage=root/STAGING;out=root/OUT
    if args.mode=='preflight':
        require(not out.exists(),f'Completed full-week exists; use --mode verify: {out}')
        budget(root,stage)
        if stage.exists():
            import numpy as np
            read_progress(stage,fp,np,cached,full_check=True)
        print('[PREFLIGHT] PASS: read-only offline; frozen first development week, resource allowance, resumable stage; no WTK calls',flush=True)
    elif args.mode=='probe':
        original=load_module(root/ORIGINAL,'phase7_c_probe_orig')
        import h5pyd
        remote_probe(original,h5pyd)
    elif args.mode=='extract':
        extract(root,fp,cached,args.max_new_hours)
    elif args.mode=='verify':verify(root,fp,cached)

if __name__=='__main__':
    try:main()
    except Exception as e:
        print('[PART-C] ERROR',type(e).__name__+':',e,file=sys.stderr,flush=True)
        sys.exit(1)
