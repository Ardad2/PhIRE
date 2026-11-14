#!/usr/bin/env python3
"""Phase 7 XCV: 2010 DEVELOPMENT ONLY seasonal 16-field source-data pilot.

2 regions x 4 predeclared seasonal weeks x offsets (0,84) hours = 16 fields.
Existing four Feb/Aug offset-0 LXXXVII previews are reused byte-for-byte;
12 other fields are fetched read-only from HSDS and checked against the
previously frozen LXXXVII independent screening measurements. No inference.

This is a feasibility sampling tranche, NOT the frozen 168h/week benchmark.
"""
from __future__ import annotations
import argparse, csv, datetime as dt, hashlib, importlib.util, json, os, re, shutil, sys, tempfile
from pathlib import Path

DEFAULT_ROOT=Path(__file__).resolve().parents[2]
SCREEN='ttk_runs_fixed/topology_finetuning/phase7_lxxxvii_wind_screen_20261015'
SPLIT='ttk_runs_fixed/topology_finetuning/phase7_lxxxix_locked_region_split_20261015'
PROTOCOL='ttk_runs_fixed/topology_finetuning/phase7_xci_frozen_evaluation_protocol_20261015'
OUTPUT='ttk_runs_fixed/topology_finetuning/phase7_xcv_development_seasonal16_20261018'
ORIGINAL='phase7_lxxxvii_update/scripts/phase7_wtk_wind_screen.py'
DEV=('gulf_inland','southwest_borderland')
LOCKED=('upper_midwest','pacific_inland')
ANCHORS=('2010-02-01T00:00:00','2010-05-01T00:00:00','2010-08-01T00:00:00','2010-11-01T00:00:00')
OFFSETS=(0,84)
PREVIEW_DATES=('20100201','20100801')
EXPECTED_RECT={'gulf_inland':(205,705,1471,1971),'southwest_borderland':(173,673,572,1072)}


def need(condition,message):
    if not condition:raise RuntimeError(message)

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for buf in iter(lambda:f.read(1024*1024),b''):h.update(buf)
    return h.hexdigest()

def hashes(folder,expected=None):
    p=folder/'SHA256SUMS';need(p.is_file(),f'Missing checksums {p}')
    found={}
    for line in p.read_text().splitlines():
        m=re.fullmatch(r'([0-9a-f]{64})  ([^\n]+)',line)
        need(m is not None,f'Malformed SHA256 entry in {p}')
        checksum,rel=m.groups()
        target=(folder/rel).resolve()
        need(target.is_relative_to(folder.resolve()) and target.is_file(),f'Missing/unsafe file: {rel}')
        need(rel not in found and sha(target)==checksum,f'Hash failed or duplicate: {rel}')
        found[rel]=checksum
    if expected is not None:need(set(found)==set(expected),f'Unexpected SHA256 entries in {p}')
    return found

def orig_module(root):
    p=root/ORIGINAL
    need(p.is_file(),f'Missing screened/source module: {p}')
    spec=importlib.util.spec_from_file_location('historical_wind_screen_xcv',p)
    obj=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj

def filename(region,start,offset):
    return f'dev2010_{region}_{start[:10].replace("-","")}_h{offset:03d}.npz'

def validate_npz(path,reg,idx,stamp,rect,reference_row,np):
    with np.load(path,allow_pickle=False) as z:
        need(set(z.files)=={'hr_uv','mr_uv','grid_rect','wtk_index','timestamp_utc','region'},f'NPZ schema mismatch: {path}')
        uv=z['hr_uv'];mr=z['mr_uv']
        need(uv.shape==(500,500,2) and mr.shape==(100,100,2),f'UV shape mismatch: {path}')
        need(uv.dtype==np.float64 and mr.dtype==np.float64,f'UV dtype mismatch: {path}')
        need(np.isfinite(uv).all() and np.isfinite(mr).all(),f'Nonfinite preview: {path}')
        need(tuple(z['grid_rect'].tolist())==tuple(rect),f'Grid rect mismatch: {path}')
        need(int(z['wtk_index'])==idx and str(z['timestamp_utc'])==stamp and str(z['region'])==reg,f'Region/UTC mismatch: {path}')
        down=uv.reshape(100,5,100,5,2).mean(axis=(1,3))
        need(float(np.max(np.abs(down-mr)))<=1e-10,f'Pooling/parity failure: {path}')
        speed=np.hypot(uv[:,:,0],uv[:,:,1]);mean=float(speed.mean());p99=float(np.percentile(speed,99))
        # WTK native [u,v] are reconstructed from speed/direction, so their
        # magnitudes must reproduce the earlier predeclared screening sample.
        for name,val in (('speed_mean',mean),('speed_p99',p99)):
            ref=float(reference_row[name]);need(abs(val-ref)<=2e-8,f'Screening covariate mismatch {name}: {path}: {val} vs {ref}')
    return {'speed_mean_mps':mean,'speed_p99_mps':p99}

def sources(root):
    screen=root/SCREEN;split=root/SPLIT;protocol=root/PROTOCOL
    sh=hashes(screen)
    need(len(sh)==11 and {'sample_screen.csv','screen_summary.json','wind_screen.md'}.issubset(sh),'Expected completed LXXXVII 11-file screening')
    hashes(split,('locked_region_split.json','locked_region_split.md'))
    hashes(protocol,('evaluation_protocol.json','evaluation_protocol.md'))
    s=json.loads((split/'locked_region_split.json').read_text())
    p=json.loads((protocol/'evaluation_protocol.json').read_text())
    need(s['status']=='FROZEN_BEFORE_MODEL_SCORES','Split has not been frozen')
    sp=s['frozen_split'];pp=p['protocol']
    need(tuple(sp['development_regions'])==DEV and tuple(sp['locked_external_test_regions'])==LOCKED,'Split roles modified')
    need(tuple(pp['geographic_roles']['development'])==DEV and tuple(pp['geographic_roles']['locked_external_test'])==LOCKED,'Protocol roles modified')
    need(tuple(sp['primary_windows_utc'])==ANCHORS and tuple(pp['times']['primary_week_start_utc'])==ANCHORS,'Seasonal anchors changed')
    need(sp['hours_per_window']==168,'Frozen week length not 168')
    screen_j=json.loads((screen/'screen_summary.json').read_text())
    original=root/ORIGINAL
    need(screen_j['source_fingerprints']['screen_script']['sha256']==sha(original),'LXXXVII extraction code no longer matches frozen screening')
    need(screen_j['n_samples']==64,'Screening sample count unexpected')
    with (screen/'sample_screen.csv').open(newline='') as f: rows=list(csv.DictReader(f))
    need(len(rows)==64,'Screening CSV invalid')
    lookup={}
    for row in rows:
        key=(row['region'],row['anchor_utc'],int(row['offset_hours']))
        need(key not in lookup,'Duplicated screening sample')
        lookup[key]=row
    items=[]
    for region in DEV:
        rect=tuple(s['region_metadata'][region]['grid_rect'])
        need(rect==EXPECTED_RECT[region],f'Frozen footprint changed for {region}')
        for start in ANCHORS:
            for off in OFFSETS:
                key=(region,start,off);need(key in lookup,f'Missing recorded screening sample: {key}')
                row=lookup[key];need(float(row['valid_fraction'])==1.,f'Predeclared sample nonfinite: {key}')
                timestamp=(dt.datetime.fromisoformat(start)+dt.timedelta(hours=off)).isoformat()
                need(row['timestamp_utc']==timestamp,'Timestamp mismatch against original screen')
                idx=int(row['wtk_index']);need(idx>7223,f'Overlapping 2007 fine-tuning: {idx}')
                date=start[:10].replace('-','')
                existing=off==0 and date in PREVIEW_DATES
                item={'region':region,'anchor_utc':start,'offset_hours':off,'timestamp_utc':timestamp,
                      'wtk_index':idx,'rect':list(rect),'filename':filename(region,start,off),
                      'cached_from_LXXXVII':existing,'screening_mean_mps':float(row['speed_mean']),
                      'screening_p99_mps':float(row['speed_p99'])}
                if existing:
                    existing_name=f'preview_{region}_{date}.npz'
                    need(existing_name in sh,f'Missing archived development preview: {existing_name}')
                    item['existing_file']=existing_name
                items.append(item)
    need(len(items)==16 and sum(x['cached_from_LXXXVII'] for x in items)==4,'Incorrect seasonal16 plan')
    fingerprints={'LXXXVII_screen_summary_sha256':sha(screen/'screen_summary.json'),
      'LXXXVII_sample_screen_sha256':sha(screen/'sample_screen.csv'),
      'LXXXVII_original_script_sha256':sha(original),
      'LXXXIX_frozen_split_sha256':sha(split/'locked_region_split.json'),
      'XCI_frozen_protocol_sha256':sha(protocol/'evaluation_protocol.json'),
      'XCV_script_sha256':sha(Path(__file__).resolve())}
    print('[SOURCES] PASS: frozen LXXXVII/LXXXIX/XCI source data, two development regions, four 2010 weeks',flush=True)
    print('[SCOPE] PASS: 16 predeclared development samples, 4 cached, 12 remote; zero locked-region predictions',flush=True)
    return items,lookup,fingerprints

def extract(root,out,items,lookup,fingerprints):
    import numpy as np
    import h5pyd
    original=orig_module(root)
    need(not out.exists(),f'Output exists: {out}; use --mode verify')
    out.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='.'+out.name+'.staging-',dir=out.parent))
    try:
        with h5pyd.File('/nrel/wtk-us.h5','r') as h:
            ws=h['windspeed_100m'];wd=h['winddirection_100m'];clock=h['datetime']
            need(tuple(ws.shape)==(61368,1602,2976) and tuple(wd.shape)==tuple(ws.shape),'WTK dataset shape mismatch')
            for n,item in enumerate(items,1):
                region=item['region'];idx=item['wtk_index'];stamp=item['timestamp_utc'];rect=item['rect']
                destination=stage/item['filename']
                if item['cached_from_LXXXVII']:
                    shutil.copy2(root/SCREEN/item['existing_file'],destination)
                    origin='cached'
                else:
                    origin='HSDS'
                    need(original.parse_time(clock[idx])==dt.datetime.fromisoformat(stamp),f'WTK time metadata changed for {idx}')
                    y0,y1,x0,x1=rect
                    a=np.asarray(ws[idx,y0:y1,x0:x1],dtype=np.float64)
                    b=np.asarray(wd[idx,y0:y1,x0:x1],dtype=np.float64)
                    need(a.shape==b.shape==(500,500),'Bad WTK wind slice shape')
                    need(np.isfinite(a).all() and np.isfinite(b).all() and (a>=0).all() and (b>=0).all() and (b<=360).all(),f'Bad wind data at {idx}')
                    u,v=original.wind_components(a,b,np)
                    hr=np.stack((u,v),axis=-1)
                    mr=hr.reshape(100,5,100,5,2).mean(axis=(1,3))
                    np.savez_compressed(destination,hr_uv=hr,mr_uv=mr,grid_rect=np.asarray(rect),
                        wtk_index=np.asarray(idx),timestamp_utc=np.asarray(stamp),region=np.asarray(region))
                validate_npz(destination,region,idx,stamp,rect,
                             lookup[(region,item['anchor_utc'],item['offset_hours'])],np)
                item['file_sha256']=sha(destination)
                print(f'[FIELD] {n:02d}/16 {region} {stamp} WTK={idx} origin={origin} VALID',flush=True)
        write_report(stage,items,fingerprints)
        files=[x['filename'] for x in items]+['seasonal16_manifest.json','seasonal16_report.md']
        (stage/'SHA256SUMS').write_text(''.join(f'{sha(stage/name)}  {name}\n' for name in files))
        need(not out.exists(),'Output appeared during extraction; refusing overwrite')
        stage.rename(out)
    except BaseException:
        shutil.rmtree(stage,ignore_errors=True)
        raise
    print('[EXTRACT-GATE] PASS: 16 verified dev 2010 fields; 12 new reads + 4 preserved previews; no models/test scores',flush=True)
    print('[OUTPUT]',out,flush=True)

def write_report(folder,items,fingerprints):
    record={'schema_version':1,'scope':'DEVELOPMENT_ONLY_SEASONAL16_SOURCE_EXTRACTION',
        'data_source':'/nrel/wtk-us.h5','sampling_rule':'2010 four frozen seasonal 168h windows, offsets 0 and 84 UTC, both development regions',
        'n_fields':16,'n_existing':4,'n_new':12,'source_sha256':fingerprints,'fields':items}
    (folder/'seasonal16_manifest.json').write_text(json.dumps(record,indent=2,sort_keys=True)+'\n')
    lines=['# Phase 7 XCV — sixteen-field 2010 development source expansion','',
     'Two fixed development regions × four primary 2010 weeks × two predeclared hours (0, 84).',
     '**This 16-field tranche is for pipeline/scaling checks, not the complete 168-hour-per-week evaluation.**','',
     '| Region | Season UTC | Offset h | WTK time index | Source |', '|---|---|---:|---:|---|']
    for x in items:
        lines.append(f"| {x['region']} | {x['anchor_utc'][:10]} | {x['offset_hours']} | {x['wtk_index']} | {'cached LXXXVII' if x['cached_from_LXXXVII'] else 'HSDS'} |")
    lines+=['','No model inference, topology metrics, new hyperparameter search, or external-test model scores.',
     'Each file stores 500×500×2 HR and 100×100×2 MR [u,v] with exact region, grid rectangle and UTC metadata.',
     'All extracted winds agree with previously frozen LXXXVII screening speed statistics within 2e-8 m/s.',
     'These locations are outside documented fine-tuning footprint; original pretrained-model WTK membership is unknown.','']
    (folder/'seasonal16_report.md').write_text('\n'.join(lines))

def verify(root,out,items,lookup,fingerprints):
    import numpy as np
    expected=[x['filename'] for x in items]+['seasonal16_manifest.json','seasonal16_report.md']
    hashes(out,expected)
    manifest=json.loads((out/'seasonal16_manifest.json').read_text())
    need(manifest['schema_version']==1 and manifest['scope']=='DEVELOPMENT_ONLY_SEASONAL16_SOURCE_EXTRACTION', 'Wrong output manifest scope')
    need(manifest['n_fields']==16 and manifest['n_existing']==4 and manifest['n_new']==12,'Wrong field counts')
    need(manifest['source_sha256']==fingerprints,'Upstream sources changed')
    observed=manifest['fields'];need(len(observed)==16,'Manifest incomplete')
    for item,seen in zip(items,observed):
        should={**item,'file_sha256':sha(out/item['filename'])}
        need(seen==should,f'Manifest field changed: {item["filename"]}')
        validate_npz(out/item['filename'],item['region'],item['wtk_index'],item['timestamp_utc'],item['rect'],
                     lookup[(item['region'],item['anchor_utc'],item['offset_hours'])],np)
        if item['cached_from_LXXXVII']:
            need(sha(out/item['filename'])==sha(root/SCREEN/item['existing_file']),f'Cached preview differs: {item["filename"]}')
    writeable_text=make_report_for_verifier(items,fingerprints)
    need((out/'seasonal16_report.md').read_text()==writeable_text,'Summary report altered')
    print('[VERIFY-GATE] PASS: exact 16 development source fields, LXXXVII screening parity, SHA256, 4 original previews, no TF or HSDS reads',flush=True)

def make_report_for_verifier(items,fingerprints):
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)
        write_report(path,items,fingerprints)
        return (path/'seasonal16_report.md').read_text()

def selftest():
    import numpy as np
    uv=np.zeros((500,500,2),dtype=np.float64);uv[:,:,0]=3;uv[:,:,1]=4
    mr=uv.reshape(100,5,100,5,2).mean((1,3))
    need(np.array_equal(mr[:,:,0],np.full((100,100),3.)),'Wrong MR vector scale')
    with tempfile.TemporaryDirectory() as t:
        d=Path(t);p=d/'demo.npz';np.savez_compressed(p,hr_uv=uv,mr_uv=mr,grid_rect=np.asarray([0,500,0,500]),
            wtk_index=np.asarray(27048),timestamp_utc=np.asarray('2010-02-01T00:00:00'),region=np.asarray('gulf_inland'))
        validate_npz(p,'gulf_inland',27048,'2010-02-01T00:00:00',[0,500,0,500],{'speed_mean':5,'speed_p99':5},np)
        (d/'SHA256SUMS').write_text(f'{sha(p)}  demo.npz\n');hashes(d,['demo.npz'])
        with p.open('ab') as f:f.write(b'altered')
        try:hashes(d,['demo.npz'])
        except RuntimeError:pass
        else:raise RuntimeError('Checksum tampering unrecognized')
    print('[SELFTEST] PASS: 500-to-100 pooled fields, checksum tamper rejection, numeric metadata/schema',flush=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--mode',choices=['selftest','preflight','extract','verify'],required=True)
    ap.add_argument('--repo-root',default=str(DEFAULT_ROOT))
    a=ap.parse_args()
    if a.mode=='selftest':selftest();return
    root=Path(a.repo_root).resolve();out=root/OUTPUT
    items,lookup,fingerprints=sources(root)
    if a.mode=='preflight':
        need(not out.exists(),'Output exists; use --mode verify')
        import numpy as np
        for x in items:
            if x['cached_from_LXXXVII']:
                validate_npz(root/SCREEN/x['existing_file'],x['region'],x['wtk_index'],x['timestamp_utc'],x['rect'],
                             lookup[(x['region'],x['anchor_utc'],x['offset_hours'])],np)
        print('[PREFLIGHT] PASS: four original previews, twelve planned read-only WTK slices; zero outputs written',flush=True)
    elif a.mode=='extract':
        need(not out.exists(),'Output exists; refusing overwrite')
        extract(root,out,items,lookup,fingerprints)
    else:
        need(out.is_dir(),f'Missing complete output {out}')
        verify(root,out,items,lookup,fingerprints)

if __name__=='__main__':
    try:main()
    except Exception as exc:
        print('[XCV] ERROR',type(exc).__name__+':',exc,file=sys.stderr)
        sys.exit(1)
