#!/usr/bin/env python3
"""Phase 7 / Part LXXXVII: meteorology-only screening of provisional WTK regions.

Does not perform model inference, inspect model metrics, train, or select test splits.
Runs from PhIRE root. Input LXXXVI inventory is immutable and checksum verified.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timedelta
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
INVENTORY_DIR = ROOT/'ttk_runs_fixed/topology_finetuning/phase7_lxxxvi_region_inventory_20261015'
OUT = ROOT/'ttk_runs_fixed/topology_finetuning/phase7_lxxxvii_wind_screen_20261015'
SCRIPT_LXXXVI = ROOT/'phase7_lxxxvi_update/scripts/phase7_region_inventory.py'
SOURCE_BUILDER = ROOT/'build_example_data_extension_500.py'
REGIONS = ('gulf_inland', 'upper_midwest', 'pacific_inland', 'southwest_borderland')
OFFSETS = (0, 84)
PREVIEW_SEASONS = ('2010-02-01T00:00:00', '2010-08-01T00:00:00')
EXPECTED_ANCHORS = (
    '2010-02-01T00:00:00','2010-05-01T00:00:00',
    '2010-08-01T00:00:00','2010-11-01T00:00:00',
    '2012-02-01T00:00:00','2012-05-01T00:00:00',
    '2012-08-01T00:00:00','2012-11-01T00:00:00'
)
# Frozen 2688 single-region fine-tuning WTK index windows (see master notes Part VIII).
TRAIN_WINDOWS = ((336,503),(504,671),(672,839),(840,1007),
                 (2160,2327),(2328,2495),(2496,2663),(2664,2831),
                 (4344,4511),(4512,4679),(4680,4847),(4848,5015),
                 (6552,6719),(6720,6887),(6888,7055),(7056,7223))
LEGACY = (720,1220,2102,2602)


def require(ok, msg):
    if not ok: raise RuntimeError(msg)


def digest(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for part in iter(lambda:f.read(1<<20),b''):h.update(part)
    return h.hexdigest()


def checksum_gate(folder, expect=None):
    p=folder/'SHA256SUMS'
    require(p.is_file(),f'Missing checksum list: {p}')
    seen=set()
    for line in p.read_text().splitlines():
        sha,rel=line.split('  ',1)
        require(re.fullmatch(r'[0-9a-f]{64}',sha) is not None, f'Malformed checksum {rel}')
        target=(folder/rel).resolve()
        require(target.is_relative_to(folder.resolve()),f'Checksum path escape: {rel}')
        require(target.is_file() and digest(target)==sha,f'Checksum mismatch: {target}')
        require(rel not in seen,f'Duplicate checksum {rel}')
        seen.add(rel)
    require(bool(seen),f'Empty checksum list {p}')
    if expect is not None:require(seen==set(expect),f'Unexpected checksum members: {seen}')
    return seen


def separated(a,b,buffer):
    return a[1]+buffer<=b[0] or b[1]+buffer<=a[0] or a[3]+buffer<=b[2] or b[3]+buffer<=a[2]


def current_sources(repo):
    paths={
        'screen_script':Path(__file__),
        'lxxxvi_inventory':repo/'ttk_runs_fixed/topology_finetuning/phase7_lxxxvi_region_inventory_20261015/inventory.json',
        'lxxxvi_notes':repo/'ttk_runs_fixed/topology_finetuning/phase7_lxxxvi_region_inventory_20261015/inventory.md',
        'lxxxvi_script':repo/'phase7_lxxxvi_update/scripts/phase7_region_inventory.py',
        'lxxxvi_candidates':repo/'phase7_lxxxvi_update/region_candidates.json',
        'legacy_builder':repo/'build_example_data_extension_500.py',
    }
    for label,p in paths.items():require(p.is_file(),f'Missing source {label}: {p}')
    return {k:{'path':str(v.relative_to(repo)) if v.is_relative_to(repo) else str(v),'sha256':digest(v)} for k,v in paths.items()}


def plan(repo):
    d=repo/'ttk_runs_fixed/topology_finetuning/phase7_lxxxvi_region_inventory_20261015'
    checksum_gate(d,('inventory.json','inventory.md'))
    data=json.loads((d/'inventory.json').read_text())
    require(data.get('schema_version')==1,'Wrong inventory schema')
    for rel, recorded in data.get('source_hashes',{}).items():
        if rel == 'scripts/phase7_region_inventory.py':
            fp=repo/'phase7_lxxxvi_update/scripts/phase7_region_inventory.py'
        elif rel == 'region_candidates.json':
            fp=repo/'phase7_lxxxvi_update/region_candidates.json'
        elif rel == 'build_example_data_extension_500.py':
            fp=repo/'build_example_data_extension_500.py'
        else:
            raise RuntimeError(f'Unknown inventory source path: {rel}')
        require(fp.is_file() and digest(fp)==recorded,f'Original LXXXVI source hash changed: {fp}')
    require(data.get('suggested_four')==list(REGIONS),'Provisional region selection changed; manual review required')
    require(tuple(data.get('legacy',{}).get('rect_rowcol',()))==LEGACY,'Historical WTK rect differs')
    require(data.get('buffer_grid_cells')==80,'Inventory buffer changed')
    grid=data['grid']
    require(tuple(grid.get('wind_dataset_shape',()))==(61368,1602,2976),'WTK shape unexpected')
    require(data['dataset']=='/nrel/wtk-us.h5','Unexpected WTK dataset')
    a={x['id']:x for x in data['candidates']}
    regions=[]
    for name in REGIONS:
        require(name in a, f'Candidate missing {name}')
        r=a[name]; rect=r['rect']
        require(r['in_domain'] and r['separated_from_legacy'],'Candidate fails geographic integrity')
        require(rect and rect[1]-rect[0]==500 and rect[3]-rect[2]==500,'Candidate not 500x500')
        require(separated(rect,LEGACY,80),'Candidate overlaps legacy with buffer')
        regions.append(r)
    for i,x in enumerate(regions):
        for y in regions[i+1:]:require(separated(x['rect'],y['rect'],80),f'Overlapping candidate regions: {x["id"]} vs {y["id"]}')
    ww=data['times']['windows']
    require(tuple(x['requested_start'] for x in ww)==EXPECTED_ANCHORS,'Timestamp window anchors differ')
    samples=[]
    for r in regions:
        for w in ww:
            require(w['available'] and int(w['index_end'])-int(w['index_start'])==167,'Broken 168h WTK window')
            for off in OFFSETS:
                ix=int(w['index_start'])+off
                require(0<=ix<61368,'WTK index out of range')
                require(ix>167 and all(not lo<=ix<=hi for lo,hi in TRAIN_WINDOWS),'Temporal fine-tune overlap')
                ts=(datetime.fromisoformat(w['requested_start'])+timedelta(hours=off)).isoformat()
                samples.append({'region':r['id'],'wtk_index':ix,'timestamp_utc':ts,
                    'anchor_utc':w['requested_start'],'offset_hours':off,'rect':r['rect'],
                    'preview':w['requested_start'] in PREVIEW_SEASONS and off==0})
    require(len(samples)==64 and sum(x['preview'] for x in samples)==8,'Unexpected predeclared pilot sizes')
    return data,regions,samples


def parse_time(raw):
    if hasattr(raw,'item') and not isinstance(raw,(str,bytes)):raw=raw.item()
    if isinstance(raw,bytes):raw=raw.decode('utf8')
    s=str(raw).strip()
    if re.fullmatch(r'\d{14}',s):return datetime.strptime(s,'%Y%m%d%H%M%S')
    return datetime.fromisoformat(s.replace('Z','+00:00')).replace(tzinfo=None)


def wind_components(ws,wd,np):
    theta=np.deg2rad(wd)
    return -ws*np.sin(theta),-ws*np.cos(theta)


def summary_field(ws,wd,np):
    good=np.isfinite(ws)&np.isfinite(wd)&(ws>=0)&(wd>=0)&(wd<=360)
    frac=float(good.mean())
    if not good.any():return {'valid_fraction':frac,'valid_points':0, 'invalid_points':int(good.size)}
    a=ws[good]
    # No artificial zeros in statistics. Gradient uses only valid adjacent cells.
    gy=good[1:,:]&good[:-1,:]
    gx=good[:,1:]&good[:,:-1]
    diffs=[]
    if np.any(gy):diffs.append(np.abs(ws[1:,:]-ws[:-1,:])[gy])
    if np.any(gx):diffs.append(np.abs(ws[:,1:]-ws[:,:-1])[gx])
    g=np.concatenate(diffs) if diffs else np.asarray([],float)
    return {'valid_fraction':frac, 'valid_points':int(good.sum()), 'invalid_points':int(good.size-good.sum()),
        'speed_mean':float(np.mean(a)),'speed_std':float(np.std(a)),
        'speed_p10':float(np.percentile(a,10)),'speed_p50':float(np.percentile(a,50)),
        'speed_p90':float(np.percentile(a,90)),'speed_p99':float(np.percentile(a,99)),
        'speed_max':float(np.max(a)), 'speed_over10_fraction':float(np.mean(a>=10)),
        'speed_over15_fraction':float(np.mean(a>=15)),
        'grad_abs_p95':float(np.percentile(g,95)) if len(g) else None,
        'grad_abs_p99':float(np.percentile(g,99)) if len(g) else None}


def save_csv(p,records):
    cols=['region','wtk_index','timestamp_utc','anchor_utc','offset_hours','preview',
          'valid_fraction','valid_points','invalid_points','speed_mean','speed_std','speed_p10','speed_p50',
          'speed_p90','speed_p99','speed_max','speed_over10_fraction','speed_over15_fraction','grad_abs_p95','grad_abs_p99']
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=cols)
        w.writeheader();w.writerows([{k:x.get(k) for k in cols} for x in records])


def pilot(repo,output,dataset,regions,samples,sources):
    require(not output.exists(),f'Output already exists, refusing overwrite: {output}')
    import numpy as np
    import h5pyd
    output.parent.mkdir(parents=True,exist_ok=True)
    # Staging is isolated; an interrupted probe cannot produce an apparently complete report.
    temp=Path(tempfile.mkdtemp(prefix='.'+output.name+'.staging-',dir=output.parent))
    records=[]; previews=[]
    try:
        with h5pyd.File(dataset,'r') as f:
            a=f['windspeed_100m'];b=f['winddirection_100m'];dt=f['datetime']
            require(tuple(a.shape)==(61368,1602,2976) and tuple(b.shape)==tuple(a.shape),'Remote WTK grid mismatch')
            for n,s in enumerate(samples,1):
                i=s['wtk_index']; y0,y1,x0,x1=s['rect']
                actual=parse_time(dt[i])
                require(actual==datetime.fromisoformat(s['timestamp_utc']),f'Datetime mismatch {i}: {actual} != {s["timestamp_utc"]}')
                ws=np.asarray(a[i,y0:y1,x0:x1],dtype=np.float64)
                wd=np.asarray(b[i,y0:y1,x0:x1],dtype=np.float64)
                require(ws.shape==wd.shape==(500,500),f'Invalid remote slice for {s["region"]} at {i}: {ws.shape} {wd.shape}')
                stats=summary_field(ws,wd,np)
                row={k:v for k,v in s.items() if k!='rect'}
                row.update(stats)
                records.append(row)
                if s['preview'] and stats['valid_fraction']>=0.999999:
                    u,v=wind_components(ws,wd,np)
                    hr=np.stack((u,v),axis=-1)
                    # Strict channel-preserving block mean; compare original tf utils when running inference.
                    mr=hr.reshape(100,5,100,5,2).mean(axis=(1,3))
                    require(np.allclose(np.hypot(hr[:,:,0],hr[:,:,1]),ws,rtol=0,atol=1e-10),'uv conversion invariant failed')
                    fname=f"preview_{s['region']}_{s['anchor_utc'][:10].replace('-','')}.npz"
                    np.savez_compressed(temp/fname,hr_uv=hr,mr_uv=mr,grid_rect=np.asarray(s['rect']),
                                        wtk_index=np.asarray(i),timestamp_utc=np.asarray(s['timestamp_utc']),region=np.asarray(s['region']))
                    previews.append(fname)
                print(f"[SCREEN] {n:02d}/{len(samples)} region={s['region']} t={i} valid={stats['valid_fraction']:.4f} speed_mean={stats.get('speed_mean',float('nan')):.3f}",flush=True)
        save_csv(temp/'sample_screen.csv',records)
        by=[]
        for r in regions:
            sub=[s for s in records if s['region']==r['id']]
            require(len(sub)==16,'Region sample-count mismatch')
            entry={'region':r['id'],'grid_center':r['center_grid'],'grid_rect':r['rect'],
                   'actual_center':r['actual_center'],'corners':r['corners'],
                   'intended_environment':r['environment'],'n_fields':len(sub),
                   'min_valid_fraction':min(x['valid_fraction'] for x in sub),
                   'mean_speed_over_fields':float(np.mean([x.get('speed_mean',float('nan')) for x in sub])),
                   'mean_p99_speed_over_fields':float(np.mean([x.get('speed_p99',float('nan')) for x in sub])),
                   'mean_grad_p95_over_fields':float(np.mean([x.get('grad_abs_p95',float('nan')) for x in sub])),
                   'suggested_data_suitability': all(x['valid_fraction']>=0.999999 for x in sub)}
            by.append(entry)
        result={'schema_version':1,'scope':'meteorology screening only, no train/eval split frozen',
                'n_samples':len(records),'n_regions':len(regions),'n_times_per_region':16,
                'source_fingerprints':sources,'input_dataset':dataset,
                'sampling':'two UTC hours (offset 0,84) per seven-day window across eight 2010/2012 seasonal windows',
                'original_train_window_ranges':[list(x) for x in TRAIN_WINDOWS],
                'historical_benchmark_window':[0,167],
                'region_summary':by,'previews':previews,'timestamps_checked_against_WTK':True}
        (temp/'screen_summary.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
        md=['# Phase 7 / LXXXVII — provisional wind-field screening','',
            '64 full 500×500 native WTK snapshots (4 provisional regions × 8 seasonal windows × 2 UTC hours).',
            'No training, inference, PD/MT calculation, model ranking, or locked test-set designation.','',
            '| Region | Valid fraction (minimum) | Mean wind speed (m/s) | Mean per-field p99 speed (m/s) | Mean per-field p95 abs pixel gradient (m/s) |',
            '|---|---:|---:|---:|---:|']
        for v in by:
            md.append(f"| {v['region']} | {v['min_valid_fraction']:.5f} | {v['mean_speed_over_fields']:.3f} | {v['mean_p99_speed_over_fields']:.3f} | {v['mean_grad_p95_over_fields']:.3f} |")
        md.extend(['','## Caveats','','- These 16 snapshots per region are a geographic screening sample, **not** a population estimate or a generalization benchmark.',
            '- WTK coordinates, labels and differences in wind fields do not certify terrain or coastal regime; inspect footprint and ancillary terrain/land metadata before assigning environment types.',
            '- Snapshot winds are from WTK speed/direction at 100 m, converted to native [u,v]. Only complete finite snapshots are saved as previews.',
            '- Derived preview MR is explicit factor-5 block-average. Exact parity with original PhIRE TensorFlow downscale must be checked independently before inference.',
            '- All 2010/2012 timestamps are disjoint from the documented original 2007 fine-tuning windows. Original PhIRE pretraining exposure remains unknown.',
            '- All regions remain provisional. Test-set labels must be frozen before looking at model scores.',''])
        (temp/'wind_screen.md').write_text('\n'.join(md))
        all_files=sorted(x.name for x in temp.iterdir() if x.is_file())
        (temp/'SHA256SUMS').write_text(''.join(f'{digest(temp/x)}  {x}\n' for x in all_files))
        require(not output.exists(),'Output appeared during run; refusing overwrite')
        temp.rename(output)
        print(f'[PILOT] PASS: 64 native WTK fields screened; {len(previews)} snapshot previews saved; 4-region report created; no models evaluated',flush=True)
        print('[OUTPUT]',output,flush=True)
    except Exception:
        shutil.rmtree(temp,ignore_errors=True)
        raise


def verify(repo,output):
    require(output.is_dir(),f'Missing pilot report directory: {output}')
    seen=checksum_gate(output)
    j=json.loads((output/'screen_summary.json').read_text())
    data,reg,samples=plan(repo)
    expected_sources=current_sources(repo)
    require(j['source_fingerprints']==expected_sources,'Pilot source fingerprints changed')
    require(j['n_samples']==64 and j['n_regions']==4 and j['n_times_per_region']==16,'Pilot manifest shape differs')
    require(len(j['region_summary'])==4,'Wrong region count')
    require(len(j['previews'])<=8 and all(x in seen for x in j['previews']),'Preview manifest invalid')
    require(seen=={'screen_summary.json','sample_screen.csv','wind_screen.md',*j['previews']},'Unexpected output file set')
    with (output/'sample_screen.csv').open(newline='') as f: rows=list(csv.DictReader(f))
    require(len(rows)==64,'Wrong CSV length')
    expect={(x['region'],x['wtk_index']) for x in samples}
    got={(x['region'],int(x['wtk_index'])) for x in rows}
    require(expect==got and len(got)==64,'Sample or WTK indices mismatch')
    require(all(0<=float(x['valid_fraction'])<=1 for x in rows),'Invalid fraction')
    for r in j['region_summary']:
        z=[x for x in rows if x['region']==r['region']]
        require(len(z)==16,'Wrong per-region count')
        require(math.isclose(float(r['min_valid_fraction']), min(float(x['valid_fraction']) for x in z),abs_tol=1e-12),'Valid fraction summary error')
    print(f'[VERIFY] PASS: all output hashes; 64 exact (region, WTK-time) rows; source continuity and per-region summaries. No HSDS reads. {output}',flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--mode',choices=('preflight','pilot','verify'),required=True)
    ap.add_argument('--repo-root',default=str(ROOT),help='PhIRE repository root (for tests; defaults to detected root)')
    ap.add_argument('--out',default=None)
    a=ap.parse_args()
    repo=Path(a.repo_root).resolve()
    out=Path(a.out).resolve() if a.out else repo/'ttk_runs_fixed/topology_finetuning/phase7_lxxxvii_wind_screen_20261015'
    if a.mode=='verify':verify(repo,out);return
    inventory,regions,samples=plan(repo)
    sources=current_sources(repo)
    require(not out.exists(),f'Output already exists; refusing overwrite: {out}')
    print(f'[INVENTORY] PASS: original LXXXVI SHA256; four nonoverlapping 500x500 regions, buffer=80',flush=True)
    print(f'[TIME] PASS: 64 predeclared samples, 2010/2012 UTC anchor+offset, disjoint from 2007 evaluation and fine-tuning windows',flush=True)
    if a.mode=='preflight':
        print('[PREFLIGHT] PASS: no network requests, data arrays, or output files; next mode=pilot',flush=True);return
    pilot(repo,out,inventory['dataset'],regions,samples,sources)

if __name__=='__main__':
    try:main()
    except Exception as e:
        print(f'[LXXXVII] ERROR {type(e).__name__}: {e}',file=sys.stderr)
        raise SystemExit(1)
