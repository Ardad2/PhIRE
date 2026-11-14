#!/usr/bin/env python3
"""Phase 7 / Part LXXXVI — read-only HSDS geography and time-index inventory.

No native wind fields, TFRecords, checkpoints, SR predictions, PD or MT results
are generated.  The only report is written to a fresh, isolated output folder.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / 'ttk_runs_fixed' / 'topology_finetuning' / 'phase7_lxxxvi_region_inventory_20261015'
REFERENCE_RECT = (720, 1220, 2102, 2602)  # [row0,row1,col0,col1), archived old-168
EXPECTED_OLD_CENTER = (970, 2352)
DEFAULT_WTK = '/nrel/wtk-us.h5'
REQ_VARS = ('coordinates', 'windspeed_100m', 'winddirection_100m', 'datetime')


def die(s: str):
    raise RuntimeError(s)


def sha256(path: Path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for part in iter(lambda: fh.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def to_datetime(raw):
    if isinstance(raw, bytes):
        raw = raw.decode('utf-8', errors='strict')
    if hasattr(raw, 'item') and not isinstance(raw, (str, bytes)):
        raw = raw.item()
    s = str(raw).strip().replace('Z', '+00:00')
    # Supported WTK strings: yyyy-mm-dd HH:MM:SS, ISO time, sometimes tz.
    try:
        out = datetime.fromisoformat(s)
    except ValueError as exc:
        raise ValueError(f'Unparseable WTK datetime item {s[:90]!r}') from exc
    return out.replace(tzinfo=None) if out.utcoffset() == timedelta(0) else out


def require(cond, msg):
    if not cond:
        die(msg)


def candidates():
    doc = json.loads((PACKAGE / 'region_candidates.json').read_text())
    require(doc['schema_version'] == 1, 'Candidate schema mismatch')
    spots = doc['candidates']
    ids = [a['id'] for a in spots]
    require(len(ids) == len(set(ids)), 'Duplicate candidate id')
    require(6 <= len(ids) <= 20, 'Unexpected candidate count')
    for a in spots:
        require(re.fullmatch('[a-z0-9_]+', a['id']) is not None, 'Unsafe candidate id')
        require(-90 <= a['latitude'] <= 90 and -180 <= a['longitude'] <= 180, 'Invalid lat/lon')
    return doc


def _extract_lat_lon(chunk, geometry_shape, np):
    """Read supported HSDS coordinate layouts without reordering grid axes."""
    if chunk.dtype.names:
        names = {x.lower(): x for x in chunk.dtype.names}
        latname = names.get('lat') or names.get('latitude')
        lonname = names.get('lon') or names.get('longitude') or names.get('long')
        require(latname is not None and lonname is not None,
                f'Unknown coordinate fields: {list(chunk.dtype.names)}')
        la, lo = chunk[latname], chunk[lonname]
    elif chunk.ndim == 3 and chunk.shape[-1] == 2:
        la, lo = chunk[..., 0], chunk[..., 1]
    elif chunk.ndim == 3 and chunk.shape[0] == 2:
        la, lo = chunk[0], chunk[1]
    else:
        die(f'Unsupported coordinate chunk shape/dtype {chunk.shape}, {chunk.dtype}')
    la = np.asarray(la, dtype=float)
    lo = np.asarray(lo, dtype=float)
    require(la.ndim == lo.ndim == 2 and la.shape == lo.shape,
            'Coordinates must produce matched 2-D lat/lon grids')
    # Numeric component order can be swapped in some HSDS exports.
    def looks_like_lat(x):
        q = x[np.isfinite(x)]
        return q.size and np.mean((q >= -90) & (q <= 90)) > 0.999
    if (not looks_like_lat(la)) and looks_like_lat(lo):
        la, lo = lo, la
    require(looks_like_lat(la), 'Latitude out of range or coordinate channel orientation unsupported')
    return la, lo


def coordinate_grid_shape(ds, wind_shape):
    h, w = wind_shape[-2:]
    shape = tuple(ds.shape)
    if len(shape) == 2 and shape == (h, w):
        return h, w, 'compound-HW'
    if len(shape) == 3 and shape == (h, w, 2):
        return h, w, 'HW2'
    if len(shape) == 3 and shape == (2, h, w):
        return h, w, '2HW'
    die(f'Coordinate shape {shape} inconsistent with wind spatial shape {(h,w)}')


def read_coordinate_slice(ds, form, rs, cs, np):
    if form == 'compound-HW':
        data = ds[rs, cs]
    elif form == 'HW2':
        data = ds[rs, cs, :]
    else:
        data = ds[:, rs, cs]
    return _extract_lat_lon(np.asarray(data), form, np)


def angular_distance_sq(la, lo, target_lat, target_lon, np):
    # Haversine central angle squared, longitude wraps handled properly.
    a1 = np.deg2rad(la)
    a2 = math.radians(target_lat)
    da = a1 - a2
    dl = np.deg2rad(lo - target_lon)
    v = np.sin(da / 2)**2 + np.cos(a1)*math.cos(a2)*np.sin(dl/2)**2
    bad = ~np.isfinite(v) | ~np.isfinite(la) | ~np.isfinite(lo)
    return np.where(bad, np.inf, v)


def locate(ds, form, lat, lon, stride, np):
    # Coarse HSDS read followed by exact neighborhood refinement.
    a, b = read_coordinate_slice(ds, form, slice(None, None, stride), slice(None, None, stride), np)
    score = angular_distance_sq(a, b, lat, lon, np)
    require(np.isfinite(score).any(), f'No finite coordinates near {lat},{lon}')
    ci, cj = np.unravel_index(np.argmin(score), score.shape)
    y = int(ci * stride)
    x = int(cj * stride)
    h, w = coord_spatial_shape(ds, form)
    r0, r1 = max(0, y - 2 * stride), min(h, y + 2 * stride + 1)
    c0, c1 = max(0, x - 2 * stride), min(w, x + 2 * stride + 1)
    al, ol = read_coordinate_slice(ds, form, slice(r0, r1), slice(c0, c1), np)
    fine = angular_distance_sq(al, ol, lat, lon, np)
    require(np.isfinite(fine).any(), f'No finite local coordinates near {lat},{lon}')
    yi, xi = np.unravel_index(np.argmin(fine), fine.shape)
    return int(r0+yi), int(c0+xi), float(al[yi, xi]), float(ol[yi, xi])


def coord_spatial_shape(ds, form):
    if form == '2HW':
        return tuple(ds.shape[1:3])
    return tuple(ds.shape[:2])


def window_rect(y, x, h, w, width=500):
    r0, c0 = int(y-width//2), int(x-width//2)
    r1, c1 = r0 + width, c0 + width
    return [r0,r1,c0,c1] if r0>=0 and c0>=0 and r1<=h and c1<=w else None


def separated(a, b, buffer):
    # Rectangle [row0,row1,col0,col1), no overlap including buffer.
    return (a[1]+buffer <= b[0] or b[1]+buffer <= a[0] or
            a[3]+buffer <= b[2] or b[3]+buffer <= a[2])


def corners(ds, form, rect, np):
    r0,r1,c0,c1 = rect
    out=[]
    for name,y,x in [('northwest_grid',r0,c0), ('northeast_grid',r0,c1-1),
                     ('southwest_grid',r1-1,c0), ('southeast_grid',r1-1,c1-1)]:
        la,lo = read_coordinate_slice(ds,form,slice(y,y+1),slice(x,x+1),np)
        out.append({'corner':name,'row':y,'col':x,'latitude':float(la[0,0]),'longitude':float(lo[0,0])})
    return out


def choose_disjoint(rows, buffer, max_regions=4):
    possible = [x for x in rows if x['in_domain'] and x['separated_from_legacy']]
    best=[]
    best_score=(-1,-1)
    for count in range(1,min(max_regions,len(possible))+1):
        for subset in itertools.combinations(possible,count):
            if all(separated(x['rect'],y['rect'],buffer) for x,y in itertools.combinations(subset,2)):
                # maximize count; tie-break using declared candidate priorities
                score=(count,-sum(x['priority'] for x in subset))
                if score>best_score:
                    best_score=score; best=list(subset)
    return [x['id'] for x in best]


def inspect_time(ds, weeks):
    # datetime vector is short (~7 years hourly); no meteorology arrays read.
    raw = ds[:]
    dates = [to_datetime(s) for s in raw]
    require(all(dates[i] < dates[i+1] for i in range(len(dates)-1)),
            'WTK datetime vector is not strictly ascending')
    lookup={t:i for i,t in enumerate(dates)}
    require(len(lookup)==len(dates), 'Duplicate WTK timestamps')
    result=[]
    for anchor in weeks:
        start=datetime.fromisoformat(anchor)
        record={'requested_start':anchor,'hours':168,'available':False}
        i=lookup.get(start)
        if i is not None and i+167<len(dates):
            matches=all(dates[i+k]==start+timedelta(hours=k) for k in range(168))
            if matches:
                record.update({'available':True,'index_start':i,'index_end':i+167,
                               'actual_first':dates[i].isoformat(sep=' '),
                               'actual_last':dates[i+167].isoformat(sep=' ')})
            else:
                record['reason']='Missing/non-hourly records in requested window'
        else:
            record['reason']='Date not found or insufficient subsequent records'
        result.append(record)
    return {'first':dates[0].isoformat(sep=' '), 'last':dates[-1].isoformat(sep=' '),
            'n_timestamps':len(dates), 'windows':result}


def run_probe(args):
    import numpy as np
    try:
        import h5pyd
    except ImportError as exc:
        die('h5pyd not installed in this Python environment; activate original WTK extraction env. No files written.')
    settings=candidates()
    out=Path(args.out).expanduser().resolve()
    require(not out.exists(), f'Output already exists; refusing overwrite: {out}')
    dataset=args.dataset
    print('[REMOTE] Connecting to WTK in read-only mode:', dataset, flush=True)
    with h5pyd.File(dataset,'r') as f:
        for v in REQ_VARS:
            require(v in f, f'Missing WTK dataset: {v}')
        speed, direction, coords=f['windspeed_100m'],f['winddirection_100m'],f['coordinates']
        require(len(speed.shape)==3 and tuple(speed.shape)==tuple(direction.shape),
                f'Unexpected WTK 100m wind shapes {speed.shape} {direction.shape}')
        h,w,form=coordinate_grid_shape(coords,speed.shape)
        require(min(h,w)>500, 'WTK grid not large enough for complete 500x500 crops')
        print(f'[GRID] WTK wind shape={speed.shape}, coordinate shape={coords.shape}, layout={form}',flush=True)
        old=locate(coords,form,39.5,-75.0,args.stride,np)
        require(abs(old[0]-EXPECTED_OLD_CENTER[0])<=8 and abs(old[1]-EXPECTED_OLD_CENTER[1])<=8,
                f'Legacy reference alignment mismatch: found {old[:2]} expected {EXPECTED_OLD_CENTER}; STOP and inspect coordinate conventions')
        print(f'[LEGACY] PASS old reference near grid={old[:2]} expected={EXPECTED_OLD_CENTER}; frozen rectangle={REFERENCE_RECT}',flush=True)
        rows=[]
        for c in settings['candidates']:
            y,x,lat,lon=locate(coords,form,c['latitude'],c['longitude'],args.stride,np)
            rect=window_rect(y,x,h,w)
            good=(rect is not None)
            disjoint=bool(good and separated(rect,REFERENCE_RECT,args.buffer))
            row={'id':c['id'],'environment':c['environment'],'priority':c['priority'],
                 'requested_center':{'latitude':c['latitude'],'longitude':c['longitude']},
                 'actual_center':{'latitude':lat,'longitude':lon},
                 'center_grid':{'row':y,'col':x},'in_domain':good,
                 'rect':rect,'separated_from_legacy':disjoint,
                 'corners':corners(coords,form,rect,np) if good else [],
                 'warning':None if good else '500x500 window falls outside available WTK grid'}
            rows.append(row)
            print(f"[CANDIDATE] {c['id']:<24} center=({y},{x}) full500={good} disjoint_legacy_buffer{args.buffer}={disjoint}",flush=True)
        pairs=[]
        for a,b in itertools.combinations(rows,2):
            both=a['in_domain'] and b['in_domain']
            pairs.append({'a':a['id'],'b':b['id'],'disjoint_with_buffer':
                          bool(both and separated(a['rect'],b['rect'],args.buffer))})
        suggested=choose_disjoint(rows,args.buffer,max_regions=4)
        print(f'[SELECTION] PROVISIONAL geography-only disjoint candidates={suggested}; NOT locked model-evaluation splits',flush=True)
        t=inspect_time(f['datetime'],settings['proposed_windows'])
        valid=sum(x['available'] for x in t['windows'])
        print(f"[TIME] {valid}/{len(t['windows'])} candidate seven-day seasonal windows contain 168 consecutive hourly records",flush=True)
        result={'schema_version':1,'dataset':dataset,'source_type':'read-only WTK HSDS',
                'grid':{'wind_dataset_shape':list(speed.shape), 'coordinates_shape':list(coords.shape),
                        'coordinate_form':form,'height_m':100},'legacy':{'nearest_row_col':list(old[:2]),
                        'nearest_lat_lon':list(old[2:]),'rect_rowcol':list(REFERENCE_RECT)},
                'buffer_grid_cells':args.buffer,'stride':args.stride,
                'candidates':rows,'pairwise':pairs,'suggested_four':suggested,
                'times':t,'note':'Coordinate/domain inventory only. No meteorological suitability certification, pretrained-weight training overlap check, or model evaluation.'}
    require(valid == len(settings['proposed_windows']),
            'Some candidate seasonal windows not available. Refusing output: inspect dates/WTK frequency; adjust via a NEW workpack.')
    require(len(suggested)>=3,
            'Fewer than three disjoint new-area candidates found. Stop and revise region candidates; no output created.')
    source_hashes={str(Path('scripts/phase7_region_inventory.py')):sha256(Path(__file__)),
                   'region_candidates.json':sha256(PACKAGE/'region_candidates.json')}
    legacy_source=ROOT/'build_example_data_extension_500.py'
    if legacy_source.exists():
        source_hashes[str(legacy_source.relative_to(ROOT))]=sha256(legacy_source)
    result['source_hashes']=source_hashes
    out.parent.mkdir(parents=True,exist_ok=True)
    # Atomic creation in fresh directory only; do not touch past output.
    out.mkdir(exist_ok=False)
    (out/'inventory.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    report=['# Phase 7 — preliminary WTK geographic and temporal inventory','',
            'No meteorological fields, SR outputs, scores or model test-set splits generated.','',
            f"- Source: `{dataset}`; grid shape {tuple(result['grid']['wind_dataset_shape'])}; coordinate format {form}",
            f"- Frozen old-168 extraction rectangle: `{REFERENCE_RECT}`; reference alignment PASS",
            f"- Required buffer: {args.buffer} WTK grid cells (only an index-space separation criterion)",
            f"- Provisional candidate set: `{', '.join(suggested)}`; subject to manual meteorology and split review",'',
            '| Candidate | Intended regime | WTK center row,col | Full 500² | Separated from old area |',
            '|---|---|---|---|---|']
    for c in rows:
        yx=c['center_grid']
        report.append(f"| {c['id']} | {c['environment']} | ({yx['row']},{yx['col']}) | {c['in_domain']} | {c['separated_from_legacy']} |")
    report += ['', '## Season candidates (not finalized)', '', '| Requested first UTC hour | First WTK index | Last WTK index | Valid 168 hourly steps |',
               '|---|---:|---:|---|']
    for trow in t['windows']:
        report.append(f"| {trow['requested_start']} | {trow.get('index_start','—')} | {trow.get('index_end','—')} | {trow['available']} |")
    report += ['', '## Important boundaries','',
               '- A disjoint 500×500 native grid rectangle does not prove statistical meteorological independence.',
               '- Original PhIRE pretrained-weight train/test membership remains unknown; this inventory guarantees neither pretraining novelty nor temporal novelty.',
               '- Geographic regime labels reflect proposed targets, not verified physical descriptors. Validate terrain, coastline, speed/gradient distributions before finalizing.',
               '- Do not tune models or inspect locked test-region results while choosing losses.',
               '- WTK timestamps are read from metadata; check local/global temporal overlap with archived training windows before test-set freezing.',
               '- The full-field/patch topological evaluation protocol is a separate following phase.','']
    (out/'inventory.md').write_text('\n'.join(report))
    hashes=[]
    for p in ('inventory.json','inventory.md'):
        hashes.append(f'{sha256(out/p)}  {p}')
    (out/'SHA256SUMS').write_text('\n'.join(hashes)+'\n')
    print(f'[PROBE] PASS: {len(rows)} candidates; {len(suggested)} provisional disjoint regions; {len(t["windows"])} valid seasonal windows; no meteorological slices extracted',flush=True)
    print(f'[OUTPUT] {out}',flush=True)


def run_verify(args):
    out=Path(args.out).expanduser().resolve()
    require(out.is_dir(), f'Missing output dir {out}')
    src=json.loads((out/'inventory.json').read_text())
    for rel,h in src['source_hashes'].items():
        p=PACKAGE/rel if rel in ('region_candidates.json','scripts/phase7_region_inventory.py') else ROOT/rel
        require(p.is_file() and sha256(p)==h, f'Inventory source changed: {p}')
    lines=(out/'SHA256SUMS').read_text().splitlines()
    require(len(lines)==2,'Unexpected inventory checksum count')
    for l in lines:
        h,fn=l.split('  ',1)
        require(fn in ('inventory.json','inventory.md') and sha256(out/fn)==h,
                f'Checksum mismatch {fn}')
    require(src['schema_version']==1 and len(src['suggested_four'])>=3,
            'Inventory schema/selection invalid')
    require(all(x['available'] for x in src['times']['windows']),
            'Incomplete candidate time windows')
    print(f'[VERIFY] PASS: output SHA256SUMS, source file hashes, candidate selection and temporal inventory; remote WTK not reprobed. {out}')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--mode',choices=('preflight','probe','verify'),required=True)
    ap.add_argument('--dataset',default=DEFAULT_WTK)
    ap.add_argument('--out',default=str(DEFAULT_OUT))
    ap.add_argument('--buffer',type=int,default=80)
    ap.add_argument('--stride',type=int,default=16)
    a=ap.parse_args()
    require(a.buffer>=0 and 1<=a.stride<=64,'Invalid buffer or stride')
    if a.mode=='preflight':
        doc=candidates()
        require((ROOT/'dataset_generation_and_repair_notes.md').is_file(),'Missing on-host master research notes')
        require((ROOT/'build_example_data_extension_500.py').is_file(), 'Missing original WTK example builder')
        require(not Path(a.out).exists(),f'Report output exists, refusing overwrite: {a.out}')
        print(f"[LOCAL] PASS: {len(doc['candidates'])} candidate geographic centers; {len(doc['proposed_windows'])} predeclared seasonal windows")
        print('[LEGACY] PASS: reference old-168 WTK rectangle 720:1220,2102:2602; check on remote WTK pending')
        print('[PREFLIGHT] PASS: no network requests or experiment files written. Next: remote --mode probe')
    elif a.mode=='probe':
        run_probe(a)
    else:
        run_verify(a)

if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        print(f'[LXXXVI] ERROR {type(exc).__name__}: {exc}',file=sys.stderr)
        sys.exit(1)
