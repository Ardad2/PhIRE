#!/usr/bin/env python3
"""LXXXVIII: source-pinned read-only preview footprint and PhIRE MR parity audit.

No inference, no split assignment, no training, no WTK/HSDS reads. A report is
atomically written to an isolated directory only after all eight checks pass.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
PILOT_REL = 'ttk_runs_fixed/topology_finetuning/phase7_lxxxvii_wind_screen_20261015'
OUT_REL = 'ttk_runs_fixed/topology_finetuning/phase7_lxxxviii_preview_parity_20261015'
REGIONS = ('gulf_inland', 'upper_midwest', 'pacific_inland', 'southwest_borderland')
DATES = ('20100201', '20100801')
PREVIEWS = tuple(f'preview_{r}_{d}.npz' for r in REGIONS for d in DATES)
TOL = 1e-8


def need(ok,msg):
    if not ok: raise RuntimeError(msg)


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def verify_sum(folder, want):
    p=folder/'SHA256SUMS'
    need(p.is_file(),f'No SHA256SUMS in {folder}')
    got={}
    for line in p.read_text().splitlines():
        d,relative=line.split('  ',1)
        need(re.fullmatch('[0-9a-f]{64}',d) is not None,'Malformed SHA256SUMS')
        q=(folder/relative).resolve()
        need(q.is_relative_to(folder.resolve()) and q.is_file(),'SHA path missing/outside folder')
        need(relative not in got and sha(q)==d,f'Checksum failure: {relative}')
        got[relative]=d
    need(set(got)==set(want),f'Wrong SHA256 member set: {sorted(got)}')
    return got


def prepare(repo):
    import numpy as np
    pilot=repo/PILOT_REL
    sumfiles=('wind_screen.md','screen_summary.json','sample_screen.csv',*PREVIEWS)
    sums=verify_sum(pilot,sumfiles)
    # Re-run the earlier validator, including original geographic source checks.
    oldscript=repo/'phase7_lxxxvii_update/scripts/phase7_wtk_wind_screen.py'
    need(oldscript.is_file(),'Missing pinned prior screening script')
    spec=importlib.util.spec_from_file_location('phase7_prior_screen_verified',oldscript)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    mod.verify(repo,pilot)
    summary=json.loads((pilot/'screen_summary.json').read_text())
    need(set(summary['previews'])==set(PREVIEWS),'Expected eight previews were not retained')
    index={r['region']:r for r in summary['region_summary']}
    need(set(index)==set(REGIONS),'Unexpected regions')
    for v in index.values():need(v['min_valid_fraction']>=0.999999,'Region has invalid WTK values')
    with (pilot/'sample_screen.csv').open(newline='') as fh:
        screen_rows=list(csv.DictReader(fh))
    row_by_id={(x['region'],x['timestamp_utc']):x for x in screen_rows}
    need(len(screen_rows)==64 and len(row_by_id)==64,'Duplicate/missing screened timestamps')
    probes=[]
    for reg in REGIONS:
        details=index[reg]
        rectangle=tuple(details['grid_rect'])
        need(len(rectangle)==4 and rectangle[1]-rectangle[0]==rectangle[3]-rectangle[2]==500,'Wrong rectangle')
        for d in DATES:
            name=f'preview_{reg}_{d}.npz'
            with np.load(pilot/name,allow_pickle=False) as x:
                need(set(x.files)=={'hr_uv','mr_uv','grid_rect','wtk_index','timestamp_utc','region'},f'Unexpected preview fields: {name}')
                hr=np.asarray(x['hr_uv']);mr=np.asarray(x['mr_uv'])
                need(hr.shape==(500,500,2) and mr.shape==(100,100,2),f'Unexpected preview shapes {name}')
                need(hr.dtype==np.float64 and mr.dtype==np.float64,'Preview numeric dtype changed')
                need(np.isfinite(hr).all() and np.isfinite(mr).all(),f'Nonfinite {name}')
                need(tuple(int(t) for t in x['grid_rect'])==rectangle,f'Rectangle mismatch: {name}')
                need(str(x['region'].item())==reg and str(x['timestamp_utc'].item()).startswith(d[:4]+'-'+d[4:6]+'-'+d[6:]),f'Identity mismatch: {name}')
                t=str(x['timestamp_utc'].item())
                match=row_by_id.get((reg,t))
                need(match is not None and int(match['wtk_index'])==int(x['wtk_index']),f'Preview WTK timestamp/index mismatch: {name}')
                record={'name':name,'region':reg,'wtk_index':int(x['wtk_index']), 'timestamp_utc':t,
                    'hr':hr,'mr':mr}
                probes.append(record)
    return sums,summary,probes


def check(repo, out, mode):
    import numpy as np
    sums,meta,probes=prepare(repo)
    print(f'[SOURCE] PASS: 11 LXXXVII checksums, previous verifier, 8 complete previews, 4 regions')
    for item in meta['region_summary']:
        print('[FOOTPRINT]',item['region'],'center=',item['actual_center'],'corners=',item['corners'],'rect=',item['grid_rect'])
    if mode=='preflight':
        print('[PREFLIGHT] PASS: no TF import, no output files and no geographic split assignment')
        return
    need(not out.exists(),f'Report output already exists; refusing overwrite: {out}')
    # Demand the same PhIRE TensorFlow downscaler, not an approximate alternative.
    utilpath=repo/'utils.py'
    need(utilpath.is_file(),'Missing repo utils.py (exact PhIRE downscaler)')
    spec=importlib.util.spec_from_file_location('phase7_original_phire_utils',utilpath)
    util=importlib.util.module_from_spec(spec);spec.loader.exec_module(util)
    need(callable(util.downscale_image),'Missing utils.downscale_image')
    results=[]
    for i,p in enumerate(probes,1):
        block=p['hr'].reshape(100,5,100,5,2).mean(axis=(1,3))
        diff_block=float(np.max(np.abs(block-p['mr'])))
        need(diff_block<=1e-12,f'Preview block average changed {p["name"]}: {diff_block}')
        tensorflow_mr=np.asarray(util.downscale_image(p['hr'],5))
        if tensorflow_mr.ndim==4:
            need(tensorflow_mr.shape==(1,100,100,2),'Unexpected TF output shape')
            tensorflow_mr=tensorflow_mr[0]
        need(tensorflow_mr.shape==(100,100,2),'Unexpected downscaler output')
        need(np.isfinite(tensorflow_mr).all(),'Nonfinite TF result')
        maxdiff=float(np.max(np.abs(tensorflow_mr-p['mr'])))
        meandiff=float(np.mean(np.abs(tensorflow_mr-p['mr'])))
        need(maxdiff<=TOL,f'PhIRE downscale parity fails {p["name"]}: max diff {maxdiff}')
        results.append({k:p[k] for k in ('name','region','wtk_index','timestamp_utc')}|{'max_abs_error':maxdiff,'mean_abs_error':meandiff})
        print(f'[PARITY] PASS {i}/8 {p["name"]} max_abs={maxdiff:.3g} mean_abs={meandiff:.3g}',flush=True)
    out.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='.'+out.name+'.staging-',dir=out.parent))
    try:
        j={'schema_version':1,'no_model_evaluation':True,'no_split_frozen':True,'source_pilot_digests':sums,
           'source_utils_sha256':sha(utilpath),'source_script_sha256':sha(HERE),
           'tolerance':TOL,'regions':meta['region_summary'],'previews':results}
        (stage/'parity_report.json').write_text(json.dumps(j,indent=2,sort_keys=True)+'\n')
        md=['# Phase 7 — LXXXVIII preview parity and footprints','','Original PhIRE `utils.downscale_image` (TF average pooling, K=5) compared against','eight frozen LXXXVII `hr_uv`/`mr_uv` previews without any inference or model scores.','',
            '| Region | Actual WTK center | WTK rectangle | Four geographic corners |','|---|---|---|---|']
        for r in meta['region_summary']:
            md.append('| '+r['region']+' | `'+str(r['actual_center'])+'` | `'+str(r['grid_rect'])+'` | `'+str(r['corners'])+'` |')
        md+=['','| Preview | Max abs MR difference | Mean abs difference |','|---|---:|---:|']
        for x in results:md.append(f"| `{x['name']}` | {x['max_abs_error']:.9g} | {x['mean_abs_error']:.9g} |")
        md+=['','## Decision boundaries','','- All geographical/environmental labels are provisional until footprint and auxiliary-land/terrain review.',
            '- No development/test roles or timestamp windows are locked by this report.',
            '- Do **not** view any model scores on prospective locked test regions before declaring roles.',
            '- This establishes input downsampling equivalence, not model output equivalence or preprocessing for a complete two-stage PhIRE cascade.',
            '- Native WTK wind orientation and projected physical distances require separate certification for 500x500 topology.','']
        (stage/'parity_report.md').write_text('\n'.join(md))
        (stage/'SHA256SUMS').write_text(''.join(f'{sha(stage/n)}  {n}\n' for n in ('parity_report.md','parity_report.json')))
        need(not out.exists(),'Output appeared during parity; refusing overwrite')
        stage.rename(out)
    except BaseException:
        shutil.rmtree(stage,ignore_errors=True)
        raise
    print(f'[PARITY-GATE] PASS: eight direct TensorFlow MR comparisons, untouched screening, output={out}')


def verify(repo,out):
    sums,_,_=prepare(repo)
    got=verify_sum(out,('parity_report.md','parity_report.json'))
    doc=json.loads((out/'parity_report.json').read_text())
    need(doc['schema_version']==1 and doc['no_model_evaluation'] and doc['no_split_frozen'],'Scope violated')
    need(doc['source_pilot_digests']==sums,'Source pilot hashes changed')
    need(doc['source_script_sha256']==sha(HERE) and doc['source_utils_sha256']==sha(repo/'utils.py'),'Source changed')
    need(len(doc['previews'])==8 and set(x['name'] for x in doc['previews'])==set(PREVIEWS),'Preview set changed')
    need(all(0<=x['max_abs_error']<=TOL for x in doc['previews']),'Parity tolerance violated')
    print('[VERIFY-GATE] PASS: current LXXXVII inputs, PhIRE source, eight matching previews and LXXXVIII SHA-256')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--mode',required=True,choices=('preflight','parity','verify'))
    ap.add_argument('--repo-root',default=str(ROOT))
    ap.add_argument('--out')
    a=ap.parse_args();repo=Path(a.repo_root).resolve()
    out=Path(a.out).resolve() if a.out else repo/OUT_REL
    if a.mode=='verify':verify(repo,out)
    else:check(repo,out,a.mode)

if __name__=='__main__':
    try:main()
    except Exception as e:
        print(f'[LXXXVIII] ERROR {type(e).__name__}: {e}',file=sys.stderr)
        raise SystemExit(1)
