#!/usr/bin/env python3
"""LXXXIX: freeze dev-vs-external WTK regions before inspecting model scores.

Source-pinned, read-only preflight; atomically isolated --mode freeze;
read-only --mode verify. No TensorFlow/HSDS or model inference.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, re, shutil, tempfile
from pathlib import Path

HERE=Path(__file__).resolve()
ROOT=HERE.parents[2]
PKG=HERE.parents[1]
PREV=Path('ttk_runs_fixed/topology_finetuning/phase7_lxxxviii_preview_parity_20261015')
SCREEN=Path('ttk_runs_fixed/topology_finetuning/phase7_lxxxvii_wind_screen_20261015')
INVENTORY=Path('ttk_runs_fixed/topology_finetuning/phase7_lxxxvi_region_inventory_20261015')
OUT=Path('ttk_runs_fixed/topology_finetuning/phase7_lxxxix_locked_region_split_20261015')
VALID=('gulf_inland','southwest_borderland','upper_midwest','pacific_inland')
EXPECTED_STARTS=(27048,29184,31392,33600,44568,46728,48936,51144)
EXPECTED_ANCHORS=('2010-02-01T00:00:00','2010-05-01T00:00:00','2010-08-01T00:00:00','2010-11-01T00:00:00',
'2012-02-01T00:00:00','2012-05-01T00:00:00','2012-08-01T00:00:00','2012-11-01T00:00:00')

def need(ok,msg):
    if not ok:raise RuntimeError(msg)

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def sums(d,expected):
    p=d/'SHA256SUMS';need(p.is_file(),f'Missing SHA256SUMS: {p}')
    found={}
    for ln in p.read_text().splitlines():
        try:digest,name=ln.split('  ',1)
        except ValueError:raise RuntimeError('Invalid SHA format')
        file=(d/name).resolve()
        need(re.fullmatch(r'[a-f0-9]{64}',digest) is not None,'Malformed hash')
        need(file.is_relative_to(d.resolve()) and file.is_file(),'Unsafe or missing source file')
        need(name not in found and sha(file)==digest,f'Source checksum fail: {name}')
        found[name]=digest
    need(set(found)==set(expected),f'Wrong source manifest files: {sorted(found)}')
    return found

def sep(a,b,pad):
    return (a[1]+pad<=b[0] or b[1]+pad<=a[0] or a[3]+pad<=b[2] or b[3]+pad<=a[2])

def inspect(root):
    plan=json.loads((PKG/'split_plan.json').read_text())
    need(plan['schema_version']==1,'Bad plan version')
    dev=plan['development_regions'];test=plan['locked_external_test_regions']
    need(dev==['gulf_inland','southwest_borderland'] and test==['upper_midwest','pacific_inland'],'Unexpected region roles')
    need(len(set(dev+test))==4 and set(dev+test)==set(VALID),'Wrong candidate set')
    need(plan['primary_windows_utc']+plan['replication_windows_utc']==list(EXPECTED_ANCHORS),'Unexpected timestamps')
    need(plan['hours_per_window']==168 and plan['reference_separation_grid_cells']==80,'Protocol change')
    need(plan['dev_pilot_previews']==[f'preview_{r}_{d}.npz' for r in dev for d in ('20100201','20100801')],'Pilot scope violation')
    prev=root/PREV;screen=root/SCREEN;inventory=root/INVENTORY
    prev_s=sums(prev,('parity_report.md','parity_report.json'))
    screen_s=sums(screen,('wind_screen.md','screen_summary.json','sample_screen.csv',*(f'preview_{r}_{d}.npz' for r in VALID for d in ('20100201','20100801'))))
    inv_s=sums(inventory,('inventory.json','inventory.md'))
    # Call the completed LXXXVIII independent source verifier (does not import tensorflow).
    old=root/'phase7_lxxxviii_update/scripts/phase7_preview_parity.py'
    need(old.is_file(),'Missing source script from LXXXVIII')
    spec=importlib.util.spec_from_file_location('phase7_lxxxviii_check',old)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.verify(root,prev)
    p=json.loads((prev/'parity_report.json').read_text())
    d=json.loads((screen/'screen_summary.json').read_text())
    inv=json.loads((inventory/'inventory.json').read_text())
    regions={r['region']:r for r in d['region_summary']}
    need(set(regions)==set(VALID),'Regions missing from screen')
    need(set(r['region'] for r in p['regions'])==set(VALID),'Region mismatch in parity')
    need(len(p['previews'])==8 and all(float(x['max_abs_error'])<=1e-8 for x in p['previews']),'Parity invalid')
    origin=plan['original_reference']['rectangle'];padding=plan['reference_separation_grid_cells']
    for name in VALID:
        r=regions[name];rect=r['grid_rect']
        need(len(rect)==4 and rect[1]-rect[0]==rect[3]-rect[2]==500,'Bad region dimensions')
        need(sep(rect,origin,padding),f'Region overlaps legacy: {name}')
        need(r['min_valid_fraction']>=0.999999,f'Invalid field region {name}')
        need(next(x for x in p['regions'] if x['region']==name)['grid_rect']==rect,'Region rectangle disagreement')
    for i,a in enumerate(VALID):
        for b in VALID[i+1:]:need(sep(regions[a]['grid_rect'],regions[b]['grid_rect'],padding),f'Regions not separated {a}/{b}')
    windows=inv['times']['windows'];need(len(windows)==8,'Wrong number of inventory time windows')
    need(tuple(x['requested_start'] for x in windows)==EXPECTED_ANCHORS,'Time starts changed')
    need(tuple(int(x['index_start']) for x in windows)==EXPECTED_STARTS,'Timestamp index regression')
    need(all(x['available'] and int(x['index_end'])==int(x['index_start'])+167 for x in windows),'Noncontiguous time blocks')
    hashes={'phase7_lxxxvi_inventory':inv_s,'phase7_lxxxvii_screen':screen_s,'phase7_lxxxviii_parity':prev_s,
        'lxxxviii_source_sha256':sha(old),'split_script_sha256':sha(HERE),'split_plan_sha256':sha(PKG/'split_plan.json')}
    counts={'primary_fields_per_region':4*168,'replication_fields_per_region':4*168,
        'primary_development_fields':2*4*168,'primary_locked_test_fields':2*4*168,
        'replication_development_fields':2*4*168,'replication_locked_test_fields':2*4*168}
    print('[SPLIT-PLAN] development='+','.join(dev)+' external_test='+','.join(test))
    print('[SEPARATION] PASS: four full-500² footprints mutually disjoint and >=80 WTK grid cells from legacy')
    print('[SEASONS] PASS: 4×168 hours in 2010 + 4×168 in 2012, per region')
    print('[PILOT-SCOPE] PASS: four development-only previews; locked-region preview files forbidden for model pilot')
    return plan,regions,windows,hashes,counts

def freeze(root,out):
    need(not out.exists(),f'Refusing overwrite of frozen split: {out}')
    plan,reg,win,h,c=inspect(root)
    out.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='.'+out.name+'.staging-',dir=out.parent))
    try:
        doc={'schema_version':1,'status':'FROZEN_BEFORE_MODEL_SCORES','frozen_split':plan,
            'source_digests':h,'region_metadata':reg,'time_windows':win,'expected_counts':c,
            'pretraining_novelty_proven':False,'no_inference_or_model_scoring':True}
        (stage/'locked_region_split.json').write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n')
        lines=['# Phase 7 — LXXXIX locked geographic evaluation split','',
          '**Frozen before any Phase 7 model predictions or rankings were inspected.**','',
          '| Role | Region | WTK grid rectangle |','|---|---|---|']
        for role,names in [('Development',plan['development_regions']),('Locked external test',plan['locked_external_test_regions'])]:
            for name in names:lines.append(f'| {role} | `{name}` | `{reg[name]["grid_rect"]}` |')
        lines+=['','- Primary evaluation: 2010 February/May/August/November, 168 hourly fields each per region.',
                '- Replication: same four seasonal weeks in 2012; do not reselect after looking at 2010 performance.',
                '- Six frozen model checkpoints; no checkpoint/loss choice may depend on locked external-test scores.',
                '- Only four specified development-region previews may enter the initial model-inference pilot.',
                '- Historical 2007 crop-based scores remain development context, not a pristine test.',
                '- Original PhIRE pretraining membership is unknown; do **not** describe regions as unseen during all training.',
                '- Geographic regime labels are descriptive hypotheses, not independently confirmed land/terrain labels.',
                '- Before final external-test evaluation, freeze inference, metrics, radii, crop positions, sample inclusion and decision criteria.','',
                '## Expected field counts','']
        for key,val in c.items():lines.append(f'- `{key}`: {val}')
        (stage/'locked_region_split.md').write_text('\n'.join(lines)+'\n')
        (stage/'SHA256SUMS').write_text(''.join(f'{sha(stage/n)}  {n}\n' for n in ('locked_region_split.json','locked_region_split.md')))
        need(not out.exists(),'Output appeared during freeze')
        stage.rename(out)
    except BaseException:
        shutil.rmtree(stage,ignore_errors=True);raise
    print('[SPLIT-FREEZE] PASS: isolated immutable SHA-indexed region assignment; no inference, model rankings or training')
    print('[OUTPUT]',out)

def verify(root,out):
    plan,reg,win,h,c=inspect(root)
    sums(out,('locked_region_split.json','locked_region_split.md'))
    doc=json.loads((out/'locked_region_split.json').read_text())
    need(doc['status']=='FROZEN_BEFORE_MODEL_SCORES' and doc['no_inference_or_model_scoring'],'Invalid status')
    need(doc['frozen_split']==plan and doc['source_digests']==h and doc['expected_counts']==c,'Frozen source/config changed')
    need(doc['region_metadata']==reg and doc['time_windows']==win,'Frozen metadata changed')
    print('[VERIFY-GATE] PASS: split assignment, source SHA256, region rectangles, timestamps and output hashes')

def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',required=True,choices=['preflight','freeze','verify']);p.add_argument('--repo-root',default=str(ROOT));a=p.parse_args()
    root=Path(a.repo_root).resolve();out=root/OUT
    if a.mode=='verify':verify(root,out)
    elif a.mode=='freeze':freeze(root,out)
    else:
        need(not out.exists(),f'Frozen report already exists; use --mode verify: {out}')
        inspect(root)
        print('[PREFLIGHT] PASS: sources validated; no files written, no TensorFlow, no model scores')
if __name__=='__main__':
    try:main()
    except Exception as e:
        import sys
        print('[LXXXIX] ERROR',type(e).__name__+':',e,file=sys.stderr);sys.exit(1)
