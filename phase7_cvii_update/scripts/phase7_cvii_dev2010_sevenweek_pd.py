#!/usr/bin/env python3
"""Phase 7 CVII: seven development-only 2010 region-weeks of canonical TTK PDs.

Uses ORIGINAL, SHA-pinned CII atomic 168h VTI/PD machinery without modifying
its source on disk. Week-specific frozen identities are bound in memory from
verified CVI outputs and XCVII's exact 16 seasonal model/GT diagram anchors.
Do not use on any locked region. CII structural, not numerical, PD verification.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

BASE = Path('ttk_runs_fixed/topology_finetuning')
CVI_SCRIPT = Path('phase7_cvi_update/scripts/phase7_cvi_dev2010_sevenweek_inference.py')
CII_SCRIPT = Path('phase7_cii_update/scripts/phase7_cii_dev_fullweek_pd.py')
# Pins from the *previously released* original workpacks. DO NOT relax these.
CVI_SHA = '3cf69b42c57a9ce16b6d37ca97285b8b54ce3696cc93d0ea6141a124a491b647'
CII_SHA = 'f0e5ae86874c7e4d853320ce0bdc45b2a4fbfec3efc9b7ce6c7630edbdc39f2d'
DATE_TAG = '20261020'
ARMS = ('gt','cnn','gan','grad','cv','pers','f1')
SCHEMA = 'phase7_cvii_dev2010_seven_week_canonical_pd168_v1'

def need(ok, message):
    if not ok: raise RuntimeError(message)

def sha(path):
    d = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1<<20), b''): d.update(chunk)
    return d.hexdigest()

def module(path, name):
    spec=importlib.util.spec_from_file_location(name,path)
    need(spec is not None and spec.loader is not None, f'Cannot load module {path}')
    x=importlib.util.module_from_spec(spec)
    sys.modules[name]=x
    spec.loader.exec_module(x)
    return x

def output_name(region, week):
    return f'phase7_cvii_dev_{region}_{week[:10].replace("-", "")}_pd168_{DATE_TAG}'

def setup(root):
    for path, expected in ((CVI_SCRIPT,CVI_SHA),(CII_SCRIPT,CII_SHA)):
        need((root/path).is_file() and sha(root/path)==expected,
             f'Frozen driver missing/modified: {path}; stop, do not edit historical source')
    cvi=module(root/CVI_SCRIPT,'cvii_frozen_cvi')
    cii=module(root/CII_SCRIPT,'cvii_frozen_cii')
    need(tuple(cii.ARMS)==ARMS and cii.N==168 and cii.STAGE_SUFFIX=='.staging',
         'Original CII diagram schema or atomic staging changed')
    cvi_ctx=cvi.setup(root)
    ci,cv,c,lookup,xcvi,xc=cvi_ctx
    original=cii.sources(root) # Includes historic Feb CI and XCVII checksum checks.
    need(tuple(original[1].ARMS)==ARMS and len(cv.weeks())==7,'Original XCVII development diagram inventory changed')
    need(tuple(cv.REGIONS)==('gulf_inland','southwest_borderland'), 'Frozen development roles changed')
    print('[SOURCES] PASS: pinned original CVI/CII drivers, seven CV sources, XCVII historical diagrams, original February closeout',flush=True)
    return cvi, cii, cvi_ctx, original

def bind(root,cvi,cii,ctx,original,region,week):
    ci,cv,c,lookup,xcvi,xc=ctx
    need((region,week) in cv.weeks(), 'Requested week is not one of seven 2010 development-only weeks')
    output,stage,cm,lineage=cvi.bind(root,ci,cv,c,lookup,xcvi,xc,region,week)
    cvi.verify_bound(root,ci,c,xc,cm,lineage,output,stage)
    cfg=json.loads((output/'run_config.json').read_text())
    raw_map=cfg['xcvi_hour_regression_mapping']
    need(set(raw_map)=={'0','84'},'CVI historical anchor mapping absent')
    mapping={0:int(raw_map['0']),84:int(raw_map['84'])}
    need(len(set(mapping.values()))==2 and all(0<=i<16 for i in mapping.values()), 'Historical XCVI mapping out of range')
    # XCVII indexed the exact same 16 XCVI seasonal samples; check their arrays
    # against CVI predictions at BOTH hours in original CII. No index guessing.
    cii.CI=BASE/cvi.output_name(region,week)
    cii.OUT=BASE/output_name(region,week)
    cii.SCHEMA=SCHEMA
    cii.ANCHORS=mapping
    xci_path=root/original[1].XCI
    original_ci,xcvii,xml,base_lineage,ci_hash,xcvii_hash=original
    need(json.loads((xci_path/'evaluation_protocol.json').read_text())['status']=='FROZEN_BEFORE_EXTERNAL_INFERENCE',
         'Geographic protocol no longer frozen')
    source={
        'cvii_driver_sha256':sha(Path(__file__).resolve()),
        'frozen_cvi_driver_sha256':CVI_SHA,
        'frozen_cii_driver_sha256':CII_SHA,
        'cvi_SHA256SUMS_sha256':sha(output/'SHA256SUMS'),
        'cvi_inference_manifest_sha256':sha(output/'fullweek_inference_manifest.json'),
        'cvi_run_config_sha256':sha(output/'run_config.json'),
        'xcvii_anchor_SHA256SUMS_sha256':sha(root/cii.XCVII/'SHA256SUMS'),
        'xci_protocol_sha256':sha(xci_path/'evaluation_protocol.json'),
        'canonical_converter_sha256':sha(root/cii.CONVERTER),
        'original_feb_cii_script_sha256':CII_SHA,
        'original_feb_cii_ancestry':base_lineage,
    }
    outcfg={
        'schema_version':1,'scope':SCHEMA,'region':region,'week_start_utc':week,
        'hours':168,'wtk_range':[cfg['wtk_first'],cfg['wtk_last']],
        'arms':list(ARMS),'source_sha256':source,
        'crop':{'x0':0,'y0':0,'height':160,'width':160},
        'ttk_command':'ttkPersistenceDiagramCmd -t 1 -i <vti> -a wind_speed -o <prefix>',
        'anchor_hours':{str(k):v for k,v in mapping.items()},
        'numerical_pd_distances_computed':False,
        'historical_XCVII_anchor_indices':list(mapping.values()),
        'no_external_locked_region':True
    }
    def current_sources(r):
        need(Path(r).resolve()==root,'Repo root changed during CII extraction')
        need(sha(output/'SHA256SUMS')==source['cvi_SHA256SUMS_sha256'] and
             sha(output/'fullweek_inference_manifest.json')==source['cvi_inference_manifest_sha256'] and
             sha(output/'run_config.json')==source['cvi_run_config_sha256'],
             'Input CVI output changed after binding')
        return original_ci,xcvii,xml,source,ci_hash,xcvii_hash
    def current_config(_lineage):
        need(_lineage==source, 'Bound source/checkout lineage changed')
        return outcfg
    def current_report(rows):
        head=[f'# Phase 7 Part CVII — {region} {week[:10]} canonical persistence extraction','',
              f'Frozen development-only region-week {region}, {week} through 168 consecutive UTC hours.',
              '**This is one region-week, not independent geographic generalization.**','',
              '168 hours x (GT + six frozen models) = 1176 port-0 VTI/VTU pairs.',
              '14 XCVII hour-0/hour-84 diagram pairs bitwise reused; 1162 freshly extracted by unchanged CII TTK command.','',
              '| Source | Reused bitwise | Newly extracted | Total |', '|---|---:|---:|---:|']
        for arm in ARMS:
            subset=[r for r in rows if r['arm']==arm]
            n=sum(r['origin']=='XCVII_REUSED_BITWISE' for r in subset)
            head.append(f'| {arm} | {n} | {len(subset)-n} | {len(subset)} |')
        head+=['', 'The fixed top-left 160x160 scalar-speed field is used; VTK x=columns, y=rows.',
               'TTK port-0 files receive structural XML checking; **numerical PD decoding remains for the next gate**.',
               'All model arrays and XCVII anchor predictions are pinned by SHA256 and numeric equality.',
               'No external locked test, trained-model modification, new inference, or numerical GUDHI/locality scoring.',
               'Original PhIRE checkpoint training-data exposure remains unverified.']
        return '\n'.join(head)+'\n'
    cii.sources=current_sources
    cii.config=current_config
    cii.report=current_report
    print(f'[BIND] PASS: {region} {week} XCVII hour0/84 indices {mapping[0]}/{mapping[84]}; 7-arm immutable output',flush=True)
    return root/cii.OUT

def invoke(cii,root,mode,max_new_pairs=None):
    saved=sys.argv
    args=[str(root/CII_SCRIPT),'--repo-root',str(root),'--mode',mode]
    if max_new_pairs is not None:args.extend(['--max-new-pairs',str(max_new_pairs)])
    try:
        sys.argv=args
        cii.main()
    finally:
        sys.argv=saved

def selftest():
    need(len(set(output_name(r,w) for r in ('gulf_inland','southwest_borderland')
         for w in ('2010-02-01T00:00:00','2010-05-01T00:00:00','2010-08-01T00:00:00','2010-11-01T00:00:00')))==8,
         'PD destinations not unique')
    need(len(ARMS)==7 and 168*len(ARMS)==1176 and 1176*2+2==2354,
         'Frozen pair/manifest cardinality drift')
    need(CVI_SHA!='' and CII_SHA!='','Source pins absent')
    print('[SELFTEST] PASS: seven development identities, 1176 pairs/week, historical anchors and unique immutable destinations',flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',required=True,choices=('selftest','inventory','preflight','extract','verify',
                                                   'preflight-all','extract-all','verify-all'))
    p.add_argument('--repo-root',default=str(Path(__file__).resolve().parents[2]))
    p.add_argument('--region',choices=('gulf_inland','southwest_borderland'))
    p.add_argument('--week-start',choices=('2010-02-01T00:00:00','2010-05-01T00:00:00',
                                          '2010-08-01T00:00:00','2010-11-01T00:00:00'))
    p.add_argument('--max-new-pairs',type=int)
    a=p.parse_args()
    if a.mode=='selftest':selftest();return
    need(a.max_new_pairs is None or (a.mode in ('extract','extract-all') and 1<=a.max_new_pairs<=1176),
         'max-new-pairs only for extraction (1..1176)')
    all_mode=a.mode=='inventory' or a.mode.endswith('-all')
    if all_mode:need(a.region is None and a.week_start is None, 'All/inventory modes cannot select a week')
    else:need(a.region is not None and a.week_start is not None,'Select a frozen development region/week')
    root=Path(a.repo_root).expanduser().resolve()
    need(root.is_dir(),'Missing PhIRE root')
    os.chdir(root);sys.path.insert(0,str(root))
    cvi,cii,ctx,original=setup(root)
    weeks=ctx[1].weeks() if all_mode else [(a.region,a.week_start)]
    for region,week in weeks:
        out=bind(root,cvi,cii,ctx,original,region,week)
        stage=out.parent/('.'+out.name+cii.STAGE_SUFFIX)
        if a.mode=='inventory':
            status='SEALED' if out.is_dir() else 'STAGED' if stage.is_dir() else 'PENDING'
            print(f'[WEEK] {region} {week} state={status}',flush=True)
            continue
        op=a.mode.removesuffix('-all')
        if op=='extract' and out.is_dir():
            invoke(cii,root,'verify')
            print(f'[SKIP-SEALED] {region} {week}',flush=True)
        else:
            invoke(cii,root,op,a.max_new_pairs)
    if a.mode.endswith('-all'):
        sealed=sum((root/BASE/output_name(r,w)).is_dir() for r,w in ctx[1].weeks())
        if a.mode=='verify-all' or (a.mode=='extract-all' and a.max_new_pairs is None):
            need(sealed==7,f'Expected 7 sealed outputs, found {sealed}')
        print(f'[ALL-{a.mode.upper()}] PASS: {sealed}/7 sealed canonical PD weeks; zero locked regions',flush=True)

if __name__=='__main__':
    try:main()
    except Exception as e:
        print(f'[CVII] ERROR: {type(e).__name__}: {e}',file=sys.stderr,flush=True)
        raise SystemExit(1)
