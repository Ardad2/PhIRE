#!/usr/bin/env python3
"""Phase 7 XCI: immutable evaluation-protocol freeze before locked external inference.

Preflight/verify re-run existing, previously verified XC and LXXXIX checks.
Does not load external-test fields, contact WTK, invoke TensorFlow, or score models.
"""
from __future__ import annotations
import argparse, hashlib, json, re, sys, tempfile, shutil, subprocess
from pathlib import Path

ROOT_DEFAULT=Path(__file__).resolve().parents[2]
SCRIPT_REL='phase7_xci_update/scripts/phase7_freeze_evaluation_protocol.py'
PLAN_REL='phase7_xci_update/evaluation_protocol.json'
FROZEN_REL='ttk_runs_fixed/topology_finetuning/phase7_lxxxix_locked_region_split_20261015'
XC_REL='ttk_runs_fixed/topology_finetuning/phase7_xc_development_inference_pilot_20261015'
OUT_REL='ttk_runs_fixed/topology_finetuning/phase7_xci_frozen_evaluation_protocol_20261015'
XC_SCRIPT_REL='phase7_xc_update/scripts/phase7_dev_six_model_pilot.py'
MODELS=['cnn','gan','grad','cv','pers','f1']

def need(b,msg):
    if not b:raise RuntimeError(msg)

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def hashes(path,required):
    need((path/'SHA256SUMS').is_file(),f'Missing SHA256SUMS: {path}')
    got={}
    for s in (path/'SHA256SUMS').read_text().splitlines():
        m=re.fullmatch(r'([a-f0-9]{64})  (.+)',s)
        need(m is not None,'Bad SHA256 record')
        digest,name=m.groups();p=(path/name).resolve()
        need(p.is_relative_to(path.resolve()) and p.is_file(),'Missing/outside file: '+name)
        need(name not in got,'Duplicate SHA record: '+name)
        need(sha(p)==digest,'Hash mismatch: '+name)
        got[name]=digest
    need(set(got)==set(required),f'Wrong SHA256 members: {sorted(got)}')
    return got

def check_plan(root):
    d=json.loads((root/PLAN_REL).read_text())
    need(d['status']=='PREDECLARED_BEFORE_EXTERNAL_INFERENCE' and d['schema_version']==1,'Invalid protocol')
    r=d['geographic_roles'];need(r['development']==['gulf_inland','southwest_borderland'],'Dev changed')
    need(r['locked_external_test']==['upper_midwest','pacific_inland'],'Test changed')
    need(d['models']['order']==MODELS,'Models changed')
    times=d['times'];need(times['hourly_fields_per_week']==168,'Week length changed')
    need(times['primary_week_start_utc']==['2010-02-01T00:00:00','2010-05-01T00:00:00','2010-08-01T00:00:00','2010-11-01T00:00:00'],'Primary weeks changed')
    need(times['replication_week_start_utc']==['2012-02-01T00:00:00','2012-05-01T00:00:00','2012-08-01T00:00:00','2012-11-01T00:00:00'],'Replication changed')
    t=d['metric_definitions']['canonical_topology'];need(t['primary_crop']=={'y0':0,'x0':0,'height':160,'width':160},'Crop changed')
    loc=d['metric_definitions']['spatial_locality'];need(loc['radius_grid_cells']==[4,8,16,32,64,96,128,'infinity'],'Radii changed')
    need(len(d['predeclared_comparisons'])==4,'Comparisons changed')
    return d

def sources(root):
    hashes(root/'phase7_xci_update',['README.md','evaluation_protocol.json','xci_append.md',
        'scripts/phase7_freeze_evaluation_protocol.py','patches/phase7_notes_XC_to_XCI.patch'])
    plan=check_plan(root)
    frozen=root/FROZEN_REL;xc=root/XC_REL
    hashes(frozen,['locked_region_split.json','locked_region_split.md'])
    hashes(xc,['reference_mr.npy','reference_gt.npy','development_4_paired.tfrecord',
               *(f'pred_{m}.npy' for m in MODELS),'dev_inference_report.json','dev_inference_report.md'])
    split=json.loads((frozen/'locked_region_split.json').read_text())
    need(split['status']=='FROZEN_BEFORE_MODEL_SCORES','Split status changed')
    sp=split['frozen_split']
    need(sp['development_regions']==plan['geographic_roles']['development'] and sp['locked_external_test_regions']==plan['geographic_roles']['locked_external_test'],'Frozen geography mismatch')
    need(sp['primary_windows_utc']==plan['times']['primary_week_start_utc'],'Primary window mismatch')
    need(sp['replication_windows_utc']==plan['times']['replication_week_start_utc'],'Replication mismatch')
    need(sp['hours_per_window']==plan['times']['hourly_fields_per_week'],'Window hours mismatch')
    for name,rel in [('XC',XC_SCRIPT_REL)]:
        p=root/rel;need(p.is_file(),f'Missing {name} verifier')
        run=subprocess.run([sys.executable,str(p),'--mode','verify','--repo-root',str(root)],
                           capture_output=True,text=True)
        need(run.returncode==0 and '[VERIFY-GATE] PASS' in run.stdout, f'{name} verifier failed: '+(run.stderr+run.stdout)[-3000:])
    report=json.loads((xc/'dev_inference_report.json').read_text())
    need(report['scope']=='DEVELOPMENT_ONLY_FOUR_FIELD_PILOT','XC scope changed')
    cks=report['sources']['checkpoint_sha256']
    need(sorted(cks)==sorted(MODELS),'Checkpoint set changed')
    source={
        'original_locked_split_sha256':sha(frozen/'locked_region_split.json'),
        'verified_XC_report_sha256':sha(xc/'dev_inference_report.json'),
        'XC_inference_script_sha256':sha(root/XC_SCRIPT_REL),
        'protocol_package_sha256':sha(root/PLAN_REL),
        'protocol_freeze_script_sha256':sha(root/SCRIPT_REL),
        'checkpoint_sha256':cks,
        'original_six_model_phire_sources':report['sources']['source_sha256'],
        'dev_pilot_fields':report['sources']['fields'],
    }
    need(all(f['region'] in plan['geographic_roles']['development'] for f in source['dev_pilot_fields']),'Non-dev pilot present')
    print('[SOURCE] PASS: frozen split, six XC prediction hashes and original verifier, fixed checkpoint set',flush=True)
    print('[PROTOCOL] PASS: models, primary crop, radius grid, 2010/2012 weeks, geographic roles',flush=True)
    return plan,source

def render_md(plan, source):
    return """# Phase 7 XCI — frozen evaluation protocol\n\n**Predeclared before any locked external-region model inference.**\n\nThis protocol is immutable. Read `evaluation_protocol.json` for all definitions.\n\n## Roles\n\n- Development: gulf_inland and southwest_borderland.\n- Locked external: upper_midwest and pacific_inland.\n- Historical reference: mid_atlantic_legacy (not a clean external test).\n\n## Time windows\n\n- Primary: 2010 Feb/May/Aug/Nov, 168 hourly fields per region-week.\n- Replication: 2012 Feb/May/Aug/Nov, same 168-hour windows per region.\n- Aggregation: paired within-field; block-level region-week means first; n=8 region-weeks per role per year.\n\n## Predeclared outcomes\n\n- Full 500x500 reconstruction UV RMSE and speed MAE.\n- **Primary topology:** original top-left 160x160 speed crop; corrected GUDHI dB, W2inf, W22, frozen Phase5E W22 locality radius grid [4,8,16,32,64,96,128,infinity], GT mass coverage.\n- **Secondary:** nine disjoint 160x160 tiles (starting x/y 0,160,320), reported separately.\n- **Development-only exploratory:** full 500x500 topology feasibility and independently checked merge-tree distances.\n- Six frozen checkpoints: cnn, gan, grad, cv, pers, f1.\n\n## Rules\n\n- No model/metric/window selection in response to external test scores.\n- If development gates require code changes, issue a versioned protocol amendment **before** external inference; never rewrite this freeze.\n- No pooled-hour independence claims. Document errors and missing records rather than silently excluding them.\n- No claim that the original PhIRE pretraining has excluded the new geography.\n\n## Provenance\n\n- Original split SHA256: `{split}`\n- XC inference-report SHA256: `{xc}`\n- XCI plan SHA256: `{plan}`\n""".format(split=source['original_locked_split_sha256'],xc=source['verified_XC_report_sha256'],plan=source['protocol_package_sha256'])

def freeze(root,out,plan,source):
    need(not out.exists(),f'Refuse overwrite; use --mode verify: {out}')
    out.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='.'+out.name+'.staging-',dir=out.parent))
    try:
        record={'schema_version':1,'status':'FROZEN_BEFORE_EXTERNAL_INFERENCE',
            'protocol':plan,'sources':source}
        (stage/'evaluation_protocol.json').write_text(json.dumps(record,indent=2,sort_keys=True)+'\n')
        (stage/'evaluation_protocol.md').write_text(render_md(plan,source))
        files=['evaluation_protocol.json','evaluation_protocol.md']
        (stage/'SHA256SUMS').write_text(''.join(f'{sha(stage/f)}  {f}\n' for f in files))
        need(not out.exists(),'Unexpected output appeared');stage.rename(out)
    except BaseException:
        shutil.rmtree(stage,ignore_errors=True);raise
    print('[PROTOCOL-FREEZE] PASS: immutable versioned protocol; no WTK access, TensorFlow, inference, training or external scores',flush=True)
    print('[OUTPUT]',out,flush=True)

def verify(out,plan,source):
    hashes(out,['evaluation_protocol.json','evaluation_protocol.md'])
    r=json.loads((out/'evaluation_protocol.json').read_text())
    need(r=={'schema_version':1,'status':'FROZEN_BEFORE_EXTERNAL_INFERENCE',
             'protocol':plan,'sources':source},'Frozen protocol sources or content changed')
    need((out/'evaluation_protocol.md').read_text()==render_md(plan,source),'Markdown not reproducible')
    print('[VERIFY-GATE] PASS: all protocol hashes, model/checkpoint lineage, region/time scope and immutable render',flush=True)

def selftest():
    import tempfile
    with tempfile.TemporaryDirectory() as z:
        base=Path(z); (base/'a').write_text('abc')
        need(sha(base/'a')==hashlib.sha256(b'abc').hexdigest(),'SHA regression')
        (base/'SHA256SUMS').write_text(sha(base/'a')+'  a\n')
        hashes(base,['a']);(base/'a').write_text('wrong')
        try:hashes(base,['a'])
        except RuntimeError:pass
        else:raise RuntimeError('Tamper check failed')
    print('[SELFTEST] PASS: hashing, verified manifest and intentional corruption',flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',required=True,choices=['selftest','preflight','freeze','verify'])
    p.add_argument('--repo-root',default=str(ROOT_DEFAULT));a=p.parse_args()
    if a.mode=='selftest':selftest();return
    root=Path(a.repo_root).resolve();out=root/OUT_REL
    plan,source=sources(root)
    if a.mode=='preflight':
        need(not out.exists(),f'Output exists; use --mode verify: {out}')
        print('[PREFLIGHT] PASS: no new files, no locked test records read; ready to freeze',flush=True)
    elif a.mode=='freeze':freeze(root,out,plan,source)
    else:verify(out,plan,source)

if __name__=='__main__':
    try:main()
    except Exception as e:
        print('[XCI] ERROR',type(e).__name__+':',e,file=sys.stderr);sys.exit(1)
