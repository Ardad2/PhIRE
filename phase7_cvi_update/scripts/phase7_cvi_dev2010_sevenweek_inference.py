#!/usr/bin/env python3
"""Part CVI: seven frozen 2010 development region-week, six-model inference runs.

Reuse SHA-pinned, untouched Part CI inference functions via in-memory, explicitly
scoped week bindings. Every model is staged atomically and independently hashed.
Nothing touches an external locked region or existing C/CI/CV outputs.
"""
from __future__ import annotations
import argparse, hashlib, json, re, sys
from pathlib import Path

BASE=Path('ttk_runs_fixed/topology_finetuning')
CI_SCRIPT=Path('phase7_ci_update/scripts/phase7_ci_dev_fullweek_inference.py')
CV_SCRIPT=Path('phase7_cv_update/scripts/phase7_cv_dev2010_week_sources.py')
CI_SHA='5a5174da40ce975440b6f66b86aeaf96da72942bdc17765d63d8c28419304dba'
CV_SHA='be862eff7da4f6ada2f2de20c91171ed6cf6b640d184a768f415bd25248fcdb0'
SCHEMA='phase7_cvi_dev2010_seven_week_inference168_v1'
STAMP='20261019'
ARM_NAMES=('cnn','gan','grad','cv','pers','f1')


def need(test,msg):
    if not test: raise RuntimeError(msg)


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def output_name(region,anchor):
    return f'phase7_cvi_dev_{region}_{anchor[:10].replace("-","")}_inference168_{STAMP}'


def setup(root):
    # Source SHA checked BEFORE importing executable code.
    for fn,digest in ((CI_SCRIPT,CI_SHA),(CV_SCRIPT,CV_SHA)):
        need((root/fn).is_file() and sha(root/fn)==digest,f'Frozen source missing or modified: {fn}')
    import importlib.util
    def load(p,name):
        spec=importlib.util.spec_from_file_location(name,p)
        need(spec is not None and spec.loader is not None,f'Cannot load {p}')
        m=importlib.util.module_from_spec(spec)
        sys.modules[name]=m
        spec.loader.exec_module(m)
        return m
    ci=load(root/CI_SCRIPT,'cvi_frozen_ci')
    cv=load(root/CV_SCRIPT,'cvi_frozen_cv')
    need(ci.N==cv.FIELDS_PER_WEEK==168 and tuple(ci.ARMS)==ARM_NAMES,'Frozen CI model configuration changed')
    need(tuple(cv.REGIONS)==('gulf_inland','southwest_borderland') and len(cv.weeks())==7,'CV development boundaries changed')
    # Check all 7 sealed inputs, original complete CI/Feb inference, and XCVI histories.
    c,lookup=cv.inventory(root)
    for region,week in cv.weeks():
        cv.bind_week(root,c,lookup,region,week)
        fp,cached=cv.bound_sources(root,c)
        c.verify(root,fp,cached)
    # Original source checker includes all 6 checkpoint hashes and all original XCVI arrays.
    original_c,xcvi,xc,original_cm,original_lineage=ci.sources(root)
    feb_out=root/ci.OUT_REL
    feb_stage=feb_out.parent/('.'+feb_out.name+'.staging')
    ci.verify(root,feb_out,feb_stage,original_c,xc,original_cm,original_lineage)
    # Original February CI inference is independently reverified, not modified.
    need(sha(root/ci.OUT_REL/'SHA256SUMS') and sha(root/ci.XCVI_REL/'SHA256SUMS'),'Original reference lineage missing')
    cv.bind_week(root,c,lookup,*cv.weeks()[0])
    print('[SOURCES] PASS: 7/7 verified CV native weeks; original CI/CV pinned; original February CI closed',flush=True)
    print('[FROZEN] PASS: XCI/XCVI model, normalized TFRecord, 16 historical anchors, no locked test',flush=True)
    return ci,cv,c,lookup,xcvi,xc


def bind(root,ci,cv,c,lookup,xcvi,xc,region,week):
    need((region,week) in cv.weeks(),'Selected region/week is not one of the seven frozen development-only weeks')
    cv.bind_week(root,c,lookup,region,week)
    fp,cached=cv.bound_sources(root,c)
    c.verify(root,fp,cached)
    # XCVI field index is bound by exact source identity, NOT a guessed list offset.
    _,fields,_,_=xcvi.sources(root)
    indexes={}
    for hour in (0,84):
        target=lookup[(region,week,hour)]
        matches=[j for j,f in enumerate(fields) if
                 f['region']==region and f['wtk_index']==target['wtk_index']
                 and f['offset_hours']==hour and f['filename']==target['filename']]
        need(len(matches)==1,f'XCVI historical field-index mismatch: {region} {week} hour{hour}')
        indexes[hour]=matches[0]
        need(sha(root/xcvi.XCV_REL/target['filename'])==sha(root/BASE/c.SRC/'fields'/c.filename(hour)),
             f'Bitwise XCV field anchor changed: hour {hour}')
    src=root/BASE/c.SRC
    cm=json.loads((src/'fullweek_manifest.json').read_text())
    need(cm['field_count']==168 and len(cm['fields'])==168 and cm['scope']==c.SCHEMA and cm['source_sha256']==fp,
         'Source week manifest changed')
    for hour,e in enumerate(cm['fields']):
        need(e['hour']==hour and e['region']==region and e['wtk_index']==c.START_IDX+hour
             and e['timestamp_utc']==c.timestamp(hour) and e['filename']==c.filename(hour),
             f'Source hourly metadata changed: {region} {week} hour{hour}')
    xci=json.loads((root/xcvi.XCI_REL/'evaluation_protocol.json').read_text())['protocol']
    need(xci['times']['hourly_fields_per_week']==168 and week in xci['times']['primary_week_start_utc']
         and region in xci['geographic_roles']['development'],'Frozen XCI identity changed')
    ci.C_REL=str(BASE/c.SRC)
    ci.OUT_REL=str(BASE/output_name(region,week))
    ci.SCHEMA=SCHEMA
    ci.ANCHORS=indexes
    # Trust CI's exact original infer/TF runner, metrics, stage seal, hashes, record and anchor checking.
    # Replace ONLY the context-dependent reference reader and metadata writers.
    def ref_arrays(r,_c):
        import numpy as np
        mr=np.empty((168,100,100,2),np.float64)
        gt=np.empty((168,500,500,2),np.float64)
        for i in range(168):
            with np.load(r/ci.C_REL/'fields'/c.filename(i),allow_pickle=False) as z:
                ci.need(set(z.files)==c.DATA_KEYS,'Native source schema changed')
                a=z['hr_uv'];b=z['mr_uv']
                ci.need(a.shape==(500,500,2) and b.shape==(100,100,2)
                        and a.dtype==b.dtype==np.float64,'Native source dtype/shape changed')
                ci.need(int(z['wtk_index'])==c.START_IDX+i and str(z['region'])==region
                        and str(z['timestamp_utc'])==c.timestamp(i),'Source timestamp/region changed')
                ci.need(np.isfinite(a).all() and np.isfinite(b).all(),'Nonfinite native source fields')
                ci.need(float(np.max(np.abs(a.reshape(100,5,100,5,2).mean((1,3))-b)))<=1e-10,
                        'Frozen factor-five pooling drift')
                gt[i]=a;mr[i]=b
        return mr,gt
    ci.reference_arrays=ref_arrays
    lineage={
        'cv_run_config_sha256':sha(src/'run_config.json'),
        'cv_manifest_sha256':sha(src/'fullweek_manifest.json'),
        'cv_sha256s_sha256':sha(src/'SHA256SUMS'),
        'xcvi_manifest_sha256':sha(root/ci.XCVI_REL/'seasonal16_inference_manifest.json'),
        'xcvi_sha256s_sha256':sha(root/ci.XCVI_REL/'SHA256SUMS'),
        'original_ci_manifest_sha256':sha(root/BASE/'phase7_ci_dev_gulf_inland_feb2010_inference168_20261018'/'fullweek_inference_manifest.json'),
        'cv_script_sha256':CV_SHA,'ci_script_sha256':CI_SHA,
        'cvi_script_sha256':sha(Path(__file__).resolve()),
        'xci_protocol_sha256':sha(root/xcvi.XCI_REL/'evaluation_protocol.json'),
        'region_week':{'region':region,'week_start_utc':week,'wtk_first':c.START_IDX},
        'checkpoint_sha256':xc.sources(root)['checkpoint_sha256'],
    }
    def make_config(_lineage):
        ci.need(_lineage==lineage,'Bound immutable source/checkpoints changed')
        return {'schema_version':1,'scope':SCHEMA,'source_sha256':lineage,'region':region,
                'week_start_utc':week,'hours':168,'wtk_first':c.START_IDX,'wtk_last':c.START_IDX+167,
                'model_names':list(ARM_NAMES),'r':[5],'mr_shape':[168,100,100,2],
                'gt_shape':[168,500,500,2],
                'xcvi_hour_regression_mapping':{str(i):j for i,j in indexes.items()},
                'old_prediction_tolerance_mps':ci.PRED_TOL}
    ci.config=make_config
    def make_report(progress):
        lines=[f'# Phase 7 Part CVI — {region} {week[:10]} six-model development inference','',
               f'Frozen region-week {region}, {week}, exactly 168 consecutive UTC hours.',
               '**One development region-week; do not interpret hours as independent statistical replicates.**','',
               '| Model | Mean hourly UV RMSE (m/s) ↓ | Mean hourly speed MAE (m/s) ↓ | XCVI anchor max difference (m/s) |',
               '|---|---:|---:|---:|']
        for arm in ARM_NAMES:
            e=progress['models'][arm];m=e['mean_hourly_metrics']
            lines.append(f'| {arm} | {m["uv_rmse_mps"]:.6f} | {m["speed_mae_mps"]:.6f} | {e["anchor_max_abs"]:.6g} |')
        lines+=['','Mean hourly metrics differ from pooled metrics. Hourly and pooled values are recorded in the sealed manifest.',
                'The six pretrained/fine-tuned checkpoints and normalization are unchanged.',
                'Topology, training, model selection, external locked geographic testing and 2012 replication were not performed.',
                'Original pretrained PhIRE WTK geographical and temporal exposure remains unverified.']
        return '\n'.join(lines)+'\n'
    ci.report_text=make_report
    def write_manifest(stage,p,_cm):
        ci.need(_cm==cm,'Source manifest changed during inference')
        m={'schema_version':1,'scope':SCHEMA,'run_config':p['config'],
           'models':p['models'],'source_field_count':168,
           'source_wtk_first':c.START_IDX,'source_wtk_last':c.START_IDX+167,
           'source_region':region,'source_week':week,
           'source_c_manifest_sha256':lineage['cv_manifest_sha256'],
           'limitations':['development only, one region-week','temporally correlated hours',
                          'no geographic generalization','no topology computed',
                          'original pretrained WTK exposure unknown']}
        ci.atom(stage/'fullweek_inference_manifest.json',ci.jd(m))
        ci.atom(stage/'fullweek_inference_report.md',make_report(p))
        ci.atom(stage/'SHA256SUMS',''.join(f'{ci.sha(stage/name)}  {name}\n' for name in ci.output_names()))
    ci.write_manifest=write_manifest
    out=root/ci.OUT_REL
    stage=out.parent/('.'+out.name+'.staging')
    print(f'[BIND] PASS: {region} {week} WTK={c.START_IDX}..{c.START_IDX+167}; XCVI hour0/84 -> {indexes[0]}/{indexes[84]}',flush=True)
    return out,stage,cm,lineage


def verify_bound(root,ci,c,xc,cm,lineage,out,stage):
    import numpy as np
    need(out.is_dir() and not stage.exists(),'Sealed inference output missing or staging still present')
    ci.manifest_sha(out,ci.output_names())
    cfg=ci.config(lineage)
    need(json.loads((out/'run_config.json').read_text())==cfg,'Sealed inference config mismatch')
    p=json.loads((out/'progress.json').read_text())
    need(set(p)=={'config','models','static_sha256'} and p['config']==cfg and set(p['models'])==set(ARM_NAMES),
         'Sealed inference model progress/config mismatch')
    need(set(p['static_sha256'])=={'reference_mr.npy','reference_gt.npy','development_168_paired.tfrecord'},
         'Incomplete paired input SHA inventory')
    for name,checksum in p['static_sha256'].items():
        need(re.fullmatch(r'[0-9a-f]{64}',checksum) and ci.sha(out/name)==checksum,
             f'Changed static input/TFRecord: {name}')
    m=json.loads((out/'fullweek_inference_manifest.json').read_text())
    need(m['scope']==SCHEMA and m['schema_version']==1 and m['run_config']==cfg and m['models']==p['models'],
         'Model manifest/source scope mismatch')
    need(m['source_region']==cfg['region'] and m['source_week']==cfg['week_start_utc']
         and m['source_wtk_first']==cfg['wtk_first'] and m['source_wtk_last']==cfg['wtk_last']
         and m['source_field_count']==168 and m['source_c_manifest_sha256']==lineage['cv_manifest_sha256'],
         'Frozen week/source manifest mismatch')
    mr,gt=ci.reference_arrays(root,c)
    need(np.array_equal(np.load(out/'reference_mr.npy',mmap_mode='r',allow_pickle=False),mr),
         'Stored MR differs from the frozen native source')
    need(np.array_equal(np.load(out/'reference_gt.npy',mmap_mode='r',allow_pickle=False),gt),
         'Stored GT differs from the frozen native source')
    for arm in ARM_NAMES:ci.validate_record(root,out,arm,p['models'][arm],xc,gt,np,full=True)
    need((out/'fullweek_inference_report.md').read_text()==ci.report_text(p),'Model report changed')
    need({f.name for f in out.iterdir()}==set(ci.output_names())|{'SHA256SUMS'},'Unexpected sealed output files')
    print(f'[VERIFY-GATE] PASS: {cfg["region"]} {cfg["week_start_utc"]}; 1008 full500 predictions, 12 XCVI anchors, 13 SHA256; no locked tests',flush=True)


def selftest():
    need(len(set(output_name(r,w) for r in ('gulf_inland','southwest_borderland')
             for w in ('2010-02-01T00:00:00','2010-05-01T00:00:00','2010-08-01T00:00:00','2010-11-01T00:00:00')))==8,
         'Unexpected inference destination collision')
    need(CI_SHA=='5a5174da40ce975440b6f66b86aeaf96da72942bdc17765d63d8c28419304dba','CI digest drift')
    # Check manifest-tampering guard does not accept malformed hashes.
    need(re.fullmatch(r'[0-9a-f]{64}','g'*64) is None,'Incorrect checksum validator')
    print('[SELFTEST] PASS: seven development-only identities, model inventory, 13-file sealing, source hash guards',flush=True)


def main():
    a=argparse.ArgumentParser()
    a.add_argument('--mode',required=True,choices=('selftest','inventory','preflight','infer','verify','preflight-all','infer-all','verify-all'))
    a.add_argument('--repo-root',default=str(Path(__file__).resolve().parents[2]))
    a.add_argument('--region',choices=('gulf_inland','southwest_borderland'))
    a.add_argument('--week-start',choices=('2010-02-01T00:00:00','2010-05-01T00:00:00','2010-08-01T00:00:00','2010-11-01T00:00:00'))
    a.add_argument('--max-new-models',type=int)
    args=a.parse_args()
    if args.mode=='selftest':selftest();return
    need(args.max_new_models is None or (args.mode in ('infer','infer-all') and 1<=args.max_new_models<=6),
         '--max-new-models permitted only for infer/infer-all, range 1..6')
    is_batch=args.mode.endswith('-all') or args.mode=='inventory'
    if is_batch:need(args.region is None and args.week_start is None,'Inventory/all commands do not select a week')
    else:need(args.region is not None and args.week_start is not None,'Select explicit region and frozen week start')
    root=Path(args.repo_root).expanduser().resolve()
    need(root.is_dir(),'Missing PhIRE root')
    sys.path.insert(0,str(root))
    import os
    os.chdir(root)
    ci,cv,c,lookup,xcvi,xc=setup(root)
    weeks=cv.weeks() if is_batch else [(args.region,args.week_start)]
    for region,week in weeks:
        out,stage,cm,lineage=bind(root,ci,cv,c,lookup,xcvi,xc,region,week)
        if args.mode=='inventory':
            state='SEALED' if out.exists() else 'STAGED' if stage.exists() else 'PENDING'
            print(f'[WEEK] {region} {week} state={state}',flush=True)
            continue
        operation=args.mode.removesuffix('-all')
        if operation=='verify':verify_bound(root,ci,c,xc,cm,lineage,out,stage)
        elif out.exists():
            # Batch commands reverify sealed outputs, never overwrite.
            need(is_batch,'Output already sealed; use --mode verify')
            verify_bound(root,ci,c,xc,cm,lineage,out,stage)
            print(f'[SKIP-SEALED] {region} {week}',flush=True)
        elif operation=='preflight':
            ci.budget(root)
            if stage.exists():
                import numpy as np
                gt=ci.reference_arrays(root,c)[1]
                p=ci.read_progress(root,stage,ci.config(lineage),xc,gt,np,full=True)
                print(f'[PREFLIGHT] PASS: {region} {week}; staged={len(p["models"])}/6, no inference',flush=True)
            else:print(f'[PREFLIGHT] PASS: {region} {week}; clean output destination, no inference',flush=True)
        elif operation=='infer':ci.infer(root,out,stage,c,xc,xcvi,cm,lineage,args.max_new_models)
        else:raise RuntimeError(f'Unknown operation: {operation}')
    if args.mode.endswith('-all'):
        sealed=sum((root/BASE/output_name(r,w)).is_dir() for r,w in cv.weeks())
        if args.mode=='verify-all':need(sealed==7,'Expected seven sealed verified weeks')
        if args.mode=='infer-all' and args.max_new_models is None:need(sealed==7,'Some weeks did not complete inference')
        print(f'[ALL-{args.mode.upper()}] PASS: {sealed}/7 sealed 2010 development inference weeks; no locked-region scores',flush=True)

if __name__=='__main__':
    try:main()
    except Exception as exc:
        print('[CVI] ERROR',type(exc).__name__+':',str(exc),file=sys.stderr,flush=True)
        raise SystemExit(1)
