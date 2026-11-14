#!/usr/bin/env python3
"""Part CI: guarded 168-hour development-only six-checkpoint PhIRE inference.

A completed output is immutable; partial model runs are resumable after verified
source/checkpoint/old-XCVI regression checks. No external/test-region inference.
"""
from __future__ import annotations
import argparse, fcntl, hashlib, importlib.util, json, os, re, shutil, sys, tempfile, time
from pathlib import Path

REL = 'ttk_runs_fixed/topology_finetuning'
C_REL = REL+'/phase7_c_dev_gulf_inland_feb2010_week168_20261018'
XCVI_REL = REL+'/phase7_xcvi_dev_seasonal16_inference_20261018'
OUT_REL = REL+'/phase7_ci_dev_gulf_inland_feb2010_inference168_20261018'
C_SCRIPT = 'phase7_c_update/scripts/phase7_c_dev_fullweek_source.py'
XCVI_SCRIPT = 'phase7_xcvi_update/scripts/phase7_xcvi_dev_seasonal16_infer.py'
C_SHA = 'dfb073530be638a87d51d90916af1e5b9293cb837a4cde8663b4dc7c4778b54b'
XCVI_SHA = '8d0f6ca62cf4f91a4ddc07fb6be16a17489e2c6f2899adac2c7b4a561db0c0f7'
ARMS = ('cnn','gan','grad','cv','pers','f1')
ANCHORS = {0:0,84:1}  # week-hour -> XCVI seasonal16 index
N = 168
SCHEMA = 'phase7_ci_gulf_inland_feb2010_inference168_v1'
REF_TOL = 2e-5
PRED_TOL = 1e-3
MIN_DISK = 12 * 1024**3
MIN_MEM = 10 * 1024**3


def need(ok, msg):
    if not ok: raise RuntimeError(msg)

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def load(path, name):
    spec=importlib.util.spec_from_file_location(name,path)
    need(spec is not None and spec.loader is not None,f'Module missing {path}')
    m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m
    spec.loader.exec_module(m)
    return m

def atom(path,contents):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile('w',dir=path.parent,prefix='.tmp-ci-',delete=False) as f:
        tmp=Path(f.name)
        try:
            f.write(contents);f.flush();os.fsync(f.fileno())
        except BaseException:
            tmp.unlink(missing_ok=True);raise
    os.replace(tmp,path)

def jd(x): return json.dumps(x,sort_keys=True,indent=2,allow_nan=False)+'\n'

def manifest_sha(folder,expect):
    folder=Path(folder);ms=folder/'SHA256SUMS'
    need(ms.is_file(),f'Missing SHA256SUMS {ms}')
    got={}
    for line in ms.read_text().splitlines():
        m=re.fullmatch(r'([0-9a-f]{64})  ([^\n]+)',line)
        need(m is not None,f'Invalid checksum line in {ms}')
        digest,rel=m.groups();f=(folder/rel).resolve()
        need(rel not in got and f.is_relative_to(folder.resolve()) and f.is_file(),f'Unexpected checksum path {rel}')
        need(sha(f)==digest,f'Checksum failed {f}')
        got[rel]=digest
    need(set(got)==set(expect),f'Wrong SHA256 entry list for {folder}')
    return got

def output_names():
    return ['reference_mr.npy','reference_gt.npy','development_168_paired.tfrecord',
            *(f'pred_{x}.npy' for x in ARMS),'run_config.json','progress.json',
            'fullweek_inference_manifest.json','fullweek_inference_report.md']

def field_name(c,i): return 'fields/'+c.filename(i)

def sources(root):
    import numpy as np
    cp=root/C_SCRIPT;xp=root/XCVI_SCRIPT
    need(cp.is_file() and xp.is_file(),'Missing archived C or XCVI source scripts')
    need(sha(cp)==C_SHA and sha(xp)==XCVI_SHA,'Archived source script SHA changed; refuse inference')
    c=load(cp,'ci_archived_part_c');xcvi=load(xp,'ci_archived_xcvi')
    fp,cached=c.frozen(root)
    c.verify(root,fp,cached)
    xc,fields,xcplan,xcvi_lineage=xcvi.sources(root)
    xcvi.verify(root,root/XCVI_REL,xc,fields,xcplan,xcvi_lineage)
    need(tuple(xc.MODELS)==ARMS,'Frozen model set or order changed')
    need(tuple(tuple(x) for x in xc.MU_SIG)==((0.7684,-0.4575),(5.02455,5.9017)),'Normalization changed')
    need(len(fields)==16 and fields[0]['wtk_index']==27048 and fields[1]['wtk_index']==27132,'XCVI anchors invalid')
    need((root/C_REL).is_dir(),'Missing complete source week')
    cm=json.loads((root/C_REL/'fullweek_manifest.json').read_text())
    need(cm['field_count']==N and cm['scope']==c.SCHEMA and cm['source_sha256']==fp,'Fullweek source manifest changed')
    need(len(cm['fields'])==N,'Incorrect source week field count')
    for i,r in enumerate(cm['fields']):
        need(r['hour']==i and r['wtk_index']==27048+i and r['region']=='gulf_inland' and
             r['timestamp_utc']==c.timestamp(i) and r['filename']==c.filename(i),'Incorrect source hour metadata')
    need(sha(root/C_REL/field_name(c,0))==sha(root/'ttk_runs_fixed/topology_finetuning/phase7_xcv_development_seasonal16_20261018'/fields[0]['filename']),'First XCV source anchor drift')
    need(sha(root/C_REL/field_name(c,84))==sha(root/'ttk_runs_fixed/topology_finetuning/phase7_xcv_development_seasonal16_20261018'/fields[1]['filename']),'Second XCV source anchor drift')
    lineage={
        'c_run_config_sha256':sha(root/C_REL/'run_config.json'),
        'c_manifest_sha256':sha(root/C_REL/'fullweek_manifest.json'),
        'c_sha256s_sha256':sha(root/C_REL/'SHA256SUMS'),
        'xcvi_manifest_sha256':sha(root/XCVI_REL/'seasonal16_inference_manifest.json'),
        'xcvi_sha256s_sha256':sha(root/XCVI_REL/'SHA256SUMS'),
        'c_script_sha256':sha(cp),'xcvi_script_sha256':sha(xp),
        'ci_script_sha256':sha(Path(__file__).resolve()),
        'checkpoint_sha256':xcplan['checkpoint_sha256'],
        'phire_source_sha256':xcplan['source_sha256'],
        'xci_protocol_sha256':sha(root/xcvi.XCI_REL/'evaluation_protocol.json'),
    }
    xci=json.loads((root/xcvi.XCI_REL/'evaluation_protocol.json').read_text())
    proto=xci['protocol']
    need(tuple(proto['geographic_roles']['development'])==('gulf_inland','southwest_borderland') and
         tuple(proto['geographic_roles']['locked_external_test'])==('upper_midwest','pacific_inland'), 'Geographic roles changed')
    need('2010-02-01T00:00:00' in proto['times']['primary_week_start_utc'] and proto['times']['hourly_fields_per_week']==N,'Frozen time window changed')
    print('[SOURCES] PASS: 168 SHA-verified native WTK fields, C/XCVI/XCI lineage and six immutable checkpoints',flush=True)
    print('[SCOPE] PASS: development gulf_inland 2010-02-01..07 UTC only; zero locked-region predictions',flush=True)
    return c,xcvi,xc,cm,lineage

def reference_arrays(root,c):
    import numpy as np
    mr=np.empty((N,100,100,2),np.float64)
    gt=np.empty((N,500,500,2),np.float64)
    for i in range(N):
        with np.load(root/C_REL/field_name(c,i),allow_pickle=False) as z:
            need(set(z.files)==c.DATA_KEYS,'Unexpected HR/MR input NPZ schema')
            a=z['hr_uv'];b=z['mr_uv']
            need(a.shape==(500,500,2) and b.shape==(100,100,2) and a.dtype==b.dtype==np.float64,'Input data shape/dtype mismatch')
            need(int(z['wtk_index'])==27048+i and str(z['region'])=='gulf_inland' and str(z['timestamp_utc'])==c.timestamp(i),f'Hour {i} identity mismatch')
            need(np.isfinite(a).all() and np.isfinite(b).all(),'Nonfinite input field')
            need(float(np.max(np.abs(a.reshape(100,5,100,5,2).mean((1,3))-b)))<=1e-10,'Pooling parity failed')
            gt[i]=a;mr[i]=b
    return mr,gt

def budget(root):
    usage=shutil.disk_usage(root)
    need(usage.free>=MIN_DISK,f'Insufficient disk for full-week inference; free={usage.free/1024**3:.2f}GiB needs 12GiB')
    avail=None
    mem=Path('/proc/meminfo')
    if mem.exists():
        m=re.search(r'^MemAvailable:\s+(\d+) kB',mem.read_text(),re.M)
        if m:avail=int(m.group(1))*1024
    if avail is not None:need(avail>=MIN_MEM,f'MemAvailable too low for full-week inference: {avail/1024**3:.2f}GiB')
    print(f'[BUDGET] free_disk={usage.free/1024**3:.2f}GiB; MemAvailable={(avail/1024**3 if avail is not None else float("nan")):.2f}GiB; one PhIRE checkpoint at a time',flush=True)
    print('[BUDGET] Raw reference GT+MR=666.50MiB, six model predictions=3.76GiB, TFRecord approx 666MiB; peak network/GPU/temporary memory NOT predicted',flush=True)

def config(lineage):
    return {'schema_version':1,'scope':SCHEMA,'source_sha256':lineage,'region':'gulf_inland',
            'week_start_utc':'2010-02-01T00:00:00','hours':N,'wtk_first':27048,'wtk_last':27215,
            'model_names':list(ARMS),'r':[5],'mr_shape':[N,100,100,2],
            'gt_shape':[N,500,500,2],'xcvi_hour_regression_mapping':{'0':0,'84':1},
            'old_prediction_tolerance_mps':PRED_TOL}

def validate_anchors(root,out,arm,np):
    old=np.load(root/XCVI_REL/f'pred_{arm}.npy',allow_pickle=False,mmap_mode='r')
    pred=np.load(out/f'pred_{arm}.npy',allow_pickle=False,mmap_mode='r')
    need(old.shape==(16,500,500,2) and pred.shape==(N,500,500,2),'Prediction shape drift')
    d=max(float(np.max(np.abs(pred[i]-old[j]))) for i,j in ANCHORS.items())
    need(d<=PRED_TOL,f'{arm} original XCVI output mismatch: max diff={d} > {PRED_TOL}')
    return d

def metric_summary(xc,pred,gt,np):
    per=[xc.metrics(pred[i],gt[i],np) for i in range(N)]
    # Week-level mean of 168 paired per-hour metrics, not 168 independent experimental blocks.
    means={k:float(np.mean([r[k] for r in per])) for k in per[0]}
    aggregate=xc.metrics(pred,gt,np)
    return {'per_hour':per,'mean_hourly_metrics':means,'pooled_metrics':aggregate}

def validate_record(root,stage,arm,entry,xc,gt,np,full=False):
    need(set(entry)=={'sha256','anchor_max_abs','mean_hourly_metrics','pooled_metrics','per_hour'},'Malformed progress item')
    predpath=stage/f'pred_{arm}.npy'
    need(predpath.is_file() and sha(predpath)==entry['sha256'],f'Prediction mutated or missing: {arm}')
    pred=np.load(predpath,mmap_mode='r',allow_pickle=False)
    need(pred.shape==(N,500,500,2) and pred.dtype.kind=='f','Bad prediction shape/dtype')
    need(bool(np.isfinite(pred).all()),'Nonfinite prediction')
    drift=validate_anchors(root,stage,arm,np)
    need(abs(drift-entry['anchor_max_abs'])<=1e-10,'Anchor drift changed after checkpoint')
    if full:
        metrics=metric_summary(xc,pred,gt,np)
        for key in ('pooled_metrics','mean_hourly_metrics'):
            need(set(metrics[key])==set(entry[key]),'Metric keys mutated')
            for k,v in metrics[key].items():need(abs(v-entry[key][k])<=1e-9,f'Metric changed: {arm}.{key}.{k}')
        need(len(entry['per_hour'])==N,'Missing hour-level metrics')
        for i,(a,b) in enumerate(zip(metrics['per_hour'],entry['per_hour'])):
            need(set(a)==set(b) and all(abs(v-b[k])<=1e-9 for k,v in a.items()),f'Hour metric changed {arm} hour{i}')
    return pred

def report_text(progress):
    lines=['# Phase 7 Part CI — full-week, six-model development inference','',
           'Frozen Gulf inland, 2010 February 1–7 UTC, exactly 168 continuous hours.',
           '**One development region-week; descriptive engineering and within-week results, not independent geographic generalization.**','',
           '| Model | Mean hourly UV RMSE (m/s) ↓ | Mean hourly speed MAE (m/s) ↓ | XCVI anchor max difference (m/s) |',
           '|---|---:|---:|---:|']
    for arm in ARMS:
        r=progress['models'][arm];m=r['mean_hourly_metrics']
        lines.append(f'| {arm} | {m["uv_rmse_mps"]:.6f} | {m["speed_mae_mps"]:.6f} | {r["anchor_max_abs"]:.6g} |')
    lines+=['','Mean hourly RMSE and pooled whole-week RMSE are different statistics; all hour-level values and pooled metrics are retained in JSON.',
            'The six original checkpoint prefixes, normalization, geographic split and XCI outcomes are unchanged.',
            'The original PhIRE pretraining geography/time membership remains unknown.',
            'No 160×160 TTK persistence diagrams or GUDHI/spatial-locality distances are computed in Part CI.',
            'No external locked regions, model selection, hyperparameter tuning, or training.']
    return '\n'.join(lines)+'\n'

def write_manifest(stage,progress,cm):
    m={'schema_version':1,'scope':SCHEMA,'run_config':progress['config'],
       'models':progress['models'],'source_field_count':N,
       'source_wtk_first':27048,'source_wtk_last':27215,
       'source_region':'gulf_inland','source_week':'2010-02-01T00:00:00',
       'source_c_manifest_sha256':progress['config']['source_sha256']['c_manifest_sha256'],
       'limitations':['development-only, one region-week','hours temporally correlated',
                      'no geographic generalization','no topology computed','original pretrained WTK exposure unknown']}
    atom(stage/'fullweek_inference_manifest.json',jd(m))
    atom(stage/'fullweek_inference_report.md',report_text(progress))
    atom(stage/'SHA256SUMS',''.join(f'{sha(stage/n)}  {n}\n' for n in output_names()))

def read_progress(root,stage,ci_cfg,xc,gt,np,full=False):
    need(json.loads((stage/'run_config.json').read_text())==ci_cfg,'Run configuration changed, stop')
    p=json.loads((stage/'progress.json').read_text())
    need(set(p)=={'config','models','static_sha256'} and p['config']==ci_cfg,'Progress config/source/checkpoints changed')
    need(set(p['static_sha256'])=={'reference_mr.npy','reference_gt.npy','development_168_paired.tfrecord'},'Static source-hash inventory changed')
    for n,d in p['static_sha256'].items():
        need(re.fullmatch(r'[0-9a-f]{64}',d) is not None and (stage/n).is_file() and sha(stage/n)==d,
             f'Staged immutable input {n} has changed; refusing resume')
    done=p['models']
    need(set(done)<=set(ARMS),'Unexpected model in progress')
    for arm in done:validate_record(root,stage,arm,done[arm],xc,gt,np,full=full)
    extra={q.name for q in stage.glob('pred_*.npy')}-{f'pred_{a}.npy' for a in done}
    need(not extra,f'Unrecorded prediction present: {extra}; inspect before resuming')
    need({q.name for q in stage.iterdir()}=={'run_config.json','progress.json','reference_mr.npy','reference_gt.npy',
                'development_168_paired.tfrecord',*(f'pred_{a}.npy' for a in done)},
        'Unexpected/partial staging file(s); inspect, do not delete automatically')
    return p

def make_stage(root,stage,c,ci_cfg,xcvi):
    import numpy as np
    mr,gt=reference_arrays(root,c)
    old_mr=np.load(root/XCVI_REL/'reference_mr.npy',allow_pickle=False,mmap_mode='r')
    old_gt=np.load(root/XCVI_REL/'reference_gt.npy',allow_pickle=False,mmap_mode='r')
    for i,j in ANCHORS.items():
        need(np.array_equal(mr[i],old_mr[j]) and np.array_equal(gt[i],old_gt[j]),f'XCVI anchor input mismatch hour{i}')
    with tempfile.TemporaryDirectory(prefix='.phase7_ci_setup-',dir=stage.parent) as t:
        tmp=Path(t)
        np.save(tmp/'reference_mr.npy',mr);np.save(tmp/'reference_gt.npy',gt)
        xcvi.write_tfrecord(tmp/'development_168_paired.tfrecord',mr,gt)
        atom(tmp/'run_config.json',jd(ci_cfg))
        atom(tmp/'progress.json',jd({'config':ci_cfg,'models':{},
            'static_sha256':{n:sha(tmp/n) for n in
                ('reference_mr.npy','reference_gt.npy','development_168_paired.tfrecord')}}))
        need(not stage.exists(),'Stage appeared while initializing')
        tmp.rename(stage)
    print('[TFRECORD] PASS: 168 ordered MR/GT pairs; XCVI hours 0/84 bitwise inputs',flush=True)

def infer(root,out,stage,c,xc,xcvi,cm,lineage,max_new):
    import numpy as np
    need(not out.exists(),'Completed immutable CI output exists; use verify')
    budget(root)
    out.parent.mkdir(parents=True,exist_ok=True)
    lock=out.parent/('.'+out.name+'.lock')
    ci_cfg=config(lineage)
    with lock.open('a+') as f:
        fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        need(not out.exists(),'Completed CI output appeared during lock')
        if not stage.exists():make_stage(root,stage,c,ci_cfg,xcvi)
        mr,gt=reference_arrays(root,c)
        need(np.array_equal(np.load(stage/'reference_mr.npy',mmap_mode='r',allow_pickle=False),mr),'Stored MR reference differs from Part C')
        need(np.array_equal(np.load(stage/'reference_gt.npy',mmap_mode='r',allow_pickle=False),gt),'Stored GT reference differs from Part C')
        p=read_progress(root,stage,ci_cfg,xc,gt,np,full=True)
        print(f'[RESUME] PASS: {len(p["models"])}/{len(ARMS)} sealed model predictions',flush=True)
        todo=[arm for arm in ARMS if arm not in p['models']]
        from PhIREGANs import PhIREGANs
        t0=time.monotonic();fresh=0
        for arm in todo:
            if max_new is not None and fresh>=max_new:break
            prefix=xc.MODELS[arm]
            print(f'[MODEL-START] {arm}: {N} development-only records',flush=True)
            with tempfile.TemporaryDirectory(prefix=f'.ci_tmp_{arm}_',dir=stage) as td:
                temp=Path(td)/'network_output'
                net=PhIREGANs(data_type='wind',mu_sig=xc.MU_SIG)
                net.set_data_out_path(str(temp))
                net.test_paired(r=[5],data_path=str(stage/'development_168_paired.tfrecord'),
                                model_path=str(root/prefix),batch_size=1,save_inputs=True)
                ids=np.load(temp/'idx.npy',allow_pickle=False)
                observed_mr=np.load(temp/'dataIN.npy',allow_pickle=False)
                observed_gt=np.load(temp/'dataGT.npy',allow_pickle=False)
                prediction=np.load(temp/'dataSR.npy',allow_pickle=False)
                need(np.array_equal(ids,np.arange(N)),'TensorFlow changed record order')
                need(observed_mr.shape==mr.shape and observed_gt.shape==gt.shape and prediction.shape==gt.shape,'Incorrect TF paired input/output shapes')
                need(np.isfinite(prediction).all() and np.isfinite(observed_mr).all() and np.isfinite(observed_gt).all(),'Nonfinite paired inference')
                dx=float(np.max(np.abs(observed_mr-mr)));dy=float(np.max(np.abs(observed_gt-gt)))
                need(dx<=REF_TOL and dy<=REF_TOL,f'Input/GT de-normalization changed: {arm}, MR={dx} GT={dy}')
                anchor=max(float(np.max(np.abs(prediction[i]-np.load(root/XCVI_REL/f'pred_{arm}.npy',mmap_mode='r',allow_pickle=False)[j]))) for i,j in ANCHORS.items())
                need(anchor<=PRED_TOL,f'XCVI prediction regression failed for {arm}: {anchor}')
                entry={**metric_summary(xc,prediction,gt,np),'anchor_max_abs':anchor}
                # Prevent half-written final prediction using atomic rename of complete .npy.
                predpath=stage/f'pred_{arm}.npy'
                need(not predpath.exists(),f'Prediction collision: {predpath}')
                scratch=stage/f'.pred_{arm}.partial.npy'
                try:
                    with scratch.open('wb') as fh:
                        np.save(fh,prediction);fh.flush();os.fsync(fh.fileno())
                    os.replace(scratch,predpath)
                finally:scratch.unlink(missing_ok=True)
            entry['sha256']=sha(predpath)
            p['models'][arm]=entry
            atom(stage/'progress.json',jd(p))
            print(f'[MODEL-PASS] {arm}: 168/168; MR={dx:.3g} GT={dy:.3g} XCVI_anchor={anchor:.3g} elapsed_s={time.monotonic()-t0:.1f}',flush=True)
            fresh+=1
        p=read_progress(root,stage,ci_cfg,xc,gt,np,full=True)
        if len(p['models'])<len(ARMS):
            print(f'[PARTIAL-GATE] PASS: {len(p["models"])}/6 model predictions verified; resume with --mode infer',flush=True)
            return
        write_manifest(stage,p,cm)
        manifest_sha(stage,output_names())
        need(not out.exists(),'Completed output collision during publish')
        stage.rename(out)
        print('[INFER-GATE] PASS: all 6 models x 168 hours = 1008 predictions; original XCVI hour 0/84 regression; immutable output',flush=True)
        print('[OUTPUT]',out,flush=True)

def verify(root,out,stage,c,xc,cm,lineage):
    import numpy as np
    need(out.is_dir() and not stage.exists(),'Completed output missing or partial stage remains')
    manifest_sha(out,output_names())
    cfg=config(lineage)
    need(json.loads((out/'run_config.json').read_text())==cfg,'Output config changed')
    p=json.loads((out/'progress.json').read_text())
    need(p['config']==cfg and set(p['models'])==set(ARMS),'Output progress changed')
    need(set(p['static_sha256'])=={'reference_mr.npy','reference_gt.npy','development_168_paired.tfrecord'},'Input-hash coverage incomplete')
    for n,d in p['static_sha256'].items():
        need(re.fullmatch(r'[0-9a-f]{64}',d) is not None and sha(out/n)==d,
             f'Sealed input {n} differs from its initial staged hash')
    m=json.loads((out/'fullweek_inference_manifest.json').read_text())
    need(m['scope']==SCHEMA and m['schema_version']==1 and m['run_config']==cfg and m['models']==p['models'],'Output manifest mismatch')
    need(m['source_field_count']==N and m['source_wtk_first']==27048 and m['source_wtk_last']==27215 and
         m['source_region']=='gulf_inland' and m['source_c_manifest_sha256']==lineage['c_manifest_sha256'],'Output region/week source scope changed')
    mr,gt=reference_arrays(root,c)
    need(np.array_equal(np.load(out/'reference_mr.npy',allow_pickle=False,mmap_mode='r'),mr),'Stored MR differs from source fields')
    need(np.array_equal(np.load(out/'reference_gt.npy',allow_pickle=False,mmap_mode='r'),gt),'Stored GT differs from source fields')
    for arm in ARMS:validate_record(root,out,arm,p['models'][arm],xc,gt,np,full=True)
    need((out/'fullweek_inference_report.md').read_text()==report_text(p),'Markdown report changed')
    need({x.name for x in out.iterdir()}==set(output_names())|{'SHA256SUMS'},'Unexpected output files')
    print('[VERIFY-GATE] PASS: full C/XCVI/XCI source and six checkpoint hashes, 1008 full500 predictions, 168 paired metrics per model, 12 XCVI anchor predictions, 13 SHA256 entries; no external tests',flush=True)

def selftest():
    import numpy as np
    need(list(ANCHORS.items())==[(0,0),(84,1)],'Old hours mapping changed')
    need(len(ARMS)==6 and len(output_names())==13,'Manifest output count changed')
    v=np.zeros((2,500,500,2),np.float64);v[...,0]=3;v[...,1]=4
    need(abs(float(np.linalg.norm(v,axis=-1).mean())-5.)<1e-12,'Speed metric sanity failed')
    with tempfile.TemporaryDirectory() as td:
        d=Path(td);(d/'a').write_bytes(b'valid')
        atom(d/'SHA256SUMS',f'{sha(d/"a")}  a\n');manifest_sha(d,['a'])
        (d/'a').write_bytes(b'changed')
        try:manifest_sha(d,['a'])
        except RuntimeError:pass
        else:raise RuntimeError('Checksum tamper undetected')
    print('[SELFTEST] PASS: frozen anchors, six-model inventory, metric sanity, tamper rejection and atomic stage bookkeeping',flush=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--mode',required=True,choices=('selftest','preflight','infer','verify'))
    ap.add_argument('--repo-root',default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument('--max-new-models',type=int,default=None)
    a=ap.parse_args()
    if a.mode=='selftest':selftest();return
    need(a.max_new_models is None or (a.mode=='infer' and 1<=a.max_new_models<=6),'--max-new-models only for inference, 1..6')
    root=Path(a.repo_root).expanduser().resolve();sys.path.insert(0,str(root));os.chdir(root)
    c,xcvi,xc,cm,lineage=sources(root)
    out=root/OUT_REL;stage=out.parent/('.'+out.name+'.staging')
    if a.mode=='preflight':
        need(not out.exists(),'CI output already exists; use --mode verify')
        budget(root)
        if stage.exists():
            import numpy as np
            gt=reference_arrays(root,c)[1]
            p=read_progress(root,stage,config(lineage),xc,gt,np,full=True)
            print(f'[PREFLIGHT] PASS: validated stage {len(p["models"])}/6 completed; no TF, no writes',flush=True)
        else:
            need(not stage.exists(),'Unexpected stage directory')
            print('[PREFLIGHT] PASS: source, two anchor fields, checkpoints, output path clean; no TF, inference or writes',flush=True)
    elif a.mode=='infer': infer(root,out,stage,c,xc,xcvi,cm,lineage,a.max_new_models)
    else:verify(root,out,stage,c,xc,cm,lineage)

if __name__=='__main__':
    try:main()
    except Exception as e:
        print('[CI] ERROR',type(e).__name__+':',str(e),file=sys.stderr,flush=True)
        raise SystemExit(1)
