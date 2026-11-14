#!/usr/bin/env python3
"""Phase 7 XCVI: immutable six-model PhIRE development-only seasonal16 inference.

Source: frozen XCV 16 native [u,v] fields; XC pilot used as numerical regression.
All 16 predictions are development-only. No TTK, GUDHI, HSDS or training.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, os, re, shutil, sys, tempfile
from pathlib import Path

DEFAULT_ROOT=Path(__file__).resolve().parents[2]
XCV_REL='ttk_runs_fixed/topology_finetuning/phase7_xcv_development_seasonal16_20261018'
XC_REL='ttk_runs_fixed/topology_finetuning/phase7_xc_development_inference_pilot_20261015'
XCI_REL='ttk_runs_fixed/topology_finetuning/phase7_xci_frozen_evaluation_protocol_20261015'
OUT_REL='ttk_runs_fixed/topology_finetuning/phase7_xcvi_dev_seasonal16_inference_20261018'
XC_SCRIPT='phase7_xc_update/scripts/phase7_dev_six_model_pilot.py'
XCV_SCRIPT='phase7_xcv_update/scripts/phase7_xcv_dev_seasonal16.py'
XC_SHA='27a0a97ce02e316a023253fdf97eed06e0139c3b4f7bff7b07367784a9a55368'
XCV_SHA='1864866025e19f957c65243623cb69ee445f35870038e66dd292606691d6ad1b'
SOURCE_FILES=('PhIREGANs.py','sr_network.py','utils.py')
EXPECTED_DEV=('gulf_inland','southwest_borderland')
MODEL_NAMES=('cnn','gan','grad','cv','pers','f1')
OLD_TO_NEW={0:0,1:4,2:8,3:12}
PRED_PILOT_MAX_ABS_TOL=1e-3  # Float32 inference can show hardware-dependent rounding
NUMERIC_REFERENCE_TOL=2e-5

def need(c,msg):
    if not c:raise RuntimeError(msg)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    need(spec is not None and spec.loader is not None,f'Cannot load {path}')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m

def checksums(folder,files=None):
    need((folder/'SHA256SUMS').is_file(),f'Missing SHA256SUMS: {folder}')
    rows=(folder/'SHA256SUMS').read_text().splitlines()
    entries={}
    for row in rows:
        m=re.fullmatch(r'([0-9a-f]{64})  ([^\n]+)',row)
        need(m is not None,'Bad checksum entry')
        digest,rel=m.groups();path=(folder/rel).resolve()
        need(rel not in entries and path.is_relative_to(folder.resolve()) and path.is_file(),f'Unsafe or missing entry {rel}')
        need(sha(path)==digest,f'Changed file {rel}')
        entries[rel]=digest
    if files is not None:need(set(entries)==set(files),f'Unexpected SHA256 files {sorted(entries)}')
    return entries

def report_files():
    return ['reference_mr.npy','reference_gt.npy','development_16_paired.tfrecord',
            *(f'pred_{n}.npy' for n in MODEL_NAMES),'seasonal16_inference_manifest.json',
            'seasonal16_inference_report.md']

def sources(root):
    import numpy as np
    xcpath=root/XC_SCRIPT;xcvpath=root/XCV_SCRIPT
    need(xcpath.is_file() and xcvpath.is_file(),'XC/XCV source scripts unavailable')
    need(sha(xcpath)==XC_SHA and sha(xcvpath)==XCV_SHA,'Historical XC or XCV script changed; STOP')
    xc=load(xcpath,'xcvi_archived_xc');xcv=load(xcvpath,'xcvi_archived_xcv')
    xcplan=xc.sources(root);xc.verify(root,root/XC_REL,xcplan)
    fields,lookup,fp=xcv.sources(root);xcv.verify(root,root/XCV_REL,fields,lookup,fp)
    need(len(fields)==16 and len(xcplan['fields'])==4,'Unexpected development field counts')
    need(tuple(xc.MODELS)==MODEL_NAMES,'Six original checkpoints changed')
    need(tuple(xc.MU_SIG[0])==(0.7684,-0.4575) and tuple(xc.MU_SIG[1])==(5.02455,5.9017),'Normalization changed')
    need(tuple(x['region'] for x in fields)==('gulf_inland',)*8+('southwest_borderland',)*8,'Incorrect or locked region in XCV manifest')
    need(tuple(x['wtk_index'] for x in fields)==(27048,27132,29184,29268,31392,31476,33600,33684)*2,'Unexpected timestamps')
    need(len({(x['region'],x['wtk_index']) for x in fields})==16,'Duplicate field')
    need(all(x['offset_hours'] in (0,84) for x in fields),'Unexpected offsets')
    # The four XC source fields must be byte-identical to the copied XCV files.
    for old_idx,new_idx in OLD_TO_NEW.items():
        old=xcplan['fields'][old_idx];new=fields[new_idx]
        need(new['cached_from_LXXXVII'] and old['region']==new['region'] and old['wtk_index']==new['wtk_index'],'XC/XCV field correspondence changed')
        need(sha(root/XCV_REL/new['filename'])==sha(root/xc.SCREEN_REL/old['name']),f'XC/XCV native preview bytes differ: {new["filename"]}')
    checksums(root/XC_REL,xc.report_files() if hasattr(xc,'report_files') else ['reference_mr.npy','reference_gt.npy','development_4_paired.tfrecord',*(f'pred_{n}.npy' for n in MODEL_NAMES),'dev_inference_report.json','dev_inference_report.md'])
    checksums(root/XCV_REL,[x['filename'] for x in fields]+['seasonal16_manifest.json','seasonal16_report.md'])
    checksums(root/XCI_REL,['evaluation_protocol.json','evaluation_protocol.md'])
    lineage={'XC_report_sha256':sha(root/XC_REL/'dev_inference_report.json'),
             'XCV_manifest_sha256':sha(root/XCV_REL/'seasonal16_manifest.json'),
             'XCI_protocol_sha256':sha(root/XCI_REL/'evaluation_protocol.json'),
             'XC_script_sha256':sha(xcpath),'XCV_script_sha256':sha(xcvpath),
             'XCVI_script_sha256':sha(Path(__file__)),
             'checkpoint_sha256':xcplan['checkpoint_sha256'],
             'model_source_sha256':{p:sha(root/p) for p in SOURCE_FILES}}
    for p in SOURCE_FILES:
        need(lineage['model_source_sha256'][p]==xcplan['source_sha256'][p],f'Original PhIRE source changed: {p}')
    print('[SOURCES] PASS: XCV 16 data+metadata, original XC six-model inference, frozen XCI protocol, SHA256',flush=True)
    print('[DEV-SCOPE] PASS: 16 records from two development regions; four original XC fields pinned; no locked-test access',flush=True)
    return xc,fields,xcplan,lineage

def reference_arrays(root,fields):
    import numpy as np
    mr=[];hr=[]
    for x in fields:
        with np.load(root/XCV_REL/x['filename'],allow_pickle=False) as z:
            need(set(z.files)=={'hr_uv','mr_uv','grid_rect','wtk_index','timestamp_utc','region'},'Bad source NPZ schema')
            a=z['hr_uv'];b=z['mr_uv']
            need(a.shape==(500,500,2) and b.shape==(100,100,2) and a.dtype==np.float64 and b.dtype==np.float64,'Bad source arrays')
            need(np.isfinite(a).all() and np.isfinite(b).all(),'Nonfinite field')
            need(str(z['region'])==x['region'] and int(z['wtk_index'])==x['wtk_index'] and str(z['timestamp_utc'])==x['timestamp_utc'],'Metadata mismatch')
            need(np.max(np.abs(a.reshape(100,5,100,5,2).mean((1,3))-b))<1e-10,'MR pooling parity changed')
            hr.append(a);mr.append(b)
    return np.stack(mr),np.stack(hr)

def write_tfrecord(path,mr,hr):
    import tensorflow as tf
    def integer(v):return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(v)]))
    def blob(v):return tf.train.Feature(bytes_list=tf.train.BytesList(value=[v]))
    with tf.io.TFRecordWriter(str(path)) as writer:
        for i in range(len(mr)):
            values={'index':integer(i),'data_LR':blob(mr[i].tobytes(order='C')),
                    'data_HR':blob(hr[i].tobytes(order='C')),
                    'h_LR':integer(100),'w_LR':integer(100),'h_HR':integer(500),'w_HR':integer(500),'c':integer(2)}
            writer.write(tf.train.Example(features=tf.train.Features(feature=values)).SerializeToString())
    print('[TFRECORD] PASS: 16 ordered paired MR/HR records (native PhIRE schema)',flush=True)

def summary_lines(fields,results):
    lines=['# Phase 7 XCVI — Six-model 2010 development seasonal16 inference','',
      '**Descriptive engineering/seasonal pilot (16 fields, two per region-week), not a generalization benchmark.**','',
      'No locked external-test region predictions, hyperparameter tuning, or topology calculations.','',
      '| Model | 16-field UV RMSE (m/s) ↓ | 16-field speed MAE (m/s) ↓ | Max XC four-field prediction difference (m/s) |',
      '|---|---:|---:|---:|']
    for arm in MODEL_NAMES:
        r=results[arm]
        lines.append(f"| {arm} | {r['pilot_mean']['uv_rmse_mps']:.6f} | {r['pilot_mean']['speed_mae_mps']:.6f} | {r['max_old_prediction_diff']:.6g} |")
    lines+=['','| Region | Season start UTC | Fields | Mean wind-speed GT (m/s) |', '|---|---|---:|---:|']
    import numpy as np
    for i in range(0,16,2):
        x=fields[i];lines.append(f"| {x['region']} | {x['anchor_utc'][:10]} | 2 | {results['cnn']['region_week_gt_speed_means'][i//2]:.6f} |")
    lines+=['','Metrics are computed over full 500×500 fields; 2010 two-hour region-week snippets are not independent hourly replications.',
       'XC four-field checkpoint-output regression is enforced before publication, without modifying original XC outputs.',
       'The original PhIRE pretrained-model WTK geography/time membership remains unknown.']
    return '\n'.join(lines)+'\n'

def infer(root,out,xc,fields,xcplan,lineage):
    import numpy as np
    need(not out.exists(),'Completed XCVI output exists; refuse overwrite')
    mr,gt=reference_arrays(root,fields)
    old_mr=np.load(root/XC_REL/'reference_mr.npy',allow_pickle=False)
    old_gt=np.load(root/XC_REL/'reference_gt.npy',allow_pickle=False)
    for oi,ni in OLD_TO_NEW.items():
        need(np.array_equal(mr[ni],old_mr[oi]) and np.array_equal(gt[ni],old_gt[oi]),'XC/XCV baseline data mismatch')
    out.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='.'+out.name+'.staging-',dir=out.parent))
    try:
        np.save(stage/'reference_mr.npy',mr)
        np.save(stage/'reference_gt.npy',gt)
        write_tfrecord(stage/'development_16_paired.tfrecord',mr,gt)
        from PhIREGANs import PhIREGANs
        results={}
        for arm,prefix in xc.MODELS.items():
            temp=stage/('tmp_'+arm)
            net=PhIREGANs(data_type='wind',mu_sig=xc.MU_SIG)
            net.set_data_out_path(str(temp))
            print(f'[MODEL-START] {arm} 16 development records',flush=True)
            net.test_paired(r=[5],data_path=str(stage/'development_16_paired.tfrecord'),
                            model_path=str(root/prefix),batch_size=1,save_inputs=True)
            index=np.load(temp/'idx.npy',allow_pickle=False)
            model_mr=np.load(temp/'dataIN.npy',allow_pickle=False)
            model_gt=np.load(temp/'dataGT.npy',allow_pickle=False)
            pred=np.load(temp/'dataSR.npy',allow_pickle=False)
            need(np.array_equal(index,np.arange(16)),'TensorFlow shuffled field order')
            need(model_mr.shape==mr.shape and model_gt.shape==gt.shape and pred.shape==gt.shape,'Incorrect paired inference shape')
            need(np.isfinite(model_mr).all() and np.isfinite(model_gt).all() and np.isfinite(pred).all(),'Nonfinite paired inference')
            dx=float(np.max(np.abs(model_mr-mr)));dy=float(np.max(np.abs(model_gt-gt)))
            need(dx<=NUMERIC_REFERENCE_TOL and dy<=NUMERIC_REFERENCE_TOL,f'{arm} input or GT de-normalization parity failed {dx} {dy}')
            old=np.load(root/XC_REL/f'pred_{arm}.npy',allow_pickle=False)
            need(old.shape==(4,500,500,2),'Original XC checkpoint output invalid')
            drift=max(float(np.max(np.abs(pred[ni]-old[oi]))) for oi,ni in OLD_TO_NEW.items())
            need(drift<=PRED_PILOT_MAX_ABS_TOL,f'{arm}: XC four-field prediction regression mismatch max_abs={drift:.8g}; investigate TF GPU/environment parity')
            np.save(stage/f'pred_{arm}.npy',pred)
            per=[xc.metrics(pred[i],gt[i],np) for i in range(16)]
            rmeans=[float(np.linalg.norm(gt[i:i+2],axis=-1).mean()) for i in range(0,16,2)]
            results[arm]={'max_mr_error':dx,'max_gt_error':dy,'max_old_prediction_diff':drift,
                          'per_field':per,'pilot_mean':xc.metrics(pred,gt,np),'region_week_gt_speed_means':rmeans}
            shutil.rmtree(temp)
            print(f'[MODEL-PASS] {arm} SR full500 16/16; mr_diff={dx:.3g} gt_diff={dy:.3g} XC_drift={drift:.3g}',flush=True)
        report={'schema_version':1,'scope':'DEVELOPMENT_ONLY_SEASONAL16_INFERENCE',
                'fields':fields,'source_sha256':lineage,'results':results,
                'notes':['2010 engineering pilot only; two sampled hours per region-week, 8 region-weeks.',
                         'Never inspect locked test model scores for method choice.',
                         'Not a full seven-day or 2012 replication evaluation.',
                         'Predictions held fixed for later corrected topology evaluation.']}
        (stage/'seasonal16_inference_manifest.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
        (stage/'seasonal16_inference_report.md').write_text(summary_lines(fields,results))
        (stage/'SHA256SUMS').write_text(''.join(f'{sha(stage/n)}  {n}\n' for n in report_files()))
        need(not out.exists(),'Output collision before commit')
        stage.rename(out)
    except BaseException:
        shutil.rmtree(stage,ignore_errors=True)
        raise
    print('[INFER-GATE] PASS: 96 full500 development predictions, original XC four-field regression, 16 exact MR/GT pairs',flush=True)
    print('[OUTPUT]',out,flush=True)

def verify(root,out,xc,fields,xcplan,lineage):
    import numpy as np
    checksums(out,report_files())
    report=json.loads((out/'seasonal16_inference_manifest.json').read_text())
    need(report['schema_version']==1 and report['scope']=='DEVELOPMENT_ONLY_SEASONAL16_INFERENCE','Wrong output schema')
    need(report['fields']==fields and report['source_sha256']==lineage,'Source lineage diverged')
    mr,gt=reference_arrays(root,fields)
    need(np.array_equal(mr,np.load(out/'reference_mr.npy',allow_pickle=False)),'Reference MR mismatch')
    need(np.array_equal(gt,np.load(out/'reference_gt.npy',allow_pickle=False)),'Reference GT mismatch')
    need(set(report['results'])==set(MODEL_NAMES),'Wrong output model set')
    for arm in MODEL_NAMES:
        pred=np.load(out/f'pred_{arm}.npy',allow_pickle=False)
        need(pred.shape==(16,500,500,2) and np.isfinite(pred).all(),f'{arm} predictions invalid')
        res=report['results'][arm]
        def compare(a,b,what):
            need(set(a)==set(b) and all(abs(float(a[k])-float(b[k]))<=1e-9 for k in a),f'{arm} {what} metric mismatch')
        compare(xc.metrics(pred,gt,np),res['pilot_mean'],'aggregate')
        need(len(res['per_field'])==16,'Per-field coverage mismatch')
        for i in range(16):compare(xc.metrics(pred[i],gt[i],np),res['per_field'][i],f'field {i}')
        old=np.load(root/XC_REL/f'pred_{arm}.npy',allow_pickle=False)
        drift=max(float(np.max(np.abs(pred[ni]-old[oi]))) for oi,ni in OLD_TO_NEW.items())
        need(abs(drift-float(res['max_old_prediction_diff']))<=1e-9 and drift<=PRED_PILOT_MAX_ABS_TOL,'XC baseline prediction mismatch')
        calc=[float(np.linalg.norm(gt[i:i+2],axis=-1).mean()) for i in range(0,16,2)]
        need(len(res['region_week_gt_speed_means'])==8 and all(abs(a-b)<=1e-9 for a,b in zip(calc,res['region_week_gt_speed_means'])),'GT season means mismatch')
    need((out/'seasonal16_inference_report.md').read_text()==summary_lines(fields,report['results']),'Markdown report edited')
    print('[VERIFY-GATE] PASS: XCV 16 native fields, 96 recomputed metrics/predictions, original XC four-field regression, 11 SHA256 entries; no locked tests',flush=True)

def selftest():
    import numpy as np
    need([*OLD_TO_NEW.items()]==[(0,0),(1,4),(2,8),(3,12)],'XC mappings changed')
    u=np.zeros((16,500,500,2),np.float64);u[...,0]=3;u[...,1]=4
    x=u.copy();x[...,0]+=1
    m=float(np.sqrt(np.mean((x-u)**2)))
    need(abs(m-np.sqrt(0.5))<1e-12,'RMSE selftest failure')
    with tempfile.TemporaryDirectory() as tmp:
        p=Path(tmp);f=p/'one.bin';f.write_bytes(b'test')
        (p/'SHA256SUMS').write_text(f'{sha(f)}  one.bin\n');checksums(p,['one.bin'])
        f.write_bytes(b'corrupt')
        try:checksums(p,['one.bin'])
        except RuntimeError:pass
        else:raise RuntimeError('Checksum tampering undetected')
    need(len(report_files())==11,'Output file set changed')
    print('[SELFTEST] PASS: 16 field ordering, four-field mapping, metrics, integrity and overwrite design',flush=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--mode',required=True,choices=('selftest','preflight','infer','verify'))
    ap.add_argument('--repo-root',default=str(DEFAULT_ROOT))
    args=ap.parse_args()
    if args.mode=='selftest':selftest();return
    root=Path(args.repo_root).resolve()
    # Required for importing original PhIREGANs and its non-package dependencies.
    sys.path.insert(0,str(root));os.chdir(root)
    xc,fields,xcplan,lineage=sources(root)
    out=root/OUT_REL
    if args.mode=='preflight':
        need(not out.exists(),'Output exists; use verify mode')
        mr,gt=reference_arrays(root,fields)
        need(mr.shape==(16,100,100,2) and gt.shape==(16,500,500,2),'Bad reference shape')
        print('[PREFLIGHT] PASS: 16 exact XCV development fields, all six archived checkpoints and clean output path; no TF',flush=True)
    elif args.mode=='infer':infer(root,out,xc,fields,xcplan,lineage)
    else:
        need(out.is_dir(),f'Completed output missing: {out}')
        verify(root,out,xc,fields,xcplan,lineage)

if __name__=='__main__':
    try:main()
    except Exception as e:
        print('[XCVI] ERROR',type(e).__name__+':',e,file=sys.stderr)
        sys.exit(1)
