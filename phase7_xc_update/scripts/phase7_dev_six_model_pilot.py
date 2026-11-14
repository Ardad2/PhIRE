#!/usr/bin/env python3
"""Phase 7 Part XC: four DEVELOPMENT-only full-500 PhIRE paired-inference pilot.

Modes: preflight (no TF, no files), infer (isolated atomic output), verify (no TF).
Never reads locked-test previews, modifies checkpoints, or selects model weights.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

ROOT_DEFAULT = Path(__file__).resolve().parents[2]
SCREEN_REL = 'ttk_runs_fixed/topology_finetuning/phase7_lxxxvii_wind_screen_20261015'
FROZEN_REL = 'ttk_runs_fixed/topology_finetuning/phase7_lxxxix_locked_region_split_20261015'
OUTPUT_REL = 'ttk_runs_fixed/topology_finetuning/phase7_xc_development_inference_pilot_20261015'
FROZEN_SCRIPT_REL = 'phase7_lxxxix_update/phase7_lxxxix_update/scripts/phase7_freeze_geographic_split.py'
SOURCES = ('PhIREGANs.py','sr_network.py','utils.py')
MODELS = {
    'cnn': 'models/wind_mr-hr/trained_cnn/cnn',
    'gan': 'models/wind_mr-hr/trained_gan/gan',
    'grad': 'models_fixed/topology_finetuning/wind_finetune_candidateF_grad_pairzero_expanded2688/cnn/cnn',
    'cv': 'models_fixed/topology_finetuning/wind_finetune_candidateF_grad_CV_only_expanded2688/cnn/cnn',
    'pers': 'models_fixed/topology_finetuning/wind_finetune_candidateF_grad_PERS_only_expanded2688/cnn/cnn',
    'f1': 'models_fixed/topology_finetuning/wind_finetune_candidateF_grad_E2_low_expanded2688/cnn/cnn',
}
EXPECTED = ('preview_gulf_inland_20100201.npz','preview_gulf_inland_20100801.npz',
            'preview_southwest_borderland_20100201.npz','preview_southwest_borderland_20100801.npz')
MU_SIG = [[0.7684,-0.4575],[5.02455,5.9017]]


def need(cond,msg):
    if not cond: raise RuntimeError(msg)


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def sums(folder, expected):
    need((folder/'SHA256SUMS').is_file(),f'Missing manifest: {folder}')
    found={}
    for line in (folder/'SHA256SUMS').read_text().splitlines():
        d,name=line.split('  ',1)
        p=(folder/name).resolve()
        need(re.fullmatch('[0-9a-f]{64}',d) is not None,'Bad SHA256 entry')
        need(p.is_relative_to(folder.resolve()) and p.is_file() and sha(p)==d,f'Integrity check failed: {name}')
        need(name not in found,'Duplicate SHA entry')
        found[name]=d
    need(set(found)==set(expected),f'Unexpected manifest files: {sorted(found)}')
    return found


def sources(root):
    import numpy as np
    screen=root/SCREEN_REL; frozen=root/FROZEN_REL
    sums(frozen,('locked_region_split.json','locked_region_split.md'))
    screen_files=('wind_screen.md','screen_summary.json','sample_screen.csv',
       *(f'preview_{r}_{d}.npz' for r in ('gulf_inland','upper_midwest','pacific_inland','southwest_borderland') for d in ('20100201','20100801')))
    sums(screen,screen_files)
    verify_script=root/FROZEN_SCRIPT_REL
    need(verify_script.is_file(),'Missing frozen-split validator')
    run=subprocess.run([sys.executable,str(verify_script),'--mode','verify','--repo-root',str(root)],capture_output=True,text=True)
    need(run.returncode==0 and '[VERIFY-GATE] PASS' in run.stdout,
         'Frozen split verification failed: '+(run.stdout+run.stderr)[-2000:])
    print('[FROZEN-SPLIT] PASS: original LXXXIX manifest and source verifier',flush=True)
    config=json.loads((frozen/'locked_region_split.json').read_text())
    split=config['frozen_split']
    need(config['status']=='FROZEN_BEFORE_MODEL_SCORES','Unexpected split status')
    need(split['development_regions']==['gulf_inland','southwest_borderland'],'Development regions changed')
    need(split['locked_external_test_regions']==['upper_midwest','pacific_inland'],'Locked regions changed')
    need(tuple(split['dev_pilot_previews'])==EXPECTED,'Preview allowlist changed')
    manifests={'frozen_split':sha(frozen/'locked_region_split.json'),
               'screen_summary':sha(screen/'screen_summary.json'),
               'pilot_script':sha(Path(__file__))}
    for name in SOURCES:
        p=root/name;need(p.is_file(),f'Missing PhIRE source {name}')
        manifests[name]=sha(p)
    cp={}
    for key,prefix in MODELS.items():
        base=root/prefix
        index=Path(str(base)+'.index')
        shard=Path(str(base)+'.data-00000-of-00001')
        need(index.is_file() and shard.is_file(),f'Incomplete checkpoint {key}: {prefix}')
        cp[key]={'prefix':prefix,'index_sha256':sha(index),'data_sha256':sha(shard)}
        print(f'[CHECKPOINT] {key}: index and data present / SHA256 recorded',flush=True)
    fields=[]
    expected_region={
      'gulf_inland':tuple(config['region_metadata']['gulf_inland']['grid_rect']),
      'southwest_borderland':tuple(config['region_metadata']['southwest_borderland']['grid_rect'])}
    for ix,name in enumerate(EXPECTED):
        path=screen/name
        with np.load(path,allow_pickle=False) as z:
            need(set(z.files)=={'hr_uv','mr_uv','grid_rect','wtk_index','timestamp_utc','region'},'Unexpected preview schema')
            hr=z['hr_uv'];mr=z['mr_uv'];rect=z['grid_rect']
            region=str(z['region']);dt=str(z['timestamp_utc']);wtkidx=int(z['wtk_index'])
            need(hr.shape==(500,500,2) and mr.shape==(100,100,2),f'Invalid shapes {name}')
            need(hr.dtype==np.float64 and mr.dtype==np.float64,'Previews must preserve float64')
            need(np.isfinite(hr).all() and np.isfinite(mr).all(),'Nonfinite native preview')
            need(tuple(rect)==expected_region[region],f'Grid rectangle mismatch {name}')
            need(name.startswith(f'preview_{region}_'),'Wrong region filename')
            need(dt.startswith('2010-02-01T') or dt.startswith('2010-08-01T'),'Date not authorized')
            # Compare cheap independent NumPy block average; TF parity previously verified in LXXXVIII.
            avg=hr.reshape(100,5,100,5,2).mean(axis=(1,3))
            need(float(np.max(np.abs(avg-mr)))<=1e-8,'MR downsample regression')
            fields.append({'index':ix,'name':name,'region':region,'wtk_index':wtkidx,
                           'timestamp_utc':dt,'grid_rect':list(map(int,rect)),'sha256':sha(path)})
    need(len(fields)==4,'Wrong pilot sample count')
    print('[DEV-SCOPE] PASS: four development-only 500x500/100x100 previews; no locked field loaded',flush=True)
    return {'source_sha256':manifests,'checkpoint_sha256':cp,'fields':fields,
            'normalization_mu_sig':MU_SIG,'model_prefixes':MODELS}


def reference_arrays(root, fields):
    import numpy as np
    mr=[];hr=[]
    for x in fields:
        with np.load(root/SCREEN_REL/x['name'],allow_pickle=False) as z:
            mr.append(z['mr_uv']);hr.append(z['hr_uv'])
    return np.stack(mr),np.stack(hr)


def write_tfrecord(path,mr,hr):
    # PhIRE _parse_train_ expects float64 tensor bytes and seven feature names.
    import tensorflow as tf
    def intf(v):return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(v)]))
    def bytef(v):return tf.train.Feature(bytes_list=tf.train.BytesList(value=[v]))
    with tf.io.TFRecordWriter(str(path)) as writer:
        for i in range(4):
            f={'index':intf(i),'data_LR':bytef(mr[i].tobytes(order='C')),
               'data_HR':bytef(hr[i].tobytes(order='C')),
               'h_LR':intf(100),'w_LR':intf(100),
               'h_HR':intf(500),'w_HR':intf(500),'c':intf(2)}
            ex=tf.train.Example(features=tf.train.Features(feature=f))
            writer.write(ex.SerializeToString())
    print('[TFRECORD] PASS: 4 ordered paired MR/HR records (PhIRE native schema)',flush=True)


def metrics(pred,gt,np):
    err=pred-gt
    speed_pred=np.linalg.norm(pred,axis=-1);speed_gt=np.linalg.norm(gt,axis=-1)
    dif=speed_pred-speed_gt
    return {'uv_rmse_mps':float(np.sqrt(np.mean(err**2))),
            'speed_mae_mps':float(np.mean(np.abs(dif))),
            'speed_rmse_mps':float(np.sqrt(np.mean(dif**2))),
            'speed_mean_pred_mps':float(speed_pred.mean()),
            'speed_mean_gt_mps':float(speed_gt.mean())}


def infer(root,out,plan):
    need(not out.exists(),f'Refusing overwrite: {out}')
    import numpy as np
    mr,gt=reference_arrays(root,plan['fields'])
    out.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='.'+out.name+'.staging-',dir=out.parent))
    try:
        np.save(stage/'reference_mr.npy',mr)
        np.save(stage/'reference_gt.npy',gt)
        write_tfrecord(stage/'development_4_paired.tfrecord',mr,gt)
        from PhIREGANs import PhIREGANs
        results={}
        for name,prefix in MODELS.items():
            temp=stage/('tmp_'+name)
            need(not temp.exists(),'Temporary output collision')
            net=PhIREGANs(data_type='wind',mu_sig=MU_SIG)
            net.set_data_out_path(str(temp))
            print(f'[MODEL-START] {name} (four development records)',flush=True)
            net.test_paired(r=[5],data_path=str(stage/'development_4_paired.tfrecord'),
                            model_path=str(root/prefix),batch_size=1,save_inputs=True)
            idx=np.load(temp/'idx.npy',allow_pickle=False)
            mr2=np.load(temp/'dataIN.npy',allow_pickle=False)
            gt2=np.load(temp/'dataGT.npy',allow_pickle=False)
            pred=np.load(temp/'dataSR.npy',allow_pickle=False)
            need(np.array_equal(idx,np.arange(4)),'TensorFlow changed sample order')
            need(mr2.shape==mr.shape and gt2.shape==gt.shape and pred.shape==gt.shape,
                 'Incorrect MR/GT/SR shapes')
            need(np.isfinite(pred).all() and np.isfinite(mr2).all() and np.isfinite(gt2).all(),
                 'Non-finite model arrays')
            err_mr=float(np.max(np.abs(mr2-mr)));err_gt=float(np.max(np.abs(gt2-gt)))
            need(err_mr<=2e-5 and err_gt<=2e-5,
                 f'{name} GT/MR orientation or de-normalization mismatch mr={err_mr:g} gt={err_gt:g}')
            for k in range(4):
                need(pred[k].shape==(500,500,2),'Model output not full-500')
            np.save(stage/f'pred_{name}.npy',pred)
            results[name]={'max_mr_error':err_mr,'max_gt_error':err_gt,
                           'per_field':[metrics(pred[i],gt[i],np) for i in range(4)],
                           'pilot_mean':metrics(pred,gt,np)}
            shutil.rmtree(temp)
            print(f'[MODEL-PASS] {name} SR full500 4/4; max_input_diff={err_mr:.3g} max_gt_diff={err_gt:.3g}',flush=True)
        report={'schema_version':1,'scope':'DEVELOPMENT_ONLY_FOUR_FIELD_PILOT',
            'sources':plan,'results':results,
            'notes':['This is a four-field development-only smoke test, not external generalization.',
                     'The pilot may inform protocol debugging; do not tune against locked geographic test.',
                     'No PD/MT or geographical calibration included; crop/topology protocol not frozen.',
                     'Original pretrained checkpoint source data membership is unverified.']}
        (stage/'dev_inference_report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
        lines=['# Phase 7 XC — four-field development-only six-model inference','',
               'Frozen before external-test access. No locked-region model scores.','',
               '| Model | UV RMSE (m/s) | Speed MAE (m/s) | GT/MR max discrepancy |',
               '|---|---:|---:|---:|']
        for n in MODELS:
            r=results[n]
            lines.append(f"| {n} | {r['pilot_mean']['uv_rmse_mps']:.5f} | {r['pilot_mean']['speed_mae_mps']:.5f} | {max(r['max_mr_error'],r['max_gt_error']):.3g} |")
        lines+=['','Development pilot only; descriptive quality checks, not model selection or a statistical conclusion.',
                 'No topology scores or external-test fields are present.']
        (stage/'dev_inference_report.md').write_text('\n'.join(lines)+'\n')
        files=['reference_mr.npy','reference_gt.npy','development_4_paired.tfrecord',
               *(f'pred_{n}.npy' for n in MODELS),'dev_inference_report.json','dev_inference_report.md']
        (stage/'SHA256SUMS').write_text(''.join(f'{sha(stage/n)}  {n}\n' for n in files))
        need(not out.exists(),'Unexpected output appeared')
        stage.rename(out)
    except BaseException:
        shutil.rmtree(stage,ignore_errors=True)
        raise
    print('[INFER-GATE] PASS: six original checkpoints, four development-only fields, GT/MR parity',flush=True)
    print('[OUTPUT]',out,flush=True)


def verify(root,out,plan):
    import numpy as np
    files=['reference_mr.npy','reference_gt.npy','development_4_paired.tfrecord',
           *(f'pred_{n}.npy' for n in MODELS),'dev_inference_report.json','dev_inference_report.md']
    sums(out,files)
    report=json.loads((out/'dev_inference_report.json').read_text())
    need(report['scope']=='DEVELOPMENT_ONLY_FOUR_FIELD_PILOT' and report['sources']==plan,'Source lineage mismatch')
    mr,gt=reference_arrays(root,plan['fields'])
    need(np.array_equal(np.load(out/'reference_mr.npy'),mr),'Reference MR tampered')
    need(np.array_equal(np.load(out/'reference_gt.npy'),gt),'Reference GT tampered')
    for name in MODELS:
        pred=np.load(out/f'pred_{name}.npy',allow_pickle=False)
        need(pred.shape==(4,500,500,2) and np.isfinite(pred).all(),'Prediction invalid')
        recorded=report['results'][name]['pilot_mean']
        observed=metrics(pred,gt,np)
        need(all(abs(observed[k]-recorded[k])<1e-9 for k in recorded),'Metric mismatch: '+name)
        for i in range(4):
            r=report['results'][name]['per_field'][i];q=metrics(pred[i],gt[i],np)
            need(all(abs(q[k]-r[k])<1e-9 for k in r),'Per-field metric mismatch: '+name)
    print('[VERIFY-GATE] PASS: source freeze, six full500 outputs, exact reference arrays, independently recomputed metrics and SHA256',flush=True)


def selftest():
    import numpy as np
    a=np.zeros((4,500,500,2),dtype=np.float64)
    a[:,:,:,0]=3.;a[:,:,:,1]=4.
    p=a.copy();p[:,:,:,0]+=1
    m=metrics(p,a,np)
    need(abs(m['uv_rmse_mps']-np.sqrt(.5))<1e-12,'UV RMSE regression')
    need(abs(m['speed_mae_mps']-(np.sqrt(32)-5))<1e-12,'Speed MAE regression')
    x=np.arange(100*5*100*5*2,dtype=np.float64).reshape(500,500,2)
    y=x.reshape(100,5,100,5,2).mean((1,3))
    need(y.shape==(100,100,2) and np.isfinite(y).all(),'Downsample selftest')
    print('[SELFTEST] PASS: metric formulas and array-layout validation (no TF, source or output files)',flush=True)


def main():
    a=argparse.ArgumentParser()
    a.add_argument('--mode',choices=('selftest','preflight','infer','verify'),required=True)
    a.add_argument('--repo-root',default=str(ROOT_DEFAULT))
    z=a.parse_args()
    if z.mode=='selftest':selftest();return
    root=Path(z.repo_root).resolve();out=root/OUTPUT_REL
    plan=sources(root)
    if z.mode=='preflight':
        need(not out.exists(),f'Output exists; use --mode verify: {out}')
        print('[PREFLIGHT] PASS: four permitted preview inputs, six complete checkpoints, frozen source SHA; no TF or output files',flush=True)
    elif z.mode=='infer':infer(root,out,plan)
    else:verify(root,out,plan)

if __name__=='__main__':
    try:main()
    except Exception as e:
        print('[XC] ERROR',type(e).__name__+':',e,file=sys.stderr)
        sys.exit(1)
