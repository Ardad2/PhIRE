#!/usr/bin/env python3
"""Phase 8 training-scale bridge: 168 separate WTK training records, three matched arms.

DEVELOPMENT EXPERIMENT ONLY. Does not load the original 168 benchmark or any
Phase-7 regional data. NOT a batch-size-4 / 2688-record manuscript-scale run.

Native TensorFlow Python modes: plan, export, fixed-cache, train, verify.
GUDHI/VTK micromamba modes: prepare, match-worker.
Any model weights saved are derived NEW artifacts under OUT. No source inputs,
original checkpoints, previous benchmark outputs or locked regions are written.

All three arms: identical pretrained generator, learning rate 1e-6, normalized
L_uv+0.05*L_grad, batch size 1, same deterministic 168-record order, 3 epochs.
  grad: no PD term.
  fixed: lambda 1e-5 x squared W2,2 using the frozen GT-to-CNN diagram
         correspondence *and predicted vertex indices* computed at initialization.
  dynamic: same lambda, new TTK critical vertices and GUDHI assignments before
           EVERY optimizer step, held fixed for that step only.

This is a controlled method-development comparison, not final manuscript proof.

Orders:
  python script --mode selftest
  python script --mode preflight
  python script --mode plan
  python script --mode export
  micromamba run -n gudhi-audit python script --mode prepare --ttk-backend docker
  python script --mode fixed-cache --max-new-samples 2 --ttk-backend docker
  python script --mode train --arm grad --max-new-steps 2
  python script --mode train --arm fixed --max-new-steps 2
  python script --mode train --arm dynamic --max-new-steps 2
  python script --mode train --arm grad|fixed|dynamic  (resume each to 504)
  python script --mode verify --arm all

Every segment is atomically published with full model AND Adam slot checkpoint,
strict SHA256 lineage and an immutable step ledger. An interruption during an
unfinished segment leaves a diagnostic staging directory and resumes from the
last committed segment; never edits sealed segments.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from collections import Counter

import numpy as np

SELF = Path(__file__).resolve()
PILOT = 'phase8_tf1_multisample_rematch_pilot_20261116.py'
ONE = 'phase8_tf1_one_step_optimizer_gate_20261116.py'
BRIDGE = 'phase8_tf1_dynamic_single_batch_gate_20261116_v2.py'
TEMPORAL = 'phase8_temporal_out_of_pilot_eval_20261117.py'
TFRECORD = 'example_data_topology_expanded_2688/wind_MR-HR.tfrecord'
CSV = 'example_data_topology_expanded_2688/manifest.csv'
BENCHMARK = 'example_data_fixed/wind_MR-HR.tfrecord'
EXISTING_BENCH_REPORT = ('ttk_runs_fixed/topology_finetuning/'
    'phase8_original168_eightstep_benchmark_20261118_r01/report/report.md')
OUT = 'ttk_runs_fixed/topology_finetuning/phase8_train168_threearm_20261118_r01'
SCHEMA='phase8-train168-threearm-freshmatch-r01'
SEASONS=('winter','spring','summer','fall')
ARMS=('grad','fixed','dynamic')
EPOCHS=3
N=168
STEPS=N*EPOCHS
CHUNK=24
LR=1e-6
LAM=1e-5
LGRAD=.05
DEFAULT_TTK_TIMEOUT=180
MAX_PAIRS=12000


def need(test,msg):
    if not test:raise RuntimeError(msg)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def jread(path):return json.loads(Path(path).read_text())


def jwrite(path,val):
    Path(path).write_text(json.dumps(val,indent=2,sort_keys=True,allow_nan=False)+'\n')


def mod(path,name):
    spec=importlib.util.spec_from_file_location(name,str(path))
    need(spec and spec.loader, f'Missing module: {path}')
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m
    spec.loader.exec_module(m)
    return m


def pick_indices():
    """42 samples/season: two 11-sample, two 10-sample weeks per season."""
    weeks=[]
    for w in range(16):
        n=11 if w%4 in (0,2) else 10
        hours=[int(round(x)) for x in np.linspace(8,159,n)]
        need(len(set(hours))==n, 'Selection degeneracy')
        weeks.append([w*168+h for h in hours])
    # Interleave the 16 windows: avoid training on an uninterrupted single week.
    return tuple(i for position in range(11) for group in weeks
                 if position<len(group) for i in [group[position]])

SELECT=pick_indices()
ORDER=SELECT*EPOCHS


def selftest():
    need(N==168 and STEPS==504 and CHUNK==24 and LR==1e-6 and LAM==1e-5 and LGRAD==.05,
         'Frozen config drift')
    need(len(SELECT)==168 and len(set(SELECT))==168 and all(0<=x<2688 for x in SELECT),
         'Wrong training selection')
    counts=Counter(x//672 for x in SELECT)
    need(counts=={0:42,1:42,2:42,3:42},f'Unbalanced seasonal counts {counts}')
    need(not set(SELECT).intersection({0,1,2,3}), 'Pilot training indices unexpectedly selected')
    need(len(ORDER)==504 and len(set(ORDER))==168, 'Training order drift')
    need('example_data_fixed' not in TFRECORD and 'phase7' not in TFRECORD.lower(),
         'Forbidden training path')
    need(not any(x in OUT for x in ('upper_midwest','pacific_inland')),'Locked path')
    print('[SELFTEST] PASS: 168 unique new training records, 42/season, 3 identical epochs,'
          ' 504 steps/arm; no files written',flush=True)


def inventory(root):
    path=root/CSV
    need(path.is_file(),f'Missing source provenance CSV: {path}')
    temporal=mod(root/TEMPORAL,'train168_immutable_temporal')
    with path.open(newline='') as f: rows=list(csv.DictReader(f))
    need(len(rows)==2688,'WTK manifest must have 2688 records')
    source=set()
    for i,row in enumerate(rows):
        rid=int(row['sample_idx']); wtk=int(row['wtk_time_index'])
        season,base=temporal.WINDOWS[i//168]
        need(rid==i and wtk==base+i%168 and row['season']==f'{season}_{i//168%4+1}',
             f'Unexpected WTK source lineage for i={i}')
        need(wtk>=168 and wtk not in source, f'Benchmark overlap or repeated WTK hour {wtk}')
        source.add(wtk)
    return rows


def deps(root):
    need(root==SELF.parent and (root/'sr_network.py').is_file(),
         'Install script at ~/PhIRE repository root')
    for rel in (PILOT,ONE,BRIDGE,TEMPORAL,TFRECORD,CSV,BENCHMARK):
        need((root/rel).is_file(),f'Missing dependency: {rel}')
    p=mod(root/PILOT,'train168_pilot')
    p.IDS=SELECT  # adapt ONLY sample inventory; matching math remains unchanged
    bridge,cp=p.check_root(root)
    return p,bridge,cp


def source_hashes(root,p,bridge):
    original=p.source_sha(root,bridge)
    original[str(SELF)]=sha(SELF)
    for rel in (PILOT,ONE,BRIDGE,TEMPORAL):
        original[str((root/rel).resolve())]=sha(root/rel)
    return original


def plan_doc(root,p,bridge,cp):
    rows=inventory(root)
    need((root/BENCHMARK).is_file(),'168 evaluation benchmark missing (not read during training)')
    need((root/EXISTING_BENCH_REPORT).is_file(),
         'Verified old 168 benchmark report missing; do not mix evaluation sources')
    return {'schema':SCHEMA+'-plan','training_indices':list(SELECT),
            'training_wtk':[{'sample_idx':i,'season':rows[i]['season'],
              'wtk_time_index':int(rows[i]['wtk_time_index']),
              'timestamp':rows[i]['timestamp']} for i in SELECT],
            'step_order':list(ORDER),'steps_per_arm':STEPS,'epochs':EPOCHS,
            'batch_size':1,'optimizer':'independent Adam, zero-initialized per arm',
            'lr':LR,'lambda_grad':LGRAD,'lambda_w22_squared':LAM,
            'fixed_policy':'pretrained CNN initial predicted PD pairs AND vertex indices frozen',
            'dynamic_policy':'new TTK/GUDHI predicted pairs before EVERY update',
            'checkpoint_hashes':bridge.filehashes(cp),
            'source_hashes':source_hashes(root,p,bridge),
            'training_tfrecord_sha256':sha(root/TFRECORD),
            'training_tfrecord_size':(root/TFRECORD).stat().st_size,
            'training_tfrecord_mtime_ns':(root/TFRECORD).stat().st_mtime_ns,
            'training_manifest_sha256':sha(root/CSV),
            'reference_benchmark_report_sha256':sha(root/EXISTING_BENCH_REPORT),
            'reference_benchmark_tfrecord_sha256':sha(root/BENCHMARK),
            'no_geographic_test_access':True,
            'note':'Train from expanded 2688 pool; original 168 benchmark remains evaluation-only.'}


def preflight(root,output,check_new=False):
    selftest()
    p,bridge,cp=deps(root)
    rows=inventory(root)
    need(len(rows)==2688,'Invalid metadata')
    if check_new:need(not (output/'plan').exists(),'Plan exists; do not overwrite')
    print('[PREFLIGHT] PASS: source 2688 WTK hours, 168 selected from 16 weeks '
          '(42 per season); original 168 benchmark NOT selected',flush=True)
    print('[SCOPE] ONLY Mid-Atlantic expanded training pool; no Phase7 data, '
          'no Upper Midwest/Pacific inland fields',flush=True)
    return p,bridge,cp


def make_plan(root,out):
    p,bridge,cp=preflight(root,out,True)
    obj=plan_doc(root,p,bridge,cp)
    p.atomic(out/'plan',lambda stage:jwrite(stage/'manifest.json',obj))
    print('[PLAN] PASS: immutable 168-field selection and 504-step schedule; '
          'TFRecord, original checkpoint, source and benchmark hashes pinned',flush=True)


def check_plan(root,out,full_data_hash=False):
    p,bridge,cp=deps(root)
    f=out/'plan'/'manifest.json'
    need(f.is_file(),'Missing immutable plan; run --mode plan')
    m=jread(f)
    need(m['schema']==SCHEMA+'-plan' and m['training_indices']==list(SELECT) and
         m['step_order']==list(ORDER) and m['steps_per_arm']==STEPS and
         m['epochs']==EPOCHS and m['batch_size']==1 and m['lr']==LR and
         m['lambda_grad']==LGRAD and m['lambda_w22_squared']==LAM and
         m['no_geographic_test_access'], 'Plan configuration drift')
    p.require_hashes(m['source_hashes'])
    need(bridge.filehashes(cp)==m['checkpoint_hashes'],'Pretrained checkpoint changed')
    need(sha(root/CSV)==m['training_manifest_sha256'],'Training metadata changed')
    need(sha(root/EXISTING_BENCH_REPORT)==m['reference_benchmark_report_sha256'],
         'Completed original168 benchmark report changed')
    need((root/TFRECORD).stat().st_size==m['training_tfrecord_size'] and
         (root/TFRECORD).stat().st_mtime_ns==m['training_tfrecord_mtime_ns'],
         'Training TFRecord size/mtime changed; stop and check provenance')
    if full_data_hash:
        need(sha(root/TFRECORD)==m['training_tfrecord_sha256'], 'Training TFRecord bytes changed')
        need(sha(root/BENCHMARK)==m['reference_benchmark_tfrecord_sha256'],
             'Original168 evaluation TFRecord bytes changed')
    return p,bridge,cp,m


def inputs_check(root,out,p):
    d=out/'inputs';need(d.is_dir(),'Run --mode export first')
    m=jread(d/'manifest.json')
    need(m['schema']==SCHEMA+'-inputs' and m['training_indices']==list(SELECT)
         and m['plan_sha256']==sha(out/'plan'/'manifest.json'),
         'Export identity changed')
    p.verify_files(d,m['artifact_hashes'])
    return m


def get_train(out,i):
    need(i in SELECT,'Record index outside frozen training selection')
    p=out/'inputs';t=f's{i:04d}'
    lr=np.load(p/f'{t}_lr.npy',allow_pickle=False)
    hr=np.load(p/f'{t}_hr.npy',allow_pickle=False)
    need(lr.shape==(1,100,100,2) and hr.shape==(1,500,500,2),f'Wrong training dimensions s{i}')
    need(np.isfinite(lr).all() and np.isfinite(hr).all(),'Nonfinite normalized training field')
    return lr,hr


def export(root,out):
    p,bridge,cp,m=check_plan(root,out,full_data_hash=True)
    need(not (out/'inputs').exists(),'Export exists; preserve immutable cache')
    tf,_,_,PhIREGANs=bridge.import_tf(root)
    tf.reset_default_graph()
    phire=PhIREGANs(data_type='phase8_train168_readonly',
                    mu_sig=[p.MU.tolist(),p.SIG.tolist()])
    ds=tf.data.TFRecordDataset(str(root/TFRECORD))
    ds=ds.map(lambda sample:phire._parse_train_(sample,phire.mu_sig))
    it=ds.make_one_shot_iterator().get_next()
    selected=set(SELECT)
    conf=tf.ConfigProto();conf.gpu_options.allow_growth=True
    def writer(stage):
        stored={};observed=0
        with tf.Session(config=conf) as sess:
            while True:
                try:idx,lr,hr=sess.run(it)
                except tf.errors.OutOfRangeError:break
                idx=int(idx);observed+=1
                need(0<=idx<2688,'Malformed expanded TFRecord index')
                if idx not in selected:continue
                need(idx not in stored,'Duplicate selected TFRecord sample')
                need(lr.shape==(100,100,2) and hr.shape==(500,500,2),
                     f'Incorrect training record dimensions {idx}')
                lr=lr[None].astype(np.float32);hr=hr[None].astype(np.float32)
                for suffix,arr in (('lr',lr),('hr',hr),('gt_crop',p.speed_from_normalized(hr))):
                    np.save(stage/f's{idx:04d}_{suffix}.npy',arr)
                stored[idx]=True
                if len(stored)%24==0:print(f'[EXPORT] {len(stored):03d}/168 selected records',flush=True)
        need(observed==2688 and set(stored)==selected,
             f'Expanded training TFRecord must have 2688 records, selected {len(stored)}')
        names=[f's{i:04d}_{suffix}.npy' for i in SELECT for suffix in ('lr','hr','gt_crop')]
        jwrite(stage/'manifest.json',{'schema':SCHEMA+'-inputs',
            'training_indices':list(SELECT),'tfrecord_total_records':observed,
            'plan_sha256':sha(out/'plan'/'manifest.json'),
            'artifact_hashes':p.hash_files(stage,names),
            'train_only':True,'no_optimizer':True})
    p.atomic(out/'inputs',writer)
    print('[EXPORT] PASS: 168 stratified training fields, 2688 source records validated; '
          'no optimizer or locked regions',flush=True)


def gt_check(root,out,p):
    d=out/'gt_cache';need(d.is_dir(),'Run --mode prepare first')
    m=jread(d/'manifest.json')
    need(m['schema']==SCHEMA+'-gt' and m['training_indices']==list(SELECT)
         and m['input_manifest_sha256']==sha(out/'inputs'/'manifest.json'),
         'GT cache lineage drift')
    p.verify_files(d,m['artifact_hashes'])
    return m


def prepare(root,out,backend,timeout):
    p,bridge,_,_=check_plan(root,out)
    inputs_check(root,out,p)
    need(not (out/'gt_cache').exists(),'GT cache sealed; do not overwrite')
    timer,pd,decoder,_,_=p.audit_modules(root,bridge)
    timer.check_backend(backend)
    def writer(stage):
        (stage/'vti').mkdir();(stage/'pd').mkdir()
        allfiles=[]
        for j,i in enumerate(SELECT,1):
            gt=np.load(out/'inputs'/f's{i:04d}_gt_crop.npy',allow_pickle=False)
            tag=f'gt_s{i:04d}'
            timer.write_vti(stage/'vti'/f'{tag}.vti',gt)
            timer.run_ttk(backend,stage,tag,timeout)
            features=pd.parse(decoder.read_pd,stage/'pd'/f'{tag}_pd_port_0.vtu')
            need(all(len(features[d])<=6000 for d in (0,1)), 'Unexpected GT PD size')
            allfiles.extend((f'vti/{tag}.vti',f'pd/{tag}_pd_port_0.vtu'))
            if j%24==0:print(f'[PREPARE] {j:03d}/168 GT speed diagrams',flush=True)
        jwrite(stage/'manifest.json',{'schema':SCHEMA+'-gt',
            'training_indices':list(SELECT),
            'input_manifest_sha256':sha(out/'inputs'/'manifest.json'),
            'artifact_hashes':p.hash_files(stage,allfiles),
            'radius_crop':160,'no_optimizer':True})
    p.atomic(out/'gt_cache',writer)
    print('[PREPARE] PASS: 168 immutable initial GT TTK diagrams',flush=True)


def check_one_sample(root,out,p,i):
    """Worker validates only current sample, avoiding 168x full input rehash/step."""
    m=jread(out/'inputs'/'manifest.json')
    g=jread(out/'gt_cache'/'manifest.json')
    need(m['plan_sha256']==sha(out/'plan'/'manifest.json') and
         g['input_manifest_sha256']==sha(out/'inputs'/'manifest.json'),
         'Worker plan/export/GT source link drift')
    files={f's{i:04d}_{suffix}.npy':m['artifact_hashes'][f's{i:04d}_{suffix}.npy']
           for suffix in ('lr','hr','gt_crop')}
    p.verify_files(out/'inputs',files)
    pd=f'pd/gt_s{i:04d}_pd_port_0.vtu'
    p.verify_files(out/'gt_cache',{pd:g['artifact_hashes'][pd]})
    return True


def match_worker(root,out,step_dir,idx,backend,timeout):
    p,bridge,cp,_=check_plan(root,out)
    need(idx in SELECT,'Matching forbidden sample')
    need(step_dir.resolve().is_relative_to(out.resolve()),'Worker stage outside output root')
    need((step_dir/'dynamic_before.npy').is_file(),'Missing before-prediction')
    # The original trusted matching routine is imported, not copied or altered.
    # Override only its full-input verifier by a per-sample verified read-only gate.
    p.cache_check=lambda r,o,b:check_one_sample(r,o,p,idx)
    p.match_worker(root,out,step_dir,idx,backend,timeout,MAX_PAIRS)
    res=jread(step_dir/'match.json')
    need(res['sample']==idx and res['pairs']>0,'Bad matching result')
    p.verify_files(step_dir,res['artifact_hashes'])
    print(f'[WORKER] PASS: selected sample={idx} original TTK/GUDHI fresh matching',flush=True)


def worker_call(root,out,stepdir,idx,backend,timeout):
    exe=os.environ.get('PHASE8_MICROMAMBA') or os.environ.get('MAMBA_EXE') or shutil.which('micromamba')
    need(exe and Path(exe).is_file(),'Set PHASE8_MICROMAMBA to executable micromamba')
    cmd=[exe,'run','-n','gudhi-audit','python',str(SELF),'--mode','match-worker',
         '--repo-root',str(root),'--out',str(out),'--step-dir',str(stepdir),
         '--index',str(idx),'--ttk-backend',backend,'--ttk-timeout',str(timeout)]
    env=dict(os.environ);env['PYTHONNOUSERSITE']='1'
    result=subprocess.run(cmd,cwd=str(root),env=env)
    need(result.returncode==0,f'GUDHI fresh matching failed for sample={idx}: exit={result.returncode}')


def cnn_graph(root):
    p,bridge,_=deps(root)
    one=mod(root/ONE,'train168_frozen_optimizer')
    tf,xl,xh,ph,fetch,trains,saver,init,vars_=one.make_tf_graph(bridge,root,LR,LAM)
    # Separate complete saver preserves model AND selected Adam slots between segments.
    full=tf.train.Saver(var_list=tf.global_variables(),max_to_keep=None)
    return p,bridge,tf,xl,xh,ph,fetch,trains,saver,init,vars_,full


def init_model_digest(sess,vars_):
    h=hashlib.sha256()
    for param in sess.run(vars_):h.update(np.asarray(param).tobytes())
    return h.hexdigest()


def fixed_pair_check(out,i,p):
    f=out/'fixed_pairs'/f's{i:04d}'
    need(f.is_dir(),f'Fixed pair for sample {i} missing; run --mode fixed-cache')
    m=jread(f/'fixed_manifest.json')
    need(m['schema']==SCHEMA+'-fixed-sample' and m['index']==i and
         m['plan_sha256']==sha(out/'plan'/'manifest.json'),
         f'Frozen-pair lineage changed {i}')
    p.verify_files(f,m['hashes'])
    need(m['match_manifest_sha256']==sha(f/'match.json'),
         f'Frozen match manifest changed s{i}')
    read_match=jread(f/'match.json')
    p.verify_files(f,read_match['artifact_hashes'])
    return f


def fixed_cache(root,out,backend,timeout,max_new):
    p,_,_,_=check_plan(root,out)
    inputs_check(root,out,p);gt_check(root,out,p)
    (out/'fixed_pairs').mkdir(exist_ok=True)
    p,bridge,tf,xl,xh,ph,fetch,trains,saver,init,vars_,full=cnn_graph(root)
    conf=tf.ConfigProto();conf.gpu_options.allow_growth=True
    completed=0;added=0
    with tf.Session(config=conf) as sess:
        sess.run(init);saver.restore(sess,str(root/p.CHECKPOINT))
        digest=init_model_digest(sess,vars_)
        for i in SELECT:
            f=out/'fixed_pairs'/f's{i:04d}'
            if f.exists():
                fixed_pair_check(out,i,p);completed+=1;continue
            if max_new is not None and added>=max_new:break
            lr,hr=get_train(out,i)
            pred=np.asarray(sess.run(fetch['crop'],feed_dict={xl:lr,xh:hr})[0],dtype=np.float32)
            need(pred.shape==(160,160) and np.isfinite(pred).all(),'Initial prediction invalid')
            def writer(stage):
                np.save(stage/'dynamic_before.npy',pred)
                worker_call(root,out,stage,i,backend,timeout)
                m=jread(stage/'match.json')
                jwrite(stage/'fixed_manifest.json',{
                    'schema':SCHEMA+'-fixed-sample','index':i,
                    'plan_sha256':sha(out/'plan'/'manifest.json'),
                    'initial_model_digest':digest,
                    'match_manifest_sha256':sha(stage/'match.json'),
                    'hashes':p.hash_files(stage,['dynamic_before.npy','feed.npz','match.json',
                       'vti/dynamic_before.vti','pd/dynamic_before_pd_port_0.vtu'])})
            p.atomic(f,writer)
            fixed_pair_check(out,i,p);completed+=1;added+=1
            if completed%16==0 or max_new is not None:
                print(f'[FIXED-CACHE] {completed:03d}/168 sealed initial pair feeds',flush=True)
    print(f'[FIXED-CACHE] PASS: cached={completed}/168, new={added}; initial pretrained weights '
          f'and vertex indices unchanged',flush=True)


def read_feed(path):
    with np.load(path,allow_pickle=False) as z:
        pairs={k:np.array(z[k]) for k in z.files}
    need(pairs['birth_yx'].shape[0]==1 and pairs['birth_yx'].shape[-1]==2 and
         np.isfinite(pairs['target']).all() and np.isfinite(pairs['gt_diagonal_cost']).all(),
         'Malformed PD optimizer feed')
    return pairs


def add_feed(ph,feed,pairs):
    out=dict(feed)
    for key,phitem in zip(('birth_yx','death_yx','target','mode','mask'),ph[:5]):
        out[phitem]=pairs[key]
    out[ph[5]]=float(pairs['gt_diagonal_cost'])
    return out


def segments(out,arm,p,strict=True):
    root=out/'training'/arm/'chunks'
    if not root.is_dir():return [],0,None
    dirs=sorted([x for x in root.iterdir() if x.is_dir() and x.name.startswith('seg_')])
    entries=[];next_step=0;prev_hash=None
    plan_sha=sha(out/'plan'/'manifest.json')
    for d in dirs:
        m=jread(d/'manifest.json')
        need(m['schema']==SCHEMA+'-chunk' and m['arm']==arm and
             m['start']==next_step and m['end']<=STEPS and
             m['end']>m['start'] and
             d.name==f'seg_{m["start"]:04d}_{m["end"]:04d}',
             f'Noncontiguous or misnamed train chunk: {d}')
        need(m['previous_manifest_sha256']==prev_hash and
             m['plan_sha256']==plan_sha,'Segment chain tampered')
        need(len(m['steps'])==m['end']-m['start'] and
             [x['index'] for x in m['steps']]==list(ORDER[m['start']:m['end']]),
             'Optimizer step schedule altered')
        p.verify_files(d,m['hashes'])
        prev_hash=sha(d/'manifest.json');next_step=m['end'];entries.append(d)
    if strict:
        # Don't silently interpret forensic partial stages as a complete snapshot.
        for path in root.iterdir():
            if path.is_dir() and path.name.startswith('.seg_'):
                print(f'[NOTICE] Incomplete forensic stage preserved: {path}',flush=True)
    return entries,next_step,prev_hash


def chunk_inputs(out,i):
    lr,hr=get_train(out,i)
    crop=np.load(out/'inputs'/f's{i:04d}_gt_crop.npy',allow_pickle=False)
    need(crop.shape==(160,160) and np.isfinite(crop).all(),'GT crop corrupt')
    return lr,hr,crop


def train(root,out,arm,backend,timeout,max_new):
    need(arm in ARMS,'Select one of grad/fixed/dynamic')
    p,bridge,cp,plan=check_plan(root,out)
    inputs_check(root,out,p);gt_check(root,out,p)
    (out/'training'/arm/'chunks').mkdir(parents=True,exist_ok=True)
    previous,next_step,prevhash=segments(out,arm,p)
    if next_step==STEPS:
        print(f'[TRAIN] PASS: arm={arm} already has {STEPS}/{STEPS} sealed optimizer updates',flush=True)
        return
    if arm=='fixed':
        # Fixed feed must be from initialization. Verify at least the next step.
        fixed_pair_check(out,ORDER[next_step],p)
    p,bridge,tf,xl,xh,ph,fetch,trains,saver,init,vars_,full=cnn_graph(root)
    conf=tf.ConfigProto();conf.gpu_options.allow_growth=True
    with tf.Session(config=conf) as sess:
        sess.run(init)
        if previous:
            full.restore(sess,str(previous[-1]/'tf_state'))
            initial_hash=jread(previous[0]/'manifest.json')['initial_model_digest']
        else:
            saver.restore(sess,str(root/p.CHECKPOINT))
            initial_hash=init_model_digest(sess,vars_)
        completed=0
        while next_step<STEPS:
            if max_new is not None and completed>=max_new:break
            remaining=STEPS-next_step
            this_many=min(CHUNK,remaining)
            if max_new is not None:this_many=min(this_many,max_new-completed)
            start=next_step;stop=start+this_many
            dest=out/'training'/arm/'chunks'/f'seg_{start:04d}_{stop:04d}'
            def writer(stage):
                (stage/'steps').mkdir()
                rows=[];files=[]
                for step in range(start,stop):
                    i=ORDER[step]
                    lr,hr,gt=chunk_inputs(out,i)
                    feed={xl:lr,xh:hr}
                    sd=stage/'steps'/f'step_{step:04d}_s{i:04d}'
                    sd.mkdir()
                    prior={}
                    if arm=='dynamic':
                        before=np.asarray(sess.run(fetch['crop'],feed_dict=feed)[0],dtype=np.float32)
                        need(before.shape==(160,160) and np.isfinite(before).all(),'Dynamic crop invalid')
                        np.save(sd/'dynamic_before.npy',before)
                        worker_call(root,out,sd,i,backend,timeout)
                        pairs=read_feed(sd/'feed.npz')
                        prior=jread(sd/'match.json')
                        need(prior['sample']==i,'Fresh match belongs to wrong training example')
                        files.extend(str(x.relative_to(stage)) for x in
                             [sd/'dynamic_before.npy',sd/'feed.npz',sd/'match.json',
                              sd/'vti'/'dynamic_before.vti',sd/'pd'/'dynamic_before_pd_port_0.vtu'])
                    elif arm=='fixed':
                        fixed=fixed_pair_check(out,i,p)
                        need(jread(fixed/'fixed_manifest.json')['initial_model_digest']==initial_hash,
                             f'Frozen initial mapping not generated with current pretrained model at s{i}')
                        pairs=read_feed(fixed/'feed.npz')
                        prior={'initial_pair_sha256':sha(fixed/'match.json')}
                    before_metrics=sess.run({'L_uv':fetch['L_uv'],
                         'L_grad':fetch['L_grad']},feed_dict=feed)
                    lossfeed=add_feed(ph,feed,pairs) if arm!='grad' else feed
                    base_norm=float(sess.run(fetch['base_grad_norm'],feed_dict=feed))
                    need(math.isfinite(base_norm),'Nonfinite base gradient norm')
                    if arm!='grad':
                        J=float(sess.run(fetch['J_frozen'],feed_dict=lossfeed))
                        need(math.isfinite(J) and J>=0,'Nonfinite / negative fixed matching J')
                        if arm=='dynamic':
                            expected=float(pairs['expected_squared_cost'])
                            need(math.isclose(J,expected,rel_tol=3e-4,abs_tol=1e-2),
                                 f'Step {step} freshly matched J={J} != GUDHI {expected}')
                        pd_norm=float(sess.run(fetch['PD_grad_norm_unscaled'],feed_dict=lossfeed))
                        total_norm=float(sess.run(fetch['dynamic_grad_norm'],feed_dict=lossfeed))
                        need(math.isfinite(pd_norm) and math.isfinite(total_norm),
                             'Nonfinite topology or combined gradient')
                    else:J=0.;pd_norm=0.;total_norm=base_norm
                    losskey='baseline' if arm=='grad' else 'dynamic'
                    sess.run(trains[losskey],feed_dict=lossfeed)
                    after=sess.run({'L_uv':fetch['L_uv'],'L_grad':fetch['L_grad'],
                        'crop':fetch['crop']},feed_dict=feed)
                    crop=np.asarray(after['crop'][0],dtype=np.float32)
                    need(crop.shape==(160,160) and np.isfinite(crop).all(),
                         f'Nonfinite model prediction after step {step}')
                    row={'step':step,'index':i,'epoch':step//N,'arm':arm,
                         'L_uv_before':float(before_metrics['L_uv']),
                         'L_grad_before':float(before_metrics['L_grad']),
                         'L_uv_after':float(after['L_uv']),
                         'L_grad_after':float(after['L_grad']),
                         'speed_crop_mae_mps_after':float(np.mean(np.abs(crop-gt),dtype=np.float64)),
                         'J_frozen_W22_squared':J,'PD_grad_norm_unscaled':pd_norm,
                         'base_grad_norm':base_norm,'combined_grad_norm':total_norm,
                         'lambda_times_PD_to_base_grad_ratio':LAM*pd_norm/max(base_norm,1e-30),
                         'match_source':prior}
                    need(all(math.isfinite(row[k]) for k in
                        ('L_uv_before','L_grad_before','L_uv_after','L_grad_after',
                         'speed_crop_mae_mps_after','J_frozen_W22_squared','PD_grad_norm_unscaled',
                         'base_grad_norm','combined_grad_norm','lambda_times_PD_to_base_grad_ratio')),
                         f'Nonfinite training metric step {step}')
                    jwrite(sd/'step.json',row)
                    files.append(str((sd/'step.json').relative_to(stage)))
                    rows.append(row)
                    if step%8==0 or max_new is not None:
                        print(f'[TRAIN-{arm.upper()}] step={step+1:03d}/504, '
                              f'source_idx={i}, L_uv={row["L_uv_after"]:.7f} '
                              f'PD-J={J:.6g}',flush=True)
                checkpoint_path=str(stage/'tf_state')
                full.save(sess,checkpoint_path,write_meta_graph=False,write_state=False)
                ckpt_files=['tf_state.index','tf_state.data-00000-of-00001']
                need(all((stage/f).is_file() for f in ckpt_files),
                     'Optimizer-state checkpoint incomplete')
                files.extend(ckpt_files)
                jwrite(stage/'manifest.json',{
                    'schema':SCHEMA+'-chunk','arm':arm,'start':start,'end':stop,
                    'plan_sha256':sha(out/'plan'/'manifest.json'),
                    'previous_manifest_sha256':prevhash,
                    'initial_model_digest':initial_hash,
                    'model_digest_end':init_model_digest(sess,vars_),
                    'steps':rows,'hashes':p.hash_files(stage,files),
                    'complete_optimizer_state':True,
                    'checkpoint_note':'Full TF1 generator and Adam-slot variables'} )
            # This atomic stage publishes ONLY at segment boundary. On failure,
            # process must stop: in-memory Adam state has advanced beyond archive.
            p.atomic(dest,writer)
            manifest=jread(dest/'manifest.json')
            p.verify_files(dest,manifest['hashes'])
            prevhash=sha(dest/'manifest.json')
            next_step=stop;completed+=this_many
            print(f'[SEGMENT] PASS: {arm} {start:03d}..{stop:03d}/504, '
                  'full Adam state persisted; safely resumable',flush=True)
    print(f'[TRAIN] PASS: arm={arm} sealed up to {next_step}/504 steps; '
          'run again to continue if not complete',flush=True)


def verify(root,out,arm,full):
    p,bridge,cp,plan=check_plan(root,out,full_data_hash=full)
    inputs_check(root,out,p);gt_check(root,out,p)
    arms=ARMS if arm=='all' else (arm,)
    before=[]
    for a in arms:
        chunks,n,last=segments(out,a,p)
        print(f'[VERIFY] {a}: {n}/{STEPS} verified steps, {len(chunks)} checkpointed segments',flush=True)
        if chunks:
            initial=jread(chunks[0]/'manifest.json')['initial_model_digest']
            before.append(initial)
            need(all(jread(s/'manifest.json')['initial_model_digest']==initial for s in chunks),
                 f'{a}: initial checkpoint digest changed')
        if a=='fixed':
            for i in set(ORDER[:n]):fixed_pair_check(out,i,p)
        if n==STEPS:
            print(f'[VERIFY-{a.upper()}] PASS: full 504-step training ledger and Adam checkpoints',flush=True)
        else:
            print(f'[VERIFY-{a.upper()}] PARTIAL: resume training to reach 504 steps',flush=True)
    if len(before)>1:need(len(set(before))==1,'Arms did not start at equal checkpoint weights')
    print('[VERIFY] PASS: source/checkpoint SHA, schedule, segment chain, sealed artifacts; '
          'no locked regions',flush=True)


def cli():
    pa=argparse.ArgumentParser(description=__doc__)
    pa.add_argument('--mode',required=True,choices=('selftest','preflight','plan','export',
        'prepare','fixed-cache','train','verify','match-worker'))
    pa.add_argument('--repo-root',default=str(SELF.parent))
    pa.add_argument('--out',default=OUT)
    pa.add_argument('--arm',choices=(*ARMS,'all'))
    pa.add_argument('--ttk-backend',default='docker',choices=('docker','native'))
    pa.add_argument('--ttk-timeout',type=int,default=DEFAULT_TTK_TIMEOUT)
    pa.add_argument('--max-new-samples',type=int)
    pa.add_argument('--max-new-steps',type=int)
    pa.add_argument('--index',type=int)
    pa.add_argument('--step-dir')
    pa.add_argument('--full-data-hash',action='store_true')
    a=pa.parse_args()
    root=Path(a.repo_root).resolve()
    out=(root/a.out).resolve()
    base=root/'ttk_runs_fixed'/'topology_finetuning'
    need(root.is_dir() and out.parent==base and out.name.startswith('phase8_train168_'),
         'Output must be a dedicated phase8_train168_* child of topology_finetuning')
    need(a.ttk_timeout>=1,'Invalid TTK timeout')
    for k in ('max_new_samples','max_new_steps'):
        val=getattr(a,k)
        if val is not None:need(0<val<=168 if k=='max_new_samples' else 0<val<=STEPS,
                               f'Invalid --{k.replace("_","-")}')
    if a.mode=='selftest':selftest()
    elif a.mode=='preflight':preflight(root,out)
    elif a.mode=='plan':make_plan(root,out)
    elif a.mode=='export':export(root,out)
    elif a.mode=='prepare':prepare(root,out,a.ttk_backend,a.ttk_timeout)
    elif a.mode=='fixed-cache':fixed_cache(root,out,a.ttk_backend,a.ttk_timeout,a.max_new_samples)
    elif a.mode=='train':
        need(a.arm in ARMS,'Training requires --arm grad|fixed|dynamic')
        train(root,out,a.arm,a.ttk_backend,a.ttk_timeout,a.max_new_steps)
    elif a.mode=='verify':verify(root,out,a.arm or 'all',a.full_data_hash)
    elif a.mode=='match-worker':
        need(a.index in SELECT and a.step_dir,'Worker requires selected --index and --step-dir')
        match_worker(root,out,Path(a.step_dir).resolve(),a.index,a.ttk_backend,a.ttk_timeout)
    else:raise RuntimeError('Unknown mode')

if __name__=='__main__':
    try:cli()
    except Exception as e:
        print(f'[PHASE8-TRAIN168] ERROR {type(e).__name__}: {e}',file=sys.stderr,flush=True)
        sys.exit(1)
