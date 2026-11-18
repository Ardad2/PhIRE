#!/usr/bin/env python3
"""Phase 8 development evaluation of FINAL 168-field/3-epoch training checkpoints.

Compares pretrained CNN, gradient control, initial-fixed-pair PD, and freshly
rematched dynamic PD on the preexisting ORIGINAL 168-field Mid-Atlantic benchmark.
The train source is the distinct expanded 2688 TFRecord; NEVER retrain or access
Phase7 geographic data / Upper Midwest / Pacific inland. All writes go to new OUT.

Modes:
  selftest, preflight (read-only); predict (native Python/TensorFlow),
  score, verify-score, locality, verify-locality, report (gudhi-audit env).

Publish each prediction arm and each scored/locality field atomically. Pinned
source digests and complete model/Adam checkpoint chain. Prediction and metrics
stages are append-only; failure preserves forensic staging but never overwrites.
"""
from __future__ import annotations
import argparse, csv, hashlib, importlib.util, json, math, sys
from collections import Counter
from pathlib import Path
from statistics import mean
import numpy as np

TRAIN_SCRIPT='phase8_train168_threearm_dynamic_pd_20261118.py'
BENCH_SCRIPT='phase8_original168_dynamic_pd_benchmark_20261118.py'
TRAIN_SHA='6dccfd25f10a85ac09dcde6425c5de2ee6cf2bf622da9e12bec800adf8301fa2'
BENCH_SHA='c28b2ed444083668472f5527d53cdd4a7cd04e3c2611868062668091f64192d0'
TRAIN_OUT='ttk_runs_fixed/topology_finetuning/phase8_train168_threearm_20261118_r01'
BENCH_OUT='ttk_runs_fixed/topology_finetuning/phase8_original168_eightstep_benchmark_20261118_r01'
OUT='ttk_runs_fixed/topology_finetuning/phase8_train168_final_original168_eval_20261118_r01'
SCHEMA='phase8-train168-final-original168-eval-v1'
ARMS=('cnn','grad','fixed','dynamic')
SAMPLES=tuple(range(168))
RADII=(4.,8.,16.,32.,64.,96.,128.,math.inf)
METRICS=('W22','W2inf','dB','L_uv','L_grad','speed_crop_MAE_mps')
SOURCE=Path(__file__).resolve()

def need(cond,msg):
    if not cond:raise RuntimeError(msg)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()

def load(path,name):
    sp=importlib.util.spec_from_file_location(name,str(path))
    need(sp and sp.loader,f'Missing importable source {path}')
    m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m)
    return m

def rd(path):return json.loads(Path(path).read_text())
def write(path,obj):Path(path).write_text(json.dumps(obj,sort_keys=True,indent=2,allow_nan=False)+'\n')
def lab(r):return 'inf' if math.isinf(r) else str(r)
def pdir(out,arm):return out/'predictions'/arm
def sdir(out,i):return out/'scores'/f's{i:03d}'
def ldir(out,i):return out/'locality_samples'/f's{i:03d}'

def selftest():
    need(len(SAMPLES)==168 and ARMS==('cnn','grad','fixed','dynamic') and
         tuple(map(lab,RADII))==('4.0','8.0','16.0','32.0','64.0','96.0','128.0','inf'),
         'Fixed scientific protocol changed')
    need('phase7' not in (TRAIN_OUT+BENCH_OUT+OUT).lower() and
         'upper_midwest' not in OUT and 'pacific_inland' not in OUT,
         'Forbidden geographic scope')
    test={'sample':0,'arms':{a:{'W22':float(i)} for i,a in enumerate(ARMS)}}
    need(test['arms']['dynamic']['W22']-test['arms']['grad']['W22']==2.,
         'Paired bookkeeping failed')
    need(json.loads(json.dumps({'0':{'0':3}}))=={'0':{'0':3}},
         'JSON normalization failed')
    print('[SELFTEST] PASS: original 168 development benchmark, 4 frozen arms, original 8 radii; no files written')

def context(root,verify=True):
    need(root==SOURCE.parent,'Install evaluator at PhIRE repository root')
    need(sha(root/TRAIN_SCRIPT)==TRAIN_SHA,'Original three-arm trainer changed; stop')
    need(sha(root/BENCH_SCRIPT)==BENCH_SHA,'Original 168 benchmark driver changed; stop')
    t=load(root/TRAIN_SCRIPT,'p8_eval_verified_train')
    b=load(root/BENCH_SCRIPT,'p8_eval_verified_benchmark')
    to=root/TRAIN_OUT;bo=root/BENCH_OUT
    tp,bridge,cp,plan=t.check_plan(root,to,full_data_hash=verify)
    t.inputs_check(root,to,tp);t.gt_check(root,to,tp)
    finals={}
    for a in ('grad','fixed','dynamic'):
        chunks,n,_=t.segments(to,a,tp)
        need(n==504 and chunks,f'Arm {a} not complete: {n}/504')
        manifest=rd(chunks[-1]/'manifest.json')
        need(manifest['end']==504 and manifest['complete_optimizer_state'] and
             set(['tf_state.index','tf_state.data-00000-of-00001'])<=set(manifest['hashes']),
             f'Incomplete TF1 checkpoint in final chunk {a}')
        finals[a]={'chunk':chunks[-1],
                   'chunk_manifest_sha256':sha(chunks[-1]/'manifest.json'),
                   'checkpoint_sha256':{k:manifest['hashes'][k] for k in
                       ('tf_state.index','tf_state.data-00000-of-00001')}}
    # Read only the existing, already SHA-verified benchmark exports and replay.
    bp=b.install(root)
    provenance,pilot,bb,original_cp,old,prior=bp.preflight(root,bo,require_new=False)
    b.replay_check(root,bo,bp,pilot,bb,original_cp,old)
    need(sha(root/t.BENCHMARK)==plan['reference_benchmark_tfrecord_sha256'],
         'Evaluation TFRecord drifted from training plan')
    return {'t':t,'b':b,'tp':tp,'bridge':bridge,'cp':cp,'plan':plan,
            'trainout':to,'benchout':bo,'pilot':pilot,'benchp':bp,'finals':finals}

def provenance(root,c):
    return {'schema':SCHEMA+'-provenance',
            'source_sha256':sha(SOURCE),
            'trainer_sha256':TRAIN_SHA,'old_benchmark_driver_sha256':BENCH_SHA,
            'training_plan_sha256':sha(c['trainout']/'plan'/'manifest.json'),
            'benchmark_inputs_manifest_sha256':sha(c['benchout']/'eval_inputs'/'manifest.json'),
            'benchmark_replay_manifest_sha256':sha(c['benchout']/'replay'/'manifest.json'),
            'benchmark_report_sha256':sha(c['benchout']/'report'/'report.md'),
            'original_168_tfrecord_sha256':sha(root/c['t'].BENCHMARK),
            'trained_final_checkpoints':{a:{k:v for k,v in f.items() if k!='chunk'}
                for a,f in c['finals'].items()},
            'scope':'development benchmark; locked geography never read',
            'training':'168 distinct training fields, 3 epochs, 504 batch-1 updates per arm',
            'models':list(ARMS)}

def verify_pred(out,arm,c,root):
    d=pdir(out,arm);need(d.is_dir(),f'Missing predictions for {arm}; --mode predict first')
    m=rd(d/'manifest.json')
    need(m['schema']==SCHEMA+'-prediction-arm' and m['arm']==arm and
         m['provenance']==provenance(root,c),'Prediction provenance drift')
    c['tp'].verify_files(d,m['hashes'])
    need(set(m['metrics'])==set(map(str,SAMPLES)),f'Incomplete prediction metric map {arm}')
    return m

def predict(root,out,c,max_arms):
    t=c['t'];to=c['trainout'];bo=c['benchout']
    (out/'predictions').mkdir(parents=True,exist_ok=True)
    # The frozen training graph exactly matches the generator and normalization.
    tp,bridge,tf,xl,xh,ph,fetch,trains,saver,init,vars_,full=t.cnn_graph(root)
    conf=tf.ConfigProto();conf.gpu_options.allow_growth=True
    done=0
    for arm in ARMS:
        dest=pdir(out,arm)
        if dest.exists():
            verify_pred(out,arm,c,root)
            print(f'[PREDICT] {arm} verified cached',flush=True)
            continue
        if max_arms is not None and done>=max_arms:
            print(f'[PARTIAL-PREDICT] {done} NEW model arms, resume predict',flush=True)
            return
        def writer(tmp):
            names=[];metrics={}
            with tf.Session(config=conf) as sess:
                sess.run(init)
                if arm=='cnn':saver.restore(sess,str(root/tp.CHECKPOINT))
                else:full.restore(sess,str(c['finals'][arm]['chunk']/'tf_state'))
                for i in SAMPLES:
                    base=bo/'eval_inputs'
                    lr=np.load(base/f's{i:04d}_lr.npy',allow_pickle=False)
                    hr=np.load(base/f's{i:04d}_hr.npy',allow_pickle=False)
                    gt=np.load(base/f's{i:04d}_gt_crop.npy',allow_pickle=False)
                    need(lr.shape==(1,100,100,2) and hr.shape==(1,500,500,2) and
                         gt.shape==(160,160),'Original168 input shapes changed')
                    result=sess.run({k:fetch[k] for k in ('crop','L_uv','L_grad')},
                                    feed_dict={xl:lr,xh:hr})
                    crop=np.asarray(result['crop'][0],dtype=np.float32)
                    need(crop.shape==(160,160) and np.isfinite(crop).all(),
                         f'Invalid crop for {arm} sample {i}')
                    if arm=='cnn':
                        oldpred=np.load(bo/'replay'/'predictions'/f's{i:04d}_cnn.npy',
                                        allow_pickle=False)
                        need(float(np.max(np.abs(crop-oldpred)))<0.005,
                             f'Pretrained-CNN prediction differs from earlier benchmark sample={i}')
                    met={'L_uv':float(result['L_uv']),'L_grad':float(result['L_grad']),
                         'speed_crop_MAE_mps':float(np.mean(np.abs(crop-gt),dtype=np.float64))}
                    need(all(math.isfinite(v) for v in met.values()),'Nonfinite eval metric')
                    metrics[str(i)]=met
                    name=f's{i:04d}.npy';np.save(tmp/name,crop);names.append(name)
                    if i%24==23:print(f'[PREDICT-{arm.upper()}] {i+1:03d}/168',flush=True)
            write(tmp/'manifest.json',{'schema':SCHEMA+'-prediction-arm',
                  'arm':arm,'provenance':provenance(root,c),'metrics':metrics,
                  'hashes':tp.hash_files(tmp,names)})
        tp.atomic(dest,writer);verify_pred(out,arm,c,root);done+=1
        print(f'[PREDICT] PASS: {arm}, 168 immutable benchmark predictions',flush=True)
    print('[PREDICT] PASS: all 4 model arms predicted on original 168 fields; no training updates',flush=True)

def verify_score_sample(root,out,c,i):
    d=sdir(out,i);need(d.is_dir(),f'Missing score for sample {i}')
    m=rd(d/'manifest.json')
    need(m['schema']==SCHEMA+'-score-sample' and m['sample']==i and
         m['evaluator_sha256']==sha(SOURCE) and
         m['prediction_manifests']=={a:sha(pdir(out,a)/'manifest.json') for a in ARMS} and
         m['original_benchmark_inputs_sha256']==sha(c['benchout']/'eval_inputs'/'manifest.json'),
         f'Score lineage drift sample {i}')
    c['tp'].verify_files(d,m['hashes'])
    return rd(d/'results.json')

def score(root,out,c,backend,timeout,max_new):
    p,bridge,_=c['t'].deps(root)
    timer,phase6,decoder,_,_=p.audit_modules(root,bridge)
    timer.check_backend(backend)
    for arm in ARMS:verify_pred(out,arm,c,root)
    (out/'scores').mkdir(parents=True,exist_ok=True)
    added=0
    for i in SAMPLES:
        dest=sdir(out,i)
        if dest.exists():
            verify_score_sample(root,out,c,i)
            if i%24==23:print(f'[SCORE] {i+1:03d}/168 verified cached',flush=True)
            continue
        if max_new is not None and added>=max_new:
            print(f'[PARTIAL-SCORE] new={added}; resume with --mode score',flush=True)
            return
        fields={'gt':np.load(c['benchout']/'eval_inputs'/f's{i:04d}_gt_crop.npy',allow_pickle=False)}
        for a in ARMS:
            fields[a]=np.load(pdir(out,a)/f's{i:04d}.npy',allow_pickle=False)
        need(all(v.shape==(160,160) and np.isfinite(v).all() for v in fields.values()),
             f'Nonfinite/shape mismatch s{i}')
        def writer(tmp):
            (tmp/'vti').mkdir();(tmp/'pd').mkdir()
            diagrams={};names=[];timings={}
            for label,field in fields.items():
                timer.write_vti(tmp/'vti'/f'{label}.vti',field)
                timings[label]=timer.run_ttk(backend,tmp,label,timeout)
                diagrams[label]=phase6.parse(decoder.read_pd,tmp/'pd'/f'{label}_pd_port_0.vtu')
                need(all(len(diagrams[label][d])<=6000 for d in (0,1)),
                     f'Excess diagram size sample {i}, {label}')
                names.extend([f'vti/{label}.vti',f'pd/{label}_pd_port_0.vtu'])
            results={}
            for arm in ARMS:
                pd=phase6.distance(diagrams['gt'],diagrams[arm])
                met=rd(pdir(out,arm)/'manifest.json')['metrics'][str(i)]
                results[arm]={**pd,**met}
                need(all(math.isfinite(float(v)) for v in results[arm].values()),
                     f'Nonfinite score sample {i} arm {arm}')
            if i==0:need(results['dynamic']['W22']>=0,'Sanity failed')
            # New CNN must agree numerically with the existing original benchmark.
            prior=c['b'].check_score_sample(c['benchout'],i,c['pilot'])
            for key in ('W22','W2inf','dB'):
                need(math.isclose(results['cnn'][key],prior['arms']['cnn'][key],
                     abs_tol=0.005,rel_tol=1e-5),
                     f'CNN historical PD regression mismatch s{i} {key}')
            write(tmp/'results.json',{'sample':i,'arms':results,'timings':timings})
            names+=['results.json']
            write(tmp/'manifest.json',{'schema':SCHEMA+'-score-sample',
                'sample':i,'evaluator_sha256':sha(SOURCE),
                'prediction_manifests':{a:sha(pdir(out,a)/'manifest.json') for a in ARMS},
                'original_benchmark_inputs_sha256':sha(c['benchout']/'eval_inputs'/'manifest.json'),
                'hashes':p.hash_files(tmp,names)})
        p.atomic(dest,writer)
        result=verify_score_sample(root,out,c,i)
        added+=1
        print(f'[SCORE] {i+1:03d}/168 grad={result["arms"]["grad"]["W22"]:.5f} '
              f'fixed={result["arms"]["fixed"]["W22"]:.5f} '
              f'dynamic={result["arms"]["dynamic"]["W22"]:.5f}',flush=True)
    print('[SCORE] PASS: original168 4 model arms, canonical GUDHI; untouched geographic tests',flush=True)

def compute_locality(solver,score_dir,score_row):
    features={};skipped={}
    for a in ('gt',*ARMS):
        features[a],skipped[a]=solver.read_features(score_dir/'pd'/f'{a}_pd_port_0.vtu')
    per_arm={}
    for arm in ARMS:
        preps={dim:solver.SparsePrepared(features['gt'][dim],features[arm][dim]) for dim in (0,1)}
        result={}
        for radius in RADII:
            dr={str(dim):preps[dim].solve(radius) for dim in (0,1)}
            total=sum(v['total_sq'] for v in dr.values())
            gt_p=sum(v['gt_total_persistence'] for v in dr.values())
            sr_p=sum(v['sr_total_persistence'] for v in dr.values())
            gt_m=sum(v['gt_matched_persistence'] for v in dr.values())
            sr_m=sum(v['sr_matched_persistence'] for v in dr.values())
            result[lab(radius)]={'w22':math.sqrt(max(0.,total)),
               'gt_persistence_coverage':gt_m/gt_p if gt_p>0 else None,
               'sr_persistence_coverage':sr_m/sr_p if sr_p>0 else None,
               'gt_to_diagonal':sum(v['gt_to_diagonal'] for v in dr.values()),
               'diagonal_to_sr':sum(v['diagonal_to_sr'] for v in dr.values()),
               'real_real':sum(v['real_real'] for v in dr.values()),
               'dimensions':dr}
        need(math.isclose(result['inf']['w22'],score_row['arms'][arm]['W22'],
            abs_tol=1e-5,rel_tol=1e-6),
            f'Spatial infinity vs canonical GUDHI differs s{score_row["sample"]} {arm}')
        per_arm[arm]=result
    return {'sample':score_row['sample'],'arms':per_arm,'reader_skipped_zeros':skipped,
            'solver':'unchanged Phase5E SparsePrepared'}

def verify_loc_sample(root,out,c,i):
    d=ldir(out,i);need(d.is_dir(),f'Locality missing sample={i}')
    m=rd(d/'manifest.json')
    need(m['schema']==SCHEMA+'-locality-sample' and m['sample']==i and
         m['source_score_manifest_sha256']==sha(sdir(out,i)/'manifest.json') and
         m['original_solver_sha256']==sha(root/c['b'].SOLVER) and
         m['evaluator_sha256']==sha(SOURCE),f'Locality provenance drift s{i}')
    c['tp'].verify_files(d,m['hashes'])
    return rd(d/'result.json')

def locality(root,out,c,max_new):
    solver=c['b'].load_solver(root)
    (out/'locality_samples').mkdir(parents=True,exist_ok=True)
    added=0
    for i in SAMPLES:
        if ldir(out,i).exists():
            verify_loc_sample(root,out,c,i)
            if i%24==23:print(f'[LOCALITY] {i+1:03d}/168 verified cached',flush=True)
            continue
        if max_new is not None and added>=max_new:
            print(f'[PARTIAL-LOCALITY] new={added}; resume --mode locality',flush=True)
            return
        row=verify_score_sample(root,out,c,i)
        def writer(tmp):
            result=compute_locality(solver,sdir(out,i),row)
            write(tmp/'result.json',result)
            write(tmp/'manifest.json',{'schema':SCHEMA+'-locality-sample','sample':i,
                'source_score_manifest_sha256':sha(sdir(out,i)/'manifest.json'),
                'original_solver_sha256':sha(root/c['b'].SOLVER),
                'evaluator_sha256':sha(SOURCE),'hashes':c['tp'].hash_files(tmp,['result.json'])})
        c['tp'].atomic(ldir(out,i),writer);verify_loc_sample(root,out,c,i)
        added+=1
        print(f'[LOCALITY] {i+1:03d}/168 original Phase5E eight radii PASS',flush=True)
    print('[LOCALITY] PASS: 168 development fields, 4 model arms, 8 radii, H0/H1',flush=True)

def verify_score(root,out,c):
    p,bridge,_=c['t'].deps(root)
    _,phase6,decoder,_,_=p.audit_modules(root,bridge)
    for i in SAMPLES:
        row=verify_score_sample(root,out,c,i)
        ss=sdir(out,i)
        diagrams={a:phase6.parse(decoder.read_pd,ss/'pd'/f'{a}_pd_port_0.vtu') for a in ('gt',*ARMS)}
        need(row['sample']==i,'Wrong score identity')
        for arm in ARMS:
            numeric=phase6.distance(diagrams['gt'],diagrams[arm])
            for metric in ('W22','W2inf','dB'):
                need(math.isclose(numeric[metric],row['arms'][arm][metric],
                     abs_tol=1e-6,rel_tol=1e-8),f'Numerical disagreement sample={i} {arm} {metric}')
        if i%24==23:print(f'[VERIFY-SCORE] {i+1:03d}/168',flush=True)
    dest=out/'verification'/'scores'
    if dest.exists():
        m=rd(dest/'manifest.json');need(m['hashes']==score_hashes(out),'Score verification seal drift')
    else:
        def writer(tmp):write(tmp/'manifest.json',{'schema':SCHEMA+'-verified-scores',
                 'evaluator_sha256':sha(SOURCE),'hashes':score_hashes(out)})
        c['tp'].atomic(dest,writer)
    print('[VERIFY-SCORE] PASS: 168x4 independently recomputed canonical PD distances',flush=True)

def score_hashes(out):return {str(i):sha(sdir(out,i)/'manifest.json') for i in SAMPLES}
def locality_hashes(out):return {str(i):sha(ldir(out,i)/'manifest.json') for i in SAMPLES}

def verify_locality(root,out,c):
    need((out/'verification'/'scores'/'manifest.json').is_file(),
         'Run verify-score first')
    need(rd(out/'verification'/'scores'/'manifest.json')['hashes']==score_hashes(out),
         'Score verification seal drift')
    solver=c['b'].load_solver(root)
    for i in SAMPLES:
        row=verify_score_sample(root,out,c,i)
        previous=verify_loc_sample(root,out,c,i)
        computed=compute_locality(solver,sdir(out,i),row)
        # JSON converts H0/H1 integer dictionary keys to strings. Match that schema
        # first, then tolerate harmless floating-point solver roundoff while requiring
        # exact identities, integer counts, and feature assignments.
        present=json.loads(json.dumps(computed,allow_nan=False))
        def close_tree(left,right):
            if isinstance(left,dict) and isinstance(right,dict):
                return left.keys()==right.keys() and all(close_tree(left[k],right[k]) for k in left)
            if isinstance(left,list) and isinstance(right,list):
                return len(left)==len(right) and all(close_tree(x,y) for x,y in zip(left,right))
            if isinstance(left,(int,float)) and not isinstance(left,bool) and \
               isinstance(right,(int,float)) and not isinstance(right,bool):
                return math.isclose(left,right,rel_tol=1e-10,abs_tol=1e-8)
            return left==right
        need(close_tree(present,previous),
             f'Independent Phase5E locality disagreement sample={i}')
        if i%24==23:print(f'[VERIFY-LOCALITY] {i+1:03d}/168',flush=True)
    dest=out/'verification'/'locality'
    if dest.exists():
        m=rd(dest/'manifest.json')
        need(m['score_hashes']==score_hashes(out) and
             m['locality_hashes']==locality_hashes(out),
             'Locality verification seal drift')
    else:
        def writer(tmp):write(tmp/'manifest.json',{'schema':SCHEMA+'-verified-locality',
                 'evaluator_sha256':sha(SOURCE),'score_hashes':score_hashes(out),
                 'locality_hashes':locality_hashes(out),'solver_sha256':sha(root/c['b'].SOLVER)})
        c['tp'].atomic(dest,writer)
    print('[VERIFY-LOCALITY] PASS: 168x4 original Phase5E independently recomputed, all radii',flush=True)

def training_audit(c):
    """Read-only descriptive training ledger (not model-selection evidence)."""
    t=c['t']; tp=c['tp']; out=c['trainout']
    print('[AUDIT] Three independently resumed 504-step training arms:')
    for arm in ('grad','fixed','dynamic'):
        chunks,n,_=t.segments(out,arm,tp)
        need(n==504,'Training audit requires sealed full training')
        steps=[x for ch in chunks for x in rd(ch/'manifest.json')['steps']]
        need(len(steps)==504 and all(x['arm']==arm for x in steps),
             f'Incorrect arm/ledger sequence {arm}')
        print('\n'+arm+' (batch 1, identical three-epoch order)')
        for epoch in range(3):
            r=steps[epoch*168:(epoch+1)*168]
            need(all(int(x['epoch'])==epoch for x in r),'Epoch ledger mismatch')
            vals={key:mean(float(x[key]) for x in r) for key in
                  ('L_uv_after','L_grad_after','speed_crop_mae_mps_after',
                   'J_frozen_W22_squared','lambda_times_PD_to_base_grad_ratio')}
            print(f'  epoch {epoch+1}: L_uv={vals["L_uv_after"]:.7g}, '
                  f'L_grad={vals["L_grad_after"]:.7g}, '
                  f'train_crop_speed_MAE={vals["speed_crop_mae_mps_after"]:.6f} m/s, '
                  f'J_fixed={vals["J_frozen_W22_squared"]:.5f}, '
                  f'lambda*gradPD/gradBase={vals["lambda_times_PD_to_base_grad_ratio"]:.5g}')
    print('\n[AUDIT] PASS: training objectives summarized; these are TRAINING measurements, not held-out model comparisons')


def report(root,out,c):
    need((out/'verification'/'locality'/'manifest.json').is_file() and
         (out/'verification'/'scores'/'manifest.json').is_file(),
         'Run both independent verification modes before report')
    need(rd(out/'verification'/'scores'/'manifest.json')['hashes']==score_hashes(out),
         'Score verification seal mismatch')
    lm=rd(out/'verification'/'locality'/'manifest.json')
    need(lm['locality_hashes']==locality_hashes(out) and lm['score_hashes']==score_hashes(out),
         'Locality verification seal mismatch')
    need(not (out/'report').exists(),'Immutable report already exists; never overwrite')
    scores=[verify_score_sample(root,out,c,i) for i in SAMPLES]
    locs=[verify_loc_sample(root,out,c,i) for i in SAMPLES]
    means={arm:{key:float(mean(r['arms'][arm][key] for r in scores)) for key in METRICS}
           for arm in ARMS}
    space={}
    for arm in ARMS:
        space[arm]={}
        for radius in map(lab,RADII):
            rows=[r['arms'][arm][radius] for r in locs]
            space[arm][radius]={
              'mean_w22':float(mean(x['w22'] for x in rows)),
              'mean_gt_coverage':float(mean(x['gt_persistence_coverage'] for x in rows
                                             if x['gt_persistence_coverage'] is not None)),
              'mean_sr_coverage':float(mean(x['sr_persistence_coverage'] for x in rows
                                             if x['sr_persistence_coverage'] is not None)),
              'mean_gt_to_diagonal':float(mean(x['gt_to_diagonal'] for x in rows)),
              'mean_diagonal_to_sr':float(mean(x['diagonal_to_sr'] for x in rows))}
        need(math.isclose(space[arm]['inf']['mean_w22'],means[arm]['W22'],
              abs_tol=1e-5,rel_tol=1e-6),'Locality infinity aggregate regression')
    pairs={}
    for a,b in (('fixed','grad'),('dynamic','grad'),('dynamic','fixed')):
        diff=[r['arms'][a]['W22']-r['arms'][b]['W22'] for r in scores]
        pairs[f'{a}_minus_{b}']={'mean_w22_delta':float(mean(diff)),
             'better_fields':sum(v<0 for v in diff),'worse_fields':sum(v>0 for v in diff),
             'daily_blocks':[{'day':j+1,'mean_delta':float(mean(diff[j*24:(j+1)*24])),
                               'better_fields':sum(x<0 for x in diff[j*24:(j+1)*24])}
                              for j in range(7)]}
    findings={'schema':SCHEMA+'-summary','evaluation_fields':168,
              'training_steps_per_arm':504,'training_fields':168,'epochs':3,
              'source':provenance(root,c),'means':means,'spatial_locality':space,'paired':pairs,
              'limitations':['Development benchmark previously used for model and hyperparameter selection; not independent test',
               'Training uses batch size 1, not manuscript-scale batch size 4 / 2688 fields',
               'Fixed pair means initial pretrained CNN correspondence and critical vertex locations held frozen',
               'Dynamic pairs updated each step; evaluation always recomputes fresh diagrams',
               'Time-adjacent hourly fields are correlated (seven 24-hour blocks)',
               'PD and spatial locality use top-left 160x160 speed crop, not full 500x500',
               'Pretraining exclusion cannot be demonstrated',
               'Upper Midwest and Pacific inland geographic tests not accessed']}
    lines=['# Phase 8 — Three-arm 168-training-field model comparison on original 168 benchmark','',
      '**Development comparison, not an independent geographic test.**','',
      'All three fine-tuning arms trained on the SAME separate 168 selected fields from the 2688 pool,',
      'three epochs, batch size 1, 504 optimizer updates, identical pretrained CNN initialization.','',
      '| Arm | PD W22 ↓ | W2inf ↓ | dB ↓ | L_uv ↓ | L_grad ↓ | Crop speed MAE m/s ↓ |',
      '|---|---:|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        lines.append('| '+arm+' | '+' | '.join(f'{means[arm][k]:.6f}' for k in METRICS)+' |')
    lines+=['','| Arm | r4 W22 ↓ | r16 W22 ↓ | r32 W22 ↓ | r∞ W22 ↓ | r16 GT coverage ↑ | r16 SR coverage ↑ |',
            '|---|---:|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        q=space[arm]
        lines.append('| '+arm+' | '+' | '.join(f'{q[r]["mean_w22"]:.6f}' for r in ('4.0','16.0','32.0','inf'))+
             f' | {q["16.0"]["mean_gt_coverage"]:.6f} | {q["16.0"]["mean_sr_coverage"]:.6f} |')
    lines+=['','## Paired unconstrained W22 comparisons','',
            '| Comparison (a − b) | Mean difference ↓ | a better hours /168 | a worse hours /168 |',
            '|---|---:|---:|---:|']
    for name,x in pairs.items():
        lines.append(f'| {name} | {x["mean_w22_delta"]:+.6f} | '
                     f'{x["better_fields"]}/168 | {x["worse_fields"]}/168 |')
    lines+=['','## Limitations','',*['- '+x for x in findings['limitations']],'']
    dest=out/'report'
    def writer(tmp):
        write(tmp/'summary.json',findings)
        (tmp/'report.md').write_text('\n'.join(lines))
        with (tmp/'per_field.csv').open('w',newline='') as f:
            cols=['sample','arm',*METRICS,'r4_w22','r16_w22','r32_w22','rinf_w22',
                  'r16_gt_coverage','r16_sr_coverage']
            writer=csv.DictWriter(f,fieldnames=cols);writer.writeheader()
            for i in SAMPLES:
                for arm in ARMS:
                    r=scores[i]['arms'][arm];q=locs[i]['arms'][arm]
                    line={'sample':i,'arm':arm,**r,
                         'r4_w22':q['4.0']['w22'],'r16_w22':q['16.0']['w22'],
                         'r32_w22':q['32.0']['w22'],'rinf_w22':q['inf']['w22'],
                         'r16_gt_coverage':q['16.0']['gt_persistence_coverage'],
                         'r16_sr_coverage':q['16.0']['sr_persistence_coverage']}
                    writer.writerow({k:line[k] for k in cols})
        write(tmp/'manifest.json',{'schema':SCHEMA+'-report-manifest',
             'evaluation_script_sha256':sha(SOURCE),
             'score_verification_manifest_sha256':sha(out/'verification'/'scores'/'manifest.json'),
             'locality_verification_manifest_sha256':sha(out/'verification'/'locality'/'manifest.json'),
             'hashes':c['tp'].hash_files(tmp,['summary.json','report.md','per_field.csv'])})
    c['tp'].atomic(dest,writer)
    print('[REPORT] PASS: final 3-arm + pretrained CNN evaluation, 168 original development fields; '
          'read '+str(dest/'report.md'),flush=True)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode',required=True,choices=('selftest','preflight','training-audit','predict','score','verify-score',
                   'locality','verify-locality','report'))
    ap.add_argument('--repo-root',type=Path,default=SOURCE.parent)
    ap.add_argument('--out',type=Path,default=Path(OUT))
    ap.add_argument('--ttk-backend',default='docker',choices=('docker','native'))
    ap.add_argument('--ttk-timeout',default=180,type=int)
    ap.add_argument('--max-new-samples',type=int)
    ap.add_argument('--max-new-arms',type=int)
    x=ap.parse_args()
    if x.mode=='selftest':selftest();return
    root=x.repo_root.resolve();out=(root/x.out).resolve()
    need(root==SOURCE.parent and out.parent==root/'ttk_runs_fixed'/'topology_finetuning' and
         out.name.startswith('phase8_train168_final_') and out!=root/TRAIN_OUT and out!=root/BENCH_OUT,
         'Refusing output path outside new isolated Phase8 evaluation directory')
    need(x.max_new_samples is None or (x.mode in ('score','locality') and x.max_new_samples>0),
         '--max-new-samples is only for score/locality')
    need(x.max_new_arms is None or (x.mode=='predict' and 1<=x.max_new_arms<=4),
         '--max-new-arms only for predict')
    selftest();c=context(root,verify=True)
    if x.mode=='preflight':
        need(not out.exists(),'Output already exists; proceed using predict/score/verify instead')
        print('[PREFLIGHT] PASS: 3 complete 504-step SHA-chained checkpoints; independent 168 benchmark input verified')
    elif x.mode=='training-audit':training_audit(c)
    elif x.mode=='predict':predict(root,out,c,x.max_new_arms)
    elif x.mode=='score':score(root,out,c,x.ttk_backend,x.ttk_timeout,x.max_new_samples)
    elif x.mode=='verify-score':verify_score(root,out,c)
    elif x.mode=='locality':locality(root,out,c,x.max_new_samples)
    elif x.mode=='verify-locality':verify_locality(root,out,c)
    elif x.mode=='report':report(root,out,c)

if __name__=='__main__':
    try:main()
    except Exception as exc:
        print(f'[PHASE8-TRAIN168-EVAL] ERROR {type(exc).__name__}: {exc}',file=sys.stderr,flush=True)
        sys.exit(1)
