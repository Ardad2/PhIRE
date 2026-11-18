#!/usr/bin/env python3
"""Phase 8: ORIGINAL 168-field (Mid-Atlantic) benchmark of verified 8-step TF1 pilot.

PURPOSE: evaluate existing 8-step control and dynamic-PD model states on the
original, previously used 168-field benchmark, using canonical TTK/GUDHI and
unchanged Phase5E locality matching. THIS IS NOT 168-field fine-tuning and NOT
an independent external test. No Phase7 development-region/locked-test paths.

Run in ~/PhIRE with prior scripts still unchanged:
  --mode selftest
  --mode preflight
  --mode export              (native Python/TensorFlow)
  --mode replay              (native Python/TensorFlow, eight archived steps)
  --mode score               (micromamba gudhi-audit, TTK docker)
  --mode verify-score        (micromamba gudhi-audit)
  --mode locality            (micromamba gudhi-audit, Phase5E sparse solver)
  --mode verify-locality     (micromamba gudhi-audit; expensive independent recomputation)
  --mode report              (micromamba gudhi-audit)

Score/locality are safely resumable at individual benchmark fields; outputs are
atomic, SHA verified, and never overwrite successful artifacts. Original models,
original TFRecord, locked region files and previously completed experiments are
read-only. There are 168 independent *records*, not 168 independent weather events.
"""
from __future__ import annotations
import argparse, csv, hashlib, importlib.util, json, math, sys, time
from collections import Counter
from pathlib import Path
from statistics import mean
import numpy as np

PARENT='phase8_temporal_out_of_pilot_eval_20261117.py'
PILOT='phase8_tf1_multisample_rematch_pilot_20261116.py'
BENCH='example_data_fixed/wind_MR-HR.tfrecord'
SRC='scripts/build_wind_mrhr_expanded_dataset_2688.py'
OUT='ttk_runs_fixed/topology_finetuning/phase8_original168_eightstep_benchmark_20261118_r01'
SOLVER='spatial_pd/phase5e_sparse_locality_solver.py'
SOLVER_SHA='47c73decca4ba151fd8b250cae2911b4c2307471fe9c881fc42bfc968e211c3a'
SAMPLE_IDS=tuple(range(168))
ARMS=('cnn','control','dynamic')
RADII=(4.,8.,16.,32.,64.,96.,128.,math.inf)
SCHEMA='phase8-original168-eightstep-benchmark-r01'
SOURCE=Path(__file__).resolve()

def need(ok,msg):
    if not ok: raise RuntimeError(msg)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def read(p):return json.loads(Path(p).read_text())
def put(p,obj):Path(p).write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n')
def load(path,name):
    spec=importlib.util.spec_from_file_location(name,str(path))
    need(spec and spec.loader,f'Missing source: {path}')
    m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m)
    return m

def metadata(root):
    path=root/BENCH
    need(path.is_file() and path.stat().st_size>0,f'Missing canonical 168 benchmark: {path}')
    # Sample identity is supplied by PhIRE's embedded TFRecord index feature,
    # checked explicitly by export. WTK timestamps are NOT fabricated here.
    return {str(i):{'sample_idx':i,'season':'original_168_Mid_Atlantic_benchmark',
                    'wtk_time_index':i,'timestamp':'not_revalidated_from_source'} for i in SAMPLE_IDS}

def install(root):
    need((root/PARENT).is_file(),f'Missing earlier verified module: {PARENT}')
    p=load(root/PARENT,'phase8_original168_parent')
    p.EVAL=SAMPLE_IDS
    p.TFRECORD=BENCH
    p.MANIFEST=BENCH  # Input integrity is a BYTE SHA256, not a CSV assertion.
    p.GENERATOR=SRC
    p.NEW_OUT=OUT
    p.SCHEMA=SCHEMA
    p.validate_metadata=lambda r:metadata(r)
    def preflight(r,out,require_new=True):
        need(Path(__file__).resolve().parent==r,'Install this driver at ~/PhIRE')
        need((r/BENCH).is_file(),'Original 168 benchmark absent')
        need((r/SRC).is_file(),'Frozen expanded training builder absent')
        need((r/SOLVER).is_file(),'Phase5E solver absent')
        if require_new:need(not out.exists(),'Output exists; use --mode replay/score instead of replacing')
        prov=metadata(r)
        pilot,bridge,cp,old,prior=p.inspect_prior(r)
        need(prior['no_heldout_fields'] and prior['n_steps_per_arm']==8,'Wrong original pilot')
        need(BENCH != str(pilot.TFRECORD) and str(pilot.TFRECORD).endswith('expanded_2688/wind_MR-HR.tfrecord'),
             'Original pilot training source changed')
        print('[BENCH] PASS: fixed original 168 benchmark, 8 original training steps '
              'use DIFFERENT 2688 data source',flush=True)
        print('[SCOPE] Original Mid-Atlantic development benchmark ONLY; no Phase7 region '
              'paths and no locked geographic fields',flush=True)
        return prov,pilot,bridge,cp,old,prior
    p.preflight=preflight
    # Every stage, including independent verification, must check this new
    # wrapper SHA and the exact original-168 TFRecord bytes.
    p.check_exports=lambda r,o,pi,od:exports_check(r,o,p,pi,od)
    return p

def selftest():
    need(SAMPLE_IDS==tuple(range(168)) and ARMS==('cnn','control','dynamic'),'Frozen protocol drift')
    need(tuple(RADII)==(4.,8.,16.,32.,64.,96.,128.,math.inf),'Radius grid drift')
    need('example_data_fixed' in BENCH and 'expanded' not in BENCH,'Wrong 168 benchmark')
    need('upper_midwest' not in str(SOURCE) and 'pacific_inland' not in str(SOURCE),'Source name invalid')
    print('[SELFTEST] PASS: 168 fixed benchmark IDs, original 8 radii, isolation scope; no files written')

def export(root,out,p):
    prov,pilot,bridge,cp,old,prior=p.preflight(root,out,require_new=True)
    tf,_,_,PhIREGANs=bridge.import_tf(root)
    tf.reset_default_graph()
    phire=PhIREGANs(data_type='phase8_original168_eval_readonly',
                    mu_sig=[pilot.MU.tolist(),pilot.SIG.tolist()])
    ds=tf.data.TFRecordDataset(str(root/BENCH))
    ds=ds.map(lambda example:phire._parse_train_(example,phire.mu_sig))
    item=ds.make_one_shot_iterator().get_next()
    found={};total=0
    c=tf.ConfigProto();c.gpu_options.allow_growth=True
    t0=time.perf_counter()
    with tf.Session(config=c) as sess:
        while True:
            try:idx,lr,hr=sess.run(item)
            except tf.errors.OutOfRangeError:break
            total+=1;idx=int(idx)
            need(idx in SAMPLE_IDS and idx not in found,
                 f'Unexpected/duplicate benchmark index {idx}')
            need(lr.shape==(100,100,2) and hr.shape==(500,500,2),
                 f'Unexpected benchmark field shape at {idx}: {lr.shape}/{hr.shape}')
            found[idx]=(lr[None].astype(np.float32),hr[None].astype(np.float32))
            if total%24==0: print(f'[EXPORT] {total:03d}/168 records checked',flush=True)
    need(total==168 and set(found)==set(SAMPLE_IDS),'Benchmark must have exactly indices 0..167')
    names=[]
    def writer(tmp):
        for i in SAMPLE_IDS:
            lr,hr=found[i]
            gt=pilot.speed_from_normalized(hr)
            for suffix,arr in [('lr',lr),('hr',hr),('gt_crop',gt)]:
                n=f's{i:04d}_{suffix}.npy';np.save(tmp/n,arr);names.append(n)
        put(tmp/'manifest.json',{'schema':SCHEMA+'-inputs','eval_indices':list(SAMPLE_IDS),
             'provenance':prov,'original_training_manifest_sha256':sha(old/'train'/'manifest.json'),
             'wtk_manifest_sha256':sha(root/BENCH),'expansion_source_sha256':sha(root/SRC),
             'script_sha256':sha(root/PARENT),'wrapper_sha256':sha(SOURCE),
             'tfrecord_size_bytes':(root/BENCH).stat().st_size,
             'artifact_hashes':pilot.hash_files(tmp,names),'pilot_training_indices_in_a_different_tfrecord':[0,1,2,3],
             'status':'EXPORTED_ORIGINAL_BENCHMARK_DEVELOPMENT_ONLY',
             'elapsed_seconds':round(time.perf_counter()-t0,6)})
    pilot.atomic(out/'eval_inputs',writer)
    print('[EXPORT] PASS: 168 canonical benchmark fields; no optimizer updates or checkpoint writes',flush=True)

def exports_check(root,out,p,pilot,old):
    d=out/'eval_inputs';need(d.is_dir(),'Run --mode export first')
    m=read(d/'manifest.json')
    need(m['schema']==SCHEMA+'-inputs' and m['eval_indices']==list(SAMPLE_IDS),'Wrong export indices')
    need(m['original_training_manifest_sha256']==sha(old/'train'/'manifest.json'),'Original pilot changed')
    need(m['wtk_manifest_sha256']==sha(root/BENCH),'168 benchmark file SHA changed')
    need(m['tfrecord_size_bytes']==(root/BENCH).stat().st_size,'Benchmark byte size changed')
    need(m['expansion_source_sha256']==sha(root/SRC),'Training builder source changed')
    need(m['script_sha256']==sha(root/PARENT),'Parent evaluator changed')
    need(m['wrapper_sha256']==sha(SOURCE),'Wrapper source changed')
    need(m['provenance']==metadata(root),'Benchmark identity changed')
    pilot.verify_files(d,m['artifact_hashes'])
    return m

def replay_check(root,out,p,pilot,bridge,cp,old):
    exports_check(root,out,p,pilot,old)
    # The parent's original replay validator checks the full eight-step fidelity,
    # the frozen checkpoints, all 168 prediction hashes and source lineage.
    return p.check_replay(root,out,pilot,bridge,cp,old)

def scored_dir(out,i):return out/'scores'/f's{i:03d}'
def locality_dir(out,i):return out/'locality_samples'/f's{i:03d}'
def radius_label(x):return 'inf' if math.isinf(x) else str(x)

def check_score_sample(out,i,pilot):
    s=scored_dir(out,i)
    need(s.is_dir(),f'Score sample {i} not completed')
    m=read(s/'manifest.json')
    need(m['schema']==SCHEMA+'-score-sample' and m['idx']==i,'Score metadata mismatch')
    need(m['input_manifest_sha256']==sha(out/'eval_inputs'/'manifest.json') and
         m['replay_manifest_sha256']==sha(out/'replay'/'manifest.json') and
         m['wrapper_sha256']==sha(SOURCE),'Score source lineage changed')
    pilot.verify_files(s,m['hashes'])
    return read(s/'results.json')

def score(root,out,p,backend,timeout,max_new):
    prov,pilot,bridge,cp,old,prior=p.preflight(root,out,require_new=False)
    m=replay_check(root,out,p,pilot,bridge,cp,old)
    timer,phase6,decoder,_,_=pilot.audit_modules(root,bridge)
    timer.check_backend(backend)
    (out/'scores').mkdir(parents=True,exist_ok=True)
    new_count=0
    for i in SAMPLE_IDS:
        if scored_dir(out,i).exists():
            check_score_sample(out,i,pilot)
            print(f'[SCORE] {i+1:03d}/168 verified cached',flush=True)
            continue
        if max_new is not None and new_count>=max_new:
            print(f'[PARTIAL-SCORE] Added {new_count} fields this invocation; resume with --mode score',flush=True)
            return
        gt=np.load(out/'eval_inputs'/f's{i:04d}_gt_crop.npy',allow_pickle=False)
        fields={'gt':gt}
        for arm in ARMS:
            fields[arm]=np.load(out/'replay'/'predictions'/f's{i:04d}_{arm}.npy',allow_pickle=False)
        need(all(f.shape==(160,160) and np.isfinite(f).all() for f in fields.values()),
             f'Score field shape/nonfinite sample {i}')
        def writer(tmp):
            (tmp/'vti').mkdir();(tmp/'pd').mkdir()
            diagrams={};names=[];timings={}
            for arm,field in fields.items():
                tag=arm
                timer.write_vti(tmp/'vti'/f'{tag}.vti',field)
                timings[arm]=timer.run_ttk(backend,tmp,tag,timeout)
                diagrams[arm]=phase6.parse(decoder.read_pd,tmp/'pd'/f'{tag}_pd_port_0.vtu')
                need(all(len(diagrams[arm][dim])<=5000 for dim in (0,1)),
                     f'Excess PD pairs sample={i}, arm={arm}')
                names.extend([f'vti/{tag}.vti',f'pd/{tag}_pd_port_0.vtu'])
            rows={}
            for arm in ARMS:
                result=phase6.distance(diagrams['gt'],diagrams[arm])
                met=m['evaluation_metrics'][str(i)][arm]
                rows[arm]={**result,**met}
                need(all(math.isfinite(float(v)) for v in rows[arm].values()),
                     f'Nonfinite metrics sample {i} {arm}')
            put(tmp/'results.json',{'sample':i,'arms':rows,'ttk_time':timings,
                'dynamic_minus_control_W22':rows['dynamic']['W22']-rows['control']['W22']})
            names.append('results.json')
            put(tmp/'manifest.json',{'schema':SCHEMA+'-score-sample','idx':i,
                'input_manifest_sha256':sha(out/'eval_inputs'/'manifest.json'),
                'replay_manifest_sha256':sha(out/'replay'/'manifest.json'),
                'wrapper_sha256':sha(SOURCE),'hashes':pilot.hash_files(tmp,names)})
        pilot.atomic(scored_dir(out,i),writer)
        rows=check_score_sample(out,i,pilot)
        new_count+=1
        print(f'[SCORE] {i+1:03d}/168 control={rows["arms"]["control"]["W22"]:.5f} '
              f'dynamic={rows["arms"]["dynamic"]["W22"]:.5f}',flush=True)
    print('[SCORE] PASS: 168 original benchmark fields, 168*4 fresh TTK diagrams; '
          'no locked regions, no training',flush=True)

def verify_score(root,out,p):
    prov,pilot,bridge,cp,old,prior=p.preflight(root,out,require_new=False)
    replay_check(root,out,p,pilot,bridge,cp,old)
    _,phase6,decoder,_,_=pilot.audit_modules(root,bridge)
    for i in SAMPLE_IDS:
        r=check_score_sample(out,i,pilot)
        root_s=scored_dir(out,i)
        diagrams={a:phase6.parse(decoder.read_pd,root_s/'pd'/f'{a}_pd_port_0.vtu')
                  for a in ('gt',*ARMS)}
        need(r['sample']==i,'Score sample order drift')
        for arm in ARMS:
            new=phase6.distance(diagrams['gt'],diagrams[arm])
            for key in ('dB','W2inf','W22'):
                need(math.isclose(new[key],r['arms'][arm][key],abs_tol=1e-6,rel_tol=1e-8),
                     f'Recomputed {key} drift sample={i} {arm}')
        if i%24==23:print(f'[VERIFY-SCORE] {i+1:03d}/168',flush=True)
    print('[VERIFY-SCORE] PASS: all 168*3 canonical GUDHI distances independently recomputed '
          'from immutable VTUs',flush=True)

def load_solver(root):
    need(sha(root/SOLVER)==SOLVER_SHA,'Frozen original Phase5E solver SHA drift')
    solver=load(root/SOLVER,'phase8_original168_original_phase5e')
    need(tuple(solver.RADII)==RADII,'Original Phase5E radii drift')
    need(tuple(solver.DIMS)==(0,1),'Phase5E dimensions drift')
    need(solver.TOL==1e-10,'Phase5E zero-persistence tolerance drift')
    return solver

def compute_locality(solver,s,score_row):
    files={a:s/'pd'/f'{a}_pd_port_0.vtu' for a in ('gt',*ARMS)}
    features={}
    skipped={}
    for arm,path in files.items():
        features[arm],skipped[arm]=solver.read_features(path)
    per_arm={}
    for arm in ARMS:
        preps={dim:solver.SparsePrepared(features['gt'][dim],features[arm][dim]) for dim in (0,1)}
        results={}
        for radius in RADII:
            dim_result={str(dim):preps[dim].solve(radius) for dim in (0,1)}
            total=sum(x['total_sq'] for x in dim_result.values())
            gtot=sum(x['gt_total_persistence'] for x in dim_result.values())
            stot=sum(x['sr_total_persistence'] for x in dim_result.values())
            gm=sum(x['gt_matched_persistence'] for x in dim_result.values())
            sm=sum(x['sr_matched_persistence'] for x in dim_result.values())
            results[radius_label(radius)]={
                'w22':math.sqrt(max(0.,total)),
                'gt_persistence_coverage':gm/gtot if gtot>0 else None,
                'sr_persistence_coverage':sm/stot if stot>0 else None,
                'gt_to_diagonal':sum(x['gt_to_diagonal'] for x in dim_result.values()),
                'diagonal_to_sr':sum(x['diagonal_to_sr'] for x in dim_result.values()),
                'real_real':sum(x['real_real'] for x in dim_result.values()),
                'dimensions':dim_result,
            }
        expected=score_row['arms'][arm]['W22']
        observed=results['inf']['w22']
        need(math.isclose(expected,observed,rel_tol=1e-6,abs_tol=1e-5),
             f'Original Phase5E r_inf mismatch sample={score_row["sample"]} '
             f'{arm}: {observed} vs canonical {expected}')
        per_arm[arm]=results
    return {'sample':score_row['sample'],'arms':per_arm,'reader_skipped_zeros':skipped,
            'solver':'original Phase5E SparsePrepared, unchanged'}

def locality(root,out,p,max_new):
    prov,pilot,bridge,cp,old,prior=p.preflight(root,out,require_new=False)
    replay_check(root,out,p,pilot,bridge,cp,old)
    solver=load_solver(root)
    (out/'locality_samples').mkdir(parents=True,exist_ok=True)
    new_count=0
    for i in SAMPLE_IDS:
        row=check_score_sample(out,i,pilot)
        dest=locality_dir(out,i)
        if dest.is_dir():
            check_locality_sample(out,i,pilot)
            print(f'[LOCALITY] {i+1:03d}/168 verified cached',flush=True)
            continue
        if max_new is not None and new_count>=max_new:
            print(f'[PARTIAL-LOCALITY] Added {new_count} fields this invocation; resume with --mode locality',flush=True)
            return
        def writer(tmp):
            result=compute_locality(solver,scored_dir(out,i),row)
            put(tmp/'result.json',result)
            put(tmp/'manifest.json',{'schema':SCHEMA+'-locality-sample',
                 'idx':i,'source_score_manifest_sha256':sha(scored_dir(out,i)/'manifest.json'),
                 'solver_sha256':sha(root/SOLVER),'wrapper_sha256':sha(SOURCE),
                 'hashes':pilot.hash_files(tmp,['result.json'])})
        pilot.atomic(dest,writer)
        check_locality_sample(out,i,pilot)
        new_count+=1
        print(f'[LOCALITY] {i+1:03d}/168 Phase5E eight radii PASS',flush=True)
    print('[LOCALITY] PASS: all 168 original development benchmark fields, '
          '8 radii x 3 model arms x 2 dimensions, no locked regions',flush=True)

def check_locality_sample(out,i,pilot):
    s=locality_dir(out,i)
    need(s.is_dir(),f'Locality field {i} not completed')
    m=read(s/'manifest.json')
    need(m['schema']==SCHEMA+'-locality-sample' and m['idx']==i,'Locality metadata drift')
    need(m['source_score_manifest_sha256']==sha(scored_dir(out,i)/'manifest.json'),
         'Source sample score changed')
    need(m['solver_sha256']==sha(SOURCE.parent/SOLVER),'Phase5E source changed')
    need(m['wrapper_sha256']==sha(SOURCE),'Wrapper source changed')
    pilot.verify_files(s,m['hashes'])
    return read(s/'result.json')

def verify_locality(root,out,p):
    prov,pilot,bridge,cp,old,prior=p.preflight(root,out,require_new=False)
    replay_check(root,out,p,pilot,bridge,cp,old)
    solver=load_solver(root)
    for i in SAMPLE_IDS:
        row=check_score_sample(out,i,pilot)
        previous=check_locality_sample(out,i,pilot)
        actual=compute_locality(solver,scored_dir(out,i),row)
        def near_struct(x,y):
            if isinstance(x,dict) and isinstance(y,dict):
                return x.keys()==y.keys() and all(near_struct(x[k],y[k]) for k in x)
            if isinstance(x,list) and isinstance(y,list):
                return len(x)==len(y) and all(near_struct(a,b) for a,b in zip(x,y))
            if isinstance(x,(int,float)) and not isinstance(x,bool) and isinstance(y,(int,float)) and not isinstance(y,bool):
                return math.isclose(x,y,rel_tol=1e-9,abs_tol=1e-6)
            return x==y
        need(near_struct(previous,actual),
             f'Independent locality re-evaluation differs at sample {i}')
        if i%24==23:print(f'[VERIFY-LOCALITY] {i+1:03d}/168',flush=True)
    print('[VERIFY-LOCALITY] PASS: original Phase5E recomputed for all 168 fields; '
          '8 radii 3 arms 2 dimensions, strict source/hash checks',flush=True)

def report(root,out,p):
    prov,pilot,bridge,cp,old,prior=p.preflight(root,out,require_new=False)
    replay=replay_check(root,out,p,pilot,bridge,cp,old)
    scores=[check_score_sample(out,i,pilot) for i in SAMPLE_IDS]
    locality_rows=[check_locality_sample(out,i,pilot) for i in SAMPLE_IDS]
    def metric(arm,key):return float(mean(x['arms'][arm][key] for x in scores))
    metrics=('W22','W2inf','dB','L_uv','L_grad','speed_crop_MAE_mps')
    summary={arm:{metric:metric(arm,metric) for metric in metrics} for arm in ARMS}
    loc={}
    for arm in ARMS:
        loc[arm]={}
        for k in ('4.0','8.0','16.0','32.0','64.0','96.0','128.0','inf'):
            r=[row['arms'][arm][k] for row in locality_rows]
            loc[arm][k]={
                'mean_w22':float(mean(x['w22'] for x in r)),
                'mean_gt_coverage':float(mean(x['gt_persistence_coverage'] for x in r if x['gt_persistence_coverage'] is not None)),
                'mean_sr_coverage':float(mean(x['sr_persistence_coverage'] for x in r if x['sr_persistence_coverage'] is not None)),
                'mean_gt_to_diagonal':float(mean(x['gt_to_diagonal'] for x in r)),
                'mean_diagonal_to_sr':float(mean(x['diagonal_to_sr'] for x in r)),
            }
        need(math.isclose(loc[arm]['inf']['mean_w22'],summary[arm]['W22'],rel_tol=1e-7,abs_tol=1e-5),
             'Mean infinity W22 mismatch')
    paired=[x['arms']['dynamic']['W22']-x['arms']['control']['W22'] for x in scores]
    blocks=[]
    for d in range(7):
        ix=range(d*24,(d+1)*24)
        block=[paired[i] for i in ix]
        blocks.append({'day':d+1,'n_fields':24,
                       'dynamic_minus_control_W22':float(mean(block)),
                       'dynamic_better_count':sum(x<0 for x in block)})
    summary_data={'schema':SCHEMA+'-report','sample_count':168,
         'training_updates_per_arm':8,'training_unique_fields':4,
         'evaluation':'original 168 Mid-Atlantic benchmark (previously used for selection)',
         'means':summary,'spatial_locality':loc,
         'paired':{'n_dynamic_W22_better':sum(x<0 for x in paired),
                   'n_dynamic_W22_worse':sum(x>0 for x in paired),
                   'mean_dynamic_minus_control_W22':float(mean(paired)),
                   'per_day':blocks},
         'limitations':['NOT comparable as matched-training-scale experiment to 2688-sample/3-epoch manuscript arms',
             'Eight updates on FOUR training fields only',
             'Benchmark used in prior model/hyperparameter selection; not an external held-out test',
             'Time-adjacent hours are correlated; seven daily blocks, not 168 independent events',
             'Top-left 160x160 scalar-speed crop for PD and locality, not full 500x500',
             'L_uv and L_grad refer to normalized model loss; speed_crop_MAE_mps is 160-crop',
             'No Phase7 regional test fields were loaded; locked geographies untouched',
             'Pretraining independence not verified'],
         'source_sha256':sha(SOURCE),'benchmark_tfrecord_sha256':sha(root/BENCH),
         'checkpoint_hashes':replay['checkpoint_hashes'],
         'scores_manifest_sha256':{str(i):sha(scored_dir(out,i)/'manifest.json') for i in SAMPLE_IDS},
         'locality_manifest_sha256':{str(i):sha(locality_dir(out,i)/'manifest.json') for i in SAMPLE_IDS}}
    dest=out/'report';need(not dest.exists(),'Report already published; never overwrite')
    lines=['# Phase 8 — Original Mid-Atlantic 168-field benchmark: eight-step pilot', '',
        '**DEVELOPMENT COMPARISON, NOT MANUSCRIPT-SCALE TRAINING OR EXTERNAL TEST.**', '',
        'Two independent Adam arms were restored and replayed from the verified pilot:',
        '`L_uv + 0.05 L_grad` versus `L_uv + 0.05 L_grad + 1e-5 * W22²`.',
        'Only 4 training fields, 8 optimizer updates per arm. All 168 ORIGINAL benchmark fields were evaluation-only.', '',
        '| Arm | PD W22 | W2inf | dB | L_uv | L_grad | Crop speed MAE m/s |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        a=summary[arm]
        lines.append('| '+arm+' | '+' | '.join(f'{a[k]:.6f}' for k in metrics)+' |')
    lines+=['','| Arm | r4 W22 | r16 W22 | r32 W22 | r∞ W22 | r16 GT coverage | r16 SR coverage |',
       '|---|---:|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        x=loc[arm]
        lines.append(f'| {arm} | {x["4.0"]["mean_w22"]:.6f} | '
           f'{x["16.0"]["mean_w22"]:.6f} | {x["32.0"]["mean_w22"]:.6f} | '
           f'{x["inf"]["mean_w22"]:.6f} | '
           f'{x["16.0"]["mean_gt_coverage"]:.6f} | '
           f'{x["16.0"]["mean_sr_coverage"]:.6f} |')
    lines+=['',f'Dynamic improved unconstrained W22 on {sum(v<0 for v in paired)} of 168 fields.',
       f'Mean dynamic-control W22 difference: {mean(paired):+.6f}.','',
       '| Consecutive 24-hour block | Mean dynamic-control W22 | Dynamic improved hours |',
       '|---|---:|---:|']
    for block in blocks:
        lines.append(f'| {block["day"]} | {block["dynamic_minus_control_W22"]:+.6f} | '
                     f'{block["dynamic_better_count"]}/24 |')
    lines+=['','## Scope cautions','',*['- '+v for v in summary_data['limitations']],'']
    def writer(tmp):
        put(tmp/'summary.json',summary_data)
        (tmp/'report.md').write_text('\n'.join(lines))
        with (tmp/'per_field.csv').open('w',newline='') as f:
            cols=['sample','arm',*metrics,'r4_w22','r16_w22','r32_w22','rinf_w22',
                  'r16_gt_coverage','r16_sr_coverage']
            w=csv.DictWriter(f,fieldnames=cols);w.writeheader()
            for i in SAMPLE_IDS:
                for arm in ARMS:
                    row={'sample':i,'arm':arm,**scores[i]['arms'][arm]}
                    q=locality_rows[i]['arms'][arm]
                    row.update({'r4_w22':q['4.0']['w22'],'r16_w22':q['16.0']['w22'],
                     'r32_w22':q['32.0']['w22'],'rinf_w22':q['inf']['w22'],
                     'r16_gt_coverage':q['16.0']['gt_persistence_coverage'],
                     'r16_sr_coverage':q['16.0']['sr_persistence_coverage']})
                    w.writerow({col:row[col] for col in cols})
        put(tmp/'manifest.json',{'schema':SCHEMA+'-report-manifest',
            'wrapper_sha256':sha(SOURCE),'input_manifest_sha256':sha(out/'eval_inputs'/'manifest.json'),
            'replay_manifest_sha256':sha(out/'replay'/'manifest.json'),
            'artifacts':pilot.hash_files(tmp,['summary.json','report.md','per_field.csv'])})
    pilot.atomic(dest,writer)
    print('[REPORT] PASS: 168 original benchmark fields, seven daily blocks; '
          'read '+str(dest/'report.md'),flush=True)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode',required=True,choices=('selftest','preflight','export','replay','score',
         'verify-score','locality','verify-locality','report'))
    ap.add_argument('--repo-root',type=Path,default=Path.cwd())
    ap.add_argument('--out-dir',type=Path,default=Path(OUT))
    ap.add_argument('--ttk-backend',choices=('docker','host'),default='docker')
    ap.add_argument('--ttk-timeout',type=int,default=180)
    ap.add_argument('--max-new-samples',type=int,default=None,
                    help='Score/locality only: stop after this many NEW samples; stages remain resumable')
    a=ap.parse_args()
    need(a.max_new_samples is None or (a.max_new_samples>0 and a.mode in ('score','locality')),
         '--max-new-samples only allowed for score/locality, and must be positive')
    if a.mode=='selftest':selftest();return
    root=a.repo_root.resolve();out=(root/a.out_dir).resolve()
    need(root==SOURCE.parent,'Install script in ~/PhIRE repository root, not elsewhere')
    p=install(root)
    if a.mode=='preflight':p.preflight(root,out,require_new=True)
    elif a.mode=='export':export(root,out,p)
    elif a.mode=='replay':
        prov,pilot,bridge,cp,old,prior=p.preflight(root,out,require_new=False)
        p.replay(root,out)
    elif a.mode=='score':score(root,out,p,a.ttk_backend,a.ttk_timeout,a.max_new_samples)
    elif a.mode=='verify-score':verify_score(root,out,p)
    elif a.mode=='locality':locality(root,out,p,a.max_new_samples)
    elif a.mode=='verify-locality':verify_locality(root,out,p)
    elif a.mode=='report':report(root,out,p)

if __name__=='__main__':
    try:main()
    except Exception as ex:
        print(f'[PHASE8-ORIGINAL168] ERROR {type(ex).__name__}: {ex}',file=sys.stderr,flush=True)
        sys.exit(1)
