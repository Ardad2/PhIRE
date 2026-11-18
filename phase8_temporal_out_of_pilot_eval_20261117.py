#!/usr/bin/env python3
"""Phase 8, development-only temporal separation evaluation of the 8-step pilot.

This is NOT an independent test of generalization: the evaluation examples come
from the SAME 2688-record *training pool*, but were not used for this pilot's
8 optimizer steps. They belong to distinct seasonal WTK source windows.

Sequence from PhIRE root:
    --mode selftest
    --mode preflight
    --mode export
    --mode replay                 # native TF; replays archived fresh-matching feeds
    --mode score --ttk-backend docker  # gudhi-audit Python
    --mode verify --ttk-backend docker # gudhi-audit Python

Replay never recomputes matchings: it uses the archived matching generated fresh
at each of the 8 original steps, and fails if predictions differ. This is a
reproducibility replay followed by development-only cross-window evaluation,
not a new experiment with additional optimizer updates.

Never modifies originals, trained model checkpoints, or 168-field benchmark.
Creates only a new, collision-safe output directory. Failures are kept as
explicitly incomplete staging directories. No checkpoint/weight file saved.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

import importlib.util

PILOT = 'phase8_tf1_multisample_rematch_pilot_20261116.py'
ONE_STEP = 'phase8_tf1_one_step_optimizer_gate_20261116.py'
GENERATOR = 'scripts/build_wind_mrhr_expanded_dataset_2688.py'
TFRECORD = 'example_data_topology_expanded_2688/wind_MR-HR.tfrecord'
MANIFEST = 'example_data_topology_expanded_2688/manifest.csv'
OLD_OUT = 'ttk_runs_fixed/topology_finetuning/phase8_multisample_rematch_20261116_r01'
NEW_OUT = 'ttk_runs_fixed/topology_finetuning/phase8_cross_season_out_of_pilot_20261117_r01'
TRAIN = (0, 1, 2, 3)
EVAL = (672, 1344, 2016, 2520)
INDEX_EXPECTATIONS = {
    0: ('winter_1',336), 1:('winter_1',337), 2:('winter_1',338),3:('winter_1',339),
    672:('spring_1',2160), 1344:('summer_1',4344),
    2016:('fall_1',6552), 2520:('fall_4',7056),
}
WINDOWS = (
    ('winter',336), ('winter',504), ('winter',672),('winter',840),
    ('spring',2160),('spring',2328),('spring',2496),('spring',2664),
    ('summer',4344),('summer',4512),('summer',4680),('summer',4848),
    ('fall',6552),('fall',6720),('fall',6888),('fall',7056),
)
LR = 1e-6
LAMBDA = 1e-5
MAX_REPLAY_FIELD_ABS_DIFF = 0.005  # m/s; fail closed rather than silently drift
SCHEMA = 'phase8-cross-season-out-of-pilot-r01'


def need(test, why):
    if not test:
        raise RuntimeError(why)


def load_module(path, name):
    spec=importlib.util.spec_from_file_location(name, str(path))
    need(spec is not None and spec.loader is not None, f'Missing module {path}')
    m=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):
            h.update(b)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj,sort_keys=True,indent=2,allow_nan=False)+'\n')


def validate_metadata(root):
    path=root/MANIFEST
    need(path.is_file(), f'Missing original WTK provenance CSV: {path}')
    need((root/GENERATOR).is_file(),f'Missing original expansion source: {GENERATOR}')
    with path.open(newline='') as f:
        records=list(csv.DictReader(f))
    need(len(records)==2688, f'Manifest contains {len(records)}, not 2688 fields')
    seen=set()
    for i,row in enumerate(records):
        rid=int(row['sample_idx'])
        wtk=int(row['wtk_time_index'])
        name=row['season']
        need(rid==i, f'CSV row ordering does not match index at row {i}')
        need(wtk not in seen and wtk >=168,f'Duplicate/benchmark source time {wtk}')
        seen.add(wtk)
        label,start=WINDOWS[i//168]
        need(wtk==start+i%168 and name==f'{label}_{i//168%4+1}',
             f'Unexpected window lineage at row {i}: {name}, {wtk}')
    for i,(season,wtk) in INDEX_EXPECTATIONS.items():
        r=records[i]
        need((r['season'],int(r['wtk_time_index']))==(season,wtk),
             f'Lineage mismatch for requested index {i}: {r}')
    need(set(TRAIN).isdisjoint(EVAL),'Evaluation overlaps optimizer examples')
    return {str(i):{'sample_idx':i,'season':records[i]['season'],
                     'wtk_time_index':int(records[i]['wtk_time_index']),
                     'timestamp':records[i]['timestamp']}
            for i in (*TRAIN,*EVAL)}


def inspect_prior(root):
    original=root/OLD_OUT
    need((original/'train'/'manifest.json').is_file(), 'Verified 8-step run missing')
    pilot=load_module(root/PILOT,'phase8_development_pilot')
    bridge,cp=pilot.check_root(root)
    pilot.cache_check(root,original,bridge)
    train=read_json(original/'train'/'manifest.json')
    need(train['schema']==pilot.SCHEMA+'-train','Wrong old training schema')
    need(train['order']==[0,1,2,3,0,1,2,3], 'Archived update order changed')
    need(train['n_steps_per_arm']==8 and train['rounds']==2,'Archived run is not 8-step pilot')
    need(math.isclose(float(train['learning_rate']),LR,rel_tol=1e-12), 'Archived LR changed')
    need(math.isclose(float(train['lambda_dynamic_W22_squared']),LAMBDA,rel_tol=1e-12),
         'Archived dynamic coefficient changed')
    need(math.isclose(float(train['lambda_grad']),0.05,rel_tol=1e-12),'Archived grad coefficient changed')
    need(train['no_heldout_fields'] and train['no_checkpoint_writes'], 'Wrong prior-run status')
    pilot.require_hashes(train['source_hashes'])
    pilot.require_hashes(train['original_checkpoint_hashes'])
    pilot.verify_files(original/'train', train['output_hashes'])
    need(bridge.filehashes(cp)==train['original_checkpoint_hashes'],'Original pretrained checkpoint changed')
    for n,idx in enumerate(train['order']):
        p=original/'train'/'steps'/f'iter_{n:03d}_s{idx:04d}'
        row=train['rows'][n]
        need(row['sample_index']==idx and row['step']==n,'Prior row order mismatch')
        need(sha(p/'match.json')==row['match_sha256'] and
             sha(p/'eval.json')==row['eval_sha256'], 'Prior matching/eval evidence drift')
    return pilot,bridge,cp,original,train


def preflight(root,out,require_new=True):
    need(Path(__file__).resolve().parent==root,'Copy this script to PhIRE repository root')
    need((root/TFRECORD).is_file(),'Missing canonical 2688 training TFRecord')
    if require_new:
        need(not out.exists(),f'Output already exists: {out}')
    provenance=validate_metadata(root)
    pilot,bridge,cp,old,train=inspect_prior(root)
    print('[LINEAGE] PASS: 2688 unique WTK hourly time indices in 16 seasonal windows',flush=True)
    print('[PILOT] PASS: original eight fresh-rematched updates and source hashes intact',flush=True)
    for i in EVAL:
        p=provenance[str(i)]
        print(f'[EVAL-PLAN] idx={i} season={p["season"]} WTK={p["wtk_time_index"]}',flush=True)
    print('[SCOPE] Out-of-pilot seasonal TRAIN-POOL fields, not independent benchmark/test',flush=True)
    return provenance,pilot,bridge,cp,old,train


def selftest():
    need(len(set(EVAL))==4 and set(EVAL).isdisjoint(TRAIN),'Evaluation overlaps training steps')
    need(len(WINDOWS)==16 and len(INDEX_EXPECTATIONS)==8,'Lineage constants invalid')
    for i,(s,t) in INDEX_EXPECTATIONS.items():
        season,start=WINDOWS[i//168]
        need(s==f'{season}_{i//168%4+1}' and t==start+i%168,'WTK map formula invalid')
    need(math.isclose(LR,1e-6) and math.isclose(LAMBDA,1e-5),'Run settings drift')
    print('[SELFTEST] PASS: season mapping, no overlap, original optimizer configuration',flush=True)


def export_fields(root,out):
    provenance,pilot,bridge,cp,old,prior=preflight(root,out,require_new=True)
    tf,_,_,PhIREGANs=bridge.import_tf(root)
    tf.reset_default_graph()
    phire=PhIREGANs(data_type='phase8_out_of_pilot_readonly',
                   mu_sig=[pilot.MU.tolist(),pilot.SIG.tolist()])
    ds=tf.data.TFRecordDataset(str(root/TFRECORD))
    ds=ds.map(lambda example:phire._parse_train_(example,phire.mu_sig))
    iterator=ds.make_one_shot_iterator().get_next()
    found={}
    t0=time.perf_counter()
    config=tf.ConfigProto();config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        while len(found)<len(EVAL):
            try:
                i,lr,hr=sess.run(iterator)
            except tf.errors.OutOfRangeError as exc:
                raise RuntimeError('Not all evaluation indices are in TFRecord') from exc
            i=int(i)
            if i not in EVAL:continue
            need(i not in found,f'Duplicate TFRecord index {i}')
            need(lr.shape==(100,100,2) and hr.shape==(500,500,2),
                 f'Unexpected TFRecord dimensions idx={i}')
            found[i]=(lr[None].astype(np.float32),hr[None].astype(np.float32))
            print(f'[EXPORT] found sample={i} ({len(found)}/4)',flush=True)
    members=[]
    def writer(temp):
        for i in EVAL:
            lr,hr=found[i]
            for suffix,array in [('lr',lr),('hr',hr),('gt_crop',pilot.speed_from_normalized(hr))]:
                p=f's{i:04d}_{suffix}.npy'
                np.save(temp/p,array)
                members.append(p)
        write_json(temp/'manifest.json',{
            'schema':SCHEMA+'-inputs','eval_indices':list(EVAL),
            'provenance':{str(i):provenance[str(i)] for i in EVAL},
            'original_training_manifest_sha256':sha(old/'train'/'manifest.json'),
            'wtk_manifest_sha256':sha(root/MANIFEST),
            'expansion_source_sha256':sha(root/GENERATOR),
            'script_sha256':sha(Path(__file__).resolve()),
            'tfrecord_size_bytes':(root/TFRECORD).stat().st_size,
            'artifact_hashes':pilot.hash_files(temp,members),
            'training_indices_omitted':list(TRAIN),
            'status':'EXPORTED_NOT_EVALUATED',
            'elapsed_seconds':round(time.perf_counter()-t0,6)})
    pilot.atomic(out/'eval_inputs',writer)
    print(f'[EXPORT] PASS: 4 out-of-pilot fields; {out/"eval_inputs"}',flush=True)


def check_exports(root,out,pilot,old):
    d=out/'eval_inputs'
    need(d.is_dir(),'Run --mode export first')
    m=read_json(d/'manifest.json')
    need(m['schema']==SCHEMA+'-inputs' and m['eval_indices']==list(EVAL),
         'Invalid export manifest')
    need(m['original_training_manifest_sha256']==sha(old/'train'/'manifest.json'),
         'Original training manifest changed')
    need(m['wtk_manifest_sha256']==sha(root/MANIFEST),'WTK metadata changed')
    need(m['expansion_source_sha256']==sha(root/GENERATOR),'Generator source changed')
    need(m['script_sha256']==sha(Path(__file__).resolve()),'Evaluation script changed')
    need(m['tfrecord_size_bytes']==(root/TFRECORD).stat().st_size,'Training source size changed')
    pilot.verify_files(d,m['artifact_hashes'])
    metadata=validate_metadata(root)
    for i in EVAL:
        need(m['provenance'][str(i)]==metadata[str(i)],'Provenance record changed')
    return m


def eval_input(out,i,pilot):
    d=out/'eval_inputs'
    lr=np.load(d/f's{i:04d}_lr.npy',allow_pickle=False)
    hr=np.load(d/f's{i:04d}_hr.npy',allow_pickle=False)
    gt=np.load(d/f's{i:04d}_gt_crop.npy',allow_pickle=False)
    need(lr.shape==(1,100,100,2) and hr.shape==(1,500,500,2),f'Bad eval input {i}')
    need(np.array_equal(gt,pilot.speed_from_normalized(hr)), f'GT speed changed {i}')
    return lr,hr,gt


def crop_eval(sess,fetch,xl,xh,lr,hr,gt):
    results=sess.run({'crop':fetch['crop'], 'L_uv':fetch['L_uv'],
                      'L_grad':fetch['L_grad']},feed_dict={xl:lr,xh:hr})
    crop=np.asarray(results['crop'][0],dtype=np.float32)
    need(crop.shape==(160,160) and np.isfinite(crop).all(),'Invalid prediction crop')
    return crop, {'L_uv':float(results['L_uv']),
                  'L_grad':float(results['L_grad']),
                  'speed_crop_MAE_mps':float(np.mean(np.abs(crop-gt),dtype=np.float64))}


def replay(root,out):
    prov,pilot,bridge,cp,old,prior=preflight(root,out,require_new=False)
    check_exports(root,out,pilot,old)
    need(not (out/'replay').exists(),'Replay already exists; no overwrite')
    one=load_module(root/ONE_STEP,'phase8_replay_graph')
    tf,xl,xh,ph,fetch,trains,saver,init,variables=one.make_tf_graph(bridge,root,LR,LAMBDA)
    start=time.perf_counter()
    original_cp=bridge.filehashes(cp)
    config=tf.ConfigProto();config.gpu_options.allow_growth=True
    def writer(temp):
        (temp/'predictions').mkdir()
        stats={}
        fidelities=[]
        images={}
        with tf.Session(config=config) as control,tf.Session(config=config) as dynamic:
            for sess in (control,dynamic):
                sess.run(init);saver.restore(sess,str(root/pilot.CHECKPOINT))
            def model_digest(sess):
                h=hashlib.sha256()
                for weight in sess.run(variables):h.update(np.asarray(weight).tobytes())
                return h.hexdigest()
            init_ctrl=model_digest(control)
            need(init_ctrl==model_digest(dynamic), 'Arms start from different checkpoint values')
            for i in EVAL:
                lr,hr,gt=eval_input(out,i,pilot)
                image,met=crop_eval(control,fetch,xl,xh,lr,hr,gt)
                images[i]={'cnn':image}
                stats[str(i)]={'cnn':met}
            for step,idx in enumerate((*TRAIN,*TRAIN)):
                lr,hr,_=pilot.get_sample(old,idx)
                feed={xl:lr,xh:hr}
                archived=old/'train'/'steps'/f'iter_{step:03d}_s{idx:04d}'
                before=np.asarray(dynamic.run(fetch['crop'],feed_dict=feed)[0],dtype=np.float32)
                prev=np.load(archived/'dynamic_before.npy',allow_pickle=False)
                pre_diff=float(np.max(np.abs(before-prev)))
                need(pre_diff<=MAX_REPLAY_FIELD_ABS_DIFF,
                     f'Step {step}: pre-update prediction differs from archived: {pre_diff}')
                with np.load(archived/'feed.npz',allow_pickle=False) as data:
                    pairs={k:np.array(data[k]) for k in data.files}
                fd=dict(feed)
                for k,p in zip(('birth_yx','death_yx','target','mode','mask'),ph[:5]):
                    fd[p]=pairs[k]
                fd[ph[5]]=float(pairs['gt_diagonal_cost'])
                J=float(dynamic.run(fetch['J_frozen'],feed_dict=fd))
                expected=float(pairs['expected_squared_cost'])
                need(math.isclose(J,expected,rel_tol=3e-4,abs_tol=1e-2),
                     f'Step {step}: replay loss {J} diverges from archived matching {expected}')
                control.run(trains['baseline'],feed_dict=feed)
                dynamic.run(trains['dynamic'],feed_dict=fd)
                b=np.asarray(control.run(fetch['crop'],feed_dict=feed)[0],dtype=np.float32)
                d=np.asarray(dynamic.run(fetch['crop'],feed_dict=feed)[0],dtype=np.float32)
                b_diff=float(np.max(np.abs(b-np.load(archived/'baseline_after.npy',allow_pickle=False))))
                d_diff=float(np.max(np.abs(d-np.load(archived/'dynamic_after.npy',allow_pickle=False))))
                need(max(b_diff,d_diff)<=MAX_REPLAY_FIELD_ABS_DIFF,
                     f'Step {step}: replayed optimizer diverges (control={b_diff}, dynamic={d_diff})')
                historical=read_json(archived/'step.json')
                full_loss_diffs={}
                for arm,sess in [('control',control),('dynamic',dynamic)]:
                    present=sess.run({'L_uv':fetch['L_uv'],'L_grad':fetch['L_grad']},feed_dict=feed)
                    for metric in ('L_uv','L_grad'):
                        recorded=float(historical[f'{metric}_after_{arm}'])
                        actual=float(present[metric])
                        need(math.isclose(actual,recorded,rel_tol=5e-3,abs_tol=1e-4),
                             f'Step {step}: full-field {metric} {arm} deviates from archive: '
                             f'{actual} versus {recorded}')
                        full_loss_diffs[f'{arm}_{metric}']=abs(actual-recorded)
                fidelities.append({'step':step,'train_index':idx,
                    'max_pre_crop_mps':pre_diff,'max_control_crop_mps':b_diff,
                    'max_dynamic_crop_mps':d_diff,'tf_J':J,'matching_J':expected,
                    'full_loss_abs_diff':full_loss_diffs})
                print(f'[REPLAY {step+1}/8] sample={idx} pre_diff={pre_diff:.3g} '
                      f'control_diff={b_diff:.3g} dynamic_diff={d_diff:.3g}',flush=True)
            need(init_ctrl!=model_digest(control) and init_ctrl!=model_digest(dynamic),
                 'No parameter updates detected')
            for i in EVAL:
                lr,hr,gt=eval_input(out,i,pilot)
                for arm,sess in [('control',control),('dynamic',dynamic)]:
                    image,met=crop_eval(sess,fetch,xl,xh,lr,hr,gt)
                    images[i][arm]=image
                    stats[str(i)][arm]=met
        names=[]
        for i in EVAL:
            for arm in ('cnn','control','dynamic'):
                name=f'predictions/s{i:04d}_{arm}.npy'
                np.save(temp/name,images[i][arm]);names.append(name)
        need(bridge.filehashes(cp)==original_cp,'Frozen pretrained checkpoint changed')
        write_json(temp/'manifest.json',{
            'schema':SCHEMA+'-replay','status':'REPLAYED_EVALUATION_PENDING',
            'train_order':list((*TRAIN,*TRAIN)), 'eval_indices':list(EVAL),
            'lr':LR,'lambda_dynamic_squared_W22':LAMBDA,
            'n_optimizer_updates_per_arm':8,'no_model_checkpoint_saved':True,
            'original_train_manifest_sha256':sha(old/'train'/'manifest.json'),
            'eval_export_manifest_sha256':sha(out/'eval_inputs'/'manifest.json'),
            'source_sha256':sha(Path(__file__).resolve()),
            'checkpoint_hashes':original_cp,
            'replay_fidelity':fidelities,'evaluation_metrics':stats,
            'artifact_hashes':pilot.hash_files(temp,names),
            'elapsed_seconds':round(time.perf_counter()-start,6)})
    pilot.atomic(out/'replay',writer)
    print('[REPLAY] PASS: archived 8-step control/dynamic updates reproduced within '
          f'{MAX_REPLAY_FIELD_ABS_DIFF:g} m/s per crop; no weights saved',flush=True)


def check_replay(root,out,pilot,bridge,cp,old):
    check_exports(root,out,pilot,old)
    d=out/'replay'
    need(d.is_dir(),'Run --mode replay first')
    m=read_json(d/'manifest.json')
    need(m['schema']==SCHEMA+'-replay' and m['eval_indices']==list(EVAL),
         'Replay metadata invalid')
    need(m['train_order']==list((*TRAIN,*TRAIN)) and m['n_optimizer_updates_per_arm']==8,
         'Replay training protocol differs')
    need(m['no_model_checkpoint_saved'],'Replay weights were saved unexpectedly')
    need(m['source_sha256']==sha(Path(__file__).resolve()),'Replay source drift')
    need(m['original_train_manifest_sha256']==sha(old/'train'/'manifest.json'),
         'Upstream training changed')
    need(m['eval_export_manifest_sha256']==sha(out/'eval_inputs'/'manifest.json'),
         'Evaluation input changed')
    need(m['checkpoint_hashes']==bridge.filehashes(cp),'Checkpoint hashes drift')
    need(len(m['replay_fidelity'])==8,'Replay did not check every step')
    for row in m['replay_fidelity']:
        need(max(row['max_pre_crop_mps'],row['max_control_crop_mps'],
                 row['max_dynamic_crop_mps'])<=MAX_REPLAY_FIELD_ABS_DIFF,
             'Archived fidelity gate violated')
    pilot.verify_files(d,m['artifact_hashes'])
    return m


def score(root,out,backend,timeout):
    prov,pilot,bridge,cp,old,prior=preflight(root,out,require_new=False)
    replaym=check_replay(root,out,pilot,bridge,cp,old)
    need(not (out/'score').exists(),'Scoring output already exists; no overwrite')
    timer,phase6,decoder,_,_=pilot.audit_modules(root,bridge)
    timer.check_backend(backend)
    t0=time.perf_counter()
    def writer(temp):
        (temp/'vti').mkdir();(temp/'pd').mkdir()
        rows=[];members=[]
        for i in EVAL:
            _,_,gt=eval_input(out,i,pilot)
            fields={'gt':gt}
            for arm in ('cnn','control','dynamic'):
                fields[arm]=np.load(out/'replay'/'predictions'/f's{i:04d}_{arm}.npy',allow_pickle=False)
            pd_by_arm={}
            for arm,field in fields.items():
                need(field.shape==(160,160) and np.isfinite(field).all(),f'Bad field {arm} {i}')
                tag=f's{i:04d}_{arm}'
                timer.write_vti(temp/'vti'/f'{tag}.vti',field)
                timer.run_ttk(backend,temp,tag,timeout)
                pd_by_arm[arm]=phase6.parse(decoder.read_pd,temp/'pd'/f'{tag}_pd_port_0.vtu')
                need(all(len(pd_by_arm[arm][dim]) <= 5000 for dim in (0,1)),
                     f'Diagram size exceeded safety gate for idx={i} arm={arm}')
                members.extend([f'vti/{tag}.vti',f'pd/{tag}_pd_port_0.vtu'])
            per_arm={}
            for arm in ('cnn','control','dynamic'):
                dist=phase6.distance(pd_by_arm['gt'],pd_by_arm[arm])
                need(all(np.isfinite(v) and v>=0 for v in dist.values()),'Invalid GUDHI score')
                met=replaym['evaluation_metrics'][str(i)][arm]
                per_arm[arm]={**dist,**met}
            rows.append({'sample_idx':i,'provenance':prov[str(i)],'arms':per_arm,
                'dynamic_minus_control_W22':per_arm['dynamic']['W22']-per_arm['control']['W22']})
            print(f'[SCORE] idx={i} CNN={per_arm["cnn"]["W22"]:.6f} '
                  f'control={per_arm["control"]["W22"]:.6f} '
                  f'dynamic={per_arm["dynamic"]["W22"]:.6f}',flush=True)
        differences=[r['dynamic_minus_control_W22'] for r in rows]
        comparison={
            'sample_count':len(rows),'dynamic_better_count':sum(v<0 for v in differences),
            'dynamic_worse_count':sum(v>0 for v in differences),
            'mean_dynamic_minus_control_W22':float(np.mean(differences)),
            'mean_metrics':{arm:{name:float(np.mean([r['arms'][arm][name] for r in rows]))
                for name in ('dB','W2inf','W22','L_uv','L_grad','speed_crop_MAE_mps')}
                for arm in ('cnn','control','dynamic')},
            'no_claim_of_independent_generalization':True,
        }
        write_json(temp/'report.json',{'schema':SCHEMA+'-scores','rows':rows,
            'summary':comparison,'limitations':[
            'All four evaluation fields are from the expanded 2688 training pool, not external test.',
            'No optimizer updates used evaluation examples; however all share spatial crop.',
            'Pretrained CNN may have been developed on other historical data.',
            'One fixed four-field selection, eight optimizer steps; no significance claim.',
            'Training was faithfully replayed using archived, originally fresh per-step matchings.',
            'Scores use canonical Phase6 TTK/GUDHI H0,H1 and unpruned diagrams.',
            ]})
        lines=['# Phase 8 — Cross-season, out-of-pilot development evaluation','','',
            'Training steps: 0,1,2,3,0,1,2,3; same original checkpoint, two TF1 Adam sessions.','','',
            '**Not an independent test:** the 4 evaluation examples remain in the 2688-field training pool.','','',
            '| Index (season) | CNN W22 | Control W22 | Dynamic W22 | Dynamic–control |',
            '|---|---:|---:|---:|---:|']
        for r in rows:
            a=r['arms'];lines.append(f'| {r["sample_idx"]} ({r["provenance"]["season"]}) | '
                f'{a["cnn"]["W22"]:.6f} | {a["control"]["W22"]:.6f} | '
                f'{a["dynamic"]["W22"]:.6f} | {r["dynamic_minus_control_W22"]:+.6f} |')
        lines.extend(['','', 'Mean metrics (lower is better):','',
            '| Arm | W22 | L_uv | L_grad | Speed crop MAE (m/s) |',
            '|---|---:|---:|---:|---:|'])
        for arm in ('cnn','control','dynamic'):
            a=comparison['mean_metrics'][arm]
            lines.append(f'| {arm} | {a["W22"]:.6f} | {a["L_uv"]:.8f} | '
                         f'{a["L_grad"]:.6f} | {a["speed_crop_MAE_mps"]:.6f} |')
        lines.extend(['','',f'Dynamic W22 better on {comparison["dynamic_better_count"]}/4 fields.',
                      '', 'Audit: replayed all 8 original steps with predictions compared to frozen artifacts; '
                      'no model checkpoint written; all diagrams recomputed from outputs.', ''])
        (temp/'report.md').write_text('\n'.join(lines))
        write_json(temp/'manifest.json',{
            'schema':SCHEMA+'-manifest','source_sha256':sha(Path(__file__).resolve()),
            'prior_train_manifest_sha256':sha(old/'train'/'manifest.json'),
            'eval_input_manifest_sha256':sha(out/'eval_inputs'/'manifest.json'),
            'replay_manifest_sha256':sha(out/'replay'/'manifest.json'),
            'artifacts':pilot.hash_files(temp,[*members,'report.json','report.md']),
            'backend':backend,'elapsed_seconds':round(time.perf_counter()-t0,6),
            'status':'SCORED_PENDING_INDEPENDENT_VERIFICATION'})
    pilot.atomic(out/'score',writer)
    print(f'[SCORE] PASS: recomputed 4 GT + 12 CNN/control/dynamic diagrams; '
          f'{out/"score"}',flush=True)


def verify(root,out,backend):
    prov,pilot,bridge,cp,old,prior=preflight(root,out,require_new=False)
    replaym=check_replay(root,out,pilot,bridge,cp,old)
    p=out/'score'
    need(p.is_dir(),'Run --mode score first')
    m=read_json(p/'manifest.json')
    need(m['schema']==SCHEMA+'-manifest','Invalid score manifest')
    need(m['source_sha256']==sha(Path(__file__).resolve()),'Source changed')
    need(m['prior_train_manifest_sha256']==sha(old/'train'/'manifest.json'), 'Prior changed')
    need(m['eval_input_manifest_sha256']==sha(out/'eval_inputs'/'manifest.json'), 'Inputs changed')
    need(m['replay_manifest_sha256']==sha(out/'replay'/'manifest.json'), 'Replay changed')
    pilot.verify_files(p,m['artifacts'])
    _,phase6,decoder,_,_=pilot.audit_modules(root,bridge)
    report=read_json(p/'report.json')
    need(len(report['rows'])==4,'Expected four evaluation rows')
    for row,i in zip(report['rows'],EVAL):
        need(row['sample_idx']==i and row['provenance']==prov[str(i)],
             'Wrong source row or seasonal identity')
        gt=phase6.parse(decoder.read_pd,p/'pd'/f's{i:04d}_gt_pd_port_0.vtu')
        for arm in ('cnn','control','dynamic'):
            pred=phase6.parse(decoder.read_pd,p/'pd'/f's{i:04d}_{arm}_pd_port_0.vtu')
            expected=row['arms'][arm]
            got=phase6.distance(gt,pred)
            for metric in ('dB','W2inf','W22'):
                need(math.isclose(got[metric],expected[metric],rel_tol=1e-8,abs_tol=1e-6),
                     f'GUDHI score changed sample={i} arm={arm} {metric}')
            crop=np.load(out/'replay'/'predictions'/f's{i:04d}_{arm}.npy',allow_pickle=False)
            reference=np.load(out/'eval_inputs'/f's{i:04d}_gt_crop.npy',allow_pickle=False)
            true_mae=float(np.mean(np.abs(crop-reference),dtype=np.float64))
            need(math.isclose(true_mae,expected['speed_crop_MAE_mps'],rel_tol=1e-9,abs_tol=1e-7),
                 f'Speed MAE discrepancy at sample={i}, arm={arm}')
        delta=row['arms']['dynamic']['W22']-row['arms']['control']['W22']
        need(math.isclose(delta,row['dynamic_minus_control_W22'],abs_tol=1e-10),
             'Paired difference mismatch')
        print(f'[VERIFY] sample={i} control_W22={row["arms"]["control"]["W22"]:.6f} '
              f'dynamic_W22={row["arms"]["dynamic"]["W22"]:.6f}',flush=True)
    diff=[x['dynamic_minus_control_W22'] for x in report['rows']]
    summary=report['summary']
    for arm in ('cnn','control','dynamic'):
        for key in ('dB','W2inf','W22','L_uv','L_grad','speed_crop_MAE_mps'):
            observed=float(np.mean([row['arms'][arm][key] for row in report['rows']]))
            need(math.isclose(observed,summary['mean_metrics'][arm][key],
                              rel_tol=1e-12,abs_tol=1e-12),
                 f'Summary mean changed for {arm}:{key}')
    need(summary['dynamic_better_count']==sum(x<0 for x in diff),'Win count mismatch')
    need(math.isclose(summary['mean_dynamic_minus_control_W22'],float(np.mean(diff)),
                      rel_tol=1e-12,abs_tol=1e-12),'Mean W22 difference mismatch')
    print('[VERIFY] PASS: source lineage, 8 replayed steps, immutable input/output hashes, '
          '4×3 recomputed canonical diagram metrics and physical-speed errors',flush=True)
    print('[SUMMARY]',json.dumps(summary,sort_keys=True),flush=True)


def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--mode',required=True,choices=('selftest','preflight','export','replay','score','verify'))
    a.add_argument('--repo-root',type=Path,default=Path.cwd())
    a.add_argument('--out-dir',type=Path,default=Path(NEW_OUT))
    a.add_argument('--ttk-backend',choices=('docker','host'),default='docker')
    a.add_argument('--ttk-timeout',type=int,default=180)
    args=a.parse_args()
    root=args.repo_root.resolve()
    out=(root/args.out_dir).resolve()
    if args.mode=='selftest':selftest()
    elif args.mode=='preflight':preflight(root,out)
    elif args.mode=='export':export_fields(root,out)
    elif args.mode=='replay':replay(root,out)
    elif args.mode=='score':score(root,out,args.ttk_backend,args.ttk_timeout)
    elif args.mode=='verify':verify(root,out,args.ttk_backend)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        print(f'[PHASE8-OUT-OF-PILOT] ERROR {type(exc).__name__}: {exc}',
              file=sys.stderr,flush=True)
        sys.exit(1)
