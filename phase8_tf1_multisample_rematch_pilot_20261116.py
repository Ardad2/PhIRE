#!/usr/bin/env python3
"""Phase 8: small TRUE re-matching native-PhIRE training experiment.

Run from the PhIRE repository root. This is a development-only training diagnostic:
  selftest -> preflight -> export -> prepare -> train -> verify

Native Python/TensorFlow: export and train. gudhi-audit Python: prepare and
verify. The train stage uses subprocess calls to micromamba/gudhi-audit for
match-worker and eval-worker: each dynamic step obtains NEW TTK predicted
critical vertices and NEW GUDHI assignments *before* its optimizer update.
The GT TTK diagrams are cached. Baseline/dynamic sessions are independent,
initialized from exactly the same frozen pretrained CNN. No model checkpoints
are written. No held-out fields are loaded. Source hashes and artifacts are
recorded. Output collision protection is mandatory.

Not a generalizable experiment: only four designated training fields with two
passes by default. No metrics are reported as validation/test results.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

BRIDGE = 'phase8_tf1_dynamic_single_batch_gate_20261116_v2.py'
ONE_STEP = 'phase8_tf1_one_step_optimizer_gate_20261116.py'
MATCH_GATE = 'phase8_matching_vertex_gradient_gate_20261116.py'
TIMING_GATE = 'phase8_ttk_w22_timing_pilot_20261116.py'
PHASE6 = 'phase6_lxxvii_update/scripts/phase6_pilot_pd_gudhi_gate.py'
EXTRACT = 'scripts/extract_ttk_pd_critical_pairs.py'
OUT_DEFAULT = Path('ttk_runs_fixed/topology_finetuning/phase8_multisample_rematch_20261116_r01')
TFRECORD = Path('example_data_topology_expanded_2688/wind_MR-HR.tfrecord')
CHECKPOINT = Path('models/wind_mr-hr/trained_cnn/cnn')
IDS = (0, 1, 2, 3)
MU = np.asarray([0.7684, -0.4575], dtype=np.float32)
SIG = np.asarray([5.02455, 5.9017], dtype=np.float32)
CROP = 160
LAMBDA_GRAD = .05
SCHEMA = 'phase8-tf1-four-sample-fresh-match-v1'


def need(condition, description):
    if not condition:
        raise RuntimeError(description)


def sha(p):
    d = hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            d.update(chunk)
    return d.hexdigest()


def import_py(p, name):
    spec = importlib.util.spec_from_file_location(name, p)
    need(spec and spec.loader, f'Cannot import source: {p}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def addjson(p, obj):
    p.write_text(json.dumps(obj, sort_keys=True, indent=2) + '\n')


def jsonread(p):
    return json.loads(p.read_text())


def atomic(dest, writer):
    """Only publish a complete result; preserve uniquely staged failures for audit."""
    need(not dest.exists(), f'Refusing pre-existing output: {dest}')
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix='.' + dest.name + '.staging-', dir=dest.parent))
    try:
        writer(tmp)
        need(not dest.exists(), f'Output collision: {dest}')
        tmp.rename(dest)
    except Exception as exc:
        # A failed training experiment must not be mistaken for a completed one.
        # Preserve its unique partial directory for inspection, not resumption.
        try:
            addjson(tmp / 'FAILED.json', {'status': 'INCOMPLETE_DO_NOT_USE',
                  'exception': f'{type(exc).__name__}: {exc}',
                  'completed_output_not_published': True})
        except Exception:
            pass
        print(f'[INCOMPLETE] Stage preserved for forensic inspection: {tmp}',
              file=sys.stderr, flush=True)
        raise


def check_root(root):
    need(root.is_dir() and (root / 'sr_network.py').is_file()
         and (root / 'PhIREGANs.py').is_file(), 'Run from the PhIRE repository root')
    need((root / TFRECORD).is_file(), 'Expanded 2,688 training TFRecord missing')
    for path in (BRIDGE, ONE_STEP, MATCH_GATE, TIMING_GATE, PHASE6, EXTRACT,
                 'scripts/run_candidateF_expanded2688_finetune.py'):
        need((root / path).is_file(), f'Missing dependency {path}')
    bridge = import_py(root / BRIDGE, 'phase8_fresh_match_bridge')
    need(bridge.CANONICAL.is_file(), 'Frozen canonical PD decoder missing')
    checkpoint = bridge.checkpoint_members(root)
    return bridge, checkpoint


def source_sha(root, bridge):
    paths = [Path(__file__).resolve(),
             *(root / p for p in [BRIDGE, ONE_STEP, MATCH_GATE, TIMING_GATE,
                                 PHASE6, EXTRACT, 'sr_network.py', 'PhIREGANs.py',
                                 'scripts/run_candidateF_expanded2688_finetune.py']),
             bridge.CANONICAL]
    return {str(p.resolve()): sha(p) for p in paths}


def require_hashes(recorded):
    for key, expected in recorded.items():
        path = Path(key)
        need(path.is_file() and sha(path) == expected, f'Source/input hash drift: {path}')


def hash_files(base, relative):
    answer = {}
    for name in relative:
        rel = Path(name)
        need(not rel.is_absolute() and '..' not in rel.parts and rel.parts,
             f'Unsafe relative path: {name}')
        file = base / rel
        need(file.is_file(), f'Missing artifact: {file}')
        answer[str(rel)] = sha(file)
    return answer


def verify_files(base, hashes):
    need(isinstance(hashes, dict) and hashes, 'Empty artifact-hash manifest')
    for name, expect in hashes.items():
        rel = Path(name)
        need(not rel.is_absolute() and '..' not in rel.parts and rel.parts,
             f'Unsafe manifest path: {name}')
        need((base / rel).is_file() and sha(base / rel) == expect,
             f'Artifact hash changed: {base / rel}')


def ids_list(indices):
    need(tuple(indices) == IDS, f'This frozen pilot requires exactly {IDS}; got {indices}')
    need(len(set(indices)) == len(indices) and all(0 <= i < 2688 for i in indices),
         'Not a unique set of TRAINING indices')
    return tuple(indices)


def speed_from_normalized(hr):
    need(hr.shape == (1, 500, 500, 2), 'Unexpected normalized HR shape')
    phys = np.asarray(hr[0], dtype=np.float32) * SIG + MU
    out = np.sqrt(np.sum(phys ** 2, axis=-1) + np.float32(1e-8))[:CROP, :CROP]
    need(out.shape == (CROP, CROP) and np.isfinite(out).all(), 'Bad GT crop')
    return np.ascontiguousarray(out, dtype=np.float32)


def selftest():
    ids_list(IDS)
    hr = np.zeros((1,500,500,2), dtype=np.float32)
    x = speed_from_normalized(hr)
    need(np.allclose(x, math.hypot(*MU), atol=2e-6), 'Speed denormalization is wrong')
    fake = np.array([[0., 1.], [2., 4.]])
    need(math.isclose(.5 * (fake[1,1] - fake[1,0])**2, 2.), 'Diagonal cost incorrect')
    need(math.isclose(1e-5 * 427.376501, .00427376501, abs_tol=1e-12),
         'Squared-W22 coefficient incorrect')
    print('[SELFTEST] PASS: frozen training indices, denormalized speed, diagonal cost, '
          'squared-W22 loss scaling; no files written', flush=True)


def preflight(root, out):
    bridge, cp = check_root(root)
    need(not out.exists(), f'Experiment output exists: {out}')
    need(Path(__file__).resolve().parent == root, 'Copy script to repository root')
    need(IDS == (0,1,2,3), 'Unexpected sample selection')
    print('[PREFLIGHT] PASS: four fixed indices exclusively from 2,688 training TFRecord, '
          'frozen CNN available, no output collision', flush=True)
    return bridge, cp


def export_inputs(root, out):
    bridge, cp = preflight(root,out)
    tf, _network, _denorm, PhIREGANs = bridge.import_tf(root)
    tf.reset_default_graph()
    phire = PhIREGANs(data_type='phase8_multisample_readonly',
                      mu_sig=[MU.tolist(), SIG.tolist()])
    ds = tf.data.TFRecordDataset(str(root / TFRECORD))
    ds = ds.map(lambda sample: phire._parse_train_(sample, phire.mu_sig))
    it = ds.make_one_shot_iterator().get_next()
    found = {}
    t0=time.perf_counter()
    with tf.Session() as sess:
        while len(found) != len(IDS):
            try:
                idx,lr,hr = sess.run(it)
            except tf.errors.OutOfRangeError as exc:
                raise RuntimeError(f'Training indices missing: {sorted(set(IDS)-set(found))}') from exc
            idx = int(idx)
            if idx not in IDS:
                continue
            need(idx not in found, f'Training sample index repeated: {idx}')
            need(lr.shape == (100,100,2) and hr.shape == (500,500,2),
                 f'Unexpected training shapes at index {idx}: {lr.shape}, {hr.shape}')
            found[idx] = (lr[np.newaxis].astype(np.float32), hr[np.newaxis].astype(np.float32))
    before = bridge.filehashes(cp)
    names=[]
    def write(p):
        for idx in IDS:
            lr,hr=found[idx]
            stem=f's{idx:04d}'
            for part, data in [('lr',lr),('hr',hr),('gt_crop',speed_from_normalized(hr))]:
                name=f'{stem}_{part}.npy'
                np.save(p/name,data)
                names.append(name)
        addjson(p/'manifest.json', {
            'schema':SCHEMA+'-inputs', 'train_indices':list(IDS),
            'source_hashes':source_sha(root,bridge), 'checkpoint_hashes':before,
            'training_tfrecord':str(TFRECORD),
            'tfrecord_bytes':(root/TFRECORD).stat().st_size,
            'output_hashes':hash_files(p,names),
            'elapsed_seconds':round(time.perf_counter()-t0,5),
            'no_heldout':True, 'no_optimizer':True})
    atomic(out/'inputs',write)
    print(f'[EXPORT] PASS: {len(found)} normalized training samples with GT crops, '
          f'no weights changed; {out/"inputs"}', flush=True)


def inputs_check(root,out,bridge):
    p=out/'inputs'
    need(p.is_dir(), f'Run --mode export first: {p}')
    manifest=jsonread(p/'manifest.json')
    need(manifest['schema']==SCHEMA+'-inputs' and manifest['train_indices']==list(IDS)
         and manifest['no_heldout'] and manifest['no_optimizer'], 'Invalid exported input manifest')
    require_hashes(manifest['source_hashes'])
    require_hashes(manifest['checkpoint_hashes'])
    verify_files(p,manifest['output_hashes'])
    need((root/TFRECORD).stat().st_size == manifest['tfrecord_bytes'], 'TFRecord source size changed')
    return manifest


def get_sample(out, idx):
    need(idx in IDS, 'Index outside selected training fields')
    p=out/'inputs'; stem=f's{idx:04d}'
    lr=np.load(p/f'{stem}_lr.npy',allow_pickle=False)
    hr=np.load(p/f'{stem}_hr.npy',allow_pickle=False)
    gt=np.load(p/f'{stem}_gt_crop.npy',allow_pickle=False)
    need(lr.shape==(1,100,100,2) and hr.shape==(1,500,500,2)
         and gt.shape==(CROP,CROP), 'Corrupt input field dimensions')
    need(np.array_equal(gt,speed_from_normalized(hr)), 'Cached GT speed does not match normalized HR')
    return lr,hr,gt


def audit_modules(root, bridge):
    timer=import_py(root/TIMING_GATE,'phase8_multisample_timing')
    pilot=import_py(root/PHASE6,'phase8_multisample_phase6')
    decoder=import_py(bridge.CANONICAL,'phase8_multisample_decoder')
    gate=import_py(root/MATCH_GATE,'phase8_multisample_gate')
    extractor=import_py(root/EXTRACT,'phase8_multisample_extractor')
    return timer,pilot,decoder,gate,extractor


def prepare(root,out,backend,timeout):
    bridge,_ = check_root(root)
    inputs_check(root,out,bridge)
    timer,pilot,decoder,_gate,_extractor=audit_modules(root,bridge)
    timer.check_backend(backend)
    def write(p):
        (p/'vti').mkdir();(p/'pd').mkdir()
        timings={}
        for idx in IDS:
            _,_,gt=get_sample(out,idx)
            tag=f'gt_s{idx:04d}'
            timer.write_vti(p/'vti'/f'{tag}.vti',gt)
            timings[str(idx)]=timer.run_ttk(backend,p,tag,timeout)
            diagram=pilot.parse(decoder.read_pd,p/'pd'/f'{tag}_pd_port_0.vtu')
            need(all(x.ndim==2 and x.shape[1]==2 for x in diagram.values()),'GT PD invalid')
            print(f'[PREPARE] GT sample {idx}, TTK={timings[str(idx)]["seconds"]:.3f}s, '
                  f'D0={len(diagram[0])}, D1={len(diagram[1])}',flush=True)
        files=[*(f'vti/gt_s{i:04d}.vti' for i in IDS),
               *(f'pd/gt_s{i:04d}_pd_port_0.vtu' for i in IDS)]
        addjson(p/'manifest.json',{'schema':SCHEMA+'-gt-cache',
            'source_hashes':source_sha(root,bridge),
            'input_manifest_sha256':sha(out/'inputs'/'manifest.json'),
            'train_indices':list(IDS),'timing_seconds':timings,
            'artifacts':hash_files(p,files),
            'note':'Cached GT diagrams: independent TTK on denormalized 160x160 speed.'})
    atomic(out/'gt_cache',write)
    print(f'[PREPARE] PASS: {len(IDS)} GT diagrams cached, no heldout fields; {out/"gt_cache"}',flush=True)


def cache_check(root,out,bridge):
    inputs_check(root,out,bridge)
    cache=out/'gt_cache'
    need(cache.is_dir(),'Run --mode prepare first')
    m=jsonread(cache/'manifest.json')
    need(m['schema']==SCHEMA+'-gt-cache' and m['train_indices']==list(IDS), 'Bad GT cache')
    require_hashes(m['source_hashes'])
    need(m['input_manifest_sha256']==sha(out/'inputs'/'manifest.json'), 'GT inputs changed')
    verify_files(cache,m['artifacts'])
    return m


def match_worker(root,out,step_dir,idx,backend,timeout,max_pairs):
    bridge,_=check_root(root)
    cache_check(root,out,bridge)
    timer,pilot,decoder,gate,extractor=audit_modules(root,bridge)
    timer.check_backend(backend)
    gt=pilot.parse(decoder.read_pd,out/'gt_cache'/'pd'/f'gt_s{idx:04d}_pd_port_0.vtu')
    sr=np.load(step_dir/'dynamic_before.npy',allow_pickle=False)
    need(sr.shape==(CROP,CROP) and np.isfinite(sr).all(),'Bad fresh dynamic field')
    for folder in ('vti','pd'):
        (step_dir/folder).mkdir(exist_ok=False)
    timer.write_vti(step_dir/'vti'/'dynamic_before.vti',sr)
    dt=timer.run_ttk(backend,step_dir,'dynamic_before',timeout)
    diagram=pilot.parse(decoder.read_pd,step_dir/'pd'/'dynamic_before_pd_port_0.vtu')
    raw=gate.raw_pairs(step_dir/'pd'/'dynamic_before_pd_port_0.vtu',sr,extractor)
    srmap,unmapped=gate.align(diagram,raw)
    from gudhi.wasserstein import wasserstein_distance as wasserstein
    vids, dvids, target, modes = [],[],[],[]
    gt_diag=0.
    squared=0.
    per_dim={}
    for dim in (0,1):
        reference=gate.positive(gt[dim]); predicted=gate.positive(diagram[dim])
        need(len(reference)+len(predicted)<=max_pairs, f'D{dim}: too many PD points')
        need(len(srmap[dim])==len(predicted), f'D{dim}: cannot map all pairs')
        t0=time.perf_counter()
        dist, matches=wasserstein(reference,predicted,order=2.,internal_p=2.,
                                   matching=True,keep_essential_parts=False)
        J,g,counts=gate.matching_obj_and_grad(sr,reference,srmap[dim],matches)
        need(math.isclose(J,float(dist)**2,rel_tol=5e-5,abs_tol=1e-4),
             f'D{dim} matched cost differs from canonical W22 squared')
        squared+=J
        seen=set()
        for i,j in np.asarray(matches,dtype=np.int64).reshape((-1,2)):
            i,j=int(i),int(j)
            if j<0:
                need(i>=0,'Invalid empty match')
                gt_diag+=.5*float(reference[i,1]-reference[i,0])**2
            else:
                need(j not in seen,'Duplicate matched predicted pair')
                seen.add(j)
                r=srmap[dim][j]
                vids.append(r['birth_vid']);dvids.append(r['death_vid'])
                modes.append(0 if i>=0 else 1)
                target.append([float(reference[i,0]),float(reference[i,1])]
                              if i>=0 else [0.,0.])
        need(len(seen)==len(predicted),'Incomplete predicted pair coverage')
        per_dim[f'D{dim}']={'W22':float(dist),'count_gt':len(reference),
                             'count_pred':len(predicted),**counts,
                             'matching_seconds':round(time.perf_counter()-t0,6)}
    true=pilot.distance(gt,diagram)['W22']
    need(math.isclose(squared,true**2,rel_tol=5e-5,abs_tol=1e-4),
         'Combined squared W22 matching divergence')
    need(len(vids)>0,'Cannot construct dynamic loss: no predicted pairs')
    coord=lambda v:np.stack([np.asarray(v,dtype=np.int32)//CROP,
                             np.asarray(v,dtype=np.int32)%CROP],axis=-1)
    np.savez_compressed(step_dir/'feed.npz',
        birth_yx=coord(vids)[None,...], death_yx=coord(dvids)[None,...],
        target=np.asarray(target,dtype=np.float32)[None,...],
        mode=np.asarray(modes,dtype=np.int32)[None,...],
        mask=np.ones((1,len(vids)),np.float32),
        gt_diagonal_cost=np.asarray(gt_diag,dtype=np.float32),
        expected_squared_cost=np.asarray(squared,dtype=np.float64))
    addjson(step_dir/'match.json',{'sample':idx,'prediction_sha256':sha(step_dir/'dynamic_before.npy'),
        'canonical_W22':float(true),'matched_squared_cost':float(squared),
        'pairs':len(vids),'ttk':dt,'per_dim':per_dim,
        'unmapped_raw_pairs':unmapped,
        'artifact_hashes':hash_files(step_dir,['feed.npz','dynamic_before.npy',
            'vti/dynamic_before.vti','pd/dynamic_before_pd_port_0.vtu']),
        'note':'NEW predicted diagram and NEW matching computed for this update.'})
    print(f'[MATCH-WORKER] sample={idx} before_W22={true:.8g} '
          f'predicted_pairs={len(vids)}',flush=True)


def evaluate_worker(root,out,step_dir,idx,backend,timeout,max_pairs):
    bridge,_=check_root(root)
    cache_check(root,out,bridge)
    timer,pilot,decoder,_gate,_extractor=audit_modules(root,bridge)
    timer.check_backend(backend)
    reference=pilot.parse(decoder.read_pd,out/'gt_cache'/'pd'/f'gt_s{idx:04d}_pd_port_0.vtu')
    stored=jsonread(step_dir/'match.json')
    verify_files(step_dir,stored['artifact_hashes'])
    scores={}
    times={}
    for arm in ('baseline','dynamic'):
        field=np.load(step_dir/f'{arm}_after.npy',allow_pickle=False)
        need(field.shape==(CROP,CROP) and np.isfinite(field).all(),'Bad updated field')
        timer.write_vti(step_dir/'vti'/f'{arm}_after.vti',field)
        times[arm]=timer.run_ttk(backend,step_dir,f'{arm}_after',timeout)
        dp=pilot.parse(decoder.read_pd,step_dir/'pd'/f'{arm}_after_pd_port_0.vtu')
        for dim in (0,1):
            positive_g=reference[dim][reference[dim][:,1]>reference[dim][:,0]]
            positive_s=dp[dim][dp[dim][:,1]>dp[dim][:,0]]
            need(len(positive_g)+len(positive_s)<=max_pairs,
                 f'{arm} D{dim}: too many diagram points for safe evaluation')
        scores[arm]=pilot.distance(reference,dp)['W22']
        need(math.isfinite(scores[arm]) and scores[arm]>=0,'Invalid true W22')
    addjson(step_dir/'eval.json',{'sample':idx,'initial_dynamic_W22':stored['canonical_W22'],
        'after_W22':scores,'timing_TTK_seconds':times,
        'artifact_hashes':hash_files(step_dir,[
            *(f'{a}_after.npy' for a in ('baseline','dynamic')),
            *(f'vti/{a}_after.vti' for a in ('baseline','dynamic')),
            *(f'pd/{a}_after_pd_port_0.vtu' for a in ('baseline','dynamic'))])})
    print(f'[EVAL-WORKER] sample={idx} true W22 control={scores["baseline"]:.8g} '
          f'dynamic={scores["dynamic"]:.8g}',flush=True)


def spawn_worker(root,out,step_dir,idx,mode,backend,timeout,max_pairs):
    executable=shutil.which('micromamba')
    need(executable is not None, 'micromamba not found; no environment changes attempted')
    command=[executable,'run','-n','gudhi-audit','python',str(Path(__file__).resolve()),
             '--mode',mode,'--repo-root',str(root),'--out-dir',str(out),
             '--step-dir',str(step_dir),'--sample-index',str(idx),
             '--ttk-backend',backend,'--ttk-timeout',str(timeout),
             '--max-matching-points',str(max_pairs)]
    env=os.environ.copy();env['PYTHONNOUSERSITE']='1'
    t0=time.perf_counter()
    p=subprocess.run(command,cwd=root,env=env,text=True,capture_output=True,
                     timeout=max(timeout*4,600))
    if p.stdout:
        print(p.stdout.strip(),flush=True)
    if p.stderr:
        print(p.stderr[-2000:].strip(),file=sys.stderr,flush=True)
    need(p.returncode==0,f'{mode} subprocess returned {p.returncode} (see above)')
    return round(time.perf_counter()-t0,6)


def train(root,out,lr,lam,rounds,backend,timeout,max_pairs):
    need(0<lr<=1e-5 and 0<lam<=1e-4,'Unsafe diagnostic LR or lambda')
    need(rounds in (1,2,3),'This pilot supports 1-3 passes, not general training')
    bridge,cp=check_root(root)
    cache_check(root,out,bridge)
    need(not (out/'train').exists(),'Training output already exists; no overwrite')
    need(shutil.which('micromamba') is not None,'micromamba not available')
    original=bridge.filehashes(cp)
    one=import_py(root/ONE_STEP,'phase8_fresh_match_one_step')
    tf,xl,xh,ph,fetch,trains,saver,init,variables=one.make_tf_graph(bridge,root,lr,lam)
    t0=time.perf_counter()
    params={}
    def writer(stage):
        (stage/'steps').mkdir()
        session_config=tf.ConfigProto()
        session_config.gpu_options.allow_growth=True
        with tf.Session(config=session_config) as ctrl, tf.Session(config=session_config) as dyn:
            for sess in (ctrl,dyn):
                sess.run(init)
                saver.restore(sess,str(root/CHECKPOINT))
            def paramsha(sess):
                h=hashlib.sha256()
                for variable in sess.run(variables):
                    h.update(np.asarray(variable).tobytes())
                return h.hexdigest()
            initial_control=paramsha(ctrl)
            initial_dynamic=paramsha(dyn)
            need(initial_control==initial_dynamic,'Arms did not begin from identical checkpoint weights')
            rows=[]
            order=IDS*rounds
            for stepno,idx in enumerate(order):
                lr_field,hr_field,_=get_sample(out,idx)
                feed={xl:lr_field,xh:hr_field}
                stepdir=stage/'steps'/f'iter_{stepno:03d}_s{idx:04d}'
                stepdir.mkdir()
                pre=dyn.run(fetch['crop'],feed_dict=feed)[0]
                need(pre.shape==(CROP,CROP) and np.isfinite(pre).all(), 'Bad pre-update prediction')
                np.save(stepdir/'dynamic_before.npy',np.asarray(pre,dtype=np.float32))
                match_secs=spawn_worker(root,out,stepdir,idx,'match-worker',backend,timeout,max_pairs)
                m=jsonread(stepdir/'match.json')
                verify_files(stepdir,m['artifact_hashes'])
                with np.load(stepdir/'feed.npz',allow_pickle=False) as z:
                    pairs={k:np.array(z[k]) for k in z.files}
                feed_dyn=dict(feed)
                for key,phitem in zip(('birth_yx','death_yx','target','mode','mask'),ph[:5]):
                    feed_dyn[phitem]=pairs[key]
                feed_dyn[ph[5]]=float(pairs['gt_diagonal_cost'])
                J=float(dyn.run(fetch['J_frozen'],feed_dict=feed_dyn))
                need(math.isclose(J,float(pairs['expected_squared_cost']),
                                  rel_tol=3e-4,abs_tol=1e-3),
                     f'iter={stepno}: TensorFlow J differs from freshly matched canonical GUDHI squared W22')
                base_pre=ctrl.run({'L_uv':fetch['L_uv'],'L_grad':fetch['L_grad'],
                                   'loss':fetch['base_loss']},feed_dict=feed)
                dyn_pre=dyn.run({'L_uv':fetch['L_uv'],'L_grad':fetch['L_grad'],
                                 'loss':fetch['dynamic_loss'],
                                 'base_grad':fetch['base_grad_norm'],
                                 'dynamic_grad':fetch['dynamic_grad_norm'],
                                 'PD_grad':fetch['PD_grad_norm_unscaled']},feed_dict=feed_dyn)
                need(all(math.isfinite(float(v)) for v in (*base_pre.values(),*dyn_pre.values())),
                     f'Nonfinite optimizer losses/gradients at step {stepno}')
                # Both arms take ONE Adam step on exactly the same training field.
                ctrl.run(trains['baseline'],feed_dict=feed)
                dyn.run(trains['dynamic'],feed_dict=feed_dyn)
                b_after=ctrl.run({'speed':fetch['crop'],'L_uv':fetch['L_uv'],
                                   'L_grad':fetch['L_grad']},feed_dict=feed)
                d_after=dyn.run({'speed':fetch['crop'],'L_uv':fetch['L_uv'],
                                  'L_grad':fetch['L_grad']},feed_dict=feed)
                for arm,data in [('baseline',b_after),('dynamic',d_after)]:
                    cropped=np.asarray(data['speed'][0],dtype=np.float32)
                    need(cropped.shape==(CROP,CROP) and np.isfinite(cropped).all(),
                         f'Bad post-step field {arm} step {stepno}')
                    np.save(stepdir/f'{arm}_after.npy',cropped)
                eval_secs=spawn_worker(root,out,stepdir,idx,'eval-worker',backend,timeout,max_pairs)
                evaluation=jsonread(stepdir/'eval.json')
                verify_files(stepdir,evaluation['artifact_hashes'])
                row={
                    'step':stepno,'sample_index':idx,'round':stepno//len(IDS),
                    'W22_before_dynamic':float(m['canonical_W22']),
                    'W22_after_control':evaluation['after_W22']['baseline'],
                    'W22_after_dynamic':evaluation['after_W22']['dynamic'],
                    'J_fresh_matching':float(pairs['expected_squared_cost']),
                    'J_in_TF':J,'predicted_pairs':int(m['pairs']),
                    'L_uv_before_control':float(base_pre['L_uv']),
                    'L_uv_before_dynamic':float(dyn_pre['L_uv']),
                    'L_uv_after_control':float(b_after['L_uv']),
                    'L_uv_after_dynamic':float(d_after['L_uv']),
                    'L_grad_after_control':float(b_after['L_grad']),
                    'L_grad_after_dynamic':float(d_after['L_grad']),
                    'base_grad_norm':float(dyn_pre['base_grad']),
                    'total_dynamic_grad_norm':float(dyn_pre['dynamic_grad']),
                    'raw_PD_grad_norm':float(dyn_pre['PD_grad']),
                    'seconds_match_worker':match_secs,'seconds_eval_worker':eval_secs,
                    'match_sha256':sha(stepdir/'match.json'),
                    'eval_sha256':sha(stepdir/'eval.json'),
                }
                rows.append(row)
                addjson(stepdir/'step.json',row)
                print(f'[TRAIN-STEP {stepno+1}/{len(order)}] sample={idx} '
                      f'control_W22={row["W22_after_control"]:.7g} '
                      f'dynamic_W22={row["W22_after_dynamic"]:.7g} '
                      f'J={J:.7g} pairs={m["pairs"]}',flush=True)
            need(paramsha(ctrl)!=initial_control and paramsha(dyn)!=initial_dynamic,
                 'At least one arm failed to update parameters')
        need(bridge.filehashes(cp)==original, 'Frozen pretrained checkpoint was modified')
        names=[]
        for stepno,idx in enumerate(IDS*rounds):
            prefix=f'steps/iter_{stepno:03d}_s{idx:04d}'
            names.extend(f'{prefix}/{p}' for p in (
                'dynamic_before.npy','baseline_after.npy','dynamic_after.npy',
                'match.json','eval.json','feed.npz','step.json',
                'vti/dynamic_before.vti','pd/dynamic_before_pd_port_0.vtu',
                'vti/baseline_after.vti','pd/baseline_after_pd_port_0.vtu',
                'vti/dynamic_after.vti','pd/dynamic_after_pd_port_0.vtu'))
        addjson(stage/'manifest.json',{
            'schema':SCHEMA+'-train','method':'actual rematching before EVERY dynamic update',
            'training_indices':list(IDS),'order':list(IDS*rounds),
            'rounds':rounds,'n_steps_per_arm':len(rows),
            'learning_rate':lr,'lambda_dynamic_W22_squared':lam,
            'lambda_grad':LAMBDA_GRAD,
            'source_hashes':source_sha(root,bridge),
            'original_checkpoint_hashes':original,
            'inputs_manifest_sha256':sha(out/'inputs'/'manifest.json'),
            'gt_cache_manifest_sha256':sha(out/'gt_cache'/'manifest.json'),
            'initial_parameter_sha256':initial_control,
            'rows':rows,'output_hashes':hash_files(stage,names),
            'elapsed_seconds':round(time.perf_counter()-t0,6),
            'no_checkpoint_writes':True,'no_heldout_fields':True,
            'limitations':['Only four training fields; no independent generalization result.',
                'Piecewise differentiable local matching surrogate; do not differentiate through TTK.',
                'Unmatched GT diagonal terms contribute constant cost, no predicted-field gradient.',
                'Raw squared W22 is summed, not averaged over persistence pairs.',
                'No top-K pruning or nonzero persistence threshold; matches canonical Phase6 W22.',
                'This is a short mini-pilot, not a 2,688-field model run.']})
    atomic(out/'train',writer)
    print(f'[TRAIN] PASS: {len(IDS)*rounds} fresh dynamic rematchings, two independent TF1 '
          f'Adam sessions, frozen checkpoint unchanged; {out/"train"}',flush=True)


def verify(root,out):
    bridge,_=check_root(root)
    cache_check(root,out,bridge)
    p=out/'train'
    need(p.is_dir(),'Run --mode train first')
    manifest=jsonread(p/'manifest.json')
    need(manifest['schema']==SCHEMA+'-train' and manifest['no_checkpoint_writes']
         and manifest['no_heldout_fields'], 'Invalid training artifact schema')
    require_hashes(manifest['source_hashes'])
    require_hashes(manifest['original_checkpoint_hashes'])
    need(sha(out/'inputs'/'manifest.json')==manifest['inputs_manifest_sha256'],
         'Input manifest changed')
    need(sha(out/'gt_cache'/'manifest.json')==manifest['gt_cache_manifest_sha256'],
         'GT cache changed')
    verify_files(p,manifest['output_hashes'])
    timer,pilot,decoder,_gate,_extractor=audit_modules(root,bridge)
    order=list(IDS)*manifest['rounds']
    need(manifest['order']==order and manifest['n_steps_per_arm']==len(order)
         and len(manifest['rows'])==len(order), 'Invalid chronological training order')
    diffs=[]
    for stepno,idx in enumerate(order):
        row=manifest['rows'][stepno]
        need(row['step']==stepno and row['sample_index']==idx,'Step mislabelled')
        step=p/'steps'/f'iter_{stepno:03d}_s{idx:04d}'
        mm=jsonread(step/'match.json');ev=jsonread(step/'eval.json')
        need(sha(step/'match.json')==row['match_sha256'] and
             sha(step/'eval.json')==row['eval_sha256'],'Per-step upstream evidence changed')
        verify_files(step,mm['artifact_hashes'])
        verify_files(step,ev['artifact_hashes'])
        with np.load(step/'feed.npz',allow_pickle=False) as z:
            J=float(z['expected_squared_cost'])
        need(math.isclose(J,row['J_fresh_matching'],rel_tol=1e-12,abs_tol=1e-9),
             'Matched J changed')
        gt=pilot.parse(decoder.read_pd,out/'gt_cache'/'pd'/f'gt_s{idx:04d}_pd_port_0.vtu')
        for tag,expect in [('dynamic_before',row['W22_before_dynamic']),
                           ('baseline_after',row['W22_after_control']),
                           ('dynamic_after',row['W22_after_dynamic'])]:
            diagram=pilot.parse(decoder.read_pd,step/'pd'/f'{tag}_pd_port_0.vtu')
            actual=pilot.distance(gt,diagram)['W22']
            need(math.isclose(actual,expect,rel_tol=2e-7,abs_tol=2e-5),
                 f'Canonical W22 drift step {stepno} tag={tag}: {actual} != {expect}')
        need(math.isclose(row['J_fresh_matching'],row['W22_before_dynamic']**2,
                          rel_tol=5e-5,abs_tol=1e-4),'Matching does not give true squared W22')
        diffs.append(row['W22_after_dynamic']-row['W22_after_control'])
        print(f'[VERIFY {stepno+1}/{len(order)}] sample={idx} control={row["W22_after_control"]:.8g} '
              f'dynamic={row["W22_after_dynamic"]:.8g}',flush=True)
    result={'train_steps':len(order),'delta_dynamic_minus_control':diffs,
            'dynamic_better_steps':sum(x<0 for x in diffs),
            'dynamic_worse_steps':sum(x>0 for x in diffs),
            'mean_delta':float(np.mean(diffs))}
    print(f'[VERIFY] PASS: {len(order)} fresh W22 matchings and 2x{len(order)} '
          f'independent true TTK/GUDHI checks; no held-out data, no checkpoint writes',flush=True)
    print('[SUMMARY]',json.dumps(result,sort_keys=True),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',required=True,choices=(
        'selftest','preflight','export','prepare','train','match-worker','eval-worker','verify'))
    parser.add_argument('--repo-root',type=Path,default=Path.cwd())
    parser.add_argument('--out-dir',type=Path,default=OUT_DEFAULT)
    parser.add_argument('--step-dir',type=Path)
    parser.add_argument('--sample-index',type=int,default=0)
    parser.add_argument('--rounds',type=int,default=2)
    parser.add_argument('--learning-rate',type=float,default=1e-6)
    parser.add_argument('--lambda-dynamic',type=float,default=1e-5)
    parser.add_argument('--ttk-backend',choices=('docker','host'),default='docker')
    parser.add_argument('--ttk-timeout',type=int,default=180)
    parser.add_argument('--max-matching-points',type=int,default=5000)
    a=parser.parse_args()
    root=a.repo_root.resolve();out=(root/a.out_dir).resolve()
    need(Path(__file__).resolve().parent==root or a.mode=='selftest',
         'Script must live at repository root; do not run a modified copy')
    if a.mode=='selftest':selftest()
    elif a.mode=='preflight':preflight(root,out)
    elif a.mode=='export':export_inputs(root,out)
    elif a.mode=='prepare':prepare(root,out,a.ttk_backend,a.ttk_timeout)
    elif a.mode=='train':train(root,out,a.learning_rate,a.lambda_dynamic,a.rounds,
                               a.ttk_backend,a.ttk_timeout,a.max_matching_points)
    elif a.mode in ('match-worker','eval-worker'):
        need(a.step_dir is not None and a.sample_index in IDS,'Worker parameters invalid')
        step=a.step_dir.resolve()
        need(step.is_dir() and out in step.parents and '.train.staging-' in str(step),
             'Worker requires isolated, in-progress training staging directory')
        if a.mode=='match-worker':match_worker(root,out,step,a.sample_index,a.ttk_backend,
                                               a.ttk_timeout,a.max_matching_points)
        else:evaluate_worker(root,out,step,a.sample_index,a.ttk_backend,a.ttk_timeout,
                             a.max_matching_points)
    else:verify(root,out)


if __name__=='__main__':
    try:
        main()
    except Exception as e:
        print(f'[PHASE8-FRESH-REMATCH] ERROR {type(e).__name__}: {e}',
              file=sys.stderr,flush=True)
        sys.exit(1)
