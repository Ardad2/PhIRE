#!/usr/bin/env python3
"""Phase 8 non-training TF1 dynamic-TTK matching bridge (training sample only).

Stages intentionally separate the PhIRE TensorFlow environment from the
GUDHI/TTK audit environment:
  selftest  -> pure NumPy target/mode/diagonal check (no writes)
  preflight-> source and training split checks (no writes)
  forward  -> TF1 pretrained generator forward only (writes isolated stage)
  match    -> TTK predicted+GT diagrams; canonical GUDHI matching (new stage)
  tfcheck  -> TF1 gather_nd scalar/weight gradient agreement (new stage)
  verify   -> read-only source/manifests and numerical trace checks

No optimizer, gradient application, checkpoint writing, model training, or
held-out 168-sample data access. All stages fail closed on existing outputs.
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

TRAIN_TFRECORD = Path('example_data_topology_expanded_2688/wind_MR-HR.tfrecord')
CHECKPOINT = Path('models/wind_mr-hr/trained_cnn/cnn')
OUTPUT = Path('ttk_runs_fixed/topology_finetuning/phase8_tf1_single_batch_bridge_20261116')
MU = np.array([0.7684, -0.4575], dtype=np.float32)
SIG = np.array([5.02455, 5.9017], dtype=np.float32)
CROP = 160
CANONICAL = Path.home() / 'phire_runtime_audit_20260809_221548/recompute_pd/canonical_pd_pilot.py'
SOURCES = [
    Path('sr_network.py'), Path('PhIREGANs.py'),
    Path('scripts/run_candidateF_expanded2688_finetune.py'),
    Path('scripts/extract_ttk_pd_critical_pairs.py'),
    Path('phase6_lxxvii_update/scripts/phase6_pilot_pd_gudhi_gate.py'),
    Path('phase8_ttk_w22_timing_pilot_20261116.py'),
    Path('phase8_matching_vertex_gradient_gate_20261116.py'),
]


def need(cond, reason):
    if not cond:
        raise RuntimeError(reason)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for piece in iter(lambda: f.read(1024 * 1024), b''):
            h.update(piece)
    return h.hexdigest()


def mod(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    need(spec is not None and spec.loader is not None, f'Cannot import {path}')
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def filehashes(paths):
    return {str(p): sha(p) for p in paths}


def check_hashes(root, recorded):
    for name, digest in recorded.items():
        p = Path(name)
        if not p.is_absolute():
            p = root / p
        need(p.is_file() and sha(p) == digest, f'Source hash mismatch: {p}')


def atomic_stage(final, writer):
    need(not final.exists(), f'Already exists; refusing overwrite: {final}')
    final.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f'.{final.name}.staging-', dir=final.parent))
    try:
        writer(work)
        need(not final.exists(), f'Output appeared during stage; aborting: {final}')
        work.rename(final)
    finally:
        if work.exists():
            shutil.rmtree(work)


def checkpoint_members(root):
    prefix = root / CHECKPOINT
    items = [Path(str(prefix) + '.index')]
    items += sorted(prefix.parent.glob(prefix.name + '.data-*'))
    need(items and all(p.is_file() for p in items), f'Missing pretrained CNN checkpoint: {prefix}')
    return items


def preflight(root, sample_index):
    need(root.is_dir() and (root / 'sr_network.py').is_file(), 'Run from the PhIRE repo root')
    need((root / TRAIN_TFRECORD).is_file(), f'Training TFRecord absent: {TRAIN_TFRECORD}')
    need(0 <= sample_index < 2688, 'Sample index must be in 2,688-field TRAINING set')
    need('expanded_2688' in str(TRAIN_TFRECORD), 'Refusing non-training source')
    members = checkpoint_members(root)
    for path in SOURCES:
        need((root / path).is_file(), f'Missing source {path}')
    need(CANONICAL.is_file(), f'Missing canonical decoder: {CANONICAL}')
    try:
        rev = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        rev = 'unknown'
    print(f'[PREFLIGHT] PASS: training index={sample_index}, sources, frozen pretrained checkpoint; repo HEAD={rev}', flush=True)
    return rev, members


def load_forward(root, out):
    p = out / 'forward'
    need(p.is_dir(), f'Run --mode forward first: {p}')
    m = json.loads((p / 'manifest.json').read_text())
    need(m['schema'] == 'phase8-tf1-forward-v1', 'Unexpected forward schema')
    check_hashes(root, m['source_hashes'])
    check_hashes(root, m['checkpoint_hashes'])
    for name, digest in m['output_hashes'].items():
        need(sha(p / name) == digest, f'Forward sample drift: {name}')
    return m


def load_match(root, out):
    p = out / 'match'
    need(p.is_dir(), f'Run --mode match first: {p}')
    m = json.loads((p / 'manifest.json').read_text())
    need(m['schema'] == 'phase8-tf1-dynamic-match-v1', 'Unexpected match schema')
    check_hashes(root, m['source_hashes'])
    for name, digest in m['output_hashes'].items():
        need(sha(p / name) == digest, f'Match artifact drift: {name}')
    need(m['upstream_manifest_sha256'] == sha(out / 'forward' / 'manifest.json'),
         'Forward manifest changed since matching')
    return m


def selftest():
    # Cost conventions: matched Euclidean squared, predicted->diag, GT->diag.
    pred = np.array([[1., 4.], [7., 9.]], dtype=np.float64)
    ref = np.array([[2., 5.], [10., 12.]], dtype=np.float64)
    mode = np.array([0, 1], dtype=np.int32)
    target = np.array([[2., 5.], [0., 0.]], dtype=np.float64)
    matched_cost = np.square(pred[mode == 0] - target[mode == 0]).sum()
    diagonal_cost = (np.diff(pred[mode == 1], axis=1) ** 2).sum() / 2
    gt_only_cost = (ref[1, 1] - ref[1, 0])**2 / 2
    expected = matched_cost + diagonal_cost + gt_only_cost
    need(math.isclose(expected, 6.0, abs_tol=1e-12), f'Selftest cost incorrect: {expected}')
    print('[SELFTEST] PASS: matched/diagonal target encoding, squared Euclidean W22, GT-only constant', flush=True)


def import_tf(root):
    sys.path.insert(0, str(root))
    try:
        import tensorflow.compat.v1 as tf
        tf.disable_v2_behavior()
        from sr_network import SR_NETWORK, _denorm_speed
        from PhIREGANs import PhIREGANs
    except ImportError as exc:
        raise RuntimeError(f'TensorFlow/PhIRE imports unavailable in this Python environment: {exc}. '
                           'Run forward/tfcheck in your existing native PhIRE TF1 environment, '
                           'not the separate gudhi-audit environment.') from exc
    return tf, SR_NETWORK, _denorm_speed, PhIREGANs


def make_model(tf, SR_NETWORK, _denorm_speed):
    xlr = tf.placeholder(tf.float32, [None, 100, 100, 2], name='phase8_xLR')
    xhr = tf.placeholder(tf.float32, [None, 500, 500, 2], name='phase8_xHR')
    model = SR_NETWORK(xlr, xhr, r=[5], status='pretraining', phys_cfg=None)
    speed = _denorm_speed(model.x_SR, tf.constant(MU), tf.constant(SIG))
    return xlr, xhr, model, speed, speed[:, :CROP, :CROP]


def read_training_sample(tf, PhIREGANs, root, index):
    phire = PhIREGANs(data_type='phase8_no_training', mu_sig=[MU.tolist(), SIG.tolist()])
    ds = tf.data.TFRecordDataset(str(root / TRAIN_TFRECORD))
    ds = ds.map(lambda xx: phire._parse_train_(xx, phire.mu_sig))
    ds = ds.filter(lambda i, _lr, _hr: tf.equal(i, tf.constant(index, dtype=tf.int64)))
    ds = ds.take(1).batch(1)
    itr = ds.make_one_shot_iterator()
    return itr.get_next()


def forward(root, out, index):
    rev, checkpoint_paths = preflight(root, index)
    tf, SR_NETWORK, _denorm_speed, PhIREGANs = import_tf(root)
    before = filehashes(checkpoint_paths)
    tf.reset_default_graph()
    xi, hi, model, speed, crop = make_model(tf, SR_NETWORK, _denorm_speed)
    idx_op, lr_op, hr_op = read_training_sample(tf, PhIREGANs, root, index)
    saver = tf.train.Saver(var_list=model.g_variables)
    t0 = time.perf_counter()
    with tf.Session() as sess:
        saver.restore(sess, str(root / CHECKPOINT))
        try:
            ids, lr, hr = sess.run([idx_op, lr_op, hr_op])
        except tf.errors.OutOfRangeError as exc:
            raise RuntimeError(f'Training index {index} not found in {TRAIN_TFRECORD}') from exc
        need(ids.tolist() == [index], f'TFRecord sample index mismatch: {ids}')
        need(lr.shape == (1, 100, 100, 2) and hr.shape == (1, 500, 500, 2),
             f'Unexpected TFRecord shape: LR={lr.shape} HR={hr.shape}')
        vals = sess.run({'sr_crop': crop, 'sr_norm': model.x_SR, 'base_loss': model.g_loss},
                        feed_dict={xi: lr.astype(np.float32), hi: hr.astype(np.float32)})
    elapsed = time.perf_counter() - t0
    after = filehashes(checkpoint_paths)
    need(before == after, 'Checkpoint changed during read-only generator forward')
    pred_speed = np.ascontiguousarray(vals['sr_crop'][0], dtype=np.float32)
    gt_phys = hr[0].astype(np.float32) * SIG + MU
    gt_speed = np.sqrt(np.sum(np.square(gt_phys), axis=-1) + np.float32(1e-8))[:CROP, :CROP]
    need(pred_speed.shape == (CROP, CROP) and np.isfinite(pred_speed).all(), 'Nonfinite SR speed')
    need(np.isfinite(gt_speed).all(), 'Nonfinite GT speed')
    def writer(p):
        np.save(p / 'lr_norm.npy', lr.astype(np.float32))
        np.save(p / 'hr_norm.npy', hr.astype(np.float32))
        np.save(p / 'sr_crop.npy', pred_speed)
        np.save(p / 'gt_crop.npy', np.ascontiguousarray(gt_speed, dtype=np.float32))
        manifest = {
            'schema': 'phase8-tf1-forward-v1', 'status': 'PASS_FORWARD_ONLY',
            'train_sample_index': index, 'source': str(TRAIN_TFRECORD),
            'git_head': rev, 'base_loss_normalized_uv': float(vals['base_loss']),
            'elapsed_seconds': round(elapsed, 6),
            'source_hashes': filehashes([root / s for s in SOURCES] + [CANONICAL]),
            'checkpoint_hashes': before,
            'output_hashes': filehashes([p / f for f in ['lr_norm.npy', 'hr_norm.npy', 'sr_crop.npy', 'gt_crop.npy']]),
            'caveat': 'Pretrained CNN forward only; no dynamic matching yet; no parameter updates.'
        }
        (p / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    atomic_stage(out / 'forward', writer)
    print(f'[FORWARD] PASS: sample={index} pretraining CNN restored, '
          f'normalized L_uv={vals["base_loss"]:.8g}, elapsed={elapsed:.3f}s; '
          f'output={out / "forward"}', flush=True)


def match(root, out, backend, timeout, max_pairs):
    fm = load_forward(root, out)
    import gudhi
    from gudhi.wasserstein import wasserstein_distance as w2
    gate = mod(root / 'phase8_matching_vertex_gradient_gate_20261116.py', 'phase8_tf1_gateg')
    timing = mod(root / 'phase8_ttk_w22_timing_pilot_20261116.py', 'phase8_tf1_timing')
    ph6 = mod(root / 'phase6_lxxvii_update/scripts/phase6_pilot_pd_gudhi_gate.py', 'phase8_tf1_ph6')
    decoder = mod(CANONICAL, 'phase8_tf1_decoder')
    extract = mod(root / 'scripts/extract_ttk_pd_critical_pairs.py', 'phase8_tf1_extract')
    timing.check_backend(backend)
    fields = {n: np.load(out / 'forward' / f'{n}_crop.npy', allow_pickle=False) for n in ('gt', 'sr')}
    for k, v in fields.items():
        need(v.shape == (CROP, CROP) and np.isfinite(v).all(), f'Invalid {k} forward crop')

    def writer(p):
        (p / 'vti').mkdir()
        (p / 'pd').mkdir()
        ttk_times = {}
        for name in ('gt', 'sr'):
            timing.write_vti(p / 'vti' / f'{name}.vti', fields[name])
            ttk_times[name] = timing.run_ttk(backend, p, name, timeout)
        diagrams = {n: ph6.parse(decoder.read_pd, p / 'pd' / f'{n}_pd_port_0.vtu')
                    for n in ('gt', 'sr')}
        raw = gate.raw_pairs(p / 'pd' / 'sr_pd_port_0.vtu', fields['sr'], extract)
        sr_maps, unmapped = gate.align(diagrams['sr'], raw)
        bvid, dvid, targets, modes = [], [], [], []
        gt_diag_const = 0.0
        check_j = 0.0
        expected_grad = np.zeros((CROP, CROP), dtype=np.float64)
        per_dim = {}
        for dim in (0, 1):
            ref = gate.positive(diagrams['gt'][dim])
            sr = gate.positive(diagrams['sr'][dim])
            need(len(ref) + len(sr) <= max_pairs,
                 f'D{dim} pair-count cap exceeded ({len(ref) + len(sr)}>{max_pairs}); '
                 'fail closed; do not prune or change evaluation conventions')
            need(len(sr_maps[dim]) == len(sr), f'D{dim} critical vertex mapping count mismatch')
            t0 = time.perf_counter()
            distance, assignment = w2(ref, sr, order=2., internal_p=2.,
                                      matching=True, keep_essential_parts=False)
            elapsed = time.perf_counter() - t0
            cost, g, counts = gate.matching_obj_and_grad(fields['sr'], ref, sr_maps[dim], assignment)
            need(math.isclose(cost, float(distance)**2, rel_tol=5e-5, abs_tol=1e-4),
                 f'D{dim} matching J differs from GUDHI W22² ({cost} != {distance**2})')
            check_j += cost
            expected_grad += g
            per_dim[f'D{dim}'] = {'n_gt':len(ref), 'n_sr':len(sr),
                                 'distance':float(distance), 'squared_cost':float(cost),
                                 'counts':counts, 'matching_seconds':round(elapsed, 6)}
            seen_pred = set()
            for i, j in np.asarray(assignment, dtype=np.int64).reshape((-1, 2)):
                i, j = int(i), int(j)
                if j < 0:
                    need(i >= 0, 'Diagonal-to-diagonal matching is invalid')
                    gt_diag_const += 0.5 * (float(ref[i,1]) - float(ref[i,0]))**2
                    continue
                need(j not in seen_pred, f'Repeated predicted pair D{dim} index {j}')
                seen_pred.add(j)
                item = sr_maps[dim][j]
                bvid.append(int(item['birth_vid']))
                dvid.append(int(item['death_vid']))
                if i >= 0:
                    modes.append(0)
                    targets.append([float(ref[i,0]), float(ref[i,1])])
                else:
                    modes.append(1)
                    targets.append([0., 0.])
            need(len(seen_pred) == len(sr), f'Incomplete D{dim} predicted pair matching')
        true_distance = float(ph6.distance(diagrams['gt'], diagrams['sr'])['W22'])
        need(math.isclose(check_j, true_distance**2, rel_tol=5e-5, abs_tol=1e-4),
             f'Combined canonical W22 mismatch: {check_j} != {true_distance**2}')
        need(len(bvid)>0, 'No finite predicted positive pairs to supervise; pilot refuses empty loss')
        coord = lambda x: np.stack([np.asarray(x,dtype=np.int32)//CROP,
                                     np.asarray(x,dtype=np.int32)%CROP],axis=1)
        np.savez_compressed(p / 'feed.npz',
            birth_yx=coord(bvid)[None, ...], death_yx=coord(dvid)[None, ...],
            target=np.asarray(targets,dtype=np.float32)[None, ...],
            mode=np.asarray(modes,dtype=np.int32)[None, ...],
            mask=np.ones((1,len(bvid)),dtype=np.float32),
            gt_diagonal_cost=np.asarray(gt_diag_const,dtype=np.float32),
            expected_gradient=expected_grad.astype(np.float32),
            expected_squared_cost=np.asarray(check_j,dtype=np.float64))
        names = ['feed.npz','vti/gt.vti','vti/sr.vti',
                 'pd/gt_pd_port_0.vtu','pd/sr_pd_port_0.vtu']
        record = {'schema':'phase8-tf1-dynamic-match-v1',
                  'status':'PASS_FULL_CANONICAL_MATCHING_NO_TRAIN',
                  'train_sample_index':fm['train_sample_index'],
                  'upstream_manifest_sha256':sha(out/'forward'/'manifest.json'),
                  'source_hashes':filehashes([root/s for s in SOURCES]+[CANONICAL]),
                  'output_hashes':filehashes([p/f for f in names]),
                  'ttk':ttk_times, 'per_dim':per_dim,
                  'canonical_W22':true_distance, 'matching_squared_cost':float(check_j),
                  'gt_diagonal_cost_constant':float(gt_diag_const),
                  'n_pred_pairs':len(bvid), 'unmapped_raw_pairs':unmapped,
                  'caveats':['No TTK/GUDHI backpropagation; fixed-pair-assignment local surrogate.',
                             'Ambiguous real-field critical point mapping causes a deliberate stop.',
                             'No top-K truncation or persistence threshold beyond zero persistence.']}
        (p/'manifest.json').write_text(json.dumps(record,indent=2,sort_keys=True)+'\n')
    atomic_stage(out/'match',writer)
    result=load_match(root,out)
    print(f'[MATCH] PASS: canonical W22={result["canonical_W22"]:.9g} '
          f'J={result["matching_squared_cost"]:.9g} '
          f'pred_pairs={result["n_pred_pairs"]}; output={out/"match"}',flush=True)


def gather_tf(tf, field, coords):
    # Same tf.gather_nd batch/y/x indexing as Candidate F's original helper.
    batch=tf.shape(field)[0]
    n=tf.shape(coords)[1]
    bix=tf.tile(tf.reshape(tf.range(batch), [batch,1,1]), [1,n,1])
    return tf.gather_nd(field, tf.concat([bix,coords],axis=2))


def tfcheck(root, out):
    fm=load_forward(root,out)
    mm=load_match(root,out)
    tf, SR_NETWORK, _denorm_speed, _PhIREGANs = import_tf(root)
    params=np.load(out/'match'/'feed.npz',allow_pickle=False)
    lr=np.load(out/'forward'/'lr_norm.npy',allow_pickle=False)
    hr=np.load(out/'forward'/'hr_norm.npy',allow_pickle=False)
    sr_expected=np.load(out/'forward'/'sr_crop.npy',allow_pickle=False)
    gt_expected=np.load(out/'forward'/'gt_crop.npy',allow_pickle=False)
    before=filehashes(checkpoint_members(root))
    tf.reset_default_graph()
    xlr,xhr,model,speed,speedcrop=make_model(tf,SR_NETWORK,_denorm_speed)
    birth=tf.placeholder(tf.int32,[None,None,2],name='phase8_dynamic_birth_yx')
    death=tf.placeholder(tf.int32,[None,None,2],name='phase8_dynamic_death_yx')
    target=tf.placeholder(tf.float32,[None,None,2],name='phase8_dynamic_target')
    mode=tf.placeholder(tf.int32,[None,None],name='phase8_dynamic_mode')
    mask=tf.placeholder(tf.float32,[None,None],name='phase8_dynamic_mask')
    gtconst=tf.placeholder(tf.float32,shape=[],name='phase8_unmatched_gt_constant')
    b=gather_tf(tf,speedcrop,birth)
    d=gather_tf(tf,speedcrop,death)
    pairmatched=tf.square(b-target[...,0])+tf.square(d-target[...,1])
    pairdiag=0.5*tf.square(d-b)
    per_pair=tf.where(tf.equal(mode,0),pairmatched,pairdiag)
    j=tf.reduce_sum(per_pair*mask)+gtconst  # SUM, not mean: exactly canonical W22²
    grad_field=tf.gradients(j,speedcrop)[0]
    grad_params=[g for g in tf.gradients(j,model.g_variables) if g is not None]
    need(grad_field is not None and len(grad_params)>0,'Missing TF gradients to model')
    weight_grad_norm=tf.global_norm(grad_params)
    saver=tf.train.Saver(var_list=model.g_variables)
    feed={xlr:lr,xhr:hr,birth:params['birth_yx'],death:params['death_yx'],
          target:params['target'],mode:params['mode'],mask:params['mask'],
          gtconst:float(params['gt_diagonal_cost'])}
    t0=time.perf_counter()
    with tf.Session() as sess:
        saver.restore(sess,str(root/CHECKPOINT))
        v=sess.run({'sr_crop':speedcrop,'gt_speed':_denorm_speed(xhr,tf.constant(MU),tf.constant(SIG))[:, :CROP, :CROP],
                    'J':j,'grad_field':grad_field,'grad_weight_norm':weight_grad_norm,
                    'uv_loss':model.g_loss},feed_dict=feed)
    elapsed=time.perf_counter()-t0
    need(before==filehashes(checkpoint_members(root)), 'Pretrained checkpoint was modified')
    field_diff=float(np.max(np.abs(v['sr_crop'][0]-sr_expected)))
    gt_diff=float(np.max(np.abs(v['gt_speed'][0]-gt_expected)))
    need(field_diff<1e-4 and gt_diff<1e-4,
         f'Forward changed since TTK extraction: SR max abs={field_diff}, GT max abs={gt_diff}')
    J=float(v['J']);expected=float(params['expected_squared_cost'])
    need(math.isclose(J,expected,rel_tol=3e-4,abs_tol=1e-3),
         f'TF local surrogate J={J:.9g} differs from canonical GUDHI²={expected:.9g}')
    g=v['grad_field'][0]
    gref=params['expected_gradient']
    maxdiff=float(np.max(np.abs(g-gref)))
    need(np.allclose(g,gref,rtol=3e-3,atol=3e-3),
         f'TF dynamic field gradient mismatch max abs={maxdiff}')
    wnorm=float(v['grad_weight_norm'])
    need(math.isfinite(wnorm) and wnorm>0, f'Generator parameter gradient absent/invalid: {wnorm}')
    def writer(p):
        report={'schema':'phase8-tf1-gradient-check-v1',
                'status':'PASS_FORWARD_MATCH_GATHER_GRAD_NO_TRAIN',
                'upstream_match_sha256':sha(out/'match'/'manifest.json'),
                'upstream_forward_sha256':sha(out/'forward'/'manifest.json'),
                'source_hashes':filehashes([root/s for s in SOURCES]+[CANONICAL]),
                'checkpoint_hashes':before,
                'sample_index':fm['train_sample_index'],
                'tf_J':J,'gudhi_W22_squared':expected,'W22':mm['canonical_W22'],
                'max_abs_tf_vs_numpy_grad':maxdiff,
                'max_abs_pred_speed_change':field_diff,
                'max_abs_gt_speed_change':gt_diff,
                'generator_weight_grad_global_norm':wnorm,
                'normalized_uv_loss':float(v['uv_loss']),
                'tf_to_normalized_uv_loss_ratio':J/max(float(v['uv_loss']),1e-12),
                'elapsed_seconds':round(elapsed,6),
                'no_optimizer':True,'no_weights_written':True,
                'caveat':'Passing validates a frozen, locally matched surrogate gradient; no optimization improvement claim.'}
        (p/'result.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    atomic_stage(out/'tfcheck',writer)
    print(f'[TF-GRAD] PASS: J={J:.9g}, W22²={expected:.9g}, '
          f'field_gradient_max_diff={maxdiff:.3g}, '
          f'generator_parameter_gradient_norm={wnorm:.6g}; '
          f'NO optimizer or training; output={out/"tfcheck"}',flush=True)


def verify(root,out):
    fm=load_forward(root,out)
    mm=load_match(root,out)
    need(fm['train_sample_index']==mm['train_sample_index'], 'Training sample mismatch')
    p=out/'tfcheck'
    need(p.is_dir(),f'Run --mode tfcheck first: {p}')
    r=json.loads((p/'result.json').read_text())
    need(r['schema']=='phase8-tf1-gradient-check-v1' and
         r['status']=='PASS_FORWARD_MATCH_GATHER_GRAD_NO_TRAIN' and
         r['no_optimizer'] is True and r['no_weights_written'] is True,
         'Invalid recorded verification status')
    check_hashes(root,r['source_hashes'])
    check_hashes(root,r['checkpoint_hashes'])
    need(sha(out/'forward'/'manifest.json')==r['upstream_forward_sha256'] and
         sha(out/'match'/'manifest.json')==r['upstream_match_sha256'],
         'Upstream manifests changed')
    params=np.load(out/'match'/'feed.npz',allow_pickle=False)
    need(math.isclose(float(params['expected_squared_cost']),r['gudhi_W22_squared'],rel_tol=1e-10),
         'Recorded matching squared-cost mismatch')
    need(math.isclose(r['tf_J'],r['gudhi_W22_squared'],rel_tol=3e-4,abs_tol=1e-3),
         'Recorded TF/GUDHI result mismatch')
    print(f'[VERIFY] PASS: frozen checkpoint + source hashes, canonical matching, TF gather gradient, '
          f'no training; {out}',flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode',required=True,
                    choices=['selftest','preflight','forward','match','tfcheck','verify'])
    ap.add_argument('--repo-root',type=Path,default=Path.cwd())
    ap.add_argument('--sample-index',type=int,default=0)
    ap.add_argument('--out-dir',type=Path,default=OUTPUT)
    ap.add_argument('--ttk-backend',choices=['docker','host'],default='docker')
    ap.add_argument('--ttk-timeout',type=int,default=180)
    ap.add_argument('--max-matching-points',type=int,default=5000)
    a=ap.parse_args();root=a.repo_root.resolve()
    out=(root/a.out_dir).resolve()
    if a.mode=='selftest':selftest();return
    preflight(root,a.sample_index)
    if a.mode=='preflight':
        need(not out.exists(),f'New output directory already exists: {out}')
        print('[PREFLIGHT] PASS: no write, no held-out data, no existing output',flush=True)
    elif a.mode=='forward': forward(root,out,a.sample_index)
    elif a.mode=='match': match(root,out,a.ttk_backend,a.ttk_timeout,a.max_matching_points)
    elif a.mode=='tfcheck':tfcheck(root,out)
    else: verify(root,out)


if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        print(f'[PHASE8-TF-BRIDGE] ERROR {type(exc).__name__}: {exc}',file=sys.stderr,flush=True)
        sys.exit(1)
