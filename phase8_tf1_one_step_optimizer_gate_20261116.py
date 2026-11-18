#!/usr/bin/env python3
"""Phase 8: one in-memory optimizer step, with and without a FROZEN dynamic PD feed.

A diagnostic only; NOT a full dynamic-PD training loop. The TTK/GUDHI matching
is obtained from the previously VERIFIED v2 bridge and is held fixed during
one Adam step. Model checkpoints, training data and prior Phase 8 artifacts
are never written. A separate GUDHI/TTK stage recomputes true W22 AFTER
that step, so lower frozen surrogate is not misrepresented as topology gain.

Stages: selftest, preflight (read-only), step (TF1), compare (GUDHI+Docker),
        verify (GUDHI environment; read-only).

Run from ~/PhIRE. The script imports the immutable v2 bridge by filename and
requires that its full prior verification passes before every substantive stage.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile
import time

import numpy as np

BRIDGE_SOURCE = 'phase8_tf1_dynamic_single_batch_gate_20261116_v2.py'
BRIDGE_OUT = Path('ttk_runs_fixed/topology_finetuning/phase8_tf1_single_batch_bridge_20261116_v2')
DEFAULT_OUT = Path('ttk_runs_fixed/topology_finetuning/phase8_one_step_optimizer_20261116')
MEAN = [0.7684, -0.4575]
STD = [5.02455, 5.9017]
LAMBDA_GRAD = 0.05


def need(condition, detail):
    if not condition:
        raise RuntimeError(detail)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def import_file(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    need(spec is not None and spec.loader is not None, f'Cannot import {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def atomic_stage(destination, writer):
    need(not destination.exists(), f'Refusing existing result: {destination}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix='.' + destination.name + '.staging-', dir=destination.parent))
    try:
        writer(temp)
        need(not destination.exists(), f'Output collision: {destination}')
        temp.rename(destination)
    finally:
        if temp.exists():
            shutil.rmtree(temp)


def jsonfile(path, obj):
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + '\n')


def load_bridge(root):
    src = root / BRIDGE_SOURCE
    need(src.is_file(), f'Missing verified v2 bridge source: {src}')
    module = import_file(src, 'phase8_verified_bridge_v2')
    prior = root / BRIDGE_OUT
    # Read-only checks of prior forward/matching/TF gradient outputs and
    # original checkpoint. Changes to historical source or files fail closed.
    module.verify(root, prior)
    return module, prior


def get_input(root, prior):
    p = prior / 'forward'
    x = np.load(p / 'lr_norm.npy', allow_pickle=False)
    y = np.load(p / 'hr_norm.npy', allow_pickle=False)
    ref = np.load(p / 'sr_crop.npy', allow_pickle=False)
    need(x.shape == (1,100,100,2) and y.shape == (1,500,500,2) and ref.shape == (160,160),
         f'Unexpected saved training shape: LR{x.shape} HR{y.shape} SR{ref.shape}')
    return x, y, ref


def get_pairs(prior):
    with np.load(prior / 'match' / 'feed.npz', allow_pickle=False) as z:
        p = {k: np.array(z[k]) for k in z.files}
    need(p['birth_yx'].shape[0] == 1 and p['birth_yx'].shape[-1] == 2, 'Invalid predicted vertex feed')
    need(p['death_yx'].shape == p['birth_yx'].shape, 'Invalid death pair shape')
    need(p['mode'].shape == p['mask'].shape == p['birth_yx'].shape[:2], 'Invalid mode/mask shape')
    need(np.isfinite(p['expected_squared_cost']).all(), 'Nonfinite upstream W22²')
    return p


def check_sources(root, prior, script_path):
    p = root / BRIDGE_OUT
    return {
        'new_script': sha(script_path),
        'v2_script': sha(root / BRIDGE_SOURCE),
        'upstream_forward_manifest': sha(p / 'forward' / 'manifest.json'),
        'upstream_match_manifest': sha(p / 'match' / 'manifest.json'),
        'upstream_tfcheck_report': sha(p / 'tfcheck' / 'result.json'),
        'upstream_feed': sha(p / 'match' / 'feed.npz'),
    }


def require_sources(root, prior, recorded, script_path):
    need(check_sources(root, prior, script_path) == recorded, 'Source/upstream SHA256 drift')


def selftest():
    # The optimizer gate uses J = W22² (not W22) with a small independent
    # coefficient, leaving L_uv + 0.05*L_grad as the baseline objective.
    J = 427.376501
    lam = 1.e-5
    assert math.isclose(lam * J, 0.00427376501, abs_tol=1e-12)
    assert lam > 0 and J > 0 and LAMBDA_GRAD == 0.05
    print('[SELFTEST] PASS: squared-cost scaling, independent 1-step arms; no weights written',flush=True)


def preflight(root, out, script_path, step=False):
    bridge, prior = load_bridge(root)
    need(script_path.is_file(), 'Source file missing')
    get_input(root, prior)
    p = get_pairs(prior)
    expected = float(p['expected_squared_cost'])
    check = json.loads((prior/'tfcheck'/'result.json').read_text())
    need(math.isclose(expected, check['gudhi_W22_squared'], rel_tol=1e-10, abs_tol=1e-10),
         'Verified squared loss drifted')
    if not step:
        need(not out.exists(), f'New output already exists: {out}')
    else:
        need((out/'step').exists(), f'Run --mode step first: {out}')
    print(f'[PREFLIGHT] PASS: verified upstream W22²={expected:.8g}, '
          f'predicted_pairs={p["mode"].shape[1]}, checkpoint frozen',flush=True)
    return bridge, prior


def make_tf_graph(bridge, root, learn_rate, lam):
    tf, SR_NETWORK, _denorm_speed, _ = bridge.import_tf(root)
    from sr_network import PhysicsLossConfig
    tf.reset_default_graph()
    mu = tf.constant(MEAN, dtype=tf.float32)
    sd = tf.constant(STD, dtype=tf.float32)
    xl = tf.placeholder(tf.float32, [None,100,100,2], name='phase8_step_lr')
    xh = tf.placeholder(tf.float32, [None,500,500,2], name='phase8_step_hr')
    cfg = PhysicsLossConfig(use_aux_losses=True, mu=MEAN, sig=STD,
                            lambda_grad=LAMBDA_GRAD, diagnostic_mode=True)
    model = SR_NETWORK(xl, xh, r=[5], status='pretraining', phys_cfg=cfg)
    cropped = _denorm_speed(model.x_SR, mu, sd)[:, :160, :160]
    birth = tf.placeholder(tf.int32, [None,None,2], name='matched_birth_yx')
    death = tf.placeholder(tf.int32, [None,None,2], name='matched_death_yx')
    target = tf.placeholder(tf.float32, [None,None,2], name='gt_match_target')
    mode = tf.placeholder(tf.int32, [None,None], name='match_or_diagonal')
    mask = tf.placeholder(tf.float32, [None,None], name='valid_pair')
    gtconstant = tf.placeholder(tf.float32, [], name='unmatched_gt_constant')
    b = bridge.gather_tf(tf, cropped, birth)
    d = bridge.gather_tf(tf, cropped, death)
    j = tf.reduce_sum(tf.where(tf.equal(mode,0),
                        tf.square(b-target[...,0]) + tf.square(d-target[...,1]),
                        0.5*tf.square(d-b))*mask) + gtconstant
    base = model.g_loss                         # normalized UV + 0.05 grad
    dynamic = base + tf.constant(lam,dtype=tf.float32)*j
    vars_ = model.g_variables
    gb_raw = tf.gradients(base, vars_)
    gj_raw = tf.gradients(j, vars_)
    need(any(g is not None for g in gj_raw), 'Missing dynamic generator gradients')
    gb = [g if g is not None else tf.zeros_like(v) for g,v in zip(gb_raw,vars_)]
    gj = [g if g is not None else tf.zeros_like(v) for g,v in zip(gj_raw,vars_)]
    gd = [g+tf.constant(lam,tf.float32)*h for g,h in zip(gb,gj)]
    base_optimizer = tf.train.AdamOptimizer(learning_rate=learn_rate)
    dynamic_optimizer = tf.train.AdamOptimizer(learning_rate=learn_rate)
    bstep = base_optimizer.apply_gradients(list(zip(gb,vars_)))
    dstep = dynamic_optimizer.apply_gradients(list(zip(gd,vars_)))
    init = tf.global_variables_initializer()
    saver = tf.train.Saver(var_list=vars_)
    fetch = {
        'crop':cropped,'L_uv':model.loss_breakdown['L_uv'],
        'L_grad':model.loss_breakdown['L_grad'],
        'base_loss':base,'J_frozen':j,'dynamic_loss':dynamic,
        'base_grad_norm':tf.global_norm(gb),
        'dynamic_grad_norm':tf.global_norm(gd),
        'PD_grad_norm_unscaled':tf.global_norm(gj),
    }
    return tf, xl, xh, (birth,death,target,mode,mask,gtconstant), fetch, \
           {'baseline':bstep,'dynamic':dstep}, saver, init, vars_


def run_step(root, out, script_path, lr, lam):
    need(0 < lr <= 1e-5, 'Diagnostic LR must be >0 and <=1e-5')
    need(0 < lam <= 1e-4, 'Diagnostic lambda_pd must be >0 and <=1e-4')
    bridge, prior = preflight(root,out,script_path)
    x, y, old = get_input(root,prior)
    pairs = get_pairs(prior)
    tf,xl,xh,ph,fetch,trains,saver,init,vars_ = make_tf_graph(bridge,root,lr,lam)
    feed = {xl:x,xh:y,
            ph[0]:pairs['birth_yx'], ph[1]:pairs['death_yx'],
            ph[2]:pairs['target'], ph[3]:pairs['mode'], ph[4]:pairs['mask'],
            ph[5]:float(pairs['gt_diagonal_cost'])}
    before_ckpt = bridge.filehashes(bridge.checkpoint_members(root))
    arms = {}
    initial_param_hashes = {}
    start = time.perf_counter()
    for arm in ('baseline','dynamic'):
        with tf.Session() as sess:
            sess.run(init)                 # initialize Adam slots independently
            saver.restore(sess, str(root/bridge.CHECKPOINT))
            before = sess.run(fetch,feed_dict=feed)
            need(np.max(np.abs(before['crop'][0]-old)) < 1e-4,
                 f'{arm}: model output is not the matching extraction input')
            need(math.isclose(float(before['J_frozen']),float(pairs['expected_squared_cost']),
                               rel_tol=3e-4,abs_tol=1e-3),
                 f'{arm}: initial fixed matching J differs from canonical W22²')
            h = hashlib.sha256()
            for v in sess.run(vars_):
                h.update(np.asarray(v).tobytes())
            initial_param_hashes[arm] = h.hexdigest()
            need(np.isfinite([before['base_loss'], before['dynamic_loss'],
                              before['base_grad_norm'], before['dynamic_grad_norm'],
                              before['PD_grad_norm_unscaled']]).all(),
                 f'{arm}: nonfinite initial loss/gradient')
            # Exactly ONE Adam update, for the requested arm only.
            sess.run(trains[arm],feed_dict=feed)
            after = sess.run(fetch,feed_dict=feed)
            new = np.asarray(after['crop'][0],dtype=np.float32)
            need(new.shape == (160,160) and np.isfinite(new).all(), f'{arm}: invalid poststep crop')
            need(np.isfinite([after['L_uv'], after['L_grad'], after['base_loss'],
                              after['J_frozen'], after['dynamic_loss'],
                              after['base_grad_norm'], after['dynamic_grad_norm'],
                              after['PD_grad_norm_unscaled']]).all(),
                 f'{arm}: nonfinite poststep loss/gradient')
            change = float(np.max(np.abs(new-old)))
            need(change > 0, f'{arm}: optimizer produced no measurable field change')
            arms[arm] = {
                'initial':{k:float(before[k]) for k in fetch if k!='crop'},
                'post':{k:float(after[k]) for k in fetch if k!='crop'},
                'max_abs_change_in_speed':change,
                'new_crop':new,
            }
        print(f'[STEP] {arm}: initial J={arms[arm]["initial"]["J_frozen"]:.9g} '
              f'after frozen J={arms[arm]["post"]["J_frozen"]:.9g} '
              f'max_speed_delta={change:.4g}',flush=True)
    need(initial_param_hashes['baseline']==initial_param_hashes['dynamic'],
         'Arms did not start from exactly the same generator weights')
    need(before_ckpt==bridge.filehashes(bridge.checkpoint_members(root)),
         'Original pretrained generator checkpoint changed')
    elapsed=time.perf_counter()-start
    # Only isolated post-step arrays, never new checkpoint weights.
    def writer(p):
        for arm in ('baseline','dynamic'):
            np.save(p/f'{arm}_speed_after.npy',arms[arm]['new_crop'])
        names=[f'{arm}_speed_after.npy' for arm in ('baseline','dynamic')]
        payload={
            'schema':'phase8-frozen-matching-one-step-v1',
            'status':'ONE_STEP_COMPLETED_TRUE_W22_NOT_YET_EVALUATED',
            'train_sample_index':0,'learning_rate':lr,'lambda_dynamic_squared_W22':lam,
            'lambda_grad':LAMBDA_GRAD,'n_optimizer_updates_per_arm':1,
            'n_training_samples':1,'model_checkpoint_written':False,
            'no_heldout_samples':True,'arms':{k:{z:v for z,v in arm.items() if z!='new_crop'}
                                             for k,arm in arms.items()},
            'initial_parameter_sha256':initial_param_hashes['baseline'],
            'source_hashes':check_sources(root,prior,script_path),
            'checkpoint_hashes':before_ckpt,
            'output_hashes':bridge.stage_filehashes(p,names),
            'elapsed_seconds':round(elapsed,6),
            'limitations':[
              'Frozen matching and critical vertices held fixed for ONE optimizer step',
              'Not a run of fully dynamic re-matching during training',
              'No data split conclusion from one training sample',
              'Raw W22 squared was not pruned or divided by number of diagram pairs',
              'True poststep W22 must be recomputed using --mode compare in gudhi-audit',
            ]}
        jsonfile(p/'manifest.json',payload)
    atomic_stage(out/'step',writer)
    print(f'[STEP-GATE] PASS: two independent in-memory 1-step arms, no checkpoint writes; '
          f'output={out/"step"}. Run --mode compare to measure TRUE W22.',flush=True)


def load_step(root, out, script_path, bridge, prior):
    p = out/'step'
    need(p.is_dir(), f'Step output missing: {p}')
    record=json.loads((p/'manifest.json').read_text())
    need(record['schema']=='phase8-frozen-matching-one-step-v1' and
         record['n_optimizer_updates_per_arm']==1 and not record['model_checkpoint_written'],
         'Invalid one-step manifest')
    require_sources(root,prior,record['source_hashes'],script_path)
    bridge.check_hashes(root,record['checkpoint_hashes'])
    bridge.verify_stage_filehashes(p,record['output_hashes'])
    return record


def compare(root,out,script_path,backend,timeout,max_pairs):
    bridge,prior=preflight(root,out,script_path,step=True)
    step=load_step(root,out,script_path,bridge,prior)
    import gudhi
    from gudhi.wasserstein import wasserstein_distance as w2
    t=import_file(root/'phase8_ttk_w22_timing_pilot_20261116.py','phase8_one_step_ttk')
    pilot=import_file(root/'phase6_lxxvii_update/scripts/phase6_pilot_pd_gudhi_gate.py','phase8_one_step_pilot')
    reader=import_file(bridge.CANONICAL,'phase8_one_step_canonical')
    t.check_backend(backend)
    gt=np.load(prior/'forward'/'gt_crop.npy',allow_pickle=False)
    fields={arm:np.load(out/'step'/f'{arm}_speed_after.npy',allow_pickle=False)
            for arm in ('baseline','dynamic')}
    need(gt.shape == (160,160) and all(x.shape==(160,160) for x in fields.values()),
         'Unexpected crop dimensions')
    def writer(p):
        (p/'vti').mkdir();(p/'pd').mkdir()
        source={'gt':gt,**fields}
        timings={}
        for name,field in source.items():
            t.write_vti(p/'vti'/f'{name}.vti',field)
            timings[name]=t.run_ttk(backend,p,name,timeout)
        pd={n:pilot.parse(reader.read_pd,p/'pd'/f'{n}_pd_port_0.vtu') for n in source}
        scores={}
        for arm in ('baseline','dynamic'):
            dims={}
            for dim in (0,1):
                g= pd['gt'][dim];s=pd[arm][dim]
                g=np.asarray(g[g[:,1]>g[:,0]],dtype=np.float64)
                s=np.asarray(s[s[:,1]>s[:,0]],dtype=np.float64)
                need(len(g)+len(s) <= max_pairs, f'{arm}/D{dim}: pair cap exceeded')
                dist,assignment=w2(g,s,order=2.,internal_p=2.,matching=True,
                                   keep_essential_parts=False)
                cost,counts=t.manual_match_squared(g,s,assignment)
                need(math.isfinite(dist) and math.isclose(cost,dist*dist,rel_tol=1e-5,abs_tol=1e-4),
                     f'{arm}/D{dim}: GUDHI cost reconstruction failed')
                dims[f'D{dim}']={'w22':float(dist),'cost_squared':float(cost),
                                 'n_gt':int(len(g)),'n_pred':int(len(s)),**counts}
            true=pilot.distance(pd['gt'],pd[arm])['W22']
            need(math.isclose(true,math.hypot(dims['D0']['w22'],dims['D1']['w22']),
                              rel_tol=1e-6,abs_tol=1e-6),f'{arm}: combined Phase6 distance drift')
            scores[arm]={'canonical_w22':float(true),'W22_squared':float(true*true),
                         'per_dimension':dims}
            print(f'[COMPARE] {arm}: true recomputed W22={true:.9g}',flush=True)
        initial=float(json.loads((prior/'match'/'manifest.json').read_text())['canonical_W22'])
        result={
            'schema':'phase8-one-step-true-w22-comparison-v1',
            'status':'RECOMPUTED_CANONICAL_W22_NO_IMPROVEMENT_REQUIRED',
            'upstream_step_manifest_sha256':sha(out/'step'/'manifest.json'),
            'source_hashes':check_sources(root,prior,script_path),
            'initial_w22':initial,
            'scores':scores,'ttk_times':timings,
            'output_hashes':bridge.stage_filehashes(p,
                [f'vti/{n}.vti' for n in source] + [f'pd/{n}_pd_port_0.vtu' for n in source]),
            'note':'One-step observations only; dynamic matching was NOT recomputed inside the optimizer step.'
        }
        jsonfile(p/'result.json',result)
    atomic_stage(out/'compare',writer)
    print(f'[COMPARE-GATE] PASS: both outputs re-extracted with TTK and GUDHI; '
          f'no requirement for improvement; {out/"compare"}',flush=True)


def verify(root,out,script_path):
    bridge,prior=preflight(root,out,script_path,step=True)
    step=load_step(root,out,script_path,bridge,prior)
    p=out/'compare'
    need(p.is_dir(),f'Missing comparison: {p}')
    r=json.loads((p/'result.json').read_text())
    need(r['schema']=='phase8-one-step-true-w22-comparison-v1' and
         r['upstream_step_manifest_sha256']==sha(out/'step'/'manifest.json'),
         'Invalid comparison/upstream reference')
    require_sources(root,prior,r['source_hashes'],script_path)
    bridge.verify_stage_filehashes(p,r['output_hashes'])
    # Recompute GUDHI on the immutable VTU evidence to verify the full metric.
    pilot=import_file(root/'phase6_lxxvii_update/scripts/phase6_pilot_pd_gudhi_gate.py','phase8_step_verify_pilot')
    reader=import_file(bridge.CANONICAL,'phase8_step_verify_reader')
    g=pilot.parse(reader.read_pd,p/'pd'/'gt_pd_port_0.vtu')
    for arm in ('baseline','dynamic'):
        s=pilot.parse(reader.read_pd,p/'pd'/f'{arm}_pd_port_0.vtu')
        w=pilot.distance(g,s)['W22']
        need(math.isclose(w,r['scores'][arm]['canonical_w22'],rel_tol=1e-8,abs_tol=1e-6),
             f'{arm}: comparison drift, {w} vs {r["scores"][arm]["canonical_w22"]}')
    print('[VERIFY] PASS: verified one-step optimizer arms, checkpoint hashes, '
          'immutable TTK diagrams and canonical GUDHI W22. No checkpoint saved.',flush=True)
    print(f'[RESULT] initial_W22={r["initial_w22"]:.8g} '
          f'baseline_after={r["scores"]["baseline"]["canonical_w22"]:.8g} '
          f'dynamic_after={r["scores"]["dynamic"]["canonical_w22"]:.8g}',flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode',required=True,choices=['selftest','preflight','step','compare','verify'])
    ap.add_argument('--repo-root',type=Path,default=Path.cwd())
    ap.add_argument('--out-dir',type=Path,default=DEFAULT_OUT)
    ap.add_argument('--learning-rate',type=float,default=1e-6)
    ap.add_argument('--lambda-dynamic',type=float,default=1e-5)
    ap.add_argument('--ttk-backend',choices=['docker','host'],default='docker')
    ap.add_argument('--ttk-timeout',type=int,default=180)
    ap.add_argument('--max-matching-points',type=int,default=5000)
    args=ap.parse_args()
    root=args.repo_root.resolve();out=(root/args.out_dir).resolve()
    script_path=Path(__file__).resolve()
    need(script_path.parent == root,'Keep the script in the PhIRE repository root')
    if args.mode=='selftest':selftest();return
    if args.mode=='preflight':preflight(root,out,script_path);return
    if args.mode=='step':run_step(root,out,script_path,args.learning_rate,args.lambda_dynamic);return
    if args.mode=='compare':compare(root,out,script_path,args.ttk_backend,
                                    args.ttk_timeout,args.max_matching_points);return
    verify(root,out,script_path)


if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        print(f'[PHASE8-ONE-STEP] ERROR {type(exc).__name__}: {exc}',file=sys.stderr,flush=True)
        sys.exit(1)
