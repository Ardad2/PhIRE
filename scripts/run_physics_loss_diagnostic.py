#!/usr/bin/env python3
"""Physics/topology loss diagnostic — no training, no checkpoint modification.

Loads the pretrained CNN checkpoint, feeds one batch through the forward pass,
and reports every individual loss term together with what each weighted
contribution would be under a user-supplied lambda configuration.

The checkpoint and model weights are NEVER modified.

Usage (on Spark, from the repo root):
    python3 scripts/run_physics_loss_diagnostic.py \\
        --data-path  example_data_fixed/wind_MR-HR.tfrecord \\
        --model-path models/wind_mr-hr/trained_cnn/cnn \\
        --batch-size 4

Optional lambda overrides (all default 0, so total loss == L_uv):
    --lambda-speed    0.1
    --lambda-grad     0.05
    --lambda-wpd      0.0
    --lambda-levelset 0.01
    --levelset-temperature 10.0
    --levelset-thresholds  5.0 10.0 15.0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure repo root is on sys.path when called as scripts/...
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT   = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from sr_network import SR_NETWORK, PhysicsLossConfig
from PhIREGANs import PhIREGANs

# ---------------------------------------------------------------------------
# Vector [u,v] normalization constants for the paper-consistent model.
# Override with --mu / --sig if needed.
# ---------------------------------------------------------------------------
DEFAULT_MU  = [0.7684, -0.4575]
DEFAULT_SIG = [5.02455, 5.9017]
DEFAULT_R   = [5]


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument('--data-path',  required=True,
                    help='Path to TFRecord (e.g. example_data_fixed/wind_MR-HR.tfrecord)')
    ap.add_argument('--model-path', required=True,
                    help='Path to CNN checkpoint (e.g. models/wind_mr-hr/trained_cnn/cnn)')
    ap.add_argument('--batch-size', type=int, default=4,
                    help='Number of samples per diagnostic batch (default 4)')
    ap.add_argument('--r', type=int, nargs='+', default=DEFAULT_R,
                    help='SR scaling factors (default 5)')
    ap.add_argument('--mu',  type=float, nargs='+', default=DEFAULT_MU,
                    help='Per-channel normalisation means (default: vector model)')
    ap.add_argument('--sig', type=float, nargs='+', default=DEFAULT_SIG,
                    help='Per-channel normalisation stds  (default: vector model)')
    # Lambda knobs — all default 0 so weighted total == L_uv
    ap.add_argument('--lambda-speed',    type=float, default=0.0)
    ap.add_argument('--lambda-grad',     type=float, default=0.0)
    ap.add_argument('--lambda-wpd',      type=float, default=0.0)
    ap.add_argument('--lambda-levelset', type=float, default=0.0)
    ap.add_argument('--levelset-temperature', type=float, default=10.0)
    ap.add_argument('--levelset-thresholds',  type=float, nargs='+',
                    default=[5.0, 10.0, 15.0],
                    help='Physical speed thresholds in m/s (default: 5 10 15)')
    return ap.parse_args()


def main() -> None:
    args = _parse_args()

    data_path  = str(args.data_path)
    model_path = str(args.model_path)

    # Verify files exist before building the graph.
    if not Path(data_path).exists():
        sys.exit(f'[error] TFRecord not found: {data_path}')
    for ext in ('.index', '.meta'):
        if not Path(model_path + ext).exists():
            sys.exit(f'[error] Checkpoint file not found: {model_path}{ext}')

    mu_sig = [args.mu, args.sig]

    # Build diagnostic PhysicsLossConfig with diagnostic_mode=True.
    # All lambdas default to 0 unless overridden — the total loss then equals
    # L_uv, matching baseline behaviour numerically, while still building
    # all aux loss tensors for inspection.
    phys_cfg = PhysicsLossConfig(
        use_aux_losses       = True,
        mu                   = args.mu,
        sig                  = args.sig,
        lambda_speed         = args.lambda_speed,
        lambda_grad          = args.lambda_grad,
        lambda_wpd           = args.lambda_wpd,
        lambda_levelset      = args.lambda_levelset,
        levelset_temperature = args.levelset_temperature,
        levelset_thresholds  = args.levelset_thresholds,
        diagnostic_mode      = True,
    )

    print('=' * 64)
    print('PhIRE physics/topology loss diagnostic')
    print('=' * 64)
    print(f'  data_path  : {data_path}')
    print(f'  model_path : {model_path}')
    print(f'  batch_size : {args.batch_size}')
    print(f'  r          : {args.r}')
    print(f'  mu         : {args.mu}')
    print(f'  sig        : {args.sig}')
    print(f'  lambdas    : speed={args.lambda_speed}  grad={args.lambda_grad}'
          f'  wpd={args.lambda_wpd}  levelset={args.lambda_levelset}')
    print(f'  levelset k : {args.levelset_temperature}')
    print(f'  thresholds : {args.levelset_thresholds} m/s')
    print()

    tf.reset_default_graph()

    # -----------------------------------------------------------------------
    # Determine LR shape from TFRecord (reuse PhIREGANs helper).
    # -----------------------------------------------------------------------
    phire = PhIREGANs(data_type='wind_diagnostic', mu_sig=mu_sig)
    phire.set_LR_data_shape(data_path)
    h, w, C = phire.LR_data_shape

    print(f'LR shape: h={h}  w={w}  C={C}')
    r = args.r

    # -----------------------------------------------------------------------
    # Build graph in pretraining mode with diagnostic phys_cfg.
    # NO optimizer is created — this is read-only.
    # -----------------------------------------------------------------------
    print('Building graph (pretraining + diagnostic) ...', end=' ', flush=True)
    x_LR = tf.placeholder(tf.float32, [None, h,            w,           C])
    x_HR = tf.placeholder(tf.float32, [None, h*np.prod(r), w*np.prod(r), C])

    model = SR_NETWORK(x_LR, x_HR, r=r, status='pretraining', phys_cfg=phys_cfg)

    if model.loss_breakdown is None:
        sys.exit('[error] loss_breakdown not built — check phys_cfg.diagnostic_mode')

    init     = tf.global_variables_initializer()
    g_saver  = tf.train.Saver(var_list=model.g_variables)
    print('Done.')

    # -----------------------------------------------------------------------
    # Data pipeline — one batch only.
    # -----------------------------------------------------------------------
    ds = tf.data.TFRecordDataset(data_path)
    ds = ds.map(lambda xx: phire._parse_train_(xx, mu_sig)).batch(args.batch_size)

    iterator  = tf.data.Iterator.from_structure(ds.output_types, ds.output_shapes)
    idx_out, LR_out, HR_out = iterator.get_next()
    init_iter = iterator.make_initializer(ds)

    # -----------------------------------------------------------------------
    # Session: restore checkpoint, run one forward pass, fetch diagnostics.
    # -----------------------------------------------------------------------
    with tf.Session() as sess:
        sess.run(init)

        print(f'Restoring checkpoint: {model_path} ...', end=' ', flush=True)
        g_saver.restore(sess, model_path)
        print('Done.')

        sess.run(init_iter)
        batch_idx, batch_LR, batch_HR = sess.run([idx_out, LR_out, HR_out])
        print(f'Loaded batch: indices={batch_idx.tolist()}')
        print()

        feed_dict = {x_LR: batch_LR, x_HR: batch_HR}

        # Fetch all breakdown tensors in one sess.run call (no training step).
        bv = sess.run(model.loss_breakdown, feed_dict=feed_dict)

    # -----------------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------------
    print('=' * 64)
    print('Loss breakdown (one batch, no training step)')
    print('=' * 64)

    _RAW  = ('L_uv', 'L_speed', 'L_grad', 'L_wpd', 'L_levelset')
    _WGT  = ('w_L_speed', 'w_L_grad', 'w_L_wpd', 'w_L_levelset')
    _TOT  = ('total_loss',)

    print('\nRaw (unweighted) loss terms:')
    for k in _RAW:
        if k in bv:
            print(f'  {k:<18s} {bv[k]:>14.6f}')

    print('\nWeighted contributions (lambda * raw):')
    lambdas = {
        'w_L_speed':    args.lambda_speed,
        'w_L_grad':     args.lambda_grad,
        'w_L_wpd':      args.lambda_wpd,
        'w_L_levelset': args.lambda_levelset,
    }
    for k in _WGT:
        if k in bv:
            lam = lambdas.get(k, 0.0)
            print(f'  {k:<18s} {bv[k]:>14.6f}  (lambda={lam})')

    print()
    for k in _TOT:
        if k in bv:
            print(f'  {k:<18s} {bv[k]:>14.6f}')

    print()
    print('Notes:')
    print('  - All raw losses are in physical units (m/s or m^2/s^2).')
    print('  - total_loss == L_uv when all lambdas are 0.')
    print('  - No checkpoint was modified.')
    print('=' * 64)


if __name__ == '__main__':
    main()
