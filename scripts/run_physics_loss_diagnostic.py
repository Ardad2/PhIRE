#!/usr/bin/env python3
"""Physics/topology loss calibration diagnostic — no training, no checkpoint modification.

Loads the pretrained CNN checkpoint, feeds batches through the forward pass,
and reports every individual loss term together with summary statistics and
candidate lambda configurations suitable for physics-aware fine-tuning.

The checkpoint and model weights are NEVER modified.

Usage (on Spark, from the repo root):
    python3 scripts/run_physics_loss_diagnostic.py \\
        --data-path  example_data_fixed/wind_MR-HR.tfrecord \\
        --model-path models/wind_mr-hr/trained_cnn/cnn

Optional arguments:
    --batch-size N          Samples per batch (default 4)
    --max-batches N         Stop after N batches; 0 = all batches (default 0)
    --out-csv PATH          Per-batch CSV output
                            (default: ttk_runs_fixed/topology_finetuning/loss_magnitude_diagnostic.csv)
    --out-md  PATH          Markdown summary report
                            (default: docs/topology_finetuning_loss_diagnostic.md)

Lambda overrides for on-the-fly inspection:
    --lambda-speed    FLOAT
    --lambda-grad     FLOAT
    --lambda-wpd      FLOAT
    --lambda-levelset FLOAT
    --levelset-temperature FLOAT
    --levelset-thresholds  FLOAT [FLOAT ...]
    --lambda-crit     FLOAT
    --crit-high-z     FLOAT
    --crit-include-minima
    --crit-low-z      FLOAT
    --crit-pool       INT
"""

from __future__ import annotations

import argparse
import csv
import io
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
# Paper-consistent normalization constants for the vector [u,v] model.
# ---------------------------------------------------------------------------
DEFAULT_MU  = [0.7684, -0.4575]
DEFAULT_SIG = [5.02455, 5.9017]
DEFAULT_R   = [5]

# Expected number of records in the canonical TFRecord.
EXPECTED_N_RECORDS = 168

# ---------------------------------------------------------------------------
# Candidate lambda configurations for physics-aware fine-tuning.
# WPD is excluded (lambda=0) from A/B/C because L_wpd magnitudes are ~100–300×
# L_uv even after switching to MAE; Candidate D includes it for completeness.
# ---------------------------------------------------------------------------
CANDIDATES = {
    'A (conservative)': dict(
        lambda_speed=0.005, lambda_grad=0.01, lambda_wpd=0.0, lambda_levelset=0.1,
        lambda_crit=0.0,
    ),
    'B (moderate)': dict(
        lambda_speed=0.01, lambda_grad=0.05, lambda_wpd=0.0, lambda_levelset=0.25,
        lambda_crit=0.0,
    ),
    'C (aggressive)': dict(
        lambda_speed=0.02, lambda_grad=0.05, lambda_wpd=0.0, lambda_levelset=0.5,
        lambda_crit=0.0,
    ),
    'D (moderate + WPD)': dict(
        lambda_speed=0.01, lambda_grad=0.05, lambda_wpd=1e-5, lambda_levelset=0.25,
        lambda_crit=0.0,
    ),
    'C1 (B + crit-value)': dict(
        lambda_speed=0.01, lambda_grad=0.05, lambda_wpd=0.0, lambda_levelset=0.25,
        lambda_crit=0.05,
    ),
}

# Regime labels for weighted-contribution fraction relative to L_uv.
def _regime(frac: float) -> str:
    if frac < 0.01:
        return 'negligible (<1 %)'
    if frac < 0.10:
        return 'minor (1–10 %)'
    if frac < 0.50:
        return 'moderate (10–50 %)'
    if frac < 1.00:
        return 'large (50–100 %)'
    return 'dominant (>100 %)'


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
                    help='Number of samples per batch (default 4)')
    ap.add_argument('--max-batches', type=int, default=0,
                    help='Stop after N batches; 0 = run all batches (default 0)')
    ap.add_argument('--out-csv', type=str,
                    default='ttk_runs_fixed/topology_finetuning/loss_magnitude_diagnostic.csv',
                    help='Per-batch CSV output path')
    ap.add_argument('--out-md', type=str,
                    default='docs/topology_finetuning_loss_diagnostic.md',
                    help='Markdown summary report path')
    ap.add_argument('--r', type=int, nargs='+', default=DEFAULT_R,
                    help='SR scaling factors (default 5)')
    ap.add_argument('--mu',  type=float, nargs='+', default=DEFAULT_MU,
                    help='Per-channel normalisation means')
    ap.add_argument('--sig', type=float, nargs='+', default=DEFAULT_SIG,
                    help='Per-channel normalisation stds')
    # Lambda knobs for optional on-the-fly inspection (all default 0).
    ap.add_argument('--lambda-speed',    type=float, default=0.0)
    ap.add_argument('--lambda-grad',     type=float, default=0.0)
    ap.add_argument('--lambda-wpd',      type=float, default=0.0)
    ap.add_argument('--lambda-levelset', type=float, default=0.0)
    ap.add_argument('--levelset-temperature', type=float, default=10.0)
    ap.add_argument('--levelset-thresholds',  type=float, nargs='+',
                    default=[5.0, 10.0, 15.0],
                    help='Physical speed thresholds in m/s (default: 5 10 15)')
    ap.add_argument('--lambda-crit',        type=float, default=0.0)
    ap.add_argument('--crit-high-z',        type=float, default=1.0)
    ap.add_argument('--crit-include-minima', action='store_true', default=False)
    ap.add_argument('--crit-low-z',         type=float, default=-1.0)
    ap.add_argument('--crit-pool',          type=int,   default=3)
    return ap.parse_args()


def _count_tfrecord_examples(path: str) -> int:
    """Count the number of serialised examples in a TFRecord file."""
    count = 0
    for _ in tf.python_io.tf_record_iterator(path):
        count += 1
    return count


def _summary_stats(values: list[float]) -> dict[str, float]:
    arr = np.array(values, dtype=np.float64)
    return {
        'mean': float(arr.mean()),
        'std':  float(arr.std()),
        'min':  float(arr.min()),
        'max':  float(arr.max()),
    }


def main() -> None:
    args = _parse_args()

    data_path  = str(args.data_path)
    model_path = str(args.model_path)

    # -----------------------------------------------------------------------
    # Pre-flight checks
    # -----------------------------------------------------------------------
    if not Path(data_path).exists():
        sys.exit(f'[error] TFRecord not found: {data_path}')
    for ext in ('.index', '.meta'):
        if not Path(model_path + ext).exists():
            sys.exit(f'[error] Checkpoint file not found: {model_path}{ext}')

    out_csv = Path(args.out_csv)
    out_md  = Path(args.out_md)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    mu_sig = [args.mu, args.sig]

    # -----------------------------------------------------------------------
    # TFRecord record-count check
    # -----------------------------------------------------------------------
    print('Counting TFRecord examples ...', end=' ', flush=True)
    n_records = _count_tfrecord_examples(data_path)
    print(f'{n_records}')
    if n_records != EXPECTED_N_RECORDS:
        print(f'[warning] Expected {EXPECTED_N_RECORDS} records, found {n_records}. '
              f'Results may not match paper baseline.')
    else:
        print(f'[ok] Record count matches expected ({EXPECTED_N_RECORDS}).')

    # -----------------------------------------------------------------------
    # Build diagnostic PhysicsLossConfig
    # -----------------------------------------------------------------------
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
        lambda_crit          = args.lambda_crit,
        crit_high_z          = args.crit_high_z,
        crit_include_minima  = args.crit_include_minima,
        crit_low_z           = args.crit_low_z,
        crit_pool            = args.crit_pool,
        diagnostic_mode      = True,
    )

    print('=' * 64)
    print('PhIRE physics/topology loss calibration diagnostic')
    print('=' * 64)
    print(f'  data_path  : {data_path}')
    print(f'  model_path : {model_path}')
    print(f'  batch_size : {args.batch_size}')
    print(f'  max_batches: {"all" if args.max_batches == 0 else args.max_batches}')
    print(f'  r          : {args.r}')
    print(f'  mu         : {args.mu}')
    print(f'  sig        : {args.sig}')
    print(f'  lambdas    : speed={args.lambda_speed}  grad={args.lambda_grad}'
          f'  wpd={args.lambda_wpd}  levelset={args.lambda_levelset}'
          f'  crit={args.lambda_crit}')
    print(f'  levelset k : {args.levelset_temperature}')
    print(f'  thresholds : {args.levelset_thresholds} m/s')
    print(f'  crit_high_z: {args.crit_high_z}  crit_pool={args.crit_pool}'
          f'  include_minima={args.crit_include_minima}'
          f'  crit_low_z={args.crit_low_z}')
    print(f'  out_csv    : {out_csv}')
    print(f'  out_md     : {out_md}')
    print()

    tf.reset_default_graph()

    # -----------------------------------------------------------------------
    # LR shape from TFRecord
    # -----------------------------------------------------------------------
    phire = PhIREGANs(data_type='wind_diagnostic', mu_sig=mu_sig)
    phire.set_LR_data_shape(data_path)
    h, w, C = phire.LR_data_shape
    print(f'LR shape: h={h}  w={w}  C={C}')
    r = args.r

    # -----------------------------------------------------------------------
    # Build graph (diagnostic, no optimizer)
    # -----------------------------------------------------------------------
    print('Building graph (pretraining + diagnostic) ...', end=' ', flush=True)
    x_LR = tf.placeholder(tf.float32, [None, h,            w,           C])
    x_HR = tf.placeholder(tf.float32, [None, h*np.prod(r), w*np.prod(r), C])

    model = SR_NETWORK(x_LR, x_HR, r=r, status='pretraining', phys_cfg=phys_cfg)

    if model.loss_breakdown is None:
        sys.exit('[error] loss_breakdown not built — check phys_cfg.diagnostic_mode')

    init    = tf.global_variables_initializer()
    g_saver = tf.train.Saver(var_list=model.g_variables)
    print('Done.')

    # -----------------------------------------------------------------------
    # Data pipeline
    # -----------------------------------------------------------------------
    ds = tf.data.TFRecordDataset(data_path)
    ds = ds.map(lambda xx: phire._parse_train_(xx, mu_sig)).batch(args.batch_size)

    iterator  = tf.data.Iterator.from_structure(ds.output_types, ds.output_shapes)
    idx_out, LR_out, HR_out = iterator.get_next()
    init_iter = iterator.make_initializer(ds)

    # -----------------------------------------------------------------------
    # Per-loss accumulators
    # -----------------------------------------------------------------------
    loss_keys   = ['L_uv', 'L_speed', 'L_grad', 'L_wpd', 'L_levelset', 'L_crit']
    accumulators: dict[str, list[float]] = {k: [] for k in loss_keys}
    batch_rows: list[dict] = []

    # -----------------------------------------------------------------------
    # Session: restore checkpoint, run forward passes
    # -----------------------------------------------------------------------
    with tf.Session() as sess:
        sess.run(init)
        print(f'Restoring checkpoint: {model_path} ...', end=' ', flush=True)
        g_saver.restore(sess, model_path)
        print('Done.')
        print()

        sess.run(init_iter)
        batch_num = 0

        while True:
            if args.max_batches > 0 and batch_num >= args.max_batches:
                break
            try:
                batch_idx, batch_LR, batch_HR = sess.run([idx_out, LR_out, HR_out])
            except tf.errors.OutOfRangeError:
                break

            feed_dict = {x_LR: batch_LR, x_HR: batch_HR}
            bv = sess.run(model.loss_breakdown, feed_dict=feed_dict)

            row = {
                'batch_number':   batch_num,
                'sample_indices': str(batch_idx.astype(int).ravel().tolist()),
            }
            for k in loss_keys:
                val = float(bv.get(k, float('nan')))
                row[k] = val
                accumulators[k].append(val)

            batch_rows.append(row)
            print(f'  batch {batch_num:3d}: indices={batch_idx.astype(int).ravel().tolist()}'
                  f'  L_uv={row["L_uv"]:.6f}  L_speed={row["L_speed"]:.6f}'
                  f'  L_grad={row["L_grad"]:.6f}  L_wpd={row["L_wpd"]:.6f}'
                  f'  L_levelset={row["L_levelset"]:.6f}'
                  f'  L_crit={row["L_crit"]:.6f}')
            batch_num += 1

    n_batches = len(batch_rows)
    print(f'\nProcessed {n_batches} batches.')

    if n_batches == 0:
        sys.exit('[error] No batches were processed. Check data path and batch size.')

    # -----------------------------------------------------------------------
    # Summary statistics
    # -----------------------------------------------------------------------
    stats: dict[str, dict[str, float]] = {}
    for k in loss_keys:
        stats[k] = _summary_stats(accumulators[k])

    mean_luv = stats['L_uv']['mean']

    # -----------------------------------------------------------------------
    # Write per-batch CSV
    # -----------------------------------------------------------------------
    csv_cols = ['batch_number', 'sample_indices'] + loss_keys
    with open(out_csv, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=csv_cols)
        writer.writeheader()
        writer.writerows(batch_rows)
    print(f'Per-batch CSV written: {out_csv}')

    # -----------------------------------------------------------------------
    # Candidate lambda analysis
    # -----------------------------------------------------------------------
    _CAND_TERMS = [
        ('L_speed',    'lambda_speed'),
        ('L_grad',     'lambda_grad'),
        ('L_wpd',      'lambda_wpd'),
        ('L_levelset', 'lambda_levelset'),
        ('L_crit',     'lambda_crit'),
    ]

    candidate_lines: list[str] = []
    for cand_name, lambdas in CANDIDATES.items():
        contribs: dict[str, float] = {}
        total_aux = 0.0
        for lk, stat_key in _CAND_TERMS:
            lam = lambdas.get(stat_key, 0.0)
            contrib = lam * stats[lk]['mean']
            contribs[lk] = contrib
            total_aux += contrib
        total_loss = mean_luv + total_aux
        lines = [f'\n### Candidate {cand_name}']
        lines.append(f'  lambda_speed={lambdas.get("lambda_speed", 0.0)}  '
                     f'lambda_grad={lambdas.get("lambda_grad", 0.0)}  '
                     f'lambda_wpd={lambdas.get("lambda_wpd", 0.0)}  '
                     f'lambda_levelset={lambdas.get("lambda_levelset", 0.0)}  '
                     f'lambda_crit={lambdas.get("lambda_crit", 0.0)}')
        lines.append(f'  {"Term":<18s}  {"Raw mean":>12s}  {"Lambda":>10s}  '
                     f'{"Weighted":>12s}  {"% of L_uv":>12s}  Regime')
        lines.append('  ' + '-' * 90)
        for lk, stat_key in _CAND_TERMS:
            lam  = lambdas.get(stat_key, 0.0)
            raw  = stats[lk]['mean']
            wt   = contribs[lk]
            pct  = 100.0 * wt / mean_luv if mean_luv > 0 else float('nan')
            reg  = _regime(wt / mean_luv) if mean_luv > 0 else 'n/a'
            lines.append(f'  {lk:<18s}  {raw:>12.6f}  {lam:>10.5f}  '
                         f'{wt:>12.6f}  {pct:>11.2f}%  {reg}')
        lines.append(f'  {"L_uv":<18s}  {mean_luv:>12.6f}  {"(baseline)":>10s}  '
                     f'{mean_luv:>12.6f}  {"100.00":>11s}%')
        lines.append(f'  {"total_loss":<18s}  {"":>12s}  {"":>10s}  '
                     f'{total_loss:>12.6f}')
        candidate_lines.extend(lines)

    # -----------------------------------------------------------------------
    # Print summary to stdout
    # -----------------------------------------------------------------------
    print()
    print('=' * 64)
    print('Summary statistics across all batches')
    print('=' * 64)
    hdr = f'  {"Term":<14s}  {"Mean":>12s}  {"Std":>12s}  {"Min":>12s}  {"Max":>12s}'
    print(hdr)
    print('  ' + '-' * 66)
    for k in loss_keys:
        s = stats[k]
        print(f'  {k:<14s}  {s["mean"]:>12.6f}  {s["std"]:>12.6f}'
              f'  {s["min"]:>12.6f}  {s["max"]:>12.6f}')

    print()
    print('Ratios to L_uv (mean):')
    for k in ['L_speed', 'L_grad', 'L_wpd', 'L_levelset', 'L_crit']:
        ratio = stats[k]['mean'] / mean_luv if mean_luv > 0 else float('nan')
        print(f'  {k:<14s} / L_uv = {ratio:>10.4f}×')

    print()
    for line in candidate_lines:
        print(line)

    print()
    print('Notes:')
    print('  L_uv       — normalized [u,v] MSE (training-space loss, dimensionless).')
    print('  L_speed    — physical wind speed MSE (m²/s²); denorm applied.')
    print('  L_grad     — physical speed-gradient magnitude MSE (grid-cell units).')
    print('  L_wpd      — MAE on speed³ proxy for wind power density (m³/s³).')
    print('  L_levelset — dimensionless soft-mask MSE at physical thresholds.')
    print('  L_crit     — MSE at GT local-speed maxima above adaptive threshold;')
    print('               critical-value proxy for superlevel-set topology.')
    print('  No checkpoint was modified.')
    print('=' * 64)

    # -----------------------------------------------------------------------
    # Markdown report
    # -----------------------------------------------------------------------
    md = io.StringIO()

    md.write('# Physics/topology loss calibration diagnostic\n\n')
    md.write(f'**Generated:** {_today()}\n\n')
    md.write('## Configuration\n\n')
    md.write(f'| Parameter | Value |\n|---|---|\n')
    md.write(f'| data_path | `{data_path}` |\n')
    md.write(f'| model_path | `{model_path}` |\n')
    md.write(f'| n_records (found / expected) | {n_records} / {EXPECTED_N_RECORDS} |\n')
    md.write(f'| batch_size | {args.batch_size} |\n')
    md.write(f'| n_batches processed | {n_batches} |\n')
    md.write(f'| r | {args.r} |\n')
    md.write(f'| mu | {args.mu} |\n')
    md.write(f'| sig | {args.sig} |\n')
    md.write(f'| levelset thresholds (m/s) | {args.levelset_thresholds} |\n')
    md.write(f'| levelset temperature k | {args.levelset_temperature} |\n')
    md.write(f'| lambda_crit | {args.lambda_crit} |\n')
    md.write(f'| crit_high_z | {args.crit_high_z} |\n')
    md.write(f'| crit_include_minima | {args.crit_include_minima} |\n')
    md.write(f'| crit_low_z | {args.crit_low_z} |\n')
    md.write(f'| crit_pool | {args.crit_pool} |\n\n')

    md.write('## Loss term descriptions\n\n')
    md.write('| Term | Description |\n|---|---|\n')
    md.write('| L_uv | Normalized [u,v] MSE — the baseline training loss; dimensionless. |\n')
    md.write('| L_speed | Physical wind speed MSE (m²/s²); requires per-channel denormalization. |\n')
    md.write('| L_grad | Physical speed-gradient magnitude MSE; backward finite differences on interior grid. |\n')
    md.write('| L_wpd | MAE on speed³ wind-power-density proxy (m³/s³); MAE used to keep magnitudes tractable. |\n')
    md.write('| L_levelset | Dimensionless soft-mask MSE at physical speed thresholds via sigmoid; '
             'topology-inspired superlevel-set proxy. |\n')
    md.write('| L_crit | MSE focused on GT local-speed maxima above an adaptive per-sample threshold; '
             'critical-value proxy for superlevel-set topology (inspired by Kissi et al.). |\n\n')

    md.write('## Summary statistics\n\n')
    md.write('| Term | Mean | Std | Min | Max |\n|---|---|---|---|---|\n')
    for k in loss_keys:
        s = stats[k]
        md.write(f'| {k} | {s["mean"]:.6f} | {s["std"]:.6f} | {s["min"]:.6f} | {s["max"]:.6f} |\n')

    md.write('\n## Ratios to L_uv\n\n')
    md.write('| Term | Ratio (×L_uv) |\n|---|---|\n')
    for k in ['L_speed', 'L_grad', 'L_wpd', 'L_levelset', 'L_crit']:
        ratio = stats[k]['mean'] / mean_luv if mean_luv > 0 else float('nan')
        md.write(f'| {k} | {ratio:.4f}× |\n')

    md.write('\n## Candidate lambda configurations\n\n')
    md.write('WPD is set to `lambda_wpd=0` in candidates A–C because L_wpd magnitudes '
             'are large even after switching to MAE; Candidate D includes a small WPD weight '
             'for reference.\n\n')

    for cand_name, lambdas in CANDIDATES.items():
        md.write(f'### Candidate {cand_name}\n\n')
        md.write(f'`lambda_speed={lambdas.get("lambda_speed", 0.0)}  '
                 f'lambda_grad={lambdas.get("lambda_grad", 0.0)}  '
                 f'lambda_wpd={lambdas.get("lambda_wpd", 0.0)}  '
                 f'lambda_levelset={lambdas.get("lambda_levelset", 0.0)}  '
                 f'lambda_crit={lambdas.get("lambda_crit", 0.0)}`\n\n')
        md.write('| Term | Raw mean | Lambda | Weighted contrib | % of L_uv | Regime |\n')
        md.write('|---|---|---|---|---|---|\n')
        for lk, stat_key in _CAND_TERMS:
            lam  = lambdas.get(stat_key, 0.0)
            raw  = stats[lk]['mean']
            wt   = lam * raw
            pct  = 100.0 * wt / mean_luv if mean_luv > 0 else float('nan')
            reg  = _regime(wt / mean_luv) if mean_luv > 0 else 'n/a'
            md.write(f'| {lk} | {raw:.6f} | {lam} | {wt:.6f} | {pct:.2f}% | {reg} |\n')
        lam_total = mean_luv + sum(lambdas.get(sk, 0.0) * stats[lk]['mean']
                                   for lk, sk in _CAND_TERMS)
        md.write(f'| L_uv (baseline) | {mean_luv:.6f} | — | {mean_luv:.6f} | 100.00% | — |\n')
        md.write(f'| **total_loss** | | | **{lam_total:.6f}** | | |\n\n')

    md.write('## Output files\n\n')
    md.write(f'- Per-batch CSV: `{out_csv}`\n')
    md.write(f'- This report: `{out_md}`\n\n')

    md.write('## Notes\n\n')
    md.write('- L_uv is the normalized [u,v] MSE loss used in baseline pretraining; '
             'it is dimensionless (both channels normalized to ~unit variance).\n')
    md.write('- L_speed and L_grad are in physical units (m²/s² and grid-cell² respectively) '
             'after per-channel denormalization: `u_phys = sig[0]*u_norm + mu[0]`, '
             '`v_phys = sig[1]*v_norm + mu[1]`, `speed = sqrt(u² + v² + ε)`.\n')
    md.write('- L_wpd uses MAE on `speed³` (not MSE) to keep the magnitude range tractable; '
             'even so it remains ~100–300× L_uv and should only be enabled with very small lambda.\n')
    md.write('- L_levelset is a soft superlevel-set proxy, not a merge-tree loss. '
             'It uses `sigmoid(k*(speed−τ))` at physical thresholds [5,10,15] m/s.\n')
    md.write('- L_crit is a critical-value proxy loss: MSE is concentrated at GT local-speed maxima '
             'that exceed `mean + crit_high_z * std` per sample.  These maxima correspond to '
             'superlevel-set component births.  Inspired by Kissi et al. (not a full PD loss).\n')
    md.write('- No checkpoint was modified during this diagnostic run.\n')

    md_text = md.getvalue()
    out_md.write_text(md_text, encoding='utf-8')
    print(f'Markdown report written: {out_md}')


def _today() -> str:
    import datetime
    return datetime.date.today().isoformat()


if __name__ == '__main__':
    main()
