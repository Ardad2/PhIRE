#!/usr/bin/env python3
"""
Candidate B factorial ablation, expanded-2688 scale: isolates which of
Candidate B's three scalar-field auxiliary losses (L_speed, L_grad,
L_levelset) drives Candidate B/C's PD improvement, and which contributes to
their MT degradation relative to CNN.

Motivation
-----------
Candidate C-expanded-2688 (PD 22.4944, MT 6.0803) and the newly completed
Candidate B-expanded-2688 (PD 22.7070, MT 6.1612, L_crit=0) are very close
on PD but B is worse on MT -- Candidate B's scalar-field scaffold
(L_speed + L_grad + L_levelset) appears to drive most of Candidate C's PD
gain, while L_crit contributes only a small additional PD improvement and
slightly mitigates the MT degradation seen with B alone. This script tests
the three terms inside Candidate B individually and pairwise, at the same
2688-sample scale, to see which term(s) are responsible.

Six factorial variants (single parameterized script, not six copies):
  speed            L_uv + 0.01 L_speed
  grad             L_uv + 0.05 L_grad
  levelset         L_uv + 0.25 L_levelset
  speed_grad       L_uv + 0.01 L_speed + 0.05 L_grad
  speed_levelset   L_uv + 0.01 L_speed          + 0.25 L_levelset
  grad_levelset    L_uv          + 0.05 L_grad  + 0.25 L_levelset

L_crit = 0.0 and no repaired-E2 TTK fixed-index losses (L_TTKCV, L_TTKpers)
are built in this script at all, in every variant -- exactly like
candidateB_expanded2688 and candidateC_expanded2688 (plain
PhIREGANs.pretrain() path, no custom training loop, no constraints NPZ).

The full Candidate B variant (all three terms) already exists as
scripts/run_candidateB_expanded2688_finetune.py
(candidateB_expanded2688). The bare UV baseline already exists as
scripts/run_candidateUV_expanded2688_finetune.py (candidateUV_expanded2688).
Neither is reproduced by this script; both remain the endpoints of this
factorial design.

Same hyperparameters/normalization/data as candidateB_expanded2688 and
candidateC_expanded2688 in every variant:
  learning_rate = 1e-5, N_epochs = 3, batch_size = 4, save_every = 1,
  print_every = 2, mu/sig = [[0.7684, -0.4575], [5.02455, 5.9017]],
  training data = example_data_topology_expanded_2688/wind_MR-HR.tfrecord
  (the SAME 2688-sample file as candidateC_expanded2688 and
  candidateB_expanded2688), evaluation = the fixed 168-sample benchmark.

Usage (from repo root, on Spark):
  python3 scripts/run_candidateB_factorial_expanded2688_finetune.py --variant speed
  python3 scripts/run_candidateB_factorial_expanded2688_finetune.py --variant grad
  python3 scripts/run_candidateB_factorial_expanded2688_finetune.py --variant levelset
  python3 scripts/run_candidateB_factorial_expanded2688_finetune.py --variant speed_grad
  python3 scripts/run_candidateB_factorial_expanded2688_finetune.py --variant speed_levelset
  python3 scripts/run_candidateB_factorial_expanded2688_finetune.py --variant grad_levelset

Dry run (preflight checks + plan only; no training):
  python3 scripts/run_candidateB_factorial_expanded2688_finetune.py --variant speed --dry-run

By default this script ABORTS if its target MODEL_OUT/DATA_OUT already
exist (never silently overwrites a completed variant). Pass --overwrite (or
--resume, which behaves identically here -- this codebase's
PhIREGANs.pretrain() always retrains from the pretrained CNN checkpoint
from scratch; there is no partial-training resume capability to invoke) to
deliberately re-run a variant.

Outputs (per variant, <variant> in {speed, grad, levelset, speed_grad,
speed_levelset, grad_levelset}):
  models_fixed/topology_finetuning/wind_finetune_candidateB_factorial_<variant>_expanded2688/
  data_out/wind_finetune_candidateB_factorial_<variant>_expanded2688/
  logs/wind_finetune_candidateB_factorial_<variant>_expanded2688.log

This script never runs the expensive TTK topology pipeline automatically --
it prints the exact command at the end, to be run manually only after the
cheap scalar/domain evaluation looks non-catastrophic (see
scripts/run_candidateB_factorial_expanded2688_batch.sh, which does run the
cheap evaluation automatically after each variant's training/inference).
"""

from __future__ import annotations

import argparse
import sys
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure repo root is on sys.path
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Factorial variants: (lambda_speed, lambda_grad, lambda_levelset)
# ---------------------------------------------------------------------------
VARIANTS = {
    'speed':          {'lambda_speed': 0.01, 'lambda_grad': 0.0,  'lambda_levelset': 0.0},
    'grad':           {'lambda_speed': 0.0,  'lambda_grad': 0.05, 'lambda_levelset': 0.0},
    'levelset':       {'lambda_speed': 0.0,  'lambda_grad': 0.0,  'lambda_levelset': 0.25},
    'speed_grad':     {'lambda_speed': 0.01, 'lambda_grad': 0.05, 'lambda_levelset': 0.0},
    'speed_levelset': {'lambda_speed': 0.01, 'lambda_grad': 0.0,  'lambda_levelset': 0.25},
    'grad_levelset':  {'lambda_speed': 0.0,  'lambda_grad': 0.05, 'lambda_levelset': 0.25},
}

# Preferred scientific run order (see scripts/run_candidateB_factorial_expanded2688_batch.sh):
# the soft high-speed level-set term is the most interesting PD hypothesis,
# so it is run first, followed by the two singletons, then the three pairs.
PREFERRED_ORDER = ['levelset', 'speed', 'grad', 'speed_levelset', 'grad_levelset', 'speed_grad']


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument('--variant', required=True, choices=sorted(VARIANTS),
                     help='Which factorial variant to train.')
    ap.add_argument('--dry-run', action='store_true',
                     help='Run preflight/safety checks and print the plan; do not train.')
    ap.add_argument('--overwrite', action='store_true',
                     help='Allow proceeding even if MODEL_OUT/DATA_OUT already exist. '
                          'Default is to abort (never silently overwrite a completed variant).')
    ap.add_argument('--resume', action='store_true',
                     help='Alias for --overwrite. PhIREGANs.pretrain() always retrains from '
                          'the pretrained CNN checkpoint from scratch in this codebase -- '
                          'there is no partial-training resume to invoke, so this flag has '
                          'the exact same effect as --overwrite.')
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    variant = args.variant
    weights = VARIANTS[variant]
    allow_existing = bool(args.overwrite or args.resume)

    # -----------------------------------------------------------------------
    # Paths (depend on --variant, so these are computed here rather than as
    # module-level constants)
    # -----------------------------------------------------------------------
    TRAIN_DATA_PATH = 'example_data_topology_expanded_2688/wind_MR-HR.tfrecord'
    EVAL_DATA_PATH  = 'example_data_fixed/wind_MR-HR.tfrecord'
    MODEL_PATH      = 'models/wind_mr-hr/trained_cnn/cnn'
    MODEL_OUT = f'models_fixed/topology_finetuning/wind_finetune_candidateB_factorial_{variant}_expanded2688'
    DATA_OUT  = f'data_out/wind_finetune_candidateB_factorial_{variant}_expanded2688'
    LOG_PATH  = REPO_ROOT / 'logs' / f'wind_finetune_candidateB_factorial_{variant}_expanded2688.log'

    # -----------------------------------------------------------------------
    # Tee: mirror stdout to a log file (after LOG_PATH is known)
    # -----------------------------------------------------------------------
    class _Tee:
        """Write to both the original stdout and a file simultaneously."""
        def __init__(self, stream, path: Path):
            path.parent.mkdir(parents=True, exist_ok=True)
            self._file   = open(path, 'a', buffering=1)
            self._stream = stream

        def write(self, data):
            self._stream.write(data)
            self._file.write(data)

        def flush(self):
            self._stream.flush()
            self._file.flush()

        def fileno(self):
            return self._stream.fileno()

    sys.stdout = _Tee(sys.stdout, LOG_PATH)
    sys.stderr = sys.stdout

    # -------------------------------------------------------------------
    # Imports (after tee is active)
    # -------------------------------------------------------------------
    import numpy as np
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()

    from sr_network import PhysicsLossConfig
    from PhIREGANs import PhIREGANs

    # -------------------------------------------------------------------
    # Protected directories -- exhaustive union of every existing candidate
    # MODEL_OUT/DATA_OUT literal in scripts/*.py at the time this script was
    # written (same list as scripts/run_candidateB_expanded2688_finetune.py),
    # plus candidateB_expanded2688 itself (the full-B endpoint of this
    # factorial design) and every sibling factorial variant's own output.
    # -------------------------------------------------------------------
    _PROTECTED = [
        # Source checkpoint
        'models/wind_mr-hr/trained_cnn',
        # Benchmark data
        'example_data_fixed',
        # Baseline model outputs
        'data_out_fixed/wind_mrhr_cnn',
        'data_out_fixed/wind_mrhr_gan',
        # Original 168-sample pilot candidate outputs (data)
        'data_out/wind_finetune_pilot_candidateB',
        'data_out/wind_finetune_pilot_candidateC',
        'data_out/wind_finetune_pilot_candidateD',
        'data_out/wind_finetune_pilot_candidateE',
        'data_out/wind_finetune_pilot_candidateE2',
        'data_out/wind_finetune_pilot_candidateUV',
        # Original 168-sample pilot candidate outputs (models)
        'models_fixed/topology_finetuning/wind_finetune_pilot_candidateB',
        'models_fixed/topology_finetuning/wind_finetune_pilot_candidateC',
        'models_fixed/topology_finetuning/wind_finetune_pilot_candidateD',
        'models_fixed/topology_finetuning/wind_finetune_pilot_candidateE',
        'models_fixed/topology_finetuning/wind_finetune_pilot_candidateE2',
        'models_fixed/topology_finetuning/wind_finetune_pilot_candidateUV',
        # Expanded UV/B/C/D/Dpd candidates at every completed scale (data)
        'data_out/wind_finetune_candidateUV_expanded672',
        'data_out/wind_finetune_candidateUV_expanded1344',
        'data_out/wind_finetune_candidateUV_expanded2688',
        'data_out/wind_finetune_candidateB_expanded672',
        'data_out/wind_finetune_candidateB_expanded2688',
        'data_out/wind_finetune_candidateC_expanded672',
        'data_out/wind_finetune_candidateC_expanded1344',
        'data_out/wind_finetune_candidateC_expanded2688',
        'data_out/wind_finetune_candidateD_expanded672',
        'data_out/wind_finetune_candidateDpd_expanded672',
        # Expanded UV/B/C/D/Dpd candidates at every completed scale (models)
        'models_fixed/topology_finetuning/wind_finetune_candidateUV_expanded672',
        'models_fixed/topology_finetuning/wind_finetune_candidateUV_expanded1344',
        'models_fixed/topology_finetuning/wind_finetune_candidateUV_expanded2688',
        'models_fixed/topology_finetuning/wind_finetune_candidateB_expanded672',
        'models_fixed/topology_finetuning/wind_finetune_candidateB_expanded2688',
        'models_fixed/topology_finetuning/wind_finetune_candidateC_expanded672',
        'models_fixed/topology_finetuning/wind_finetune_candidateC_expanded1344',
        'models_fixed/topology_finetuning/wind_finetune_candidateC_expanded2688',
        'models_fixed/topology_finetuning/wind_finetune_candidateD_expanded672',
        'models_fixed/topology_finetuning/wind_finetune_candidateDpd_expanded672',
        # PyTorch residual-refiner E2 family -- distinct architecture (data)
        'data_out/wind_finetune_candidateE2_expanded672',
        'data_out/wind_finetune_candidateE2_fixed',
        'data_out/wind_finetune_candidateE2_fixed_lowlambda',
        'data_out/wind_finetune_candidateE2_fixed_lowlambda_expanded1344',
        'data_out/wind_finetune_candidateE2_fixed_lowlambda_expanded2688',
        # PyTorch residual-refiner E2 family (models)
        'models_fixed/topology_finetuning/wind_finetune_candidateE2_expanded672',
        'models_fixed/topology_finetuning/wind_finetune_candidateE2_fixed',
        'models_fixed/topology_finetuning/wind_finetune_candidateE2_fixed_lowlambda',
        'models_fixed/topology_finetuning/wind_finetune_candidateE2_fixed_lowlambda_expanded1344',
        'models_fixed/topology_finetuning/wind_finetune_candidateE2_fixed_lowlambda_expanded2688',
        # Native TF C+E2-low family, all scales (data + models)
        'data_out/wind_finetune_candidateE2_tf_lowlambda_expanded672',
        'data_out/wind_finetune_candidateE2_tf_lowlambda_expanded1344',
        'data_out/wind_finetune_candidateE2_tf_lowlambda_expanded2688',
        'models_fixed/topology_finetuning/wind_finetune_candidateE2_tf_lowlambda_expanded672',
        'models_fixed/topology_finetuning/wind_finetune_candidateE2_tf_lowlambda_expanded1344',
        'models_fixed/topology_finetuning/wind_finetune_candidateE2_tf_lowlambda_expanded2688',
        # Native TF B+E2-low family, all scales (data + models)
        'data_out/wind_finetune_candidateB_plus_E2_tf_lowlambda_expanded672',
        'data_out/wind_finetune_candidateB_plus_E2_tf_lowlambda_expanded1344',
        'data_out/wind_finetune_candidateB_plus_E2_tf_lowlambda_expanded2688',
        'models_fixed/topology_finetuning/wind_finetune_candidateB_plus_E2_tf_lowlambda_expanded672',
        'models_fixed/topology_finetuning/wind_finetune_candidateB_plus_E2_tf_lowlambda_expanded1344',
        'models_fixed/topology_finetuning/wind_finetune_candidateB_plus_E2_tf_lowlambda_expanded2688',
        # Native TF UV+E2-low family, all scales (data + models)
        'data_out/wind_finetune_candidateUV_plus_E2_tf_lowlambda_expanded672',
        'data_out/wind_finetune_candidateUV_plus_E2_tf_lowlambda_expanded1344',
        'data_out/wind_finetune_candidateUV_plus_E2_tf_lowlambda_expanded2688',
        'models_fixed/topology_finetuning/wind_finetune_candidateUV_plus_E2_tf_lowlambda_expanded672',
        'models_fixed/topology_finetuning/wind_finetune_candidateUV_plus_E2_tf_lowlambda_expanded1344',
        'models_fixed/topology_finetuning/wind_finetune_candidateUV_plus_E2_tf_lowlambda_expanded2688',
        # Native TF UV+crit family, all scales (data + models)
        'data_out/wind_finetune_candidateUV_plus_crit_expanded672',
        'data_out/wind_finetune_candidateUV_plus_crit_expanded1344',
        'data_out/wind_finetune_candidateUV_plus_crit_expanded2688',
        'models_fixed/topology_finetuning/wind_finetune_candidateUV_plus_crit_expanded672',
        'models_fixed/topology_finetuning/wind_finetune_candidateUV_plus_crit_expanded1344',
        'models_fixed/topology_finetuning/wind_finetune_candidateUV_plus_crit_expanded2688',
        # Alias path referenced (but not actually used as a training script's own
        # MODEL_OUT/DATA_OUT) in scripts/run_superlevel_topology_robustness.py's
        # DEFAULT_METHODS; protected defensively in case it is ever populated.
        'data_out/wind_finetune_candidateC_plus_E2_tf_lowlambda_expanded2688',
        'models_fixed/topology_finetuning/wind_finetune_candidateC_plus_E2_tf_lowlambda_expanded2688',
    ]
    # Sibling cross-protection: every OTHER factorial variant's own output
    # (defense in depth on top of the dynamic existing-directory abort below).
    for other in VARIANTS:
        if other == variant:
            continue
        _PROTECTED.append(f'models_fixed/topology_finetuning/wind_finetune_candidateB_factorial_{other}_expanded2688')
        _PROTECTED.append(f'data_out/wind_finetune_candidateB_factorial_{other}_expanded2688')

    MU_SIG = [[0.7684, -0.4575], [5.02455, 5.9017]]
    R      = [5]
    N_TRAIN = 2688
    N_EVAL  = 168

    def _safety_check() -> None:
        for protected in _PROTECTED:
            for candidate_path in (MODEL_OUT, DATA_OUT):
                p = Path(protected).resolve()
                c = Path(candidate_path).resolve()
                if c == p or str(c).startswith(str(p) + os.sep):
                    sys.exit(
                        f'[error] Output path {candidate_path!r} overlaps protected '
                        f'path {protected!r}. Aborting to avoid overwriting baselines.'
                    )
        print('[safety] Output paths do not overlap any protected directory. OK.')

    def _count_tfrecord_records(data_path: str) -> int:
        n = 0
        for _ in tf.python_io.tf_record_iterator(data_path):
            n += 1
        return n

    _safety_check()

    print('=' * 64)
    print(f'Candidate B factorial ablation, expanded-2688: variant={variant!r}')
    print('=' * 64)
    print(f'  Training data     : {TRAIN_DATA_PATH}')
    print(f'  Evaluation data   : {EVAL_DATA_PATH}')
    print(f'  Source checkpoint : {MODEL_PATH}')
    print(f'  Output model dir  : {MODEL_OUT}')
    print(f'  Inference output  : {DATA_OUT}')
    print(f'  Log file          : {LOG_PATH}')
    print()
    print('  Loss configuration:')
    print(f'  lambda_speed    = {weights["lambda_speed"]}')
    print(f'  lambda_grad     = {weights["lambda_grad"]}')
    print('  lambda_wpd      = 0.0')
    print(f'  lambda_levelset = {weights["lambda_levelset"]}')
    print('  lambda_crit     = 0.0  (DISABLED -- factorial ablation never includes L_crit)')
    print('  lambda_TTKCV    = n/a  (no repaired-E2 losses are built in this script)')
    print('  lambda_TTKpers  = n/a  (no repaired-E2 losses are built in this script)')
    print()
    print('  Hyperparameters:')
    print('  learning_rate   = 1e-5')
    print('  N_epochs        = 3')
    print('  save_every      = 1')
    print('  print_every     = 2')
    print('  batch_size      = 4')
    print()
    print('  Normalisation (pretrained; unchanged):')
    print('  mu  = [0.7684, -0.4575]')
    print('  sig = [5.02455, 5.9017]')
    print('=' * 64)
    print()

    # -----------------------------------------------------------------------
    # Preflight checks
    # -----------------------------------------------------------------------
    print('=' * 64)
    print('Preflight checks')
    print('=' * 64)

    if not Path(TRAIN_DATA_PATH).exists():
        sys.exit(
            f'[error] Training TFRecord not found: {TRAIN_DATA_PATH}\n'
            '  Run scripts/build_wind_mrhr_expanded_dataset_2688.py first:\n'
            '    python3 scripts/build_wind_mrhr_expanded_dataset_2688.py \\\n'
            '      --out-dir example_data_topology_expanded_2688'
        )
    if not Path(EVAL_DATA_PATH).exists():
        sys.exit(
            f'[error] Evaluation TFRecord not found: {EVAL_DATA_PATH}\n'
            '  The corrected 168-sample benchmark is required for inference.'
        )
    for ext in ('.index', '.meta'):
        if not Path(MODEL_PATH + ext).exists():
            sys.exit(f'[error] Checkpoint not found: {MODEL_PATH}{ext}')

    print('  Counting training TFRecord records …')
    n_records = _count_tfrecord_records(TRAIN_DATA_PATH)
    print(f'  Training TFRecord  : {n_records} records')
    if n_records != N_TRAIN:
        sys.exit(f'[error] Training TFRecord has {n_records} records; expected {N_TRAIN}.')
    print(f'  Confirmed: {N_TRAIN}-sample training set matches candidateC_expanded2688 / candidateB_expanded2688.')
    print('  Confirmed: evaluation benchmark is the fixed 168-sample benchmark.')
    print('  Confirmed: lambda_crit = 0.0 in every factorial variant.')
    print('  Confirmed: no L_TTKCV/L_TTKpers ops are built in this script (repaired-E2 OFF).')
    print('  Confirmed: no E2 constraints NPZ and no custom E2 training loop are used.')

    for out_dir, label in ((MODEL_OUT, 'MODEL_OUT'), (DATA_OUT, 'DATA_OUT')):
        p = Path(out_dir)
        if p.exists() and any(p.iterdir()) and not allow_existing:
            sys.exit(
                f'[error] {label} already exists and is non-empty: {out_dir}\n'
                '  Refusing to overwrite a possibly-completed variant. Pass --overwrite '
                '(or --resume, identical effect here) to proceed deliberately.'
            )
    print(f'  Confirmed: MODEL_OUT/DATA_OUT do not already contain a completed run '
          f'(or --overwrite/--resume was passed).')
    print('=' * 64)
    print()

    if args.dry_run:
        print('[dry-run] Preflight checks passed. No training performed.')
        print(f'[dry-run] Would fine-tune variant={variant!r} for {3} epochs on {N_TRAIN} samples,')
        print(f'[dry-run] then run paired inference on the {N_EVAL}-sample benchmark, writing to:')
        print(f'[dry-run]   {MODEL_OUT}')
        print(f'[dry-run]   {DATA_OUT}')
        return

    # -----------------------------------------------------------------------
    # Phase 1: Fine-tuning on the 2688-sample expanded dataset.
    # -----------------------------------------------------------------------
    cfg = PhysicsLossConfig(
        use_aux_losses       = True,
        mu                   = MU_SIG[0],
        sig                  = MU_SIG[1],
        lambda_speed         = weights['lambda_speed'],
        lambda_grad          = weights['lambda_grad'],
        lambda_wpd           = 0.0,
        lambda_levelset      = weights['lambda_levelset'],
        levelset_temperature = 10.0,
        levelset_thresholds  = [5.0, 10.0, 15.0],
        lambda_crit          = 0.0,
        crit_high_z          = 1.0,
        crit_include_minima  = False,
        crit_low_z           = -1.0,
        crit_pool            = 3,
        diagnostic_mode      = True,
    )

    phire = PhIREGANs(
        data_type    = f'wind_finetune_candidateB_factorial_{variant}_expanded2688',
        learning_rate= 1e-5,
        N_epochs     = 3,
        save_every   = 1,
        print_every  = 2,
        mu_sig       = MU_SIG,
    )
    phire.setModel_name(MODEL_OUT)

    print(f'Phase 1: Fine-tuning on 2688-sample expanded dataset (variant={variant!r}) …')
    saved_model = phire.pretrain(
        r          = R,
        data_path  = TRAIN_DATA_PATH,
        model_path = MODEL_PATH,
        batch_size = 4,
        phys_cfg   = cfg,
    )

    print()
    print(f'Fine-tuning complete. Final checkpoint: {saved_model}')
    print()

    # -----------------------------------------------------------------------
    # Phase 2: Paired inference on the original 168-sample benchmark.
    # -----------------------------------------------------------------------
    print('=' * 64)
    print('Phase 2: Paired inference on 168-sample benchmark')
    print('=' * 64)
    print(f'  Checkpoint    : {saved_model}')
    print(f'  Eval data     : {EVAL_DATA_PATH}')
    print(f'  Output dir    : {DATA_OUT}')
    print()

    phire.set_data_out_path(DATA_OUT)
    phire.test_paired(
        r          = R,
        data_path  = EVAL_DATA_PATH,
        model_path = saved_model,
        batch_size = 1,
        save_inputs= True,
    )

    # -----------------------------------------------------------------------
    # Post-inference validation
    # -----------------------------------------------------------------------
    print()
    print('--- Post-inference validation ---')
    expected = {
        'idx.npy':    (N_EVAL,),
        'dataIN.npy': (N_EVAL, 100, 100, 2),
        'dataGT.npy': (N_EVAL, 500, 500, 2),
        'dataSR.npy': (N_EVAL, 500, 500, 2),
    }
    all_ok = True
    for fname, exp_shape in expected.items():
        p = Path(DATA_OUT) / fname
        if not p.exists():
            print(f'[MISSING] {p}')
            all_ok = False
            continue
        arr = np.load(str(p), mmap_mode='r')
        shape_ok = arr.shape == exp_shape
        finite_ok = True
        if fname != 'idx.npy':
            finite_ok = bool(np.isfinite(np.asarray(arr)).all())
        lo = float(np.nanmin(arr))
        hi = float(np.nanmax(arr))
        status = 'OK' if (shape_ok and finite_ok) else ('SHAPE-MISMATCH' if not shape_ok else 'NAN-OR-INF')
        print(f'[{status}] {fname}  shape={arr.shape}  min={lo:.4f}  max={hi:.4f}')
        if not shape_ok:
            print(f'         expected shape {exp_shape}')
            all_ok = False
        if not finite_ok:
            print(f'         contains NaN/inf values')
            all_ok = False

    idx_p = Path(DATA_OUT) / 'idx.npy'
    if idx_p.exists():
        idx_vals = np.load(idx_p)
        if not np.array_equal(idx_vals, np.arange(N_EVAL)):
            print(
                f'[FAIL] idx.npy: values are not exactly ordered 0..{N_EVAL - 1} '
                f'(got range [{int(idx_vals.min())}, {int(idx_vals.max())}], '
                f'{len(idx_vals)} entries)'
            )
            all_ok = False
        else:
            print(f'[OK]   idx.npy values: exactly ordered 0..{N_EVAL - 1}')

    if not all_ok:
        sys.exit('[error] Post-inference validation failed. See above.')
    print('[validate] All output arrays present, correctly shaped, and finite.')

    method_name = f'candidateB_factorial_{variant}_expanded2688'
    print()
    print('=' * 64)
    print('Run complete.')
    print(f'  Variant          : {variant}')
    print(f'  Final checkpoint : {saved_model}')
    print(f'  SR outputs       : {DATA_OUT}/dataSR.npy')
    print(f'  GT outputs       : {DATA_OUT}/dataGT.npy')
    print(f'  LR inputs        : {DATA_OUT}/dataIN.npy')
    print(f'  Sample indices   : {DATA_OUT}/idx.npy')
    print()
    print('Next steps:')
    print('  Cheap scalar/domain evaluation (compare to CNN, GAN, UV-2688, B-2688, C-2688):')
    print('    python3 scripts/evaluate_finetune_candidate.py \\')
    print(f'      --candidate-name {method_name} \\')
    print(f'      --candidate-dir  {DATA_OUT} \\')
    print('      --cnn-dir        data_out_fixed/wind_mrhr_cnn \\')
    print('      --gan-dir        data_out_fixed/wind_mrhr_gan \\')
    print('      --merged-csv     ttk_runs_fixed/combined/psnr_topology_physics_merged.csv \\')
    print(f'      --out-dir        ttk_runs_fixed/topology_finetuning/{method_name}_eval')
    print()
    print('  Only run the expensive TTK topology pipeline AFTER confirming')
    print('  the cheap evaluation above is non-catastrophic:')
    print('    bash scripts/run_candidate_topology_pipeline.sh \\')
    print(f'      --method     {method_name} \\')
    print(f'      --data-dir   {DATA_OUT} \\')
    print(f'      --vti-dir    ttk_runs_fixed/topology_finetuning/{method_name}_topology_vti \\')
    print(f'      --out-base   ttk_runs_fixed/topology_finetuning/{method_name}_topology \\')
    print('      --n-samples  168 \\')
    print('      --threads    1 \\')
    print('      --skip-viz')
    print('=' * 64)


if __name__ == '__main__':
    main()
