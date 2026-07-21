#!/usr/bin/env python3
"""
Candidate B-expanded-2688: Candidate B physics/level-set losses, trained on
the expanded 2688-sample seasonal dataset, evaluated on the 168-sample
benchmark.

This fills the missing rung of the Candidate B scale ladder (672 already
exists at scripts/run_candidateB_expanded672_finetune.py) at the same
2688-sample scale already used for Candidate C
(scripts/run_candidateC_expanded2688_finetune.py), so Candidate B and
Candidate C can finally be compared directly at 2688 samples -- exactly
mirroring the 672/1344/2688 scale ladder already completed for Candidate C
and for the repaired-E2 families (B+E2-low, C+E2-low, UV+E2-low, UV+crit).

This script:
  1. Fine-tunes the pretrained vector [u,v] CNN for 3 epochs using the
     Candidate B physics loss configuration:
       lambda_speed    = 0.01
       lambda_grad     = 0.05
       lambda_wpd      = 0.0
       lambda_levelset = 0.25
       lambda_crit     = 0.0   (disabled; no critical-value proxy)
     No repaired-E2 TTK fixed-index losses (L_TTKCV, L_TTKpers) are built or
     used in this script at all -- there is no constraints NPZ and no custom
     training loop, exactly like candidateC_expanded2688 and
     candidateB_expanded672.
     Training data: example_data_topology_expanded_2688/wind_MR-HR.tfrecord
       (the SAME 2688-sample expanded seasonal dataset used by
        candidateC_expanded2688 -- 16 windows x 168 hours:
        WTK 336-503, 504-671, 672-839, 840-1007 (winter),
            2160-2327, 2328-2495, 2496-2663, 2664-2831 (spring),
            4344-4511, 4512-4679, 4680-4847, 4848-5015 (summer),
            6552-6719, 6720-6887, 6888-7055, 7056-7223 (fall);
        no overlap with the 168-sample benchmark)
  2. Saves per-epoch checkpoints to:
       models_fixed/topology_finetuning/wind_finetune_candidateB_expanded2688/
  3. Runs paired inference from the final checkpoint on the original corrected
     168-sample benchmark:
       example_data_fixed/wind_MR-HR.tfrecord
  4. Writes SR/GT/IN/idx outputs to:
       data_out/wind_finetune_candidateB_expanded2688/

Loss/normalization conventions (unchanged from every other native-TF
candidate script in this repo, in particular candidateC_expanded2688 and
candidateB_expanded672):
  - The model predicts NORMALIZED [u, v]. L_uv (built into
    sr_network.SR_NETWORK / PhysicsLossConfig, not re-implemented here) is
    MSE on normalized [u, v].
  - L_speed and L_grad are computed on DENORMALIZED physical scalar speed
    (sqrt(u_phys^2 + v_phys^2)), via the same PhysicsLossConfig machinery
    Candidate B/C already use.
  - L_levelset uses the same soft high-speed mask thresholds
    {5, 10, 15} m/s as every other Candidate B/C script.
  - lambda_crit = 0.0: the L_crit op is still constructed by
    PhysicsLossConfig (diagnostic_mode=True exposes its raw value), but
    contributes exactly 0.0 to the training loss -- this script does not
    edit sr_network.py or PhIREGANs.py.
  - No L_TTKCV / L_TTKpers ops exist in this script at all (unlike the
    B+E2/C+E2/UV+E2 family scripts, which build a custom training loop to
    feed TTK fixed-index constraints). This script uses the plain
    PhIREGANs.pretrain() path, exactly like candidateB_expanded672 and
    candidateC_expanded2688.

Scientific motivation:
  Candidate C-expanded-2688 (lambda_crit=0.001) is the strongest completed
  PD-oriented result in this repo's controlled scale ladder. Candidate
  B-expanded-2688 (lambda_crit=0.0, otherwise identical) isolates the
  contribution of L_crit at the SAME 2688-sample scale, completing the
  matched B-vs-C comparison already available at 672 (B-672 vs C-672) and
  giving a Candidate-B counterpart to the completed B+E2-low-2688 /
  C+E2-low-2688 / UV+E2-low-2688 / UV+crit-2688 2688-scale results.

Normalisation constants are unchanged from the pretrained checkpoint:
  mu  = [0.7684, -0.4575]
  sig = [5.02455, 5.9017]

Protected directories -- never overwritten:
  models/wind_mr-hr/trained_cnn/
  example_data_fixed/
  data_out_fixed/wind_mrhr_cnn/
  data_out_fixed/wind_mrhr_gan/
  All existing pilot/expanded/repaired-E2/B+E2/C+E2/UV+E2/UV+crit candidate
  output directories (data_out/ and models_fixed/topology_finetuning/) --
  see _PROTECTED below for the exhaustive list, derived directly from every
  MODEL_OUT/DATA_OUT literal defined anywhere in scripts/*.py at the time
  this script was written.

Usage (from repo root, on Spark):
  python3 scripts/run_candidateB_expanded2688_finetune.py \\
    2>&1 | tee logs/wind_finetune_candidateB_expanded2688.log

Or let the script handle logging automatically (default):
  python3 scripts/run_candidateB_expanded2688_finetune.py
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure repo root is on sys.path
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT   = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Tee: mirror stdout to a log file
# ---------------------------------------------------------------------------
LOG_PATH = REPO_ROOT / 'logs' / 'wind_finetune_candidateB_expanded2688.log'


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

# ---------------------------------------------------------------------------
# Imports (after tee is active)
# ---------------------------------------------------------------------------
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from sr_network import PhysicsLossConfig
from PhIREGANs import PhIREGANs

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Training data: the exact same expanded 2688-sample seasonal dataset used
# by candidateC_expanded2688 -- reused unchanged.
TRAIN_DATA_PATH = 'example_data_topology_expanded_2688/wind_MR-HR.tfrecord'

# Inference/evaluation data: original corrected 168-sample benchmark
EVAL_DATA_PATH = 'example_data_fixed/wind_MR-HR.tfrecord'

# Pretrained source checkpoint (read-only) -- same as every other candidate
MODEL_PATH = 'models/wind_mr-hr/trained_cnn/cnn'

# Outputs -- new, distinct names; do not overwrite any existing candidate
MODEL_OUT = 'models_fixed/topology_finetuning/wind_finetune_candidateB_expanded2688'
DATA_OUT  = 'data_out/wind_finetune_candidateB_expanded2688'

# Exhaustive list of every existing candidate MODEL_OUT/DATA_OUT literal
# defined anywhere in scripts/*.py at the time this script was written
# (derived via `grep -rhoE 'data_out/wind_finetune_[A-Za-z0-9_]+' scripts/*.py`
# and the models_fixed/topology_finetuning/ equivalent). Candidate C-2688 in
# particular is the direct comparison partner for this run and must never be
# touched.
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

# ---------------------------------------------------------------------------
# Normalisation constants (pretrained checkpoint; unchanged)
# ---------------------------------------------------------------------------
MU_SIG = [[0.7684, -0.4575], [5.02455, 5.9017]]
R      = [5]

# ---------------------------------------------------------------------------
# Dataset size parameters (for preflight checks)
# ---------------------------------------------------------------------------
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
    """Count Examples in a TFRecord via pure protobuf iteration (no TF graph,
    no decoding of the large LR/HR payloads)."""
    n = 0
    for _ in tf.python_io.tf_record_iterator(data_path):
        n += 1
    return n


def main() -> None:
    _safety_check()

    print('=' * 64)
    print('Candidate B-expanded-2688 fine-tuning')
    print('=' * 64)
    print(f'  Training data     : {TRAIN_DATA_PATH}')
    print(f'  Evaluation data   : {EVAL_DATA_PATH}')
    print(f'  Source checkpoint : {MODEL_PATH}')
    print(f'  Output model dir  : {MODEL_OUT}')
    print(f'  Inference output  : {DATA_OUT}')
    print(f'  Log file          : {LOG_PATH}')
    print()
    print('  Loss configuration (Candidate B):')
    print('  lambda_speed    = 0.01')
    print('  lambda_grad     = 0.05')
    print('  lambda_wpd      = 0.0')
    print('  lambda_levelset = 0.25')
    print('  lambda_crit     = 0.0  (DISABLED -- this is Candidate B, not Candidate C)')
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
    print(f'  Training TFRecord  : {n_records} records (same file as candidateC_expanded2688)')
    if n_records != N_TRAIN:
        sys.exit(
            f'[error] Training TFRecord has {n_records} records; expected {N_TRAIN}.'
        )
    print(f'  Confirmed: {N_TRAIN}-sample training set matches candidateC_expanded2688.')
    print('  Confirmed: evaluation benchmark is the fixed 168-sample benchmark.')
    print('  Confirmed: lambda_crit = 0.0 (Candidate C\'s critical-value proxy is OFF).')
    print('  Confirmed: no L_TTKCV/L_TTKpers ops are built in this script (repaired-E2 OFF).')
    print('=' * 64)
    print()

    # -----------------------------------------------------------------------
    # Phase 1: Fine-tuning on the 2688-sample expanded dataset.
    # Candidate B: speed + gradient + level-set losses; lambda_crit = 0.
    # Identical hyperparameters/normalization to candidateC_expanded2688 --
    # the only intended difference is lambda_crit (0.0 here vs 0.001 there).
    # -----------------------------------------------------------------------
    cfg = PhysicsLossConfig(
        use_aux_losses       = True,
        mu                   = MU_SIG[0],
        sig                  = MU_SIG[1],
        lambda_speed         = 0.01,
        lambda_grad          = 0.05,
        lambda_wpd           = 0.0,
        lambda_levelset      = 0.25,
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
        data_type    = 'wind_finetune_candidateB_expanded2688',
        learning_rate= 1e-5,
        N_epochs     = 3,
        save_every   = 1,
        print_every  = 2,
        mu_sig       = MU_SIG,
    )
    phire.setModel_name(MODEL_OUT)

    print('Phase 1: Fine-tuning on 2688-sample expanded dataset …')
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
    # Uses example_data_fixed/wind_MR-HR.tfrecord so results are directly
    # comparable with candidateC_expanded2688 and every other candidate.
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

    print()
    print('=' * 64)
    print('Run complete.')
    print(f'  Final checkpoint : {saved_model}')
    print(f'  SR outputs       : {DATA_OUT}/dataSR.npy')
    print(f'  GT outputs       : {DATA_OUT}/dataGT.npy')
    print(f'  LR inputs        : {DATA_OUT}/dataIN.npy')
    print(f'  Sample indices   : {DATA_OUT}/idx.npy')
    print()
    print('Next steps:')
    print('  Cheap scalar/domain evaluation (compare to CNN, GAN, and candidateC_expanded2688):')
    print('    python3 scripts/evaluate_finetune_candidate.py \\')
    print('      --candidate-name candidateB_expanded2688 \\')
    print('      --candidate-dir  data_out/wind_finetune_candidateB_expanded2688 \\')
    print('      --cnn-dir        data_out_fixed/wind_mrhr_cnn \\')
    print('      --gan-dir        data_out_fixed/wind_mrhr_gan \\')
    print('      --merged-csv     ttk_runs_fixed/combined/psnr_topology_physics_merged.csv \\')
    print('      --out-dir        ttk_runs_fixed/topology_finetuning/candidateB_expanded2688_eval')
    print()
    print('  Only run the expensive TTK topology pipeline AFTER confirming')
    print('  the cheap evaluation above is non-catastrophic:')
    print('    bash scripts/run_candidate_topology_pipeline.sh \\')
    print('      --method     candidateB_expanded2688 \\')
    print('      --data-dir   data_out/wind_finetune_candidateB_expanded2688 \\')
    print('      --vti-dir    ttk_runs_fixed/topology_finetuning/candidateB_expanded2688_topology_vti \\')
    print('      --out-base   ttk_runs_fixed/topology_finetuning/candidateB_expanded2688_topology \\')
    print('      --n-samples  168 \\')
    print('      --threads    1 \\')
    print('      --skip-viz')
    print()
    print('  This completes the Candidate B scale ladder (672 already exists) and')
    print('  gives the matched B-vs-C comparison at 2688 samples. If the sublevel')
    print('  topology evaluation above completes cleanly, candidateB_expanded2688')
    print('  can optionally be added to scripts/run_superlevel_topology_robustness.py')
    print('  DEFAULT_METHODS as an additional superlevel robustness check.')
    print('=' * 64)


if __name__ == '__main__':
    main()
