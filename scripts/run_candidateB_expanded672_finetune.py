#!/usr/bin/env python3
"""
Candidate B-expanded-672: Candidate B physics/level-set losses, trained on
the expanded 672-sample seasonal dataset, evaluated on the 168-sample benchmark.

This script:
  1. Fine-tunes the pretrained vector [u,v] CNN for 3 epochs using the
     Candidate B physics loss configuration:
       lambda_speed    = 0.01
       lambda_grad     = 0.05
       lambda_wpd      = 0.0
       lambda_levelset = 0.25
       lambda_crit     = 0.0   (disabled; no critical-value proxy)
     Training data: example_data_topology_expanded_672/wind_MR-HR.tfrecord
       (672 seasonally diverse samples; WTK indices 336-503, 2160-2327,
        4344-4511, 6552-6719; no overlap with the 168-sample benchmark)
  2. Saves per-epoch checkpoints to:
       models_fixed/topology_finetuning/wind_finetune_candidateB_expanded672/
  3. Runs paired inference from the final checkpoint on the original corrected
     168-sample benchmark:
       example_data_fixed/wind_MR-HR.tfrecord
  4. Writes SR/GT/IN/idx outputs to:
       data_out/wind_finetune_candidateB_expanded672/

Scientific motivation:
  Candidate B applies speed, gradient, and level-set auxiliary losses without
  the critical-value proxy (L_crit = 0).  Training on 672 seasonally diverse
  samples and evaluating on the 168-sample benchmark tests whether the physics
  losses alone generalise across seasons.  Comparison with
  candidateC_expanded672 isolates the contribution of L_crit; comparison with
  candidateB (original, trained on 168 samples) isolates the effect of training
  set size and diversity.

Normalisation constants are unchanged from the pretrained checkpoint:
  mu  = [0.7684, -0.4575]
  sig = [5.02455, 5.9017]

Protected directories — never overwritten:
  models/wind_mr-hr/trained_cnn/
  example_data_fixed/
  data_out_fixed/wind_mrhr_cnn/
  data_out_fixed/wind_mrhr_gan/
  data_out/wind_finetune_pilot_candidateB/
  data_out/wind_finetune_pilot_candidateC/
  data_out/wind_finetune_pilot_candidateD/
  data_out/wind_finetune_pilot_candidateE/
  data_out/wind_finetune_pilot_candidateE2/
  data_out/wind_finetune_pilot_candidateUV/
  data_out/wind_finetune_candidateC_expanded672/
  data_out/wind_finetune_candidateUV_expanded672/
  models_fixed/topology_finetuning/wind_finetune_pilot_candidateB/
  models_fixed/topology_finetuning/wind_finetune_pilot_candidateC/
  models_fixed/topology_finetuning/wind_finetune_pilot_candidateD/
  models_fixed/topology_finetuning/wind_finetune_pilot_candidateE/
  models_fixed/topology_finetuning/wind_finetune_pilot_candidateE2/
  models_fixed/topology_finetuning/wind_finetune_pilot_candidateUV/
  models_fixed/topology_finetuning/wind_finetune_candidateC_expanded672/
  models_fixed/topology_finetuning/wind_finetune_candidateUV_expanded672/

Usage (from repo root, on Spark):
  python3 scripts/run_candidateB_expanded672_finetune.py \\
    2>&1 | tee logs/wind_finetune_candidateB_expanded672.log

Or let the script handle logging automatically (default):
  python3 scripts/run_candidateB_expanded672_finetune.py
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT   = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

LOG_PATH = REPO_ROOT / 'logs' / 'wind_finetune_candidateB_expanded672.log'

class _Tee:
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

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from sr_network import PhysicsLossConfig
from PhIREGANs import PhIREGANs

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TRAIN_DATA_PATH = 'example_data_topology_expanded_672/wind_MR-HR.tfrecord'
EVAL_DATA_PATH  = 'example_data_fixed/wind_MR-HR.tfrecord'
MODEL_PATH      = 'models/wind_mr-hr/trained_cnn/cnn'
MODEL_OUT       = 'models_fixed/topology_finetuning/wind_finetune_candidateB_expanded672'
DATA_OUT        = 'data_out/wind_finetune_candidateB_expanded672'

_PROTECTED = [
    'models/wind_mr-hr/trained_cnn',
    'example_data_fixed',
    'data_out_fixed/wind_mrhr_cnn',
    'data_out_fixed/wind_mrhr_gan',
    'data_out/wind_finetune_pilot_candidateB',
    'data_out/wind_finetune_pilot_candidateC',
    'data_out/wind_finetune_pilot_candidateD',
    'data_out/wind_finetune_pilot_candidateE',
    'data_out/wind_finetune_pilot_candidateE2',
    'data_out/wind_finetune_pilot_candidateUV',
    'data_out/wind_finetune_candidateC_expanded672',
    'data_out/wind_finetune_candidateUV_expanded672',
    'models_fixed/topology_finetuning/wind_finetune_pilot_candidateB',
    'models_fixed/topology_finetuning/wind_finetune_pilot_candidateC',
    'models_fixed/topology_finetuning/wind_finetune_pilot_candidateD',
    'models_fixed/topology_finetuning/wind_finetune_pilot_candidateE',
    'models_fixed/topology_finetuning/wind_finetune_pilot_candidateE2',
    'models_fixed/topology_finetuning/wind_finetune_pilot_candidateUV',
    'models_fixed/topology_finetuning/wind_finetune_candidateC_expanded672',
    'models_fixed/topology_finetuning/wind_finetune_candidateUV_expanded672',
]

MU_SIG = [[0.7684, -0.4575], [5.02455, 5.9017]]
R      = [5]


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


def main() -> None:
    _safety_check()

    print('=' * 64)
    print('Candidate B-expanded-672 fine-tuning')
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
    print('  lambda_crit     = 0.0  (disabled)')
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

    if not Path(TRAIN_DATA_PATH).exists():
        sys.exit(
            f'[error] Training TFRecord not found: {TRAIN_DATA_PATH}\n'
            '  Run scripts/build_wind_mrhr_expanded_dataset.py first:\n'
            '    python3 scripts/build_wind_mrhr_expanded_dataset.py \\\n'
            '      --out-dir example_data_topology_expanded_672'
        )
    if not Path(EVAL_DATA_PATH).exists():
        sys.exit(
            f'[error] Evaluation TFRecord not found: {EVAL_DATA_PATH}\n'
            '  The corrected 168-sample benchmark is required for inference.'
        )
    for ext in ('.index', '.meta'):
        if not Path(MODEL_PATH + ext).exists():
            sys.exit(f'[error] Checkpoint not found: {MODEL_PATH}{ext}')

    # -----------------------------------------------------------------------
    # Phase 1: Fine-tuning on the 672-sample expanded dataset.
    # Candidate B: speed + gradient + level-set losses; lambda_crit = 0.
    # -----------------------------------------------------------------------
    cfg = PhysicsLossConfig(
        use_aux_losses       = True,
        mu                   = [0.7684, -0.4575],
        sig                  = [5.02455, 5.9017],
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
        data_type    = 'wind_finetune_candidateB_expanded672',
        learning_rate= 1e-5,
        N_epochs     = 3,
        save_every   = 1,
        print_every  = 2,
        mu_sig       = MU_SIG,
    )
    phire.setModel_name(MODEL_OUT)

    print('Phase 1: Fine-tuning on 672-sample expanded dataset …')
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
    print('  Evaluate:')
    print('    python3 scripts/evaluate_finetune_candidate.py \\')
    print('      --candidate-name candidateB_expanded672 \\')
    print('      --candidate-dir  data_out/wind_finetune_candidateB_expanded672 \\')
    print('      --cnn-dir        data_out_fixed/wind_mrhr_cnn \\')
    print('      --gan-dir        data_out_fixed/wind_mrhr_gan \\')
    print('      --merged-csv     ttk_runs_fixed/combined/psnr_topology_physics_merged.csv \\')
    print('      --out-dir        ttk_runs_fixed/topology_finetuning/candidateB_expanded672_eval')
    print()
    print('  TTK topology pipeline:')
    print('    bash scripts/run_candidate_topology_pipeline.sh \\')
    print('      --method   candidateB_expanded672 \\')
    print('      --data-dir data_out/wind_finetune_candidateB_expanded672')
    print('=' * 64)


if __name__ == '__main__':
    main()
