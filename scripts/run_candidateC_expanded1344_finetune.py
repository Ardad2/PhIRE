#!/usr/bin/env python3
"""
Candidate C-expanded-1344: Candidate C topology losses, trained on the
expanded 1344-sample seasonal dataset, evaluated on the 168-sample benchmark.

This script:
  1. Fine-tunes the pretrained vector [u,v] CNN for 3 epochs using the
     Candidate C1 physics/topology loss configuration:
       lambda_speed    = 0.01
       lambda_grad     = 0.05
       lambda_wpd      = 0.0
       lambda_levelset = 0.25
       lambda_crit     = 0.001   (critical-value / topological-extrema proxy)
       crit_high_z     = 1.0
       crit_pool       = 3
     Training data: example_data_topology_expanded_1344/wind_MR-HR.tfrecord
       (1344 seasonally diverse samples; 8 windows × 168 hours:
        WTK 336-503, 504-671, 2160-2327, 2328-2495,
            4344-4511, 4512-4679, 6552-6719, 6720-6887;
        no overlap with the 168-sample benchmark)
  2. Saves per-epoch checkpoints to:
       models_fixed/topology_finetuning/wind_finetune_candidateC_expanded1344/
  3. Runs paired inference from the final checkpoint on the original corrected
     168-sample benchmark:
       example_data_fixed/wind_MR-HR.tfrecord
  4. Writes SR/GT/IN/idx outputs to:
       data_out/wind_finetune_candidateC_expanded1344/

Scientific motivation:
  Scaling Candidate C-expanded-672 to 1344 training samples (two windows per
  season instead of one) tests whether intra-season diversity provides
  additional gains beyond inter-season diversity.  All hyperparameters and
  loss weights are identical to candidateC_expanded672 so the comparison is
  controlled: the only difference is the volume of training data (1344 vs 672).

Normalisation constants are unchanged from the pretrained checkpoint:
  mu  = [0.7684, -0.4575]
  sig = [5.02455, 5.9017]

Protected directories — never overwritten:
  models/wind_mr-hr/trained_cnn/           (source checkpoint)
  example_data_fixed/                      (168-sample benchmark data)
  data_out_fixed/wind_mrhr_cnn/            (baseline CNN outputs)
  data_out_fixed/wind_mrhr_gan/            (baseline GAN outputs)
  data_out/wind_finetune_pilot_candidateB/
  data_out/wind_finetune_pilot_candidateC/
  data_out/wind_finetune_pilot_candidateD/
  data_out/wind_finetune_pilot_candidateE/
  data_out/wind_finetune_pilot_candidateE2/
  data_out/wind_finetune_pilot_candidateUV/
  data_out/wind_finetune_candidateC_expanded672/
  models_fixed/topology_finetuning/wind_finetune_pilot_candidateB/
  models_fixed/topology_finetuning/wind_finetune_pilot_candidateC/
  models_fixed/topology_finetuning/wind_finetune_pilot_candidateD/
  models_fixed/topology_finetuning/wind_finetune_pilot_candidateE/
  models_fixed/topology_finetuning/wind_finetune_pilot_candidateE2/
  models_fixed/topology_finetuning/wind_finetune_pilot_candidateUV/
  models_fixed/topology_finetuning/wind_finetune_candidateC_expanded672/

Usage (from repo root, on Spark):
  python3 scripts/run_candidateC_expanded1344_finetune.py \\
    2>&1 | tee logs/wind_finetune_candidateC_expanded1344.log

Or let the script handle logging automatically (default):
  python3 scripts/run_candidateC_expanded1344_finetune.py
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
LOG_PATH = REPO_ROOT / 'logs' / 'wind_finetune_candidateC_expanded1344.log'

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

# Training data: expanded 1344-sample seasonal dataset
TRAIN_DATA_PATH = 'example_data_topology_expanded_1344/wind_MR-HR.tfrecord'

# Inference/evaluation data: original corrected 168-sample benchmark
EVAL_DATA_PATH  = 'example_data_fixed/wind_MR-HR.tfrecord'

# Pretrained source checkpoint (read-only)
MODEL_PATH = 'models/wind_mr-hr/trained_cnn/cnn'

# Outputs
MODEL_OUT = 'models_fixed/topology_finetuning/wind_finetune_candidateC_expanded1344'
DATA_OUT  = 'data_out/wind_finetune_candidateC_expanded1344'

_PROTECTED = [
    # Source checkpoint
    'models/wind_mr-hr/trained_cnn',
    # Benchmark data
    'example_data_fixed',
    # Baseline model outputs
    'data_out_fixed/wind_mrhr_cnn',
    'data_out_fixed/wind_mrhr_gan',
    # Previous candidate outputs (data)
    'data_out/wind_finetune_pilot_candidateB',
    'data_out/wind_finetune_pilot_candidateC',
    'data_out/wind_finetune_pilot_candidateD',
    'data_out/wind_finetune_pilot_candidateE',
    'data_out/wind_finetune_pilot_candidateE2',
    'data_out/wind_finetune_pilot_candidateUV',
    'data_out/wind_finetune_candidateC_expanded672',
    # Previous candidate outputs (models)
    'models_fixed/topology_finetuning/wind_finetune_pilot_candidateB',
    'models_fixed/topology_finetuning/wind_finetune_pilot_candidateC',
    'models_fixed/topology_finetuning/wind_finetune_pilot_candidateD',
    'models_fixed/topology_finetuning/wind_finetune_pilot_candidateE',
    'models_fixed/topology_finetuning/wind_finetune_pilot_candidateE2',
    'models_fixed/topology_finetuning/wind_finetune_pilot_candidateUV',
    'models_fixed/topology_finetuning/wind_finetune_candidateC_expanded672',
]

# ---------------------------------------------------------------------------
# Normalisation constants (pretrained checkpoint; unchanged)
# ---------------------------------------------------------------------------
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
    print('Candidate C-expanded-1344 fine-tuning')
    print('=' * 64)
    print(f'  Training data     : {TRAIN_DATA_PATH}')
    print(f'  Evaluation data   : {EVAL_DATA_PATH}')
    print(f'  Source checkpoint : {MODEL_PATH}')
    print(f'  Output model dir  : {MODEL_OUT}')
    print(f'  Inference output  : {DATA_OUT}')
    print(f'  Log file          : {LOG_PATH}')
    print()
    print('  Loss configuration (Candidate C1):')
    print('  lambda_speed    = 0.01')
    print('  lambda_grad     = 0.05')
    print('  lambda_wpd      = 0.0')
    print('  lambda_levelset = 0.25')
    print('  lambda_crit     = 0.001  (critical-value proxy; pool=3, z=1.0)')
    print('  crit_high_z     = 1.0   (adaptive threshold: mean + 1σ)')
    print('  crit_pool       = 3')
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
    # Verify required files exist
    # -----------------------------------------------------------------------
    if not Path(TRAIN_DATA_PATH).exists():
        sys.exit(
            f'[error] Training TFRecord not found: {TRAIN_DATA_PATH}\n'
            '  Run scripts/build_wind_mrhr_expanded_dataset_1344.py first:\n'
            '    python3 scripts/build_wind_mrhr_expanded_dataset_1344.py \\\n'
            '      --out-dir example_data_topology_expanded_1344'
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
    # Phase 1: Fine-tuning on the 1344-sample expanded dataset.
    # Loss configuration is identical to Candidate C1 and candidateC_expanded672.
    # diagnostic_mode=True logs per-iteration loss breakdown.
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
        lambda_crit          = 0.001,
        crit_high_z          = 1.0,
        crit_include_minima  = False,
        crit_low_z           = -1.0,
        crit_pool            = 3,
        diagnostic_mode      = True,
    )

    phire = PhIREGANs(
        data_type    = 'wind_finetune_candidateC_expanded1344',
        learning_rate= 1e-5,
        N_epochs     = 3,
        save_every   = 1,
        print_every  = 2,
        mu_sig       = MU_SIG,
    )
    phire.setModel_name(MODEL_OUT)

    print('Phase 1: Fine-tuning on 1344-sample expanded dataset …')
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
    # comparable with all other candidates evaluated on the same 168 samples.
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
    print('  Evaluate (compare to CNN baseline and candidateC_expanded672):')
    print('    python3 scripts/evaluate_finetune_candidate.py \\')
    print('      --candidate-name candidateC_expanded1344 \\')
    print('      --candidate-dir  data_out/wind_finetune_candidateC_expanded1344 \\')
    print('      --cnn-dir        data_out_fixed/wind_mrhr_cnn \\')
    print('      --gan-dir        data_out_fixed/wind_mrhr_gan \\')
    print('      --merged-csv     ttk_runs_fixed/combined/psnr_topology_physics_merged.csv \\')
    print('      --out-dir        ttk_runs_fixed/topology_finetuning/candidateC_expanded1344_eval')
    print()
    print('  TTK topology pipeline:')
    print('    bash scripts/run_candidate_topology_pipeline.sh \\')
    print('      --method   candidateC_expanded1344 \\')
    print('      --data-dir data_out/wind_finetune_candidateC_expanded1344')
    print('=' * 64)


if __name__ == '__main__':
    main()
