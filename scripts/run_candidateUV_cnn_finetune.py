#!/usr/bin/env python3
r"""
UV-only ablation fine-tuning — Candidate UV.

ABLATION RUN — tests whether B/C improvements are due to auxiliary
physics/topology losses or simply to fine-tuning on the 168-sample
evaluation set.

This script:
  1. Fine-tunes the pretrained vector [u,v] CNN for 3 epochs using ONLY
     the direct reconstruction loss (L_uv).  All auxiliary losses are
     disabled (lambda = 0):
       lambda_speed    = 0.0
       lambda_grad     = 0.0
       lambda_wpd      = 0.0
       lambda_levelset = 0.0
       lambda_crit     = 0.0
     Auxiliary losses are still computed and logged in diagnostic_mode
     for comparison, but contribute zero weight to training.
  2. Saves per-epoch checkpoints to:
       models_fixed/topology_finetuning/wind_finetune_pilot_candidateUV/
  3. Runs paired inference from the final checkpoint on all 168 samples.
  4. Writes SR outputs to:
       data_out/wind_finetune_pilot_candidateUV/

All other settings match Candidate C exactly:
  - Same pretrained CNN checkpoint (models/wind_mr-hr/trained_cnn/cnn)
  - Same 168 samples / idx.npy order
  - Same learning rate (1e-5), batch size (4), optimizer (Adam)
  - Same 3 epochs

Protected outputs — never overwritten:
  models/wind_mr-hr/trained_cnn/           (source checkpoint)
  data_out_fixed/wind_mrhr_cnn/            (baseline CNN outputs)
  data_out_fixed/wind_mrhr_gan/            (baseline GAN outputs)
  data_out/wind_finetune_pilot_candidateB/
  data_out/wind_finetune_pilot_candidateC/
  data_out/wind_finetune_pilot_candidateD/
  data_out/wind_finetune_pilot_candidateE/
  data_out/wind_finetune_pilot_candidateE2/
  models_fixed/topology_finetuning/wind_finetune_pilot_candidateB/
  models_fixed/topology_finetuning/wind_finetune_pilot_candidateC/
  models_fixed/topology_finetuning/wind_finetune_pilot_candidateD/
  models_fixed/topology_finetuning/wind_finetune_pilot_candidateE/
  models_fixed/topology_finetuning/wind_finetune_pilot_candidateE2/

Usage (from repo root):
  python3 scripts/run_candidateUV_cnn_finetune.py 2>&1 | tee logs/wind_finetune_pilot_candidateUV.log

Or let the script handle logging automatically (default):
  python3 scripts/run_candidateUV_cnn_finetune.py
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure repo root is on sys.path when called as scripts/...
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT   = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Tee: mirror stdout to a log file.
# ---------------------------------------------------------------------------
LOG_PATH = REPO_ROOT / 'logs' / 'wind_finetune_pilot_candidateUV.log'

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
# Imports
# ---------------------------------------------------------------------------
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from sr_network import PhysicsLossConfig
from PhIREGANs import PhIREGANs

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_PATH  = 'example_data_fixed/wind_MR-HR.tfrecord'
MODEL_PATH = 'models/wind_mr-hr/trained_cnn/cnn'
MODEL_OUT  = 'models_fixed/topology_finetuning/wind_finetune_pilot_candidateUV'
DATA_OUT   = 'data_out/wind_finetune_pilot_candidateUV'

_PROTECTED = [
    'models/wind_mr-hr/trained_cnn',
    'data_out_fixed/wind_mrhr_cnn',
    'data_out_fixed/wind_mrhr_gan',
    'data_out/wind_finetune_pilot_candidateB',
    'data_out/wind_finetune_pilot_candidateC',
    'data_out/wind_finetune_pilot_candidateD',
    'data_out/wind_finetune_pilot_candidateE',
    'data_out/wind_finetune_pilot_candidateE2',
    'models_fixed/topology_finetuning/wind_finetune_pilot_candidateB',
    'models_fixed/topology_finetuning/wind_finetune_pilot_candidateC',
    'models_fixed/topology_finetuning/wind_finetune_pilot_candidateD',
    'models_fixed/topology_finetuning/wind_finetune_pilot_candidateE',
    'models_fixed/topology_finetuning/wind_finetune_pilot_candidateE2',
]

# ---------------------------------------------------------------------------
# Normalisation constants (vector [u,v] model, paper baseline)
# ---------------------------------------------------------------------------
MU_SIG = [[0.7684, -0.4575], [5.02455, 5.9017]]
R      = [5]


def _safety_check() -> None:
    for protected in _PROTECTED:
        for candidate in (MODEL_OUT, DATA_OUT):
            p = Path(protected).resolve()
            c = Path(candidate).resolve()
            if c == p or str(c).startswith(str(p) + os.sep):
                sys.exit(
                    f'[error] Output path {candidate!r} overlaps protected path '
                    f'{protected!r}. Aborting to avoid overwriting baselines.'
                )
    print('[safety] Output paths do not overlap any protected baseline directory. OK.')


def main() -> None:
    _safety_check()

    print('=' * 64)
    print('PhIRE UV-only ablation fine-tuning — Candidate UV')
    print('=' * 64)
    print(f'  Source checkpoint : {MODEL_PATH}')
    print(f'  Output model dir  : {MODEL_OUT}')
    print(f'  Inference output  : {DATA_OUT}')
    print(f'  Log file          : {LOG_PATH}')
    print()
    print('  ABLATION: L_total = L_uv only')
    print('  lambda_speed    = 0.0  (disabled)')
    print('  lambda_grad     = 0.0  (disabled)')
    print('  lambda_wpd      = 0.0  (disabled)')
    print('  lambda_levelset = 0.0  (disabled)')
    print('  lambda_crit     = 0.0  (disabled)')
    print('  [Auxiliary losses are logged in diagnostic_mode but not trained on]')
    print()
    print('  learning_rate   = 1e-5')
    print('  N_epochs        = 3')
    print('  save_every      = 1')
    print('  print_every     = 2')
    print('  batch_size      = 4')
    print('=' * 64)
    print()

    if not Path(DATA_PATH).exists():
        sys.exit(f'[error] TFRecord not found: {DATA_PATH}')
    for ext in ('.index', '.meta'):
        if not Path(MODEL_PATH + ext).exists():
            sys.exit(f'[error] Checkpoint not found: {MODEL_PATH}{ext}')

    # -----------------------------------------------------------------------
    # Phase 1: Fine-tuning — L_uv only.
    # All auxiliary lambda values are 0.0; diagnostic_mode logs their raw
    # magnitudes so we can verify they are not driving the update.
    # -----------------------------------------------------------------------
    cfg = PhysicsLossConfig(
        use_aux_losses       = True,
        mu                   = [0.7684, -0.4575],
        sig                  = [5.02455, 5.9017],
        lambda_speed         = 0.0,
        lambda_grad          = 0.0,
        lambda_wpd           = 0.0,
        lambda_levelset      = 0.0,
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
        data_type    = 'wind_finetune_pilot_candidateUV',
        learning_rate= 1e-5,
        N_epochs     = 3,
        save_every   = 1,
        print_every  = 2,
        mu_sig       = MU_SIG,
    )
    phire.setModel_name(MODEL_OUT)

    print('Starting UV-only fine-tuning ...')
    saved_model = phire.pretrain(
        r          = R,
        data_path  = DATA_PATH,
        model_path = MODEL_PATH,
        batch_size = 4,
        phys_cfg   = cfg,
    )

    print()
    print(f'Fine-tuning complete. Final checkpoint: {saved_model}')
    print()

    # -----------------------------------------------------------------------
    # Phase 2: Paired inference from the final checkpoint.
    # -----------------------------------------------------------------------
    print('=' * 64)
    print('Phase 2: Paired inference from fine-tuned checkpoint')
    print('=' * 64)
    print(f'  Checkpoint : {saved_model}')
    print(f'  Output dir : {DATA_OUT}')
    print()

    phire.set_data_out_path(DATA_OUT)

    phire.test_paired(
        r          = R,
        data_path  = DATA_PATH,
        model_path = saved_model,
        batch_size = 1,
        save_inputs= True,
    )

    print()
    print('=' * 64)
    print('Pilot run complete.')
    print(f'  Final checkpoint : {saved_model}')
    print(f'  SR outputs       : {DATA_OUT}/dataSR.npy')
    print(f'  GT outputs       : {DATA_OUT}/dataGT.npy')
    print(f'  LR inputs        : {DATA_OUT}/dataIN.npy')
    print(f'  Sample indices   : {DATA_OUT}/idx.npy')
    print()
    print('Next steps:')
    print('  Evaluate:')
    print('    python3 scripts/evaluate_finetune_candidate.py \\')
    print('      --candidate-name candidateUV \\')
    print('      --candidate-dir  data_out/wind_finetune_pilot_candidateUV \\')
    print('      --cnn-dir        data_out_fixed/wind_mrhr_cnn \\')
    print('      --gan-dir        data_out_fixed/wind_mrhr_gan \\')
    print('      --merged-csv     ttk_runs_fixed/combined/psnr_topology_physics_merged.csv \\')
    print('      --out-dir        ttk_runs_fixed/topology_finetuning/candidateUV_eval')
    print()
    print('  TTK topology pipeline:')
    print('    bash scripts/run_candidate_topology_pipeline.sh \\')
    print('      --method candidateUV \\')
    print('      --data-dir data_out/wind_finetune_pilot_candidateUV')
    print('=' * 64)


if __name__ == '__main__':
    main()
