#!/usr/bin/env python3
"""
Physics/topology fine-tuning pilot — Candidate C1 (B + critical-value loss).

PILOT RUN ONLY — not the final experiment. This script:
  1. Fine-tunes the pretrained vector [u,v] CNN for 3 epochs using Candidate C1
     physics/topology auxiliary losses:
       lambda_speed    = 0.01
       lambda_grad     = 0.05
       lambda_wpd      = 0.0
       lambda_levelset = 0.25
       lambda_crit     = 0.001   ← new: critical-value / topological-extrema proxy
       crit_high_z     = 1.0
       crit_pool       = 3
  2. Saves per-epoch checkpoints to:
       models_fixed/topology_finetuning/wind_finetune_pilot_candidateC/
  3. Runs paired inference from the final checkpoint on all 168 samples.
  4. Writes SR outputs to:
       data_out/wind_finetune_pilot_candidateC/

Baseline outputs and Candidate B outputs are never overwritten:
  models/wind_mr-hr/trained_cnn/       (read-only source checkpoint)
  data_out_fixed/wind_mrhr_cnn/        (paper baseline CNN outputs)
  data_out_fixed/wind_mrhr_gan/        (paper baseline GAN outputs)
  data_out/wind_finetune_pilot_candidateB/
  models_fixed/topology_finetuning/wind_finetune_pilot_candidateB/

Scientific motivation:
  Candidate C adds a critical-value loss (L_crit) on top of Candidate B.
  L_crit focuses MSE on GT local-speed maxima above a per-sample adaptive
  threshold (mean + crit_high_z * std).  These maxima correspond to births
  of connected components in the superlevel-set filtration, so matching their
  scalar values should help preserve topological feature prominence.
  Inspired by Kissi et al.'s critical-value loss for topology-aware
  scalar-field interpolation; this is a TF1-compatible proxy, not a full
  PD Wasserstein loss.

Usage (from repo root):
  python3 scripts/run_candidateC_crit_finetune.py 2>&1 | tee logs/wind_finetune_pilot_candidateC.log

Or let the script handle logging automatically (default):
  python3 scripts/run_candidateC_crit_finetune.py
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
# Tee: mirror stdout to a log file so the full per-iteration loss breakdown
# is preserved. 3 epochs × 42 batches × print_every=2 → ~63 printed iters.
# ---------------------------------------------------------------------------
LOG_PATH = REPO_ROOT / 'logs' / 'wind_finetune_pilot_candidateC.log'

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
sys.stderr = sys.stdout   # capture TensorFlow warnings/errors in the log too

# ---------------------------------------------------------------------------
# Imports (after sys.path is set and tee is active).
# ---------------------------------------------------------------------------
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from sr_network import PhysicsLossConfig
from PhIREGANs import PhIREGANs

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_PATH   = 'example_data_fixed/wind_MR-HR.tfrecord'
MODEL_PATH  = 'models/wind_mr-hr/trained_cnn/cnn'    # source checkpoint (read-only)
MODEL_OUT   = 'models_fixed/topology_finetuning/wind_finetune_pilot_candidateC'
DATA_OUT    = 'data_out/wind_finetune_pilot_candidateC'

# Baseline and Candidate B outputs — must never be touched.
_PROTECTED = [
    'models/wind_mr-hr/trained_cnn',
    'data_out_fixed/wind_mrhr_cnn',
    'data_out_fixed/wind_mrhr_gan',
    'data_out/wind_finetune_pilot_candidateB',
    'models_fixed/topology_finetuning/wind_finetune_pilot_candidateB',
]

# ---------------------------------------------------------------------------
# Normalisation constants (vector [u,v] model, paper baseline)
# ---------------------------------------------------------------------------
MU_SIG = [[0.7684, -0.4575], [5.02455, 5.9017]]
R      = [5]


def _safety_check() -> None:
    """Abort if output paths would collide with paper baselines or Candidate B."""
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
    print('PhIRE physics/topology fine-tuning pilot — Candidate C1')
    print('=' * 64)
    print(f'  Source checkpoint : {MODEL_PATH}')
    print(f'  Output model dir  : {MODEL_OUT}')
    print(f'  Inference output  : {DATA_OUT}')
    print(f'  Log file          : {LOG_PATH}')
    print()
    print('  lambda_speed   = 0.01')
    print('  lambda_grad    = 0.05')
    print('  lambda_wpd     = 0.0')
    print('  lambda_levelset= 0.25')
    print('  lambda_crit    = 0.001  (critical-value proxy; calibrated: ~19% of L_uv)')
    print('  crit_high_z    = 1.0   (adaptive threshold: mean + 1σ)')
    print('  crit_pool      = 3     (max-pool neighbourhood size)')
    print('  learning_rate  = 1e-5')
    print('  N_epochs       = 3  (pilot; not the final experiment)')
    print('  save_every     = 1')
    print('  print_every    = 2')
    print('  batch_size     = 4')
    print('=' * 64)
    print()

    # -----------------------------------------------------------------------
    # Verify source files exist before starting.
    # -----------------------------------------------------------------------
    if not Path(DATA_PATH).exists():
        sys.exit(f'[error] TFRecord not found: {DATA_PATH}')
    for ext in ('.index', '.meta'):
        if not Path(MODEL_PATH + ext).exists():
            sys.exit(f'[error] Checkpoint not found: {MODEL_PATH}{ext}')

    # -----------------------------------------------------------------------
    # Phase 1: Fine-tuning
    # diagnostic_mode=True keeps the per-iteration loss breakdown visible
    # so we can verify that L_crit converges alongside the other losses.
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
        data_type    = 'wind_finetune_pilot_candidateC',
        learning_rate= 1e-5,
        N_epochs     = 3,
        save_every   = 1,
        print_every  = 2,
        mu_sig       = MU_SIG,
    )
    phire.setModel_name(MODEL_OUT)

    print('Starting fine-tuning ...')
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
    # Uses the same test_paired() path as the Candidate B pilot so output
    # format is identical (idx.npy, dataSR.npy, dataIN.npy, dataGT.npy).
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
    print('  - Compare dataSR.npy speed MAE/PSNR against data_out_fixed/wind_mrhr_cnn/')
    print('    and data_out/wind_finetune_pilot_candidateB/')
    print('  - Run scripts/evaluate_finetune_candidateB.py with --candidate-dir pointing')
    print('    to this output directory to compare all three candidates')
    print('  - Run scripts/run_candidateB_topology_pipeline.sh (adapted for candidateC)')
    print('    to compute PD/MT distances')
    print('=' * 64)


if __name__ == '__main__':
    main()
