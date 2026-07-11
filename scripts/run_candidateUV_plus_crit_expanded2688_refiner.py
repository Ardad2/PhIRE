#!/usr/bin/env python3
"""
Candidate UV+crit-expanded-2688: scale-up of the completed
candidateUV_plus_crit_expanded672 run to 2688 training samples, using the
same loss configuration -- Candidate UV's bare vector-reconstruction loss
(L_uv) plus Candidate C's high-speed local-maxima / critical-value proxy
(L_crit), with Candidate B's scalar-speed, gradient, and level-set losses
kept disabled, exactly as in the completed 672 run. Tests whether L_crit
alone -- without Candidate B's scalar-field scaffold -- can produce
topology-relevant training signal at a larger training scale.

This is NOT a repaired-E2 script: it contains no TTK fixed-index critical-
pair supervision (no constraints NPZ, no per-sample feed_dict of birth/death
vertices). LAMBDA_TTKCV and LAMBDA_TTKPERS are defined below purely for
documentation/verification symmetry with the E2 family and are always 0.0;
no TTK loss ops are built in this script at all.

Same native PhIRE/TensorFlow generator fine-tuning path (calls
PhIREGANs.pretrain() directly, exactly like candidateUV_plus_crit_expanded672
and Candidate C-expanded-2688 do -- no custom training loop is needed here
since there is no extra per-sample constraint data to feed), same 2688-sample
expanded seasonal training data as Candidate C-expanded-2688, same pretrained
CNN initialization, same optimizer/epochs/batch size, same output/inference
shape conventions as candidateUV_plus_crit_expanded672 -- the only intended
difference from the 672 script is the training data/sample count.

Loss configuration (Candidate UV + Candidate C's critical-value proxy only):
    L_total = L_uv                              (normalized [u, v])
            + 0.001  * L_crit                    (physical; Candidate C proxy)

    L_speed, L_grad, and L_levelset remain ALL EXPLICITLY DISABLED
    (LAMBDA_SPEED = LAMBDA_GRAD = LAMBDA_LEVELSET = 0.0), identical to
    candidateUV_plus_crit_expanded672. Candidate B is NOT reintroduced at
    this scale.

L_uv/L_speed/L_grad/L_levelset/L_crit are built by sr_network.SR_NETWORK /
PhysicsLossConfig via PhIREGANs.pretrain(), completely unmodified -- this
script does not edit sr_network.py or PhIREGANs.py. The graph still
constructs the L_speed/L_grad/L_levelset ops (all with zero weight, i.e.
they always contribute exactly 0.0 to g_loss) so the diagnostic printout
(PhysicsLossConfig's diagnostic_mode=True) can display their raw
(unweighted) values for comparison purposes -- they are excluded from
training only via their zero weights, not by removing the ops. L_crit is
computed on denormalized scalar speed using the exact same Candidate C
implementation/convention (adaptive per-sample threshold mean + crit_high_z
* std, 3x3 local-maxima pooling with crit_pool=3).

Scientific readout (see docs/dataset_generation_and_repair_notes.md,
"Next robustness check: scale UV+E2-low and UV+crit to 1344 and 2688
samples"): the completed candidateUV_plus_crit_expanded672 run showed that
L_crit alone was NOT sufficient to reproduce Candidate C's PD improvement
(PD mean 29.4764, worse than the CNN baseline's 27.4063) and did not beat
CNN on mean MT either (5.9217 vs 5.8678), despite strong direct-fidelity and
scalar-speed gains. This 2688-sample run tests whether that conclusion holds
at a larger training scale:
  - If UV+crit remains topology-weak at 2688 samples, Candidate C's PD
    success should be attributed to the combination of Candidate B and
    L_crit, not L_crit alone.
  - If UV+crit improves substantially with scale, L_crit may be data-hungry
    and should be reconsidered as an independent local-extrema signal.
  - If UV+crit improves MT but not PD at this scale, L_crit is likely a
    local-extrema value proxy rather than a merge-tree hierarchy proxy,
    consistent with the 672-sample result.

Workflow:
  Phase 1 -- Training: fine-tune the pretrained vector [u,v] CNN for 3
             epochs on the 2688-sample expanded dataset.
             Output: models_fixed/topology_finetuning/
                       wind_finetune_candidateUV_plus_crit_expanded2688/
  Phase 2 -- Evaluation: paired inference on the 168-sample benchmark via
             PhIREGANs.test_paired(), unchanged from Candidate C.
             Output: data_out/wind_finetune_candidateUV_plus_crit_expanded2688/
                       dataSR.npy (168, 500, 500, 2)
                       dataGT.npy (168, 500, 500, 2)
                       dataIN.npy (168, 100, 100, 2)
                       idx.npy    (168,)

Usage (from repo root, on Spark):
  python3 scripts/run_candidateUV_plus_crit_expanded2688_refiner.py

Next steps (not run by this script):
  Cheap scalar/domain evaluation:
    python3 scripts/evaluate_finetune_candidate.py \\
      --candidate-name candidateUV_plus_crit_expanded2688 \\
      --candidate-dir  data_out/wind_finetune_candidateUV_plus_crit_expanded2688 \\
      --cnn-dir        data_out_fixed/wind_mrhr_cnn \\
      --gan-dir        data_out_fixed/wind_mrhr_gan \\
      --merged-csv     ttk_runs_fixed/combined/psnr_topology_physics_merged.csv \\
      --out-dir        ttk_runs_fixed/topology_finetuning/candidateUV_plus_crit_expanded2688_eval

  Only after confirming the cheap evaluation is non-catastrophic, the TTK
  topology pipeline (--threads 1 to avoid the segfault-under-contention
  issue seen on other candidates; --skip-viz since only the distances/
  comparison report are needed for this ablation):
    bash scripts/run_candidate_topology_pipeline.sh \\
      --method     candidateUV_plus_crit_expanded2688 \\
      --data-dir   data_out/wind_finetune_candidateUV_plus_crit_expanded2688 \\
      --vti-dir    ttk_runs_fixed/topology_finetuning/candidateUV_plus_crit_expanded2688_topology_vti \\
      --out-base   ttk_runs_fixed/topology_finetuning/candidateUV_plus_crit_expanded2688_topology \\
      --n-samples  168 \\
      --threads    1 \\
      --skip-viz
"""

from __future__ import annotations

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
# Tee: mirror stdout to a log file
# ---------------------------------------------------------------------------
LOG_PATH = REPO_ROOT / 'logs' / 'wind_finetune_candidateUV_plus_crit_expanded2688.log'


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

# Training data: the exact same expanded 2688-sample seasonal dataset used by
# Candidate C-expanded-2688 -- reused unchanged.
TRAIN_DATA_PATH = 'example_data_topology_expanded_2688/wind_MR-HR.tfrecord'

# Inference/evaluation data: original corrected 168-sample benchmark
EVAL_DATA_PATH = 'example_data_fixed/wind_MR-HR.tfrecord'

# Pretrained source checkpoint (read-only) -- same as Candidate C
MODEL_PATH = 'models/wind_mr-hr/trained_cnn/cnn'

# Outputs -- new, distinct names; do not overwrite any existing candidate
MODEL_OUT = 'models_fixed/topology_finetuning/wind_finetune_candidateUV_plus_crit_expanded2688'
DATA_OUT  = 'data_out/wind_finetune_candidateUV_plus_crit_expanded2688'

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
    # Expanded-672 candidates UV/B/C/D/Dpd (the direct apples-to-apples
    # comparison partners)
    'data_out/wind_finetune_candidateUV_expanded672',
    'data_out/wind_finetune_candidateB_expanded672',
    'data_out/wind_finetune_candidateC_expanded672',
    'data_out/wind_finetune_candidateD_expanded672',
    'data_out/wind_finetune_candidateDpd_expanded672',
    'models_fixed/topology_finetuning/wind_finetune_candidateUV_expanded672',
    'models_fixed/topology_finetuning/wind_finetune_candidateB_expanded672',
    'models_fixed/topology_finetuning/wind_finetune_candidateC_expanded672',
    'models_fixed/topology_finetuning/wind_finetune_candidateD_expanded672',
    'models_fixed/topology_finetuning/wind_finetune_candidateDpd_expanded672',
    # Candidate C at the larger completed scales (the main submitted result)
    # -- must not be overwritten.
    'data_out/wind_finetune_candidateC_expanded1344',
    'data_out/wind_finetune_candidateC_expanded2688',
    'models_fixed/topology_finetuning/wind_finetune_candidateC_expanded1344',
    'models_fixed/topology_finetuning/wind_finetune_candidateC_expanded2688',
    # Existing PyTorch residual-refiner E2 family -- distinct architecture,
    # must not collide with this TF fine-tuning family
    'data_out/wind_finetune_candidateE2_expanded672',
    'data_out/wind_finetune_candidateE2_fixed',
    'data_out/wind_finetune_candidateE2_fixed_lowlambda',
    'data_out/wind_finetune_candidateE2_fixed_lowlambda_expanded1344',
    'data_out/wind_finetune_candidateE2_fixed_lowlambda_expanded2688',
    'models_fixed/topology_finetuning/wind_finetune_candidateE2_expanded672',
    'models_fixed/topology_finetuning/wind_finetune_candidateE2_fixed',
    'models_fixed/topology_finetuning/wind_finetune_candidateE2_fixed_lowlambda',
    'models_fixed/topology_finetuning/wind_finetune_candidateE2_fixed_lowlambda_expanded1344',
    'models_fixed/topology_finetuning/wind_finetune_candidateE2_fixed_lowlambda_expanded2688',
    # Completed native TF B+E2-low runs at all three scales -- distinct
    # experiment (Candidate B stack + repaired TTK terms, no L_crit), must
    # not be overwritten.
    'data_out/wind_finetune_candidateB_plus_E2_tf_lowlambda_expanded672',
    'models_fixed/topology_finetuning/wind_finetune_candidateB_plus_E2_tf_lowlambda_expanded672',
    'data_out/wind_finetune_candidateB_plus_E2_tf_lowlambda_expanded1344',
    'models_fixed/topology_finetuning/wind_finetune_candidateB_plus_E2_tf_lowlambda_expanded1344',
    'data_out/wind_finetune_candidateB_plus_E2_tf_lowlambda_expanded2688',
    'models_fixed/topology_finetuning/wind_finetune_candidateB_plus_E2_tf_lowlambda_expanded2688',
    # Completed native TF C+E2-low runs at all three scales -- distinct
    # experiment (Candidate C stack + repaired TTK terms), must not be
    # overwritten.
    'data_out/wind_finetune_candidateE2_tf_lowlambda_expanded672',
    'models_fixed/topology_finetuning/wind_finetune_candidateE2_tf_lowlambda_expanded672',
    'data_out/wind_finetune_candidateE2_tf_lowlambda_expanded1344',
    'models_fixed/topology_finetuning/wind_finetune_candidateE2_tf_lowlambda_expanded1344',
    'data_out/wind_finetune_candidateE2_tf_lowlambda_expanded2688',
    'models_fixed/topology_finetuning/wind_finetune_candidateE2_tf_lowlambda_expanded2688',
    # Completed native TF UV+E2-low-672 run -- closest cross-family sibling
    # ablation (bare L_uv + repaired TTK terms, no L_crit), must not be
    # overwritten.
    'data_out/wind_finetune_candidateUV_plus_E2_tf_lowlambda_expanded672',
    'models_fixed/topology_finetuning/wind_finetune_candidateUV_plus_E2_tf_lowlambda_expanded672',
    # The completed native TF UV+crit-672 run this scale-up ladder traces
    # back to -- must not be overwritten by this or any sibling script.
    'data_out/wind_finetune_candidateUV_plus_crit_expanded672',
    'models_fixed/topology_finetuning/wind_finetune_candidateUV_plus_crit_expanded672',
    # The sibling 1344-scale UV+crit script's output -- defense in
    # depth against a copy-paste mistake between the two scale-up scripts.
    'data_out/wind_finetune_candidateUV_plus_crit_expanded1344',
    'models_fixed/topology_finetuning/wind_finetune_candidateUV_plus_crit_expanded1344',
]

# ---------------------------------------------------------------------------
# Normalisation constants (pretrained checkpoint; unchanged)
# ---------------------------------------------------------------------------
MU_SIG = [[0.7684, -0.4575], [5.02455, 5.9017]]
R      = [5]

# ---------------------------------------------------------------------------
# UV+crit loss weights: bare Candidate UV vector loss (L_uv only) plus
# Candidate C's critical-value proxy (L_crit) -- Candidate B's scalar-speed/
# gradient/level-set scaffold is explicitly zeroed, and repaired E2 TTK
# fixed-index losses are not implemented in this script at all (LAMBDA_TTKCV/
# LAMBDA_TTKPERS below are documentation-only and always 0.0).
# ---------------------------------------------------------------------------
LAMBDA_SPEED    = 0.0    # DISABLED (was 0.01 in Candidate B/C)
LAMBDA_GRAD     = 0.0    # DISABLED (was 0.05 in Candidate B/C)
LAMBDA_WPD      = 0.0
LAMBDA_LEVELSET = 0.0    # DISABLED (was 0.25 in Candidate B/C)
LAMBDA_CRIT     = 0.001  # ENABLED -- Candidate C's critical-value proxy, unchanged
CRIT_HIGH_Z     = 1.0    # adaptive threshold: mean + 1 sigma (Candidate C convention)
CRIT_POOL       = 3      # 3x3 local-maxima pooling (Candidate C convention)

# Repaired E2 TTK fixed-index losses are NOT used in this script (no
# constraints NPZ, no custom training loop, no per-sample feed_dict). These
# constants exist only so this ablation's lambda configuration is fully
# documented and machine-verifiable alongside the E2 family scripts.
LAMBDA_TTKCV   = 0.0
LAMBDA_TTKPERS = 0.0

# ---------------------------------------------------------------------------
# Training hyperparameters (identical to Candidate C-expanded-2688)
# ---------------------------------------------------------------------------
LEARNING_RATE = 1e-5
N_EPOCHS      = 3
SAVE_EVERY    = 1
PRINT_EVERY   = 2
BATCH_SIZE    = 4

# ---------------------------------------------------------------------------
# Dataset size parameters
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


# ---------------------------------------------------------------------------
# Preflight checks
# ---------------------------------------------------------------------------

def _run_preflight_checks() -> None:
    print('=' * 64)
    print('Preflight checks')
    print('=' * 64)

    if not Path(TRAIN_DATA_PATH).exists():
        sys.exit(
            f'[error] Training TFRecord not found: {TRAIN_DATA_PATH}\n'
            '  Run scripts/build_wind_mrhr_expanded_dataset.py first:\n'
            '    python3 scripts/build_wind_mrhr_expanded_dataset.py \\\n'
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
        sys.exit(
            f'[error] Training TFRecord has {n_records} records; expected {N_TRAIN}.'
        )

    print()
    print('  Lambda values:')
    print(f'    LAMBDA_SPEED    = {LAMBDA_SPEED}  (DISABLED)')
    print(f'    LAMBDA_GRAD     = {LAMBDA_GRAD}  (DISABLED)')
    print(f'    LAMBDA_WPD      = {LAMBDA_WPD}')
    print(f'    LAMBDA_LEVELSET = {LAMBDA_LEVELSET}  (DISABLED)')
    print(f'    LAMBDA_CRIT     = {LAMBDA_CRIT}  (ENABLED -- Candidate C critical-value proxy)')
    print('  UV+crit: L_speed/L_grad/L_levelset disabled; only L_uv + L_crit active.')
    print(f'    LAMBDA_TTKCV    = {LAMBDA_TTKCV}  (repaired E2 not used in this script)')
    print(f'    LAMBDA_TTKPERS  = {LAMBDA_TTKPERS}  (repaired E2 not used in this script)')
    print()
    print('  Output directories:')
    print(f'    MODEL_OUT = {MODEL_OUT}')
    print(f'    DATA_OUT  = {DATA_OUT}')
    print(f'    LOG_PATH  = {LOG_PATH}')
    print('=' * 64)
    print()


# ---------------------------------------------------------------------------
# Output validation
# ---------------------------------------------------------------------------

def _validate_outputs() -> None:
    print('\n[validate] Checking output shapes …')
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
            print(f'  [FAIL] {fname}: NOT FOUND')
            all_ok = False
            continue
        arr = np.load(p, mmap_mode='r')
        if arr.shape != exp_shape:
            print(f'  [FAIL] {fname}: shape {arr.shape} != {exp_shape}')
            all_ok = False
        else:
            print(f'  [OK]   {fname}: {arr.shape}')
        if fname != 'idx.npy':
            sample = np.load(p)
            if not np.isfinite(sample).all():
                print(f'  [FAIL] {fname}: contains NaN/inf values')
                all_ok = False

    idx_p = Path(DATA_OUT) / 'idx.npy'
    if idx_p.exists():
        idx_vals = np.load(idx_p)
        if not np.array_equal(np.sort(idx_vals), np.arange(N_EVAL)):
            print(
                f'  [FAIL] idx.npy: values are not exactly 0..{N_EVAL - 1} '
                f'(got range [{int(idx_vals.min())}, {int(idx_vals.max())}], '
                f'{len(idx_vals)} entries)'
            )
            all_ok = False
        else:
            print(f'  [OK]   idx.npy values: exactly 0..{N_EVAL - 1}')

    if not all_ok:
        sys.exit('[error] Output shape validation failed.')
    print('[validate] All outputs OK.')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _safety_check()

    print('=' * 64)
    print('Candidate UV+crit-expanded-2688 fine-tuning (Candidate B scaffold ablation)')
    print('=' * 64)
    print(f'  Training data     : {TRAIN_DATA_PATH}')
    print(f'  Evaluation data   : {EVAL_DATA_PATH}')
    print(f'  Source checkpoint : {MODEL_PATH}')
    print(f'  Output model dir  : {MODEL_OUT}')
    print(f'  Inference output  : {DATA_OUT}')
    print(f'  Log file          : {LOG_PATH}')
    print('=' * 64)
    print()

    # ── Preflight ────────────────────────────────────────────────────────
    _run_preflight_checks()

    # ── Phase 1: Fine-tuning on the 2688-sample expanded dataset ───────────
    # diagnostic_mode=True logs per-iteration loss breakdown, including the
    # raw (unweighted) values of the disabled L_speed/L_grad/L_levelset ops.
    cfg = PhysicsLossConfig(
        use_aux_losses       = True,
        mu                   = MU_SIG[0],
        sig                  = MU_SIG[1],
        lambda_speed         = LAMBDA_SPEED,
        lambda_grad          = LAMBDA_GRAD,
        lambda_wpd           = LAMBDA_WPD,
        lambda_levelset      = LAMBDA_LEVELSET,
        levelset_temperature = 10.0,
        levelset_thresholds  = [5.0, 10.0, 15.0],
        lambda_crit          = LAMBDA_CRIT,
        crit_high_z          = CRIT_HIGH_Z,
        crit_include_minima  = False,
        crit_low_z           = -1.0,
        crit_pool            = CRIT_POOL,
        diagnostic_mode      = True,
    )

    phire = PhIREGANs(
        data_type    = 'wind_finetune_candidateUV_plus_crit_expanded2688',
        learning_rate= LEARNING_RATE,
        N_epochs     = N_EPOCHS,
        save_every   = SAVE_EVERY,
        print_every  = PRINT_EVERY,
        mu_sig       = MU_SIG,
    )
    phire.setModel_name(MODEL_OUT)

    print('Phase 1: Fine-tuning on 2688-sample expanded dataset …')
    saved_model = phire.pretrain(
        r          = R,
        data_path  = TRAIN_DATA_PATH,
        model_path = MODEL_PATH,
        batch_size = BATCH_SIZE,
        phys_cfg   = cfg,
    )

    print()
    print(f'Fine-tuning complete. Final checkpoint: {saved_model}')
    print()

    # ── Phase 2: Paired inference on the 168-sample benchmark ─────────────
    print('=' * 64)
    print('Phase 2: Paired inference on 168-sample benchmark')
    print('=' * 64)
    print(f'  Checkpoint : {saved_model}')
    print(f'  Eval data  : {EVAL_DATA_PATH}')
    print(f'  Output dir : {DATA_OUT}')
    print()

    phire.set_data_out_path(DATA_OUT)
    phire.test_paired(
        r          = R,
        data_path  = EVAL_DATA_PATH,
        model_path = saved_model,
        batch_size = 1,
        save_inputs= True,
    )

    # ── Validate outputs ────────────────────────────────────────────────
    _validate_outputs()

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
    print('  Cheap scalar/domain evaluation (run this first):')
    print('    python3 scripts/evaluate_finetune_candidate.py \\')
    print('      --candidate-name candidateUV_plus_crit_expanded2688 \\')
    print('      --candidate-dir  data_out/wind_finetune_candidateUV_plus_crit_expanded2688 \\')
    print('      --cnn-dir        data_out_fixed/wind_mrhr_cnn \\')
    print('      --gan-dir        data_out_fixed/wind_mrhr_gan \\')
    print('      --merged-csv     ttk_runs_fixed/combined/psnr_topology_physics_merged.csv \\')
    print('      --out-dir        ttk_runs_fixed/topology_finetuning/candidateUV_plus_crit_expanded2688_eval')
    print()
    print('  Only run the expensive TTK topology pipeline AFTER confirming')
    print('  the cheap evaluation above is non-catastrophic:')
    print('    bash scripts/run_candidate_topology_pipeline.sh \\')
    print('      --method     candidateUV_plus_crit_expanded2688 \\')
    print('      --data-dir   data_out/wind_finetune_candidateUV_plus_crit_expanded2688 \\')
    print('      --vti-dir    ttk_runs_fixed/topology_finetuning/candidateUV_plus_crit_expanded2688_topology_vti \\')
    print('      --out-base   ttk_runs_fixed/topology_finetuning/candidateUV_plus_crit_expanded2688_topology \\')
    print('      --n-samples  168 \\')
    print('      --threads    1 \\')
    print('      --skip-viz')
    print('=' * 64)


if __name__ == '__main__':
    main()
