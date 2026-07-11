#!/usr/bin/env python3
"""
Candidate UV+E2-tf-lowlambda-expanded-1344: scale-up of the completed
candidateUV_plus_E2_tf_lowlambda_expanded672 run to 1344 training samples,
using the same loss configuration -- Candidate UV's bare vector-
reconstruction loss (L_uv) plus repaired low-lambda TTK fixed-index losses
(L_ttkcv, L_ttkpers), with Candidate B's scalar-speed/gradient/level-set
scaffold and Candidate C's L_crit all kept disabled, exactly as in the
completed 672 run. This is an apples-to-apples scale comparison against
Candidate C-expanded-1344, candidateB_plus_E2_tf_lowlambda_expanded1344, and
candidateE2_tf_lowlambda_expanded1344, following the same 672->1344->2688
training-scale ladder already used for those families.

Background (unchanged from the 672 script): this tests whether Candidate
B's scalar-field scaffold is necessary for the repaired TTK fixed-index
losses to produce a useful training signal on their own. At 672 samples,
candidateUV_plus_E2_tf_lowlambda_expanded672 gave the strongest 672-scale
mean MT among the repaired native TF E2 variants (PD 25.2675, MT 5.6645) --
this scale-up tests whether that MT signal persists at 1344 samples. Same
PhIRE/TensorFlow generator fine-tuning path (sr_network.SR_NETWORK), same
pretrained CNN initialization, same optimizer/epochs/batch size, same
repaired-E2 constraint convention (rebuilt at this training scale), same
normalized L_uv convention, and same output/inference shape conventions as
candidateUV_plus_E2_tf_lowlambda_expanded672 -- the only intended
differences from the 672 script are the training data/sample count and the
constraints file.

This is "vector-only reconstruction + repaired TTK fixed-index losses": same
architecture (fine-tunes the actual generator, not a frozen-CNN post-hoc
residual head) as Candidate C-expanded-1344 and candidateUV_plus_E2_tf_
lowlambda_expanded672, with TTK fixed-index losses (L_ttkcv, L_ttkpers)
added directly on top of Candidate UV's bare vector-reconstruction loss --
no Candidate B scalar-speed/gradient/level-set scaffold, and no Candidate
C's L_crit.

Loss configuration (Candidate UV + repaired low-lambda TTK terms only):
    L_total = L_uv                              (normalized [u, v])
            + 0.004  * L_ttkcv                   (physical; TTK critical vertices)
            + 0.002  * L_ttkpers                 (physical; TTK persistence gap)

    L_speed, L_grad, L_levelset, and L_crit remain ALL EXPLICITLY DISABLED
    (LAMBDA_SPEED = LAMBDA_GRAD = LAMBDA_LEVELSET = LAMBDA_CRIT = 0.0),
    identical to candidateUV_plus_E2_tf_lowlambda_expanded672. Candidate B
    and Candidate C are NOT reintroduced at this scale.

L_uv/L_speed/L_grad/L_levelset/L_crit are built by sr_network.SR_NETWORK /
PhysicsLossConfig, completely unmodified -- this script does not edit
sr_network.py or PhIREGANs.py. The graph still constructs the L_speed/
L_grad/L_levelset/L_crit ops (all with zero weight, i.e. they always
contribute exactly 0.0 to g_loss) so the diagnostic printout can display
their raw (unweighted) values for comparison purposes -- they are excluded
from training only via their zero weights, not by removing the ops.
L_ttkcv/L_ttkpers are new TF ops added in this script only, operating on
model.x_SR (normalized predicted [u,v]) via the same _denorm_speed() helper
Candidate C's losses already use internally.

TTK constraints (built at this training scale, distinct from the 672
script's):
    ttk_runs_fixed/topology_finetuning/candidateE2_fixed_lowlambda_expanded1344_constraints/
      ttk_pd_critical_pairs_gtvalues.npz

Design decision (see docstring section "Feeding constraints into TF1" below
for the reasoning): constraints are fed via feed_dict, keyed by the
TFRecord-embedded 'index' feature that PhIREGANs._parse_train_ already
decodes and PhIREGANs.pretrain() already fetches every training step but
discards. No changes to PhIREGANs.py's data pipeline were needed or made.

Feeding constraints into TF1
-----------------------------
PhIREGANs.pretrain()'s tf.data pipeline already yields (idx, LR, HR) per
example, where idx = example['index'] is a TFRecord-embedded per-example
integer id (NOT a read-order position) -- stable under .shuffle(). This
script cannot call phire.pretrain() as a black box (no hook for extra
placeholders/losses), so it builds its own training loop that mirrors
pretrain()'s structure closely, reusing sr_network.SR_NETWORK and
PhIREGANs' set_mu_sig/set_LR_data_shape/_parse_train_ as building blocks.
Each training step fetches (batch_idx, batch_LR, batch_HR) exactly as
pretrain() does, then looks up the TTK constraint rows for each sample in
the batch (by batch_idx) from a constraints NPZ preloaded into a Python
dict, and feeds them into new TTK placeholders in the same feed_dict.

Fixed-shape batching and the valid-pairs mask
-----------------------------------------------
The constraints NPZ stores up to MAX_PAIRS=64 birth/death pairs per sample
(persistence_frac=0.01, top_k=64 filtering at constraint-build time). A TF
feed_dict batch needs uniform array shape across samples, so every sample's
pairs are padded to exactly MAX_PAIRS with an explicit per-pair valid mask
(1 for a real pair, 0 for padding) rather than assuming every sample has
exactly 64 real pairs. If every training sample happens to have exactly 64
(the observed case for this constraint file), the mask is all-ones and this
reduces to the literal "no missing pairs" case. Preflight aborts if any
sample has MORE than 64 pairs (would silently truncate data); anything up
to 64 is handled correctly via the mask.

Vertex-coordinate convention (matches build_candidateE2_expanded672_ttk_
constraints.py and TTKConstraints.get() in the PyTorch scripts exactly):
    iy = vid // PATCH   (PATCH = 160)
    ix = vid % PATCH
These (iy, ix) values are used to index DIRECTLY into the full 500x500
predicted speed field with no offset -- correct only because the TTK VTI
crop is top-left-anchored (x0=y0=0), so crop-local and full-field indices
coincide for iy, ix < 160. If the crop origin is ever changed, this direct
indexing would need an explicit (y0, x0) offset added.

Scientific readout (see docs/dataset_generation_and_repair_notes.md,
"Next robustness check: scale UV+E2-low and UV+crit to 1344 and 2688
samples"): the completed candidateUV_plus_E2_tf_lowlambda_expanded672 run
gave the strongest 672-scale mean MT among the repaired native TF E2
variants (PD 25.2675, MT 5.6645), without Candidate B's scalar-speed/
gradient/level-set scaffold. This 1344-sample run tests whether that
conclusion holds at a larger training scale:
  - If UV+E2-low remains MT-strong at 1344 samples, repaired fixed-index TTK
    supervision is robustly MT-oriented even without Candidate B, across
    scale.
  - If UV+E2-low weakens at 1344 samples relative to the 672 result, the
    672-scale MT signal may have been partly a small-data artifact.
  - If UV+E2-low improves MT but damages fidelity at this scale, Candidate B
    likely acts as a stabilizer rather than as the main topology signal.

Workflow:
  Phase 1 -- Training: fine-tune the pretrained vector [u,v] CNN for 3
             epochs on the 1344-sample expanded dataset.
             Output: models_fixed/topology_finetuning/
                       wind_finetune_candidateUV_plus_E2_tf_lowlambda_expanded1344/
  Phase 2 -- Evaluation: paired inference on the 168-sample benchmark via
             PhIREGANs.test_paired(), unchanged from Candidate C.
             Output: data_out/wind_finetune_candidateUV_plus_E2_tf_lowlambda_expanded1344/
                       dataSR.npy (168, 500, 500, 2)
                       dataGT.npy (168, 500, 500, 2)
                       dataIN.npy (168, 100, 100, 2)
                       idx.npy    (168,)

Usage (from repo root, on Spark):
  python3 scripts/run_candidateUV_plus_E2_tf_lowlambda_expanded1344_ttkcrit_refiner.py

Next steps (not run by this script):
  Cheap scalar/domain evaluation:
    python3 scripts/evaluate_finetune_candidate.py \\
      --candidate-name candidateUV_plus_E2_tf_lowlambda_expanded1344 \\
      --candidate-dir  data_out/wind_finetune_candidateUV_plus_E2_tf_lowlambda_expanded1344 \\
      --cnn-dir        data_out_fixed/wind_mrhr_cnn \\
      --gan-dir        data_out_fixed/wind_mrhr_gan \\
      --merged-csv     ttk_runs_fixed/combined/psnr_topology_physics_merged.csv \\
      --out-dir        ttk_runs_fixed/topology_finetuning/candidateUV_plus_E2_tf_lowlambda_expanded1344_eval

  Only after confirming the cheap evaluation is non-catastrophic, the TTK
  topology pipeline (--threads 1 to avoid the segfault-under-contention
  issue seen on candidateE2_tf_lowlambda_expanded672; --skip-viz since only
  the distances/comparison report are needed for this ablation):
    bash scripts/run_candidate_topology_pipeline.sh \\
      --method     candidateUV_plus_E2_tf_lowlambda_expanded1344 \\
      --data-dir   data_out/wind_finetune_candidateUV_plus_E2_tf_lowlambda_expanded1344 \\
      --vti-dir    ttk_runs_fixed/topology_finetuning/candidateUV_plus_E2_tf_lowlambda_expanded1344_topology_vti \\
      --out-base   ttk_runs_fixed/topology_finetuning/candidateUV_plus_E2_tf_lowlambda_expanded1344_topology \\
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
LOG_PATH = REPO_ROOT / 'logs' / 'wind_finetune_candidateUV_plus_E2_tf_lowlambda_expanded1344.log'


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

from sr_network import PhysicsLossConfig, SR_NETWORK, _denorm_speed
from PhIREGANs import PhIREGANs

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Training data: same expanded 1344-sample seasonal dataset as Candidate C-expanded-1344
# and candidateB_plus_E2_tf_lowlambda_expanded1344 -- reused unchanged.
TRAIN_DATA_PATH = 'example_data_topology_expanded_1344/wind_MR-HR.tfrecord'

# Inference/evaluation data: original corrected 168-sample benchmark
EVAL_DATA_PATH = 'example_data_fixed/wind_MR-HR.tfrecord'

# Pretrained source checkpoint (read-only) -- same as Candidate C
MODEL_PATH = 'models/wind_mr-hr/trained_cnn/cnn'

# Repaired E2 TTK constraints (built with the fixed VTI coordinate mapping) --
# the exact same file used by candidateB_plus_E2_tf_lowlambda_expanded1344 and
# candidateE2_tf_lowlambda_expanded1344 (same 1344-sample training scale).
CONSTRAINTS_NPZ = (
    REPO_ROOT / 'ttk_runs_fixed' / 'topology_finetuning'
    / 'candidateE2_fixed_lowlambda_expanded1344_constraints' / 'ttk_pd_critical_pairs_gtvalues.npz'
)

# Outputs -- new, distinct names; do not overwrite any existing candidate
MODEL_OUT = 'models_fixed/topology_finetuning/wind_finetune_candidateUV_plus_E2_tf_lowlambda_expanded1344'
DATA_OUT  = 'data_out/wind_finetune_candidateUV_plus_E2_tf_lowlambda_expanded1344'

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
    # Expanded-672 candidates (the direct apples-to-apples comparison partners)
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
    # Candidate C at all completed scales (the main submitted result) --
    # must not be overwritten.
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
    # The completed native TF C+E2-low runs at all three scales -- distinct
    # experiment (Candidate C stack + repaired TTK terms), must not be
    # overwritten.
    'data_out/wind_finetune_candidateE2_tf_lowlambda_expanded672',
    'models_fixed/topology_finetuning/wind_finetune_candidateE2_tf_lowlambda_expanded672',
    'data_out/wind_finetune_candidateE2_tf_lowlambda_expanded1344',
    'models_fixed/topology_finetuning/wind_finetune_candidateE2_tf_lowlambda_expanded1344',
    'data_out/wind_finetune_candidateE2_tf_lowlambda_expanded2688',
    'models_fixed/topology_finetuning/wind_finetune_candidateE2_tf_lowlambda_expanded2688',
    # The completed native TF B+E2-low runs at all three scales -- this
    # script's closest sibling ablation (Candidate B stack + repaired TTK
    # terms, no L_crit), must not be overwritten.
    'data_out/wind_finetune_candidateB_plus_E2_tf_lowlambda_expanded672',
    'models_fixed/topology_finetuning/wind_finetune_candidateB_plus_E2_tf_lowlambda_expanded672',
    'data_out/wind_finetune_candidateB_plus_E2_tf_lowlambda_expanded1344',
    'models_fixed/topology_finetuning/wind_finetune_candidateB_plus_E2_tf_lowlambda_expanded1344',
    'data_out/wind_finetune_candidateB_plus_E2_tf_lowlambda_expanded2688',
    'models_fixed/topology_finetuning/wind_finetune_candidateB_plus_E2_tf_lowlambda_expanded2688',
    # The completed native TF UV+E2-low-672 run this scale-up ladder traces
    # back to -- must not be overwritten by this or any sibling script.
    'data_out/wind_finetune_candidateUV_plus_E2_tf_lowlambda_expanded672',
    'models_fixed/topology_finetuning/wind_finetune_candidateUV_plus_E2_tf_lowlambda_expanded672',
    # The completed native TF UV+crit-672 run -- cross-family sibling
    # ablation (Candidate C's L_crit without Candidate B, no repaired E2),
    # must not be overwritten.
    'data_out/wind_finetune_candidateUV_plus_crit_expanded672',
    'models_fixed/topology_finetuning/wind_finetune_candidateUV_plus_crit_expanded672',
    # The sibling 2688-scale UV+E2 script's output -- defense in depth
    # against a copy-paste mistake between the two scale-up scripts.
    'data_out/wind_finetune_candidateUV_plus_E2_tf_lowlambda_expanded2688',
    'models_fixed/topology_finetuning/wind_finetune_candidateUV_plus_E2_tf_lowlambda_expanded2688',
    # Constraints (read-only input to this script; never written)
    'ttk_runs_fixed/topology_finetuning/candidateE2_fixed_constraints',
    'ttk_runs_fixed/topology_finetuning/candidateE2_expanded672_constraints',
    'ttk_runs_fixed/topology_finetuning/candidateE2_fixed_lowlambda_expanded1344_constraints',
    'ttk_runs_fixed/topology_finetuning/candidateE2_fixed_lowlambda_expanded2688_constraints',
]

# ---------------------------------------------------------------------------
# Normalisation constants (pretrained checkpoint; unchanged)
# ---------------------------------------------------------------------------
MU_SIG = [[0.7684, -0.4575], [5.02455, 5.9017]]
R      = [5]

# ---------------------------------------------------------------------------
# UV+E2 loss weights: bare Candidate UV vector loss (L_uv only) plus repaired
# low-lambda TTK terms -- ALL Candidate B/C scalar proxy terms (L_speed,
# L_grad, L_levelset, L_crit) are explicitly zeroed. This isolates whether
# the repaired TTK fixed-index losses can drive topology on their own,
# without Candidate B's scalar-field scaffold or Candidate C's L_crit.
# ---------------------------------------------------------------------------
LAMBDA_SPEED    = 0.0    # DISABLED (was 0.01 in Candidate B/C and B+E2/C+E2)
LAMBDA_GRAD     = 0.0    # DISABLED (was 0.05 in Candidate B/C and B+E2/C+E2)
LAMBDA_WPD      = 0.0
LAMBDA_LEVELSET = 0.0    # DISABLED (was 0.25 in Candidate B/C and B+E2/C+E2)
LAMBDA_CRIT     = 0.0    # DISABLED (was 0.001 in Candidate C/C+E2; already 0.0 in B+E2)
CRIT_HIGH_Z     = 1.0    # inert while LAMBDA_CRIT=0.0; kept only so PhysicsLossConfig's
CRIT_POOL       = 3      # signature matches the B+E2/C+E2-tf-lowlambda scripts unchanged

# Repaired low-lambda TTK weights. Do NOT use the old high-lambda 0.04/0.02
# values as the default here -- see docs/dataset_generation_and_repair_notes.md
# Part XIII for why the high-lambda configuration degraded fidelity.
LAMBDA_TTKCV   = 0.004
LAMBDA_TTKPERS = 0.002

# ---------------------------------------------------------------------------
# Training hyperparameters (identical to Candidate C-expanded-1344 /
# candidateB_plus_E2_tf_lowlambda_expanded1344)
# ---------------------------------------------------------------------------
LEARNING_RATE = 1e-5
N_EPOCHS      = 3
SAVE_EVERY    = 1
PRINT_EVERY   = 2
BATCH_SIZE    = 4

# ---------------------------------------------------------------------------
# TTK constraint parameters
# ---------------------------------------------------------------------------
PATCH      = 160   # vertex-ID encoding width; matches the constraint builder
MAX_PAIRS  = 64    # padded per-sample pair count; see module docstring
N_TRAIN    = 1344
N_EVAL     = 168


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


# ---------------------------------------------------------------------------
# TTK constraints loading (numpy side)
# ---------------------------------------------------------------------------

def _read_tfrecord_indices(data_path: str) -> np.ndarray:
    """Read only the 'index' feature from every Example in a TFRecord, without
    decoding the (large) LR/HR payloads. Pure protobuf parsing, no TF graph."""
    indices = []
    for raw in tf.python_io.tf_record_iterator(data_path):
        example = tf.train.Example()
        example.ParseFromString(raw)
        indices.append(example.features.feature['index'].int64_list.value[0])
    return np.array(indices, dtype=np.int64)


def _load_constraints(npz_path: Path):
    """Load the constraints NPZ into a dict: sample_idx -> padded per-sample
    arrays (birth_yx, death_yx, birth_val, death_val, persistence, valid_mask),
    all padded/truncated to exactly MAX_PAIRS with valid_mask marking real
    pairs. Returns (constraints_dict, sample_count_array, sample_idx_array).
    """
    npz = np.load(str(npz_path), allow_pickle=True)

    required = {
        'n_samples', 'sample_idx', 'sample_start', 'sample_count',
        'birth_vid', 'death_vid', 'birth_val', 'death_val', 'persistence',
    }
    missing = required - set(npz.files)
    if missing:
        sys.exit(f'[error] Constraints NPZ missing required keys: {missing}')

    sample_idx   = npz['sample_idx'].astype(np.int64)
    sample_start = npz['sample_start'].astype(np.int64)
    sample_count = npz['sample_count'].astype(np.int64)
    birth_vid    = npz['birth_vid'].astype(np.int64)
    death_vid    = npz['death_vid'].astype(np.int64)
    birth_val    = npz['birth_val'].astype(np.float32)
    death_val    = npz['death_val'].astype(np.float32)
    persistence  = npz['persistence'].astype(np.float32)

    constraints = {}
    for row in range(len(sample_idx)):
        sidx  = int(sample_idx[row])
        start = int(sample_start[row])
        count = int(sample_count[row])
        n_use = min(count, MAX_PAIRS)

        birth_yx    = np.zeros((MAX_PAIRS, 2), dtype=np.int32)
        death_yx    = np.zeros((MAX_PAIRS, 2), dtype=np.int32)
        birth_v     = np.zeros((MAX_PAIRS,),   dtype=np.float32)
        death_v     = np.zeros((MAX_PAIRS,),   dtype=np.float32)
        pers_v      = np.zeros((MAX_PAIRS,),   dtype=np.float32)
        valid_mask  = np.zeros((MAX_PAIRS,),   dtype=np.float32)

        if n_use > 0:
            bvid = birth_vid[start:start + n_use]
            dvid = death_vid[start:start + n_use]
            birth_yx[:n_use, 0] = bvid // PATCH   # iy
            birth_yx[:n_use, 1] = bvid % PATCH    # ix
            death_yx[:n_use, 0] = dvid // PATCH
            death_yx[:n_use, 1] = dvid % PATCH
            birth_v[:n_use] = birth_val[start:start + n_use]
            death_v[:n_use] = death_val[start:start + n_use]
            pers_v[:n_use]  = persistence[start:start + n_use]
            valid_mask[:n_use] = 1.0

        constraints[sidx] = {
            'birth_yx': birth_yx, 'death_yx': death_yx,
            'birth_val': birth_v, 'death_val': death_v,
            'persistence': pers_v, 'valid_mask': valid_mask,
        }

    return constraints, sample_count, sample_idx


def _build_batch_ttk_feed(constraints: dict, batch_idx: np.ndarray):
    """Stack per-sample padded constraint arrays for one batch, keyed by the
    TFRecord-embedded sample indices in batch_idx."""
    B = len(batch_idx)
    birth_yx   = np.zeros((B, MAX_PAIRS, 2), dtype=np.int32)
    death_yx   = np.zeros((B, MAX_PAIRS, 2), dtype=np.int32)
    birth_val  = np.zeros((B, MAX_PAIRS),    dtype=np.float32)
    death_val  = np.zeros((B, MAX_PAIRS),    dtype=np.float32)
    persistence = np.zeros((B, MAX_PAIRS),   dtype=np.float32)
    valid_mask = np.zeros((B, MAX_PAIRS),    dtype=np.float32)

    for i, sidx in enumerate(batch_idx):
        sidx = int(sidx)
        if sidx not in constraints:
            # Preflight guarantees every training sample_idx is present in
            # constraints; this should be unreachable. Fail loudly rather
            # than silently training with zeroed/masked-out constraints.
            raise RuntimeError(
                f'[error] sample_idx={sidx} not found in constraints during '
                'training, despite passing preflight. Aborting.'
            )
        c = constraints[sidx]
        birth_yx[i]    = c['birth_yx']
        death_yx[i]    = c['death_yx']
        birth_val[i]   = c['birth_val']
        death_val[i]   = c['death_val']
        persistence[i] = c['persistence']
        valid_mask[i]  = c['valid_mask']

    return birth_yx, death_yx, birth_val, death_val, persistence, valid_mask


# ---------------------------------------------------------------------------
# TF-side TTK loss ops
# ---------------------------------------------------------------------------

def _gather_speed_at_yx(speed_field: tf.Tensor, yx: tf.Tensor) -> tf.Tensor:
    """speed_field: [B, H, W]; yx: [B, K, 2] (iy, ix) int32 -> [B, K]."""
    B = tf.shape(speed_field)[0]
    K = tf.shape(yx)[1]
    batch_idx = tf.reshape(tf.range(B), [B, 1, 1])
    batch_idx = tf.tile(batch_idx, [1, K, 1])
    gather_idx = tf.concat([batch_idx, yx], axis=2)   # [B, K, 3]
    return tf.gather_nd(speed_field, gather_idx)       # [B, K]


def _masked_mse(pred: tf.Tensor, target: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
    sq_err   = tf.square(pred - target) * mask
    n_valid  = tf.maximum(tf.reduce_sum(mask), 1.0)
    return tf.reduce_sum(sq_err) / n_valid


def build_ttk_loss_ops(model, mu_sig):
    """Build L_ttkcv/L_ttkpers ops and the placeholders that feed them.

    Returns (placeholders_dict, L_ttkcv, L_ttkpers).
    """
    mu_tf  = tf.constant(mu_sig[0], dtype=tf.float32)
    sig_tf = tf.constant(mu_sig[1], dtype=tf.float32)
    speed_sr = _denorm_speed(model.x_SR, mu_tf, sig_tf)   # [B, 500, 500]

    ph = {
        'birth_yx':    tf.placeholder(tf.int32,   [None, MAX_PAIRS, 2], name='ttk_birth_yx'),
        'death_yx':    tf.placeholder(tf.int32,   [None, MAX_PAIRS, 2], name='ttk_death_yx'),
        'birth_val':   tf.placeholder(tf.float32, [None, MAX_PAIRS],    name='ttk_birth_val'),
        'death_val':   tf.placeholder(tf.float32, [None, MAX_PAIRS],    name='ttk_death_val'),
        'persistence': tf.placeholder(tf.float32, [None, MAX_PAIRS],    name='ttk_persistence'),
        'valid_mask':  tf.placeholder(tf.float32, [None, MAX_PAIRS],    name='ttk_valid_mask'),
    }

    sr_birth = _gather_speed_at_yx(speed_sr, ph['birth_yx'])   # [B, MAX_PAIRS]
    sr_death = _gather_speed_at_yx(speed_sr, ph['death_yx'])

    L_ttkcv = 0.5 * (
        _masked_mse(sr_birth, ph['birth_val'], ph['valid_mask'])
        + _masked_mse(sr_death, ph['death_val'], ph['valid_mask'])
    )
    sr_persistence = tf.abs(sr_death - sr_birth)
    L_ttkpers = _masked_mse(sr_persistence, ph['persistence'], ph['valid_mask'])

    return ph, L_ttkcv, L_ttkpers


# ---------------------------------------------------------------------------
# Preflight checks
# ---------------------------------------------------------------------------

def _run_preflight_checks(phire: PhIREGANs):
    print('=' * 64)
    print('Preflight checks')
    print('=' * 64)

    if not Path(TRAIN_DATA_PATH).exists():
        sys.exit(
            f'[error] Training TFRecord not found: {TRAIN_DATA_PATH}\n'
            '  Run scripts/build_wind_mrhr_expanded_dataset.py first:\n'
            '    python3 scripts/build_wind_mrhr_expanded_dataset.py \\\n'
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

    if not CONSTRAINTS_NPZ.exists():
        sys.exit(
            f'[error] Constraints NPZ not found: {CONSTRAINTS_NPZ}\n'
            '  Build it first:\n'
            '    python3 scripts/build_candidateE2_expanded672_ttk_constraints.py \\\n'
            '      --gt-path    data_out/wind_mrhr_cnn_expanded1344/dataGT.npy \\\n'
            '      --idx-path   data_out/wind_mrhr_cnn_expanded1344/idx.npy \\\n'
            f'      --n-expected {N_TRAIN} \\\n'
            f'      --out-dir    {CONSTRAINTS_NPZ.parent}'
        )

    constraints, sample_count, constraint_idx = _load_constraints(CONSTRAINTS_NPZ)
    n_samples = len(constraint_idx)
    print(f'  Constraints NPZ    : {CONSTRAINTS_NPZ}')
    print(f'  n_samples          : {n_samples}')
    if n_samples != N_TRAIN:
        sys.exit(f'[error] Constraints NPZ has {n_samples} samples; expected {N_TRAIN}.')

    sc_min, sc_mean, sc_max = int(sample_count.min()), float(sample_count.mean()), int(sample_count.max())
    print(f'  sample_count       : min={sc_min} mean={sc_mean:.2f} max={sc_max}  (MAX_PAIRS={MAX_PAIRS})')
    if sc_max > MAX_PAIRS:
        sys.exit(
            f'[error] sample_count max ({sc_max}) exceeds MAX_PAIRS ({MAX_PAIRS}); '
            'padding/masking would silently truncate real pairs. Aborting.'
        )
    if sc_min != MAX_PAIRS or sc_max != MAX_PAIRS:
        print(
            f'  [WARN] sample_count is not uniformly {MAX_PAIRS} (min={sc_min}, max={sc_max}). '
            'This is handled correctly via the per-pair valid mask, but is worth '
            'double-checking against the constraint-building run if unexpected.'
        )

    print('  Reading training TFRecord sample indices …')
    train_indices = _read_tfrecord_indices(TRAIN_DATA_PATH)
    print(f'  Training TFRecord  : {len(train_indices)} records')
    if len(train_indices) != N_TRAIN:
        sys.exit(
            f'[error] Training TFRecord has {len(train_indices)} records; expected {N_TRAIN}.'
        )
    if len(set(train_indices.tolist())) != N_TRAIN:
        sys.exit('[error] Training TFRecord contains duplicate index values.')

    missing = sorted(set(train_indices.tolist()) - set(constraint_idx.tolist()))
    if missing:
        sys.exit(
            f'[error] {len(missing)} training sample indices not found in constraints NPZ '
            f'(first 10): {missing[:10]}\n'
            '  Re-run the constraint builder for this dataset.'
        )
    print(f'  All {N_TRAIN} training indices confirmed present in constraints.')

    print()
    print('  Lambda values:')
    print(f'    LAMBDA_SPEED    = {LAMBDA_SPEED}  (DISABLED)')
    print(f'    LAMBDA_GRAD     = {LAMBDA_GRAD}  (DISABLED)')
    print(f'    LAMBDA_WPD      = {LAMBDA_WPD}')
    print(f'    LAMBDA_LEVELSET = {LAMBDA_LEVELSET}  (DISABLED)')
    print(f'    LAMBDA_CRIT     = {LAMBDA_CRIT}  (DISABLED)')
    print('  UV+E2-low: L_speed/L_grad/L_levelset/L_crit disabled; only L_uv + repaired E2 active.')
    print(f'    LAMBDA_TTKCV    = {LAMBDA_TTKCV}  (low-lambda)')
    print(f'    LAMBDA_TTKPERS  = {LAMBDA_TTKPERS}  (low-lambda)')
    print()
    print('  Output directories:')
    print(f'    MODEL_OUT = {MODEL_OUT}')
    print(f'    DATA_OUT  = {DATA_OUT}')
    print(f'    LOG_PATH  = {LOG_PATH}')
    print('=' * 64)
    print()

    return constraints


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
    print('Candidate UV+E2-tf-lowlambda-expanded-1344 fine-tuning (Candidate B scaffold ablation)')
    print('=' * 64)
    print(f'  Training data     : {TRAIN_DATA_PATH}')
    print(f'  Evaluation data   : {EVAL_DATA_PATH}')
    print(f'  Source checkpoint : {MODEL_PATH}')
    print(f'  Output model dir  : {MODEL_OUT}')
    print(f'  Inference output  : {DATA_OUT}')
    print(f'  Log file          : {LOG_PATH}')
    print('=' * 64)
    print()

    phire = PhIREGANs(
        data_type    = 'wind_finetune_candidateUV_plus_E2_tf_lowlambda_expanded1344',
        learning_rate= LEARNING_RATE,
        N_epochs     = N_EPOCHS,
        save_every   = SAVE_EVERY,
        print_every  = PRINT_EVERY,
        mu_sig       = MU_SIG,
    )
    phire.setModel_name(MODEL_OUT)

    # ── Preflight ────────────────────────────────────────────────────────
    constraints = _run_preflight_checks(phire)

    # ── Phase 1: Build graph ────────────────────────────────────────────
    tf.reset_default_graph()
    phire.set_LR_data_shape(TRAIN_DATA_PATH)
    h, w, C = phire.LR_data_shape

    print('Initializing network …', end=' ')
    x_LR = tf.placeholder(tf.float32, [None, h,                w,               C])
    x_HR = tf.placeholder(tf.float32, [None, h * np.prod(R),   w * np.prod(R),  C])

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
    model = SR_NETWORK(x_LR, x_HR, r=R, status='pretraining', phys_cfg=cfg)

    ttk_ph, L_ttkcv, L_ttkpers = build_ttk_loss_ops(model, MU_SIG)
    total_loss = model.g_loss + LAMBDA_TTKCV * L_ttkcv + LAMBDA_TTKPERS * L_ttkpers

    optimizer = tf.train.AdamOptimizer(learning_rate=phire.learning_rate)
    train_op  = optimizer.minimize(total_loss, var_list=model.g_variables)
    init      = tf.global_variables_initializer()
    g_saver   = tf.train.Saver(var_list=model.g_variables, max_to_keep=10000)
    print('Done.')

    print('Building data pipeline …', end=' ')
    ds = tf.data.TFRecordDataset(TRAIN_DATA_PATH)
    ds = ds.map(lambda xx: phire._parse_train_(xx, phire.mu_sig)).shuffle(1000).batch(BATCH_SIZE)
    iterator = tf.data.Iterator.from_structure(ds.output_types, ds.output_shapes)
    idx_out, LR_out, HR_out = iterator.get_next()
    init_iter = iterator.make_initializer(ds)
    print('Done.')

    with tf.Session() as sess:
        sess.run(init)

        print('Loading pretrained checkpoint …', end=' ')
        g_saver.restore(sess, MODEL_PATH)
        print('Done.')

        # ── Loss-magnitude diagnostic (before any training step) ───────
        print()
        print('=' * 64)
        print('Loss-magnitude diagnostic (first batch, no training step)')
        print('=' * 64)
        sess.run(init_iter)
        diag_idx, diag_LR, diag_HR = sess.run([idx_out, LR_out, HR_out])
        (d_birth_yx, d_death_yx, d_birth_val, d_death_val,
         d_persistence, d_valid_mask) = _build_batch_ttk_feed(constraints, diag_idx)

        diag_feed = {
            x_LR: diag_LR, x_HR: diag_HR,
            ttk_ph['birth_yx']: d_birth_yx, ttk_ph['death_yx']: d_death_yx,
            ttk_ph['birth_val']: d_birth_val, ttk_ph['death_val']: d_death_val,
            ttk_ph['persistence']: d_persistence, ttk_ph['valid_mask']: d_valid_mask,
        }
        diag_vals = sess.run(
            {
                'L_uv': model.loss_breakdown['L_uv'],
                'L_speed (raw, DISABLED)': model.loss_breakdown['L_speed'],
                'L_grad (raw, DISABLED)': model.loss_breakdown['L_grad'],
                'L_levelset (raw, DISABLED)': model.loss_breakdown['L_levelset'],
                'L_crit (raw, DISABLED)': model.loss_breakdown['L_crit'],
                'L_ttkcv': L_ttkcv,
                'L_ttkpers': L_ttkpers,
                'g_loss (Candidate UV stack)': model.g_loss,
                'total_loss (+ TTK)': total_loss,
            },
            feed_dict=diag_feed,
        )
        print('  NOTE: L_speed/L_grad/L_levelset/L_crit are shown for reference only')
        print('  (raw, unweighted values). LAMBDA_SPEED=LAMBDA_GRAD=LAMBDA_LEVELSET=')
        print('  LAMBDA_CRIT=0.0, so they contribute exactly 0.0 to g_loss/total_loss below.')
        for key, val in diag_vals.items():
            print(f'  {key:<28} {val:.6f}')
        print('=' * 64)
        print()

        # ── Phase 1: Training ────────────────────────────────────────────
        print('Phase 1: Training on 1344 expanded samples …')
        iters = 0
        first_batch = (diag_idx, diag_LR, diag_HR)   # reuse the diagnostic batch as step 1

        for epoch in range(1, N_EPOCHS + 1):
            print(f'Epoch: {epoch}')
            if epoch > 1:
                sess.run(init_iter)

            epoch_loss, N = 0.0, 0
            try:
                while True:
                    if epoch == 1 and N == 0:
                        batch_idx, batch_LR, batch_HR = first_batch
                    else:
                        batch_idx, batch_LR, batch_HR = sess.run([idx_out, LR_out, HR_out])

                    N_batch = batch_LR.shape[0]
                    (b_birth_yx, b_death_yx, b_birth_val, b_death_val,
                     b_persistence, b_valid_mask) = _build_batch_ttk_feed(constraints, batch_idx)

                    feed_dict = {
                        x_LR: batch_LR, x_HR: batch_HR,
                        ttk_ph['birth_yx']: b_birth_yx, ttk_ph['death_yx']: b_death_yx,
                        ttk_ph['birth_val']: b_birth_val, ttk_ph['death_val']: b_death_val,
                        ttk_ph['persistence']: b_persistence, ttk_ph['valid_mask']: b_valid_mask,
                    }

                    _, tl = sess.run([train_op, total_loss], feed_dict=feed_dict)

                    epoch_loss += tl * N_batch
                    N += N_batch
                    iters += 1

                    if (iters % PRINT_EVERY) == 0:
                        print(f'Iteration={iters}, total loss={tl:.5f}')
                        bv = sess.run(
                            {**{k: v for k, v in model.loss_breakdown.items()},
                             'L_ttkcv': L_ttkcv, 'L_ttkpers': L_ttkpers,
                             'total_loss_with_ttk': total_loss},
                            feed_dict=feed_dict,
                        )
                        disabled_notes = {
                            'L_speed':    '  <- raw, unweighted; DISABLED (LAMBDA_SPEED=0.0)',
                            'L_grad':     '  <- raw, unweighted; DISABLED (LAMBDA_GRAD=0.0)',
                            'L_levelset': '  <- raw, unweighted; DISABLED (LAMBDA_LEVELSET=0.0)',
                            'L_crit':     '  <- raw, unweighted; DISABLED (LAMBDA_CRIT=0.0)',
                        }
                        print('  Loss breakdown:')
                        for key in ('L_uv', 'L_speed', 'L_grad', 'L_wpd', 'L_levelset',
                                    'L_crit', 'L_ttkcv', 'L_ttkpers', 'total_loss_with_ttk'):
                            note = disabled_notes.get(key, '')
                            print(f'    {key + ":":<22} {bv[key]:.6f}{note}')
            except tf.errors.OutOfRangeError:
                pass

            if (epoch % SAVE_EVERY) == 0:
                model_dir = '/'.join([phire.model_name, 'cnn{0:05d}'.format(epoch)])
                os.makedirs(model_dir, exist_ok=True)
                saved_model = '/'.join([model_dir, 'cnn'])
                g_saver.save(sess, saved_model)
                print(f'Checkpoint: {saved_model}')

            epoch_loss = epoch_loss / max(N, 1)
            print(f'Epoch total loss (with TTK) = {epoch_loss:.5f}')
            print()

        model_dir = '/'.join([phire.model_name, 'cnn'])
        os.makedirs(model_dir, exist_ok=True)
        saved_model = '/'.join([model_dir, 'cnn'])
        g_saver.save(sess, saved_model)
        print(f'Final checkpoint: {saved_model}')

    print()
    print('Fine-tuning complete.')
    print()

    # ── Phase 2: Paired inference on the 168-sample benchmark ─────────────
    # Identical to candidateB_plus_E2_tf_lowlambda_expanded1344 -- TTK losses
    # are training-time only, no special handling needed at inference time.
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
    print('      --candidate-name candidateUV_plus_E2_tf_lowlambda_expanded1344 \\')
    print('      --candidate-dir  data_out/wind_finetune_candidateUV_plus_E2_tf_lowlambda_expanded1344 \\')
    print('      --cnn-dir        data_out_fixed/wind_mrhr_cnn \\')
    print('      --gan-dir        data_out_fixed/wind_mrhr_gan \\')
    print('      --merged-csv     ttk_runs_fixed/combined/psnr_topology_physics_merged.csv \\')
    print('      --out-dir        ttk_runs_fixed/topology_finetuning/candidateUV_plus_E2_tf_lowlambda_expanded1344_eval')
    print()
    print('  Only run the expensive TTK topology pipeline AFTER confirming')
    print('  the cheap evaluation above is non-catastrophic:')
    print('    bash scripts/run_candidate_topology_pipeline.sh \\')
    print('      --method     candidateUV_plus_E2_tf_lowlambda_expanded1344 \\')
    print('      --data-dir   data_out/wind_finetune_candidateUV_plus_E2_tf_lowlambda_expanded1344 \\')
    print('      --vti-dir    ttk_runs_fixed/topology_finetuning/candidateUV_plus_E2_tf_lowlambda_expanded1344_topology_vti \\')
    print('      --out-base   ttk_runs_fixed/topology_finetuning/candidateUV_plus_E2_tf_lowlambda_expanded1344_topology \\')
    print('      --n-samples  168 \\')
    print('      --threads    1 \\')
    print('      --skip-viz')
    print('=' * 64)


if __name__ == '__main__':
    main()
