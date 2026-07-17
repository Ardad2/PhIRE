#!/usr/bin/env python3
"""
Candidate F: native PhIRE/TensorFlow generator fine-tuning, expanded-2688
scale. Three variants that recombine the two dominant descriptor-specific
signals identified so far -- L_grad (the strongest PD-oriented term inside
Candidate B) and repaired low-lambda TTK fixed-index supervision (the
strongest MT-oriented signal, from the UV+E2-low family) -- plus a control
that pairs L_grad directly with Candidate C's local-maxima proxy instead of
the full Candidate B scaffold.

Motivation
-----------
Candidate B-2688 (PD 22.7070, MT 6.1612) is close to Candidate C-2688
(PD 22.4944, MT 6.0803) on PD but worse on MT; the Candidate B factorial
ablation (scripts/run_candidateB_factorial_expanded2688_finetune.py)
isolated L_grad as the PD-driving term inside Candidate B. Separately, the
UV+E2-low family showed that repaired TTK fixed-index supervision alone
drives a strong MT signal without any Candidate B scaffold at all. Candidate
F asks: what happens when L_grad (PD) and repaired E2 (MT) are combined
directly, without the rest of Candidate B, and does adding L_levelset back
help or hurt? A third control pairs L_grad with Candidate C's L_crit
directly (no Candidate B scaffold, no E2) to isolate whether L_crit adds
anything beyond L_grad alone.

Three variants (single parameterized script, not three copies -- see
"Implementation structure" below):

  grad_e2_low            method: candidateF_grad_E2_low_expanded2688
    L_total = L_uv + 0.05 L_grad + 0.004 L_TTKCV + 0.002 L_TTKpers
    L_speed=L_levelset=L_crit=0. Uses the repaired native-TF E2 fixed-index
    constraint machinery (same as the completed *_plus_E2_tf_lowlambda_*
    scripts), reading the completed 2688-sample constraints NPZ.

  grad_levelset_e2_low   method: candidateF_grad_levelset_E2_low_expanded2688
    L_total = L_uv + 0.05 L_grad + 0.25 L_levelset + 0.004 L_TTKCV + 0.002 L_TTKpers
    L_speed=L_crit=0. Same E2 machinery as grad_e2_low.

  grad_crit               method: candidateF_grad_crit_expanded2688
    L_total = L_uv + 0.05 L_grad + 0.001 L_crit
    L_speed=L_levelset=L_TTKCV=L_TTKpers=0. NO E2 constraints NPZ is loaded
    and NO E2 loss ops are built for this variant at all -- it uses the
    plain PhIREGANs.pretrain() path, exactly like
    candidateB_expanded2688/candidateC_expanded2688/the Candidate B
    factorial script.

Implementation structure
--------------------------
This single script covers all three variants via a VARIANTS configuration
map (--variant grad_e2_low|grad_levelset_e2_low|grad_crit), following
scripts/run_candidateB_factorial_expanded2688_finetune.py's pattern for
configurable loss-weight handling, protection, and CLI conventions. Because
two variants need repaired TTK fixed-index supervision and one does not,
this script also reuses the custom TF1 training-loop machinery from
scripts/run_candidateUV_plus_E2_tf_lowlambda_expanded2688_ttkcrit_refiner.py
/ scripts/run_candidateB_plus_E2_tf_lowlambda_expanded2688_ttkcrit_refiner.py
(constraint loading + validation, _gather_speed_at_yx, _masked_mse,
build_ttk_loss_ops, the feed_dict-based training loop keyed by the
TFRecord-embedded 'index' feature) for the two E2 variants, and the plain
PhIREGANs.pretrain() path for grad_crit. VARIANTS['requires_e2'] selects
which training path runs; grad_crit's branch never imports/calls the
constraint loader or builds any TTK op.

Repaired E2 constraints (grad_e2_low / grad_levelset_e2_low only)
---------------------------------------------------------------------
    ttk_runs_fixed/topology_finetuning/candidateE2_fixed_lowlambda_expanded2688_constraints/
      ttk_pd_critical_pairs_gtvalues.npz

This is the exact same constraints file already used, unmodified, by
candidateE2_tf_lowlambda_expanded2688 (C+E2), candidateB_plus_E2_tf_lowlambda_expanded2688
(B+E2), and candidateUV_plus_E2_tf_lowlambda_expanded2688 (UV+E2) -- verified
by grepping CONSTRAINTS_NPZ in all three scripts before writing this one.
This script never regenerates or modifies that file; preflight only reads
and validates it (see _load_constraints_and_report below), loaded with
allow_pickle=False (every field is a plain numeric array; no script in this
repo's E2 family relies on pickled objects inside this NPZ). Hard errors
(not warnings) on any of: stored n_samples != 2688; sample_idx/sample_start/
sample_count shape != (2688,); sample_idx is not exactly the set
{0, ..., 2687} (each integer exactly once); sample_count != 64 for any
sample; any sample_start < 0; any sample_start+sample_count exceeding the
birth/death pair-array length; birth_vid/death_vid/birth_val/death_val/
persistence lengths disagreeing; any birth/death vertex ID outside
[0, PATCH*PATCH-1] (PATCH=160, matching the top-left-anchored 160x160 VTK
crop); any non-finite target/persistence value; or any negative persistence
value. The training TFRecord's sample-index set must equal the NPZ's
sample-index set EXACTLY (not merely be a subset of it).

Normalization / physical-unit conventions (identical to every other native
TF candidate script in this repo)
-----------------------------------------------------------------------------
  - L_uv is MSE in NORMALIZED [u, v] space (sr_network.SR_NETWORK's
    normal loss_breakdown['L_uv'], built by PhysicsLossConfig/SR_NETWORK,
    completely unmodified).
  - L_grad, L_levelset, L_crit, L_TTKCV, L_TTKpers all operate on
    DENORMALIZED physical scalar speed: u_phys = sigma_u*u_norm + mu_u,
    v_phys = sigma_v*v_norm + mu_v, speed = sqrt(u_phys^2 + v_phys^2),
    using the same MU_SIG = [[0.7684, -0.4575], [5.02455, 5.9017]] and the
    same _denorm_speed() helper the E2 scripts already use internally.
  - This script does not edit sr_network.py or PhIREGANs.py.

Same hyperparameters/data/inference procedure as the completed Candidate B
factorial and native E2-low 2688 runs in every variant: learning_rate=1e-5,
N_epochs=3, batch_size=4, save_every=1, print_every=2, training data =
example_data_topology_expanded_2688/wind_MR-HR.tfrecord (2688 samples, the
SAME file as candidateC_expanded2688/candidateB_expanded2688/every E2-low-2688
script), evaluation = example_data_fixed/wind_MR-HR.tfrecord (the fixed
168-sample benchmark -- never trained on).

Usage (from repo root, on Spark)
-----------------------------------
  Print the full static configuration table for all three variants (no
  training, no file I/O beyond reading argv):
    python3 scripts/run_candidateF_expanded2688_finetune.py --print-config

  Dry run for one variant (preflight + collision checks + NPZ validation +
  plan; no training):
    python3 scripts/run_candidateF_expanded2688_finetune.py --variant grad_e2_low --dry-run
    python3 scripts/run_candidateF_expanded2688_finetune.py --variant grad_levelset_e2_low --dry-run
    python3 scripts/run_candidateF_expanded2688_finetune.py --variant grad_crit --dry-run

  Full run:
    python3 scripts/run_candidateF_expanded2688_finetune.py --variant grad_e2_low
    python3 scripts/run_candidateF_expanded2688_finetune.py --variant grad_levelset_e2_low
    python3 scripts/run_candidateF_expanded2688_finetune.py --variant grad_crit

Collision protection
-----------------------
By default (and ALWAYS -- there is no override flag) this script ABORTS if
ANY of its target artifact paths already exist and look completed -- model
dir, data-out dir, cheap-eval dir, TTK VTI dir, TTK topology-out dir, the
eval/topology markdown reports, or the canonical training log -- printing
every colliding path, not just the first one. There is deliberately no
--overwrite or --resume flag: PhIREGANs.pretrain() always retrains from the
pretrained CNN checkpoint from scratch in this codebase (there is no
partial-training resume capability), so an "overwrite" flag would silently
retrain under the same method name while potentially leaving stale
cheap-evaluation or TTK outputs from an older model version behind. If a
prior run for this exact method needs to be redone, inspect its artifacts
first, then deliberately archive or delete them yourself (this script never
deletes anything automatically) before rerunning.

Dry runs and --print-config are artifact-free: no canonical training log,
model checkpoint, inference output, or any other file is ever created by
--print-config or --dry-run. The canonical training log is opened for the
first time only after every collision/preflight/NPZ-validation check has
already passed for an actual (non-dry-run) invocation -- a failed preflight
or a --dry-run therefore can never "poison" a subsequent real run by
pre-creating a log file that then looks like a collision. If you want
preflight/dry-run output captured to a file, tee the command yourself, e.g.
`python3 scripts/run_candidateF_expanded2688_finetune.py --variant
grad_e2_low --dry-run 2>&1 | tee /tmp/preflight.log` -- that file is not
treated as an experiment artifact by this script.

This script never runs the expensive TTK topology pipeline automatically --
it prints the exact command at the end (see
scripts/run_candidateF_expanded2688_batch.sh for the batch driver, which
runs cheap evaluation automatically after a successful run but still never
runs TTK automatically).
"""

from __future__ import annotations

import argparse
import sys
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure repo root is on sys.path AND is the working directory, so every
# relative path used below (TRAIN_DATA_PATH, EVAL_DATA_PATH, model_dir,
# data_out_dir, etc.) resolves correctly regardless of the directory this
# script was invoked from.
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

# ---------------------------------------------------------------------------
# Shared constants (identical to candidateB_expanded2688 / candidateC_expanded2688
# / the completed native E2-low-2688 scripts)
# ---------------------------------------------------------------------------
TRAIN_DATA_PATH = 'example_data_topology_expanded_2688/wind_MR-HR.tfrecord'
EVAL_DATA_PATH  = 'example_data_fixed/wind_MR-HR.tfrecord'
MODEL_PATH      = 'models/wind_mr-hr/trained_cnn/cnn'
CNN_BASELINE_DIR = 'data_out_fixed/wind_mrhr_cnn'  # authoritative fixed-benchmark GT/IN reference

MU_SIG = [[0.7684, -0.4575], [5.02455, 5.9017]]
R      = [5]

N_TRAIN = 2688
N_EVAL  = 168
PATCH      = 160   # vertex-ID encoding width; matches the constraint builder
MAX_PAIRS  = 64    # padded per-sample pair count
EXPECTED_PAIRS_PER_SAMPLE = 64  # the known completed constraint set: exactly 64 pairs/sample

LEARNING_RATE = 1e-5
N_EPOCHS      = 3
SAVE_EVERY    = 1
PRINT_EVERY   = 2
BATCH_SIZE    = 4

# Repaired E2 constraints -- identical file already used, unmodified, by
# candidateE2_tf_lowlambda_expanded2688 / candidateB_plus_E2_tf_lowlambda_expanded2688 /
# candidateUV_plus_E2_tf_lowlambda_expanded2688 (verified by inspection before writing
# this script; no path mismatch found, so no alternate NPZ search was needed).
CONSTRAINTS_NPZ_2688 = (
    REPO_ROOT / 'ttk_runs_fixed' / 'topology_finetuning'
    / 'candidateE2_fixed_lowlambda_expanded2688_constraints' / 'ttk_pd_critical_pairs_gtvalues.npz'
)

# ---------------------------------------------------------------------------
# Variant configuration map
# ---------------------------------------------------------------------------
VARIANTS = {
    'grad_e2_low': {
        'method_name':    'candidateF_grad_E2_low_expanded2688',
        'lambda_speed':    0.0,
        'lambda_grad':     0.05,
        'lambda_levelset': 0.0,
        'lambda_crit':     0.0,
        'lambda_ttkcv':    0.004,
        'lambda_ttkpers':  0.002,
        'requires_e2':     True,
    },
    'grad_levelset_e2_low': {
        'method_name':    'candidateF_grad_levelset_E2_low_expanded2688',
        'lambda_speed':    0.0,
        'lambda_grad':     0.05,
        'lambda_levelset': 0.25,
        'lambda_crit':     0.0,
        'lambda_ttkcv':    0.004,
        'lambda_ttkpers':  0.002,
        'requires_e2':     True,
    },
    'grad_crit': {
        'method_name':    'candidateF_grad_crit_expanded2688',
        'lambda_speed':    0.0,
        'lambda_grad':     0.05,
        'lambda_levelset': 0.0,
        'lambda_crit':     0.001,
        'lambda_ttkcv':    0.0,
        'lambda_ttkpers':  0.0,
        'requires_e2':     False,
    },
}

# Recommended run order: the cleanest direct PD+MT combination first, then
# the level-set interaction test, then the local-maxima-only control.
PREFERRED_ORDER = ['grad_e2_low', 'grad_levelset_e2_low', 'grad_crit']

# Exact per-variant assertions required by this task -- checked verbatim
# against VARIANTS at startup (both --print-config and real runs). A
# mismatch here means the configuration map was edited incorrectly; fail
# immediately rather than silently training the wrong objective.
EXPECTED_ASSERTIONS = {
    'grad_e2_low': {
        'lambda_speed': 0.0, 'lambda_grad': 0.05, 'lambda_levelset': 0.0,
        'lambda_crit': 0.0, 'lambda_ttkcv': 0.004, 'lambda_ttkpers': 0.002,
        'requires_e2': True,
    },
    'grad_levelset_e2_low': {
        'lambda_speed': 0.0, 'lambda_grad': 0.05, 'lambda_levelset': 0.25,
        'lambda_crit': 0.0, 'lambda_ttkcv': 0.004, 'lambda_ttkpers': 0.002,
        'requires_e2': True,
    },
    'grad_crit': {
        'lambda_speed': 0.0, 'lambda_grad': 0.05, 'lambda_levelset': 0.0,
        'lambda_crit': 0.001, 'lambda_ttkcv': 0.0, 'lambda_ttkpers': 0.0,
        'requires_e2': False,
    },
}


def _validate_variant_configs() -> None:
    """Fail immediately if VARIANTS drifted from the exact per-variant
    assertions this task requires (an unexpected loss active, or E2 wired
    up for grad_crit)."""
    for name, expected in EXPECTED_ASSERTIONS.items():
        cfg = VARIANTS[name]
        for key, exp_val in expected.items():
            got = cfg[key]
            if got != exp_val:
                sys.exit(
                    f"[error] Variant {name!r} configuration assertion failed: "
                    f"{key}={got!r}, expected {exp_val!r}. Refusing to proceed with an "
                    f"unexpected loss configuration."
                )
    # grad_crit-specific: must not be able to load or require E2 constraints.
    if VARIANTS['grad_crit']['requires_e2'] is not False:
        sys.exit("[error] grad_crit must have requires_e2=False.")
    if constraints_npz_for('grad_crit') is not None:
        sys.exit("[error] grad_crit must resolve to constraints_npz=None.")


def constraints_npz_for(variant: str):
    return CONSTRAINTS_NPZ_2688 if VARIANTS[variant]['requires_e2'] else None


def method_paths(method_name: str) -> dict:
    """Derive every artifact path for a method_name, following the exact
    naming convention already used by every other native-TF candidate
    script in this repo (data_out/wind_finetune_<name>, models_fixed/
    topology_finetuning/wind_finetune_<name>, ttk_runs_fixed/
    topology_finetuning/<name>_eval, etc.)."""
    return {
        'method_name':      method_name,
        'model_dir':        f'models_fixed/topology_finetuning/wind_finetune_{method_name}',
        'data_out_dir':     f'data_out/wind_finetune_{method_name}',
        'log_path':         REPO_ROOT / 'logs' / f'wind_finetune_{method_name}.log',
        'cheap_eval_dir':   f'ttk_runs_fixed/topology_finetuning/{method_name}_eval',
        'eval_report':      f'docs/topology_finetuning_{method_name}_eval.md',
        'topology_vti_dir': f'ttk_runs_fixed/topology_finetuning/{method_name}_topology_vti',
        'topology_out_dir': f'ttk_runs_fixed/topology_finetuning/{method_name}_topology',
        'topology_report':  f'docs/topology_finetuning_{method_name}_topology_eval.md',
    }


# ---------------------------------------------------------------------------
# Exhaustive list of every completed candidate's method_name, used to
# derive the full _PROTECTED path list below (data_out, models_fixed,
# cheap-eval, TTK VTI/topology, and doc-report paths for each). Compiled by
# grepping every data_out/wind_finetune_<name> literal in scripts/*.py
# before writing this script.
# ---------------------------------------------------------------------------
COMPLETED_METHOD_NAMES = [
    # Original 168-sample pilot candidates
    'pilot_candidateB', 'pilot_candidateC', 'pilot_candidateD',
    'pilot_candidateE', 'pilot_candidateE2', 'pilot_candidateUV',
    # Expanded UV/B/C/D/Dpd at every completed scale
    'candidateUV_expanded672', 'candidateUV_expanded1344', 'candidateUV_expanded2688',
    'candidateB_expanded672', 'candidateB_expanded2688',
    'candidateC_expanded672', 'candidateC_expanded1344', 'candidateC_expanded2688',
    'candidateD_expanded672', 'candidateDpd_expanded672',
    # PyTorch residual-refiner E2 family -- distinct architecture
    'candidateE2_expanded672', 'candidateE2_fixed', 'candidateE2_fixed_lowlambda',
    'candidateE2_fixed_lowlambda_expanded1344', 'candidateE2_fixed_lowlambda_expanded2688',
    # Native TF C+E2-low family, all scales
    'candidateE2_tf_lowlambda_expanded672', 'candidateE2_tf_lowlambda_expanded1344',
    'candidateE2_tf_lowlambda_expanded2688',
    # Native TF B+E2-low family, all scales
    'candidateB_plus_E2_tf_lowlambda_expanded672', 'candidateB_plus_E2_tf_lowlambda_expanded1344',
    'candidateB_plus_E2_tf_lowlambda_expanded2688',
    # Native TF UV+E2-low family, all scales
    'candidateUV_plus_E2_tf_lowlambda_expanded672', 'candidateUV_plus_E2_tf_lowlambda_expanded1344',
    'candidateUV_plus_E2_tf_lowlambda_expanded2688',
    # Native TF UV+crit family, all scales
    'candidateUV_plus_crit_expanded672', 'candidateUV_plus_crit_expanded1344',
    'candidateUV_plus_crit_expanded2688',
    # Candidate B factorial (all six variants)
    'candidateB_factorial_speed_expanded2688', 'candidateB_factorial_grad_expanded2688',
    'candidateB_factorial_levelset_expanded2688', 'candidateB_factorial_speed_grad_expanded2688',
    'candidateB_factorial_speed_levelset_expanded2688', 'candidateB_factorial_grad_levelset_expanded2688',
    # Stale alias defensively protected elsewhere in the repo (never a real
    # candidate output; see scripts/run_superlevel_topology_robustness.py history)
    'candidateC_plus_E2_tf_lowlambda_expanded2688',
]


def _build_protected_list() -> list:
    protected = [
        # Source checkpoint
        'models/wind_mr-hr/trained_cnn',
        # Benchmark data
        'example_data_fixed',
        # Baseline model outputs
        'data_out_fixed/wind_mrhr_cnn',
        'data_out_fixed/wind_mrhr_gan',
        # Priority 6 superlevel topology robustness outputs (a whole separate
        # output root; never touched by this script, protected defensively)
        'ttk_runs_fixed/superlevel_topology',
        # Combined baseline CSVs/reports
        'ttk_runs_fixed/combined',
        # Repaired E2 constraint artifacts (read-only input; never written)
        'ttk_runs_fixed/topology_finetuning/candidateE2_fixed_constraints',
        'ttk_runs_fixed/topology_finetuning/candidateE2_expanded672_constraints',
        'ttk_runs_fixed/topology_finetuning/candidateE2_fixed_lowlambda_expanded1344_constraints',
        'ttk_runs_fixed/topology_finetuning/candidateE2_fixed_lowlambda_expanded2688_constraints',
    ]
    for name in COMPLETED_METHOD_NAMES:
        p = method_paths(name)
        protected += [
            p['data_out_dir'], p['model_dir'], p['cheap_eval_dir'],
            p['topology_vti_dir'], p['topology_out_dir'],
            p['eval_report'], p['topology_report'],
        ]
    return protected


# ---------------------------------------------------------------------------
# Static configuration table (--print-config)
# ---------------------------------------------------------------------------

def _print_config_table() -> None:
    _validate_variant_configs()
    cols = [
        'variant', 'method_name', 'training_tfrecord', 'evaluation_tfrecord',
        'constraints_npz', 'N_TRAIN', 'N_EVAL', 'lambda_uv', 'lambda_speed',
        'lambda_grad', 'lambda_levelset', 'lambda_crit', 'lambda_ttkcv',
        'lambda_ttkpers', 'model_dir', 'data_out_dir', 'cheap_eval_dir',
        'topology_vti_dir', 'topology_out_dir',
    ]
    print('=' * 100)
    print('Candidate F expanded-2688: static configuration')
    print('=' * 100)
    for variant in PREFERRED_ORDER:
        cfg = VARIANTS[variant]
        paths = method_paths(cfg['method_name'])
        npz = constraints_npz_for(variant)
        row = {
            'variant': variant,
            'method_name': cfg['method_name'],
            'training_tfrecord': TRAIN_DATA_PATH,
            'evaluation_tfrecord': EVAL_DATA_PATH,
            'constraints_npz': str(npz) if npz is not None else 'NONE',
            'N_TRAIN': N_TRAIN,
            'N_EVAL': N_EVAL,
            'lambda_uv': 1.0,
            'lambda_speed': cfg['lambda_speed'],
            'lambda_grad': cfg['lambda_grad'],
            'lambda_levelset': cfg['lambda_levelset'],
            'lambda_crit': cfg['lambda_crit'],
            'lambda_ttkcv': cfg['lambda_ttkcv'],
            'lambda_ttkpers': cfg['lambda_ttkpers'],
            'model_dir': paths['model_dir'],
            'data_out_dir': paths['data_out_dir'],
            'cheap_eval_dir': paths['cheap_eval_dir'],
            'topology_vti_dir': paths['topology_vti_dir'],
            'topology_out_dir': paths['topology_out_dir'],
        }
        print(f'--- {variant} ---')
        for c in cols:
            print(f'  {c:<20}: {row[c]}')
        print()
    print('=' * 100)
    print('All per-variant assertions verified: no unexpected loss is active in any variant.')
    print("grad_crit confirmed: requires_e2=False, constraints_npz=NONE "
          "(no E2 constraint loader or E2 loss ops can execute for this variant).")
    print('=' * 100)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument('--variant', choices=sorted(VARIANTS), default=None,
                     help='Which Candidate F variant to train/inspect.')
    ap.add_argument('--print-config', action='store_true',
                     help='Print the static configuration table for all three variants and exit. '
                          'No training, no Docker/TTK, no file writes.')
    ap.add_argument('--dry-run', action='store_true',
                     help='Run preflight/collision/NPZ-validation checks for --variant and print '
                          'the plan; do not train. Creates no files at all (no canonical training '
                          'log, no model/data output).')
    return ap.parse_args()


def main() -> int:
    args = _parse_args()

    if args.print_config:
        _print_config_table()
        return 0

    if not args.variant:
        print('[error] --variant is required unless --print-config is given.')
        return 2

    _validate_variant_configs()

    variant = args.variant
    cfg = VARIANTS[variant]
    paths = method_paths(cfg['method_name'])
    constraints_npz = constraints_npz_for(variant)

    # -------------------------------------------------------------------
    # _Tee is defined here but NOT instantiated yet. It is only ever
    # instantiated after every collision/preflight/NPZ-validation check has
    # passed for a REAL (non-dry-run) invocation -- see the "if args.dry_run:
    # return" gate below. This guarantees --dry-run and --print-config never
    # create the canonical training log (or any other file), and that a
    # failed preflight can never leave behind a log file that a later run
    # would mistake for a collision.
    # -------------------------------------------------------------------
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

    # -------------------------------------------------------------------
    # Imports (plain stdout; no Tee has been created yet)
    # -------------------------------------------------------------------
    import numpy as np
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()

    from sr_network import PhysicsLossConfig, SR_NETWORK, _denorm_speed
    from PhIREGANs import PhIREGANs

    # -------------------------------------------------------------------
    # Protected paths: exhaustive completed-candidate list + sibling
    # Candidate F variants (never this variant's own paths).
    # -------------------------------------------------------------------
    protected = _build_protected_list()
    for other_variant, other_cfg in VARIANTS.items():
        if other_variant == variant:
            continue
        op = method_paths(other_cfg['method_name'])
        protected += [
            op['data_out_dir'], op['model_dir'], op['cheap_eval_dir'],
            op['topology_vti_dir'], op['topology_out_dir'],
            op['eval_report'], op['topology_report'],
        ]

    def _safety_check() -> None:
        own_dirs = (paths['model_dir'], paths['data_out_dir'],
                    paths['cheap_eval_dir'], paths['topology_vti_dir'],
                    paths['topology_out_dir'])
        for prot in protected:
            p = Path(prot).resolve()
            for candidate_path in own_dirs:
                c = Path(candidate_path).resolve()
                if c == p or str(c).startswith(str(p) + os.sep):
                    sys.exit(
                        f'[error] Output path {candidate_path!r} overlaps protected '
                        f'path {prot!r}. Aborting to avoid overwriting baselines.'
                    )
        print('[safety] Output paths do not overlap any protected directory. OK.')

    def _check_collisions() -> None:
        """Abort if any method-specific model, inference, evaluation, VTI,
        topology, report, or completed training-log artifact exists for
        this method. There is no override: this check cannot be bypassed
        by any flag. If a prior run needs to be redone, inspect and
        deliberately archive or remove all artifacts for this method
        yourself before rerunning -- this script never deletes anything.

        Note: the canonical training log (paths['log_path']) is only ever
        created below AFTER this check (and only for a real, non-dry-run
        invocation), so if it exists here, it can only be from a genuine
        prior real run -- never from this invocation's own dry-run or
        preflight output, which are never written to it."""
        existing = []
        for key in ('model_dir', 'data_out_dir', 'cheap_eval_dir',
                    'topology_vti_dir', 'topology_out_dir'):
            p = Path(paths[key])
            if p.exists() and any(p.iterdir()):
                existing.append(str(p))
        for key in ('eval_report', 'topology_report'):
            p = Path(paths[key])
            if p.exists():
                existing.append(str(p))
        if Path(paths['log_path']).exists() and Path(paths['log_path']).stat().st_size > 0:
            existing.append(str(paths['log_path']))

        if not existing:
            print('[collision-check] No completed-looking output exists yet for '
                  f'{paths["method_name"]!r}. OK to proceed.')
            return

        sys.exit(
            '[error] Completed-looking output(s) already exist for method '
            f'{paths["method_name"]!r}:\n'
            + '\n'.join(f'  - {e}' for e in existing)
            + '\n  Refusing to proceed automatically. There is no --overwrite/--resume flag: '
              'PhIREGANs.pretrain() always retrains from scratch in this codebase, so an '
              'automatic overwrite could silently retrain under this method name while leaving '
              'stale cheap-evaluation or TTK outputs from an older model behind. Inspect the '
              'path(s) above, then deliberately archive or remove ALL of them yourself before '
              'rerunning this exact variant. Nothing is deleted automatically.'
        )

    def _count_tfrecord_records(data_path: str) -> int:
        n = 0
        for _ in tf.python_io.tf_record_iterator(data_path):
            n += 1
        return n

    # =====================================================================
    # Repaired E2 constraint loading + validation (shared with the
    # completed *_plus_E2_tf_lowlambda_expanded2688 scripts; only invoked
    # when cfg['requires_e2'] is True)
    # =====================================================================

    def _read_tfrecord_indices(data_path: str) -> "np.ndarray":
        indices = []
        for raw in tf.python_io.tf_record_iterator(data_path):
            example = tf.train.Example()
            example.ParseFromString(raw)
            indices.append(example.features.feature['index'].int64_list.value[0])
        return np.array(indices, dtype=np.int64)

    def _load_constraints_and_report(npz_path: Path):
        """Load + strictly validate the constraints NPZ. allow_pickle=False:
        every field used here (sample_idx/sample_start/sample_count/
        birth_vid/death_vid/birth_val/death_val/persistence) is a plain
        numeric array in every reference E2 script in this repo; none
        requires pickled Python objects.

        Hard errors (never warnings) on:
          - stored n_samples != N_TRAIN
          - sample_idx/sample_start/sample_count shape != (N_TRAIN,)
          - sample_idx is not exactly {0, ..., N_TRAIN-1}, each exactly once
          - sample_count != EXPECTED_PAIRS_PER_SAMPLE for any sample
          - any sample_start < 0
          - any sample_start + sample_count exceeding the pair-array length
          - birth_vid/death_vid/birth_val/death_val/persistence length mismatch
          - any birth/death vertex ID outside [0, PATCH*PATCH-1]
          - any non-finite target or persistence value
          - any negative persistence value
        """
        npz = np.load(str(npz_path), allow_pickle=False)

        required = {
            'n_samples', 'sample_idx', 'sample_start', 'sample_count',
            'birth_vid', 'death_vid', 'birth_val', 'death_val', 'persistence',
        }
        missing = required - set(npz.files)
        if missing:
            sys.exit(f'[error] Constraints NPZ missing required keys: {missing}')

        stored_n_samples = npz['n_samples']
        # May be stored as a 0-d array or Python scalar depending on how it
        # was saved; np.asarray(...).item() handles both uniformly.
        stored_n_samples = int(np.asarray(stored_n_samples).item())

        sample_idx   = npz['sample_idx'].astype(np.int64)
        sample_start = npz['sample_start'].astype(np.int64)
        sample_count = npz['sample_count'].astype(np.int64)
        birth_vid    = npz['birth_vid'].astype(np.int64)
        death_vid    = npz['death_vid'].astype(np.int64)
        birth_val    = npz['birth_val'].astype(np.float32)
        death_val    = npz['death_val'].astype(np.float32)
        persistence  = npz['persistence'].astype(np.float32)

        print('  --- Constraints NPZ metadata validation ---')
        print(f'  stored n_samples field  : {stored_n_samples} (expected {N_TRAIN})')
        if stored_n_samples != N_TRAIN:
            sys.exit(f'[error] Constraints NPZ stored n_samples={stored_n_samples}; expected {N_TRAIN}.')

        for arr_name, arr in (('sample_idx', sample_idx), ('sample_start', sample_start),
                               ('sample_count', sample_count)):
            print(f'  {arr_name}.shape'.ljust(26) + f': {arr.shape} (expected ({N_TRAIN},))')
            if arr.shape != (N_TRAIN,):
                sys.exit(f'[error] Constraints NPZ {arr_name}.shape={arr.shape}; expected ({N_TRAIN},).')

        sample_idx_set = set(sample_idx.tolist())
        expected_set = set(range(N_TRAIN))
        print(f'  sample_idx range        : [{int(sample_idx.min())}, {int(sample_idx.max())}]')
        if sample_idx_set != expected_set:
            extra = sorted(sample_idx_set - expected_set)[:10]
            missing_idx = sorted(expected_set - sample_idx_set)[:10]
            sys.exit(
                f'[error] Constraints NPZ sample_idx is not exactly {{0, ..., {N_TRAIN - 1}}} '
                f'(each exactly once). Unexpected values (first 10): {extra}. '
                f'Missing values (first 10): {missing_idx}.'
            )
        if len(sample_idx.tolist()) != len(sample_idx_set):
            sys.exit('[error] Constraints NPZ sample_idx contains duplicate values.')
        print(f'  sample_idx == exactly {{0..{N_TRAIN - 1}}}, each once: confirmed')

        sc_min, sc_mean, sc_max = int(sample_count.min()), float(sample_count.mean()), int(sample_count.max())
        print(f'  pairs per sample       : min={sc_min} mean={sc_mean:.2f} max={sc_max} '
              f'(expected exactly {EXPECTED_PAIRS_PER_SAMPLE} for every sample)')
        if sc_min != EXPECTED_PAIRS_PER_SAMPLE or sc_max != EXPECTED_PAIRS_PER_SAMPLE:
            sys.exit(
                f'[error] Constraints NPZ does not have exactly {EXPECTED_PAIRS_PER_SAMPLE} '
                f'pairs per sample for every sample (min={sc_min}, max={sc_max}). The known '
                'completed constraint set has exactly 64 pairs/sample; treating this as an '
                'error rather than silently continuing, per this variant\'s validation policy.'
            )
        if sc_max > MAX_PAIRS:
            sys.exit(
                f'[error] sample_count max ({sc_max}) exceeds MAX_PAIRS ({MAX_PAIRS}); '
                'padding/masking would silently truncate real pairs. Aborting.'
            )

        if bool((sample_start < 0).any()):
            sys.exit('[error] Constraints NPZ contains negative sample_start values.')
        print('  sample_start all nonnegative: confirmed')

        pair_lengths = {name: len(arr) for name, arr in (
            ('birth_vid', birth_vid), ('death_vid', death_vid),
            ('birth_val', birth_val), ('death_val', death_val),
            ('persistence', persistence),
        )}
        print(f'  pair-array lengths      : {pair_lengths}')
        if len(set(pair_lengths.values())) != 1:
            sys.exit(f'[error] birth_vid/death_vid/birth_val/death_val/persistence have '
                      f'unequal lengths: {pair_lengths}.')
        pair_array_len = next(iter(pair_lengths.values()))

        row_end = sample_start + sample_count
        if bool((row_end > pair_array_len).any()):
            bad = int(np.argmax(row_end > pair_array_len))
            sys.exit(
                f'[error] sample_start+sample_count exceeds the pair-array length '
                f'({pair_array_len}) for at least one sample (first offending row index '
                f'{bad}: sample_start={int(sample_start[bad])}, '
                f'sample_count={int(sample_count[bad])}).'
            )
        print(f'  sample_start+sample_count never exceeds pair-array length ({pair_array_len}): confirmed')

        bv_min, bv_max = int(birth_vid.min()), int(birth_vid.max())
        dv_min, dv_max = int(death_vid.min()), int(death_vid.max())
        print(f'  birth vertex ID bounds : [{bv_min}, {bv_max}]')
        print(f'  death vertex ID bounds : [{dv_min}, {dv_max}]')
        max_valid_vid = PATCH * PATCH - 1
        if bv_min < 0 or bv_max > max_valid_vid or dv_min < 0 or dv_max > max_valid_vid:
            sys.exit(
                f'[error] birth/death vertex IDs out of the expected [0, {max_valid_vid}] range '
                f'for a {PATCH}x{PATCH} C-order VTK point-ID crop '
                f'(got birth=[{bv_min},{bv_max}], death=[{dv_min},{dv_max}]).'
            )

        finite_birth_val = bool(np.isfinite(birth_val).all())
        finite_death_val = bool(np.isfinite(death_val).all())
        finite_persistence = bool(np.isfinite(persistence).all())
        print(f'  finite birth_val target values      : {finite_birth_val}')
        print(f'  finite death_val target values      : {finite_death_val}')
        print(f'  finite persistence target values    : {finite_persistence}')
        if not (finite_birth_val and finite_death_val and finite_persistence):
            sys.exit('[error] Constraints NPZ contains non-finite target/persistence values.')

        persistence_nonneg = bool((persistence >= 0).all())
        print(f'  persistence nonnegative              : {persistence_nonneg}')
        if not persistence_nonneg:
            sys.exit('[error] Constraints NPZ contains negative persistence values.')

        print('  --- Constraints NPZ metadata validation: PASSED ---')

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
                # C-order VTK point-ID convention: iy = vid // PATCH, ix = vid % PATCH,
                # indexed directly into the full field (valid because the TTK VTI crop
                # is top-left-anchored, x0=y0=0).
                birth_yx[:n_use, 0] = bvid // PATCH
                birth_yx[:n_use, 1] = bvid % PATCH
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

    def _build_batch_ttk_feed(constraints: dict, batch_idx):
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

    def _gather_speed_at_yx(speed_field, yx):
        B = tf.shape(speed_field)[0]
        K = tf.shape(yx)[1]
        batch_idx = tf.reshape(tf.range(B), [B, 1, 1])
        batch_idx = tf.tile(batch_idx, [1, K, 1])
        gather_idx = tf.concat([batch_idx, yx], axis=2)
        return tf.gather_nd(speed_field, gather_idx)

    def _masked_mse(pred, target, mask):
        sq_err  = tf.square(pred - target) * mask
        n_valid = tf.maximum(tf.reduce_sum(mask), 1.0)
        return tf.reduce_sum(sq_err) / n_valid

    def build_ttk_loss_ops(model, mu_sig):
        mu_tf  = tf.constant(mu_sig[0], dtype=tf.float32)
        sig_tf = tf.constant(mu_sig[1], dtype=tf.float32)
        speed_sr = _denorm_speed(model.x_SR, mu_tf, sig_tf)

        ph = {
            'birth_yx':    tf.placeholder(tf.int32,   [None, MAX_PAIRS, 2], name='ttk_birth_yx'),
            'death_yx':    tf.placeholder(tf.int32,   [None, MAX_PAIRS, 2], name='ttk_death_yx'),
            'birth_val':   tf.placeholder(tf.float32, [None, MAX_PAIRS],    name='ttk_birth_val'),
            'death_val':   tf.placeholder(tf.float32, [None, MAX_PAIRS],    name='ttk_death_val'),
            'persistence': tf.placeholder(tf.float32, [None, MAX_PAIRS],    name='ttk_persistence'),
            'valid_mask':  tf.placeholder(tf.float32, [None, MAX_PAIRS],    name='ttk_valid_mask'),
        }

        sr_birth = _gather_speed_at_yx(speed_sr, ph['birth_yx'])
        sr_death = _gather_speed_at_yx(speed_sr, ph['death_yx'])

        L_ttkcv = 0.5 * (
            _masked_mse(sr_birth, ph['birth_val'], ph['valid_mask'])
            + _masked_mse(sr_death, ph['death_val'], ph['valid_mask'])
        )
        sr_persistence = tf.abs(sr_death - sr_birth)
        L_ttkpers = _masked_mse(sr_persistence, ph['persistence'], ph['valid_mask'])

        return ph, L_ttkcv, L_ttkpers

    # =====================================================================
    # Header
    # =====================================================================
    print('=' * 72)
    print(f'Candidate F expanded-2688 fine-tuning: variant={variant!r} '
          f'(method={paths["method_name"]!r})')
    print('=' * 72)
    print(f'  Training data     : {TRAIN_DATA_PATH}')
    print(f'  Evaluation data   : {EVAL_DATA_PATH}')
    print(f'  Source checkpoint : {MODEL_PATH}')
    print(f'  Output model dir  : {paths["model_dir"]}')
    print(f'  Inference output  : {paths["data_out_dir"]}')
    print(f'  Cheap-eval dir    : {paths["cheap_eval_dir"]}')
    print(f'  Topology VTI dir  : {paths["topology_vti_dir"]}')
    print(f'  Topology out dir  : {paths["topology_out_dir"]}')
    print(f'  Log file          : {paths["log_path"]}')
    print()
    print('  Loss configuration:')
    print('  lambda_uv       = 1.0  (always on; normalized [u, v])')
    print(f'  lambda_speed    = {cfg["lambda_speed"]}')
    print(f'  lambda_grad     = {cfg["lambda_grad"]}')
    print('  lambda_wpd      = 0.0')
    print(f'  lambda_levelset = {cfg["lambda_levelset"]}')
    print(f'  lambda_crit     = {cfg["lambda_crit"]}'
          + ('  (ENABLED)' if cfg['lambda_crit'] > 0 else '  (DISABLED)'))
    print(f'  lambda_TTKCV    = {cfg["lambda_ttkcv"]}'
          + ('  (repaired E2 ACTIVE)' if cfg['requires_e2'] else '  (repaired E2 NOT USED in this script)'))
    print(f'  lambda_TTKpers  = {cfg["lambda_ttkpers"]}'
          + ('  (repaired E2 ACTIVE)' if cfg['requires_e2'] else '  (repaired E2 NOT USED in this script)'))
    print(f'  requires_e2     = {cfg["requires_e2"]}')
    print(f'  constraints_npz = {constraints_npz if constraints_npz is not None else "NONE"}')
    print()
    print('  Hyperparameters:')
    print(f'  learning_rate   = {LEARNING_RATE}')
    print(f'  N_epochs        = {N_EPOCHS}')
    print(f'  save_every      = {SAVE_EVERY}')
    print(f'  print_every     = {PRINT_EVERY}')
    print(f'  batch_size      = {BATCH_SIZE}')
    print()
    print('  Normalisation (pretrained; unchanged):')
    print(f'  mu  = {MU_SIG[0]}')
    print(f'  sig = {MU_SIG[1]}')
    print('=' * 72)
    print()

    # =====================================================================
    # Safety + collision checks
    # =====================================================================
    _safety_check()
    _check_collisions()

    # =====================================================================
    # Preflight
    # =====================================================================
    print('=' * 72)
    print('Preflight checks')
    print('=' * 72)

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
    print(f'  Confirmed: {N_TRAIN}-sample training set matches candidateC_expanded2688 / '
          f'candidateB_expanded2688 / the completed E2-low-2688 scripts.')
    print(f'  Confirmed: evaluation benchmark is the fixed {N_EVAL}-sample benchmark '
          f'(never trained on).')

    constraints = None
    if cfg['requires_e2']:
        print()
        print(f'  Constraints NPZ    : {constraints_npz}')
        if not constraints_npz.exists():
            sys.exit(
                f'[error] Constraints NPZ not found: {constraints_npz}\n'
                '  This script does not regenerate constraint artifacts. If the listed path '
                'is wrong, search under ttk_runs_fixed/topology_finetuning/ for '
                'ttk_pd_critical_pairs_gtvalues.npz files associated with the completed '
                'native 2688-sample repaired-E2 runs before editing this script.'
            )
        constraints, sample_count, constraint_idx = _load_constraints_and_report(constraints_npz)

        print('  Reading training TFRecord sample indices …')
        train_indices = _read_tfrecord_indices(TRAIN_DATA_PATH)
        train_idx_set = set(train_indices.tolist())
        if len(train_indices.tolist()) != len(train_idx_set):
            sys.exit('[error] Training TFRecord contains duplicate index values.')
        constraint_idx_set = set(constraint_idx.tolist())
        # The training TFRecord's index set must equal the NPZ's index set
        # EXACTLY -- not merely be a subset of it. A superset on the NPZ
        # side (extra indices never trained on) is just as much a mismatch
        # as a missing index, so both directions are checked and reported.
        missing_from_npz = sorted(train_idx_set - constraint_idx_set)
        extra_in_npz = sorted(constraint_idx_set - train_idx_set)
        if missing_from_npz or extra_in_npz:
            sys.exit(
                '[error] Training TFRecord index set does not exactly equal the constraints '
                'NPZ index set.\n'
                f'  In TFRecord but not in NPZ (first 10 of {len(missing_from_npz)}): '
                f'{missing_from_npz[:10]}\n'
                f'  In NPZ but not in TFRecord (first 10 of {len(extra_in_npz)}): '
                f'{extra_in_npz[:10]}\n'
                '  Re-run the constraint builder for this exact dataset.'
            )
        print(f'  Training TFRecord index set == constraints NPZ index set EXACTLY '
              f'({N_TRAIN} indices, each maps to the correct row of the 2688-sample '
              'constraint file).')
        print('  Confirmed: lambda_crit = 0.0 for this variant (E2 present, L_crit absent).')
    else:
        print('  Confirmed: requires_e2=False -- no constraints NPZ will be loaded, no E2 '
              'constraint loader or E2 loss op will be built or executed for this variant.')
        print(f'  Confirmed: lambda_TTKCV=0.0, lambda_TTKpers=0.0, lambda_crit={cfg["lambda_crit"]}.')

    print('=' * 72)
    print()

    if args.dry_run:
        print('[dry-run] All preflight/collision/NPZ-validation checks passed. No training performed.')
        print(f'[dry-run] Would fine-tune variant={variant!r} for {N_EPOCHS} epochs on {N_TRAIN} samples,')
        print(f'[dry-run] then run paired inference on the {N_EVAL}-sample benchmark, writing to:')
        print(f'[dry-run]   {paths["model_dir"]}')
        print(f'[dry-run]   {paths["data_out_dir"]}')
        print('[dry-run] No canonical training log or any other file was created by this dry run.')
        return 0

    # =====================================================================
    # Every check has passed and this is a real (non-dry-run) invocation --
    # only now do we create the canonical training log. Everything printed
    # above (header, safety, collision, preflight, NPZ validation) went to
    # plain stdout only; tee the whole command yourself if you want that
    # captured too. Everything printed from here on is mirrored into the
    # canonical log.
    # =====================================================================
    sys.stdout = _Tee(sys.stdout, paths['log_path'])
    sys.stderr = sys.stdout
    print(f'[log] Canonical training log opened: {paths["log_path"]}')

    # =====================================================================
    # Training
    # =====================================================================
    phire = PhIREGANs(
        data_type    = f'wind_finetune_{paths["method_name"]}',
        learning_rate= LEARNING_RATE,
        N_epochs     = N_EPOCHS,
        save_every   = SAVE_EVERY,
        print_every  = PRINT_EVERY,
        mu_sig       = MU_SIG,
    )
    phire.setModel_name(paths['model_dir'])

    if cfg['requires_e2']:
        saved_model = _train_with_e2(
            phire, cfg, paths, constraints,
            tf=tf, np=np, PhysicsLossConfig=PhysicsLossConfig, SR_NETWORK=SR_NETWORK,
            build_ttk_loss_ops=build_ttk_loss_ops, _build_batch_ttk_feed=_build_batch_ttk_feed,
        )
    else:
        saved_model = _train_plain(phire, cfg, paths, PhysicsLossConfig=PhysicsLossConfig)

    print()
    print(f'Fine-tuning complete. Final checkpoint: {saved_model}')
    print()

    # =====================================================================
    # Paired inference
    # =====================================================================
    print('=' * 72)
    print('Phase 2: Paired inference on 168-sample benchmark')
    print('=' * 72)
    print(f'  Checkpoint : {saved_model}')
    print(f'  Eval data  : {EVAL_DATA_PATH}')
    print(f'  Output dir : {paths["data_out_dir"]}')
    print()

    phire.set_data_out_path(paths['data_out_dir'])
    phire.test_paired(
        r          = R,
        data_path  = EVAL_DATA_PATH,
        model_path = saved_model,
        batch_size = 1,
        save_inputs= True,
    )

    # =====================================================================
    # Post-inference validation
    # =====================================================================
    _validate_post_inference(paths, np=np)

    method_name = paths['method_name']
    print()
    print('=' * 72)
    print('Run complete.')
    print(f'  Variant          : {variant}')
    print(f'  Final checkpoint : {saved_model}')
    print(f'  SR outputs       : {paths["data_out_dir"]}/dataSR.npy')
    print(f'  GT outputs       : {paths["data_out_dir"]}/dataGT.npy')
    print(f'  LR inputs        : {paths["data_out_dir"]}/dataIN.npy')
    print(f'  Sample indices   : {paths["data_out_dir"]}/idx.npy')
    print()
    print('Next steps:')
    print('  Cheap scalar/domain evaluation (run this first):')
    print('    cd ~/PhIRE')
    print()
    print('    python3 scripts/evaluate_finetune_candidate.py \\')
    print(f'      --candidate-name {method_name} \\')
    print(f'      --candidate-dir  {paths["data_out_dir"]} \\')
    print('      --cnn-dir        data_out_fixed/wind_mrhr_cnn \\')
    print('      --gan-dir        data_out_fixed/wind_mrhr_gan \\')
    print('      --merged-csv     ttk_runs_fixed/combined/psnr_topology_physics_merged.csv \\')
    print(f'      --out-dir        {paths["cheap_eval_dir"]}')
    print()
    print('  Only run the expensive TTK topology pipeline AFTER confirming')
    print('  the cheap evaluation above is non-catastrophic:')
    print('    cd ~/PhIRE')
    print()
    print('    bash scripts/run_candidate_topology_pipeline.sh \\')
    print(f'      --method     {method_name} \\')
    print(f'      --data-dir   {paths["data_out_dir"]} \\')
    print(f'      --vti-dir    {paths["topology_vti_dir"]} \\')
    print(f'      --out-base   {paths["topology_out_dir"]} \\')
    print('      --n-samples  168 \\')
    print('      --threads    1 \\')
    print('      --skip-viz \\')
    print(f'      2>&1 | tee logs/topology_{method_name}.log')
    print('=' * 72)
    return 0


def _train_with_e2(phire, cfg, paths, constraints, *, tf, np, PhysicsLossConfig, SR_NETWORK,
                    build_ttk_loss_ops, _build_batch_ttk_feed):
    """Custom TF1 training loop with repaired E2 fixed-index TTK supervision,
    mirroring candidateB_plus_E2_tf_lowlambda_expanded2688/
    candidateUV_plus_E2_tf_lowlambda_expanded2688 exactly."""
    tf.reset_default_graph()
    phire.set_LR_data_shape(TRAIN_DATA_PATH)
    h, w, C = phire.LR_data_shape

    print('Initializing network …', end=' ')
    x_LR = tf.placeholder(tf.float32, [None, h,                w,               C])
    x_HR = tf.placeholder(tf.float32, [None, h * np.prod(R),   w * np.prod(R),  C])

    plc = PhysicsLossConfig(
        use_aux_losses       = True,
        mu                   = MU_SIG[0],
        sig                  = MU_SIG[1],
        lambda_speed         = cfg['lambda_speed'],
        lambda_grad          = cfg['lambda_grad'],
        lambda_wpd           = 0.0,
        lambda_levelset      = cfg['lambda_levelset'],
        levelset_temperature = 10.0,
        levelset_thresholds  = [5.0, 10.0, 15.0],
        lambda_crit          = cfg['lambda_crit'],
        crit_high_z          = 1.0,
        crit_include_minima  = False,
        crit_low_z           = -1.0,
        crit_pool            = 3,
        diagnostic_mode      = True,
    )
    model = SR_NETWORK(x_LR, x_HR, r=R, status='pretraining', phys_cfg=plc)

    ttk_ph, L_ttkcv, L_ttkpers = build_ttk_loss_ops(model, MU_SIG)
    total_loss = model.g_loss + cfg['lambda_ttkcv'] * L_ttkcv + cfg['lambda_ttkpers'] * L_ttkpers

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
                'L_speed (raw)': model.loss_breakdown['L_speed'],
                'L_grad (raw)': model.loss_breakdown['L_grad'],
                'L_levelset (raw)': model.loss_breakdown['L_levelset'],
                'L_crit (raw)': model.loss_breakdown['L_crit'],
                'L_ttkcv': L_ttkcv,
                'L_ttkpers': L_ttkpers,
                'g_loss': model.g_loss,
                'total_loss (+ TTK)': total_loss,
            },
            feed_dict=diag_feed,
        )
        print('  NOTE: raw values are unweighted; only terms with a nonzero lambda above '
              'actually contribute to total_loss.')
        for key, val in diag_vals.items():
            print(f'  {key:<28} {val:.6f}')
        print('=' * 64)
        print()

        print(f'Phase 1: Training on {N_TRAIN} expanded samples …')
        iters = 0
        first_batch = (diag_idx, diag_LR, diag_HR)

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
                        print('  Loss breakdown:')
                        for key in ('L_uv', 'L_speed', 'L_grad', 'L_wpd', 'L_levelset',
                                    'L_crit', 'L_ttkcv', 'L_ttkpers', 'total_loss_with_ttk'):
                            print(f'    {key + ":":<22} {bv[key]:.6f}')
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
    print('Fine-tuning complete (repaired E2 path).')
    return saved_model


def _train_plain(phire, cfg, paths, *, PhysicsLossConfig):
    """Plain PhIREGANs.pretrain() path for grad_crit -- no E2 constraints,
    no custom training loop, exactly like candidateB_expanded2688/
    candidateC_expanded2688."""
    plc = PhysicsLossConfig(
        use_aux_losses       = True,
        mu                   = MU_SIG[0],
        sig                  = MU_SIG[1],
        lambda_speed         = cfg['lambda_speed'],
        lambda_grad          = cfg['lambda_grad'],
        lambda_wpd           = 0.0,
        lambda_levelset      = cfg['lambda_levelset'],
        levelset_temperature = 10.0,
        levelset_thresholds  = [5.0, 10.0, 15.0],
        lambda_crit          = cfg['lambda_crit'],
        crit_high_z          = 1.0,
        crit_include_minima  = False,
        crit_low_z           = -1.0,
        crit_pool            = 3,
        diagnostic_mode      = True,
    )

    print(f'Phase 1: Fine-tuning on {N_TRAIN}-sample expanded dataset (plain pretrain() path) …')
    saved_model = phire.pretrain(
        r          = R,
        data_path  = TRAIN_DATA_PATH,
        model_path = MODEL_PATH,
        batch_size = BATCH_SIZE,
        phys_cfg   = plc,
    )
    print()
    print('Fine-tuning complete (plain pretrain() path; no E2 constraints were loaded).')
    return saved_model


def _validate_post_inference(paths, *, np) -> None:
    print()
    print('--- Post-inference validation ---')
    data_out_dir = paths['data_out_dir']
    expected = {
        'idx.npy':    (N_EVAL,),
        'dataIN.npy': (N_EVAL, 100, 100, 2),
        'dataGT.npy': (N_EVAL, 500, 500, 2),
        'dataSR.npy': (N_EVAL, 500, 500, 2),
    }
    all_ok = True
    arrays = {}
    for fname, exp_shape in expected.items():
        p = Path(data_out_dir) / fname
        if not p.exists():
            print(f'[MISSING] {p}')
            all_ok = False
            continue
        arr = np.load(str(p))
        arrays[fname] = arr
        shape_ok = arr.shape == exp_shape
        finite_ok = True
        if fname != 'idx.npy':
            finite_ok = bool(np.isfinite(arr).all())
        status = 'OK' if (shape_ok and finite_ok) else ('SHAPE-MISMATCH' if not shape_ok else 'NAN-OR-INF')
        print(f'[{status}] {fname}  shape={arr.shape}')
        if not shape_ok:
            print(f'         expected shape {exp_shape}')
            all_ok = False
        if not finite_ok:
            print(f'         contains NaN/inf values')
            all_ok = False

    if not all_ok:
        sys.exit('[error] Post-inference shape/finiteness validation failed. See above; '
                  'stopping before cheap evaluation.')

    idx_vals = arrays['idx.npy']
    if not np.array_equal(idx_vals, np.arange(N_EVAL)):
        sys.exit(
            f'[error] idx.npy is not exactly ordered 0..{N_EVAL - 1} '
            f'(got range [{int(idx_vals.min())}, {int(idx_vals.max())}], '
            f'{len(idx_vals)} entries). A sorted-set/uniqueness check is not sufficient here; '
            'stopping before cheap evaluation.'
        )
    print(f'[OK]   idx.npy values: exactly ordered 0..{N_EVAL - 1} (no duplicate/missing samples).')

    # ---------------------------------------------------------------
    # GT/IN alignment with the fixed CNN/GAN benchmark -- MANDATORY.
    # On Spark the fixed CNN benchmark arrays must exist; their absence is a
    # fatal validation error, not a warning (this repo's authoritative
    # benchmark is data_out_fixed/wind_mrhr_cnn/, produced once and never
    # regenerated by any candidate script).
    # ---------------------------------------------------------------
    cnn_gt_p  = Path(CNN_BASELINE_DIR) / 'dataGT.npy'
    cnn_in_p  = Path(CNN_BASELINE_DIR) / 'dataIN.npy'
    cnn_idx_p = Path(CNN_BASELINE_DIR) / 'idx.npy'
    missing_baseline = [str(p) for p in (cnn_idx_p, cnn_in_p, cnn_gt_p) if not p.exists()]
    if missing_baseline:
        sys.exit(
            '[error] Fixed CNN benchmark array(s) required for GT/IN alignment are missing:\n'
            + '\n'.join(f'  - {p}' for p in missing_baseline)
            + '\n  This is a fatal validation error, not a warning -- stopping before cheap '
              'evaluation. GT/IN alignment against the fixed benchmark cannot be skipped.'
        )

    cnn_idx = np.load(cnn_idx_p).astype(int).ravel()
    if not np.array_equal(cnn_idx, np.arange(N_EVAL)):
        sys.exit(
            f'[error] Fixed CNN baseline idx.npy ({cnn_idx_p}) is not exactly ordered '
            f'0..{N_EVAL - 1} (got range [{int(cnn_idx.min())}, {int(cnn_idx.max())}], '
            f'{len(cnn_idx)} entries). Stopping before cheap evaluation.'
        )
    print(f'[OK]   fixed CNN baseline idx.npy: exactly ordered 0..{N_EVAL - 1}.')

    cnn_gt = np.load(cnn_gt_p)
    cnn_in = np.load(cnn_in_p)
    # idx.npy is exactly ordered 0..N_EVAL-1 for both the candidate and the
    # CNN baseline (just confirmed above), so position i in both arrays
    # already refers to the same sample_idx == i; no idx-based reordering
    # is needed, but we still assert the shapes agree defensively.
    if cnn_gt.shape != arrays['dataGT.npy'].shape or cnn_in.shape != arrays['dataIN.npy'].shape:
        sys.exit(
            f'[error] Fixed CNN baseline array shape(s) do not match this candidate\'s output '
            f'(cnn dataGT.npy={cnn_gt.shape} vs candidate={arrays["dataGT.npy"].shape}; '
            f'cnn dataIN.npy={cnn_in.shape} vs candidate={arrays["dataIN.npy"].shape}). '
            'Stopping before cheap evaluation.'
        )

    gt_exact = bool(np.array_equal(cnn_gt, arrays['dataGT.npy']))
    in_exact = bool(np.array_equal(cnn_in, arrays['dataIN.npy']))
    gt_diff = float(np.abs(cnn_gt.astype(np.float64) - arrays['dataGT.npy'].astype(np.float64)).max())
    in_diff = float(np.abs(cnn_in.astype(np.float64) - arrays['dataIN.npy'].astype(np.float64)).max())
    print(f'[{"OK" if gt_exact else "FAIL"}] dataGT.npy exactly aligned with fixed CNN/GAN '
          f'benchmark GT (np.array_equal={gt_exact}, max_abs_diff={gt_diff:.4e} for diagnostics).')
    print(f'[{"OK" if in_exact else "FAIL"}] dataIN.npy exactly aligned with fixed benchmark '
          f'input (np.array_equal={in_exact}, max_abs_diff={in_diff:.4e} for diagnostics).')
    # No repo script establishes 1e-3 (or any other tolerance) as an
    # authoritative "aligned" criterion -- scripts/evaluate_finetune_candidate.py
    # only ever uses 1e-3 as a non-fatal warning threshold. Both dataGT.npy
    # and dataIN.npy are deterministic decodes of the same fixed evaluation
    # TFRecord, so exact equality is the correct bar here; any difference is
    # fatal.
    if not (gt_exact and in_exact):
        sys.exit('[error] GT/IN alignment against the fixed CNN/GAN benchmark is not exact '
                  '(see max_abs_diff above). Stopping before cheap evaluation.')

    # ---------------------------------------------------------------
    # Vector/scalar-speed range + mean abs error reporting
    # ---------------------------------------------------------------
    gt = arrays['dataGT.npy'].astype(np.float64)
    sr = arrays['dataSR.npy'].astype(np.float64)
    print(f'[report] dataGT.npy [u,v] range : u=[{gt[...,0].min():.4f}, {gt[...,0].max():.4f}] '
          f'v=[{gt[...,1].min():.4f}, {gt[...,1].max():.4f}]')
    print(f'[report] dataSR.npy [u,v] range : u=[{sr[...,0].min():.4f}, {sr[...,0].max():.4f}] '
          f'v=[{sr[...,1].min():.4f}, {sr[...,1].max():.4f}]')

    mu_u, mu_v = MU_SIG[0]
    sig_u, sig_v = MU_SIG[1]
    # dataGT.npy/dataSR.npy from PhIREGANs.test_paired() are already
    # denormalized physical [u,v] (test_paired's output convention, unlike
    # the normalized [u,v] used internally for L_uv) -- compute physical
    # scalar speed directly, matching every existing evaluate_finetune_candidate.py
    # convention in this repo.
    speed_gt = np.sqrt(gt[..., 0] ** 2 + gt[..., 1] ** 2)
    speed_sr = np.sqrt(sr[..., 0] ** 2 + sr[..., 1] ** 2)
    print(f'[report] GT scalar-speed range : [{speed_gt.min():.4f}, {speed_gt.max():.4f}] m/s')
    print(f'[report] SR scalar-speed range : [{speed_sr.min():.4f}, {speed_sr.max():.4f}] m/s')
    mae = float(np.abs(speed_sr - speed_gt).mean())
    print(f'[report] Mean absolute scalar-speed error (SR vs GT): {mae:.4f} m/s')

    # Non-catastrophic plausibility report only -- never fatal. An experimental
    # fine-tuned model slightly exceeding the fixed GT maximum is expected and
    # must not abort the run; this is a prompt for manual judgement before the
    # expensive TTK stage, not a pass/fail gate.
    gt_max = float(speed_gt.max())
    sr_max = float(speed_sr.max())
    sr_gt_max_ratio = (sr_max / gt_max) if gt_max > 0 else float('inf')

    plausible_max = 100.0        # generous absolute bound; established CNN/GAN speeds are well under this
    relative_ratio_bound = 1.25  # SR max materially above GT max: >25% over is worth a manual look

    absolute_flag = gt_max > plausible_max or sr_max > plausible_max or speed_sr.min() < 0
    relative_flag = sr_gt_max_ratio > relative_ratio_bound
    if absolute_flag or relative_flag:
        print('=' * 72)
        print('[WARNING] Scalar-speed range looks implausible for physical wind speed, or the '
              'SR maximum is materially above the fixed GT maximum. This is NOT automatically '
              'fatal -- an experimental model slightly exceeding the GT maximum is expected -- '
              'but do not proceed to the expensive TTK stage until this has been manually reviewed.')
        print(f'[WARNING]   GT maximum speed      : {gt_max:.4f} m/s')
        print(f'[WARNING]   SR maximum speed      : {sr_max:.4f} m/s')
        print(f'[WARNING]   SR/GT maximum ratio   : {sr_gt_max_ratio:.4f}')
        print(f'[WARNING]   Speed MAE (SR vs GT)  : {mae:.4f} m/s')
        print('=' * 72)

    print('[validate] Post-inference validation complete: shapes, exact idx order, '
          'finiteness, and (where the baseline was available) GT/IN alignment all passed.')


if __name__ == '__main__':
    sys.exit(main())
