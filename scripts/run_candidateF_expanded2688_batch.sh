#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Batch driver for Candidate F (scripts/run_candidateF_expanded2688_finetune.py):
# three native PhIRE/TensorFlow fine-tuning variants at the 2688-sample scale
# that recombine L_grad (the PD-driving term) with repaired low-lambda E2
# TTK fixed-index supervision (grad_e2_low, grad_levelset_e2_low) or with
# Candidate C's local-maxima proxy directly (grad_crit, no E2).
#
# For each selected variant this script:
#   1. Runs scripts/run_candidateF_expanded2688_finetune.py --variant <name>,
#      which validates the constraints NPZ (for the two E2 variants), trains,
#      runs paired inference on the fixed 168-sample benchmark, and validates
#      the output arrays (idx == exactly ordered 0..167, shapes, finiteness,
#      GT/IN alignment against the fixed CNN/GAN benchmark, scalar-speed
#      ranges + mean absolute error).
#   2. If training + validation succeeded and --skip-eval was not passed,
#      runs the cheap scalar/domain evaluation
#      (scripts/evaluate_finetune_candidate.py) automatically.
#   3. Prints (but never runs) the exact TTK topology pipeline command for
#      that variant -- the expensive TTK stage is always manual, and should
#      only be run after confirming the cheap evaluation above is
#      non-catastrophic.
#
# A variant's failure (training script exits non-zero -- e.g. its output
# already exists, or post-inference validation failed) is reported and does
# NOT abort the batch; remaining variants still run, since each is an
# independent training job. A final PASS/FAIL summary is printed at the end
# and the script exits non-zero if any variant failed.
#
# There is deliberately no --overwrite/--resume option here or in the
# training script: PhIREGANs.pretrain() always retrains from the pretrained
# CNN checkpoint from scratch in this codebase, so an overwrite flag would
# silently retrain under the same method name while potentially leaving
# stale cheap-evaluation or TTK outputs from an older model version behind.
# If a prior run for a variant needs to be redone, inspect its artifacts
# first, then deliberately archive or remove them yourself (nothing here
# ever deletes anything automatically) before rerunning.
#
# Usage (from repo root, on Spark):
#   bash scripts/run_candidateF_expanded2688_batch.sh --variant grad_e2_low
#   bash scripts/run_candidateF_expanded2688_batch.sh --all
#
# Options:
#   --variant NAME   Run exactly one variant (grad_e2_low|
#                    grad_levelset_e2_low|grad_crit).
#   --all            Run all three variants, in the recommended order:
#                    grad_e2_low, grad_levelset_e2_low, grad_crit (the
#                    cleanest direct PD+MT combination first, then the
#                    level-set interaction test, then the local-maxima-only
#                    control).
#   --print-config   Forwarded to the training script: print the static
#                    configuration table for all three variants and exit
#                    (ignores --variant/--all; runs once, not once per
#                    variant).
#   --dry-run        Forwarded to the training script (preflight + collision
#                    + NPZ-validation checks + plan only, no training); the
#                    cheap-evaluation step is also skipped in this mode.
#   --skip-eval      Do not automatically run the cheap scalar/domain
#                    evaluation after a variant's training completes.
#   -h, --help       Show this help.
#
# Recommended run order rationale: grad_e2_low is the cleanest direct
# combination of the established PD (L_grad) and MT (repaired E2) signals;
# grad_levelset_e2_low tests whether adding L_levelset back helps or hurts;
# grad_crit isolates whether L_crit adds anything beyond L_grad alone.
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

TRAIN_SCRIPT="scripts/run_candidateF_expanded2688_finetune.py"
EVAL_SCRIPT="scripts/evaluate_finetune_candidate.py"

# Recommended run order (see module docstring above and the training script).
PREFERRED_ORDER=(grad_e2_low grad_levelset_e2_low grad_crit)

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
usage() {
  sed -n '/^# Usage (from repo root/,/^# Recommended run order/p' "$0" | grep '^#' | sed 's/^# \?//'
}

VARIANT=""
MODE=""   # "single" | "all" | "print-config"
DRY_RUN=0
SKIP_EVAL=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --variant)       VARIANT="$2"; MODE="single"; shift 2 ;;
    --all)           MODE="all"; shift ;;
    --print-config)  MODE="print-config"; shift ;;
    --dry-run)       DRY_RUN=1; shift ;;
    --skip-eval)     SKIP_EVAL=1; shift ;;
    -h|--help)       usage; exit 0 ;;
    *) echo "[error] Unknown argument: $1"; usage; exit 2 ;;
  esac
done

if [[ "$MODE" == "print-config" ]]; then
  python3 "$TRAIN_SCRIPT" --print-config
  exit $?
fi

if [[ -z "$MODE" ]]; then
  echo "[error] Exactly one of --variant, --all, --print-config is required."
  usage
  exit 2
fi

VALID_VARIANTS=(grad_e2_low grad_levelset_e2_low grad_crit)
is_valid_variant() {
  local v="$1"
  for candidate in "${VALID_VARIANTS[@]}"; do
    [[ "$candidate" == "$v" ]] && return 0
  done
  return 1
}

case "$MODE" in
  single)
    if [[ -z "$VARIANT" ]]; then
      echo "[error] --variant requires a name."
      exit 2
    fi
    if ! is_valid_variant "$VARIANT"; then
      echo "[error] Unknown variant: $VARIANT. Valid: ${VALID_VARIANTS[*]}"
      exit 2
    fi
    VARIANTS_TO_RUN=("$VARIANT")
    ;;
  all) VARIANTS_TO_RUN=("${PREFERRED_ORDER[@]}") ;;
esac

echo "========================================"
echo "Candidate F expanded-2688 batch"
echo "========================================"
echo "  Mode              : $MODE"
echo "  Variants to run    : ${VARIANTS_TO_RUN[*]}"
echo "  Dry run             : $DRY_RUN"
echo "  Auto cheap-eval     : $([[ "$SKIP_EVAL" -eq 1 || "$DRY_RUN" -eq 1 ]] && echo no || echo yes)"
echo "========================================"
echo

RESULTS=()

for variant in "${VARIANTS_TO_RUN[@]}"; do
  echo "########################################"
  echo "# Variant: $variant"
  echo "########################################"

  TRAIN_ARGS=(--variant "$variant")
  [[ "$DRY_RUN" -eq 1 ]] && TRAIN_ARGS+=(--dry-run)

  set +e
  python3 "$TRAIN_SCRIPT" "${TRAIN_ARGS[@]}"
  train_rc=$?
  set -e

  if [[ "$train_rc" -ne 0 ]]; then
    echo "[error] Variant $variant: training script exited with code $train_rc. Skipping evaluation/TTK for this variant."
    echo "[error] If this was a collision (artifacts already exist for this method), inspect them and"
    echo "[error] deliberately archive or remove ALL of them yourself before rerunning -- nothing is"
    echo "[error] deleted automatically, and there is no --overwrite/--resume flag to bypass this."
    RESULTS+=("$variant:FAIL(train_rc=$train_rc)")
    echo
    continue
  fi

  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[dry-run] Variant $variant: preflight/collision/NPZ checks OK, no training performed."
    RESULTS+=("$variant:DRY-RUN-OK")
    echo
    continue
  fi

  case "$variant" in
    grad_e2_low)          METHOD="candidateF_grad_E2_low_expanded2688" ;;
    grad_levelset_e2_low)  METHOD="candidateF_grad_levelset_E2_low_expanded2688" ;;
    grad_crit)              METHOD="candidateF_grad_crit_expanded2688" ;;
  esac
  DATA_OUT="data_out/wind_finetune_${METHOD}"

  if [[ "$SKIP_EVAL" -eq 0 ]]; then
    echo "--- Variant $variant: cheap scalar/domain evaluation ---"
    set +e
    python3 "$EVAL_SCRIPT" \
      --candidate-name "$METHOD" \
      --candidate-dir  "$DATA_OUT" \
      --cnn-dir        data_out_fixed/wind_mrhr_cnn \
      --gan-dir        data_out_fixed/wind_mrhr_gan \
      --merged-csv     ttk_runs_fixed/combined/psnr_topology_physics_merged.csv \
      --out-dir        "ttk_runs_fixed/topology_finetuning/${METHOD}_eval"
    eval_rc=$?
    set -e
    if [[ "$eval_rc" -ne 0 ]]; then
      echo "[warn] Variant $variant: cheap evaluation exited with code $eval_rc."
      RESULTS+=("$variant:TRAIN-OK,EVAL-FAIL(rc=$eval_rc)")
    else
      RESULTS+=("$variant:TRAIN-OK,EVAL-OK")
    fi
  else
    echo "[skip-eval] Variant $variant: cheap evaluation skipped (--skip-eval)."
    RESULTS+=("$variant:TRAIN-OK,EVAL-SKIPPED")
  fi

  echo
  echo "--- Variant $variant: TTK topology command (NOT run automatically) ---"
  echo "  Only run this after confirming the cheap evaluation above is non-catastrophic:"
  echo "    bash scripts/run_candidate_topology_pipeline.sh \\"
  echo "      --method     ${METHOD} \\"
  echo "      --data-dir   ${DATA_OUT} \\"
  echo "      --vti-dir    ttk_runs_fixed/topology_finetuning/${METHOD}_topology_vti \\"
  echo "      --out-base   ttk_runs_fixed/topology_finetuning/${METHOD}_topology \\"
  echo "      --n-samples  168 \\"
  echo "      --threads    1 \\"
  echo "      --skip-viz \\"
  echo "      2>&1 | tee logs/topology_${METHOD}.log"
  echo
done

echo "========================================"
echo "Batch complete."
echo "========================================"
overall_rc=0
for r in "${RESULTS[@]}"; do
  echo "  $r"
  [[ "$r" == *FAIL* ]] && overall_rc=1
done
echo "========================================"

exit "$overall_rc"
