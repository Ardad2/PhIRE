#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Batch driver for the Candidate B factorial ablation at the 2688-sample
# scale (scripts/run_candidateB_factorial_expanded2688_finetune.py).
#
# For each selected variant this script:
#   1. Runs scripts/run_candidateB_factorial_expanded2688_finetune.py
#      --variant <name>, which trains, runs paired inference on the fixed
#      168-sample benchmark, and validates the output arrays (idx == 0..167,
#      shapes, no NaN/Inf).
#   2. If training succeeded and --skip-eval was not passed, runs the cheap
#      scalar/domain evaluation (scripts/evaluate_finetune_candidate.py)
#      automatically, comparing the variant against CNN and GAN.
#   3. Prints (but never runs) the exact TTK topology pipeline command for
#      that variant -- the expensive TTK stage is always manual, and should
#      only be run after confirming the cheap evaluation above is
#      non-catastrophic.
#
# A variant's failure (training script exits non-zero, e.g. because its
# output directory already exists and --overwrite/--resume was not passed)
# is reported and does NOT abort the batch -- remaining variants still run,
# since each variant is an independent training job. A final PASS/FAIL
# summary is printed at the end and the script exits non-zero if any
# variant failed.
#
# Usage (from repo root, on Spark):
#   bash scripts/run_candidateB_factorial_expanded2688_batch.sh --variant levelset
#   bash scripts/run_candidateB_factorial_expanded2688_batch.sh --all-singletons
#   bash scripts/run_candidateB_factorial_expanded2688_batch.sh --all-pairs
#   bash scripts/run_candidateB_factorial_expanded2688_batch.sh --all
#
# Options:
#   --variant NAME   Run exactly one variant (speed|grad|levelset|
#                    speed_grad|speed_levelset|grad_levelset).
#   --all-singletons Run levelset, speed, grad (in that preferred order).
#   --all-pairs      Run speed_levelset, grad_levelset, speed_grad
#                    (in that preferred order).
#   --all            Run all six variants, in the full preferred order:
#                    levelset, speed, grad, speed_levelset, grad_levelset,
#                    speed_grad.
#   --overwrite      Forwarded to the training script: allow proceeding even
#                    if a variant's output directory already exists.
#   --resume         Alias for --overwrite (identical effect; forwarded).
#   --dry-run        Forwarded to the training script (preflight checks +
#                    plan only, no training); the cheap-evaluation step is
#                    also skipped in this mode.
#   --skip-eval      Do not automatically run the cheap scalar/domain
#                    evaluation after a variant's training completes.
#   -h, --help       Show this help.
#
# Preferred run order rationale: the soft high-speed level-set term is the
# most interesting hypothesis for Candidate B's PD improvement, so
# `levelset` runs first.
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

TRAIN_SCRIPT="scripts/run_candidateB_factorial_expanded2688_finetune.py"
EVAL_SCRIPT="scripts/evaluate_finetune_candidate.py"

# Full preferred order (see module docstring above and the training script).
PREFERRED_ORDER=(levelset speed grad speed_levelset grad_levelset speed_grad)
SINGLETONS=(levelset speed grad)
PAIRS=(speed_levelset grad_levelset speed_grad)

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
usage() {
  sed -n '/^# Usage (from repo root/,/^# Preferred run order/p' "$0" | grep '^#' | sed 's/^# \?//'
}

VARIANT=""
MODE=""   # "single" | "singletons" | "pairs" | "all"
OVERWRITE=0
DRY_RUN=0
SKIP_EVAL=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --variant)        VARIANT="$2"; MODE="single"; shift 2 ;;
    --all-singletons) MODE="singletons"; shift ;;
    --all-pairs)      MODE="pairs"; shift ;;
    --all)            MODE="all"; shift ;;
    --overwrite)      OVERWRITE=1; shift ;;
    --resume)         OVERWRITE=1; shift ;;
    --dry-run)        DRY_RUN=1; shift ;;
    --skip-eval)      SKIP_EVAL=1; shift ;;
    -h|--help)        usage; exit 0 ;;
    *) echo "[error] Unknown argument: $1"; usage; exit 2 ;;
  esac
done

if [[ -z "$MODE" ]]; then
  echo "[error] Exactly one of --variant, --all-singletons, --all-pairs, --all is required."
  usage
  exit 2
fi

VALID_VARIANTS=(speed grad levelset speed_grad speed_levelset grad_levelset)
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
  singletons) VARIANTS_TO_RUN=("${SINGLETONS[@]}") ;;
  pairs)      VARIANTS_TO_RUN=("${PAIRS[@]}") ;;
  all)        VARIANTS_TO_RUN=("${PREFERRED_ORDER[@]}") ;;
esac

echo "========================================"
echo "Candidate B factorial ablation batch (2688-sample scale)"
echo "========================================"
echo "  Mode              : $MODE"
echo "  Variants to run    : ${VARIANTS_TO_RUN[*]}"
echo "  Overwrite/resume   : $OVERWRITE"
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
  [[ "$OVERWRITE" -eq 1 ]] && TRAIN_ARGS+=(--overwrite)
  [[ "$DRY_RUN" -eq 1 ]] && TRAIN_ARGS+=(--dry-run)

  set +e
  python3 "$TRAIN_SCRIPT" "${TRAIN_ARGS[@]}"
  train_rc=$?
  set -e

  if [[ "$train_rc" -ne 0 ]]; then
    echo "[error] Variant $variant: training script exited with code $train_rc. Skipping evaluation/TTK for this variant."
    RESULTS+=("$variant:FAIL(train_rc=$train_rc)")
    echo
    continue
  fi

  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[dry-run] Variant $variant: preflight OK, no training performed."
    RESULTS+=("$variant:DRY-RUN-OK")
    echo
    continue
  fi

  METHOD="candidateB_factorial_${variant}_expanded2688"
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
  echo "      --skip-viz"
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
