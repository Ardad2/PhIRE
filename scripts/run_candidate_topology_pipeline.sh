#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Generic topology pipeline for any fine-tuned CNN candidate.
#
# Uses the SAME Docker image, TTK settings, patch size (160), and scalar
# field ("speed") as the fixed CNN/GAN baseline topology pipeline, so
# candidate distances are directly comparable.
#
# Pipeline stages:
#   1. VTI generation  (Docker: convert_phire_to_vti.py)
#   2. TTK PD/MT extraction per VTI file  (Docker: ttkPersistenceDiagramCmd /
#      ttkMergeTreeCmd)
#   3. Distance computation  (native python3: compute_composite_tree_distance.py)
#   4. Optional visualization  (native python3: phase_c_final_visualization.py)
#   5. Topology comparison report  (native python3: build_candidate_topology_comparison.py)
#
# Prerequisites (run from repo root on Spark):
#   - <data-dir>/{dataSR,dataGT,idx}.npy
#   - Docker image phire-ttk:latest
#   - native python3 with vtk + topologytoolkit (for steps 3–5)
#
# Usage:
#   bash scripts/run_candidate_topology_pipeline.sh [options]
#
# Required:
#   --method NAME      Candidate identifier used in VTI labels (e.g. candidateC)
#   --data-dir DIR     Directory with dataSR.npy, dataGT.npy, idx.npy
#
# Optional:
#   --vti-dir DIR      Output VTI directory
#                      (default: ttk_runs_fixed/topology_finetuning/<method>_vti)
#   --out-base DIR     Output topology directory
#                      (default: ttk_runs_fixed/topology_finetuning/<method>_topology)
#   --n-samples N      Number of samples to process (default: 168, indices 0..N-1)
#   --scalar NAME      Scalar field name for convert_phire_to_vti.py (default: speed)
#   --patch N          Patch size (default: 160)
#   --x0 N             Patch x offset (default: 0)
#   --y0 N             Patch y offset (default: 0)
#   --threads N        TTK thread count per Docker call (default: 20)
#   --docker-image IMG Docker image name (default: phire-ttk:latest)
#   --skip-viz         Skip phase_c_final_visualization.py
#   --debug            Pass --debug to compute_composite_tree_distance.py
#   -h, --help         Show this help
#
# Outputs:
#   <vti-dir>/         VTI files
#   <out-base>/        TTK outputs + CSV
# ---------------------------------------------------------------------------
set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults (matching the fixed CNN/GAN baseline exactly)
# ---------------------------------------------------------------------------
DOCKER_IMAGE="phire-ttk:latest"
N_SAMPLES=168
SCALAR="speed"
PATCH=160
X0=0
Y0=0
THREADS=20
SKIP_VIZ=0
DEBUG=0

WORKDIR="$(pwd)"

# Required args (no defaults — must be supplied)
METHOD=""
DATA_DIR=""

# Optional args with method-derived defaults (set after arg parsing)
VTI_DIR=""
OUT_BASE=""

# Baseline and Candidate B outputs — must never be modified.
_PROTECTED_DIRS=(
  "ttk_runs_fixed/cnn"
  "ttk_runs_fixed/gan"
  "ttk_runs_fixed/combined"
  "ttk_runs_fixed/topology_finetuning/candidateB_vti"
  "ttk_runs_fixed/topology_finetuning/candidateB_topology"
)

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
usage() {
  sed -n '/^# Usage:/,/^# Outputs:/p' "$0" | grep '^#' | sed 's/^# \?//'
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --method)       METHOD="$2";       shift 2 ;;
    --data-dir)     DATA_DIR="$2";     shift 2 ;;
    --vti-dir)      VTI_DIR="$2";      shift 2 ;;
    --out-base)     OUT_BASE="$2";     shift 2 ;;
    --n-samples)    N_SAMPLES="$2";    shift 2 ;;
    --scalar)       SCALAR="$2";       shift 2 ;;
    --patch)        PATCH="$2";        shift 2 ;;
    --x0)           X0="$2";           shift 2 ;;
    --y0)           Y0="$2";           shift 2 ;;
    --threads)      THREADS="$2";      shift 2 ;;
    --docker-image) DOCKER_IMAGE="$2"; shift 2 ;;
    --skip-viz)     SKIP_VIZ=1;        shift   ;;
    --debug)        DEBUG=1;           shift   ;;
    -h|--help)      usage; exit 0      ;;
    *) echo "[error] Unknown argument: $1"; usage; exit 2 ;;
  esac
done

# ---------------------------------------------------------------------------
# Validate required args
# ---------------------------------------------------------------------------
if [[ -z "$METHOD" ]]; then
  echo "[error] --method is required (e.g. --method candidateC)"
  usage; exit 2
fi
if [[ -z "$DATA_DIR" ]]; then
  echo "[error] --data-dir is required (e.g. --data-dir data_out/wind_finetune_pilot_candidateC)"
  usage; exit 2
fi

# Derive defaults from METHOD if not supplied.
if [[ -z "$VTI_DIR" ]]; then
  VTI_DIR="ttk_runs_fixed/topology_finetuning/${METHOD}_vti"
fi
if [[ -z "$OUT_BASE" ]]; then
  OUT_BASE="ttk_runs_fixed/topology_finetuning/${METHOD}_topology"
fi

# ---------------------------------------------------------------------------
# Safety check: refuse if OUT_BASE or VTI_DIR would touch protected dirs
# ---------------------------------------------------------------------------
for p in "${_PROTECTED_DIRS[@]}"; do
  for target in "$OUT_BASE" "$VTI_DIR"; do
    if [[ "$(realpath -m "$target")" == "$(realpath -m "$p")"* ]]; then
      echo "[error] $target would write inside protected $p. Aborting."
      exit 1
    fi
  done
done

# ---------------------------------------------------------------------------
# Pre-flight: verify inputs exist
# ---------------------------------------------------------------------------
for f in "$DATA_DIR/dataGT.npy" "$DATA_DIR/dataSR.npy"; do
  if [[ ! -f "$f" ]]; then
    echo "[error] Required input not found: $f"
    echo "        Run the fine-tuning pilot script first."
    exit 1
  fi
done

# Build sample index list 0 .. N_SAMPLES-1  (space-separated to avoid newline bug)
SAMPLE_LIST=$(seq -s ' ' 0 $((N_SAMPLES - 1)))
N_LAST=$((N_SAMPLES - 1))

echo "========================================"
echo "Candidate topology pipeline: $METHOD"
echo "========================================"
echo "  data_dir     : $DATA_DIR"
echo "  n_samples    : $N_SAMPLES  (indices 0–$N_LAST)"
echo "  scalar       : $SCALAR"
echo "  patch        : ${PATCH}×${PATCH} at (x0=$X0, y0=$Y0)"
echo "  threads      : $THREADS"
echo "  docker_image : $DOCKER_IMAGE"
echo "  vti_dir      : $VTI_DIR"
echo "  out_base     : $OUT_BASE"
echo "========================================"

mkdir -p "$VTI_DIR" "$OUT_BASE"

# ---------------------------------------------------------------------------
# Stage 1: VTI generation
# ---------------------------------------------------------------------------
echo ""
echo "--- Stage 1: VTI generation ---"

docker run --rm -v "$WORKDIR":/work -w /work "$DOCKER_IMAGE" \
  bash -lc "python scripts/convert_phire_to_vti.py \
    --input $DATA_DIR/dataGT.npy \
    --outdir $VTI_DIR \
    --label ${METHOD}_GT \
    --scalar $SCALAR \
    --patch $PATCH --x0 $X0 --y0 $Y0 \
    --samples $SAMPLE_LIST"

echo "[stage1] GT VTI generation complete."

docker run --rm -v "$WORKDIR":/work -w /work "$DOCKER_IMAGE" \
  bash -lc "python scripts/convert_phire_to_vti.py \
    --input $DATA_DIR/dataSR.npy \
    --outdir $VTI_DIR \
    --label ${METHOD}_SR \
    --scalar $SCALAR \
    --patch $PATCH --x0 $X0 --y0 $Y0 \
    --samples $SAMPLE_LIST"

echo "[stage1] SR VTI generation complete."

GT_VTI_COUNT=$(find "$VTI_DIR" -name "${METHOD}_GT_*.vti" | wc -l)
SR_VTI_COUNT=$(find "$VTI_DIR" -name "${METHOD}_SR_*.vti" | wc -l)
echo "[stage1] VTI counts: GT=$GT_VTI_COUNT  SR=$SR_VTI_COUNT  (expected $N_SAMPLES each)"

# ---------------------------------------------------------------------------
# Stage 2: TTK PD and MT extraction
# convert_phire_to_vti.py writes --scalar speed as VTK array "wind_speed",
# so TTK must use -a wind_speed regardless of the --scalar argument.
# ---------------------------------------------------------------------------
echo ""
echo "--- Stage 2: TTK PD and MT extraction ---"

PD_DIR="$OUT_BASE/pd"
MT_DIR="$OUT_BASE/mt"
mkdir -p "$PD_DIR" "$MT_DIR"

shopt -s nullglob
VTI_FILES=("$VTI_DIR"/${METHOD}_*.vti)
shopt -u nullglob

if [[ ${#VTI_FILES[@]} -eq 0 ]]; then
  echo "[error] No VTI files found in $VTI_DIR matching ${METHOD}_*.vti"
  exit 1
fi

echo "  Processing ${#VTI_FILES[@]} VTI files …"

for f in "${VTI_FILES[@]}"; do
  base="$(basename "$f" .vti)"
  echo "  === PD: $base ==="
  docker run --rm -v "$WORKDIR":/work -w /work "$DOCKER_IMAGE" \
    bash -lc "ttkPersistenceDiagramCmd -t $THREADS \
      -i $VTI_DIR/${base}.vti \
      -a wind_speed \
      -o $PD_DIR/${base}_pd"

  echo "  === MT: $base ==="
  docker run --rm -v "$WORKDIR":/work -w /work "$DOCKER_IMAGE" \
    bash -lc "ttkMergeTreeCmd -t $THREADS \
      -i $VTI_DIR/${base}.vti \
      -a wind_speed \
      -o $MT_DIR/${base}_mt"
done

# Organise into GT/ and SR/ subdirectories (mirrors baseline layout)
mkdir -p "$PD_DIR/GT" "$PD_DIR/SR" "$MT_DIR/GT" "$MT_DIR/SR"

shopt -s nullglob
PD_GT=("$PD_DIR"/${METHOD}_GT_*)
PD_SR=("$PD_DIR"/${METHOD}_SR_*)
MT_GT=("$MT_DIR"/${METHOD}_GT_*)
MT_SR=("$MT_DIR"/${METHOD}_SR_*)
shopt -u nullglob

[[ ${#PD_GT[@]} -gt 0 ]] && command mv -f -- "${PD_GT[@]}" "$PD_DIR/GT/"
[[ ${#PD_SR[@]} -gt 0 ]] && command mv -f -- "${PD_SR[@]}" "$PD_DIR/SR/"
[[ ${#MT_GT[@]} -gt 0 ]] && command mv -f -- "${MT_GT[@]}" "$MT_DIR/GT/"
[[ ${#MT_SR[@]} -gt 0 ]] && command mv -f -- "${MT_SR[@]}" "$MT_DIR/SR/"

echo "[stage2] PD files: $(find "$PD_DIR" -name '*.vtu' | wc -l)"
echo "[stage2] MT port0: $(find "$MT_DIR" -name '*_port_0.vtu' | wc -l)"
echo "[stage2] MT port1: $(find "$MT_DIR" -name '*_port_1.vtu' | wc -l)"
echo "[stage2] MT port2: $(find "$MT_DIR" -name '*_port_2.vti'  | wc -l)"

# ---------------------------------------------------------------------------
# Stage 3: Distance computation (native python3 — same as baseline pipeline)
# ---------------------------------------------------------------------------
echo ""
echo "--- Stage 3: Distance computation ---"

FINAL_DIR="$OUT_BASE/phase_c_final"
DEBUG_FLAG=""
[[ "$DEBUG" -eq 1 ]] && DEBUG_FLAG="--debug"

python3 -X faulthandler scripts/compute_composite_tree_distance.py \
  --pd-dir "$PD_DIR" \
  --mt-dir "$MT_DIR" \
  --outdir "$FINAL_DIR" \
  --isolate-mt \
  $DEBUG_FLAG

echo "[stage3] Distance computation complete."
echo "[stage3] Results: $FINAL_DIR/phase_c_results.csv"

# ---------------------------------------------------------------------------
# Stage 4: Optional visualization
# ---------------------------------------------------------------------------
if [[ "$SKIP_VIZ" -eq 0 ]] && [[ -f "scripts/phase_c_final_visualization.py" ]]; then
  echo ""
  echo "--- Stage 4: Visualization ---"
  python3 scripts/phase_c_final_visualization.py --indir "$FINAL_DIR" --strict
  echo "[stage4] Visualization complete."
fi

# ---------------------------------------------------------------------------
# Stage 5: Build topology comparison report
# ---------------------------------------------------------------------------
echo ""
echo "--- Stage 5: Topology comparison report ---"

python3 scripts/build_candidate_topology_comparison.py \
  --candidate-name     "$METHOD" \
  --candidate-results  "$FINAL_DIR/phase_c_results.csv" \
  --baseline-results   "ttk_runs_fixed/combined/phase_c_results.csv" \
  --candidate-idx      "$DATA_DIR/idx.npy" \
  --out-dir            "$OUT_BASE" \
  --report-path        "docs/topology_finetuning_${METHOD}_topology_eval.md"

echo ""
echo "========================================"
echo "Pipeline complete."
echo "  VTI files    : $VTI_DIR"
echo "  TTK outputs  : $OUT_BASE"
echo "  Distances    : $FINAL_DIR/phase_c_results.csv"
echo "  Comparison   : $OUT_BASE/${METHOD}_topology_comparison.csv"
echo "  Report       : docs/topology_finetuning_${METHOD}_topology_eval.md"
echo "========================================"
