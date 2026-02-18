#!/usr/bin/env bash
set -euo pipefail

# End-to-end topology pipeline for one or more methods (gan/cnn).
# Steps: VTI generation -> TTK PD/MT extraction -> distance compute -> visualization -> combine summary.

DOCKER_IMAGE="phire-ttk:latest"
METHODS=(gan)
SAMPLES=(0 1 2 3 4)
SCALAR="speed"
PATCH=160
X0=0
Y0=0
THREADS=20
WORKDIR="$(pwd)"
VTI_ROOT="vtk_inputs"
OUT_ROOT="ttk_runs"
DEBUG=0

usage() {
  cat <<EOF
Usage: $0 [options]

Options:
  --methods "gan cnn"   Methods to run (default: "gan")
  --samples "0 1 2 3 4" Sample indices (default: "0 1 2 3 4")
  --scalar speed|u|v     Scalar for VTI generation (default: speed)
  --patch N              Patch size (default: 160)
  --x0 N                 Patch x offset (default: 0)
  --y0 N                 Patch y offset (default: 0)
  --threads N            TTK threads (default: 20)
  --docker-image IMG     Docker image (default: phire-ttk:latest)
  --vti-root DIR         VTI output root (default: vtk_inputs)
  --out-root DIR         Output root (default: ttk_runs)
  --debug                Enable debug mode for distance script
  -h, --help             Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --methods) read -r -a METHODS <<< "$2"; shift 2 ;;
    --samples) read -r -a SAMPLES <<< "$2"; shift 2 ;;
    --scalar) SCALAR="$2"; shift 2 ;;
    --patch) PATCH="$2"; shift 2 ;;
    --x0) X0="$2"; shift 2 ;;
    --y0) Y0="$2"; shift 2 ;;
    --threads) THREADS="$2"; shift 2 ;;
    --docker-image) DOCKER_IMAGE="$2"; shift 2 ;;
    --vti-root) VTI_ROOT="$2"; shift 2 ;;
    --out-root) OUT_ROOT="$2"; shift 2 ;;
    --debug) DEBUG=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 2 ;;
  esac
done

mkdir -p "$OUT_ROOT"

run_convert() {
  local method="$1"
  local vti_dir="$2"
  local data_dir="data_out/wind_mrhr_${method}"

  if [[ ! -f "$data_dir/dataGT.npy" || ! -f "$data_dir/dataSR.npy" ]]; then
    echo "[error] missing inputs under $data_dir"
    exit 1
  fi

  mkdir -p "$vti_dir"

  docker run --rm -v "$WORKDIR":/work -w /work "$DOCKER_IMAGE" \
    bash -lc "python scripts/convert_phire_to_vti.py \
      --input $data_dir/dataGT.npy \
      --outdir $vti_dir \
      --label ${method}_GT \
      --scalar $SCALAR \
      --patch $PATCH --x0 $X0 --y0 $Y0 \
      --samples ${SAMPLES[*]}"

  docker run --rm -v "$WORKDIR":/work -w /work "$DOCKER_IMAGE" \
    bash -lc "python scripts/convert_phire_to_vti.py \
      --input $data_dir/dataSR.npy \
      --outdir $vti_dir \
      --label ${method}_SR \
      --scalar $SCALAR \
      --patch $PATCH --x0 $X0 --y0 $Y0 \
      --samples ${SAMPLES[*]}"
}

run_ttk_extract() {
  local method="$1"
  local vti_dir="$2"
  local out_base="$3"
  local pd_dir="$out_base/pd"
  local mt_dir="$out_base/mt"

  mkdir -p "$pd_dir" "$mt_dir"

  shopt -s nullglob
  local files=("$vti_dir"/${method}_*.vti)
  shopt -u nullglob
  if [[ ${#files[@]} -eq 0 ]]; then
    echo "[error] no VTI files found for method=$method in $vti_dir"
    exit 1
  fi

  for f in "${files[@]}"; do
    local base
    base="$(basename "$f" .vti)"
    echo "=== PD: $base ==="
    docker run --rm -v "$WORKDIR":/work -w /work "$DOCKER_IMAGE" \
      bash -lc "ttkPersistenceDiagramCmd -t $THREADS -i $f -a wind_speed -o $pd_dir/${base}_pd"

    echo "=== MT: $base ==="
    docker run --rm -v "$WORKDIR":/work -w /work "$DOCKER_IMAGE" \
      bash -lc "ttkMergeTreeCmd -t $THREADS -i $f -a wind_speed -o $mt_dir/${base}_mt"
  done

  mkdir -p "$pd_dir/GT" "$pd_dir/SR" "$mt_dir/GT" "$mt_dir/SR"
  mv "$pd_dir"/${method}_GT_* "$pd_dir/GT/" 2>/dev/null || true
  mv "$pd_dir"/${method}_SR_* "$pd_dir/SR/" 2>/dev/null || true
  mv "$mt_dir"/${method}_GT_* "$mt_dir/GT/" 2>/dev/null || true
  mv "$mt_dir"/${method}_SR_* "$mt_dir/SR/" 2>/dev/null || true

  echo "[check] PD count: $(find "$pd_dir" -name '*.vtu' | wc -l)"
  echo "[check] MT port0 count: $(find "$mt_dir" -name '*_port_0.vtu' | wc -l)"
  echo "[check] MT port1 count: $(find "$mt_dir" -name '*_port_1.vtu' | wc -l)"
  echo "[check] MT port2 count: $(find "$mt_dir" -name '*_port_2.vti' | wc -l)"
}

run_dist_and_viz() {
  local out_base="$1"
  local pd_dir="$out_base/pd"
  local mt_dir="$out_base/mt"
  local final_dir="$out_base/phase_c_final"

  local debug_flag=""
  if [[ "$DEBUG" -eq 1 ]]; then
    debug_flag="--debug"
  fi

  python3 -X faulthandler scripts/compute_composite_tree_distance.py \
    --pd-dir "$pd_dir" \
    --mt-dir "$mt_dir" \
    --outdir "$final_dir" \
    --isolate-mt \
    $debug_flag

  python3 scripts/phase_c_final_visualization.py --indir "$final_dir" --strict
}

final_inputs=()
for method in "${METHODS[@]}"; do
  echo "\n================ METHOD: $method ================"
  vti_dir="$VTI_ROOT"
  out_base="$OUT_ROOT/$method"
  mkdir -p "$out_base"

  run_convert "$method" "$vti_dir"
  run_ttk_extract "$method" "$vti_dir" "$out_base"
  run_dist_and_viz "$out_base"

  final_inputs+=("$out_base/phase_c_final")
done

if [[ ${#final_inputs[@]} -ge 2 ]]; then
  echo "\n================ COMBINED REPORT ================"
  python3 scripts/combine_phase_c_results.py --inputs "${final_inputs[@]}" --outdir "$OUT_ROOT/combined"
fi

echo "\n[done] pipeline complete"
