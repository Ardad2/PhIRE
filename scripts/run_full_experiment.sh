#!/usr/bin/env bash
set -euo pipefail

# Full end-to-end experiment runner with optional resume behavior.
# Stages:
#  0) Ensure 500x500 TFRecords are staged in example_data/
#  1) Run paired GAN/CNN inference and snapshot to stable dirs data_out/wind_mrhr_{gan,cnn}
#  2) Generate VTI files for selected samples
#  3) Extract PD/MT using TTK docker image (resume-friendly per-sample checks)
#  4) Compute distances, combine reports, PSNR/topology analysis, physics merge, add SSIM, and visualize

METHODS=(gan cnn)
PATCH=160
X0=0
Y0=0
THREADS=20
OUT_ROOT="ttk_runs"
DOCKER_IMAGE="phire-ttk:latest"
SAMPLES=($(seq 0 167))
RESUME=0
DO_MODELS=1
DO_TOPO=1
DO_POST=1

usage() {
  cat <<EOF
Usage: $0 [options]

Options:
  --samples "0 1 ..."      Sample indices (default: 0..167)
  --patch N                Patch size (default: 160)
  --x0 N                   Crop x0 (default: 0)
  --y0 N                   Crop y0 (default: 0)
  --threads N              TTK threads (default: 20)
  --out-root DIR           Output root (default: ttk_runs)
  --docker-image IMG       Docker image (default: phire-ttk:latest)
  --resume                 Skip already-created artifacts where possible
  --skip-models            Skip GAN/CNN inference stage
  --skip-topology          Skip VTI/PD/MT extraction stage
  --skip-post              Skip distance/combine/analysis/plot stage
  -h, --help               Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --samples) shift; read -r -a SAMPLES <<< "${1:-}"; shift ;;
    --patch) PATCH="$2"; shift 2 ;;
    --x0) X0="$2"; shift 2 ;;
    --y0) Y0="$2"; shift 2 ;;
    --threads) THREADS="$2"; shift 2 ;;
    --out-root) OUT_ROOT="$2"; shift 2 ;;
    --docker-image) DOCKER_IMAGE="$2"; shift 2 ;;
    --resume) RESUME=1; shift ;;
    --skip-models) DO_MODELS=0; shift ;;
    --skip-topology) DO_TOPO=0; shift ;;
    --skip-post) DO_POST=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

for c in python3 docker; do
  command -v "$c" >/dev/null 2>&1 || { echo "Missing command: $c"; exit 1; }
done

max_s=0
for s in "${SAMPLES[@]}"; do
  [[ "$s" =~ ^[0-9]+$ ]] || { echo "Bad sample index: $s"; exit 1; }
  (( s > max_s )) && max_s=$s
done

echo "============================================================"
echo "PhIRE full experiment"
echo "methods      : ${METHODS[*]}"
echo "samples      : ${SAMPLES[*]} (max=$max_s)"
echo "patch/x0/y0  : ${PATCH}/${X0}/${Y0}"
echo "threads      : ${THREADS}"
echo "out-root     : ${OUT_ROOT}"
echo "docker-image : ${DOCKER_IMAGE}"
echo "resume       : ${RESUME}"
echo "============================================================"

echo
echo "[0] Staging 500x500 TFRecords into example_data/"
if [[ -f example_data_extension_500/wind_MR-HR.tfrecord ]]; then
  cp -f example_data_extension_500/wind_*.tfrecord example_data/
else
  echo "[warn] example_data_extension_500 not found; skipping TFRecord staging"
fi

run_model() {
  local method="$1"
  local runner=""
  if [[ "$method" == "gan" ]]; then runner="run_paired_wind_mr_hr.py"; fi
  if [[ "$method" == "cnn" ]]; then runner="run_paired_wind_mr_hr_cnn.py"; fi
  [[ -f "$runner" ]] || { echo "Missing $runner"; exit 1; }

  python3 "$runner"

  local latest stable
  latest="$(ls -dt data_out/wind-* | head -n 1)"
  stable="data_out/wind_mrhr_${method}"
  mkdir -p "$stable"
  cp -f "$latest"/dataGT.npy "$stable"/dataGT.npy
  cp -f "$latest"/dataIN.npy "$stable"/dataIN.npy
  cp -f "$latest"/dataSR.npy "$stable"/dataSR.npy
  cp -f "$latest"/idx.npy "$stable"/idx.npy
  echo "  [$method] snapshotted to $stable from $latest"
}

verify_method_n() {
  local method="$1"
  local d="data_out/wind_mrhr_${method}"
  python3 - <<PY
import numpy as np
from pathlib import Path

d = Path('$d')
idx = np.load(d/'idx.npy', mmap_mode='r')
gt = np.load(d/'dataGT.npy', mmap_mode='r')
sr = np.load(d/'dataSR.npy', mmap_mode='r')
print('$method', 'N=', idx.shape[0], 'GT=', gt.shape, 'SR=', sr.shape)
assert idx.shape[0] >= $((max_s+1)), 'Not enough samples in $d'
PY
}

if [[ "$DO_MODELS" -eq 1 ]]; then
  echo
  echo "[1] Running GAN/CNN paired inference + stable snapshots"
  run_model gan
  run_model cnn
fi

echo
echo "[1b] Verifying stable output dirs"
verify_method_n gan
verify_method_n cnn

missing_vtis_for_label() {
  local label="$1"
  local miss=()
  local s
  for s in "${SAMPLES[@]}"; do
    local p="vtk_inputs/${label}_s${s}_speed_p${PATCH}_x${X0}_y${Y0}.vti"
    if [[ ! -f "$p" ]]; then
      miss+=("$s")
    fi
  done
  echo "${miss[*]:-}"
}

convert_vtis() {
  local npy="$1"
  local label="$2"
  local sample_list

  if [[ "$RESUME" -eq 1 ]]; then
    sample_list="$(missing_vtis_for_label "$label")"
    if [[ -z "${sample_list// }" ]]; then
      echo "  [$label] all VTIs present; skipping"
      return
    fi
    echo "  [$label] missing VTIs for samples: $sample_list"
  else
    sample_list="${SAMPLES[*]}"
    echo "  [$label] writing all requested samples"
  fi

  docker run --rm -v "$PWD":/work -w /work "$DOCKER_IMAGE" \
    bash -lc "python scripts/convert_phire_to_vti.py \
      --input '$npy' \
      --outdir vtk_inputs \
      --label '$label' \
      --scalar speed \
      --patch ${PATCH} --x0 ${X0} --y0 ${Y0} \
      --samples ${sample_list}"
}

if [[ "$DO_TOPO" -eq 1 ]]; then
  echo
  echo "[2] Generating VTIs"
  mkdir -p vtk_inputs
  convert_vtis "data_out/wind_mrhr_gan/dataGT.npy" "gan_GT"
  convert_vtis "data_out/wind_mrhr_gan/dataSR.npy" "gan_SR"
  convert_vtis "data_out/wind_mrhr_cnn/dataGT.npy" "cnn_GT"
  convert_vtis "data_out/wind_mrhr_cnn/dataSR.npy" "cnn_SR"
fi

ensure_pd_mt_one() {
  local method="$1" kind="$2" s="$3"
  local stem="${method}_${kind}_s${s}_speed_p${PATCH}_x${X0}_y${Y0}"
  local vti="vtk_inputs/${stem}.vti"
  local pd0="${OUT_ROOT}/${method}/pd/${stem}_pd_port_0.vtu"
  local mt0="${OUT_ROOT}/${method}/mt/${stem}_mt_port_0.vtu"
  local mt1="${OUT_ROOT}/${method}/mt/${stem}_mt_port_1.vtu"
  local mt2="${OUT_ROOT}/${method}/mt/${stem}_mt_port_2.vti"

  [[ -f "$vti" ]] || { echo "Missing VTI: $vti"; exit 1; }

  if [[ "$RESUME" -eq 1 && -f "$pd0" && -f "$mt0" && -f "$mt1" && -f "$mt2" ]]; then
    return
  fi

  if [[ ! -f "$pd0" ]]; then
    docker run --rm -v "$PWD":/work -w /work "$DOCKER_IMAGE" \
      bash -lc "ttkPersistenceDiagramCmd -t ${THREADS} -i '$vti' -a wind_speed -o '${OUT_ROOT}/${method}/pd/${stem}_pd'"
  fi

  if [[ ! -f "$mt0" || ! -f "$mt1" || ! -f "$mt2" ]]; then
    docker run --rm -v "$PWD":/work -w /work "$DOCKER_IMAGE" \
      bash -lc "ttkMergeTreeCmd -t ${THREADS} -i '$vti' -a wind_speed -o '${OUT_ROOT}/${method}/mt/${stem}_mt'"
  fi
}

if [[ "$DO_TOPO" -eq 1 ]]; then
  echo
  echo "[3] Extracting PD/MT with resume checks"
  mkdir -p "${OUT_ROOT}/gan/pd" "${OUT_ROOT}/gan/mt" "${OUT_ROOT}/cnn/pd" "${OUT_ROOT}/cnn/mt"
  local_method=""
  local_kind=""
  local_s=""
  for local_method in "${METHODS[@]}"; do
    for local_kind in GT SR; do
      echo "  [${local_method}/${local_kind}]"
      for local_s in "${SAMPLES[@]}"; do
        ensure_pd_mt_one "$local_method" "$local_kind" "$local_s"
      done
    done
  done
fi

if [[ "$DO_POST" -eq 1 ]]; then
  echo
  echo "[4] Postprocessing and analysis"

  for method in "${METHODS[@]}"; do
    python3 -X faulthandler scripts/compute_composite_tree_distance.py \
      --pd-dir "${OUT_ROOT}/${method}/pd" \
      --mt-dir "${OUT_ROOT}/${method}/mt" \
      --outdir "${OUT_ROOT}/${method}/phase_c_final" \
      --isolate-mt --debug

    python3 scripts/phase_c_final_visualization.py --indir "${OUT_ROOT}/${method}/phase_c_final" --strict
  done

  python3 scripts/combine_phase_c_results.py \
    --inputs "${OUT_ROOT}/gan/phase_c_final" "${OUT_ROOT}/cnn/phase_c_final" \
    --outdir "${OUT_ROOT}/combined"

  cp -f "${OUT_ROOT}/combined/combined_pairwise_results.csv" "${OUT_ROOT}/combined/phase_c_results.csv"
  python3 scripts/phase_c_final_visualization.py --indir "${OUT_ROOT}/combined" || true

  python3 scripts/analyze_psnr_vs_ttk_topology.py \
    --combined "${OUT_ROOT}/combined/combined_pairwise_results.csv" \
    --gan-dir data_out/wind_mrhr_gan \
    --cnn-dir data_out/wind_mrhr_cnn \
    --outdir "${OUT_ROOT}/combined"

  python3 scripts/analysis_compare.py \
    --topology-csv "${OUT_ROOT}/combined/combined_pairwise_results.csv" \
    --method-dir gan=data_out/wind_mrhr_gan \
    --method-dir cnn=data_out/wind_mrhr_cnn \
    --out-csv "${OUT_ROOT}/combined/psnr_topology_physics_merged.csv" \
    --out-report "${OUT_ROOT}/combined/psnr_topology_physics_report.txt" \
    --out-delta-csv "${OUT_ROOT}/combined/psnr_topology_physics_delta.csv"

  PYTHONNOUSERSITE=1 /usr/bin/python3 scripts/add_ssim_to_merged.py \
    --in-csv "${OUT_ROOT}/combined/psnr_topology_physics_merged.csv" \
    --out-csv "${OUT_ROOT}/combined/psnr_topology_physics_merged.csv" \
    --method-dir gan=data_out/wind_mrhr_gan \
    --method-dir cnn=data_out/wind_mrhr_cnn \
    --patch "${PATCH}" --x0 "${X0}" --y0 "${Y0}"

  PYTHONNOUSERSITE=1 /usr/bin/python3 scripts/plot_topology_physics_analysis.py \
    --merged-csv "${OUT_ROOT}/combined/psnr_topology_physics_merged.csv" \
    --delta-csv "${OUT_ROOT}/combined/psnr_topology_physics_delta.csv" \
    --outdir "${OUT_ROOT}/combined/figs_topology_physics"
fi

echo
echo "[done] Full experiment complete."
echo "Combined outputs: ${OUT_ROOT}/combined"
