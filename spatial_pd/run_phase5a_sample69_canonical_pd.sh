#!/usr/bin/env bash
set -euo pipefail

ROOT="${HOME}/PhIRE"
OUT="${1:-${ROOT}/spatial_pd/phase5a_sample69_canonical_pd}"
THREADS="${THREADS:-20}"
IMAGE="${IMAGE:-phire-ttk:latest}"

VTI_DIR="${OUT}/vti"
PD_DIR="${OUT}/pd"
LOG_DIR="${OUT}/logs"

CNN_DATA="data_out_fixed/wind_mrhr_cnn"
UV_DATA="data_out/wind_finetune_candidateUV_expanded2688"
F1_DATA="data_out/wind_finetune_candidateF_grad_E2_low_expanded2688"

if [[ -e "$OUT" ]] && find "$OUT" -mindepth 1 -print -quit | grep -q .; then
  echo "[error] Refusing to overwrite non-empty output directory:"
  echo "        $OUT"
  echo "[error] Move/remove it intentionally, or provide a new output path."
  exit 2
fi

mkdir -p "$VTI_DIR" "$PD_DIR" "$LOG_DIR"

cd "$ROOT"
WORKDIR="$(pwd)"

echo "===== PHASE 5A STEP 3 — CANONICAL SAMPLE-69 PD REGENERATION ====="
date -Is
hostname
echo "root      : $ROOT"
echo "out       : $OUT"
echo "threads   : $THREADS"
echo "image     : $IMAGE"
echo

echo "===== INPUT / SCRIPT HASHES ====="
sha256sum \
  scripts/convert_phire_to_vti.py \
  scripts/run_candidate_topology_pipeline.sh \
  "$CNN_DATA/dataGT.npy" \
  "$CNN_DATA/dataSR.npy" \
  "$UV_DATA/dataGT.npy" \
  "$UV_DATA/dataSR.npy" \
  "$F1_DATA/dataGT.npy" \
  "$F1_DATA/dataSR.npy" \
  | tee "$OUT/input_sha256.txt"

echo
echo "===== DOCKER IMAGE ====="
docker image inspect "$IMAGE" \
  --format 'ID={{.Id}} RepoDigests={{json .RepoDigests}}' \
  | tee "$OUT/docker_image.txt"

generate_vti () {
  local input="$1"
  local label="$2"

  docker run --rm \
    -v "$WORKDIR":/work \
    -w /work \
    "$IMAGE" \
    bash -lc "python scripts/convert_phire_to_vti.py \
      --input '$input' \
      --outdir '${VTI_DIR#${ROOT}/}' \
      --label '$label' \
      --scalar speed \
      --patch 160 \
      --x0 0 \
      --y0 0 \
      --samples 69"
}

echo
echo "===== GENERATE FOUR CANONICAL C-ORDER VTIs ====="

generate_vti "$CNN_DATA/dataGT.npy" "phase5_GT"
generate_vti "$CNN_DATA/dataSR.npy" "phase5_CNN_SR"
generate_vti "$UV_DATA/dataSR.npy" "phase5_UV_SR"
generate_vti "$F1_DATA/dataSR.npy" "phase5_F1_SR"

echo
echo "===== VTI INVENTORY ====="
find "$VTI_DIR" -maxdepth 1 -type f -name '*.vti' -print | sort | tee "$OUT/vti_inventory.txt"

VTI_COUNT="$(find "$VTI_DIR" -maxdepth 1 -type f -name '*.vti' | wc -l)"
if [[ "$VTI_COUNT" -ne 4 ]]; then
  echo "[error] Expected exactly 4 VTIs, found $VTI_COUNT"
  exit 3
fi

echo
echo "===== CANONICALIZATION GATE AGAINST HISTORICAL EXACT TTK INPUTS ====="

PYTHONNOUSERSITE=1 /usr/bin/python3 - "$VTI_DIR" <<'PY'
import sys
from pathlib import Path
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

root = Path.home() / "PhIRE"
vti_dir = Path(sys.argv[1])

def one(pattern):
    hits = sorted(vti_dir.glob(pattern))
    if len(hits) != 1:
        raise RuntimeError(f"{pattern}: expected 1 hit, got {hits}")
    return hits[0]

new = {
    "GT": one("phase5_GT*s69*speed*p160*x0*y0*.vti"),
    "CNN_SR": one("phase5_CNN_SR*s69*speed*p160*x0*y0*.vti"),
    "UV_SR": one("phase5_UV_SR*s69*speed*p160*x0*y0*.vti"),
    "F1_SR": one("phase5_F1_SR*s69*speed*p160*x0*y0*.vti"),
}

old = {
    "GT": root / "ttk_runs_fixed/cnn/mt/cnn_GT_s69_speed_p160_x0_y0_mt_port_2.vti",
    "CNN_SR": root / "ttk_runs_fixed/cnn/mt/cnn_SR_s69_speed_p160_x0_y0_mt_port_2.vti",
    "UV_SR": root / "ttk_runs_fixed/topology_finetuning/candidateUV_expanded2688_topology/mt/SR/candidateUV_expanded2688_SR_s69_speed_p160_x0_y0_mt_port_2.vti",
    "F1_SR": root / "ttk_runs_fixed/topology_finetuning/candidateF_grad_E2_low_expanded2688_topology/mt/SR/candidateF_grad_E2_low_expanded2688_SR_s69_speed_p160_x0_y0_mt_port_2.vti",
}

def read(path):
    if not path.exists():
        raise FileNotFoundError(path)
    r = vtk.vtkXMLImageDataReader()
    r.SetFileName(str(path))
    r.Update()
    img = r.GetOutput()
    dims = img.GetDimensions()
    arr = img.GetPointData().GetArray("wind_speed")
    if arr is None:
        raise RuntimeError(f"{path}: wind_speed missing")
    a = np.asarray(vtk_to_numpy(arr))
    W, H, Z = dims
    if Z != 1:
        raise RuntimeError((path, dims))
    return a.reshape(H, W, order="C")

passed = True

for key in ("GT", "CNN_SR", "UV_SR"):
    a = read(new[key])
    b = read(old[key])
    eq = np.array_equal(a, b)
    maxdiff = float(np.max(np.abs(a.astype(np.float64)-b.astype(np.float64))))
    print(f"{key:7s} new == historical C-order: {eq} max_abs_diff={maxdiff}")
    passed &= eq

a = read(new["F1_SR"])
b = read(old["F1_SR"])
eq_raw = np.array_equal(a, b)
eq_t = np.array_equal(a, b.T)
rawdiff = float(np.max(np.abs(a.astype(np.float64)-b.astype(np.float64))))
tdiff = float(np.max(np.abs(a.astype(np.float64)-b.T.astype(np.float64))))
print(
    "F1_SR   new == historical raw:",
    eq_raw,
    "max_abs_diff=",
    rawdiff,
)
print(
    "F1_SR   new == historical transpose:",
    eq_t,
    "max_abs_diff=",
    tdiff,
)
passed &= eq_t

if not passed:
    raise SystemExit(
        "Canonicalization gate FAILED. Do not run TTK PD extraction."
    )

print("CANONICALIZATION GATE: PASS")
PY

echo
echo "===== GENERATE FOUR CANONICAL TTK PERSISTENCE DIAGRAMS ====="

while IFS= read -r f; do
  base="$(basename "$f" .vti)"
  rel="${f#${ROOT}/}"
  outprefix="${PD_DIR#${ROOT}/}/${base}_pd"

  echo "--- PD: $base ---"

  docker run --rm \
    -v "$WORKDIR":/work \
    -w /work \
    -e OMP_NUM_THREADS="$THREADS" \
    -e TTK_NUM_THREADS="$THREADS" \
    "$IMAGE" \
    bash -lc "ttkPersistenceDiagramCmd -t '$THREADS' \
      -i '$rel' \
      -a wind_speed \
      -o '$outprefix'" \
    2>&1 | tee "$LOG_DIR/${base}_pd.log"
done < <(find "$VTI_DIR" -maxdepth 1 -type f -name '*.vti' | sort)

echo
echo "===== PD INVENTORY ====="
find "$PD_DIR" -maxdepth 1 -type f -print | sort | tee "$OUT/pd_inventory.txt"

PD_COUNT="$(find "$PD_DIR" -maxdepth 1 -type f -name '*_pd_port_0.vtu' | wc -l)"
if [[ "$PD_COUNT" -ne 4 ]]; then
  echo "[error] Expected 4 PD port_0 VTUs, found $PD_COUNT"
  exit 4
fi

echo
echo "===== BASIC PD COUNTS ====="

PYTHONNOUSERSITE=1 /usr/bin/python3 - "$PD_DIR" <<'PY'
import sys
from pathlib import Path
import vtk

pd_dir = Path(sys.argv[1])
files = sorted(pd_dir.glob("*_pd_port_0.vtu"))

for path in files:
    r = vtk.vtkXMLUnstructuredGridReader()
    r.SetFileName(str(path))
    r.Update()
    g = r.GetOutput()

    cd = g.GetCellData()
    finite = cd.GetArray("IsFinite")
    ptype = cd.GetArray("PairType")

    nfinite = 0
    types = {}
    for i in range(g.GetNumberOfCells()):
        f = int(round(finite.GetTuple1(i))) if finite is not None else -999
        t = int(round(ptype.GetTuple1(i))) if ptype is not None else -999
        nfinite += (f == 1)
        types[t] = types.get(t, 0) + 1

    print(
        path.name,
        "points=", g.GetNumberOfPoints(),
        "cells=", g.GetNumberOfCells(),
        "finite=", nfinite,
        "pair_types=", types,
    )
PY

echo
echo "===== FREEZE PHASE 5A STEP 3 OUTPUT ====="

find "$OUT" -type f ! -name sha256_manifest.txt -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  > "$OUT/sha256_manifest.txt"

cat "$OUT/sha256_manifest.txt"

echo
echo "===== PHASE 5A STEP 3 COMPLETE ====="
echo "Outputs preserved under:"
echo "  $OUT"
