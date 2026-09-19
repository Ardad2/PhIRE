#!/usr/bin/env bash
set -euo pipefail

ROOT="${HOME}/PhIRE"
OUT="${1:-${ROOT}/spatial_pd/phase5a_sample69_canonical_pd}"
THREADS="${THREADS:-20}"
IMAGE="${IMAGE:-phire-ttk:latest}"

VTI_DIR="${OUT}/vti"
PD_DIR="${OUT}/pd"
LOG_DIR="${OUT}/logs_step3b"

mkdir -p "$PD_DIR" "$LOG_DIR"

cd "$ROOT"
WORKDIR="$(pwd)"

echo "===== PHASE 5A STEP 3B — CANONICAL SAMPLE-69 PD EXTRACTION ====="
date -Is
hostname
echo "root      : $ROOT"
echo "out       : $OUT"
echo "threads   : $THREADS"
echo "image     : $IMAGE"
echo

if [[ ! -d "$VTI_DIR" ]]; then
  echo "[error] Missing VTI directory: $VTI_DIR"
  exit 2
fi

VTI_COUNT="$(find "$VTI_DIR" -maxdepth 1 -type f -name '*.vti' | wc -l)"
if [[ "$VTI_COUNT" -ne 4 ]]; then
  echo "[error] Expected exactly four canonical VTIs; found $VTI_COUNT"
  exit 3
fi

if find "$PD_DIR" -maxdepth 1 -type f -print -quit | grep -q .; then
  echo "[error] Refusing to overwrite non-empty PD directory:"
  echo "        $PD_DIR"
  echo "[error] Move/remove it intentionally before rerunning Step 3B."
  exit 4
fi

echo "===== CANONICAL VTI INVENTORY / HASHES ====="
find "$VTI_DIR" -maxdepth 1 -type f -name '*.vti' -print0 \
  | sort -z | xargs -0 sha256sum \
  | tee "$OUT/step3b_vti_sha256.txt"

echo
echo "===== CORRECTED EXACT ORIENTATION GATE ====="

PYTHONNOUSERSITE=1 /usr/bin/python3 - "$VTI_DIR" <<'PY'
import sys
from pathlib import Path
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

root = Path.home() / "PhIRE"
vti = Path(sys.argv[1])

def one(name):
    p = vti / name
    if not p.exists():
        raise FileNotFoundError(p)
    return p

new_paths = {
    "GT": one("phase5_GT_s69_speed_p160_x0_y0.vti"),
    "CNN_SR": one("phase5_CNN_SR_s69_speed_p160_x0_y0.vti"),
    "UV_SR": one("phase5_UV_SR_s69_speed_p160_x0_y0.vti"),
    "F1_SR": one("phase5_F1_SR_s69_speed_p160_x0_y0.vti"),
}

old_paths = {
    "CNN_GT": root / "ttk_runs_fixed/cnn/mt/cnn_GT_s69_speed_p160_x0_y0_mt_port_2.vti",
    "CNN_SR": root / "ttk_runs_fixed/cnn/mt/cnn_SR_s69_speed_p160_x0_y0_mt_port_2.vti",
    "UV_GT": root / "ttk_runs_fixed/topology_finetuning/candidateUV_expanded2688_topology/mt/GT/candidateUV_expanded2688_GT_s69_speed_p160_x0_y0_mt_port_2.vti",
    "UV_SR": root / "ttk_runs_fixed/topology_finetuning/candidateUV_expanded2688_topology/mt/SR/candidateUV_expanded2688_SR_s69_speed_p160_x0_y0_mt_port_2.vti",
    "F1_GT": root / "ttk_runs_fixed/topology_finetuning/candidateF_grad_E2_low_expanded2688_topology/mt/GT/candidateF_grad_E2_low_expanded2688_GT_s69_speed_p160_x0_y0_mt_port_2.vti",
    "F1_SR": root / "ttk_runs_fixed/topology_finetuning/candidateF_grad_E2_low_expanded2688_topology/mt/SR/candidateF_grad_E2_low_expanded2688_SR_s69_speed_p160_x0_y0_mt_port_2.vti",
}

def read(path):
    r = vtk.vtkXMLImageDataReader()
    r.SetFileName(str(path))
    r.Update()
    img = r.GetOutput()
    W,H,Z = img.GetDimensions()
    arr = img.GetPointData().GetArray("wind_speed")
    if arr is None:
        raise RuntimeError(f"{path}: missing wind_speed")
    a = np.asarray(vtk_to_numpy(arr)).reshape(H,W,order="C")
    return a

new = {k: read(p) for k,p in new_paths.items()}
old = {k: read(p) for k,p in old_paths.items()}

tests = [
    ("GT == transpose(CNN_GT)", new["GT"], old["CNN_GT"].T),
    ("GT == transpose(UV_GT)", new["GT"], old["UV_GT"].T),
    ("GT == F1_GT raw", new["GT"], old["F1_GT"]),
    ("CNN_SR == transpose(CNN_SR historical)", new["CNN_SR"], old["CNN_SR"].T),
    ("UV_SR == transpose(UV_SR historical)", new["UV_SR"], old["UV_SR"].T),
    ("F1_SR == F1_SR historical raw", new["F1_SR"], old["F1_SR"]),
]

ok = True
for label,a,b in tests:
    eq = np.array_equal(a,b)
    md = float(np.max(np.abs(a.astype(np.float64)-b.astype(np.float64))))
    print(f"{label}: equal={eq} max_abs_diff={md}")
    ok &= eq and md == 0.0

if not ok:
    raise SystemExit("CORRECTED ORIENTATION GATE: FAIL")

print("CORRECTED ORIENTATION GATE: PASS")
PY

echo
echo "===== DOCKER IMAGE ====="
docker image inspect "$IMAGE" \
  --format 'ID={{.Id}} RepoDigests={{json .RepoDigests}}' \
  | tee "$OUT/step3b_docker_image.txt"

echo
echo "===== GENERATE FOUR CANONICAL TTK PERSISTENCE DIAGRAMS ====="

while IFS= read -r f; do
  base="$(basename "$f" .vti)"
  rel="${f#${ROOT}/}"
  outprefix="${PD_DIR#${ROOT}/}/${base}_pd"

  echo
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
find "$PD_DIR" -maxdepth 1 -type f -print | sort \
  | tee "$OUT/step3b_pd_inventory.txt"

PD_COUNT="$(find "$PD_DIR" -maxdepth 1 -type f -name '*_pd_port_0.vtu' | wc -l)"
if [[ "$PD_COUNT" -ne 4 ]]; then
  echo "[error] Expected four PD port_0 VTUs; found $PD_COUNT"
  exit 5
fi

echo
echo "===== BASIC PD COUNTS / ARRAY CHECK ====="

PYTHONNOUSERSITE=1 /usr/bin/python3 - "$PD_DIR" <<'PY'
import sys
from pathlib import Path
import vtk

pd_dir = Path(sys.argv[1])

for path in sorted(pd_dir.glob("*_pd_port_0.vtu")):
    r = vtk.vtkXMLUnstructuredGridReader()
    r.SetFileName(str(path))
    r.Update()
    g = r.GetOutput()

    cd = g.GetCellData()
    pd = g.GetPointData()

    required_cell = ["PairIdentifier","PairType","Persistence","Birth","IsFinite"]
    required_point = ["ttkVertexScalarField","CriticalType","Coordinates"]

    missing_cell = [x for x in required_cell if cd.GetArray(x) is None]
    missing_point = [x for x in required_point if pd.GetArray(x) is None]
    if missing_cell or missing_point:
        raise RuntimeError(
            f"{path}: missing cell={missing_cell} point={missing_point}"
        )

    finite = cd.GetArray("IsFinite")
    ptype = cd.GetArray("PairType")

    nfinite = 0
    types = {}
    for i in range(g.GetNumberOfCells()):
        f = int(round(finite.GetTuple1(i)))
        t = int(round(ptype.GetTuple1(i)))
        nfinite += (f == 1)
        types[t] = types.get(t, 0) + 1

    print(
        path.name,
        "points=", g.GetNumberOfPoints(),
        "cells=", g.GetNumberOfCells(),
        "finite=", nfinite,
        "pair_types=", types,
        "schema=PASS",
    )
PY

echo
echo "===== FREEZE STEP 3B OUTPUT ====="

find "$OUT" -type f ! -name 'step3b_sha256_manifest.txt' -print0 \
  | sort -z | xargs -0 sha256sum \
  > "$OUT/step3b_sha256_manifest.txt"

cat "$OUT/step3b_sha256_manifest.txt"

echo
echo "===== PHASE 5A STEP 3B COMPLETE ====="
echo "Canonical PDs:"
echo "  $PD_DIR"
