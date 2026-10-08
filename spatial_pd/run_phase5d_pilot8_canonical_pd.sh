#!/usr/bin/env bash
set -euo pipefail

ROOT="${HOME}/PhIRE"
OUT="${1:-${ROOT}/spatial_pd/phase5d_pilot8_canonical_pd}"
THREADS="${THREADS:-20}"
IMAGE="${IMAGE:-phire-ttk:latest}"

SAMPLES=(0 24 48 69 96 120 144 167)

VTI_DIR="${OUT}/vti"
PD_DIR="${OUT}/pd"
LOG_DIR="${OUT}/logs"

CNN_DATA="data_out_fixed/wind_mrhr_cnn"
UV_DATA="data_out/wind_finetune_candidateUV_expanded2688"
F1_DATA="data_out/wind_finetune_candidateF_grad_E2_low_expanded2688"

cd "$ROOT"

if [[ -e "$OUT" ]] && find "$OUT" -mindepth 1 -print -quit | grep -q .; then
  echo "[error] Refusing to overwrite non-empty output directory:"
  echo "        $OUT"
  exit 2
fi

mkdir -p "$VTI_DIR" "$PD_DIR" "$LOG_DIR"

echo "===== PHASE 5D — PREDECLARED 8-SAMPLE CANONICAL PD PILOT ====="
date -Is
hostname
echo "root    : $ROOT"
echo "out     : $OUT"
echo "samples : ${SAMPLES[*]}"
echo "threads : $THREADS"
echo "image   : $IMAGE"
echo

echo "===== SCRIPT / DATA HASHES ====="
sha256sum \
  scripts/convert_phire_to_vti.py \
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

sample_args="${SAMPLES[*]}"

generate_vti () {
  local input="$1"
  local label="$2"

  docker run --rm \
    --user "$(id -u):$(id -g)" \
    -v "$ROOT":/work \
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
      --samples $sample_args"
}

echo
echo "===== GENERATE 32 CANONICAL C-ORDER VTIs ====="

generate_vti "$CNN_DATA/dataGT.npy" "phase5_GT"
generate_vti "$CNN_DATA/dataSR.npy" "phase5_CNN_SR"
generate_vti "$UV_DATA/dataSR.npy" "phase5_UV_SR"
generate_vti "$F1_DATA/dataSR.npy" "phase5_F1_SR"

mapfile -t VTI_FILES < <(
  find "$VTI_DIR" -maxdepth 1 -type f -name '*.vti' | sort
)

echo
echo "===== VTI INVENTORY ====="
printf '%s\n' "${VTI_FILES[@]}" | tee "$OUT/vti_inventory.txt"

if [[ "${#VTI_FILES[@]}" -ne 32 ]]; then
  echo "[error] Expected exactly 32 VTIs; found ${#VTI_FILES[@]}"
  exit 3
fi

echo
echo "===== CANONICAL ORIENTATION GATE ====="

PYTHONNOUSERSITE=1 /usr/bin/python3 - "$VTI_DIR" "${SAMPLES[@]}" <<'PY'
import sys
from pathlib import Path
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

vti_dir = Path(sys.argv[1])
samples = [int(x) for x in sys.argv[2:]]
root = Path.home() / "PhIRE"

def read(path):
    if not path.exists():
        raise FileNotFoundError(path)

    r = vtk.vtkXMLImageDataReader()
    r.SetFileName(str(path))
    r.Update()
    img = r.GetOutput()

    W, H, Z = img.GetDimensions()
    if Z != 1:
        raise RuntimeError((path, img.GetDimensions()))

    arr = img.GetPointData().GetArray("wind_speed")
    if arr is None:
        raise RuntimeError(f"{path}: missing wind_speed")

    return np.asarray(vtk_to_numpy(arr)).reshape(H, W, order="C")

def new_path(label, s):
    hits = sorted(vti_dir.glob(
        f"{label}_s{s}_speed_p160_x0_y0.vti"
    ))
    if len(hits) != 1:
        raise RuntimeError((label, s, hits))
    return hits[0]

for s in samples:
    new = {
        "GT": read(new_path("phase5_GT", s)),
        "CNN": read(new_path("phase5_CNN_SR", s)),
        "UV": read(new_path("phase5_UV_SR", s)),
        "F1": read(new_path("phase5_F1_SR", s)),
    }

    hist_paths = {
        "CNN_GT":
            root / f"ttk_runs_fixed/cnn/mt/cnn_GT_s{s}_speed_p160_x0_y0_mt_port_2.vti",
        "CNN_SR":
            root / f"ttk_runs_fixed/cnn/mt/cnn_SR_s{s}_speed_p160_x0_y0_mt_port_2.vti",
        "UV_GT":
            root / (
                "ttk_runs_fixed/topology_finetuning/"
                "candidateUV_expanded2688_topology/mt/GT/"
                f"candidateUV_expanded2688_GT_s{s}_speed_p160_x0_y0_mt_port_2.vti"
            ),
        "UV_SR":
            root / (
                "ttk_runs_fixed/topology_finetuning/"
                "candidateUV_expanded2688_topology/mt/SR/"
                f"candidateUV_expanded2688_SR_s{s}_speed_p160_x0_y0_mt_port_2.vti"
            ),
        "F1_GT":
            root / (
                "ttk_runs_fixed/topology_finetuning/"
                "candidateF_grad_E2_low_expanded2688_topology/mt/GT/"
                f"candidateF_grad_E2_low_expanded2688_GT_s{s}_speed_p160_x0_y0_mt_port_2.vti"
            ),
        "F1_SR":
            root / (
                "ttk_runs_fixed/topology_finetuning/"
                "candidateF_grad_E2_low_expanded2688_topology/mt/SR/"
                f"candidateF_grad_E2_low_expanded2688_SR_s{s}_speed_p160_x0_y0_mt_port_2.vti"
            ),
    }

    hist = {k: read(p) for k, p in hist_paths.items()}

    tests = [
        ("GT==CNN_GT.T", new["GT"], hist["CNN_GT"].T),
        ("GT==UV_GT.T", new["GT"], hist["UV_GT"].T),
        ("GT==F1_GT", new["GT"], hist["F1_GT"]),
        ("CNN==hist_CNN.T", new["CNN"], hist["CNN_SR"].T),
        ("UV==hist_UV.T", new["UV"], hist["UV_SR"].T),
        ("F1==hist_F1", new["F1"], hist["F1_SR"]),
    ]

    for label, a, b in tests:
        eq = np.array_equal(a, b)
        md = float(np.max(np.abs(
            a.astype(np.float64) - b.astype(np.float64)
        )))
        print(
            f"sample={s:3d} {label:18s} "
            f"equal={eq} max_abs_diff={md}"
        )
        if not eq or md != 0.0:
            raise SystemExit(
                f"CANONICAL ORIENTATION GATE: FAIL sample={s} {label}"
            )

print("CANONICAL ORIENTATION GATE: PASS (8/8 samples)")
PY

echo
echo "===== FREEZE VTI HASHES ====="
printf '%s\0' "${VTI_FILES[@]}" \
  | xargs -0 sha256sum \
  | tee "$OUT/vti_sha256.txt"

echo
echo "===== GENERATE 32 TTK PERSISTENCE DIAGRAMS ====="

for f in "${VTI_FILES[@]}"; do
  base="$(basename "$f" .vti)"
  rel="${f#${ROOT}/}"
  outprefix="${PD_DIR#${ROOT}/}/${base}_pd"

  echo
  echo "--- $base ---"

  docker run --rm \
    --user "$(id -u):$(id -g)" \
    -v "$ROOT":/work \
    -w /work \
    -e OMP_NUM_THREADS="$THREADS" \
    -e TTK_NUM_THREADS="$THREADS" \
    "$IMAGE" \
    bash -lc "ttkPersistenceDiagramCmd -t '$THREADS' \
      -i '$rel' \
      -a wind_speed \
      -o '$outprefix'" \
    2>&1 | tee "$LOG_DIR/${base}_pd.log"
done

mapfile -t PD_FILES < <(
  find "$PD_DIR" -maxdepth 1 -type f -name '*_pd_port_0.vtu' | sort
)

echo
echo "===== PD INVENTORY ====="
printf '%s\n' "${PD_FILES[@]}" | tee "$OUT/pd_inventory.txt"

if [[ "${#PD_FILES[@]}" -ne 32 ]]; then
  echo "[error] Expected exactly 32 PD port_0 VTUs; found ${#PD_FILES[@]}"
  exit 4
fi

echo
echo "===== PD SCHEMA / COUNTS ====="

PYTHONNOUSERSITE=1 /usr/bin/python3 - "$PD_DIR" <<'PY'
import sys
from pathlib import Path
import vtk

pd_dir = Path(sys.argv[1])
files = sorted(pd_dir.glob("*_pd_port_0.vtu"))

required_cell = [
    "PairIdentifier", "PairType", "Persistence", "Birth", "IsFinite"
]
required_point = [
    "ttkVertexScalarField", "CriticalType", "Coordinates"
]

for path in files:
    r = vtk.vtkXMLUnstructuredGridReader()
    r.SetFileName(str(path))
    r.Update()
    g = r.GetOutput()

    cd = g.GetCellData()
    pd = g.GetPointData()

    mc = [x for x in required_cell if cd.GetArray(x) is None]
    mp = [x for x in required_point if pd.GetArray(x) is None]
    if mc or mp:
        raise RuntimeError(
            f"{path}: missing cell={mc}, point={mp}"
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
echo "===== FREEZE PD HASHES ====="
printf '%s\0' "${PD_FILES[@]}" \
  | xargs -0 sha256sum \
  | tee "$OUT/pd_sha256.txt"

echo
echo "===== FINAL MANIFEST ====="
find "$OUT" -type f ! -name 'sha256_manifest.txt' -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  > "$OUT/sha256_manifest.txt"

echo "VTI count: ${#VTI_FILES[@]}"
echo "PD count : ${#PD_FILES[@]}"
echo "PHASE 5D PREDECLARED 8-SAMPLE CANONICAL PD PILOT: PASS"
