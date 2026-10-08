#!/usr/bin/env bash
set -euo pipefail

ROOT="${HOME}/PhIRE"
OUT="${1:-${ROOT}/spatial_pd/phase5e_all168_canonical_pd}"
THREADS="${THREADS:-20}"
IMAGE="${IMAGE:-phire-ttk:latest}"

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

echo "===== PHASE 5E — ALL-168 CANONICAL PD GENERATION ====="
date -Is
hostname
echo "root    : $ROOT"
echo "out     : $OUT"
echo "threads : $THREADS"
echo "image   : $IMAGE"
echo

echo "===== INPUT HASHES ====="
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
echo "===== GT EQUALITY GATE ====="
PYTHONNOUSERSITE=1 /usr/bin/python3 - <<'PY'
import numpy as np
pairs = [
    ("CNN", "data_out_fixed/wind_mrhr_cnn/dataGT.npy"),
    ("UV", "data_out/wind_finetune_candidateUV_expanded2688/dataGT.npy"),
    ("F1", "data_out/wind_finetune_candidateF_grad_E2_low_expanded2688/dataGT.npy"),
]
arr = {k: np.load(p, mmap_mode="r") for k,p in pairs}
for k,v in arr.items():
    print(k, v.shape, v.dtype)
if arr["CNN"].shape[0] != 168:
    raise SystemExit("Expected 168 samples")
for other in ("UV","F1"):
    if not np.array_equal(arr["CNN"], arr[other]):
        raise SystemExit(f"GT equality FAIL: CNN vs {other}")
print("GT EQUALITY GATE: PASS")
PY

echo
echo "===== DOCKER IMAGE ====="
docker image inspect "$IMAGE" \
  --format 'ID={{.Id}} RepoDigests={{json .RepoDigests}}' \
  | tee "$OUT/docker_image.txt"

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
      --y0 0"
}

echo
echo "===== GENERATE 672 CANONICAL C-ORDER VTIs ====="
generate_vti "$CNN_DATA/dataGT.npy" "phase5_GT"
generate_vti "$CNN_DATA/dataSR.npy" "phase5_CNN_SR"
generate_vti "$UV_DATA/dataSR.npy" "phase5_UV_SR"
generate_vti "$F1_DATA/dataSR.npy" "phase5_F1_SR"

mapfile -t VTI_FILES < <(
  find "$VTI_DIR" -maxdepth 1 -type f -name '*.vti' | sort
)

if [[ "${#VTI_FILES[@]}" -ne 672 ]]; then
  echo "[error] Expected exactly 672 VTIs; found ${#VTI_FILES[@]}"
  exit 3
fi

echo "VTI count: ${#VTI_FILES[@]}"

echo
echo "===== PILOT-8 CONTINUITY GATE ====="
PILOT="$ROOT/spatial_pd/phase5d_pilot8_canonical_pd/vti"
SAMPLES=(0 24 48 69 96 120 144 167)
LABELS=(phase5_GT phase5_CNN_SR phase5_UV_SR phase5_F1_SR)

for s in "${SAMPLES[@]}"; do
  for label in "${LABELS[@]}"; do
    new="$VTI_DIR/${label}_s${s}_speed_p160_x0_y0.vti"
    old="$PILOT/${label}_s${s}_speed_p160_x0_y0.vti"
    if [[ ! -f "$old" ]]; then
      echo "[error] Missing pilot reference: $old"
      exit 4
    fi
    hnew="$(sha256sum "$new" | awk '{print $1}')"
    hold="$(sha256sum "$old" | awk '{print $1}')"
    if [[ "$hnew" != "$hold" ]]; then
      echo "[error] Pilot continuity FAIL s=$s label=$label"
      echo "new=$hnew"
      echo "old=$hold"
      exit 5
    fi
  done
done
echo "PILOT-8 VTI CONTINUITY GATE: PASS (32/32)"

printf '%s\0' "${VTI_FILES[@]}" \
  | xargs -0 sha256sum \
  | tee "$OUT/vti_sha256.txt" >/dev/null

echo
echo "===== GENERATE 672 TTK PERSISTENCE DIAGRAMS ====="

count=0
for f in "${VTI_FILES[@]}"; do
  base="$(basename "$f" .vti)"
  rel="${f#${ROOT}/}"
  outprefix="${PD_DIR#${ROOT}/}/${base}_pd"

  count=$((count+1))
  echo "[$count/672] $base"

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
    > "$LOG_DIR/${base}_pd.log" 2>&1
done

mapfile -t PD_FILES < <(
  find "$PD_DIR" -maxdepth 1 -type f -name '*_pd_port_0.vtu' | sort
)

if [[ "${#PD_FILES[@]}" -ne 672 ]]; then
  echo "[error] Expected exactly 672 PD port_0 VTUs; found ${#PD_FILES[@]}"
  exit 6
fi

echo "PD count: ${#PD_FILES[@]}"

echo
echo "===== PILOT-8 PD CONTINUITY GATE ====="
PILOT_PD="$ROOT/spatial_pd/phase5d_pilot8_canonical_pd/pd"

for s in "${SAMPLES[@]}"; do
  for label in "${LABELS[@]}"; do
    new="$PD_DIR/${label}_s${s}_speed_p160_x0_y0_pd_port_0.vtu"
    old="$PILOT_PD/${label}_s${s}_speed_p160_x0_y0_pd_port_0.vtu"
    hnew="$(sha256sum "$new" | awk '{print $1}')"
    hold="$(sha256sum "$old" | awk '{print $1}')"
    if [[ "$hnew" != "$hold" ]]; then
      echo "[error] Pilot PD continuity FAIL s=$s label=$label"
      echo "new=$hnew"
      echo "old=$hold"
      exit 7
    fi
  done
done
echo "PILOT-8 PD CONTINUITY GATE: PASS (32/32)"

echo
echo "===== PD SCHEMA / COUNT GATE ====="
PYTHONNOUSERSITE=1 /usr/bin/python3 - "$PD_DIR" <<'PY'
import sys
from pathlib import Path
import vtk

pd_dir = Path(sys.argv[1])
files = sorted(pd_dir.glob("*_pd_port_0.vtu"))
if len(files) != 672:
    raise SystemExit(f"expected 672, found {len(files)}")

required_cell = ["PairIdentifier","PairType","Persistence","Birth","IsFinite"]
required_point = ["ttkVertexScalarField","CriticalType","Coordinates"]

bad = []
for idx,path in enumerate(files,1):
    r=vtk.vtkXMLUnstructuredGridReader()
    r.SetFileName(str(path)); r.Update()
    g=r.GetOutput()
    cd=g.GetCellData(); pd=g.GetPointData()
    mc=[x for x in required_cell if cd.GetArray(x) is None]
    mp=[x for x in required_point if pd.GetArray(x) is None]
    if mc or mp:
        bad.append((path.name,mc,mp))
    if idx % 100 == 0 or idx == len(files):
        print(f"schema checked {idx}/{len(files)}")
if bad:
    raise SystemExit(f"schema failures: {bad[:10]}")
print("PD SCHEMA GATE: PASS (672/672)")
PY

printf '%s\0' "${PD_FILES[@]}" \
  | xargs -0 sha256sum \
  | tee "$OUT/pd_sha256.txt" >/dev/null

echo
echo "===== FINAL MANIFEST ====="
find "$OUT" -type f ! -name 'sha256_manifest.txt' -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  > "$OUT/sha256_manifest.txt"

echo "VTI count: ${#VTI_FILES[@]}"
echo "PD count : ${#PD_FILES[@]}"
echo "PHASE 5E ALL-168 CANONICAL PD GENERATION: PASS"
