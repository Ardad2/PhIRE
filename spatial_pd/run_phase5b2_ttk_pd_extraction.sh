#!/usr/bin/env bash
set -euo pipefail

ROOT="${HOME}/PhIRE"
OUT="${1:-${ROOT}/spatial_pd/phase5b2_near_degenerate}"
THREADS="${THREADS:-20}"
IMAGE="${IMAGE:-phire-ttk:latest}"

VTI_DIR="${OUT}/vti"
PD_DIR="${OUT}/pd"
LOG_DIR="${OUT}/logs_pd"

cd "$ROOT"

echo "===== PHASE 5B2 — TTK PD EXTRACTION ====="
date -Is
hostname
echo "out       : $OUT"
echo "threads   : $THREADS"
echo "image     : $IMAGE"
echo

if [[ ! -d "$VTI_DIR" ]]; then
  echo "[error] Missing VTI directory: $VTI_DIR"
  exit 2
fi

mapfile -t VTI_FILES < <(find "$VTI_DIR" -maxdepth 1 -type f -name '*.vti' | sort)

if [[ "${#VTI_FILES[@]}" -ne 8 ]]; then
  echo "[error] Expected exactly 8 VTI files; found ${#VTI_FILES[@]}"
  printf '  %s\n' "${VTI_FILES[@]}"
  exit 3
fi

if [[ -e "$PD_DIR" ]] && find "$PD_DIR" -mindepth 1 -print -quit | grep -q .; then
  echo "[error] Refusing to overwrite non-empty PD directory:"
  echo "        $PD_DIR"
  exit 4
fi

mkdir -p "$PD_DIR" "$LOG_DIR"

echo "===== INPUT VTI INVENTORY / HASHES ====="
printf '%s\0' "${VTI_FILES[@]}" | xargs -0 sha256sum \
  | tee "$OUT/phase5b2_vti_sha256.txt"

echo
echo "===== DOCKER IMAGE ====="
docker image inspect "$IMAGE" \
  --format 'ID={{.Id}} RepoDigests={{json .RepoDigests}}' \
  | tee "$OUT/phase5b2_docker_image.txt"

echo
echo "===== RUN TTK PERSISTENCE DIAGRAMS ====="

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

echo
echo "===== OUTPUT PD INVENTORY ====="

mapfile -t PD_FILES < <(
  find "$PD_DIR" -maxdepth 1 -type f -name '*_pd_port_0.vtu' | sort
)

printf '%s\n' "${PD_FILES[@]}"

if [[ "${#PD_FILES[@]}" -ne 8 ]]; then
  echo "[error] Expected exactly 8 PD port_0 VTUs; found ${#PD_FILES[@]}"
  exit 5
fi

echo
echo "===== BASIC SCHEMA / PAIR COUNTS ====="

PYTHONNOUSERSITE=1 /usr/bin/python3 - "$PD_DIR" <<'PY'
import sys
from pathlib import Path
import vtk

pd_dir = Path(sys.argv[1])
files = sorted(pd_dir.glob("*_pd_port_0.vtu"))

required_cell = ["PairIdentifier", "PairType", "Persistence", "Birth", "IsFinite"]
required_point = ["ttkVertexScalarField", "CriticalType", "Coordinates"]

for path in files:
    r = vtk.vtkXMLUnstructuredGridReader()
    r.SetFileName(str(path))
    r.Update()
    g = r.GetOutput()

    cd = g.GetCellData()
    pd = g.GetPointData()

    missing_cell = [x for x in required_cell if cd.GetArray(x) is None]
    missing_point = [x for x in required_point if pd.GetArray(x) is None]
    if missing_cell or missing_point:
        raise RuntimeError(
            f"{path}: missing cell={missing_cell}, point={missing_point}"
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
echo "===== FREEZE PD OUTPUT ====="

printf '%s\0' "${PD_FILES[@]}" | xargs -0 sha256sum \
  | tee "$OUT/phase5b2_pd_sha256.txt"

echo
echo "PD count: ${#PD_FILES[@]}"
echo "PHASE 5B2 PD EXTRACTION: PASS"
