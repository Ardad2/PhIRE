#!/usr/bin/env bash
# Phase 6 isolated topology extraction pilot, sample 0 only.
# Usage (from ~/PhIRE): bash scripts/phase6_topology_s0_extraction_pilot.sh
# No legacy PD/MT distance is computed. No prior model/baseline outputs touched.
set -euo pipefail

[[ -f scripts/convert_phire_to_vti.py ]] || { echo '[ERROR] Run from PhIRE repository root'; exit 2; }
METHOD=candidateF_grad_pairzero_expanded2688
IMAGE=phire-ttk:latest
DATA_DIR="data_out/wind_finetune_${METHOD}"
ROOT="ttk_runs_fixed/topology_finetuning/${METHOD}_s0_extraction_pilot_20261014"
VTI_DIR="$ROOT/vti"
PD_DIR="$ROOT/pd"
MT_DIR="$ROOT/mt"

for f in dataGT.npy dataSR.npy idx.npy; do
  [[ -f "$DATA_DIR/$f" ]] || { echo "[ERROR] Missing $DATA_DIR/$f"; exit 2; }
done
docker image inspect "$IMAGE" --format 'Docker image: {{.Id}}'
mkdir -p "$VTI_DIR" "$PD_DIR" "$MT_DIR"
printf '[PILOT] Method=%s\n[PILOT] Root=%s\n' "$METHOD" "$ROOT"

for SIDE in GT SR; do
  VTI="${VTI_DIR}/${METHOD}_${SIDE}_s0_speed_p160_x0_y0.vti"
  SOURCE="${DATA_DIR}/data${SIDE}.npy"
  if [[ ! -s "$VTI" ]]; then
    echo "[VTI] Creating $VTI"
    docker run --rm -v "$PWD":/work -w /work "$IMAGE" \
      bash -lc "python scripts/convert_phire_to_vti.py \
      --input '$SOURCE' --outdir '$VTI_DIR' --label '${METHOD}_${SIDE}' \
      --scalar speed --patch 160 --x0 0 --y0 0 --samples 0"
  else
    echo "[VTI] Existing nonempty file: $VTI"
  fi
done

# Verify readback against the exact speed patch, including array name and
# orientation. No reliance on the host Python's NumPy/scikit-image ABI.
echo '[CHECK] VTI numerical round-trip and x/y orientation'
docker run --rm -i -v "$PWD":/work -w /work \
  -e PILOT_ROOT="$ROOT" -e METHOD="$METHOD" -e DATA_DIR="$DATA_DIR" \
  "$IMAGE" python - <<'PY'
import os
from pathlib import Path
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

root = Path(os.environ['PILOT_ROOT'])
method = os.environ['METHOD']
data_dir = Path(os.environ['DATA_DIR'])
for side in ('GT', 'SR'):
    fn = root / 'vti' / f'{method}_{side}_s0_speed_p160_x0_y0.vti'
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(str(fn))
    reader.Update()
    image = reader.GetOutput()
    assert image.GetDimensions() == (160, 160, 1), (side, image.GetDimensions())
    arr = image.GetPointData().GetArray('wind_speed')
    assert arr is not None, f'{side}: missing wind_speed point array'
    measured = vtk_to_numpy(arr).reshape((160, 160))
    raw = np.load(data_dir / f'data{side}.npy', mmap_mode='r')[0, :160, :160, :]
    u = np.asarray(raw[..., 0], dtype=np.float32)
    v = np.asarray(raw[..., 1], dtype=np.float32)
    expected = np.sqrt(u * u + v * v)
    diff = float(np.max(np.abs(measured - expected)))
    np.testing.assert_allclose(measured, expected, rtol=1e-6, atol=2e-5)
    print(f'[PASS] {side}: (160,160), wind_speed, orientation; max_abs_diff={diff:.6g}')
PY

# Extract exactly two samples (GT and SR for index 0), each with PD and MT.
# Avoid the historical distance stage in run_candidate_topology_pipeline.sh.
for SIDE in GT SR; do
  BASE="${METHOD}_${SIDE}_s0_speed_p160_x0_y0"
  PD_FILE="$PD_DIR/${BASE}_pd_port_0.vtu"
  MT0="$MT_DIR/${BASE}_mt_port_0.vtu"
  MT1="$MT_DIR/${BASE}_mt_port_1.vtu"
  MT2="$MT_DIR/${BASE}_mt_port_2.vti"
  if [[ ! -s "$PD_FILE" ]]; then
    echo "[TTK] PD $BASE"
    docker run --rm -v "$PWD":/work -w /work \
      -e OMP_NUM_THREADS=1 -e TTK_NUM_THREADS=1 "$IMAGE" \
      bash -lc "ttkPersistenceDiagramCmd -t 1 -i '$VTI_DIR/$BASE.vti' \
      -a wind_speed -o '$PD_DIR/${BASE}_pd'"
  else
    echo "[TTK] Skipping existing PD $BASE"
  fi
  if [[ ! -s "$MT0" || ! -s "$MT1" || ! -s "$MT2" ]]; then
    echo "[TTK] MT $BASE"
    docker run --rm -v "$PWD":/work -w /work \
      -e OMP_NUM_THREADS=1 -e TTK_NUM_THREADS=1 "$IMAGE" \
      bash -lc "ttkMergeTreeCmd -t 1 -i '$VTI_DIR/$BASE.vti' \
      -a wind_speed -o '$MT_DIR/${BASE}_mt'"
  else
    echo "[TTK] Skipping existing MT $BASE"
  fi
  for f in "$PD_FILE" "$MT0" "$MT1" "$MT2"; do
    [[ -s "$f" ]] || { echo "[ERROR] Missing/empty TTK output: $f"; exit 1; }
    ls -lh "$f"
  done
done

echo '[PILOT] PASS: GT and SR sample 0 VTI, PD and three MT ports present.'
echo '[PILOT] Historical TTK distances deliberately NOT computed.'
echo "[PILOT] Files in: $ROOT"
