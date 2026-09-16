#!/usr/bin/env bash
set -euo pipefail

ROOT="${HOME}/PhIRE"
AUDIT="${HOME}/phire_runtime_audit_20260809_221548"
W22="${AUDIT}/recompute_pd_w22"
BIN="${W22}/mt_matching_extractor_build_host96/extract_ttk_mt_matching"
OUT="${1:-${ROOT}/join_tree_harness/phase4e_sample127_ttk_gt_distance}"

CNN_N="${ROOT}/ttk_runs_fixed/cnn/mt/cnn_GT_s127_speed_p160_x0_y0_mt_port_0.vtu"
CNN_A="${ROOT}/ttk_runs_fixed/cnn/mt/cnn_GT_s127_speed_p160_x0_y0_mt_port_1.vtu"

UV_N="${ROOT}/ttk_runs_fixed/topology_finetuning/candidateUV_expanded2688_topology/mt/GT/candidateUV_expanded2688_GT_s127_speed_p160_x0_y0_mt_port_0.vtu"
UV_A="${ROOT}/ttk_runs_fixed/topology_finetuning/candidateUV_expanded2688_topology/mt/GT/candidateUV_expanded2688_GT_s127_speed_p160_x0_y0_mt_port_1.vtu"

mkdir -p "$OUT"

echo "===== PHASE 4E SAMPLE-127 CNN-GT VS UV-GT TTK DISTANCE ====="
date -Is
hostname
echo

echo "===== HOST ENVIRONMENT ====="
/usr/bin/python3 - <<'PY'
import sys
import vtk
import topologytoolkit as ttk
print("python =", sys.executable)
print("vtk =", vtk.vtkVersion.GetVTKVersion())
print("vtk module =", vtk.__file__)
print("ttk module =", ttk.__file__)
print("ttk version =", getattr(ttk, "__version__", "n/a"))
PY

echo
echo "===== EXTRACTOR / INPUT HASHES ====="
sha256sum "$BIN" "$CNN_N" "$CNN_A" "$UV_N" "$UV_A"

echo
echo "===== EXTRACTOR LINKAGE ====="
ldd "$BIN" | grep -E 'libvtk|libttk' | head -80 || true

for pp in 0 1; do
  if [[ "$pp" == "0" ]]; then
    mode="raw"
  else
    mode="converted"
  fi

  echo
  echo "===== RUN: ${mode} / postprocess=${pp} ====="

  "$BIN" \
    --nodes1 "$CNN_N" \
    --arcs1  "$CNN_A" \
    --nodes2 "$UV_N" \
    --arcs2  "$UV_A" \
    --postprocess "$pp" \
    --matching-csv "$OUT/sample127_cnnGT_vs_uvGT_${mode}_matching.csv" \
    --summary-csv  "$OUT/sample127_cnnGT_vs_uvGT_${mode}_summary.csv" \
    2>&1 | tee "$OUT/sample127_cnnGT_vs_uvGT_${mode}.log"

  echo
  echo "----- ${mode} summary.csv -----"
  cat "$OUT/sample127_cnnGT_vs_uvGT_${mode}_summary.csv"

  echo
  echo "----- ${mode} matching rows -----"
  wc -l "$OUT/sample127_cnnGT_vs_uvGT_${mode}_matching.csv"
done

echo
echo "===== FREEZE PHASE 4E OUTPUT ====="
find "$OUT" -type f ! -name sha256_manifest.txt -print0 \
  | sort -z | xargs -0 sha256sum > "$OUT/sha256_manifest.txt"

cat "$OUT/sha256_manifest.txt"
