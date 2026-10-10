#!/usr/bin/env bash
set -euo pipefail

ROOT="$HOME/PhIRE"
AUDIT="$HOME/phire_runtime_audit_20260809_221548"
W22="$AUDIT/recompute_pd_w22"

SRC="$W22/mt_cost_attribution_src"
BUILD="$W22/mt_cost_attribution_build_host96"
OUT="$W22/corrected_pd_mt/discordance_visuals/sample_069/ttk_matching_host96/cost_attribution"

rm -rf "$BUILD"
mkdir -p "$BUILD" "$OUT"

echo "================================================================================"
echo "BUILD NODEWISE MT COST ATTRIBUTION"
echo "================================================================================"

cmake \
  -S "$SRC" \
  -B "$BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DVTK_DIR="/usr/local/lib/cmake/vtk-9.6" \
  -DTTKBase_DIR="/usr/local/lib/cmake/ttkBase"

cmake --build "$BUILD" --parallel 2

BIN="$BUILD/extract_ttk_mt_cost_attribution"

run_pair() {
  local label="$1"
  local n1="$2"
  local a1="$3"
  local n2="$4"
  local a2="$5"

  echo
  echo "================================================================================"
  echo "$label"
  echo "================================================================================"

  "$BIN" \
    --nodes1 "$ROOT/$n1" \
    --arcs1 "$ROOT/$a1" \
    --nodes2 "$ROOT/$n2" \
    --arcs2 "$ROOT/$a2" \
    --summary-csv "$OUT/${label}_cost_attribution_summary.csv" \
    --unmatched-csv "$OUT/${label}_unmatched_branches.csv" \
    --matching-csv "$OUT/${label}_raw_matching_manual_preprocess.csv"
}

run_pair \
  cnn \
  "ttk_runs_fixed/cnn/mt/cnn_GT_s69_speed_p160_x0_y0_mt_port_0.vtu" \
  "ttk_runs_fixed/cnn/mt/cnn_GT_s69_speed_p160_x0_y0_mt_port_1.vtu" \
  "ttk_runs_fixed/cnn/mt/cnn_SR_s69_speed_p160_x0_y0_mt_port_0.vtu" \
  "ttk_runs_fixed/cnn/mt/cnn_SR_s69_speed_p160_x0_y0_mt_port_1.vtu"

run_pair \
  uv \
  "ttk_runs_fixed/topology_finetuning/candidateUV_expanded2688_topology/mt/GT/candidateUV_expanded2688_GT_s69_speed_p160_x0_y0_mt_port_0.vtu" \
  "ttk_runs_fixed/topology_finetuning/candidateUV_expanded2688_topology/mt/GT/candidateUV_expanded2688_GT_s69_speed_p160_x0_y0_mt_port_1.vtu" \
  "ttk_runs_fixed/topology_finetuning/candidateUV_expanded2688_topology/mt/SR/candidateUV_expanded2688_SR_s69_speed_p160_x0_y0_mt_port_0.vtu" \
  "ttk_runs_fixed/topology_finetuning/candidateUV_expanded2688_topology/mt/SR/candidateUV_expanded2688_SR_s69_speed_p160_x0_y0_mt_port_1.vtu"

run_pair \
  f1 \
  "ttk_runs_fixed/topology_finetuning/candidateF_grad_E2_low_expanded2688_topology/mt/GT/candidateF_grad_E2_low_expanded2688_GT_s69_speed_p160_x0_y0_mt_port_0.vtu" \
  "ttk_runs_fixed/topology_finetuning/candidateF_grad_E2_low_expanded2688_topology/mt/GT/candidateF_grad_E2_low_expanded2688_GT_s69_speed_p160_x0_y0_mt_port_1.vtu" \
  "ttk_runs_fixed/topology_finetuning/candidateF_grad_E2_low_expanded2688_topology/mt/SR/candidateF_grad_E2_low_expanded2688_SR_s69_speed_p160_x0_y0_mt_port_0.vtu" \
  "ttk_runs_fixed/topology_finetuning/candidateF_grad_E2_low_expanded2688_topology/mt/SR/candidateF_grad_E2_low_expanded2688_SR_s69_speed_p160_x0_y0_mt_port_1.vtu"

echo
echo "================================================================================"
echo "SUMMARIES"
echo "================================================================================"
cat "$OUT/"*_cost_attribution_summary.csv

echo
echo "================================================================================"
echo "TOP 12 INDIVIDUAL UNMATCHED BRANCH COSTS PER METHOD"
echo "================================================================================"

export OUT
/usr/bin/python3 - <<'PY'
import csv, os
from pathlib import Path

out = Path(os.environ["OUT"])

for label in ("cnn", "uv", "f1"):
    p = out / f"{label}_unmatched_branches.csv"
    with p.open(newline="") as f:
        rows = list(csv.DictReader(f))

    rows.sort(key=lambda r: float(r["nonmatching_cost"]), reverse=True)

    print()
    print(label.upper())
    print("-" * 88)
    print("rank side      node origin root persistence_raw nonmatching_cost")
    for rank, r in enumerate(rows[:12], 1):
        print(
            f"{rank:>4d} "
            f"{r['side']:<9s} "
            f"{int(r['node_id']):>5d} "
            f"{int(r['origin_id']):>6d} "
            f"{int(r['is_root']):>4d} "
            f"{float(r['persistence_raw']):>15.8g} "
            f"{float(r['nonmatching_cost']):>16.10g}"
        )
PY

echo
echo "ALL NODEWISE DELETE/INSERT RECOMPOSITION CHECKS PASSED."
