#!/usr/bin/env bash
set -euo pipefail

ROOT="${HOME}/PhIRE"
AUDIT="${HOME}/phire_runtime_audit_20260809_221548"
W22="${AUDIT}/recompute_pd_w22"

SRC="${W22}/mt_matching_extractor_src"
BUILD="${W22}/mt_matching_extractor_build_host96"
OUT="${W22}/corrected_pd_mt/discordance_visuals/sample_069/ttk_matching_host96"

mkdir -p "$OUT"

# Never reuse the prior Docker / VTK 9.1 build cache.
rm -rf "$BUILD"
mkdir -p "$BUILD"

echo "================================================================================"
echo "HOST NUMERICAL-DISTANCE ENVIRONMENT"
echo "================================================================================"

/usr/bin/python3 - <<'PY'
import sys
import vtk
import topologytoolkit as ttk

print("python =", sys.executable)
print("vtk =", vtk.vtkVersion.GetVTKVersion())
print("vtk module =", vtk.__file__)
print("ttk module =", ttk.__file__)
PY

echo
echo "TTK VTK linkage:"
ldd /usr/local/lib/libttkMergeTreeDistanceMatrix.so \
  | grep -E 'libvtk.*9\.6' \
  | head -30 \
  || true

echo
echo "================================================================================"
echo "BUILD EXTRACTOR AGAINST HOST VTK 9.6 / TTK"
echo "================================================================================"

cmake \
  -S "$SRC" \
  -B "$BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DVTK_DIR="/usr/local/lib/cmake/vtk-9.6" \
  -DTTKBase_DIR="/usr/local/lib/cmake/ttkBase"

cmake \
  --build "$BUILD" \
  --parallel 2

BIN="${BUILD}/extract_ttk_mt_matching"

if [[ ! -x "$BIN" ]]; then
  echo "ERROR: extractor binary not found: $BIN" >&2
  exit 30
fi

echo
echo "Extractor linkage:"
ldd "$BIN" \
  | grep -E 'libvtk|libttk' \
  | head -80 \
  || true

run_pair() {
  local label="$1"
  local gt_nodes="$2"
  local gt_arcs="$3"
  local sr_nodes="$4"
  local sr_arcs="$5"

  for pp in 0 1; do
    local mode

    if [[ "$pp" == "0" ]]; then
      mode="raw"
    else
      mode="converted"
    fi

    echo
    echo "================================================================================"
    echo "${label}: ${mode}"
    echo "================================================================================"

    "$BIN" \
      --nodes1 "${ROOT}/${gt_nodes}" \
      --arcs1  "${ROOT}/${gt_arcs}" \
      --nodes2 "${ROOT}/${sr_nodes}" \
      --arcs2  "${ROOT}/${sr_arcs}" \
      --postprocess "$pp" \
      --matching-csv "${OUT}/${label}_${mode}_matching.csv" \
      --summary-csv  "${OUT}/${label}_${mode}_summary.csv"
  done
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
echo "VALIDATE AGAINST FROZEN SAMPLE-69 MT DISTANCES"
echo "================================================================================"

export OUT W22

/usr/bin/python3 - <<'PY'
import csv
import os
from pathlib import Path

W22 = Path(os.environ["W22"])
OUT = Path(os.environ["OUT"])

joined = (
    W22
    / "corrected_pd_mt"
    / "corrected_pd_mt_joined.csv"
)

with joined.open(newline="") as f:
    rows = list(csv.DictReader(f))

# The joined table in this audit uses "MT"; tolerate the older/internal name
# "mt_distance" as well.
mt_col = None
for candidate in ("MT", "mt_distance", "mt"):
    if candidate in rows[0]:
        mt_col = candidate
        break

if mt_col is None:
    raise RuntimeError(
        f"Could not identify MT column. Columns={list(rows[0])}"
    )

def find_frozen_row(label):
    aliases = {
        "cnn": [
            "cnn",
            "CNN baseline (pretrained)",
        ],
        "uv": [
            "candidateUV_expanded2688",
            "candidateUV_expanded2688_topology",
            "uv",
            "L_uv",
        ],
        "f1": [
            "candidateF_grad_E2_low_expanded2688",
            "candidateF_grad_E2_low_expanded2688_topology",
            "f1_grad_e2",
            "Candidate F",
        ],
    }[label]

    sample_rows = [
        r
        for r in rows
        if int(r["sample_idx"]) == 69
    ]

    # First prefer exact method identifiers / corrected run names.
    exact_fields = (
        "method_id",
        "corrected_pd_run",
        "original_method_name",
    )

    exact = []
    for r in sample_rows:
        for field in exact_fields:
            val = str(r.get(field, ""))
            if val in aliases:
                exact.append(r)
                break

    if len(exact) == 1:
        return exact[0]

    # Then use unambiguous substring matching across descriptive fields.
    descriptive_fields = (
        "method_id",
        "display_name",
        "corrected_pd_run",
        "original_method_name",
    )

    fuzzy = []
    for r in sample_rows:
        hay = " | ".join(
            str(r.get(field, ""))
            for field in descriptive_fields
        ).lower()

        if any(a.lower() in hay for a in aliases):
            fuzzy.append(r)

    # Deduplicate identical row objects by their field tuple.
    unique = []
    seen = set()

    for r in fuzzy:
        key = tuple(sorted(r.items()))
        if key not in seen:
            seen.add(key)
            unique.append(r)

    if len(unique) != 1:
        print(f"{label}: aliases={aliases}")
        print(f"{label}: candidate frozen rows={len(unique)}")

        for r in unique:
            print({
                k: r.get(k, "")
                for k in (
                    "sample_idx",
                    "method_id",
                    "display_name",
                    "corrected_pd_run",
                    "original_method_name",
                    mt_col,
                )
            })

        raise RuntimeError(
            f"{label}: expected one frozen row, got {len(unique)}"
        )

    return unique[0]

expected = {}

for label in ("cnn", "uv", "f1"):
    row = find_frozen_row(label)
    expected[label] = float(row[mt_col])

    print(
        f"Frozen {label}: "
        f"method_id={row.get('method_id','')} "
        f"{mt_col}={row[mt_col]}"
    )

validation = []

# VTK table output in the historical Python wrapper is float-valued; this
# tolerance is intentionally tight but not bitwise.
TOL = 1e-6

for label in ("cnn", "uv", "f1"):
    vals = {}

    for mode in ("raw", "converted"):
        p = OUT / f"{label}_{mode}_summary.csv"

        with p.open(newline="") as f:
            rr = list(csv.DictReader(f))

        if len(rr) != 1:
            raise RuntimeError(f"Bad summary: {p}")

        vals[mode] = float(rr[0]["distance"])

    e = expected[label]

    raw_diff = abs(vals["raw"] - e)
    conv_diff = abs(vals["converted"] - e)
    cross_diff = abs(vals["raw"] - vals["converted"])

    ok = (
        raw_diff <= TOL
        and conv_diff <= TOL
        and cross_diff <= TOL
    )

    validation.append({
        "label": label,
        "expected_mt": e,
        "raw_mt": vals["raw"],
        "converted_mt": vals["converted"],
        "raw_abs_diff": raw_diff,
        "converted_abs_diff": conv_diff,
        "raw_converted_abs_diff": cross_diff,
        "pass": int(ok),
    })

out_csv = OUT / "sample69_mt_matching_distance_validation.csv"

with out_csv.open("w", newline="") as f:
    w = csv.DictWriter(
        f,
        fieldnames=list(validation[0].keys()),
    )
    w.writeheader()
    w.writerows(validation)

print()
print(
    "label      expected             raw          converted       "
    "|raw-exp|       |conv-exp|      |raw-conv|   PASS"
)
print("-" * 120)

for r in validation:
    print(
        f"{r['label']:>5s}  "
        f"{r['expected_mt']:16.10f} "
        f"{r['raw_mt']:16.10f} "
        f"{r['converted_mt']:16.10f} "
        f"{r['raw_abs_diff']:13.3e} "
        f"{r['converted_abs_diff']:13.3e} "
        f"{r['raw_converted_abs_diff']:13.3e} "
        f"{r['pass']}"
    )

if not all(int(r["pass"]) == 1 for r in validation):
    raise SystemExit(
        "FAIL: numerical reproduction did not pass; "
        "do not interpret matching tuples."
    )

print()
print("ALL THREE SAMPLE-69 MT DISTANCES REPRODUCED: PASS")
print("Matching tuples may proceed to mapping/audit.")
print()
print("Validation CSV:", out_csv)
PY
