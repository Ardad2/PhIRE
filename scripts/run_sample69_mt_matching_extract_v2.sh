#!/usr/bin/env bash
set -euo pipefail

ROOT="${HOME}/PhIRE"
AUDIT="${HOME}/phire_runtime_audit_20260809_221548"
W22="${AUDIT}/recompute_pd_w22"

SRC="${W22}/mt_matching_extractor_src"
BUILD="${W22}/mt_matching_extractor_build"
OUT="${W22}/corrected_pd_mt/discordance_visuals/sample_069/ttk_matching"

IMAGE="${IMAGE:-phire-ttk}"

mkdir -p "$OUT"

# IMPORTANT:
# The container has both distro VTK 9.1 and the /usr/local VTK used by TTK.
# Never reuse a CMake cache that may have selected the distro VTK.
rm -rf "$BUILD"
mkdir -p "$BUILD"

echo "================================================================================"
echo "BUILD EXTRACTOR"
echo "================================================================================"

docker run --rm \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -v "${ROOT}:/work" \
  -v "${W22}:/audit" \
  "$IMAGE" \
  bash -lc '
    set -euo pipefail

    echo "Locating the VTK CMake package installed alongside the audited TTK build..."

    VTK_CONFIG="$(
      find /usr/local /opt \
        -type f \
        \( -name "VTKConfig.cmake" -o -name "vtk-config.cmake" \) \
        2>/dev/null \
        | sort \
        | head -1
    )"

    if [[ -z "${VTK_CONFIG}" ]]; then
      echo "ERROR: no /usr/local or /opt VTKConfig.cmake found." >&2
      echo "TTK runtime linkage:" >&2
      ldd /usr/local/lib/libttkMergeTreeDistanceMatrix.so \
        | grep -i vtk \
        || true
      exit 20
    fi

    VTK_DIR="$(dirname "${VTK_CONFIG}")"

    echo "Using VTK_CONFIG=${VTK_CONFIG}"
    echo "Using VTK_DIR=${VTK_DIR}"

    # Reject the distro VTK 9.1 package explicitly. TTK in this audit was
    # built against the /usr/local VTK installation.
    case "${VTK_DIR}" in
      /usr/lib/*/cmake/vtk-9.1*|/usr/lib/cmake/vtk-9.1*)
        echo "ERROR: resolved the wrong distro VTK 9.1 package: ${VTK_DIR}" >&2
        exit 21
        ;;
    esac

    cmake \
      -S /audit/mt_matching_extractor_src \
      -B /audit/mt_matching_extractor_build \
      -DCMAKE_BUILD_TYPE=Release \
      -DVTK_DIR="${VTK_DIR}" \
      -DCMAKE_PREFIX_PATH="/usr/local/lib/cmake/ttkVTK;/usr/local/lib/cmake/ttkBase;${VTK_DIR}"

    cmake \
      --build /audit/mt_matching_extractor_build \
      --parallel 2
  '

BIN="/audit/mt_matching_extractor_build/extract_ttk_mt_matching"

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

    docker run --rm \
      --user "$(id -u):$(id -g)" \
      -e HOME=/tmp \
      -v "${ROOT}:/work" \
      -v "${W22}:/audit" \
      "$IMAGE" \
      "$BIN" \
        --nodes1 "/work/${gt_nodes}" \
        --arcs1  "/work/${gt_arcs}" \
        --nodes2 "/work/${sr_nodes}" \
        --arcs2  "/work/${sr_arcs}" \
        --postprocess "$pp" \
        --matching-csv "/audit/corrected_pd_mt/discordance_visuals/sample_069/ttk_matching/${label}_${mode}_matching.csv" \
        --summary-csv  "/audit/corrected_pd_mt/discordance_visuals/sample_069/ttk_matching/${label}_${mode}_summary.csv"
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
echo "VALIDATE AGAINST FROZEN AUDITED SAMPLE-69 MT DISTANCES"
echo "================================================================================"

export OUT W22

python3 - <<'PY'
import csv
import math
import os
from pathlib import Path

W22 = Path(os.environ["W22"])
OUT = Path(os.environ["OUT"])

joined = (
    W22
    / "corrected_pd_mt"
    / "corrected_pd_mt_joined.csv"
)

wanted = {
    "cnn": "cnn",
    "uv": "uv",
    "f1": "f1_grad_e2",
}

with joined.open(newline="") as f:
    rows = list(csv.DictReader(f))

expected = {}

for label, method_id in wanted.items():
    matches = [
        r for r in rows
        if (
            r["method_id"] == method_id
            and int(r["sample_idx"]) == 69
        )
    ]

    if len(matches) != 1:
        raise RuntimeError(
            f"{label}: expected one frozen row, got {len(matches)}"
        )

    expected[label] = float(matches[0]["mt_distance"])

validation = []

TOL = 1e-8

for label in wanted:
    vals = {}

    for mode in ["raw", "converted"]:
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
        "FAIL: do not interpret matching tuples."
    )

print()
print("ALL THREE SAMPLE-69 MT DISTANCES REPRODUCED: PASS")
print("Matching tuples may proceed to the next mapping/audit stage.")
print()
print("Validation CSV:", out_csv)
PY
