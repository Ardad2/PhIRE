#!/usr/bin/env bash

set -euo pipefail

ROOT="${HOME}/PhIRE"
AUDIT="${HOME}/phire_runtime_audit_20260809_221548"
W22="${AUDIT}/recompute_pd_w22"

BASE="${W22}/corrected_pd_mt/discordance_visuals/sample_069"
INPUT_DIR="${BASE}/inputs"
OUTBASE="${BASE}/fixed_threshold_t3"

IMAGE="${IMAGE:-phire-ttk:latest}"

THRESHOLD="3"
ARC_SAMPLING="10"
THREADS="20"

echo "================================================================================"
echo "VERIFY FROZEN INPUTS / THRESHOLD"
echo "================================================================================"

sha256sum -c \
    "${BASE}/authoritative_inputs_sha256.txt"

sha256sum -c \
    "${BASE}/mt_display_threshold_selection_sha256.txt"

mkdir -p "$OUTBASE"


run_method() {

    local method="$1"
    local input="${INPUT_DIR}/${method}_s069.vti"
    local out="${OUTBASE}/${method}"

    if [[ ! -s "$input" ]]; then
        echo "Missing input:"
        echo "  $input"
        exit 1
    fi

    required=(
        "${out}/nodes.vtu"
        "${out}/arcs.vtu"
        "${out}/summary.json"
        "${out}/nodes_display.vtu"
        "${out}/arcs_display.vtu"
        "${out}/display_geometry_report.json"
    )

    complete=1

    for p in "${required[@]}"
    do
        if [[ ! -s "$p" ]]; then
            complete=0
        fi
    done

    if [[ "$complete" -eq 1 ]]; then
        echo
        echo "SKIP ${method}: already complete"
        return
    fi

    rm -rf "$out"
    mkdir -p "$out"

    input_rel="${input#${W22}/}"
    out_rel="${out#${W22}/}"

    echo
    echo "================================================================================"
    echo "METHOD: ${method}"
    echo "DISPLAY THRESHOLD: ${THRESHOLD}"
    echo "================================================================================"

    docker run --rm \
        --user "$(id -u):$(id -g)" \
        -e HOME=/tmp \
        -v "${ROOT}:/work" \
        -v "${W22}:/audit" \
        -w /work \
        "$IMAGE" \
        bash -lc "
            set -euo pipefail

            export PYTHONPATH=\"/usr/local/lib/python3/dist-packages:/opt/ttk/build/lib/python3/dist-packages:/usr/lib/python3/dist-packages:\${PYTHONPATH:-}\"

            python3 \
                scripts/phase2db_extract_simplified_mt.py \
                --input \"/audit/${input_rel}\" \
                --output-dir \"/audit/${out_rel}\" \
                --threshold \"${THRESHOLD}\" \
                --arc-sampling \"${ARC_SAMPLING}\" \
                --threads \"${THREADS}\"
        "

    python3 \
        "${ROOT}/scripts/phase2db_sanitize_mt_geometry.py" \
        --input-vti "$input" \
        --nodes "${out}/nodes.vtu" \
        --arcs "${out}/arcs.vtu" \
        --output-dir "$out"
}


run_method cnn
run_method uv
run_method f1


echo
echo "================================================================================"
echo "SUMMARIZE FIXED-THRESHOLD TREE COMPLEXITY"
echo "================================================================================"

export BASE
export OUTBASE

python3 - <<'PY'
import csv
import os
from pathlib import Path

import vtk


BASE = Path(os.environ["BASE"])
OUTBASE = Path(os.environ["OUTBASE"])


ROOTS = {
    # Reuse the already-generated GT threshold-3 result.
    "gt": BASE / "gt_threshold_sweep" / "t3",

    "cnn": OUTBASE / "cnn",
    "uv": OUTBASE / "uv",
    "f1": OUTBASE / "f1",
}


def read_grid(path):
    if not path.is_file():
        raise FileNotFoundError(path)

    r = vtk.vtkXMLUnstructuredGridReader()
    r.SetFileName(str(path))
    r.Update()

    g = r.GetOutput()

    if g is None:
        raise RuntimeError(f"Failed to read {path}")

    return g


rows = []

for method, root in ROOTS.items():

    nodes = read_grid(root / "nodes.vtu")
    arcs = read_grid(root / "arcs.vtu")

    nodes_display = read_grid(
        root / "nodes_display.vtu"
    )

    arcs_display = read_grid(
        root / "arcs_display.vtu"
    )

    rows.append({
        "method": method,
        "threshold": 3.0,

        "node_points":
            nodes.GetNumberOfPoints(),

        "node_cells":
            nodes.GetNumberOfCells(),

        "arc_points":
            arcs.GetNumberOfPoints(),

        "arc_cells":
            arcs.GetNumberOfCells(),

        "display_node_points":
            nodes_display.GetNumberOfPoints(),

        "display_node_cells":
            nodes_display.GetNumberOfCells(),

        "display_arc_points":
            arcs_display.GetNumberOfPoints(),

        "display_arc_cells":
            arcs_display.GetNumberOfCells(),
    })


out = (
    BASE
    / "sample69_mt_fixed_t3_structure_counts.csv"
)

with out.open("w", newline="") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(rows[0].keys()),
    )

    writer.writeheader()
    writer.writerows(rows)


print()
print(
    "method  nodes  node_cells  arc_points  arc_segments"
)

print("-" * 62)

for r in rows:

    print(
        f"{r['method']:>6s} "
        f"{r['node_points']:6d} "
        f"{r['node_cells']:11d} "
        f"{r['arc_points']:11d} "
        f"{r['arc_cells']:13d}"
    )


print()
print("CSV:")
print(out)

print()
print(
    "No visual threshold changes are permitted after this point "
    "for the sample-69 primary discordance study."
)
PY


echo
echo "================================================================================"
echo "FIXED-THRESHOLD ALL-METHOD EXTRACTION: COMPLETE"
echo "================================================================================"
