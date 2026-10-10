#!/usr/bin/env bash

set -euo pipefail

ROOT="${HOME}/PhIRE"
AUDIT="${HOME}/phire_runtime_audit_20260809_221548"
W22="${AUDIT}/recompute_pd_w22"

SAMPLE=69

BASE="${W22}/corrected_pd_mt/discordance_visuals/sample_069"
INPUT="${BASE}/inputs/gt_s069.vti"
SWEEP="${BASE}/gt_threshold_sweep"

IMAGE="${IMAGE:-phire-ttk:latest}"

THRESHOLDS=(
    0
    1
    2
    3
)


echo "================================================================================"
echo "STEP 1: PREPARE AUTHORITATIVE INPUTS"
echo "================================================================================"

python3 \
    "${W22}/prepare_sample69_mt_authoritative_inputs.py"


if [[ ! -s "$INPUT" ]]; then
    echo "Missing authoritative GT VTI:"
    echo "  $INPUT"
    exit 1
fi


mkdir -p "$SWEEP"


echo
echo "================================================================================"
echo "STEP 2: GT-ONLY MERGE-TREE DISPLAY-THRESHOLD SWEEP"
echo "================================================================================"

echo "Important:"
echo "  These thresholds are DISPLAY / simplification thresholds only."
echo "  They do NOT modify the already-audited numerical MT distance."
echo


for threshold in "${THRESHOLDS[@]}"
do

    label="t${threshold//./p}"

    OUT="${SWEEP}/${label}"

    REQUIRED=(
        "${OUT}/nodes.vtu"
        "${OUT}/arcs.vtu"
        "${OUT}/summary.json"
        "${OUT}/nodes_display.vtu"
        "${OUT}/arcs_display.vtu"
        "${OUT}/display_geometry_report.json"
    )

    complete=1

    for p in "${REQUIRED[@]}"
    do
        if [[ ! -s "$p" ]]; then
            complete=0
        fi
    done

    if [[ "$complete" -eq 1 ]]; then
        echo
        echo "SKIP threshold=${threshold}: already complete"
        continue
    fi

    rm -rf "$OUT"
    mkdir -p "$OUT"

    input_rel="${INPUT#${W22}/}"
    out_rel="${OUT#${W22}/}"

    echo
    echo "--------------------------------------------------------------------------------"
    echo "GT threshold = ${threshold}"
    echo "--------------------------------------------------------------------------------"

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
                --threshold \"${threshold}\" \
                --arc-sampling 10 \
                --threads 20
        "

    python3 \
        "${ROOT}/scripts/phase2db_sanitize_mt_geometry.py" \
        --input-vti "$INPUT" \
        --nodes "${OUT}/nodes.vtu" \
        --arcs "${OUT}/arcs.vtu" \
        --output-dir "$OUT"

done


echo
echo "================================================================================"
echo "STEP 3: SUMMARIZE GT TREE COMPLEXITY"
echo "================================================================================"

export BASE

python3 - <<'PY'
import csv
import json
import os
from pathlib import Path

import vtk


BASE = Path(os.environ["BASE"])

SWEEP = (
    BASE
    / "gt_threshold_sweep"
)

THRESHOLDS = [
    ("0", "t0"),
    ("1", "t1"),
    ("2", "t2"),
    ("3", "t3"),
]


def read_grid(path):

    r = vtk.vtkXMLUnstructuredGridReader()
    r.SetFileName(str(path))
    r.Update()

    g = r.GetOutput()

    if (
        g is None
        or g.GetNumberOfPoints() == 0
    ):
        raise RuntimeError(
            f"Could not read usable VTU: {path}"
        )

    return g


rows = []

for threshold, label in THRESHOLDS:

    root = SWEEP / label

    nodes = read_grid(
        root / "nodes.vtu"
    )

    arcs = read_grid(
        root / "arcs.vtu"
    )

    nodes_display = read_grid(
        root / "nodes_display.vtu"
    )

    arcs_display = read_grid(
        root / "arcs_display.vtu"
    )

    row = {
        "threshold": threshold,

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
    }

    rows.append(row)


out_csv = (
    BASE
    / "gt_mt_display_threshold_sweep.csv"
)

with out_csv.open(
    "w",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(rows[0].keys()),
    )

    writer.writeheader()
    writer.writerows(rows)


print()
print(
    "threshold "
    "nodes "
    "node_cells "
    "arc_points "
    "arc_segments"
)

print("-" * 70)

for r in rows:

    print(
        f"{r['threshold']:>9s} "
        f"{r['node_points']:>5d} "
        f"{r['node_cells']:>10d} "
        f"{r['arc_points']:>10d} "
        f"{r['arc_cells']:>12d}"
    )


print()
print("CSV:")
print(out_csv)

print()
print(
    "NOTE: arc_cells are rendered/sampled line segments, "
    "not necessarily the number of logical merge-tree branches."
)

print()
print(
    "Do not inspect CNN / Ablation / Candidate-F tree geometry yet. "
    "First freeze the display threshold using GT complexity only."
)
PY


echo
echo "================================================================================"
echo "GT THRESHOLD SWEEP: COMPLETE"
echo "================================================================================"
