#!/usr/bin/env bash
set -euo pipefail

REPO="${HOME}/PhIRE"
INPUT_ROOT="/tmp/phase2db_mt_batch_inputs"
STAGING_REL="ttk_runs_fixed/unified_candidate_analysis/phase2db/_staging/spark_simplified_mt_all"
STAGING="${REPO}/${STAGING_REL}"
PACKAGE_ROOT="/tmp/phase2db_mt_precomputed_geometry"
ARCHIVE="/tmp/phase2db_mt_precomputed_geometry.tar.gz"
IMAGE="phire-ttk:latest"

EXTRACTOR="${REPO}/scripts/phase2db_extract_simplified_mt.py"
SANITIZER="${REPO}/scripts/phase2db_sanitize_mt_geometry.py"

cd "$REPO"

for required in \
  "$INPUT_ROOT/manifest.json" \
  "$EXTRACTOR" \
  "$SANITIZER"
do
  if [[ ! -f "$required" ]]; then
    echo "Missing required file: $required" >&2
    exit 1
  fi
done

if [[ -e "$STAGING" ]]; then
  echo "Resume mode: using existing staging directory:"
  echo "  $STAGING"
fi

if [[ -e "$PACKAGE_ROOT" || -e "$ARCHIVE" ]]; then
  echo "Refusing to overwrite existing package output:" >&2
  echo "  $PACKAGE_ROOT" >&2
  echo "  $ARCHIVE" >&2
  exit 1
fi

mkdir -p "$STAGING/logs"

PANELS=(
  "1:GT"
  "1:cnn"
  "1:gan"
  "1:candidate_c"
  "1:f3_grad_crit"
  "1:f2_grad_levelset_e2"
  "1:uv_e2"
  "2:GT"
  "2:bicubic"
  "2:cnn"
  "2:gan"
  "3:GT"
  "3:cnn"
  "3:f3_grad_crit"
  "3:uv_e2"
  "5:GT"
  "5:cnn"
  "5:candidate_c"
  "5:f3_grad_crit"
  "5:f2_grad_levelset_e2"
  "5:uv_e2"
)

completed=0

for spec in "${PANELS[@]}"; do
  figure="${spec%%:*}"
  method="${spec#*:}"
  figure_padded="$(printf '%02d' "$figure")"
  input_host="$INPUT_ROOT/inputs/figure_${figure_padded}/${method}.vti"
  output_rel="$STAGING_REL/figure_${figure_padded}/${method}"
  output_host="$REPO/$output_rel"
  log="$STAGING/logs/figure_${figure_padded}_${method}.log"

  if [[ ! -f "$input_host" ]]; then
    echo "Missing input for $spec: $input_host" >&2
    exit 1
  fi

  required_outputs=(
    "$output_host/simplified_p11.vti"
    "$output_host/segmentation.vti"
    "$output_host/summary.json"
    "$output_host/nodes_display.vtu"
    "$output_host/arcs_display.vtu"
    "$output_host/display_geometry_report.json"
  )

  panel_complete=1

  for required in "${required_outputs[@]}"; do
    if [[ ! -s "$required" ]]; then
      panel_complete=0
      break
    fi
  done

  if [[ "$panel_complete" -eq 1 ]]; then
    completed=$((completed + 1))
    echo "SKIP: $spec already complete ($completed/21)"
    continue
  fi

  # Remove only this panel's incomplete staging directory.
  rm -rf "$output_host"
  mkdir -p "$output_host"

  echo
  echo "======================================================================"
  echo "Extracting $spec"
  echo "Log: $log"
  echo "======================================================================"

  docker run --rm \
    --user "$(id -u):$(id -g)" \
    -e HOME=/tmp \
    -v "$REPO:/work" \
    -v "$INPUT_ROOT:/inputs:ro" \
    -w /work \
    "$IMAGE" \
    bash -lc "
set -e
export PYTHONPATH=\"/usr/local/lib/python3/dist-packages:/opt/ttk/build/lib/python3/dist-packages:/usr/lib/python3/dist-packages:\${PYTHONPATH:-}\"

python3 scripts/phase2db_extract_simplified_mt.py \
  --input /inputs/inputs/figure_${figure_padded}/${method}.vti \
  --output-dir /work/${output_rel} \
  --threshold 11.0 \
  --arc-sampling 10 \
  --threads 20
" >"$log" 2>&1

  python3 "$SANITIZER" \
    --input-vti "$input_host" \
    --nodes "$output_host/nodes.vtu" \
    --arcs "$output_host/arcs.vtu" \
    --output-dir "$output_host" \
    >>"$log" 2>&1

  for required in \
    "$output_host/simplified_p11.vti" \
    "$output_host/segmentation.vti" \
    "$output_host/summary.json" \
    "$output_host/nodes_display.vtu" \
    "$output_host/arcs_display.vtu" \
    "$output_host/display_geometry_report.json"
  do
    if [[ ! -s "$required" ]]; then
      echo "FAILED: $spec did not produce $required" >&2
      tail -n 120 "$log" >&2 || true
      exit 1
    fi
  done

  completed=$((completed + 1))
  echo "PASS: $spec ($completed/21)"
done

python3 - <<'PY'
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

repo = Path.home() / "PhIRE"
input_root = Path("/tmp/phase2db_mt_batch_inputs")
staging = (
    repo
    / "ttk_runs_fixed"
    / "unified_candidate_analysis"
    / "phase2db"
    / "_staging"
    / "spark_simplified_mt_all"
)
package_root = Path("/tmp/phase2db_mt_precomputed_geometry")

manifest = json.loads((input_root / "manifest.json").read_text(encoding="utf-8"))
panels = manifest["panels"]
assert len(panels) == 21

package_root.mkdir(parents=True)

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

for panel in panels:
    figure_id = int(panel["figure_id"])
    method_id = str(panel["method_id"])
    figure_dir = f"figure_{figure_id:02d}"
    source_dir = staging / figure_dir / method_id
    destination_dir = package_root / "geometry" / figure_dir / method_id
    destination_dir.mkdir(parents=True)

    copied = {}
    for name in (
        "nodes_display.vtu",
        "arcs_display.vtu",
        "summary.json",
        "display_geometry_report.json",
    ):
        source = source_dir / name
        destination = destination_dir / name
        shutil.copy2(source, destination)
        copied[name] = {
            "package_path": destination.relative_to(package_root).as_posix(),
            "sha256": sha256(destination),
            "bytes": destination.stat().st_size,
        }

    final_source_root = (
        "ttk_runs_fixed/unified_candidate_analysis/phase2db/"
        f"manual_topology_sources/{figure_dir}/{method_id}"
    )

    panel["nodes_display_package_path"] = copied["nodes_display.vtu"]["package_path"]
    panel["arcs_display_package_path"] = copied["arcs_display.vtu"]["package_path"]
    panel["geometry_summary_package_path"] = copied["summary.json"]["package_path"]
    panel["geometry_report_package_path"] = copied["display_geometry_report.json"]["package_path"]
    panel["nodes_display_sha256"] = copied["nodes_display.vtu"]["sha256"]
    panel["arcs_display_sha256"] = copied["arcs_display.vtu"]["sha256"]
    panel["source_vtu_path"] = f"{final_source_root}/arcs_display.vtu"
    panel["source_nodes_vtu_path"] = f"{final_source_root}/nodes_display.vtu"

output_manifest = {
    "schema_version": 2,
    "panel_count": len(panels),
    "geometry_computation": {
        "environment": "Spark phire-ttk:latest",
        "vtk_version": "9.1.0",
        "ttk_version": "1.3.0",
        "persistence_threshold": 11.0,
        "threshold_is_absolute": True,
        "pair_type": "EXTREMUM_SADDLE",
        "compute_perturbation": False,
        "merge_tree_backend": "FTM",
        "tree_type": "Join",
        "arc_sampling": 10,
        "with_normalize": True,
        "with_advanced_statistics": True,
        "display_geometry_arrays_removed": True,
    },
    "panels": panels,
}

(package_root / "manifest.json").write_text(
    json.dumps(output_manifest, indent=2) + "\n",
    encoding="utf-8",
)

print(f"Wrote geometry manifest with {len(panels)} panels")
PY

tar -czf "$ARCHIVE" \
  -C /tmp \
  phase2db_mt_precomputed_geometry

echo
echo "======================================================================"
echo "SPARK PRECOMPUTED GEOMETRY BATCH COMPLETE"
echo "Panels:  21"
echo "Archive: $ARCHIVE"
echo "Bytes:   $(stat -c %s "$ARCHIVE")"
echo "SHA-256: $(sha256sum "$ARCHIVE" | awk '{print $1}')"
echo "======================================================================"
