#!/usr/bin/env bash
set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PHIRE_ROOT_DEFAULT="$(cd "$SCRIPT_DIR/../.." 2>/dev/null && pwd || true)"
PHIRE_ROOT="${PHIRE_ROOT:-${PHIRE_ROOT_DEFAULT:-$HOME/PhIRE}}"

# If user passes a VTU path, use it. Otherwise pick a default PD VTU automatically.
VTU1="${1:-}"
if [[ -z "${VTU1}" ]]; then
  PD_DIR="$PHIRE_ROOT/ttk_outputs/direct/pd_vtu"
  if [[ -d "$PD_DIR" ]]; then
    VTU1="$(find "$PD_DIR" -maxdepth 1 -type f -name '*.vtu' | head -n 1)"
  else
    VTU1=""
  fi
fi

# Optional second VTU (for “distance” tools if you want to try later)
VTU2="${2:-}"

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
OUTDIR="$SCRIPT_DIR/ttk_vtu_repro_${timestamp}"
mkdir -p "$OUTDIR"/{logs,inputs,bin_info,python}

LOG="$OUTDIR/logs/run.log"
exec > >(tee -a "$LOG") 2>&1

echo "=== REPRO BUNDLE START ==="
echo "SCRIPT_DIR=$SCRIPT_DIR"
echo "PHIRE_ROOT=$PHIRE_ROOT"
echo "OUTDIR=$OUTDIR"
echo "DATE_UTC=$(date -u)"
echo

run() {
  echo
  echo ">>> $*"
  "$@" || echo "!!! Command failed (continuing): $*"
}

run_section() {
  echo
  echo "=============================="
  echo "== $1"
  echo "=============================="
}

need_file_or_die() {
  local f="$1"
  if [[ -z "$f" || ! -f "$f" ]]; then
    echo
    echo "ERROR: VTU file not found."
    echo "Tried VTU1='$f'"
    echo
    echo "Quick help:"
    echo "  find ~/PhIRE -type f -name '*.vtu' | head -n 30"
    echo
    echo "Then rerun with:"
    echo "  ./run_ttk_vtu_bundle.sh /full/path/to/file.vtu"
    exit 2
  fi
}

run_section "SYSTEM"
run uname -a
run uname -m
run bash -lc 'cat /etc/os-release || true'
run bash -lc 'lsb_release -a || true'
run ldd --version
run python3 --version
run which python3
run cmake --version
run gcc --version
run bash -lc 'nvidia-smi || true'
run bash -lc 'nvcc --version || true'
run bash -lc 'ldconfig -p | egrep -i "vtk|ttk|paraview" | head -n 200 || true'

run_section "TTK / VTK / PARAVIEW DISCOVERY"
run bash -lc 'command -v ttkPersistenceDiagramDistance || true'
run bash -lc 'command -v ttkMergeTreeDistance || true'
run bash -lc 'command -v paraview || true'
run bash -lc 'command -v pvpython || true'
run bash -lc 'python3 -c "import vtk; print(\"vtk python:\", vtk.vtkVersion.GetVTKVersion())" || true'
run bash -lc 'python3 -c "import sys; print(sys.version)"'

# If TTK binaries exist, capture file/ldd/help into bin_info
for exe in ttkPersistenceDiagramDistance ttkMergeTreeDistance; do
  p="$(command -v "$exe" 2>/dev/null || true)"
  if [[ -n "$p" ]]; then
    run_section "BIN INFO: $exe"
    run file "$p"
    run ldd "$p"
    run "$p" --help
  fi
done

run_section "INPUT VTU SELECTION"
need_file_or_die "$VTU1"
VTU1_ABS="$(python3 -c "import os; print(os.path.abspath('$VTU1'))")"
echo "VTU1_ABS=$VTU1_ABS"

# If VTU2 not provided, pick a different PD VTU (if available) just for convenience
if [[ -z "$VTU2" ]]; then
  PD_DIR="$(dirname "$VTU1_ABS")"
  VTU2="$(find "$PD_DIR" -maxdepth 1 -type f -name '*.vtu' ! -samefile "$VTU1_ABS" | head -n 1 || true)"
fi
if [[ -n "$VTU2" && -f "$VTU2" ]]; then
  VTU2_ABS="$(python3 -c "import os; print(os.path.abspath('$VTU2'))")"
  echo "VTU2_ABS=$VTU2_ABS"
else
  VTU2_ABS=""
  echo "VTU2_ABS=(none)"
fi

run_section "COPY INPUTS"
run cp -av "$VTU1_ABS" "$OUTDIR/inputs/"
if [[ -n "$VTU2_ABS" ]]; then
  run cp -av "$VTU2_ABS" "$OUTDIR/inputs/"
fi

run_section "VTU QUICK INSPECTION"
for f in "$OUTDIR/inputs/"*.vtu; do
  echo
  echo "--- FILE: $f"
  run ls -lh "$f"
  run stat "$f"
  run sha256sum "$f"
  run file "$f"
  echo
  echo "Header (first ~40 lines):"
  run bash -lc "head -n 40 \"$f\""
  echo
  echo "AppendedData lines (if any):"
  run bash -lc "grep -n \"AppendedData\" -n \"$f\" || true"
  echo
  echo "Snippet around AppendedData (if any):"
  run bash -lc "grep -n -C 3 \"AppendedData\" \"$f\" || true"
done

run_section "YOUR PYTHON DEBUG SCRIPTS (IF PRESENT)"
# Copy these scripts into the bundle too (if they exist)
for s in read_vtu_debug.py check_vtu_bytes.py check_vtu_appended_base64.py; do
  if [[ -f "$SCRIPT_DIR/$s" ]]; then
    run cp -av "$SCRIPT_DIR/$s" "$OUTDIR/python/"
  fi
done

for f in "$OUTDIR/inputs/"*.vtu; do
  echo
  echo "--- Running checks on: $f"

  if [[ -f "$SCRIPT_DIR/read_vtu_debug.py" ]]; then
    run python3 "$SCRIPT_DIR/read_vtu_debug.py" "$f"
  else
    echo "(read_vtu_debug.py not found; doing inline VTK read attempt)"
    run python3 - <<PY
import sys
try:
    import vtk
    r = vtk.vtkXMLUnstructuredGridReader()
    r.SetFileName(sys.argv[1])
    ok = r.CanReadFile(sys.argv[1])
    print("CanReadFile:", ok)
    r.Update()
    out = r.GetOutput()
    if out is None:
        print("Output: None")
    else:
        print("Points:", out.GetNumberOfPoints(), "Cells:", out.GetNumberOfCells())
except Exception as e:
    print("VTK read exception:", repr(e))
PY "$f"
  fi

  if [[ -f "$SCRIPT_DIR/check_vtu_bytes.py" ]]; then
    run python3 "$SCRIPT_DIR/check_vtu_bytes.py" "$f"
  fi

  if [[ -f "$SCRIPT_DIR/check_vtu_appended_base64.py" ]]; then
    run python3 "$SCRIPT_DIR/check_vtu_appended_base64.py" "$f"
  fi
done

run_section "OPTIONAL: TRY DISTANCE BINARIES (LOG ONLY)"
# We don’t guess flags; we only run --help above so developers can see expected CLI.
# If you want to attempt a run anyway, uncomment and edit once you see the help output.
echo "See bin_info sections above for '--help' output."
echo "If you want, you can manually try commands and append logs into: $LOG"

run_section "BUNDLE"
TARBALL="$SCRIPT_DIR/ttk_vtu_repro_${timestamp}.tar.gz"
run tar -czf "$TARBALL" -C "$SCRIPT_DIR" "$(basename "$OUTDIR")"
echo
echo "=== DONE ==="
echo "Bundle directory: $OUTDIR"
echo "Tarball: $TARBALL"
