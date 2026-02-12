#!/usr/bin/env bash
set -euo pipefail

OUTDIR="${1:-ttk_vtu_repro_$(date -u +%Y%m%dT%H%M%SZ)}"
VTU_FILE="${2:-ttk_outputs/direct/pd_vtu/gan_GT_s0_speed_p160_x0_y0_pd.vtu_port_0.vtu}"

mkdir -p "$OUTDIR"
exec > >(tee "$OUTDIR/run.log") 2>&1

echo "=== SYSTEM ==="
date -u
uname -a
uname -m
ldd --version 2>/dev/null | head -n 2 || true
python3 --version || true
which python3 || true
cmake --version || true
gcc --version | head -n 1 || true
nvidia-smi || true

echo
echo "=== INPUT VTU ==="
echo "VTU_FILE=$VTU_FILE"
ls -lh "$VTU_FILE"
sha256sum "$VTU_FILE" | head -n 1

echo
echo "=== VTU HEADER (first 80 lines) ==="
head -80 "$VTU_FILE" > "$OUTDIR/vtu_head.txt"
sed -n '1,80p' "$OUTDIR/vtu_head.txt"

echo
echo "=== KEY TAG LOCATIONS ==="
grep -n "<VTKFile" "$VTU_FILE" | tee "$OUTDIR/grep_vtkfile.txt" || true
grep -n "<AppendedData" "$VTU_FILE" | tee "$OUTDIR/grep_appended.txt" || true
grep -n "</VTKFile" "$VTU_FILE" | tee "$OUTDIR/grep_vtkfile_end.txt" || true

echo
echo "=== PYTHON VTK INTROSPECTION ==="
python3 - <<'PY' | tee "$OUTDIR/python_vtk_info.txt"
import sys, platform
print("python:", sys.version)
print("platform:", platform.platform())
try:
    import vtk
    print("vtk version:", vtk.vtkVersion.GetVTKVersion())
    print("vtk module file:", vtk.__file__)
    c = vtk.vtkZLibDataCompressor()
    print("zlib compressor:", c.GetClassName())
except Exception as e:
    print("VTK import/usage failed:", repr(e))
PY

echo
echo "=== VTK READ REPRO ==="
python3 read_vtu_debug.py "$VTU_FILE" 2>&1 | tee "$OUTDIR/vtk_read_fail.txt" || true

echo
echo "=== BYTE-LEVEL CHECKS AROUND REPORTED INDEX ==="
python3 check_vtu_bytes.py "$VTU_FILE" 2>&1 | tee "$OUTDIR/byte_checks.txt" || true

echo
echo "=== APPENDED DATA BASE64 CHECK ==="
python3 check_vtu_appended_base64.py "$VTU_FILE" 2>&1 | tee "$OUTDIR/appended_base64_check.txt" || true

echo
echo "=== DONE ==="
echo "Bundle dir: $OUTDIR"
echo "To share: tar -czf ${OUTDIR}.tar.gz $OUTDIR"
