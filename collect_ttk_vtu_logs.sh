#!/usr/bin/env bash
set -euo pipefail

TS="$(date -u +%Y%m%dT%H%M%SZ)"
OUTDIR="ttk_vtu_debug_$TS"
mkdir -p "$OUTDIR"

LOG="$OUTDIR/run.log"
exec > >(tee -i "$LOG") 2>&1

echo "=== TIMESTAMP (UTC) ==="
date -u
echo

echo "=== SYSTEM ==="
uname -a
uname -m
ldd --version 2>/dev/null | head -n 2 || true
echo

f="ttk_outputs/direct/pd_vtu/gan_GT_s0_speed_p160_x0_y0_pd.vtu_port_0.vtu"
echo "=== FILE IDENTITY ==="
ls -lh "$f"
sha256sum "$f" | head -n 1
echo

echo "=== TAG COUNTS ==="
grep -n "<VTKFile" "$f" || true
grep -n "</VTKFile" "$f" || true
grep -n "<AppendedData" "$f" || true
echo

echo "=== HEADER (first 40 lines) ==="
sed -n '1,40p' "$f"
echo

echo "=== APPENDEDDATA BOUNDARY (cat -A) ==="
# adjust range if needed; yours looked like the end of file too, which is fine
sed -n '25,32p' "$f" | cat -A
echo

echo "=== PYTHON / VTK ==="
python3 - <<'PY'
import sys
import vtk
print("python:", sys.version)
print("VTK:", vtk.vtkVersion.GetVTKVersion())
print("VTK source:", vtk.vtkVersion.GetVTKSourceVersion())
try:
    c = vtk.vtkZLibDataCompressor()
    print("zlib compressor:", c.GetClassName())
except Exception as e:
    print("zlib compressor error:", e)
PY
echo

echo "=== XML WELL-FORMED CHECK ==="
python3 - <<'PY'
import xml.etree.ElementTree as ET
ET.parse("ttk_outputs/direct/pd_vtu/gan_GT_s0_speed_p160_x0_y0_pd.vtu_port_0.vtu")
print("XML is well-formed")
PY
echo

echo "=== TRAILING BYTES CHECK ==="
python3 - <<'PY'
from pathlib import Path
f = Path("ttk_outputs/direct/pd_vtu/gan_GT_s0_speed_p160_x0_y0_pd.vtu_port_0.vtu")
b = f.read_bytes()
tag = b"</VTKFile>"
end = b.rfind(tag)
print("file bytes:", len(b), "last </VTKFile> at:", end)
tail = b[end+len(tag):]
non_ws = [c for c in tail if c not in b" \t\r\n"]
print("trailing non-whitespace bytes:", len(non_ws))
print("first trailing bytes:", tail[:40])
PY
echo

echo "=== UNDERSCORE-PLACEMENT EVIDENCE ==="
python3 - <<'PY'
from pathlib import Path
f = Path("ttk_outputs/direct/pd_vtu/gan_GT_s0_speed_p160_x0_y0_pd.vtu_port_0.vtu")
s = f.read_text(errors="replace")
tag = '<AppendedData encoding="base64">'
i = s.find(tag)
print("tag idx:", i)
print("after tag:", repr(s[i+len(tag):i+len(tag)+40]))
PY
echo

echo "=== TTK HELP (full) ==="
ttkPersistenceDiagramCmd --help > "$OUTDIR/ttkPersistenceDiagramCmd_help.txt" 2>&1 || true
ttkMergeTreeCmd --help > "$OUTDIR/ttkMergeTreeCmd_help.txt" 2>&1 || true
echo "Saved help to $OUTDIR/*.txt"
echo

echo "=== VTK READ TEST (original vs patched) ==="
python3 - <<'PY'
import vtk
from pathlib import Path

def try_read(path):
    # capture vtk error output in-memory
    ow = vtk.vtkStringOutputWindow()
    vtk.vtkOutputWindow.SetInstance(ow)

    r = vtk.vtkXMLUnstructuredGridReader()
    r.SetFileName(str(path))
    ok = r.Update()  # may still be 1 even with errors; we inspect output
    ug = r.GetOutput()
    msg = ow.GetOutput()
    npts = ug.GetNumberOfPoints() if ug is not None else -1
    ncells = ug.GetNumberOfCells() if ug is not None else -1
    print(f"\n--- {path} ---")
    print("Update() returned:", ok)
    print("points:", npts, "cells:", ncells)
    if msg.strip():
        print("VTK output window:")
        print(msg)

f = Path("ttk_outputs/direct/pd_vtu/gan_GT_s0_speed_p160_x0_y0_pd.vtu_port_0.vtu")
patched = Path(str(f).replace(".vtu", "_patched.vtu"))
try_read(f)
if patched.exists():
    try_read(patched)
else:
    print("patched file not found:", patched)
PY
echo

echo "=== DONE ==="
echo "Log directory: $OUTDIR"
echo "To share: tar -czf ${OUTDIR}.tar.gz $OUTDIR"
