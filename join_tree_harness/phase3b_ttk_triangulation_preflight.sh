#!/usr/bin/env bash
set -euo pipefail

OUT="${1:-$HOME/PhIRE/join_tree_harness/phase3b_preflight}"
mkdir -p "$OUT"

LOG="$OUT/phase3b_ttk_triangulation_preflight.txt"

{
  echo "===== PHASE 3B TTK TRIANGULATION / ARTIFACT PREFLIGHT ====="
  date -Is
  echo "host=$(hostname)"
  echo

  echo "===== HOST TTK / VTK ====="
  /usr/bin/python3 - <<'PY'
try:
    import vtk
    print("VTK", vtk.vtkVersion.GetVTKVersion())
except Exception as e:
    print("VTK import failed:", repr(e))
try:
    import topologytoolkit as ttk
    print("topologytoolkit module:", ttk.__file__)
    print("TTK version attr:", getattr(ttk, "__version__", "n/a"))
except Exception as e:
    print("TTK import failed:", repr(e))
PY
  echo

  echo "===== INSTALLED TRIANGULATION HEADERS ====="
  find /usr/local/include/ttk -type f 2>/dev/null \
    | grep -Ei '(ImplicitTriangulation|PeriodicImplicitTriangulation|Triangulation)' \
    | sort || true
  echo

  echo "===== TARGETED IMPLICIT-TRIANGULATION SOURCE GREP ====="
  CANDS=(
    /usr/local/include/ttk/base/ImplicitTriangulation.h
    /usr/local/include/ttk/base/ImplicitTriangulation.cpp
    /usr/local/include/ttk/base/PeriodicImplicitTriangulation.h
    /usr/local/include/ttk/base/PeriodicImplicitTriangulation.cpp
  )

  for f in "${CANDS[@]}"; do
    if [[ -f "$f" ]]; then
      echo
      echo "----- $f -----"
      grep -nE -A8 -B8 \
        'getVertexNeighbor|vertexNeighbor|preconditionVertexNeighbors|dimensionality_|Diagonal|diagonal|Freudenthal|shift|offset|case[[:space:]]+2' \
        "$f" || true
    fi
  done
  echo

  echo "===== BROADER TRIANGULATION GREP (capped) ====="
  grep -RniE \
    'Freudenthal|getVertexNeighbor|preconditionVertexNeighbors|vertexNeighbor' \
    /usr/local/include/ttk 2>/dev/null \
    | head -n 500 || true
  echo

  echo "===== SAMPLE-69 NUMERICAL MT PORT-0 CANDIDATES ====="
  find "$HOME/PhIRE/ttk_runs_fixed" -type f \
    \( -name '*69*mt_port_0.vtu' -o -name '*s69*port_0.vtu' \) \
    2>/dev/null \
    | sort \
    | grep -E \
      'candidateF_grad_E2_low_expanded2688|candidateUV_expanded2688|/cnn/|wind_mrhr_cnn|CNN|candidateF|candidateUV' \
    || true
  echo

  echo "===== ALL SAMPLE-69 MT PORT-0 CANDIDATES (if naming differs) ====="
  find "$HOME/PhIRE/ttk_runs_fixed" -type f \
    \( -name '*69*port_0.vtu' -o -name '*69*mt_port_0.vtu' \) \
    2>/dev/null \
    | sort \
    | head -n 300 || true
  echo

  echo "===== NODE-VTU METADATA FOR LIKELY CANDIDATES ====="
  /usr/bin/python3 - <<'PY'
from pathlib import Path
import re
import vtk

root = Path.home() / "PhIRE" / "ttk_runs_fixed"
patterns = [
    re.compile(r"candidateF_grad_E2_low_expanded2688.*(?:s69|sample[_-]?0*69|_69_).*mt_port_0\.vtu$", re.I),
    re.compile(r"candidateUV_expanded2688.*(?:s69|sample[_-]?0*69|_69_).*mt_port_0\.vtu$", re.I),
    re.compile(r"(?:cnn|wind_mrhr_cnn).*(?:s69|sample[_-]?0*69|_69_).*mt_port_0\.vtu$", re.I),
]

files=[]
for p in root.rglob("*port_0.vtu"):
    s=str(p)
    if any(rx.search(s) for rx in patterns):
        files.append(p)

for p in sorted(set(files)):
    reader=vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(p))
    reader.Update()
    g=reader.GetOutput()
    pd=g.GetPointData()
    names=[pd.GetArrayName(i) for i in range(pd.GetNumberOfArrays())]
    print("FILE", p)
    print("  points", g.GetNumberOfPoints())
    print("  point arrays", names)
    for name in ("NodeId","VertexId","Scalar","CriticalType"):
        a=pd.GetArray(name)
        if a is not None:
            vals=[a.GetTuple1(i) for i in range(min(5,a.GetNumberOfTuples()))]
            print(f"  {name}: tuples={a.GetNumberOfTuples()} first5={vals}")
    print()
PY

} | tee "$LOG"

sha256sum "$LOG" > "$OUT/sha256_manifest.txt"

echo
echo "===== PREFLIGHT SHA256 ====="
cat "$OUT/sha256_manifest.txt"
