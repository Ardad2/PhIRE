#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./ttk_repro_bundle.sh /absolute/path/to/problem_file.vtu
#
# Example:
#   ./ttk_repro_bundle.sh /home/adadhwal/PhIRE/ttk_outputs/direct/pd_vtu/gan_GT_s0_speed_p160_x0_y0_pd.vtu_port_0.vtu

VTU_FILE="${1:-}"
if [[ -z "${VTU_FILE}" ]]; then
  echo "USAGE: $0 /absolute/path/to/file.vtu"
  exit 2
fi
if [[ ! -f "${VTU_FILE}" ]]; then
  echo "ERROR: VTU file not found: ${VTU_FILE}"
  exit 2
fi

TS="$(date -u +%Y%m%dT%H%M%SZ)"
OUTDIR="ttk_vtu_repro_${TS}"
mkdir -p "${OUTDIR}"/{vtu,env,logs,python}

cp -av "${VTU_FILE}" "${OUTDIR}/vtu/"

VTU_BASENAME="$(basename "${VTU_FILE}")"
VTU_COPY="${OUTDIR}/vtu/${VTU_BASENAME}"

log() { echo "[$(date -u +%H:%M:%SZ)] $*"; }

log "Collecting system/environment info..."
{
  echo "=== DATE (UTC) ==="
  date -u
  echo

  echo "=== UNAME ==="
  uname -a
  uname -m
  echo

  echo "=== OS RELEASE ==="
  (cat /etc/os-release || true)
  echo

  echo "=== GLIBC (ldd) ==="
  (ldd --version 2>&1 | head -n 5) || true
  echo

  echo "=== TOOLCHAIN ==="
  (gcc --version | head -n 2) || true
  (cmake --version | head -n 1) || true
  echo

  echo "=== PYTHON ==="
  (python3 --version) || true
  (command -v python3) || true
  echo

  echo "=== NVIDIA ==="
  (nvidia-smi || true)
  echo

  echo "=== KEY ENV VARS ==="
  env | egrep '^(PATH|LD_LIBRARY_PATH|PYTHONPATH|CONDA_PREFIX|VIRTUAL_ENV)=' || true
  echo
} | tee "${OUTDIR}/env/system.txt" >/dev/null

log "Collecting TTK CLI presence + ldd..."
TTK_BINS=(ttkPersistenceDiagramCmd ttkMergeTreeCmd ttkMergeTreeDistanceMatrixCmd)
for b in "${TTK_BINS[@]}"; do
  if command -v "${b}" >/dev/null 2>&1; then
    BINPATH="$(command -v "${b}")"
    echo "${b} -> ${BINPATH}" | tee -a "${OUTDIR}/env/ttk_bins.txt" >/dev/null
    (ldd "${BINPATH}" > "${OUTDIR}/env/ldd_${b}.txt" 2>&1) || true
    ("${b}" --help > "${OUTDIR}/env/help_${b}.txt" 2>&1) || true
  else
    echo "${b} -> NOT FOUND IN PATH" | tee -a "${OUTDIR}/env/ttk_bins.txt" >/dev/null
  fi
done

log "VTU quick fingerprint..."
{
  echo "=== FILE ==="
  ls -lh "${VTU_COPY}"
  echo

  echo "=== SHA256 ==="
  (sha256sum "${VTU_COPY}" || shasum -a 256 "${VTU_COPY}") 2>/dev/null || true
  echo

  echo "=== FIRST 40 LINES (NO HUGE DUMPS) ==="
  sed -n '1,40p' "${VTU_COPY}" || true
  echo

  echo "=== AppendedData lines (line numbers) ==="
  grep -n "AppendedData" "${VTU_COPY}" || true
  echo

  echo "=== VTKFile header line ==="
  grep -n "<VTKFile" "${VTU_COPY}" || true
  echo
} | tee "${OUTDIR}/logs/vtu_fingerprint.txt" >/dev/null

log "Python: module discovery (no importing TTK yet)..."
python3 - <<'PY' | tee "${OUTDIR}/python/python_discovery.txt" >/dev/null
import sys, importlib.util, platform
print("sys.executable:", sys.executable)
print("sys.version:", sys.version.replace("\n"," "))
print("platform:", platform.platform())
for name in ["vtk", "ttk"]:
    spec = importlib.util.find_spec(name)
    print(f"find_spec({name!r}):", None if spec is None else spec.origin)
PY

log "Python: attempt to read VTU with vtk (if available)..."
set +e
python3 - <<PY > "${OUTDIR}/python/vtk_read_vtu.txt" 2>&1
import sys
vtu_path = r"${VTU_COPY}"
print("VTU:", vtu_path)
try:
    import vtk
    print("vtk version:", vtk.vtkVersion.GetVTKVersion())
    r = vtk.vtkXMLUnstructuredGridReader()
    r.SetFileName(vtu_path)
    ok = r.Update()
    out = r.GetOutput()
    print("Update() returned:", ok)
    if out is None:
        print("Output: None")
    else:
        print("Points:", out.GetNumberOfPoints())
        print("Cells :", out.GetNumberOfCells())
except Exception as e:
    print("EXCEPTION:", repr(e))
    raise
PY
echo "EXIT_CODE=$?" >> "${OUTDIR}/python/vtk_read_vtu.txt"
set -e

log "Python: AppendedData base64 decode sanity check (no full dumps)..."
python3 - <<PY > "${OUTDIR}/python/vtu_appended_decode_check.txt" 2>&1
import base64, re, sys
path = r"${VTU_COPY}"
data = open(path, "rb").read()

m = re.search(rb"<AppendedData[^>]*>(.*)</AppendedData>", data, flags=re.S)
if not m:
    print("No <AppendedData> block found.")
    sys.exit(0)

blob = m.group(1).strip()
# VTK XML appended data commonly starts with '_' sentinel before base64 payload
if blob[:1] == b"_":
    blob = blob[1:]
# remove whitespace/newlines
blob = re.sub(rb"\s+", b"", blob)

print("Base64 chars:", len(blob))
try:
    raw = base64.b64decode(blob, validate=False)
    print("Decoded bytes:", len(raw))
    print("First 32 bytes (hex):", raw[:32].hex())
except Exception as e:
    print("BASE64 DECODE FAILED:", repr(e))
    raise
PY

log "Python: try importing ttk with faulthandler (may segfault)..."
set +e
python3 -X faulthandler -c "import ttk; print('Imported ttk OK')" > "${OUTDIR}/python/import_ttk.txt" 2>&1
echo "EXIT_CODE=$?" >> "${OUTDIR}/python/import_ttk.txt"
set -e

log "Optional: gdb backtrace for python import ttk (if gdb present)..."
if command -v gdb >/dev/null 2>&1; then
  set +e
  gdb -q \
    -ex "set pagination off" \
    -ex "run" \
    -ex "bt" \
    -ex "quit" \
    --args python3 -c "import ttk; print('Imported ttk OK')" \
    > "${OUTDIR}/logs/gdb_python_import_ttk.txt" 2>&1
  set -e
else
  echo "gdb not found; skipping gdb backtrace." > "${OUTDIR}/logs/gdb_python_import_ttk.txt"
fi

log "Packaging tarball..."
tar -czf "${OUTDIR}.tar.gz" "${OUTDIR}"
log "DONE: ${OUTDIR}.tar.gz"
echo
echo "Bundle created:"
echo "  $(pwd)/${OUTDIR}.tar.gz"

