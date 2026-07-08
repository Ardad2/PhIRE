#!/usr/bin/env python3
"""
Build CandidateE2-expanded-672 constraints using actual TTK persistence diagrams.

This is the faithful constraint builder for CandidateE2-expanded-672.
It requires Docker (phire-ttk:latest) for TTK computation.

For each of the 672 expanded GT samples:
  1. Write a 160×160 ASCII VTI file (wind_speed scalar field; no VTK required).
  2. Run ttkPersistenceDiagramCmd via Docker on each VTI.
  3. Parse the resulting VTU file using the same parser as extract_ttk_pd_critical_pairs.py.
  4. Extract birth/death vertex IDs (TTK persistence pairs, same filtering as E pilot).
  5. Correct birth_val/death_val by reading GT numpy at the TTK vertex positions
     (the "corrected E2" convention: targets from GT numpy, not from VTU coordinates).
  6. Write the NPZ in TTKConstraints format.

Vertex-ID coordinate convention (consistent with TTKConstraints and l_ttkcv):
  vid // PATCH = iy    (TTK y-index → first index in numpy speed array)
  vid % PATCH  = ix    (TTK x-index → second index in numpy speed array)
  corrected_val = gt_speed[iy, ix]   (same mapping as TTKConstraints.get)

Intermediate files are written to:
  ttk_runs_fixed/topology_finetuning/candidateE2_expanded672_constraints/work/

and are kept after the run for inspection / re-parsing without re-running TTK.

Output:
  ttk_runs_fixed/topology_finetuning/candidateE2_expanded672_constraints/
    ttk_pd_critical_pairs_gtvalues.npz

NPZ schema (TTKConstraints-compatible):
  n_samples     int64 scalar — 672
  sample_idx    (672,) int64 — WTK training indices from idx.npy
  sample_start  (672,) int64 — start offset per sample into flat arrays
  sample_count  (672,) int64 — pair count per sample
  birth_vid     (P,) int64  — flat vertex ID (iy*160 + ix) in 160×160 grid
  death_vid     (P,) int64
  birth_val     (P,) float32 — GT speed at birth vertex (corrected from numpy)
  death_val     (P,) float32 — GT speed at death vertex (corrected from numpy)
  persistence   (P,) float32 — |birth_val - death_val|

Filtering (same as Candidate E pilot):
  persistence_frac = 0.01   (min persistence as fraction of sample max persistence)
  top_k            = 64     (top pairs by descending persistence)

Usage (default: the original 672-sample dataset, unchanged):
  docker pull phire-ttk:latest   # if not already cached
  python3 scripts/build_candidateE2_expanded672_ttk_constraints.py

Usage (generalized: any expanded scale, e.g. 1344):
  python3 scripts/build_candidateE2_expanded672_ttk_constraints.py \\
    --gt-path    data_out/wind_mrhr_cnn_expanded1344/dataGT.npy \\
    --idx-path   data_out/wind_mrhr_cnn_expanded1344/idx.npy \\
    --n-expected 1344 \\
    --out-dir    ttk_runs_fixed/topology_finetuning/candidateE2_fixed_lowlambda_expanded1344_constraints \\
    --vti-dir    ttk_runs_fixed/topology_finetuning/candidateE2_fixed_lowlambda_expanded1344_vti \\
    --vti-label  candidateE2fixedlowlambda1344_GT

Requires:
  - Docker with image phire-ttk:latest
  - <gt-path>  (N, 500, 500, 2), N == --n-expected
  - <idx-path> (N,), N == --n-expected  (optional; falls back to 0..N-1 if absent)
  - numpy only (no VTK or TTK in native Python env)

See also:
  scripts/build_candidateE2approx_expanded672_constraints.py
    — APPROXIMATE version using scipy local maxima; do NOT use for main E2 result.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent

# Historical defaults (unchanged): used when --gt-path/--idx-path/--n-expected
# are not passed, so the original zero-arg 672-sample invocation keeps working
# exactly as before. Pass --gt-path/--idx-path/--n-expected to build
# constraints for a different expanded dataset (e.g. 1344, 2688).
_DEFAULT_GT_PATH  = REPO_ROOT / "data_out" / "wind_mrhr_cnn_expanded672" / "dataGT.npy"
_DEFAULT_IDX_PATH = REPO_ROOT / "data_out" / "wind_mrhr_cnn_expanded672" / "idx.npy"
_DEFAULT_N_EXPECTED = 672

# Historical defaults (unchanged): used when --out-dir/--vti-dir/--pd-dir are
# not passed on the command line, so old invocations keep working exactly as
# before. Pass --out-dir (and optionally --vti-dir/--pd-dir/--vti-label) to
# write to a fresh location instead of overwriting these -- e.g. to rebuild
# constraints against the fixed VTI writer without touching the pre-fix
# candidateE2_expanded672_constraints/ output.
_DEFAULT_OUT_DIR  = (
    REPO_ROOT
    / "ttk_runs_fixed"
    / "topology_finetuning"
    / "candidateE2_expanded672_constraints"
)
_DEFAULT_VTI_DIR  = _DEFAULT_OUT_DIR / "work" / "vti"
_DEFAULT_PD_DIR   = _DEFAULT_OUT_DIR / "work" / "pd"
_DEFAULT_VTI_LABEL = "candidateE2expanded672_GT"

# ---------------------------------------------------------------------------
# Parameters (matching Candidate E pilot for comparability)
# ---------------------------------------------------------------------------
PATCH            = 160       # crop size; VTK Dimensions (160, 160, 1)
TOP_K            = 64        # max pairs per sample
PERSISTENCE_FRAC = 0.01      # min persistence as fraction of per-sample max
DOCKER_IMAGE     = "phire-ttk:latest"
THREADS          = 4         # TTK threads per VTI; each TTK call is sequential

# Docker TTK timeout, scaled per sample from the original 672-sample budget
# (4 hours) so larger expanded datasets (1344, 2688, ...) aren't killed
# partway through by a timeout sized for 672.
_DOCKER_TIMEOUT_SEC_PER_SAMPLE = (4 * 3600) / 672

_SAMPLE_IDX_RE   = re.compile(r"_s(\d+)_")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gt-path", type=Path, default=None,
                    help=f"Path to expanded dataGT.npy, shape (N,500,500,2) "
                         f"(default: {_DEFAULT_GT_PATH})")
    p.add_argument("--idx-path", type=Path, default=None,
                    help=f"Path to expanded idx.npy, shape (N,) "
                         f"(default: {_DEFAULT_IDX_PATH})")
    p.add_argument("--n-expected", type=int, default=None,
                    help=f"Expected sample count N (default: {_DEFAULT_N_EXPECTED})")
    p.add_argument("--out-dir", type=Path, default=None,
                    help=f"Output directory for the constraints NPZ "
                         f"(default: {_DEFAULT_OUT_DIR})")
    p.add_argument("--vti-dir", type=Path, default=None,
                    help="Directory to write GT VTI files to "
                         "(default: <out-dir>/work/vti)")
    p.add_argument("--pd-dir", type=Path, default=None,
                    help="Directory for TTK's VTU output "
                         "(default: <out-dir>/work/pd)")
    p.add_argument("--vti-label", type=str, default=_DEFAULT_VTI_LABEL,
                    help=f"VTI filename label prefix (default: {_DEFAULT_VTI_LABEL})")
    return p


# ---------------------------------------------------------------------------
# Import VTU parser from extract_ttk_pd_critical_pairs
# ---------------------------------------------------------------------------

def _import_extract_module():
    """Import parse_vtu and extract_constraints from extract_ttk_pd_critical_pairs.py."""
    import importlib.util
    src = REPO_ROOT / "scripts" / "extract_ttk_pd_critical_pairs.py"
    if not src.exists():
        sys.exit(
            f"[error] Required script not found: {src}\n"
            "  This script must coexist with extract_ttk_pd_critical_pairs.py."
        )
    spec = importlib.util.spec_from_file_location("_extract_ttk", src)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.parse_vtu, mod.extract_constraints


# ---------------------------------------------------------------------------
# Speed field
# ---------------------------------------------------------------------------

def _speed(uv: np.ndarray) -> np.ndarray:
    """(H, W, 2) float → (H, W) float32 scalar speed."""
    return np.sqrt(
        uv[..., 0].astype(np.float32) ** 2
        + uv[..., 1].astype(np.float32) ** 2
    )


# ---------------------------------------------------------------------------
# ASCII VTI writer (no VTK required)
# ---------------------------------------------------------------------------

def _write_vti_ascii(
    path: Path,
    field2d: np.ndarray,
    array_name: str = "wind_speed",
) -> None:
    """
    Write a 2D scalar field as an ASCII VTK ImageData (.vti) file.

    Coordinate convention: field2d[y, x] -> VTK point (x, y), matching the
    fixed make_vti_from_scalar in convert_phire_to_vti.py. VTK point i is at
    (ix=i%W, iy=i//W) given SetDimensions(W, H, 1) (x fastest), so the flat
    buffer must vary the column axis (W, x) fastest -- ordinary C/row-major
    ravel of a (H, W) array: flat[i] = field[i//W, i%W].

    (Historical note: an earlier version used ravel(order="F"), which
    varies the row axis fastest instead and incorrectly claimed "F-order ==
    C-order" for square patches -- that claim is false in general (it only
    holds if the field happens to be symmetric across its diagonal, which
    wind fields are not) and silently transposed every constraint vertex
    coordinate. See docs/candidateD_E_topology_audit.md Section 2.3 and
    docs/candidateD_E_infra_fix_notes.md.)
    """
    H, W = field2d.shape
    flat  = field2d.astype(np.float32).ravel(order="C")

    # Write 8 values per line for readability
    chunks = []
    for j in range(0, len(flat), 8):
        chunks.append(" ".join(f"{v:.8g}" for v in flat[j : j + 8]))
    data_str = "\n          ".join(chunks)

    content = (
        '<?xml version="1.0"?>\n'
        '<VTKFile type="ImageData" version="0.1" byte_order="LittleEndian">\n'
        f'  <ImageData WholeExtent="0 {W-1} 0 {H-1} 0 0"'
        f' Origin="0 0 0" Spacing="1 1 1">\n'
        f'    <Piece Extent="0 {W-1} 0 {H-1} 0 0">\n'
        f'      <PointData Scalars="{array_name}">\n'
        f'        <DataArray type="Float32" Name="{array_name}"'
        f' format="ascii" NumberOfComponents="1">\n'
        f'          {data_str}\n'
        f'        </DataArray>\n'
        f'      </PointData>\n'
        f'    </Piece>\n'
        f'  </ImageData>\n'
        f'</VTKFile>\n'
    )
    path.write_text(content)


# ---------------------------------------------------------------------------
# Docker availability check
# ---------------------------------------------------------------------------

def _check_docker() -> None:
    """Abort with a clear message if Docker or the required image is missing."""
    result = subprocess.run(
        ["docker", "info"],
        capture_output=True,
        timeout=30,
    )
    if result.returncode != 0:
        sys.exit(
            "[error] Docker is not available or not running.\n"
            "  Start Docker and retry:\n"
            "    sudo systemctl start docker   # Linux\n"
            "    open -a Docker               # macOS"
        )

    result2 = subprocess.run(
        ["docker", "image", "inspect", DOCKER_IMAGE],
        capture_output=True,
        timeout=30,
    )
    if result2.returncode != 0:
        sys.exit(
            f"[error] Docker image '{DOCKER_IMAGE}' not found locally.\n"
            f"  Pull it first:\n"
            f"    docker pull {DOCKER_IMAGE}"
        )
    print(f"[docker] Image '{DOCKER_IMAGE}' is available.")


# ---------------------------------------------------------------------------
# Phase 3: Run TTK via Docker (one bash loop for all 672 VTIs)
# ---------------------------------------------------------------------------

def _run_ttk_docker(vti_dir: Path, pd_dir: Path, n_expected: int) -> None:
    """
    Run ttkPersistenceDiagramCmd on every VTI in vti_dir via a single Docker call.

    TTK output for each VTI named 'base.vti' is written to:
      pd_dir/base_pd_port_0.vtu
    """
    pd_dir.mkdir(parents=True, exist_ok=True)

    # One bash script that loops over all VTIs; avoids one Docker start-up per sample.
    bash_script = (
        "for f in /work_vti/*.vti; do\n"
        "    base=$(basename \"$f\" .vti)\n"
        "    echo \"=== TTK: ${base} ===\"\n"
        f"    ttkPersistenceDiagramCmd -t {THREADS}"
        " -i \"$f\" -a wind_speed"
        " -o \"/work_pd/${base}_pd\"\n"
        "done"
    )

    print(f"[phase3] Running Docker ({DOCKER_IMAGE}) for {len(list(vti_dir.glob('*.vti')))} VTI files …")
    print(f"[phase3] Threads per VTI: {THREADS}  (sequential in one Docker container)")

    # Timeout scaled from the original 672-sample budget (4h) so larger
    # expanded datasets aren't killed partway through.
    timeout_sec = max(4 * 3600, int(n_expected * _DOCKER_TIMEOUT_SEC_PER_SAMPLE))
    print(f"[phase3] Docker timeout: {timeout_sec/3600:.1f}h for {n_expected} samples")

    t0 = time.perf_counter()
    subprocess.run(
        [
            "docker", "run", "--rm",
            "-v", f"{vti_dir.resolve()}:/work_vti",
            "-v", f"{pd_dir.resolve()}:/work_pd",
            DOCKER_IMAGE,
            "bash", "-c", bash_script,
        ],
        check=True,
        timeout=timeout_sec,
    )
    elapsed = time.perf_counter() - t0
    print(f"[phase3] Docker TTK complete in {elapsed:.1f}s  ({elapsed/n_expected:.2f}s/sample)")


# ---------------------------------------------------------------------------
# Phase 5: Parse VTU files and apply corrected GT values
# ---------------------------------------------------------------------------

def _parse_vtu_with_correction(
    vtu_path: Path,
    wtk_idx: int,
    gt_speed_crop: np.ndarray,
    parse_vtu,
    extract_constraints,
) -> dict | None:
    """
    Parse one TTK persistence-diagram VTU file and apply corrected GT values.

    Returns a dict with TTK vertex IDs and GT-numpy-corrected birth_val/death_val,
    or None if no pairs survive the persistence filter.
    """
    try:
        arrays = parse_vtu(vtu_path)
    except Exception as exc:
        print(f"  [warn] parse_vtu failed for {vtu_path.name}: {exc}")
        return None

    constraints = extract_constraints(
        arrays,
        sample_idx=wtk_idx,
        sample_name=vtu_path.stem,
        persistence_frac=PERSISTENCE_FRAC,
        top_k=TOP_K,
        patch=PATCH,
    )
    if constraints is None:
        return None

    W = PATCH
    bvids = constraints["birth_vid"].astype(np.int64)
    dvids = constraints["death_vid"].astype(np.int64)

    # Corrected values: read GT numpy at TTK vertex positions.
    # Coordinate convention: vid // W = iy (first numpy index),
    #                        vid % W  = ix (second numpy index).
    # This matches the TTKConstraints.get decoding used during training.
    b_iy, b_ix = bvids // W, bvids % W
    d_iy, d_ix = dvids // W, dvids % W

    corrected_bval = gt_speed_crop[b_iy, b_ix].astype(np.float32)
    corrected_dval = gt_speed_crop[d_iy, d_ix].astype(np.float32)
    corrected_pers = np.abs(corrected_bval - corrected_dval).astype(np.float32)

    return {
        "sample_idx": wtk_idx,
        "birth_vid":  bvids,
        "death_vid":  dvids,
        "birth_val":  corrected_bval,
        "death_val":  corrected_dval,
        "persistence": corrected_pers,
    }


# ---------------------------------------------------------------------------
# Sanity check: stored values match GT speed at stored vertex positions
# ---------------------------------------------------------------------------

def _sanity_check(gt: np.ndarray, idx_all: np.ndarray, npz_path: Path) -> None:
    """
    Verify that every stored birth_val/death_val matches GT speed at the
    stored vertex within float32 numerical tolerance.

    Also verifies sample_idx alignment with idx_all.
    """
    print("\n[sanity] Verifying stored values match GT speed at each vertex …")
    npz = np.load(npz_path, allow_pickle=True)

    # --- sample_idx alignment ---
    stored_idx = set(npz["sample_idx"].tolist())
    expected_idx = set(idx_all.tolist())
    missing = expected_idx - stored_idx
    extra   = stored_idx   - expected_idx
    if missing:
        sys.exit(
            f"[error] Sanity FAIL: {len(missing)} expected WTK indices missing from NPZ.\n"
            f"  Missing (first 10): {sorted(missing)[:10]}"
        )
    if extra:
        sys.exit(
            f"[error] Sanity FAIL: {len(extra)} unexpected WTK indices in NPZ.\n"
            f"  Extra (first 10): {sorted(extra)[:10]}"
        )

    # --- per-sample value check ---
    sample_idx_npz = npz["sample_idx"].astype(np.int64)
    sample_start   = npz["sample_start"].astype(np.int64)
    sample_count   = npz["sample_count"].astype(np.int64)
    birth_vid      = npz["birth_vid"].astype(np.int64)
    death_vid      = npz["death_vid"].astype(np.int64)
    birth_val      = npz["birth_val"].astype(np.float32)
    death_val      = npz["death_val"].astype(np.float32)

    W = PATCH
    max_b_err = 0.0
    max_d_err = 0.0

    # Build WTK idx → array position map for GT access
    wtk_to_pos = {int(idx_all[i]): i for i in range(len(idx_all))}

    for row_i in range(len(sample_idx_npz)):
        wtk_idx = int(sample_idx_npz[row_i])
        arr_pos = wtk_to_pos[wtk_idx]
        start = int(sample_start[row_i])
        count = int(sample_count[row_i])
        if count == 0:
            continue

        gt_crop = _speed(np.asarray(gt[arr_pos, :PATCH, :PATCH, :]))

        bvids = birth_vid[start : start + count]
        dvids = death_vid[start : start + count]
        bvals = birth_val[start : start + count]
        dvals = death_val[start : start + count]

        b_iy, b_ix = bvids // W, bvids % W
        d_iy, d_ix = dvids // W, dvids % W

        computed_bval = gt_crop[b_iy, b_ix].astype(np.float32)
        computed_dval = gt_crop[d_iy, d_ix].astype(np.float32)

        b_err = float(np.abs(computed_bval - bvals).max())
        d_err = float(np.abs(computed_dval - dvals).max())
        max_b_err = max(max_b_err, b_err)
        max_d_err = max(max_d_err, d_err)

        if b_err > 1e-4 or d_err > 1e-4:
            print(
                f"  [WARN] wtk_idx={wtk_idx}: birth_err={b_err:.2e}  death_err={d_err:.2e}"
            )

    print(f"  max birth_val error : {max_b_err:.2e}")
    print(f"  max death_val error : {max_d_err:.2e}")

    if max_b_err >= 1e-4 or max_d_err >= 1e-4:
        sys.exit(
            f"[error] Sanity FAILED: birth_err={max_b_err:.2e}, death_err={max_d_err:.2e}.\n"
            "  This should not happen with the corrected-value approach.\n"
            "  Check the VTI writing and GT numpy loading."
        )
    print("  Sanity check PASSED.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = build_arg_parser().parse_args()

    gt_path    = args.gt_path    if args.gt_path    is not None else _DEFAULT_GT_PATH
    idx_path   = args.idx_path   if args.idx_path   is not None else _DEFAULT_IDX_PATH
    n_expected = args.n_expected if args.n_expected is not None else _DEFAULT_N_EXPECTED

    out_dir   = args.out_dir if args.out_dir is not None else _DEFAULT_OUT_DIR
    out_npz   = out_dir / "ttk_pd_critical_pairs_gtvalues.npz"
    work_dir  = out_dir / "work"
    vti_dir   = args.vti_dir if args.vti_dir is not None else (
        _DEFAULT_VTI_DIR if args.out_dir is None else work_dir / "vti"
    )
    pd_dir    = args.pd_dir if args.pd_dir is not None else (
        _DEFAULT_PD_DIR if args.out_dir is None else work_dir / "pd"
    )
    vti_label = args.vti_label

    print("=" * 64)
    print("Build CandidateE2 TTK constraints (faithful)")
    print("=" * 64)
    print(f"  GT input    : {gt_path}")
    print(f"  IDX input   : {idx_path}")
    print(f"  N expected  : {n_expected}")
    print(f"  Output NPZ  : {out_npz}")
    print(f"  Work dir    : {work_dir}")
    print(f"  VTI dir     : {vti_dir}")
    print(f"  PD dir      : {pd_dir}")
    print(f"  VTI label   : {vti_label}")
    print(f"  PATCH       : {PATCH}")
    print(f"  TOP_K       : {TOP_K}")
    print(f"  PERS_FRAC   : {PERSISTENCE_FRAC}")
    print(f"  Docker      : {DOCKER_IMAGE}")
    print(f"  Threads/VTI : {THREADS}")
    print()

    # ── Pre-condition checks ──────────────────────────────────────────────
    if not gt_path.exists():
        sys.exit(
            f"[error] Expanded GT arrays not found: {gt_path}\n"
            "  Generate them first (requires TF1):\n"
            "    python3 - <<'PY'\n"
            "    import sys; sys.path.insert(0, '.')\n"
            "    import tensorflow.compat.v1 as tf; tf.disable_v2_behavior()\n"
            "    from PhIREGANs import PhIREGANs\n"
            "    phire = PhIREGANs(\n"
            "        data_type='wind_mrhr_cnn_expanded672',\n"
            "        mu_sig=[[0.7684, -0.4575], [5.02455, 5.9017]],\n"
            "    )\n"
            "    phire.set_data_out_path('data_out/wind_mrhr_cnn_expanded672')\n"
            "    phire.test_paired(\n"
            "        r=[5],\n"
            "        data_path='example_data_topology_expanded_672/wind_MR-HR.tfrecord',\n"
            "        model_path='models/wind_mr-hr/trained_cnn/cnn',\n"
            "        batch_size=1, save_inputs=True,\n"
            "    )\n"
            "    PY"
        )

    gt_shape = np.load(gt_path, mmap_mode="r").shape
    if gt_shape[0] != n_expected:
        sys.exit(
            f"[error] GT array has {gt_shape[0]} samples; expected {n_expected}."
        )
    if gt_shape[1] < PATCH or gt_shape[2] < PATCH:
        sys.exit(
            f"[error] GT spatial size {gt_shape[1]}×{gt_shape[2]} < PATCH={PATCH}."
        )

    if idx_path.exists():
        idx_all = np.load(idx_path).astype(np.int64)
        if len(idx_all) != n_expected:
            sys.exit(
                f"[error] idx.npy has {len(idx_all)} entries; expected {n_expected}."
            )
        print(
            f"  idx range: [{int(idx_all.min())}, {int(idx_all.max())}] "
            f"({len(idx_all)} entries)"
        )
    else:
        print(f"  [warn] idx.npy not found; using 0..{n_expected - 1}")
        idx_all = np.arange(n_expected, dtype=np.int64)

    _check_docker()
    print()

    # ── Import VTU parser ─────────────────────────────────────────────────
    print("[setup] Importing VTU parser from extract_ttk_pd_critical_pairs.py …")
    parse_vtu, extract_constraints = _import_extract_module()
    print("[setup] Import OK.")
    print()

    # ── Phase 1: Load GT (mmap) ───────────────────────────────────────────
    print("[phase1] Loading GT arrays (mmap) …")
    gt = np.load(gt_path, mmap_mode="r")   # (672, 500, 500, 2)
    print(f"  GT shape: {gt.shape}  dtype: {gt.dtype}")
    print()

    # ── Phase 2: Write VTI files ──────────────────────────────────────────
    vti_dir.mkdir(parents=True, exist_ok=True)
    vti_paths = sorted(vti_dir.glob(f"{vti_label}_s*.vti"))
    if len(vti_paths) == n_expected:
        print(f"[phase2] {n_expected} VTI files already exist in {vti_dir} — skipping.")
    else:
        print(f"[phase2] Writing {n_expected} ASCII VTI files to {vti_dir} …")
        t0 = time.perf_counter()
        for i in range(n_expected):
            wtk_idx = int(idx_all[i])
            speed_crop = _speed(np.asarray(gt[i, :PATCH, :PATCH, :]))
            vti_name = f"{vti_label}_s{wtk_idx}_speed_p{PATCH}_x0_y0.vti"
            _write_vti_ascii(vti_dir / vti_name, speed_crop)
            if (i + 1) % 100 == 0 or i == 0:
                print(f"  [{i + 1:4d}/{n_expected}]  {vti_name}")
        elapsed = time.perf_counter() - t0
        vti_paths = sorted(vti_dir.glob(f"{vti_label}_s*.vti"))
        print(f"  Written {len(vti_paths)} VTI files in {elapsed:.1f}s")
        if len(vti_paths) != n_expected:
            sys.exit(
                f"[error] Expected {n_expected} VTI files; wrote {len(vti_paths)}."
            )
    print()

    # ── Phase 3: Run TTK via Docker ────────────────────────────────────────
    n_vtu_existing = len(list(pd_dir.glob("*_port_0.vtu")))
    if n_vtu_existing >= n_expected:
        print(
            f"[phase3] {n_vtu_existing} *_port_0.vtu files already exist in {pd_dir} — "
            "skipping Docker run."
        )
    else:
        _run_ttk_docker(vti_dir, pd_dir, n_expected)
    print()

    # ── Phase 4: Discover VTU files ───────────────────────────────────────
    # TTK writes: {prefix}_pd_port_0.vtu  for ttkPersistenceDiagramCmd
    vtu_files = sorted(pd_dir.glob("*_port_0.vtu"))
    if len(vtu_files) == 0:
        # Fallback: any .vtu (handles TTK versions that don't append _port_0)
        vtu_files = sorted(pd_dir.glob("*.vtu"))

    print(f"[phase4] Found {len(vtu_files)} VTU files in {pd_dir}")
    if len(vtu_files) == 0:
        sys.exit(
            f"[error] No VTU files found in {pd_dir}.\n"
            "  Check Docker logs above for TTK errors."
        )

    # Build sample_idx → VTU path dict
    vtu_by_idx: dict[int, Path] = {}
    for vtu_path in vtu_files:
        m = _SAMPLE_IDX_RE.search(vtu_path.stem)
        if m:
            vtu_by_idx[int(m.group(1))] = vtu_path
        else:
            print(f"  [warn] Cannot parse sample index from {vtu_path.name}; skipping.")

    print(f"[phase4] Parsed {len(vtu_by_idx)} sample indices from VTU filenames.")

    # Check for missing samples
    missing_vtu = [int(idx_all[i]) for i in range(n_expected)
                   if int(idx_all[i]) not in vtu_by_idx]
    if missing_vtu:
        sys.exit(
            f"[error] {len(missing_vtu)} training samples have no VTU output.\n"
            f"  Missing WTK indices (first 10): {missing_vtu[:10]}\n"
            "  Check Docker TTK log above for errors."
        )
    print()

    # ── Phase 5: Parse VTU files and correct GT values ────────────────────
    print(f"[phase5] Parsing {n_expected} VTU files and applying GT-corrected values …")
    t0 = time.perf_counter()

    all_results: list[dict] = []
    n_zero_pairs = 0

    for i in range(n_expected):
        wtk_idx = int(idx_all[i])
        vtu_path = vtu_by_idx[wtk_idx]
        gt_crop  = _speed(np.asarray(gt[i, :PATCH, :PATCH, :]))

        result = _parse_vtu_with_correction(
            vtu_path, wtk_idx, gt_crop, parse_vtu, extract_constraints
        )

        if result is None:
            # No pairs survived — store empty entry so sample_idx is still recorded
            all_results.append({
                "sample_idx": wtk_idx,
                "birth_vid":  np.array([], dtype=np.int64),
                "death_vid":  np.array([], dtype=np.int64),
                "birth_val":  np.array([], dtype=np.float32),
                "death_val":  np.array([], dtype=np.float32),
                "persistence": np.array([], dtype=np.float32),
            })
            n_zero_pairs += 1
        else:
            all_results.append(result)

        if (i + 1) % 100 == 0 or i == 0:
            n_pairs = len(all_results[-1]["birth_vid"])
            print(f"  [{i + 1:4d}/{n_expected}]  wtk_idx={wtk_idx:6d}  pairs={n_pairs}")

    elapsed = time.perf_counter() - t0
    print(f"  Parsed {n_expected} samples in {elapsed:.1f}s")
    if n_zero_pairs > 0:
        print(
            f"  [warn] {n_zero_pairs} samples have 0 pairs "
            "(persistence threshold may be too high or TTK found no finite pairs)."
        )
    print()

    # ── Phase 6: Assemble flat arrays ─────────────────────────────────────
    print("[phase6] Assembling flat arrays …")

    # Sort by WTK index for deterministic ordering
    all_results.sort(key=lambda d: d["sample_idx"])

    all_sidx:  list[int]        = []
    all_start: list[int]        = []
    all_count: list[int]        = []
    flat_bvid: list[np.ndarray] = []
    flat_dvid: list[np.ndarray] = []
    flat_bval: list[np.ndarray] = []
    flat_dval: list[np.ndarray] = []
    flat_pers: list[np.ndarray] = []

    offset = 0
    for d in all_results:
        n = len(d["birth_vid"])
        all_sidx.append(d["sample_idx"])
        all_start.append(offset)
        all_count.append(n)
        flat_bvid.append(d["birth_vid"])
        flat_dvid.append(d["death_vid"])
        flat_bval.append(d["birth_val"])
        flat_dval.append(d["death_val"])
        flat_pers.append(d["persistence"])
        offset += n

    birth_vid_all  = np.concatenate(flat_bvid)  if flat_bvid else np.array([], dtype=np.int64)
    death_vid_all  = np.concatenate(flat_dvid)  if flat_dvid else np.array([], dtype=np.int64)
    birth_val_all  = np.concatenate(flat_bval)  if flat_bval else np.array([], dtype=np.float32)
    death_val_all  = np.concatenate(flat_dval)  if flat_dval else np.array([], dtype=np.float32)
    persistence_all = np.concatenate(flat_pers) if flat_pers else np.array([], dtype=np.float32)

    sample_idx_arr  = np.array(all_sidx,  dtype=np.int64)
    sample_start_arr = np.array(all_start, dtype=np.int64)
    sample_count_arr = np.array(all_count, dtype=np.int64)

    counts_arr = sample_count_arr
    total_pairs = int(counts_arr.sum())
    print(f"  Total pairs : {total_pairs}")
    print(
        f"  Pairs/sample: min={counts_arr.min()}  "
        f"mean={counts_arr.mean():.1f}  max={counts_arr.max()}"
    )
    print()

    # ── Phase 7: Write NPZ ────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[phase7] Writing: {out_npz}")
    np.savez(
        str(out_npz),
        n_samples    = np.int64(n_expected),
        sample_idx   = sample_idx_arr,
        sample_start = sample_start_arr,
        sample_count = sample_count_arr,
        birth_vid    = birth_vid_all,
        death_vid    = death_vid_all,
        birth_val    = birth_val_all,
        death_val    = death_val_all,
        persistence  = persistence_all,
    )

    # Quick structural verification
    check = np.load(str(out_npz), allow_pickle=True)
    required = {
        "n_samples", "sample_idx", "sample_start", "sample_count",
        "birth_vid", "death_vid", "birth_val", "death_val", "persistence",
    }
    missing_keys = required - set(check.files)
    if missing_keys:
        sys.exit(f"[error] Written NPZ is missing keys: {missing_keys}")
    assert int(check["n_samples"]) == n_expected
    assert len(check["sample_idx"]) == n_expected
    print(f"  n_samples   : {int(check['n_samples'])}")
    print(f"  total pairs : {len(check['birth_vid'])}")
    if len(birth_val_all) > 0:
        print(
            f"  birth_val range: [{float(birth_val_all.min()):.4f}, "
            f"{float(birth_val_all.max()):.4f}]"
        )
        print(
            f"  persistence range: [{float(persistence_all.min()):.4f}, "
            f"{float(persistence_all.max()):.4f}]"
        )
    print()

    # ── Phase 8: Sanity check ─────────────────────────────────────────────
    _sanity_check(gt, idx_all, out_npz)

    # ── Done ──────────────────────────────────────────────────────────────
    print()
    print("=" * 64)
    print("Done.")
    print(f"  Output NPZ : {out_npz}")
    print(f"  Work dir   : {work_dir}  (VTI + VTU files kept for inspection)")
    print()
    print("Next step: point CONSTRAINTS_NPZ in your refiner training script at:")
    print(f"  {out_npz}")
    print("=" * 64)


if __name__ == "__main__":
    main()
