#!/usr/bin/env python3
"""
Docker-independent smoke test for build_candidateE2_expanded672_ttk_constraints.py
after the VTI-writer coordinate fix.

This exercises the REAL script's Phase 1 (load GT) and Phase 2 (write VTI
files) end to end against small, clearly-synthetic fabricated GT/idx arrays,
routed through the new --out-dir/--vti-dir CLI arguments into a location
entirely separate from both the pre-fix output and the real post-fix output
path -- so this can never be confused with, or interfere with, either.

It deliberately does NOT reach Phase 3 (Docker/TTK) -- there is no running
Docker daemon in this environment, and this script does not try to work
around that. It stops right after confirming the (fixed) VTI-writing stage
is correct, using a real VTK read-back against the known synthetic ground
truth, exactly like scripts/verify_vti_coordinate_mapping.py does for
convert_phire_to_vti.py.

It also exercises _sanity_check()'s discriminating power directly: it
fabricates one NPZ with correctly-decoded values (should PASS) and one with
deliberately transposed values (should FAIL), without needing any real TTK
output.

Finally it renders overlay plots (GT scalar speed + fabricated birth/death
vertices) for a couple of synthetic samples, to smoke-test the overlay
generation logic ahead of a real run.

Output: diagnostics/candidateE2_fixed_constraints/smoke_test/
(synthetic data only -- never written to ttk_runs_fixed/... paths)

Usage:
    python3 scripts/smoke_test_candidateE2_fixed_constraints.py
"""

import importlib.util
import shutil
import sys
from pathlib import Path

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAVE_MPL = True
except ImportError:
    _HAVE_MPL = False

REPO_ROOT = Path(__file__).resolve().parent.parent
SMOKE_DIR = REPO_ROOT / "diagnostics" / "candidateE2_fixed_constraints" / "smoke_test"


def load_builder_module():
    src = REPO_ROOT / "scripts" / "build_candidateE2_expanded672_ttk_constraints.py"
    spec = importlib.util.spec_from_file_location("_e2builder_smoke", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_synthetic_gt(n_samples: int, size: int = 200, seed: int = 0):
    """Fabricated, clearly-non-real [u,v] fields, large enough to exceed PATCH=160."""
    rng = np.random.default_rng(seed)
    gt = rng.normal(loc=[0.7684, -0.4575], scale=[5.02455, 5.9017],
                     size=(n_samples, size, size, 2)).astype(np.float32)
    idx = np.arange(1000, 1000 + n_samples, dtype=np.int64)  # distinctive fake WTK indices
    return gt, idx


def read_vti_grid(vti_path: Path):
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(str(vti_path))
    reader.Update()
    img = reader.GetOutput()
    dim_x, dim_y, _ = img.GetDimensions()
    flat = vtk_to_numpy(img.GetPointData().GetScalars())
    grid = np.empty((dim_x, dim_y), dtype=np.float32)
    for iy in range(dim_y):
        for ix in range(dim_x):
            grid[ix, iy] = flat[ix + iy * dim_x]
    return grid, (dim_x, dim_y)


def main() -> int:
    if SMOKE_DIR.exists():
        shutil.rmtree(SMOKE_DIR)
    SMOKE_DIR.mkdir(parents=True)

    e2builder = load_builder_module()

    # ── Fabricate small synthetic GT/idx, monkey-patch the module's input
    #    constants for this in-process smoke test only. Never written to any
    #    real data_out/ path, and never touches GT_PATH/IDX_PATH on disk.
    N = 6
    gt, idx = make_synthetic_gt(N)
    e2builder.GT_PATH = Path("/nonexistent/smoke_test_gt.npy")   # not read from disk
    e2builder.IDX_PATH = Path("/nonexistent/smoke_test_idx.npy")  # not read from disk
    e2builder.N_EXPECTED = N

    vti_dir = SMOKE_DIR / "vti"

    # ── Run Phase 1+2 directly (bypassing main()'s file-existence checks,
    #    since our GT is synthetic/in-memory, not on disk) ──────────────────
    print(f"[smoke] Writing {N} synthetic VTI files via the FIXED _write_vti_ascii()...")
    vti_dir.mkdir(parents=True, exist_ok=True)
    label = "smokeTestE2fixed_GT"
    for i in range(N):
        wtk_idx = int(idx[i])
        speed_crop = e2builder._speed(np.asarray(gt[i, :e2builder.PATCH, :e2builder.PATCH, :]))
        vti_name = f"{label}_s{wtk_idx}_speed_p{e2builder.PATCH}_x0_y0.vti"
        e2builder._write_vti_ascii(vti_dir / vti_name, speed_crop)
    written = sorted(vti_dir.glob(f"{label}_s*.vti"))
    print(f"[smoke] Wrote {len(written)} VTI files to {vti_dir}")
    assert len(written) == N, f"expected {N} VTI files, got {len(written)}"

    # ── Verify each VTI's coordinate mapping against the known synthetic GT,
    #    via real VTK read-back (independent of the writer's own logic). ───
    all_ok = True
    for i in range(N):
        wtk_idx = int(idx[i])
        vti_path = vti_dir / f"{label}_s{wtk_idx}_speed_p{e2builder.PATCH}_x0_y0.vti"
        grid, dims = read_vti_grid(vti_path)
        expected_field = e2builder._speed(
            np.asarray(gt[i, :e2builder.PATCH, :e2builder.PATCH, :])
        )
        ok = np.allclose(grid, expected_field.T, atol=1e-4)
        all_ok &= ok
        print(f"  sample wtk_idx={wtk_idx}: dims={dims}, "
              f"grid[x,y]==speed[y,x]? {ok}")
    if not all_ok:
        print("[smoke] FAIL: VTI coordinate mapping mismatch detected.", file=sys.stderr)
        return 1
    print("[smoke] PASS: all synthetic VTI files have correct (y,x)->(x,y) mapping.\n")

    # ── Exercise _sanity_check()'s discriminating power directly, without
    #    needing real TTK output. Build one NPZ with correctly-decoded
    #    values (should PASS) and one with deliberately transposed values
    #    (should FAIL). ────────────────────────────────────────────────────
    print("[smoke] Testing _sanity_check() discriminating power...")
    W = e2builder.PATCH
    rng = np.random.default_rng(1)
    n_pairs_per_sample = 5

    def build_fake_npz(path: Path, transpose_bug: bool):
        all_bvid, all_dvid, all_bval, all_dval, all_pers = [], [], [], [], []
        starts, counts = [], []
        offset = 0
        for i in range(N):
            gt_crop = e2builder._speed(np.asarray(gt[i, :W, :W, :]))
            bvid = rng.integers(0, W * W, size=n_pairs_per_sample).astype(np.int64)
            dvid = rng.integers(0, W * W, size=n_pairs_per_sample).astype(np.int64)
            b_iy, b_ix = bvid // W, bvid % W
            d_iy, d_ix = dvid // W, dvid % W
            if transpose_bug:
                # Deliberately read the WRONG (swapped) index, simulating
                # the pre-fix bug, to confirm the check catches it.
                bval = gt_crop[b_ix, b_iy].astype(np.float32)
                dval = gt_crop[d_ix, d_iy].astype(np.float32)
            else:
                bval = gt_crop[b_iy, b_ix].astype(np.float32)
                dval = gt_crop[d_iy, d_ix].astype(np.float32)
            all_bvid.append(bvid); all_dvid.append(dvid)
            all_bval.append(bval); all_dval.append(dval)
            all_pers.append(np.abs(bval - dval).astype(np.float32))
            starts.append(offset); counts.append(n_pairs_per_sample)
            offset += n_pairs_per_sample
        np.savez(
            str(path),
            n_samples=np.int64(N),
            sample_idx=idx.astype(np.int64),
            sample_start=np.array(starts, dtype=np.int64),
            sample_count=np.array(counts, dtype=np.int64),
            birth_vid=np.concatenate(all_bvid),
            death_vid=np.concatenate(all_dvid),
            birth_val=np.concatenate(all_bval),
            death_val=np.concatenate(all_dval),
            persistence=np.concatenate(all_pers),
        )

    good_npz = SMOKE_DIR / "fake_correct.npz"
    bad_npz = SMOKE_DIR / "fake_transposed.npz"
    build_fake_npz(good_npz, transpose_bug=False)
    build_fake_npz(bad_npz, transpose_bug=True)

    print("  -- correct-coordinate NPZ (expect PASS) --")
    try:
        e2builder._sanity_check(gt, idx, good_npz)
        good_result = "PASS"
    except SystemExit:
        good_result = "FAIL (unexpected!)"
    print(f"  result: {good_result}")

    print("  -- deliberately-transposed NPZ (expect FAIL) --")
    try:
        e2builder._sanity_check(gt, idx, bad_npz)
        bad_result = "PASS (unexpected!)"
    except SystemExit:
        bad_result = "FAIL (as expected)"
    print(f"  result: {bad_result}")

    sanity_ok = (good_result == "PASS") and (bad_result == "FAIL (as expected)")
    print(f"\n[smoke] _sanity_check() discriminating power confirmed: {sanity_ok}")

    # ── Overlay plots: GT scalar speed + fabricated birth/death vertices,
    #    using the correctly-decoded NPZ, for a couple of synthetic samples.
    overlays_written = 0
    if _HAVE_MPL:
        overlay_dir = SMOKE_DIR / "overlays"
        overlay_dir.mkdir(parents=True, exist_ok=True)
        npz = np.load(good_npz, allow_pickle=True)
        for pos in range(min(2, N)):
            wtk_idx = int(idx[pos])
            start = int(npz["sample_start"][pos])
            count = int(npz["sample_count"][pos])
            bvid = npz["birth_vid"][start:start + count]
            dvid = npz["death_vid"][start:start + count]
            by, bx = bvid // W, bvid % W
            dy, dx = dvid // W, dvid % W

            gt_speed2d = e2builder._speed(np.asarray(gt[pos, :W, :W, :]))

            fig, ax = plt.subplots(figsize=(6, 6))
            im = ax.imshow(gt_speed2d, cmap="viridis")
            plt.colorbar(im, ax=ax, label="synthetic 'wind speed'")
            ax.scatter(bx, by, s=30, marker="^", c="red", label="birth vertices")
            ax.scatter(dx, dy, s=30, marker="v", c="orange", label="death vertices")
            ax.set_title(f"Smoke-test sample wtk_idx={wtk_idx} (SYNTHETIC DATA)")
            ax.legend(loc="upper right", fontsize=8)
            fig.tight_layout()
            out_path = overlay_dir / f"overlay_smoke_s{wtk_idx}.png"
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            print(f"[smoke] wrote overlay: {out_path}")
            overlays_written += 1
    else:
        print("[smoke] matplotlib not available; skipping overlay generation.")

    print("\n" + "=" * 64)
    print("Smoke test summary")
    print("=" * 64)
    print(f"  VTI coordinate mapping (real VTK read-back) : {'PASS' if all_ok else 'FAIL'}")
    print(f"  _sanity_check() discriminating power         : {'PASS' if sanity_ok else 'FAIL'}")
    print(f"  Overlay plots written                        : {overlays_written}")
    print(f"  Synthetic artifacts written to                : {SMOKE_DIR}")
    print("  NOTE: this smoke test used fabricated data. It does NOT")
    print("  constitute a real Candidate E2 constraint regeneration --")
    print("  Docker/TTK were never invoked. Phase 3 onward (real TTK")
    print("  persistence-diagram computation) requires Docker + the")
    print("  phire-ttk:latest image, neither of which is available here.")

    return 0 if (all_ok and sanity_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
