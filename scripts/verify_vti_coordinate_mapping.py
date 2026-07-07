#!/usr/bin/env python3
"""
End-to-end regression test for the VTI writer coordinate-mapping fix
(docs/candidateD_E_topology_audit.md Section 2.3;
docs/candidateD_E_infra_fix_notes.md).

Unlike scripts/verify_vti_transpose_bug.py (which is a standalone, stdlib-only
*simulation* kept as a historical record of the bug), this script exercises
the *actual* production function `make_vti_from_scalar` in
scripts/convert_phire_to_vti.py end to end: it writes a real .vti file to a
temp directory, reads it back with VTK's own XML reader, and asserts the
stored field matches the intended convention.

Uses a NON-SQUARE (H != W) synthetic field on purpose. The historical bug
(ravel(order="F") with SetDimensions(W, H, 1)) reduced to a pure transpose
only for square patches -- for non-square patches it produced fully
scrambled data. A square test field cannot distinguish "transpose" from
"correctly fixed", so this test intentionally uses an asymmetric H x W with
H != W to make sure the coordinate mapping cannot silently pass by
coincidence.

Requires numpy and vtk (the same dependencies convert_phire_to_vti.py itself
requires). Run with:

    python3 scripts/verify_vti_coordinate_mapping.py

Exit code 0: mapping is correct (scalar_2d[y, x] == VTK point (x, y)).
Exit code 1: mapping is wrong -- regression detected.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from convert_phire_to_vti import make_vti_from_scalar  # noqa: E402

try:
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy
except ImportError:
    print("[error] vtk is required for this end-to-end test. "
          "scripts/verify_vti_transpose_bug.py provides a vtk-free simulation instead.",
          file=sys.stderr)
    sys.exit(2)


def build_asymmetric_field(H: int, W: int) -> np.ndarray:
    """Every cell distinct and H != W, so neither a transpose nor a generic
    scramble can accidentally satisfy the correctness check."""
    return (np.arange(H * W, dtype=np.float32).reshape(H, W)) * 1.0 + 0.5


def read_vti_as_grid(vti_path: str):
    """Read a .vti file back and return (grid[ix][iy], dims) using VTK's own
    reader and point-id convention -- no reimplementation of VTK's layout
    logic, so this is a genuine independent check on make_vti_from_scalar's
    output."""
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(vti_path)
    reader.Update()
    img = reader.GetOutput()

    dims = img.GetDimensions()  # (dimX, dimY, dimZ)
    dim_x, dim_y, _ = dims

    arr = img.GetPointData().GetScalars()
    flat = vtk_to_numpy(arr)

    grid = np.empty((dim_x, dim_y), dtype=np.float32)
    for iy in range(dim_y):
        for ix in range(dim_x):
            point_id = ix + iy * dim_x  # VTK's documented point-id convention
            grid[ix, iy] = flat[point_id]
    return grid, dims


def main() -> int:
    H, W = 5, 8  # deliberately non-square
    scalar_2d = build_asymmetric_field(H, W)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_vti = str(Path(tmpdir) / "coord_mapping_test.vti")
        make_vti_from_scalar(scalar_2d, array_name="test_scalar", out_vti=out_vti, ascii=True)

        grid, dims = read_vti_as_grid(out_vti)

    dim_x, dim_y, _ = dims
    print(f"Wrote/read a ({H}, {W}) [H, W] field; VTK dims reported: {dims}")
    assert dim_x == W and dim_y == H, (
        f"Unexpected VTK dimensions {dims}; expected dimX=W={W}, dimY=H={H}"
    )

    # Intended convention: scalar_2d[y, x] -> VTK point (x, y), i.e. grid[x, y].
    intended = scalar_2d.T  # intended[x, y] == scalar_2d[y, x]
    correct = np.array_equal(grid, intended)

    # Also explicitly check the (previously buggy) transposed-vs-scrambled
    # failure modes so a regression is diagnosed, not just detected.
    matches_old_bug_transpose = (
        H == W and np.array_equal(grid, scalar_2d)
    )

    print("\nVTK point (x, y) as read back [grid[x][y]]:")
    print(grid)
    print("\nIntended scalar_2d[y, x] [== scalar_2d.T]:")
    print(intended)

    print(f"\ngrid[x, y] == scalar_2d[y, x]  (CORRECT mapping)?  {correct}")
    if H == W:
        print(f"grid[x, y] == scalar_2d[x, y]  (old transpose bug, only meaningful for H==W)?  "
              f"{matches_old_bug_transpose}")

    if correct:
        print("\n[RESULT] PASS -- VTI writer coordinate mapping is correct "
              "(scalar_2d[y, x] -> VTK point (x, y)).")
        return 0
    else:
        print("\n[RESULT] FAIL -- coordinate mapping regression detected. "
              "Check scripts/convert_phire_to_vti.py's ravel()/SetDimensions() "
              "combination against docs/candidateD_E_topology_audit.md Section 2.3.",
              file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
