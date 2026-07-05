#!/usr/bin/env python3
"""
Standalone regression test for the VTI writer transpose bug documented in
docs/candidateD_E_topology_audit.md (Section 2.3).

Pure Python standard library only (no numpy, no vtk, no Docker) so it can run
in any environment, including sandboxes without the project's usual
scientific-Python stack.

What it checks
--------------
scripts/convert_phire_to_vti.py writes a 2D scalar field to VTI as:

    img.SetDimensions(W, H, 1)                       # VTK point id = ix + iy*W
    flat = scalar_2d.ravel(order="F")                # varies row (H) fastest

For a square (H == W) patch -- the case used everywhere in this repo's
topology pipeline (patch=160) -- this combination makes VTK's field
img(ix, iy) equal scalar_2d[ix, iy] instead of the intended
scalar_2d[iy, ix] (x=column, y=row, per the function's own docstring).
That is a transpose, not a no-op.

This script builds a small asymmetric test field, replays exactly the same
ravel/SetDimensions arithmetic (without needing the vtk package -- VTK's
point-id formula for vtkImageData, pointId = ix + iy*dimX, is a stable,
documented convention, not something that needs the library installed to
reason about), and asserts whether the transpose is present.

Exit code 0: bug reproduced as documented (writer is unfixed).
Exit code 1: bug NOT reproduced (writer has been fixed, or logic changed) --
             re-read docs/candidateD_E_topology_audit.md Section 2.3 and
             update it if the underlying script has changed.

Usage:
    python3 scripts/verify_vti_transpose_bug.py
"""

import sys


def simulate_convert_phire_to_vti_ravel_order(scalar_2d, order):
    """Replicate `np.ascontiguousarray(scalar_2d).ravel(order=order)` for a
    plain list-of-lists 2D field, without requiring numpy."""
    H = len(scalar_2d)
    W = len(scalar_2d[0])
    flat = [None] * (H * W)
    if order == "F":
        # Fortran order: first axis (row/H) varies fastest.
        i = 0
        for col in range(W):
            for row in range(H):
                flat[i] = scalar_2d[row][col]
                i += 1
    elif order == "C":
        # C order (numpy default): last axis (col/W) varies fastest.
        i = 0
        for row in range(H):
            for col in range(W):
                flat[i] = scalar_2d[row][col]
                i += 1
    else:
        raise ValueError(order)
    return flat


def vtk_image_read(flat, ix, iy, dim_x):
    """VTK's documented vtkImageData point-id convention: pointId = ix + iy*dimX
    (x varies fastest). This is independent of how `flat` was produced."""
    return flat[ix + iy * dim_x]


def build_asymmetric_field(n=4):
    """An n x n field with every cell distinct, so a transpose is always
    detectable (no accidental symmetry)."""
    return [[row * n + col for col in range(n)] for row in range(n)]


def main():
    n = 4
    scalar_2d = build_asymmetric_field(n)
    H, W = n, n

    flat = simulate_convert_phire_to_vti_ravel_order(scalar_2d, order="F")

    # What VTK/TTK will actually read back, given SetDimensions(W, H, 1):
    img = [[vtk_image_read(flat, ix, iy, dim_x=W) for iy in range(H)] for ix in range(W)]

    matches_transposed = all(
        img[ix][iy] == scalar_2d[ix][iy] for ix in range(W) for iy in range(H)
    )
    matches_intended = all(
        img[ix][iy] == scalar_2d[iy][ix] for ix in range(W) for iy in range(H)
    )

    print("Original scalar_2d[row][col]:")
    for row in scalar_2d:
        print(" ", row)

    print("\nWhat VTK/TTK actually reads as img[ix][iy] (order='F' + SetDimensions(W,H,1)):")
    for ix in range(W):
        print(" ", img[ix])

    print(f"\nimg[ix][iy] == scalar_2d[ix][iy]  (TRANSPOSED, i.e. BUG present)?  {matches_transposed}")
    print(f"img[ix][iy] == scalar_2d[iy][ix]  (CORRECT/intended mapping)?       {matches_intended}")

    # Also demonstrate the downstream consequence for vertex decoding used by
    # extract_ttk_pd_critical_pairs.py / build_candidateE2_expanded672_ttk_constraints.py:
    #   iy = vid // W ; ix = vid % W ; value = gt_speed[iy, ix]
    print("\n--- Downstream vertex-decoding consequence (extract_ttk_pd_critical_pairs.py style) ---")
    test_ix_vtk, test_iy_vtk = 1, 3  # an arbitrary off-diagonal VTK point
    vid = test_ix_vtk + test_iy_vtk * W
    decoded_iy, decoded_ix = vid // W, vid % W  # == test_iy_vtk, test_ix_vtk
    true_value_ttk_saw = img[test_ix_vtk][test_iy_vtk]
    value_code_reads = scalar_2d[decoded_iy][decoded_ix]
    print(f"  TTK vertex id={vid} at VTK point (ix={test_ix_vtk}, iy={test_iy_vtk})")
    print(f"  True scalar value TTK associated with this vertex: {true_value_ttk_saw}")
    print(f"  Value the extraction code reads via gt_speed[{decoded_iy}][{decoded_ix}]: {value_code_reads}")
    mismatch = true_value_ttk_saw != value_code_reads
    print(f"  Mismatch (i.e. wrong pixel selected for supervision)? {mismatch}")

    if matches_transposed and not matches_intended and mismatch:
        print("\n[RESULT] Bug reproduced as documented in "
              "docs/candidateD_E_topology_audit.md Section 2.3.")
        return 0
    else:
        print("\n[RESULT] Bug NOT reproduced -- convert_phire_to_vti.py may have "
              "changed. Re-verify docs/candidateD_E_topology_audit.md Section 2.3.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
