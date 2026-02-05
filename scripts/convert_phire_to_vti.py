#!/usr/bin/env python3
"""
Convert PhIRE wind .npy data to VTK .vti for TTK (Persistence Diagrams + Merge Trees).

Supports common tensor layouts:
- (N, H, W, C) where C>=2 and [u,v] are in last dim
- (N, C, H, W) where C>=2 and [u,v] are in first dim
- single sample layouts (H, W, C) or (C, H, W)

Writes ASCII VTI (reliable).
"""

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk


def infer_uv(sample: np.ndarray):
    """
    Given a sample tensor, return (u, v) as 2D arrays [H, W].
    Handles (H,W,C) or (C,H,W).
    """
    if sample.ndim != 3:
        raise ValueError(f"Expected a 3D sample after indexing; got shape {sample.shape}")

    # Case A: (H, W, C)
    if sample.shape[-1] >= 2:
        u = sample[..., 0]
        v = sample[..., 1]
        return u, v

    # Case B: (C, H, W)
    if sample.shape[0] >= 2:
        u = sample[0, ...]
        v = sample[1, ...]
        return u, v

    raise ValueError(
        f"Could not infer (u,v) channels from sample shape {sample.shape}. "
        "Expected last dim>=2 or first dim>=2."
    )


def load_sample(npy_path: str, idx: int) -> np.ndarray:
    arr = np.load(npy_path)
    if arr.ndim < 3:
        raise ValueError(f"Unexpected array shape {arr.shape} from {npy_path}")

    # If there's a leading sample dimension, index it
    if arr.ndim == 4:
        # likely (N,H,W,C) or (N,C,H,W)
        if idx < 0 or idx >= arr.shape[0]:
            raise IndexError(f"sample_idx {idx} out of range for N={arr.shape[0]}")
        return arr[idx]
    elif arr.ndim == 3:
        # already a single sample (H,W,C) or (C,H,W)
        if idx != 0:
            raise ValueError(f"{npy_path} looks like a single sample (ndim=3); use --sample 0")
        return arr
    else:
        # If you ever have (N,T,H,W,...) etc, you can extend this here
        raise ValueError(f"Unsupported npy ndim={arr.ndim} with shape {arr.shape} for {npy_path}")


def make_vti_from_scalar(scalar_2d: np.ndarray, array_name: str, out_vti: str, ascii: bool = True) -> None:
    """
    Write a 2D scalar field as VTK ImageData (.vti). Uses x=width (cols), y=height (rows).
    """
    scalar_2d = np.asarray(scalar_2d, dtype=np.float32)
    if scalar_2d.ndim != 2:
        raise ValueError(f"Expected 2D scalar, got shape {scalar_2d.shape}")

    H, W = scalar_2d.shape  # rows, cols

    img = vtk.vtkImageData()
    img.SetDimensions(W, H, 1)
    img.SetExtent(0, W - 1, 0, H - 1, 0, 0)
    img.SetSpacing(1.0, 1.0, 1.0)
    img.SetOrigin(0.0, 0.0, 0.0)

    # VTK expects data ordered with x varying fastest; ravel(order="F") is reliable here.
    flat = np.ascontiguousarray(scalar_2d).ravel(order="F")
    vtk_arr = numpy_to_vtk(flat, deep=True, array_type=vtk.VTK_FLOAT)
    vtk_arr.SetName(array_name)

    pd = img.GetPointData()
    pd.AddArray(vtk_arr)
    pd.SetActiveScalars(array_name)  # important: makes -a <name> consistently visible
    pd.SetScalars(vtk_arr)

    w = vtk.vtkXMLImageDataWriter()
    w.SetFileName(out_vti)
    w.SetInputData(img)

    if ascii:
        w.SetDataModeToAscii()
    else:
        # If you ever revisit appended, do it carefully (and test readback).
        w.SetDataModeToAppended()
        w.EncodeAppendedDataOn()
        w.SetCompressorTypeToNone()

    ok = w.Write()
    if not ok:
        raise RuntimeError(f"VTK failed to write {out_vti}")


def sanitize_label(s: str) -> str:
    s = s.strip()
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to .npy file (e.g., dataGT.npy or dataSR.npy)")
    ap.add_argument("--outdir", default="vtk_inputs", help="Output directory for .vti files")
    ap.add_argument("--label", default=None, help="Label prefix for output filenames (default: derived from input path)")
    ap.add_argument("--scalar", choices=["speed", "u", "v"], default="speed", help="Which scalar field to write")
    ap.add_argument("--patch", type=int, default=160, help="Patch size (square). Use 0 for full field.")
    ap.add_argument("--x0", type=int, default=0, help="Patch x0 (column start)")
    ap.add_argument("--y0", type=int, default=0, help="Patch y0 (row start)")
    ap.add_argument("--samples", type=int, nargs="+", default=[0], help="Sample indices to export (e.g., 0 1 2 3 4)")
    args = ap.parse_args()

    in_path = args.input
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.label is None:
        # derive label from parent folder + filename stem
        p = Path(in_path)
        label = sanitize_label(f"{p.parent.name}_{p.stem}")
    else:
        label = sanitize_label(args.label)

    for sidx in args.samples:
        sample = load_sample(in_path, sidx)
        u, v = infer_uv(sample)

        # compute requested scalar
        if args.scalar == "u":
            field = u.astype(np.float32)
            array_name = "wind_u"
        elif args.scalar == "v":
            field = v.astype(np.float32)
            array_name = "wind_v"
        else:
            field = np.sqrt(u.astype(np.float32) ** 2 + v.astype(np.float32) ** 2)
            array_name = "wind_speed"

        H, W = field.shape
        if args.patch and args.patch > 0:
            x0, y0 = args.x0, args.y0
            x1 = min(x0 + args.patch, W)
            y1 = min(y0 + args.patch, H)
            if x0 < 0 or y0 < 0 or x0 >= W or y0 >= H:
                raise ValueError(f"Invalid patch origin (x0={x0}, y0={y0}) for field shape (H={H}, W={W})")
            field2d = field[y0:y1, x0:x1]
            patch_tag = f"p{args.patch}_x{x0}_y{y0}"
        else:
            field2d = field
            patch_tag = "full"

        out_vti = outdir / f"{label}_s{sidx}_{args.scalar}_{patch_tag}.vti"
        print(f"[WRITE] {out_vti}")
        print(f"  sample shape: {sample.shape}, scalar shape: {field2d.shape}, range: [{field2d.min():.4f}, {field2d.max():.4f}]")

        make_vti_from_scalar(field2d, array_name=array_name, out_vti=str(out_vti), ascii=True)

    print("[OK] Done.")


if __name__ == "__main__":
    main()


