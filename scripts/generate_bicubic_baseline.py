#!/usr/bin/env python3
"""Generate a bicubic-interpolation baseline from the fixed MR inputs.

Reads:
    data_out_fixed/wind_mrhr_cnn/dataIN.npy   (168, 100, 100, 2)  MR inputs
    data_out_fixed/wind_mrhr_cnn/dataGT.npy   (168, 500, 500, 2)  HR ground truth
    data_out_fixed/wind_mrhr_cnn/idx.npy      (168,)              sample indices

Writes:
    data_out_fixed/wind_mrhr_bicubic/
        dataSR.npy   bicubic upsampled MR → HR  (168, 500, 500, 2)  float32
        dataIN.npy   copy of CNN dataIN
        dataGT.npy   copy of CNN dataGT
        idx.npy      copy of CNN idx

The two wind-vector channels (u, v) are interpolated separately so no
scalar conversion is performed before resampling.  Cubic order-3 spline
interpolation is used via scipy.ndimage.zoom.

Run from either the repo root or the scripts/ subdirectory:
    cd ~/PhIRE
    PYTHONNOUSERSITE=1 /usr/bin/python3 scripts/generate_bicubic_baseline.py

    cd ~/PhIRE/scripts
    PYTHONNOUSERSITE=1 /usr/bin/python3 generate_bicubic_baseline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import zoom


# ---------------------------------------------------------------------------
# Locate repo root robustly: script lives in <repo>/scripts/
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SRC_DIR = REPO_ROOT / "data_out_fixed" / "wind_mrhr_cnn"
OUT_DIR = REPO_ROOT / "data_out_fixed" / "wind_mrhr_bicubic"

IN_NPY  = SRC_DIR / "dataIN.npy"
GT_NPY  = SRC_DIR / "dataGT.npy"
IDX_NPY = SRC_DIR / "idx.npy"


def check_sources() -> None:
    missing = [p for p in (IN_NPY, GT_NPY, IDX_NPY) if not p.exists()]
    if missing:
        print("[error] The following required files were not found:")
        for p in missing:
            print(f"  {p}")
        print()
        print("Make sure the fixed CNN inference has been run and outputs saved to:")
        print(f"  {SRC_DIR}")
        sys.exit(1)


def bicubic_upsample(data_in: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Upsample (N, H, W, C) to (N, target_h, target_w, C) with cubic splines.

    Each channel of each sample is zoomed independently so that the
    interpolation never mixes the u and v wind components.

    Parameters
    ----------
    data_in : ndarray, shape (N, H, W, C)
    target_h : int  desired output height
    target_w : int  desired output width

    Returns
    -------
    ndarray, shape (N, target_h, target_w, C), dtype float32
    """
    n, h, w, c = data_in.shape

    if h == target_h and w == target_w:
        return data_in.astype(np.float32)

    zoom_h = target_h / h
    zoom_w = target_w / w

    out = np.empty((n, target_h, target_w, c), dtype=np.float32)

    for i in range(n):
        for ch in range(c):
            # order=3 → cubic B-spline interpolation
            # mode='reflect' handles boundary; prefilter=True (default) applies
            # the spline pre-filter for accuracy.
            out[i, :, :, ch] = zoom(
                data_in[i, :, :, ch].astype(np.float64),
                (zoom_h, zoom_w),
                order=3,
                mode="reflect",
                prefilter=True,
            )

        if (i + 1) % 20 == 0 or i == n - 1:
            print(f"  interpolated {i + 1}/{n} samples", flush=True)

    return out


def main() -> None:
    print("=" * 60)
    print("PhIRE bicubic baseline generator")
    print(f"  repo root : {REPO_ROOT}")
    print(f"  source    : {SRC_DIR}")
    print(f"  output    : {OUT_DIR}")
    print("=" * 60)

    check_sources()

    # ------------------------------------------------------------------
    # Load inputs
    # ------------------------------------------------------------------
    print("\n[1] Loading source arrays …")
    data_in = np.load(IN_NPY)   # (N, H_mr, W_mr, C)
    data_gt = np.load(GT_NPY)   # (N, H_hr, W_hr, C)
    idx     = np.load(IDX_NPY)  # (N,)

    print(f"  dataIN  shape={data_in.shape}  dtype={data_in.dtype}")
    print(f"  dataGT  shape={data_gt.shape}  dtype={data_gt.dtype}")
    print(f"  idx     shape={idx.shape}       dtype={idx.dtype}")

    # Sanity check that we have the expected dimensionality
    if data_in.ndim != 4 or data_gt.ndim != 4:
        print("[error] Expected 4-D arrays (N, H, W, C).  Check source files.")
        sys.exit(1)

    n, h_mr, w_mr, c = data_in.shape
    _, h_hr, w_hr, _ = data_gt.shape

    print(f"\n  MR spatial size : {h_mr} × {w_mr}")
    print(f"  HR spatial size : {h_hr} × {w_hr}")
    print(f"  Scale factor    : {h_hr / h_mr:.1f} × {w_hr / w_mr:.1f}")
    print(f"  Channels        : {c}  (u=channel 0, v=channel 1)")

    # ------------------------------------------------------------------
    # Value range summary before interpolation
    # ------------------------------------------------------------------
    print("\n[2] Value ranges before interpolation:")
    print(f"  dataIN  min={data_in.min():.4f}  max={data_in.max():.4f}")
    print(f"  dataGT  min={data_gt.min():.4f}  max={data_gt.max():.4f}")

    # ------------------------------------------------------------------
    # Bicubic upsampling
    # ------------------------------------------------------------------
    print(f"\n[3] Bicubic upsampling {h_mr}×{w_mr} → {h_hr}×{w_hr} …")
    data_sr = bicubic_upsample(data_in, h_hr, w_hr)

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------
    print("\n[4] Verification:")
    print(f"  bicubic SR shape : {data_sr.shape}   dtype={data_sr.dtype}")
    print(f"  bicubic SR min   : {data_sr.min():.4f}")
    print(f"  bicubic SR max   : {data_sr.max():.4f}")

    shape_ok = data_sr.shape == (n, h_hr, w_hr, c)
    print(f"  output shape matches GT shape : {shape_ok}")
    if not shape_ok:
        print("[error] Shape mismatch — aborting before writing.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Write outputs
    # ------------------------------------------------------------------
    print(f"\n[5] Writing outputs to {OUT_DIR} …")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    np.save(OUT_DIR / "dataSR.npy", data_sr)
    np.save(OUT_DIR / "dataIN.npy", data_in)
    np.save(OUT_DIR / "dataGT.npy", data_gt)
    np.save(OUT_DIR / "idx.npy",    idx)

    print("  dataSR.npy  written")
    print("  dataIN.npy  copied from CNN source")
    print("  dataGT.npy  copied from CNN source")
    print(f"  idx.npy     copied — indices: {idx[0]} … {idx[-1]}")

    print("\n[done] Bicubic baseline complete.")
    print(f"  Output directory: {OUT_DIR}")


if __name__ == "__main__":
    main()
