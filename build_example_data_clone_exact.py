#!/usr/bin/env python3
"""
Build an exact-behavior clone of PhIRE's shipped wind example_data.

Goals:
- Write exactly two files:
    example_data_clone_exact/wind_LR-MR.tfrecord
    example_data_clone_exact/wind_MR-HR.tfrecord
- Exactly 5 records in each TFRecord
- Use native WTK [u, v] data shaped like PhIRE expects: (N, H, W, 2)
- Use PhIRE's TFRecord schema and TF1-style generate_TFRecords/downscale behavior
- Match the shipped wind stage behavior as closely as possible:
    LR->MR = 10x, MR->HR = 5x
  which implies:
    LR = 4x4, MR = 40x40, HR = 200x200

Important:
- This is a behavior/schema clone of the shipped example_data flow, not a byte-for-byte
  clone of the original sample values.
- Because the final HR is 200x200, crop to 160x160 (e.g. x0=20, y0=20) later when
  converting .npy outputs to VTI for your existing TTK workflow.
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import h5pyd
import tensorflow as tf

# -----------------------------------------------------------------------------
# TensorFlow 2 -> TensorFlow 1 compatibility for PhIRE helper behavior
# -----------------------------------------------------------------------------
# PhIRE's repo targets TF 1.12. Under TF2 we need compat.v1 symbols and disabled
# eager execution so the old graph/session code works.
if tf.executing_eagerly():
    tf.compat.v1.disable_eager_execution()

if not hasattr(tf, "reset_default_graph"):
    tf.reset_default_graph = tf.compat.v1.reset_default_graph
if not hasattr(tf, "placeholder"):
    tf.placeholder = tf.compat.v1.placeholder
if not hasattr(tf, "Session"):
    tf.Session = tf.compat.v1.Session
if not hasattr(tf, "python_io"):
    tf.python_io = tf.compat.v1.python_io

# Import repo helper after TF compatibility patch.
import utils  # noqa: E402

# -----------------------------------------------------------------------------
# Configuration: exact-behavior clone target
# -----------------------------------------------------------------------------
N_SAMPLES = 5
SAMPLE_INDICES = [0, 1, 2, 3, 4]
HEIGHT_M = 100

# Clean integer ladder matching shipped wind stages:
# LR->MR = 10x, MR->HR = 5x
LR_SIZE = 4
MR_SIZE = 40
HR_SIZE = 200

# Region large enough to support a native 200x200 crop.
LAT_RANGE = (37.0, 42.0)
LON_RANGE = (-77.0, -73.0)

OUTDIR = Path("example_data_clone_exact")
OUTDIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Exact-ish TF1 downscaling helper matching PhIRE utils.downscale_image
# -----------------------------------------------------------------------------
def phire_downscale_image_exact(x: np.ndarray, K: int) -> np.ndarray:
    """
    Reproduce PhIRE utils.downscale_image() behavior under TF2 using compat.v1.

    Notes:
    - Preserves float64, graph/session execution, SAME padding, and the repo's
      channel-mixing filter construction.
    - This intentionally mirrors the original helper semantics as closely as
      possible so the generated TFRecords behave like an extension of example_data.
    """
    tf.reset_default_graph()

    if x.ndim == 3:
        x = x.reshape((1, x.shape[0], x.shape[1], x.shape[2]))

    x = np.asarray(x, dtype=np.float64)

    x_in = tf.placeholder(tf.float64, [None, x.shape[1], x.shape[2], x.shape[3]])
    weight = tf.constant(1.0 / K**2, shape=[K, K, x.shape[3], x.shape[3]], dtype=tf.float64)
    # Keep the original TF1-style behavior and padding.
    downscaled = tf.compat.v1.nn.conv2d(input=x_in, filters=weight, strides=[1, K, K, 1], padding="SAME")

    with tf.Session() as sess:
        ds_out = sess.run(downscaled, feed_dict={x_in: x})
    return ds_out


# Monkey-patch the repo helper so generate_TFRecords() uses this exact TF2-safe path.
utils.downscale_image = phire_downscale_image_exact


# -----------------------------------------------------------------------------
# WIND Toolkit extraction
# -----------------------------------------------------------------------------
def compute_uv_from_speed_direction(speed: np.ndarray, direction_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert meteorological wind speed+direction to [u, v] components."""
    theta = np.radians(direction_deg)
    u = -speed * np.sin(theta)
    v = -speed * np.cos(theta)
    return u.astype(np.float64), v.astype(np.float64)


def extract_native_hr_stack() -> np.ndarray:
    """
    Extract a native 200x200 center crop from WTK and return shape (5, 200, 200, 2).
    """
    print("=" * 80)
    print("Building exact-behavior PhIRE example_data clone")
    print("=" * 80)
    print(f"Using native HR size: {HR_SIZE}x{HR_SIZE}")
    print(f"Stage sizes: LR={LR_SIZE}, MR={MR_SIZE}, HR={HR_SIZE}")
    print(f"Sample indices: {SAMPLE_INDICES}")
    print(f"Region: lat {LAT_RANGE}, lon {LON_RANGE}, height={HEIGHT_M}m")

    f = h5pyd.File("/nrel/wtk-us.h5", "r")
    coords = f["coordinates"][...]
    if coords.dtype.names:
        lats = coords["lat"]
        lons = coords["lon"]
    else:
        lats = coords[0]
        lons = coords[1]

    lat_mask = (lats[:, 0] >= LAT_RANGE[0]) & (lats[:, 0] <= LAT_RANGE[1])
    lon_mask = (lons[0, :] >= LON_RANGE[0]) & (lons[0, :] <= LON_RANGE[1])
    lat_idx = np.where(lat_mask)[0]
    lon_idx = np.where(lon_mask)[0]

    native_h = len(lat_idx)
    native_w = len(lon_idx)
    print(f"Available region size: {native_h}x{native_w}")

    if native_h < HR_SIZE or native_w < HR_SIZE:
        f.close()
        raise RuntimeError(
            f"Region too small for native {HR_SIZE}x{HR_SIZE} HR crop. "
            f"Available: {native_h}x{native_w}. Expand LAT_RANGE/LON_RANGE."
        )

    h0 = (native_h - HR_SIZE) // 2
    w0 = (native_w - HR_SIZE) // 2
    lat_start = lat_idx[h0]
    lat_end = lat_start + HR_SIZE
    lon_start = lon_idx[w0]
    lon_end = lon_start + HR_SIZE
    print(f"Cropping native center patch: rows {lat_start}:{lat_end}, cols {lon_start}:{lon_end}")

    hr_stack = np.zeros((N_SAMPLES, HR_SIZE, HR_SIZE, 2), dtype=np.float64)

    for i, t in enumerate(SAMPLE_INDICES):
        speed = f[f"windspeed_{HEIGHT_M}m"][t, lat_start:lat_end, lon_start:lon_end]
        direction = f[f"winddirection_{HEIGHT_M}m"][t, lat_start:lat_end, lon_start:lon_end]
        u, v = compute_uv_from_speed_direction(speed, direction)
        hr_stack[i, :, :, 0] = u
        hr_stack[i, :, :, 1] = v
        print(
            f"Sample {t}: u in [{u.min():.2f}, {u.max():.2f}], "
            f"v in [{v.min():.2f}, {v.max():.2f}]"
        )

    f.close()
    return hr_stack


# -----------------------------------------------------------------------------
# Main clone builder
# -----------------------------------------------------------------------------
def main() -> int:
    hr_stack = extract_native_hr_stack()

    print("\nPreparing stage-consistent arrays with PhIRE downscale_image()...")
    mr_stack = phire_downscale_image_exact(hr_stack, 5)   # 200 -> 40
    lr_stack = phire_downscale_image_exact(mr_stack, 10)  # 40 -> 4

    print(f"hr_stack shape: {hr_stack.shape}")
    print(f"mr_stack shape: {mr_stack.shape}")
    print(f"lr_stack shape: {lr_stack.shape}")

    if mr_stack.shape[1:4] != (MR_SIZE, MR_SIZE, 2):
        raise RuntimeError(f"Unexpected MR shape {mr_stack.shape}; expected (*, {MR_SIZE}, {MR_SIZE}, 2)")
    if lr_stack.shape[1:4] != (LR_SIZE, LR_SIZE, 2):
        raise RuntimeError(f"Unexpected LR shape {lr_stack.shape}; expected (*, {LR_SIZE}, {LR_SIZE}, 2)")

    # Write the two shipped-style files using PhIRE's own TFRecord helper.
    out_lr_mr = OUTDIR / "wind_LR-MR.tfrecord"
    out_mr_hr = OUTDIR / "wind_MR-HR.tfrecord"

    print("\nWriting PhIRE-native TFRecords via utils.generate_TFRecords()...")
    # For LR->MR: provide MR as the 'HR' array and let K=10 generate LR internally.
    utils.generate_TFRecords(str(out_lr_mr), mr_stack.astype(np.float64), mode="train", K=10)
    # For MR->HR: provide HR as the 'HR' array and let K=5 generate MR internally.
    utils.generate_TFRecords(str(out_mr_hr), hr_stack.astype(np.float64), mode="train", K=5)

    print("\n" + "=" * 80)
    print("EXACT-BEHAVIOR CLONE CREATED")
    print("=" * 80)
    print(f"Output directory: {OUTDIR.resolve()}")
    print(f"Created: {out_lr_mr.name}, {out_mr_hr.name}")
    print("\nBehavioral clone characteristics:")
    print(f"- records per file: {N_SAMPLES}")
    print(f"- wind_LR-MR.tfrecord: {LR_SIZE}x{LR_SIZE}x2 -> {MR_SIZE}x{MR_SIZE}x2")
    print(f"- wind_MR-HR.tfrecord: {MR_SIZE}x{MR_SIZE}x2 -> {HR_SIZE}x{HR_SIZE}x2")
    print("- dtype: float64")
    print("- channels: [u, v]")
    print("- schema: PhIRE native via utils.generate_TFRecords()")

    print("\nInstall into repo example_data/ after backing up originals:")
    print("  mkdir -p example_data_backup")
    print("  mv example_data/wind_*.tfrecord example_data_backup/")
    print(f"  cp {OUTDIR}/wind_*.tfrecord example_data/")

    print("\nThen test the normal flow, for example:")
    print("  python test_paired.py --data_in example_data/wind_MR-HR.tfrecord")

    print("\nFor your topology flow, crop 200x200 -> 160x160 during VTI conversion, e.g.")
    print("  --patch 160 --x0 20 --y0 20")
    return 0


if __name__ == "__main__":
    sys.exit(main())
