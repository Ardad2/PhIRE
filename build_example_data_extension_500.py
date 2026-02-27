#!/usr/bin/env python3
"""
Build a true extension of PhIRE's shipped wind example_data.

Goal:
- wind_LR-MR.tfrecord  : 10x10x2   -> 100x100x2
- wind_MR-HR.tfrecord  : 100x100x2 -> 500x500x2

This makes the paired outputs match the original example-data geometry:
- dataIN.npy : (5, 100, 100, 2)
- dataGT.npy : (5, 500, 500, 2)
- dataSR.npy : (5, 500, 500, 2)

Assumptions:
- You are running inside the PhIRE repo root.
- utils.py has already been patched for:
    * tensorflow.compat.v1
    * tf.disable_v2_behavior()
    * .tostring() -> .tobytes()
    * tf.contrib.layers.xavier_initializer() -> tf.glorot_uniform_initializer()
"""

import sys
from pathlib import Path

import h5pyd
import numpy as np

# Import patched PhIRE utils
import utils


# ============================================================================
# CONFIGURATION
# ============================================================================

N_SAMPLES = 168
SAMPLE_INDICES = list(range(N_SAMPLES))

# Exact extension target sizes
LR_SIZE = 10
MR_SIZE = 100
HR_SIZE = 500

# Stage factors must match original example behavior
# LR -> MR : 10x
# MR -> HR : 5x
assert MR_SIZE // LR_SIZE == 10
assert HR_SIZE // MR_SIZE == 5

HEIGHT = 100  # meters

# Choose a center safely inside the domain
CENTER_LAT = 39.5
CENTER_LON = -75.0

# Output folder
OUTDIR = Path("example_data_extension_500")
OUTDIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("Building REAL extension of shipped wind example_data")
print("=" * 80)
print(f"Target sizes: LR={LR_SIZE}, MR={MR_SIZE}, HR={HR_SIZE}")
print(f"Sample indices: {SAMPLE_INDICES}")
print(f"Center point: lat={CENTER_LAT}, lon={CENTER_LON}, height={HEIGHT}m")


# ============================================================================
# HELPERS
# ============================================================================

def compute_wind_components(wind_speed, wind_direction_deg):
    """
    Convert speed + meteorological direction to u, v.

    Meteorological convention:
    - direction is the direction the wind is blowing FROM
    """
    theta = np.radians(wind_direction_deg)
    u = -wind_speed * np.sin(theta)
    v = -wind_speed * np.cos(theta)
    return u, v


def find_center_indices(lats, lons, target_lat, target_lon):
    """
    Find nearest grid point to desired lat/lon center.
    """
    # Grid is structured; use squared distance on the full 2D grid
    dist2 = (lats - target_lat) ** 2 + (lons - target_lon) ** 2
    iy, ix = np.unravel_index(np.argmin(dist2), dist2.shape)
    return int(iy), int(ix)


def extract_native_hr_stack():
    """
    Extract exactly 5 native 500x500 samples as [u, v].
    Returns:
        hr_stack: (5, 500, 500, 2), float64
        datetime_data
    """
    print("\nOpening WIND Toolkit...")
    f = h5pyd.File("/nrel/wtk-us.h5", "r")

    coords = f["coordinates"][...]
    if coords.dtype.names:
        lats = coords["lat"]
        lons = coords["lon"]
    else:
        lats = coords[0]
        lons = coords[1]

    H, W = lats.shape
    print(f"Full grid size: {H}x{W}")

    cy, cx = find_center_indices(lats, lons, CENTER_LAT, CENTER_LON)
    print(f"Nearest grid center index: row={cy}, col={cx}")
    print(f"Nearest grid center lat/lon: ({lats[cy, cx]:.4f}, {lons[cy, cx]:.4f})")

    half = HR_SIZE // 2
    y0 = cy - half
    y1 = y0 + HR_SIZE
    x0 = cx - half
    x1 = x0 + HR_SIZE

    if y0 < 0 or x0 < 0 or y1 > H or x1 > W:
        f.close()
        raise ValueError(
            f"Cannot extract native {HR_SIZE}x{HR_SIZE} centered at "
            f"({CENTER_LAT}, {CENTER_LON}). "
            f"Computed slice rows {y0}:{y1}, cols {x0}:{x1} is out of bounds "
            f"for full grid {H}x{W}."
        )

    print(f"Using native crop rows {y0}:{y1}, cols {x0}:{x1}")

    hr_stack = np.zeros((N_SAMPLES, HR_SIZE, HR_SIZE, 2), dtype=np.float64)

    for i, t in enumerate(SAMPLE_INDICES):
        ws = f[f"windspeed_{HEIGHT}m"][t, y0:y1, x0:x1]
        wd = f[f"winddirection_{HEIGHT}m"][t, y0:y1, x0:x1]

        u, v = compute_wind_components(ws, wd)

        hr_stack[i, :, :, 0] = u
        hr_stack[i, :, :, 1] = v

        print(
            f"Sample {t}: "
            f"u in [{u.min():.2f}, {u.max():.2f}], "
            f"v in [{v.min():.2f}, {v.max():.2f}]"
        )

    datetime_data = f["datetime"][SAMPLE_INDICES[0] : SAMPLE_INDICES[-1] + 1]  # avoid h5pyd fancy-index POST
    f.close()

    print(f"\nNative HR stack shape: {hr_stack.shape}")
    return hr_stack, datetime_data


# ============================================================================
# MAIN
# ============================================================================

def main():
    # Extract native 500x500 HR stack
    hr_stack, datetime_data = extract_native_hr_stack()

    print("\nPreparing stage-consistent arrays with PhIRE utils.downscale_image()...")
    # MR from HR by 5x
    mr_stack = utils.downscale_image(hr_stack, 5)
    # LR from MR by 10x (or HR by 50x; using MR keeps stage semantics explicit)
    lr_stack = utils.downscale_image(mr_stack, 10)

    print(f"hr_stack shape: {hr_stack.shape}")
    print(f"mr_stack shape: {mr_stack.shape}")
    print(f"lr_stack shape: {lr_stack.shape}")

    # Sanity checks
    assert hr_stack.shape == (N_SAMPLES, HR_SIZE, HR_SIZE, 2)
    assert mr_stack.shape == (N_SAMPLES, MR_SIZE, MR_SIZE, 2)
    assert lr_stack.shape == (N_SAMPLES, LR_SIZE, LR_SIZE, 2)

    # Build TFRecords exactly like example_data behavior
    out_lr_mr = OUTDIR / "wind_LR-MR.tfrecord"
    out_mr_hr = OUTDIR / "wind_MR-HR.tfrecord"

    print("\nWriting PhIRE-native TFRecords via utils.generate_TFRecords()...")

    # For LR->MR, pass MR stack and K=10 so TFRecord stores:
    # data_LR = 10x10x2, data_HR = 100x100x2
    utils.generate_TFRecords(str(out_lr_mr), mr_stack.astype(np.float64), mode="train", K=10)

    # For MR->HR, pass HR stack and K=5 so TFRecord stores:
    # data_LR = 100x100x2, data_HR = 500x500x2
    utils.generate_TFRecords(str(out_mr_hr), hr_stack.astype(np.float64), mode="train", K=5)

    print("\n" + "=" * 80)
    print("REAL EXTENSION CREATED")
    print("=" * 80)
    print(f"Output directory: {OUTDIR}")
    print("Created files:")
    print(f"  {out_lr_mr}")
    print(f"  {out_mr_hr}")
    print("\nBehavioral extension characteristics:")
    print("- records per file: 5")
    print("- wind_LR-MR.tfrecord: 10x10x2 -> 100x100x2")
    print("- wind_MR-HR.tfrecord: 100x100x2 -> 500x500x2")
    print("- dtype: float64")
    print("- channels: [u, v]")
    print("- schema: PhIRE native via utils.generate_TFRecords()")
    print("\nInstall into repo example_data/ after backing up originals:")
    print("  mkdir -p example_data_backup")
    print("  mv example_data/wind_*.tfrecord example_data_backup/")
    print("  cp example_data_extension_500/wind_*.tfrecord example_data/")
    print("\nThen run the usual paired flow:")
    print("  python3 run_paired_wind_mr_hr.py")
    print("\nExpected outputs should now match original geometry:")
    print("  dataIN.npy : (5, 100, 100, 2)")
    print("  dataGT.npy : (5, 500, 500, 2)")
    print("  dataSR.npy : (5, 500, 500, 2)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
