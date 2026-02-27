#!/usr/bin/env python3
"""
Build EXACT clone of example_data behavior
- Exactly 2 files: wind_LR-MR.tfrecord, wind_MR-HR.tfrecord
- Exactly 5 samples each
- Native 160×160 HR (ENFORCED)
- Drop-in replacement for shipped example_data
"""

import h5pyd
import numpy as np
import tensorflow as tf
from pathlib import Path

print("="*80)
print("PhIRE example_data Clone Builder")
print("="*80)

# ============================================================================
# CONFIGURATION - EXACT CLONE TARGET
# ============================================================================

# CRITICAL: Must get native 160×160 HR!
HR_SIZE = 160   # Required for --patch 160

# Work backwards for proper scales
# Option 1: Match shipped example behavior (if possible)
# Option 2: Use practical scales that work with 160×160

# Let's use practical 2×5× for now:
MR_SIZE = 32    # 5× smaller than HR (160/5 = 32)
LR_SIZE = 16    # 2× smaller than MR (32/2 = 16)
# This gives: 16→32 (2×), 32→160 (5×)

# EXACTLY 5 samples (like example_data)
N_SAMPLES = 5
SAMPLE_INDICES = [0, 1, 2, 3, 4]  # First 5 hours

# Region - MUST be large enough for 160×160!
# Mid-Atlantic is too small (~112×112)
# Need to find larger region or different area

# Option A: Expand to larger Mid-Atlantic
LAT_RANGE = (37.0, 42.0)  # Wider range
LON_RANGE = (-77.0, -73.0)

# Option B: Different region (e.g., Great Plains)
# LAT_RANGE = (38.0, 43.0)
# LON_RANGE = (-100.0, -95.0)

HEIGHT = 100

OUTPUT_DIR = Path("/home/adadhwal/PhIRE/example_data_clone")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"\n⚙️  Clone Specifications:")
print(f"  Target: Exact example_data behavior")
print(f"  Files: wind_LR-MR.tfrecord, wind_MR-HR.tfrecord")
print(f"  Samples: {N_SAMPLES}")
print(f"  HR: {HR_SIZE}×{HR_SIZE}×2 (REQUIRED for --patch 160)")
print(f"  MR: {MR_SIZE}×{MR_SIZE}×2")
print(f"  LR: {LR_SIZE}×{LR_SIZE}×2")

# ============================================================================
# WIND COMPONENT EXTRACTION
# ============================================================================

def compute_wind_components(wind_speed, wind_direction):
    """Convert to u, v components"""
    dir_rad = np.radians(wind_direction)
    u = -wind_speed * np.sin(dir_rad)
    v = -wind_speed * np.cos(dir_rad)
    return u, v


def extract_wind_for_clone(lat_range, lon_range, sample_indices, height=100):
    """
    Extract wind data for clone - ENFORCES 160×160 minimum
    """
    print(f"\n📥 Opening WIND Toolkit...")
    f = h5pyd.File("/nrel/wtk-us.h5", "r")
    
    # Get coordinates
    coords = f['coordinates'][...]
    if coords.dtype.names:
        lats = coords['lat']
        lons = coords['lon']
    else:
        lats = coords[0]
        lons = coords[1]
    
    # Find region
    lat_mask = (lats[:, 0] >= lat_range[0]) & (lats[:, 0] <= lat_range[1])
    lon_mask = (lons[0, :] >= lon_range[0]) & (lons[0, :] <= lon_range[1])
    
    lat_idx = np.where(lat_mask)[0]
    lon_idx = np.where(lon_mask)[0]
    
    native_h = len(lat_idx)
    native_w = len(lon_idx)
    
    print(f"  Available region: {native_h}×{native_w}")
    
    # ENFORCE minimum size
    if native_h < HR_SIZE or native_w < HR_SIZE:
        f.close()
        raise ValueError(
            f"\n❌ FATAL: Region too small!\n"
            f"   Available: {native_h}×{native_w}\n"
            f"   Required: {HR_SIZE}×{HR_SIZE}\n"
            f"   Solution: Expand LAT_RANGE and LON_RANGE\n"
        )
    
    print(f"  ✅ Sufficient for {HR_SIZE}×{HR_SIZE} native HR")
    
    # Center crop to exact HR_SIZE
    h_start = (native_h - HR_SIZE) // 2
    w_start = (native_w - HR_SIZE) // 2
    
    print(f"  Cropping: [{h_start}:{h_start+HR_SIZE}, {w_start}:{w_start+HR_SIZE}]")
    
    # Extract for selected samples only
    n_samples = len(sample_indices)
    u_wind = np.zeros((n_samples, HR_SIZE, HR_SIZE), dtype=np.float64)
    v_wind = np.zeros((n_samples, HR_SIZE, HR_SIZE), dtype=np.float64)
    
    print(f"\n  Extracting {n_samples} samples at {height}m...")
    
    for i, time_idx in enumerate(sample_indices):
        # Extract wind speed and direction
        wind_speed = f[f'windspeed_{height}m'][
            time_idx,
            lat_idx[h_start]:lat_idx[h_start]+HR_SIZE,
            lon_idx[w_start]:lon_idx[w_start]+HR_SIZE
        ]
        
        wind_direction = f[f'winddirection_{height}m'][
            time_idx,
            lat_idx[h_start]:lat_idx[h_start]+HR_SIZE,
            lon_idx[w_start]:lon_idx[w_start]+HR_SIZE
        ]
        
        # Compute components
        u_wind[i], v_wind[i] = compute_wind_components(wind_speed, wind_direction)
        
        if i == 0:
            print(f"    Sample {time_idx}: u∈[{u_wind[i].min():.1f}, {u_wind[i].max():.1f}], "
                  f"v∈[{v_wind[i].min():.1f}, {v_wind[i].max():.1f}]")
    
    # Get timestamps
    datetime_data = f['datetime'][sample_indices]
    
    f.close()
    
    print(f"  ✅ Extracted {n_samples} samples, shape: {u_wind.shape}")
    
    return u_wind, v_wind, datetime_data


# ============================================================================
# FLOAT-PRESERVING BLOCK AVERAGING
# ============================================================================

def block_average_downsample(data, scale_factor):
    """
    PhIRE-style block averaging - NO PIL, NO uint8
    Pure float64 throughout
    """
    h, w = data.shape
    new_h, new_w = h // scale_factor, w // scale_factor
    
    # Crop to divisible size
    h_crop = new_h * scale_factor
    w_crop = new_w * scale_factor
    data = data[:h_crop, :w_crop]
    
    # Block average
    result = data.reshape(new_h, scale_factor, new_w, scale_factor)
    result = result.mean(axis=(1, 3))
    
    return result


# ============================================================================
# PHIRE TFRECORD SCHEMA
# ============================================================================

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_phire_example(lr_data, hr_data, index):
    """PhIRE-exact schema"""
    lr_bytes = lr_data.astype(np.float64).tobytes()
    hr_bytes = hr_data.astype(np.float64).tobytes()
    
    h_lr, w_lr, c = lr_data.shape
    h_hr, w_hr, _ = hr_data.shape
    
    feature = {
        'data_LR': _bytes_feature(lr_bytes),
        'h_LR': _int64_feature(h_lr),
        'w_LR': _int64_feature(w_lr),
        'data_HR': _bytes_feature(hr_bytes),
        'h_HR': _int64_feature(h_hr),
        'w_HR': _int64_feature(w_hr),
        'c': _int64_feature(c),
        'index': _int64_feature(index),
    }
    
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()


# ============================================================================
# BUILD CLONE TFRECORDS
# ============================================================================

def build_clone_tfrecords(u_wind, v_wind):
    """
    Build EXACT clone:
    - wind_LR-MR.tfrecord (5 records)
    - wind_MR-HR.tfrecord (5 records)
    """
    n_samples = u_wind.shape[0]
    
    print(f"\n🔄 Building Clone TFRecords ({n_samples} samples)...")
    
    # Compute all resolutions
    print(f"\n  Preparing resolution hierarchy:")
    print(f"    HR: {HR_SIZE}×{HR_SIZE}")
    print(f"    MR: {MR_SIZE}×{MR_SIZE} (downsample by {HR_SIZE//MR_SIZE}×)")
    print(f"    LR: {LR_SIZE}×{LR_SIZE} (downsample by {HR_SIZE//LR_SIZE}×)")
    
    # Pre-compute all samples at all resolutions
    hr_stack = []
    mr_stack = []
    lr_stack = []
    
    for i in range(n_samples):
        # HR (native)
        hr_u = u_wind[i]
        hr_v = v_wind[i]
        hr = np.stack([hr_u, hr_v], axis=-1).astype(np.float64)
        hr_stack.append(hr)
        
        # MR (block average from HR)
        mr_u = block_average_downsample(hr_u, HR_SIZE // MR_SIZE)
        mr_v = block_average_downsample(hr_v, HR_SIZE // MR_SIZE)
        mr = np.stack([mr_u, mr_v], axis=-1).astype(np.float64)
        mr_stack.append(mr)
        
        # LR (block average from HR)
        lr_u = block_average_downsample(hr_u, HR_SIZE // LR_SIZE)
        lr_v = block_average_downsample(hr_v, HR_SIZE // LR_SIZE)
        lr = np.stack([lr_u, lr_v], axis=-1).astype(np.float64)
        lr_stack.append(lr)
    
    # Build LR→MR TFRecord
    print(f"\n  Writing wind_LR-MR.tfrecord...")
    tfr_lr_mr = OUTPUT_DIR / "wind_LR-MR.tfrecord"
    with tf.io.TFRecordWriter(str(tfr_lr_mr)) as writer:
        for i in range(n_samples):
            example = serialize_phire_example(
                lr_data=lr_stack[i],
                hr_data=mr_stack[i],
                index=i
            )
            writer.write(example)
    print(f"    ✅ {n_samples} records, {tfr_lr_mr.stat().st_size/1024/1024:.2f} MB")
    
    # Build MR→HR TFRecord
    print(f"\n  Writing wind_MR-HR.tfrecord...")
    tfr_mr_hr = OUTPUT_DIR / "wind_MR-HR.tfrecord"
    with tf.io.TFRecordWriter(str(tfr_mr_hr)) as writer:
        for i in range(n_samples):
            example = serialize_phire_example(
                lr_data=mr_stack[i],
                hr_data=hr_stack[i],
                index=i
            )
            writer.write(example)
    print(f"    ✅ {n_samples} records, {tfr_mr_hr.stat().st_size/1024/1024:.2f} MB")
    
    return tfr_lr_mr, tfr_mr_hr


# ============================================================================
# MAIN
# ============================================================================

def main():
    try:
        # Extract wind components
        u_wind, v_wind, datetime_data = extract_wind_for_clone(
            LAT_RANGE, LON_RANGE, SAMPLE_INDICES, HEIGHT
        )
        
        # Build clone TFRecords
        tfr_lr_mr, tfr_mr_hr = build_clone_tfrecords(u_wind, v_wind)
        
        # Summary
        print("\n" + "="*80)
        print("✅ EXAMPLE_DATA CLONE CREATED")
        print("="*80)
        
        print(f"\n📁 Output Directory: {OUTPUT_DIR}")
        print(f"\n📊 Files Created:")
        print(f"  {tfr_lr_mr.name}")
        print(f"  {tfr_mr_hr.name}")
        
        print(f"\n🔬 Specifications:")
        print(f"  Samples: {N_SAMPLES}")
        print(f"  HR: {HR_SIZE}×{HR_SIZE}×2 (native, no interpolation)")
        print(f"  MR: {MR_SIZE}×{MR_SIZE}×2")
        print(f"  LR: {LR_SIZE}×{LR_SIZE}×2")
        print(f"  Channels: [u, v]")
        print(f"  Dtype: float64")
        print(f"  Downsampling: Block averaging (float-preserving)")
        
        print(f"\n📝 Installation:")
        print(f"  # Backup original (if desired):")
        print(f"  mv /path/to/PhIRE/example_data/wind_*.tfrecord /path/to/backup/")
        print(f"  ")
        print(f"  # Install clone:")
        print(f"  cp {OUTPUT_DIR}/wind_*.tfrecord /path/to/PhIRE/example_data/")
        
        print(f"\n🧪 Testing:")
        print(f"  cd /path/to/PhIRE")
        print(f"  python test_paired.py --data_in example_data/wind_MR-HR.tfrecord")
        print(f"  # Should produce dataSR.npy, dataGT.npy with 5 samples")
        
        print(f"\n📐 Topology Pipeline:")
        print(f"  python scripts/convert_phire_to_vti.py --patch 160  # ✓ Matches HR size")
        print(f"  python compute_composite_tree_distance.py ...")
        
        print(f"\n✅ Clone complete! Drop-in replacement for example_data")
        
    except ValueError as e:
        print(f"\n{e}")
        print(f"\n💡 Solutions:")
        print(f"  1. Expand LAT_RANGE and LON_RANGE to cover larger area")
        print(f"  2. Try different geographic region (e.g., Great Plains)")
        print(f"  3. Accept smaller HR_SIZE (but must update --patch accordingly)")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
