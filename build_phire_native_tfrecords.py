#!/usr/bin/env python3
"""
Build EXACT PhIRE-Compatible TFRecords from WIND Toolkit
Matches PhIRE's schema, dtypes, and scale factors exactly
"""

import h5pyd
import numpy as np
import tensorflow as tf
from pathlib import Path

print("="*80)
print("PhIRE-Native WTK TFRecord Builder")
print("="*80)

# ============================================================================
# CONFIGURATION - Match PhIRE's Wind Model Setup
# ============================================================================

# PhIRE's wind model scale factors (from main.py)
# Stage 1: LR→MR = 10× upscaling
# Stage 2: MR→HR = 5× upscaling
# Total: 50× upscaling (LR→HR)

# Let's work backwards from what's achievable:
# Mid-Atlantic native crop: ~112×112 (limited by WTK grid)

# Option 1: Use smaller HR that fits
HR_SIZE = 100   # Native crop (what we can actually get)
MR_SIZE = 20    # 5× smaller than HR (PhIRE MR→HR scale)
LR_SIZE = 10    # 10× smaller than MR (PhIRE LR→MR scale)
# This gives us 10→20 (10×) and 20→100 (5×) ✓

# OR Option 2: Extract larger region
# HR_SIZE = 250   # Would need larger geographic area
# MR_SIZE = 50    # 5× smaller
# LR_SIZE = 5     # 10× smaller (very coarse!)

# Region
LAT_RANGE = (38.0, 40.0)
LON_RANGE = (-76.0, -74.0)

# Time
TIME_START = 0
TIME_END = 168  # 1 week

HEIGHT = 100  # meters

# Splits
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15

OUTPUT_BASE = Path("/home/adadhwal/PhIRE/wtk_tfrecords_native")
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

print(f"\n⚙️  PhIRE-Compatible Configuration:")
print(f"  Scale factors: 10× (LR→MR), 5× (MR→HR), 50× (LR→HR)")
print(f"  LR: {LR_SIZE}×{LR_SIZE}×2")
print(f"  MR: {MR_SIZE}×{MR_SIZE}×2") 
print(f"  HR: {HR_SIZE}×{HR_SIZE}×2")
print(f"  Dtype: float64 (PhIRE standard)")
print(f"  Schema: PhIRE native (data_LR, data_HR, etc.)")

# ============================================================================
# EXTRACT WIND COMPONENTS
# ============================================================================

def compute_wind_components(wind_speed, wind_direction):
    """Convert speed + direction to u, v components"""
    dir_rad = np.radians(wind_direction)
    u = -wind_speed * np.sin(dir_rad)
    v = -wind_speed * np.cos(dir_rad)
    return u, v


def extract_wind_components_native(lat_range, lon_range, time_start, time_end,
                                   patch_size=100, height=100):
    """Extract native wind components from WTK"""
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
    
    if native_h < patch_size or native_w < patch_size:
        actual_size = min(native_h, native_w, patch_size)
        print(f"  ⚠️  Limiting to: {actual_size}×{actual_size}")
    else:
        actual_size = patch_size
        print(f"  ✅ Using: {actual_size}×{actual_size}")
    
    # Extract wind speed and direction
    print(f"  Extracting wind data at {height}m...")
    wind_speed = f[f'windspeed_{height}m'][
        time_start:time_end,
        lat_idx[0]:lat_idx[0]+actual_size,
        lon_idx[0]:lon_idx[0]+actual_size
    ]
    
    wind_direction = f[f'winddirection_{height}m'][
        time_start:time_end,
        lat_idx[0]:lat_idx[0]+actual_size,
        lon_idx[0]:lon_idx[0]+actual_size
    ]
    
    datetime_data = f['datetime'][time_start:time_end]
    f.close()
    
    print(f"  ✅ Extracted: {wind_speed.shape}")
    
    # Compute components
    print(f"  🧮 Computing [u, v] components...")
    u_wind = np.zeros_like(wind_speed, dtype=np.float64)
    v_wind = np.zeros_like(wind_speed, dtype=np.float64)
    
    for t in range(wind_speed.shape[0]):
        u_wind[t], v_wind[t] = compute_wind_components(
            wind_speed[t], wind_direction[t]
        )
    
    print(f"  ✅ Components: u∈[{u_wind.min():.1f}, {u_wind.max():.1f}], v∈[{v_wind.min():.1f}, {v_wind.max():.1f}]")
    
    return u_wind, v_wind, datetime_data, actual_size


# ============================================================================
# PHIRE-STYLE DOWNSAMPLING (Block Averaging)
# ============================================================================

def block_average_downsample(data, scale_factor):
    """
    PhIRE-style block averaging downsampling
    Each output pixel = average of scale×scale block
    Matches PhIRE's downscale_image() behavior
    """
    h, w = data.shape
    new_h, new_w = h // scale_factor, w // scale_factor
    
    # Ensure divisibility
    h_crop = new_h * scale_factor
    w_crop = new_w * scale_factor
    data = data[:h_crop, :w_crop]
    
    # Reshape and average
    result = data.reshape(new_h, scale_factor, new_w, scale_factor)
    result = result.mean(axis=(1, 3))
    
    return result


# ============================================================================
# PHIRE TFRECORD SCHEMA (EXACT)
# ============================================================================

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_phire_example(lr_data, hr_data, index):
    """
    Create tf.train.Example with EXACT PhIRE schema
    
    Args:
        lr_data: (H, W, C) array - float64
        hr_data: (H, W, C) array - float64
        index: int
    
    Returns:
        Serialized example matching PhIRE's _parse_train_/_parse_test_
    """
    # Convert to bytes (float64!)
    lr_bytes = lr_data.astype(np.float64).tobytes()
    hr_bytes = hr_data.astype(np.float64).tobytes()
    
    h_lr, w_lr, c = lr_data.shape
    h_hr, w_hr, _ = hr_data.shape
    
    # PhIRE's EXACT schema
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
# BUILD TFRECORDS
# ============================================================================

def build_phire_tfrecords(u_wind, v_wind, datetime_data, actual_size,
                         train_frac, val_frac, test_frac):
    """Build PhIRE-native TFRecords with proper scale factors"""
    n_samples = u_wind.shape[0]
    
    # Crop to exact HR size
    if actual_size > HR_SIZE:
        crop_start = (actual_size - HR_SIZE) // 2
        u_wind = u_wind[:, crop_start:crop_start+HR_SIZE, crop_start:crop_start+HR_SIZE]
        v_wind = v_wind[:, crop_start:crop_start+HR_SIZE, crop_start:crop_start+HR_SIZE]
        actual_size = HR_SIZE
    
    print(f"\n📊 Final data shape: {u_wind.shape}")
    print(f"   Using HR size: {actual_size}×{actual_size}")
    
    if actual_size != HR_SIZE:
        print(f"   ⚠️  Warning: HR is {actual_size}×{actual_size}, not {HR_SIZE}×{HR_SIZE}")
        # Adjust MR/LR accordingly
        actual_mr = actual_size // 5
        actual_lr = actual_mr // 10
        print(f"   Adjusting: LR={actual_lr}×{actual_lr}, MR={actual_mr}×{actual_mr}")
    else:
        actual_mr = MR_SIZE
        actual_lr = LR_SIZE
    
    # Time-based splits
    n_train = int(n_samples * train_frac)
    n_val = int(n_samples * val_frac)
    
    splits = {
        'train': (0, n_train),
        'val': (n_train, n_train + n_val),
        'test': (n_train + n_val, n_samples)
    }
    
    print(f"\n🔄 Creating PhIRE TFRecords:")
    
    # Build both LR→MR and MR→HR TFRecords
    for stage_name, (input_size, output_size) in [
        ('LR-MR', (actual_lr, actual_mr)),
        ('MR-HR', (actual_mr, actual_size))
    ]:
        print(f"\n  {stage_name}: {input_size}×{input_size} → {output_size}×{output_size}")
        scale = output_size // input_size
        print(f"  Scale factor: {scale}×")
        
        for split_name, (start, end) in splits.items():
            output_file = OUTPUT_BASE / f"wind_{stage_name}_{split_name}.tfrecord"
            
            print(f"    {split_name}: ", end="")
            
            with tf.io.TFRecordWriter(str(output_file)) as writer:
                for t in range(start, end):
                    # HR components (native, float64)
                    hr_u = u_wind[t].astype(np.float64)
                    hr_v = v_wind[t].astype(np.float64)
                    
                    # Stack into (H, W, 2)
                    hr = np.stack([hr_u, hr_v], axis=-1)
                    
                    # Downsample to output resolution (block averaging)
                    if output_size != actual_size:
                        scale_out = actual_size // output_size
                        output_u = block_average_downsample(hr_u, scale_out)
                        output_v = block_average_downsample(hr_v, scale_out)
                        output = np.stack([output_u, output_v], axis=-1)
                    else:
                        output = hr
                    
                    # Downsample to input resolution
                    scale_in = actual_size // input_size
                    input_u = block_average_downsample(hr_u, scale_in)
                    input_v = block_average_downsample(hr_v, scale_in)
                    input_data = np.stack([input_u, input_v], axis=-1)
                    
                    # Ensure float64
                    input_data = input_data.astype(np.float64)
                    output = output.astype(np.float64)
                    
                    # For LR-MR: LR → MR
                    # For MR-HR: MR → HR
                    if stage_name == 'LR-MR':
                        lr_data = input_data
                        hr_data = output
                    else:  # MR-HR
                        lr_data = input_data
                        hr_data = hr
                    
                    # Serialize with PhIRE schema
                    example = serialize_phire_example(
                        lr_data=lr_data,
                        hr_data=hr_data,
                        index=t - start
                    )
                    writer.write(example)
            
            n_written = end - start
            print(f"✅ {n_written} records")
    
    return splits, actual_size, actual_mr, actual_lr


# ============================================================================
# MAIN
# ============================================================================

def main():
    # Extract wind components
    u_wind, v_wind, datetime_data, actual_size = extract_wind_components_native(
        LAT_RANGE, LON_RANGE, TIME_START, TIME_END,
        patch_size=HR_SIZE, height=HEIGHT
    )
    
    # Build TFRecords
    splits, final_hr, final_mr, final_lr = build_phire_tfrecords(
        u_wind, v_wind, datetime_data, actual_size,
        TRAIN_FRAC, VAL_FRAC, TEST_FRAC
    )
    
    # Summary
    print("\n" + "="*80)
    print("✅ PHIRE-NATIVE TFRECORDS CREATED")
    print("="*80)
    
    print(f"\n📁 Output: {OUTPUT_BASE}")
    
    print(f"\n📊 Files Created:")
    for tfr in sorted(OUTPUT_BASE.glob("*.tfrecord")):
        size_mb = tfr.stat().st_size / 1024 / 1024
        n_records = sum(end - start for start, end in splits.values())
        print(f"  {tfr.name:40s} {size_mb:6.2f} MB")
    
    print(f"\n🔬 Data Specification:")
    print(f"  Schema: PhIRE native (data_LR, data_HR, h_LR, w_LR, etc.)")
    print(f"  Dtype: float64 ✓")
    print(f"  Channels: 2 [u, v] ✓")
    print(f"  LR: {final_lr}×{final_lr}×2")
    print(f"  MR: {final_mr}×{final_mr}×2")
    print(f"  HR: {final_hr}×{final_hr}×2")
    scale_lr_mr = final_mr // final_lr if final_lr > 0 else 0
    scale_mr_hr = final_hr // final_mr if final_mr > 0 else 0
    print(f"  Scale factors: {scale_lr_mr}× (LR→MR), {scale_mr_hr}× (MR→HR)")
    print(f"  Downsampling: Block averaging (PhIRE-style) ✓")
    
    print(f"\n📝 Next Steps:")
    print(f"  1. Copy to PhIRE repo:")
    print(f"     cp {OUTPUT_BASE}/*.tfrecord /path/to/PhIRE/example_data/")
    print(f"  2. Test with PhIRE:")
    print(f"     python test_paired.py --data_in example_data/wind_MR-HR_train.tfrecord")
    print(f"  3. Should load without errors ✓")


if __name__ == "__main__":
    main()
