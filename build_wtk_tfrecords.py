#!/usr/bin/env python3
"""
Build PhIRE-Compatible TFRecords from WIND Toolkit
Extracts wind components (u, v) and creates proper TFRecord format
"""

import h5pyd
import numpy as np
import tensorflow as tf
from pathlib import Path
from PIL import Image

print("="*80)
print("WTK → PhIRE TFRecord Builder")
print("="*80)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Resolution (matches your topology pipeline)
HR_SIZE = 160
MR_SIZE = 80
LR_SIZE = 40

# Region
LAT_RANGE = (38.0, 40.0)
LON_RANGE = (-76.0, -74.0)

# Time coverage
TIME_START = 0
TIME_END = 168  # 1 week

# Height for wind components
HEIGHT = 100  # meters

# Splits
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15

OUTPUT_BASE = Path("/home/adadhwal/PhIRE/wtk_tfrecords")
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

print(f"\n⚙️  Configuration:")
print(f"  Format: (H, W, 2) with channels [u, v]")
print(f"  HR: {HR_SIZE}×{HR_SIZE}")
print(f"  MR: {MR_SIZE}×{MR_SIZE}")
print(f"  LR: {LR_SIZE}×{LR_SIZE}")
print(f"  Time: {TIME_END} hours")

# ============================================================================
# EXTRACT WIND COMPONENTS (u, v)
# ============================================================================

def compute_wind_components(wind_speed, wind_direction):
    """
    Convert wind speed and direction to u, v components
    
    Convention:
    - wind_direction: meteorological (direction FROM which wind blows)
    - u: zonal (west-to-east) component
    - v: meridional (south-to-north) component
    """
    # Convert direction from degrees to radians
    # Meteorological convention: 0° = North, 90° = East, etc.
    dir_rad = np.radians(wind_direction)
    
    # Convert to math convention and get components
    # u = -speed * sin(dir) (negative because wind FROM direction)
    # v = -speed * cos(dir)
    u = -wind_speed * np.sin(dir_rad)
    v = -wind_speed * np.cos(dir_rad)
    
    return u, v


def extract_wind_components_native(lat_range, lon_range, time_start, time_end,
                                   patch_size=160, height=100):
    """
    Extract NATIVE wind component patches from WTK
    Returns (u, v) components, not just speed!
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
    
    print(f"  Native region: {native_h}×{native_w}")
    
    use_size = min(patch_size, native_h, native_w)
    print(f"  Using: {use_size}×{use_size}")
    
    # Extract BOTH wind speed AND direction
    print(f"  Extracting wind speed at {height}m...")
    wind_speed = f[f'windspeed_{height}m'][
        time_start:time_end,
        lat_idx[0]:lat_idx[0]+use_size,
        lon_idx[0]:lon_idx[0]+use_size
    ]
    
    print(f"  Extracting wind direction at {height}m...")
    wind_direction = f[f'winddirection_{height}m'][
        time_start:time_end,
        lat_idx[0]:lat_idx[0]+use_size,
        lon_idx[0]:lon_idx[0]+use_size
    ]
    
    datetime_data = f['datetime'][time_start:time_end]
    
    f.close()
    
    print(f"  ✅ Extracted: {wind_speed.shape}")
    
    # Convert to u, v components
    print(f"  🧮 Computing wind components (u, v)...")
    u_wind = np.zeros_like(wind_speed)
    v_wind = np.zeros_like(wind_speed)
    
    for t in range(wind_speed.shape[0]):
        u_wind[t], v_wind[t] = compute_wind_components(
            wind_speed[t], wind_direction[t]
        )
    
    print(f"  ✅ Components computed")
    print(f"     u range: {u_wind.min():.2f} to {u_wind.max():.2f} m/s")
    print(f"     v range: {v_wind.min():.2f} to {v_wind.max():.2f} m/s")
    
    return u_wind, v_wind, datetime_data, use_size


# ============================================================================
# ANTI-ALIASED DOWNSAMPLING
# ============================================================================

def downsample_antialiased(data, target_size):
    """Anti-aliased downsampling"""
    data_min, data_max = data.min(), data.max()
    data_norm = ((data - data_min) / (data_max - data_min) * 255).astype(np.uint8)
    
    img = Image.fromarray(data_norm)
    img_resized = img.resize((target_size, target_size), Image.LANCZOS)
    
    data_resized = np.array(img_resized, dtype=np.float32)
    data_resized = (data_resized / 255.0) * (data_max - data_min) + data_min
    
    return data_resized


# ============================================================================
# TFRECORD UTILITIES
# ============================================================================

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_example(lr_data, hr_data, index):
    """
    Create a tf.train.Example for LR→HR or MR→HR pair
    
    Args:
        lr_data: (H, W, 2) array - low/medium resolution
        hr_data: (H, W, 2) array - high resolution
        index: sample index
    """
    # Convert to bytes
    lr_bytes = lr_data.tobytes()
    hr_bytes = hr_data.tobytes()
    
    # Create feature dictionary
    feature = {
        'index': _int64_feature(index),
        'lr_shape_0': _int64_feature(lr_data.shape[0]),
        'lr_shape_1': _int64_feature(lr_data.shape[1]),
        'lr_shape_2': _int64_feature(lr_data.shape[2]),
        'hr_shape_0': _int64_feature(hr_data.shape[0]),
        'hr_shape_1': _int64_feature(hr_data.shape[1]),
        'hr_shape_2': _int64_feature(hr_data.shape[2]),
        'lr_data': _bytes_feature(lr_bytes),
        'hr_data': _bytes_feature(hr_bytes),
    }
    
    # Create Example
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()


# ============================================================================
# BUILD TFRECORDS WITH SPLITS
# ============================================================================

def build_tfrecords(u_wind, v_wind, datetime_data, patch_size,
                   train_frac, val_frac, test_frac):
    """
    Build TFRecords for LR→MR and MR→HR pairs
    """
    n_samples = u_wind.shape[0]
    
    # Time-based splits
    n_train = int(n_samples * train_frac)
    n_val = int(n_samples * val_frac)
    
    splits = {
        'train': (0, n_train),
        'val': (n_train, n_train + n_val),
        'test': (n_train + n_val, n_samples)
    }
    
    print(f"\n📊 Creating TFRecords:")
    
    # For each resolution pair
    for scale_name, (input_size, output_size) in [
        ('LR-MR', (LR_SIZE, MR_SIZE)),
        ('MR-HR', (MR_SIZE, HR_SIZE))
    ]:
        print(f"\n  {scale_name}: {input_size}×{input_size} → {output_size}×{output_size}")
        
        # For each split
        for split_name, (start, end) in splits.items():
            output_file = OUTPUT_BASE / f"wind_{scale_name}_{split_name}.tfrecord"
            
            print(f"    Writing {split_name}: {output_file.name}...", end=" ")
            
            with tf.io.TFRecordWriter(str(output_file)) as writer:
                for t in range(start, end):
                    # Get HR wind components (native)
                    hr_u = u_wind[t]
                    hr_v = v_wind[t]
                    
                    # Crop to correct size if needed
                    if hr_u.shape[0] != patch_size:
                        h, w = hr_u.shape
                        s_h = (h - patch_size) // 2
                        s_w = (w - patch_size) // 2
                        hr_u = hr_u[s_h:s_h+patch_size, s_w:s_w+patch_size]
                        hr_v = hr_v[s_h:s_h+patch_size, s_w:s_w+patch_size]
                    
                    # Create HR array (H, W, 2)
                    hr = np.stack([hr_u, hr_v], axis=-1).astype(np.float32)
                    
                    # Downsample to output resolution
                    output_u = downsample_antialiased(hr_u, output_size)
                    output_v = downsample_antialiased(hr_v, output_size)
                    output = np.stack([output_u, output_v], axis=-1).astype(np.float32)
                    
                    # Downsample to input resolution
                    input_u = downsample_antialiased(hr_u, input_size)
                    input_v = downsample_antialiased(hr_v, input_size)
                    input_data = np.stack([input_u, input_v], axis=-1).astype(np.float32)
                    
                    # For LR-MR: input=LR, output=MR
                    # For MR-HR: input=MR, output=HR
                    
                    # Serialize and write
                    example = serialize_example(
                        lr_data=input_data,
                        hr_data=output if scale_name == 'LR-MR' else hr,
                        index=t - start
                    )
                    writer.write(example)
            
            n_written = end - start
            print(f"✅ {n_written} records")
    
    return splits


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
    splits = build_tfrecords(
        u_wind, v_wind, datetime_data, actual_size,
        TRAIN_FRAC, VAL_FRAC, TEST_FRAC
    )
    
    # Summary
    print("\n" + "="*80)
    print("✅ TFRECORDS CREATED SUCCESSFULLY")
    print("="*80)
    
    print(f"\n📁 Output: {OUTPUT_BASE}")
    print(f"\n📊 Created Files:")
    for tfr_file in sorted(OUTPUT_BASE.glob("*.tfrecord")):
        size_mb = tfr_file.stat().st_size / 1024 / 1024
        print(f"  {tfr_file.name:40s} {size_mb:6.2f} MB")
    
    print(f"\n🔬 Data Format:")
    print(f"  Shape: (H, W, 2) where channels = [u, v]")
    print(f"  HR: {HR_SIZE}×{HR_SIZE}×2")
    print(f"  MR: {MR_SIZE}×{MR_SIZE}×2")
    print(f"  LR: {LR_SIZE}×{LR_SIZE}×2")
    print(f"  Native: ✅ YES")
    print(f"  Downsampling: Anti-aliased LANCZOS")
    
    print(f"\n📝 Usage:")
    print(f"  1. Copy to PhIRE/example_data/:")
    print(f"     cp {OUTPUT_BASE}/wind_*_train.tfrecord PhIRE/example_data/")
    print(f"  2. Run PhIRE inference:")
    print(f"     python test_paired.py --data_in example_data/wind_MR-HR_train.tfrecord")
    print(f"  3. Get output in data_out/:")
    print(f"     dataSR.npy, dataGT.npy, etc.")
    print(f"  4. Convert to VTI:")
    print(f"     python scripts/convert_phire_to_vti.py")
    print(f"  5. Run topology analysis:")
    print(f"     python compute_composite_tree_distance.py")


if __name__ == "__main__":
    main()
