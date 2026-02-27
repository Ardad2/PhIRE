#!/usr/bin/env python3
"""
Build Research-Grade WTK Dataset
Native HR patches → Proper downsampling → Train/Val/Test splits
"""

import h5pyd
import numpy as np
from pathlib import Path
from PIL import Image
import argparse

print("="*80)
print("WIND Toolkit Research-Grade Dataset Builder")
print("="*80)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Resolution hierarchy (matches your topology pipeline)
HR_SIZE = 160   # Native crop size (matches --patch 160)
MR_SIZE = 80    # 2× downscale
LR_SIZE = 40    # 4× downscale

# Region (Mid-Atlantic)
LAT_RANGE = (38.0, 40.0)
LON_RANGE = (-76.0, -74.0)

# Time coverage (in hours)
TIME_START = 0
TIME_END = 168  # 1 week (7 days × 24 hours)

# Train/Val/Test splits (time-based, not random!)
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15

# Height
HEIGHT = 100  # meters

# Output
OUTPUT_BASE = Path("/home/adadhwal/PhIRE/wind_toolkit_research")

print(f"\n⚙️  Configuration:")
print(f"  Native HR: {HR_SIZE}×{HR_SIZE} (no interpolation!)")
print(f"  MR: {MR_SIZE}×{MR_SIZE} (2× downscale)")
print(f"  LR: {LR_SIZE}×{LR_SIZE} (4× downscale)")
print(f"  Time window: {TIME_END} hours ({TIME_END//24} days)")
print(f"  Train/Val/Test: {TRAIN_FRAC:.0%}/{VAL_FRAC:.0%}/{TEST_FRAC:.0%}")

# ============================================================================
# ANTI-ALIASED DOWNSAMPLING (Better than cubic!)
# ============================================================================

def downsample_antialiased(data, target_size):
    """
    Proper anti-aliased downsampling using PIL
    Much better than scipy zoom for degradation
    """
    # Convert to PIL Image
    # Normalize to 0-255 for PIL
    data_min, data_max = data.min(), data.max()
    data_norm = ((data - data_min) / (data_max - data_min) * 255).astype(np.uint8)
    
    img = Image.fromarray(data_norm)
    
    # Resize with LANCZOS (anti-aliased)
    img_resized = img.resize((target_size, target_size), Image.LANCZOS)
    
    # Convert back to float and denormalize
    data_resized = np.array(img_resized, dtype=np.float32)
    data_resized = (data_resized / 255.0) * (data_max - data_min) + data_min
    
    return data_resized


def downsample_area_average(data, scale_factor):
    """
    Alternative: Block averaging (physically motivated)
    Each MR pixel = average of scale×scale HR block
    """
    h, w = data.shape
    new_h, new_w = h // scale_factor, w // scale_factor
    
    result = np.zeros((new_h, new_w))
    for i in range(new_h):
        for j in range(new_w):
            block = data[
                i*scale_factor:(i+1)*scale_factor,
                j*scale_factor:(j+1)*scale_factor
            ]
            result[i, j] = block.mean()
    
    return result


# ============================================================================
# EXTRACT NATIVE PATCHES
# ============================================================================

def extract_native_patches(lat_range, lon_range, time_start, time_end, 
                          patch_size=160, height=100):
    """
    Extract NATIVE patches from WTK (no resizing!)
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
    
    if native_h < patch_size or native_w < patch_size:
        print(f"  ⚠️  Native region too small for {patch_size}×{patch_size} patches!")
        print(f"     Extracting maximum available: {native_h}×{native_w}")
        use_size = min(native_h, native_w)
    else:
        use_size = patch_size
        print(f"  ✅ Can extract native {patch_size}×{patch_size} patches")
    
    # Extract wind speed
    print(f"  Extracting timesteps {time_start} to {time_end}...")
    wind_data = f[f'windspeed_{height}m'][
        time_start:time_end,
        lat_idx[0]:lat_idx[0]+use_size,
        lon_idx[0]:lon_idx[0]+use_size
    ]
    
    datetime_data = f['datetime'][time_start:time_end]
    
    f.close()
    
    print(f"  ✅ Extracted: {wind_data.shape}")
    print(f"  ✅ Timestamps: {len(datetime_data)}")
    
    return wind_data, datetime_data, use_size


# ============================================================================
# CREATE DATASET WITH SPLITS
# ============================================================================

def create_dataset_with_splits(wind_data, datetime_data, patch_size,
                               train_frac, val_frac, test_frac):
    """
    Create train/val/test splits (TIME-BASED, not random!)
    """
    n_samples = wind_data.shape[0]
    
    # Time-based splits
    n_train = int(n_samples * train_frac)
    n_val = int(n_samples * val_frac)
    # n_test = remaining
    
    splits = {
        'train': (0, n_train),
        'val': (n_train, n_train + n_val),
        'test': (n_train + n_val, n_samples)
    }
    
    print(f"\n📊 Dataset Splits:")
    for split_name, (start, end) in splits.items():
        print(f"  {split_name:5s}: timesteps {start:3d}-{end:3d} ({end-start:3d} samples)")
    
    # Create output directories
    for split_name in splits.keys():
        split_dir = OUTPUT_BASE / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each split
    for split_name, (start, end) in splits.items():
        print(f"\n🔄 Creating {split_name} set...")
        split_dir = OUTPUT_BASE / split_name
        
        for t in range(start, end):
            # Native HR (no resizing!)
            hr = wind_data[t]
            
            # Ensure square and correct size
            if hr.shape[0] != patch_size or hr.shape[1] != patch_size:
                # Center crop if needed
                h, w = hr.shape
                start_h = (h - patch_size) // 2
                start_w = (w - patch_size) // 2
                hr = hr[start_h:start_h+patch_size, start_w:start_w+patch_size]
            
            # Downsample with anti-aliasing
            mr = downsample_antialiased(hr, MR_SIZE)
            lr = downsample_antialiased(hr, LR_SIZE)
            
            # Add channel dimension (H, W, C) format
            hr = hr[..., np.newaxis]  # (160, 160, 1)
            mr = mr[..., np.newaxis]  # (80, 80, 1)
            lr = lr[..., np.newaxis]  # (40, 40, 1)
            
            # Get timestamp
            timestamp = str(datetime_data[t]).replace("b'", "").replace("'", "")
            sample_id = f"wind_{timestamp}"
            
            # Save
            np.savez_compressed(
                split_dir / f"{sample_id}.npz",
                HR=hr,
                MR=mr,
                LR=lr,
                metadata={
                    'timestamp': timestamp,
                    'timestep': t,
                    'split': split_name,
                    'hr_size': HR_SIZE,
                    'mr_size': MR_SIZE,
                    'lr_size': LR_SIZE,
                    'height_m': HEIGHT,
                    'native': True,  # Flag: this is native HR!
                    'downsampling': 'antialiased_lanczos',
                    'mean_windspeed_hr': float(hr.mean()),
                    'max_windspeed_hr': float(hr.max()),
                    'min_windspeed_hr': float(hr.min())
                }
            )
            
            if (t - start) % 24 == 0:
                print(f"  ✅ {sample_id} (day {(t-start)//24 + 1})")
    
    return splits


# ============================================================================
# MAIN
# ============================================================================

def main():
    # Extract native patches
    wind_data, datetime_data, actual_size = extract_native_patches(
        LAT_RANGE, LON_RANGE, TIME_START, TIME_END,
        patch_size=HR_SIZE, height=HEIGHT
    )
    
    # Create dataset with splits
    splits = create_dataset_with_splits(
        wind_data, datetime_data, actual_size,
        TRAIN_FRAC, VAL_FRAC, TEST_FRAC
    )
    
    # Summary
    print("\n" + "="*80)
    print("✅ DATASET CREATED SUCCESSFULLY")
    print("="*80)
    
    print(f"\n📁 Output: {OUTPUT_BASE}")
    print(f"\n📊 Summary:")
    total_samples = sum(end - start for start, end in splits.values())
    print(f"  Total samples: {total_samples}")
    print(f"  Resolution chain: {LR_SIZE}×{LR_SIZE} → {MR_SIZE}×{MR_SIZE} → {HR_SIZE}×{HR_SIZE}")
    print(f"  Format: (H, W, C) with C=1 (scalar wind speed)")
    print(f"  Downsampling: Anti-aliased LANCZOS")
    print(f"  Native HR: ✅ YES (no interpolation)")
    
    print(f"\n🔬 Data Statistics:")
    print(f"  Mean wind speed: {wind_data.mean():.2f} m/s")
    print(f"  Std: {wind_data.std():.2f} m/s")
    print(f"  Range: {wind_data.min():.2f} - {wind_data.max():.2f} m/s")
    
    print(f"\n📝 Next Steps:")
    print(f"  1. Verify format matches your SR pipeline")
    print(f"  2. Train SR model on train set")
    print(f"  3. Validate on val set")
    print(f"  4. Test on test set")
    print(f"  5. Run topology analysis on test set outputs")
    print(f"  6. Validate r=0.92 correlation!")
    
    print(f"\n💡 To expand dataset:")
    print(f"  - Edit TIME_END for more data (168→720 for 1 month)")
    print(f"  - Run for different seasons (Jan, Apr, Jul, Oct)")
    print(f"  - Extract multiple regions")


if __name__ == "__main__":
    main()
