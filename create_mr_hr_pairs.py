#!/usr/bin/env python3
"""
Create MR→HR pairs from WIND Toolkit data
MR (Medium Resolution) → HR (High Resolution)
"""

import numpy as np
from pathlib import Path
from scipy.ndimage import zoom

print("="*80)
print("Creating MR→HR Pairs from WIND Toolkit")
print("="*80)

# Configuration
MR_SIZE = 32   # Medium resolution (input to SR model)
HR_SIZE = 64   # High resolution (target/ground truth)
SCALE = HR_SIZE // MR_SIZE

# Load extracted data
data_dir = Path("/home/adadhwal/PhIRE/wind_toolkit/midatlantic")
input_file = data_dir / "wind_midatlantic_t0_24.npz"

if not input_file.exists():
    print(f"❌ File not found: {input_file}")
    print("   Run extract_midatlantic_grid.py first!")
    exit(1)

print(f"\n📥 Loading: {input_file.name}")
data = np.load(input_file, allow_pickle=True)
wind_speed_hr = data['wind_speed']  # Shape: (24, 64, 64)
datetime_data = data['datetime']

print(f"  Shape: {wind_speed_hr.shape}")
print(f"  Mean wind speed: {wind_speed_hr.mean():.2f} m/s")

# Output configuration
print(f"\n⚙️  Configuration:")
print(f"  HR (Ground Truth): {HR_SIZE}×{HR_SIZE}")
print(f"  MR (Input): {MR_SIZE}×{MR_SIZE}")
print(f"  Upscale factor: {SCALE}×")

output_dir = Path("/home/adadhwal/PhIRE/wind_toolkit/mr_hr_pairs")
output_dir.mkdir(parents=True, exist_ok=True)

# Create pairs
n_samples = wind_speed_hr.shape[0]
print(f"\n🔄 Creating {n_samples} MR→HR pairs...")

for t in range(n_samples):
    # Ground truth (HR) - already at target resolution
    hr = wind_speed_hr[t]
    
    # Medium resolution (MR) - downsample from HR
    mr = zoom(hr, MR_SIZE / HR_SIZE, order=3)
    
    # Verify sizes
    assert hr.shape == (HR_SIZE, HR_SIZE), f"HR wrong size: {hr.shape}"
    assert mr.shape == (MR_SIZE, MR_SIZE), f"MR wrong size: {mr.shape}"
    
    # Get timestamp string
    timestamp = str(datetime_data[t]).replace("b'", "").replace("'", "")
    sample_id = f"wind_toolkit_{timestamp}"
    
    # Save as numpy array
    np.savez(
        output_dir / f"{sample_id}.npz",
        HR=hr,        # Ground truth (64×64)
        MR=mr,        # Medium resolution input (32×32)
        metadata={
            'timestamp': timestamp,
            'timestep': t,
            'hr_size': HR_SIZE,
            'mr_size': MR_SIZE,
            'scale': SCALE,
            'mean_windspeed_hr': hr.mean(),
            'max_windspeed_hr': hr.max(),
            'mean_windspeed_mr': mr.mean()
        }
    )
    
    if t % 6 == 0:
        print(f"  ✅ {sample_id}")

print(f"\n💾 Saved {n_samples} MR→HR pairs to: {output_dir}")

# Summary statistics
print(f"\n📊 Dataset Summary:")
print(f"  Total pairs: {n_samples}")
print(f"  HR (Ground Truth): {HR_SIZE}×{HR_SIZE}")
print(f"  MR (Input): {MR_SIZE}×{MR_SIZE}")
print(f"  Scale factor: {SCALE}×")
print(f"  Wind speed range: {wind_speed_hr.min():.2f} - {wind_speed_hr.max():.2f} m/s")
print(f"  Temporal coverage: {datetime_data[0]} to {datetime_data[-1]}")

print(f"\n✅ Ready for SR training/testing!")
print(f"\n📝 Next steps:")
print(f"  1. Train your SR model on MR (32×32) → HR (64×64)")
print(f"  2. Generate SR predictions from MR inputs")
print(f"  3. Compare SR vs HR using your topology pipeline")
print(f"  4. Compute PD/MT distances")
print(f"  5. Validate physics correlations!")
