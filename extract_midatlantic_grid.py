#!/usr/bin/env python3
"""
Extract Mid-Atlantic region - FIXED for structured coordinates
"""

import h5pyd
import numpy as np
from pathlib import Path

print("="*80)
print("Mid-Atlantic Grid Extraction")
print("="*80)

# Configuration
LAT_RANGE = (38.0, 40.0)
LON_RANGE = (-76.0, -74.0)
TIME_START = 0
TIME_END = 24  # First 24 hours
TARGET_SIZE = 64

output_dir = Path("/home/adadhwal/PhIRE/wind_toolkit/midatlantic")
output_dir.mkdir(parents=True, exist_ok=True)

print(f"\n📍 Configuration:")
print(f"  Region: {LAT_RANGE[0]}° to {LAT_RANGE[1]}°N")
print(f"          {LON_RANGE[0]}° to {LON_RANGE[1]}°E")
print(f"  Time: indices {TIME_START} to {TIME_END}")
print(f"  Target size: {TARGET_SIZE}×{TARGET_SIZE}")

# Open dataset
print(f"\n📥 Opening dataset...")
f = h5pyd.File("/nrel/wtk-us.h5", "r")

# Get coordinates - handle structured array
coords = f['coordinates'][...]
if coords.dtype.names:
    lats = coords['lat']
    lons = coords['lon']
else:
    lats = coords[0]
    lons = coords[1]

print(f"  Grid size: {lats.shape}")
print(f"  Lat range: {lats.min():.2f}° to {lats.max():.2f}°")
print(f"  Lon range: {lons.min():.2f}° to {lons.max():.2f}°")

# Find region indices
lat_mask = (lats[:, 0] >= LAT_RANGE[0]) & (lats[:, 0] <= LAT_RANGE[1])
lon_mask = (lons[0, :] >= LON_RANGE[0]) & (lons[0, :] <= LON_RANGE[1])

lat_idx = np.where(lat_mask)[0]
lon_idx = np.where(lon_mask)[0]

if len(lat_idx) == 0 or len(lon_idx) == 0:
    print(f"\n❌ ERROR: Region not in dataset!")
    print(f"   Requested: Lat {LAT_RANGE}, Lon {LON_RANGE}")
    print(f"   Available: Lat {lats.min():.1f}-{lats.max():.1f}, Lon {lons.min():.1f}-{lons.max():.1f}")
    f.close()
    exit(1)

print(f"\n🔍 Found region:")
print(f"  Lat indices: {lat_idx[0]} to {lat_idx[-1]} ({len(lat_idx)} points)")
print(f"  Lon indices: {lon_idx[0]} to {lon_idx[-1]} ({len(lon_idx)} points)")
print(f"  Native grid: {len(lat_idx)}×{len(lon_idx)}")

# Extract wind speed
print(f"\n📥 Extracting wind speed at 100m...")
print(f"  Time range: {TIME_START} to {TIME_END}")

wind_data = f['windspeed_100m'][
    TIME_START:TIME_END,
    lat_idx[0]:lat_idx[-1]+1,
    lon_idx[0]:lon_idx[-1]+1
]

print(f"  ✅ Extracted: {wind_data.shape}")

# Get timestamps
datetime_data = f['datetime'][TIME_START:TIME_END]
print(f"  ✅ Timestamps: {len(datetime_data)}")

f.close()

# Resize if needed
if wind_data.shape[1] != TARGET_SIZE or wind_data.shape[2] != TARGET_SIZE:
    print(f"\n🔄 Resizing to {TARGET_SIZE}×{TARGET_SIZE}...")
    from scipy.ndimage import zoom
    
    zoom_factors = (
        1.0,
        TARGET_SIZE / wind_data.shape[1],
        TARGET_SIZE / wind_data.shape[2]
    )
    
    wind_data_resized = zoom(wind_data, zoom_factors, order=3)
    print(f"  ✅ Resized: {wind_data_resized.shape}")
else:
    wind_data_resized = wind_data
    print(f"  ℹ️  No resize needed")

# Save
output_file = output_dir / f"wind_midatlantic_t{TIME_START}_{TIME_END}.npz"
np.savez_compressed(
    output_file,
    wind_speed=wind_data_resized,
    datetime=datetime_data,
    lat_indices=lat_idx,
    lon_indices=lon_idx,
    metadata={
        'lat_range': LAT_RANGE,
        'lon_range': LON_RANGE,
        'time_start': TIME_START,
        'time_end': TIME_END,
        'native_shape': wind_data.shape,
        'resized_shape': wind_data_resized.shape
    }
)

print(f"\n💾 Saved: {output_file}")
print(f"   Size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")

# Summary
print(f"\n📊 Data Summary:")
print(f"  Shape: {wind_data_resized.shape} (time, lat, lon)")
print(f"  Mean: {wind_data_resized.mean():.2f} m/s")
print(f"  Std:  {wind_data_resized.std():.2f} m/s")
print(f"  Range: {wind_data_resized.min():.2f} - {wind_data_resized.max():.2f} m/s")

print("\n✅ Extraction complete!")
