#!/usr/bin/env python3
"""
Explore WIND Toolkit gridded data - FIXED for structured coordinates
"""

import h5pyd
import numpy as np

print("="*80)
print("WIND Toolkit Data Explorer")
print("="*80)

f = h5pyd.File("/nrel/wtk-us.h5", "r")

# Data dimensions
ws_shape = f['windspeed_100m'].shape
print(f"\n📐 Grid Dimensions:")
print(f"  Time steps: {ws_shape[0]:,}")
print(f"  Latitude points: {ws_shape[1]:,}")
print(f"  Longitude points: {ws_shape[2]:,}")
print(f"  Total data points: {np.prod(ws_shape):,}")

# Coordinates - handle structured array
coords = f['coordinates'][...]
if coords.dtype.names:
    lats = coords['lat']
    lons = coords['lon']
else:
    lats = coords[0]
    lons = coords[1]

print(f"\n🗺️  Geographic Coverage:")
print(f"  Latitude:  {lats.min():.3f}° to {lats.max():.3f}°")
print(f"  Longitude: {lons.min():.3f}° to {lons.max():.3f}°")

# Grid resolution
lat_res = np.abs(np.diff(lats[:, 0])).mean()
lon_res = np.abs(np.diff(lons[0, :])).mean()
print(f"  Resolution: {lat_res:.4f}° lat × {lon_res:.4f}° lon")
print(f"  (~{lat_res*111:.1f} km × {lon_res*111*np.cos(np.radians(lats.mean())):.1f} km)")

# Time coverage
datetime_arr = f['datetime'][...]
print(f"\n⏰ Temporal Coverage:")
print(f"  Start: {datetime_arr[0]}")
print(f"  End: {datetime_arr[-1]}")
print(f"  Total hours: {len(datetime_arr):,}")
print(f"  Total years: ~{len(datetime_arr) / 8760:.1f}")

# Mid-Atlantic region
print(f"\n🎯 Mid-Atlantic Region (38-40°N, -76 to -74°W):")

# Find indices
lat_mask = (lats[:, 0] >= 38) & (lats[:, 0] <= 40)
lon_mask = (lons[0, :] >= -76) & (lons[0, :] <= -74)

lat_idx = np.where(lat_mask)[0]
lon_idx = np.where(lon_mask)[0]

if len(lat_idx) > 0 and len(lon_idx) > 0:
    print(f"  Lat indices: {lat_idx[0]} to {lat_idx[-1]} ({len(lat_idx)} points)")
    print(f"  Lon indices: {lon_idx[0]} to {lon_idx[-1]} ({len(lon_idx)} points)")
    print(f"  Subgrid size: {len(lat_idx)} × {len(lon_idx)} = {len(lat_idx)*len(lon_idx):,} points")
else:
    print(f"  ⚠️  Region not in dataset bounds")
    print(f"  Dataset covers: {lats.min():.1f}° to {lats.max():.1f}°N")
    print(f"                  {lons.min():.1f}° to {lons.max():.1f}°W")

# Sample statistics
print(f"\n📊 Wind Speed Sample (100m, first timestep):")
ws_sample = f['windspeed_100m'][0, :, :]
print(f"  Mean: {ws_sample.mean():.2f} m/s")
print(f"  Std:  {ws_sample.std():.2f} m/s")
print(f"  Min:  {ws_sample.min():.2f} m/s")
print(f"  Max:  {ws_sample.max():.2f} m/s")

# Dataset size
total_size_gb = ws_shape[0] * ws_shape[1] * ws_shape[2] * 4 / 1e9  # 4 bytes per float32
print(f"\n💾 Storage:")
print(f"  Single variable: ~{total_size_gb:.1f} GB")
print(f"  All wind variables: ~{total_size_gb * 7:.1f} GB")

f.close()
print("\n✅ Exploration complete!")
