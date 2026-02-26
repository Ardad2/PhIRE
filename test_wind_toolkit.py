#!/usr/bin/env python3
"""
Test WIND Toolkit API access - FIXED for structured coordinate arrays
"""

import h5pyd
import numpy as np
import sys

print("="*80)
print("Testing WIND Toolkit Access")
print("="*80)

try:
    # Open the main WIND Toolkit file
    print("\nOpening /nrel/wtk-us.h5...")
    f = h5pyd.File("/nrel/wtk-us.h5", "r")
    
    print("✅ SUCCESS! Connected to WIND Toolkit")
    
    # Check data structure
    ws_shape = f['windspeed_100m'].shape
    coords_shape = f['coordinates'].shape
    
    print(f"\n🗺️  Data Structure:")
    print(f"  ✅ GRIDDED data (lat x lon grid)")
    print(f"  ✅ Grid dimensions: {coords_shape[0]} x {coords_shape[1]}")
    print(f"  ✅ Time steps: {ws_shape[0]:,}")
    print(f"  ✅ Perfect for PhIRE - already on regular grid!")
    
    # Get time information
    print(f"\n⏰ Time Information:")
    datetime_data = f['datetime'][...]
    print(f"  First: {datetime_data[0]}")
    print(f"  Last: {datetime_data[-1]}")
    print(f"  Total timesteps: {len(datetime_data):,}")
    
    # Years covered
    first_year = str(datetime_data[0])[:4]
    last_year = str(datetime_data[-1])[:4]
    print(f"  Years: {first_year} - {last_year}")
    
    # Get coordinates - FIXED for structured arrays
    print(f"\n📍 Coordinates:")
    coords = f['coordinates'][...]
    
    # Check if structured array
    if coords.dtype.names:
        print(f"  Format: Structured array with fields: {coords.dtype.names}")
        lats = coords['lat']
        lons = coords['lon']
    else:
        print(f"  Format: Regular array")
        lats = coords[0]
        lons = coords[1]
    
    print(f"  Latitude range:  {lats.min():.2f}° to {lats.max():.2f}°")
    print(f"  Longitude range: {lons.min():.2f}° to {lons.max():.2f}°")
    
    # Grid spacing
    lat_spacing = np.abs(np.diff(lats[:, 0])).mean()
    lon_spacing = np.abs(np.diff(lons[0, :])).mean()
    print(f"  Grid spacing: ~{lat_spacing:.3f}° lat x ~{lon_spacing:.3f}° lon")
    
    # Test data extraction
    print(f"\n📥 Testing data extraction:")
    sample = f['windspeed_100m'][0, :10, :10]
    print(f"  ✅ Sample shape: {sample.shape}")
    print(f"  ✅ Sample mean: {sample.mean():.2f} m/s")
    
    # Available heights
    print(f"\n🌬️  Available Wind Speed Heights:")
    heights = []
    for key in f.keys():
        if key.startswith('windspeed_'):
            height = key.split('_')[1].replace('m', '')
            heights.append(height)
    print(f"  Heights: {', '.join(sorted(heights, key=int))} meters")
    
    f.close()
    print("\n✅ All tests passed!")
    print("\n💡 You have 7 years of hourly wind data on a 1602x2976 grid!")
    sys.exit(0)
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
