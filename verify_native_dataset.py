#!/usr/bin/env python3
"""
Verify the research-grade dataset
"""

import numpy as np
from pathlib import Path

dataset_dir = Path("/home/adadhwal/PhIRE/wind_toolkit_research")

print("="*80)
print("Dataset Verification")
print("="*80)

for split in ['train', 'val', 'test']:
    split_dir = dataset_dir / split
    if not split_dir.exists():
        print(f"\n❌ {split}: NOT FOUND")
        continue
    
    files = sorted(split_dir.glob("*.npz"))
    print(f"\n✅ {split}: {len(files)} samples")
    
    if files:
        # Check first file
        data = np.load(files[0], allow_pickle=True)
        hr = data['HR']
        mr = data['MR']
        lr = data['LR']
        meta = data['metadata'].item()
        
        print(f"  HR shape: {hr.shape} (expect (160,160,1))")
        print(f"  MR shape: {mr.shape} (expect (80,80,1))")
        print(f"  LR shape: {lr.shape} (expect (40,40,1))")
        print(f"  Native: {meta.get('native', False)}")
        print(f"  Downsampling: {meta.get('downsampling', 'unknown')}")
        print(f"  Wind range: {meta['min_windspeed_hr']:.2f} - {meta['max_windspeed_hr']:.2f} m/s")

print("\n" + "="*80)
