#!/usr/bin/env python3
"""
Simplified REAL Distance Calculator

Uses only GUDHI (no TTK Python filters) to avoid segfaults.
Works with your existing VTU files.

Usage:
    python3 compute_distances_simple.py
"""

import sys
from pathlib import Path

# Check imports
missing = []
try:
    import numpy as np
except ImportError:
    missing.append("numpy")

try:
    import pandas as pd
except ImportError:
    missing.append("pandas")

try:
    import vtk
except ImportError:
    missing.append("vtk")

if missing:
    print("❌ Missing packages:", ", ".join(missing))
    print("\nInstall with:")
    print(f"  python3 -m pip install --break-system-packages {' '.join(missing)}")
    sys.exit(1)

# Install GUDHI if needed
try:
    import gudhi
except ImportError:
    print("📦 Installing GUDHI...")
    import subprocess
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", 
        "--break-system-packages", "gudhi"
    ])
    import gudhi

print("="*80)
print("SIMPLIFIED REAL DISTANCE CALCULATOR")
print("="*80)
print(f"VTK: {vtk.vtkVersion.GetVTKVersion()}")
print(f"GUDHI: {gudhi.__version__}")
print(f"NumPy: {np.__version__}")
print()

# ============================================================================
# READ PERSISTENCE DIAGRAMS FROM VTU
# ============================================================================

def read_pd_from_vtu(vtu_path):
    """Extract persistence diagram from TTK VTU output."""
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(vtu_path))
    reader.Update()
    
    output = reader.GetOutput()
    if output.GetNumberOfPoints() == 0:
        return None
    
    point_data = output.GetPointData()
    
    # Find Birth and Death arrays
    birth_arr = point_data.GetArray('Birth')
    death_arr = point_data.GetArray('Death')
    
    if not birth_arr or not death_arr:
        return None
    
    # Convert to numpy array for GUDHI
    diagram = []
    for i in range(birth_arr.GetNumberOfTuples()):
        b = birth_arr.GetValue(i)
        d = death_arr.GetValue(i)
        if d > b:  # Valid pair
            diagram.append([b, d])
    
    return np.array(diagram) if diagram else None


# ============================================================================
# READ MERGE TREE FEATURES FROM VTU
# ============================================================================

def read_mt_features(vtu_path):
    """Extract merge tree structural features."""
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(vtu_path))
    reader.Update()
    
    output = reader.GetOutput()
    if output.GetNumberOfPoints() == 0:
        return None
    
    return {
        'nodes': output.GetNumberOfPoints(),
        'arcs': output.GetNumberOfCells(),
    }


# ============================================================================
# COMPUTE DISTANCES
# ============================================================================

def compute_pd_distance(pd1, pd2):
    """Compute Wasserstein-2 distance using GUDHI."""
    if pd1 is None or pd2 is None or len(pd1) == 0 or len(pd2) == 0:
        return None
    
    try:
        return gudhi.wasserstein.wasserstein_distance(pd1, pd2, order=2)
    except Exception as e:
        print(f"  ⚠️  PD distance error: {e}")
        return None


def compute_mt_distance(mt1, mt2):
    """Compute structural distance between merge trees."""
    if mt1 is None or mt2 is None:
        return None
    
    # Simple structural dissimilarity
    node_diff = abs(mt1['nodes'] - mt2['nodes']) / max(mt1['nodes'], mt2['nodes'], 1)
    arc_diff = abs(mt1['arcs'] - mt2['arcs']) / max(mt1['arcs'], mt2['arcs'], 1)
    
    return (node_diff + arc_diff) / 2


# ============================================================================
# PROCESS ALL FILES
# ============================================================================

def find_matching_file(sample_id, file_list):
    """Find file matching sample ID."""
    for f in file_list:
        if sample_id in f.stem:
            return f
    return None


def main():
    # Setup paths
    base_dir = Path("/home/adadhwal/PhIRE/ttk_outputs")
    pd_dir = base_dir / "direct" / "pd_vtu"
    mt_dir = base_dir / "mt"
    
    # Check directories exist
    if not pd_dir.exists():
        print(f"❌ PD directory not found: {pd_dir}")
        return
    
    if not mt_dir.exists():
        print(f"❌ MT directory not found: {mt_dir}")
        return
    
    # Find GT and SR subdirectories
    gt_pd_dir = pd_dir / "GT"
    sr_pd_dir = pd_dir / "SR"
    gt_mt_dir = mt_dir / "GT"
    sr_mt_dir = mt_dir / "SR"
    
    if not gt_pd_dir.exists():
        print(f"❌ GT PD directory not found: {gt_pd_dir}")
        return
    
    if not sr_pd_dir.exists():
        print(f"❌ SR PD directory not found: {sr_pd_dir}")
        return
    
    # Get file lists
    gt_pd_files = sorted(gt_pd_dir.glob("*.vtu"))
    sr_pd_files = sorted(sr_pd_dir.glob("*.vtu"))
    gt_mt_files = sorted(gt_mt_dir.glob("*.vtu")) if gt_mt_dir.exists() else []
    sr_mt_files = sorted(sr_mt_dir.glob("*.vtu")) if sr_mt_dir.exists() else []
    
    print(f"Found:")
    print(f"  GT PD files: {len(gt_pd_files)}")
    print(f"  SR PD files: {len(sr_pd_files)}")
    print(f"  GT MT files: {len(gt_mt_files)}")
    print(f"  SR MT files: {len(sr_mt_files)}")
    print()
    
    # Process each GT sample
    results = []
    
    for gt_pd_file in gt_pd_files:
        # Extract sample ID (e.g., "s0" from filename)
        sample_id = None
        for part in gt_pd_file.stem.split('_'):
            if part.startswith('s') and part[1:].isdigit():
                sample_id = part
                break
        
        if not sample_id:
            print(f"⚠️  Skipping {gt_pd_file.name} (no sample ID)")
            continue
        
        print(f"Processing sample: {sample_id}")
        
        # Find matching SR PD file
        sr_pd_file = find_matching_file(sample_id, sr_pd_files)
        if not sr_pd_file:
            print(f"  ⚠️  No matching SR PD file")
            continue
        
        # Read PD diagrams
        pd_gt = read_pd_from_vtu(gt_pd_file)
        pd_sr = read_pd_from_vtu(sr_pd_file)
        
        pd_dist = None
        if pd_gt is not None and pd_sr is not None:
            pd_dist = compute_pd_distance(pd_gt, pd_sr)
            if pd_dist is not None:
                print(f"  ✅ PD distance: {pd_dist:.6f}")
        
        # Find matching MT files
        mt_dist = None
        if gt_mt_files and sr_mt_files:
            gt_mt_file = find_matching_file(sample_id, gt_mt_files)
            sr_mt_file = find_matching_file(sample_id, sr_mt_files)
            
            if gt_mt_file and sr_mt_file:
                mt_gt = read_mt_features(gt_mt_file)
                mt_sr = read_mt_features(sr_mt_file)
                
                if mt_gt and mt_sr:
                    mt_dist = compute_mt_distance(mt_gt, mt_sr)
                    if mt_dist is not None:
                        print(f"  ✅ MT distance: {mt_dist:.6f}")
        
        # Store result
        results.append({
            'sample': sample_id,
            'method': 'SR',
            'dataset': 'SR',
            'pd_distance': pd_dist if pd_dist is not None else np.nan,
            'mt_distance': mt_dist if mt_dist is not None else np.nan,
        })
        print()
    
    # Create DataFrame and save
    if not results:
        print("❌ No results computed!")
        return
    
    df = pd.DataFrame(results)
    
    # Save results
    output_dir = base_dir / "direct"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "distances_simple.csv"
    df.to_csv(output_file, index=False)
    
    print("="*80)
    print("✅ RESULTS")
    print("="*80)
    print(f"\nSaved: {output_file}")
    print(f"\nProcessed {len(results)} samples:")
    print(df.to_string(index=False))
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    pd_valid = df['pd_distance'].dropna()
    mt_valid = df['mt_distance'].dropna()
    
    if len(pd_valid) > 0:
        print(f"\nPD Distances (Wasserstein-2):")
        print(f"  Mean: {pd_valid.mean():.6f}")
        print(f"  Std:  {pd_valid.std():.6f}")
        print(f"  Min:  {pd_valid.min():.6f}")
        print(f"  Max:  {pd_valid.max():.6f}")
    
    if len(mt_valid) > 0:
        print(f"\nMT Distances (Structural):")
        print(f"  Mean: {mt_valid.mean():.6f}")
        print(f"  Std:  {mt_valid.std():.6f}")
        print(f"  Min:  {mt_valid.min():.6f}")
        print(f"  Max:  {mt_valid.max():.6f}")
    
    print("\n" + "="*80)
    print("Next: Run visualization script")
    print("="*80)


if __name__ == "__main__":
    main()
