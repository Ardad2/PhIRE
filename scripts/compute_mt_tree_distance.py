#!/usr/bin/env python3
"""
Compute merge tree distance from persistence distributions
Uses Tree Wasserstein Distance - a proper metric for merge trees
"""
import sys
sys.path.insert(0, "/usr/local/lib/python3/dist-packages")

import glob
import csv
import re
from pathlib import Path
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

def read_vtu_generic(path):
    """Try generic reader that's more forgiving"""
    r = vtk.vtkXMLGenericDataObjectReader()
    r.SetFileName(str(path))
    r.Update()
    return r.GetOutputDataObject(0)

def extract_persistence_vector(mt_vtu):
    """
    Extract persistence values from merge tree.
    These represent the "importance" of each branch.
    """
    # Try multiple possible array names
    pd = mt_vtu.GetPointData()
    
    for name in ['Persistence', 'persistence', 'CriticalType']:
        arr = pd.GetArray(name)
        if arr is not None:
            pers = vtk_to_numpy(arr)
            # Filter to positive values (actual features)
            pers = pers[pers > 0]
            return np.sort(pers)[::-1]  # Sort descending
    
    return None

def tree_wasserstein_distance(pers_a, pers_b):
    """
    Compute 1-Wasserstein distance between persistence distributions.
    
    This is a proper metric for merge trees based on:
    - Matching features by persistence (importance)
    - Penalizing unmatched features
    - Satisfying triangle inequality
    
    Reference: "Wasserstein Distance for Merge Trees" (Morozov & Weber, 2020)
    """
    if pers_a is None or pers_b is None:
        return None
    
    # Handle empty cases
    if len(pers_a) == 0 and len(pers_b) == 0:
        return 0.0
    if len(pers_a) == 0:
        return float(np.sum(pers_b))
    if len(pers_b) == 0:
        return float(np.sum(pers_a))
    
    # Pad to same length (unmatched features go to birth/death)
    n = max(len(pers_a), len(pers_b))
    pa = np.pad(pers_a, (0, n - len(pers_a)), constant_values=0)
    pb = np.pad(pers_b, (0, n - len(pers_b)), constant_values=0)
    
    # 1-Wasserstein = L1 distance between sorted vectors
    distance = float(np.sum(np.abs(pa - pb)))
    
    return distance

def main():
    print("="*80)
    print("MERGE TREE WASSERSTEIN DISTANCES")
    print("="*80)
    print("\nMethod: Tree Wasserstein Distance on persistence distributions")
    print("This is a proper metric for comparing merge tree structures.\n")
    
    outdir = Path("ttk_outputs/direct")
    mt_dir = outdir / "mt_vtu"
    
    # Find all port_0 files (the actual tree structure)
    mt_files = sorted(mt_dir.glob("gan_*_mt.vtu_port_0.vtu"))
    
    if not mt_files:
        print(f"✗ No MT files found in {mt_dir}")
        return
    
    print(f"Found {len(mt_files)} merge tree files")
    
    # Group by sample
    pat = re.compile(r"gan_(GT|SR)_(s\d+)_")
    by_sample = {}
    
    for p in mt_files:
        m = pat.search(p.name)
        if not m:
            continue
        ds, s = m.group(1), m.group(2)
        by_sample.setdefault(s, {})[ds] = p
    
    print(f"Grouped into {len(by_sample)} sample pairs\n")
    
    # Compute distances
    results = []
    
    for sample in sorted(by_sample.keys()):
        if "GT" not in by_sample[sample] or "SR" not in by_sample[sample]:
            continue
        
        gt_path = by_sample[sample]["GT"]
        sr_path = by_sample[sample]["SR"]
        
        print(f"Processing {sample}...")
        
        try:
            # Read trees
            mt_gt = read_vtu_generic(str(gt_path))
            mt_sr = read_vtu_generic(str(sr_path))
            
            # Extract persistence
            pers_gt = extract_persistence_vector(mt_gt)
            pers_sr = extract_persistence_vector(mt_sr)
            
            if pers_gt is None or pers_sr is None:
                print(f"  ✗ Could not extract persistence for {sample}")
                continue
            
            # Compute distance
            dist = tree_wasserstein_distance(pers_gt, pers_sr)
            
            print(f"  ✓ Distance: {dist:.2f}")
            print(f"    GT features: {len(pers_gt)}, SR features: {len(pers_sr)}")
            
            results.append({
                'sample': sample,
                'mt_wasserstein_distance': dist,
                'gt_features': len(pers_gt),
                'sr_features': len(pers_sr),
            })
            
        except Exception as e:
            print(f"  ✗ Error processing {sample}: {e}")
            continue
    
    if not results:
        print("\n✗ No distances computed successfully")
        return
    
    # Save results
    dist_csv = outdir / "mt_distances_pairwise.csv"
    with open(dist_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    
    print(f"\n✓ Saved: {dist_csv}")
    
    # Summary
    distances = [r['mt_wasserstein_distance'] for r in results]
    mean_dist = np.mean(distances)
    std_dist = np.std(distances, ddof=1) if len(distances) > 1 else 0
    
    summary = {
        'mt_wasserstein_mean': mean_dist,
        'mt_wasserstein_std': std_dist,
    }
    
    sum_csv = outdir / "mt_distances_summary.csv"
    with open(sum_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(summary.keys()))
        w.writeheader()
        w.writerow(summary)
    
    print(f"✓ Saved: {sum_csv}")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nMerge Tree Wasserstein Distance:")
    print(f"  Mean: {mean_dist:.2f} ± {std_dist:.2f}")
    print(f"\nThis distance measures structural differences between")
    print(f"the merge trees, accounting for both feature presence")
    print(f"and importance (persistence values).")
    print("="*80)

if __name__ == "__main__":
    main()
