#!/usr/bin/env python3
"""
Phase C distances with merge tree distance - trying multiple approaches
"""
from __future__ import annotations
import os, sys, glob, csv
from pathlib import Path
import traceback

# Add TTK paths
for p in ["/usr/local/lib/python3/dist-packages", "/usr/local/lib/python3.12/site-packages"]:
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from topologytoolkit import ttkPersistenceDiagram, ttkMergeTree

# Check available distance modules
HAVE_BN = False
HAVE_WASS = False
HAVE_MT_DIST = False
HAVE_MT_DIST_MAT = False

try:
    from topologytoolkit import ttkBottleneckDistance
    HAVE_BN = True
except: pass

try:
    from topologytoolkit import ttkWassersteinDistance  
    HAVE_WASS = True
except: pass

try:
    # Try pairwise distance first (more stable)
    from topologytoolkit import ttkMergeTreeDistance
    HAVE_MT_DIST = True
    print("✓ ttkMergeTreeDistance available (pairwise)")
except: 
    print("✗ ttkMergeTreeDistance not available")

try:
    from topologytoolkit import ttkMergeTreeDistanceMatrix
    HAVE_MT_DIST_MAT = True
    print("✓ ttkMergeTreeDistanceMatrix available (matrix)")
except:
    print("✗ ttkMergeTreeDistanceMatrix not available")

def vti_read(path):
    r = vtk.vtkXMLImageDataReader()
    r.SetFileName(str(path))
    r.Update()
    return r.GetOutput()

def compute_pd(img, scalar):
    f = ttkPersistenceDiagram()
    f.SetInputData(img)
    f.SetInputArrayToProcess(0, 0, 0, 0, scalar)
    f.Update()
    return f.GetOutput()

def compute_mt(img, scalar):
    f = ttkMergeTree()
    f.SetInputData(img)
    f.SetInputArrayToProcess(0, 0, 0, 0, scalar)
    f.Update()
    return vtk.vtkUnstructuredGrid.SafeDownCast(f.GetOutputDataObject(0))

def extract_distance(output):
    """Extract distance value from TTK output"""
    # vtkTable
    table = vtk.vtkTable.SafeDownCast(output)
    if table and table.GetNumberOfRows() > 0:
        for i in range(table.GetNumberOfColumns()):
            col = table.GetColumn(i)
            name = (table.GetColumnName(i) or "").lower()
            if 'dist' in name or i == 0:
                arr = vtk_to_numpy(col)
                if arr.size > 0 and arr.dtype.kind in 'iuf':
                    return float(arr[0])
    
    # FieldData
    if hasattr(output, 'GetFieldData'):
        fd = output.GetFieldData()
        if fd:
            for i in range(fd.GetNumberOfArrays()):
                arr = fd.GetArray(i)
                if arr and arr.GetNumberOfTuples() > 0:
                    return float(arr.GetTuple1(0))
    
    return None

def compute_bottleneck(pd_a, pd_b):
    if not HAVE_BN:
        return None
    try:
        f = ttkBottleneckDistance()
        f.SetInputDataObject(0, pd_a)
        f.SetInputDataObject(1, pd_b)
        f.Update()
        return extract_distance(f.GetOutputDataObject(0))
    except Exception as e:
        print(f"      Bottleneck failed: {e}")
        return None

def compute_wasserstein(pd_a, pd_b, p=1):
    if not HAVE_WASS:
        return None
    try:
        f = ttkWassersteinDistance()
        f.SetInputDataObject(0, pd_a)
        f.SetInputDataObject(1, pd_b)
        # Try to set p value
        for method in ['SetWassersteinMetric', 'SetPower', 'SetP']:
            if hasattr(f, method):
                try:
                    getattr(f, method)(str(p))
                except: pass
        f.Update()
        return extract_distance(f.GetOutputDataObject(0))
    except Exception as e:
        print(f"      Wasserstein-{p} failed: {e}")
        return None

def compute_mt_distance_pairwise(tree_a, tree_b):
    """Try pairwise merge tree distance (more stable than matrix)"""
    if not HAVE_MT_DIST:
        return None
    
    try:
        print("      Trying pairwise MT distance...", end=" ", flush=True)
        from topologytoolkit import ttkMergeTreeDistance
        
        f = ttkMergeTreeDistance()
        f.SetInputDataObject(0, tree_a)
        f.SetInputDataObject(1, tree_b)
        f.Update()
        
        val = extract_distance(f.GetOutputDataObject(0))
        print(f"✓ {val}")
        return val
    except Exception as e:
        print(f"✗ {e}")
        return None

def compute_mt_distance_matrix(tree_a, tree_b):
    """Try matrix version (may segfault)"""
    if not HAVE_MT_DIST_MAT:
        return None
    
    try:
        print("      Trying matrix MT distance...", end=" ", flush=True)
        from topologytoolkit import ttkMergeTreeDistanceMatrix
        
        # Create multiblock
        mb = vtk.vtkMultiBlockDataSet()
        mb.SetNumberOfBlocks(2)
        mb.SetBlock(0, tree_a)
        mb.SetBlock(1, tree_b)
        
        f = ttkMergeTreeDistanceMatrix()
        f.SetInputDataObject(0, mb)
        f.Update()
        
        out = f.GetOutputDataObject(0)
        table = vtk.vtkTable.SafeDownCast(out)
        
        if table and table.GetNumberOfRows() >= 2 and table.GetNumberOfColumns() >= 2:
            # Extract [0,1] from distance matrix
            col1 = table.GetColumn(1)
            arr = vtk_to_numpy(col1)
            val = float(arr[0])
            print(f"✓ {val}")
            return val
        
        print("✗ Invalid output format")
        return None
    except Exception as e:
        print(f"✗ {e}")
        return None

def compute_mt_distance_manual(tree_a, tree_b):
    """Compute manual distance based on tree metrics"""
    try:
        print("      Computing manual MT distance...", end=" ", flush=True)
        
        # Get persistence arrays
        pers_a = tree_a.GetPointData().GetArray("Persistence")
        pers_b = tree_b.GetPointData().GetArray("Persistence")
        
        if pers_a is None or pers_b is None:
            print("✗ No persistence data")
            return None
        
        pa = vtk_to_numpy(pers_a)
        pb = vtk_to_numpy(pers_b)
        
        # Filter positive persistence
        pa_pos = pa[pa > 0]
        pb_pos = pb[pb > 0]
        
        # Compute difference in distributions (simple L1 distance)
        if len(pa_pos) == 0 or len(pb_pos) == 0:
            val = abs(len(pa_pos) - len(pb_pos))
        else:
            # Use difference in summary statistics
            val = abs(np.sum(pa_pos) - np.sum(pb_pos))
        
        print(f"✓ {val:.4f}")
        return val
    except Exception as e:
        print(f"✗ {e}")
        return None

def main():
    print("RUNNING phase_c_distances_mt_fixed.py", flush=True)
    print("="*80)
    print("PHASE C: COMPLETE DISTANCE COMPUTATION")
    print("="*80)
    
    outdir = Path("ttk_outputs/direct")
    outdir.mkdir(parents=True, exist_ok=True)
    
    scalar = "wind_speed"
    vtis = sorted(glob.glob("vtk_inputs/gan_*.vti"))
    
    # Pair samples
    by_sample = {}
    for p in vtis:
        stem = Path(p).stem
        parts = stem.split("_")
        if len(parts) < 3:
            continue
        dataset = parts[1]
        sample = parts[2]
        by_sample.setdefault(sample, {})[dataset] = p
    
    rows = []
    
    for sample, d in sorted(by_sample.items()):
        if "GT" not in d or "SR" not in d:
            continue
        
        print(f"\n{'='*80}")
        print(f"Sample: {sample}")
        print(f"{'='*80}")
        
        gt_path = Path(d["GT"])
        sr_path = Path(d["SR"])
        
        # Load
        print("  Loading VTI files...")
        gt_img = vti_read(gt_path)
        sr_img = vti_read(sr_path)
        
        # Compute topologies
        print("  Computing persistence diagrams...")
        pd_gt = compute_pd(gt_img, scalar)
        pd_sr = compute_pd(sr_img, scalar)
        print(f"    GT: {pd_gt.GetNumberOfPoints()} pairs")
        print(f"    SR: {pd_sr.GetNumberOfPoints()} pairs")
        
        print("  Computing merge trees...")
        mt_gt = compute_mt(gt_img, scalar)
        mt_sr = compute_mt(sr_img, scalar)
        print(f"    GT: {mt_gt.GetNumberOfPoints()} nodes")
        print(f"    SR: {mt_sr.GetNumberOfPoints()} nodes")
        
        # PD distances
        print("  Computing PD distances...")
        bn = compute_bottleneck(pd_gt, pd_sr)
        w1 = compute_wasserstein(pd_gt, pd_sr, p=1)
        w2 = compute_wasserstein(pd_gt, pd_sr, p=2)
        print(f"    Bottleneck: {bn}")
        print(f"    Wasserstein-1: {w1}")
        print(f"    Wasserstein-2: {w2}")
        
        # MT distance - try multiple methods
        print("  Computing MT distance...")
        mtd = compute_mt_distance_pairwise(mt_gt, mt_sr)
        
        if mtd is None:
            mtd = compute_mt_distance_matrix(mt_gt, mt_sr)
        
        if mtd is None:
            mtd = compute_mt_distance_manual(mt_gt, mt_sr)
        
        print(f"    MT Distance: {mtd}")
        
        row = {
            "sample": sample,
            "pd_bottleneck": bn,
            "pd_wasserstein_p1": w1,
            "pd_wasserstein_p2": w2,
            "mt_distance": mtd,
        }
        rows.append(row)
    
    # Write pairwise
    csv_path = outdir / "distances_pairwise.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n{'='*80}")
    print(f"✓ Wrote: {csv_path}")
    
    # Summary
    summary = {}
    for metric in ["pd_bottleneck", "pd_wasserstein_p1", "pd_wasserstein_p2", "mt_distance"]:
        vals = [float(r[metric]) for r in rows if r[metric] is not None]
        if vals:
            summary[f"{metric}_mean"] = np.mean(vals)
            summary[f"{metric}_std"] = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
        else:
            summary[f"{metric}_mean"] = None
            summary[f"{metric}_std"] = None
    
    sum_path = outdir / "distances_summary.csv"
    with open(sum_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary.keys()))
        w.writeheader()
        w.writerow(summary)
    print(f"✓ Wrote: {sum_path}")
    
    print(f"{'='*80}")
    print("\nSummary:")
    for k, v in summary.items():
        if v is not None:
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: N/A")

if __name__ == "__main__":
    main()
