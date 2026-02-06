#!/usr/bin/env python3
"""
Extract metrics from TTK CLI-generated VTU files
"""
import sys
sys.path.insert(0, "/usr/local/lib/python3/dist-packages")

import glob
import csv
from pathlib import Path
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

def read_vtu(path):
    """Read VTU file"""
    r = vtk.vtkXMLUnstructuredGridReader()
    r.SetFileName(str(path))
    r.Update()
    return r.GetOutput()

def extract_pd_metrics(vtu):
    """Extract persistence diagram metrics"""
    # Get PairType array to count by dimension
    pair_type = vtu.GetCellData().GetArray("PairType")
    if pair_type is None:
        return {'n_pd0': 0, 'n_pd1': 0, 'n_pd2': 0}
    
    types = vtk_to_numpy(pair_type)
    return {
        'n_pd0': int(np.sum(types == 0)),  # H0 (components)
        'n_pd1': int(np.sum(types == 1)),  # H1 (loops)
        'n_pd2': int(np.sum(types == 2)),  # H2 (voids)
    }

def extract_mt_metrics(vtu):
    """Extract merge tree metrics from port_0"""
    n_nodes = vtu.GetNumberOfPoints()
    n_cells = vtu.GetNumberOfCells()
    
    # Count branches (cells with specific types)
    cell_dim = vtu.GetCellData().GetArray("CellDimension")
    if cell_dim:
        dims = vtk_to_numpy(cell_dim)
        n_branches = int(np.sum(dims == 1))  # 1D cells = branches
    else:
        n_branches = n_cells
    
    return {
        'mt_nodes': n_nodes,
        'mt_arcs': n_cells,
        'mt_branches': n_branches,
    }

def main():
    print("="*80)
    print("EXTRACTING METRICS FROM VTU FILES")
    print("="*80)
    
    outdir = Path('ttk_outputs/direct')
    
    # Process PD files
    pd_files = sorted(glob.glob('ttk_outputs/direct/pd_vtu/*_port_0.vtu'))
    mt_files = sorted(glob.glob('ttk_outputs/direct/mt_vtu/*_port_0.vtu'))
    
    print(f"\nFound {len(pd_files)} PD files, {len(mt_files)} MT files")
    
    rows = []
    
    for pd_path in pd_files:
        # Parse filename: gan_GT_s0_speed_p160_x0_y0_pd.vtu_port_0.vtu
        stem = Path(pd_path).stem.replace('_port_0', '')
        parts = stem.split('_')
        dataset = parts[1]  # GT or SR
        sample = parts[2]   # s0, s1, etc.
        
        # Find corresponding MT file
        mt_pattern = f"*_{dataset}_{sample}_*_port_0.vtu"
        mt_matches = glob.glob(f'ttk_outputs/direct/mt_vtu/{mt_pattern}')
        
        if not mt_matches:
            print(f"  ✗ No MT file for {dataset} {sample}")
            continue
        
        mt_path = mt_matches[0]
        
        print(f"  Processing {dataset} {sample}...")
        
        # Read and extract
        pd_vtu = read_vtu(pd_path)
        mt_vtu = read_vtu(mt_path)
        
        pd_metrics = extract_pd_metrics(pd_vtu)
        mt_metrics = extract_mt_metrics(mt_vtu)
        
        row = {
            'sample': sample,
            'dataset': dataset,
            **pd_metrics,
            **mt_metrics,
        }
        rows.append(row)
        
        print(f"    PD: {pd_metrics['n_pd1']} loops, MT: {mt_metrics['mt_branches']} branches")
    
    # Save results
    csv_path = outdir / 'phase_c_results_from_cli.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    
    print(f"\n✓ Saved: {csv_path}")
    
    # Compute summary
    print("\n" + "="*80)
    print("SUMMARY BY DATASET")
    print("="*80)
    
    for dataset in ['GT', 'SR']:
        subset = [r for r in rows if r['dataset'] == dataset]
        if not subset:
            continue
        
        pd1_mean = np.mean([r['n_pd1'] for r in subset])
        mt_br_mean = np.mean([r['mt_branches'] for r in subset])
        
        print(f"\n{dataset}:")
        print(f"  PD1 loops: {pd1_mean:.1f}")
        print(f"  MT branches: {mt_br_mean:.1f}")
    
    # Compute differences
    print("\n" + "="*80)
    print("GT vs SR DIFFERENCES")
    print("="*80)
    
    gt_rows = [r for r in rows if r['dataset'] == 'GT']
    sr_rows = [r for r in rows if r['dataset'] == 'SR']
    
    gt_pd1 = np.mean([r['n_pd1'] for r in gt_rows])
    sr_pd1 = np.mean([r['n_pd1'] for r in sr_rows])
    pd1_diff_pct = (sr_pd1 - gt_pd1) / gt_pd1 * 100
    
    gt_mt = np.mean([r['mt_branches'] for r in gt_rows])
    sr_mt = np.mean([r['mt_branches'] for r in sr_rows])
    mt_diff_pct = (sr_mt - gt_mt) / gt_mt * 100
    
    print(f"\nPD1 loops: {pd1_diff_pct:+.1f}%")
    print(f"MT branches: {mt_diff_pct:+.1f}%")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
