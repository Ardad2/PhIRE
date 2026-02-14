#!/usr/bin/env python3
"""
Analyze TTK outputs: extract topology metrics from PDs and Merge Trees
(No pandas dependency - pure Python + VTK + NumPy)
"""
import vtk
import numpy as np
from pathlib import Path
import json

def extract_pd_features(vtu_path):
    """Extract features from persistence diagram"""
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(vtu_path))
    reader.Update()
    pd = reader.GetOutput()
    
    n_points = pd.GetNumberOfPoints()
    
    features = {
        'n_critical_points': n_points,
        'n_cells': pd.GetNumberOfCells()
    }
    
    # Get persistence values
    if pd.GetPointData().GetArray("Persistence"):
        pers = pd.GetPointData().GetArray("Persistence")
        pers_vals = [pers.GetValue(i) for i in range(n_points)]
        pers_vals = [p for p in pers_vals if p > 0]  # Filter positive
        
        if pers_vals:
            features['max_persistence'] = max(pers_vals)
            features['mean_persistence'] = np.mean(pers_vals)
            features['total_persistence'] = sum(pers_vals)
            features['n_significant'] = len([p for p in pers_vals if p > 0.1])
        else:
            features['max_persistence'] = 0
            features['mean_persistence'] = 0
            features['total_persistence'] = 0
            features['n_significant'] = 0
    
    # Get pair types (PD0 vs PD1)
    if pd.GetPointData().GetArray("PairType"):
        pair_types = pd.GetPointData().GetArray("PairType")
        n_pd0 = sum(1 for i in range(n_points) if pair_types.GetValue(i) == 0)
        n_pd1 = sum(1 for i in range(n_points) if pair_types.GetValue(i) == 1)
        features['n_pd0'] = n_pd0
        features['n_pd1'] = n_pd1
    
    return features

def extract_mt_features(vtu_path):
    """Extract features from merge tree"""
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(vtu_path))
    reader.Update()
    tree = reader.GetOutput()
    
    features = {
        'n_nodes': tree.GetNumberOfPoints(),
        'n_arcs': tree.GetNumberOfCells()
    }
    
    # Get persistence from tree
    if tree.GetPointData().GetArray("Persistence"):
        pers = tree.GetPointData().GetArray("Persistence")
        pers_vals = [pers.GetValue(i) for i in range(features['n_nodes'])]
        pers_vals = [p for p in pers_vals if p > 0]
        
        if pers_vals:
            features['max_pers'] = max(pers_vals)
            features['mean_pers'] = np.mean(pers_vals)
            features['n_branches'] = len([p for p in pers_vals if p > 0.1])
        else:
            features['max_pers'] = 0
            features['mean_pers'] = 0
            features['n_branches'] = 0
    
    return features

def main():
    results = []
    
    # Process PDs
    print("="*70)
    print("ANALYZING PERSISTENCE DIAGRAMS")
    print("="*70)
    pd_dir = Path('ttk_outputs/pd')
    for vtu_file in sorted(pd_dir.glob('*_pd_port_0.vtu')):
        # Parse filename: gan_GT_s0_speed_p160_x0_y0_pd_port_0.vtu
        parts = vtu_file.stem.replace('_pd_port_0', '').split('_')
        
        method = parts[0]  # gan
        dataset = parts[1]  # GT or SR
        sample = parts[2]   # s0, s1, etc.
        
        features = extract_pd_features(vtu_file)
        features['method'] = method
        features['dataset'] = dataset
        features['sample'] = sample
        features['type'] = 'PD'
        
        results.append(features)
        print(f"  {dataset} {sample}: {features.get('n_pd1', 0)} PD1 features, "
              f"{features.get('n_pd0', 0)} PD0 features")
    
    # Process MTs
    print("\n" + "="*70)
    print("ANALYZING MERGE TREES")
    print("="*70)
    mt_dir = Path('ttk_outputs/mt')
    for vtu_file in sorted(mt_dir.glob('*_mt_port_0.vtu')):
        if 'toy' in str(vtu_file):  # Skip toy file
            continue
            
        parts = vtu_file.stem.replace('_mt_port_0', '').split('_')
        
        method = parts[0]
        dataset = parts[1]
        sample = parts[2]
        
        features = extract_mt_features(vtu_file)
        features['method'] = method
        features['dataset'] = dataset
        features['sample'] = sample
        features['type'] = 'MT'
        
        results.append(features)
        print(f"  {dataset} {sample}: {features['n_branches']} branches, "
              f"{features['n_nodes']} nodes")
    
    # Save results as JSON (no pandas needed)
    with open('ttk_outputs/topology_metrics.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved: ttk_outputs/topology_metrics.json")
    
    # Compute summary statistics
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    
    # PD1 features by dataset
    pd_results = [r for r in results if r['type'] == 'PD']
    print("\nPersistence Diagram PD1 Features (loops/holes):")
    
    for dataset in ['GT', 'SR']:
        subset = [r for r in pd_results if r['dataset'] == dataset]
        if subset and 'n_pd1' in subset[0]:
            pd1_vals = [r['n_pd1'] for r in subset]
            print(f"  {dataset}: {np.mean(pd1_vals):.1f} ± {np.std(pd1_vals):.1f}")
            print(f"       Range: [{min(pd1_vals)}, {max(pd1_vals)}]")
    
    # Merge tree branches by dataset
    mt_results = [r for r in results if r['type'] == 'MT']
    print("\nMerge Tree Branches:")
    
    for dataset in ['GT', 'SR']:
        subset = [r for r in mt_results if r['dataset'] == dataset]
        if subset and 'n_branches' in subset[0]:
            branch_vals = [r['n_branches'] for r in subset]
            print(f"  {dataset}: {np.mean(branch_vals):.1f} ± {np.std(branch_vals):.1f}")
            print(f"       Range: [{min(branch_vals)}, {max(branch_vals)}]")
    
    print("\n" + "="*70)
    print("KEY FINDINGS")
    print("="*70)
    
    # Compare GT vs SR
    gt_pd1 = [r['n_pd1'] for r in pd_results if r['dataset'] == 'GT' and 'n_pd1' in r]
    sr_pd1 = [r['n_pd1'] for r in pd_results if r['dataset'] == 'SR' and 'n_pd1' in r]
    
    if gt_pd1 and sr_pd1:
        diff_pct = (np.mean(sr_pd1) - np.mean(gt_pd1)) / np.mean(gt_pd1) * 100
        print(f"\nPD1 Features:")
        print(f"  GT: {np.mean(gt_pd1):.1f} loops/holes")
        print(f"  SR: {np.mean(sr_pd1):.1f} loops/holes")
        print(f"  Difference: {diff_pct:+.1f}%")
        
        if diff_pct > 50:
            print("  → GAN SR creates MORE topological features than ground truth")
            print("     (possibly hallucinating structure)")
        elif diff_pct < -20:
            print("  → GAN SR loses significant topology")
        else:
            print("  → GAN SR preserves topology reasonably well")
    
    gt_branches = [r['n_branches'] for r in mt_results if r['dataset'] == 'GT' and 'n_branches' in r]
    sr_branches = [r['n_branches'] for r in mt_results if r['dataset'] == 'SR' and 'n_branches' in r]
    
    if gt_branches and sr_branches:
        diff_pct = (np.mean(sr_branches) - np.mean(gt_branches)) / np.mean(gt_branches) * 100
        print(f"\nMerge Tree Branches:")
        print(f"  GT: {np.mean(gt_branches):.1f} branches")
        print(f"  SR: {np.mean(sr_branches):.1f} branches")
        print(f"  Difference: {diff_pct:+.1f}%")

if __name__ == '__main__':
    main()
