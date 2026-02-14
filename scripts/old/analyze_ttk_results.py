#!/usr/bin/env python3
"""
Analyze TTK outputs: extract topology metrics from PDs and Merge Trees
"""
import vtk
import numpy as np
from pathlib import Path
import pandas as pd

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
    print("Analyzing Persistence Diagrams...")
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
        print(f"  {vtu_file.name}: {features['n_pd1']} PD1 features")
    
    # Process MTs
    print("\nAnalyzing Merge Trees...")
    mt_dir = Path('ttk_outputs/mt')
    for vtu_file in sorted(mt_dir.glob('*_mt_port_0.vtu')):
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
        print(f"  {vtu_file.name}: {features['n_branches']} branches")
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv('ttk_outputs/topology_metrics.csv', index=False)
    print(f"\n✓ Saved: ttk_outputs/topology_metrics.csv")
    
    # Summary statistics
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    
    # PD1 features by dataset
    pd_df = df[df['type'] == 'PD']
    if 'n_pd1' in pd_df.columns:
        print("\nPD1 Features (loops/holes):")
        for dataset in ['GT', 'SR']:
            subset = pd_df[pd_df['dataset'] == dataset]
            if len(subset) > 0:
                print(f"  {dataset}: {subset['n_pd1'].mean():.1f} ± {subset['n_pd1'].std():.1f}")
    
    # Merge tree branches by dataset
    mt_df = df[df['type'] == 'MT']
    if 'n_branches' in mt_df.columns:
        print("\nMerge Tree Branches:")
        for dataset in ['GT', 'SR']:
            subset = mt_df[mt_df['dataset'] == dataset]
            if len(subset) > 0:
                print(f"  {dataset}: {subset['n_branches'].mean():.1f} ± {subset['n_branches'].std():.1f}")

if __name__ == '__main__':
    main()
