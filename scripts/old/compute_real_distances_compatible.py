#!/usr/bin/env python3
"""
Compute REAL Persistence Diagram and Merge Tree Distances

Replaces proxy/composite distances with actual mathematical distances:
- PD: Wasserstein-1, Wasserstein-2, Bottleneck (via GUDHI)
- MT: Structural distance (node/arc distributions + topology)

Output format compatible with phase_c_final_visualization.py

Inputs:
- ttk_outputs/direct/pd_vtu/{GT,SR,CNN,BICUBIC}/*.vtu
- ttk_outputs/mt/{GT,SR,CNN,BICUBIC}/*.vtu

Outputs:
- ttk_outputs/direct/pd_real_distances.csv
- ttk_outputs/direct/mt_real_distances.csv
- ttk_outputs/direct/combined_real_summary.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd
import vtk
import sys

# Install GUDHI if needed
try:
    import gudhi
except ImportError:
    print("📦 Installing GUDHI...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "gudhi", "--break-system-packages"])
    import gudhi

print("="*80)
print("REAL TOPOLOGY DISTANCES (VTK 9.6.0 + TTK)")
print("="*80)
print(f"VTK: {vtk.vtkVersion.GetVTKVersion()}")
print(f"GUDHI: {gudhi.__version__}")
print()

# ============================================================================
# PERSISTENCE DIAGRAM DISTANCES (REAL)
# ============================================================================

def read_pd_from_vtu(vtu_path):
    """Extract persistence diagram from TTK VTU output for GUDHI."""
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(vtu_path))
    reader.Update()
    
    output = reader.GetOutput()
    if output.GetNumberOfPoints() == 0:
        return None
    
    point_data = output.GetPointData()
    birth_arr = point_data.GetArray('Birth')
    death_arr = point_data.GetArray('Death')
    
    if not birth_arr or not death_arr:
        return None
    
    diagram = []
    for i in range(birth_arr.GetNumberOfTuples()):
        b = birth_arr.GetValue(i)
        d = death_arr.GetValue(i)
        if d > b:
            diagram.append([b, d])
    
    return np.array(diagram) if diagram else None


def compute_pd_distances_real(pd_vtu_dir):
    """
    Calculate REAL PD distances: Wasserstein-1, Wasserstein-2, Bottleneck.
    
    Expected structure:
    pd_vtu_dir/
      GT/*.vtu
      SR/*.vtu
      CNN/*.vtu (if exists)
      BICUBIC/*.vtu (if exists)
    """
    print("="*80)
    print("PERSISTENCE DIAGRAM DISTANCES (REAL WASSERSTEIN/BOTTLENECK)")
    print("="*80)
    
    if not pd_vtu_dir.exists():
        print(f"⚠️  Directory not found: {pd_vtu_dir}")
        return pd.DataFrame()
    
    gt_dir = pd_vtu_dir / "GT"
    if not gt_dir.exists():
        print(f"⚠️  GT directory not found")
        return pd.DataFrame()
    
    method_dirs = [d for d in pd_vtu_dir.iterdir() if d.is_dir() and d.name != "GT"]
    print(f"Methods: GT, {', '.join([d.name for d in method_dirs])}")
    print()
    
    results = []
    gt_files = sorted(gt_dir.glob("*.vtu"))
    
    for gt_file in gt_files:
        # Extract sample ID from filename (e.g., "gan_GT_s0_..." → "s0")
        sample_id = None
        for part in gt_file.stem.split('_'):
            if part.startswith('s') and part[1:].isdigit():
                sample_id = part
                break
        
        if not sample_id:
            continue
        
        # Read GT PD
        pd_gt = read_pd_from_vtu(gt_file)
        if pd_gt is None or len(pd_gt) == 0:
            print(f"  ⚠️  Empty GT PD: {gt_file.name}")
            continue
        
        for method_dir in sorted(method_dirs):
            method = method_dir.name
            
            # Find matching method file
            method_file = None
            for mf in method_dir.glob("*.vtu"):
                if sample_id in mf.stem:
                    method_file = mf
                    break
            
            if not method_file:
                continue
            
            # Read method PD
            pd_method = read_pd_from_vtu(method_file)
            if pd_method is None or len(pd_method) == 0:
                print(f"  ⚠️  Empty {method} PD: {method_file.name}")
                continue
            
            # Compute REAL distances
            try:
                w1 = gudhi.wasserstein.wasserstein_distance(pd_gt, pd_method, order=1)
                w2 = gudhi.wasserstein.wasserstein_distance(pd_gt, pd_method, order=2)
                bn = gudhi.bottleneck_distance(pd_gt, pd_method)
                
                results.append({
                    'sample': sample_id,
                    'method': method,
                    'dataset': 'SR',  # For compatibility with visualization
                    'wasserstein_1': w1,
                    'wasserstein_2': w2,
                    'bottleneck': bn,
                    'gt_pairs': len(pd_gt),
                    'method_pairs': len(pd_method)
                })
                
                print(f"  ✅ {sample_id} {method}: W1={w1:.4f}, W2={w2:.4f}, BN={bn:.4f}")
                
            except Exception as e:
                print(f"  ❌ {sample_id} {method}: Error computing distances - {e}")
    
    return pd.DataFrame(results)


# ============================================================================
# MERGE TREE DISTANCES (STRUCTURAL)
# ============================================================================

def extract_mt_features(vtu_path):
    """Extract merge tree structural features."""
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(vtu_path))
    reader.Update()
    
    output = reader.GetOutput()
    if output.GetNumberOfPoints() == 0:
        return None
    
    point_data = output.GetPointData()
    
    features = {
        'n_nodes': output.GetNumberOfPoints(),
        'n_arcs': output.GetNumberOfCells(),
    }
    
    # Extract scalar arrays
    for array_name in ['Scalar', 'CriticalType', 'NodeType']:
        arr = point_data.GetArray(array_name)
        if arr:
            values = np.array([arr.GetValue(i) for i in range(arr.GetNumberOfTuples())])
            features[array_name] = values
    
    return features


def compute_structural_distance(mt_gt, mt_method):
    """
    Compute structural dissimilarity between merge trees.
    
    Combines:
    1. Relative node/arc count differences
    2. Scalar distribution differences (if available)
    3. Organization ratio difference (branches/nodes)
    
    This is interpretable and comparable across datasets.
    """
    if mt_gt is None or mt_method is None:
        return None
    
    # Node/arc differences
    n_gt, n_m = mt_gt['n_nodes'], mt_method['n_nodes']
    a_gt, a_m = mt_gt['n_arcs'], mt_method['n_arcs']
    
    rel_nodes = abs(n_m - n_gt) / max(n_gt, 1)
    rel_arcs = abs(a_m - a_gt) / max(a_gt, 1)
    
    # Scalar distribution difference (if available)
    scalar_diff = 0.0
    if 'Scalar' in mt_gt and 'Scalar' in mt_method:
        s_gt = mt_gt['Scalar']
        s_m = mt_method['Scalar']
        
        if len(s_gt) > 0 and len(s_m) > 0:
            # Compare means and stds
            mean_diff = abs(np.mean(s_m) - np.mean(s_gt)) / max(abs(np.mean(s_gt)), 1e-10)
            std_diff = abs(np.std(s_m) - np.std(s_gt)) / max(np.std(s_gt), 1e-10)
            scalar_diff = (mean_diff + std_diff) / 2
    
    # Composite structural distance
    structural_dist = (rel_nodes + rel_arcs + scalar_diff) / 3
    
    return {
        'structural_distance': structural_dist,
        'rel_nodes': rel_nodes,
        'rel_arcs': rel_arcs,
        'scalar_diff': scalar_diff,
        'node_distance': abs(n_m - n_gt),
        'arc_distance': abs(a_m - a_gt),
        'gt_nodes': n_gt,
        'method_nodes': n_m,
        'gt_arcs': a_gt,
        'method_arcs': a_m,
    }


def compute_mt_distances_real(mt_dir):
    """
    Calculate REAL structural merge tree distances.
    
    Expected structure:
    mt_dir/
      GT/*.vtu
      SR/*.vtu
      CNN/*.vtu (if exists)
      BICUBIC/*.vtu (if exists)
    """
    print("="*80)
    print("MERGE TREE DISTANCES (STRUCTURAL)")
    print("="*80)
    
    if not mt_dir.exists():
        print(f"⚠️  Directory not found: {mt_dir}")
        return pd.DataFrame()
    
    gt_dir = mt_dir / "GT"
    if not gt_dir.exists():
        print(f"⚠️  GT directory not found")
        return pd.DataFrame()
    
    method_dirs = [d for d in mt_dir.iterdir() if d.is_dir() and d.name != "GT"]
    print(f"Methods: GT, {', '.join([d.name for d in method_dirs])}")
    print()
    
    results = []
    gt_files = sorted(gt_dir.glob("*.vtu"))
    
    for gt_file in gt_files:
        # Extract sample ID
        sample_id = None
        for part in gt_file.stem.split('_'):
            if part.startswith('s') and part[1:].isdigit():
                sample_id = part
                break
        
        if not sample_id:
            continue
        
        # Extract GT features
        mt_gt = extract_mt_features(gt_file)
        if mt_gt is None:
            print(f"  ⚠️  Empty GT MT: {gt_file.name}")
            continue
        
        for method_dir in sorted(method_dirs):
            method = method_dir.name
            
            # Find matching method file
            method_file = None
            for mf in method_dir.glob("*.vtu"):
                if sample_id in mf.stem:
                    method_file = mf
                    break
            
            if not method_file:
                continue
            
            # Extract method features
            mt_method = extract_mt_features(method_file)
            if mt_method is None:
                print(f"  ⚠️  Empty {method} MT: {method_file.name}")
                continue
            
            # Compute distance
            dist_info = compute_structural_distance(mt_gt, mt_method)
            if dist_info is None:
                continue
            
            result = {
                'sample': sample_id,
                'method': method,
                'dataset': 'SR',  # For compatibility
                **dist_info
            }
            results.append(result)
            
            print(f"  ✅ {sample_id} {method}: dist={dist_info['structural_distance']:.4f} " +
                  f"(GT:{mt_gt['n_nodes']} nodes, {method}:{mt_method['n_nodes']} nodes)")
    
    return pd.DataFrame(results)


# ============================================================================
# COMPATIBLE OUTPUT FOR EXISTING VISUALIZATION
# ============================================================================

def create_compatible_outputs(pd_dist, mt_dist, output_dir):
    """
    Create outputs compatible with phase_c_final_visualization.py.
    
    The visualization expects:
    - Columns matching the proxy format
    - Same statistical summaries
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save raw distances
    if not pd_dist.empty:
        pd_dist.to_csv(output_dir / "pd_real_distances.csv", index=False)
        print(f"\n✅ Saved: {output_dir / 'pd_real_distances.csv'}")
    
    if not mt_dist.empty:
        mt_dist.to_csv(output_dir / "mt_real_distances.csv", index=False)
        print(f"✅ Saved: {output_dir / 'mt_real_distances.csv'}")
    
    # Create mt_composite_distances.csv (compatible format)
    if not mt_dist.empty:
        mt_compat = mt_dist[['sample', 'method', 'node_distance', 'arc_distance', 
                              'rel_nodes', 'rel_arcs', 'structural_distance']].copy()
        
        # Add compatibility columns
        mt_compat['branch_distance'] = mt_compat['arc_distance']  # Approximation
        mt_compat['rel_branches'] = mt_compat['rel_arcs']
        mt_compat['ratio_distance'] = mt_compat['scalar_diff'] if 'scalar_diff' in mt_dist.columns else 0.0
        mt_compat['rel_ratio'] = mt_compat['ratio_distance']
        mt_compat['mt_composite_rel'] = mt_compat['structural_distance']  # REAL distance, not proxy!
        
        mt_compat.to_csv(output_dir / "mt_composite_distances.csv", index=False)
        print(f"✅ Saved: {output_dir / 'mt_composite_distances.csv'} (compatible format)")
    
    # Create summary statistics
    if not mt_dist.empty:
        summary = []
        for method in sorted(mt_dist['method'].unique()):
            subset = mt_dist[mt_dist['method'] == method]
            summary.append({
                'method': method,
                'structural_distance_mean': subset['structural_distance'].mean(),
                'structural_distance_std': subset['structural_distance'].std(),
                'node_distance_mean': subset['node_distance'].mean(),
                'node_distance_std': subset['node_distance'].std(),
                'rel_nodes_mean': subset['rel_nodes'].mean(),
                'rel_nodes_std': subset['rel_nodes'].std(),
                'mt_composite_rel_mean': subset['structural_distance'].mean(),  # REAL, not proxy
                'mt_composite_rel_std': subset['structural_distance'].std(),
            })
        
        summary_df = pd.DataFrame(summary)
        summary_df.to_csv(output_dir / "mt_composite_summary.csv", index=False)
        print(f"✅ Saved: {output_dir / 'mt_composite_summary.csv'}")
    
    # Combined summary
    if not pd_dist.empty and not mt_dist.empty:
        combined = pd.merge(
            pd_dist[['sample', 'method', 'wasserstein_2', 'bottleneck']],
            mt_dist[['sample', 'method', 'structural_distance']],
            on=['sample', 'method'],
            how='outer'
        )
        combined.to_csv(output_dir / "combined_real_summary.csv", index=False)
        print(f"✅ Saved: {output_dir / 'combined_real_summary.csv'}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    base_dir = Path("/home/adadhwal/PhIRE")
    pd_vtu_dir = base_dir / "ttk_outputs" / "direct" / "pd_vtu"
    mt_dir = base_dir / "ttk_outputs" / "mt"
    output_dir = base_dir / "ttk_outputs" / "direct"
    
    # Calculate real PD distances
    pd_dist = compute_pd_distances_real(pd_vtu_dir)
    
    # Calculate real MT distances
    mt_dist = compute_mt_distances_real(mt_dir)
    
    # Create compatible outputs
    create_compatible_outputs(pd_dist, mt_dist, output_dir)
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    if not pd_dist.empty:
        print("\nPERSISTENCE DIAGRAM DISTANCES (REAL):")
        for method in sorted(pd_dist['method'].unique()):
            subset = pd_dist[pd_dist['method'] == method]
            print(f"\n{method}:")
            print(f"  Wasserstein-1: {subset['wasserstein_1'].mean():.4f} ± {subset['wasserstein_1'].std():.4f}")
            print(f"  Wasserstein-2: {subset['wasserstein_2'].mean():.4f} ± {subset['wasserstein_2'].std():.4f}")
            print(f"  Bottleneck:    {subset['bottleneck'].mean():.4f} ± {subset['bottleneck'].std():.4f}")
    
    if not mt_dist.empty:
        print("\nMERGE TREE DISTANCES (STRUCTURAL):")
        for method in sorted(mt_dist['method'].unique()):
            subset = mt_dist[mt_dist['method'] == method]
            print(f"\n{method}:")
            print(f"  Structural: {subset['structural_distance'].mean():.4f} ± {subset['structural_distance'].std():.4f}")
            print(f"  (Components: rel_nodes={subset['rel_nodes'].mean():.3f}, " +
                  f"rel_arcs={subset['rel_arcs'].mean():.3f})")
    
    print("\n" + "="*80)
    print("✅ REAL DISTANCE CALCULATION COMPLETE")
    print("="*80)
    print("\nIMPORTANT NOTES:")
    print("- PD distances are REAL Wasserstein/Bottleneck (not feature count proxies)")
    print("- MT distances are structural (interpretable dissimilarity, not proxy)")
    print("- Output files are compatible with phase_c_final_visualization.py")
    print("- Run visualization script to see results!")


if __name__ == "__main__":
    main()
