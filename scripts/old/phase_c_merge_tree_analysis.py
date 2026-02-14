#!/usr/bin/env python3
"""
Phase C: Merge Tree Analysis (matching Phase B style)
Compare GT vs SR using TTK merge tree metrics
"""
import vtk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict

# Set style to match Phase B
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

def extract_pd_features(vtu_path):
    """Extract features from persistence diagram"""
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(vtu_path))
    reader.Update()
    pd = reader.GetOutput()
    
    n_points = pd.GetNumberOfPoints()
    
    features = {
        'n_critical_points': n_points,
    }
    
    # Get persistence values
    pers_vals = []
    if pd.GetPointData().GetArray("Persistence"):
        pers = pd.GetPointData().GetArray("Persistence")
        pers_vals = [pers.GetValue(i) for i in range(n_points)]
        pers_vals = [p for p in pers_vals if p > 0]
        
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
    pers_vals = []
    if tree.GetPointData().GetArray("Persistence"):
        pers = tree.GetPointData().GetArray("Persistence")
        pers_vals = [pers.GetValue(i) for i in range(features['n_nodes'])]
        pers_vals = [p for p in pers_vals if p > 0]
        
        if pers_vals:
            features['max_pers'] = max(pers_vals)
            features['mean_pers'] = np.mean(pers_vals)
            features['total_pers'] = sum(pers_vals)
            features['n_branches'] = len([p for p in pers_vals if p > 0.1])
        else:
            features['max_pers'] = 0
            features['mean_pers'] = 0
            features['total_pers'] = 0
            features['n_branches'] = 0
    
    return features

def main():
    print("="*70)
    print("PHASE C: MERGE TREE ANALYSIS")
    print("="*70)
    
    # Collect all results
    results = []
    
    # Process Persistence Diagrams
    print("\nProcessing Persistence Diagrams...")
    pd_dir = Path('ttk_outputs/pd')
    for vtu_file in sorted(pd_dir.glob('gan_*_pd_port_0.vtu')):
        parts = vtu_file.stem.replace('_pd_port_0', '').split('_')
        
        method = parts[0]  # gan
        dataset = parts[1]  # GT or SR
        sample = parts[2]   # s0, s1, etc.
        
        features = extract_pd_features(vtu_file)
        features.update({
            'method': method.upper(),
            'dataset': dataset,
            'sample': sample,
            'label': f'{method.upper()}_{dataset}' if dataset == 'SR' else dataset,
        })
        
        results.append(features)
        print(f"  {dataset} {sample}: {features.get('n_pd1', 0)} PD1, {features.get('n_pd0', 0)} PD0")
    
    # Process Merge Trees
    print("\nProcessing Merge Trees...")
    mt_dir = Path('ttk_outputs/mt')
    for vtu_file in sorted(mt_dir.glob('gan_*_mt_port_0.vtu')):
        parts = vtu_file.stem.replace('_mt_port_0', '').split('_')
        
        method = parts[0]
        dataset = parts[1]
        sample = parts[2]
        
        features = extract_mt_features(vtu_file)
        
        # Find matching PD entry and merge
        matching = [r for r in results if r['dataset'] == dataset and r['sample'] == sample]
        if matching:
            matching[0].update({
                'mt_nodes': features['n_nodes'],
                'mt_arcs': features['n_arcs'],
                'mt_branches': features['n_branches'],
                'mt_max_pers': features['max_pers'],
                'mt_mean_pers': features['mean_pers'],
                'mt_total_pers': features['total_pers'],
            })
        
        print(f"  {dataset} {sample}: {features['n_branches']} branches, {features['n_nodes']} nodes")
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save full results
    df.to_csv('ttk_outputs/phase_c_results.csv', index=False)
    print(f"\n✓ Saved: ttk_outputs/phase_c_results.csv")
    
    # Compute summary statistics (per-sample averages)
    summary_stats = df.groupby('label').agg({
        'n_pd0': ['mean', 'std'],
        'n_pd1': ['mean', 'std'],
        'mt_nodes': ['mean', 'std'],
        'mt_branches': ['mean', 'std'],
        'mt_total_pers': ['mean', 'std'],
    }).round(2)
    
    summary_stats.columns = ['_'.join(col).strip() for col in summary_stats.columns.values]
    summary_stats['n_samples'] = df.groupby('label').size()
    
    summary_stats.to_csv('ttk_outputs/phase_c_summary_stats.csv')
    print(f"✓ Saved: ttk_outputs/phase_c_summary_stats.csv")
    
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    print(summary_stats)
    
    # Create visualizations
    print("\n" + "="*70)
    print("CREATING VISUALIZATIONS")
    print("="*70)
    
    fig_dir = Path('ttk_outputs/figs')
    fig_dir.mkdir(exist_ok=True)
    
    # Bar plot: PD1 features
    plt.figure(figsize=(10, 6))
    summary = df.groupby('label')['n_pd1'].agg(['mean', 'std']).reset_index()
    plt.bar(summary['label'], summary['mean'], yerr=summary['std'], capsize=10)
    plt.ylabel('Mean PD1 Features (loops/holes)')
    plt.title('Mean PD1 Features vs GT on speed (per-sample avg)')
    plt.tight_layout()
    plt.savefig(fig_dir / 'bar_mean_pd1_features.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created: bar_mean_pd1_features.png")
    
    # Bar plot: Merge Tree Branches
    plt.figure(figsize=(10, 6))
    summary = df.groupby('label')['mt_branches'].agg(['mean', 'std']).reset_index()
    plt.bar(summary['label'], summary['mean'], yerr=summary['std'], capsize=10)
    plt.ylabel('Mean MT Branches')
    plt.title('Mean Merge Tree Branches on speed (per-sample avg)')
    plt.tight_layout()
    plt.savefig(fig_dir / 'bar_mean_mt_branches.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created: bar_mean_mt_branches.png")
    
    # Bar plot: MT Nodes
    plt.figure(figsize=(10, 6))
    summary = df.groupby('label')['mt_nodes'].agg(['mean', 'std']).reset_index()
    plt.bar(summary['label'], summary['mean'], yerr=summary['std'], capsize=10)
    plt.ylabel('Mean MT Nodes')
    plt.title('Mean Merge Tree Nodes on speed (per-sample avg)')
    plt.tight_layout()
    plt.savefig(fig_dir / 'bar_mean_mt_nodes.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created: bar_mean_mt_nodes.png")
    
    # Bar plot: MT Total Persistence
    plt.figure(figsize=(10, 6))
    summary = df.groupby('label')['mt_total_pers'].agg(['mean', 'std']).reset_index()
    plt.bar(summary['label'], summary['mean'], yerr=summary['std'], capsize=10)
    plt.ylabel('Mean MT Total Persistence')
    plt.title('Mean Merge Tree Total Persistence on speed (per-sample avg)')
    plt.tight_layout()
    plt.savefig(fig_dir / 'bar_mean_mt_total_pers.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created: bar_mean_mt_total_pers.png")
    
    # Scatter plot: PD1 vs MT branches
    plt.figure(figsize=(10, 6))
    for label in df['label'].unique():
        subset = df[df['label'] == label]
        plt.scatter(subset['n_pd1'], subset['mt_branches'], label=label, alpha=0.7, s=100)
    plt.xlabel('PD1 Features (loops/holes)')
    plt.ylabel('MT Branches')
    plt.title('PD1 Features vs Merge Tree Branches (per-sample)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / 'scatter_pd1_vs_mt_branches.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created: scatter_pd1_vs_mt_branches.png")
    
    # Key findings
    print("\n" + "="*70)
    print("KEY FINDINGS")
    print("="*70)
    
    gt = df[df['dataset'] == 'GT']
    sr = df[df['dataset'] == 'SR']
    
    if len(gt) > 0 and len(sr) > 0:
        # PD1 comparison
        gt_pd1 = gt['n_pd1'].mean()
        sr_pd1 = sr['n_pd1'].mean()
        pd1_diff = ((sr_pd1 - gt_pd1) / gt_pd1 * 100) if gt_pd1 > 0 else 0
        
        print(f"\nPD1 Features (loops/holes):")
        print(f"  GT:  {gt_pd1:.1f} ± {gt['n_pd1'].std():.1f}")
        print(f"  SR:  {sr_pd1:.1f} ± {sr['n_pd1'].std():.1f}")
        print(f"  Diff: {pd1_diff:+.1f}%")
        
        if pd1_diff > 50:
            print("  → GAN SR creates MORE topological features (possible hallucination)")
        elif pd1_diff < -20:
            print("  → GAN SR loses significant topology")
        else:
            print("  → GAN SR preserves PD1 topology reasonably well")
        
        # MT branches comparison
        gt_branches = gt['mt_branches'].mean()
        sr_branches = sr['mt_branches'].mean()
        branch_diff = ((sr_branches - gt_branches) / gt_branches * 100) if gt_branches > 0 else 0
        
        print(f"\nMerge Tree Branches:")
        print(f"  GT:  {gt_branches:.1f} ± {gt['mt_branches'].std():.1f}")
        print(f"  SR:  {sr_branches:.1f} ± {sr['mt_branches'].std():.1f}")
        print(f"  Diff: {branch_diff:+.1f}%")
        
        if branch_diff > 50:
            print("  → GAN SR creates MORE hierarchical structure")
        elif branch_diff < -20:
            print("  → GAN SR simplifies hierarchical structure")
        else:
            print("  → GAN SR preserves hierarchical structure reasonably well")
        
        # MT nodes comparison  
        gt_nodes = gt['mt_nodes'].mean()
        sr_nodes = sr['mt_nodes'].mean()
        node_diff = ((sr_nodes - gt_nodes) / gt_nodes * 100) if gt_nodes > 0 else 0
        
        print(f"\nMerge Tree Nodes:")
        print(f"  GT:  {gt_nodes:.1f} ± {gt['mt_nodes'].std():.1f}")
        print(f"  SR:  {sr_nodes:.1f} ± {sr['mt_nodes'].std():.1f}")
        print(f"  Diff: {node_diff:+.1f}%")
    
    print("\n" + "="*70)
    print("✓ Phase C Analysis Complete!")
    print("="*70)
    print(f"\nOutputs in: ttk_outputs/")
    print("  - phase_c_results.csv")
    print("  - phase_c_summary_stats.csv")
    print("  - figs/*.png")

if __name__ == '__main__':
    main()
