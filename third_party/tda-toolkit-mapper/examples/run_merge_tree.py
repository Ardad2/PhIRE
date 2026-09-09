import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tda_toolkit.merge_tree import get_merge_tree_graph_1d, plot_merge_tree_graph_1d

def main():
    # Load Data
    df = pd.read_csv('./avg_pow_ts.csv')
    scalar_field = df['avg_pow'].values

    # Compute Merge Tree (using default direction=1 for Minima-based)
    G_mt, coords_plot, _ = get_merge_tree_graph_1d(scalar_field, direction=1)

    # Plot 2: Merge Tree Graph Only (Saves as merge_tree_graph_only.png)
    plot_merge_tree_graph_1d(G_mt, coords_plot, scalar_field, overlay=False)
    plt.savefig('merge_tree_graph_only.png')
    plt.close()
    print("Successfully saved: merge_tree_graph_only.png")

    # Plot 3: Merge Tree Overlay (Saves as merge_tree_overlay.png)
    plot_merge_tree_graph_1d(G_mt, coords_plot, scalar_field, overlay=True)
    plt.savefig('merge_tree_overlay.png')
    plt.close()
    print("Successfully saved: merge_tree_overlay.png")

    # Save Original Data Plot 
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(np.arange(len(scalar_field)), scalar_field, color='blue', linewidth=1.5, alpha=0.7)
    ax.set_title("Original 1D Scalar Field (Time Series)")
    ax.set_xlabel("Time Index (Hours)")
    ax.set_ylabel("Average Power Value")
    plt.tight_layout()
    plt.savefig('original_data.png')
    plt.close(fig)
    print("Successfully saved: original_data.png")

if __name__ == "__main__":
    main()