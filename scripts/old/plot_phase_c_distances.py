#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    outdir = Path("ttk_outputs/direct")
    df = pd.read_csv(outdir / "distances_pairwise.csv")

    plots = outdir / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    x = np.arange(len(df))
    labels = df["sample"].tolist()

    def bar(metric, title, ylabel, fname):
        if metric not in df.columns:
            return
        y = df[metric].astype(float).values
        plt.figure()
        plt.bar(x, y)
        plt.xticks(x, labels)
        plt.title(title)
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(plots / fname, dpi=300, bbox_inches="tight")
        plt.close()
        print("✓", fname)

    bar("pd_bottleneck", "PD Bottleneck Distance (GT vs SR)", "Distance", "bar_pd_bottleneck.png")
    bar("pd_wasserstein_p1", "PD Wasserstein-1 Distance (GT vs SR)", "Distance", "bar_pd_w1.png")
    bar("pd_wasserstein_p2", "PD Wasserstein-2 Distance (GT vs SR)", "Distance", "bar_pd_w2.png")
    bar("mt_distance", "Merge Tree Distance (GT vs SR)", "Distance", "bar_mt_distance.png")

    # Combined line plot
    plt.figure()
    for m in ["pd_bottleneck", "pd_wasserstein_p1", "pd_wasserstein_p2", "mt_distance"]:
        if m in df.columns and df[m].notna().any():
            plt.plot(x, df[m].astype(float).values, marker="o", label=m)
    plt.xticks(x, labels)
    plt.title("Distances per sample (GT vs SR)")
    plt.ylabel("Distance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots / "lines_all_distances.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✓ lines_all_distances.png")

if __name__ == "__main__":
    main()
