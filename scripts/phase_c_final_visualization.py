#!/usr/bin/env python3
"""
Phase C Visualization for REAL distances computed by compute_composite_tree_distance.py

Reads (from --outdir):
- pd_pairwise_distances.csv
- mt_pairwise_distances.csv
- pd_summary_by_method.csv
- mt_summary_by_method.csv

Writes (to same outdir):
- phase_c_all_distances.png
- phase_c_summary.png
- phase_c_summary.txt
"""

import argparse
import csv
import math
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except Exception as e:
    plt = None
    _PLOT_IMPORT_ERROR = e



def _read_csv(path: Path):
    with path.open("r", newline="") as f:
        r = csv.DictReader(f)
        return list(r)


def _to_float(x):
    try:
        return float(x)
    except Exception:
        return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True, help="Same outdir used by compute script")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    pd_path = outdir / "pd_pairwise_distances.csv"
    mt_path = outdir / "mt_pairwise_distances.csv"
    pd_sum_path = outdir / "pd_summary_by_method.csv"
    mt_sum_path = outdir / "mt_summary_by_method.csv"

    all_png = outdir / "phase_c_all_distances.png"
    sum_png = outdir / "phase_c_summary.png"
    pd_rows = _read_csv(pd_path) if pd_path.exists() else []
    mt_rows = _read_csv(mt_path) if mt_path.exists() else []
    pd_sum = _read_csv(pd_sum_path) if pd_sum_path.exists() else []
    mt_sum = _read_csv(mt_sum_path) if mt_sum_path.exists() else []

    # Choose a PD column to plot (prefer port0)
    pd_col = None
    if pd_rows:
        cols = list(pd_rows[0].keys())
        if "pd_bottleneck_port0" in cols:
            pd_col = "pd_bottleneck_port0"
        else:
            for c in cols:
                if c.startswith("pd_bottleneck_port"):
                    pd_col = c
                    break

    # ------------------ Plot: all distances (sorted) ------------------
    if plt is None:
        print("matplotlib not available; skipping PNG plots. Error:", _PLOT_IMPORT_ERROR)
        # Still write the text summary below.
    
    if plt is not None:
        fig = plt.figure(figsize=(14, 8))
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2)

        if pd_rows and pd_col:
        vals = [_to_float(r.get(pd_col)) for r in pd_rows]
        vals_sorted = sorted([v for v in vals if math.isfinite(v)])
        ax1.plot(vals_sorted, marker="o", linestyle="-")
        ax1.set_title(f"PD Bottleneck Distances ({pd_col})")
        ax1.set_ylabel("Distance")
        ax1.set_xlabel("Pairs (sorted)")

        if mt_rows:
        vals = [_to_float(r.get("mt_distance")) for r in mt_rows]
        vals_sorted = sorted([v for v in vals if math.isfinite(v)])
        ax2.plot(vals_sorted, marker="o", linestyle="-")
        ax2.set_title("Merge Tree Distances (ttkMergeTreeDistanceMatrix)")
        ax2.set_ylabel("Distance")
        ax2.set_xlabel("Pairs (sorted)")

            fig.tight_layout()
        all_png = outdir / "phase_c_all_distances.png"
        fig.savefig(all_png, dpi=200)
            plt.close(fig)

    # ------------------ Plot: summary by method ------------------
    if plt is not None:

    # Simple bar plots of means
        fig = plt.figure(figsize=(14, 5))
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)

        if pd_sum:
        methods = [r["method"] for r in pd_sum]
        means = [_to_float(r["mean"]) for r in pd_sum]
        ax1.bar(methods, means)
        ax1.set_title("PD mean bottleneck by method")
        ax1.set_ylabel("Mean distance")
        ax1.tick_params(axis="x", rotation=45)

        if mt_sum:
        methods = [r["method"] for r in mt_sum]
        means = [_to_float(r["mean"]) for r in mt_sum]
        ax2.bar(methods, means)
        ax2.set_title("MT mean distance by method")
        ax2.set_ylabel("Mean distance")
        ax2.tick_params(axis="x", rotation=45)

        fig.tight_layout()
        sum_png = outdir / "phase_c_summary.png"
        fig.savefig(sum_png, dpi=200)
        plt.close(fig)

    # ------------------ Text summary ------------------
    txt = outdir / "phase_c_summary.txt"
    with txt.open("w") as f:
        f.write("Phase C REAL distances summary\n")
        f.write("=" * 60 + "\n\n")
        if pd_col:
            f.write(f"PD column plotted: {pd_col}\n")
        f.write(f"PD pairs: {len(pd_rows)}\n")
        f.write(f"MT pairs: {len(mt_rows)}\n\n")

            if pd_sum:
            f.write("PD summary by method\n")
            f.write("-" * 60 + "\n")
            for r in pd_sum:
                f.write(f"{r['method']}: n={r['n']} mean={r['mean']} median={r['median']} min={r['min']} max={r['max']}\n")
            f.write("\n")

            if mt_sum:
            f.write("MT summary by method\n")
            f.write("-" * 60 + "\n")
            for r in mt_sum:
                f.write(f"{r['method']}: n={r['n']} mean={r['mean']} median={r['median']} min={r['min']} max={r['max']}\n")

    print("Wrote:")
    print(" ", all_png)
    print(" ", sum_png)
    print(" ", txt)


if __name__ == "__main__":
    main()
