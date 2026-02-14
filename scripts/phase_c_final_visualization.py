#!/usr/bin/env python3
"""
Visualize the outputs produced by compute_composite_tree_distance.py

Reads from --outdir:
  - pd_pairwise_distances.csv  (method,key,port,distance)
  - mt_pairwise_distances.csv  (method,key,distance)
  - pd_summary_by_method.csv
  - mt_summary_by_method.csv

Writes:
  - phase_c_all_distances.png
  - phase_c_summary.png

This script tries to be dependency-light:
  - uses only Python stdlib + matplotlib (if available)
If matplotlib isn't available, it will still print a textual summary.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

def _read_csv(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="") as f:
        r = csv.DictReader(f)
        return list(r)

def _to_float(x: str) -> float:
    try:
        v = float(x)
        return v
    except Exception:
        return float("nan")

def _safe(vals: List[float]) -> List[float]:
    return [v for v in vals if not (v is None or math.isnan(v))]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, required=True)
    args = ap.parse_args()

    outdir: Path = args.outdir
    pd_rows = _read_csv(outdir / "pd_pairwise_distances.csv")
    mt_rows = _read_csv(outdir / "mt_pairwise_distances.csv")
    pd_summary = _read_csv(outdir / "pd_summary_by_method.csv")
    mt_summary = _read_csv(outdir / "mt_summary_by_method.csv")

    print("Loaded:")
    print("  PD rows:", len(pd_rows))
    print("  MT rows:", len(mt_rows))

    # If matplotlib isn't installed, stop after printing.
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:
        print("\nmatplotlib not available:", e)
        print("Skipping plots; CSV outputs are still ready.")
        return 0

    # --- Plot 1: per-sample distances (scatter-ish using dots) ---
    # Prepare PD by method+port
    pd_by = defaultdict(list)  # (method,port) -> [dist]
    for r in pd_rows:
        m = r.get("method", "")
        port = int(r.get("port", "0"))
        d = _to_float(r.get("distance", "nan"))
        if not math.isnan(d):
            pd_by[(m, port)].append(d)

    mt_by = defaultdict(list)  # method -> [dist]
    for r in mt_rows:
        m = r.get("method", "")
        d = _to_float(r.get("distance", "nan"))
        if not math.isnan(d):
            mt_by[m].append(d)

    # Make a simple figure: PD (port0) and MT side-by-side by method
    methods = sorted(set([k[0] for k in pd_by.keys()] + list(mt_by.keys())))
    pd_port0_means = []
    mt_means = []
    for m in methods:
        pd0 = _safe(pd_by.get((m, 0), []))
        mtv = _safe(mt_by.get(m, []))
        pd_port0_means.append(sum(pd0)/len(pd0) if pd0 else float("nan"))
        mt_means.append(sum(mtv)/len(mtv) if mtv else float("nan"))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = list(range(len(methods)))
    # plot two series as points
    ax.plot(x, pd_port0_means, marker="o", linestyle="None", label="PD mean (port0)")
    ax.plot(x, mt_means, marker="s", linestyle="None", label="MT mean")

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=30, ha="right")
    ax.set_ylabel("Distance")
    ax.set_title("Phase C: Mean distances by method")
    ax.legend()
    fig.tight_layout()
    out1 = outdir / "phase_c_summary.png"
    fig.savefig(out1, dpi=200)
    print("Wrote:", out1)

    # --- Plot 2: distributions (boxplot) ---
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)

    # For each method, build two distributions: PD port0 and MT
    labels = []
    data = []
    for m in methods:
        pd0 = _safe(pd_by.get((m, 0), []))
        if pd0:
            labels.append(f"{m}\nPD0")
            data.append(pd0)
        mtv = _safe(mt_by.get(m, []))
        if mtv:
            labels.append(f"{m}\nMT")
            data.append(mtv)

    if data:
        ax2.boxplot(data, labels=labels, showfliers=False)
        ax2.set_ylabel("Distance")
        ax2.set_title("Phase C: Distance distributions (PD port0 vs MT)")
        fig2.tight_layout()
        out2 = outdir / "phase_c_all_distances.png"
        fig2.savefig(out2, dpi=200)
        print("Wrote:", out2)
    else:
        print("No finite distances found to plot distributions.")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
