#!/usr/bin/env python3
"""phase_c_final_visualization_real.py

Visualization for REAL TTK distances produced by compute_composite_tree_distance_real.py.

Input directory (--indir) must contain:
  - pd_pairwise_distances.csv
  - mt_pairwise_distances.csv
  - phase_c_results.csv

Outputs written into --indir (by default):
  - phase_c_all_distances.png
  - phase_c_summary.png

Dependencies:
  - matplotlib
  - (optional) pandas

Usage:
  python3 phase_c_final_visualization_real.py --indir /home/adadhwal/PhIRE/ttk_outputs/phase_c_final
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception as e:
    raise SystemExit(f"ERROR: matplotlib required: {e}\nInstall: sudo apt-get install -y python3-matplotlib")

try:
    import pandas as pd  # type: ignore
except Exception:
    pd = None


def _parse_float(x: object) -> Optional[float]:
    try:
        if x is None:
            return None
        s = str(x).strip()
        if not s:
            return None
        return float(s)
    except Exception:
        return None


def _load_cols_csv(path: Path, col: str) -> List[float]:
    vals: List[float] = []
    if pd is not None:
        df = pd.read_csv(path)
        for v in df[col].tolist():
            fv = _parse_float(v)
            if fv is not None:
                vals.append(fv)
        return vals

    with path.open("r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            fv = _parse_float(row.get(col, ""))
            if fv is not None:
                vals.append(fv)
    return vals


def _load_results(path: Path) -> Tuple[List[float], List[float]]:
    if not path.exists():
        return [], []
    pd_vals = _load_cols_csv(path, "pd_distance")
    mt_vals = _load_cols_csv(path, "mt_distance")
    return pd_vals, mt_vals


def plot_all(pd_vals: List[float], mt_vals: List[float], outpath: Path) -> None:
    fig = plt.figure(figsize=(12, 8))

    ax1 = fig.add_subplot(221)
    ax1.set_title("PD bottleneck distances")
    if pd_vals:
        ax1.hist(pd_vals, bins=30)
    ax1.set_xlabel("PD distance")
    ax1.set_ylabel("count")
    ax1.grid(True, linestyle="--", linewidth=0.5, alpha=0.3)

    ax2 = fig.add_subplot(222)
    ax2.set_title("Merge tree distances")
    if mt_vals:
        ax2.hist(mt_vals, bins=30)
    ax2.set_xlabel("MT distance")
    ax2.set_ylabel("count")
    ax2.grid(True, linestyle="--", linewidth=0.5, alpha=0.3)

    ax3 = fig.add_subplot(223)
    ax3.set_title("PD vs MT (scatter)")
    if pd_vals and mt_vals:
        n = min(len(pd_vals), len(mt_vals))
        ax3.scatter(pd_vals[:n], mt_vals[:n], s=12)
    ax3.set_xlabel("PD distance")
    ax3.set_ylabel("MT distance")
    ax3.grid(True, linestyle="--", linewidth=0.5, alpha=0.3)

    ax4 = fig.add_subplot(224)
    ax4.axis("off")
    lines = [
        f"rows with PD: {len(pd_vals)}",
        f"rows with MT: {len(mt_vals)}",
        "",
        "If counts are low:",
        " • check phase_c_results.csv error column",
        " • verify merge tree ports 0/1 exist as .vtu",
    ]
    ax4.text(0.01, 0.99, "\n".join(lines), va="top")

    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def plot_summary(pd_vals: List[float], mt_vals: List[float], outpath: Path) -> None:
    fig = plt.figure(figsize=(8, 4.5))
    ax = fig.add_subplot(111)
    ax.set_title("Phase C — summary (REAL distances)")

    labels = []
    means = []

    if pd_vals:
        labels.append("PD")
        means.append(sum(pd_vals) / len(pd_vals))
    if mt_vals:
        labels.append("MT")
        means.append(sum(mt_vals) / len(mt_vals))

    x = list(range(len(labels)))
    ax.bar(x, means)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("mean distance")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.3)

    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", type=Path, required=True)
    args = ap.parse_args()

    results = args.indir / "phase_c_results.csv"
    pd_vals, mt_vals = _load_results(results)

    plot_all(pd_vals, mt_vals, args.indir / "phase_c_all_distances.png")
    plot_summary(pd_vals, mt_vals, args.indir / "phase_c_summary.png")

    print(f"Loaded PD={len(pd_vals)} MT={len(mt_vals)}")
    print(f"Wrote: {args.indir/'phase_c_all_distances.png'}")
    print(f"Wrote: {args.indir/'phase_c_summary.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
