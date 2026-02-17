#!/usr/bin/env python3
"""
phase_c_final_visualization_ttk.py

Visualization for the CSV produced by compute_composite_tree_distance_ttk.py.

It makes:
  - A quick summary PNG (means/medians + IQR) for PD and MT distances
  - A "full" PNG with histograms + PD-vs-MT scatter

Usage:
  python phase_c_final_visualization_ttk.py --csv distances_ttk.csv --outdir figs

Dependencies:
  - matplotlib (required)
  - pandas (optional; script falls back to csv module if pandas not installed)

"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception as e:
    raise SystemExit(f"ERROR: matplotlib is required for visualization: {e}")

# Optional pandas for convenience
try:
    import pandas as pd  # type: ignore
except Exception:
    pd = None


def _parse_float(x: str) -> Optional[float]:
    try:
        if x is None:
            return None
        x = str(x).strip()
        if not x:
            return None
        return float(x)
    except Exception:
        return None


def load_distances(csv_path: Path) -> Tuple[List[str], List[float], List[float], List[str]]:
    keys: List[str] = []
    pd_d: List[float] = []
    mt_d: List[float] = []
    errors: List[str] = []

    if pd is not None:
        df = pd.read_csv(csv_path)
        for _, r in df.iterrows():
            keys.append(str(r.get("key", "")))
            pdv = _parse_float(r.get("pd_distance", ""))
            mtv = _parse_float(r.get("mt_distance", ""))
            if pdv is not None:
                pd_d.append(pdv)
            if mtv is not None:
                mt_d.append(mtv)
            errors.append(str(r.get("error", "")))
        return keys, pd_d, mt_d, errors

    # Fallback: csv module
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            keys.append(r.get("key", "") or "")
            pdv = _parse_float(r.get("pd_distance", ""))
            mtv = _parse_float(r.get("mt_distance", ""))
            if pdv is not None:
                pd_d.append(pdv)
            if mtv is not None:
                mt_d.append(mtv)
            errors.append(r.get("error", "") or "")
    return keys, pd_d, mt_d, errors


def summary_stats(vals: List[float]) -> Dict[str, float]:
    if not vals:
        return {"n": 0}
    v = sorted(vals)
    n = len(v)
    mean = sum(v) / n
    median = v[n // 2] if n % 2 == 1 else 0.5 * (v[n // 2 - 1] + v[n // 2])
    q1 = v[int(0.25 * (n - 1))]
    q3 = v[int(0.75 * (n - 1))]
    stdev = statistics.pstdev(v) if n >= 2 else 0.0
    return {
        "n": float(n),
        "mean": mean,
        "median": median,
        "q1": q1,
        "q3": q3,
        "stdev": stdev,
        "min": v[0],
        "max": v[-1],
    }


def plot_summary(pd_vals: List[float], mt_vals: List[float], outpath: Path) -> None:
    s_pd = summary_stats(pd_vals)
    s_mt = summary_stats(mt_vals)

    fig = plt.figure(figsize=(9, 4.5))
    ax = fig.add_subplot(111)
    ax.set_title("TTK distances summary (PD + Merge Tree)")

    labels = []
    medians = []
    q1s = []
    q3s = []
    means = []

    if pd_vals:
        labels.append("PD")
        medians.append(s_pd["median"])
        q1s.append(s_pd["q1"])
        q3s.append(s_pd["q3"])
        means.append(s_pd["mean"])
    if mt_vals:
        labels.append("MergeTree")
        medians.append(s_mt["median"])
        q1s.append(s_mt["q1"])
        q3s.append(s_mt["q3"])
        means.append(s_mt["mean"])

    x = list(range(len(labels)))
    # IQR as error bars around median
    yerr_low = [m - q1 for m, q1 in zip(medians, q1s)]
    yerr_high = [q3 - m for m, q3 in zip(medians, q3s)]
    ax.errorbar(x, medians, yerr=[yerr_low, yerr_high], fmt="o", capsize=6)
    ax.plot(x, means, "x")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Distance")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)

    # annotate
    txt = []
    if pd_vals:
        txt.append(f"PD: n={int(s_pd['n'])}, mean={s_pd['mean']:.4g}, median={s_pd['median']:.4g}")
    if mt_vals:
        txt.append(f"MT: n={int(s_mt['n'])}, mean={s_mt['mean']:.4g}, median={s_mt['median']:.4g}")
    ax.text(0.02, 0.02, "\n".join(txt), transform=ax.transAxes, va="bottom")

    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def plot_full(pd_vals: List[float], mt_vals: List[float], outpath: Path) -> None:
    fig = plt.figure(figsize=(12, 8))

    ax1 = fig.add_subplot(221)
    ax1.set_title("PD distances histogram")
    if pd_vals:
        ax1.hist(pd_vals, bins=30)
    ax1.set_xlabel("PD distance")
    ax1.set_ylabel("count")
    ax1.grid(True, linestyle="--", linewidth=0.5, alpha=0.3)

    ax2 = fig.add_subplot(222)
    ax2.set_title("Merge tree distances histogram")
    if mt_vals:
        ax2.hist(mt_vals, bins=30)
    ax2.set_xlabel("Merge tree distance")
    ax2.set_ylabel("count")
    ax2.grid(True, linestyle="--", linewidth=0.5, alpha=0.3)

    ax3 = fig.add_subplot(223)
    ax3.set_title("PD vs Merge tree (scatter)")
    if pd_vals and mt_vals:
        # If lengths differ, just scatter on aligned ranks (simple visualization)
        n = min(len(pd_vals), len(mt_vals))
        ax3.scatter(pd_vals[:n], mt_vals[:n], s=12)
    ax3.set_xlabel("PD distance")
    ax3.set_ylabel("Merge tree distance")
    ax3.grid(True, linestyle="--", linewidth=0.5, alpha=0.3)

    ax4 = fig.add_subplot(224)
    ax4.set_title("Notes")
    lines = [
        "• Distances come from TTK distance filters (not proxies).",
        "• If you see many blanks, check the 'error' column in the CSV.",
        "• If MT distances are slow, try running compute script with --max first.",
    ]
    ax4.axis("off")
    ax4.text(0.01, 0.99, "\n".join(lines), va="top")

    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True, help="CSV produced by compute_composite_tree_distance_ttk.py")
    ap.add_argument("--outdir", type=Path, default=Path("figs_ttk"), help="Output directory for figures")
    args = ap.parse_args()

    keys, pd_vals, mt_vals, errors = load_distances(args.csv)

    args.outdir.mkdir(parents=True, exist_ok=True)
    plot_summary(pd_vals, mt_vals, args.outdir / "ttk_distance_summary.png")
    plot_full(pd_vals, mt_vals, args.outdir / "ttk_distance_all.png")

    n_err = sum(1 for e in errors if e.strip())
    print(f"Loaded rows={len(keys)}, pd_n={len(pd_vals)}, mt_n={len(mt_vals)}, error_rows={n_err}")
    print(f"Wrote: {args.outdir/'ttk_distance_summary.png'}")
    print(f"Wrote: {args.outdir/'ttk_distance_all.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
