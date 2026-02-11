#!/usr/bin/env python3
"""
Compute a composite MERGE-TREE structural dissimilarity proxy from TTK-derived metrics.

Important framing:
- This is a proxy / index, not a canonical merge-tree Wasserstein/edit distance.
- We compute BOTH:
  (1) mt_composite_minmax: min-max normalized within the evaluated set (relative rank)
  (2) mt_composite_rel: per-sample relative composite (comparable across datasets)

Inputs:
- ttk_outputs/phase_c_results.csv with columns:
  sample, dataset, mt_nodes, mt_arcs, mt_branches
Where dataset includes 'GT' and one or more methods (e.g., 'SR', 'BICUBIC', 'CNN').

Outputs:
- ttk_outputs/direct/mt_composite_distances.csv
- ttk_outputs/direct/mt_composite_summary.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd

EPS = 1e-12

def minmax(values: np.ndarray) -> np.ndarray:
    vmin, vmax = np.min(values), np.max(values)
    if vmax <= vmin + EPS:
        return np.zeros_like(values, dtype=float)
    return (values - vmin) / (vmax - vmin)

def safe_div(a: float, b: float) -> float:
    return a / (b + EPS)

def compute_components(gt_row: pd.Series, m_row: pd.Series) -> dict:
    """Return absolute + relative components."""
    gt_nodes = float(gt_row["mt_nodes"])
    gt_br   = float(gt_row["mt_branches"])
    gt_arcs = float(gt_row["mt_arcs"])

    m_nodes = float(m_row["mt_nodes"])
    m_br    = float(m_row["mt_branches"])
    m_arcs  = float(m_row["mt_arcs"])

    # Absolute differences (scale-dependent)
    abs_nodes = abs(m_nodes - gt_nodes)
    abs_br    = abs(m_br - gt_br)
    abs_arcs  = abs(m_arcs - gt_arcs)

    # Unitless “shape/bushiness” difference
    gt_ratio = safe_div(gt_br, gt_nodes) if gt_nodes > 0 else 0.0
    m_ratio  = safe_div(m_br, m_nodes) if m_nodes > 0 else 0.0
    abs_ratio = abs(m_ratio - gt_ratio)

    # Relative (dataset-comparable) differences
    rel_nodes = safe_div(abs_nodes, gt_nodes) if gt_nodes > 0 else 0.0
    rel_br    = safe_div(abs_br, gt_br) if gt_br > 0 else 0.0
    rel_arcs  = safe_div(abs_arcs, gt_arcs) if gt_arcs > 0 else 0.0
    rel_ratio = abs_ratio  # already unitless

    return {
        "node_distance": abs_nodes,
        "branch_distance": abs_br,
        "arc_distance": abs_arcs,
        "ratio_distance": abs_ratio,
        "rel_nodes": rel_nodes,
        "rel_branches": rel_br,
        "rel_arcs": rel_arcs,
        "rel_ratio": rel_ratio,
    }

def main():
    print("=" * 80)
    print("COMPOSITE MERGE-TREE STRUCTURAL DISSIMILARITY (PROXY)")
    print("=" * 80)

    df = pd.read_csv("ttk_outputs/phase_c_results.csv")

    required = {"sample", "dataset", "mt_nodes", "mt_arcs", "mt_branches"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    methods = sorted([d for d in df["dataset"].unique() if d != "GT"])
    if not methods:
        raise ValueError("No methods found besides 'GT' in dataset column.")

    rows = []
    for sample in sorted(df["sample"].unique()):
        gt = df[(df["sample"] == sample) & (df["dataset"] == "GT")]
        if len(gt) != 1:
            continue
        gt_row = gt.iloc[0]

        for method in methods:
            m = df[(df["sample"] == sample) & (df["dataset"] == method)]
            if len(m) != 1:
                continue
            m_row = m.iloc[0]

            comps = compute_components(gt_row, m_row)
            rows.append({"sample": sample, "method": method, **comps})

    res = pd.DataFrame(rows)
    if res.empty:
        raise RuntimeError("No (sample, GT, method) pairs found to compare.")

    # Composite (min-max): relative rank within this evaluation set
    n_norm = minmax(res["node_distance"].to_numpy(dtype=float))
    b_norm = minmax(res["branch_distance"].to_numpy(dtype=float))
    r_norm = minmax(res["ratio_distance"].to_numpy(dtype=float))
    res["mt_composite_minmax"] = (n_norm + b_norm + r_norm) / 3.0

    # Composite (relative): comparable across datasets
    res["mt_composite_rel"] = (res["rel_nodes"] + res["rel_branches"] + res["rel_ratio"]) / 3.0

    outdir = Path("ttk_outputs/direct")
    outdir.mkdir(parents=True, exist_ok=True)

    out_csv = outdir / "mt_composite_distances.csv"
    res.to_csv(out_csv, index=False)
    print(f"✓ Saved: {out_csv}")

    # Summary by method
    summary_rows = []
    for method in methods:
        sub = res[res["method"] == method].copy()
        if sub.empty:
            continue
        summary_rows.append({
            "method": method,

            "node_distance_mean": sub["node_distance"].mean(),
            "node_distance_std": sub["node_distance"].std(),

            "branch_distance_mean": sub["branch_distance"].mean(),
            "branch_distance_std": sub["branch_distance"].std(),

            "ratio_distance_mean": sub["ratio_distance"].mean(),
            "ratio_distance_std": sub["ratio_distance"].std(),

            "mt_composite_minmax_mean": sub["mt_composite_minmax"].mean(),
            "mt_composite_minmax_std": sub["mt_composite_minmax"].std(),

            "mt_composite_rel_mean": sub["mt_composite_rel"].mean(),
            "mt_composite_rel_std": sub["mt_composite_rel"].std(),
        })

    sum_df = pd.DataFrame(summary_rows)
    out_sum = outdir / "mt_composite_summary.csv"
    sum_df.to_csv(out_sum, index=False)
    print(f"✓ Saved: {out_sum}")

    print("\nNotes for reporting:")
    print("- mt_composite_minmax is a within-run rank (depends on what samples/methods are included).")
    print("- mt_composite_rel is per-sample relative and is comparable across datasets.")

if __name__ == "__main__":
    main()
