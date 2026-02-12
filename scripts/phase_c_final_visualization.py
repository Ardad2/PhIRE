#!/usr/bin/env python3
"""
Phase C: Final visualization suite (multi-method + defensible proxy distances)

Key points (for reporting):
- PD "distances" here are **feature-count differences** (Δ#features), not bottleneck/Wasserstein.
- Merge-tree "distance" here is a **composite structural dissimilarity index (proxy)** built from
  real TTK-computed merge-tree statistics (nodes/arcs/branches) and a simple organization ratio.
- mt_composite_minmax is a within-run rank (depends on which samples/methods are included).
- mt_composite_rel is per-sample relative and is comparable across datasets.

Inputs expected:
1) ttk_outputs/phase_c_results.csv
   Columns used:
     method, dataset (GT/SR), sample, n_pd0, n_pd1, mt_nodes, mt_arcs, mt_branches
   (Other columns are allowed but ignored.)

2) ttk_outputs/direct/mt_composite_distances.csv and mt_composite_summary.csv
   (Produced by scripts/compute_composite_tree_distance.py)
   If these files are missing or incomplete for some methods, this script will
   compute mt_composite_rel + components directly from phase_c_results.csv.

Outputs:
- ttk_outputs/phase_c_final/phase_c_all_distances.png
- ttk_outputs/phase_c_final/phase_c_summary.png
- ttk_outputs/phase_c_final/phase_c_summary.txt
"""
from __future__ import annotations

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")  # helps inside Docker as non-root

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------- helpers ---------------------------------

def _pick_gt_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Return GT rows keyed by sample. If multiple GT rows exist, keep the first per sample."""
    gt = df[df["dataset"].str.upper() == "GT"].copy()
    if gt.empty:
        raise RuntimeError("No GT rows found (dataset == 'GT') in ttk_outputs/phase_c_results.csv")
    gt = gt.sort_values(["sample", "method"]).drop_duplicates(subset=["sample"], keep="first")
    return gt.set_index("sample")


def compute_pd_pairwise(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each method, compute per-sample PD feature-count differences vs GT:
      pd0_distance = |SR.n_pd0 - GT.n_pd0|
      pd1_distance = |SR.n_pd1 - GT.n_pd1|
    """
    gt = _pick_gt_rows(df)

    sr = df[df["dataset"].str.upper() == "SR"].copy()
    if sr.empty:
        raise RuntimeError("No SR rows found (dataset == 'SR') in ttk_outputs/phase_c_results.csv")

    rows = []
    for (method, sample), r in sr.set_index(["method", "sample"]).iterrows():
        if sample not in gt.index:
            continue
        g = gt.loc[sample]
        rows.append({
            "method": method,
            "sample": sample,
            "pd0_distance": float(abs(r["n_pd0"] - g["n_pd0"])),
            "pd1_distance": float(abs(r["n_pd1"] - g["n_pd1"])),
            # optional relative versions (safe when GT counts >0)
            "pd0_rel": float(abs(r["n_pd0"] - g["n_pd0"]) / g["n_pd0"]) if g["n_pd0"] else np.nan,
            "pd1_rel": float(abs(r["n_pd1"] - g["n_pd1"]) / g["n_pd1"]) if g["n_pd1"] else np.nan,
        })

    out = pd.DataFrame(rows).sort_values(["method", "sample"])
    return out


def compute_mt_proxy_from_results(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute MT structural proxy per sample/method vs GT from phase_c_results.csv.

    Components (absolute):
      node_distance  = |SR_nodes - GT_nodes|
      branch_distance= |SR_branches - GT_branches|
      arc_distance   = |SR_arcs - GT_arcs|
      ratio_distance = |(SR_branches/SR_nodes) - (GT_branches/GT_nodes)|

    Relative (per-sample, cross-dataset comparable):
      rel_nodes    = node_distance / GT_nodes
      rel_branches = branch_distance / GT_branches
      rel_arcs     = arc_distance / GT_arcs
      rel_ratio    = ratio_distance

    Composite relative proxy (defensible + interpretable):
      mt_composite_rel = mean(rel_nodes, rel_branches, rel_ratio)

    Note: arcs are reported but not included in mt_composite_rel (to avoid redundancy).
    """
    gt = _pick_gt_rows(df)
    sr = df[df["dataset"].str.upper() == "SR"].copy()

    rows = []
    for (method, sample), r in sr.set_index(["method", "sample"]).iterrows():
        if sample not in gt.index:
            continue
        g = gt.loc[sample]

        gt_nodes, sr_nodes = float(g["mt_nodes"]), float(r["mt_nodes"])
        gt_br, sr_br = float(g["mt_branches"]), float(r["mt_branches"])
        gt_arcs, sr_arcs = float(g["mt_arcs"]), float(r["mt_arcs"])

        node_distance = abs(sr_nodes - gt_nodes)
        branch_distance = abs(sr_br - gt_br)
        arc_distance = abs(sr_arcs - gt_arcs)

        gt_ratio = (gt_br / gt_nodes) if gt_nodes else 0.0
        sr_ratio = (sr_br / sr_nodes) if sr_nodes else 0.0
        ratio_distance = abs(sr_ratio - gt_ratio)

        rel_nodes = node_distance / gt_nodes if gt_nodes else np.nan
        rel_branches = branch_distance / gt_br if gt_br else np.nan
        rel_arcs = arc_distance / gt_arcs if gt_arcs else np.nan
        rel_ratio = ratio_distance

        # Composite proxy (relative, per-sample)
        mt_composite_rel = np.nanmean([rel_nodes, rel_branches, rel_ratio])

        rows.append({
            "method": method,
            "sample": sample,
            "node_distance": node_distance,
            "branch_distance": branch_distance,
            "arc_distance": arc_distance,
            "ratio_distance": ratio_distance,
            "rel_nodes": rel_nodes,
            "rel_branches": rel_branches,
            "rel_arcs": rel_arcs,
            "rel_ratio": rel_ratio,
            "mt_composite_rel": mt_composite_rel,
        })

    out = pd.DataFrame(rows).sort_values(["method", "sample"])
    return out


def summarize_by_method(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for method, g in df.groupby("method"):
        d = {"method": method}
        for c in cols:
            d[f"{c}_mean"] = float(np.nanmean(g[c].values))
            d[f"{c}_std"] = float(np.nanstd(g[c].values, ddof=1)) if g[c].notna().sum() > 1 else 0.0
        rows.append(d)
    return pd.DataFrame(rows).sort_values("method")


# ------------------------------ main -----------------------------------

def main():
    results_path = Path("ttk_outputs/phase_c_results.csv")
    if not results_path.exists():
        raise SystemExit(f"Missing: {results_path}")

    df = pd.read_csv(results_path)
    # normalize columns
    df["dataset"] = df["dataset"].astype(str)
    df["method"] = df["method"].astype(str)
    df["sample"] = df["sample"].astype(str)

    outdir = Path("ttk_outputs/phase_c_final")
    outdir.mkdir(parents=True, exist_ok=True)

    # --- PD pairwise distances vs GT (feature-count diffs) ---
    pd_pair = compute_pd_pairwise(df)
    pd_pair.to_csv(outdir / "pd_pairwise_distances.csv", index=False)

    pd_sum = summarize_by_method(pd_pair, ["pd0_distance", "pd1_distance"])
    pd_sum.to_csv(outdir / "pd_summary_by_method.csv", index=False)

    # --- MT proxy distances ---
    mt_csv = Path("ttk_outputs/direct/mt_composite_distances.csv")
    mt_sum_csv = Path("ttk_outputs/direct/mt_composite_summary.csv")

    if mt_csv.exists():
        mt_dist = pd.read_csv(mt_csv)
        # If the file uses 'method' as 'SR' but your actual method name is, e.g., 'GAN',
        # patch it when there is only one SR method in phase_c_results.
        sr_methods = sorted(df[df["dataset"].str.upper() == "SR"]["method"].unique().tolist())
        if "method" in mt_dist.columns and mt_dist["method"].nunique() == 1:
            only = str(mt_dist["method"].iloc[0])
            if only.upper() == "SR" and len(sr_methods) == 1:
                mt_dist["method"] = sr_methods[0]
    else:
        mt_dist = pd.DataFrame()

    # Fill/compute missing MT proxy rows directly from phase_c_results.csv
    mt_proxy = compute_mt_proxy_from_results(df)

    if mt_dist.empty:
        mt_dist = mt_proxy
    else:
        # ensure we have required columns; if not, prefer computed
        need = {"method", "sample", "node_distance", "branch_distance", "ratio_distance", "mt_composite_rel"}
        if not need.issubset(set(mt_dist.columns)):
            mt_dist = mt_proxy
        else:
            # merge: keep existing rows and append any missing (method,sample) pairs
            key = ["method", "sample"]
            have = set(map(tuple, mt_dist[key].values.tolist()))
            missing = mt_proxy[~mt_proxy[key].apply(tuple, axis=1).isin(have)]
            if not missing.empty:
                mt_dist = pd.concat([mt_dist, missing], ignore_index=True)

    # MT summary by method (use mt_composite_rel + key components)
    mt_cols = [c for c in ["node_distance", "branch_distance", "arc_distance", "ratio_distance",
                          "rel_nodes", "rel_branches", "rel_ratio", "mt_composite_rel"]
               if c in mt_dist.columns]
    mt_sum_by_method = summarize_by_method(mt_dist, mt_cols)
    mt_sum_by_method.to_csv(outdir / "mt_summary_by_method.csv", index=False)

    # --- Figure 1: Per-sample PD + MT distances (grouped by method) ---
    methods = sorted(pd_pair["method"].unique().tolist())
    samples = sorted(pd_pair["sample"].unique().tolist())

    # pivots
    pd0_piv = pd_pair.pivot(index="sample", columns="method", values="pd0_distance").reindex(samples)
    pd1_piv = pd_pair.pivot(index="sample", columns="method", values="pd1_distance").reindex(samples)

    mtc_piv = None
    if "mt_composite_rel" in mt_dist.columns:
        mtc_piv = mt_dist.pivot(index="sample", columns="method", values="mt_composite_rel").reindex(samples)

    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.25)

    # helper to make grouped bars
    def grouped_bar(ax, piv: pd.DataFrame, title: str, ylabel: str):
        x = np.arange(piv.shape[0])
        m = piv.shape[1]
        width = 0.8 / max(m, 1)
        for i, col in enumerate(piv.columns):
            ax.bar(x + (i - (m-1)/2)*width, piv[col].values, width=width, alpha=0.8, label=str(col))
        ax.set_xticks(x)
        ax.set_xticklabels(piv.index.tolist())
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Sample")
        ax.grid(axis="y", alpha=0.3)
        if m > 1:
            ax.legend(fontsize=9, ncol=min(3, m), frameon=False)

    ax1 = fig.add_subplot(gs[0, 0])
    grouped_bar(ax1, pd0_piv, "PD0 feature-count difference vs GT (components)", "Δ count")

    ax2 = fig.add_subplot(gs[0, 1])
    grouped_bar(ax2, pd1_piv, "PD1 feature-count difference vs GT (loops)", "Δ count")

    ax3 = fig.add_subplot(gs[1, :])
    if mtc_piv is not None and not mtc_piv.empty:
        grouped_bar(ax3, mtc_piv, "Merge-tree composite structural dissimilarity (relative proxy)", "Composite (rel)")
    else:
        ax3.axis("off")
        ax3.text(0.5, 0.5, "No MT composite proxy found.\nRun compute_composite_tree_distance.py first.",
                 ha="center", va="center", fontsize=12)

    fig.suptitle("Phase C: Topological Differences vs GT (feature-count PD + MT structural proxy)",
                 fontsize=16, fontweight="bold", y=0.99)

    fig_path = outdir / "phase_c_all_distances.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # --- Figure 2: Summary (mean ± std) across methods ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: PD1 mean±std + MT composite rel mean±std
    sum_methods = sorted(set(pd_sum["method"]).union(set(mt_sum_by_method["method"])))
    x = np.arange(len(sum_methods))

    pd1_mean = []
    pd1_std = []
    mt_mean = []
    mt_std = []

    pd_sum_idx = pd_sum.set_index("method")
    mt_sum_idx = mt_sum_by_method.set_index("method")

    for m in sum_methods:
        if m in pd_sum_idx.index:
            pd1_mean.append(pd_sum_idx.loc[m, "pd1_distance_mean"])
            pd1_std.append(pd_sum_idx.loc[m, "pd1_distance_std"])
        else:
            pd1_mean.append(np.nan); pd1_std.append(np.nan)

        if m in mt_sum_idx.index and "mt_composite_rel_mean" in mt_sum_idx.columns:
            mt_mean.append(mt_sum_idx.loc[m, "mt_composite_rel_mean"])
            mt_std.append(mt_sum_idx.loc[m, "mt_composite_rel_std"])
        else:
            mt_mean.append(np.nan); mt_std.append(np.nan)

    w = 0.38
    axes[0].bar(x - w/2, pd1_mean, width=w, yerr=pd1_std, capsize=4, alpha=0.8, label="PD1 Δcount")
    axes[0].bar(x + w/2, mt_mean, width=w, yerr=mt_std, capsize=4, alpha=0.8, label="MT composite (rel proxy)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(sum_methods, rotation=0)
    axes[0].set_title("Summary vs GT (mean ± std)", fontweight="bold")
    axes[0].set_ylabel("Value")
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].legend(frameon=False)

    # Right: MT components (relative) if present
    comp_cols = [c for c in ["rel_nodes_mean", "rel_branches_mean", "rel_ratio_mean"] if c in mt_sum_idx.columns]
    if comp_cols:
        # plot stacked-ish grouped
        rel_nodes = [mt_sum_idx.loc[m, "rel_nodes_mean"] if m in mt_sum_idx.index else np.nan for m in sum_methods]
        rel_br = [mt_sum_idx.loc[m, "rel_branches_mean"] if m in mt_sum_idx.index else np.nan for m in sum_methods]
        rel_ra = [mt_sum_idx.loc[m, "rel_ratio_mean"] if m in mt_sum_idx.index else np.nan for m in sum_methods]

        axes[1].bar(x - w, rel_nodes, width=w, alpha=0.8, label="rel_nodes")
        axes[1].bar(x,      rel_br,   width=w, alpha=0.8, label="rel_branches")
        axes[1].bar(x + w,  rel_ra,   width=w, alpha=0.8, label="rel_ratio")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(sum_methods, rotation=0)
        axes[1].set_title("MT proxy components (relative means)", fontweight="bold")
        axes[1].set_ylabel("Relative value")
        axes[1].grid(axis="y", alpha=0.3)
        axes[1].legend(frameon=False, ncol=3, fontsize=9)
    else:
        axes[1].axis("off")
        axes[1].text(0.5, 0.5, "MT relative components not available.", ha="center", va="center")

    sum_path = outdir / "phase_c_summary.png"
    plt.tight_layout()
    plt.savefig(sum_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # --- Summary text ---
    def fmt(mu, sd, digits=3):
        if np.isnan(mu):
            return "NA"
        return f"{mu:.{digits}f} ± {sd:.{digits}f}"

    txt_path = outdir / "phase_c_summary.txt"
    with open(txt_path, "w") as f:
        f.write("="*80 + "\n")
        f.write("PHASE C: TOPOLOGICAL DIFFERENCE ANALYSIS (DEFENSIBLE PROXIES)\n")
        f.write("="*80 + "\n\n")

        f.write("METHODS / DEFINITIONS:\n")
        f.write("- PD distances: feature-count differences vs GT (Δ#components, Δ#loops)\n")
        f.write("  (Not bottleneck/Wasserstein; no distributional-distance claim.)\n")
        f.write("- MT distance: composite structural dissimilarity index (proxy)\n")
        f.write("  built from real TTK merge-tree metrics (nodes/branches) + organization ratio.\n")
        f.write("  Report mt_composite_rel for cross-dataset comparability.\n\n")

        f.write("SUMMARY BY METHOD (mean ± std across samples):\n")
        f.write("-"*80 + "\n")
        for m in sum_methods:
            pd1_mu = pd_sum_idx.loc[m, "pd1_distance_mean"] if m in pd_sum_idx.index else np.nan
            pd1_sd = pd_sum_idx.loc[m, "pd1_distance_std"] if m in pd_sum_idx.index else np.nan
            mt_mu = mt_sum_idx.loc[m, "mt_composite_rel_mean"] if (m in mt_sum_idx.index and "mt_composite_rel_mean" in mt_sum_idx.columns) else np.nan
            mt_sd = mt_sum_idx.loc[m, "mt_composite_rel_std"] if (m in mt_sum_idx.index and "mt_composite_rel_std" in mt_sum_idx.columns) else np.nan

            f.write(f"{m}:\n")
            f.write(f"  PD1 Δcount: {fmt(pd1_mu, pd1_sd, digits=2)}\n")
            f.write(f"  MT composite_rel: {fmt(mt_mu, mt_sd, digits=3)}\n")

            if m in mt_sum_idx.index and "rel_nodes_mean" in mt_sum_idx.columns:
                f.write(f"  MT rel_nodes mean: {mt_sum_idx.loc[m,'rel_nodes_mean']:.3f}\n")
                f.write(f"  MT rel_branches mean: {mt_sum_idx.loc[m,'rel_branches_mean']:.3f}\n")
                f.write(f"  MT rel_ratio mean: {mt_sum_idx.loc[m,'rel_ratio_mean']:.6f}\n")
            f.write("\n")

        f.write("="*80 + "\n")
        f.write("NOTES:\n")
        f.write("="*80 + "\n")
        f.write("- mt_composite_minmax (if present) is a within-run rank; use mt_composite_rel for reporting.\n")
        f.write("- When you add BiCUBIC and CNN rows into phase_c_results.csv (dataset='SR'),\n")
        f.write("  re-run compute_composite_tree_distance.py, then re-run this script.\n")

    print(f"✓ Saved: {fig_path}")
    print(f"✓ Saved: {sum_path}")
    print(f"✓ Saved: {txt_path}")
    print(f"✓ Also saved: {outdir / 'pd_pairwise_distances.csv'} and {outdir / 'mt_summary_by_method.csv'}")


if __name__ == "__main__":
    main()
