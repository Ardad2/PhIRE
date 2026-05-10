#!/usr/bin/env python3
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


ROOT = Path(__file__).resolve().parents[1]
OUT  = ROOT / "ttk_runs_fixed" / "merge_tree_proxy_report_figures"
OUT.mkdir(parents=True, exist_ok=True)

BASE_METRICS = ROOT / "ttk_runs_fixed" / "baseline_metrics"
COMP         = ROOT / "ttk_runs_fixed" / "component_counts"
MTP          = ROOT / "ttk_runs_fixed" / "merge_tree_proxy"
VIS          = ROOT / "ttk_runs_fixed" / "baseline_visual_panels"


def savefig(name: str) -> None:
    for ext in ("png", "pdf"):
        path = OUT / f"{name}.{ext}"
        plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"wrote {OUT / (name + '.png')}")
    print(f"wrote {OUT / (name + '.pdf')}")


def norm_method(x: str) -> str:
    s = str(x).strip().lower()
    if s in {"bicubic", "bic"}: return "BICUBIC"
    if s == "cnn":               return "CNN"
    if s == "gan":               return "GAN"
    if s in {"tie", "ties"}:     return "Tie"
    return str(x).strip()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


# ── Figure 2: three-method metric winner summary ──────────────────────────────

def make_three_method_metric_summary() -> None:
    path = BASE_METRICS / "all_methods_winner_counts.csv"
    df   = read_csv(path)

    cols       = {c.lower(): c for c in df.columns}
    measure_col = cols.get("measure") or cols.get("metric") or df.columns[0]

    method_cols = []
    for target in ["bicubic", "cnn", "gan", "tie"]:
        for c in df.columns:
            if c.lower() == target or c.lower() == f"{target}_wins":
                method_cols.append(c)
                break

    if not method_cols:
        raise RuntimeError(
            f"Could not identify method-count columns in {path}. Columns={list(df.columns)}"
        )

    plot_df = df[[measure_col] + method_cols].copy().rename(columns={measure_col: "Measure"})
    label_map = {c: norm_method(c.replace("_wins", "")) for c in method_cols}
    plot_df = plot_df.rename(columns=label_map)
    methods = [m for m in ["BICUBIC", "CNN", "GAN", "Tie"] if m in plot_df.columns]

    y    = np.arange(len(plot_df))
    left = np.zeros(len(plot_df))

    plt.figure(figsize=(10.5, max(5.5, 0.35 * len(plot_df))))
    for m in methods:
        vals = pd.to_numeric(plot_df[m], errors="coerce").fillna(0).to_numpy()
        plt.barh(y, vals, left=left, label=m)
        left += vals

    plt.yticks(y, plot_df["Measure"])
    plt.xlabel("Winner count across 168 samples")
    plt.title("Three-method metric winner counts")
    plt.legend(loc="lower right")
    plt.gca().invert_yaxis()
    savefig("fig02_three_method_metric_summary")


# ── Figure 3: component-count threshold heatmap ───────────────────────────────

def make_component_threshold_heatmap() -> None:
    path = COMP / "component_count_winner_counts.csv"
    df   = read_csv(path)

    threshold_col = "threshold_label" if "threshold_label" in df.columns else df.columns[0]
    methods = [m for m in ["bicubic", "cnn", "gan", "tie"] if m in df.columns]
    if not methods:
        methods = [m for m in ["BICUBIC", "CNN", "GAN", "Tie"] if m in df.columns]

    data     = df[methods].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy()
    labels_y = df[threshold_col].astype(str).tolist()
    labels_x = [norm_method(m) for m in methods]

    plt.figure(figsize=(7.5, 4.5))
    im = plt.imshow(data, aspect="auto")
    plt.colorbar(im, label="Winner count")
    plt.xticks(np.arange(len(labels_x)), labels_x)
    plt.yticks(np.arange(len(labels_y)), labels_y)
    plt.title("Component-count winner counts by threshold")

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            plt.text(j, i, f"{int(data[i, j])}", ha="center", va="center", fontsize=9)

    savefig("fig03_component_count_threshold_heatmap")


# ── Figure 4: MT-GAN threshold-signature matrix ───────────────────────────────

def parse_signature(sig: str) -> dict[str, str]:
    out = {}
    for part in str(sig).replace(",", ";").split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = norm_method(v.strip())
    return out


def make_mt_gan_signature_matrix() -> None:
    path = COMP / "special_component_case_summary.csv"
    df   = read_csv(path)

    if "group_name" in df.columns:
        df = df[df["group_name"].astype(str).str.lower().eq("all_mt_gan")].copy()

    if "winner_signature" not in df.columns:
        raise RuntimeError(f"winner_signature column missing from {path}")

    df["sample_idx"] = pd.to_numeric(df["sample_idx"], errors="coerce").astype(int)
    df = df.sort_values("sample_idx")

    thresholds     = ["t5", "t10", "t15", "p90", "p95", "p99"]
    method_to_int  = {"BICUBIC": 0, "CNN": 1, "GAN": 2, "Tie": 3, "": 4}
    cmap = ListedColormap(["#440154", "#31688e", "#35b779", "#fde725", "#dddddd"])

    mat = [
        [method_to_int.get(parse_signature(row["winner_signature"]).get(t, ""), 4)
         for t in thresholds]
        for _, row in df.iterrows()
    ]
    mat = np.array(mat)

    plt.figure(figsize=(8.5, max(5, 0.28 * len(df))))
    plt.imshow(mat, aspect="auto", cmap=cmap, vmin=0, vmax=4)
    plt.xticks(np.arange(len(thresholds)), thresholds)
    plt.yticks(np.arange(len(df)), df["sample_idx"].astype(str))
    plt.xlabel("Threshold")
    plt.ylabel("MT-GAN sample")
    plt.title("Component-count winner signatures for MT-GAN cases")

    cbar = plt.colorbar(ticks=[0, 1, 2, 3, 4])
    cbar.ax.set_yticklabels(["Bicubic", "CNN", "GAN", "Tie", "Missing"])

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            plt.text(
                j, i, ["B","C","G","T","?"][mat[i,j]],
                ha="center", va="center", color="white", fontsize=8
            )

    savefig("fig04_mt_gan_signature_matrix")


# ── Figure 5: adjacent cluster component-count curves ─────────────────────────

def make_adjacent_cluster_curve(cluster: list[int], name: str) -> None:
    path = COMP / "special_component_case_thresholds.csv"
    df   = read_csv(path)
    df["sample_idx"] = pd.to_numeric(df["sample_idx"], errors="coerce").astype(int)

    thresholds = ["t5", "t10", "t15", "p90", "p95", "p99"]
    methods    = [
        ("gt_count",      "GT"),
        ("bicubic_count", "Bicubic"),
        ("cnn_count",     "CNN"),
        ("gan_count",     "GAN"),
    ]

    sub = df[df["sample_idx"].isin(cluster)].copy()
    if sub.empty:
        raise RuntimeError(f"No rows found for cluster {cluster} in {path}")

    fig, axes = plt.subplots(len(cluster), 1, figsize=(9.5, 2.25 * len(cluster)), sharex=True)
    if len(cluster) == 1:
        axes = [axes]

    for ax, sid in zip(axes, cluster):
        s = sub[sub["sample_idx"].eq(sid)].copy()
        s["threshold_label"] = pd.Categorical(s["threshold_label"], categories=thresholds, ordered=True)
        s = s.sort_values("threshold_label")

        title_sig = (
            "; ".join(f"{r['threshold_label']}={norm_method(r['winner'])}" for _, r in s.iterrows())
            if "winner" in s.columns else ""
        )

        for col, label in methods:
            if col in s.columns:
                ax.plot(
                    s["threshold_label"].astype(str),
                    pd.to_numeric(s[col], errors="coerce"),
                    marker="o", label=label,
                )

        ax.set_ylabel("Component count")
        ax.set_title(f"Sample {sid}: {title_sig}", fontsize=10)
        ax.grid(True, alpha=0.25)

    axes[0].legend(loc="best")
    axes[-1].set_xlabel("Threshold")
    fig.suptitle(f"Adjacent cluster {name}: component-count curves", y=1.01, fontsize=13)
    fig.tight_layout()
    savefig(f"fig05_adjacent_cluster_{name.replace('-', '_')}_component_curves")


# ── Figure 8: rare topology-CNN controls ──────────────────────────────────────

def make_rare_topology_cnn_controls() -> None:
    samples   = [161, 162, 163, 164]
    img_paths = []
    for sid in samples:
        p = VIS / "panels_crop" / f"sample_{sid:03d}_crop.png"
        if not p.exists():
            p = VIS / "panels_full" / f"sample_{sid:03d}_full.png"
        if not p.exists():
            raise FileNotFoundError(f"Missing visual panel for sample {sid}: {p}")
        img_paths.append(p)

    fig, axes = plt.subplots(2, 2, figsize=(13, 7))
    axes = axes.ravel()

    for ax, sid, p in zip(axes, samples, img_paths):
        ax.imshow(plt.imread(p))
        ax.axis("off")
        ax.set_title(f"Sample {sid}", fontsize=12)

    fig.suptitle("Rare topology-CNN controls: samples 161–164", fontsize=14)
    fig.tight_layout()
    savefig("fig08_rare_topology_cnn_controls_161_164")


# ── Figure 9: dense component-count curves ────────────────────────────────────

def make_dense_component_curves() -> None:
    path = MTP / "merge_tree_proxy_threshold_curves.csv"
    df   = read_csv(path)
    df["sample_idx"] = pd.to_numeric(df["sample_idx"], errors="coerce").astype(int)

    if "min_area" in df.columns:
        df = df[pd.to_numeric(df["min_area"], errors="coerce").eq(10)].copy()

    samples   = [12, 77, 92, 154, 162, 163]
    methods   = ["gt", "bicubic", "cnn", "gan"]
    label_map = {"gt": "GT", "bicubic": "Bicubic", "cnn": "CNN", "gan": "GAN"}

    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    axes = axes.ravel()

    for ax, sid in zip(axes, samples):
        s = df[df["sample_idx"].eq(sid)].copy()
        if s.empty:
            ax.set_title(f"Sample {sid}: missing")
            ax.axis("off")
            continue

        for m in methods:
            sm = s[s["method"].astype(str).str.lower().eq(m)].sort_values("threshold_percentile")
            if sm.empty:
                continue
            ax.plot(
                pd.to_numeric(sm["threshold_percentile"], errors="coerce"),
                pd.to_numeric(sm["component_count"],     errors="coerce"),
                label=label_map[m],
            )

        ax.set_title(f"Sample {sid}")
        ax.set_ylabel("Component count")
        ax.grid(True, alpha=0.25)

    axes[-2].set_xlabel("GT percentile threshold")
    axes[-1].set_xlabel("GT percentile threshold")
    axes[0].legend(loc="best")
    fig.suptitle("Dense area-filtered component-count curves, min_area=10", fontsize=14)
    fig.tight_layout()
    savefig("fig09_dense_component_curves_representative_samples")


# ── Figure 10: hierarchy-proxy summary ───────────────────────────────────────

def make_hierarchy_proxy_summary() -> None:
    path = MTP / "special_merge_tree_proxy_summary.csv"
    df   = read_csv(path)
    df["sample_idx"] = pd.to_numeric(df["sample_idx"], errors="coerce").astype(int)

    if "group_name" not in df.columns:
        df["group_name"] = "all"

    group_order = [
        "all_mt_gan",
        "gan_majority_but_mt_not_gan",
        "cluster_10_13",
        "cluster_76_78",
        "cluster_90_93",
        "cluster_161_164",
        "limitation_cases",
    ]

    rows = []
    for g in group_order:
        sub = df[df["group_name"].astype(str).eq(g)].copy()
        if sub.empty:
            continue

        for col in ["gt_branch_count", "cnn_branch_count", "gan_branch_count", "bic_branch_count"]:
            if col in sub.columns:
                sub[col] = pd.to_numeric(sub[col], errors="coerce")

        gan_ratio = (
            (sub["gan_branch_count"] / sub["gt_branch_count"]).replace([np.inf, -np.inf], np.nan)
            if {"gan_branch_count", "gt_branch_count"}.issubset(sub.columns)
            else pd.Series(dtype=float)
        )

        cnn_ratio = (
            (sub["cnn_branch_count"] / sub["gt_branch_count"]).replace([np.inf, -np.inf], np.nan)
            if {"cnn_branch_count", "gt_branch_count"}.issubset(sub.columns)
            else pd.Series(dtype=float)
        )

        gan_pers_bias = (
            pd.to_numeric(sub["gan_mean_pers"], errors="coerce")
            - pd.to_numeric(sub["gt_mean_pers"], errors="coerce")
            if {"gan_mean_pers", "gt_mean_pers"}.issubset(sub.columns)
            else pd.Series(dtype=float)
        )

        rows.append({
            "group":                 g,
            "n":                     len(sub),
            "gan_branch_ratio_mean": gan_ratio.mean(),
            "cnn_branch_ratio_mean": cnn_ratio.mean(),
            "gan_mean_pers_bias":    gan_pers_bias.mean(),
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "hierarchy_proxy_group_summary.csv", index=False)

    if out.empty:
        raise RuntimeError("No hierarchy proxy group rows found.")

    x      = np.arange(len(out))
    labels = out["group"].str.replace("_", "\n")

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    axes[0].bar(x - 0.18, out["cnn_branch_ratio_mean"], width=0.36, label="CNN / GT")
    axes[0].bar(x + 0.18, out["gan_branch_ratio_mean"], width=0.36, label="GAN / GT")
    axes[0].axhline(1.0, linestyle="--", linewidth=1)
    axes[0].set_ylabel("Mean branch-count ratio")
    axes[0].set_title("Branch-count ratio by case family")
    axes[0].legend()
    axes[0].grid(True, axis="y", alpha=0.25)

    axes[1].bar(x, out["gan_mean_pers_bias"])
    axes[1].axhline(0.0, linestyle="--", linewidth=1)
    axes[1].set_ylabel("GAN mean persistence − GT")
    axes[1].set_title("GAN mean-persistence bias by case family")
    axes[1].grid(True, axis="y", alpha=0.25)

    plt.xticks(x, labels, fontsize=8)
    fig.tight_layout()
    savefig("fig10_hierarchy_proxy_summary")


# ── LaTeX tables ─────────────────────────────────────────────────────────────

def make_summary_tables_tex() -> None:
    table_dir = OUT / "tables_tex"
    table_dir.mkdir(exist_ok=True)

    p = COMP / "component_count_summary.csv"
    if p.exists():
        tex = pd.read_csv(p).to_latex(index=False, escape=False, float_format=lambda x: f"{x:.2f}")
        (table_dir / "table_component_count_summary.tex").write_text(tex)

    p = MTP / "merge_tree_proxy_summary.csv"
    if p.exists():
        tex = pd.read_csv(p).to_latex(index=False, escape=False, float_format=lambda x: f"{x:.3f}")
        (table_dir / "table_merge_tree_proxy_summary.tex").write_text(tex)

    p = MTP / "special_merge_tree_proxy_summary.csv"
    if p.exists():
        df = pd.read_csv(p)
        if "group_name" in df.columns:
            df = df[df["group_name"].astype(str).eq("all_mt_gan")].copy()
        keep = [c for c in [
            "sample_idx",
            "gt_branch_count", "cnn_branch_count", "gan_branch_count",
            "gt_mean_pers",    "cnn_mean_pers",    "gan_mean_pers",
            "branch_count_winner", "mean_pers_winner",
        ] if c in df.columns]
        tex = df[keep].to_latex(index=False, escape=False, float_format=lambda x: f"{x:.3f}")
        (table_dir / "table_mt_gan_branch_proxy_summary.tex").write_text(tex)

    print(f"wrote LaTeX tables under {table_dir}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    make_three_method_metric_summary()
    make_component_threshold_heatmap()
    make_mt_gan_signature_matrix()
    make_adjacent_cluster_curve([10, 11, 12, 13], "10-13")
    make_adjacent_cluster_curve([76, 77, 78],     "76-78")
    make_adjacent_cluster_curve([90, 91, 92, 93], "90-93")
    make_rare_topology_cnn_controls()
    make_dense_component_curves()
    make_hierarchy_proxy_summary()
    make_summary_tables_tex()

    print(f"\nAll report figures written to:\n{OUT}")


if __name__ == "__main__":
    main()
