#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


ROOT   = Path(__file__).resolve().parents[1]
FIG    = ROOT / "ttk_runs_fixed" / "merge_tree_proxy_report_figures"
TABLES = FIG / "tables_tex"
FIG.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)

BASE = ROOT / "ttk_runs_fixed" / "baseline_metrics"
COMP = ROOT / "ttk_runs_fixed" / "component_counts"
MTP  = ROOT / "ttk_runs_fixed" / "merge_tree_proxy"
VIS  = ROOT / "ttk_runs_fixed" / "baseline_visual_panels"


def savefig(name: str):
    for ext in ("png", "pdf"):
        out = FIG / f"{name}.{ext}"
        plt.savefig(out, dpi=220, bbox_inches="tight")
        print("wrote", out)
    plt.close()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    return pd.read_csv(path)


def human_metric_name(x: str) -> str:
    mapping = {
        "psnr":                 "PSNR",
        "ssim":                 "SSIM",
        "speed_mae":            "Speed MAE",
        "speed_rmse":           "Speed RMSE",
        "wpd_bias":             "WPD bias |·|",
        "wpd_mae":              "WPD MAE",
        "wpd_rmse":             "WPD RMSE",
        "wpd_w1":               "WPD Wasserstein-1",
        "psd_log_l2":           "PSD log-L2",
        "psd_slope_delta":      "PSD slope |Δ|",
        "grad_mae":             "Gradient MAE",
        "grad_w1":              "Gradient Wasserstein-1",
        "grad_kurtosis_delta":  "Gradient kurtosis |Δ|",
        "exceed_err_5ms":       "Exceedance error, s > 5",
        "exceed_err_10ms":      "Exceedance error, s > 10",
        "exceed_err_15ms":      "Exceedance error, s > 15",
        "exceed_err_20ms":      "Exceedance error, s > 20",
        "exceed_err_p90":       "Exceedance error, p90",
        "exceed_err_p95":       "Exceedance error, p95",
        "exceed_err_p99":       "Exceedance error, p99",
    }
    return mapping.get(str(x), str(x).replace("_", " "))


# ── Tables ────────────────────────────────────────────────────────────────────

def write_component_count_table():
    df   = read_csv(COMP / "component_count_summary.csv")
    keep = [c for c in [
        "threshold_label",
        "mean_gt_count",
        "mean_bicubic_abs_error",
        "mean_cnn_abs_error",
        "mean_gan_abs_error",
        "bicubic_wins", "cnn_wins", "gan_wins", "ties",
    ] if c in df.columns]
    df   = df[keep].rename(columns={
        "threshold_label":       "Threshold",
        "mean_gt_count":         "GT count",
        "mean_bicubic_abs_error":"Bicubic AE",
        "mean_cnn_abs_error":    "CNN AE",
        "mean_gan_abs_error":    "GAN AE",
        "bicubic_wins":          "Bicubic wins",
        "cnn_wins":              "CNN wins",
        "gan_wins":              "GAN wins",
        "ties":                  "Ties",
    })
    out = TABLES / "table_component_count_summary.tex"
    out.write_text(df.to_latex(index=False, escape=False, float_format=lambda v: f"{v:.2f}"))
    print("wrote", out)


def write_merge_tree_proxy_table():
    df        = read_csv(MTP / "merge_tree_proxy_summary.csv")
    preferred = [
        "method",
        "mean_afc_l1_a1",   "mean_afc_l1_a10",  "mean_afc_l1_a50",
        "mean_exceedance_l1",
        "mean_branch_count_ae",     "mean_branch_count_gt_1_ae",
        "mean_mean_persistence_ae", "mean_max_persistence_ae",
        "mean_total_persistence_ae","mean_top5_persistence_sum_ae",
        "mean_persistence_entropy_ae",
    ]
    keep = [c for c in preferred if c in df.columns]
    if len(keep) <= 1:
        keep = list(df.columns[: min(len(df.columns), 10)])
    df   = df[keep].rename(columns={
        "method":                       "Method",
        "mean_afc_l1_a1":               "AFC L1 a1",
        "mean_afc_l1_a10":              "AFC L1 a10",
        "mean_afc_l1_a50":              "AFC L1 a50",
        "mean_exceedance_l1":           "Exceedance L1",
        "mean_branch_count_ae":         "Branch count AE",
        "mean_branch_count_gt_1_ae":    "Branches >1 AE",
        "mean_mean_persistence_ae":     "Mean pers. AE",
        "mean_max_persistence_ae":      "Max pers. AE",
        "mean_total_persistence_ae":    "Total pers. AE",
        "mean_top5_persistence_sum_ae": "Top-5 pers. AE",
        "mean_persistence_entropy_ae":  "Pers. entropy AE",
    })
    out = TABLES / "table_merge_tree_proxy_summary.tex"
    out.write_text(df.to_latex(index=False, escape=False, float_format=lambda v: f"{v:.3f}"))
    print("wrote", out)


# ── Figure 2: three-method metric summary ─────────────────────────────────────

def regenerate_metric_summary():
    path = BASE / "all_methods_winner_counts.csv"
    if not path.exists():
        print("skip fig02: missing", path)
        return

    df   = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    measure_col = cols.get("measure") or cols.get("metric") or df.columns[0]

    method_cols = []
    for target in ["bicubic", "cnn", "gan", "tie"]:
        for c in df.columns:
            cl = c.lower()
            if cl == target or cl == f"{target}_wins":
                method_cols.append(c)
                break

    if not method_cols:
        print("skip fig02: could not identify method columns")
        return

    df = df[[measure_col] + method_cols].copy()
    df[measure_col] = df[measure_col].map(human_metric_name)
    rename = {}
    for c in method_cols:
        cl = c.lower().replace("_wins", "")
        rename[c] = {"bicubic": "Bicubic", "cnn": "CNN", "gan": "GAN"}.get(cl, "Tie")
    df = df.rename(columns=rename)

    methods = [m for m in ["Bicubic", "CNN", "GAN", "Tie"] if m in df.columns]
    y    = np.arange(len(df))
    left = np.zeros(len(df))

    plt.figure(figsize=(10.5, max(5.5, 0.38 * len(df))))
    for m in methods:
        vals = pd.to_numeric(df[m], errors="coerce").fillna(0).to_numpy()
        plt.barh(y, vals, left=left, label=m)
        left += vals

    plt.yticks(y, df[measure_col], fontsize=8)
    plt.xlabel("Winner count across 168 samples")
    plt.title("Three-method metric winner counts")
    plt.legend(loc="lower right")
    plt.gca().invert_yaxis()
    savefig("fig02_three_method_metric_summary")


# ── Figure 4: MT-GAN signature matrix ────────────────────────────────────────

def parse_signature(sig: str) -> dict[str, str]:
    out = {}
    for part in str(sig).replace(",", ";").split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = v.strip().upper()
    return out


def regenerate_mt_gan_signature_matrix():
    path = COMP / "special_component_case_summary.csv"
    if not path.exists():
        print("skip fig04: missing", path)
        return

    df = pd.read_csv(path)
    if "group_name" in df.columns:
        df = df[df["group_name"].astype(str).eq("all_mt_gan")].copy()
    if "winner_signature" not in df.columns:
        print("skip fig04: missing winner_signature")
        return

    df["sample_idx"] = pd.to_numeric(df["sample_idx"], errors="coerce").astype(int)
    df = df.drop_duplicates(subset=["sample_idx"]).sort_values("sample_idx")

    thresholds    = ["t5", "t10", "t15", "p90", "p95", "p99"]
    method_to_int = {"BICUBIC": 0, "BIC": 0, "CNN": 1, "GAN": 2, "TIE": 3, "": 4}
    cmap = ListedColormap(["#440154", "#31688e", "#35b779", "#fde725", "#dddddd"])

    mat = [
        [method_to_int.get(parse_signature(row["winner_signature"]).get(t, ""), 4)
         for t in thresholds]
        for _, row in df.iterrows()
    ]
    mat = np.asarray(mat)

    plt.figure(figsize=(8.5, max(5, 0.32 * len(df))))
    im = plt.imshow(mat, aspect="auto", cmap=cmap, vmin=0, vmax=4, interpolation="nearest")
    plt.xticks(np.arange(len(thresholds)), thresholds)
    plt.yticks(np.arange(len(df)), df["sample_idx"].astype(str))
    plt.xlabel("Threshold")
    plt.ylabel("MT-GAN sample")
    plt.title("Component-count winner signatures for MT-GAN cases")

    cbar = plt.colorbar(im, ticks=[0, 1, 2, 3, 4])
    cbar.ax.set_yticklabels(["Bicubic", "CNN", "GAN", "Tie", "Missing"])

    letter = {0: "B", 1: "C", 2: "G", 3: "T", 4: "?"}
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            plt.text(j, i, letter[int(mat[i, j])],
                     ha="center", va="center", color="white", fontsize=8)

    savefig("fig04_mt_gan_signature_matrix")


# ── Figure 5: adjacent cluster curves ────────────────────────────────────────

def regenerate_adjacent_cluster(cluster: list[int], name: str):
    path = COMP / "special_component_case_thresholds.csv"
    if not path.exists():
        print("skip cluster", name, ": missing", path)
        return

    df = pd.read_csv(path)
    df["sample_idx"] = pd.to_numeric(df["sample_idx"], errors="coerce").astype(int)
    sub = df[df["sample_idx"].isin(cluster)].copy()

    # Critical fix: deduplicate sample-threshold rows to eliminate repeated labels
    if "threshold_label" in sub.columns:
        sub = sub.drop_duplicates(subset=["sample_idx", "threshold_label"])

    thresholds = ["t5", "t10", "t15", "p90", "p95", "p99"]
    methods    = [
        ("gt_count",      "GT"),
        ("bicubic_count", "Bicubic"),
        ("cnn_count",     "CNN"),
        ("gan_count",     "GAN"),
    ]

    fig, axes = plt.subplots(len(cluster), 1, figsize=(9.5, 2.2 * len(cluster)), sharex=True)
    if len(cluster) == 1:
        axes = [axes]

    for ax, sid in zip(axes, cluster):
        s = sub[sub["sample_idx"].eq(sid)].copy()
        if s.empty:
            ax.set_title(f"Sample {sid}: missing")
            continue

        s["threshold_label"] = pd.Categorical(
            s["threshold_label"], categories=thresholds, ordered=True
        )
        s = s.sort_values("threshold_label")

        sig = (
            "; ".join(
                f"{r['threshold_label']}={str(r['winner']).upper()}"
                for _, r in s.iterrows()
            )
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
        ax.set_title(f"Sample {sid}: {sig}", fontsize=9)
        ax.grid(True, alpha=0.25)

    axes[0].legend(loc="best")
    axes[-1].set_xlabel("Threshold")
    fig.suptitle(f"Adjacent cluster {name}: component-count curves", y=1.01, fontsize=13)
    fig.tight_layout()
    savefig(f"fig05_adjacent_cluster_{name.replace('-', '_')}_component_curves")


# ── Figure 8: rare topology-CNN controls (full-width stacked) ─────────────────

def regenerate_rare_topology_controls():
    samples = [161, 162, 163, 164]
    paths   = []

    for sid in samples:
        p = VIS / "panels_crop" / f"sample_{sid:03d}_crop.png"
        if not p.exists():
            p = VIS / "panels_full" / f"sample_{sid:03d}_full.png"
        if not p.exists():
            print("missing rare-control panel:", p)
            return
        paths.append(p)

    # Full-width stacked layout — far more readable than 2×2 for wide panels
    fig, axes = plt.subplots(4, 1, figsize=(16, 10.5))
    for ax, sid, p in zip(axes, samples, paths):
        ax.imshow(plt.imread(p))
        ax.axis("off")
        ax.set_title(f"Sample {sid}", fontsize=11)

    fig.suptitle("Rare topology-CNN control neighborhood: samples 161–164", fontsize=15)
    fig.tight_layout()
    savefig("fig08_rare_topology_cnn_controls_161_164")


# ── Figure 10: hierarchy proxy summary ───────────────────────────────────────

def find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def regenerate_fig10_hierarchy_proxy_summary():
    path = MTP / "special_merge_tree_proxy_summary.csv"
    if not path.exists():
        print("skip fig10: missing", path)
        return

    df = pd.read_csv(path)
    if "sample_idx" not in df.columns:
        print("skip fig10: sample_idx missing")
        return

    df["sample_idx"] = pd.to_numeric(df["sample_idx"], errors="coerce").astype(int)

    gt_bc  = find_col(df, ["gt_branch_count",  "branch_count_gt"])
    cnn_bc = find_col(df, ["cnn_branch_count", "branch_count_cnn"])
    gan_bc = find_col(df, ["gan_branch_count", "branch_count_gan"])
    gt_mp  = find_col(df, ["gt_mean_persistence",  "gt_mean_pers",  "mean_persistence_gt",  "mean_pers_gt"])
    gan_mp = find_col(df, ["gan_mean_persistence", "gan_mean_pers", "mean_persistence_gan", "mean_pers_gan"])

    if not all([gt_bc, cnn_bc, gan_bc]):
        print("fig10 fallback: branch count columns not found.")
        print("Available columns:", list(df.columns))
        return

    groups = {
        "MT-GAN":     sorted(df[df["group_name"].astype(str).eq("all_mt_gan")]["sample_idx"].unique().tolist())
                      if "group_name" in df.columns else [],
        "10-13":      [10, 11, 12, 13],
        "76-78":      [76, 77, 78],
        "90-93":      [90, 91, 92, 93],
        "161-164":    [161, 162, 163, 164],
        "Limitations":[25, 80, 154],
    }

    if not groups["MT-GAN"] and "mt_winner" in df.columns:
        groups["MT-GAN"] = sorted(
            df[df["mt_winner"].astype(str).str.upper().eq("GAN")]["sample_idx"].unique().tolist()
        )

    rows = []
    for gname, samples in groups.items():
        sub = df[df["sample_idx"].isin(samples)].drop_duplicates(subset=["sample_idx"]).copy()
        if sub.empty:
            continue

        gt  = pd.to_numeric(sub[gt_bc],  errors="coerce").replace(0, np.nan)
        cnn = pd.to_numeric(sub[cnn_bc], errors="coerce")
        gan = pd.to_numeric(sub[gan_bc], errors="coerce")

        row = {
            "group":             gname,
            "n":                 len(sub),
            "cnn_branch_ratio":  (cnn / gt).replace([np.inf, -np.inf], np.nan).mean(),
            "gan_branch_ratio":  (gan / gt).replace([np.inf, -np.inf], np.nan).mean(),
            "gan_mean_persistence_bias": (
                pd.to_numeric(sub[gan_mp], errors="coerce") -
                pd.to_numeric(sub[gt_mp],  errors="coerce")
            ).mean() if (gt_mp and gan_mp) else np.nan,
        }
        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(FIG / "hierarchy_proxy_group_summary.csv", index=False)
    print("wrote", FIG / "hierarchy_proxy_group_summary.csv")

    if out.empty:
        print("skip fig10: no group rows")
        return

    x      = np.arange(len(out))
    labels = out["group"].astype(str)

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.5), sharex=True)

    axes[0].bar(x - 0.18, out["cnn_branch_ratio"], width=0.36, label="CNN / GT")
    axes[0].bar(x + 0.18, out["gan_branch_ratio"], width=0.36, label="GAN / GT")
    axes[0].axhline(1.0, linestyle="--", linewidth=1)
    axes[0].set_ylabel("Mean branch-count ratio")
    axes[0].set_title("Branch-count ratio by case family")
    axes[0].legend()
    axes[0].grid(True, axis="y", alpha=0.25)

    axes[1].bar(x, out["gan_mean_persistence_bias"])
    axes[1].axhline(0.0, linestyle="--", linewidth=1)
    axes[1].set_ylabel("GAN mean persistence − GT")
    axes[1].set_title("GAN mean-persistence bias by case family")
    axes[1].grid(True, axis="y", alpha=0.25)

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=20, ha="right")

    fig.tight_layout()
    savefig("fig10_hierarchy_proxy_summary")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    write_component_count_table()
    write_merge_tree_proxy_table()
    regenerate_metric_summary()
    regenerate_mt_gan_signature_matrix()
    regenerate_adjacent_cluster([10, 11, 12, 13], "10-13")
    regenerate_adjacent_cluster([76, 77, 78],     "76-78")
    regenerate_adjacent_cluster([90, 91, 92, 93], "90-93")
    regenerate_rare_topology_controls()
    regenerate_fig10_hierarchy_proxy_summary()

    print("\nRepair complete.")
    print("Figures:", FIG)
    print("Tables :", TABLES)


if __name__ == "__main__":
    main()
