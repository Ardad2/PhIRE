#!/usr/bin/env python3
"""
Generate figures and LaTeX tables for the component/hierarchy report.

Run:
  cd ~/PhIRE
  PYTHONNOUSERSITE=1 /usr/bin/python3 reports/component_hierarchy_report/scripts/generate_report_assets.py
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy import ndimage
except Exception:
    ndimage = None


ROOT = Path(__file__).resolve().parents[3]
REPORT_DIR = ROOT / "reports" / "component_hierarchy_report"
FIG_DIR = REPORT_DIR / "figures"
TAB_DIR = REPORT_DIR / "tables"

FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

BASELINE_METRICS = ROOT / "ttk_runs_fixed" / "baseline_metrics"
COMP_DIR = ROOT / "ttk_runs_fixed" / "component_counts"
VIS_DIR = ROOT / "ttk_runs_fixed" / "baseline_visual_panels"

ALL_METHODS_PER_SAMPLE = BASELINE_METRICS / "all_methods_per_sample.csv"
ALL_METHODS_SUMMARY = BASELINE_METRICS / "all_methods_summary.csv"
ALL_METHODS_WINNERS = BASELINE_METRICS / "all_methods_winner_counts.csv"

COMP_SUMMARY = COMP_DIR / "component_count_summary.csv"
COMP_WINNERS = COMP_DIR / "component_count_winner_counts.csv"
COMP_PER_SAMPLE = COMP_DIR / "component_counts_per_sample.csv"
SPECIAL_SUMMARY = COMP_DIR / "special_component_case_summary.csv"
SPECIAL_THRESHOLDS = COMP_DIR / "special_component_case_thresholds.csv"

METHODS = ["bicubic", "cnn", "gan"]
THRESHOLDS = ["t5", "t10", "t15", "p90", "p95", "p99"]

SIGNED_ABS_METRICS = {
    "wpd_bias",
    "psd_slope_delta",
    "psd_slope_abs_delta",
    "grad_kurtosis_delta",
    "grad_kurtosis_abs_delta",
    "exceed_frac_delta_t5",
    "exceed_frac_delta_t10",
    "exceed_frac_delta_t15",
    "exceed_frac_delta_p90",
    "exceed_frac_delta_p95",
    "exceed_frac_delta_p99",
    "exceed_frac_abs_delta_t5",
    "exceed_frac_abs_delta_t10",
    "exceed_frac_abs_delta_t15",
    "exceed_frac_abs_delta_p90",
    "exceed_frac_abs_delta_p95",
    "exceed_frac_abs_delta_p99",
}

HIGHER_IS_BETTER = {"psnr", "ssim"}


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return path


def savefig(name: str) -> None:
    out = FIG_DIR / name
    plt.tight_layout()
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out}")


def clean_metric_name(metric: str) -> str:
    return (
        metric.replace("_", " ")
        .replace("wpd", "WPD")
        .replace("psd", "PSD")
        .replace("mae", "MAE")
        .replace("rmse", "RMSE")
        .replace("ssim", "SSIM")
        .replace("psnr", "PSNR")
    )


def load_all_method_long() -> pd.DataFrame:
    df = pd.read_csv(require(ALL_METHODS_PER_SAMPLE))
    df.columns = [c.strip() for c in df.columns]
    if "method" not in df.columns:
        raise RuntimeError("all_methods_per_sample.csv must contain a 'method' column.")
    df["method"] = df["method"].astype(str).str.lower()
    return df


# ---------------------------------------------------------------------
# Figure 1: three-method metric winner summary
# ---------------------------------------------------------------------

def make_three_method_metric_summary() -> None:
    df = load_all_method_long()

    exclude = {
        "sample_idx", "sample_id", "sample", "idx", "method",
        "time", "timestamp", "date",
    }

    metric_cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            metric_cols.append(c)
        else:
            try:
                pd.to_numeric(df[c])
                metric_cols.append(c)
            except Exception:
                pass

    preferred_order = [
        "psnr", "ssim", "speed_mae", "speed_rmse",
        "wpd_bias", "wpd_mae", "wpd_rmse", "wpd_w1",
        "psd_log_l2", "psd_slope_abs_delta",
        "grad_mae", "grad_w1", "grad_kurtosis_abs_delta",
        "exceed_frac_abs_delta_t5", "exceed_frac_abs_delta_t10",
        "exceed_frac_abs_delta_t15", "exceed_frac_abs_delta_p90",
        "exceed_frac_abs_delta_p95", "exceed_frac_abs_delta_p99",
    ]
    metric_cols = [m for m in preferred_order if m in metric_cols]

    rows = []
    for metric in metric_cols:
        wide = df.pivot(index="sample_idx", columns="method", values=metric)
        wide = wide[[m for m in METHODS if m in wide.columns]].apply(pd.to_numeric, errors="coerce")
        if len(wide.columns) < 2:
            continue

        if metric in SIGNED_ABS_METRICS or metric.endswith("_abs_delta"):
            wide = wide.abs()

        counts = {m: 0 for m in METHODS}
        ties = 0

        for _, r in wide.iterrows():
            vals = r.dropna()
            if vals.empty:
                continue
            if metric in HIGHER_IS_BETTER:
                best_val = vals.max()
            else:
                best_val = vals.min()
            winners = vals.index[vals == best_val].tolist()
            if len(winners) == 1:
                counts[winners[0]] += 1
            else:
                ties += 1

        rows.append({"metric": metric, **counts, "tie": ties})

    out_csv = TAB_DIR / "three_method_metric_winner_counts.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")

    plot_df = pd.DataFrame(rows)
    labels = [clean_metric_name(x) for x in plot_df["metric"]]
    y = np.arange(len(plot_df))

    fig_h = max(4, 0.35 * len(plot_df))
    plt.figure(figsize=(9.5, fig_h))

    left = np.zeros(len(plot_df))
    for m in ["bicubic", "cnn", "gan", "tie"]:
        if m in plot_df.columns:
            vals = plot_df[m].values
            plt.barh(y, vals, left=left, label=m.upper() if m != "tie" else "Tie")
            left += vals

    plt.yticks(y, labels, fontsize=8)
    plt.xlabel("Winner count across 168 samples")
    plt.title("Three-method metric winner counts")
    plt.legend(loc="lower right")
    plt.gca().invert_yaxis()
    savefig("three_method_metric_summary.png")


# ---------------------------------------------------------------------
# Figure 2: component-count threshold heatmap
# ---------------------------------------------------------------------

def make_component_threshold_heatmap() -> None:
    winners = pd.read_csv(require(COMP_WINNERS))
    winners["threshold_label"] = pd.Categorical(
        winners["threshold_label"], categories=THRESHOLDS, ordered=True
    )
    winners = winners.sort_values("threshold_label")

    cols = [c for c in ["bicubic", "cnn", "gan", "tie"] if c in winners.columns]
    mat = winners[cols].to_numpy(dtype=float)

    plt.figure(figsize=(7.5, 3.8))
    im = plt.imshow(mat, aspect="auto")
    plt.colorbar(im, label="Winner count")
    plt.xticks(np.arange(len(cols)), [c.upper() if c != "tie" else "Tie" for c in cols])
    plt.yticks(np.arange(len(winners)), winners["threshold_label"].astype(str))
    plt.title("Component-count winner counts by threshold")

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            plt.text(j, i, f"{int(mat[i, j])}", ha="center", va="center", fontsize=9)

    savefig("component_threshold_heatmap.png")


# ---------------------------------------------------------------------
# Figure 3: MT-GAN threshold-signature matrix
# ---------------------------------------------------------------------

def parse_signature(sig: str) -> list[str]:
    parts = {}
    for token in str(sig).replace(",", ";").split(";"):
        if "=" not in token:
            continue
        k, v = token.split("=", 1)
        parts[k.strip()] = v.strip().upper()
    return [parts.get(t, "?").upper() for t in THRESHOLDS]


def make_mt_gan_signature_matrix() -> None:
    df = pd.read_csv(require(SPECIAL_SUMMARY))
    mtgan = df[df["group_name"] == "all_mt_gan"].copy()
    mtgan = mtgan.drop_duplicates("sample_idx").sort_values("sample_idx")

    if mtgan.empty:
        print("No all_mt_gan rows found; skipping MT-GAN signature matrix.")
        return

    code = {"BICUBIC": 0, "CNN": 1, "GAN": 2, "TIE": 3, "?": 4, "": 4}
    names = ["Bicubic", "CNN", "GAN", "Tie", "Missing"]

    matrix = []
    for _, row in mtgan.iterrows():
        winners = parse_signature(row["winner_signature"])
        matrix.append([code.get(w, 4) for w in winners])

    matrix = np.array(matrix)
    sample_labels = mtgan["sample_idx"].astype(int).astype(str).tolist()

    plt.figure(figsize=(7.5, max(4, 0.28 * len(sample_labels))))
    im = plt.imshow(matrix, aspect="auto", vmin=0, vmax=4)
    cbar = plt.colorbar(im, ticks=list(range(len(names))))
    cbar.ax.set_yticklabels(names)

    plt.xticks(np.arange(len(THRESHOLDS)), THRESHOLDS)
    plt.yticks(np.arange(len(sample_labels)), sample_labels, fontsize=8)
    plt.xlabel("Threshold")
    plt.ylabel("MT-GAN sample")
    plt.title("Component-count winner signatures for MT-GAN cases")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            plt.text(j, i, names[matrix[i, j]][0], ha="center", va="center", fontsize=7)

    savefig("mt_gan_signature_matrix.png")


# ---------------------------------------------------------------------
# Figures 4-6: adjacent cluster component-count curves
# ---------------------------------------------------------------------

def make_adjacent_cluster_curves() -> None:
    comp = pd.read_csv(require(COMP_PER_SAMPLE))
    clusters = {
        "10_13": [10, 11, 12, 13],
        "76_78": [76, 77, 78],
        "90_93": [90, 91, 92, 93],
    }

    count_cols = ["gt_count", "bicubic_count", "cnn_count", "gan_count"]
    display = {
        "gt_count": "GT",
        "bicubic_count": "Bicubic",
        "cnn_count": "CNN",
        "gan_count": "GAN",
    }

    for name, samples in clusters.items():
        fig, axes = plt.subplots(len(samples), 1, figsize=(8, 2.4 * len(samples)), sharex=True)
        if len(samples) == 1:
            axes = [axes]

        for ax, sid in zip(axes, samples):
            sdf = comp[comp["sample_idx"] == sid].copy()
            sdf["threshold_label"] = pd.Categorical(
                sdf["threshold_label"], categories=THRESHOLDS, ordered=True
            )
            sdf = sdf.sort_values("threshold_label")

            x = np.arange(len(sdf))
            for col in count_cols:
                ax.plot(x, sdf[col].astype(float), marker="o", label=display[col])

            winners = "; ".join(
                f"{r.threshold_label}={str(r.winner).upper()}"
                for _, r in sdf.iterrows()
            )
            ax.set_title(f"Sample {sid}: {winners}", fontsize=9)
            ax.set_ylabel("Component count")
            ax.grid(True, alpha=0.3)

        axes[-1].set_xticks(np.arange(len(THRESHOLDS)))
        axes[-1].set_xticklabels(THRESHOLDS)
        axes[0].legend(loc="upper right", ncol=4, fontsize=8)
        fig.suptitle(f"Adjacent cluster {name.replace('_', '–')}: component-count curves", y=1.01)
        savefig(f"adjacent_component_curves_{name}.png")


# ---------------------------------------------------------------------
# Figure 7: rare topology-CNN controls montage
# ---------------------------------------------------------------------

def find_panel(sid: int) -> Path | None:
    for sub in ["panels_crop", "panels_full"]:
        d = VIS_DIR / sub
        if not d.exists():
            continue
        hits = sorted(d.glob(f"*{sid:03d}*.png"))
        if hits:
            return hits[0]
    return None


def make_panel_montages() -> None:
    montages = {
        "topology_cnn_controls.png": [161, 162, 163, 164],
        "representative_case_montage.png": [6, 12, 62, 77, 92, 154, 162, 163],
    }

    for out_name, samples in montages.items():
        paths = [(sid, find_panel(sid)) for sid in samples]
        paths = [(sid, p) for sid, p in paths if p is not None]
        if not paths:
            print(f"No panels found for {out_name}; skipping.")
            continue

        ncols = 2
        nrows = math.ceil(len(paths) / ncols)
        plt.figure(figsize=(12, 4.2 * nrows))

        for idx, (sid, path) in enumerate(paths, 1):
            img = plt.imread(path)
            ax = plt.subplot(nrows, ncols, idx)
            ax.imshow(img)
            ax.set_title(f"Sample {sid}", fontsize=10)
            ax.axis("off")

        savefig(out_name)


# ---------------------------------------------------------------------
# Dense component-count curves
# ---------------------------------------------------------------------

def speed(a: np.ndarray) -> np.ndarray:
    return np.sqrt(a[..., 0].astype(np.float32) ** 2 + a[..., 1].astype(np.float32) ** 2)


def count_components(mask: np.ndarray, min_area: int = 0) -> int:
    if ndimage is None:
        raise RuntimeError("scipy.ndimage is required for dense component curves.")
    structure = np.ones((3, 3), dtype=int)
    labels, n = ndimage.label(mask, structure=structure)
    if min_area <= 1 or n == 0:
        return int(n)
    sizes = np.bincount(labels.ravel())
    return int(np.sum(sizes[1:] >= min_area))


def make_dense_component_curves() -> None:
    if ndimage is None:
        print("scipy.ndimage not available; skipping dense component curves.")
        return

    dirs = {
        "bicubic": ROOT / "data_out_fixed" / "wind_mrhr_bicubic",
        "cnn": ROOT / "data_out_fixed" / "wind_mrhr_cnn",
        "gan": ROOT / "data_out_fixed" / "wind_mrhr_gan",
    }

    for d in dirs.values():
        if not d.exists():
            print("Missing data_out_fixed arrays; skipping dense component curves.")
            return

    idx = np.load(dirs["cnn"] / "idx.npy")
    pos = {int(v): i for i, v in enumerate(idx.tolist())}

    gt = np.load(dirs["cnn"] / "dataGT.npy", mmap_mode="r")
    sr = {
        "bicubic": np.load(dirs["bicubic"] / "dataSR.npy", mmap_mode="r"),
        "cnn": np.load(dirs["cnn"] / "dataSR.npy", mmap_mode="r"),
        "gan": np.load(dirs["gan"] / "dataSR.npy", mmap_mode="r"),
    }

    samples = [6, 12, 77, 92, 154, 162, 163]
    min_area = 25
    rows = []

    for sid in samples:
        if sid not in pos:
            continue
        i = pos[sid]
        gt_s = speed(np.asarray(gt[i]))
        lo = float(np.percentile(gt_s, 1))
        hi = float(np.percentile(gt_s, 99))
        thresholds = np.linspace(lo, hi, 40)

        fields = {"gt": gt_s}
        for method in METHODS:
            fields[method] = speed(np.asarray(sr[method][i]))

        for t in thresholds:
            for method, field in fields.items():
                rows.append({
                    "sample_idx": sid,
                    "threshold": t,
                    "method": method,
                    "component_count": count_components(field >= t, min_area=min_area),
                    "min_area": min_area,
                })

    curve_df = pd.DataFrame(rows)
    out_csv = TAB_DIR / "dense_component_curves_minarea25.csv"
    curve_df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")

    # Plot representative dense curves
    fig, axes = plt.subplots(len(samples), 1, figsize=(8, 2.2 * len(samples)), sharex=False)
    if len(samples) == 1:
        axes = [axes]

    for ax, sid in zip(axes, samples):
        sdf = curve_df[curve_df["sample_idx"] == sid]
        if sdf.empty:
            continue
        for method in ["gt", "bicubic", "cnn", "gan"]:
            mdf = sdf[sdf["method"] == method]
            ax.plot(mdf["threshold"], mdf["component_count"], label=method.upper())
        ax.set_title(f"Sample {sid} dense component curve; min component area = {min_area} px")
        ax.set_ylabel("# components")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Speed threshold")
    axes[0].legend(loc="upper right", ncol=4, fontsize=8)
    savefig("dense_component_curves.png")


# ---------------------------------------------------------------------
# Hierarchy-proxy summary from current component features
# ---------------------------------------------------------------------

def make_hierarchy_proxy_summary() -> None:
    df = pd.read_csv(require(SPECIAL_SUMMARY))

    groups = []
    for name, sub in [
        ("MT-GAN", df[df["group_name"] == "all_mt_gan"]),
        ("GAN-majority\nMT-CNN", df[df["group_name"] == "gan_majority_but_mt_not_gan"]),
        ("PD=MT=CNN\ncontrols", df[
            (df.get("pd_winner", "") == "CNN") &
            (df.get("mt_winner", "") == "CNN")
        ]),
        ("Limitation\ncases", df[df["group_name"] == "limitation_cases"]),
    ]:
        if sub.empty:
            continue
        sub = sub.drop_duplicates("sample_idx").copy()
        sub["plot_group"] = name
        groups.append(sub)

    if not groups:
        print("No groups available for hierarchy proxy summary.")
        return

    plot_df = pd.concat(groups, ignore_index=True)
    plot_df["gan_mean_bias"] = pd.to_numeric(plot_df["gan_mean_bias"], errors="coerce")
    plot_df["gan_wins"] = pd.to_numeric(plot_df["gan_wins"], errors="coerce")

    labels = []
    data_bias = []
    data_wins = []
    for name, sub in plot_df.groupby("plot_group", sort=False):
        labels.append(name)
        data_bias.append(sub["gan_mean_bias"].dropna().values)
        data_wins.append(sub["gan_wins"].dropna().values)

    plt.figure(figsize=(8, 4.2))
    plt.boxplot(data_bias, labels=labels, showmeans=True)
    plt.ylabel("Mean GAN component-count bias\n(GAN count - GT count)")
    plt.title("Current hierarchy proxy: GAN over-fragmentation by case family")
    plt.grid(True, axis="y", alpha=0.3)
    savefig("hierarchy_proxy_summary.png")

    plt.figure(figsize=(8, 4.2))
    plt.boxplot(data_wins, labels=labels, showmeans=True)
    plt.ylabel("GAN component-count wins out of 6 thresholds")
    plt.title("Current hierarchy proxy: GAN threshold wins by case family")
    plt.grid(True, axis="y", alpha=0.3)
    savefig("hierarchy_proxy_gan_wins.png")


# ---------------------------------------------------------------------
# LaTeX tables
# ---------------------------------------------------------------------

def make_latex_tables() -> None:
    # Component summary table
    comp = pd.read_csv(require(COMP_SUMMARY))
    comp_tex = comp.to_latex(index=False, escape=False, float_format="%.2f")
    (TAB_DIR / "component_count_summary.tex").write_text(comp_tex)
    print(f"Wrote {TAB_DIR / 'component_count_summary.tex'}")

    # Special MT-GAN summary table
    sp = pd.read_csv(require(SPECIAL_SUMMARY))
    mtgan = sp[sp["group_name"] == "all_mt_gan"].drop_duplicates("sample_idx").copy()
    keep = [
        "sample_idx", "pd_winner", "mt_winner",
        "direct_error_group_winner", "distributional_group_winner",
        "tail_group_winner", "configured_physics_group_winner",
        "bicubic_wins", "cnn_wins", "gan_wins",
        "gan_wins_low", "gan_wins_high",
        "winner_signature", "interpretation_label",
        "phys_bicubic_wins", "phys_cnn_wins", "phys_gan_wins", "phys_winner",
    ]
    keep = [c for c in keep if c in mtgan.columns]
    mtgan_tex = mtgan[keep].to_latex(index=False, escape=False)
    (TAB_DIR / "mt_gan_case_summary.tex").write_text(mtgan_tex)
    print(f"Wrote {TAB_DIR / 'mt_gan_case_summary.tex'}")

    # Adjacent cluster summary
    adjacent_ids = [10, 11, 12, 13, 76, 77, 78, 90, 91, 92, 93]
    adj = sp[sp["sample_idx"].isin(adjacent_ids)].drop_duplicates(["group_name", "sample_idx"])
    keep2 = [
        "group_name", "sample_idx", "pd_winner", "mt_winner",
        "winner_signature", "interpretation_label",
        "phys_bicubic_wins", "phys_cnn_wins", "phys_gan_wins", "phys_winner",
    ]
    keep2 = [c for c in keep2 if c in adj.columns]
    adj_tex = adj[keep2].to_latex(index=False, escape=False)
    (TAB_DIR / "adjacent_cluster_summary.tex").write_text(adj_tex)
    print(f"Wrote {TAB_DIR / 'adjacent_cluster_summary.tex'}")

    # Interpretation label distribution
    label_counts = (
        sp[~sp["group_name"].isin(["all_mt_gan", "gan_majority_but_mt_not_gan"])]
        .drop_duplicates("sample_idx")["interpretation_label"]
        .value_counts()
        .rename_axis("interpretation_label")
        .reset_index(name="count")
    )
    label_counts.to_csv(TAB_DIR / "curated_label_distribution.csv", index=False)
    label_tex = label_counts.to_latex(index=False, escape=False)
    (TAB_DIR / "curated_label_distribution.tex").write_text(label_tex)
    print(f"Wrote {TAB_DIR / 'curated_label_distribution.tex'}")


def main() -> None:
    print(f"Repo root: {ROOT}")
    print(f"Figures:   {FIG_DIR}")
    print(f"Tables:    {TAB_DIR}")

    make_three_method_metric_summary()
    make_component_threshold_heatmap()
    make_mt_gan_signature_matrix()
    make_adjacent_cluster_curves()
    make_panel_montages()
    make_dense_component_curves()
    make_hierarchy_proxy_summary()
    make_latex_tables()

    print("\nDone. Generated figures:")
    for p in sorted(FIG_DIR.glob("*.png")):
        print(f"  {p.relative_to(REPORT_DIR)}")

    print("\nGenerated tables:")
    for p in sorted(TAB_DIR.glob("*")):
        print(f"  {p.relative_to(REPORT_DIR)}")


if __name__ == "__main__":
    main()
