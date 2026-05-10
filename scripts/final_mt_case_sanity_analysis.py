#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm


ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "ttk_runs_fixed" / "merge_tree_proxy_report_figures"
FIG.mkdir(parents=True, exist_ok=True)

COMP_SUMMARY = ROOT / "ttk_runs_fixed" / "component_counts" / "special_component_case_summary.csv"
MTP_PER_SAMPLE = ROOT / "ttk_runs_fixed" / "merge_tree_proxy" / "merge_tree_proxy_per_sample.csv"
MANIFEST = ROOT / "ttk_runs_fixed" / "baseline_visual_panels" / "baseline_visual_manifest.csv"
OUT_CSV = ROOT / "ttk_runs_fixed" / "merge_tree_proxy" / "mt_case_branch_ratio_comparison.csv"


THRESHOLDS = ["t5", "t10", "t15", "p90", "p95", "p99"]


def savefig(name: str) -> None:
    for ext in ("png", "pdf"):
        out = FIG / f"{name}.{ext}"
        plt.savefig(out, dpi=220, bbox_inches="tight")
        print("wrote", out)
    plt.close()


def parse_signature(sig: str) -> dict[str, str]:
    out = {}
    for part in str(sig).replace(",", ";").split(";"):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        k = k.strip()
        v = v.strip().upper()
        if v in {"BIC", "BICUBIC"}:
            v = "BICUBIC"
        elif v in {"CNN", "GAN", "TIE"}:
            pass
        else:
            v = "MISSING"
        out[k] = v
    return out


def regenerate_fig4() -> None:
    df = pd.read_csv(COMP_SUMMARY)

    if "group_name" in df.columns:
        sub = df[df["group_name"].astype(str).eq("all_mt_gan")].copy()
    else:
        sub = df.copy()

    if sub.empty:
        # Fallback: infer MT-GAN rows if manifest fields are present.
        if "mt_winner" in df.columns:
            sub = df[df["mt_winner"].astype(str).str.upper().eq("GAN")].copy()

    if sub.empty:
        raise RuntimeError("No MT-GAN rows found in special_component_case_summary.csv")

    if "winner_signature" not in sub.columns:
        raise RuntimeError("winner_signature column missing from special_component_case_summary.csv")

    sub["sample_idx"] = pd.to_numeric(sub["sample_idx"], errors="coerce").astype(int)
    sub = sub.drop_duplicates(subset=["sample_idx"]).sort_values("sample_idx")

    method_to_int = {
        "BICUBIC": 0,
        "CNN": 1,
        "GAN": 2,
        "TIE": 3,
        "MISSING": 4,
    }

    mat = []
    text = []
    for _, row in sub.iterrows():
        sig = parse_signature(row["winner_signature"])
        mat_row = []
        text_row = []
        for t in THRESHOLDS:
            v = sig.get(t, "MISSING")
            mat_row.append(method_to_int[v])
            text_row.append({"BICUBIC": "B", "CNN": "C", "GAN": "G", "TIE": "T", "MISSING": "?"}[v])
        mat.append(mat_row)
        text.append(text_row)

    mat = np.asarray(mat, dtype=int)

    print("\nFigure 4 matrix check")
    print("samples:", sub["sample_idx"].tolist())
    print("unique matrix values:", sorted(np.unique(mat).tolist()))
    print("value meaning: 0=Bicubic, 1=CNN, 2=GAN, 3=Tie, 4=Missing")
    if np.all(mat == 4):
        raise RuntimeError("All Figure 4 cells are MISSING. Check winner_signature parsing.")

    # Explicit discrete colors. Missing is light gray, not transparent.
    cmap = ListedColormap(["#4b0055", "#37769b", "#39b77a", "#ffe11a", "#d9d9d9"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], cmap.N)

    fig, ax = plt.subplots(figsize=(8.5, max(5.0, 0.30 * len(sub))))

    # Use pcolormesh instead of imshow to avoid rendering weirdness.
    x = np.arange(len(THRESHOLDS) + 1)
    y = np.arange(len(sub) + 1)
    mesh = ax.pcolormesh(x, y, mat, cmap=cmap, norm=norm, edgecolors="white", linewidth=0.5)

    ax.set_xlim(0, len(THRESHOLDS))
    ax.set_ylim(0, len(sub))
    ax.invert_yaxis()

    ax.set_xticks(np.arange(len(THRESHOLDS)) + 0.5)
    ax.set_xticklabels(THRESHOLDS)
    ax.set_yticks(np.arange(len(sub)) + 0.5)
    ax.set_yticklabels(sub["sample_idx"].astype(str))

    ax.set_xlabel("Threshold")
    ax.set_ylabel("MT-GAN sample")
    ax.set_title("Component-count winner signatures for MT-GAN cases")

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j + 0.5, i + 0.5, text[i][j], ha="center", va="center", fontsize=8, color="white")

    cbar = fig.colorbar(mesh, ax=ax, ticks=[0, 1, 2, 3, 4])
    cbar.ax.set_yticklabels(["Bicubic", "CNN", "GAN", "Tie", "Missing"])

    savefig("fig04_mt_gan_signature_matrix")


def find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def branch_ratio_analysis() -> None:
    manifest = pd.read_csv(MANIFEST)
    per = pd.read_csv(MTP_PER_SAMPLE)

    manifest["sample_idx"] = pd.to_numeric(manifest["sample_idx"], errors="coerce").astype(int)
    per["sample_idx"] = pd.to_numeric(per["sample_idx"], errors="coerce").astype(int)

    gt_col = find_col(per, ["gt_branch_count", "branch_count_gt"])
    cnn_col = find_col(per, ["cnn_branch_count", "branch_count_cnn"])
    gan_col = find_col(per, ["gan_branch_count", "branch_count_gan"])
    bic_col = find_col(per, ["bicubic_branch_count", "bic_branch_count", "branch_count_bicubic"])

    if not all([gt_col, cnn_col, gan_col]):
        print("Could not find expected branch-count columns.")
        print("Columns in merge_tree_proxy_per_sample.csv:")
        print(list(per.columns))
        raise RuntimeError("Missing branch-count columns.")

    keep_cols = ["sample_idx", gt_col, cnn_col, gan_col]
    if bic_col:
        keep_cols.append(bic_col)

    df = manifest.merge(per[keep_cols], on="sample_idx", how="left")
    df["mt_winner"] = df["mt_winner"].astype(str).str.upper()

    gt = pd.to_numeric(df[gt_col], errors="coerce").replace(0, np.nan)
    df["cnn_branch_ratio"] = pd.to_numeric(df[cnn_col], errors="coerce") / gt
    df["gan_branch_ratio"] = pd.to_numeric(df[gan_col], errors="coerce") / gt
    if bic_col:
        df["bicubic_branch_ratio"] = pd.to_numeric(df[bic_col], errors="coerce") / gt

    df["case_group"] = np.where(df["mt_winner"].eq("GAN"), "MT-GAN", "MT-CNN")

    clusters = {
        "cluster_10_13": [10, 11, 12, 13],
        "cluster_76_78": [76, 77, 78],
        "cluster_90_93": [90, 91, 92, 93],
        "cluster_161_164": [161, 162, 163, 164],
        "limitation_cases": [25, 80, 154],
    }

    df["curated_group"] = ""
    for name, ids in clusters.items():
        df.loc[df["sample_idx"].isin(ids), "curated_group"] = name

    df.to_csv(OUT_CSV, index=False)
    print("wrote", OUT_CSV)

    print("\nMT-GAN vs MT-CNN branch-ratio summary")
    summary = (
        df.groupby("case_group")
          .agg(
              n=("sample_idx", "count"),
              mean_gan_ratio=("gan_branch_ratio", "mean"),
              median_gan_ratio=("gan_branch_ratio", "median"),
              std_gan_ratio=("gan_branch_ratio", "std"),
              mean_cnn_ratio=("cnn_branch_ratio", "mean"),
              median_cnn_ratio=("cnn_branch_ratio", "median"),
          )
          .reset_index()
    )
    print(summary.to_string(index=False))

    # Save summary CSV too
    summary.to_csv(ROOT / "ttk_runs_fixed" / "merge_tree_proxy" / "mt_case_branch_ratio_group_summary.csv", index=False)

    # Fig 11: MT-GAN vs MT-CNN boxplot
    data = [
        df[df["case_group"].eq("MT-GAN")]["gan_branch_ratio"].dropna().to_numpy(),
        df[df["case_group"].eq("MT-CNN")]["gan_branch_ratio"].dropna().to_numpy(),
    ]

    plt.figure(figsize=(6.8, 4.8))
    plt.boxplot(data, labels=["MT-GAN", "MT-CNN"], showmeans=True)
    plt.axhline(1.0, linestyle="--", linewidth=1)
    plt.ylabel("GAN branch count / GT branch count")
    plt.title("Does MT-GAN have lower GAN over-fragmentation?")
    plt.grid(True, axis="y", alpha=0.25)
    savefig("fig11_branch_ratio_mt_gan_vs_mt_cnn")

    # Fig 12: adjacent clusters
    rows = []
    for cluster_name, ids in clusters.items():
        for sid in ids:
            row = df[df["sample_idx"].eq(sid)]
            if row.empty:
                continue
            r = row.iloc[0]
            rows.append({
                "cluster": cluster_name.replace("cluster_", "").replace("_", "-"),
                "sample_idx": sid,
                "mt_winner": r["mt_winner"],
                "cnn_ratio": r["cnn_branch_ratio"],
                "gan_ratio": r["gan_branch_ratio"],
            })

    cdf = pd.DataFrame(rows)
    if not cdf.empty:
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharey=True)
        plot_clusters = ["10-13", "76-78", "90-93"]

        for ax, cname in zip(axes, plot_clusters):
            sub = cdf[cdf["cluster"].eq(cname)].copy()
            x = np.arange(len(sub))
            ax.plot(x, sub["cnn_ratio"], marker="o", label="CNN / GT")
            ax.plot(x, sub["gan_ratio"], marker="o", label="GAN / GT")
            ax.axhline(1.0, linestyle="--", linewidth=1)
            ax.set_xticks(x)
            labels = [
                f"{int(r.sample_idx)}\nMT={r.mt_winner}"
                for _, r in sub.iterrows()
            ]
            ax.set_xticklabels(labels)
            ax.set_title(f"Adjacent cluster {cname}")
            ax.grid(True, axis="y", alpha=0.25)

        axes[0].set_ylabel("Branch count / GT branch count")
        axes[0].legend(loc="best")
        fig.suptitle("Adjacent cluster branch-count ratios")
        fig.tight_layout()
        savefig("fig12_adjacent_branch_ratio_clusters")


def main() -> None:
    regenerate_fig4()
    branch_ratio_analysis()


if __name__ == "__main__":
    main()
