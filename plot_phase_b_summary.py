from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV = "phase_b_persistence_results.csv"
OUTDIR = Path("figs_phase_b")

def ensure_outdir():
    OUTDIR.mkdir(parents=True, exist_ok=True)

ensure_outdir()

df = pd.read_csv(CSV)
runs = sorted(df["run"].unique())
if len(runs) == 0:
    raise RuntimeError("No runs found in CSV.")

# Detect fields present by columns like "<field>_psnr_sr"
fields = set()
for c in df.columns:
    m = re.match(r"(.+)_psnr_sr$", c)
    if m:
        fields.add(m.group(1))
FIELDS = sorted(fields)
if not FIELDS:
    raise RuntimeError("No fields detected (expected columns like speed_psnr_sr).")

def mean_std(x):
    x = np.asarray(x, dtype=float)
    if len(x) <= 1:
        return float(np.mean(x)) if len(x) else float("nan"), 0.0
    return float(np.mean(x)), float(np.std(x, ddof=1))

def savefig(path: Path):
    ensure_outdir()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

def scatter_plots(field):
    for run in runs:
        sub = df[df["run"] == run]
        n = len(sub)

        plt.figure(figsize=(6, 4))
        plt.scatter(sub[f"{field}_psnr_sr"], sub[f"{field}_w1_pd0_sr"], s=14, alpha=0.75)
        plt.title(f"{run}: PSNR vs W1(PD0) on {field} (per-patch, n={n})")
        plt.xlabel("PSNR (dB) (SR vs GT)")
        plt.ylabel("W1 distance (PD0) (SR vs GT)")
        savefig(OUTDIR / f"scatter_psnr_w1_pd0_{field}_{run}.png")

        plt.figure(figsize=(6, 4))
        plt.scatter(sub[f"{field}_psnr_sr"], sub[f"{field}_w1_pd1_sr"], s=14, alpha=0.75)
        plt.title(f"{run}: PSNR vs W1(PD1) on {field} (per-patch, n={n})")
        plt.xlabel("PSNR (dB) (SR vs GT)")
        plt.ylabel("W1 distance (PD1) (SR vs GT)")
        savefig(OUTDIR / f"scatter_psnr_w1_pd1_{field}_{run}.png")

def make_summary(field):
    """
    Summarize per-sample by averaging patches for each sample first,
    then compute mean/std across samples.
    """
    base = df[df["run"] == runs[0]].copy()

    base_g = base.groupby("sample").agg(
        psnr_bic=(f"{field}_psnr_bic", "mean"),
        w1_pd0_bic=(f"{field}_w1_pd0_bic", "mean"),
        w1_pd1_bic=(f"{field}_w1_pd1_bic", "mean"),
        w2_pd0_bic=(f"{field}_w2_pd0_bic", "mean"),
        w2_pd1_bic=(f"{field}_w2_pd1_bic", "mean"),
        w2_pd0_bic_n=(f"{field}_w2_pd0_bic", lambda x: int(np.sum(np.isfinite(x)))),
        w2_pd1_bic_n=(f"{field}_w2_pd1_bic", lambda x: int(np.sum(np.isfinite(x)))),
        patches=("top", "count"),
    ).reset_index()

    bic_psnr_mu, bic_psnr_sd = mean_std(base_g["psnr_bic"])
    bic_pd0_mu, bic_pd0_sd = mean_std(base_g["w1_pd0_bic"])
    bic_pd1_mu, bic_pd1_sd = mean_std(base_g["w1_pd1_bic"])
    bic_w2_pd0_mu, bic_w2_pd0_sd = mean_std(base_g["w2_pd0_bic"])
    bic_w2_pd1_mu, bic_w2_pd1_sd = mean_std(base_g["w2_pd1_bic"])
    bic_w2_pd0_n = int(np.sum(base_g["w2_pd0_bic_n"]))
    bic_w2_pd1_n = int(np.sum(base_g["w2_pd1_bic_n"]))

    summary_rows = [{
        "label": "BICUBIC",
        "n_samples": int(base_g.shape[0]),
        "patches_total": int(base_g["patches"].sum()),
        "psnr_mean": bic_psnr_mu,
        "psnr_std": bic_psnr_sd,
        "w1_pd0_mean": bic_pd0_mu,
        "w1_pd0_std": bic_pd0_sd,
        "w1_pd1_mean": bic_pd1_mu,
        "w1_pd1_std": bic_pd1_sd,
        "w2_pd0_mean": bic_w2_pd0_mu,
        "w2_pd0_std": bic_w2_pd0_sd,
        "w2_pd1_mean": bic_w2_pd1_mu,
        "w2_pd1_std": bic_w2_pd1_sd,
        "w2_pd0_count": bic_w2_pd0_n,
        "w2_pd1_count": bic_w2_pd1_n,
    }]

    for run in runs:
        sub = df[df["run"] == run].copy()
        g = sub.groupby("sample").agg(
            psnr_sr=(f"{field}_psnr_sr", "mean"),
            w1_pd0_sr=(f"{field}_w1_pd0_sr", "mean"),
            w1_pd1_sr=(f"{field}_w1_pd1_sr", "mean"),
            w2_pd0_sr=(f"{field}_w2_pd0_sr", "mean"),
            w2_pd1_sr=(f"{field}_w2_pd1_sr", "mean"),
            w2_pd0_sr_n=(f"{field}_w2_pd0_sr", lambda x: int(np.sum(np.isfinite(x)))),
            w2_pd1_sr_n=(f"{field}_w2_pd1_sr", lambda x: int(np.sum(np.isfinite(x)))),
            patches=("top", "count"),
        ).reset_index()

        psnr_mu, psnr_sd = mean_std(g["psnr_sr"])
        pd0_mu, pd0_sd = mean_std(g["w1_pd0_sr"])
        pd1_mu, pd1_sd = mean_std(g["w1_pd1_sr"])
        w2_pd0_mu, w2_pd0_sd = mean_std(g["w2_pd0_sr"])
        w2_pd1_mu, w2_pd1_sd = mean_std(g["w2_pd1_sr"])
        w2_pd0_n = int(np.sum(g["w2_pd0_sr_n"]))
        w2_pd1_n = int(np.sum(g["w2_pd1_sr_n"]))

        summary_rows.append({
            "label": run,
            "n_samples": int(g.shape[0]),
            "patches_total": int(g["patches"].sum()),
            "psnr_mean": psnr_mu,
            "psnr_std": psnr_sd,
            "w1_pd0_mean": pd0_mu,
            "w1_pd0_std": pd0_sd,
            "w1_pd1_mean": pd1_mu,
            "w1_pd1_std": pd1_sd,
            "w2_pd0_mean": w2_pd0_mu,
            "w2_pd0_std": w2_pd0_sd,
            "w2_pd1_mean": w2_pd1_mu,
            "w2_pd1_std": w2_pd1_sd,
            "w2_pd0_count": w2_pd0_n,
            "w2_pd1_count": w2_pd1_n,
        })

    summary = pd.DataFrame(summary_rows)
    ensure_outdir()
    summary.to_csv(OUTDIR / f"phase_b_summary_stats_{field}.csv", index=False)
    return summary

def bar_plots(field, summary):
    labels = summary["label"].tolist()
    x = np.arange(len(labels))

    plt.figure(figsize=(7.8, 4.2))
    plt.bar(x, summary["w1_pd0_mean"].values, yerr=summary["w1_pd0_std"].values, capsize=4)
    plt.xticks(x, labels, rotation=15)
    plt.title(f"Mean W1(PD0) vs GT on {field} (per-sample avg of patches)")
    plt.ylabel("Mean W1(PD0)")
    savefig(OUTDIR / f"bar_mean_w1_pd0_{field}.png")

    plt.figure(figsize=(7.8, 4.2))
    plt.bar(x, summary["w1_pd1_mean"].values, yerr=summary["w1_pd1_std"].values, capsize=4)
    plt.xticks(x, labels, rotation=15)
    plt.title(f"Mean W1(PD1) vs GT on {field} (per-sample avg of patches)")
    plt.ylabel("Mean W1(PD1)")
    savefig(OUTDIR / f"bar_mean_w1_pd1_{field}.png")

    plt.figure(figsize=(7.8, 4.2))
    plt.bar(x, summary["psnr_mean"].values, yerr=summary["psnr_std"].values, capsize=4)
    plt.xticks(x, labels, rotation=15)
    plt.title(f"Mean PSNR vs GT on {field} (per-sample avg of patches)")
    plt.ylabel("PSNR (dB)")
    savefig(OUTDIR / f"bar_mean_psnr_{field}.png")

    # W2 bars only if some values exist
    if np.isfinite(summary["w2_pd0_mean"]).any():
        plt.figure(figsize=(7.8, 4.2))
        plt.bar(x, summary["w2_pd0_mean"].values, yerr=summary["w2_pd0_std"].values, capsize=4)
        plt.xticks(x, labels, rotation=15)
        plt.title(f"Mean W2(PD0) vs GT on {field} (AUDIT subset)")
        plt.ylabel("Mean W2(PD0)")
        savefig(OUTDIR / f"bar_mean_w2_pd0_{field}.png")

    if np.isfinite(summary["w2_pd1_mean"]).any():
        plt.figure(figsize=(7.8, 4.2))
        plt.bar(x, summary["w2_pd1_mean"].values, yerr=summary["w2_pd1_std"].values, capsize=4)
        plt.xticks(x, labels, rotation=15)
        plt.title(f"Mean W2(PD1) vs GT on {field} (AUDIT subset)")
        plt.ylabel("Mean W2(PD1)")
        savefig(OUTDIR / f"bar_mean_w2_pd1_{field}.png")

for field in FIELDS:
    scatter_plots(field)
    summary = make_summary(field)
    bar_plots(field, summary)

print("[OK] Wrote plots + stats CSVs to:", OUTDIR)
