import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV = "phase_b_persistence_results.csv"
OUTDIR = Path("figs_phase_b")
OUTDIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV)

# Error handling.

runs = sorted(df["run"].unique())
if len(runs) == 0:
    raise RuntimeError("No runs found in CSV.")

for run in runs:
    sub = df[df["run"] == run]
    n = len(sub)

# --- Scatter: PSNR vs W1(PD0) ---

    plt.figure(figsize=(6, 4))
    plt.scatter(sub["psnr_sr"], sub["w1_pd0_sr"], s=18, alpha=0.85)
    plt.title(f"{run}: PSNR vs W1(PD0) on |v| (normalized)  (n={n})")
    plt.xlabel("PSNR (dB) (SR vs GT)")
    plt.ylabel("W1 distance (PD0) (SR vs GT)")
    plt.tight_layout()
    plt.savefig(OUTDIR / f"scatter_psnr_w1_pd0_{run}.png", dpi=200)
    plt.close()

# --- Scatter: PSNR vs W1(PD1) ---

    plt.figure(figsize=(6, 4))
    plt.scatter(sub["psnr_sr"], sub["w1_pd1_sr"], s=18, alpha=0.85)
    plt.title(f"{run}: PSNR vs W1(PD1) on |v| (normalized)  (n={n})")
    plt.xlabel("PSNR (dB) (SR vs GT)")
    plt.ylabel("W1 distance (PD1) (SR vs GT)")
    plt.tight_layout()
    plt.savefig(OUTDIR / f"scatter_psnr_w1_pd1_{run}.png", dpi=200)
    plt.close()



def mean_std(x):
    x = np.asarray(x, dtype=float)
    return float(np.mean(x)), float(np.std(x, ddof=1)) if len(x) > 1 else 0.0

# Note: Bicubic metrics are duplicated across the run (same baseline for eachy run), so bicubic baseline is only computed from one run to avoid duplicates.

base_run = runs[0]
base = df[df["run"] == base_run].copy()

# Statistics for the bicubic baseline (computed per-sample)
bic_pd0_mu, bic_pd0_sd = mean_std(base["w1_pd0_bic"])
bic_pd1_mu, bic_pd1_sd = mean_std(base["w1_pd1_bic"])
bic_psnr_mu, bic_psnr_sd = mean_std(base["psnr_bic"])

summary_rows = []
summary_rows.append({
    "label": "BICUBIC",
    "n": len(base),
    "psnr_mean": bic_psnr_mu,
    "psnr_std": bic_psnr_sd,
    "w1_pd0_mean": bic_pd0_mu,
    "w1_pd0_std": bic_pd0_sd,
    "w1_pd1_mean": bic_pd1_mu,
    "w1_pd1_std": bic_pd1_sd,
})

# Statistics per SR run.
for run in runs:
    sub = df[df["run"] == run]
    psnr_mu, psnr_sd = mean_std(sub["psnr_sr"])
    pd0_mu, pd0_sd = mean_std(sub["w1_pd0_sr"])
    pd1_mu, pd1_sd = mean_std(sub["w1_pd1_sr"])
    summary_rows.append({
        "label": run,
        "n": len(sub),
        "psnr_mean": psnr_mu,
        "psnr_std": psnr_sd,
        "w1_pd0_mean": pd0_mu,
        "w1_pd0_std": pd0_sd,
        "w1_pd1_mean": pd1_mu,
        "w1_pd1_std": pd1_sd,
    })

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUTDIR / "phase_b_summary_stats.csv", index=False)

labels = summary["label"].tolist()
x = np.arange(len(labels))

# Bar plots with error bars

plt.figure(figsize=(7.4, 4.2))
plt.bar(x, summary["w1_pd0_mean"].values, yerr=summary["w1_pd0_std"].values, capsize=4)
plt.xticks(x, labels, rotation=15)
plt.title("Mean W1(PD0) vs GT (lower is better)")
plt.ylabel("Mean W1(PD0)")
plt.tight_layout()
plt.savefig(OUTDIR / "bar_mean_w1_pd0.png", dpi=200)
plt.close()

plt.figure(figsize=(7.4, 4.2))
plt.bar(x, summary["w1_pd1_mean"].values, yerr=summary["w1_pd1_std"].values, capsize=4)
plt.xticks(x, labels, rotation=15)
plt.title("Mean W1(PD1) vs GT (lower is better)")
plt.ylabel("Mean W1(PD1)")
plt.tight_layout()
plt.savefig(OUTDIR / "bar_mean_w1_pd1.png", dpi=200)
plt.close()

# PSNR Bar plots
plt.figure(figsize=(7.4, 4.2))
plt.bar(x, summary["psnr_mean"].values, yerr=summary["psnr_std"].values, capsize=4)
plt.xticks(x, labels, rotation=15)
plt.title("Mean PSNR vs GT (higher is better)")
plt.ylabel("PSNR (dB)")
plt.tight_layout()
plt.savefig(OUTDIR / "bar_mean_psnr.png", dpi=200)
plt.close()

print("[OK] Wrote plots + stats CSV to:", OUTDIR)
print("     Stats:", (OUTDIR / "phase_b_summary_stats.csv"))
