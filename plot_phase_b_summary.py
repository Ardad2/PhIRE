import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

CSV = "phase_b_persistence_results.csv"
OUTDIR = Path("figs_phase_b")
OUTDIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV)

# --- Scatter: PSNR vs W1(PD0) ---
for run in sorted(df["run"].unique()):
    sub = df[df["run"] == run]
    plt.figure(figsize=(6, 4))
    plt.scatter(sub["psnr_sr"], sub["w1_pd0_sr"], s=12)
    plt.title(f"{run}: PSNR vs W1(PD0) on |v| (normalized)")
    plt.xlabel("PSNR (dB) (SR vs GT)")
    plt.ylabel("W1 distance (PD0) (SR vs GT)")
    plt.tight_layout()
    out = OUTDIR / f"scatter_psnr_w1_pd0_{run}.png"
    plt.savefig(out, dpi=200)
    plt.close()

# --- Scatter: PSNR vs W1(PD1) ---
for run in sorted(df["run"].unique()):
    sub = df[df["run"] == run]
    plt.figure(figsize=(6, 4))
    plt.scatter(sub["psnr_sr"], sub["w1_pd1_sr"], s=12)
    plt.title(f"{run}: PSNR vs W1(PD1) on |v| (normalized)")
    plt.xlabel("PSNR (dB) (SR vs GT)")
    plt.ylabel("W1 distance (PD1) (SR vs GT)")
    plt.tight_layout()
    out = OUTDIR / f"scatter_psnr_w1_pd1_{run}.png"
    plt.savefig(out, dpi=200)
    plt.close()

# --- Bar: mean topo distances (SR vs GT) + bicubic baseline ---
means = []
for run in sorted(df["run"].unique()):
    sub = df[df["run"] == run]
    means.append((run, sub["w1_pd0_sr"].mean(), sub["w1_pd1_sr"].mean(), sub["psnr_sr"].mean()))

# Bicubic aggregated (same rows contain bicubic metrics)
bic_pd0 = df["w1_pd0_bic"].mean()
bic_pd1 = df["w1_pd1_bic"].mean()
bic_psnr = df["psnr_bic"].mean()

labels = ["BICUBIC"] + [m[0] for m in means]
pd0_vals = [bic_pd0] + [m[1] for m in means]
pd1_vals = [bic_pd1] + [m[2] for m in means]

plt.figure(figsize=(7, 4))
x = range(len(labels))
plt.bar(x, pd0_vals)
plt.xticks(x, labels, rotation=15)
plt.title("Mean W1(PD0) vs GT (lower is better)")
plt.ylabel("Mean W1(PD0)")
plt.tight_layout()
plt.savefig(OUTDIR / "bar_mean_w1_pd0.png", dpi=200)
plt.close()

plt.figure(figsize=(7, 4))
plt.bar(x, pd1_vals)
plt.xticks(x, labels, rotation=15)
plt.title("Mean W1(PD1) vs GT (lower is better)")
plt.ylabel("Mean W1(PD1)")
plt.tight_layout()
plt.savefig(OUTDIR / "bar_mean_w1_pd1.png", dpi=200)
plt.close()

print("[OK] Wrote plots to:", OUTDIR)
