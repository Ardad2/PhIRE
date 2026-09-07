#!/usr/bin/env python3
from pathlib import Path
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--out", type=Path, default=Path("poster_main_results_A0"))
args = parser.parse_args()

methods = ["CNN", "Ablation", "Topo-inspired"]

metrics = [
    ("Bottleneck $d_B$\n(lower is better)", [3.124, 3.288, 2.495], ".3f", "Topo vs Ablation: −24.1%"),
    ("2-Wasserstein $W_2$\n(lower is better)", [19.152, 21.062, 15.867], ".3f", "Topo vs Ablation: −24.7%"),
    ("PSNR$_{uv}$\n(higher is better)", [31.1925, 33.7892, 33.4807], ".3f", "Topo vs Ablation: −0.9%"),
    ("SSIM$_{speed}$\n(higher is better)", [0.7412, 0.8134, 0.8126], ".4f", "Topo vs Ablation: −0.1%"),
]

fig, axes = plt.subplots(2, 2, figsize=(16.8, 9.4), constrained_layout=False)
axes = axes.ravel()

fig.subplots_adjust(
    left=0.10,
    right=0.985,
    top=0.90,
    bottom=0.14,
    wspace=0.34,
    hspace=0.52
)

markers = ["o", "s", "D"]
colors = ["#1f77b4", "#ff7f0e", "#1b9e77"]

delta_pos = (0.33, 0.36)

for ax, (title, values, fmt, delta) in zip(axes, metrics):
    vals = np.asarray(values, dtype=float)
    y = np.arange(len(methods))

    ax.plot([vals[1], vals[2]], [1, 2], linewidth=1.4, alpha=0.55, zorder=1, color="#5fa2d9")

    for i, (method, value, marker, color) in enumerate(zip(methods, vals, markers, colors)):
        ax.scatter(value, i, s=115, marker=marker, zorder=3, color=color, edgecolor="none")

        if method == "Ablation":
            xytext = (-10, 0)
            ha = "right"
        else:
            xytext = (10, 0)
            ha = "left"

        ax.annotate(
            format(value, fmt),
            xy=(value, i),
            xytext=xytext,
            textcoords="offset points",
            ha=ha,
            va="center",
            fontsize=18,
            fontweight="bold"
        )

    ax.set_yticks(y, methods)
    ax.invert_yaxis()

    ax.tick_params(axis="y", labelsize=18, pad=4)
    ax.tick_params(axis="x", labelsize=16)

    ax.grid(axis="x", alpha=0.22, linewidth=0.8)
    ax.set_axisbelow(True)

    ax.set_title(title, fontsize=20, fontweight="bold", pad=10)

    span = max(vals) - min(vals)
    if span == 0:
        span = max(abs(vals[0]), 1.0) * 0.1

    ax.set_xlim(min(vals) - 0.32 * span, max(vals) + 0.30 * span)

    ax.text(
        delta_pos[0], delta_pos[1], delta,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=15,
        bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="0.6", alpha=0.95)
    )

fig.text(
    0.5, 0.045,
    "Controlled comparison: Ablation = reconstruction-only fine-tuning.",
    ha="center",
    fontsize=18
)

args.out.parent.mkdir(parents=True, exist_ok=True)
for ext in ("svg", "pdf", "png"):
    path = Path(str(args.out) + f".{ext}")
    fig.savefig(
        path,
        dpi=(350 if ext == "png" else None),
        bbox_inches="tight",
        facecolor="white"
    )
    print("Saved:", path)

plt.close(fig)