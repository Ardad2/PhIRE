#!/usr/bin/env python3
from pathlib import Path
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--out", type=Path, default=Path("poster_broader_metrics_A0"))
args = parser.parse_args()

labels = [
    r"Fidelity · PSNR$_{uv}$",
    r"Fidelity · SSIM$_{speed}$",
    "Direct · Speed MAE",
    "Wind power · WPD bias",
    "Wind power · WPD MAE",
    r"Wind power · WPD $W_1$",
    r"Spectrum · PSD log-$L_2$",
    r"Gradients · Gradient $W_1$",
    r"Extremes · p95 exceedance",
    r"Structure · Component curve $L_1$",
]
changes = np.array([-0.9, -0.1, -3.8, 74.5, -4.4, 59.0, 16.0, 22.2, 69.4, 20.6])

fig, ax = plt.subplots(figsize=(15.2, 6.8))
plt.subplots_adjust(left=0.34, right=0.98, bottom=0.19, top=0.96)

y = np.arange(len(labels))

bar_colors = ["#0072B2" if v >= 0 else "#9E9E9E" for v in changes]
bars = ax.barh(
    y, changes,
    color=bar_colors,
    edgecolor="black",
    linewidth=0.5
)

# Hatch negative bars so color is not the only cue
for rect, value in zip(bars, changes):
    if value < 0:
        rect.set_hatch("//")

ax.axvline(0, color="black", linewidth=1.2)
ax.set_yticks(y, labels)
ax.invert_yaxis()
ax.tick_params(axis="y", labelsize=17)
ax.tick_params(axis="x", labelsize=16)
ax.grid(axis="x", alpha=0.22, linewidth=0.8)
ax.set_axisbelow(True)

xmin, xmax = -12, 82
ax.set_xlim(xmin, xmax)
ax.set_xlabel("Relative change vs Ablation (%)", fontsize=19)

# Value labels: keep all of them inside the figure bounds
for rect, value in zip(bars, changes):
    yc = rect.get_y() + rect.get_height() / 2

    if value >= 0:
        x = min(value + 1.1, xmax - 0.8)
        ha = "left"
    else:
        x = max(value - 1.1, xmin + 0.8)
        ha = "right"

    ax.text(
        x, yc, f"{value:+.1f}%",
        va="center", ha=ha,
        fontsize=16,
        fontweight="bold",
        clip_on=False
    )

# Small accessibility note
ax.text(
    0.98, 0.02,
    "Blue solid = better   ·   Gray hatched = worse",
    transform=ax.transAxes,
    ha="right", va="bottom",
    fontsize=13, color="0.30"
)

fig.text(
    0.5, 0.06,
    "Positive values favor Topology-inspired. Selected metrics shown; not every non-topology metric improves.",
    ha="center", fontsize=16
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