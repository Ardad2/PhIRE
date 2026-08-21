#!/usr/bin/env python3

import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

AUDIT = Path.home() / "phire_runtime_audit_20260809_221548"

INPUT = (
    AUDIT
    / "recompute_pd"
    / "canonical_pd_full_sweep.csv"
)

OUT = Path("poster_figures")
OUT.mkdir(exist_ok=True)

RUNS = {
    "cnn": "CNN",
    "gan": "GAN",
    "topology_finetuning/candidateUV_expanded2688_topology":
        "Vector-only",
    "topology_finetuning/candidateC_expanded2688_topology":
        "Topology-inspired",
}

# Colorblind-friendly and consistent across poster figures
COLORS = {
    "CNN": "#0072B2",
    "GAN": "#E69F00",
    "Vector-only": "#009E73",
    "Topology-inspired": "#CC79A7",
}

values = {
    name: {"db": [], "w2": []}
    for name in RUNS.values()
}

with INPUT.open(newline="") as f:
    for r in csv.DictReader(f):
        if r["run"] not in RUNS:
            continue

        name = RUNS[r["run"]]

        values[name]["db"].append(
            float(r["pd_bottleneck_all"])
        )

        values[name]["w2"].append(
            float(r["pd_w2_all"])
        )

names = list(RUNS.values())

db = np.array([
    np.mean(values[n]["db"])
    for n in names
])

w2 = np.array([
    np.mean(values[n]["w2"])
    for n in names
])

print("Means:")
for n, a, b in zip(names, db, w2):
    print(f"{n:20s} dB={a:.6f} W2={b:.6f}")

fig, axes = plt.subplots(
    1, 2,
    figsize=(10.5, 4.2),
    constrained_layout=True,
)

for ax, vals, ylabel, title in [
    (
        axes[0],
        db,
        r"$d_B$",
        r"PD bottleneck $d_B$",
    ),
    (
        axes[1],
        w2,
        r"$W_2$",
        r"PD 2-Wasserstein $W_2$",
    ),
]:
    bars = ax.bar(
        names,
        vals,
        color=[COLORS[n] for n in names],
        width=0.67,
    )

    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(
        title + "\n(lower is better)",
        fontsize=14,
        weight="bold",
    )

    ax.tick_params(
        axis="x",
        labelrotation=15,
        labelsize=10,
    )

    ax.grid(
        axis="y",
        alpha=0.25,
        linestyle="--",
    )

    ax.set_axisbelow(True)

    for bar, value in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height(),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=11,
            weight="bold",
        )

fig.suptitle(
    "Persistence-Diagram Agreement — 168 Held-Out Samples",
    fontsize=16,
    weight="bold",
)

for suffix in ["pdf", "svg"]:
    fig.savefig(
        OUT / f"poster_pd_summary.{suffix}",
        bbox_inches="tight",
    )

fig.savefig(
    OUT / "poster_pd_summary.png",
    dpi=400,
    bbox_inches="tight",
)

print("Wrote poster_figures/poster_pd_summary.[pdf|svg|png]")
