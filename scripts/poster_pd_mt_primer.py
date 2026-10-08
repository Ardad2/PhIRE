#!/usr/bin/env python3

"""
Compact PD/MT primer for the TopoInVis poster.

This is an explanatory schematic, not an experimental result.

Outputs:
    figures/poster_background/pd_mt_primer.png
    figures/poster_background/pd_mt_primer.pdf
"""

from pathlib import Path
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.colors import ListedColormap


ROOT = Path.home() / "PhIRE"
OUT = ROOT / "figures" / "poster_background"
OUT.mkdir(parents=True, exist_ok=True)

DPI = 300

C_TITLE = "#2A6654"
C_TEAL = "#3C947C"
C_TEAL_LIGHT = "#DCEFE9"
C_YELLOW = "#F4D06F"
C_ORANGE = "#F28E2B"
C_BLUE = "#DCEAF4"
C_TEXT = "#222222"
C_MUTED = "#666666"
C_BORDER = "#B6C8C3"


def rounded_box(
    ax,
    x,
    y,
    w,
    h,
    fc,
    ec=C_BORDER,
    lw=1.2,
    radius=0.02,
):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        transform=ax.transAxes,
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
    )
    ax.add_patch(patch)
    return patch


def arrow(
    ax,
    x1,
    y1,
    x2,
    y2,
    color=C_TEAL,
    lw=1.8,
):
    a = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        transform=ax.transAxes,
        arrowstyle="-|>",
        mutation_scale=16,
        linewidth=lw,
        color=color,
    )
    ax.add_patch(a)


fig = plt.figure(figsize=(15.8, 4.8))

ax = fig.add_axes([0, 0, 1, 1])
ax.set_axis_off()

fig.text(
    0.5,
    0.945,
    "Topology summarizes how threshold-connected regions persist and merge",
    ha="center",
    va="top",
    fontsize=22,
    fontweight="bold",
    color=C_TITLE,
)


# ============================================================
# LEFT: synthetic scalar field
# ============================================================

rounded_box(
    ax,
    0.02,
    0.22,
    0.18,
    0.58,
    C_BLUE,
)

fig.text(
    0.11,
    0.765,
    r"Wind-speed field  $s(x,y)$",
    ha="center",
    fontsize=16,
    fontweight="bold",
    color=C_TEXT,
)

field_ax = fig.add_axes(
    [0.045, 0.315, 0.13, 0.34]
)

n = 120
yy, xx = np.mgrid[-1:1:complex(n), -1:1:complex(n)]

field = (
    1.00 * np.exp(
        -((xx + 0.38)**2 + (yy + 0.10)**2) / 0.12
    )
    + 0.86 * np.exp(
        -((xx - 0.33)**2 + (yy - 0.35)**2) / 0.09
    )
    + 0.72 * np.exp(
        -((xx - 0.28)**2 + (yy + 0.37)**2) / 0.13
    )
)

field_ax.imshow(
    field,
    origin="upper",
    cmap="YlGn",
    interpolation="bilinear",
)

field_ax.set_xticks([])
field_ax.set_yticks([])

for s in field_ax.spines.values():
    s.set_color("#AABBB6")
    s.set_linewidth(1.0)


# ============================================================
# MIDDLE: threshold sweep
# ============================================================

fig.text(
    0.405,
    0.78,
    "Connected regions across thresholds",
    ha="center",
    fontsize=16,
    fontweight="bold",
    color=C_TEXT,
)

thresholds = [
    (0.72, r"high $\tau$"),
    (0.47, r"medium $\tau$"),
    (0.25, r"low $\tau$"),
]

x0s = [
    0.255,
    0.36,
    0.465,
]

mask_cmap = ListedColormap(
    ["#FFFFFF", C_TEAL]
)

for x0, (tau, label) in zip(
    x0s,
    thresholds,
):
    rounded_box(
        ax,
        x0,
        0.31,
        0.085,
        0.34,
        "#FFFFFF",
    )

    sub = fig.add_axes(
        [x0 + 0.010, 0.365, 0.065, 0.20]
    )

    mask = field >= tau

    sub.imshow(
        mask,
        origin="upper",
        cmap=mask_cmap,
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )

    sub.set_axis_off()

    fig.text(
        x0 + 0.0425,
        0.285,
        label,
        ha="center",
        fontsize=12.5,
        color=C_MUTED,
    )


arrow(
    ax,
    0.205,
    0.49,
    0.247,
    0.49,
)

fig.text(
    0.225,
    0.535,
    "sweep",
    ha="center",
    fontsize=12.5,
    fontweight="bold",
    color=C_TEAL,
)

arrow(
    ax,
    0.342,
    0.49,
    0.355,
    0.49,
    color="#999999",
)

arrow(
    ax,
    0.447,
    0.49,
    0.46,
    0.49,
    color="#999999",
)


# ============================================================
# BRANCH TO PD / MT
# ============================================================

arrow(
    ax,
    0.555,
    0.49,
    0.605,
    0.49,
)

ax.plot(
    [0.602, 0.625],
    [0.49, 0.64],
    transform=ax.transAxes,
    color=C_BORDER,
    linewidth=1.4,
)

ax.plot(
    [0.602, 0.625],
    [0.49, 0.35],
    transform=ax.transAxes,
    color=C_BORDER,
    linewidth=1.4,
)


# ============================================================
# PD BOX
# ============================================================

rounded_box(
    ax,
    0.625,
    0.51,
    0.165,
    0.31,
    "#FFF5D8",
)

fig.text(
    0.7075,
    0.78,
    "Persistence diagram (PD)",
    ha="center",
    fontsize=16,
    fontweight="bold",
    color=C_TEXT,
)

pd_ax = fig.add_axes(
    [0.65, 0.59, 0.105, 0.145]
)

pd_ax.plot(
    [0, 1],
    [0, 1],
    "--",
    color="#AAAAAA",
    linewidth=1.0,
)

birth = np.array([
    0.18,
    0.34,
    0.48,
    0.65,
])

death = np.array([
    0.48,
    0.77,
    0.89,
    0.94,
])

pd_ax.scatter(
    birth,
    death,
    s=38,
    color=C_TEAL,
    edgecolor="white",
    linewidth=0.7,
    zorder=3,
)

pd_ax.set_xlim(0, 1)
pd_ax.set_ylim(0, 1)

pd_ax.set_xlabel(
    "birth",
    fontsize=9,
)

pd_ax.set_ylabel(
    "death",
    fontsize=9,
)

pd_ax.tick_params(
    labelsize=7,
    length=2,
)

for side in [
    "top",
    "right",
]:
    pd_ax.spines[side].set_visible(False)

fig.text(
    0.7075,
    0.535,
    "birth/death values + persistence",
    ha="center",
    fontsize=11.5,
    color=C_MUTED,
)


# ============================================================
# MT BOX
# ============================================================

rounded_box(
    ax,
    0.625,
    0.17,
    0.165,
    0.27,
    C_TEAL_LIGHT,
)

fig.text(
    0.7075,
    0.405,
    "Merge tree (MT)",
    ha="center",
    fontsize=16,
    fontweight="bold",
    color=C_TEXT,
)

mt_ax = fig.add_axes(
    [0.655, 0.22, 0.105, 0.14]
)

mt_ax.set_xlim(0, 1)
mt_ax.set_ylim(0, 1)
mt_ax.axis("off")

nodes = {
    "root": (0.50, 0.87),
    "mid": (0.50, 0.56),
    "left": (0.22, 0.12),
    "right": (0.78, 0.12),
}

edges = [
    ("root", "mid"),
    ("mid", "left"),
    ("mid", "right"),
]

for a, b in edges:
    xa, ya = nodes[a]
    xb, yb = nodes[b]

    mt_ax.plot(
        [xa, xb],
        [ya, yb],
        color="#6A6A6A",
        linewidth=2.1,
    )

node_colors = {
    "root": C_ORANGE,
    "mid": C_YELLOW,
    "left": C_TEAL,
    "right": C_TEAL,
}

for key, (x, y) in nodes.items():
    mt_ax.scatter(
        [x],
        [y],
        s=80,
        color=node_colors[key],
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )

fig.text(
    0.7075,
    0.185,
    "critical values + merge hierarchy",
    ha="center",
    fontsize=11.5,
    color=C_MUTED,
)


# ============================================================
# RIGHT: questions
# ============================================================

rounded_box(
    ax,
    0.82,
    0.22,
    0.16,
    0.58,
    "#FAFAFA",
)

fig.text(
    0.90,
    0.715,
    "PD",
    ha="center",
    fontsize=16,
    fontweight="bold",
    color=C_TITLE,
)

fig.text(
    0.90,
    0.63,
    "Which features persist,\nand for how long?",
    ha="center",
    fontsize=13.5,
    color=C_TEXT,
)

fig.text(
    0.90,
    0.455,
    "MT",
    ha="center",
    fontsize=16,
    fontweight="bold",
    color=C_TITLE,
)

fig.text(
    0.90,
    0.365,
    "How are connected regions\norganized and merged?",
    ha="center",
    fontsize=13.5,
    color=C_TEXT,
)


# ============================================================
# BOTTOM SYNTHESIS
# ============================================================

fig.text(
    0.5,
    0.055,
    (
        "Complementary views: PD emphasizes feature persistence; "
        "MT additionally emphasizes merge hierarchy."
    ),
    ha="center",
    fontsize=15.5,
    fontweight="bold",
    color=C_TITLE,
)

fig.text(
    0.975,
    0.015,
    "schematic",
    ha="right",
    fontsize=9,
    color="#999999",
)

fig.savefig(
    OUT / "pd_mt_primer.png",
    dpi=DPI,
    bbox_inches="tight",
    facecolor="white",
)

fig.savefig(
    OUT / "pd_mt_primer.pdf",
    bbox_inches="tight",
    facecolor="white",
)

plt.close(fig)

print("Saved:")
print(OUT / "pd_mt_primer.png")
print(OUT / "pd_mt_primer.pdf")

