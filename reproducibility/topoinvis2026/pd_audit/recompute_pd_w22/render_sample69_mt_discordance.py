#!/usr/bin/env python3

import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

import vtk
from vtk.util.numpy_support import vtk_to_numpy


W22 = Path(os.environ["W22"])

BASE = (
    W22
    / "corrected_pd_mt"
    / "discordance_visuals"
    / "sample_069"
)

INPUT = BASE / "inputs"
T3 = BASE / "fixed_threshold_t3"

OUT = BASE / "figures"
OUT.mkdir(parents=True, exist_ok=True)

METHODS = [
    (
        "gt",
        "GT",
        INPUT / "gt_s069.vti",
        BASE / "gt_threshold_sweep/t3",
    ),
    (
        "cnn",
        "CNN",
        INPUT / "cnn_s069.vti",
        T3 / "cnn",
    ),
    (
        "uv",
        r"Ablation ($L_{uv}$ only)",
        INPUT / "uv_s069.vti",
        T3 / "uv",
    ),
    (
        "f1",
        "Candidate F",
        INPUT / "f1_s069.vti",
        T3 / "f1",
    ),
]


def read_vti(path):
    r = vtk.vtkXMLImageDataReader()
    r.SetFileName(str(path))
    r.Update()

    img = r.GetOutput()

    dims = img.GetDimensions()

    arr = img.GetPointData().GetArray("wind_speed")

    if arr is None:
        raise RuntimeError(
            f"wind_speed missing: {path}"
        )

    flat = vtk_to_numpy(arr)

    W = dims[0]
    H = dims[1]

    field = np.asarray(
        flat,
        dtype=np.float64,
    ).reshape(
        H,
        W,
        order="C",
    )

    return field


def read_grid(path):
    r = vtk.vtkXMLUnstructuredGridReader()
    r.SetFileName(str(path))
    r.Update()

    g = r.GetOutput()

    if g is None:
        raise RuntimeError(
            f"Could not read {path}"
        )

    return g


def extract_nodes(grid):
    pts = vtk_to_numpy(
        grid.GetPoints().GetData()
    )

    return np.asarray(
        pts[:, :2],
        dtype=np.float64,
    )


def extract_segments(grid):
    pts = vtk_to_numpy(
        grid.GetPoints().GetData()
    )

    segments = []

    for cell_id in range(
        grid.GetNumberOfCells()
    ):
        cell = grid.GetCell(cell_id)

        ids = cell.GetPointIds()

        if ids.GetNumberOfIds() != 2:
            raise RuntimeError(
                f"Expected VTK_LINE with 2 ids; "
                f"cell {cell_id} has "
                f"{ids.GetNumberOfIds()}"
            )

        p0 = pts[
            ids.GetId(0),
            :2,
        ]

        p1 = pts[
            ids.GetId(1),
            :2,
        ]

        segments.append(
            [
                [float(p0[0]), float(p0[1])],
                [float(p1[0]), float(p1[1])],
            ]
        )

    return np.asarray(
        segments,
        dtype=np.float64,
    )


data = {}

global_min = float("inf")
global_max = float("-inf")

for key, label, vti, root in METHODS:
    field = read_vti(vti)

    nodes = extract_nodes(
        read_grid(
            root / "nodes_display.vtu"
        )
    )

    segments = extract_segments(
        read_grid(
            root / "arcs_display.vtu"
        )
    )

    global_min = min(
        global_min,
        float(np.min(field)),
    )

    global_max = max(
        global_max,
        float(np.max(field)),
    )

    data[key] = {
        "label": label,
        "field": field,
        "nodes": nodes,
        "segments": segments,
    }


# ============================================================================
# Figure 1: fields + spatially embedded Join Trees
# ============================================================================

fig, axes = plt.subplots(
    1,
    4,
    figsize=(16.0, 4.25),
    sharex=True,
    sharey=True,
)

last_im = None

for ax, (key, _, _, _) in zip(
    axes,
    METHODS,
):
    d = data[key]

    last_im = ax.imshow(
        d["field"],
        origin="lower",
        extent=(0, 159, 0, 159),
        vmin=global_min,
        vmax=global_max,
        cmap="viridis",
        interpolation="nearest",
    )

    lc_shadow = LineCollection(
        d["segments"],
        linewidths=2.8,
        colors="black",
        alpha=0.70,
        zorder=3,
    )

    lc = LineCollection(
        d["segments"],
        linewidths=1.35,
        colors="white",
        alpha=0.95,
        zorder=4,
    )

    ax.add_collection(lc_shadow)
    ax.add_collection(lc)

    ax.scatter(
        d["nodes"][:, 0],
        d["nodes"][:, 1],
        s=24,
        facecolors="white",
        edgecolors="black",
        linewidths=0.75,
        zorder=5,
    )

    ax.set_title(
        d["label"],
        fontsize=13,
        fontweight="bold",
    )

    ax.set_xlim(0, 159)
    ax.set_ylim(0, 159)
    ax.set_aspect("equal")

    ax.set_xlabel("x")

axes[0].set_ylabel("y")

cbar = fig.colorbar(
    last_im,
    ax=axes,
    fraction=0.025,
    pad=0.015,
)

cbar.set_label(
    "Wind speed magnitude",
)

fig.suptitle(
    "Sample 69: spatially embedded Join Trees "
    "(display persistence threshold = 3.0)",
    fontsize=15,
    fontweight="bold",
)

fig.subplots_adjust(
    left=0.055,
    right=0.93,
    bottom=0.11,
    top=0.84,
    wspace=0.09,
)

overlay_png = (
    OUT
    / "sample_069_mt_t3_field_overlay.png"
)

overlay_pdf = (
    OUT
    / "sample_069_mt_t3_field_overlay.pdf"
)

fig.savefig(
    overlay_png,
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    overlay_pdf,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================================
# Figure 2: tree-only spatial embedding
# ============================================================================

fig, axes = plt.subplots(
    1,
    4,
    figsize=(15.5, 4.1),
    sharex=True,
    sharey=True,
)

for ax, (key, _, _, _) in zip(
    axes,
    METHODS,
):
    d = data[key]

    lc = LineCollection(
        d["segments"],
        linewidths=1.35,
        colors="black",
        alpha=0.88,
        zorder=2,
    )

    ax.add_collection(lc)

    ax.scatter(
        d["nodes"][:, 0],
        d["nodes"][:, 1],
        s=20,
        facecolors="white",
        edgecolors="black",
        linewidths=0.8,
        zorder=3,
    )

    ax.set_title(
        (
            f"{d['label']}\n"
            f"{len(d['nodes'])} nodes, "
            f"{len(d['segments'])} arc segments"
        ),
        fontsize=11.5,
        fontweight="bold",
    )

    ax.set_xlim(0, 159)
    ax.set_ylim(0, 159)
    ax.set_aspect("equal")

    ax.set_xlabel("x")

axes[0].set_ylabel("y")

fig.suptitle(
    "Sample 69: simplified Join-Tree spatial embeddings",
    fontsize=15,
    fontweight="bold",
)

fig.subplots_adjust(
    left=0.055,
    right=0.985,
    bottom=0.12,
    top=0.80,
    wspace=0.10,
)

tree_png = (
    OUT
    / "sample_069_mt_t3_tree_only.png"
)

tree_pdf = (
    OUT
    / "sample_069_mt_t3_tree_only.pdf"
)

fig.savefig(
    tree_png,
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    tree_pdf,
    bbox_inches="tight",
)

plt.close(fig)


print("=" * 88)
print("SAMPLE-69 MT VISUALIZATION COMPLETE")
print("=" * 88)

print()
print("threshold: 3.0 (frozen before method inspection)")
print()

for key, _, _, _ in METHODS:
    d = data[key]

    print(
        f"{key:4s}: "
        f"nodes={len(d['nodes']):3d} "
        f"arc_segments={len(d['segments']):4d}"
    )

print()
print("overlay PNG:", overlay_png)
print("overlay PDF:", overlay_pdf)
print("tree PNG:   ", tree_png)
print("tree PDF:   ", tree_pdf)
