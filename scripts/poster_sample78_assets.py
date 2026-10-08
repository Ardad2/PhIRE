#!/usr/bin/env python3

"""
Poster assets for Sample 78.

Produces:
  - a compact real-data motivation strip for Section 1:
        GT | Pretrained CNN | Topology-inspired
  - separate clean field panels:
        GT, CNN, fine-tuning without topology losses, topology-inspired
  - separate absolute-error maps:
        CNN, fine-tuning without topology losses, topology-inspired
  - shared speed and error colorbars
  - a simple full qualitative-layout preview

All field/error panels use the same fixed 160x160 topology-evaluation crop:
    x0 = 0, y0 = 0

The script resolves SAMPLE_ID using idx.npy rather than assuming that
the array row is identical to the sample identifier.
"""

from pathlib import Path
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


# ============================================================
# CONFIG
# ============================================================

ROOT = Path.home() / "PhIRE"

SAMPLE_ID = 78

CNN_DIR = ROOT / "data_out_fixed" / "wind_mrhr_cnn"

# Matched fine-tuning control:
CONTROL_DIR = (
    ROOT
    / "data_out"
    / "wind_finetune_candidateUV_expanded2688"
)

# Final topology-inspired model:
TOPO_DIR = (
    ROOT
    / "data_out"
    / "wind_finetune_candidateF_grad_E2_low_expanded2688"
)

OUT_DIR = ROOT / "figures" / "poster_sample78_assets"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Audited topology evaluation domain
X0 = 0
Y0 = 0
PATCH = 160

CMAP_FIELD = "viridis"
CMAP_ERROR = "magma"

# Poster colors
TITLE_COLOR = "#007FB6"
TEXT_COLOR = "#222222"
MUTED_COLOR = "#666666"

# Export
DPI = 300


# ============================================================
# HELPERS
# ============================================================

def require(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"\nMissing required path:\n  {path}\n"
        )


def load_method(method_dir: Path):
    require(method_dir)
    require(method_dir / "idx.npy")
    require(method_dir / "dataSR.npy")

    idx = np.load(method_dir / "idx.npy")
    sr = np.load(method_dir / "dataSR.npy", mmap_mode="r")

    return idx, sr


def locate_sample(idx, sample_id, label):
    hits = np.flatnonzero(idx == sample_id)

    if len(hits) != 1:
        raise RuntimeError(
            f"{label}: expected exactly one row for sample {sample_id}, "
            f"found {len(hits)}"
        )

    return int(hits[0])


def speed_from_uv(uv):
    """
    uv shape must be (H, W, 2).
    No transpose is performed.
    """
    uv = np.asarray(uv)

    if uv.ndim != 3 or uv.shape[-1] != 2:
        raise ValueError(
            f"Expected (H,W,2), got {uv.shape}"
        )

    u = uv[..., 0]
    v = uv[..., 1]

    return np.sqrt(u * u + v * v)


def crop(field):
    return field[
        Y0:Y0 + PATCH,
        X0:X0 + PATCH
    ]


def save_clean_panel(
    arr,
    basename,
    cmap,
    vmin,
    vmax,
):
    """
    Clean image panel with no title, axes, or colorbar.
    Intended for manual poster placement.
    """

    fig, ax = plt.subplots(figsize=(3.2, 3.2))

    ax.imshow(
        arr,
        origin="upper",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        aspect="equal",
    )

    ax.set_axis_off()

    fig.subplots_adjust(
        left=0,
        right=1,
        bottom=0,
        top=1,
    )

    fig.savefig(
        OUT_DIR / f"{basename}.png",
        dpi=DPI,
        bbox_inches="tight",
        pad_inches=0,
        facecolor="white",
    )

    fig.savefig(
        OUT_DIR / f"{basename}.pdf",
        bbox_inches="tight",
        pad_inches=0,
        facecolor="white",
    )

    plt.close(fig)


def save_vertical_colorbar(
    basename,
    cmap,
    vmin,
    vmax,
    label,
):
    fig, ax = plt.subplots(figsize=(1.25, 4.2))

    fig.subplots_adjust(
        left=0.10,
        right=0.45,
        bottom=0.06,
        top=0.97,
    )

    ax.set_visible(False)

    cax = fig.add_axes(
        [0.20, 0.08, 0.20, 0.86]
    )

    sm = ScalarMappable(
        norm=Normalize(vmin=vmin, vmax=vmax),
        cmap=cmap,
    )
    sm.set_array([])

    cb = fig.colorbar(sm, cax=cax)

    cb.set_label(
        label,
        fontsize=13,
        rotation=90,
        labelpad=10,
    )

    cb.ax.tick_params(labelsize=11)

    fig.savefig(
        OUT_DIR / f"{basename}.png",
        dpi=DPI,
        bbox_inches="tight",
        facecolor="white",
    )

    fig.savefig(
        OUT_DIR / f"{basename}.pdf",
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)


# ============================================================
# LOAD DATA
# ============================================================

require(CNN_DIR / "dataGT.npy")

cnn_idx, cnn_sr_all = load_method(CNN_DIR)
ctl_idx, ctl_sr_all = load_method(CONTROL_DIR)
topo_idx, topo_sr_all = load_method(TOPO_DIR)

cnn_row = locate_sample(
    cnn_idx,
    SAMPLE_ID,
    "Pretrained CNN",
)

ctl_row = locate_sample(
    ctl_idx,
    SAMPLE_ID,
    "Fine-tuning without topology losses",
)

topo_row = locate_sample(
    topo_idx,
    SAMPLE_ID,
    "Topology-inspired",
)

gt_all = np.load(
    CNN_DIR / "dataGT.npy",
    mmap_mode="r",
)

gt_uv = gt_all[cnn_row]
cnn_uv = cnn_sr_all[cnn_row]
ctl_uv = ctl_sr_all[ctl_row]
topo_uv = topo_sr_all[topo_row]

print("=" * 72)
print(f"SAMPLE_ID: {SAMPLE_ID}")
print("Rows resolved through idx.npy:")
print(f"  GT/CNN row       : {cnn_row}")
print(f"  Control row       : {ctl_row}")
print(f"  Topology row      : {topo_row}")
print()
print("Vector-field shapes:")
print(f"  GT                : {gt_uv.shape}")
print(f"  Pretrained CNN    : {cnn_uv.shape}")
print(f"  No-topology FT    : {ctl_uv.shape}")
print(f"  Topology-inspired : {topo_uv.shape}")


# ============================================================
# WIND SPEED + TOPOLOGY CROP
# ============================================================

gt_full = speed_from_uv(gt_uv)
cnn_full = speed_from_uv(cnn_uv)
ctl_full = speed_from_uv(ctl_uv)
topo_full = speed_from_uv(topo_uv)

gt = crop(gt_full)
cnn = crop(cnn_full)
ctl = crop(ctl_full)
topo = crop(topo_full)

assert gt.shape == (PATCH, PATCH)
assert cnn.shape == (PATCH, PATCH)
assert ctl.shape == (PATCH, PATCH)
assert topo.shape == (PATCH, PATCH)

print()
print("160x160 crop wind-speed ranges:")
for name, arr in [
    ("GT", gt),
    ("CNN", cnn),
    ("Control", ctl),
    ("Topology-inspired", topo),
]:
    print(
        f"  {name:20s}: "
        f"min={np.min(arr):.5f}, "
        f"max={np.max(arr):.5f}, "
        f"mean={np.mean(arr):.5f}"
    )


# ============================================================
# ERRORS
# ============================================================

cnn_err = np.abs(cnn - gt)
ctl_err = np.abs(ctl - gt)
topo_err = np.abs(topo - gt)

print()
print("Crop absolute-error summaries:")
for name, arr in [
    ("CNN", cnn_err),
    ("Control", ctl_err),
    ("Topology-inspired", topo_err),
]:
    print(
        f"  {name:20s}: "
        f"MAE={np.mean(arr):.6f}, "
        f"RMSE={np.sqrt(np.mean(arr**2)):.6f}, "
        f"max={np.max(arr):.6f}"
    )


# ============================================================
# SHARED SCALES
# ============================================================

FIELD_VMIN = 0.0
FIELD_VMAX = max(
    float(np.max(gt)),
    float(np.max(cnn)),
    float(np.max(ctl)),
    float(np.max(topo)),
)

ERROR_VMIN = 0.0
ERROR_VMAX = max(
    float(np.max(cnn_err)),
    float(np.max(ctl_err)),
    float(np.max(topo_err)),
)

print()
print(
    f"Shared field scale: "
    f"[{FIELD_VMIN:.4f}, {FIELD_VMAX:.4f}]"
)
print(
    f"Shared error scale: "
    f"[{ERROR_VMIN:.4f}, {ERROR_VMAX:.4f}]"
)
print("=" * 72)


# ============================================================
# EXPORT CLEAN INDIVIDUAL PANELS
# ============================================================

save_clean_panel(
    gt,
    "sample078_gt_field",
    CMAP_FIELD,
    FIELD_VMIN,
    FIELD_VMAX,
)

save_clean_panel(
    cnn,
    "sample078_pretrained_cnn_field",
    CMAP_FIELD,
    FIELD_VMIN,
    FIELD_VMAX,
)

save_clean_panel(
    ctl,
    "sample078_no_topology_ft_field",
    CMAP_FIELD,
    FIELD_VMIN,
    FIELD_VMAX,
)

save_clean_panel(
    topo,
    "sample078_topology_inspired_field",
    CMAP_FIELD,
    FIELD_VMIN,
    FIELD_VMAX,
)

save_clean_panel(
    cnn_err,
    "sample078_pretrained_cnn_error",
    CMAP_ERROR,
    ERROR_VMIN,
    ERROR_VMAX,
)

save_clean_panel(
    ctl_err,
    "sample078_no_topology_ft_error",
    CMAP_ERROR,
    ERROR_VMIN,
    ERROR_VMAX,
)

save_clean_panel(
    topo_err,
    "sample078_topology_inspired_error",
    CMAP_ERROR,
    ERROR_VMIN,
    ERROR_VMAX,
)

save_vertical_colorbar(
    "sample078_speed_colorbar",
    CMAP_FIELD,
    FIELD_VMIN,
    FIELD_VMAX,
    r"Wind speed (m s$^{-1}$)",
)

save_vertical_colorbar(
    "sample078_error_colorbar",
    CMAP_ERROR,
    ERROR_VMIN,
    ERROR_VMAX,
    r"Absolute speed error (m s$^{-1}$)",
)


# ============================================================
# SECTION 1 MINI MOTIVATION STRIP
# ============================================================
#
# Use GT / pretrained CNN / topology-inspired.
#
# This is safer than GT / control / topology-inspired for the
# introductory near-tie visual because Sample 78 was explicitly
# selected from a conventional near-tie cohort relative to CNN.
#

fig, axes = plt.subplots(
    1,
    3,
    figsize=(11.2, 4.25),
)

motivation_data = [
    ("Ground truth", gt),
    ("Pretrained CNN", cnn),
    ("Topology-inspired", topo),
]

im = None

for ax, (title, arr) in zip(
    axes,
    motivation_data,
):
    im = ax.imshow(
        arr,
        origin="upper",
        cmap=CMAP_FIELD,
        vmin=FIELD_VMIN,
        vmax=FIELD_VMAX,
        interpolation="nearest",
    )

    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_title(
        title,
        fontsize=18,
        fontweight="bold",
        pad=8,
        color=TEXT_COLOR,
    )

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#444444")


fig.suptitle(
    "Why look beyond pointwise fidelity?",
    fontsize=23,
    fontweight="bold",
    color=TITLE_COLOR,
    y=0.99,
)

fig.text(
    0.5,
    0.055,
    (
        "Sample 78: a conventional near-tie can still differ "
        "in persistent structural organization."
    ),
    ha="center",
    va="center",
    fontsize=15,
    color=MUTED_COLOR,
)

cbar = fig.colorbar(
    im,
    ax=axes,
    fraction=0.025,
    pad=0.025,
)

cbar.set_label(
    r"Wind speed (m s$^{-1}$)",
    fontsize=13,
)

cbar.ax.tick_params(labelsize=10)

fig.subplots_adjust(
    left=0.02,
    right=0.92,
    bottom=0.15,
    top=0.83,
    wspace=0.08,
)

fig.savefig(
    OUT_DIR / "sample078_motivation_strip.png",
    dpi=DPI,
    bbox_inches="tight",
    facecolor="white",
)

fig.savefig(
    OUT_DIR / "sample078_motivation_strip.pdf",
    bbox_inches="tight",
    facecolor="white",
)

plt.close(fig)


# ============================================================
# QUALITATIVE LAYOUT PREVIEW
# ============================================================

fig = plt.figure(
    figsize=(14.6, 7.3)
)

gs = fig.add_gridspec(
    2,
    4,
    height_ratios=[1.0, 1.0],
    hspace=0.20,
    wspace=0.08,
)

field_titles = [
    "GT",
    "Pretrained CNN",
    "Fine-tuning without\ntopology losses",
    "Topology-inspired",
]

field_arrays = [
    gt,
    cnn,
    ctl,
    topo,
]

top_axes = []

for j, (title, arr) in enumerate(
    zip(field_titles, field_arrays)
):
    ax = fig.add_subplot(gs[0, j])
    top_axes.append(ax)

    ax.imshow(
        arr,
        origin="upper",
        cmap=CMAP_FIELD,
        vmin=FIELD_VMIN,
        vmax=FIELD_VMAX,
        interpolation="nearest",
    )

    ax.set_title(
        title,
        fontsize=17,
        fontweight="bold",
        pad=7,
    )

    ax.set_axis_off()


# Empty lower-left cell under GT
ax_blank = fig.add_subplot(gs[1, 0])
ax_blank.axis("off")

error_titles = [
    "|CNN − GT|",
    "|No-topology FT − GT|",
    "|Topology-inspired − GT|",
]

error_arrays = [
    cnn_err,
    ctl_err,
    topo_err,
]

for j, (title, arr) in enumerate(
    zip(error_titles, error_arrays),
    start=1,
):
    ax = fig.add_subplot(gs[1, j])

    ax.imshow(
        arr,
        origin="upper",
        cmap=CMAP_ERROR,
        vmin=ERROR_VMIN,
        vmax=ERROR_VMAX,
        interpolation="nearest",
    )

    ax.set_title(
        title,
        fontsize=16,
        fontweight="bold",
        pad=7,
    )

    ax.set_axis_off()


fig.suptitle(
    "Sample 78 — qualitative field comparison",
    fontsize=23,
    fontweight="bold",
    color=TITLE_COLOR,
    y=0.985,
)

fig.text(
    0.5,
    0.025,
    (
        "All panels use the same fixed 160×160 topology-evaluation "
        "domain; field panels share one speed scale and error maps "
        "share one error scale."
    ),
    ha="center",
    fontsize=12.5,
    color=MUTED_COLOR,
)

fig.savefig(
    OUT_DIR / "sample078_qualitative_preview.png",
    dpi=DPI,
    bbox_inches="tight",
    facecolor="white",
)

fig.savefig(
    OUT_DIR / "sample078_qualitative_preview.pdf",
    bbox_inches="tight",
    facecolor="white",
)

plt.close(fig)


# ============================================================
# DONE
# ============================================================

print()
print("Wrote poster assets to:")
print(f"  {OUT_DIR}")
print()
for p in sorted(OUT_DIR.iterdir()):
    print(" ", p.name)

