#!/usr/bin/env python3
"""
Section 1, panel 1 - "Wind-field super-resolution"
Poster slot: 9.45 x 3.20 in

What the reader should get in 2 seconds:
    coarse (blocky) wind  --x5-->  sharp SR output,  compared with the true
    high-resolution field.

Changes from the previous version
* Shows a zoomed window instead of the full 100x100 field.  At 2.2 in wide,
  full-field 10 km cells are 0.02 in and invisible from 2 m; in a 32-cell
  window they are ~2 mm, so "coarse vs fine" is visible without grid lines.
* Plain labels: "Coarse input / SR output / Ground truth", "10 km cells" vs
  "2 km cells", "x5" on the arrow, "vs." between SR and GT (GT is compared
  with, not produced by, the model).
* Shared wind-speed range (0-20 m/s) with every other wind figure.
* All text >= 24 pt, checked automatically before saving.
"""

import numpy as np

from poster_style import (
    ROOT, FS, INK, MUTED, RULE,
    WIND_CMAP, WIND_VMIN, WIND_VMAX,
    new_figure, inch_axes, fx, fy, style_image_axes,
    wind_colorbar, check_wind_range, save_figure,
)
from matplotlib.patches import FancyArrowPatch, Rectangle


# ============================================================
# CONFIG
# ============================================================

DATA_DIR = ROOT / "data_out_fixed" / "wind_mrhr_cnn"

SAMPLE_ROW = 69          # row in dataIN/dataSR/dataGT (as before)

SCALE = 5                # MR -> HR factor (100 -> 500)
MR_KM, HR_KM = 10, 2

# Zoom window, in coarse (MR) cells.  WINDOW_MR = None shows the full field.
WINDOW_MR = 32           # 32 MR cells = 160 HR cells = 320 km
WIN_X0_MR = 0
WIN_Y0_MR = 0

# Locator inset: a small thumbnail of the full field, drawn in a corner of
# the coarse-input panel, with a box showing where the zoom window is.
# Answers "is this the whole domain / is it cherry-picked?" without text.
SHOW_LOCATOR = True
LOCATOR_SIZE = 0.72      # inches
LOCATOR_BOX = "#E34948"  # red outline (with a white halo for contrast)

OUT_STEM = "wind_sr_final"

FIG_W, FIG_H = 9.45, 3.20


# ============================================================
# LOAD
# ============================================================

def speed(uv):
    uv = np.asarray(uv, dtype=np.float32)
    return np.hypot(uv[..., 0], uv[..., 1])


data_in = np.load(DATA_DIR / "dataIN.npy", mmap_mode="r")
data_sr = np.load(DATA_DIR / "dataSR.npy", mmap_mode="r")
data_gt = np.load(DATA_DIR / "dataGT.npy", mmap_mode="r")

mr = speed(data_in[SAMPLE_ROW])
sr = speed(data_sr[SAMPLE_ROW])
gt = speed(data_gt[SAMPLE_ROW])
gt_full = gt                     # kept for the locator thumbnail

if WINDOW_MR is not None:
    y0, x0, n = WIN_Y0_MR, WIN_X0_MR, WINDOW_MR
    mr = mr[y0:y0 + n, x0:x0 + n]
    Y0, X0, N = y0 * SCALE, x0 * SCALE, n * SCALE
    sr = sr[Y0:Y0 + N, X0:X0 + N]
    gt = gt[Y0:Y0 + N, X0:X0 + N]

else:
    X0 = Y0 = 0
    N = gt.shape[0]

print(f"Sample row {SAMPLE_ROW}: MR {mr.shape}, SR {sr.shape}, GT {gt.shape}")
check_wind_range(mr, sr, gt)


# ============================================================
# LAYOUT  (inches from bottom-left)
# ============================================================
#
#   [ image ]  x5 -->  [ image ]   vs.   [ image ]  | cbar
#   Coarse input       SR output         Ground truth
#   10 km cells        2 km cells        2 km cells

IMG = 2.20                   # image side
GAP = 0.78                   # room for arrow / "vs."
LEFT = 0.12
IMG_BOTTOM = 0.86            # leaves two 24-pt text lines underneath
LINE1_Y = 0.58               # centre of bold label
LINE2_Y = 0.20               # centre of muted detail

xs = [LEFT + i * (IMG + GAP) for i in range(3)]
CBAR_LEFT = xs[-1] + IMG + 0.22

fig = new_figure(FIG_W, FIG_H)

panels = [
    (mr, "Coarse input", f"{MR_KM} km cells"),
    (sr, "SR output", f"{HR_KM} km cells"),
    (gt, "Ground truth", f"{HR_KM} km cells"),
]

im = None
axes = []
for x, (arr, name, detail) in zip(xs, panels):
    ax = inch_axes(fig, x, IMG_BOTTOM, IMG, IMG)
    axes.append(ax)
    im = ax.imshow(
        arr, origin="upper", cmap=WIND_CMAP,
        vmin=WIND_VMIN, vmax=WIND_VMAX,
        interpolation="nearest",     # keep coarse cells visibly blocky
        aspect="equal",
    )
    style_image_axes(ax)

    cx = fx(fig, x + IMG / 2)
    fig.text(cx, fy(fig, LINE1_Y), name, ha="center", va="center",
             fontsize=FS.LABEL, fontweight="bold", color=INK)
    fig.text(cx, fy(fig, LINE2_Y), detail, ha="center", va="center",
             fontsize=FS.NOTE, color=MUTED)


# --- locator inset (full field + window box) ------------------
if SHOW_LOCATOR and WINDOW_MR is not None:
    pad = 0.06
    loc = inch_axes(fig, xs[0] + IMG - LOCATOR_SIZE - pad,
                    IMG_BOTTOM + pad, LOCATOR_SIZE, LOCATOR_SIZE)
    loc.imshow(gt_full, origin="upper", cmap=WIND_CMAP,
               vmin=WIND_VMIN, vmax=WIND_VMAX, interpolation="antialiased")
    loc.set_xticks([]); loc.set_yticks([])
    for sp in loc.spines.values():
        sp.set_color("white"); sp.set_linewidth(2.5)
    for color, lw in (("white", 5.0), (LOCATOR_BOX, 2.6)):
        loc.add_patch(Rectangle((X0 - 0.5, Y0 - 0.5), N, N, fill=False,
                                edgecolor=color, linewidth=lw, clip_on=False,
                                zorder=5))


# --- x5 arrow (the model step) --------------------------------
mid_y = IMG_BOTTOM + IMG / 2
a0 = xs[0] + IMG + 0.10
a1 = xs[1] - 0.10
fig.patches.append(FancyArrowPatch(
    (fx(fig, a0), fy(fig, mid_y)), (fx(fig, a1), fy(fig, mid_y)),
    transform=fig.transFigure, arrowstyle="-|>", mutation_scale=26,
    linewidth=2.4, color=RULE,
))
fig.text(fx(fig, (a0 + a1) / 2), fy(fig, mid_y + 0.30), "×5",
         ha="center", va="center", fontsize=FS.ANNOT, fontweight="bold",
         color=INK)

# --- "vs." (comparison, not generation) -----------------------
fig.text(fx(fig, xs[1] + IMG + GAP / 2), fy(fig, mid_y), "vs.",
         ha="center", va="center", fontsize=FS.ANNOT, color=MUTED)

# --- colourbar -------------------------------------------------
wind_colorbar(fig, im, CBAR_LEFT, IMG_BOTTOM, IMG_BOTTOM + IMG)

save_figure(fig, OUT_STEM)
