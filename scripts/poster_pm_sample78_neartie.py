#!/usr/bin/env python3
"""
Section 1, panel 2 - "Similar error, different structure"
Poster slot: 9.45 x 3.20 in

What the reader should get in 2 seconds:
    the two model outputs have almost the same pointwise error (numbers under
    the images are nearly equal) - yet they look structurally different.

Changes from the previous version
* Under each prediction, the pointwise error is printed, so "similar error"
  is visible instead of only claimed.  Optionally a second line shows the
  PD distance, so "different structure" is also a number.
* Labels no longer collide ("Topology-inspired" is 2.9 in wide at 24 pt;
  column pitch is now 2.95 in).
* The 20-pt "Sample 78 - fixed 160x160 domain" note is removed (below the
  floor); put it in the poster text box if you need it.
* Shared 0-20 m/s range; all text >= 24 pt, checked before saving.
"""

import numpy as np

from poster_style import (
    ROOT, FS, INK, MUTED,
    WIND_CMAP, WIND_VMIN, WIND_VMAX, WIND_UNITS,
    new_figure, inch_axes, fx, fy, style_image_axes,
    wind_colorbar, check_wind_range, save_figure,
)


# ============================================================
# CONFIG
# ============================================================

CNN_DIR = ROOT / "data_out_fixed" / "wind_mrhr_cnn"
TOPO_DIR = ROOT / "data_out" / "wind_finetune_candidateF_grad_E2_low_expanded2688"

SAMPLE_ID = 78
X0, Y0, PATCH = 0, 0, 160          # the fixed topology domain

# Line under each prediction.
#   "rmse"  -> RMSE of wind speed over the displayed 160x160 window
#              (computed here; printed to the console so you can check it)
#   "psnr"  -> use PSNR_TEXT below (e.g. your reported full-field PSNR)
ERROR_METRIC = "rmse"
PSNR_TEXT = {"cnn": None, "topo": None}          # e.g. "PSNR 31.2 dB"

# Optional second line: PD distance to ground truth for this sample
# (from your TTK evaluation).  Leave None to omit the line; the images
# then get taller automatically.
PD_DIST = {"cnn": None, "topo": None}            # e.g. 3.12, 1.79
PD_LABEL = "PD dist."                            # W2 / bottleneck: say which in caption

OUT_STEM = "sample078_neartie_final"

FIG_W, FIG_H = 9.45, 3.20


# ============================================================
# LOAD
# ============================================================

def speed(uv):
    uv = np.asarray(uv, dtype=np.float32)
    return np.hypot(uv[..., 0], uv[..., 1])


def row_of(idx_path, sample_id, name):
    idx = np.load(idx_path)
    hits = np.flatnonzero(idx == sample_id)
    if len(hits) != 1:
        raise RuntimeError(f"{name}: expected one sample {sample_id}, got {len(hits)}")
    return int(hits[0])


def crop(a):
    return a[Y0:Y0 + PATCH, X0:X0 + PATCH]


cnn_row = row_of(CNN_DIR / "idx.npy", SAMPLE_ID, "CNN")
topo_row = row_of(TOPO_DIR / "idx.npy", SAMPLE_ID, "Topology")

gt = crop(speed(np.load(CNN_DIR / "dataGT.npy", mmap_mode="r")[cnn_row]))
cnn = crop(speed(np.load(CNN_DIR / "dataSR.npy", mmap_mode="r")[cnn_row]))
topo = crop(speed(np.load(TOPO_DIR / "dataSR.npy", mmap_mode="r")[topo_row]))

check_wind_range(gt, cnn, topo)


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


err = {"cnn": rmse(cnn, gt), "topo": rmse(topo, gt)}
print(f"Sample {SAMPLE_ID}: speed RMSE on window  CNN {err['cnn']:.3f}  "
      f"topo {err['topo']:.3f} {WIND_UNITS}")


def error_text(key):
    if ERROR_METRIC == "psnr":
        if PSNR_TEXT[key] is None:
            raise ValueError("ERROR_METRIC='psnr' but PSNR_TEXT is not filled in")
        return PSNR_TEXT[key]
    return f"RMSE {err[key]:.2f} {WIND_UNITS}"


show_pd = all(v is not None for v in PD_DIST.values())


# ============================================================
# LAYOUT  (inches from bottom-left)
# ============================================================
#
#   [ image ]        [ image ]          [ image ]          | cbar
#   Ground truth     Pretrained CNN     Topology-inspired
#   reference        RMSE 0.84 m/s      RMSE 0.85 m/s
#                    PD dist. 3.12      PD dist. 1.79      (optional)

LINE = 0.37                              # one 24-pt line
n_lines = 3 if show_pd else 2
TEXT_H = n_lines * LINE + 0.06

PITCH = 2.95                             # >= widest label (2.89 in) + gap
LEFT = 0.10
IMG = min(FIG_H - TEXT_H - 0.10, 2.35)   # square image side
IMG_BOTTOM = TEXT_H + 0.02

cols = [LEFT + i * PITCH for i in range(3)]      # left edge of each column
CBAR_LEFT = FIG_W - 0.64                 # fixed, clear of the widest label

fig = new_figure(FIG_W, FIG_H)

panels = [
    (gt, "Ground truth", ["reference", None]),
    (cnn, "Pretrained CNN", [error_text("cnn"),
                             f"{PD_LABEL} {PD_DIST['cnn']:.2f}" if show_pd else None]),
    (topo, "Topology-inspired", [error_text("topo"),
                                 f"{PD_LABEL} {PD_DIST['topo']:.2f}" if show_pd else None]),
]

im = None
for c, (arr, name, lines) in zip(cols, panels):
    cx_in = c + PITCH / 2
    ax = inch_axes(fig, cx_in - IMG / 2, IMG_BOTTOM, IMG, IMG)
    im = ax.imshow(arr, origin="upper", cmap=WIND_CMAP,
                   vmin=WIND_VMIN, vmax=WIND_VMAX,
                   interpolation="nearest", aspect="equal")
    style_image_axes(ax)

    y = TEXT_H - LINE / 2
    fig.text(fx(fig, cx_in), fy(fig, y), name, ha="center", va="center",
             fontsize=FS.LABEL, fontweight="bold", color=INK)
    for ln in lines:
        y -= LINE
        if ln:
            fig.text(fx(fig, cx_in), fy(fig, y), ln, ha="center", va="center",
                     fontsize=FS.NOTE, color=MUTED)

wind_colorbar(fig, im, CBAR_LEFT, IMG_BOTTOM + 0.18, IMG_BOTTOM + IMG)

save_figure(fig, OUT_STEM)