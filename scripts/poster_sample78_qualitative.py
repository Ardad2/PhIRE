#!/usr/bin/env python3
"""
Section 4 - Qualitative example (Sample 78), one figure for the content area.

Output (in ~/PhIRE/figures/poster_sample78_final/):
    sample078_section4.png / .pdf      31.20 x 4.65 in   (X=0.95, Y=31.80)

Keep the Keynote headline ("Sample 78: ...") above it and the caption below.

The story, left to right:
    GT | Pretrained CNN | Reconstruction-only | Topology-inspired
      each column: wind-speed field + persistence diagram overlaid on GT's
      under each:  fidelity (RMSE, MAE)  -> similar
                   PD distance (W2,2)    -> not similar
    then: persistence survival (how many features outlive each threshold)

* Fields share one colour scale (same 0-20 m/s as Section 1).
* PDs: GT in every panel as open black rings, the method as filled dots in
  its Section 3 colour, so "closer to GT" reads directly.  D0 and D1 are
  combined and only pairs with persistence >= 3 m/s are drawn (display only;
  the audited distances use every strictly positive finite pair).
* Survival curves use the same colours; the column titles carry the line
  styles, so no separate legend box.

Validation carried over from the previous script: rows looked up via idx.npy,
GT vector fields must match exactly across runs, GT persistence diagrams must
match exactly across runs, PD distances read from the audited W22 sweep.
"""

import csv
import os
import sys
from pathlib import Path

import numpy as np

from poster_style import (
    ROOT, INK, MUTED, METHOD_COLORS,
    WIND_CMAP, WIND_VMIN, WIND_VMAX,
    new_figure, inch_axes, fx, fy, style_image_axes,
    wind_colorbar, check_wind_range, save_figure,
)
from poster_results import wrap, text_bottom

# ============================================================
# CONFIG
# ============================================================

CNN_DIR = ROOT / "data_out_fixed" / "wind_mrhr_cnn"
CONTROL_DIR = ROOT / "data_out" / "wind_finetune_candidateUV_expanded2688"
TOPO_DIR = ROOT / "data_out" / "wind_finetune_candidateF_grad_E2_low_expanded2688"

SAMPLE_ID = 78
X0, Y0, PATCH = 0, 0, 160                     # fixed topology-evaluation crop

AUDIT = Path(os.environ.get("AUDIT", str(Path.home() / "phire_runtime_audit_20260809_221548")))
W22_SWEEP = Path(os.environ.get("W22", str(AUDIT / "recompute_pd_w22"))) / "w22_full_sweep.csv"
sys.path.insert(0, str(AUDIT / "recompute_pd"))
import canonical_pd_pilot as canonical        # noqa: E402

CNN_RUN = "cnn"
CONTROL_RUN = "topology_finetuning/candidateUV_expanded2688_topology"
TOPO_RUN = "topology_finetuning/candidateF_grad_E2_low_expanded2688_topology"

# PD panels and survival curve show the high-persistence part only
MIN_PERSISTENCE = 3.0        # m/s (display threshold, as in the audited figure)

# Optional: full-field PSNR from your report tables, e.g. {1: 31.19, 2: 33.79, 3: 32.49}
# (keys: 1 = pretrained, 2 = reconstruction-only, 3 = topology-inspired).
# When filled, PSNR replaces MAE in the fidelity line.
PSNR_DB = {}

# Labels (match the Method pipeline wording)
NAMES = ["Ground truth", "Pretrained CNN", "Reconstruction-only", "Topology-inspired"]
COLORS = [INK, METHOD_COLORS["pretrained"], "#D08A2E", METHOD_COLORS["topology"]]
LINESTYLES = ["-", (0, (5, 3)), (0, (1.5, 2)), "-"]

OUT_STEM = "sample078_section4"
SUBDIR = "poster_sample78_final"
FIG_W, FIG_H = 31.20, 4.65

F_COL = 30          # column titles
F_SUB = 24          # sub-labels, numbers, ticks, legend
F_TITLE = 30        # survival title


# ============================================================
# LOAD + VALIDATE (same guards as the previous script)
# ============================================================

def require(path):
    if not Path(path).is_file():
        raise FileNotFoundError(str(path))


def repo_path(v):
    p = Path(v)
    return p if p.is_absolute() else ROOT / p


def load(directory):
    for n in ("idx.npy", "dataGT.npy", "dataSR.npy"):
        require(directory / n)
    idx = np.load(directory / "idx.npy").astype(int)
    hits = np.flatnonzero(idx == SAMPLE_ID)
    if len(hits) != 1:
        raise RuntimeError(f"{directory}: expected one row for sample {SAMPLE_ID}, got {len(hits)}")
    r = int(hits[0])
    gt = np.load(directory / "dataGT.npy", mmap_mode="r")[r]
    sr = np.load(directory / "dataSR.npy", mmap_mode="r")[r]
    return np.asarray(gt, np.float64), np.asarray(sr, np.float64), r


def speed(uv):
    return np.hypot(uv[..., 0], uv[..., 1])


def crop(a):
    return a[Y0:Y0 + PATCH, X0:X0 + PATCH]


def read_sweep():
    require(W22_SWEEP)
    with W22_SWEEP.open(newline="") as f:
        return {(r["run"], int(r["sample"])): r for r in csv.DictReader(f)}


def positive(D):
    D = np.asarray(D, np.float64).reshape(-1, 2)
    p = D[:, 1] - D[:, 0]
    if np.any(p < 0):
        raise RuntimeError("negative persistence found")
    D = D[p > 0]
    return D[np.lexsort((D[:, 1], D[:, 0]))] if len(D) else D


def read_pds(sweep, run):
    row = sweep[(run, SAMPLE_ID)]
    gt_pd, _ = canonical.read_pd(str(repo_path(row["gt_path"])))
    sr_pd, _ = canonical.read_pd(str(repo_path(row["sr_path"])))
    clean = lambda pd: {d: positive(pd[d]) for d in (0, 1)}
    return clean(gt_pd), clean(sr_pd), row


def metric(row, *keys):
    for k in keys:
        if row.get(k) not in (None, ""):
            return float(row[k])
    return None


gt_c, cnn_sr, r1 = load(CNN_DIR)
gt_k, ctl_sr, r2 = load(CONTROL_DIR)
gt_t, top_sr, r3 = load(TOPO_DIR)
if max(np.abs(gt_c - gt_k).max(), np.abs(gt_c - gt_t).max()) > 1e-6:
    raise RuntimeError("GT vector fields differ between runs")

fields = [crop(speed(a)) for a in (gt_c, cnn_sr, ctl_sr, top_sr)]
check_wind_range(*fields)
rmse = [None] + [float(np.sqrt(np.mean((f - fields[0]) ** 2))) for f in fields[1:]]
mae = [None] + [float(np.mean(np.abs(f - fields[0]))) for f in fields[1:]]

sweep = read_sweep()
pds, rows = [], []
gt_ref = None
for run in (CNN_RUN, CONTROL_RUN, TOPO_RUN):
    g, s, row = read_pds(sweep, run)
    if gt_ref is None:
        gt_ref = g
    elif any(not np.array_equal(gt_ref[d], g[d]) for d in (0, 1)):
        raise RuntimeError(f"GT persistence diagrams differ for {run}")
    pds.append(s)
    rows.append(row)
all_pds = [gt_ref] + pds
combined = [np.vstack([pd[d] for d in (0, 1) if len(pd[d])]) for pd in all_pds]
pers = [D[:, 1] - D[:, 0] for D in combined]
w22 = [None] + [metric(r, "w22_all") for r in rows]
w2inf = [None] + [metric(r, "w2inf_all", "w2_inf_all") for r in rows]
dbot = [None] + [metric(r, "bottleneck_all", "db_all", "dB_all") for r in rows]
# (symbol, values, decimals) - same symbols and order as the Section 3 PD panels
PD_METRICS = [(r"$W_{2,2}$", w22, 1), (r"$W_{2,\infty}$", w2inf, 1), (r"$d_B$", dbot, 2)]



print(f"Sample {SAMPLE_ID}  rows {r1}/{r2}/{r3}")
for k, name in enumerate(NAMES):
    shown = int(np.sum(pers[k] >= MIN_PERSISTENCE))
    print(f"  {name:20s} RMSE {'—' if rmse[k] is None else f'{rmse[k]:.3f}':>6}  "
          f"MAE {'—' if mae[k] is None else f'{mae[k]:.3f}':>6}  "
          f"pairs {len(pers[k]):5d} (shown {shown:3d})  "
          f"W22 {'—' if w22[k] is None else f'{w22[k]:.2f}'}")


# ============================================================
# FIGURE
# ============================================================

fig = new_figure(FIG_W, FIG_H)
R = fig.canvas.get_renderer()

SURV_W = 7.40
CBAR_COL = 0.95
LEFT = 0.06
cols_w = FIG_W - LEFT - SURV_W - CBAR_COL - 0.25
COL_GAP = 0.30
col_w = (cols_w - 3 * COL_GAP) / 4
DEATH_W = 0.36                                  # room for the rotated "death" label
TITLE_Y = FIG_H - 0.04
SUBLBL_Y = FIG_H - 0.74
IMG_TOP = SUBLBL_Y - 0.22
LINE_STEP = 0.42                                # maths lines are taller than plain ones
TEXT_BLOCK = 3 * LINE_STEP + 0.08               # fidelity + PD distances + % change
IMG = min((col_w - DEATH_W - 0.06) / 2, IMG_TOP - TEXT_BLOCK)
PAIR_OFF = (col_w - (2 * IMG + DEATH_W + 0.06)) / 2   # centre the pair in its column
IMG_BOTTOM = IMG_TOP - IMG
LINE1_Y = IMG_BOTTOM - 0.24                     # fidelity
LINE2_Y = LINE1_Y - LINE_STEP                   # three PD distances
LINE3_Y = LINE2_Y - LINE_STEP                   # their change vs. pretrained

# shared PD axis range (all methods, persistence >= threshold)
shown = [D[(D[:, 1] - D[:, 0]) >= MIN_PERSISTENCE] for D in combined]
pd_hi = max(float(D.max()) for D in shown if len(D)) * 1.04
pd_lo = min(0.0, min(float(D.min()) for D in shown if len(D)))
GT_OPEN = INK            # GT = black everywhere (open rings under a method)

for k in range(4):
    x = LEFT + k * (col_w + COL_GAP)
    cx = x + col_w / 2

    # column title + line sample (the titles double as the survival legend)
    t = fig.text(fx(fig, cx + 0.45), fy(fig, TITLE_Y), NAMES[k], ha="center", va="top",
                 fontsize=F_COL, fontweight="bold", color=INK)
    bb = t.get_window_extent(R)
    ty = (bb.y0 + bb.y1) / 2 / fig.dpi
    lax = inch_axes(fig, bb.x0 / fig.dpi - 0.85, ty - 0.1, 0.7, 0.2)
    lax.set_xlim(0, 1); lax.set_ylim(-1, 1); lax.set_axis_off()
    lax.plot([0, 1], [0, 0], color=COLORS[k], ls=LINESTYLES[k], lw=4,
             solid_capstyle="butt", dash_capstyle="butt")

    # wind-speed field
    ax = inch_axes(fig, x + PAIR_OFF, IMG_BOTTOM, IMG, IMG)
    im = ax.imshow(fields[k], cmap=WIND_CMAP, vmin=WIND_VMIN, vmax=WIND_VMAX,
                   origin="upper", interpolation="nearest")
    style_image_axes(ax)

    # persistence diagram: GT open black rings + method filled dots
    px = x + PAIR_OFF + IMG + DEATH_W + 0.06
    pax = inch_axes(fig, px, IMG_BOTTOM, IMG, IMG)
    pax.plot([pd_lo, pd_hi], [pd_lo, pd_hi], color="#BBBBBB", lw=1.4, ls=(0, (4, 3)), zorder=1)
    G = shown[0]
    if k == 0:
        pax.scatter(G[:, 0], G[:, 1], s=42, facecolor="none", edgecolor=INK,
                    linewidth=1.2, zorder=3)
    else:
        pax.scatter(G[:, 0], G[:, 1], s=42, facecolor="none", edgecolor=GT_OPEN,
                    linewidth=0.9, alpha=0.55, zorder=2)
        D = shown[k]
        pax.scatter(D[:, 0], D[:, 1], s=46, facecolor=COLORS[k], edgecolor=INK,
                    linewidth=0.6, zorder=3)
    pax.set_xlim(pd_lo, pd_hi); pax.set_ylim(pd_lo, pd_hi)
    pax.set_xticks([]); pax.set_yticks([])
    for sp in ("top", "right"):
        pax.spines[sp].set_visible(False)
    pax.spines["left"].set_color(MUTED); pax.spines["bottom"].set_color(MUTED)
    # axis names: "birth" inside the empty lower-right triangle, "death" outside left
    pax.text(0.97, 0.04, "birth →", transform=pax.transAxes, ha="right", va="bottom",
             fontsize=F_SUB, color=MUTED)
    fig.text(fx(fig, px - 0.06), fy(fig, IMG_BOTTOM + IMG / 2), "death →", rotation=90,
             ha="right", va="center", fontsize=F_SUB, color=MUTED)

    for xc, lbl in ((x + PAIR_OFF + IMG / 2, "Wind speed"),
                    (px + IMG / 2, "PD  (○ = GT)" if k == 0 else "PD vs. GT")):
        fig.text(fx(fig, xc), fy(fig, SUBLBL_Y), lbl, ha="center", va="center",
                 fontsize=F_SUB, color=MUTED)

    # numbers: fidelity (similar), then all three PD distances (not similar)
    if k == 0:
        l1, l2, l3 = "reference", None, None
    else:
        second = (f"PSNR {PSNR_DB[k]:.2f} dB" if k in PSNR_DB else f"MAE {mae[k]:.2f}")
        l1 = f"RMSE {rmse[k]:.2f}  ·  {second}"
        have = [(sym, v[k], v[1], nd) for sym, v, nd in PD_METRICS if v[k] is not None]
        l2 = "  ·  ".join(f"{sym} {val:.{nd}f}" for sym, val, _, nd in have) or None
        if k == 1:
            l3 = "baseline for % change"
        else:
            l3 = "  ·  ".join(f"{100 * (val - ref) / ref:+.0f}%" for _, val, ref, _ in have)
            l3 = l3.replace("-", "−")
    bold = "bold" if k == 3 else "normal"
    fig.text(fx(fig, cx), fy(fig, LINE1_Y), l1, ha="center", va="center",
             fontsize=F_SUB, color=MUTED if k == 0 else INK,
             style="italic" if k == 0 else "normal")
    if l2:
        fig.text(fx(fig, cx), fy(fig, LINE2_Y), l2, ha="center", va="center",
                 fontsize=F_SUB, fontweight=bold, color=INK)
    if l3:
        fig.text(fx(fig, cx), fy(fig, LINE3_Y), l3, ha="center", va="center",
                 fontsize=F_SUB, fontweight=bold,
                 color=MUTED if k == 1 else INK, style="italic" if k == 1 else "normal")
    if k == 0:   # say what the numbers under the other columns are
        fig.text(fx(fig, cx), fy(fig, LINE2_Y), "PD distance to GT:", ha="center",
                 va="center", fontsize=F_SUB, color=MUTED)
        fig.text(fx(fig, cx), fy(fig, LINE3_Y), "change vs. pretrained:", ha="center",
                 va="center", fontsize=F_SUB, color=MUTED)

wind_colorbar(fig, im, LEFT + cols_w + 0.14, IMG_BOTTOM, IMG_TOP)

# ---- survival curve ------------------------------------------------------
sx = FIG_W - SURV_W
title = fig.text(fx(fig, sx + 0.05), fy(fig, TITLE_Y),
                 wrap(fig, "Topology-inspired better matches the GT long-lived tail",
                      F_TITLE, SURV_W - 0.1, "bold"),
                 ha="left", va="top", fontsize=F_TITLE, fontweight="bold",
                 color=INK, linespacing=1.0)
ax_top = text_bottom(fig, title) - 0.18
ax_bottom = 0.95
ax_left = sx + 1.05
ax = inch_axes(fig, ax_left, ax_bottom, FIG_W - ax_left - 0.12, ax_top - ax_bottom)

allp = np.concatenate(pers)
lo, hi = MIN_PERSISTENCE, float(allp.max())
th = np.linspace(lo, hi, 300)
for p, c, ls, lw in zip(pers, COLORS, LINESTYLES, (2.8, 2.6, 3.0, 3.4)):
    counts = np.array([np.count_nonzero(p >= t) for t in th], float)
    counts[counts == 0] = np.nan          # end the curve where no features remain
    ax.step(th, counts, where="post", color=c, ls=ls, lw=lw)
ax.set_yscale("log")
ax.set_ylim(0.8, None)
ax.set_xlim(lo, hi)
ax.set_xlabel("Persistence at least (m/s)", fontsize=F_SUB, labelpad=4)
ax.set_ylabel("Features", fontsize=F_SUB, labelpad=4)
ax.tick_params(labelsize=F_SUB, length=5)
ax.grid(True, which="major", color="#E3E3E3", lw=1.2)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)

save_figure(fig, OUT_STEM, subdir=SUBDIR)
