#!/usr/bin/env python3
"""
Section 1, panel 3 - "PD & MT capture complementary structure"
Poster slot: 11.80 x 4.20 in

What the reader should get in 2 seconds:
    lower a threshold on wind speed -> regions A, B, C appear, then merge.
    The PD records how long each region lives; the MT records what merges
    into what.

Changes from the previous version
* One story told with three lettered, coloured features (A, B, C) that are
  the same in every panel: the field, the threshold frames, the PD and the
  MT.  Each region in a frame is coloured by the oldest peak it contains
  (the elder rule), so merging is visible, not just described.
* The PD and MT are computed from the toy field (0-dim superlevel-set
  persistence), so the picture is exactly consistent: every PD point's
  distance to the diagonal equals its coloured branch length in the MT.
* Short headers that fit their columns (the old ones overlapped), no tiny
  "tau" labels, and one plain-language summary line.
* All text >= 24 pt, checked automatically before saving.
"""

import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyArrowPatch
from scipy import ndimage

from poster_style import (
    FS, INK, MUTED, RULE, FRAME, FEATURE_COLORS, WIND_CMAP,
    new_figure, inch_axes, fx, fy, halo, rich_line, save_figure,
)


OUT_STEM = "pd_mt_primer_final"
FIG_W, FIG_H = 11.80, 4.20

LETTERS = ["A", "B", "C"]


# ============================================================
# TOY WIND-SPEED FIELD: three peaks of clearly different height
# ============================================================

n = 160
yy, xx = np.mgrid[-1:1:complex(n), -1:1:complex(n)]


def bump(cx, cy, h, w):
    # flat-topped bump: regions are already a readable size just below a peak
    r2 = ((xx - cx) ** 2 + (yy - cy) ** 2) / w
    return h * np.exp(-r2 ** 2.2)


# centres are (x, y) with y pointing *down* in the image
field = (
    bump(-0.40, -0.02, 1.00, 0.20)    # A  tallest, broad
    + bump(0.50, -0.50, 0.80, 0.13)   # B  far from A -> merges late
    + bump(0.18, 0.55, 0.64, 0.11)    # C  close to A -> merges early
    + 0.04
)


# ============================================================
# 0-DIM SUPERLEVEL-SET PERSISTENCE (union-find, elder rule)
# ============================================================

def superlevel_persistence(f):
    H, W = f.shape
    order = np.argsort(-f, axis=None)
    parent = -np.ones(H * W, dtype=np.int64)
    peak = {}                                  # root -> index of its peak

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    flat = f.ravel()
    pairs = []                                 # (peak_idx, birth, death, merged_into_peak)
    for i in order:
        parent[i] = i
        peak[i] = i
        r, c = divmod(int(i), W)
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1),
                       (-1, -1), (-1, 1), (1, -1), (1, 1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < H and 0 <= cc < W:
                j = rr * W + cc
                if parent[j] < 0:
                    continue
                ri, rj = find(i), find(j)
                if ri == rj:
                    continue
                # elder rule: the component with the lower peak dies here
                if flat[peak[ri]] < flat[peak[rj]]:
                    young_root, old_root = ri, rj
                else:
                    young_root, old_root = rj, ri
                young, old = peak[young_root], peak[old_root]
                if flat[young] > flat[i] + 1e-9:
                    pairs.append((young, flat[young], flat[i], old))
                parent[young_root] = old_root
    gmax = int(order[0])
    pairs.append((gmax, flat[gmax], flat.min(), None))    # essential class
    return pairs


pairs = [p for p in superlevel_persistence(field) if p[1] - p[2] > 0.02]
pairs.sort(key=lambda p: -p[1])                           # by birth: A, B, C
assert len(pairs) == 3, f"expected 3 features, got {len(pairs)}"

H, W = field.shape
feat = []
for k, (pk, b, d, into) in enumerate(pairs):
    r, c = divmod(int(pk), W)
    feat.append(dict(letter=LETTERS[k], color=FEATURE_COLORS[k],
                     peak=pk, rc=(r, c), birth=b, death=d, into=into))
peak_to_k = {f_["peak"]: k for k, f_ in enumerate(feat)}
for f_ in feat:
    f_["into_k"] = None if f_["into"] is None else peak_to_k[f_["into"]]
    print(f"{f_['letter']}: appears {f_['birth']:.3f}  merges {f_['death']:.3f}"
          + ("" if f_["into_k"] is None else f"  into {LETTERS[f_['into_k']]}"))

# Three thresholds: {A,B}  ->  {A,B,C} separate  ->  all merged
merges = sorted(f_["death"] for f_ in feat[1:])
# thresholds sit just above the next event so every region is big enough to see
tau_high = feat[2]["birth"] + 0.04            # A and B visible, C not yet
tau_mid = merges[-1] + 0.04                   # all three, still separate
tau_low = merges[0] - 0.30 * (merges[0] - field.min())   # all merged
assert feat[2]["birth"] > merges[-1], "C must appear before the first merge"
THRESHOLDS = [(tau_high, "high"), (tau_mid, "mid"), (tau_low, "low")]


def elder_labels(tau):
    """Label each superlevel component by the tallest feature it contains."""
    comp, _ = ndimage.label(field >= tau, structure=np.ones((3, 3)))
    out = np.full(field.shape, -1)
    present = []
    for k, f_ in enumerate(feat):                 # tallest first
        lab = comp[f_["rc"]]
        if lab == 0:
            continue
        mask = (comp == lab) & (out < 0)
        if mask.any():
            out[mask] = k
            present.append(k)
    return out, present


# ============================================================
# LAYOUT (inches from bottom-left)
# ============================================================
#
#  Wind speed   Lower the threshold      Persistence    Merge
#                                        diagram        tree
#  [field] -->  [high] [mid] [low]  -->  [PD]           [MT]
#                high   mid   low
#      PD: how long each region lives  ·  MT: which region merges into which

HEAD_Y = 3.36                     # bottom of the header band
SUMMARY_Y = 0.36

FIELD_X, FIELD_S = 0.16, 1.92
FIELD_Y = 1.08

FR_X0, FR_S, FR_GAP = 2.56, 1.25, 0.14
FR_Y = 1.50
fr_xs = [FR_X0 + i * (FR_S + FR_GAP) for i in range(3)]
FR_RIGHT = fr_xs[-1] + FR_S

PD_X, PD_S = 7.55, 1.66
PD_Y = 1.40
MT_X, MT_W = 9.62, 2.05
MT_Y, MT_H = 1.15, 2.00

fig = new_figure(FIG_W, FIG_H)


def header(x_center_in, text):
    fig.text(fx(fig, x_center_in), fy(fig, HEAD_Y), text, ha="center",
             va="bottom", fontsize=FS.PANEL_TITLE, fontweight="bold",
             color=INK, linespacing=1.05)


def arrow(x0, x1, y):
    fig.patches.append(FancyArrowPatch(
        (fx(fig, x0), fy(fig, y)), (fx(fig, x1), fy(fig, y)),
        transform=fig.transFigure, arrowstyle="-|>", mutation_scale=24,
        linewidth=2.2, color=RULE))


def letter(ax, r, c, s, size=FS.LABEL):
    ax.text(c, r, s, ha="center", va="center", fontsize=size,
            fontweight="bold", color=INK, path_effects=halo(4))


# --- 1. field --------------------------------------------------
ax = inch_axes(fig, FIELD_X, FIELD_Y, FIELD_S, FIELD_S)
ax.imshow(field, origin="upper", cmap=WIND_CMAP, interpolation="bilinear")
ax.set_axis_off()
for f_ in feat:
    letter(ax, *f_["rc"], f_["letter"])
header(FIELD_X + FIELD_S / 2, "Wind speed")

arrow(FIELD_X + FIELD_S + 0.08, FR_X0 - 0.08, FIELD_Y + FIELD_S / 2)

# --- 2. threshold frames --------------------------------------
cmap = ListedColormap(["white"] + FEATURE_COLORS)
# small regions get their letter beside them (pixels, +right) so it doesn't hide them
FRAME_LETTER_DX = {0: 0, 1: -30, 2: 40}
for x, (tau, lab) in zip(fr_xs, THRESHOLDS):
    a = inch_axes(fig, x, FR_Y, FR_S, FR_S)
    lbl, present = elder_labels(tau)
    a.imshow(lbl + 1, origin="upper", cmap=cmap, vmin=0, vmax=3,
             interpolation="nearest")
    a.set_xticks([]); a.set_yticks([])
    for s in a.spines.values():
        s.set_color(FRAME); s.set_linewidth(1.0)
    for k in present:
        r, c = feat[k]["rc"]
        dc = FRAME_LETTER_DX[k] if lbl.max() > 0 and (lbl == k).sum() < 0.12 * lbl.size else 0
        letter(a, r, c + dc, feat[k]["letter"])
    fig.text(fx(fig, x + FR_S / 2), fy(fig, FR_Y - 0.26), lab,
             ha="center", va="center", fontsize=FS.NOTE, color=MUTED)
header((FR_X0 + FR_RIGHT) / 2, "Lower the threshold")

arrow(FR_RIGHT + 0.10, PD_X - 0.50, FR_Y + FR_S / 2)

# --- value range shared by PD and MT --------------------------
lo = field.min() - 0.10
hi = field.max() + 0.08

# --- 3. persistence diagram -----------------------------------
pd = inch_axes(fig, PD_X, PD_Y, PD_S, PD_S)
pd.plot([lo, hi], [lo, hi], color=RULE, lw=1.6, ls=(0, (4, 3)), zorder=1)
for f_ in feat:
    b, d = f_["birth"], f_["death"]
    pd.plot([b, b], [d, b], color=f_["color"], lw=2.0, alpha=0.55, zorder=2)
    pd.scatter([b], [d], s=190, color=f_["color"], edgecolor="white",
               linewidth=2.0, zorder=3)
    if f_["into_k"] is None:          # A never merges: label to its right
        pd.text(b + 0.06, d, f_["letter"], ha="left", va="center",
                fontsize=FS.LABEL, fontweight="bold", color=INK)
    else:                             # others: label to the left
        pd.text(b - 0.06, d, f_["letter"], ha="right", va="center",
                fontsize=FS.LABEL, fontweight="bold", color=INK)
pd.set_xlim(lo, hi + 0.16); pd.set_ylim(lo, hi)   # room for A's label
pd.set_xticks([]); pd.set_yticks([])
for s in ("top", "right"):
    pd.spines[s].set_visible(False)
pd.set_xlabel("appears", fontsize=FS.AXIS, labelpad=6)
pd.set_ylabel("merges", fontsize=FS.AXIS, labelpad=6)
header(PD_X + PD_S / 2, "Persistence\ndiagram")

# --- 4. merge tree (branch decomposition) ---------------------
mt = inch_axes(fig, MT_X, MT_Y, MT_W, MT_H)
mt.set_xlim(0, 1); mt.set_ylim(lo, hi + 0.18)   # headroom for leaf letters
mt.set_axis_off()
xpos = {0: 0.42, 1: 0.84, 2: 0.10}             # A centre, B right, C left
root_y = field.min() - 0.01
for k, f_ in enumerate(feat):
    x = xpos[k]
    bottom = root_y if f_["into_k"] is None else f_["death"]
    mt.plot([x, x], [f_["birth"], bottom], color=f_["color"], lw=5,
            solid_capstyle="round", zorder=2)
    if f_["into_k"] is not None:                # connector to the parent branch
        mt.plot([x, xpos[f_["into_k"]]], [bottom, bottom], color=RULE,
                lw=2.4, zorder=1)
        mt.scatter([xpos[f_["into_k"]]], [bottom], s=60, color=INK, zorder=3)
    mt.text(x, f_["birth"] + 0.05, f_["letter"], ha="center", va="bottom",
            fontsize=FS.LABEL, fontweight="bold", color=INK)
header(MT_X + MT_W / 2, "Merge\ntree")

# --- summary line ---------------------------------------------
rich_line(fig, 0.5, fy(fig, SUMMARY_Y), [
    ("PD", "bold", INK), (": how long each region lives", "normal", INK),
    ("   ·   ", "normal", MUTED),
    ("MT", "bold", INK), (": which region merges into which", "normal", INK),
])

save_figure(fig, OUT_STEM)
