#!/usr/bin/env python3
"""
Section 2 - Method: objective + three loss cards

Outputs (in ~/PhIRE/figures/poster_method/):
    method_objective.png      22.70 x 0.72 in   (place at X=5.20, Y=13.48)
    method_card_grad.png      10.23 x 2.72 in   (place at X=0.95,  Y=14.38)
    method_card_cv.png        10.23 x 2.72 in   (place at X=11.43, Y=14.38)
    method_card_pers.png      10.23 x 2.72 in   (place at X=21.91, Y=14.38)

Why whole cards instead of separate title / text / diagram / formula boxes:
at the required sizes a 38-pt title is ~0.55 in tall and two lines of 28-pt
text are ~0.85 in, so the planned 0.40 in and 0.65 in Slides boxes overflow.
Rendering each card as one image keeps every element on the same grid and
lets the audit check the whole card.  The footnote stays as Slides text.

The three diagrams share one visual language: the same 1-D wind-speed
profile, ground truth (GT) solid dark, SR prediction dashed green, so each
card only changes *what is compared*:
    gradient  -> slopes across a front
    CV        -> values at GT critical points
    PERS      -> the rise from a pair's death to its birth (persistence)

Formulas match the Candidate F implementation (MSE losses at fixed GT TTK
pair locations); see the comment above CARDS.
"""

import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from poster_style import (
    INK, MUTED, METHOD_COLORS,
    new_figure, inch_axes, fx, fy, save_figure,
)


# ============================================================
# CONTENT  (edit here)
# ============================================================

TITLE_COLOR = "#0182B9"      # poster heading blue
BODY_COLOR = "#52575A"       # poster body grey
FRAME_COLOR = "#255B4C"      # poster box outline (Tulane green)
DRAW_FRAME = True            # set False to keep your Slides card boxes instead

MATH = "cm"                  # Computer Modern for maths (matches LaTeX look)

L = r"\mathcal{L}"
SUB = {
    "uv": r"_{\mathrm{uv}}",
    "grad": r"_{\mathrm{grad}}",
    "cv": r"_{\mathrm{TTK\text{-}CV}}",
    "pers": r"_{\mathrm{TTK\text{-}PERS}}",
}

OBJECTIVE = (rf"${L} = {L}{SUB['uv']} + 0.05\,{L}{SUB['grad']}"
             rf" + 0.004\,{L}{SUB['cv']} + 0.002\,{L}{SUB['pers']}$")

# Formulas verified against the Candidate F implementation:
#   L_grad     = F.mse_loss(|grad speed(SR)|, |grad speed(GT)|)
#   L_TTK-CV   = 0.5 * (masked_mse(SR@birth, GT birth_val) + masked_mse(SR@death, GT death_val))
#   L_TTK-PERS = masked_mse(|SR@death - SR@birth|, GT persistence)
# where the birth/death locations are GT TTK pairs, precomputed and fixed.
HAT_B, HAT_D = r"\hat{b}", r"\hat{d}"

CARDS = {
    "grad": dict(
        title=rf"Gradient Structure (${L}{SUB['grad']}$)",
        body="Matches wind-speed gradient magnitude, encouraging sharper "
             "fronts and fine-scale transitions.",
        formula=rf"${L}{SUB['grad']} = \mathrm{{MSE}}\,(\Vert\nabla\hat{{s}}\Vert,\ \Vert\nabla s\Vert)$",
        gloss=r"$s$: GT wind speed,   $\hat{s}$: SR prediction",
    ),
    "cv": dict(
        title=rf"Critical-Value Supervision (${L}{SUB['cv']}$)",
        body="Matches predicted scalar values at fixed GT persistence-pair endpoints.",
        # 0.5 rather than a stacked 1/2: a \frac shrinks the digits below 24 pt
        formula=(rf"${L}{SUB['cv']} = 0.5\,[\mathrm{{MSE}}({HAT_B},b)"
                 rf" + \mathrm{{MSE}}({HAT_D},d)]$"),
        gloss=rf"$b, d$: GT endpoint values,   ${HAT_B}, {HAT_D}$: SR values",
    ),
    "pers": dict(
        title=rf"Persistence Supervision (${L}{SUB['pers']}$)",
        body="Matches the predicted scalar contrast of each fixed GT persistence pair.",
        formula=rf"${L}{SUB['pers']} = \mathrm{{MSE}}\,(\hat{{p}},\ p)$",
        gloss=rf"$\hat{{p}} = |{HAT_D} - {HAT_B}|$,   $p$: GT persistence",
    ),
}

# Sizes (pt)
F_EQ = 42
F_TITLE = 38
F_BODY = 28
F_FORMULA = 26
F_GLOSS = 24
F_LABEL = 24

CARD_W, CARD_H = 10.23, 2.72
EQ_W, EQ_H = 22.70, 0.72

GT_COLOR = INK
SR_COLOR = METHOD_COLORS["topology"]
MARK = "#1F1F1F"


# ============================================================
# OBJECTIVE
# ============================================================

fig = new_figure(EQ_W, EQ_H)
fig.text(0.5, 0.5, OBJECTIVE, ha="center", va="center", fontsize=F_EQ,
         color=INK, math_fontfamily=MATH)
save_figure(fig, "method_objective", subdir="poster_method")


# ============================================================
# SHARED 1-D PROFILE  (toy wind speed along a transect)
# ============================================================

x = np.linspace(0, 1, 400)


def profile(sharp=1.0, amp=1.0, p2=0.62):
    """Two peaks separated by a valley; `sharp` < 1 blurs the front and
    `p2` sets the second peak's height (lower in SR = weaker feature)."""
    front = 1 / (1 + np.exp(-(x - 0.22) * 38 * sharp))          # steep rise
    fall = np.exp(-((x - 0.36) / (0.16 / sharp**0.3)) ** 2)
    peak2 = p2 * np.exp(-((x - 0.76) / (0.09 / sharp**0.3)) ** 2)
    base = 0.18
    s = base + amp * (0.80 * front * (0.35 + 0.65 * fall) + peak2)
    return s


s_gt = profile(sharp=1.0, amp=1.0)
s_sr = profile(sharp=0.32, amp=0.84, p2=0.50)

# GT critical points (1-D): maxima and the valley between the two peaks
i_max1 = int(np.argmax(np.where(x < 0.55, s_gt, -1)))
i_max2 = int(np.argmax(np.where(x > 0.55, s_gt, -1)))
i_val = i_max1 + int(np.argmin(s_gt[i_max1:i_max2]))
crit = [i_max1, i_val, i_max2]
i_front = int(np.argmax(np.gradient(s_gt)))


GT_STYLE = dict(color=GT_COLOR, lw=3.0, solid_capstyle="round")
SR_STYLE = dict(color=SR_COLOR, lw=3.0, ls=(0, (3.2, 1.8)))


def profile_axes(fig, left, bottom, w, h):
    ax = inch_axes(fig, left, bottom, w, h)
    ax.set_axis_off()
    return ax


def draw_profiles(ax, lo, hi, x_right=None, pad_top=0.14, pad_bot=0.06):
    """Plot GT and SR on the window [lo, hi]; x_right extends the axes for labels."""
    m = (x >= lo) & (x <= hi)
    ax.plot(x[m], s_gt[m], zorder=3, **GT_STYLE)
    ax.plot(x[m], s_sr[m], zorder=3, **SR_STYLE)
    y0 = min(s_gt[m].min(), s_sr[m].min())
    y1 = max(s_gt[m].max(), s_sr[m].max())
    span = y1 - y0
    ax.set_xlim(lo - 0.01, x_right if x_right else hi + 0.01)
    ax.set_ylim(y0 - pad_bot * span, y1 + pad_top * span)


def key(fig, left, y, w):
    """One-row key under the diagram:  ── GT     ‒ ‒ SR"""
    kax = inch_axes(fig, left, y - 0.17, w, 0.34)
    kax.set_xlim(0, w); kax.set_ylim(-1, 1); kax.set_axis_off()
    kax.plot([0.02, 0.42], [0, 0], **GT_STYLE)
    kax.text(0.52, 0, "GT", fontsize=F_LABEL, va="center", ha="left", color=INK)
    kax.plot([1.30, 1.70], [0, 0], **SR_STYLE)
    kax.text(1.80, 0, "SR", fontsize=F_LABEL, va="center", ha="left", color=INK)


def dot(ax, i, s, color, ms=10):
    ax.plot(x[i], s[i], "o", ms=ms, color=color, mec="white", mew=1.8, zorder=5,
            clip_on=False)   # peak dots may sit in the headroom above the axes


def diag_grad(ax):
    """Zoom on the front: GT rises steeply, SR is smeared; compare slopes there."""
    draw_profiles(ax, 0.02, 0.44, pad_top=0.10, pad_bot=0.10)
    i = i_front
    # shaded band marks the local front region where slopes are compared
    ax.axvspan(x[i] - 0.075, x[i] + 0.075, color="#DCE9EE", lw=0, zorder=0)
    # tangent segments of equal vertical extent: the steeper one is shorter
    rise = 0.26
    for s, c in ((s_gt, GT_COLOR), (s_sr, SR_COLOR)):
        slope = np.gradient(s, x)[i]
        half = rise / slope
        xs = np.array([x[i] - half, x[i] + half])
        ax.plot(xs, s[i] + slope * (xs - x[i]), color=c, lw=7.0,
                solid_capstyle="round", alpha=0.85, zorder=4)


def diag_cv(ax):
    """Same GT pair as the persistence card: compare values at b and d."""
    ib, id_ = i_max2, i_val
    draw_profiles(ax, 0.44, 1.0, x_right=1.10, pad_top=0.04, pad_bot=0.30)
    for i in (ib, id_):
        ax.plot([x[i], x[i]], [s_sr[i], s_gt[i]], color=MARK, lw=2.4,
                ls=(0, (1, 1.3)), zorder=2)
        dot(ax, i, s_gt, MARK)
        dot(ax, i, s_sr, SR_COLOR, ms=9)
    ax.text(x[ib] - 0.07, s_gt[ib], "b", fontsize=F_LABEL, ha="right",
            va="center", color=INK, style="italic")
    ax.text(x[id_] - 0.03, min(s_gt[id_], s_sr[id_]) - 0.02, "d", fontsize=F_LABEL,
            ha="right", va="top", color=INK, style="italic")


def diag_pers(ax):
    """Pair born at peak b, dying at valley d: compare the rise p vs p-hat."""
    ib, id_ = i_max2, i_val
    draw_profiles(ax, 0.44, 1.0, x_right=1.34, pad_top=0.04, pad_bot=0.30)
    dot(ax, ib, s_gt, MARK)
    dot(ax, id_, s_gt, MARK)
    ax.text(x[ib] - 0.07, s_gt[ib], "b", fontsize=F_LABEL, ha="right",
            va="center", color=INK, style="italic")
    ax.text(x[id_] - 0.03, s_gt[id_] - 0.02, "d", fontsize=F_LABEL, ha="right",
            va="top", color=INK, style="italic")
    for xb, s, c, lab, ha, dx in ((1.07, s_gt, GT_COLOR, r"$p$", "right", -0.03),
                                  (1.19, s_sr, SR_COLOR, r"$\hat{p}$", "left", 0.03)):
        ax.add_patch(FancyArrowPatch((xb, s[id_]), (xb, s[ib]), arrowstyle="|-|,widthA=0.5,widthB=0.5",
                                     mutation_scale=14, lw=3.0, color=c, zorder=4,
                                     shrinkA=0, shrinkB=0))
        for i in (ib, id_):
            ax.plot([x[i], xb], [s[i], s[i]], color=c, lw=1.5, ls=(0, (1, 1.3)), zorder=2)
        ax.text(xb + dx, (s[ib] + s[id_]) / 2, lab, fontsize=F_LABEL, ha=ha,
                va="center", color=INK, math_fontfamily=MATH)


DIAGRAMS = {"grad": diag_grad, "cv": diag_cv, "pers": diag_pers}


# ============================================================
# CARDS
# ============================================================
#
#  ┌───────────────────────────────────────────────────────┐
#  │ Title (38 pt, blue)                                    │
#  │ Explanation, 28 pt, wraps to two lines                 │
#  │ [ 1-D diagram ]   L = formula (32 pt)                  │
#  │   GT / SR         gloss (24 pt)                        │
#  └───────────────────────────────────────────────────────┘

PAD_X = 0.15
TITLE_TOP = CARD_H - 0.10
BODY_W = CARD_W - 2 * PAD_X
DIAG = dict(left=0.22, bottom=0.42, w=3.15)   # height is set from the text above
KEY_Y = 0.23                                  # centre of the GT/SR key and gloss row
LABEL_HEADROOM = 0.19                         # half a 24-pt label + margin
FORM_X = 3.55


def wrap_to_width(fig, text, size, width_in):
    """Greedy word wrap using real text widths."""
    r = fig.canvas.get_renderer()
    words, lines, cur = text.split(), [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        t = fig.text(0, 0, trial, fontsize=size)
        ok = t.get_window_extent(r).width / fig.dpi <= width_in
        t.remove()
        if ok or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    lines.append(cur)
    return "\n".join(lines)


for name, c in CARDS.items():
    fig = new_figure(CARD_W, CARD_H)
    if DRAW_FRAME:
        bg = fig.add_axes([0, 0, 1, 1])
        bg.set_axis_off()
        bg.add_patch(FancyBboxPatch((0.004, 0.006), 0.992, 0.988, transform=bg.transAxes,
                                    boxstyle="square,pad=0", facecolor="white",
                                    edgecolor=FRAME_COLOR, linewidth=1.5))

    R = fig.canvas.get_renderer()

    def bottom_in(t):
        """Bottom edge of a text item in inches, measured with the fonts on
        *this* machine (Arial vs Liberation Sans differ slightly)."""
        return t.get_window_extent(R).y0 / fig.dpi

    title = fig.text(fx(fig, PAD_X), fy(fig, TITLE_TOP), c["title"], ha="left",
                     va="top", fontsize=F_TITLE, fontweight="bold",
                     color=TITLE_COLOR, math_fontfamily=MATH)

    body_text = wrap_to_width(fig, c["body"], F_BODY, BODY_W)
    body = fig.text(fx(fig, PAD_X), fy(fig, bottom_in(title) - 0.06), body_text,
                    ha="left", va="top", fontsize=F_BODY, color=BODY_COLOR,
                    linespacing=1.08)

    # Everything below is laid out from where the body text actually ends.
    row_top = bottom_in(body) - 0.07
    diag_top = row_top - LABEL_HEADROOM          # room for the b label at the peak
    ax = profile_axes(fig, DIAG["left"], DIAG["bottom"], DIAG["w"],
                      diag_top - DIAG["bottom"])
    DIAGRAMS[name](ax)
    key(fig, DIAG["left"] + 0.55, KEY_Y, 2.4)

    gloss = fig.text(fx(fig, FORM_X), fy(fig, KEY_Y), c["gloss"], ha="left",
                     va="center", fontsize=F_GLOSS, color=BODY_COLOR,
                     math_fontfamily=MATH)
    gloss_top = gloss.get_window_extent(R).y1 / fig.dpi
    # formula centred in the space left between the body text and the gloss
    fig.text(fx(fig, FORM_X), fy(fig, (row_top + gloss_top) / 2), c["formula"],
             ha="left", va="center", fontsize=F_FORMULA, color=INK,
             math_fontfamily=MATH)

    save_figure(fig, f"method_card_{name}", subdir="poster_method")
