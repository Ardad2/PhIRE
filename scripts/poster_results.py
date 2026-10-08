#!/usr/bin/env python3
"""
Section 3 - Results: three figures at their exact poster slots.

Outputs (in ~/PhIRE/figures/poster_results/):
    results_pd_agreement.png     31.20 x 5.45 in   (X=0.95,  Y=18.83)
    results_broader_effects.png  20.55 x 5.50 in   (X=0.95,  Y=24.53)
    results_mt_agreement.png     10.40 x 5.50 in   (X=21.75, Y=24.53)

What changed from the old figures
* Built at placed size: the old 15-17 in figures were shrunk ~2x, so their
  text printed at ~8-12 pt.  Everything here is >= 24 pt, audited.
* Each figure opens with a 38-pt blue headline that states the finding
  (VIS guideline: informative headlines), matching the Section 1 headings.
* Colours are the poster's method colours: grey = pretrained,
  hatched orange = control, green = topology-inspired.
* PD figure: values sit on the bars, so the y-axis is dropped; the legend and
  the joint 150/168 result move to a right-hand column.
* Broader effects: PSNR becomes its own row (in dB, no bar) instead of a
  separate callout box; metric names are rewritten in plain words, grouped by
  the same families as the Method pipeline (Fidelity, Wind power, Gradients,
  Connectivity).
* MT figure: two stacked bars and a one-line mean-MT summary.

All numbers are copied unchanged from the previous scripts (DATA sections).
"""

import numpy as np
from matplotlib.patches import Rectangle

from poster_style import (
    INK, MUTED, METHOD_COLORS,
    new_figure, inch_axes, fx, fy, save_figure,
)

SUBDIR = "poster_results"

# ============================================================
# SHARED STYLE
# ============================================================

HEAD_COLOR = "#0182B9"          # poster heading blue
F_HEAD = 38                     # figure headline (matches Section 1 headings)
F_SUB = 26                      # one-line explanation under the headline
F_PANEL = 28                    # panel titles
F_VALUE = 26                    # numbers on bars
F_SMALL = 24                    # ticks, notes, legend

C_PRE = METHOD_COLORS["pretrained"]
C_CTL_FILL, C_CTL_EDGE = "#F2C27A", "#B8792A"
C_OURS = METHOD_COLORS["topology"]
C_WORSE = "#A6A6A6"             # "comparison model better"
C_GRID = "#E3E3E3"

NAME_PRE = "Pretrained CNN"
NAME_CTL = "Reconstruction-only fine-tuning (control)"
NAME_OURS = "Topology-inspired fine-tuning"

PAD = 0.12                      # outer margin (in)


def wrap(fig, text, size, width_in, weight="normal"):
    """Greedy word wrap using real rendered widths."""
    r = fig.canvas.get_renderer()
    out, cur = [], ""
    for word in text.split():
        trial = (cur + " " + word).strip()
        t = fig.text(0, 0, trial, fontsize=size, fontweight=weight)
        ok = t.get_window_extent(r).width / fig.dpi <= width_in
        t.remove()
        if ok or not cur:
            cur = trial
        else:
            out.append(cur)
            cur = word
    out.append(cur)
    return "\n".join(out)


def text_bottom(fig, t):
    return t.get_window_extent(fig.canvas.get_renderer()).y0 / fig.dpi


def headline(fig, title, sub, width_in):
    """Blue headline + grey one-liner, top-left.  Returns the y (in) below them."""
    W, H = fig.get_size_inches()
    h = fig.text(fx(fig, PAD), fy(fig, H - 0.06), wrap(fig, title, F_HEAD, width_in, "bold"),
                 ha="left", va="top", fontsize=F_HEAD, fontweight="bold",
                 color=HEAD_COLOR, linespacing=1.0)
    if not sub:
        return text_bottom(fig, h)
    s = fig.text(fx(fig, PAD), fy(fig, text_bottom(fig, h) - 0.06),
                 wrap(fig, sub, F_SUB, width_in), ha="left", va="top",
                 fontsize=F_SUB, color=MUTED, linespacing=1.05)
    return text_bottom(fig, s)


def signed(p, unit="%", nd=1):
    return f"{'+' if p >= 0 else '−'}{abs(p):.{nd}f}{unit}"


# ============================================================
# FIGURE 1 - PD AGREEMENT                      31.20 x 5.45 in
# ============================================================

N_FIELDS = 168
PD_PANELS = [
    # title,                           (pretrained, control, ours), (fields lower vs pre, vs ctl)
    (r"$W_{2,2}$  ·  Euclidean",       (24.552, 27.509, 17.832), (166, 168)),
    (r"$W_{2,\infty}$  ·  $L_\infty$", (19.152, 21.062, 14.326), (167, 168)),
    (r"Bottleneck $d_B$",              (3.124, 3.288, 2.150),    (151, 156)),
]
JOINT_VS_PRE, JOINT_VS_CTL = 150, 156

PD_TITLE = "Topology-inspired fine-tuning lowers PD distance by 25–31%"
PD_SUB = ("Reconstruction-only fine-tuning (the control) makes PD agreement worse, "
          "so the gain does not come from fine-tuning alone.")
PD_TICKS = ["Pretrained", "Recon.-\nonly", "Topology-\ninspired"]


def pd_figure():
    W, H = 31.20, 5.45
    fig = new_figure(W, H)
    RIGHT_COL = 7.30                                 # legend + joint result
    panels_w = W - 2 * PAD - RIGHT_COL - 0.40
    y_top = headline(fig, PD_TITLE, PD_SUB, W - 2 * PAD) - 0.12

    GAP = 0.55
    pw = (panels_w - 2 * GAP) / 3
    FIELD_Y = PAD + 0.18                             # centre of the fieldwise line
    TICK_H = 0.78                                    # two lines of 24-pt ticks
    ax_bottom = FIELD_Y + 0.26 + TICK_H
    ax_top = y_top - 0.50                            # room for the panel title
    ax_h = ax_top - ax_bottom

    for k, (title, (pre, ctl, ours), (w_pre, w_ctl)) in enumerate(PD_PANELS):
        x0 = PAD + k * (pw + GAP)
        ax = inch_axes(fig, x0, ax_bottom, pw, ax_h)
        xs = np.array([0, 1, 2])
        # headroom: control value label + bracket + bracket label above the tallest bar
        top = ctl / (1 - 1.05 / ax_h)
        ax.set_ylim(0, top)
        ax.set_xlim(-0.6, 2.6)
        bw = 0.62
        ax.bar(0, pre, bw, color=C_PRE, edgecolor="none", zorder=2)
        ax.bar(1, ctl, bw, color=C_CTL_FILL, edgecolor=C_CTL_EDGE, hatch="//",
               lw=1.6, zorder=2)
        ax.bar(2, ours, bw, color=C_OURS, edgecolor="none", zorder=2)
        ax.axhline(pre, color=INK, ls=(0, (4, 3)), lw=1.6, zorder=3)
        ax.axhline(0, color=INK, lw=1.4, zorder=3)

        u = top / ax_h                                # data units per inch
        ax.text(0, pre + 0.06 * u, f"{pre:.2f}", ha="center", va="bottom",
                fontsize=F_VALUE, color=INK, zorder=4,
                bbox=dict(fc="white", ec="none", pad=1))
        ax.text(1, ctl + 0.06 * u, f"{ctl:.2f} ({signed(100 * (ctl - pre) / pre, '%', 0)})",
                ha="center", va="bottom", fontsize=F_SMALL, color=INK, zorder=4)
        # headline number for our bar, inside the bar in white
        ax.text(2, ours - 0.10 * u, f"{ours:.2f}\n{signed(100 * (ours - pre) / pre)}",
                ha="center", va="top", fontsize=F_VALUE, fontweight="bold",
                color="white", linespacing=1.05, zorder=4)

        # bracket: ours vs control
        yb = ctl + 0.55 * u
        tick = 0.10 * u
        ax.plot([1, 1, 2, 2], [yb - tick, yb, yb, yb - tick], color=MUTED, lw=1.6,
                zorder=3, clip_on=False)
        ax.text(1.5, yb + 0.05 * u, f"{signed(100 * (ours - ctl) / ctl)} vs. control",
                ha="center", va="bottom", fontsize=F_SMALL, color=MUTED, zorder=4)

        ax.set_xticks(xs, PD_TICKS, fontsize=F_SMALL, linespacing=1.0)
        ax.tick_params(axis="x", length=0, pad=6)
        ax.set_yticks([])
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_visible(False)

        fig.text(fx(fig, x0 + pw / 2), fy(fig, ax_top + 0.06), title, ha="center",
                 va="bottom", fontsize=F_PANEL, fontweight="bold", color=INK)
        fig.text(fx(fig, x0 + pw / 2), fy(fig, FIELD_Y),
                 f"Lower on {w_pre}/{N_FIELDS} fields vs. pretrained",
                 ha="center", va="center", fontsize=F_SMALL, color=INK)

    # ---- right column: legend + joint result ------------------------
    rx = W - PAD - RIGHT_COL
    y = y_top - 0.05
    items = [
        ("patch", dict(fc=C_PRE, ec="none"), NAME_PRE),
        ("patch", dict(fc=C_CTL_FILL, ec=C_CTL_EDGE, hatch="//", lw=1.6), NAME_CTL),
        ("patch", dict(fc=C_OURS, ec="none"), NAME_OURS),
        ("line", None, "Pretrained level"),
    ]
    lax = inch_axes(fig, rx, 0, RIGHT_COL, H)
    lax.set_xlim(0, RIGHT_COL); lax.set_ylim(0, H); lax.set_axis_off()
    for kind, style, label in items:
        label = wrap(fig, label, F_SMALL, RIGHT_COL - 0.75)
        n = label.count("\n") + 1
        cy = y - 0.19
        if kind == "patch":
            lax.add_patch(Rectangle((0.02, cy - 0.14), 0.50, 0.28, **style))
        else:
            lax.plot([0.02, 0.52], [cy, cy], color=INK, ls=(0, (4, 3)), lw=1.6)
        t = fig.text(fx(fig, rx + 0.68), fy(fig, y), label, ha="left", va="top",
                     fontsize=F_SMALL, color=INK, linespacing=1.05)
        y = text_bottom(fig, t) - 0.12

    y -= 0.10
    lax.plot([0, RIGHT_COL], [y, y], color=C_GRID, lw=1.5)
    t = fig.text(fx(fig, rx), fy(fig, y - 0.12), f"{JOINT_VS_PRE}/{N_FIELDS}",
                 ha="left", va="top", fontsize=46, fontweight="bold", color=INK)
    t2 = fig.text(fx(fig, rx), fy(fig, text_bottom(fig, t) - 0.06),
                  wrap(fig, f"fields with all three PD distances lower vs. pretrained "
                            f"({JOINT_VS_CTL}/{N_FIELDS} vs. control)", F_SMALL,
                       RIGHT_COL - 0.05),
                  ha="left", va="top", fontsize=F_SMALL, color=INK, linespacing=1.05)

    save_figure(fig, "results_pd_agreement", subdir=SUBDIR)


# ============================================================
# FIGURE 2 - BROADER EFFECTS                   20.55 x 5.50 in
# ============================================================

# family, label, vs pretrained (%), vs control (%)   [oriented: + = ours better]
BE_ROWS = [
    ("Fidelity", "PSNR (dB)", None, None),               # filled from PSNR below
    ("Fidelity", "SSIM", 7.0, -2.5),
    ("Fidelity", "Speed error (MAE)", 15.0, -19.0),
    ("Wind power", "Mean bias", 36.0, 27.7),
    ("Wind power", "Pixel error (MAE)", 17.7, -19.9),
    ("Wind power", r"Distribution ($W_1$)", 45.4, 37.2),
    ("Gradients", r"Distribution ($W_1$)", 40.7, 44.7),
    ("Connectivity", r"Component curve ($L_1$)", 29.7, 42.8),
]
PSNR = {"pretrained": 31.1925, "control": 33.7892, "ours": 32.4949}

BE_TITLE = "Fidelity gains retained, broader structure improved"
BE_SUB = None     # the direction cue sits under each panel instead


def be_figure():
    W, H = 20.55, 5.50
    fig = new_figure(W, H)
    y_top = headline(fig, BE_TITLE, BE_SUB, W - 2 * PAD) - 0.10

    LABEL_W = 6.15
    GAP = 0.55
    main_w = 7.35
    ctl_w = W - 2 * PAD - LABEL_W - main_w - GAP - 0.05
    x_main = PAD + LABEL_W
    x_ctl = x_main + main_w + GAP

    ax_top = y_top - 0.50                       # panel titles above
    ax_bottom = PAD + 0.42                      # direction cue below
    ax_h = ax_top - ax_bottom

    # rows with extra space between families
    ys, y, prev = [], 0.0, None
    for fam, *_ in BE_ROWS:
        if prev is not None and fam != prev:
            y -= 0.35
        ys.append(y)
        y -= 1.0
        prev = fam
    y_lo, y_hi = ys[-1] - 0.55, ys[0] + 0.55

    d_pre = PSNR["ours"] - PSNR["pretrained"]
    d_ctl = PSNR["ours"] - PSNR["control"]
    vals_pre = [r[2] for r in BE_ROWS]
    vals_ctl = [r[3] for r in BE_ROWS]
    xlim = (-28, 62)

    def panel(x0, w, vals, db, title, primary):
        ax = inch_axes(fig, x0, ax_bottom, w, ax_h)
        ax.set_xlim(*xlim)
        ax.set_ylim(y_lo, y_hi)
        span_in = w / (xlim[1] - xlim[0])            # inches per % unit
        for yy, v in zip(ys, vals):
            if v is None:
                continue
            good = v >= 0
            ax.barh(yy, v, 0.62, color=C_OURS if good else C_WORSE,
                    hatch=None if good else "//",
                    edgecolor="none" if good else "#6E6E6E", lw=1.2,
                    alpha=1.0 if primary else 0.8, zorder=2)
            off = 0.08 / span_in
            ax.text(v + (off if good else -off), yy, signed(v),
                    ha="left" if good else "right", va="center",
                    fontsize=F_SMALL, fontweight="bold" if primary else "normal",
                    color=INK, zorder=3)
        # PSNR row: text in dB, no bar (log scale is not comparable to %)
        off = 0.08 / span_in
        ax.text(off if db >= 0 else -off, ys[0], signed(db, " dB", 2),
                ha="left" if db >= 0 else "right", va="center", fontsize=F_SMALL,
                fontweight="bold" if primary else "normal", color=INK, zorder=3)
        ax.axvline(0, color=INK, lw=1.6, zorder=3)
        for i in range(1, len(BE_ROWS)):
            if BE_ROWS[i][0] != BE_ROWS[i - 1][0]:
                ax.axhline((ys[i] + ys[i - 1]) / 2, color=C_GRID, lw=1.5, zorder=1)
        ax.set_yticks([])
        ax.set_xticks([-20, 20, 40, 60])            # light grid only; values are on the bars
        ax.set_xticklabels([])
        ax.tick_params(axis="x", length=0)
        ax.grid(axis="x", color=C_GRID, lw=1.2, zorder=0)
        for s_ in ("top", "right", "left", "bottom"):
            ax.spines[s_].set_visible(False)
        # direction cue instead of numeric ticks
        x_zero = x0 + (0 - xlim[0]) / (xlim[1] - xlim[0]) * w
        fig.text(fx(fig, x_zero - 0.12), fy(fig, PAD + 0.20), "← worse", ha="right",
                 va="center", fontsize=F_SMALL, color=MUTED)
        fig.text(fx(fig, x_zero + 0.12), fy(fig, PAD + 0.20), "better →", ha="left",
                 va="center", fontsize=F_SMALL, color=MUTED)
        fig.text(fx(fig, x0 + w / 2), fy(fig, ax_top + 0.08), title, ha="center",
                 va="bottom", fontsize=F_PANEL, fontweight="bold" if primary else "normal",
                 color=INK if primary else MUTED)
        return ax

    ax_main = panel(x_main, main_w, vals_pre, d_pre, "vs. pretrained CNN", True)
    panel(x_ctl, ctl_w, vals_ctl, d_ctl, "vs. control", False)

    # row labels: family (bold, left) + metric (right-aligned)
    prev = None
    for yy, (fam, label, *_) in zip(ys, BE_ROWS):
        y_in = ax_bottom + (yy - y_lo) / (y_hi - y_lo) * ax_h
        if fam != prev:
            fig.text(fx(fig, PAD), fy(fig, y_in), fam, ha="left", va="center",
                     fontsize=F_SUB, fontweight="bold", color=INK)
            prev = fam
        fig.text(fx(fig, x_main - 0.15), fy(fig, y_in), label, ha="right", va="center",
                 fontsize=F_SMALL, color=INK)

    save_figure(fig, "results_broader_effects", subdir=SUBDIR)


# ============================================================
# FIGURE 3 - MT AGREEMENT                      10.40 x 5.50 in
# ============================================================

MT_ROWS = [
    # label,              fields where all 3 PD improve, MT also improves, primary
    ("vs. pretrained CNN", 150, 91, True),
    ("vs. control", 156, 119, False),
]
MT_MEAN = {"pretrained": 5.8678, "control": 6.0119, "ours": 5.6566}

MT_TITLE = "MT mostly agrees with PD"
MT_SUB = "Of the fields where all three PD distances improve, how many also improve in MT?"


def mt_figure():
    W, H = 10.40, 5.50
    fig = new_figure(W, H)
    y = headline(fig, MT_TITLE, MT_SUB, W - 2 * PAD) - 0.14

    BAR_H = 0.62
    bar_w = W - 2 * PAD
    ax = inch_axes(fig, PAD, 0, bar_w, H)
    ax.set_xlim(0, bar_w); ax.set_ylim(0, H); ax.set_axis_off()

    for label, n, agree, primary in MT_ROWS:
        t = fig.text(fx(fig, PAD), fy(fig, y), f"{label}  ·  {n} fields", ha="left",
                     va="top", fontsize=F_SMALL, fontweight="bold" if primary else "normal",
                     color=INK if primary else MUTED)
        yb = text_bottom(fig, t) - 0.08 - BAR_H
        a = agree / n
        alpha = 1.0 if primary else 0.8
        ax.add_patch(Rectangle((0, yb), bar_w * a, BAR_H, fc=C_OURS, ec="none", alpha=alpha))
        ax.add_patch(Rectangle((bar_w * a, yb), bar_w * (1 - a), BAR_H, fc=C_WORSE,
                               ec="#6E6E6E", hatch="//", lw=1.2, alpha=alpha))
        ax.text(0.15, yb + BAR_H / 2, f"{100 * a:.0f}%  ({agree})", ha="left",
                va="center", fontsize=F_VALUE, fontweight="bold", color="white")
        ax.text(bar_w - 0.15, yb + BAR_H / 2, f"{100 * (1 - a):.0f}%  ({n - agree})",
                ha="right", va="center", fontsize=F_SMALL, color=INK,
                bbox=dict(fc="white", ec="none", pad=2, alpha=0.9))
        y = yb - 0.22

    # legend
    ly = y - 0.20
    ax.add_patch(Rectangle((0, ly - 0.13), 0.45, 0.26, fc=C_OURS, ec="none"))
    t = fig.text(fx(fig, PAD + 0.58), fy(fig, ly), "MT also improves", ha="left",
                 va="center", fontsize=F_SMALL, color=INK)
    x2 = t.get_window_extent(fig.canvas.get_renderer()).x1 / fig.dpi - PAD + 0.45
    ax.add_patch(Rectangle((x2, ly - 0.13), 0.45, 0.26, fc=C_WORSE, ec="#6E6E6E",
                           hatch="//", lw=1.2))
    fig.text(fx(fig, PAD + x2 + 0.58), fy(fig, ly), "MT does not", ha="left",
             va="center", fontsize=F_SMALL, color=INK)

    # mean MT distance
    ours, pre, ctl = MT_MEAN["ours"], MT_MEAN["pretrained"], MT_MEAN["control"]
    y = ly - 0.34
    ax.plot([0, bar_w], [y, y], color=C_GRID, lw=1.5)
    t = fig.text(fx(fig, PAD), fy(fig, y - 0.10),
                 f"Mean MT distance: {ours:.2f}  (lower is better)",
                 ha="left", va="top", fontsize=F_SMALL, fontweight="bold", color=INK)
    fig.text(fx(fig, PAD), fy(fig, text_bottom(fig, t) - 0.04),
             wrap(fig, f"{signed(100 * (ours - pre) / pre)} vs. pretrained ({pre:.2f}),  "
                       f"{signed(100 * (ours - ctl) / ctl)} vs. control ({ctl:.2f})",
                  F_SMALL, bar_w),
             ha="left", va="top", fontsize=F_SMALL, color=INK, linespacing=1.05)

    save_figure(fig, "results_mt_agreement", subdir=SUBDIR)


if __name__ == "__main__":
    pd_figure()
    be_figure()
    mt_figure()
