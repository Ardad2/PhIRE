"""
Shared style for every poster figure.

Import this in each figure script so all figures use the same fonts, sizes,
colours and export rules, and so every figure is checked against the
poster's typography convention before it is saved.

    from poster_style import (FS, INK, MUTED, new_figure, save_figure, ...)

Rules enforced here
-------------------
* Figures are generated at their final placed size (inches), never resized
  in Slides.  1 pt in Matplotlib == 1 pt on the printed A0 poster.
* 24 pt is the floor for any text.  `audit()` fails the export otherwise.
* No text may leave the canvas or overlap other text.
* No bbox_inches="tight" (it silently changes the figure size).
"""

from __future__ import annotations

import logging
import os
from itertools import combinations
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")

# Arial's embedded timestamps make fontTools log harmless
# "'created' timestamp seems very low" lines when writing PDFs.
logging.getLogger("fontTools").setLevel(logging.ERROR)
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402


# ============================================================
# PATHS
# ============================================================

# Override with:  PHIRE_ROOT=/some/other/path python3 script.py
ROOT = Path(os.environ.get("PHIRE_ROOT", Path.home() / "PhIRE"))

OUT_DIR = ROOT / "figures" / "poster_problem_motivation"


# ============================================================
# TYPOGRAPHY  (frozen convention: 90 - 54 - 38 - 30 - 28 - 26 - 24)
# ============================================================

class FS:
    """Font sizes in points *at print size*."""

    FLOOR = 24          # hard minimum anywhere on the poster
    TICK = 24
    LEGEND = 24
    AXIS = 24
    NOTE = 24           # small explanatory text inside a figure
    LABEL = 24          # panel labels under / over images
    ANNOT = 26          # numeric annotations, bar values
    PANEL_TITLE = 28    # A/B/C panel titles, column headers
    HEADLINE = 30       # figure headline (only if the poster has none)
    EMPHASIS = 32       # large emphasised result inside a figure


# Arial on macOS; Liberation Sans has identical metrics on Linux.
# Matching the poster's sans-serif keeps figures visually integrated and is
# ~10% narrower than Matplotlib's default DejaVu Sans.
FONT_FAMILY = ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"]


# ============================================================
# COLOUR
# ============================================================

INK = "#1F1F1F"         # primary text
MUTED = "#5C5C5C"       # secondary text (>= 4.5:1 on white)
RULE = "#9A9A9A"        # arrows, thin connectors
FRAME = "#4A4A4A"       # image borders
PANEL_BG = "#F4F6F7"    # very light card fill

# Magnitude (wind speed): one perceptually uniform sequential map, one fixed
# range, used by *every* wind figure on the poster (Sections 1 and 4) so the
# same colour always means the same speed.
WIND_CMAP = "viridis"
WIND_VMIN = 0.0
WIND_VMAX = 20.0        # m/s; raise if any field is clipped (the scripts warn)
WIND_UNITS = "m/s"

# Method identity, matching the Results figures.
METHOD_COLORS = {
    "pretrained": "#8C8C8C",   # grey
    "control": "#E8A34A",      # orange (hatched in results)
    "topology": "#0F9D74",     # green
}

# Feature identity in the PD/MT primer (validated categorical set:
# all-pairs CVD dE >= 13, normal-vision dE >= 16).  Chosen to avoid the
# grey/orange/green that mean "method" elsewhere on the poster.
FEATURE_COLORS = ["#2A78D6", "#E87BA4", "#4A3AA7"]   # A, B, C


def _first_installed(families):
    """First family in the list that is actually installed (Arial on macOS,
    Liberation Sans on the Spark), so maths and text use the same face."""
    from matplotlib import font_manager
    installed = {f.name for f in font_manager.fontManager.ttflist}
    return next((f for f in families if f in installed), "DejaVu Sans")


BODY_FONT = _first_installed(FONT_FAMILY)


def apply_rcparams() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [BODY_FONT] + [f for f in FONT_FAMILY if f != BODY_FONT],
        "font.size": FS.NOTE,
        "axes.titlesize": FS.PANEL_TITLE,
        "axes.labelsize": FS.AXIS,
        "xtick.labelsize": FS.TICK,
        "ytick.labelsize": FS.TICK,
        "legend.fontsize": FS.LEGEND,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": FRAME,
        "xtick.color": INK,
        "ytick.color": INK,
        "mathtext.fontset": "custom",
        "mathtext.rm": BODY_FONT,
        "mathtext.it": f"{BODY_FONT}:italic",
        "mathtext.bf": f"{BODY_FONT}:bold",
        "mathtext.sf": BODY_FONT,
        "mathtext.cal": f"{BODY_FONT}:italic",   # silences the 'cursive' lookup
        # Embed TrueType so the PDF keeps real text (Type 42).
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    })


# ============================================================
# FIGURE HELPERS
# ============================================================

def new_figure(width_in: float, height_in: float):
    """Create a figure at the exact size of its poster slot."""
    apply_rcparams()
    return plt.figure(figsize=(width_in, height_in), facecolor="white")


def inch_axes(fig, left, bottom, width, height, **kw):
    """Add axes positioned in inches from the bottom-left corner.

    Working in inches keeps gutters and image sizes identical across
    figures of different sizes.
    """
    W, H = fig.get_size_inches()
    return fig.add_axes([left / W, bottom / H, width / W, height / H], **kw)


def fx(fig, x_in):
    """Inches -> figure fraction (x)."""
    return x_in / fig.get_size_inches()[0]


def fy(fig, y_in):
    """Inches -> figure fraction (y)."""
    return y_in / fig.get_size_inches()[1]


def halo(width=3.5, color="white"):
    """Path effect that keeps text readable on top of images."""
    return [pe.withStroke(linewidth=width, foreground=color)]


def rich_line(fig, x_center, y, parts, fontsize=FS.NOTE, va="center"):
    """One centred line built from (text, weight, color) runs.

    Lets a sentence carry bold keywords ("PD: ...") without mathtext.
    x_center, y are figure fractions.
    """
    r = fig.canvas.get_renderer()
    group = f"rich{len(fig.texts)}"
    texts = [fig.text(0, y, s, fontsize=fontsize, fontweight=w, color=c,
                      ha="left", va=va, gid=group) for s, w, c in parts]
    widths = [t.get_window_extent(r).width / fig.bbox.width for t in texts]
    x = x_center - sum(widths) / 2
    for t, w in zip(texts, widths):
        t.set_x(x)
        x += w
    return texts


def style_image_axes(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(FRAME)
        s.set_linewidth(1.0)


UNIT_H = 0.36          # height of a 24-pt line, inches
TICK_CLEAR = 0.24      # half a 24-pt tick label + gap, so the top tick never hits the unit


def wind_colorbar(fig, im, left, bottom, top, width=0.14):
    """Slim vertical colourbar with the unit written above it.

    `bottom` and `top` (inches) are the vertical band the colourbar block may
    use - normally the bottom and top edge of the images next to it.  The unit
    label is pinned to `top`; the bar fills the rest, leaving room for the top
    tick label whether or not the bar has an extension arrow.

    A short horizontal unit label is much narrower than a rotated
    'Wind speed (m s^-1)' label, which buys the images ~0.35 in.
    """
    clipped = _DATA_MAX[0] > WIND_VMAX
    bar_top = top - UNIT_H - TICK_CLEAR
    cax = inch_axes(fig, left, bottom, width, bar_top - bottom)
    cb = fig.colorbar(im, cax=cax, extend="max" if clipped else "neither")
    # at most 3 ticks: a short bar can't fit more 24-pt labels
    span = WIND_VMAX - WIND_VMIN
    step = next(s for s in (1, 2, 5, 10, 20, 25, 50, 100) if s >= span / 2)
    cb.set_ticks(np.arange(WIND_VMIN, WIND_VMAX + 1e-9, step))
    cb.ax.tick_params(labelsize=FS.TICK, length=4, width=1.0, pad=4)
    cb.outline.set_linewidth(1.0)
    # left-aligned over the bar so it spans the bar + tick column, not the image
    fig.text(
        fx(fig, left),
        fy(fig, top),
        WIND_UNITS,
        ha="left", va="top",
        fontsize=FS.NOTE, color=MUTED,
    )
    return cb


_DATA_MAX = [0.0]


def check_wind_range(*arrays) -> None:
    """Warn if a field exceeds the shared colour range (it would be clipped)."""
    m = max(float(a.max()) for a in arrays)
    _DATA_MAX[0] = max(_DATA_MAX[0], m)
    if m > WIND_VMAX:
        print(
            f"  NOTE: max wind speed {m:.1f} {WIND_UNITS} > WIND_VMAX "
            f"{WIND_VMAX:.0f}; colours above are clipped (colourbar shows an arrow). "
            f"Raise WIND_VMAX in poster_style.py to use one range everywhere."
        )


# ============================================================
# AUDIT
# ============================================================

def _undrawn_tick_labels(fig):
    """Tick labels Matplotlib keeps around but does not draw (outside limits)."""
    hidden = set()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            if ax.axison and axis.get_visible():
                drawn = {id(t) for t in axis._update_ticks()}
            else:
                drawn = set()          # axis switched off: nothing is drawn
            for t in axis.get_major_ticks() + axis.get_minor_ticks():
                if id(t) not in drawn:
                    hidden.update({t.label1, t.label2})
    return hidden


def _visible_texts(fig):
    hidden = _undrawn_tick_labels(fig)
    for t in fig.findobj(matplotlib.text.Text):
        if not t.get_visible() or t in hidden:
            continue
        s = t.get_text()
        if not s or not s.strip():
            continue
        yield t


MIN_GAP_IN = 0.04   # minimum clear space between any two text items


def audit(fig, name: str, min_pt: float = FS.FLOOR, strict: bool = True) -> bool:
    """Check the figure against the poster convention.

    * every visible text >= min_pt
    * no text outside the canvas
    * no two texts overlapping

    Returns True if clean.  With strict=True a problem raises, so a figure
    that breaks the convention can never be exported by accident.
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    W, H = fig.bbox.width, fig.bbox.height
    tol = 1.0  # px

    problems = []
    boxes = []

    for t in _visible_texts(fig):
        label = repr(t.get_text()[:40])
        size = t.get_fontsize()
        if size < min_pt - 1e-6:
            problems.append(f"{label}: {size:.0f} pt < {min_pt:.0f} pt floor")
        bb = t.get_window_extent(r)
        if bb.x0 < -tol or bb.y0 < -tol or bb.x1 > W + tol or bb.y1 > H + tol:
            problems.append(f"{label}: runs off the canvas")
        boxes.append((label, bb, t.get_gid()))

    # text must not sit on top of an image it doesn't belong to
    for ax in fig.axes:
        if not ax.images:
            continue
        abb = ax.get_window_extent(r)
        own = set(ax.texts)
        for t in _visible_texts(fig):
            if t in own or t.axes is ax:
                continue
            bb = t.get_window_extent(r)
            ox = min(abb.x1, bb.x1) - max(abb.x0, bb.x0)
            oy = min(abb.y1, bb.y1) - max(abb.y0, bb.y0)
            if ox > 2 and oy > 2:
                problems.append(f"{t.get_text()[:40]!r} covers an image")

    # Require a small clear gap, not just "not touching": fonts differ a
    # little between machines (Arial on macOS vs Liberation Sans on Linux),
    # so a figure that only just passes here could fail elsewhere.
    gap = MIN_GAP_IN * fig.dpi
    for (la, a, ga), (lb, b, gb) in combinations(boxes, 2):
        if ga and ga == gb and ga.startswith("rich"):
            continue        # runs of one rich_line sit flush by design
        ox = min(a.x1, b.x1) - max(a.x0, b.x0) + gap
        oy = min(a.y1, b.y1) - max(a.y0, b.y0) + gap
        if ox > 0 and oy > 0:
            problems.append(f"{la} overlaps or is within {MIN_GAP_IN} in of {lb}")

    w_in, h_in = fig.get_size_inches()
    print(f"[audit] {name}: {w_in:.2f} x {h_in:.2f} in, {len(boxes)} text items")
    if problems:
        for p in problems:
            print("   x", p)
        if strict:
            raise SystemExit(f"[audit] {name}: {len(problems)} problem(s) - not saved.")
        return False
    print("   ok: all text >= %d pt, inside canvas, no overlaps" % min_pt)
    return True


def _tag_srgb(path, dpi):
    """Embed an sRGB colour profile in the PNG.

    Matplotlib writes untagged PNGs; Keynote/macOS then guess the colour space,
    which can shift hues slightly against text coloured with the same hex code
    on the slide.  Tagging the file as sRGB makes #0182B9 in the figure match
    #0182B9 typed into Keynote.
    """
    try:
        from PIL import Image, ImageCms
        profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
        with Image.open(path) as im:
            im.load()
            im.save(path, icc_profile=profile, dpi=(dpi, dpi))
    except Exception as e:                      # never block an export over this
        print(f"   (could not embed sRGB profile: {e})")


def save_figure(fig, stem: str, dpi: int = 300, strict: bool = True,
                subdir: str | None = None):
    """Audit, then write PNG (for Slides) and PDF (vector) at the exact size.

    subdir: folder under ~/PhIRE/figures (default: poster_problem_motivation).
    """
    out = OUT_DIR if subdir is None else ROOT / "figures" / subdir
    out.mkdir(parents=True, exist_ok=True)
    audit(fig, stem, strict=strict)
    png = out / f"{stem}.png"
    pdf = out / f"{stem}.pdf"
    fig.savefig(png, dpi=dpi)      # never bbox_inches="tight"
    _tag_srgb(png, dpi)
    fig.savefig(pdf)
    w, h = fig.get_size_inches()
    print(f"   saved {png}  ({round(w * dpi)} x {round(h * dpi)} px)")
    print(f"   saved {pdf}")
    print(f"   -> insert at exactly {w:.2f} x {h:.2f} in")
    plt.close(fig)
    return png, pdf
