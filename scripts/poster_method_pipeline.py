#!/usr/bin/env python3
"""
Section 2 - Method pipeline strip
Poster slot: 31.20 x 1.18 in

What the reader should get in 3 seconds:
    one pretrained checkpoint is fine-tuned two ways (control vs
    topology-inspired), both are evaluated on the same held-out fields, and
    scored on three families of metrics.

Layout (one row: 1.18 in fits one row of two-line boxes at 24-30 pt):

  [Pretrained SR CNN    ]  fine-tune  ┌[Control: reconstruction loss only             ]┐ evaluate on the same  [Fidelity] [Domain & structure] [Topology]
  [shared starting ckpt ] ────────────┤                                                 ├───────────────────► [metrics ] [metrics           ] [metrics ]
                                      └[Topology-inspired: adds gradient + topology ...]┘ 168 held-out fields

* Box widths are measured from their text, so editing the wording below
  re-flows the layout; the script stops if the text no longer fits 31.2 in.
* Method colours match the Results figures (grey = pretrained,
  orange = control, green = topology-inspired).  The metric families are
  neutral: colour is reserved for methods.
* All text >= 24 pt; the export is refused if anything overlaps.
"""

from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from poster_style import (
    INK, MUTED, RULE, FRAME, PANEL_BG, METHOD_COLORS,
    new_figure, fx, fy, rich_line, save_figure,
)


# ============================================================
# WORDING  (edit freely; the layout re-flows)
# ============================================================

OUT_STEM = "method_pipeline_final"
FIG_W, FIG_H = 31.20, 1.18

START = ("Pretrained SR CNN", "shared starting checkpoint")
FORK_LABEL = "fine-tune"

# (bold name, regular description) - "adds" keeps it clear that the
# reconstruction loss is still there in the topology-inspired objective.
CONTROL = ("Reconstruction-only (control)", "")
OURS = ("Reconstruction + gradient + persistence-pair supervision", "")

EVAL_LABEL = ("same evaluation on", "168 held-out fields")   # above / below arrow

# Metric *families* (examples, not exhaustive lists):
# pixel agreement -> physically meaningful behaviour -> topology
METRICS = [
    ("Fidelity", "PSNR · SSIM · MAE"),
    ("Domain & structure", "wind power · gradients · connectivity"),
    ("Topology", "PD · MT"),
]

# Font sizes (pt).  Titles 30 (pipeline text), branch lines 26, secondary 24.
F_TITLE = 30
F_BRANCH = 26
F_SUB = 24


# ============================================================
# SET-UP
# ============================================================

fig = new_figure(FIG_W, FIG_H)
canvas = fig.add_axes([0, 0, 1, 1])
canvas.set_xlim(0, FIG_W)
canvas.set_ylim(0, FIG_H)
canvas.set_axis_off()
R = fig.canvas.get_renderer()

MID = FIG_H / 2
BOX_H = 1.04            # two-line boxes
BR_H, BR_GAP = 0.47, 0.10
PAD = 0.17              # horizontal padding inside a box (each side)
M = 0.06                # outer margin


def text_w(s, size, weight="normal"):
    t = fig.text(0, 0, s, fontsize=size, fontweight=weight)
    w = t.get_window_extent(R).width / fig.dpi
    t.remove()
    return w


def tint(hex_color, amount):
    """Mix a colour with white; amount=0 -> white, 1 -> the colour."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    mix = lambda c: round(255 - (255 - c) * amount)
    return f"#{mix(r):02X}{mix(g):02X}{mix(b):02X}"


def box(x, y, w, h, fc, ec, lw=1.6):
    canvas.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0,rounding_size=0.08",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=2))


def two_line_box(x, w, title, sub, fc, ec):
    box(x, MID - BOX_H / 2, w, BOX_H, fc, ec)
    cx = fx(fig, x + w / 2)
    fig.text(cx, fy(fig, MID + 0.19), title, ha="center", va="center",
             fontsize=F_TITLE, fontweight="bold", color=INK)
    fig.text(cx, fy(fig, MID - 0.21), sub, ha="center", va="center",
             fontsize=F_SUB, color=MUTED)


def path(xs, ys, arrow=False):
    if len(xs) > 2 or not arrow:
        end = -1 if arrow else None
        canvas.plot(xs[:end] if arrow else xs, ys[:end] if arrow else ys,
                    color=RULE, lw=2.4, solid_joinstyle="miter", zorder=1)
    if arrow:
        canvas.add_patch(FancyArrowPatch(
            (xs[-2], ys[-2]), (xs[-1], ys[-1]), arrowstyle="-|>",
            mutation_scale=26, linewidth=2.4, color=RULE, zorder=1,
            shrinkA=0, shrinkB=0))


# ============================================================
# WIDTHS (measured from the text)
# ============================================================

w_start = max(text_w(START[0], F_TITLE, "bold"), text_w(START[1], F_SUB)) + 2 * PAD
w_branch = max(text_w(CONTROL[0], F_BRANCH, "bold") + text_w(CONTROL[1], F_BRANCH),
               text_w(OURS[0], F_BRANCH, "bold") + text_w(OURS[1], F_BRANCH)) + 2 * PAD
w_chips = [max(text_w(t, F_TITLE, "bold"), text_w(s, F_SUB)) + 2 * PAD for t, s in METRICS]
CHIP_GAP = 0.15

MIN_FORK = text_w(FORK_LABEL, F_SUB) + 0.45          # label sits on the stem
MIN_EVAL = max(text_w(s, F_SUB) for s in EVAL_LABEL) + 0.62

fixed = 2 * M + w_start + w_branch + sum(w_chips) + CHIP_GAP * (len(METRICS) - 1)
slack = FIG_W - fixed - MIN_FORK - MIN_EVAL
if slack < 0:
    raise SystemExit(f"Pipeline text is {-slack:.2f} in too wide for {FIG_W} in - "
                     f"shorten the wording or reduce F_BRANCH/F_SUB.")
# spare width goes to the two connectors, then to wider chips
W_FORK = MIN_FORK + min(slack * 0.25, 0.6)
W_EVAL = MIN_EVAL + min(slack * 0.35, 0.8)
extra = FIG_W - fixed - W_FORK - W_EVAL
w_chips = [w + extra / len(w_chips) for w in w_chips]

print(f"widths (in): start {w_start:.2f}  fork {W_FORK:.2f}  branches {w_branch:.2f}  "
      f"eval {W_EVAL:.2f}  chips {', '.join(f'{w:.2f}' for w in w_chips)}")


# ============================================================
# DRAW
# ============================================================

x = M

# --- 1. shared start -------------------------------------------
pre = METHOD_COLORS["pretrained"]
two_line_box(x, w_start, *START, tint(pre, 0.18), pre)
x0 = x + w_start
x += w_start

# --- 2. fork: "fine-tune" -> two branches ----------------------
c_top = MID + BR_GAP / 2 + BR_H / 2
c_bot = MID - BR_GAP / 2 - BR_H / 2
x_br = x + W_FORK
x_split = x_br - 0.30
path([x0, x_split], [MID, MID])
path([x_split, x_split, x_br], [MID, c_top, c_top], arrow=True)
path([x_split, x_split, x_br], [MID, c_bot, c_bot], arrow=True)
fig.text(fx(fig, (x0 + x_split) / 2), fy(fig, MID + 0.07), FORK_LABEL,
         ha="center", va="bottom", fontsize=F_SUB, color=MUTED)

ctl, ours = METHOD_COLORS["control"], METHOD_COLORS["topology"]
box(x_br, c_top - BR_H / 2, w_branch, BR_H, tint(ctl, 0.22), ctl)
box(x_br, c_bot - BR_H / 2, w_branch, BR_H, tint(ours, 0.20), ours, lw=2.6)
cx = fx(fig, x_br + w_branch / 2)
rich_line(fig, cx, fy(fig, c_top), [(CONTROL[0], "bold", INK), (CONTROL[1], "normal", INK)],
          fontsize=F_BRANCH)
rich_line(fig, cx, fy(fig, c_bot), [(OURS[0], "bold", INK), (OURS[1], "normal", INK)],
          fontsize=F_BRANCH)
x = x_br + w_branch

# --- 3. merge -> evaluate on the same held-out fields ----------
x_join = x + 0.30
x_chips = x + W_EVAL
path([x, x_join, x_join], [c_top, c_top, MID])
path([x, x_join, x_join], [c_bot, c_bot, MID])
path([x_join, x_chips], [MID, MID], arrow=True)
ex = fx(fig, (x_join + x_chips) / 2 - 0.08)
fig.text(ex, fy(fig, MID + 0.07), EVAL_LABEL[0], ha="center", va="bottom",
         fontsize=F_SUB, color=MUTED)
fig.text(ex, fy(fig, MID - 0.07), EVAL_LABEL[1], ha="center", va="top",
         fontsize=F_SUB, color=MUTED)
x = x_chips

# --- 4. three metric families (neutral) ------------------------
for (title, sub), w in zip(METRICS, w_chips):
    two_line_box(x, w, title, sub, PANEL_BG, FRAME)
    x += w + CHIP_GAP

save_figure(fig, OUT_STEM, subdir="poster_method")