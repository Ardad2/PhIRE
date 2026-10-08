"""
Real wind-field SR motivation figure for the poster.

Uses actual repaired PhIRE CNN batched arrays on Spark:
    ~/PhIRE/data_out_fixed/wind_mrhr_cnn/dataIN.npy
    ~/PhIRE/data_out_fixed/wind_mrhr_cnn/dataSR.npy
    ~/PhIRE/data_out_fixed/wind_mrhr_cnn/dataGT.npy

Expected shapes:
    dataIN : (168, 100, 100, 2)
    dataSR : (168, 500, 500, 2)
    dataGT : (168, 500, 500, 2)

The same SAMPLE_IDX is extracted from all three arrays.
Each (u,v) field is converted to wind-speed magnitude:
    s = sqrt(u^2 + v^2)

Output:
    motivation_real.png   (300 dpi)
    motivation_real.pdf   (vector)
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

# ============================================================
# CONFIG
# ============================================================

SAMPLE_IDX = 69

DATA_IN_PATH = Path("~/PhIRE/data_out_fixed/wind_mrhr_cnn/dataIN.npy").expanduser()
DATA_SR_PATH = Path("~/PhIRE/data_out_fixed/wind_mrhr_cnn/dataSR.npy").expanduser()
DATA_GT_PATH = Path("~/PhIRE/data_out_fixed/wind_mrhr_cnn/dataGT.npy").expanduser()

OUT_PNG = "motivation_real.png"
OUT_PDF = "motivation_real.pdf"

TITLE = "Wind-field super-resolution across the same physical region"

PANEL_LABELS = [
    "Lower-resolution input",
    "SR prediction",
    "High-resolution ground truth",
]

PANEL_META = [
    "100 × 100 · 10 km / cell",
    "500 × 500 · 2 km / cell",
    "500 × 500 · 2 km / cell",
]

FOOTER_1 = r"SR predicts wind velocity $(u,v)$; topology is evaluated on wind-speed magnitude $s=\sqrt{u^2+v^2}$."
FOOTER_2 = r"Topology metrics use the same fixed $160\times160$ SR/GT evaluation crop for every method."

# Grid settings
MR_GRID_DIVISIONS = 5
HR_GRID_DIVISIONS = 25

# Keep HR grid lighter/thinner so it does not obscure the data
MR_GRID_COLOR = "white"
MR_GRID_ALPHA = 0.55
MR_GRID_LW = 1.0

HR_GRID_COLOR = "white"
HR_GRID_ALPHA = 0.28
HR_GRID_LW = 0.35

# Display settings
CMAP = "viridis"
FIGSIZE = (16.5, 6.6)
TITLE_COLOR = "#1A6B5A"
TEXT_COLOR = "#111111"
MUTED_COLOR = "#666666"
ARROW_COLOR = "#7A838C"

# ============================================================
# HELPERS
# ============================================================

def uv_to_speed(arr):
    """
    Convert a single sample from (H, W, 2) to wind-speed magnitude (H, W).
    No transpose is performed.
    """
    if arr.ndim != 3 or arr.shape[-1] != 2:
        raise ValueError(f"Expected shape (H, W, 2), got {arr.shape}")
    u = arr[..., 0]
    v = arr[..., 1]
    return np.sqrt(u**2 + v**2)


def draw_grid(ax, h, w, divisions, color="white", alpha=0.4, lw=0.5):
    """
    Draw representative grid lines only (not every cell).
    """
    xs = np.linspace(-0.5, w - 0.5, divisions + 1)
    ys = np.linspace(-0.5, h - 0.5, divisions + 1)

    for x in xs:
        ax.plot([x, x], [-0.5, h - 0.5], color=color, lw=lw, alpha=alpha, zorder=3)
    for y in ys:
        ax.plot([-0.5, w - 0.5], [y, y], color=color, lw=lw, alpha=alpha, zorder=3)


def add_arrow(fig, ax_from, ax_to, text):
    """
    Add a simple arrow between panels in figure coordinates.
    """
    b1 = ax_from.get_position()
    b2 = ax_to.get_position()

    x1 = b1.x1 + 0.012
    x2 = b2.x0 - 0.012
    y = 0.5 * (b1.y0 + b1.y1)

    arrow = FancyArrowPatch(
        (x1, y), (x2, y),
        transform=fig.transFigure,
        arrowstyle="simple",
        mutation_scale=28,
        lw=0.0,
        color=ARROW_COLOR,
        alpha=0.9,
        zorder=10,
    )
    fig.patches.append(arrow)

    fig.text(
        0.5 * (x1 + x2), y + 0.02, text,
        ha="center", va="bottom",
        fontsize=15, fontweight="bold", color=TEXT_COLOR
    )


# ============================================================
# LOAD ARRAYS
# ============================================================

data_in = np.load(DATA_IN_PATH)
data_sr = np.load(DATA_SR_PATH)
data_gt = np.load(DATA_GT_PATH)

print("Loaded array shapes:")
print(f"  dataIN: {data_in.shape}")
print(f"  dataSR: {data_sr.shape}")
print(f"  dataGT: {data_gt.shape}")
print(f"Selected SAMPLE_IDX: {SAMPLE_IDX}")

# Validate shapes
expected_in_shape_tail = (100, 100, 2)
expected_hr_shape_tail = (500, 500, 2)

if data_in.ndim != 4 or data_in.shape[1:] != expected_in_shape_tail:
    raise ValueError(f"dataIN shape mismatch: expected (*, 100, 100, 2), got {data_in.shape}")
if data_sr.ndim != 4 or data_sr.shape[1:] != expected_hr_shape_tail:
    raise ValueError(f"dataSR shape mismatch: expected (*, 500, 500, 2), got {data_sr.shape}")
if data_gt.ndim != 4 or data_gt.shape[1:] != expected_hr_shape_tail:
    raise ValueError(f"dataGT shape mismatch: expected (*, 500, 500, 2), got {data_gt.shape}")

n_samples = data_in.shape[0]
if not (0 <= SAMPLE_IDX < n_samples):
    raise IndexError(f"SAMPLE_IDX={SAMPLE_IDX} is out of range for {n_samples} samples")

# Extract one sample from each
mr_uv = data_in[SAMPLE_IDX]   # (100, 100, 2)
sr_uv = data_sr[SAMPLE_IDX]   # (500, 500, 2)
gt_uv = data_gt[SAMPLE_IDX]   # (500, 500, 2)

# Convert to wind-speed magnitude
mr = uv_to_speed(mr_uv)
sr = uv_to_speed(sr_uv)
gt = uv_to_speed(gt_uv)

print("Wind-speed ranges for selected sample:")
print(f"  MR: min={mr.min():.4f}, max={mr.max():.4f}")
print(f"  SR: min={sr.min():.4f}, max={sr.max():.4f}")
print(f"  GT: min={gt.min():.4f}, max={gt.max():.4f}")

# Shared color scale
vmin = 0.0
vmax = max(float(mr.max()), float(sr.max()), float(gt.max()))
print(f"Shared color scale: vmin={vmin:.4f}, vmax={vmax:.4f}")

# ============================================================
# FIGURE
# ============================================================

plt.rcParams.update({
    "font.size": 12,
    "font.family": "sans-serif",
})

fig = plt.figure(figsize=FIGSIZE)

# 3 panels + 1 shared colorbar
gs = fig.add_gridspec(
    1, 4,
    width_ratios=[1.0, 1.0, 1.0, 0.045],
    left=0.045,
    right=0.98,
    top=0.80,
    bottom=0.24,
    wspace=0.18
)

ax_mr = fig.add_subplot(gs[0, 0])
ax_sr = fig.add_subplot(gs[0, 1])
ax_gt = fig.add_subplot(gs[0, 2])
cax   = fig.add_subplot(gs[0, 3])

# Main title
fig.suptitle(
    TITLE,
    fontsize=24,
    fontweight="bold",
    color=TITLE_COLOR,
    y=0.92
)

# Plot MR
im = ax_mr.imshow(
    mr,
    cmap=CMAP,
    vmin=vmin,
    vmax=vmax,
    origin="upper",
    interpolation="nearest",
    aspect="equal",
)
draw_grid(
    ax_mr, mr.shape[0], mr.shape[1],
    divisions=MR_GRID_DIVISIONS,
    color=MR_GRID_COLOR,
    alpha=MR_GRID_ALPHA,
    lw=MR_GRID_LW
)

# Plot SR
ax_sr.imshow(
    sr,
    cmap=CMAP,
    vmin=vmin,
    vmax=vmax,
    origin="upper",
    interpolation="nearest",
    aspect="equal",
)
draw_grid(
    ax_sr, sr.shape[0], sr.shape[1],
    divisions=HR_GRID_DIVISIONS,
    color=HR_GRID_COLOR,
    alpha=HR_GRID_ALPHA,
    lw=HR_GRID_LW
)

# Plot GT
ax_gt.imshow(
    gt,
    cmap=CMAP,
    vmin=vmin,
    vmax=vmax,
    origin="upper",
    interpolation="nearest",
    aspect="equal",
)
draw_grid(
    ax_gt, gt.shape[0], gt.shape[1],
    divisions=HR_GRID_DIVISIONS,
    color=HR_GRID_COLOR,
    alpha=HR_GRID_ALPHA,
    lw=HR_GRID_LW
)

# Shared colorbar at far right
cb = fig.colorbar(im, cax=cax)
cb.ax.tick_params(labelsize=10)
cb.set_label("Wind speed (m s$^{-1}$)", fontsize=12)

# Clean axes + labels beneath panels
for ax, label, meta in zip(
    [ax_mr, ax_sr, ax_gt],
    PANEL_LABELS,
    PANEL_META
):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#333333")
        spine.set_linewidth(0.8)

    ax.text(
        0.5, -0.08, label,
        transform=ax.transAxes,
        ha="center", va="top",
        fontsize=17, fontweight="bold", color=TEXT_COLOR
    )
    ax.text(
        0.5, -0.19, meta,
        transform=ax.transAxes,
        ha="center", va="top",
        fontsize=14, color=MUTED_COLOR
    )

# Arrows
add_arrow(fig, ax_mr, ax_sr, "SR model")
add_arrow(fig, ax_sr, ax_gt, "compare")

# Footer notes only
fig.text(
    0.5, 0.105, FOOTER_1,
    ha="center", va="center",
    fontsize=13.5, color=MUTED_COLOR
)
fig.text(
    0.5, 0.065, FOOTER_2,
    ha="center", va="center",
    fontsize=13.5, color=MUTED_COLOR
)

# Save
fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(OUT_PDF, bbox_inches="tight", facecolor="white")

print(f"Saved {OUT_PNG}")
print(f"Saved {OUT_PDF}")