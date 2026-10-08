#!/usr/bin/env python3
"""
Final poster assets for Section 4: Qualitative Example (Sample 78).

Outputs separate, poster-ready assets:
  - 4 wind-speed field panels
  - 3 absolute-error panels
  - shared speed + error colorbars
  - one compact persistence-survival curve
  - one layout preview
  - a text audit/summary

Scientific choices:
  * fixed Sample 78
  * fixed 160x160 topology-evaluation crop at (x0,y0)=(0,0)
  * wind speed s = sqrt(u^2 + v^2)
  * same field color scale across GT / CNN / control / topology-inspired
  * same error color scale across all three absolute-error maps
  * survival curve uses actual audited finite D0+D1 PD pairs
  * GT persistence diagrams are checked for exact positive-persistence equality
    across CNN / control / topology runs before plotting

Run from ~/PhIRE:
    python3 scripts/poster_sample78_qualitative_final.py

If your canonical PD audit lives elsewhere:
    AUDIT=/path/to/phire_runtime_audit_xxx python3 scripts/poster_sample78_qualitative_final.py
"""

from pathlib import Path
import csv
import math
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.colorbar import ColorbarBase


# =============================================================================
# CONFIG
# =============================================================================

ROOT = Path.home() / "PhIRE"

CNN_DIR = ROOT / "data_out_fixed" / "wind_mrhr_cnn"
CONTROL_DIR = ROOT / "data_out" / "wind_finetune_candidateUV_expanded2688"
TOPO_DIR = ROOT / "data_out" / "wind_finetune_candidateF_grad_E2_low_expanded2688"

SAMPLE_ID = 78

# Fixed topology-evaluation crop.
X0, Y0, PATCH = 0, 0, 160

# Historical audited PD machinery.
AUDIT = Path(
    os.environ.get(
        "AUDIT",
        str(Path.home() / "phire_runtime_audit_20260809_221548"),
    )
)
W22 = Path(os.environ.get("W22", str(AUDIT / "recompute_pd_w22")))
W22_SWEEP = W22 / "w22_full_sweep.csv"

CANONICAL_DIR = AUDIT / "recompute_pd"
sys.path.insert(0, str(CANONICAL_DIR))
import canonical_pd_pilot as canonical

CNN_RUN = "cnn"
CONTROL_RUN = "topology_finetuning/candidateUV_expanded2688_topology"
TOPO_RUN = "topology_finetuning/candidateF_grad_E2_low_expanded2688_topology"

OUT_DIR = ROOT / "figures" / "poster_sample78_final"

# Poster physical asset sizes (inches).
FIELD_SIZE = 2.60
ERROR_SIZE = 1.40
CBAR_W = 1.00
SPEED_CBAR_H = FIELD_SIZE
ERROR_CBAR_H = ERROR_SIZE
SURV_W, SURV_H = 8.10, 4.68

# Content-only preview: exact width of the Section-4 inner region.
PREVIEW_W, PREVIEW_H = 31.20, 4.68

# Display style.
WIND_CMAP = "viridis"
ERROR_CMAP = "magma"
GT_COLOR = "#111111"
CNN_COLOR = "#777777"
CONTROL_COLOR = "#C77C22"
TOPO_COLOR = "#009E73"
MUTED = "#666666"
GRID = "#E5E5E5"

FS = 24
TITLE_FS = 30
PANEL_FS = 30
NOTE_FS = 24

# Persistence-survival is intentionally a high-persistence-tail view.
SURVIVAL_MIN_PERSISTENCE = 3.0  # m/s


# =============================================================================
# IO / validation
# =============================================================================

def require_file(path: Path):
    if not path.is_file():
        raise FileNotFoundError(str(path))


def read_csv(path: Path):
    require_file(path)
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def repo_path(value):
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def load_model_arrays(directory: Path, label: str):
    for name in ("idx.npy", "dataGT.npy", "dataSR.npy"):
        require_file(directory / name)

    return {
        "label": label,
        "idx": np.load(directory / "idx.npy"),
        "gt": np.load(directory / "dataGT.npy", mmap_mode="r"),
        "sr": np.load(directory / "dataSR.npy", mmap_mode="r"),
    }


def row_for_sample(idx, sample_id, label):
    idx = np.asarray(idx).astype(int)
    hits = np.flatnonzero(idx == int(sample_id))
    if len(hits) != 1:
        raise RuntimeError(
            f"{label}: expected exactly one row for sample {sample_id}, got {len(hits)}"
        )
    return int(hits[0])


def speed(uv):
    uv = np.asarray(uv, dtype=np.float64)
    if uv.ndim != 3 or uv.shape[-1] != 2:
        raise RuntimeError(f"Expected [H,W,2] vector field, got {uv.shape}")
    return np.hypot(uv[..., 0], uv[..., 1])


def crop(a):
    if a.shape[0] < Y0 + PATCH or a.shape[1] < X0 + PATCH:
        raise RuntimeError(f"Array too small for {PATCH}x{PATCH} crop: {a.shape}")
    return np.asarray(a[Y0:Y0 + PATCH, X0:X0 + PATCH])


def build_sweep_index():
    out = {}
    for r in read_csv(W22_SWEEP):
        key = (r["run"], int(r["sample"]))
        if key in out:
            raise RuntimeError(f"Duplicate W22 row: {key}")
        out[key] = r
    return out


def read_diagrams(sweep_index, run, sample):
    key = (run, sample)
    if key not in sweep_index:
        raise RuntimeError(f"Missing W22 sweep row: {key}")

    row = sweep_index[key]
    gt_path = repo_path(row["gt_path"])
    sr_path = repo_path(row["sr_path"])
    require_file(gt_path)
    require_file(sr_path)

    gt_pd, _ = canonical.read_pd(str(gt_path))
    sr_pd, _ = canonical.read_pd(str(sr_path))
    return gt_pd, sr_pd, row


def positive_diagram(D, label):
    D = np.asarray(D, dtype=np.float64)
    if D.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    if D.ndim != 2 or D.shape[1] != 2:
        raise RuntimeError(f"{label}: expected [N,2], got {D.shape}")

    p = D[:, 1] - D[:, 0]
    if np.any(p < 0):
        raise RuntimeError(f"{label}: found negative persistence")

    D = D[p > 0]
    if len(D):
        order = np.lexsort((D[:, 1], D[:, 0]))
        D = D[order]
    return D


def clean_pd(pd, label):
    return {
        0: positive_diagram(pd[0], f"{label} D0"),
        1: positive_diagram(pd[1], f"{label} D1"),
    }


def assert_same_gt(*named_pds):
    ref_name, ref = named_pds[0]
    ref = clean_pd(ref, f"{ref_name} GT")
    for name, pd in named_pds[1:]:
        other = clean_pd(pd, f"{name} GT")
        for dim in (0, 1):
            if ref[dim].shape != other[dim].shape or not np.array_equal(ref[dim], other[dim]):
                raise RuntimeError(
                    f"GT PD mismatch: {ref_name} vs {name}, D{dim}: "
                    f"{ref[dim].shape} vs {other[dim].shape}"
                )
    return ref


def combine_pd(pd):
    parts = [np.asarray(pd[d]) for d in (0, 1) if len(pd[d])]
    return np.vstack(parts) if parts else np.empty((0, 2), dtype=np.float64)


def persistence(D):
    D = np.asarray(D, dtype=np.float64)
    return D[:, 1] - D[:, 0] if len(D) else np.empty(0, dtype=np.float64)


def logical_metric(row, key):
    choices = {
        "db": ("bottleneck_all", "db_all", "dB_all"),
        "w2inf": ("w2inf_all", "w2_inf_all"),
        "w22": ("w22_all",),
    }[key]
    for c in choices:
        if c in row and row[c] not in ("", None):
            return float(row[c])
    return None


# =============================================================================
# Plot helpers
# =============================================================================

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": FS,
    "axes.titlesize": TITLE_FS,
    "axes.labelsize": FS,
    "xtick.labelsize": FS,
    "ytick.labelsize": FS,
    "legend.fontsize": FS,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def save_exact(fig, stem: str):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{stem}.pdf", facecolor="white")
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=300, facecolor="white")
    plt.close(fig)


def save_image_panel(arr, stem, cmap, vmin, vmax, size):
    fig = plt.figure(figsize=(size, size))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(
        arr,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        origin="upper",
        interpolation="nearest",
    )
    ax.set_axis_off()
    save_exact(fig, stem)


def save_vertical_colorbar(stem, cmap, vmin, vmax, height, ticks, ticklabels=None):
    fig = plt.figure(figsize=(CBAR_W, height))
    # Give tick labels enough horizontal room.
    ax = fig.add_axes([0.20, 0.08, 0.26, 0.84])
    cb = ColorbarBase(
        ax,
        cmap=plt.get_cmap(cmap),
        norm=Normalize(vmin=vmin, vmax=vmax),
        orientation="vertical",
        extend="max" if vmax > max(ticks) else "neither",
    )
    cb.set_ticks(ticks)
    if ticklabels is not None:
        cb.set_ticklabels(ticklabels)
    cb.ax.tick_params(labelsize=FS, width=1.0, length=5)
    cb.outline.set_linewidth(1.0)
    fig.text(0.07, 0.985, "m/s", ha="left", va="top", fontsize=FS, color=MUTED)
    save_exact(fig, stem)


def survival_counts(D, thresholds):
    p = persistence(D)
    return np.asarray([np.count_nonzero(p >= t) for t in thresholds])


def draw_survival_axis(ax, gt, cnn, control, topo):
    all_p = np.concatenate([
        persistence(gt),
        persistence(cnn),
        persistence(control),
        persistence(topo),
    ])
    all_p = all_p[all_p > 0]
    if len(all_p) == 0:
        raise RuntimeError("No positive-persistence pairs found.")

    lo = max(SURVIVAL_MIN_PERSISTENCE, float(np.min(all_p)))
    hi = float(np.max(all_p))
    thresholds = np.linspace(lo, hi, 260)

    specs = [
        (gt, GT_COLOR, "-", 2.6, "GT"),
        (cnn, CNN_COLOR, "--", 2.5, "Pretrained CNN"),
        (control, CONTROL_COLOR, ":", 2.8, "Reconstruction-only"),
        (topo, TOPO_COLOR, "-", 3.0, "Topology-inspired"),
    ]
    for D, color, ls, lw, label in specs:
        ax.step(
            thresholds,
            survival_counts(D, thresholds),
            where="post",
            color=color,
            linestyle=ls,
            linewidth=lw,
            label=label,
        )

    ax.set_yscale("log")
    ax.set_ylim(bottom=0.8)
    ax.set_xlim(lo, hi)
    ax.set_xlabel("Persistence threshold (m/s)")
    ax.set_ylabel("Surviving finite pairs")
    ax.grid(True, which="both", axis="both", color=GRID, linewidth=0.8, zorder=0)
    ax.legend(loc="upper right", frameon=False, fontsize=FS)
    ax.tick_params(labelsize=FS)


def save_survival(gt, cnn, control, topo):
    fig = plt.figure(figsize=(SURV_W, SURV_H))
    ax = fig.add_axes([0.15, 0.20, 0.81, 0.67])
    draw_survival_axis(ax, gt, cnn, control, topo)
    fig.text(
        0.53, 0.95,
        "High-persistence survival",
        ha="center", va="top",
        fontsize=TITLE_FS, fontweight="bold",
    )
    fig.text(
        0.53, 0.885,
        r"combined finite $D_0 + D_1$ pairs",
        ha="center", va="top",
        fontsize=FS, color=MUTED,
    )
    save_exact(fig, "sample078_persistence_survival")


# =============================================================================
# Main
# =============================================================================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cnn = load_model_arrays(CNN_DIR, "Pretrained CNN")
    control = load_model_arrays(CONTROL_DIR, "Reconstruction-only")
    topo = load_model_arrays(TOPO_DIR, "Topology-inspired")

    r_cnn = row_for_sample(cnn["idx"], SAMPLE_ID, "CNN")
    r_ctl = row_for_sample(control["idx"], SAMPLE_ID, "Control")
    r_top = row_for_sample(topo["idx"], SAMPLE_ID, "Topology")

    print("=" * 76)
    print(f"SAMPLE_ID = {SAMPLE_ID}")
    print(f"rows: CNN={r_cnn}, control={r_ctl}, topology={r_top}")

    # Verify the actual GT vector fields correspond exactly to the same sample.
    gt_cnn = np.asarray(cnn["gt"][r_cnn])
    gt_ctl = np.asarray(control["gt"][r_ctl])
    gt_top = np.asarray(topo["gt"][r_top])
    d_ctl = float(np.max(np.abs(gt_cnn - gt_ctl)))
    d_top = float(np.max(np.abs(gt_cnn - gt_top)))
    print(f"GT vector alignment max abs: CNN/control={d_ctl:.3e}, CNN/topology={d_top:.3e}")
    if d_ctl > 1e-6 or d_top > 1e-6:
        raise RuntimeError("GT vector arrays are not aligned.")

    # 160x160 scalar-speed topology domain.
    gt = crop(speed(gt_cnn))
    cnn_s = crop(speed(cnn["sr"][r_cnn]))
    ctl_s = crop(speed(control["sr"][r_ctl]))
    top_s = crop(speed(topo["sr"][r_top]))

    fields = {
        "GT": gt,
        "CNN": cnn_s,
        "Control": ctl_s,
        "Topology": top_s,
    }

    errors = {
        "CNN": np.abs(cnn_s - gt),
        "Control": np.abs(ctl_s - gt),
        "Topology": np.abs(top_s - gt),
    }

    field_vmin = 0.0
    field_vmax = max(float(np.max(x)) for x in fields.values())
    err_vmin = 0.0
    err_vmax = max(float(np.max(x)) for x in errors.values())

    print("\n160x160 speed ranges:")
    for name, x in fields.items():
        print(
            f"  {name:10s}: min={np.min(x):.5f}, max={np.max(x):.5f}, "
            f"mean={np.mean(x):.5f}"
        )

    print("\nAbsolute-error summaries:")
    for name, e in errors.items():
        print(
            f"  {name:10s}: MAE={np.mean(e):.6f}, "
            f"RMSE={np.sqrt(np.mean(e**2)):.6f}, max={np.max(e):.6f}"
        )

    print(f"\nshared field scale = [{field_vmin:.3f}, {field_vmax:.3f}] m/s")
    print(f"shared error scale = [{err_vmin:.3f}, {err_vmax:.3f}] m/s")

    # Save plain square panels. Labels live in the poster, not inside these files.
    save_image_panel(gt, "sample078_gt_field", WIND_CMAP, field_vmin, field_vmax, FIELD_SIZE)
    save_image_panel(cnn_s, "sample078_pretrained_cnn_field", WIND_CMAP, field_vmin, field_vmax, FIELD_SIZE)
    save_image_panel(ctl_s, "sample078_reconstruction_only_field", WIND_CMAP, field_vmin, field_vmax, FIELD_SIZE)
    save_image_panel(top_s, "sample078_topology_inspired_field", WIND_CMAP, field_vmin, field_vmax, FIELD_SIZE)

    save_image_panel(errors["CNN"], "sample078_pretrained_cnn_error", ERROR_CMAP, err_vmin, err_vmax, ERROR_SIZE)
    save_image_panel(errors["Control"], "sample078_reconstruction_only_error", ERROR_CMAP, err_vmin, err_vmax, ERROR_SIZE)
    save_image_panel(errors["Topology"], "sample078_topology_inspired_error", ERROR_CMAP, err_vmin, err_vmax, ERROR_SIZE)

    # Colorbar ticks chosen for readability, but scales use the exact shared maxima.
    speed_ticks = [0.0, 10.0, 20.0]
    speed_ticks = [t for t in speed_ticks if t <= field_vmax + 1e-9]
    if not speed_ticks or speed_ticks[-1] < 0.75 * field_vmax:
        speed_ticks.append(field_vmax)

    error_ticks = [0.0, err_vmax / 2.0, err_vmax]
    error_labels = [f"{x:.0f}" if x >= 10 else f"{x:.1f}" for x in error_ticks]

    save_vertical_colorbar(
        "sample078_speed_colorbar",
        WIND_CMAP, field_vmin, field_vmax,
        SPEED_CBAR_H,
        speed_ticks,
        [f"{x:g}" for x in speed_ticks],
    )
    save_vertical_colorbar(
        "sample078_error_colorbar",
        ERROR_CMAP, err_vmin, err_vmax,
        ERROR_CBAR_H,
        error_ticks,
        error_labels,
    )

    # Actual audited PDs.
    sweep = build_sweep_index()

    cnn_gt_raw, cnn_pd_raw, cnn_row = read_diagrams(sweep, CNN_RUN, SAMPLE_ID)
    ctl_gt_raw, ctl_pd_raw, ctl_row = read_diagrams(sweep, CONTROL_RUN, SAMPLE_ID)
    top_gt_raw, top_pd_raw, top_row = read_diagrams(sweep, TOPO_RUN, SAMPLE_ID)

    gt_pd = assert_same_gt(
        ("CNN", cnn_gt_raw),
        ("Control", ctl_gt_raw),
        ("Topology", top_gt_raw),
    )
    cnn_pd = clean_pd(cnn_pd_raw, "CNN")
    ctl_pd = clean_pd(ctl_pd_raw, "Control")
    top_pd = clean_pd(top_pd_raw, "Topology")

    gt_all = combine_pd(gt_pd)
    cnn_all = combine_pd(cnn_pd)
    ctl_all = combine_pd(ctl_pd)
    top_all = combine_pd(top_pd)

    print("\nPositive-persistence pair counts (D0 + D1):")
    print(f"  GT        : {len(gt_all)}")
    print(f"  CNN       : {len(cnn_all)}")
    print(f"  Control   : {len(ctl_all)}")
    print(f"  Topology  : {len(top_all)}")

    save_survival(gt_all, cnn_all, ctl_all, top_all)

    # Print audited sample-level PD distances when the sweep exposes them.
    metric_rows = {"CNN": cnn_row, "Control": ctl_row, "Topology": top_row}
    metric_text = []
    print("\nSample-level audited PD distances:")
    for name, row in metric_rows.items():
        vals = {
            "db": logical_metric(row, "db"),
            "w2inf": logical_metric(row, "w2inf"),
            "w22": logical_metric(row, "w22"),
        }
        metric_text.append((name, vals))
        print(
            f"  {name:10s}: "
            f"dB={vals['db'] if vals['db'] is not None else 'NA'}  "
            f"W2inf={vals['w2inf'] if vals['w2inf'] is not None else 'NA'}  "
            f"W22={vals['w22'] if vals['w22'] is not None else 'NA'}"
        )

    # ---------------------------------------------------------------------
    # Content-only preview using the final placement logic.
    # The actual poster should still use the separate assets above.
    # ---------------------------------------------------------------------
    fig = plt.figure(figsize=(PREVIEW_W, PREVIEW_H))
    fig.patch.set_facecolor("white")

    left_x = 0.0
    left_w = 21.60
    col_gap = 0.20
    col_w = (left_w - 3 * col_gap) / 4.0
    centers = [
        left_x + col_w / 2 + i * (col_w + col_gap)
        for i in range(4)
    ]

    # Preview-coordinate heights in inches from bottom.
    field_y = 1.81
    error_y = 0.17

    # Helper to convert inch coordinates into figure fractions.
    def ax_at(x, y, w, h):
        return fig.add_axes([x / PREVIEW_W, y / PREVIEW_H, w / PREVIEW_W, h / PREVIEW_H])

    for center, arr, title in zip(
        centers,
        [gt, cnn_s, ctl_s, top_s],
        ["Ground truth", "Pretrained CNN", "Reconstruction-only", "Topology-inspired"],
    ):
        ax = ax_at(center - FIELD_SIZE / 2, field_y, FIELD_SIZE, FIELD_SIZE)
        ax.imshow(arr, cmap=WIND_CMAP, vmin=field_vmin, vmax=field_vmax,
                  origin="upper", interpolation="nearest")
        ax.set_axis_off()
        fig.text(
            center / PREVIEW_W,
            (field_y + FIELD_SIZE + 0.10) / PREVIEW_H,
            title,
            ha="center", va="bottom",
            fontsize=PANEL_FS, fontweight="bold",
        )

    # Bottom error maps under the three predictions.
    err_specs = [
        (centers[1], errors["CNN"], r"$|\mathrm{Pretrained}-\mathrm{GT}|$"),
        (centers[2], errors["Control"], r"$|\mathrm{Recon.}-\mathrm{GT}|$"),
        (centers[3], errors["Topology"], r"$|\mathrm{Topology}-\mathrm{GT}|$"),
    ]
    for center, arr, title in err_specs:
        ax = ax_at(center - ERROR_SIZE / 2, error_y, ERROR_SIZE, ERROR_SIZE)
        ax.imshow(arr, cmap=ERROR_CMAP, vmin=err_vmin, vmax=err_vmax,
                  origin="upper", interpolation="nearest")
        ax.set_axis_off()
        fig.text(
            center / PREVIEW_W,
            (error_y + ERROR_SIZE + 0.04) / PREVIEW_H,
            title,
            ha="center", va="bottom",
            fontsize=NOTE_FS,
        )

    fig.text(
        centers[0] / PREVIEW_W,
        (error_y + ERROR_SIZE / 2) / PREVIEW_H,
        "reference",
        ha="center", va="center",
        fontsize=NOTE_FS, color=MUTED, style="italic",
    )

    # Shared colorbars.
    cb_x = 21.85
    cax1 = ax_at(cb_x, field_y, 0.23, FIELD_SIZE)
    cb1 = ColorbarBase(cax1, cmap=plt.get_cmap(WIND_CMAP),
                       norm=Normalize(field_vmin, field_vmax), orientation="vertical")
    cb1.ax.tick_params(labelsize=NOTE_FS)
    fig.text((cb_x - 0.02) / PREVIEW_W,
             (field_y + FIELD_SIZE + 0.10) / PREVIEW_H,
             "m/s", ha="left", va="bottom", fontsize=NOTE_FS, color=MUTED)

    cax2 = ax_at(cb_x, error_y, 0.23, ERROR_SIZE)
    cb2 = ColorbarBase(cax2, cmap=plt.get_cmap(ERROR_CMAP),
                       norm=Normalize(err_vmin, err_vmax), orientation="vertical")
    cb2.ax.tick_params(labelsize=NOTE_FS)
    fig.text((cb_x - 0.02) / PREVIEW_W,
             (error_y + ERROR_SIZE + 0.04) / PREVIEW_H,
             "m/s", ha="left", va="bottom", fontsize=NOTE_FS, color=MUTED)

    # Survival at far right.
    surv_x = 23.10
    surv_w = PREVIEW_W - surv_x
    ax = ax_at(surv_x + 0.70, 0.68, surv_w - 0.95, 3.28)
    draw_survival_axis(ax, gt_all, cnn_all, ctl_all, top_all)
    fig.text(
        (surv_x + surv_w / 2) / PREVIEW_W,
        4.51 / PREVIEW_H,
        "High-persistence survival",
        ha="center", va="top",
        fontsize=TITLE_FS, fontweight="bold",
    )
    fig.text(
        (surv_x + surv_w / 2) / PREVIEW_W,
        4.13 / PREVIEW_H,
        r"combined finite $D_0 + D_1$ pairs",
        ha="center", va="top",
        fontsize=NOTE_FS, color=MUTED,
    )

    save_exact(fig, "sample078_section4_content_preview")

    # Audit summary.
    summary = OUT_DIR / "sample078_section4_audit.txt"
    with summary.open("w") as f:
        f.write(f"SAMPLE_ID={SAMPLE_ID}\n")
        f.write(f"rows CNN/control/topology={r_cnn}/{r_ctl}/{r_top}\n")
        f.write(f"GT_alignment_max_abs CNN-control={d_ctl:.12g}\n")
        f.write(f"GT_alignment_max_abs CNN-topology={d_top:.12g}\n")
        f.write(f"field_scale=[{field_vmin:.8g},{field_vmax:.8g}] m/s\n")
        f.write(f"error_scale=[{err_vmin:.8g},{err_vmax:.8g}] m/s\n")
        for name, e in errors.items():
            f.write(
                f"{name}_error MAE={np.mean(e):.8g} "
                f"RMSE={np.sqrt(np.mean(e**2)):.8g} "
                f"MAX={np.max(e):.8g}\n"
            )
        f.write(
            f"PD_counts GT/CNN/control/topology="
            f"{len(gt_all)}/{len(cnn_all)}/{len(ctl_all)}/{len(top_all)}\n"
        )
        for name, vals in metric_text:
            f.write(
                f"{name}_PD dB={vals['db']} W2inf={vals['w2inf']} W22={vals['w22']}\n"
            )

    print("\nWrote:")
    for p in sorted(OUT_DIR.iterdir()):
        if p.is_file():
            print(" ", p.name)
    print("=" * 76)


if __name__ == "__main__":
    main()
