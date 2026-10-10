#!/usr/bin/env python3

"""
Generate REAL-DATA near-tie visualizations for:

    CNN vs candidateF_grad_E2_low_expanded2688

No synthetic / AI-generated field content is used.

Inputs
------
- Fixed CNN benchmark arrays
- Candidate-F benchmark arrays
- Frozen W22 full-sweep manifest
- Frozen candidate PD robustness table
- Audited canonical PD parser

Outputs per selected sample
---------------------------
1. Poster-style real-data overview
2. Detailed W22 assignment/matching diagnostic

Primary predeclared samples:
    78, 71, 80, 63, 69
"""

import argparse
import csv
import math
import os
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch

from scipy.optimize import linear_sum_assignment


# ============================================================================
# Paths
# ============================================================================

ROOT = Path.home() / "PhIRE"

AUDIT = Path(
    os.environ.get(
        "AUDIT",
        str(
            Path.home()
            / "phire_runtime_audit_20260809_221548"
        ),
    )
)

W22 = Path(
    os.environ.get(
        "W22",
        str(AUDIT / "recompute_pd_w22"),
    )
)

CANONICAL_DIR = (
    AUDIT
    / "recompute_pd"
)

sys.path.insert(
    0,
    str(CANONICAL_DIR),
)

import canonical_pd_pilot as canonical


# ============================================================================
# Data paths
# ============================================================================

CNN_DIR = (
    ROOT
    / "data_out_fixed"
    / "wind_mrhr_cnn"
)

F_DIR = (
    ROOT
    / "data_out"
    / "wind_finetune_candidateF_grad_E2_low_expanded2688"
)

W22_SWEEP = (
    W22
    / "w22_full_sweep.csv"
)

ROBUSTNESS_CSV = (
    W22
    / "candidate_pd_robustness_samples.csv"
)

NEAR_TIE_MASTER = (
    W22
    / "near_tie_candidateF_grad_E2_vs_cnn_master.csv"
)

OUT_DIR = (
    ROOT
    / "figures"
    / "candidateF_near_tie_real"
)


CNN_RUN = "cnn"

F_RUN = (
    "topology_finetuning/"
    "candidateF_grad_E2_low_expanded2688_topology"
)

ROBUSTNESS_CANDIDATE = (
    "candidateF_grad_E2_low_2688"
)

DEFAULT_SAMPLES = [
    78,
    71,
    80,
    63,
    69,
]


# ============================================================================
# Topology field crop
#
# Candidate-F topology extraction used the top-left 160 x 160 crop.
# Keep the visual scalar fields identical to the spatial domain used to
# construct the PDs.
# ============================================================================

X0 = 0
Y0 = 0
PATCH = 160


# ============================================================================
# Display controls
# ============================================================================

DEFAULT_PD_DISPLAY_THRESHOLD = 5.0

# Number of most expensive W22 assignments shown per homology dimension
# in the detailed matching diagnostic.
DEFAULT_TOP_MATCHES = 30


# ============================================================================
# IO helpers
# ============================================================================

def load_csv(path):

    with path.open(newline="") as f:
        return list(
            csv.DictReader(f)
        )


def require_file(path):

    if not path.is_file():
        raise FileNotFoundError(
            str(path)
        )


def load_arrays(directory):

    names = [
        "idx.npy",
        "dataGT.npy",
        "dataSR.npy",
    ]

    for name in names:
        require_file(
            directory / name
        )

    return {
        "idx": np.load(
            directory / "idx.npy"
        ),
        "gt": np.load(
            directory / "dataGT.npy",
            mmap_mode="r",
        ),
        "sr": np.load(
            directory / "dataSR.npy",
            mmap_mode="r",
        ),
    }


# ============================================================================
# Wind-speed helpers
# ============================================================================

def vector_to_speed(x):

    x = np.asarray(
        x,
        dtype=np.float64,
    )

    if (
        x.ndim != 3
        or x.shape[-1] != 2
    ):
        raise ValueError(
            f"Expected [H,W,2], got {x.shape}"
        )

    return np.sqrt(
        x[..., 0] ** 2
        + x[..., 1] ** 2
    )


def crop_field(x):

    h, w = x.shape[:2]

    if (
        h < Y0 + PATCH
        or w < X0 + PATCH
    ):
        raise RuntimeError(
            "Array is too small for "
            f"{PATCH}x{PATCH} crop: "
            f"{x.shape}"
        )

    return x[
        Y0:Y0 + PATCH,
        X0:X0 + PATCH,
        ...
    ]


# ============================================================================
# PD helpers
# ============================================================================

def combine_pd(pd_dict):

    parts = []

    for dim in [0, 1]:

        arr = np.asarray(
            pd_dict[dim],
            dtype=np.float64,
        )

        if len(arr):
            parts.append(arr)

    if not parts:
        return np.empty(
            (0, 2),
            dtype=np.float64,
        )

    return np.vstack(parts)


def persistence(D):

    if len(D) == 0:
        return np.empty(
            0,
            dtype=np.float64,
        )

    return (
        D[:, 1]
        - D[:, 0]
    )


def display_pd(D, threshold):

    if len(D) == 0:
        return D

    mask = (
        persistence(D)
        >= threshold
    )

    return D[mask]


def diagonal_projection(D):

    if len(D) == 0:
        return np.empty_like(D)

    mid = (
        D[:, 0]
        + D[:, 1]
    ) / 2.0

    return np.column_stack(
        [mid, mid]
    )


# ============================================================================
# W22 assignment
#
# Same standard construction used in the audited custom W22 implementation.
# ============================================================================

def pairwise_l2(A, B):

    if (
        len(A) == 0
        or len(B) == 0
    ):
        return np.empty(
            (len(A), len(B)),
            dtype=np.float64,
        )

    db = (
        A[:, None, 0]
        - B[None, :, 0]
    )

    dd = (
        A[:, None, 1]
        - B[None, :, 1]
    )

    return np.hypot(
        db,
        dd,
    )


def diagonal_cost_l2(D):

    if len(D) == 0:
        return np.empty(
            0,
            dtype=np.float64,
        )

    return (
        persistence(D)
        / math.sqrt(2.0)
    )


def w22_assignment(A, B):
    """
    A = GT diagram
    B = reconstruction diagram

    Returns:
        distance,
        match records

    match types:
        real-real
        gt-diagonal
        model-diagonal
    """

    A = np.asarray(
        A,
        dtype=np.float64,
    )

    B = np.asarray(
        B,
        dtype=np.float64,
    )

    m = len(A)
    n = len(B)

    if m == 0 and n == 0:
        return 0.0, []

    size = m + n

    C = np.full(
        (size, size),
        np.inf,
        dtype=np.float64,
    )

    # Real-real block
    if m and n:
        C[:m, :n] = pairwise_l2(
            A,
            B,
        )

    # GT -> diagonal
    da = diagonal_cost_l2(A)

    for i in range(m):
        C[
            i,
            n + i
        ] = da[i]

    # diagonal -> model
    db = diagonal_cost_l2(B)

    for j in range(n):
        C[
            m + j,
            j
        ] = db[j]

    # diagonal-diagonal zero block
    if n and m:
        C[
            m:,
            n:
        ] = 0.0

    rows, cols = (
        linear_sum_assignment(
            C * C
        )
    )

    costs = C[
        rows,
        cols
    ]

    distance = float(
        np.sqrt(
            np.sum(
                costs * costs
            )
        )
    )

    matches = []

    for r, c, cost in zip(
        rows,
        cols,
        costs,
    ):

        if r < m and c < n:

            matches.append({
                "type":
                    "real-real",
                "gt":
                    A[r],
                "model":
                    B[c],
                "cost":
                    float(cost),
            })

        elif r < m and c >= n:

            proj = diagonal_projection(
                A[r:r+1]
            )[0]

            matches.append({
                "type":
                    "gt-diagonal",
                "gt":
                    A[r],
                "model":
                    proj,
                "cost":
                    float(cost),
            })

        elif r >= m and c < n:

            proj = diagonal_projection(
                B[c:c+1]
            )[0]

            matches.append({
                "type":
                    "model-diagonal",
                "gt":
                    proj,
                "model":
                    B[c],
                "cost":
                    float(cost),
            })

        # diagonal-diagonal is irrelevant

    return distance, matches


# ============================================================================
# Resolve audited PD files
# ============================================================================

def build_sweep_index():

    rows = load_csv(
        W22_SWEEP
    )

    out = {}

    for r in rows:

        key = (
            r["run"],
            int(r["sample"]),
        )

        if key in out:
            raise RuntimeError(
                f"Duplicate sweep key: {key}"
            )

        out[key] = r

    return out


def read_diagrams(
    sweep_index,
    run,
    sample,
):

    key = (
        run,
        sample,
    )

    if key not in sweep_index:
        raise RuntimeError(
            f"Missing W22 row: {key}"
        )

    row = sweep_index[key]

    gt_path = Path(
        row["gt_path"]
    )

    sr_path = Path(
        row["sr_path"]
    )

    require_file(gt_path)
    require_file(sr_path)

    gt_pd, gt_nonfinite = (
        canonical.read_pd(
            str(gt_path)
        )
    )

    sr_pd, sr_nonfinite = (
        canonical.read_pd(
            str(sr_path)
        )
    )

    return (
        gt_pd,
        sr_pd,
        row,
    )


# ============================================================================
# Frozen robustness / near-tie metadata
# ============================================================================

def build_robustness_index():

    rows = load_csv(
        ROBUSTNESS_CSV
    )

    out = {}

    for r in rows:

        if (
            r["candidate"]
            != ROBUSTNESS_CANDIDATE
        ):
            continue

        if (
            r["baseline"]
            != "cnn"
        ):
            continue

        sample = int(
            r["sample"]
        )

        out[sample] = r

    if len(out) != 168:
        raise RuntimeError(
            "Expected 168 Candidate-F "
            "vs CNN robustness rows; "
            f"got {len(out)}"
        )

    return out


def build_near_tie_index():

    rows = load_csv(
        NEAR_TIE_MASTER
    )

    out = {}

    for r in rows:

        sample = int(
            r["sample"]
        )

        out[sample] = r

    if len(out) != 168:
        raise RuntimeError(
            "Expected 168 near-tie rows; "
            f"got {len(out)}"
        )

    return out


# ============================================================================
# Plotting helpers
# ============================================================================

def set_publication_defaults():

    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "figure.titlesize": 18,
        "font.family":
            "DejaVu Sans",
        "savefig.dpi": 300,
    })


def pd_limits(*diagrams):

    arrays = [
        D
        for D in diagrams
        if len(D)
    ]

    if not arrays:
        return 0.0, 1.0

    X = np.vstack(arrays)

    lo = float(
        np.min(X)
    )

    hi = float(
        np.max(X)
    )

    span = max(
        hi - lo,
        1.0,
    )

    pad = (
        0.04
        * span
    )

    return (
        min(0.0, lo - pad),
        hi + pad,
    )


def gt_zoom_bounds(
    gt_full,
    n_top=6,
):

    if len(gt_full) == 0:
        return None

    pers = persistence(
        gt_full
    )

    order = np.argsort(
        pers
    )[::-1]

    take = gt_full[
        order[
            :min(
                n_top,
                len(order),
            )
        ]
    ]

    xmin = float(
        np.min(
            take[:, 0]
        )
    )

    xmax = float(
        np.max(
            take[:, 0]
        )
    )

    ymin = float(
        np.min(
            take[:, 1]
        )
    )

    ymax = float(
        np.max(
            take[:, 1]
        )
    )

    dx = max(
        xmax - xmin,
        1.0,
    )

    dy = max(
        ymax - ymin,
        1.0,
    )

    return (
        xmin - 0.15 * dx,
        xmax + 0.15 * dx,
        ymin - 0.15 * dy,
        ymax + 0.15 * dy,
    )


def plot_pd_overlay(
    ax,
    gt,
    model,
    model_label,
    model_color,
    display_threshold,
    title,
):

    gt_disp = display_pd(
        gt,
        display_threshold,
    )

    model_disp = display_pd(
        model,
        display_threshold,
    )

    lo, hi = pd_limits(
        gt_disp,
        model_disp,
    )

    ax.plot(
        [lo, hi],
        [lo, hi],
        linestyle="--",
        linewidth=1.0,
        color="0.65",
        zorder=0,
    )

    if len(gt_disp):

        ax.scatter(
            gt_disp[:, 0],
            gt_disp[:, 1],
            facecolors="none",
            edgecolors="black",
            s=28,
            linewidths=1.0,
            label="GT",
            zorder=3,
        )

    if len(model_disp):

        ax.scatter(
            model_disp[:, 0],
            model_disp[:, 1],
            marker="x",
            color=model_color,
            s=30,
            linewidths=1.25,
            label=model_label,
            zorder=4,
        )

    ax.set_xlim(
        lo,
        hi,
    )

    ax.set_ylim(
        lo,
        hi,
    )

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    ax.set_xlabel(
        "Birth"
    )

    ax.set_ylabel(
        "Death"
    )

    ax.set_title(
        title,
        fontweight="bold",
        pad=16,
    )

    ax.text(
        0.5,
        1.015,
        (
            "finite D0+D1; "
            f"display persistence ≥ "
            f"{display_threshold:g}"
        ),
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=8.5,
    )

    ax.legend(
        loc="upper left",
        frameon=True,
    )

    # --------------------------------------------------------
    # GT-defined high-persistence inset
    # --------------------------------------------------------

    bounds = gt_zoom_bounds(
        gt
    )

    if bounds is not None:

        x0, x1, y0, y1 = bounds

        inset = ax.inset_axes(
            [
                0.56,
                0.05,
                0.39,
                0.33,
            ]
        )

        inset.plot(
            [lo, hi],
            [lo, hi],
            "--",
            linewidth=0.7,
            color="0.75",
        )

        if len(gt_disp):

            inset.scatter(
                gt_disp[:, 0],
                gt_disp[:, 1],
                facecolors="none",
                edgecolors="black",
                s=20,
                linewidths=0.9,
            )

        if len(model_disp):

            inset.scatter(
                model_disp[:, 0],
                model_disp[:, 1],
                marker="x",
                color=model_color,
                s=22,
                linewidths=1.0,
            )

        inset.set_xlim(
            x0,
            x1,
        )

        inset.set_ylim(
            y0,
            y1,
        )

        inset.set_title(
            "GT-defined zoom",
            fontsize=8,
        )

        inset.tick_params(
            labelsize=7,
        )


def survival_curve(
    D,
    thresholds,
):

    p = persistence(D)

    return np.asarray([
        np.count_nonzero(
            p >= t
        )
        for t in thresholds
    ])


def plot_survival(
    ax,
    gt,
    cnn,
    cand,
):

    all_p = np.concatenate([
        persistence(gt),
        persistence(cnn),
        persistence(cand),
    ])

    positive = all_p[
        all_p > 0
    ]

    if len(positive) == 0:
        return

    upper = float(
        np.max(positive)
    )

    # Start high enough to focus the
    # visually meaningful tail while using
    # complete diagrams for every count.
    lower = max(
        0.0,
        float(
            np.percentile(
                positive,
                65,
            )
        ),
    )

    if lower >= upper:
        lower = 0.0

    thresholds = np.linspace(
        lower,
        upper,
        220,
    )

    curves = {
        "GT":
            survival_curve(
                gt,
                thresholds,
            ),
        "CNN":
            survival_curve(
                cnn,
                thresholds,
            ),
        "Candidate F":
            survival_curve(
                cand,
                thresholds,
            ),
    }

    ax.step(
        thresholds,
        curves["GT"],
        where="post",
        color="black",
        linewidth=1.7,
        label="GT",
    )

    ax.step(
        thresholds,
        curves["CNN"],
        where="post",
        color="#1696a7",
        linestyle="--",
        linewidth=1.7,
        label="CNN",
    )

    ax.step(
        thresholds,
        curves["Candidate F"],
        where="post",
        color="#d95f02",
        linewidth=1.7,
        label="Candidate F",
    )

    ax.set_yscale(
        "log"
    )

    ax.set_ylim(
        bottom=0.8
    )

    ax.set_xlabel(
        "Persistence threshold"
    )

    ax.set_ylabel(
        "Surviving finite pairs"
    )

    ax.set_title(
        "Persistence Survival — High-Persistence Tail",
        fontweight="bold",
    )

    ax.text(
        0.5,
        1.01,
        "complete finite D0+D1 diagrams",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=8.5,
    )

    ax.grid(
        alpha=0.2,
        which="both",
    )

    ax.legend(
        loc="upper right"
    )


def summary_box(
    ax,
    title,
    lines,
):

    ax.axis(
        "off"
    )

    bbox = dict(
        boxstyle="round,pad=0.55",
        facecolor="white",
        edgecolor="#2b83ba",
        linewidth=1.1,
    )

    text = (
        f"$\\bf{{{title}}}$\n"
        + "\n".join(lines)
    )

    ax.text(
        0.5,
        0.5,
        text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=10,
        bbox=bbox,
    )


# ============================================================================
# Main overview figure
# ============================================================================

def make_overview(
    sample,
    gt_speed,
    cnn_speed,
    f_speed,
    gt_pd,
    cnn_pd,
    f_pd,
    robust,
    near_tie,
    out_png,
    out_pdf,
    display_threshold,
):

    gt_all = combine_pd(
        gt_pd
    )

    cnn_all = combine_pd(
        cnn_pd
    )

    f_all = combine_pd(
        f_pd
    )

    set_publication_defaults()

    fig = plt.figure(
        figsize=(18, 10.5),
        constrained_layout=False,
    )

    gs = fig.add_gridspec(
        nrows=4,
        ncols=12,
        height_ratios=[
            1.55,
            0.95,
            0.76,
            0.62,
        ],
        hspace=0.42,
        wspace=0.65,
        left=0.045,
        right=0.965,
        top=0.88,
        bottom=0.07,
    )

    fig.suptitle(
        (
            f"Sample {sample}: "
            "Candidate F Shows Better Persistence Agreement "
            "at Similar Conventional Fidelity"
        ),
        fontsize=18,
        fontweight="bold",
        y=0.94,
    )

    fig.text(
        0.02,
        0.975,
        "Representative Near-Tie Example",
        fontsize=24,
        fontweight="bold",
        color="#0072B2",
        va="top",
    )

    # --------------------------------------------------------
    # Top row: actual PDs and actual persistence survival
    # --------------------------------------------------------

    ax_pd_cnn = fig.add_subplot(
        gs[0, 0:4]
    )

    ax_pd_f = fig.add_subplot(
        gs[0, 4:8]
    )

    ax_surv = fig.add_subplot(
        gs[0, 8:12]
    )

    plot_pd_overlay(
        ax_pd_cnn,
        gt_all,
        cnn_all,
        "CNN",
        "#1696a7",
        display_threshold,
        "CNN vs GT",
    )

    plot_pd_overlay(
        ax_pd_f,
        gt_all,
        f_all,
        "Candidate F",
        "#d95f02",
        display_threshold,
        "Candidate F vs GT",
    )

    plot_survival(
        ax_surv,
        gt_all,
        cnn_all,
        f_all,
    )

    # --------------------------------------------------------
    # Middle row: actual scalar fields
    # --------------------------------------------------------

    vmin = float(
        min(
            gt_speed.min(),
            cnn_speed.min(),
            f_speed.min(),
        )
    )

    vmax = float(
        max(
            gt_speed.max(),
            cnn_speed.max(),
            f_speed.max(),
        )
    )

    cmap_field = "viridis"

    ax_gt = fig.add_subplot(
        gs[1, 0:4]
    )

    ax_cnn = fig.add_subplot(
        gs[1, 4:8]
    )

    ax_f = fig.add_subplot(
        gs[1, 8:12]
    )

    for ax, arr, label in [
        (
            ax_gt,
            gt_speed,
            "GT",
        ),
        (
            ax_cnn,
            cnn_speed,
            "CNN",
        ),
        (
            ax_f,
            f_speed,
            "Candidate F",
        ),
    ]:

        im = ax.imshow(
            arr,
            origin="lower",
            cmap=cmap_field,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )

        ax.set_title(
            label,
            fontweight="bold",
        )

        ax.set_xticks([])
        ax.set_yticks([])

    cbar = fig.colorbar(
        im,
        ax=[
            ax_gt,
            ax_cnn,
            ax_f,
        ],
        fraction=0.018,
        pad=0.015,
    )

    cbar.set_label(
        "Wind-speed magnitude"
    )

    # --------------------------------------------------------
    # Error row: REAL absolute scalar-speed errors
    # --------------------------------------------------------

    err_cnn = np.abs(
        cnn_speed
        - gt_speed
    )

    err_f = np.abs(
        f_speed
        - gt_speed
    )

    err_vmax = float(
        max(
            err_cnn.max(),
            err_f.max(),
        )
    )

    ax_err_cnn = fig.add_subplot(
        gs[2, 1:6]
    )

    ax_err_f = fig.add_subplot(
        gs[2, 6:11]
    )

    for ax, arr, label in [
        (
            ax_err_cnn,
            err_cnn,
            "|CNN - GT|",
        ),
        (
            ax_err_f,
            err_f,
            "|Candidate F - GT|",
        ),
    ]:

        im_err = ax.imshow(
            arr,
            origin="lower",
            cmap="viridis",
            vmin=0.0,
            vmax=err_vmax,
            interpolation="nearest",
        )

        ax.set_title(
            label,
            fontweight="bold",
        )

        ax.set_xticks([])
        ax.set_yticks([])

    cbar_err = fig.colorbar(
        im_err,
        ax=[
            ax_err_cnn,
            ax_err_f,
        ],
        fraction=0.022,
        pad=0.02,
    )

    cbar_err.set_label(
        "Absolute speed error"
    )

    # --------------------------------------------------------
    # Real frozen quantitative summaries
    # --------------------------------------------------------

    db_gain = (
        100.0
        * float(
            robust[
                "db_relative_improvement"
            ]
        )
    )

    w2inf_gain = (
        100.0
        * float(
            robust[
                "w2inf_relative_improvement"
            ]
        )
    )

    w22_gain = (
        100.0
        * float(
            robust[
                "w22_relative_improvement"
            ]
        )
    )

    min_gain = min(
        db_gain,
        w2inf_gain,
        w22_gain,
    )

    conv_rank = int(
        near_tie[
            "conventional_rank"
        ]
    )

    dpsnr = float(
        near_tie[
            "abs_delta_psnruv"
        ]
    )

    mae_gap = (
        100.0
        * float(
            near_tie[
                "relative_gap_speed_mae"
            ]
        )
    )

    rmse_gap = (
        100.0
        * float(
            near_tie[
                "relative_gap_speed_rmse"
            ]
        )
    )

    ax_box1 = fig.add_subplot(
        gs[3, 0:4]
    )

    ax_box2 = fig.add_subplot(
        gs[3, 4:8]
    )

    ax_box3 = fig.add_subplot(
        gs[3, 8:12]
    )

    summary_box(
        ax_box1,
        "Persistence improvement",
        [
            f"$d_B$ gain: {db_gain:.2f}%",
            (
                "$W_{2,\\infty}$ gain: "
                f"{w2inf_gain:.2f}%"
            ),
            (
                "$W_{2,2}$ gain: "
                f"{w22_gain:.2f}%"
            ),
            (
                "minimum PD gain: "
                f"{min_gain:.2f}%"
            ),
        ],
    )

    summary_box(
        ax_box2,
        "Conventional fidelity remains close",
        [
            (
                "conventional rank: "
                f"{conv_rank} / 168"
            ),
            (
                "$|\\Delta PSNR_{uv}|$: "
                f"{dpsnr:.4f} dB"
            ),
            (
                "relative speed MAE gap: "
                f"{mae_gap:.2f}%"
            ),
            (
                "relative speed RMSE gap: "
                f"{rmse_gap:.2f}%"
            ),
        ],
    )

    summary_box(
        ax_box3,
        "Interpretation",
        [
            (
                "Candidate F is conventionally close to CNN,"
            ),
            (
                "yet all three validated PD metrics favor"
            ),
            (
                "Candidate F relative to ground truth."
            ),
        ],
    )

    fig.text(
        0.5,
        0.018,
        (
            "All field maps and PD points are computed from the "
            "actual benchmark artifacts. "
            f"PD scatter displays finite D0+D1 pairs with persistence ≥ "
            f"{display_threshold:g} for readability; "
            "reported distances use the complete finite diagrams. "
            "The scalar maps show the same 160×160 domain used for topology."
        ),
        ha="center",
        va="bottom",
        fontsize=8.5,
    )

    fig.savefig(
        out_png,
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        out_pdf,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================================
# Detailed W22 assignment figure
# ============================================================================

def plot_matching_panel(
    ax,
    gt,
    model,
    model_label,
    model_color,
    top_matches,
    title,
):

    distance, matches = (
        w22_assignment(
            gt,
            model,
        )
    )

    lo, hi = pd_limits(
        gt,
        model,
    )

    ax.plot(
        [lo, hi],
        [lo, hi],
        "--",
        color="0.7",
        linewidth=1.0,
        zorder=0,
    )

    # Plot all points faintly
    if len(gt):

        ax.scatter(
            gt[:, 0],
            gt[:, 1],
            s=9,
            facecolors="none",
            edgecolors="black",
            linewidths=0.45,
            alpha=0.30,
        )

    if len(model):

        ax.scatter(
            model[:, 0],
            model[:, 1],
            s=9,
            marker="x",
            color=model_color,
            linewidths=0.45,
            alpha=0.30,
        )

    # Highest-cost W22 assignments
    real_matches = [
        m
        for m in matches
        if m["cost"] > 0
    ]

    real_matches.sort(
        key=lambda x:
            x["cost"],
        reverse=True,
    )

    shown = real_matches[
        :min(
            top_matches,
            len(real_matches),
        )
    ]

    if shown:

        costs = np.asarray(
            [
                m["cost"]
                for m in shown
            ],
            dtype=np.float64,
        )

        cmin = float(
            costs.min()
        )

        cmax = float(
            costs.max()
        )

        denom = max(
            cmax - cmin,
            1e-12,
        )

        for m in shown:

            frac = (
                m["cost"] - cmin
            ) / denom

            # darker/thicker = larger contribution
            alpha = (
                0.25
                + 0.65 * frac
            )

            lw = (
                0.8
                + 2.0 * frac
            )

            p = m["gt"]
            q = m["model"]

            ax.plot(
                [
                    p[0],
                    q[0],
                ],
                [
                    p[1],
                    q[1],
                ],
                color=model_color,
                alpha=alpha,
                linewidth=lw,
                zorder=2,
            )

            ax.scatter(
                [p[0]],
                [p[1]],
                s=30,
                facecolors="white",
                edgecolors="black",
                linewidths=0.9,
                zorder=4,
            )

            ax.scatter(
                [q[0]],
                [q[1]],
                s=30,
                marker="x",
                color=model_color,
                linewidths=1.1,
                zorder=5,
            )

    ax.set_xlim(
        lo,
        hi,
    )

    ax.set_ylim(
        lo,
        hi,
    )

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    ax.set_xlabel(
        "Birth"
    )

    ax.set_ylabel(
        "Death"
    )

    ax.set_title(
        (
            f"{title}\n"
            f"$W_{{2,2}}={distance:.4f}$"
        ),
        fontweight="bold",
    )

    return distance


def make_matching_detail(
    sample,
    gt_pd,
    cnn_pd,
    f_pd,
    out_png,
    out_pdf,
    top_matches,
):

    set_publication_defaults()

    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(12.5, 11.0),
        constrained_layout=True,
    )

    fig.suptitle(
        (
            f"Sample {sample}: "
            "Largest Contributions to the Actual "
            "$W_{2,2}$ Assignment"
        ),
        fontsize=17,
        fontweight="bold",
    )

    values = {}

    for row, dim in enumerate(
        [0, 1]
    ):

        values[
            ("cnn", dim)
        ] = plot_matching_panel(
            axes[row, 0],
            gt_pd[dim],
            cnn_pd[dim],
            "CNN",
            "#1696a7",
            top_matches,
            (
                f"CNN vs GT — $D_{dim}$"
            ),
        )

        values[
            ("f", dim)
        ] = plot_matching_panel(
            axes[row, 1],
            gt_pd[dim],
            f_pd[dim],
            "Candidate F",
            "#d95f02",
            top_matches,
            (
                f"Candidate F vs GT — $D_{dim}$"
            ),
        )

    cnn_all = math.hypot(
        values[("cnn", 0)],
        values[("cnn", 1)],
    )

    f_all = math.hypot(
        values[("f", 0)],
        values[("f", 1)],
    )

    gain = (
        100.0
        * (
            cnn_all - f_all
        )
        / cnn_all
    )

    fig.text(
        0.5,
        0.01,
        (
            f"Aggregate actual W2,2: "
            f"CNN={cnn_all:.4f}, "
            f"Candidate F={f_all:.4f}, "
            f"relative improvement={gain:.2f}%. "
            f"Only the {top_matches} largest nonzero assignment costs "
            "per panel are connected for readability; "
            "the numeric distances use every finite persistence pair."
        ),
        ha="center",
        fontsize=9,
    )

    fig.savefig(
        out_png,
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        out_pdf,
        bbox_inches="tight",
    )

    plt.close(fig)

    return {
        "cnn_w22_d0":
            values[("cnn", 0)],
        "cnn_w22_d1":
            values[("cnn", 1)],
        "cnn_w22_all":
            cnn_all,
        "f_w22_d0":
            values[("f", 0)],
        "f_w22_d1":
            values[("f", 1)],
        "f_w22_all":
            f_all,
        "relative_gain":
            gain / 100.0,
    }


# ============================================================================
# Main
# ============================================================================

def parse_args():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--samples",
        nargs="+",
        type=int,
        default=DEFAULT_SAMPLES,
    )

    ap.add_argument(
        "--pd-display-threshold",
        type=float,
        default=DEFAULT_PD_DISPLAY_THRESHOLD,
    )

    ap.add_argument(
        "--top-matches",
        type=int,
        default=DEFAULT_TOP_MATCHES,
    )

    ap.add_argument(
        "--out-dir",
        type=Path,
        default=OUT_DIR,
    )

    return ap.parse_args()


def main():

    args = parse_args()

    args.out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load actual benchmark arrays
    # --------------------------------------------------------

    cnn = load_arrays(
        CNN_DIR
    )

    cand = load_arrays(
        F_DIR
    )

    if len(cnn["idx"]) != 168:
        raise RuntimeError(
            "CNN idx.npy is not 168 samples"
        )

    if len(cand["idx"]) != 168:
        raise RuntimeError(
            "Candidate-F idx.npy is not 168 samples"
        )

    if not np.array_equal(
        cnn["idx"],
        cand["idx"],
    ):
        raise RuntimeError(
            "CNN and Candidate-F idx.npy differ"
        )

    # Strong GT alignment check
    if (
        cnn["gt"].shape
        != cand["gt"].shape
    ):
        raise RuntimeError(
            "CNN/Candidate-F GT shapes differ: "
            f"{cnn['gt'].shape} vs "
            f"{cand['gt'].shape}"
        )

    max_gt_diff = float(
        np.max(
            np.abs(
                np.asarray(cnn["gt"])
                - np.asarray(cand["gt"])
            )
        )
    )

    print(
        "CNN/Candidate-F GT "
        f"max abs difference: {max_gt_diff}"
    )

    if max_gt_diff > 1e-6:
        raise RuntimeError(
            "GT alignment failed"
        )

    # --------------------------------------------------------
    # Indices / audited metadata
    # --------------------------------------------------------

    sweep_index = (
        build_sweep_index()
    )

    robust_index = (
        build_robustness_index()
    )

    near_tie_index = (
        build_near_tie_index()
    )

    summary_rows = []

    # --------------------------------------------------------
    # Per-sample figure generation
    # --------------------------------------------------------

    for sample in args.samples:

        if sample not in range(168):
            raise ValueError(
                f"Invalid sample: {sample}"
            )

        print()
        print("=" * 80)
        print(
            f"SAMPLE {sample}"
        )
        print("=" * 80)

        # ----------------------------------------------------
        # Actual scalar fields on exact topology crop
        # ----------------------------------------------------

        gt_uv = crop_field(
            cnn["gt"][sample]
        )

        cnn_uv = crop_field(
            cnn["sr"][sample]
        )

        f_uv = crop_field(
            cand["sr"][sample]
        )

        gt_speed = vector_to_speed(
            gt_uv
        )

        cnn_speed = vector_to_speed(
            cnn_uv
        )

        f_speed = vector_to_speed(
            f_uv
        )

        # ----------------------------------------------------
        # Actual audited persistence diagrams
        # ----------------------------------------------------

        (
            gt_pd_cnn,
            cnn_pd,
            cnn_row,
        ) = read_diagrams(
            sweep_index,
            CNN_RUN,
            sample,
        )

        (
            gt_pd_f,
            f_pd,
            f_row,
        ) = read_diagrams(
            sweep_index,
            F_RUN,
            sample,
        )

        # GT PD should be identical between comparisons.
        for dim in [0, 1]:

            A = np.asarray(
                gt_pd_cnn[dim]
            )

            B = np.asarray(
                gt_pd_f[dim]
            )

            if (
                A.shape != B.shape
                or not np.array_equal(A, B)
            ):
                raise RuntimeError(
                    f"GT PD mismatch "
                    f"sample={sample} "
                    f"dim={dim}"
                )

        gt_pd = gt_pd_cnn

        robust = (
            robust_index[sample]
        )

        near_tie = (
            near_tie_index[sample]
        )

        # ----------------------------------------------------
        # Poster-style overview
        # ----------------------------------------------------

        stem = (
            f"sample_{sample:03d}"
        )

        overview_png = (
            args.out_dir
            / f"{stem}_overview.png"
        )

        overview_pdf = (
            args.out_dir
            / f"{stem}_overview.pdf"
        )

        make_overview(
            sample,
            gt_speed,
            cnn_speed,
            f_speed,
            gt_pd,
            cnn_pd,
            f_pd,
            robust,
            near_tie,
            overview_png,
            overview_pdf,
            args.pd_display_threshold,
        )

        # ----------------------------------------------------
        # Detailed W22 matching diagnostic
        # ----------------------------------------------------

        match_png = (
            args.out_dir
            / f"{stem}_w22_matching.png"
        )

        match_pdf = (
            args.out_dir
            / f"{stem}_w22_matching.pdf"
        )

        w22_detail = (
            make_matching_detail(
                sample,
                gt_pd,
                cnn_pd,
                f_pd,
                match_png,
                match_pdf,
                args.top_matches,
            )
        )

        # ----------------------------------------------------
        # Cross-check against frozen robustness W22 values
        # ----------------------------------------------------

        frozen_cnn_w22 = float(
            robust[
                "w22_baseline"
            ]
        )

        frozen_f_w22 = float(
            robust[
                "w22_candidate"
            ]
        )

        cnn_diff = abs(
            w22_detail[
                "cnn_w22_all"
            ]
            - frozen_cnn_w22
        )

        f_diff = abs(
            w22_detail[
                "f_w22_all"
            ]
            - frozen_f_w22
        )

        print(
            "W22 recomputation check:"
        )

        print(
            "  CNN:"
            f" recomputed="
            f"{w22_detail['cnn_w22_all']:.15g}"
            f" frozen="
            f"{frozen_cnn_w22:.15g}"
            f" diff={cnn_diff:.3e}"
        )

        print(
            "  F:  "
            f" recomputed="
            f"{w22_detail['f_w22_all']:.15g}"
            f" frozen="
            f"{frozen_f_w22:.15g}"
            f" diff={f_diff:.3e}"
        )

        if (
            cnn_diff > 1e-10
            or f_diff > 1e-10
        ):
            raise RuntimeError(
                "W22 figure-time recomputation "
                "does not match frozen audit"
            )

        print(
            "W22 audit consistency: PASS"
        )

        print(
            "overview:",
            overview_png,
        )

        print(
            "matching:",
            match_png,
        )

        summary_rows.append({
            "sample":
                sample,

            "conventional_rank":
                int(
                    near_tie[
                        "conventional_rank"
                    ]
                ),

            "overview_png":
                str(
                    overview_png
                ),

            "overview_pdf":
                str(
                    overview_pdf
                ),

            "matching_png":
                str(
                    match_png
                ),

            "matching_pdf":
                str(
                    match_pdf
                ),

            "cnn_w22_recomputed":
                w22_detail[
                    "cnn_w22_all"
                ],

            "cnn_w22_frozen":
                frozen_cnn_w22,

            "candidate_w22_recomputed":
                w22_detail[
                    "f_w22_all"
                ],

            "candidate_w22_frozen":
                frozen_f_w22,

            "w22_recompute_max_abs_diff":
                max(
                    cnn_diff,
                    f_diff,
                ),
        })

    # --------------------------------------------------------
    # Manifest
    # --------------------------------------------------------

    manifest = (
        args.out_dir
        / "figure_manifest.csv"
    )

    with manifest.open(
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                summary_rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            summary_rows
        )

    print()
    print("=" * 80)
    print(
        "REAL-DATA FIGURE GENERATION COMPLETE"
    )
    print("=" * 80)

    print(
        "output directory:",
        args.out_dir,
    )

    print(
        "manifest:",
        manifest,
    )


if __name__ == "__main__":
    main()
