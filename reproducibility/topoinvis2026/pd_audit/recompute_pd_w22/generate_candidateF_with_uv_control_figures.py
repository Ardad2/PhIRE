#!/usr/bin/env python3
"""
Generate publication-oriented REAL-DATA figures for the matched-control comparison:

    GT
    pretrained CNN
    Candidate UV (L_uv-only matched fine-tuning control)
    Candidate F (grad + repaired E2 topology-aware fine-tuning)

The overview intentionally resembles the earlier poster layout:
    - CNN / Ablation / Candidate-F PD overlays across the top
    - one persistence-survival panel
    - GT / CNN / Ablation / Candidate-F scalar fields in one aligned row
    - CNN / Ablation / Candidate-F absolute-error maps directly beneath
    - one clean summary band, not floating annotation boxes

The W2,2 diagnostic is a separate 2 x 3 figure:
    columns = CNN, Ablation, Candidate F
    rows    = D0, D1

All numerical topology values come from the frozen W22 audit.
All plotted fields and PDs come from the actual benchmark artifacts.
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

from scipy.optimize import linear_sum_assignment


# =============================================================================
# Paths / experiment IDs
# =============================================================================

ROOT = Path.home() / "PhIRE"

AUDIT = Path(
    os.environ.get(
        "AUDIT",
        str(Path.home() / "phire_runtime_audit_20260809_221548"),
    )
)

W22 = Path(
    os.environ.get(
        "W22",
        str(AUDIT / "recompute_pd_w22"),
    )
)

CANONICAL_DIR = AUDIT / "recompute_pd"
sys.path.insert(0, str(CANONICAL_DIR))

import canonical_pd_pilot as canonical


CNN_DIR = ROOT / "data_out_fixed" / "wind_mrhr_cnn"

UV_DIR = (
    ROOT
    / "data_out"
    / "wind_finetune_candidateUV_expanded2688"
)

F_DIR = (
    ROOT
    / "data_out"
    / "wind_finetune_candidateF_grad_E2_low_expanded2688"
)

UV_EVAL_CSV = (
    ROOT
    / "ttk_runs_fixed"
    / "topology_finetuning"
    / "candidateUV_expanded2688_eval"
    / "all_sample_metrics_candidateUV_expanded2688.csv"
)

F_EVAL_CSV = (
    ROOT
    / "ttk_runs_fixed"
    / "topology_finetuning"
    / "candidateF_grad_E2_low_expanded2688_eval"
    / "all_sample_metrics_candidateF_grad_E2_low_expanded2688.csv"
)

W22_SWEEP = W22 / "w22_full_sweep.csv"

NEAR_TIE_MASTER = (
    W22
    / "near_tie_candidateF_grad_E2_vs_cnn_master.csv"
)

OUT_DIR = (
    ROOT
    / "figures"
    / "candidateF_uv_control_real"
)


CNN_RUN = "cnn"

UV_RUN = (
    "topology_finetuning/"
    "candidateUV_expanded2688_topology"
)

F_RUN = (
    "topology_finetuning/"
    "candidateF_grad_E2_low_expanded2688_topology"
)


CNN_METHOD = "cnn"
UV_METHOD = "candidateUV_expanded2688"
F_METHOD = "candidateF_grad_E2_low_expanded2688"


DEFAULT_SAMPLES = [78, 71, 80, 63, 69]

# Display-only threshold. Numerical distances use every strictly
# positive-persistence finite point.
DEFAULT_PD_DISPLAY_THRESHOLD = 3.0

DEFAULT_TOP_MATCHES = 30

X0 = 0
Y0 = 0
PATCH = 160


# =============================================================================
# Generic IO
# =============================================================================

def require_file(path):
    if not path.is_file():
        raise FileNotFoundError(str(path))


def read_csv(path):
    require_file(path)
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def repo_path(value):
    p = Path(value)
    if p.is_absolute():
        return p
    return ROOT / p


def load_arrays(directory):
    for name in ["idx.npy", "dataGT.npy", "dataSR.npy"]:
        require_file(directory / name)

    return {
        "idx": np.load(directory / "idx.npy"),
        "gt": np.load(directory / "dataGT.npy", mmap_mode="r"),
        "sr": np.load(directory / "dataSR.npy", mmap_mode="r"),
    }


def build_method_index(path):
    rows = read_csv(path)
    out = {}

    for r in rows:
        method = r["method"]
        sample = int(r["sample_idx"])
        key = (method, sample)

        if key in out:
            raise RuntimeError(f"Duplicate conventional-metric row: {key}")

        out[key] = r

    return out


def build_sweep_index():
    rows = read_csv(W22_SWEEP)
    out = {}

    for r in rows:
        key = (r["run"], int(r["sample"]))

        if key in out:
            raise RuntimeError(f"Duplicate W22 sweep row: {key}")

        out[key] = r

    return out


def build_near_tie_index():
    rows = read_csv(NEAR_TIE_MASTER)
    out = {}

    for r in rows:
        sample = int(r["sample"])
        if sample in out:
            raise RuntimeError(f"Duplicate near-tie sample: {sample}")
        out[sample] = r

    if len(out) != 168:
        raise RuntimeError(
            f"Expected 168 frozen near-tie rows, found {len(out)}"
        )

    return out


def row_metric(row, logical_name):
    candidates = {
        "db": [
            "bottleneck_all",
            "db_all",
            "dB_all",
        ],
        "w2inf": [
            "w2inf_all",
            "w2_inf_all",
        ],
        "w22": [
            "w22_all",
        ],
    }[logical_name]

    for name in candidates:
        if name in row and row[name] not in ("", None):
            value = float(row[name])
            if not math.isfinite(value):
                raise RuntimeError(
                    f"Nonfinite {logical_name}: column={name}, value={row[name]}"
                )
            return value

    raise KeyError(
        f"Could not find {logical_name}; tried {candidates}. "
        f"Available columns: {list(row.keys())}"
    )


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


# =============================================================================
# Array / field helpers
# =============================================================================

def crop_field(x):
    h, w = x.shape[:2]

    if h < Y0 + PATCH or w < X0 + PATCH:
        raise RuntimeError(
            f"Array is too small for {PATCH}x{PATCH} topology crop: {x.shape}"
        )

    return x[
        Y0:Y0 + PATCH,
        X0:X0 + PATCH,
        ...
    ]


def vector_to_speed(x):
    x = np.asarray(x, dtype=np.float64)

    if x.ndim != 3 or x.shape[-1] != 2:
        raise ValueError(f"Expected [H,W,2], got {x.shape}")

    return np.hypot(x[..., 0], x[..., 1])


def validate_idx_alignment(*named):
    reference_name, reference = named[0]
    expected = np.asarray(reference["idx"])

    if len(expected) != 168:
        raise RuntimeError(
            f"{reference_name}: expected 168 indices, found {len(expected)}"
        )

    for name, data in named[1:]:
        idx = np.asarray(data["idx"])

        if not np.array_equal(expected, idx):
            raise RuntimeError(
                f"Index mismatch: {reference_name} vs {name}"
            )


def validate_sample_gt_alignment(sample, cnn, uv, cand):
    A = np.asarray(cnn["gt"][sample])
    B = np.asarray(uv["gt"][sample])
    C = np.asarray(cand["gt"][sample])

    uv_diff = float(np.max(np.abs(A - B)))
    f_diff = float(np.max(np.abs(A - C)))

    if uv_diff > 1e-6 or f_diff > 1e-6:
        raise RuntimeError(
            f"GT alignment failed sample={sample}: "
            f"CNN-vs-UV={uv_diff}, CNN-vs-F={f_diff}"
        )

    print(
        "GT array alignment: PASS "
        f"(CNN-vs-UV={uv_diff:.3e}, CNN-vs-F={f_diff:.3e})"
    )


# =============================================================================
# Persistence-diagram helpers
# =============================================================================

def persistence(D):
    D = np.asarray(D, dtype=np.float64)

    if len(D) == 0:
        return np.empty(0, dtype=np.float64)

    return D[:, 1] - D[:, 0]


def canonical_positive_pd(D, label):
    D = np.asarray(D, dtype=np.float64)

    if D.size == 0:
        return D.reshape(0, 2), 0

    if D.ndim != 2 or D.shape[1] != 2:
        raise RuntimeError(f"{label}: invalid shape {D.shape}")

    pers = D[:, 1] - D[:, 0]

    if np.any(pers < 0.0):
        bad = D[pers < 0.0]
        raise RuntimeError(
            f"{label}: negative-persistence points: {bad[:5]}"
        )

    zero_count = int(np.count_nonzero(pers == 0.0))
    D = D[pers > 0.0]

    if len(D) == 0:
        return D.reshape(0, 2), zero_count

    order = np.lexsort((D[:, 1], D[:, 0]))
    return D[order], zero_count


def canonicalize_pd_dict(pd, label):
    result = {}

    for dim in [0, 1]:
        result[dim], zero_count = canonical_positive_pd(
            pd[dim],
            f"{label} D{dim}",
        )

        print(
            f"{label} D{dim}: "
            f"positive_pairs={len(result[dim])}, "
            f"zero_removed={zero_count}"
        )

    return result


def assert_same_gt(sample, gt_candidates):
    """
    gt_candidates: list of (label, pd_dict)
    """

    canonical = []

    for label, pd in gt_candidates:
        canonical.append(
            (
                label,
                canonicalize_pd_dict(
                    pd,
                    f"{label} GT sample={sample}",
                ),
            )
        )

    ref_label, ref = canonical[0]

    for other_label, other in canonical[1:]:
        for dim in [0, 1]:
            A = ref[dim]
            B = other[dim]

            if A.shape != B.shape or not np.array_equal(A, B):
                max_abs = (
                    float(np.max(np.abs(A - B)))
                    if A.shape == B.shape and A.size
                    else math.inf
                )

                raise RuntimeError(
                    "GT positive-persistence mismatch "
                    f"sample={sample} D{dim}: "
                    f"{ref_label}={A.shape}, "
                    f"{other_label}={B.shape}, "
                    f"max_abs={max_abs}"
                )

    print(
        "GT positive-persistence equality across "
        "CNN / Ablation / Candidate F: PASS"
    )

    return ref


def combine_pd(pd):
    parts = []

    for dim in [0, 1]:
        D = np.asarray(pd[dim], dtype=np.float64)
        if len(D):
            parts.append(D)

    if not parts:
        return np.empty((0, 2), dtype=np.float64)

    return np.vstack(parts)


def display_pd(D, threshold):
    D = np.asarray(D, dtype=np.float64)

    if len(D) == 0:
        return D

    return D[persistence(D) >= threshold]


def pd_limits(*diagrams):
    arrays = [
        np.asarray(D, dtype=np.float64)
        for D in diagrams
        if len(D)
    ]

    if not arrays:
        return 0.0, 1.0

    X = np.vstack(arrays)
    lo = float(np.min(X))
    hi = float(np.max(X))

    span = max(hi - lo, 1.0)
    pad = 0.04 * span

    return min(0.0, lo - pad), hi + pad


# =============================================================================
# W22 assignment helpers
# =============================================================================

def diagonal_projection(D):
    D = np.asarray(D, dtype=np.float64)

    if len(D) == 0:
        return np.empty_like(D)

    mid = (D[:, 0] + D[:, 1]) / 2.0
    return np.column_stack([mid, mid])


def pairwise_l2(A, B):
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)

    if len(A) == 0 or len(B) == 0:
        return np.empty((len(A), len(B)), dtype=np.float64)

    db = A[:, None, 0] - B[None, :, 0]
    dd = A[:, None, 1] - B[None, :, 1]

    return np.hypot(db, dd)


def diagonal_cost_l2(D):
    return persistence(D) / math.sqrt(2.0)


def w22_assignment(A, B):
    """
    A = GT diagram
    B = reconstruction diagram
    """

    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)

    m = len(A)
    n = len(B)

    if m == 0 and n == 0:
        return 0.0, []

    C = np.full(
        (m + n, m + n),
        np.inf,
        dtype=np.float64,
    )

    if m and n:
        C[:m, :n] = pairwise_l2(A, B)

    da = diagonal_cost_l2(A)

    for i in range(m):
        C[i, n + i] = da[i]

    db = diagonal_cost_l2(B)

    for j in range(n):
        C[m + j, j] = db[j]

    if m and n:
        C[m:, n:] = 0.0

    rows, cols = linear_sum_assignment(C * C)
    costs = C[rows, cols]

    distance = float(
        np.sqrt(
            np.sum(costs * costs)
        )
    )

    matches = []

    for r, c, cost in zip(rows, cols, costs):
        if r < m and c < n:
            matches.append({
                "type": "real-real",
                "gt": A[r],
                "model": B[c],
                "cost": float(cost),
            })

        elif r < m and c >= n:
            proj = diagonal_projection(A[r:r+1])[0]
            matches.append({
                "type": "gt-diagonal",
                "gt": A[r],
                "model": proj,
                "cost": float(cost),
            })

        elif r >= m and c < n:
            proj = diagonal_projection(B[c:c+1])[0]
            matches.append({
                "type": "model-diagonal",
                "gt": proj,
                "model": B[c],
                "cost": float(cost),
            })

    return distance, matches


# =============================================================================
# Conventional metrics
# =============================================================================

def finite_float(value, label):
    x = float(value)

    if not math.isfinite(x):
        raise RuntimeError(f"Nonfinite {label}: {value}")

    return x


def conventional_record(index, method, sample):
    key = (method, sample)

    if key not in index:
        raise RuntimeError(f"Missing conventional row: {key}")

    return index[key]


def conventional_gaps(reference, candidate):
    ref_psnr = finite_float(reference["psnruv"], "reference psnruv")
    cand_psnr = finite_float(candidate["psnruv"], "candidate psnruv")

    ref_mae = finite_float(reference["speed_mae"], "reference speed_mae")
    cand_mae = finite_float(candidate["speed_mae"], "candidate speed_mae")

    ref_rmse = finite_float(reference["speed_rmse"], "reference speed_rmse")
    cand_rmse = finite_float(candidate["speed_rmse"], "candidate speed_rmse")

    return {
        "abs_psnr": abs(cand_psnr - ref_psnr),
        "rel_mae": abs(cand_mae - ref_mae) / ref_mae,
        "rel_rmse": abs(cand_rmse - ref_rmse) / ref_rmse,
    }


# =============================================================================
# Plot styling
# =============================================================================

def set_publication_defaults():
    plt.rcParams.update({
        "font.size": 9.5,
        "axes.titlesize": 11.0,
        "axes.labelsize": 9.5,
        "axes.linewidth": 0.8,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 7.8,
        "figure.titlesize": 18,
        "font.family": "DejaVu Sans",
        "savefig.dpi": 300,
    })


def plot_combined_pd(
    ax,
    gt_pd,
    model_pd,
    model_label,
    color,
    threshold,
    limits,
):
    lo, hi = limits

    ax.plot(
        [lo, hi],
        [lo, hi],
        "--",
        color="0.68",
        linewidth=0.9,
        zorder=0,
    )

    gt0 = display_pd(gt_pd[0], threshold)
    gt1 = display_pd(gt_pd[1], threshold)
    m0 = display_pd(model_pd[0], threshold)
    m1 = display_pd(model_pd[1], threshold)

    if len(gt0):
        ax.scatter(
            gt0[:, 0],
            gt0[:, 1],
            s=22,
            marker="o",
            facecolors="none",
            edgecolors="black",
            linewidths=0.9,
            label="GT $D_0$",
            zorder=3,
        )

    if len(gt1):
        ax.scatter(
            gt1[:, 0],
            gt1[:, 1],
            s=21,
            marker="s",
            facecolors="none",
            edgecolors="0.25",
            linewidths=0.85,
            label="GT $D_1$",
            zorder=3,
        )

    if len(m0):
        ax.scatter(
            m0[:, 0],
            m0[:, 1],
            s=24,
            marker="x",
            color=color,
            linewidths=1.1,
            label=f"{model_label} $D_0$",
            zorder=4,
        )

    if len(m1):
        ax.scatter(
            m1[:, 0],
            m1[:, 1],
            s=28,
            marker="+",
            color=color,
            linewidths=1.1,
            label=f"{model_label} $D_1$",
            zorder=4,
        )

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel("Birth")
    ax.set_ylabel("Death")

    ax.grid(
        alpha=0.10,
        linewidth=0.5,
    )

    ax.legend(
        loc="upper left",
        frameon=True,
        framealpha=0.90,
        borderpad=0.35,
        handletextpad=0.35,
        labelspacing=0.25,
    )


def survival_curve(D, thresholds):
    p = persistence(D)

    return np.asarray([
        np.count_nonzero(p >= t)
        for t in thresholds
    ])


def plot_survival(
    ax,
    gt,
    cnn,
    uv,
    cand,
    display_threshold,
):
    arrays = [
        persistence(gt),
        persistence(cnn),
        persistence(uv),
        persistence(cand),
    ]

    positive = np.concatenate(arrays)
    positive = positive[positive > 0]

    if not len(positive):
        return

    lower = max(
        float(display_threshold),
        float(np.min(positive)),
    )

    upper = float(np.max(positive))

    thresholds = np.linspace(
        lower,
        upper,
        240,
    )

    ax.step(
        thresholds,
        survival_curve(gt, thresholds),
        where="post",
        color="black",
        linewidth=1.7,
        label="GT",
    )

    ax.step(
        thresholds,
        survival_curve(cnn, thresholds),
        where="post",
        color="#1696a7",
        linestyle="--",
        linewidth=1.7,
        label="CNN",
    )

    ax.step(
        thresholds,
        survival_curve(uv, thresholds),
        where="post",
        color="#2e8b57",
        linestyle=":",
        linewidth=1.9,
        label=r"Ablation ($L_{uv}$ only)",
    )

    ax.step(
        thresholds,
        survival_curve(cand, thresholds),
        where="post",
        color="#d95f02",
        linewidth=1.8,
        label="Candidate F",
    )

    ax.set_yscale("log")
    ax.set_ylim(bottom=0.8)

    ax.set_xlabel("Persistence threshold")
    ax.set_ylabel("Surviving finite pairs")

    ax.grid(
        alpha=0.18,
        which="both",
        linewidth=0.6,
    )

    ax.legend(
        loc="upper right",
        frameon=True,
        framealpha=0.92,
    )


def format_metric_chain(name, values):
    cnn, uv, cand = values

    return (
        f"{name}: "
        f"CNN {cnn:.3f} → "
        f"Abl. {uv:.3f} → "
        f"F {cand:.3f}"
    )


def draw_summary_band(
    ax,
    sample,
    metrics,
    cnn_f_gaps,
    uv_f_gaps,
    original_rank,
):
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    # separators only — no floating rounded boxes
    for x in [1/3, 2/3]:
        ax.plot(
            [x, x],
            [0.10, 0.92],
            color="0.78",
            linewidth=1.0,
            transform=ax.transAxes,
        )

    # Section 1
    ax.text(
        1/6,
        0.88,
        "Topology chain (lower is better)",
        ha="center",
        va="top",
        fontweight="bold",
        fontsize=10.2,
        transform=ax.transAxes,
    )

    ax.text(
        1/6,
        0.61,
        "\n".join([
            format_metric_chain(
                r"$d_B$",
                (
                    metrics["cnn"]["db"],
                    metrics["uv"]["db"],
                    metrics["f"]["db"],
                ),
            ),
            format_metric_chain(
                r"$W_{2,\infty}$",
                (
                    metrics["cnn"]["w2inf"],
                    metrics["uv"]["w2inf"],
                    metrics["f"]["w2inf"],
                ),
            ),
            format_metric_chain(
                r"$W_{2,2}$",
                (
                    metrics["cnn"]["w22"],
                    metrics["uv"]["w22"],
                    metrics["f"]["w22"],
                ),
            ),
        ]),
        ha="center",
        va="center",
        fontsize=9.0,
        linespacing=1.45,
        transform=ax.transAxes,
    )

    # Section 2
    ax.text(
        0.5,
        0.88,
        "Conventional fidelity gaps",
        ha="center",
        va="top",
        fontweight="bold",
        fontsize=10.2,
        transform=ax.transAxes,
    )

    ax.text(
        0.5,
        0.61,
        "\n".join([
            (
                "F vs CNN "
                f"(original closeness rank {original_rank}/168): "
                f"|ΔPSNR|={cnn_f_gaps['abs_psnr']:.3f} dB, "
                f"MAE={100*cnn_f_gaps['rel_mae']:.1f}%, "
                f"RMSE={100*cnn_f_gaps['rel_rmse']:.1f}%"
            ),
            (
                r"F vs $L_{uv}$ ablation: "
                f"|ΔPSNR|={uv_f_gaps['abs_psnr']:.3f} dB, "
                f"MAE={100*uv_f_gaps['rel_mae']:.1f}%, "
                f"RMSE={100*uv_f_gaps['rel_rmse']:.1f}%"
            ),
        ]),
        ha="center",
        va="center",
        fontsize=8.7,
        linespacing=1.55,
        transform=ax.transAxes,
        wrap=True,
    )

    # Section 3
    aux_wins = all(
        metrics["f"][m] < metrics["uv"][m]
        for m in ["db", "w2inf", "w22"]
    )

    ax.text(
        5/6,
        0.88,
        "Matched-control interpretation",
        ha="center",
        va="top",
        fontweight="bold",
        fontsize=10.2,
        transform=ax.transAxes,
    )

    if aux_wins:
        interp = (
            "Candidate F is lower than the matched "
            r"$L_{uv}$-only fine-tuning control under all three "
            "validated PD metrics. This sample therefore supports "
            "an auxiliary-loss effect beyond ordinary fine-tuning."
        )
    else:
        interp = (
            "The matched control does not show an all-three "
            "Candidate-F advantage on this sample; interpret the "
            "topological gain relative to CNN with caution."
        )

    ax.text(
        5/6,
        0.56,
        interp,
        ha="center",
        va="center",
        fontsize=8.8,
        linespacing=1.35,
        transform=ax.transAxes,
        wrap=True,
    )


# =============================================================================
# Overview figure
# =============================================================================

def make_overview(
    sample,
    gt_speed,
    cnn_speed,
    uv_speed,
    f_speed,
    gt_pd,
    cnn_pd,
    uv_pd,
    f_pd,
    metrics,
    cnn_f_gaps,
    uv_f_gaps,
    original_rank,
    out_png,
    out_pdf,
    display_threshold,
):
    set_publication_defaults()

    gt_all = combine_pd(gt_pd)
    cnn_all = combine_pd(cnn_pd)
    uv_all = combine_pd(uv_pd)
    f_all = combine_pd(f_pd)

    # One shared PD scale across all three overview panels.
    pd_display_arrays = []

    for pd in [gt_pd, cnn_pd, uv_pd, f_pd]:
        for dim in [0, 1]:
            D = display_pd(pd[dim], display_threshold)
            if len(D):
                pd_display_arrays.append(D)

    limits = pd_limits(*pd_display_arrays)

    fig = plt.figure(
        figsize=(20.0, 10.8),
        constrained_layout=False,
    )

    gs = fig.add_gridspec(
        nrows=4,
        ncols=4,
        height_ratios=[
            1.38,
            1.05,
            0.92,
            0.52,
        ],
        left=0.045,
        right=0.970,
        top=0.885,
        bottom=0.070,
        hspace=0.42,
        wspace=0.34,
    )

    fig.suptitle(
        (
            f"Sample {sample}: "
            "Topology-Aware Gains Beyond Matched Fine-Tuning"
        ),
        fontsize=19,
        fontweight="bold",
        y=0.965,
    )

    fig.text(
        0.5,
        0.927,
        (
            r"CNN = pretrained baseline; Ablation = matched $L_{uv}$-only "
            "fine-tuning; Candidate F = "
            r"$L_{uv}+L_{grad}+$ repaired E2 supervision."
        ),
        ha="center",
        va="center",
        fontsize=10.8,
    )

    # ---------------------------------------------------------
    # Row 1: three poster-style combined PD panels + survival
    # ---------------------------------------------------------

    pd_specs = [
        (
            gs[0, 0],
            cnn_pd,
            "CNN",
            "#1696a7",
            "CNN vs GT",
        ),
        (
            gs[0, 1],
            uv_pd,
            r"Ablation ($L_{uv}$ only)",
            "#2e8b57",
            "Ablation vs GT",
        ),
        (
            gs[0, 2],
            f_pd,
            "Candidate F",
            "#d95f02",
            "Candidate F vs GT",
        ),
    ]

    for spec, model_pd, label, color, title in pd_specs:
        ax = fig.add_subplot(spec)

        plot_combined_pd(
            ax,
            gt_pd,
            model_pd,
            label,
            color,
            display_threshold,
            limits,
        )

        ax.set_title(
            title,
            fontweight="bold",
            pad=14,
        )

        ax.text(
            0.5,
            1.01,
            (
                "combined display; "
                f"persistence ≥ {display_threshold:g}"
            ),
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=8.0,
        )

    ax_surv = fig.add_subplot(gs[0, 3])

    plot_survival(
        ax_surv,
        gt_all,
        cnn_all,
        uv_all,
        f_all,
        display_threshold,
    )

    ax_surv.set_title(
        "Persistence Survival",
        fontweight="bold",
        pad=14,
    )

    ax_surv.text(
        0.5,
        1.01,
        (
            "combined finite D0+D1; "
            f"thresholds ≥ {display_threshold:g}"
        ),
        transform=ax_surv.transAxes,
        ha="center",
        va="bottom",
        fontsize=8.0,
    )

    # ---------------------------------------------------------
    # Row 2: aligned GT / CNN / Ablation / F scalar fields
    # ---------------------------------------------------------

    fields = [
        ("GT", gt_speed),
        ("CNN", cnn_speed),
        ("Ablation", uv_speed),
        ("Candidate F", f_speed),
    ]

    vmin = float(
        min(np.min(arr) for _, arr in fields)
    )

    vmax = float(
        max(np.max(arr) for _, arr in fields)
    )

    field_axes = []

    for col, (label, arr) in enumerate(fields):
        ax = fig.add_subplot(gs[1, col])
        field_axes.append(ax)

        im = ax.imshow(
            arr,
            origin="lower",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )

        ax.set_title(
            label,
            fontweight="bold",
            pad=6,
        )

        ax.set_xticks([])
        ax.set_yticks([])

    cbar = fig.colorbar(
        im,
        ax=field_axes,
        fraction=0.014,
        pad=0.010,
    )

    cbar.set_label(
        "Wind-speed magnitude"
    )

    # ---------------------------------------------------------
    # Row 3: errors aligned directly beneath learned methods
    # ---------------------------------------------------------

    err_gs = gs[2, :].subgridspec(
        1,
        3,
        wspace=0.20,
    )

    errors = [
        ("|CNN - GT|", np.abs(cnn_speed - gt_speed)),
        ("|Ablation - GT|", np.abs(uv_speed - gt_speed)),
        ("|Candidate F - GT|", np.abs(f_speed - gt_speed)),
    ]

    err_vmax = float(
        max(np.max(arr) for _, arr in errors)
    )

    error_axes = []

    for col, (label, arr) in enumerate(errors):
        ax = fig.add_subplot(err_gs[0, col])
        error_axes.append(ax)

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
            pad=6,
        )

        ax.set_xticks([])
        ax.set_yticks([])

    cbar_err = fig.colorbar(
        im_err,
        ax=error_axes,
        fraction=0.015,
        pad=0.010,
    )

    cbar_err.set_label(
        "Absolute speed error"
    )

    # ---------------------------------------------------------
    # Row 4: one clean summary band
    # ---------------------------------------------------------

    ax_summary = fig.add_subplot(gs[3, :])

    draw_summary_band(
        ax_summary,
        sample,
        metrics,
        cnn_f_gaps,
        uv_f_gaps,
        original_rank,
    )

    fig.text(
        0.5,
        0.018,
        (
            "Actual benchmark artifacts only. PD panels are display-thresholded "
            f"at persistence ≥ {display_threshold:g}; audited distances use all "
            "strictly positive finite pairs and preserve homology dimension. "
            "Field/error maps use the same 160×160 topology domain."
        ),
        ha="center",
        va="bottom",
        fontsize=8.2,
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


# =============================================================================
# Matching-detail figure
# =============================================================================

def prepare_matching_panel(gt, model, top_matches):
    distance, matches = w22_assignment(gt, model)

    nonzero = [
        m
        for m in matches
        if m["cost"] > 0
    ]

    nonzero.sort(
        key=lambda x: x["cost"],
        reverse=True,
    )

    shown = nonzero[:min(top_matches, len(nonzero))]

    total_sq = distance * distance

    shown_sq = sum(
        m["cost"] * m["cost"]
        for m in shown
    )

    fraction = (
        shown_sq / total_sq
        if total_sq > 0
        else 1.0
    )

    return {
        "distance": distance,
        "shown": shown,
        "shown_fraction": fraction,
    }


def plot_matching_panel(
    ax,
    gt,
    model,
    color,
    prepared,
    title,
    limits,
    global_max_cost,
):
    lo, hi = limits

    ax.plot(
        [lo, hi],
        [lo, hi],
        "--",
        color="0.68",
        linewidth=0.9,
        zorder=0,
    )

    if len(gt):
        ax.scatter(
            gt[:, 0],
            gt[:, 1],
            s=8,
            facecolors="none",
            edgecolors="0.35",
            linewidths=0.40,
            alpha=0.22,
            zorder=1,
        )

    if len(model):
        ax.scatter(
            model[:, 0],
            model[:, 1],
            s=8,
            marker="x",
            color=color,
            linewidths=0.40,
            alpha=0.22,
            zorder=1,
        )

    for m in prepared["shown"]:
        frac = (
            m["cost"]
            / max(global_max_cost, 1e-12)
        )

        alpha = 0.28 + 0.62 * frac
        lw = 0.75 + 2.25 * frac

        p = m["gt"]
        q = m["model"]

        linestyle = (
            "-"
            if m["type"] == "real-real"
            else "--"
        )

        ax.plot(
            [p[0], q[0]],
            [p[1], q[1]],
            color=color,
            alpha=alpha,
            linewidth=lw,
            linestyle=linestyle,
            zorder=2,
        )

        if m["type"] == "real-real":
            ax.scatter(
                [p[0]],
                [p[1]],
                s=25,
                facecolors="white",
                edgecolors="black",
                linewidths=0.8,
                zorder=4,
            )

            ax.scatter(
                [q[0]],
                [q[1]],
                s=27,
                marker="x",
                color=color,
                linewidths=1.0,
                zorder=5,
            )

        elif m["type"] == "gt-diagonal":
            ax.scatter(
                [p[0]],
                [p[1]],
                s=25,
                facecolors="white",
                edgecolors="black",
                linewidths=0.8,
                zorder=4,
            )

            ax.scatter(
                [q[0]],
                [q[1]],
                s=16,
                marker="D",
                facecolors="0.75",
                edgecolors="0.35",
                linewidths=0.6,
                zorder=5,
            )

        elif m["type"] == "model-diagonal":
            ax.scatter(
                [p[0]],
                [p[1]],
                s=16,
                marker="D",
                facecolors="0.75",
                edgecolors="0.35",
                linewidths=0.6,
                zorder=5,
            )

            ax.scatter(
                [q[0]],
                [q[1]],
                s=27,
                marker="x",
                color=color,
                linewidths=1.0,
                zorder=5,
            )

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel("Birth")
    ax.set_ylabel("Death")

    ax.set_title(
        (
            f"{title}\n"
            f"$W_{{2,2}}={prepared['distance']:.4f}$; "
            f"displayed={100*prepared['shown_fraction']:.1f}% "
            r"of $W_{2,2}^2$"
        ),
        fontweight="bold",
        fontsize=9.6,
        pad=7,
    )

    ax.grid(
        alpha=0.10,
        linewidth=0.5,
    )


def make_matching_detail(
    sample,
    gt_pd,
    cnn_pd,
    uv_pd,
    f_pd,
    frozen_metrics,
    out_png,
    out_pdf,
    top_matches,
):
    set_publication_defaults()

    models = [
        ("cnn", "CNN", cnn_pd, "#1696a7"),
        ("uv", r"Ablation ($L_{uv}$ only)", uv_pd, "#2e8b57"),
        ("f", "Candidate F", f_pd, "#d95f02"),
    ]

    prepared = {}

    for key, _, pd, _ in models:
        for dim in [0, 1]:
            prepared[(key, dim)] = prepare_matching_panel(
                gt_pd[dim],
                pd[dim],
                top_matches,
            )

    shown_costs = [
        m["cost"]
        for item in prepared.values()
        for m in item["shown"]
    ]

    global_max_cost = (
        max(shown_costs)
        if shown_costs
        else 1.0
    )

    d0_limits = pd_limits(
        gt_pd[0],
        cnn_pd[0],
        uv_pd[0],
        f_pd[0],
    )

    d1_limits = pd_limits(
        gt_pd[1],
        cnn_pd[1],
        uv_pd[1],
        f_pd[1],
    )

    fig = plt.figure(
        figsize=(16.0, 10.2),
        constrained_layout=False,
    )

    gs = fig.add_gridspec(
        nrows=2,
        ncols=3,
        left=0.055,
        right=0.975,
        top=0.855,
        bottom=0.155,
        hspace=0.31,
        wspace=0.22,
    )

    fig.suptitle(
        (
            f"Sample {sample}: "
            r"$W_{2,2}$ Matching With Matched Fine-Tuning Control"
        ),
        fontsize=18,
        fontweight="bold",
        y=0.965,
    )

    fig.text(
        0.5,
        0.920,
        (
            "Solid = real-to-real; dashed = assignment to diagonal. "
            "Line weight uses one shared absolute assignment-cost scale."
        ),
        ha="center",
        va="center",
        fontsize=9.8,
    )

    recomputed = {}

    for col, (key, label, pd, color) in enumerate(models):
        for row, dim in enumerate([0, 1]):
            ax = fig.add_subplot(gs[row, col])

            limits = (
                d0_limits
                if dim == 0
                else d1_limits
            )

            item = prepared[(key, dim)]

            plot_matching_panel(
                ax,
                gt_pd[dim],
                pd[dim],
                color,
                item,
                f"{label} vs GT — $D_{dim}$",
                limits,
                global_max_cost,
            )

            recomputed[(key, dim)] = item["distance"]

    aggregate = {}

    for key, _, _, _ in models:
        aggregate[key] = math.hypot(
            recomputed[(key, 0)],
            recomputed[(key, 1)],
        )

    # Exact guard against the frozen W22 audit.
    for key in ["cnn", "uv", "f"]:
        diff = abs(
            aggregate[key]
            - frozen_metrics[key]["w22"]
        )

        print(
            f"W22 figure-time check {key}: "
            f"recomputed={aggregate[key]:.15g} "
            f"frozen={frozen_metrics[key]['w22']:.15g} "
            f"diff={diff:.3e}"
        )

        if diff > 1e-10:
            raise RuntimeError(
                f"W22 figure-time audit mismatch for {key}: {diff}"
            )

    legend_handles = [
        Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            markerfacecolor="white",
            markeredgecolor="black",
            label="GT point",
        ),
        Line2D(
            [0], [0],
            marker="x",
            linestyle="None",
            color="#1696a7",
            label="CNN point",
        ),
        Line2D(
            [0], [0],
            marker="x",
            linestyle="None",
            color="#2e8b57",
            label="Ablation point",
        ),
        Line2D(
            [0], [0],
            marker="x",
            linestyle="None",
            color="#d95f02",
            label="Candidate-F point",
        ),
        Line2D(
            [0], [0],
            linestyle="-",
            color="0.35",
            label="real-to-real",
        ),
        Line2D(
            [0], [0],
            linestyle="--",
            marker="D",
            markersize=4,
            color="0.35",
            label="to diagonal",
        ),
    ]

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.080),
        ncol=6,
        frameon=False,
        fontsize=8.7,
    )

    fig.text(
        0.5,
        0.025,
        (
            r"Aggregate $W_{2,2}$: "
            f"CNN={aggregate['cnn']:.4f}, "
            f"Ablation={aggregate['uv']:.4f}, "
            f"Candidate F={aggregate['f']:.4f}. "
            f"At most {top_matches} largest nonzero assignments per panel "
            "are drawn; numeric distances use every strictly positive finite pair."
        ),
        ha="center",
        va="bottom",
        fontsize=8.8,
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

    return aggregate


# =============================================================================
# Main
# =============================================================================

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

    cnn = load_arrays(CNN_DIR)
    uv = load_arrays(UV_DIR)
    cand = load_arrays(F_DIR)

    validate_idx_alignment(
        ("CNN", cnn),
        ("Ablation", uv),
        ("Candidate F", cand),
    )

    sweep = build_sweep_index()
    near_tie = build_near_tie_index()

    uv_eval = build_method_index(UV_EVAL_CSV)
    f_eval = build_method_index(F_EVAL_CSV)

    manifest_rows = []

    for sample in args.samples:
        if sample not in range(168):
            raise ValueError(f"Invalid sample: {sample}")

        print()
        print("=" * 92)
        print(f"SAMPLE {sample}")
        print("=" * 92)

        validate_sample_gt_alignment(
            sample,
            cnn,
            uv,
            cand,
        )

        gt_uv = crop_field(cnn["gt"][sample])
        cnn_uv = crop_field(cnn["sr"][sample])
        abl_uv = crop_field(uv["sr"][sample])
        f_uv = crop_field(cand["sr"][sample])

        gt_speed = vector_to_speed(gt_uv)
        cnn_speed = vector_to_speed(cnn_uv)
        uv_speed = vector_to_speed(abl_uv)
        f_speed = vector_to_speed(f_uv)

        cnn_gt_raw, cnn_pd_raw, cnn_row = read_diagrams(
            sweep,
            CNN_RUN,
            sample,
        )

        uv_gt_raw, uv_pd_raw, uv_row = read_diagrams(
            sweep,
            UV_RUN,
            sample,
        )

        f_gt_raw, f_pd_raw, f_row = read_diagrams(
            sweep,
            F_RUN,
            sample,
        )

        gt_pd = assert_same_gt(
            sample,
            [
                ("CNN", cnn_gt_raw),
                ("Ablation", uv_gt_raw),
                ("Candidate F", f_gt_raw),
            ],
        )

        cnn_pd = canonicalize_pd_dict(
            cnn_pd_raw,
            f"CNN sample={sample}",
        )

        uv_pd = canonicalize_pd_dict(
            uv_pd_raw,
            f"Ablation sample={sample}",
        )

        f_pd = canonicalize_pd_dict(
            f_pd_raw,
            f"Candidate F sample={sample}",
        )

        metrics = {
            "cnn": {
                "db": row_metric(cnn_row, "db"),
                "w2inf": row_metric(cnn_row, "w2inf"),
                "w22": row_metric(cnn_row, "w22"),
            },
            "uv": {
                "db": row_metric(uv_row, "db"),
                "w2inf": row_metric(uv_row, "w2inf"),
                "w22": row_metric(uv_row, "w22"),
            },
            "f": {
                "db": row_metric(f_row, "db"),
                "w2inf": row_metric(f_row, "w2inf"),
                "w22": row_metric(f_row, "w22"),
            },
        }

        cnn_eval_row = conventional_record(
            f_eval,
            CNN_METHOD,
            sample,
        )

        f_eval_row = conventional_record(
            f_eval,
            F_METHOD,
            sample,
        )

        uv_eval_row = conventional_record(
            uv_eval,
            UV_METHOD,
            sample,
        )

        # The CNN rows should be identical across the candidate-specific
        # cheap-evaluation tables for the relevant fields. Use the F table
        # as the canonical CNN record and check the UV table.
        uv_table_cnn = conventional_record(
            uv_eval,
            CNN_METHOD,
            sample,
        )

        for field in ["psnruv", "speed_mae", "speed_rmse"]:
            a = finite_float(
                cnn_eval_row[field],
                f"F-table CNN {field}",
            )
            b = finite_float(
                uv_table_cnn[field],
                f"UV-table CNN {field}",
            )

            if abs(a - b) > 1e-12:
                raise RuntimeError(
                    f"CNN conventional metric mismatch sample={sample} "
                    f"field={field}: {a} vs {b}"
                )

        cnn_f_gaps = conventional_gaps(
            cnn_eval_row,
            f_eval_row,
        )

        uv_f_gaps = conventional_gaps(
            uv_eval_row,
            f_eval_row,
        )

        original_rank = int(
            near_tie[sample]["conventional_rank"]
        )

        stem = f"sample_{sample:03d}"

        overview_png = (
            args.out_dir
            / f"{stem}_uv_control_overview.png"
        )

        overview_pdf = (
            args.out_dir
            / f"{stem}_uv_control_overview.pdf"
        )

        match_png = (
            args.out_dir
            / f"{stem}_uv_control_w22_matching.png"
        )

        match_pdf = (
            args.out_dir
            / f"{stem}_uv_control_w22_matching.pdf"
        )

        make_overview(
            sample,
            gt_speed,
            cnn_speed,
            uv_speed,
            f_speed,
            gt_pd,
            cnn_pd,
            uv_pd,
            f_pd,
            metrics,
            cnn_f_gaps,
            uv_f_gaps,
            original_rank,
            overview_png,
            overview_pdf,
            args.pd_display_threshold,
        )

        aggregate = make_matching_detail(
            sample,
            gt_pd,
            cnn_pd,
            uv_pd,
            f_pd,
            metrics,
            match_png,
            match_pdf,
            args.top_matches,
        )

        print("W22 audit consistency: PASS")

        print("overview:", overview_png)
        print("matching:", match_png)

        manifest_rows.append({
            "sample": sample,
            "original_cnn_near_tie_rank": original_rank,
            "pd_display_threshold":
                args.pd_display_threshold,

            "cnn_db": metrics["cnn"]["db"],
            "ablation_db": metrics["uv"]["db"],
            "candidateF_db": metrics["f"]["db"],

            "cnn_w2inf": metrics["cnn"]["w2inf"],
            "ablation_w2inf": metrics["uv"]["w2inf"],
            "candidateF_w2inf": metrics["f"]["w2inf"],

            "cnn_w22": metrics["cnn"]["w22"],
            "ablation_w22": metrics["uv"]["w22"],
            "candidateF_w22": metrics["f"]["w22"],

            "candidateF_vs_ablation_all3":
                int(
                    metrics["f"]["db"] < metrics["uv"]["db"]
                    and metrics["f"]["w2inf"] < metrics["uv"]["w2inf"]
                    and metrics["f"]["w22"] < metrics["uv"]["w22"]
                ),

            "F_vs_CNN_abs_delta_psnruv":
                cnn_f_gaps["abs_psnr"],

            "F_vs_CNN_relative_gap_speed_mae":
                cnn_f_gaps["rel_mae"],

            "F_vs_CNN_relative_gap_speed_rmse":
                cnn_f_gaps["rel_rmse"],

            "F_vs_ablation_abs_delta_psnruv":
                uv_f_gaps["abs_psnr"],

            "F_vs_ablation_relative_gap_speed_mae":
                uv_f_gaps["rel_mae"],

            "F_vs_ablation_relative_gap_speed_rmse":
                uv_f_gaps["rel_rmse"],

            "overview_png": str(overview_png),
            "overview_pdf": str(overview_pdf),
            "matching_png": str(match_png),
            "matching_pdf": str(match_pdf),

            "cnn_w22_recomputed": aggregate["cnn"],
            "ablation_w22_recomputed": aggregate["uv"],
            "candidateF_w22_recomputed": aggregate["f"],
        })

    manifest = (
        args.out_dir
        / "uv_control_figure_manifest.csv"
    )

    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(manifest_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    print()
    print("=" * 92)
    print("MATCHED-CONTROL REAL-DATA FIGURE GENERATION COMPLETE")
    print("=" * 92)
    print("output directory:", args.out_dir)
    print("manifest:", manifest)


if __name__ == "__main__":
    main()
