#!/usr/bin/env python3

from pathlib import Path
import sys
import math

import numpy as np
import scipy
import vtk
import gudhi

from gudhi.wasserstein import wasserstein_distance


# ----------------------------------------------------------------------
# Import the EXACT custom implementation already used in the audit
# ----------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import canonical_pd_pilot as canonical


ROOT = Path.home() / "PhIRE"


# ----------------------------------------------------------------------
# GUDHI wrappers
# ----------------------------------------------------------------------

def gudhi_db(A, B):
    """
    Exact bottleneck distance using GUDHI.
    Ground metric is L_infinity / sup norm.
    """
    return float(
        gudhi.bottleneck_distance(
            A,
            B,
            e=0.0
        )
    )


def gudhi_w2(A, B):
    """
    W_2 using L_infinity as the ground metric.
    Only finite persistence points are supplied/considered.
    """
    return float(
        wasserstein_distance(
            A,
            B,
            matching=False,
            order=2.0,
            internal_p=np.inf,
            keep_essential_parts=False,
        )
    )


# ----------------------------------------------------------------------
# Comparison helper
# ----------------------------------------------------------------------

ABS_TOL = 1e-10
REL_TOL = 1e-12


def close(a, b):
    return math.isclose(
        a,
        b,
        rel_tol=REL_TOL,
        abs_tol=ABS_TOL
    )


def compare_case(name, A, B):

    ours_db = canonical.bottleneck(A, B)
    ours_w2 = canonical.w2(A, B)

    g_db = gudhi_db(A, B)
    g_w2 = gudhi_w2(A, B)

    diff_db = abs(ours_db - g_db)
    diff_w2 = abs(ours_w2 - g_w2)

    db_ok = close(ours_db, g_db)
    w2_ok = close(ours_w2, g_w2)

    print()
    print(name)
    print("-" * len(name))

    print(
        f"dB: ours={ours_db:.17g}  "
        f"GUDHI={g_db:.17g}  "
        f"|diff|={diff_db:.17g}  "
        f"{'PASS' if db_ok else 'FAIL'}"
    )

    print(
        f"W2: ours={ours_w2:.17g}  "
        f"GUDHI={g_w2:.17g}  "
        f"|diff|={diff_w2:.17g}  "
        f"{'PASS' if w2_ok else 'FAIL'}"
    )

    return {
        "ours_db": ours_db,
        "gudhi_db": g_db,
        "diff_db": diff_db,
        "db_ok": db_ok,
        "ours_w2": ours_w2,
        "gudhi_w2": g_w2,
        "diff_w2": diff_w2,
        "w2_ok": w2_ok,
    }


# ----------------------------------------------------------------------
# Environment provenance
# ----------------------------------------------------------------------

print("=" * 80)
print("ENVIRONMENT")
print("=" * 80)

print("Python:", sys.version.replace("\n", " "))
print("Python executable:", sys.executable)
print("NumPy:", np.__version__)
print("SciPy:", scipy.__version__)
print("VTK:", vtk.vtkVersion.GetVTKVersion())
print("GUDHI:", gudhi.__version__)


# ----------------------------------------------------------------------
# Synthetic independent cross-checks
# ----------------------------------------------------------------------

print()
print("=" * 80)
print("SYNTHETIC GUDHI CROSS-CHECKS")
print("=" * 80)

E = np.empty((0, 2), dtype=np.float64)

A = np.array([
    [0.0, 2.0]
], dtype=np.float64)

B = np.array([
    [1.0, 3.0]
], dtype=np.float64)

A2 = np.array([
    [0.0, 2.0],
    [4.0, 6.0]
], dtype=np.float64)

FAR1 = np.array([
    [0.0, 4.0]
], dtype=np.float64)

FAR2 = np.array([
    [100.0, 104.0]
], dtype=np.float64)


synthetic_cases = [
    ("empty / empty", E, E),
    ("single / empty", A, E),
    ("shifted point", A, B),
    ("two points / empty", A2, E),
    ("diagonal wins", FAR1, FAR2),
    ("identical diagrams", A2, A2),
]


all_ok = True

for name, X, Y in synthetic_cases:

    result = compare_case(
        name,
        X,
        Y
    )

    all_ok = (
        all_ok
        and result["db_ok"]
        and result["w2_ok"]
    )


# ----------------------------------------------------------------------
# Real TTK persistence diagrams — CNN sample 0
# ----------------------------------------------------------------------

print()
print("=" * 80)
print("REAL-DATA CROSS-CHECK — CNN SAMPLE 0")
print("=" * 80)


gt_path = (
    ROOT
    / "ttk_runs_fixed"
    / "cnn"
    / "pd"
    / "cnn_GT_s0_speed_p160_x0_y0_pd_port_0.vtu"
)

sr_path = (
    ROOT
    / "ttk_runs_fixed"
    / "cnn"
    / "pd"
    / "cnn_SR_s0_speed_p160_x0_y0_pd_port_0.vtu"
)


print("GT:", gt_path)
print("SR:", sr_path)


GT, gt_global = canonical.read_pd(gt_path)
SR, sr_global = canonical.read_pd(sr_path)


results = {}


for dim in (0, 1):

    print()
    print(
        f"D{dim} cardinalities: "
        f"GT={len(GT[dim])}, "
        f"SR={len(SR[dim])}"
    )

    result = compare_case(
        f"CNN sample 0 — D{dim}",
        GT[dim],
        SR[dim]
    )

    results[dim] = result

    all_ok = (
        all_ok
        and result["db_ok"]
        and result["w2_ok"]
    )


# ----------------------------------------------------------------------
# Dimension-wise aggregation
#
# IMPORTANT:
# D0 and D1 must NOT be concatenated into one unlabeled GUDHI diagram,
# because that would permit cross-dimension matching.
#
# Instead, independently validated D0/D1 distances are combined using
# the definitions used by the canonical sweep.
# ----------------------------------------------------------------------

ours_db_all = max(
    results[0]["ours_db"],
    results[1]["ours_db"]
)

gudhi_db_all = max(
    results[0]["gudhi_db"],
    results[1]["gudhi_db"]
)

ours_w2_all = math.hypot(
    results[0]["ours_w2"],
    results[1]["ours_w2"]
)

gudhi_w2_all = math.hypot(
    results[0]["gudhi_w2"],
    results[1]["gudhi_w2"]
)


db_all_diff = abs(
    ours_db_all - gudhi_db_all
)

w2_all_diff = abs(
    ours_w2_all - gudhi_w2_all
)


db_all_ok = close(
    ours_db_all,
    gudhi_db_all
)

w2_all_ok = close(
    ours_w2_all,
    gudhi_w2_all
)


all_ok = (
    all_ok
    and db_all_ok
    and w2_all_ok
)


print()
print("=" * 80)
print("COMBINED FINITE-PD RESULT")
print("=" * 80)

print(
    f"dB_all: ours={ours_db_all:.17g}  "
    f"GUDHI-derived={gudhi_db_all:.17g}  "
    f"|diff|={db_all_diff:.17g}  "
    f"{'PASS' if db_all_ok else 'FAIL'}"
)

print(
    f"W2_all: ours={ours_w2_all:.17g}  "
    f"GUDHI-derived={gudhi_w2_all:.17g}  "
    f"|diff|={w2_all_diff:.17g}  "
    f"{'PASS' if w2_all_ok else 'FAIL'}"
)


# ----------------------------------------------------------------------
# Nonfinite/global pair
# ----------------------------------------------------------------------

print()
print("=" * 80)
print("NONFINITE GLOBAL PAIR — EXCLUDED FROM FINITE-PD DISTANCES")
print("=" * 80)

print("GT :", gt_global)
print("SR :", sr_global)

print(
    "These are intentionally kept separate and are NOT supplied "
    "to either finite-diagram distance implementation."
)


# ----------------------------------------------------------------------
# Final result
# ----------------------------------------------------------------------

print()
print("=" * 80)

if all_ok:
    print("OVERALL GUDHI CROSS-CHECK: PASS")
    print(
        "Custom dB and W2 agree with the independent "
        "GUDHI implementation within numerical tolerance."
    )
else:
    print("OVERALL GUDHI CROSS-CHECK: FAIL")
    print(
        "At least one custom/GUDHI comparison differs beyond "
        "the specified tolerance."
    )

print("=" * 80)


if not all_ok:
    sys.exit(1)
