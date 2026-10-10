#!/usr/bin/env python3

from pathlib import Path
import math
import os
import sys

import numpy as np
import gudhi

from scipy.optimize import linear_sum_assignment
from gudhi.wasserstein import wasserstein_distance


# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------

AUDIT = Path(
    os.environ.get(
        "AUDIT",
        str(
            Path.home()
            / "phire_runtime_audit_20260809_221548"
        ),
    )
).resolve()

ORIGINAL_AUDIT = AUDIT / "recompute_pd"

sys.path.insert(
    0,
    str(ORIGINAL_AUDIT)
)

# Reuse ONLY the already-audited TTK VTU parser.
# Do not modify the frozen canonical implementation.
import canonical_pd_pilot as canonical


ROOT = Path.home() / "PhIRE"

ABS_TOL = 1e-10
REL_TOL = 1e-12


# ----------------------------------------------------------------------
# Standard W_{2,2}
#
# Wasserstein exponent q = 2
# Birth/death ground norm p = 2
#
# Real-real:
#
#   ||(b1,d1)-(b2,d2)||_2
#
# Point-diagonal:
#
#   persistence / sqrt(2)
#
# ----------------------------------------------------------------------

def pairwise_l2(A, B):

    if len(A) == 0 or len(B) == 0:
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

    persistence = (
        D[:, 1]
        - D[:, 0]
    )

    return (
        persistence
        / math.sqrt(2.0)
    )


def augmented_cost_l2(A, B):

    n = len(A)
    m = len(B)

    if n + m == 0:
        return np.empty(
            (0, 0),
            dtype=np.float64,
        )

    C = np.full(
        (n + m, n + m),
        np.inf,
        dtype=np.float64,
    )

    # --------------------------------------------------------------
    # Real A -> real B
    # --------------------------------------------------------------

    if n and m:
        C[:n, :m] = pairwise_l2(
            A,
            B,
        )

    # --------------------------------------------------------------
    # Real A -> own diagonal copy
    # --------------------------------------------------------------

    if n:

        i = np.arange(n)

        C[
            i,
            m + i
        ] = diagonal_cost_l2(A)

    # --------------------------------------------------------------
    # Corresponding diagonal copy -> real B
    # --------------------------------------------------------------

    if m:

        j = np.arange(m)

        C[
            n + j,
            j
        ] = diagonal_cost_l2(B)

    # --------------------------------------------------------------
    # Diagonal copy -> diagonal copy
    # --------------------------------------------------------------

    if n and m:
        C[
            n:,
            m:
        ] = 0.0

    return C


def w22(A, B):

    C = augmented_cost_l2(
        A,
        B,
    )

    if C.size == 0:
        return 0.0

    rows, cols = (
        linear_sum_assignment(
            C * C
        )
    )

    costs = C[
        rows,
        cols
    ]

    return float(
        np.sqrt(
            np.sum(
                costs * costs
            )
        )
    )


# ----------------------------------------------------------------------
# Independent GUDHI W_{2,2}
# ----------------------------------------------------------------------

def gudhi_w22(A, B):

    return float(
        wasserstein_distance(
            A,
            B,
            matching=False,
            order=2.0,
            internal_p=2.0,
            keep_essential_parts=False,
        )
    )


# ----------------------------------------------------------------------
# Comparison helpers
# ----------------------------------------------------------------------

def close(a, b):

    return math.isclose(
        a,
        b,
        rel_tol=REL_TOL,
        abs_tol=ABS_TOL,
    )


def compare(
    name,
    A,
    B,
    expected=None,
):

    ours = w22(
        A,
        B,
    )

    g = gudhi_w22(
        A,
        B,
    )

    diff = abs(
        ours - g
    )

    ok = close(
        ours,
        g,
    )

    print()
    print(name)

    print(
        f"ours     = "
        f"{ours:.17g}"
    )

    print(
        f"GUDHI    = "
        f"{g:.17g}"
    )

    print(
        f"|diff|   = "
        f"{diff:.17g}"
    )

    print(
        "ours/GUDHI: "
        + (
            "PASS"
            if ok
            else "FAIL"
        )
    )

    if expected is not None:

        expected_ok = close(
            ours,
            expected,
        )

        print(
            f"expected = "
            f"{expected:.17g}"
        )

        print(
            "analytic: "
            + (
                "PASS"
                if expected_ok
                else "FAIL"
            )
        )

        ok = (
            ok
            and expected_ok
        )

    return ok


# ----------------------------------------------------------------------
# Synthetic analytical tests
# ----------------------------------------------------------------------

def synthetic_tests():

    print("=" * 80)
    print(
        "STANDARD W_{2,2} SYNTHETIC TESTS"
    )
    print("=" * 80)

    ok = True

    E = np.empty(
        (0, 2),
        dtype=np.float64,
    )

    A = np.array(
        [
            [0.0, 2.0],
        ],
        dtype=np.float64,
    )

    B = np.array(
        [
            [1.0, 3.0],
        ],
        dtype=np.float64,
    )

    A2 = np.array(
        [
            [0.0, 2.0],
            [4.0, 6.0],
        ],
        dtype=np.float64,
    )

    FAR1 = np.array(
        [
            [0.0, 4.0],
        ],
        dtype=np.float64,
    )

    FAR2 = np.array(
        [
            [100.0, 104.0],
        ],
        dtype=np.float64,
    )

    C = np.array(
        [
            [1.0, 2.0],
        ],
        dtype=np.float64,
    )

    # Empty / empty
    ok &= compare(
        "empty / empty",
        E,
        E,
        0.0,
    )

    # Point [0,2] -> diagonal:
    #
    # persistence / sqrt(2)
    # = 2 / sqrt(2)
    # = sqrt(2)
    ok &= compare(
        "single / empty",
        A,
        E,
        math.sqrt(2.0),
    )

    # [0,2] -> [1,3]
    #
    # Euclidean:
    # sqrt(1^2 + 1^2)
    # = sqrt(2)
    ok &= compare(
        "shifted point",
        A,
        B,
        math.sqrt(2.0),
    )

    # Two persistence-2 points -> diagonal:
    #
    # each cost sqrt(2)
    #
    # W2 = sqrt(2 + 2) = 2
    ok &= compare(
        "two points / empty",
        A2,
        E,
        2.0,
    )

    # Direct real-real matching is huge.
    #
    # Each persistence-4 point matches diagonal:
    #
    # cost = 4/sqrt(2) = 2*sqrt(2)
    #
    # W2 = sqrt(8 + 8) = 4
    ok &= compare(
        "diagonal wins",
        FAR1,
        FAR2,
        4.0,
    )

    # Identical
    ok &= compare(
        "identical diagrams",
        A2,
        A2,
        0.0,
    )

    # Extra test that differentiates L2 from L_inf.
    #
    # [0,2] -> [1,2]
    # Euclidean distance = 1.
    ok &= compare(
        "single-coordinate shift",
        A,
        C,
        1.0,
    )

    print()

    if not ok:
        raise AssertionError(
            "Synthetic W22 tests failed"
        )

    print(
        "ALL W22 SYNTHETIC TESTS: PASSED"
    )


# ----------------------------------------------------------------------
# Real CNN sample-0 pilot
# ----------------------------------------------------------------------

def real_test():

    gt_path = (
        ROOT
        / "ttk_runs_fixed"
        / "cnn"
        / "pd"
        / (
            "cnn_GT_s0_speed_"
            "p160_x0_y0_pd_port_0.vtu"
        )
    )

    sr_path = (
        ROOT
        / "ttk_runs_fixed"
        / "cnn"
        / "pd"
        / (
            "cnn_SR_s0_speed_"
            "p160_x0_y0_pd_port_0.vtu"
        )
    )

    GT, gt_global = (
        canonical.read_pd(
            gt_path
        )
    )

    SR, sr_global = (
        canonical.read_pd(
            sr_path
        )
    )

    print()
    print("=" * 80)
    print(
        "REAL W22 PILOT — CNN SAMPLE 0"
    )
    print("=" * 80)

    vals = {}

    overall_ok = True

    for dim in (0, 1):

        A = GT[dim]
        B = SR[dim]

        ours = w22(
            A,
            B,
        )

        g = gudhi_w22(
            A,
            B,
        )

        diff = abs(
            ours - g
        )

        ok = close(
            ours,
            g,
        )

        overall_ok &= ok

        vals[dim] = (
            ours,
            g,
        )

        print()
        print(
            f"D{dim} cardinalities: "
            f"GT={len(A)}, "
            f"SR={len(B)}"
        )

        print(
            f"W22 ours  = "
            f"{ours:.17g}"
        )

        print(
            f"W22 GUDHI = "
            f"{g:.17g}"
        )

        print(
            f"|diff|    = "
            f"{diff:.17g}"
        )

        print(
            "status     = "
            + (
                "PASS"
                if ok
                else "FAIL"
            )
        )

    ours_all = math.hypot(
        vals[0][0],
        vals[1][0],
    )

    gudhi_all = math.hypot(
        vals[0][1],
        vals[1][1],
    )

    diff_all = abs(
        ours_all
        - gudhi_all
    )

    all_ok = close(
        ours_all,
        gudhi_all,
    )

    overall_ok &= all_ok

    print()
    print(
        "COMBINED FINITE PD"
    )

    print(
        f"W22_all ours  = "
        f"{ours_all:.17g}"
    )

    print(
        f"W22_all GUDHI = "
        f"{gudhi_all:.17g}"
    )

    print(
        f"|diff|        = "
        f"{diff_all:.17g}"
    )

    print(
        "status         = "
        + (
            "PASS"
            if all_ok
            else "FAIL"
        )
    )

    print()
    print(
        "GLOBAL NONFINITE PAIR — EXCLUDED"
    )

    print(
        "GT:",
        gt_global
    )

    print(
        "SR:",
        sr_global
    )

    if not overall_ok:
        raise AssertionError(
            "Real W22 GUDHI cross-check failed"
        )

    print()
    print(
        "OVERALL STANDARD W22 GUDHI PILOT: PASS"
    )


def main():

    print(
        "Python:",
        sys.version.replace(
            "\n",
            " "
        )
    )

    print(
        "Executable:",
        sys.executable
    )

    print(
        "NumPy:",
        np.__version__
    )

    print(
        "GUDHI:",
        gudhi.__version__
    )

    print()
    print(
        "Definition:"
    )

    print(
        "  Wasserstein order q = 2"
    )

    print(
        "  ground norm p = 2 (Euclidean)"
    )

    print(
        "  diagonal cost = persistence/sqrt(2)"
    )

    print(
        "  D0 and D1 evaluated separately"
    )

    print(
        "  W22_all = hypot(W22_D0, W22_D1)"
    )

    synthetic_tests()
    real_test()


if __name__ == "__main__":
    main()
