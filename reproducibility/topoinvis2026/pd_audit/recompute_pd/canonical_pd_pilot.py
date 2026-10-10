#!/usr/bin/env python3

from pathlib import Path
from time import perf_counter
import math

import numpy as np
import vtk

from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching


ROOT = Path.home() / "PhIRE"


# ----------------------------------------------------------------------
# Read finite persistence pairs from audited TTK VTU
# ----------------------------------------------------------------------

def read_pd(path):

    r = vtk.vtkXMLUnstructuredGridReader()
    r.SetFileName(str(path))
    r.Update()

    g = r.GetOutput()

    if g is None:
        raise RuntimeError(f"Could not read {path}")

    cd = g.GetCellData()

    pid = cd.GetArray("PairIdentifier")
    ptype = cd.GetArray("PairType")
    birth = cd.GetArray("Birth")
    pers = cd.GetArray("Persistence")
    finite = cd.GetArray("IsFinite")

    for name, arr in [
        ("PairIdentifier", pid),
        ("PairType", ptype),
        ("Birth", birth),
        ("Persistence", pers),
        ("IsFinite", finite),
    ]:
        if arr is None:
            raise RuntimeError(
                f"{path}: missing {name}"
            )

    diagrams = {0: [], 1: []}
    nonfinite = []

    for i in range(g.GetNumberOfCells()):

        pair_id = int(pid.GetTuple1(i))

        # TTK's synthetic display diagonal
        if pair_id == -1:
            continue

        typ = int(ptype.GetTuple1(i))
        fin = int(finite.GetTuple1(i))

        b = float(birth.GetTuple1(i))
        p = float(pers.GetTuple1(i))
        d = b + p

        if p < 0:
            raise RuntimeError(
                f"{path}: negative persistence"
            )

        if fin == 1:

            if typ not in (0, 1):
                raise RuntimeError(
                    f"{path}: unexpected finite "
                    f"PairType={typ}"
                )

            diagrams[typ].append(
                (b, d)
            )

        else:

            nonfinite.append(
                (typ, b, d)
            )

    if len(nonfinite) != 1:
        raise RuntimeError(
            f"{path}: expected 1 real nonfinite "
            f"pair; got {len(nonfinite)}"
        )

    for k in (0, 1):
        diagrams[k] = np.asarray(
            diagrams[k],
            dtype=np.float64
        ).reshape(-1, 2)

    return diagrams, nonfinite[0]


# ----------------------------------------------------------------------
# Canonical persistence-diagram costs
# L_infinity ground metric
# ----------------------------------------------------------------------

def pairwise_linf(A, B):

    if len(A) == 0 or len(B) == 0:
        return np.empty(
            (len(A), len(B)),
            dtype=np.float64
        )

    return np.maximum(
        np.abs(
            A[:, None, 0]
            - B[None, :, 0]
        ),
        np.abs(
            A[:, None, 1]
            - B[None, :, 1]
        ),
    )


def diagonal_cost(D):

    if len(D) == 0:
        return np.empty(
            0,
            dtype=np.float64
        )

    return 0.5 * (
        D[:, 1] - D[:, 0]
    )


def augmented_cost(A, B):

    n = len(A)
    m = len(B)

    if n + m == 0:
        return np.empty(
            (0, 0),
            dtype=np.float64
        )

    C = np.full(
        (n + m, n + m),
        np.inf,
        dtype=np.float64
    )

    # Real A -> real B
    if n and m:
        C[:n, :m] = pairwise_linf(
            A, B
        )

    # A -> diagonal
    if n:
        i = np.arange(n)

        C[
            i,
            m + i
        ] = diagonal_cost(A)

    # diagonal -> B
    if m:
        j = np.arange(m)

        C[
            n + j,
            j
        ] = diagonal_cost(B)

    # diagonal copies -> diagonal copies
    if n and m:
        C[
            n:,
            m:
        ] = 0.0

    return C


# ----------------------------------------------------------------------
# Canonical order-2 Wasserstein
# ----------------------------------------------------------------------

def w2(A, B):

    C = augmented_cost(A, B)

    if C.size == 0:
        return 0.0

    rows, cols = linear_sum_assignment(
        C * C
    )

    costs = C[
        rows,
        cols
    ]

    if not np.all(
        np.isfinite(costs)
    ):
        raise RuntimeError(
            "W2 assignment used infinite edge"
        )

    return float(
        np.sqrt(
            np.sum(costs ** 2)
        )
    )


# ----------------------------------------------------------------------
# Canonical bottleneck
# ----------------------------------------------------------------------

def feasible_bottleneck(
    C,
    threshold
):

    allowed = (
        np.isfinite(C)
        & (C <= threshold)
    )

    graph = csr_matrix(
        allowed.astype(np.int8)
    )

    match = (
        maximum_bipartite_matching(
            graph,
            perm_type="column"
        )
    )

    return bool(
        np.all(match >= 0)
    )


def bottleneck(A, B):

    C = augmented_cost(A, B)

    if C.size == 0:
        return 0.0

    candidates = np.unique(
        C[
            np.isfinite(C)
        ]
    )

    lo = 0
    hi = len(candidates) - 1

    if not feasible_bottleneck(
        C,
        candidates[hi]
    ):
        raise RuntimeError(
            "No perfect matching "
            "at maximum threshold"
        )

    while lo < hi:

        mid = (
            lo + hi
        ) // 2

        if feasible_bottleneck(
            C,
            candidates[mid]
        ):
            hi = mid
        else:
            lo = mid + 1

    return float(
        candidates[lo]
    )


# ----------------------------------------------------------------------
# Analytical unit tests
# ----------------------------------------------------------------------

def check(
    name,
    actual,
    expected
):

    if not math.isclose(
        actual,
        expected,
        rel_tol=0.0,
        abs_tol=1e-10
    ):
        raise AssertionError(
            f"{name}: "
            f"got {actual}, "
            f"expected {expected}"
        )

    print(
        f"PASS {name}: "
        f"{actual:.15g}"
    )


def synthetic_tests():

    print("=" * 80)
    print("SYNTHETIC TESTS")
    print("=" * 80)

    E = np.empty(
        (0, 2),
        dtype=np.float64
    )

    check(
        "empty/empty dB",
        bottleneck(E, E),
        0.0
    )

    check(
        "empty/empty W2",
        w2(E, E),
        0.0
    )

    A = np.array([
        [0.0, 2.0]
    ])

    check(
        "single/empty dB",
        bottleneck(A, E),
        1.0
    )

    check(
        "single/empty W2",
        w2(A, E),
        1.0
    )

    B = np.array([
        [1.0, 3.0]
    ])

    check(
        "shifted dB",
        bottleneck(A, B),
        1.0
    )

    check(
        "shifted W2",
        w2(A, B),
        1.0
    )

    A2 = np.array([
        [0.0, 2.0],
        [4.0, 6.0]
    ])

    check(
        "two/empty dB",
        bottleneck(A2, E),
        1.0
    )

    check(
        "two/empty W2",
        w2(A2, E),
        math.sqrt(2.0)
    )

    FAR1 = np.array([
        [0.0, 4.0]
    ])

    FAR2 = np.array([
        [100.0, 104.0]
    ])

    check(
        "diagonal-wins dB",
        bottleneck(
            FAR1,
            FAR2
        ),
        2.0
    )

    check(
        "diagonal-wins W2",
        w2(
            FAR1,
            FAR2
        ),
        math.sqrt(8.0)
    )

    check(
        "identical dB",
        bottleneck(
            A2,
            A2
        ),
        0.0
    )

    check(
        "identical W2",
        w2(
            A2,
            A2
        ),
        0.0
    )

    print()
    print(
        "ALL SYNTHETIC TESTS: PASSED"
    )


# ----------------------------------------------------------------------
# CNN sample-0 real-data pilot
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

    GT, gt_global = read_pd(
        gt_path
    )

    SR, sr_global = read_pd(
        sr_path
    )

    print()
    print("=" * 80)
    print(
        "REAL PILOT — CNN SAMPLE 0"
    )
    print("=" * 80)

    results = {}

    total_start = perf_counter()

    for dim in (0, 1):

        print()
        print(
            f"D{dim}: "
            f"GT={len(GT[dim])}, "
            f"SR={len(SR[dim])}"
        )

        start = perf_counter()

        db = bottleneck(
            GT[dim],
            SR[dim]
        )

        db_time = (
            perf_counter()
            - start
        )

        start = perf_counter()

        wd = w2(
            GT[dim],
            SR[dim]
        )

        w_time = (
            perf_counter()
            - start
        )

        if db > wd + 1e-9:
            raise AssertionError(
                f"D{dim}: "
                f"dB={db} > W2={wd}"
            )

        results[dim] = (
            db,
            wd
        )

        print(
            f"dB = {db:.15g}"
        )

        print(
            f"W2 = {wd:.15g}"
        )

        print(
            "dB <= W2: PASS"
        )

        print(
            f"dB time = "
            f"{db_time:.3f} s"
        )

        print(
            f"W2 time = "
            f"{w_time:.3f} s"
        )

    db_all = max(
        results[0][0],
        results[1][0]
    )

    w2_all = math.hypot(
        results[0][1],
        results[1][1]
    )

    if (
        db_all
        > w2_all + 1e-9
    ):
        raise AssertionError(
            "combined dB > combined W2"
        )

    _, gt_min, gt_max = (
        gt_global
    )

    _, sr_min, sr_max = (
        sr_global
    )

    range_endpoint_linf = max(
        abs(gt_min - sr_min),
        abs(gt_max - sr_max)
    )

    print()
    print(
        "COMBINED FINITE PD"
    )

    print(
        f"dB_all = "
        f"{db_all:.15g}"
    )

    print(
        f"W2_all = "
        f"{w2_all:.15g}"
    )

    print(
        "dB_all <= W2_all: PASS"
    )

    print()
    print(
        "GLOBAL MIN/MAX — SEPARATE"
    )

    print(
        f"GT = "
        f"({gt_min:.15g}, "
        f"{gt_max:.15g})"
    )

    print(
        f"SR = "
        f"({sr_min:.15g}, "
        f"{sr_max:.15g})"
    )

    print(
        "endpoint L_inf "
        "discrepancy = "
        f"{range_endpoint_linf:.15g}"
    )

    print()
    print(
        "TOTAL REAL PILOT = "
        f"{perf_counter() - total_start:.3f} s"
    )


if __name__ == "__main__":

    synthetic_tests()
    real_test()
