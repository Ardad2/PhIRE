#!/usr/bin/env python3
"""
Phase 5B3 — locality-constrained persistence benchmark.

This benchmark uses the already-generated Phase-5B2 GT/SR persistence diagrams.

Definition
----------
For each spatial radius r:

    minimize the ordinary order-2 PD matching objective

subject to:

    a real-real D0 match is allowed only if the birth/minimum coordinate
    displacement <= r

Diagonal matches remain allowed exactly as in the ordinary PD metric.

Two persistence ground norms are evaluated:
    W2,2        internal p = 2
    W2,infinity internal p = infinity

The implementation uses the standard augmented assignment construction with
SciPy linear_sum_assignment.  The r=infinity result is required to reproduce
GUDHI before any constrained result is accepted.

This is a controlled methodological benchmark, not a wind-field result.
"""

from pathlib import Path
import argparse
import csv
import json
import math

import numpy as np
import vtk
import gudhi
from gudhi.wasserstein import wasserstein_distance
from scipy.optimize import linear_sum_assignment


DELTAS = [0.00, 0.10, 0.19, 0.20, 0.21, 0.30, 0.40]
RADII = [0.0, 1.0, 2.0, 3.0, 3.999, 4.0, 5.0, 8.0, 16.0,
         32.0, 64.0, 69.0, 72.0, 76.0, 100.0, math.inf]
TOL = 1e-10
BIG = 1e12

GT_EXPECTED = {
    "A": (50.0, 50.0),
    "B": (110.0, 90.0),
    "C": (90.0, 120.0),
}
SR_EXPECTED = {
    "A": (54.0, 50.0),
    "B": (114.0, 90.0),
    "C": (94.0, 120.0),
}


def read_d0(path):
    r = vtk.vtkXMLUnstructuredGridReader()
    r.SetFileName(str(path))
    r.Update()
    g = r.GetOutput()

    cd = g.GetCellData()
    pd = g.GetPointData()

    ptype = cd.GetArray("PairType")
    pers = cd.GetArray("Persistence")
    birth = cd.GetArray("Birth")
    finite = cd.GetArray("IsFinite")
    coord = pd.GetArray("Coordinates")

    out = []
    for ci in range(g.GetNumberOfCells()):
        typ = int(round(ptype.GetTuple1(ci)))
        fin = int(round(finite.GetTuple1(ci)))
        p = float(pers.GetTuple1(ci))

        if typ != 0 or fin != 1 or p <= TOL:
            continue

        c = g.GetCell(ci)
        p0 = int(c.GetPointId(0))
        b = float(birth.GetTuple1(ci))
        d = b + p
        xy = coord.GetTuple(p0)

        out.append({
            "index": len(out),
            "birth": b,
            "death": d,
            "persistence": p,
            "x": float(xy[0]),
            "y": float(xy[1]),
        })
    return out


def pts(fs):
    return np.asarray([[f["birth"], f["death"]] for f in fs], dtype=float)


def spatial(a, b):
    return math.hypot(a["x"] - b["x"], a["y"] - b["y"])


def real_cost(a, b, p):
    db = abs(a["birth"] - b["birth"])
    dd = abs(a["death"] - b["death"])

    if math.isinf(p):
        return max(db, dd)

    return math.hypot(db, dd)


def diag_cost(a, p):
    persistence = a["death"] - a["birth"]

    if math.isinf(p):
        return persistence / 2.0

    # Euclidean distance to the diagonal.
    return persistence / math.sqrt(2.0)


def augmented_match(gt, sr, p, radius):
    """
    Standard augmented assignment for q=2 Wasserstein.

    Rows:
        n GT real points
        m dummy rows for SR->diagonal

    Columns:
        m SR real points
        n dummy columns for GT->diagonal
    """
    n = len(gt)
    m = len(sr)
    N = n + m

    C = np.full((N, N), BIG, dtype=float)

    # real-real
    for i, a in enumerate(gt):
        for j, b in enumerate(sr):
            if math.isinf(radius) or spatial(a, b) <= radius + 1e-12:
                c = real_cost(a, b, p)
                C[i, j] = c * c

    # GT -> diagonal; each real GT gets its own diagonal column
    for i, a in enumerate(gt):
        c = diag_cost(a, p)
        C[i, m + i] = c * c

    # diagonal -> SR; each SR gets its own dummy row
    for j, b in enumerate(sr):
        c = diag_cost(b, p)
        C[n + j, j] = c * c

    # dummy <-> dummy completion
    C[n:, m:] = 0.0

    rr, cc = linear_sum_assignment(C)
    total = float(C[rr, cc].sum())

    if np.any(C[rr, cc] >= BIG / 2):
        raise RuntimeError("Assignment required a forbidden BIG-cost edge.")

    matches = []
    for r, c in zip(rr, cc):
        if r < n and c < m:
            matches.append({
                "type": "real_real",
                "gt_index": int(r),
                "sr_index": int(c),
                "spatial": spatial(gt[r], sr[c]),
                "ground_cost": real_cost(gt[r], sr[c], p),
            })
        elif r < n and c >= m:
            matches.append({
                "type": "gt_to_diagonal",
                "gt_index": int(r),
                "sr_index": -1,
                "spatial": None,
                "ground_cost": diag_cost(gt[r], p),
            })
        elif r >= n and c < m:
            matches.append({
                "type": "diagonal_to_sr",
                "gt_index": -1,
                "sr_index": int(c),
                "spatial": None,
                "ground_cost": diag_cost(sr[c], p),
            })
        # dummy-dummy rows are intentionally omitted

    return math.sqrt(total), matches


def label_features(fs, expected):
    remaining = set(range(len(fs)))
    result = {}

    for label, (x, y) in expected.items():
        j = min(
            remaining,
            key=lambda k: math.hypot(fs[k]["x"] - x, fs[k]["y"] - y),
        )
        result[label] = j
        remaining.remove(j)

    return result


def mapping(matches):
    return {
        r["gt_index"]: r["sr_index"]
        for r in matches
        if r["gt_index"] >= 0
    }


def gudhi_distance(gt, sr, p):
    return float(
        wasserstein_distance(
            pts(gt),
            pts(sr),
            matching=False,
            order=2.0,
            internal_p=p,
            keep_essential_parts=False,
        )
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        default=str(Path.home() / "PhIRE/spatial_pd/phase5b2_near_degenerate"),
    )
    ap.add_argument(
        "--out",
        default=str(Path.home() / "PhIRE/spatial_pd/phase5b3_locality_constrained"),
    )
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    pdroot = root / "pd"
    gt = read_d0(pdroot / "GT_pd_port_0.vtu")
    gt_labels = label_features(gt, GT_EXPECTED)

    rows = []
    report = {
        "gudhi_version": gudhi.__version__,
        "radii": ["inf" if math.isinf(r) else r for r in RADII],
        "deltas": DELTAS,
        "results": [],
    }

    for delta in DELTAS:
        tag = f"{delta:.2f}".replace(".", "p")
        sr = read_d0(pdroot / f"SR_delta_{tag}_pd_port_0.vtu")
        sr_labels = label_features(sr, SR_EXPECTED)

        print()
        print("=" * 72)
        print("delta =", delta)

        for norm_name, p in (("W22", 2.0), ("W2inf", math.inf)):
            gd = gudhi_distance(gt, sr, p)
            unconstrained, m_inf = augmented_match(gt, sr, p, math.inf)

            if not math.isclose(
                gd, unconstrained, rel_tol=1e-12, abs_tol=1e-10
            ):
                raise RuntimeError(
                    f"{delta} {norm_name}: augmented unconstrained "
                    f"{unconstrained} != GUDHI {gd}"
                )

            print()
            print(norm_name)
            print("  GUDHI / unconstrained =", gd)
            print("  r        constrained    locality_gap   A_correct B_correct RR")

            for radius in RADII:
                d, matches = augmented_match(gt, sr, p, radius)
                mp = mapping(matches)

                a_ok = mp.get(gt_labels["A"], -1) == sr_labels["A"]
                b_ok = mp.get(gt_labels["B"], -1) == sr_labels["B"]
                rr_count = sum(x["type"] == "real_real" for x in matches)

                row = {
                    "delta": delta,
                    "norm": norm_name,
                    "radius": "inf" if math.isinf(radius) else radius,
                    "unconstrained_distance": gd,
                    "constrained_distance": d,
                    "locality_gap": d - gd,
                    "A_correct": a_ok,
                    "B_correct": b_ok,
                    "real_real": rr_count,
                }
                rows.append(row)
                report["results"].append(row)

                rtxt = "inf" if math.isinf(radius) else f"{radius:g}"
                print(
                    f"  {rtxt:>7s}  {d:13.9f}  {d-gd:+13.9f}  "
                    f"{str(a_ok):>8s} {str(b_ok):>8s} {rr_count:2d}"
                )

    with (out / "locality_constrained_summary.csv").open(
        "w", newline=""
    ) as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    (out / "locality_constrained_summary.json").write_text(
        json.dumps(report, indent=2)
    )

    print()
    print("LOCALITY-CONSTRAINED BENCHMARK: PASS")
    print("Wrote:", out)


if __name__ == "__main__":
    main()
