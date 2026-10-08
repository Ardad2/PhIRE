#!/usr/bin/env python3
"""
Phase 5C pilot — sample-69 locality-constrained W2,2 persistence cost.

Uses canonical Phase-5 sample-69 PDs:
    GT
    pretrained CNN
    reconstruction-only
    topology-inspired F1

Primary objective:
    W2,2 (order=2, Euclidean birth/death-plane ground norm)

D0 and D1 are matched separately.

Spatial admissibility for a real-real pair:
    BOTH critical endpoints must be within radius r:

        max(
            ||birth_GT - birth_SR||_2,
            ||death_GT - death_SR||_2
        ) <= r

Diagonal matches remain allowed exactly as in ordinary persistence matching.

At r=infinity, the augmented assignment MUST reproduce GUDHI W2,2 before
finite-r results are accepted.

Exact zero-persistence pairs are excluded, consistent with prior Phase-5
spatial correspondence work.

Terminology:
    "locality-constrained persistence cost", not a proven mathematical metric.
"""

from pathlib import Path
import argparse
import csv
import json
import math
import hashlib

import numpy as np
import vtk
import gudhi
from gudhi.wasserstein import wasserstein_distance
from scipy.optimize import linear_sum_assignment


RADII = [4.0, 8.0, 16.0, 32.0, 64.0, 96.0, 128.0, math.inf]
TOL = 1e-10
BIG = 1e12


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def read_features(path):
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
    pid = cd.GetArray("PairIdentifier")

    coord = pd.GetArray("Coordinates")
    vid = pd.GetArray("ttkVertexScalarField")

    out = {0: [], 1: []}
    zero = {0: [], 1: []}

    for ci in range(g.GetNumberOfCells()):
        typ = int(round(ptype.GetTuple1(ci)))
        fin = int(round(finite.GetTuple1(ci)))
        p = float(pers.GetTuple1(ci))

        if fin != 1 or typ not in (0, 1):
            continue

        cell = g.GetCell(ci)
        p0 = int(cell.GetPointId(0))
        p1 = int(cell.GetPointId(1))

        b = float(birth.GetTuple1(ci))
        d = b + p

        c0 = coord.GetTuple(p0)
        c1 = coord.GetTuple(p1)

        f = {
            "index": None,
            "cell_index": ci,
            "pair_identifier": int(round(pid.GetTuple1(ci))),
            "birth": b,
            "death": d,
            "persistence": p,
            "birth_x": float(c0[0]),
            "birth_y": float(c0[1]),
            "death_x": float(c1[0]),
            "death_y": float(c1[1]),
            "birth_vertex_id": int(round(vid.GetTuple1(p0))),
            "death_vertex_id": int(round(vid.GetTuple1(p1))),
        }

        if abs(p) <= TOL:
            zero[typ].append(f)
        elif p > TOL:
            f["index"] = len(out[typ])
            out[typ].append(f)
        else:
            raise RuntimeError(f"{path}: negative persistence {p}")

    return out, zero


def points(fs):
    if not fs:
        return np.empty((0, 2), dtype=float)
    return np.asarray([[f["birth"], f["death"]] for f in fs], dtype=float)


def endpoint_displacements(a, b):
    bd = math.hypot(
        a["birth_x"] - b["birth_x"],
        a["birth_y"] - b["birth_y"],
    )
    dd = math.hypot(
        a["death_x"] - b["death_x"],
        a["death_y"] - b["death_y"],
    )
    return bd, dd


def real_cost_w22(a, b):
    return math.hypot(
        a["birth"] - b["birth"],
        a["death"] - b["death"],
    )


def diagonal_cost_w22(a):
    return (a["death"] - a["birth"]) / math.sqrt(2.0)


def gudhi_w22(gt, sr):
    return float(
        wasserstein_distance(
            points(gt),
            points(sr),
            matching=False,
            order=2.0,
            internal_p=2.0,
            keep_essential_parts=False,
        )
    )


def constrained_match(gt, sr, radius):
    n = len(gt)
    m = len(sr)
    N = n + m

    C = np.full((N, N), BIG, dtype=float)

    # real-real block
    for i, a in enumerate(gt):
        for j, b in enumerate(sr):
            bd, dd = endpoint_displacements(a, b)
            local = math.isinf(radius) or max(bd, dd) <= radius + 1e-12
            if local:
                c = real_cost_w22(a, b)
                C[i, j] = c * c

    # GT -> diagonal
    for i, a in enumerate(gt):
        c = diagonal_cost_w22(a)
        C[i, m + i] = c * c

    # diagonal -> SR
    for j, b in enumerate(sr):
        c = diagonal_cost_w22(b)
        C[n + j, j] = c * c

    # dummy-dummy completion
    C[n:, m:] = 0.0

    rr, cc = linear_sum_assignment(C)
    selected = C[rr, cc]

    if np.any(selected >= BIG / 2):
        raise RuntimeError("Assignment used a forbidden edge.")

    matches = []

    for r, c in zip(rr, cc):
        if r < n and c < m:
            bd, dd = endpoint_displacements(gt[r], sr[c])
            matches.append({
                "type": "real_real",
                "gt_index": int(r),
                "sr_index": int(c),
                "birth_displacement": bd,
                "death_displacement": dd,
                "max_endpoint_displacement": max(bd, dd),
                "gt_persistence": gt[r]["persistence"],
                "sr_persistence": sr[c]["persistence"],
                "ground_cost": real_cost_w22(gt[r], sr[c]),
            })
        elif r < n and c >= m:
            matches.append({
                "type": "gt_to_diagonal",
                "gt_index": int(r),
                "sr_index": -1,
                "gt_persistence": gt[r]["persistence"],
                "sr_persistence": None,
                "birth_displacement": None,
                "death_displacement": None,
                "max_endpoint_displacement": None,
                "ground_cost": diagonal_cost_w22(gt[r]),
            })
        elif r >= n and c < m:
            matches.append({
                "type": "diagonal_to_sr",
                "gt_index": -1,
                "sr_index": int(c),
                "gt_persistence": None,
                "sr_persistence": sr[c]["persistence"],
                "birth_displacement": None,
                "death_displacement": None,
                "max_endpoint_displacement": None,
                "ground_cost": diagonal_cost_w22(sr[c]),
            })

    return math.sqrt(float(selected.sum())), matches


def summarize(gt, matches):
    rr = [x for x in matches if x["type"] == "real_real"]
    gd = [x for x in matches if x["type"] == "gt_to_diagonal"]
    ds = [x for x in matches if x["type"] == "diagonal_to_sr"]

    gt_total_persistence = sum(f["persistence"] for f in gt)
    rr_gt_persistence = sum(
        gt[x["gt_index"]]["persistence"] for x in rr
    )

    max_disp = max(
        (x["max_endpoint_displacement"] for x in rr),
        default=None,
    )

    return {
        "real_real": len(rr),
        "gt_to_diagonal": len(gd),
        "diagonal_to_sr": len(ds),
        "gt_real_real_fraction":
            len(rr) / len(gt) if gt else None,
        "gt_persistence_weighted_real_real_coverage":
            rr_gt_persistence / gt_total_persistence
            if gt_total_persistence > 0 else None,
        "max_endpoint_displacement_real_real": max_disp,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--phase5",
        default=str(
            Path.home() / "PhIRE/spatial_pd/phase5a_sample69_canonical_pd"
        ),
    )
    ap.add_argument(
        "--out",
        default=str(
            Path.home() / "PhIRE/spatial_pd/phase5c_sample69_locality_w22"
        ),
    )
    args = ap.parse_args()

    phase5 = Path(args.phase5).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    pdroot = phase5 / "pd"

    paths = {
        "GT": pdroot / "phase5_GT_s69_speed_p160_x0_y0_pd_port_0.vtu",
        "CNN": pdroot / "phase5_CNN_SR_s69_speed_p160_x0_y0_pd_port_0.vtu",
        "UV": pdroot / "phase5_UV_SR_s69_speed_p160_x0_y0_pd_port_0.vtu",
        "F1": pdroot / "phase5_F1_SR_s69_speed_p160_x0_y0_pd_port_0.vtu",
    }

    features = {}
    zeros = {}

    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(path)
        features[name], zeros[name] = read_features(path)

    report = {
        "sample": 69,
        "gudhi_version": gudhi.__version__,
        "objective": "W2,2",
        "spatial_rule":
            "max(birth_endpoint_L2, death_endpoint_L2) <= r",
        "radii": ["inf" if math.isinf(r) else r for r in RADII],
        "files": {
            k: {"path": str(v), "sha256": sha256(v)}
            for k, v in paths.items()
        },
        "positive_counts": {
            k: {"D0": len(features[k][0]), "D1": len(features[k][1])}
            for k in paths
        },
        "zero_persistence_counts": {
            k: {"D0": len(zeros[k][0]), "D1": len(zeros[k][1])}
            for k in paths
        },
        "methods": {},
    }

    dim_rows = []
    aggregate_rows = []

    for method in ("CNN", "UV", "F1"):
        print()
        print("=" * 80)
        print(method)
        report["methods"][method] = {"dimensions": {}, "aggregate": {}}

        unconstrained_dim = {}

        for dim in (0, 1):
            gt = features["GT"][dim]
            sr = features[method][dim]

            gd = gudhi_w22(gt, sr)
            aug_inf, m_inf = constrained_match(gt, sr, math.inf)

            if not math.isclose(
                gd, aug_inf, rel_tol=1e-12, abs_tol=1e-10
            ):
                raise RuntimeError(
                    f"{method} D{dim}: augmented infinity {aug_inf} "
                    f"!= GUDHI {gd}"
                )

            unconstrained_dim[dim] = gd
            report["methods"][method]["dimensions"][f"D{dim}"] = {
                "gudhi_unconstrained": gd,
                "radii": {},
            }

            print()
            print(f"D{dim} GUDHI/unconstrained W22 = {gd:.15g}")
            print(
                "r      constrained      gap        RR  GTdiag  SRdiag  "
                "GT_RR_frac  pers_cov  max_disp"
            )

            for radius in RADII:
                d, matches = constrained_match(gt, sr, radius)
                s = summarize(gt, matches)
                gap = d - gd

                key = "inf" if math.isinf(radius) else str(radius)

                payload = {
                    "constrained_w22": d,
                    "locality_gap": gap,
                    **s,
                }
                report["methods"][method]["dimensions"][f"D{dim}"]["radii"][key] = payload

                dim_rows.append({
                    "method": method,
                    "dimension": dim,
                    "radius": key,
                    "unconstrained_w22": gd,
                    **payload,
                })

                rtxt = "inf" if math.isinf(radius) else f"{radius:g}"
                print(
                    f"{rtxt:>4s}  {d:14.8f}  {gap:+10.8f}  "
                    f"{s['real_real']:3d}  {s['gt_to_diagonal']:6d}  "
                    f"{s['diagonal_to_sr']:6d}  "
                    f"{s['gt_real_real_fraction']:.4f}  "
                    f"{s['gt_persistence_weighted_real_real_coverage']:.4f}  "
                    f"{s['max_endpoint_displacement_real_real']}"
                )

        base_all = math.hypot(
            unconstrained_dim[0],
            unconstrained_dim[1],
        )

        print()
        print("AGGREGATE D0+D1")
        print("r      constrained_all   locality_gap_all")

        for radius in RADII:
            key = "inf" if math.isinf(radius) else str(radius)
            d0 = report["methods"][method]["dimensions"]["D0"]["radii"][key]["constrained_w22"]
            d1 = report["methods"][method]["dimensions"]["D1"]["radii"][key]["constrained_w22"]
            all_d = math.hypot(d0, d1)
            gap = all_d - base_all

            report["methods"][method]["aggregate"][key] = {
                "unconstrained_w22_all": base_all,
                "constrained_w22_all": all_d,
                "locality_gap_all": gap,
            }

            aggregate_rows.append({
                "method": method,
                "radius": key,
                "unconstrained_w22_all": base_all,
                "constrained_w22_all": all_d,
                "locality_gap_all": gap,
            })

            rtxt = "inf" if math.isinf(radius) else f"{radius:g}"
            print(f"{rtxt:>4s}  {all_d:15.8f}  {gap:+15.8f}")

    with (out / "sample69_locality_w22_dimension.csv").open(
        "w", newline=""
    ) as f:
        w = csv.DictWriter(f, fieldnames=list(dim_rows[0].keys()))
        w.writeheader()
        w.writerows(dim_rows)

    with (out / "sample69_locality_w22_aggregate.csv").open(
        "w", newline=""
    ) as f:
        w = csv.DictWriter(f, fieldnames=list(aggregate_rows[0].keys()))
        w.writeheader()
        w.writerows(aggregate_rows)

    (out / "sample69_locality_w22_summary.json").write_text(
        json.dumps(report, indent=2)
    )

    print()
    print("SAMPLE-69 LOCALITY-CONSTRAINED W22 PILOT: PASS")
    print("Wrote:", out)


if __name__ == "__main__":
    main()
