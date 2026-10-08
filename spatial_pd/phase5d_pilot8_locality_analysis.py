#!/usr/bin/env python3
"""
Phase 5D Step 2 — predeclared 8-sample locality-constrained W2,2 pilot.

Samples:
    0, 24, 48, 69, 96, 120, 144, 167

Methods:
    CNN
    UV   (matched reconstruction-only control)
    F1   (topology-inspired)

For D0 and D1 separately:
    minimize ordinary W2,2 persistence matching cost
    subject to a real-real match satisfying

        max(
            Euclidean birth-endpoint displacement,
            Euclidean death-endpoint displacement
        ) <= r

Diagonal matching is unchanged.

This script reports:
- constrained W2,2 and locality gap;
- real-real / diagonal counts;
- GT-side and SR-side persistence-mass coverage;
- squared-cost decomposition;
- aggregate D0+D1 cost;
- macro summaries across the 8 fixed samples;
- pairwise sample counts for F1 vs CNN / UV.

At r=infinity, the optimized augmented assignment MUST reproduce GUDHI.

The implementation is vectorized relative to the original sample-69 pilot so
the fixed 8-sample screening run is practical.  A sample-69 numerical parity
gate against the previously validated pilot is also enforced.
"""

from pathlib import Path
import argparse
import csv
import hashlib
import json
import math

import numpy as np
import vtk
import gudhi
from gudhi.wasserstein import wasserstein_distance
from scipy.optimize import linear_sum_assignment


SAMPLES = [0, 24, 48, 69, 96, 120, 144, 167]
METHODS = ["CNN", "UV", "F1"]
DIMS = [0, 1]
RADII = [4.0, 8.0, 16.0, 32.0, 64.0, 96.0, 128.0, math.inf]
TOL = 1e-10
BIG = 1e18

S69_HASH = {
    "GT":  "3b4cb20b2d830ff2edc4b3c13c9f8e866a176824d52a27be9b7517ae392ae7d5",
    "CNN": "67c5a4f53834e2b4d401852cd7b730fcbeea695a03d0d853e781fa746693520c",
    "UV":  "3fd05fedee7317b4fa51bd8a838e2e99928dd3bc2f610a3d686c56d2b97f6f02",
    "F1":  "908dc080096767063ba88d40b17b03affd24126025b4d1b6f672d41d90ad73be",
}

S69_AGG_EXPECTED = {
    "CNN": {4.0: 31.05331012, math.inf: 18.72241898},
    "UV":  {4.0: 29.91023776, math.inf: 19.91258125},
    "F1":  {4.0: 32.25549755, math.inf: 12.32296250},
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def pd_path(root, label, sample):
    p = root / f"phase5_{label}_s{sample}_speed_p160_x0_y0_pd_port_0.vtu"
    if not p.exists():
        raise FileNotFoundError(p)
    return p


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
    coord = pd.GetArray("Coordinates")

    out = {0: [], 1: []}
    zero = {0: 0, 1: 0}

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

        if abs(p) <= TOL:
            zero[typ] += 1
            continue
        if p < -TOL:
            raise RuntimeError(f"{path}: negative persistence {p}")

        out[typ].append({
            "birth": b,
            "death": d,
            "persistence": p,
            "birth_x": float(c0[0]),
            "birth_y": float(c0[1]),
            "death_x": float(c1[0]),
            "death_y": float(c1[1]),
        })

    return out, zero


def points(fs):
    if not fs:
        return np.empty((0, 2), dtype=float)
    return np.asarray([[f["birth"], f["death"]] for f in fs], dtype=float)


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


class Prepared:
    def __init__(self, gt, sr):
        self.gt = gt
        self.sr = sr
        self.n = len(gt)
        self.m = len(sr)

        gb = np.asarray([x["birth"] for x in gt], dtype=float)
        gd = np.asarray([x["death"] for x in gt], dtype=float)
        sb = np.asarray([x["birth"] for x in sr], dtype=float)
        sd = np.asarray([x["death"] for x in sr], dtype=float)

        gbxy = np.asarray([[x["birth_x"], x["birth_y"]] for x in gt], dtype=float)
        gdxy = np.asarray([[x["death_x"], x["death_y"]] for x in gt], dtype=float)
        sbxy = np.asarray([[x["birth_x"], x["birth_y"]] for x in sr], dtype=float)
        sdxy = np.asarray([[x["death_x"], x["death_y"]] for x in sr], dtype=float)

        gp = np.asarray([x["persistence"] for x in gt], dtype=float)
        sp = np.asarray([x["persistence"] for x in sr], dtype=float)

        self.gt_p = gp
        self.sr_p = sp

        # squared Euclidean ground cost in (birth, death)
        self.real_sq = (
            (gb[:, None] - sb[None, :]) ** 2
            + (gd[:, None] - sd[None, :]) ** 2
        )

        # spatial admissibility: BOTH endpoints local
        bdisp = np.sqrt(
            ((gbxy[:, None, :] - sbxy[None, :, :]) ** 2).sum(axis=2)
        )
        ddisp = np.sqrt(
            ((gdxy[:, None, :] - sdxy[None, :, :]) ** 2).sum(axis=2)
        )
        self.max_disp = np.maximum(bdisp, ddisp)

        # Euclidean point-to-diagonal cost squared = persistence^2 / 2
        self.gt_diag_sq = gp * gp / 2.0
        self.sr_diag_sq = sp * sp / 2.0

        self.gt_total_p = float(gp.sum())
        self.sr_total_p = float(sp.sum())

    def solve(self, radius):
        n, m = self.n, self.m
        N = n + m
        C = np.full((N, N), BIG, dtype=float)

        if math.isinf(radius):
            C[:n, :m] = self.real_sq
        else:
            C[:n, :m] = np.where(
                self.max_disp <= radius + 1e-12,
                self.real_sq,
                BIG,
            )

        # GT -> own diagonal slot
        C[np.arange(n), m + np.arange(n)] = self.gt_diag_sq

        # diagonal slot -> SR
        C[n + np.arange(m), np.arange(m)] = self.sr_diag_sq

        # dummy-dummy completion
        C[n:, m:] = 0.0

        rr, cc = linear_sum_assignment(C)
        selected = C[rr, cc]

        if np.any(selected >= BIG / 2):
            raise RuntimeError("Assignment used forbidden BIG-cost edge.")

        total_sq = float(selected.sum())

        real_mask = (rr < n) & (cc < m)
        gt_diag_mask = (rr < n) & (cc >= m)
        sr_diag_mask = (rr >= n) & (cc < m)

        rr_i = rr[real_mask]
        rr_j = cc[real_mask]

        rr_sq = float(selected[real_mask].sum())
        gt_diag_sq = float(selected[gt_diag_mask].sum())
        sr_diag_sq = float(selected[sr_diag_mask].sum())

        gt_matched_p = float(self.gt_p[rr_i].sum()) if len(rr_i) else 0.0
        sr_matched_p = float(self.sr_p[rr_j].sum()) if len(rr_j) else 0.0

        max_disp = (
            float(self.max_disp[rr_i, rr_j].max())
            if len(rr_i) else None
        )

        return {
            "w22": math.sqrt(total_sq),
            "total_sq": total_sq,
            "rr_sq": rr_sq,
            "gt_diag_sq": gt_diag_sq,
            "sr_diag_sq": sr_diag_sq,
            "rr_sq_fraction": rr_sq / total_sq if total_sq else 0.0,
            "gt_diag_sq_fraction": gt_diag_sq / total_sq if total_sq else 0.0,
            "sr_diag_sq_fraction": sr_diag_sq / total_sq if total_sq else 0.0,
            "real_real": int(real_mask.sum()),
            "gt_to_diagonal": int(gt_diag_mask.sum()),
            "diagonal_to_sr": int(sr_diag_mask.sum()),
            "gt_persistence_coverage":
                gt_matched_p / self.gt_total_p if self.gt_total_p > 0 else None,
            "sr_persistence_coverage":
                sr_matched_p / self.sr_total_p if self.sr_total_p > 0 else None,
            "gt_total_persistence": self.gt_total_p,
            "sr_total_persistence": self.sr_total_p,
            "gt_matched_persistence": gt_matched_p,
            "sr_matched_persistence": sr_matched_p,
            "max_endpoint_displacement_real_real": max_disp,
        }


def mean(xs):
    xs = list(xs)
    return float(np.mean(xs)) if xs else None


def median(xs):
    xs = list(xs)
    return float(np.median(xs)) if xs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pd-root",
        default=str(
            Path.home()
            / "PhIRE/spatial_pd/phase5d_pilot8_canonical_pd/pd"
        ),
    )
    ap.add_argument(
        "--out",
        default=str(
            Path.home()
            / "PhIRE/spatial_pd/phase5d_pilot8_locality_analysis"
        ),
    )
    args = ap.parse_args()

    pdroot = Path(args.pd_root).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    labels = {
        "GT": "GT",
        "CNN": "CNN_SR",
        "UV": "UV_SR",
        "F1": "F1_SR",
    }

    # ---------- sample-69 hash continuity ----------
    print("===== SAMPLE-69 HASH CONTINUITY =====")
    for name in ("GT", "CNN", "UV", "F1"):
        p = pd_path(pdroot, labels[name], 69)
        got = sha256(p)
        exp = S69_HASH[name]
        print(name, got, "PASS" if got == exp else "FAIL")
        if got != exp:
            raise RuntimeError(
                f"sample69 {name}: hash mismatch {got} != {exp}"
            )

    detail_rows = []
    aggregate_rows = []
    sample_cache = {}
    report = {
        "samples": SAMPLES,
        "methods": METHODS,
        "radii": ["inf" if math.isinf(r) else r for r in RADII],
        "gudhi_version": gudhi.__version__,
        "results": {},
    }

    for s in SAMPLES:
        print()
        print("#" * 96)
        print("SAMPLE", s)

        gt_feat, gt_zero = read_features(pd_path(pdroot, "GT", s))
        sample_cache[s] = {}

        report["results"][str(s)] = {
            "zero_persistence_GT": gt_zero,
            "methods": {},
        }

        for method in METHODS:
            sr_feat, sr_zero = read_features(
                pd_path(pdroot, labels[method], s)
            )

            report["results"][str(s)]["methods"][method] = {
                "zero_persistence_SR": sr_zero,
                "dimensions": {},
                "aggregate": {},
            }

            dim_results = {}

            print()
            print(method)

            for dim in DIMS:
                gt = gt_feat[dim]
                sr = sr_feat[dim]
                prep = Prepared(gt, sr)

                gd = gudhi_w22(gt, sr)
                inf_result = prep.solve(math.inf)

                if not math.isclose(
                    gd, inf_result["w22"],
                    rel_tol=1e-12, abs_tol=1e-10
                ):
                    raise RuntimeError(
                        f"s{s} {method} D{dim}: "
                        f"augmented infinity {inf_result['w22']} "
                        f"!= GUDHI {gd}"
                    )

                print(
                    f"  D{dim}: GT+={len(gt)} SR+={len(sr)} "
                    f"W22_inf={gd:.8f}"
                )

                dim_results[dim] = {}
                report["results"][str(s)]["methods"][method][
                    "dimensions"
                ][f"D{dim}"] = {
                    "unconstrained_w22": gd,
                    "radii": {},
                }

                for radius in RADII:
                    res = prep.solve(radius)
                    res["locality_gap"] = res["w22"] - gd

                    key = "inf" if math.isinf(radius) else str(radius)
                    dim_results[dim][key] = res
                    report["results"][str(s)]["methods"][method][
                        "dimensions"
                    ][f"D{dim}"]["radii"][key] = res

                    detail_rows.append({
                        "sample": s,
                        "method": method,
                        "dimension": dim,
                        "radius": key,
                        "unconstrained_w22": gd,
                        **res,
                    })

            # aggregate D0 + D1
            for radius in RADII:
                key = "inf" if math.isinf(radius) else str(radius)
                d0 = dim_results[0][key]
                d1 = dim_results[1][key]

                total_sq = d0["total_sq"] + d1["total_sq"]
                rr_sq = d0["rr_sq"] + d1["rr_sq"]
                gd_sq = d0["gt_diag_sq"] + d1["gt_diag_sq"]
                sd_sq = d0["sr_diag_sq"] + d1["sr_diag_sq"]

                base_all = math.hypot(
                    report["results"][str(s)]["methods"][method][
                        "dimensions"
                    ]["D0"]["unconstrained_w22"],
                    report["results"][str(s)]["methods"][method][
                        "dimensions"
                    ]["D1"]["unconstrained_w22"],
                )
                w_all = math.sqrt(total_sq)

                gt_total = (
                    d0["gt_total_persistence"]
                    + d1["gt_total_persistence"]
                )
                sr_total = (
                    d0["sr_total_persistence"]
                    + d1["sr_total_persistence"]
                )
                gt_match = (
                    d0["gt_matched_persistence"]
                    + d1["gt_matched_persistence"]
                )
                sr_match = (
                    d0["sr_matched_persistence"]
                    + d1["sr_matched_persistence"]
                )

                agg = {
                    "unconstrained_w22_all": base_all,
                    "constrained_w22_all": w_all,
                    "locality_gap_all": w_all - base_all,
                    "total_sq": total_sq,
                    "rr_sq": rr_sq,
                    "gt_diag_sq": gd_sq,
                    "sr_diag_sq": sd_sq,
                    "rr_sq_fraction": rr_sq / total_sq if total_sq else 0.0,
                    "gt_diag_sq_fraction": gd_sq / total_sq if total_sq else 0.0,
                    "sr_diag_sq_fraction": sd_sq / total_sq if total_sq else 0.0,
                    "gt_persistence_coverage_all":
                        gt_match / gt_total if gt_total > 0 else None,
                    "sr_persistence_coverage_all":
                        sr_match / sr_total if sr_total > 0 else None,
                    "real_real_all":
                        d0["real_real"] + d1["real_real"],
                    "gt_to_diagonal_all":
                        d0["gt_to_diagonal"] + d1["gt_to_diagonal"],
                    "diagonal_to_sr_all":
                        d0["diagonal_to_sr"] + d1["diagonal_to_sr"],
                }

                report["results"][str(s)]["methods"][method][
                    "aggregate"
                ][key] = agg

                aggregate_rows.append({
                    "sample": s,
                    "method": method,
                    "radius": key,
                    **agg,
                })

        # ---------- sample69 numerical parity with previous pilot ----------
        if s == 69:
            print()
            print("===== SAMPLE-69 NUMERICAL PARITY =====")
            for method in METHODS:
                for radius, exp in S69_AGG_EXPECTED[method].items():
                    key = "inf" if math.isinf(radius) else str(radius)
                    got = report["results"]["69"]["methods"][method][
                        "aggregate"
                    ][key]["constrained_w22_all"]
                    ok = math.isclose(got, exp, rel_tol=0.0, abs_tol=5e-8)
                    print(
                        method, key,
                        f"got={got:.10f}",
                        f"expected={exp:.10f}",
                        "PASS" if ok else "FAIL"
                    )
                    if not ok:
                        raise RuntimeError(
                            f"s69 parity fail {method} r={key}: "
                            f"{got} vs {exp}"
                        )

    # ---------- macro summary across fixed samples ----------
    macro_rows = []
    pairwise_rows = []

    print()
    print("=" * 96)
    print("MACRO SUMMARY ACROSS 8 FIXED SAMPLES")
    print(
        "radius method  mean_cost median_cost mean_gap "
        "mean_GTcov mean_SRcov"
    )

    for radius in RADII:
        key = "inf" if math.isinf(radius) else str(radius)

        by_method = {}
        for method in METHODS:
            vals = [
                report["results"][str(s)]["methods"][method][
                    "aggregate"
                ][key]
                for s in SAMPLES
            ]
            row = {
                "radius": key,
                "method": method,
                "mean_constrained_w22_all":
                    mean(v["constrained_w22_all"] for v in vals),
                "median_constrained_w22_all":
                    median(v["constrained_w22_all"] for v in vals),
                "mean_locality_gap_all":
                    mean(v["locality_gap_all"] for v in vals),
                "median_locality_gap_all":
                    median(v["locality_gap_all"] for v in vals),
                "mean_gt_persistence_coverage_all":
                    mean(v["gt_persistence_coverage_all"] for v in vals),
                "median_gt_persistence_coverage_all":
                    median(v["gt_persistence_coverage_all"] for v in vals),
                "mean_sr_persistence_coverage_all":
                    mean(v["sr_persistence_coverage_all"] for v in vals),
                "median_sr_persistence_coverage_all":
                    median(v["sr_persistence_coverage_all"] for v in vals),
                "mean_gt_diag_sq_fraction":
                    mean(v["gt_diag_sq_fraction"] for v in vals),
                "mean_sr_diag_sq_fraction":
                    mean(v["sr_diag_sq_fraction"] for v in vals),
            }
            by_method[method] = vals
            macro_rows.append(row)

            print(
                f"{key:>5s} {method:>4s} "
                f"{row['mean_constrained_w22_all']:9.4f} "
                f"{row['median_constrained_w22_all']:10.4f} "
                f"{row['mean_locality_gap_all']:8.4f} "
                f"{row['mean_gt_persistence_coverage_all']:.4f} "
                f"{row['mean_sr_persistence_coverage_all']:.4f}"
            )

        f1_lt_cnn = 0
        f1_lt_uv = 0
        f1_gtcov_cnn = 0
        f1_gtcov_uv = 0
        f1_both_cnn = 0
        f1_both_uv = 0

        for idx, s in enumerate(SAMPLES):
            f = by_method["F1"][idx]
            c = by_method["CNN"][idx]
            u = by_method["UV"][idx]

            lc = f["constrained_w22_all"] < c["constrained_w22_all"]
            lu = f["constrained_w22_all"] < u["constrained_w22_all"]
            gc = (
                f["gt_persistence_coverage_all"]
                > c["gt_persistence_coverage_all"]
            )
            gu = (
                f["gt_persistence_coverage_all"]
                > u["gt_persistence_coverage_all"]
            )

            f1_lt_cnn += int(lc)
            f1_lt_uv += int(lu)
            f1_gtcov_cnn += int(gc)
            f1_gtcov_uv += int(gu)
            f1_both_cnn += int(lc and gc)
            f1_both_uv += int(lu and gu)

        pairwise = {
            "radius": key,
            "n_samples": len(SAMPLES),
            "F1_lower_cost_than_CNN": f1_lt_cnn,
            "F1_lower_cost_than_UV": f1_lt_uv,
            "F1_higher_GTcoverage_than_CNN": f1_gtcov_cnn,
            "F1_higher_GTcoverage_than_UV": f1_gtcov_uv,
            "F1_lower_cost_AND_higher_GTcoverage_than_CNN": f1_both_cnn,
            "F1_lower_cost_AND_higher_GTcoverage_than_UV": f1_both_uv,
        }
        pairwise_rows.append(pairwise)

    print()
    print("PAIRWISE SAMPLE COUNTS")
    print(
        "radius  F1cost<CNN F1cost<UV "
        "F1GTcov>CNN F1GTcov>UV both_vs_CNN both_vs_UV"
    )
    for r in pairwise_rows:
        print(
            f"{r['radius']:>5s} "
            f"{r['F1_lower_cost_than_CNN']:10d} "
            f"{r['F1_lower_cost_than_UV']:9d} "
            f"{r['F1_higher_GTcoverage_than_CNN']:11d} "
            f"{r['F1_higher_GTcoverage_than_UV']:10d} "
            f"{r['F1_lower_cost_AND_higher_GTcoverage_than_CNN']:11d} "
            f"{r['F1_lower_cost_AND_higher_GTcoverage_than_UV']:10d}"
        )

    # ---------- outputs ----------
    with (out / "pilot8_locality_dimension.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(detail_rows[0].keys()))
        w.writeheader()
        w.writerows(detail_rows)

    with (out / "pilot8_locality_aggregate_by_sample.csv").open(
        "w", newline=""
    ) as f:
        w = csv.DictWriter(f, fieldnames=list(aggregate_rows[0].keys()))
        w.writeheader()
        w.writerows(aggregate_rows)

    with (out / "pilot8_locality_macro_summary.csv").open(
        "w", newline=""
    ) as f:
        w = csv.DictWriter(f, fieldnames=list(macro_rows[0].keys()))
        w.writeheader()
        w.writerows(macro_rows)

    with (out / "pilot8_locality_pairwise_counts.csv").open(
        "w", newline=""
    ) as f:
        w = csv.DictWriter(f, fieldnames=list(pairwise_rows[0].keys()))
        w.writeheader()
        w.writerows(pairwise_rows)

    (out / "pilot8_locality_summary.json").write_text(
        json.dumps(report, indent=2)
    )

    print()
    print("PHASE 5D 8-SAMPLE LOCALITY ANALYSIS: PASS")
    print("Wrote:", out)


if __name__ == "__main__":
    main()
