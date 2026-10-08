#!/usr/bin/env python3
"""
Phase 5E — sparse locality-constrained W2,2 solver.

Why this solver
---------------
For diagrams G and S, start from the valid assignment in which every point is
matched to the diagonal:

    B = sum_i d_G(i,Delta)^2 + sum_j d_S(j,Delta)^2

If GT point i and SR point j are matched real-real instead, the squared-cost
saving is:

    saving_ij
      = d_G(i,Delta)^2 + d_S(j,Delta)^2 - d_real(i,j)^2

Therefore the W2,2 optimum is the all-diagonal baseline minus the maximum total
saving over a one-to-one set of admissible real-real edges.

Edges with saving <= 0 never need to be selected.  Spatial locality restricts
which positive-savings edges are admissible.

This is equivalent to the dense augmented assignment used in Phase 5C/D, but
it can be solved as a sparse rectangular bipartite matching:
- one row per GT point;
- one real column per SR point;
- one private dummy column per GT point (unmatched GT);
- unmatched SR columns simply remain unused.

Modes
-----
validate-pilot:
    Reproduce the frozen Phase-5D 8-sample dense-solver CSV exactly/numerically.

all168:
    Run the same metric on the full canonical 168-sample PD layer.
"""

from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import time

import numpy as np
import vtk
import gudhi
from gudhi.wasserstein import wasserstein_distance
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching


PILOT_SAMPLES = [0, 24, 48, 69, 96, 120, 144, 167]
ALL_SAMPLES = list(range(168))
METHODS = ["CNN", "UV", "F1"]
DIMS = [0, 1]
RADII = [4.0, 8.0, 16.0, 32.0, 64.0, 96.0, 128.0, math.inf]

TOL = 1e-10
DUMMY_EDGE_WEIGHT = 1e-12


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

        if abs(p) <= TOL:
            zero[typ] += 1
            continue
        if p < -TOL:
            raise RuntimeError(f"{path}: negative persistence {p}")

        cell = g.GetCell(ci)
        p0 = int(cell.GetPointId(0))
        p1 = int(cell.GetPointId(1))
        b = float(birth.GetTuple1(ci))
        d = b + p
        c0 = coord.GetTuple(p0)
        c1 = coord.GetTuple(p1)

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
    return np.asarray([[x["birth"], x["death"]] for x in fs], dtype=float)


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


class SparsePrepared:
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

        self.gt_p = np.asarray([x["persistence"] for x in gt], dtype=float)
        self.sr_p = np.asarray([x["persistence"] for x in sr], dtype=float)

        self.real_sq = (
            (gb[:, None] - sb[None, :]) ** 2
            + (gd[:, None] - sd[None, :]) ** 2
        )

        bdisp = np.sqrt(
            ((gbxy[:, None, :] - sbxy[None, :, :]) ** 2).sum(axis=2)
        )
        ddisp = np.sqrt(
            ((gdxy[:, None, :] - sdxy[None, :, :]) ** 2).sum(axis=2)
        )
        self.max_disp = np.maximum(bdisp, ddisp)

        self.gt_diag_sq = self.gt_p * self.gt_p / 2.0
        self.sr_diag_sq = self.sr_p * self.sr_p / 2.0

        self.baseline_sq = float(
            self.gt_diag_sq.sum() + self.sr_diag_sq.sum()
        )

        self.savings = (
            self.gt_diag_sq[:, None]
            + self.sr_diag_sq[None, :]
            - self.real_sq
        )

        self.gt_total_p = float(self.gt_p.sum())
        self.sr_total_p = float(self.sr_p.sum())

    def solve(self, radius):
        n, m = self.n, self.m

        if n == 0:
            total_sq = float(self.sr_diag_sq.sum())
            return {
                "w22": math.sqrt(total_sq),
                "total_sq": total_sq,
                "rr_sq": 0.0,
                "gt_diag_sq": 0.0,
                "sr_diag_sq": total_sq,
                "real_real": 0,
                "gt_to_diagonal": 0,
                "diagonal_to_sr": m,
                "gt_persistence_coverage": None,
                "sr_persistence_coverage": 0.0 if self.sr_total_p > 0 else None,
                "gt_total_persistence": 0.0,
                "sr_total_persistence": self.sr_total_p,
                "gt_matched_persistence": 0.0,
                "sr_matched_persistence": 0.0,
                "max_endpoint_displacement_real_real": None,
                "candidate_real_edges": 0,
            }

        positive = self.savings > TOL
        if math.isinf(radius):
            admissible = positive
        else:
            admissible = positive & (self.max_disp <= radius + 1e-12)

        ri, cj = np.nonzero(admissible)

        # Sparse rectangular assignment:
        # rows = GT features
        # columns [0:m) = real SR features
        # columns [m:m+n) = one private GT dummy each
        rows = np.concatenate([ri, np.arange(n, dtype=int)])
        cols = np.concatenate([cj, m + np.arange(n, dtype=int)])

        # Real edges have negative cost = -saving.
        # Private dummies use a tiny positive stored weight because explicit
        # zero entries are dropped by scipy sparse matching.
        data = np.concatenate([
            -self.savings[ri, cj],
            np.full(n, DUMMY_EDGE_WEIGHT, dtype=float),
        ])

        graph = coo_matrix(
            (data, (rows, cols)),
            shape=(n, m + n),
        ).tocsr()

        row_ind, col_ind = min_weight_full_bipartite_matching(graph)

        # min_weight_full_bipartite_matching returns one selected column for
        # every GT row because n <= m+n and every row has a private dummy.
        real_mask = col_ind < m
        rr_i = row_ind[real_mask]
        rr_j = col_ind[real_mask]

        # Every selected real edge was stored only if it was admissible and had
        # positive savings.
        rr_sq = float(self.real_sq[rr_i, rr_j].sum()) if len(rr_i) else 0.0

        matched_gt = np.zeros(n, dtype=bool)
        matched_gt[rr_i] = True
        matched_sr = np.zeros(m, dtype=bool)
        matched_sr[rr_j] = True

        gt_diag_sq = float(self.gt_diag_sq[~matched_gt].sum())
        sr_diag_sq = float(self.sr_diag_sq[~matched_sr].sum())
        total_sq = rr_sq + gt_diag_sq + sr_diag_sq

        # Independent savings identity check.
        selected_savings = (
            float(self.savings[rr_i, rr_j].sum()) if len(rr_i) else 0.0
        )
        total_via_savings = self.baseline_sq - selected_savings
        if not math.isclose(
            total_sq, total_via_savings, rel_tol=1e-11, abs_tol=1e-9
        ):
            raise RuntimeError(
                f"savings identity mismatch {total_sq} vs {total_via_savings}"
            )

        gt_match_p = float(self.gt_p[rr_i].sum()) if len(rr_i) else 0.0
        sr_match_p = float(self.sr_p[rr_j].sum()) if len(rr_j) else 0.0

        max_disp = (
            float(self.max_disp[rr_i, rr_j].max()) if len(rr_i) else None
        )

        return {
            "w22": math.sqrt(max(total_sq, 0.0)),
            "total_sq": total_sq,
            "rr_sq": rr_sq,
            "gt_diag_sq": gt_diag_sq,
            "sr_diag_sq": sr_diag_sq,
            "rr_sq_fraction": rr_sq / total_sq if total_sq else 0.0,
            "gt_diag_sq_fraction": gt_diag_sq / total_sq if total_sq else 0.0,
            "sr_diag_sq_fraction": sr_diag_sq / total_sq if total_sq else 0.0,
            "real_real": int(len(rr_i)),
            "gt_to_diagonal": int(n - len(rr_i)),
            "diagonal_to_sr": int(m - len(rr_j)),
            "gt_persistence_coverage":
                gt_match_p / self.gt_total_p if self.gt_total_p > 0 else None,
            "sr_persistence_coverage":
                sr_match_p / self.sr_total_p if self.sr_total_p > 0 else None,
            "gt_total_persistence": self.gt_total_p,
            "sr_total_persistence": self.sr_total_p,
            "gt_matched_persistence": gt_match_p,
            "sr_matched_persistence": sr_match_p,
            "max_endpoint_displacement_real_real": max_disp,
            "candidate_real_edges": int(len(ri)),
        }


def load_dense_reference(path):
    rows = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            key = (
                int(r["sample"]),
                r["method"],
                int(r["dimension"]),
                r["radius"],
            )
            rows[key] = r
    return rows


def validate_pilot(args):
    pdroot = Path(args.pd_root).expanduser().resolve()
    ref = Path(args.reference_csv).expanduser().resolve()

    dense = load_dense_reference(ref)
    tested = 0
    max_w22 = 0.0
    max_cov = 0.0
    max_sq = 0.0
    start = time.perf_counter()

    print("===== SPARSE SOLVER PILOT VALIDATION =====")
    print("PD root :", pdroot)
    print("reference:", ref)
    print()

    for s in PILOT_SAMPLES:
        gt_feat, _ = read_features(pd_path(pdroot, "GT", s))

        for method, label in (
            ("CNN", "CNN_SR"),
            ("UV", "UV_SR"),
            ("F1", "F1_SR"),
        ):
            sr_feat, _ = read_features(pd_path(pdroot, label, s))

            for dim in DIMS:
                prep = SparsePrepared(gt_feat[dim], sr_feat[dim])

                # Independent infinity validation against GUDHI.
                gd = gudhi_w22(gt_feat[dim], sr_feat[dim])
                inf_res = prep.solve(math.inf)
                if not math.isclose(
                    gd, inf_res["w22"], rel_tol=1e-11, abs_tol=1e-9
                ):
                    raise RuntimeError(
                        f"GUDHI mismatch s{s} {method} D{dim}: "
                        f"{inf_res['w22']} vs {gd}"
                    )

                for radius in RADII:
                    keyr = "inf" if math.isinf(radius) else str(radius)
                    got = prep.solve(radius)
                    key = (s, method, dim, keyr)
                    if key not in dense:
                        raise KeyError(key)
                    exp = dense[key]

                    def ef(name):
                        return float(exp[name])

                    dw = abs(got["w22"] - ef("w22"))
                    dsq = abs(got["total_sq"] - ef("total_sq"))
                    dgc = abs(
                        got["gt_persistence_coverage"]
                        - ef("gt_persistence_coverage")
                    )
                    dsc = abs(
                        got["sr_persistence_coverage"]
                        - ef("sr_persistence_coverage")
                    )

                    max_w22 = max(max_w22, dw)
                    max_sq = max(max_sq, dsq)
                    max_cov = max(max_cov, dgc, dsc)

                    if dw > 1e-8 or dsq > 1e-6 or dgc > 1e-10 or dsc > 1e-10:
                        raise RuntimeError(
                            f"numeric mismatch {key}: "
                            f"dw={dw} dsq={dsq} dgc={dgc} dsc={dsc}"
                        )

                    for field in (
                        "real_real",
                        "gt_to_diagonal",
                        "diagonal_to_sr",
                    ):
                        if int(got[field]) != int(exp[field]):
                            raise RuntimeError(
                                f"count mismatch {key} {field}: "
                                f"{got[field]} vs {exp[field]}"
                            )

                    tested += 1

        print(f"sample {s}: PASS")

    elapsed = time.perf_counter() - start
    print()
    print("comparison rows:", tested)
    print("max |W22 sparse-dense|:", max_w22)
    print("max |total_sq sparse-dense|:", max_sq)
    print("max |coverage sparse-dense|:", max_cov)
    print("elapsed_seconds:", elapsed)
    print("SPARSE SOLVER PILOT VALIDATION: PASS")


def mean(xs):
    return float(np.mean(list(xs)))


def median(xs):
    return float(np.median(list(xs)))


def analyze_all168(args):
    pdroot = Path(args.pd_root).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    labels = {"CNN": "CNN_SR", "UV": "UV_SR", "F1": "F1_SR"}

    dim_rows = []
    agg_rows = []
    report = {
        "samples": ALL_SAMPLES,
        "methods": METHODS,
        "radii": ["inf" if math.isinf(r) else r for r in RADII],
        "gudhi_version": gudhi.__version__,
    }

    start_all = time.perf_counter()

    for si, s in enumerate(ALL_SAMPLES, 1):
        t0 = time.perf_counter()
        gt_feat, _ = read_features(pd_path(pdroot, "GT", s))

        for method in METHODS:
            sr_feat, _ = read_features(pd_path(pdroot, labels[method], s))
            per_dim = {}

            for dim in DIMS:
                gt = gt_feat[dim]
                sr = sr_feat[dim]
                prep = SparsePrepared(gt, sr)

                gd = gudhi_w22(gt, sr)
                inf_res = prep.solve(math.inf)
                if not math.isclose(
                    gd, inf_res["w22"], rel_tol=1e-11, abs_tol=1e-9
                ):
                    raise RuntimeError(
                        f"GUDHI mismatch s{s} {method} D{dim}: "
                        f"{inf_res['w22']} vs {gd}"
                    )

                per_dim[dim] = {}

                for radius in RADII:
                    keyr = "inf" if math.isinf(radius) else str(radius)
                    res = prep.solve(radius)
                    res["locality_gap"] = res["w22"] - gd
                    per_dim[dim][keyr] = res

                    dim_rows.append({
                        "sample": s,
                        "method": method,
                        "dimension": dim,
                        "radius": keyr,
                        "unconstrained_w22": gd,
                        **res,
                    })

            for radius in RADII:
                keyr = "inf" if math.isinf(radius) else str(radius)
                a = per_dim[0][keyr]
                b = per_dim[1][keyr]

                total_sq = a["total_sq"] + b["total_sq"]
                rr_sq = a["rr_sq"] + b["rr_sq"]
                gd_sq = a["gt_diag_sq"] + b["gt_diag_sq"]
                sd_sq = a["sr_diag_sq"] + b["sr_diag_sq"]

                w_all = math.sqrt(total_sq)
                base_all = math.hypot(
                    next(
                        r["unconstrained_w22"]
                        for r in reversed(dim_rows)
                        if r["sample"] == s
                        and r["method"] == method
                        and r["dimension"] == 0
                    ),
                    next(
                        r["unconstrained_w22"]
                        for r in reversed(dim_rows)
                        if r["sample"] == s
                        and r["method"] == method
                        and r["dimension"] == 1
                    ),
                )

                gt_total = a["gt_total_persistence"] + b["gt_total_persistence"]
                sr_total = a["sr_total_persistence"] + b["sr_total_persistence"]
                gt_match = a["gt_matched_persistence"] + b["gt_matched_persistence"]
                sr_match = a["sr_matched_persistence"] + b["sr_matched_persistence"]

                agg_rows.append({
                    "sample": s,
                    "method": method,
                    "radius": keyr,
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
                    "real_real_all": a["real_real"] + b["real_real"],
                    "gt_to_diagonal_all":
                        a["gt_to_diagonal"] + b["gt_to_diagonal"],
                    "diagonal_to_sr_all":
                        a["diagonal_to_sr"] + b["diagonal_to_sr"],
                })

        elapsed = time.perf_counter() - t0
        if si % 8 == 0 or si == 1 or si == len(ALL_SAMPLES):
            print(
                f"[{si:3d}/168] sample={s:3d} "
                f"sample_seconds={elapsed:.2f} "
                f"elapsed_minutes={(time.perf_counter()-start_all)/60:.1f}"
            )

    # Summary tables.
    macro_rows = []
    pairwise_rows = []

    for radius in RADII:
        keyr = "inf" if math.isinf(radius) else str(radius)

        by_method = {
            m: [
                r for r in agg_rows
                if r["method"] == m and r["radius"] == keyr
            ]
            for m in METHODS
        }

        for method in METHODS:
            vals = by_method[method]
            macro_rows.append({
                "radius": keyr,
                "method": method,
                "n": len(vals),
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
            })

        # Index rows by sample.
        idx = {
            m: {int(v["sample"]): v for v in by_method[m]}
            for m in METHODS
        }

        counts = {
            "F1_lower_cost_than_CNN": 0,
            "F1_lower_cost_than_UV": 0,
            "F1_higher_GTcoverage_than_CNN": 0,
            "F1_higher_GTcoverage_than_UV": 0,
            "F1_lower_cost_AND_higher_GTcoverage_than_CNN": 0,
            "F1_lower_cost_AND_higher_GTcoverage_than_UV": 0,
        }

        for s in ALL_SAMPLES:
            f, c, u = idx["F1"][s], idx["CNN"][s], idx["UV"][s]
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
            counts["F1_lower_cost_than_CNN"] += int(lc)
            counts["F1_lower_cost_than_UV"] += int(lu)
            counts["F1_higher_GTcoverage_than_CNN"] += int(gc)
            counts["F1_higher_GTcoverage_than_UV"] += int(gu)
            counts["F1_lower_cost_AND_higher_GTcoverage_than_CNN"] += int(lc and gc)
            counts["F1_lower_cost_AND_higher_GTcoverage_than_UV"] += int(lu and gu)

        pairwise_rows.append({
            "radius": keyr,
            "n_samples": len(ALL_SAMPLES),
            **counts,
        })

    # Save outputs.
    def write_csv(path, rows):
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    write_csv(out / "all168_locality_dimension.csv", dim_rows)
    write_csv(out / "all168_locality_aggregate_by_sample.csv", agg_rows)
    write_csv(out / "all168_locality_macro_summary.csv", macro_rows)
    write_csv(out / "all168_locality_pairwise_counts.csv", pairwise_rows)

    manifest = {
        "pd_root": str(pdroot),
        "samples": 168,
        "methods": METHODS,
        "radii": ["inf" if math.isinf(r) else r for r in RADII],
        "gudhi_version": gudhi.__version__,
        "elapsed_seconds": time.perf_counter() - start_all,
    }
    (out / "all168_locality_manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )

    print()
    print("===== MACRO SUMMARY =====")
    print(
        "radius method mean_cost median_cost mean_gap "
        "mean_GTcov mean_SRcov"
    )
    for r in macro_rows:
        print(
            f"{r['radius']:>5s} {r['method']:>4s} "
            f"{r['mean_constrained_w22_all']:.4f} "
            f"{r['median_constrained_w22_all']:.4f} "
            f"{r['mean_locality_gap_all']:.4f} "
            f"{r['mean_gt_persistence_coverage_all']:.4f} "
            f"{r['mean_sr_persistence_coverage_all']:.4f}"
        )

    print()
    print("===== PAIRWISE COUNTS =====")
    for r in pairwise_rows:
        print(r)

    print()
    print("PHASE 5E ALL-168 LOCALITY ANALYSIS: PASS")
    print("Wrote:", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode",
        required=True,
        choices=["validate-pilot", "all168"],
    )
    ap.add_argument("--pd-root")
    ap.add_argument("--reference-csv")
    ap.add_argument("--out")
    args = ap.parse_args()

    if args.mode == "validate-pilot":
        if not args.pd_root or not args.reference_csv:
            raise SystemExit(
                "--pd-root and --reference-csv are required for validate-pilot"
            )
        validate_pilot(args)
    else:
        if not args.pd_root or not args.out:
            raise SystemExit("--pd-root and --out are required for all168")
        analyze_all168(args)


if __name__ == "__main__":
    main()
