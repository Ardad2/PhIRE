#!/usr/bin/env python3
"""
Phase 5C Step 2 — sample-69 locality-cost decomposition.

Reuses the exact functions from:
    phase5c_sample69_locality_constrained_w22.py

For each method, dimension, and radius, decompose the order-2 objective:

    W22(r)^2
      = sum_real-real c^2
      + sum_GT->diag c^2
      + sum_diag->SR c^2

This distinguishes:
- scalar mismatch among accepted local correspondences;
- unmatched GT topological mass;
- unmatched SR topological mass.

Also evaluates a refined radius grid between 8 and 16 pixels.

This is a diagnostic decomposition of the same constrained objective, not a
new metric.
"""

from pathlib import Path
import argparse
import csv
import hashlib
import importlib.util
import json
import math


RADII = [4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0,
         20.0, 24.0, 32.0, 64.0, 96.0, 128.0, math.inf]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def load_solver(path):
    spec = importlib.util.spec_from_file_location("phase5c_solver", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load solver: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def components(matches):
    sq = {
        "real_real": 0.0,
        "gt_to_diagonal": 0.0,
        "diagonal_to_sr": 0.0,
    }
    count = {k: 0 for k in sq}

    for m in matches:
        typ = m["type"]
        c = float(m["ground_cost"])
        sq[typ] += c * c
        count[typ] += 1

    total_sq = sum(sq.values())

    return {
        "total_sq": total_sq,
        "rr_sq": sq["real_real"],
        "gt_diag_sq": sq["gt_to_diagonal"],
        "sr_diag_sq": sq["diagonal_to_sr"],
        "rr_count": count["real_real"],
        "gt_diag_count": count["gt_to_diagonal"],
        "sr_diag_count": count["diagonal_to_sr"],
        "rr_sq_fraction": sq["real_real"] / total_sq if total_sq else 0.0,
        "gt_diag_sq_fraction":
            sq["gt_to_diagonal"] / total_sq if total_sq else 0.0,
        "sr_diag_sq_fraction":
            sq["diagonal_to_sr"] / total_sq if total_sq else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--solver",
        default=str(
            Path.home()
            / "PhIRE/spatial_pd/phase5c_sample69_locality_constrained_w22.py"
        ),
    )
    ap.add_argument(
        "--phase5",
        default=str(
            Path.home() / "PhIRE/spatial_pd/phase5a_sample69_canonical_pd"
        ),
    )
    ap.add_argument(
        "--out",
        default=str(
            Path.home() / "PhIRE/spatial_pd/phase5c_sample69_locality_decomposition"
        ),
    )
    args = ap.parse_args()

    solver_path = Path(args.solver).expanduser().resolve()
    phase5 = Path(args.phase5).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    solver = load_solver(solver_path)
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
        features[name], zeros[name] = solver.read_features(path)

    rows = []
    aggregate_rows = []
    report = {
        "sample": 69,
        "solver_path": str(solver_path),
        "solver_sha256": sha256(solver_path),
        "radii": ["inf" if math.isinf(r) else r for r in RADII],
        "methods": {},
    }

    for method in ("CNN", "UV", "F1"):
        print()
        print("=" * 88)
        print(method)
        report["methods"][method] = {"dimensions": {}, "aggregate": {}}

        base_dim = {}

        for dim in (0, 1):
            gt = features["GT"][dim]
            sr = features[method][dim]
            base = solver.gudhi_w22(gt, sr)
            base_dim[dim] = base

            report["methods"][method]["dimensions"][f"D{dim}"] = {}
            print()
            print(f"D{dim} unconstrained W22 = {base:.12f}")
            print(
                "r    W22       gap       RR_sq%  GTdiag_sq%  SRdiag_sq%  "
                "RR  GTdiag SRdiag"
            )

            for radius in RADII:
                d, matches = solver.constrained_match(gt, sr, radius)
                c = components(matches)

                if not math.isclose(
                    d * d, c["total_sq"], rel_tol=1e-12, abs_tol=1e-10
                ):
                    raise RuntimeError(
                        f"{method} D{dim} r={radius}: decomposition mismatch"
                    )

                key = "inf" if math.isinf(radius) else str(radius)
                payload = {
                    "w22": d,
                    "gap": d - base,
                    **c,
                }
                report["methods"][method]["dimensions"][f"D{dim}"][key] = payload

                rows.append({
                    "method": method,
                    "dimension": dim,
                    "radius": key,
                    "unconstrained_w22": base,
                    **payload,
                })

                rtxt = "inf" if math.isinf(radius) else f"{radius:g}"
                print(
                    f"{rtxt:>4s} {d:9.4f} {d-base:+9.4f} "
                    f"{100*c['rr_sq_fraction']:8.2f} "
                    f"{100*c['gt_diag_sq_fraction']:10.2f} "
                    f"{100*c['sr_diag_sq_fraction']:10.2f} "
                    f"{c['rr_count']:4d} "
                    f"{c['gt_diag_count']:6d} "
                    f"{c['sr_diag_count']:6d}"
                )

        base_all = math.hypot(base_dim[0], base_dim[1])

        print()
        print("AGGREGATE")
        print(
            "r    W22_all    gap_all    RR_sq%  GTdiag_sq%  SRdiag_sq%"
        )

        for radius in RADII:
            key = "inf" if math.isinf(radius) else str(radius)

            d0 = report["methods"][method]["dimensions"]["D0"][key]
            d1 = report["methods"][method]["dimensions"]["D1"][key]

            total_sq = d0["total_sq"] + d1["total_sq"]
            rr_sq = d0["rr_sq"] + d1["rr_sq"]
            gd_sq = d0["gt_diag_sq"] + d1["gt_diag_sq"]
            sd_sq = d0["sr_diag_sq"] + d1["sr_diag_sq"]

            all_d = math.sqrt(total_sq)

            payload = {
                "w22_all": all_d,
                "gap_all": all_d - base_all,
                "total_sq": total_sq,
                "rr_sq": rr_sq,
                "gt_diag_sq": gd_sq,
                "sr_diag_sq": sd_sq,
                "rr_sq_fraction": rr_sq / total_sq if total_sq else 0.0,
                "gt_diag_sq_fraction": gd_sq / total_sq if total_sq else 0.0,
                "sr_diag_sq_fraction": sd_sq / total_sq if total_sq else 0.0,
            }

            report["methods"][method]["aggregate"][key] = payload
            aggregate_rows.append({
                "method": method,
                "radius": key,
                "unconstrained_w22_all": base_all,
                **payload,
            })

            rtxt = "inf" if math.isinf(radius) else f"{radius:g}"
            print(
                f"{rtxt:>4s} {all_d:10.4f} {all_d-base_all:+10.4f} "
                f"{100*payload['rr_sq_fraction']:8.2f} "
                f"{100*payload['gt_diag_sq_fraction']:10.2f} "
                f"{100*payload['sr_diag_sq_fraction']:10.2f}"
            )

    # Cross-method aggregate table around the transition.
    print()
    print("=" * 88)
    print("CROSS-METHOD AGGREGATE COST")
    print("r       CNN        UV         F1")

    for radius in RADII:
        key = "inf" if math.isinf(radius) else str(radius)
        vals = {
            m: report["methods"][m]["aggregate"][key]["w22_all"]
            for m in ("CNN", "UV", "F1")
        }
        rtxt = "inf" if math.isinf(radius) else f"{radius:g}"
        print(
            f"{rtxt:>4s}  {vals['CNN']:9.4f}  "
            f"{vals['UV']:9.4f}  {vals['F1']:9.4f}"
        )

    with (out / "sample69_locality_cost_decomposition_dimension.csv").open(
        "w", newline=""
    ) as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    with (out / "sample69_locality_cost_decomposition_aggregate.csv").open(
        "w", newline=""
    ) as f:
        w = csv.DictWriter(f, fieldnames=list(aggregate_rows[0].keys()))
        w.writeheader()
        w.writerows(aggregate_rows)

    (out / "sample69_locality_cost_decomposition.json").write_text(
        json.dumps(report, indent=2)
    )

    print()
    print("SAMPLE-69 LOCALITY COST DECOMPOSITION: PASS")
    print("Wrote:", out)


if __name__ == "__main__":
    main()
