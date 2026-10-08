#!/usr/bin/env python3
"""
Phase 5C Step 3 — bidirectional persistence-mass coverage on sample 69.

Reuses the exact locality-constrained W2,2 solver.

For each method, dimension, and radius, report:
  GT-side persistence coverage:
      GT persistence mass participating in real-real matches / total GT mass

  SR-side persistence coverage:
      SR persistence mass participating in real-real matches / total SR mass

and their complementary unmatched fractions.

No new matching rule is introduced.
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


def coverage(gt, sr, matches):
    rr = [m for m in matches if m["type"] == "real_real"]

    gt_total = sum(float(f["persistence"]) for f in gt)
    sr_total = sum(float(f["persistence"]) for f in sr)

    gt_matched = sum(float(gt[m["gt_index"]]["persistence"]) for m in rr)
    sr_matched = sum(float(sr[m["sr_index"]]["persistence"]) for m in rr)

    gt_cov = gt_matched / gt_total if gt_total > 0 else None
    sr_cov = sr_matched / sr_total if sr_total > 0 else None

    return {
        "gt_total_persistence": gt_total,
        "sr_total_persistence": sr_total,
        "gt_matched_persistence": gt_matched,
        "sr_matched_persistence": sr_matched,
        "gt_persistence_coverage": gt_cov,
        "sr_persistence_coverage": sr_cov,
        "gt_unmatched_persistence_fraction": 1.0 - gt_cov if gt_cov is not None else None,
        "sr_unmatched_persistence_fraction": 1.0 - sr_cov if sr_cov is not None else None,
        "real_real": len(rr),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--solver",
        default=str(
            Path.home() / "PhIRE/spatial_pd/phase5c_sample69_locality_constrained_w22.py"
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
            Path.home() / "PhIRE/spatial_pd/phase5c_sample69_bidirectional_coverage"
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
    for name, path in paths.items():
        features[name], _ = solver.read_features(path)

    rows = []
    report = {
        "sample": 69,
        "solver_path": str(solver_path),
        "solver_sha256": sha256(solver_path),
        "radii": ["inf" if math.isinf(r) else r for r in RADII],
        "methods": {},
    }

    for method in ("CNN", "UV", "F1"):
        print()
        print("=" * 92)
        print(method)
        report["methods"][method] = {}

        for dim in (0, 1):
            gt = features["GT"][dim]
            sr = features[method][dim]

            print()
            print(f"D{dim}")
            print(
                "r      RR   GT_cov   SR_cov   GT_unmatched   SR_unmatched"
            )

            report["methods"][method][f"D{dim}"] = {}

            for radius in RADII:
                _, matches = solver.constrained_match(gt, sr, radius)
                c = coverage(gt, sr, matches)
                key = "inf" if math.isinf(radius) else str(radius)

                report["methods"][method][f"D{dim}"][key] = c
                rows.append({
                    "method": method,
                    "dimension": dim,
                    "radius": key,
                    **c,
                })

                rtxt = "inf" if math.isinf(radius) else f"{radius:g}"
                print(
                    f"{rtxt:>4s}  {c['real_real']:4d}  "
                    f"{c['gt_persistence_coverage']:.4f}  "
                    f"{c['sr_persistence_coverage']:.4f}  "
                    f"{c['gt_unmatched_persistence_fraction']:.4f}  "
                    f"{c['sr_unmatched_persistence_fraction']:.4f}"
                )

    with (out / "sample69_bidirectional_persistence_coverage.csv").open(
        "w", newline=""
    ) as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    (out / "sample69_bidirectional_persistence_coverage.json").write_text(
        json.dumps(report, indent=2)
    )

    print()
    print("SAMPLE-69 BIDIRECTIONAL PERSISTENCE COVERAGE: PASS")
    print("Wrote:", out)


if __name__ == "__main__":
    main()
