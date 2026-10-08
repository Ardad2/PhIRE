#!/usr/bin/env python3
"""
Phase 5E closeout summary extractor.

Reads only the already-computed:
    all168_locality_aggregate_by_sample.csv

Produces compact presentation-ready tables:
1) paired F1-minus-baseline constrained-cost distributions by radius;
2) paired GT-coverage differences by radius;
3) first stable joint-advantage radius histogram;
4) exception samples by radius.

No persistence matching or new experiment is performed.
"""

from pathlib import Path
import argparse
import csv
import json
import numpy as np

RADII = ["4.0", "8.0", "16.0", "32.0", "64.0", "96.0", "128.0", "inf"]
BASELINES = ["CNN", "UV"]


def read_rows(path):
    rows = []
    with path.open(newline="") as f:
        for r in csv.DictReader(f):
            rows.append({
                "sample": int(r["sample"]),
                "method": r["method"],
                "radius": r["radius"],
                "constrained_w22_all": float(r["constrained_w22_all"]),
                "gt_persistence_coverage_all":
                    float(r["gt_persistence_coverage_all"]),
                "sr_persistence_coverage_all":
                    float(r["sr_persistence_coverage_all"]),
            })
    return rows


def stats(values):
    a = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(a)),
        "median": float(np.median(a)),
        "q25": float(np.quantile(a, 0.25)),
        "q75": float(np.quantile(a, 0.75)),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
    }


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv",
        default=str(
            Path.home()
            / "PhIRE/spatial_pd/phase5e_all168_locality_analysis"
            / "all168_locality_aggregate_by_sample.csv"
        ),
    )
    ap.add_argument(
        "--out",
        default=str(
            Path.home()
            / "PhIRE/spatial_pd/phase5e_all168_closeout_summary"
        ),
    )
    args = ap.parse_args()

    src = Path(args.csv).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    rows = read_rows(src)
    expected = 168 * 3 * 8

    if len(rows) != expected:
        raise SystemExit(f"Expected {expected} rows; found {len(rows)}")

    by = {
        (r["sample"], r["method"], r["radius"]): r
        for r in rows
    }

    # ---------- paired-difference distributions ----------
    paired_rows = []

    for radius in RADII:
        for base in BASELINES:
            dcost = []
            dgt = []
            dsr = []

            for s in range(168):
                f = by[(s, "F1", radius)]
                b = by[(s, base, radius)]

                dcost.append(
                    f["constrained_w22_all"]
                    - b["constrained_w22_all"]
                )
                dgt.append(
                    f["gt_persistence_coverage_all"]
                    - b["gt_persistence_coverage_all"]
                )
                dsr.append(
                    f["sr_persistence_coverage_all"]
                    - b["sr_persistence_coverage_all"]
                )

            cs = stats(dcost)
            gs = stats(dgt)
            ss = stats(dsr)

            paired_rows.append({
                "radius": radius,
                "contrast": f"F1-{base}",
                "cost_mean": cs["mean"],
                "cost_median": cs["median"],
                "cost_q25": cs["q25"],
                "cost_q75": cs["q75"],
                "cost_min": cs["min"],
                "cost_max": cs["max"],
                "gtcov_mean": gs["mean"],
                "gtcov_median": gs["median"],
                "gtcov_q25": gs["q25"],
                "gtcov_q75": gs["q75"],
                "srcov_mean": ss["mean"],
                "srcov_median": ss["median"],
                "srcov_q25": ss["q25"],
                "srcov_q75": ss["q75"],
                "cost_lower_count":
                    int(np.sum(np.asarray(dcost) < 0.0)),
                "gtcov_higher_count":
                    int(np.sum(np.asarray(dgt) > 0.0)),
            })

    # ---------- first stable joint-advantage radius ----------
    stable_per_sample = []
    hist = {r: 0 for r in RADII}
    hist["none"] = 0

    for s in range(168):
        flags = []

        for radius in RADII:
            f = by[(s, "F1", radius)]
            c = by[(s, "CNN", radius)]
            u = by[(s, "UV", radius)]

            flags.append(
                f["constrained_w22_all"] < c["constrained_w22_all"]
                and
                f["constrained_w22_all"] < u["constrained_w22_all"]
            )

        first = "none"
        for i, radius in enumerate(RADII):
            if all(flags[i:]):
                first = radius
                break

        hist[first] += 1
        stable_per_sample.append({
            "sample": s,
            "first_stable_joint_advantage_radius": first,
        })

    hist_rows = [
        {
            "first_stable_radius": key,
            "count": hist[key],
            "fraction": hist[key] / 168.0,
        }
        for key in RADII + ["none"]
        if hist[key] > 0
    ]

    # ---------- exception inventory ----------
    exception_rows = []

    for radius in RADII:
        for base in BASELINES:
            for s in range(168):
                f = by[(s, "F1", radius)]
                b = by[(s, base, radius)]
                delta = (
                    f["constrained_w22_all"]
                    - b["constrained_w22_all"]
                )

                if delta >= 0:
                    exception_rows.append({
                        "radius": radius,
                        "baseline": base,
                        "sample": s,
                        "cost_difference_F1_minus_baseline": delta,
                        "gtcov_difference_F1_minus_baseline":
                            f["gt_persistence_coverage_all"]
                            - b["gt_persistence_coverage_all"],
                    })

    write_csv(out / "paired_difference_summary.csv", paired_rows)
    write_csv(out / "stable_radius_histogram.csv", hist_rows)
    write_csv(out / "stable_radius_by_sample.csv", stable_per_sample)
    write_csv(out / "cost_exception_inventory.csv", exception_rows)

    summary = {
        "source_csv": str(src),
        "n_samples": 168,
        "radii": RADII,
        "stable_radius_histogram": hist,
        "paired_difference_summary": paired_rows,
    }
    (out / "phase5e_closeout_summary.json").write_text(
        json.dumps(summary, indent=2)
    )

    # ---------- concise terminal output ----------
    print("===== STABLE JOINT-ADVANTAGE RADIUS HISTOGRAM =====")
    for r in hist_rows:
        print(
            f"{r['first_stable_radius']:>5s}: "
            f"{r['count']:3d}/168 "
            f"({100*r['fraction']:.2f}%)"
        )

    print()
    print("===== PAIRED DIFFERENCE QUARTILES =====")
    print(
        "radius contrast "
        "cost_med [q25,q75] "
        "GTcov_med [q25,q75] "
        "lower_cost_count"
    )

    for r in paired_rows:
        print(
            f"{r['radius']:>5s} {r['contrast']:<6s} "
            f"{r['cost_median']:+.4f} "
            f"[{r['cost_q25']:+.4f},{r['cost_q75']:+.4f}] "
            f"{r['gtcov_median']:+.4f} "
            f"[{r['gtcov_q25']:+.4f},{r['gtcov_q75']:+.4f}] "
            f"{r['cost_lower_count']:3d}/168"
        )

    print()
    print("===== NEVER-STABLE SAMPLES =====")
    print([
        r["sample"]
        for r in stable_per_sample
        if r["first_stable_joint_advantage_radius"] == "none"
    ])

    print()
    print("PHASE 5E CLOSEOUT SUMMARY EXTRACTION: PASS")
    print("Wrote:", out)


if __name__ == "__main__":
    main()
