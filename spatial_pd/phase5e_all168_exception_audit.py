#!/usr/bin/env python3
"""
Phase 5E closeout — paired-difference and exception audit.

Uses ONLY the already-computed:
    all168_locality_aggregate_by_sample.csv

No new persistence matching is performed.

Reports:
- paired F1-CNN and F1-UV constrained-cost differences;
- paired GT-coverage differences;
- quartiles and sample counts;
- exception sample IDs where F1 cost is not lower;
- first tested radius where F1 becomes lower than BOTH baselines and stays
  lower through every wider tested radius.
"""

from pathlib import Path
import argparse
import csv
import math
import numpy as np
import pandas as pd

RADII = ["4.0", "8.0", "16.0", "32.0", "64.0", "96.0", "128.0", "inf"]


def qstats(x):
    a = np.asarray(x, dtype=float)
    return {
        "mean": float(np.mean(a)),
        "median": float(np.median(a)),
        "q25": float(np.quantile(a, 0.25)),
        "q75": float(np.quantile(a, 0.75)),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
    }


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
    args = ap.parse_args()

    path = Path(args.csv).expanduser().resolve()
    df = pd.read_csv(path, dtype={"radius": str})

    expected = 168 * 3 * 8
    if len(df) != expected:
        raise SystemExit(f"Expected {expected} rows, found {len(df)}")

    print("===== INPUT GATE =====")
    print("rows:", len(df))
    print("samples:", df["sample"].nunique())
    print("methods:", sorted(df["method"].unique()))
    print("radii:", list(df["radius"].drop_duplicates()))
    print("INPUT GATE: PASS")
    print()

    by = {
        (int(r.sample), r.method, str(r.radius)): r
        for r in df.itertuples(index=False)
    }

    print("===== PAIRED DIFFERENCE DISTRIBUTIONS =====")
    print(
        "radius contrast "
        "mean_dcost median_dcost q25_dcost q75_dcost "
        "mean_dGTcov median_dGTcov"
    )

    for radius in RADII:
        for base in ("CNN", "UV"):
            dc = []
            dg = []
            for s in range(168):
                f = by[(s, "F1", radius)]
                b = by[(s, base, radius)]
                dc.append(
                    float(f.constrained_w22_all)
                    - float(b.constrained_w22_all)
                )
                dg.append(
                    float(f.gt_persistence_coverage_all)
                    - float(b.gt_persistence_coverage_all)
                )

            cs = qstats(dc)
            gs = qstats(dg)

            print(
                f"{radius:>5s} F1-{base:<3s} "
                f"{cs['mean']:+.5f} {cs['median']:+.5f} "
                f"{cs['q25']:+.5f} {cs['q75']:+.5f} "
                f"{gs['mean']:+.5f} {gs['median']:+.5f}"
            )

    print()
    print("===== COST EXCEPTIONS =====")
    for radius in RADII:
        print()
        print("radius =", radius)

        for base in ("CNN", "UV"):
            exc = []
            for s in range(168):
                f = by[(s, "F1", radius)]
                b = by[(s, base, radius)]
                delta = (
                    float(f.constrained_w22_all)
                    - float(b.constrained_w22_all)
                )
                if delta >= 0.0:
                    exc.append((
                        s,
                        delta,
                        float(f.constrained_w22_all),
                        float(b.constrained_w22_all),
                        float(f.gt_persistence_coverage_all)
                        - float(b.gt_persistence_coverage_all),
                    ))

            exc.sort(key=lambda x: x[1], reverse=True)

            print(
                f"  vs {base}: exceptions={len(exc)}/168"
            )

            for s, delta, fcost, bcost, dcov in exc[:20]:
                print(
                    f"    s={s:3d} "
                    f"dcost={delta:+.6f} "
                    f"F1={fcost:.6f} "
                    f"{base}={bcost:.6f} "
                    f"dGTcov={dcov:+.6f}"
                )

    print()
    print("===== FIRST STABLE JOINT-ADVANTAGE RADIUS =====")
    print(
        "Definition: first tested radius where F1 has lower cost than BOTH "
        "CNN and UV, and remains lower at every wider tested radius."
    )

    counts = {}
    sample_results = []

    for s in range(168):
        flags = []
        for radius in RADII:
            f = by[(s, "F1", radius)]
            c = by[(s, "CNN", radius)]
            u = by[(s, "UV", radius)]
            flags.append(
                float(f.constrained_w22_all)
                < float(c.constrained_w22_all)
                and
                float(f.constrained_w22_all)
                < float(u.constrained_w22_all)
            )

        first = "none"
        for i, radius in enumerate(RADII):
            if all(flags[i:]):
                first = radius
                break

        counts[first] = counts.get(first, 0) + 1
        sample_results.append((s, first))

    order = RADII + ["none"]
    for key in order:
        if key in counts:
            print(f"{key:>5s}: {counts[key]:3d}")

    print()
    print("Samples with no stable joint advantage:")
    print([s for s, first in sample_results if first == "none"])

    print()
    print("PHASE 5E PAIRED-DIFFERENCE / EXCEPTION AUDIT: PASS")


if __name__ == "__main__":
    main()
