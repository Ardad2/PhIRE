#!/usr/bin/env python3
"""
Exact aggregate decomposition of the validated sample-69 TTK merge-tree
distance into:

  (A) explicit real-to-real relabel contribution
  (B) nonmatching contribution = delete/insert residual

This is exact for the audited configuration because:
  - wassersteinPower = 2
  - distanceSquaredRoot = true
  - keepSubtree = false
  - useMinMaxPair = true
  - minMaxPairWeight = 1
  - raw outputMatching contains every explicit real-to-real backtracked match
    and stores relabelCost for that match
  - TTK's dummy assignment costs are included in the DP objective but omitted
    from outputMatching

Hence:
    total_squared = distance**2
    relabel_squared = sum(raw relabel_cost)
    nonmatching_squared = total_squared - relabel_squared

The two squared contributions add exactly. Their square roots do NOT add
linearly; they are reported only as component magnitudes.

This script does not yet split nonmatching_squared into tree1 deletions vs
tree2 insertions. That requires explicit dummy-assignment instrumentation.
"""

from __future__ import annotations

import csv
import math
import os
from pathlib import Path

AUDIT = Path.home() / "phire_runtime_audit_20260809_221548"
W22 = Path(os.environ.get("W22", AUDIT / "recompute_pd_w22"))
OUT = (
    W22
    / "corrected_pd_mt"
    / "discordance_visuals"
    / "sample_069"
    / "ttk_matching_host96"
)

LABELS = ("cnn", "uv", "f1")

rows_out = []

print("SAMPLE-69 EXACT TTK MT AGGREGATE COST DECOMPOSITION")
print("=" * 112)

for label in LABELS:
    match_path = OUT / f"{label}_raw_matching.csv"
    summary_path = OUT / f"{label}_raw_summary.csv"

    with match_path.open(newline="") as f:
        matches = list(csv.DictReader(f))

    with summary_path.open(newline="") as f:
        summary = list(csv.DictReader(f))

    if len(summary) != 1:
        raise RuntimeError(f"Expected exactly one row in {summary_path}")

    d = float(summary[0]["distance"])
    costs = [float(r["relabel_cost"]) for r in matches]
    raw_pairs = [
        (int(r["tree1_node_id"]), int(r["tree2_node_id"]))
        for r in matches
    ]
    unique_raw_pairs = len(set(raw_pairs))

    if unique_raw_pairs != len(raw_pairs):
        raise RuntimeError(
            f"{label}: raw matching contains duplicate real-real pairs: "
            f"{unique_raw_pairs} unique of {len(raw_pairs)} rows"
        )

    total_sq = d * d
    relabel_sq = math.fsum(costs)
    nonmatching_sq = total_sq - relabel_sq

    # Float implementation + CSV serialization can create tiny residual noise.
    tol = 1e-5 * max(1.0, total_sq)
    if nonmatching_sq < -tol:
        raise RuntimeError(
            f"{label}: negative nonmatching residual {nonmatching_sq}; "
            f"total_sq={total_sq}, relabel_sq={relabel_sq}"
        )
    if abs(nonmatching_sq) <= tol and nonmatching_sq < 0:
        nonmatching_sq = 0.0

    relabel_pct = 100.0 * relabel_sq / total_sq if total_sq else float("nan")
    nonmatching_pct = (
        100.0 * nonmatching_sq / total_sq if total_sq else float("nan")
    )

    row = {
        "label": label,
        "distance": d,
        "distance_squared": total_sq,
        "raw_real_to_real_match_count": len(matches),
        "unique_raw_real_to_real_pairs": unique_raw_pairs,
        "relabel_squared_contribution": relabel_sq,
        "nonmatching_delete_insert_squared_contribution": nonmatching_sq,
        "relabel_percent_of_squared_objective": relabel_pct,
        "nonmatching_percent_of_squared_objective": nonmatching_pct,
        "sqrt_relabel_component": math.sqrt(max(0.0, relabel_sq)),
        "sqrt_nonmatching_component": math.sqrt(max(0.0, nonmatching_sq)),
        "recomposition_squared": relabel_sq + nonmatching_sq,
        "recomposition_abs_error": abs(
            total_sq - (relabel_sq + nonmatching_sq)
        ),
        "min_real_relabel_cost": min(costs) if costs else float("nan"),
        "mean_real_relabel_cost": (
            relabel_sq / len(costs) if costs else float("nan")
        ),
        "max_real_relabel_cost": max(costs) if costs else float("nan"),
    }
    rows_out.append(row)

    print()
    print(label.upper())
    print("-" * 112)
    print(f"distance                               = {d:.17g}")
    print(f"distance^2                             = {total_sq:.17g}")
    print(f"explicit real-real matches             = {len(matches)}")
    print(f"unique real-real match pairs           = {unique_raw_pairs}")
    print(f"relabel squared contribution           = {relabel_sq:.17g}")
    print(f"delete/insert residual contribution    = {nonmatching_sq:.17g}")
    print(f"relabel share of squared objective     = {relabel_pct:.9f}%")
    print(f"nonmatching share of squared objective = {nonmatching_pct:.9f}%")
    print(
        f"sqrt component magnitudes              = "
        f"{math.sqrt(max(0.0, relabel_sq)):.12g} relabel, "
        f"{math.sqrt(max(0.0, nonmatching_sq)):.12g} nonmatching"
    )
    print(
        "NOTE: sqrt component magnitudes combine by quadrature, not addition."
    )

csv_path = OUT / "sample69_exact_mt_aggregate_cost_decomposition.csv"
with csv_path.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
    w.writeheader()
    w.writerows(rows_out)

print()
print("=" * 112)
print("PAIRWISE DIFFERENCE DECOMPOSITION IN SQUARED OBJECTIVE")
print("=" * 112)

by = {r["label"]: r for r in rows_out}

for a, b in (("f1", "cnn"), ("f1", "uv"), ("uv", "cnn")):
    da = by[a]
    db = by[b]

    total_delta = da["distance_squared"] - db["distance_squared"]
    relabel_delta = (
        da["relabel_squared_contribution"]
        - db["relabel_squared_contribution"]
    )
    nonmatching_delta = (
        da["nonmatching_delete_insert_squared_contribution"]
        - db["nonmatching_delete_insert_squared_contribution"]
    )

    print()
    print(f"{a.upper()} - {b.upper()}")
    print(f"  delta total squared objective = {total_delta:.17g}")
    print(f"  delta relabel contribution    = {relabel_delta:.17g}")
    print(f"  delta nonmatching contribution= {nonmatching_delta:.17g}")
    print(
        f"  recomposition error           = "
        f"{abs(total_delta - relabel_delta - nonmatching_delta):.3e}"
    )

print()
print("CSV:", csv_path)
print()
print(
    "STATUS: aggregate real-relabel vs delete/insert residual decomposition "
    "computed. Individual deletion-vs-insertion attribution remains a "
    "separate instrumentation step."
)
