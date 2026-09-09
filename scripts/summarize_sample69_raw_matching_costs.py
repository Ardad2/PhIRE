#!/usr/bin/env python3
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

print("SAMPLE-69 RAW TTK MATCHING COST SUMMARY")
print("=" * 100)

rows_out = []

for label in ("cnn", "uv", "f1"):
    match_path = OUT / f"{label}_raw_matching.csv"
    summary_path = OUT / f"{label}_raw_summary.csv"

    with match_path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    with summary_path.open(newline="") as f:
        summary = list(csv.DictReader(f))

    if len(summary) != 1:
        raise RuntimeError(f"Bad summary file: {summary_path}")

    costs = [float(r["relabel_cost"]) for r in rows]
    distance = float(summary[0]["distance"])

    s1 = sum(costs)
    s2 = sum(c * c for c in costs)
    sqrt_s1 = math.sqrt(s1) if s1 >= 0 else float("nan")
    sqrt_s2 = math.sqrt(s2) if s2 >= 0 else float("nan")
    cmin = min(costs) if costs else float("nan")
    cmax = max(costs) if costs else float("nan")
    cmean = s1 / len(costs) if costs else float("nan")

    r = {
        "label": label,
        "distance": distance,
        "raw_match_count": len(costs),
        "sum_relabel_cost": s1,
        "sum_cost_squared": s2,
        "sqrt_sum_relabel_cost": sqrt_s1,
        "sqrt_sum_cost_squared": sqrt_s2,
        "min_cost": cmin,
        "mean_cost": cmean,
        "max_cost": cmax,
        "distance_minus_sqrt_sum": distance - sqrt_s1,
        "distance_minus_sqrt_sumsq": distance - sqrt_s2,
    }
    rows_out.append(r)

    print()
    print(label.upper())
    print("-" * 100)
    for k, v in r.items():
        if k == "label":
            continue
        if isinstance(v, float):
            print(f"{k:30s} {v:.17g}")
        else:
            print(f"{k:30s} {v}")

out_csv = OUT / "sample69_raw_matching_cost_summary.csv"
with out_csv.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
    w.writeheader()
    w.writerows(rows_out)

print()
print("CSV:", out_csv)
print()
print(
    "NOTE: These are descriptive aggregates only. Do NOT interpret them as a "
    "distance decomposition until the exact TTK delete/insert/relabel and "
    "final-normalization source path has been audited."
)
