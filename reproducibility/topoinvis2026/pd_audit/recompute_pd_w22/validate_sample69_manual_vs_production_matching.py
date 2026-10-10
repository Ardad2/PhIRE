#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
import os
from pathlib import Path

AUDIT = Path.home() / "phire_runtime_audit_20260809_221548"
W22 = Path(os.environ.get("W22", AUDIT / "recompute_pd_w22"))
BASE = (
    W22
    / "corrected_pd_mt"
    / "discordance_visuals"
    / "sample_069"
    / "ttk_matching_host96"
)
MANUAL = BASE / "cost_attribution"

TOL = 1e-8

print("SAMPLE-69 PRODUCTION-STYLE VS MANUAL-PREPROCESS RAW MATCHING AUDIT")
print("=" * 104)

summary_rows = []

for label in ("cnn", "uv", "f1"):
    prod_path = BASE / f"{label}_raw_matching.csv"
    man_path = MANUAL / f"{label}_raw_matching_manual_preprocess.csv"

    def load(path: Path):
        with path.open(newline="") as f:
            rows = list(csv.DictReader(f))
        out = {}
        duplicates = []
        for r in rows:
            pair = (int(r["tree1_node_id"]), int(r["tree2_node_id"]))
            cost = float(r["relabel_cost"])
            if pair in out:
                duplicates.append(pair)
            out[pair] = cost
        return rows, out, duplicates

    prod_rows, prod, prod_dups = load(prod_path)
    man_rows, man, man_dups = load(man_path)

    prod_only = sorted(set(prod) - set(man))
    man_only = sorted(set(man) - set(prod))
    common = sorted(set(prod) & set(man))

    cost_diffs = [
        (pair, abs(prod[pair] - man[pair]))
        for pair in common
    ]
    max_cost_diff = max((d for _, d in cost_diffs), default=0.0)
    bad_costs = [(p, d) for p, d in cost_diffs if d > TOL]

    ok = (
        len(prod_rows) == len(man_rows)
        and not prod_dups
        and not man_dups
        and not prod_only
        and not man_only
        and not bad_costs
    )

    print()
    print(label.upper())
    print("-" * 104)
    print(f"production rows       = {len(prod_rows)}")
    print(f"manual rows           = {len(man_rows)}")
    print(f"production duplicates = {len(prod_dups)}")
    print(f"manual duplicates     = {len(man_dups)}")
    print(f"production-only pairs = {len(prod_only)}")
    print(f"manual-only pairs     = {len(man_only)}")
    print(f"max relabel cost diff = {max_cost_diff:.17g}")
    print(f"bad cost pairs > tol  = {len(bad_costs)}")
    print(f"PASS                   = {int(ok)}")

    if prod_only[:10]:
        print("first production-only:", prod_only[:10])
    if man_only[:10]:
        print("first manual-only:", man_only[:10])
    if bad_costs[:10]:
        print("first bad costs:", bad_costs[:10])

    summary_rows.append({
        "label": label,
        "production_rows": len(prod_rows),
        "manual_rows": len(man_rows),
        "production_duplicates": len(prod_dups),
        "manual_duplicates": len(man_dups),
        "production_only_pairs": len(prod_only),
        "manual_only_pairs": len(man_only),
        "max_relabel_cost_abs_diff": max_cost_diff,
        "bad_cost_pairs": len(bad_costs),
        "pass": int(ok),
    })

out = MANUAL / "sample69_manual_vs_production_raw_matching_validation.csv"
with out.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
    w.writeheader()
    w.writerows(summary_rows)

print()
print("CSV:", out)

if not all(r["pass"] == 1 for r in summary_rows):
    raise SystemExit(
        "FAIL: manual preprocessing did not reproduce the exact production-style raw matching."
    )

print()
print("ALL THREE RAW MATCHINGS IDENTICAL: PASS")
print("Nodewise delete/insert attribution may proceed to spatial mapping.")
