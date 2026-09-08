#!/usr/bin/env python3

"""
Corrected persistence-diagram / audited merge-tree tradeoff analysis.

STRICT SOURCE POLICY
--------------------
Corrected PD metrics:
    $W22/w22_full_sweep.csv
        bottleneck_all
        w2inf_all
        w22_all

Audited MT metric:
    ~/PhIRE/ttk_runs_fixed/unified_candidate_evaluation/
        unified_primary_per_sample_long.csv::mt_distance

Metadata / mapping only:
    ~/PhIRE/ttk_runs_fixed/unified_candidate_evaluation/
        method_inventory.csv

FORBIDDEN AS PD INPUTS
----------------------
The historical TTK "2" fields are intentionally not used:
    unified_primary_per_sample_long.csv::pd_distance
    method_inventory.csv::topology_mean_pd
    method_inventory.csv::expected_pd
    raw pd_distance_* columns

The script hard-fails on incomplete sample coverage, missing corrected W22 runs,
nonfinite/negative values, or MT mean disagreement with method_inventory.csv.
"""

import csv
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np


HOME = Path.home()
ROOT = HOME / "PhIRE"

if "W22" not in os.environ:
    raise RuntimeError(
        "W22 is not set. Run:\n"
        '  export AUDIT="$HOME/phire_runtime_audit_20260809_221548"\n'
        '  export W22="$AUDIT/recompute_pd_w22"'
    )

W22 = Path(os.environ["W22"])

PD_CSV = W22 / "w22_full_sweep.csv"
LONG_CSV = (
    ROOT
    / "ttk_runs_fixed"
    / "unified_candidate_evaluation"
    / "unified_primary_per_sample_long.csv"
)
INV_CSV = (
    ROOT
    / "ttk_runs_fixed"
    / "unified_candidate_evaluation"
    / "method_inventory.csv"
)

OUT = W22 / "corrected_pd_mt"
OUT.mkdir(parents=True, exist_ok=True)

JOINED_CSV = OUT / "corrected_pd_mt_joined.csv"
METHOD_MEANS_CSV = OUT / "corrected_pd_mt_method_means.csv"
METHOD_CORR_CSV = OUT / "corrected_pd_mt_method_mean_correlations.csv"
WITHIN_METHOD_CSV = OUT / "corrected_pd_mt_within_method_correlations.csv"
WITHIN_METHOD_SUMMARY_CSV = OUT / "corrected_pd_mt_within_method_summary.csv"
SAMPLEWISE_CSV = OUT / "corrected_pd_mt_cross_method_correlations_by_sample.csv"
SAMPLEWISE_SUMMARY_CSV = OUT / "corrected_pd_mt_cross_method_summary.csv"
RESIDUAL_CSV = OUT / "corrected_pd_mt_two_way_residual_correlations.csv"
QUADRANTS_CSV = OUT / "corrected_pd_mt_quadrants_vs_cnn.csv"
CONSENSUS_CSV = OUT / "corrected_pd_mt_all3_consensus_vs_cnn.csv"
PARETO_CSV = OUT / "corrected_pd_mt_pareto_membership.csv"
ARCHETYPES_CSV = OUT / "corrected_pd_mt_sample_archetypes_vs_cnn.csv"
FOCAL_CSV = OUT / "corrected_pd_mt_focal_comparisons.csv"
SUMMARY_TXT = OUT / "corrected_pd_mt_summary.txt"

EXPECTED_SAMPLES = list(range(168))
EPS = 1e-12
MT_MEAN_TOL = 1e-10

PD_METRICS = {
    "dB": "bottleneck_all",
    "W2inf": "w2inf_all",
    "W22": "w22_all",
}


def read_csv(path):
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def fnum(x, label):
    try:
        v = float(x)
    except Exception as e:
        raise RuntimeError(f"Cannot parse {label}: {x!r}") from e
    if not math.isfinite(v):
        raise RuntimeError(f"Nonfinite {label}: {x!r}")
    return v


def average_ranks(values):
    """Average 1-based ranks for ties."""
    a = np.asarray(values, dtype=np.float64)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=np.float64)

    i = 0
    while i < len(a):
        j = i + 1
        while j < len(a) and a[order[j]] == a[order[i]]:
            j += 1
        avg = (i + 1 + j) / 2.0
        ranks[order[i:j]] = avg
        i = j

    return ranks


def pearson(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) != len(y):
        raise RuntimeError("pearson length mismatch")
    if len(x) < 2:
        return float("nan")
    sx = float(np.std(x))
    sy = float(np.std(y))
    if sx == 0.0 or sy == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    return pearson(average_ranks(x), average_ranks(y))


def sign_lower_is_better(candidate, baseline):
    d = candidate - baseline
    if d < -EPS:
        return "improve"
    if d > EPS:
        return "worsen"
    return "tie"


def run_name_for_inventory_row(r):
    mid = r["method_id"]
    if mid in {"cnn", "gan"}:
        return mid

    original = r["original_method_name"].strip()
    if not original:
        raise RuntimeError(f"{mid}: missing original_method_name")

    return f"topology_finetuning/{original}_topology"


def pareto_front(method_rows, objectives):
    """
    Lower is better for every objective.
    Return method_ids on the nondominated front.
    """
    ids = [r["method_id"] for r in method_rows]
    front = []

    for i, a in enumerate(method_rows):
        dominated = False
        for j, b in enumerate(method_rows):
            if i == j:
                continue

            weak = all(
                float(b[obj]) <= float(a[obj]) + EPS
                for obj in objectives
            )
            strict = any(
                float(b[obj]) < float(a[obj]) - EPS
                for obj in objectives
            )

            if weak and strict:
                dominated = True
                break

        if not dominated:
            front.append(ids[i])

    return set(front)


# -----------------------------------------------------------------------------
# Load inventory and define the 18 topology-bearing primary methods
# -----------------------------------------------------------------------------

inventory = read_csv(INV_CSV)

primary_topology = []
for r in inventory:
    if r.get("include_primary", "").strip().lower() != "true":
        continue

    n_top = r.get("row_count_topology", "").strip()
    mt_mean = r.get("topology_mean_mt", "").strip()

    if n_top == "168" and mt_mean:
        primary_topology.append(r)

method_ids = [r["method_id"] for r in primary_topology]

if len(primary_topology) != 18:
    raise RuntimeError(
        "Expected 18 primary topology-bearing methods "
        f"(19 primary methods minus bicubic), found {len(primary_topology)}:\n"
        f"{method_ids}"
    )

if "bicubic" in method_ids:
    raise RuntimeError("Bicubic must not be included: it has no MT topology result.")

if "cnn" not in method_ids or "gan" not in method_ids:
    raise RuntimeError("CNN/GAN missing from topology-bearing primary set.")

inv_by_method = {r["method_id"]: r for r in primary_topology}
run_for_method = {
    mid: run_name_for_inventory_row(inv_by_method[mid])
    for mid in method_ids
}

# -----------------------------------------------------------------------------
# Load corrected PD sweep
# -----------------------------------------------------------------------------

pd_rows = read_csv(PD_CSV)
pd_lookup = {}

required_pd_columns = {"run", "sample"} | set(PD_METRICS.values())
missing = required_pd_columns - set(pd_rows[0].keys())
if missing:
    raise RuntimeError(
        f"{PD_CSV} is missing required corrected-PD columns: {sorted(missing)}"
    )

for r in pd_rows:
    key = (r["run"], int(r["sample"]))
    if key in pd_lookup:
        raise RuntimeError(f"Duplicate corrected PD key: {key}")
    pd_lookup[key] = r

# -----------------------------------------------------------------------------
# Load audited MT from unified long table.
#
# IMPORTANT: pd_distance is deliberately never accessed below.
# -----------------------------------------------------------------------------

long_rows = read_csv(LONG_CSV)

if "mt_distance" not in long_rows[0]:
    raise RuntimeError("unified long table has no mt_distance column")

mt_lookup = {}

for r in long_rows:
    mid = r["method_id"]
    if mid not in inv_by_method:
        continue

    s = int(r["sample_idx"])
    key = (mid, s)

    if key in mt_lookup:
        raise RuntimeError(f"Duplicate MT key: {key}")

    mt = fnum(r["mt_distance"], f"MT {mid} sample {s}")
    if mt < 0:
        raise RuntimeError(f"Negative MT distance: {mid} sample {s}: {mt}")

    mt_lookup[key] = mt

# -----------------------------------------------------------------------------
# Coverage and inventory-mean validation
# -----------------------------------------------------------------------------

for mid in method_ids:
    run = run_for_method[mid]

    mt_samples = sorted(
        s for (m, s) in mt_lookup
        if m == mid
    )
    if mt_samples != EXPECTED_SAMPLES:
        raise RuntimeError(
            f"{mid}: MT sample coverage mismatch: "
            f"n={len(mt_samples)} first={mt_samples[:5]} last={mt_samples[-5:]}"
        )

    pd_samples = sorted(
        s for (rname, s) in pd_lookup
        if rname == run
    )
    if pd_samples != EXPECTED_SAMPLES:
        raise RuntimeError(
            f"{mid}: corrected-PD run {run!r} coverage mismatch: "
            f"n={len(pd_samples)}"
        )

    mt_vals = [mt_lookup[(mid, s)] for s in EXPECTED_SAMPLES]
    recomputed_mt_mean = float(np.mean(mt_vals))
    inventory_mt_mean = fnum(
        inv_by_method[mid]["topology_mean_mt"],
        f"inventory topology_mean_mt {mid}",
    )

    if abs(recomputed_mt_mean - inventory_mt_mean) > MT_MEAN_TOL:
        raise RuntimeError(
            f"{mid}: MT mean mismatch: recomputed={recomputed_mt_mean:.17g} "
            f"inventory={inventory_mt_mean:.17g} "
            f"diff={abs(recomputed_mt_mean - inventory_mt_mean):.3e}"
        )

# -----------------------------------------------------------------------------
# Build corrected joined table
# -----------------------------------------------------------------------------

joined = []

for mid in method_ids:
    inv = inv_by_method[mid]
    run = run_for_method[mid]

    for s in EXPECTED_SAMPLES:
        pr = pd_lookup[(run, s)]

        vals = {}
        for label, col in PD_METRICS.items():
            v = fnum(pr[col], f"{label} {mid} sample {s}")
            if v < 0:
                raise RuntimeError(
                    f"Negative corrected PD distance: {label} {mid} sample {s}"
                )
            vals[label] = v

        mt = mt_lookup[(mid, s)]

        joined.append({
            "sample_idx": s,
            "method_id": mid,
            "display_name": inv["display_name"],
            "candidate_family": inv["candidate_family"],
            "original_method_name": inv["original_method_name"],
            "corrected_pd_run": run,
            "dB": vals["dB"],
            "W2inf": vals["W2inf"],
            "W22": vals["W22"],
            "MT": mt,
        })

with JOINED_CSV.open("w", newline="") as f:
    fieldnames = list(joined[0].keys())
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(joined)

by_method = defaultdict(list)
by_sample = defaultdict(list)

for r in joined:
    by_method[r["method_id"]].append(r)
    by_sample[int(r["sample_idx"])].append(r)

for mid in method_ids:
    if len(by_method[mid]) != 168:
        raise RuntimeError(f"{mid}: joined rows != 168")

for s in EXPECTED_SAMPLES:
    if len(by_sample[s]) != 18:
        raise RuntimeError(f"sample {s}: joined methods != 18")

# -----------------------------------------------------------------------------
# Method means
# -----------------------------------------------------------------------------

method_means = []

for mid in method_ids:
    rows = by_method[mid]
    inv = inv_by_method[mid]

    out = {
        "method_id": mid,
        "display_name": inv["display_name"],
        "candidate_family": inv["candidate_family"],
    }

    for metric in ["dB", "W2inf", "W22", "MT"]:
        a = np.array([float(r[metric]) for r in rows], dtype=np.float64)
        out[f"mean_{metric}"] = float(np.mean(a))
        out[f"median_{metric}"] = float(np.median(a))

    method_means.append(out)

method_means_by_id = {r["method_id"]: r for r in method_means}

with METHOD_MEANS_CSV.open("w", newline="") as f:
    fieldnames = list(method_means[0].keys())
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(method_means)

# -----------------------------------------------------------------------------
# A. Correlations across 18 method means
# -----------------------------------------------------------------------------

method_corr_rows = []

for pd_metric in ["dB", "W2inf", "W22"]:
    x = [r[f"mean_{pd_metric}"] for r in method_means]
    y = [r["mean_MT"] for r in method_means]

    method_corr_rows.append({
        "analysis_level": "across_18_method_means",
        "pd_metric": pd_metric,
        "n": len(x),
        "pearson": pearson(x, y),
        "spearman": spearman(x, y),
    })

with METHOD_CORR_CSV.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(method_corr_rows[0].keys()))
    w.writeheader()
    w.writerows(method_corr_rows)

# -----------------------------------------------------------------------------
# B. Within each method across 168 samples
# -----------------------------------------------------------------------------

within_method_rows = []

for mid in method_ids:
    rows = sorted(by_method[mid], key=lambda r: int(r["sample_idx"]))
    mt = [float(r["MT"]) for r in rows]

    for pd_metric in ["dB", "W2inf", "W22"]:
        pdv = [float(r[pd_metric]) for r in rows]
        within_method_rows.append({
            "method_id": mid,
            "pd_metric": pd_metric,
            "n": 168,
            "pearson": pearson(pdv, mt),
            "spearman": spearman(pdv, mt),
        })

with WITHIN_METHOD_CSV.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(within_method_rows[0].keys()))
    w.writeheader()
    w.writerows(within_method_rows)

within_method_summary = []

for pd_metric in ["dB", "W2inf", "W22"]:
    rr = [r for r in within_method_rows if r["pd_metric"] == pd_metric]
    p = np.array([float(r["pearson"]) for r in rr], dtype=np.float64)
    s = np.array([float(r["spearman"]) for r in rr], dtype=np.float64)

    within_method_summary.append({
        "pd_metric": pd_metric,
        "n_methods": len(rr),
        "median_pearson": float(np.nanmedian(p)),
        "median_spearman": float(np.nanmedian(s)),
        "min_spearman": float(np.nanmin(s)),
        "max_spearman": float(np.nanmax(s)),
    })

with WITHIN_METHOD_SUMMARY_CSV.open("w", newline="") as f:
    w = csv.DictWriter(
        f,
        fieldnames=list(within_method_summary[0].keys()),
    )
    w.writeheader()
    w.writerows(within_method_summary)

# -----------------------------------------------------------------------------
# C. Across methods within each sample
# -----------------------------------------------------------------------------

sample_corr_rows = []

for s in EXPECTED_SAMPLES:
    rows = sorted(by_sample[s], key=lambda r: r["method_id"])
    mt = [float(r["MT"]) for r in rows]

    for pd_metric in ["dB", "W2inf", "W22"]:
        pdv = [float(r[pd_metric]) for r in rows]
        sample_corr_rows.append({
            "sample_idx": s,
            "pd_metric": pd_metric,
            "n_methods": 18,
            "pearson": pearson(pdv, mt),
            "spearman": spearman(pdv, mt),
        })

with SAMPLEWISE_CSV.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(sample_corr_rows[0].keys()))
    w.writeheader()
    w.writerows(sample_corr_rows)

samplewise_summary = []

for pd_metric in ["dB", "W2inf", "W22"]:
    rr = [r for r in sample_corr_rows if r["pd_metric"] == pd_metric]
    p = np.array([float(r["pearson"]) for r in rr], dtype=np.float64)
    s = np.array([float(r["spearman"]) for r in rr], dtype=np.float64)

    samplewise_summary.append({
        "pd_metric": pd_metric,
        "n_samples": len(rr),
        "median_pearson": float(np.nanmedian(p)),
        "median_spearman": float(np.nanmedian(s)),
        "fraction_spearman_positive": float(np.mean(s > 0)),
        "fraction_spearman_negative": float(np.mean(s < 0)),
    })

with SAMPLEWISE_SUMMARY_CSV.open("w", newline="") as f:
    w = csv.DictWriter(
        f,
        fieldnames=list(samplewise_summary[0].keys()),
    )
    w.writeheader()
    w.writerows(samplewise_summary)

# -----------------------------------------------------------------------------
# D. Two-way-centered residual correlations
# -----------------------------------------------------------------------------

# Complete 18 x 168 rectangle for every metric.
ordered_methods = sorted(method_ids)

def matrix_for(metric):
    M = np.empty((len(ordered_methods), 168), dtype=np.float64)
    for i, mid in enumerate(ordered_methods):
        vals = {
            int(r["sample_idx"]): float(r[metric])
            for r in by_method[mid]
        }
        for s in EXPECTED_SAMPLES:
            M[i, s] = vals[s]
    return M


def two_way_residual(M):
    row_mean = np.mean(M, axis=1, keepdims=True)
    col_mean = np.mean(M, axis=0, keepdims=True)
    grand = float(np.mean(M))
    R = M - row_mean - col_mean + grand

    # Numerical guard: row/column residual means should be near zero.
    margin = max(
        float(np.max(np.abs(np.mean(R, axis=0)))),
        float(np.max(np.abs(np.mean(R, axis=1)))),
    )
    if margin > 1e-9:
        raise RuntimeError(
            f"two-way residual centering failure: margin={margin:.3e}"
        )
    return R, margin


MT_M = matrix_for("MT")
MT_R, mt_margin = two_way_residual(MT_M)

residual_rows = []

for pd_metric in ["dB", "W2inf", "W22"]:
    P = matrix_for(pd_metric)
    PR, p_margin = two_way_residual(P)

    residual_rows.append({
        "pd_metric": pd_metric,
        "n_methods": 18,
        "n_samples": 168,
        "n_cells": 18 * 168,
        "pearson": pearson(PR.ravel(), MT_R.ravel()),
        "spearman": spearman(PR.ravel(), MT_R.ravel()),
        "max_centering_margin": max(p_margin, mt_margin),
    })

with RESIDUAL_CSV.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(residual_rows[0].keys()))
    w.writeheader()
    w.writerows(residual_rows)

# -----------------------------------------------------------------------------
# E. Per-metric PD/MT improvement quadrants vs CNN
# -----------------------------------------------------------------------------

cnn_by_sample = {
    int(r["sample_idx"]): r
    for r in by_method["cnn"]
}

quadrant_rows = []

for mid in method_ids:
    if mid == "cnn":
        continue

    rows = {
        int(r["sample_idx"]): r
        for r in by_method[mid]
    }

    for pd_metric in ["dB", "W2inf", "W22"]:
        counts = defaultdict(int)

        for s in EXPECTED_SAMPLES:
            cand = rows[s]
            base = cnn_by_sample[s]

            p = sign_lower_is_better(
                float(cand[pd_metric]),
                float(base[pd_metric]),
            )
            m = sign_lower_is_better(
                float(cand["MT"]),
                float(base["MT"]),
            )

            counts[f"pd_{p}__mt_{m}"] += 1

        quadrant_rows.append({
            "method_id": mid,
            "pd_metric": pd_metric,
            "pd_improve_mt_improve": counts["pd_improve__mt_improve"],
            "pd_improve_mt_worsen": counts["pd_improve__mt_worsen"],
            "pd_worsen_mt_improve": counts["pd_worsen__mt_improve"],
            "pd_worsen_mt_worsen": counts["pd_worsen__mt_worsen"],
            "any_tie_cases": sum(
                v for k, v in counts.items()
                if "tie" in k
            ),
        })

with QUADRANTS_CSV.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(quadrant_rows[0].keys()))
    w.writeheader()
    w.writerows(quadrant_rows)

# -----------------------------------------------------------------------------
# F. All-three corrected-PD consensus vs MT, plus per-sample archetypes
# -----------------------------------------------------------------------------

consensus_rows = []
archetype_rows = []

for mid in method_ids:
    if mid == "cnn":
        continue

    rows = {
        int(r["sample_idx"]): r
        for r in by_method[mid]
    }

    counts = defaultdict(int)

    for s in EXPECTED_SAMPLES:
        cand = rows[s]
        base = cnn_by_sample[s]

        pd_signs = [
            sign_lower_is_better(
                float(cand[m]),
                float(base[m]),
            )
            for m in ["dB", "W2inf", "W22"]
        ]

        mt_sign = sign_lower_is_better(
            float(cand["MT"]),
            float(base["MT"]),
        )

        if all(x == "improve" for x in pd_signs):
            pd_state = "all3_improve"
        elif all(x == "worsen" for x in pd_signs):
            pd_state = "all3_worsen"
        elif any(x == "tie" for x in pd_signs):
            pd_state = "mixed_or_tie"
        else:
            pd_state = "mixed"

        archetype = f"{pd_state}__mt_{mt_sign}"
        counts[archetype] += 1

        pd_gains = {}
        for metric in ["dB", "W2inf", "W22"]:
            b = float(base[metric])
            c = float(cand[metric])
            pd_gains[metric] = (b - c) / b if b != 0 else float("nan")

        bmt = float(base["MT"])
        cmt = float(cand["MT"])
        mt_gain = (bmt - cmt) / bmt if bmt != 0 else float("nan")

        if pd_state == "all3_improve" and mt_sign == "worsen":
            discordance_score = min(pd_gains.values()) + max(0.0, -mt_gain)
        elif pd_state == "all3_worsen" and mt_sign == "improve":
            discordance_score = min(-g for g in pd_gains.values()) + max(0.0, mt_gain)
        else:
            discordance_score = 0.0

        archetype_rows.append({
            "method_id": mid,
            "sample_idx": s,
            "pd_state": pd_state,
            "mt_state": mt_sign,
            "archetype": archetype,
            "dB_gain_vs_cnn": pd_gains["dB"],
            "W2inf_gain_vs_cnn": pd_gains["W2inf"],
            "W22_gain_vs_cnn": pd_gains["W22"],
            "MT_gain_vs_cnn": mt_gain,
            "discordance_score": discordance_score,
        })

    consensus_rows.append({
        "method_id": mid,
        "all3_pd_improve__mt_improve": counts["all3_improve__mt_improve"],
        "all3_pd_improve__mt_worsen": counts["all3_improve__mt_worsen"],
        "all3_pd_worsen__mt_improve": counts["all3_worsen__mt_improve"],
        "all3_pd_worsen__mt_worsen": counts["all3_worsen__mt_worsen"],
        "mixed_pd_cases": sum(
            v for k, v in counts.items()
            if k.startswith("mixed")
        ),
        "mt_tie_cases": sum(
            v for k, v in counts.items()
            if k.endswith("mt_tie")
        ),
    })

with CONSENSUS_CSV.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(consensus_rows[0].keys()))
    w.writeheader()
    w.writerows(consensus_rows)

with ARCHETYPES_CSV.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(archetype_rows[0].keys()))
    w.writeheader()
    w.writerows(archetype_rows)

# -----------------------------------------------------------------------------
# G. Corrected PD/MT Pareto fronts across method means
# -----------------------------------------------------------------------------

pareto_sets = {}

for pd_metric in ["dB", "W2inf", "W22"]:
    pareto_sets[f"{pd_metric}_MT"] = pareto_front(
        method_means,
        [f"mean_{pd_metric}", "mean_MT"],
    )

pareto_sets["all3PD_MT"] = pareto_front(
    method_means,
    ["mean_dB", "mean_W2inf", "mean_W22", "mean_MT"],
)

pareto_rows = []

for r in method_means:
    mid = r["method_id"]
    pareto_rows.append({
        "method_id": mid,
        "mean_dB": r["mean_dB"],
        "mean_W2inf": r["mean_W2inf"],
        "mean_W22": r["mean_W22"],
        "mean_MT": r["mean_MT"],
        "pareto_dB_MT": int(mid in pareto_sets["dB_MT"]),
        "pareto_W2inf_MT": int(mid in pareto_sets["W2inf_MT"]),
        "pareto_W22_MT": int(mid in pareto_sets["W22_MT"]),
        "pareto_all3PD_MT": int(mid in pareto_sets["all3PD_MT"]),
    })

with PARETO_CSV.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(pareto_rows[0].keys()))
    w.writeheader()
    w.writerows(pareto_rows)

# -----------------------------------------------------------------------------
# H. Focal pair comparisons: F1 vs CNN and UV
# -----------------------------------------------------------------------------

focal_pairs = [
    ("f1_grad_e2", "cnn"),
    ("f1_grad_e2", "uv"),
    ("f2_grad_levelset_e2", "cnn"),
    ("candidate_c", "cnn"),
]

focal_rows = []

for cand_mid, base_mid in focal_pairs:
    if cand_mid not in by_method or base_mid not in by_method:
        continue

    cand_map = {
        int(r["sample_idx"]): r
        for r in by_method[cand_mid]
    }
    base_map = {
        int(r["sample_idx"]): r
        for r in by_method[base_mid]
    }

    for s in EXPECTED_SAMPLES:
        c = cand_map[s]
        b = base_map[s]

        row = {
            "candidate_method": cand_mid,
            "baseline_method": base_mid,
            "sample_idx": s,
        }

        pd_states = []
        for metric in ["dB", "W2inf", "W22"]:
            cv = float(c[metric])
            bv = float(b[metric])
            row[f"{metric}_candidate"] = cv
            row[f"{metric}_baseline"] = bv
            row[f"{metric}_gain"] = (bv - cv) / bv if bv != 0 else float("nan")
            pd_states.append(sign_lower_is_better(cv, bv))

        cmt = float(c["MT"])
        bmt = float(b["MT"])
        row["MT_candidate"] = cmt
        row["MT_baseline"] = bmt
        row["MT_gain"] = (bmt - cmt) / bmt if bmt != 0 else float("nan")
        row["MT_state"] = sign_lower_is_better(cmt, bmt)
        row["all3_PD_improve"] = int(all(x == "improve" for x in pd_states))
        row["all3_PD_worsen"] = int(all(x == "worsen" for x in pd_states))

        focal_rows.append(row)

with FOCAL_CSV.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(focal_rows[0].keys()))
    w.writeheader()
    w.writerows(focal_rows)

# -----------------------------------------------------------------------------
# Human-readable summary
# -----------------------------------------------------------------------------

lines = []

def emit(s=""):
    lines.append(s)
    print(s)


emit("CORRECTED PD / AUDITED MT TRADEOFF ANALYSIS")
emit("=" * 92)
emit()
emit("SOURCE POLICY")
emit("-" * 92)
emit("  historical TTK PD used: NO")
emit(f"  corrected PD source: {PD_CSV}")
emit("    columns: bottleneck_all, w2inf_all, w22_all")
emit(f"  audited MT source: {LONG_CSV}::mt_distance")
emit(f"  metadata mapping: {INV_CSV}")
emit("  bicubic excluded from topology integration: YES")
emit()
emit("COVERAGE")
emit("-" * 92)
emit(f"  topology-bearing primary methods: {len(method_ids)}")
emit(f"  samples per method: 168")
emit(f"  joined rows: {len(joined)}")
emit("  expected joined rows: 18 x 168 = 3024")
emit()

emit("ACROSS METHOD MEANS: corrected PD vs MT")
emit("-" * 92)
for r in method_corr_rows:
    emit(
        f"  {r['pd_metric']:5s}: "
        f"Pearson={r['pearson']:+.4f} "
        f"Spearman={r['spearman']:+.4f} "
        f"(n={r['n']})"
    )
emit()

emit("WITHIN-METHOD MEDIAN CORRELATION ACROSS 168 SAMPLES")
emit("-" * 92)
for r in within_method_summary:
    emit(
        f"  {r['pd_metric']:5s}: "
        f"median Pearson={r['median_pearson']:+.4f} "
        f"median Spearman={r['median_spearman']:+.4f}"
    )
emit()

emit("WITHIN-SAMPLE MEDIAN CROSS-METHOD CORRELATION")
emit("-" * 92)
for r in samplewise_summary:
    emit(
        f"  {r['pd_metric']:5s}: "
        f"median Pearson={r['median_pearson']:+.4f} "
        f"median Spearman={r['median_spearman']:+.4f}"
    )
emit()

emit("TWO-WAY-CENTERED RESIDUAL CORRELATION")
emit("-" * 92)
for r in residual_rows:
    emit(
        f"  {r['pd_metric']:5s}: "
        f"Pearson={r['pearson']:+.4f} "
        f"Spearman={r['spearman']:+.4f} "
        f"margin={r['max_centering_margin']:.3e}"
    )
emit()

emit("ALL-THREE CORRECTED-PD CONSENSUS vs CNN")
emit("-" * 92)
for mid in ["gan", "uv", "candidate_c", "uv_e2", "b_e2", "c_e2",
            "f1_grad_e2", "f2_grad_levelset_e2", "f3_grad_crit"]:
    rr = next((x for x in consensus_rows if x["method_id"] == mid), None)
    if rr is None:
        continue
    emit(
        f"  {mid:24s} "
        f"PD+ MT+={rr['all3_pd_improve__mt_improve']:3d}  "
        f"PD+ MT-={rr['all3_pd_improve__mt_worsen']:3d}  "
        f"PD- MT+={rr['all3_pd_worsen__mt_improve']:3d}  "
        f"PD- MT-={rr['all3_pd_worsen__mt_worsen']:3d}  "
        f"mixedPD={rr['mixed_pd_cases']:3d}"
    )
emit()

emit("CORRECTED PD / MT METHOD-MEAN PARETO FRONTS")
emit("-" * 92)
for key in ["dB_MT", "W2inf_MT", "W22_MT", "all3PD_MT"]:
    emit(f"  {key}: {', '.join(sorted(pareto_sets[key]))}")
emit()

# Top discordant samples for F1 vs CNN.
f1_arch = [
    r for r in archetype_rows
    if r["method_id"] == "f1_grad_e2"
]

pd_up_mt_down = sorted(
    [
        r for r in f1_arch
        if r["archetype"] == "all3_improve__mt_worsen"
    ],
    key=lambda r: float(r["discordance_score"]),
    reverse=True,
)

pd_down_mt_up = sorted(
    [
        r for r in f1_arch
        if r["archetype"] == "all3_worsen__mt_improve"
    ],
    key=lambda r: float(r["discordance_score"]),
    reverse=True,
)

emit("F1 (grad+E2) vs CNN: strongest descriptor-discordant samples")
emit("-" * 92)
emit("  all three corrected PD metrics improve, MT worsens:")
if pd_up_mt_down:
    for r in pd_up_mt_down[:10]:
        emit(
            f"    sample={int(r['sample_idx']):3d} "
            f"dB_gain={100*float(r['dB_gain_vs_cnn']):+7.2f}% "
            f"W2inf_gain={100*float(r['W2inf_gain_vs_cnn']):+7.2f}% "
            f"W22_gain={100*float(r['W22_gain_vs_cnn']):+7.2f}% "
            f"MT_gain={100*float(r['MT_gain_vs_cnn']):+7.2f}%"
        )
else:
    emit("    NONE")

emit("  all three corrected PD metrics worsen, MT improves:")
if pd_down_mt_up:
    for r in pd_down_mt_up[:10]:
        emit(
            f"    sample={int(r['sample_idx']):3d} "
            f"dB_gain={100*float(r['dB_gain_vs_cnn']):+7.2f}% "
            f"W2inf_gain={100*float(r['W2inf_gain_vs_cnn']):+7.2f}% "
            f"W22_gain={100*float(r['W22_gain_vs_cnn']):+7.2f}% "
            f"MT_gain={100*float(r['MT_gain_vs_cnn']):+7.2f}%"
        )
else:
    emit("    NONE")

emit()
emit("OUTPUTS")
emit("-" * 92)
for p in [
    JOINED_CSV,
    METHOD_MEANS_CSV,
    METHOD_CORR_CSV,
    WITHIN_METHOD_CSV,
    WITHIN_METHOD_SUMMARY_CSV,
    SAMPLEWISE_CSV,
    SAMPLEWISE_SUMMARY_CSV,
    RESIDUAL_CSV,
    QUADRANTS_CSV,
    CONSENSUS_CSV,
    PARETO_CSV,
    ARCHETYPES_CSV,
    FOCAL_CSV,
]:
    emit(f"  {p}")

SUMMARY_TXT.write_text("\n".join(lines) + "\n")
print()
print("Summary:", SUMMARY_TXT)
print()
print("CORRECTED PD / AUDITED MT ANALYSIS: COMPLETE")
