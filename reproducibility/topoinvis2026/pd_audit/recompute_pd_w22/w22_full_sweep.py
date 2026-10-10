#!/usr/bin/env python3

from pathlib import Path
import argparse
import csv
import math
import os
import sys
import time
import traceback
from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment


EXPECTED_COMPARISONS = 8568

ABS_TOL = 1e-10
REL_TOL = 1e-12

SQRT2 = math.sqrt(2.0)


# ==========================================================================
# Explicit standard W_{2,2}
#
# Wasserstein aggregation exponent q = 2
# birth/death-plane ground norm p = 2
#
# real-real:
#
#   sqrt((b1-b2)^2 + (d1-d2)^2)
#
# real-diagonal:
#
#   (death-birth)/sqrt(2)
#
# ==========================================================================

def pairwise_l2(A, B):

    if len(A) == 0 or len(B) == 0:
        return np.empty(
            (len(A), len(B)),
            dtype=np.float64,
        )

    db = (
        A[:, None, 0]
        - B[None, :, 0]
    )

    dd = (
        A[:, None, 1]
        - B[None, :, 1]
    )

    return np.hypot(
        db,
        dd,
    )


def diagonal_cost_l2(D):

    if len(D) == 0:
        return np.empty(
            0,
            dtype=np.float64,
        )

    persistence = (
        D[:, 1]
        - D[:, 0]
    )

    return (
        persistence
        / SQRT2
    )


def augmented_cost_l2(A, B):

    n = len(A)
    m = len(B)

    if n + m == 0:
        return np.empty(
            (0, 0),
            dtype=np.float64,
        )

    C = np.full(
        (n + m, n + m),
        np.inf,
        dtype=np.float64,
    )

    # --------------------------------------------------------------
    # Real A -> real B
    # --------------------------------------------------------------

    if n and m:
        C[:n, :m] = pairwise_l2(
            A,
            B,
        )

    # --------------------------------------------------------------
    # Real A -> own diagonal copy
    # --------------------------------------------------------------

    if n:

        i = np.arange(n)

        C[
            i,
            m + i
        ] = diagonal_cost_l2(A)

    # --------------------------------------------------------------
    # Corresponding diagonal copy -> real B
    # --------------------------------------------------------------

    if m:

        j = np.arange(m)

        C[
            n + j,
            j
        ] = diagonal_cost_l2(B)

    # --------------------------------------------------------------
    # Diagonal copy -> diagonal copy
    # --------------------------------------------------------------

    if n and m:
        C[
            n:,
            m:
        ] = 0.0

    return C


def w22(A, B):

    C = augmented_cost_l2(
        A,
        B,
    )

    if C.size == 0:
        return 0.0

    rows, cols = (
        linear_sum_assignment(
            C * C
        )
    )

    costs = C[
        rows,
        cols
    ]

    return float(
        np.sqrt(
            np.sum(
                costs * costs
            )
        )
    )


# ==========================================================================
# Helpers
# ==========================================================================

def close(a, b):

    return math.isclose(
        a,
        b,
        rel_tol=REL_TOL,
        abs_tol=ABS_TOL,
    )


def lower_bound_ok(w2inf, w22_value):

    return (
        w22_value >= w2inf
        or close(
            w22_value,
            w2inf,
        )
    )


def upper_bound_ok(w2inf, w22_value):

    upper = SQRT2 * w2inf

    return (
        w22_value <= upper
        or close(
            w22_value,
            upper,
        )
    )


def optional_float(value):

    if value is None:
        return math.nan

    value = str(value).strip()

    if not value:
        return math.nan

    try:
        return float(value)

    except ValueError:
        return math.nan


def read_manifest(path):

    with path.open(newline="") as f:
        rows = list(
            csv.DictReader(f)
        )

    if not rows:
        raise RuntimeError(
            f"Manifest is empty: {path}"
        )

    required = {
        "run",
        "sample",
        "gt_path",
        "sr_path",
        "gt_d0_count",
        "sr_d0_count",
        "gt_d1_count",
        "sr_d1_count",
        "pd_bottleneck_d0",
        "pd_bottleneck_d1",
        "pd_bottleneck_all",
        "pd_w2_d0",
        "pd_w2_d1",
        "pd_w2_all",
        "historical_ttk_metric2",
    }

    missing = required.difference(
        rows[0].keys()
    )

    if missing:

        raise RuntimeError(
            "Missing manifest columns: "
            + ", ".join(
                sorted(missing)
            )
        )

    keys = [
        (
            row["run"],
            int(row["sample"]),
        )
        for row in rows
    ]

    if len(keys) != len(set(keys)):

        raise RuntimeError(
            "Duplicate (run,sample) "
            "keys in canonical manifest"
        )

    if len(rows) != EXPECTED_COMPARISONS:

        raise RuntimeError(
            f"Expected {EXPECTED_COMPARISONS} "
            f"manifest rows; found {len(rows)}"
        )

    return rows


def load_recorded(path):

    if (
        not path.exists()
        or path.stat().st_size == 0
    ):
        return set()

    with path.open(newline="") as f:

        reader = csv.DictReader(f)

        return {
            (
                row["run"],
                int(row["sample"]),
            )
            for row in reader
            if (
                row.get("run")
                and row.get("sample")
                not in ("", None)
            )
        }


# ==========================================================================
# Final summaries
# ==========================================================================

def summarize(
    output_path,
    summary_path,
    run_summary_path,
):

    with output_path.open(newline="") as f:

        rows = list(
            csv.DictReader(f)
        )

    keys = [
        (
            r["run"],
            int(r["sample"]),
        )
        for r in rows
    ]

    unique_keys = set(keys)

    pass_rows = [
        r
        for r in rows
        if r["status"] == "PASS"
    ]

    invariant_rows = [
        r
        for r in rows
        if r["status"] == "INVARIANT_FAIL"
    ]

    error_rows = [
        r
        for r in rows
        if r["status"] == "ERROR"
    ]

    count_mismatch = [
        r
        for r in rows
        if r.get("count_match") == "0"
    ]

    lower_violations = []

    upper_violations = []

    ratios_all = []

    w22_minus_w2inf = []

    ttk_minus_w22 = []

    for r in rows:

        if r["status"] == "ERROR":
            continue

        w2inf = float(
            r["w2inf_all"]
        )

        w22_value = float(
            r["w22_all"]
        )

        lower_violations.append(
            max(
                0.0,
                w2inf - w22_value,
            )
        )

        upper_violations.append(
            max(
                0.0,
                w22_value
                - SQRT2 * w2inf,
            )
        )

        if w2inf != 0:
            ratios_all.append(
                w22_value / w2inf
            )

        w22_minus_w2inf.append(
            w22_value - w2inf
        )

        ttk = optional_float(
            r.get(
                "historical_ttk_metric2"
            )
        )

        if math.isfinite(ttk):

            ttk_minus_w22.append(
                ttk - w22_value
            )

    overall_ok = (
        len(rows)
        == EXPECTED_COMPARISONS
        and len(unique_keys)
        == EXPECTED_COMPARISONS
        and len(pass_rows)
        == EXPECTED_COMPARISONS
        and not invariant_rows
        and not error_rows
        and not count_mismatch
    )

    max_lower = (
        max(lower_violations)
        if lower_violations
        else math.nan
    )

    max_upper = (
        max(upper_violations)
        if upper_violations
        else math.nan
    )

    min_ratio = (
        min(ratios_all)
        if ratios_all
        else math.nan
    )

    max_ratio = (
        max(ratios_all)
        if ratios_all
        else math.nan
    )

    mean_ratio = (
        sum(ratios_all)
        / len(ratios_all)
        if ratios_all
        else math.nan
    )

    mean_delta_inf = (
        sum(w22_minus_w2inf)
        / len(w22_minus_w2inf)
        if w22_minus_w2inf
        else math.nan
    )

    mean_ttk_minus = (
        sum(ttk_minus_w22)
        / len(ttk_minus_w22)
        if ttk_minus_w22
        else math.nan
    )

    lines = [
        "STANDARD W_{2,2} FULL-SWEEP SUMMARY",
        "=" * 80,
        f"expected comparisons:          {EXPECTED_COMPARISONS}",
        f"rows in W22 CSV:               {len(rows)}",
        f"unique (run,sample):           {len(unique_keys)}",
        f"PASS rows:                     {len(pass_rows)}",
        f"INVARIANT_FAIL rows:           {len(invariant_rows)}",
        f"ERROR rows:                    {len(error_rows)}",
        f"count mismatches:              {len(count_mismatch)}",
        "",
        "Cross-norm invariant:",
        "    W2_inf <= W2_2 <= sqrt(2) * W2_inf",
        f"max lower-bound violation:     {max_lower:.17g}",
        f"max upper-bound violation:     {max_upper:.17g}",
        f"min W22/W2inf ratio:           {min_ratio:.17g}",
        f"max W22/W2inf ratio:           {max_ratio:.17g}",
        f"mean W22/W2inf ratio:          {mean_ratio:.17g}",
        f"mean W22-W2inf:                {mean_delta_inf:.17g}",
        "",
        f"mean historical_TTK2-W22:      {mean_ttk_minus:.17g}",
        "",
        f"abs tolerance:                 {ABS_TOL}",
        f"rel tolerance:                 {REL_TOL}",
        "",
        "OVERALL INTERNAL CHECK: "
        + (
            "PASS"
            if overall_ok
            else "FAIL"
        ),
        "",
        "NOTE:",
        "This validates internal consistency and the cross-norm inequality.",
        "Independent numerical validation against GUDHI is a separate next stage.",
    ]

    summary_path.write_text(
        "\n".join(lines)
        + "\n"
    )

    print()
    print(
        "\n".join(lines)
    )

    # ------------------------------------------------------------------
    # Per-run summary
    # ------------------------------------------------------------------

    by_run = defaultdict(list)

    for r in rows:

        if r["status"] == "PASS":
            by_run[
                r["run"]
            ].append(r)

    run_fields = [
        "run",
        "n",
        "mean_bottleneck_all",
        "mean_w2inf_all",
        "mean_w22_all",
        "mean_historical_ttk2",
        "mean_w22_minus_w2inf",
        "mean_w22_over_w2inf",
        "mean_historical_ttk2_minus_w22",
        "historical_ttk2_gt_w22",
        "historical_ttk2_lt_w22",
    ]

    run_rows = []

    for run in sorted(by_run):

        sub = by_run[run]

        db = np.asarray(
            [
                float(r["bottleneck_all"])
                for r in sub
            ],
            dtype=float,
        )

        wi = np.asarray(
            [
                float(r["w2inf_all"])
                for r in sub
            ],
            dtype=float,
        )

        w22_values = np.asarray(
            [
                float(r["w22_all"])
                for r in sub
            ],
            dtype=float,
        )

        ttk = np.asarray(
            [
                optional_float(
                    r[
                        "historical_ttk_metric2"
                    ]
                )
                for r in sub
            ],
            dtype=float,
        )

        finite_ttk = np.isfinite(
            ttk
        )

        ratios = np.divide(
            w22_values,
            wi,
            out=np.full_like(
                w22_values,
                np.nan,
            ),
            where=wi != 0,
        )

        if np.any(finite_ttk):

            mean_ttk = float(
                np.mean(
                    ttk[finite_ttk]
                )
            )

            mean_ttk_minus = float(
                np.mean(
                    ttk[finite_ttk]
                    - w22_values[finite_ttk]
                )
            )

            ttk_gt = int(
                np.sum(
                    ttk[finite_ttk]
                    > w22_values[finite_ttk]
                )
            )

            ttk_lt = int(
                np.sum(
                    ttk[finite_ttk]
                    < w22_values[finite_ttk]
                )
            )

        else:

            mean_ttk = math.nan
            mean_ttk_minus = math.nan
            ttk_gt = 0
            ttk_lt = 0

        run_rows.append({
            "run": run,
            "n": len(sub),
            "mean_bottleneck_all":
                f"{np.mean(db):.17g}",
            "mean_w2inf_all":
                f"{np.mean(wi):.17g}",
            "mean_w22_all":
                f"{np.mean(w22_values):.17g}",
            "mean_historical_ttk2":
                f"{mean_ttk:.17g}",
            "mean_w22_minus_w2inf":
                f"{np.mean(w22_values-wi):.17g}",
            "mean_w22_over_w2inf":
                f"{np.nanmean(ratios):.17g}",
            "mean_historical_ttk2_minus_w22":
                f"{mean_ttk_minus:.17g}",
            "historical_ttk2_gt_w22":
                ttk_gt,
            "historical_ttk2_lt_w22":
                ttk_lt,
        })

    with run_summary_path.open(
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=run_fields,
        )

        writer.writeheader()
        writer.writerows(
            run_rows
        )

    print()
    print(
        "Per-run summary:",
        run_summary_path
    )

    return (
        0
        if overall_ok
        else 1
    )


# ==========================================================================
# Main
# ==========================================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--canonical-csv",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--summary",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--run-summary",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
    )

    args = parser.parse_args()

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.summary.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------------
    # Import ONLY the frozen audited TTK-VTU parser.
    # ------------------------------------------------------------------

    canonical_dir = (
        args.canonical_csv
        .resolve()
        .parent
    )

    sys.path.insert(
        0,
        str(canonical_dir)
    )

    import canonical_pd_pilot as canonical

    rows = read_manifest(
        args.canonical_csv
    )

    recorded = load_recorded(
        args.output
    )

    print("=" * 80)
    print(
        "STANDARD W_{2,2} FULL SWEEP"
    )
    print("=" * 80)

    print(
        "Manifest:",
        args.canonical_csv
    )

    print(
        "Rows:",
        len(rows)
    )

    print(
        "Already recorded:",
        len(recorded)
    )

    print()
    print(
        "Definition:"
    )

    print(
        "  Wasserstein order q = 2"
    )

    print(
        "  ground norm p = 2"
    )

    print(
        "  diagonal cost = persistence/sqrt(2)"
    )

    print(
        "  finite D0 and D1 separately"
    )

    print(
        "  W22_all = hypot(W22_D0, W22_D1)"
    )

    print()
    print(
        "Invariant checked:"
    )

    print(
        "  W2_inf <= W2_2 <= sqrt(2)*W2_inf"
    )

    fieldnames = [
        "run",
        "sample",
        "gt_path",
        "sr_path",

        "gt_d0_count",
        "sr_d0_count",
        "gt_d1_count",
        "sr_d1_count",

        "frozen_gt_d0_count",
        "frozen_sr_d0_count",
        "frozen_gt_d1_count",
        "frozen_sr_d1_count",

        "count_match",

        "bottleneck_d0",
        "bottleneck_d1",
        "bottleneck_all",

        "w2inf_d0",
        "w22_d0",
        "w22_over_w2inf_d0",
        "w22_minus_w2inf_d0",
        "sqrt2_w2inf_minus_w22_d0",
        "lower_ok_d0",
        "upper_ok_d0",

        "w2inf_d1",
        "w22_d1",
        "w22_over_w2inf_d1",
        "w22_minus_w2inf_d1",
        "sqrt2_w2inf_minus_w22_d1",
        "lower_ok_d1",
        "upper_ok_d1",

        "w2inf_all",
        "w22_all",
        "w22_over_w2inf_all",
        "w22_minus_w2inf_all",
        "sqrt2_w2inf_minus_w22_all",
        "lower_ok_all",
        "upper_ok_all",

        "historical_ttk_metric2",
        "historical_ttk2_minus_w22",
        "w22_minus_historical_ttk2",
        "historical_ttk2_over_w22",

        "status",
        "error",
        "elapsed_seconds",
    ]

    file_exists = (
        args.output.exists()
        and args.output.stat().st_size > 0
    )

    mode = (
        "a"
        if file_exists
        else "w"
    )

    attempted = 0
    new_pass = 0
    new_invariant_fail = 0
    new_error = 0

    t_all = time.time()

    with args.output.open(
        mode,
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        if not file_exists:

            writer.writeheader()
            f.flush()

        for manifest_index, row in enumerate(
            rows,
            1,
        ):

            run = row["run"]

            sample = int(
                row["sample"]
            )

            key = (
                run,
                sample,
            )

            if key in recorded:
                continue

            attempted += 1

            t0 = time.time()

            out = {
                name: ""
                for name in fieldnames
            }

            out["run"] = run
            out["sample"] = sample

            try:

                gt_path = Path(
                    row["gt_path"]
                )

                sr_path = Path(
                    row["sr_path"]
                )

                if not gt_path.exists():
                    raise FileNotFoundError(
                        gt_path
                    )

                if not sr_path.exists():
                    raise FileNotFoundError(
                        sr_path
                    )

                out["gt_path"] = str(
                    gt_path
                )

                out["sr_path"] = str(
                    sr_path
                )

                GT, _ = canonical.read_pd(
                    gt_path
                )

                SR, _ = canonical.read_pd(
                    sr_path
                )

                # --------------------------------------------------
                # Cardinalities
                # --------------------------------------------------

                counts = {
                    "gt_d0_count":
                        len(GT[0]),
                    "sr_d0_count":
                        len(SR[0]),
                    "gt_d1_count":
                        len(GT[1]),
                    "sr_d1_count":
                        len(SR[1]),
                }

                for k, v in counts.items():
                    out[k] = v

                frozen_counts = {
                    "frozen_gt_d0_count":
                        int(
                            row[
                                "gt_d0_count"
                            ]
                        ),
                    "frozen_sr_d0_count":
                        int(
                            row[
                                "sr_d0_count"
                            ]
                        ),
                    "frozen_gt_d1_count":
                        int(
                            row[
                                "gt_d1_count"
                            ]
                        ),
                    "frozen_sr_d1_count":
                        int(
                            row[
                                "sr_d1_count"
                            ]
                        ),
                }

                for k, v in frozen_counts.items():
                    out[k] = v

                count_ok = (
                    counts["gt_d0_count"]
                    == frozen_counts[
                        "frozen_gt_d0_count"
                    ]
                    and counts["sr_d0_count"]
                    == frozen_counts[
                        "frozen_sr_d0_count"
                    ]
                    and counts["gt_d1_count"]
                    == frozen_counts[
                        "frozen_gt_d1_count"
                    ]
                    and counts["sr_d1_count"]
                    == frozen_counts[
                        "frozen_sr_d1_count"
                    ]
                )

                out["count_match"] = int(
                    count_ok
                )

                # --------------------------------------------------
                # Frozen metrics
                # --------------------------------------------------

                db0 = float(
                    row[
                        "pd_bottleneck_d0"
                    ]
                )

                db1 = float(
                    row[
                        "pd_bottleneck_d1"
                    ]
                )

                db_all = float(
                    row[
                        "pd_bottleneck_all"
                    ]
                )

                wi0 = float(
                    row[
                        "pd_w2_d0"
                    ]
                )

                wi1 = float(
                    row[
                        "pd_w2_d1"
                    ]
                )

                wi_all = float(
                    row[
                        "pd_w2_all"
                    ]
                )

                out["bottleneck_d0"] = (
                    f"{db0:.17g}"
                )

                out["bottleneck_d1"] = (
                    f"{db1:.17g}"
                )

                out["bottleneck_all"] = (
                    f"{db_all:.17g}"
                )

                # --------------------------------------------------
                # New W22
                # --------------------------------------------------

                w220 = w22(
                    GT[0],
                    SR[0],
                )

                w221 = w22(
                    GT[1],
                    SR[1],
                )

                w22_all = math.hypot(
                    w220,
                    w221,
                )

                # --------------------------------------------------
                # Dimension-wise invariant checks
                # --------------------------------------------------

                all_internal_ok = count_ok

                for dim, wi, wv in [
                    (0, wi0, w220),
                    (1, wi1, w221),
                ]:

                    lower_ok = lower_bound_ok(
                        wi,
                        wv,
                    )

                    upper_ok = upper_bound_ok(
                        wi,
                        wv,
                    )

                    ratio = (
                        wv / wi
                        if wi != 0
                        else math.nan
                    )

                    out[
                        f"w2inf_d{dim}"
                    ] = f"{wi:.17g}"

                    out[
                        f"w22_d{dim}"
                    ] = f"{wv:.17g}"

                    out[
                        f"w22_over_w2inf_d{dim}"
                    ] = f"{ratio:.17g}"

                    out[
                        f"w22_minus_w2inf_d{dim}"
                    ] = (
                        f"{wv-wi:.17g}"
                    )

                    out[
                        f"sqrt2_w2inf_minus_w22_d{dim}"
                    ] = (
                        f"{SQRT2*wi-wv:.17g}"
                    )

                    out[
                        f"lower_ok_d{dim}"
                    ] = int(
                        lower_ok
                    )

                    out[
                        f"upper_ok_d{dim}"
                    ] = int(
                        upper_ok
                    )

                    all_internal_ok &= (
                        lower_ok
                        and upper_ok
                    )

                # --------------------------------------------------
                # Aggregate invariant
                # --------------------------------------------------

                lower_all = lower_bound_ok(
                    wi_all,
                    w22_all,
                )

                upper_all = upper_bound_ok(
                    wi_all,
                    w22_all,
                )

                ratio_all = (
                    w22_all / wi_all
                    if wi_all != 0
                    else math.nan
                )

                out["w2inf_all"] = (
                    f"{wi_all:.17g}"
                )

                out["w22_all"] = (
                    f"{w22_all:.17g}"
                )

                out[
                    "w22_over_w2inf_all"
                ] = (
                    f"{ratio_all:.17g}"
                )

                out[
                    "w22_minus_w2inf_all"
                ] = (
                    f"{w22_all-wi_all:.17g}"
                )

                out[
                    "sqrt2_w2inf_minus_w22_all"
                ] = (
                    f"{SQRT2*wi_all-w22_all:.17g}"
                )

                out["lower_ok_all"] = int(
                    lower_all
                )

                out["upper_ok_all"] = int(
                    upper_all
                )

                all_internal_ok &= (
                    lower_all
                    and upper_all
                )

                # --------------------------------------------------
                # Historical TTK "2"
                # --------------------------------------------------

                ttk2 = optional_float(
                    row[
                        "historical_ttk_metric2"
                    ]
                )

                if math.isfinite(ttk2):

                    out[
                        "historical_ttk_metric2"
                    ] = (
                        f"{ttk2:.17g}"
                    )

                    out[
                        "historical_ttk2_minus_w22"
                    ] = (
                        f"{ttk2-w22_all:.17g}"
                    )

                    out[
                        "w22_minus_historical_ttk2"
                    ] = (
                        f"{w22_all-ttk2:.17g}"
                    )

                    if w22_all != 0:

                        out[
                            "historical_ttk2_over_w22"
                        ] = (
                            f"{ttk2/w22_all:.17g}"
                        )

                # --------------------------------------------------
                # Status
                # --------------------------------------------------

                if all_internal_ok:

                    out["status"] = "PASS"

                    new_pass += 1

                else:

                    out[
                        "status"
                    ] = "INVARIANT_FAIL"

                    new_invariant_fail += 1

                    print()
                    print(
                        "INVARIANT_FAIL",
                        f"run={run}",
                        f"sample={sample}",
                        f"W2inf={wi_all:.17g}",
                        f"W22={w22_all:.17g}",
                        flush=True,
                    )

            except Exception as exc:

                out["status"] = "ERROR"

                out["error"] = repr(
                    exc
                )

                new_error += 1

                print()
                print(
                    "ERROR",
                    f"run={run}",
                    f"sample={sample}",
                    repr(exc),
                    flush=True,
                )

                traceback.print_exc()

            out["elapsed_seconds"] = (
                f"{time.time()-t0:.6f}"
            )

            writer.writerow(out)
            f.flush()

            if (
                attempted
                % args.progress_every
                == 0
                or manifest_index
                == len(rows)
                or out["status"]
                != "PASS"
            ):

                elapsed = (
                    time.time()
                    - t_all
                )

                print(
                    f"progress "
                    f"manifest_index="
                    f"{manifest_index}/"
                    f"{len(rows)} "
                    f"attempted={attempted} "
                    f"pass={new_pass} "
                    f"invariant_fail="
                    f"{new_invariant_fail} "
                    f"error={new_error} "
                    f"elapsed={elapsed:.1f}s",
                    flush=True,
                )

    return summarize(
        args.output,
        args.summary,
        args.run_summary,
    )


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
