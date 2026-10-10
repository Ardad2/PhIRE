#!/usr/bin/env python3

from pathlib import Path
import argparse
import csv
import math
import os
import sys
import time
import traceback

import numpy as np
import gudhi

from gudhi.wasserstein import (
    wasserstein_distance
)


EXPECTED = 8568

ABS_TOL = 1e-10
REL_TOL = 1e-12


def close(a, b):

    return math.isclose(
        a,
        b,
        abs_tol=ABS_TOL,
        rel_tol=REL_TOL,
    )


def gudhi_w22(A, B):

    return float(
        wasserstein_distance(
            A,
            B,
            matching=False,
            order=2.0,
            internal_p=2.0,
            keep_essential_parts=False,
        )
    )


def read_rows(path):

    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    if len(rows) != EXPECTED:
        raise RuntimeError(
            f"Expected {EXPECTED} rows; "
            f"found {len(rows)}"
        )

    required = {
        "run",
        "sample",
        "gt_path",
        "sr_path",
        "w22_d0",
        "w22_d1",
        "w22_all",
    }

    missing = required.difference(
        rows[0].keys()
    )

    if missing:
        raise RuntimeError(
            "Missing columns: "
            + ", ".join(sorted(missing))
        )

    keys = [
        (
            r["run"],
            int(r["sample"]),
        )
        for r in rows
    ]

    if len(keys) != len(set(keys)):
        raise RuntimeError(
            "Duplicate (run,sample) keys"
        )

    return rows


def load_recorded(path):

    if (
        not path.exists()
        or path.stat().st_size == 0
    ):
        return set()

    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    return {
        (
            r["run"],
            int(r["sample"]),
        )
        for r in rows
        if r.get("run")
    }


def preflight(rows):

    print("=" * 80)
    print("GUDHI W22 PREFLIGHT")
    print("=" * 80)

    errors = 0

    for i, row in enumerate(
        rows,
        1
    ):

        gt = Path(
            row["gt_path"]
        )

        sr = Path(
            row["sr_path"]
        )

        if not gt.exists():

            print(
                "MISSING GT:",
                gt
            )

            errors += 1

        if not sr.exists():

            print(
                "MISSING SR:",
                sr
            )

            errors += 1

        if (
            i % 500 == 0
            or i == len(rows)
        ):

            print(
                f"preflight "
                f"{i}/{len(rows)} "
                f"errors={errors}"
            )

    print()

    print(
        "Rows:",
        len(rows)
    )

    print(
        "Errors:",
        errors
    )

    if errors:

        print(
            "PREFLIGHT RESULT: FAIL"
        )

        return 1

    print(
        "PREFLIGHT RESULT: PASS"
    )

    return 0


def summarize(
    output,
    summary,
):

    with output.open(newline="") as f:
        rows = list(csv.DictReader(f))

    keys = {
        (
            r["run"],
            int(r["sample"])
        )
        for r in rows
    }

    passes = [
        r for r in rows
        if r["status"] == "PASS"
    ]

    mismatch = [
        r for r in rows
        if r["status"] == "MISMATCH"
    ]

    errors = [
        r for r in rows
        if r["status"] == "ERROR"
    ]

    def values(col):

        return [
            float(r[col])
            for r in rows
            if (
                r.get(col)
                not in ("", None)
            )
        ]

    d0 = values(
        "abs_diff_w22_d0"
    )

    d1 = values(
        "abs_diff_w22_d1"
    )

    da = values(
        "abs_diff_w22_all"
    )

    overall = (
        len(rows) == EXPECTED
        and len(keys) == EXPECTED
        and len(passes) == EXPECTED
        and not mismatch
        and not errors
    )

    lines = [
        "GUDHI W22 FULL-SWEEP CROSS-CHECK",
        "=" * 80,
        f"expected comparisons:  {EXPECTED}",
        f"rows:                  {len(rows)}",
        f"unique keys:           {len(keys)}",
        f"PASS:                  {len(passes)}",
        f"MISMATCH:              {len(mismatch)}",
        f"ERROR:                 {len(errors)}",
        "",
        f"max |delta W22_D0|:    {max(d0) if d0 else math.nan:.17g}",
        f"max |delta W22_D1|:    {max(d1) if d1 else math.nan:.17g}",
        f"max |delta W22_all|:   {max(da) if da else math.nan:.17g}",
        "",
        f"abs tolerance:         {ABS_TOL}",
        f"rel tolerance:         {REL_TOL}",
        "",
        "OVERALL: "
        + (
            "PASS"
            if overall
            else "FAIL"
        ),
    ]

    summary.write_text(
        "\n".join(lines)
        + "\n"
    )

    print()
    print(
        "\n".join(lines)
    )

    return (
        0
        if overall
        else 1
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
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
        "--preflight",
        action="store_true",
    )

    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
    )

    args = parser.parse_args()

    # --------------------------------------------------------------
    # Import the same frozen audited VTU parser
    # --------------------------------------------------------------

    AUDIT = Path(
        os.environ["AUDIT"]
    )

    sys.path.insert(
        0,
        str(
            AUDIT
            / "recompute_pd"
        )
    )

    import canonical_pd_pilot as canonical

    rows = read_rows(
        args.input
    )

    if args.preflight:
        return preflight(
            rows
        )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    recorded = load_recorded(
        args.output
    )

    fields = [
        "run",
        "sample",

        "custom_w22_d0",
        "gudhi_w22_d0",
        "abs_diff_w22_d0",
        "match_d0",

        "custom_w22_d1",
        "gudhi_w22_d1",
        "abs_diff_w22_d1",
        "match_d1",

        "custom_w22_all",
        "gudhi_w22_all",
        "abs_diff_w22_all",
        "match_all",

        "status",
        "error",
        "seconds",
    ]

    exists = (
        args.output.exists()
        and args.output.stat().st_size > 0
    )

    mode = (
        "a"
        if exists
        else "w"
    )

    attempted = 0
    passed = 0
    mismatched = 0
    errored = 0

    start_all = time.time()

    print("=" * 80)
    print(
        "GUDHI W22 FULL-SWEEP CROSS-CHECK"
    )
    print("=" * 80)

    print(
        "GUDHI:",
        gudhi.__version__
    )

    print(
        "Rows:",
        len(rows)
    )

    print(
        "Already recorded:",
        len(recorded)
    )

    print(
        "Definition:"
    )

    print(
        "  order=2"
    )

    print(
        "  internal_p=2"
    )

    print(
        "  finite points only"
    )

    with args.output.open(
        mode,
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        if not exists:

            writer.writeheader()
            f.flush()

        for index, row in enumerate(
            rows,
            1
        ):

            key = (
                row["run"],
                int(row["sample"]),
            )

            if key in recorded:
                continue

            attempted += 1

            t0 = time.time()

            out = {
                c: ""
                for c in fields
            }

            out["run"] = row["run"]
            out["sample"] = row["sample"]

            try:

                GT, _ = canonical.read_pd(
                    Path(
                        row["gt_path"]
                    )
                )

                SR, _ = canonical.read_pd(
                    Path(
                        row["sr_path"]
                    )
                )

                custom0 = float(
                    row["w22_d0"]
                )

                custom1 = float(
                    row["w22_d1"]
                )

                custom_all = float(
                    row["w22_all"]
                )

                g0 = gudhi_w22(
                    GT[0],
                    SR[0]
                )

                g1 = gudhi_w22(
                    GT[1],
                    SR[1]
                )

                gall = math.hypot(
                    g0,
                    g1
                )

                diff0 = abs(
                    custom0 - g0
                )

                diff1 = abs(
                    custom1 - g1
                )

                diffa = abs(
                    custom_all - gall
                )

                ok0 = close(
                    custom0,
                    g0
                )

                ok1 = close(
                    custom1,
                    g1
                )

                oka = close(
                    custom_all,
                    gall
                )

                out.update({
                    "custom_w22_d0":
                        f"{custom0:.17g}",
                    "gudhi_w22_d0":
                        f"{g0:.17g}",
                    "abs_diff_w22_d0":
                        f"{diff0:.17g}",
                    "match_d0":
                        int(ok0),

                    "custom_w22_d1":
                        f"{custom1:.17g}",
                    "gudhi_w22_d1":
                        f"{g1:.17g}",
                    "abs_diff_w22_d1":
                        f"{diff1:.17g}",
                    "match_d1":
                        int(ok1),

                    "custom_w22_all":
                        f"{custom_all:.17g}",
                    "gudhi_w22_all":
                        f"{gall:.17g}",
                    "abs_diff_w22_all":
                        f"{diffa:.17g}",
                    "match_all":
                        int(oka),
                })

                if (
                    ok0
                    and ok1
                    and oka
                ):

                    out["status"] = "PASS"

                    passed += 1

                else:

                    out["status"] = (
                        "MISMATCH"
                    )

                    mismatched += 1

                    print(
                        "MISMATCH",
                        key,
                        diff0,
                        diff1,
                        diffa,
                    )

            except Exception as exc:

                out["status"] = "ERROR"

                out["error"] = repr(
                    exc
                )

                errored += 1

                traceback.print_exc()

            out["seconds"] = (
                f"{time.time()-t0:.6f}"
            )

            writer.writerow(out)
            f.flush()

            if (
                attempted
                % args.progress_every
                == 0
                or index == len(rows)
                or out["status"]
                != "PASS"
            ):

                print(
                    f"progress "
                    f"index={index}/"
                    f"{len(rows)} "
                    f"attempted={attempted} "
                    f"pass={passed} "
                    f"mismatch={mismatched} "
                    f"error={errored} "
                    f"elapsed="
                    f"{time.time()-start_all:.1f}s",
                    flush=True,
                )

    return summarize(
        args.output,
        args.summary,
    )


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
