#!/usr/bin/env python3

from pathlib import Path
import argparse
import csv
import math
import re
import sys
from time import perf_counter

AUDIT = (
    Path.home()
    / "phire_runtime_audit_20260809_221548"
)

ROOT = Path.home() / "PhIRE"
BASE = ROOT / "ttk_runs_fixed"

MANIFEST = (
    AUDIT
    / "manifests"
    / "pd_result_run_manifest_v2.csv"
)

sys.path.insert(
    0,
    str(AUDIT / "recompute_pd")
)

import canonical_pd_pilot as pdm


SAMPLE_RE = re.compile(r"_s(\d+)_")


def sample_map(files):

    out = {}

    for p in files:

        m = SAMPLE_RE.search(p.name)

        if not m:
            raise RuntimeError(
                f"Cannot parse sample ID: {p}"
            )

        s = int(m.group(1))

        if s in out:
            raise RuntimeError(
                f"Duplicate sample {s}: "
                f"{out[s]} and {p}"
            )

        out[s] = p

    return out


def discover(run_root):

    pd_root = run_root / "pd"

    gt_dir = pd_root / "GT"
    sr_dir = pd_root / "SR"

    if gt_dir.exists() or sr_dir.exists():

        gt_files = sorted(
            gt_dir.glob("*_pd_port_0.vtu")
        )

        sr_files = sorted(
            sr_dir.glob("*_pd_port_0.vtu")
        )

    else:

        gt_files = sorted(
            p
            for p in pd_root.glob(
                "*_pd_port_0.vtu"
            )
            if "_GT_" in p.name
        )

        sr_files = sorted(
            p
            for p in pd_root.glob(
                "*_pd_port_0.vtu"
            )
            if "_SR_" in p.name
        )

    return (
        sample_map(gt_files),
        sample_map(sr_files),
    )


def historical_map(run_root):

    p = (
        run_root
        / "phase_c_final"
        / "pd_pairwise_distances.csv"
    )

    out = {}

    with p.open(newline="") as f:

        for row in csv.DictReader(f):

            m = SAMPLE_RE.search(
                row["key"]
            )

            if not m:
                raise RuntimeError(
                    f"Cannot parse historical key: "
                    f"{row['key']}"
                )

            s = int(m.group(1))

            out[s] = float(
                row["pd_distance"]
            )

    return out


def read_completed(path):

    if not path.exists():
        return set()

    done = set()

    with path.open(newline="") as f:

        for row in csv.DictReader(f):

            done.add(
                (
                    row["run"],
                    int(row["sample"]),
                )
            )

    return done


def compute_one(
    run,
    sample,
    gt_path,
    sr_path,
    historical,
):

    start = perf_counter()

    GT, gt_global = pdm.read_pd(
        gt_path
    )

    SR, sr_global = pdm.read_pd(
        sr_path
    )

    db0 = pdm.bottleneck(
        GT[0],
        SR[0],
    )

    w20 = pdm.w2(
        GT[0],
        SR[0],
    )

    db1 = pdm.bottleneck(
        GT[1],
        SR[1],
    )

    w21 = pdm.w2(
        GT[1],
        SR[1],
    )

    db_all = max(
        db0,
        db1,
    )

    w2_all = math.hypot(
        w20,
        w21,
    )

    # Fundamental metric sanity checks.
    if db0 > w20 + 1e-9:
        raise AssertionError(
            f"{run} s{sample}: "
            f"D0 dB > W2"
        )

    if db1 > w21 + 1e-9:
        raise AssertionError(
            f"{run} s{sample}: "
            f"D1 dB > W2"
        )

    if db_all > w2_all + 1e-9:
        raise AssertionError(
            f"{run} s{sample}: "
            f"combined dB > W2"
        )

    gt_type, gt_min, gt_max = (
        gt_global
    )

    sr_type, sr_min, sr_max = (
        sr_global
    )

    if gt_type != 0 or sr_type != 0:
        raise AssertionError(
            f"{run} s{sample}: "
            "unexpected nonfinite PairType"
        )

    delta_min = abs(
        gt_min - sr_min
    )

    delta_max = abs(
        gt_max - sr_max
    )

    endpoint_linf = max(
        delta_min,
        delta_max,
    )

    gt_span = gt_max - gt_min
    sr_span = sr_max - sr_min

    span_abs_error = abs(
        gt_span - sr_span
    )

    elapsed = (
        perf_counter() - start
    )

    return {
        "run": run,
        "sample": sample,

        "gt_path":
            str(gt_path.relative_to(ROOT)),
        "sr_path":
            str(sr_path.relative_to(ROOT)),

        "gt_d0_count": len(GT[0]),
        "sr_d0_count": len(SR[0]),
        "gt_d1_count": len(GT[1]),
        "sr_d1_count": len(SR[1]),

        "pd_bottleneck_d0": db0,
        "pd_bottleneck_d1": db1,
        "pd_bottleneck_all": db_all,

        "pd_w2_d0": w20,
        "pd_w2_d1": w21,
        "pd_w2_all": w2_all,

        "gt_global_min": gt_min,
        "gt_global_max": gt_max,
        "sr_global_min": sr_min,
        "sr_global_max": sr_max,

        "global_min_abs_error":
            delta_min,
        "global_max_abs_error":
            delta_max,
        "global_endpoint_linf":
            endpoint_linf,
        "global_span_abs_error":
            span_abs_error,

        # Historical value retained only for provenance.
        "historical_ttk_metric2":
            historical,

        "canonical_w2_minus_historical":
            w2_all - historical,

        "elapsed_seconds":
            elapsed,
    }


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--output",
        required=True,
    )

    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Optional number of comparisons "
            "for a smoke test."
        ),
    )

    args = ap.parse_args()

    out = Path(args.output)

    with MANIFEST.open(newline="") as f:
        manifest = list(
            csv.DictReader(f)
        )

    expected = []

    run_data = {}

    for row in manifest:

        run = row["run"]

        run_root = BASE / run

        gt, sr = discover(run_root)

        hist = historical_map(
            run_root
        )

        if set(gt) != set(sr):
            raise RuntimeError(
                f"{run}: GT/SR mismatch"
            )

        if set(gt) != set(hist):
            raise RuntimeError(
                f"{run}: PD/historical mismatch"
            )

        run_data[run] = (
            gt,
            sr,
            hist,
        )

        for sample in sorted(gt):
            expected.append(
                (run, sample)
            )

    if len(expected) != 8568:
        raise RuntimeError(
            f"Expected 8568 comparisons, "
            f"found {len(expected)}"
        )

    completed = read_completed(
        out
    )

    remaining = [
        x
        for x in expected
        if x not in completed
    ]

    if args.limit is not None:
        remaining = remaining[
            :args.limit
        ]

    print(
        "expected comparisons:",
        len(expected),
    )

    print(
        "already completed:",
        len(completed),
    )

    print(
        "to process this run:",
        len(remaining),
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
        "pd_bottleneck_d0",
        "pd_bottleneck_d1",
        "pd_bottleneck_all",
        "pd_w2_d0",
        "pd_w2_d1",
        "pd_w2_all",
        "gt_global_min",
        "gt_global_max",
        "sr_global_min",
        "sr_global_max",
        "global_min_abs_error",
        "global_max_abs_error",
        "global_endpoint_linf",
        "global_span_abs_error",
        "historical_ttk_metric2",
        "canonical_w2_minus_historical",
        "elapsed_seconds",
    ]

    write_header = (
        not out.exists()
        or out.stat().st_size == 0
    )

    out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    total_start = perf_counter()

    with out.open(
        "a",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        if write_header:
            writer.writeheader()
            f.flush()

        for i, (
            run,
            sample,
        ) in enumerate(
            remaining,
            start=1,
        ):

            gt, sr, hist = (
                run_data[run]
            )

            result = compute_one(
                run,
                sample,
                gt[sample],
                sr[sample],
                hist[sample],
            )

            writer.writerow(
                result
            )

            # Preserve progress if interrupted.
            f.flush()

            print(
                f"[{i}/{len(remaining)}] "
                f"{run} s{sample}: "
                f"dB={result['pd_bottleneck_all']:.6f} "
                f"W2={result['pd_w2_all']:.6f} "
                f"time={result['elapsed_seconds']:.3f}s"
            )

    elapsed = (
        perf_counter()
        - total_start
    )

    final_done = read_completed(
        out
    )

    print()
    print("=" * 100)
    print("CANONICAL PD SWEEP STATUS")
    print("=" * 100)

    print(
        "rows now present:",
        len(final_done),
    )

    print(
        "expected:",
        len(expected),
    )

    print(
        "this-run elapsed seconds:",
        elapsed,
    )

    print(
        "output:",
        out,
    )

    if (
        args.limit is None
        and len(final_done) != len(expected)
    ):
        raise SystemExit(
            "FAIL: full sweep incomplete"
        )

    if args.limit is None:
        print()
        print(
            "ALL 8568 CANONICAL PD "
            "COMPARISONS COMPLETE"
        )


if __name__ == "__main__":
    main()
