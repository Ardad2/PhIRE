#!/usr/bin/env python3

from pathlib import Path
import argparse
import csv
import math
import sys
import time
import traceback

import numpy as np
import gudhi
from gudhi.wasserstein import wasserstein_distance


ABS_TOL = 1e-10
REL_TOL = 1e-12
EXPECTED_COMPARISONS = 8568


def close(a, b):
    return math.isclose(a, b, rel_tol=REL_TOL, abs_tol=ABS_TOL)


def gudhi_db(A, B):
    return float(gudhi.bottleneck_distance(A, B, e=0.0))


def gudhi_w2(A, B):
    return float(
        wasserstein_distance(
            A,
            B,
            matching=False,
            order=2.0,
            internal_p=np.inf,
            keep_essential_parts=False,
        )
    )


def _prefer_pd_candidates(candidates, kind):
    """Resolve a unique PD VTU conservatively without guessing across run roots."""
    candidates = sorted(set(Path(p).resolve() for p in candidates))
    if len(candidates) == 1:
        return candidates[0]

    # Newer candidate topology outputs commonly separate pd/GT and pd/SR.
    preferred = [
        p for p in candidates
        if "pd" in p.parts and kind in p.parts
    ]
    if len(preferred) == 1:
        return preferred[0]

    # Historical CNN/GAN layouts commonly place both files directly under pd/.
    flat = [p for p in candidates if p.parent.name == "pd"]
    if len(flat) == 1:
        return flat[0]

    pretty = "\n    ".join(str(p) for p in candidates)
    raise RuntimeError(
        f"Expected exactly one {kind} PD file after conservative filtering; "
        f"found {len(candidates)} candidates:\n    {pretty}"
    )


def resolve_pd_paths(root, run, sample):
    run_root = root / "ttk_runs_fixed" / run
    if not run_root.is_dir():
        raise FileNotFoundError(f"Run root does not exist: {run_root}")

    gt_pattern = f"*_GT_s{sample}_speed_p160_x0_y0_pd_port_0.vtu"
    sr_pattern = f"*_SR_s{sample}_speed_p160_x0_y0_pd_port_0.vtu"

    gt_candidates = list(run_root.rglob(gt_pattern))
    sr_candidates = list(run_root.rglob(sr_pattern))

    if not gt_candidates:
        raise FileNotFoundError(
            f"No GT PD found under {run_root} for sample {sample} "
            f"with pattern {gt_pattern}"
        )
    if not sr_candidates:
        raise FileNotFoundError(
            f"No SR PD found under {run_root} for sample {sample} "
            f"with pattern {sr_pattern}"
        )

    gt = _prefer_pd_candidates(gt_candidates, "GT")
    sr = _prefer_pd_candidates(sr_candidates, "SR")
    return gt, sr


def read_canonical_rows(path):
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise RuntimeError(f"Canonical CSV is empty: {path}")

    required = {"run", "sample", "pd_bottleneck_all", "pd_w2_all"}
    missing = required.difference(rows[0].keys())
    if missing:
        raise RuntimeError(
            "Canonical CSV is missing required columns: "
            + ", ".join(sorted(missing))
            + "\nObserved columns: "
            + ", ".join(rows[0].keys())
        )

    keys = [(r["run"], int(r["sample"])) for r in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError("Canonical CSV contains duplicate (run, sample) rows")

    return rows


def preflight(rows, root):
    print("=" * 80)
    print("PREFLIGHT — RESOLVE ALL CANONICAL GT/SR PD PATHS")
    print("=" * 80)
    print(f"Rows: {len(rows)}")

    errors = []
    resolved = 0
    start = time.time()

    for i, row in enumerate(rows, 1):
        run = row["run"]
        sample = int(row["sample"])
        try:
            resolve_pd_paths(root, run, sample)
            resolved += 1
        except Exception as exc:
            errors.append((run, sample, repr(exc)))
            print(f"ERROR run={run} sample={sample}: {exc}")

        if i % 250 == 0 or i == len(rows):
            print(f"preflight {i}/{len(rows)} resolved={resolved} errors={len(errors)}")

    print()
    print(f"Resolved: {resolved}")
    print(f"Errors:   {len(errors)}")
    print(f"Elapsed:  {time.time() - start:.2f} s")

    if len(rows) != EXPECTED_COMPARISONS:
        print(
            f"ERROR: expected {EXPECTED_COMPARISONS} canonical rows, "
            f"found {len(rows)}"
        )
        return 1

    if errors:
        print("PREFLIGHT RESULT: FAIL")
        return 1

    print("PREFLIGHT RESULT: PASS")
    return 0


def load_recorded(output_csv):
    if not output_csv.exists() or output_csv.stat().st_size == 0:
        return set()

    with output_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        return {
            (row["run"], int(row["sample"]))
            for row in reader
            if row.get("run") and row.get("sample") not in (None, "")
        }


def summarize(output_csv, summary_path, expected_rows):
    with output_csv.open(newline="") as f:
        rows = list(csv.DictReader(f))

    unique = {(r["run"], int(r["sample"])) for r in rows}
    passes = [r for r in rows if r["status"] == "PASS"]
    mismatches = [r for r in rows if r["status"] == "MISMATCH"]
    errors = [r for r in rows if r["status"] == "ERROR"]

    def finite_float_values(column):
        vals = []
        for r in rows:
            text = r.get(column, "")
            if not text:
                continue
            try:
                v = float(text)
            except ValueError:
                continue
            if math.isfinite(v):
                vals.append(v)
        return vals

    db_diffs = finite_float_values("abs_diff_bottleneck_all")
    w2_diffs = finite_float_values("abs_diff_w2_all")

    max_db = max(db_diffs) if db_diffs else math.nan
    max_w2 = max(w2_diffs) if w2_diffs else math.nan

    complete = (
        len(rows) == expected_rows
        and len(unique) == expected_rows
        and not mismatches
        and not errors
    )

    lines = [
        "GUDHI FULL-SWEEP CROSS-CHECK SUMMARY",
        "=" * 80,
        f"expected comparisons: {expected_rows}",
        f"rows in GUDHI CSV:     {len(rows)}",
        f"unique (run,sample):   {len(unique)}",
        f"PASS rows:             {len(passes)}",
        f"MISMATCH rows:         {len(mismatches)}",
        f"ERROR rows:            {len(errors)}",
        f"max |delta dB_all|:    {max_db:.17g}",
        f"max |delta W2_all|:    {max_w2:.17g}",
        f"abs tolerance:         {ABS_TOL}",
        f"rel tolerance:         {REL_TOL}",
        "",
        "OVERALL: " + ("PASS" if complete else "FAIL"),
    ]

    summary_path.write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    return 0 if complete else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--canonical-csv",
        type=Path,
        required=True,
        help="Frozen canonical_pd_full_sweep.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Resume-safe GUDHI cross-check CSV",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        required=True,
        help="Summary text output",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.home() / "PhIRE",
        help="PhIRE repository root",
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Resolve all 8,568 GT/SR PD paths but do not compute distances",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
    )
    args = parser.parse_args()

    audit_dir = args.canonical_csv.resolve().parent
    sys.path.insert(0, str(audit_dir))
    import canonical_pd_pilot as canonical

    rows = read_canonical_rows(args.canonical_csv)

    if args.preflight:
        return preflight(rows, args.root.resolve())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)

    recorded = load_recorded(args.output)

    fieldnames = [
        "run",
        "sample",
        "gt_pd_path",
        "sr_pd_path",
        "gt_d0_count",
        "sr_d0_count",
        "gt_d1_count",
        "sr_d1_count",
        "gudhi_bottleneck_d0",
        "gudhi_bottleneck_d1",
        "gudhi_bottleneck_all",
        "canonical_bottleneck_all",
        "abs_diff_bottleneck_all",
        "gudhi_w2_d0",
        "gudhi_w2_d1",
        "gudhi_w2_all",
        "canonical_w2_all",
        "abs_diff_w2_all",
        "bottleneck_match",
        "w2_match",
        "status",
        "error",
        "seconds",
    ]

    file_exists = args.output.exists() and args.output.stat().st_size > 0
    mode = "a" if file_exists else "w"

    print("=" * 80)
    print("GUDHI FULL-SWEEP CROSS-CHECK")
    print("=" * 80)
    print("GUDHI:", gudhi.__version__)
    print("Canonical CSV:", args.canonical_csv)
    print("Output CSV:   ", args.output)
    print("Rows:         ", len(rows))
    print("Already recorded:", len(recorded))
    print("Exact dB:     gudhi.bottleneck_distance(..., e=0.0)")
    print("W2:           order=2, internal_p=inf, finite points only")
    print()

    start_all = time.time()
    attempted = 0
    new_pass = 0
    new_mismatch = 0
    new_error = 0

    with args.output.open(mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
            f.flush()

        for index, row in enumerate(rows, 1):
            run = row["run"]
            sample = int(row["sample"])
            key = (run, sample)

            if key in recorded:
                continue

            attempted += 1
            t0 = time.time()
            out = {name: "" for name in fieldnames}
            out["run"] = run
            out["sample"] = sample

            try:
                gt_path, sr_path = resolve_pd_paths(args.root.resolve(), run, sample)
                out["gt_pd_path"] = str(gt_path)
                out["sr_pd_path"] = str(sr_path)

                GT, _ = canonical.read_pd(gt_path)
                SR, _ = canonical.read_pd(sr_path)

                out["gt_d0_count"] = len(GT[0])
                out["sr_d0_count"] = len(SR[0])
                out["gt_d1_count"] = len(GT[1])
                out["sr_d1_count"] = len(SR[1])

                db0 = gudhi_db(GT[0], SR[0])
                db1 = gudhi_db(GT[1], SR[1])
                db_all = max(db0, db1)

                w20 = gudhi_w2(GT[0], SR[0])
                w21 = gudhi_w2(GT[1], SR[1])
                w2_all = math.hypot(w20, w21)

                canonical_db = float(row["pd_bottleneck_all"])
                canonical_w2 = float(row["pd_w2_all"])

                diff_db = abs(db_all - canonical_db)
                diff_w2 = abs(w2_all - canonical_w2)

                db_ok = close(db_all, canonical_db)
                w2_ok = close(w2_all, canonical_w2)

                out.update({
                    "gudhi_bottleneck_d0": f"{db0:.17g}",
                    "gudhi_bottleneck_d1": f"{db1:.17g}",
                    "gudhi_bottleneck_all": f"{db_all:.17g}",
                    "canonical_bottleneck_all": f"{canonical_db:.17g}",
                    "abs_diff_bottleneck_all": f"{diff_db:.17g}",
                    "gudhi_w2_d0": f"{w20:.17g}",
                    "gudhi_w2_d1": f"{w21:.17g}",
                    "gudhi_w2_all": f"{w2_all:.17g}",
                    "canonical_w2_all": f"{canonical_w2:.17g}",
                    "abs_diff_w2_all": f"{diff_w2:.17g}",
                    "bottleneck_match": int(db_ok),
                    "w2_match": int(w2_ok),
                    "status": "PASS" if db_ok and w2_ok else "MISMATCH",
                })

                if db_ok and w2_ok:
                    new_pass += 1
                else:
                    new_mismatch += 1
                    print(
                        f"MISMATCH run={run} sample={sample} "
                        f"dB diff={diff_db:.3e} W2 diff={diff_w2:.3e}"
                    )

            except Exception as exc:
                new_error += 1
                out["status"] = "ERROR"
                out["error"] = repr(exc)
                print(f"ERROR run={run} sample={sample}: {exc}")
                traceback.print_exc()

            out["seconds"] = f"{time.time() - t0:.6f}"
            writer.writerow(out)
            f.flush()

            if (
                attempted % args.progress_every == 0
                or index == len(rows)
                or out["status"] != "PASS"
            ):
                elapsed = time.time() - start_all
                print(
                    f"progress canonical_index={index}/{len(rows)} "
                    f"attempted={attempted} pass={new_pass} "
                    f"mismatch={new_mismatch} error={new_error} "
                    f"elapsed={elapsed:.1f}s",
                    flush=True,
                )

    return summarize(args.output, args.summary, EXPECTED_COMPARISONS)


if __name__ == "__main__":
    raise SystemExit(main())
