#!/usr/bin/env python3

from pathlib import Path
from collections import Counter
import csv
import re

import vtk


ROOT = Path.home() / "PhIRE"
BASE = ROOT / "ttk_runs_fixed"

AUDIT = (
    Path.home()
    / "phire_runtime_audit_20260809_221548"
)

MANIFEST = (
    AUDIT
    / "manifests"
    / "pd_result_run_manifest_v2.csv"
)

OUT = (
    AUDIT
    / "recompute_pd"
    / "pd_schema_all_files.csv"
)

SAMPLE_RE = re.compile(r"_s(\d+)_")

TOL = 1e-5


def read_vtu(path):
    r = vtk.vtkXMLUnstructuredGridReader()
    r.SetFileName(str(path))
    r.Update()

    g = r.GetOutput()

    if g is None:
        raise RuntimeError(
            f"VTK reader returned None: {path}"
        )

    return g


def sample_map(files):
    out = {}

    for p in files:
        m = SAMPLE_RE.search(p.name)

        if not m:
            raise RuntimeError(
                f"Cannot parse sample from {p}"
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

        gt = sorted(
            gt_dir.glob("*_pd_port_0.vtu")
        )

        sr = sorted(
            sr_dir.glob("*_pd_port_0.vtu")
        )

    else:

        gt = sorted(
            p for p in pd_root.glob(
                "*_pd_port_0.vtu"
            )
            if "_GT_" in p.name
        )

        sr = sorted(
            p for p in pd_root.glob(
                "*_pd_port_0.vtu"
            )
            if "_SR_" in p.name
        )

    return sample_map(gt), sample_map(sr)


def inspect(path):

    g = read_vtu(path)

    cd = g.GetCellData()
    pd = g.GetPointData()

    required_cell = [
        "PairIdentifier",
        "PairType",
        "Persistence",
        "Birth",
        "IsFinite",
    ]

    required_point = [
        "CriticalType",
    ]

    anomalies = []

    for name in required_cell:
        if cd.GetArray(name) is None:
            anomalies.append(
                f"missing_cell_array:{name}"
            )

    for name in required_point:
        if pd.GetArray(name) is None:
            anomalies.append(
                f"missing_point_array:{name}"
            )

    if anomalies:
        return {
            "cells": g.GetNumberOfCells(),
            "real_pairs": -1,
            "finite_type0": -1,
            "finite_type1": -1,
            "nonfinite_real": -1,
            "display_diagonal": -1,
            "max_embedding_error": "",
            "status": "CHECK",
            "notes": ";".join(anomalies),
        }

    pair_id = cd.GetArray("PairIdentifier")
    pair_type = cd.GetArray("PairType")
    persistence = cd.GetArray("Persistence")
    birth = cd.GetArray("Birth")
    finite = cd.GetArray("IsFinite")
    crit = pd.GetArray("CriticalType")

    counts = Counter()

    diagonal_count = 0
    nonfinite_count = 0

    max_err = 0.0

    for ci in range(g.GetNumberOfCells()):

        pid = int(pair_id.GetTuple1(ci))
        ptype = int(pair_type.GetTuple1(ci))
        fin = int(finite.GetTuple1(ci))

        cell = g.GetCell(ci)

        if cell.GetNumberOfPoints() != 2:
            anomalies.append(
                f"cell{ci}:not_two_points"
            )
            continue

        id0 = cell.GetPointId(0)
        id1 = cell.GetPointId(1)

        ct0 = int(crit.GetTuple1(id0))
        ct1 = int(crit.GetTuple1(id1))

        # --------------------------------------------------
        # Synthetic display diagonal
        # --------------------------------------------------
        if pid == -1:

            diagonal_count += 1

            if ptype != -1:
                anomalies.append(
                    f"cell{ci}:"
                    f"diagonal_pairtype={ptype}"
                )

            if fin != 0:
                anomalies.append(
                    f"cell{ci}:"
                    f"diagonal_isfinite={fin}"
                )

            continue

        # --------------------------------------------------
        # Real persistence pair
        # --------------------------------------------------
        b = float(birth.GetTuple1(ci))
        pers = float(
            persistence.GetTuple1(ci)
        )
        d = b + pers

        p0 = g.GetPoint(id0)
        p1 = g.GetPoint(id1)

        err = max(
            abs(float(p0[0]) - b),
            abs(float(p0[1]) - b),
            abs(float(p1[0]) - b),
            abs(float(p1[1]) - d),
        )

        max_err = max(max_err, err)

        if fin == 1:

            if ptype == 0:

                counts["finite_type0"] += 1

                if (ct0, ct1) != (0, 1):
                    anomalies.append(
                        f"cell{ci}:"
                        f"type0_endpoints="
                        f"{ct0},{ct1}"
                    )

            elif ptype == 1:

                counts["finite_type1"] += 1

                if (ct0, ct1) != (2, 3):
                    anomalies.append(
                        f"cell{ci}:"
                        f"type1_endpoints="
                        f"{ct0},{ct1}"
                    )

            else:

                anomalies.append(
                    f"cell{ci}:"
                    f"unexpected_finite_type="
                    f"{ptype}"
                )

        else:

            nonfinite_count += 1

            if not (
                ptype == 0
                and (ct0, ct1) == (0, 3)
            ):
                anomalies.append(
                    f"cell{ci}:"
                    f"unexpected_nonfinite="
                    f"type{ptype}_"
                    f"{ct0},{ct1}"
                )

    if diagonal_count != 1:
        anomalies.append(
            f"display_diagonal_count="
            f"{diagonal_count}"
        )

    if nonfinite_count != 1:
        anomalies.append(
            f"real_nonfinite_count="
            f"{nonfinite_count}"
        )

    if max_err > TOL:
        anomalies.append(
            f"embedding_error="
            f"{max_err:.17g}"
        )

    real_pairs = (
        counts["finite_type0"]
        + counts["finite_type1"]
        + nonfinite_count
    )

    return {
        "cells": g.GetNumberOfCells(),
        "real_pairs": real_pairs,
        "finite_type0":
            counts["finite_type0"],
        "finite_type1":
            counts["finite_type1"],
        "nonfinite_real":
            nonfinite_count,
        "display_diagonal":
            diagonal_count,
        "max_embedding_error":
            max_err,
        "status":
            "OK" if not anomalies else "CHECK",
        "notes":
            ";".join(anomalies),
    }


with MANIFEST.open(newline="") as f:
    manifest = list(csv.DictReader(f))


rows = []
files_checked = 0

for ri, run_row in enumerate(
    manifest,
    start=1,
):

    run = run_row["run"]
    run_root = BASE / run

    gt, sr = discover(run_root)

    if set(gt) != set(sr):
        raise RuntimeError(
            f"{run}: GT/SR sample mismatch"
        )

    print(
        f"[{ri:02d}/{len(manifest)}] "
        f"{run}"
    )

    for sample in sorted(gt):

        for label, path in (
            ("GT", gt[sample]),
            ("SR", sr[sample]),
        ):

            x = inspect(path)

            files_checked += 1

            rows.append({
                "run": run,
                "sample": sample,
                "label": label,
                "path":
                    str(path.relative_to(ROOT)),
                **x,
            })

            if x["status"] != "OK":
                print(
                    "  CHECK",
                    sample,
                    label,
                    x["notes"],
                )


with OUT.open("w", newline="") as f:

    fieldnames = list(rows[0].keys())

    w = csv.DictWriter(
        f,
        fieldnames=fieldnames,
    )

    w.writeheader()
    w.writerows(rows)


bad = [
    r for r in rows
    if r["status"] != "OK"
]

max_err = max(
    float(r["max_embedding_error"])
    for r in rows
    if r["max_embedding_error"] != ""
)


print()
print("=" * 100)
print("ALL-PD SCHEMA AUDIT")
print("=" * 100)

print("runs:", len(manifest))
print("comparisons:", files_checked // 2)
print("file instances checked:", files_checked)

print(
    "OK:",
    len(rows) - len(bad)
)

print(
    "CHECK:",
    len(bad)
)

print(
    "maximum real-pair embedding error:",
    max_err
)

print("output:", OUT)

if bad:

    print()
    print("PROBLEMS")

    for r in bad[:100]:
        print(
            r["run"],
            r["sample"],
            r["label"],
            r["notes"],
        )

    raise SystemExit(
        "FAIL: one or more PD artifacts "
        "violated the audited schema"
    )


print()
print("ALL QUANTITATIVE PD SCHEMAS: VERIFIED")
