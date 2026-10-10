#!/usr/bin/env python3

from pathlib import Path
import argparse
import csv
import math
import re

import vtk
import topologytoolkit as ttk


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

SAMPLE_RE = re.compile(r"_s(\d+)_")


# ----------------------------------------------------------------------
# VTK / TTK helpers
# ----------------------------------------------------------------------

def read_dataset(path: Path):
    if path.suffix == ".vtu":
        r = vtk.vtkXMLUnstructuredGridReader()
    elif path.suffix == ".vtp":
        r = vtk.vtkXMLPolyDataReader()
    elif path.suffix == ".vti":
        r = vtk.vtkXMLImageDataReader()
    else:
        raise RuntimeError(f"Unsupported file type: {path}")

    r.SetFileName(str(path))
    r.Update()

    out = r.GetOutputDataObject(0)

    if out is None:
        raise RuntimeError(f"VTK reader returned None: {path}")

    return out


def get_field_distance(ds):
    if ds is None or not hasattr(ds, "GetFieldData"):
        return None

    fd = ds.GetFieldData()

    if fd is None:
        return None

    for name in (
        "BottleneckDistance",
        "WassersteinDistance",
        "Distance",
    ):
        arr = fd.GetArray(name)

        if arr is not None and arr.GetNumberOfTuples() > 0:
            return float(arr.GetTuple1(0))

    return None


def compute_pd_distance(pd_gt: Path, pd_sr: Path, metric: str):
    """
    metric:
        "2"   -> explicit 2-Wasserstein
        "inf" -> explicit bottleneck
    """

    gt = read_dataset(pd_gt)
    sr = read_dataset(pd_sr)

    mb = vtk.vtkMultiBlockDataSet()
    mb.SetNumberOfBlocks(2)
    mb.SetBlock(0, gt)
    mb.SetBlock(1, sr)

    bd = ttk.ttkBottleneckDistance()

    # Preserve historical non-metric configuration.
    if hasattr(bd, "SetPVAlgorithm"):
        bd.SetPVAlgorithm(0)

    if hasattr(bd, "SetDistanceAlgorithm"):
        bd.SetDistanceAlgorithm("ttk")

    if hasattr(bd, "SetUseOutputMatching"):
        bd.SetUseOutputMatching(1)

    # The actual correction:
    bd.SetWassersteinMetric(metric)

    # Hard assertion: never silently fall back.
    actual_metric = bd.GetWassersteinMetric()

    if str(actual_metric) != metric:
        raise RuntimeError(
            f"Wasserstein setter failed: "
            f"requested={metric!r}, got={actual_metric!r}"
        )

    if hasattr(bd, "SetInputDataObject"):
        bd.SetInputDataObject(0, mb)
    else:
        bd.SetInputData(mb)

    bd.Update()

    dist = get_field_distance(
        bd.GetOutputDataObject(1)
    )

    if dist is None:
        dist = get_field_distance(
            bd.GetOutputDataObject(0)
        )

    if dist is None:
        raise RuntimeError(
            "Could not extract PD distance from TTK output"
        )

    return float(dist)


# ----------------------------------------------------------------------
# Artifact discovery
# ----------------------------------------------------------------------

def discover_pd_files(run_root: Path):
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
            p for p in pd_root.glob("*_pd_port_0.vtu")
            if "_GT_" in p.name
        )

        sr_files = sorted(
            p for p in pd_root.glob("*_pd_port_0.vtu")
            if "_SR_" in p.name
        )

    def map_samples(files):
        out = {}

        for p in files:
            m = SAMPLE_RE.search(p.name)

            if m:
                out[int(m.group(1))] = p

        return out

    return map_samples(gt_files), map_samples(sr_files)


def historical_map(run_root: Path):
    p = (
        run_root
        / "phase_c_final"
        / "pd_pairwise_distances.csv"
    )

    out = {}

    with p.open(newline="") as f:
        for row in csv.DictReader(f):

            m = SAMPLE_RE.search(row["key"])

            if not m:
                raise RuntimeError(
                    f"Cannot parse historical key: {row['key']}"
                )

            s = int(m.group(1))

            out[s] = {
                "historical_key": row["key"],
                "historical_method": row["method"],
                "historical_pd_distance":
                    float(row["pd_distance"]),
            }

    return out


# ----------------------------------------------------------------------
# Pilot
# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--samples",
        default="0,30,34,99,119,120"
    )

    args = ap.parse_args()

    samples = [
        int(x)
        for x in args.samples.split(",")
        if x.strip()
    ]

    # Deliberately span baseline, learned candidate,
    # and superlevel artifact families.
    pilot_runs = [
        "cnn",
        "gan",
        (
            "topology_finetuning/"
            "candidateF_grad_levelset_E2_low_expanded2688_topology"
        ),
        "superlevel_topology/cnn/topology",
    ]

    rows = []

    print("Python / VTK / TTK")
    print("------------------")
    print("VTK =", vtk.vtkVersion().GetVTKVersion())
    print("TTK =", ttk.__file__)
    print()

    for run in pilot_runs:

        run_root = BASE / run

        gt_map, sr_map = discover_pd_files(run_root)
        hist = historical_map(run_root)

        print("=" * 100)
        print("RUN:", run)

        for sample in samples:

            gt = gt_map[sample]
            sr = sr_map[sample]

            historical = hist[sample][
                "historical_pd_distance"
            ]

            w2 = compute_pd_distance(
                gt, sr, "2"
            )

            bottleneck = compute_pd_distance(
                gt, sr, "inf"
            )

            abs_diff = abs(
                historical - w2
            )

            matches = math.isclose(
                historical,
                w2,
                rel_tol=0.0,
                abs_tol=1e-10,
            )

            row = {
                "run": run,
                "sample": sample,
                "historical_pd_distance":
                    historical,
                "explicit_pd_w2": w2,
                "historical_minus_w2":
                    historical - w2,
                "historical_matches_w2":
                    matches,
                "explicit_pd_bottleneck":
                    bottleneck,
            }

            rows.append(row)

            print(
                f"s{sample:03d} "
                f"historical={historical:.15g} "
                f"W2={w2:.15g} "
                f"|diff|={abs_diff:.3e} "
                f"match={matches} "
                f"bottleneck={bottleneck:.15g}"
            )

    out = (
        AUDIT
        / "recompute_pd"
        / "pd_explicit_metric_pilot.csv"
    )

    with out.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys())
        )

        w.writeheader()
        w.writerows(rows)

    failures = [
        r for r in rows
        if not r["historical_matches_w2"]
    ]

    print()
    print("=" * 100)
    print("PILOT SUMMARY")
    print("=" * 100)

    print("comparisons:", len(rows))
    print(
        "historical == explicit W2:",
        len(rows) - len(failures),
        "/",
        len(rows),
    )

    print(
        "W2 mismatches:",
        len(failures)
    )

    print("output:", out)

    if failures:
        raise SystemExit(
            "FAIL: historical values did not all "
            "reproduce as explicit W2"
        )


if __name__ == "__main__":
    main()
