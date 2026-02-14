#!/usr/bin/env python3
"""
Phase C distances (Phase-B style):
- For each sample sK: compare GT vs SR
- PD distances: Bottleneck + Wasserstein(p=1) + Wasserstein(p=2) if available
- Merge tree distance: via ttkMergeTreeDistanceMatrix (2x2) if available
Outputs:
  ttk_outputs/direct/distances_pairwise.csv
  ttk_outputs/direct/distances_summary.csv
  ttk_outputs/direct/mt_distance_matrix_all.csv (optional)
"""

from __future__ import annotations
import os, sys, glob, csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

# Make sure typical install locations are visible (even if PYTHONPATH wasn't set)
CANDIDATE_PATHS = [
    "/usr/local/lib/python3/dist-packages",
    "/usr/local/lib/python3.12/site-packages",
    "/usr/local/lib/python3.12/dist-packages",
]
for p in CANDIDATE_PATHS:
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

# --- TTK imports (only what we can actually use) ---
import topologytoolkit as ttkpkg
from topologytoolkit import ttkPersistenceDiagram, ttkMergeTree

HAVE_PRECOND = False
try:
    from topologytoolkit import ttkArrayPreconditioning
    HAVE_PRECOND = True
except Exception:
    pass

HAVE_BN = False
try:
    from topologytoolkit import ttkBottleneckDistance
    HAVE_BN = True
except Exception:
    pass

HAVE_WASS = False
try:
    from topologytoolkit import ttkWassersteinDistance
    HAVE_WASS = True
except Exception:
    pass

HAVE_MT_DIST = False
try:
    from topologytoolkit import ttkMergeTreeDistanceMatrix
    HAVE_MT_DIST = True
except Exception:
    pass


def set_if_exists(obj: Any, name: str, value: Any) -> None:
    if hasattr(obj, name):
        try:
            getattr(obj, name)(value)
        except Exception:
            pass


def set_threads(filt: Any, n: int) -> None:
    for m in ("SetThreadNumber", "SetThreads"):
        if hasattr(filt, m):
            try:
                getattr(filt, m)(int(n))
                return
            except Exception:
                pass


def set_input(filt: Any, port: int, data: vtk.vtkDataObject) -> None:
    # wrappers differ; try common APIs
    if hasattr(filt, "SetInputDataObject"):
        filt.SetInputDataObject(port, data)
        return
    if port == 0 and hasattr(filt, "SetInputData"):
        filt.SetInputData(data)
        return
    # fallback: SetInputData on ported pipelines
    if hasattr(filt, "SetInputData"):
        filt.SetInputData(data)
        return
    raise RuntimeError("Could not set input on this filter")


def vti_read(path: Path) -> vtk.vtkImageData:
    r = vtk.vtkXMLImageDataReader()
    r.SetFileName(str(path))
    r.Update()
    img = r.GetOutput()
    if img is None or img.GetNumberOfPoints() == 0:
        raise RuntimeError(f"Failed to read: {path}")
    return img


def ensure_scalar(img: vtk.vtkImageData, scalar: str) -> None:
    if img.GetPointData().GetArray(scalar) is None:
        names = [img.GetPointData().GetArrayName(i) for i in range(img.GetPointData().GetNumberOfArrays())]
        raise RuntimeError(f"Scalar '{scalar}' not found. Arrays: {names}")


def compute_pd(img: vtk.vtkImageData, scalar: str, threads: int) -> vtk.vtkUnstructuredGrid:
    inp = img
    if HAVE_PRECOND:
        pre = ttkArrayPreconditioning()
        set_input(pre, 0, inp)
        set_threads(pre, threads)
        pre.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, scalar)
        pre.Update()
        inp = pre.GetOutput()

    f = ttkPersistenceDiagram()
    set_input(f, 0, inp)
    set_threads(f, threads)
    f.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, scalar)
    f.Update()
    out = f.GetOutput()
    if out is None:
        raise RuntimeError("PD output is None")
    return out


def compute_mt_tree(img: vtk.vtkImageData, scalar: str, threads: int) -> vtk.vtkUnstructuredGrid:
    inp = img
    if HAVE_PRECOND:
        pre = ttkArrayPreconditioning()
        set_input(pre, 0, inp)
        set_threads(pre, threads)
        pre.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, scalar)
        pre.Update()
        inp = pre.GetOutput()

    mt = ttkMergeTree()
    set_input(mt, 0, inp)
    set_threads(mt, threads)
    mt.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, scalar)
    mt.Update()

    tree0 = vtk.vtkUnstructuredGrid.SafeDownCast(mt.GetOutputDataObject(0))
    if tree0 is None:
        raise RuntimeError("MergeTree port0 is not a vtkUnstructuredGrid")
    return tree0


def extract_single_distance(out: vtk.vtkDataObject) -> float:
    """
    TTK distance filters typically output vtkTable OR dataset with FieldData.
    This tries to pull the first numeric scalar that looks like a distance.
    """
    # vtkTable case
    table = vtk.vtkTable.SafeDownCast(out)
    if table is not None:
        ncols = table.GetNumberOfColumns()
        nrows = table.GetNumberOfRows()
        # Prefer columns whose name contains "dist"
        col_order = list(range(ncols))
        names = [table.GetColumnName(i) or "" for i in range(ncols)]
        dist_cols = [i for i, nm in enumerate(names) if "dist" in nm.lower()]
        if dist_cols:
            col_order = dist_cols + [i for i in col_order if i not in dist_cols]

        for ci in col_order:
            col = table.GetColumn(ci)
            arr = vtk_to_numpy(col)
            if arr.dtype.kind in "iuf" and arr.size > 0:
                return float(arr.ravel()[0])

    # FieldData case
    if hasattr(out, "GetFieldData") and out.GetFieldData() is not None:
        fd = out.GetFieldData()
        for i in range(fd.GetNumberOfArrays()):
            a = fd.GetArray(i)
            if a is None:
                continue
            name = (a.GetName() or "").lower()
            if "dist" in name and a.GetNumberOfTuples() > 0:
                v = a.GetTuple1(0)
                return float(v)
        # fallback: first numeric field array
        for i in range(fd.GetNumberOfArrays()):
            a = fd.GetArray(i)
            if a is None:
                continue
            if a.GetNumberOfTuples() > 0:
                try:
                    return float(a.GetTuple1(0))
                except Exception:
                    pass

    raise RuntimeError("Could not extract distance from output")


def pd_bottleneck(pd_a: vtk.vtkUnstructuredGrid, pd_b: vtk.vtkUnstructuredGrid, threads: int) -> Optional[float]:
    if not HAVE_BN:
        return None
    f = ttkBottleneckDistance()
    set_input(f, 0, pd_a)
    set_input(f, 1, pd_b)
    set_threads(f, threads)
    f.Update()
    return extract_single_distance(f.GetOutputDataObject(0))


def pd_wasserstein(pd_a: vtk.vtkUnstructuredGrid, pd_b: vtk.vtkUnstructuredGrid, threads: int, p: int) -> Optional[float]:
    if not HAVE_WASS:
        return None
    f = ttkWassersteinDistance()
    set_input(f, 0, pd_a)
    set_input(f, 1, pd_b)
    set_threads(f, threads)

    # different builds expose different setter names; try common ones
    for name in ("SetP", "SetPower", "SetWassersteinPower", "SetWassersteinOrder"):
        set_if_exists(f, name, int(p))

    f.Update()
    return extract_single_distance(f.GetOutputDataObject(0))


def mt_distance(tree_a: vtk.vtkUnstructuredGrid, tree_b: vtk.vtkUnstructuredGrid, threads: int) -> Optional[float]:
    if not HAVE_MT_DIST:
        return None

    mb = vtk.vtkMultiBlockDataSet()
    mb.SetNumberOfBlocks(2)
    mb.SetBlock(0, tree_a)
    mb.SetBlock(1, tree_b)

    f = ttkMergeTreeDistanceMatrix()
    # this one usually takes a multiblock on port 0
    set_input(f, 0, mb)
    set_threads(f, threads)

    f.Update()
    out = f.GetOutputDataObject(0)

    # Output is typically a vtkTable representing a matrix.
    table = vtk.vtkTable.SafeDownCast(out)
    if table is None:
        # fallback: maybe it returns a dataset with field data
        return extract_single_distance(out)

    # Build numeric matrix and pick [0,1]
    nrows = table.GetNumberOfRows()
    ncols = table.GetNumberOfColumns()
    if nrows >= 2 and ncols >= 2:
        M = np.zeros((nrows, ncols), dtype=np.float64)
        for j in range(ncols):
            arr = vtk_to_numpy(table.GetColumn(j))
            if arr.dtype.kind in "iuf":
                M[:, j] = arr
        return float(M[0, 1])

    # Or possibly a single-value table
    return extract_single_distance(out)


def mean_std(xs: List[float]) -> Tuple[float, float]:
    a = np.asarray(xs, dtype=np.float64)
    if a.size == 0:
        return (float("nan"), float("nan"))
    return float(a.mean()), float(a.std(ddof=1)) if a.size > 1 else 0.0


def main():
    print("topologytoolkit loaded from:", ttkpkg.__file__)
    print("Have: bottleneck=", HAVE_BN, " wasserstein=", HAVE_WASS, " mt_distance=", HAVE_MT_DIST)

    scalar = "wind_speed"
    threads = 20
    outdir = Path("ttk_outputs/direct")
    outdir.mkdir(parents=True, exist_ok=True)

    # Pair up GT/SR by sample id from filename gan_GT_sK_... / gan_SR_sK_...
    vtis = sorted(glob.glob("vtk_inputs/gan_*.vti"))
    by_sample: Dict[str, Dict[str, str]] = {}
    for p in vtis:
        stem = Path(p).stem
        parts = stem.split("_")
        if len(parts) < 3:
            continue
        dataset = parts[1]   # GT/SR
        sample = parts[2]    # s0...
        by_sample.setdefault(sample, {})[dataset] = p

    rows = []
    for sample, d in sorted(by_sample.items()):
        if "GT" not in d or "SR" not in d:
            continue

        gt_path = Path(d["GT"])
        sr_path = Path(d["SR"])
        print(f"\n=== {sample}: GT vs SR ===")
        print(" GT:", gt_path.name)
        print(" SR:", sr_path.name)

        gt_img = vti_read(gt_path); ensure_scalar(gt_img, scalar)
        sr_img = vti_read(sr_path); ensure_scalar(sr_img, scalar)

        pd_gt = compute_pd(gt_img, scalar, threads)
        pd_sr = compute_pd(sr_img, scalar, threads)

        tree_gt = compute_mt_tree(gt_img, scalar, threads)
        tree_sr = compute_mt_tree(sr_img, scalar, threads)

        bn = pd_bottleneck(pd_gt, pd_sr, threads)
#        w1 = pd_wasserstein(pd_gt, pd_sr, threads, p=1)
#        w2 = pd_wasserstein(pd_gt, pd_sr, threads, p=2)
#        mtd = mt_distance(tree_gt, tree_sr, threads)

        row = {
            "sample": sample,
            "pd_bottleneck": bn,
            "pd_wasserstein_p1": w1,
            "pd_wasserstein_p2": w2,
            "mt_distance": mtd,
        }
        rows.append(row)
        print("  PD bottleneck:", bn)
        print("  PD W1:", w1, " PD W2:", w2)
        print("  MT distance:", mtd)

    # Write pairwise CSV
    pair_csv = outdir / "distances_pairwise.csv"
    with open(pair_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("\n✓ Wrote:", pair_csv)

    # Summary CSV
    summary = {}
    for k in ["pd_bottleneck", "pd_wasserstein_p1", "pd_wasserstein_p2", "mt_distance"]:
        vals = [r[k] for r in rows if r[k] is not None]
        mu, sd = mean_std([float(x) for x in vals])
        summary[k + "_mean"] = mu
        summary[k + "_std"] = sd

    sum_csv = outdir / "distances_summary.csv"
    with open(sum_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary.keys()))
        w.writeheader()
        w.writerow(summary)
    print("✓ Wrote:", sum_csv)


if __name__ == "__main__":
    main()
