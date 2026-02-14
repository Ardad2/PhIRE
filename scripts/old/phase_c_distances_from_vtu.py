#!/usr/bin/env python3
from __future__ import annotations
import os, sys, re, glob, csv
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

# Ensure TTK module importable in case PYTHONPATH isn't set in runtime
for p in ["/usr/local/lib/python3/dist-packages", "/usr/local/lib/python3.12/site-packages", "/usr/local/lib/python3.12/dist-packages"]:
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

# TTK distance filters (NO ttkMergeTree usage here)
HAVE_BN = False
HAVE_WASS = False
try:
    from topologytoolkit import ttkBottleneckDistance
    HAVE_BN = True
except Exception:
    pass

try:
    from topologytoolkit import ttkWassersteinDistance
    HAVE_WASS = True
except Exception:
    pass


def read_ugrid(path: Path) -> vtk.vtkUnstructuredGrid:
    r = vtk.vtkXMLUnstructuredGridReader()
    r.SetFileName(str(path))
    r.Update()
    out = r.GetOutput()
    if out is None or out.GetNumberOfPoints() == 0:
        raise RuntimeError(f"Failed to read VTU: {path}")
    return out


def extract_distance(output: vtk.vtkDataObject) -> float:
    # often vtkTable
    t = vtk.vtkTable.SafeDownCast(output)
    if t is not None and t.GetNumberOfRows() > 0:
        for ci in range(t.GetNumberOfColumns()):
            arr = vtk_to_numpy(t.GetColumn(ci))
            if arr.size and arr.dtype.kind in "iuf":
                return float(arr.ravel()[0])

    # or FieldData numeric
    if hasattr(output, "GetFieldData") and output.GetFieldData() is not None:
        fd = output.GetFieldData()
        for i in range(fd.GetNumberOfArrays()):
            a = fd.GetArray(i)
            if a is None or a.GetNumberOfTuples() == 0:
                continue
            try:
                return float(a.GetTuple1(0))
            except Exception:
                continue

    raise RuntimeError("Could not extract distance from output")


def pd_bottleneck(pd_a: vtk.vtkUnstructuredGrid, pd_b: vtk.vtkUnstructuredGrid) -> Optional[float]:
    if not HAVE_BN:
        return None
    f = ttkBottleneckDistance()
    f.SetInputDataObject(0, pd_a)
    f.SetInputDataObject(1, pd_b)
    f.Update()
    return extract_distance(f.GetOutputDataObject(0))


def pd_wasserstein(pd_a: vtk.vtkUnstructuredGrid, pd_b: vtk.vtkUnstructuredGrid, p: int) -> Optional[float]:
    if not HAVE_WASS:
        return None
    f = ttkWassersteinDistance()
    f.SetInputDataObject(0, pd_a)
    f.SetInputDataObject(1, pd_b)

    # setters differ across builds; try a few
    for name in ("SetP", "SetPower", "SetWassersteinPower", "SetWassersteinOrder"):
        if hasattr(f, name):
            try:
                getattr(f, name)(int(p))
            except Exception:
                pass

    f.Update()
    return extract_distance(f.GetOutputDataObject(0))


def find_first_array_like(ds: vtk.vtkDataSet, needle: str) -> Optional[np.ndarray]:
    pd = ds.GetPointData()
    if pd is None:
        return None
    # exact
    a = pd.GetArray(needle)
    if a is not None:
        return vtk_to_numpy(a)
    # case-insensitive contains
    for i in range(pd.GetNumberOfArrays()):
        arr = pd.GetArray(i)
        if arr is None:
            continue
        nm = (arr.GetName() or "")
        if needle.lower() in nm.lower():
            return vtk_to_numpy(arr)
    return None


def mt_distance_persistence_proxy(mt_a: vtk.vtkUnstructuredGrid, mt_b: vtk.vtkUnstructuredGrid) -> Optional[float]:
    """
    Stable proxy distance from merge tree VTU:
      - Take per-node persistence values (array name usually contains 'Persistence')
      - Compute 1D Wasserstein-1 between persistence value multisets
      - Add small structural penalty for feature-count mismatch
    """
    pa = find_first_array_like(mt_a, "Persistence")
    pb = find_first_array_like(mt_b, "Persistence")
    if pa is None or pb is None:
        return None

    pa = np.asarray(pa, dtype=np.float64)
    pb = np.asarray(pb, dtype=np.float64)

    pa = np.sort(pa[pa > 0])
    pb = np.sort(pb[pb > 0])

    if pa.size == 0 and pb.size == 0:
        return 0.0
    if pa.size == 0 or pb.size == 0:
        return float(pa.sum() + pb.sum())

    n = max(pa.size, pb.size)
    pa = np.pad(pa, (0, n - pa.size))
    pb = np.pad(pb, (0, n - pb.size))

    wass1 = np.sum(np.abs(pa - pb))

    # structural penalty (small)
    total = float(pa.sum() + pb.sum())
    struct = abs((pa > 0).sum() - (pb > 0).sum()) / max((pa > 0).sum(), (pb > 0).sum()) * total * 0.10

    return float(wass1 + struct)


def mean_std(vals: List[float]) -> Tuple[float, float]:
    a = np.asarray(vals, dtype=np.float64)
    if a.size == 0:
        return float("nan"), float("nan")
    return float(a.mean()), float(a.std(ddof=1)) if a.size > 1 else 0.0


def main():
    outdir = Path("ttk_outputs/direct")
    outdir.mkdir(parents=True, exist_ok=True)

    # Your actual filenames end with *_port_0.vtu for PD and MT
    pd_dir = outdir / "pd_vtu"
    mt_dir = outdir / "mt_vtu"

    # infer samples from filenames
    pd_files = sorted(pd_dir.glob("gan_*_pd.vtu_port_0.vtu"))
    if not pd_files:
        raise SystemExit(f"No PD files found in {pd_dir} (expected gan_*_pd.vtu_port_0.vtu)")

    # group by (dataset, sample)
    # e.g. gan_GT_s0_..._pd.vtu_port_0.vtu
    pat = re.compile(r"gan_(GT|SR)_(s\d+)_")
    by_sample: Dict[str, Dict[str, Path]] = {}
    for p in pd_files:
        m = pat.search(p.name)
        if not m:
            continue
        ds, s = m.group(1), m.group(2)
        by_sample.setdefault(s, {})[ds] = p

    rows = []
    for s, d in sorted(by_sample.items()):
        if "GT" not in d or "SR" not in d:
            continue

        pd_gt = read_ugrid(d["GT"])
        pd_sr = read_ugrid(d["SR"])

        # MT: use port_0 tree (nodes) to extract Persistence
        mt_gt_files = list(mt_dir.glob(f"gan_GT_{s}_*_mt.vtu_port_0.vtu"))
        mt_sr_files = list(mt_dir.glob(f"gan_SR_{s}_*_mt.vtu_port_0.vtu"))
        if not mt_gt_files or not mt_sr_files:
            raise RuntimeError(f"Missing MT port_0 VTU for {s} in {mt_dir}")

        mt_gt = read_ugrid(mt_gt_files[0])
        mt_sr = read_ugrid(mt_sr_files[0])

        bn = pd_bottleneck(pd_gt, pd_sr)
        w1 = pd_wasserstein(pd_gt, pd_sr, 1)
        w2 = pd_wasserstein(pd_gt, pd_sr, 2)
        mtd = mt_distance_persistence_proxy(mt_gt, mt_sr)

        rows.append({
            "sample": s,
            "pd_bottleneck": bn,
            "pd_wasserstein_p1": w1,
            "pd_wasserstein_p2": w2,
            "mt_distance_proxy": mtd,
        })

        print(s, "bn=", bn, "w1=", w1, "w2=", w2, "mt_proxy=", mtd)

    pair_csv = outdir / "distances_pairwise.csv"
    with open(pair_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("✓ wrote", pair_csv)

    summary = {}
    for k in ["pd_bottleneck", "pd_wasserstein_p1", "pd_wasserstein_p2", "mt_distance_proxy"]:
        vals = [float(r[k]) for r in rows if r[k] is not None]
        mu, sd = mean_std(vals)
        summary[f"{k}_mean"] = mu
        summary[f"{k}_std"] = sd

    sum_csv = outdir / "distances_summary.csv"
    with open(sum_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary.keys()))
        w.writeheader()
        w.writerow(summary)
    print("✓ wrote", sum_csv)

    print("\nNOTE:")
    print(" - PD distances are TTK bottleneck / wasserstein (if available in your build).")
    print(" - MT distance here is a stable persistence-based proxy from merge-tree VTU (no TTK MT distance kernel).")


if __name__ == "__main__":
    main()


