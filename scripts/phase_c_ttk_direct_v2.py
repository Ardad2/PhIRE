#!/usr/bin/env python3
from __future__ import annotations

"""
Direct TTK computation in-memory (avoids CLI VTU I/O issues).
- Reads VTI
- Runs TTK PersistenceDiagram + MergeTree in memory
- Extracts robust metrics
- Optionally writes PD/MT outputs as ASCII VTU (readable)
- Saves per-file CSV + per-label summary CSV
"""

import argparse
import csv
import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy


import os, sys

CANDIDATE_PATHS = [
    "/usr/local/lib/python3/dist-packages",
    "/usr/local/lib/python3.12/site-packages",
    "/usr/local/lib/python3.12/dist-packages",
]

for p in CANDIDATE_PATHS:
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

# --- TTK imports ---
try:
    import topologytoolkit as ttkpkg
    from topologytoolkit import ttkPersistenceDiagram, ttkMergeTree
    try:
        from topologytoolkit import ttkArrayPreconditioning
        HAVE_PRECOND = True
    except Exception:
        HAVE_PRECOND = False
    HAVE_TTK = True
except Exception as e:
    HAVE_TTK = False
    HAVE_PRECOND = False
    TTK_IMPORT_ERROR = e


def set_threads(filt: Any, n: int) -> None:
    for m in ("SetThreadNumber", "SetThreads"):
        if hasattr(filt, m):
            try:
                getattr(filt, m)(int(n))
                return
            except Exception:
                pass


def vti_read(path: Path) -> vtk.vtkImageData:
    r = vtk.vtkXMLImageDataReader()
    r.SetFileName(str(path))
    r.Update()
    out = r.GetOutput()
    if out is None or out.GetNumberOfPoints() == 0:
        raise RuntimeError(f"Failed to read VTI or empty dataset: {path}")
    return out


def ensure_scalar_exists(img: vtk.vtkImageData, scalar_name: str) -> None:
    pd = img.GetPointData()
    if pd is None or pd.GetArray(scalar_name) is None:
        names = [pd.GetArrayName(i) for i in range(pd.GetNumberOfArrays())] if pd else []
        raise RuntimeError(f"Scalar '{scalar_name}' not found. Arrays: {names}")


def get_scalar_range(img: vtk.vtkImageData, scalar_name: str) -> Tuple[float, float]:
    arr = img.GetPointData().GetArray(scalar_name)
    lo, hi = arr.GetRange()
    return float(lo), float(hi)


def get_point_array(ds: vtk.vtkDataSet, name: str) -> Optional[np.ndarray]:
    pd = ds.GetPointData()
    if pd is None:
        return None
    arr = pd.GetArray(name)
    if arr is None:
        return None
    return vtk_to_numpy(arr)


def parse_label_from_filename(stem: str) -> Dict[str, str]:
    # gan_GT_s0_speed_p160_x0_y0
    parts = stem.split("_")
    dataset = parts[1] if len(parts) > 1 else "UNK"
    sample = parts[2] if len(parts) > 2 else "UNK"
    label = f"GAN_{dataset}" if dataset in ("GT", "SR") else "GAN_UNK"
    return {"dataset": dataset, "sample": sample, "label": label}


def write_ascii_vtu(path: Path, ug: vtk.vtkUnstructuredGrid) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    w = vtk.vtkXMLUnstructuredGridWriter()
    w.SetFileName(str(path))
    w.SetInputData(ug)
    w.SetDataModeToAscii()
    w.SetCompressorTypeToNone()
    ok = w.Write()
    if not ok:
        raise RuntimeError(f"Failed to write VTU: {path}")


@dataclass
class PersStats:
    count_pos: int
    maxv: float
    meanv: float
    sumv: float
    count_gt_eps: int


def summarize_persistence(pers: np.ndarray, eps: float) -> PersStats:
    pers = pers.astype(np.float64, copy=False)
    pos = pers[pers > 0]
    if pos.size == 0:
        return PersStats(0, 0.0, 0.0, 0.0, 0)
    return PersStats(
        count_pos=int(pos.size),
        maxv=float(np.max(pos)),
        meanv=float(np.mean(pos)),
        sumv=float(np.sum(pos)),
        count_gt_eps=int(np.sum(pos > eps)),
    )


def compute_pd(img: vtk.vtkImageData, scalar_name: str, threads: int, eps: float):
    inp = img
    if HAVE_PRECOND:
        pre = ttkArrayPreconditioning()
        pre.SetInputData(inp)
        set_threads(pre, threads)
        pre.SetInputArrayToProcess(
            0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, scalar_name
        )
        pre.Update()
        inp = pre.GetOutput()

    f = ttkPersistenceDiagram()
    f.SetInputData(inp)
    set_threads(f, threads)
    f.SetInputArrayToProcess(
        0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, scalar_name
    )
    f.Update()

    ug = f.GetOutput()
    if ug is None:
        raise RuntimeError("PD filter returned no output")

    pt = get_point_array(ug, "PairType")
    pairtype_counts: Dict[int, int] = {}
    if pt is not None and pt.size:
        pti = pt.astype(np.int64, copy=False)
        for k in np.unique(pti):
            pairtype_counts[int(k)] = int(np.sum(pti == k))

    pers = get_point_array(ug, "Persistence")
    if pers is None:
        b = get_point_array(ug, "Birth")
        d = get_point_array(ug, "Death")
        if b is not None and d is not None and b.size and d.size:
            pers = (d - b)
        else:
            pers = np.array([], dtype=np.float64)

    pers_stats = summarize_persistence(pers, eps)
    return ug, pairtype_counts, pers_stats


def compute_mt(img: vtk.vtkImageData, scalar_name: str, threads: int, eps: float):
    inp = img
    if HAVE_PRECOND:
        pre = ttkArrayPreconditioning()
        pre.SetInputData(inp)
        set_threads(pre, threads)
        pre.SetInputArrayToProcess(
            0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, scalar_name
        )
        pre.Update()
        inp = pre.GetOutput()

    mt = ttkMergeTree()
    mt.SetInputData(inp)
    set_threads(mt, threads)
    mt.SetInputArrayToProcess(
        0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, scalar_name
    )
    mt.Update()

    tree0 = vtk.vtkUnstructuredGrid.SafeDownCast(mt.GetOutputDataObject(0))
    seg1 = vtk.vtkUnstructuredGrid.SafeDownCast(mt.GetOutputDataObject(1))

    if tree0 is None:
        raise RuntimeError("MergeTree port 0 is not an UnstructuredGrid")

    pers = get_point_array(tree0, "Persistence")
    if pers is None:
        pers = np.array([], dtype=np.float64)

    pers_stats = summarize_persistence(pers, eps)
    return tree0, seg1, pers_stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="vtk_inputs/gan_*.vti")
    ap.add_argument("--scalar", default="wind_speed")
    ap.add_argument("--threads", type=int, default=20)
    ap.add_argument("--outdir", default="ttk_outputs/direct")
    ap.add_argument("--eps-frac", type=float, default=0.01)
    ap.add_argument("--write-ascii-vtu", action="store_true")
    args = ap.parse_args()

    if not HAVE_TTK:
        raise SystemExit(f"TTK python bindings not available: {TTK_IMPORT_ERROR}")

    print("TTK python module loaded from:", ttkpkg.__file__)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    inputs = [Path(p) for p in sorted(glob.glob(args.glob))]
    if not inputs:
        raise SystemExit(f"No input VTIs matched: {args.glob}")

    rows: List[Dict[str, Any]] = []

    for vti_path in inputs:
        stem = vti_path.stem
        meta = parse_label_from_filename(stem)

        img = vti_read(vti_path)
        ensure_scalar_exists(img, args.scalar)
        lo, hi = get_scalar_range(img, args.scalar)
        eps = float(args.eps_frac) * (hi - lo)

        print(f"\n=== {vti_path.name} ===")
        print(f"  dims={img.GetDimensions()} range=[{lo:.4f},{hi:.4f}] eps={eps:.4f}")

        pd_ug, pairtype_counts, pd_pers = compute_pd(img, args.scalar, args.threads, eps)
        mt_tree0, mt_seg1, mt_pers = compute_mt(img, args.scalar, args.threads, eps)

        row = {
            "file": str(vti_path),
            "label": meta["label"],
            "dataset": meta["dataset"],
            "sample": meta["sample"],
            "scalar": args.scalar,
            "scalar_min": lo,
            "scalar_max": hi,
            "eps": eps,

            "pd_pairs_total": int(pd_ug.GetNumberOfPoints()),
            "pd_pairtype_counts": str(pairtype_counts),

            "pd_pers_count_pos": pd_pers.count_pos,
            "pd_pers_max": pd_pers.maxv,
            "pd_pers_mean": pd_pers.meanv,
            "pd_pers_sum": pd_pers.sumv,
            "pd_pers_count_gt_eps": pd_pers.count_gt_eps,

            "mt_nodes": int(mt_tree0.GetNumberOfPoints()),
            "mt_arcs": int(mt_tree0.GetNumberOfCells()),

            "mt_pers_count_pos": mt_pers.count_pos,
            "mt_pers_max": mt_pers.maxv,
            "mt_pers_mean": mt_pers.meanv,
            "mt_pers_sum": mt_pers.sumv,
            "mt_pers_count_gt_eps": mt_pers.count_gt_eps,
        }
        rows.append(row)

        print(f"  PD: pairs={row['pd_pairs_total']} maxPers={row['pd_pers_max']:.4f} pairTypes={pairtype_counts}")
        print(f"  MT: nodes={row['mt_nodes']} arcs={row['mt_arcs']} maxPers={row['mt_pers_max']:.4f}")

        if args.write_ascii_vtu:
            write_ascii_vtu(outdir / "ascii_vtu/pd" / f"{stem}_pd_ascii.vtu", pd_ug)
            write_ascii_vtu(outdir / "ascii_vtu/mt" / f"{stem}_mt_port0_ascii.vtu", mt_tree0)
            if mt_seg1 is not None:
                write_ascii_vtu(outdir / "ascii_vtu/mt" / f"{stem}_mt_port1_ascii.vtu", mt_seg1)

    # per-file CSV
    per_csv = outdir / "phase_c_results.csv"
    with open(per_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n✓ Wrote: {per_csv}")

    # summary CSV (mean/std per label)
    def mean_std(xs: List[float]) -> Tuple[float, float]:
        a = np.asarray(xs, dtype=np.float64)
        return float(a.mean()), float(a.std(ddof=1)) if a.size > 1 else 0.0

    metrics = [
        "pd_pairs_total", "pd_pers_max", "pd_pers_sum", "pd_pers_count_gt_eps",
        "mt_nodes", "mt_arcs", "mt_pers_max", "mt_pers_sum", "mt_pers_count_gt_eps",
    ]

    by_label: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_label.setdefault(r["label"], []).append(r)

    summary_rows: List[Dict[str, Any]] = []
    for label, rr in by_label.items():
        out = {"label": label, "n": len(rr)}
        for m in metrics:
            mu, sd = mean_std([float(x[m]) for x in rr])
            out[f"{m}_mean"] = mu
            out[f"{m}_std"] = sd
        summary_rows.append(out)

    sum_csv = outdir / "phase_c_summary.csv"
    with open(sum_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    print(f"✓ Wrote: {sum_csv}")
    print("\nDone.")

if __name__ == "__main__":
    main()
