#!/usr/bin/env python3
"""
Phase C: Direct TTK computation (avoid CLI VTU I/O issues)
- Reads VTI (ASCII VTI you already generate)
- Runs TTK PD + MergeTree in memory
- Extracts robust metrics
- Optionally rewrites PD/MT outputs as ASCII VTU (readable by VTK/ParaView)
- Saves per-file CSV + group summary CSV
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

# --- TTK imports ---
try:
    from topologytoolkit import ttkPersistenceDiagram, ttkMergeTree
    # Optional (performance): order array preconditioning
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


# -------------------------
# Helpers
# -------------------------

def set_threads(filt: Any, n: int) -> None:
    """Best-effort thread setter (TTK wrappers differ slightly)."""
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


def get_point_array(ds: vtk.vtkDataSet, name: str) -> Optional[np.ndarray]:
    pd = ds.GetPointData()
    if pd is None:
        return None
    arr = pd.GetArray(name)
    if arr is None:
        return None
    return vtk_to_numpy(arr)


def get_scalar_range(img: vtk.vtkImageData, scalar_name: str) -> Tuple[float, float]:
    arr = img.GetPointData().GetArray(scalar_name)
    if arr is None:
        raise RuntimeError(f"Scalar array not found: {scalar_name}")
    lo, hi = arr.GetRange()
    return float(lo), float(hi)


def ensure_scalar_exists(img: vtk.vtkImageData, scalar_name: str) -> None:
    if img.GetPointData().GetArray(scalar_name) is None:
        names = [img.GetPointData().GetArrayName(i) for i in range(img.GetPointData().GetNumberOfArrays())]
        raise RuntimeError(f"Scalar '{scalar_name}' not found. Arrays: {names}")


def parse_label_from_filename(stem: str) -> Dict[str, str]:
    """
    Expected stems like: gan_GT_s0_speed_p160_x0_y0
    We extract dataset (GT/SR) + sample (s0..).
    """
    parts = stem.split("_")
    dataset = "UNK"
    sample = "UNK"
    if len(parts) >= 3:
        dataset = parts[1]  # GT / SR
        sample = parts[2]   # s0 / s1 ...
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
class PDStats:
    n_pairs_total: int
    # PairType counts (TTK-specific; for 2D fields type_0~H0, type_1~H1 typically)
    pairtype_counts: Dict[int, int]
    pers_count_pos: int
    pers_max: float
    pers_mean: float
    pers_sum: float
    pers_count_gt_eps: int


@dataclass
class MTStats:
    nodes: int
    arcs: int
    pers_count_pos: int
    pers_max: float
    pers_mean: float
    pers_sum: float
    pers_count_gt_eps: int


def compute_pd(img: vtk.vtkImageData, scalar_name: str, threads: int, eps: float) -> Tuple[PDStats, vtk.vtkUnstructuredGrid]:
    ensure_scalar_exists(img, scalar_name)

    inp = img
    if HAVE_PRECOND:
        # This can remove the "No pre-existing order" warnings and speed up
        pre = ttkArrayPreconditioning()
        pre.SetInputData(inp)
        set_threads(pre, threads)
        pre.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, scalar_name)
        pre.Update()
        inp = pre.GetOutput()

    pd_f = ttkPersistenceDiagram()
    pd_f.SetInputData(inp)
    set_threads(pd_f, threads)
    pd_f.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, scalar_name)
    pd_f.Update()

    ug = pd_f.GetOutput()
    if ug is None:
        raise RuntimeError("PD filter returned no output")

    n = int(ug.GetNumberOfPoints())

    pairtype_counts: Dict[int, int] = {}
    pt = get_point_array(ug, "PairType")
    if pt is not None and pt.size:
        # PairType can be float in some builds; normalize to int
        pt_i = pt.astype(np.int64, copy=False)
        for k in np.unique(pt_i):
            pairtype_counts[int(k)] = int(np.sum(pt_i == k))

    pers = get_point_array(ug, "Persistence")
    if pers is None:
        # Fallback: Death - Birth if present
        b = get_point_array(ug, "Birth")
        d = get_point_array(ug, "Death")
        if b is not None and d is not None and b.size and d.size:
            pers = (d - b).astype(np.float64)
        else:
            pers = np.array([], dtype=np.float64)

    pers = pers.astype(np.float64, copy=False)
    pers_pos = pers[pers > 0]

    if pers_pos.size:
        pers_max = float(np.max(pers_pos))
        pers_mean = float(np.mean(pers_pos))
        pers_sum = float(np.sum(pers_pos))
        pers_count_pos = int(pers_pos.size)
        pers_count_gt_eps = int(np.sum(pers_pos > eps))
    else:
        pers_max = pers_mean = pers_sum = 0.0
        pers_count_pos = pers_count_gt_eps = 0

    return PDStats(
        n_pairs_total=n,
        pairtype_counts=pairtype_counts,
        pers_count_pos=pers_count_pos,
        pers_max=pers_max,
        pers_mean=pers_mean,
        pers_sum=pers_sum,
        pers_count_gt_eps=pers_count_gt_eps,
    ), ug


def compute_mt(img: vtk.vtkImageData, scalar_name: str, threads: int, eps: float) -> Tuple[MTStats, vtk.vtkUnstructuredGrid, vtk.vtkUnstructuredGrid]:
    ensure_scalar_exists(img, scalar_name)

    inp = img
    if HAVE_PRECOND:
        pre = ttkArrayPreconditioning()
        pre.SetInputData(inp)
        set_threads(pre, threads)
        pre.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, scalar_name)
        pre.Update()
        inp = pre.GetOutput()

    mt = ttkMergeTree()
    mt.SetInputData(inp)
    set_threads(mt, threads)
    mt.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, scalar_name)
    mt.Update()

    # Port 0: tree (vtu), Port 1: segmentation (vtu), Port 2: simplified scalar field (vti)
    # In python wrappers, GetOutput() can behave differently, so be explicit:
    tree0 = mt.GetOutputDataObject(0)
    seg1 = mt.GetOutputDataObject(1)

    # cast to UG
    tree0 = vtk.vtkUnstructuredGrid.SafeDownCast(tree0)
    seg1 = vtk.vtkUnstructuredGrid.SafeDownCast(seg1)

    if tree0 is None:
        raise RuntimeError("MergeTree port 0 is not an UnstructuredGrid")

    nodes = int(tree0.GetNumberOfPoints())
    arcs = int(tree0.GetNumberOfCells())

    pers = get_point_array(tree0, "Persistence")
    if pers is None:
        pers = np.array([], dtype=np.float64)
    pers = pers.astype(np.float64, copy=False)
    pers_pos = pers[pers > 0]

    if pers_pos.size:
        pers_max = float(np.max(pers_pos))
        pers_mean = float(np.mean(pers_pos))
        pers_sum = float(np.sum(pers_pos))
        pers_count_pos = int(pers_pos.size)
        pers_count_gt_eps = int(np.sum(pers_pos > eps))
    else:
        pers_max = pers_mean = pers_sum = 0.0
        pers_count_pos = pers_count_gt_eps = 0

    return MTStats(
        nodes=nodes,
        arcs=arcs,
        pers_count_pos=pers_count_pos,
        pers_max=pers_max,
        pers_mean=pers_mean,
        pers_sum=pers_sum,
        pers_count_gt_eps=pers_count_gt_eps,
    ), tree0, seg1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="vtk_inputs/gan_*.vti", help="Input VTI glob")
    ap.add_argument("--scalar", default="wind_speed", help="Scalar array name in VTI")
    ap.add_argument("--threads", type=int, default=20)
    ap.add_argument("--outdir", default="ttk_outputs/direct", help="Output directory")
    ap.add_argument("--eps-frac", type=float, default=0.01,
                    help="Persistence threshold as fraction of scalar range (used for count_gt_eps metrics)")
    ap.add_argument("--write-ascii-vtu", action="store_true",
                    help="Also write PD and MT outputs as ASCII VTU (readable)")
    args = ap.parse_args()

    if not HAVE_TTK:
        raise SystemExit(f"TTK python bindings not available: {TTK_IMPORT_ERROR}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    inputs = sorted(Path(".").glob(args.glob)) if ("*" not in args.glob and "?" not in args.glob and "[" not in args.glob) else sorted(Path(".").glob(args.glob))
    # The above line is defensive; globbing via Path.glob is fine for your use.
    if not inputs:
        # fallback: use glob module semantics
        import glob as _glob
        inputs = [Path(p) for p in sorted(_glob.glob(args.glob))]

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
        print(f"  dims: {img.GetDimensions()} scalar: {args.scalar} range=[{lo:.4f}, {hi:.4f}] eps={eps:.4f}")

        pd_stats, pd_ug = compute_pd(img, args.scalar, args.threads, eps)
        mt_stats, mt_tree0, mt_seg1 = compute_mt(img, args.scalar, args.threads, eps)

        # PD type_0/type_1 columns (common for 2D); keep generic dict too
        pd_t0 = pd_stats.pairtype_counts.get(0, 0)
        pd_t1 = pd_stats.pairtype_counts.get(1, 0)

        row = {
            "file": str(vti_path),
            "label": meta["label"],
            "dataset": meta["dataset"],
            "sample": meta["sample"],
            "scalar": args.scalar,
            "scalar_min": lo,
            "scalar_max": hi,
            "eps": eps,

            # PD
            "pd_pairs_total": pd_stats.n_pairs_total,
            "pd_pairs_type_0": pd_t0,
            "pd_pairs_type_1": pd_t1,
            "pd_pers_count_pos": pd_stats.pers_count_pos,
            "pd_pers_max": pd_stats.pers_max,
            "pd_pers_mean": pd_stats.pers_mean,
            "pd_pers_sum": pd_stats.pers_sum,
            "pd_pers_count_gt_eps": pd_stats.pers_count_gt_eps,

            # MT
            "mt_nodes": mt_stats.nodes,
            "mt_arcs": mt_stats.arcs,
            "mt_pers_count_pos": mt_stats.pers_count_pos,
            "mt_pers_max": mt_stats.pers_max,
            "mt_pers_mean": mt_stats.pers_mean,
            "mt_pers_sum": mt_stats.pers_sum,
            "mt_pers_count_gt_eps": mt_stats.pers_count_gt_eps,
        }
        rows.append(row)

        print(f"  PD: total={row['pd_pairs_total']} type0={pd_t0} type1={pd_t1} maxPers={row['pd_pers_max']:.4f}")
        print(f"  MT: nodes={row['mt_nodes']} arcs={row['mt_arcs']} maxPers={row['mt_pers_max']:.4f}")

        if args.write_ascii_vtu:
            pd_out = outdir / "ascii_vtu" / "pd" / f"{stem}_pd_ascii.vtu"
            mt0_out = outdir / "ascii_vtu" / "mt" / f"{stem}_mt_port0_ascii.vtu"
            mt1_out = outdir / "ascii_vtu" / "mt" / f"{stem}_mt_port1_ascii.vtu"
            write_ascii_vtu(pd_out, pd_ug)
            write_ascii_vtu(mt0_out, mt_tree0)
            if mt_seg1 is not None:
                write_ascii_vtu(mt1_out, mt_seg1)

    # Write per-file CSV
    per_csv = outdir / "phase_c_results.csv"
    with open(per_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n✓ Wrote: {per_csv}")

    # Group summary CSV (mean/std over samples per label)
    # Do it without pandas to keep dependencies minimal.
    def agg(vals: List[float]) -> Tuple[float, float]:
        a = np.asarray(vals, dtype=np.float64)
        return float(np.mean(a)), float(np.std(a, ddof=1)) if a.size > 1 else 0.0

    by_label: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_label.setdefault(r["label"], []).append(r)

    summary_rows = []
    metrics = ["pd_pairs_total", "pd_pairs_type_0", "pd_pairs_type_1", "pd_pers_max", "pd_pers_sum",
               "mt_nodes", "mt_arcs", "mt_pers_max", "mt_pers_sum", "mt_pers_count_gt_eps"]
    for label, rr in by_label.items():
        out = {"label": label, "n": len(rr)}
        for m in metrics:
            mean, std = agg([float(x[m]) for x in rr])
            out[f"{m}_mean"] = mean
            out[f"{m}_std"] = std
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
