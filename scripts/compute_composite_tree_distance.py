#!/usr/bin/env python3
"""
Compute REAL Persistence Diagram and Merge Tree distances with TTK.

What this does
- PD: bottleneck distance via ttkBottleneckDistance
- MT: merge tree distance via ttkMergeTreeDistanceMatrix

Expected file naming (from your outputs)
- PD: <method>_<GT|SR>_<sample>_pd.vtu_port_<p>.vtu
- MT: <method>_<GT|SR>_<sample>_mt_port_<p>.<vtu|vti>

Example
- gan_GT_s0_speed_p160_x0_y0_pd.vtu_port_0.vtu
- gan_SR_s0_speed_p160_x0_y0_mt_port_1.vtu

This script is dependency-light: only needs VTK + TTK Python bindings.
"""

import argparse
import csv
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import vtk  # type: ignore
import topologytoolkit as ttk  # type: ignore


PD_RE = re.compile(r"^(?P<method>[^_]+)_(?P<label>GT|SR)_(?P<sample>.+)_pd\.vtu_port_(?P<port>\d+)\.vtu$")
MT_RE = re.compile(r"^(?P<method>[^_]+)_(?P<label>GT|SR)_(?P<sample>.+)_mt_port_(?P<port>\d+)\.(?P<ext>vtu|vti)$")


def _read_vtu(path: str) -> "vtk.vtkUnstructuredGrid":
    r = vtk.vtkXMLUnstructuredGridReader()
    r.SetFileName(path)
    r.Update()
    out = r.GetOutput()
    return out


def _read_vti(path: str) -> "vtk.vtkImageData":
    r = vtk.vtkXMLImageDataReader()
    r.SetFileName(path)
    r.Update()
    out = r.GetOutput()
    return out


def compute_pd_bottleneck(pd_a_vtu: str, pd_b_vtu: str, debug: bool = False) -> float:
    """
    Bottleneck distance between two persistence diagrams (two .vtu grids).
    """
    a = _read_vtu(pd_a_vtu)
    b = _read_vtu(pd_b_vtu)

    if a is None or a.GetNumberOfPoints() == 0:
        raise RuntimeError(f"PD read failed or empty: {pd_a_vtu}")
    if b is None or b.GetNumberOfPoints() == 0:
        raise RuntimeError(f"PD read failed or empty: {pd_b_vtu}")

    mb = vtk.vtkMultiBlockDataSet()
    mb.SetNumberOfBlocks(2)
    mb.SetBlock(0, a)
    mb.SetBlock(1, b)

    bd = ttk.ttkBottleneckDistance()
    # TTK expects a single MultiBlock input (not 2 separate inputs)
    bd.SetInputDataObject(0, mb)

    # REQUIRED on some TTK builds, otherwise you see:
    # "[BottleneckDistance] [ERROR] You must specify a valid assignment algorithm."
    if hasattr(bd, "SetPVAlgorithm"):
        bd.SetPVAlgorithm(0)  # 0 = TTK approach (see ttk::BottleneckDistance docs)
    # keep single-thread while debugging stability
    if hasattr(bd, "SetThreadNumber"):
        bd.SetThreadNumber(1)

    bd.Update()

    # Try to read the distance from output field data (varies across TTK versions)
    out0 = bd.GetOutputDataObject(0)
    # sometimes output0 is a vtkMultiBlockDataSet
    fd = None
    if out0 is not None and hasattr(out0, "GetFieldData"):
        fd = out0.GetFieldData()
    dist_candidates: List[float] = []
    if fd is not None:
        for name in ("BottleneckDistance", "Distance", "bottleneck", "Bottleneck"):
            arr = fd.GetArray(name) if hasattr(fd, "GetArray") else None
            if arr is not None and arr.GetNumberOfTuples() >= 1:
                try:
                    dist_candidates.append(float(arr.GetTuple1(0)))
                except Exception:
                    pass

    # fallback: if output1 is an unstructured grid with a scalar array
    if not dist_candidates and bd.GetNumberOfOutputPorts() > 1:
        out1 = bd.GetOutputDataObject(1)
        if out1 is not None and hasattr(out1, "GetFieldData"):
            fd1 = out1.GetFieldData()
            if fd1 is not None:
                for name in ("BottleneckDistance", "Distance"):
                    arr = fd1.GetArray(name)
                    if arr is not None and arr.GetNumberOfTuples() >= 1:
                        dist_candidates.append(float(arr.GetTuple1(0)))

    if not dist_candidates:
        # last resort: look for any 1-tuple numeric field array
        if fd is not None:
            for i in range(fd.GetNumberOfArrays()):
                arr = fd.GetAbstractArray(i)
                if arr is None:
                    continue
                if hasattr(arr, "GetNumberOfTuples") and arr.GetNumberOfTuples() == 1:
                    try:
                        dist_candidates.append(float(arr.GetTuple1(0)))
                    except Exception:
                        continue

    if not dist_candidates:
        raise RuntimeError("Could not extract bottleneck distance from ttkBottleneckDistance output")

    dist = dist_candidates[0]
    if debug:
        print(f"[debug] bottleneck extracted={dist}")
    return float(dist)


def _load_mt_port(path: str, port: int):
    if path.endswith(".vti"):
        return _read_vti(path)
    return _read_vtu(path)


def _build_merge_tree_stack(
    gt_ports: Dict[int, str],
    sr_ports: Dict[int, str],
    debug: bool = False,
) -> "vtk.vtkMultiBlockDataSet":
    """
    Build the exact MultiBlock layout expected by ttkMergeTreeDistanceMatrix.

    IMPORTANT: ttkMergeTreeDistanceMatrix expects a "stacked by port" layout:
        top-level blocks = ports (0=nodes, 1=arcs, [2=segmentation optional])
        each port-block is a MultiBlock whose blocks are the trees (GT, SR, ...)

    For two trees:
        top[0] = MultiBlock( nodesGT, nodesSR )
        top[1] = MultiBlock( arcsGT,  arcsSR  )
        (optionally) top[2] = MultiBlock( segGT, segSR )
    """
    # We only need ports 0 & 1 for distance; port 2 can be omitted.
    needed = [0, 1]
    for p in needed:
        if p not in gt_ports or p not in sr_ports:
            raise RuntimeError(f"Missing merge-tree port {p} for GT/SR (need ports 0 and 1).")

    top = vtk.vtkMultiBlockDataSet()
    top.SetNumberOfBlocks(2)

    # port 0 (nodes)
    mb0 = vtk.vtkMultiBlockDataSet()
    mb0.SetNumberOfBlocks(2)
    a0 = _load_mt_port(gt_ports[0], 0)
    b0 = _load_mt_port(sr_ports[0], 0)
    mb0.SetBlock(0, a0)
    mb0.SetBlock(1, b0)
    top.SetBlock(0, mb0)

    # port 1 (arcs)
    mb1 = vtk.vtkMultiBlockDataSet()
    mb1.SetNumberOfBlocks(2)
    a1 = _load_mt_port(gt_ports[1], 1)
    b1 = _load_mt_port(sr_ports[1], 1)
    mb1.SetBlock(0, a1)
    mb1.SetBlock(1, b1)
    top.SetBlock(1, mb1)

    if debug:
        print(f"[debug] MT stack built: top blocks={top.GetNumberOfBlocks()} (ports 0&1), each with 2 trees")
    return top


def _extract_distance_from_vtk_table(tbl) -> float:
    """
    Robustly extract the off-diagonal value from a 2x2 distance matrix vtkTable.
    """
    if tbl is None or tbl.GetNumberOfRows() < 2:
        raise RuntimeError("DistanceMatrix output is empty (expected at least 2 rows).")
    # Collect numeric columns (skip string id columns if any)
    numeric_cols = []
    for ci in range(tbl.GetNumberOfColumns()):
        col = tbl.GetColumn(ci)
        if col is None:
            continue
        # vtkStringArray has GetValue but not GetTuple1; numeric arrays do.
        if hasattr(col, "GetTuple1"):
            numeric_cols.append(ci)

    if not numeric_cols:
        raise RuntimeError("No numeric columns found in vtkTable output.")

    # Build matrix: rows x numeric_cols
    mat = []
    for ri in range(tbl.GetNumberOfRows()):
        row = []
        for ci in numeric_cols:
            col = tbl.GetColumn(ci)
            try:
                row.append(float(col.GetTuple1(ri)))
            except Exception:
                row.append(float("nan"))
        mat.append(row)

    # Off-diagonal for 2x2 is [0][1] in most cases; but numeric_cols may exclude a leading label col.
    # If there are >=2 numeric cols, use mat[0][1] (second numeric col).
    if len(numeric_cols) >= 2 and len(mat) >= 2:
        d = mat[0][1]
        if not math.isnan(d):
            return float(d)

    # Fallback: pick the first finite, non-zero off-diagonal candidate.
    candidates = []
    for i in range(min(10, len(mat))):
        for j in range(min(10, len(mat[i]))):
            if i == j:
                continue
            v = mat[i][j]
            if math.isfinite(v) and v != 0.0:
                candidates.append(v)
    if not candidates:
        # maybe the metric genuinely returns 0; accept first finite off-diagonal
        for i in range(min(10, len(mat))):
            for j in range(min(10, len(mat[i]))):
                if i == j:
                    continue
                v = mat[i][j]
                if math.isfinite(v):
                    candidates.append(v)
    if not candidates:
        raise RuntimeError("Could not find any finite distance value in vtkTable output.")
    return float(candidates[0])


def compute_mt_distance_inprocess(
    gt_ports: Dict[int, str],
    sr_ports: Dict[int, str],
    debug: bool = False,
) -> float:
    """
    Merge tree distance between GT and SR using ttkMergeTreeDistanceMatrix.
    """
    stack = _build_merge_tree_stack(gt_ports, sr_ports, debug=debug)

    flt = ttk.ttkMergeTreeDistanceMatrix()
    flt.SetInputDataObject(0, stack)

    # Some builds expose knobs; set conservative defaults if present
    for meth, val in (
        ("SetThreadNumber", 1),
        ("SetAssignmentSolver", 0),
        ("SetAssignmentAlgorithm", 0),
        ("SetEpsilon", 1e-7),
    ):
        if hasattr(flt, meth):
            try:
                getattr(flt, meth)(val)
            except Exception:
                pass

    flt.Update()

    out0 = flt.GetOutputDataObject(0)
    # Most builds output a vtkTable directly
    if out0 is not None and out0.IsA("vtkTable"):
        return _extract_distance_from_vtk_table(out0)

    # Some builds wrap it inside a multiblock
    if out0 is not None and out0.IsA("vtkMultiBlockDataSet"):
        mb = out0
        # try first block that is a vtkTable
        for i in range(mb.GetNumberOfBlocks()):
            b = mb.GetBlock(i)
            if b is not None and b.IsA("vtkTable"):
                return _extract_distance_from_vtk_table(b)

    raise RuntimeError(f"Unexpected output type from ttkMergeTreeDistanceMatrix: {out0.GetClassName() if out0 else 'None'}")


def compute_mt_distance_isolated(
    gt_ports: Dict[int, str],
    sr_ports: Dict[int, str],
    debug: bool = False,
) -> float:
    """
    Compute MT distance in a child process so a segfault doesn't kill the whole batch.
    """
    args = [
        sys.executable, __file__, "--_mt_worker",
        "--gt0", gt_ports.get(0, ""), "--gt1", gt_ports.get(1, ""),
        "--sr0", sr_ports.get(0, ""), "--sr1", sr_ports.get(1, ""),
    ]
    if debug:
        args.append("--debug")
    cp = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if cp.returncode != 0:
        if debug:
            print("[debug] mt worker failed rc=", cp.returncode)
            print(cp.stderr)
        return float("nan")
    try:
        return float(cp.stdout.strip().splitlines()[-1])
    except Exception:
        if debug:
            print("[debug] mt worker output parse failed:")
            print(cp.stdout)
            print(cp.stderr)
        return float("nan")


def _index_files(root: Path, regex: re.Pattern) -> List[Tuple[Tuple[str, str, str], Dict[int, str]]]:
    """
    Build an index mapping (method, label, sample) -> {port: path}
    for both PD and MT file patterns.
    """
    tmp: Dict[Tuple[str, str, str], Dict[int, str]] = {}
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        m = regex.match(p.name)
        if not m:
            continue
        method = m.group("method")
        label = m.group("label")
        sample = m.group("sample")
        port = int(m.group("port"))
        key = (method, label, sample)
        tmp.setdefault(key, {})[port] = str(p)
    return list(tmp.items())


def _pair_keys(pd_index: Dict[Tuple[str, str], Dict[str, Dict[int, str]]], max_items: Optional[int] = None) -> List[Tuple[str, str]]:
    # pd_index: (method, sample) -> {"GT": {port:path}, "SR": {port:path}}
    keys = sorted(pd_index.keys())
    if max_items is not None:
        keys = keys[:max_items]
    return keys


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pd-dir", required=False, help="Directory containing PD .vtu outputs")
    ap.add_argument("--mt-dir", required=False, help="Directory containing MT outputs (ports 0,1,...)")
    ap.add_argument("--outdir", required=True, help="Output directory for CSVs and logs")
    ap.add_argument("--max", type=int, default=None, help="Limit number of (method,sample) pairs for quick tests")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--no-mt", action="store_true", help="Skip merge tree distances")
    ap.add_argument("--no-pd", action="store_true", help="Skip persistence diagram distances")
    ap.add_argument("--isolate-mt", action="store_true", help="Compute each MT distance in a subprocess (survives segfaults)")
    ap.add_argument("--_mt_worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--gt0", default="", help=argparse.SUPPRESS)
    ap.add_argument("--gt1", default="", help=argparse.SUPPRESS)
    ap.add_argument("--sr0", default="", help=argparse.SUPPRESS)
    ap.add_argument("--sr1", default="", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    if args._mt_worker:
        gt_ports = {0: args.gt0, 1: args.gt1}
        sr_ports = {0: args.sr0, 1: args.sr1}
        d = compute_mt_distance_inprocess(gt_ports, sr_ports, debug=args.debug)
        print(d)
        return 0

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ------------------ Index PD files ------------------
    pd_pairs: Dict[Tuple[str, str], Dict[str, Dict[int, str]]] = {}
    if not args.no_pd:
        if not args.pd_dir:
            raise SystemExit("--pd-dir is required unless --no-pd is used")
        pd_root = Path(args.pd_dir)
        for (method, label, sample), ports in _index_files(pd_root, PD_RE):
            pd_pairs.setdefault((method, sample), {})[label] = ports

    # ------------------ Index MT files ------------------
    mt_pairs: Dict[Tuple[str, str], Dict[str, Dict[int, str]]] = {}
    if not args.no_mt:
        if not args.mt_dir:
            raise SystemExit("--mt-dir is required unless --no-mt is used")
        mt_root = Path(args.mt_dir)
        for (method, label, sample), ports in _index_files(mt_root, MT_RE):
            mt_pairs.setdefault((method, sample), {})[label] = ports

    # Build the intersection of keys
    keys = set()
    if not args.no_pd:
        keys |= set(pd_pairs.keys())
    if not args.no_mt:
        keys |= set(mt_pairs.keys())
    keys = sorted(keys)

    if args.max is not None:
        keys = keys[: args.max]

    if args.debug:
        print(f"[debug] discovered {len(keys)} keys")

    # ------------------ Compute distances ------------------
    pd_rows: List[Dict[str, object]] = []
    mt_rows: List[Dict[str, object]] = []

    for method, sample in keys:
        if not args.no_pd:
            if (method, sample) in pd_pairs and "GT" in pd_pairs[(method, sample)] and "SR" in pd_pairs[(method, sample)]:
                gt_ports = pd_pairs[(method, sample)]["GT"]
                sr_ports = pd_pairs[(method, sample)]["SR"]
                # compute for all shared PD ports
                common_ports = sorted(set(gt_ports.keys()) & set(sr_ports.keys()))
                row = {"method": method, "sample": sample}
                for p in common_ports:
                    try:
                        d = compute_pd_bottleneck(gt_ports[p], sr_ports[p], debug=args.debug)
                        row[f"pd_bottleneck_port{p}"] = d
                        if args.debug:
                            print(f"[debug] PD bottleneck {Path(gt_ports[p]).name} vs {Path(sr_ports[p]).name} = {d}")
                    except Exception as e:
                        row[f"pd_bottleneck_port{p}"] = float("nan")
                        if args.debug:
                            print(f"[debug] PD failed for ({method},{sample},port={p}): {e}")
                pd_rows.append(row)

        if not args.no_mt:
            if (method, sample) in mt_pairs and "GT" in mt_pairs[(method, sample)] and "SR" in mt_pairs[(method, sample)]:
                gt_ports = mt_pairs[(method, sample)]["GT"]
                sr_ports = mt_pairs[(method, sample)]["SR"]
                try:
                    if args.isolate_mt:
                        d = compute_mt_distance_isolated(gt_ports, sr_ports, debug=args.debug)
                    else:
                        d = compute_mt_distance_inprocess(gt_ports, sr_ports, debug=args.debug)
                    mt_rows.append({"method": method, "sample": sample, "mt_distance": d})
                    if args.debug:
                        print(f"[debug] MT distance {method} {sample} = {d}")
                except Exception as e:
                    mt_rows.append({"method": method, "sample": sample, "mt_distance": float('nan')})
                    if args.debug:
                        print(f"[debug] MT failed for ({method},{sample}): {e}")

    # ------------------ Write outputs ------------------
    if pd_rows:
        pd_path = outdir / "pd_pairwise_distances.csv"
        with pd_path.open("w", newline="") as f:
            cols = sorted({k for r in pd_rows for k in r.keys()})
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in pd_rows:
                w.writerow(r)
        if args.debug:
            print(f"[debug] wrote {pd_path}")

    if mt_rows:
        mt_path = outdir / "mt_pairwise_distances.csv"
        with mt_path.open("w", newline="") as f:
            cols = ["method", "sample", "mt_distance"]
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in mt_rows:
                w.writerow(r)
        if args.debug:
            print(f"[debug] wrote {mt_path}")

    # ------------------ Summaries by method ------------------
    def _stats(vals: List[float]) -> Dict[str, float]:
        finite = [v for v in vals if v is not None and math.isfinite(v)]
        if not finite:
            return {"n": 0, "mean": float("nan"), "median": float("nan"), "min": float("nan"), "max": float("nan")}
        finite.sort()
        n = len(finite)
        mean = sum(finite) / n
        med = finite[n // 2] if n % 2 == 1 else 0.5 * (finite[n // 2 - 1] + finite[n // 2])
        return {"n": n, "mean": mean, "median": med, "min": finite[0], "max": finite[-1]}

    # PD summary (use port0 if present, else first pd_bottleneck_port*)
    if pd_rows:
        by_method: Dict[str, List[float]] = {}
        for r in pd_rows:
            m = str(r["method"])
            # prefer port0
            v = None
            if "pd_bottleneck_port0" in r:
                v = r["pd_bottleneck_port0"]
            else:
                for k in sorted(r.keys()):
                    if k.startswith("pd_bottleneck_port"):
                        v = r[k]
                        break
            try:
                v = float(v)
            except Exception:
                v = float("nan")
            by_method.setdefault(m, []).append(v)

        pd_sum_path = outdir / "pd_summary_by_method.csv"
        with pd_sum_path.open("w", newline="") as f:
            cols = ["method", "n", "mean", "median", "min", "max"]
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for m, vals in sorted(by_method.items()):
                s = _stats(vals)
                w.writerow({"method": m, **s})
        if args.debug:
            print(f"[debug] wrote {pd_sum_path}")

    if mt_rows:
        by_method2: Dict[str, List[float]] = {}
        for r in mt_rows:
            by_method2.setdefault(str(r["method"]), []).append(float(r["mt_distance"]))
        mt_sum_path = outdir / "mt_summary_by_method.csv"
        with mt_sum_path.open("w", newline="") as f:
            cols = ["method", "n", "mean", "median", "min", "max"]
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for m, vals in sorted(by_method2.items()):
                s = _stats(vals)
                w.writerow({"method": m, **s})
        if args.debug:
            print(f"[debug] wrote {mt_sum_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
