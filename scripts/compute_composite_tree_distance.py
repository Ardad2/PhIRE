#!/usr/bin/env python3
"""compute_composite_tree_distance_real.py

Compute **real** distances (not proxies) between GT and SR outputs using TTK:

- Persistence diagram distance: `ttkBottleneckDistance` (bottleneck distance).
- Merge tree distance: `ttkMergeTreeDistanceMatrix` (Wasserstein-type MT distance).

This script is designed for your file layout where:
  - PD outputs are VTU files (unstructured grids) produced by TTK.
  - Merge tree outputs are produced as multiple "ports" per sample:
      *_mt_port_0.vtu, *_mt_port_1.vtu, *_mt_port_2.vti (segmentation)
    and the merge tree distance filter expects a vtkMultiBlockDataSet.

Outputs (written to --outdir):
  - pd_pairwise_distances.csv
  - mt_pairwise_distances.csv
  - phase_c_results.csv
  - pd_summary_by_method.csv
  - mt_summary_by_method.csv
  - phase_c_summary.csv
  - phase_c_summary.txt

Example:
  python3 compute_composite_tree_distance_real.py \
    --pd-dir /home/adadhwal/PhIRE/ttk_outputs/direct/pd_vtu \
    --mt-dir /home/adadhwal/PhIRE/ttk_outputs/mt \
    --outdir /home/adadhwal/PhIRE/ttk_outputs/phase_c_final \
    --max 5 --debug

Notes:
- If your TTK build is correctly matched to your VTK (9.6.0), imports should work.
- This script avoids numpy/pandas; it uses only stdlib + VTK/TTK.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import vtk  # type: ignore
import topologytoolkit as ttk  # type: ignore


# -------------------------------
# Parsing / discovery
# -------------------------------

_LABEL_RE = re.compile(r"_(GT|SR)_")


@dataclass(frozen=True)
class PDEntry:
    key: str
    method: str
    label: str  # GT or SR
    path: Path


@dataclass(frozen=True)
class MTEntry:
    key: str
    method: str
    label: str  # GT or SR
    port: int
    path: Path


def _extract_method_label_and_tail(stem: str) -> Optional[Tuple[str, str, str]]:
    """Return (method, label, tail) from a file stem that contains _GT_ or _SR_."""
    m = _LABEL_RE.search(stem)
    if not m:
        return None
    label = m.group(1)
    method = stem[: m.start()]  # everything before _GT_/_SR_
    tail = stem[m.end() :]  # everything after
    if not method or not tail:
        return None
    return method, label, tail


def discover_pd(pd_dir: Path) -> Dict[Tuple[str, str], PDEntry]:
    """Find PD VTU files and map (key,label)->entry.

    If multiple ports exist, prefer port_0, else lowest port number.
    """
    best: Dict[Tuple[str, str], Tuple[int, PDEntry]] = {}

    for p in pd_dir.rglob("*.vtu"):
        stem = p.name
        if "_pd" not in stem:
            continue

        # strip any trailing .vtu_port_X.vtu decorations for parsing
        # e.g. gan_GT_..._pd.vtu_port_0.vtu
        base = stem
        if base.endswith(".vtu"):
            base = base[:-4]
        base = re.sub(r"\.vtu_port_\d+$", "", base)
        base = re.sub(r"_port_\d+$", "", base)

        parsed = _extract_method_label_and_tail(base)
        if not parsed:
            continue
        method, label, tail = parsed

        # tail still contains something like s0_..._pd.vtu OR ..._pd
        tail = re.sub(r"_pd(\.vtu)?$", "", tail)
        key = f"{method}_{tail}"

        # pick port preference
        port_m = re.search(r"port_(\d+)", stem)
        port = int(port_m.group(1)) if port_m else 0

        entry = PDEntry(key=key, method=method, label=label, path=p)
        k = (key, label)
        prev = best.get(k)
        if prev is None:
            best[k] = (port, entry)
        else:
            prev_port, _ = prev
            # prefer port 0, then lower port
            if (port == 0 and prev_port != 0) or (port < prev_port):
                best[k] = (port, entry)

    return {k: v for k, (_, v) in best.items()}


def discover_mt(mt_dir: Path) -> Dict[Tuple[str, str], Dict[int, MTEntry]]:
    """Find MT port files (vtu/vti) and map (key,label)->{port:entry}."""
    out: Dict[Tuple[str, str], Dict[int, MTEntry]] = {}

    for p in mt_dir.rglob("*"):
        if not p.is_file():
            continue
        if "_mt_port_" not in p.name:
            continue
        if p.suffix not in {".vtu", ".vti", ".vtm", ".vtp", ".vtk"}:
            continue

        base = p.name
        # strip extension for parsing
        stem = base
        if stem.endswith(p.suffix):
            stem = stem[: -len(p.suffix)]

        # remove trailing _mt_port_X
        mport = re.search(r"_mt_port_(\d+)$", stem)
        if not mport:
            continue
        port = int(mport.group(1))
        stem_noport = stem[: mport.start()]

        parsed = _extract_method_label_and_tail(stem_noport)
        if not parsed:
            continue
        method, label, tail = parsed
        tail = re.sub(r"_mt$", "", tail)
        key = f"{method}_{tail}"

        entry = MTEntry(key=key, method=method, label=label, port=port, path=p)
        k = (key, label)
        out.setdefault(k, {})[port] = entry

    return out


# -------------------------------
# VTK readers / dataset helpers
# -------------------------------


def read_dataset(path: Path) -> vtk.vtkDataObject:
    """Read a VTK XML dataset based on extension."""
    suf = path.suffix.lower()

    if suf == ".vtu":
        r = vtk.vtkXMLUnstructuredGridReader()
    elif suf == ".vti":
        r = vtk.vtkXMLImageDataReader()
    elif suf == ".vtp":
        r = vtk.vtkXMLPolyDataReader()
    elif suf == ".vtm":
        r = vtk.vtkXMLMultiBlockDataReader()
    elif suf == ".vtk":
        r = vtk.vtkDataSetReader()
    else:
        raise ValueError(f"Unsupported extension: {path}")

    r.SetFileName(str(path))
    r.Update()
    out = r.GetOutputDataObject(0)
    if out is None:
        raise RuntimeError(f"VTK reader produced None for {path}")
    return out


# -------------------------------
# Distance computations
# -------------------------------


def _get_field_distance(ds: vtk.vtkDataObject) -> Optional[float]:
    if not hasattr(ds, "GetFieldData"):
        return None
    fd = ds.GetFieldData()
    if fd is None:
        return None
    for name in ("BottleneckDistance", "WassersteinDistance", "Distance"):
        arr = fd.GetArray(name)
        if arr is not None and arr.GetNumberOfTuples() > 0:
            try:
                return float(arr.GetTuple1(0))
            except Exception:
                pass
    return None


def compute_pd_bottleneck(pd_gt: Path, pd_sr: Path, debug: bool = False) -> float:
    """Compute bottleneck distance between two PD VTU files."""
    ug_gt = read_dataset(pd_gt)
    ug_sr = read_dataset(pd_sr)

    # BottleneckDistance expects a vtkMultiBlockDataSet with two blocks.
    mb = vtk.vtkMultiBlockDataSet()
    mb.SetNumberOfBlocks(2)
    mb.SetBlock(0, ug_gt)
    mb.SetBlock(1, ug_sr)

    bd = ttk.ttkBottleneckDistance()

    # REQUIRED in recent TTK: choose the assignment algorithm implementation.
    # In TTK source, PVAlgorithm==0 corresponds to the "ttk" backend.
    if hasattr(bd, "SetPVAlgorithm"):
        bd.SetPVAlgorithm(0)
    if hasattr(bd, "SetDistanceAlgorithm"):
        try:
            bd.SetDistanceAlgorithm("ttk")
        except Exception:
            pass

    # Make sure we get the matching output (port 1) where distances are stored in field data.
    if hasattr(bd, "SetUseOutputMatching"):
        bd.SetUseOutputMatching(1)

    # Prefer bottleneck distance (Wasserstein infinity) when available.
    # Different TTK versions expose this differently.
    if hasattr(bd, "SetWassersteinMetric"):
        try:
            bd.SetWassersteinMetric(-1)  # often encodes infinity
        except Exception:
            try:
                bd.SetWassersteinMetric(0)  # fallback to bottleneck in some builds
            except Exception:
                pass

    # Set input (single port)
    if hasattr(bd, "SetInputDataObject"):
        bd.SetInputDataObject(0, mb)
    else:
        bd.SetInputData(mb)

    bd.Update()

    # Distance is saved in output matchings field data
    out_match = bd.GetOutputDataObject(1)
    dist = _get_field_distance(out_match)
    if dist is None:
        # sometimes stored on port 0
        dist = _get_field_distance(bd.GetOutputDataObject(0))

    if dist is None:
        raise RuntimeError("Could not read distance from ttkBottleneckDistance outputs")

    if debug:
        print(f"[debug] PD bottleneck {pd_gt.name} vs {pd_sr.name} = {dist}")

    return float(dist)


def _read_port_dataset(port_map: Dict[int, MTEntry], port: int, debug: bool = False) -> Optional[vtk.vtkDataObject]:
    e = port_map.get(port)
    if e is None:
        return None
    ds = read_dataset(e.path)
    if debug:
        print(f"[debug] mt port={port} type={ds.GetClassName()} path={e.path.name}")
    return ds


def _build_mt_stack_port_major(
    gt_ports: Dict[int, MTEntry], sr_ports: Dict[int, MTEntry], debug: bool = False
) -> vtk.vtkMultiBlockDataSet:
    """Build merge-tree stack in a port-major layout.

    top[0] -> mb(gt_port0, sr_port0)
    top[1] -> mb(gt_port1, sr_port1)
    top[2] -> mb(gt_port2, sr_port2) optional
    """
    gt0 = _read_port_dataset(gt_ports, 0, debug=debug)
    sr0 = _read_port_dataset(sr_ports, 0, debug=debug)
    gt1 = _read_port_dataset(gt_ports, 1, debug=debug)
    sr1 = _read_port_dataset(sr_ports, 1, debug=debug)

    if gt0 is None or sr0 is None or gt1 is None or sr1 is None:
        raise RuntimeError("Missing required merge tree ports 0 and 1 for GT/SR")

    top = vtk.vtkMultiBlockDataSet()
    top.SetNumberOfBlocks(3)

    mb0 = vtk.vtkMultiBlockDataSet()
    mb0.SetNumberOfBlocks(2)
    mb0.SetBlock(0, gt0)
    mb0.SetBlock(1, sr0)
    top.SetBlock(0, mb0)

    mb1 = vtk.vtkMultiBlockDataSet()
    mb1.SetNumberOfBlocks(2)
    mb1.SetBlock(0, gt1)
    mb1.SetBlock(1, sr1)
    top.SetBlock(1, mb1)

    gt2 = _read_port_dataset(gt_ports, 2, debug=debug)
    sr2 = _read_port_dataset(sr_ports, 2, debug=debug)
    if gt2 is not None and sr2 is not None:
        mb2 = vtk.vtkMultiBlockDataSet()
        mb2.SetNumberOfBlocks(2)
        mb2.SetBlock(0, gt2)
        mb2.SetBlock(1, sr2)
        top.SetBlock(2, mb2)

    return top


def _build_mt_stack_sample_major(
    gt_ports: Dict[int, MTEntry], sr_ports: Dict[int, MTEntry], debug: bool = False
) -> vtk.vtkMultiBlockDataSet:
    """Build merge-tree stack in a sample-major layout.

    trees[0] -> mb(gt_port0, gt_port1[, gt_port2])
    trees[1] -> mb(sr_port0, sr_port1[, sr_port2])
    """
    gt0 = _read_port_dataset(gt_ports, 0, debug=debug)
    sr0 = _read_port_dataset(sr_ports, 0, debug=debug)
    gt1 = _read_port_dataset(gt_ports, 1, debug=debug)
    sr1 = _read_port_dataset(sr_ports, 1, debug=debug)

    if gt0 is None or sr0 is None or gt1 is None or sr1 is None:
        raise RuntimeError("Missing required merge tree ports 0 and 1 for GT/SR")

    gt_tree = vtk.vtkMultiBlockDataSet()
    gt_tree.SetNumberOfBlocks(3)
    gt_tree.SetBlock(0, gt0)
    gt_tree.SetBlock(1, gt1)

    sr_tree = vtk.vtkMultiBlockDataSet()
    sr_tree.SetNumberOfBlocks(3)
    sr_tree.SetBlock(0, sr0)
    sr_tree.SetBlock(1, sr1)

    gt2 = _read_port_dataset(gt_ports, 2, debug=debug)
    sr2 = _read_port_dataset(sr_ports, 2, debug=debug)
    if gt2 is not None and sr2 is not None:
        gt_tree.SetBlock(2, gt2)
        sr_tree.SetBlock(2, sr2)

    trees = vtk.vtkMultiBlockDataSet()
    trees.SetNumberOfBlocks(2)
    trees.SetBlock(0, gt_tree)
    trees.SetBlock(1, sr_tree)
    return trees


def _table_value(table: vtk.vtkTable, row: int, col: int) -> Optional[float]:
    if table is None:
        return None
    if row < 0 or col < 0:
        return None
    if table.GetNumberOfRows() <= row or table.GetNumberOfColumns() <= col:
        return None
    try:
        v = table.GetValue(row, col)
        try:
            return float(v.ToDouble())
        except Exception:
            return float(str(v))
    except Exception:
        return None


def _extract_mt_distance_from_table(table: vtk.vtkTable) -> Optional[float]:
    # Preferred index for 2x2 matrix output
    d = _table_value(table, 0, 1)
    if d is not None:
        return d

    # Some builds use named columns (Tree0/Tree1) with rows as tree ids.
    if table is not None:
        for col_name in ("Tree0", "tree0", "Tree1", "tree1"):
            col = table.GetColumnByName(col_name)
            if col is not None and col.GetNumberOfTuples() > 1:
                try:
                    return float(col.GetTuple1(1))
                except Exception:
                    pass

    # Diagonal fallback (often zero).
    return _table_value(table, 0, 0)


def _compute_mt_distance_once(
    mt_gt_ports: Dict[int, MTEntry],
    mt_sr_ports: Dict[int, MTEntry],
    layout: str,
    debug: bool = False,
) -> float:
    if layout == "port":
        stack = _build_mt_stack_port_major(mt_gt_ports, mt_sr_ports, debug=debug)
    elif layout == "sample":
        stack = _build_mt_stack_sample_major(mt_gt_ports, mt_sr_ports, debug=debug)
    else:
        raise ValueError(f"Unknown MT layout: {layout}")

    flt = ttk.ttkMergeTreeDistanceMatrix()
    if hasattr(flt, "SetInputDataObject"):
        flt.SetInputDataObject(0, stack)
    else:
        flt.SetInputData(0, stack)

    for meth, val in (
        ("SetOutputDistanceMatrix", 1),
        ("SetComputeDistanceMatrix", 1),
        ("SetThreadNumber", 1),
    ):
        if hasattr(flt, meth):
            try:
                getattr(flt, meth)(val)
            except Exception:
                pass

    flt.Update()
    out0 = flt.GetOutputDataObject(0)

    if isinstance(out0, vtk.vtkTable):
        dist = _extract_mt_distance_from_table(out0)
        if dist is not None:
            if debug:
                print(f"[debug] MT distance ({layout} layout) = {dist}")
            return float(dist)

    if isinstance(out0, vtk.vtkMultiBlockDataSet):
        for i in range(out0.GetNumberOfBlocks()):
            blk = out0.GetBlock(i)
            if isinstance(blk, vtk.vtkTable):
                dist = _extract_mt_distance_from_table(blk)
                if dist is not None:
                    if debug:
                        print(f"[debug] MT distance ({layout} layout wrapped table[{i}]) = {dist}")
                    return float(dist)

    dist = _get_field_distance(out0)
    if dist is not None:
        return float(dist)

    raise RuntimeError(f"Could not extract MT distance from output type {out0.GetClassName()}")


def _run_mt_worker(
    mt_gt_ports: Dict[int, MTEntry], mt_sr_ports: Dict[int, MTEntry], layout: str, debug: bool = False
) -> Tuple[Optional[float], str]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--_mt-worker",
        "--worker-layout",
        layout,
        "--gt-port0",
        str(mt_gt_ports[0].path),
        "--gt-port1",
        str(mt_gt_ports[1].path),
        "--sr-port0",
        str(mt_sr_ports[0].path),
        "--sr-port1",
        str(mt_sr_ports[1].path),
    ]
    if 2 in mt_gt_ports and 2 in mt_sr_ports:
        cmd.extend(["--gt-port2", str(mt_gt_ports[2].path), "--sr-port2", str(mt_sr_ports[2].path)])

    env = os.environ.copy()
    env.setdefault("PYTHONFAULTHANDLER", "1")
    cp = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    if cp.returncode != 0:
        msg = f"worker layout={layout} failed rc={cp.returncode}"
        if debug:
            print(f"[debug] {msg}")
            if cp.stderr.strip():
                print(cp.stderr.strip())
        return None, msg

    lines = [ln for ln in cp.stdout.splitlines() if ln.strip()]
    if not lines:
        return None, f"worker layout={layout} returned empty output"
    try:
        payload = json.loads(lines[-1])
    except Exception:
        return None, f"worker layout={layout} returned non-json output"

    if not payload.get("ok", False):
        return None, f"worker layout={layout} error: {payload.get('error', 'unknown')}"

    try:
        return float(payload["distance"]), ""
    except Exception:
        return None, f"worker layout={layout} returned invalid distance"


def compute_mt_distance(
    mt_gt_ports: Dict[int, MTEntry],
    mt_sr_ports: Dict[int, MTEntry],
    debug: bool = False,
    isolate_mt: bool = False,
) -> float:
    if 0 not in mt_gt_ports or 1 not in mt_gt_ports or 0 not in mt_sr_ports or 1 not in mt_sr_ports:
        raise RuntimeError("Missing required MT ports 0/1 for GT/SR")

    layouts = ["port", "sample"]

    if not isolate_mt:
        # In-process: try the historically safer layout first.
        return _compute_mt_distance_once(mt_gt_ports, mt_sr_ports, layout="port", debug=debug)

    errs: List[str] = []
    for layout in layouts:
        dist, err = _run_mt_worker(mt_gt_ports, mt_sr_ports, layout=layout, debug=debug)
        if dist is not None:
            return dist
        errs.append(err)

    raise RuntimeError("; ".join(e for e in errs if e))


# -------------------------------
# Summaries / output
# -------------------------------


def _stats(vals: List[float]) -> Dict[str, float]:
    if not vals:
        return {"n": 0}
    v = sorted(vals)
    n = len(v)
    mean = sum(v) / n
    median = v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])
    stdev = statistics.pstdev(v) if n >= 2 else 0.0
    return {
        "n": n,
        "mean": mean,
        "median": median,
        "stdev": stdev,
        "min": v[0],
        "max": v[-1],
    }


def write_csv(path: Path, fieldnames: List[str], rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pd-dir", type=Path, required=False)
    ap.add_argument("--mt-dir", type=Path, required=False)
    ap.add_argument("--outdir", type=Path, required=False)
    ap.add_argument("--max", type=int, default=0, help="limit number of keys (for quick testing)")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--isolate-mt", action="store_true", help="Run MT distance in a worker process; retry multiple layouts.")
    ap.add_argument("--_mt-worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--worker-layout", choices=["port", "sample"], default="port", help=argparse.SUPPRESS)
    ap.add_argument("--gt-port0", type=Path, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--gt-port1", type=Path, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--gt-port2", type=Path, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--sr-port0", type=Path, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--sr-port1", type=Path, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--sr-port2", type=Path, default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args._mt_worker:
        try:
            if args.gt_port0 is None or args.gt_port1 is None or args.sr_port0 is None or args.sr_port1 is None:
                raise RuntimeError("Missing required worker ports")
            gt_ports = {
                0: MTEntry(key="worker", method="worker", label="GT", port=0, path=args.gt_port0),
                1: MTEntry(key="worker", method="worker", label="GT", port=1, path=args.gt_port1),
            }
            sr_ports = {
                0: MTEntry(key="worker", method="worker", label="SR", port=0, path=args.sr_port0),
                1: MTEntry(key="worker", method="worker", label="SR", port=1, path=args.sr_port1),
            }
            if args.gt_port2 is not None and args.sr_port2 is not None:
                gt_ports[2] = MTEntry(key="worker", method="worker", label="GT", port=2, path=args.gt_port2)
                sr_ports[2] = MTEntry(key="worker", method="worker", label="SR", port=2, path=args.sr_port2)

            d = _compute_mt_distance_once(gt_ports, sr_ports, layout=args.worker_layout, debug=args.debug)
            print(json.dumps({"ok": True, "distance": d}))
            return 0
        except Exception as e:
            print(json.dumps({"ok": False, "error": str(e)}))
            return 2

    if args.pd_dir is None or args.mt_dir is None or args.outdir is None:
        ap.error("--pd-dir, --mt-dir, and --outdir are required unless running --_mt-worker")

    pd_map = discover_pd(args.pd_dir)
    mt_map = discover_mt(args.mt_dir)

    # Determine keys that have both GT and SR PDs, and also MT port maps.
    keys = sorted({k for (k, lab) in pd_map.keys()})
    keys = [k for k in keys if (k, "GT") in pd_map and (k, "SR") in pd_map]

    # For MT we require both GT and SR entries exist
    keys = [k for k in keys if (k, "GT") in mt_map and (k, "SR") in mt_map]

    if args.max and args.max > 0:
        keys = keys[: args.max]

    if args.debug:
        print(f"[debug] discovered {len(keys)} keys")

    rows_results: List[Dict[str, object]] = []
    rows_pd: List[Dict[str, object]] = []
    rows_mt: List[Dict[str, object]] = []

    for key in keys:
        pd_gt = pd_map[(key, "GT")]
        pd_sr = pd_map[(key, "SR")]
        mt_gt_ports = mt_map[(key, "GT")]
        mt_sr_ports = mt_map[(key, "SR")]

        method = pd_gt.method
        pd_dist: Optional[float] = None
        mt_dist: Optional[float] = None
        err: str = ""

        try:
            pd_dist = compute_pd_bottleneck(pd_gt.path, pd_sr.path, debug=args.debug)
        except Exception as e:
            err = f"pd:{e}"

        try:
            mt_dist = compute_mt_distance(mt_gt_ports, mt_sr_ports, debug=args.debug, isolate_mt=args.isolate_mt)
        except Exception as e:
            err = (err + "; " if err else "") + f"mt:{e}"

        if pd_dist is not None:
            rows_pd.append({"key": key, "method": method, "pd_distance": pd_dist})
        if mt_dist is not None:
            rows_mt.append({"key": key, "method": method, "mt_distance": mt_dist})

        rows_results.append(
            {
                "key": key,
                "method": method,
                "pd_distance": pd_dist if pd_dist is not None else "",
                "mt_distance": mt_dist if mt_dist is not None else "",
                "error": err,
            }
        )

    outdir = args.outdir
    write_csv(outdir / "pd_pairwise_distances.csv", ["key", "method", "pd_distance"], rows_pd)
    write_csv(outdir / "mt_pairwise_distances.csv", ["key", "method", "mt_distance"], rows_mt)
    write_csv(outdir / "phase_c_results.csv", ["key", "method", "pd_distance", "mt_distance", "error"], rows_results)

    # Summaries by method
    pd_by_method: Dict[str, List[float]] = {}
    for r in rows_pd:
        pd_by_method.setdefault(str(r["method"]), []).append(float(r["pd_distance"]))

    mt_by_method: Dict[str, List[float]] = {}
    for r in rows_mt:
        mt_by_method.setdefault(str(r["method"]), []).append(float(r["mt_distance"]))

    pd_summary_rows: List[Dict[str, object]] = []
    for m, vals in sorted(pd_by_method.items()):
        s = _stats(vals)
        pd_summary_rows.append({"method": m, **s})

    mt_summary_rows: List[Dict[str, object]] = []
    for m, vals in sorted(mt_by_method.items()):
        s = _stats(vals)
        mt_summary_rows.append({"method": m, **s})

    write_csv(outdir / "pd_summary_by_method.csv", ["method", "n", "mean", "median", "stdev", "min", "max"], pd_summary_rows)
    write_csv(outdir / "mt_summary_by_method.csv", ["method", "n", "mean", "median", "stdev", "min", "max"], mt_summary_rows)

    # Combined summary
    combined_rows: List[Dict[str, object]] = []
    methods = sorted(set(pd_by_method.keys()) | set(mt_by_method.keys()))
    for m in methods:
        row: Dict[str, object] = {"method": m}
        s_pd = _stats(pd_by_method.get(m, []))
        s_mt = _stats(mt_by_method.get(m, []))
        row.update({f"pd_{k}": v for k, v in s_pd.items()})
        row.update({f"mt_{k}": v for k, v in s_mt.items()})
        combined_rows.append(row)

    # stable field order
    comb_fields = ["method", "pd_n", "pd_mean", "pd_median", "pd_stdev", "pd_min", "pd_max", "mt_n", "mt_mean", "mt_median", "mt_stdev", "mt_min", "mt_max"]
    write_csv(outdir / "phase_c_summary.csv", comb_fields, combined_rows)

    # Text report
    report_lines: List[str] = []
    report_lines.append("Phase C — REAL distances (TTK)\n")
    report_lines.append(f"Rows computed: {len(rows_results)}")
    report_lines.append(f"PD distances:  {len(rows_pd)}")
    report_lines.append(f"MT distances:  {len(rows_mt)}\n")

    for m in methods:
        s_pd = _stats(pd_by_method.get(m, []))
        s_mt = _stats(mt_by_method.get(m, []))
        report_lines.append(f"[{m}]")
        if s_pd.get("n", 0):
            report_lines.append(f"  PD  n={s_pd['n']} mean={s_pd['mean']:.6g} median={s_pd['median']:.6g} stdev={s_pd['stdev']:.6g}")
        else:
            report_lines.append("  PD  n=0")
        if s_mt.get("n", 0):
            report_lines.append(f"  MT  n={s_mt['n']} mean={s_mt['mean']:.6g} median={s_mt['median']:.6g} stdev={s_mt['stdev']:.6g}")
        else:
            report_lines.append("  MT  n=0")
        report_lines.append("")

    (outdir / "phase_c_summary.txt").write_text("\n".join(report_lines))

    print(f"Wrote: {outdir/'phase_c_results.csv'}")
    print(f"Wrote: {outdir/'pd_pairwise_distances.csv'}")
    print(f"Wrote: {outdir/'mt_pairwise_distances.csv'}")
    print(f"Wrote: {outdir/'phase_c_summary.csv'}")
    print(f"Wrote: {outdir/'phase_c_summary.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
