#!/usr/bin/env python3
"""Experimental alternate topology-distance pipeline (kept separate).

This script mirrors the existing Phase-C pairing logic but uses alternate TTK filters:
- PD: ttkPersistenceDiagramDistanceMatrix
- MT: ttkMergeTreeDistance (fallback/error if unavailable)

It is intended for comparison runs against the main pipeline outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import vtk  # type: ignore
import topologytoolkit as ttk  # type: ignore

_LABEL_RE = re.compile(r"_(GT|SR)_")


@dataclass(frozen=True)
class PDEntry:
    key: str
    method: str
    label: str
    path: Path


@dataclass(frozen=True)
class MTEntry:
    key: str
    method: str
    label: str
    port: int
    path: Path


def _extract_method_label_and_tail(stem: str) -> Optional[Tuple[str, str, str]]:
    m = _LABEL_RE.search(stem)
    if not m:
        return None
    label = m.group(1)
    method = stem[: m.start()]
    tail = stem[m.end() :]
    if not method or not tail:
        return None
    return method, label, tail


def discover_pd(pd_dir: Path) -> Dict[Tuple[str, str], PDEntry]:
    best: Dict[Tuple[str, str], Tuple[int, PDEntry]] = {}
    for p in pd_dir.rglob("*.vtu"):
        stem = p.name
        if "_pd" not in stem:
            continue
        base = stem[:-4] if stem.endswith(".vtu") else stem
        base = re.sub(r"\.vtu_port_\d+$", "", base)
        base = re.sub(r"_port_\d+$", "", base)
        parsed = _extract_method_label_and_tail(base)
        if not parsed:
            continue
        method, label, tail = parsed
        tail = re.sub(r"_pd(\.vtu)?$", "", tail)
        key = f"{method}_{tail}"
        m_port = re.search(r"port_(\d+)", stem)
        port = int(m_port.group(1)) if m_port else 0
        entry = PDEntry(key=key, method=method, label=label, path=p)
        k = (key, label)
        prev = best.get(k)
        if prev is None or (port == 0 and prev[0] != 0) or (port < prev[0]):
            best[k] = (port, entry)
    return {k: v for k, (_, v) in best.items()}


def discover_mt(mt_dir: Path) -> Dict[Tuple[str, str], Dict[int, MTEntry]]:
    out: Dict[Tuple[str, str], Dict[int, MTEntry]] = {}
    for p in mt_dir.rglob("*"):
        if not p.is_file() or "_mt_port_" not in p.name:
            continue
        if p.suffix not in {".vtu", ".vti", ".vtm", ".vtp", ".vtk"}:
            continue
        stem = p.name[: -len(p.suffix)]
        m_port = re.search(r"_mt_port_(\d+)$", stem)
        if not m_port:
            continue
        port = int(m_port.group(1))
        stem_noport = stem[: m_port.start()]
        parsed = _extract_method_label_and_tail(stem_noport)
        if not parsed:
            continue
        method, label, tail = parsed
        key = f"{method}_{tail}"
        out.setdefault((key, label), {})[port] = MTEntry(key=key, method=method, label=label, port=port, path=p)
    return out


def read_dataset(path: Path) -> vtk.vtkDataObject:
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
        raise RuntimeError(f"Reader produced None for {path}")
    return out


def _table_value(table: vtk.vtkTable, row: int, col: int) -> Optional[float]:
    if row < 0 or row >= table.GetNumberOfRows() or col < 0 or col >= table.GetNumberOfColumns():
        return None
    arr = table.GetColumn(col)
    if arr is None:
        return None
    try:
        v = float(arr.GetTuple1(row))
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


def _extract_distance_from_table(table: vtk.vtkTable) -> Optional[float]:
    nrows = table.GetNumberOfRows()
    ncols = table.GetNumberOfColumns()
    if nrows == 0 or ncols == 0:
        return None
    candidates = [(0, 1), (1, 0), (0, 0), (1, 1)]
    for r, c in candidates:
        v = _table_value(table, r, c)
        if v is not None:
            return v
    for r in range(nrows):
        for c in range(ncols):
            v = _table_value(table, r, c)
            if v is not None:
                return v
    return None


def compute_pd_distance_matrix(pd_gt: Path, pd_sr: Path) -> float:
    ug_gt = read_dataset(pd_gt)
    ug_sr = read_dataset(pd_sr)
    mb = vtk.vtkMultiBlockDataSet()
    mb.SetNumberOfBlocks(2)
    mb.SetBlock(0, ug_gt)
    mb.SetBlock(1, ug_sr)

    flt = ttk.ttkPersistenceDiagramDistanceMatrix()
    if hasattr(flt, "SetInputDataObject"):
        flt.SetInputDataObject(0, mb)
    else:
        flt.SetInputData(0, mb)
    if hasattr(flt, "SetThreadNumber"):
        flt.SetThreadNumber(1)
    flt.Update()

    out0 = flt.GetOutputDataObject(0)
    if isinstance(out0, vtk.vtkTable):
        dist = _extract_distance_from_table(out0)
        if dist is not None:
            return dist
    if isinstance(out0, vtk.vtkMultiBlockDataSet):
        for i in range(out0.GetNumberOfBlocks()):
            blk = out0.GetBlock(i)
            if isinstance(blk, vtk.vtkTable):
                dist = _extract_distance_from_table(blk)
                if dist is not None:
                    return dist
    raise RuntimeError("Could not extract PD distance from ttkPersistenceDiagramDistanceMatrix")


def _make_tree(port_map: Dict[int, MTEntry]) -> vtk.vtkMultiBlockDataSet:
    if 0 not in port_map or 1 not in port_map:
        raise RuntimeError("Missing required MT ports 0 and 1")
    tree = vtk.vtkMultiBlockDataSet()
    tree.SetNumberOfBlocks(3)
    tree.SetBlock(0, read_dataset(port_map[0].path))
    tree.SetBlock(1, read_dataset(port_map[1].path))
    if 2 in port_map:
        tree.SetBlock(2, read_dataset(port_map[2].path))
    return tree


def _compute_mt_distance_direct(mt_gt_ports: Dict[int, MTEntry], mt_sr_ports: Dict[int, MTEntry]) -> float:
    if not hasattr(ttk, "ttkMergeTreeDistance"):
        raise RuntimeError("This TTK build has no ttkMergeTreeDistance python wrapper")

    gt_tree = _make_tree(mt_gt_ports)
    sr_tree = _make_tree(mt_sr_ports)

    flt = ttk.ttkMergeTreeDistance()

    # Try the most common two-input wiring first.
    wired = False
    if hasattr(flt, "SetInputDataObject"):
        try:
            flt.SetInputDataObject(0, gt_tree)
            flt.SetInputDataObject(1, sr_tree)
            wired = True
        except Exception:
            wired = False
    if not wired and hasattr(flt, "SetInputData"):
        try:
            flt.SetInputData(0, gt_tree)
            flt.SetInputData(1, sr_tree)
            wired = True
        except Exception:
            wired = False
    if not wired:
        raise RuntimeError("Failed to wire ttkMergeTreeDistance inputs")

    if hasattr(flt, "SetThreadNumber"):
        flt.SetThreadNumber(1)
    flt.Update()

    out0 = flt.GetOutputDataObject(0)
    if isinstance(out0, vtk.vtkTable):
        dist = _extract_distance_from_table(out0)
        if dist is not None:
            return dist

    # Fallback: some builds may expose a scalar getter.
    for getter in ("GetDistance", "GetOutputDistance", "Getresult"):
        if hasattr(flt, getter):
            try:
                return float(getattr(flt, getter)())
            except Exception:
                pass

    raise RuntimeError("Could not extract MT distance from ttkMergeTreeDistance")


def _run_mt_worker(mt_gt_ports: Dict[int, MTEntry], mt_sr_ports: Dict[int, MTEntry], debug: bool = False) -> Tuple[Optional[float], str]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--_mt-worker",
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
        detail = cp.stderr.strip()
        # Worker errors are commonly emitted as JSON on stdout.
        lines = [ln for ln in cp.stdout.splitlines() if ln.strip()]
        if lines:
            try:
                payload = json.loads(lines[-1])
                if isinstance(payload, dict) and payload.get("error"):
                    detail = str(payload.get("error"))
            except Exception:
                if not detail:
                    detail = lines[-1]
        if debug:
            print(f"[debug] alt MT worker failed rc={cp.returncode}: {detail}")
        return None, f"worker failed rc={cp.returncode}: {detail}"
    lines = [ln for ln in cp.stdout.splitlines() if ln.strip()]
    if not lines:
        return None, "worker returned empty output"
    try:
        payload = json.loads(lines[-1])
    except Exception:
        return None, "worker returned non-json"
    if not payload.get("ok", False):
        return None, str(payload.get("error", "unknown worker error"))
    try:
        return float(payload["distance"]), ""
    except Exception:
        return None, "worker returned invalid distance"


def write_csv(path: Path, fields: List[str], rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Compute alternate TTK distances for comparison experiments")
    ap.add_argument("--pd-dir", type=Path, required=False)
    ap.add_argument("--mt-dir", type=Path, required=False)
    # Keep optional at argparse-level so hidden worker mode can run without it.
    # We validate --outdir explicitly in normal (non-worker) mode below.
    ap.add_argument("--outdir", type=Path, required=False)
    ap.add_argument("--max", type=int, default=0)
    ap.add_argument("--debug", action="store_true")
    mt_mode = ap.add_mutually_exclusive_group()
    mt_mode.add_argument("--isolate-mt", dest="isolate_mt", action="store_true", help="Run MT in worker subprocesses (default)")
    mt_mode.add_argument("--no-isolate-mt", dest="isolate_mt", action="store_false", help="Run MT direct in-process")
    ap.set_defaults(isolate_mt=True)

    # worker mode
    ap.add_argument("--_mt-worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--gt-port0", type=Path, default=Path(""), help=argparse.SUPPRESS)
    ap.add_argument("--gt-port1", type=Path, default=Path(""), help=argparse.SUPPRESS)
    ap.add_argument("--gt-port2", type=Path, default=Path(""), help=argparse.SUPPRESS)
    ap.add_argument("--sr-port0", type=Path, default=Path(""), help=argparse.SUPPRESS)
    ap.add_argument("--sr-port1", type=Path, default=Path(""), help=argparse.SUPPRESS)
    ap.add_argument("--sr-port2", type=Path, default=Path(""), help=argparse.SUPPRESS)

    args = ap.parse_args(argv)

    if args._mt_worker:
        try:
            gt_ports: Dict[int, MTEntry] = {
                0: MTEntry("", "", "GT", 0, args.gt_port0),
                1: MTEntry("", "", "GT", 1, args.gt_port1),
            }
            sr_ports: Dict[int, MTEntry] = {
                0: MTEntry("", "", "SR", 0, args.sr_port0),
                1: MTEntry("", "", "SR", 1, args.sr_port1),
            }
            if str(args.gt_port2):
                gt_ports[2] = MTEntry("", "", "GT", 2, args.gt_port2)
            if str(args.sr_port2):
                sr_ports[2] = MTEntry("", "", "SR", 2, args.sr_port2)
            d = _compute_mt_distance_direct(gt_ports, sr_ports)
            print(json.dumps({"ok": True, "distance": d}))
            return 0
        except Exception as e:
            print(json.dumps({"ok": False, "error": str(e)}))
            return 2

    if args.pd_dir is None or args.mt_dir is None or args.outdir is None:
        raise SystemExit("--pd-dir, --mt-dir, and --outdir are required in normal mode")

    pd_map = discover_pd(args.pd_dir)
    mt_map = discover_mt(args.mt_dir)

    keys = sorted({k for (k, lab) in pd_map.keys()})
    keys = [k for k in keys if (k, "GT") in pd_map and (k, "SR") in pd_map]
    keys = [k for k in keys if (k, "GT") in mt_map and (k, "SR") in mt_map]
    if args.max and args.max > 0:
        keys = keys[: args.max]

    rows_results: List[Dict[str, object]] = []
    rows_pd: List[Dict[str, object]] = []
    rows_mt: List[Dict[str, object]] = []

    if args.debug:
        print(f"[debug] discovered {len(keys)} keys")

    for key in keys:
        pd_gt = pd_map[(key, "GT")]
        pd_sr = pd_map[(key, "SR")]
        mt_gt_ports = mt_map[(key, "GT")]
        mt_sr_ports = mt_map[(key, "SR")]
        method = pd_gt.method

        pd_dist: Optional[float] = None
        mt_dist: Optional[float] = None
        err = ""

        try:
            pd_dist = compute_pd_distance_matrix(pd_gt.path, pd_sr.path)
            if args.debug:
                print(f"[debug] alt PD matrix {pd_gt.path.name} vs {pd_sr.path.name} = {pd_dist}")
        except Exception as e:
            err = f"pd:{e}"

        try:
            if args.isolate_mt:
                mt_dist, mt_err = _run_mt_worker(mt_gt_ports, mt_sr_ports, debug=args.debug)
                if mt_dist is None:
                    raise RuntimeError(mt_err)
            else:
                mt_dist = _compute_mt_distance_direct(mt_gt_ports, mt_sr_ports)
            if args.debug:
                print(f"[debug] alt MT direct {key} = {mt_dist}")
        except Exception as e:
            err = (err + "; " if err else "") + f"mt:{e}"

        if pd_dist is not None:
            rows_pd.append({"key": key, "method": method, "pd_distance_alt": pd_dist})
        if mt_dist is not None:
            rows_mt.append({"key": key, "method": method, "mt_distance_alt": mt_dist})

        rows_results.append(
            {
                "key": key,
                "method": method,
                "pd_distance_alt": "" if pd_dist is None else pd_dist,
                "mt_distance_alt": "" if mt_dist is None else mt_dist,
                "error": err,
            }
        )

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    write_csv(outdir / "alt_pd_pairwise_distances.csv", ["key", "method", "pd_distance_alt"], rows_pd)
    write_csv(outdir / "alt_mt_pairwise_distances.csv", ["key", "method", "mt_distance_alt"], rows_mt)
    write_csv(
        outdir / "alt_phase_c_results.csv",
        ["key", "method", "pd_distance_alt", "mt_distance_alt", "error"],
        rows_results,
    )

    print(f"Wrote: {outdir/'alt_pd_pairwise_distances.csv'}")
    print(f"Wrote: {outdir/'alt_mt_pairwise_distances.csv'}")
    print(f"Wrote: {outdir/'alt_phase_c_results.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
