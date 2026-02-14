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
import re
import statistics
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


def make_multiblock_from_ports(port_map: Dict[int, MTEntry], debug: bool = False) -> Optional[vtk.vtkMultiBlockDataSet]:
    """Build a vtkMultiBlockDataSet in port order (0,1,2,...) from a set of mt_port files.

    Heuristics:
      - Must contain at least one VTU (unstructured grid) (ports 0/1 usually).
      - If only a port_2.vti exists, we skip (that's typically segmentation image only).
    """
    if not port_map:
        return None

    # Ensure we have at least one unstructured grid
    has_vtu = any(e.path.suffix.lower() == ".vtu" for e in port_map.values())
    if not has_vtu:
        if debug:
            print(f"[debug] merge tree ports exist but no .vtu blocks: {[str(e.path) for e in port_map.values()]}")
        return None

    ports_sorted = sorted(port_map.keys())

    mb = vtk.vtkMultiBlockDataSet()
    mb.SetNumberOfBlocks(len(ports_sorted))

    for i, port in enumerate(ports_sorted):
        e = port_map[port]
        ds = read_dataset(e.path)
        mb.SetBlock(i, ds)
        # put a readable name
        md = mb.GetMetaData(i)
        if md is not None:
            md.Set(vtk.vtkCompositeDataSet.NAME(), f"port_{port}")

        if debug:
            print(f"[debug] mt block[{i}] port={port} type={ds.GetClassName()} path={e.path.name}")

    return mb


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


def compute_mt_distance(mt_gt_ports: Dict[int, MTEntry], mt_sr_ports: Dict[int, MTEntry], debug: bool = False) -> float:
    """Compute merge tree distance between two samples.

    We build vtkMultiBlockDataSet objects from the available ports (0/1/2...).
    """
    mb_gt = make_multiblock_from_ports(mt_gt_ports, debug=debug)
    mb_sr = make_multiblock_from_ports(mt_sr_ports, debug=debug)
    if mb_gt is None or mb_sr is None:
        raise RuntimeError("Missing required merge tree port blocks (.vtu) for MT distance")

    flt = ttk.ttkMergeTreeDistanceMatrix()

    # Connect as two ports when available.
    # Most builds have 2 ports (0: set A trees, 1: set B trees).
    used_two_ports = False
    if hasattr(flt, "SetInputDataObject"):
        try:
            flt.SetInputDataObject(0, mb_gt)
            flt.SetInputDataObject(1, mb_sr)
            used_two_ports = True
        except Exception:
            used_two_ports = False

    if not used_two_ports:
        # Fallback: many-vtk-connection style on port 0
        try:
            flt.AddInputDataObject(0, mb_gt)
            flt.AddInputDataObject(0, mb_sr)
        except Exception:
            flt.SetInputData(mb_gt)

    # Some versions require explicitly enabling computation
    for meth, val in (
        ("SetOutputDistanceMatrix", 1),
        ("SetComputeDistanceMatrix", 1),
    ):
        if hasattr(flt, meth):
            try:
                getattr(flt, meth)(val)
            except Exception:
                pass

    flt.Update()

    out0 = flt.GetOutputDataObject(0)

    # Common case: vtkTable distance matrix
    if isinstance(out0, vtk.vtkTable):
        if out0.GetNumberOfRows() < 1 or out0.GetNumberOfColumns() < 1:
            raise RuntimeError("MergeTreeDistanceMatrix produced empty vtkTable")
        v = out0.GetValue(0, 0)
        try:
            dist = float(v.ToDouble())
        except Exception:
            dist = float(str(v))
        if debug:
            print(f"[debug] MT distance (table[0,0]) = {dist}")
        return dist

    # Fallback: look for distance in field data
    dist = _get_field_distance(out0)
    if dist is None and hasattr(flt, "GetOutputDataObject"):
        for i in range(1, 4):
            try:
                dist = _get_field_distance(flt.GetOutputDataObject(i))
            except Exception:
                pass
            if dist is not None:
                break

    if dist is None:
        raise RuntimeError(f"Could not extract MT distance from output type {out0.GetClassName()}")

    if debug:
        print(f"[debug] MT distance = {dist}")

    return float(dist)


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
    ap.add_argument("--pd-dir", type=Path, required=True)
    ap.add_argument("--mt-dir", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--max", type=int, default=0, help="limit number of keys (for quick testing)")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

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
            mt_dist = compute_mt_distance(mt_gt_ports, mt_sr_ports, debug=args.debug)
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
