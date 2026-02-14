#!/usr/bin/env python3
"""
compute_composite_tree_distance_ttk.py

Compute *actual* TTK distances between:
  - Persistence diagrams (PD) stored as .vtu (vtkUnstructuredGrid)
  - Merge trees (MT) stored as .vtu (vtkUnstructuredGrid)

Outputs a CSV with one row per matched sample key.

Typical usage (auto-split GT/SR by filename containing "GT" / "SR"):

  python compute_composite_tree_distance_ttk.py \
    --pd-dir /home/adadhwal/PhIRE/ttk_outputs/direct/pd_vtu \
    --mt-dir /home/adadhwal/PhIRE/ttk_outputs/mt \
    --out distances_ttk.csv

If your PD/MT are in separate folders:

  python compute_composite_tree_distance_ttk.py \
    --pd-gt /path/to/pd/GT --pd-sr /path/to/pd/SR \
    --mt-gt /path/to/mt/GT --mt-sr /path/to/mt/SR \
    --out distances_ttk.csv

Notes
-----
- This script does NOT use any "feature-count" proxy. It runs TTK distance filters.
- It tries multiple filters and extracts the distance robustly from the output.
- Requires: VTK + TTK python bindings in the same Python environment (your VTK 9.6 + rebuilt TTK).

"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# --------------------------
# Import VTK + TTK
# --------------------------
try:
    import vtk  # type: ignore
except Exception as e:
    print("ERROR: failed to import vtk:", e, file=sys.stderr)
    sys.exit(2)

try:
    import topologytoolkit as ttk  # type: ignore
except Exception as e:
    print("ERROR: failed to import topologytoolkit (TTK):", e, file=sys.stderr)
    sys.exit(3)


@dataclass(frozen=True)
class SampleFiles:
    key: str
    label_gt: str
    label_sr: str
    pd_gt: Optional[Path] = None
    pd_sr: Optional[Path] = None
    mt_gt: Optional[Path] = None
    mt_sr: Optional[Path] = None


def _normalize_name_for_key(name: str) -> str:
    """Normalize filenames so GT/SR pairing is stable.

    We strip common ParaView/TTK save-data suffixes and file extensions.
    """
    # Remove common suffixes like .vtu_port_0.vtu etc.
    name = re.sub(r"\.vtu_port_\d+\.vtu$", ".vtu", name)
    # Strip one of the common VTK XML extensions.
    name = re.sub(r"\.(vtu|vtm|vtmb|vti|vtp)$", "", name)
    return name


def parse_label_and_key(path: Path, kind: str) -> Optional[Tuple[str, str]]:
    """
    Extract (label, key) from filename, where label is GT or SR and key is the sample id.

    Works for names like:
      gan_GT_s0_speed_p160_x0_y0_pd.vtu_port_0.vtu
      <prefix>SR_<whatever>_mt.vtu
    """
    stem = _normalize_name_for_key(path.name)

    # Make sure we remove the kind suffix to isolate the sample key.
    # e.g. "..._pd" or "..._mt"
    if f"_{kind}" in stem:
        stem = stem.split(f"_{kind}")[0]

    # Detect GT/SR
    m = re.search(r"^(?P<prefix>.*?)(?P<label>GT|SR)_(?P<key>.+)$", stem)
    if not m:
        # also handle "..._GT_...": sometimes GT/SR is in middle
        m2 = re.search(r"^(?P<prefix>.*?)[_\-](?P<label>GT|SR)_(?P<key>.+)$", stem)
        if not m2:
            return None
        label = m2.group("label")
        key = m2.group("key")
        return label, key

    label = m.group("label")
    key = m.group("key")
    return label, key


def discover_vtk_files(root: Path) -> List[Path]:
    """Discover VTK XML files we might read (.vtu/.vtm/.vti/etc.)."""
    if not root.exists():
        return []
    exts = (".vtu", ".vtm", ".vtmb", ".vti", ".vtp")
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix in exts]
    # Also catch ParaView's annoying ".vtu_port_0.vtu" naming.
    files += [p for p in root.rglob("*.vtu_port_*.vtu") if p.is_file()]
    return sorted(set(files))


def read_data_object(path: Path) -> "vtk.vtkDataObject":
    """Read a VTK XML file into a vtkDataObject."""
    ext = path.suffix.lower()
    if path.name.endswith(".vtu"):
        r = vtk.vtkXMLUnstructuredGridReader()
    elif ext in (".vtm", ".vtmb"):
        r = vtk.vtkXMLMultiBlockDataReader()
    elif ext == ".vti":
        r = vtk.vtkXMLImageDataReader()
    elif ext == ".vtp":
        r = vtk.vtkXMLPolyDataReader()
    else:
        # Best-effort fallback.
        r = vtk.vtkXMLGenericDataObjectReader()
    r.SetFileName(str(path))
    r.Update()
    out = r.GetOutputDataObject(0)
    if out is None:
        raise RuntimeError(f"Failed to read: {path}")
    return out


def read_unstructured_grid(path: Path) -> "vtk.vtkUnstructuredGrid":
    obj = read_data_object(path)
    if obj is None or not obj.IsA("vtkUnstructuredGrid"):
        raise TypeError(f"Expected vtkUnstructuredGrid from {path}, got {type(obj).__name__}")
    return obj  # type: ignore[return-value]


def read_multiblock(path: Path) -> "vtk.vtkMultiBlockDataSet":
    obj = read_data_object(path)
    if obj is None or not obj.IsA("vtkMultiBlockDataSet"):
        raise TypeError(f"Expected vtkMultiBlockDataSet from {path}, got {type(obj).__name__}")
    return obj  # type: ignore[return-value]


def read_unstructured_grid(path: Path) -> "vtk.vtkUnstructuredGrid":
    r = vtk.vtkXMLUnstructuredGridReader()
    r.SetFileName(str(path))
    r.Update()
    ug = r.GetOutput()
    if ug is None:
        raise ValueError(f"VTU read failed (output is None): {path}")
    if ug.GetNumberOfPoints() == 0:
        # still return, but signal likely failed; caller can decide
        raise ValueError(f"VTU read returned 0 points: {path}")
    return ug


def _try_extract_distance_from_table(tbl: "vtk.vtkTable") -> Optional[float]:
    nrows = tbl.GetNumberOfRows()
    ncols = tbl.GetNumberOfColumns()
    if nrows == 0 or ncols == 0:
        return None

    # 1) If it's a 2x2 or NxN matrix, take (0,1) if possible
    if nrows >= 2 and ncols >= 2:
        try:
            v = tbl.GetValue(0, 1)
            return float(v.ToDouble())
        except Exception:
            pass

    # 2) Look for a column that smells like "distance"
    for ci in range(ncols):
        cname = tbl.GetColumnName(ci) or ""
        if "dist" in cname.lower():
            try:
                v = tbl.GetValue(0, ci)
                return float(v.ToDouble())
            except Exception:
                continue

    # 3) Fallback: first cell
    try:
        v = tbl.GetValue(0, 0)
        return float(v.ToDouble())
    except Exception:
        return None


def _try_extract_distance_from_fielddata(fd: "vtk.vtkFieldData") -> Optional[float]:
    if fd is None:
        return None
    for ai in range(fd.GetNumberOfArrays()):
        arr = fd.GetArray(ai)
        if arr is None:
            continue
        name = (arr.GetName() or "").lower()
        if "dist" in name or "wasser" in name or "bottleneck" in name:
            try:
                if arr.GetNumberOfTuples() >= 1:
                    return float(arr.GetTuple1(0))
            except Exception:
                pass
    return None


def extract_distance_from_algorithm(algo: "vtk.vtkAlgorithm", debug: bool = False) -> Optional[float]:
    """
    Try to extract a single distance value from a TTK algorithm output.

    We try:
      - vtkTable outputs (common for distance matrices)
      - FieldData arrays containing "Distance"/"Wasserstein"/"Bottleneck"
      - First numeric array in FieldData as fallback
    """
    nports = getattr(algo, "GetNumberOfOutputPorts", lambda: 1)()
    for pi in range(max(1, nports)):
        try:
            obj = algo.GetOutputDataObject(pi)
        except Exception:
            obj = None
        if obj is None:
            continue

        # vtkTable?
        if obj.IsA("vtkTable"):
            d = _try_extract_distance_from_table(obj)
            if d is not None:
                return d

        # Try FieldData
        try:
            fd = obj.GetFieldData()
            d = _try_extract_distance_from_fielddata(fd)
            if d is not None:
                return d
        except Exception:
            pass

        if debug:
            try:
                print(f"[debug] output port {pi}: {obj.GetClassName()}", file=sys.stderr)
                fd = obj.GetFieldData()
                if fd:
                    names = []
                    for ai in range(fd.GetNumberOfArrays()):
                        a = fd.GetArray(ai)
                        if a:
                            names.append(a.GetName())
                    print(f"[debug] field arrays: {names}", file=sys.stderr)
            except Exception:
                pass

    return None


def compute_pd_distance(pd_gt: Path, pd_sr: Path, debug: bool = False) -> float:
    """
    Compute PD distance between 2 PD VTUs.

    Preference order:
      1) ttkBottleneckDistance (pairwise, **single** multiblock input with 2 PD blocks)
      2) ttkPersistenceDiagramDistanceMatrix (multiblock -> distance matrix)
      3) ttkPersistenceDiagramClustering (multiblock -> distances often in field data)

    Returns a single float distance.
    """
    ug_gt = read_unstructured_grid(pd_gt)
    ug_sr = read_unstructured_grid(pd_sr)

    # 1) Bottleneck distance (most direct if available)
    # NOTE: ttkBottleneckDistance has **one** input port and expects a vtkMultiBlockDataSet
    # with *at least* two vtkUnstructuredGrid blocks (the 2 persistence diagrams).
    if hasattr(ttk, "ttkBottleneckDistance"):
        mb_pair = vtk.vtkMultiBlockDataSet()
        mb_pair.SetNumberOfBlocks(2)
        mb_pair.SetBlock(0, ug_gt)
        mb_pair.SetBlock(1, ug_sr)

        f = ttk.ttkBottleneckDistance()

        # Force bottleneck metric ("inf") where possible
        if hasattr(f, "SetWassersteinMetric"):
            try:
                f.SetWassersteinMetric("inf")
            except Exception:
                pass
        elif hasattr(f, "SetWassersteinMetricToInf"):
            try:
                f.SetWassersteinMetricToInf()
            except Exception:
                pass

        if hasattr(f, "SetInputDataObject"):
            f.SetInputDataObject(mb_pair)
        else:
            f.SetInputData(mb_pair)

        f.Update()

        # ttkBottleneckDistance stores the distance in the FieldData of output port #1
        # (the matchings `vtkUnstructuredGrid`) as an array named either
        # "BottleneckDistance" (if metric is inf) or "WassersteinDistance".
        match = None
        try:
            match = f.GetOutputDataObject(1)
        except Exception:
            pass
        if match is None:
            try:
                match = f.GetOutput(1)
            except Exception:
                pass

        if match is not None and hasattr(match, "GetFieldData"):
            fd = match.GetFieldData()
            if fd is not None:
                for nm in ("BottleneckDistance", "WassersteinDistance"):
                    arr = fd.GetArray(nm)
                    if arr is not None and arr.GetNumberOfTuples() > 0:
                        return float(arr.GetTuple1(0))

        # Fallback: heuristic extraction (covers odd wrapper variations)
        d = extract_distance_from_algorithm(f, debug=debug)
        if d is not None:
            return float(d)

    # 2) Distance matrix on a multiblock
    if hasattr(ttk, "ttkPersistenceDiagramDistanceMatrix"):
        mb = vtk.vtkMultiBlockDataSet()
        mb.SetNumberOfBlocks(2)
        mb.SetBlock(0, ug_gt)
        mb.SetBlock(1, ug_sr)

        f = ttk.ttkPersistenceDiagramDistanceMatrix()
        if hasattr(f, "SetInputDataObject"):
            f.SetInputDataObject(mb)
        else:
            f.SetInputData(mb)
        f.Update()
        d = extract_distance_from_algorithm(f, debug=debug)
        if d is not None:
            return float(d)

    # 3) Clustering (also computes distances in many pipelines)
    if hasattr(ttk, "ttkPersistenceDiagramClustering"):
        mb = vtk.vtkMultiBlockDataSet()
        mb.SetNumberOfBlocks(2)
        mb.SetBlock(0, ug_gt)
        mb.SetBlock(1, ug_sr)

        f = ttk.ttkPersistenceDiagramClustering()
        if hasattr(f, "SetInputDataObject"):
            f.SetInputDataObject(mb)
        else:
            f.SetInputData(mb)
        f.Update()
        d = extract_distance_from_algorithm(f, debug=debug)
        if d is not None:
            return float(d)

    raise RuntimeError("Could not compute PD distance: no suitable TTK PD distance filter found or could not extract distance.")


def compute_mt_distance(mt_gt: Path, mt_sr: Path, debug: bool = False) -> float:
    """
    Compute merge tree distance between 2 merge tree VTUs.

    Preference order:
      1) ttkMergeTreeDistanceMatrix (multiblock -> distance matrix)
      2) ttkMergeTreeClustering (multiblock -> distances in field data)
    """
    mt_gt_obj = read_data_object(mt_gt)
    mt_sr_obj = read_data_object(mt_sr)

    # ttkMergeTreeDistanceMatrix expects a vtkMultiBlockDataSet where each BLOCK is
    # itself a vtkMultiBlockDataSet representing a merge tree produced by ttkMergeTree.
    def as_merge_tree_mb(obj: vtk.vtkDataObject, src: Path) -> vtk.vtkMultiBlockDataSet:
        if obj is None:
            raise RuntimeError(f"Failed to read merge tree file: {src}")
        if obj.IsA("vtkMultiBlockDataSet"):
            return vtk.vtkMultiBlockDataSet.SafeDownCast(obj)
        # Fallback: wrap single dataset as a 1-block multiblock.
        if debug:
            print(f"[debug] merge tree {src.name} is {obj.GetClassName()} (expected vtkMultiBlockDataSet); wrapping as 1-block multiblock", file=sys.stderr)
        mb = vtk.vtkMultiBlockDataSet()
        mb.SetNumberOfBlocks(1)
        mb.SetBlock(0, obj)
        return mb

    tree_gt = as_merge_tree_mb(mt_gt_obj, mt_gt)
    tree_sr = as_merge_tree_mb(mt_sr_obj, mt_sr)

    trees = vtk.vtkMultiBlockDataSet()
    trees.SetNumberOfBlocks(2)
    trees.SetBlock(0, tree_gt)
    trees.SetBlock(1, tree_sr)

    def extract_01_from_table(table: vtk.vtkTable) -> Optional[float]:
        if table is None or (not table.IsA("vtkTable")):
            return None
        if table.GetNumberOfRows() < 2 or table.GetNumberOfColumns() < 2:
            return None
        # Tables are stored as columns (vtkAbstractArray). Value at row i, col j.
        col1 = table.GetColumn(1)
        if col1 is None or col1.GetNumberOfTuples() < 1:
            return None
        try:
            return float(col1.GetTuple1(0))
        except Exception:
            try:
                return float(col1.GetValue(0))
            except Exception:
                return None

    if hasattr(ttk, "ttkMergeTreeDistanceMatrix"):
        f = ttk.ttkMergeTreeDistanceMatrix()
        # Use only input port 0 (single list) -> 2x2 matrix. We take entry [0,1].
        if hasattr(f, "SetInputDataObject"):
            f.SetInputDataObject(0, trees)
        else:
            f.SetInputData(0, trees)
        f.Update()

        out0 = None
        for getter in ("GetOutput", "GetOutputDataObject"):
            if hasattr(f, getter):
                try:
                    out0 = getattr(f, getter)(0)
                    break
                except Exception:
                    pass
        if out0 is None and hasattr(f, "GetOutput"):
            try:
                out0 = f.GetOutput()
            except Exception:
                out0 = None

        d01 = extract_01_from_table(out0) if (out0 is not None and out0.IsA("vtkTable")) else None
        if d01 is not None:
            return float(d01)

        # fallback: heuristic extractor (may pick wrong entry)
        d = extract_distance_from_algorithm(f, debug=debug)
        if d is not None:
            return float(d)

    if hasattr(ttk, "ttkMergeTreeClustering"):
        f = ttk.ttkMergeTreeClustering()
        if hasattr(f, "SetInputDataObject"):
            f.SetInputDataObject(trees)
        else:
            f.SetInputData(trees)
        f.Update()
        d = extract_distance_from_algorithm(f, debug=debug)
        if d is not None:
            return float(d)

    raise RuntimeError("Could not compute merge tree distance: no suitable TTK merge tree distance filter found or could not extract distance.")


def build_pairs_from_dirs(
    pd_dir: Optional[Path],
    mt_dir: Optional[Path],
    pd_gt: Optional[Path],
    pd_sr: Optional[Path],
    mt_gt: Optional[Path],
    mt_sr: Optional[Path],
    debug: bool = False,
) -> Dict[str, SampleFiles]:
    """
    Build mapping: sample_key -> SampleFiles, discovering PD/MT files either from combined dirs
    (auto split by GT/SR naming) or from explicit GT/SR dirs.
    """
    pairs: Dict[str, SampleFiles] = {}

    def upsert(label: str, key: str, kind: str, path: Path) -> None:
        if key not in pairs:
            pairs[key] = SampleFiles(key=key, label_gt="GT", label_sr="SR")
        cur = pairs[key]
        if kind == "pd":
            if label == "GT":
                pairs[key] = SampleFiles(**{**cur.__dict__, "pd_gt": path})
            else:
                pairs[key] = SampleFiles(**{**cur.__dict__, "pd_sr": path})
        elif kind == "mt":
            if label == "GT":
                pairs[key] = SampleFiles(**{**cur.__dict__, "mt_gt": path})
            else:
                pairs[key] = SampleFiles(**{**cur.__dict__, "mt_sr": path})

    # PD discovery
    if pd_gt and pd_sr:
        for p in discover_vtk_files(pd_gt):
            lk = parse_label_and_key(p, "pd")
            label = "GT"
            key = lk[1] if lk else _normalize_name_for_key(p.name)
            upsert(label, key, "pd", p)
        for p in discover_vtk_files(pd_sr):
            lk = parse_label_and_key(p, "pd")
            label = "SR"
            key = lk[1] if lk else _normalize_name_for_key(p.name)
            upsert(label, key, "pd", p)
    elif pd_dir:
        for p in discover_vtk_files(pd_dir):
            lk = parse_label_and_key(p, "pd")
            if not lk:
                continue
            label, key = lk
            upsert(label, key, "pd", p)

    # MT discovery
    if mt_gt and mt_sr:
        for p in discover_vtk_files(mt_gt):
            lk = parse_label_and_key(p, "mt")
            label = "GT"
            key = lk[1] if lk else _normalize_name_for_key(p.name)
            upsert(label, key, "mt", p)
        for p in discover_vtk_files(mt_sr):
            lk = parse_label_and_key(p, "mt")
            label = "SR"
            key = lk[1] if lk else _normalize_name_for_key(p.name)
            upsert(label, key, "mt", p)
    elif mt_dir:
        for p in discover_vtk_files(mt_dir):
            lk = parse_label_and_key(p, "mt")
            if not lk:
                continue
            label, key = lk
            upsert(label, key, "mt", p)

    if debug:
        print(f"[debug] discovered {len(pairs)} keys", file=sys.stderr)

    return pairs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pd-dir", type=Path, default=None, help="Directory containing PD VTUs for both GT/SR (filenames contain GT/SR).")
    ap.add_argument("--mt-dir", type=Path, default=None, help="Directory containing MT VTUs for both GT/SR (filenames contain GT/SR).")
    ap.add_argument("--pd-gt", type=Path, default=None, help="Directory containing PD VTUs for GT only.")
    ap.add_argument("--pd-sr", type=Path, default=None, help="Directory containing PD VTUs for SR only.")
    ap.add_argument("--mt-gt", type=Path, default=None, help="Directory containing MT VTUs for GT only.")
    ap.add_argument("--mt-sr", type=Path, default=None, help="Directory containing MT VTUs for SR only.")
    ap.add_argument("--out", type=Path, default=Path("distances_ttk.csv"), help="Output CSV path.")
    ap.add_argument("--max", type=int, default=0, help="If >0, only process first N samples (sorted by key).")
    ap.add_argument("--debug", action="store_true", help="Verbose debug output (prints array names if extraction fails).")
    args = ap.parse_args()

    # Sanity: at least PD or MT specified
    if not any([args.pd_dir, args.mt_dir, (args.pd_gt and args.pd_sr), (args.mt_gt and args.mt_sr)]):
        print("ERROR: Provide --pd-dir/--mt-dir or --pd-gt/--pd-sr and/or --mt-gt/--mt-sr", file=sys.stderr)
        return 2

    pairs = build_pairs_from_dirs(
        pd_dir=args.pd_dir,
        mt_dir=args.mt_dir,
        pd_gt=args.pd_gt,
        pd_sr=args.pd_sr,
        mt_gt=args.mt_gt,
        mt_sr=args.mt_sr,
        debug=args.debug,
    )

    keys = sorted(pairs.keys())
    if args.max and args.max > 0:
        keys = keys[: args.max]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "key",
                "pd_distance",
                "mt_distance",
                "pd_gt",
                "pd_sr",
                "mt_gt",
                "mt_sr",
                "error",
            ],
        )
        w.writeheader()

        n_ok = 0
        for key in keys:
            item = pairs[key]
            row = {
                "key": key,
                "pd_distance": "",
                "mt_distance": "",
                "pd_gt": str(item.pd_gt) if item.pd_gt else "",
                "pd_sr": str(item.pd_sr) if item.pd_sr else "",
                "mt_gt": str(item.mt_gt) if item.mt_gt else "",
                "mt_sr": str(item.mt_sr) if item.mt_sr else "",
                "error": "",
            }

            try:
                if item.pd_gt and item.pd_sr:
                    row["pd_distance"] = f"{compute_pd_distance(item.pd_gt, item.pd_sr, debug=args.debug):.10g}"
                if item.mt_gt and item.mt_sr:
                    row["mt_distance"] = f"{compute_mt_distance(item.mt_gt, item.mt_sr, debug=args.debug):.10g}"
                n_ok += 1
            except Exception as e:
                row["error"] = str(e)

            w.writerow(row)

    print(f"Wrote: {args.out} (rows={len(keys)}, ok_rows={n_ok})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
