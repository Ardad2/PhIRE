#!/usr/bin/env python3
"""
Audit raw TTK MergeTree node/arc arrays and geometry for sample 69 at the
already-frozen qualitative display persistence threshold 3.0.

Purpose
-------
Determine whether the raw TTK outputs contain trustworthy node/arc hierarchy
attributes that can support an abstract tree rendering.

This script DOES NOT:
- alter the frozen display threshold;
- alter the audited numerical merge-tree distance;
- use historical F-order VTIs;
- assume that suspicious historical upNodeId/downNodeId arrays are valid.

Inputs
------
$W22/corrected_pd_mt/discordance_visuals/sample_069/

GT:
  gt_threshold_sweep/t3/{nodes.vtu,arcs.vtu}

CNN / UV / F1:
  fixed_threshold_t3/<method>/{nodes.vtu,arcs.vtu}

Authoritative scalar fields:
  inputs/{gt,cnn,uv,f1}_s069.vti

Outputs
-------
sample69_mt_raw_array_inventory.csv
sample69_mt_raw_array_audit.json
sample69_mt_raw_array_audit_summary.txt
"""

from __future__ import annotations

import csv
import json
import math
import os
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy


W22 = Path(os.environ["W22"])
BASE = (
    W22
    / "corrected_pd_mt"
    / "discordance_visuals"
    / "sample_069"
)

METHODS = {
    "gt": {
        "field": BASE / "inputs/gt_s069.vti",
        "root": BASE / "gt_threshold_sweep/t3",
    },
    "cnn": {
        "field": BASE / "inputs/cnn_s069.vti",
        "root": BASE / "fixed_threshold_t3/cnn",
    },
    "uv": {
        "field": BASE / "inputs/uv_s069.vti",
        "root": BASE / "fixed_threshold_t3/uv",
    },
    "f1": {
        "field": BASE / "inputs/f1_s069.vti",
        "root": BASE / "fixed_threshold_t3/f1",
    },
}

OUT_CSV = BASE / "sample69_mt_raw_array_inventory.csv"
OUT_JSON = BASE / "sample69_mt_raw_array_audit.json"
OUT_TXT = BASE / "sample69_mt_raw_array_audit_summary.txt"

ABS_TOL = 1e-6
GRID_TOL = 1e-6
SCALAR_TOL = 5e-5


def read_vti(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)

    r = vtk.vtkXMLImageDataReader()
    r.SetFileName(str(path))
    r.Update()
    img = r.GetOutput()

    if img is None:
        raise RuntimeError(f"Failed to read VTI: {path}")

    arr = img.GetPointData().GetArray("wind_speed")
    if arr is None:
        raise RuntimeError(f"wind_speed missing: {path}")

    dims = img.GetDimensions()
    W, H, Z = dims
    if Z != 1:
        raise RuntimeError(f"Expected Z=1 in {path}, got {dims}")

    flat = vtk_to_numpy(arr).astype(np.float64, copy=False)
    field = flat.reshape(H, W, order="C")

    return img, field


def read_grid(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)

    r = vtk.vtkXMLUnstructuredGridReader()
    r.SetFileName(str(path))
    r.Update()
    g = r.GetOutput()

    if g is None:
        raise RuntimeError(f"Failed to read VTU: {path}")

    return g


def vtk_array_to_numpy(arr):
    try:
        a = vtk_to_numpy(arr)
    except Exception:
        return None

    if a.ndim == 1:
        a = a[:, None]

    return np.asarray(a)


def numeric_summary(a: np.ndarray | None) -> dict[str, Any]:
    if a is None or a.size == 0:
        return {
            "numeric": False,
            "finite": None,
            "min": None,
            "max": None,
            "unique_count": None,
            "integer_like_fraction": None,
        }

    if not np.issubdtype(a.dtype, np.number):
        return {
            "numeric": False,
            "finite": None,
            "min": None,
            "max": None,
            "unique_count": None,
            "integer_like_fraction": None,
        }

    af = a.astype(np.float64, copy=False)
    finite = np.isfinite(af)

    if finite.any():
        vals = af[finite]
        mn = float(np.min(vals))
        mx = float(np.max(vals))
        integer_like = float(
            np.mean(np.abs(vals - np.round(vals)) <= 1e-9)
        )
        # Cap exact unique calculation for safety.
        if vals.size <= 1_000_000:
            uniq = int(np.unique(vals).size)
        else:
            uniq = None
    else:
        mn = None
        mx = None
        integer_like = None
        uniq = 0

    return {
        "numeric": True,
        "finite": bool(finite.all()),
        "min": mn,
        "max": mx,
        "unique_count": uniq,
        "integer_like_fraction": integer_like,
    }


def inventory_attributes(method: str, obj_name: str, grid) -> list[dict[str, Any]]:
    rows = []

    associations = [
        ("point", grid.GetPointData()),
        ("cell", grid.GetCellData()),
        ("field", grid.GetFieldData()),
    ]

    for assoc_name, attrs in associations:
        for i in range(attrs.GetNumberOfArrays()):
            arr = attrs.GetAbstractArray(i)
            name = arr.GetName() if arr.GetName() else f"<unnamed_{i}>"
            comps = int(arr.GetNumberOfComponents())
            tuples = int(arr.GetNumberOfTuples())
            dtype = arr.GetDataTypeAsString()

            np_arr = vtk_array_to_numpy(arr)
            summary = numeric_summary(np_arr)

            rows.append({
                "method": method,
                "object": obj_name,
                "association": assoc_name,
                "array_name": name,
                "vtk_dtype": dtype,
                "components": comps,
                "tuples": tuples,
                **summary,
            })

    return rows


def get_named_numeric_arrays(grid) -> dict[str, dict[str, np.ndarray]]:
    out = {"point": {}, "cell": {}, "field": {}}

    for assoc, attrs in [
        ("point", grid.GetPointData()),
        ("cell", grid.GetCellData()),
        ("field", grid.GetFieldData()),
    ]:
        for i in range(attrs.GetNumberOfArrays()):
            arr = attrs.GetAbstractArray(i)
            name = arr.GetName() or f"<unnamed_{i}>"
            a = vtk_array_to_numpy(arr)
            if a is not None and np.issubdtype(a.dtype, np.number):
                out[assoc][name] = a

    return out


def node_geometry_audit(nodes, field: np.ndarray) -> dict[str, Any]:
    pts = vtk_to_numpy(nodes.GetPoints().GetData()).astype(np.float64)
    xy = pts[:, :2]

    rounded = np.round(xy)
    grid_exact = np.all(np.abs(xy - rounded) <= GRID_TOL, axis=1)

    H, W = field.shape
    in_bounds = (
        (rounded[:, 0] >= 0)
        & (rounded[:, 0] < W)
        & (rounded[:, 1] >= 0)
        & (rounded[:, 1] < H)
    )

    sampled = np.full(len(xy), np.nan, dtype=np.float64)
    valid = grid_exact & in_bounds

    xi = rounded[valid, 0].astype(int)
    yi = rounded[valid, 1].astype(int)
    sampled[valid] = field[yi, xi]

    arrays = get_named_numeric_arrays(nodes)["point"]

    scalar_candidates = {}
    for name, a in arrays.items():
        if a.shape[0] != len(xy) or a.shape[1] != 1:
            continue

        lname = name.lower()
        if (
            "scalar" in lname
            or "value" in lname
            or "function" in lname
        ):
            vals = a[:, 0].astype(np.float64)
            finite_mask = np.isfinite(vals) & np.isfinite(sampled)
            if finite_mask.any():
                diff = np.abs(vals[finite_mask] - sampled[finite_mask])
                scalar_candidates[name] = {
                    "finite_pairs": int(finite_mask.sum()),
                    "max_abs_diff_vs_field_at_node": float(np.max(diff)),
                    "mean_abs_diff_vs_field_at_node": float(np.mean(diff)),
                    "matches_field_within_tol": bool(np.max(diff) <= SCALAR_TOL),
                }

    return {
        "node_count": int(len(xy)),
        "all_xy_integer_grid": bool(grid_exact.all()),
        "integer_grid_fraction": float(np.mean(grid_exact)),
        "all_xy_in_bounds": bool(in_bounds.all()),
        "sampled_field_min": float(np.nanmin(sampled)),
        "sampled_field_max": float(np.nanmax(sampled)),
        "scalar_candidates": scalar_candidates,
    }


def arc_connectivity_audit(arcs, nodes) -> dict[str, Any]:
    pts = vtk_to_numpy(arcs.GetPoints().GetData()).astype(np.float64)
    n_points = int(arcs.GetNumberOfPoints())
    n_cells = int(arcs.GetNumberOfCells())

    adjacency = defaultdict(set)
    edge_counter = Counter()
    bad_cells = []
    cell_types = Counter()

    for cid in range(n_cells):
        cell = arcs.GetCell(cid)
        ctype = int(cell.GetCellType())
        cell_types[ctype] += 1

        ids = cell.GetPointIds()
        nids = ids.GetNumberOfIds()

        if nids != 2:
            bad_cells.append({
                "cell_id": cid,
                "num_point_ids": int(nids),
                "cell_type": ctype,
            })
            continue

        u = int(ids.GetId(0))
        v = int(ids.GetId(1))

        if not (0 <= u < n_points and 0 <= v < n_points):
            bad_cells.append({
                "cell_id": cid,
                "bad_endpoint": [u, v],
                "cell_type": ctype,
            })
            continue

        if u == v:
            bad_cells.append({
                "cell_id": cid,
                "self_loop": u,
                "cell_type": ctype,
            })

        adjacency[u].add(v)
        adjacency[v].add(u)
        edge_counter[tuple(sorted((u, v)))] += 1

    degrees = np.zeros(n_points, dtype=int)
    for p in range(n_points):
        degrees[p] = len(adjacency[p])

    # Connected components over segment graph.
    seen = set()
    components = []

    for start in range(n_points):
        if start in seen:
            continue

        q = deque([start])
        seen.add(start)
        comp = []

        while q:
            u = q.popleft()
            comp.append(u)
            for v in adjacency[u]:
                if v not in seen:
                    seen.add(v)
                    q.append(v)

        components.append(comp)

    node_pts = vtk_to_numpy(nodes.GetPoints().GetData()).astype(np.float64)[:, :2]
    arc_xy = pts[:, :2]

    # Geometric matching of critical nodes to raw arc points.
    node_matches = []
    for i, p in enumerate(node_pts):
        d = np.linalg.norm(arc_xy - p[None, :], axis=1)
        exact_ids = np.where(d <= ABS_TOL)[0]
        node_matches.append({
            "node_index": i,
            "matched_arc_point_count": int(len(exact_ids)),
            "matched_arc_point_ids": [int(x) for x in exact_ids[:20]],
            "nearest_arc_point_distance": float(np.min(d)),
        })

    matched_counts = [x["matched_arc_point_count"] for x in node_matches]

    # Segment graph tree test per connected component.
    comp_summaries = []
    for comp in components:
        comp_set = set(comp)
        edge_count = 0
        for u in comp:
            edge_count += sum(1 for v in adjacency[u] if v in comp_set)
        edge_count //= 2

        comp_summaries.append({
            "points": int(len(comp)),
            "edges": int(edge_count),
            "is_tree": bool(edge_count == len(comp) - 1),
        })

    return {
        "arc_point_count": n_points,
        "arc_cell_count": n_cells,
        "cell_types": {str(k): int(v) for k, v in cell_types.items()},
        "bad_cell_count": int(len(bad_cells)),
        "bad_cells_first20": bad_cells[:20],
        "duplicate_segment_pairs": int(
            sum(1 for _, count in edge_counter.items() if count > 1)
        ),
        "degree_histogram": {
            str(k): int(v)
            for k, v in Counter(degrees.tolist()).items()
        },
        "connected_component_count": int(len(components)),
        "component_summaries": comp_summaries,
        "all_components_are_trees": bool(
            all(c["is_tree"] for c in comp_summaries)
        ),
        "node_to_arc_exact_match_count": int(
            sum(c > 0 for c in matched_counts)
        ),
        "node_to_arc_exact_match_fraction": float(
            np.mean(np.array(matched_counts) > 0)
        ),
        "node_to_arc_match_multiplicity_histogram": {
            str(k): int(v)
            for k, v in Counter(matched_counts).items()
        },
        "node_matches": node_matches,
    }


def candidate_id_array_audit(nodes, arcs) -> dict[str, Any]:
    node_arrays = get_named_numeric_arrays(nodes)
    arc_arrays = get_named_numeric_arrays(arcs)

    n_nodes = nodes.GetNumberOfPoints()

    node_id_candidates = {}
    for assoc in ["point", "cell", "field"]:
        for name, a in node_arrays[assoc].items():
            lname = name.lower()
            if "id" not in lname:
                continue

            vals = a.reshape(-1).astype(np.float64)
            finite = np.isfinite(vals)
            intlike = finite & (np.abs(vals - np.round(vals)) <= 1e-9)

            node_id_candidates[f"{assoc}:{name}"] = {
                "n_values": int(vals.size),
                "finite_fraction": float(np.mean(finite)),
                "integer_like_fraction": float(np.mean(intlike)),
                "min": float(np.min(vals[finite])) if finite.any() else None,
                "max": float(np.max(vals[finite])) if finite.any() else None,
                "unique_count": int(np.unique(vals[finite]).size) if finite.any() else 0,
            }

    arc_id_candidates = {}
    for assoc in ["point", "cell", "field"]:
        for name, a in arc_arrays[assoc].items():
            lname = name.lower()
            if not (
                "id" in lname
                or "up" in lname
                or "down" in lname
                or "node" in lname
            ):
                continue

            vals = a.reshape(-1).astype(np.float64)
            finite = np.isfinite(vals)
            intlike = finite & (np.abs(vals - np.round(vals)) <= 1e-9)
            in_zero_based = (
                intlike
                & (vals >= 0)
                & (vals < n_nodes)
            )

            arc_id_candidates[f"{assoc}:{name}"] = {
                "n_values": int(vals.size),
                "finite_fraction": float(np.mean(finite)),
                "integer_like_fraction": float(np.mean(intlike)),
                "zero_based_node_range_fraction": float(np.mean(in_zero_based)),
                "min": float(np.min(vals[finite])) if finite.any() else None,
                "max": float(np.max(vals[finite])) if finite.any() else None,
                "unique_count": int(np.unique(vals[finite]).size) if finite.any() else 0,
            }

    return {
        "node_id_candidates": node_id_candidates,
        "arc_endpoint_or_id_candidates": arc_id_candidates,
    }


inventory_rows = []
report = {
    "sample": 69,
    "display_threshold": 3.0,
    "methods": {},
}

for method, paths in METHODS.items():
    _, field = read_vti(paths["field"])
    nodes = read_grid(paths["root"] / "nodes.vtu")
    arcs = read_grid(paths["root"] / "arcs.vtu")

    inventory_rows.extend(inventory_attributes(method, "nodes", nodes))
    inventory_rows.extend(inventory_attributes(method, "arcs", arcs))

    report["methods"][method] = {
        "paths": {
            "field": str(paths["field"]),
            "nodes": str(paths["root"] / "nodes.vtu"),
            "arcs": str(paths["root"] / "arcs.vtu"),
        },
        "node_geometry": node_geometry_audit(nodes, field),
        "arc_geometry_connectivity": arc_connectivity_audit(arcs, nodes),
        "candidate_arrays": candidate_id_array_audit(nodes, arcs),
    }

# -------------------------------------------------------------------------
# Write CSV inventory
# -------------------------------------------------------------------------

fieldnames = [
    "method",
    "object",
    "association",
    "array_name",
    "vtk_dtype",
    "components",
    "tuples",
    "numeric",
    "finite",
    "min",
    "max",
    "unique_count",
    "integer_like_fraction",
]

with OUT_CSV.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(inventory_rows)

OUT_JSON.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

# -------------------------------------------------------------------------
# Human-readable summary
# -------------------------------------------------------------------------

lines = []


def emit(s=""):
    lines.append(s)
    print(s)


emit("SAMPLE-69 RAW TTK MERGE-TREE ARRAY / HIERARCHY AUDIT")
emit("=" * 100)
emit()
emit("Purpose:")
emit("  Decide whether raw TTK hierarchy arrays are trustworthy enough")
emit("  to construct an abstract tree visualization.")
emit()
emit("Frozen qualitative display threshold: 3.0")
emit()

for method in ["gt", "cnn", "uv", "f1"]:
    m = report["methods"][method]
    ng = m["node_geometry"]
    ag = m["arc_geometry_connectivity"]
    ca = m["candidate_arrays"]

    emit(method.upper())
    emit("-" * 100)
    emit(
        f"  nodes={ng['node_count']} "
        f"arc_points={ag['arc_point_count']} "
        f"arc_cells={ag['arc_cell_count']}"
    )
    emit(
        f"  node XY integer-grid fraction={ng['integer_grid_fraction']:.3f}; "
        f"all in bounds={ng['all_xy_in_bounds']}"
    )
    emit(
        f"  raw arc cell validity: bad_cells={ag['bad_cell_count']}, "
        f"duplicate_segment_pairs={ag['duplicate_segment_pairs']}"
    )
    emit(
        f"  segment-graph connected components={ag['connected_component_count']}; "
        f"all components trees={ag['all_components_are_trees']}"
    )
    emit(
        f"  critical nodes matching an arc point exactly: "
        f"{ag['node_to_arc_exact_match_count']}/{ng['node_count']} "
        f"({100*ag['node_to_arc_exact_match_fraction']:.1f}%)"
    )

    emit()
    emit("  Scalar-like node arrays vs authoritative field:")
    if ng["scalar_candidates"]:
        for name, x in ng["scalar_candidates"].items():
            emit(
                f"    {name}: "
                f"max_abs_diff={x['max_abs_diff_vs_field_at_node']:.6g}; "
                f"matches={x['matches_field_within_tol']}"
            )
    else:
        emit("    NONE")

    emit()
    emit("  Candidate node-ID arrays:")
    if ca["node_id_candidates"]:
        for name, x in ca["node_id_candidates"].items():
            emit(
                f"    {name}: "
                f"finite={100*x['finite_fraction']:.1f}% "
                f"integerlike={100*x['integer_like_fraction']:.1f}% "
                f"range=[{x['min']}, {x['max']}] "
                f"unique={x['unique_count']}"
            )
    else:
        emit("    NONE")

    emit()
    emit("  Candidate arc endpoint/ID arrays:")
    if ca["arc_endpoint_or_id_candidates"]:
        for name, x in ca["arc_endpoint_or_id_candidates"].items():
            emit(
                f"    {name}: "
                f"finite={100*x['finite_fraction']:.1f}% "
                f"integerlike={100*x['integer_like_fraction']:.1f}% "
                f"in_0..N-1={100*x['zero_based_node_range_fraction']:.1f}% "
                f"range=[{x['min']}, {x['max']}] "
                f"unique={x['unique_count']}"
            )
    else:
        emit("    NONE")

    emit()

emit("ARRAY INVENTORY")
emit("-" * 100)

for r in inventory_rows:
    emit(
        f"  {r['method']:3s} {r['object']:5s} {r['association']:5s} "
        f"{r['array_name']!r:32s} "
        f"dtype={r['vtk_dtype']:12s} "
        f"comps={r['components']} tuples={r['tuples']} "
        f"finite={r['finite']} "
        f"range=[{r['min']}, {r['max']}]"
    )

emit()
emit("DECISION POLICY")
emit("-" * 100)
emit("  Do NOT use an ID/connectivity array merely because its name looks plausible.")
emit("  Abstract hierarchy reconstruction should proceed only if:")
emit("    1. endpoint/node-ID arrays are finite and integer-like;")
emit("    2. their values map coherently to the node set;")
emit("    3. scalar-like node arrays agree with authoritative wind_speed at node coordinates;")
emit("    4. reconstructed logical connectivity is acyclic and tree-consistent;")
emit("    5. the result is consistent across GT/CNN/UV/F1.")
emit()
emit(f"CSV:  {OUT_CSV}")
emit(f"JSON: {OUT_JSON}")
emit(f"TXT:  {OUT_TXT}")

OUT_TXT.write_text("\n".join(lines) + "\n")

print()
print("RAW TTK ARRAY / HIERARCHY AUDIT: COMPLETE")
