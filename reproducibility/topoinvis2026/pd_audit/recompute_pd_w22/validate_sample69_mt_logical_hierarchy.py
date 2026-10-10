#!/usr/bin/env python3
"""
Validate the logical merge-tree hierarchy encoded in TTK's raw threshold-3
sample-69 outputs.

This is a second-stage audit after the raw-array inventory. It tests whether
NodeId / SegmentationId / upNodeId / downNodeId encode a coherent rooted tree
and whether each logical superarc's sampled geometry terminates at the
corresponding critical nodes.

It deliberately does NOT require TTK's node Scalar array to equal the original
authoritative wind_speed field. Instead it separately tests:
  (a) TTK-internal scalar consistency between node and arc outputs;
  (b) monotonicity along logical up/down node relationships;
  (c) original-field consistency as a diagnostic only.

If the hierarchy checks pass but original-field scalar equality does not, an
abstract hierarchy may be rendered using topology/depth only, while scalar-axis
claims should remain qualified.
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

OUT_ARCS = BASE / "sample69_mt_logical_arcs.csv"
OUT_NODES = BASE / "sample69_mt_logical_nodes.csv"
OUT_JSON = BASE / "sample69_mt_logical_hierarchy_validation.json"
OUT_TXT = BASE / "sample69_mt_logical_hierarchy_validation_summary.txt"

XY_TOL = 1e-6
SCALAR_TOL = 5e-5


def read_vti(path: Path):
    r = vtk.vtkXMLImageDataReader()
    r.SetFileName(str(path))
    r.Update()
    img = r.GetOutput()

    if img is None:
        raise RuntimeError(f"Failed to read VTI: {path}")

    arr = img.GetPointData().GetArray("wind_speed")
    if arr is None:
        raise RuntimeError(f"wind_speed missing: {path}")

    W, H, Z = img.GetDimensions()
    if Z != 1:
        raise RuntimeError(f"Expected Z=1, got {(W,H,Z)} for {path}")

    flat = vtk_to_numpy(arr).astype(np.float64, copy=False)
    field = flat.reshape(H, W, order="C")
    return img, field


def read_grid(path: Path):
    r = vtk.vtkXMLUnstructuredGridReader()
    r.SetFileName(str(path))
    r.Update()
    g = r.GetOutput()
    if g is None:
        raise RuntimeError(f"Failed to read VTU: {path}")
    return g


def require_array(attrs, name: str, tuples: int | None = None):
    arr = attrs.GetArray(name)
    if arr is None:
        raise RuntimeError(f"Required array missing: {name}")

    a = np.asarray(vtk_to_numpy(arr))
    if tuples is not None and len(a) != tuples:
        raise RuntimeError(
            f"{name}: expected {tuples} tuples, got {len(a)}"
        )
    return a


def graph_connected_acyclic(nodes, undirected_edges):
    adj = {int(n): set() for n in nodes}
    for u, v in undirected_edges:
        adj[int(u)].add(int(v))
        adj[int(v)].add(int(u))

    if not adj:
        return False, False, 0

    start = next(iter(adj))
    seen = set()
    q = deque([start])
    seen.add(start)

    while q:
        u = q.popleft()
        for v in adj[u]:
            if v not in seen:
                seen.add(v)
                q.append(v)

    connected = len(seen) == len(adj)
    acyclic = len(undirected_edges) == len(adj) - 1 if connected else False
    return connected, acyclic, len(seen)


def simple_path_audit(cell_ids, arcs):
    """
    Audit sampled line cells for one logical SegmentationId.
    Return point set, endpoints, connected/path flags.
    """
    adjacency = defaultdict(set)
    used_points = set()
    bad = []

    for cid in cell_ids:
        cell = arcs.GetCell(int(cid))
        ids = cell.GetPointIds()

        if ids.GetNumberOfIds() != 2:
            bad.append(int(cid))
            continue

        u = int(ids.GetId(0))
        v = int(ids.GetId(1))
        used_points.add(u)
        used_points.add(v)
        adjacency[u].add(v)
        adjacency[v].add(u)

    if bad:
        return {
            "valid_lines": False,
            "connected": False,
            "simple_path": False,
            "used_point_ids": sorted(used_points),
            "endpoint_point_ids": [],
            "max_degree": None,
        }

    if not used_points:
        return {
            "valid_lines": True,
            "connected": False,
            "simple_path": False,
            "used_point_ids": [],
            "endpoint_point_ids": [],
            "max_degree": 0,
        }

    start = next(iter(used_points))
    seen = {start}
    q = deque([start])

    while q:
        u = q.popleft()
        for v in adjacency[u]:
            if v not in seen:
                seen.add(v)
                q.append(v)

    connected = len(seen) == len(used_points)
    degrees = {p: len(adjacency[p]) for p in used_points}
    endpoints = sorted([p for p, d in degrees.items() if d == 1])
    max_degree = max(degrees.values())

    edge_count = len(cell_ids)
    vertex_count = len(used_points)

    simple_path = bool(
        connected
        and edge_count == vertex_count - 1
        and len(endpoints) == 2
        and max_degree <= 2
    )

    return {
        "valid_lines": True,
        "connected": connected,
        "simple_path": simple_path,
        "used_point_ids": sorted(used_points),
        "endpoint_point_ids": endpoints,
        "max_degree": int(max_degree),
    }


def coord_key(xy):
    return tuple(float(v) for v in xy[:2])


logical_arc_rows = []
logical_node_rows = []
report: dict[str, Any] = {
    "sample": 69,
    "display_threshold": 3.0,
    "methods": {},
}


for method, paths in METHODS.items():
    img, field = read_vti(paths["field"])
    nodes = read_grid(paths["root"] / "nodes.vtu")
    arcs = read_grid(paths["root"] / "arcs.vtu")

    n_nodes = int(nodes.GetNumberOfPoints())
    n_arc_cells = int(arcs.GetNumberOfCells())

    node_pts = np.asarray(
        vtk_to_numpy(nodes.GetPoints().GetData()),
        dtype=np.float64,
    )
    arc_pts = np.asarray(
        vtk_to_numpy(arcs.GetPoints().GetData()),
        dtype=np.float64,
    )

    node_ids = require_array(
        nodes.GetPointData(),
        "NodeId",
        n_nodes,
    ).astype(int)

    vertex_ids = require_array(
        nodes.GetPointData(),
        "VertexId",
        n_nodes,
    ).astype(int)

    node_scalar = require_array(
        nodes.GetPointData(),
        "Scalar",
        n_nodes,
    ).astype(np.float64)

    critical_type = require_array(
        nodes.GetPointData(),
        "CriticalType",
        n_nodes,
    ).astype(int)

    seg_ids = require_array(
        arcs.GetCellData(),
        "SegmentationId",
        n_arc_cells,
    ).astype(int)

    up_ids = require_array(
        arcs.GetCellData(),
        "upNodeId",
        n_arc_cells,
    ).astype(int)

    down_ids = require_array(
        arcs.GetCellData(),
        "downNodeId",
        n_arc_cells,
    ).astype(int)

    arc_scalar = require_array(
        arcs.GetPointData(),
        "Scalar",
        arcs.GetNumberOfPoints(),
    ).astype(np.float64)

    # ------------------------------------------------------------------
    # Node identity / coordinate / VertexId checks
    # ------------------------------------------------------------------

    node_id_unique = len(np.unique(node_ids)) == n_nodes
    node_id_contiguous = np.array_equal(
        np.sort(node_ids),
        np.arange(n_nodes, dtype=int),
    )

    node_row_for_id = {
        int(node_id): int(row)
        for row, node_id in enumerate(node_ids)
    }

    if len(node_row_for_id) != n_nodes:
        raise RuntimeError(f"{method}: duplicate NodeId values")

    W, H, _ = img.GetDimensions()

    xy_rounded = np.round(node_pts[:, :2]).astype(int)
    xy_integer = np.all(
        np.abs(node_pts[:, :2] - xy_rounded) <= XY_TOL,
        axis=1,
    )

    expected_vertex_ids = (
        xy_rounded[:, 0]
        + xy_rounded[:, 1] * W
    )

    vertex_id_matches_c_order = (
        vertex_ids == expected_vertex_ids
    )

    authoritative_scalar = field[
        xy_rounded[:, 1],
        xy_rounded[:, 0],
    ]

    original_scalar_diff = np.abs(
        node_scalar - authoritative_scalar
    )

    # TTK-internal scalar consistency: node scalar vs coincident arc point scalars.
    arc_points_by_coord = defaultdict(list)
    for pid, p in enumerate(arc_pts[:, :2]):
        arc_points_by_coord[coord_key(p)].append(pid)

    internal_node_scalar_diffs = []
    internal_node_scalar_match_counts = []

    for row in range(n_nodes):
        ids = arc_points_by_coord.get(
            coord_key(node_pts[row, :2]),
            [],
        )

        internal_node_scalar_match_counts.append(len(ids))

        if ids:
            diffs = np.abs(
                arc_scalar[np.asarray(ids, dtype=int)]
                - node_scalar[row]
            )
            internal_node_scalar_diffs.extend(
                float(x) for x in diffs
            )

    internal_scalar_max_diff = (
        max(internal_node_scalar_diffs)
        if internal_node_scalar_diffs
        else None
    )

    # ------------------------------------------------------------------
    # Collapse sampled cells into logical superarcs via SegmentationId
    # ------------------------------------------------------------------

    unique_seg = sorted(int(x) for x in np.unique(seg_ids))
    seg_contiguous = unique_seg == list(range(len(unique_seg)))
    expected_logical_arc_count = n_nodes - 1

    cells_by_seg = defaultdict(list)
    for cid, sid in enumerate(seg_ids):
        cells_by_seg[int(sid)].append(int(cid))

    logical_edges = []
    all_groups_constant = True
    all_groups_paths = True
    all_endpoints_match = True
    all_monotone = True

    for sid in unique_seg:
        cids = cells_by_seg[sid]

        ups = sorted(set(int(up_ids[c]) for c in cids))
        downs = sorted(set(int(down_ids[c]) for c in cids))

        constant_pair = len(ups) == 1 and len(downs) == 1
        all_groups_constant &= constant_pair

        up = ups[0] if len(ups) == 1 else None
        down = downs[0] if len(downs) == 1 else None

        if not constant_pair:
            path_info = {
                "valid_lines": False,
                "connected": False,
                "simple_path": False,
                "endpoint_point_ids": [],
                "used_point_ids": [],
                "max_degree": None,
            }
            endpoint_match = False
            scalar_monotone = False
            down_scalar = None
            up_scalar = None
        else:
            if up not in node_row_for_id or down not in node_row_for_id:
                raise RuntimeError(
                    f"{method}: logical arc {sid} references "
                    f"unknown node(s): down={down}, up={up}"
                )

            logical_edges.append((down, up))

            path_info = simple_path_audit(cids, arcs)
            all_groups_paths &= bool(path_info["simple_path"])

            down_row = node_row_for_id[down]
            up_row = node_row_for_id[up]

            down_xy = node_pts[down_row, :2]
            up_xy = node_pts[up_row, :2]

            endpoint_coords = [
                arc_pts[pid, :2]
                for pid in path_info["endpoint_point_ids"]
            ]

            endpoint_match = False
            if len(endpoint_coords) == 2:
                a, b = endpoint_coords
                direct = (
                    np.linalg.norm(a - down_xy) <= XY_TOL
                    and np.linalg.norm(b - up_xy) <= XY_TOL
                )
                reverse = (
                    np.linalg.norm(b - down_xy) <= XY_TOL
                    and np.linalg.norm(a - up_xy) <= XY_TOL
                )
                endpoint_match = bool(direct or reverse)

            all_endpoints_match &= endpoint_match

            down_scalar = float(node_scalar[down_row])
            up_scalar = float(node_scalar[up_row])

            # "up" should not be lower in TTK's own scalar field.
            scalar_monotone = bool(
                up_scalar + SCALAR_TOL >= down_scalar
            )
            all_monotone &= scalar_monotone

        logical_arc_rows.append({
            "method": method,
            "segmentation_id": sid,
            "sampled_cell_count": len(cids),
            "up_node_id": up,
            "down_node_id": down,
            "constant_up_down_pair": int(constant_pair),
            "sampled_geometry_simple_path": int(path_info["simple_path"]),
            "sampled_path_endpoint_matches_nodes": int(endpoint_match),
            "down_node_scalar": down_scalar,
            "up_node_scalar": up_scalar,
            "up_scalar_ge_down_scalar": int(scalar_monotone),
        })

    # ------------------------------------------------------------------
    # Rooted logical-tree checks
    # ------------------------------------------------------------------

    logical_edge_unique = len(set(logical_edges)) == len(logical_edges)
    no_self_loops = all(d != u for d, u in logical_edges)

    connected, acyclic, seen_count = graph_connected_acyclic(
        node_ids.tolist(),
        [(d, u) for d, u in logical_edges],
    )

    down_counter = Counter(d for d, _ in logical_edges)
    up_counter = Counter(u for _, u in logical_edges)

    root_candidates = sorted(
        set(int(x) for x in node_ids)
        - set(down_counter.keys())
    )

    every_nonroot_once_as_down = bool(
        len(root_candidates) == 1
        and all(
            down_counter[int(n)] == 1
            for n in node_ids
            if int(n) not in root_candidates
        )
        and down_counter[root_candidates[0]] == 0
    )

    root_id = (
        root_candidates[0]
        if len(root_candidates) == 1
        else None
    )

    # Node table
    for row in range(n_nodes):
        nid = int(node_ids[row])
        logical_node_rows.append({
            "method": method,
            "node_id": nid,
            "vertex_id": int(vertex_ids[row]),
            "x": float(node_pts[row, 0]),
            "y": float(node_pts[row, 1]),
            "critical_type": int(critical_type[row]),
            "ttk_scalar": float(node_scalar[row]),
            "authoritative_field_scalar": float(authoritative_scalar[row]),
            "scalar_abs_diff": float(original_scalar_diff[row]),
            "vertex_id_matches_c_order_xy": int(vertex_id_matches_c_order[row]),
            "is_root_candidate": int(nid == root_id),
            "parent_count_via_up_relation": int(down_counter[nid]),
            "child_count_via_up_relation": int(up_counter[nid]),
            "incident_arc_point_matches": int(
                internal_node_scalar_match_counts[row]
            ),
        })

    hierarchy_pass = bool(
        node_id_unique
        and node_id_contiguous
        and np.all(xy_integer)
        and np.all(vertex_id_matches_c_order)
        and len(unique_seg) == expected_logical_arc_count
        and seg_contiguous
        and all_groups_constant
        and all_groups_paths
        and all_endpoints_match
        and logical_edge_unique
        and no_self_loops
        and connected
        and acyclic
        and every_nonroot_once_as_down
        and all_monotone
    )

    internal_scalar_pass = bool(
        internal_scalar_max_diff is not None
        and internal_scalar_max_diff <= SCALAR_TOL
    )

    original_scalar_pass = bool(
        np.max(original_scalar_diff) <= SCALAR_TOL
    )

    report["methods"][method] = {
        "counts": {
            "nodes": n_nodes,
            "sampled_arc_cells": n_arc_cells,
            "logical_arcs": len(unique_seg),
            "expected_logical_arcs": expected_logical_arc_count,
        },
        "node_identity": {
            "node_id_unique": node_id_unique,
            "node_id_contiguous_0_to_nminus1": node_id_contiguous,
            "all_xy_integer": bool(np.all(xy_integer)),
            "vertex_id_matches_c_order_xy_all": bool(
                np.all(vertex_id_matches_c_order)
            ),
        },
        "logical_arc_grouping": {
            "segmentation_ids_contiguous": seg_contiguous,
            "all_cells_within_segmentation_share_up_down": all_groups_constant,
            "all_sampled_logical_arcs_are_simple_paths": all_groups_paths,
            "all_sampled_path_endpoints_match_referenced_nodes":
                all_endpoints_match,
            "logical_edge_pairs_unique": logical_edge_unique,
            "no_self_loops": no_self_loops,
        },
        "logical_tree": {
            "connected": connected,
            "acyclic": acyclic,
            "seen_nodes": seen_count,
            "root_candidates": root_candidates,
            "root_id": root_id,
            "every_nonroot_node_occurs_once_as_downNodeId":
                every_nonroot_once_as_down,
        },
        "scalar_checks": {
            "ttk_node_vs_coincident_arc_scalar_max_abs_diff":
                internal_scalar_max_diff,
            "ttk_internal_scalar_consistency_pass": internal_scalar_pass,
            "ttk_up_scalar_ge_down_scalar_all": all_monotone,
            "ttk_node_vs_original_field_max_abs_diff":
                float(np.max(original_scalar_diff)),
            "ttk_node_vs_original_field_mean_abs_diff":
                float(np.mean(original_scalar_diff)),
            "original_field_scalar_equality_pass": original_scalar_pass,
        },
        "hierarchy_pass": hierarchy_pass,
    }


# -------------------------------------------------------------------------
# Write artifacts
# -------------------------------------------------------------------------

node_fields = list(logical_node_rows[0].keys())
with OUT_NODES.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=node_fields)
    w.writeheader()
    w.writerows(logical_node_rows)

arc_fields = list(logical_arc_rows[0].keys())
with OUT_ARCS.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=arc_fields)
    w.writeheader()
    w.writerows(logical_arc_rows)

OUT_JSON.write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n"
)


# -------------------------------------------------------------------------
# Summary
# -------------------------------------------------------------------------

lines = []


def emit(s=""):
    lines.append(s)
    print(s)


emit("SAMPLE-69 LOGICAL MERGE-TREE HIERARCHY VALIDATION")
emit("=" * 100)
emit()
emit("Frozen qualitative display persistence threshold: 3.0")
emit()

all_hierarchy_pass = True

for method in ["gt", "cnn", "uv", "f1"]:
    m = report["methods"][method]
    c = m["counts"]
    ni = m["node_identity"]
    lg = m["logical_arc_grouping"]
    lt = m["logical_tree"]
    sc = m["scalar_checks"]

    emit(method.upper())
    emit("-" * 100)
    emit(
        f"  nodes={c['nodes']} logical_arcs={c['logical_arcs']} "
        f"expected=N-1={c['expected_logical_arcs']} "
        f"sampled_arc_cells={c['sampled_arc_cells']}"
    )
    emit(
        f"  NodeId unique/contiguous: "
        f"{ni['node_id_unique']} / "
        f"{ni['node_id_contiguous_0_to_nminus1']}"
    )
    emit(
        f"  VertexId == x + y*W for every node: "
        f"{ni['vertex_id_matches_c_order_xy_all']}"
    )
    emit(
        f"  SegmentationId contiguous: "
        f"{lg['segmentation_ids_contiguous']}"
    )
    emit(
        f"  each logical arc has one constant (down,up) pair: "
        f"{lg['all_cells_within_segmentation_share_up_down']}"
    )
    emit(
        f"  each sampled logical arc is one simple path: "
        f"{lg['all_sampled_logical_arcs_are_simple_paths']}"
    )
    emit(
        f"  every sampled-path endpoint matches its referenced nodes: "
        f"{lg['all_sampled_path_endpoints_match_referenced_nodes']}"
    )
    emit(
        f"  logical graph connected / acyclic: "
        f"{lt['connected']} / {lt['acyclic']}"
    )
    emit(
        f"  root candidate(s): {lt['root_candidates']}"
    )
    emit(
        f"  every non-root occurs exactly once as downNodeId: "
        f"{lt['every_nonroot_node_occurs_once_as_downNodeId']}"
    )
    emit(
        f"  TTK node Scalar vs coincident arc Scalar max diff: "
        f"{sc['ttk_node_vs_coincident_arc_scalar_max_abs_diff']:.6g}"
    )
    emit(
        f"  TTK-internal scalar consistency: "
        f"{sc['ttk_internal_scalar_consistency_pass']}"
    )
    emit(
        f"  up-node Scalar >= down-node Scalar on every logical arc: "
        f"{sc['ttk_up_scalar_ge_down_scalar_all']}"
    )
    emit(
        f"  TTK node Scalar vs ORIGINAL wind_speed: "
        f"max diff={sc['ttk_node_vs_original_field_max_abs_diff']:.6g}, "
        f"mean diff={sc['ttk_node_vs_original_field_mean_abs_diff']:.6g}, "
        f"equal={sc['original_field_scalar_equality_pass']}"
    )
    emit(
        f"  HIERARCHY VALIDATION PASS: {m['hierarchy_pass']}"
    )
    emit()

    all_hierarchy_pass &= bool(m["hierarchy_pass"])


emit("OVERALL DECISION")
emit("-" * 100)

if all_hierarchy_pass:
    emit("  ABSTRACT HIERARCHY RECONSTRUCTION: ALLOWED")
    emit()
    emit("  NodeId + SegmentationId + downNodeId + upNodeId form a validated")
    emit("  rooted logical tree for GT/CNN/UV/F1 at the frozen threshold.")
    emit()
    emit("  Recommended first abstract rendering:")
    emit("    - topology/depth-based vertical placement;")
    emit("    - do not use original wind_speed as if it were identical to TTK Scalar;")
    emit("    - optionally show TTK Scalar only as a separately labeled simplified-tree")
    emit("      scalar after noting its distinction from the original field.")
else:
    emit("  ABSTRACT HIERARCHY RECONSTRUCTION: NOT YET ALLOWED")
    emit("  At least one structural validation failed.")

emit()
emit(f"Logical nodes CSV: {OUT_NODES}")
emit(f"Logical arcs CSV:  {OUT_ARCS}")
emit(f"JSON report:       {OUT_JSON}")
emit(f"Summary:           {OUT_TXT}")

OUT_TXT.write_text("\n".join(lines) + "\n")

print()
print(
    "LOGICAL HIERARCHY VALIDATION: "
    + ("PASS" if all_hierarchy_pass else "FAIL")
)
