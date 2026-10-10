#!/usr/bin/env python3
"""
Map the validated sample-69 unmatched TTK branch-decomposition nodes back to
original numerical-tree VertexId values and then into the authoritative C-order
160x160 spatial frame.

This stage does NOT alter the matching or recompute any costs.

Frozen provenance/orientation bridge:
    CNN numerical GT/SR -> authoritative C-order : transpose
    UV  numerical GT/SR -> authoritative C-order : transpose
    F1  numerical GT/SR -> authoritative C-order : identity

The script first validates that every preprocessed unmatched node_id and
origin_id still resolves exactly to the original numerical-tree port-0 VTU
(NodeId == VTK point row). Only then are VertexIds spatially transformed.

Outputs:
    cost_attribution/{cnn,uv,f1}_unmatched_branches_spatial.csv
    cost_attribution/sample69_unmatched_spatial_mapping_validation.csv
    cost_attribution/sample69_unmatched_spatial_top_summary.txt
"""

from __future__ import annotations

import csv
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import vtk

W = 160
H = 160

AUDIT = Path.home() / "phire_runtime_audit_20260809_221548"
W22 = Path(os.environ.get("W22", AUDIT / "recompute_pd_w22"))
ROOT = Path.home() / "PhIRE"
BASE = (
    W22
    / "corrected_pd_mt"
    / "discordance_visuals"
    / "sample_069"
    / "ttk_matching_host96"
    / "cost_attribution"
)


@dataclass(frozen=True)
class TreeSpec:
    label: str
    side: str
    role: str
    orientation: str
    nodes_vtu: Path


SPECS: Dict[Tuple[str, str], TreeSpec] = {
    ("cnn", "1_delete"): TreeSpec(
        "cnn", "1_delete", "GT", "transpose",
        ROOT / "ttk_runs_fixed/cnn/mt/"
        "cnn_GT_s69_speed_p160_x0_y0_mt_port_0.vtu",
    ),
    ("cnn", "2_insert"): TreeSpec(
        "cnn", "2_insert", "SR", "transpose",
        ROOT / "ttk_runs_fixed/cnn/mt/"
        "cnn_SR_s69_speed_p160_x0_y0_mt_port_0.vtu",
    ),
    ("uv", "1_delete"): TreeSpec(
        "uv", "1_delete", "GT", "transpose",
        ROOT / "ttk_runs_fixed/topology_finetuning/"
        "candidateUV_expanded2688_topology/mt/GT/"
        "candidateUV_expanded2688_GT_s69_speed_p160_x0_y0_mt_port_0.vtu",
    ),
    ("uv", "2_insert"): TreeSpec(
        "uv", "2_insert", "SR", "transpose",
        ROOT / "ttk_runs_fixed/topology_finetuning/"
        "candidateUV_expanded2688_topology/mt/SR/"
        "candidateUV_expanded2688_SR_s69_speed_p160_x0_y0_mt_port_0.vtu",
    ),
    ("f1", "1_delete"): TreeSpec(
        "f1", "1_delete", "GT", "identity",
        ROOT / "ttk_runs_fixed/topology_finetuning/"
        "candidateF_grad_E2_low_expanded2688_topology/mt/GT/"
        "candidateF_grad_E2_low_expanded2688_GT_s69_speed_p160_x0_y0_mt_port_0.vtu",
    ),
    ("f1", "2_insert"): TreeSpec(
        "f1", "2_insert", "SR", "identity",
        ROOT / "ttk_runs_fixed/topology_finetuning/"
        "candidateF_grad_E2_low_expanded2688_topology/mt/SR/"
        "candidateF_grad_E2_low_expanded2688_SR_s69_speed_p160_x0_y0_mt_port_0.vtu",
    ),
}


def read_csv(path: Path) -> List[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def read_grid(path: Path) -> vtk.vtkUnstructuredGrid:
    if not path.is_file():
        raise FileNotFoundError(path)

    r = vtk.vtkXMLUnstructuredGridReader()
    r.SetFileName(str(path))
    r.Update()

    out = vtk.vtkUnstructuredGrid()
    out.ShallowCopy(r.GetOutput())

    if out.GetNumberOfPoints() <= 0:
        raise RuntimeError(f"{path}: zero points")

    return out


def get_array(grid, name: str):
    a = grid.GetPointData().GetArray(name)
    if a is None:
        names = []
        pd = grid.GetPointData()
        for i in range(pd.GetNumberOfArrays()):
            aa = pd.GetArray(i)
            if aa is not None and aa.GetName():
                names.append(aa.GetName())
        raise RuntimeError(f"Missing array {name!r}; arrays={names}")
    return a


def intlike(a, row: int, name: str) -> int:
    v = float(a.GetTuple1(row))
    iv = int(round(v))
    if not math.isfinite(v) or abs(v - iv) > 1e-8:
        raise RuntimeError(f"{name}[{row}]={v} is not integer-like")
    return iv


def spatialize_vertex_id(v: int, orientation: str) -> Tuple[int, int, int]:
    if not (0 <= v < W * H):
        raise RuntimeError(f"VertexId {v} outside {W}x{H}")

    x = v % W
    y = v // W

    if orientation == "identity":
        cx, cy = x, y
    elif orientation == "transpose":
        cx, cy = y, x
    else:
        raise ValueError(orientation)

    cv = cx + W * cy
    return cv, cx, cy


def endpoint_record(
    grid,
    node_id_arr,
    vertex_id_arr,
    row_id: int,
    orientation: str,
    prefix: str,
) -> dict:
    n = grid.GetNumberOfPoints()
    if not (0 <= row_id < n):
        raise RuntimeError(
            f"{prefix}: id {row_id} outside original VTU rows 0..{n-1}"
        )

    node_id = intlike(node_id_arr, row_id, "NodeId")
    if node_id != row_id:
        raise RuntimeError(
            f"{prefix}: row {row_id} has NodeId {node_id}; "
            "preprocessed id cannot be treated as original row"
        )

    vertex_id = intlike(vertex_id_arr, row_id, "VertexId")
    cv, cx, cy = spatialize_vertex_id(vertex_id, orientation)
    px, py, pz = grid.GetPoint(row_id)

    return {
        f"{prefix}_original_point_row": row_id,
        f"{prefix}_original_NodeId": node_id,
        f"{prefix}_original_VertexId": vertex_id,
        f"{prefix}_original_vtk_x": px,
        f"{prefix}_original_vtk_y": py,
        f"{prefix}_original_vtk_z": pz,
        f"{prefix}_corder_VertexId": cv,
        f"{prefix}_corder_x": cx,
        f"{prefix}_corder_y": cy,
    }


def main() -> int:
    grids = {}
    arrays = {}

    for key, spec in SPECS.items():
        g = read_grid(spec.nodes_vtu)
        grids[key] = g
        arrays[key] = (
            get_array(g, "NodeId"),
            get_array(g, "VertexId"),
        )

    validation_rows = []
    summary_lines = [
        "SAMPLE-69 UNMATCHED-BRANCH SPATIAL MAPPING",
        "=" * 108,
        "",
        "Frozen orientation bridge:",
        "  CNN / UV numerical trees -> transpose -> authoritative C-order",
        "  F1 numerical trees       -> identity  -> authoritative C-order",
        "",
    ]

    for label in ("cnn", "uv", "f1"):
        in_path = BASE / f"{label}_unmatched_branches.csv"
        rows = read_csv(in_path)

        enriched = []
        validation = defaultdict(int)

        # Side totals used for ranking/cumulative contribution.
        side_totals = defaultdict(float)
        for r in rows:
            side_totals[r["side"]] += float(r["nonmatching_cost"])

        # Rank by cost within each method and within each side.
        ordered_all = sorted(
            range(len(rows)),
            key=lambda i: float(rows[i]["nonmatching_cost"]),
            reverse=True,
        )
        rank_all = {idx: rank + 1 for rank, idx in enumerate(ordered_all)}

        rank_side = {}
        for side in ("1_delete", "2_insert"):
            ids = [i for i, r in enumerate(rows) if r["side"] == side]
            ids.sort(
                key=lambda i: float(rows[i]["nonmatching_cost"]),
                reverse=True,
            )
            for rank, idx in enumerate(ids, 1):
                rank_side[idx] = rank

        # Cumulative within each side in descending order.
        cumulative_by_idx = {}
        for side in ("1_delete", "2_insert"):
            ids = [i for i, r in enumerate(rows) if r["side"] == side]
            ids.sort(
                key=lambda i: float(rows[i]["nonmatching_cost"]),
                reverse=True,
            )
            c = 0.0
            total = side_totals[side]
            for idx in ids:
                c += float(rows[idx]["nonmatching_cost"])
                cumulative_by_idx[idx] = (
                    100.0 * c / total if total else float("nan")
                )

        for i, r in enumerate(rows):
            side = r["side"]
            key = (label, side)
            spec = SPECS[key]
            g = grids[key]
            node_id_arr, vertex_id_arr = arrays[key]

            node_id = int(r["node_id"])
            origin_id = int(r["origin_id"])

            rec = dict(r)
            rec["method"] = label
            rec["tree_role"] = spec.role
            rec["orientation_to_corder"] = spec.orientation
            rec["rank_all_unmatched"] = rank_all[i]
            rec["rank_within_side"] = rank_side[i]
            rec["cumulative_side_cost_percent"] = cumulative_by_idx[i]

            rec.update(
                endpoint_record(
                    g, node_id_arr, vertex_id_arr,
                    node_id, spec.orientation, "node"
                )
            )
            rec.update(
                endpoint_record(
                    g, node_id_arr, vertex_id_arr,
                    origin_id, spec.orientation, "origin"
                )
            )

            rec["corder_mid_x"] = (
                float(rec["node_corder_x"])
                + float(rec["origin_corder_x"])
            ) / 2.0
            rec["corder_mid_y"] = (
                float(rec["node_corder_y"])
                + float(rec["origin_corder_y"])
            ) / 2.0

            validation[f"{side}_rows"] += 1
            validation[f"{side}_nodeid_exact"] += 1
            validation[f"{side}_originid_exact"] += 1

            enriched.append(rec)

        out_path = BASE / f"{label}_unmatched_branches_spatial.csv"
        with out_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(enriched[0].keys()))
            w.writeheader()
            w.writerows(enriched)

        val_row = {
            "method": label,
            "total_rows": len(enriched),
            "delete_rows": validation["1_delete_rows"],
            "insert_rows": validation["2_insert_rows"],
            "all_node_ids_resolve_to_original_NodeId_row": 1,
            "all_origin_ids_resolve_to_original_NodeId_row": 1,
            "delete_orientation": SPECS[(label, "1_delete")].orientation,
            "insert_orientation": SPECS[(label, "2_insert")].orientation,
            "output_csv": str(out_path),
        }
        validation_rows.append(val_row)

        summary_lines.extend([
            label.upper(),
            "-" * 108,
            (
                f"mapped rows: {len(enriched)} "
                f"(delete={validation['1_delete_rows']}, "
                f"insert={validation['2_insert_rows']})"
            ),
            (
                "NodeId/row validation: PASS for every unmatched node "
                "and every branch origin"
            ),
            f"spatial CSV: {out_path}",
            "",
            "Top 8 unmatched branches by exact nonmatching cost:",
            (
                "rank side      cost          node->origin  "
                "C-order node(x,y) -> origin(x,y)"
            ),
        ])

        top = sorted(
            enriched,
            key=lambda r: float(r["nonmatching_cost"]),
            reverse=True,
        )[:8]

        for r in top:
            summary_lines.append(
                f"{int(r['rank_all_unmatched']):>4d} "
                f"{r['side']:<9s} "
                f"{float(r['nonmatching_cost']):>12.8f} "
                f"{int(r['node_id']):>5d}->{int(r['origin_id']):<5d} "
                f"({int(r['node_corder_x']):>3d},"
                f"{int(r['node_corder_y']):>3d}) -> "
                f"({int(r['origin_corder_x']):>3d},"
                f"{int(r['origin_corder_y']):>3d})"
            )

        summary_lines.append("")

    validation_path = BASE / "sample69_unmatched_spatial_mapping_validation.csv"
    with validation_path.open("w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=list(validation_rows[0].keys())
        )
        w.writeheader()
        w.writerows(validation_rows)

    summary_path = BASE / "sample69_unmatched_spatial_top_summary.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n")

    print("\n".join(summary_lines))
    print("VALIDATION CSV:", validation_path)
    print("SUMMARY:", summary_path)
    print()
    print("UNMATCHED BRANCH -> AUTHORITATIVE C-ORDER SPATIAL MAPPING: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
