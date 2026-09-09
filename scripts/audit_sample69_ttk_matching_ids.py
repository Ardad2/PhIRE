#!/usr/bin/env python3
"""
Audit sample-69 TTK merge-tree matching IDs against the ORIGINAL numerical-tree
node VTUs used by the host VTK 9.6 / TTK distance computation.

This script deliberately does NOT attempt to map the numerical tree to the
new threshold-3 qualitative tree.  Its job is only to establish what the
postprocessed/converted matching node IDs mean relative to the original
numerical MT node grids.

Expected environment:
    /usr/bin/python3
    VTK 9.6.0
    host topologytoolkit installation

Inputs:
    $W22/corrected_pd_mt/discordance_visuals/sample_069/ttk_matching_host96/
        {cnn,uv,f1}_{raw,converted}_matching.csv
        {cnn,uv,f1}_{raw,converted}_summary.csv

    Original numerical MT port-0 VTUs for GT/SR.

Outputs:
    .../ttk_matching_host96/
        {cnn,uv,f1}_converted_matching_enriched.csv
        sample69_ttk_matching_id_audit_summary.txt
        sample69_ttk_matching_id_audit.csv
"""

from __future__ import annotations

import csv
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import vtk


@dataclass(frozen=True)
class PairSpec:
    label: str
    gt_nodes: Path
    sr_nodes: Path


def read_ugrid(path: Path) -> vtk.vtkUnstructuredGrid:
    if not path.is_file():
        raise FileNotFoundError(path)

    r = vtk.vtkXMLUnstructuredGridReader()
    r.SetFileName(str(path))
    r.Update()

    out = vtk.vtkUnstructuredGrid()
    out.ShallowCopy(r.GetOutput())

    if out.GetNumberOfPoints() <= 0:
        raise RuntimeError(f"{path}: reader returned zero points")

    return out


def read_csv(path: Path) -> List[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def array_names(grid: vtk.vtkUnstructuredGrid) -> List[str]:
    pd = grid.GetPointData()
    ans = []
    for i in range(pd.GetNumberOfArrays()):
        a = pd.GetArray(i)
        if a is not None and a.GetName():
            ans.append(a.GetName())
    return ans


def find_array_name(
    grid: vtk.vtkUnstructuredGrid,
    candidates: Tuple[str, ...],
) -> Optional[str]:
    names = array_names(grid)
    lower_to_actual = {n.lower(): n for n in names}
    for c in candidates:
        if c.lower() in lower_to_actual:
            return lower_to_actual[c.lower()]
    return None


def get_scalar_value(grid, array_name: Optional[str], row: int):
    if array_name is None:
        return ""
    a = grid.GetPointData().GetArray(array_name)
    if a is None:
        return ""
    if a.GetNumberOfComponents() == 1:
        return a.GetTuple1(row)
    return "|".join(str(x) for x in a.GetTuple(row))


def get_int_value(grid, array_name: Optional[str], row: int):
    v = get_scalar_value(grid, array_name, row)
    if v == "":
        return ""
    try:
        fv = float(v)
    except Exception:
        return v
    if math.isfinite(fv) and abs(fv - round(fv)) <= 1e-9:
        return int(round(fv))
    return fv


@dataclass
class NodeIndexAudit:
    npoints: int
    node_id_name: Optional[str]
    vertex_id_name: Optional[str]
    scalar_name: Optional[str]
    critical_type_name: Optional[str]
    node_id_unique: Optional[bool]
    node_id_identity: Optional[bool]
    node_id_lookup: Dict[int, int]


def audit_node_index(grid: vtk.vtkUnstructuredGrid) -> NodeIndexAudit:
    n = grid.GetNumberOfPoints()

    node_id_name = find_array_name(grid, ("NodeId", "NodeID", "nodeId"))
    vertex_id_name = find_array_name(grid, ("VertexId", "VertexID", "vertexId"))
    scalar_name = find_array_name(grid, ("Scalar", "scalar"))
    critical_type_name = find_array_name(
        grid, ("CriticalType", "criticalType", "Critical Type")
    )

    node_id_lookup: Dict[int, int] = {}
    node_id_unique: Optional[bool] = None
    node_id_identity: Optional[bool] = None

    if node_id_name is not None:
        a = grid.GetPointData().GetArray(node_id_name)
        vals: List[int] = []

        for i in range(n):
            v = float(a.GetTuple1(i))
            if not math.isfinite(v) or abs(v - round(v)) > 1e-9:
                raise RuntimeError(
                    f"{node_id_name}: non-integer-like value at row {i}: {v}"
                )
            vals.append(int(round(v)))

        node_id_unique = len(set(vals)) == len(vals)
        node_id_identity = all(v == i for i, v in enumerate(vals))

        if node_id_unique:
            node_id_lookup = {v: i for i, v in enumerate(vals)}

    return NodeIndexAudit(
        npoints=n,
        node_id_name=node_id_name,
        vertex_id_name=vertex_id_name,
        scalar_name=scalar_name,
        critical_type_name=critical_type_name,
        node_id_unique=node_id_unique,
        node_id_identity=node_id_identity,
        node_id_lookup=node_id_lookup,
    )


def resolve_matching_id(
    match_id: int,
    audit: NodeIndexAudit,
) -> Tuple[int, str]:
    """
    Return (point-row, semantics).

    Prefer an exact NodeId-array lookup when available.  If NodeId is identical
    to point-row index, report that identity explicitly.
    """
    in_row_range = 0 <= match_id < audit.npoints

    if audit.node_id_name is not None and audit.node_id_unique:
        if match_id in audit.node_id_lookup:
            row = audit.node_id_lookup[match_id]
            if in_row_range and row == match_id:
                return row, "NodeId==point_row"
            return row, "NodeId_value"

    if in_row_range:
        return match_id, "point_row_only"

    raise RuntimeError(
        f"matching id {match_id} cannot be resolved against {audit.npoints} points"
    )


def node_record(
    grid: vtk.vtkUnstructuredGrid,
    audit: NodeIndexAudit,
    match_id: int,
    prefix: str,
) -> Dict[str, object]:
    row, semantics = resolve_matching_id(match_id, audit)
    x, y, z = grid.GetPoint(row)

    return {
        f"{prefix}_match_id": match_id,
        f"{prefix}_point_row": row,
        f"{prefix}_id_semantics": semantics,
        f"{prefix}_NodeId": get_int_value(grid, audit.node_id_name, row),
        f"{prefix}_VertexId": get_int_value(grid, audit.vertex_id_name, row),
        f"{prefix}_Scalar": get_scalar_value(grid, audit.scalar_name, row),
        f"{prefix}_CriticalType": get_int_value(
            grid, audit.critical_type_name, row
        ),
        f"{prefix}_x": x,
        f"{prefix}_y": y,
        f"{prefix}_z": z,
    }


def main() -> int:
    root = Path.home() / "PhIRE"
    audit_root = Path.home() / "phire_runtime_audit_20260809_221548"
    w22 = Path(os.environ.get("W22", audit_root / "recompute_pd_w22"))

    out = (
        w22
        / "corrected_pd_mt"
        / "discordance_visuals"
        / "sample_069"
        / "ttk_matching_host96"
    )

    specs = [
        PairSpec(
            "cnn",
            root / "ttk_runs_fixed/cnn/mt/"
            "cnn_GT_s69_speed_p160_x0_y0_mt_port_0.vtu",
            root / "ttk_runs_fixed/cnn/mt/"
            "cnn_SR_s69_speed_p160_x0_y0_mt_port_0.vtu",
        ),
        PairSpec(
            "uv",
            root / "ttk_runs_fixed/topology_finetuning/"
            "candidateUV_expanded2688_topology/mt/GT/"
            "candidateUV_expanded2688_GT_s69_speed_p160_x0_y0_mt_port_0.vtu",
            root / "ttk_runs_fixed/topology_finetuning/"
            "candidateUV_expanded2688_topology/mt/SR/"
            "candidateUV_expanded2688_SR_s69_speed_p160_x0_y0_mt_port_0.vtu",
        ),
        PairSpec(
            "f1",
            root / "ttk_runs_fixed/topology_finetuning/"
            "candidateF_grad_E2_low_expanded2688_topology/mt/GT/"
            "candidateF_grad_E2_low_expanded2688_GT_s69_speed_p160_x0_y0_mt_port_0.vtu",
            root / "ttk_runs_fixed/topology_finetuning/"
            "candidateF_grad_E2_low_expanded2688_topology/mt/SR/"
            "candidateF_grad_E2_low_expanded2688_SR_s69_speed_p160_x0_y0_mt_port_0.vtu",
        ),
    ]

    audit_rows: List[Dict[str, object]] = []
    report: List[str] = []

    report.append("SAMPLE-69 TTK MATCHING-ID AUDIT")
    report.append("=" * 88)
    report.append(f"python: {sys.executable}")
    report.append(f"VTK: {vtk.vtkVersion.GetVTKVersion()}")
    report.append("")
    report.append(
        "Scope: converted matching IDs vs ORIGINAL numerical-tree port-0 VTUs."
    )
    report.append(
        "No threshold-3 qualitative-tree mapping is attempted in this stage."
    )
    report.append("")

    for spec in specs:
        raw_matching_path = out / f"{spec.label}_raw_matching.csv"
        conv_matching_path = out / f"{spec.label}_converted_matching.csv"

        raw = read_csv(raw_matching_path)
        conv = read_csv(conv_matching_path)

        gt = read_ugrid(spec.gt_nodes)
        sr = read_ugrid(spec.sr_nodes)

        ga = audit_node_index(gt)
        sa = audit_node_index(sr)

        enriched: List[Dict[str, object]] = []
        gt_ids: List[int] = []
        sr_ids: List[int] = []

        for i, r in enumerate(conv):
            g_id = int(r["tree1_node_id"])
            s_id = int(r["tree2_node_id"])
            cost = float(r["relabel_cost"])

            gt_ids.append(g_id)
            sr_ids.append(s_id)

            row = {
                "matching_row": i,
                "relabel_cost": cost,
            }
            row.update(node_record(gt, ga, g_id, "gt"))
            row.update(node_record(sr, sa, s_id, "sr"))
            enriched.append(row)

        unique_pairs = len(
            {
                (int(r["tree1_node_id"]), int(r["tree2_node_id"]))
                for r in conv
            }
        )
        unique_gt = len(set(gt_ids))
        unique_sr = len(set(sr_ids))

        conv_is_exact_2x = len(conv) == 2 * len(raw)

        enriched_path = out / f"{spec.label}_converted_matching_enriched.csv"
        with enriched_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(enriched[0].keys()))
            w.writeheader()
            w.writerows(enriched)

        audit_row = {
            "label": spec.label,
            "raw_matching_count": len(raw),
            "converted_matching_count": len(conv),
            "converted_is_exact_2x_raw": int(conv_is_exact_2x),
            "unique_converted_pairs": unique_pairs,
            "unique_gt_matched_ids": unique_gt,
            "unique_sr_matched_ids": unique_sr,
            "gt_input_nodes": ga.npoints,
            "sr_input_nodes": sa.npoints,
            "gt_NodeId_array": ga.node_id_name or "",
            "sr_NodeId_array": sa.node_id_name or "",
            "gt_NodeId_unique": "" if ga.node_id_unique is None else int(ga.node_id_unique),
            "sr_NodeId_unique": "" if sa.node_id_unique is None else int(sa.node_id_unique),
            "gt_NodeId_identity": "" if ga.node_id_identity is None else int(ga.node_id_identity),
            "sr_NodeId_identity": "" if sa.node_id_identity is None else int(sa.node_id_identity),
            "gt_VertexId_array": ga.vertex_id_name or "",
            "sr_VertexId_array": sa.vertex_id_name or "",
            "gt_Scalar_array": ga.scalar_name or "",
            "sr_Scalar_array": sa.scalar_name or "",
            "gt_CriticalType_array": ga.critical_type_name or "",
            "sr_CriticalType_array": sa.critical_type_name or "",
            "enriched_csv": str(enriched_path),
        }
        audit_rows.append(audit_row)

        report.append(spec.label.upper())
        report.append("-" * 88)
        report.append(
            f"raw matches: {len(raw)}; converted: {len(conv)}; "
            f"converted == 2*raw: {conv_is_exact_2x}"
        )
        report.append(
            f"unique converted pairs: {unique_pairs}; "
            f"unique GT matched IDs: {unique_gt}/{ga.npoints}; "
            f"unique SR matched IDs: {unique_sr}/{sa.npoints}"
        )
        report.append(
            f"GT arrays: NodeId={ga.node_id_name!r}, "
            f"VertexId={ga.vertex_id_name!r}, Scalar={ga.scalar_name!r}, "
            f"CriticalType={ga.critical_type_name!r}"
        )
        report.append(
            f"SR arrays: NodeId={sa.node_id_name!r}, "
            f"VertexId={sa.vertex_id_name!r}, Scalar={sa.scalar_name!r}, "
            f"CriticalType={sa.critical_type_name!r}"
        )
        report.append(
            f"GT NodeId unique={ga.node_id_unique}, "
            f"NodeId==point-row={ga.node_id_identity}"
        )
        report.append(
            f"SR NodeId unique={sa.node_id_unique}, "
            f"NodeId==point-row={sa.node_id_identity}"
        )
        report.append(f"enriched: {enriched_path}")
        report.append("")

    audit_csv = out / "sample69_ttk_matching_id_audit.csv"
    with audit_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(audit_rows[0].keys()))
        w.writeheader()
        w.writerows(audit_rows)

    summary = out / "sample69_ttk_matching_id_audit_summary.txt"
    summary.write_text("\n".join(report) + "\n")

    print("\n".join(report))
    print("AUDIT CSV:", audit_csv)
    print("SUMMARY:", summary)
    print()
    print("MATCHING-ID AUDIT: COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
