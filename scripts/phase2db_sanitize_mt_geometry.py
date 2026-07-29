#!/usr/bin/env python3
"""
Validate TTK merge-tree node/arc geometry and write display-only VTUs.

The display-only files preserve points and cells but remove all point, cell,
and field arrays. This is useful when sampled TTK arc attributes contain
uninitialized values but the geometry itself is valid and will be rendered
with a solid color.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import vtk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-vti", type=Path, required=True)
    parser.add_argument("--nodes", type=Path, required=True)
    parser.add_argument("--arcs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_image(path: Path):
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    data = reader.GetOutput()
    if data is None or data.GetNumberOfPoints() == 0:
        raise RuntimeError(f"Could not read image data: {path}")
    return data


def read_grid(path: Path):
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(path))
    reader.Update()
    data = reader.GetOutput()
    if data is None or data.GetNumberOfPoints() == 0:
        raise RuntimeError(f"Could not read unstructured grid: {path}")
    return data


def validate_geometry(label: str, data, input_bounds) -> dict[str, object]:
    n_points = data.GetNumberOfPoints()
    n_cells = data.GetNumberOfCells()

    coords = []
    for point_id in range(n_points):
        point = data.GetPoint(point_id)
        if point is None or len(point) != 3:
            raise RuntimeError(f"{label}: invalid point {point_id}")
        if not all(math.isfinite(float(value)) for value in point):
            raise RuntimeError(f"{label}: non-finite point {point_id}: {point}")
        coords.append(tuple(float(value) for value in point))

    tolerance = 1e-5
    for axis in range(3):
        lower = float(input_bounds[2 * axis]) - tolerance
        upper = float(input_bounds[2 * axis + 1]) + tolerance
        for point_id, point in enumerate(coords):
            value = point[axis]
            if not (lower <= value <= upper):
                raise RuntimeError(
                    f"{label}: point {point_id} axis {axis}={value} lies outside "
                    f"input bound [{lower}, {upper}]"
                )

    cell_types = set()
    total_references = 0
    for cell_id in range(n_cells):
        cell = data.GetCell(cell_id)
        if cell is None:
            raise RuntimeError(f"{label}: missing cell {cell_id}")
        cell_types.add(int(cell.GetCellType()))
        ids = cell.GetPointIds()
        if ids is None or ids.GetNumberOfIds() == 0:
            raise RuntimeError(f"{label}: empty cell {cell_id}")
        total_references += ids.GetNumberOfIds()
        for local_id in range(ids.GetNumberOfIds()):
            point_id = ids.GetId(local_id)
            if point_id < 0 or point_id >= n_points:
                raise RuntimeError(
                    f"{label}: cell {cell_id} references invalid point {point_id}"
                )

    allowed = {
        "nodes": {vtk.VTK_VERTEX, vtk.VTK_POLY_VERTEX},
        "arcs": {vtk.VTK_LINE, vtk.VTK_POLY_LINE},
    }[label]
    if not cell_types.issubset(allowed):
        raise RuntimeError(
            f"{label}: unexpected cell types {sorted(cell_types)}; "
            f"allowed={sorted(allowed)}"
        )

    return {
        "vtk_class": data.GetClassName(),
        "points": int(n_points),
        "cells": int(n_cells),
        "bounds": [float(value) for value in data.GetBounds()],
        "cell_types": sorted(cell_types),
        "cell_point_references": int(total_references),
    }


def geometry_only(data):
    output = vtk.vtkUnstructuredGrid()
    output.DeepCopy(data)
    output.GetPointData().Initialize()
    output.GetCellData().Initialize()
    output.GetFieldData().Initialize()
    return output


def write_grid(data, path: Path) -> None:
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(data)
    writer.SetDataModeToBinary()
    if writer.Write() != 1:
        raise RuntimeError(f"Failed to write {path}")


def array_counts(data) -> dict[str, int]:
    return {
        "point_arrays": int(data.GetPointData().GetNumberOfArrays()),
        "cell_arrays": int(data.GetCellData().GetNumberOfArrays()),
        "field_arrays": int(data.GetFieldData().GetNumberOfArrays()),
    }


def main() -> None:
    args = parse_args()
    input_path = args.input_vti.resolve()
    nodes_path = args.nodes.resolve()
    arcs_path = args.arcs.resolve()
    output_dir = args.output_dir.resolve()

    for path in (input_path, nodes_path, arcs_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise SystemExit(f"Missing or empty input: {path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    field = read_image(input_path)
    nodes = read_grid(nodes_path)
    arcs = read_grid(arcs_path)
    input_bounds = field.GetBounds()

    node_geometry = validate_geometry("nodes", nodes, input_bounds)
    arc_geometry = validate_geometry("arcs", arcs, input_bounds)

    nodes_display = geometry_only(nodes)
    arcs_display = geometry_only(arcs)

    nodes_output = output_dir / "nodes_display.vtu"
    arcs_output = output_dir / "arcs_display.vtu"
    report_output = output_dir / "display_geometry_report.json"

    write_grid(nodes_display, nodes_output)
    write_grid(arcs_display, arcs_output)

    nodes_check = read_grid(nodes_output)
    arcs_check = read_grid(arcs_output)

    expected_counts = {
        "point_arrays": 0,
        "cell_arrays": 0,
        "field_arrays": 0,
    }
    if array_counts(nodes_check) != expected_counts:
        raise RuntimeError("nodes_display.vtu still contains data arrays")
    if array_counts(arcs_check) != expected_counts:
        raise RuntimeError("arcs_display.vtu still contains data arrays")

    report = {
        "input_vti": str(input_path),
        "input_bounds": [float(value) for value in input_bounds],
        "raw_nodes": str(nodes_path),
        "raw_arcs": str(arcs_path),
        "nodes_geometry": node_geometry,
        "arcs_geometry": arc_geometry,
        "display_nodes": str(nodes_output),
        "display_arcs": str(arcs_output),
        "display_nodes_arrays": array_counts(nodes_check),
        "display_arcs_arrays": array_counts(arcs_check),
    }
    report_output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print("DISPLAY GEOMETRY SANITIZATION PASSED")
    print(f"Input bounds: {input_bounds}")
    print(f"Nodes: {node_geometry}")
    print(f"Arcs:  {arc_geometry}")
    print(f"Nodes output: {nodes_output}")
    print(f"Arcs output:  {arcs_output}")
    print(f"Report:       {report_output}")


if __name__ == "__main__":
    main()
