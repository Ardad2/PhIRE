#!/usr/bin/env python3
"""
Compute a persistence-simplified Join Tree with TTK's native Python wrappers.

Designed for the phire-ttk:latest Docker image.

Example:
    python3 phase2db_extract_simplified_mt.py \
      --input /inputs/inputs/figure_01/cnn.vti \
      --output-dir /outputs/figure_01/cnn \
      --threshold 11.0 \
      --arc-sampling 10 \
      --threads 20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import vtk
from topologytoolkit.ttkMergeTree import ttkMergeTree
from topologytoolkit.ttkTopologicalSimplificationByPersistence import (
    ttkTopologicalSimplificationByPersistence,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=11.0)
    parser.add_argument("--arc-sampling", type=int, default=10)
    parser.add_argument("--threads", type=int, default=20)
    return parser.parse_args()


def require_file(path: Path) -> Path:
    path = path.resolve()
    if not path.is_file():
        raise SystemExit(f"Missing input file: {path}")
    return path


def write_vtu(data, path: Path) -> None:
    grid = vtk.vtkUnstructuredGrid.SafeDownCast(data)
    if grid is None:
        raise RuntimeError(
            f"Expected vtkUnstructuredGrid for {path.name}, "
            f"got {data.GetClassName() if data else 'None'}"
        )

    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(grid)
    writer.SetDataModeToBinary()
    if writer.Write() != 1:
        raise RuntimeError(f"Failed to write {path}")


def write_vti(data, path: Path) -> None:
    image = vtk.vtkImageData.SafeDownCast(data)
    if image is None:
        raise RuntimeError(
            f"Expected vtkImageData for {path.name}, "
            f"got {data.GetClassName() if data else 'None'}"
        )

    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(image)
    writer.SetDataModeToBinary()
    if writer.Write() != 1:
        raise RuntimeError(f"Failed to write {path}")


def array_summary(attributes) -> list[dict[str, object]]:
    rows = []
    for index in range(attributes.GetNumberOfArrays()):
        array = attributes.GetArray(index)
        if array is None:
            continue
        value_range = array.GetRange(0) if array.GetNumberOfComponents() else (None, None)
        rows.append(
            {
                "name": array.GetName(),
                "components": array.GetNumberOfComponents(),
                "tuples": array.GetNumberOfTuples(),
                "range": [float(value_range[0]), float(value_range[1])],
            }
        )
    return rows


def dataset_summary(data) -> dict[str, object]:
    return {
        "vtk_class": data.GetClassName(),
        "points": int(data.GetNumberOfPoints()),
        "cells": int(data.GetNumberOfCells()),
        "point_arrays": array_summary(data.GetPointData()),
        "cell_arrays": array_summary(data.GetCellData()),
    }


def main() -> None:
    args = parse_args()
    input_path = require_file(args.input)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(str(input_path))
    reader.Update()

    input_data = reader.GetOutput()
    if input_data is None or input_data.GetNumberOfPoints() == 0:
        raise RuntimeError(f"Reader produced no data for {input_path}")

    scalar = input_data.GetPointData().GetArray("wind_speed")
    if scalar is None:
        available = [
            input_data.GetPointData().GetArrayName(i)
            for i in range(input_data.GetPointData().GetNumberOfArrays())
        ]
        raise RuntimeError(
            f"'wind_speed' is missing from {input_path}; available={available}"
        )

    simplification = ttkTopologicalSimplificationByPersistence()
    simplification.SetInputConnection(reader.GetOutputPort())
    simplification.SetInputArrayToProcess(
        0,
        0,
        0,
        vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS,
        "wind_speed",
    )
    simplification.SetPairType(0)  # EXTREMUM_SADDLE
    simplification.SetPersistenceThreshold(float(args.threshold))
    simplification.SetThresholdIsAbsolute(True)
    simplification.SetComputePerturbation(False)
    simplification.SetThreadNumber(int(args.threads))
    simplification.SetDebugLevel(3)
    simplification.Update()

    simplified = simplification.GetOutputDataObject(0)
    if simplified is None:
        raise RuntimeError("Persistence simplification produced no output")

    merge_tree = ttkMergeTree()
    merge_tree.SetInputConnection(simplification.GetOutputPort())
    merge_tree.SetInputArrayToProcess(
        0,
        0,
        0,
        vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS,
        "wind_speed",
    )
    merge_tree.SetBackend(0)  # FTM
    merge_tree.SetTreeType(0)  # Join Tree
    merge_tree.SetSuperArcSamplingLevel(int(args.arc_sampling))
    merge_tree.SetWithSegmentation(True)
    merge_tree.SetWithNormalize(True)
    merge_tree.SetWithAdvStats(True)
    merge_tree.SetThreadNumber(int(args.threads))
    merge_tree.SetDebugLevel(3)
    merge_tree.Update()

    nodes = merge_tree.GetOutputDataObject(0)
    arcs = merge_tree.GetOutputDataObject(1)
    segmentation = merge_tree.GetOutputDataObject(2)

    if nodes is None or arcs is None or segmentation is None:
        raise RuntimeError(
            "Merge tree did not produce all three expected output ports"
        )

    simplified_path = output_dir / "simplified_p11.vti"
    nodes_path = output_dir / "nodes.vtu"
    arcs_path = output_dir / "arcs.vtu"
    segmentation_path = output_dir / "segmentation.vti"
    summary_path = output_dir / "summary.json"

    write_vti(simplified, simplified_path)
    write_vtu(nodes, nodes_path)
    write_vtu(arcs, arcs_path)
    write_vti(segmentation, segmentation_path)

    summary = {
        "input": str(input_path),
        "settings": {
            "scalar_array": "wind_speed",
            "pair_type": "EXTREMUM_SADDLE",
            "pair_type_value": 0,
            "persistence_threshold": float(args.threshold),
            "threshold_is_absolute": True,
            "compute_perturbation": False,
            "merge_tree_backend": "FTM",
            "merge_tree_backend_value": 0,
            "tree_type": "Join",
            "tree_type_value": 0,
            "arc_sampling": int(args.arc_sampling),
            "with_segmentation": True,
            "with_normalize": True,
            "with_advanced_statistics": True,
            "threads": int(args.threads),
            "vtk_version": vtk.vtkVersion.GetVTKVersion(),
        },
        "input_data": dataset_summary(input_data),
        "simplified_data": dataset_summary(simplified),
        "nodes": dataset_summary(nodes),
        "arcs": dataset_summary(arcs),
        "segmentation": dataset_summary(segmentation),
        "outputs": {
            "simplified": simplified_path.name,
            "nodes": nodes_path.name,
            "arcs": arcs_path.name,
            "segmentation": segmentation_path.name,
        },
    }

    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print()
    print("SPARK SIMPLIFIED MERGE-TREE EXTRACTION PASSED")
    print(f"Input:        {input_path}")
    print(f"Output dir:   {output_dir}")
    print(f"Nodes:        {nodes.GetNumberOfPoints()} points, {nodes.GetNumberOfCells()} cells")
    print(f"Arcs:         {arcs.GetNumberOfPoints()} points, {arcs.GetNumberOfCells()} cells")
    print(
        f"Segmentation: {segmentation.GetNumberOfPoints()} points, "
        f"{segmentation.GetNumberOfCells()} cells"
    )
    print(f"Summary:      {summary_path}")

    # TTK 1.3.0's Python wrapper can abort while native objects are
    # destroyed after successful output generation. Flush all messages and
    # terminate without running the faulty native teardown path.
    import os
    import sys

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
