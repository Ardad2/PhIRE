#!/usr/bin/env python3
"""
Phase 5A Step 2 — sample-69 persistence-pair provenance extraction.

Purpose
-------
Extract an auditable table for every persistence-pair cell from the six
authoritative sample-69 TTK PD VTUs used by the current three-method study:

  - Pretrained CNN: GT / SR
  - Reconstruction-only: GT / SR
  - Topology-inspired: GT / SR

For every pair, preserve:
  * PairIdentifier, PairType, IsFinite
  * Birth, Death, Persistence
  * birth/death critical type
  * raw TTK input vertex id
  * raw TTK spatial Coordinates
  * canonicalized x/y Coordinates

Important conventions
---------------------
1. The VTU geometry points themselves are persistence-diagram display
   coordinates, NOT physical field coordinates.
2. The point-data array "Coordinates" stores the original scalar-grid
   coordinates of the critical points.
3. The point-data array "ttkVertexScalarField" stores the corresponding
   input-grid vertex id.
4. The historical topology-inspired track is transposed relative to the
   CNN / reconstruction-only orientation. Therefore this script preserves
   BOTH raw coordinates and canonical coordinates, with swap_xy applied
   to the topology-inspired track.
5. No GT<->SR feature matching is performed in this step.

Expected raw vertex convention for a 160x160 field:
    vertex_id = y * 160 + x
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import vtk


GRID_W = 160
GRID_H = 160


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_vtu(path: Path):
    r = vtk.vtkXMLUnstructuredGridReader()
    r.SetFileName(str(path))
    r.Update()
    g = r.GetOutput()
    if g is None:
        raise RuntimeError(f"Could not read {path}")
    return g


def scalar(arr, i):
    return arr.GetTuple1(i)


def get_required_point_array(pd, name):
    a = pd.GetArray(name)
    if a is None:
        raise RuntimeError(f"Missing point array {name}")
    return a


def get_required_cell_array(cd, name):
    a = cd.GetArray(name)
    if a is None:
        raise RuntimeError(f"Missing cell array {name}")
    return a


def canonical_xy(x: int, y: int, transform: str):
    if transform == "identity":
        return x, y
    if transform == "swap_xy":
        return y, x
    raise ValueError(transform)


def parse_pairs(path: Path, method: str, field_kind: str, transform: str):
    g = read_vtu(path)
    pd = g.GetPointData()
    cd = g.GetCellData()

    vid_arr = get_required_point_array(pd, "ttkVertexScalarField")
    crit_arr = get_required_point_array(pd, "CriticalType")
    coord_arr = get_required_point_array(pd, "Coordinates")

    pair_id_arr = get_required_cell_array(cd, "PairIdentifier")
    pair_type_arr = get_required_cell_array(cd, "PairType")
    pers_arr = get_required_cell_array(cd, "Persistence")
    birth_arr = get_required_cell_array(cd, "Birth")
    finite_arr = get_required_cell_array(cd, "IsFinite")

    rows = []
    vid_coord_failures = []
    pd_geometry_failures = []

    for ci in range(g.GetNumberOfCells()):
        cell = g.GetCell(ci)
        if cell.GetNumberOfPoints() != 2:
            raise RuntimeError(f"{path}: cell {ci} has {cell.GetNumberOfPoints()} points")

        p0 = int(cell.GetPointId(0))
        p1 = int(cell.GetPointId(1))

        pair_id = int(round(scalar(pair_id_arr, ci)))
        pair_type = int(round(scalar(pair_type_arr, ci)))
        persistence = float(scalar(pers_arr, ci))
        birth = float(scalar(birth_arr, ci))
        death = birth + persistence
        is_finite = int(round(scalar(finite_arr, ci)))

        # TTK PD geometry convention observed in authoritative files:
        # point 0 = (birth, birth, 0), point 1 = (birth, death, 0).
        xyz0 = tuple(float(v) for v in g.GetPoint(p0))
        xyz1 = tuple(float(v) for v in g.GetPoint(p1))

        tol = 2e-5
        geom_ok = (
            abs(xyz0[0] - birth) <= tol
            and abs(xyz0[1] - birth) <= tol
            and abs(xyz1[0] - birth) <= tol
            and abs(xyz1[1] - death) <= tol
        )
        if not geom_ok:
            pd_geometry_failures.append({
                "cell": ci,
                "pair_id": pair_id,
                "birth": birth,
                "death": death,
                "p0_xyz": xyz0,
                "p1_xyz": xyz1,
            })

        b_vid = int(round(scalar(vid_arr, p0)))
        d_vid = int(round(scalar(vid_arr, p1)))
        b_crit = int(round(scalar(crit_arr, p0)))
        d_crit = int(round(scalar(crit_arr, p1)))

        b_xyz = tuple(float(v) for v in coord_arr.GetTuple(p0))
        d_xyz = tuple(float(v) for v in coord_arr.GetTuple(p1))

        b_x_raw = int(round(b_xyz[0]))
        b_y_raw = int(round(b_xyz[1]))
        d_x_raw = int(round(d_xyz[0]))
        d_y_raw = int(round(d_xyz[1]))

        if not (0 <= b_x_raw < GRID_W and 0 <= b_y_raw < GRID_H):
            raise RuntimeError(f"{path}: birth coordinate out of bounds: {(b_x_raw,b_y_raw)}")
        if not (0 <= d_x_raw < GRID_W and 0 <= d_y_raw < GRID_H):
            raise RuntimeError(f"{path}: death coordinate out of bounds: {(d_x_raw,d_y_raw)}")

        b_vid_expected = b_y_raw * GRID_W + b_x_raw
        d_vid_expected = d_y_raw * GRID_W + d_x_raw

        if b_vid != b_vid_expected:
            vid_coord_failures.append({
                "cell": ci, "endpoint": "birth",
                "vid": b_vid, "expected": b_vid_expected,
                "xy": [b_x_raw, b_y_raw],
            })
        if d_vid != d_vid_expected:
            vid_coord_failures.append({
                "cell": ci, "endpoint": "death",
                "vid": d_vid, "expected": d_vid_expected,
                "xy": [d_x_raw, d_y_raw],
            })

        b_x_can, b_y_can = canonical_xy(b_x_raw, b_y_raw, transform)
        d_x_can, d_y_can = canonical_xy(d_x_raw, d_y_raw, transform)

        rows.append({
            "method": method,
            "field_kind": field_kind,
            "source_vtu": str(path),
            "source_sha256": sha256(path),
            "cell_index": ci,
            "pair_identifier": pair_id,
            "pair_type": pair_type,
            "is_finite": is_finite,
            "birth": birth,
            "death": death,
            "persistence": persistence,
            "birth_critical_type": b_crit,
            "death_critical_type": d_crit,
            "birth_vertex_id_raw": b_vid,
            "death_vertex_id_raw": d_vid,
            "birth_x_raw": b_x_raw,
            "birth_y_raw": b_y_raw,
            "death_x_raw": d_x_raw,
            "death_y_raw": d_y_raw,
            "coord_transform_to_canonical": transform,
            "birth_x_canonical": b_x_can,
            "birth_y_canonical": b_y_can,
            "death_x_canonical": d_x_can,
            "death_y_canonical": d_y_can,
            "pd_point0_x": xyz0[0],
            "pd_point0_y": xyz0[1],
            "pd_point1_x": xyz1[0],
            "pd_point1_y": xyz1[1],
        })

    return rows, {
        "path": str(path),
        "sha256": sha256(path),
        "points": g.GetNumberOfPoints(),
        "cells": g.GetNumberOfCells(),
        "finite_cells": sum(r["is_finite"] == 1 for r in rows),
        "nonfinite_cells": sum(r["is_finite"] == 0 for r in rows),
        "pair_type_counts": {
            str(t): sum(r["pair_type"] == t for r in rows)
            for t in sorted(set(r["pair_type"] for r in rows))
        },
        "vid_coordinate_failures": vid_coord_failures,
        "pd_geometry_failures": pd_geometry_failures,
        "transform": transform,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phire", default=str(Path.home() / "PhIRE"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    root = Path(args.phire).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    specs = [
        (
            "pretrained_cnn", "GT", "identity",
            root / "ttk_runs_fixed/cnn/pd/cnn_GT_s69_speed_p160_x0_y0_pd_port_0.vtu"
        ),
        (
            "pretrained_cnn", "SR", "identity",
            root / "ttk_runs_fixed/cnn/pd/cnn_SR_s69_speed_p160_x0_y0_pd_port_0.vtu"
        ),
        (
            "reconstruction_only", "GT", "identity",
            root / "ttk_runs_fixed/topology_finetuning/candidateUV_expanded2688_topology/pd/GT/candidateUV_expanded2688_GT_s69_speed_p160_x0_y0_pd_port_0.vtu"
        ),
        (
            "reconstruction_only", "SR", "identity",
            root / "ttk_runs_fixed/topology_finetuning/candidateUV_expanded2688_topology/pd/SR/candidateUV_expanded2688_SR_s69_speed_p160_x0_y0_pd_port_0.vtu"
        ),
        (
            "topology_inspired", "GT", "swap_xy",
            root / "ttk_runs_fixed/topology_finetuning/candidateF_grad_E2_low_expanded2688_topology/pd/GT/candidateF_grad_E2_low_expanded2688_GT_s69_speed_p160_x0_y0_pd_port_0.vtu"
        ),
        (
            "topology_inspired", "SR", "swap_xy",
            root / "ttk_runs_fixed/topology_finetuning/candidateF_grad_E2_low_expanded2688_topology/pd/SR/candidateF_grad_E2_low_expanded2688_SR_s69_speed_p160_x0_y0_pd_port_0.vtu"
        ),
    ]

    all_rows = []
    summaries = []

    for method, field_kind, transform, path in specs:
        if not path.exists():
            raise FileNotFoundError(path)

        rows, summary = parse_pairs(path, method, field_kind, transform)
        all_rows.extend(rows)
        summary["method"] = method
        summary["field_kind"] = field_kind
        summaries.append(summary)

        csv_path = out / f"{method}_{field_kind}_sample69_pair_provenance.csv"
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    combined = out / "sample69_all_pair_provenance.csv"
    with combined.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)

    # GT canonical-coordinate cross-track audit for cells that share the exact
    # same (PairType, Birth, Persistence) key. This is diagnostic only.
    gt_by_method = {}
    for method in ("pretrained_cnn", "reconstruction_only", "topology_inspired"):
        gt_by_method[method] = [
            r for r in all_rows
            if r["method"] == method and r["field_kind"] == "GT"
        ]

    def key(r):
        return (
            r["pair_type"],
            r["is_finite"],
            round(r["birth"], 7),
            round(r["persistence"], 7),
        )

    maps = {m: {key(r): r for r in rs} for m, rs in gt_by_method.items()}
    common_keys = set.intersection(*(set(m.keys()) for m in maps.values()))

    mismatch = []
    for k in sorted(common_keys):
        base = maps["pretrained_cnn"][k]
        expected = (
            base["birth_x_canonical"], base["birth_y_canonical"],
            base["death_x_canonical"], base["death_y_canonical"],
        )
        for m in ("reconstruction_only", "topology_inspired"):
            r = maps[m][k]
            got = (
                r["birth_x_canonical"], r["birth_y_canonical"],
                r["death_x_canonical"], r["death_y_canonical"],
            )
            if got != expected:
                mismatch.append({
                    "key": k,
                    "method": m,
                    "expected": expected,
                    "got": got,
                })

    audit = {
        "sample": 69,
        "grid": [GRID_W, GRID_H],
        "files": summaries,
        "gt_common_exact_pair_keys_after_round7": len(common_keys),
        "gt_canonical_coordinate_mismatches_on_common_keys": mismatch,
        "notes": [
            "VTU point geometry is persistence-diagram display geometry.",
            'Point-data "Coordinates" is the spatial critical-point coordinate source.',
            'Point-data "ttkVertexScalarField" is the raw input-grid vertex id.',
            "Topology-inspired raw coordinates are canonicalized by swap_xy.",
            "No GT-SR optimal persistence matching is performed in this phase.",
        ],
    }

    (out / "sample69_pair_provenance_audit.json").write_text(
        json.dumps(audit, indent=2)
    )

    # concise human summary
    print("===== PHASE 5A STEP 2 — SAMPLE-69 PAIR PROVENANCE =====")
    for s in summaries:
        print(
            f'{s["method"]:22s} {s["field_kind"]:2s} '
            f'cells={s["cells"]:4d} finite={s["finite_cells"]:4d} '
            f'nonfinite={s["nonfinite_cells"]:2d} '
            f'pair_types={s["pair_type_counts"]} '
            f'vid_xy_fail={len(s["vid_coordinate_failures"]):2d} '
            f'pd_geom_fail={len(s["pd_geometry_failures"]):2d} '
            f'transform={s["transform"]}'
        )

    print()
    print("GT common exact pair keys:", len(common_keys))
    print("GT canonical-coordinate mismatches on common keys:", len(mismatch))
    if mismatch:
        print("First 10 mismatches:")
        for x in mismatch[:10]:
            print(x)

    print()
    print("Wrote:")
    for p in sorted(out.iterdir()):
        if p.is_file():
            print(" ", p)


if __name__ == "__main__":
    main()
