#!/usr/bin/env python3
"""
Preflight bridge from ORIGINAL numerical sample-69 merge trees (threshold 0,
legacy provenance/orientation) to the regenerated authoritative threshold-3
logical trees.

The script tests, rather than assumes, whether VertexId correspondence is best
explained by:
    1) identity mapping, or
    2) the legacy square-grid transpose induced by F-order flattening.

It also reports how many exact converted TTK matching endpoint pairs survive
onto the threshold-3 display trees under each orientation.

No figure is produced and no mapping is frozen automatically.
"""

from __future__ import annotations

import csv
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import vtk

W = 160
H = 160


@dataclass(frozen=True)
class NumericTree:
    name: str
    display_method: str
    path: Path


def read_csv(path: Path) -> List[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def infer_column(fieldnames: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    lut = {norm(x): x for x in fieldnames}
    for c in candidates:
        if norm(c) in lut:
            return lut[norm(c)]
    return None


def canonical_method(value: str) -> Optional[str]:
    n = norm(value)

    exact = {
        "gt": "gt",
        "groundtruth": "gt",
        "truth": "gt",
        "cnn": "cnn",
        "uv": "uv",
        "f1": "f1",
    }
    if n in exact:
        return exact[n]

    if "groundtruth" in n or n.startswith("gt"):
        return "gt"
    if "cnn" in n:
        return "cnn"
    if "candidatef" in n or "f1" in n:
        return "f1"
    if "ablation" in n or "luv" in n or "candidateuv" in n:
        return "uv"

    return None


def read_vtu(path: Path) -> vtk.vtkUnstructuredGrid:
    if not path.is_file():
        raise FileNotFoundError(path)

    r = vtk.vtkXMLUnstructuredGridReader()
    r.SetFileName(str(path))
    r.Update()

    g = vtk.vtkUnstructuredGrid()
    g.ShallowCopy(r.GetOutput())

    if g.GetNumberOfPoints() == 0:
        raise RuntimeError(f"{path}: zero points")

    return g


def point_array(grid: vtk.vtkUnstructuredGrid, name: str):
    a = grid.GetPointData().GetArray(name)
    if a is None:
        names = []
        for i in range(grid.GetPointData().GetNumberOfArrays()):
            x = grid.GetPointData().GetArray(i)
            if x is not None and x.GetName():
                names.append(x.GetName())
        raise RuntimeError(f"Missing {name!r}; arrays={names}")
    return a


def int_array_values(grid: vtk.vtkUnstructuredGrid, name: str) -> List[int]:
    a = point_array(grid, name)
    out = []
    for i in range(grid.GetNumberOfPoints()):
        v = float(a.GetTuple1(i))
        iv = int(round(v))
        if abs(v - iv) > 1e-8:
            raise RuntimeError(f"{name}[{i}]={v} is not integer-like")
        out.append(iv)
    return out


def transpose_vid(v: int) -> int:
    if not (0 <= v < W * H):
        raise ValueError(f"VertexId outside {W}x{H} grid: {v}")
    old_x = v % W
    old_y = v // W
    new_x = old_y
    new_y = old_x
    return new_x + W * new_y


def map_vid(v: int, mode: str) -> int:
    if mode == "identity":
        return v
    if mode == "transpose":
        return transpose_vid(v)
    raise ValueError(mode)


def pct(a: int, b: int) -> float:
    return 100.0 * a / b if b else float("nan")


def main() -> int:
    root = Path.home() / "PhIRE"
    audit = Path.home() / "phire_runtime_audit_20260809_221548"
    w22 = Path(os.environ.get("W22", audit / "recompute_pd_w22"))
    base = (
        w22
        / "corrected_pd_mt"
        / "discordance_visuals"
        / "sample_069"
    )
    match_dir = base / "ttk_matching_host96"

    logical_path = base / "sample69_mt_logical_nodes.csv"
    logical = read_csv(logical_path)
    if not logical:
        raise RuntimeError(f"No rows in {logical_path}")

    fields = list(logical[0].keys())
    method_col = infer_column(
        fields,
        (
            "method",
            "method_id",
            "tree",
            "tree_name",
            "label",
            "source",
        ),
    )
    vertex_col = infer_column(
        fields,
        (
            "VertexId",
            "vertex_id",
            "vertexId",
            "vertexid",
        ),
    )
    node_col = infer_column(
        fields,
        (
            "NodeId",
            "node_id",
            "nodeId",
            "nodeid",
        ),
    )

    print("SAMPLE-69 NUMERICAL -> THRESHOLD-3 VERTEX BRIDGE PREFLIGHT")
    print("=" * 100)
    print("python:", sys.executable)
    print("VTK:", vtk.vtkVersion.GetVTKVersion())
    print("logical nodes:", logical_path)
    print("logical fields:", fields)
    print("inferred method column:", method_col)
    print("inferred VertexId column:", vertex_col)
    print("inferred NodeId column:", node_col)
    print()

    if method_col is None or vertex_col is None:
        raise RuntimeError(
            "Could not infer method/VertexId columns. "
            "Paste the printed logical fields."
        )

    t3: Dict[str, Set[int]] = defaultdict(set)
    unknown_methods = Counter()

    for r in logical:
        m = canonical_method(r[method_col])
        if m is None:
            unknown_methods[r[method_col]] += 1
            continue
        t3[m].add(int(round(float(r[vertex_col]))))

    print("Threshold-3 logical-node counts by canonical method:")
    for m in ("gt", "cnn", "uv", "f1"):
        print(f"  {m:>3s}: {len(t3[m])}")
    if unknown_methods:
        print("Unrecognized method labels:", dict(unknown_methods))
    print()

    expected_t3 = {"gt": 58, "cnn": 12, "uv": 18, "f1": 38}
    for m, expected in expected_t3.items():
        if len(t3[m]) != expected:
            print(
                f"WARNING: {m} threshold-3 count is {len(t3[m])}, "
                f"expected {expected} from the validated hierarchy."
            )
    print()

    numeric = [
        NumericTree(
            "cnn_GT",
            "gt",
            root
            / "ttk_runs_fixed/cnn/mt/"
            "cnn_GT_s69_speed_p160_x0_y0_mt_port_0.vtu",
        ),
        NumericTree(
            "cnn_SR",
            "cnn",
            root
            / "ttk_runs_fixed/cnn/mt/"
            "cnn_SR_s69_speed_p160_x0_y0_mt_port_0.vtu",
        ),
        NumericTree(
            "uv_GT",
            "gt",
            root
            / "ttk_runs_fixed/topology_finetuning/"
            "candidateUV_expanded2688_topology/mt/GT/"
            "candidateUV_expanded2688_GT_s69_speed_p160_x0_y0_mt_port_0.vtu",
        ),
        NumericTree(
            "uv_SR",
            "uv",
            root
            / "ttk_runs_fixed/topology_finetuning/"
            "candidateUV_expanded2688_topology/mt/SR/"
            "candidateUV_expanded2688_SR_s69_speed_p160_x0_y0_mt_port_0.vtu",
        ),
        NumericTree(
            "f1_GT",
            "gt",
            root
            / "ttk_runs_fixed/topology_finetuning/"
            "candidateF_grad_E2_low_expanded2688_topology/mt/GT/"
            "candidateF_grad_E2_low_expanded2688_GT_s69_speed_p160_x0_y0_mt_port_0.vtu",
        ),
        NumericTree(
            "f1_SR",
            "f1",
            root
            / "ttk_runs_fixed/topology_finetuning/"
            "candidateF_grad_E2_low_expanded2688_topology/mt/SR/"
            "candidateF_grad_E2_low_expanded2688_SR_s69_speed_p160_x0_y0_mt_port_0.vtu",
        ),
    ]

    numeric_vids: Dict[str, Set[int]] = {}

    print("A. THRESHOLD-3 NODE COVERAGE BY ORIGINAL NUMERICAL VertexId")
    print("-" * 100)
    print(
        "tree      t3_method    numeric_nodes    t3_nodes    "
        "identity_overlap    transpose_overlap"
    )

    coverage_rows = []

    for spec in numeric:
        g = read_vtu(spec.path)
        vids = set(int_array_values(g, "VertexId"))
        numeric_vids[spec.name] = vids

        target = t3[spec.display_method]
        identity = target & vids
        trans_vids = {transpose_vid(v) for v in vids}
        transpose = target & trans_vids

        print(
            f"{spec.name:<9s} {spec.display_method:<11s} "
            f"{len(vids):>13d} {len(target):>11d} "
            f"{len(identity):>8d}/{len(target):<3d} "
            f"({pct(len(identity), len(target)):6.2f}%)    "
            f"{len(transpose):>8d}/{len(target):<3d} "
            f"({pct(len(transpose), len(target)):6.2f}%)"
        )

        coverage_rows.append(
            {
                "numeric_tree": spec.name,
                "t3_method": spec.display_method,
                "numeric_unique_vertexids": len(vids),
                "t3_nodes": len(target),
                "identity_overlap": len(identity),
                "identity_percent": pct(len(identity), len(target)),
                "transpose_overlap": len(transpose),
                "transpose_percent": pct(len(transpose), len(target)),
            }
        )

    print()
    print("B. CONVERTED TTK MATCHING PAIRS THAT SURVIVE ON BOTH THRESHOLD-3 TREES")
    print("-" * 100)
    print(
        "pair      converted_pairs    identity_both_t3    transpose_both_t3    "
        "identity_unique_pairs    transpose_unique_pairs"
    )

    pair_specs = {
        "cnn": ("gt", "cnn"),
        "uv": ("gt", "uv"),
        "f1": ("gt", "f1"),
    }

    survival_rows = []

    for label, (m1, m2) in pair_specs.items():
        p = match_dir / f"{label}_converted_matching_enriched.csv"
        rows = read_csv(p)

        required = ("gt_VertexId", "sr_VertexId")
        if not all(c in rows[0] for c in required):
            raise RuntimeError(
                f"{p}: expected columns {required}; have {list(rows[0])}"
            )

        stats = {}

        for mode in ("identity", "transpose"):
            pairs = []
            both = 0

            for r in rows:
                v1 = int(round(float(r["gt_VertexId"])))
                v2 = int(round(float(r["sr_VertexId"])))

                mv1 = map_vid(v1, mode)
                mv2 = map_vid(v2, mode)

                if mv1 in t3[m1] and mv2 in t3[m2]:
                    both += 1
                    pairs.append((mv1, mv2))

            stats[mode] = (both, len(set(pairs)))

        print(
            f"{label:<8s} {len(rows):>15d} "
            f"{stats['identity'][0]:>18d} "
            f"{stats['transpose'][0]:>20d} "
            f"{stats['identity'][1]:>22d} "
            f"{stats['transpose'][1]:>23d}"
        )

        survival_rows.append(
            {
                "pair": label,
                "converted_pairs": len(rows),
                "identity_both_t3": stats["identity"][0],
                "identity_unique_pairs": stats["identity"][1],
                "transpose_both_t3": stats["transpose"][0],
                "transpose_unique_pairs": stats["transpose"][1],
            }
        )

    cov_csv = match_dir / "sample69_numerical_to_t3_vertex_coverage.csv"
    with cov_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(coverage_rows[0].keys()))
        w.writeheader()
        w.writerows(coverage_rows)

    surv_csv = match_dir / "sample69_converted_matching_t3_survival.csv"
    with surv_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(survival_rows[0].keys()))
        w.writeheader()
        w.writerows(survival_rows)

    print()
    print("OUTPUTS")
    print("-" * 100)
    print("coverage:", cov_csv)
    print("matching survival:", surv_csv)
    print()
    print(
        "Do not freeze an orientation from this script automatically. "
        "Inspect whether identity or transpose gives dominant/exact coverage "
        "across all six numerical-tree variants."
    )
    print()
    print("NUMERICAL -> THRESHOLD-3 VERTEX BRIDGE PREFLIGHT: COMPLETE")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
