#!/usr/bin/env python3
"""Compare the TDA toolkit's 2D join tree with the stored TTK join trees (ttkMergeTreeCmd).

For every sample / method / field it reads the authoritative TTK outputs
  *_mt_port_0.vtu  critical nodes   (NodeId, VertexId, Scalar)
  *_mt_port_1.vtu  tree arcs        (upNodeId, downNodeId)
  *_mt_port_2.vti  the exact float32 wind_speed input TTK used
rebuilds the tree with tda_toolkit.merge_tree.get_merge_tree_graph_2d on that same
float32 field, and compares vertex IDs, parent->child edges and the root exactly.
Both sides index vertices by VTK point ID of the same VTI (x fastest), so no
transpose bridge is needed.

Run on Spark (see the usage notes sent with this script):
  micromamba run -n tda-toolkit-check python compare_toolkit_vs_ttk.py \
      --toolkit-src ~/tda-toolkit-main/src --out ~/toolkit_ttk_check
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOTS = {
    "CNN": "ttk_runs_fixed/cnn/mt",
    "UV": "ttk_runs_fixed/topology_finetuning/candidateUV_expanded2688_topology/mt",
    "F1": "ttk_runs_fixed/topology_finetuning/candidateF_grad_E2_low_expanded2688_topology/mt",
}
SCALAR_NAME = "wind_speed"


def find_one(root: Path, kind: str, sample: int, port: int, ext: str) -> Path:
    pattern = f"*_{kind}_s{sample}_speed_p160_x0_y0_mt_port_{port}.{ext}"
    hits = sorted(p for p in root.rglob(pattern) if "superlevel" not in str(p))
    if len(hits) != 1:
        raise FileNotFoundError(f"{len(hits)} files match {root}/**/{pattern}")
    return hits[0]


def read_ttk_tree(root: Path, kind: str, sample: int):
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy

    def read(path, reader_cls):
        r = reader_cls()
        r.SetFileName(str(path))
        r.Update()
        return r.GetOutput()

    def arr(data, name):
        a = data.GetArray(name)
        if a is None:
            names = [data.GetArrayName(i) for i in range(data.GetNumberOfArrays())]
            raise KeyError(f"array {name!r} not found; available: {names}")
        return vtk_to_numpy(a)

    p0 = read(find_one(root, kind, sample, 0, "vtu"), vtk.vtkXMLUnstructuredGridReader)
    p1 = read(find_one(root, kind, sample, 1, "vtu"), vtk.vtkXMLUnstructuredGridReader)
    p2 = read(find_one(root, kind, sample, 2, "vti"), vtk.vtkXMLImageDataReader)

    node_id = arr(p0.GetPointData(), "NodeId").astype(np.int64)
    vertex_id = arr(p0.GetPointData(), "VertexId").astype(np.int64)
    vid = dict(zip(node_id.tolist(), vertex_id.tolist()))
    up = arr(p1.GetCellData(), "upNodeId").astype(np.int64)
    down = arr(p1.GetCellData(), "downNodeId").astype(np.int64)

    nx_, ny_, nz_ = p2.GetDimensions()
    if nz_ != 1:
        raise ValueError(f"expected a 2D image, got dimensions {(nx_, ny_, nz_)}")
    flat = arr(p2.GetPointData(), SCALAR_NAME)
    field = np.asarray(flat, dtype=np.float64).reshape(ny_, nx_)  # VTK point order: x fastest

    nodes = set(vertex_id.tolist())
    edges = {(vid[int(u)], vid[int(d)]) for u, d in zip(up, down)}
    values = field.ravel()
    flipped = sum(values[a] < values[b] for a, b in edges)  # orientation sanity check
    if flipped > len(edges) / 2:
        edges = {(b, a) for a, b in edges}
    return field, nodes, edges, str(flat.dtype), int(flipped)


def compare(tree_nodes, tree_edges, ttk_nodes, ttk_edges):
    tk_only = tree_nodes - ttk_nodes
    ttk_only = ttk_nodes - tree_nodes
    inter, union = tree_edges & ttk_edges, tree_edges | ttk_edges
    return {
        "exact": int(not tk_only and not ttk_only and tree_edges == ttk_edges),
        "toolkit_nodes": len(tree_nodes),
        "ttk_nodes": len(ttk_nodes),
        "toolkit_only_nodes": len(tk_only),
        "ttk_only_nodes": len(ttk_only),
        "edge_jaccard": round(len(inter) / len(union), 6) if union else 1.0,
        "toolkit_only_examples": sorted(tk_only)[:5],
        "ttk_only_examples": sorted(ttk_only)[:5],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phire", default="~/PhIRE")
    ap.add_argument("--toolkit-src", required=True, help="path to tda-toolkit-main/src")
    ap.add_argument("--samples", type=int, nargs="*", default=list(range(168)))
    ap.add_argument("--methods", nargs="*", default=list(ROOTS))
    ap.add_argument("--control-8", action="store_true",
                    help="also test the toolkit's default 8-neighbour tree (expected NOT to match TTK)")
    ap.add_argument("--out", default="~/toolkit_ttk_check")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(args.toolkit_src).expanduser()))
    from tda_toolkit.merge_tree import get_merge_tree_graph_2d

    phire = Path(args.phire).expanduser()
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    rows, t0 = [], time.time()
    for method in args.methods:
        root = phire / ROOTS[method]
        for sample in args.samples:
            for kind in ("GT", "SR"):
                row = {"method": method, "field": kind, "sample": sample}
                try:
                    field, ttk_nodes, ttk_edges, dtype, flipped = read_ttk_tree(root, kind, sample)
                    tree = get_merge_tree_graph_2d(field, 1, connectivity="triangulated")[0]
                    row.update(compare(set(tree.nodes), set(tree.edges), ttk_nodes, ttk_edges))
                    row.update(scalar_dtype=dtype, ttk_arcs_upside_down=flipped)
                    if args.control_8:
                        t8 = get_merge_tree_graph_2d(field, 1, connectivity=8)[0]
                        row["default8_exact"] = compare(set(t8.nodes), set(t8.edges), ttk_nodes, ttk_edges)["exact"]
                except Exception as exc:  # keep going; report the failure
                    row.update(exact="", error=repr(exc))
                rows.append(row)
            if sample % 20 == 0:
                done = sum(r.get("exact") == 1 for r in rows)
                print(f"{method} s{sample:03d}  exact so far {done}/{len(rows)}  ({time.time() - t0:.0f}s)", flush=True)

    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("method", "field", "sample", "exact"), k))
    with open(out / "toolkit_vs_ttk_per_tree.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    ok = [r for r in rows if r.get("exact") in (0, 1)]
    summary = {
        "trees_compared": len(ok),
        "errors": len(rows) - len(ok),
        "exact_matches": sum(r["exact"] for r in ok),
        "by_method_field": {
            f"{m} {k}": f"{sum(r['exact'] for r in ok if r['method'] == m and r['field'] == k)}"
                        f"/{sum(1 for r in ok if r['method'] == m and r['field'] == k)}"
            for m in args.methods for k in ("GT", "SR")
        },
        "mismatches": [{key: r.get(key) for key in ("method", "field", "sample", "toolkit_only_nodes",
                                                     "ttk_only_nodes", "edge_jaccard",
                                                     "toolkit_only_examples", "ttk_only_examples")}
                       for r in ok if not r["exact"]],
        "first_errors": [{key: r.get(key) for key in ("method", "field", "sample", "error")}
                         for r in rows if r.get("error")][:10],
    }
    if args.control_8:
        summary["default8_exact_matches"] = sum(r.get("default8_exact", 0) for r in ok)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
