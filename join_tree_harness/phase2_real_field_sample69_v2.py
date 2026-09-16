#!/usr/bin/env python3
"""
Phase 2: real-field structural Join-Tree comparison using the colleague
toolkit's existing _build_join_tree_graph() on the exact 160x160 top-left
topology crop used by the authoritative PhIRE TTK/PD/MT analysis.

This script DOES NOT compute or claim TTK MergeTreeDistance parity.
It compares structural properties under explicit 4-neighbor and 8-neighbor
regular-grid adjacency.

Expected vector-array roots:
  CNN: data_out_fixed/wind_mrhr_cnn
  UV : data_out/wind_finetune_candidateUV_expanded2688
  F1 : data_out/wind_finetune_candidateF_grad_E2_low_expanded2688
"""

from __future__ import annotations
import argparse, csv, inspect, json, sys, hashlib
from pathlib import Path

import numpy as np
import networkx as nx


def add_repo_src(repo: Path) -> None:
    sys.path.insert(0, str(repo / "src"))


def grid_adjacency(h: int, w: int, connectivity: int):
    if connectivity == 4:
        offsets = [(-1,0),(1,0),(0,-1),(0,1)]
    elif connectivity == 8:
        offsets = [
            (-1,0),(1,0),(0,-1),(0,1),
            (-1,-1),(-1,1),(1,-1),(1,1)
        ]
    else:
        raise ValueError("connectivity must be 4 or 8")

    return {
        r*w+c: [
            (r+dr)*w+(c+dc)
            for dr,dc in offsets
            if 0 <= r+dr < h and 0 <= c+dc < w
        ]
        for r in range(h)
        for c in range(w)
    }


def load_speed_from_root(root: Path, which: str, sample: int, crop: int):
    p = root / which
    a = np.load(p, mmap_mode="r")
    if a.ndim != 4 or a.shape[-1] != 2:
        raise ValueError(f"Expected [N,H,W,2] in {p}, got {a.shape}")
    if sample < 0 or sample >= a.shape[0]:
        raise IndexError(f"sample {sample} out of range for {p}: N={a.shape[0]}")
    v = np.asarray(a[sample, :crop, :crop, :], dtype=np.float64)
    if v.shape != (crop, crop, 2):
        raise ValueError(f"crop mismatch for {p}: got {v.shape}")
    speed = np.hypot(v[...,0], v[...,1])
    if not np.isfinite(speed).all():
        raise ValueError(f"non-finite speed values in {p}")
    return speed


def components(G):
    if G.number_of_nodes() == 0:
        return 0
    return (nx.number_weakly_connected_components(G)
            if G.is_directed()
            else nx.number_connected_components(G))


def is_tree(G):
    if G.number_of_nodes() == 0:
        return False
    UG = G.to_undirected() if G.is_directed() else G
    return nx.is_tree(UG)


def scalar_for_node(G, n, flat):
    for k in ("value","scalar","f","height"):
        if k in G.nodes[n]:
            try:
                return float(G.nodes[n][k])
            except Exception:
                pass
    if isinstance(n, (int, np.integer)) and 0 <= int(n) < len(flat):
        return float(flat[int(n)])
    return float("nan")


def tree_depth_if_dag(G):
    if not G.is_directed() or G.number_of_nodes() == 0 or not nx.is_directed_acyclic_graph(G):
        return None
    roots = [n for n in G.nodes if G.in_degree(n) == 0]
    sinks = [n for n in G.nodes if G.out_degree(n) == 0]
    # Orientation is implementation-dependent. Report the maximum DAG path length
    # without assuming which end is the semantic root.
    try:
        return int(nx.dag_longest_path_length(G))
    except Exception:
        return None


def summarize(label, connectivity, G, flat):
    UG = G.to_undirected() if G.is_directed() else G
    deg = dict(UG.degree())
    vals = [scalar_for_node(G,n,flat) for n in G.nodes]
    vals = [v for v in vals if np.isfinite(v)]
    return dict(
        label=label,
        connectivity=connectivity,
        directed=bool(G.is_directed()),
        nodes=int(G.number_of_nodes()),
        edges=int(G.number_of_edges()),
        components=int(components(G)),
        tree_undirected=bool(is_tree(G)),
        dag=bool(nx.is_directed_acyclic_graph(G)) if G.is_directed() else None,
        dag_longest_path_edges=tree_depth_if_dag(G),
        leaves_degree1=int(sum(d == 1 for d in deg.values())),
        branch_nodes_degree_ge3=int(sum(d >= 3 for d in deg.values())),
        degree2_nodes=int(sum(d == 2 for d in deg.values())),
        self_loops=int(nx.number_of_selfloops(UG)),
        min_node_scalar=min(vals) if vals else None,
        max_node_scalar=max(vals) if vals else None,
        field_min=float(np.min(flat)),
        field_max=float(np.max(flat)),
        field_mean=float(np.mean(flat)),
    )


def sha256_file(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--sample", type=int, default=69)
    ap.add_argument("--crop", type=int, default=160)
    ap.add_argument("--cnn-root", required=True)
    ap.add_argument("--uv-root", required=True)
    ap.add_argument("--f1-root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    repo = Path(args.repo).expanduser().resolve()
    cnn = Path(args.cnn_root).expanduser().resolve()
    uv  = Path(args.uv_root).expanduser().resolve()
    f1  = Path(args.f1_root).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    add_repo_src(repo)
    from tda_toolkit.merge_tree import _build_join_tree_graph

    # One GT is enough only after exact identity is verified below.
    gt_cnn = load_speed_from_root(cnn, "dataGT.npy", args.sample, args.crop)
    gt_uv  = load_speed_from_root(uv,  "dataGT.npy", args.sample, args.crop)
    gt_f1  = load_speed_from_root(f1,  "dataGT.npy", args.sample, args.crop)

    gt_equal_cnn_uv = np.array_equal(gt_cnn, gt_uv)
    gt_equal_cnn_f1 = np.array_equal(gt_cnn, gt_f1)
    if not (gt_equal_cnn_uv and gt_equal_cnn_f1):
        raise RuntimeError("GT identity check failed across CNN / UV / F1 roots")

    fields = {
        "GT": gt_cnn,
        "CNN": load_speed_from_root(cnn, "dataSR.npy", args.sample, args.crop),
        "UV":  load_speed_from_root(uv,  "dataSR.npy", args.sample, args.crop),
        "F1":  load_speed_from_root(f1,  "dataSR.npy", args.sample, args.crop),
    }

    provenance = {
        "sample": args.sample,
        "crop": args.crop,
        "crop_rule": "top-left [:crop, :crop]",
        "scalar": "wind_speed = hypot(u,v)",
        "flattening": "C-order via ravel(order='C')",
        "gt_cnn_equals_gt_uv": bool(gt_equal_cnn_uv),
        "gt_cnn_equals_gt_f1": bool(gt_equal_cnn_f1),
        "roots": {
            "CNN": str(cnn),
            "UV": str(uv),
            "F1": str(f1),
        },
        "builder_signature": str(inspect.signature(_build_join_tree_graph)),
    }

    rows = []
    for label, field in fields.items():
        np.save(out / f"{label}_sample{args.sample:03d}_speed_160.npy", field)
        flat = np.asarray(field, dtype=np.float64).ravel(order="C")

        for conn in (4, 8):
            adj = grid_adjacency(args.crop, args.crop, conn)
            G = _build_join_tree_graph(flat, adj)
            row = summarize(label, conn, G, flat)
            rows.append(row)

            with (out / f"{label}_join{conn}_nodes.csv").open("w", newline="") as f:
                wr = csv.writer(f)
                wr.writerow(["node","scalar","degree","in_degree","out_degree"])
                UG = G.to_undirected() if G.is_directed() else G
                for n in G.nodes:
                    wr.writerow([
                        n,
                        scalar_for_node(G,n,flat),
                        UG.degree(n),
                        G.in_degree(n) if G.is_directed() else "",
                        G.out_degree(n) if G.is_directed() else "",
                    ])

            with (out / f"{label}_join{conn}_edges.csv").open("w", newline="") as f:
                wr = csv.writer(f)
                wr.writerow(["source","target","source_scalar","target_scalar"])
                for u,v in G.edges:
                    wr.writerow([
                        u, v,
                        scalar_for_node(G,u,flat),
                        scalar_for_node(G,v,flat),
                    ])

    with (out / "structural_summary.csv").open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    (out / "structural_summary.json").write_text(json.dumps(rows, indent=2))
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2))

    print("===== PHASE 2 SAMPLE-69 REAL-FIELD STRUCTURAL SUMMARY =====")
    print(json.dumps(provenance, indent=2))
    print()
    for r in rows:
        print(r)

    # Freeze all produced files except a manifest that may be written later.
    manifest = []
    for p in sorted(out.iterdir()):
        if p.is_file() and p.name != "sha256_manifest.txt":
            manifest.append(f"{sha256_file(p)}  {p.name}")
    (out / "sha256_manifest.txt").write_text("\n".join(manifest) + "\n")
    print()
    print("===== SHA256 MANIFEST =====")
    print((out / "sha256_manifest.txt").read_text())


if __name__ == "__main__":
    main()
