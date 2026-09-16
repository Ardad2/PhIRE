#!/usr/bin/env python3
"""
Phase 3A: candidate grid-triangulation audit for the colleague toolkit's
hierarchical Join-Tree builder.

Tests four explicit adjacency conventions on the exact sample-69 160x160
wind-speed crop:
  4-neighbor
  6-neighbor main-diagonal  (NW <-> SE)
  6-neighbor anti-diagonal  (NE <-> SW)
  8-neighbor

The two 6-neighbor cases are candidate triangulated-grid conventions.
This script does NOT assume either is TTK's actual implicit triangulation.

It compares structural counts with frozen audited TTK numerical-tree node
counts as a diagnostic only; matching node counts are not treated as proof
of semantic parity.
"""

from __future__ import annotations
import argparse, csv, hashlib, json, sys
from pathlib import Path
import numpy as np
import networkx as nx

TTK_NUMERICAL_NODE_COUNTS = {
    "GT_low": 2193,
    "GT_high": 2195,
    "CNN": 1144,
    "UV": 808,
    "F1": 1765,
}

PHASE2_EXPECTED = {
    ("GT","4"): 2503, ("GT","8"): 1701,
    ("CNN","4"): 1446, ("CNN","8"): 939,
    ("UV","4"): 937, ("UV","8"): 625,
    ("F1","4"): 2769, ("F1","8"): 1150,
}

def add_repo_src(repo):
    sys.path.insert(0, str(repo / "src"))

def adjacency(h, w, mode):
    cardinal = [(-1,0),(1,0),(0,-1),(0,1)]
    if mode == "4":
        offs = cardinal
    elif mode == "6_main":
        offs = cardinal + [(-1,-1),(1,1)]
    elif mode == "6_anti":
        offs = cardinal + [(-1,1),(1,-1)]
    elif mode == "8":
        offs = cardinal + [(-1,-1),(-1,1),(1,-1),(1,1)]
    else:
        raise ValueError(mode)
    return {
        r*w+c: [(r+dr)*w+(c+dc) for dr,dc in offs
                if 0 <= r+dr < h and 0 <= c+dc < w]
        for r in range(h) for c in range(w)
    }

def load_speed(root, filename, sample, crop):
    a=np.load(root/filename,mmap_mode="r")
    v=np.asarray(a[sample,:crop,:crop,:],dtype=np.float64)
    if v.shape != (crop,crop,2):
        raise ValueError((root,filename,v.shape))
    s=np.hypot(v[...,0],v[...,1])
    if not np.isfinite(s).all():
        raise ValueError("nonfinite field")
    return s

def n_components(G):
    if not G.number_of_nodes(): return 0
    return nx.number_weakly_connected_components(G) if G.is_directed() else nx.number_connected_components(G)

def target_range(label):
    if label=="GT":
        return TTK_NUMERICAL_NODE_COUNTS["GT_low"], TTK_NUMERICAL_NODE_COUNTS["GT_high"]
    x=TTK_NUMERICAL_NODE_COUNTS[label]
    return x,x

def range_error(n,lo,hi):
    if lo <= n <= hi: return 0
    return min(abs(n-lo),abs(n-hi))

def summarize(label,mode,G):
    UG=G.to_undirected() if G.is_directed() else G
    deg=dict(UG.degree())
    lo,hi=target_range(label)
    mid=(lo+hi)/2
    return {
        "label":label,
        "adjacency":mode,
        "nodes":int(G.number_of_nodes()),
        "edges":int(G.number_of_edges()),
        "components":int(n_components(G)),
        "tree_undirected":bool(nx.is_tree(UG)) if G.number_of_nodes() else False,
        "dag":bool(nx.is_directed_acyclic_graph(G)) if G.is_directed() else None,
        "dag_longest_path_edges":int(nx.dag_longest_path_length(G))
            if G.is_directed() and nx.is_directed_acyclic_graph(G) else None,
        "leaves_degree1":int(sum(d==1 for d in deg.values())),
        "degree2_nodes":int(sum(d==2 for d in deg.values())),
        "branch_nodes_degree_ge3":int(sum(d>=3 for d in deg.values())),
        "self_loops":int(nx.number_of_selfloops(UG)),
        "ttk_target_nodes_low":lo,
        "ttk_target_nodes_high":hi,
        "abs_node_count_error_to_range":int(range_error(G.number_of_nodes(),lo,hi)),
        "relative_node_count_error_to_midpoint_pct":
            float(100*(G.number_of_nodes()-mid)/mid),
    }

def sha256(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):
            h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--repo",required=True)
    ap.add_argument("--sample",type=int,default=69)
    ap.add_argument("--crop",type=int,default=160)
    ap.add_argument("--cnn-root",required=True)
    ap.add_argument("--uv-root",required=True)
    ap.add_argument("--f1-root",required=True)
    ap.add_argument("--out",required=True)
    args=ap.parse_args()

    repo=Path(args.repo).expanduser().resolve()
    cnn=Path(args.cnn_root).expanduser().resolve()
    uv=Path(args.uv_root).expanduser().resolve()
    f1=Path(args.f1_root).expanduser().resolve()
    out=Path(args.out).expanduser().resolve()
    out.mkdir(parents=True,exist_ok=True)

    add_repo_src(repo)
    from tda_toolkit.merge_tree import _build_join_tree_graph

    gt1=load_speed(cnn,"dataGT.npy",args.sample,args.crop)
    gt2=load_speed(uv ,"dataGT.npy",args.sample,args.crop)
    gt3=load_speed(f1 ,"dataGT.npy",args.sample,args.crop)
    if not (np.array_equal(gt1,gt2) and np.array_equal(gt1,gt3)):
        raise RuntimeError("GT identity check failed")

    fields={
        "GT":gt1,
        "CNN":load_speed(cnn,"dataSR.npy",args.sample,args.crop),
        "UV":load_speed(uv,"dataSR.npy",args.sample,args.crop),
        "F1":load_speed(f1,"dataSR.npy",args.sample,args.crop),
    }

    rows=[]
    for label,field in fields.items():
        flat=np.asarray(field,dtype=np.float64).ravel(order="C")
        for mode in ("4","6_main","6_anti","8"):
            G=_build_join_tree_graph(flat,adjacency(args.crop,args.crop,mode))
            row=summarize(label,mode,G)
            rows.append(row)
            if (label,mode) in PHASE2_EXPECTED:
                exp=PHASE2_EXPECTED[(label,mode)]
                if row["nodes"] != exp:
                    raise RuntimeError(
                        f"Phase-2 reproduction failed {label}/{mode}: "
                        f"{row['nodes']} != {exp}"
                    )

    with (out/"candidate_connectivity_summary.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    (out/"candidate_connectivity_summary.json").write_text(json.dumps(rows,indent=2))

    print("===== PHASE 3A CANDIDATE TRIANGULATION AUDIT =====")
    print("4/8-neighbor Phase-2 reproduction: PASS")
    print()
    for label in ("GT","CNN","UV","F1"):
        print(f"--- {label} ---")
        for r in rows:
            if r["label"]==label:
                target=(str(r["ttk_target_nodes_low"]) if r["ttk_target_nodes_low"]==r["ttk_target_nodes_high"]
                        else f'{r["ttk_target_nodes_low"]}-{r["ttk_target_nodes_high"]}')
                print(f'{r["adjacency"]:7s} nodes={r["nodes"]:5d} '
                      f'TTK={target:9s} err={r["abs_node_count_error_to_range"]:4d} '
                      f'rel={r["relative_node_count_error_to_midpoint_pct"]:+7.2f}% '
                      f'tree={r["tree_undirected"]} dag={r["dag"]}')
        print()

    print("COUNT-NEAREST ADJACENCY (diagnostic only; NOT semantic parity)")
    for label in ("GT","CNN","UV","F1"):
        rr=[r for r in rows if r["label"]==label]
        best=min(rr,key=lambda r:(r["abs_node_count_error_to_range"],r["adjacency"]))
        print(label,best["adjacency"],best["nodes"],
              "error",best["abs_node_count_error_to_range"])

    manifest=[]
    for p in sorted(out.iterdir()):
        if p.is_file() and p.name!="sha256_manifest.txt":
            manifest.append(f"{sha256(p)}  {p.name}")
    (out/"sha256_manifest.txt").write_text("\n".join(manifest)+"\n")
    print()
    print("===== SHA256 MANIFEST =====")
    print((out/"sha256_manifest.txt").read_text())

if __name__=="__main__":
    main()
