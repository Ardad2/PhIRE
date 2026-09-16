#!/usr/bin/env python3
"""
Phase 3E step 2: rebuild colleague 6_anti trees using the exact scalar arrays
read from the VTI files that fed TTK, then compare vertices and hierarchy.

Comparison is performed directly in each VTI's raw VertexId coordinate system;
no historical transpose bridge is needed because TTK and the colleague builder
are now fed the same VTI-indexed scalar sequence.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, sys
from pathlib import Path
import numpy as np
import networkx as nx

def add_repo_src(repo): sys.path.insert(0,str(repo/"src"))

def adjacency(h,w):
    card=[(-1,0),(1,0),(0,-1),(0,1)]
    offs=card+[(-1,1),(1,-1)]  # 6_anti
    return {r*w+c:[(r+dr)*w+(c+dc) for dr,dc in offs
                    if 0<=r+dr<h and 0<=c+dc<w]
            for r in range(h) for c in range(w)}

def read_nodes(path):
    m={}
    with path.open(newline="") as f:
        for r in csv.DictReader(f):
            m[int(r["NodeId"])]={
              "v":int(r["VertexId"]),
              "s":float(r["Scalar"]),
              "ct":int(r["CriticalType"]),
            }
    return m

def read_edges(path,m):
    out=set()
    with path.open(newline="") as f:
        for r in csv.DictReader(f):
            u=int(r["upNodeId"]); d=int(r["downNodeId"])
            a=m[d]["v"]; b=m[u]["v"]
            if a!=b: out.add(tuple(sorted((a,b))))
    return out

def metrics(A,B):
    I=A&B; U=A|B
    return dict(
      intersection=len(I), union=len(U),
      jaccard=len(I)/len(U) if U else 1.0,
      a_recall=len(I)/len(A) if A else 1.0,
      b_precision=len(I)/len(B) if B else 1.0,
      a_only=len(A-I), b_only=len(B-I),
    )

def sha256(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--repo",required=True)
    ap.add_argument("--exact-scalars",required=True)
    ap.add_argument("--ttk-tree-extract",required=True)
    ap.add_argument("--out",required=True)
    args=ap.parse_args()

    repo=Path(args.repo).expanduser().resolve()
    scal=Path(args.exact_scalars).expanduser().resolve()
    tex=Path(args.ttk_tree_extract).expanduser().resolve()
    out=Path(args.out).expanduser().resolve(); out.mkdir(parents=True,exist_ok=True)

    add_repo_src(repo)
    from tda_toolkit.merge_tree import _build_join_tree_graph

    rows=[]
    diffs=[]
    for label in ("CNN_GT","CNN_SR","UV_GT","UV_SR","F1_GT","F1_SR"):
        x=np.load(scal/f"{label}_exact_vti_scalar.npy")
        x=np.asarray(x).reshape(-1)
        if x.size!=160*160: raise RuntimeError((label,x.shape))

        G=_build_join_tree_graph(x,adjacency(160,160))
        c_nodes={int(v) for v in G.nodes}
        c_edges={tuple(sorted((int(a),int(b)))) for a,b in G.edges if a!=b}

        tm=read_nodes(tex/f"{label}_nodes.csv")
        t_nodes={z["v"] for z in tm.values()}
        t_edges=read_edges(tex/f"{label}_edges.csv",tm)

        nm=metrics(t_nodes,c_nodes)

        common=t_nodes&c_nodes
        te={e for e in t_edges if e[0] in common and e[1] in common}
        ce={e for e in c_edges if e[0] in common and e[1] in common}
        em=metrics(te,ce)

        ttk_only=sorted(t_nodes-c_nodes)
        col_only=sorted(c_nodes-t_nodes)

        row={
          "label":label,
          "scalar_dtype":str(x.dtype),
          "ttk_nodes":len(t_nodes),
          "colleague_nodes":len(c_nodes),
          "common_nodes":len(common),
          "node_jaccard":nm["jaccard"],
          "ttk_node_recall":nm["a_recall"],
          "colleague_node_precision":nm["b_precision"],
          "ttk_only_nodes":nm["a_only"],
          "colleague_only_nodes":nm["b_only"],
          "common_ttk_edges":len(te),
          "common_colleague_edges":len(ce),
          "common_edge_jaccard":em["jaccard"],
          "common_ttk_edge_recall":em["a_recall"],
          "common_colleague_edge_precision":em["b_precision"],
        }
        rows.append(row)

        for v in ttk_only:
            diffs.append([label,"TTK_ONLY",v,float(x[v])])
        for v in col_only:
            diffs.append([label,"COLLEAGUE_ONLY",v,float(x[v])])

    with (out/"exact_vti_parity_summary.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    (out/"exact_vti_parity_summary.json").write_text(json.dumps(rows,indent=2))
    with (out/"exact_vti_parity_differences.csv").open("w",newline="") as f:
        w=csv.writer(f); w.writerow(["label","status","vertex_id","scalar"])
        w.writerows(diffs)

    print("===== PHASE 3E EXACT-TTK-INPUT PARITY =====")
    for r in rows:
        print(
          f'{r["label"]:7s} dtype={r["scalar_dtype"]:8s} '
          f'TTK={r["ttk_nodes"]:4d} COL={r["colleague_nodes"]:4d} '
          f'nodeJ={r["node_jaccard"]:.6f} '
          f'TTKonly={r["ttk_only_nodes"]:3d} COLonly={r["colleague_only_nodes"]:3d} '
          f'edgeJ={r["common_edge_jaccard"]:.6f} '
          f'edgeRecall={r["common_ttk_edge_recall"]:.6f} '
          f'edgePrecision={r["common_colleague_edge_precision"]:.6f}'
        )

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
