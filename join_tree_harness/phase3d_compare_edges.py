#!/usr/bin/env python3
"""
Phase 3D step 2: compare TTK Join-Tree arcs with colleague hierarchical-builder
edges for sample 69.

Primary adjacency: 6_anti
Control adjacency: 6_main

Comparison is done in authoritative C-order VertexId space.
CNN/UV TTK vertices use the already-audited legacy transpose bridge.
F1 uses identity.

Primary metrics:
  raw undirected edge overlap
  overlap after restricting TTK to the shared vertex set
  scalar-monotone directed edge overlap
  degree/root diagnostics

For SR fields, if the only TTK-only node is the global maximum/root and has
degree one, removing that root should allow exact hierarchy parity if the
remaining trees are truly the same.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, sys
from pathlib import Path
import numpy as np
import networkx as nx

EXPECTED_COUNTS={
    ("GT","6_main"):1982, ("GT","6_anti"):2189,
    ("CNN","6_main"):1229, ("CNN","6_anti"):1143,
    ("UV","6_main"):755, ("UV","6_anti"):807,
    ("F1","6_main"):1791, ("F1","6_anti"):1764,
}

def add_repo_src(repo): sys.path.insert(0,str(repo/"src"))

def adjacency(h,w,mode):
    card=[(-1,0),(1,0),(0,-1),(0,1)]
    if mode=="6_main": offs=card+[(-1,-1),(1,1)]
    elif mode=="6_anti": offs=card+[(-1,1),(1,-1)]
    else: raise ValueError(mode)
    return {r*w+c:[(r+dr)*w+(c+dc) for dr,dc in offs
                    if 0<=r+dr<h and 0<=c+dc<w]
            for r in range(h) for c in range(w)}

def load_speed(root,fn,sample,crop):
    a=np.load(root/fn,mmap_mode="r")
    v=np.asarray(a[sample,:crop,:crop,:],dtype=np.float64)
    return np.hypot(v[...,0],v[...,1])

def orient_vid(v,n,orientation):
    if orientation=="identity": return v
    if orientation=="transpose":
        x=v%n; y=v//n
        return y+n*x
    raise ValueError(orientation)

def read_nodes(path, n, orientation):
    node_to_vid={}
    node_scalar={}
    ctype={}
    with path.open(newline="") as f:
        for r in csv.DictReader(f):
            nid=int(r["NodeId"]); raw=int(r["VertexId"])
            vid=orient_vid(raw,n,orientation)
            node_to_vid[nid]=vid
            node_scalar[nid]=float(r["Scalar"])
            ctype[nid]=int(r["CriticalType"])
    return node_to_vid,node_scalar,ctype

def read_edges(path,node_to_vid):
    directed=[]
    with path.open(newline="") as f:
        for r in csv.DictReader(f):
            up=int(r["upNodeId"]); down=int(r["downNodeId"])
            directed.append((node_to_vid[down],node_to_vid[up])) # low/down -> high/up convention
    return directed

def undirected(edges):
    return {tuple(sorted((int(a),int(b)))) for a,b in edges if a!=b}

def scalar_direct(edges, flat):
    out=set()
    ties=0
    for a,b in edges:
        fa=float(flat[a]); fb=float(flat[b])
        if fa<fb: out.add((a,b))
        elif fb<fa: out.add((b,a))
        else:
            ties+=1
            out.add((min(a,b),max(a,b)))
    return out,ties

def metrics(A,B):
    I=A&B; U=A|B
    return {
      "a":len(A),"b":len(B),"intersection":len(I),"union":len(U),
      "jaccard":len(I)/len(U) if U else 1.0,
      "a_recall":len(I)/len(A) if A else 1.0,
      "b_precision":len(I)/len(B) if B else 1.0,
      "a_only":len(A-I),"b_only":len(B-I),
    }

def sha256(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--repo",required=True)
    ap.add_argument("--sample",type=int,default=69)
    ap.add_argument("--crop",type=int,default=160)
    ap.add_argument("--cnn-root",required=True)
    ap.add_argument("--uv-root",required=True)
    ap.add_argument("--f1-root",required=True)
    ap.add_argument("--ttk-tree-extract",required=True)
    ap.add_argument("--out",required=True)
    args=ap.parse_args()

    repo=Path(args.repo).expanduser().resolve()
    cnn=Path(args.cnn_root).expanduser().resolve()
    uv=Path(args.uv_root).expanduser().resolve()
    f1=Path(args.f1_root).expanduser().resolve()
    tex=Path(args.ttk_tree_extract).expanduser().resolve()
    out=Path(args.out).expanduser().resolve()
    out.mkdir(parents=True,exist_ok=True)

    add_repo_src(repo)
    from tda_toolkit.merge_tree import _build_join_tree_graph

    gt=load_speed(cnn,"dataGT.npy",args.sample,args.crop)
    gt2=load_speed(uv,"dataGT.npy",args.sample,args.crop)
    gt3=load_speed(f1,"dataGT.npy",args.sample,args.crop)
    if not (np.array_equal(gt,gt2) and np.array_equal(gt,gt3)):
        raise RuntimeError("GT identity failed")

    fields={
      "GT":gt,
      "CNN":load_speed(cnn,"dataSR.npy",args.sample,args.crop),
      "UV":load_speed(uv,"dataSR.npy",args.sample,args.crop),
      "F1":load_speed(f1,"dataSR.npy",args.sample,args.crop),
    }

    specs=[
      ("CNN_GT","GT","transpose"),
      ("UV_GT","GT","transpose"),
      ("F1_GT","GT","identity"),
      ("CNN_SR","CNN","transpose"),
      ("UV_SR","UV","transpose"),
      ("F1_SR","F1","identity"),
    ]

    # Build colleague graphs once.
    cgraphs={}
    for flabel,field in fields.items():
        flat=field.ravel(order="C")
        for mode in ("6_main","6_anti"):
            G=_build_join_tree_graph(flat,adjacency(args.crop,args.crop,mode))
            if G.number_of_nodes()!=EXPECTED_COUNTS[(flabel,mode)]:
                raise RuntimeError(f"{flabel}/{mode}: frozen node count mismatch")
            cgraphs[(flabel,mode)]=G

    summary=[]
    diff_rows=[]

    for ttk_label,flabel,orientation in specs:
        flat=fields[flabel].ravel(order="C")
        node_to_vid,node_scalar,ctype=read_nodes(tex/f"{ttk_label}_nodes.csv",args.crop,orientation)
        ttk_dir_raw=read_edges(tex/f"{ttk_label}_edges.csv",node_to_vid)
        ttk_u=undirected(ttk_dir_raw)
        ttk_nodes=set(node_to_vid.values())

        # TTK topology sanity.
        T=nx.Graph(); T.add_nodes_from(ttk_nodes); T.add_edges_from(ttk_u)
        if not nx.is_tree(T):
            raise RuntimeError(f"{ttk_label}: extracted TTK graph not a tree")

        for mode in ("6_main","6_anti"):
            G=cgraphs[(flabel,mode)]
            c_nodes={int(v) for v in G.nodes}
            c_dir_raw=[(int(a),int(b)) for a,b in G.edges]
            c_u=undirected(c_dir_raw)

            # Raw whole-tree edge overlap.
            raw=metrics(ttk_u,c_u)

            # Common-vertex induced edge comparison.
            common=ttk_nodes & c_nodes
            ttk_common={e for e in ttk_u if e[0] in common and e[1] in common}
            c_common={e for e in c_u if e[0] in common and e[1] in common}
            induced=metrics(ttk_common,c_common)

            # Canonical scalar-monotone direction over the shared endpoints.
            ttk_scalar_dir,tties=scalar_direct(ttk_common,flat)
            c_scalar_dir,cties=scalar_direct(c_common,flat)
            directed=metrics(ttk_scalar_dir,c_scalar_dir)

            # TTK-only node degrees and role diagnostics.
            ttk_only=ttk_nodes-c_nodes
            root_info=[]
            for v in sorted(ttk_only):
                root_info.append({
                  "vertex_id":v,
                  "degree":int(T.degree(v)),
                  "scalar":float(flat[v]),
                  "is_argmin":bool(v==int(np.argmin(flat))),
                  "is_argmax":bool(v==int(np.argmax(flat))),
                })

            row={
              "ttk_label":ttk_label,
              "field":flabel,
              "orientation":orientation,
              "adjacency":mode,
              "ttk_nodes":len(ttk_nodes),
              "colleague_nodes":len(c_nodes),
              "common_nodes":len(common),
              "ttk_only_nodes":len(ttk_nodes-c_nodes),
              "colleague_only_nodes":len(c_nodes-ttk_nodes),

              "raw_ttk_edges":raw["a"],
              "raw_colleague_edges":raw["b"],
              "raw_edge_intersection":raw["intersection"],
              "raw_edge_jaccard":raw["jaccard"],
              "raw_ttk_edge_recall":raw["a_recall"],
              "raw_colleague_edge_precision":raw["b_precision"],

              "common_ttk_edges":induced["a"],
              "common_colleague_edges":induced["b"],
              "common_edge_intersection":induced["intersection"],
              "common_edge_jaccard":induced["jaccard"],
              "common_ttk_edge_recall":induced["a_recall"],
              "common_colleague_edge_precision":induced["b_precision"],
              "common_ttk_only_edges":induced["a_only"],
              "common_colleague_only_edges":induced["b_only"],

              "scalar_directed_edge_jaccard":directed["jaccard"],
              "scalar_directed_ttk_recall":directed["a_recall"],
              "scalar_directed_colleague_precision":directed["b_precision"],
              "scalar_tie_edges_ttk":tties,
              "scalar_tie_edges_colleague":cties,

              "ttk_only_node_info":json.dumps(root_info,sort_keys=True),
            }
            summary.append(row)

            for e in sorted(ttk_common-c_common):
                diff_rows.append([ttk_label,mode,"TTK_ONLY_EDGE",e[0],e[1],
                                  float(flat[e[0]]),float(flat[e[1]])])
            for e in sorted(c_common-ttk_common):
                diff_rows.append([ttk_label,mode,"COLLEAGUE_ONLY_EDGE",e[0],e[1],
                                  float(flat[e[0]]),float(flat[e[1]])])

    with (out/"edge_hierarchy_summary.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0].keys()))
        w.writeheader(); w.writerows(summary)
    (out/"edge_hierarchy_summary.json").write_text(json.dumps(summary,indent=2))

    with (out/"edge_hierarchy_differences.csv").open("w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["ttk_label","adjacency","status","vertex_a","vertex_b","scalar_a","scalar_b"])
        w.writerows(diff_rows)

    print("===== PHASE 3D EDGE / HIERARCHY OVERLAP =====")
    for r in summary:
        print(
          f'{r["ttk_label"]:7s} {r["adjacency"]:7s} '
          f'nodes_common={r["common_nodes"]:4d} '
          f'RAW_J={r["raw_edge_jaccard"]:.6f} '
          f'COMMON_J={r["common_edge_jaccard"]:.6f} '
          f'COMMON_recall={r["common_ttk_edge_recall"]:.6f} '
          f'COMMON_precision={r["common_colleague_edge_precision"]:.6f} '
          f'DIR_J={r["scalar_directed_edge_jaccard"]:.6f} '
          f'TTKonlyN={r["ttk_only_nodes"]:3d} COLonlyN={r["colleague_only_nodes"]:3d}'
        )
        if r["adjacency"]=="6_anti":
            print("   TTK-only node info:",r["ttk_only_node_info"])

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
