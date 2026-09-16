#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, inspect, json, sys
from pathlib import Path
import numpy as np
import networkx as nx

def add_repo_src(repo): sys.path.insert(0,str(repo/"src"))

def grid_adjacency(h,w,connectivity=4):
    offs=[(-1,0),(1,0),(0,-1),(0,1)]
    if connectivity==8: offs += [(-1,-1),(-1,1),(1,-1),(1,1)]
    elif connectivity!=4: raise ValueError("connectivity must be 4 or 8")
    return {r*w+c:[(r+dr)*w+(c+dc) for dr,dc in offs if 0<=r+dr<h and 0<=c+dc<w]
            for r in range(h) for c in range(w)}

def to_scalar(arr,sample):
    if arr.ndim==4 and arr.shape[-1]==2:
        v=np.asarray(arr[sample]); return np.sqrt(v[...,0]**2+v[...,1]**2)
    if arr.ndim==3 and arr.shape[-1]==2:
        return np.sqrt(arr[...,0]**2+arr[...,1]**2)
    if arr.ndim==3: return np.asarray(arr[sample],dtype=float)
    if arr.ndim==2: return np.asarray(arr,dtype=float)
    raise ValueError(f"Unsupported shape {arr.shape}")

def comps(G):
    return nx.number_weakly_connected_components(G) if G.is_directed() else nx.number_connected_components(G)

def tree(G):
    return nx.is_tree(G.to_undirected() if G.is_directed() else G) if G.number_of_nodes() else False

def sval(G,n,flat):
    for k in ("value","scalar","f","height"):
        if k in G.nodes[n]:
            try:return float(G.nodes[n][k])
            except:pass
    if isinstance(n,(int,np.integer)) and 0<=int(n)<len(flat): return float(flat[int(n)])
    return float("nan")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--repo",required=True)
    ap.add_argument("--sample",type=int,default=69)
    ap.add_argument("--input",action="append",required=True,help="LABEL=/path/to/array.npy")
    ap.add_argument("--out",required=True)
    args=ap.parse_args()
    repo=Path(args.repo).expanduser().resolve()
    out=Path(args.out).expanduser().resolve(); out.mkdir(parents=True,exist_ok=True)
    add_repo_src(repo)
    from tda_toolkit.merge_tree import _build_join_tree_graph

    rows=[]
    for spec in args.input:
        label,pth=spec.split("=",1)
        arr=np.load(Path(pth).expanduser(),mmap_mode="r")
        field=to_scalar(arr,args.sample)
        flat=field.ravel()
        work=flat + 1e-12*np.arange(flat.size,dtype=float)
        np.save(out/f"{label}_speed.npy",field)
        h,w=field.shape
        for conn in (4,8):
            G=_build_join_tree_graph(work,grid_adjacency(h,w,conn))
            UG=G.to_undirected() if G.is_directed() else G
            deg=dict(UG.degree())
            vals=[sval(G,n,work) for n in G.nodes]
            vals=[v for v in vals if np.isfinite(v)]
            row=dict(label=label,connectivity=conn,nodes=G.number_of_nodes(),
                     edges=G.number_of_edges(),components=comps(G),tree=tree(G),
                     leaves_degree1=sum(d==1 for d in deg.values()),
                     branch_nodes_degree_ge3=sum(d>=3 for d in deg.values()),
                     self_loops=nx.number_of_selfloops(UG),
                     min_node_scalar=min(vals) if vals else None,
                     max_node_scalar=max(vals) if vals else None)
            rows.append(row)
            with open(out/f"{label}_join{conn}_edges.csv","w",newline="") as f:
                wr=csv.writer(f); wr.writerow(["source","target","source_scalar","target_scalar"])
                for u,v in G.edges: wr.writerow([u,v,sval(G,u,work),sval(G,v,work)])

    with open(out/"structural_summary.csv","w",newline="") as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
    (out/"structural_summary.json").write_text(json.dumps(rows,indent=2))
    print("===== PHASE 2 REAL-FIELD STRUCTURAL SUMMARY =====")
    print("builder signature:",inspect.signature(_build_join_tree_graph))
    print("sample:",args.sample)
    print("out:",out)
    for r in rows: print(r)

if __name__=="__main__":
    main()
