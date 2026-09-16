#!/usr/bin/env python3
"""
Phase 4D: diagnose the sole remaining construction-parity exception:
sample 127 CNN GT.

Compare the TTK GT trees across:
  CNN raw orientation
  UV raw orientation
  F1 mapped by transpose into CNN/UV orientation

The exact CNN and UV scalar fields are bit-identical. F1 is the exact transpose.

Questions:
1) Are the TTK critical-vertex sets identical after alignment?
2) Are the TTK edge sets identical?
3) Which track agrees with the colleague 6_anti hierarchy?
4) Do CriticalType labels differ on the locally affected vertices?
"""
from __future__ import annotations
import argparse, csv, hashlib, json, sys
from pathlib import Path
import numpy as np
import networkx as nx

N=160
SAMPLE=127

def add_repo_src(repo):
    sys.path.insert(0,str(repo/"src"))

def adjacency(h=N,w=N):
    offs=[(-1,0),(1,0),(0,-1),(0,1),(-1,1),(1,-1)]
    return {r*w+c:[(r+dr)*w+(c+dc) for dr,dc in offs
                    if 0<=r+dr<h and 0<=c+dc<w]
            for r in range(h) for c in range(w)}

def tr(v):
    v=int(v)
    return (v % N)*N + (v // N)

def sha256(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def metrics(A,B):
    I=A&B; U=A|B
    return {
      "a":len(A),"b":len(B),"intersection":len(I),
      "jaccard":len(I)/len(U) if U else 1.0,
      "a_only":len(A-I),"b_only":len(B-I),
    }

def load_tree(npz_path: Path, transpose=False):
    z=np.load(npz_path)
    field=np.asarray(z["field"]).reshape(N,N)
    node_id=np.asarray(z["node_id"],dtype=np.int64)
    vertex_id=np.asarray(z["vertex_id"],dtype=np.int64)
    ctype=np.asarray(z["critical_type"],dtype=np.int16)
    up=np.asarray(z["up_id"],dtype=np.int64)
    down=np.asarray(z["down_id"],dtype=np.int64)

    if transpose:
        mapped_vid=np.asarray([tr(v) for v in vertex_id],dtype=np.int64)
        aligned_field=field.T.copy()
    else:
        mapped_vid=vertex_id.copy()
        aligned_field=field.copy()

    node_to_vid={int(n):int(v) for n,v in zip(node_id,mapped_vid)}
    ctype_by_vid={int(v):int(c) for v,c in zip(mapped_vid,ctype)}
    nodes=set(node_to_vid.values())
    edges={tuple(sorted((node_to_vid[int(d)],node_to_vid[int(u)])))
           for d,u in zip(down,up)
           if node_to_vid[int(d)] != node_to_vid[int(u)]}

    G=nx.Graph(); G.add_nodes_from(nodes); G.add_edges_from(edges)
    if not nx.is_tree(G):
        raise RuntimeError(f"{npz_path}: mapped TTK graph is not a tree")

    return {
      "field":aligned_field,
      "nodes":nodes,
      "edges":edges,
      "ctype":ctype_by_vid,
      "graph":G,
      "source":str(npz_path),
      "npz_sha256":sha256(npz_path),
    }

def edge_records(edges, field, ctype_maps):
    flat=field.reshape(-1)
    rows=[]
    for a,b in sorted(edges):
        rec={
          "a":int(a),"b":int(b),
          "scalar_a":float(flat[a]),"scalar_b":float(flat[b]),
          "abs_gap":float(abs(float(flat[a])-float(flat[b]))),
        }
        for name,cmap in ctype_maps.items():
            rec[f"ctype_{name}_a"]=cmap.get(a)
            rec[f"ctype_{name}_b"]=cmap.get(b)
        rows.append(rec)
    return rows

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--repo",required=True)
    ap.add_argument("--extract",required=True)
    ap.add_argument("--out",required=True)
    args=ap.parse_args()

    repo=Path(args.repo).expanduser().resolve()
    ext=Path(args.extract).expanduser().resolve()
    out=Path(args.out).expanduser().resolve(); out.mkdir(parents=True,exist_ok=True)

    add_repo_src(repo)
    from tda_toolkit.merge_tree import _build_join_tree_graph

    cnn=load_tree(ext/"CNN_GT_s127.npz",transpose=False)
    uv =load_tree(ext/"UV_GT_s127.npz", transpose=False)
    f1 =load_tree(ext/"F1_GT_s127.npz", transpose=True)

    # Exact aligned field equality gate
    if not np.array_equal(cnn["field"],uv["field"]):
        raise RuntimeError("CNN != UV exact aligned field")
    if not np.array_equal(cnn["field"],f1["field"]):
        raise RuntimeError("CNN != transposed F1 exact aligned field")

    field=cnn["field"]
    flat=field.reshape(-1)

    # Colleague reference
    CG=_build_join_tree_graph(flat,adjacency())
    c_nodes={int(v) for v in CG.nodes}
    c_edges={tuple(sorted((int(a),int(b)))) for a,b in CG.edges if a!=b}

    tracks={"CNN":cnn,"UV":uv,"F1T":f1}
    pairwise={}
    for a,b in (("CNN","UV"),("CNN","F1T"),("UV","F1T")):
        A=tracks[a]; B=tracks[b]
        pairwise[f"{a}_vs_{b}"]={
          "node_metrics":metrics(A["nodes"],B["nodes"]),
          "edge_metrics":metrics(A["edges"],B["edges"]),
          "a_only_edges":edge_records(A["edges"]-B["edges"],field,
                                      {a:A["ctype"],b:B["ctype"]}),
          "b_only_edges":edge_records(B["edges"]-A["edges"],field,
                                      {a:A["ctype"],b:B["ctype"]}),
        }

    versus_colleague={}
    for name,T in tracks.items():
        common=T["nodes"] & c_nodes
        te={e for e in T["edges"] if e[0] in common and e[1] in common}
        ce={e for e in c_edges if e[0] in common and e[1] in common}
        versus_colleague[name]={
          "node_metrics":metrics(T["nodes"],c_nodes),
          "common_edge_metrics":metrics(te,ce),
          "ttk_only_common_edges":edge_records(te-ce,field,{name:T["ctype"]}),
          "colleague_only_common_edges":edge_records(ce-te,field,{name:T["ctype"]}),
          "ttk_only_nodes":sorted(T["nodes"]-c_nodes),
          "colleague_only_nodes":sorted(c_nodes-T["nodes"]),
        }

    affected=set()
    for obj in pairwise.values():
        for k in ("a_only_edges","b_only_edges"):
            for r in obj[k]:
                affected.update((r["a"],r["b"]))

    affected_info=[]
    for v in sorted(affected):
        affected_info.append({
          "vertex_id":v,
          "scalar":float(flat[v]),
          "CNN_ctype":cnn["ctype"].get(v),
          "UV_ctype":uv["ctype"].get(v),
          "F1T_ctype":f1["ctype"].get(v),
          "CNN_degree":int(cnn["graph"].degree(v)) if v in cnn["nodes"] else None,
          "UV_degree":int(uv["graph"].degree(v)) if v in uv["nodes"] else None,
          "F1T_degree":int(f1["graph"].degree(v)) if v in f1["nodes"] else None,
          "COL_degree":int(nx.Graph(CG).degree(v)) if v in c_nodes else None,
        })

    report={
      "sample":127,
      "aligned_field_equal_CNN_UV":True,
      "aligned_field_equal_CNN_F1T":True,
      "field_sha256":hashlib.sha256(np.ascontiguousarray(field).view(np.uint8)).hexdigest(),
      "tracks":{
        k:{
          "nodes":len(v["nodes"]),
          "edges":len(v["edges"]),
          "npz_sha256":v["npz_sha256"],
        } for k,v in tracks.items()
      },
      "colleague":{"nodes":len(c_nodes),"edges":len(c_edges)},
      "pairwise_ttk":pairwise,
      "versus_colleague":versus_colleague,
      "affected_vertices":affected_info,
    }

    (out/"phase4d_cross_track_report.json").write_text(json.dumps(report,indent=2))

    # concise edge diff CSV
    csv_rows=[]
    for pair,obj in pairwise.items():
        for status,key in (("A_ONLY","a_only_edges"),("B_ONLY","b_only_edges")):
            for r in obj[key]:
                csv_rows.append([pair,status,r["a"],r["b"],
                                 r["scalar_a"],r["scalar_b"],r["abs_gap"]])
    with (out/"phase4d_pairwise_edge_differences.csv").open("w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["pair","status","vertex_a","vertex_b",
                    "scalar_a","scalar_b","abs_gap"])
        w.writerows(csv_rows)

    print("===== PHASE 4D SAMPLE-127 CROSS-TRACK TTK DIAGNOSTIC =====")
    print("Aligned exact fields: CNN == UV == transpose(F1): PASS")
    print()
    print("TTK TRACK SIZES")
    for k,v in tracks.items():
        print(k,"nodes=",len(v["nodes"]),"edges=",len(v["edges"]))
    print("COL nodes=",len(c_nodes),"edges=",len(c_edges))

    print()
    print("PAIRWISE TTK COMPARISONS")
    for k,v in pairwise.items():
        print(k)
        print("  nodes:",v["node_metrics"])
        print("  edges:",v["edge_metrics"])
        print("  A-only edges:",json.dumps(v["a_only_edges"],indent=2))
        print("  B-only edges:",json.dumps(v["b_only_edges"],indent=2))

    print()
    print("EACH TTK TRACK VS COLLEAGUE")
    for k,v in versus_colleague.items():
        print(k)
        print("  nodes:",v["node_metrics"])
        print("  common edges:",v["common_edge_metrics"])
        print("  TTK-only common:",json.dumps(v["ttk_only_common_edges"],indent=2))
        print("  COL-only common:",json.dumps(v["colleague_only_common_edges"],indent=2))
        print("  TTK-only nodes:",v["ttk_only_nodes"])
        print("  COL-only nodes:",v["colleague_only_nodes"])

    print()
    print("AFFECTED VERTICES")
    print(json.dumps(affected_info,indent=2))

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
