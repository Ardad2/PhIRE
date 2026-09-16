#!/usr/bin/env python3
"""
Phase 4B step 2: full exact-input Join-Tree construction-parity sweep.

Primary criterion:
  colleague-only nodes == 0
  TTK-only nodes == 1
  TTK-only node == field argmax
  TTK-only node degree == 1
  common-edge Jaccard/recall/precision == 1

Expected total:
  1008 trees = 168 x 3 methods x GT/SR
"""
from __future__ import annotations
import argparse, csv, hashlib, json, sys, time
from pathlib import Path
import numpy as np
import networkx as nx

def add_repo_src(repo): sys.path.insert(0,str(repo/"src"))

def adjacency(h=160,w=160):
    offsets=[(-1,0),(1,0),(0,-1),(0,1),(-1,1),(1,-1)]
    return {r*w+c:[(r+dr)*w+(c+dc) for dr,dc in offsets
                    if 0<=r+dr<h and 0<=c+dc<w]
            for r in range(h) for c in range(w)}

def metrics(A,B):
    I=A&B; U=A|B
    return dict(
      intersection=len(I),
      jaccard=len(I)/len(U) if U else 1.0,
      a_recall=len(I)/len(A) if A else 1.0,
      b_precision=len(I)/len(B) if B else 1.0,
      a_only=len(A-I),b_only=len(B-I),
    )

def sha256(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

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
    adj=adjacency()

    files=sorted(
        ext.glob("*.npz"),
        key=lambda p: (
            int(p.stem.rsplit("_s",1)[1]),
            p.stem.split("_")[0],
            p.stem.split("_")[1],
        )
    )
    if len(files)!=1008:
        raise RuntimeError(f"Expected 1008 extracted trees, found {len(files)}")

    rows=[]; diff=[]
    t0=time.time()

    for idx,npz in enumerate(files,1):
        z=np.load(npz)
        field=np.asarray(z["field"]).reshape(-1)
        node_id=np.asarray(z["node_id"],dtype=np.int64)
        vertex_id=np.asarray(z["vertex_id"],dtype=np.int64)
        up_id=np.asarray(z["up_id"],dtype=np.int64)
        down_id=np.asarray(z["down_id"],dtype=np.int64)

        parts=npz.stem.split("_")
        method=parts[0]; kind=parts[1]; sample=int(parts[2][1:])

        node_to_vid={int(n):int(v) for n,v in zip(node_id,vertex_id)}
        ttk_nodes=set(node_to_vid.values())
        ttk_edges={tuple(sorted((node_to_vid[int(d)],node_to_vid[int(u)])))
                   for d,u in zip(down_id,up_id)
                   if node_to_vid[int(d)] != node_to_vid[int(u)]}
        T=nx.Graph(); T.add_nodes_from(ttk_nodes); T.add_edges_from(ttk_edges)
        if not nx.is_tree(T):
            raise RuntimeError(f"{npz.stem}: extracted TTK is not tree")

        G=_build_join_tree_graph(field,adj)
        c_nodes={int(v) for v in G.nodes}
        c_edges={tuple(sorted((int(a),int(b)))) for a,b in G.edges if a!=b}
        C=nx.Graph(); C.add_nodes_from(c_nodes); C.add_edges_from(c_edges)
        if not nx.is_tree(C):
            raise RuntimeError(f"{npz.stem}: colleague graph is not tree")

        nm=metrics(ttk_nodes,c_nodes)
        common=ttk_nodes&c_nodes
        te={e for e in ttk_edges if e[0] in common and e[1] in common}
        ce={e for e in c_edges if e[0] in common and e[1] in common}
        em=metrics(te,ce)

        ttk_only=sorted(ttk_nodes-c_nodes)
        col_only=sorted(c_nodes-ttk_nodes)
        argmax=int(np.argmax(field)); argmin=int(np.argmin(field))

        extra=[]
        for v in ttk_only:
            extra.append({
              "vertex_id":v,
              "degree":int(T.degree(v)),
              "is_argmax":bool(v==argmax),
              "is_argmin":bool(v==argmin),
              "scalar":float(field[v]),
            })

        exact=(
          len(ttk_only)==1 and
          len(col_only)==0 and
          ttk_only[0]==argmax and
          T.degree(ttk_only[0])==1 and
          abs(em["jaccard"]-1.0)<1e-15 and
          abs(em["a_recall"]-1.0)<1e-15 and
          abs(em["b_precision"]-1.0)<1e-15
        )

        row={
          "sample":sample,"method":method,"kind":kind,
          "ttk_nodes":len(ttk_nodes),
          "colleague_nodes":len(c_nodes),
          "common_nodes":len(common),
          "node_jaccard":nm["jaccard"],
          "ttk_only_nodes":len(ttk_only),
          "colleague_only_nodes":len(col_only),
          "common_edge_jaccard":em["jaccard"],
          "common_edge_ttk_recall":em["a_recall"],
          "common_edge_colleague_precision":em["b_precision"],
          "ttk_only_info":json.dumps(extra,sort_keys=True),
          "root_only_exact_parity":bool(exact),
        }
        rows.append(row)

        if not exact:
            for v in ttk_only:
                diff.append([sample,method,kind,"TTK_ONLY",v,int(T.degree(v)),
                             v==argmax,v==argmin,float(field[v])])
            for v in col_only:
                diff.append([sample,method,kind,"COLLEAGUE_ONLY",v,int(C.degree(v)),
                             v==argmax,v==argmin,float(field[v])])

        if idx % 48 == 0 or idx==len(files):
            passed=sum(r["root_only_exact_parity"] for r in rows)
            elapsed=time.time()-t0
            print(f"processed {idx}/{len(files)} trees; exact={passed}/{idx}; "
                  f"elapsed={elapsed:.1f}s")

    with (out/"full_parity_summary.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    (out/"full_parity_summary.json").write_text(json.dumps(rows,indent=2))

    with (out/"full_parity_failures.csv").open("w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["sample","method","kind","status","vertex_id","degree",
                    "is_argmax","is_argmin","scalar"])
        w.writerows(diff)

    passed=sum(r["root_only_exact_parity"] for r in rows)
    report={
      "trees":len(rows),
      "root_only_exact_parity":passed,
      "fraction":passed/len(rows),
      "failures":len(rows)-passed,
      "GT":{},
      "SR":{},
      "by_method":{},
    }

    for kind in ("GT","SR"):
        rr=[r for r in rows if r["kind"]==kind]
        report[kind]={
          "passed":sum(r["root_only_exact_parity"] for r in rr),
          "total":len(rr)
        }
    for method in ("CNN","UV","F1"):
        rr=[r for r in rows if r["method"]==method]
        report["by_method"][method]={
          "passed":sum(r["root_only_exact_parity"] for r in rr),
          "total":len(rr),
          "GT_passed":sum(r["root_only_exact_parity"] for r in rr if r["kind"]=="GT"),
          "SR_passed":sum(r["root_only_exact_parity"] for r in rr if r["kind"]=="SR"),
        }

    (out/"full_report.json").write_text(json.dumps(report,indent=2))

    print("===== PHASE 4B FULL 168-SAMPLE CONSTRUCTION-PARITY SWEEP =====")
    print(json.dumps(report,indent=2))
    if report["failures"]:
        print()
        print("FAILURE ROWS:")
        for r in rows:
            if not r["root_only_exact_parity"]:
                print(r)

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
