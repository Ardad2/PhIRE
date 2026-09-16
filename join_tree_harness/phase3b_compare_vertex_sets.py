#!/usr/bin/env python3
"""
Phase 3B proper: compare colleague hierarchical-builder critical vertex IDs
against exact TTK numerical-tree VertexId sets for sample 69.

Primary candidate: 6_anti
Control:           6_main

CNN/UV TTK VertexIds are mapped through the already-audited legacy transpose.
F1 TTK VertexIds use identity mapping.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, sys
from pathlib import Path
import numpy as np

EXPECTED_COLLEAGUE_COUNTS={
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
    if v.shape!=(crop,crop,2): raise ValueError((root,fn,v.shape))
    return np.hypot(v[...,0],v[...,1])

def read_ttk_csv(path):
    rows=[]
    with path.open(newline="") as f:
        for r in csv.DictReader(f):
            rows.append({
                "row":int(r["row"]),
                "NodeId":int(r["NodeId"]),
                "VertexId":int(r["VertexId"]),
                "Scalar":float(r["Scalar"]),
                "CriticalType":int(r["CriticalType"]),
            })
    return rows

def orient_vid(v,n,orientation):
    if orientation=="identity":
        return v
    if orientation=="transpose":
        x=v % n
        y=v // n
        return y + n*x
    raise ValueError(orientation)

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
    ap.add_argument("--ttk-extract",required=True)
    ap.add_argument("--out",required=True)
    args=ap.parse_args()

    repo=Path(args.repo).expanduser().resolve()
    cnn=Path(args.cnn_root).expanduser().resolve()
    uv=Path(args.uv_root).expanduser().resolve()
    f1=Path(args.f1_root).expanduser().resolve()
    tex=Path(args.ttk_extract).expanduser().resolve()
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

    # Build each colleague graph once.
    graphs={}
    node_sets={}
    for field_label,field in fields.items():
        flat=np.asarray(field,dtype=np.float64).ravel(order="C")
        for mode in ("6_main","6_anti"):
            G=_build_join_tree_graph(flat,adjacency(args.crop,args.crop,mode))
            nodes=list(G.nodes)
            if not all(isinstance(x,(int,np.integer)) for x in nodes):
                raise RuntimeError(f"{field_label}/{mode}: graph nodes are not integer grid IDs")
            nodes=[int(x) for x in nodes]
            if not all(0<=x<args.crop*args.crop for x in nodes):
                raise RuntimeError(f"{field_label}/{mode}: graph node outside grid range")
            if len(set(nodes)) != len(nodes):
                raise RuntimeError(f"{field_label}/{mode}: duplicate graph node IDs")
            exp=EXPECTED_COLLEAGUE_COUNTS[(field_label,mode)]
            if len(nodes)!=exp:
                raise RuntimeError(f"{field_label}/{mode}: count {len(nodes)} != frozen {exp}")
            graphs[(field_label,mode)]=G
            node_sets[(field_label,mode)]=set(nodes)

    specs=[
      ("CNN_GT","GT","transpose"),
      ("UV_GT","GT","transpose"),
      ("F1_GT","GT","identity"),
      ("CNN_SR","CNN","transpose"),
      ("UV_SR","UV","transpose"),
      ("F1_SR","F1","identity"),
    ]

    summary=[]
    detail_rows=[]
    for ttk_label,field_label,orientation in specs:
        rows=read_ttk_csv(tex/f"{ttk_label}_ttk_nodes.csv")
        raw_vids=[r["VertexId"] for r in rows]
        if len(set(raw_vids))!=len(raw_vids):
            raise RuntimeError(f"{ttk_label}: TTK VertexId is not unique")
        mapped={orient_vid(v,args.crop,orientation) for v in raw_vids}
        if not all(0<=v<args.crop*args.crop for v in mapped):
            raise RuntimeError(f"{ttk_label}: mapped VertexId outside range")

        scalar_by_mapped={}
        ctype_by_mapped={}
        for r in rows:
            mv=orient_vid(r["VertexId"],args.crop,orientation)
            scalar_by_mapped[mv]=r["Scalar"]
            ctype_by_mapped[mv]=r["CriticalType"]

        flat=np.asarray(fields[field_label],dtype=np.float64).ravel(order="C")

        for mode in ("6_main","6_anti"):
            cset=node_sets[(field_label,mode)]
            inter=mapped & cset
            union=mapped | cset
            ttk_only=mapped-cset
            col_only=cset-mapped

            diffs=np.asarray([abs(scalar_by_mapped[v]-flat[v]) for v in inter],dtype=float)
            row={
              "ttk_label":ttk_label,
              "field":field_label,
              "orientation":orientation,
              "adjacency":mode,
              "ttk_nodes":len(mapped),
              "colleague_nodes":len(cset),
              "intersection":len(inter),
              "union":len(union),
              "jaccard":len(inter)/len(union) if union else 1.0,
              "ttk_recall":len(inter)/len(mapped) if mapped else 1.0,
              "colleague_precision":len(inter)/len(cset) if cset else 1.0,
              "ttk_only":len(ttk_only),
              "colleague_only":len(col_only),
              "shared_scalar_mae":float(diffs.mean()) if len(diffs) else None,
              "shared_scalar_max_abs":float(diffs.max()) if len(diffs) else None,
            }
            summary.append(row)

            for v in sorted(ttk_only):
                detail_rows.append([ttk_label,mode,"TTK_ONLY",v,
                                    scalar_by_mapped.get(v,""),
                                    float(flat[v]),ctype_by_mapped.get(v,"")])
            for v in sorted(col_only):
                detail_rows.append([ttk_label,mode,"COLLEAGUE_ONLY",v,
                                    "",float(flat[v]),""])

    with (out/"vertex_overlap_summary.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0].keys()))
        w.writeheader(); w.writerows(summary)
    (out/"vertex_overlap_summary.json").write_text(json.dumps(summary,indent=2))

    with (out/"vertex_overlap_differences.csv").open("w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["ttk_label","adjacency","status","corrected_vertex_id",
                    "ttk_scalar","field_scalar","ttk_critical_type"])
        w.writerows(detail_rows)

    print("===== PHASE 3B CRITICAL-VERTEX OVERLAP =====")
    print("colleague graph-node ID semantics: PASS (integer original grid IDs)")
    print("frozen Phase-3A counts: PASS")
    print()
    for r in summary:
        print(
          f'{r["ttk_label"]:7s} {r["adjacency"]:7s} '
          f'TTK={r["ttk_nodes"]:4d} COL={r["colleague_nodes"]:4d} '
          f'I={r["intersection"]:4d} J={r["jaccard"]:.6f} '
          f'recall={r["ttk_recall"]:.6f} precision={r["colleague_precision"]:.6f} '
          f'TTKonly={r["ttk_only"]:3d} COLonly={r["colleague_only"]:3d} '
          f'scalar_max={r["shared_scalar_max_abs"]:.3e}'
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
