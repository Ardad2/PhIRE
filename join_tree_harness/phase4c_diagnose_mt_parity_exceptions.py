#!/usr/bin/env python3
"""
Phase 4C: diagnose the four Phase-4B strict-criterion exceptions.

Targets:
  sample 86: CNN/UV/F1 GT
  sample 127: CNN GT

Questions:
1) Is the sample-86 TTK-only degree-1 vertex one of multiple *exact* float32
   global maxima, even though it is not the first index returned by np.argmax?
2) For sample-127 CNN GT, exactly which shared-node arcs differ?
3) Do the differing sample-127 endpoints involve equal / nearly equal float32
   scalar values, consistent with a tie-breaking / ordering issue?

Uses the already frozen Phase-4B extracted NPZ files.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, sys
from pathlib import Path
import numpy as np
import networkx as nx

TARGETS=[
    (86,"CNN","GT"),
    (86,"UV","GT"),
    (86,"F1","GT"),
    (127,"CNN","GT"),
]

def add_repo_src(repo):
    sys.path.insert(0,str(repo/"src"))

def adjacency(h=160,w=160):
    offsets=[(-1,0),(1,0),(0,-1),(0,1),(-1,1),(1,-1)]
    return {r*w+c:[(r+dr)*w+(c+dc) for dr,dc in offsets
                    if 0<=r+dr<h and 0<=c+dc<w]
            for r in range(h) for c in range(w)}

def sha256_bytes(a):
    return hashlib.sha256(np.ascontiguousarray(a).view(np.uint8)).hexdigest()

def sha256(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):
            h.update(b)
    return h.hexdigest()

def edge_details(edges, field, ctype_by_vid):
    out=[]
    for a,b in sorted(edges):
        fa=float(field[a]); fb=float(field[b])
        out.append({
            "a":int(a),"b":int(b),
            "scalar_a":fa,"scalar_b":fb,
            "abs_scalar_gap":abs(fa-fb),
            "ulp_gap_float32":int(abs(
                np.asarray(np.float32(fa)).view(np.int32).item()
                - np.asarray(np.float32(fb)).view(np.int32).item()
            )),
            "critical_type_a":ctype_by_vid.get(int(a)),
            "critical_type_b":ctype_by_vid.get(int(b)),
        })
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--repo",required=True)
    ap.add_argument("--extract",required=True)
    ap.add_argument("--out",required=True)
    args=ap.parse_args()

    repo=Path(args.repo).expanduser().resolve()
    ext=Path(args.extract).expanduser().resolve()
    out=Path(args.out).expanduser().resolve()
    out.mkdir(parents=True,exist_ok=True)

    add_repo_src(repo)
    from tda_toolkit.merge_tree import _build_join_tree_graph
    adj=adjacency()

    report={}
    csv_rows=[]

    for sample,method,kind in TARGETS:
        label=f"{method}_{kind}_s{sample}"
        p=ext/f"{label}.npz"
        if not p.exists():
            raise FileNotFoundError(p)

        z=np.load(p)
        field=np.asarray(z["field"]).reshape(-1)
        node_id=np.asarray(z["node_id"],dtype=np.int64)
        vertex_id=np.asarray(z["vertex_id"],dtype=np.int64)
        ctype=np.asarray(z["critical_type"],dtype=np.int16)
        up_id=np.asarray(z["up_id"],dtype=np.int64)
        down_id=np.asarray(z["down_id"],dtype=np.int64)

        node_to_vid={int(n):int(v) for n,v in zip(node_id,vertex_id)}
        ctype_by_vid={int(v):int(c) for v,c in zip(vertex_id,ctype)}

        ttk_nodes=set(node_to_vid.values())
        ttk_edges={tuple(sorted((node_to_vid[int(d)],node_to_vid[int(u)])))
                   for d,u in zip(down_id,up_id)
                   if node_to_vid[int(d)]!=node_to_vid[int(u)]}
        T=nx.Graph(); T.add_nodes_from(ttk_nodes); T.add_edges_from(ttk_edges)
        if not nx.is_tree(T):
            raise RuntimeError(f"{label}: TTK not tree")

        G=_build_join_tree_graph(field,adj)
        col_nodes={int(v) for v in G.nodes}
        col_edges={tuple(sorted((int(a),int(b)))) for a,b in G.edges if a!=b}
        C=nx.Graph(); C.add_nodes_from(col_nodes); C.add_edges_from(col_edges)
        if not nx.is_tree(C):
            raise RuntimeError(f"{label}: colleague not tree")

        ttk_only=sorted(ttk_nodes-col_nodes)
        col_only=sorted(col_nodes-ttk_nodes)

        max_val=np.max(field)
        min_val=np.min(field)
        max_vids=np.flatnonzero(field==max_val).astype(int).tolist()
        min_vids=np.flatnonzero(field==min_val).astype(int).tolist()

        common=ttk_nodes & col_nodes
        te={e for e in ttk_edges if e[0] in common and e[1] in common}
        ce={e for e in col_edges if e[0] in common and e[1] in common}
        t_only_edges=te-ce
        c_only_edges=ce-te

        extra_info=[]
        for v in ttk_only:
            extra_info.append({
                "vertex_id":v,
                "degree":int(T.degree(v)),
                "scalar":float(field[v]),
                "equals_global_max_value":bool(field[v]==max_val),
                "is_first_np_argmax":bool(v==int(np.argmax(field))),
                "is_any_exact_max_vertex":bool(v in max_vids),
                "critical_type":ctype_by_vid.get(v),
            })

        # Show the top few distinct float32 values and multiplicities.
        vals,counts=np.unique(field,return_counts=True)
        order=np.argsort(vals)[::-1]
        top_levels=[
            {"value":float(vals[i]),"count":int(counts[i])}
            for i in order[:8]
        ]

        rec={
            "label":label,
            "field_dtype":str(field.dtype),
            "field_sha256_raw":sha256_bytes(field),
            "field_max":float(max_val),
            "field_min":float(min_val),
            "np_argmax":int(np.argmax(field)),
            "exact_max_vertex_count":len(max_vids),
            "exact_max_vertices":max_vids,
            "exact_min_vertex_count":len(min_vids),
            "exact_min_vertices":min_vids,
            "ttk_only_nodes":extra_info,
            "colleague_only_nodes":col_only,
            "common_ttk_edges":len(te),
            "common_colleague_edges":len(ce),
            "common_edge_intersection":len(te&ce),
            "ttk_only_common_edges":edge_details(t_only_edges,field,ctype_by_vid),
            "colleague_only_common_edges":edge_details(c_only_edges,field,ctype_by_vid),
            "top_distinct_scalar_levels":top_levels,
        }
        report[label]=rec

        for status,edges in (("TTK_ONLY_EDGE",t_only_edges),
                             ("COLLEAGUE_ONLY_EDGE",c_only_edges)):
            for e in edge_details(edges,field,ctype_by_vid):
                csv_rows.append([
                    sample,method,kind,status,
                    e["a"],e["b"],e["scalar_a"],e["scalar_b"],
                    e["abs_scalar_gap"],e["ulp_gap_float32"],
                    e["critical_type_a"],e["critical_type_b"],
                ])

    # Cross-track exact-field relationships for the two target GT samples.
    cross={}
    for sample in (86,127):
        fields={}
        for method in ("CNN","UV","F1"):
            z=np.load(ext/f"{method}_GT_s{sample}.npz")
            fields[method]=np.asarray(z["field"]).reshape(160,160)
        cross[f"s{sample}"]={
            "CNN_equals_UV":bool(np.array_equal(fields["CNN"],fields["UV"])),
            "CNN_equals_F1":bool(np.array_equal(fields["CNN"],fields["F1"])),
            "CNN_transpose_equals_F1":bool(np.array_equal(fields["CNN"].T,fields["F1"])),
            "UV_transpose_equals_F1":bool(np.array_equal(fields["UV"].T,fields["F1"])),
            "CNN_UV_max_abs_diff":float(np.max(np.abs(fields["CNN"].astype(np.float64)-fields["UV"].astype(np.float64)))),
            "CNN_T_F1_max_abs_diff":float(np.max(np.abs(fields["CNN"].T.astype(np.float64)-fields["F1"].astype(np.float64)))),
        }

    report["cross_track_field_relationships"]=cross

    (out/"phase4c_exception_diagnostic.json").write_text(json.dumps(report,indent=2))
    with (out/"phase4c_edge_differences.csv").open("w",newline="") as f:
        w=csv.writer(f)
        w.writerow([
            "sample","method","kind","status","vertex_a","vertex_b",
            "scalar_a","scalar_b","abs_scalar_gap","ulp_gap_float32",
            "critical_type_a","critical_type_b"
        ])
        w.writerows(csv_rows)

    print("===== PHASE 4C EXCEPTION DIAGNOSTIC =====")
    for sample,method,kind in TARGETS:
        label=f"{method}_{kind}_s{sample}"
        r=report[label]
        print()
        print("---",label,"---")
        print("field dtype:",r["field_dtype"])
        print("field max:",r["field_max"])
        print("np.argmax:",r["np_argmax"])
        print("exact max vertex count:",r["exact_max_vertex_count"])
        print("exact max vertices:",r["exact_max_vertices"])
        print("TTK-only nodes:",json.dumps(r["ttk_only_nodes"],sort_keys=True))
        print("colleague-only nodes:",r["colleague_only_nodes"])
        print("common edges TTK/COL/intersection:",
              r["common_ttk_edges"],r["common_colleague_edges"],
              r["common_edge_intersection"])
        print("TTK-only common edges:")
        print(json.dumps(r["ttk_only_common_edges"],indent=2))
        print("COL-only common edges:")
        print(json.dumps(r["colleague_only_common_edges"],indent=2))
        print("top distinct levels:")
        print(json.dumps(r["top_distinct_scalar_levels"],indent=2))

    print()
    print("===== CROSS-TRACK FIELD RELATIONSHIPS =====")
    print(json.dumps(cross,indent=2))

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
