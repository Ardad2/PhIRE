#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, inspect, json, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def add_repo_src(repo: Path):
    src = repo / "src"
    if not src.exists():
        raise FileNotFoundError(src)
    sys.path.insert(0, str(src))

def make_three_basin_field(h=17, w=23):
    yy, xx = np.mgrid[0:h, 0:w]
    A = 0.00 + 0.055 * ((xx - 5.0)**2 + (yy - 5.0)**2)
    B = 0.18 + 0.060 * ((xx - 11.0)**2 + (yy - 5.0)**2)
    C = 0.08 + 0.040 * ((xx - 17.0)**2 + (yy - 12.0)**2)
    field = np.minimum(np.minimum(A, B), C)
    field += 1e-9*np.arange(h*w, dtype=float).reshape(h, w)
    return field

def grid_adjacency(h, w, connectivity=4):
    offs = [(-1,0),(1,0),(0,-1),(0,1)]
    if connectivity == 8:
        offs += [(-1,-1),(-1,1),(1,-1),(1,1)]
    elif connectivity != 4:
        raise ValueError("connectivity must be 4 or 8")
    adj = {}
    for r in range(h):
        for c in range(w):
            i = r*w+c
            adj[i] = [(r+dr)*w+(c+dc) for dr,dc in offs
                      if 0 <= r+dr < h and 0 <= c+dc < w]
    return adj

def n_components(G):
    if G.number_of_nodes() == 0: return 0
    return nx.number_weakly_connected_components(G) if G.is_directed() else nx.number_connected_components(G)

def is_tree(G):
    if G.number_of_nodes() == 0: return False
    UG = G.to_undirected() if G.is_directed() else G
    return nx.is_tree(UG)

def scalar_for_node(G, n, flat):
    for k in ("value","scalar","f","height"):
        if k in G.nodes[n]:
            try: return float(G.nodes[n][k])
            except Exception: pass
    if isinstance(n, (int, np.integer)) and 0 <= int(n) < len(flat):
        return float(flat[int(n)])
    return float("nan")

def summarize(name, G, flat):
    UG = G.to_undirected() if G.is_directed() else G
    deg = dict(UG.degree())
    vals = [scalar_for_node(G,n,flat) for n in G.nodes]
    vals = [v for v in vals if np.isfinite(v)]
    return dict(
        name=name,
        directed=bool(G.is_directed()),
        nodes=int(G.number_of_nodes()),
        edges=int(G.number_of_edges()),
        components=int(n_components(G)),
        is_tree_undirected=bool(is_tree(G)),
        leaves_degree1=int(sum(d==1 for d in deg.values())),
        branch_nodes_degree_ge3=int(sum(d>=3 for d in deg.values())),
        self_loops=int(nx.number_of_selfloops(UG)),
        min_node_scalar=min(vals) if vals else None,
        max_node_scalar=max(vals) if vals else None,
    )

def export_graph(prefix, G, flat):
    UG = G.to_undirected() if G.is_directed() else G
    with open(str(prefix)+"_nodes.csv","w",newline="") as f:
        w=csv.writer(f); w.writerow(["node","scalar","degree"])
        for n in G.nodes:
            w.writerow([n, scalar_for_node(G,n,flat), UG.degree(n)])
    with open(str(prefix)+"_edges.csv","w",newline="") as f:
        w=csv.writer(f); w.writerow(["source","target","source_scalar","target_scalar"])
        for u,v in G.edges:
            w.writerow([u,v,scalar_for_node(G,u,flat),scalar_for_node(G,v,flat)])

def draw_graph(path, title, G, flat):
    fig, ax = plt.subplots(figsize=(8,6))
    if G.number_of_nodes()==0:
        ax.text(.5,.5,"Empty graph",ha="center",va="center"); ax.set_axis_off()
    else:
        UG = G.to_undirected() if G.is_directed() else G
        pos = nx.spring_layout(UG, seed=7)
        nx.draw_networkx_edges(UG,pos,ax=ax,width=.8,alpha=.6)
        nx.draw_networkx_nodes(UG,pos,ax=ax,node_size=45)
        if G.number_of_nodes() <= 40:
            labels={n:f"{n}\n{scalar_for_node(G,n,flat):.2f}" for n in G.nodes}
            nx.draw_networkx_labels(UG,pos,labels=labels,font_size=6,ax=ax)
        ax.set_title(title); ax.set_axis_off()
    fig.savefig(path,dpi=180,bbox_inches="tight"); plt.close(fig)

def call_current_2d(fn, field):
    sig = inspect.signature(fn)
    kwargs={}
    if "direction" in sig.parameters:
        kwargs["direction"]="join"
    result = fn(field, **kwargs)
    if isinstance(result, tuple):
        return result[0], result[1:], str(sig)
    return result, (), str(sig)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--repo", default=str(Path.home()/ "PhIRE/third_party/tda-toolkit-mapper"))
    ap.add_argument("--out", default=str(Path.home()/ "PhIRE/join_tree_harness/phase1_synthetic"))
    args=ap.parse_args()
    repo=Path(args.repo).expanduser().resolve()
    out=Path(args.out).expanduser().resolve()
    out.mkdir(parents=True,exist_ok=True)
    add_repo_src(repo)
    from tda_toolkit.merge_tree import _build_join_tree_graph, get_merge_tree_graph_2d

    field=make_three_basin_field()
    h,w=field.shape
    flat=field.ravel()
    np.save(out/"synthetic_field.npy",field)

    fig,ax=plt.subplots(figsize=(8,5))
    im=ax.imshow(field,origin="upper")
    ax.set_title("Synthetic three-basin scalar field")
    fig.colorbar(im,ax=ax,label="scalar value")
    fig.savefig(out/"synthetic_field.png",dpi=180,bbox_inches="tight")
    plt.close(fig)

    Gpair, extras, sig2d = call_current_2d(get_merge_tree_graph_2d,field)
    G4 = _build_join_tree_graph(flat, grid_adjacency(h,w,4))
    G8 = _build_join_tree_graph(flat, grid_adjacency(h,w,8))

    graphs={
        "current_2d_pair_graph":Gpair,
        "hierarchical_join_tree_4nbr":G4,
        "hierarchical_join_tree_8nbr":G8
    }
    rows=[]
    for name,G in graphs.items():
        rows.append(summarize(name,G,flat))
        export_graph(out/name,G,flat)
        draw_graph(out/f"{name}.png",name,G,flat)

    sigs={
        "get_merge_tree_graph_2d":sig2d,
        "_build_join_tree_graph":str(inspect.signature(_build_join_tree_graph)),
        "pair_route_extra_return_objects":len(extras)
    }
    (out/"function_signatures.json").write_text(json.dumps(sigs,indent=2))
    (out/"summary.json").write_text(json.dumps(rows,indent=2))

    lines=["===== JOIN-TREE PHASE 1 SYNTHETIC SUMMARY =====",
           f"repo: {repo}",f"out: {out}",f"field shape: {field.shape}","",
           "FUNCTION SIGNATURES"]
    lines += [f"{k}: {v}" for k,v in sigs.items()]
    lines.append("")
    for r in rows:
        lines.append(f"--- {r['name']} ---")
        lines += [f"{k}: {v}" for k,v in r.items() if k!="name"]
        lines.append("")
    pair=next(r for r in rows if r["name"]=="current_2d_pair_graph")
    j4=next(r for r in rows if r["name"]=="hierarchical_join_tree_4nbr")
    lines += ["DIAGNOSTIC INTERPRETATION",
              f"pair_route_fragmented: {pair['components']>1}",
              f"hierarchical_4nbr_connected_tree: {j4['components']==1 and j4['is_tree_undirected']}",
              "",
              "Do not interpret this as TTK parity yet. If the hierarchical 4-neighbor route is a connected tree while the current 2D route is fragmented, Phase 1 supports the semantic distinction found in the source audit."]
    txt="\n".join(lines)+"\n"
    (out/"summary.txt").write_text(txt)
    print(txt)

if __name__=="__main__":
    main()
