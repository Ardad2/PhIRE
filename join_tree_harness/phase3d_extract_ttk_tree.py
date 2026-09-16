#!/usr/bin/env python3
"""
Phase 3D step 1: extract exact TTK sample-69 node and arc tables.

Port 0 provides:
  NodeId -> VertexId, Scalar, CriticalType

Port 1 provides:
  upNodeId / downNodeId per merge-tree arc

The output keeps raw TTK IDs. Orientation correction into authoritative C-order
is applied only in the comparison step.
"""
from __future__ import annotations
import argparse, csv, hashlib, json
from pathlib import Path
import vtk

def sha256(path: Path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1<<20), b""):
            h.update(b)
    return h.hexdigest()

def load_vtu(path: Path):
    r=vtk.vtkXMLUnstructuredGridReader()
    r.SetFileName(str(path)); r.Update()
    return r.GetOutput()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--phire", default=str(Path.home()/"PhIRE"))
    ap.add_argument("--out", required=True)
    args=ap.parse_args()

    phire=Path(args.phire).expanduser().resolve()
    out=Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    specs={
      "CNN_GT": phire/"ttk_runs_fixed/cnn/mt/cnn_GT_s69_speed_p160_x0_y0_mt_port_0.vtu",
      "CNN_SR": phire/"ttk_runs_fixed/cnn/mt/cnn_SR_s69_speed_p160_x0_y0_mt_port_0.vtu",
      "UV_GT": phire/"ttk_runs_fixed/topology_finetuning/candidateUV_expanded2688_topology/mt/GT/candidateUV_expanded2688_GT_s69_speed_p160_x0_y0_mt_port_0.vtu",
      "UV_SR": phire/"ttk_runs_fixed/topology_finetuning/candidateUV_expanded2688_topology/mt/SR/candidateUV_expanded2688_SR_s69_speed_p160_x0_y0_mt_port_0.vtu",
      "F1_GT": phire/"ttk_runs_fixed/topology_finetuning/candidateF_grad_E2_low_expanded2688_topology/mt/GT/candidateF_grad_E2_low_expanded2688_GT_s69_speed_p160_x0_y0_mt_port_0.vtu",
      "F1_SR": phire/"ttk_runs_fixed/topology_finetuning/candidateF_grad_E2_low_expanded2688_topology/mt/SR/candidateF_grad_E2_low_expanded2688_SR_s69_speed_p160_x0_y0_mt_port_0.vtu",
    }

    summary={}
    for label,p0 in specs.items():
        p1=Path(str(p0).replace("_mt_port_0.vtu","_mt_port_1.vtu"))
        if not p0.exists() or not p1.exists():
            raise FileNotFoundError((p0,p1))

        g0=load_vtu(p0); pd0=g0.GetPointData()
        req0=["NodeId","VertexId","Scalar","CriticalType"]
        a0={}
        for name in req0:
            a=pd0.GetArray(name)
            if a is None: raise RuntimeError(f"{p0}: missing {name}")
            a0[name]=a

        node_rows=[]
        node_ids=set()
        for i in range(g0.GetNumberOfPoints()):
            nid=int(round(a0["NodeId"].GetTuple1(i)))
            if nid in node_ids:
                raise RuntimeError(f"{label}: duplicate NodeId {nid}")
            node_ids.add(nid)
            node_rows.append([
                nid,
                int(round(a0["VertexId"].GetTuple1(i))),
                float(a0["Scalar"].GetTuple1(i)),
                int(round(a0["CriticalType"].GetTuple1(i))),
            ])

        nodes_csv=out/f"{label}_nodes.csv"
        with nodes_csv.open("w",newline="") as f:
            w=csv.writer(f)
            w.writerow(["NodeId","VertexId","Scalar","CriticalType"])
            w.writerows(node_rows)

        g1=load_vtu(p1); cd1=g1.GetCellData()
        up=cd1.GetArray("upNodeId"); down=cd1.GetArray("downNodeId")
        if up is None or down is None:
            raise RuntimeError(f"{p1}: missing upNodeId/downNodeId")
        if up.GetNumberOfTuples()!=g1.GetNumberOfCells() or down.GetNumberOfTuples()!=g1.GetNumberOfCells():
            raise RuntimeError(f"{label}: cell-array tuple mismatch")

        edge_rows=[]
        seen=set()
        for i in range(g1.GetNumberOfCells()):
            u=int(round(up.GetTuple1(i)))
            d=int(round(down.GetTuple1(i)))
            if u not in node_ids or d not in node_ids:
                raise RuntimeError(f"{label}: edge references unknown NodeId {(u,d)}")
            if u==d:
                raise RuntimeError(f"{label}: self edge {(u,d)}")
            key=(u,d)
            if key in seen:
                raise RuntimeError(f"{label}: duplicate directed edge {key}")
            seen.add(key)
            edge_rows.append([i,u,d])

        edges_csv=out/f"{label}_edges.csv"
        with edges_csv.open("w",newline="") as f:
            w=csv.writer(f)
            w.writerow(["cell","upNodeId","downNodeId"])
            w.writerows(edge_rows)

        if g1.GetNumberOfCells()!=g0.GetNumberOfPoints()-1:
            raise RuntimeError(f"{label}: expected tree edge count n-1")

        summary[label]={
          "port0":str(p0),"port1":str(p1),
          "nodes":int(g0.GetNumberOfPoints()),
          "edges":int(g1.GetNumberOfCells()),
          "nodes_csv_sha256":sha256(nodes_csv),
          "edges_csv_sha256":sha256(edges_csv),
          "port0_sha256":sha256(p0),
          "port1_sha256":sha256(p1),
        }

    (out/"ttk_tree_extract_summary.json").write_text(json.dumps(summary,indent=2))
    print("===== PHASE 3D TTK TREE EXTRACTION PASS =====")
    for k,v in summary.items():
        print(k, "nodes=",v["nodes"], "edges=",v["edges"])
    print("summary:", out/"ttk_tree_extract_summary.json")

if __name__=="__main__":
    main()
