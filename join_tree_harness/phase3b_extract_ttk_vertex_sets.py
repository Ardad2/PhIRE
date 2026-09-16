#!/usr/bin/env python3
"""Extract NodeId/VertexId/Scalar/CriticalType from exact TTK sample-69 node VTUs."""
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

def read_vtu(path: Path):
    r=vtk.vtkXMLUnstructuredGridReader()
    r.SetFileName(str(path))
    r.Update()
    g=r.GetOutput()
    pd=g.GetPointData()
    req=["NodeId","VertexId","Scalar","CriticalType"]
    arr={}
    for name in req:
        a=pd.GetArray(name)
        if a is None:
            raise RuntimeError(f"{path}: missing point array {name}")
        if a.GetNumberOfTuples()!=g.GetNumberOfPoints():
            raise RuntimeError(f"{path}: tuple count mismatch for {name}")
        arr[name]=a
    return g,arr

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
    for label,p in specs.items():
        if not p.exists():
            raise FileNotFoundError(p)
        g,a=read_vtu(p)
        csvp=out/f"{label}_ttk_nodes.csv"
        with csvp.open("w",newline="") as f:
            w=csv.writer(f)
            w.writerow(["row","NodeId","VertexId","Scalar","CriticalType"])
            for i in range(g.GetNumberOfPoints()):
                w.writerow([
                    i,
                    int(round(a["NodeId"].GetTuple1(i))),
                    int(round(a["VertexId"].GetTuple1(i))),
                    float(a["Scalar"].GetTuple1(i)),
                    int(round(a["CriticalType"].GetTuple1(i))),
                ])
        summary[label]={
            "source":str(p),
            "points":int(g.GetNumberOfPoints()),
            "csv":str(csvp),
            "source_sha256":sha256(p),
            "csv_sha256":sha256(csvp),
        }

    (out/"ttk_extract_summary.json").write_text(json.dumps(summary,indent=2))
    print("===== TTK NODE EXTRACTION PASS =====")
    for k,v in summary.items():
        print(k, "points=",v["points"], "source=",v["source"])
    print("summary:", out/"ttk_extract_summary.json")

if __name__=="__main__":
    main()
