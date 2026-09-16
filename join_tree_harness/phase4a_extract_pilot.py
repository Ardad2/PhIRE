#!/usr/bin/env python3
"""
Phase 4A step 1: extract compact exact-TTK-input bundles for a predeclared
multi-sample Join-Tree construction-parity pilot.

Default pilot samples:
  0, 24, 48, 69, 96, 120, 144, 167

For each method (CNN, UV, F1), each sample, and GT/SR:
  - exact float32 wind_speed from MT port_2 VTI
  - TTK port_0 NodeId/VertexId/Scalar/CriticalType
  - TTK port_1 upNodeId/downNodeId arcs

No scientific parity claim is made in this extraction step.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

DEFAULT_SAMPLES=[0,24,48,69,96,120,144,167]

def sha256(path: Path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1<<20), b""):
            h.update(b)
    return h.hexdigest()

def read_vtu(path: Path):
    r=vtk.vtkXMLUnstructuredGridReader()
    r.SetFileName(str(path)); r.Update()
    return r.GetOutput()

def read_vti(path: Path):
    r=vtk.vtkXMLImageDataReader()
    r.SetFileName(str(path)); r.Update()
    return r.GetOutput()

def paths(phire: Path, method: str, kind: str, s: int):
    if method=="CNN":
        base=phire/"ttk_runs_fixed/cnn/mt"
        stem=f"cnn_{kind}_s{s}_speed_p160_x0_y0_mt"
    elif method=="UV":
        base=phire/"ttk_runs_fixed/topology_finetuning/candidateUV_expanded2688_topology/mt"/kind
        stem=f"candidateUV_expanded2688_{kind}_s{s}_speed_p160_x0_y0_mt"
    elif method=="F1":
        base=phire/"ttk_runs_fixed/topology_finetuning/candidateF_grad_E2_low_expanded2688_topology/mt"/kind
        stem=f"candidateF_grad_E2_low_expanded2688_{kind}_s{s}_speed_p160_x0_y0_mt"
    else:
        raise ValueError(method)
    return (
      base/f"{stem}_port_0.vtu",
      base/f"{stem}_port_1.vtu",
      base/f"{stem}_port_2.vti",
    )

def extract_one(p0,p1,p2):
    for p in (p0,p1,p2):
        if not p.exists(): raise FileNotFoundError(p)

    g0=read_vtu(p0); pd0=g0.GetPointData()
    names=["NodeId","VertexId","Scalar","CriticalType"]
    aa={}
    for n in names:
        a=pd0.GetArray(n)
        if a is None: raise RuntimeError(f"{p0}: missing {n}")
        aa[n]=a
    node_id=np.array([int(round(aa["NodeId"].GetTuple1(i))) for i in range(g0.GetNumberOfPoints())],dtype=np.int64)
    vertex_id=np.array([int(round(aa["VertexId"].GetTuple1(i))) for i in range(g0.GetNumberOfPoints())],dtype=np.int64)
    scalar=np.array([float(aa["Scalar"].GetTuple1(i)) for i in range(g0.GetNumberOfPoints())],dtype=np.float64)
    ctype=np.array([int(round(aa["CriticalType"].GetTuple1(i))) for i in range(g0.GetNumberOfPoints())],dtype=np.int16)

    g1=read_vtu(p1); cd=g1.GetCellData()
    up=cd.GetArray("upNodeId"); down=cd.GetArray("downNodeId")
    if up is None or down is None: raise RuntimeError(f"{p1}: missing arc arrays")
    up_id=np.array([int(round(up.GetTuple1(i))) for i in range(g1.GetNumberOfCells())],dtype=np.int64)
    down_id=np.array([int(round(down.GetTuple1(i))) for i in range(g1.GetNumberOfCells())],dtype=np.int64)

    img=read_vti(p2)
    dims=img.GetDimensions()
    arr=img.GetPointData().GetArray("wind_speed")
    if arr is None: raise RuntimeError(f"{p2}: wind_speed missing")
    field=np.asarray(vtk_to_numpy(arr)).reshape(-1)
    if tuple(dims)!=(160,160,1) or field.size!=25600:
        raise RuntimeError(f"{p2}: unexpected dims={dims} n={field.size}")
    if field.dtype != np.float32:
        raise RuntimeError(f"{p2}: expected float32, got {field.dtype}")

    if len(node_id)-1 != len(up_id):
        raise RuntimeError(f"{p0}: expected n-1 arcs")
    if len(set(node_id.tolist())) != len(node_id):
        raise RuntimeError(f"{p0}: duplicate NodeId")

    return dict(
      field=field,
      node_id=node_id,
      vertex_id=vertex_id,
      ttk_scalar=scalar,
      critical_type=ctype,
      up_id=up_id,
      down_id=down_id,
    )

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--phire",default=str(Path.home()/"PhIRE"))
    ap.add_argument("--out",required=True)
    ap.add_argument("--samples",default=",".join(map(str,DEFAULT_SAMPLES)))
    args=ap.parse_args()

    phire=Path(args.phire).expanduser().resolve()
    out=Path(args.out).expanduser().resolve(); out.mkdir(parents=True,exist_ok=True)
    samples=[int(x) for x in args.samples.split(",") if x.strip()]

    manifest={}
    for s in samples:
        for method in ("CNN","UV","F1"):
            for kind in ("GT","SR"):
                label=f"{method}_{kind}_s{s}"
                p0,p1,p2=paths(phire,method,kind,s)
                d=extract_one(p0,p1,p2)
                npz=out/f"{label}.npz"
                np.savez_compressed(npz,**d)
                manifest[label]={
                  "sample":s,"method":method,"kind":kind,
                  "port0":str(p0),"port1":str(p1),"port2":str(p2),
                  "port0_sha256":sha256(p0),
                  "port1_sha256":sha256(p1),
                  "port2_sha256":sha256(p2),
                  "npz":str(npz),
                  "npz_sha256":sha256(npz),
                  "ttk_nodes":int(len(d["node_id"])),
                  "ttk_edges":int(len(d["up_id"])),
                }

    (out/"pilot_extract_manifest.json").write_text(json.dumps(manifest,indent=2))
    print("===== PHASE 4A PILOT EXTRACTION PASS =====")
    print("samples:",samples)
    print("trees:",len(manifest))
    for k,v in manifest.items():
        print(k,"nodes=",v["ttk_nodes"],"edges=",v["ttk_edges"])

if __name__=="__main__":
    main()
