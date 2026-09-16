#!/usr/bin/env python3
"""
Phase 4B step 1: extract exact-TTK-input bundles for the full 168-sample
Join-Tree construction-parity sweep.

For every sample 0..167, each method (CNN, UV, F1), and GT/SR:
  - exact float32 wind_speed from MT port_2 VTI
  - TTK port_0 NodeId/VertexId/Scalar/CriticalType
  - TTK port_1 upNodeId/downNodeId arcs

Total expected trees:
  168 x 3 x 2 = 1008
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

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
    aa={}
    for n in ("NodeId","VertexId","Scalar","CriticalType"):
        a=pd0.GetArray(n)
        if a is None: raise RuntimeError(f"{p0}: missing {n}")
        aa[n]=a

    node_id=np.array([int(round(aa["NodeId"].GetTuple1(i)))
                      for i in range(g0.GetNumberOfPoints())],dtype=np.int64)
    vertex_id=np.array([int(round(aa["VertexId"].GetTuple1(i)))
                        for i in range(g0.GetNumberOfPoints())],dtype=np.int64)
    scalar=np.array([float(aa["Scalar"].GetTuple1(i))
                     for i in range(g0.GetNumberOfPoints())],dtype=np.float64)
    ctype=np.array([int(round(aa["CriticalType"].GetTuple1(i)))
                    for i in range(g0.GetNumberOfPoints())],dtype=np.int16)

    g1=read_vtu(p1); cd=g1.GetCellData()
    up=cd.GetArray("upNodeId"); down=cd.GetArray("downNodeId")
    if up is None or down is None:
        raise RuntimeError(f"{p1}: missing upNodeId/downNodeId")
    up_id=np.array([int(round(up.GetTuple1(i)))
                    for i in range(g1.GetNumberOfCells())],dtype=np.int64)
    down_id=np.array([int(round(down.GetTuple1(i)))
                      for i in range(g1.GetNumberOfCells())],dtype=np.int64)

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
    ap.add_argument("--start",type=int,default=0)
    ap.add_argument("--stop",type=int,default=168,
                    help="exclusive stop; default 168")
    args=ap.parse_args()

    if not (0 <= args.start < args.stop <= 168):
        raise ValueError((args.start,args.stop))

    phire=Path(args.phire).expanduser().resolve()
    out=Path(args.out).expanduser().resolve(); out.mkdir(parents=True,exist_ok=True)

    manifest={}
    count=0
    for s in range(args.start,args.stop):
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
                count += 1
        if (s-args.start+1) % 8 == 0 or s==args.stop-1:
            print(f"extracted through sample {s}: {count} trees")

    (out/"full_extract_manifest.json").write_text(json.dumps(manifest,indent=2))
    print("===== PHASE 4B FULL EXTRACTION PASS =====")
    print("sample range:",args.start,args.stop)
    print("trees:",count)

if __name__=="__main__":
    main()
