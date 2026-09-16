#!/usr/bin/env python3
"""
Phase 3E step 1: locate and extract the exact sample-69 VTI scalar arrays that
fed the authoritative numerical TTK Join Trees.

Goal: test whether the remaining GT mismatch is caused by scalar precision /
near-tie differences from recomputing speed in float64.

This script uses host VTK only.
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

def candidates(root: Path, token: str):
    # Prefer exact sample-69/crop naming. Exclude MT/PD output files.
    out=[]
    for p in root.rglob("*.vti"):
        s=p.name.lower()
        if "s69" in s and "speed" in s and "p160" in s and token.lower() in s:
            out.append(p)
    return sorted(out)

def choose(root: Path, token: str):
    cc=candidates(root,token)
    if len(cc)==1:
        return cc[0]
    # Prefer paths containing /vti/ and exact x0_y0.
    preferred=[p for p in cc if "/vti/" in str(p).replace("\\","/").lower()
               and "x0_y0" in p.name.lower()]
    if len(preferred)==1:
        return preferred[0]
    print("CANDIDATES", root, token)
    for p in cc: print(" ",p)
    raise RuntimeError(f"Expected one exact VTI for {root} / {token}, got {len(cc)}")

def extract(path: Path):
    r=vtk.vtkXMLImageDataReader()
    r.SetFileName(str(path)); r.Update()
    img=r.GetOutput()
    dims=img.GetDimensions()
    pd=img.GetPointData()
    names=[pd.GetArrayName(i) for i in range(pd.GetNumberOfArrays())]
    arr=pd.GetArray("wind_speed")
    if arr is None:
        raise RuntimeError(f"{path}: wind_speed missing; arrays={names}")
    x=vtk_to_numpy(arr)
    if x.ndim!=1:
        x=np.asarray(x).reshape(-1)
    if dims[0]*dims[1]*max(dims[2],1) != x.size:
        raise RuntimeError(f"{path}: dimension/array mismatch dims={dims} n={x.size}")
    return np.asarray(x),dims,names

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--phire",default=str(Path.home()/"PhIRE"))
    ap.add_argument("--out",required=True)
    args=ap.parse_args()
    phire=Path(args.phire).expanduser().resolve()
    out=Path(args.out).expanduser().resolve(); out.mkdir(parents=True,exist_ok=True)

    roots={
      "CNN":phire/"ttk_runs_fixed/cnn",
      "UV":phire/"ttk_runs_fixed/topology_finetuning/candidateUV_expanded2688_topology",
      "F1":phire/"ttk_runs_fixed/topology_finetuning/candidateF_grad_E2_low_expanded2688_topology",
    }

    meta={}
    for method,root in roots.items():
        for kind in ("GT","SR"):
            p=choose(root,kind)
            x,dims,names=extract(p)
            if dims[0]!=160 or dims[1]!=160:
                raise RuntimeError(f"{method}/{kind}: expected 160x160, got {dims}")
            npy=out/f"{method}_{kind}_exact_vti_scalar.npy"
            np.save(npy,x)
            meta[f"{method}_{kind}"]={
              "source":str(p),
              "source_sha256":sha256(p),
              "dims":list(dims),
              "dtype":str(x.dtype),
              "n":int(x.size),
              "point_arrays":names,
              "min":float(x.min()),
              "max":float(x.max()),
              "npy":str(npy),
              "npy_sha256":sha256(npy),
            }

    (out/"exact_vti_scalar_manifest.json").write_text(json.dumps(meta,indent=2))
    print("===== PHASE 3E EXACT VTI SCALAR EXTRACTION PASS =====")
    for k,v in meta.items():
        print(k, "dtype=",v["dtype"], "dims=",v["dims"],
              "min=",v["min"],"max=",v["max"])
        print("  ",v["source"])

if __name__=="__main__":
    main()
