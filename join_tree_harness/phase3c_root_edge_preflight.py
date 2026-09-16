#!/usr/bin/env python3
"""
Phase 3C preflight:
1) Diagnose the single TTK-only node in CNN/UV/F1 6_anti SR comparisons.
2) Compare it to authoritative field extrema.
3) Inventory exact TTK MT port-1 edge arrays/cell arrays for the same trees.

No hierarchy-parity claim is made by this script.
"""
from __future__ import annotations
import argparse, csv, hashlib, json
from pathlib import Path
import numpy as np
import vtk

def sha256(path: Path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):
            h.update(b)
    return h.hexdigest()

def speed(root: Path, filename: str, sample=69, crop=160):
    a=np.load(root/filename,mmap_mode="r")
    v=np.asarray(a[sample,:crop,:crop,:],dtype=np.float64)
    return np.hypot(v[...,0],v[...,1])

def read_diff(diff_csv: Path):
    rows=[]
    with diff_csv.open(newline="") as f:
        for r in csv.DictReader(f):
            if r["adjacency"]=="6_anti" and r["status"]=="TTK_ONLY":
                rows.append(r)
    return rows

def vtk_inventory(path: Path):
    if not path.exists():
        return {"exists":False,"path":str(path)}
    r=vtk.vtkXMLUnstructuredGridReader()
    r.SetFileName(str(path)); r.Update()
    g=r.GetOutput()
    pd=g.GetPointData(); cd=g.GetCellData()
    pnames=[pd.GetArrayName(i) for i in range(pd.GetNumberOfArrays())]
    cnames=[cd.GetArrayName(i) for i in range(cd.GetNumberOfArrays())]
    out={
        "exists":True,
        "path":str(path),
        "points":int(g.GetNumberOfPoints()),
        "cells":int(g.GetNumberOfCells()),
        "point_arrays":pnames,
        "cell_arrays":cnames,
    }
    # First few cell connectivities.
    cells=[]
    for i in range(min(5,g.GetNumberOfCells())):
        c=g.GetCell(i)
        ids=[int(c.GetPointId(j)) for j in range(c.GetNumberOfPoints())]
        cells.append(ids)
    out["first_cells_point_ids"]=cells

    samples={}
    for name in pnames:
        a=pd.GetArray(name)
        if a is not None:
            samples["point:"+name]=[a.GetTuple1(i) for i in range(min(5,a.GetNumberOfTuples()))]
    for name in cnames:
        a=cd.GetArray(name)
        if a is not None:
            samples["cell:"+name]=[a.GetTuple1(i) for i in range(min(5,a.GetNumberOfTuples()))]
    out["samples"]=samples
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--phire", default=str(Path.home()/"PhIRE"))
    ap.add_argument("--phase3b", required=True)
    ap.add_argument("--out", required=True)
    args=ap.parse_args()

    phire=Path(args.phire).expanduser().resolve()
    p3b=Path(args.phase3b).expanduser().resolve()
    out=Path(args.out).expanduser().resolve()
    out.mkdir(parents=True,exist_ok=True)

    fields={
      "CNN_SR": speed(phire/"data_out_fixed/wind_mrhr_cnn","dataSR.npy"),
      "UV_SR": speed(phire/"data_out/wind_finetune_candidateUV_expanded2688","dataSR.npy"),
      "F1_SR": speed(phire/"data_out/wind_finetune_candidateF_grad_E2_low_expanded2688","dataSR.npy"),
      "GT": speed(phire/"data_out_fixed/wind_mrhr_cnn","dataGT.npy"),
    }

    diff_rows=read_diff(p3b/"compare/vertex_overlap_differences.csv")
    by_label={}
    for r in diff_rows:
        by_label.setdefault(r["ttk_label"],[]).append(r)

    print("===== PHASE 3C-A: TTK-ONLY NODE DIAGNOSIS =====")
    diag=[]
    for label in ("CNN_SR","UV_SR","F1_SR"):
        rr=by_label.get(label,[])
        f=fields[label]
        flat=f.ravel(order="C")
        amin=int(np.argmin(flat)); amax=int(np.argmax(flat))
        print(f"--- {label} ---")
        print("TTK-only rows:", len(rr))
        print("field argmin:", amin, float(flat[amin]))
        print("field argmax:", amax, float(flat[amax]))
        for r in rr:
            v=int(r["corrected_vertex_id"])
            ts=float(r["ttk_scalar"])
            fs=float(r["field_scalar"])
            ct=int(r["ttk_critical_type"])
            print("TTK_ONLY", "vid=",v, "ttk_scalar=",ts,
                  "field_scalar=",fs, "critical_type=",ct,
                  "is_argmin=",v==amin, "is_argmax=",v==amax)
            diag.append({
                "label":label,"vertex_id":v,"ttk_scalar":ts,"field_scalar":fs,
                "critical_type":ct,"field_argmin":amin,"field_argmax":amax,
                "is_argmin":v==amin,"is_argmax":v==amax,
            })

    # Also summarize GT mismatch extrema involvement.
    print()
    print("===== GT 6_anti mismatch extrema membership =====")
    for label in ("CNN_GT","UV_GT","F1_GT"):
        rr=by_label.get(label,[])
        f=fields["GT"]; flat=f.ravel(order="C")
        amin=int(np.argmin(flat)); amax=int(np.argmax(flat))
        vids={int(r["corrected_vertex_id"]) for r in rr}
        print(label, "TTK-only=",len(rr),
              "contains_argmin=",amin in vids,
              "contains_argmax=",amax in vids)

    (out/"ttk_only_root_diagnostic.json").write_text(json.dumps(diag,indent=2))

    print()
    print("===== PHASE 3C-B: TTK PORT-1 EDGE ARTIFACT INVENTORY =====")

    # Exact positive-scalar numerical-tree prefixes.
    node_paths={
      "CNN_GT": phire/"ttk_runs_fixed/cnn/mt/cnn_GT_s69_speed_p160_x0_y0_mt_port_0.vtu",
      "CNN_SR": phire/"ttk_runs_fixed/cnn/mt/cnn_SR_s69_speed_p160_x0_y0_mt_port_0.vtu",
      "UV_GT": phire/"ttk_runs_fixed/topology_finetuning/candidateUV_expanded2688_topology/mt/GT/candidateUV_expanded2688_GT_s69_speed_p160_x0_y0_mt_port_0.vtu",
      "UV_SR": phire/"ttk_runs_fixed/topology_finetuning/candidateUV_expanded2688_topology/mt/SR/candidateUV_expanded2688_SR_s69_speed_p160_x0_y0_mt_port_0.vtu",
      "F1_GT": phire/"ttk_runs_fixed/topology_finetuning/candidateF_grad_E2_low_expanded2688_topology/mt/GT/candidateF_grad_E2_low_expanded2688_GT_s69_speed_p160_x0_y0_mt_port_0.vtu",
      "F1_SR": phire/"ttk_runs_fixed/topology_finetuning/candidateF_grad_E2_low_expanded2688_topology/mt/SR/candidateF_grad_E2_low_expanded2688_SR_s69_speed_p160_x0_y0_mt_port_0.vtu",
    }

    inventory={}
    for label,p0 in node_paths.items():
        p1=Path(str(p0).replace("_mt_port_0.vtu","_mt_port_1.vtu"))
        inv=vtk_inventory(p1)
        inventory[label]=inv
        print("---",label,"---")
        print(json.dumps(inv,indent=2))

    (out/"port1_inventory.json").write_text(json.dumps(inventory,indent=2))

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
