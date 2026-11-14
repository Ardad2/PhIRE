#!/usr/bin/env python3
"""Read-only forensic comparison of historical CNN GT and Phase-6 sample-0 GT.

Modes:
  inventory  - source locations, artifact hashes and sizes (stdlib only)
  arrays-vti - verify ordered GT data arrays and VTI scalar fields (Docker/VTK)
  pd        - compare canonical finite D0/D1 diagrams (archived GUDHI environment)
No mode overwrites or writes any project file.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib
import sys
from pathlib import Path

METHOD = "candidateF_grad_pairzero_expanded2688"
STEM_NEW = f"{METHOD}_GT_s0_speed_p160_x0_y0"
STEM_OLD = "cnn_GT_s0_speed_p160_x0_y0"
PILOT_NAME = METHOD + "_s0_extraction_pilot_20261014"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def one(folder: Path, glob_pattern: str) -> Path:
    paths = sorted(x for x in folder.rglob(glob_pattern) if x.is_file())
    require(len(paths) == 1, f"Expected one file {glob_pattern} under {folder}; found {len(paths)}: {paths[:6]}")
    return paths[0].resolve()


def source_paths(repo: Path):
    legacy = repo / "ttk_runs_fixed" / "cnn"
    pilot = repo / "ttk_runs_fixed" / "topology_finetuning" / PILOT_NAME
    require(legacy.is_dir() and pilot.is_dir(), f"Missing historic or pilot folder: {legacy} / {pilot}")
    return {
        "hist_vti": one(legacy, STEM_OLD + ".vti"),
        "new_vti": one(pilot / "vti", STEM_NEW + ".vti"),
        "hist_pd": one(legacy / "pd", STEM_OLD + "_pd_port_0.vtu"),
        "new_pd": one(pilot / "pd", STEM_NEW + "_pd_port_0.vtu"),
        "hist_gt_array": one(repo / "data_out_fixed" / "wind_mrhr_cnn", "dataGT.npy"),
        "hist_idx": one(repo / "data_out_fixed" / "wind_mrhr_cnn", "idx.npy"),
        "new_gt_array": one(repo / "data_out" / f"wind_finetune_{METHOD}", "dataGT.npy"),
        "new_idx": one(repo / "data_out" / f"wind_finetune_{METHOD}", "idx.npy"),
    }


def hash_file(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def inventory(paths):
    for label,p in paths.items():
        print(f"[FILE] {label} bytes={p.stat().st_size} sha256={hash_file(p)} path={p}", flush=True)
    print("[INVENTORY] Found all GT references (no fields or diagrams parsed)", flush=True)


def compare_arrays_vti(paths):
    import numpy as np
    from vtkmodules.vtkIOXML import vtkXMLImageDataReader
    from vtkmodules.util.numpy_support import vtk_to_numpy

    ids={}
    vectors={}
    for tag in ('hist', 'new'):
        ids[tag]=np.load(paths[f'{tag}_idx'], allow_pickle=False)
        vectors[tag]=np.load(paths[f'{tag}_gt_array'], mmap_mode='r', allow_pickle=False)
        require(ids[tag].ndim==1 and len(ids[tag])>=1, f'{tag} invalid index array')
        require(vectors[tag].ndim==4 and vectors[tag].shape[-1]==2, f'{tag} GT array invalid: {vectors[tag].shape}')
        require(int(ids[tag][0])==0, f'{tag} first benchmark index is not 0: {ids[tag][0]}')
        print(f'[ARRAY] {tag} shape={vectors[tag].shape} dtype={vectors[tag].dtype} sample0_idx={ids[tag][0]}',flush=True)
    a=np.asarray(vectors['hist'][0]); b=np.asarray(vectors['new'][0])
    require(a.shape==b.shape, f'GT sample0 shapes mismatch: {a.shape} vs {b.shape}')
    are_equal=bool(np.array_equal(a,b))
    print(f'[ARRAY] sample-0 CNN-vs-gradient GT exact={are_equal} max_abs_diff={float(np.max(np.abs(a.astype(float)-b.astype(float)))):.12g}',flush=True)
    print(f'[ARRAY] entire idx vectors equal={bool(np.array_equal(ids["hist"],ids["new"]))}',flush=True)

    def vti(p):
        reader=vtkXMLImageDataReader()
        reader.SetFileName(str(p))
        reader.Update()
        o=reader.GetOutput()
        require(o is not None and tuple(o.GetDimensions())==(160,160,1), f'Unexpected VTI dimensions: {p}')
        point=o.GetPointData()
        arr=point.GetArray('wind_speed')
        require(arr is not None, f'wind_speed array missing: {p}')
        require(point.GetScalars() is not None and point.GetScalars().GetName()=='wind_speed', f'Active scalar is not wind_speed: {p}')
        numpy_arr=np.asarray(vtk_to_numpy(arr)).reshape((160,160))
        require(np.isfinite(numpy_arr).all(), f'Nonfinite VTI scalars in {p}')
        return numpy_arr

    va=vti(paths['hist_vti']); vb=vti(paths['new_vti'])
    exact=bool(np.array_equal(va,vb))
    print(f'[VTI] CNN GT vs new GT exact={exact} max_abs_diff={float(np.max(np.abs(va.astype(float)-vb.astype(float)))):.12g}',flush=True)
    for label,src,field in [('hist',a,va),('new',b,vb)]:
        u=src[:160,:160,0].astype(np.float32)
        v=src[:160,:160,1].astype(np.float32)
        speed=np.sqrt(u*u+v*v)
        diff=np.abs(field.astype(float)-speed.astype(float))
        print(f'[VTI] {label} versus expected NumPy speed: max_abs_diff={float(np.max(diff)):.12g} allclose_atol_1e-5={bool(np.allclose(field,speed,rtol=0,atol=1e-5))}',flush=True)
    if are_equal and exact:
        print('[GT FIELD] MATCH: source vector sample 0 and read-back VTI speed fields are exactly identical',flush=True)
    else:
        print('[GT FIELD] DISCREPANCY: inspect source/VTI differences before comparing historical and new metrics',flush=True)


def compare_pd(paths, audit_root):
    import numpy as np
    parser=audit_root/'recompute_pd'/'canonical_pd_pilot.py'
    require(parser.is_file(), f'Archived parser unavailable: {parser}')
    sys.path.insert(0,str(parser.parent))
    canonical=importlib.import_module('canonical_pd_pilot')
    h,_=canonical.read_pd(paths['hist_pd'])
    n,_=canonical.read_pd(paths['new_pd'])
    all_equal=True
    for d in (0,1):
        a=np.asarray(h[d],dtype=float)
        b=np.asarray(n[d],dtype=float)
        require(a.ndim==b.ndim==2 and a.shape[1]==b.shape[1]==2, f'Invalid D{d} shapes: {a.shape} / {b.shape}')
        require(np.isfinite(a).all() and np.isfinite(b).all(), f'Nonfinite D{d} pairs')
        sa=a[np.lexsort((a[:,1],a[:,0]))]
        sb=b[np.lexsort((b[:,1],b[:,0]))]
        match=bool(np.array_equal(sa,sb))
        all_equal=all_equal and match
        z_old=int(np.count_nonzero(a[:,1]==a[:,0]))
        z_new=int(np.count_nonzero(b[:,1]==b[:,0]))
        print(f'[PD] D{d} historical={len(a)} new={len(b)} exact_birth_death_multiset={match} zero_persistence={z_old}/{z_new}',flush=True)
        if len(a)==len(b):
            max_diff=float(np.max(np.abs(sa-sb))) if len(a)>0 else 0.0
            print(f'[PD] D{d} lexicographically-sorted coordinate max_abs_diff={max_diff:.12g}',flush=True)
        if not match:
            ca=Counter(map(tuple,a)); cb=Counter(map(tuple,b))
            extra_old=list((ca-cb).items())[:4]
            extra_new=list((cb-ca).items())[:4]
            print(f'[PD] D{d} unmatched historical examples={extra_old}',flush=True)
            print(f'[PD] D{d} unmatched new examples={extra_new}',flush=True)
    print('[GT DIAGRAM] '+ ('MATCH: finite D0/D1 birth-death multisets identical' if all_equal else 'DISCREPANCY: GT finite persistence-pair multisets differ'),flush=True)
    print('[NOTE] The historical CNN regression tests the metric implementation, not GT parity. This diagnostic never runs distance calculations.',flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo-root',type=Path,default=Path.home()/'PhIRE')
    ap.add_argument('--audit-root',type=Path,default=Path.home()/'phire_runtime_audit_20260809_221548')
    ap.add_argument('--mode',choices=('inventory','arrays-vti','pd'),required=True)
    args=ap.parse_args()
    paths=source_paths(args.repo_root.resolve())
    if args.mode=='inventory': inventory(paths)
    elif args.mode=='arrays-vti': compare_arrays_vti(paths)
    else: compare_pd(paths,args.audit_root.resolve())
    return 0

if __name__=='__main__':
    try: raise SystemExit(main())
    except Exception as ex:
        print(f'[PARITY] ERROR {type(ex).__name__}: {ex}',file=sys.stderr)
        raise SystemExit(1)
