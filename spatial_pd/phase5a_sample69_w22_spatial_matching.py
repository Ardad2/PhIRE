#!/usr/bin/env python3
"""
Phase 5A Step 4A/4B — W2,2 matching + spatial provenance for sample 69.

Run inside the validated gudhi-audit environment.

Primary matching:
    Wasserstein order q=2, internal_p=2  (W2,2)

Important conventions:
- finite PairType 0/1 only
- D0 and D1 matched separately
- exact zero-persistence pairs are preserved in counts/provenance but excluded
  from the spatial correspondence input because they lie exactly on the
  diagonal and have zero-cost / potentially non-unique matches
- real-real matches receive birth/death spatial displacement
- GT->diagonal and diagonal->SR matches are classified but have no spatial
  counterpart distance
"""

from pathlib import Path
import argparse, csv, hashlib, json, math
import numpy as np
import vtk
import gudhi
from gudhi.wasserstein import wasserstein_distance

GRID_N = 160
GRID_DIAG = math.hypot(GRID_N - 1, GRID_N - 1)
TOL = 1e-10

def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for c in iter(lambda:f.read(1<<20),b""): h.update(c)
    return h.hexdigest()

def read_features(path):
    r=vtk.vtkXMLUnstructuredGridReader()
    r.SetFileName(str(path)); r.Update()
    g=r.GetOutput()
    cd=g.GetCellData(); pd=g.GetPointData()

    req_c=["PairIdentifier","PairType","Persistence","Birth","IsFinite"]
    req_p=["ttkVertexScalarField","CriticalType","Coordinates"]
    for n in req_c:
        if cd.GetArray(n) is None: raise RuntimeError(f"{path}: missing cell {n}")
    for n in req_p:
        if pd.GetArray(n) is None: raise RuntimeError(f"{path}: missing point {n}")

    pid=cd.GetArray("PairIdentifier")
    ptype=cd.GetArray("PairType")
    pers=cd.GetArray("Persistence")
    birth=cd.GetArray("Birth")
    finite=cd.GetArray("IsFinite")
    vid=pd.GetArray("ttkVertexScalarField")
    crit=pd.GetArray("CriticalType")
    coord=pd.GetArray("Coordinates")

    out={0:[],1:[]}
    zero={0:[],1:[]}

    for ci in range(g.GetNumberOfCells()):
        typ=int(round(ptype.GetTuple1(ci)))
        fin=int(round(finite.GetTuple1(ci)))
        if fin!=1 or typ not in (0,1): continue

        cell=g.GetCell(ci)
        if cell.GetNumberOfPoints()!=2:
            raise RuntimeError(f"{path}: cell {ci} not 2-point")
        p0=int(cell.GetPointId(0)); p1=int(cell.GetPointId(1))

        b=float(birth.GetTuple1(ci))
        p=float(pers.GetTuple1(ci))
        d=b+p

        c0=coord.GetTuple(p0); c1=coord.GetTuple(p1)
        f={
            "source_path":str(path),
            "cell_index":ci,
            "pair_identifier":int(round(pid.GetTuple1(ci))),
            "pair_type":typ,
            "birth":b,
            "death":d,
            "persistence":p,
            "birth_vertex_id":int(round(vid.GetTuple1(p0))),
            "death_vertex_id":int(round(vid.GetTuple1(p1))),
            "birth_critical_type":int(round(crit.GetTuple1(p0))),
            "death_critical_type":int(round(crit.GetTuple1(p1))),
            "birth_x":float(c0[0]),"birth_y":float(c0[1]),
            "death_x":float(c1[0]),"death_y":float(c1[1]),
        }
        if abs(p) <= TOL:
            zero[typ].append(f)
        elif p > TOL:
            out[typ].append(f)
        else:
            raise RuntimeError(f"{path}: negative persistence {p}")

    return out,zero

def points(features):
    if not features: return np.empty((0,2),dtype=float)
    return np.asarray([[f["birth"],f["death"]] for f in features],dtype=float)

def api_selftest():
    cases=[
        ("real-real", np.array([[0.,2.]]), np.array([[0.1,2.1]]), "rr"),
        ("gt-diag", np.array([[0.,2.]]), np.empty((0,2)), "gd"),
        ("diag-sr", np.empty((0,2)), np.array([[0.,2.]]), "ds"),
    ]
    results=[]
    for name,A,B,kind in cases:
        dist,matching=wasserstein_distance(
            A,B,matching=True,order=2.0,internal_p=2.0,
            keep_essential_parts=False
        )
        M=np.asarray(matching,dtype=int).reshape(-1,2)
        results.append((name,float(dist),M.tolist()))
        if kind=="rr" and not any(i==0 and j==0 for i,j in M):
            raise RuntimeError(f"Unexpected GUDHI real-real matching convention: {M}")
        if kind=="gd" and not any(i==0 and j==-1 for i,j in M):
            raise RuntimeError(f"Unexpected GUDHI GT-diagonal convention: {M}")
        if kind=="ds" and not any(i==-1 and j==0 for i,j in M):
            raise RuntimeError(f"Unexpected GUDHI diagonal-SR convention: {M}")
    return results

def spatial_dist(a,b,prefix):
    dx=a[f"{prefix}_x"]-b[f"{prefix}_x"]
    dy=a[f"{prefix}_y"]-b[f"{prefix}_y"]
    return math.hypot(dx,dy)

def match_dimension(gt,sr,method,dim):
    A=points(gt); B=points(sr)
    dist,M=wasserstein_distance(
        A,B,matching=True,order=2.0,internal_p=2.0,
        keep_essential_parts=False
    )
    M=np.asarray(M,dtype=int).reshape(-1,2)

    # independent distance-only consistency check
    d2=float(wasserstein_distance(
        A,B,matching=False,order=2.0,internal_p=2.0,
        keep_essential_parts=False
    ))
    if not math.isclose(float(dist),d2,rel_tol=1e-12,abs_tol=1e-10):
        raise RuntimeError((method,dim,dist,d2))

    rows=[]
    for rank,(i,j) in enumerate(M):
        base={
            "method":method,"dimension":dim,"matching_row":rank,
            "gt_index":int(i),"sr_index":int(j)
        }
        if i>=0 and j>=0:
            g=gt[i]; s=sr[j]
            bd=spatial_dist(g,s,"birth")
            dd=spatial_dist(g,s,"death")
            rec={**base,"match_type":"real_real",
                 "gt_birth":g["birth"],"gt_death":g["death"],"gt_persistence":g["persistence"],
                 "sr_birth":s["birth"],"sr_death":s["death"],"sr_persistence":s["persistence"],
                 "persistence_plane_L2":math.hypot(g["birth"]-s["birth"],g["death"]-s["death"]),
                 "persistence_plane_Linf":max(abs(g["birth"]-s["birth"]),abs(g["death"]-s["death"])),
                 "birth_gt_x":g["birth_x"],"birth_gt_y":g["birth_y"],
                 "birth_sr_x":s["birth_x"],"birth_sr_y":s["birth_y"],
                 "birth_displacement_px":bd,"birth_displacement_norm":bd/GRID_DIAG,
                 "death_gt_x":g["death_x"],"death_gt_y":g["death_y"],
                 "death_sr_x":s["death_x"],"death_sr_y":s["death_y"],
                 "death_displacement_px":dd,"death_displacement_norm":dd/GRID_DIAG,
                 "gt_pair_identifier":g["pair_identifier"],"sr_pair_identifier":s["pair_identifier"],
                 "gt_birth_vertex_id":g["birth_vertex_id"],"gt_death_vertex_id":g["death_vertex_id"],
                 "sr_birth_vertex_id":s["birth_vertex_id"],"sr_death_vertex_id":s["death_vertex_id"],
                 }
        elif i>=0 and j==-1:
            g=gt[i]
            rec={**base,"match_type":"gt_to_diagonal",
                 "gt_birth":g["birth"],"gt_death":g["death"],"gt_persistence":g["persistence"],
                 "sr_birth":"","sr_death":"","sr_persistence":"",
                 "persistence_plane_L2":g["persistence"]/math.sqrt(2),
                 "persistence_plane_Linf":g["persistence"]/2,
                 "birth_gt_x":g["birth_x"],"birth_gt_y":g["birth_y"],
                 "birth_sr_x":"","birth_sr_y":"","birth_displacement_px":"","birth_displacement_norm":"",
                 "death_gt_x":g["death_x"],"death_gt_y":g["death_y"],
                 "death_sr_x":"","death_sr_y":"","death_displacement_px":"","death_displacement_norm":"",
                 "gt_pair_identifier":g["pair_identifier"],"sr_pair_identifier":"",
                 "gt_birth_vertex_id":g["birth_vertex_id"],"gt_death_vertex_id":g["death_vertex_id"],
                 "sr_birth_vertex_id":"","sr_death_vertex_id":""}
        elif i==-1 and j>=0:
            s=sr[j]
            rec={**base,"match_type":"diagonal_to_sr",
                 "gt_birth":"","gt_death":"","gt_persistence":"",
                 "sr_birth":s["birth"],"sr_death":s["death"],"sr_persistence":s["persistence"],
                 "persistence_plane_L2":s["persistence"]/math.sqrt(2),
                 "persistence_plane_Linf":s["persistence"]/2,
                 "birth_gt_x":"","birth_gt_y":"",
                 "birth_sr_x":s["birth_x"],"birth_sr_y":s["birth_y"],
                 "birth_displacement_px":"","birth_displacement_norm":"",
                 "death_gt_x":"","death_gt_y":"",
                 "death_sr_x":s["death_x"],"death_sr_y":s["death_y"],
                 "death_displacement_px":"","death_displacement_norm":"",
                 "gt_pair_identifier":"","sr_pair_identifier":s["pair_identifier"],
                 "gt_birth_vertex_id":"","gt_death_vertex_id":"",
                 "sr_birth_vertex_id":s["birth_vertex_id"],"sr_death_vertex_id":s["death_vertex_id"]}
        else:
            raise RuntimeError(f"Invalid matching row {(i,j)}")
        rows.append(rec)

    return float(dist),rows

def summary(rows):
    rr=[r for r in rows if r["match_type"]=="real_real"]
    gd=[r for r in rows if r["match_type"]=="gt_to_diagonal"]
    ds=[r for r in rows if r["match_type"]=="diagonal_to_sr"]
    out={"real_real":len(rr),"gt_to_diagonal":len(gd),"diagonal_to_sr":len(ds)}
    for key in ("birth_displacement_px","death_displacement_px"):
        vals=np.asarray([float(r[key]) for r in rr],dtype=float)
        out[key]={
            "count":int(len(vals)),
            "mean":float(vals.mean()) if len(vals) else None,
            "median":float(np.median(vals)) if len(vals) else None,
            "p90":float(np.quantile(vals,.9)) if len(vals) else None,
            "max":float(vals.max()) if len(vals) else None,
        }
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--phase5",default=str(Path.home()/"PhIRE/spatial_pd/phase5a_sample69_canonical_pd"))
    ap.add_argument("--out",default=str(Path.home()/"PhIRE/spatial_pd/phase5a_sample69_w22_spatial_matching"))
    args=ap.parse_args()
    phase5=Path(args.phase5).expanduser().resolve()
    out=Path(args.out).expanduser().resolve(); out.mkdir(parents=True,exist_ok=True)
    pdroot=phase5/"pd"

    paths={
      "GT":pdroot/"phase5_GT_s69_speed_p160_x0_y0_pd_port_0.vtu",
      "CNN":pdroot/"phase5_CNN_SR_s69_speed_p160_x0_y0_pd_port_0.vtu",
      "UV":pdroot/"phase5_UV_SR_s69_speed_p160_x0_y0_pd_port_0.vtu",
      "F1":pdroot/"phase5_F1_SR_s69_speed_p160_x0_y0_pd_port_0.vtu"}

    api=api_selftest()
    print("===== GUDHI MATCHING API SELF-TEST =====")
    for x in api: print(x)
    print("MATCHING API SELF-TEST: PASS")

    feats={}; zeros={}
    for k,p in paths.items():
        feats[k],zeros[k]=read_features(p)

    print("\n===== POSITIVE / ZERO-PERSISTENCE COUNTS =====")
    for k in ("GT","CNN","UV","F1"):
        print(k, "D0+",len(feats[k][0]),"D1+",len(feats[k][1]),
              "D0zero",len(zeros[k][0]),"D1zero",len(zeros[k][1]))

    all_rows=[]; report={"sample":69,"gudhi_version":gudhi.__version__,"grid_diagonal":GRID_DIAG,
                         "api_selftest":api,"files":{},"methods":{}}
    for k,p in paths.items():
        report["files"][k]={"path":str(p),"sha256":sha256(p)}

    for method in ("CNN","UV","F1"):
        report["methods"][method]={}
        print("\n=====",method,"=====")
        for dim in (0,1):
            dist,rows=match_dimension(feats["GT"][dim],feats[method][dim],method,dim)
            all_rows.extend(rows)
            s=summary(rows)
            report["methods"][method][f"D{dim}"]={"W22_dimension":dist,"summary":s}
            print(f"D{dim} W22={dist:.15g}")
            print(" ",s)

        d0=report["methods"][method]["D0"]["W22_dimension"]
        d1=report["methods"][method]["D1"]["W22_dimension"]
        report["methods"][method]["W22_all"]=math.hypot(d0,d1)
        print("W22 all =",report["methods"][method]["W22_all"])

    if all_rows:
        csv_path=out/"sample69_w22_spatial_matches.csv"
        with csv_path.open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(all_rows[0].keys()))
            w.writeheader(); w.writerows(all_rows)

    (out/"sample69_w22_spatial_matching_summary.json").write_text(json.dumps(report,indent=2))

    print("\nWrote:",out)

if __name__=="__main__":
    main()
