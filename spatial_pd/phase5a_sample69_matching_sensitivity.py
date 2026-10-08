#!/usr/bin/env python3
"""
Phase 5A Step 4C — matching sensitivity and self-match diagnostic.

Compares GUDHI optimal matchings for:
  - W2,2       : order=2, internal_p=2
  - W2,infinity: order=2, internal_p=inf

Also self-matches canonical GT against itself under both conventions.

Goals:
1. detect any non-identity zero-cost self assignments;
2. quantify how often W2,2 and W2inf choose the same SR partner;
3. inspect the highest-persistence GT features before freezing a spatial
   correspondence convention.

Exact zero-persistence features are excluded, matching Step 4.
"""

from pathlib import Path
import argparse, csv, json, math, hashlib
import numpy as np
import vtk
import gudhi
from gudhi.wasserstein import wasserstein_distance

TOL=1e-10

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
    ptype=cd.GetArray("PairType"); pers=cd.GetArray("Persistence")
    birth=cd.GetArray("Birth"); finite=cd.GetArray("IsFinite")
    pid=cd.GetArray("PairIdentifier")
    vid=pd.GetArray("ttkVertexScalarField")
    coord=pd.GetArray("Coordinates")
    out={0:[],1:[]}
    for ci in range(g.GetNumberOfCells()):
        typ=int(round(ptype.GetTuple1(ci))); fin=int(round(finite.GetTuple1(ci)))
        p=float(pers.GetTuple1(ci))
        if fin!=1 or typ not in (0,1) or p<=TOL: continue
        c=g.GetCell(ci); p0=int(c.GetPointId(0)); p1=int(c.GetPointId(1))
        b=float(birth.GetTuple1(ci)); d=b+p
        c0=coord.GetTuple(p0); c1=coord.GetTuple(p1)
        out[typ].append({
            "index":len(out[typ]),"cell_index":ci,
            "pair_identifier":int(round(pid.GetTuple1(ci))),
            "birth":b,"death":d,"persistence":p,
            "birth_x":float(c0[0]),"birth_y":float(c0[1]),
            "death_x":float(c1[0]),"death_y":float(c1[1]),
            "birth_vid":int(round(vid.GetTuple1(p0))),
            "death_vid":int(round(vid.GetTuple1(p1))),
        })
    return out

def pts(fs):
    return np.asarray([[f["birth"],f["death"]] for f in fs],dtype=float).reshape(-1,2)

def matching(A,B,p):
    d,M=wasserstein_distance(
        pts(A),pts(B),matching=True,order=2.0,internal_p=p,
        keep_essential_parts=False)
    M=np.asarray(M,dtype=int).reshape(-1,2)
    return float(d),M

def map_gt(M):
    # mapping GT index -> SR index, with -1 for diagonal
    return {int(i):int(j) for i,j in M if i>=0}

def disp(a,b,which):
    return math.hypot(a[f"{which}_x"]-b[f"{which}_x"],
                      a[f"{which}_y"]-b[f"{which}_y"])

def self_diag(gt,p):
    d,M=matching(gt,gt,p)
    rows=[]
    nonidentity=0
    for i,j in M:
        if i<0 or j<0:
            rows.append({"i":int(i),"j":int(j),"type":"diagonal"})
            continue
        if i!=j: nonidentity+=1
        rows.append({
            "i":int(i),"j":int(j),"type":"real_real",
            "same_index":bool(i==j),
            "same_scalar":bool(gt[i]["birth"]==gt[j]["birth"] and gt[i]["death"]==gt[j]["death"]),
            "birth_disp":disp(gt[i],gt[j],"birth"),
            "death_disp":disp(gt[i],gt[j],"death"),
        })
    return d,nonidentity,rows

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--phase5",default=str(Path.home()/"PhIRE/spatial_pd/phase5a_sample69_canonical_pd"))
    ap.add_argument("--out",default=str(Path.home()/"PhIRE/spatial_pd/phase5a_sample69_matching_sensitivity"))
    ap.add_argument("--top-k",type=int,default=12)
    args=ap.parse_args()
    phase5=Path(args.phase5).expanduser().resolve()
    out=Path(args.out).expanduser().resolve(); out.mkdir(parents=True,exist_ok=True)
    pdroot=phase5/"pd"
    paths={
      "GT":pdroot/"phase5_GT_s69_speed_p160_x0_y0_pd_port_0.vtu",
      "CNN":pdroot/"phase5_CNN_SR_s69_speed_p160_x0_y0_pd_port_0.vtu",
      "UV":pdroot/"phase5_UV_SR_s69_speed_p160_x0_y0_pd_port_0.vtu",
      "F1":pdroot/"phase5_F1_SR_s69_speed_p160_x0_y0_pd_port_0.vtu"}
    F={k:read_features(p) for k,p in paths.items()}

    report={"sample":69,"gudhi":gudhi.__version__,"files":{k:{"path":str(p),"sha256":sha256(p)} for k,p in paths.items()},
            "self_match":{},"methods":{}}
    detail=[]

    print("===== GT SELF-MATCH =====")
    for dim in (0,1):
        report["self_match"][f"D{dim}"]={}
        for label,p in (("W22",2.0),("W2inf",np.inf)):
            d,nid,rows=self_diag(F["GT"][dim],p)
            maxbd=max([r.get("birth_disp",0) for r in rows],default=0)
            maxdd=max([r.get("death_disp",0) for r in rows],default=0)
            report["self_match"][f"D{dim}"][label]={
                "distance":d,"nonidentity_real_real":nid,
                "max_birth_displacement":maxbd,"max_death_displacement":maxdd,
                "nonidentity_rows":[r for r in rows if r.get("same_index") is False][:20]}
            print(f"D{dim} {label}: distance={d:.15g} nonidentity={nid} max_birth_disp={maxbd:.6g} max_death_disp={maxdd:.6g}")

    for method in ("CNN","UV","F1"):
        print("\n=====",method,"=====")
        report["methods"][method]={}
        for dim in (0,1):
            d22,M22=matching(F["GT"][dim],F[method][dim],2.0)
            di,Mi=matching(F["GT"][dim],F[method][dim],np.inf)
            a=map_gt(M22); b=map_gt(Mi)
            common=set(a)&set(b)
            same=sum(a[i]==b[i] for i in common)
            both_real=[i for i in common if a[i]>=0 and b[i]>=0]
            same_real=sum(a[i]==b[i] for i in both_real)
            rr22=sum(j>=0 for j in a.values()); rri=sum(j>=0 for j in b.values())

            report["methods"][method][f"D{dim}"]={
                "W22_distance":d22,"W2inf_distance":di,
                "gt_features":len(F["GT"][dim]),
                "W22_real_real_gt":rr22,"W2inf_real_real_gt":rri,
                "same_assignment_all_gt":same,
                "same_assignment_all_gt_fraction":same/len(common),
                "real_real_under_both":len(both_real),
                "same_sr_partner_among_both_real":same_real,
                "same_sr_partner_among_both_real_fraction":same_real/len(both_real) if both_real else None}

            print(f"D{dim}: W22={d22:.12g} W2inf={di:.12g}")
            print(f"  same assignment all GT: {same}/{len(common)} = {same/len(common):.4f}")
            print(f"  real-real W22={rr22} W2inf={rri}")
            print(f"  both real, same SR partner: {same_real}/{len(both_real)} = {(same_real/len(both_real) if both_real else float('nan')):.4f}")

            top=sorted(range(len(F["GT"][dim])),key=lambda i:F["GT"][dim][i]["persistence"],reverse=True)[:args.top_k]
            for rank,i in enumerate(top,1):
                g=F["GT"][dim][i]
                j22=a.get(i,-999); ji=b.get(i,-999)
                row={"method":method,"dimension":dim,"rank":rank,"gt_index":i,
                     "gt_birth":g["birth"],"gt_death":g["death"],"gt_persistence":g["persistence"],
                     "gt_birth_x":g["birth_x"],"gt_birth_y":g["birth_y"],
                     "gt_death_x":g["death_x"],"gt_death_y":g["death_y"],
                     "W22_sr_index":j22,"W2inf_sr_index":ji,
                     "same_assignment":bool(j22==ji)}
                for lab,j in (("W22",j22),("W2inf",ji)):
                    if j>=0:
                        s=F[method][dim][j]
                        row[f"{lab}_sr_birth"]=s["birth"]; row[f"{lab}_sr_death"]=s["death"]; row[f"{lab}_sr_persistence"]=s["persistence"]
                        row[f"{lab}_birth_disp"]=disp(g,s,"birth"); row[f"{lab}_death_disp"]=disp(g,s,"death")
                        row[f"{lab}_sr_birth_x"]=s["birth_x"]; row[f"{lab}_sr_birth_y"]=s["birth_y"]
                        row[f"{lab}_sr_death_x"]=s["death_x"]; row[f"{lab}_sr_death_y"]=s["death_y"]
                    else:
                        for suffix in ("sr_birth","sr_death","sr_persistence","birth_disp","death_disp","sr_birth_x","sr_birth_y","sr_death_x","sr_death_y"):
                            row[f"{lab}_{suffix}"]=""
                detail.append(row)

    (out/"sample69_matching_sensitivity_summary.json").write_text(json.dumps(report,indent=2))
    if detail:
        with (out/"sample69_top_persistence_matching_sensitivity.csv").open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=detail[0].keys()); w.writeheader(); w.writerows(detail)

    print("\nWrote:",out)

if __name__=="__main__":
    main()
