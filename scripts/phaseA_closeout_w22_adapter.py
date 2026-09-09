#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, math, sys
from pathlib import Path
import numpy as np

ROOT_DEFAULT = Path.home() / "PhIRE"
AUDIT_DEFAULT = Path.home() / "phire_runtime_audit_20260809_221548"
W22_DEFAULT = AUDIT_DEFAULT / "recompute_pd_w22"
SAMPLE = 69
TOL = 1e-10

RUNS = {
    "cnn": "cnn",
    "uv": "topology_finetuning/candidateUV_expanded2688_topology",
    "f1": "topology_finetuning/candidateF_grad_E2_low_expanded2688_topology",
}

PD_PATHS = {
    "cnn": {
        "GT": "ttk_runs_fixed/cnn/pd/cnn_GT_s69_speed_p160_x0_y0_pd_port_0.vtu",
        "SR": "ttk_runs_fixed/cnn/pd/cnn_SR_s69_speed_p160_x0_y0_pd_port_0.vtu",
    },
    "uv": {
        "GT": "ttk_runs_fixed/topology_finetuning/candidateUV_expanded2688_topology/pd/GT/candidateUV_expanded2688_GT_s69_speed_p160_x0_y0_pd_port_0.vtu",
        "SR": "ttk_runs_fixed/topology_finetuning/candidateUV_expanded2688_topology/pd/SR/candidateUV_expanded2688_SR_s69_speed_p160_x0_y0_pd_port_0.vtu",
    },
    "f1": {
        "GT": "ttk_runs_fixed/topology_finetuning/candidateF_grad_E2_low_expanded2688_topology/pd/GT/candidateF_grad_E2_low_expanded2688_GT_s69_speed_p160_x0_y0_pd_port_0.vtu",
        "SR": "ttk_runs_fixed/topology_finetuning/candidateF_grad_E2_low_expanded2688_topology/pd/SR/candidateF_grad_E2_low_expanded2688_SR_s69_speed_p160_x0_y0_pd_port_0.vtu",
    },
}

def args():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,default=ROOT_DEFAULT)
    p.add_argument("--w22",type=Path,default=W22_DEFAULT)
    p.add_argument("--toolkit-root",type=Path,default=ROOT_DEFAULT/"third_party"/"tda-toolkit-mapper")
    p.add_argument("--out",type=Path,default=None)
    return p.parse_args()

def import_colleague(root):
    src=root/"src"
    if not (src/"tda_toolkit").is_dir():
        raise RuntimeError(f"Missing colleague package: {src/'tda_toolkit'}")
    sys.path.insert(0,str(src))
    import tda_toolkit
    from tda_toolkit import persistence as cp
    if cp.wasserstein_distance is None:
        raise RuntimeError("Colleague persistence module has no GUDHI Wasserstein callable")
    return tda_toolkit,cp

def import_canonical(w22):
    d=w22.parent/"recompute_pd"
    py=d/"canonical_pd_pilot.py"
    if not py.is_file(): raise FileNotFoundError(py)
    sys.path.insert(0,str(d))
    import canonical_pd_pilot as canonical
    return canonical,py

def read_frozen(path):
    with path.open(newline="") as f: rows=list(csv.DictReader(f))
    out={}
    for m,run in RUNS.items():
        hits=[r for r in rows if r["run"]==run and int(r["sample"])==SAMPLE]
        if len(hits)!=1: raise RuntimeError(f"Frozen lookup {m}: {len(hits)} rows")
        out[m]=float(hits[0]["w22_all"])
    return out

def finite(canonical,path):
    diags,_=canonical.read_pd(path)
    def arr(x):
        a=np.asarray(x,dtype=np.float64)
        return np.empty((0,2),dtype=np.float64) if a.size==0 else a.reshape((-1,2))
    return {0:arr(diags[0]),1:arr(diags[1])}

def w22(cp,a,b):
    return float(cp.wasserstein_distance(
        a,b,matching=False,order=2.0,internal_p=2.0,keep_essential_parts=False))

def main():
    a=args(); root=a.root.expanduser().resolve(); w22root=a.w22.expanduser().resolve()
    tk=a.toolkit_root.expanduser().resolve()
    out=(a.out.expanduser().resolve() if a.out else
         w22root/"pd_colleague_compatibility"/"phaseA_w22_closeout")
    out.mkdir(parents=True,exist_ok=True)
    toolkit,cp=import_colleague(tk); canonical,cpy=import_canonical(w22root)
    frozen_csv=w22root/"w22_full_sweep.csv"; frozen=read_frozen(frozen_csv)

    print("PHASE A W22 COLLEAGUE-ADAPTER PARITY CLOSEOUT")
    print("="*96)
    print("python:",sys.executable)
    print("toolkit:",toolkit.__file__)
    print("toolkit version:",getattr(toolkit,"__version__","unknown"))
    print("canonical parser:",cpy)
    print("frozen source:",frozen_csv); print()

    rows=[]; overall=True
    for m in ("cnn","uv","f1"):
        gt=finite(canonical,root/PD_PATHS[m]["GT"])
        sr=finite(canonical,root/PD_PATHS[m]["SR"])
        d0=w22(cp,gt[0],sr[0]); d1=w22(cp,gt[1],sr[1]); total=math.hypot(d0,d1)
        diff=abs(total-frozen[m]); ok=diff<=TOL; overall &= ok
        row={"method":m,"sample":SAMPLE,"GT_D0_count":len(gt[0]),"GT_D1_count":len(gt[1]),
             "SR_D0_count":len(sr[0]),"SR_D1_count":len(sr[1]),"adapter_W22_D0":d0,
             "adapter_W22_D1":d1,"adapter_W22_all":total,"frozen_W22_all":frozen[m],
             "abs_diff":diff,"tolerance":TOL,"pass":int(ok)}
        rows.append(row)
        print(m.upper(),f"adapter={total:.17g} frozen={frozen[m]:.17g} diff={diff:.3e} PASS={int(ok)}")

    csvp=out/"phaseA_colleague_adapter_W22_parity.csv"
    with csvp.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    txt=out/"phaseA_colleague_adapter_W22_parity_summary.txt"
    txt.write_text("\n".join(
        ["PHASE A W22 COLLEAGUE-ADAPTER PARITY CLOSEOUT","="*96]+
        [f"{r['method']}: adapter={r['adapter_W22_all']:.17g} frozen={r['frozen_W22_all']:.17g} abs_diff={r['abs_diff']:.3e} PASS={r['pass']}" for r in rows]+
        ["","OVERALL: "+("PASS" if overall else "FAIL")]
    )+"\n")
    print("CSV:",csvp); print("SUMMARY:",txt)
    if not overall: raise RuntimeError("PHASE A W22 PARITY CLOSEOUT: FAIL")
    print("PHASE A W22 PARITY CLOSEOUT: PASS")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
