#!/usr/bin/env python3
from pathlib import Path
from collections import Counter
import argparse, csv, hashlib, importlib.util, json, math
import numpy as np
import gudhi
from gudhi.wasserstein import wasserstein_distance

ABS_TOL=1e-10
REL_TOL=1e-12
FROZEN={
 "CNN":{"dB":3.06704616546631,"W2inf":14.57226729106,"W22":18.7224189780594},
 "UV":{"dB":2.43620783090591,"W2inf":15.2562312172225,"W22":19.9125812541732},
 "F1":{"dB":1.23333263397217,"W2inf":9.98731279438054,"W22":12.3229624979698},
}

def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for c in iter(lambda:f.read(1<<20),b""): h.update(c)
    return h.hexdigest()

def load_parser(path):
    spec=importlib.util.spec_from_file_location("canonical_pd_frozen",path)
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod

def arr(x):
    a=np.asarray(x,dtype=float)
    return np.empty((0,2)) if a.size==0 else a.reshape(-1,2)

def db(A,B):
    A,B=arr(A),arr(B)
    return 0.0 if len(A)==len(B)==0 else float(gudhi.bottleneck_distance(A,B,e=0.0))

def w2(A,B,p):
    A,B=arr(A),arr(B)
    if len(A)==len(B)==0: return 0.0
    return float(wasserstein_distance(A,B,matching=False,order=2.0,internal_p=p,keep_essential_parts=False))

def metrics(G,S):
    db0,db1=db(G[0],S[0]),db(G[1],S[1])
    wi0,wi1=w2(G[0],S[0],np.inf),w2(G[1],S[1],np.inf)
    w20,w21=w2(G[0],S[0],2.0),w2(G[1],S[1],2.0)
    return {"dB":max(db0,db1),"W2inf":math.hypot(wi0,wi1),"W22":math.hypot(w20,w21),
            "dB_D0":db0,"dB_D1":db1,"W2inf_D0":wi0,"W2inf_D1":wi1,"W22_D0":w20,"W22_D1":w21}

def key(pair):
    b,d=map(float,pair); return (b.hex(),d.hex())

def counter(diag,dim):
    return Counter(key(x) for x in diag[dim])

def compare(A,B):
    out={}
    for dim in (0,1):
        a,b=counter(A,dim),counter(B,dim)
        common=a&b; ao=a-b; bo=b-a
        out[f"D{dim}"]={
            "new":sum(a.values()),"historical":sum(b.values()),"common":sum(common.values()),
            "new_only":sum(ao.values()),"historical_only":sum(bo.values()),
            "new_only_examples":[[k[0],k[1],v] for k,v in list(ao.items())[:10]],
            "historical_only_examples":[[k[0],k[1],v] for k,v in list(bo.items())[:10]],
        }
    return out

def close(a,b): return math.isclose(a,b,rel_tol=REL_TOL,abs_tol=ABS_TOL)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--phire",default=str(Path.home()/"PhIRE"))
    ap.add_argument("--audit",default=str(Path.home()/"phire_runtime_audit_20260809_221548"))
    ap.add_argument("--phase5",default=str(Path.home()/"PhIRE/spatial_pd/phase5a_sample69_canonical_pd"))
    ap.add_argument("--out",default=str(Path.home()/"PhIRE/spatial_pd/phase5a_sample69_pd_scalar_audit"))
    args=ap.parse_args()
    root=Path(args.phire).expanduser().resolve()
    audit=Path(args.audit).expanduser().resolve()
    phase5=Path(args.phase5).expanduser().resolve()
    out=Path(args.out).expanduser().resolve(); out.mkdir(parents=True,exist_ok=True)
    parser_path=audit/"recompute_pd/canonical_pd_pilot.py"
    canonical=load_parser(parser_path)

    P5=phase5/"pd"
    paths={
      "canonical":{
        "GT":P5/"phase5_GT_s69_speed_p160_x0_y0_pd_port_0.vtu",
        "CNN":P5/"phase5_CNN_SR_s69_speed_p160_x0_y0_pd_port_0.vtu",
        "UV":P5/"phase5_UV_SR_s69_speed_p160_x0_y0_pd_port_0.vtu",
        "F1":P5/"phase5_F1_SR_s69_speed_p160_x0_y0_pd_port_0.vtu"},
      "historical":{
        "CNN_GT":root/"ttk_runs_fixed/cnn/pd/cnn_GT_s69_speed_p160_x0_y0_pd_port_0.vtu",
        "CNN":root/"ttk_runs_fixed/cnn/pd/cnn_SR_s69_speed_p160_x0_y0_pd_port_0.vtu",
        "UV_GT":root/"ttk_runs_fixed/topology_finetuning/candidateUV_expanded2688_topology/pd/GT/candidateUV_expanded2688_GT_s69_speed_p160_x0_y0_pd_port_0.vtu",
        "UV":root/"ttk_runs_fixed/topology_finetuning/candidateUV_expanded2688_topology/pd/SR/candidateUV_expanded2688_SR_s69_speed_p160_x0_y0_pd_port_0.vtu",
        "F1_GT":root/"ttk_runs_fixed/topology_finetuning/candidateF_grad_E2_low_expanded2688_topology/pd/GT/candidateF_grad_E2_low_expanded2688_GT_s69_speed_p160_x0_y0_pd_port_0.vtu",
        "F1":root/"ttk_runs_fixed/topology_finetuning/candidateF_grad_E2_low_expanded2688_topology/pd/SR/candidateF_grad_E2_low_expanded2688_SR_s69_speed_p160_x0_y0_pd_port_0.vtu"}}

    D={}; meta={}
    for group,gpaths in paths.items():
        D[group]={}; meta[group]={}
        for name,p in gpaths.items():
            if not p.exists(): raise FileNotFoundError(p)
            diag,nf=canonical.read_pd(p)
            D[group][name]=diag
            meta[group][name]={"path":str(p),"sha256":sha256(p),"D0":len(diag[0]),"D1":len(diag[1]),"nonfinite":len(nf)}

    scalar_cmp={
      "GT_vs_CNN_GT":compare(D["canonical"]["GT"],D["historical"]["CNN_GT"]),
      "GT_vs_UV_GT":compare(D["canonical"]["GT"],D["historical"]["UV_GT"]),
      "GT_vs_F1_GT":compare(D["canonical"]["GT"],D["historical"]["F1_GT"]),
      "CNN_SR":compare(D["canonical"]["CNN"],D["historical"]["CNN"]),
      "UV_SR":compare(D["canonical"]["UV"],D["historical"]["UV"]),
      "F1_SR":compare(D["canonical"]["F1"],D["historical"]["F1"]),
    }

    rows=[]; report_metrics={}
    for m in ("CNN","UV","F1"):
        H=metrics(D["historical"][f"{m}_GT"],D["historical"][m])
        C=metrics(D["canonical"]["GT"],D["canonical"][m])
        checks={k:close(H[k],FROZEN[m][k]) for k in ("dB","W2inf","W22")}
        report_metrics[m]={"historical":H,"canonical":C,"frozen":FROZEN[m],"historical_matches_frozen":checks,
                           "canonical_minus_historical":{k:C[k]-H[k] for k in ("dB","W2inf","W22")},
                           "canonical_relative_change_percent":{k:100*(C[k]-H[k])/H[k] for k in ("dB","W2inf","W22")}}
        for k in ("dB","W2inf","W22"):
            rows.append({"method":m,"metric":k,"historical":H[k],"frozen":FROZEN[m][k],
                         "historical_minus_frozen":H[k]-FROZEN[m][k],"historical_matches_frozen":checks[k],
                         "canonical":C[k],"canonical_minus_historical":C[k]-H[k],
                         "canonical_relative_change_percent":100*(C[k]-H[k])/H[k]})

    if not all(v for m in report_metrics.values() for v in m["historical_matches_frozen"].values()):
        raise RuntimeError("Historical recomputation failed frozen-reference gate.")

    report={"sample":69,"gudhi_version":gudhi.__version__,
            "canonical_parser":{"path":str(parser_path),"sha256":sha256(parser_path)},
            "files":meta,"scalar_pair_multiset_comparisons":scalar_cmp,"metrics":report_metrics}
    (out/"phase5a_step3c_pd_scalar_topology_audit.json").write_text(json.dumps(report,indent=2))

    with (out/"phase5a_step3c_metric_comparison.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)

    print("===== PHASE 5A STEP 3C — CANONICAL VS HISTORICAL PD AUDIT =====")
    print("GUDHI:",gudhi.__version__)
    print("\n===== SCALAR-PAIR MULTISET COMPARISONS =====")
    for label,x in scalar_cmp.items():
        print("\n"+label)
        for dim in ("D0","D1"):
            y=x[dim]
            print(f"  {dim}: new={y['new']} historical={y['historical']} common={y['common']} new_only={y['new_only']} historical_only={y['historical_only']}")
            if y["new_only"] or y["historical_only"]:
                print("    new-only examples:",y["new_only_examples"])
                print("    historical-only examples:",y["historical_only_examples"])

    print("\n===== DISTANCE RECOMPUTATION =====")
    for m in ("CNN","UV","F1"):
        print("\n"+m)
        R=report_metrics[m]
        for k in ("dB","W2inf","W22"):
            print(f"  {k:6s} historical={R['historical'][k]:.15g} frozen={R['frozen'][k]:.15g} frozen_match={R['historical_matches_frozen'][k]} canonical={R['canonical'][k]:.15g} delta={R['canonical_minus_historical'][k]:+.15g} ({R['canonical_relative_change_percent'][k]:+.6f}%)")
    print("\nHISTORICAL FROZEN-REFERENCE GATE: PASS")
    print("Wrote:",out)

if __name__=="__main__":
    main()
