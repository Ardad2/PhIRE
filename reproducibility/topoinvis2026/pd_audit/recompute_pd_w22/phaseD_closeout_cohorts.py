#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv
from pathlib import Path
from statistics import mean

W22_DEFAULT=Path.home()/"phire_runtime_audit_20260809_221548"/"recompute_pd_w22"
METRICS=("dB","W2inf","W22"); EPS=1e-12

def args():
    p=argparse.ArgumentParser(); p.add_argument("--w22",type=Path,default=W22_DEFAULT); p.add_argument("--out",type=Path,default=None); return p.parse_args()
def read(path):
    if not path.is_file(): raise FileNotFoundError(path)
    with path.open(newline="") as f:return list(csv.DictReader(f))
def fv(x): return float(x)
def sample(r):
    for k in ("sample","sample_idx","sample_index"):
        if k in r and str(r[k]).strip()!="": return int(float(r[k]))
    raise KeyError("sample column")

def near_rank(rows):
    for c in ("conventional_rank","conventional_closeness_rank","conventional_order_rank","rank_conventional","rank"):
        if c in rows[0]:
            try:
                pairs=[(int(float(r[c])),sample(r)) for r in rows]
                if len({a for a,_ in pairs})==len(rows): return {s:r for r,s in pairs},c
            except Exception: pass
    sc=next((c for c in ("conventional_closeness_score","closeness_score") if c in rows[0]),None)
    mc=next((c for c in ("conventional_mean_percentile","mean_percentile") if c in rows[0]),None)
    if sc is None or mc is None: raise RuntimeError("Cannot reconstruct frozen conventional order")
    ordered=sorted(rows,key=lambda r:(fv(r[sc]),fv(r[mc]),sample(r)))
    return {sample(r):i+1 for i,r in enumerate(ordered)},f"derived:{sc},{mc},sample"

def gain(cnn,f1): return (cnn-f1)/cnn if cnn!=0 else float("nan")

def summarize(name,samples,lookup):
    ss=sorted(samples); out={"group":name,"n_samples":len(ss)}
    for m in METRICS:
        tw=cw=0; tg=[]; cg=[]
        for s in ss:
            c=lookup[(s,"cnn")]; f=lookup[(s,"f1")]
            tc,tf=fv(c[f"ttk_{m}"]),fv(f[f"ttk_{m}"]); cc,cf=fv(c[f"cubical_{m}"]),fv(f[f"cubical_{m}"])
            tw+=int(tf<tc-EPS); cw+=int(cf<cc-EPS); tg.append(gain(tc,tf)); cg.append(gain(cc,cf))
        out[f"ttk_{m}_F1_win_count"]=tw; out[f"cubical_{m}_F1_win_count"]=cw
        out[f"ttk_{m}_mean_gain"]=mean(tg); out[f"cubical_{m}_mean_gain"]=mean(cg)
    ta=ca=0
    for s in ss:
        c=lookup[(s,"cnn")]; f=lookup[(s,"f1")]
        ta+=int(all(fv(f[f"ttk_{m}"])<fv(c[f"ttk_{m}"])-EPS for m in METRICS))
        ca+=int(all(fv(f[f"cubical_{m}"])<fv(c[f"cubical_{m}"])-EPS for m in METRICS))
    out["ttk_all3_F1_win_count"]=ta; out["cubical_all3_F1_win_count"]=ca
    return out

def main():
    a=args(); w22=a.w22.expanduser().resolve()
    out=(a.out.expanduser().resolve() if a.out else w22/"pd_colleague_compatibility"/"phaseD_closeout"); out.mkdir(parents=True,exist_ok=True)
    backend=w22/"pd_colleague_compatibility"/"all168_phaseB"/"all168_ttk_vs_colleague_cubical_per_sample.csv"
    focal=w22/"corrected_pd_mt"/"corrected_pd_mt_focal_comparisons.csv"
    near=w22/"near_tie_candidateF_grad_E2_vs_cnn_master.csv"
    br,fr,nr=read(backend),read(focal),read(near)
    lookup={(sample(r),r["method"]):r for r in br}
    expected={(s,m) for s in range(168) for m in ("cnn","uv","f1")}
    if set(lookup)!=expected: raise RuntimeError("Backend coverage mismatch")

    f1cnn=[r for r in fr if r.get("candidate_method")=="f1_grad_e2" and r.get("baseline_method")=="cnn"]
    plus={int(r["sample_idx"]) for r in f1cnn if int(float(r["all3_PD_improve"]))==1 and r["MT_state"]=="improve"}
    minus={int(r["sample_idx"]) for r in f1cnn if int(float(r["all3_PD_improve"]))==1 and r["MT_state"]=="worsen"}
    if (len(plus),len(minus))!=(91,59): raise RuntimeError(f"Frozen cohorts changed: {len(plus)},{len(minus)}")

    ranks,rank_src=near_rank(nr)
    tiers={"near_tie_strict10":{s for s,r in ranks.items() if r<=17},
           "near_tie_primary20":{s for s,r in ranks.items() if r<=34},
           "near_tie_broad30":{s for s,r in ranks.items() if r<=51}}
    if [len(tiers[k]) for k in tiers]!=[17,34,51]: raise RuntimeError("Near-tie sizes changed")
    groups={"PDplus_MTplus":plus,"PDplus_MTminus":minus,**tiers,"predeclared_visual_5":{78,71,80,63,69}}
    rows=[summarize(k,v,lookup) for k,v in groups.items()]

    csvp=out/"phaseD_cohort_consistency_summary.csv"
    fields=[]; seen=set()
    for r in rows:
        for k in r:
            if k not in seen: seen.add(k); fields.append(k)
    with csvp.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)

    detail=[]
    for g,ss in groups.items():
        for s in sorted(ss):
            c=lookup[(s,"cnn")]; ff=lookup[(s,"f1")]; r={"group":g,"sample":s}
            for m in METRICS:
                tc,tf=fv(c[f"ttk_{m}"]),fv(ff[f"ttk_{m}"]); cc,cf=fv(c[f"cubical_{m}"]),fv(ff[f"cubical_{m}"])
                r[f"ttk_{m}_F1_win"]=int(tf<tc-EPS); r[f"cubical_{m}_F1_win"]=int(cf<cc-EPS)
                r[f"ttk_{m}_gain"]=gain(tc,tf); r[f"cubical_{m}_gain"]=gain(cc,cf)
            r["ttk_all3_F1_win"]=int(all(r[f"ttk_{m}_F1_win"] for m in METRICS))
            r["cubical_all3_F1_win"]=int(all(r[f"cubical_{m}_F1_win"] for m in METRICS))
            detail.append(r)
    detailp=out/"phaseD_cohort_consistency_per_sample.csv"
    with detailp.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(detail[0].keys())); w.writeheader(); w.writerows(detail)

    lines=["PHASE D CROSS-CONSTRUCTION COHORT CONSISTENCY CLOSEOUT","="*104,f"near-tie rank source: {rank_src}",""]
    for r in rows:
        n=int(r["n_samples"]); lines += [f"{r['group']} n={n}","-"*104]
        for m in METRICS:
            lines.append(f"{m:6s}: TTK F1 wins {r[f'ttk_{m}_F1_win_count']}/{n}; cubical F1 wins {r[f'cubical_{m}_F1_win_count']}/{n}; mean gain TTK={100*r[f'ttk_{m}_mean_gain']:+.2f}% cubical={100*r[f'cubical_{m}_mean_gain']:+.2f}%")
        lines += [f"all3  : TTK {r['ttk_all3_F1_win_count']}/{n}; cubical {r['cubical_all3_F1_win_count']}/{n}",""]
    txt=out/"phaseD_cohort_consistency_summary.txt"; txt.write_text("\n".join(lines)+"\n")
    print("\n".join(lines)); print("SUMMARY CSV:",csvp); print("DETAIL CSV:",detailp); print("SUMMARY TXT:",txt); print("PHASE D COHORT CONSISTENCY CLOSEOUT: COMPLETE")
    return 0

if __name__=="__main__": raise SystemExit(main())
