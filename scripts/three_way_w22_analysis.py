#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,math
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr,spearmanr

ROOT_DEFAULT=Path.home()/"PhIRE"
W22_DEFAULT=Path.home()/"phire_runtime_audit_20260809_221548"/"recompute_pd_w22"
METHODS=("cnn","uv","f1")
HIST_ID={"cnn":"cnn","uv":"uv","f1":"f1_grad_e2"}
DISPLAY={"cnn":"CNN","uv":"Matched UV control","f1":"Candidate F1"}
EPS=1e-12

def args():
    p=argparse.ArgumentParser(); p.add_argument("--root",type=Path,default=ROOT_DEFAULT); p.add_argument("--w22",type=Path,default=W22_DEFAULT); p.add_argument("--out",type=Path,default=None); return p.parse_args()
def read(path):
    if not path.is_file(): raise FileNotFoundError(path)
    with path.open(newline="") as f:return list(csv.DictReader(f))
def fv(x): return float(x)
def corr(x,y):
    return float(pearsonr(x,y).statistic),float(spearmanr(x,y).statistic)

def main():
    a=args(); root=a.root.expanduser().resolve(); w22=a.w22.expanduser().resolve()
    out=(a.out.expanduser().resolve() if a.out else w22/"pd_colleague_compatibility"/"three_way_w22"); out.mkdir(parents=True,exist_ok=True)
    histp=root/"ttk_runs_fixed"/"unified_candidate_evaluation"/"unified_primary_per_sample_long.csv"
    stdp=w22/"pd_colleague_compatibility"/"all168_phaseB"/"all168_ttk_vs_colleague_cubical_per_sample.csv"
    hist,std=read(histp),read(stdp)

    hl={}
    for m,mid in HIST_ID.items():
        rr=[r for r in hist if r.get("method_id")==mid]
        if len(rr)!=168: raise RuntimeError(f"{m}/{mid}: {len(rr)} historical rows")
        for r in rr:
            s=int(r["sample_idx"]); v=fv(r["pd_distance"])
            if not math.isfinite(v) or v<0: raise RuntimeError(f"bad historical value {m},{s}")
            hl[(s,m)]=v
    sl={}
    for r in std:
        if r["method"] in METHODS: sl[(int(r["sample"]),r["method"])]={"corrected":fv(r["ttk_W22"]),"cubical":fv(r["cubical_W22"])}
    exp={(s,m) for s in range(168) for m in METHODS}
    if set(hl)!=exp or set(sl)!=exp: raise RuntimeError("coverage mismatch")

    joined=[]
    for s in range(168):
        for m in METHODS:
            h=hl[(s,m)]; c=sl[(s,m)]["corrected"]; g=sl[(s,m)]["cubical"]
            joined.append({"sample":s,"method":m,"display_name":DISPLAY[m],"historical_ttk2":h,"corrected_ttkpd_w22":c,"cubical_w22":g,
                           "hist_minus_corrected":h-c,"cubical_minus_corrected":g-c,"hist_minus_cubical":h-g,
                           "hist_rel_vs_corrected_percent":100*(h-c)/c,"cubical_rel_vs_corrected_percent":100*(g-c)/c})
    joinedp=out/"three_way_w22_per_sample.csv"
    with joinedp.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(joined[0].keys())); w.writeheader(); w.writerows(joined)

    by={m:[r for r in joined if r["method"]==m] for m in METHODS}; summaries=[]; lines=["THREE-WAY W22 DISTANCE-BEHAVIOR ANALYSIS","="*104,""]
    for m in METHODS:
        rr=by[m]; h=np.array([fv(r["historical_ttk2"]) for r in rr]); c=np.array([fv(r["corrected_ttkpd_w22"]) for r in rr]); g=np.array([fv(r["cubical_w22"]) for r in rr])
        hp,hs=corr(h,c); gp,gs=corr(g,c); hgp,hgs=corr(h,g)
        rec={"method":m,"historical_mean":float(h.mean()),"corrected_mean":float(c.mean()),"cubical_mean":float(g.mean()),
             "historical_median":float(np.median(h)),"corrected_median":float(np.median(c)),"cubical_median":float(np.median(g)),
             "hist_minus_corrected_mean":float(np.mean(h-c)),"hist_minus_corrected_mean_abs":float(np.mean(np.abs(h-c))),
             "hist_minus_corrected_mean_rel_percent":float(np.mean(100*(h-c)/c)),
             "cubical_minus_corrected_mean":float(np.mean(g-c)),"cubical_minus_corrected_mean_abs":float(np.mean(np.abs(g-c))),
             "cubical_minus_corrected_mean_rel_percent":float(np.mean(100*(g-c)/c)),
             "hist_corrected_pearson":hp,"hist_corrected_spearman":hs,"cubical_corrected_pearson":gp,"cubical_corrected_spearman":gs,
             "hist_cubical_pearson":hgp,"hist_cubical_spearman":hgs}
        summaries.append(rec)
        lines += [DISPLAY[m],"-"*104,
                  f"means: historical={rec['historical_mean']:.8g}, corrected={rec['corrected_mean']:.8g}, cubical={rec['cubical_mean']:.8g}",
                  f"historical-corrected mean bias={rec['hist_minus_corrected_mean']:+.8g}; mean |bias|={rec['hist_minus_corrected_mean_abs']:.8g}; mean relative bias={rec['hist_minus_corrected_mean_rel_percent']:+.3f}%",
                  f"cubical-corrected mean bias={rec['cubical_minus_corrected_mean']:+.8g}; mean |bias|={rec['cubical_minus_corrected_mean_abs']:.8g}; mean relative bias={rec['cubical_minus_corrected_mean_rel_percent']:+.3f}%",
                  f"historical vs corrected: Pearson={hp:+.4f}, Spearman={hs:+.4f}",
                  f"cubical vs corrected: Pearson={gp:+.4f}, Spearman={gs:+.4f}",""]

        for x,y,xlab,ylab,title,name in [
            (c,h,"Standard W2,2 on TTK-derived PD",'Historical TTK "2" PD dissimilarity',f"{DISPLAY[m]}: historical TTK2 vs standard W2,2",f"{m}_scatter_historical_vs_corrected_w22.png"),
            (c,g,"Standard W2,2 on TTK-derived PD","Standard W2,2 on GUDHI cubical PD",f"{DISPLAY[m]}: cross-construction W2,2",f"{m}_scatter_cubical_vs_corrected_w22.png")]:
            fig,ax=plt.subplots(figsize=(7.5,6.5)); ax.scatter(x,y,alpha=.75); lo=min(float(x.min()),float(y.min())); hi=max(float(x.max()),float(y.max())); ax.plot([lo,hi],[lo,hi],linestyle="--"); ax.set_xlabel(xlab); ax.set_ylabel(ylab); ax.set_title(title); fig.savefig(out/name,dpi=220,bbox_inches="tight"); plt.close(fig)

        for a1,a2,xlab,ylab,title,name in [
            (h,c,'Mean of historical TTK "2" and standard W2,2','Historical TTK "2" - standard W2,2',f"{DISPLAY[m]}: historical-vs-standard difference",f"{m}_difference_historical_vs_corrected_w22.png"),
            (g,c,"Mean of cubical and TTK-derived standard W2,2","Cubical W2,2 - TTK-derived W2,2",f"{DISPLAY[m]}: cross-construction difference",f"{m}_difference_cubical_vs_corrected_w22.png")]:
            avg=.5*(a1+a2); diff=a1-a2; md=float(diff.mean()); sd=float(diff.std(ddof=1))
            fig,ax=plt.subplots(figsize=(7.5,6.5)); ax.scatter(avg,diff,alpha=.75); ax.axhline(md); ax.axhline(md+1.96*sd,linestyle="--"); ax.axhline(md-1.96*sd,linestyle="--"); ax.set_xlabel(xlab); ax.set_ylabel(ylab); ax.set_title(title); fig.savefig(out/name,dpi=220,bbox_inches="tight"); plt.close(fig)

    sump=out/"three_way_w22_method_summary.csv"
    with sump.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(summaries[0].keys())); w.writeheader(); w.writerows(summaries)

    effects=[]
    for aa,bb in (("f1","cnn"),("f1","uv"),("uv","cnn")):
        for src,col in (("historical_ttk2","historical_ttk2"),("corrected_w22","corrected_ttkpd_w22"),("cubical_w22","cubical_w22")):
            am={int(r["sample"]):fv(r[col]) for r in by[aa]}; bm={int(r["sample"]):fv(r[col]) for r in by[bb]}
            gains=[(bm[s]-am[s])/bm[s] for s in range(168)]; wins=sum(am[s]<bm[s]-EPS for s in range(168)); ties=sum(abs(am[s]-bm[s])<=EPS for s in range(168))
            effects.append({"comparison":f"{aa}_vs_{bb}","source":src,"mean_relative_gain_percent":100*float(np.mean(gains)),"median_relative_gain_percent":100*float(np.median(gains)),"A_win_count":wins,"ties":ties})
    effp=out/"three_way_w22_pairwise_effects.csv"
    with effp.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(effects[0].keys())); w.writeheader(); w.writerows(effects)

    txt=out/"three_way_w22_summary.txt"; lines += ["OUTPUTS","-"*104,f"joined: {joinedp}",f"method summary: {sump}",f"pairwise effects: {effp}","","Interpretation boundary:",'historical TTK "2" is a related legacy quantity, not standard W2,2.',"Cubical and TTK-derived standard W2,2 use the same explicit metric but different PD constructions."]
    txt.write_text("\n".join(lines)+"\n"); print("\n".join(lines)); print("THREE-WAY W22 ANALYSIS: COMPLETE"); return 0

if __name__=="__main__": raise SystemExit(main())
