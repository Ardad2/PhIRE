#!/usr/bin/env python3
"""Read Phase 6 four-cell *summary reports* and print descriptive factorial contrasts.

Run from PhIRE repository root:
  python3 phase6_lxxii_update/scripts/compare_phase6_2x2_reports.py --repo-root .
Python standard library only. Read-only. No PD/MT or spatial-locality claims.
"""
import argparse
from pathlib import Path
import sys

REPORT_PATHS = {
    "00_gradient_only": "docs/topology_finetuning_candidateF_grad_pairzero_expanded2688_eval.md",
    "10_cv_only": "docs/topology_finetuning_candidateF_grad_CV_only_expanded2688_eval.md",
    "01_pers_only": "docs/topology_finetuning_candidateF_grad_PERS_only_expanded2688_eval.md",
    "11_full_f1": "docs/topology_finetuning_candidateF_grad_E2_low_expanded2688_eval.md",
}
METRICS = (
    "PSNRuv (dB)", "Speed MAE (m/s)", "Speed RMSE (m/s)",
    "WPD MAE (m³/s³)", "WPD Wasserstein-1", "WPD bias abs (m³/s³)",
    "Gradient MAE", "Gradient W1", "Gradient kurtosis abs Δ",
    "PSD log-L2", "PSD slope abs Δ",
    "Exceedance abs Δ s>5", "Exceedance abs Δ s>10",
    "Exceedance abs Δ s>15", "Exceedance abs Δ p90",
    "Component-count curve L1",
)

def parse_report(path):
    text = Path(path).read_text(encoding="utf-8")
    if "## Summary metrics table" not in text:
        raise ValueError(f"Report has no Summary metrics table: {path}")
    table = text.split("## Summary metrics table", 1)[1].split("## ",1)[0]
    out = {}
    for line in table.splitlines():
        if not line.startswith("|"):
            continue
        parts = [x.strip() for x in line.strip().strip("|").split("|")]
        if len(parts) != 5:
            continue
        try:
            # Bicubic, CNN, GAN, candidate; only collect finite numeric values.
            vals = list(map(float, parts[1:]))
        except ValueError:
            continue
        out[parts[0]] = vals
    return out

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", type=Path, default=Path("."))
    args = p.parse_args()
    summaries = {k: parse_report(args.repo_root / path) for k,path in REPORT_PATHS.items()}
    print("Source reports (read-only):")
    for k,v in REPORT_PATHS.items(): print(f"  {k}: {v}")
    print()
    print("| Metric | CNN | Grad (00) | CV (10) | PERS (01) | F1 (11) | CV avg | PERS avg | Interaction |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for metric in METRICS:
        if not all(metric in rows for rows in summaries.values()):
            missing=[name for name,rows in summaries.items() if metric not in rows]
            raise ValueError(f"Metric {metric!r} missing from: {missing}")
        baseline=[rows[metric][1] for rows in summaries.values()]
        if max(baseline)-min(baseline)>0.00011:
            raise ValueError(f"CNN baseline mismatch for {metric}: {baseline}")
        g=summaries["00_gradient_only"][metric][3]
        c=summaries["10_cv_only"][metric][3]
        r=summaries["01_pers_only"][metric][3]
        f=summaries["11_full_f1"][metric][3]
        cv=((c-g)+(f-r))/2
        pers=((r-g)+(f-c))/2
        interaction=f-c-r+g
        print(f"| {metric} | {baseline[0]:.4f} | {g:.4f} | {c:.4f} | {r:.4f} | {f:.4f} | {cv:+.4f} | {pers:+.4f} | {interaction:+.4f} |")
    print("\nAll means are rounded in source reports; contrasts are descriptive only. SSIM and new-arm true PD/MT/locality remain unavailable.")
    return 0

if __name__=="__main__":
    sys.exit(main())
