#!/usr/bin/env python3
"""Recompute SSIM for the current topology-inspired model on the fixed 168-sample benchmark.

Public method names:
  pretrained_cnn
  reconstruction_only
  topology_inspired

This script is intentionally standalone and does not touch training or topology outputs.
It expects saved physical-unit [u,v] arrays in dataGT.npy/dataSR.npy.

The earlier clean SSIM audit recorded these mean values:
  pretrained CNN:        speed=0.741175, uv_mean=0.771031
  reconstruction-only:   speed=0.813443, uv_mean=0.853551

Because the exact earlier helper script is not present in the supplied audit bundle, this
script computes the standard skimage SSIM convention with per-sample GT dynamic range and
prints the frozen means beside the newly computed values. If CNN/control reproduce the
frozen values to the displayed precision, the convention is confirmed; otherwise stop and
recover the original helper before using the topology-inspired SSIM in the poster.
"""
from pathlib import Path
import argparse
import csv
import numpy as np
from skimage.metrics import structural_similarity

EXPECTED = {
    "pretrained_cnn": (0.741175, 0.771031),
    "reconstruction_only": (0.813443, 0.853551),
}

def ssim2d(gt, sr):
    gt = np.asarray(gt, dtype=np.float64)
    sr = np.asarray(sr, dtype=np.float64)
    dr = float(np.max(gt) - np.min(gt))
    if not np.isfinite(dr) or dr <= 0:
        dr = 1.0
    return float(structural_similarity(gt, sr, data_range=dr))

def load_pair(root: Path):
    gt = np.load(root / "dataGT.npy")
    sr = np.load(root / "dataSR.npy")
    if gt.shape != sr.shape:
        raise ValueError(f"shape mismatch in {root}: GT={gt.shape}, SR={sr.shape}")
    if gt.ndim != 4 or gt.shape[-1] != 2:
        raise ValueError(f"expected [N,H,W,2] arrays in {root}, got {gt.shape}")
    if not (np.isfinite(gt).all() and np.isfinite(sr).all()):
        raise ValueError(f"non-finite values in {root}")
    return gt, sr

def evaluate(name, root):
    gt, sr = load_pair(root)
    rows = []
    for i in range(gt.shape[0]):
        gu, gv = gt[i, ..., 0], gt[i, ..., 1]
        su, sv = sr[i, ..., 0], sr[i, ..., 1]
        gspeed = np.hypot(gu, gv)
        sspeed = np.hypot(su, sv)
        su_ssim = ssim2d(gu, su)
        sv_ssim = ssim2d(gv, sv)
        rows.append({
            "method": name,
            "sample_idx": i,
            "ssim_speed": ssim2d(gspeed, sspeed),
            "ssim_u": su_ssim,
            "ssim_v": sv_ssim,
            "ssim_uv_mean": 0.5 * (su_ssim + sv_ssim),
        })
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=str(Path.home() / "PhIRE"))
    ap.add_argument("--cnn", default="data_out_fixed/wind_mrhr_cnn")
    ap.add_argument("--control", default="data_out/wind_finetune_candidateUV_expanded2688")
    ap.add_argument("--topology", default="data_out/wind_finetune_candidateF_grad_E2_low_expanded2688")
    ap.add_argument("--outdir", default="ttk_runs_fixed/ssim_recomputed_current_topology_inspired")
    args = ap.parse_args()

    repo = Path(args.repo).expanduser().resolve()
    paths = {
        "pretrained_cnn": repo / args.cnn,
        "reconstruction_only": repo / args.control,
        "topology_inspired": repo / args.topology,
    }
    outdir = repo / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for name, path in paths.items():
        print(f"Evaluating {name}: {path}")
        all_rows.extend(evaluate(name, path))

    per_sample = outdir / "ssim_per_sample_current_topology_inspired.csv"
    fields = ["method", "sample_idx", "ssim_speed", "ssim_u", "ssim_v", "ssim_uv_mean"]
    with per_sample.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(all_rows)

    summary = []
    for name in paths:
        rr = [r for r in all_rows if r["method"] == name]
        speed = float(np.mean([r["ssim_speed"] for r in rr]))
        uv = float(np.mean([r["ssim_uv_mean"] for r in rr]))
        row = {"method": name, "n": len(rr), "ssim_speed_mean": speed, "ssim_uv_mean": uv}
        if name in EXPECTED:
            es, euv = EXPECTED[name]
            row["frozen_speed_mean"] = es
            row["frozen_uv_mean"] = euv
            row["abs_diff_speed"] = abs(speed-es)
            row["abs_diff_uv"] = abs(uv-euv)
        else:
            row["frozen_speed_mean"] = ""
            row["frozen_uv_mean"] = ""
            row["abs_diff_speed"] = ""
            row["abs_diff_uv"] = ""
        summary.append(row)

    summary_csv = outdir / "ssim_summary_current_topology_inspired.csv"
    sf = ["method","n","ssim_speed_mean","ssim_uv_mean","frozen_speed_mean","frozen_uv_mean","abs_diff_speed","abs_diff_uv"]
    with summary_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sf); w.writeheader(); w.writerows(summary)

    print("\nSUMMARY")
    for r in summary:
        print(f"{r['method']:24s} n={r['n']:3d} speed={r['ssim_speed_mean']:.6f} uv_mean={r['ssim_uv_mean']:.6f}")
        if r["method"] in EXPECTED:
            print(f"  frozen target: speed={r['frozen_speed_mean']:.6f} uv_mean={r['frozen_uv_mean']:.6f}")
            print(f"  abs diff:      speed={r['abs_diff_speed']:.3e} uv_mean={r['abs_diff_uv']:.3e}")

    print(f"\nWrote: {per_sample}")
    print(f"Wrote: {summary_csv}")
    print("\nIMPORTANT: only use the topology-inspired SSIM on the poster if the CNN/control")
    print("sanity-check means reproduce the frozen values closely. If they do not, recover")
    print("the original clean-SSIM helper/convention rather than tuning this script to fit.")

if __name__ == "__main__":
    main()
