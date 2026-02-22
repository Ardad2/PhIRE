#!/usr/bin/env python3
"""Physics-based metrics for scalar-field SR evaluation.

Computes per-sample metrics from GT/SR arrays (N,H,W,C), focusing on:
- Wind-power-density proxies via speed^3
- Spectrum mismatch (radial PSD)
- Gradient/intermittency mismatch
- Exceedance-area mismatch at fixed and percentile-based thresholds
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


def _ensure_4d(a: np.ndarray) -> np.ndarray:
    if a.ndim == 3:
        return a[..., None]
    if a.ndim != 4:
        raise ValueError(f"Expected array with ndim 3 or 4, got shape={a.shape}")
    return a


def _speed_from_field(x: np.ndarray) -> np.ndarray:
    x = _ensure_4d(x)
    if x.shape[-1] >= 2:
        return np.sqrt(np.square(x[..., 0]) + np.square(x[..., 1]))
    return np.abs(x[..., 0])


def _wasserstein_1d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.size == 0 or b.size == 0:
        return float("nan")
    a = np.sort(a)
    b = np.sort(b)
    n = min(len(a), len(b))
    if n < 2:
        return float(abs(a[0] - b[0]))
    qa = np.interp(np.linspace(0.0, 1.0, n), np.linspace(0.0, 1.0, len(a)), a)
    qb = np.interp(np.linspace(0.0, 1.0, n), np.linspace(0.0, 1.0, len(b)), b)
    return float(np.mean(np.abs(qa - qb)))


def _radial_psd(img2d: np.ndarray, bins: int = 32) -> Tuple[np.ndarray, np.ndarray]:
    f = np.fft.rfft2(img2d)
    p = np.abs(f) ** 2
    h, w2 = p.shape
    yy, xx = np.indices((h, w2))
    r = np.sqrt((yy - (h / 2.0)) ** 2 + xx**2)
    r = r / (r.max() + 1e-12)
    edges = np.linspace(0.0, 1.0, bins + 1)
    vals = np.zeros(bins, dtype=float)
    for i in range(bins):
        m = (r >= edges[i]) & (r < edges[i + 1])
        if np.any(m):
            vals[i] = float(np.mean(p[m]))
    return 0.5 * (edges[:-1] + edges[1:]), vals


def compute_physics_metrics(
    gt: np.ndarray,
    sr: np.ndarray,
    exceedance_thresholds: Sequence[float] = (5.0, 10.0, 15.0),
    percentile_thresholds: Sequence[float] = (90.0, 95.0, 99.0),
    psd_bins: int = 32,
) -> List[Dict[str, float]]:
    gt = _ensure_4d(gt)
    sr = _ensure_4d(sr)
    if gt.shape != sr.shape:
        raise ValueError(f"Shape mismatch gt={gt.shape} sr={sr.shape}")

    speed_gt = _speed_from_field(gt)
    speed_sr = _speed_from_field(sr)

    out: List[Dict[str, float]] = []
    for i in range(gt.shape[0]):
        sgt = speed_gt[i]
        ssr = speed_sr[i]

        wpd_gt = sgt**3
        wpd_sr = ssr**3
        wpd_bias = float(np.mean(wpd_sr - wpd_gt))
        wpd_mae = float(np.mean(np.abs(wpd_sr - wpd_gt)))
        wpd_rmse = float(np.sqrt(np.mean((wpd_sr - wpd_gt) ** 2)))
        wpd_w1 = _wasserstein_1d(wpd_gt, wpd_sr)

        k, psd_gt = _radial_psd(sgt, bins=psd_bins)
        _, psd_sr = _radial_psd(ssr, bins=psd_bins)
        psd_log_l2 = float(np.sqrt(np.mean((np.log1p(psd_sr) - np.log1p(psd_gt)) ** 2)))
        fit_lo = max(2, psd_bins // 8)
        fit_hi = max(fit_lo + 2, psd_bins // 2)
        x = np.log(k[fit_lo:fit_hi] + 1e-8)
        y_gt = np.log(psd_gt[fit_lo:fit_hi] + 1e-8)
        y_sr = np.log(psd_sr[fit_lo:fit_hi] + 1e-8)
        slope_gt = float(np.polyfit(x, y_gt, 1)[0]) if len(x) >= 2 else float("nan")
        slope_sr = float(np.polyfit(x, y_sr, 1)[0]) if len(x) >= 2 else float("nan")
        psd_slope_delta = float(slope_sr - slope_gt) if not (math.isnan(slope_gt) or math.isnan(slope_sr)) else float("nan")

        gy_gt, gx_gt = np.gradient(sgt)
        gy_sr, gx_sr = np.gradient(ssr)
        gm_gt = np.sqrt(gx_gt**2 + gy_gt**2)
        gm_sr = np.sqrt(gx_sr**2 + gy_sr**2)
        grad_mae = float(np.mean(np.abs(gm_sr - gm_gt)))
        grad_w1 = _wasserstein_1d(gm_gt, gm_sr)
        kurt_gt = float(np.mean((gm_gt - gm_gt.mean()) ** 4) / (np.std(gm_gt) ** 4 + 1e-12))
        kurt_sr = float(np.mean((gm_sr - gm_sr.mean()) ** 4) / (np.std(gm_sr) ** 4 + 1e-12))
        grad_kurtosis_delta = float(kurt_sr - kurt_gt)

        row: Dict[str, float] = {
            "sample_idx": float(i),
            "wpd_bias": wpd_bias,
            "wpd_mae": wpd_mae,
            "wpd_rmse": wpd_rmse,
            "wpd_w1": wpd_w1,
            "psd_log_l2": psd_log_l2,
            "psd_slope_gt": slope_gt,
            "psd_slope_sr": slope_sr,
            "psd_slope_delta": psd_slope_delta,
            "psd_slope_abs_delta": float(abs(psd_slope_delta)) if not math.isnan(psd_slope_delta) else float("nan"),
            "grad_mae": grad_mae,
            "grad_w1": grad_w1,
            "grad_kurtosis_gt": kurt_gt,
            "grad_kurtosis_sr": kurt_sr,
            "grad_kurtosis_delta": grad_kurtosis_delta,
            "grad_kurtosis_abs_delta": float(abs(grad_kurtosis_delta)),
        }

        for t in exceedance_thresholds:
            gt_frac = float(np.mean(sgt > t))
            sr_frac = float(np.mean(ssr > t))
            delta = sr_frac - gt_frac
            row[f"exceed_frac_gt_t{t:g}"] = gt_frac
            row[f"exceed_frac_sr_t{t:g}"] = sr_frac
            row[f"exceed_frac_delta_t{t:g}"] = delta
            row[f"exceed_frac_abs_delta_t{t:g}"] = float(abs(delta))

        for p in percentile_thresholds:
            thr = float(np.percentile(sgt, p))
            gt_frac = float(np.mean(sgt > thr))
            sr_frac = float(np.mean(ssr > thr))
            delta = sr_frac - gt_frac
            tag = f"p{int(p)}"
            row[f"exceed_frac_gt_{tag}"] = gt_frac
            row[f"exceed_frac_sr_{tag}"] = sr_frac
            row[f"exceed_frac_delta_{tag}"] = delta
            row[f"exceed_frac_abs_delta_{tag}"] = float(abs(delta))

        out.append(row)

    return out


def _write_csv(path: Path, rows: List[Dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    cols = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _parse_list(vals: str) -> List[float]:
    return [float(x.strip()) for x in vals.split(",") if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute physics metrics from GT/SR arrays")
    ap.add_argument("--gt", required=True, help="Path to dataGT.npy")
    ap.add_argument("--sr", required=True, help="Path to dataSR.npy")
    ap.add_argument("--out", required=True, help="Output CSV path")
    ap.add_argument("--thresholds", default="5,10,15", help="Comma-separated fixed exceedance thresholds")
    ap.add_argument("--percentiles", default="90,95,99", help="Comma-separated GT percentile exceedance thresholds")
    ap.add_argument("--psd-bins", type=int, default=32)
    args = ap.parse_args()

    gt = np.load(args.gt)
    sr = np.load(args.sr)
    thresholds = _parse_list(str(args.thresholds))
    percentiles = _parse_list(str(args.percentiles))
    rows = compute_physics_metrics(gt, sr, thresholds, percentiles, psd_bins=args.psd_bins)
    _write_csv(Path(args.out), rows)
    print(f"Wrote: {args.out}")


if __name__ == "__main__":
    main()
