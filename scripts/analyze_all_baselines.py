#!/usr/bin/env python3
"""
analyze_all_baselines.py
Compute wind-field metrics for CNN, GAN, and bicubic outputs and write
comparison CSVs.

Usage:
    cd ~/PhIRE/scripts
    PYTHONNOUSERSITE=1 /usr/bin/python3 analyze_all_baselines.py

Outputs (relative to repo root):
    ttk_runs_fixed/baseline_metrics/all_methods_per_sample.csv
    ttk_runs_fixed/baseline_metrics/all_methods_summary.csv
    ttk_runs_fixed/baseline_metrics/all_methods_winner_counts.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binned_statistic, kurtosis, wasserstein_distance
from skimage.metrics import structural_similarity

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parent
DATA_ROOT  = REPO_ROOT / "data_out_fixed"
OUT_DIR    = REPO_ROOT / "ttk_runs_fixed" / "baseline_metrics"

METHODS = ["cnn", "gan", "bicubic"]

# ---------------------------------------------------------------------------
# Physical constants and threshold tables
# ---------------------------------------------------------------------------

RHO_AIR = 1.225                        # kg m-3, standard sea-level air density
FIXED_THRESHOLDS_MS = [5.0, 10.0, 15.0, 20.0]   # m s-1
PERCENTILE_THRESHOLDS = [90, 95, 99]

# Metrics where the signed value matters vs. absolute value
_HIGHER_IS_BETTER: frozenset[str] = frozenset({"psnr", "ssim"})
# For these, |val| determines winner (signed bias / delta / frac error)
_ABS_BEST_PREFIX = ("wpd_bias", "psd_slope_delta", "grad_kurtosis_delta",
                    "exceed_err_")

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _method_dir(method: str) -> Path:
    return DATA_ROOT / f"wind_mrhr_{method}"


def load_method(method: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (idx, dataIN, dataGT, dataSR) for one method."""
    d = _method_dir(method)
    idx = np.load(d / "idx.npy")
    din = np.load(d / "dataIN.npy")
    dgt = np.load(d / "dataGT.npy")
    dsr = np.load(d / "dataSR.npy")
    return idx, din, dgt, dsr


def to_nhwc(arr: np.ndarray) -> np.ndarray:
    """Guarantee shape is (N, H, W, 2); transpose from (N, 2, H, W) if needed."""
    if arr.ndim == 4 and arr.shape[1] == 2 and arr.shape[-1] != 2:
        return arr.transpose(0, 2, 3, 1)
    if arr.ndim != 4 or arr.shape[-1] != 2:
        raise ValueError(
            f"Expected 4-D array with 2 channels; got shape {arr.shape}"
        )
    return arr


def wind_speed(uv: np.ndarray) -> np.ndarray:
    """(…, 2) → (…) scalar speed magnitude."""
    return np.sqrt(uv[..., 0] ** 2 + uv[..., 1] ** 2)

# ---------------------------------------------------------------------------
# Individual metrics
# ---------------------------------------------------------------------------

def _psnr(gt: np.ndarray, sr: np.ndarray) -> float:
    """PSNR on the full [u, v] tensor of one sample."""
    mse = float(np.mean((gt.astype(np.float64) - sr.astype(np.float64)) ** 2))
    if mse == 0.0:
        return np.inf
    data_range = float(gt.max() - gt.min())
    if data_range == 0.0:
        return np.nan
    return 20.0 * np.log10(data_range / np.sqrt(mse))


def _ssim(gt_spd: np.ndarray, sr_spd: np.ndarray) -> float:
    """SSIM on scalar wind-speed magnitude (2-D, one sample)."""
    data_range = float(gt_spd.max() - gt_spd.min())
    if data_range == 0.0:
        return np.nan
    return float(structural_similarity(gt_spd, sr_spd, data_range=data_range))


def _wpd_metrics(gt_spd: np.ndarray, sr_spd: np.ndarray) -> dict[str, float]:
    wgt  = (0.5 * RHO_AIR * gt_spd ** 3).ravel()
    wsr  = (0.5 * RHO_AIR * sr_spd ** 3).ravel()
    diff = wsr - wgt
    return {
        "wpd_bias": float(diff.mean()),
        "wpd_mae":  float(np.abs(diff).mean()),
        "wpd_rmse": float(np.sqrt((diff ** 2).mean())),
        "wpd_w1":   float(wasserstein_distance(wsr, wgt)),
    }


def _radial_psd(field: np.ndarray, n_bins: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Radially averaged 1-D power spectrum of a 2-D field."""
    f = field - field.mean()
    ny, nx = f.shape
    F = np.fft.rfft2(f)
    P = np.abs(F) ** 2 / (nx * ny)

    fy = np.fft.fftfreq(ny)
    fx = np.fft.rfftfreq(nx)
    FX, FY = np.meshgrid(fx, fy)
    K = np.sqrt(FX ** 2 + FY ** 2).ravel()
    P_flat = P.ravel()

    bin_means, bin_edges, _ = binned_statistic(
        K, P_flat, statistic="mean",
        bins=np.linspace(1e-9, 0.5, n_bins + 1),
    )
    k_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    valid = np.isfinite(bin_means) & (bin_means > 0)
    return k_centers[valid], bin_means[valid]


def _psd_slope(field: np.ndarray) -> float:
    k, P = _radial_psd(field)
    if len(k) < 5:
        return np.nan
    lk = np.log(k)
    lP = np.log(P)
    lo, hi = np.percentile(lk, [20, 80])
    sel = (lk >= lo) & (lk <= hi)
    if sel.sum() < 3:
        return np.nan
    slope, _ = np.polyfit(lk[sel], lP[sel], 1)
    return float(slope)


def _psd_metrics(gt_spd: np.ndarray, sr_spd: np.ndarray) -> dict[str, float]:
    """PSD log-L2 and slope delta for one sample pair."""
    _, P_gt = _radial_psd(gt_spd)
    _, P_sr = _radial_psd(sr_spd)
    n = min(len(P_gt), len(P_sr))
    log_l2 = float(np.sqrt(np.mean((np.log(P_sr[:n]) - np.log(P_gt[:n])) ** 2)))

    s_gt = _psd_slope(gt_spd)
    s_sr = _psd_slope(sr_spd)
    slope_delta = (float(s_sr - s_gt)
                   if np.isfinite(s_gt) and np.isfinite(s_sr) else np.nan)
    return {"psd_log_l2": log_l2, "psd_slope_delta": slope_delta}


def _gradient_magnitude(spd: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(spd)
    return np.sqrt(gx ** 2 + gy ** 2)


def _gradient_metrics(gt_spd: np.ndarray, sr_spd: np.ndarray) -> dict[str, float]:
    g_gt = _gradient_magnitude(gt_spd).ravel()
    g_sr = _gradient_magnitude(sr_spd).ravel()
    return {
        "grad_mae":             float(np.abs(g_sr - g_gt).mean()),
        "grad_w1":              float(wasserstein_distance(g_sr, g_gt)),
        "grad_kurtosis_delta":  float(kurtosis(g_sr) - kurtosis(g_gt)),
    }


def _exceedance_metrics(gt_spd: np.ndarray, sr_spd: np.ndarray) -> dict[str, float]:
    """Fraction-error at fixed m/s thresholds and GT-distribution percentiles."""
    g = gt_spd.ravel()
    s = sr_spd.ravel()
    out: dict[str, float] = {}
    for thr in FIXED_THRESHOLDS_MS:
        out[f"exceed_err_{int(thr)}ms"] = float(np.mean(s > thr) - np.mean(g > thr))
    for pct in PERCENTILE_THRESHOLDS:
        thr = float(np.percentile(g, pct))
        out[f"exceed_err_p{pct}"] = float(np.mean(s > thr) - np.mean(g > thr))
    return out

# ---------------------------------------------------------------------------
# Per-sample driver
# ---------------------------------------------------------------------------

def metrics_for_sample(gt_uv: np.ndarray, sr_uv: np.ndarray) -> dict[str, float]:
    """All metrics for one (H, W, 2) ground-truth / SR pair."""
    gt_spd = wind_speed(gt_uv)
    sr_spd = wind_speed(sr_uv)
    spd_diff = sr_spd - gt_spd

    row: dict[str, float] = {
        "psnr":       _psnr(gt_uv, sr_uv),
        "ssim":       _ssim(gt_spd, sr_spd),
        "speed_mae":  float(np.abs(spd_diff).mean()),
        "speed_rmse": float(np.sqrt((spd_diff ** 2).mean())),
    }
    row.update(_wpd_metrics(gt_spd, sr_spd))
    row.update(_psd_metrics(gt_spd, sr_spd))
    row.update(_gradient_metrics(gt_spd, sr_spd))
    row.update(_exceedance_metrics(gt_spd, sr_spd))
    return row

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── load ────────────────────────────────────────────────────────────────
    print("=" * 60)
    print("Loading method outputs")
    print("=" * 60)
    loaded: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for method in METHODS:
        mdir = _method_dir(method)
        if not mdir.exists():
            print(f"  MISSING: {mdir}", file=sys.stderr)
            sys.exit(1)
        idx, din, dgt, dsr = load_method(method)
        dgt = to_nhwc(dgt)
        dsr = to_nhwc(dsr)
        loaded[method] = (idx, din, dgt, dsr)
        print(
            f"  {method:<10}  idx={idx.shape}  "
            f"dataIN={din.shape}  dataGT={dgt.shape}  dataSR={dsr.shape}  "
            f"GT range=[{dgt.min():.3g}, {dgt.max():.3g}]"
        )

    # ── idx alignment ────────────────────────────────────────────────────────
    print()
    print("Verifying idx alignment across methods …", end="  ")
    idx_ref = loaded[METHODS[0]][0]
    for method in METHODS[1:]:
        if not np.array_equal(idx_ref, loaded[method][0]):
            print("FAIL")
            raise ValueError(
                f"idx mismatch between {METHODS[0]} and {method}. "
                "Re-run bicubic with the same sample list."
            )
    n_samples = len(idx_ref)
    print(f"OK  ({n_samples} samples, methods: {METHODS})")

    # ── per-sample metrics ────────────────────────────────────────────────────
    print()
    print("Computing per-sample metrics …")
    rows: list[dict] = []
    report_every = max(1, n_samples // 10)
    for i in range(n_samples):
        if i % report_every == 0:
            print(f"  {i:>5} / {n_samples}")
        for method in METHODS:
            idx, _, dgt, dsr = loaded[method]
            m = metrics_for_sample(dgt[i], dsr[i])
            rows.append({"method": method, "sample_idx": int(idx[i]), **m})

    df = pd.DataFrame(rows)
    metric_cols = [c for c in df.columns if c not in ("method", "sample_idx")]

    per_sample_path = OUT_DIR / "all_methods_per_sample.csv"
    df.to_csv(per_sample_path, index=False)
    print(f"\n  → {per_sample_path}")

    # ── summary ──────────────────────────────────────────────────────────────
    summary = (
        df.groupby("method")[metric_cols]
        .agg(["mean", "std"])
        .round(6)
    )
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary_path = OUT_DIR / "all_methods_summary.csv"
    summary.to_csv(summary_path)
    print(f"  → {summary_path}")

    # ── winner counts ─────────────────────────────────────────────────────────
    # For each metric × sample, find which method wins.
    # Higher-is-better: max wins. Signed-bias/delta: |val| min wins.
    # All others: min wins.
    abs_best_cols = {
        c for c in metric_cols
        if any(c.startswith(pfx) for pfx in _ABS_BEST_PREFIX)
    }
    winner_records: list[dict] = []
    for sample_idx_val in idx_ref.tolist():
        sub = df[df["sample_idx"] == int(sample_idx_val)].set_index("method")
        for col in metric_cols:
            vals = sub[col]
            if col in _HIGHER_IS_BETTER:
                winner = vals.idxmax()
            elif col in abs_best_cols:
                winner = vals.abs().idxmin()
            else:
                winner = vals.idxmin()
            winner_records.append({"metric": col, "winner": winner})

    win_df = pd.DataFrame(winner_records)
    win_counts = (
        win_df.groupby(["metric", "winner"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=METHODS, fill_value=0)
    )
    win_counts.columns.name = None
    win_path = OUT_DIR / "all_methods_winner_counts.csv"
    win_counts.to_csv(win_path)
    print(f"  → {win_path}")

    # ── console summary ────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("Mean metrics per method")
    print("=" * 60)
    mean_df = df.groupby("method")[metric_cols].mean()
    print(mean_df.round(5).to_string())

    print()
    print("Win counts (total across all samples × metrics)")
    print(win_counts.sum(axis=0).to_string())

    print()
    print("Done.")


if __name__ == "__main__":
    main()
