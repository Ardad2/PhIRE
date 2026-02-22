#!/usr/bin/env python3
"""Analyze and visualize PSNR vs TTK topology distances.

This script is intentionally Phase-C/TTK based:
- topology comes from `combined_pairwise_results.csv` produced by
  `scripts/combine_phase_c_results.py` over TTK distance outputs.
- PSNR is computed directly from per-method `dataGT.npy`/`dataSR.npy`.

Outputs:
- psnr_topology_merged.csv
- psnr_topology_stats.txt
- psnr_vs_topology_global.png
- psnr_vs_topology_by_method.png
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover
    plt = None


def _to_float(x: object) -> Optional[float]:
    try:
        if x is None:
            return None
        s = str(x).strip()
        if not s:
            return None
        v = float(s)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


def mse(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = a - b
    return np.mean(d * d, axis=(1, 2, 3))


def psnr(sr: np.ndarray, gt: np.ndarray) -> np.ndarray:
    dr = gt.max(axis=(1, 2, 3)) - gt.min(axis=(1, 2, 3))
    m = mse(sr, gt)
    out: List[float] = []
    for i in range(len(m)):
        if m[i] == 0:
            out.append(float("inf"))
        else:
            out.append(20 * math.log10(float(dr[i])) - 10 * math.log10(float(m[i])))
    return np.asarray(out, dtype=float)


def parse_sample_idx(key: str) -> Optional[int]:
    m = re.search(r"_s(\d+)_", key)
    if not m:
        return None
    return int(m.group(1))


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def rankdata(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(a), dtype=float)
    vals, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    for i, c in enumerate(counts):
        if c > 1:
            idx = np.where(inv == i)[0]
            ranks[idx] = ranks[idx].mean()
    return ranks + 1.0


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return pearson(rankdata(x), rankdata(y))


def bootstrap_mean_diff(a: np.ndarray, b: np.ndarray, nboot: int = 20000, seed: int = 0) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(nboot):
        aa = a[rng.integers(0, len(a), len(a))]
        bb = b[rng.integers(0, len(b), len(b))]
        out.append(float(np.mean(aa) - np.mean(bb)))
    boots = np.asarray(out)
    point = float(np.mean(a) - np.mean(b))
    return point, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def load_ttk_combined(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing topology input: {path}")
    rows: List[Dict[str, object]] = []
    with path.open("r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            method = str(row.get("method", "")).strip().lower()
            key = str(row.get("key", "")).strip()
            pdv = _to_float(row.get("pd_distance", ""))
            mtv = _to_float(row.get("mt_distance", ""))
            sample_idx = parse_sample_idx(key)
            if not method or sample_idx is None or pdv is None or mtv is None:
                continue
            rows.append(
                {
                    "method": method,
                    "key": key,
                    "sample_idx": sample_idx,
                    "pd_distance": pdv,
                    "mt_distance": mtv,
                }
            )
    return rows


def build_psnr_rows(method_dirs: Dict[str, Path]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for method, d in method_dirs.items():
        gt_path = d / "dataGT.npy"
        sr_path = d / "dataSR.npy"
        if not gt_path.exists() or not sr_path.exists():
            raise FileNotFoundError(f"Missing {gt_path} or {sr_path}")
        gt = np.load(gt_path)
        sr = np.load(sr_path)
        vals = psnr(sr, gt)
        for i, v in enumerate(vals):
            rows.append({"method": method, "sample_idx": i, "psnr": float(v)})
    return rows


def merge_rows(topo_rows: List[Dict[str, object]], psnr_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    by = {(str(r["method"]), int(r["sample_idx"])): float(r["psnr"]) for r in psnr_rows}
    out: List[Dict[str, object]] = []
    for r in topo_rows:
        k = (str(r["method"]), int(r["sample_idx"]))
        if k not in by:
            continue
        out.append({**r, "psnr": by[k]})
    return sorted(out, key=lambda z: (str(z["method"]), int(z["sample_idx"])))


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["method", "key", "sample_idx", "psnr", "pd_distance", "mt_distance"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def summarize(rows: List[Dict[str, object]]) -> str:
    if not rows:
        return "No merged rows."

    ps = np.array([float(r["psnr"]) for r in rows], dtype=float)
    pd = np.array([float(r["pd_distance"]) for r in rows], dtype=float)
    mt = np.array([float(r["mt_distance"]) for r in rows], dtype=float)

    lines = []
    lines.append("=== Correlations (PSNR vs topology; lower topology is better) ===")
    lines.append(f"global Pearson(PSNR, PD): {pearson(ps, pd)}")
    lines.append(f"global Pearson(PSNR, MT): {pearson(ps, mt)}")
    lines.append(f"global Spearman(PSNR, PD): {spearman(ps, pd)}")
    lines.append(f"global Spearman(PSNR, MT): {spearman(ps, mt)}")
    lines.append("")

    methods = sorted({str(r["method"]) for r in rows})
    for m in methods:
        sub = [r for r in rows if str(r["method"]) == m]
        ps_m = np.array([float(r["psnr"]) for r in sub])
        pd_m = np.array([float(r["pd_distance"]) for r in sub])
        mt_m = np.array([float(r["mt_distance"]) for r in sub])
        lines.append(f"[{m}] n={len(sub)}")
        lines.append(f"  Pearson(PSNR, PD): {pearson(ps_m, pd_m)}")
        lines.append(f"  Pearson(PSNR, MT): {pearson(ps_m, mt_m)}")
        lines.append(f"  Spearman(PSNR, PD): {spearman(ps_m, pd_m)}")
        lines.append(f"  Spearman(PSNR, MT): {spearman(ps_m, mt_m)}")
        lines.append("")

    if "gan" in methods and "cnn" in methods:
        gan = [r for r in rows if str(r["method"]) == "gan"]
        cnn = [r for r in rows if str(r["method"]) == "cnn"]
        gan_pd = np.array([float(r["pd_distance"]) for r in gan])
        cnn_pd = np.array([float(r["pd_distance"]) for r in cnn])
        gan_mt = np.array([float(r["mt_distance"]) for r in gan])
        cnn_mt = np.array([float(r["mt_distance"]) for r in cnn])

        d_pd, lo_pd, hi_pd = bootstrap_mean_diff(gan_pd, cnn_pd)
        d_mt, lo_mt, hi_mt = bootstrap_mean_diff(gan_mt, cnn_mt)
        lines.append("=== Mean deltas (GAN - CNN); >0 means CNN lower/better ===")
        lines.append(f"PD delta: {d_pd:.4f}  95% CI [{lo_pd:.4f}, {hi_pd:.4f}]")
        lines.append(f"MT delta: {d_mt:.4f}  95% CI [{lo_mt:.4f}, {hi_mt:.4f}]")

    return "\n".join(lines)


def _scatter(ax, x, y, title, ylab):
    ax.scatter(x, y, s=35)
    ax.set_title(title)
    ax.set_xlabel("PSNR (dB)")
    ax.set_ylabel(ylab)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.3)


def make_plots(rows: List[Dict[str, object]], outdir: Path) -> List[Path]:
    if plt is None:
        print("[warn] matplotlib unavailable; skipping plots")
        return []
    outs: List[Path] = []

    # Global figure (PD, MT)
    ps = [float(r["psnr"]) for r in rows]
    pd = [float(r["pd_distance"]) for r in rows]
    mt = [float(r["mt_distance"]) for r in rows]

    fig = plt.figure(figsize=(11, 4.5))
    ax1 = fig.add_subplot(121)
    _scatter(ax1, ps, pd, "Global: PSNR vs TTK PD", "TTK PD distance")
    ax2 = fig.add_subplot(122)
    _scatter(ax2, ps, mt, "Global: PSNR vs TTK MT", "TTK MT distance")
    fig.tight_layout()
    p1 = outdir / "psnr_vs_topology_global.png"
    fig.savefig(p1, dpi=200)
    plt.close(fig)
    outs.append(p1)

    methods = sorted({str(r["method"]) for r in rows})
    if methods:
        fig2 = plt.figure(figsize=(12, 4 * len(methods)))
        for i, m in enumerate(methods, start=1):
            sub = [r for r in rows if str(r["method"]) == m]
            x = [float(r["psnr"]) for r in sub]
            y_pd = [float(r["pd_distance"]) for r in sub]
            y_mt = [float(r["mt_distance"]) for r in sub]
            axp = fig2.add_subplot(len(methods), 2, 2 * i - 1)
            _scatter(axp, x, y_pd, f"{m.upper()}: PSNR vs TTK PD", "TTK PD distance")
            axm = fig2.add_subplot(len(methods), 2, 2 * i)
            _scatter(axm, x, y_mt, f"{m.upper()}: PSNR vs TTK MT", "TTK MT distance")
        fig2.tight_layout()
        p2 = outdir / "psnr_vs_topology_by_method.png"
        fig2.savefig(p2, dpi=200)
        plt.close(fig2)
        outs.append(p2)

    return outs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--combined", type=Path, default=Path("ttk_runs/combined/combined_pairwise_results.csv"))
    ap.add_argument("--gan-dir", type=Path, default=Path("data_out/wind_mrhr_gan"))
    ap.add_argument("--cnn-dir", type=Path, default=Path("data_out/wind_mrhr_cnn"))
    ap.add_argument("--outdir", type=Path, default=Path("ttk_runs/combined"))
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    topo_rows = load_ttk_combined(args.combined)
    psnr_rows = build_psnr_rows({"gan": args.gan_dir, "cnn": args.cnn_dir})
    merged = merge_rows(topo_rows, psnr_rows)

    out_csv = args.outdir / "psnr_topology_merged.csv"
    write_csv(out_csv, merged)

    report = summarize(merged)
    out_txt = args.outdir / "psnr_topology_stats.txt"
    out_txt.write_text(report)
    print(report)
    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_txt}")

    plots = make_plots(merged, args.outdir)
    for p in plots:
        print(f"Wrote: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

