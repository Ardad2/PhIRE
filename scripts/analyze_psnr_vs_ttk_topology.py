#!/usr/bin/env python3
"""Analyze and visualize quality metric vs TTK topology distances.

By default this reproduces PSNR artifacts:
- psnr_topology_merged.csv
- psnr_topology_stats.txt
- psnr_vs_topology_global.png
- psnr_vs_topology_by_method.png

With --metric ssim it produces:
- ssim_topology_merged.csv
- ssim_topology_stats.txt
- ssim_vs_topology_global.png
- ssim_vs_topology_by_method.png
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
            rows.append({"method": method, "key": key, "sample_idx": sample_idx, "pd_distance": pdv, "mt_distance": mtv})
    return rows


def build_psnr_rows(method_dirs: Dict[str, Path]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for method, d in method_dirs.items():
        gt = np.load(d / "dataGT.npy")
        sr = np.load(d / "dataSR.npy")
        vals = psnr(sr, gt)
        for i, v in enumerate(vals):
            rows.append({"method": method, "sample_idx": i, "metric": float(v)})
    return rows


def build_metric_rows_from_merged(path: Path, metric: str) -> List[Dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing merged CSV for metric={metric}: {path}")
    rows: List[Dict[str, object]] = []
    with path.open("r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            method = str(row.get("method", "")).strip().lower()
            key = str(row.get("key", "")).strip()
            si = _to_float(row.get("sample_idx", ""))
            if si is None:
                si2 = parse_sample_idx(key)
                si = float(si2) if si2 is not None else None
            mv = _to_float(row.get(metric, ""))
            if not method or si is None or mv is None:
                continue
            rows.append({"method": method, "sample_idx": int(si), "metric": mv})
    return rows


def merge_rows(topo_rows: List[Dict[str, object]], metric_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    by = {(str(r["method"]), int(r["sample_idx"])): float(r["metric"]) for r in metric_rows}
    out: List[Dict[str, object]] = []
    for r in topo_rows:
        k = (str(r["method"]), int(r["sample_idx"]))
        if k not in by:
            continue
        out.append({**r, "metric": by[k]})
    return sorted(out, key=lambda z: (str(z["method"]), int(z["sample_idx"])))


def write_csv(path: Path, rows: List[Dict[str, object]], metric: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["method", "key", "sample_idx", metric, "pd_distance", "mt_distance"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({"method": r["method"], "key": r["key"], "sample_idx": r["sample_idx"], metric: r["metric"], "pd_distance": r["pd_distance"], "mt_distance": r["mt_distance"]})


def summarize(rows: List[Dict[str, object]], metric: str) -> str:
    if not rows:
        return "No merged rows."

    q = np.array([float(r["metric"]) for r in rows], dtype=float)
    pd = np.array([float(r["pd_distance"]) for r in rows], dtype=float)
    mt = np.array([float(r["mt_distance"]) for r in rows], dtype=float)

    mtag = metric.upper()
    lines: List[str] = []
    lines.append(f"=== Correlations ({mtag} vs topology; lower topology is better) ===")
    lines.append(f"global Pearson({mtag}, PD): {pearson(q, pd)}")
    lines.append(f"global Pearson({mtag}, MT): {pearson(q, mt)}")
    lines.append(f"global Spearman({mtag}, PD): {spearman(q, pd)}")
    lines.append(f"global Spearman({mtag}, MT): {spearman(q, mt)}")
    lines.append("")

    methods = sorted({str(r["method"]) for r in rows})
    for m in methods:
        sub = [r for r in rows if str(r["method"]) == m]
        q_m = np.array([float(r["metric"]) for r in sub])
        pd_m = np.array([float(r["pd_distance"]) for r in sub])
        mt_m = np.array([float(r["mt_distance"]) for r in sub])
        lines.append(f"[{m}] n={len(sub)}")
        lines.append(f"  Pearson({mtag}, PD): {pearson(q_m, pd_m)}")
        lines.append(f"  Pearson({mtag}, MT): {pearson(q_m, mt_m)}")
        lines.append(f"  Spearman({mtag}, PD): {spearman(q_m, pd_m)}")
        lines.append(f"  Spearman({mtag}, MT): {spearman(q_m, mt_m)}")
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
        lines.append("=== Mean deltas (GAN - CNN); >0 means CNN lower/better topology ===")
        lines.append(f"PD delta: {d_pd:.4f}  95% CI [{lo_pd:.4f}, {hi_pd:.4f}]")
        lines.append(f"MT delta: {d_mt:.4f}  95% CI [{lo_mt:.4f}, {hi_mt:.4f}]")

    return "\n".join(lines)


def _scatter(ax, x, y, title, xlabel):
    ax.scatter(x, y, s=35)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Topology distance")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.3)


def make_plots(rows: List[Dict[str, object]], outdir: Path, metric: str) -> List[Path]:
    outs: List[Path] = []
    if plt is None or not rows:
        return outs

    label = metric.upper()
    x = [float(r["metric"]) for r in rows]
    pd = [float(r["pd_distance"]) for r in rows]
    mt = [float(r["mt_distance"]) for r in rows]

    fig = plt.figure(figsize=(10, 4.5))
    ax1 = fig.add_subplot(1, 2, 1)
    _scatter(ax1, x, pd, f"Global: {label} vs PD", label)
    ax1.set_ylabel("TTK PD distance")
    ax2 = fig.add_subplot(1, 2, 2)
    _scatter(ax2, x, mt, f"Global: {label} vs MT", label)
    ax2.set_ylabel("TTK MT distance")
    fig.tight_layout()
    p1 = outdir / f"{metric}_vs_topology_global.png"
    fig.savefig(p1, dpi=200)
    plt.close(fig)
    outs.append(p1)

    methods = sorted({str(r["method"]) for r in rows})
    if methods:
        fig2 = plt.figure(figsize=(10, max(4.5, 3.2 * len(methods))))
        for i, m in enumerate(methods, start=1):
            sub = [r for r in rows if str(r["method"]) == m]
            x_m = [float(r["metric"]) for r in sub]
            y_pd = [float(r["pd_distance"]) for r in sub]
            y_mt = [float(r["mt_distance"]) for r in sub]
            axp = fig2.add_subplot(len(methods), 2, 2 * i - 1)
            _scatter(axp, x_m, y_pd, f"{m.upper()}: {label} vs PD", label)
            axp.set_ylabel("TTK PD distance")
            axm = fig2.add_subplot(len(methods), 2, 2 * i)
            _scatter(axm, x_m, y_mt, f"{m.upper()}: {label} vs MT", label)
            axm.set_ylabel("TTK MT distance")
        fig2.tight_layout()
        p2 = outdir / f"{metric}_vs_topology_by_method.png"
        fig2.savefig(p2, dpi=200)
        plt.close(fig2)
        outs.append(p2)

    return outs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--combined", type=Path, default=Path("ttk_runs/combined/combined_pairwise_results.csv"))
    ap.add_argument("--gan-dir", type=Path, default=Path("data_out/wind_mrhr_gan"))
    ap.add_argument("--cnn-dir", type=Path, default=Path("data_out/wind_mrhr_cnn"))
    ap.add_argument("--merged-csv", type=Path, default=Path("ttk_runs/combined/psnr_topology_physics_merged.csv"), help="Used when --metric ssim")
    ap.add_argument("--metric", choices=["psnr", "ssim"], default="psnr")
    ap.add_argument("--outdir", type=Path, default=Path("ttk_runs/combined"))
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    topo_rows = load_ttk_combined(args.combined)
    if args.metric == "psnr":
        metric_rows = build_psnr_rows({"gan": args.gan_dir, "cnn": args.cnn_dir})
    else:
        metric_rows = build_metric_rows_from_merged(args.merged_csv, "ssim")

    merged = merge_rows(topo_rows, metric_rows)

    out_csv = args.outdir / f"{args.metric}_topology_merged.csv"
    write_csv(out_csv, merged, args.metric)

    report = summarize(merged, args.metric)
    out_txt = args.outdir / f"{args.metric}_topology_stats.txt"
    out_txt.write_text(report)
    print(report)
    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_txt}")

    plots = make_plots(merged, args.outdir, args.metric)
    for p in plots:
        print(f"Wrote: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
