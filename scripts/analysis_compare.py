#!/usr/bin/env python3
"""Compare PSNR, topology distances, and physics metrics.

Inputs:
- combined pairwise topology CSV (method,key,sample_idx,pd_distance,mt_distance)
- one or more method directories containing dataGT.npy/dataSR.npy

Outputs:
- merged CSV for downstream plotting
- text report with correlations and "PSNR-tie topology-break" candidates
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from metrics_physics import compute_physics_metrics


def _to_float(x: object) -> Optional[float]:
    try:
        v = float(str(x).strip())
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


def parse_sample_idx(key: str) -> Optional[int]:
    m = re.search(r"_s(\d+)_", key)
    return int(m.group(1)) if m else None


def psnr(sr: np.ndarray, gt: np.ndarray) -> np.ndarray:
    d = sr - gt
    mse = np.mean(d * d, axis=(1, 2, 3))
    dr = gt.max(axis=(1, 2, 3)) - gt.min(axis=(1, 2, 3))
    out = np.zeros(len(mse), dtype=float)
    for i, m in enumerate(mse):
        out[i] = float("inf") if m == 0 else 20 * math.log10(float(dr[i])) - 10 * math.log10(float(m))
    return out


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def load_topology(path: Path) -> List[Dict[str, object]]:
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
            pd = _to_float(row.get("pd_distance", ""))
            mt = _to_float(row.get("mt_distance", ""))
            if not method or si is None or pd is None or mt is None:
                continue
            rows.append({"method": method, "sample_idx": int(si), "key": key, "pd_distance": pd, "mt_distance": mt})
    return rows


def build_metric_rows(method_dirs: Dict[str, Path]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for method, d in method_dirs.items():
        gt = np.load(d / "dataGT.npy")
        sr = np.load(d / "dataSR.npy")
        ps = psnr(sr, gt)
        phys = compute_physics_metrics(gt, sr)
        for i in range(len(ps)):
            r: Dict[str, object] = {"method": method, "sample_idx": i, "psnr": float(ps[i])}
            r.update(phys[i])
            rows.append(r)
    return rows


def merge_rows(topo: List[Dict[str, object]], metrics: List[Dict[str, object]]) -> List[Dict[str, object]]:
    by = {(str(r["method"]), int(r["sample_idx"])): r for r in metrics}
    out: List[Dict[str, object]] = []
    for t in topo:
        k = (str(t["method"]), int(t["sample_idx"]))
        if k in by:
            r = dict(by[k])
            r.update(t)
            out.append(r)
    return sorted(out, key=lambda z: (str(z["method"]), int(z["sample_idx"])))


def summarize(rows: List[Dict[str, object]], tie_eps: float = 0.25, topo_gap_z: float = 1.0) -> str:
    if not rows:
        return "No merged rows"
    lines: List[str] = []
    methods = sorted({str(r["method"]) for r in rows})

    ps = np.array([float(r["psnr"]) for r in rows])
    pd = np.array([float(r["pd_distance"]) for r in rows])
    mt = np.array([float(r["mt_distance"]) for r in rows])
    wpd = np.array([float(r["wpd_rmse"]) for r in rows])
    lines.append("=== Global correlations ===")
    lines.append(f"corr(PSNR, PD) = {pearson(ps, pd):.4f}")
    lines.append(f"corr(PSNR, MT) = {pearson(ps, mt):.4f}")
    lines.append(f"corr(PSNR, WPD_RMSE) = {pearson(ps, wpd):.4f}")
    lines.append(f"corr(PD, WPD_RMSE) = {pearson(pd, wpd):.4f}")
    lines.append(f"corr(MT, WPD_RMSE) = {pearson(mt, wpd):.4f}")
    lines.append("")

    for m in methods:
        sub = [r for r in rows if str(r["method"]) == m]
        lines.append(f"[{m}] n={len(sub)} mean PSNR={np.mean([float(r['psnr']) for r in sub]):.3f}")
    lines.append("")

    if len(methods) >= 2:
        lines.append("=== PSNR-tie / topology-break candidates ===")
        # pair methods by sample index
        by_idx: Dict[int, List[Dict[str, object]]] = {}
        for r in rows:
            by_idx.setdefault(int(r["sample_idx"]), []).append(r)
        pd_all = np.array([float(r["pd_distance"]) for r in rows])
        mt_all = np.array([float(r["mt_distance"]) for r in rows])
        pd_std = float(np.std(pd_all) + 1e-12)
        mt_std = float(np.std(mt_all) + 1e-12)
        for si in sorted(by_idx):
            rr = by_idx[si]
            if len(rr) < 2:
                continue
            for i in range(len(rr)):
                for j in range(i + 1, len(rr)):
                    a, b = rr[i], rr[j]
                    dps = abs(float(a["psnr"]) - float(b["psnr"]))
                    if dps > tie_eps:
                        continue
                    dpd = abs(float(a["pd_distance"]) - float(b["pd_distance"])) / pd_std
                    dmt = abs(float(a["mt_distance"]) - float(b["mt_distance"])) / mt_std
                    if max(dpd, dmt) >= topo_gap_z:
                        lines.append(
                            f"s{si}: {a['method']} vs {b['method']} | ΔPSNR={dps:.3f} dB | zΔPD={dpd:.2f}, zΔMT={dmt:.2f}"
                        )
    return "\n".join(lines)


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def parse_method_dirs(values: List[str]) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for x in values:
        if "=" not in x:
            raise ValueError(f"Expected method=dir, got: {x}")
        m, d = x.split("=", 1)
        out[m.strip().lower()] = Path(d).expanduser().resolve()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare PSNR + topology + physics metrics")
    ap.add_argument("--topology-csv", required=True)
    ap.add_argument("--method-dir", action="append", required=True, help="method=/path/to/data_out_dir")
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-report", required=True)
    ap.add_argument("--tie-eps", type=float, default=0.25)
    ap.add_argument("--topo-gap-z", type=float, default=1.0)
    args = ap.parse_args()

    topo = load_topology(Path(args.topology_csv))
    methods = parse_method_dirs(args.method_dir)
    metrics = build_metric_rows(methods)
    merged = merge_rows(topo, metrics)
    write_csv(Path(args.out_csv), merged)
    report = summarize(merged, tie_eps=args.tie_eps, topo_gap_z=args.topo_gap_z)
    Path(args.out_report).write_text(report)
    print(f"Wrote: {args.out_csv}")
    print(f"Wrote: {args.out_report}")


if __name__ == "__main__":
    main()
