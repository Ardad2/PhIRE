#!/usr/bin/env python3
"""Derive metric-specific topology/physics artifacts from merged CSV.

Creates:
- <metric>_topology_physics_merged.csv
- <metric>_topology_physics_delta.csv
- <metric>_topology_physics_report.txt
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def _to_float(x: object) -> Optional[float]:
    try:
        v = float(str(x).strip())
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


def _read(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with path.open("r", newline="") as f:
        r = csv.DictReader(f)
        if r.fieldnames is None:
            raise ValueError(f"No header: {path}")
        rows = list(r)
    return list(r.fieldnames), rows


def _write(path: Path, cols: List[str], rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    n = min(len(x), len(y))
    return float(np.corrcoef(x[:n], y[:n])[0, 1])


def _rank(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(a), dtype=float)
    return ranks + 1.0


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    return _corr(_rank(x), _rank(y))


def _bootstrap_mean_diff(a: np.ndarray, b: np.ndarray, nboot: int = 4000, seed: int = 0) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(nboot):
        aa = a[rng.integers(0, len(a), len(a))]
        bb = b[rng.integers(0, len(b), len(b))]
        vals.append(float(np.mean(aa) - np.mean(bb)))
    v = np.asarray(vals)
    return float(np.mean(a) - np.mean(b)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-csv", type=Path, required=True)
    ap.add_argument("--metric", choices=["psnr", "ssim"], required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--nboot", type=int, default=4000)
    args = ap.parse_args()

    cols, rows = _read(args.in_csv)
    if not rows:
        raise SystemExit(f"No rows in {args.in_csv}")
    if args.metric not in cols:
        raise SystemExit(f"Missing metric column {args.metric} in {args.in_csv}")

    # metric-specific merged (full row copy for convenience)
    m_csv = args.outdir / f"{args.metric}_topology_physics_merged.csv"
    _write(m_csv, cols, [dict(r) for r in rows])

    # paired deltas (first two methods by name)
    methods = sorted({str(r.get("method", "")).strip().lower() for r in rows if str(r.get("method", "")).strip()})
    d_rows: List[Dict[str, object]] = []
    numeric_cols = [c for c in cols if c not in {"method", "key"}]
    if len(methods) >= 2:
        a, b = methods[0], methods[1]
        by: Dict[int, Dict[str, Dict[str, str]]] = {}
        for r in rows:
            si = _to_float(r.get("sample_idx", ""))
            if si is None:
                continue
            by.setdefault(int(si), {})[str(r.get("method", "")).strip().lower()] = r
        for si in sorted(by):
            rr = by[si]
            if a not in rr or b not in rr:
                continue
            ra, rb = rr[a], rr[b]
            out: Dict[str, object] = {"sample_idx": si, "method_a": a, "method_b": b}
            for c in numeric_cols:
                va = _to_float(ra.get(c, "")); vb = _to_float(rb.get(c, ""))
                if va is None or vb is None:
                    continue
                out[f"delta_{c}"] = float(va - vb)
            d_rows.append(out)
    d_csv = args.outdir / f"{args.metric}_topology_physics_delta.csv"
    d_cols = list(d_rows[0].keys()) if d_rows else ["sample_idx", "method_a", "method_b"]
    _write(d_csv, d_cols, d_rows)

    # report
    q = np.asarray([_to_float(r.get(args.metric, "")) for r in rows if _to_float(r.get(args.metric, "")) is not None], dtype=float)
    pd = np.asarray([_to_float(r.get("pd_distance", "")) for r in rows if _to_float(r.get("pd_distance", "")) is not None], dtype=float)
    mt = np.asarray([_to_float(r.get("mt_distance", "")) for r in rows if _to_float(r.get("mt_distance", "")) is not None], dtype=float)

    lines: List[str] = []
    tag = args.metric.upper()
    lines.append(f"=== Correlations ({tag} vs topology) ===")
    lines.append(f"global Pearson({tag}, PD): {_corr(q,pd)}")
    lines.append(f"global Pearson({tag}, MT): {_corr(q,mt)}")
    lines.append(f"global Spearman({tag}, PD): {_spearman(q,pd)}")
    lines.append(f"global Spearman({tag}, MT): {_spearman(q,mt)}")
    lines.append("")

    for m in methods:
        sub = [r for r in rows if str(r.get("method", "")).strip().lower() == m]
        qq = np.asarray([_to_float(r.get(args.metric, "")) for r in sub if _to_float(r.get(args.metric, "")) is not None], dtype=float)
        pp = np.asarray([_to_float(r.get("pd_distance", "")) for r in sub if _to_float(r.get("pd_distance", "")) is not None], dtype=float)
        mm = np.asarray([_to_float(r.get("mt_distance", "")) for r in sub if _to_float(r.get("mt_distance", "")) is not None], dtype=float)
        lines.append(f"[{m}] n={len(sub)}")
        lines.append(f"  Pearson({tag}, PD): {_corr(qq, pp)}")
        lines.append(f"  Pearson({tag}, MT): {_corr(qq, mm)}")
        lines.append(f"  Spearman({tag}, PD): {_spearman(qq, pp)}")
        lines.append(f"  Spearman({tag}, MT): {_spearman(qq, mm)}")
        lines.append("")

    if "gan" in methods and "cnn" in methods:
        gan = [r for r in rows if str(r.get("method", "")).strip().lower() == "gan"]
        cnn = [r for r in rows if str(r.get("method", "")).strip().lower() == "cnn"]
        gpd = np.asarray([_to_float(r.get("pd_distance", "")) for r in gan if _to_float(r.get("pd_distance", "")) is not None], dtype=float)
        cpd = np.asarray([_to_float(r.get("pd_distance", "")) for r in cnn if _to_float(r.get("pd_distance", "")) is not None], dtype=float)
        gmt = np.asarray([_to_float(r.get("mt_distance", "")) for r in gan if _to_float(r.get("mt_distance", "")) is not None], dtype=float)
        cmt = np.asarray([_to_float(r.get("mt_distance", "")) for r in cnn if _to_float(r.get("mt_distance", "")) is not None], dtype=float)
        d_pd, lo_pd, hi_pd = _bootstrap_mean_diff(gpd, cpd, nboot=args.nboot)
        d_mt, lo_mt, hi_mt = _bootstrap_mean_diff(gmt, cmt, nboot=args.nboot)
        lines.append("=== Mean deltas (GAN - CNN); >0 means CNN lower/better topology ===")
        lines.append(f"PD delta: {d_pd:.4f}  95% CI [{lo_pd:.4f}, {hi_pd:.4f}]")
        lines.append(f"MT delta: {d_mt:.4f}  95% CI [{lo_mt:.4f}, {hi_mt:.4f}]")

    rep = args.outdir / f"{args.metric}_topology_physics_report.txt"
    rep.write_text("\n".join(lines))

    print(f"Wrote: {m_csv}")
    print(f"Wrote: {d_csv}")
    print(f"Wrote: {rep}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
