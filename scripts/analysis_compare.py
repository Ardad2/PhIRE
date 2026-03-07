#!/usr/bin/env python3
"""Compare quality metric, topology distances, and physics metrics.

Two modes:
1) Build merged table from topology CSV + method data dirs (original mode).
2) Re-analyze an existing merged CSV (new mode via --in-csv), useful for SSIM report parity.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from metrics_physics import compute_physics_metrics


BASE_NUMERIC_COLS = {"sample_idx", "psnr", "ssim", "pd_distance", "mt_distance"}


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


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return pearson(rankdata(x), rankdata(y))


def bootstrap_corr_ci(x: np.ndarray, y: np.ndarray, kind: str = "pearson", nboot: int = 2000, seed: int = 0) -> Tuple[float, float]:
    if len(x) < 3:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    stats: List[float] = []
    fn = pearson if kind == "pearson" else spearman
    for _ in range(nboot):
        idx = rng.integers(0, len(x), len(x))
        v = fn(x[idx], y[idx])
        if not math.isnan(v):
            stats.append(v)
    if not stats:
        return float("nan"), float("nan")
    arr = np.asarray(stats, dtype=float)
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


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


def _numeric_columns(rows: List[Dict[str, object]]) -> List[str]:
    if not rows:
        return []
    out: List[str] = []
    for k in rows[0].keys():
        if k in {"method", "key"}:
            continue
        vals = [_to_float(r.get(k)) for r in rows]
        if sum(v is not None for v in vals) >= 2:
            out.append(k)
    return out


def _arr(rows: List[Dict[str, object]], col: str) -> np.ndarray:
    vals = [_to_float(r.get(col)) for r in rows]
    clean = [v for v in vals if v is not None]
    return np.asarray(clean, dtype=float)


def _paired_delta_rows(rows: List[Dict[str, object]], methods: Sequence[str]) -> List[Dict[str, object]]:
    if len(methods) < 2:
        return []
    a, b = methods[0], methods[1]
    by: Dict[int, Dict[str, Dict[str, object]]] = {}
    for r in rows:
        si = int(r["sample_idx"])
        by.setdefault(si, {})[str(r["method"])] = r

    numeric_cols = _numeric_columns(rows)
    deltas: List[Dict[str, object]] = []
    for si in sorted(by.keys()):
        m = by[si]
        if a not in m or b not in m:
            continue
        ra, rb = m[a], m[b]
        row: Dict[str, object] = {"sample_idx": si, "method_a": a, "method_b": b}
        for c in numeric_cols:
            va = _to_float(ra.get(c))
            vb = _to_float(rb.get(c))
            if va is None or vb is None:
                continue
            row[f"delta_{c}"] = float(va - vb)
        deltas.append(row)
    return deltas


def _corr_line(x: np.ndarray, y: np.ndarray, label: str, nboot: int, seed: int) -> str:
    if len(x) < 3 or len(y) < 3:
        return f"{label}: n<3"
    n = min(len(x), len(y))
    xx = x[:n]
    yy = y[:n]
    p = pearson(xx, yy)
    s = spearman(xx, yy)
    plo, phi = bootstrap_corr_ci(xx, yy, "pearson", nboot=nboot, seed=seed)
    slo, shi = bootstrap_corr_ci(xx, yy, "spearman", nboot=nboot, seed=seed + 17)
    return f"{label}: pearson={p:.4f} CI[{plo:.4f},{phi:.4f}] spearman={s:.4f} CI[{slo:.4f},{shi:.4f}]"


def summarize(rows: List[Dict[str, object]], metric_column: str = "psnr", tie_eps: float = 0.25, topo_gap_z: float = 1.0, nboot: int = 2000, seed: int = 0) -> Tuple[str, List[Dict[str, object]]]:
    if not rows:
        return "No merged rows", []
    if metric_column not in rows[0]:
        raise ValueError(f"metric-column {metric_column} not found in merged rows")

    methods = sorted({str(r["method"]) for r in rows})
    physics_cols = [c for c in _numeric_columns(rows) if c not in BASE_NUMERIC_COLS and not c.endswith("_gt") and not c.endswith("_sr")]

    mtag = metric_column.upper()
    lines: List[str] = []
    lines.append("=== CONFIG ===")
    lines.append(f"rows={len(rows)} methods={methods}")
    lines.append(f"metric_column={metric_column}")
    lines.append(f"physics_columns={len(physics_cols)}")
    lines.append("")

    lines.append(f"=== GLOBAL correlations (physics vs {mtag}/PD/MT) ===")
    for col in physics_cols:
        vals = _arr(rows, col)
        q = _arr(rows, metric_column)
        pd = _arr(rows, "pd_distance")
        mt = _arr(rows, "mt_distance")
        lines.append(_corr_line(q, vals, f"{col} :: {mtag}", nboot, seed))
        lines.append(_corr_line(pd, vals, f"{col} :: PD", nboot, seed + 1))
        lines.append(_corr_line(mt, vals, f"{col} :: MT", nboot, seed + 2))
    lines.append("")

    lines.append("=== PER-METHOD correlations ===")
    for m in methods:
        sub = [r for r in rows if str(r["method"]) == m]
        lines.append(f"[{m}] n={len(sub)}")
        for col in physics_cols:
            vals = _arr(sub, col)
            lines.append(_corr_line(_arr(sub, metric_column), vals, f"  {col} :: {mtag}", nboot, seed + 3))
            lines.append(_corr_line(_arr(sub, "pd_distance"), vals, "  {col} :: PD".format(col=col), nboot, seed + 4))
            lines.append(_corr_line(_arr(sub, "mt_distance"), vals, "  {col} :: MT".format(col=col), nboot, seed + 5))
        lines.append("")

    delta_rows = _paired_delta_rows(rows, methods)
    if delta_rows:
        dmetric = f"delta_{metric_column}"
        lines.append(f"=== PAIRED-DELTA correlations ({methods[0]} - {methods[1]}) ===")
        for col in physics_cols:
            dphys = _arr(delta_rows, f"delta_{col}")
            lines.append(_corr_line(_arr(delta_rows, dmetric), dphys, f"delta_{col} :: d{mtag}", nboot, seed + 6))
            lines.append(_corr_line(_arr(delta_rows, "delta_pd_distance"), dphys, f"delta_{col} :: dPD", nboot, seed + 7))
            lines.append(_corr_line(_arr(delta_rows, "delta_mt_distance"), dphys, f"delta_{col} :: dMT", nboot, seed + 8))
        lines.append("")

    if len(methods) >= 2:
        lines.append(f"=== {mtag}-tie / topology-break candidates ===")
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
                    dq = abs(float(a[metric_column]) - float(b[metric_column]))
                    if dq > tie_eps:
                        continue
                    dpd = abs(float(a["pd_distance"]) - float(b["pd_distance"])) / pd_std
                    dmt = abs(float(a["mt_distance"]) - float(b["mt_distance"])) / mt_std
                    if max(dpd, dmt) >= topo_gap_z:
                        lines.append(f"s{si}: {a['method']} vs {b['method']} | Δ{mtag}={dq:.4f} | zΔPD={dpd:.2f}, zΔMT={dmt:.2f}")

    return "\n".join(lines), delta_rows


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def read_merged_csv(path: Path) -> List[Dict[str, object]]:
    with path.open("r", newline="") as f:
        r = csv.DictReader(f)
        return [dict(row) for row in r]


def parse_method_dirs(values: List[str]) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for x in values:
        if "=" not in x:
            raise ValueError(f"Expected method=dir, got: {x}")
        m, d = x.split("=", 1)
        out[m.strip().lower()] = Path(d).expanduser().resolve()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare quality metric + topology + physics metrics")
    ap.add_argument("--topology-csv", default="", help="Build mode: topology CSV")
    ap.add_argument("--method-dir", action="append", default=[], help="Build mode: method=/path/to/data_out_dir")
    ap.add_argument("--in-csv", default="", help="Analysis mode: existing merged CSV")
    ap.add_argument("--metric-column", choices=["psnr", "ssim"], default="psnr", help="Metric used in report/tie-break")
    ap.add_argument("--out-csv", default="", help="Optional output merged CSV path")
    ap.add_argument("--out-report", required=True)
    ap.add_argument("--out-delta-csv", default="", help="Optional CSV path for paired-delta table")
    ap.add_argument("--tie-eps", type=float, default=0.25)
    ap.add_argument("--topo-gap-z", type=float, default=1.0)
    ap.add_argument("--bootstrap", type=int, default=2000, help="Bootstrap iterations")
    ap.add_argument("--nboot", type=int, default=None, help="Alias for --bootstrap")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.in_csv:
        merged = read_merged_csv(Path(args.in_csv))
        if args.out_csv:
            write_csv(Path(args.out_csv), merged)
    else:
        if not args.topology_csv:
            raise SystemExit("Either --in-csv or --topology-csv must be provided")
        if not args.method_dir:
            raise SystemExit("Build mode requires one or more --method-dir")
        topo = load_topology(Path(args.topology_csv))
        methods = parse_method_dirs(args.method_dir)
        metrics = build_metric_rows(methods)
        merged = merge_rows(topo, metrics)
        if args.out_csv:
            write_csv(Path(args.out_csv), merged)

    nboot = args.bootstrap if args.nboot is None else args.nboot
    report, delta_rows = summarize(merged, metric_column=args.metric_column, tie_eps=args.tie_eps, topo_gap_z=args.topo_gap_z, nboot=nboot, seed=args.seed)
    Path(args.out_report).write_text(report)

    if args.out_delta_csv:
        write_csv(Path(args.out_delta_csv), delta_rows)
        print(f"Wrote: {args.out_delta_csv}")
    if args.out_csv:
        print(f"Wrote: {args.out_csv}")
    print(f"Wrote: {args.out_report}")


if __name__ == "__main__":
    main()
