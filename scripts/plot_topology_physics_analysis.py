#!/usr/bin/env python3
"""Visualize merged topology/physics analysis tables.

Focus: physics/topology heatmaps, paired-delta scatter grid, and tie-break table.
(SSIM-vs-topology standalone plots are handled by analyze_psnr_vs_ttk_topology.py)
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception as e:  # pragma: no cover
    raise SystemExit(f"matplotlib required: {e}")

BASE_COLS = {"method", "key", "sample_idx", "psnr", "ssim", "pd_distance", "mt_distance"}


def _to_float(x: object) -> Optional[float]:
    try:
        v = float(str(x).strip())
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def _num_cols(rows: List[Dict[str, str]]) -> List[str]:
    if not rows:
        return []
    cols: List[str] = []
    for k in rows[0].keys():
        if k in {"method", "key"}:
            continue
        vals = [_to_float(r.get(k)) for r in rows]
        if sum(v is not None for v in vals) >= 2:
            cols.append(k)
    return cols


def _arr(rows: List[Dict[str, str]], col: str) -> np.ndarray:
    vals = [_to_float(r.get(col)) for r in rows]
    clean = [v for v in vals if v is not None]
    return np.asarray(clean, dtype=float)


def _paired_arrays(rows: List[Dict[str, str]], x_col: str, y_col: str) -> Tuple[np.ndarray, np.ndarray]:
    xs: List[float] = []
    ys: List[float] = []
    for r in rows:
        xv = _to_float(r.get(x_col))
        yv = _to_float(r.get(y_col))
        if xv is None or yv is None:
            continue
        xs.append(xv)
        ys.append(yv)
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _matrix(rows: List[Dict[str, str]], physics_cols: Sequence[str]) -> Tuple[np.ndarray, List[str], List[str]]:
    targets = ["psnr", "ssim", "pd_distance", "mt_distance"]
    m = np.zeros((len(physics_cols), len(targets)), dtype=float)
    m[:] = np.nan
    for i, pcol in enumerate(physics_cols):
        for j, t in enumerate(targets):
            xt, yp = _paired_arrays(rows, t, pcol)
            m[i, j] = _corr(xt, yp)
    return m, list(physics_cols), targets


def _plot_heatmap(mat: np.ndarray, ylabels: Sequence[str], xlabels: Sequence[str], title: str, out: Path) -> None:
    h = max(4.5, 0.28 * len(ylabels) + 1.5)
    fig, ax = plt.subplots(figsize=(8, h))
    im = ax.imshow(mat, aspect="auto", cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_xticklabels(xlabels)
    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_yticklabels(ylabels)
    ax.set_title(title)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            txt = "nan" if math.isnan(v) else f"{v:.2f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025)
    cbar.set_label("Pearson r")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220)
    plt.close(fig)


def _plot_delta_scatters(delta_rows: List[Dict[str, str]], physics_cols: Sequence[str], out: Path, top_n: int = 9) -> None:
    candidates = [c for c in physics_cols if c in {"wpd_rmse", "wpd_mae", "grad_mae", "psd_log_l2"}]
    if not candidates:
        candidates = list(physics_cols[:top_n])
    fig, axes = plt.subplots(len(candidates), 2, figsize=(10, max(4.5, 2.7 * len(candidates))), squeeze=False)
    for i, c in enumerate(candidates):
        x1, y1 = _paired_arrays(delta_rows, "delta_pd_distance", f"delta_{c}")
        x2, y2 = _paired_arrays(delta_rows, "delta_mt_distance", f"delta_{c}")
        axes[i, 0].scatter(x1, y1, s=36)
        axes[i, 0].set_xlabel("ΔPD")
        axes[i, 0].set_ylabel(f"Δ{c}")
        axes[i, 0].set_title(f"ΔPD vs Δ{c}")
        axes[i, 0].grid(True, linestyle="--", alpha=0.3)

        axes[i, 1].scatter(x2, y2, s=36)
        axes[i, 1].set_xlabel("ΔMT")
        axes[i, 1].set_ylabel(f"Δ{c}")
        axes[i, 1].set_title(f"ΔMT vs Δ{c}")
        axes[i, 1].grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220)
    plt.close(fig)


def _top_tie_break_rows(rows: List[Dict[str, str]], metric_col: str, tie_eps: float, min_z: float, k: int) -> List[List[str]]:
    by_idx: Dict[int, List[Dict[str, str]]] = {}
    for r in rows:
        si = _to_float(r.get("sample_idx"))
        if si is None:
            continue
        by_idx.setdefault(int(si), []).append(r)

    pd = np.asarray([_to_float(r.get("pd_distance")) for r in rows if _to_float(r.get("pd_distance")) is not None], dtype=float)
    mt = np.asarray([_to_float(r.get("mt_distance")) for r in rows if _to_float(r.get("mt_distance")) is not None], dtype=float)
    pd_std = float(np.std(pd) + 1e-12)
    mt_std = float(np.std(mt) + 1e-12)

    out: List[Tuple[float, List[str]]] = []
    for si, rr in sorted(by_idx.items()):
        for i in range(len(rr)):
            for j in range(i + 1, len(rr)):
                a = rr[i]
                b = rr[j]
                qa = _to_float(a.get(metric_col))
                qb = _to_float(b.get(metric_col))
                pd_a = _to_float(a.get("pd_distance"))
                pd_b = _to_float(b.get("pd_distance"))
                mt_a = _to_float(a.get("mt_distance"))
                mt_b = _to_float(b.get("mt_distance"))
                if None in (qa, qb, pd_a, pd_b, mt_a, mt_b):
                    continue
                dq = abs(qa - qb)
                if dq > tie_eps:
                    continue
                zpd = abs(pd_a - pd_b) / pd_std
                zmt = abs(mt_a - mt_b) / mt_std
                score = max(zpd, zmt)
                if score < min_z:
                    continue
                out.append((score, [str(si), str(a.get("method", "?")), str(b.get("method", "?")), f"{dq:.3f}", f"{zpd:.2f}", f"{zmt:.2f}"]))
    out.sort(key=lambda x: x[0], reverse=True)
    return [x[1] for x in out[:k]]


def _plot_tie_table(rows: List[List[str]], metric_col: str, out: Path) -> None:
    headers = ["sample", "method_a", "method_b", f"Δ{metric_col.upper()}", "zΔPD", "zΔMT"]
    fig, ax = plt.subplots(figsize=(8, max(2.4, 0.5 * (len(rows) + 1))))
    ax.axis("off")
    table = ax.table(cellText=rows if rows else [["-", "-", "-", "-", "-", "-"]], colLabels=headers, loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.3)
    ax.set_title(f"{metric_col.upper()}-tie / topology-break candidates")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220)
    plt.close(fig)


def _write_tie_csv(rows: List[List[str]], metric_col: str, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    headers = ["sample", "method_a", "method_b", f"delta_{metric_col}", "z_delta_pd", "z_delta_mt"]
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for r in rows:
            w.writerow(r)


def main() -> int:
    ap = argparse.ArgumentParser(description="Visualize merged topology/physics analysis outputs")
    ap.add_argument("--merged-csv", type=Path, required=True, help="CSV from analysis_compare.py --out-csv")
    ap.add_argument("--delta-csv", type=Path, default=None, help="Optional CSV from analysis_compare.py --out-delta-csv")
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--metric-column", choices=["psnr", "ssim"], default="psnr", help="Metric used for tie/break table")
    ap.add_argument("--tie-eps", type=float, default=0.25)
    ap.add_argument("--topo-gap-z", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--skip-heatmaps", action="store_true", help="Skip generic joint heatmaps")
    ap.add_argument("--only-heatmaps", action="store_true", help="Generate only generic joint heatmaps")
    args = ap.parse_args()

    rows = _read_csv(args.merged_csv)
    if not rows:
        raise SystemExit(f"No rows in {args.merged_csv}")

    physics_cols = [c for c in _num_cols(rows) if c not in BASE_COLS and not c.endswith("_gt") and not c.endswith("_sr")]
    methods = sorted({str(r.get("method", "")) for r in rows if str(r.get("method", "")).strip()})

    if not args.skip_heatmaps:
        mat, ylabels, xlabels = _matrix(rows, physics_cols)
        _plot_heatmap(mat, ylabels, xlabels, "Global correlations: physics vs PSNR/SSIM/PD/MT", args.outdir / "corr_heatmap_global.png")
        print(f"Wrote: {args.outdir/'corr_heatmap_global.png'}")
        for m in methods:
            sub = [r for r in rows if str(r.get("method", "")) == m]
            if len(sub) < 2:
                continue
            mm, yy, xx = _matrix(sub, physics_cols)
            p = args.outdir / f"corr_heatmap_{m}.png"
            _plot_heatmap(mm, yy, xx, f"{m.upper()} correlations: physics vs PSNR/SSIM/PD/MT", p)
            print(f"Wrote: {p}")

    if args.only_heatmaps:
        return 0

    tie_rows = _top_tie_break_rows(rows, metric_col=args.metric_column, tie_eps=args.tie_eps, min_z=args.topo_gap_z, k=args.top_k)
    tie_png = args.outdir / f"{args.metric_column}_tie_topology_break_table.png"
    _plot_tie_table(tie_rows, metric_col=args.metric_column, out=tie_png)
    print(f"Wrote: {tie_png}")

    tie_csv = args.outdir.parent / f"{args.metric_column}_tie_break_candidates.csv"
    _write_tie_csv(tie_rows, metric_col=args.metric_column, out=tie_csv)
    print(f"Wrote: {tie_csv}")

    if args.delta_csv is not None and args.delta_csv.exists():
        drows = _read_csv(args.delta_csv)
        if drows:
            p = args.outdir / f"{args.metric_column}_paired_delta_scatter_grid.png"
            _plot_delta_scatters(drows, physics_cols, p)
            print(f"Wrote: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
