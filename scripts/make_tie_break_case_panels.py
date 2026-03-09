#!/usr/bin/env python3
"""Generate qualitative tie-break case-study panels for PSNR/SSIM.

Selects samples where quality metric deltas are small but topology deltas are large,
and renders GT/CNN/GAN speed/error panels with metric annotations.
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


def _pick_col(row: Dict[str, str], names: Sequence[str]) -> Optional[str]:
    for n in names:
        if n in row:
            return n
    return None


def _group_by_sample(merged_rows: List[Dict[str, str]]) -> Dict[int, Dict[str, Dict[str, str]]]:
    out: Dict[int, Dict[str, Dict[str, str]]] = {}
    for r in merged_rows:
        idx_col = _pick_col(r, ["sample_idx", "sample", "sidx", "index", "idx"])
        if idx_col is None:
            continue
        si = _to_float(r.get(idx_col))
        if si is None:
            continue
        m = str(r.get("method", "")).strip().lower()
        if not m:
            continue
        out.setdefault(int(si), {})[m] = r
    return out


def _score_candidates(delta_rows: List[Dict[str, str]], metric: str, top_k: int, explicit_samples: Optional[Sequence[int]]) -> List[Dict[str, object]]:
    if not delta_rows:
        return []

    sample_col = _pick_col(delta_rows[0], ["sample_idx", "sample", "sidx", "index", "idx"])
    if sample_col is None:
        raise SystemExit("Delta CSV missing sample index column")

    dmetric_col = f"delta_{metric}"
    if dmetric_col not in delta_rows[0]:
        raise SystemExit(f"Delta CSV missing required column: {dmetric_col}")

    dpd = np.asarray([abs(_to_float(r.get("delta_pd_distance")) or np.nan) for r in delta_rows], dtype=float)
    dmt = np.asarray([abs(_to_float(r.get("delta_mt_distance")) or np.nan) for r in delta_rows], dtype=float)
    pd_std = float(np.nanstd(dpd) + 1e-12)
    mt_std = float(np.nanstd(dmt) + 1e-12)

    out: List[Dict[str, object]] = []
    explicit_set = set(explicit_samples or [])
    for r in delta_rows:
        si = _to_float(r.get(sample_col))
        dm = _to_float(r.get(dmetric_col))
        dpd_v = _to_float(r.get("delta_pd_distance"))
        dmt_v = _to_float(r.get("delta_mt_distance"))
        if None in (si, dm, dpd_v, dmt_v):
            continue
        zpd = abs(dpd_v) / pd_std
        zmt = abs(dmt_v) / mt_std
        topo_signal = max(zpd, zmt)
        # prioritize strong topology disagreement under tied quality metric
        score = topo_signal / (abs(dm) + 1e-6)
        chosen = int(si) in explicit_set
        out.append(
            {
                "sample_idx": int(si),
                "delta_metric": float(dm),
                "delta_pd": float(dpd_v),
                "delta_mt": float(dmt_v),
                "z_delta_pd": float(zpd),
                "z_delta_mt": float(zmt),
                "topology_signal": float(topo_signal),
                "selection_score": float(score),
                "selected_by_explicit": chosen,
            }
        )

    if explicit_set:
        out = [r for r in out if r["sample_idx"] in explicit_set]
    out.sort(key=lambda r: (not bool(r["selected_by_explicit"]), -float(r["selection_score"])))
    return out[:top_k] if not explicit_set else out


def _speed(arr_uv: np.ndarray) -> np.ndarray:
    return np.sqrt(arr_uv[..., 0] ** 2 + arr_uv[..., 1] ** 2)


def _gradmag(a: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(a)
    return np.sqrt(gx * gx + gy * gy)


def _extract_patch(a: np.ndarray, patch: int, x0: int, y0: int) -> np.ndarray:
    return a[y0 : y0 + patch, x0 : x0 + patch, :]


def _annot(row: Dict[str, str], metric: str) -> str:
    vals = []
    for c in [metric, "psnr", "ssim", "pd_distance", "mt_distance", "wpd_rmse", "grad_mae", "psd_log_l2"]:
        if c in row and _to_float(row.get(c)) is not None:
            vals.append(f"{c}={float(row[c]):.4f}")
    return ", ".join(vals)


def _plot_panel(out_png: Path, si: int, cnn_row: Dict[str, str], gan_row: Dict[str, str], cnn_gt: np.ndarray, cnn_sr: np.ndarray, gan_sr: np.ndarray, case: Dict[str, object], metric: str) -> None:
    err_cnn = np.abs(cnn_gt - cnn_sr)
    err_gan = np.abs(cnn_gt - gan_sr)
    g_gt = _gradmag(cnn_gt)
    g_cnn = _gradmag(cnn_sr)
    g_gan = _gradmag(gan_sr)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    ims = [
        (cnn_gt, "GT speed", "viridis"),
        (cnn_sr, "CNN SR speed", "viridis"),
        (gan_sr, "GAN SR speed", "viridis"),
        (np.abs(cnn_sr - gan_sr), "|CNN-GAN| speed", "magma"),
        (err_cnn, "|GT-CNN|", "magma"),
        (err_gan, "|GT-GAN|", "magma"),
        (g_gt, "|∇GT|", "cividis"),
        (np.abs(g_cnn - g_gan), "|∇CNN-∇GAN|", "cividis"),
    ]
    for ax, (img, ttl, cmap) in zip(axes.ravel(), ims):
        h = ax.imshow(img, cmap=cmap)
        ax.set_title(ttl)
        ax.axis("off")
        fig.colorbar(h, ax=ax, fraction=0.046, pad=0.04)

    topo_better = "mixed"
    if float(case["delta_pd"]) < 0 and float(case["delta_mt"]) < 0:
        topo_better = "gan"
    elif float(case["delta_pd"]) > 0 and float(case["delta_mt"]) > 0:
        topo_better = "cnn"

    fig.suptitle(
        f"sample={si} | Δ{metric.upper()}={float(case['delta_metric']):.4f} | "
        f"ΔPD={float(case['delta_pd']):.4f} (z={float(case['z_delta_pd']):.2f}) | "
        f"ΔMT={float(case['delta_mt']):.4f} (z={float(case['z_delta_mt']):.2f}) | topology-closer={topo_better}\n"
        f"CNN: {_annot(cnn_row, metric)}\nGAN: {_annot(gan_row, metric)}",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def _write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="Build qualitative tie-break case-study panels")
    ap.add_argument("--metric", choices=["psnr", "ssim"], required=True)
    ap.add_argument("--merged-csv", type=Path, required=True)
    ap.add_argument("--delta-csv", type=Path, required=True)
    ap.add_argument("--cnn-dir", type=Path, required=True)
    ap.add_argument("--gan-dir", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, default=Path("ttk_runs/combined"))
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--samples", nargs="*", type=int, default=None)
    ap.add_argument("--patch", type=int, default=160)
    ap.add_argument("--x0", type=int, default=0)
    ap.add_argument("--y0", type=int, default=0)
    args = ap.parse_args()

    merged_rows = _read_csv(args.merged_csv)
    delta_rows = _read_csv(args.delta_csv)
    by_sample = _group_by_sample(merged_rows)
    selected = _score_candidates(delta_rows, args.metric, args.top_k, args.samples)
    if not selected:
        raise SystemExit("No tie-break candidates selected")

    cnn_gt = np.load(args.cnn_dir / "dataGT.npy", mmap_mode="r")
    cnn_sr = np.load(args.cnn_dir / "dataSR.npy", mmap_mode="r")
    gan_sr = np.load(args.gan_dir / "dataSR.npy", mmap_mode="r")

    case_dir = args.outdir / "tie_break_cases" / args.metric
    top_cases_csv = args.outdir / f"{args.metric}_tie_break_top_cases.csv"
    summary_txt = args.outdir / f"{args.metric}_tie_break_summary.txt"

    summary_lines: List[str] = [f"=== {args.metric.upper()} tie-break case summary ==="]
    rows_for_csv: List[Dict[str, object]] = []

    for rank, c in enumerate(selected, start=1):
        si = int(c["sample_idx"])
        rec = by_sample.get(si, {})
        cnn_row = rec.get("cnn")
        gan_row = rec.get("gan")
        if cnn_row is None or gan_row is None:
            continue

        gt_spd = _speed(_extract_patch(np.asarray(cnn_gt[si]), args.patch, args.x0, args.y0))
        cnn_spd = _speed(_extract_patch(np.asarray(cnn_sr[si]), args.patch, args.x0, args.y0))
        gan_spd = _speed(_extract_patch(np.asarray(gan_sr[si]), args.patch, args.x0, args.y0))

        out_png = case_dir / f"{args.metric}_tie_case_rank{rank:02d}_s{si}.png"
        _plot_panel(out_png, si, cnn_row, gan_row, gt_spd, cnn_spd, gan_spd, c, args.metric)

        preferred = "gan" if float(c["delta_pd"]) < 0 and float(c["delta_mt"]) < 0 else "cnn" if float(c["delta_pd"]) > 0 and float(c["delta_mt"]) > 0 else "mixed"
        reason = (
            f"small |Δ{args.metric.upper()}|={abs(float(c['delta_metric'])):.4f} with "
            f"large topology gap max(zΔPD,zΔMT)={float(c['topology_signal']):.2f}"
        )
        summary_lines.append(
            f"rank {rank} sample {si}: {reason}; ΔPD={float(c['delta_pd']):.4f}, ΔMT={float(c['delta_mt']):.4f}, "
            f"zΔPD={float(c['z_delta_pd']):.2f}, zΔMT={float(c['z_delta_mt']):.2f}, topology-closer={preferred}"
        )

        row = dict(c)
        row["rank"] = rank
        row["panel_png"] = str(out_png)
        row["topology_closer"] = preferred
        rows_for_csv.append(row)

    _write_csv(top_cases_csv, rows_for_csv)
    summary_txt.write_text("\n".join(summary_lines) + "\n")
    print(f"Wrote: {top_cases_csv}")
    print(f"Wrote: {summary_txt}")
    print(f"Wrote case panels under: {case_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
