#!/usr/bin/env python3
"""Generate qualitative tie-break case-study panels for PSNR/SSIM."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence

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


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        # still write headerless empty file marker for traceability
        path.write_text("\n")
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _topology_score(zpd: float, zmt: float, mode: str) -> float:
    if mode == "l2":
        return float(math.sqrt(zpd * zpd + zmt * zmt))
    return float(max(zpd, zmt))


def _build_candidates(
    delta_rows: List[Dict[str, str]],
    metric: str,
    score_mode: str,
) -> List[Dict[str, object]]:
    if not delta_rows:
        return []

    sample_col = _pick_col(delta_rows[0], ["sample_idx", "sample", "sidx", "index", "idx"])
    if sample_col is None:
        raise SystemExit("Delta CSV missing sample index column")

    dmetric_col = f"delta_{metric}"
    if dmetric_col not in delta_rows[0]:
        raise SystemExit(f"Delta CSV missing required column: {dmetric_col}")

    pd_vals = np.asarray([_to_float(r.get("delta_pd_distance")) for r in delta_rows], dtype=float)
    mt_vals = np.asarray([_to_float(r.get("delta_mt_distance")) for r in delta_rows], dtype=float)
    pd_std = float(np.nanstd(np.abs(pd_vals)) + 1e-12)
    mt_std = float(np.nanstd(np.abs(mt_vals)) + 1e-12)

    out: List[Dict[str, object]] = []
    for r in delta_rows:
        si = _to_float(r.get(sample_col))
        dm = _to_float(r.get(dmetric_col))
        dpd = _to_float(r.get("delta_pd_distance"))
        dmt = _to_float(r.get("delta_mt_distance"))
        if None in (si, dm, dpd, dmt):
            continue
        zpd = abs(float(dpd)) / pd_std
        zmt = abs(float(dmt)) / mt_std
        topo_consistent = float(dpd) * float(dmt) > 0.0
        out.append(
            {
                "sample_idx": int(si),
                "delta_metric": float(dm),
                "abs_delta_metric": abs(float(dm)),
                "delta_pd": float(dpd),
                "delta_mt": float(dmt),
                "z_delta_pd": float(zpd),
                "z_delta_mt": float(zmt),
                "topology_score": _topology_score(zpd, zmt, score_mode),
                "topology_consistent": topo_consistent,
            }
        )

    out.sort(key=lambda r: (-float(r["topology_score"]), float(r["abs_delta_metric"]), int(r["sample_idx"])))
    return out


def _filtered_candidates(
    ranked: List[Dict[str, object]],
    tie_thresh: float,
    require_consistent: bool,
    explicit_samples: Optional[Sequence[int]],
) -> List[Dict[str, object]]:
    explicit_set = set(explicit_samples or [])
    out: List[Dict[str, object]] = []
    for r in ranked:
        if float(r["abs_delta_metric"]) > tie_thresh:
            continue
        if require_consistent and not bool(r["topology_consistent"]):
            continue
        if explicit_set and int(r["sample_idx"]) not in explicit_set:
            continue
        out.append(r)
    out.sort(key=lambda r: (-float(r["topology_score"]), float(r["abs_delta_metric"]), int(r["sample_idx"])))
    return out


def _plot_panel(
    out_png: Path,
    si: int,
    cnn_row: Dict[str, str],
    gan_row: Dict[str, str],
    cnn_gt: np.ndarray,
    cnn_sr: np.ndarray,
    gan_sr: np.ndarray,
    case: Dict[str, object],
    metric: str,
    label_mode: str,
) -> None:
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
        topo_better = "cnn"
    elif float(case["delta_pd"]) > 0 and float(case["delta_mt"]) > 0:
        topo_better = "gan"

    fig.suptitle(
        f"[{label_mode}] sample={si} | Δ{metric.upper()}={float(case['delta_metric']):.4f} | "
        f"ΔPD={float(case['delta_pd']):.4f} (z={float(case['z_delta_pd']):.2f}) | "
        f"ΔMT={float(case['delta_mt']):.4f} (z={float(case['z_delta_mt']):.2f}) | "
        f"topology_score={float(case['topology_score']):.2f} | topology-closer={topo_better}\n"
        f"CNN: {_annot(cnn_row, metric)}\nGAN: {_annot(gan_row, metric)}",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def _threshold_tag(v: float) -> str:
    return f"{v:.3f}".replace(".", "p")


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
    ap.add_argument("--tie-thresh", type=float, default=None, help="Require |Δmetric| <= tie threshold")
    ap.add_argument("--topology-score", choices=["max", "l2"], default="max", help="Topology ranking score")
    ap.add_argument("--require-consistent-topology", action="store_true", help="Require sign(ΔPD) == sign(ΔMT)")
    ap.add_argument("--report-thresholds", default="0.05,0.075,0.10", help="Comma-separated thresholds for summary/exports")
    args = ap.parse_args()

    merged_rows = _read_csv(args.merged_csv)
    delta_rows = _read_csv(args.delta_csv)
    by_sample = _group_by_sample(merged_rows)

    ranked = _build_candidates(delta_rows, args.metric, args.topology_score)
    if not ranked:
        raise SystemExit("No valid candidate rows in delta CSV")

    report_thresholds = [float(x.strip()) for x in args.report_thresholds.split(",") if x.strip()]
    tie_thresh = args.tie_thresh
    if tie_thresh is None:
        tie_thresh = 0.075 if args.metric == "ssim" else 0.25

    all_ranked_csv = args.outdir / f"{args.metric}_tie_break_all_ranked_candidates.csv"
    _write_csv(all_ranked_csv, ranked)

    filtered_for_thresh: Dict[float, List[Dict[str, object]]] = {}
    for th in report_thresholds:
        frows = _filtered_candidates(ranked, tie_thresh=th, require_consistent=args.require_consistent_topology, explicit_samples=None)
        filtered_for_thresh[th] = frows
        _write_csv(args.outdir / f"{args.metric}_tie_break_candidates_thresh_{_threshold_tag(th)}.csv", frows)

    selected = _filtered_candidates(
        ranked,
        tie_thresh=tie_thresh,
        require_consistent=args.require_consistent_topology,
        explicit_samples=args.samples,
    )
    selected = selected[: args.top_k] if not args.samples else selected

    top_cases_csv = args.outdir / f"{args.metric}_tie_break_top_cases.csv"
    summary_txt = args.outdir / f"{args.metric}_tie_break_summary.txt"
    case_dir = args.outdir / "tie_break_cases" / args.metric

    abs_dm = np.asarray([float(r["abs_delta_metric"]) for r in ranked], dtype=float)
    consistent_count = sum(bool(r["topology_consistent"]) for r in ranked)
    lines: List[str] = [f"=== {args.metric.upper()} tie-break case summary ==="]
    lines.append(f"total_paired_samples={len(ranked)}")
    lines.append(
        f"abs_delta_{args.metric}: min={float(np.min(abs_dm)):.6f}, median={float(np.median(abs_dm)):.6f}, max={float(np.max(abs_dm)):.6f}"
    )
    lines.append(f"topology_consistent_count={consistent_count} / {len(ranked)}")
    for th in report_thresholds:
        base = sum(float(r["abs_delta_metric"]) <= th for r in ranked)
        cons = sum(float(r["abs_delta_metric"]) <= th and bool(r["topology_consistent"]) for r in ranked)
        lines.append(f"count(|Δ{args.metric.upper()}| <= {th:.3f})={base}")
        lines.append(f"count(|Δ{args.metric.upper()}| <= {th:.3f} and consistent_topology)={cons}")

    strict = 0.05
    strict_pass = sum(float(r["abs_delta_metric"]) <= strict for r in ranked)
    if args.metric == "ssim" and strict_pass == 0:
        lines.append("NOTE: No rows pass strict SSIM tie threshold (0.05).")
        lines.append("Interpret selected rows as close-SSIM topology-separation / agreement cases, not strict tie-breaks.")

    label_mode = "tie-break"
    if args.metric == "ssim" and strict_pass == 0:
        label_mode = "close-SSIM topology-separation"

    rows_for_csv: List[Dict[str, object]] = []
    if not selected:
        lines.append("No selected cases after tie threshold + consistency filters.")
        _write_csv(top_cases_csv, rows_for_csv)
        summary_txt.write_text("\n".join(lines) + "\n")
        print(f"Wrote: {all_ranked_csv}")
        print(f"Wrote: {top_cases_csv}")
        print(f"Wrote: {summary_txt}")
        return 0

    cnn_gt = np.load(args.cnn_dir / "dataGT.npy", mmap_mode="r")
    cnn_sr = np.load(args.cnn_dir / "dataSR.npy", mmap_mode="r")
    gan_sr = np.load(args.gan_dir / "dataSR.npy", mmap_mode="r")

    for rank, case in enumerate(selected, start=1):
        si = int(case["sample_idx"])
        rec = by_sample.get(si, {})
        cnn_row = rec.get("cnn")
        gan_row = rec.get("gan")
        if cnn_row is None or gan_row is None:
            continue

        gt_spd = _speed(_extract_patch(np.asarray(cnn_gt[si]), args.patch, args.x0, args.y0))
        cnn_spd = _speed(_extract_patch(np.asarray(cnn_sr[si]), args.patch, args.x0, args.y0))
        gan_spd = _speed(_extract_patch(np.asarray(gan_sr[si]), args.patch, args.x0, args.y0))

        out_png = case_dir / f"{args.metric}_tie_case_rank{rank:02d}_s{si}.png"
        _plot_panel(out_png, si, cnn_row, gan_row, gt_spd, cnn_spd, gan_spd, case, args.metric, label_mode)

        pref = "mixed"
        if float(case["delta_pd"]) < 0 and float(case["delta_mt"]) < 0:
            pref = "cnn"
        elif float(case["delta_pd"]) > 0 and float(case["delta_mt"]) > 0:
            pref = "gan"
        lines.append(
            f"rank {rank} sample {si}: |Δ{args.metric.upper()}|={float(case['abs_delta_metric']):.4f}, "
            f"ΔPD={float(case['delta_pd']):.4f}, ΔMT={float(case['delta_mt']):.4f}, "
            f"zΔPD={float(case['z_delta_pd']):.2f}, zΔMT={float(case['z_delta_mt']):.2f}, "
            f"score={float(case['topology_score']):.2f}, consistent={bool(case['topology_consistent'])}, topology-closer={pref}"
        )

        row = dict(case)
        row["rank"] = rank
        row["panel_png"] = str(out_png)
        row["topology_closer"] = pref
        row["label_mode"] = label_mode
        rows_for_csv.append(row)

    _write_csv(top_cases_csv, rows_for_csv)
    summary_txt.write_text("\n".join(lines) + "\n")

    print(f"Wrote: {all_ranked_csv}")
    for th in report_thresholds:
        print(f"Wrote: {args.outdir / f'{args.metric}_tie_break_candidates_thresh_{_threshold_tag(th)}.csv'}")
    print(f"Wrote: {top_cases_csv}")
    print(f"Wrote: {summary_txt}")
    print(f"Wrote case panels under: {case_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
