#!/usr/bin/env python3
"""Run a post-processing near-tie validation study for PSNR/SSIM topology selection.

This script ONLY reads existing artifacts and model outputs; it does not run inference
or topology extraction.

Example (Spark-friendly):
  PYTHONNOUSERSITE=1 /usr/bin/python3 scripts/run_near_tie_study.py \
    --combined-csv ttk_runs/combined/combined_pairwise_results.csv \
    --merged-csv ttk_runs/combined/psnr_topology_physics_merged.csv \
    --cnn-dir data_out/wind_mrhr_cnn \
    --gan-dir data_out/wind_mrhr_gan \
    --out-root ttk_runs/near_tie_study
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


BASE_COLS = {
    "method",
    "key",
    "sample_idx",
    "sample",
    "sidx",
    "idx",
    "index",
    "psnr",
    "ssim",
    "pd_distance",
    "mt_distance",
}

PHYSICS_PREFERRED_LOW_BETTER = [
    "wpd_rmse",
    "wpd_mae",
    "psd_log_l2",
    "grad_mae",
    "w1_speed",
    "w1_grad",
]


@dataclass
class MethodRow:
    method: str
    sample_idx: int
    psnr: float
    ssim: float
    mt_distance: float
    pd_distance: float
    extras: Dict[str, float]


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


def _sample_idx(row: Dict[str, str]) -> Optional[int]:
    for k in ("sample_idx", "sample", "sidx", "idx", "index"):
        if k in row:
            v = _to_float(row.get(k))
            if v is not None:
                return int(v)
    key = str(row.get("key", ""))
    marker = "_s"
    if marker in key:
        i = key.find(marker)
        j = i + 2
        digits = []
        while j < len(key) and key[j].isdigit():
            digits.append(key[j])
            j += 1
        if digits:
            return int("".join(digits))
    return None


def _winner_low(a: float, b: float, names: Tuple[str, str] = ("cnn", "gan"), eps: float = 1e-12) -> str:
    if abs(a - b) <= eps:
        return "tie"
    return names[0] if a < b else names[1]


def _winner_high(a: float, b: float, names: Tuple[str, str] = ("cnn", "gan"), eps: float = 1e-12) -> str:
    if abs(a - b) <= eps:
        return "tie"
    return names[0] if a > b else names[1]


def _vote_winner(winners: Iterable[str], prefer: str = "tie") -> str:
    counts: Dict[str, int] = {}
    for w in winners:
        counts[w] = counts.get(w, 0) + 1
    cnn = counts.get("cnn", 0)
    gan = counts.get("gan", 0)
    if cnn > gan:
        return "cnn"
    if gan > cnn:
        return "gan"
    return prefer


def _parse_thresholds(x: str) -> List[float]:
    out: List[float] = []
    for t in x.split(","):
        t = t.strip()
        if not t:
            continue
        out.append(float(t))
    return out


def _speed(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 3 and arr.shape[-1] >= 2:
        return np.sqrt(arr[..., 0] ** 2 + arr[..., 1] ** 2)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        return arr[..., 0]
    return np.asarray(arr)


def _downsample_mean_2d(a: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    h, w = a.shape
    if h == out_h and w == out_w:
        return a
    if h % out_h == 0 and w % out_w == 0:
        fh = h // out_h
        fw = w // out_w
        return a.reshape(out_h, fh, out_w, fw).mean(axis=(1, 3))
    ys = np.linspace(0, h - 1, out_h).astype(int)
    xs = np.linspace(0, w - 1, out_w).astype(int)
    return a[np.ix_(ys, xs)]


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    d = a - b
    return float(np.mean(d * d))


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    m = _mse(a, b)
    if m == 0.0:
        return float("inf")
    dr = float(np.max(b) - np.min(b))
    if dr == 0.0:
        dr = 1.0
    return 20.0 * math.log10(dr) - 10.0 * math.log10(m)


def _exceed_frac(a: np.ndarray, thr: float) -> float:
    return float(np.mean(a >= thr))


def _load_rows(merged_csv: Path) -> Dict[int, Dict[str, MethodRow]]:
    rows = _read_csv(merged_csv)
    out: Dict[int, Dict[str, MethodRow]] = {}
    for r in rows:
        method = str(r.get("method", "")).strip().lower()
        if method not in {"cnn", "gan"}:
            continue
        si = _sample_idx(r)
        psnr = _to_float(r.get("psnr"))
        ssim = _to_float(r.get("ssim"))
        mt = _to_float(r.get("mt_distance"))
        pd = _to_float(r.get("pd_distance"))
        if si is None or psnr is None or ssim is None or mt is None or pd is None:
            continue
        extras: Dict[str, float] = {}
        for k, v in r.items():
            if k in BASE_COLS or k.endswith("_gt") or k.endswith("_sr"):
                continue
            fv = _to_float(v)
            if fv is None:
                continue
            extras[k] = fv
        out.setdefault(si, {})[method] = MethodRow(method=method, sample_idx=si, psnr=psnr, ssim=ssim, mt_distance=mt, pd_distance=pd, extras=extras)
    return out


def _pick_physics_cols(sample_rows: Dict[int, Dict[str, MethodRow]]) -> List[str]:
    cols = set()
    for by_method in sample_rows.values():
        for m in ("cnn", "gan"):
            row = by_method.get(m)
            if row is not None:
                cols.update(row.extras.keys())
    picked = [c for c in PHYSICS_PREFERRED_LOW_BETTER if c in cols]
    return picked


def _metrics_for_sample(
    si: int,
    cnn_row: MethodRow,
    gan_row: MethodRow,
    in_cnn: np.ndarray,
    sr_cnn: np.ndarray,
    gt_cnn: np.ndarray,
    sr_gan: np.ndarray,
    physics_cols: Sequence[str],
    tail_q: float,
    exceed_qs: Sequence[float],
) -> Dict[str, object]:
    metric_delta_psnr = abs(cnn_row.psnr - gan_row.psnr)
    metric_delta_ssim = abs(cnn_row.ssim - gan_row.ssim)

    mt_winner = _winner_low(cnn_row.mt_distance, gan_row.mt_distance)
    pd_winner = _winner_low(cnn_row.pd_distance, gan_row.pd_distance)

    lr_target = _speed(in_cnn[si])
    cnn_speed = _speed(sr_cnn[si])
    gan_speed = _speed(sr_gan[si])
    gt_speed = _speed(gt_cnn[si])

    lr_h, lr_w = lr_target.shape[:2]
    cnn_lr = _downsample_mean_2d(cnn_speed, lr_h, lr_w)
    gan_lr = _downsample_mean_2d(gan_speed, lr_h, lr_w)

    cnn_lr_mae = float(np.mean(np.abs(cnn_lr - lr_target)))
    gan_lr_mae = float(np.mean(np.abs(gan_lr - lr_target)))
    cnn_lr_mse = _mse(cnn_lr, lr_target)
    gan_lr_mse = _mse(gan_lr, lr_target)
    cnn_lr_psnr = _psnr(cnn_lr, lr_target)
    gan_lr_psnr = _psnr(gan_lr, lr_target)

    w_lr_mae = _winner_low(cnn_lr_mae, gan_lr_mae)
    w_lr_mse = _winner_low(cnn_lr_mse, gan_lr_mse)
    w_lr_psnr = _winner_high(cnn_lr_psnr, gan_lr_psnr)
    w_lr_group = _vote_winner([w_lr_mae, w_lr_mse, w_lr_psnr])

    gt_max = float(np.max(gt_speed))
    cnn_max_abs = abs(float(np.max(cnn_speed)) - gt_max)
    gan_max_abs = abs(float(np.max(gan_speed)) - gt_max)
    w_ext_max = _winner_low(cnn_max_abs, gan_max_abs)

    ext_winners = [w_ext_max]
    out: Dict[str, object] = {
        "sample_idx": si,
        "psnr_cnn": cnn_row.psnr,
        "psnr_gan": gan_row.psnr,
        "ssim_cnn": cnn_row.ssim,
        "ssim_gan": gan_row.ssim,
        "abs_delta_psnr": metric_delta_psnr,
        "abs_delta_ssim": metric_delta_ssim,
        "mt_cnn": cnn_row.mt_distance,
        "mt_gan": gan_row.mt_distance,
        "pd_cnn": cnn_row.pd_distance,
        "pd_gan": gan_row.pd_distance,
        "delta_mt_cnn_minus_gan": cnn_row.mt_distance - gan_row.mt_distance,
        "delta_pd_cnn_minus_gan": cnn_row.pd_distance - gan_row.pd_distance,
        "winner_mt": mt_winner,
        "winner_pd": pd_winner,
        "cnn_lr_mae": cnn_lr_mae,
        "gan_lr_mae": gan_lr_mae,
        "cnn_lr_mse": cnn_lr_mse,
        "gan_lr_mse": gan_lr_mse,
        "cnn_lr_psnr": cnn_lr_psnr,
        "gan_lr_psnr": gan_lr_psnr,
        "winner_lr_mae": w_lr_mae,
        "winner_lr_mse": w_lr_mse,
        "winner_lr_psnr": w_lr_psnr,
        "winner_lr_group": w_lr_group,
        "cnn_err_abs_max": cnn_max_abs,
        "gan_err_abs_max": gan_max_abs,
        "winner_extreme_abs_max": w_ext_max,
    }

    for q in exceed_qs:
        thr = float(np.quantile(gt_speed, q))
        cnn_ex = _exceed_frac(cnn_speed, thr)
        gan_ex = _exceed_frac(gan_speed, thr)
        gt_ex = _exceed_frac(gt_speed, thr)
        cnn_ex_err = abs(cnn_ex - gt_ex)
        gan_ex_err = abs(gan_ex - gt_ex)
        w = _winner_low(cnn_ex_err, gan_ex_err)
        tag = f"{int(round(q * 100))}"
        out[f"cnn_exceed_err_q{tag}"] = cnn_ex_err
        out[f"gan_exceed_err_q{tag}"] = gan_ex_err
        out[f"winner_exceed_q{tag}"] = w
        ext_winners.append(w)

    tail_thr = float(np.quantile(gt_speed, tail_q))
    mask = gt_speed >= tail_thr
    if np.any(mask):
        cnn_tail_mae = float(np.mean(np.abs(cnn_speed[mask] - gt_speed[mask])))
        gan_tail_mae = float(np.mean(np.abs(gan_speed[mask] - gt_speed[mask])))
    else:
        cnn_tail_mae = float(np.mean(np.abs(cnn_speed - gt_speed)))
        gan_tail_mae = float(np.mean(np.abs(gan_speed - gt_speed)))
    w_tail = _winner_low(cnn_tail_mae, gan_tail_mae)
    out["cnn_tail_mae"] = cnn_tail_mae
    out["gan_tail_mae"] = gan_tail_mae
    out["winner_tail_mae"] = w_tail
    ext_winners.append(w_tail)
    w_ext_group = _vote_winner(ext_winners)
    out["winner_extreme_group"] = w_ext_group

    phys_winners = []
    for c in physics_cols:
        cv = cnn_row.extras.get(c)
        gv = gan_row.extras.get(c)
        if cv is None or gv is None:
            continue
        w = _winner_low(cv, gv)
        out[f"cnn_{c}"] = cv
        out[f"gan_{c}"] = gv
        out[f"winner_{c}"] = w
        phys_winners.append(w)
    out["winner_physics_group"] = _vote_winner(phys_winners, prefer="na") if phys_winners else "na"

    for wkey in [
        "winner_lr_mae",
        "winner_lr_mse",
        "winner_lr_psnr",
        "winner_lr_group",
        "winner_extreme_abs_max",
        "winner_extreme_group",
        "winner_tail_mae",
        "winner_physics_group",
    ]:
        ww = str(out.get(wkey, "na"))
        out[f"agree_mt_{wkey}"] = int(ww == mt_winner) if ww in {"cnn", "gan"} else "na"

    return out


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", newline="") as f:
            pass
        return
    keys = []
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


def _count(rows: List[Dict[str, object]], key: str, val: str) -> int:
    return sum(1 for r in rows if str(r.get(key, "")) == val)


def _agreement(rows: List[Dict[str, object]], key: str) -> Tuple[int, int, float]:
    vals = [r.get(key) for r in rows]
    valid = [int(v) for v in vals if str(v) in {"0", "1"}]
    if not valid:
        return 0, 0, float("nan")
    k = sum(valid)
    n = len(valid)
    return k, n, k / n


def _summarize(rows: List[Dict[str, object]], metric: str, thresh: float, out_txt: Path, physics_cols: Sequence[str]) -> Dict[str, object]:
    lines: List[str] = []
    n = len(rows)
    lines.append(f"metric={metric} threshold={thresh:.5f}")
    lines.append(f"near_ties={n}")
    lines.append(f"mt_select_cnn={_count(rows, 'winner_mt', 'cnn')}")
    lines.append(f"mt_select_gan={_count(rows, 'winner_mt', 'gan')}")

    agree_keys = [
        "agree_mt_winner_lr_group",
        "agree_mt_winner_extreme_group",
        "agree_mt_winner_physics_group",
        "agree_mt_winner_lr_mae",
        "agree_mt_winner_lr_mse",
        "agree_mt_winner_lr_psnr",
        "agree_mt_winner_extreme_abs_max",
        "agree_mt_winner_tail_mae",
    ]
    summary: Dict[str, object] = {
        "metric": metric,
        "threshold": thresh,
        "near_ties": n,
        "mt_select_cnn": _count(rows, "winner_mt", "cnn"),
        "mt_select_gan": _count(rows, "winner_mt", "gan"),
    }

    lines.append("")
    lines.append("agreement_with_mt:")
    for k in agree_keys:
        a, b, r = _agreement(rows, k)
        lines.append(f"  {k}: {a}/{b} ({r:.4f})" if b > 0 else f"  {k}: n/a")
        summary[k] = r if b > 0 else "na"

    if n == 0:
        lines.append("NOTE: no near-tie samples at this threshold.")
    elif n < 5:
        lines.append("NOTE: very small near-tie sample size; interpret proportions cautiously.")

    if physics_cols:
        lines.append(f"physics_cols_used={','.join(physics_cols)}")
    else:
        lines.append("physics_cols_used=none")

    out_txt.write_text("\n".join(lines) + "\n")
    return summary


def _top_cases(rows: List[Dict[str, object]], metric: str, thresh: float, out_csv: Path, top_n: int = 10) -> None:
    for r in rows:
        r["topology_gap_abs"] = abs(float(r["delta_mt_cnn_minus_gan"]))
        agrees = [r.get("agree_mt_winner_lr_group"), r.get("agree_mt_winner_extreme_group"), r.get("agree_mt_winner_physics_group")]
        vv = [int(v) for v in agrees if str(v) in {"0", "1"}]
        r["agreement_score"] = float(sum(vv) / len(vv)) if vv else float("nan")
        r["metric"] = metric
        r["threshold"] = thresh

    strongest = sorted(rows, key=lambda x: float(x["topology_gap_abs"]), reverse=True)[:top_n]
    agree = [r for r in rows if not math.isnan(float(r.get("agreement_score", float("nan"))))]
    most_agree = sorted(agree, key=lambda x: float(x["agreement_score"]), reverse=True)[:top_n]
    most_disagree = sorted(agree, key=lambda x: float(x["agreement_score"]))[:top_n]

    tagged: List[Dict[str, object]] = []
    for group, items in [
        ("strongest_topology_gap", strongest),
        ("highest_agreement", most_agree),
        ("highest_disagreement", most_disagree),
    ]:
        for r in items:
            rr = dict(r)
            rr["top_case_group"] = group
            tagged.append(rr)
    _write_csv(out_csv, tagged)


def _plot_agreement(summary_rows: List[Dict[str, object]], outdir: Path) -> None:
    if plt is None or not summary_rows:
        return
    check_keys = [
        "agree_mt_winner_lr_group",
        "agree_mt_winner_extreme_group",
        "agree_mt_winner_physics_group",
    ]
    labels = ["LR group", "Extreme group", "Physics group"]

    for metric in sorted({str(r["metric"]) for r in summary_rows}):
        rows = [r for r in summary_rows if str(r["metric"]) == metric]
        if not rows:
            continue
        fig, ax = plt.subplots(figsize=(8.5, 4.5))
        x = np.arange(len(rows))
        width = 0.22
        for i, (k, lab) in enumerate(zip(check_keys, labels)):
            vals = []
            for r in rows:
                v = r.get(k, "na")
                vals.append(float(v) if isinstance(v, float) else np.nan)
            ax.bar(x + (i - 1) * width, vals, width=width, label=lab)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{float(r['threshold']):.3f}" for r in rows])
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel(f"{metric.upper()} near-tie threshold")
        ax.set_ylabel("Agreement rate vs MT winner")
        ax.set_title(f"MT-winner agreement rates ({metric.upper()})")
        ax.legend()
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        fig.tight_layout()
        p = outdir / f"{metric}_agreement_rates.png"
        fig.savefig(p, dpi=200)
        plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(8.5, 4.5))
    metrics = sorted({str(r["metric"]) for r in summary_rows})
    width = 0.35
    max_len = max((len([r for r in summary_rows if str(r["metric"]) == m]) for m in metrics), default=0)
    base = np.arange(max_len)
    for i, m in enumerate(metrics):
        rows = [r for r in summary_rows if str(r["metric"]) == m]
        cnts = [int(r["near_ties"]) for r in rows]
        pos = base[: len(cnts)] + (i - (len(metrics) - 1) / 2) * width
        ax2.bar(pos, cnts, width=width, label=m.upper())
    ax2.set_xlabel("Threshold index (sorted per metric)")
    ax2.set_ylabel("Near-tie count")
    ax2.set_title("Near-tie sample counts by threshold")
    ax2.legend()
    ax2.grid(True, axis="y", linestyle="--", alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(outdir / "near_tie_counts_by_threshold.png", dpi=200)
    plt.close(fig2)


def main() -> int:
    ap = argparse.ArgumentParser(description="Near-tie validation study for PSNR/SSIM topology selection")
    ap.add_argument("--combined-csv", type=Path, default=Path("ttk_runs/combined/combined_pairwise_results.csv"), help="Retained for explicit provenance")
    ap.add_argument("--merged-csv", type=Path, default=Path("ttk_runs/combined/psnr_topology_physics_merged.csv"), help="Merged CSV containing PSNR/SSIM/MT/PD and optional physics metrics")
    ap.add_argument("--cnn-dir", type=Path, default=Path("data_out/wind_mrhr_cnn"))
    ap.add_argument("--gan-dir", type=Path, default=Path("data_out/wind_mrhr_gan"))
    ap.add_argument("--out-root", type=Path, default=Path("ttk_runs/near_tie_study"))
    ap.add_argument("--psnr-thresholds", default="0.05,0.10,0.20")
    ap.add_argument("--ssim-thresholds", default="0.02,0.03,0.05,0.075")
    ap.add_argument("--tail-quantile", type=float, default=0.95)
    ap.add_argument("--exceed-quantiles", default="0.90,0.95")
    ap.add_argument("--include-dual", action="store_true", help="Also analyze intersection: |ΔPSNR|<=dual-psnr and |ΔSSIM|<=dual-ssim")
    ap.add_argument("--dual-psnr", type=float, default=0.10)
    ap.add_argument("--dual-ssim", type=float, default=0.03)
    ap.add_argument("--top-cases", type=int, default=10)
    args = ap.parse_args()

    _ = args.combined_csv  # explicit arg for provenance; merged csv is primary data source.

    psnr_thresholds = _parse_thresholds(args.psnr_thresholds)
    ssim_thresholds = _parse_thresholds(args.ssim_thresholds)
    exceed_qs = _parse_thresholds(args.exceed_quantiles)

    sample_rows = _load_rows(args.merged_csv)
    if not sample_rows:
        raise SystemExit(f"No usable rows found in {args.merged_csv}")

    in_cnn = np.load(args.cnn_dir / "dataIN.npy", mmap_mode="r")
    sr_cnn = np.load(args.cnn_dir / "dataSR.npy", mmap_mode="r")
    gt_cnn = np.load(args.cnn_dir / "dataGT.npy", mmap_mode="r")
    sr_gan = np.load(args.gan_dir / "dataSR.npy", mmap_mode="r")

    physics_cols = _pick_physics_cols(sample_rows)

    per_sample: List[Dict[str, object]] = []
    for si, by_method in sorted(sample_rows.items()):
        cnn = by_method.get("cnn")
        gan = by_method.get("gan")
        if cnn is None or gan is None:
            continue
        per_sample.append(
            _metrics_for_sample(
                si=si,
                cnn_row=cnn,
                gan_row=gan,
                in_cnn=in_cnn,
                sr_cnn=sr_cnn,
                gt_cnn=gt_cnn,
                sr_gan=sr_gan,
                physics_cols=physics_cols,
                tail_q=args.tail_quantile,
                exceed_qs=exceed_qs,
            )
        )

    if not per_sample:
        raise SystemExit("No paired CNN/GAN samples with complete metrics were found")

    summary_rows: List[Dict[str, object]] = []

    def run_metric(metric: str, thresholds: Sequence[float], predicate) -> None:
        metric_dir = args.out_root / metric
        metric_dir.mkdir(parents=True, exist_ok=True)
        for th in thresholds:
            rows = [r for r in per_sample if predicate(r, th)]
            rows = sorted(rows, key=lambda x: abs(float(x[f"delta_mt_cnn_minus_gan"])), reverse=True)
            out_csv = metric_dir / f"near_tie_{metric}_thr_{th:.3f}.csv"
            out_txt = metric_dir / f"near_tie_{metric}_thr_{th:.3f}_summary.txt"
            out_top = metric_dir / f"near_tie_{metric}_thr_{th:.3f}_top_cases.csv"
            _write_csv(out_csv, rows)
            summary_rows.append(_summarize(rows, metric, th, out_txt, physics_cols))
            _top_cases(rows, metric, th, out_top, top_n=args.top_cases)

    run_metric("psnr", psnr_thresholds, lambda r, th: float(r["abs_delta_psnr"]) <= th)
    run_metric("ssim", ssim_thresholds, lambda r, th: float(r["abs_delta_ssim"]) <= th)

    if args.include_dual:
        dual_dir = args.out_root / "dual"
        dual_dir.mkdir(parents=True, exist_ok=True)
        rows = [
            r
            for r in per_sample
            if float(r["abs_delta_psnr"]) <= args.dual_psnr and float(r["abs_delta_ssim"]) <= args.dual_ssim
        ]
        rows = sorted(rows, key=lambda x: abs(float(x["delta_mt_cnn_minus_gan"])), reverse=True)
        out_csv = dual_dir / f"near_tie_dual_psnr_{args.dual_psnr:.3f}_ssim_{args.dual_ssim:.3f}.csv"
        out_txt = dual_dir / f"near_tie_dual_psnr_{args.dual_psnr:.3f}_ssim_{args.dual_ssim:.3f}_summary.txt"
        out_top = dual_dir / f"near_tie_dual_psnr_{args.dual_psnr:.3f}_ssim_{args.dual_ssim:.3f}_top_cases.csv"
        _write_csv(out_csv, rows)
        summary_rows.append(_summarize(rows, "dual", float(args.dual_psnr), out_txt, physics_cols))
        _top_cases(rows, "dual", float(args.dual_psnr), out_top, top_n=args.top_cases)

    summary_rows = sorted(summary_rows, key=lambda x: (str(x["metric"]), float(x["threshold"])))
    _write_csv(args.out_root / "near_tie_summary_all.csv", summary_rows)
    _plot_agreement(summary_rows, args.out_root)

    print(f"Wrote study outputs to: {args.out_root}")
    print("Start here:")
    print(f"  - {args.out_root/'near_tie_summary_all.csv'}")
    print(f"  - {args.out_root/'psnr'}")
    print(f"  - {args.out_root/'ssim'}")
    if args.include_dual:
        print(f"  - {args.out_root/'dual'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
