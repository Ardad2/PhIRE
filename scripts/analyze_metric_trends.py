#!/usr/bin/env python3
"""Analyze per-sample trend alignment between SSIM, MT, and PD deltas."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from run_near_tie_study import _read_csv, _sample_idx, _to_float

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def _winner_from_delta(delta: float, eps: float = 1e-12) -> str:
    if abs(delta) <= eps:
        return "tie"
    return "cnn" if delta > 0 else "gan"


def _rankdata(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a)
    ranks = np.empty(len(a), dtype=float)
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and a[order[j + 1]] == a[order[i]]:
            j += 1
        rank = (i + j) / 2.0 + 1.0
        ranks[order[i : j + 1]] = rank
        i = j + 1
    return ranks


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    return _pearson(_rankdata(x), _rankdata(y))


def _sign_agreement(a: np.ndarray, b: np.ndarray) -> Tuple[int, int]:
    sa = np.sign(a)
    sb = np.sign(b)
    valid = (sa != 0) & (sb != 0)
    if int(np.sum(valid)) == 0:
        return 0, 0
    agree = int(np.sum(sa[valid] == sb[valid]))
    return agree, int(np.sum(valid))


def _load_pairs(merged_csv: Path) -> List[Dict[str, object]]:
    rows = _read_csv(merged_csv)
    by_sample: Dict[int, Dict[str, Dict[str, str]]] = {}
    for row in rows:
        method = str(row.get("method", "")).strip().lower()
        if method not in {"cnn", "gan"}:
            continue
        si = _sample_idx(row)
        if si is None:
            continue
        by_sample.setdefault(si, {})[method] = row

    out: List[Dict[str, object]] = []
    for si in sorted(by_sample):
        pair = by_sample[si]
        if "cnn" not in pair or "gan" not in pair:
            continue
        cnn, gan = pair["cnn"], pair["gan"]
        ssim_cnn = _to_float(cnn.get("ssim"))
        ssim_gan = _to_float(gan.get("ssim"))
        mt_cnn = _to_float(cnn.get("mt_distance"))
        mt_gan = _to_float(gan.get("mt_distance"))
        pd_cnn = _to_float(cnn.get("pd_distance"))
        pd_gan = _to_float(gan.get("pd_distance"))
        if None in (ssim_cnn, ssim_gan, mt_cnn, mt_gan, pd_cnn, pd_gan):
            continue

        delta_ssim = float(ssim_cnn - ssim_gan)
        delta_mt = float(mt_gan - mt_cnn)  # positive => MT favors CNN
        delta_pd = float(pd_gan - pd_cnn)  # positive => PD favors CNN

        out.append(
            {
                "sample_idx": si,
                "delta_ssim": delta_ssim,
                "delta_mt": delta_mt,
                "delta_pd": delta_pd,
                "winner_ssim": _winner_from_delta(delta_ssim),
                "winner_mt": _winner_from_delta(delta_mt),
                "winner_pd": _winner_from_delta(delta_pd),
            }
        )
    return out


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    with path.open("w", newline="") as f:
        if not rows:
            f.write("\n")
            return
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _plot_scatter(x: np.ndarray, y: np.ndarray, out_png: Path, ylabel: str) -> None:
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(x, y, s=14, alpha=0.8)
    ax.axhline(0.0, color="gray", linewidth=1)
    ax.axvline(0.0, color="gray", linewidth=1)
    ax.set_xlabel("delta_ssim (cnn - gan)")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--merged-csv", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    rows = _load_pairs(args.merged_csv)
    if not rows:
        raise SystemExit("No paired CNN/GAN rows with required columns found.")

    per_sample_csv = args.outdir / "metric_trends_per_sample.csv"
    summary_txt = args.outdir / "metric_trends_summary.txt"

    _write_csv(per_sample_csv, rows)

    delta_ssim = np.asarray([float(r["delta_ssim"]) for r in rows], dtype=float)
    delta_mt = np.asarray([float(r["delta_mt"]) for r in rows], dtype=float)
    delta_pd = np.asarray([float(r["delta_pd"]) for r in rows], dtype=float)

    pearson_ssim_mt = _pearson(delta_ssim, delta_mt)
    pearson_ssim_pd = _pearson(delta_ssim, delta_pd)
    spearman_ssim_mt = _spearman(delta_ssim, delta_mt)
    spearman_ssim_pd = _spearman(delta_ssim, delta_pd)

    agree_ssim_mt, n_ssim_mt = _sign_agreement(delta_ssim, delta_mt)
    agree_ssim_pd, n_ssim_pd = _sign_agreement(delta_ssim, delta_pd)

    opposite_ssim_mt = [r for r in rows if float(r["delta_ssim"]) * float(r["delta_mt"]) < 0]
    opposite_ssim_mt = sorted(opposite_ssim_mt, key=lambda r: abs(float(r["delta_ssim"]) * float(r["delta_mt"])), reverse=True)

    lines = [
        f"merged_csv={args.merged_csv}",
        f"num_samples={len(rows)}",
        f"pearson_delta_ssim_vs_delta_mt={pearson_ssim_mt:.6f}",
        f"pearson_delta_ssim_vs_delta_pd={pearson_ssim_pd:.6f}",
        f"spearman_delta_ssim_vs_delta_mt={spearman_ssim_mt:.6f}",
        f"spearman_delta_ssim_vs_delta_pd={spearman_ssim_pd:.6f}",
        f"sign_agreement_delta_ssim_vs_delta_mt={agree_ssim_mt}/{n_ssim_mt}",
        f"sign_agreement_delta_ssim_vs_delta_pd={agree_ssim_pd}/{n_ssim_pd}",
        "opposite_sign_outliers_ssim_vs_mt(sample|delta_ssim|delta_mt)="
        + ",".join(f"{r['sample_idx']}|{float(r['delta_ssim']):.6f}|{float(r['delta_mt']):.6f}" for r in opposite_ssim_mt),
    ]

    by_id = {int(r["sample_idx"]): r for r in rows}
    for sid in (8, 12, 25):
        r = by_id.get(sid)
        if r is None:
            lines.append(f"flag_sample_{sid}=missing")
            continue
        is_opposite = float(r["delta_ssim"]) * float(r["delta_mt"]) < 0
        lines.append(
            f"flag_sample_{sid}=delta_ssim:{float(r['delta_ssim']):.6f},delta_mt:{float(r['delta_mt']):.6f},delta_pd:{float(r['delta_pd']):.6f},"
            f"winner_ssim:{r['winner_ssim']},winner_mt:{r['winner_mt']},winner_pd:{r['winner_pd']},opposite_ssim_mt:{is_opposite}"
        )

    summary_txt.write_text("\n".join(lines) + "\n")

    _plot_scatter(delta_ssim, delta_mt, args.outdir / "scatter_delta_ssim_vs_delta_mt.png", "delta_mt (gan - cnn distance)")
    _plot_scatter(delta_ssim, delta_pd, args.outdir / "scatter_delta_ssim_vs_delta_pd.png", "delta_pd (gan - cnn distance)")

    print(f"Wrote: {per_sample_csv}")
    print(f"Wrote: {summary_txt}")
    if plt is not None:
        print(f"Wrote: {args.outdir / 'scatter_delta_ssim_vs_delta_mt.png'}")
        print(f"Wrote: {args.outdir / 'scatter_delta_ssim_vs_delta_pd.png'}")
    else:
        print("matplotlib not available; scatter plots skipped.")


if __name__ == "__main__":
    main()
