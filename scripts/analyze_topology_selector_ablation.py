#!/usr/bin/env python3
"""Post-process repaired merged CSV to analyze topology selector behavior in SSIM near ties."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from run_near_tie_study import (
    _agreement,
    _count,
    _load_rows,
    _metrics_for_sample,
    _parse_thresholds,
    _pick_physics_cols,
    _validate_alignment,
)

AGREE_WINNER_KEYS = [
    "winner_lr_group",
    "winner_extreme_group",
    "winner_physics_group",
    "winner_lr_mae",
    "winner_lr_mse",
    "winner_lr_psnr",
    "winner_extreme_abs_max",
    "winner_tail_mae",
]


def _threshold_tag(threshold: float) -> str:
    return f"{threshold:.6f}".rstrip("0").rstrip(".").replace("-", "m").replace(".", "p") or "0"


def _with_pd_agreements(row: Dict[str, object]) -> Dict[str, object]:
    out = dict(row)
    pd_winner = str(out.get("winner_pd", "na"))
    for key in AGREE_WINNER_KEYS:
        winner = str(out.get(key, "na"))
        out[f"agree_pd_{key}"] = int(winner == pd_winner) if winner in {"cnn", "gan"} else "na"

    # Backward-compatible aliases from initial ablation output.
    out["mt_winner"] = out.get("winner_mt", "na")
    out["pd_winner"] = out.get("winner_pd", "na")
    out["physics_group_winner"] = out.get("winner_physics_group", "na")
    mt_w = str(out.get("winner_mt", "na"))
    pd_w = str(out.get("winner_pd", "na"))
    out["consensus_topology_winner"] = mt_w if mt_w == pd_w else "abstain"
    out["mt_pd_disagree"] = int(mt_w != pd_w)
    return out


def _compute_per_sample_rows(
    merged_csv: Path,
    cnn_dir: Path,
    gan_dir: Path,
    tail_q: float,
    exceed_qs: Sequence[float],
) -> List[Dict[str, object]]:
    sample_rows = _load_rows(merged_csv)
    if not sample_rows:
        raise SystemExit(f"No usable rows found in {merged_csv}")

    in_cnn = np.load(cnn_dir / "dataIN.npy", mmap_mode="r")
    sr_cnn = np.load(cnn_dir / "dataSR.npy", mmap_mode="r")
    gt_cnn = np.load(cnn_dir / "dataGT.npy", mmap_mode="r")
    sr_gan = np.load(gan_dir / "dataSR.npy", mmap_mode="r")

    _validate_alignment(
        sample_rows=sample_rows,
        cnn_dir=cnn_dir,
        gan_dir=gan_dir,
        in_cnn=in_cnn,
        sr_cnn=sr_cnn,
        gt_cnn=gt_cnn,
        sr_gan=sr_gan,
    )

    physics_cols = _pick_physics_cols(sample_rows)

    rows: List[Dict[str, object]] = []
    for si, by_method in sorted(sample_rows.items()):
        cnn = by_method.get("cnn")
        gan = by_method.get("gan")
        if cnn is None or gan is None:
            continue
        base = _metrics_for_sample(
            si=si,
            cnn_row=cnn,
            gan_row=gan,
            in_cnn=in_cnn,
            sr_cnn=sr_cnn,
            gt_cnn=gt_cnn,
            sr_gan=sr_gan,
            physics_cols=physics_cols,
            tail_q=tail_q,
            exceed_qs=exceed_qs,
        )
        rows.append(_with_pd_agreements(base))
    return rows


def _analyze_threshold(per_sample_rows: List[Dict[str, object]], threshold: float) -> List[Dict[str, object]]:
    return [r for r in per_sample_rows if float(r["abs_delta_ssim"]) <= threshold]


def _write_outputs(rows: List[Dict[str, object]], threshold: float, outdir: Path) -> None:
    tag = _threshold_tag(threshold)
    csv_path = outdir / f"selector_ablation_threshold_{tag}.csv"
    txt_path = outdir / f"selector_ablation_threshold_{tag}_summary.txt"

    if rows:
        fieldnames = []
        seen = set()
        for r in rows:
            for k in r.keys():
                if k not in seen:
                    seen.add(k)
                    fieldnames.append(k)
    else:
        fieldnames = ["sample_idx"]

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    disagree_ids = [str(r["sample_idx"]) for r in rows if int(r.get("mt_pd_disagree", 0)) == 1]

    def _agree_line(key: str) -> str:
        a, b, ratio = _agreement(rows, key)
        return f"{a}/{b} ({ratio:.4f})" if b > 0 else "n/a"

    lines = [
        f"threshold={threshold}",
        f"near_tie_count={len(rows)}",
        f"mt_winner_cnn={_count(rows, 'winner_mt', 'cnn')}",
        f"mt_winner_gan={_count(rows, 'winner_mt', 'gan')}",
        f"mt_winner_tie={_count(rows, 'winner_mt', 'tie')}",
        f"pd_winner_cnn={_count(rows, 'winner_pd', 'cnn')}",
        f"pd_winner_gan={_count(rows, 'winner_pd', 'gan')}",
        f"pd_winner_tie={_count(rows, 'winner_pd', 'tie')}",
        f"agree_mt_vs_lr_group={_agree_line('agree_mt_winner_lr_group')}",
        f"agree_mt_vs_extreme_group={_agree_line('agree_mt_winner_extreme_group')}",
        f"agree_mt_vs_physics_group={_agree_line('agree_mt_winner_physics_group')}",
        f"agree_pd_vs_lr_group={_agree_line('agree_pd_winner_lr_group')}",
        f"agree_pd_vs_extreme_group={_agree_line('agree_pd_winner_extreme_group')}",
        f"agree_pd_vs_physics_group={_agree_line('agree_pd_winner_physics_group')}",
        f"agree_mt_vs_lr_mae={_agree_line('agree_mt_winner_lr_mae')}",
        f"agree_mt_vs_lr_mse={_agree_line('agree_mt_winner_lr_mse')}",
        f"agree_mt_vs_lr_psnr={_agree_line('agree_mt_winner_lr_psnr')}",
        f"agree_mt_vs_extreme_abs_max={_agree_line('agree_mt_winner_extreme_abs_max')}",
        f"agree_mt_vs_tail_mae={_agree_line('agree_mt_winner_tail_mae')}",
        f"agree_pd_vs_lr_mae={_agree_line('agree_pd_winner_lr_mae')}",
        f"agree_pd_vs_lr_mse={_agree_line('agree_pd_winner_lr_mse')}",
        f"agree_pd_vs_lr_psnr={_agree_line('agree_pd_winner_lr_psnr')}",
        f"agree_pd_vs_extreme_abs_max={_agree_line('agree_pd_winner_extreme_abs_max')}",
        f"agree_pd_vs_tail_mae={_agree_line('agree_pd_winner_tail_mae')}",
        f"consensus_count={_count(rows, 'consensus_topology_winner', 'cnn') + _count(rows, 'consensus_topology_winner', 'gan') + _count(rows, 'consensus_topology_winner', 'tie')}",
        f"mt_pd_disagree_sample_ids={','.join(disagree_ids)}",
        f"per_sample_csv={csv_path.name}",
    ]
    txt_path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merged-csv", type=Path, required=True, help="Repaired merged CSV (one row per sample/method).")
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        required=True,
        help="One or more SSIM near-tie thresholds, e.g. --thresholds 0.02 0.05 0.075",
    )
    parser.add_argument("--outdir", type=Path, required=True, help="Directory for per-threshold CSV and summary outputs.")
    parser.add_argument("--combined-csv", type=Path, default=Path("ttk_runs/combined/combined_pairwise_results.csv"), help="Optional provenance argument (unused, aligned with run_near_tie_study).")
    parser.add_argument("--cnn-dir", type=Path, default=Path("data_out/wind_mrhr_cnn"), help="CNN artifact directory containing dataIN.npy/dataSR.npy/dataGT.npy.")
    parser.add_argument("--gan-dir", type=Path, default=Path("data_out/wind_mrhr_gan"), help="GAN artifact directory containing dataSR.npy (and optionally idx.npy/dataIN.npy).")
    parser.add_argument("--tail-quantile", type=float, default=0.95)
    parser.add_argument("--exceed-quantiles", default="0.90,0.95")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _ = args.combined_csv

    args.outdir.mkdir(parents=True, exist_ok=True)
    exceed_qs = _parse_thresholds(args.exceed_quantiles)
    per_sample_rows = _compute_per_sample_rows(
        merged_csv=args.merged_csv,
        cnn_dir=args.cnn_dir,
        gan_dir=args.gan_dir,
        tail_q=args.tail_quantile,
        exceed_qs=exceed_qs,
    )
    if not per_sample_rows:
        raise SystemExit("No paired cnn/gan samples with complete metrics found.")

    for threshold in args.thresholds:
        if threshold < 0:
            raise SystemExit(f"Threshold must be non-negative: {threshold}")
        rows = _analyze_threshold(per_sample_rows, threshold)
        _write_outputs(rows, threshold, args.outdir)


if __name__ == "__main__":
    main()
