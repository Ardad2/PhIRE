#!/usr/bin/env python3
"""Post-process repaired merged CSV to analyze topology selector behavior in SSIM near ties."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

from run_near_tie_study import _read_csv, _sample_idx, _to_float, _vote_winner, _winner_low

REQUIRED_COLS = {
    "sample_idx",
    "method",
    "ssim",
    "psnr",
    "pd_distance",
    "mt_distance",
    "wpd_rmse",
    "wpd_mae",
    "psd_log_l2",
    "grad_mae",
}
PHYSICS_COLS = ["wpd_rmse", "wpd_mae", "psd_log_l2", "grad_mae"]


def _threshold_tag(threshold: float) -> str:
    return f"{threshold:.6f}".rstrip("0").rstrip(".").replace("-", "m").replace(".", "p") or "0"


def _validate_columns(rows: List[Dict[str, str]]) -> None:
    if not rows:
        raise SystemExit("Merged CSV has no rows.")
    missing = sorted(c for c in REQUIRED_COLS if c not in rows[0])
    if missing:
        raise SystemExit(f"Merged CSV is missing required columns: {', '.join(missing)}")


def _load_paired_samples(merged_csv: Path) -> Dict[int, Dict[str, Dict[str, float]]]:
    rows = _read_csv(merged_csv)
    _validate_columns(rows)
    paired: Dict[int, Dict[str, Dict[str, float]]] = {}
    for row in rows:
        method = str(row.get("method", "")).strip().lower()
        if method not in {"cnn", "gan"}:
            continue
        sample_idx = _sample_idx(row)
        if sample_idx is None:
            continue
        values: Dict[str, float] = {}
        needed = ["ssim", "psnr", "pd_distance", "mt_distance", *PHYSICS_COLS]
        ok = True
        for key in needed:
            fv = _to_float(row.get(key))
            if fv is None:
                ok = False
                break
            values[key] = fv
        if not ok:
            continue
        paired.setdefault(sample_idx, {})[method] = values
    return {k: v for k, v in paired.items() if {"cnn", "gan"}.issubset(v.keys())}


def _count(rows: List[Dict[str, object]], key: str, value: str) -> int:
    return sum(1 for r in rows if r.get(key) == value)


def _agreement(rows: List[Dict[str, object]], a: str, b: str) -> str:
    valid = [r for r in rows if r.get(a) in {"cnn", "gan"} and r.get(b) in {"cnn", "gan"}]
    if not valid:
        return "na"
    matches = sum(1 for r in valid if r[a] == r[b])
    return f"{matches}/{len(valid)} ({matches / len(valid):.3f})"


def _analyze_threshold(samples: Dict[int, Dict[str, Dict[str, float]]], threshold: float) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for sample_idx in sorted(samples):
        cnn = samples[sample_idx]["cnn"]
        gan = samples[sample_idx]["gan"]
        ssim_gap = abs(cnn["ssim"] - gan["ssim"])
        if ssim_gap > threshold:
            continue

        mt_winner = _winner_low(cnn["mt_distance"], gan["mt_distance"])
        pd_winner = _winner_low(cnn["pd_distance"], gan["pd_distance"])
        physics_votes = [_winner_low(cnn[col], gan[col]) for col in PHYSICS_COLS]
        physics_group_winner = _vote_winner(physics_votes)
        consensus_topology_winner = mt_winner if mt_winner == pd_winner else "abstain"

        out.append(
            {
                "sample_idx": sample_idx,
                "ssim_cnn": cnn["ssim"],
                "ssim_gan": gan["ssim"],
                "ssim_abs_gap": ssim_gap,
                "mt_distance_cnn": cnn["mt_distance"],
                "mt_distance_gan": gan["mt_distance"],
                "pd_distance_cnn": cnn["pd_distance"],
                "pd_distance_gan": gan["pd_distance"],
                "mt_winner": mt_winner,
                "pd_winner": pd_winner,
                "physics_group_winner": physics_group_winner,
                "consensus_topology_winner": consensus_topology_winner,
                "mt_pd_disagree": int(mt_winner != pd_winner),
                "wpd_rmse_winner": physics_votes[0],
                "wpd_mae_winner": physics_votes[1],
                "psd_log_l2_winner": physics_votes[2],
                "grad_mae_winner": physics_votes[3],
            }
        )
    return out


def _write_outputs(rows: List[Dict[str, object]], threshold: float, outdir: Path) -> None:
    tag = _threshold_tag(threshold)
    csv_path = outdir / f"selector_ablation_threshold_{tag}.csv"
    txt_path = outdir / f"selector_ablation_threshold_{tag}_summary.txt"

    fieldnames = [
        "sample_idx",
        "ssim_cnn",
        "ssim_gan",
        "ssim_abs_gap",
        "mt_distance_cnn",
        "mt_distance_gan",
        "pd_distance_cnn",
        "pd_distance_gan",
        "mt_winner",
        "pd_winner",
        "physics_group_winner",
        "consensus_topology_winner",
        "mt_pd_disagree",
        "wpd_rmse_winner",
        "wpd_mae_winner",
        "psd_log_l2_winner",
        "grad_mae_winner",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    disagree_ids = [str(r["sample_idx"]) for r in rows if r["mt_pd_disagree"] == 1]
    lines = [
        f"threshold={threshold}",
        f"near_tie_count={len(rows)}",
        f"mt_winner_cnn={_count(rows, 'mt_winner', 'cnn')}",
        f"mt_winner_gan={_count(rows, 'mt_winner', 'gan')}",
        f"mt_winner_tie={_count(rows, 'mt_winner', 'tie')}",
        f"pd_winner_cnn={_count(rows, 'pd_winner', 'cnn')}",
        f"pd_winner_gan={_count(rows, 'pd_winner', 'gan')}",
        f"pd_winner_tie={_count(rows, 'pd_winner', 'tie')}",
        f"agree_mt_vs_physics_group={_agreement(rows, 'mt_winner', 'physics_group_winner')}",
        f"agree_pd_vs_physics_group={_agreement(rows, 'pd_winner', 'physics_group_winner')}",
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
        help="One or more SSIM near-tie thresholds, e.g. --thresholds 0.0005 0.001 0.002",
    )
    parser.add_argument("--outdir", type=Path, required=True, help="Directory for per-threshold CSV and summary outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    samples = _load_paired_samples(args.merged_csv)
    if not samples:
        raise SystemExit("No paired cnn/gan samples with required metrics found in merged CSV.")

    for threshold in args.thresholds:
        if threshold < 0:
            raise SystemExit(f"Threshold must be non-negative: {threshold}")
        rows = _analyze_threshold(samples, threshold)
        _write_outputs(rows, threshold, args.outdir)


if __name__ == "__main__":
    main()
