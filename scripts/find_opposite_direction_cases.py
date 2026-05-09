#!/usr/bin/env python3
"""Find and summarize opposite-direction SSIM-vs-MT cases from selector ablation outputs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional

from run_near_tie_study import _winner_high

VALIDATOR_WINNERS = ["winner_lr_group", "winner_extreme_group", "winner_physics_group"]


def _to_float(x: object) -> Optional[float]:
    try:
        return float(str(x).strip())
    except Exception:
        return None


def _winner_from_pair(row: Dict[str, str], prefix: str) -> str:
    a = _to_float(row.get(f"{prefix}_cnn"))
    b = _to_float(row.get(f"{prefix}_gan"))
    if a is None or b is None:
        return "na"
    return _winner_high(a, b)


def _resolve_winner(row: Dict[str, str], key: str, fallback_prefix: Optional[str] = None) -> str:
    v = str(row.get(key, "")).strip().lower()
    if v in {"cnn", "gan", "tie", "na"}:
        return v
    if fallback_prefix is not None:
        return _winner_from_pair(row, fallback_prefix)
    return "na"


def _support_count(row: Dict[str, str], mt_winner: str) -> int:
    if mt_winner not in {"cnn", "gan"}:
        return 0
    return sum(1 for k in VALIDATOR_WINNERS if str(row.get(k, "na")).strip().lower() == mt_winner)


def summarize(ablation_csv: Path) -> Path:
    with ablation_csv.open("r", newline="") as f:
        rows = list(csv.DictReader(f))

    enriched: List[Dict[str, object]] = []
    for row in rows:
        sample_idx = int(float(row["sample_idx"]))
        winner_ssim = _resolve_winner(row, "winner_ssim", fallback_prefix="ssim")
        winner_psnr = _resolve_winner(row, "winner_psnr", fallback_prefix="psnr")
        winner_mt = _resolve_winner(row, "winner_mt")
        winner_pd = _resolve_winner(row, "winner_pd")
        winner_lr_group = _resolve_winner(row, "winner_lr_group")
        winner_extreme_group = _resolve_winner(row, "winner_extreme_group")
        winner_physics_group = _resolve_winner(row, "winner_physics_group")

        abs_delta_ssim = _to_float(row.get("abs_delta_ssim"))
        if abs_delta_ssim is None:
            ssim_cnn = _to_float(row.get("ssim_cnn"))
            ssim_gan = _to_float(row.get("ssim_gan"))
            abs_delta_ssim = abs(ssim_cnn - ssim_gan) if ssim_cnn is not None and ssim_gan is not None else float("inf")

        enriched.append(
            {
                "sample_idx": sample_idx,
                "winner_ssim": winner_ssim,
                "winner_psnr": winner_psnr,
                "winner_mt": winner_mt,
                "winner_pd": winner_pd,
                "winner_lr_group": winner_lr_group,
                "winner_extreme_group": winner_extreme_group,
                "winner_physics_group": winner_physics_group,
                "mt_support_count": _support_count(
                    {
                        "winner_lr_group": winner_lr_group,
                        "winner_extreme_group": winner_extreme_group,
                        "winner_physics_group": winner_physics_group,
                    },
                    winner_mt,
                ),
                "abs_delta_ssim": abs_delta_ssim,
            }
        )

    gan_ssim_cnn_mt = [r for r in enriched if r["winner_ssim"] == "gan" and r["winner_mt"] == "cnn"]
    cnn_ssim_gan_mt = [r for r in enriched if r["winner_ssim"] == "cnn" and r["winner_mt"] == "gan"]

    def _rank(rows_in: List[Dict[str, object]]) -> List[Dict[str, object]]:
        return sorted(rows_in, key=lambda r: (-int(r["mt_support_count"]), float(r["abs_delta_ssim"]), int(r["sample_idx"])))

    ranked_gan_ssim_cnn_mt = _rank(gan_ssim_cnn_mt)
    ranked_cnn_ssim_gan_mt = _rank(cnn_ssim_gan_mt)

    out_path = ablation_csv.with_name(ablation_csv.stem + "_opposite_direction_summary.txt")
    lines = [
        f"ablation_csv={ablation_csv}",
        f"num_rows={len(enriched)}",
        "",
        "bucket_1: ssim_favors_gan_but_mt_favors_cnn",
        f"count={len(gan_ssim_cnn_mt)}",
        f"sample_ids={','.join(str(r['sample_idx']) for r in gan_ssim_cnn_mt)}",
        "ranked_top_candidates(sample_id|mt_support_count|abs_delta_ssim)="
        + ",".join(f"{r['sample_idx']}|{r['mt_support_count']}|{float(r['abs_delta_ssim']):.6f}" for r in ranked_gan_ssim_cnn_mt),
        "",
        "bucket_2: ssim_favors_cnn_but_mt_favors_gan",
        f"count={len(cnn_ssim_gan_mt)}",
        f"sample_ids={','.join(str(r['sample_idx']) for r in cnn_ssim_gan_mt)}",
        "ranked_top_candidates(sample_id|mt_support_count|abs_delta_ssim)="
        + ",".join(f"{r['sample_idx']}|{r['mt_support_count']}|{float(r['abs_delta_ssim']):.6f}" for r in ranked_cnn_ssim_gan_mt),
    ]
    out_path.write_text("\n".join(lines) + "\n")
    return out_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ablation-csv", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.ablation_csv.exists():
        raise SystemExit(f"Missing ablation CSV: {args.ablation_csv}")
    out_path = summarize(args.ablation_csv)
    print(out_path)


if __name__ == "__main__":
    main()
