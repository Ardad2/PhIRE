#!/usr/bin/env python3
"""Find validator-supported MT-primary cases from selector ablation outputs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional

VALIDATOR_KEYS = ["winner_lr_group", "winner_extreme_group", "winner_physics_group"]


def _to_float(x: object) -> Optional[float]:
    try:
        return float(str(x).strip())
    except Exception:
        return None


def _norm_winner(v: object) -> str:
    s = str(v or "").strip().lower()
    return s if s in {"cnn", "gan", "tie", "na"} else "na"


def _support_count(row: Dict[str, str]) -> int:
    mt = _norm_winner(row.get("winner_mt"))
    if mt not in {"cnn", "gan"}:
        return 0
    return sum(1 for k in VALIDATOR_KEYS if _norm_winner(row.get(k)) == mt)


def summarize(ablation_csv: Path) -> Path:
    with ablation_csv.open("r", newline="") as f:
        rows = list(csv.DictReader(f))

    enriched: List[Dict[str, object]] = []
    for row in rows:
        sample_idx = int(float(row["sample_idx"]))
        abs_delta_ssim = _to_float(row.get("abs_delta_ssim"))
        if abs_delta_ssim is None:
            ssim_cnn = _to_float(row.get("ssim_cnn"))
            ssim_gan = _to_float(row.get("ssim_gan"))
            abs_delta_ssim = abs(ssim_cnn - ssim_gan) if ssim_cnn is not None and ssim_gan is not None else float("inf")
        enriched.append(
            {
                "sample_idx": sample_idx,
                "mt_support_count": _support_count(row),
                "abs_delta_ssim": abs_delta_ssim,
            }
        )

    groups = {
        3: sorted([int(r["sample_idx"]) for r in enriched if int(r["mt_support_count"]) == 3]),
        2: sorted([int(r["sample_idx"]) for r in enriched if int(r["mt_support_count"]) == 2]),
        1: sorted([int(r["sample_idx"]) for r in enriched if int(r["mt_support_count"]) == 1]),
        0: sorted([int(r["sample_idx"]) for r in enriched if int(r["mt_support_count"]) == 0]),
    }

    ranked = sorted(
        enriched,
        key=lambda r: (-int(r["mt_support_count"]), float(r["abs_delta_ssim"]), int(r["sample_idx"])),
    )

    out_path = ablation_csv.with_name(ablation_csv.stem + "_mt_supported_summary.txt")
    lines = [
        f"ablation_csv={ablation_csv}",
        f"num_rows={len(enriched)}",
        f"support_count_3_sample_ids={','.join(map(str, groups[3]))}",
        f"support_count_2_sample_ids={','.join(map(str, groups[2]))}",
        f"support_count_1_sample_ids={','.join(map(str, groups[1]))}",
        f"support_count_0_sample_ids={','.join(map(str, groups[0]))}",
        "ranked_top_candidates(sample_id|mt_support_count|abs_delta_ssim)="
        + ",".join(f"{r['sample_idx']}|{r['mt_support_count']}|{float(r['abs_delta_ssim']):.6f}" for r in ranked),
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
    out = summarize(args.ablation_csv)
    print(out)


if __name__ == "__main__":
    main()
