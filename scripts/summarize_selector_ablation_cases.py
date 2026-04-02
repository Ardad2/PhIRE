#!/usr/bin/env python3
"""Summarize selector ablation subset/grouping for a single threshold CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List


def _read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def _as_int(x: str) -> int:
    return int(float(str(x).strip()))


def _is_consensus(row: Dict[str, str]) -> bool:
    return row.get("consensus_topology_winner", "").strip().lower() in {"cnn", "gan", "tie"}


def summarize(ablation_csv: Path) -> Path:
    rows = _read_rows(ablation_csv)
    ids = sorted(_as_int(r["sample_idx"]) for r in rows)
    consensus_ids = sorted(_as_int(r["sample_idx"]) for r in rows if _is_consensus(r))
    mt_only_ids = sorted(_as_int(r["sample_idx"]) for r in rows if not _is_consensus(r))

    consensus_set = set(consensus_ids)
    checks = {sid: (sid in consensus_set) for sid in (25, 8, 12)}

    out_path = ablation_csv.with_suffix("").with_name(ablation_csv.stem + "_case_summary.txt")
    lines = [
        f"ablation_csv={ablation_csv}",
        f"subset_count={len(ids)}",
        f"subset_sample_ids={','.join(map(str, ids))}",
        f"consensus_count={len(consensus_ids)}",
        f"consensus_sample_ids={','.join(map(str, consensus_ids))}",
        f"mt_only_count={len(mt_only_ids)}",
        f"mt_only_sample_ids={','.join(map(str, mt_only_ids))}",
        f"sample_25_in_consensus={checks[25]}",
        f"sample_8_in_consensus={checks[8]}",
        f"sample_12_in_consensus={checks[12]}",
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
