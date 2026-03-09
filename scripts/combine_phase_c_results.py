#!/usr/bin/env python3
"""Combine per-method Phase C outputs into one report table.

Expected inputs:
  - One or more directories, each containing phase_c_results.csv

Outputs (in --outdir):
  - combined_pairwise_results.csv
  - combined_method_summary.csv
  - combined_summary.txt
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Optional


def _f(x: object) -> Optional[float]:
    try:
        if x is None:
            return None
        s = str(x).strip()
        if not s:
            return None
        v = float(s)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


def _stats(vals: List[float]) -> Dict[str, float]:
    if not vals:
        return {"n": 0}
    s = sorted(vals)
    n = len(s)
    mean = sum(s) / n
    med = s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])
    var = sum((v - mean) ** 2 for v in s) / n
    return {
        "n": n,
        "mean": mean,
        "median": med,
        "stdev": math.sqrt(var),
        "min": s[0],
        "max": s[-1],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", type=Path, required=True, help="Per-method phase_c_final directories")
    ap.add_argument("--outdir", type=Path, required=True)
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict[str, object]] = []
    pd_by_m: Dict[str, List[float]] = {}
    mt_by_m: Dict[str, List[float]] = {}

    for d in args.inputs:
        f = d / "phase_c_results.csv"
        if not f.exists():
            print(f"[warn] missing {f}")
            continue
        with f.open("r", newline="") as fh:
            r = csv.DictReader(fh)
            for row in r:
                method = str(row.get("method", "")).strip() or d.name
                key = str(row.get("key", "")).strip()
                pdv = _f(row.get("pd_distance", ""))
                mtv = _f(row.get("mt_distance", ""))
                err = str(row.get("error", "")).strip()
                all_rows.append(
                    {
                        "method": method,
                        "key": key,
                        "pd_distance": "" if pdv is None else pdv,
                        "mt_distance": "" if mtv is None else mtv,
                        "error": err,
                        "source_dir": str(d),
                    }
                )
                if pdv is not None:
                    pd_by_m.setdefault(method, []).append(pdv)
                if mtv is not None:
                    mt_by_m.setdefault(method, []).append(mtv)

    all_rows = sorted(all_rows, key=lambda x: (str(x["method"]), str(x["key"])))

    pair_path = args.outdir / "combined_pairwise_results.csv"
    with pair_path.open("w", newline="") as fh:
        cols = ["method", "key", "pd_distance", "mt_distance", "error", "source_dir"]
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(all_rows)

    methods = sorted(set(pd_by_m) | set(mt_by_m))
    sum_rows: List[Dict[str, object]] = []
    for m in methods:
        spd = _stats(pd_by_m.get(m, []))
        smt = _stats(mt_by_m.get(m, []))
        sum_rows.append(
            {
                "method": m,
                "pd_n": spd.get("n", 0),
                "pd_mean": spd.get("mean", ""),
                "pd_median": spd.get("median", ""),
                "pd_stdev": spd.get("stdev", ""),
                "pd_min": spd.get("min", ""),
                "pd_max": spd.get("max", ""),
                "mt_n": smt.get("n", 0),
                "mt_mean": smt.get("mean", ""),
                "mt_median": smt.get("median", ""),
                "mt_stdev": smt.get("stdev", ""),
                "mt_min": smt.get("min", ""),
                "mt_max": smt.get("max", ""),
            }
        )

    sum_path = args.outdir / "combined_method_summary.csv"
    with sum_path.open("w", newline="") as fh:
        cols = [
            "method",
            "pd_n",
            "pd_mean",
            "pd_median",
            "pd_stdev",
            "pd_min",
            "pd_max",
            "mt_n",
            "mt_mean",
            "mt_median",
            "mt_stdev",
            "mt_min",
            "mt_max",
        ]
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(sum_rows)

    txt = args.outdir / "combined_summary.txt"
    lines = ["Phase C combined summary", ""]
    for r in sum_rows:
        lines.append(f"[{r['method']}]")
        lines.append(
            f"  PD n={r['pd_n']} mean={r['pd_mean']} median={r['pd_median']} stdev={r['pd_stdev']}"
        )
        lines.append(
            f"  MT n={r['mt_n']} mean={r['mt_mean']} median={r['mt_median']} stdev={r['mt_stdev']}"
        )
        lines.append("")
    txt.write_text("\n".join(lines))

    print(f"Wrote: {pair_path}")
    print(f"Wrote: {sum_path}")
    print(f"Wrote: {txt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

