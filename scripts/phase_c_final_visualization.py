#!/usr/bin/env python3
"""phase_c_final_visualization_real.py

Visualization + sanity checks for REAL TTK distances produced by
`compute_composite_tree_distance.py`.

Input directory (--indir) must contain:
  - phase_c_results.csv
  - (optional) pd_pairwise_distances.csv
  - (optional) mt_pairwise_distances.csv

Outputs written to --indir:
  - phase_c_all_distances.png
  - phase_c_summary.png
  - phase_c_sanity_report.txt
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception as e:
    raise SystemExit(f"ERROR: matplotlib required: {e}\nInstall: sudo apt-get install -y python3-matplotlib")


def _parse_float(x: object) -> Optional[float]:
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


def _load_phase_c_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def _extract_values(rows: List[Dict[str, str]]) -> Tuple[List[float], List[float], List[Tuple[float, float]], List[Dict[str, str]]]:
    pd_vals: List[float] = []
    mt_vals: List[float] = []
    pairs: List[Tuple[float, float]] = []

    failed_rows: List[Dict[str, str]] = []
    for r in rows:
        pd_v = _parse_float(r.get("pd_distance", ""))
        mt_v = _parse_float(r.get("mt_distance", ""))
        err = (r.get("error", "") or "").strip()

        if pd_v is not None:
            pd_vals.append(pd_v)
        if mt_v is not None:
            mt_vals.append(mt_v)
        if pd_v is not None and mt_v is not None:
            pairs.append((pd_v, mt_v))

        if err or mt_v is None:
            failed_rows.append(r)

    return pd_vals, mt_vals, pairs, failed_rows


def _mean(vals: List[float]) -> Optional[float]:
    return (sum(vals) / len(vals)) if vals else None


def _median(vals: List[float]) -> Optional[float]:
    if not vals:
        return None
    s = sorted(vals)
    n = len(s)
    if n % 2:
        return s[n // 2]
    return 0.5 * (s[n // 2 - 1] + s[n // 2])


def _stdev(vals: List[float]) -> Optional[float]:
    if len(vals) < 2:
        return None
    m = _mean(vals)
    assert m is not None
    var = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
    return math.sqrt(var)


def _pearson(xs: List[float], ys: List[float]) -> Optional[float]:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mx = _mean(xs)
    my = _mean(ys)
    if mx is None or my is None:
        return None
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - my) ** 2 for y in ys))
    if den_x == 0.0 or den_y == 0.0:
        return None
    return num / (den_x * den_y)


def _group_by_method(rows: List[Dict[str, str]]) -> Dict[str, Dict[str, List[float]]]:
    grouped: Dict[str, Dict[str, List[float]]] = {}
    for r in rows:
        method = (r.get("method", "") or "").strip() or "unknown"
        bucket = grouped.setdefault(method, {"pd": [], "mt": [], "pair_pd": [], "pair_mt": []})
        pd_v = _parse_float(r.get("pd_distance", ""))
        mt_v = _parse_float(r.get("mt_distance", ""))
        if pd_v is not None:
            bucket["pd"].append(pd_v)
        if mt_v is not None:
            bucket["mt"].append(mt_v)
        if pd_v is not None and mt_v is not None:
            bucket["pair_pd"].append(pd_v)
            bucket["pair_mt"].append(mt_v)
    return grouped


def write_method_summary(indir: Path, grouped: Dict[str, Dict[str, List[float]]]) -> Optional[Path]:
    if len(grouped) < 2:
        return None
    out = indir / "phase_c_method_comparison.csv"
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "pd_n", "pd_mean", "mt_n", "mt_mean", "paired_n", "pd_mt_pearson"])
        for method in sorted(grouped.keys()):
            g = grouped[method]
            pd_vals = g["pd"]
            mt_vals = g["mt"]
            pair_pd = g["pair_pd"]
            pair_mt = g["pair_mt"]
            w.writerow(
                [
                    method,
                    len(pd_vals),
                    "" if _mean(pd_vals) is None else f"{_mean(pd_vals):.12g}",
                    len(mt_vals),
                    "" if _mean(mt_vals) is None else f"{_mean(mt_vals):.12g}",
                    len(pair_pd),
                    "" if _pearson(pair_pd, pair_mt) is None else f"{_pearson(pair_pd, pair_mt):.12g}",
                ]
            )
    return out


def write_sanity_report(indir: Path, rows: List[Dict[str, str]], pd_vals: List[float], mt_vals: List[float], pairs: List[Tuple[float, float]], failed_rows: List[Dict[str, str]]) -> Path:
    out = indir / "phase_c_sanity_report.txt"

    mt_success = sum(1 for r in rows if _parse_float(r.get("mt_distance", "")) is not None)
    pd_success = sum(1 for r in rows if _parse_float(r.get("pd_distance", "")) is not None)

    lines: List[str] = []
    lines.append("Phase C sanity report")
    lines.append("")
    lines.append(f"rows_total={len(rows)}")
    lines.append(f"pd_success={pd_success}")
    lines.append(f"mt_success={mt_success}")
    lines.append(f"paired_pd_mt={len(pairs)}")
    lines.append("")

    for label, vals in (("PD", pd_vals), ("MT", mt_vals)):
        lines.append(f"[{label}] n={len(vals)}")
        if vals:
            lines.append(f"  mean={_mean(vals):.6g}")
            lines.append(f"  median={_median(vals):.6g}")
            sd = _stdev(vals)
            lines.append(f"  stdev={(sd if sd is not None else float('nan')):.6g}")
            lines.append(f"  min={min(vals):.6g}")
            lines.append(f"  max={max(vals):.6g}")
        lines.append("")

    if failed_rows:
        lines.append("Rows with missing MT or non-empty error:")
        for r in failed_rows[:20]:
            lines.append(
                f"  key={r.get('key','')} method={r.get('method','')} mt={r.get('mt_distance','')} error={r.get('error','')}"
            )
        if len(failed_rows) > 20:
            lines.append(f"  ... {len(failed_rows)-20} more rows omitted")

    out.write_text("\n".join(lines))
    return out


def plot_all(pd_vals: List[float], mt_vals: List[float], pairs: List[Tuple[float, float]], outpath: Path) -> None:
    fig = plt.figure(figsize=(12, 8))

    ax1 = fig.add_subplot(221)
    ax1.set_title("PD bottleneck distances")
    if pd_vals:
        ax1.hist(pd_vals, bins=min(30, max(5, len(pd_vals))))
    ax1.set_xlabel("PD distance")
    ax1.set_ylabel("count")
    ax1.grid(True, linestyle="--", linewidth=0.5, alpha=0.3)

    ax2 = fig.add_subplot(222)
    ax2.set_title("Merge tree distances")
    if mt_vals:
        ax2.hist(mt_vals, bins=min(30, max(5, len(mt_vals))))
    ax2.set_xlabel("MT distance")
    ax2.set_ylabel("count")
    ax2.grid(True, linestyle="--", linewidth=0.5, alpha=0.3)

    ax3 = fig.add_subplot(223)
    ax3.set_title("PD vs MT (paired rows)")
    if pairs:
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        ax3.scatter(xs, ys, s=20)
    ax3.set_xlabel("PD distance")
    ax3.set_ylabel("MT distance")
    ax3.grid(True, linestyle="--", linewidth=0.5, alpha=0.3)

    ax4 = fig.add_subplot(224)
    ax4.axis("off")
    lines = [
        f"PD count: {len(pd_vals)}",
        f"MT count: {len(mt_vals)}",
        f"Paired count: {len(pairs)}",
        "",
        "Sanity tips:",
        " • paired count should match successful rows",
        " • if MT count is 0, check phase_c_results.csv error",
    ]
    ax4.text(0.01, 0.99, "\n".join(lines), va="top")

    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def plot_summary(pd_vals: List[float], mt_vals: List[float], outpath: Path) -> None:
    fig = plt.figure(figsize=(8, 4.5))
    ax = fig.add_subplot(111)
    ax.set_title("Phase C — summary (REAL distances)")

    labels: List[str] = []
    means: List[float] = []
    if pd_vals:
        labels.append("PD")
        means.append(_mean(pd_vals) or 0.0)
    if mt_vals:
        labels.append("MT")
        means.append(_mean(mt_vals) or 0.0)

    x = list(range(len(labels)))
    ax.bar(x, means)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("mean distance")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.3)

    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def plot_method_comparison(grouped: Dict[str, Dict[str, List[float]]], outpath: Path) -> None:
    methods = sorted(grouped.keys())
    if len(methods) < 2:
        return

    fig = plt.figure(figsize=(12, 8))

    ax1 = fig.add_subplot(221)
    ax1.set_title("PD distance by method")
    pd_data = [grouped[m]["pd"] for m in methods]
    ax1.boxplot(pd_data, tick_labels=methods, showfliers=True)
    ax1.set_ylabel("PD distance")
    ax1.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.3)

    ax2 = fig.add_subplot(222)
    ax2.set_title("MT distance by method")
    mt_data = [grouped[m]["mt"] for m in methods]
    ax2.boxplot(mt_data, tick_labels=methods, showfliers=True)
    ax2.set_ylabel("MT distance")
    ax2.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.3)

    ax3 = fig.add_subplot(223)
    ax3.set_title("PD vs MT by method")
    for method in methods:
        xs = grouped[method]["pair_pd"]
        ys = grouped[method]["pair_mt"]
        if xs and ys:
            ax3.scatter(xs, ys, s=24, label=method)
    ax3.set_xlabel("PD distance")
    ax3.set_ylabel("MT distance")
    ax3.grid(True, linestyle="--", linewidth=0.5, alpha=0.3)
    ax3.legend(loc="best")

    ax4 = fig.add_subplot(224)
    ax4.axis("off")
    lines: List[str] = ["Method summary (mean values):"]
    for method in methods:
        pd_m = _mean(grouped[method]["pd"])
        mt_m = _mean(grouped[method]["mt"])
        corr = _pearson(grouped[method]["pair_pd"], grouped[method]["pair_mt"])
        lines.append(
            f"{method}: PD={pd_m:.4g} MT={mt_m:.4g} corr(PD,MT)="
            f"{('nan' if corr is None else f'{corr:.4g}') }"
        )
    ax4.text(0.01, 0.99, "\n".join(lines), va="top")

    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", type=Path, required=True)
    ap.add_argument("--strict", action="store_true", help="Exit non-zero if MT successful rows are zero.")
    args = ap.parse_args()

    results = args.indir / "phase_c_results.csv"
    rows = _load_phase_c_rows(results)
    if not rows:
        raise SystemExit(f"No rows found in {results}")

    pd_vals, mt_vals, pairs, failed_rows = _extract_values(rows)

    out_all = args.indir / "phase_c_all_distances.png"
    out_summary = args.indir / "phase_c_summary.png"
    out_report = write_sanity_report(args.indir, rows, pd_vals, mt_vals, pairs, failed_rows)
    grouped = _group_by_method(rows)
    out_method_plot = args.indir / "phase_c_method_comparison.png"
    out_method_csv = write_method_summary(args.indir, grouped)

    plot_all(pd_vals, mt_vals, pairs, out_all)
    plot_summary(pd_vals, mt_vals, out_summary)
    if len(grouped) >= 2:
        plot_method_comparison(grouped, out_method_plot)

    mt_success = sum(1 for r in rows if _parse_float(r.get("mt_distance", "")) is not None)
    print(f"Loaded rows={len(rows)} PD={len(pd_vals)} MT={len(mt_vals)} paired={len(pairs)}")
    print(f"Wrote: {out_all}")
    print(f"Wrote: {out_summary}")
    print(f"Wrote: {out_report}")
    if len(grouped) >= 2:
        print(f"Wrote: {out_method_plot}")
    if out_method_csv is not None:
        print(f"Wrote: {out_method_csv}")

    if args.strict and mt_success == 0:
        raise SystemExit("STRICT MODE: MT success count is 0")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

