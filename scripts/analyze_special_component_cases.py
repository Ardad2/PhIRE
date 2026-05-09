#!/usr/bin/env python3
"""Summarize special component-count cases with manifest-enriched winners and HTML report."""

from __future__ import annotations

import argparse
import csv
import html
from pathlib import Path
from typing import Dict, List, Tuple

_METHODS = ["bicubic", "cnn", "gan"]
_PHYS_COLS = [
    "wpd_bias",
    "wpd_rmse",
    "wpd_mae",
    "wpd_w1",
    "psd_log_l2",
    "psd_slope_abs_delta",
    "grad_mae",
    "grad_w1",
    "grad_kurtosis_abs_delta",
    "exceed_frac_abs_delta_t5",
    "exceed_frac_abs_delta_t10",
    "exceed_frac_abs_delta_t15",
    "exceed_frac_abs_delta_p90",
    "exceed_frac_abs_delta_p95",
    "exceed_frac_abs_delta_p99",
]
_LOW_KEYS = ["t5", "t10"]
_HIGH_KEYS = ["t15", "p90", "p95", "p99"]
_MANIFEST_COLS = [
    "pd_winner",
    "mt_winner",
    "direct_error_group_winner",
    "distributional_group_winner",
    "tail_group_winner",
    "configured_physics_group_winner",
    "groups",
    "question",
]


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def _to_int(x: str) -> int:
    return int(float(str(x).strip()))


def _to_float(x: str) -> float:
    return float(str(x).strip())


def _winner_lower(values: Dict[str, float]) -> Tuple[str, Dict[str, int]]:
    best = min(values.values())
    winners = [m for m, v in values.items() if v == best]
    counts = {m: 0 for m in _METHODS}
    if len(winners) == 1:
        counts[winners[0]] = 1
        return winners[0], counts
    return "TIE", counts


def _get_metric_values(row: Dict[str, str], metric: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for m in _METHODS:
        key = f"{m}_{metric}"
        if key in row and row[key] not in {"", "na", "NA", "NaN"}:
            out[m] = _to_float(row[key])
    return out


def _write_csv(path: Path, rows: List[Dict[str, object]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _curated_groups() -> Dict[str, List[int]]:
    return {
        "curated_10_13": [10, 11, 12, 13],
        "curated_76_78": [76, 77, 78],
        "curated_90_93": [90, 91, 92, 93],
        "curated_161_164": [161, 162, 163, 164],
        "curated_16_20": [16, 17, 18, 19, 20],
        "curated_25_80_154": [25, 80, 154],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--component-csv", type=Path, default=Path("ttk_runs_fixed/component_counts/component_counts_merged.csv"))
    ap.add_argument("--manifest-csv", type=Path, default=Path("ttk_runs_fixed/baseline_visual_panels/baseline_visual_manifest.csv"))
    ap.add_argument("--outdir", type=Path, default=Path("ttk_runs_fixed/component_counts"))
    args = ap.parse_args()

    rows = _read_csv(args.component_csv)
    manifest_rows = _read_csv(args.manifest_csv)
    manifest_by_idx = {_to_int(r.get("sample_idx", r.get("sample", ""))): r for r in manifest_rows}

    by_idx = {_to_int(r.get("sample_idx", r.get("sample", ""))): r for r in rows}
    groups = _curated_groups()

    # Dynamic groups from manifest
    all_mt_gan = [sid for sid, m in manifest_by_idx.items() if str(m.get("mt_winner", "")).strip().upper() == "GAN"]
    groups["all_mt_gan"] = sorted(all_mt_gan)

    gm: List[int] = []
    for sid, m in manifest_by_idx.items():
        gtxt = str(m.get("groups", ""))
        mt = str(m.get("mt_winner", "")).strip().upper()
        direct = str(m.get("direct_error_group_winner", "")).strip().upper()
        distr = str(m.get("distributional_group_winner", "")).strip().upper()
        tail = str(m.get("tail_group_winner", "")).strip().upper()
        phys = str(m.get("configured_physics_group_winner", "")).strip().upper()
        votes = [v for v in [direct, distr, tail, phys] if v in {"BICUBIC", "CNN", "GAN"}]
        majority_gan = votes.count("GAN") > max(votes.count("CNN"), votes.count("BICUBIC")) if votes else False
        if "gan_majority_mt_rejects_gan" in gtxt or (majority_gan and mt != "GAN"):
            gm.append(sid)
    groups["gan_majority_but_mt_not_gan"] = sorted(set(gm))

    threshold_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []

    for gname, sids in groups.items():
        for sid in sids:
            if sid not in by_idx:
                continue
            row = by_idx[sid]
            mrow = manifest_by_idx.get(sid, {})

            bic, cnn, gan, ties = 0, 0, 0, 0
            gan_low = 0
            gan_high = 0
            bias_sum = 0.0
            bias_n = 0
            phys_counts = {"bicubic": 0, "cnn": 0, "gan": 0}

            for metric in _PHYS_COLS:
                vals = _get_metric_values(row, metric)
                if len(vals) < 2:
                    continue
                winner, onehot = _winner_lower(vals)
                if winner == "TIE":
                    ties += 1
                else:
                    if winner == "bicubic":
                        bic += 1
                    elif winner == "cnn":
                        cnn += 1
                    else:
                        gan += 1
                    if metric.endswith("_t5") or metric.endswith("_t10"):
                        if winner == "gan":
                            gan_low += 1
                    if metric.endswith("_t15") or metric.endswith("_p90") or metric.endswith("_p95") or metric.endswith("_p99"):
                        if winner == "gan":
                            gan_high += 1
                    if metric in _PHYS_COLS:
                        phys_counts[winner] += 1
                threshold_rows.append({
                    "sample_idx": sid,
                    "group_name": gname,
                    "metric": metric,
                    "winner": winner,
                    "bicubic": vals.get("bicubic", ""),
                    "cnn": vals.get("cnn", ""),
                    "gan": vals.get("gan", ""),
                })

            if "gan_wpd_bias" in row and "bicubic_wpd_bias" in row and "cnn_wpd_bias" in row:
                try:
                    bias_sum = _to_float(row["gan_wpd_bias"])
                    bias_n = 1
                except Exception:
                    pass

            winner_signature = f"B{bic}-C{cnn}-G{gan}-T{ties}"
            gan_mean_bias = (bias_sum / bias_n) if bias_n else 0.0
            gan_direction = "positive" if gan_mean_bias > 0 else ("negative" if gan_mean_bias < 0 else "zero")

            phys_winner = max(phys_counts.items(), key=lambda kv: kv[1])[0].upper() if sum(phys_counts.values()) else "TIE"
            label = "gan_dominant" if gan > max(cnn, bic) else ("cnn_dominant" if cnn > max(gan, bic) else ("bicubic_dominant" if bic > max(gan, cnn) else "mixed"))

            out = {
                "sample_idx": sid,
                "group_name": gname,
                "pd_winner": mrow.get("pd_winner", ""),
                "mt_winner": mrow.get("mt_winner", ""),
                "direct_error_group_winner": mrow.get("direct_error_group_winner", ""),
                "distributional_group_winner": mrow.get("distributional_group_winner", ""),
                "tail_group_winner": mrow.get("tail_group_winner", ""),
                "configured_physics_group_winner": mrow.get("configured_physics_group_winner", ""),
                "bicubic_wins": bic,
                "cnn_wins": cnn,
                "gan_wins": gan,
                "tie_count": ties,
                "gan_wins_low": gan_low,
                "gan_wins_high": gan_high,
                "winner_signature": winner_signature,
                "gan_mean_bias": gan_mean_bias,
                "gan_direction": gan_direction,
                "interpretation_label": label,
                "phys_bicubic_wins": phys_counts["bicubic"],
                "phys_cnn_wins": phys_counts["cnn"],
                "phys_gan_wins": phys_counts["gan"],
                "phys_winner": phys_winner,
                "groups": mrow.get("groups", ""),
                "question": mrow.get("question", ""),
            }
            summary_rows.append(out)

    summary_rows.sort(key=lambda r: (str(r["group_name"]), int(r["sample_idx"])))
    threshold_rows.sort(key=lambda r: (str(r["group_name"]), int(r["sample_idx"]), str(r["metric"])))

    summary_fields = [
        "sample_idx",
        "group_name",
        "pd_winner",
        "mt_winner",
        "direct_error_group_winner",
        "distributional_group_winner",
        "tail_group_winner",
        "configured_physics_group_winner",
        "bicubic_wins",
        "cnn_wins",
        "gan_wins",
        "tie_count",
        "gan_wins_low",
        "gan_wins_high",
        "winner_signature",
        "gan_mean_bias",
        "gan_direction",
        "interpretation_label",
        "phys_bicubic_wins",
        "phys_cnn_wins",
        "phys_gan_wins",
        "phys_winner",
    ]
    _write_csv(args.outdir / "special_component_case_summary.csv", summary_rows, summary_fields)
    _write_csv(args.outdir / "special_component_case_thresholds.csv", threshold_rows, ["sample_idx", "group_name", "metric", "winner", "bicubic", "cnn", "gan"])

    html_rows = []
    for r in summary_rows:
        sid = int(r["sample_idx"])
        link = f"../baseline_visual_panels/index.html#sample-{sid}"
        html_rows.append(
            "<tr>"
            f"<td>{sid}</td><td>{html.escape(str(r['group_name']))}</td>"
            f"<td>{html.escape(str(r['pd_winner']))}</td><td>{html.escape(str(r['mt_winner']))}</td>"
            f"<td>{html.escape(str(r['winner_signature']))}</td>"
            f"<td><a href=\"{link}\">baseline panel</a></td>"
            f"<td>{html.escape(str(r.get('groups','')))}</td>"
            f"<td>{html.escape(str(r.get('question','')))}</td>"
            "</tr>"
        )
    report = (
        "<html><body><h1>Special Component Cases</h1>"
        "<table border='1'><thead><tr>"
        "<th>sample_idx</th><th>group</th><th>pd_winner</th><th>mt_winner</th><th>winner_signature</th><th>baseline</th><th>groups</th><th>question</th>"
        "</tr></thead><tbody>"
        + "".join(html_rows)
        + "</tbody></table></body></html>"
    )
    (args.outdir / "special_component_case_report.html").write_text(report)


if __name__ == "__main__":
    main()
