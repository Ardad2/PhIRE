#!/usr/bin/env python3
"""Analyse special-case component-count patterns for PhIRE baseline comparison.

Reads the component_counts_per_sample.csv produced by
analyze_component_counts_all_baselines.py and produces focused reports for
the curated sample groups defined in generate_visual_inspection_panels.py:
  - Adjacent MT-GAN cluster 10–13  (anchor: 12)
  - Adjacent MT-GAN cluster 76–78  (anchor: 77)
  - Adjacent MT-GAN cluster 90–93  (anchor: 92)
  - Ridge-rich motif 16–20         (anchors: 17, 19)
  - Rare PD-CNN/MT-CNN cluster 161–164
  - Lower-confidence limitation cases (25, 80, 154)

For each sample a per-threshold winner signature, GAN bias direction, and
an interpretation label from a fixed vocabulary are derived:
  GAN_component_strong                  — GAN wins ≥ 4 of 6 thresholds
  GAN_component_low_threshold_only      — GAN wins at low speeds only
  GAN_component_overfragmented_high_threshold — GAN over-fragments at high percentiles
  CNN_component_control                 — CNN wins ≥ 4 of 6 thresholds
  mixed_or_ambiguous                    — no dominant winner

Inputs
------
  ttk_runs_fixed/component_counts/component_counts_per_sample.csv  (required)
  ttk_runs_fixed/baseline_metrics/all_methods_per_sample.csv       (optional)

Outputs (under ttk_runs_fixed/component_counts/)
-------
  special_component_case_summary.csv    — one row per special sample
  special_component_case_thresholds.csv — one row per (sample, threshold)
  special_component_case_report.html    — grouped HTML report with panel links

Run from repo root or scripts/:
    cd ~/PhIRE
    PYTHONNOUSERSITE=1 /usr/bin/python3 scripts/analyze_special_component_cases.py
"""

from __future__ import annotations

import argparse
import csv
import html as _html
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Repo layout
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parent

COMP_CSV = REPO_ROOT / "ttk_runs_fixed" / "component_counts" / "component_counts_per_sample.csv"
PHYS_CSV = REPO_ROOT / "ttk_runs_fixed" / "baseline_metrics" / "all_methods_per_sample.csv"
OUT_DIR  = REPO_ROOT / "ttk_runs_fixed" / "component_counts"

# Relative path from OUT_DIR to baseline visual panels index.
_PANELS_REL = "../../baseline_visual_panels/index.html"


# ---------------------------------------------------------------------------
# Curated sample groups — mirrors FORCED dict in generate_visual_inspection_panels.py
# ---------------------------------------------------------------------------
FORCED: Dict[int, str] = {
    10: "adjacent control before sample 12",
    11: "adjacent control before sample 12",
    12: "strong MT-GAN anchor",
    13: "adjacent control after sample 12",

    76: "adjacent control before sample 77",
    77: "strong MT-GAN anchor",
    78: "adjacent control after sample 77",

    90: "GAN-majority / MT-CNN adjacent control near sample 92",
    91: "GAN-majority / MT-CNN adjacent control near sample 92",
    92: "strong MT-GAN anchor",
    93: "GAN-majority / MT-CNN adjacent control near sample 92",

    161: "adjacent control before rare topology-CNN samples 162-163",
    162: "rare PD-CNN and MT-CNN topology-consensus control",
    163: "rare PD-CNN and MT-CNN topology-consensus control",
    164: "adjacent control after rare topology-CNN samples 162-163",

    16: "moderate MT-GAN ridge-rich motif",
    17: "strong MT-GAN anchor",
    18: "moderate/lower MT-GAN ridge-rich motif",
    19: "strong MT-GAN anchor",
    20: "moderate/lower MT-GAN ridge-rich motif",

    25: "lower-confidence MT-GAN limitation case",
    80: "lower-confidence MT-GAN limitation case",
    154: "lower-confidence MT-GAN limitation case",
}

# Each group: (id, display title, list of sample indices)
GROUPS: List[Tuple[str, str, List[int]]] = [
    ("cluster_10_13",     "Adjacent cluster 10–13 (MT-GAN anchor: 12)",              [10, 11, 12, 13]),
    ("cluster_76_78",     "Adjacent cluster 76–78 (MT-GAN anchor: 77)",              [76, 77, 78]),
    ("cluster_90_93",     "Adjacent cluster 90–93 (MT-GAN anchor: 92)",              [90, 91, 92, 93]),
    ("ridge_motif_16_20", "Ridge-rich motif 16–20 (MT-GAN anchors: 17, 19)",         [16, 17, 18, 19, 20]),
    ("cluster_161_164",   "Rare PD-CNN/MT-CNN cluster 161–164",                      [161, 162, 163, 164]),
    ("limitation_cases",  "Lower-confidence MT-GAN limitation cases (25, 80, 154)",  [25, 80, 154]),
]

METHODS     = ("bicubic", "cnn", "gan")
ALL_TLABELS = ["t5", "t10", "t15", "p90", "p95", "p99"]
LOW_TLABELS  = ["t5", "t10", "t15"]    # fixed-threshold, lower speeds
HIGH_TLABELS = ["p90", "p95", "p99"]   # percentile-threshold, high speeds

# Physics columns to use when computing per-sample physics winner (lower-is-better)
_PHYS_COLS = [
    "wpd_mae", "wpd_rmse", "wpd_w1",
    "psd_log_l2", "psd_slope_abs_delta",
    "grad_mae", "grad_w1", "grad_kurtosis_abs_delta",
    "exceed_frac_abs_delta_t5",  "exceed_frac_abs_delta_t10",
    "exceed_frac_abs_delta_t15", "exceed_frac_abs_delta_p90",
    "exceed_frac_abs_delta_p95", "exceed_frac_abs_delta_p99",
]


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def _load_component_csv(path: Path) -> Dict[int, Dict[str, Dict[str, str]]]:
    """Load component_counts_per_sample.csv → {sample_idx: {threshold_label: row}}."""
    if not path.exists():
        print(f"[error] Required component CSV not found:\n  {path}", file=sys.stderr)
        sys.exit(1)

    data: Dict[int, Dict[str, Dict[str, str]]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                sid = int(float(row["sample_idx"]))
            except (KeyError, ValueError):
                continue
            tlabel = row.get("threshold_label", "").strip()
            if tlabel:
                data.setdefault(sid, {})[tlabel] = dict(row)
    return data


def _load_physics_csv(path: Path) -> Optional[Dict[int, Dict[str, str]]]:
    """Load all_methods_per_sample.csv and compute per-sample physics winner.

    Returns None when the file is absent or unparseable.
    Each entry: {"phys_bicubic_wins": str, "phys_cnn_wins": str,
                 "phys_gan_wins": str, "phys_winner": str}.
    """
    if not path.exists():
        print(f"[info] Physics CSV not found (optional): {path}")
        return None

    # Pivot to {sample_idx: {method: {col: val}}}
    by_sample: Dict[int, Dict[str, Dict[str, str]]] = {}
    try:
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sid = None
                for k in ("sample_idx", "sample_id", "sample"):
                    if k in row:
                        try:
                            sid = int(float(row[k]))
                        except (ValueError, TypeError):
                            pass
                        break
                if sid is None:
                    continue

                raw = row.get("method", "").strip().lower()
                method = next((m for m in METHODS if m in raw), None)
                if method is None:
                    continue

                by_sample.setdefault(sid, {})[method] = dict(row)
    except Exception as exc:
        print(f"[info] Could not parse physics CSV: {exc}")
        return None

    if not by_sample:
        return None

    result: Dict[int, Dict[str, str]] = {}
    for sid, by_method in by_sample.items():
        wins: Dict[str, int] = {m: 0 for m in METHODS}
        for col in _PHYS_COLS:
            vals: Dict[str, float] = {}
            for m in METHODS:
                if m not in by_method:
                    continue
                try:
                    v = float(by_method[m].get(col, "nan"))
                    if not math.isnan(v):
                        vals[m] = v
                except (ValueError, TypeError):
                    pass
            if not vals:
                continue
            min_v = min(vals.values())
            best  = [m for m, v in vals.items() if v == min_v]
            if len(best) == 1:
                wins[best[0]] += 1

        top = max(wins, key=wins.__getitem__)
        result[sid] = {
            "phys_bicubic_wins": str(wins["bicubic"]),
            "phys_cnn_wins":     str(wins["cnn"]),
            "phys_gan_wins":     str(wins["gan"]),
            "phys_winner":       top,
        }

    return result


# ---------------------------------------------------------------------------
# Per-sample analysis
# ---------------------------------------------------------------------------

def _interpret(
    gan_wins: int,
    cnn_wins: int,
    gan_wins_low: int,
    gan_wins_high: int,
    gan_mean_bias: float,
) -> str:
    """Assign one of five interpretation labels."""
    if gan_wins >= 4:
        return "GAN_component_strong"
    if gan_wins_low >= 2 and gan_wins_high <= 1:
        return "GAN_component_low_threshold_only"
    bias_ok = not math.isnan(gan_mean_bias)
    if gan_wins_high == 0 and bias_ok and gan_mean_bias > 1.0:
        return "GAN_component_overfragmented_high_threshold"
    if cnn_wins >= 4:
        return "CNN_component_control"
    return "mixed_or_ambiguous"


def analyse_sample(sid: int, thresholds: Dict[str, Dict[str, str]]) -> Dict:
    """Derive summary statistics for one sample from its 6 threshold rows."""
    wins:  Dict[str, int] = {m: 0 for m in METHODS}
    ties   = 0
    sig_parts: List[str] = []
    gan_biases: List[float] = []
    gan_wins_low  = 0
    gan_wins_high = 0

    for tl in ALL_TLABELS:
        row = thresholds.get(tl, {})
        w   = row.get("winner", "").strip()

        if w in METHODS:
            wins[w] += 1
        elif w == "tie":
            ties += 1

        sig_parts.append(f"{tl}={w or '?'}")

        try:
            gan_n = float(row.get("gan_count", "nan"))
            gt_n  = float(row.get("gt_count",  "nan"))
            if not (math.isnan(gan_n) or math.isnan(gt_n)):
                gan_biases.append(gan_n - gt_n)
        except (ValueError, TypeError):
            pass

        if w == "gan":
            if tl in LOW_TLABELS:
                gan_wins_low  += 1
            else:
                gan_wins_high += 1

    gan_mean_bias = float(np.mean(gan_biases)) if gan_biases else float("nan")

    if not math.isnan(gan_mean_bias):
        if gan_mean_bias > 0.5:
            gan_direction = "over_fragmented"
        elif gan_mean_bias < -0.5:
            gan_direction = "under_fragmented"
        else:
            gan_direction = "accurate"
    else:
        gan_direction = "unknown"

    interpretation = _interpret(
        wins["gan"], wins["cnn"], gan_wins_low, gan_wins_high, gan_mean_bias
    )

    return {
        "sample_idx":          sid,
        "forced_note":         FORCED.get(sid, ""),
        "bicubic_wins":        wins["bicubic"],
        "cnn_wins":            wins["cnn"],
        "gan_wins":            wins["gan"],
        "tie_count":           ties,
        "gan_wins_low":        gan_wins_low,
        "gan_wins_high":       gan_wins_high,
        "winner_signature":    ",".join(sig_parts),
        "gan_mean_bias":       f"{gan_mean_bias:.3f}" if not math.isnan(gan_mean_bias) else "nan",
        "gan_direction":       gan_direction,
        "interpretation_label": interpretation,
    }


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def _write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    cols = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

_INTERP_DESC: Dict[str, str] = {
    "GAN_component_strong":
        "GAN wins ≥ 4 of 6 thresholds — consistently better component structure",
    "GAN_component_low_threshold_only":
        "GAN leads at low-speed thresholds only; advantage fades at high speeds",
    "GAN_component_overfragmented_high_threshold":
        "GAN over-fragments at high speed percentiles relative to GT",
    "CNN_component_control":
        "CNN wins majority of component comparisons across all thresholds",
    "mixed_or_ambiguous":
        "No method shows a clear component-count advantage",
}

_INTERP_BG: Dict[str, str] = {
    "GAN_component_strong":                       "#bbf7d0",
    "GAN_component_low_threshold_only":            "#fef08a",
    "GAN_component_overfragmented_high_threshold": "#fecaca",
    "CNN_component_control":                       "#bfdbfe",
    "mixed_or_ambiguous":                          "#e5e7eb",
}

_WINNER_BG: Dict[str, str] = {
    "bicubic": "#dbeafe",
    "cnn":     "#dbeafe",
    "gan":     "#bbf7d0",
    "tie":     "#fef9c3",
}


def _esc(s: object) -> str:
    return _html.escape(str(s))


def _fmt(val: object) -> str:
    try:
        return f"{float(val):.2f}"   # type: ignore[arg-type]
    except (ValueError, TypeError):
        return _esc(val)


def _threshold_table_html(thresholds: Dict[str, Dict[str, str]]) -> str:
    rows_html = ""
    for tl in ALL_TLABELS:
        row = thresholds.get(tl, {})
        w   = row.get("winner", "").strip()
        bg  = _WINNER_BG.get(w, "#f9fafb")

        def _c(col: str) -> str:
            return _fmt(row.get(col, "—"))

        rows_html += (
            f"<tr>"
            f"<td>{_esc(tl)}</td>"
            f"<td>{_esc(row.get('threshold_name', ''))}</td>"
            f"<td>{_esc(row.get('threshold_type', ''))}</td>"
            f"<td>{_c('threshold_value')}</td>"
            f"<td>{_c('gt_count')}</td>"
            f"<td>{_c('bicubic_count')}</td>"
            f"<td>{_c('cnn_count')}</td>"
            f"<td>{_c('gan_count')}</td>"
            f"<td>{_c('bicubic_abs_error')}</td>"
            f"<td>{_c('cnn_abs_error')}</td>"
            f"<td>{_c('gan_abs_error')}</td>"
            f"<td style='background:{bg};font-weight:600'>{_esc(w or '—')}</td>"
            f"</tr>\n"
        )

    return (
        "<table class='data-table'>"
        "<thead><tr>"
        "<th>Label</th><th>Name</th><th>Type</th><th>Value</th>"
        "<th>GT</th><th>Bicubic</th><th>CNN</th><th>GAN</th>"
        "<th>Bic|err|</th><th>CNN|err|</th><th>GAN|err|</th>"
        "<th>Winner</th>"
        "</tr></thead>"
        f"<tbody>{rows_html}</tbody>"
        "</table>"
    )


def _sample_card_html(
    sid: int,
    comp_data: Dict[int, Dict[str, Dict[str, str]]],
    summaries: Dict[int, Dict],
    physics: Optional[Dict[int, Dict[str, str]]],
    panels_rel: str,
) -> str:
    if sid not in comp_data:
        return f"<p class='muted'>[info] Sample {sid} not found in component CSV.</p>\n"

    s      = summaries.get(sid, {})
    interp = s.get("interpretation_label", "mixed_or_ambiguous")
    bg     = _INTERP_BG.get(interp, "#e5e7eb")
    desc   = _INTERP_DESC.get(interp, interp)

    phys_html = ""
    if physics and sid in physics:
        p = physics[sid]
        phys_html = (
            "<p style='margin:6px 0'>"
            f"<b>Physics winner:</b> {_esc(p.get('phys_winner', '—'))} "
            f"(Bic {p.get('phys_bicubic_wins', '—')}, "
            f"CNN {p.get('phys_cnn_wins', '—')}, "
            f"GAN {p.get('phys_gan_wins', '—')} wins/{len(_PHYS_COLS)})"
            "</p>"
        )

    panel_href = f"{panels_rel}#sample-{sid}"

    return (
        f"<div class='card sample-card' id='sc-{sid}'>"
        f"<div style='display:flex;justify-content:space-between;align-items:flex-start'>"
        f"<div>"
        f"<h3 style='margin:0'>Sample {sid}</h3>"
        f"<p style='color:#6b7280;margin:2px 0'>{_esc(s.get('forced_note', ''))}</p>"
        f"</div>"
        f"<a href='{panel_href}' target='_blank' style='font-size:0.85em;white-space:nowrap'>View panel ↗</a>"
        f"</div>"
        f"<div style='margin:8px 0;padding:8px 12px;background:{bg};border-radius:6px'>"
        f"<b>{_esc(interp)}</b><br>"
        f"<span style='font-size:0.9em;color:#374151'>{_esc(desc)}</span>"
        f"</div>"
        f"<div style='display:flex;gap:24px;flex-wrap:wrap;margin:6px 0'>"
        f"<div><b>GAN wins:</b> {s.get('gan_wins', '—')}/6 "
        f"(low={s.get('gan_wins_low', '—')}, high={s.get('gan_wins_high', '—')})</div>"
        f"<div><b>CNN wins:</b> {s.get('cnn_wins', '—')}/6</div>"
        f"<div><b>GAN bias:</b> {_esc(s.get('gan_direction', '—'))} "
        f"({s.get('gan_mean_bias', '—')} components/threshold)</div>"
        f"</div>"
        f"<p style='margin:4px 0'><b>Winner signature:</b> "
        f"<code>{_esc(s.get('winner_signature', '—'))}</code></p>"
        f"{phys_html}"
        + _threshold_table_html(comp_data[sid])
        + "</div>\n"
    )


def _write_html(
    path: Path,
    comp_data: Dict[int, Dict[str, Dict[str, str]]],
    summaries: Dict[int, Dict],
    physics: Optional[Dict[int, Dict[str, str]]],
) -> None:
    # path is inside OUT_DIR, so panels_rel is relative from there
    panels_rel = _PANELS_REL

    nav_links = "\n".join(
        f'<a href="#{gid}">{_esc(gtitle)}</a>'
        for gid, gtitle, _ in GROUPS
    )

    sections = ""
    for gid, gtitle, samples in GROUPS:
        cards = "".join(
            _sample_card_html(sid, comp_data, summaries, physics, panels_rel)
            for sid in samples
        )
        sections += (
            f"<section class='card group-card' id='{gid}'>"
            f"<h2 style='margin-top:0'>{_esc(gtitle)}</h2>"
            f"{cards}"
            f"</section>\n"
        )

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Special Component Case Report – PhIRE</title>
<style>
  body {{
    font-family: system-ui, sans-serif;
    background: #f0f2f5;
    margin: 0;
    padding: 24px;
    color: #111827;
  }}
  h1 {{ font-size: 1.5rem; margin-bottom: 4px; }}
  h2 {{ font-size: 1.2rem; }}
  h3 {{ font-size: 1rem; margin-bottom: 0; }}
  .card {{
    background: #fff;
    border: 1px solid #dde1e7;
    border-radius: 8px;
    box-shadow: 0 1px 4px rgba(0,0,0,.07);
    padding: 20px;
    margin-bottom: 16px;
  }}
  .group-card {{ margin-bottom: 32px; }}
  .sample-card {{ margin-bottom: 12px; border-left: 4px solid #e5e7eb; }}
  .muted {{ color: #9ca3af; font-style: italic; }}
  .data-table {{
    border-collapse: collapse;
    font-size: 0.83em;
    width: 100%;
    margin-top: 10px;
  }}
  .data-table th, .data-table td {{
    border: 1px solid #d1d5db;
    padding: 4px 8px;
    text-align: right;
    white-space: nowrap;
  }}
  .data-table th:nth-child(-n+3),
  .data-table td:nth-child(-n+3) {{ text-align: left; }}
  .data-table thead {{ background: #f3f4f6; }}
  code {{
    background: #f3f4f6;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 0.88em;
  }}
  nav.jump-nav {{
    background: #fff;
    border: 1px solid #dde1e7;
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 24px;
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
  }}
  nav.jump-nav a {{
    padding: 4px 10px;
    background: #f3f4f6;
    border-radius: 4px;
    text-decoration: none;
    color: #1d4ed8;
    font-size: 0.88em;
  }}
  nav.jump-nav a:hover {{ background: #dbeafe; }}
</style>
</head>
<body>
<h1>Special Component-Count Case Report</h1>
<p style="color:#6b7280;margin-bottom:20px">
  Topology proxy analysis: 8-connected superlevel-set component counts for GT,
  Bicubic, CNN, and GAN across six thresholds
  (t5/t10/t15&nbsp;=&nbsp;fixed&nbsp;m/s; p90/p95/p99&nbsp;=&nbsp;GT percentile).
  Winner = method whose count is closest to GT.
</p>

<nav class="jump-nav">
  <b style="align-self:center">Jump:&nbsp;</b>
  {nav_links}
</nav>

{sections}
</body>
</html>
"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page, encoding="utf-8")
    print(f"  Wrote: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Analyse special-case component-count patterns for curated sample groups."
    )
    ap.add_argument(
        "--comp-csv", type=Path, default=COMP_CSV,
        help=f"component_counts_per_sample.csv (default: {COMP_CSV})",
    )
    ap.add_argument(
        "--phys-csv", type=Path, default=PHYS_CSV,
        help=f"all_methods_per_sample.csv — optional physics winner data (default: {PHYS_CSV})",
    )
    ap.add_argument(
        "--outdir", type=Path, default=OUT_DIR,
        help=f"Output directory (default: {OUT_DIR})",
    )
    args = ap.parse_args()

    print("=" * 60)
    print("PhIRE special component case analyser")
    print(f"  comp CSV : {args.comp_csv}")
    print(f"  phys CSV : {args.phys_csv}")
    print(f"  output   : {args.outdir}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load component counts (required)
    # ------------------------------------------------------------------
    print("\n[1] Loading component counts CSV …")
    comp_data = _load_component_csv(args.comp_csv)
    print(f"  Loaded {len(comp_data)} samples × {len(ALL_TLABELS)} thresholds")

    # ------------------------------------------------------------------
    # 2. Load physics metrics (optional)
    # ------------------------------------------------------------------
    print("\n[2] Loading physics metrics CSV (optional) …")
    physics = _load_physics_csv(args.phys_csv)
    if physics:
        print(f"  Loaded physics winner data for {len(physics)} samples")

    # ------------------------------------------------------------------
    # 3. Analyse each curated sample
    # ------------------------------------------------------------------
    print("\n[3] Analysing curated sample groups …")
    all_special: List[int] = sorted({sid for _, _, sids in GROUPS for sid in sids})

    summaries: Dict[int, Dict] = {}
    for sid in all_special:
        if sid not in comp_data:
            print(f"  [warn] Sample {sid} not found in component CSV; skipping")
            continue
        summaries[sid] = analyse_sample(sid, comp_data[sid])

    print(f"  Analysed {len(summaries)} / {len(all_special)} requested samples")

    # ------------------------------------------------------------------
    # 4. Build output rows
    # ------------------------------------------------------------------

    # Summary: one row per sample, with group name prepended
    summary_rows: List[Dict] = []
    for gid, gtitle, samples in GROUPS:
        for sid in samples:
            if sid not in summaries:
                continue
            row: Dict = {"group_name": gid, "group_title": gtitle}
            row.update(summaries[sid])
            if physics and sid in physics:
                row.update(physics[sid])
            summary_rows.append(row)

    # Thresholds: one row per (sample, threshold) for all special samples
    threshold_rows: List[Dict] = []
    for gid, _, samples in GROUPS:
        for sid in samples:
            if sid not in comp_data:
                continue
            for tl in ALL_TLABELS:
                r = comp_data[sid].get(tl)
                if r is None:
                    continue
                tr: Dict = {"group_name": gid, "sample_idx": sid}
                tr.update(r)
                threshold_rows.append(tr)

    # ------------------------------------------------------------------
    # 5. Write CSVs
    # ------------------------------------------------------------------
    print("\n[4] Writing CSVs …")
    args.outdir.mkdir(parents=True, exist_ok=True)

    p_summary    = args.outdir / "special_component_case_summary.csv"
    p_thresholds = args.outdir / "special_component_case_thresholds.csv"

    _write_csv(p_summary, summary_rows)
    print(f"  Wrote: {p_summary}  ({len(summary_rows)} rows)")

    _write_csv(p_thresholds, threshold_rows)
    print(f"  Wrote: {p_thresholds}  ({len(threshold_rows)} rows)")

    # ------------------------------------------------------------------
    # 6. Write HTML report
    # ------------------------------------------------------------------
    print("\n[5] Writing HTML report …")
    _write_html(
        args.outdir / "special_component_case_report.html",
        comp_data,
        summaries,
        physics,
    )

    # ------------------------------------------------------------------
    # 7. Interpretation summary
    # ------------------------------------------------------------------
    print("\n[6] Interpretation label distribution across curated samples:")
    counts = Counter(s.get("interpretation_label", "") for s in summaries.values())
    for label, n in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {label:<52s}: {n}")

    print("\n[done] Special component case analysis complete.")
    print(f"  Output directory: {args.outdir}")


if __name__ == "__main__":
    main()
