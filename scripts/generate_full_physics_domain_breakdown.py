#!/usr/bin/env python3
"""
Generate full physics/domain metric breakdown tables for all 168 samples.

Run from:
    cd ~/PhIRE/scripts
    PYTHONNOUSERSITE=1 /usr/bin/python3 generate_full_physics_domain_breakdown.py

Expected input:
    ~/PhIRE/ttk_runs_fixed/report_tables/metric_sweep_all_samples_wide.csv

Optional inputs:
    ~/PhIRE/ttk_runs_fixed/observation_groups/observation_groups_per_sample.csv
    ~/PhIRE/ttk_runs_fixed/visual_inspection/visual_inspection_manifest.csv

Outputs:
    ~/PhIRE/ttk_runs_fixed/report_tables/full_physics_domain_breakdown/
"""

from __future__ import annotations

import csv
import html
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def find_repo_root() -> Path:
    here = Path(__file__).resolve().parent
    cwd = Path.cwd().resolve()
    candidates = [
        here.parent if here.name == "scripts" else here,
        cwd.parent if cwd.name == "scripts" else cwd,
        here,
        cwd,
    ]
    for p in candidates:
        if (p / "ttk_runs_fixed").exists():
            return p
    raise FileNotFoundError("Could not find repo root containing ttk_runs_fixed/")


ROOT = find_repo_root()
INPUT_WIDE = ROOT / "ttk_runs_fixed" / "report_tables" / "metric_sweep_all_samples_wide.csv"
OBS_PER_SAMPLE = ROOT / "ttk_runs_fixed" / "observation_groups" / "observation_groups_per_sample.csv"
VIS_MANIFEST = ROOT / "ttk_runs_fixed" / "visual_inspection" / "visual_inspection_manifest.csv"
OUTDIR = ROOT / "ttk_runs_fixed" / "report_tables" / "full_physics_domain_breakdown"


@dataclass(frozen=True)
class MetricSpec:
    key: str
    label: str
    group: str
    subgroup: str
    signed_abs: bool = False


METRICS: list[MetricSpec] = [
    MetricSpec("wpd_bias", "WPD Bias |·|", "Physics/WPD", "wpd", True),
    MetricSpec("wpd_mae", "WPD MAE", "Physics/WPD", "wpd", False),
    MetricSpec("wpd_rmse", "WPD RMSE", "Physics/WPD", "wpd", False),
    MetricSpec("wpd_w1", "WPD Wasserstein-1", "Distributional", "distributional", False),
    MetricSpec("psd_log_l2", "PSD log-L2", "Distributional", "distributional", False),
    MetricSpec("psd_slope_abs_delta", "PSD slope |Δ|", "Distributional", "distributional", False),
    MetricSpec("grad_mae", "Gradient MAE", "Physics/Gradient", "gradient", False),
    MetricSpec("grad_w1", "Gradient Wasserstein-1", "Distributional", "distributional", False),
    MetricSpec("grad_kurtosis_abs_delta", "Gradient kurtosis |Δ|", "Distributional", "distributional", False),
    MetricSpec("exceed_frac_abs_delta_t5", "Exceedance |Δ|, s>5", "Tail/Exceedance", "tail", False),
    MetricSpec("exceed_frac_abs_delta_t10", "Exceedance |Δ|, s>10", "Tail/Exceedance", "tail", False),
    MetricSpec("exceed_frac_abs_delta_t15", "Exceedance |Δ|, s>15", "Tail/Exceedance", "tail", False),
    MetricSpec("exceed_frac_abs_delta_p90", "Exceedance |Δ|, p90", "Tail/Exceedance", "tail", False),
    MetricSpec("exceed_frac_abs_delta_p95", "Exceedance |Δ|, p95", "Tail/Exceedance", "tail", False),
    MetricSpec("exceed_frac_abs_delta_p99", "Exceedance |Δ|, p99", "Tail/Exceedance", "tail", False),
]

DIST_KEYS = {m.key for m in METRICS if m.subgroup == "distributional"}
TAIL_KEYS = {m.key for m in METRICS if m.subgroup == "tail"}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def sample_id(row: dict[str, str]) -> int:
    for k in ("sample_idx", "sample_id", "sample", "id"):
        if k in row and str(row[k]).strip() != "":
            return int(float(str(row[k]).strip()))
    raise ValueError(f"No sample id column found in row keys: {list(row.keys())}")


def as_float(x) -> Optional[float]:
    if x is None:
        return None
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def norm_winner(x) -> str:
    if x is None:
        return ""
    s = str(x).strip().upper()
    if s in {"CNN", "GAN", "TIE"}:
        return s
    if s in {"TIED", "EQUAL"}:
        return "TIE"
    return s


def pick(row: dict[str, str], *cols: str) -> str:
    for c in cols:
        if c in row and str(row[c]).strip():
            return str(row[c]).strip()
    return ""


def boolish(x) -> bool:
    return str(x).strip().lower() in {"true", "1", "yes", "y"}


def load_optional_metadata() -> dict[int, dict]:
    meta: dict[int, dict] = defaultdict(dict)

    for path in (OBS_PER_SAMPLE, VIS_MANIFEST):
        for row in read_csv(path):
            try:
                sid = sample_id(row)
            except Exception:
                continue

            for col in [
                "psnr_winner", "ssim_winner", "pd_winner", "mt_winner",
                "direct_error_group_winner", "distributional_group_winner",
                "tail_group_winner", "configured_physics_group_winner",
                "recommendation_group", "reason", "question"
            ]:
                if col in row and str(row[col]).strip():
                    meta[sid][col] = row[col]

            groups = []
            for k, v in row.items():
                if k.startswith("group_") and boolish(v):
                    groups.append(k[len("group_"):])
            if groups:
                old = meta[sid].get("groups", "")
                merged = sorted(set([g for g in old.split(";") if g] + groups))
                meta[sid]["groups"] = ";".join(merged)

    return dict(meta)


def infer_mt_pd_winners(row: dict[str, str], meta: dict) -> tuple[str, str]:
    mt = norm_winner(pick(row, "mt_winner", "winner_mt", "merge_tree_winner"))
    pd = norm_winner(pick(row, "pd_winner", "winner_pd", "bottleneck_pd_winner"))
    if not mt:
        mt = norm_winner(meta.get("mt_winner", ""))
    if not pd:
        pd = norm_winner(meta.get("pd_winner", ""))

    if not mt:
        d = as_float(pick(row, "delta_mt_cnn_positive", "delta_mt"))
        if d is not None:
            mt = "CNN" if d > 0 else "GAN" if d < 0 else "TIE"
    if not pd:
        d = as_float(pick(row, "delta_pd_cnn_positive", "delta_pd"))
        if d is not None:
            pd = "CNN" if d > 0 else "GAN" if d < 0 else "TIE"

    return mt, pd


def metric_winner(row: dict[str, str], spec: MetricSpec) -> tuple[str, Optional[float], Optional[float]]:
    w = norm_winner(pick(row, f"{spec.key}_winner", f"{spec.key}_win"))
    cnn = as_float(pick(row, f"{spec.key}_cnn", f"cnn_{spec.key}"))
    gan = as_float(pick(row, f"{spec.key}_gan", f"gan_{spec.key}"))

    if w in {"CNN", "GAN", "TIE"}:
        return w, cnn, gan

    if cnn is None or gan is None:
        return "MISSING", cnn, gan

    cv = abs(cnn) if spec.signed_abs else cnn
    gv = abs(gan) if spec.signed_abs else gan
    if math.isclose(cv, gv, rel_tol=0.0, abs_tol=1e-12):
        return "TIE", cnn, gan
    return ("CNN" if cv < gv else "GAN"), cnn, gan


def classify_mt_gan_tier(gan_wins: int) -> str:
    if gan_wins >= 8:
        return "Strong"
    if gan_wins >= 6:
        return "Moderate"
    return "Lower"


def build_breakdown_rows(wide_rows: list[dict[str, str]], meta_by_sample: dict[int, dict]) -> list[dict]:
    out = []

    for row in wide_rows:
        sid = sample_id(row)
        meta = meta_by_sample.get(sid, {})
        mt_winner, pd_winner = infer_mt_pd_winners(row, meta)

        gan_measures = []
        cnn_measures = []
        tie_measures = []
        dist_gan = 0
        tail_gan = 0

        record: dict[str, object] = {
            "sample_idx": sid,
            "psnr_winner": norm_winner(pick(row, "psnr_winner")) or norm_winner(meta.get("psnr_winner", "")),
            "ssim_winner": norm_winner(pick(row, "ssim_winner")) or norm_winner(meta.get("ssim_winner", "")),
            "pd_winner": pd_winner,
            "mt_winner": mt_winner,
            "direct_error_group_winner": norm_winner(meta.get("direct_error_group_winner", "")),
            "distributional_group_winner": norm_winner(meta.get("distributional_group_winner", "")),
            "tail_group_winner": norm_winner(meta.get("tail_group_winner", "")),
            "configured_physics_group_winner": norm_winner(meta.get("configured_physics_group_winner", "")),
            "groups": meta.get("groups", ""),
            "recommendation_group": meta.get("recommendation_group", ""),
        }

        for spec in METRICS:
            w, cnn, gan = metric_winner(row, spec)

            record[f"{spec.key}_cnn"] = cnn if cnn is not None else ""
            record[f"{spec.key}_gan"] = gan if gan is not None else ""
            record[f"{spec.key}_winner"] = w

            if w == "GAN":
                gan_measures.append(spec.label)
                if spec.key in DIST_KEYS:
                    dist_gan += 1
                if spec.key in TAIL_KEYS:
                    tail_gan += 1
            elif w == "CNN":
                cnn_measures.append(spec.label)
            elif w == "TIE":
                tie_measures.append(spec.label)

        cnn_wins = len(cnn_measures)
        gan_wins = len(gan_measures)
        ties = len(tie_measures)

        record.update({
            "cnn_metric_wins": cnn_wins,
            "gan_metric_wins": gan_wins,
            "ties": ties,
            "overall_metric_majority": "GAN" if gan_wins > cnn_wins else "CNN" if cnn_wins > gan_wins else "TIE",
            "distributional_gan_wins_out_of_5": dist_gan,
            "tail_gan_wins_out_of_6": tail_gan,
            "gan_winning_measures": " · ".join(gan_measures),
            "cnn_winning_measures": " · ".join(cnn_measures),
            "tie_measures": " · ".join(tie_measures),
            "mt_gan_case": mt_winner == "GAN",
            "pd_cnn_case": pd_winner == "CNN",
            "gan_metric_majority": gan_wins > cnn_wins,
            "gan_metric_majority_but_mt_not_gan": (gan_wins > cnn_wins and mt_winner != "GAN"),
            "mt_gan_tier": classify_mt_gan_tier(gan_wins) if mt_winner == "GAN" else "",
        })

        out.append(record)

    return sorted(out, key=lambda r: int(r["sample_idx"]))


def fieldnames() -> list[str]:
    base = [
        "sample_idx",
        "psnr_winner", "ssim_winner", "pd_winner", "mt_winner",
        "direct_error_group_winner", "distributional_group_winner",
        "tail_group_winner", "configured_physics_group_winner",
        "cnn_metric_wins", "gan_metric_wins", "ties", "overall_metric_majority",
        "distributional_gan_wins_out_of_5", "tail_gan_wins_out_of_6",
        "mt_gan_case", "pd_cnn_case", "gan_metric_majority",
        "gan_metric_majority_but_mt_not_gan", "mt_gan_tier",
        "groups", "recommendation_group",
        "gan_winning_measures", "cnn_winning_measures", "tie_measures",
    ]

    metric_cols = []
    for spec in METRICS:
        metric_cols.extend([f"{spec.key}_cnn", f"{spec.key}_gan", f"{spec.key}_winner"])

    return base + metric_cols


def contiguous_runs(sample_ids: list[int], label: str) -> list[dict]:
    ids = sorted(set(sample_ids))
    if not ids:
        return []

    runs = []
    start = prev = ids[0]
    for x in ids[1:]:
        if x == prev + 1:
            prev = x
        else:
            runs.append({
                "label": label,
                "start_sample": start,
                "end_sample": prev,
                "length": prev - start + 1,
                "sample_ids": " ".join(map(str, range(start, prev + 1))),
            })
            start = prev = x
    runs.append({
        "label": label,
        "start_sample": start,
        "end_sample": prev,
        "length": prev - start + 1,
        "sample_ids": " ".join(map(str, range(start, prev + 1))),
    })
    return runs


def transition_pairs(rows: list[dict]) -> list[dict]:
    by_id = {int(r["sample_idx"]): r for r in rows}
    ids = sorted(by_id.keys())
    out = []

    for a, b in zip(ids, ids[1:]):
        if b != a + 1:
            continue

        ra, rb = by_id[a], by_id[b]
        if ra["mt_winner"] != rb["mt_winner"] or ra["overall_metric_majority"] != rb["overall_metric_majority"]:
            out.append({
                "sample_a": a,
                "sample_b": b,
                "mt_a": ra["mt_winner"],
                "mt_b": rb["mt_winner"],
                "pd_a": ra["pd_winner"],
                "pd_b": rb["pd_winner"],
                "gan_metric_wins_a": ra["gan_metric_wins"],
                "gan_metric_wins_b": rb["gan_metric_wins"],
                "overall_majority_a": ra["overall_metric_majority"],
                "overall_majority_b": rb["overall_metric_majority"],
                "dist_gan_a": ra["distributional_gan_wins_out_of_5"],
                "dist_gan_b": rb["distributional_gan_wins_out_of_5"],
                "tail_gan_a": ra["tail_gan_wins_out_of_6"],
                "tail_gan_b": rb["tail_gan_wins_out_of_6"],
                "note": "Adjacent timesteps with MT or physics-domain majority transition; inspect visually for threshold/bifurcation behavior.",
            })

    return out


def summary_rows(rows: list[dict]) -> list[dict]:
    mt_gan = [r for r in rows if r["mt_winner"] == "GAN"]
    pd_cnn = [r for r in rows if r["pd_winner"] == "CNN"]
    gan_majority = [r for r in rows if r["gan_metric_majority"]]
    gan_majority_mt_not = [r for r in rows if r["gan_metric_majority_but_mt_not_gan"]]
    c = Counter(r["mt_gan_tier"] for r in mt_gan)

    return [
        {"item": "total_samples", "count": len(rows), "sample_ids": ""},
        {"item": "mt_gan_cases", "count": len(mt_gan), "sample_ids": " ".join(str(r["sample_idx"]) for r in mt_gan)},
        {"item": "pd_cnn_cases", "count": len(pd_cnn), "sample_ids": " ".join(str(r["sample_idx"]) for r in pd_cnn)},
        {"item": "gan_metric_majority_cases", "count": len(gan_majority), "sample_ids": " ".join(str(r["sample_idx"]) for r in gan_majority)},
        {"item": "gan_metric_majority_but_mt_not_gan_cases", "count": len(gan_majority_mt_not), "sample_ids": " ".join(str(r["sample_idx"]) for r in gan_majority_mt_not)},
        {"item": "mt_gan_strong", "count": c.get("Strong", 0), "sample_ids": " ".join(str(r["sample_idx"]) for r in mt_gan if r["mt_gan_tier"] == "Strong")},
        {"item": "mt_gan_moderate", "count": c.get("Moderate", 0), "sample_ids": " ".join(str(r["sample_idx"]) for r in mt_gan if r["mt_gan_tier"] == "Moderate")},
        {"item": "mt_gan_lower", "count": c.get("Lower", 0), "sample_ids": " ".join(str(r["sample_idx"]) for r in mt_gan if r["mt_gan_tier"] == "Lower")},
    ]



# ============================================================
# Static HTML dashboard
# ============================================================

def _json_safe_rows(rows: list[dict]) -> list[dict]:
    safe = []
    for r in rows:
        item = {}
        for k, v in r.items():
            if isinstance(v, float):
                item[k] = None if math.isnan(v) or math.isinf(v) else v
            else:
                item[k] = v
        safe.append(item)
    return safe


def write_html_dashboard(rows: list[dict], transitions: list[dict]) -> None:
    """Write a self-contained dashboard for easier browser inspection."""
    metric_specs = [m.__dict__ for m in METRICS]
    rows_json = json.dumps(_json_safe_rows(rows), ensure_ascii=False, allow_nan=False)
    metrics_json = json.dumps(metric_specs, ensure_ascii=False, allow_nan=False)
    transitions_json = json.dumps(transitions, ensure_ascii=False, allow_nan=False)

    mt_gan = [r for r in rows if r["mt_winner"] == "GAN"]
    pd_cnn = [r for r in rows if r["pd_winner"] == "CNN"]
    gan_majority = [r for r in rows if r["gan_metric_majority"]]
    gan_majority_mt_not = [r for r in rows if r["gan_metric_majority_but_mt_not_gan"]]

    html_text = f"""<!doctype html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\">
<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
<title>TopoAware SR — Full Physics/Domain Breakdown</title>
<style>
  :root {{
    --bg: #f6f7fb;
    --card: #ffffff;
    --ink: #111827;
    --muted: #6b7280;
    --line: #e5e7eb;
    --blue: #2563eb;
    --green-bg: #dcfce7;
    --green-ink: #166534;
    --orange-bg: #ffedd5;
    --orange-ink: #9a3412;
    --purple-bg: #ede9fe;
    --purple-ink: #5b21b6;
    --red-bg: #fee2e2;
    --red-ink: #991b1b;
    --gray-bg: #f3f4f6;
  }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, \"Segoe UI\", Roboto, Helvetica, Arial, sans-serif; color: var(--ink); background: var(--bg); }}
  header {{ padding: 24px 28px 12px; background: linear-gradient(180deg, #ffffff 0%, #f7f8fc 100%); border-bottom: 1px solid var(--line); position: sticky; top: 0; z-index: 20; }}
  h1 {{ margin: 0 0 6px; font-size: 28px; }}
  h2 {{ margin-top: 0; }}
  .subtitle {{ color: var(--muted); font-size: 14px; line-height: 1.45; }}
  main {{ padding: 18px 28px 40px; }}
  .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin: 14px 0 18px; }}
  .card {{ background: var(--card); border: 1px solid var(--line); border-radius: 14px; padding: 14px; box-shadow: 0 1px 2px rgba(0,0,0,.04); }}
  .card .num {{ font-weight: 800; font-size: 26px; }}
  .card .label {{ color: var(--muted); font-size: 13px; margin-top: 4px; }}
  .panel {{ background: var(--card); border: 1px solid var(--line); border-radius: 14px; padding: 14px; margin: 14px 0; box-shadow: 0 1px 2px rgba(0,0,0,.04); }}
  .controls {{ display: grid; grid-template-columns: minmax(220px, 1fr) repeat(3, minmax(140px, max-content)); gap: 10px; align-items: center; }}
  input[type=\"search\"], select {{ width: 100%; padding: 9px 10px; border: 1px solid var(--line); border-radius: 10px; background: #fff; }}
  label.check {{ display: inline-flex; gap: 6px; align-items: center; white-space: nowrap; color: #374151; font-size: 14px; }}
  .links a {{ margin-right: 12px; color: var(--blue); text-decoration: none; font-weight: 600; }}
  .links a:hover {{ text-decoration: underline; }}
  table {{ width: 100%; border-collapse: collapse; }}
  th, td {{ padding: 8px 9px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; font-size: 13px; }}
  th {{ background: #f9fafb; position: sticky; top: 112px; z-index: 10; cursor: pointer; user-select: none; }}
  tr:hover td {{ background: #fcfcfd; }}
  .pill {{ display: inline-flex; align-items: center; padding: 2px 8px; border-radius: 999px; font-size: 12px; font-weight: 700; border: 1px solid transparent; white-space: nowrap; }}
  .pill.cnn {{ background: var(--green-bg); color: var(--green-ink); border-color: #bbf7d0; }}
  .pill.gan {{ background: var(--orange-bg); color: var(--orange-ink); border-color: #fed7aa; }}
  .pill.tie {{ background: var(--gray-bg); color: #374151; border-color: #d1d5db; }}
  .pill.flag {{ background: var(--purple-bg); color: var(--purple-ink); border-color: #ddd6fe; margin: 2px 3px 2px 0; }}
  .pill.warn {{ background: var(--red-bg); color: var(--red-ink); border-color: #fecaca; }}
  .small {{ color: var(--muted); font-size: 12px; }}
  .measure-list {{ max-width: 480px; line-height: 1.35; }}
  details summary {{ cursor: pointer; color: var(--blue); font-weight: 700; }}
  .metric-grid {{ margin-top: 8px; border: 1px solid var(--line); border-radius: 10px; overflow: hidden; }}
  .metric-grid th {{ position: static; cursor: default; }}
  .metric-grid td, .metric-grid th {{ font-size: 12px; padding: 6px 8px; }}
  .transition-list {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 8px; }}
  .transition-card {{ border: 1px solid var(--line); border-radius: 10px; padding: 10px; background: #fff; font-size: 13px; }}
  .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, \"Liberation Mono\", monospace; }}
  @media (max-width: 900px) {{ header {{ position: static; }} th {{ position: static; }} .controls {{ grid-template-columns: 1fr; }} main {{ padding: 14px; }} table {{ min-width: 1100px; }} .table-wrap {{ overflow-x: auto; }} }}
</style>
</head>
<body>
<header>
  <h1>TopoAware SR — Full Physics/Domain Breakdown</h1>
  <div class=\"subtitle\">Static dashboard generated from <span class=\"mono\">metric_sweep_all_samples_wide.csv</span>. Lower is better for all displayed components; signed quantities use absolute error relative to GT. Adjacent sample IDs may be adjacent hourly timesteps.</div>
</header>
<main>
  <section class=\"cards\">
    <div class=\"card\"><div class=\"num\">{len(rows)}</div><div class=\"label\">Total samples</div></div>
    <div class=\"card\"><div class=\"num\">{len(mt_gan)}</div><div class=\"label\">MT picks GAN</div></div>
    <div class=\"card\"><div class=\"num\">{len(pd_cnn)}</div><div class=\"label\">PD picks CNN</div></div>
    <div class=\"card\"><div class=\"num\">{len(gan_majority)}</div><div class=\"label\">GAN majority among 15 measures</div></div>
    <div class=\"card\"><div class=\"num\">{len(gan_majority_mt_not)}</div><div class=\"label\">GAN majority but MT not GAN</div></div>
    <div class=\"card\"><div class=\"num\">{len(transitions)}</div><div class=\"label\">Adjacent transition pairs</div></div>
  </section>
  <section class=\"panel links\"><strong>CSV outputs:</strong>
    <a href=\"physics_domain_breakdown_all_samples.csv\">all samples</a>
    <a href=\"physics_domain_breakdown_mt_gan.csv\">MT-GAN</a>
    <a href=\"physics_domain_breakdown_non_mt_gan.csv\">non-MT-GAN</a>
    <a href=\"physics_domain_breakdown_pd_cnn_cases.csv\">PD-CNN cases</a>
    <a href=\"physics_domain_breakdown_gan_majority_cases.csv\">GAN-majority</a>
    <a href=\"physics_domain_breakdown_gan_majority_mt_not_gan.csv\">GAN-majority / MT-not-GAN</a>
    <a href=\"adjacency_transition_pairs.csv\">adjacent transitions</a>
  </section>
  <section class=\"panel\"><div class=\"controls\">
      <input id=\"search\" type=\"search\" placeholder=\"Search sample id, measures, groups, tier...\">
      <select id=\"preset\"><option value=\"all\">All samples</option><option value=\"mt_gan\">MT picks GAN</option><option value=\"pd_cnn\">PD picks CNN</option><option value=\"gan_majority\">GAN metric majority</option><option value=\"gan_majority_mt_not\">GAN majority but MT not GAN</option><option value=\"strong_mt_gan\">Strong MT-GAN</option><option value=\"moderate_mt_gan\">Moderate MT-GAN</option><option value=\"lower_mt_gan\">Lower MT-GAN</option></select>
      <label class=\"check\"><input id=\"showOnlyInteresting\" type=\"checkbox\"> only diagnostic rows</label>
      <label class=\"check\"><input id=\"compact\" type=\"checkbox\" checked> compact measures</label>
    </div><p class=\"small\"><span id=\"visibleCount\"></span> visible. Click column headers to sort. Use the details link in each row for the 15-measure breakdown.</p></section>
  <section class=\"panel table-wrap\"><table id=\"mainTable\"><thead><tr>
    <th data-sort=\"sample_idx\">Sample</th><th data-sort=\"pd_winner\">PD</th><th data-sort=\"mt_winner\">MT</th><th data-sort=\"gan_metric_wins\">GAN wins</th><th data-sort=\"overall_metric_majority\">Overall</th><th data-sort=\"distributional_gan_wins_out_of_5\">Dist.</th><th data-sort=\"tail_gan_wins_out_of_6\">Tail</th><th data-sort=\"mt_gan_tier\">MT-GAN tier</th><th>Flags</th><th>GAN-winning measures</th><th>Details</th>
  </tr></thead><tbody></tbody></table></section>
  <section class=\"panel\"><h2>Adjacent transition pairs</h2><p class=\"small\">Adjacent pairs where MT winner or 15-measure majority changes.</p><div id=\"transitionList\" class=\"transition-list\"></div></section>
</main>
<script>
const ROWS = {rows_json};
const METRICS = {metrics_json};
const TRANSITIONS = {transitions_json};
let sortKey = 'sample_idx';
let sortDir = 1;
function esc(s) {{ return String(s ?? '').replace(/[&<>\"']/g, ch => ({{'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;',"'":'&#39;'}}[ch])); }}
function pill(w) {{ const v = String(w || '').toUpperCase(); const cls = v === 'CNN' ? 'cnn' : v === 'GAN' ? 'gan' : 'tie'; return `<span class=\"pill ${{cls}}\">${{esc(v || 'NA')}}</span>`; }}
function flag(text, warn=false) {{ return `<span class=\"pill ${{warn ? 'warn' : 'flag'}}\">${{esc(text)}}</span>`; }}
function fmt(x) {{ if (x === null || x === undefined || x === '') return ''; const n = Number(x); if (!Number.isFinite(n)) return esc(x); if (Math.abs(n) >= 100) return n.toFixed(3); if (Math.abs(n) >= 1) return n.toFixed(4); if (Math.abs(n) >= 0.001) return n.toFixed(5); return n.toExponential(3); }}
function flagsFor(r) {{ const out = []; if (r.mt_gan_case) out.push(flag('MT→GAN')); if (r.pd_cnn_case) out.push(flag('PD→CNN')); if (r.gan_metric_majority) out.push(flag('GAN metric majority')); if (r.gan_metric_majority_but_mt_not_gan) out.push(flag('GAN majority / MT≠GAN', true)); if (r.groups) {{ for (const g of String(r.groups).split(';').filter(Boolean).slice(0, 3)) out.push(flag(g)); }} return out.join(' '); }}
function metricDetails(r) {{ const rs = METRICS.map(m => `<tr><td>${{esc(m.label)}}</td><td>${{esc(m.group)}}</td><td class=\"mono\">${{fmt(r[m.key + '_cnn'])}}</td><td class=\"mono\">${{fmt(r[m.key + '_gan'])}}</td><td>${{pill(r[m.key + '_winner'])}}</td></tr>`).join(''); return `<details><summary>details</summary><table class=\"metric-grid\"><thead><tr><th>Measure</th><th>Group</th><th>CNN</th><th>GAN</th><th>Winner</th></tr></thead><tbody>${{rs}}</tbody></table></details>`; }}
function rowMatchesPreset(r, preset) {{ if (preset === 'all') return true; if (preset === 'mt_gan') return r.mt_winner === 'GAN'; if (preset === 'pd_cnn') return r.pd_winner === 'CNN'; if (preset === 'gan_majority') return !!r.gan_metric_majority; if (preset === 'gan_majority_mt_not') return !!r.gan_metric_majority_but_mt_not_gan; if (preset === 'strong_mt_gan') return r.mt_gan_tier === 'Strong'; if (preset === 'moderate_mt_gan') return r.mt_gan_tier === 'Moderate'; if (preset === 'lower_mt_gan') return r.mt_gan_tier === 'Lower'; return true; }}
function getFilteredRows() {{ const q = document.getElementById('search').value.trim().toLowerCase(); const preset = document.getElementById('preset').value; const onlyDiag = document.getElementById('showOnlyInteresting').checked; let rows = ROWS.filter(r => rowMatchesPreset(r, preset)); if (onlyDiag) rows = rows.filter(r => r.mt_gan_case || r.pd_cnn_case || r.gan_metric_majority || r.gan_metric_majority_but_mt_not_gan); if (q) rows = rows.filter(r => JSON.stringify(r).toLowerCase().includes(q)); rows.sort((a, b) => {{ const av = a[sortKey], bv = b[sortKey]; const an = Number(av), bn = Number(bv); if (Number.isFinite(an) && Number.isFinite(bn)) return sortDir * (an - bn); return sortDir * String(av ?? '').localeCompare(String(bv ?? '')); }}); return rows; }}
function render() {{ const rows = getFilteredRows(); const compact = document.getElementById('compact').checked; document.getElementById('visibleCount').textContent = `${{rows.length}} / ${{ROWS.length}} rows`; const tbody = document.querySelector('#mainTable tbody'); tbody.innerHTML = rows.map(r => {{ const measures = compact && String(r.gan_winning_measures || '').length > 120 ? esc(String(r.gan_winning_measures).slice(0, 120)) + '…' : esc(r.gan_winning_measures || '—'); return `<tr><td class=\"mono\"><strong>${{esc(r.sample_idx)}}</strong></td><td>${{pill(r.pd_winner)}}</td><td>${{pill(r.mt_winner)}}</td><td><strong>${{esc(r.gan_metric_wins)}}</strong>/15 <span class=\"small\">CNN ${{esc(r.cnn_metric_wins)}}</span></td><td>${{pill(r.overall_metric_majority)}}</td><td>${{esc(r.distributional_gan_wins_out_of_5)}}/5</td><td>${{esc(r.tail_gan_wins_out_of_6)}}/6</td><td>${{esc(r.mt_gan_tier || '—')}}</td><td>${{flagsFor(r)}}</td><td class=\"measure-list\">${{measures}}</td><td>${{metricDetails(r)}}</td></tr>`; }}).join(''); }}
function renderTransitions() {{ const box = document.getElementById('transitionList'); box.innerHTML = TRANSITIONS.map(t => `<div class=\"transition-card\"><strong class=\"mono\">${{esc(t.sample_a)}} → ${{esc(t.sample_b)}}</strong><br>MT: ${{pill(t.mt_a)}} → ${{pill(t.mt_b)}} &nbsp; PD: ${{pill(t.pd_a)}} → ${{pill(t.pd_b)}}<br>GAN wins: <span class=\"mono\">${{esc(t.gan_metric_wins_a)}} → ${{esc(t.gan_metric_wins_b)}}</span>; overall: ${{pill(t.overall_majority_a)}} → ${{pill(t.overall_majority_b)}}<br><span class=\"small\">Dist: ${{esc(t.dist_gan_a)}}→${{esc(t.dist_gan_b)}}; Tail: ${{esc(t.tail_gan_a)}}→${{esc(t.tail_gan_b)}}</span></div>`).join('') || '<p class=\"small\">No adjacent transition pairs found.</p>'; }}
document.querySelectorAll('th[data-sort]').forEach(th => {{ th.addEventListener('click', () => {{ const k = th.dataset.sort; if (sortKey === k) sortDir *= -1; else {{ sortKey = k; sortDir = 1; }} render(); }}); }});
for (const id of ['search', 'preset', 'showOnlyInteresting', 'compact']) {{ document.getElementById(id).addEventListener('input', render); document.getElementById(id).addEventListener('change', render); }}
renderTransitions(); render();
</script>
</body>
</html>
"""
    (OUTDIR / "physics_domain_breakdown_index.html").write_text(html_text, encoding="utf-8")

def write_readme(rows: list[dict]) -> None:
    mt_gan = [r for r in rows if r["mt_winner"] == "GAN"]
    pd_cnn = [r for r in rows if r["pd_winner"] == "CNN"]
    gan_majority = [r for r in rows if r["gan_metric_majority"]]
    gan_majority_mt_not = [r for r in rows if r["gan_metric_majority_but_mt_not_gan"]]

    readme = f"""# Full physics/domain breakdown tables

Generated by `scripts/generate_full_physics_domain_breakdown.py`.

## Inputs

- `{INPUT_WIDE.relative_to(ROOT)}`

## Main outputs

- `physics_domain_breakdown_all_samples.csv`: all 168 samples.
- `physics_domain_breakdown_non_mt_gan.csv`: all samples where MT does not pick GAN.
- `physics_domain_breakdown_mt_gan.csv`: all samples where MT picks GAN.
- `physics_domain_breakdown_pd_cnn_cases.csv`: rare cases where PD picks CNN.
- `physics_domain_breakdown_gan_majority_cases.csv`: samples where GAN wins a majority of the 15 physics/domain measures.
- `physics_domain_breakdown_gan_majority_mt_not_gan.csv`: especially important counterexamples where GAN wins the metric majority but MT does not pick GAN.
- `adjacency_runs_*.csv`: contiguous sample-index runs.
- `adjacency_transition_pairs.csv`: adjacent timesteps where MT winner or physics/domain majority changes.
- `physics_domain_breakdown_index.html`: static browser dashboard with filters, summary cards, metric details, and adjacency-transition cards.

## Counts

- Total samples: {len(rows)}
- MT picks GAN: {len(mt_gan)}
- PD picks CNN: {len(pd_cnn)}
- GAN majority among 15 physics/domain measures: {len(gan_majority)}
- GAN majority but MT does not pick GAN: {len(gan_majority_mt_not)}

## Interpretation notes

1. Conservative fidelity measures usually favor CNN: WPD MAE, WPD RMSE, Gradient MAE, PSNR, SSIM.
2. Distributional / multiscale measures often favor GAN: PSD log-L2, Gradient Wasserstein-1, some PSD slope and extreme-tail measures.
3. PD often favors GAN because it is sensitive to added topological feature richness.
4. MT is more selective and should be interpreted as a structural diagnostic rather than a standalone ground-truth judge.
5. Adjacent sample IDs should be treated carefully. Because the evaluation set uses consecutive hourly samples, adjacent IDs may represent similar or slowly evolving meteorological regimes rather than independent cases.
"""
    (OUTDIR / "README_full_physics_domain_breakdown.md").write_text(readme, encoding="utf-8")


def main() -> None:
    if not INPUT_WIDE.exists():
        raise FileNotFoundError(f"Missing required wide table: {INPUT_WIDE}")

    OUTDIR.mkdir(parents=True, exist_ok=True)

    wide_rows = read_csv(INPUT_WIDE)
    meta = load_optional_metadata()
    rows = build_breakdown_rows(wide_rows, meta)

    fns = fieldnames()

    mt_gan = [r for r in rows if r["mt_winner"] == "GAN"]
    non_mt_gan = [r for r in rows if r["mt_winner"] != "GAN"]
    pd_cnn = [r for r in rows if r["pd_winner"] == "CNN"]
    gan_majority = [r for r in rows if r["gan_metric_majority"]]
    gan_majority_mt_not = [r for r in rows if r["gan_metric_majority_but_mt_not_gan"]]

    by_gan_desc = lambda r: (-int(r["gan_metric_wins"]), int(r["sample_idx"]))

    write_csv(OUTDIR / "physics_domain_breakdown_all_samples.csv", rows, fns)
    write_csv(OUTDIR / "physics_domain_breakdown_non_mt_gan.csv", sorted(non_mt_gan, key=by_gan_desc), fns)
    write_csv(OUTDIR / "physics_domain_breakdown_mt_gan.csv", sorted(mt_gan, key=by_gan_desc), fns)
    write_csv(OUTDIR / "physics_domain_breakdown_pd_cnn_cases.csv", sorted(pd_cnn, key=lambda r: int(r["sample_idx"])), fns)
    write_csv(OUTDIR / "physics_domain_breakdown_gan_majority_cases.csv", sorted(gan_majority, key=by_gan_desc), fns)
    write_csv(OUTDIR / "physics_domain_breakdown_gan_majority_mt_not_gan.csv", sorted(gan_majority_mt_not, key=by_gan_desc), fns)
    write_csv(
        OUTDIR / "physics_domain_breakdown_mt_gan_strong_moderate_lower.csv",
        sorted(mt_gan, key=lambda r: ({"Strong": 0, "Moderate": 1, "Lower": 2}.get(r["mt_gan_tier"], 9), -int(r["gan_metric_wins"]), int(r["sample_idx"]))),
        fns,
    )

    run_fields = ["label", "start_sample", "end_sample", "length", "sample_ids"]
    write_csv(OUTDIR / "adjacency_runs_all_samples.csv", contiguous_runs([int(r["sample_idx"]) for r in rows], "all_samples"), run_fields)
    write_csv(OUTDIR / "adjacency_runs_mt_gan.csv", contiguous_runs([int(r["sample_idx"]) for r in mt_gan], "mt_gan"), run_fields)
    write_csv(OUTDIR / "adjacency_runs_gan_majority.csv", contiguous_runs([int(r["sample_idx"]) for r in gan_majority], "gan_metric_majority"), run_fields)

    trans = transition_pairs(rows)
    trans_fields = [
        "sample_a", "sample_b", "mt_a", "mt_b", "pd_a", "pd_b",
        "gan_metric_wins_a", "gan_metric_wins_b",
        "overall_majority_a", "overall_majority_b",
        "dist_gan_a", "dist_gan_b", "tail_gan_a", "tail_gan_b", "note"
    ]
    write_csv(OUTDIR / "adjacency_transition_pairs.csv", trans, trans_fields)

    write_csv(OUTDIR / "summary_counts.csv", summary_rows(rows), ["item", "count", "sample_ids"])
    write_html_dashboard(rows, trans)
    write_readme(rows)

    print(f"repo_root={ROOT}")
    print(f"input={INPUT_WIDE}")
    print(f"outdir={OUTDIR}")
    print(f"html={OUTDIR / 'physics_domain_breakdown_index.html'}")
    print()
    print("Summary:")
    print(f"  total samples: {len(rows)}")
    print(f"  MT picks GAN: {len(mt_gan)}")
    print(f"  PD picks CNN: {len(pd_cnn)} -> {' '.join(str(r['sample_idx']) for r in pd_cnn)}")
    print(f"  GAN majority among 15 physics/domain measures: {len(gan_majority)}")
    print(f"  GAN majority but MT does not pick GAN: {len(gan_majority_mt_not)}")
    print(f"  adjacent transition pairs: {len(trans)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
