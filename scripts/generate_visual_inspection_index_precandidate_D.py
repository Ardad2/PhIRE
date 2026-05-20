#!/usr/bin/env python3
"""
Build a readable card-based TopoAware SR visual inspection index with
per-sample physics/domain metric breakdowns.

Run from:
    cd ~/PhIRE/scripts
    PYTHONNOUSERSITE=1 /usr/bin/python3 generate_visual_inspection_index.py

Inputs preferred:
    ~/PhIRE/ttk_runs_fixed/visual_inspection/visual_inspection_manifest.csv
    ~/PhIRE/ttk_runs_fixed/combined/psnr_topology_physics_merged.csv

Output:
    ~/PhIRE/ttk_runs_fixed/visual_inspection/index.html

This script does not regenerate PNG panels. It only rebuilds index.html.
"""

from __future__ import annotations

import csv
import html
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Optional


def repo_root() -> Path:
    here = Path(__file__).resolve().parent
    cwd = Path.cwd().resolve()
    candidates = [here.parent, cwd.parent if cwd.name == "scripts" else cwd, here, cwd]
    for p in candidates:
        if (p / "ttk_runs_fixed").exists():
            return p
    raise RuntimeError("Could not find repo root containing ttk_runs_fixed/")


def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def first(row: dict, *keys: str) -> str:
    for k in keys:
        v = str(row.get(k, "")).strip()
        if v:
            return v
    return ""


def sample_id(row: dict) -> int:
    raw = first(row, "sample_idx", "sample_id", "sample", "Sample", "id")
    if not raw:
        raise ValueError(f"Could not find sample id. Row keys: {list(row.keys())}")
    m = re.search(r"\d+", raw)
    if not m:
        raise ValueError(f"Could not parse sample id from {raw}")
    return int(m.group(0))


def split_groups(text: str) -> list[str]:
    if not text:
        return []
    return [x.strip() for x in re.split(r"[;,|]", text) if x.strip()]


def boolish(x: str) -> bool:
    return str(x).strip().lower() in {"true", "1", "yes", "y"}


def groups_for(row: dict) -> list[str]:
    groups = split_groups(first(row, "groups", "group_membership", "membership", "group_list"))
    rec = first(row, "recommendation_group")
    if rec:
        groups.append(rec)
    for k, v in row.items():
        if k.startswith("group_") and boolish(v):
            groups.append(k[len("group_"):])
    out, seen = [], set()
    for g in groups:
        if g and g not in seen:
            seen.add(g)
            out.append(g)
    return out


def pretty(s: str) -> str:
    return s.replace("_", " ")


def to_float(x: object) -> Optional[float]:
    try:
        if x is None:
            return None
        s = str(x).strip()
        if s == "" or s.lower() in {"nan", "none", "null"}:
            return None
        return float(s)
    except Exception:
        return None


def fmt_num(x: Optional[float]) -> str:
    if x is None:
        return "—"
    ax = abs(x)
    if ax == 0:
        return "0"
    if ax < 1e-3 or ax >= 1e4:
        return f"{x:.3e}"
    if ax < 1:
        return f"{x:.4f}"
    return f"{x:.3f}"


def resolve_path(raw: str, vis_dir: Path, root: Path) -> Path | None:
    if not raw:
        return None
    p = Path(raw)
    candidates = [p] if p.is_absolute() else [vis_dir / raw, root / raw, root / "ttk_runs_fixed" / "visual_inspection" / raw]
    for c in candidates:
        if c.exists():
            return c
    return None


def fallback_asset(vis_dir: Path, sid: int, kind: str) -> Path | None:
    if kind == "crop":
        candidates = [vis_dir / "panels_crop160" / f"sample_{sid:03d}_speed_error_crop.png"]
    elif kind == "full":
        candidates = [vis_dir / "panels_full" / f"sample_{sid:03d}_speed_error_full.png"]
    else:
        candidates = [
            vis_dir / "panels_crop160" / f"sample_{sid:03d}_speed_error_crop.png",
            vis_dir / "panels_full" / f"sample_{sid:03d}_speed_error_full.png",
        ]
    for c in candidates:
        if c.exists():
            return c
    return None


def rel(p: Path | None, start: Path) -> str:
    if p is None or not p.exists():
        return ""
    return os.path.relpath(p, start)


# label, metric stem, group, compare absolute value?
METRIC_SPECS = [
    ("WPD bias |·|", "wpd_bias", "Physics / WPD", True),
    ("WPD MAE", "wpd_mae", "Physics / WPD", False),
    ("WPD RMSE", "wpd_rmse", "Physics / WPD", False),
    ("WPD Wasserstein-1", "wpd_w1", "Distributional", False),
    ("PSD log-L2", "psd_log_l2", "Distributional", False),
    ("PSD slope |Δ|", "psd_slope_abs_delta", "Distributional", False),
    ("PSD slope |Δ|", "psd_slope_delta", "Distributional", True),
    ("Gradient MAE", "grad_mae", "Physics / Gradient", False),
    ("Gradient Wasserstein-1", "grad_w1", "Distributional", False),
    ("Gradient kurtosis |Δ|", "grad_kurtosis_abs_delta", "Distributional", False),
    ("Gradient kurtosis |Δ|", "grad_kurtosis_delta", "Distributional", True),
    ("Exceedance |Δ|, s > 5", "exceed_frac_abs_delta_t5", "Tail / Exceedance", False),
    ("Exceedance |Δ|, s > 5", "exceed_frac_delta_t5", "Tail / Exceedance", True),
    ("Exceedance |Δ|, s > 10", "exceed_frac_abs_delta_t10", "Tail / Exceedance", False),
    ("Exceedance |Δ|, s > 10", "exceed_frac_delta_t10", "Tail / Exceedance", True),
    ("Exceedance |Δ|, s > 15", "exceed_frac_abs_delta_t15", "Tail / Exceedance", False),
    ("Exceedance |Δ|, s > 15", "exceed_frac_delta_t15", "Tail / Exceedance", True),
    ("Exceedance |Δ|, p90", "exceed_frac_abs_delta_p90", "Tail / Exceedance", False),
    ("Exceedance |Δ|, p90", "exceed_frac_delta_p90", "Tail / Exceedance", True),
    ("Exceedance |Δ|, p95", "exceed_frac_abs_delta_p95", "Tail / Exceedance", False),
    ("Exceedance |Δ|, p95", "exceed_frac_delta_p95", "Tail / Exceedance", True),
    ("Exceedance |Δ|, p99", "exceed_frac_abs_delta_p99", "Tail / Exceedance", False),
    ("Exceedance |Δ|, p99", "exceed_frac_delta_p99", "Tail / Exceedance", True),
]
DISPLAY_ORDER = [
    "WPD bias |·|", "WPD MAE", "WPD RMSE", "WPD Wasserstein-1", "PSD log-L2", "PSD slope |Δ|",
    "Gradient MAE", "Gradient Wasserstein-1", "Gradient kurtosis |Δ|",
    "Exceedance |Δ|, s > 5", "Exceedance |Δ|, s > 10", "Exceedance |Δ|, s > 15",
    "Exceedance |Δ|, p90", "Exceedance |Δ|, p95", "Exceedance |Δ|, p99",
]


def find_model_col(row: dict, metric: str, model: str) -> Optional[str]:
    model_l, model_u = model.lower(), model.upper()
    candidates = [
        f"{model_l}_{metric}", f"{model_u}_{metric}", f"{metric}_{model_l}", f"{metric}_{model_u}",
        f"{model_l}.{metric}", f"{metric}.{model_l}", f"{model_l}-{metric}", f"{metric}-{model_l}",
    ]
    for c in candidates:
        if c in row:
            return c
    metric_parts = metric.lower().split("_")
    for c in row.keys():
        cl = c.lower()
        if model_l in cl and all(part in cl for part in metric_parts):
            return c
    return None


def metric_winner(cnn_val: Optional[float], gan_val: Optional[float], use_abs: bool) -> str:
    if cnn_val is None or gan_val is None:
        return "missing"
    cv = abs(cnn_val) if use_abs else cnn_val
    gv = abs(gan_val) if use_abs else gan_val
    if abs(cv - gv) <= 1e-12:
        return "tie"
    return "CNN" if cv < gv else "GAN"


def metric_row_html(label: str, group: str, cnn_val: Optional[float], gan_val: Optional[float], winner: str, use_abs: bool) -> str:
    cv = abs(cnn_val) if (use_abs and cnn_val is not None) else cnn_val
    gv = abs(gan_val) if (use_abs and gan_val is not None) else gan_val
    badge_cls = {"CNN": "cnn", "GAN": "gan", "tie": "tie"}.get(winner, "missing")
    return f"""
      <tr>
        <td>{html.escape(label)}</td>
        <td class="metric-group">{html.escape(group)}</td>
        <td class="num">{html.escape(fmt_num(cv))}</td>
        <td class="num">{html.escape(fmt_num(gv))}</td>
        <td><span class="winner-badge {badge_cls}">{html.escape(winner)}</span></td>
      </tr>
    """


def build_metric_breakdown(metric_row: Optional[dict]) -> tuple[str, str]:
    if metric_row is None:
        return ('<details class="metric-details"><summary>Physics/domain metric breakdown unavailable</summary>'
                '<div class="muted">No row found for this sample in psnr_topology_physics_merged.csv.</div></details>', "")
    raw_rows = []
    for desired_label in DISPLAY_ORDER:
        candidates = [x for x in METRIC_SPECS if x[0] == desired_label]
        chosen = None
        for label, metric, group, use_abs in candidates:
            cnn_col = find_model_col(metric_row, metric, "cnn")
            gan_col = find_model_col(metric_row, metric, "gan")
            if cnn_col and gan_col:
                chosen = (label, metric, group, use_abs, cnn_col, gan_col)
                break
        if not chosen:
            continue
        label, metric, group, use_abs, cnn_col, gan_col = chosen
        cnn_val = to_float(metric_row.get(cnn_col))
        gan_val = to_float(metric_row.get(gan_col))
        winner = metric_winner(cnn_val, gan_val, use_abs)
        raw_rows.append((label, group, cnn_val, gan_val, winner, use_abs))
    if not raw_rows:
        return ('<details class="metric-details"><summary>Physics/domain metric breakdown unavailable</summary>'
                '<div class="muted">Could not find paired CNN/GAN metric columns in the merged CSV.</div></details>', "")
    counts = Counter([r[4] for r in raw_rows])
    summary = f"CNN {counts.get('CNN', 0)} | GAN {counts.get('GAN', 0)} | ties {counts.get('tie', 0)}"
    rows_html = "\n".join(metric_row_html(*r) for r in raw_rows)
    search_text = " ".join([f"{label} {winner}" for label, _, _, _, winner, _ in raw_rows])
    html_block = f"""
    <details class="metric-details">
      <summary>Physics/domain metric breakdown <span class="mini-counts">{html.escape(summary)}</span></summary>
      <div class="metric-note">
        Lower is better. For signed quantities such as WPD bias, PSD slope delta,
        gradient-kurtosis delta, and exceedance deltas, the displayed comparison
        uses absolute error relative to GT.
      </div>
      <div class="metric-table-wrap">
        <table class="metric-table">
          <thead><tr><th>Measure</th><th>Group</th><th>CNN</th><th>GAN</th><th>Winner</th></tr></thead>
          <tbody>{rows_html}</tbody>
        </table>
      </div>
    </details>
    """
    return html_block, search_text


def question_for(row: dict) -> str:
    q = first(row, "question", "suggested_question", "inspection_question", "reason")
    if q:
        return q
    ssim, pd, mt = first(row, "ssim_winner"), first(row, "pd_winner"), first(row, "mt_winner")
    if ssim or pd or mt:
        return f"Inspect structure. Winners: SSIM={ssim or '?'}, PD={pd or '?'}, MT={mt or '?'} ."
    return "Inspect GT, CNN, GAN, and error maps."


def winners_for(row: dict) -> str:
    pairs = [
        ("psnr_winner", "PSNR"), ("ssim_winner", "SSIM"), ("pd_winner", "PD"), ("mt_winner", "MT"),
        ("direct_error_group_winner", "Direct"), ("distributional_group_winner", "Distributional"),
        ("tail_group_winner", "Tail"), ("configured_physics_group_winner", "Physics"),
    ]
    vals = []
    for k, label in pairs:
        v = first(row, k)
        if v:
            vals.append(f"{label}: {v}")
    return " | ".join(vals)


def load_metric_rows(root: Path) -> dict[int, dict]:
    candidates = [
        # Prefer the report-table wide CSV because it has columns like:
        # wpd_mae_cnn, wpd_mae_gan, wpd_mae_winner
        root / "ttk_runs_fixed" / "report_tables" / "metric_sweep_all_samples_wide.csv",

        # Fallbacks. These are long-format or subset files, so they are less ideal.
        root / "ttk_runs_fixed" / "selector_ablation_full" / "selector_ablation_threshold_0p075.csv",
        root / "ttk_runs_fixed" / "selector_ablation_full" / "selector_ablation_threshold_0p05.csv",
        root / "ttk_runs_fixed" / "near_tie_study" / "ssim" / "near_tie_ssim_thr_0.075.csv",
        root / "ttk_runs_fixed" / "near_tie_study" / "ssim" / "near_tie_ssim_thr_0.050.csv",
        root / "ttk_runs_fixed" / "combined" / "psnr_topology_physics_merged.csv",
        root / "ttk_runs_fixed" / "combined" / "ssim_topology_physics_merged.csv",
    ]

    for p in candidates:
        if not p.exists():
            continue

        rows = read_rows(p)
        if not rows:
            continue

        # Only accept files that actually contain paired CNN/GAN columns.
        cols = set(rows[0].keys())
        has_paired_cols = any(c.endswith("_cnn") for c in cols) and any(c.endswith("_gan") for c in cols)

        if not has_paired_cols:
            print(f"skipping non-wide metric CSV={p}")
            continue

        out = {}
        for r in rows:
            try:
                out[sample_id(r)] = r
            except Exception:
                continue

        print(f"metric_csv={p}")
        print(f"metric_rows={len(out)}")
        return out

    print("WARNING: no wide paired metric CSV found; metric breakdowns will be unavailable.")
    return {}


def build_cards(rows: list[dict], metric_by_sample: dict[int, dict], vis_dir: Path, root: Path) -> str:
    cards = []
    for row in rows:
        sid = sample_id(row)
        groups = groups_for(row)
        question = question_for(row)
        winners = winners_for(row)
        metric_html, metric_search = build_metric_breakdown(metric_by_sample.get(sid))
        crop = resolve_path(first(row, "crop_panel_path", "crop_panel", "crop"), vis_dir, root) or fallback_asset(vis_dir, sid, "crop")
        full = resolve_path(first(row, "full_panel_path", "full_panel", "panel_path", "panel"), vis_dir, root) or fallback_asset(vis_dir, sid, "full")
        preview = crop or full or fallback_asset(vis_dir, sid, "preview")
        crop_rel, full_rel, preview_rel = rel(crop, vis_dir), rel(full, vis_dir), rel(preview, vis_dir)
        target_rel = full_rel or crop_rel or preview_rel
        tag_html = "\n".join(f'<span class="tag">{html.escape(pretty(g))}</span>' for g in groups) or '<span class="muted">No groups listed</span>'
        links = []
        if crop_rel:
            links.append(f'<a href="{html.escape(crop_rel)}" target="_blank">Open crop panel</a>')
        if full_rel:
            links.append(f'<a href="{html.escape(full_rel)}" target="_blank">Open full panel</a>')
        links_html = " ".join(links) if links else '<span class="muted">No linked panel found</span>'
        preview_html = (f'<a href="{html.escape(target_rel)}" target="_blank"><img src="{html.escape(preview_rel)}" alt="Sample {sid} preview"></a>'
                        if preview_rel else '<div class="no-preview">No preview found</div>')
        search = " ".join([str(sid), " ".join(groups), question, winners, metric_search]).lower()
        cards.append(f"""
<article class="sample-card" data-search="{html.escape(search)}" data-groups="{' '.join(html.escape(g) for g in groups)}">
  <div class="sample-meta">
    <h2>Sample {sid}</h2>
    {'<div class="winner-strip">' + html.escape(winners) + '</div>' if winners else ''}
    <div class="meta-block"><div class="meta-label">Question</div><div class="meta-value">{html.escape(question)}</div></div>
    <div class="meta-block"><div class="meta-label">Groups</div><div class="tag-list">{tag_html}</div></div>
    {metric_html}
    <div class="meta-block links">{links_html}</div>
  </div>
  <div class="sample-preview">{preview_html}</div>
</article>
""")
    return "\n".join(cards)


def build_html(rows: list[dict], metric_by_sample: dict[int, dict], vis_dir: Path, root: Path, src_csv: Path) -> str:
    all_groups = []
    for r in rows:
        all_groups.extend(groups_for(r))
    counts = Counter(all_groups)
    buttons = "\n".join(f'<button class="group-filter" data-group="{html.escape(g)}" type="button">{html.escape(pretty(g))} <span>{n}</span></button>'
                         for g, n in sorted(counts.items(), key=lambda x: (-x[1], x[0])))
    cards = build_cards(rows, metric_by_sample, vis_dir, root)
    return f'''<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>TopoAware SR visual inspection index</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; padding: 24px; font-family: Arial, Helvetica, sans-serif; background: #f7f7f9; color: #111; line-height: 1.45; }}
  .page {{ max-width: 1550px; margin: 0 auto; }}
  h1 {{ margin: 0 0 8px 0; font-size: 2.2rem; }}
  .subtitle {{ margin-bottom: 6px; font-size: 1.05rem; }}
  .helper {{ color: #666; margin-bottom: 20px; font-size: 0.95rem; }}
  .controls {{ position: sticky; top: 10px; z-index: 10; background: white; border: 1px solid #ddd; border-radius: 10px; padding: 16px; margin-bottom: 22px; box-shadow: 0 1px 4px rgba(0,0,0,0.05); }}
  .controls-top {{ display: flex; gap: 12px; flex-wrap: wrap; align-items: center; margin-bottom: 14px; }}
  #searchBox {{ flex: 1 1 360px; min-width: 260px; padding: 12px 14px; border: 1px solid #ccc; border-radius: 8px; font-size: 15px; }}
  .clear-btn {{ border: 1px solid #ccc; background: #fff; padding: 10px 14px; border-radius: 8px; cursor: pointer; font-weight: 600; }}
  .stat-pill {{ background: #f1f1f1; border: 1px solid #ddd; border-radius: 999px; padding: 7px 12px; font-size: 14px; }}
  .filter-title {{ font-weight: 700; margin-bottom: 8px; }}
  .group-filters {{ display: flex; flex-wrap: wrap; gap: 8px; }}
  .group-filter {{ background: #fff; border: 1px solid #ccc; border-radius: 999px; padding: 8px 12px; cursor: pointer; font-size: 13px; }}
  .group-filter.active {{ background: #eef3ff; border-color: #c8d7ff; font-weight: 700; }}
  .index-container {{ display: flex; flex-direction: column; gap: 18px; }}
  .sample-card {{ display: grid; grid-template-columns: minmax(0, 1.45fr) minmax(420px, 1fr); gap: 20px; background: white; border: 1px solid #ddd; border-radius: 10px; padding: 18px; box-shadow: 0 1px 4px rgba(0,0,0,0.05); align-items: start; }}
  .sample-card h2 {{ margin: 0 0 8px 0; font-size: 1.55rem; }}
  .winner-strip {{ color: #333; background: #f5f5f5; border: 1px solid #e1e1e1; border-radius: 8px; padding: 8px 10px; font-size: 13px; margin-bottom: 12px; word-break: break-word; }}
  .meta-block {{ margin-bottom: 14px; }}
  .meta-label {{ font-weight: 700; margin-bottom: 6px; }}
  .tag-list {{ display: flex; flex-wrap: wrap; gap: 8px; }}
  .tag {{ background: #eef3ff; border: 1px solid #c8d7ff; border-radius: 999px; padding: 5px 10px; font-size: 13px; }}
  .links {{ display: flex; flex-wrap: wrap; gap: 14px; margin-top: 14px; }}
  .links a {{ color: #0056b3; font-weight: 600; text-decoration: none; }}
  .links a:hover {{ text-decoration: underline; }}
  .sample-preview {{ display: flex; justify-content: flex-end; }}
  .sample-preview img {{ width: 100%; max-width: 680px; border: 1px solid #ccc; border-radius: 6px; background: white; }}
  .no-preview {{ width: 100%; min-height: 220px; border: 1px dashed #bbb; border-radius: 6px; display: flex; align-items: center; justify-content: center; color: #666; background: #fafafa; }}
  .muted {{ color: #666; }}
  .metric-details {{ margin: 12px 0 14px 0; border: 1px solid #ddd; border-radius: 8px; background: #fcfcfc; padding: 8px 10px; }}
  .metric-details summary {{ cursor: pointer; font-weight: 700; }}
  .mini-counts {{ color: #444; font-weight: 500; margin-left: 8px; font-size: 13px; }}
  .metric-note {{ color: #555; font-size: 12.5px; margin: 8px 0; }}
  .metric-table-wrap {{ overflow-x: auto; }}
  .metric-table {{ border-collapse: collapse; width: 100%; font-size: 12.5px; }}
  .metric-table th, .metric-table td {{ border-bottom: 1px solid #e2e2e2; padding: 5px 6px; text-align: left; vertical-align: top; }}
  .metric-table th {{ background: #f2f2f2; position: sticky; top: 0; }}
  .metric-table .num {{ text-align: right; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace; }}
  .metric-group {{ color: #555; }}
  .winner-badge {{ display: inline-block; min-width: 44px; text-align: center; border-radius: 999px; padding: 2px 8px; font-weight: 700; font-size: 12px; border: 1px solid #ddd; }}
  .winner-badge.cnn {{ background: #e9f7ef; border-color: #b9e4c9; color: #146c2e; }}
  .winner-badge.gan {{ background: #fff2df; border-color: #ffd49a; color: #8a4b00; }}
  .winner-badge.tie {{ background: #f0f0ff; border-color: #ccccff; color: #333366; }}
  .winner-badge.missing {{ background: #f4f4f4; border-color: #ddd; color: #666; }}
  @media (max-width: 1150px) {{ .sample-card {{ grid-template-columns: 1fr; }} .sample-preview {{ justify-content: flex-start; }} .sample-preview img {{ max-width: 100%; }} }}
</style>
</head>
<body>
<div class="page">
  <h1>TopoAware SR visual inspection index</h1>
  <div class="subtitle">Each panel shows: GT speed | CNN speed | GAN speed | CNN-GT | GAN-GT.</div>
  <div class="helper">Source CSV: <code>{html.escape(str(src_csv))}</code>. Use search and group filters to narrow samples. Open each card's metric breakdown for per-sample physics/domain winners.</div>
  <section class="controls">
    <div class="controls-top">
      <input id="searchBox" type="text" placeholder="Search sample ID, group, question, winner, or metric...">
      <button class="clear-btn" id="clearBtn" type="button">Clear filters</button>
      <span class="stat-pill">Total: <strong>{len(rows)}</strong></span>
      <span class="stat-pill">Visible: <strong id="visibleCount">{len(rows)}</strong></span>
    </div>
    <div class="filter-title">Filter by group</div><div class="group-filters">{buttons}</div>
  </section>
  <section class="index-container" id="cards">{cards}</section>
</div>
<script>
const searchBox = document.getElementById("searchBox");
const clearBtn = document.getElementById("clearBtn");
const cards = Array.from(document.querySelectorAll(".sample-card"));
const buttons = Array.from(document.querySelectorAll(".group-filter"));
const visibleCount = document.getElementById("visibleCount");
function activeGroups() {{ return buttons.filter(b => b.classList.contains("active")).map(b => b.dataset.group); }}
function applyFilters() {{
  const q = searchBox.value.trim().toLowerCase();
  const groups = activeGroups();
  let visible = 0;
  cards.forEach(card => {{
    const text = card.dataset.search || "";
    const cardGroups = (card.dataset.groups || "").split(/\s+/).filter(Boolean);
    const queryMatch = !q || text.includes(q);
    const groupMatch = groups.length === 0 || groups.every(g => cardGroups.includes(g));
    const show = queryMatch && groupMatch;
    card.style.display = show ? "" : "none";
    if (show) visible++;
  }});
  visibleCount.textContent = visible;
}}
buttons.forEach(b => b.addEventListener("click", () => {{ b.classList.toggle("active"); applyFilters(); }}));
searchBox.addEventListener("input", applyFilters);
clearBtn.addEventListener("click", () => {{ searchBox.value = ""; buttons.forEach(b => b.classList.remove("active")); applyFilters(); }});
applyFilters();
</script>
</body>
</html>
'''


def main() -> None:
    root = repo_root()
    vis_dir = root / "ttk_runs_fixed" / "visual_inspection"
    obs_dir = root / "ttk_runs_fixed" / "observation_groups"
    src_csv = vis_dir / "visual_inspection_manifest.csv"
    if not src_csv.exists():
        src_csv = obs_dir / "recommended_visual_inspection_cases.csv"
    if not src_csv.exists():
        raise FileNotFoundError("Could not find visual_inspection_manifest.csv or fallback recommendation CSV.")
    rows = read_rows(src_csv)
    rows = sorted(rows, key=sample_id)
    metric_by_sample = load_metric_rows(root)
    out = vis_dir / "index.html"
    out.write_text(build_html(rows, metric_by_sample, vis_dir, root, src_csv), encoding="utf-8")
    print(f"repo_root={root}")
    print(f"source_csv={src_csv}")
    print(f"wrote={out}")
    print(f"sample_count={len(rows)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
