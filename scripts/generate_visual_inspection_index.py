#!/usr/bin/env python3
from __future__ import annotations

import csv
import html
import os
import re
import sys
from collections import Counter
from pathlib import Path


def repo_root() -> Path:
    here = Path(__file__).resolve().parent
    cwd = Path.cwd().resolve()

    for p in [here.parent, cwd.parent if cwd.name == "scripts" else cwd, here, cwd]:
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


def resolve_path(raw: str, vis_dir: Path, root: Path) -> Path | None:
    if not raw:
        return None

    p = Path(raw)
    candidates = []
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.extend([
            vis_dir / raw,
            root / raw,
            root / "ttk_runs_fixed" / "visual_inspection" / raw,
        ])

    for c in candidates:
        if c.exists():
            return c
    return None


def fallback_asset(vis_dir: Path, sid: int, kind: str) -> Path | None:
    if kind == "crop":
        candidates = [
            vis_dir / "panels_crop160" / f"sample_{sid:03d}_speed_error_crop.png",
        ]
    elif kind == "full":
        candidates = [
            vis_dir / "panels_full" / f"sample_{sid:03d}_speed_error_full.png",
        ]
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


def question_for(row: dict) -> str:
    q = first(row, "question", "suggested_question", "inspection_question", "reason")
    if q:
        return q

    ssim = first(row, "ssim_winner")
    pd = first(row, "pd_winner")
    mt = first(row, "mt_winner")
    if ssim or pd or mt:
        return f"Inspect structure. Winners: SSIM={ssim or '?'}, PD={pd or '?'}, MT={mt or '?'}."

    return "Inspect GT, CNN, GAN, and error maps."


def winners_for(row: dict) -> str:
    pairs = [
        ("psnr_winner", "PSNR"),
        ("ssim_winner", "SSIM"),
        ("pd_winner", "PD"),
        ("mt_winner", "MT"),
        ("direct_error_group_winner", "Direct"),
        ("distributional_group_winner", "Distributional"),
        ("tail_group_winner", "Tail"),
        ("configured_physics_group_winner", "Physics"),
    ]
    vals = []
    for k, label in pairs:
        v = first(row, k)
        if v:
            vals.append(f"{label}: {v}")
    return " | ".join(vals)


def build_cards(rows: list[dict], vis_dir: Path, root: Path) -> str:
    cards = []

    for row in rows:
        sid = sample_id(row)
        groups = groups_for(row)
        question = question_for(row)
        winners = winners_for(row)

        crop = (
            resolve_path(first(row, "crop_panel_path", "crop_panel", "crop"), vis_dir, root)
            or fallback_asset(vis_dir, sid, "crop")
        )
        full = (
            resolve_path(first(row, "full_panel_path", "full_panel", "panel_path", "panel"), vis_dir, root)
            or fallback_asset(vis_dir, sid, "full")
        )
        preview = crop or full or fallback_asset(vis_dir, sid, "preview")

        crop_rel = rel(crop, vis_dir)
        full_rel = rel(full, vis_dir)
        preview_rel = rel(preview, vis_dir)
        target_rel = full_rel or crop_rel or preview_rel

        tag_html = "\n".join(
            f'<span class="tag">{html.escape(pretty(g))}</span>'
            for g in groups
        ) or '<span class="muted">No groups listed</span>'

        links = []
        if crop_rel:
            links.append(f'<a href="{html.escape(crop_rel)}" target="_blank">Open crop panel</a>')
        if full_rel:
            links.append(f'<a href="{html.escape(full_rel)}" target="_blank">Open full panel</a>')
        links_html = " ".join(links) if links else '<span class="muted">No linked panel found</span>'

        preview_html = (
            f'<a href="{html.escape(target_rel)}" target="_blank">'
            f'<img src="{html.escape(preview_rel)}" alt="Sample {sid} preview"></a>'
            if preview_rel
            else '<div class="no-preview">No preview found</div>'
        )

        search = " ".join([str(sid), " ".join(groups), question, winners]).lower()

        cards.append(f"""
<article class="sample-card"
         data-search="{html.escape(search)}"
         data-groups="{' '.join(html.escape(g) for g in groups)}">
  <div class="sample-meta">
    <h2>Sample {sid}</h2>
    {'<div class="winner-strip">' + html.escape(winners) + '</div>' if winners else ''}

    <div class="meta-block">
      <div class="meta-label">Question</div>
      <div class="meta-value">{html.escape(question)}</div>
    </div>

    <div class="meta-block">
      <div class="meta-label">Groups</div>
      <div class="tag-list">{tag_html}</div>
    </div>

    <div class="meta-block links">{links_html}</div>
  </div>

  <div class="sample-preview">{preview_html}</div>
</article>
""")

    return "\n".join(cards)


def build_html(rows: list[dict], vis_dir: Path, root: Path, src_csv: Path) -> str:
    all_groups = []
    for r in rows:
        all_groups.extend(groups_for(r))

    counts = Counter(all_groups)

    buttons = "\n".join(
        f'<button class="group-filter" data-group="{html.escape(g)}" type="button">'
        f'{html.escape(pretty(g))} <span>{n}</span></button>'
        for g, n in sorted(counts.items(), key=lambda x: (-x[1], x[0]))
    )

    cards = build_cards(rows, vis_dir, root)

    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>TopoAware SR visual inspection index</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    padding: 24px;
    font-family: Arial, Helvetica, sans-serif;
    background: #f7f7f9;
    color: #111;
    line-height: 1.45;
  }}
  .page {{
    max-width: 1500px;
    margin: 0 auto;
  }}
  h1 {{
    margin: 0 0 8px 0;
    font-size: 2.2rem;
  }}
  .subtitle {{
    margin-bottom: 6px;
    font-size: 1.05rem;
  }}
  .helper {{
    color: #666;
    margin-bottom: 20px;
    font-size: 0.95rem;
  }}
  .controls {{
    position: sticky;
    top: 10px;
    z-index: 10;
    background: white;
    border: 1px solid #ddd;
    border-radius: 10px;
    padding: 16px;
    margin-bottom: 22px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
  }}
  .controls-top {{
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    align-items: center;
    margin-bottom: 14px;
  }}
  #searchBox {{
    flex: 1 1 360px;
    min-width: 260px;
    padding: 12px 14px;
    border: 1px solid #ccc;
    border-radius: 8px;
    font-size: 15px;
  }}
  .clear-btn {{
    border: 1px solid #ccc;
    background: #fff;
    padding: 10px 14px;
    border-radius: 8px;
    cursor: pointer;
    font-weight: 600;
  }}
  .stat-pill {{
    background: #f1f1f1;
    border: 1px solid #ddd;
    border-radius: 999px;
    padding: 7px 12px;
    font-size: 14px;
  }}
  .filter-title {{
    font-weight: 700;
    margin-bottom: 8px;
  }}
  .group-filters {{
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
  }}
  .group-filter {{
    background: #fff;
    border: 1px solid #ccc;
    border-radius: 999px;
    padding: 8px 12px;
    cursor: pointer;
    font-size: 13px;
  }}
  .group-filter.active {{
    background: #eef3ff;
    border-color: #c8d7ff;
    font-weight: 700;
  }}
  .index-container {{
    display: flex;
    flex-direction: column;
    gap: 18px;
  }}
  .sample-card {{
    display: grid;
    grid-template-columns: minmax(0, 1.45fr) minmax(360px, 1fr);
    gap: 20px;
    background: white;
    border: 1px solid #ddd;
    border-radius: 10px;
    padding: 18px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    align-items: start;
  }}
  .sample-card h2 {{
    margin: 0 0 8px 0;
    font-size: 1.55rem;
  }}
  .winner-strip {{
    color: #333;
    background: #f5f5f5;
    border: 1px solid #e1e1e1;
    border-radius: 8px;
    padding: 8px 10px;
    font-size: 13px;
    margin-bottom: 12px;
    word-break: break-word;
  }}
  .meta-block {{
    margin-bottom: 14px;
  }}
  .meta-label {{
    font-weight: 700;
    margin-bottom: 6px;
  }}
  .tag-list {{
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
  }}
  .tag {{
    background: #eef3ff;
    border: 1px solid #c8d7ff;
    border-radius: 999px;
    padding: 5px 10px;
    font-size: 13px;
  }}
  .links {{
    display: flex;
    flex-wrap: wrap;
    gap: 14px;
  }}
  .links a {{
    color: #0056b3;
    font-weight: 600;
    text-decoration: none;
  }}
  .links a:hover {{
    text-decoration: underline;
  }}
  .sample-preview {{
    display: flex;
    justify-content: flex-end;
  }}
  .sample-preview img {{
    width: 100%;
    max-width: 620px;
    border: 1px solid #ccc;
    border-radius: 6px;
    background: white;
  }}
  .no-preview {{
    width: 100%;
    min-height: 220px;
    border: 1px dashed #bbb;
    border-radius: 6px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #666;
    background: #fafafa;
  }}
  .muted {{
    color: #666;
  }}
  @media (max-width: 1100px) {{
    .sample-card {{ grid-template-columns: 1fr; }}
    .sample-preview {{ justify-content: flex-start; }}
    .sample-preview img {{ max-width: 100%; }}
  }}
</style>
</head>
<body>
<div class="page">
  <h1>TopoAware SR visual inspection index</h1>
  <div class="subtitle">Each panel shows: GT speed | CNN speed | GAN speed | CNN-GT | GAN-GT.</div>
  <div class="helper">Source CSV: <code>{html.escape(str(src_csv))}</code>. Use search and group filters to narrow samples.</div>

  <section class="controls">
    <div class="controls-top">
      <input id="searchBox" type="text" placeholder="Search sample ID, group, question, or winner...">
      <button class="clear-btn" id="clearBtn" type="button">Clear filters</button>
      <span class="stat-pill">Total: <strong>{len(rows)}</strong></span>
      <span class="stat-pill">Visible: <strong id="visibleCount">{len(rows)}</strong></span>
    </div>

    <div class="filter-title">Filter by group</div>
    <div class="group-filters">{buttons}</div>
  </section>

  <section class="index-container" id="cards">
    {cards}
  </section>
</div>

<script>
const searchBox = document.getElementById("searchBox");
const clearBtn = document.getElementById("clearBtn");
const cards = Array.from(document.querySelectorAll(".sample-card"));
const buttons = Array.from(document.querySelectorAll(".group-filter"));
const visibleCount = document.getElementById("visibleCount");

function activeGroups() {{
  return buttons.filter(b => b.classList.contains("active")).map(b => b.dataset.group);
}}

function applyFilters() {{
  const q = searchBox.value.trim().toLowerCase();
  const groups = activeGroups();
  let visible = 0;

  cards.forEach(card => {{
    const text = card.dataset.search || "";
    const cardGroups = (card.dataset.groups || "").split(/\\s+/).filter(Boolean);

    const queryMatch = !q || text.includes(q);
    const groupMatch = groups.length === 0 || groups.every(g => cardGroups.includes(g));

    const show = queryMatch && groupMatch;
    card.style.display = show ? "" : "none";
    if (show) visible++;
  }});

  visibleCount.textContent = visible;
}}

buttons.forEach(b => b.addEventListener("click", () => {{
  b.classList.toggle("active");
  applyFilters();
}}));

searchBox.addEventListener("input", applyFilters);

clearBtn.addEventListener("click", () => {{
  searchBox.value = "";
  buttons.forEach(b => b.classList.remove("active"));
  applyFilters();
}});

applyFilters();
</script>
</body>
</html>
"""


def main() -> None:
    root = repo_root()
    vis_dir = root / "ttk_runs_fixed" / "visual_inspection"
    obs_dir = root / "ttk_runs_fixed" / "observation_groups"

    src_csv = vis_dir / "visual_inspection_manifest.csv"
    if not src_csv.exists():
        # Fallback only if needed
        src_csv = obs_dir / "recommended_visual_inspection_cases.csv"

    if not src_csv.exists():
        raise FileNotFoundError(f"Could not find visual inspection manifest or fallback CSV.")

    rows = read_rows(src_csv)
    rows = sorted(rows, key=sample_id)

    out = vis_dir / "index.html"
    out.write_text(build_html(rows, vis_dir, root, src_csv), encoding="utf-8")

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
