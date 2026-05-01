#!/usr/bin/env python3
"""
Generate / rebuild the TopoAware SR visual inspection index in a more readable
card-based layout.

This script is designed to be run from:
    ~/PhIRE/scripts

It looks for:
    ~/PhIRE/ttk_runs_fixed/observation_groups/recommended_visual_inspection_cases.csv
and existing visual inspection assets under:
    ~/PhIRE/ttk_runs_fixed/visual_inspection/

It then writes:
    ~/PhIRE/ttk_runs_fixed/visual_inspection/index.html

The generated HTML is a responsive, card-based interface with:
- search
- clickable group filters
- larger preview images
- wrapped group tags
- full panel / crop panel links
"""

from __future__ import annotations

import csv
import html
import os
import re
import sys
from pathlib import Path
from collections import Counter


# ============================================================
# Path helpers
# ============================================================

def resolve_repo_root() -> Path:
    """
    Resolve repo root robustly whether the script is run from:
      - ~/PhIRE/scripts
      - ~/PhIRE
      - another working directory
    """
    candidates = []

    try:
        candidates.append(Path(__file__).resolve().parent)
        candidates.append(Path(__file__).resolve().parent.parent)
    except NameError:
        pass

    candidates.append(Path.cwd().resolve())
    candidates.append(Path.cwd().resolve().parent)

    checked = []
    for c in candidates:
        if c in checked:
            continue
        checked.append(c)

        # direct repo root
        if (c / "ttk_runs_fixed").exists():
            return c

        # if candidate is scripts/
        if c.name == "scripts" and (c.parent / "ttk_runs_fixed").exists():
            return c.parent

    raise RuntimeError(
        "Could not locate repo root containing 'ttk_runs_fixed'. "
        "Please run this from ~/PhIRE/scripts or ~/PhIRE."
    )


# ============================================================
# CSV helpers
# ============================================================

def read_csv_rows(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def get_first(row: dict, *keys, default=""):
    for key in keys:
        if key in row and str(row[key]).strip():
            return str(row[key]).strip()
    return default


def parse_sample_id(row: dict) -> int:
    raw = get_first(row, "sample", "sample_id", "Sample", "id")
    if not raw:
        raise ValueError(f"Could not find sample id in row: {row}")
    m = re.search(r"\d+", str(raw))
    if not m:
        raise ValueError(f"Could not parse sample id from: {raw}")
    return int(m.group(0))


def split_groups(text: str):
    if not text:
        return []
    parts = re.split(r"[;,|]", text)
    return [p.strip() for p in parts if p.strip()]


def pretty_group_name(name: str) -> str:
    return name.replace("_", " ")


# ============================================================
# Input discovery
# ============================================================

def find_cases_csv(obs_dir: Path, vis_dir: Path) -> Path:
    candidates = [
        vis_dir / "recommended_visual_inspection_cases.csv",
        vis_dir / "visual_inspection_cases.csv",
        vis_dir / "visual_inspection_index.csv",
        obs_dir / "recommended_visual_inspection_cases.csv",
        obs_dir / "recommended_visual_inspection_unique_samples.csv",
        obs_dir / "recommended_visual_inspection.csv",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        "Could not find a visual inspection cases CSV.\n"
        "Looked in:\n  - " + "\n  - ".join(str(x) for x in candidates)
    )


def resolve_existing_asset(vis_dir: Path, sample_id: int, kind: str) -> Path | None:
    """
    Tries common filenames/folders for existing generated assets.
    """
    if kind == "panel":
        candidates = [
            vis_dir / "panels" / f"sample_{sample_id:03d}_panel.png",
            vis_dir / "full_panels" / f"sample_{sample_id:03d}_panel.png",
            vis_dir / f"sample_{sample_id:03d}_panel.png",
        ]
    elif kind == "crop":
        candidates = [
            vis_dir / "crop_panels" / f"sample_{sample_id:03d}_crop_panel.png",
            vis_dir / "crops" / f"sample_{sample_id:03d}_crop_panel.png",
            vis_dir / f"sample_{sample_id:03d}_crop_panel.png",
        ]
    elif kind == "thumb":
        candidates = [
            vis_dir / "thumbnails" / f"sample_{sample_id:03d}_thumb.png",
            vis_dir / "thumbs" / f"sample_{sample_id:03d}_thumb.png",
            vis_dir / "thumbnails" / f"sample_{sample_id:03d}_panel.png",
            vis_dir / "panels" / f"sample_{sample_id:03d}_panel.png",
            vis_dir / "full_panels" / f"sample_{sample_id:03d}_panel.png",
        ]
    else:
        raise ValueError(f"Unknown asset kind: {kind}")

    for c in candidates:
        if c.exists():
            return c
    return None


# ============================================================
# HTML generation
# ============================================================

def relpath_if_exists(path: Path | None, start: Path) -> str:
    if path is None or not path.exists():
        return ""
    return os.path.relpath(path, start)


def build_group_filter_buttons(group_counts: Counter) -> str:
    items = []
    for group, count in sorted(group_counts.items(), key=lambda x: (-x[1], x[0])):
        items.append(
            f"""
            <button class="group-filter" data-group="{html.escape(group)}" type="button">
              {html.escape(pretty_group_name(group))} <span class="group-count">{count}</span>
            </button>
            """
        )
    return "\n".join(items)


def build_cards(records: list[dict], vis_dir: Path) -> str:
    cards = []

    for row in records:
        sample_id = parse_sample_id(row)

        groups_text = get_first(row, "groups", "group_membership", "membership", "group_list")
        groups = split_groups(groups_text)

        question = get_first(
            row,
            "question",
            "inspection_question",
            "prompt_question",
            default="Visual inspection: compare GT, CNN, and GAN structure."
        )

        note = get_first(row, "note", "notes", "comment", default="")

        # Prefer paths in CSV if present, otherwise infer from common filenames
        panel_from_csv = get_first(row, "panel_path", "full_panel", "full_panel_path", "panel")
        crop_from_csv = get_first(row, "crop_panel", "crop_panel_path", "crop")
        thumb_from_csv = get_first(row, "thumbnail", "thumb", "thumbnail_path")

        def resolve_from_csv(raw: str) -> Path | None:
            if not raw:
                return None
            p = Path(raw)
            if p.is_absolute() and p.exists():
                return p
            # Try relative to repo visual directory
            p1 = vis_dir / raw
            if p1.exists():
                return p1
            # Try relative to repo root shape like ttk_runs_fixed/...
            p2 = vis_dir.parent.parent / raw
            if p2.exists():
                return p2
            return None

        panel_path = resolve_from_csv(panel_from_csv) or resolve_existing_asset(vis_dir, sample_id, "panel")
        crop_path = resolve_from_csv(crop_from_csv) or resolve_existing_asset(vis_dir, sample_id, "crop")
        thumb_path = resolve_from_csv(thumb_from_csv) or resolve_existing_asset(vis_dir, sample_id, "thumb")

        # Fallback preview to full panel if no thumb exists
        preview_path = thumb_path or panel_path or crop_path

        panel_rel = relpath_if_exists(panel_path, vis_dir)
        crop_rel = relpath_if_exists(crop_path, vis_dir)
        preview_rel = relpath_if_exists(preview_path, vis_dir)

        tags_html = "\n".join(
            f'<span class="tag">{html.escape(pretty_group_name(g))}</span>'
            for g in groups
        )

        search_blob = " ".join(
            [str(sample_id), groups_text, question, note]
        ).lower()

        links = []
        if panel_rel:
            links.append(f'<a href="{html.escape(panel_rel)}" target="_blank" rel="noopener">Open full panel</a>')
        if crop_rel:
            links.append(f'<a href="{html.escape(crop_rel)}" target="_blank" rel="noopener">Open crop panel</a>')

        links_html = " ".join(links) if links else '<span class="muted">No linked panel files found.</span>'

        preview_html = (
            f'<a href="{html.escape(panel_rel or crop_rel or preview_rel)}" target="_blank" rel="noopener">'
            f'  <img src="{html.escape(preview_rel)}" alt="Sample {sample_id} preview">'
            f'</a>'
            if preview_rel else
            '<div class="no-preview">No preview found</div>'
        )

        cards.append(
            f"""
            <article class="sample-card"
                     data-sample="{sample_id}"
                     data-groups="{' '.join(html.escape(g) for g in groups)}"
                     data-search="{html.escape(search_blob)}">
              <div class="sample-meta">
                <div class="sample-header">
                  <h2>Sample {sample_id}</h2>
                </div>

                <div class="meta-block">
                  <div class="meta-label">Question</div>
                  <div class="meta-value">{html.escape(question)}</div>
                </div>

                <div class="meta-block">
                  <div class="meta-label">Groups</div>
                  <div class="tag-list">
                    {tags_html if tags_html else '<span class="muted">No groups listed</span>'}
                  </div>
                </div>

                {f'''
                <div class="meta-block">
                  <div class="meta-label">Notes</div>
                  <div class="meta-value">{html.escape(note)}</div>
                </div>
                ''' if note else ''}

                <div class="meta-block links">
                  {links_html}
                </div>
              </div>

              <div class="sample-preview">
                {preview_html}
              </div>
            </article>
            """
        )

    return "\n".join(cards)


def build_html(records: list[dict], vis_dir: Path) -> str:
    all_groups = []
    for row in records:
        all_groups.extend(split_groups(get_first(row, "groups", "group_membership", "membership", "group_list")))
    group_counts = Counter(all_groups)

    buttons_html = build_group_filter_buttons(group_counts)
    cards_html = build_cards(records, vis_dir)

    total = len(records)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>TopoAware SR visual inspection index</title>
  <style>
    :root {{
      --bg: #f7f7f9;
      --card: #ffffff;
      --border: #dddddd;
      --text: #111111;
      --muted: #666666;
      --blue-bg: #eef3ff;
      --blue-border: #c8d7ff;
      --accent: #0056b3;
      --shadow: 0 1px 4px rgba(0,0,0,0.05);
    }}

    * {{
      box-sizing: border-box;
    }}

    body {{
      font-family: Arial, Helvetica, sans-serif;
      margin: 0;
      padding: 24px;
      background: var(--bg);
      color: var(--text);
      line-height: 1.45;
    }}

    .page {{
      max-width: 1500px;
      margin: 0 auto;
    }}

    h1 {{
      margin: 0 0 8px 0;
      font-size: 2.25rem;
      line-height: 1.15;
    }}

    .subtitle {{
      margin-bottom: 10px;
      color: #222;
      font-size: 1.1rem;
    }}

    .helper {{
      color: var(--muted);
      margin-bottom: 20px;
    }}

    .controls {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 16px;
      margin-bottom: 22px;
      box-shadow: var(--shadow);
      position: sticky;
      top: 10px;
      z-index: 10;
    }}

    .controls-top {{
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      align-items: center;
      margin-bottom: 14px;
    }}

    .search-box {{
      flex: 1 1 360px;
      min-width: 240px;
    }}

    .search-box input {{
      width: 100%;
      padding: 12px 14px;
      border: 1px solid #ccc;
      border-radius: 8px;
      font-size: 15px;
    }}

    .stats {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
      font-size: 14px;
      color: #333;
    }}

    .stat-pill {{
      background: #f1f1f1;
      border: 1px solid #ddd;
      border-radius: 999px;
      padding: 7px 12px;
    }}

    .clear-btn {{
      border: 1px solid #ccc;
      background: #fff;
      padding: 10px 14px;
      border-radius: 8px;
      cursor: pointer;
      font-weight: 600;
    }}

    .clear-btn:hover {{
      background: #f4f4f4;
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
      line-height: 1.2;
    }}

    .group-filter.active {{
      background: var(--blue-bg);
      border-color: var(--blue-border);
      font-weight: 700;
    }}

    .group-count {{
      color: var(--muted);
      margin-left: 4px;
    }}

    .index-container {{
      display: flex;
      flex-direction: column;
      gap: 18px;
    }}

    .sample-card {{
      display: grid;
      grid-template-columns: 1.55fr 1fr;
      gap: 20px;
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 18px;
      box-shadow: var(--shadow);
      align-items: start;
    }}

    .sample-header h2 {{
      margin: 0 0 10px 0;
      font-size: 1.6rem;
    }}

    .meta-block {{
      margin-bottom: 14px;
    }}

    .meta-label {{
      font-weight: 700;
      margin-bottom: 6px;
      color: #333;
    }}

    .meta-value {{
      color: #222;
      white-space: normal;
      word-break: break-word;
    }}

    .tag-list {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }}

    .tag {{
      background: var(--blue-bg);
      border: 1px solid var(--blue-border);
      border-radius: 999px;
      padding: 5px 10px;
      font-size: 13px;
      white-space: normal;
      word-break: break-word;
    }}

    .links {{
      display: flex;
      flex-wrap: wrap;
      gap: 14px;
      align-items: center;
    }}

    .links a {{
      text-decoration: none;
      color: var(--accent);
      font-weight: 600;
    }}

    .links a:hover {{
      text-decoration: underline;
    }}

    .sample-preview {{
      display: flex;
      justify-content: flex-end;
      align-items: flex-start;
    }}

    .sample-preview img {{
      width: 100%;
      max-width: 560px;
      height: auto;
      border: 1px solid #ccc;
      border-radius: 6px;
      display: block;
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
      color: var(--muted);
      background: #fafafa;
      padding: 20px;
      text-align: center;
    }}

    .muted {{
      color: var(--muted);
    }}

    .footer-note {{
      margin-top: 28px;
      color: var(--muted);
      font-size: 13px;
    }}

    @media (max-width: 1100px) {{
      .sample-card {{
        grid-template-columns: 1fr;
      }}

      .sample-preview {{
        justify-content: flex-start;
      }}

      .sample-preview img {{
        max-width: 100%;
      }}
    }}

    @media (max-width: 700px) {{
      body {{
        padding: 14px;
      }}

      h1 {{
        font-size: 1.8rem;
      }}

      .controls {{
        position: static;
      }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <h1>TopoAware SR visual inspection index</h1>
    <div class="subtitle">
      Each panel shows: GT speed | CNN speed | GAN speed | CNN-GT | GAN-GT.
    </div>
    <div class="helper">
      Use the search box and group filters to narrow the inspection set. Click a preview to open the full panel.
    </div>

    <section class="controls">
      <div class="controls-top">
        <div class="search-box">
          <input type="text" id="searchBox" placeholder="Search sample ID, group, or question...">
        </div>
        <button type="button" class="clear-btn" id="clearFiltersBtn">Clear filters</button>
        <div class="stats">
          <span class="stat-pill">Total samples: <strong>{total}</strong></span>
          <span class="stat-pill">Visible: <strong id="visibleCount">{total}</strong></span>
          <span class="stat-pill">Group filters: <strong>{len(group_counts)}</strong></span>
        </div>
      </div>

      <div class="filter-title">Filter by group</div>
      <div class="group-filters">
        {buttons_html}
      </div>
    </section>

    <section class="index-container" id="cardContainer">
      {cards_html}
    </section>

    <div class="footer-note">
      Generated automatically by <code>scripts/generate_visual_inspection_panels.py</code>.
    </div>
  </div>

  <script>
    const searchBox = document.getElementById('searchBox');
    const clearBtn = document.getElementById('clearFiltersBtn');
    const groupButtons = Array.from(document.querySelectorAll('.group-filter'));
    const cards = Array.from(document.querySelectorAll('.sample-card'));
    const visibleCount = document.getElementById('visibleCount');

    function activeGroups() {{
      return groupButtons
        .filter(btn => btn.classList.contains('active'))
        .map(btn => btn.dataset.group);
    }}

    function applyFilters() {{
      const q = searchBox.value.trim().toLowerCase();
      const groups = activeGroups();
      let visible = 0;

      cards.forEach(card => {{
        const hay = (card.dataset.search || '').toLowerCase();
        const cardGroups = (card.dataset.groups || '').split(/\\s+/).filter(Boolean);

        const queryMatch = !q || hay.includes(q);
        const groupMatch = groups.length === 0 || groups.every(g => cardGroups.includes(g));

        const show = queryMatch && groupMatch;
        card.style.display = show ? '' : 'none';
        if (show) visible += 1;
      }});

      visibleCount.textContent = visible;
    }}

    groupButtons.forEach(btn => {{
      btn.addEventListener('click', () => {{
        btn.classList.toggle('active');
        applyFilters();
      }});
    }});

    searchBox.addEventListener('input', applyFilters);

    clearBtn.addEventListener('click', () => {{
      searchBox.value = '';
      groupButtons.forEach(btn => btn.classList.remove('active'));
      applyFilters();
    }});

    applyFilters();
  </script>
</body>
</html>
"""


# ============================================================
# Main
# ============================================================

def main():
    repo_root = resolve_repo_root()

    vis_dir = repo_root / "ttk_runs_fixed" / "visual_inspection"
    obs_dir = repo_root / "ttk_runs_fixed" / "observation_groups"

    vis_dir.mkdir(parents=True, exist_ok=True)

    cases_csv = find_cases_csv(obs_dir, vis_dir)
    rows = read_csv_rows(cases_csv)

    if not rows:
        raise RuntimeError(f"No rows found in CSV: {cases_csv}")

    # Sort by sample id if available
    rows = sorted(rows, key=parse_sample_id)

    html_text = build_html(rows, vis_dir)

    out_html = vis_dir / "index.html"
    out_html.write_text(html_text, encoding="utf-8")

    print(f"repo_root={repo_root}")
    print(f"visual_inspection_dir={vis_dir}")
    print(f"cases_csv={cases_csv}")
    print(f"wrote={out_html}")
    print(f"sample_count={len(rows)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)