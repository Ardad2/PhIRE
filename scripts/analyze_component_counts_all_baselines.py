#!/usr/bin/env python3
"""Count connected components in wind-speed superlevel sets for GT, Bicubic, CNN, and GAN.

This is a lightweight topology proxy that does not require TTK or Docker.
For each sample and threshold the superlevel set  { x : speed(x) >= t }  is
computed; its 2-D connected components (8-connectivity) are counted; and the
per-method absolute error relative to GT is recorded.

Thresholds
----------
  Fixed (m/s):     5, 10, 15
  GT-percentile:   p90, p95, p99 of that sample's GT speed field

Inputs
------
  data_out_fixed/wind_mrhr_cnn/{idx,dataGT,dataSR}.npy
  data_out_fixed/wind_mrhr_gan/{idx,dataGT,dataSR}.npy
  data_out_fixed/wind_mrhr_bicubic/{idx,dataGT,dataSR}.npy

Outputs (all under ttk_runs_fixed/component_counts/)
-------
  component_counts_per_sample.csv   — one row per (sample, threshold)
  component_count_winner_counts.csv — wins per method per threshold
  component_count_summary.csv       — mean errors / win counts per threshold
  index.html                        — lightweight HTML report

Run from scripts/ or repo root:
    cd ~/PhIRE/scripts
    PYTHONNOUSERSITE=1 /usr/bin/python3 analyze_component_counts_all_baselines.py
"""

from __future__ import annotations

import argparse
import csv
import html
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import label as _nd_label


# ---------------------------------------------------------------------------
# Repo layout
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parent

DATA_ROOT = REPO_ROOT / "data_out_fixed"
CNN_DIR   = DATA_ROOT / "wind_mrhr_cnn"
GAN_DIR   = DATA_ROOT / "wind_mrhr_gan"
BIC_DIR   = DATA_ROOT / "wind_mrhr_bicubic"

OUT_DIR   = REPO_ROOT / "ttk_runs_fixed" / "component_counts"

METHODS = ("bicubic", "cnn", "gan")
METHOD_DIRS = {"bicubic": BIC_DIR, "cnn": CNN_DIR, "gan": GAN_DIR}
METHOD_LABELS = {"bicubic": "Bicubic", "cnn": "CNN", "gan": "GAN"}

FIXED_THRESHOLDS: List[float] = [5.0, 10.0, 15.0]
PERCENTILES: List[float]      = [90.0, 95.0, 99.0]

# 8-connectivity kernel for scipy.ndimage.label
_STRUCT8 = np.ones((3, 3), dtype=np.int32)


# ---------------------------------------------------------------------------
# Array loading and verification
# ---------------------------------------------------------------------------

def _load(directory: Path, name: str, mmap: bool = True) -> np.ndarray:
    p = directory / name
    if not p.exists():
        raise SystemExit(
            f"[error] Missing {p}\n"
            f"  Ensure {directory.name} outputs exist before running this script."
        )
    return np.load(p, mmap_mode="r" if mmap else None)


def load_method(directory: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (idx, dataGT, dataSR) from a method directory."""
    return (
        _load(directory, "idx.npy", mmap=False),
        _load(directory, "dataGT.npy", mmap=True),
        _load(directory, "dataSR.npy", mmap=True),
    )


def verify_alignment(
    cnn_idx: np.ndarray, gan_idx: np.ndarray, bic_idx: np.ndarray,
    cnn_gt:  np.ndarray, gan_gt:  np.ndarray, bic_gt:  np.ndarray,
) -> None:
    if not np.array_equal(cnn_idx, gan_idx):
        raise SystemExit("[error] CNN and GAN idx arrays differ.")
    if not np.array_equal(cnn_idx, bic_idx):
        raise SystemExit("[error] CNN and Bicubic idx arrays differ.")
    print(f"  idx OK  — shape={cnn_idx.shape}, range=[{int(cnn_idx.min())}..{int(cnn_idx.max())}]")

    max_cnn_gan = float(np.max(np.abs(cnn_gt[:] - gan_gt[:])))
    max_cnn_bic = float(np.max(np.abs(cnn_gt[:] - bic_gt[:])))
    if max_cnn_gan > 1e-9:
        print(f"  [warn] CNN GT vs GAN GT max_abs_diff={max_cnn_gan:.3e}")
    else:
        print(f"  CNN GT == GAN GT  ✓  (max_diff={max_cnn_gan:.1e})")
    if max_cnn_bic > 1e-9:
        print(f"  [warn] CNN GT vs Bicubic GT max_abs_diff={max_cnn_bic:.3e}")
    else:
        print(f"  CNN GT == Bicubic GT  ✓  (max_diff={max_cnn_bic:.1e})")


# ---------------------------------------------------------------------------
# Speed and component-count helpers
# ---------------------------------------------------------------------------

def _speed(uv: np.ndarray) -> np.ndarray:
    """(H, W, 2) [u, v] → (H, W) speed magnitude, float32."""
    return np.sqrt(
        uv[..., 0].astype(np.float32) ** 2 + uv[..., 1].astype(np.float32) ** 2
    )


def _count_components(speed_2d: np.ndarray, threshold: float) -> int:
    """Number of 8-connected components in the superlevel set speed >= threshold."""
    mask = speed_2d >= threshold
    if not np.any(mask):
        return 0
    _, n = _nd_label(mask, structure=_STRUCT8)
    return int(n)


def threshold_label(t: float, is_percentile: bool, pct: Optional[float] = None) -> str:
    """Canonical string key for a threshold, e.g. 't5', 'p90'."""
    if is_percentile and pct is not None:
        return f"p{int(pct)}"
    return f"t{t:g}"


# ---------------------------------------------------------------------------
# Per-sample computation
# ---------------------------------------------------------------------------

def process_sample(
    si: int,
    gt_uv:  np.ndarray,   # (H, W, 2)
    sr_uvs: Dict[str, np.ndarray],  # method → (H, W, 2)
) -> List[Dict]:
    """Return one dict per threshold for sample si."""
    gt_spd = _speed(gt_uv)
    sr_spds = {m: _speed(sr_uvs[m]) for m in METHODS}

    rows: List[Dict] = []

    # fixed thresholds
    for t in FIXED_THRESHOLDS:
        tlabel = threshold_label(t, False)
        gt_n   = _count_components(gt_spd, t)
        counts = {m: _count_components(sr_spds[m], t) for m in METHODS}
        rows.append(_make_row(si, tlabel, f"{t:g}", "fixed", t, gt_n, counts))

    # GT-percentile thresholds
    for pct in PERCENTILES:
        t = float(np.percentile(gt_spd, pct))
        tlabel = threshold_label(t, True, pct)
        gt_n   = _count_components(gt_spd, t)
        counts = {m: _count_components(sr_spds[m], t) for m in METHODS}
        rows.append(_make_row(si, tlabel, f"{pct:.0f}th pct", "percentile", t, gt_n, counts))

    return rows


def _make_row(
    si:       int,
    tlabel:   str,
    tname:    str,
    ttype:    str,
    tval:     float,
    gt_n:     int,
    counts:   Dict[str, int],
) -> Dict:
    abs_errs = {m: abs(counts[m] - gt_n) for m in METHODS}

    # winner: method(s) with smallest abs error; ties reported as "tie"
    min_err = min(abs_errs.values())
    winners = [m for m in METHODS if abs_errs[m] == min_err]
    winner  = winners[0] if len(winners) == 1 else "tie"

    row: Dict = {
        "sample_idx":       si,
        "threshold_label":  tlabel,
        "threshold_name":   tname,
        "threshold_type":   ttype,
        "threshold_value":  round(tval, 6),
        "gt_count":         gt_n,
    }
    for m in METHODS:
        row[f"{m}_count"]     = counts[m]
        row[f"{m}_abs_error"] = abs_errs[m]
    row["winner"] = winner
    return row


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

ALL_TLABELS = [f"t{t:g}" for t in FIXED_THRESHOLDS] + [f"p{int(p)}" for p in PERCENTILES]


def compute_winner_counts(per_sample: List[Dict]) -> List[Dict]:
    """Wins per (method, threshold_label), including tie counts."""
    counts: Dict[str, Dict[str, int]] = {
        tl: {m: 0 for m in METHODS + ("tie",)}
        for tl in ALL_TLABELS
    }
    for row in per_sample:
        tl = row["threshold_label"]
        w  = row["winner"]
        if tl in counts:
            if w in counts[tl]:
                counts[tl][w] += 1
    out: List[Dict] = []
    for tl in ALL_TLABELS:
        entry = {"threshold_label": tl}
        entry.update(counts[tl])
        out.append(entry)
    return out


def compute_summary(per_sample: List[Dict]) -> List[Dict]:
    """Mean component counts and errors plus win/tie totals per threshold."""
    from collections import defaultdict
    buckets: Dict[str, List[Dict]] = defaultdict(list)
    for row in per_sample:
        buckets[row["threshold_label"]].append(row)

    out: List[Dict] = []
    for tl in ALL_TLABELS:
        rows = buckets.get(tl, [])
        if not rows:
            continue
        n = len(rows)

        def _mean(col: str) -> float:
            vals = [r[col] for r in rows if r[col] is not None]
            return sum(vals) / len(vals) if vals else float("nan")

        win_counts = {m: sum(1 for r in rows if r["winner"] == m) for m in METHODS}
        ties       = sum(1 for r in rows if r["winner"] == "tie")

        entry: Dict = {
            "threshold_label":       tl,
            "n_samples":             n,
            "mean_gt_count":         round(_mean("gt_count"), 3),
        }
        for m in METHODS:
            entry[f"mean_{m}_count"]     = round(_mean(f"{m}_count"), 3)
            entry[f"mean_{m}_abs_error"] = round(_mean(f"{m}_abs_error"), 3)
            entry[f"{m}_wins"]           = win_counts[m]
        entry["ties"] = ties
        out.append(entry)
    return out


# ---------------------------------------------------------------------------
# CSV writers
# ---------------------------------------------------------------------------

def _write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n")
        return
    cols = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def _fmt(v: object, decimals: int = 4) -> str:
    if v is None:
        return "—"
    try:
        f = float(str(v))  # type: ignore[arg-type]
        if math.isnan(f):
            return "—"
        if isinstance(v, int) or (f == int(f) and decimals == 0):
            return str(int(f))
        return f"{f:.{decimals}f}"
    except (TypeError, ValueError):
        return html.escape(str(v))


def _winner_badge(w: str) -> str:
    colours = {
        "bicubic": ("#dbeafe", "#1d4ed8"),
        "cnn":     ("#dcfce7", "#166534"),
        "gan":     ("#fef9c3", "#854d0e"),
        "tie":     ("#f3f4f6", "#4b5563"),
    }
    bg, fg = colours.get(w, ("#f3f4f6", "#1a1a1a"))
    label  = METHOD_LABELS.get(w, w.upper())
    return (
        f'<span style="background:{bg};color:{fg};padding:1px 6px;'
        f'border-radius:4px;font-size:0.8em;font-weight:600">{label}</span>'
    )


def _summary_table_html(summary: List[Dict]) -> str:
    hdr = (
        "<tr>"
        "<th>Threshold</th>"
        "<th>N</th>"
        "<th>Mean GT</th>"
        + "".join(
            f"<th>Mean {METHOD_LABELS[m]}</th><th>Mean |err| {METHOD_LABELS[m]}</th>"
            for m in METHODS
        )
        + "<th>Bic wins</th><th>CNN wins</th><th>GAN wins</th><th>Ties</th>"
        "</tr>"
    )
    body_rows: List[str] = []
    for r in summary:
        cells = (
            f"<td><code>{html.escape(str(r['threshold_label']))}</code></td>"
            f"<td>{r['n_samples']}</td>"
            f"<td>{_fmt(r['mean_gt_count'], 1)}</td>"
        )
        for m in METHODS:
            cells += (
                f"<td>{_fmt(r.get(f'mean_{m}_count'), 1)}</td>"
                f"<td>{_fmt(r.get(f'mean_{m}_abs_error'), 2)}</td>"
            )
        cells += (
            f"<td>{r.get('bicubic_wins', 0)}</td>"
            f"<td>{r.get('cnn_wins', 0)}</td>"
            f"<td>{r.get('gan_wins', 0)}</td>"
            f"<td>{r.get('ties', 0)}</td>"
        )
        body_rows.append(f"<tr>{cells}</tr>")
    return f"<table class='data-table'><thead>{hdr}</thead><tbody>{''.join(body_rows)}</tbody></table>"


def _per_threshold_section(tlabel: str, rows: List[Dict]) -> str:
    """One card per threshold showing the per-sample breakdown."""
    hdr = (
        "<tr>"
        "<th>Sample</th>"
        "<th>GT</th>"
        + "".join(
            f"<th>{METHOD_LABELS[m]}</th><th>|err|</th>"
            for m in METHODS
        )
        + "<th>Winner</th>"
        "</tr>"
    )
    body_rows: List[str] = []
    for r in sorted(rows, key=lambda x: int(x["sample_idx"])):
        cells = (
            f"<td>{int(r['sample_idx'])}</td>"
            f"<td>{int(r['gt_count'])}</td>"
        )
        for m in METHODS:
            cnt = int(r[f"{m}_count"])
            err = int(r[f"{m}_abs_error"])
            win_cls = ' class="winner-cell"' if r["winner"] == m else ""
            cells += f"<td{win_cls}>{cnt}</td><td{win_cls}>{err}</td>"
        cells += f"<td>{_winner_badge(r['winner'])}</td>"
        body_rows.append(f"<tr>{cells}</tr>")

    return (
        f'<div class="card">'
        f'<h2>Threshold: <code>{html.escape(tlabel)}</code></h2>'
        f"<p>GT percentile thresholds are computed per-sample from the GT field.</p>"
        f"<table class='data-table'><thead>{hdr}</thead><tbody>{''.join(body_rows)}</tbody></table>"
        f"</div>"
    )


CSS = """
*, *::before, *::after { box-sizing: border-box; }
body {
  font-family: system-ui, sans-serif;
  background: #f0f2f5;
  color: #1a1a1a;
  margin: 0; padding: 1.5em;
}
h1  { font-size: 1.3em; margin: 0 0 0.3em; }
h2  { font-size: 1em; margin: 0 0 0.6em; color: #1e293b; }
.intro { color: #555; font-size: 0.88em; margin: 0 0 1.5em; line-height: 1.6; }
.card {
  background: #fff;
  border: 1px solid #dde1e7;
  border-radius: 8px;
  box-shadow: 0 1px 4px rgba(0,0,0,.07);
  margin-bottom: 1.5em;
  padding: 1.25em 1.5em;
  overflow-x: auto;
}
.data-table {
  border-collapse: collapse;
  font-size: 0.82em;
  width: 100%;
}
.data-table th {
  background: #f7f8fa;
  color: #444;
  font-weight: 600;
  text-align: left;
  padding: 5px 9px;
  border: 1px solid #e2e5ea;
  white-space: nowrap;
}
.data-table td {
  padding: 3px 9px;
  border: 1px solid #e2e5ea;
  font-variant-numeric: tabular-nums;
  white-space: nowrap;
}
.data-table tr:nth-child(even) td { background: #fafbfc; }
.data-table td.winner-cell {
  background: #d1fae5;
  color: #065f46;
  font-weight: 600;
}
"""


def write_html(
    out_path: Path,
    summary: List[Dict],
    per_sample: List[Dict],
) -> None:
    from collections import defaultdict
    by_threshold: Dict[str, List[Dict]] = defaultdict(list)
    for row in per_sample:
        by_threshold[row["threshold_label"]].append(row)

    summary_html  = _summary_table_html(summary)
    sections_html = "\n".join(
        _per_threshold_section(tl, by_threshold.get(tl, []))
        for tl in ALL_TLABELS
    )

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Component count analysis — Bicubic / CNN / GAN</title>
<style>{CSS}</style>
</head>
<body>
<h1>Superlevel-set connected-component counts</h1>
<p class="intro">
  Proxy topology metric: number of 8-connected components in the superlevel set
  {{ x : speed(x) &ge; t }} for each method and threshold.<br>
  Fixed thresholds: 5, 10, 15 m/s &nbsp;|&nbsp;
  GT-percentile thresholds: p90, p95, p99 (computed per sample from GT).<br>
  Winner = method whose component count is closest to GT (lower absolute error).
</p>

<div class="card">
<h2>Summary across all {len(per_sample) // len(ALL_TLABELS)} samples</h2>
{summary_html}
</div>

{sections_html}
</body>
</html>
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(page, encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cnn-dir",     type=Path, default=CNN_DIR)
    ap.add_argument("--gan-dir",     type=Path, default=GAN_DIR)
    ap.add_argument("--bicubic-dir", type=Path, default=BIC_DIR)
    ap.add_argument("--outdir",      type=Path, default=OUT_DIR)
    ap.add_argument("--samples",     nargs="*", type=int, default=None,
                    help="Subset of sample indices (default: all)")
    return ap.parse_args()


def main() -> int:
    args = _parse_args()

    print("=" * 60)
    print("PhIRE component-count baseline analysis")
    print(f"  CNN dir     : {args.cnn_dir}")
    print(f"  GAN dir     : {args.gan_dir}")
    print(f"  Bicubic dir : {args.bicubic_dir}")
    print(f"  Output dir  : {args.outdir}")
    print("=" * 60)

    # ---- load ---------------------------------------------------------------
    print("\n[1] Loading arrays …")
    cnn_idx, cnn_gt, cnn_sr = load_method(args.cnn_dir)
    gan_idx, gan_gt, gan_sr = load_method(args.gan_dir)
    bic_idx, bic_gt, bic_sr = load_method(args.bicubic_dir)

    print(f"  CNN  GT={cnn_gt.shape}  SR={cnn_sr.shape}")
    print(f"  GAN  GT={gan_gt.shape}  SR={gan_sr.shape}")
    print(f"  Bic  GT={bic_gt.shape}  SR={bic_sr.shape}")

    # ---- verify -------------------------------------------------------------
    print("\n[2] Verifying alignment …")
    verify_alignment(cnn_idx, gan_idx, bic_idx, cnn_gt, gan_gt, bic_gt)

    n_total = cnn_gt.shape[0]
    samples = sorted(set(args.samples)) if args.samples else list(range(n_total))
    bad = [s for s in samples if s < 0 or s >= n_total]
    if bad:
        raise SystemExit(f"[error] Sample indices out of range [0,{n_total-1}]: {bad}")

    print(f"\n[3] Processing {len(samples)} samples × {len(ALL_TLABELS)} thresholds …")

    sr_arrays = {
        "bicubic": bic_sr,
        "cnn":     cnn_sr,
        "gan":     gan_sr,
    }

    per_sample_rows: List[Dict] = []
    for i, si in enumerate(samples):
        gt_uv  = np.asarray(cnn_gt[si])
        sr_uvs = {m: np.asarray(sr_arrays[m][si]) for m in METHODS}
        rows   = process_sample(si, gt_uv, sr_uvs)
        per_sample_rows.extend(rows)
        if (i + 1) % 20 == 0 or i == len(samples) - 1:
            print(f"  {i+1}/{len(samples)} samples done", flush=True)

    # ---- aggregate ----------------------------------------------------------
    print("\n[4] Aggregating …")
    winner_counts = compute_winner_counts(per_sample_rows)
    summary       = compute_summary(per_sample_rows)

    # quick console summary
    print(f"\n  {'Threshold':<10}  {'GT mean':>8}  {'Bic err':>8}  {'CNN err':>8}  {'GAN err':>8}  {'Bic W':>6}  {'CNN W':>6}  {'GAN W':>6}")
    print("  " + "-" * 75)
    for s in summary:
        print(
            f"  {s['threshold_label']:<10}"
            f"  {s['mean_gt_count']:>8.1f}"
            f"  {s['mean_bicubic_abs_error']:>8.2f}"
            f"  {s['mean_cnn_abs_error']:>8.2f}"
            f"  {s['mean_gan_abs_error']:>8.2f}"
            f"  {s['bicubic_wins']:>6}"
            f"  {s['cnn_wins']:>6}"
            f"  {s['gan_wins']:>6}"
        )

    # ---- write --------------------------------------------------------------
    print(f"\n[5] Writing outputs to {args.outdir} …")
    args.outdir.mkdir(parents=True, exist_ok=True)

    p_per   = args.outdir / "component_counts_per_sample.csv"
    p_wins  = args.outdir / "component_count_winner_counts.csv"
    p_sum   = args.outdir / "component_count_summary.csv"
    p_html  = args.outdir / "index.html"

    _write_csv(p_per,  per_sample_rows)
    _write_csv(p_wins, winner_counts)
    _write_csv(p_sum,  summary)
    write_html(p_html, summary, per_sample_rows)

    print(f"  {p_per.name}     ({p_per.stat().st_size:,} bytes)")
    print(f"  {p_wins.name} ({p_wins.stat().st_size:,} bytes)")
    print(f"  {p_sum.name}     ({p_sum.stat().st_size:,} bytes)")
    print(f"  {p_html.name}             ({p_html.stat().st_size:,} bytes)")
    print(f"\n[done]  {len(per_sample_rows)} rows written "
          f"({len(samples)} samples × {len(ALL_TLABELS)} thresholds).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
