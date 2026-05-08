#!/usr/bin/env python3
"""Generate 7-column visual comparison panels for CNN, GAN, and bicubic outputs.

Each panel shows one sample as a single row of seven subplots:
    GT speed | Bicubic speed | CNN speed | GAN speed |
    |Bicubic-GT| | |CNN-GT| | |GAN-GT|

Colour scales are shared within each group per sample:
  - Speed columns (GT / Bicubic / CNN / GAN): same vmin / vmax
  - Error columns (|Bicubic-GT| / |CNN-GT| / |GAN-GT|): same vmin=0 / vmax

If ttk_runs_fixed/baseline_metrics/all_methods_per_sample.csv is present,
each sample card in index.html also shows a 15-row physics / domain breakdown
table (Measure | Bicubic | CNN | GAN | Winner), with the winner determined
by lower-is-better for all measures.

Inputs (all under <repo>/data_out_fixed/):
    wind_mrhr_cnn/{idx,dataIN,dataGT,dataSR}.npy
    wind_mrhr_gan/{idx,dataIN,dataGT,dataSR}.npy
    wind_mrhr_bicubic/{idx,dataIN,dataGT,dataSR}.npy

Optional:
    ttk_runs_fixed/baseline_metrics/all_methods_per_sample.csv

Outputs:
    ttk_runs_fixed/baseline_visual_panels/
        panel_s<idx>.png   one PNG per selected sample
        index.html         browsable index of all panels

Run from scripts/ or the repo root:
    cd ~/PhIRE/scripts
    PYTHONNOUSERSITE=1 /usr/bin/python3 generate_baseline_visual_panels.py

    cd ~/PhIRE
    PYTHONNOUSERSITE=1 /usr/bin/python3 scripts/generate_baseline_visual_panels.py
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

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
except ImportError as exc:
    raise SystemExit(f"matplotlib required: {exc}") from exc


# ---------------------------------------------------------------------------
# Repo layout
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parent

DATA_ROOT = REPO_ROOT / "data_out_fixed"
CNN_DIR   = DATA_ROOT / "wind_mrhr_cnn"
GAN_DIR   = DATA_ROOT / "wind_mrhr_gan"
BIC_DIR   = DATA_ROOT / "wind_mrhr_bicubic"

OUT_DIR      = REPO_ROOT / "ttk_runs_fixed" / "baseline_visual_panels"
METRICS_CSV  = REPO_ROOT / "ttk_runs_fixed" / "baseline_metrics" / "all_methods_per_sample.csv"

DEFAULT_SAMPLES: List[int] = [
    10, 11, 12, 13, 17, 19, 25,
    76, 77, 78, 80,
    90, 91, 92, 93,
    154, 162, 163,
]


# ---------------------------------------------------------------------------
# Physics metric table definitions
# ---------------------------------------------------------------------------

# (display label, CSV column, needs_abs)
# needs_abs=True  → value is signed in the CSV; we take abs before comparing
# needs_abs=False → CSV already stores the absolute / unsigned version
_PHYSICS_MEASURES: List[Tuple[str, str, bool]] = [
    ("WPD bias |·|",          "wpd_bias",                   True),
    ("WPD MAE",               "wpd_mae",                    False),
    ("WPD RMSE",              "wpd_rmse",                   False),
    ("WPD Wasserstein-1",     "wpd_w1",                     False),
    ("PSD log-L2",            "psd_log_l2",                 False),
    ("PSD slope |Δ|",         "psd_slope_abs_delta",        False),
    ("Gradient MAE",          "grad_mae",                   False),
    ("Gradient Wasserstein-1","grad_w1",                    False),
    ("Gradient kurtosis |Δ|", "grad_kurtosis_abs_delta",    False),
    ("Exceedance |Δ|, s > 5", "exceed_frac_abs_delta_t5",   False),
    ("Exceedance |Δ|, s > 10","exceed_frac_abs_delta_t10",  False),
    ("Exceedance |Δ|, s > 15","exceed_frac_abs_delta_t15",  False),
    ("Exceedance |Δ|, p90",   "exceed_frac_abs_delta_p90",  False),
    ("Exceedance |Δ|, p95",   "exceed_frac_abs_delta_p95",  False),
    ("Exceedance |Δ|, p99",   "exceed_frac_abs_delta_p99",  False),
]

# Canonical method names as they appear in the Winner column.
# The CSV loader normalises whatever strings it finds to these keys.
_METHODS = ("bicubic", "cnn", "gan")
_METHOD_LABELS = {"bicubic": "Bicubic", "cnn": "CNN", "gan": "GAN"}

# PhysicsData maps (normalised_method, sample_idx) → {col: raw_string}
PhysicsData = Dict[Tuple[str, int], Dict[str, str]]


def _normalise_method(raw: str) -> Optional[str]:
    """Map raw method string from CSV to one of ('bicubic', 'cnn', 'gan')."""
    s = raw.strip().lower()
    # accept prefixed names like "wind_mrhr_cnn" or "mrhr_bicubic"
    for key in _METHODS:
        if key in s:
            return key
    return None


def _load_physics_csv(path: Path) -> Optional[PhysicsData]:
    """Load all_methods_per_sample.csv into a (method, sample_idx) → row dict.

    Returns None (with a printed warning) if the file is missing or malformed.
    Missing method / sample_idx columns are also reported and None returned.
    """
    if not path.exists():
        print(f"  [info] Physics CSV not found — skipping table: {path}")
        return None
    try:
        with path.open(newline="") as f:
            rows = list(csv.DictReader(f))
    except Exception as exc:
        print(f"  [warn] Could not read physics CSV ({exc}) — skipping table.")
        return None

    if not rows:
        print("  [warn] Physics CSV is empty — skipping table.")
        return None

    # Detect method and sample_idx columns
    method_col  = next((c for c in rows[0] if c.strip().lower() in ("method", "model")), None)
    sample_col  = next(
        (c for c in rows[0] if c.strip().lower() in ("sample_idx", "sample", "idx", "sidx", "index")),
        None,
    )
    if method_col is None or sample_col is None:
        print(
            f"  [warn] Physics CSV missing 'method' or 'sample_idx' column "
            f"(found: {list(rows[0].keys())[:10]}) — skipping table."
        )
        return None

    data: PhysicsData = {}
    skipped = 0
    for r in rows:
        meth = _normalise_method(r.get(method_col, ""))
        try:
            si = int(float(r.get(sample_col, "nan")))
        except (ValueError, TypeError):
            skipped += 1
            continue
        if meth is None:
            skipped += 1
            continue
        data[(meth, si)] = r

    if skipped:
        print(f"  [info] Physics CSV: skipped {skipped} unrecognised rows.")
    print(
        f"  Physics CSV loaded — {len(data)} (method, sample) entries "
        f"from {path.name}"
    )
    return data


def _fval(row: Dict[str, str], col: str, needs_abs: bool) -> Optional[float]:
    """Extract a float from a CSV row, optionally taking abs; returns None on NaN/missing."""
    raw = row.get(col)
    if raw is None:
        # try the abs_delta variant automatically
        alt = col.replace("_delta", "_abs_delta") if "_delta" in col else None
        if alt:
            raw = row.get(alt)
    if raw is None:
        return None
    try:
        v = float(raw)
        if math.isnan(v) or math.isinf(v):
            return None
        return abs(v) if needs_abs else v
    except (ValueError, TypeError):
        return None


def _physics_table_html(si: int, physics: PhysicsData) -> str:
    """Return an HTML table fragment for the physics breakdown of sample si.

    Returns an empty string if none of the three methods have data for si.
    """
    rows_by_method = {m: physics.get((m, si)) for m in _METHODS}
    if all(r is None for r in rows_by_method.values()):
        return (
            '<p style="color:#999;font-size:0.85em;margin:0.5em 0">'
            "Physics metrics not available for this sample.</p>"
        )

    # Build table rows
    tbody_rows: List[str] = []
    for label, col, needs_abs in _PHYSICS_MEASURES:
        vals: Dict[str, Optional[float]] = {
            m: _fval(rows_by_method[m], col, needs_abs) if rows_by_method[m] else None
            for m in _METHODS
        }
        # Determine winner (lowest non-None value)
        finite = {m: v for m, v in vals.items() if v is not None}
        winner: Optional[str] = min(finite, key=lambda m: finite[m]) if finite else None

        cells: List[str] = [f'<td class="m-label">{html.escape(label)}</td>']
        for m in _METHODS:
            v = vals[m]
            is_win = (m == winner) and (winner is not None)
            cls = ' class="winner"' if is_win else ""
            text = f"{v:.5g}" if v is not None else "—"
            cells.append(f"<td{cls}>{html.escape(text)}</td>")
        # Winner label cell
        if winner is not None:
            w_label = html.escape(_METHOD_LABELS[winner])
            cells.append(f'<td class="winner-label">{w_label}</td>')
        else:
            cells.append('<td class="winner-label">—</td>')

        tbody_rows.append("<tr>" + "".join(cells) + "</tr>")

    tbody = "\n      ".join(tbody_rows)
    return f"""<table class="phys-table">
  <thead>
    <tr>
      <th>Measure</th>
      <th>Bicubic</th><th>CNN</th><th>GAN</th>
      <th>Winner</th>
    </tr>
  </thead>
  <tbody>
      {tbody}
  </tbody>
</table>"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _speed(uv: np.ndarray) -> np.ndarray:
    """Compute wind speed magnitude from (H, W, 2) [u, v] array."""
    return np.sqrt(uv[..., 0] ** 2 + uv[..., 1] ** 2).astype(np.float32)


def _load_arrays(directory: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load (idx, dataIN, dataGT, dataSR) from a method directory with mmap."""
    for name in ("idx.npy", "dataGT.npy", "dataSR.npy"):
        if not (directory / name).exists():
            raise SystemExit(
                f"[error] Missing {directory / name}\n"
                f"  Ensure {directory.name} outputs exist before running this script."
            )
    idx    = np.load(directory / "idx.npy")
    data_in = np.load(directory / "dataIN.npy", mmap_mode="r") if (directory / "dataIN.npy").exists() else None
    data_gt = np.load(directory / "dataGT.npy", mmap_mode="r")
    data_sr = np.load(directory / "dataSR.npy", mmap_mode="r")
    return idx, data_in, data_gt, data_sr


def _verify_alignment(
    cnn_idx: np.ndarray, gan_idx: np.ndarray, bic_idx: np.ndarray,
    cnn_gt:  np.ndarray, gan_gt:  np.ndarray, bic_gt:  np.ndarray,
) -> None:
    """Abort with a clear message if idx or GT arrays disagree across methods."""
    print("[verify] Checking idx alignment …")
    if not np.array_equal(cnn_idx, gan_idx):
        raise SystemExit("[error] CNN and GAN idx arrays differ — cannot align samples.")
    if not np.array_equal(cnn_idx, bic_idx):
        raise SystemExit("[error] CNN and Bicubic idx arrays differ — cannot align samples.")
    print(f"  idx OK  — shape={cnn_idx.shape}, range=[{cnn_idx.min()}..{cnn_idx.max()}]")

    print("[verify] Checking GT array identity …")
    max_cnn_gan = float(np.max(np.abs(cnn_gt[:] - gan_gt[:])))
    max_cnn_bic = float(np.max(np.abs(cnn_gt[:] - bic_gt[:])))
    if max_cnn_gan > 1e-9:
        print(f"  [warn] CNN GT vs GAN GT max_abs_diff={max_cnn_gan:.6e}  (expected 0)")
    else:
        print(f"  CNN GT == GAN GT  ✓  (max_abs_diff={max_cnn_gan:.2e})")
    if max_cnn_bic > 1e-9:
        print(f"  [warn] CNN GT vs Bicubic GT max_abs_diff={max_cnn_bic:.6e}  (expected 0)")
    else:
        print(f"  CNN GT == Bicubic GT  ✓  (max_abs_diff={max_cnn_bic:.2e})")


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a.astype(float) - b.astype(float)) ** 2)))


def _panel_filename(si: int) -> str:
    return f"panel_s{si:03d}.png"


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_panel(
    out_png: Path,
    si: int,
    gt_spd:  np.ndarray,
    bic_spd: np.ndarray,
    cnn_spd: np.ndarray,
    gan_spd: np.ndarray,
) -> None:
    """Render a 1×7 panel and write to out_png.

    Colour scales:
      - Columns 0–3 (speed):   shared vmin/vmax across GT/Bicubic/CNN/GAN
      - Columns 4–6 (|error|): shared vmin=0, vmax=max of the three error maps
    """
    # ---- speed scale -------------------------------------------------------
    spd_vmin = float(min(gt_spd.min(),  bic_spd.min(), cnn_spd.min(), gan_spd.min()))
    spd_vmax = float(max(gt_spd.max(),  bic_spd.max(), cnn_spd.max(), gan_spd.max()))

    # ---- error maps --------------------------------------------------------
    err_bic = np.abs(bic_spd - gt_spd)
    err_cnn = np.abs(cnn_spd - gt_spd)
    err_gan = np.abs(gan_spd - gt_spd)
    err_vmax = float(max(err_bic.max(), err_cnn.max(), err_gan.max()))
    err_vmin = 0.0

    # ---- RMSE for title ----------------------------------------------------
    rmse_bic = _rmse(bic_spd, gt_spd)
    rmse_cnn = _rmse(cnn_spd, gt_spd)
    rmse_gan = _rmse(gan_spd, gt_spd)

    # ---- layout ------------------------------------------------------------
    fig, axes = plt.subplots(1, 7, figsize=(24, 4))

    speed_norm = mcolors.Normalize(vmin=spd_vmin, vmax=spd_vmax)
    error_norm = mcolors.Normalize(vmin=err_vmin, vmax=err_vmax)

    panels = [
        (gt_spd,  "GT speed",      "viridis",  speed_norm),
        (bic_spd, "Bicubic speed", "viridis",  speed_norm),
        (cnn_spd, "CNN SR speed",  "viridis",  speed_norm),
        (gan_spd, "GAN SR speed",  "viridis",  speed_norm),
        (err_bic, "|Bicubic − GT|","magma",    error_norm),
        (err_cnn, "|CNN − GT|",    "magma",    error_norm),
        (err_gan, "|GAN − GT|",    "magma",    error_norm),
    ]

    for ax, (img, title, cmap, norm) in zip(axes, panels):
        im = ax.imshow(img, cmap=cmap, norm=norm, interpolation="nearest", origin="upper")
        ax.set_title(title, fontsize=9, pad=3)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"Sample {si}  |  "
        f"RMSE (speed m/s):  Bicubic={rmse_bic:.3f}  CNN={rmse_cnn:.3f}  GAN={rmse_gan:.3f}  |  "
        f"Speed scale [{spd_vmin:.1f}, {spd_vmax:.1f}] m/s  |  "
        f"Error scale [0, {err_vmax:.2f}] m/s",
        fontsize=9,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# HTML index
# ---------------------------------------------------------------------------

def _write_index(
    out_dir: Path,
    sample_indices: List[int],
    physics: Optional[PhysicsData],
) -> Path:
    """Write index.html with one white card per sample.

    Each card contains the panel image and, if physics data is available,
    a 15-row physics/domain breakdown table.
    """
    cards: List[str] = []
    for si in sample_indices:
        fname   = _panel_filename(si)
        fpath   = out_dir / fname
        missing = "" if fpath.exists() else " <span class='warn'>⚠ image missing</span>"
        esc_fname = html.escape(fname)

        # physics table (empty string when not available)
        phys_html = _physics_table_html(si, physics) if physics is not None else (
            '<p class="phys-missing">Physics CSV not loaded — '
            'run baseline_metrics pipeline first.</p>'
        )

        cards.append(f"""  <div class="card">
    <h2>Sample {si}{missing}</h2>
    <a href="{esc_fname}">
      <img src="{esc_fname}" alt="sample {si} comparison panel" loading="lazy">
    </a>
    <div class="phys-section">
      <h3>Physics / domain breakdown</h3>
      {phys_html}
    </div>
  </div>""")

    cards_html = "\n".join(cards)

    has_phys = physics is not None
    phys_note = (
        "Physics breakdown is shown below each panel "
        "(lower&nbsp;=&nbsp;better;&nbsp;winner&nbsp;highlighted&nbsp;in&nbsp;green)."
        if has_phys else
        "Physics CSV not found — run the baseline_metrics pipeline to add the table."
    )

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Baseline visual panels — Bicubic / CNN / GAN</title>
<style>
  /* ---- page ---- */
  *, *::before, *::after {{ box-sizing: border-box; }}
  body {{
    font-family: system-ui, sans-serif;
    background: #f0f2f5;
    color: #1a1a1a;
    margin: 0;
    padding: 1.5em;
  }}
  h1 {{ font-size: 1.25em; margin: 0 0 0.35em; }}
  .intro {{ color: #555; font-size: 0.9em; margin: 0 0 1.5em; line-height: 1.5; }}

  /* ---- white cards ---- */
  .card {{
    background: #fff;
    border: 1px solid #dde1e7;
    border-radius: 8px;
    box-shadow: 0 1px 4px rgba(0,0,0,.07);
    margin-bottom: 2em;
    padding: 1.25em 1.5em;
  }}
  .card h2 {{
    font-size: 1em;
    margin: 0 0 0.75em;
    color: #222;
  }}
  .warn {{ color: #b45309; }}
  .card img {{
    display: block;
    max-width: 100%;
    border-radius: 4px;
    margin-bottom: 1em;
  }}

  /* ---- physics section ---- */
  .phys-section h3 {{
    font-size: 0.85em;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #666;
    margin: 0 0 0.5em;
  }}
  .phys-missing {{ color: #999; font-size: 0.85em; margin: 0; }}

  /* ---- physics table ---- */
  .phys-table {{
    border-collapse: collapse;
    font-size: 0.82em;
    width: 100%;
    margin: 0;
  }}
  .phys-table th {{
    background: #f7f8fa;
    color: #444;
    font-weight: 600;
    text-align: left;
    padding: 5px 10px;
    border: 1px solid #e2e5ea;
  }}
  .phys-table td {{
    padding: 4px 10px;
    border: 1px solid #e2e5ea;
    color: #222;
    font-variant-numeric: tabular-nums;
    white-space: nowrap;
  }}
  .phys-table td.m-label {{ color: #444; font-weight: 500; }}
  .phys-table tr:nth-child(even) td {{ background: #fafbfc; }}
  .phys-table td.winner {{
    background: #d1fae5;
    color: #065f46;
    font-weight: 700;
  }}
  .phys-table td.winner-label {{
    background: #ecfdf5;
    color: #047857;
    font-weight: 600;
  }}
</style>
</head>
<body>
<h1>Baseline visual panels: GT | Bicubic | CNN SR | GAN SR | errors (speed m/s)</h1>
<p class="intro">
  <strong>Panel columns:</strong>
  GT speed &nbsp;|&nbsp; Bicubic speed &nbsp;|&nbsp; CNN speed &nbsp;|&nbsp; GAN speed &nbsp;|&nbsp;
  |Bicubic&minus;GT| &nbsp;|&nbsp; |CNN&minus;GT| &nbsp;|&nbsp; |GAN&minus;GT|<br>
  Speed columns share one colour scale per sample &mdash;
  error columns share one colour scale per sample.<br>
  {phys_note}
</p>
{cards_html}
</body>
</html>
"""
    idx_path = out_dir / "index.html"
    idx_path.write_text(page, encoding="utf-8")
    return idx_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cnn-dir",     type=Path, default=CNN_DIR)
    ap.add_argument("--gan-dir",     type=Path, default=GAN_DIR)
    ap.add_argument("--bicubic-dir", type=Path, default=BIC_DIR)
    ap.add_argument("--outdir",      type=Path, default=OUT_DIR)
    ap.add_argument("--metrics-csv", type=Path, default=METRICS_CSV,
                    help="Path to all_methods_per_sample.csv (omit to skip physics table)")
    ap.add_argument(
        "--samples", nargs="*", type=int, default=None,
        help="Sample indices to render (default: hard-coded 18-sample set)",
    )
    ap.add_argument("--dpi", type=int, default=150, help="Output DPI (default 150)")
    return ap.parse_args()


def main() -> int:
    args = _parse_args()

    samples: List[int] = sorted(set(args.samples if args.samples is not None else DEFAULT_SAMPLES))

    print("=" * 65)
    print("PhIRE baseline visual panel generator")
    print(f"  CNN dir     : {args.cnn_dir}")
    print(f"  GAN dir     : {args.gan_dir}")
    print(f"  Bicubic dir : {args.bicubic_dir}")
    print(f"  Metrics CSV : {args.metrics_csv}")
    print(f"  Output dir  : {args.outdir}")
    print(f"  Samples     : {samples}")
    print("=" * 65)

    # ---- load physics CSV ---------------------------------------------------
    print("\n[1] Loading physics CSV …")
    physics = _load_physics_csv(args.metrics_csv)

    # ---- load arrays --------------------------------------------------------
    print("\n[2] Loading arrays …")
    cnn_idx, cnn_in, cnn_gt, cnn_sr = _load_arrays(args.cnn_dir)
    gan_idx, gan_in, gan_gt, gan_sr = _load_arrays(args.gan_dir)
    bic_idx, bic_in, bic_gt, bic_sr = _load_arrays(args.bicubic_dir)

    print(f"  CNN  GT={cnn_gt.shape} SR={cnn_sr.shape} dtype={cnn_gt.dtype}")
    print(f"  GAN  GT={gan_gt.shape} SR={gan_sr.shape} dtype={gan_gt.dtype}")
    print(f"  Bic  GT={bic_gt.shape} SR={bic_sr.shape} dtype={bic_gt.dtype}")

    # ---- verify -------------------------------------------------------------
    print("\n[3] Verifying alignment …")
    _verify_alignment(cnn_idx, gan_idx, bic_idx, cnn_gt, gan_gt, bic_gt)

    # ---- check sample bounds ------------------------------------------------
    n = cnn_gt.shape[0]
    bad = [si for si in samples if si < 0 or si >= n]
    if bad:
        raise SystemExit(f"[error] Sample indices out of range [0, {n-1}]: {bad}")

    # ---- render panels ------------------------------------------------------
    print(f"\n[4] Rendering {len(samples)} panels …")
    generated: List[int] = []
    for si in samples:
        print(f"  sample {si} …", end=" ", flush=True)

        gt_spd  = _speed(np.asarray(cnn_gt[si]))
        bic_spd = _speed(np.asarray(bic_sr[si]))
        cnn_spd = _speed(np.asarray(cnn_sr[si]))
        gan_spd = _speed(np.asarray(gan_sr[si]))

        out_png = args.outdir / _panel_filename(si)
        _plot_panel(out_png, si, gt_spd, bic_spd, cnn_spd, gan_spd)
        generated.append(si)
        print("done")

    # ---- HTML index ---------------------------------------------------------
    print("\n[5] Writing index.html …")
    idx_path = _write_index(args.outdir, samples, physics)
    print(f"  {idx_path}")

    # ---- summary ------------------------------------------------------------
    print(f"\n[done] {len(generated)} panels written to {args.outdir}")
    print(f"  Open: {idx_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
