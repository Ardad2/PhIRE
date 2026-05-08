#!/usr/bin/env python3
"""Generate 7-column visual comparison panels for CNN, GAN, and bicubic outputs.

Each panel shows one sample as a single row of seven subplots:
    GT speed | Bicubic speed | CNN speed | GAN speed |
    |Bicubic-GT| | |CNN-GT| | |GAN-GT|

Colour scales are shared within each group per sample:
  - Speed columns (GT / Bicubic / CNN / GAN): same vmin / vmax
  - Error columns (|Bicubic-GT| / |CNN-GT| / |GAN-GT|): same vmin=0 / vmax

Inputs (all under <repo>/data_out_fixed/):
    wind_mrhr_cnn/{idx,dataIN,dataGT,dataSR}.npy
    wind_mrhr_gan/{idx,dataIN,dataGT,dataSR}.npy
    wind_mrhr_bicubic/{idx,dataIN,dataGT,dataSR}.npy

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
import html
import sys
from pathlib import Path
from typing import List, Tuple

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

OUT_DIR   = REPO_ROOT / "ttk_runs_fixed" / "baseline_visual_panels"

DEFAULT_SAMPLES: List[int] = [
    10, 11, 12, 13, 17, 19, 25,
    76, 77, 78, 80,
    90, 91, 92, 93,
    154, 162, 163,
]


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

def _write_index(out_dir: Path, sample_indices: List[int]) -> Path:
    """Write a minimal index.html linking all generated panel PNGs."""
    rows_html = []
    for si in sample_indices:
        fname = _panel_filename(si)
        fpath = out_dir / fname
        exists_mark = "" if fpath.exists() else " ⚠ missing"
        escaped = html.escape(fname)
        rows_html.append(
            f'  <div class="panel">\n'
            f'    <p><strong>Sample {si}</strong>{html.escape(exists_mark)}</p>\n'
            f'    <a href="{escaped}">'
            f'<img src="{escaped}" alt="sample {si}" width="1200"></a>\n'
            f'  </div>'
        )

    content = "\n".join(rows_html)
    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Baseline visual panels — CNN / GAN / Bicubic</title>
<style>
  body {{ font-family: sans-serif; background: #111; color: #eee; }}
  h1   {{ font-size: 1.2em; margin: 1em; }}
  .panel {{ margin: 1.5em 1em; border-bottom: 1px solid #333; padding-bottom: 1em; }}
  .panel p {{ margin: 0 0 0.4em; }}
  img {{ max-width: 100%; display: block; }}
</style>
</head>
<body>
<h1>Baseline visual panels: GT | Bicubic | CNN SR | GAN SR | errors (speed m/s)</h1>
<p style="margin:0 1em 1em">
  Columns: GT speed | Bicubic speed | CNN speed | GAN speed |
  |Bicubic−GT| | |CNN−GT| | |GAN−GT|<br>
  Speed columns share one colour scale per sample.
  Error columns share one colour scale per sample.
</p>
{content}
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
    print(f"  Output dir  : {args.outdir}")
    print(f"  Samples     : {samples}")
    print("=" * 65)

    # ---- load ---------------------------------------------------------------
    print("\n[1] Loading arrays …")
    cnn_idx, cnn_in, cnn_gt, cnn_sr = _load_arrays(args.cnn_dir)
    gan_idx, gan_in, gan_gt, gan_sr = _load_arrays(args.gan_dir)
    bic_idx, bic_in, bic_gt, bic_sr = _load_arrays(args.bicubic_dir)

    print(f"  CNN  GT={cnn_gt.shape} SR={cnn_sr.shape} dtype={cnn_gt.dtype}")
    print(f"  GAN  GT={gan_gt.shape} SR={gan_sr.shape} dtype={gan_gt.dtype}")
    print(f"  Bic  GT={bic_gt.shape} SR={bic_sr.shape} dtype={bic_gt.dtype}")

    # ---- verify -------------------------------------------------------------
    print("\n[2] Verifying alignment …")
    _verify_alignment(cnn_idx, gan_idx, bic_idx, cnn_gt, gan_gt, bic_gt)

    # ---- check sample bounds ------------------------------------------------
    n = cnn_gt.shape[0]
    bad = [si for si in samples if si < 0 or si >= n]
    if bad:
        raise SystemExit(f"[error] Sample indices out of range [0, {n-1}]: {bad}")

    # ---- render panels ------------------------------------------------------
    print(f"\n[3] Rendering {len(samples)} panels …")
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
    print("\n[4] Writing index.html …")
    idx_path = _write_index(args.outdir, samples)
    print(f"  {idx_path}")

    # ---- summary ------------------------------------------------------------
    print(f"\n[done] {len(generated)} panels written to {args.outdir}")
    print(f"  Open: {idx_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
