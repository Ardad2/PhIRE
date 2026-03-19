#!/usr/bin/env python3
"""Reproducible qualitative/quantitative comparison for selected wind SR samples.

Compares:
- original vector CNN paired inference
- original vector GAN paired inference
- optional direct scalar-speed CNN paired inference

Outputs per-sample figures, a metrics CSV, and a markdown summary.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as e:  # pragma: no cover
    raise SystemExit(f"matplotlib is required: {e}")

try:  # optional
    from skimage.metrics import structural_similarity as ssim
except Exception:  # pragma: no cover
    ssim = None

# Repo-observed showcase/default sample ids:
# - quick_figs.py renders sample 0
# - scripts/run_full_experiment.sh highlights 2, 165, 166 in tie-break panels
REPO_SHOWCASE_SAMPLES = (0, 2, 165, 166)


@dataclass
class RunData:
    label: str
    path: Path
    idx: np.ndarray
    lr: np.ndarray
    gt: np.ndarray
    sr: np.ndarray
    is_scalar: bool


@dataclass
class SampleViews:
    sample_id: int
    row: int
    lr_speed: np.ndarray
    gt_speed: np.ndarray
    cnn_speed: np.ndarray
    gan_speed: np.ndarray
    scalar_speed: Optional[np.ndarray]
    u_gt: np.ndarray
    u_cnn: np.ndarray
    u_gan: np.ndarray
    v_gt: np.ndarray
    v_cnn: np.ndarray
    v_gan: np.ndarray
    lr_u: np.ndarray
    lr_v: np.ndarray


def _to_uv_last(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr)
    if a.ndim == 4 and a.shape[-1] == 2:
        return a
    if a.ndim == 4 and a.shape[1] == 2:
        return np.transpose(a, (0, 2, 3, 1))
    raise ValueError(f"Expected vector array with shape (N,H,W,2) or (N,2,H,W); got {a.shape}")


def _to_scalar_last(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr)
    if a.ndim == 3:
        return a[..., None]
    if a.ndim == 4 and a.shape[-1] == 1:
        return a
    if a.ndim == 4 and a.shape[1] == 1:
        return np.transpose(a, (0, 2, 3, 1))
    raise ValueError(f"Expected scalar array with shape (N,H,W,1), (N,1,H,W), or (N,H,W); got {a.shape}")


def _speed(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    if a.ndim == 3 and a.shape[-1] >= 2:
        return np.sqrt(np.square(a[..., 0]) + np.square(a[..., 1]))
    if a.ndim == 3 and a.shape[-1] == 1:
        return a[..., 0]
    if a.ndim == 2:
        return a
    raise ValueError(f"Cannot compute speed from shape {a.shape}")


def _crop2d(a: np.ndarray, patch: int, x0: int, y0: int) -> np.ndarray:
    if patch <= 0:
        return a
    h, w = a.shape
    if not (0 <= x0 < w and 0 <= y0 < h):
        raise ValueError(f"Invalid crop origin ({x0}, {y0}) for shape {(h, w)}")
    return a[y0:min(h, y0 + patch), x0:min(w, x0 + patch)]


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return float(np.mean(d * d))


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(math.sqrt(_mse(a, b)))


def _mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))))


def _psnr(pred: np.ndarray, gt: np.ndarray) -> float:
    mse = _mse(pred, gt)
    if mse == 0.0:
        return float("inf")
    dr = float(np.max(gt) - np.min(gt))
    if dr <= 0.0:
        dr = 1.0
    return float(20.0 * math.log10(dr) - 10.0 * math.log10(mse))


def _ssim(pred: np.ndarray, gt: np.ndarray) -> Optional[float]:
    if ssim is None:
        return None
    gt = np.asarray(gt, dtype=float)
    pred = np.asarray(pred, dtype=float)
    dr = float(np.max(gt) - np.min(gt))
    if dr <= 0.0:
        dr = 1.0
    return float(ssim(gt, pred, data_range=dr))


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _load_run(path: Path, label: str, is_scalar: bool) -> RunData:
    idx = np.load(path / "idx.npy")
    lr = np.load(path / "dataIN.npy")
    gt = np.load(path / "dataGT.npy")
    sr = np.load(path / "dataSR.npy")
    if is_scalar:
        lr = _to_scalar_last(lr)
        gt = _to_scalar_last(gt)
        sr = _to_scalar_last(sr)
    else:
        lr = _to_uv_last(lr)
        gt = _to_uv_last(gt)
        sr = _to_uv_last(sr)
    return RunData(label=label, path=path, idx=np.asarray(idx), lr=lr, gt=gt, sr=sr, is_scalar=is_scalar)


def _index_map(idx: np.ndarray) -> Dict[int, int]:
    out: Dict[int, int] = {}
    for i, v in enumerate(np.asarray(idx).tolist()):
        out[int(v)] = i
    return out


def _validate_alignment(cnn: RunData, gan: RunData, scalar: Optional[RunData]) -> None:
    if cnn.idx.shape != gan.idx.shape or not np.array_equal(cnn.idx, gan.idx):
        raise SystemExit("Vector CNN/GAN idx mismatch; cannot compare per-sample outputs fairly")
    if cnn.gt.shape != gan.gt.shape:
        raise SystemExit(f"Vector CNN/GAN GT shape mismatch: {cnn.gt.shape} vs {gan.gt.shape}")
    if scalar is not None:
        if scalar.idx.shape != cnn.idx.shape or not np.array_equal(scalar.idx, cnn.idx):
            raise SystemExit("Scalar idx mismatch against vector runs; sample IDs no longer align")
        if scalar.gt.shape[0] != cnn.gt.shape[0]:
            raise SystemExit("Scalar and vector runs have different numbers of samples")


def _selected_ids(cnn: RunData, requested: Sequence[int], include_showcase: bool) -> List[int]:
    ids = list(requested)
    if include_showcase:
        ids.extend(REPO_SHOWCASE_SAMPLES)
    seen = set()
    out = []
    valid = set(int(x) for x in cnn.idx.tolist())
    for sid in ids:
        sid = int(sid)
        if sid in seen:
            continue
        if sid in valid:
            seen.add(sid)
            out.append(sid)
    return out


def _sample_views(cnn: RunData, gan: RunData, scalar: Optional[RunData], sample_id: int, patch: int, x0: int, y0: int) -> SampleViews:
    row = _index_map(cnn.idx)[sample_id]

    lr_speed = _crop2d(_speed(cnn.lr[row]), patch=max(0, patch // 5) if patch > 0 else 0, x0=max(0, x0 // 5), y0=max(0, y0 // 5))
    gt_speed = _crop2d(_speed(cnn.gt[row]), patch, x0, y0)
    cnn_speed = _crop2d(_speed(cnn.sr[row]), patch, x0, y0)
    gan_speed = _crop2d(_speed(gan.sr[row]), patch, x0, y0)
    scalar_speed = _crop2d(_speed(scalar.sr[row]), patch, x0, y0) if scalar is not None else None

    u_gt = _crop2d(cnn.gt[row, ..., 0], patch, x0, y0)
    u_cnn = _crop2d(cnn.sr[row, ..., 0], patch, x0, y0)
    u_gan = _crop2d(gan.sr[row, ..., 0], patch, x0, y0)
    v_gt = _crop2d(cnn.gt[row, ..., 1], patch, x0, y0)
    v_cnn = _crop2d(cnn.sr[row, ..., 1], patch, x0, y0)
    v_gan = _crop2d(gan.sr[row, ..., 1], patch, x0, y0)
    lr_u = _crop2d(cnn.lr[row, ..., 0], patch=max(0, patch // 5) if patch > 0 else 0, x0=max(0, x0 // 5), y0=max(0, y0 // 5))
    lr_v = _crop2d(cnn.lr[row, ..., 1], patch=max(0, patch // 5) if patch > 0 else 0, x0=max(0, x0 // 5), y0=max(0, y0 // 5))
    return SampleViews(sample_id=sample_id, row=row, lr_speed=lr_speed, gt_speed=gt_speed, cnn_speed=cnn_speed, gan_speed=gan_speed, scalar_speed=scalar_speed, u_gt=u_gt, u_cnn=u_cnn, u_gan=u_gan, v_gt=v_gt, v_cnn=v_cnn, v_gan=v_gan, lr_u=lr_u, lr_v=lr_v)


def _panel_limits(images: Iterable[np.ndarray]) -> Tuple[float, float]:
    vals = [np.asarray(im, dtype=float) for im in images if im is not None]
    return float(min(np.min(v) for v in vals)), float(max(np.max(v) for v in vals))


def _save_speed_panel(out_png: Path, views: SampleViews) -> None:
    errors = [np.abs(views.cnn_speed - views.gt_speed), np.abs(views.gan_speed - views.gt_speed)]
    titles = ["LR speed", "GT speed", "Vector CNN speed", "Vector GAN speed", "|CNN-GT|", "|GAN-GT|"]
    images: List[np.ndarray] = [views.lr_speed, views.gt_speed, views.cnn_speed, views.gan_speed, errors[0], errors[1]]
    if views.scalar_speed is not None:
        titles.extend(["Scalar CNN speed", "|Scalar-GT|"])
        images.extend([views.scalar_speed, np.abs(views.scalar_speed - views.gt_speed)])
    n = len(images)
    ncols = 4
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 4.2 * nrows), squeeze=False)
    speed_min, speed_max = _panel_limits([views.gt_speed, views.cnn_speed, views.gan_speed, views.scalar_speed, views.lr_speed])
    err_max = float(max(np.max(img) for img in images if img is not None and img.shape == views.gt_speed.shape and np.min(img) >= 0.0))

    for ax in axes.ravel():
        ax.axis("off")
    for ax, title, img in zip(axes.ravel(), titles, images):
        cmap = "magma" if title.startswith("|") else "viridis"
        if title.startswith("|"):
            vmin, vmax = 0.0, err_max
        else:
            vmin, vmax = speed_min, speed_max
        im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
        ax.set_title(f"s{views.sample_id}: {title}")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.axis("on")

    fig.suptitle(f"Speed comparison for sample {views.sample_id} (row {views.row})", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_velocity_panel(out_png: Path, sample_id: int, component_name: str, gt: np.ndarray, cnn: np.ndarray, gan: np.ndarray, lr: np.ndarray) -> None:
    err_cnn = np.abs(cnn - gt)
    err_gan = np.abs(gan - gt)
    value_min, value_max = _panel_limits([gt, cnn, gan, lr])
    err_max = float(max(np.max(err_cnn), np.max(err_gan)))

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.5), squeeze=False)
    top = [(lr, f"LR {component_name}"), (gt, f"GT {component_name}"), (cnn, f"CNN SR {component_name}")]
    bot = [(gan, f"GAN SR {component_name}"), (err_cnn, f"|CNN-GT| {component_name}"), (err_gan, f"|GAN-GT| {component_name}")]
    for row_axes, row_items in zip(axes, [top, bot]):
        for ax, (img, title) in zip(row_axes, row_items):
            is_err = title.startswith("|")
            im = ax.imshow(img, cmap="magma" if is_err else "coolwarm", vmin=(0.0 if is_err else value_min), vmax=(err_max if is_err else value_max), origin="lower")
            ax.set_title(f"s{sample_id}: {title}")
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"Velocity component {component_name} for sample {sample_id}", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _metrics_row(views: SampleViews) -> Dict[str, object]:
    row: Dict[str, object] = {"sample_id": views.sample_id, "row": views.row}
    methods = {"vector_cnn": views.cnn_speed, "vector_gan": views.gan_speed}
    if views.scalar_speed is not None:
        methods["scalar_cnn"] = views.scalar_speed
    for name, pred in methods.items():
        row[f"{name}_speed_mae"] = _mae(pred, views.gt_speed)
        row[f"{name}_speed_rmse"] = _rmse(pred, views.gt_speed)
        row[f"{name}_speed_psnr"] = _psnr(pred, views.gt_speed)
        ssim_val = _ssim(pred, views.gt_speed)
        row[f"{name}_speed_ssim"] = "" if ssim_val is None else ssim_val
        row[f"{name}_speed_min"] = float(np.min(pred))
        row[f"{name}_speed_max"] = float(np.max(pred))
    for prefix, pred_u, pred_v in [("vector_cnn", views.u_cnn, views.v_cnn), ("vector_gan", views.u_gan, views.v_gan)]:
        row[f"{prefix}_u_mae"] = _mae(pred_u, views.u_gt)
        row[f"{prefix}_u_rmse"] = _rmse(pred_u, views.u_gt)
        row[f"{prefix}_v_mae"] = _mae(pred_v, views.v_gt)
        row[f"{prefix}_v_rmse"] = _rmse(pred_v, views.v_gt)
    row["gt_speed_min"] = float(np.min(views.gt_speed))
    row["gt_speed_max"] = float(np.max(views.gt_speed))
    row["lr_speed_min"] = float(np.min(views.lr_speed))
    row["lr_speed_max"] = float(np.max(views.lr_speed))
    return row


def _summary_markdown(out_path: Path, rows: List[Dict[str, object]], selected: Sequence[int], include_showcase: bool, patch: int, x0: int, y0: int, scalar_present: bool) -> None:
    lines: List[str] = []
    lines.append("# Wind SR diagnostic summary\n")
    lines.append(f"Selected sample IDs: {', '.join(str(x) for x in selected)}.\n")
    if include_showcase:
        lines.append(f"Repo showcase/sample defaults included when present: {', '.join(str(x) for x in REPO_SHOWCASE_SAMPLES)}.\n")
    lines.append(f"Comparison crop: {'full frame' if patch <= 0 else f'patch={patch}, x0={x0}, y0={y0}'}.\n")
    lines.append(f"Direct scalar CNN available: {'yes' if scalar_present else 'no'}.\n")
    lines.append("\n## Per-sample speed metrics\n")
    headers = ["sample_id", "vector_cnn_speed_mae", "vector_gan_speed_mae"] + (["scalar_cnn_speed_mae"] if scalar_present else [])
    lines.append("| " + " | ".join(headers) + " |\n")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|\n")
    for r in rows:
        vals = []
        for h in headers:
            v = r.get(h, "")
            if isinstance(v, float):
                vals.append(f"{v:.4f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |\n")

    def _avg(col: str) -> Optional[float]:
        vals = [float(r[col]) for r in rows if col in r and str(r[col]) != ""]
        return None if not vals else float(np.mean(vals))

    lines.append("\n## Aggregate notes\n")
    for col in ["vector_cnn_speed_mae", "vector_gan_speed_mae"] + (["scalar_cnn_speed_mae"] if scalar_present else []):
        avg = _avg(col)
        if avg is not None:
            lines.append(f"- Mean {col}: {avg:.4f}\n")
    lines.append("- Use the per-sample speed and velocity figures in this directory to judge whether structural mismatch persists across methods and whether it is already present in the vector components.\n")
    out_path.write_text("".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare selected wind SR samples across vector CNN/GAN and scalar CNN runs")
    ap.add_argument("--cnn-dir", type=Path, default=Path("data_out/wind_mrhr_cnn"))
    ap.add_argument("--gan-dir", type=Path, default=Path("data_out/wind_mrhr_gan"))
    ap.add_argument("--scalar-dir", type=Path, default=None, help="Optional scalar-speed CNN paired output dir")
    ap.add_argument("--samples", nargs="+", type=int, required=True, help="Original sample IDs to compare")
    ap.add_argument("--include-repo-showcase", action="store_true", help="Also include repo-observed showcase/default sample IDs when present")
    ap.add_argument("--patch", type=int, default=0, help="Patch size in HR pixels; 0 means full frame")
    ap.add_argument("--x0", type=int, default=0)
    ap.add_argument("--y0", type=int, default=0)
    ap.add_argument("--outdir", type=Path, default=Path("analysis/wind_diagnostics"))
    args = ap.parse_args()

    cnn = _load_run(args.cnn_dir, "vector_cnn", is_scalar=False)
    gan = _load_run(args.gan_dir, "vector_gan", is_scalar=False)
    scalar = _load_run(args.scalar_dir, "scalar_cnn", is_scalar=True) if args.scalar_dir is not None else None
    _validate_alignment(cnn, gan, scalar)

    selected = _selected_ids(cnn, args.samples, args.include_repo_showcase)
    if not selected:
        raise SystemExit("None of the requested sample IDs were found in the provided idx.npy")

    outdir = args.outdir
    speed_dir = outdir / "speed_panels"
    vel_dir = outdir / "velocity_panels"
    rows: List[Dict[str, object]] = []
    found = []
    for sid in selected:
        views = _sample_views(cnn, gan, scalar, sid, args.patch, args.x0, args.y0)
        _save_speed_panel(speed_dir / f"sample_{sid}_speed.png", views)
        _save_velocity_panel(vel_dir / f"sample_{sid}_u.png", sid, "u", views.u_gt, views.u_cnn, views.u_gan, views.lr_u)
        _save_velocity_panel(vel_dir / f"sample_{sid}_v.png", sid, "v", views.v_gt, views.v_cnn, views.v_gan, views.lr_v)
        rows.append(_metrics_row(views))
        found.append(sid)

    rows.sort(key=lambda r: int(r["sample_id"]))
    _write_csv(outdir / "selected_sample_metrics.csv", rows)
    _summary_markdown(outdir / "summary.md", rows, found, args.include_repo_showcase, args.patch, args.x0, args.y0, scalar is not None)
    print(f"[OK] wrote figures/metrics to {outdir}")
    print(f"[OK] compared sample IDs: {found}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
