#!/usr/bin/env python3
"""Audit whether paired wind MR fields are valid low-resolution versions of paired HR fields.

This script reads a paired wind MR->HR TFRecord, downsamples each HR sample to MR
resolution using 5x5 averaging per channel, and compares the result against:
  1) the paired MR sample, and
  2) wrong-sample MR candidates.

It also runs a small orientation/layout sanity check on representative samples.
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
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"matplotlib is required: {exc}")


def _import_tensorflow():
    try:
        import tensorflow as tf  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "TensorFlow is required to parse TFRecords for this audit. "
            f"Import failed: {exc}"
        )
    return tf


@dataclass
class WindRecord:
    row: int
    index: int
    mr: np.ndarray  # [Hm, Wm, 2]
    hr: np.ndarray  # [Hh, Wh, 2]


@dataclass
class PairMetrics:
    mae_total: float
    rmse_total: float
    max_abs_diff: float
    mae_ua: float
    mae_va: float
    rmse_ua: float
    rmse_va: float


@dataclass
class RankingResult:
    paired_rank_rmse: int
    paired_rank_mae: int
    paired_rmse: float
    paired_mae: float
    best_wrong_rmse: float
    best_wrong_mae: float
    paired_beats_wrong_rmse_gap: float
    paired_beats_wrong_mae_gap: float
    candidate_count: int


TRANSFORMS: Tuple[str, ...] = (
    "identity",
    "transpose",
    "flip_ud",
    "flip_lr",
    "swap_channels",
    "swap_channels_transpose",
    "swap_channels_flip_ud",
    "swap_channels_flip_lr",
)


def _mean_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))))


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return float(np.sqrt(np.mean(d * d)))


def _pair_metrics(mr: np.ndarray, ds: np.ndarray) -> PairMetrics:
    diff = np.asarray(mr, dtype=np.float64) - np.asarray(ds, dtype=np.float64)
    abs_diff = np.abs(diff)
    return PairMetrics(
        mae_total=float(np.mean(abs_diff)),
        rmse_total=float(np.sqrt(np.mean(diff * diff))),
        max_abs_diff=float(np.max(abs_diff)),
        mae_ua=float(np.mean(abs_diff[..., 0])),
        mae_va=float(np.mean(abs_diff[..., 1])),
        rmse_ua=float(np.sqrt(np.mean(diff[..., 0] * diff[..., 0]))),
        rmse_va=float(np.sqrt(np.mean(diff[..., 1] * diff[..., 1]))),
    )


def _apply_transform(arr: np.ndarray, name: str) -> np.ndarray:
    out = np.asarray(arr)
    if name == "identity":
        return out
    if name == "transpose":
        return np.transpose(out, (1, 0, 2))
    if name == "flip_ud":
        return np.flipud(out)
    if name == "flip_lr":
        return np.fliplr(out)
    if name == "swap_channels":
        return out[..., [1, 0]]
    if name == "swap_channels_transpose":
        return np.transpose(out[..., [1, 0]], (1, 0, 2))
    if name == "swap_channels_flip_ud":
        return np.flipud(out[..., [1, 0]])
    if name == "swap_channels_flip_lr":
        return np.fliplr(out[..., [1, 0]])
    raise ValueError(f"Unknown transform: {name}")


def _downsample_mean_2d(a: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {a.shape}")
    kh = a.shape[0] // out_h
    kw = a.shape[1] // out_w
    if kh <= 0 or kw <= 0:
        raise ValueError(f"Invalid output size {(out_h, out_w)} for input {a.shape}")
    trim_h = out_h * kh
    trim_w = out_w * kw
    a = a[:trim_h, :trim_w]
    return a.reshape(out_h, kh, out_w, kw).mean(axis=(1, 3))


def downsample_hr_to_mr(hr: np.ndarray, mr_shape: Tuple[int, int, int]) -> np.ndarray:
    hm, wm, cm = mr_shape
    if hr.ndim != 3 or hr.shape[-1] != cm:
        raise ValueError(f"HR shape {hr.shape} incompatible with MR shape {mr_shape}")
    out = np.zeros((hm, wm, cm), dtype=np.float64)
    for c in range(cm):
        out[..., c] = _downsample_mean_2d(hr[..., c], hm, wm)
    return out


def iter_wind_records(path: Path) -> Iterable[WindRecord]:
    tf = _import_tensorflow()
    for row, raw in enumerate(tf.compat.v1.io.tf_record_iterator(str(path))):
        ex = tf.train.Example()
        ex.ParseFromString(raw)
        feat = ex.features.feature

        def _i(name: str) -> int:
            return int(feat[name].int64_list.value[0])

        def _b(name: str) -> bytes:
            return bytes(feat[name].bytes_list.value[0])

        index = _i("index")
        h_mr, w_mr = _i("h_LR"), _i("w_LR")
        h_hr, w_hr = _i("h_HR"), _i("w_HR")
        c = _i("c")

        mr = np.frombuffer(_b("data_LR"), dtype=np.float64).reshape(h_mr, w_mr, c)
        hr = np.frombuffer(_b("data_HR"), dtype=np.float64).reshape(h_hr, w_hr, c)
        yield WindRecord(row=row, index=index, mr=mr, hr=hr)


def load_records(path: Path) -> List[WindRecord]:
    records = list(iter_wind_records(path))
    if not records:
        raise SystemExit(f"No records found in {path}")
    return records


def ranking_for_sample(records: Sequence[WindRecord], sample_idx: int, candidate_rows: Optional[Sequence[int]] = None) -> RankingResult:
    target = records[sample_idx]
    ds = downsample_hr_to_mr(target.hr, target.mr.shape)

    candidates = list(candidate_rows) if candidate_rows is not None else list(range(len(records)))
    if sample_idx not in candidates:
        candidates.append(sample_idx)
    candidates = sorted(set(candidates))

    rmse_scores: List[Tuple[int, float]] = []
    mae_scores: List[Tuple[int, float]] = []
    for j in candidates:
        rmse_scores.append((j, _rmse(records[j].mr, ds)))
        mae_scores.append((j, _mean_abs(records[j].mr, ds)))

    rmse_scores.sort(key=lambda t: (t[1], t[0]))
    mae_scores.sort(key=lambda t: (t[1], t[0]))

    paired_rmse = next(v for j, v in rmse_scores if j == sample_idx)
    paired_mae = next(v for j, v in mae_scores if j == sample_idx)
    paired_rank_rmse = 1 + next(i for i, (j, _) in enumerate(rmse_scores) if j == sample_idx)
    paired_rank_mae = 1 + next(i for i, (j, _) in enumerate(mae_scores) if j == sample_idx)
    best_wrong_rmse = next(v for j, v in rmse_scores if j != sample_idx)
    best_wrong_mae = next(v for j, v in mae_scores if j != sample_idx)

    return RankingResult(
        paired_rank_rmse=paired_rank_rmse,
        paired_rank_mae=paired_rank_mae,
        paired_rmse=paired_rmse,
        paired_mae=paired_mae,
        best_wrong_rmse=best_wrong_rmse,
        best_wrong_mae=best_wrong_mae,
        paired_beats_wrong_rmse_gap=best_wrong_rmse - paired_rmse,
        paired_beats_wrong_mae_gap=best_wrong_mae - paired_mae,
        candidate_count=len(candidates),
    )


def orientation_check(record: WindRecord) -> Dict[str, float]:
    ds = downsample_hr_to_mr(record.hr, record.mr.shape)
    return {name: _rmse(_apply_transform(record.mr, name), ds) for name in TRANSFORMS}


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _panel(ax, img: np.ndarray, title: str, cmap: str = "viridis", vmin: Optional[float] = None, vmax: Optional[float] = None) -> None:
    im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def save_diagnostic_figure(out_png: Path, record: WindRecord) -> None:
    ds = downsample_hr_to_mr(record.hr, record.mr.shape)
    mr_speed = np.sqrt(np.square(record.mr[..., 0]) + np.square(record.mr[..., 1]))
    ds_speed = np.sqrt(np.square(ds[..., 0]) + np.square(ds[..., 1]))
    err_speed = np.abs(mr_speed - ds_speed)

    fig, axes = plt.subplots(3, 3, figsize=(13, 12), squeeze=False)

    ua_min = float(min(record.mr[..., 0].min(), ds[..., 0].min()))
    ua_max = float(max(record.mr[..., 0].max(), ds[..., 0].max()))
    va_min = float(min(record.mr[..., 1].min(), ds[..., 1].min()))
    va_max = float(max(record.mr[..., 1].max(), ds[..., 1].max()))
    sp_min = float(min(mr_speed.min(), ds_speed.min()))
    sp_max = float(max(mr_speed.max(), ds_speed.max()))

    _panel(axes[0, 0], record.mr[..., 0], f"sample {record.index} MR ua", cmap="coolwarm", vmin=ua_min, vmax=ua_max)
    _panel(axes[0, 1], ds[..., 0], f"sample {record.index} downsampled HR ua", cmap="coolwarm", vmin=ua_min, vmax=ua_max)
    _panel(axes[0, 2], np.abs(record.mr[..., 0] - ds[..., 0]), "|ua diff|", cmap="magma", vmin=0.0)

    _panel(axes[1, 0], record.mr[..., 1], f"sample {record.index} MR va", cmap="coolwarm", vmin=va_min, vmax=va_max)
    _panel(axes[1, 1], ds[..., 1], f"sample {record.index} downsampled HR va", cmap="coolwarm", vmin=va_min, vmax=va_max)
    _panel(axes[1, 2], np.abs(record.mr[..., 1] - ds[..., 1]), "|va diff|", cmap="magma", vmin=0.0)

    _panel(axes[2, 0], mr_speed, f"sample {record.index} MR speed", cmap="viridis", vmin=sp_min, vmax=sp_max)
    _panel(axes[2, 1], ds_speed, f"sample {record.index} downsampled HR speed", cmap="viridis", vmin=sp_min, vmax=sp_max)
    _panel(axes[2, 2], err_speed, "|speed diff|", cmap="magma", vmin=0.0)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def choose_representatives(rows: List[Dict[str, object]], limit: int) -> List[int]:
    if not rows:
        return []
    ordered = sorted(rows, key=lambda r: float(r["paired_rmse"]))
    picks = [int(ordered[0]["row"])]
    if len(ordered) > 2:
        picks.append(int(ordered[len(ordered) // 2]["row"]))
    if len(ordered) > 1:
        picks.append(int(ordered[-1]["row"]))
    extras = [int(r["row"]) for r in ordered[:limit] if int(r["row"]) not in picks]
    return (picks + extras)[:limit]


def summarize_metrics(rows: List[Dict[str, object]], orient_rows: List[Dict[str, object]], tfrecord: Path, candidates_desc: str) -> str:
    paired_rank1_rmse = sum(1 for r in rows if int(r["paired_rank_rmse"]) == 1)
    paired_rank1_mae = sum(1 for r in rows if int(r["paired_rank_mae"]) == 1)
    rank_rmse = np.asarray([int(r["paired_rank_rmse"]) for r in rows], dtype=int)
    gap_rmse = np.asarray([float(r["paired_beats_wrong_rmse_gap"]) for r in rows], dtype=float)
    paired_rmse = np.asarray([float(r["paired_rmse"]) for r in rows], dtype=float)

    lines: List[str] = []
    lines.append("# Wind MR→HR pairing audit\n\n")
    lines.append(f"TFRecord: `{tfrecord}`\n\n")
    lines.append(f"Candidate comparison set: {candidates_desc}.\n\n")
    lines.append("## Pairing result summary\n")
    lines.append(f"- Samples audited: {len(rows)}\n")
    lines.append(f"- Paired MR rank-1 by RMSE: {paired_rank1_rmse}/{len(rows)}\n")
    lines.append(f"- Paired MR rank-1 by MAE: {paired_rank1_mae}/{len(rows)}\n")
    lines.append(f"- Median paired-rank by RMSE: {int(np.median(rank_rmse))}\n")
    lines.append(f"- Mean paired RMSE: {paired_rmse.mean():.6g}\n")
    lines.append(f"- Median paired RMSE: {np.median(paired_rmse):.6g}\n")
    lines.append(f"- Mean best-wrong minus paired RMSE gap: {gap_rmse.mean():.6g}\n")
    lines.append(f"- Min best-wrong minus paired RMSE gap: {gap_rmse.min():.6g}\n")

    if orient_rows:
        lines.append("\n## Orientation sanity check\n")
        better = [r for r in orient_rows if str(r["best_transform"]) != "identity"]
        lines.append(f"- Representative samples checked: {len(orient_rows)}\n")
        lines.append(f"- Cases where a non-identity transform beat identity: {len(better)}\n")
        if better:
            for r in better:
                lines.append(
                    f"  - sample {int(r['index'])}: best={r['best_transform']} identity_rmse={float(r['identity_rmse']):.6g} best_rmse={float(r['best_rmse']):.6g}\n"
                )
        else:
            lines.append("- Identity was best on all representative orientation checks.\n")

    lines.append("\n## Interpretation guide\n")
    lines.append("- If the paired MR is usually rank-1 or near-rank-1 against wrong MR candidates, that strongly supports valid per-sample pairing.\n")
    lines.append("- If the best-wrong minus paired gap is consistently positive, the paired MR is meaningfully closer to its own downsampled HR than to wrong samples.\n")
    lines.append("- If identity beats transpose/flip/channel-swap checks, that argues against an obvious layout/orientation bug.\n")
    return "".join(lines)


def parse_sample_list(text: Optional[str], max_n: int) -> Optional[List[int]]:
    if text is None or text.strip() == "":
        return None
    out = []
    for part in text.replace(",", " ").split():
        v = int(part)
        if v < 0 or v >= max_n:
            raise SystemExit(f"Sample row {v} out of range for N={max_n}")
        out.append(v)
    return sorted(set(out))


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit whether wind MR samples are valid downsampled partners of wind HR samples")
    ap.add_argument("--tfrecord", type=Path, default=Path("example_data/wind_MR-HR.tfrecord"))
    ap.add_argument("--outdir", type=Path, default=Path("analysis/wind_mr_hr_pairing_audit"))
    ap.add_argument("--sample-rows", default=None, help="Optional row indices to audit (default: all rows)")
    ap.add_argument("--candidate-rows", default=None, help="Optional MR candidate row indices for wrong-sample ranking (default: all rows)")
    ap.add_argument("--orientation-rows", default=None, help="Optional row indices for orientation sanity checks")
    ap.add_argument("--figure-count", type=int, default=3, help="How many representative figures to write")
    args = ap.parse_args()

    records = load_records(args.tfrecord)
    n = len(records)

    sample_rows = parse_sample_list(args.sample_rows, n) or list(range(n))
    candidate_rows = parse_sample_list(args.candidate_rows, n)
    orientation_rows = parse_sample_list(args.orientation_rows, n)

    first = records[0]
    if first.mr.ndim != 3 or first.hr.ndim != 3:
        raise SystemExit(f"Unexpected tensor dims: MR={first.mr.shape} HR={first.hr.shape}")
    if first.mr.shape[-1] != 2 or first.hr.shape[-1] != 2:
        raise SystemExit(f"Expected 2 wind channels [ua, va], got MR={first.mr.shape} HR={first.hr.shape}")

    rows: List[Dict[str, object]] = []
    for i in sample_rows:
        rec = records[i]
        ds = downsample_hr_to_mr(rec.hr, rec.mr.shape)
        pm = _pair_metrics(rec.mr, ds)
        rr = ranking_for_sample(records, i, candidate_rows)
        rows.append({
            "row": rec.row,
            "index": rec.index,
            "mr_shape": f"{rec.mr.shape}",
            "hr_shape": f"{rec.hr.shape}",
            "mae_total": pm.mae_total,
            "rmse_total": pm.rmse_total,
            "max_abs_diff": pm.max_abs_diff,
            "mae_ua": pm.mae_ua,
            "mae_va": pm.mae_va,
            "rmse_ua": pm.rmse_ua,
            "rmse_va": pm.rmse_va,
            "paired_rank_rmse": rr.paired_rank_rmse,
            "paired_rank_mae": rr.paired_rank_mae,
            "paired_rmse": rr.paired_rmse,
            "paired_mae": rr.paired_mae,
            "best_wrong_rmse": rr.best_wrong_rmse,
            "best_wrong_mae": rr.best_wrong_mae,
            "paired_beats_wrong_rmse_gap": rr.paired_beats_wrong_rmse_gap,
            "paired_beats_wrong_mae_gap": rr.paired_beats_wrong_mae_gap,
            "candidate_count": rr.candidate_count,
        })

    orient_out: List[Dict[str, object]] = []
    orient_rows_eff = orientation_rows or choose_representatives(rows, limit=max(1, args.figure_count))
    for i in orient_rows_eff:
        rec = records[i]
        scores = orientation_check(rec)
        best_name, best_val = min(scores.items(), key=lambda kv: (kv[1], kv[0]))
        orient_out.append({
            "row": rec.row,
            "index": rec.index,
            **scores,
            "identity_rmse": scores["identity"],
            "best_transform": best_name,
            "best_rmse": best_val,
        })

    outdir = args.outdir
    figdir = outdir / "figures"
    outdir.mkdir(parents=True, exist_ok=True)
    write_csv(outdir / "per_sample_metrics.csv", rows)
    write_csv(outdir / "orientation_check.csv", orient_out)

    fig_rows = choose_representatives(rows, limit=max(1, args.figure_count))
    for i in fig_rows:
        rec = records[i]
        save_diagnostic_figure(figdir / f"sample_{rec.index}_row_{rec.row}.png", rec)

    candidates_desc = "all rows" if candidate_rows is None else f"rows {candidate_rows}"
    summary = summarize_metrics(rows, orient_out, args.tfrecord, candidates_desc)
    (outdir / "summary.md").write_text(summary)

    print(f"[OK] loaded {n} records from {args.tfrecord}")
    print(f"[OK] inferred MR shape {first.mr.shape} and HR shape {first.hr.shape}")
    print(f"[OK] wrote {outdir / 'per_sample_metrics.csv'}")
    print(f"[OK] wrote {outdir / 'orientation_check.csv'}")
    print(f"[OK] wrote {outdir / 'summary.md'}")
    print(f"[OK] wrote {len(fig_rows)} figure(s) under {figdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
