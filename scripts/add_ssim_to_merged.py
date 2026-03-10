#!/usr/bin/env python3
"""Add SSIM values to merged topology/physics CSV outputs.

Typical usage:
  python3 scripts/add_ssim_to_merged.py \
    --in-csv ttk_runs/combined/psnr_topology_physics_merged.csv \
    --out-csv ttk_runs/combined/psnr_topology_physics_merged.csv \
    --method-dir gan=data_out/wind_mrhr_gan \
    --method-dir cnn=data_out/wind_mrhr_cnn
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


def _parse_method_dirs(values: List[str]) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for kv in values:
        if "=" not in kv:
            raise ValueError(f"Expected method=dir, got: {kv}")
        k, v = kv.split("=", 1)
        out[k.strip().lower()] = Path(v.strip()).expanduser().resolve()
    return out


def _to_int(x: object) -> Optional[int]:
    try:
        s = str(x).strip()
        if not s:
            return None
        return int(float(s))
    except Exception:
        return None


def _speed_from_uv(arr: np.ndarray) -> np.ndarray:
    # arr shape: (H, W, C)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D per-sample array, got shape={arr.shape}")
    if arr.shape[-1] >= 2:
        return np.sqrt(np.square(arr[..., 0]) + np.square(arr[..., 1]))
    return np.abs(arr[..., 0])


def _pick_sample_col(fieldnames: List[str]) -> str:
    for c in ["sample_idx", "sample", "sidx", "index", "idx"]:
        if c in fieldnames:
            return c
    raise ValueError("Could not find sample index column (tried sample_idx/sample/sidx/index/idx)")


def _read_rows(path: Path) -> tuple[List[str], List[Dict[str, str]]]:
    with path.open("r", newline="") as f:
        r = csv.DictReader(f)
        if r.fieldnames is None:
            raise ValueError(f"No header found in {path}")
        rows = list(r)
        return list(r.fieldnames), rows


def _write_rows(path: Path, fieldnames: List[str], rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def main() -> int:
    ap = argparse.ArgumentParser(description="Add SSIM column to merged topology/physics CSV")
    ap.add_argument("--in-csv", type=Path, required=True)
    ap.add_argument("--out-csv", type=Path, required=True)
    ap.add_argument("--method-dir", action="append", required=True, help="method=/path/to/data_out_dir")
    ap.add_argument("--patch", type=int, default=160)
    ap.add_argument("--x0", type=int, default=0)
    ap.add_argument("--y0", type=int, default=0)
    ap.add_argument("--data-range", type=float, default=None, help="If omitted, use per-sample GT patch max-min")
    args = ap.parse_args()

    try:
        from skimage.metrics import structural_similarity as ssim  # type: ignore
    except Exception as e:  # pragma: no cover
        raise SystemExit("scikit-image not installed. Install via: python3 -m pip install --user scikit-image") from e

    method_dirs = _parse_method_dirs(args.method_dir)
    fieldnames, rows = _read_rows(args.in_csv)
    if not rows:
        raise SystemExit(f"No rows in {args.in_csv}")

    sample_col = _pick_sample_col(fieldnames)
    if "method" not in fieldnames:
        raise SystemExit("Input CSV must contain a 'method' column")

    cache: Dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def _load_arrays(method: str) -> tuple[np.ndarray, np.ndarray]:
        method = method.lower()
        if method in cache:
            return cache[method]
        if method not in method_dirs:
            raise KeyError(f"No method-dir supplied for method={method}")
        d = method_dirs[method]
        gt = np.load(d / "dataGT.npy", mmap_mode="r")
        sr = np.load(d / "dataSR.npy", mmap_mode="r")
        cache[method] = (gt, sr)
        return gt, sr

    out_rows: List[Dict[str, object]] = []
    p = int(args.patch)
    x0 = int(args.x0)
    y0 = int(args.y0)

    for row in rows:
        method = str(row.get("method", "")).strip().lower()
        sidx = _to_int(row.get(sample_col, ""))
        if not method or sidx is None:
            row["ssim"] = ""
            out_rows.append(dict(row))
            continue

        gt, sr = _load_arrays(method)
        if sidx < 0 or sidx >= len(gt):
            row["ssim"] = ""
            out_rows.append(dict(row))
            continue

        gt_s = gt[sidx]
        sr_s = sr[sidx]
        gt_patch = gt_s[y0 : y0 + p, x0 : x0 + p, :]
        sr_patch = sr_s[y0 : y0 + p, x0 : x0 + p, :]

        gt_speed = _speed_from_uv(gt_patch)
        sr_speed = _speed_from_uv(sr_patch)

        dr = args.data_range
        if dr is None:
            dr = float(gt_speed.max() - gt_speed.min())
            if dr == 0.0 or math.isnan(dr) or math.isinf(dr):
                dr = 1.0

        val = float(ssim(gt_speed, sr_speed, data_range=float(dr)))
        row2 = dict(row)
        row2["ssim"] = val
        out_rows.append(row2)

    out_fields = list(fieldnames)
    if "ssim" not in out_fields:
        out_fields.append("ssim")

    _write_rows(args.out_csv, out_fields, out_rows)
    print(f"Wrote: {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
