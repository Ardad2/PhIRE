#!/usr/bin/env python3
"""
Add missing exceedance metrics to:
  ttk_runs_fixed/baseline_metrics/all_methods_per_sample.csv

Computes, for Bicubic/CNN/GAN, the absolute difference from GT in the
fraction of pixels whose scalar speed exceeds:
  fixed thresholds : 5, 10, 15 m/s
  GT percentile thresholds : p90, p95, p99

Run:
  cd ~/PhIRE/scripts
  PYTHONNOUSERSITE=1 /usr/bin/python3 add_baseline_exceedance_metrics.py
"""

from __future__ import annotations

import csv
import shutil
import sys
from pathlib import Path

import numpy as np


def repo_root() -> Path:
    here = Path(__file__).resolve().parent
    cwd = Path.cwd().resolve()
    candidates = [
        here.parent if here.name == "scripts" else here,
        cwd.parent if cwd.name == "scripts" else cwd,
        here,
        cwd,
    ]
    for c in candidates:
        if (c / "data_out_fixed").exists() and (c / "ttk_runs_fixed").exists():
            return c
    raise FileNotFoundError("Could not locate PhIRE repo root.")


ROOT = repo_root()

CNN_DIR = ROOT / "data_out_fixed" / "wind_mrhr_cnn"
GAN_DIR = ROOT / "data_out_fixed" / "wind_mrhr_gan"
BIC_DIR = ROOT / "data_out_fixed" / "wind_mrhr_bicubic"

CSV_PATH = ROOT / "ttk_runs_fixed" / "baseline_metrics" / "all_methods_per_sample.csv"

METHOD_DIRS = {
    "cnn":     CNN_DIR,
    "gan":     GAN_DIR,
    "bicubic": BIC_DIR,
}

METRIC_KEYS = [
    "exceed_frac_abs_delta_t5",
    "exceed_frac_abs_delta_t10",
    "exceed_frac_abs_delta_t15",
    "exceed_frac_abs_delta_p90",
    "exceed_frac_abs_delta_p95",
    "exceed_frac_abs_delta_p99",
]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def speed(a: np.ndarray) -> np.ndarray:
    if a.ndim == 3 and a.shape[-1] == 2:
        return np.sqrt(a[..., 0].astype(np.float32) ** 2 + a[..., 1].astype(np.float32) ** 2)
    if a.ndim == 3 and a.shape[-1] == 1:
        return a[..., 0].astype(np.float32)
    if a.ndim == 2:
        return a.astype(np.float32)
    raise ValueError(f"Unexpected array shape: {a.shape}")


def normalize_method(x: str) -> str:
    s = str(x or "").strip().lower()
    if "bic" in s: return "bicubic"
    if "cnn" in s: return "cnn"
    if "gan" in s: return "gan"
    return s


def find_col(cols: list[str], options: list[str]) -> str | None:
    lower = {c.lower(): c for c in cols}
    for opt in options:
        if opt.lower() in lower:
            return lower[opt.lower()]
    return None


def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Missing baseline metrics CSV: {CSV_PATH}")

    print("=" * 72)
    print("Adding exceedance metrics to baseline all-method CSV")
    print(f"repo root : {ROOT}")
    print(f"csv       : {CSV_PATH}")
    print("=" * 72)

    rows = read_csv_rows(CSV_PATH)
    if not rows:
        raise RuntimeError(f"No rows found in {CSV_PATH}")

    cols = list(rows[0].keys())
    sample_col = find_col(cols, ["sample_idx", "sample", "idx", "index", "id"])
    method_col = find_col(cols, ["method", "model", "name"])

    if sample_col is None or method_col is None:
        print("ERROR: Could not find sample/method columns.")
        print("Columns found:")
        for c in cols:
            print(" ", c)
        sys.exit(1)

    print(f"sample column : {sample_col}")
    print(f"method column : {method_col}")
    print(f"rows          : {len(rows)}")

    arrays = {}
    for method, d in METHOD_DIRS.items():
        missing = [name for name in ["idx.npy", "dataGT.npy", "dataSR.npy"]
                   if not (d / name).exists()]
        if missing:
            raise FileNotFoundError(f"Missing files in {d}: {missing}")
        arrays[method] = {
            "idx": np.load(d / "idx.npy"),
            "gt":  np.load(d / "dataGT.npy", mmap_mode="r"),
            "sr":  np.load(d / "dataSR.npy",  mmap_mode="r"),
        }

    ref_idx = arrays["cnn"]["idx"]
    for method in ["gan", "bicubic"]:
        if not np.array_equal(ref_idx, arrays[method]["idx"]):
            raise RuntimeError(f"idx mismatch between CNN and {method}")

    pos = {int(v): i for i, v in enumerate(ref_idx.tolist())}

    print("\nVerifying GT alignment...")
    for method in ["gan", "bicubic"]:
        max_diff = float(np.max(np.abs(arrays["cnn"]["gt"][:] - arrays[method]["gt"][:])))
        print(f"  CNN GT vs {method} GT max abs diff = {max_diff:.3e}")
        if max_diff > 1e-9:
            raise RuntimeError(f"GT arrays differ for {method}")

    print("\nComputing exceedance metrics...")
    cache: dict[tuple[int, str], dict[str, float]] = {}

    for sid in sorted(pos):
        i = pos[sid]
        gt_speed = speed(np.asarray(arrays["cnn"]["gt"][i]))

        thresholds = {
            "t5":  5.0,
            "t10": 10.0,
            "t15": 15.0,
            "p90": float(np.percentile(gt_speed, 90)),
            "p95": float(np.percentile(gt_speed, 95)),
            "p99": float(np.percentile(gt_speed, 99)),
        }

        gt_fracs = {
            label: float(np.mean(gt_speed > threshold))
            for label, threshold in thresholds.items()
        }

        for method in ["bicubic", "cnn", "gan"]:
            sr_speed = speed(np.asarray(arrays[method]["sr"][i]))
            values = {}
            for label, threshold in thresholds.items():
                pred_frac = float(np.mean(sr_speed > threshold))
                values[f"exceed_frac_abs_delta_{label}"] = abs(pred_frac - gt_fracs[label])
            cache[(sid, method)] = values

        if (sid + 1) % 20 == 0 or sid == sorted(pos)[-1]:
            print(f"  processed sample {sid}")

    for key in METRIC_KEYS:
        if key not in cols:
            cols.append(key)

    updated = 0
    skipped = 0

    for row in rows:
        try:
            sid    = int(float(str(row[sample_col]).strip()))
            method = normalize_method(row[method_col])
        except Exception:
            skipped += 1
            continue

        values = cache.get((sid, method))
        if not values:
            skipped += 1
            continue

        for key in METRIC_KEYS:
            row[key] = f"{values[key]:.10g}"
        updated += 1

    backup = CSV_PATH.with_suffix(".before_exceedance_backup.csv")
    shutil.copy2(CSV_PATH, backup)
    write_csv_rows(CSV_PATH, rows, cols)

    print("\nDone.")
    print(f"Updated rows : {updated}")
    print(f"Skipped rows : {skipped}")
    print(f"Backup       : {backup}")
    print(f"Rewrote      : {CSV_PATH}")


if __name__ == "__main__":
    main()