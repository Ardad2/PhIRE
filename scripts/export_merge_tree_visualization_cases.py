#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import csv
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

CNN_DIR = ROOT / "data_out_fixed" / "wind_mrhr_cnn"
GAN_DIR = ROOT / "data_out_fixed" / "wind_mrhr_gan"

OUTDIR = ROOT / "ttk_runs_fixed" / "merge_tree_visualization_cases"
OUTDIR.mkdir(parents=True, exist_ok=True)

SAMPLES = [10, 11, 12, 13, 76, 77, 78, 90, 91, 92, 93, 162, 163]

def load_arrays():
    gt = np.load(CNN_DIR / "dataGT.npy", mmap_mode="r")
    cnn = np.load(CNN_DIR / "dataSR.npy", mmap_mode="r")
    gan = np.load(GAN_DIR / "dataSR.npy", mmap_mode="r")
    idx = np.load(CNN_DIR / "idx.npy")

    idx_gan = np.load(GAN_DIR / "idx.npy")
    if not np.array_equal(idx, idx_gan):
        raise RuntimeError("CNN idx.npy and GAN idx.npy do not match.")

    gt_gan = np.load(GAN_DIR / "dataGT.npy", mmap_mode="r")
    max_gt_diff = float(np.max(np.abs(np.asarray(gt, dtype=np.float64) - np.asarray(gt_gan, dtype=np.float64))))
    if max_gt_diff > 1e-9:
        raise RuntimeError(f"GT mismatch between CNN and GAN directories: max_abs_diff={max_gt_diff:.3e}")

    pos = {int(v): i for i, v in enumerate(idx.tolist())}
    return gt, cnn, gan, idx, pos

def speed(a: np.ndarray) -> np.ndarray:
    return np.sqrt(a[..., 0] ** 2 + a[..., 1] ** 2).astype(np.float32)

def main():
    gt, cnn, gan, idx, pos = load_arrays()

    manifest_rows = []

    for sid in SAMPLES:
        if sid not in pos:
            print(f"WARNING: sample {sid} not found, skipping")
            continue

        i = pos[sid]
        sdir = OUTDIR / f"sample_{sid:03d}"
        sdir.mkdir(parents=True, exist_ok=True)

        gt_speed = speed(np.asarray(gt[i]))
        cnn_speed = speed(np.asarray(cnn[i]))
        gan_speed = speed(np.asarray(gan[i]))

        gt_path = sdir / "gt_speed.npy"
        cnn_path = sdir / "cnn_speed.npy"
        gan_path = sdir / "gan_speed.npy"

        np.save(gt_path, gt_speed)
        np.save(cnn_path, cnn_speed)
        np.save(gan_path, gan_speed)

        vmin = float(min(gt_speed.min(), cnn_speed.min(), gan_speed.min()))
        vmax = float(max(gt_speed.max(), cnn_speed.max(), gan_speed.max()))

        manifest_rows.extend([
            {"sample_idx": sid, "method": "gt",  "npy_file": gt_path.name,  "vmin": vmin, "vmax": vmax},
            {"sample_idx": sid, "method": "cnn", "npy_file": cnn_path.name, "vmin": vmin, "vmax": vmax},
            {"sample_idx": sid, "method": "gan", "npy_file": gan_path.name, "vmin": vmin, "vmax": vmax},
        ])

        print(f"wrote sample {sid}")

    manifest_path = OUTDIR / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["sample_idx", "method", "npy_file", "vmin", "vmax"])
        w.writeheader()
        w.writerows(manifest_rows)

    print(f"wrote {manifest_path}")
    print(f"output directory: {OUTDIR}")

if __name__ == "__main__":
    main()