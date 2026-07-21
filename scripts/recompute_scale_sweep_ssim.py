#!/usr/bin/env python3
"""
Recompute SSIM for Candidate C and Candidate UV scale-sweep outputs.

This script avoids the Spark skimage/NumPy incompatibility by being run inside
a clean venv. It does not import TensorFlow, TTK, or project training code.

It computes:
  - speed_ssim: SSIM between scalar speed fields sqrt(u^2 + v^2)
  - u_ssim:     SSIM on u component
  - v_ssim:     SSIM on v component
  - uv_ssim:    average of u_ssim and v_ssim

All methods are evaluated on the common 168-sample benchmark outputs.
"""

from __future__ import annotations

from pathlib import Path
import csv
import math
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as ssim


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "ttk_runs_fixed" / "ssim_recomputed_scale_sweep"
OUT_DIR.mkdir(parents=True, exist_ok=True)


METHODS = [
    # Baselines
    {
        "method": "cnn",
        "family": "baseline",
        "training_size": "baseline",
        "dir": ROOT / "data_out_fixed" / "wind_mrhr_cnn",
    },
    {
        "method": "gan",
        "family": "baseline",
        "training_size": "baseline",
        "dir": ROOT / "data_out_fixed" / "wind_mrhr_gan",
    },

    # Candidate C scale sweep
    {
        "method": "candidateC_168",
        "family": "candidateC",
        "training_size": "168",
        "dir": ROOT / "data_out" / "wind_finetune_pilot_candidateC",
    },
    {
        "method": "candidateC_672",
        "family": "candidateC",
        "training_size": "672",
        "dir": ROOT / "data_out" / "wind_finetune_candidateC_expanded672",
    },
    {
        "method": "candidateC_1344",
        "family": "candidateC",
        "training_size": "1344",
        "dir": ROOT / "data_out" / "wind_finetune_candidateC_expanded1344",
    },
    {
        "method": "candidateC_2688",
        "family": "candidateC",
        "training_size": "2688",
        "dir": ROOT / "data_out" / "wind_finetune_candidateC_expanded2688",
    },

    # Candidate UV scale sweep
    {
        "method": "candidateUV_168",
        "family": "candidateUV",
        "training_size": "168",
        "dir": ROOT / "data_out" / "wind_finetune_pilot_candidateUV",
    },
    {
        "method": "candidateUV_672",
        "family": "candidateUV",
        "training_size": "672",
        "dir": ROOT / "data_out" / "wind_finetune_candidateUV_expanded672",
    },
    {
        "method": "candidateUV_1344",
        "family": "candidateUV",
        "training_size": "1344",
        "dir": ROOT / "data_out" / "wind_finetune_candidateUV_expanded1344",
    },
    {
        "method": "candidateUV_2688",
        "family": "candidateUV",
        "training_size": "2688",
        "dir": ROOT / "data_out" / "wind_finetune_candidateUV_expanded2688",
    },
]


def load_arrays(d: Path):
    required = ["idx.npy", "dataGT.npy", "dataSR.npy"]
    missing = [name for name in required if not (d / name).exists()]
    if missing:
        raise FileNotFoundError(f"{d}: missing {missing}")

    idx = np.load(d / "idx.npy")
    gt = np.load(d / "dataGT.npy", mmap_mode="r")
    sr = np.load(d / "dataSR.npy", mmap_mode="r")

    if idx.shape != (168,):
        raise ValueError(f"{d}: expected idx shape (168,), got {idx.shape}")
    if gt.shape != (168, 500, 500, 2):
        raise ValueError(f"{d}: expected GT shape (168,500,500,2), got {gt.shape}")
    if sr.shape != (168, 500, 500, 2):
        raise ValueError(f"{d}: expected SR shape (168,500,500,2), got {sr.shape}")

    return idx, gt, sr


def safe_range(a: np.ndarray, b: np.ndarray) -> float:
    """Use combined dynamic range, with a small fallback for constant fields."""
    lo = float(min(np.nanmin(a), np.nanmin(b)))
    hi = float(max(np.nanmax(a), np.nanmax(b)))
    r = hi - lo
    if not np.isfinite(r) or r <= 1e-12:
        r = 1.0
    return r


def compute_sample_ssim(gt_uv: np.ndarray, sr_uv: np.ndarray):
    gt_u = np.asarray(gt_uv[..., 0], dtype=np.float64)
    gt_v = np.asarray(gt_uv[..., 1], dtype=np.float64)
    sr_u = np.asarray(sr_uv[..., 0], dtype=np.float64)
    sr_v = np.asarray(sr_uv[..., 1], dtype=np.float64)

    gt_speed = np.sqrt(gt_u * gt_u + gt_v * gt_v)
    sr_speed = np.sqrt(sr_u * sr_u + sr_v * sr_v)

    u_val = ssim(gt_u, sr_u, data_range=safe_range(gt_u, sr_u))
    v_val = ssim(gt_v, sr_v, data_range=safe_range(gt_v, sr_v))
    speed_val = ssim(gt_speed, sr_speed, data_range=safe_range(gt_speed, sr_speed))
    uv_val = 0.5 * (u_val + v_val)

    return float(u_val), float(v_val), float(uv_val), float(speed_val)


def main():
    print("=" * 72)
    print("Recomputing SSIM for Candidate C / UV scale sweep")
    print("=" * 72)
    print("Output:", OUT_DIR)
    print()

    rows = []
    skipped = []

    # Load CNN GT as alignment reference.
    cnn_idx, cnn_gt, _ = load_arrays(ROOT / "data_out_fixed" / "wind_mrhr_cnn")

    for spec in METHODS:
        method = spec["method"]
        d = spec["dir"]

        if not d.exists():
            print(f"[skip] {method}: directory not found: {d}")
            skipped.append((method, str(d), "missing_dir"))
            continue

        print(f"[load] {method}: {d}")
        try:
            idx, gt, sr = load_arrays(d)
        except Exception as e:
            print(f"[skip] {method}: {e}")
            skipped.append((method, str(d), repr(e)))
            continue

        if not np.array_equal(idx, cnn_idx):
            raise ValueError(f"{method}: idx does not match CNN baseline idx")

        # Check GT alignment against CNN baseline.
        max_gt_diff = float(np.nanmax(np.abs(np.asarray(gt) - np.asarray(cnn_gt))))
        if max_gt_diff > 1e-8:
            raise ValueError(f"{method}: GT mismatch vs CNN baseline, max diff={max_gt_diff}")

        for i in range(len(idx)):
            u_s, v_s, uv_s, speed_s = compute_sample_ssim(gt[i], sr[i])
            rows.append({
                "sample_idx": int(idx[i]),
                "method": method,
                "family": spec["family"],
                "training_size": spec["training_size"],
                "ssim_u": u_s,
                "ssim_v": v_s,
                "ssim_uv_mean": uv_s,
                "ssim_speed": speed_s,
            })

        print(f"  done: 168 samples")

    per_sample = pd.DataFrame(rows)
    per_sample_path = OUT_DIR / "ssim_per_sample_scale_sweep.csv"
    per_sample.to_csv(per_sample_path, index=False)
    print()
    print("Written:", per_sample_path)

    # Summary by method.
    summary_rows = []
    for (method, family, training_size), g in per_sample.groupby(["method", "family", "training_size"], sort=False):
        row = {
            "method": method,
            "family": family,
            "training_size": training_size,
            "n": len(g),
        }
        for col in ["ssim_speed", "ssim_uv_mean", "ssim_u", "ssim_v"]:
            row[f"{col}_mean"] = float(g[col].mean())
            row[f"{col}_std"] = float(g[col].std(ddof=1))
            row[f"{col}_median"] = float(g[col].median())
            row[f"{col}_min"] = float(g[col].min())
            row[f"{col}_max"] = float(g[col].max())
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary_path = OUT_DIR / "ssim_summary_scale_sweep.csv"
    summary.to_csv(summary_path, index=False)
    print("Written:", summary_path)

    # Pairwise comparisons vs CNN for each available method.
    if "cnn" in set(per_sample["method"]):
        cnn = per_sample[per_sample["method"] == "cnn"].set_index("sample_idx")
        pair_rows = []
        for method in sorted(set(per_sample["method"]) - {"cnn"}):
            m = per_sample[per_sample["method"] == method].set_index("sample_idx")
            common = sorted(set(cnn.index).intersection(set(m.index)))
            for sid in common:
                pair_rows.append({
                    "sample_idx": sid,
                    "method": method,
                    "training_size": m.loc[sid, "training_size"],
                    "family": m.loc[sid, "family"],
                    "delta_ssim_speed_vs_cnn": float(m.loc[sid, "ssim_speed"] - cnn.loc[sid, "ssim_speed"]),
                    "delta_ssim_uv_mean_vs_cnn": float(m.loc[sid, "ssim_uv_mean"] - cnn.loc[sid, "ssim_uv_mean"]),
                    "speed_ssim_gt_cnn": bool(m.loc[sid, "ssim_speed"] > cnn.loc[sid, "ssim_speed"]),
                    "uv_ssim_gt_cnn": bool(m.loc[sid, "ssim_uv_mean"] > cnn.loc[sid, "ssim_uv_mean"]),
                })
        pair = pd.DataFrame(pair_rows)
        pair_path = OUT_DIR / "ssim_pairwise_vs_cnn.csv"
        pair.to_csv(pair_path, index=False)
        print("Written:", pair_path)

        # Compact wins table vs CNN.
        wins_rows = []
        for method, g in pair.groupby("method", sort=False):
            wins_rows.append({
                "method": method,
                "family": g["family"].iloc[0],
                "training_size": g["training_size"].iloc[0],
                "n": len(g),
                "speed_ssim_gt_cnn_count": int(g["speed_ssim_gt_cnn"].sum()),
                "uv_ssim_gt_cnn_count": int(g["uv_ssim_gt_cnn"].sum()),
                "mean_delta_ssim_speed_vs_cnn": float(g["delta_ssim_speed_vs_cnn"].mean()),
                "mean_delta_ssim_uv_mean_vs_cnn": float(g["delta_ssim_uv_mean_vs_cnn"].mean()),
            })
        wins = pd.DataFrame(wins_rows)
        wins_path = OUT_DIR / "ssim_wins_vs_cnn.csv"
        wins.to_csv(wins_path, index=False)
        print("Written:", wins_path)

    # Direct C-vs-UV by training size where both exist.
    c_uv_rows = []
    for size in ["168", "672", "1344", "2688"]:
        c_name = f"candidateC_{size}"
        u_name = f"candidateUV_{size}"
        if c_name not in set(per_sample["method"]) or u_name not in set(per_sample["method"]):
            continue
        c = per_sample[per_sample["method"] == c_name].set_index("sample_idx")
        u = per_sample[per_sample["method"] == u_name].set_index("sample_idx")
        common = sorted(set(c.index).intersection(set(u.index)))
        for sid in common:
            c_uv_rows.append({
                "sample_idx": sid,
                "training_size": size,
                "delta_c_minus_uv_ssim_speed": float(c.loc[sid, "ssim_speed"] - u.loc[sid, "ssim_speed"]),
                "delta_c_minus_uv_ssim_uv_mean": float(c.loc[sid, "ssim_uv_mean"] - u.loc[sid, "ssim_uv_mean"]),
                "c_speed_ssim_gt_uv": bool(c.loc[sid, "ssim_speed"] > u.loc[sid, "ssim_speed"]),
                "c_uv_ssim_gt_uv": bool(c.loc[sid, "ssim_uv_mean"] > u.loc[sid, "ssim_uv_mean"]),
            })
    if c_uv_rows:
        c_uv = pd.DataFrame(c_uv_rows)
        c_uv_path = OUT_DIR / "ssim_candidateC_vs_UV_by_scale.csv"
        c_uv.to_csv(c_uv_path, index=False)
        print("Written:", c_uv_path)

        c_uv_summary = []
        for size, g in c_uv.groupby("training_size", sort=False):
            c_uv_summary.append({
                "training_size": size,
                "n": len(g),
                "c_speed_ssim_gt_uv_count": int(g["c_speed_ssim_gt_uv"].sum()),
                "c_uv_ssim_gt_uv_count": int(g["c_uv_ssim_gt_uv"].sum()),
                "mean_delta_c_minus_uv_ssim_speed": float(g["delta_c_minus_uv_ssim_speed"].mean()),
                "mean_delta_c_minus_uv_ssim_uv_mean": float(g["delta_c_minus_uv_ssim_uv_mean"].mean()),
            })
        c_uv_summary_df = pd.DataFrame(c_uv_summary)
        c_uv_summary_path = OUT_DIR / "ssim_candidateC_vs_UV_by_scale_summary.csv"
        c_uv_summary_df.to_csv(c_uv_summary_path, index=False)
        print("Written:", c_uv_summary_path)

    # Markdown summary.
    md_path = OUT_DIR / "ssim_scale_sweep_summary.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Recomputed SSIM scale sweep\n\n")
        f.write("Computed in an isolated environment to avoid the Spark skimage/NumPy incompatibility.\n\n")
        f.write("Definitions:\n\n")
        f.write("- `ssim_speed`: SSIM on scalar wind-speed magnitude fields.\n")
        f.write("- `ssim_uv_mean`: average of separate SSIM on u and v components.\n\n")
        f.write("## Summary by method\n\n")
        f.write(summary[[
            "method", "family", "training_size", "n",
            "ssim_speed_mean", "ssim_uv_mean_mean",
            "ssim_speed_std", "ssim_uv_mean_std"
        ]].to_markdown(index=False))
        f.write("\n\n")
        if 'wins' in locals():
            f.write("## Pairwise wins vs CNN\n\n")
            f.write(wins.to_markdown(index=False))
            f.write("\n\n")
        if 'c_uv_summary_df' in locals():
            f.write("## Candidate C vs UV by training size\n\n")
            f.write(c_uv_summary_df.to_markdown(index=False))
            f.write("\n\n")
        if skipped:
            f.write("## Skipped methods\n\n")
            for method, path, reason in skipped:
                f.write(f"- `{method}`: `{path}` — {reason}\n")

    print("Written:", md_path)

    print()
    print("=" * 72)
    print("SSIM recomputation complete.")
    print("=" * 72)
    print(summary[["method", "training_size", "ssim_speed_mean", "ssim_uv_mean_mean"]].to_string(index=False))

    if skipped:
        print()
        print("Skipped methods:")
        for s in skipped:
            print("  ", s)


if __name__ == "__main__":
    main()
