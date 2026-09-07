#!/usr/bin/env python3
"""
Select poster-friendly qualitative samples for the Candidate-C comparison.

Selection principle:
  1) Topology-inspired (Candidate C) must beat BOTH CNN and the matched
     reconstruction-only ablation on corrected bottleneck d_B and W2.
  2) The ablation should remain a strong conventional-fidelity control.
  3) Candidate C should remain close to the ablation in PSNR/SSIM.
  4) Rank by balanced PD improvement rather than one exceptional metric.

This script does NOT select from visual appearance. It creates a short,
pre-defined candidate list; visually inspect only that shortlist afterward.

Run from ~/PhIRE:
    python3 select_poster_qualitative_sample.py

Outputs:
    poster_qualitative_candidates.csv
"""

from __future__ import annotations

from pathlib import Path
import math
import re
import sys

import numpy as np
import pandas as pd

try:
    from skimage.metrics import structural_similarity
except Exception as exc:
    raise SystemExit(
        "scikit-image is required for SSIM. Run this inside the clean SSIM "
        "environment you used previously.\nOriginal import error: %r" % (exc,)
    )

ROOT = Path.home() / "PhIRE"

CANONICAL_PD = (
    Path.home()
    / "phire_runtime_audit_20260809_221548"
    / "recompute_pd"
    / "canonical_pd_full_sweep.csv"
)

METHOD_DIRS = {
    "CNN": ROOT / "data_out_fixed/wind_mrhr_cnn",
    "Ablation": ROOT / "data_out/wind_finetune_candidateUV_expanded2688",
    "Topology-inspired": ROOT / "data_out/wind_finetune_candidateC_expanded2688",
}

# Poster-selection tolerances. Adjust only BEFORE inspecting sample visuals.
MAX_PSNR_DROP_VS_ABLATION_DB = 0.75
MAX_SSIM_DROP_VS_ABLATION = 0.015
TOP_K = 20


def choose_col(columns, aliases, contains=()):
    lower = {c.lower(): c for c in columns}
    for a in aliases:
        if a.lower() in lower:
            return lower[a.lower()]
    for c in columns:
        lc = c.lower()
        if all(x.lower() in lc for x in contains):
            return c
    return None


def normalize_sample(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, float) and float(x).is_integer():
        return int(x)
    m = re.search(r"(\d+)", str(x))
    return int(m.group(1)) if m else np.nan


def find_target_run(unique_runs, role):
    """
    Resolve canonical-sweep run names without mixing in superlevel variants.
    """
    runs = [str(x) for x in unique_runs]

    def valid(r):
        lr = r.lower()
        return "superlevel" not in lr and "negated" not in lr

    runs = [r for r in runs if valid(r)]

    if role == "CNN":
        exact = [r for r in runs if r.lower() == "cnn"]
        if exact:
            return exact[0]
        candidates = [r for r in runs if r.lower().endswith("/cnn") or "cnn" == Path(r).name.lower()]

    elif role == "Ablation":
        candidates = [
            r for r in runs
            if "candidateuv_expanded2688" in r.lower()
            and "plus" not in r.lower()
            and "crit" not in r.lower()
            and "e2" not in r.lower()
        ]

    elif role == "Topology-inspired":
        candidates = [
            r for r in runs
            if "candidatec_expanded2688" in r.lower()
            and "plus" not in r.lower()
            and "e2" not in r.lower()
        ]
    else:
        raise ValueError(role)

    if not candidates:
        print("\nAvailable run names:")
        for r in sorted(runs):
            print(" ", r)
        raise RuntimeError(f"Could not resolve canonical-PD run for {role}")

    # Prefer the default topology_finetuning namespace.
    candidates.sort(
        key=lambda r: (
            "topology_finetuning" not in r.lower(),
            len(r),
            r,
        )
    )
    return candidates[0]


def load_pd_table():
    if not CANONICAL_PD.exists():
        raise FileNotFoundError(CANONICAL_PD)

    df = pd.read_csv(CANONICAL_PD)

    run_col = choose_col(df.columns, ["run", "method", "run_name"])
    sample_col = choose_col(
        df.columns,
        ["sample", "sample_idx", "sample_id", "idx", "pos_idx", "index"],
    )
    db_col = choose_col(
        df.columns,
        [
            "pd_bottleneck_all",
            "pd_bottleneck_all_distance",
            "pd_bottleneck",
            "bottleneck_all",
            "bottleneck",
            "d_b",
            "db",
        ],
        contains=("bottleneck", "all"),
    )
    w2_col = choose_col(
        df.columns,
        [
            "pd_w2_all",
            "pd_w2_all_distance",
            "pd_w2",
            "w2_all",
            "wasserstein_2",
            "wasserstein2",
            "w2",
        ],
        contains=("w2", "all"),
    )

    missing = [
        name
        for name, col in {
            "run": run_col,
            "sample": sample_col,
            "d_B": db_col,
            "W2": w2_col,
        }.items()
        if col is None
    ]
    if missing:
        print("Columns in canonical sweep:")
        print("\n".join(map(str, df.columns)))
        raise RuntimeError(f"Could not infer columns: {missing}")

    df = df[[run_col, sample_col, db_col, w2_col]].copy()
    df.columns = ["run", "sample", "d_B", "W2"]
    df["sample"] = df["sample"].map(normalize_sample)
    df = df.dropna(subset=["sample", "d_B", "W2"])
    df["sample"] = df["sample"].astype(int)

    targets = {
        role: find_target_run(df["run"].unique(), role)
        for role in METHOD_DIRS
    }

    print("Resolved canonical-PD runs:")
    for role, run in targets.items():
        print(f"  {role:19s} -> {run}")

    pieces = []
    for role, run in targets.items():
        x = df[df["run"] == run][["sample", "d_B", "W2"]].copy()
        if x["sample"].duplicated().any():
            raise RuntimeError(f"Duplicate sample rows for {role}: {run}")
        if len(x) != 168:
            print(f"WARNING: {role} has {len(x)} canonical rows, expected 168.")
        x = x.rename(columns={"d_B": f"d_B_{role}", "W2": f"W2_{role}"})
        pieces.append(x)

    out = pieces[0]
    for p in pieces[1:]:
        out = out.merge(p, on="sample", how="inner")

    return out.sort_values("sample").reset_index(drop=True)


def speed(uv):
    return np.sqrt(np.square(uv[..., 0]) + np.square(uv[..., 1]))


def psnr_uv(gt, sr):
    """
    Matches the project's vector-field PSNR convention:
      MSE over H,W,C and per-sample GT data range.
    """
    gt64 = np.asarray(gt, dtype=np.float64)
    sr64 = np.asarray(sr, dtype=np.float64)
    mse = float(np.mean((sr64 - gt64) ** 2))
    dr = float(np.max(gt64) - np.min(gt64))
    if mse == 0:
        return float("inf")
    if dr <= 0:
        return float("nan")
    return 20.0 * math.log10(dr) - 10.0 * math.log10(mse)


def ssim_speed(gt, sr):
    g = speed(np.asarray(gt, dtype=np.float64))
    s = speed(np.asarray(sr, dtype=np.float64))
    dr = float(g.max() - g.min())
    if dr <= 0:
        return float("nan")
    return float(structural_similarity(g, s, data_range=dr))


def load_fidelity():
    arrays = {}
    idxs = {}

    for role, d in METHOD_DIRS.items():
        idx = np.load(d / "idx.npy")
        if not np.array_equal(idx, np.arange(168)):
            raise RuntimeError(f"{role}: idx.npy is not exactly 0..167")
        idxs[role] = idx
        arrays[role] = {
            "GT": np.load(d / "dataGT.npy", mmap_mode="r"),
            "SR": np.load(d / "dataSR.npy", mmap_mode="r"),
        }

    # Ground truth must be identical across the controlled comparison.
    gt_ref = arrays["CNN"]["GT"]
    for role in ("Ablation", "Topology-inspired"):
        gt = arrays[role]["GT"]
        # chunked equality check
        for i in range(168):
            if not np.array_equal(gt_ref[i], gt[i]):
                raise RuntimeError(f"GT mismatch for {role}, sample {i}")

    rows = []
    for i in range(168):
        row = {"sample": i}
        gt = gt_ref[i]
        for role in ("CNN", "Ablation", "Topology-inspired"):
            sr = arrays[role]["SR"][i]
            row[f"PSNR_{role}"] = psnr_uv(gt, sr)
            row[f"SSIM_{role}"] = ssim_speed(gt, sr)
        rows.append(row)

        if (i + 1) % 20 == 0:
            print(f"Computed fidelity for {i+1}/168 samples")

    return pd.DataFrame(rows)


def relative_reduction(baseline, candidate):
    baseline = np.asarray(baseline, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    return 100.0 * (baseline - candidate) / baseline


def harmonic_positive(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    out = np.full_like(a, -np.inf, dtype=float)
    good = (a > 0) & (b > 0)
    out[good] = 2.0 * a[good] * b[good] / (a[good] + b[good])
    return out


def main():
    pdm = load_pd_table()
    fid = load_fidelity()

    df = pdm.merge(fid, on="sample", how="inner")

    df["gain_dB_vs_Ablation_pct"] = relative_reduction(
        df["d_B_Ablation"], df["d_B_Topology-inspired"]
    )
    df["gain_W2_vs_Ablation_pct"] = relative_reduction(
        df["W2_Ablation"], df["W2_Topology-inspired"]
    )
    df["gain_dB_vs_CNN_pct"] = relative_reduction(
        df["d_B_CNN"], df["d_B_Topology-inspired"]
    )
    df["gain_W2_vs_CNN_pct"] = relative_reduction(
        df["W2_CNN"], df["W2_Topology-inspired"]
    )

    df["PSNR_drop_vs_Ablation"] = (
        df["PSNR_Ablation"] - df["PSNR_Topology-inspired"]
    )
    df["SSIM_drop_vs_Ablation"] = (
        df["SSIM_Ablation"] - df["SSIM_Topology-inspired"]
    )

    # Require a clean qualitative story.
    eligible = (
        (df["gain_dB_vs_Ablation_pct"] > 0)
        & (df["gain_W2_vs_Ablation_pct"] > 0)
        & (df["gain_dB_vs_CNN_pct"] > 0)
        & (df["gain_W2_vs_CNN_pct"] > 0)
        & (df["PSNR_Ablation"] > df["PSNR_CNN"])
        & (df["SSIM_Ablation"] > df["SSIM_CNN"])
        & (df["PSNR_Topology-inspired"] >= df["PSNR_CNN"])
        & (df["SSIM_Topology-inspired"] >= df["SSIM_CNN"])
        & (df["PSNR_drop_vs_Ablation"] <= MAX_PSNR_DROP_VS_ABLATION_DB)
        & (df["SSIM_drop_vs_Ablation"] <= MAX_SSIM_DROP_VS_ABLATION)
    )

    df["balanced_gain_vs_Ablation"] = harmonic_positive(
        df["gain_dB_vs_Ablation_pct"],
        df["gain_W2_vs_Ablation_pct"],
    )
    df["balanced_gain_vs_CNN"] = harmonic_positive(
        df["gain_dB_vs_CNN_pct"],
        df["gain_W2_vs_CNN_pct"],
    )

    # Prioritize the controlled comparison, with CNN as secondary evidence.
    df["poster_score"] = (
        0.75 * df["balanced_gain_vs_Ablation"]
        + 0.25 * df["balanced_gain_vs_CNN"]
    )

    ranked = df[eligible].sort_values(
        ["poster_score", "balanced_gain_vs_Ablation"],
        ascending=False,
    )

    cols = [
        "sample",
        "poster_score",
        "d_B_CNN",
        "d_B_Ablation",
        "d_B_Topology-inspired",
        "gain_dB_vs_Ablation_pct",
        "W2_CNN",
        "W2_Ablation",
        "W2_Topology-inspired",
        "gain_W2_vs_Ablation_pct",
        "PSNR_CNN",
        "PSNR_Ablation",
        "PSNR_Topology-inspired",
        "PSNR_drop_vs_Ablation",
        "SSIM_CNN",
        "SSIM_Ablation",
        "SSIM_Topology-inspired",
        "SSIM_drop_vs_Ablation",
    ]

    out = ROOT / "poster_qualitative_candidates.csv"
    ranked[cols].to_csv(out, index=False)

    print()
    print(f"Eligible samples: {len(ranked)}/168")
    print(f"Saved: {out}")
    print()
    print("Top poster candidates:")
    with pd.option_context(
        "display.max_columns", None,
        "display.width", 240,
        "display.float_format", lambda x: f"{x:.4f}",
    ):
        print(ranked[cols].head(TOP_K).to_string(index=False))

    print(
        "\nNEXT STEP: render only the top ~8-10 samples and choose among those "
        "for visual clarity. Do not change the quantitative selection criteria "
        "after looking at the plots."
    )


if __name__ == "__main__":
    main()
