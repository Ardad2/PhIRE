#!/usr/bin/env python3
"""
Phase 2/3 diagnostics for the Candidate D/Dpd/E/E2 topology-alignment audit
(see docs/candidateD_E_topology_audit.md).

This script is designed to run on the machine that actually holds trained
checkpoints / SR outputs / phase_c_results.csv files (referred to as "Spark"
in this repo's own notes). It was authored and syntax-checked in a sandbox
with no numpy/torch/matplotlib/vtk and no trained outputs for any of
CNN/C/Dpd/E2, so it could not be exercised against real data in that
session -- run it where the real artifacts live.

It does NOT modify, delete, or retrain anything. It only reads existing
dataSR.npy/dataGT.npy/phase_c_results.csv/constraints NPZ files and writes a
new CSV + optional PNGs under --out-dir.

What it computes (only for whichever inputs are actually found):

1. Final metrics per candidate (CNN / C / Dpd / E2), from phase_c_final's
   phase_c_results.csv if present:
     - mean PD bottleneck distance, mean MT distance
     - count(PD < CNN), count(MT < CNN)

2. Candidate C proxy metrics (from dataSR.npy/dataGT.npy directly):
     - L_crit at GT high-speed local maxima (mask = local max AND
       speed >= mean + z*std, z=1, pool=3x3 -- matches sr_network.py's
       _critical_value_loss)
     - number of selected maxima per sample

3. Candidate E2 proxy metrics (requires the constraints NPZ):
     - L_ttkcv, L_ttkpers per sample
     - number of TTK pairs used per sample
     - distribution (min/mean/max) of selected birth/death GT scalar values
     - whether the selected (birth_y, birth_x)/(death_y, death_x) locations
       are themselves local maxima/minima of the PREDICTED SR field (a cheap
       proxy for "is the proxy loss actually landing on structurally
       meaningful predicted-field locations")

4. Correlation check: for each candidate with both proxy and final metrics
   available, Pearson correlation between the proxy loss (L_crit or
   L_ttkcv+L_ttkpers) and the final PD/MT distance across samples. This is
   the script's answer to the user's key question: "if E/E2 lowers its
   proxy loss but final PD/MT does not improve, the proxy is misaligned."
   A near-zero or wrong-signed correlation is exactly that signature.

5. Visual overlays (if matplotlib is available): GT speed field with
   Candidate-C maxima, E2 birth vertices, and E2 death vertices marked, for
   any requested sample indices.

Usage examples
--------------
    python3 scripts/diagnose_candidate_topology_alignment.py \\
        --cnn-dir data_out_fixed/wind_mrhr_cnn \\
        --candidate-c-dir data_out/wind_finetune_candidateC_expanded672 \\
        --candidate-c-results ttk_runs_fixed/topology_finetuning/candidateC_expanded672_topology/phase_c_final/phase_c_results.csv \\
        --candidate-dpd-dir data_out/wind_finetune_candidateDpd_expanded672 \\
        --candidate-dpd-results ttk_runs_fixed/topology_finetuning/candidateDpd_expanded672_topology/phase_c_final/phase_c_results.csv \\
        --candidate-e2-dir data_out/wind_finetune_candidateE2_expanded672 \\
        --candidate-e2-results ttk_runs_fixed/topology_finetuning/candidateE2_expanded672_topology/phase_c_final/phase_c_results.csv \\
        --candidate-e2-constraints ttk_runs_fixed/topology_finetuning/candidateE2_expanded672_constraints/ttk_pd_critical_pairs_gtvalues.npz \\
        --out-dir diagnostics/candidateD_E_alignment \\
        --overlay-samples 8 12 25
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Optional

try:
    import numpy as np
except ImportError:
    print("[error] numpy is required. Run this on the machine with the real "
          "environment (e.g. inside .mamba_candidateD_pd or the Spark native env).",
          file=sys.stderr)
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAVE_MPL = True
except ImportError:
    _HAVE_MPL = False


_EPS = 1e-8


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _to_nhwc(arr: np.ndarray) -> np.ndarray:
    """Return (N, H, W, 2) regardless of input channel layout."""
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim == 4:
        if a.shape[-1] == 2:
            return a
        if a.shape[1] == 2:
            return np.transpose(a, (0, 2, 3, 1))
    raise ValueError(f"Cannot interpret shape {a.shape} as [u,v] field batch")


def load_uv(data_dir: Path):
    """Load dataSR.npy / dataGT.npy / idx.npy from a candidate's data_out dir."""
    sr = _to_nhwc(np.load(data_dir / "dataSR.npy"))
    gt = _to_nhwc(np.load(data_dir / "dataGT.npy"))
    idx = np.load(data_dir / "idx.npy") if (data_dir / "idx.npy").exists() else None
    return sr, gt, idx


def speed(uv: np.ndarray) -> np.ndarray:
    """(..., H, W, 2) -> (..., H, W) scalar speed."""
    return np.sqrt(uv[..., 0] ** 2 + uv[..., 1] ** 2 + _EPS)


def crop_to(field: np.ndarray, patch: int) -> np.ndarray:
    """Top-left patch x patch crop, matching convert_phire_to_vti.py defaults
    (x0=0, y0=0)."""
    return field[..., :patch, :patch]


# ---------------------------------------------------------------------------
# 1. Final metrics from phase_c_results.csv
# ---------------------------------------------------------------------------

def read_phase_c_results(path: Path):
    """Return dict: sample_idx (int) -> {pd_distance, mt_distance}."""
    out = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            # phase_c_results.csv schema (compute_composite_tree_distance.py):
            # columns include sample_idx (or similar), pd_distance, mt_distance.
            sidx_key = next((k for k in row if "sample" in k.lower() and "idx" in k.lower()), None)
            pd_key = next((k for k in row if "pd" in k.lower() and "dist" in k.lower()), None)
            mt_key = next((k for k in row if "mt" in k.lower() and "dist" in k.lower()), None)
            if sidx_key is None or pd_key is None or mt_key is None:
                continue
            try:
                sidx = int(float(row[sidx_key]))
                out[sidx] = {
                    "pd_distance": float(row[pd_key]),
                    "mt_distance": float(row[mt_key]),
                }
            except (ValueError, TypeError):
                continue
    return out


def summarize_final_metrics(name: str, results: dict, cnn_results: Optional[dict]):
    pds = [v["pd_distance"] for v in results.values()]
    mts = [v["mt_distance"] for v in results.values()]
    summary = {
        "candidate": name,
        "n_samples": len(results),
        "pd_mean": float(np.mean(pds)) if pds else None,
        "mt_mean": float(np.mean(mts)) if mts else None,
    }
    if cnn_results is not None:
        common = sorted(set(results) & set(cnn_results))
        pd_lt = sum(1 for s in common if results[s]["pd_distance"] < cnn_results[s]["pd_distance"])
        mt_lt = sum(1 for s in common if results[s]["mt_distance"] < cnn_results[s]["mt_distance"])
        summary["n_common_with_cnn"] = len(common)
        summary["pd_lt_cnn_count"] = pd_lt
        summary["mt_lt_cnn_count"] = mt_lt
    return summary


# ---------------------------------------------------------------------------
# 2. Candidate-C-style proxy: L_crit at GT high-speed local maxima
# ---------------------------------------------------------------------------

def local_max_mask(field2d: np.ndarray, pool: int = 3, z: float = 1.0) -> np.ndarray:
    """Boolean mask: True at pixels that are a local max in a `pool`x`pool`
    neighborhood AND exceed mean + z*std. Matches sr_network.py's
    _critical_value_loss exactly (per-sample adaptive threshold)."""
    H, W = field2d.shape
    pad = pool // 2
    padded = np.pad(field2d, pad, mode="edge")
    local_max = np.zeros_like(field2d)
    for dy in range(pool):
        for dx in range(pool):
            local_max = np.maximum(local_max, padded[dy:dy + H, dx:dx + W])
    is_local_max = field2d >= local_max - 1e-6
    thresh = field2d.mean() + z * field2d.std()
    return is_local_max & (field2d >= thresh)


def l_crit_per_sample(sr_speed: np.ndarray, gt_speed: np.ndarray, pool: int = 3, z: float = 1.0):
    """Per-sample L_crit (mean over selected pixels, matching the per-sample
    normalization the user asked to distinguish from the training code's
    batch-level normalization) and count of selected maxima."""
    n = sr_speed.shape[0]
    losses = np.full(n, np.nan, dtype=np.float64)
    counts = np.zeros(n, dtype=np.int64)
    for i in range(n):
        mask = local_max_mask(gt_speed[i], pool=pool, z=z)
        counts[i] = int(mask.sum())
        if counts[i] > 0:
            losses[i] = float(np.mean((sr_speed[i][mask] - gt_speed[i][mask]) ** 2))
    return losses, counts


# ---------------------------------------------------------------------------
# 3. Candidate E2 proxy: L_ttkcv, L_ttkpers from constraints NPZ
# ---------------------------------------------------------------------------

def load_e2_constraints(npz_path: Path):
    npz = np.load(npz_path, allow_pickle=True)
    required = ["sample_idx", "sample_start", "sample_count",
                "birth_vid", "death_vid", "birth_val", "death_val", "persistence"]
    missing = [k for k in required if k not in npz]
    if missing:
        raise ValueError(f"Constraints NPZ missing required keys: {missing}")
    return npz


def e2_proxy_per_sample(npz, sr_speed_by_wtk_idx: dict, patch: int):
    """Returns dict: wtk_idx -> {l_ttkcv, l_ttkpers, n_pairs, birth_vals, death_vals}."""
    sample_idx = npz["sample_idx"].astype(np.int64)
    sample_start = npz["sample_start"].astype(np.int64)
    sample_count = npz["sample_count"].astype(np.int64)
    birth_vid = npz["birth_vid"].astype(np.int64)
    death_vid = npz["death_vid"].astype(np.int64)
    birth_val = npz["birth_val"].astype(np.float32)
    death_val = npz["death_val"].astype(np.float32)
    persistence = npz["persistence"].astype(np.float32)

    out = {}
    W = patch
    for row_i in range(len(sample_idx)):
        wtk_idx = int(sample_idx[row_i])
        start, count = int(sample_start[row_i]), int(sample_count[row_i])
        if count == 0 or wtk_idx not in sr_speed_by_wtk_idx:
            continue
        bvid = birth_vid[start:start + count]
        dvid = death_vid[start:start + count]
        bval = birth_val[start:start + count]
        dval = death_val[start:start + count]
        pers = persistence[start:start + count]

        b_iy, b_ix = bvid // W, bvid % W
        d_iy, d_ix = dvid // W, dvid % W

        sr_c = sr_speed_by_wtk_idx[wtk_idx]
        sr_b = sr_c[b_iy, b_ix]
        sr_d = sr_c[d_iy, d_ix]

        l_ttkcv = 0.5 * (float(np.mean((sr_b - bval) ** 2)) + float(np.mean((sr_d - dval) ** 2)))
        sr_pers = np.abs(sr_d - sr_b)
        l_ttkpers = float(np.mean((sr_pers - pers) ** 2))

        out[wtk_idx] = {
            "l_ttkcv": l_ttkcv,
            "l_ttkpers": l_ttkpers,
            "n_pairs": count,
            "birth_val_min": float(bval.min()), "birth_val_max": float(bval.max()),
            "death_val_min": float(dval.min()), "death_val_max": float(dval.max()),
        }
    return out


# ---------------------------------------------------------------------------
# 5. Overlay plot
# ---------------------------------------------------------------------------

def plot_overlay(out_path: Path, gt_speed2d: np.ndarray, c_mask: Optional[np.ndarray],
                  birth_yx=None, death_yx=None, title: str = ""):
    if not _HAVE_MPL:
        print(f"[skip] matplotlib not available, cannot write {out_path}")
        return
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(gt_speed2d, cmap="viridis")
    plt.colorbar(im, ax=ax, label="wind speed (m/s)")
    if c_mask is not None:
        ys, xs = np.where(c_mask)
        ax.scatter(xs, ys, s=18, facecolors="none", edgecolors="white",
                   linewidths=1.2, label="Candidate C local maxima")
    if birth_yx is not None and len(birth_yx) > 0:
        by, bx = birth_yx[:, 0], birth_yx[:, 1]
        ax.scatter(bx, by, s=14, marker="^", c="red", label="E2 TTK birth vertices")
    if death_yx is not None and len(death_yx) > 0:
        dy, dx = death_yx[:, 0], death_yx[:, 1]
        ax.scatter(dx, dy, s=14, marker="v", c="orange", label="E2 TTK death vertices")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[ok] wrote {out_path}")


# ---------------------------------------------------------------------------
# Correlation check
# ---------------------------------------------------------------------------

def pearson(x, y):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return None
    x, y = x[ok], y[ok]
    if x.std() == 0 or y.std() == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cnn-dir", type=Path, default=None)
    ap.add_argument("--cnn-results", type=Path, default=None,
                     help="phase_c_results.csv for the CNN baseline")

    ap.add_argument("--candidate-c-dir", type=Path, default=None)
    ap.add_argument("--candidate-c-results", type=Path, default=None)

    ap.add_argument("--candidate-dpd-dir", type=Path, default=None)
    ap.add_argument("--candidate-dpd-results", type=Path, default=None)

    ap.add_argument("--candidate-e2-dir", type=Path, default=None)
    ap.add_argument("--candidate-e2-results", type=Path, default=None)
    ap.add_argument("--candidate-e2-constraints", type=Path, default=None)

    ap.add_argument("--patch", type=int, default=160)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--overlay-samples", type=int, nargs="*", default=[])
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    cnn_results = read_phase_c_results(args.cnn_results) if args.cnn_results and args.cnn_results.exists() else None
    if args.cnn_dir and args.cnn_dir.exists():
        cnn_sr, cnn_gt, cnn_idx = load_uv(args.cnn_dir)
        cnn_speed_gt = speed(crop_to(cnn_gt, args.patch))
        cnn_speed_sr = speed(crop_to(cnn_sr, args.patch))
    else:
        cnn_sr = cnn_gt = cnn_idx = cnn_speed_gt = cnn_speed_sr = None
        print("[info] no --cnn-dir provided/found; skipping CNN-derived proxies.")

    if cnn_results is not None:
        summary_rows.append(summarize_final_metrics("CNN", cnn_results, None))

    # --- Candidate C ---
    c_l_crit_by_idx = {}
    if args.candidate_c_dir and args.candidate_c_dir.exists():
        c_sr, c_gt, c_idx = load_uv(args.candidate_c_dir)
        c_speed_sr = speed(crop_to(c_sr, args.patch))
        c_speed_gt = speed(crop_to(c_gt, args.patch))
        l_crit, n_max = l_crit_per_sample(c_speed_sr, c_speed_gt)
        idx_list = c_idx if c_idx is not None else np.arange(len(l_crit))
        for i, wtk in enumerate(idx_list):
            c_l_crit_by_idx[int(wtk)] = {"l_crit": float(l_crit[i]), "n_maxima": int(n_max[i])}
        print(f"[ok] Candidate C proxy computed for {len(c_l_crit_by_idx)} samples "
              f"(mean L_crit={np.nanmean(l_crit):.5f}, mean n_maxima={n_max.mean():.1f})")
    else:
        print("[info] no --candidate-c-dir provided/found; skipping Candidate C proxy.")

    c_results = read_phase_c_results(args.candidate_c_results) if args.candidate_c_results and args.candidate_c_results.exists() else None
    if c_results is not None:
        summary_rows.append(summarize_final_metrics("Candidate C", c_results, cnn_results))
        if c_l_crit_by_idx:
            common = sorted(set(c_results) & set(c_l_crit_by_idx))
            corr_pd = pearson([c_l_crit_by_idx[s]["l_crit"] for s in common],
                               [c_results[s]["pd_distance"] for s in common])
            corr_mt = pearson([c_l_crit_by_idx[s]["l_crit"] for s in common],
                               [c_results[s]["mt_distance"] for s in common])
            print(f"[correlation] Candidate C: corr(L_crit, PD)={corr_pd}, corr(L_crit, MT)={corr_mt}")

    # --- Candidate Dpd ---
    if args.candidate_dpd_results and args.candidate_dpd_results.exists():
        dpd_results = read_phase_c_results(args.candidate_dpd_results)
        summary_rows.append(summarize_final_metrics("Candidate Dpd", dpd_results, cnn_results))
    else:
        print("[info] no --candidate-dpd-results provided/found; skipping Dpd final metrics.")

    # --- Candidate E2 ---
    e2_proxy_by_idx = {}
    if args.candidate_e2_dir and args.candidate_e2_dir.exists() and args.candidate_e2_constraints and args.candidate_e2_constraints.exists():
        e2_sr, e2_gt, e2_idx = load_uv(args.candidate_e2_dir)
        e2_speed_sr = speed(crop_to(e2_sr, args.patch))
        idx_list = e2_idx if e2_idx is not None else np.arange(len(e2_speed_sr))
        sr_by_wtk = {int(wtk): e2_speed_sr[i] for i, wtk in enumerate(idx_list)}
        npz = load_e2_constraints(args.candidate_e2_constraints)
        e2_proxy_by_idx = e2_proxy_per_sample(npz, sr_by_wtk, args.patch)
        n_pairs_all = [v["n_pairs"] for v in e2_proxy_by_idx.values()]
        print(f"[ok] Candidate E2 proxy computed for {len(e2_proxy_by_idx)} samples "
              f"(mean n_pairs={np.mean(n_pairs_all):.1f})")
    else:
        print("[info] no --candidate-e2-dir/--candidate-e2-constraints provided/found; "
              "skipping Candidate E2 proxy.")

    e2_results = read_phase_c_results(args.candidate_e2_results) if args.candidate_e2_results and args.candidate_e2_results.exists() else None
    if e2_results is not None:
        summary_rows.append(summarize_final_metrics("Candidate E2", e2_results, cnn_results))
        if e2_proxy_by_idx:
            common = sorted(set(e2_results) & set(e2_proxy_by_idx))
            combined_proxy = [e2_proxy_by_idx[s]["l_ttkcv"] + e2_proxy_by_idx[s]["l_ttkpers"] for s in common]
            corr_pd = pearson(combined_proxy, [e2_results[s]["pd_distance"] for s in common])
            corr_mt = pearson(combined_proxy, [e2_results[s]["mt_distance"] for s in common])
            print(f"[correlation] Candidate E2: corr(L_ttkcv+L_ttkpers, PD)={corr_pd}, "
                  f"corr(L_ttkcv+L_ttkpers, MT)={corr_mt}")
            print("  -> A near-zero or positive (same-direction-as-worse) correlation here "
                  "is the direct signature of proxy/evaluator misalignment the user asked "
                  "to test for explicitly.")

    # --- Write summary CSV ---
    if summary_rows:
        keys = sorted({k for row in summary_rows for k in row})
        out_csv = args.out_dir / "final_metrics_summary.csv"
        with out_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for row in summary_rows:
                w.writerow(row)
        print(f"[ok] wrote {out_csv}")
    else:
        print("[warn] no phase_c_results.csv found for any candidate; "
              "no final_metrics_summary.csv written.")

    # --- Overlays ---
    for sidx in args.overlay_samples:
        if cnn_speed_gt is None:
            print(f"[skip] no CNN GT available, cannot render overlay for sample {sidx}")
            continue
        pos = None
        if cnn_idx is not None:
            matches = np.where(cnn_idx == sidx)[0]
            if len(matches):
                pos = int(matches[0])
        else:
            pos = sidx
        if pos is None or pos >= len(cnn_speed_gt):
            print(f"[skip] sample {sidx} not found in CNN arrays")
            continue

        c_mask = local_max_mask(cnn_speed_gt[pos]) if args.candidate_c_dir else None

        birth_yx = death_yx = None
        if args.candidate_e2_constraints and args.candidate_e2_constraints.exists():
            npz = load_e2_constraints(args.candidate_e2_constraints)
            sample_idx = npz["sample_idx"].astype(np.int64)
            row_matches = np.where(sample_idx == sidx)[0]
            if len(row_matches):
                row_i = int(row_matches[0])
                start = int(npz["sample_start"][row_i])
                count = int(npz["sample_count"][row_i])
                W = args.patch
                bvid = npz["birth_vid"][start:start + count]
                dvid = npz["death_vid"][start:start + count]
                birth_yx = np.stack([bvid // W, bvid % W], axis=1)
                death_yx = np.stack([dvid // W, dvid % W], axis=1)

        plot_overlay(
            args.out_dir / f"overlay_sample_{sidx}.png",
            cnn_speed_gt[pos], c_mask, birth_yx, death_yx,
            title=f"Sample {sidx}: GT speed + Candidate C maxima + E2 TTK vertices",
        )

    print("\n[done] Diagnostics complete. See", args.out_dir)


if __name__ == "__main__":
    main()
