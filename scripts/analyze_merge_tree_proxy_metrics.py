#!/usr/bin/env python3
"""Compute hierarchy-aware merge-tree proxy statistics for scalar wind-speed super-resolution.

For each sample in the PhIRE dataset (GT, Bicubic, CNN, GAN), this script computes:

1. Area-filtered component counts at 40 GT-percentile thresholds (p1..p99).
   For each threshold tau and minimum area a, count the number of 8-connected
   superlevel-set components whose pixel area >= a.

2. Component size statistics (n_components, largest_area, largest_area_frac,
   mean_area, median_area, area_entropy, area_gini) swept over the same 40 thresholds.

3. A branch-persistence sweep inspired by merge-tree analysis: components are tracked
   as the threshold descends from max to min speed; births/merges are detected and
   recorded as (birth_value, death_value, persistence, final_area) tuples.

4. Branch-persistence summary statistics (branch_count, mean/median/max/total
   persistence, top-5 sum, persistence entropy, and counts above 4 cut values).

All statistics are computed for GT and each SR method. Per-method absolute errors
vs GT are also computed.

Inputs
------
  data_out_fixed/wind_mrhr_{bicubic,cnn,gan}/{idx,dataGT,dataSR}.npy

Outputs (all under ttk_runs_fixed/merge_tree_proxy/)
-------
  merge_tree_proxy_per_sample.csv      — one row per sample (wide format)
  merge_tree_proxy_summary.csv         — one row per method (mean abs errors)
  special_merge_tree_proxy_summary.csv — per-sample rows for special-case indices
  index.html                           — HTML report with white-card style

Run:
    cd ~/PhIRE/scripts
    python3 analyze_merge_tree_proxy_metrics.py
"""

from __future__ import annotations

import argparse
import collections
import csv
import html
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import label as _nd_label


# ---------------------------------------------------------------------------
# Repo layout / constants
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parent

DATA_ROOT    = REPO_ROOT / "data_out_fixed"
CNN_DIR      = DATA_ROOT / "wind_mrhr_cnn"
GAN_DIR      = DATA_ROOT / "wind_mrhr_gan"
BIC_DIR      = DATA_ROOT / "wind_mrhr_bicubic"
MANIFEST_CSV = REPO_ROOT / "ttk_runs_fixed" / "baseline_visual_panels" / "baseline_visual_manifest.csv"
SPECIAL_CSV  = REPO_ROOT / "ttk_runs_fixed" / "component_counts" / "special_component_case_summary.csv"
OUT_DIR      = REPO_ROOT / "ttk_runs_fixed" / "merge_tree_proxy"

N_THRESHOLDS  = 40          # GT percentile thresholds p1..p99
N_BRANCH_LVLS = 80          # levels for branch-persistence sweep
MIN_AREAS     = [1, 5, 10, 25, 50, 100]
METHODS       = ("bicubic", "cnn", "gan")
METHOD_DIRS   = {"bicubic": BIC_DIR, "cnn": CNN_DIR, "gan": GAN_DIR}
STRUCT8       = np.ones((3, 3), dtype=np.int32)
PERS_CUTS     = [0.5, 1.0, 2.0, 5.0]   # persistence thresholds for branch counts

# ---------------------------------------------------------------------------
# FORCED dict (mirror from generate_visual_inspection_panels.py)
# ---------------------------------------------------------------------------

FORCED: Dict[int, str] = {
    10: "adjacent control before sample 12",
    11: "adjacent control before sample 12",
    12: "strong MT-GAN anchor",
    13: "adjacent control after sample 12",
    76: "adjacent control before sample 77",
    77: "strong MT-GAN anchor",
    78: "adjacent control after sample 77",
    90: "GAN-majority / MT-CNN adjacent control near sample 92",
    91: "GAN-majority / MT-CNN adjacent control near sample 92",
    92: "strong MT-GAN anchor",
    93: "GAN-majority / MT-CNN adjacent control near sample 92",
    161: "adjacent control before rare topology-CNN samples 162-163",
    162: "rare PD-CNN and MT-CNN topology-consensus control",
    163: "rare PD-CNN and MT-CNN topology-consensus control",
    164: "adjacent control after rare topology-CNN samples 162-163",
    16: "moderate MT-GAN ridge-rich motif",
    17: "strong MT-GAN anchor",
    18: "moderate/lower MT-GAN ridge-rich motif",
    19: "strong MT-GAN anchor",
    20: "moderate/lower MT-GAN ridge-rich motif",
    25: "lower-confidence MT-GAN limitation case",
    80: "lower-confidence MT-GAN limitation case",
    154: "lower-confidence MT-GAN limitation case",
}

# ---------------------------------------------------------------------------
# Static fallback groups for HTML sections
# ---------------------------------------------------------------------------

_STATIC_GROUPS = [
    ("cluster_10_13",   "Adjacent cluster 10–13",             [10, 11, 12, 13]),
    ("cluster_76_78",   "Adjacent cluster 76–78",             [76, 77, 78]),
    ("cluster_90_93",   "Adjacent cluster 90–93",             [90, 91, 92, 93]),
    ("cluster_161_164", "Rare PD-CNN/MT-CNN cluster 161–164", [161, 162, 163, 164]),
    ("limitation_cases","Limitation cases 25, 80, 154",       [25, 80, 154]),
]

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_method(directory: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (idx_array, gt_array_mmap, sr_array_mmap). Exits if files missing."""
    for name in ("idx.npy", "dataGT.npy", "dataSR.npy"):
        p = directory / name
        if not p.exists():
            sys.exit(
                f"[error] Missing file: {p}\n"
                f"  Ensure {directory.name} outputs exist before running this script."
            )
    idx  = np.load(directory / "idx.npy", mmap_mode=None)
    gt   = np.load(directory / "dataGT.npy", mmap_mode="r")
    sr   = np.load(directory / "dataSR.npy", mmap_mode="r")
    return idx, gt, sr


def _load_manifest(path: Path) -> Optional[Dict[int, Dict[str, str]]]:
    """Load baseline_visual_manifest.csv -> {sample_idx: row}. Returns None if absent."""
    if not path.exists():
        print(f"[info] Manifest CSV not found: {path}")
        return None
    result: Dict[int, Dict[str, str]] = {}
    try:
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    si = int(row.get("sample_idx", row.get("idx", -1)))
                    result[si] = dict(row)
                except (ValueError, KeyError):
                    pass
    except Exception as e:
        print(f"[warn] Could not read manifest CSV {path}: {e}")
        return None
    return result


def _load_special_csv(path: Path) -> Optional[Dict[int, Dict[str, str]]]:
    """Load special_component_case_summary.csv -> {sample_idx: row}. Returns None if absent."""
    if not path.exists():
        print(f"[info] Special CSV not found: {path}")
        return None
    result: Dict[int, Dict[str, str]] = {}
    try:
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    si = int(row.get("sample_idx", row.get("idx", -1)))
                    result[si] = dict(row)
                except (ValueError, KeyError):
                    pass
    except Exception as e:
        print(f"[warn] Could not read special CSV {path}: {e}")
        return None
    return result


# ---------------------------------------------------------------------------
# Core computation helpers
# ---------------------------------------------------------------------------

def _speed(uv: np.ndarray) -> np.ndarray:
    """uv shape (..., 2) -> speed scalar."""
    return np.sqrt(uv[..., 0].astype(np.float32) ** 2 + uv[..., 1].astype(np.float32) ** 2)


def _gt_thresholds(speed_gt_2d: np.ndarray, n: int = N_THRESHOLDS) -> np.ndarray:
    """n evenly-spaced percentile thresholds from p1 to p99 (ascending values)."""
    percs = np.linspace(1.0, 99.0, n)
    return np.percentile(speed_gt_2d, percs)


def _gini(areas: np.ndarray) -> float:
    """Gini coefficient of area distribution."""
    if len(areas) == 0:
        return 0.0
    areas_s = np.sort(areas.astype(float))
    n_a = len(areas_s)
    cumsum = np.cumsum(areas_s)
    return max(0.0, (n_a + 1 - 2 * cumsum.sum() / (cumsum[-1] + 1e-12)) / (n_a + 1e-12))


def _area_entropy(areas: np.ndarray) -> float:
    """Entropy of the area distribution: -sum(p * log(p + 1e-12)) where p = areas/sum."""
    if len(areas) == 0:
        return 0.0
    total = float(areas.sum())
    if total == 0.0:
        return 0.0
    p = areas.astype(float) / total
    return float(-np.sum(p * np.log(p + 1e-12)))


def _main_stats(
    speed_2d: np.ndarray,
    thresholds: np.ndarray,
) -> Dict:
    """
    Compute area-filtered counts and size statistics over the 40 thresholds.

    Returns dict with keys:
        afc: {min_area: np.array(shape=(40,))}
        n_components, largest_area, largest_area_frac, mean_area, median_area,
        area_entropy, area_gini, exceedance: np.array(shape=(40,))
        thresholds: np.array(shape=(40,))
    """
    n = len(thresholds)
    total_pixels = float(speed_2d.size)

    afc = {a: np.zeros(n, dtype=np.int64) for a in MIN_AREAS}
    n_components   = np.zeros(n, dtype=np.float64)
    largest_area   = np.zeros(n, dtype=np.float64)
    largest_area_frac = np.zeros(n, dtype=np.float64)
    mean_area      = np.zeros(n, dtype=np.float64)
    median_area    = np.zeros(n, dtype=np.float64)
    area_entropy   = np.zeros(n, dtype=np.float64)
    area_gini      = np.zeros(n, dtype=np.float64)
    exceedance     = np.zeros(n, dtype=np.float64)

    flat_speed = speed_2d.ravel()

    for i, tau in enumerate(thresholds):
        mask = speed_2d >= tau
        exc  = mask.sum()
        exceedance[i] = exc / total_pixels

        if exc == 0:
            # all stats remain 0
            continue

        labeled, n_comp = _nd_label(mask, structure=STRUCT8)
        if n_comp == 0:
            continue

        flat_labels = labeled.ravel()
        # areas per component (label 0 = background)
        bc = np.bincount(flat_labels)
        if len(bc) > 1:
            areas = bc[1:]  # drop background
        else:
            areas = np.array([], dtype=np.int64)

        if len(areas) == 0:
            continue

        n_components[i]      = float(n_comp)
        la                   = float(areas.max())
        largest_area[i]      = la
        largest_area_frac[i] = la / total_pixels
        mean_area[i]         = float(areas.mean())
        median_area[i]       = float(np.median(areas))
        area_entropy[i]      = _area_entropy(areas)
        area_gini[i]         = _gini(areas)

        for a in MIN_AREAS:
            afc[a][i] = int(np.sum(areas >= a))

    return {
        "afc":               afc,
        "n_components":      n_components,
        "largest_area":      largest_area,
        "largest_area_frac": largest_area_frac,
        "mean_area":         mean_area,
        "median_area":       median_area,
        "area_entropy":      area_entropy,
        "area_gini":         area_gini,
        "exceedance":        exceedance,
        "thresholds":        thresholds,
    }


# ---------------------------------------------------------------------------
# Branch persistence sweep
# ---------------------------------------------------------------------------

def _branch_sweep(
    speed_2d: np.ndarray,
    n_levels: int = N_BRANCH_LVLS,
) -> List[Tuple[float, float, float, int]]:
    """
    Sweep from max to min speed. Track component births/merges via scipy.ndimage.label.

    Algorithm:
    1. Create n_levels+1 threshold values linearly from max(speed) to min(speed), inclusive.
    2. Process thresholds descending (index 0 = highest).
    3. At each step, label the superlevel set.
    4. For each newly-appeared component: birth_value = max(speed_2d[comp_mask]).
    5. For merges (multiple old components -> one new): elder rule (highest birth survives);
       dying components recorded with death_value = current threshold.
    6. Surviving components at end: death_value = min(speed_2d).

    Returns list of (birth_value, death_value, persistence, final_area) tuples.
    persistence = birth_value - death_value.
    """
    sp_max = float(speed_2d.max())
    sp_min = float(speed_2d.min())

    if sp_max == sp_min:
        # Trivial constant field
        return [(sp_max, sp_min, 0.0, int(speed_2d.size))]

    thresholds_desc = np.linspace(sp_max, sp_min, n_levels + 1)

    # State tracking
    comp_births: Dict[int, float] = {}      # component_id -> birth_value
    comp_rep:    Dict[int, Tuple[int, int]] = {}  # component_id -> (row, col) representative pixel
    comp_areas:  Dict[int, int]  = {}       # component_id -> pixel area

    branches: List[Tuple[float, float, float, int]] = []

    prev_labeled:    Optional[np.ndarray] = None
    prev_comp_ids:   set = set()

    flat_speed = speed_2d.ravel()
    H, W = speed_2d.shape

    for step_idx, tau in enumerate(thresholds_desc):
        mask = speed_2d >= tau
        if not np.any(mask):
            # No components at this threshold; previously alive comps die here
            death_val = float(tau)
            for cid in list(comp_births.keys()):
                bv = comp_births[cid]
                pers = bv - death_val
                area = comp_areas.get(cid, 0)
                branches.append((bv, death_val, pers, area))
            comp_births.clear()
            comp_rep.clear()
            comp_areas.clear()
            prev_labeled = None
            prev_comp_ids = set()
            continue

        labeled, n_comp = _nd_label(mask, structure=STRUCT8)

        if n_comp == 0:
            prev_labeled = None
            prev_comp_ids = set()
            continue

        flat_labels = labeled.ravel()

        # Compute areas and representative pixels for new labeled image
        bc = np.bincount(flat_labels)
        new_areas: Dict[int, int] = {}
        if len(bc) > 1:
            for cid_new in range(1, len(bc)):
                new_areas[cid_new] = int(bc[cid_new])

        # Representative pixels: first occurrence of each label
        uniq_labels, first_idxs = np.unique(flat_labels, return_index=True)
        new_reps: Dict[int, Tuple[int, int]] = {}
        for lbl, fidx in zip(uniq_labels, first_idxs):
            if lbl == 0:
                continue
            r, c = int(fidx // W), int(fidx % W)
            new_reps[lbl] = (r, c)

        current_comp_ids = set(new_areas.keys())

        if prev_labeled is None:
            # First non-empty level: birth all components
            for cid_new in current_comp_ids:
                ri, ci = new_reps[cid_new]
                # Birth value: max speed in this component
                comp_mask_idxs = np.where(flat_labels == cid_new)[0]
                bv = float(flat_speed[comp_mask_idxs].max())
                comp_births[cid_new] = bv
                comp_rep[cid_new]    = new_reps[cid_new]
                comp_areas[cid_new]  = new_areas[cid_new]
        else:
            # Map old component ids to new component ids using representative pixels
            old_to_new: Dict[int, int] = {}
            for old_cid, (ri, ci) in comp_rep.items():
                # Check where this old rep pixel maps in the new labeling
                new_cid = int(labeled[ri, ci])
                if new_cid > 0:
                    old_to_new[old_cid] = new_cid
                else:
                    # Rep pixel fell below threshold; sample the component more broadly
                    # find any pixel from the old component that's still in mask
                    # (skip expensive search; treat as died)
                    old_to_new[old_cid] = 0  # mark as died

            # Build new_to_olds: which old components map to each new component
            new_to_olds: Dict[int, List[int]] = collections.defaultdict(list)
            for old_cid, new_cid in old_to_new.items():
                if new_cid > 0:
                    new_to_olds[new_cid].append(old_cid)

            # Handle old components that died (mapped to background)
            death_val = float(tau)
            for old_cid, new_cid in old_to_new.items():
                if new_cid == 0:
                    bv = comp_births.get(old_cid, death_val)
                    pers = bv - death_val
                    area = comp_areas.get(old_cid, 0)
                    branches.append((bv, death_val, pers, area))

            # Process each new component
            new_comp_births:  Dict[int, float]           = {}
            new_comp_reps:    Dict[int, Tuple[int, int]] = {}
            new_comp_areas:   Dict[int, int]             = {}

            for cid_new in current_comp_ids:
                olds = new_to_olds.get(cid_new, [])
                area_new = new_areas[cid_new]

                if len(olds) == 0:
                    # Born now
                    comp_mask_idxs = np.where(flat_labels == cid_new)[0]
                    bv = float(flat_speed[comp_mask_idxs].max())
                    new_comp_births[cid_new] = bv
                    new_comp_reps[cid_new]   = new_reps[cid_new]
                    new_comp_areas[cid_new]  = area_new

                elif len(olds) == 1:
                    # Continuation
                    old_cid = olds[0]
                    new_comp_births[cid_new] = comp_births[old_cid]
                    new_comp_reps[cid_new]   = new_reps[cid_new]
                    new_comp_areas[cid_new]  = area_new

                else:
                    # Merge: elder rule - highest birth_value survives
                    elder_cid = max(olds, key=lambda c: comp_births.get(c, 0.0))
                    elder_bv  = comp_births[elder_cid]

                    for old_cid in olds:
                        if old_cid == elder_cid:
                            continue
                        bv = comp_births.get(old_cid, death_val)
                        pers = bv - death_val
                        area = comp_areas.get(old_cid, 0)
                        branches.append((bv, death_val, pers, area))

                    new_comp_births[cid_new] = elder_bv
                    new_comp_reps[cid_new]   = new_reps[cid_new]
                    new_comp_areas[cid_new]  = area_new

            comp_births = new_comp_births
            comp_rep    = new_comp_reps
            comp_areas  = new_comp_areas

        prev_labeled  = labeled
        prev_comp_ids = current_comp_ids

    # Surviving components at end: death_value = min(speed_2d)
    death_val = sp_min
    for cid, bv in comp_births.items():
        pers = bv - death_val
        area = comp_areas.get(cid, 0)
        branches.append((bv, death_val, pers, area))

    return branches


# ---------------------------------------------------------------------------
# Branch summary
# ---------------------------------------------------------------------------

_BRANCH_STAT_NAMES: List[str] = []  # filled after function definition


def _branch_summary(branches: List[Tuple[float, float, float, int]]) -> Dict[str, float]:
    """
    Summarize branch list into 11 statistics.
    branches: list of (birth_value, death_value, persistence, final_area)
    """
    if not branches:
        stat_names = [
            "branch_count",
            "branch_count_gt_0p5",
            "branch_count_gt_1",
            "branch_count_gt_2",
            "branch_count_gt_5",
            "mean_persistence",
            "median_persistence",
            "max_persistence",
            "total_persistence",
            "top5_persistence_sum",
            "persistence_entropy",
        ]
        return {k: 0.0 for k in stat_names}

    persis = np.array([b[2] for b in branches], dtype=float)
    persis = np.maximum(persis, 0.0)  # guard negatives from floating point

    branch_count = len(branches)
    counts_gt = {cut: int(np.sum(persis > cut)) for cut in PERS_CUTS}

    mean_p   = float(persis.mean())
    median_p = float(np.median(persis))
    max_p    = float(persis.max())
    total_p  = float(persis.sum())
    top5_p   = float(np.sort(persis)[-5:].sum()) if len(persis) >= 5 else total_p

    # Entropy of persistence distribution
    if total_p > 0:
        p_norm = persis / total_p
        ent = float(-np.sum(p_norm * np.log(p_norm + 1e-12)))
    else:
        ent = 0.0

    return {
        "branch_count":         float(branch_count),
        "branch_count_gt_0p5":  float(counts_gt[0.5]),
        "branch_count_gt_1":    float(counts_gt[1.0]),
        "branch_count_gt_2":    float(counts_gt[2.0]),
        "branch_count_gt_5":    float(counts_gt[5.0]),
        "mean_persistence":     mean_p,
        "median_persistence":   median_p,
        "max_persistence":      max_p,
        "total_persistence":    total_p,
        "top5_persistence_sum": top5_p,
        "persistence_entropy":  ent,
    }


_BRANCH_STAT_NAMES = list(_branch_summary([]).keys())


# ---------------------------------------------------------------------------
# Per-sample computation
# ---------------------------------------------------------------------------

_SIZE_STAT_NAMES = [
    "n_components", "largest_area", "largest_area_frac",
    "mean_area", "median_area", "area_entropy", "area_gini",
]


def _squeeze_2d(arr: np.ndarray) -> np.ndarray:
    """Squeeze leading batch dimensions if present: (1,H,W,2) -> (H,W,2)."""
    while arr.ndim > 3:
        if arr.shape[0] == 1:
            arr = arr[0]
        else:
            break
    return arr


def _process_sample(
    si: int,
    gt_uv: np.ndarray,                      # (H,W,2) or (1,H,W,2)
    sr_uvs: Dict[str, np.ndarray],           # method -> (H,W,2) or (1,H,W,2)
) -> Dict:
    """Return one wide-format row for this sample."""
    gt_uv = _squeeze_2d(np.array(gt_uv, dtype=np.float32))
    sr_uvs = {m: _squeeze_2d(np.array(v, dtype=np.float32)) for m, v in sr_uvs.items()}

    gt_speed = _speed(gt_uv)
    sr_speeds = {m: _speed(sr_uvs[m]) for m in METHODS}

    # GT thresholds (from GT only)
    thresholds = _gt_thresholds(gt_speed)

    # --- Main stats at 40 thresholds ---
    gt_stats  = _main_stats(gt_speed, thresholds)
    sr_stats  = {m: _main_stats(sr_speeds[m], thresholds) for m in METHODS}

    # --- Branch persistence sweep ---
    gt_branches  = _branch_sweep(gt_speed)
    sr_branches  = {m: _branch_sweep(sr_speeds[m]) for m in METHODS}

    gt_bsum  = _branch_summary(gt_branches)
    sr_bsums = {m: _branch_summary(sr_branches[m]) for m in METHODS}

    # --- Assemble row ---
    row: Dict = {"sample_idx": si}

    # Area-filtered count L1 errors
    for m in METHODS:
        for a in MIN_AREAS:
            gt_curve  = gt_stats["afc"][a].astype(float)
            sr_curve  = sr_stats[m]["afc"][a].astype(float)
            row[f"{m}_afc_l1_a{a}"] = float(np.abs(sr_curve - gt_curve).sum())

    # Component size stats (mean of curve over 40 thresholds) + abs errors
    for stat_name in _SIZE_STAT_NAMES:
        gt_mean = float(gt_stats[stat_name].mean())
        row[f"gt_sz_{stat_name}"] = gt_mean
        for m in METHODS:
            sr_mean = float(sr_stats[m][stat_name].mean())
            row[f"{m}_sz_{stat_name}"]     = sr_mean
            row[f"{m}_sz_{stat_name}_mae"] = abs(sr_mean - gt_mean)

    # Exceedance L1 error
    for m in METHODS:
        gt_exc = gt_stats["exceedance"]
        sr_exc = sr_stats[m]["exceedance"]
        row[f"{m}_exceedance_l1"] = float(np.abs(sr_exc - gt_exc).sum())

    # Branch stats
    for bstat in _BRANCH_STAT_NAMES:
        row[f"gt_{bstat}"] = gt_bsum[bstat]
        for m in METHODS:
            sr_val = sr_bsums[m][bstat]
            row[f"{m}_{bstat}"]     = sr_val
            row[f"{m}_{bstat}_ae"]  = abs(sr_val - gt_bsum[bstat])

    return row


# ---------------------------------------------------------------------------
# Summary aggregation
# ---------------------------------------------------------------------------

def _compute_summary(per_sample: List[Dict]) -> List[Dict]:
    """One row per method. Mean absolute errors across all samples."""
    if not per_sample:
        return []

    # Identify error columns per method
    all_keys = list(per_sample[0].keys())

    summary_rows: List[Dict] = []
    for m in METHODS:
        err_cols = [k for k in all_keys if k.startswith(f"{m}_") and (k.endswith("_ae") or k.endswith("_mae") or k.endswith("_l1"))]
        raw_cols = [k for k in all_keys if k.startswith(f"{m}_") and not k.endswith("_ae") and not k.endswith("_mae") and not k.endswith("_l1")]

        row: Dict = {"method": m}
        for col in err_cols:
            vals = [r[col] for r in per_sample if col in r]
            row[f"mean_{col}"] = float(np.mean(vals)) if vals else float("nan")
        for col in raw_cols:
            vals = [r[col] for r in per_sample if col in r]
            row[f"mean_{col}"] = float(np.mean(vals)) if vals else float("nan")
        summary_rows.append(row)

    return summary_rows


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------

def _write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    cols = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------

CSS = """
*, *::before, *::after { box-sizing: border-box; }
body {
  font-family: system-ui, sans-serif;
  background: #f0f2f5;
  color: #1a1a1a;
  margin: 0; padding: 1.5em;
}
h1  { font-size: 1.3em; margin: 0 0 0.3em; }
h2  { font-size: 1em; margin: 0 0 0.6em; color: #1e293b; }
.intro { color: #555; font-size: 0.88em; margin: 0 0 1.5em; line-height: 1.6; }
.card {
  background: #fff;
  border: 1px solid #dde1e7;
  border-radius: 8px;
  box-shadow: 0 1px 4px rgba(0,0,0,.07);
  margin-bottom: 1.5em;
  padding: 1.25em 1.5em;
  overflow-x: auto;
}
nav {
  margin-bottom: 1.2em;
  font-size: 0.85em;
}
nav a {
  margin-right: 0.8em;
  color: #1d4ed8;
  text-decoration: none;
}
nav a:hover { text-decoration: underline; }
.data-table {
  border-collapse: collapse;
  font-size: 0.82em;
  width: 100%;
}
.data-table th {
  background: #f7f8fa;
  color: #444;
  font-weight: 600;
  text-align: left;
  padding: 5px 9px;
  border: 1px solid #e2e5ea;
  white-space: nowrap;
}
.data-table td {
  padding: 3px 9px;
  border: 1px solid #e2e5ea;
  font-variant-numeric: tabular-nums;
  white-space: nowrap;
}
.data-table tr:nth-child(even) td { background: #fafbfc; }
.winner-cell { background: #bbf7d0 !important; color: #065f46; font-weight: 600; }
"""


def _fmt(v: object, decimals: int = 3) -> str:
    if v is None:
        return "—"
    try:
        f = float(str(v))  # type: ignore[arg-type]
        if math.isnan(f):
            return "—"
        if isinstance(v, int) or (f == int(f) and abs(f) < 1e9 and decimals == 0):
            return str(int(f))
        return f"{f:.{decimals}f}"
    except (TypeError, ValueError):
        return html.escape(str(v))


def _summary_table_html(summary_rows: List[Dict]) -> str:
    """HTML table: methods x key metrics."""
    key_metrics = [
        ("mean_branch_count_ae",         "branch_count MAE",          0),
        ("mean_mean_persistence_ae",      "mean_pers MAE",             3),
        ("mean_max_persistence_ae",       "max_pers MAE",              3),
        ("mean_top5_persistence_sum_ae",  "top5_pers_sum MAE",         3),
        ("mean_afc_l1_a10",               "afc_l1_a10 mean",           3),
    ]

    hdr = "<tr><th>Method</th>" + "".join(f"<th>{label}</th>" for _, label, _ in key_metrics) + "</tr>"
    body_rows: List[str] = []
    for row in summary_rows:
        m = row.get("method", "")
        cells = f"<td><strong>{html.escape(m.upper())}</strong></td>"
        for col, _, dec in key_metrics:
            v = row.get(col, None)
            cells += f"<td>{_fmt(v, dec)}</td>"
        body_rows.append(f"<tr>{cells}</tr>")

    return (
        "<table class='data-table'>"
        f"<thead>{hdr}</thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table>"
    )


def _group_section_html(
    section_id: str,
    title: str,
    sample_ids: List[int],
    per_sample_by_idx: Dict[int, Dict],
) -> str:
    """One card for a group of samples."""
    hdr = (
        "<tr>"
        "<th>sample_idx</th>"
        "<th>forced_note</th>"
        "<th>gt_branch_count</th>"
        "<th>bic branch_count</th>"
        "<th>cnn branch_count</th>"
        "<th>gan branch_count</th>"
        "<th>gt_mean_pers</th>"
        "<th>bic mean_pers</th>"
        "<th>cnn mean_pers</th>"
        "<th>gan mean_pers</th>"
        "<th>best_method</th>"
        "<th>panel link</th>"
        "</tr>"
    )

    body_rows: List[str] = []
    for sid in sorted(sample_ids):
        row = per_sample_by_idx.get(sid)
        forced_note = html.escape(FORCED.get(sid, ""))

        if row is None:
            body_rows.append(
                f"<tr>"
                f"<td>{sid}</td>"
                f"<td>{forced_note}</td>"
                + "<td>—</td>" * 9
                + "</tr>"
            )
            continue

        gt_bc  = row.get("gt_branch_count", float("nan"))
        bic_bc = row.get("bicubic_branch_count", float("nan"))
        cnn_bc = row.get("cnn_branch_count", float("nan"))
        gan_bc = row.get("gan_branch_count", float("nan"))

        gt_mp  = row.get("gt_mean_persistence", float("nan"))
        bic_mp = row.get("bicubic_mean_persistence", float("nan"))
        cnn_mp = row.get("cnn_mean_persistence", float("nan"))
        gan_mp = row.get("gan_mean_persistence", float("nan"))

        # Best method: lowest branch_count absolute error
        aes = {
            "bicubic": row.get("bicubic_branch_count_ae", float("inf")),
            "cnn":     row.get("cnn_branch_count_ae",     float("inf")),
            "gan":     row.get("gan_branch_count_ae",     float("inf")),
        }
        try:
            best = min(aes, key=lambda k: aes[k])
        except Exception:
            best = "—"

        link = f'<a href="../../baseline_visual_panels/index.html#sample-{sid}">panel</a>'

        def _win(m: str, val: object) -> str:
            win_cls = ' class="winner-cell"' if m == best else ""
            return f"<td{win_cls}>{_fmt(val, 1)}</td>"

        row_html = (
            f"<tr>"
            f"<td>{sid}</td>"
            f"<td style='font-size:0.78em;color:#555'>{forced_note}</td>"
            f"<td>{_fmt(gt_bc, 1)}</td>"
            + _win("bicubic", bic_bc)
            + _win("cnn", cnn_bc)
            + _win("gan", gan_bc)
            + f"<td>{_fmt(gt_mp, 3)}</td>"
            + _win("bicubic", bic_mp)
            + _win("cnn", cnn_mp)
            + _win("gan", gan_mp)
            + f"<td><strong>{html.escape(str(best))}</strong></td>"
            f"<td>{link}</td>"
            f"</tr>"
        )
        body_rows.append(row_html)

    return (
        f'<div class="card" id="{html.escape(section_id)}">'
        f"<h2>{html.escape(title)}</h2>"
        f"<table class='data-table'>"
        f"<thead>{hdr}</thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        f"</table>"
        f"</div>"
    )


def _build_html_groups(
    special_data: Optional[Dict[int, Dict[str, str]]],
    manifest_data: Optional[Dict[int, Dict[str, str]]],
) -> List[Tuple[str, str, List[int]]]:
    """
    Build the ordered list of (section_id, title, sample_ids) for HTML sections.

    Section order:
    1. All MT-GAN
    2. Cluster 10-13
    3. Cluster 76-78
    4. Cluster 90-93
    5. Rare PD=MT=CNN controls 161-164
    6. Limitation cases 25, 80, 154
    7. GAN-majority but MT-CNN
    """
    sections: List[Tuple[str, str, List[int]]] = []

    # Section 1: All MT-GAN
    mt_gan_ids: List[int] = []
    if special_data is not None:
        mt_gan_ids = [si for si, row in special_data.items() if row.get("group_name") == "all_mt_gan"]
    if not mt_gan_ids and manifest_data is not None:
        # Fallback: look for MT-GAN in manifest
        mt_gan_ids = []
    if not mt_gan_ids:
        # Hardcoded fallback
        mt_gan_ids = [12, 17, 19, 77, 92]
    sections.append(("all_mt_gan", "All MT-GAN Anchors", sorted(mt_gan_ids)))

    # Sections 2-6: static groups
    for sid, stitle, sids in _STATIC_GROUPS:
        sections.append((sid, stitle, sids))

    # Section 7: GAN-majority but MT-CNN
    gan_maj_ids: List[int] = []
    if special_data is not None:
        gan_maj_ids = [si for si, row in special_data.items()
                       if row.get("group_name") == "gan_majority_but_mt_not_gan"]
    if not gan_maj_ids:
        gan_maj_ids = [90, 91, 93]  # hardcoded fallback
    sections.append(("gan_majority_mt_cnn", "GAN-majority but MT-CNN", sorted(gan_maj_ids)))

    return sections


def _write_html(
    out_path: Path,
    summary_rows: List[Dict],
    per_sample: List[Dict],
    special_data: Optional[Dict[int, Dict[str, str]]],
    manifest_data: Optional[Dict[int, Dict[str, str]]],
) -> None:
    per_sample_by_idx = {int(r["sample_idx"]): r for r in per_sample}

    groups = _build_html_groups(special_data, manifest_data)

    # Navigation bar
    nav_links = "".join(
        f'<a href="#{html.escape(sid)}">{html.escape(title)}</a>'
        for sid, title, _ in groups
    )
    nav_html = f'<nav><strong>Jump to:</strong> {nav_links}</nav>'

    # Summary table
    summary_html = _summary_table_html(summary_rows)

    # Group sections
    sections_html = "\n".join(
        _group_section_html(sid, title, ids, per_sample_by_idx)
        for sid, title, ids in groups
    )

    n_samples = len(per_sample)

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Merge-Tree Proxy Report – PhIRE</title>
<style>{CSS}</style>
</head>
<body>
<h1>Merge-Tree Proxy Report – PhIRE</h1>
<p class="intro">
  Hierarchy-aware merge-tree proxy statistics for scalar wind-speed super-resolution.<br>
  GT-percentile thresholds: p1..p99 ({N_THRESHOLDS} levels). Branch sweep levels: {N_BRANCH_LVLS}.<br>
  {n_samples} samples analysed across Bicubic, CNN, GAN baselines.
</p>

{nav_html}

<div class="card">
<h2>Method Summary (mean errors across all {n_samples} samples)</h2>
{summary_html}
</div>

{sections_html}
</body>
</html>
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(page, encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--cnn-dir",       type=Path, default=CNN_DIR)
    ap.add_argument("--gan-dir",       type=Path, default=GAN_DIR)
    ap.add_argument("--bicubic-dir",   type=Path, default=BIC_DIR)
    ap.add_argument("--manifest-csv",  type=Path, default=MANIFEST_CSV)
    ap.add_argument("--special-csv",   type=Path, default=SPECIAL_CSV)
    ap.add_argument("--outdir",        type=Path, default=OUT_DIR)
    ap.add_argument("--n-thresholds",  type=int,  default=N_THRESHOLDS)
    ap.add_argument("--n-branch-lvls", type=int,  default=N_BRANCH_LVLS)
    args = ap.parse_args()

    # Override module-level constants if overridden
    n_thresholds  = args.n_thresholds
    n_branch_lvls = args.n_branch_lvls

    # Check data directories exist
    for method, d in [("bicubic", args.bicubic_dir), ("cnn", args.cnn_dir), ("gan", args.gan_dir)]:
        if not d.exists():
            sys.exit(
                f"[error] Data directory not found: {d}\n"
                f"  Run the {method} inference pipeline first, or adjust --{method}-dir."
            )

    # [1] Loading data
    print("[1] Loading data ...")
    METHOD_DIRS_ARGS = {
        "bicubic": args.bicubic_dir,
        "cnn":     args.cnn_dir,
        "gan":     args.gan_dir,
    }
    loaded: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for m, d in METHOD_DIRS_ARGS.items():
        print(f"  Loading {m} from {d} ...")
        loaded[m] = _load_method(d)

    # Validate alignment
    idx_bic, gt_bic, sr_bic = loaded["bicubic"]
    idx_cnn, gt_cnn, sr_cnn = loaded["cnn"]
    idx_gan, gt_gan, sr_gan = loaded["gan"]

    if not np.array_equal(idx_bic, idx_cnn):
        sys.exit("[error] Bicubic and CNN idx arrays differ.")
    if not np.array_equal(idx_bic, idx_gan):
        sys.exit("[error] Bicubic and GAN idx arrays differ.")

    sample_indices = idx_bic.astype(int).ravel().tolist()
    N = len(sample_indices)
    print(f"  Aligned: {N} samples, idx range=[{min(sample_indices)}..{max(sample_indices)}]")

    # Load optional CSVs
    manifest_data = _load_manifest(args.manifest_csv)
    special_data  = _load_special_csv(args.special_csv)

    # [2] Processing samples
    print(f"[2] Processing {N} samples ...")
    per_sample: List[Dict] = []

    for i, si in enumerate(sample_indices):
        if i % 10 == 0:
            print(f"  Sample {i + 1}/{N}: idx={si} ...")

        gt_uv_raw = gt_bic[i]
        sr_uvs_raw = {
            "bicubic": sr_bic[i],
            "cnn":     sr_cnn[i],
            "gan":     sr_gan[i],
        }
        row = _process_sample(si, gt_uv_raw, sr_uvs_raw)
        per_sample.append(row)

    # [3] Writing CSVs
    print("[3] Writing CSVs ...")
    args.outdir.mkdir(parents=True, exist_ok=True)

    per_sample_path = args.outdir / "merge_tree_proxy_per_sample.csv"
    _write_csv(per_sample_path, per_sample)
    print(f"  Wrote {per_sample_path}")

    summary_rows = _compute_summary(per_sample)
    summary_path = args.outdir / "merge_tree_proxy_summary.csv"
    _write_csv(summary_path, summary_rows)
    print(f"  Wrote {summary_path}")

    # Special cases: union of all group sample ids
    groups_for_special = _build_html_groups(special_data, manifest_data)
    special_ids: set = set()
    for _, _, ids in groups_for_special:
        special_ids.update(ids)
    special_rows = [r for r in per_sample if int(r["sample_idx"]) in special_ids]
    special_path = args.outdir / "special_merge_tree_proxy_summary.csv"
    _write_csv(special_path, special_rows)
    print(f"  Wrote {special_path}")

    # [4] Writing HTML
    print("[4] Writing HTML ...")
    html_path = args.outdir / "index.html"
    _write_html(html_path, summary_rows, per_sample, special_data, manifest_data)
    print(f"  Wrote {html_path}")

    print("Done.")


if __name__ == "__main__":
    main()
