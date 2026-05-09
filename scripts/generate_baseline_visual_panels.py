#!/usr/bin/env python3
"""
Generate a baseline visual-inspection report for TopoAware/PhIRE SR.

This version is intended to replace scripts/generate_baseline_visual_panels.py.
It keeps the white-card visual-inspection UI style, but adds bicubic as a
third visual baseline next to CNN and GAN.

Default behavior:
  - Generates all available samples, usually 168.
  - Writes crop and full-field 7-column panels:
        GT speed | Bicubic speed | CNN speed | GAN speed |
        |Bicubic-GT| | |CNN-GT| | |GAN-GT|
  - Uses shared color limits across all speed panels in a sample.
  - Uses shared error limits across all error panels in a sample.
  - Uses origin='lower' to match the earlier visual-inspection convention.
  - Builds ttk_runs_fixed/baseline_visual_panels/index.html.
  - Adds:
        (1) baseline direct-error summary for Bicubic/CNN/GAN
        (2) three-method physics/domain table from
            ttk_runs_fixed/baseline_metrics/all_methods_per_sample.csv
        (3) original CNN/GAN physics/domain table when available.

Run from scripts/:
    cd ~/PhIRE/scripts
    PYTHONNOUSERSITE=1 /usr/bin/python3 generate_baseline_visual_panels.py

Useful options:
    PYTHONNOUSERSITE=1 /usr/bin/python3 generate_baseline_visual_panels.py --samples 10,11,12,13
    PYTHONNOUSERSITE=1 /usr/bin/python3 generate_baseline_visual_panels.py --no-panels
"""

from __future__ import annotations

import argparse
import csv
import html
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Repo paths
# -----------------------------------------------------------------------------

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
        if (c / "data_out_fixed").exists() or (c / "ttk_runs_fixed").exists():
            return c
    raise FileNotFoundError("Could not locate repo root containing data_out_fixed/ or ttk_runs_fixed/.")


ROOT = repo_root()
DATA_ROOT = ROOT / "data_out_fixed"
CNN_DIR = DATA_ROOT / "wind_mrhr_cnn"
GAN_DIR = DATA_ROOT / "wind_mrhr_gan"
BIC_DIR = DATA_ROOT / "wind_mrhr_bicubic"

OUTDIR = ROOT / "ttk_runs_fixed" / "baseline_visual_panels"
CROP_DIR = OUTDIR / "panels_crop"
FULL_DIR = OUTDIR / "panels_full"

# Original CNN/GAN metadata and metric outputs
CNN_GAN_PHYSICS = ROOT / "ttk_runs_fixed" / "report_tables" / "full_physics_domain_breakdown" / "physics_domain_breakdown_all_samples.csv"
WIDE_TABLE = ROOT / "ttk_runs_fixed" / "report_tables" / "metric_sweep_all_samples_wide.csv"
OBS_PER_SAMPLE = ROOT / "ttk_runs_fixed" / "observation_groups" / "observation_groups_per_sample.csv"
RECOMMENDED_UNIQUE = ROOT / "ttk_runs_fixed" / "observation_groups" / "recommended_visual_inspection_unique_samples.csv"
OLD_MANIFEST = ROOT / "ttk_runs_fixed" / "visual_inspection" / "visual_inspection_manifest.csv"

# New three-method baseline metrics
BASELINE_METRICS = ROOT / "ttk_runs_fixed" / "baseline_metrics" / "all_methods_per_sample.csv"


# -----------------------------------------------------------------------------
# Important qualitative anchors / controls
# -----------------------------------------------------------------------------

FORCED = {
    10: "adjacent control before sample 12",
    11: "adjacent control before sample 12",
    12: "strong MT-GAN anchor",
    13: "adjacent control after sample 12",
    16: "moderate MT-GAN ridge-rich motif",
    17: "strong MT-GAN anchor",
    18: "moderate/lower MT-GAN ridge-rich motif",
    19: "strong MT-GAN anchor",
    20: "moderate/lower MT-GAN ridge-rich motif",
    25: "lower-confidence MT-GAN limitation case",
    76: "adjacent control before sample 77",
    77: "strong MT-GAN anchor",
    78: "adjacent control after sample 77",
    80: "lower-confidence MT-GAN limitation case",
    90: "GAN-majority / MT-CNN adjacent control near sample 92",
    91: "GAN-majority / MT-CNN adjacent control near sample 92",
    92: "strong MT-GAN anchor",
    93: "GAN-majority / MT-CNN adjacent control near sample 92",
    154: "lower-confidence MT-GAN limitation case",
    161: "adjacent control before rare topology-CNN samples 162-163",
    162: "rare PD-CNN and MT-CNN topology-consensus control",
    163: "rare PD-CNN and MT-CNN topology-consensus control",
    164: "adjacent control after rare topology-CNN samples 162-163",
}


PHYSICS_METRICS = [
    ("wpd_bias", "WPD bias |·|", "Physics / WPD", True),
    ("wpd_mae", "WPD MAE", "Physics / WPD", False),
    ("wpd_rmse", "WPD RMSE", "Physics / WPD", False),
    ("wpd_w1", "WPD Wasserstein-1", "Distributional", False),
    ("psd_log_l2", "PSD log-L2", "Distributional", False),
    ("psd_slope_abs_delta",    "PSD slope |Δ|",          "Distributional", True),
    ("grad_mae", "Gradient MAE", "Physics / Gradient", False),
    ("grad_w1", "Gradient Wasserstein-1", "Distributional", False),
    ("grad_kurtosis_abs_delta","Gradient kurtosis |Δ|",   "Distributional", True),
    ("exceed_frac_abs_delta_t5", "Exceedance |Δ|, s > 5", "Tail / Exceedance", False),
    ("exceed_frac_abs_delta_t10", "Exceedance |Δ|, s > 10", "Tail / Exceedance", False),
    ("exceed_frac_abs_delta_t15", "Exceedance |Δ|, s > 15", "Tail / Exceedance", False),
    ("exceed_frac_abs_delta_p90", "Exceedance |Δ|, p90", "Tail / Exceedance", False),
    ("exceed_frac_abs_delta_p95", "Exceedance |Δ|, p95", "Tail / Exceedance", False),
    ("exceed_frac_abs_delta_p99", "Exceedance |Δ|, p99", "Tail / Exceedance", False),
]

METHODS = ("bicubic", "cnn", "gan")
METHOD_LABELS = {"bicubic": "Bicubic", "cnn": "CNN", "gan": "GAN"}


# -----------------------------------------------------------------------------
# General helpers
# -----------------------------------------------------------------------------

def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def sid_from(row: dict[str, str]) -> int:
    for k in ("sample_idx", "sample_id", "sample", "idx", "id", "index"):
        if k in row and str(row[k]).strip():
            return int(float(str(row[k]).strip()))
    raise ValueError("row has no sample id")


def H(x) -> str:
    return html.escape(str(x))


def norm(x) -> str:
    s = str(x or "").strip().upper()
    if s in {"CNN", "GAN", "BICUBIC", "TIE"}:
        return s
    if s in {"BIC", "BICUB"}:
        return "BICUBIC"
    if s in {"TIED", "EQUAL"}:
        return "TIE"
    return s


def boolish(x) -> bool:
    return str(x or "").strip().lower() in {"true", "1", "yes", "y"}


def safe_float(x) -> Optional[float]:
    try:
        v = float(str(x).strip())
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


def fmt_num(x) -> str:
    v = safe_float(x)
    if v is None:
        return ""
    av = abs(v)
    if av == 0:
        return "0"
    if av < 1e-3:
        return f"{v:.3e}"
    if av < 1:
        return f"{v:.4f}"
    if av < 100:
        return f"{v:.3f}"
    return f"{v:.1f}"


def pick(row: dict, obs: dict, *keys: str) -> str:
    for k in keys:
        if k in row and str(row[k]).strip():
            return str(row[k]).strip()
        if k in obs and str(obs[k]).strip():
            return str(obs[k]).strip()
    return ""


# -----------------------------------------------------------------------------
# Metadata loaders
# -----------------------------------------------------------------------------

def load_cnn_gan_rows() -> dict[int, dict[str, str]]:
    """Load the original CNN/GAN wide metrics table, one row per sample."""
    for p in (CNN_GAN_PHYSICS, WIDE_TABLE, OLD_MANIFEST):
        rows = read_csv(p)
        if rows:
            out: dict[int, dict[str, str]] = {}
            for r in rows:
                try:
                    out[sid_from(r)] = r
                except Exception:
                    pass
            print(f"Loaded {len(out)} CNN/GAN metric rows from {p}")
            return out
    print("WARNING: no CNN/GAN metric table found; index will have limited winner metadata.")
    return {}


def load_obs_rows() -> dict[int, dict[str, str]]:
    out: dict[int, dict[str, str]] = {}
    for p in (OBS_PER_SAMPLE, RECOMMENDED_UNIQUE, OLD_MANIFEST):
        for r in read_csv(p):
            try:
                sid = sid_from(r)
            except Exception:
                continue
            out.setdefault(sid, {})
            for k, v in r.items():
                if v is not None and str(v).strip():
                    out[sid][k] = v
    if out:
        print(f"Loaded observation metadata for {len(out)} samples")
    return out


def normalize_method(raw: str) -> Optional[str]:
    s = str(raw or "").strip().lower()
    for m in METHODS:
        if m in s:
            return m
    return None


def load_baseline_metrics(path: Path = BASELINE_METRICS) -> dict[tuple[str, int], dict[str, str]]:
    """Load all_methods_per_sample.csv as long-form rows keyed by (method, sample_idx)."""
    rows = read_csv(path)
    if not rows:
        print(f"WARNING: baseline metric CSV not found or empty: {path}")
        return {}

    cols = list(rows[0].keys())
    method_col = next((c for c in cols if c.strip().lower() in {"method", "model", "name"}), None)
    sample_col = next((c for c in cols if c.strip().lower() in {"sample_idx", "sample", "idx", "index", "id"}), None)
    if method_col is None or sample_col is None:
        print(f"WARNING: baseline metric CSV missing method/sample columns: {path}")
        return {}

    out: dict[tuple[str, int], dict[str, str]] = {}
    skipped = 0
    for r in rows:
        m = normalize_method(r.get(method_col, ""))
        try:
            sid = int(float(str(r.get(sample_col, "")).strip()))
        except Exception:
            sid = -1
        if m is None or sid < 0:
            skipped += 1
            continue
        out[(m, sid)] = r

    print(f"Loaded {len(out)} all-baseline metric rows from {path}")
    if skipped:
        print(f"  skipped {skipped} unrecognized baseline metric rows")
    return out


# -----------------------------------------------------------------------------
# Array loading / panels
# -----------------------------------------------------------------------------

def load_method_arrays(directory: Path) -> tuple[np.ndarray, Optional[np.ndarray], np.ndarray, np.ndarray]:
    missing = [name for name in ("idx.npy", "dataGT.npy", "dataSR.npy") if not (directory / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required files in {directory}: {missing}")
    idx = np.load(directory / "idx.npy")
    data_in = np.load(directory / "dataIN.npy", mmap_mode="r") if (directory / "dataIN.npy").exists() else None
    data_gt = np.load(directory / "dataGT.npy", mmap_mode="r")
    data_sr = np.load(directory / "dataSR.npy", mmap_mode="r")
    return idx, data_in, data_gt, data_sr


def load_arrays():
    cnn_idx, cnn_in, cnn_gt, cnn_sr = load_method_arrays(CNN_DIR)
    gan_idx, gan_in, gan_gt, gan_sr = load_method_arrays(GAN_DIR)
    bic_idx, bic_in, bic_gt, bic_sr = load_method_arrays(BIC_DIR)

    if not np.array_equal(cnn_idx, gan_idx):
        raise ValueError("CNN and GAN idx arrays differ.")
    if not np.array_equal(cnn_idx, bic_idx):
        raise ValueError("CNN and bicubic idx arrays differ.")

    max_cnn_gan = float(np.max(np.abs(cnn_gt[:] - gan_gt[:])))
    max_cnn_bic = float(np.max(np.abs(cnn_gt[:] - bic_gt[:])))
    if max_cnn_gan > 1e-9 or max_cnn_bic > 1e-9:
        print(f"WARNING: GT arrays differ: vs GAN={max_cnn_gan:.3e}, vs bicubic={max_cnn_bic:.3e}")
    else:
        print("Array alignment verified:")
        print(f"  idx shape/range: {cnn_idx.shape}, {int(cnn_idx.min())} … {int(cnn_idx.max())}")
        print(f"  GT shape:        {cnn_gt.shape}")
        print(f"  Bicubic SR:      {bic_sr.shape}")
        print(f"  CNN SR:          {cnn_sr.shape}")
        print(f"  GAN SR:          {gan_sr.shape}")
        print(f"  GT max diff vs GAN={max_cnn_gan:.2e}, vs bicubic={max_cnn_bic:.2e}")

    pos = {int(v): i for i, v in enumerate(cnn_idx.tolist())}
    arrays = {
        "idx": cnn_idx,
        "pos": pos,
        "gt": cnn_gt,
        "bicubic": bic_sr,
        "cnn": cnn_sr,
        "gan": gan_sr,
    }
    return arrays


def speed(a: np.ndarray) -> np.ndarray:
    if a.ndim == 3 and a.shape[-1] == 2:
        return np.sqrt(a[..., 0].astype(np.float32) ** 2 + a[..., 1].astype(np.float32) ** 2)
    if a.ndim == 3 and a.shape[-1] == 1:
        return a[..., 0].astype(np.float32)
    if a.ndim == 2:
        return a.astype(np.float32)
    raise ValueError(f"Unexpected sample shape: {a.shape}")


def scalar_stats(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    diff = pred.astype(np.float64) - gt.astype(np.float64)
    return {
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
    }


def panel_title(sid: int, row: dict, obs: dict) -> str:
    return (
        f"sample {sid} | "
        f"SSIM={norm(pick(row, obs, 'ssim_winner')) or '?'} | "
        f"PD={norm(pick(row, obs, 'pd_winner')) or '?'} | "
        f"MT={norm(pick(row, obs, 'mt_winner')) or '?'} | "
        f"direct={norm(pick(row, obs, 'direct_error_group_winner')) or '?'} | "
        f"dist={norm(pick(row, obs, 'distributional_group_winner')) or row.get('overall_metric_majority', '?')}"
    )


def get_sample_speeds(sid: int, arrays: dict, crop: Optional[tuple[int, int, int, int]] = None):
    pos = arrays["pos"]
    if sid not in pos:
        raise KeyError(f"sample {sid} not found in idx.npy")
    i = pos[sid]
    gt_s = speed(np.asarray(arrays["gt"][i]))
    bic_s = speed(np.asarray(arrays["bicubic"][i]))
    cnn_s = speed(np.asarray(arrays["cnn"][i]))
    gan_s = speed(np.asarray(arrays["gan"][i]))

    if crop is not None:
        y0, y1, x0, x1 = crop
        gt_s = gt_s[y0:y1, x0:x1]
        bic_s = bic_s[y0:y1, x0:x1]
        cnn_s = cnn_s[y0:y1, x0:x1]
        gan_s = gan_s[y0:y1, x0:x1]

    return gt_s, bic_s, cnn_s, gan_s


def make_panel(
    sid: int,
    arrays: dict,
    row: dict,
    obs: dict,
    crop: Optional[tuple[int, int, int, int]],
    out: Path,
) -> tuple[bool, dict[str, dict[str, float]]]:
    if sid not in arrays["pos"]:
        print(f"WARNING: sample {sid} not found in idx.npy; skipping panel.")
        return False, {}

    gt_s, bic_s, cnn_s, gan_s = get_sample_speeds(sid, arrays, crop)

    desc = "full field"
    if crop is not None:
        y0, y1, x0, x1 = crop
        desc = f"crop y={y0}:{y1}, x={x0}:{x1}"

    err_bic = np.abs(bic_s - gt_s)
    err_cnn = np.abs(cnn_s - gt_s)
    err_gan = np.abs(gan_s - gt_s)

    speed_fields = [gt_s, bic_s, cnn_s, gan_s]
    vmin = float(min(np.nanmin(a) for a in speed_fields))
    vmax = float(max(np.nanmax(a) for a in speed_fields))
    emax = float(max(np.nanmax(err_bic), np.nanmax(err_cnn), np.nanmax(err_gan)))
    if not np.isfinite(emax) or emax <= 0:
        emax = 1.0

    stats = {
        "bicubic": scalar_stats(bic_s, gt_s),
        "cnn": scalar_stats(cnn_s, gt_s),
        "gan": scalar_stats(gan_s, gt_s),
    }

    fig, axes = plt.subplots(1, 7, figsize=(30, 5.2))
    fields = [gt_s, bic_s, cnn_s, gan_s, err_bic, err_cnn, err_gan]
    titles = ["GT speed", "Bicubic speed", "CNN speed", "GAN speed", "|Bicubic-GT|", "|CNN-GT|", "|GAN-GT|"]

    for ax, field, title in zip(axes, fields, titles):
        if "GT|" in title:
            im = ax.imshow(field, origin="lower", vmin=0, vmax=emax)
        else:
            im = ax.imshow(field, origin="lower", vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    fig.suptitle(panel_title(sid, row, obs), fontsize=12)
    fig.text(
        0.5,
        0.02,
        (
            f"{desc}; speed RMSE "
            f"bicubic={stats['bicubic']['rmse']:.3f}, "
            f"CNN={stats['cnn']['rmse']:.3f}, GAN={stats['gan']['rmse']:.3f}"
        ),
        ha="center",
        fontsize=9,
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    fig.savefig(out, dpi=125)
    plt.close(fig)
    return True, stats


# -----------------------------------------------------------------------------
# Winner/group logic
# -----------------------------------------------------------------------------

def cnn_gan_counts(row: dict) -> tuple[int, int, int]:
    cnn = gan = tie = 0
    for key, _label, _group, _abs in PHYSICS_METRICS:
        w = norm(row.get(f"{key}_winner", ""))
        if w == "CNN":
            cnn += 1
        elif w == "GAN":
            gan += 1
        elif w == "TIE":
            tie += 1
    return cnn, gan, tie


def baseline_value(base_rows: dict[tuple[str, int], dict[str, str]], sid: int, method: str, key: str, needs_abs: bool = False) -> Optional[float]:
    row = base_rows.get((method, sid))
    if not row:
        return None

    candidates = [key]
    # Useful fallbacks for slightly different CSV conventions.
    if key.endswith("_abs_delta"):
        candidates.append(key.replace("_abs_delta", "_delta"))
    if "_abs_delta" not in key and key.endswith("_delta"):
        candidates.append(key.replace("_delta", "_abs_delta"))

    for c in candidates:
        if c in row and str(row[c]).strip() != "":
            v = safe_float(row[c])
            if v is not None:
                return abs(v) if needs_abs else v
    return None


def three_method_counts(base_rows: dict[tuple[str, int], dict[str, str]], sid: int) -> dict[str, int]:
    counts = {"bicubic": 0, "cnn": 0, "gan": 0, "tie": 0}
    for key, _label, _group, needs_abs in PHYSICS_METRICS:
        vals = {m: baseline_value(base_rows, sid, m, key, needs_abs) for m in METHODS}
        finite = {m: v for m, v in vals.items() if v is not None}
        if not finite:
            continue
        minv = min(finite.values())
        winners = [m for m, v in finite.items() if abs(v - minv) <= 1e-12]
        if len(winners) == 1:
            counts[winners[0]] += 1
        else:
            counts["tie"] += 1
    return counts


def infer_groups(sid: int, row: dict, obs: dict, base_rows: dict[tuple[str, int], dict[str, str]]) -> list[str]:
    groups = set()

    for source in (row, obs):
        raw = str(source.get("groups", ""))
        for g in raw.replace(",", ";").split(";"):
            g = g.strip()
            if g:
                groups.add(g)
        for k, v in source.items():
            if k.startswith("group_") and boolish(v):
                groups.add(k[len("group_"):])

    pd = norm(pick(row, obs, "pd_winner"))
    mt = norm(pick(row, obs, "mt_winner"))
    direct = norm(pick(row, obs, "direct_error_group_winner"))
    dist = norm(pick(row, obs, "distributional_group_winner")) or norm(row.get("overall_metric_majority", ""))
    tail = norm(pick(row, obs, "tail_group_winner"))
    physics = norm(pick(row, obs, "configured_physics_group_winner", "physics_group_winner"))

    if mt == "GAN":
        groups.add("mt_gan_diagnostic")
        groups.add("mt_gan_all")
    elif mt == "CNN":
        groups.add("mt_cnn_all")
    if pd == "GAN":
        groups.add("pd_gan_all")
    elif pd == "CNN":
        groups.add("pd_cnn_all")

    if pd == "GAN" and mt == "GAN":
        groups.add("topology_consensus_gan")
    if pd == "GAN" and mt == "CNN":
        groups.add("pd_gan_mt_cnn_control")
        groups.add("candidate_structural_hallucination_signature")
    if pd == "CNN" and mt == "CNN":
        groups.add("topology_consensus_cnn")

    for name, val in [("direct", direct), ("distributional", dist), ("tail", tail), ("physics", physics)]:
        if val in {"CNN", "GAN"}:
            groups.add(f"{name}_{val.lower()}")

    old_cnn, old_gan, _old_tie = cnn_gan_counts(row)
    if old_gan > old_cnn:
        groups.add("gan_metric_majority")
        if mt != "GAN":
            groups.add("gan_majority_mt_rejects_gan")

    base_counts = three_method_counts(base_rows, sid)
    if base_counts["bicubic"] > max(base_counts["cnn"], base_counts["gan"]):
        groups.add("bicubic_metric_majority")
    if base_counts["gan"] > max(base_counts["cnn"], base_counts["bicubic"]):
        groups.add("gan_three_method_metric_majority")
    if base_counts["cnn"] > max(base_counts["gan"], base_counts["bicubic"]):
        groups.add("cnn_three_method_metric_majority")

    if sid in FORCED:
        groups.add("forced_qualitative_set")
    if sid in {10, 11, 12, 13}:
        groups.add("adjacent_cluster_10_13")
    if sid in {76, 77, 78}:
        groups.add("adjacent_cluster_76_78")
    if sid in {90, 91, 92, 93}:
        groups.add("adjacent_cluster_90_93")
    if sid in {161, 162, 163, 164}:
        groups.add("adjacent_cluster_161_164")

    groups.add("psnr_cnn")
    groups.add("ssim_cnn")

    return sorted(groups)


def question(sid: int, row: dict, obs: dict) -> str:
    pd = norm(pick(row, obs, "pd_winner"))
    mt = norm(pick(row, obs, "mt_winner"))
    old_cnn, old_gan, _ = cnn_gan_counts(row)
    gan_majority = old_gan > old_cnn

    if sid in {90, 91, 92, 93}:
        return "Adjacent transition cluster: why does MT accept GAN in sample 92 but reject nearby GAN-majority samples?"
    if sid in {10, 11, 12, 13}:
        return "Adjacent transition cluster around sample 12: what visual/topological change makes MT favor GAN?"
    if sid in {76, 77, 78}:
        return "Neighbor controls around sample 77: is the MT-GAN selection locally stable or sample-specific?"
    if sid in {161, 162, 163, 164}:
        return "Rare topology-CNN control neighborhood: when do both PD and MT reject GAN texture?"
    if pd == "CNN" and mt == "CNN":
        return "Rare topology-CNN control: why do both topological descriptors prefer CNN?"
    if mt == "GAN" and gan_majority:
        return "Strong MT-GAN candidate: does GAN preserve meaningful multiscale/topological structure?"
    if mt == "GAN":
        return "MT favors GAN: is GAN structurally closer to GT, or sharper but misaligned?"
    if pd == "GAN" and mt == "CNN" and gan_majority:
        return "GAN wins many domain metrics but MT favors CNN: is GAN distributionally plausible but hierarchically misaligned?"
    if pd == "GAN" and mt == "CNN":
        return "PD favors GAN but MT favors CNN: are GAN features plausible but hierarchically/spatially misaligned?"
    return "Control sample: compare bicubic interpolation, conservative CNN fidelity, GAN texture, and topology choices."


def select_samples(args: argparse.Namespace, arrays: Optional[dict], cnn_gan_rows: dict[int, dict]) -> list[int]:
    if args.samples:
        return sorted({int(x.strip()) for x in args.samples.split(",") if x.strip()})
    if args.default_subset:
        return sorted(FORCED.keys())
    if arrays is not None:
        return sorted(int(x) for x in arrays["idx"].tolist())
    if cnn_gan_rows:
        return sorted(cnn_gan_rows)
    return list(range(168))


# -----------------------------------------------------------------------------
# HTML fragments
# -----------------------------------------------------------------------------

def badge(label: str, cls: str) -> str:
    return f'<span class="win {H(cls)}">{H(label)}</span>'


def direct_error_table(stats: dict[str, dict[str, float]]) -> str:
    if not stats:
        return """
        <details class="metric-box" open>
          <summary><b>Baseline direct-error summary</b> <span class="warn">unavailable</span></summary>
          <p class="muted">No scalar-speed error stats were computed for this panel.</p>
        </details>
        """

    rows = []
    for metric, label in [("mae", "Speed MAE"), ("rmse", "Speed RMSE")]:
        vals = {m: stats.get(m, {}).get(metric, float("nan")) for m in METHODS}
        finite = {m: v for m, v in vals.items() if np.isfinite(v)}
        minv = min(finite.values()) if finite else float("nan")
        winners = [m for m, v in finite.items() if abs(v - minv) <= 1e-12]
        w = winners[0] if len(winners) == 1 else "tie"
        rows.append(
            "<tr>"
            f"<td>{H(label)}</td>"
            f"<td class='num'>{fmt_num(vals['bicubic'])}</td>"
            f"<td class='num'>{fmt_num(vals['cnn'])}</td>"
            f"<td class='num'>{fmt_num(vals['gan'])}</td>"
            f"<td>{winner_badge(w)}</td>"
            "</tr>"
        )

    return f"""
    <details class="metric-box" open>
      <summary><b>Baseline direct-error summary</b>
        <span class="count">lower is better; computed from the crop panel</span>
      </summary>
      <table class="metrics compact">
        <thead><tr><th>Measure</th><th>Bicubic</th><th>CNN</th><th>GAN</th><th>Winner</th></tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
    </details>
    """


def winner_badge(w: str) -> str:
    w = str(w or "").lower()
    if w == "cnn":
        return badge("CNN", "cnn")
    if w == "gan":
        return badge("GAN", "gan")
    if w in {"bicubic", "bic"}:
        return badge("Bicubic", "bicubic")
    if w == "tie":
        return badge("TIE", "tie")
    return "?"


def three_method_physics_table(sid: int, base_rows: dict[tuple[str, int], dict[str, str]]) -> str:
    if not base_rows or all((m, sid) not in base_rows for m in METHODS):
        return """
        <details class="metric-box">
          <summary><b>Three-method physics/domain breakdown</b> <span class="warn">unavailable</span></summary>
          <p class="muted">Could not find baseline metric rows for Bicubic/CNN/GAN.</p>
        </details>
        """

    body = []
    counts = {"bicubic": 0, "cnn": 0, "gan": 0, "tie": 0}
    for key, label, group, needs_abs in PHYSICS_METRICS:
        vals = {m: baseline_value(base_rows, sid, m, key, needs_abs) for m in METHODS}
        finite = {m: v for m, v in vals.items() if v is not None}
        if finite:
            minv = min(finite.values())
            winners = [m for m, v in finite.items() if abs(v - minv) <= 1e-12]
            w = winners[0] if len(winners) == 1 else "tie"
            counts[w] += 1
        else:
            w = ""

        body.append(
            "<tr>"
            f"<td>{H(label)}</td>"
            f"<td>{H(group)}</td>"
            f"<td class='num'>{H(fmt_num(vals['bicubic']))}</td>"
            f"<td class='num'>{H(fmt_num(vals['cnn']))}</td>"
            f"<td class='num'>{H(fmt_num(vals['gan']))}</td>"
            f"<td>{winner_badge(w)}</td>"
            "</tr>"
        )

    return f"""
    <details class="metric-box">
      <summary><b>Three-method physics/domain breakdown</b>
        <span class="count">Bicubic {counts['bicubic']} | CNN {counts['cnn']} | GAN {counts['gan']} | ties {counts['tie']}</span>
      </summary>
      <p class="muted">Lower is better. Signed quantities use absolute values for comparison when needed.</p>
      <table class="metrics">
        <thead><tr><th>Measure</th><th>Group</th><th>Bicubic</th><th>CNN</th><th>GAN</th><th>Winner</th></tr></thead>
        <tbody>{''.join(body)}</tbody>
      </table>
    </details>
    """


def cnn_gan_physics_table(row: dict) -> str:
    if not row:
        return """
        <details class="metric-box">
          <summary><b>CNN/GAN physics-domain breakdown</b> <span class="warn">unavailable</span></summary>
          <p class="muted">Could not find original paired CNN/GAN metric columns.</p>
        </details>
        """

    body = []
    cnn_count = gan_count = tie_count = 0

    for key, label, group, _needs_abs in PHYSICS_METRICS:
        w = norm(row.get(f"{key}_winner", ""))
        if w == "CNN":
            cnn_count += 1
            b = winner_badge("cnn")
        elif w == "GAN":
            gan_count += 1
            b = winner_badge("gan")
        elif w == "TIE":
            tie_count += 1
            b = winner_badge("tie")
        else:
            b = "?"

        body.append(
            "<tr>"
            f"<td>{H(label)}</td>"
            f"<td>{H(group)}</td>"
            f"<td class='num'>{H(fmt_num(row.get(key + '_cnn', '')))}</td>"
            f"<td class='num'>{H(fmt_num(row.get(key + '_gan', '')))}</td>"
            f"<td>{b}</td>"
            "</tr>"
        )

    return f"""
    <details class="metric-box">
      <summary><b>CNN/GAN physics-domain breakdown</b>
        <span class="count">CNN {cnn_count} | GAN {gan_count} | ties {tie_count}</span>
      </summary>
      <p class="muted">This is the original paired CNN/GAN physics table; the bicubic table above adds the classical interpolation baseline.</p>
      <table class="metrics">
        <thead><tr><th>Measure</th><th>Group</th><th>CNN</th><th>GAN</th><th>Winner</th></tr></thead>
        <tbody>{''.join(body)}</tbody>
      </table>
    </details>
    """


def card(entry: dict) -> str:
    sid = entry["sample_idx"]
    row = entry["row"]
    obs = entry["obs"]
    groups = entry["groups"]
    crop = entry["crop_panel"]
    full = entry["full_panel"]
    base_rows = entry["base_rows"]
    stats = entry.get("crop_stats", {})

    cls = " ".join("tag-" + g.replace("_", "-") for g in groups)
    chips = " ".join(f"<span class='chip'>{H(g.replace('_', ' '))}</span>" for g in groups)

    winners = (
        f"PSNR: {H(norm(pick(row, obs, 'psnr_winner')) or '?')} | "
        f"SSIM: {H(norm(pick(row, obs, 'ssim_winner')) or '?')} | "
        f"PD: {H(norm(pick(row, obs, 'pd_winner')) or '?')} | "
        f"MT: {H(norm(pick(row, obs, 'mt_winner')) or '?')} | "
        f"Direct: {H(norm(pick(row, obs, 'direct_error_group_winner')) or '?')} | "
        f"Distributional: {H(norm(pick(row, obs, 'distributional_group_winner')) or row.get('overall_metric_majority', '?'))} | "
        f"Tail: {H(norm(pick(row, obs, 'tail_group_winner')) or '?')} | "
        f"Physics: {H(norm(pick(row, obs, 'configured_physics_group_winner', 'physics_group_winner')) or '?')}"
    )

    links = []
    if crop:
        links.append(f"<a href='{H(crop)}' target='_blank'>Open crop panel</a>")
    if full:
        links.append(f"<a href='{H(full)}' target='_blank'>Open full panel</a>")

    note = FORCED.get(sid, "")

    return f"""
    <section class="card {cls}" id="sample-{sid}">
      <div class="card-grid">
        <div>
          <h2>Sample {sid}</h2>
          <div class="winner-line">{winners}</div>

          <h3>Question</h3>
          <p>{H(entry['question'])}</p>

          {f"<p class='forced'><b>Added because:</b> {H(note)}</p>" if note else ""}

          <h3>Groups</h3>
          <div>{chips}</div>

          {direct_error_table(stats)}
          {three_method_physics_table(sid, base_rows)}
          {cnn_gan_physics_table(row)}

          <div class="links">{' '.join(links)}</div>
        </div>
        <div class="thumb">
          {f"<a href='{H(crop)}' target='_blank'><img src='{H(crop)}' alt='sample {sid} crop panel'></a>" if crop else "<p class='muted'>No panel available.</p>"}
        </div>
      </div>
    </section>
    """


def write_index(entries: list[dict]) -> None:
    def count(group: str) -> int:
        return sum(group in e["groups"] for e in entries)

    cards = "\n".join(card(e) for e in entries)

    page = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>TopoAware SR baseline visual inspection index</title>
<style>
body {{ margin:0; background:#f7f7f8; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; color:#111; }}
header {{ position:sticky; top:0; z-index:10; background:white; border-bottom:1px solid #ddd; padding:16px 24px; }}
h1 {{ margin:0 0 10px 0; font-size:30px; }}
button {{ border:1px solid #ddd; background:white; border-radius:999px; padding:8px 12px; margin:0 6px 6px 0; cursor:pointer; }}
main {{ padding:16px; }}
.card {{ background:white; border:1px solid #ddd; border-radius:14px; padding:18px; margin-bottom:18px; box-shadow:0 1px 4px rgba(0,0,0,.06); }}
.card-grid {{ display:grid; grid-template-columns:minmax(520px,1fr) minmax(420px,.9fr); gap:20px; align-items:start; }}
h2 {{ font-size:26px; margin:0 0 10px 0; }}
h3 {{ font-size:16px; margin:18px 0 8px 0; }}
.winner-line {{ display:inline-block; background:#f5f5f5; border:1px solid #ddd; border-radius:8px; padding:10px 12px; }}
.forced {{ background:#fff8e7; border-left:4px solid #ff9f1a; padding:8px 10px; border-radius:6px; }}
.chip {{ display:inline-block; background:#eef5ff; border:1px solid #b9d1ff; border-radius:999px; padding:6px 10px; margin:0 6px 8px 0; }}
.metric-box {{ margin-top:16px; border:1px solid #ddd; border-radius:10px; padding:10px 12px; background:#fcfcfc; }}
.metric-box summary {{ cursor:pointer; }}
.count {{ margin-left:12px; color:#333; font-size:14px; }}
.muted {{ color:#666; }}
.metrics {{ border-collapse:collapse; width:100%; margin-top:10px; font-size:13px; }}
.metrics.compact {{ max-width:760px; }}
.metrics th {{ text-align:left; background:#f0f0f0; padding:6px; }}
.metrics td {{ border-top:1px solid #e5e5e5; padding:6px; }}
.num {{ text-align:right; font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; }}
.win {{ display:inline-block; border-radius:999px; padding:3px 8px; font-weight:700; font-size:12px; }}
.cnn {{ background:#dff7e9; color:#11733b; border:1px solid #a8e6c1; }}
.gan {{ background:#fff0d9; color:#a35b00; border:1px solid #ffd39a; }}
.bicubic {{ background:#e8e0ff; color:#4c1d95; border:1px solid #c4b5fd; }}
.tie {{ background:#eee; color:#555; border:1px solid #ccc; }}
.warn {{ background:yellow; padding:2px 4px; }}
.links {{ margin-top:14px; display:flex; gap:14px; flex-wrap:wrap; }}
.links a {{ color:#0057b8; font-weight:700; text-decoration:none; }}
.thumb {{ position:sticky; top:118px; }}
.thumb img {{ max-width:100%; border:1px solid #ddd; border-radius:8px; background:white; }}
@media(max-width:1200px) {{ .card-grid {{ grid-template-columns:1fr; }} .thumb {{ position:static; }} }}
</style>
<script>
function showOnly(cls) {{
  document.querySelectorAll('.card').forEach(c => {{
    c.style.display = cls === 'all' || c.classList.contains(cls) ? '' : 'none';
  }});
}}
</script>
</head>
<body>
<header>
  <h1>TopoAware SR baseline visual inspection index</h1>
  <button onclick="showOnly('all')">All ({len(entries)})</button>
  <button onclick="showOnly('tag-forced-qualitative-set')">Forced qualitative set ({count('forced_qualitative_set')})</button>
  <button onclick="showOnly('tag-mt-gan-diagnostic')">MT picks GAN ({count('mt_gan_diagnostic')})</button>
  <button onclick="showOnly('tag-topology-consensus-cnn')">PD=MT=CNN ({count('topology_consensus_cnn')})</button>
  <button onclick="showOnly('tag-gan-metric-majority')">GAN metric majority, CNN/GAN ({count('gan_metric_majority')})</button>
  <button onclick="showOnly('tag-gan-majority-mt-rejects-gan')">GAN majority but MT≠GAN ({count('gan_majority_mt_rejects_gan')})</button>
  <button onclick="showOnly('tag-bicubic-metric-majority')">Bicubic metric majority ({count('bicubic_metric_majority')})</button>
  <button onclick="showOnly('tag-adjacent-cluster-10-13')">Cluster 10–13</button>
  <button onclick="showOnly('tag-adjacent-cluster-76-78')">Cluster 76–78</button>
  <button onclick="showOnly('tag-adjacent-cluster-90-93')">Cluster 90–93</button>
  <button onclick="showOnly('tag-adjacent-cluster-161-164')">Cluster 161–164</button>
  <p class="muted">Each panel shows GT speed | Bicubic speed | CNN speed | GAN speed | |Bicubic-GT| | |CNN-GT| | |GAN-GT|.</p>
</header>
<main>
{cards}
</main>
</body>
</html>
"""
    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / "index.html").write_text(page, encoding="utf-8")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate baseline visual-inspection panels for Bicubic/CNN/GAN.")
    parser.add_argument("--samples", default="", help="Comma-separated sample ids. Default: all samples in idx.npy.")
    parser.add_argument("--default-subset", action="store_true", help="Generate only the forced qualitative subset.")
    parser.add_argument("--no-panels", action="store_true", help="Only rebuild index/manifest using existing PNGs.")
    parser.add_argument("--metrics-csv", type=Path, default=BASELINE_METRICS, help="Three-method all_methods_per_sample.csv")
    args = parser.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    CROP_DIR.mkdir(parents=True, exist_ok=True)
    FULL_DIR.mkdir(parents=True, exist_ok=True)

    cnn_gan_rows = load_cnn_gan_rows()
    obs_rows = load_obs_rows()
    base_rows = load_baseline_metrics(args.metrics_csv)

    arrays = None
    if not args.no_panels:
        arrays = load_arrays()
    else:
        try:
            arrays = load_arrays()
        except Exception:
            arrays = None

    samples = select_samples(args, arrays, cnn_gan_rows)
    if not samples:
        raise RuntimeError("No samples selected.")

    print(f"repo_root={ROOT}")
    print(f"outdir={OUTDIR}")
    print(f"selected_samples={len(samples)}")
    print("forced/extra samples:", " ".join(map(str, sorted(FORCED))))

    entries = []
    manifest = []

    for sid in samples:
        row = cnn_gan_rows.get(sid, {})
        ob = obs_rows.get(sid, {})

        crop_path = CROP_DIR / f"sample_{sid:03d}_crop.png"
        full_path = FULL_DIR / f"sample_{sid:03d}_full.png"

        crop_ok = crop_path.exists()
        full_ok = full_path.exists()
        crop_stats = {}
        full_stats = {}

        if not args.no_panels:
            if arrays is None:
                raise RuntimeError("Arrays were not loaded but panel generation was requested.")
            crop_ok, crop_stats = make_panel(sid, arrays, row, ob, (0, 160, 0, 160), crop_path)
            full_ok, full_stats = make_panel(sid, arrays, row, ob, None, full_path)
        elif arrays is not None:
            try:
                gt_s, bic_s, cnn_s, gan_s = get_sample_speeds(sid, arrays, (0, 160, 0, 160))
                crop_stats = {
                    "bicubic": scalar_stats(bic_s, gt_s),
                    "cnn": scalar_stats(cnn_s, gt_s),
                    "gan": scalar_stats(gan_s, gt_s),
                }
            except Exception:
                crop_stats = {}

        crop_rel = crop_path.relative_to(OUTDIR).as_posix() if crop_ok else ""
        full_rel = full_path.relative_to(OUTDIR).as_posix() if full_ok else ""

        groups = infer_groups(sid, row, ob, base_rows)
        q = question(sid, row, ob)
        old_cnn, old_gan, old_tie = cnn_gan_counts(row)
        base_counts = three_method_counts(base_rows, sid)

        entry = {
            "sample_idx": sid,
            "row": row,
            "obs": ob,
            "base_rows": base_rows,
            "groups": groups,
            "question": q,
            "crop_panel": crop_rel,
            "full_panel": full_rel,
            "crop_stats": crop_stats,
            "full_stats": full_stats,
        }
        entries.append(entry)

        manifest.append({
            "sample_idx": sid,
            "psnr_winner": norm(pick(row, ob, "psnr_winner")),
            "ssim_winner": norm(pick(row, ob, "ssim_winner")),
            "pd_winner": norm(pick(row, ob, "pd_winner")),
            "mt_winner": norm(pick(row, ob, "mt_winner")),
            "direct_error_group_winner": norm(pick(row, ob, "direct_error_group_winner")),
            "distributional_group_winner": norm(pick(row, ob, "distributional_group_winner")) or row.get("overall_metric_majority", ""),
            "tail_group_winner": norm(pick(row, ob, "tail_group_winner")),
            "configured_physics_group_winner": norm(pick(row, ob, "configured_physics_group_winner", "physics_group_winner")),
            "cnn_gan_cnn_metric_wins": old_cnn,
            "cnn_gan_gan_metric_wins": old_gan,
            "cnn_gan_ties": old_tie,
            "baseline_bicubic_metric_wins": base_counts["bicubic"],
            "baseline_cnn_metric_wins": base_counts["cnn"],
            "baseline_gan_metric_wins": base_counts["gan"],
            "baseline_metric_ties": base_counts["tie"],
            "crop_mae_bicubic": crop_stats.get("bicubic", {}).get("mae", ""),
            "crop_mae_cnn": crop_stats.get("cnn", {}).get("mae", ""),
            "crop_mae_gan": crop_stats.get("gan", {}).get("mae", ""),
            "crop_rmse_bicubic": crop_stats.get("bicubic", {}).get("rmse", ""),
            "crop_rmse_cnn": crop_stats.get("cnn", {}).get("rmse", ""),
            "crop_rmse_gan": crop_stats.get("gan", {}).get("rmse", ""),
            "question": q,
            "groups": ";".join(groups),
            "crop_panel": crop_rel,
            "full_panel": full_rel,
            "forced_reason": FORCED.get(sid, ""),
        })

    entries.sort(key=lambda e: e["sample_idx"])
    manifest.sort(key=lambda r: int(r["sample_idx"]))

    write_csv(
        OUTDIR / "baseline_visual_manifest.csv",
        manifest,
        [
            "sample_idx", "psnr_winner", "ssim_winner", "pd_winner", "mt_winner",
            "direct_error_group_winner", "distributional_group_winner",
            "tail_group_winner", "configured_physics_group_winner",
            "cnn_gan_cnn_metric_wins", "cnn_gan_gan_metric_wins", "cnn_gan_ties",
            "baseline_bicubic_metric_wins", "baseline_cnn_metric_wins", "baseline_gan_metric_wins", "baseline_metric_ties",
            "crop_mae_bicubic", "crop_mae_cnn", "crop_mae_gan",
            "crop_rmse_bicubic", "crop_rmse_cnn", "crop_rmse_gan",
            "question", "groups", "crop_panel", "full_panel", "forced_reason",
        ],
    )
    write_index(entries)

    print(f"Wrote {OUTDIR / 'index.html'}")
    print(f"Wrote {OUTDIR / 'baseline_visual_manifest.csv'}")
    print(f"Wrote panels under {CROP_DIR} and {FULL_DIR}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
