#!/usr/bin/env python3
"""
Replacement visual-inspection generator for TopoAware SR.

Run from the scripts directory:

    cd ~/PhIRE/scripts
    PYTHONNOUSERSITE=1 /usr/bin/python3 generate_visual_inspection_panels.py

What this version changes:
  - Keeps samples already present in ttk_runs_fixed/visual_inspection/visual_inspection_manifest.csv
    when that file exists.
  - Force-adds the extra qualitative/adjacent-control samples:
        10, 11, 13, 76, 78, 90, 93, 161, 164
    plus the anchor/context samples:
        12, 16, 17, 18, 19, 20, 25, 77, 80, 91, 92, 154, 162, 163.
  - Regenerates crop and full-field PNG panels.
  - Adds Candidate UV, Candidate UV-expanded-672, Candidate B, Candidate B-expanded-672, Candidate C, Candidate C-expanded-672, Candidate C-expanded-1344, Candidate C-expanded-2688, Candidate UV-expanded-1344, Candidate UV-expanded-2688, Candidate D, Candidate Dpd-expanded-672, Candidate E2, and Candidate E2-expanded-672 panels and topology comparison metadata when available.
  - Rebuilds ttk_runs_fixed/visual_inspection/index.html with physics/domain breakdowns.

Optional:
    PYTHONNOUSERSITE=1 /usr/bin/python3 generate_visual_inspection_panels.py --samples 6,18,20,25,62,63,65,68,79,80,92
    PYTHONNOUSERSITE=1 /usr/bin/python3 generate_visual_inspection_panels.py --samples 90,91,92,93
    PYTHONNOUSERSITE=1 /usr/bin/python3 generate_visual_inspection_panels.py --all
    PYTHONNOUSERSITE=1 /usr/bin/python3 generate_visual_inspection_panels.py --no-panels
"""

from __future__ import annotations

import argparse
import csv
import html
import sys
from pathlib import Path
from typing import Optional

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------------
# Repo paths
# -----------------------------

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
        if (c / "ttk_runs_fixed").exists() or (c / "data_out").exists():
            return c
    raise FileNotFoundError("Could not locate repo root containing ttk_runs_fixed/ or data_out/.")


ROOT = repo_root()
OUTDIR = ROOT / "ttk_runs_fixed" / "visual_inspection"
CROP_DIR = OUTDIR / "panels_crop"
FULL_DIR = OUTDIR / "panels_full"

def first_existing(*paths: Path) -> Path:
    """Return the first existing path, otherwise the first candidate."""
    for p in paths:
        if p.exists():
            return p
    return paths[0]


CNN_DIR = first_existing(
    ROOT / "data_out_fixed" / "wind_mrhr_cnn",
    ROOT / "data_out" / "wind_mrhr_cnn",
)
GAN_DIR = first_existing(
    ROOT / "data_out_fixed" / "wind_mrhr_gan",
    ROOT / "data_out" / "wind_mrhr_gan",
)
CANDIDATEB_DIR = ROOT / "data_out" / "wind_finetune_pilot_candidateB"
CANDIDATEB_EXPANDED672_DIR = ROOT / "data_out" / "wind_finetune_candidateB_expanded672"
CANDIDATEC_DIR = ROOT / "data_out" / "wind_finetune_pilot_candidateC"
CANDIDATEC_EXPANDED672_DIR = ROOT / "data_out" / "wind_finetune_candidateC_expanded672"
CANDIDATEC_EXPANDED1344_DIR = ROOT / "data_out" / "wind_finetune_candidateC_expanded1344"
CANDIDATEC_EXPANDED2688_DIR = ROOT / "data_out" / "wind_finetune_candidateC_expanded2688"
CANDIDATEUV_DIR = ROOT / "data_out" / "wind_finetune_pilot_candidateUV"
CANDIDATEUV_EXPANDED672_DIR = ROOT / "data_out" / "wind_finetune_candidateUV_expanded672"
CANDIDATEUV_EXPANDED1344_DIR = ROOT / "data_out" / "wind_finetune_candidateUV_expanded1344"
CANDIDATEUV_EXPANDED2688_DIR = ROOT / "data_out" / "wind_finetune_candidateUV_expanded2688"
CANDIDATED_DIR = ROOT / "data_out" / "wind_finetune_pilot_candidateD"
CANDIDATEDPD_EXPANDED672_DIR = ROOT / "data_out" / "wind_finetune_candidateDpd_expanded672"
CANDIDATEE2_DIR = ROOT / "data_out" / "wind_finetune_pilot_candidateE2"
CANDIDATEE2_EXPANDED672_DIR = ROOT / "data_out" / "wind_finetune_candidateE2_expanded672"

CANDIDATEB_TOPOLOGY_COMPARISON = (
    ROOT
    / "ttk_runs_fixed"
    / "topology_finetuning"
    / "candidateB_topology"
    / "candidateB_topology_comparison.csv"
)
CANDIDATEB_EXPANDED672_TOPOLOGY_COMPARISON = (
    ROOT
    / "ttk_runs_fixed"
    / "topology_finetuning"
    / "candidateB_expanded672_topology"
    / "candidateB_expanded672_topology_comparison.csv"
)
CANDIDATEC_TOPOLOGY_COMPARISON = (
    ROOT
    / "ttk_runs_fixed"
    / "topology_finetuning"
    / "candidateC_topology"
    / "candidateC_topology_comparison.csv"
)
CANDIDATEC_EXPANDED672_TOPOLOGY_COMPARISON = (
    ROOT
    / "ttk_runs_fixed"
    / "topology_finetuning"
    / "candidateC_expanded672_topology"
    / "candidateC_expanded672_topology_comparison.csv"
)
CANDIDATEC_EXPANDED1344_TOPOLOGY_COMPARISON = (
    ROOT
    / "ttk_runs_fixed"
    / "topology_finetuning"
    / "candidateC_expanded1344_topology"
    / "candidateC_expanded1344_topology_comparison.csv"
)
CANDIDATEC_EXPANDED2688_TOPOLOGY_COMPARISON = (
    ROOT
    / "ttk_runs_fixed"
    / "topology_finetuning"
    / "candidateC_expanded2688_topology"
    / "candidateC_expanded2688_topology_comparison.csv"
)
CANDIDATEUV_TOPOLOGY_COMPARISON = (
    ROOT
    / "ttk_runs_fixed"
    / "topology_finetuning"
    / "candidateUV_topology"
    / "candidateUV_topology_comparison.csv"
)
CANDIDATEUV_EXPANDED672_TOPOLOGY_COMPARISON = (
    ROOT
    / "ttk_runs_fixed"
    / "topology_finetuning"
    / "candidateUV_expanded672_topology"
    / "candidateUV_expanded672_topology_comparison.csv"
)
CANDIDATEUV_EXPANDED1344_TOPOLOGY_COMPARISON = (
    ROOT
    / "ttk_runs_fixed"
    / "topology_finetuning"
    / "candidateUV_expanded1344_topology"
    / "candidateUV_expanded1344_topology_comparison.csv"
)
CANDIDATEUV_EXPANDED2688_TOPOLOGY_COMPARISON = (
    ROOT
    / "ttk_runs_fixed"
    / "topology_finetuning"
    / "candidateUV_expanded2688_topology"
    / "candidateUV_expanded2688_topology_comparison.csv"
)
CANDIDATED_TOPOLOGY_COMPARISON = (
    ROOT
    / "ttk_runs_fixed"
    / "topology_finetuning"
    / "candidateD_topology"
    / "candidateD_topology_comparison.csv"
)
CANDIDATEDPD_EXPANDED672_TOPOLOGY_COMPARISON = (
    ROOT
    / "ttk_runs_fixed"
    / "topology_finetuning"
    / "candidateDpd_expanded672_topology"
    / "candidateDpd_expanded672_topology_comparison.csv"
)
CANDIDATEE2_TOPOLOGY_COMPARISON = (
    ROOT
    / "ttk_runs_fixed"
    / "topology_finetuning"
    / "candidateE2_topology"
    / "candidateE2_topology_comparison.csv"
)
CANDIDATEE2_EXPANDED672_TOPOLOGY_COMPARISON = (
    ROOT
    / "ttk_runs_fixed"
    / "topology_finetuning"
    / "candidateE2_expanded672_topology"
    / "candidateE2_expanded672_topology_comparison.csv"
)

CANDIDATEB_EVAL = (
    ROOT
    / "ttk_runs_fixed"
    / "topology_finetuning"
    / "candidateB_eval"
    / "all_sample_metrics_candidateB.csv"
)
CANDIDATEB_EXPANDED672_EVAL = (
    ROOT
    / "ttk_runs_fixed"
    / "topology_finetuning"
    / "candidateB_expanded672_eval"
    / "all_sample_metrics_candidateB_expanded672.csv"
)
CANDIDATEC_EVAL = (
    ROOT
    / "ttk_runs_fixed"
    / "topology_finetuning"
    / "candidateC_eval"
    / "all_sample_metrics_candidateC.csv"
)
CANDIDATEC_EXPANDED672_EVAL = (
    ROOT
    / "ttk_runs_fixed"
    / "topology_finetuning"
    / "candidateC_expanded672_eval"
    / "all_sample_metrics_candidateC_expanded672.csv"
)
CANDIDATEC_EXPANDED1344_EVAL = (
    ROOT
    / "ttk_runs_fixed"
    / "topology_finetuning"
    / "candidateC_expanded1344_eval"
    / "all_sample_metrics_candidateC_expanded1344.csv"
)
CANDIDATEC_EXPANDED2688_EVAL = (
    ROOT
    / "ttk_runs_fixed"
    / "topology_finetuning"
    / "candidateC_expanded2688_eval"
    / "all_sample_metrics_candidateC_expanded2688.csv"
)
CANDIDATEUV_EVAL = (
    ROOT
    / "ttk_runs_fixed"
    / "topology_finetuning"
    / "candidateUV_eval"
    / "all_sample_metrics_candidateUV.csv"
)
CANDIDATEUV_EXPANDED672_EVAL = (
    ROOT
    / "ttk_runs_fixed"
    / "topology_finetuning"
    / "candidateUV_expanded672_eval"
    / "all_sample_metrics_candidateUV_expanded672.csv"
)
CANDIDATEUV_EXPANDED1344_EVAL = (
    ROOT
    / "ttk_runs_fixed"
    / "topology_finetuning"
    / "candidateUV_expanded1344_eval"
    / "all_sample_metrics_candidateUV_expanded1344.csv"
)
CANDIDATEUV_EXPANDED2688_EVAL = (
    ROOT
    / "ttk_runs_fixed"
    / "topology_finetuning"
    / "candidateUV_expanded2688_eval"
    / "all_sample_metrics_candidateUV_expanded2688.csv"
)
CANDIDATED_EVAL = (
    ROOT
    / "ttk_runs_fixed"
    / "topology_finetuning"
    / "candidateD_eval"
    / "all_sample_metrics_candidateD.csv"
)
CANDIDATEDPD_EXPANDED672_EVAL = (
    ROOT
    / "ttk_runs_fixed"
    / "topology_finetuning"
    / "candidateDpd_expanded672_eval"
    / "all_sample_metrics_candidateDpd_expanded672.csv"
)
CANDIDATEE2_EVAL = (
    ROOT
    / "ttk_runs_fixed"
    / "topology_finetuning"
    / "candidateE2_eval"
    / "all_sample_metrics_candidateE2.csv"
)
CANDIDATEE2_EXPANDED672_EVAL = (
    ROOT
    / "ttk_runs_fixed"
    / "topology_finetuning"
    / "candidateE2_expanded672_eval"
    / "all_sample_metrics_candidateE2_expanded672.csv"
)

FULL_BREAKDOWN = ROOT / "ttk_runs_fixed" / "report_tables" / "full_physics_domain_breakdown" / "physics_domain_breakdown_all_samples.csv"
WIDE_TABLE = ROOT / "ttk_runs_fixed" / "report_tables" / "metric_sweep_all_samples_wide.csv"
OBS_PER_SAMPLE = ROOT / "ttk_runs_fixed" / "observation_groups" / "observation_groups_per_sample.csv"
RECOMMENDED_UNIQUE = ROOT / "ttk_runs_fixed" / "observation_groups" / "recommended_visual_inspection_unique_samples.csv"
OLD_MANIFEST = OUTDIR / "visual_inspection_manifest.csv"


# -----------------------------
# Forced sample set
# -----------------------------

FORCED = {
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
    18: "MT-GAN case recovered by Candidate B and Candidate C",
    19: "strong MT-GAN anchor",
    20: "MT-GAN case recovered by Candidate C",

    25: "MT-GAN case recovered by Candidate B, Candidate C, and Candidate D",
    63: "MT-GAN case recovered by Candidate B and Candidate C",
    80: "MT-GAN case recovered by Candidate B and Candidate C",
    6: "MT-GAN case recovered by Candidate C",
    62: "MT-GAN case recovered by Candidate C",
    65: "MT-GAN case recovered by Candidate C",
    68: "MT-GAN case recovered by Candidate C",
    79: "MT-GAN case recovered by Candidate C",
    154: "lower-confidence MT-GAN limitation case",
}

# From full topology comparisons:
# original MT-GAN cases recovered by each fine-tuned candidate.
CANDIDATEB_MT_GAN_RECOVERED = {18, 25, 63, 80}
CANDIDATEC_MT_GAN_RECOVERED = {6, 18, 20, 25, 62, 63, 65, 68, 79, 80, 92}
# Candidate C-expanded-672 recovered cases are inferred from its topology comparison CSV at runtime.
CANDIDATEUV_MT_GAN_RECOVERED = {25}
CANDIDATEUV_EXPANDED1344_MT_GAN_RECOVERED = {18, 25, 63, 68, 92}
CANDIDATEC_EXPANDED2688_MT_GAN_RECOVERED = {6, 18, 20, 25, 62, 63, 65, 68, 77, 79}
CANDIDATEUV_EXPANDED2688_MT_GAN_RECOVERED = {18, 25, 63, 68, 92}
CANDIDATED_MT_GAN_RECOVERED = {25}
CANDIDATEDPD_EXPANDED672_MT_GAN_RECOVERED = {25}
CANDIDATEE2_MT_GAN_RECOVERED = {25}
CANDIDATEE2_EXPANDED672_MT_GAN_RECOVERED = {25, 92}

# Map legacy physics/domain table keys to all_sample_metrics_candidate*.csv keys.
EVAL_METRIC_KEYS = {
    "wpd_bias": "wpd_bias_abs",
    "wpd_mae": "wpd_mae",
    "wpd_rmse": "wpd_rmse",
    "wpd_w1": "wpd_w1",
    "psd_log_l2": "psd_log_l2",
    "psd_slope_abs_delta": "psd_slope_abs_delta",
    "grad_mae": "grad_mae",
    "grad_w1": "grad_w1",
    "grad_kurtosis_abs_delta": "grad_kurtosis_abs_delta",
    "exceed_frac_abs_delta_t5": "exceed_abs_t5",
    "exceed_frac_abs_delta_t10": "exceed_abs_t10",
    "exceed_frac_abs_delta_t15": "exceed_abs_t15",
    "exceed_frac_abs_delta_p90": "exceed_abs_p90",
    "exceed_frac_abs_delta_p95": "exceed_abs_p95",
    "exceed_frac_abs_delta_p99": "exceed_abs_p99",
}

RANK_METHODS = [
    ("candidateC", "Candidate C"),
    ("candidateC_expanded672", "Candidate C-expanded-672"),
    ("candidateC_expanded1344", "Candidate C-expanded-1344"),
    ("candidateC_expanded2688", "Candidate C-expanded-2688"),
    ("candidateB_expanded672", "Candidate B-expanded-672"),
    ("candidateUV_expanded672", "Candidate UV-expanded-672"),
    ("candidateUV_expanded1344", "Candidate UV-expanded-1344"),
    ("candidateUV_expanded2688", "Candidate UV-expanded-2688"),
    ("candidateUV", "Candidate UV"),
    ("candidateD", "Candidate D"),
    ("candidateDpd_expanded672", "Candidate Dpd-expanded-672"),
    ("candidateE2", "Candidate E2"),
    ("candidateE2_expanded672", "Candidate E2-expanded-672"),
    ("candidateB", "Candidate B"),
    ("cnn", "CNN"),
    ("gan", "GAN"),
]


METRICS = [
    ("wpd_bias", "WPD bias |·|", "Physics / WPD"),
    ("wpd_mae", "WPD MAE", "Physics / WPD"),
    ("wpd_rmse", "WPD RMSE", "Physics / WPD"),
    ("wpd_w1", "WPD Wasserstein-1", "Distributional"),
    ("psd_log_l2", "PSD log-L2", "Distributional"),
    ("psd_slope_abs_delta", "PSD slope |Δ|", "Distributional"),
    ("grad_mae", "Gradient MAE", "Physics / Gradient"),
    ("grad_w1", "Gradient Wasserstein-1", "Distributional"),
    ("grad_kurtosis_abs_delta", "Gradient kurtosis |Δ|", "Distributional"),
    ("exceed_frac_abs_delta_t5", "Exceedance |Δ|, s > 5", "Tail / Exceedance"),
    ("exceed_frac_abs_delta_t10", "Exceedance |Δ|, s > 10", "Tail / Exceedance"),
    ("exceed_frac_abs_delta_t15", "Exceedance |Δ|, s > 15", "Tail / Exceedance"),
    ("exceed_frac_abs_delta_p90", "Exceedance |Δ|, p90", "Tail / Exceedance"),
    ("exceed_frac_abs_delta_p95", "Exceedance |Δ|, p95", "Tail / Exceedance"),
    ("exceed_frac_abs_delta_p99", "Exceedance |Δ|, p99", "Tail / Exceedance"),
]


# -----------------------------
# Helpers
# -----------------------------

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
    for k in ("sample_idx", "sample_id", "sample", "id"):
        if k in row and str(row[k]).strip():
            return int(float(str(row[k]).strip()))
    raise ValueError("row has no sample id")


def H(x) -> str:
    return html.escape(str(x))


def norm(x) -> str:
    s = str(x or "").strip().upper()
    if s in {"CNN", "GAN", "TIE"}:
        return s
    if s in {"TIED", "EQUAL"}:
        return "TIE"
    return s


def boolish(x) -> bool:
    return str(x or "").strip().lower() in {"true", "1", "yes", "y"}


def num(x) -> str:
    try:
        v = float(str(x).strip())
    except Exception:
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


def fnum(x, default=np.nan) -> float:
    try:
        return float(str(x).strip())
    except Exception:
        return default


def pick(row: dict, obs: dict, *keys: str) -> str:
    for k in keys:
        if k in row and str(row[k]).strip():
            return str(row[k]).strip()
        if k in obs and str(obs[k]).strip():
            return str(obs[k]).strip()
    return ""


# -----------------------------
# Load metadata
# -----------------------------

def load_metric_rows() -> dict[int, dict[str, str]]:
    for p in (FULL_BREAKDOWN, WIDE_TABLE):
        rows = read_csv(p)
        if rows:
            out = {}
            for r in rows:
                try:
                    out[sid_from(r)] = r
                except Exception:
                    pass
            print(f"Loaded {len(out)} metric rows from {p}")
            return out
    print("WARNING: no metric table found; index will have limited metric info.")
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
    return out


def _candidate_column(row: dict[str, str], candidate: str, stem: str) -> str:
    """Return a candidate-specific topology column robustly across script versions."""
    candidates = [
        f"{stem}_{candidate}",
        f"{stem}_{candidate.lower()}",
        f"{stem}_candidate",
    ]
    for k in candidates:
        if k in row and str(row[k]).strip():
            return str(row[k]).strip()
    return ""


def load_candidate_topology_rows(path: Path, candidate: str, label: str) -> dict[int, dict[str, str]]:
    """Load 3-way CNN/GAN/candidate topology comparison and normalize key names."""
    rows = read_csv(path)
    out: dict[int, dict[str, str]] = {}
    for r in rows:
        try:
            sid = sid_from(r)
        except Exception:
            continue

        normalized = {k: v for k, v in r.items() if v is not None}

        # Normalize common columns so downstream HTML can use stable names:
        # pd_distance_candidateB / mt_winner_after_candidateC / etc.
        for stem in ("pd_distance", "mt_distance", "pd_winner_after", "mt_winner_after"):
            val = _candidate_column(r, candidate, stem)
            if val:
                normalized[f"{stem}_{candidate}"] = val

        # Carry over CNN/GAN/baseline columns.
        for k in [
            "pd_distance_cnn", "pd_distance_gan", "mt_distance_cnn", "mt_distance_gan",
            "pd_winner_before", "mt_winner_before", "was_mt_gan_win_before",
        ]:
            if k in r and str(r[k]).strip():
                normalized[k] = r[k]

        out[sid] = normalized

    if out:
        print(f"Loaded {len(out)} {label} topology rows from {path}")
    else:
        print(f"WARNING: {label} topology comparison not found; {label} tags will be limited.")
    return out


def load_candidate_eval_rows() -> dict[int, dict[str, dict[str, str]]]:
    """Load per-sample rows from Candidate B/C evaluation CSVs for 4-way metric rankings."""
    out: dict[int, dict[str, dict[str, str]]] = {}

    for path in (CANDIDATEB_EVAL, CANDIDATEB_EXPANDED672_EVAL, CANDIDATEC_EVAL, CANDIDATEC_EXPANDED672_EVAL, CANDIDATEC_EXPANDED1344_EVAL, CANDIDATEC_EXPANDED2688_EVAL, CANDIDATEUV_EVAL, CANDIDATEUV_EXPANDED672_EVAL, CANDIDATEUV_EXPANDED1344_EVAL, CANDIDATEUV_EXPANDED2688_EVAL, CANDIDATED_EVAL, CANDIDATEDPD_EXPANDED672_EVAL, CANDIDATEE2_EVAL, CANDIDATEE2_EXPANDED672_EVAL):
        for r in read_csv(path):
            try:
                sid = sid_from(r)
            except Exception:
                continue
            method = str(r.get("method", "")).strip()
            if not method:
                continue
            out.setdefault(sid, {})[method] = r

    # Normalize candidate method names in case a report used different capitalization.
    for sid, methods in out.items():
        for k in list(methods):
            lk = k.lower()
            if lk == "candidateb" and "candidateB" not in methods:
                methods["candidateB"] = methods[k]
            elif lk == "candidatec" and "candidateC" not in methods:
                methods["candidateC"] = methods[k]
            elif lk in {"candidateb_expanded672", "candidatebexpanded672"} and "candidateB_expanded672" not in methods:
                methods["candidateB_expanded672"] = methods[k]
            elif lk in {"candidatec_expanded672", "candidatecexpanded672"} and "candidateC_expanded672" not in methods:
                methods["candidateC_expanded672"] = methods[k]
            elif lk in {"candidatec_expanded1344", "candidatecexpanded1344"} and "candidateC_expanded1344" not in methods:
                methods["candidateC_expanded1344"] = methods[k]
            elif lk in {"candidatec_expanded2688", "candidatecexpanded2688"} and "candidateC_expanded2688" not in methods:
                methods["candidateC_expanded2688"] = methods[k]
            elif lk in {"candidateuv_expanded672", "candidateuvexpanded672"} and "candidateUV_expanded672" not in methods:
                methods["candidateUV_expanded672"] = methods[k]
            elif lk in {"candidateuv_expanded1344", "candidateuvexpanded1344"} and "candidateUV_expanded1344" not in methods:
                methods["candidateUV_expanded1344"] = methods[k]
            elif lk in {"candidateuv_expanded2688", "candidateuvexpanded2688"} and "candidateUV_expanded2688" not in methods:
                methods["candidateUV_expanded2688"] = methods[k]
            elif lk == "candidateuv" and "candidateUV" not in methods:
                methods["candidateUV"] = methods[k]
            elif lk == "candidated" and "candidateD" not in methods:
                methods["candidateD"] = methods[k]
            elif lk in {"candidatedpd_expanded672", "candidatedpdexpanded672"} and "candidateDpd_expanded672" not in methods:
                methods["candidateDpd_expanded672"] = methods[k]
            elif lk == "candidatee2" and "candidateE2" not in methods:
                methods["candidateE2"] = methods[k]
            elif lk in {"candidatee2_expanded672", "candidatee2expanded672"} and "candidateE2_expanded672" not in methods:
                methods["candidateE2_expanded672"] = methods[k]

    if out:
        print(f"Loaded candidate evaluation metric rows for {len(out)} samples")
    else:
        print("WARNING: Candidate B/C/UV/D/E2 and expanded evaluation CSVs not found; metric ranking unavailable.")
    return out


def infer_groups(sid: int, row: dict, obs: dict) -> list[str]:
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
    gan_majority = boolish(row.get("gan_metric_majority", "")) or str(row.get("overall_metric_majority", "")).upper() == "GAN"

    if mt == "GAN":
        groups.add("mt_gan_diagnostic")
    if pd == "GAN" and mt == "GAN":
        groups.add("topology_consensus_gan")
    if pd == "GAN" and mt == "CNN":
        groups.add("pd_gan_mt_cnn_control")
        groups.add("candidate_structural_hallucination_signature")
    if pd == "CNN" and mt == "CNN":
        groups.add("topology_consensus_cnn")
    if gan_majority:
        groups.add("gan_metric_majority")
    if gan_majority and mt != "GAN":
        groups.add("gan_majority_mt_rejects_gan")
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

    mt_after = str(obs.get("mt_winner_after_candidateB", "")).strip().lower()
    pd_after = str(obs.get("pd_winner_after_candidateB", "")).strip().lower()
    was_mt_gan = boolish(obs.get("was_mt_gan_win_before", ""))

    if sid in CANDIDATEB_MT_GAN_RECOVERED:
        groups.add("candidateB_mt_gan_recovered")
    if mt_after == "candidateb":
        groups.add("candidateB_mt_winner")
    if was_mt_gan and mt_after == "candidateb":
        groups.add("mt_gan_flipped_to_candidateB")
    if pd_after == "candidateb":
        groups.add("candidateB_pd_winner")

    cb_pd = fnum(obs.get("pd_distance_candidateB", ""))
    cnn_pd = fnum(obs.get("pd_distance_cnn", ""))
    cb_mt = fnum(obs.get("mt_distance_candidateB", ""))
    cnn_mt = fnum(obs.get("mt_distance_cnn", ""))
    if np.isfinite(cb_pd) and np.isfinite(cnn_pd) and cb_pd < cnn_pd:
        groups.add("candidateB_pd_improves_vs_cnn")
    if np.isfinite(cb_mt) and np.isfinite(cnn_mt) and cb_mt < cnn_mt:
        groups.add("candidateB_mt_improves_vs_cnn")

    mt_after_c = str(obs.get("mt_winner_after_candidateC", "")).strip().lower()
    pd_after_c = str(obs.get("pd_winner_after_candidateC", "")).strip().lower()

    if sid in CANDIDATEC_MT_GAN_RECOVERED:
        groups.add("candidateC_mt_gan_recovered")
    if mt_after_c == "candidatec":
        groups.add("candidateC_mt_winner")
    if was_mt_gan and mt_after_c == "candidatec":
        groups.add("mt_gan_flipped_to_candidateC")
    if pd_after_c == "candidatec":
        groups.add("candidateC_pd_winner")

    cc_pd = fnum(obs.get("pd_distance_candidateC", ""))
    cc_mt = fnum(obs.get("mt_distance_candidateC", ""))
    if np.isfinite(cc_pd) and np.isfinite(cnn_pd) and cc_pd < cnn_pd:
        groups.add("candidateC_pd_improves_vs_cnn")
    if np.isfinite(cc_mt) and np.isfinite(cnn_mt) and cc_mt < cnn_mt:
        groups.add("candidateC_mt_improves_vs_cnn")
    if np.isfinite(cc_pd) and np.isfinite(cb_pd) and cc_pd < cb_pd:
        groups.add("candidateC_pd_improves_vs_candidateB")
    if np.isfinite(cc_mt) and np.isfinite(cb_mt) and cc_mt < cb_mt:
        groups.add("candidateC_mt_improves_vs_candidateB")

    mt_after_cexp = str(obs.get("mt_winner_after_candidateC_expanded672", "")).strip().lower()
    pd_after_cexp = str(obs.get("pd_winner_after_candidateC_expanded672", "")).strip().lower()

    if mt_after_cexp == "candidatec_expanded672":
        groups.add("candidateC_expanded672_mt_winner")
    if was_mt_gan and mt_after_cexp == "candidatec_expanded672":
        groups.add("mt_gan_flipped_to_candidateC_expanded672")
    if pd_after_cexp == "candidatec_expanded672":
        groups.add("candidateC_expanded672_pd_winner")

    cexp_pd = fnum(obs.get("pd_distance_candidateC_expanded672", ""))
    cexp_mt = fnum(obs.get("mt_distance_candidateC_expanded672", ""))
    if np.isfinite(cexp_pd) and np.isfinite(cnn_pd) and cexp_pd < cnn_pd:
        groups.add("candidateC_expanded672_pd_improves_vs_cnn")
    if np.isfinite(cexp_mt) and np.isfinite(cnn_mt) and cexp_mt < cnn_mt:
        groups.add("candidateC_expanded672_mt_improves_vs_cnn")
    if np.isfinite(cexp_pd) and np.isfinite(cc_pd) and cexp_pd < cc_pd:
        groups.add("candidateC_expanded672_pd_improves_vs_candidateC")
    if np.isfinite(cexp_mt) and np.isfinite(cc_mt) and cexp_mt < cc_mt:
        groups.add("candidateC_expanded672_mt_improves_vs_candidateC")

    mt_after_c1344 = str(obs.get("mt_winner_after_candidateC_expanded1344", "")).strip().lower()
    pd_after_c1344 = str(obs.get("pd_winner_after_candidateC_expanded1344", "")).strip().lower()
    c1344_pd = fnum(obs.get("pd_distance_candidateC_expanded1344", ""))
    c1344_mt = fnum(obs.get("mt_distance_candidateC_expanded1344", ""))
    if mt_after_c1344 == "candidatec_expanded1344":
        groups.add("candidateC_expanded1344_mt_winner")
    if was_mt_gan and mt_after_c1344 == "candidatec_expanded1344":
        groups.add("mt_gan_flipped_to_candidateC_expanded1344")
    if pd_after_c1344 == "candidatec_expanded1344":
        groups.add("candidateC_expanded1344_pd_winner")
    if np.isfinite(c1344_pd) and np.isfinite(cnn_pd) and c1344_pd < cnn_pd:
        groups.add("candidateC_expanded1344_pd_improves_vs_cnn")
    if np.isfinite(c1344_mt) and np.isfinite(cnn_mt) and c1344_mt < cnn_mt:
        groups.add("candidateC_expanded1344_mt_improves_vs_cnn")
    if np.isfinite(c1344_pd) and np.isfinite(cexp_pd) and c1344_pd < cexp_pd:
        groups.add("candidateC_expanded1344_pd_improves_vs_candidateC_expanded672")
    if np.isfinite(c1344_mt) and np.isfinite(cexp_mt) and c1344_mt < cexp_mt:
        groups.add("candidateC_expanded1344_mt_improves_vs_candidateC_expanded672")

    uv1344_pd = fnum(obs.get("pd_distance_candidateUV_expanded1344", ""))
    uv1344_mt = fnum(obs.get("mt_distance_candidateUV_expanded1344", ""))
    if np.isfinite(c1344_pd) and np.isfinite(uv1344_pd) and c1344_pd < uv1344_pd:
        groups.add("candidateC_expanded1344_pd_improves_vs_candidateUV_expanded1344")
    if np.isfinite(c1344_mt) and np.isfinite(uv1344_mt) and c1344_mt < uv1344_mt:
        groups.add("candidateC_expanded1344_mt_improves_vs_candidateUV_expanded1344")

    mt_after_c2688 = str(obs.get("mt_winner_after_candidateC_expanded2688", "")).strip().lower()
    pd_after_c2688 = str(obs.get("pd_winner_after_candidateC_expanded2688", "")).strip().lower()
    c2688_pd = fnum(obs.get("pd_distance_candidateC_expanded2688", ""))
    c2688_mt = fnum(obs.get("mt_distance_candidateC_expanded2688", ""))
    if mt_after_c2688 == "candidatec_expanded2688":
        groups.add("candidateC_expanded2688_mt_winner")
    if was_mt_gan and mt_after_c2688 == "candidatec_expanded2688":
        groups.add("mt_gan_flipped_to_candidateC_expanded2688")
    if pd_after_c2688 == "candidatec_expanded2688":
        groups.add("candidateC_expanded2688_pd_winner")
    if np.isfinite(c2688_pd) and np.isfinite(cnn_pd) and c2688_pd < cnn_pd:
        groups.add("candidateC_expanded2688_pd_improves_vs_cnn")
    if np.isfinite(c2688_mt) and np.isfinite(cnn_mt) and c2688_mt < cnn_mt:
        groups.add("candidateC_expanded2688_mt_improves_vs_cnn")
    if np.isfinite(c2688_pd) and np.isfinite(c1344_pd) and c2688_pd < c1344_pd:
        groups.add("candidateC_expanded2688_pd_improves_vs_candidateC_expanded1344")
    if np.isfinite(c2688_mt) and np.isfinite(c1344_mt) and c2688_mt < c1344_mt:
        groups.add("candidateC_expanded2688_mt_improves_vs_candidateC_expanded1344")

    uv2688_pd = fnum(obs.get("pd_distance_candidateUV_expanded2688", ""))
    uv2688_mt = fnum(obs.get("mt_distance_candidateUV_expanded2688", ""))
    if np.isfinite(c2688_pd) and np.isfinite(uv2688_pd) and c2688_pd < uv2688_pd:
        groups.add("candidateC_expanded2688_pd_improves_vs_candidateUV_expanded2688")
    if np.isfinite(c2688_mt) and np.isfinite(uv2688_mt) and c2688_mt < uv2688_mt:
        groups.add("candidateC_expanded2688_mt_improves_vs_candidateUV_expanded2688")

    # Expanded-data ablation controls: Candidate B-expanded-672 and Candidate UV-expanded-672.
    for exp_key, exp_group in [
        ("candidateB_expanded672", "candidateB_expanded672"),
        ("candidateUV_expanded672", "candidateUV_expanded672"),
        ("candidateUV_expanded1344", "candidateUV_expanded1344"),
        ("candidateUV_expanded2688", "candidateUV_expanded2688"),
    ]:
        mt_after_exp = str(obs.get(f"mt_winner_after_{exp_key}", "")).strip().lower()
        pd_after_exp = str(obs.get(f"pd_winner_after_{exp_key}", "")).strip().lower()
        exp_pd = fnum(obs.get(f"pd_distance_{exp_key}", ""))
        exp_mt = fnum(obs.get(f"mt_distance_{exp_key}", ""))
        if mt_after_exp == exp_key.lower():
            groups.add(f"{exp_group}_mt_winner")
        if was_mt_gan and mt_after_exp == exp_key.lower():
            groups.add(f"mt_gan_flipped_to_{exp_group}")
        if pd_after_exp == exp_key.lower():
            groups.add(f"{exp_group}_pd_winner")
        if np.isfinite(exp_pd) and np.isfinite(cnn_pd) and exp_pd < cnn_pd:
            groups.add(f"{exp_group}_pd_improves_vs_cnn")
        if np.isfinite(exp_mt) and np.isfinite(cnn_mt) and exp_mt < cnn_mt:
            groups.add(f"{exp_group}_mt_improves_vs_cnn")
        if np.isfinite(exp_pd) and np.isfinite(cexp_pd) and exp_pd < cexp_pd:
            groups.add(f"{exp_group}_pd_improves_vs_candidateC_expanded672")
        if np.isfinite(exp_mt) and np.isfinite(cexp_mt) and exp_mt < cexp_mt:
            groups.add(f"{exp_group}_mt_improves_vs_candidateC_expanded672")

    mt_after_uv = str(obs.get("mt_winner_after_candidateUV", "")).strip().lower()
    pd_after_uv = str(obs.get("pd_winner_after_candidateUV", "")).strip().lower()

    if sid in CANDIDATEUV_MT_GAN_RECOVERED:
        groups.add("candidateUV_mt_gan_recovered")
    if mt_after_uv == "candidateuv":
        groups.add("candidateUV_mt_winner")
    if was_mt_gan and mt_after_uv == "candidateuv":
        groups.add("mt_gan_flipped_to_candidateUV")
    if pd_after_uv == "candidateuv":
        groups.add("candidateUV_pd_winner")

    cuv_pd = fnum(obs.get("pd_distance_candidateUV", ""))
    cuv_mt = fnum(obs.get("mt_distance_candidateUV", ""))
    if np.isfinite(cuv_pd) and np.isfinite(cnn_pd) and cuv_pd < cnn_pd:
        groups.add("candidateUV_pd_improves_vs_cnn")
    if np.isfinite(cuv_mt) and np.isfinite(cnn_mt) and cuv_mt < cnn_mt:
        groups.add("candidateUV_mt_improves_vs_cnn")
    if np.isfinite(cuv_pd) and np.isfinite(cc_pd) and cuv_pd < cc_pd:
        groups.add("candidateUV_pd_improves_vs_candidateC")
    if np.isfinite(cuv_mt) and np.isfinite(cc_mt) and cuv_mt < cc_mt:
        groups.add("candidateUV_mt_improves_vs_candidateC")

    mt_after_d = str(obs.get("mt_winner_after_candidateD", "")).strip().lower()
    pd_after_d = str(obs.get("pd_winner_after_candidateD", "")).strip().lower()

    if sid in CANDIDATED_MT_GAN_RECOVERED:
        groups.add("candidateD_mt_gan_recovered")
    if mt_after_d == "candidated":
        groups.add("candidateD_mt_winner")
    if was_mt_gan and mt_after_d == "candidated":
        groups.add("mt_gan_flipped_to_candidateD")
    if pd_after_d == "candidated":
        groups.add("candidateD_pd_winner")

    cd_pd = fnum(obs.get("pd_distance_candidateD", ""))
    cd_mt = fnum(obs.get("mt_distance_candidateD", ""))
    if np.isfinite(cd_pd) and np.isfinite(cnn_pd) and cd_pd < cnn_pd:
        groups.add("candidateD_pd_improves_vs_cnn")
    if np.isfinite(cd_mt) and np.isfinite(cnn_mt) and cd_mt < cnn_mt:
        groups.add("candidateD_mt_improves_vs_cnn")
    if np.isfinite(cd_pd) and np.isfinite(cc_pd) and cd_pd < cc_pd:
        groups.add("candidateD_pd_improves_vs_candidateC")
    if np.isfinite(cd_mt) and np.isfinite(cc_mt) and cd_mt < cc_mt:
        groups.add("candidateD_mt_improves_vs_candidateC")

    mt_after_dpd = str(obs.get("mt_winner_after_candidateDpd_expanded672", "")).strip().lower()
    pd_after_dpd = str(obs.get("pd_winner_after_candidateDpd_expanded672", "")).strip().lower()

    if sid in CANDIDATEDPD_EXPANDED672_MT_GAN_RECOVERED:
        groups.add("candidateDpd_expanded672_mt_gan_recovered")
    if mt_after_dpd == "candidatedpd_expanded672":
        groups.add("candidateDpd_expanded672_mt_winner")
    if was_mt_gan and mt_after_dpd == "candidatedpd_expanded672":
        groups.add("mt_gan_flipped_to_candidateDpd_expanded672")
    if pd_after_dpd == "candidatedpd_expanded672":
        groups.add("candidateDpd_expanded672_pd_winner")

    cdpd_pd = fnum(obs.get("pd_distance_candidateDpd_expanded672", ""))
    cdpd_mt = fnum(obs.get("mt_distance_candidateDpd_expanded672", ""))
    if np.isfinite(cdpd_pd) and np.isfinite(cnn_pd) and cdpd_pd < cnn_pd:
        groups.add("candidateDpd_expanded672_pd_improves_vs_cnn")
    if np.isfinite(cdpd_mt) and np.isfinite(cnn_mt) and cdpd_mt < cnn_mt:
        groups.add("candidateDpd_expanded672_mt_improves_vs_cnn")
    if np.isfinite(cdpd_pd) and np.isfinite(cexp_pd) and cdpd_pd < cexp_pd:
        groups.add("candidateDpd_expanded672_pd_improves_vs_candidateC_expanded672")
    if np.isfinite(cdpd_mt) and np.isfinite(cexp_mt) and cdpd_mt < cexp_mt:
        groups.add("candidateDpd_expanded672_mt_improves_vs_candidateC_expanded672")

    mt_after_e2 = str(obs.get("mt_winner_after_candidateE2", "")).strip().lower()
    pd_after_e2 = str(obs.get("pd_winner_after_candidateE2", "")).strip().lower()

    if sid in CANDIDATEE2_MT_GAN_RECOVERED:
        groups.add("candidateE2_mt_gan_recovered")
    if mt_after_e2 == "candidatee2":
        groups.add("candidateE2_mt_winner")
    if was_mt_gan and mt_after_e2 == "candidatee2":
        groups.add("mt_gan_flipped_to_candidateE2")
    if pd_after_e2 == "candidatee2":
        groups.add("candidateE2_pd_winner")

    ce2_pd = fnum(obs.get("pd_distance_candidateE2", ""))
    ce2_mt = fnum(obs.get("mt_distance_candidateE2", ""))
    if np.isfinite(ce2_pd) and np.isfinite(cnn_pd) and ce2_pd < cnn_pd:
        groups.add("candidateE2_pd_improves_vs_cnn")
    if np.isfinite(ce2_mt) and np.isfinite(cnn_mt) and ce2_mt < cnn_mt:
        groups.add("candidateE2_mt_improves_vs_cnn")
    if np.isfinite(ce2_pd) and np.isfinite(cc_pd) and ce2_pd < cc_pd:
        groups.add("candidateE2_pd_improves_vs_candidateC")
    if np.isfinite(ce2_mt) and np.isfinite(cc_mt) and ce2_mt < cc_mt:
        groups.add("candidateE2_mt_improves_vs_candidateC")

    return sorted(groups)


def question(sid: int, row: dict, obs: dict) -> str:
    if sid in CANDIDATEDPD_EXPANDED672_MT_GAN_RECOVERED:
        return "Candidate Dpd-expanded-672 is the active PD-loss run's recovered MT-GAN case: does direct PD supervision visibly improve structure, or is the gain localized?"
    if sid in CANDIDATEE2_MT_GAN_RECOVERED:
        return "Candidate E2 is the corrected TTK critical-pair loss recovered MT-GAN case: does fixed critical-pair supervision change visible structure, or is the improvement isolated?"
    if sid in CANDIDATED_MT_GAN_RECOVERED:
        return "Candidate D is the true PD-loss pilot's only recovered MT-GAN case: does the PD-refiner change visible structure, or is the improvement localized and metric-specific?"
    if str(obs.get("mt_winner_after_candidateC_expanded1344", "")).strip().lower() == "candidatec_expanded1344":
        return "Candidate C-expanded-1344 recovers this case after larger seasonal training: does the extra training data strengthen topology-relevant peaks or change merge-tree hierarchy relative to C-expanded-672?"
    if str(obs.get("mt_winner_after_candidateUV_expanded1344", "")).strip().lower() == "candidateuv_expanded1344":
        return "Candidate UV-expanded-1344 recovers this case under UV-only training: compare against Candidate C-expanded-1344 to isolate whether auxiliary topology-aware losses are helpful."
    if str(obs.get("mt_winner_after_candidateC_expanded672", "")).strip().lower() == "candidatec_expanded672":
        return "Candidate C-expanded-672 recovers this case after training on non-overlapping seasonal data: does the expanded model improve visible PD-style structure or merge-tree hierarchy relative to Candidate C?"
    if str(obs.get("mt_winner_after_candidateB_expanded672", "")).strip().lower() == "candidateb_expanded672":
        return "Candidate B-expanded-672 recovers this case with physics/level-set losses: does expanded physics supervision improve visible threshold structure?"
    if str(obs.get("mt_winner_after_candidateUV_expanded672", "")).strip().lower() == "candidateuv_expanded672":
        return "Candidate UV-expanded-672 recovers this case with UV-only expanded fine-tuning: is this a generic fine-tuning effect or an isolated MT fluctuation?"
    if sid in CANDIDATEC_MT_GAN_RECOVERED:
        return "Candidate C recovered this original MT-GAN case: does the critical-value/topological-extrema proxy improve merge-tree hierarchy while retaining CNN-like fidelity?"
    if sid in CANDIDATEUV_MT_GAN_RECOVERED:
        return "Candidate UV is the UV-only ablation's only recovered MT-GAN case: is this an isolated fine-tuning effect compared with Candidate C's broader MT improvement?"
    if sid in CANDIDATEB_MT_GAN_RECOVERED:
        return "Candidate B recovered this original MT-GAN case: does the fine-tuned CNN restore broad level-set/hierarchical structure while retaining CNN fidelity?"

    pd = norm(pick(row, obs, "pd_winner"))
    mt = norm(pick(row, obs, "mt_winner"))
    gan_majority = boolish(row.get("gan_metric_majority", "")) or str(row.get("overall_metric_majority", "")).upper() == "GAN"

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
    return "Control sample: compare conservative fidelity, GAN texture, and topology choices."


def select_samples(args, metrics: dict[int, dict], obs: dict[int, dict]) -> list[int]:
    if args.all:
        return sorted(metrics.keys())
    if args.samples:
        return sorted({int(x.strip()) for x in args.samples.split(",") if x.strip()})

    selected = set(FORCED.keys())

    # Preserve any samples already in the existing manifest/index workflow.
    for sid in obs:
        if OLD_MANIFEST.exists():
            selected.add(sid)
        elif boolish(obs[sid].get("group_recommended_visual_inspection_unique", "")):
            selected.add(sid)
        elif obs[sid].get("recommendation_group", ""):
            selected.add(sid)

    return sorted(selected)


# -----------------------------
# Array panels
# -----------------------------

def load_arrays():
    gt_p = CNN_DIR / "dataGT.npy"
    cnn_p = CNN_DIR / "dataSR.npy"
    gan_p = GAN_DIR / "dataSR.npy"
    candb_p = CANDIDATEB_DIR / "dataSR.npy"
    candbexp_p = CANDIDATEB_EXPANDED672_DIR / "dataSR.npy"
    candc_p = CANDIDATEC_DIR / "dataSR.npy"
    candcexp_p = CANDIDATEC_EXPANDED672_DIR / "dataSR.npy"
    candc1344_p = CANDIDATEC_EXPANDED1344_DIR / "dataSR.npy"
    candc2688_p = CANDIDATEC_EXPANDED2688_DIR / "dataSR.npy"
    canduv_p = CANDIDATEUV_DIR / "dataSR.npy"
    canduvexp_p = CANDIDATEUV_EXPANDED672_DIR / "dataSR.npy"
    canduv1344_p = CANDIDATEUV_EXPANDED1344_DIR / "dataSR.npy"
    canduv2688_p = CANDIDATEUV_EXPANDED2688_DIR / "dataSR.npy"
    candd_p = CANDIDATED_DIR / "dataSR.npy"
    canddpd_p = CANDIDATEDPD_EXPANDED672_DIR / "dataSR.npy"
    cande2_p = CANDIDATEE2_DIR / "dataSR.npy"
    cande2exp_p = CANDIDATEE2_EXPANDED672_DIR / "dataSR.npy"
    missing = [str(p) for p in (gt_p, cnn_p, gan_p, candb_p, candbexp_p, candc_p, candcexp_p, candc1344_p, candc2688_p, canduv_p, canduvexp_p, canduv1344_p, canduv2688_p, candd_p, canddpd_p, cande2_p, cande2exp_p) if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing NPY arrays:\n" + "\n".join(missing))

    gt = np.load(gt_p, mmap_mode="r")
    cnn = np.load(cnn_p, mmap_mode="r")
    gan = np.load(gan_p, mmap_mode="r")
    candb = np.load(candb_p, mmap_mode="r")
    candbexp = np.load(candbexp_p, mmap_mode="r")
    candc = np.load(candc_p, mmap_mode="r")
    candcexp = np.load(candcexp_p, mmap_mode="r")
    candc1344 = np.load(candc1344_p, mmap_mode="r")
    candc2688 = np.load(candc2688_p, mmap_mode="r")
    canduv = np.load(canduv_p, mmap_mode="r")
    canduvexp = np.load(canduvexp_p, mmap_mode="r")
    canduv1344 = np.load(canduv1344_p, mmap_mode="r")
    canduv2688 = np.load(canduv2688_p, mmap_mode="r")
    candd = np.load(candd_p, mmap_mode="r")
    canddpd = np.load(canddpd_p, mmap_mode="r")
    cande2 = np.load(cande2_p, mmap_mode="r")
    cande2exp = np.load(cande2exp_p, mmap_mode="r")

    idx_p = CNN_DIR / "idx.npy"
    if idx_p.exists():
        idx = np.load(idx_p)
    else:
        idx = np.arange(gt.shape[0])
    pos = {int(v): i for i, v in enumerate(idx.tolist())}

    candb_idx_p = CANDIDATEB_DIR / "idx.npy"
    if candb_idx_p.exists():
        candb_idx = np.load(candb_idx_p)
    else:
        candb_idx = np.arange(candb.shape[0])
    candb_pos = {int(v): i for i, v in enumerate(candb_idx.tolist())}

    candbexp_idx_p = CANDIDATEB_EXPANDED672_DIR / "idx.npy"
    if candbexp_idx_p.exists():
        candbexp_idx = np.load(candbexp_idx_p)
    else:
        candbexp_idx = np.arange(candbexp.shape[0])
    candbexp_pos = {int(v): i for i, v in enumerate(candbexp_idx.tolist())}

    candc_idx_p = CANDIDATEC_DIR / "idx.npy"
    if candc_idx_p.exists():
        candc_idx = np.load(candc_idx_p)
    else:
        candc_idx = np.arange(candc.shape[0])
    candc_pos = {int(v): i for i, v in enumerate(candc_idx.tolist())}

    candcexp_idx_p = CANDIDATEC_EXPANDED672_DIR / "idx.npy"
    if candcexp_idx_p.exists():
        candcexp_idx = np.load(candcexp_idx_p)
    else:
        candcexp_idx = np.arange(candcexp.shape[0])
    candcexp_pos = {int(v): i for i, v in enumerate(candcexp_idx.tolist())}

    candc1344_idx_p = CANDIDATEC_EXPANDED1344_DIR / "idx.npy"
    if candc1344_idx_p.exists():
        candc1344_idx = np.load(candc1344_idx_p)
    else:
        candc1344_idx = np.arange(candc1344.shape[0])
    candc1344_pos = {int(v): i for i, v in enumerate(candc1344_idx.tolist())}

    candc2688_idx_p = CANDIDATEC_EXPANDED2688_DIR / "idx.npy"
    if candc2688_idx_p.exists():
        candc2688_idx = np.load(candc2688_idx_p)
    else:
        candc2688_idx = np.arange(candc2688.shape[0])
    candc2688_pos = {int(v): i for i, v in enumerate(candc2688_idx.tolist())}

    canduv_idx_p = CANDIDATEUV_DIR / "idx.npy"
    if canduv_idx_p.exists():
        canduv_idx = np.load(canduv_idx_p)
    else:
        canduv_idx = np.arange(canduv.shape[0])
    canduv_pos = {int(v): i for i, v in enumerate(canduv_idx.tolist())}

    canduvexp_idx_p = CANDIDATEUV_EXPANDED672_DIR / "idx.npy"
    if canduvexp_idx_p.exists():
        canduvexp_idx = np.load(canduvexp_idx_p)
    else:
        canduvexp_idx = np.arange(canduvexp.shape[0])
    canduvexp_pos = {int(v): i for i, v in enumerate(canduvexp_idx.tolist())}

    canduv1344_idx_p = CANDIDATEUV_EXPANDED1344_DIR / "idx.npy"
    if canduv1344_idx_p.exists():
        canduv1344_idx = np.load(canduv1344_idx_p)
    else:
        canduv1344_idx = np.arange(canduv1344.shape[0])
    canduv1344_pos = {int(v): i for i, v in enumerate(canduv1344_idx.tolist())}

    canduv2688_idx_p = CANDIDATEUV_EXPANDED2688_DIR / "idx.npy"
    if canduv2688_idx_p.exists():
        canduv2688_idx = np.load(canduv2688_idx_p)
    else:
        canduv2688_idx = np.arange(canduv2688.shape[0])
    canduv2688_pos = {int(v): i for i, v in enumerate(canduv2688_idx.tolist())}

    candd_idx_p = CANDIDATED_DIR / "idx.npy"
    if candd_idx_p.exists():
        candd_idx = np.load(candd_idx_p)
    else:
        candd_idx = np.arange(candd.shape[0])
    candd_pos = {int(v): i for i, v in enumerate(candd_idx.tolist())}

    canddpd_idx_p = CANDIDATEDPD_EXPANDED672_DIR / "idx.npy"
    if canddpd_idx_p.exists():
        canddpd_idx = np.load(canddpd_idx_p)
    else:
        canddpd_idx = np.arange(canddpd.shape[0])
    canddpd_pos = {int(v): i for i, v in enumerate(canddpd_idx.tolist())}

    cande2_idx_p = CANDIDATEE2_DIR / "idx.npy"
    if cande2_idx_p.exists():
        cande2_idx = np.load(cande2_idx_p)
    else:
        cande2_idx = np.arange(cande2.shape[0])
    cande2_pos = {int(v): i for i, v in enumerate(cande2_idx.tolist())}

    cande2exp_idx_p = CANDIDATEE2_EXPANDED672_DIR / "idx.npy"
    if cande2exp_idx_p.exists():
        cande2exp_idx = np.load(cande2exp_idx_p)
    else:
        cande2exp_idx = np.arange(cande2exp.shape[0])
    cande2exp_pos = {int(v): i for i, v in enumerate(cande2exp_idx.tolist())}

    return gt, cnn, gan, candb, candbexp, candc, candcexp, candc1344, candc2688, canduv, canduvexp, canduv1344, canduv2688, candd, canddpd, cande2, cande2exp, pos, candb_pos, candbexp_pos, candc_pos, candcexp_pos, candc1344_pos, candc2688_pos, canduv_pos, canduvexp_pos, canduv1344_pos, canduv2688_pos, candd_pos, canddpd_pos, cande2_pos, cande2exp_pos

def speed(a: np.ndarray) -> np.ndarray:
    if a.ndim == 3 and a.shape[-1] == 2:
        return np.sqrt(a[..., 0] ** 2 + a[..., 1] ** 2)
    if a.ndim == 3 and a.shape[-1] == 1:
        return a[..., 0]
    if a.ndim == 2:
        return a
    raise ValueError(f"Unexpected sample shape: {a.shape}")


def panel_title(sid: int, row: dict, obs: dict) -> str:
    return (
        f"sample {sid} | "
        f"SSIM={norm(pick(row, obs, 'ssim_winner')) or '?'} | "
        f"PD={norm(pick(row, obs, 'pd_winner')) or '?'} | "
        f"MT={norm(pick(row, obs, 'mt_winner')) or '?'} | "
        f"direct={norm(pick(row, obs, 'direct_error_group_winner')) or '?'} | "
        f"dist={norm(pick(row, obs, 'distributional_group_winner')) or row.get('overall_metric_majority', '?')}"
    )


def make_panel(sid: int, gt, cnn, gan, candb, candbexp, candc, candcexp, candc1344, candc2688, canduv, canduvexp, canduv1344, canduv2688, candd, canddpd, cande2, cande2exp,
               pos: dict[int, int], candb_pos: dict[int, int], candbexp_pos: dict[int, int], candc_pos: dict[int, int], candcexp_pos: dict[int, int], candc1344_pos: dict[int, int], candc2688_pos: dict[int, int], canduv_pos: dict[int, int], canduvexp_pos: dict[int, int], canduv1344_pos: dict[int, int], canduv2688_pos: dict[int, int], candd_pos: dict[int, int], canddpd_pos: dict[int, int], cande2_pos: dict[int, int], cande2exp_pos: dict[int, int],
               row: dict, obs: dict, crop, out: Path) -> bool:
    if sid not in pos:
        print(f"WARNING: sample {sid} not found in baseline idx.npy; skipping panel.")
        return False
    if sid not in candb_pos:
        print(f"WARNING: sample {sid} not found in Candidate B idx.npy; skipping panel.")
        return False
    if sid not in candbexp_pos:
        print(f"WARNING: sample {sid} not found in Candidate B-expanded-672 idx.npy; skipping panel.")
        return False
    if sid not in candc_pos:
        print(f"WARNING: sample {sid} not found in Candidate C idx.npy; skipping panel.")
        return False
    if sid not in candcexp_pos:
        print(f"WARNING: sample {sid} not found in Candidate C-expanded-672 idx.npy; skipping panel.")
        return False
    if sid not in candc1344_pos:
        print(f"WARNING: sample {sid} not found in Candidate C-expanded-1344 idx.npy; skipping panel.")
        return False
    if sid not in candc2688_pos:
        print(f"WARNING: sample {sid} not found in Candidate C-expanded-2688 idx.npy; skipping panel.")
        return False
    if sid not in canduv_pos:
        print(f"WARNING: sample {sid} not found in Candidate UV idx.npy; skipping panel.")
        return False
    if sid not in canduvexp_pos:
        print(f"WARNING: sample {sid} not found in Candidate UV-expanded-672 idx.npy; skipping panel.")
        return False
    if sid not in canduv1344_pos:
        print(f"WARNING: sample {sid} not found in Candidate UV-expanded-1344 idx.npy; skipping panel.")
        return False
    if sid not in canduv2688_pos:
        print(f"WARNING: sample {sid} not found in Candidate UV-expanded-2688 idx.npy; skipping panel.")
        return False
    if sid not in candd_pos:
        print(f"WARNING: sample {sid} not found in Candidate D idx.npy; skipping panel.")
        return False
    if sid not in canddpd_pos:
        print(f"WARNING: sample {sid} not found in Candidate Dpd-expanded-672 idx.npy; skipping panel.")
        return False
    if sid not in cande2_pos:
        print(f"WARNING: sample {sid} not found in Candidate E2 idx.npy; skipping panel.")
        return False
    if sid not in cande2exp_pos:
        print(f"WARNING: sample {sid} not found in Candidate E2-expanded-672 idx.npy; skipping panel.")
        return False

    i = pos[sid]
    j = candb_pos[sid]
    jb = candbexp_pos[sid]
    k = candc_pos[sid]
    x = candcexp_pos[sid]
    x1344 = candc1344_pos[sid]
    x2688 = candc2688_pos[sid]
    u = canduv_pos[sid]
    ux = canduvexp_pos[sid]
    ux1344 = canduv1344_pos[sid]
    ux2688 = canduv2688_pos[sid]
    m = candd_pos[sid]
    mpd = canddpd_pos[sid]
    n = cande2_pos[sid]
    ne2x = cande2exp_pos[sid]
    gt_s = speed(np.asarray(gt[i]))
    cnn_s = speed(np.asarray(cnn[i]))
    gan_s = speed(np.asarray(gan[i]))
    candb_s = speed(np.asarray(candb[j]))
    candbexp_s = speed(np.asarray(candbexp[jb]))
    candc_s = speed(np.asarray(candc[k]))
    candcexp_s = speed(np.asarray(candcexp[x]))
    candc1344_s = speed(np.asarray(candc1344[x1344]))
    candc2688_s = speed(np.asarray(candc2688[x2688]))
    canduv_s = speed(np.asarray(canduv[u]))
    canduvexp_s = speed(np.asarray(canduvexp[ux]))
    canduv1344_s = speed(np.asarray(canduv1344[ux1344]))
    canduv2688_s = speed(np.asarray(canduv2688[ux2688]))
    candd_s = speed(np.asarray(candd[m]))
    canddpd_s = speed(np.asarray(canddpd[mpd]))
    cande2_s = speed(np.asarray(cande2[n]))
    cande2exp_s = speed(np.asarray(cande2exp[ne2x]))

    desc = "full field"
    if crop is not None:
        y0, y1, x0, x1 = crop
        gt_s = gt_s[y0:y1, x0:x1]
        cnn_s = cnn_s[y0:y1, x0:x1]
        candb_s = candb_s[y0:y1, x0:x1]
        candbexp_s = candbexp_s[y0:y1, x0:x1]
        candc_s = candc_s[y0:y1, x0:x1]
        candcexp_s = candcexp_s[y0:y1, x0:x1]
        candc1344_s = candc1344_s[y0:y1, x0:x1]
        candc2688_s = candc2688_s[y0:y1, x0:x1]
        canduv_s = canduv_s[y0:y1, x0:x1]
        canduvexp_s = canduvexp_s[y0:y1, x0:x1]
        canduv1344_s = canduv1344_s[y0:y1, x0:x1]
        canduv2688_s = canduv2688_s[y0:y1, x0:x1]
        candd_s = candd_s[y0:y1, x0:x1]
        canddpd_s = canddpd_s[y0:y1, x0:x1]
        cande2_s = cande2_s[y0:y1, x0:x1]
        cande2exp_s = cande2exp_s[y0:y1, x0:x1]
        gan_s = gan_s[y0:y1, x0:x1]
        desc = f"crop y={y0}:{y1}, x={x0}:{x1}"

    err_cnn = np.abs(cnn_s - gt_s)
    err_candb = np.abs(candb_s - gt_s)
    err_candbexp = np.abs(candbexp_s - gt_s)
    err_candc = np.abs(candc_s - gt_s)
    err_canduv = np.abs(canduv_s - gt_s)
    err_canduvexp = np.abs(canduvexp_s - gt_s)
    err_canduv1344 = np.abs(canduv1344_s - gt_s)
    err_canduv2688 = np.abs(canduv2688_s - gt_s)
    err_candcexp = np.abs(candcexp_s - gt_s)
    err_candc1344 = np.abs(candc1344_s - gt_s)
    err_candc2688 = np.abs(candc2688_s - gt_s)
    err_candd = np.abs(candd_s - gt_s)
    err_canddpd = np.abs(canddpd_s - gt_s)
    err_cande2 = np.abs(cande2_s - gt_s)
    err_cande2exp = np.abs(cande2exp_s - gt_s)
    err_gan = np.abs(gan_s - gt_s)
    diff_c_uv = np.abs(candc_s - canduv_s)
    diff_cexp_c = np.abs(candcexp_s - candc_s)
    diff_c1344_cexp = np.abs(candc1344_s - candcexp_s)
    diff_c1344_c = np.abs(candc1344_s - candc_s)
    diff_cexp_bexp = np.abs(candcexp_s - candbexp_s)
    diff_cexp_uvexp = np.abs(candcexp_s - canduvexp_s)
    diff_c1344_uv1344 = np.abs(candc1344_s - canduv1344_s)
    diff_c2688_uv2688 = np.abs(candc2688_s - canduv2688_s)
    diff_c2688_c1344 = np.abs(candc2688_s - candc1344_s)
    diff_uv2688_uv1344 = np.abs(canduv2688_s - canduv1344_s)
    diff_uv1344_uvexp = np.abs(canduv1344_s - canduvexp_s)
    diff_e2_cc = np.abs(cande2_s - candc_s)
    diff_dpd_cexp = np.abs(canddpd_s - candcexp_s)
    diff_e2exp_e2 = np.abs(cande2exp_s - cande2_s)
    diff_e2exp_cexp = np.abs(cande2exp_s - candcexp_s)

    vmin = float(min(np.nanmin(gt_s), np.nanmin(cnn_s), np.nanmin(candb_s), np.nanmin(candbexp_s), np.nanmin(candc_s), np.nanmin(candcexp_s), np.nanmin(candc1344_s), np.nanmin(candc2688_s), np.nanmin(canduv_s), np.nanmin(canduvexp_s), np.nanmin(canduv1344_s), np.nanmin(canduv2688_s), np.nanmin(candd_s), np.nanmin(canddpd_s), np.nanmin(cande2_s), np.nanmin(cande2exp_s), np.nanmin(gan_s)))
    vmax = float(max(np.nanmax(gt_s), np.nanmax(cnn_s), np.nanmax(candb_s), np.nanmax(candbexp_s), np.nanmax(candc_s), np.nanmax(candcexp_s), np.nanmax(candc1344_s), np.nanmax(candc2688_s), np.nanmax(canduv_s), np.nanmax(canduvexp_s), np.nanmax(canduv1344_s), np.nanmax(canduv2688_s), np.nanmax(candd_s), np.nanmax(canddpd_s), np.nanmax(cande2_s), np.nanmax(cande2exp_s), np.nanmax(gan_s)))
    emax = float(max(np.nanmax(err_cnn), np.nanmax(err_canduv), np.nanmax(err_canduvexp), np.nanmax(err_canduv1344), np.nanmax(err_canduv2688), np.nanmax(err_candb), np.nanmax(err_candbexp), np.nanmax(err_candc), np.nanmax(err_candcexp), np.nanmax(err_candc1344), np.nanmax(err_candc2688), np.nanmax(err_candd), np.nanmax(err_canddpd), np.nanmax(err_cande2), np.nanmax(err_cande2exp), np.nanmax(err_gan)))
    dmax = float(max(np.nanmax(diff_c_uv), np.nanmax(diff_cexp_c), np.nanmax(diff_cexp_bexp), np.nanmax(diff_cexp_uvexp), np.nanmax(diff_c1344_cexp), np.nanmax(diff_c1344_c), np.nanmax(diff_c1344_uv1344), np.nanmax(diff_c2688_uv2688), np.nanmax(diff_c2688_c1344), np.nanmax(diff_uv2688_uv1344), np.nanmax(diff_uv1344_uvexp), np.nanmax(diff_e2_cc), np.nanmax(diff_dpd_cexp), np.nanmax(diff_e2exp_e2), np.nanmax(diff_e2exp_cexp)))
    if not np.isfinite(emax) or emax <= 0:
        emax = 1.0
    if not np.isfinite(dmax) or dmax <= 0:
        dmax = 1.0

    fig, axes = plt.subplots(1, 1, figsize=(5, 5.2))  # replaced below after fields are assembled
    fields = [
        gt_s, cnn_s, canduv_s, canduvexp_s, canduv1344_s, canduv2688_s, candb_s, candbexp_s, candc_s, candcexp_s, candc1344_s, candc2688_s, candd_s, canddpd_s, cande2_s, cande2exp_s, gan_s,
        err_cnn, err_canduv, err_canduvexp, err_canduv1344, err_canduv2688, err_candb, err_candbexp, err_candc, err_candcexp, err_candc1344, err_candc2688, err_candd, err_canddpd, err_cande2, err_cande2exp, err_gan,
        diff_cexp_uvexp, diff_c1344_uv1344, diff_c2688_uv2688, diff_c2688_c1344, diff_uv2688_uv1344, diff_uv1344_uvexp, diff_cexp_bexp, diff_cexp_c, diff_c1344_cexp, diff_c1344_c, diff_dpd_cexp, diff_e2exp_e2, diff_e2exp_cexp,
    ]
    titles = [
        "GT speed", "CNN speed", "CandUV speed", "CandUV-exp speed", "CandUV-1344 speed", "CandUV-2688 speed", "CandB speed", "CandB-exp speed", "CandC speed", "CandC-exp speed", "CandC-1344 speed", "CandC-2688 speed", "CandD speed", "CandDpd-exp speed", "CandE2 speed", "CandE2-exp speed", "GAN speed",
        "|CNN-GT|", "|UV-GT|", "|UVexp-GT|", "|UV1344-GT|", "|UV2688-GT|", "|B-GT|", "|Bexp-GT|", "|C-GT|", "|Cexp-GT|", "|C1344-GT|", "|C2688-GT|", "|D-GT|", "|Dpd-GT|", "|E2-GT|", "|E2exp-GT|", "|GAN-GT|",
        "|Cexp-UVexp|", "|C1344-UV1344|", "|C2688-UV2688|", "|C2688-C1344|", "|UV2688-UV1344|", "|UV1344-UVexp|", "|Cexp-Bexp|", "|Cexp-C|", "|C1344-Cexp|", "|C1344-C|", "|Dpd-Cexp|", "|E2exp-E2|", "|E2exp-Cexp|",
    ]
    fig, axes = plt.subplots(1, len(fields), figsize=(5 * len(fields), 5.2))
    if len(fields) == 1:
        axes = [axes]


    for ax, field, title in zip(axes, fields, titles):
        if title.startswith("|Cexp-") or title.startswith("|CandC-") or title.startswith("|CandE2-"):
            im = ax.imshow(field, origin="lower", vmin=0, vmax=dmax)
        elif title.startswith("|"):
            im = ax.imshow(field, origin="lower", vmin=0, vmax=emax)
        else:
            im = ax.imshow(field, origin="lower", vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    fig.suptitle(panel_title(sid, row, obs), fontsize=12)
    fig.text(0.5, 0.02, desc, ha="center", fontsize=9)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    fig.savefig(out, dpi=125)
    plt.close(fig)
    return True


# -----------------------------
# HTML
# -----------------------------

def _rank_methods_for_metric(sid: int, key: str, eval_rows: dict[int, dict[str, dict[str, str]]]) -> tuple[str, dict[str, str]]:
    """Return better-to-worse ranking and raw values for CNN/GAN/CandidateB/C/D."""
    metric_key = EVAL_METRIC_KEYS.get(key, key)
    methods = eval_rows.get(sid, {})
    values: list[tuple[float, str, str]] = []
    raw: dict[str, str] = {}

    for method, label in RANK_METHODS:
        row = methods.get(method) or methods.get(method.lower()) or methods.get(method.upper())
        if not row:
            raw[method] = ""
            continue
        v = fnum(row.get(metric_key, ""))
        raw[method] = num(row.get(metric_key, ""))
        if np.isfinite(v):
            values.append((v, method, label))

    if not values:
        return "", raw

    # All metrics listed in METRICS are lower-is-better absolute errors/distances.
    values.sort(key=lambda x: x[0])
    return " > ".join(label for _, _, label in values), raw


def metric_table(row: dict, sid: int, eval_rows: dict[int, dict[str, dict[str, str]]]) -> str:
    if not row and sid not in eval_rows:
        return '<details class="metric-box"><summary><b>Physics/domain metric breakdown</b> <span class="warn">unavailable</span></summary></details>'

    body = []
    top_counts = {method: 0 for method, _ in RANK_METHODS}
    legacy_cnn_count = legacy_gan_count = legacy_tie_count = 0

    for key, label, group in METRICS:
        rank, raw = _rank_methods_for_metric(sid, key, eval_rows)
        if rank:
            top_label = rank.split(" > ")[0]
            for method, label_name in RANK_METHODS:
                if top_label == label_name:
                    top_counts[method] += 1
                    break
            rank_cell = f"<span class='rank'>{H(rank)}</span>"
            cnn_val = raw.get("cnn", "")
            gan_val = raw.get("gan", "")
            b_val = raw.get("candidateB", "")
            c_val = raw.get("candidateC", "")
            cexp_val = raw.get("candidateC_expanded672", "")
            bexp_val = raw.get("candidateB_expanded672", "")
            uv_val = raw.get("candidateUV", "")
            uvexp_val = raw.get("candidateUV_expanded672", "")
            d_val = raw.get("candidateD", "")
            dpd_val = raw.get("candidateDpd_expanded672", "")
            e2_val = raw.get("candidateE2", "")
            e2exp_val = raw.get("candidateE2_expanded672", "")
        else:
            # Fallback to legacy CNN/GAN-only breakdown.
            w = norm(row.get(f"{key}_winner", ""))
            if w == "CNN":
                legacy_cnn_count += 1
                badge = '<span class="win cnn">CNN</span>'
            elif w == "GAN":
                legacy_gan_count += 1
                badge = '<span class="win gan">GAN</span>'
            elif w == "TIE":
                legacy_tie_count += 1
                badge = '<span class="win tie">TIE</span>'
            else:
                badge = "?"
            rank_cell = badge
            cnn_val = num(row.get(key + "_cnn", ""))
            gan_val = num(row.get(key + "_gan", ""))
            b_val = ""
            c_val = ""
            cexp_val = ""
            bexp_val = ""
            uv_val = ""
            uvexp_val = ""
            d_val = ""
            dpd_val = ""
            e2_val = ""
            e2exp_val = ""

        body.append(
            f"<tr><td>{H(label)}</td><td>{H(group)}</td>"
            f"<td class='num'>{H(cnn_val)}</td>"
            f"<td class='num'>{H(gan_val)}</td>"
            f"<td class='num'>{H(b_val)}</td>"
            f"<td class='num'>{H(c_val)}</td>"
            f"<td class='num'>{H(cexp_val)}</td>"
            f"<td class='num'>{H(bexp_val)}</td>"
            f"<td class='num'>{H(uv_val)}</td>"
            f"<td class='num'>{H(uvexp_val)}</td>"
            f"<td class='num'>{H(d_val)}</td>"
            f"<td class='num'>{H(dpd_val)}</td>"
            f"<td class='num'>{H(e2_val)}</td>"
            f"<td class='num'>{H(e2exp_val)}</td>"
            f"<td>{rank_cell}</td></tr>"
        )

    count_line = (
        f"Top-ranked: Candidate C {top_counts['candidateC']} | "
        f"Candidate C-expanded-672 {top_counts['candidateC_expanded672']} | Candidate C-expanded-1344 {top_counts['candidateC_expanded1344']} | "
        f"Candidate B-expanded-672 {top_counts['candidateB_expanded672']} | "
        f"Candidate UV {top_counts['candidateUV']} | "
        f"Candidate UV-expanded-672 {top_counts['candidateUV_expanded672']} | "
        f"Candidate D {top_counts['candidateD']} | Dpd-exp {top_counts['candidateDpd_expanded672']} | "
        f"Candidate E2 {top_counts['candidateE2']} | E2-exp {top_counts['candidateE2_expanded672']} | "
        f"Candidate B {top_counts['candidateB']} | "
        f"CNN {top_counts['cnn']} | GAN {top_counts['gan']}"
    )
    if legacy_cnn_count or legacy_gan_count or legacy_tie_count:
        count_line += f" | legacy CNN {legacy_cnn_count} GAN {legacy_gan_count} ties {legacy_tie_count}"

    return f"""
    <details class="metric-box">
      <summary><b>Physics/domain metric breakdown</b>
        <span class="count">{H(count_line)}</span>
      </summary>
      <p class="muted">Lower is better for these physics/domain error metrics. Ranking is better -> worse using candidate evaluation CSVs when available.</p>
      <table class="metrics">
        <thead><tr><th>Measure</th><th>Group</th><th>CNN</th><th>GAN</th><th>Candidate B</th><th>Candidate C</th><th>Candidate C-expanded-672</th><th>Candidate C-expanded-1344</th><th>Candidate B-expanded-672</th><th>Candidate UV</th><th>Candidate UV-expanded-672</th><th>Candidate D</th><th>Candidate Dpd-exp</th><th>Candidate E2</th><th>Candidate E2-exp</th><th>Ranking / Winner</th></tr></thead>
        <tbody>{''.join(body)}</tbody>
      </table>
    </details>
    """


def candidate_topology_box(obs: dict, candidate: str, label: str, open_box: bool = False) -> str:
    """Small per-sample topology summary for Candidate B/C."""
    if not obs.get(f"pd_distance_{candidate}") and not obs.get(f"mt_distance_{candidate}"):
        return ""

    pd_vals = (
        num(obs.get("pd_distance_cnn", "")),
        num(obs.get("pd_distance_gan", "")),
        num(obs.get(f"pd_distance_{candidate}", "")),
    )
    mt_vals = (
        num(obs.get("mt_distance_cnn", "")),
        num(obs.get("mt_distance_gan", "")),
        num(obs.get(f"mt_distance_{candidate}", "")),
    )
    pd_before = obs.get("pd_winner_before", "")
    pd_after = obs.get(f"pd_winner_after_{candidate}", "")
    mt_before = obs.get("mt_winner_before", "")
    mt_after = obs.get(f"mt_winner_after_{candidate}", "")
    was_mt_gan = "yes" if boolish(obs.get("was_mt_gan_win_before", "")) else "no"
    open_attr = " open" if open_box else ""

    return f"""
    <details class="metric-box"{open_attr}>
      <summary><b>{H(label)} topology comparison</b>
        <span class="count">MT-GAN baseline case: {H(was_mt_gan)}</span>
      </summary>
      <table class="metrics">
        <thead><tr><th>Distance</th><th>CNN</th><th>GAN</th><th>{H(label)}</th><th>Winner before &rarr; after</th></tr></thead>
        <tbody>
          <tr><td>PD bottleneck</td><td class="num">{H(pd_vals[0])}</td><td class="num">{H(pd_vals[1])}</td><td class="num">{H(pd_vals[2])}</td><td>{H(pd_before)} &rarr; {H(pd_after)}</td></tr>
          <tr><td>MT Wasserstein</td><td class="num">{H(mt_vals[0])}</td><td class="num">{H(mt_vals[1])}</td><td class="num">{H(mt_vals[2])}</td><td>{H(mt_before)} &rarr; {H(mt_after)}</td></tr>
        </tbody>
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
        f"Physics: {H(norm(pick(row, obs, 'configured_physics_group_winner')) or '?')}"
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

          {candidate_topology_box(obs, "candidateB", "Candidate B", open_box=False)}
          {candidate_topology_box(obs, "candidateB_expanded672", "Candidate B-expanded-672", open_box=False)}
          {candidate_topology_box(obs, "candidateC", "Candidate C", open_box=True)}
          {candidate_topology_box(obs, "candidateC_expanded672", "Candidate C-expanded-672", open_box=False)}
          {candidate_topology_box(obs, "candidateUV", "Candidate UV", open_box=False)}
          {candidate_topology_box(obs, "candidateUV_expanded672", "Candidate UV-expanded-672", open_box=False)}
          {candidate_topology_box(obs, "candidateUV_expanded1344", "Candidate UV-expanded-1344", open_box=False)}
          {candidate_topology_box(obs, "candidateUV_expanded2688", "Candidate UV-expanded-2688", open_box=False)}
          {candidate_topology_box(obs, "candidateD", "Candidate D", open_box=False)}
          {candidate_topology_box(obs, "candidateDpd_expanded672", "Candidate Dpd-expanded-672", open_box=False)}
          {candidate_topology_box(obs, "candidateE2", "Candidate E2", open_box=False)}
          {candidate_topology_box(obs, "candidateE2_expanded672", "Candidate E2-expanded-672", open_box=False)}

          {metric_table(row, sid, entry.get("eval_rows", {}))}

          <div class="links">{' '.join(links)}</div>
        </div>
        <div class="thumb">
          {f"<a href='{H(crop)}' target='_blank'><img src='{H(crop)}'></a>" if crop else "<p class='muted'>No panel available.</p>"}
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
<title>TopoAware SR visual inspection index</title>
<style>
body {{ margin:0; background:#f7f7f8; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; color:#111; }}
header {{ position:sticky; top:0; z-index:10; background:white; border-bottom:1px solid #ddd; padding:16px 24px; }}
h1 {{ margin:0 0 10px 0; font-size:30px; }}
button {{ border:1px solid #ddd; background:white; border-radius:999px; padding:8px 12px; margin:0 6px 6px 0; cursor:pointer; }}
main {{ padding:16px; }}
.card {{ background:white; border:1px solid #ddd; border-radius:14px; padding:18px; margin-bottom:18px; box-shadow:0 1px 4px rgba(0,0,0,.06); }}
.card-grid {{ display:grid; grid-template-columns:minmax(520px,1fr) minmax(360px,.8fr); gap:20px; align-items:start; }}
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
.metrics th {{ text-align:left; background:#f0f0f0; padding:6px; }}
.metrics td {{ border-top:1px solid #e5e5e5; padding:6px; }}
.num {{ text-align:right; font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; }}
.win {{ display:inline-block; border-radius:999px; padding:3px 8px; font-weight:700; font-size:12px; }}
.rank {{ font-weight:700; }}
.cnn {{ background:#dff7e9; color:#11733b; border:1px solid #a8e6c1; }}
.gan {{ background:#fff0d9; color:#a35b00; border:1px solid #ffd39a; }}
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
  <h1>TopoAware SR visual inspection index</h1>
  <button onclick="showOnly('all')">All ({len(entries)})</button>
  <button onclick="showOnly('tag-forced-qualitative-set')">Forced qualitative set ({count('forced_qualitative_set')})</button>
  <button onclick="showOnly('tag-mt-gan-diagnostic')">MT picks GAN ({count('mt_gan_diagnostic')})</button>
  <button onclick="showOnly('tag-mt-gan-flipped-to-candidateB')">MT-GAN &rarr; Candidate B ({count('mt_gan_flipped_to_candidateB')})</button>
  <button onclick="showOnly('tag-mt-gan-flipped-to-candidateC')">MT-GAN &rarr; Candidate C ({count('mt_gan_flipped_to_candidateC')})</button>
  <button onclick="showOnly('tag-mt-gan-flipped-to-candidateC-expanded672')">MT-GAN &rarr; Candidate C-exp672 ({count('mt_gan_flipped_to_candidateC_expanded672')})</button>
  <button onclick="showOnly('tag-mt-gan-flipped-to-candidateC-expanded1344')">MT-GAN &rarr; Candidate C-exp1344 ({count('mt_gan_flipped_to_candidateC_expanded1344')})</button>
  <button onclick="showOnly('tag-mt-gan-flipped-to-candidateUV')">MT-GAN &rarr; Candidate UV ({count('mt_gan_flipped_to_candidateUV')})</button>
  <button onclick="showOnly('tag-mt-gan-flipped-to-candidateD')">MT-GAN &rarr; Candidate D ({count('mt_gan_flipped_to_candidateD')})</button>
  <button onclick="showOnly('tag-mt-gan-flipped-to-candidateDpd-expanded672')">MT-GAN &rarr; Candidate Dpd-exp672 ({count('mt_gan_flipped_to_candidateDpd_expanded672')})</button>
  <button onclick="showOnly('tag-mt-gan-flipped-to-candidateE2')">MT-GAN &rarr; Candidate E2 ({count('mt_gan_flipped_to_candidateE2')})</button>
  <button onclick="showOnly('tag-candidateB-mt-winner')">Candidate B MT winner ({count('candidateB_mt_winner')})</button>
  <button onclick="showOnly('tag-candidateC-mt-winner')">Candidate C MT winner ({count('candidateC_mt_winner')})</button>
  <button onclick="showOnly('tag-candidateC-expanded672-mt-winner')">Candidate C-exp672 MT winner ({count('candidateC_expanded672_mt_winner')})</button>
  <button onclick="showOnly('tag-candidateC-expanded1344-mt-winner')">Candidate C-exp1344 MT winner ({count('candidateC_expanded1344_mt_winner')})</button>
  <button onclick="showOnly('tag-candidateUV-mt-winner')">Candidate UV MT winner ({count('candidateUV_mt_winner')})</button>
  <button onclick="showOnly('tag-candidateD-mt-winner')">Candidate D MT winner ({count('candidateD_mt_winner')})</button>
  <button onclick="showOnly('tag-candidateDpd-expanded672-mt-winner')">Candidate Dpd-expanded MT winner ({count('candidateDpd_expanded672_mt_winner')})</button>
  <button onclick="showOnly('tag-candidateE2-mt-winner')">Candidate E2 MT winner ({count('candidateE2_mt_winner')})</button>
  <button onclick="showOnly('tag-candidateB-pd-improves-vs-cnn')">Candidate B improves PD vs CNN ({count('candidateB_pd_improves_vs_cnn')})</button>
  <button onclick="showOnly('tag-candidateC-pd-improves-vs-cnn')">Candidate C improves PD vs CNN ({count('candidateC_pd_improves_vs_cnn')})</button>
  <button onclick="showOnly('tag-candidateC-expanded672-pd-improves-vs-cnn')">Candidate C-exp672 improves PD vs CNN ({count('candidateC_expanded672_pd_improves_vs_cnn')})</button>
  <button onclick="showOnly('tag-candidateC-expanded1344-pd-improves-vs-cnn')">Candidate C-exp1344 improves PD vs CNN ({count('candidateC_expanded1344_pd_improves_vs_cnn')})</button>
  <button onclick="showOnly('tag-candidateUV-pd-improves-vs-cnn')">Candidate UV improves PD vs CNN ({count('candidateUV_pd_improves_vs_cnn')})</button>
  <button onclick="showOnly('tag-candidateD-pd-improves-vs-cnn')">Candidate D improves PD vs CNN ({count('candidateD_pd_improves_vs_cnn')})</button>
  <button onclick="showOnly('tag-candidateDpd-expanded672-pd-improves-vs-cnn')">Candidate Dpd-exp improves PD vs CNN ({count('candidateDpd_expanded672_pd_improves_vs_cnn')})</button>
  <button onclick="showOnly('tag-candidateE2-pd-improves-vs-cnn')">Candidate E2 improves PD vs CNN ({count('candidateE2_pd_improves_vs_cnn')})</button>
  <button onclick="showOnly('tag-candidateB-mt-improves-vs-cnn')">Candidate B improves MT vs CNN ({count('candidateB_mt_improves_vs_cnn')})</button>
  <button onclick="showOnly('tag-candidateC-mt-improves-vs-cnn')">Candidate C improves MT vs CNN ({count('candidateC_mt_improves_vs_cnn')})</button>
  <button onclick="showOnly('tag-candidateC-expanded672-mt-improves-vs-cnn')">Candidate C-exp672 improves MT vs CNN ({count('candidateC_expanded672_mt_improves_vs_cnn')})</button>
  <button onclick="showOnly('tag-candidateC-expanded1344-mt-improves-vs-cnn')">Candidate C-exp1344 improves MT vs CNN ({count('candidateC_expanded1344_mt_improves_vs_cnn')})</button>
  <button onclick="showOnly('tag-mt-gan-flipped-to-candidateUV-expanded1344')">MT-GAN &rarr; Candidate UV-exp1344 ({count('mt_gan_flipped_to_candidateUV_expanded1344')})</button>
  <button onclick="showOnly('tag-candidateUV-expanded1344-pd-improves-vs-cnn')">Candidate UV-exp1344 improves PD vs CNN ({count('candidateUV_expanded1344_pd_improves_vs_cnn')})</button>
  <button onclick="showOnly('tag-candidateUV-expanded1344-mt-improves-vs-cnn')">Candidate UV-exp1344 improves MT vs CNN ({count('candidateUV_expanded1344_mt_improves_vs_cnn')})</button>
  <button onclick="showOnly('tag-candidateUV-mt-improves-vs-cnn')">Candidate UV improves MT vs CNN ({count('candidateUV_mt_improves_vs_cnn')})</button>
  <button onclick="showOnly('tag-candidateD-mt-improves-vs-cnn')">Candidate D improves MT vs CNN ({count('candidateD_mt_improves_vs_cnn')})</button>
  <button onclick="showOnly('tag-candidateDpd-expanded672-mt-improves-vs-cnn')">Candidate Dpd-exp improves MT vs CNN ({count('candidateDpd_expanded672_mt_improves_vs_cnn')})</button>
  <button onclick="showOnly('tag-candidateE2-mt-improves-vs-cnn')">Candidate E2 improves MT vs CNN ({count('candidateE2_mt_improves_vs_cnn')})</button>
  <button onclick="showOnly('tag-candidateC-mt-improves-vs-candidateB')">Candidate C improves MT vs B ({count('candidateC_mt_improves_vs_candidateB')})</button>
  <button onclick="showOnly('tag-candidateC-expanded672-mt-improves-vs-candidateC')">Candidate C-exp672 improves MT vs C ({count('candidateC_expanded672_mt_improves_vs_candidateC')})</button>
  <button onclick="showOnly('tag-candidateUV-mt-improves-vs-candidateC')">Candidate UV improves MT vs C ({count('candidateUV_mt_improves_vs_candidateC')})</button>
  <button onclick="showOnly('tag-candidateD-mt-improves-vs-candidateC')">Candidate D improves MT vs C ({count('candidateD_mt_improves_vs_candidateC')})</button>
  <button onclick="showOnly('tag-candidateE2-mt-improves-vs-candidateC')">Candidate E2 improves MT vs C ({count('candidateE2_mt_improves_vs_candidateC')})</button>
  <button onclick="showOnly('tag-mt-gan-flipped-to-candidateB-expanded672')">MT-GAN &rarr; Candidate B-exp672 ({count('mt_gan_flipped_to_candidateB_expanded672')})</button>
  <button onclick="showOnly('tag-mt-gan-flipped-to-candidateUV-expanded672')">MT-GAN &rarr; Candidate UV-exp672 ({count('mt_gan_flipped_to_candidateUV_expanded672')})</button>
  <button onclick="showOnly('tag-candidateB-expanded672-pd-improves-vs-cnn')">Candidate B-exp672 improves PD vs CNN ({count('candidateB_expanded672_pd_improves_vs_cnn')})</button>
  <button onclick="showOnly('tag-candidateUV-expanded672-pd-improves-vs-cnn')">Candidate UV-exp672 improves PD vs CNN ({count('candidateUV_expanded672_pd_improves_vs_cnn')})</button>
  <button onclick="showOnly('tag-candidateB-expanded672-mt-improves-vs-cnn')">Candidate B-exp672 improves MT vs CNN ({count('candidateB_expanded672_mt_improves_vs_cnn')})</button>
  <button onclick="showOnly('tag-candidateUV-expanded672-mt-improves-vs-cnn')">Candidate UV-exp672 improves MT vs CNN ({count('candidateUV_expanded672_mt_improves_vs_cnn')})</button>
  <button onclick="showOnly('tag-topology-consensus-cnn')">PD=MT=CNN ({count('topology_consensus_cnn')})</button>
  <button onclick="showOnly('tag-gan-metric-majority')">GAN metric majority ({count('gan_metric_majority')})</button>
  <button onclick="showOnly('tag-gan-majority-mt-rejects-gan')">GAN majority but MT≠GAN ({count('gan_majority_mt_rejects_gan')})</button>
  <button onclick="showOnly('tag-adjacent-cluster-10-13')">Cluster 10–13</button>
  <button onclick="showOnly('tag-adjacent-cluster-76-78')">Cluster 76–78</button>
  <button onclick="showOnly('tag-adjacent-cluster-90-93')">Cluster 90–93</button>
  <button onclick="showOnly('tag-adjacent-cluster-161-164')">Cluster 161–164</button>
  <p class="muted">Each panel shows GT speed | CNN speed | Candidate UV | Candidate UV-expanded-672 | Candidate B | Candidate B-expanded-672 | Candidate C | Candidate C-expanded-672 | Candidate C-expanded-1344 | Candidate D | Candidate Dpd-expanded-672 | Candidate E2 | Candidate E2-expanded-672 | GAN, followed by absolute-error maps and expanded-candidate difference maps.</p>
</header>
<main>
{cards}
</main>
</body>
</html>
"""
    (OUTDIR / "index.html").write_text(page, encoding="utf-8")


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="Generate index/panels for all samples in the metric table.")
    parser.add_argument("--samples", default="", help="Comma-separated sample ids to generate instead of default selection.")
    parser.add_argument("--no-panels", action="store_true", help="Only rebuild index/manifest using existing PNGs.")
    args = parser.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    CROP_DIR.mkdir(parents=True, exist_ok=True)
    FULL_DIR.mkdir(parents=True, exist_ok=True)

    metrics = load_metric_rows()
    obs = load_obs_rows()
    eval_rows = load_candidate_eval_rows()

    candb_topology = load_candidate_topology_rows(CANDIDATEB_TOPOLOGY_COMPARISON, "candidateB", "Candidate B")
    candbexp_topology = load_candidate_topology_rows(CANDIDATEB_EXPANDED672_TOPOLOGY_COMPARISON, "candidateB_expanded672", "Candidate B-expanded-672")
    candc_topology = load_candidate_topology_rows(CANDIDATEC_TOPOLOGY_COMPARISON, "candidateC", "Candidate C")
    candcexp_topology = load_candidate_topology_rows(CANDIDATEC_EXPANDED672_TOPOLOGY_COMPARISON, "candidateC_expanded672", "Candidate C-expanded-672")
    candc1344_topology = load_candidate_topology_rows(CANDIDATEC_EXPANDED1344_TOPOLOGY_COMPARISON, "candidateC_expanded1344", "Candidate C-expanded-1344")
    candc2688_topology = load_candidate_topology_rows(CANDIDATEC_EXPANDED2688_TOPOLOGY_COMPARISON, "candidateC_expanded2688", "Candidate C-expanded-2688")
    canduv_topology = load_candidate_topology_rows(CANDIDATEUV_TOPOLOGY_COMPARISON, "candidateUV", "Candidate UV")
    canduvexp_topology = load_candidate_topology_rows(CANDIDATEUV_EXPANDED672_TOPOLOGY_COMPARISON, "candidateUV_expanded672", "Candidate UV-expanded-672")
    canduv1344_topology = load_candidate_topology_rows(CANDIDATEUV_EXPANDED1344_TOPOLOGY_COMPARISON, "candidateUV_expanded1344", "Candidate UV-expanded-1344")
    canduv2688_topology = load_candidate_topology_rows(CANDIDATEUV_EXPANDED2688_TOPOLOGY_COMPARISON, "candidateUV_expanded2688", "Candidate UV-expanded-2688")
    candd_topology = load_candidate_topology_rows(CANDIDATED_TOPOLOGY_COMPARISON, "candidateD", "Candidate D")
    canddpd_topology = load_candidate_topology_rows(CANDIDATEDPD_EXPANDED672_TOPOLOGY_COMPARISON, "candidateDpd_expanded672", "Candidate Dpd-expanded-672")
    cande2_topology = load_candidate_topology_rows(CANDIDATEE2_TOPOLOGY_COMPARISON, "candidateE2", "Candidate E2")
    cande2exp_topology = load_candidate_topology_rows(CANDIDATEE2_EXPANDED672_TOPOLOGY_COMPARISON, "candidateE2_expanded672", "Candidate E2-expanded-672")
    for topology_rows in (candb_topology, candbexp_topology, candc_topology, candcexp_topology, candc1344_topology, candc2688_topology, canduv_topology, canduvexp_topology, canduv1344_topology, canduv2688_topology, candd_topology, canddpd_topology, cande2_topology, cande2exp_topology):
        for sid, r in topology_rows.items():
            obs.setdefault(sid, {})
            for k, v in r.items():
                if v is not None and str(v).strip():
                    obs[sid][k] = v

    samples = select_samples(args, metrics, obs)

    if not samples:
        raise RuntimeError("No samples selected.")

    print(f"repo_root={ROOT}")
    print(f"outdir={OUTDIR}")
    print(f"selected_samples={len(samples)}")
    print("forced/extra samples:", " ".join(map(str, sorted(FORCED))))

    gt = cnn = gan = candb = candbexp = candc = candcexp = candc1344 = candc2688 = canduv = canduvexp = canduv1344 = canduv2688 = candd = canddpd = cande2 = cande2exp = pos = candb_pos = candbexp_pos = candc_pos = candcexp_pos = candc1344_pos = candc2688_pos = canduv_pos = canduvexp_pos = canduv1344_pos = canduv2688_pos = candd_pos = canddpd_pos = cande2_pos = cande2exp_pos = None
    if not args.no_panels:
        gt, cnn, gan, candb, candbexp, candc, candcexp, candc1344, candc2688, canduv, canduvexp, canduv1344, canduv2688, candd, canddpd, cande2, cande2exp, pos, candb_pos, candbexp_pos, candc_pos, candcexp_pos, candc1344_pos, candc2688_pos, canduv_pos, canduvexp_pos, canduv1344_pos, canduv2688_pos, candd_pos, canddpd_pos, cande2_pos, cande2exp_pos = load_arrays()

    entries = []
    manifest = []

    for sid in samples:
        row = metrics.get(sid, {})
        ob = obs.get(sid, {})

        crop_path = CROP_DIR / f"sample_{sid:03d}_crop.png"
        full_path = FULL_DIR / f"sample_{sid:03d}_full.png"

        crop_ok = crop_path.exists()
        full_ok = full_path.exists()

        if not args.no_panels:
            crop_ok = make_panel(sid, gt, cnn, gan, candb, candbexp, candc, candcexp, candc1344, candc2688, canduv, canduvexp, canduv1344, canduv2688, candd, canddpd, cande2, cande2exp, pos, candb_pos, candbexp_pos, candc_pos, candcexp_pos, candc1344_pos, candc2688_pos, canduv_pos, canduvexp_pos, canduv1344_pos, canduv2688_pos, candd_pos, canddpd_pos, cande2_pos, cande2exp_pos, row, ob, (0, 160, 0, 160), crop_path)
            full_ok = make_panel(sid, gt, cnn, gan, candb, candbexp, candc, candcexp, candc1344, candc2688, canduv, canduvexp, canduv1344, canduv2688, candd, canddpd, cande2, cande2exp, pos, candb_pos, candbexp_pos, candc_pos, candcexp_pos, candc1344_pos, candc2688_pos, canduv_pos, canduvexp_pos, canduv1344_pos, canduv2688_pos, candd_pos, canddpd_pos, cande2_pos, cande2exp_pos, row, ob, None, full_path)

        crop_rel = crop_path.relative_to(OUTDIR).as_posix() if crop_ok else ""
        full_rel = full_path.relative_to(OUTDIR).as_posix() if full_ok else ""

        groups = infer_groups(sid, row, ob)
        q = question(sid, row, ob)

        entry = {
            "sample_idx": sid,
            "row": row,
            "obs": ob,
            "eval_rows": eval_rows,
            "groups": groups,
            "question": q,
            "crop_panel": crop_rel,
            "full_panel": full_rel,
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
            "configured_physics_group_winner": norm(pick(row, ob, "configured_physics_group_winner")),
            "cnn_metric_wins": row.get("cnn_metric_wins", ""),
            "gan_metric_wins": row.get("gan_metric_wins", ""),
            "overall_metric_majority": row.get("overall_metric_majority", ""),
            "question": q,
            "groups": ";".join(groups),
            "crop_panel": crop_rel,
            "full_panel": full_rel,
            "mt_winner_after_candidateB": ob.get("mt_winner_after_candidateB", ""),
            "pd_winner_after_candidateB": ob.get("pd_winner_after_candidateB", ""),
            "pd_distance_candidateB": ob.get("pd_distance_candidateB", ""),
            "mt_distance_candidateB": ob.get("mt_distance_candidateB", ""),
            "mt_winner_after_candidateC": ob.get("mt_winner_after_candidateC", ""),
            "pd_winner_after_candidateC": ob.get("pd_winner_after_candidateC", ""),
            "pd_distance_candidateC": ob.get("pd_distance_candidateC", ""),
            "mt_distance_candidateC": ob.get("mt_distance_candidateC", ""),
            "mt_winner_after_candidateC_expanded672": ob.get("mt_winner_after_candidateC_expanded672", ""),
            "pd_winner_after_candidateC_expanded672": ob.get("pd_winner_after_candidateC_expanded672", ""),
            "pd_distance_candidateC_expanded672": ob.get("pd_distance_candidateC_expanded672", ""),
            "mt_distance_candidateC_expanded672": ob.get("mt_distance_candidateC_expanded672", ""),
            "mt_winner_after_candidateC_expanded1344": ob.get("mt_winner_after_candidateC_expanded1344", ""),
            "pd_winner_after_candidateC_expanded1344": ob.get("pd_winner_after_candidateC_expanded1344", ""),
            "pd_distance_candidateC_expanded1344": ob.get("pd_distance_candidateC_expanded1344", ""),
            "mt_distance_candidateC_expanded1344": ob.get("mt_distance_candidateC_expanded1344", ""),
            "mt_winner_after_candidateC_expanded2688": ob.get("mt_winner_after_candidateC_expanded2688", ""),
            "pd_winner_after_candidateC_expanded2688": ob.get("pd_winner_after_candidateC_expanded2688", ""),
            "pd_distance_candidateC_expanded2688": ob.get("pd_distance_candidateC_expanded2688", ""),
            "mt_distance_candidateC_expanded2688": ob.get("mt_distance_candidateC_expanded2688", ""),
            "mt_winner_after_candidateUV_expanded1344": ob.get("mt_winner_after_candidateUV_expanded1344", ""),
            "pd_winner_after_candidateUV_expanded1344": ob.get("pd_winner_after_candidateUV_expanded1344", ""),
            "pd_distance_candidateUV_expanded1344": ob.get("pd_distance_candidateUV_expanded1344", ""),
            "mt_distance_candidateUV_expanded1344": ob.get("mt_distance_candidateUV_expanded1344", ""),
            "mt_winner_after_candidateUV_expanded2688": ob.get("mt_winner_after_candidateUV_expanded2688", ""),
            "pd_winner_after_candidateUV_expanded2688": ob.get("pd_winner_after_candidateUV_expanded2688", ""),
            "pd_distance_candidateUV_expanded2688": ob.get("pd_distance_candidateUV_expanded2688", ""),
            "mt_distance_candidateUV_expanded2688": ob.get("mt_distance_candidateUV_expanded2688", ""),
            "mt_winner_after_candidateUV": ob.get("mt_winner_after_candidateUV", ""),
            "pd_winner_after_candidateUV": ob.get("pd_winner_after_candidateUV", ""),
            "pd_distance_candidateUV": ob.get("pd_distance_candidateUV", ""),
            "mt_distance_candidateUV": ob.get("mt_distance_candidateUV", ""),
            "mt_winner_after_candidateD": ob.get("mt_winner_after_candidateD", ""),
            "pd_winner_after_candidateD": ob.get("pd_winner_after_candidateD", ""),
            "pd_distance_candidateD": ob.get("pd_distance_candidateD", ""),
            "mt_distance_candidateD": ob.get("mt_distance_candidateD", ""),
            "mt_winner_after_candidateDpd_expanded672": ob.get("mt_winner_after_candidateDpd_expanded672", ""),
            "pd_winner_after_candidateDpd_expanded672": ob.get("pd_winner_after_candidateDpd_expanded672", ""),
            "pd_distance_candidateDpd_expanded672": ob.get("pd_distance_candidateDpd_expanded672", ""),
            "mt_distance_candidateDpd_expanded672": ob.get("mt_distance_candidateDpd_expanded672", ""),
            "mt_winner_after_candidateE2": ob.get("mt_winner_after_candidateE2", ""),
            "pd_winner_after_candidateE2": ob.get("pd_winner_after_candidateE2", ""),
            "pd_distance_candidateE2": ob.get("pd_distance_candidateE2", ""),
            "mt_distance_candidateE2": ob.get("mt_distance_candidateE2", ""),
            "mt_winner_after_candidateE2_expanded672": ob.get("mt_winner_after_candidateE2_expanded672", ""),
            "pd_winner_after_candidateE2_expanded672": ob.get("pd_winner_after_candidateE2_expanded672", ""),
            "pd_distance_candidateE2_expanded672": ob.get("pd_distance_candidateE2_expanded672", ""),
            "mt_distance_candidateE2_expanded672": ob.get("mt_distance_candidateE2_expanded672", ""),
            "was_mt_gan_win_before": ob.get("was_mt_gan_win_before", ""),
            "forced_reason": FORCED.get(sid, ""),
        })

    entries.sort(key=lambda e: e["sample_idx"])
    manifest.sort(key=lambda r: int(r["sample_idx"]))

    write_csv(
        OUTDIR / "visual_inspection_manifest.csv",
        manifest,
        [
            "sample_idx", "psnr_winner", "ssim_winner", "pd_winner", "mt_winner",
            "direct_error_group_winner", "distributional_group_winner",
            "tail_group_winner", "configured_physics_group_winner",
            "cnn_metric_wins", "gan_metric_wins", "overall_metric_majority",
            "question", "groups", "crop_panel", "full_panel",
            "mt_winner_after_candidateB", "pd_winner_after_candidateB",
            "pd_distance_candidateB", "mt_distance_candidateB",
            "mt_winner_after_candidateC", "pd_winner_after_candidateC",
            "pd_distance_candidateC", "mt_distance_candidateC",
            "mt_winner_after_candidateC_expanded672", "pd_winner_after_candidateC_expanded672",
            "pd_distance_candidateC_expanded672", "mt_distance_candidateC_expanded672",
            "mt_winner_after_candidateC_expanded1344", "pd_winner_after_candidateC_expanded1344",
            "pd_distance_candidateC_expanded1344", "mt_distance_candidateC_expanded1344",
            "mt_winner_after_candidateC_expanded2688", "pd_winner_after_candidateC_expanded2688",
            "pd_distance_candidateC_expanded2688", "mt_distance_candidateC_expanded2688",
            "mt_winner_after_candidateUV_expanded1344", "pd_winner_after_candidateUV_expanded1344",
            "pd_distance_candidateUV_expanded1344", "mt_distance_candidateUV_expanded1344",
            "mt_winner_after_candidateUV_expanded2688", "pd_winner_after_candidateUV_expanded2688",
            "pd_distance_candidateUV_expanded2688", "mt_distance_candidateUV_expanded2688",
            "mt_winner_after_candidateUV", "pd_winner_after_candidateUV",
            "pd_distance_candidateUV", "mt_distance_candidateUV",
            "mt_winner_after_candidateD", "pd_winner_after_candidateD",
            "pd_distance_candidateD", "mt_distance_candidateD",
            "mt_winner_after_candidateDpd_expanded672", "pd_winner_after_candidateDpd_expanded672",
            "pd_distance_candidateDpd_expanded672", "mt_distance_candidateDpd_expanded672",
            "mt_winner_after_candidateE2", "pd_winner_after_candidateE2",
            "pd_distance_candidateE2", "mt_distance_candidateE2",
            "mt_winner_after_candidateE2_expanded672", "pd_winner_after_candidateE2_expanded672",
            "pd_distance_candidateE2_expanded672", "mt_distance_candidateE2_expanded672",
            "was_mt_gan_win_before", "forced_reason"
        ],
    )
    write_index(entries)

    print(f"Wrote {OUTDIR / 'index.html'}")
    print(f"Wrote {OUTDIR / 'visual_inspection_manifest.csv'}")
    print(f"Wrote panels under {CROP_DIR} and {FULL_DIR}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
