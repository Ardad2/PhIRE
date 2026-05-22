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
  - Adds Candidate B, Candidate C, Candidate D, and Candidate E2 panels and topology comparison metadata when available.
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
CANDIDATEC_DIR = ROOT / "data_out" / "wind_finetune_pilot_candidateC"
CANDIDATED_DIR = ROOT / "data_out" / "wind_finetune_pilot_candidateD"
CANDIDATEE2_DIR = ROOT / "data_out" / "wind_finetune_pilot_candidateE2"

CANDIDATEB_TOPOLOGY_COMPARISON = (
    ROOT
    / "ttk_runs_fixed"
    / "topology_finetuning"
    / "candidateB_topology"
    / "candidateB_topology_comparison.csv"
)
CANDIDATEC_TOPOLOGY_COMPARISON = (
    ROOT
    / "ttk_runs_fixed"
    / "topology_finetuning"
    / "candidateC_topology"
    / "candidateC_topology_comparison.csv"
)
CANDIDATED_TOPOLOGY_COMPARISON = (
    ROOT
    / "ttk_runs_fixed"
    / "topology_finetuning"
    / "candidateD_topology"
    / "candidateD_topology_comparison.csv"
)
CANDIDATEE2_TOPOLOGY_COMPARISON = (
    ROOT
    / "ttk_runs_fixed"
    / "topology_finetuning"
    / "candidateE2_topology"
    / "candidateE2_topology_comparison.csv"
)

CANDIDATEB_EVAL = (
    ROOT
    / "ttk_runs_fixed"
    / "topology_finetuning"
    / "candidateB_eval"
    / "all_sample_metrics_candidateB.csv"
)
CANDIDATEC_EVAL = (
    ROOT
    / "ttk_runs_fixed"
    / "topology_finetuning"
    / "candidateC_eval"
    / "all_sample_metrics_candidateC.csv"
)
CANDIDATED_EVAL = (
    ROOT
    / "ttk_runs_fixed"
    / "topology_finetuning"
    / "candidateD_eval"
    / "all_sample_metrics_candidateD.csv"
)
CANDIDATEE2_EVAL = (
    ROOT
    / "ttk_runs_fixed"
    / "topology_finetuning"
    / "candidateE2_eval"
    / "all_sample_metrics_candidateE2.csv"
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
CANDIDATED_MT_GAN_RECOVERED = {25}
CANDIDATEE2_MT_GAN_RECOVERED = {25}

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
    ("candidateD", "Candidate D"),
    ("candidateE2", "Candidate E2"),
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

    for path in (CANDIDATEB_EVAL, CANDIDATEC_EVAL, CANDIDATED_EVAL, CANDIDATEE2_EVAL):
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
            elif lk == "candidated" and "candidateD" not in methods:
                methods["candidateD"] = methods[k]
            elif lk == "candidatee2" and "candidateE2" not in methods:
                methods["candidateE2"] = methods[k]

    if out:
        print(f"Loaded candidate evaluation metric rows for {len(out)} samples")
    else:
        print("WARNING: Candidate B/C evaluation CSVs not found; 4-way metric ranking unavailable.")
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
    if sid in CANDIDATEE2_MT_GAN_RECOVERED:
        return "Candidate E2 is the corrected TTK critical-pair loss recovered MT-GAN case: does fixed critical-pair supervision change visible structure, or is the improvement isolated?"
    if sid in CANDIDATED_MT_GAN_RECOVERED:
        return "Candidate D is the true PD-loss pilot's only recovered MT-GAN case: does the PD-refiner change visible structure, or is the improvement localized and metric-specific?"
    if sid in CANDIDATEC_MT_GAN_RECOVERED:
        return "Candidate C recovered this original MT-GAN case: does the critical-value/topological-extrema proxy improve merge-tree hierarchy while retaining CNN-like fidelity?"
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
    candc_p = CANDIDATEC_DIR / "dataSR.npy"
    candd_p = CANDIDATED_DIR / "dataSR.npy"
    cande2_p = CANDIDATEE2_DIR / "dataSR.npy"
    missing = [str(p) for p in (gt_p, cnn_p, gan_p, candb_p, candc_p, candd_p, cande2_p) if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing NPY arrays:\n" + "\n".join(missing))

    gt = np.load(gt_p, mmap_mode="r")
    cnn = np.load(cnn_p, mmap_mode="r")
    gan = np.load(gan_p, mmap_mode="r")
    candb = np.load(candb_p, mmap_mode="r")
    candc = np.load(candc_p, mmap_mode="r")
    candd = np.load(candd_p, mmap_mode="r")
    cande2 = np.load(cande2_p, mmap_mode="r")

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

    candc_idx_p = CANDIDATEC_DIR / "idx.npy"
    if candc_idx_p.exists():
        candc_idx = np.load(candc_idx_p)
    else:
        candc_idx = np.arange(candc.shape[0])
    candc_pos = {int(v): i for i, v in enumerate(candc_idx.tolist())}

    candd_idx_p = CANDIDATED_DIR / "idx.npy"
    if candd_idx_p.exists():
        candd_idx = np.load(candd_idx_p)
    else:
        candd_idx = np.arange(candd.shape[0])
    candd_pos = {int(v): i for i, v in enumerate(candd_idx.tolist())}

    cande2_idx_p = CANDIDATEE2_DIR / "idx.npy"
    if cande2_idx_p.exists():
        cande2_idx = np.load(cande2_idx_p)
    else:
        cande2_idx = np.arange(cande2.shape[0])
    cande2_pos = {int(v): i for i, v in enumerate(cande2_idx.tolist())}

    return gt, cnn, gan, candb, candc, candd, cande2, pos, candb_pos, candc_pos, candd_pos, cande2_pos

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


def make_panel(sid: int, gt, cnn, gan, candb, candc, candd, cande2,
               pos: dict[int, int], candb_pos: dict[int, int], candc_pos: dict[int, int], candd_pos: dict[int, int], cande2_pos: dict[int, int],
               row: dict, obs: dict, crop, out: Path) -> bool:
    if sid not in pos:
        print(f"WARNING: sample {sid} not found in baseline idx.npy; skipping panel.")
        return False
    if sid not in candb_pos:
        print(f"WARNING: sample {sid} not found in Candidate B idx.npy; skipping panel.")
        return False
    if sid not in candc_pos:
        print(f"WARNING: sample {sid} not found in Candidate C idx.npy; skipping panel.")
        return False
    if sid not in candd_pos:
        print(f"WARNING: sample {sid} not found in Candidate D idx.npy; skipping panel.")
        return False
    if sid not in cande2_pos:
        print(f"WARNING: sample {sid} not found in Candidate E2 idx.npy; skipping panel.")
        return False

    i = pos[sid]
    j = candb_pos[sid]
    k = candc_pos[sid]
    m = candd_pos[sid]
    n = cande2_pos[sid]
    gt_s = speed(np.asarray(gt[i]))
    cnn_s = speed(np.asarray(cnn[i]))
    gan_s = speed(np.asarray(gan[i]))
    candb_s = speed(np.asarray(candb[j]))
    candc_s = speed(np.asarray(candc[k]))
    candd_s = speed(np.asarray(candd[m]))
    cande2_s = speed(np.asarray(cande2[n]))

    desc = "full field"
    if crop is not None:
        y0, y1, x0, x1 = crop
        gt_s = gt_s[y0:y1, x0:x1]
        cnn_s = cnn_s[y0:y1, x0:x1]
        candb_s = candb_s[y0:y1, x0:x1]
        candc_s = candc_s[y0:y1, x0:x1]
        candd_s = candd_s[y0:y1, x0:x1]
        cande2_s = cande2_s[y0:y1, x0:x1]
        gan_s = gan_s[y0:y1, x0:x1]
        desc = f"crop y={y0}:{y1}, x={x0}:{x1}"

    err_cnn = np.abs(cnn_s - gt_s)
    err_candb = np.abs(candb_s - gt_s)
    err_candc = np.abs(candc_s - gt_s)
    err_candd = np.abs(candd_s - gt_s)
    err_cande2 = np.abs(cande2_s - gt_s)
    err_gan = np.abs(gan_s - gt_s)
    diff_e2_cc = np.abs(cande2_s - candc_s)
    diff_e2_cd = np.abs(cande2_s - candd_s)

    vmin = float(min(np.nanmin(gt_s), np.nanmin(cnn_s), np.nanmin(candb_s), np.nanmin(candc_s), np.nanmin(candd_s), np.nanmin(cande2_s), np.nanmin(gan_s)))
    vmax = float(max(np.nanmax(gt_s), np.nanmax(cnn_s), np.nanmax(candb_s), np.nanmax(candc_s), np.nanmax(candd_s), np.nanmax(cande2_s), np.nanmax(gan_s)))
    emax = float(max(np.nanmax(err_cnn), np.nanmax(err_candb), np.nanmax(err_candc), np.nanmax(err_candd), np.nanmax(err_cande2), np.nanmax(err_gan)))
    dmax = float(max(np.nanmax(diff_e2_cc), np.nanmax(diff_e2_cd)))
    if not np.isfinite(emax) or emax <= 0:
        emax = 1.0
    if not np.isfinite(dmax) or dmax <= 0:
        dmax = 1.0

    fig, axes = plt.subplots(1, 15, figsize=(74, 5.2))
    fields = [
        gt_s, cnn_s, candb_s, candc_s, candd_s, cande2_s, gan_s,
        err_cnn, err_candb, err_candc, err_candd, err_cande2, err_gan,
        diff_e2_cc, diff_e2_cd,
    ]
    titles = [
        "GT speed", "CNN speed", "Candidate B speed", "Candidate C speed", "Candidate D speed", "Candidate E2 speed", "GAN speed",
        "|CNN-GT|", "|CandB-GT|", "|CandC-GT|", "|CandD-GT|", "|CandE2-GT|", "|GAN-GT|",
        "|CandE2-CandC|", "|CandE2-CandD|",
    ]

    for ax, field, title in zip(axes, fields, titles):
        if title.startswith("|CandE2-Cand"):
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
            d_val = raw.get("candidateD", "")
            e2_val = raw.get("candidateE2", "")
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
            d_val = ""
            e2_val = ""

        body.append(
            f"<tr><td>{H(label)}</td><td>{H(group)}</td>"
            f"<td class='num'>{H(cnn_val)}</td>"
            f"<td class='num'>{H(gan_val)}</td>"
            f"<td class='num'>{H(b_val)}</td>"
            f"<td class='num'>{H(c_val)}</td>"
            f"<td class='num'>{H(d_val)}</td>"
            f"<td class='num'>{H(e2_val)}</td>"
            f"<td>{rank_cell}</td></tr>"
        )

    count_line = (
        f"Top-ranked: Candidate C {top_counts['candidateC']} | "
        f"Candidate D {top_counts['candidateD']} | "
        f"Candidate E2 {top_counts['candidateE2']} | "
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
      <p class="muted">Lower is better for these physics/domain error metrics. Ranking is better → worse using Candidate B/C/D/E2 evaluation CSVs when available.</p>
      <table class="metrics">
        <thead><tr><th>Measure</th><th>Group</th><th>CNN</th><th>GAN</th><th>Candidate B</th><th>Candidate C</th><th>Candidate D</th><th>Candidate E2</th><th>Ranking / Winner</th></tr></thead>
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
        <thead><tr><th>Distance</th><th>CNN</th><th>GAN</th><th>{H(label)}</th><th>Winner before → after</th></tr></thead>
        <tbody>
          <tr><td>PD bottleneck</td><td class="num">{H(pd_vals[0])}</td><td class="num">{H(pd_vals[1])}</td><td class="num">{H(pd_vals[2])}</td><td>{H(pd_before)} → {H(pd_after)}</td></tr>
          <tr><td>MT Wasserstein</td><td class="num">{H(mt_vals[0])}</td><td class="num">{H(mt_vals[1])}</td><td class="num">{H(mt_vals[2])}</td><td>{H(mt_before)} → {H(mt_after)}</td></tr>
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
          {candidate_topology_box(obs, "candidateC", "Candidate C", open_box=True)}
          {candidate_topology_box(obs, "candidateD", "Candidate D", open_box=False)}
          {candidate_topology_box(obs, "candidateE2", "Candidate E2", open_box=False)}

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
  <button onclick="showOnly('tag-mt-gan-flipped-to-candidateB')">MT-GAN → Candidate B ({count('mt_gan_flipped_to_candidateB')})</button>
  <button onclick="showOnly('tag-mt-gan-flipped-to-candidateC')">MT-GAN → Candidate C ({count('mt_gan_flipped_to_candidateC')})</button>
  <button onclick="showOnly('tag-mt-gan-flipped-to-candidateD')">MT-GAN → Candidate D ({count('mt_gan_flipped_to_candidateD')})</button>
  <button onclick="showOnly('tag-mt-gan-flipped-to-candidateE2')">MT-GAN → Candidate E2 ({count('mt_gan_flipped_to_candidateE2')})</button>
  <button onclick="showOnly('tag-candidateB-mt-winner')">Candidate B MT winner ({count('candidateB_mt_winner')})</button>
  <button onclick="showOnly('tag-candidateC-mt-winner')">Candidate C MT winner ({count('candidateC_mt_winner')})</button>
  <button onclick="showOnly('tag-candidateD-mt-winner')">Candidate D MT winner ({count('candidateD_mt_winner')})</button>
  <button onclick="showOnly('tag-candidateE2-mt-winner')">Candidate E2 MT winner ({count('candidateE2_mt_winner')})</button>
  <button onclick="showOnly('tag-candidateB-pd-improves-vs-cnn')">Candidate B improves PD vs CNN ({count('candidateB_pd_improves_vs_cnn')})</button>
  <button onclick="showOnly('tag-candidateC-pd-improves-vs-cnn')">Candidate C improves PD vs CNN ({count('candidateC_pd_improves_vs_cnn')})</button>
  <button onclick="showOnly('tag-candidateD-pd-improves-vs-cnn')">Candidate D improves PD vs CNN ({count('candidateD_pd_improves_vs_cnn')})</button>
  <button onclick="showOnly('tag-candidateE2-pd-improves-vs-cnn')">Candidate E2 improves PD vs CNN ({count('candidateE2_pd_improves_vs_cnn')})</button>
  <button onclick="showOnly('tag-candidateB-mt-improves-vs-cnn')">Candidate B improves MT vs CNN ({count('candidateB_mt_improves_vs_cnn')})</button>
  <button onclick="showOnly('tag-candidateC-mt-improves-vs-cnn')">Candidate C improves MT vs CNN ({count('candidateC_mt_improves_vs_cnn')})</button>
  <button onclick="showOnly('tag-candidateD-mt-improves-vs-cnn')">Candidate D improves MT vs CNN ({count('candidateD_mt_improves_vs_cnn')})</button>
  <button onclick="showOnly('tag-candidateE2-mt-improves-vs-cnn')">Candidate E2 improves MT vs CNN ({count('candidateE2_mt_improves_vs_cnn')})</button>
  <button onclick="showOnly('tag-candidateC-mt-improves-vs-candidateB')">Candidate C improves MT vs B ({count('candidateC_mt_improves_vs_candidateB')})</button>
  <button onclick="showOnly('tag-candidateD-mt-improves-vs-candidateC')">Candidate D improves MT vs C ({count('candidateD_mt_improves_vs_candidateC')})</button>
  <button onclick="showOnly('tag-candidateE2-mt-improves-vs-candidateC')">Candidate E2 improves MT vs C ({count('candidateE2_mt_improves_vs_candidateC')})</button>
  <button onclick="showOnly('tag-topology-consensus-cnn')">PD=MT=CNN ({count('topology_consensus_cnn')})</button>
  <button onclick="showOnly('tag-gan-metric-majority')">GAN metric majority ({count('gan_metric_majority')})</button>
  <button onclick="showOnly('tag-gan-majority-mt-rejects-gan')">GAN majority but MT≠GAN ({count('gan_majority_mt_rejects_gan')})</button>
  <button onclick="showOnly('tag-adjacent-cluster-10-13')">Cluster 10–13</button>
  <button onclick="showOnly('tag-adjacent-cluster-76-78')">Cluster 76–78</button>
  <button onclick="showOnly('tag-adjacent-cluster-90-93')">Cluster 90–93</button>
  <button onclick="showOnly('tag-adjacent-cluster-161-164')">Cluster 161–164</button>
  <p class="muted">Each panel shows GT speed | CNN speed | Candidate B speed | Candidate C speed | Candidate D speed | Candidate E2 speed | GAN speed | absolute-error maps | Candidate-E2-minus-C/D difference maps.</p>
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
    candc_topology = load_candidate_topology_rows(CANDIDATEC_TOPOLOGY_COMPARISON, "candidateC", "Candidate C")
    candd_topology = load_candidate_topology_rows(CANDIDATED_TOPOLOGY_COMPARISON, "candidateD", "Candidate D")
    cande2_topology = load_candidate_topology_rows(CANDIDATEE2_TOPOLOGY_COMPARISON, "candidateE2", "Candidate E2")
    for topology_rows in (candb_topology, candc_topology, candd_topology, cande2_topology):
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

    gt = cnn = gan = candb = candc = candd = cande2 = pos = candb_pos = candc_pos = candd_pos = cande2_pos = None
    if not args.no_panels:
        gt, cnn, gan, candb, candc, candd, cande2, pos, candb_pos, candc_pos, candd_pos, cande2_pos = load_arrays()

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
            crop_ok = make_panel(sid, gt, cnn, gan, candb, candc, candd, cande2, pos, candb_pos, candc_pos, candd_pos, cande2_pos, row, ob, (0, 160, 0, 160), crop_path)
            full_ok = make_panel(sid, gt, cnn, gan, candb, candc, candd, cande2, pos, candb_pos, candc_pos, candd_pos, cande2_pos, row, ob, None, full_path)

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
            "mt_winner_after_candidateD": ob.get("mt_winner_after_candidateD", ""),
            "pd_winner_after_candidateD": ob.get("pd_winner_after_candidateD", ""),
            "pd_distance_candidateD": ob.get("pd_distance_candidateD", ""),
            "mt_distance_candidateD": ob.get("mt_distance_candidateD", ""),
            "mt_winner_after_candidateE2": ob.get("mt_winner_after_candidateE2", ""),
            "pd_winner_after_candidateE2": ob.get("pd_winner_after_candidateE2", ""),
            "pd_distance_candidateE2": ob.get("pd_distance_candidateE2", ""),
            "mt_distance_candidateE2": ob.get("mt_distance_candidateE2", ""),
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
            "mt_winner_after_candidateD", "pd_winner_after_candidateD",
            "pd_distance_candidateD", "mt_distance_candidateD",
            "mt_winner_after_candidateE2", "pd_winner_after_candidateE2",
            "pd_distance_candidateE2", "mt_distance_candidateE2",
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
