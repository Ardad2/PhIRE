#!/usr/bin/env python3
"""
Generate baseline visual-inspection cards for TopoAware SR.

This is a bicubic-aware version of generate_visual_inspection_panels.py.
It keeps the same white card-style HTML UI as the original visual-inspection
index, but each panel compares:

    GT speed | Bicubic speed | CNN speed | GAN speed |
    |Bicubic-GT| | |CNN-GT| | |GAN-GT|

Run from scripts/:

    cd ~/PhIRE/scripts
    PYTHONNOUSERSITE=1 /usr/bin/python3 generate_baseline_visual_panels.py

Optional:

    PYTHONNOUSERSITE=1 /usr/bin/python3 generate_baseline_visual_panels.py --samples 10,11,12,13
    PYTHONNOUSERSITE=1 /usr/bin/python3 generate_baseline_visual_panels.py --all
    PYTHONNOUSERSITE=1 /usr/bin/python3 generate_baseline_visual_panels.py --no-panels

Outputs:

    ttk_runs_fixed/baseline_visual_panels/
        index.html
        baseline_visual_manifest.csv
        panels_crop/sample_010_crop.png
        panels_full/sample_010_full.png
"""

from __future__ import annotations

import argparse
import csv
import html
import sys
from pathlib import Path

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
        if (c / "ttk_runs_fixed").exists() or (c / "data_out_fixed").exists() or (c / "data_out").exists():
            return c
    raise FileNotFoundError("Could not locate repo root containing ttk_runs_fixed/, data_out_fixed/, or data_out/.")


ROOT = repo_root()
OUTDIR = ROOT / "ttk_runs_fixed" / "baseline_visual_panels"
CROP_DIR = OUTDIR / "panels_crop"
FULL_DIR = OUTDIR / "panels_full"

DATA_ROOT = ROOT / "data_out_fixed"
CNN_DIR = DATA_ROOT / "wind_mrhr_cnn"
GAN_DIR = DATA_ROOT / "wind_mrhr_gan"
BIC_DIR = DATA_ROOT / "wind_mrhr_bicubic"

# Optional metadata from the earlier CNN/GAN analysis.
FULL_BREAKDOWN = ROOT / "ttk_runs_fixed" / "report_tables" / "full_physics_domain_breakdown" / "physics_domain_breakdown_all_samples.csv"
WIDE_TABLE = ROOT / "ttk_runs_fixed" / "report_tables" / "metric_sweep_all_samples_wide.csv"
BASELINE_PER_SAMPLE = ROOT / "ttk_runs_fixed" / "baseline_metrics" / "all_methods_per_sample.csv"
OBS_PER_SAMPLE = ROOT / "ttk_runs_fixed" / "observation_groups" / "observation_groups_per_sample.csv"
RECOMMENDED_UNIQUE = ROOT / "ttk_runs_fixed" / "observation_groups" / "recommended_visual_inspection_unique_samples.csv"
OLD_VISUAL_MANIFEST = ROOT / "ttk_runs_fixed" / "visual_inspection" / "visual_inspection_manifest.csv"
OLD_BASELINE_MANIFEST = OUTDIR / "baseline_visual_manifest.csv"


# -----------------------------
# Default qualitative sample set
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
    18: "moderate/lower MT-GAN ridge-rich motif",
    19: "strong MT-GAN anchor",
    20: "moderate/lower MT-GAN ridge-rich motif",

    25: "lower-confidence MT-GAN limitation case",
    80: "lower-confidence MT-GAN limitation case",
    154: "lower-confidence MT-GAN limitation case",
}


PHYSICS_METRICS = [
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
# CSV / formatting helpers
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
    if s in {"CNN", "GAN", "BICUBIC", "BIC", "TIE"}:
        return "BICUBIC" if s == "BIC" else s
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


def pick(row: dict, obs: dict, *keys: str) -> str:
    for k in keys:
        if k in row and str(row[k]).strip():
            return str(row[k]).strip()
        if k in obs and str(obs[k]).strip():
            return str(obs[k]).strip()
    return ""


def get_first(row: dict, *keys: str) -> str:
    for k in keys:
        if k in row and str(row[k]).strip():
            return str(row[k]).strip()
    return ""


# -----------------------------
# Metadata loading
# -----------------------------

def load_metric_rows() -> dict[int, dict[str, str]]:
    """Load the earlier CNN/GAN report table if available."""
    for p in (FULL_BREAKDOWN, WIDE_TABLE):
        rows = read_csv(p)
        if rows:
            out = {}
            for r in rows:
                try:
                    out[sid_from(r)] = r
                except Exception:
                    pass
            print(f"Loaded {len(out)} CNN/GAN metric rows from {p}")
            return out
    print("WARNING: no CNN/GAN metric table found; index will have limited metric info.")
    return {}


def load_baseline_metric_rows() -> dict[int, dict[str, str]]:
    """Load optional all-method metric rows if available."""
    rows = read_csv(BASELINE_PER_SAMPLE)
    out: dict[int, dict[str, str]] = {}
    for r in rows:
        try:
            sid = sid_from(r)
        except Exception:
            continue
        out[sid] = r
    if out:
        print(f"Loaded {len(out)} all-baseline metric rows from {BASELINE_PER_SAMPLE}")
    return out


def load_obs_rows() -> dict[int, dict[str, str]]:
    out: dict[int, dict[str, str]] = {}
    for p in (OBS_PER_SAMPLE, RECOMMENDED_UNIQUE, OLD_VISUAL_MANIFEST, OLD_BASELINE_MANIFEST):
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

    return sorted(groups)


def question(sid: int, row: dict, obs: dict) -> str:
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
    return "Control sample: compare bicubic smoothness, CNN direct fidelity, GAN texture, and topology choices."


def select_samples(args, metrics: dict[int, dict], obs: dict[int, dict], n_available: int) -> list[int]:
    if args.all:
        return list(range(n_available))
    if args.samples:
        raw = args.samples.replace(";", ",").replace(" ", ",")
        return sorted({int(x.strip()) for x in raw.split(",") if x.strip()})

    selected = set(FORCED.keys())

    # Preserve samples from previous visual-inspection workflows when available.
    for sid in obs:
        if OLD_VISUAL_MANIFEST.exists() or OLD_BASELINE_MANIFEST.exists():
            selected.add(sid)
        elif boolish(obs[sid].get("group_recommended_visual_inspection_unique", "")):
            selected.add(sid)
        elif obs[sid].get("recommendation_group", ""):
            selected.add(sid)

    return sorted(s for s in selected if 0 <= s < n_available)


# -----------------------------
# Array loading and panels
# -----------------------------

def require_files(directory: Path, names: list[str]) -> None:
    missing = [str(directory / name) for name in names if not (directory / name).exists()]
    if missing:
        raise FileNotFoundError("Missing required NPY files:\n" + "\n".join(missing))


def load_arrays():
    for d in (CNN_DIR, GAN_DIR, BIC_DIR):
        require_files(d, ["idx.npy", "dataGT.npy", "dataSR.npy"])

    cnn_idx = np.load(CNN_DIR / "idx.npy")
    gan_idx = np.load(GAN_DIR / "idx.npy")
    bic_idx = np.load(BIC_DIR / "idx.npy")

    if not np.array_equal(cnn_idx, gan_idx):
        raise ValueError("CNN and GAN idx arrays differ; cannot align samples.")
    if not np.array_equal(cnn_idx, bic_idx):
        raise ValueError("CNN and bicubic idx arrays differ; cannot align samples.")

    gt = np.load(CNN_DIR / "dataGT.npy", mmap_mode="r")
    gan_gt = np.load(GAN_DIR / "dataGT.npy", mmap_mode="r")
    bic_gt = np.load(BIC_DIR / "dataGT.npy", mmap_mode="r")

    max_gt_gan = float(np.max(np.abs(gt[:] - gan_gt[:])))
    max_gt_bic = float(np.max(np.abs(gt[:] - bic_gt[:])))
    if max_gt_gan != 0.0:
        print(f"WARNING: CNN GT and GAN GT differ, max abs diff={max_gt_gan:.6e}")
    if max_gt_bic != 0.0:
        print(f"WARNING: CNN GT and bicubic GT differ, max abs diff={max_gt_bic:.6e}")

    bic = np.load(BIC_DIR / "dataSR.npy", mmap_mode="r")
    cnn = np.load(CNN_DIR / "dataSR.npy", mmap_mode="r")
    gan = np.load(GAN_DIR / "dataSR.npy", mmap_mode="r")

    pos = {int(v): i for i, v in enumerate(cnn_idx.tolist())}

    print("Array alignment verified:")
    print(f"  idx shape/range: {cnn_idx.shape}, {int(cnn_idx.min())} … {int(cnn_idx.max())}")
    print(f"  GT shape:        {gt.shape}")
    print(f"  Bicubic SR:      {bic.shape}")
    print(f"  CNN SR:          {cnn.shape}")
    print(f"  GAN SR:          {gan.shape}")
    print(f"  GT max diff vs GAN={max_gt_gan:.2e}, vs bicubic={max_gt_bic:.2e}")

    return gt, bic, cnn, gan, pos


def speed(a: np.ndarray) -> np.ndarray:
    if a.ndim == 3 and a.shape[-1] == 2:
        return np.sqrt(a[..., 0] ** 2 + a[..., 1] ** 2)
    if a.ndim == 3 and a.shape[-1] == 1:
        return a[..., 0]
    if a.ndim == 2:
        return a
    raise ValueError(f"Unexpected sample shape: {a.shape}")


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)))


def panel_title(sid: int, row: dict, obs: dict) -> str:
    return (
        f"sample {sid} | "
        f"SSIM={norm(pick(row, obs, 'ssim_winner')) or '?'} | "
        f"PD={norm(pick(row, obs, 'pd_winner')) or '?'} | "
        f"MT={norm(pick(row, obs, 'mt_winner')) or '?'} | "
        f"direct={norm(pick(row, obs, 'direct_error_group_winner')) or '?'} | "
        f"dist={norm(pick(row, obs, 'distributional_group_winner')) or row.get('overall_metric_majority', '?')}"
    )


def compute_baseline_stats(gt_s: np.ndarray, bic_s: np.ndarray, cnn_s: np.ndarray, gan_s: np.ndarray) -> dict:
    stats = {}
    for name, arr in (("bicubic", bic_s), ("cnn", cnn_s), ("gan", gan_s)):
        stats[f"{name}_speed_mae"] = mae(arr, gt_s)
        stats[f"{name}_speed_rmse"] = rmse(arr, gt_s)
    return stats


def make_panel(
    sid: int,
    gt,
    bic,
    cnn,
    gan,
    pos: dict[int, int],
    row: dict,
    obs: dict,
    crop,
    out: Path,
) -> tuple[bool, dict]:
    if sid not in pos:
        print(f"WARNING: sample {sid} not found in idx.npy; skipping panel.")
        return False, {}

    i = pos[sid]
    gt_s = speed(np.asarray(gt[i]))
    bic_s = speed(np.asarray(bic[i]))
    cnn_s = speed(np.asarray(cnn[i]))
    gan_s = speed(np.asarray(gan[i]))

    desc = "full field"
    if crop is not None:
        y0, y1, x0, x1 = crop
        gt_s = gt_s[y0:y1, x0:x1]
        bic_s = bic_s[y0:y1, x0:x1]
        cnn_s = cnn_s[y0:y1, x0:x1]
        gan_s = gan_s[y0:y1, x0:x1]
        desc = f"crop y={y0}:{y1}, x={x0}:{x1}"

    err_bic = np.abs(bic_s - gt_s)
    err_cnn = np.abs(cnn_s - gt_s)
    err_gan = np.abs(gan_s - gt_s)

    vmin = float(min(np.nanmin(gt_s), np.nanmin(bic_s), np.nanmin(cnn_s), np.nanmin(gan_s)))
    vmax = float(max(np.nanmax(gt_s), np.nanmax(bic_s), np.nanmax(cnn_s), np.nanmax(gan_s)))
    emax = float(max(np.nanmax(err_bic), np.nanmax(err_cnn), np.nanmax(err_gan)))
    if not np.isfinite(emax) or emax <= 0:
        emax = 1.0

    stats = compute_baseline_stats(gt_s, bic_s, cnn_s, gan_s)

    # Keep the original visual style: white figure, viridis/default colormap,
    # lower origin, visible colorbars, and compact full-width row.
    fig, axes = plt.subplots(1, 7, figsize=(30, 5.2))
    fields = [gt_s, bic_s, cnn_s, gan_s, err_bic, err_cnn, err_gan]
    titles = [
        "GT speed",
        "Bicubic speed",
        "CNN speed",
        "GAN speed",
        "|Bicubic-GT|",
        "|CNN-GT|",
        "|GAN-GT|",
    ]

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
        f"{desc}; speed RMSE bicubic={stats['bicubic_speed_rmse']:.3f}, "
        f"CNN={stats['cnn_speed_rmse']:.3f}, GAN={stats['gan_speed_rmse']:.3f}",
        ha="center",
        fontsize=9,
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    fig.savefig(out, dpi=125)
    plt.close(fig)
    return True, stats


# -----------------------------
# HTML
# -----------------------------

def badge(label: str) -> str:
    label = norm(label)
    if label == "CNN":
        return '<span class="win cnn">CNN</span>'
    if label == "GAN":
        return '<span class="win gan">GAN</span>'
    if label == "BICUBIC":
        return '<span class="win bicubic">Bicubic</span>'
    if label == "TIE":
        return '<span class="win tie">TIE</span>'
    return H(label or "?")


def baseline_summary_table(stats: dict) -> str:
    if not stats:
        return ""

    methods = ["bicubic", "cnn", "gan"]
    labels = {"bicubic": "Bicubic", "cnn": "CNN", "gan": "GAN"}
    rows = [
        ("Speed MAE", "speed_mae"),
        ("Speed RMSE", "speed_rmse"),
    ]

    body = []
    for label, key in rows:
        vals = {m: float(stats.get(f"{m}_{key}", np.nan)) for m in methods}
        finite = {m: v for m, v in vals.items() if np.isfinite(v)}
        winner = min(finite, key=finite.get) if finite else ""
        body.append(
            "<tr>"
            f"<td>{H(label)}</td>"
            + "".join(f"<td class='num'>{num(vals[m])}</td>" for m in methods)
            + f"<td>{badge(winner)}</td>"
            "</tr>"
        )

    return f"""
    <details class="metric-box" open>
      <summary><b>Baseline direct-error summary</b>
        <span class="count">Lower is better; computed from the displayed scalar-speed field.</span>
      </summary>
      <table class="metrics">
        <thead><tr><th>Measure</th><th>Bicubic</th><th>CNN</th><th>GAN</th><th>Winner</th></tr></thead>
        <tbody>{''.join(body)}</tbody>
      </table>
    </details>
    """


def physics_metric_table(row: dict) -> str:
    """Preserve the original CNN/GAN physics-domain breakdown when available."""
    if not row:
        return '<details class="metric-box"><summary><b>CNN/GAN physics-domain breakdown</b> <span class="warn">unavailable</span></summary></details>'

    body = []
    cnn_count = gan_count = tie_count = 0

    for key, label, group in PHYSICS_METRICS:
        w = norm(row.get(f"{key}_winner", ""))
        if w == "CNN":
            cnn_count += 1
        elif w == "GAN":
            gan_count += 1
        elif w == "TIE":
            tie_count += 1

        body.append(
            f"<tr><td>{H(label)}</td><td>{H(group)}</td>"
            f"<td class='num'>{H(num(row.get(key + '_cnn', '')))}</td>"
            f"<td class='num'>{H(num(row.get(key + '_gan', '')))}</td>"
            f"<td>{badge(w)}</td></tr>"
        )

    return f"""
    <details class="metric-box">
      <summary><b>CNN/GAN physics-domain breakdown</b>
        <span class="count">CNN {cnn_count} | GAN {gan_count} | ties {tie_count}</span>
      </summary>
      <p class="muted">This is the original CNN/GAN physics table. The bicubic panel above adds the classical interpolation baseline visually.</p>
      <table class="metrics">
        <thead><tr><th>Measure</th><th>Group</th><th>CNN</th><th>GAN</th><th>Winner</th></tr></thead>
        <tbody>{''.join(body)}</tbody>
      </table>
    </details>
    """


def baseline_metric_table(row: dict) -> str:
    """Optional all-method metric table if all_methods_per_sample.csv has recognizable columns."""
    if not row:
        return ""

    metric_candidates = [
        ("psnr_uv", "PSNR_uv", ["bicubic", "cnn", "gan"], "higher"),
        ("ssim_speed", "SSIM speed", ["bicubic", "cnn", "gan"], "higher"),
        ("speed_mae", "Speed MAE", ["bicubic", "cnn", "gan"], "lower"),
        ("speed_rmse", "Speed RMSE", ["bicubic", "cnn", "gan"], "lower"),
    ]

    body = []
    for base, label, methods, direction in metric_candidates:
        vals = {}
        for m in methods:
            candidates = [
                f"{m}_{base}", f"{base}_{m}", f"{m}.{base}",
                f"{m.upper()}_{base}", f"{base}_{m.upper()}",
            ]
            raw = get_first(row, *candidates)
            try:
                vals[m] = float(raw)
            except Exception:
                vals[m] = np.nan
        if not any(np.isfinite(v) for v in vals.values()):
            continue
        finite = {m: v for m, v in vals.items() if np.isfinite(v)}
        winner = (max(finite, key=finite.get) if direction == "higher" else min(finite, key=finite.get)) if finite else ""
        body.append(
            "<tr>"
            f"<td>{H(label)}</td><td>{H(direction)}</td>"
            + "".join(f"<td class='num'>{num(vals[m])}</td>" for m in methods)
            + f"<td>{badge(winner)}</td>"
            "</tr>"
        )

    if not body:
        return ""

    return f"""
    <details class="metric-box">
      <summary><b>All-baseline metric table</b>
        <span class="count">from baseline_metrics/all_methods_per_sample.csv</span>
      </summary>
      <table class="metrics">
        <thead><tr><th>Measure</th><th>Better</th><th>Bicubic</th><th>CNN</th><th>GAN</th><th>Winner</th></tr></thead>
        <tbody>{''.join(body)}</tbody>
      </table>
    </details>
    """


def card(entry: dict) -> str:
    sid = entry["sample_idx"]
    row = entry["row"]
    obs = entry["obs"]
    base_row = entry.get("baseline_row", {})
    groups = entry["groups"]
    crop = entry["crop_panel"]
    full = entry["full_panel"]
    stats = entry.get("stats", {})

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

          {baseline_summary_table(stats)}
          {baseline_metric_table(base_row)}
          {physics_metric_table(row)}

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
<title>TopoAware SR baseline visual inspection index</title>
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
.cnn {{ background:#dff7e9; color:#11733b; border:1px solid #a8e6c1; }}
.gan {{ background:#fff0d9; color:#a35b00; border:1px solid #ffd39a; }}
.bicubic {{ background:#e9e4ff; color:#4930a3; border:1px solid #c8bcff; }}
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
  <button onclick="showOnly('tag-gan-metric-majority')">GAN metric majority ({count('gan_metric_majority')})</button>
  <button onclick="showOnly('tag-gan-majority-mt-rejects-gan')">GAN majority but MT≠GAN ({count('gan_majority_mt_rejects_gan')})</button>
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
    (OUTDIR / "index.html").write_text(page, encoding="utf-8")


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="Generate index/panels for all samples in idx.npy.")
    parser.add_argument("--samples", default="", help="Comma- or space-separated sample ids to generate instead of default selection.")
    parser.add_argument("--no-panels", action="store_true", help="Only rebuild index/manifest using existing PNGs.")
    args = parser.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    CROP_DIR.mkdir(parents=True, exist_ok=True)
    FULL_DIR.mkdir(parents=True, exist_ok=True)

    metrics = load_metric_rows()
    baseline_metrics = load_baseline_metric_rows()
    obs = load_obs_rows()

    gt = bic = cnn = gan = pos = None
    gt, bic, cnn, gan, pos = load_arrays()
    samples = select_samples(args, metrics, obs, n_available=len(pos))

    if not samples:
        raise RuntimeError("No samples selected.")

    print(f"repo_root={ROOT}")
    print(f"outdir={OUTDIR}")
    print(f"selected_samples={len(samples)}")
    print("forced/extra samples:", " ".join(map(str, sorted(FORCED))))

    entries = []
    manifest = []

    for sid in samples:
        row = metrics.get(sid, {})
        base_row = baseline_metrics.get(sid, {})
        ob = obs.get(sid, {})

        crop_path = CROP_DIR / f"sample_{sid:03d}_crop.png"
        full_path = FULL_DIR / f"sample_{sid:03d}_full.png"

        crop_ok = crop_path.exists()
        full_ok = full_path.exists()
        crop_stats = {}
        full_stats = {}

        if not args.no_panels:
            crop_ok, crop_stats = make_panel(sid, gt, bic, cnn, gan, pos, row, ob, (0, 160, 0, 160), crop_path)
            full_ok, full_stats = make_panel(sid, gt, bic, cnn, gan, pos, row, ob, None, full_path)
        else:
            # Compute full-field stats for the card even when reusing existing panels.
            if sid in pos:
                i = pos[sid]
                full_stats = compute_baseline_stats(
                    speed(np.asarray(gt[i])),
                    speed(np.asarray(bic[i])),
                    speed(np.asarray(cnn[i])),
                    speed(np.asarray(gan[i])),
                )

        crop_rel = crop_path.relative_to(OUTDIR).as_posix() if crop_ok else ""
        full_rel = full_path.relative_to(OUTDIR).as_posix() if full_ok else ""

        groups = infer_groups(sid, row, ob)
        q = question(sid, row, ob)

        # Use crop stats in the card because the thumbnail is the crop panel.
        stats = crop_stats or full_stats

        entry = {
            "sample_idx": sid,
            "row": row,
            "baseline_row": base_row,
            "obs": ob,
            "groups": groups,
            "question": q,
            "crop_panel": crop_rel,
            "full_panel": full_rel,
            "stats": stats,
        }
        entries.append(entry)

        manifest_row = {
            "sample_idx": sid,
            "psnr_winner": norm(pick(row, ob, "psnr_winner")),
            "ssim_winner": norm(pick(row, ob, "ssim_winner")),
            "pd_winner": norm(pick(row, ob, "pd_winner")),
            "mt_winner": norm(pick(row, ob, "mt_winner")),
            "direct_error_group_winner": norm(pick(row, ob, "direct_error_group_winner")),
            "distributional_group_winner": norm(pick(row, ob, "distributional_group_winner")) or row.get("overall_metric_majority", ""),
            "tail_group_winner": norm(pick(row, ob, "tail_group_winner")),
            "configured_physics_group_winner": norm(pick(row, ob, "configured_physics_group_winner")),
            "question": q,
            "groups": ";".join(groups),
            "crop_panel": crop_rel,
            "full_panel": full_rel,
            "forced_reason": FORCED.get(sid, ""),
        }
        for k, v in stats.items():
            manifest_row[k] = v
        manifest.append(manifest_row)

    entries.sort(key=lambda e: e["sample_idx"])
    manifest.sort(key=lambda r: int(r["sample_idx"]))

    fields = [
        "sample_idx", "psnr_winner", "ssim_winner", "pd_winner", "mt_winner",
        "direct_error_group_winner", "distributional_group_winner",
        "tail_group_winner", "configured_physics_group_winner",
        "bicubic_speed_mae", "bicubic_speed_rmse",
        "cnn_speed_mae", "cnn_speed_rmse",
        "gan_speed_mae", "gan_speed_rmse",
        "question", "groups", "crop_panel", "full_panel", "forced_reason",
    ]
    write_csv(OUTDIR / "baseline_visual_manifest.csv", manifest, fields)
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
