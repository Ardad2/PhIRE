#!/usr/bin/env python3
"""
Visual-inspection generator for TopoAware SR.

Run from the scripts directory:

    cd ~/PhIRE/scripts
    PYTHONNOUSERSITE=1 /usr/bin/python3 generate_visual_inspection_panels.py

Recommended full report:

    PYTHONNOUSERSITE=1 /usr/bin/python3 generate_visual_inspection_panels.py --all-samples

Useful options:

    PYTHONNOUSERSITE=1 /usr/bin/python3 generate_visual_inspection_panels.py --samples 90,91,92,93
    PYTHONNOUSERSITE=1 /usr/bin/python3 generate_visual_inspection_panels.py --all
    PYTHONNOUSERSITE=1 /usr/bin/python3 generate_visual_inspection_panels.py --all-samples
    PYTHONNOUSERSITE=1 /usr/bin/python3 generate_visual_inspection_panels.py --no-panels

What this script does:
  - Uses the fixed/repaired arrays:
        data_out_fixed/wind_mrhr_cnn/
        data_out_fixed/wind_mrhr_gan/
  - Builds crop and full-field panels.
  - Builds ttk_runs_fixed/visual_inspection/index.html.
  - Adds per-sample physics/domain metric breakdowns when available.
  - Supports generating all 168 samples through --all-samples / --all.
  - Preserves forced qualitative/adjacent-control sample labels.
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


# ============================================================
# Repo paths
# ============================================================

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
        if (
            (c / "ttk_runs_fixed").exists()
            or (c / "data_out_fixed").exists()
            or (c / "data_out").exists()
        ):
            return c

    raise FileNotFoundError(
        "Could not locate repo root containing ttk_runs_fixed/, data_out_fixed/, or data_out/."
    )


ROOT = repo_root()

OUTDIR = ROOT / "ttk_runs_fixed" / "visual_inspection"
CROP_DIR = OUTDIR / "panels_crop"
FULL_DIR = OUTDIR / "panels_full"

CNN_DIR = ROOT / "data_out_fixed" / "wind_mrhr_cnn"
GAN_DIR = ROOT / "data_out_fixed" / "wind_mrhr_gan"

MERGED_CSV = ROOT / "ttk_runs_fixed" / "combined" / "psnr_topology_physics_merged.csv"
FULL_BREAKDOWN = (
    ROOT
    / "ttk_runs_fixed"
    / "report_tables"
    / "full_physics_domain_breakdown"
    / "physics_domain_breakdown_all_samples.csv"
)
WIDE_TABLE = ROOT / "ttk_runs_fixed" / "report_tables" / "metric_sweep_all_samples_wide.csv"

OBS_PER_SAMPLE = ROOT / "ttk_runs_fixed" / "observation_groups" / "observation_groups_per_sample.csv"
RECOMMENDED_UNIQUE = ROOT / "ttk_runs_fixed" / "observation_groups" / "recommended_visual_inspection_unique_samples.csv"
OLD_MANIFEST = OUTDIR / "visual_inspection_manifest.csv"


# ============================================================
# Forced qualitative / adjacent-control sample set
# ============================================================

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


# ============================================================
# Generic helpers
# ============================================================

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
    raise ValueError(f"row has no sample id: {row}")


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


def pick(row: dict, obs: dict, *keys: str) -> str:
    for k in keys:
        if k in row and str(row[k]).strip():
            return str(row[k]).strip()
        if k in obs and str(obs[k]).strip():
            return str(obs[k]).strip()
    return ""


def merge_rows_by_sample(paths: list[Path]) -> dict[int, dict[str, str]]:
    """
    Merge multiple CSV files keyed by sample id.

    Later files can add missing columns, but non-empty values from any file
    are preserved. This lets us combine:
      - merged all-sample CSV,
      - wide metric sweep table,
      - detailed physics/domain table.
    """
    out: dict[int, dict[str, str]] = {}

    for p in paths:
        rows = read_csv(p)
        if not rows:
            continue

        loaded = 0
        for r in rows:
            try:
                sid = sid_from(r)
            except Exception:
                continue

            out.setdefault(sid, {})
            out[sid]["sample_idx"] = str(sid)

            for k, v in r.items():
                if v is None:
                    continue
                sv = str(v).strip()
                if sv != "":
                    out[sid][k] = sv

            loaded += 1

        print(f"Loaded {loaded} rows from {p}")

    return out


# ============================================================
# Metadata loading
# ============================================================

def load_metric_rows(extra_merged_csv: Path | None = None) -> dict[int, dict[str, str]]:
    paths = []

    if extra_merged_csv is not None:
        paths.append(extra_merged_csv)

    paths.extend([
        MERGED_CSV,
        WIDE_TABLE,
        FULL_BREAKDOWN,
    ])

    # Remove duplicate paths while preserving order.
    seen = set()
    unique_paths = []
    for p in paths:
        rp = p.resolve()
        if rp not in seen:
            unique_paths.append(p)
            seen.add(rp)

    rows = merge_rows_by_sample(unique_paths)

    if not rows:
        print("WARNING: no metric rows found; index will have limited metric info.")

    return rows


def load_obs_rows() -> dict[int, dict[str, str]]:
    out: dict[int, dict[str, str]] = {}

    for p in (OBS_PER_SAMPLE, RECOMMENDED_UNIQUE, OLD_MANIFEST):
        rows = read_csv(p)
        if not rows:
            continue

        loaded = 0
        for r in rows:
            try:
                sid = sid_from(r)
            except Exception:
                continue

            out.setdefault(sid, {})
            out[sid]["sample_idx"] = str(sid)

            for k, v in r.items():
                if v is not None and str(v).strip():
                    out[sid][k] = str(v).strip()

            loaded += 1

        print(f"Loaded {loaded} observation rows from {p}")

    return out


# ============================================================
# Metric breakdown helpers
# ============================================================

def metric_value(row: dict, key: str, model: str) -> str:
    """
    Robustly retrieve a metric value.

    Supports common layouts:
      key_cnn / key_gan
      cnn_key / gan_key
      key_CNN / key_GAN
      CNN_key / GAN_key
    """
    model_l = model.lower()
    model_u = model.upper()

    candidates = [
        f"{key}_{model_l}",
        f"{model_l}_{key}",
        f"{key}_{model_u}",
        f"{model_u}_{key}",
    ]

    for c in candidates:
        if c in row and str(row[c]).strip():
            return str(row[c]).strip()

    return ""


def metric_winner(row: dict, key: str) -> str:
    """
    Retrieve or infer the winner for a metric.

    Lower is better for all metrics in METRICS because signed quantities
    are represented as absolute errors / absolute deltas.
    """
    candidates = [
        f"{key}_winner",
        f"winner_{key}",
        f"{key}_win",
    ]

    for c in candidates:
        w = norm(row.get(c, ""))
        if w in {"CNN", "GAN", "TIE"}:
            return w

    cnn = metric_value(row, key, "cnn")
    gan = metric_value(row, key, "gan")

    try:
        c = float(cnn)
        g = float(gan)
    except Exception:
        return ""

    if np.isclose(c, g):
        return "TIE"
    return "CNN" if c < g else "GAN"


def metric_counts(row: dict) -> tuple[int, int, int]:
    cnn_count = gan_count = tie_count = 0

    for key, _, _ in METRICS:
        w = metric_winner(row, key)
        if w == "CNN":
            cnn_count += 1
        elif w == "GAN":
            gan_count += 1
        elif w == "TIE":
            tie_count += 1

    return cnn_count, gan_count, tie_count


# ============================================================
# Groups / questions / sample selection
# ============================================================

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

    psnr = norm(pick(row, obs, "psnr_winner", "winner_psnr"))
    ssim = norm(pick(row, obs, "ssim_winner", "winner_ssim"))
    pd = norm(pick(row, obs, "pd_winner", "winner_pd", "bottleneck_pd_winner"))
    mt = norm(pick(row, obs, "mt_winner", "winner_mt", "merge_tree_winner"))
    direct = norm(pick(row, obs, "direct_error_group_winner", "winner_direct_error_group"))
    dist = norm(pick(row, obs, "distributional_group_winner", "winner_distributional_group"))
    tail = norm(pick(row, obs, "tail_group_winner", "winner_tail_group"))
    physics = norm(pick(row, obs, "configured_physics_group_winner", "physics_group_winner", "winner_physics_group"))

    cnn_count, gan_count, tie_count = metric_counts(row)
    gan_majority = (
        boolish(row.get("gan_metric_majority", ""))
        or norm(row.get("overall_metric_majority", "")) == "GAN"
        or gan_count > cnn_count
    )

    if psnr == "CNN":
        groups.add("psnr_cnn")
    elif psnr == "GAN":
        groups.add("psnr_gan")

    if ssim == "CNN":
        groups.add("ssim_cnn")
    elif ssim == "GAN":
        groups.add("ssim_gan")

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
        groups.add("topology_consensus_gan_all")

    if pd == "GAN" and mt == "CNN":
        groups.add("pd_gan_mt_cnn_control")
        groups.add("pd_gan_mt_cnn_all")
        groups.add("candidate_structural_hallucination_signature")

    if pd == "CNN" and mt == "CNN":
        groups.add("topology_consensus_cnn")
        groups.add("topology_consensus_cnn_all")

    if gan_majority:
        groups.add("gan_metric_majority")

    if gan_majority and mt != "GAN":
        groups.add("gan_majority_mt_rejects_gan")

    if direct == "CNN":
        groups.add("direct_cnn")
    elif direct == "GAN":
        groups.add("direct_gan")

    if dist == "CNN":
        groups.add("distributional_cnn")
    elif dist == "GAN":
        groups.add("distributional_gan")

    if tail == "CNN":
        groups.add("tail_cnn")
    elif tail == "GAN":
        groups.add("tail_gan")

    if physics == "CNN":
        groups.add("physics_cnn")
    elif physics == "GAN":
        groups.add("physics_gan")

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

    return sorted(g for g in groups if g)


def question(sid: int, row: dict, obs: dict) -> str:
    pd = norm(pick(row, obs, "pd_winner", "winner_pd", "bottleneck_pd_winner"))
    mt = norm(pick(row, obs, "mt_winner", "winner_mt", "merge_tree_winner"))

    cnn_count, gan_count, _ = metric_counts(row)
    gan_majority = (
        boolish(row.get("gan_metric_majority", ""))
        or norm(row.get("overall_metric_majority", "")) == "GAN"
        or gan_count > cnn_count
    )

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
    if args.all or args.all_samples:
        selected = set(metrics.keys()) | set(obs.keys())

        # If metrics are missing but arrays exist, fall back to idx.npy later.
        if selected:
            return sorted(selected)

        return []

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


# ============================================================
# Array loading / panels
# ============================================================

def load_arrays():
    gt_p = CNN_DIR / "dataGT.npy"
    cnn_p = CNN_DIR / "dataSR.npy"
    gan_p = GAN_DIR / "dataSR.npy"

    missing = [str(p) for p in (gt_p, cnn_p, gan_p) if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing NPY arrays:\n" + "\n".join(missing))

    gt = np.load(gt_p, mmap_mode="r")
    cnn = np.load(cnn_p, mmap_mode="r")
    gan = np.load(gan_p, mmap_mode="r")

    idx_p = CNN_DIR / "idx.npy"
    if idx_p.exists():
        idx = np.load(idx_p)
    else:
        idx = np.arange(gt.shape[0])

    pos = {int(v): i for i, v in enumerate(idx.tolist())}

    return gt, cnn, gan, pos


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
        f"SSIM={norm(pick(row, obs, 'ssim_winner', 'winner_ssim')) or '?'} | "
        f"PD={norm(pick(row, obs, 'pd_winner', 'winner_pd', 'bottleneck_pd_winner')) or '?'} | "
        f"MT={norm(pick(row, obs, 'mt_winner', 'winner_mt', 'merge_tree_winner')) or '?'} | "
        f"direct={norm(pick(row, obs, 'direct_error_group_winner', 'winner_direct_error_group')) or '?'} | "
        f"dist={norm(pick(row, obs, 'distributional_group_winner', 'winner_distributional_group')) or row.get('overall_metric_majority', '?')}"
    )


def make_panel(
    sid: int,
    gt,
    cnn,
    gan,
    pos: dict[int, int],
    row: dict,
    obs: dict,
    crop,
    out: Path,
) -> bool:
    if sid not in pos:
        print(f"WARNING: sample {sid} not found in idx.npy; skipping panel.")
        return False

    i = pos[sid]

    gt_s = speed(np.asarray(gt[i]))
    cnn_s = speed(np.asarray(cnn[i]))
    gan_s = speed(np.asarray(gan[i]))

    desc = "full field"
    if crop is not None:
        y0, y1, x0, x1 = crop
        gt_s = gt_s[y0:y1, x0:x1]
        cnn_s = cnn_s[y0:y1, x0:x1]
        gan_s = gan_s[y0:y1, x0:x1]
        desc = f"crop y={y0}:{y1}, x={x0}:{x1}"

    err_cnn = np.abs(cnn_s - gt_s)
    err_gan = np.abs(gan_s - gt_s)

    # Speed panels share one scale across GT/CNN/GAN for fair visual comparison.
    vmin = float(min(np.nanmin(gt_s), np.nanmin(cnn_s), np.nanmin(gan_s)))
    vmax = float(max(np.nanmax(gt_s), np.nanmax(cnn_s), np.nanmax(gan_s)))

    # Error panels share one scale across CNN/GAN errors.
    emax = float(max(np.nanmax(err_cnn), np.nanmax(err_gan)))
    if not np.isfinite(emax) or emax <= 0:
        emax = 1.0

    fig, axes = plt.subplots(1, 5, figsize=(24, 5.2))

    fields = [gt_s, cnn_s, gan_s, err_cnn, err_gan]
    titles = ["GT speed", "CNN speed", "GAN speed", "|CNN-GT|", "|GAN-GT|"]

    for j, (ax, field, title) in enumerate(zip(axes, fields, titles)):
        if j >= 3:
            im = ax.imshow(field, origin="lower", vmin=0, vmax=emax)
        else:
            im = ax.imshow(field, origin="lower", vmin=vmin, vmax=vmax)

        ax.set_title(title, fontsize=11)
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


# ============================================================
# HTML generation
# ============================================================

def metric_table(row: dict) -> str:
    if not row:
        return """
        <details class="metric-box">
          <summary><b>Physics/domain metric breakdown</b> <span class="warn">unavailable</span></summary>
          <p class="muted">No metric row was available for this sample.</p>
        </details>
        """

    body = []
    cnn_count = gan_count = tie_count = 0

    for key, label, group in METRICS:
        w = metric_winner(row, key)

        if w == "CNN":
            cnn_count += 1
            badge = '<span class="win cnn">CNN</span>'
        elif w == "GAN":
            gan_count += 1
            badge = '<span class="win gan">GAN</span>'
        elif w == "TIE":
            tie_count += 1
            badge = '<span class="win tie">TIE</span>'
        else:
            badge = '<span class="muted">?</span>'

        cnn_v = metric_value(row, key, "cnn")
        gan_v = metric_value(row, key, "gan")

        body.append(
            f"<tr><td>{H(label)}</td><td>{H(group)}</td>"
            f"<td class='num'>{H(num(cnn_v))}</td>"
            f"<td class='num'>{H(num(gan_v))}</td>"
            f"<td>{badge}</td></tr>"
        )

    return f"""
    <details class="metric-box">
      <summary><b>Physics/domain metric breakdown</b>
        <span class="count">CNN {cnn_count} | GAN {gan_count} | ties {tie_count}</span>
      </summary>
      <p class="muted">
        Lower is better. For signed quantities such as WPD bias, PSD slope delta,
        gradient-kurtosis delta, and exceedance deltas, the displayed comparison
        uses absolute error relative to GT.
      </p>
      <table class="metrics">
        <thead>
          <tr><th>Measure</th><th>Group</th><th>CNN</th><th>GAN</th><th>Winner</th></tr>
        </thead>
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

    cls = " ".join("tag-" + g.replace("_", "-") for g in groups)
    chips = " ".join(f"<span class='chip'>{H(g.replace('_', ' '))}</span>" for g in groups)

    winners = (
        f"PSNR: {H(norm(pick(row, obs, 'psnr_winner', 'winner_psnr')) or '?')} | "
        f"SSIM: {H(norm(pick(row, obs, 'ssim_winner', 'winner_ssim')) or '?')} | "
        f"PD: {H(norm(pick(row, obs, 'pd_winner', 'winner_pd', 'bottleneck_pd_winner')) or '?')} | "
        f"MT: {H(norm(pick(row, obs, 'mt_winner', 'winner_mt', 'merge_tree_winner')) or '?')} | "
        f"Direct: {H(norm(pick(row, obs, 'direct_error_group_winner', 'winner_direct_error_group')) or '?')} | "
        f"Distributional: {H(norm(pick(row, obs, 'distributional_group_winner', 'winner_distributional_group')) or row.get('overall_metric_majority', '?'))} | "
        f"Tail: {H(norm(pick(row, obs, 'tail_group_winner', 'winner_tail_group')) or '?')} | "
        f"Physics: {H(norm(pick(row, obs, 'configured_physics_group_winner', 'physics_group_winner', 'winner_physics_group')) or '?')}"
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

          {metric_table(row)}

          <div class="links">{' '.join(links)}</div>
        </div>

        <div class="thumb">
          {f"<a href='{H(crop)}' target='_blank'><img src='{H(crop)}'></a>" if crop else "<p class='muted'>No crop panel available.</p>"}
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
body {{
  margin: 0;
  background: #f7f7f8;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  color: #111;
}}
header {{
  position: sticky;
  top: 0;
  z-index: 10;
  background: white;
  border-bottom: 1px solid #ddd;
  padding: 16px 24px;
}}
h1 {{
  margin: 0 0 10px 0;
  font-size: 30px;
}}
button {{
  border: 1px solid #ddd;
  background: white;
  border-radius: 999px;
  padding: 8px 12px;
  margin: 0 6px 6px 0;
  cursor: pointer;
}}
button:hover {{
  background: #f0f4ff;
}}
main {{
  padding: 16px;
}}
.card {{
  background: white;
  border: 1px solid #ddd;
  border-radius: 14px;
  padding: 18px;
  margin-bottom: 18px;
  box-shadow: 0 1px 4px rgba(0,0,0,.06);
}}
.card-grid {{
  display: grid;
  grid-template-columns: minmax(520px, 1fr) minmax(360px, .8fr);
  gap: 20px;
  align-items: start;
}}
h2 {{
  font-size: 26px;
  margin: 0 0 10px 0;
}}
h3 {{
  font-size: 16px;
  margin: 18px 0 8px 0;
}}
.winner-line {{
  display: inline-block;
  background: #f5f5f5;
  border: 1px solid #ddd;
  border-radius: 8px;
  padding: 10px 12px;
  word-break: break-word;
}}
.forced {{
  background: #fff8e7;
  border-left: 4px solid #ff9f1a;
  padding: 8px 10px;
  border-radius: 6px;
}}
.chip {{
  display: inline-block;
  background: #eef5ff;
  border: 1px solid #b9d1ff;
  border-radius: 999px;
  padding: 6px 10px;
  margin: 0 6px 8px 0;
}}
.metric-box {{
  margin-top: 16px;
  border: 1px solid #ddd;
  border-radius: 10px;
  padding: 10px 12px;
  background: #fcfcfc;
}}
.metric-box summary {{
  cursor: pointer;
}}
.count {{
  margin-left: 12px;
  color: #333;
  font-size: 14px;
}}
.muted {{
  color: #666;
}}
.metrics {{
  border-collapse: collapse;
  width: 100%;
  margin-top: 10px;
  font-size: 13px;
}}
.metrics th {{
  text-align: left;
  background: #f0f0f0;
  padding: 6px;
}}
.metrics td {{
  border-top: 1px solid #e5e5e5;
  padding: 6px;
}}
.num {{
  text-align: right;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}}
.win {{
  display: inline-block;
  border-radius: 999px;
  padding: 3px 8px;
  font-weight: 700;
  font-size: 12px;
}}
.cnn {{
  background: #dff7e9;
  color: #11733b;
  border: 1px solid #a8e6c1;
}}
.gan {{
  background: #fff0d9;
  color: #a35b00;
  border: 1px solid #ffd39a;
}}
.tie {{
  background: #eee;
  color: #555;
  border: 1px solid #ccc;
}}
.warn {{
  background: yellow;
  padding: 2px 4px;
}}
.links {{
  margin-top: 14px;
  display: flex;
  gap: 14px;
  flex-wrap: wrap;
}}
.links a {{
  color: #0057b8;
  font-weight: 700;
  text-decoration: none;
}}
.links a:hover {{
  text-decoration: underline;
}}
.thumb {{
  position: sticky;
  top: 118px;
}}
.thumb img {{
  max-width: 100%;
  border: 1px solid #ddd;
  border-radius: 8px;
  background: white;
}}
@media(max-width: 1200px) {{
  .card-grid {{
    grid-template-columns: 1fr;
  }}
  .thumb {{
    position: static;
  }}
}}
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
  <button onclick="showOnly('tag-mt-cnn-all')">MT picks CNN ({count('mt_cnn_all')})</button>
  <button onclick="showOnly('tag-pd-gan-all')">PD picks GAN ({count('pd_gan_all')})</button>
  <button onclick="showOnly('tag-pd-cnn-all')">PD picks CNN ({count('pd_cnn_all')})</button>
  <button onclick="showOnly('tag-topology-consensus-gan')">PD=MT=GAN ({count('topology_consensus_gan')})</button>
  <button onclick="showOnly('tag-topology-consensus-cnn')">PD=MT=CNN ({count('topology_consensus_cnn')})</button>
  <button onclick="showOnly('tag-pd-gan-mt-cnn-control')">PD=GAN, MT=CNN ({count('pd_gan_mt_cnn_control')})</button>
  <button onclick="showOnly('tag-gan-metric-majority')">GAN metric majority ({count('gan_metric_majority')})</button>
  <button onclick="showOnly('tag-gan-majority-mt-rejects-gan')">GAN majority but MT≠GAN ({count('gan_majority_mt_rejects_gan')})</button>

  <br>

  <button onclick="showOnly('tag-adjacent-cluster-10-13')">Cluster 10–13</button>
  <button onclick="showOnly('tag-adjacent-cluster-76-78')">Cluster 76–78</button>
  <button onclick="showOnly('tag-adjacent-cluster-90-93')">Cluster 90–93</button>
  <button onclick="showOnly('tag-adjacent-cluster-161-164')">Cluster 161–164</button>

  <p class="muted">Each panel shows GT speed | CNN speed | GAN speed | |CNN-GT| | |GAN-GT|.</p>
</header>

<main>
{cards}
</main>
</body>
</html>
"""

    (OUTDIR / "index.html").write_text(page, encoding="utf-8")


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--all-samples",
        action="store_true",
        help="Generate the visual-inspection report for all samples from the metric CSVs, normally 168.",
    )
    parser.add_argument(
        "--merged-csv",
        default=None,
        help="Optional merged metrics CSV. Defaults to ttk_runs_fixed/combined/psnr_topology_physics_merged.csv.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Alias for --all-samples.",
    )
    parser.add_argument(
        "--samples",
        default="",
        help="Comma-separated sample ids to generate instead of default selection.",
    )
    parser.add_argument(
        "--no-panels",
        action="store_true",
        help="Only rebuild index/manifest using existing PNGs.",
    )

    args = parser.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    CROP_DIR.mkdir(parents=True, exist_ok=True)
    FULL_DIR.mkdir(parents=True, exist_ok=True)

    extra_merged_csv = Path(args.merged_csv).resolve() if args.merged_csv else None

    metrics = load_metric_rows(extra_merged_csv)
    obs = load_obs_rows()
    samples = select_samples(args, metrics, obs)

    gt = cnn = gan = pos = None

    if not args.no_panels:
        gt, cnn, gan, pos = load_arrays()

        # If --all-samples was requested but CSV metadata is unavailable,
        # fall back to all sample ids available in idx.npy.
        if (args.all or args.all_samples) and not samples:
            samples = sorted(pos.keys())

    if not samples:
        raise RuntimeError("No samples selected.")

    if args.all or args.all_samples:
        if len(samples) != 168:
            print(f"WARNING: all-sample mode selected {len(samples)} samples, expected 168.")

    print(f"repo_root={ROOT}")
    print(f"cnn_dir={CNN_DIR}")
    print(f"gan_dir={GAN_DIR}")
    print(f"outdir={OUTDIR}")
    print(f"selected_samples={len(samples)}")
    print("forced/extra samples:", " ".join(map(str, sorted(FORCED))))

    entries = []
    manifest = []

    for sid in samples:
        row = metrics.get(sid, {"sample_idx": str(sid)})
        ob = obs.get(sid, {"sample_idx": str(sid)})

        crop_path = CROP_DIR / f"sample_{sid:03d}_crop.png"
        full_path = FULL_DIR / f"sample_{sid:03d}_full.png"

        crop_ok = crop_path.exists()
        full_ok = full_path.exists()

        if not args.no_panels:
            crop_ok = make_panel(sid, gt, cnn, gan, pos, row, ob, (0, 160, 0, 160), crop_path)
            full_ok = make_panel(sid, gt, cnn, gan, pos, row, ob, None, full_path)

        crop_rel = crop_path.relative_to(OUTDIR).as_posix() if crop_ok else ""
        full_rel = full_path.relative_to(OUTDIR).as_posix() if full_ok else ""

        groups = infer_groups(sid, row, ob)
        q = question(sid, row, ob)

        cnn_metric_wins, gan_metric_wins, metric_ties = metric_counts(row)

        entry = {
            "sample_idx": sid,
            "row": row,
            "obs": ob,
            "groups": groups,
            "question": q,
            "crop_panel": crop_rel,
            "full_panel": full_rel,
        }
        entries.append(entry)

        manifest.append({
            "sample_idx": sid,
            "psnr_winner": norm(pick(row, ob, "psnr_winner", "winner_psnr")),
            "ssim_winner": norm(pick(row, ob, "ssim_winner", "winner_ssim")),
            "pd_winner": norm(pick(row, ob, "pd_winner", "winner_pd", "bottleneck_pd_winner")),
            "mt_winner": norm(pick(row, ob, "mt_winner", "winner_mt", "merge_tree_winner")),
            "direct_error_group_winner": norm(pick(row, ob, "direct_error_group_winner", "winner_direct_error_group")),
            "distributional_group_winner": (
                norm(pick(row, ob, "distributional_group_winner", "winner_distributional_group"))
                or row.get("overall_metric_majority", "")
            ),
            "tail_group_winner": norm(pick(row, ob, "tail_group_winner", "winner_tail_group")),
            "configured_physics_group_winner": norm(
                pick(row, ob, "configured_physics_group_winner", "physics_group_winner", "winner_physics_group")
            ),
            "cnn_metric_wins": cnn_metric_wins,
            "gan_metric_wins": gan_metric_wins,
            "metric_ties": metric_ties,
            "overall_metric_majority": row.get("overall_metric_majority", ""),
            "question": q,
            "groups": ";".join(groups),
            "crop_panel": crop_rel,
            "full_panel": full_rel,
            "forced_reason": FORCED.get(sid, ""),
        })

    entries.sort(key=lambda e: e["sample_idx"])
    manifest.sort(key=lambda r: int(r["sample_idx"]))

    write_csv(
        OUTDIR / "visual_inspection_manifest.csv",
        manifest,
        [
            "sample_idx",
            "psnr_winner",
            "ssim_winner",
            "pd_winner",
            "mt_winner",
            "direct_error_group_winner",
            "distributional_group_winner",
            "tail_group_winner",
            "configured_physics_group_winner",
            "cnn_metric_wins",
            "gan_metric_wins",
            "metric_ties",
            "overall_metric_majority",
            "question",
            "groups",
            "crop_panel",
            "full_panel",
            "forced_reason",
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