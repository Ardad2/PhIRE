#!/usr/bin/env python3
"""
Generate qualitative-observation groups for the TopoAware SR paper.

Purpose
-------
This script turns the final repaired merged metric table into observation groups
for manual/visual inspection. It is designed to be run after the report-table
metric sweep, using the authoritative repaired merged table:

    ttk_runs_fixed/combined/psnr_topology_physics_merged.csv

The script writes per-sample winner summaries, diagnostic groups, sample-id
lists, and a README explaining how to interpret the groups.

Core idea
---------
The groups are meant for observation, not for proving a universal winner.

They separate:
  1. CNN-consensus / distortion-faithful cases,
  2. GAN-distributional cases,
  3. PD-vs-MT disagreement cases,
  4. MT-GAN diagnostic cases,
  5. candidate structural-hallucination cases where PD favors GAN but MT favors CNN.

Winner convention
-----------------
For high-is-better metrics such as PSNR and SSIM:
    higher value wins.

For distance/error metrics such as WPD MAE, PSD log-L2, gradient W1, PD distance,
and MT distance:
    lower value wins.

For signed-error columns such as wpd_bias:
    lower absolute value wins.

For delta columns that already encode absolute error, such as
exceed_frac_abs_delta_p99:
    lower value wins.

Outputs
-------
The default output directory is:

    ttk_runs_fixed/observation_groups/

Important generated files:
  - observation_groups_per_sample.csv
  - observation_group_summary.csv
  - group_cnn_consensus_core.csv
  - group_gan_distributional_cases.csv
  - group_pd_mt_disagreement.csv
  - group_candidate_structural_hallucination_signature.csv
  - group_mt_gan_diagnostic.csv
  - group_topology_consensus_gan.csv
  - recommended_visual_inspection_cases.csv
  - sample_ids_*.txt
  - README_observation_groups.md

Example
-------
PYTHONNOUSERSITE=1 /usr/bin/python3 scripts/generate_observation_groups.py \
  --merged-csv ttk_runs_fixed/combined/psnr_topology_physics_merged.csv \
  --outdir ttk_runs_fixed/observation_groups
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class MetricSpec:
    name: str
    column: str
    direction: str  # "higher", "lower", or "lower_abs"
    family: str
    description: str


METRICS: List[MetricSpec] = [
    # Standard / topology
    MetricSpec("PSNR_uv", "psnr", "higher", "standard", "Vector-field PSNR; higher is better."),
    MetricSpec("SSIM_speed", "ssim", "higher", "standard", "Speed-magnitude SSIM; higher is better."),
    MetricSpec("PD_distance", "pd_distance", "lower", "topology", "Persistence-diagram bottleneck distance; lower is better."),
    MetricSpec("MT_distance", "mt_distance", "lower", "topology", "Merge-tree distance; lower is better."),

    # Direct physics/domain error measures
    MetricSpec("WPD_bias_abs", "wpd_bias", "lower_abs", "direct_error", "Absolute WPD signed bias; lower absolute value is better."),
    MetricSpec("WPD_MAE", "wpd_mae", "lower", "direct_error", "WPD MAE; lower is better."),
    MetricSpec("WPD_RMSE", "wpd_rmse", "lower", "direct_error", "WPD RMSE; lower is better."),
    MetricSpec("WPD_W1", "wpd_w1", "lower", "distributional", "WPD Wasserstein-1 distance; lower is better."),

    # Spectral / distributional / texture-like measures
    MetricSpec("PSD_log_L2", "psd_log_l2", "lower", "distributional", "Radial PSD log-L2 mismatch; lower is better."),
    MetricSpec("PSD_slope_abs_delta", "psd_slope_abs_delta", "lower", "distributional", "Absolute PSD slope mismatch; lower is better."),
    MetricSpec("Gradient_MAE", "grad_mae", "lower", "direct_error", "Gradient MAE; lower is better."),
    MetricSpec("Gradient_W1", "grad_w1", "lower", "distributional", "Gradient-distribution Wasserstein-1 distance; lower is better."),
    MetricSpec("Gradient_kurtosis_abs_delta", "grad_kurtosis_abs_delta", "lower", "distributional", "Absolute gradient-kurtosis mismatch; lower is better."),

    # Tail / exceedance measures
    MetricSpec("Exceed_abs_t5", "exceed_frac_abs_delta_t5", "lower", "tail", "Absolute exceedance-fraction error, speed > 5; lower is better."),
    MetricSpec("Exceed_abs_t10", "exceed_frac_abs_delta_t10", "lower", "tail", "Absolute exceedance-fraction error, speed > 10; lower is better."),
    MetricSpec("Exceed_abs_t15", "exceed_frac_abs_delta_t15", "lower", "tail", "Absolute exceedance-fraction error, speed > 15; lower is better."),
    MetricSpec("Exceed_abs_p90", "exceed_frac_abs_delta_p90", "lower", "tail", "Absolute exceedance-fraction error, p90 threshold; lower is better."),
    MetricSpec("Exceed_abs_p95", "exceed_frac_abs_delta_p95", "lower", "tail", "Absolute exceedance-fraction error, p95 threshold; lower is better."),
    MetricSpec("Exceed_abs_p99", "exceed_frac_abs_delta_p99", "lower", "tail", "Absolute exceedance-fraction error, p99 threshold; lower is better."),
]


# Group definitions. These are intentionally explicit so the qualitative
# grouping logic is reproducible and easy to audit.
DIRECT_ERROR_GROUP = [
    "PSNR_uv",
    "SSIM_speed",
    "WPD_MAE",
    "WPD_RMSE",
    "Gradient_MAE",
]

DISTRIBUTIONAL_GROUP = [
    "PSD_log_L2",
    "PSD_slope_abs_delta",
    "Gradient_W1",
    "Exceed_abs_p95",
    "Exceed_abs_p99",
]

TAIL_GROUP = [
    "Exceed_abs_p90",
    "Exceed_abs_p95",
    "Exceed_abs_p99",
]

# This matches the compact physics group used earlier in the paper discussion.
CONFIGURED_PHYSICS_GROUP = [
    "WPD_RMSE",
    "WPD_MAE",
    "PSD_log_L2",
    "Gradient_MAE",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--merged-csv",
        type=Path,
        default=Path("ttk_runs_fixed/combined/psnr_topology_physics_merged.csv"),
        help="Final repaired merged metric table with one CNN row and one GAN row per sample.",
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=Path("ttk_runs_fixed/observation_groups"),
        help="Output directory for generated groups.",
    )
    p.add_argument(
        "--near-tie-thresholds",
        type=float,
        nargs="*",
        default=[0.05, 0.075],
        help="SSIM near-tie thresholds to add as flags.",
    )
    p.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of cases to include from each ranked diagnostic group in the recommendation file.",
    )
    return p.parse_args()


def read_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise SystemExit(f"Missing merged CSV: {path}")
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def to_float(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def metric_value(row: Dict[str, str], spec: MetricSpec) -> float:
    raw = to_float(row.get(spec.column, "nan"))
    if spec.direction == "lower_abs":
        return abs(raw)
    return raw


def winner(cnn_row: Dict[str, str], gan_row: Dict[str, str], spec: MetricSpec) -> str:
    cv = metric_value(cnn_row, spec)
    gv = metric_value(gan_row, spec)

    if math.isnan(cv) or math.isnan(gv):
        return "unavailable"

    if math.isclose(cv, gv, rel_tol=0.0, abs_tol=0.0):
        return "tie"

    if spec.direction == "higher":
        return "CNN" if cv > gv else "GAN"

    # lower / lower_abs
    return "CNN" if cv < gv else "GAN"


def cnn_positive_delta(cnn_row: Dict[str, str], gan_row: Dict[str, str], spec: MetricSpec) -> float:
    """Return a signed delta where positive means CNN wins, negative means GAN wins."""
    cv = metric_value(cnn_row, spec)
    gv = metric_value(gan_row, spec)
    if math.isnan(cv) or math.isnan(gv):
        return float("nan")
    if spec.direction == "higher":
        return cv - gv
    return gv - cv


def majority_winner(metric_winners: Dict[str, str], metric_names: Iterable[str]) -> Tuple[str, int, int, int]:
    cnn = gan = tie = 0
    for m in metric_names:
        w = metric_winners.get(m, "unavailable")
        if w == "CNN":
            cnn += 1
        elif w == "GAN":
            gan += 1
        else:
            tie += 1

    if cnn > gan:
        return "CNN", cnn, gan, tie
    if gan > cnn:
        return "GAN", cnn, gan, tie
    return "tie", cnn, gan, tie


def percentile_scores(values_by_sample: Dict[int, float]) -> Dict[int, float]:
    """Return percentile scores in [0,1], where larger input gets larger percentile."""
    clean = [(sid, v) for sid, v in values_by_sample.items() if not math.isnan(v)]
    if not clean:
        return {sid: float("nan") for sid in values_by_sample}

    clean_sorted = sorted(clean, key=lambda kv: kv[1])
    n = len(clean_sorted)
    out: Dict[int, float] = {}
    if n == 1:
        out[clean_sorted[0][0]] = 1.0
    else:
        for rank, (sid, _v) in enumerate(clean_sorted):
            out[sid] = rank / (n - 1)
    for sid in values_by_sample:
        out.setdefault(sid, float("nan"))
    return out


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: List[str] = []
        for r in rows:
            for k in r.keys():
                if k not in keys:
                    keys.append(k)
        fieldnames = keys
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def write_sample_ids(path: Path, rows: List[Dict[str, object]]) -> None:
    ids = [str(int(r["sample_idx"])) for r in rows]
    path.write_text(",".join(ids) + ("\n" if ids else ""))


def ids_string(rows: List[Dict[str, object]], limit: Optional[int] = None) -> str:
    vals = [str(int(r["sample_idx"])) for r in rows]
    if limit is not None and len(vals) > limit:
        return ",".join(vals[:limit]) + f",... (+{len(vals)-limit} more)"
    return ",".join(vals)


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    input_rows = read_rows(args.merged_csv)

    by_sample: Dict[int, Dict[str, Dict[str, str]]] = {}
    for row in input_rows:
        method = row.get("method", "").strip().lower()
        if method not in {"cnn", "gan"}:
            continue
        sample_idx = int(float(row["sample_idx"]))
        by_sample.setdefault(sample_idx, {})[method] = row

    metric_by_name = {m.name: m for m in METRICS}
    available_metrics = [m for m in METRICS if m.column in input_rows[0]]

    missing_pairs = [sid for sid, d in by_sample.items() if "cnn" not in d or "gan" not in d]
    if missing_pairs:
        raise SystemExit(f"Found samples without both CNN and GAN rows: {missing_pairs[:10]}")

    per_sample: List[Dict[str, object]] = []

    for sample_idx in sorted(by_sample):
        cnn = by_sample[sample_idx]["cnn"]
        gan = by_sample[sample_idx]["gan"]

        winners: Dict[str, str] = {}
        deltas: Dict[str, float] = {}
        for spec in available_metrics:
            winners[spec.name] = winner(cnn, gan, spec)
            deltas[spec.name] = cnn_positive_delta(cnn, gan, spec)

        direct_winner, direct_cnn, direct_gan, direct_tie = majority_winner(winners, DIRECT_ERROR_GROUP)
        distributional_winner, dist_cnn, dist_gan, dist_tie = majority_winner(winners, DISTRIBUTIONAL_GROUP)
        tail_winner, tail_cnn, tail_gan, tail_tie = majority_winner(winners, TAIL_GROUP)
        physics_winner, phys_cnn, phys_gan, phys_tie = majority_winner(winners, CONFIGURED_PHYSICS_GROUP)

        psnr_w = winners.get("PSNR_uv", "unavailable")
        ssim_w = winners.get("SSIM_speed", "unavailable")
        pd_w = winners.get("PD_distance", "unavailable")
        mt_w = winners.get("MT_distance", "unavailable")

        ssim_delta = deltas.get("SSIM_speed", float("nan"))
        pd_delta = deltas.get("PD_distance", float("nan"))
        mt_delta = deltas.get("MT_distance", float("nan"))

        # Positive strengths below are used for ranking diagnostic cases.
        # For PD/MT deltas, negative means GAN and positive means CNN.
        pd_gan_strength = max(0.0, -pd_delta) if not math.isnan(pd_delta) else float("nan")
        mt_cnn_strength = max(0.0, mt_delta) if not math.isnan(mt_delta) else float("nan")
        mt_gan_strength = max(0.0, -mt_delta) if not math.isnan(mt_delta) else float("nan")

        row: Dict[str, object] = {
            "sample_idx": sample_idx,
            "psnr_winner": psnr_w,
            "ssim_winner": ssim_w,
            "pd_winner": pd_w,
            "mt_winner": mt_w,
            "direct_error_group_winner": direct_winner,
            "direct_error_cnn_votes": direct_cnn,
            "direct_error_gan_votes": direct_gan,
            "direct_error_ties": direct_tie,
            "distributional_group_winner": distributional_winner,
            "distributional_cnn_votes": dist_cnn,
            "distributional_gan_votes": dist_gan,
            "distributional_ties": dist_tie,
            "tail_group_winner": tail_winner,
            "tail_cnn_votes": tail_cnn,
            "tail_gan_votes": tail_gan,
            "tail_ties": tail_tie,
            "configured_physics_group_winner": physics_winner,
            "configured_physics_cnn_votes": phys_cnn,
            "configured_physics_gan_votes": phys_gan,
            "configured_physics_ties": phys_tie,
            "delta_ssim_cnn_positive": deltas.get("SSIM_speed", ""),
            "delta_psnr_cnn_positive": deltas.get("PSNR_uv", ""),
            "delta_pd_cnn_positive": deltas.get("PD_distance", ""),
            "delta_mt_cnn_positive": deltas.get("MT_distance", ""),
            "pd_gan_strength": pd_gan_strength,
            "mt_cnn_strength": mt_cnn_strength,
            "mt_gan_strength": mt_gan_strength,
        }

        for eps in args.near_tie_thresholds:
            tag = str(eps).replace(".", "p")
            row[f"ssim_near_tie_{tag}"] = bool(not math.isnan(ssim_delta) and abs(ssim_delta) <= eps)

        # Observation group flags.
        row["group_cnn_consensus_core"] = (
            psnr_w == "CNN"
            and ssim_w == "CNN"
            and mt_w == "CNN"
            and direct_winner == "CNN"
            and physics_winner == "CNN"
        )

        row["group_gan_distributional_cases"] = (
            psnr_w == "CNN"
            and ssim_w == "CNN"
            and direct_winner == "CNN"
            and distributional_winner == "GAN"
        )

        row["group_pd_mt_disagreement"] = (
            pd_w in {"CNN", "GAN"}
            and mt_w in {"CNN", "GAN"}
            and pd_w != mt_w
        )

        row["group_candidate_structural_hallucination_signature"] = (
            pd_w == "GAN"
            and mt_w == "CNN"
        )

        row["group_mt_gan_diagnostic"] = mt_w == "GAN"
        row["group_topology_consensus_gan"] = (mt_w == "GAN" and pd_w == "GAN")
        row["group_topology_consensus_cnn"] = (mt_w == "CNN" and pd_w == "CNN")

        # This is the special near-tie pattern discussed in the paper:
        # SSIM says CNN, both topology descriptors say GAN, and configured physics/direct validators say CNN.
        row["group_neartie_topology_validator_disagreement"] = (
            bool(row.get("ssim_near_tie_0p075", False))
            and ssim_w == "CNN"
            and mt_w == "GAN"
            and pd_w == "GAN"
            and direct_winner == "CNN"
            and physics_winner == "CNN"
        )

        per_sample.append(row)

    # Percentile-based scores for prioritizing visual inspection.
    pd_strength_by_sid = {int(r["sample_idx"]): float(r["pd_gan_strength"]) for r in per_sample}
    mt_cnn_strength_by_sid = {int(r["sample_idx"]): float(r["mt_cnn_strength"]) for r in per_sample}
    mt_gan_strength_by_sid = {int(r["sample_idx"]): float(r["mt_gan_strength"]) for r in per_sample}

    pd_gan_pct = percentile_scores(pd_strength_by_sid)
    mt_cnn_pct = percentile_scores(mt_cnn_strength_by_sid)
    mt_gan_pct = percentile_scores(mt_gan_strength_by_sid)

    for r in per_sample:
        sid = int(r["sample_idx"])
        dist_frac = float(r["distributional_gan_votes"]) / max(1, len(DISTRIBUTIONAL_GROUP))
        tail_frac = float(r["tail_gan_votes"]) / max(1, len(TAIL_GROUP))

        r["pd_gan_strength_percentile"] = pd_gan_pct[sid]
        r["mt_cnn_strength_percentile"] = mt_cnn_pct[sid]
        r["mt_gan_strength_percentile"] = mt_gan_pct[sid]
        r["distributional_gan_vote_fraction"] = dist_frac
        r["tail_gan_vote_fraction"] = tail_frac

        r["candidate_hallucination_score"] = (
            0.40 * pd_gan_pct[sid]
            + 0.40 * mt_cnn_pct[sid]
            + 0.20 * dist_frac
        )

        r["mt_gan_visual_score"] = (
            0.50 * mt_gan_pct[sid]
            + 0.30 * dist_frac
            + 0.20 * tail_frac
        )

    # Write full per-sample table.
    write_csv(args.outdir / "observation_groups_per_sample.csv", per_sample)

    # Define output groups and their sorting keys.
    groups = [
        (
            "cnn_consensus_core",
            "group_cnn_consensus_core",
            "CNN-consensus / distortion-faithful cases",
            "Samples where PSNR, SSIM, MT, the direct-error group, and the configured physics group all favor CNN.",
            "sample_idx",
            False,
        ),
        (
            "gan_distributional_cases",
            "group_gan_distributional_cases",
            "GAN-distributional cases",
            "Samples where PSNR/SSIM/direct-error metrics favor CNN, but the distributional group favors GAN.",
            "distributional_gan_vote_fraction",
            True,
        ),
        (
            "pd_mt_disagreement",
            "group_pd_mt_disagreement",
            "PD-vs-MT disagreement cases",
            "Samples where PD and MT choose different topology winners.",
            "candidate_hallucination_score",
            True,
        ),
        (
            "candidate_structural_hallucination_signature",
            "group_candidate_structural_hallucination_signature",
            "Candidate structural-hallucination signature cases",
            "Samples where PD favors GAN but MT favors CNN; these are candidates where GAN may have plausible persistence statistics but different merge hierarchy.",
            "candidate_hallucination_score",
            True,
        ),
        (
            "mt_gan_diagnostic",
            "group_mt_gan_diagnostic",
            "MT-GAN diagnostic cases",
            "Samples where MT favors GAN despite SSIM/PSNR being CNN-favoring in this dataset.",
            "mt_gan_visual_score",
            True,
        ),
        (
            "topology_consensus_gan",
            "group_topology_consensus_gan",
            "Topology-consensus GAN cases",
            "Samples where both PD and MT favor GAN.",
            "mt_gan_visual_score",
            True,
        ),
        (
            "topology_consensus_cnn",
            "group_topology_consensus_cnn",
            "Topology-consensus CNN cases",
            "Samples where both PD and MT favor CNN.",
            "sample_idx",
            False,
        ),
        (
            "neartie_topology_validator_disagreement",
            "group_neartie_topology_validator_disagreement",
            "Near-tie topology-validator disagreement cases",
            "SSIM-near-tie samples where SSIM and configured validators favor CNN but PD and MT favor GAN.",
            "mt_gan_visual_score",
            True,
        ),
    ]

    summary_rows: List[Dict[str, object]] = []
    group_rows_by_name: Dict[str, List[Dict[str, object]]] = {}

    for short_name, flag, title, description, sort_key, reverse in groups:
        rows = [r for r in per_sample if bool(r.get(flag, False))]
        rows = sorted(rows, key=lambda r: (float(r.get(sort_key, 0.0)) if sort_key != "sample_idx" else int(r["sample_idx"])), reverse=reverse)
        group_rows_by_name[short_name] = rows
        write_csv(args.outdir / f"group_{short_name}.csv", rows)
        write_sample_ids(args.outdir / f"sample_ids_{short_name}.txt", rows)
        summary_rows.append({
            "group": short_name,
            "title": title,
            "count": len(rows),
            "sample_ids": ids_string(rows),
            "sample_ids_preview": ids_string(rows, limit=40),
            "description": description,
        })

    write_csv(args.outdir / "observation_group_summary.csv", summary_rows)

    # Recommended visual inspection list.
    recommendations: List[Dict[str, object]] = []

    def add_recs(group_name: str, rows: List[Dict[str, object]], reason: str, top_n: int) -> None:
        for rank, r in enumerate(rows[:top_n], start=1):
            out = dict(r)
            out["recommendation_group"] = group_name
            out["recommendation_rank_within_group"] = rank
            out["reason"] = reason
            recommendations.append(out)

    add_recs(
        "candidate_structural_hallucination_signature",
        group_rows_by_name["candidate_structural_hallucination_signature"],
        "PD favors GAN while MT favors CNN; inspect whether GAN has plausible but spatially/hierarchically misaligned structures.",
        args.top_n,
    )
    add_recs(
        "mt_gan_diagnostic",
        group_rows_by_name["mt_gan_diagnostic"],
        "MT favors GAN; inspect whether MT is rewarding real structure or GAN artifacts.",
        args.top_n,
    )
    add_recs(
        "gan_distributional_cases",
        group_rows_by_name["gan_distributional_cases"],
        "Distributional metrics favor GAN while direct-error metrics favor CNN; inspect perception-distortion tradeoff.",
        args.top_n,
    )
    add_recs(
        "neartie_topology_validator_disagreement",
        group_rows_by_name["neartie_topology_validator_disagreement"],
        "Near-tie topology-consensus GAN but configured validators favor CNN; highest-priority qualitative audit cases.",
        args.top_n,
    )

    # Deduplicate recommendations by (sample_idx, recommendation_group) first, then create a compact unique sample list.
    write_csv(args.outdir / "recommended_visual_inspection_cases.csv", recommendations)

    unique_recommended: Dict[int, Dict[str, object]] = {}
    for r in recommendations:
        sid = int(r["sample_idx"])
        if sid not in unique_recommended:
            unique_recommended[sid] = {
                "sample_idx": sid,
                "appears_in_groups": [],
                "psnr_winner": r["psnr_winner"],
                "ssim_winner": r["ssim_winner"],
                "pd_winner": r["pd_winner"],
                "mt_winner": r["mt_winner"],
                "direct_error_group_winner": r["direct_error_group_winner"],
                "distributional_group_winner": r["distributional_group_winner"],
                "tail_group_winner": r["tail_group_winner"],
                "configured_physics_group_winner": r["configured_physics_group_winner"],
                "candidate_hallucination_score": r["candidate_hallucination_score"],
                "mt_gan_visual_score": r["mt_gan_visual_score"],
            }
        unique_recommended[sid]["appears_in_groups"].append(str(r["recommendation_group"]))

    unique_rows = []
    for sid, r in sorted(unique_recommended.items()):
        rr = dict(r)
        rr["appears_in_groups"] = ";".join(sorted(set(rr["appears_in_groups"])))
        unique_rows.append(rr)

    write_csv(args.outdir / "recommended_visual_inspection_unique_samples.csv", unique_rows)
    (args.outdir / "sample_ids_recommended_visual_inspection_unique.txt").write_text(
        ",".join(str(int(r["sample_idx"])) for r in unique_rows) + ("\n" if unique_rows else "")
    )

    # Metric winner summary for this script's groups.
    metric_summary_rows: List[Dict[str, object]] = []
    for spec in available_metrics:
        cnn_count = gan_count = tie_count = unavailable = 0
        for r in per_sample:
            w = r.get(f"{spec.name}_winner")
            # Winners are not stored with this suffix in per_sample; recompute from summary below.
        sample_winners = []
        for sid in sorted(by_sample):
            w = winner(by_sample[sid]["cnn"], by_sample[sid]["gan"], spec)
            sample_winners.append(w)
        cnn_count = sample_winners.count("CNN")
        gan_count = sample_winners.count("GAN")
        tie_count = sample_winners.count("tie")
        unavailable = sample_winners.count("unavailable")
        metric_summary_rows.append({
            "metric_name": spec.name,
            "column": spec.column,
            "family": spec.family,
            "winner_rule": spec.direction,
            "cnn_wins": cnn_count,
            "gan_wins": gan_count,
            "ties": tie_count,
            "unavailable": unavailable,
            "description": spec.description,
        })
    write_csv(args.outdir / "observation_metric_winner_summary.csv", metric_summary_rows)

    # README
    summary_by_group = {r["group"]: r for r in summary_rows}
    readme_lines = [
        "# Observation Groups for Qualitative Audit",
        "",
        "This directory was generated by `generate_observation_groups.py`.",
        "",
        "The groups are for **manual inspection and qualitative interpretation**, not for declaring a universal model winner.",
        "",
        "## Input",
        "",
        f"- merged CSV: `{args.merged_csv}`",
        "",
        "## Key interpretation",
        "",
        "- CNN is expected to dominate direct-error metrics such as PSNR, SSIM, WPD MAE/RMSE, and gradient MAE.",
        "- GAN may dominate distributional or tail-oriented metrics such as PSD log-L2, gradient W1, and upper-tail exceedance.",
        "- PD and MT should be interpreted as different topological diagnostics rather than interchangeable topology signals.",
        "- PD-GAN / MT-CNN disagreement is a candidate signature that GAN has plausible topological feature lifetimes but different merge hierarchy.",
        "- MT-GAN cases are useful diagnostic cases where topology favors GAN despite the conventional CNN-favoring baseline.",
        "",
        "## Generated groups",
        "",
    ]
    for row in summary_rows:
        readme_lines.extend([
            f"### `{row['group']}`",
            "",
            f"- count: {row['count']}",
            f"- sample IDs: `{row['sample_ids_preview']}`",
            f"- definition: {row['description']}",
            "",
        ])

    readme_lines.extend([
        "## Recommended next visual panels",
        "",
        "A useful visual panel for each recommended sample is:",
        "",
        "```text",
        "GT speed | CNN speed | GAN speed | |CNN-GT| |GAN-GT|",
        "```",
        "",
        "The most useful question is not simply whether GAN looks sharper, but whether the sharper structures are present in the GT and correctly organized.",
        "",
        "## Example command",
        "",
        "```bash",
        "PYTHONNOUSERSITE=1 /usr/bin/python3 scripts/generate_observation_groups.py \\",
        "  --merged-csv ttk_runs_fixed/combined/psnr_topology_physics_merged.csv \\",
        "  --outdir ttk_runs_fixed/observation_groups",
        "```",
        "",
    ])
    (args.outdir / "README_observation_groups.md").write_text("\n".join(readme_lines))

    print(f"Wrote observation groups to: {args.outdir}")
    print("Summary:")
    for row in summary_rows:
        print(f"  {row['group']}: {row['count']} samples")
    print(f"Recommended unique visual samples: {len(unique_rows)}")


if __name__ == "__main__":
    main()
