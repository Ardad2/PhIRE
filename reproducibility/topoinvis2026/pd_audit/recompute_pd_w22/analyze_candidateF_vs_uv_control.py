#!/usr/bin/env python3
"""
Matched-control analysis for:

    pretrained CNN
    Candidate UV expanded-2688 (L_uv-only matched fine-tuning control)
    Candidate F grad+E2-low expanded-2688

Purpose
-------
Quantify whether Candidate F's persistence-diagram improvements are merely
associated with ordinary fine-tuning, or whether they exceed the matched
L_uv-only control.

This is a descriptive matched-control decomposition, not a formal causal
percentage attribution.

Outputs
-------
- candidateF_vs_uv_control_summary.txt
- candidateF_vs_uv_control_samples.csv
- candidateF_vs_uv_control_metric_summary.csv
"""

import csv
import math
import os
from pathlib import Path

import numpy as np


HOME = Path.home()
ROOT = HOME / "PhIRE"

W22 = Path(os.environ["W22"])

SWEEP = W22 / "w22_full_sweep.csv"

UV_EVAL = (
    ROOT
    / "ttk_runs_fixed"
    / "topology_finetuning"
    / "candidateUV_expanded2688_eval"
    / "all_sample_metrics_candidateUV_expanded2688.csv"
)

F_EVAL = (
    ROOT
    / "ttk_runs_fixed"
    / "topology_finetuning"
    / "candidateF_grad_E2_low_expanded2688_eval"
    / "all_sample_metrics_candidateF_grad_E2_low_expanded2688.csv"
)

OUT_SUMMARY = W22 / "candidateF_vs_uv_control_summary.txt"
OUT_SAMPLES = W22 / "candidateF_vs_uv_control_samples.csv"
OUT_METRICS = W22 / "candidateF_vs_uv_control_metric_summary.csv"


CNN_RUN = "cnn"

UV_RUN = (
    "topology_finetuning/"
    "candidateUV_expanded2688_topology"
)

F_RUN = (
    "topology_finetuning/"
    "candidateF_grad_E2_low_expanded2688_topology"
)


CNN_METHOD = "cnn"
UV_METHOD = "candidateUV_expanded2688"
F_METHOD = "candidateF_grad_E2_low_expanded2688"

N = 168

SELECTED = [78, 71, 80, 63, 69]


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def metric(row, logical):
    candidates = {
        "db": ["bottleneck_all", "db_all", "dB_all"],
        "w2inf": ["w2inf_all", "w2_inf_all"],
        "w22": ["w22_all"],
    }[logical]

    for col in candidates:
        if col in row and row[col] not in ("", None):
            x = float(row[col])
            if not math.isfinite(x):
                raise RuntimeError(
                    f"Nonfinite {logical} from {col}: {row[col]}"
                )
            return x

    raise KeyError(
        f"Could not find {logical}; tried {candidates}. "
        f"Columns={list(row.keys())}"
    )


def pct_change(new, old):
    return 100.0 * (new - old) / old


def improvement(lower_better_baseline, lower_better_candidate):
    return lower_better_baseline - lower_better_candidate


def sign(x, eps=1e-12):
    if x > eps:
        return 1
    if x < -eps:
        return -1
    return 0


def build_sweep_index():
    rows = read_csv(SWEEP)
    out = {}

    for r in rows:
        key = (r["run"], int(r["sample"]))
        if key in out:
            raise RuntimeError(f"Duplicate sweep row: {key}")
        out[key] = r

    return out


def build_eval_index(path):
    rows = read_csv(path)
    out = {}

    for r in rows:
        key = (r["method"], int(r["sample_idx"]))
        if key in out:
            raise RuntimeError(f"Duplicate eval row: {key}")
        out[key] = r

    return out


def finite(row, field):
    x = float(row[field])
    if not math.isfinite(x):
        raise RuntimeError(
            f"Nonfinite {field}: {row[field]}"
        )
    return x


sweep = build_sweep_index()
uv_eval = build_eval_index(UV_EVAL)
f_eval = build_eval_index(F_EVAL)

per_sample = []

for sample in range(N):
    rows = {
        "cnn": sweep[(CNN_RUN, sample)],
        "uv": sweep[(UV_RUN, sample)],
        "f": sweep[(F_RUN, sample)],
    }

    m = {
        method: {
            name: metric(row, name)
            for name in ["db", "w2inf", "w22"]
        }
        for method, row in rows.items()
    }

    cnn_eval = f_eval[(CNN_METHOD, sample)]
    uv_row = uv_eval[(UV_METHOD, sample)]
    f_row = f_eval[(F_METHOD, sample)]

    # Check CNN conventional metric consistency between candidate-specific tables.
    uv_table_cnn = uv_eval[(CNN_METHOD, sample)]

    for field in ["psnruv", "speed_mae", "speed_rmse"]:
        a = finite(cnn_eval, field)
        b = finite(uv_table_cnn, field)

        if abs(a - b) > 1e-12:
            raise RuntimeError(
                f"CNN eval mismatch sample={sample}, field={field}: {a} vs {b}"
            )

    rec = {
        "sample": sample,
    }

    for name in ["db", "w2inf", "w22"]:
        cnn = m["cnn"][name]
        uv = m["uv"][name]
        cand = m["f"][name]

        rec[f"cnn_{name}"] = cnn
        rec[f"uv_{name}"] = uv
        rec[f"f_{name}"] = cand

        # Positive means the second method is topologically better.
        rec[f"uv_gain_vs_cnn_{name}"] = cnn - uv
        rec[f"f_gain_vs_uv_{name}"] = uv - cand
        rec[f"f_gain_vs_cnn_{name}"] = cnn - cand

        rec[f"uv_better_than_cnn_{name}"] = int(uv < cnn)
        rec[f"f_better_than_uv_{name}"] = int(cand < uv)
        rec[f"f_better_than_cnn_{name}"] = int(cand < cnn)

    rec["f_better_than_uv_all3"] = int(
        rec["f_better_than_uv_db"]
        and rec["f_better_than_uv_w2inf"]
        and rec["f_better_than_uv_w22"]
    )

    rec["uv_better_than_cnn_all3"] = int(
        rec["uv_better_than_cnn_db"]
        and rec["uv_better_than_cnn_w2inf"]
        and rec["uv_better_than_cnn_w22"]
    )

    rec["f_better_than_cnn_all3"] = int(
        rec["f_better_than_cnn_db"]
        and rec["f_better_than_cnn_w2inf"]
        and rec["f_better_than_cnn_w22"]
    )

    cnn_psnr = finite(cnn_eval, "psnruv")
    uv_psnr = finite(uv_row, "psnruv")
    f_psnr = finite(f_row, "psnruv")

    cnn_mae = finite(cnn_eval, "speed_mae")
    uv_mae = finite(uv_row, "speed_mae")
    f_mae = finite(f_row, "speed_mae")

    cnn_rmse = finite(cnn_eval, "speed_rmse")
    uv_rmse = finite(uv_row, "speed_rmse")
    f_rmse = finite(f_row, "speed_rmse")

    rec.update({
        "cnn_psnruv": cnn_psnr,
        "uv_psnruv": uv_psnr,
        "f_psnruv": f_psnr,

        "cnn_speed_mae": cnn_mae,
        "uv_speed_mae": uv_mae,
        "f_speed_mae": f_mae,

        "cnn_speed_rmse": cnn_rmse,
        "uv_speed_rmse": uv_rmse,
        "f_speed_rmse": f_rmse,

        "f_vs_uv_abs_delta_psnruv":
            abs(f_psnr - uv_psnr),

        "f_vs_uv_relative_gap_speed_mae":
            abs(f_mae - uv_mae) / uv_mae,

        "f_vs_uv_relative_gap_speed_rmse":
            abs(f_rmse - uv_rmse) / uv_rmse,
    })

    per_sample.append(rec)


# -------------------------------------------------------------------------
# Write per-sample table
# -------------------------------------------------------------------------

with OUT_SAMPLES.open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=list(per_sample[0].keys()),
    )
    writer.writeheader()
    writer.writerows(per_sample)


# -------------------------------------------------------------------------
# Metric-level matched-control summary
# -------------------------------------------------------------------------

metric_rows = []
summary_lines = []

summary_lines.append(
    "CANDIDATE F vs MATCHED L_uv-ONLY FINE-TUNING CONTROL"
)
summary_lines.append("=" * 88)
summary_lines.append("")
summary_lines.append(
    "Candidate UV expanded-2688 is the matched reconstruction-only "
    "fine-tuning control."
)
summary_lines.append(
    "Candidate F grad+E2-low uses the same 2688-sample scale but adds "
    "gradient + repaired E2 supervision."
)
summary_lines.append("")
summary_lines.append(
    "Important: the decomposition below is descriptive. "
    "It is not a formal causal percentage attribution."
)
summary_lines.append("")

for name, label in [
    ("db", "d_B"),
    ("w2inf", "W2inf"),
    ("w22", "W22"),
]:
    cnn_values = np.asarray(
        [r[f"cnn_{name}"] for r in per_sample],
        dtype=np.float64,
    )

    uv_values = np.asarray(
        [r[f"uv_{name}"] for r in per_sample],
        dtype=np.float64,
    )

    f_values = np.asarray(
        [r[f"f_{name}"] for r in per_sample],
        dtype=np.float64,
    )

    cnn_mean = float(np.mean(cnn_values))
    uv_mean = float(np.mean(uv_values))
    f_mean = float(np.mean(f_values))

    cnn_median = float(np.median(cnn_values))
    uv_median = float(np.median(uv_values))
    f_median = float(np.median(f_values))

    total = improvement(cnn_mean, f_mean)
    generic_ft = improvement(cnn_mean, uv_mean)
    auxiliary_increment = improvement(uv_mean, f_mean)

    if abs(total) > 1e-15:
        ft_share = 100.0 * generic_ft / total
        auxiliary_share = 100.0 * auxiliary_increment / total
    else:
        ft_share = math.nan
        auxiliary_share = math.nan

    uv_win = int(np.count_nonzero(uv_values < cnn_values))
    f_uv_win = int(np.count_nonzero(f_values < uv_values))
    f_cnn_win = int(np.count_nonzero(f_values < cnn_values))

    metric_rows.append({
        "metric": name,
        "cnn_mean": cnn_mean,
        "uv_mean": uv_mean,
        "f_mean": f_mean,

        "cnn_median": cnn_median,
        "uv_median": uv_median,
        "f_median": f_median,

        "uv_percent_change_vs_cnn":
            pct_change(uv_mean, cnn_mean),

        "f_percent_change_vs_uv":
            pct_change(f_mean, uv_mean),

        "f_percent_change_vs_cnn":
            pct_change(f_mean, cnn_mean),

        "observed_cnn_to_f_improvement":
            total,

        "generic_finetuning_component":
            generic_ft,

        "auxiliary_increment_beyond_uv":
            auxiliary_increment,

        "generic_finetuning_share_of_observed_change":
            ft_share,

        "auxiliary_share_of_observed_change":
            auxiliary_share,

        "uv_better_than_cnn_count":
            uv_win,

        "f_better_than_uv_count":
            f_uv_win,

        "f_better_than_cnn_count":
            f_cnn_win,
    })

    summary_lines.append(label)
    summary_lines.append("-" * 88)

    summary_lines.append(
        f"  means:   CNN={cnn_mean:.6f}  "
        f"UV={uv_mean:.6f}  F={f_mean:.6f}"
    )

    summary_lines.append(
        f"  medians: CNN={cnn_median:.6f}  "
        f"UV={uv_median:.6f}  F={f_median:.6f}"
    )

    summary_lines.append(
        f"  UV vs CNN mean change: "
        f"{pct_change(uv_mean, cnn_mean):+.2f}% "
        "(positive = worse because lower is better)"
    )

    summary_lines.append(
        f"  F vs UV mean change:   "
        f"{pct_change(f_mean, uv_mean):+.2f}% "
        "(negative = better)"
    )

    summary_lines.append(
        f"  F vs CNN mean change:  "
        f"{pct_change(f_mean, cnn_mean):+.2f}%"
    )

    summary_lines.append(
        "  additive mean-distance decomposition:"
    )

    summary_lines.append(
        f"    total CNN->F improvement:       {total:+.6f}"
    )

    summary_lines.append(
        f"    ordinary fine-tuning component: {generic_ft:+.6f}"
    )

    summary_lines.append(
        f"    extra F improvement beyond UV:  {auxiliary_increment:+.6f}"
    )

    summary_lines.append(
        f"    descriptive shares: "
        f"fine-tuning={ft_share:+.2f}%, "
        f"beyond-UV={auxiliary_share:+.2f}%"
    )

    summary_lines.append(
        f"  sample wins: "
        f"UV<CNN {uv_win}/168, "
        f"F<UV {f_uv_win}/168, "
        f"F<CNN {f_cnn_win}/168"
    )

    summary_lines.append("")


with OUT_METRICS.open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=list(metric_rows[0].keys()),
    )
    writer.writeheader()
    writer.writerows(metric_rows)


all3_f_uv = sum(
    r["f_better_than_uv_all3"]
    for r in per_sample
)

all3_uv_cnn = sum(
    r["uv_better_than_cnn_all3"]
    for r in per_sample
)

all3_f_cnn = sum(
    r["f_better_than_cnn_all3"]
    for r in per_sample
)

summary_lines.append(
    "THREE-METRIC CONSENSUS"
)
summary_lines.append("-" * 88)

summary_lines.append(
    f"  UV better than CNN under all three: "
    f"{all3_uv_cnn}/168"
)

summary_lines.append(
    f"  Candidate F better than UV under all three: "
    f"{all3_f_uv}/168"
)

summary_lines.append(
    f"  Candidate F better than CNN under all three: "
    f"{all3_f_cnn}/168"
)

summary_lines.append("")
summary_lines.append(
    "CONVENTIONAL FIDELITY: F vs UV"
)
summary_lines.append("-" * 88)

psnr_gap = np.asarray(
    [r["f_vs_uv_abs_delta_psnruv"] for r in per_sample]
)

mae_gap = np.asarray(
    [r["f_vs_uv_relative_gap_speed_mae"] for r in per_sample]
)

rmse_gap = np.asarray(
    [r["f_vs_uv_relative_gap_speed_rmse"] for r in per_sample]
)

summary_lines.append(
    f"  median |delta PSNRuv|: "
    f"{np.median(psnr_gap):.6f} dB"
)

summary_lines.append(
    f"  median relative speed-MAE gap: "
    f"{100*np.median(mae_gap):.3f}%"
)

summary_lines.append(
    f"  median relative speed-RMSE gap: "
    f"{100*np.median(rmse_gap):.3f}%"
)

summary_lines.append("")
summary_lines.append(
    "PREDECLARED VISUAL SAMPLES"
)
summary_lines.append("-" * 88)

sample_map = {
    r["sample"]: r
    for r in per_sample
}

for sample in SELECTED:
    r = sample_map[sample]

    summary_lines.append(
        f"  sample {sample}: "
        f"F<UV all3={r['f_better_than_uv_all3']} | "
        f"dB {r['cnn_db']:.3f}->{r['uv_db']:.3f}->{r['f_db']:.3f} | "
        f"W2inf {r['cnn_w2inf']:.3f}->{r['uv_w2inf']:.3f}->{r['f_w2inf']:.3f} | "
        f"W22 {r['cnn_w22']:.3f}->{r['uv_w22']:.3f}->{r['f_w22']:.3f} | "
        f"F-vs-UV dPSNR={r['f_vs_uv_abs_delta_psnruv']:.4f} dB, "
        f"relMAE={100*r['f_vs_uv_relative_gap_speed_mae']:.2f}%, "
        f"relRMSE={100*r['f_vs_uv_relative_gap_speed_rmse']:.2f}%"
    )

summary_lines.append("")
summary_lines.append(
    "Interpretation rule:"
)
summary_lines.append(
    "  If UV is worse than CNN while Candidate F is better than both, "
    "then the observed topology improvement cannot be described as a "
    "generic benefit of fine-tuning alone at the corresponding aggregate level."
)
summary_lines.append(
    "  Shares above 100% are possible when ordinary L_uv-only fine-tuning "
    "moves the topology metric in the wrong direction; interpret this as "
    "overcoming a negative fine-tuning component, not as literal causal percent."
)

OUT_SUMMARY.write_text(
    "\n".join(summary_lines)
    + "\n"
)

print(
    "\n".join(summary_lines)
)

print()
print("Summary:", OUT_SUMMARY)
print("Per-sample:", OUT_SAMPLES)
print("Metric summary:", OUT_METRICS)
