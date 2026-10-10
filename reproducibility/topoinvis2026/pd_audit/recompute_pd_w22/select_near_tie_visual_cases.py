#!/usr/bin/env python3

import csv
import math
import os
from pathlib import Path


# ---------------------------------------------------------------------------
# Paths / configuration
# ---------------------------------------------------------------------------

HOME = Path.home()
W22 = Path(os.environ["W22"])

CONVENTIONAL_CSV = (
    HOME
    / "PhIRE"
    / "ttk_runs_fixed"
    / "topology_finetuning"
    / "candidateF_grad_E2_low_expanded2688_eval"
    / "all_sample_metrics_candidateF_grad_E2_low_expanded2688.csv"
)

PD_ROBUSTNESS_CSV = (
    W22
    / "candidate_pd_robustness_samples.csv"
)

OUT_MASTER = (
    W22
    / "near_tie_candidateF_grad_E2_vs_cnn_master.csv"
)

OUT_TOPOLOGY = (
    W22
    / "near_tie_candidateF_grad_E2_vs_cnn_topology_ranked.csv"
)

OUT_SUMMARY = (
    W22
    / "near_tie_candidateF_grad_E2_vs_cnn_summary.txt"
)


CNN_METHOD = "cnn"

CANDIDATE_METHOD = (
    "candidateF_grad_E2_low_expanded2688"
)

ROBUSTNESS_CANDIDATE = (
    "candidateF_grad_E2_low_2688"
)

ROBUSTNESS_BASELINE = "cnn"

N_EXPECTED = 168


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_csv(path):

    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def finite_float(x, name, sample):

    v = float(x)

    if not math.isfinite(v):
        raise RuntimeError(
            f"Nonfinite {name} at sample {sample}: {x}"
        )

    return v


def average_ranks(values):
    """
    Average ranks for ascending values.

    Smallest value -> rank near 0.
    Largest value  -> rank near N-1.

    Exact ties receive their average rank.
    """

    n = len(values)

    order = sorted(
        range(n),
        key=lambda i: (values[i], i)
    )

    ranks = [None] * n

    pos = 0

    while pos < n:

        end = pos + 1

        while (
            end < n
            and values[order[end]]
            == values[order[pos]]
        ):
            end += 1

        avg_rank = (
            (pos + end - 1)
            / 2.0
        )

        for k in range(pos, end):
            ranks[order[k]] = avg_rank

        pos = end

    return ranks


def percentile_ranks(values):
    """
    Convert ascending average ranks to [0,1].

    0 = closest/smallest gap.
    1 = farthest/largest gap.
    """

    ranks = average_ranks(values)

    n = len(values)

    if n <= 1:
        return [0.0] * n

    return [
        r / (n - 1)
        for r in ranks
    ]


def pct_improvement(
    baseline,
    candidate,
):
    """
    Lower-is-better metric.

    Positive means candidate is better.

    (baseline - candidate) / baseline
    """

    if baseline <= 0:
        raise RuntimeError(
            "Expected positive PD distance"
        )

    return (
        baseline - candidate
    ) / baseline


# ---------------------------------------------------------------------------
# Conventional-metric data
# ---------------------------------------------------------------------------

rows = read_csv(
    CONVENTIONAL_CSV
)

by_method = {}

for r in rows:

    method = r["method"]
    sample = int(r["sample_idx"])

    by_method.setdefault(
        method,
        {}
    )

    if sample in by_method[method]:
        raise RuntimeError(
            f"Duplicate sample {sample} "
            f"for method {method}"
        )

    by_method[method][sample] = r


for method in [
    CNN_METHOD,
    CANDIDATE_METHOD,
]:

    if method not in by_method:
        raise RuntimeError(
            f"Missing method: {method}"
        )

    samples = set(
        by_method[method]
    )

    expected = set(
        range(N_EXPECTED)
    )

    if samples != expected:
        raise RuntimeError(
            f"{method}: expected samples "
            f"0..167; got {len(samples)}"
        )


# ---------------------------------------------------------------------------
# Validated PD robustness data
# ---------------------------------------------------------------------------

pd_rows = read_csv(
    PD_ROBUSTNESS_CSV
)

pd_by_sample = {}

for r in pd_rows:

    if (
        r["candidate"]
        != ROBUSTNESS_CANDIDATE
    ):
        continue

    if (
        r["baseline"]
        != ROBUSTNESS_BASELINE
    ):
        continue

    sample = int(
        r["sample"]
    )

    if sample in pd_by_sample:
        raise RuntimeError(
            f"Duplicate PD robustness "
            f"sample {sample}"
        )

    pd_by_sample[sample] = r


if set(pd_by_sample) != set(
    range(N_EXPECTED)
):
    raise RuntimeError(
        "Expected exactly 168 Candidate-F "
        "vs CNN PD robustness rows; found "
        f"{len(pd_by_sample)}"
    )


# ---------------------------------------------------------------------------
# Build conventional gaps
# ---------------------------------------------------------------------------

records = []

for sample in range(
    N_EXPECTED
):

    cnn = by_method[
        CNN_METHOD
    ][sample]

    cand = by_method[
        CANDIDATE_METHOD
    ][sample]

    cnn_psnr = finite_float(
        cnn["psnruv"],
        "cnn psnruv",
        sample,
    )

    cand_psnr = finite_float(
        cand["psnruv"],
        "candidate psnruv",
        sample,
    )

    cnn_mae = finite_float(
        cnn["speed_mae"],
        "cnn speed_mae",
        sample,
    )

    cand_mae = finite_float(
        cand["speed_mae"],
        "candidate speed_mae",
        sample,
    )

    cnn_rmse = finite_float(
        cnn["speed_rmse"],
        "cnn speed_rmse",
        sample,
    )

    cand_rmse = finite_float(
        cand["speed_rmse"],
        "candidate speed_rmse",
        sample,
    )

    # -------------------------------------------------------
    # Conventional closeness:
    #
    # PSNR: absolute dB difference
    #
    # MAE/RMSE: relative difference w.r.t. CNN value,
    # because these errors vary naturally by sample.
    # -------------------------------------------------------

    psnr_gap = abs(
        cand_psnr - cnn_psnr
    )

    mae_rel_gap = abs(
        cand_mae - cnn_mae
    ) / cnn_mae

    rmse_rel_gap = abs(
        cand_rmse - cnn_rmse
    ) / cnn_rmse

    pd = pd_by_sample[
        sample
    ]

    db_cnn = finite_float(
        pd["db_baseline"],
        "dB CNN",
        sample,
    )

    db_cand = finite_float(
        pd["db_candidate"],
        "dB candidate",
        sample,
    )

    w2inf_cnn = finite_float(
        pd["w2inf_baseline"],
        "W2inf CNN",
        sample,
    )

    w2inf_cand = finite_float(
        pd["w2inf_candidate"],
        "W2inf candidate",
        sample,
    )

    w22_cnn = finite_float(
        pd["w22_baseline"],
        "W22 CNN",
        sample,
    )

    w22_cand = finite_float(
        pd["w22_candidate"],
        "W22 candidate",
        sample,
    )

    db_impr = pct_improvement(
        db_cnn,
        db_cand,
    )

    w2inf_impr = pct_improvement(
        w2inf_cnn,
        w2inf_cand,
    )

    w22_impr = pct_improvement(
        w22_cnn,
        w22_cand,
    )

    all3 = (
        db_impr > 0
        and w2inf_impr > 0
        and w22_impr > 0
    )

    # Conservative topology score:
    #
    # A sample scores highly only when ALL THREE
    # PD metrics show a substantial relative gain.
    topology_consensus_score = min(
        db_impr,
        w2inf_impr,
        w22_impr,
    )

    topology_mean_improvement = (
        db_impr
        + w2inf_impr
        + w22_impr
    ) / 3.0

    records.append({
        "sample": sample,

        "cnn_psnruv": cnn_psnr,
        "candidate_psnruv": cand_psnr,
        "abs_delta_psnruv": psnr_gap,

        "cnn_speed_mae": cnn_mae,
        "candidate_speed_mae": cand_mae,
        "relative_gap_speed_mae":
            mae_rel_gap,

        "cnn_speed_rmse": cnn_rmse,
        "candidate_speed_rmse":
            cand_rmse,
        "relative_gap_speed_rmse":
            rmse_rel_gap,

        "db_cnn": db_cnn,
        "db_candidate": db_cand,
        "db_relative_improvement":
            db_impr,

        "w2inf_cnn": w2inf_cnn,
        "w2inf_candidate":
            w2inf_cand,
        "w2inf_relative_improvement":
            w2inf_impr,

        "w22_cnn": w22_cnn,
        "w22_candidate": w22_cand,
        "w22_relative_improvement":
            w22_impr,

        "all3_pd_improved":
            int(all3),

        "topology_consensus_score":
            topology_consensus_score,

        "topology_mean_improvement":
            topology_mean_improvement,
    })


# ---------------------------------------------------------------------------
# Rank ONLY by conventional metrics first
# ---------------------------------------------------------------------------

psnr_vals = [
    r["abs_delta_psnruv"]
    for r in records
]

mae_vals = [
    r["relative_gap_speed_mae"]
    for r in records
]

rmse_vals = [
    r["relative_gap_speed_rmse"]
    for r in records
]


psnr_pct = percentile_ranks(
    psnr_vals
)

mae_pct = percentile_ranks(
    mae_vals
)

rmse_pct = percentile_ranks(
    rmse_vals
)


for i, r in enumerate(
    records
):

    r["psnr_gap_percentile"] = (
        psnr_pct[i]
    )

    r["mae_gap_percentile"] = (
        mae_pct[i]
    )

    r["rmse_gap_percentile"] = (
        rmse_pct[i]
    )

    # Worst conventional percentile.
    #
    # A low score means the sample is close
    # under ALL THREE conventional metrics.
    r["conventional_closeness_score"] = max(
        psnr_pct[i],
        mae_pct[i],
        rmse_pct[i],
    )

    r["conventional_mean_percentile"] = (
        psnr_pct[i]
        + mae_pct[i]
        + rmse_pct[i]
    ) / 3.0


# ---------------------------------------------------------------------------
# Freeze deterministic conventional ordering
#
# No PD quantity appears in this sorting key.
# ---------------------------------------------------------------------------

conventional_order = sorted(
    records,
    key=lambda r: (
        r[
            "conventional_closeness_score"
        ],
        r[
            "conventional_mean_percentile"
        ],
        r["sample"],
    )
)


for rank, r in enumerate(
    conventional_order,
    start=1,
):

    r[
        "conventional_rank"
    ] = rank

    # Predeclared tiers:
    #
    # strict  = closest 10% = first 17
    # primary = closest 20% = first 34
    # broad   = closest 30% = first 51
    #
    # These are based ONLY on conventional
    # metric closeness.
    r["strict_10pct"] = int(
        rank <= 17
    )

    r["primary_20pct"] = int(
        rank <= 34
    )

    r["broad_30pct"] = int(
        rank <= 51
    )


# ---------------------------------------------------------------------------
# Write frozen master table in conventional rank order
# ---------------------------------------------------------------------------

fields = list(
    conventional_order[0].keys()
)

with OUT_MASTER.open(
    "w",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fields,
    )

    writer.writeheader()
    writer.writerows(
        conventional_order
    )


# ---------------------------------------------------------------------------
# Only AFTER conventional selection is frozen:
# topology ranking inside each tier.
# ---------------------------------------------------------------------------

topology_rows = []

for tier_name, tier_col in [
    ("strict_10pct", "strict_10pct"),
    ("primary_20pct", "primary_20pct"),
    ("broad_30pct", "broad_30pct"),
]:

    selected = [
        r.copy()
        for r in conventional_order
        if r[tier_col] == 1
    ]

    consensus = [
        r
        for r in selected
        if r["all3_pd_improved"] == 1
    ]

    # Rank by conservative topology score:
    #
    # largest minimum relative improvement
    # across dB, W2inf, W22.
    consensus.sort(
        key=lambda r: (
            -r[
                "topology_consensus_score"
            ],
            -r[
                "topology_mean_improvement"
            ],
            r[
                "conventional_rank"
            ],
            r["sample"],
        )
    )

    for topo_rank, r in enumerate(
        consensus,
        start=1,
    ):

        out = r.copy()

        out["tier"] = tier_name

        out[
            "topology_rank_within_tier"
        ] = topo_rank

        topology_rows.append(
            out
        )


topology_fields = (
    fields
    + [
        "tier",
        "topology_rank_within_tier",
    ]
)

with OUT_TOPOLOGY.open(
    "w",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=topology_fields,
    )

    writer.writeheader()
    writer.writerows(
        topology_rows
    )


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def tier_records(col):

    return [
        r
        for r in conventional_order
        if r[col] == 1
    ]


summary_lines = []

summary_lines.append(
    "CANDIDATE-F CONVENTIONAL NEAR-TIE / "
    "PD-SEPARATION STUDY"
)

summary_lines.append(
    "=" * 80
)

summary_lines.append("")

summary_lines.append(
    "Comparison:"
)

summary_lines.append(
    "  CNN vs "
    "candidateF_grad_E2_low_expanded2688"
)

summary_lines.append("")

summary_lines.append(
    "Conventional metrics used:"
)

summary_lines.append(
    "  |delta PSNRuv|"
)

summary_lines.append(
    "  |delta speed MAE| / CNN speed MAE"
)

summary_lines.append(
    "  |delta speed RMSE| / CNN speed RMSE"
)

summary_lines.append("")

summary_lines.append(
    "SSIM:"
)

summary_lines.append(
    "  excluded because all values are NaN"
)

summary_lines.append("")

summary_lines.append(
    "Conventional selection was performed "
    "before topology ranking."
)

summary_lines.append("")

for name, col in [
    ("STRICT 10%", "strict_10pct"),
    ("PRIMARY 20%", "primary_20pct"),
    ("BROAD 30%", "broad_30pct"),
]:

    subset = tier_records(
        col
    )

    consensus = [
        r
        for r in subset
        if r[
            "all3_pd_improved"
        ] == 1
    ]

    summary_lines.append(
        f"{name}"
    )

    summary_lines.append(
        f"  conventional cases: "
        f"{len(subset)}"
    )

    summary_lines.append(
        f"  all-three-PD-improved: "
        f"{len(consensus)}"
    )

    summary_lines.append("")

    ranked = sorted(
        consensus,
        key=lambda r: (
            -r[
                "topology_consensus_score"
            ],
            -r[
                "topology_mean_improvement"
            ],
            r[
                "conventional_rank"
            ],
        )
    )

    summary_lines.append(
        "  top topology-separated cases:"
    )

    for r in ranked[:10]:

        summary_lines.append(
            "    "
            f"sample={r['sample']:3d} "
            f"conv_rank={r['conventional_rank']:3d} "
            f"C={r['conventional_closeness_score']:.4f} "
            f"min_PD_gain={100*r['topology_consensus_score']:.2f}% "
            f"mean_PD_gain={100*r['topology_mean_improvement']:.2f}% "
            f"dB={100*r['db_relative_improvement']:.2f}% "
            f"W2inf={100*r['w2inf_relative_improvement']:.2f}% "
            f"W22={100*r['w22_relative_improvement']:.2f}%"
        )

    summary_lines.append("")


summary_lines.append(
    "Primary interpretation rule:"
)

summary_lines.append(
    "  Visual inspection must begin only "
    "after this table is frozen."
)

summary_lines.append(
    "  Prefer samples in the PRIMARY 20% "
    "conventional near-tie subset"
)

summary_lines.append(
    "  where all three validated PD metrics "
    "favor Candidate F."
)

summary_lines.append(
    "  Rank topology separation by the minimum "
    "relative improvement across"
)

summary_lines.append(
    "  dB, W2inf, and W22."
)

summary_lines.append("")

summary_lines.append(
    f"Master CSV: {OUT_MASTER}"
)

summary_lines.append(
    f"Topology-ranked CSV: {OUT_TOPOLOGY}"
)


OUT_SUMMARY.write_text(
    "\n".join(
        summary_lines
    )
    + "\n"
)


print(
    "\n".join(
        summary_lines
    )
)

print()
print(
    "NEAR-TIE SELECTION: COMPLETE"
)
