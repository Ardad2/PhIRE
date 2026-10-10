#!/usr/bin/env python3
"""
Freeze a secondary PD/MT-discordance visual-case selection for Candidate F vs CNN.

Selection policy
----------------
This is a SECONDARY study and does not modify the original predeclared
Candidate-F-vs-CNN visual set.

Primary discordance rule:
  1. sample belongs to the already-frozen PRIMARY 20% conventional-near-tie tier;
  2. Candidate F improves all three corrected PD metrics vs CNN:
         d_B, W2inf, W22
  3. Candidate F worsens the audited merge-tree distance vs CNN;
  4. rank first by the minimum relative PD improvement across the three corrected
     PD metrics (descending);
  5. break ties by MT worsening magnitude (descending), then by original
     conventional rank (ascending), then sample index.

The same logic is also reported for STRICT 10% and BROAD 30% tiers for context.

Historical TTK "2" PD values are NOT used.
"""

import csv
import math
import os
from pathlib import Path


W22 = Path(os.environ["W22"])
PDMT = W22 / "corrected_pd_mt"

NEAR_TIE = (
    W22
    / "near_tie_candidateF_grad_E2_vs_cnn_topology_ranked.csv"
)

ARCHETYPES = (
    PDMT
    / "corrected_pd_mt_sample_archetypes_vs_cnn.csv"
)

FOCAL = (
    PDMT
    / "corrected_pd_mt_focal_comparisons.csv"
)

OUT_MASTER = (
    PDMT
    / "candidateF_pd_mt_discordance_master.csv"
)

OUT_RANKED = (
    PDMT
    / "candidateF_pd_mt_discordance_ranked.csv"
)

OUT_SUMMARY = (
    PDMT
    / "candidateF_pd_mt_discordance_summary.txt"
)


CANDIDATE_METHOD = "f1_grad_e2"

TIERS = [
    ("strict_10pct", "STRICT 10%"),
    ("primary_20pct", "PRIMARY 20%"),
    ("broad_30pct", "BROAD 30%"),
]


def read_csv(path):
    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def fnum(x, label):
    try:
        v = float(x)
    except Exception as e:
        raise RuntimeError(
            f"Cannot parse {label}: {x!r}"
        ) from e

    if not math.isfinite(v):
        raise RuntimeError(
            f"Nonfinite {label}: {x!r}"
        )

    return v


near_rows = read_csv(NEAR_TIE)

required_near = {
    "tier",
    "sample",
    "conventional_rank",
}

missing = required_near - set(near_rows[0].keys())

if missing:
    raise RuntimeError(
        f"Near-tie table missing columns: {sorted(missing)}"
    )

near_by_key = {}

for r in near_rows:
    tier = r["tier"]
    sample = int(r["sample"])

    if tier not in {
        "strict_10pct",
        "primary_20pct",
        "broad_30pct",
    }:
        continue

    key = (tier, sample)

    if key in near_by_key:
        raise RuntimeError(
            f"Duplicate near-tie row: {key}"
        )

    near_by_key[key] = r


arch_rows = read_csv(ARCHETYPES)

arch = {}

for r in arch_rows:
    if r["method_id"] != CANDIDATE_METHOD:
        continue

    sample = int(r["sample_idx"])

    if sample in arch:
        raise RuntimeError(
            f"Duplicate Candidate-F archetype sample {sample}"
        )

    arch[sample] = r


if sorted(arch) != list(range(168)):
    raise RuntimeError(
        "Candidate-F corrected PD/MT archetype table "
        "does not cover exactly samples 0..167."
    )


focal_rows = read_csv(FOCAL)

f_vs_uv = {}

for r in focal_rows:
    if (
        r["candidate_method"] == "f1_grad_e2"
        and r["baseline_method"] == "uv"
    ):
        sample = int(r["sample_idx"])

        if sample in f_vs_uv:
            raise RuntimeError(
                f"Duplicate F-vs-UV sample {sample}"
            )

        f_vs_uv[sample] = r


if sorted(f_vs_uv) != list(range(168)):
    raise RuntimeError(
        "F-vs-UV focal table does not cover exactly samples 0..167."
    )


master = []

for tier, tier_label in TIERS:
    tier_samples = sorted(
        sample
        for (t, sample) in near_by_key
        if t == tier
    )

    for sample in tier_samples:
        n = near_by_key[(tier, sample)]
        a = arch[sample]
        u = f_vs_uv[sample]

        db_gain = fnum(
            a["dB_gain_vs_cnn"],
            f"sample {sample} dB gain",
        )

        w2inf_gain = fnum(
            a["W2inf_gain_vs_cnn"],
            f"sample {sample} W2inf gain",
        )

        w22_gain = fnum(
            a["W22_gain_vs_cnn"],
            f"sample {sample} W22 gain",
        )

        mt_gain = fnum(
            a["MT_gain_vs_cnn"],
            f"sample {sample} MT gain",
        )

        all3_pd_improve = (
            a["pd_state"] == "all3_improve"
        )

        mt_worsens = (
            a["mt_state"] == "worsen"
        )

        qualifies = (
            all3_pd_improve
            and mt_worsens
        )

        min_pd_gain = min(
            db_gain,
            w2inf_gain,
            w22_gain,
        )

        mean_pd_gain = (
            db_gain
            + w2inf_gain
            + w22_gain
        ) / 3.0

        mt_worsening = max(
            0.0,
            -mt_gain,
        )

        row = {
            "tier": tier,
            "tier_label": tier_label,
            "sample": sample,
            "conventional_rank":
                int(n["conventional_rank"]),

            "qualifies_pdplus_mtminus":
                int(qualifies),

            "dB_gain_vs_cnn":
                db_gain,

            "W2inf_gain_vs_cnn":
                w2inf_gain,

            "W22_gain_vs_cnn":
                w22_gain,

            "min_pd_gain_vs_cnn":
                min_pd_gain,

            "mean_pd_gain_vs_cnn":
                mean_pd_gain,

            "MT_gain_vs_cnn":
                mt_gain,

            "MT_worsening_magnitude":
                mt_worsening,

            "F_vs_UV_all3_PD_improve":
                int(u["all3_PD_improve"]),

            "F_vs_UV_MT_state":
                u["MT_state"],

            "F_vs_UV_dB_gain":
                fnum(
                    u["dB_gain"],
                    f"sample {sample} F-vs-UV dB gain",
                ),

            "F_vs_UV_W2inf_gain":
                fnum(
                    u["W2inf_gain"],
                    f"sample {sample} F-vs-UV W2inf gain",
                ),

            "F_vs_UV_W22_gain":
                fnum(
                    u["W22_gain"],
                    f"sample {sample} F-vs-UV W22 gain",
                ),

            "F_vs_UV_MT_gain":
                fnum(
                    u["MT_gain"],
                    f"sample {sample} F-vs-UV MT gain",
                ),
        }

        for col in [
            "abs_delta_psnruv",
            "relative_gap_speed_mae",
            "relative_gap_speed_rmse",
            "topology_consensus_score",
        ]:
            if col in n and n[col] not in ("", None):
                row[col] = fnum(
                    n[col],
                    f"sample {sample} {col}",
                )

        master.append(row)


if not master:
    raise RuntimeError(
        "No near-tie rows were loaded."
    )


all_fields = []

for r in master:
    for key in r:
        if key not in all_fields:
            all_fields.append(key)


with OUT_MASTER.open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=all_fields,
    )
    writer.writeheader()
    writer.writerows(master)


ranked = []

for tier, tier_label in TIERS:
    rows = [
        r
        for r in master
        if (
            r["tier"] == tier
            and int(r["qualifies_pdplus_mtminus"]) == 1
        )
    ]

    rows.sort(
        key=lambda r: (
            -float(r["min_pd_gain_vs_cnn"]),
            -float(r["MT_worsening_magnitude"]),
            int(r["conventional_rank"]),
            int(r["sample"]),
        )
    )

    for rank, r in enumerate(
        rows,
        start=1,
    ):
        x = dict(r)
        x["discordance_rank"] = rank
        ranked.append(x)


rank_fields = [
    "discordance_rank"
] + all_fields


with OUT_RANKED.open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=rank_fields,
    )
    writer.writeheader()
    writer.writerows(ranked)


lines = []


def emit(s=""):
    lines.append(s)
    print(s)


emit(
    "CANDIDATE-F CORRECTED-PD / MT DISCORDANCE SELECTION"
)

emit("=" * 92)
emit()

emit("SOURCE POLICY")
emit("-" * 92)

emit(
    "  historical TTK PD used: NO"
)

emit(
    "  corrected PD / audited MT source:"
)

emit(
    f"    {ARCHETYPES}"
)

emit(
    "  frozen conventional-near-tie source:"
)

emit(
    f"    {NEAR_TIE}"
)

emit(
    "  matched L_uv-only control is context only, not a selection criterion."
)

emit()

emit("FROZEN SECONDARY SELECTION RULE")
emit("-" * 92)

emit(
    "  1. retain membership in an already-frozen conventional near-tie tier;"
)

emit(
    "  2. require Candidate F to improve d_B, W2inf, and W22 vs CNN;"
)

emit(
    "  3. require audited MT distance to worsen vs CNN;"
)

emit(
    "  4. rank by minimum relative PD gain descending;"
)

emit(
    "  5. tie-break by MT worsening magnitude descending, "
    "then conventional rank ascending, then sample index."
)

emit()

for tier, tier_label in TIERS:
    all_tier = [
        r
        for r in master
        if r["tier"] == tier
    ]

    q = [
        r
        for r in ranked
        if r["tier"] == tier
    ]

    emit(tier_label)
    emit("-" * 92)

    emit(
        f"  frozen near-tie cases: {len(all_tier)}"
    )

    emit(
        f"  all3 corrected-PD improve + MT worsen: {len(q)}"
    )

    if q:
        emit()
        emit("  ranked discordant cases:")

        for r in q[:15]:
            emit(
                f"    rank={int(r['discordance_rank']):2d} "
                f"sample={int(r['sample']):3d} "
                f"conv_rank={int(r['conventional_rank']):3d} "
                f"min_PD={100*float(r['min_pd_gain_vs_cnn']):6.2f}% "
                f"mean_PD={100*float(r['mean_pd_gain_vs_cnn']):6.2f}% "
                f"dB={100*float(r['dB_gain_vs_cnn']):6.2f}% "
                f"W2inf={100*float(r['W2inf_gain_vs_cnn']):6.2f}% "
                f"W22={100*float(r['W22_gain_vs_cnn']):6.2f}% "
                f"MT={100*float(r['MT_gain_vs_cnn']):7.2f}% "
                f"F-vs-UV-MT={100*float(r['F_vs_UV_MT_gain']):+7.2f}%"
            )

    emit()


primary = [
    r
    for r in ranked
    if r["tier"] == "primary_20pct"
]

emit("PRIMARY INTERPRETATION")
emit("-" * 92)

if primary:
    top = primary[0]

    emit(
        "  Primary secondary-study case under the frozen rule:"
    )

    emit(
        f"    sample {int(top['sample'])}"
    )

    emit(
        f"    conventional rank = {int(top['conventional_rank'])}/168"
    )

    emit(
        f"    minimum corrected-PD gain = "
        f"{100*float(top['min_pd_gain_vs_cnn']):.2f}%"
    )

    emit(
        f"    audited MT gain = "
        f"{100*float(top['MT_gain_vs_cnn']):+.2f}% "
        "(negative = MT worsens)"
    )

    emit()
    emit(
        "  This case should be visualized before any appearance-based substitution."
    )

else:
    emit(
        "  No PRIMARY-20% case satisfies all3-PD-improve + MT-worsen."
    )


emit()
emit("OUTPUTS")
emit("-" * 92)

emit(
    f"  master:  {OUT_MASTER}"
)

emit(
    f"  ranked:  {OUT_RANKED}"
)

emit(
    f"  summary: {OUT_SUMMARY}"
)


OUT_SUMMARY.write_text(
    "\n".join(lines)
    + "\n"
)

print()
print(
    "PD/MT DISCORDANCE SELECTION: COMPLETE / FROZEN"
)
