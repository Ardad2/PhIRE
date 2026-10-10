#!/usr/bin/env python3

import csv
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np


W22 = Path(os.environ["W22"])

INPUT = W22 / "w22_full_sweep.csv"

OUT_SUMMARY = (
    W22
    / "candidate_pd_robustness_summary.csv"
)

OUT_SAMPLES = (
    W22
    / "candidate_pd_robustness_samples.csv"
)


BASELINES = {
    "cnn": "cnn",
    "gan": "gan",
}


CANDIDATES = {
    "candidateC_2688":
        "topology_finetuning/"
        "candidateC_expanded2688_topology",

    "candidateB_E2_low_2688":
        "topology_finetuning/"
        "candidateB_plus_E2_tf_lowlambda_"
        "expanded2688_topology",

    "candidateF_grad_E2_low_2688":
        "topology_finetuning/"
        "candidateF_grad_E2_low_"
        "expanded2688_topology",

    "candidateF_grad_levelset_E2_low_2688":
        "topology_finetuning/"
        "candidateF_grad_levelset_E2_low_"
        "expanded2688_topology",
}


METRICS = {
    "db": "bottleneck_all",
    "w2inf": "w2inf_all",
    "w22": "w22_all",
}


EPS = 1e-12


def relation(candidate, baseline):

    d = candidate - baseline

    if d < -EPS:
        return "improved"

    if d > EPS:
        return "worsened"

    return "tie"


def load():

    with INPUT.open(newline="") as f:
        rows = list(csv.DictReader(f))

    by_run = defaultdict(dict)

    for r in rows:

        run = r["run"]
        sample = int(r["sample"])

        if sample in by_run[run]:
            raise RuntimeError(
                f"duplicate sample {sample} "
                f"in run {run}"
            )

        by_run[run][sample] = r

    required = (
        set(BASELINES.values())
        | set(CANDIDATES.values())
    )

    missing = sorted(
        required - set(by_run)
    )

    if missing:
        raise RuntimeError(
            "Missing runs:\n"
            + "\n".join(missing)
        )

    for run in required:

        samples = set(by_run[run])

        if samples != set(range(168)):
            raise RuntimeError(
                f"{run}: expected 0..167; "
                f"got {len(samples)} samples"
            )

    return by_run


def compare_candidate(
    by_run,
    candidate_label,
    candidate_run,
    baseline_label,
    baseline_run,
):

    detailed = []

    counts = defaultdict(int)

    metric_deltas = {
        m: []
        for m in METRICS
    }

    for sample in range(168):

        c = by_run[candidate_run][sample]
        b = by_run[baseline_run][sample]

        status = {}

        values = {}

        for short, col in METRICS.items():

            cv = float(c[col])
            bv = float(b[col])

            delta = cv - bv

            rel = relation(
                cv,
                bv,
            )

            status[short] = rel

            values[f"{short}_baseline"] = bv
            values[f"{short}_candidate"] = cv
            values[f"{short}_delta"] = delta

            counts[
                f"{short}_{rel}"
            ] += 1

            metric_deltas[short].append(
                delta
            )

        improved = {
            m
            for m, s in status.items()
            if s == "improved"
        }

        worsened = {
            m
            for m, s in status.items()
            if s == "worsened"
        }

        all3 = (
            improved
            == {"db", "w2inf", "w22"}
        )

        wasserstein_both = (
            status["w2inf"] == "improved"
            and status["w22"] == "improved"
        )

        w_ground_agree = (
            status["w2inf"]
            == status["w22"]
        )

        if all3:
            category = "ALL3_IMPROVED"

        elif (
            wasserstein_both
            and status["db"] == "worsened"
        ):
            category = (
                "WASSERSTEINS_IMPROVED_DB_WORSE"
            )

        elif (
            status["db"] == "improved"
            and status["w2inf"] == "worsened"
            and status["w22"] == "worsened"
        ):
            category = (
                "DB_IMPROVED_WASSERSTEINS_WORSE"
            )

        elif not w_ground_agree:
            category = (
                "W2INF_W22_DISAGREE"
            )

        elif len(improved) == 2:
            category = "TWO_OF_THREE_IMPROVED"

        elif len(improved) == 1:
            category = "ONE_OF_THREE_IMPROVED"

        elif len(worsened) == 3:
            category = "ALL3_WORSENED"

        else:
            category = "MIXED_OR_TIE"

        counts[
            f"category_{category}"
        ] += 1

        detailed.append({
            "candidate": candidate_label,
            "candidate_run": candidate_run,
            "baseline": baseline_label,
            "sample": sample,

            **{
                k: f"{v:.17g}"
                for k, v in values.items()
            },

            "db_status":
                status["db"],

            "w2inf_status":
                status["w2inf"],

            "w22_status":
                status["w22"],

            "all3_improved":
                int(all3),

            "both_wassersteins_improved":
                int(wasserstein_both),

            "w2inf_w22_same_direction":
                int(w_ground_agree),

            "category":
                category,
        })

    summary = {
        "candidate":
            candidate_label,

        "candidate_run":
            candidate_run,

        "baseline":
            baseline_label,

        "n":
            168,

        "db_improved":
            counts["db_improved"],

        "db_worsened":
            counts["db_worsened"],

        "db_tie":
            counts["db_tie"],

        "w2inf_improved":
            counts["w2inf_improved"],

        "w2inf_worsened":
            counts["w2inf_worsened"],

        "w2inf_tie":
            counts["w2inf_tie"],

        "w22_improved":
            counts["w22_improved"],

        "w22_worsened":
            counts["w22_worsened"],

        "w22_tie":
            counts["w22_tie"],

        "all3_improved":
            counts[
                "category_ALL3_IMPROVED"
            ],

        "all3_worsened":
            counts[
                "category_ALL3_WORSENED"
            ],

        "wassersteins_improved_db_worse":
            counts[
                "category_"
                "WASSERSTEINS_IMPROVED_DB_WORSE"
            ],

        "db_improved_wassersteins_worse":
            counts[
                "category_"
                "DB_IMPROVED_WASSERSTEINS_WORSE"
            ],

        "w2inf_w22_disagree":
            counts[
                "category_W2INF_W22_DISAGREE"
            ],

        "mean_delta_db":
            float(
                np.mean(
                    metric_deltas["db"]
                )
            ),

        "median_delta_db":
            float(
                np.median(
                    metric_deltas["db"]
                )
            ),

        "mean_delta_w2inf":
            float(
                np.mean(
                    metric_deltas["w2inf"]
                )
            ),

        "median_delta_w2inf":
            float(
                np.median(
                    metric_deltas["w2inf"]
                )
            ),

        "mean_delta_w22":
            float(
                np.mean(
                    metric_deltas["w22"]
                )
            ),

        "median_delta_w22":
            float(
                np.median(
                    metric_deltas["w22"]
                )
            ),
    }

    return summary, detailed


def main():

    by_run = load()

    summaries = []
    details = []

    for candidate_label, candidate_run \
            in CANDIDATES.items():

        for baseline_label, baseline_run \
                in BASELINES.items():

            summary, detail = (
                compare_candidate(
                    by_run,
                    candidate_label,
                    candidate_run,
                    baseline_label,
                    baseline_run,
                )
            )

            summaries.append(summary)
            details.extend(detail)

    with OUT_SUMMARY.open(
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                summaries[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            summaries
        )

    with OUT_SAMPLES.open(
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                details[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            details
        )

    print("=" * 100)
    print(
        "SAMPLE-WISE PD ROBUSTNESS"
    )
    print("=" * 100)

    for s in summaries:

        print()
        print(
            s["candidate"],
            "vs",
            s["baseline"],
        )

        print(
            "  dB improved:      "
            f"{s['db_improved']:3d}/168"
        )

        print(
            "  W2inf improved:   "
            f"{s['w2inf_improved']:3d}/168"
        )

        print(
            "  W22 improved:     "
            f"{s['w22_improved']:3d}/168"
        )

        print(
            "  ALL THREE:        "
            f"{s['all3_improved']:3d}/168"
        )

        print(
            "  all 3 worsened:   "
            f"{s['all3_worsened']:3d}/168"
        )

        print(
            "  W2s improve, "
            "dB worse:      "
            f"{s['wassersteins_improved_db_worse']:3d}"
        )

        print(
            "  dB improves, "
            "W2s worse:     "
            f"{s['db_improved_wassersteins_worse']:3d}"
        )

        print(
            "  W2inf/W22 "
            "direction disagreements: "
            f"{s['w2inf_w22_disagree']:3d}"
        )

        print(
            "  mean deltas "
            "(candidate-baseline):"
        )

        print(
            "    dB:    "
            f"{s['mean_delta_db']:+.6f}"
        )

        print(
            "    W2inf: "
            f"{s['mean_delta_w2inf']:+.6f}"
        )

        print(
            "    W22:   "
            f"{s['mean_delta_w22']:+.6f}"
        )

        print(
            "  median deltas:"
        )

        print(
            "    dB:    "
            f"{s['median_delta_db']:+.6f}"
        )

        print(
            "    W2inf: "
            f"{s['median_delta_w2inf']:+.6f}"
        )

        print(
            "    W22:   "
            f"{s['median_delta_w22']:+.6f}"
        )

    print()
    print(
        "Summary:",
        OUT_SUMMARY
    )

    print(
        "Per-sample:",
        OUT_SAMPLES
    )


if __name__ == "__main__":
    main()
