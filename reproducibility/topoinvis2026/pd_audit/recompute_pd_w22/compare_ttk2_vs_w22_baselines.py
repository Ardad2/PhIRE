#!/usr/bin/env python3

from pathlib import Path
import csv
import math
import os
import re
import sys

import numpy as np


AUDIT = Path(
    os.environ.get(
        "AUDIT",
        str(
            Path.home()
            / "phire_runtime_audit_20260809_221548"
        ),
    )
).resolve()

W22 = Path(
    os.environ.get(
        "W22",
        str(AUDIT / "recompute_pd_w22"),
    )
).resolve()

ROOT = Path.home() / "PhIRE"


# ----------------------------------------------------------------------
# Import audited parser + validated W22 implementation
# ----------------------------------------------------------------------

sys.path.insert(
    0,
    str(AUDIT / "recompute_pd")
)

import canonical_pd_pilot as canonical


sys.path.insert(
    0,
    str(W22)
)

import w22_gudhi_pilot as w22mod


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

KEY_RE = re.compile(
    r"^(?P<method>.+)_s(?P<sample>\d+)_speed_p160_x0_y0$"
)


def read_ttk2(method):

    path = (
        ROOT
        / "ttk_runs_fixed"
        / method
        / "phase_c_final"
        / "pd_pairwise_distances.csv"
    )

    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    result = {}

    for row in rows:

        m = KEY_RE.match(
            row["key"]
        )

        if not m:
            raise RuntimeError(
                f"Could not parse key: {row['key']}"
            )

        sample = int(
            m.group("sample")
        )

        if sample in result:
            raise RuntimeError(
                f"Duplicate sample {sample}"
            )

        result[sample] = float(
            row["pd_distance"]
        )

    if set(result) != set(range(168)):
        raise RuntimeError(
            f"{method}: expected samples 0..167; "
            f"found {len(result)}"
        )

    return result


def pd_paths(method, sample):

    pd_dir = (
        ROOT
        / "ttk_runs_fixed"
        / method
        / "pd"
    )

    gt = (
        pd_dir
        / (
            f"{method}_GT_s{sample}_speed_"
            f"p160_x0_y0_pd_port_0.vtu"
        )
    )

    sr = (
        pd_dir
        / (
            f"{method}_SR_s{sample}_speed_"
            f"p160_x0_y0_pd_port_0.vtu"
        )
    )

    if not gt.exists():
        raise FileNotFoundError(gt)

    if not sr.exists():
        raise FileNotFoundError(sr)

    return gt, sr


def explicit_w22(method, sample):

    gt_path, sr_path = pd_paths(
        method,
        sample
    )

    GT, _ = canonical.read_pd(
        gt_path
    )

    SR, _ = canonical.read_pd(
        sr_path
    )

    w0 = w22mod.w22(
        GT[0],
        SR[0]
    )

    w1 = w22mod.w22(
        GT[1],
        SR[1]
    )

    return (
        w0,
        w1,
        math.hypot(w0, w1),
    )


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():

    out_path = (
        W22
        / "ttk2_vs_w22_baselines.csv"
    )

    methods = [
        "cnn",
        "gan",
    ]

    ttk = {
        method: read_ttk2(method)
        for method in methods
    }

    rows = []

    print("=" * 80)
    print("HISTORICAL TTK \"2\" VS STANDARD W_{2,2}")
    print("=" * 80)

    for method in methods:

        print()
        print(method.upper())

        for sample in range(168):

            w0, w1, wall = explicit_w22(
                method,
                sample
            )

            old = ttk[method][sample]

            diff = old - wall

            abs_diff = abs(diff)

            ratio = (
                old / wall
                if wall != 0
                else math.nan
            )

            rows.append({
                "method": method,
                "sample": sample,
                "ttk2": old,
                "w22_d0": w0,
                "w22_d1": w1,
                "w22_all": wall,
                "ttk2_minus_w22": diff,
                "abs_diff": abs_diff,
                "ttk2_over_w22": ratio,
            })

            if sample % 20 == 0 or sample == 167:
                print(
                    f"sample {sample:3d}: "
                    f"TTK2={old:.9f} "
                    f"W22={wall:.9f} "
                    f"diff={diff:+.9f}"
                )

    fieldnames = list(
        rows[0].keys()
    )

    with out_path.open(
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rows)

    print()
    print("=" * 80)
    print("PER-METHOD SUMMARY")
    print("=" * 80)

    for method in methods:

        sub = [
            r for r in rows
            if r["method"] == method
        ]

        a = np.asarray(
            [
                r["ttk2"]
                for r in sub
            ],
            dtype=float,
        )

        b = np.asarray(
            [
                r["w22_all"]
                for r in sub
            ],
            dtype=float,
        )

        diff = a - b

        ratio = a / b

        corr = np.corrcoef(
            a,
            b
        )[0, 1]

        print()
        print(method.upper())

        print(
            "TTK2 mean:",
            float(np.mean(a))
        )

        print(
            "W22 mean:",
            float(np.mean(b))
        )

        print(
            "mean TTK2-W22:",
            float(np.mean(diff))
        )

        print(
            "max |difference|:",
            float(np.max(np.abs(diff)))
        )

        print(
            "mean TTK2/W22:",
            float(np.mean(ratio))
        )

        print(
            "Pearson correlation:",
            float(corr)
        )

        print(
            "TTK2 > W22:",
            int(np.sum(a > b)),
            "/ 168"
        )

        print(
            "TTK2 < W22:",
            int(np.sum(a < b)),
            "/ 168"
        )

        print(
            "TTK2 == W22:",
            int(np.sum(a == b)),
            "/ 168"
        )

    # --------------------------------------------------------------
    # CNN-vs-GAN winner agreement
    # --------------------------------------------------------------

    by_method = {
        method: {
            r["sample"]: r
            for r in rows
            if r["method"] == method
        }
        for method in methods
    }

    agree = 0
    disagree = 0
    ties = 0

    ttk_cnn_wins = 0
    ttk_gan_wins = 0

    w22_cnn_wins = 0
    w22_gan_wins = 0

    disagreement_rows = []

    for sample in range(168):

        c = by_method["cnn"][sample]
        g = by_method["gan"][sample]

        ttk_delta = (
            c["ttk2"]
            - g["ttk2"]
        )

        w22_delta = (
            c["w22_all"]
            - g["w22_all"]
        )

        if ttk_delta < 0:
            ttk_winner = "cnn"
            ttk_cnn_wins += 1

        elif ttk_delta > 0:
            ttk_winner = "gan"
            ttk_gan_wins += 1

        else:
            ttk_winner = "tie"

        if w22_delta < 0:
            w22_winner = "cnn"
            w22_cnn_wins += 1

        elif w22_delta > 0:
            w22_winner = "gan"
            w22_gan_wins += 1

        else:
            w22_winner = "tie"

        if (
            ttk_winner == "tie"
            or w22_winner == "tie"
        ):
            ties += 1

        elif ttk_winner == w22_winner:
            agree += 1

        else:
            disagree += 1

            disagreement_rows.append(
                (
                    sample,
                    ttk_winner,
                    w22_winner,
                    ttk_delta,
                    w22_delta,
                )
            )

    print()
    print("=" * 80)
    print("CNN-vs-GAN WINNER AGREEMENT")
    print("=" * 80)

    print(
        "TTK2 winners: "
        f"CNN={ttk_cnn_wins}, "
        f"GAN={ttk_gan_wins}"
    )

    print(
        "W22 winners: "
        f"CNN={w22_cnn_wins}, "
        f"GAN={w22_gan_wins}"
    )

    print(
        "winner agreement:",
        agree
    )

    print(
        "winner disagreement:",
        disagree
    )

    print(
        "ties:",
        ties
    )

    if disagreement_rows:

        print()
        print(
            "Disagreement samples:"
        )

        for row in disagreement_rows:

            print(
                "sample=%d "
                "TTK2=%s "
                "W22=%s "
                "TTK2_delta=%+.9f "
                "W22_delta=%+.9f"
                % row
            )

    print()
    print(
        "CSV:",
        out_path
    )


if __name__ == "__main__":
    main()
